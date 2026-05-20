#!/usr/bin/env python3
"""Rewrite a Cobre case's ``initial_conditions.json::past_inflows`` to match the
historical-replay lag state used by the forward pass.

Background
----------
When a Cobre case is configured with ``inflow.scheme: "historical"``, the
forward pass calls ``ClassSampler::Historical::apply_initial_state`` before
solving stage 0. That call **overwrites** the inflow-lag portion of the state
vector with the historical window's lag values (``HistoricalScenarioLibrary
::lag_slice``).

The ``evaluate_lower_bound`` path in ``cobre-sddp`` does **not** apply this
rewrite — it consumes the raw ``initial_state`` built from
``past_inflows.values_m3s``. The result is that LB and UB measure
``V_0(x_0)`` at two different ``x_0`` lag states, which produces a
non-vanishing (typically negative) gap even on a deterministic single-historical-year
run.

This script forces both paths to operate on the same ``x_0`` by copying the
forward pass's effective lag state (which is dumped to
``output/simulation/inflow_lags/scenario_id=0000/data.parquet`` at stage 0)
back into ``initial_conditions.json::past_inflows``. After the rewrite,
``build_initial_state`` puts those same lags into the state vector before
``apply_initial_state`` overwrites them idempotently, so the LB calc sees the
identical state.

This is a diagnostic/workaround tool. The proper fix lives in
``crates/cobre-sddp`` (apply the historical-state rewrite inside
``evaluate_lower_bound`` or bake it into ``training_ctx.initial_state`` at
setup time).

Usage
-----
    python scripts/inject_historical_lags.py <case_dir>

Reads
    <case_dir>/output/simulation/inflow_lags/scenario_id=0000/data.parquet
Writes
    <case_dir>/initial_conditions.json   (with a .bak.<unix> sibling)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl


def rewrite_past_inflows(case_dir: Path, scenario_id: int = 0) -> None:
    ic_path = case_dir / "initial_conditions.json"
    parquet_path = (
        case_dir
        / "output"
        / "simulation"
        / "inflow_lags"
        / f"scenario_id={scenario_id:04d}"
        / "data.parquet"
    )

    if not ic_path.is_file():
        raise FileNotFoundError(ic_path)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    # Load the forward pass's effective lag state at stage 0.
    df = pl.read_parquet(parquet_path).filter(pl.col("stage_id") == 0)
    if df.is_empty():
        raise RuntimeError(f"{parquet_path} has no rows with stage_id=0")

    # Build {hydro_id: [inflow_m3s at lag_index 0, 1, 2, ...]}
    by_hydro = (
        df.sort(["hydro_id", "lag_index"])
        .group_by("hydro_id")
        .agg(pl.col("inflow_m3s"))
        .sort("hydro_id")
    )
    lags_by_hydro: dict[int, list[float]] = dict(
        zip(
            by_hydro["hydro_id"].to_list(),
            by_hydro["inflow_m3s"].to_list(),
            strict=True,
        )
    )

    # Load and rewrite initial_conditions.json.
    with ic_path.open() as f:
        ic = json.load(f)

    past_inflows = ic.get("past_inflows")
    if not isinstance(past_inflows, list):
        raise RuntimeError("initial_conditions.json is missing 'past_inflows' list")

    n_rewritten = 0
    n_skipped = 0
    n_truncated = 0
    for entry in past_inflows:
        hid = entry["hydro_id"]
        new_lags = lags_by_hydro.get(hid)
        if new_lags is None:
            n_skipped += 1
            continue
        original_len = len(entry["values_m3s"])
        # Preserve the original lag count (typically 12).  Truncate or zero-pad
        # the parquet values to match so the JSON schema stays untouched.
        if len(new_lags) >= original_len:
            entry["values_m3s"] = [float(v) for v in new_lags[:original_len]]
            if len(new_lags) > original_len:
                n_truncated += 1
        else:
            entry["values_m3s"] = [float(v) for v in new_lags] + [0.0] * (
                original_len - len(new_lags)
            )
        n_rewritten += 1

    # Back up the original (keep history with a unix-timestamp suffix).
    backup_path = ic_path.with_suffix(f".json.bak.{int(time.time())}")
    backup_path.write_bytes(ic_path.read_bytes())

    with ic_path.open("w") as f:
        json.dump(ic, f, indent=2)
        f.write("\n")

    print(f"Rewrote past_inflows for {n_rewritten} hydros in {ic_path}")
    print(f"  Truncated (parquet had more lags than JSON): {n_truncated}")
    print(f"  Skipped (no parquet lags for hydro_id):      {n_skipped}")
    print(f"  Backup written to:                           {backup_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case_dir", type=Path, help="Path to a Cobre case directory.")
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=0,
        help="Simulation scenario_id to read lag state from (default: 0).",
    )
    args = parser.parse_args()

    try:
        rewrite_past_inflows(args.case_dir, scenario_id=args.scenario_id)
    except FileNotFoundError as e:
        print(f"error: missing file {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
