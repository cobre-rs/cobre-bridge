"""Generic constraint LHS evaluation from NEWAVE and Cobre simulation outputs.

Reads the converted Cobre case's ``constraints/generic_constraints.json``
and ``constraints/generic_constraint_bounds.parquet`` and evaluates each
constraint's LHS using both:

- NEWAVE outputs — ``MEDIAS-USIH.CSV`` (``GHIDUH``, ``VARMUH`` per plant
  per stage) and ``int*.out`` (per-line per-stage interchange).
- Cobre simulation outputs — per-(scenario, stage, block) hydro and
  exchange data, collapsed to one value per (constraint, stage) by
  averaging across scenarios and blocks.

The output is consumed by the compare-results report's Constraints tab,
mirroring the dashboard's per-constraint LHS-vs-Bound visualisation with a
NEWAVE overlay so the user can see whether a binding constraint matches
across the two solvers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl

from cobre_bridge.comparators.alignment import EntityAlignment
from cobre_bridge.dashboard.tabs.constraints_utils import (
    _parse_expression,
    _resolve_param_to_column,
)
from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)


def _load_generic_constraints(cobre_input_dir: Path) -> list[dict]:
    """Load constraint definitions from ``constraints/generic_constraints.json``.

    Returns an empty list when the file is missing or malformed.
    """
    path = cobre_input_dir / "constraints" / "generic_constraints.json"
    if not path.exists():
        return []
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("generic_constraints.json could not be parsed: %s", exc)
        return []
    return list(data.get("constraints", []))


def _load_generic_constraint_bounds(cobre_input_dir: Path) -> pl.DataFrame:
    """Load bound table from ``constraints/generic_constraint_bounds.parquet``."""
    path = cobre_input_dir / "constraints" / "generic_constraint_bounds.parquet"
    if not path.exists():
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "block_id": pl.Int32,
                "bound": pl.Float64,
            }
        )
    return pl.read_parquet(path)


def evaluate_lhs_newave(
    constraints: list[dict],
    nw_hydro_df: pl.DataFrame,
    nw_line_means: pl.DataFrame,
    alignment: EntityAlignment,
    id_map: NewaveIdMap,
    nw_offset: int,
) -> pl.DataFrame:
    """Evaluate each constraint's LHS from NEWAVE simulation outputs.

    The constraint expressions reference Cobre entity IDs. We translate
    via ``id_map`` (hydro Cobre id → NEWAVE code) and
    ``alignment.lines`` (Cobre line id → NEWAVE submarket pair) before
    looking values up in the NEWAVE data.

    NEWAVE outputs are stage-level (block-collapsed) — MEDIAS values are
    monthly means and ``int*.out`` ``TOTAL`` rows already average over
    blocks weighted by their duration. The result therefore has one value
    per (constraint, stage_0based) pair.

    Parameters
    ----------
    constraints:
        Constraint dicts from ``generic_constraints.json``.
    nw_hydro_df:
        Output of :func:`cobre_bridge.comparators.newave_readers.read_medias_hydro`.
        Columns ``newave_code`` (Int64), ``stage`` (Int64, 1-based as in
        MEDIAS — Sep-start studies use 9 for the first stage),
        ``variable`` (Utf8), ``value`` (Float64).
    nw_line_means:
        Output of
        :func:`cobre_bridge.comparators.newave_readers.read_nwlistop_intercambio`.
        One row per directional pair × stage.  We map each Cobre line to
        the matching directional row via the alignment's ``newave_de`` /
        ``newave_para`` fields (respecting the ``reversed`` flag for
        sign).
    alignment:
        Pre-built entity alignment.  Provides Cobre-line → (newave_de,
        newave_para) mappings and the ``reversed`` flag.
    id_map:
        Used to translate Cobre hydro IDs back to NEWAVE plant codes.
    nw_offset:
        MEDIAS stage offset (e.g. 9 for a September-start study).  Used
        to convert NEWAVE 1-based MEDIAS stages to Cobre 0-based.

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32, 0-based),
        ``lhs_value`` (Float64).  One row per (constraint, stage) pair
        that has at least one resolvable term.  Stages with missing
        variables produce no row (rather than a zero); this avoids
        misrepresenting "no NEWAVE data" as "LHS = 0".
    """
    if not constraints:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # --- Build hydro generation lookup: (cobre_hydro_id, stage_0based) -> MW
    hydro_gen: dict[tuple[int, int], float] = {}
    if not nw_hydro_df.is_empty():
        ghiduh = nw_hydro_df.filter(pl.col("variable") == "GHIDUH")
        for row in ghiduh.iter_rows(named=True):
            nw_code = int(row["newave_code"])
            try:
                cobre_id = id_map.hydro_id(nw_code)
            except KeyError:
                # Skip plants that are not in the LP (e.g. FICT, NE/NC).
                continue
            stage_0based = int(row["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            hydro_gen[(cobre_id, stage_0based)] = float(row["value"])

    # --- Build hydro storage lookup: (cobre_hydro_id, stage_0based) -> hm3
    hydro_storage: dict[tuple[int, int], float] = {}
    if not nw_hydro_df.is_empty():
        varmuh = nw_hydro_df.filter(pl.col("variable") == "VARMUH")
        for row in varmuh.iter_rows(named=True):
            nw_code = int(row["newave_code"])
            try:
                cobre_id = id_map.hydro_id(nw_code)
            except KeyError:
                continue
            stage_0based = int(row["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            hydro_storage[(cobre_id, stage_0based)] = float(row["value"])

    # --- Build line exchange lookup: (cobre_line_id, stage_0based) -> MW
    # Aligned via EntityAlignment.lines: each Cobre line records the
    # NEWAVE directional pair (newave_de → newave_para) plus a
    # ``reversed`` flag.  When ``reversed`` is True the NEWAVE row's
    # flow is the opposite sign of Cobre's net_flow_mw.
    line_flow: dict[tuple[int, int], float] = {}
    if not nw_line_means.is_empty() and alignment.lines:
        # Build (de, para, stage_0based) -> value for both directions.
        # NWLISTOP files are per (de, para) ordered pair; if our alignment
        # uses the opposite ordering we negate.
        df = nw_line_means.with_columns(
            (pl.col("stage") - nw_offset).cast(pl.Int64).alias("stage_0based")
        ).filter(pl.col("stage_0based") >= 0)
        nw_by_pair: dict[tuple[int, int, int], float] = {}
        for row in df.iter_rows(named=True):
            key = (
                int(row["from_submarket_code"]),
                int(row["to_submarket_code"]),
                int(row["stage_0based"]),
            )
            nw_by_pair[key] = float(row["value"])
        for line in alignment.lines:
            de = line.newave_de
            para = line.newave_para
            if de is None or para is None:
                continue
            for (de_k, para_k, s), val in nw_by_pair.items():
                if de_k == de and para_k == para:
                    sign = -1.0 if line.reversed else 1.0
                    line_flow[(line.cobre_line_id, s)] = sign * val
                elif de_k == para and para_k == de:
                    # Reversed-direction NWLISTOP row supplies the
                    # opposite sign of what our alignment expects.
                    sign = 1.0 if line.reversed else -1.0
                    # Only fill if the canonical-direction row hasn't
                    # already populated this slot.
                    line_flow.setdefault((line.cobre_line_id, s), sign * val)

    # --- Per-constraint LHS evaluation ---
    rows: list[dict] = []
    for c in constraints:
        cid = int(c["id"])
        terms = _parse_expression(c["expression"])
        if not terms:
            continue
        # Determine the set of stages we can evaluate for this
        # constraint: a stage is evaluable iff every referenced variable
        # has a NEWAVE value at that stage.
        per_stage: dict[int, float] = {}
        all_stages: set[int] = set()
        for _, _, vtype, eid in terms:
            if vtype == "hydro_storage":
                for h, s in hydro_storage:
                    if h == eid:
                        all_stages.add(s)
            elif vtype == "hydro_generation":
                for h, s in hydro_gen:
                    if h == eid:
                        all_stages.add(s)
            elif vtype in ("line_exchange", "line_direct", "line_reverse"):
                for line_id, s in line_flow:
                    if line_id == eid:
                        all_stages.add(s)

        for stage in all_stages:
            lhs = 0.0
            stage_complete = True
            for coeff, param_name, vtype, eid in terms:
                if vtype == "hydro_storage":
                    val = hydro_storage.get((eid, stage))
                elif vtype == "hydro_generation":
                    val = hydro_gen.get((eid, stage))
                elif vtype == "line_direct":
                    # ``line_direct`` is the non-negative flow in the
                    # canonical (src<tgt) direction. NEWAVE int*.out
                    # gives a single signed value per (stage, pair); we
                    # take its positive part as the stage-mean
                    # approximation. Per-block decomposition would
                    # require parsing the per-patamar rows separately.
                    signed = line_flow.get((eid, stage))
                    val = max(0.0, signed) if signed is not None else None
                elif vtype == "line_reverse":
                    signed = line_flow.get((eid, stage))
                    val = max(0.0, -signed) if signed is not None else None
                else:  # line_exchange (signed net flow)
                    val = line_flow.get((eid, stage))
                if val is None:
                    stage_complete = False
                    break
                # @rho_eq / @rho_acum parameters scale the coefficient at
                # solve time in Cobre.  We don't have NEWAVE-side
                # productivity per (hydro, stage) handy here, so skip
                # constraints with such parameters by treating them as
                # missing on this side — the chart will simply lack a
                # NEWAVE trace.  This affects VminOP only; RE/AGRINT
                # never reference @-parameters.
                if param_name is not None:
                    resolved = _resolve_param_to_column(param_name)
                    if resolved is not None:
                        stage_complete = False
                        break
                lhs += coeff * val
            if stage_complete:
                per_stage[stage] = lhs

        for stage, lhs in sorted(per_stage.items()):
            rows.append(
                {
                    "constraint_id": cid,
                    "stage_id": stage,
                    "lhs_value": lhs,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


def evaluate_lhs_cobre(
    constraints: list[dict],
    cobre_output_dir: Path,
) -> pl.DataFrame:
    """Evaluate each constraint's LHS from Cobre simulation outputs.

    Reuses the dashboard's
    :func:`cobre_bridge.dashboard.tabs.constraints_utils.evaluate_constraint_expressions`
    (which returns one row per (constraint, scenario, stage, block)) and
    collapses to mean across scenarios and blocks per (constraint, stage).

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32),
        ``lhs_value`` (Float64).  Empty when no simulation data is
        available or no constraints reference the simulation entities.
    """
    if not constraints:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    from cobre_bridge.dashboard.data import scan_entity
    from cobre_bridge.dashboard.tabs.constraints_utils import (
        evaluate_constraint_expressions,
    )

    # The dashboard's scan_entity expects a *case* directory (it appends
    # ``output/simulation/<entity>`` itself); we instead receive the
    # ``output/`` directory in the comparator, so emulate the same scan
    # one level shallower by constructing an artificial case_dir whose
    # ``output`` subdir is what we already hold.
    artificial_case_dir = cobre_output_dir.parent
    hydros_lf = scan_entity(artificial_case_dir, "hydros")
    exchanges_lf = scan_entity(artificial_case_dir, "exchanges")

    lhs_pd: pd.DataFrame = evaluate_constraint_expressions(
        constraints, hydros_lf, exchanges_lf
    )
    if lhs_pd.empty:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # Collapse to mean across scenarios and blocks per (constraint, stage).
    grouped = lhs_pd.groupby(["constraint_id", "stage_id"], as_index=False)
    means = pd.DataFrame(grouped["lhs_value"].mean())
    return pl.from_pandas(means).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


def per_stage_bounds(
    gc_bounds: pl.DataFrame,
    max_stage: int | None = None,
) -> dict[int, dict[int, float]]:
    """Reduce per-(stage, block) bounds to one value per (constraint, stage).

    When per-block bounds disagree (rare; AGRINT/RE bounds are usually
    block-invariant), the block_id=0 value is preferred.  Returns
    ``{constraint_id: {stage_id: bound}}``; stages beyond ``max_stage``
    are dropped when ``max_stage`` is supplied.
    """
    out: dict[int, dict[int, float]] = {}
    if gc_bounds.is_empty():
        return out
    df = gc_bounds
    if max_stage is not None:
        df = df.filter(pl.col("stage_id") <= max_stage)
    for cid, group in df.group_by("constraint_id"):
        cid_int = int(cid[0])
        per_stage: dict[int, float] = {}
        if "block_id" in group.columns:
            preferred = group.filter(pl.col("block_id") == 0)
            if preferred.is_empty():
                preferred = group
        else:
            preferred = group
        for row in preferred.iter_rows(named=True):
            stage = int(row["stage_id"])
            if stage not in per_stage:
                per_stage[stage] = float(row["bound"])
        out[cid_int] = per_stage
    return out
