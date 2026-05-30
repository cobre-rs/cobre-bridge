"""Read Cobre FCF cut coefficients (``output/policy/cuts/stage_NNN.bin``).

Counterpart to ``newave_cut_investigation.py``. Cobre stores per-stage cuts as
FlatBuffers (root ``StageCuts``, schema ``crates/cobre-io/schemas/policy.fbs``).
Each ``Cut`` has ``intercept`` + positional ``coefficients[i]`` mapped to a
reservoir via ``state_dictionary.json``; ``coefficients[i]`` is the marginal water
value of Cobre hydro ``entity_id``. ``stage_s.bin`` holds the cost-to-go that
stage ``s`` accesses; the terminal stage's file is empty.

Units: Cobre cut values are in **10⁶ R$**; ``cobre_water_values`` normalizes to R$
(R$/hm³ for the gradients) via ``COBRE_MONETARY_UNIT_RS``.

We decode ``.bin -> JSON`` with ``flatc`` (no Python flatbuffers dependency).

Importable API:
    decode_stage_cuts(case_dir, stage) -> dict
    cobre_water_values(case_dir, stage, *, iteration, forward_pass, reduce)
        -> DataFrame[entity_id, nome, water_value_cobre]
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

FLATC = "flatc"
SCHEMA_PATH = Path("/home/rjalves/git/cobre/crates/cobre-io/schemas/policy.fbs")
COBRE_MONETARY_UNIT_RS = 1.0e6  # Cobre cut values are in 10⁶ R$ -> ×1e6 = R$


def _reduce_rows(array: np.ndarray, reduce: str) -> np.ndarray:
    """Reduce a (n_cuts, …) array along axis 0 to a single cut's values."""
    if reduce == "first":
        return array[0]
    if reduce == "last":
        return array[-1]
    if reduce == "mean":
        return array.mean(axis=0)
    if reduce == "median":
        return np.median(array, axis=0)
    raise ValueError(f"unknown reduce={reduce!r} (use first|last|mean|median)")


def decode_stage_cuts(case_dir: Path | str, stage_id: int) -> dict:
    """Decode ``policy/cuts/stage_NNN.bin`` (FlatBuffers) to a dict via ``flatc``."""
    bin_path = (
        Path(case_dir) / "output" / "policy" / "cuts" / f"stage_{stage_id:03d}.bin"
    )
    if not bin_path.exists():
        raise FileNotFoundError(f"cut file not found: {bin_path}")
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"FlatBuffers schema not found: {SCHEMA_PATH} "
            "(point SCHEMA_PATH at your Cobre checkout's policy.fbs)"
        )
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                FLATC,
                "-t",
                "--strict-json",
                "--raw-binary",
                "--root-type",
                "StageCuts",
                "-o",
                tmp,
                str(SCHEMA_PATH),
                "--",
                str(bin_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads((Path(tmp) / f"stage_{stage_id:03d}.json").read_text())


def hydro_id_to_name(case_dir: Path | str) -> dict[int, str]:
    """Map Cobre hydro id -> name (from system/hydros.json)."""
    hydros = json.loads((Path(case_dir) / "system" / "hydros.json").read_text())[
        "hydros"
    ]
    return {h["id"]: h["name"] for h in hydros}


def cobre_water_values(
    case_dir: Path | str,
    stage_id: int,
    *,
    iteration: int | None = None,
    forward_pass: int | None = None,
    reduce: str | None = None,
    select: str | None = None,
) -> pd.DataFrame:
    """Per-reservoir water value (cut coefficient, R$/hm³) for selected cut(s).

    Filter with ``iteration`` and/or ``forward_pass`` (``forward_pass_index``). If
    more than one cut remains, pass ``reduce`` (first|last|mean|median); with
    ``reduce=None`` exactly one cut must remain. ``select``
    ('oldest'|'newest'|'first_real') picks one cut by ``cut_id`` ('first_real' ==
    oldest; Cobre has no placeholder). Returns ``[entity_id, nome,
    water_value_cobre]``; ``.attrs`` has ``intercept`` and ``n_cuts``. Empty for a
    terminal (no-cut) stage.
    """
    data = decode_stage_cuts(case_dir, stage_id)
    cuts = data["cuts"]
    state_vars = json.loads(
        (
            Path(case_dir)
            / "output"
            / "training"
            / "dictionaries"
            / "state_dictionary.json"
        ).read_text()
    )["state_variables"]
    names = hydro_id_to_name(case_dir)

    if not cuts:
        out = pd.DataFrame(columns=["entity_id", "nome", "water_value_cobre"])
        out.attrs["intercept"] = float("nan")
        out.attrs["n_cuts"] = 0
        return out

    sel = cuts
    if iteration is not None:
        sel = [c for c in sel if c["iteration"] == iteration]
    if forward_pass is not None:
        sel = [c for c in sel if c["forward_pass_index"] == forward_pass]
    if len(sel) == 0:
        raise ValueError(
            f"no Cobre cut for iteration={iteration}, forward_pass={forward_pass} "
            f"at stage {stage_id}"
        )
    if select in ("oldest", "first_real"):  # min cut_id; Cobre has no placeholder
        sel = [min(sel, key=lambda c: c["cut_id"])]
    elif select == "newest":  # last cut built = latest iteration
        sel = [max(sel, key=lambda c: c["cut_id"])]
    elif select is not None:
        raise ValueError(f"unknown select={select!r} (use oldest|newest|first_real)")
    if len(sel) > 1 and reduce is None:
        raise ValueError(
            f"{len(sel)} Cobre cuts match; pass reduce= or filter further "
            "(iteration / forward_pass)"
        )

    coef = _reduce_rows(
        np.array([c["coefficients"] for c in sel], dtype=float), reduce or "first"
    )
    intercept = float(
        _reduce_rows(
            np.array([c["intercept"] for c in sel], dtype=float), reduce or "first"
        )
    )

    out = pd.DataFrame(
        {
            "entity_id": [sv["entity_id"] for sv in state_vars],
            "nome": [names.get(sv["entity_id"], "") for sv in state_vars],
            "water_value_cobre": coef * COBRE_MONETARY_UNIT_RS,
        }
    )
    out.attrs["intercept"] = intercept * COBRE_MONETARY_UNIT_RS
    out.attrs["n_cuts"] = int(len(sel))
    return out


def _main() -> None:
    case_dir = Path("./example/cobre_rodada")
    wv = cobre_water_values(case_dir, 0, iteration=1)
    print(
        f"Cobre stage 0, iteração 1: {wv.attrs['n_cuts']} corte, "
        f"intercept={wv.attrs['intercept']:.4e} R$"
    )
    top = wv.reindex(wv["water_value_cobre"].abs().sort_values(ascending=False).index)
    print("top 12 |coef| (valor da água, R$/hm³):")
    for _, r in top.head(12).iterrows():
        print(
            f"  hydro {int(r['entity_id']):>3} {r['nome']:<13} "
            f"{r['water_value_cobre']:>16.2f}"
        )


if __name__ == "__main__":
    _main()
