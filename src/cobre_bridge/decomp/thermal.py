"""Thermal conversion for DECOMP-like decks (``CT`` records).

``CT`` declares, per (plant, stage), the per-block incremental cost
(``cvu``), availability (``disponibilidade``) and inflexibility
(``inflexibilidade``), sparsely by stage (later stages inherit the last
declared record). ``convert_thermal_bounds`` emits a stage-level base row
(hours-weighted ``min``/``max``/``cost``, ``block_id = None``) plus, only
where a stage's per-block ``disp``/``inflex`` values actually differ across
blocks, sparse per-block override rows (``block_id = 0..n-1``) carrying the
exact per-block ``min``/``max`` — the block-hour fold that used to be the
only representation is now just the base row's summary. ``cost_per_mwh`` is
not block-eligible (cobre rule 37) and stays on the base row only.

GNL plants live in the anticipated-dispatch file and are converted by the
anticipation track, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.converters.thermal import _SCHEMA_URL
from cobre_bridge.decomp.temporal import hours_weighted as _hours_weighted

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage


def _ct_dense(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict]:
    """Read ``CT`` into dense per-stage block values per plant.

    Returns ``{code: {"name", "bus_code", "stages": [{cvu, disp, inflex}
    per block, one entry per stage]}}`` with declared stages forward-filled
    (stage 1 mandatory).
    """
    ct = dadger.ct(df=True)
    if ct is None or ct.empty:
        return {}

    declared: dict[int, dict] = {}
    for _, row in ct.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"CT stage {int(row['estagio'])} outside the calendar "
                f"(1..{len(calendar)})"
            )
        n_blocks = len(calendar[stage_index].block_hours)

        def _blocks(
            prefix: str, row: pd.Series = row, n: int = n_blocks
        ) -> list[float]:
            return [
                0.0 if pd.isna(row[f"{prefix}_{k}"]) else float(row[f"{prefix}_{k}"])
                for k in range(1, n + 1)
            ]

        code = int(row["codigo_usina"])
        plant = declared.setdefault(
            code,
            {
                "name": str(row["nome_usina"]).strip(),
                "bus_code": int(row["codigo_submercado"]),
                "declared": {},
            },
        )
        plant["declared"][stage_index] = {
            "cvu": _blocks("cvu"),
            "disp": _blocks("disponibilidade"),
            "inflex": _blocks("inflexibilidade"),
        }

    for code, plant in declared.items():
        if 0 not in plant["declared"]:
            raise ValueError(
                f"CT plant {code} ({plant['name']}) does not declare stage 1; "
                "sparse-stage inheritance has no base"
            )
        dense: list[dict] = []
        for stage in calendar:
            dense.append(plant["declared"].get(stage.index, dense[-1] if dense else {}))
        plant["stages"] = dense
        del plant["declared"]
    return declared


def convert_thermals(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    start_date: date,
) -> dict:
    """Build ``thermals.json`` from the ``CT`` records (stage-1 base values)."""
    plants = _ct_dense(dadger, calendar)
    op_date = start_date.isoformat()
    first = calendar[0]

    thermals: list[dict] = []
    for code in id_map.thermal_codes:
        plant = plants.get(code)
        if plant is None:
            raise ValueError(f"thermal code {code} has no CT records")
        base = plant["stages"][0]
        thermals.append(
            {
                "id": id_map.thermal_id(code),
                "name": plant["name"],
                "operational_start_date": op_date,
                "bus_id": id_map.bus_id(plant["bus_code"]),
                "cost_per_mwh": _hours_weighted(base["cvu"], first),
                "generation": {
                    "min_mw": _hours_weighted(base["inflex"], first),
                    "max_mw": _hours_weighted(base["disp"], first),
                },
            }
        )
    return {"$schema": _SCHEMA_URL, "thermals": thermals}


_THERMAL_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("thermal_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("min_generation_mw", pa.float64(), nullable=False),
        pa.field("max_generation_mw", pa.float64(), nullable=False),
        pa.field("cost_per_mwh", pa.float64(), nullable=True),
        pa.field("block_id", pa.int32(), nullable=True),
    ]
)


def convert_thermal_bounds(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """Thermal bounds: a stage-level base row plus sparse per-block overrides.

    Every ``(thermal, stage)`` gets a base row (``block_id = None``) carrying
    the hours-weighted ``min_generation_mw`` / ``max_generation_mw`` /
    ``cost_per_mwh`` — unchanged from the pre-block-axis fold, so any
    stage-level consumer sees the same numbers as before. Where — and only
    where — the stage's per-block ``disponibilidade`` (max) or
    ``inflexibilidade`` (min) values are not block-uniform, one additional
    override row per block is emitted with the block's own exact ``min``/
    ``max`` and ``cost_per_mwh = None`` (cost has no per-block LP variable;
    cobre rule 37 rejects it there). This mirrors ``convert_lines``' sparse
    base-plus-override convention (``decomp/network.py``) exactly.
    """
    plants = _ct_dense(dadger, calendar)

    thermal_ids: list[int] = []
    stage_ids: list[int] = []
    mins: list[float] = []
    maxs: list[float] = []
    costs: list[float | None] = []
    block_ids: list[int | None] = []
    for code in id_map.thermal_codes:
        plant = plants.get(code)
        if plant is None:
            raise ValueError(f"thermal code {code} has no CT records")
        thermal_id = id_map.thermal_id(code)
        for stage in calendar:
            values = plant["stages"][stage.index]
            disp = values["disp"]
            inflex = values["inflex"]

            thermal_ids.append(thermal_id)
            stage_ids.append(stage.index)
            mins.append(_hours_weighted(inflex, stage))
            maxs.append(_hours_weighted(disp, stage))
            costs.append(_hours_weighted(values["cvu"], stage))
            block_ids.append(None)

            uniform = all(d == disp[0] for d in disp) and all(
                m == inflex[0] for m in inflex
            )
            if not uniform:
                for b in range(len(disp)):
                    thermal_ids.append(thermal_id)
                    stage_ids.append(stage.index)
                    mins.append(inflex[b])
                    maxs.append(disp[b])
                    costs.append(None)
                    block_ids.append(b)

    return pa.table(
        {
            "thermal_id": pa.array(thermal_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "min_generation_mw": pa.array(mins, type=pa.float64()),
            "max_generation_mw": pa.array(maxs, type=pa.float64()),
            "cost_per_mwh": pa.array(costs, type=pa.float64()),
            "block_id": pa.array(block_ids, type=pa.int32()),
        },
        schema=_THERMAL_BOUNDS_SCHEMA,
    )
