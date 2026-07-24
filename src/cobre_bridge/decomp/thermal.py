"""Thermal conversion for DECOMP-like decks (``CT`` records).

``CT`` declares, per (plant, stage), the per-block incremental cost
(``cvu``), availability (``disponibilidade``) and inflexibility
(``inflexibilidade``), sparsely by stage (later stages inherit the last
declared record). Cobre's thermal bounds carry no block dimension until
the unit-groups work activates the ``block_id`` column, so per-block
values are **hours-weighted** into per-stage bounds — a documented interim
approximation: one summary warning reports the worst per-block relative
spread folded away per conversion.

GNL plants live in the anticipated-dispatch file and are converted by the
anticipation track, not here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.converters.thermal import _SCHEMA_URL

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)


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


def _hours_weighted(values: Sequence[float], stage: OperativeStage) -> float:
    return (
        sum(v * h for v, h in zip(values, stage.block_hours, strict=True))
        / stage.total_hours
    )


def _relative_spread(values: Sequence[float]) -> float:
    top = max(values)
    if top == 0.0:
        return 0.0
    return (top - min(values)) / top


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


def convert_thermal_bounds(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """Per-stage thermal bounds, hours-weighted from the per-block values.

    One summary warning reports the largest per-block relative spread the
    aggregation folds away (availability and inflexibility); it goes to
    zero when the unit-groups ``block_id`` convention lands and the emitter
    switches to per-block rows.
    """
    plants = _ct_dense(dadger, calendar)

    thermal_ids: list[int] = []
    stage_ids: list[int] = []
    mins: list[float] = []
    maxs: list[float] = []
    costs: list[float] = []
    worst_spread = 0.0
    for code in id_map.thermal_codes:
        plant = plants.get(code)
        if plant is None:
            raise ValueError(f"thermal code {code} has no CT records")
        for stage in calendar:
            values = plant["stages"][stage.index]
            thermal_ids.append(id_map.thermal_id(code))
            stage_ids.append(stage.index)
            mins.append(_hours_weighted(values["inflex"], stage))
            maxs.append(_hours_weighted(values["disp"], stage))
            costs.append(_hours_weighted(values["cvu"], stage))
            worst_spread = max(
                worst_spread,
                _relative_spread(values["disp"]),
                _relative_spread(values["inflex"]),
            )

    if worst_spread > 0.0:
        _LOG.warning(
            "thermal per-block values hours-weighted into per-stage bounds "
            "(worst per-block relative spread folded away: %.1f%%); exact "
            "per-block bounds land with the unit-groups block_id convention",
            worst_spread * 100.0,
        )

    return pa.table(
        {
            "thermal_id": pa.array(thermal_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "min_generation_mw": pa.array(mins, type=pa.float64()),
            "max_generation_mw": pa.array(maxs, type=pa.float64()),
            "cost_per_mwh": pa.array(costs, type=pa.float64()),
        }
    )
