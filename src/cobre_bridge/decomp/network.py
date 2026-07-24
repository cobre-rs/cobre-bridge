"""Bus conversion for DECOMP-like decks (``SB`` + ``CD`` → ``buses.json``).

Declared subsystems become buses with their deficit curves; the implicit
transhipment node becomes a converter-created zero-load bus with no
deficit curve of its own (it carries no demand, so no deficit is ever
priced there). Exchange lines (``IA``) join this module once the upstream
accessor fix lands.

The deficit emitters are deliberately gated: the decks in hand carry a
single 100 %-depth segment with one cost, uniform across blocks and
stages. Anything richer fails loudly rather than being silently
approximated (per-block deficit costs and stage-varying deficit costs
have no Cobre encoding today).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.converters.network import (
    _BUSES_SCHEMA_URL,
    _EXCHANGE_FACTORS_SCHEMA_URL,
    _LINES_SCHEMA_URL,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_COST_COLUMN = re.compile(r"^custo_(\d+)$")
_LIMIT_COLUMN = re.compile(r"^limite_superior_(\d+)$")
_FULL_DEPTH_PERCENT = 100.0
_PUMPING_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/pumping_stations.schema.json"
)


def _bus_deficit_costs(dadger: Dadger) -> dict[int, float]:
    """Extract one deficit cost per subsystem code from the ``CD`` records.

    Enforces the single-segment shape the Cobre bus model encodes: one
    curve per subsystem, 100 % depth, one cost uniform across blocks and
    across every declared stage.
    """
    cd = dadger.cd(df=True)
    if cd is None or cd.empty:
        return {}

    cost_columns = [c for c in cd.columns if _COST_COLUMN.match(c)]
    limit_columns = [c for c in cd.columns if _LIMIT_COLUMN.match(c)]

    costs: dict[int, float] = {}
    for code, group in cd.groupby("codigo_submercado"):
        code = int(code)
        if group["codigo_curva"].nunique() != 1:
            raise ValueError(
                f"subsystem {code}: multi-segment deficit curves are not "
                "supported yet (multiple codigo_curva values)"
            )
        values: set[float] = set()
        for _, row in group.iterrows():
            for column in cost_columns:
                value = row[column]
                if not pd.isna(value):
                    values.add(float(value))
            for column in limit_columns:
                limit = row[column]
                if not pd.isna(limit) and float(limit) != _FULL_DEPTH_PERCENT:
                    raise ValueError(
                        f"subsystem {code}: deficit segment depth "
                        f"{float(limit)} % is not supported yet (expected "
                        f"{_FULL_DEPTH_PERCENT} %)"
                    )
        if len(values) != 1:
            raise ValueError(
                f"subsystem {code}: deficit cost must be one value uniform "
                f"across blocks and stages; got {sorted(values)}"
            )
        costs[code] = values.pop()
    return costs


def convert_buses(
    dadger: Dadger,
    id_map: DecompIdMap,
    start_date: date,
) -> dict:
    """Build the ``buses.json`` dict from ``SB`` + ``CD`` records.

    Buses model subsystems, which have no commissioning date; the study
    start date serves as the canonical-ordering key. Buses without a
    ``CD`` record (the fictitious subsystem, the transhipment bus) omit
    ``deficit_segments`` and defer to the global default — they carry no
    load, so the value is never priced.
    """
    costs = _bus_deficit_costs(dadger)
    op_date = start_date.isoformat()

    buses: list[dict] = []
    for code in id_map.bus_codes:
        entry: dict = {
            "id": id_map.bus_id(code),
            "name": id_map.bus_name(id_map.bus_id(code)),
            "operational_start_date": op_date,
        }
        if code in costs:
            entry["deficit_segments"] = [{"depth_mw": None, "cost": costs[code]}]
        buses.append(entry)

    buses.append(
        {
            "id": id_map.transhipment_bus_id,
            "name": id_map.bus_name(id_map.transhipment_bus_id),
            "operational_start_date": op_date,
        }
    )

    return {"$schema": _BUSES_SCHEMA_URL, "buses": buses}


def convert_lines_placeholder() -> dict:
    """An empty (but structurally required) ``lines.json``.

    The exchange network waits on the ``IA`` accessor fix upstream; until
    it lands the buses are deliberately unconnected — each subsystem
    self-balances against its own deficit cost. (The file being mandatory
    for a lineless study is tracked cobre-gap C4,
    ``plans/conversion-found-improvements.md`` in the cobre repo.)
    """
    _LOG_MESSAGE = (
        "exchange network deferred (IA accessor fix upstream): emitting an "
        "empty lines.json — subsystems are unconnected and self-balance"
    )
    logging.getLogger(__name__).warning(_LOG_MESSAGE)
    return {"$schema": _LINES_SCHEMA_URL, "lines": []}


def _ia_dense(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> dict[tuple[str, str], list[dict]]:
    """Read ``IA`` into dense per-stage per-block limits per exchange pair.

    Returns ``{(name_de, name_para): [{"de_para": [...], "para_de": [...]}
    per stage]}`` with declared stages forward-filled (stage 1 mandatory).
    """
    ia = dadger.ia(df=True)
    if ia is None or ia.empty:
        return {}

    declared: dict[tuple[str, str], dict[int, dict]] = {}
    for _, row in ia.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"IA stage {int(row['estagio'])} outside the calendar "
                f"(1..{len(calendar)})"
            )
        n_blocks = len(calendar[stage_index].block_hours)

        def _limits(
            prefix: str, row: pd.Series = row, n: int = n_blocks
        ) -> list[float]:
            values = []
            for k in range(1, n + 1):
                value = row[f"{prefix}_{k}"]
                if pd.isna(value):
                    raise ValueError(
                        f"IA {row['nome_submercado_de']}-"
                        f"{row['nome_submercado_para']} stage "
                        f"{int(row['estagio'])}: missing {prefix}_{k}"
                    )
                values.append(float(value))
            return values

        pair = (
            str(row["nome_submercado_de"]).strip(),
            str(row["nome_submercado_para"]).strip(),
        )
        declared.setdefault(pair, {})[stage_index] = {
            "de_para": _limits("limite_de_para"),
            "para_de": _limits("limite_para_de"),
        }

    dense: dict[tuple[str, str], list[dict]] = {}
    for pair, stages in declared.items():
        if 0 not in stages:
            raise ValueError(
                f"IA pair {pair[0]}-{pair[1]} does not declare stage 1; "
                "sparse-stage inheritance has no base"
            )
        per_stage: list[dict] = []
        for stage in calendar:
            per_stage.append(
                stages.get(stage.index, per_stage[-1] if per_stage else {})
            )
        dense[pair] = per_stage
    return dense


def convert_lines(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    start_date: date,
) -> tuple[dict, pa.Table, dict]:
    """Convert the ``IA`` exchange network.

    Returns ``(lines.json dict, line_bounds table, exchange_factors dict)``:
    one line per declared pair, the per-stage base capacity as the maximum
    block limit of each direction, and the per-block shape as exchange
    factors (``factor_b = limit_b / base``) — emitted only for
    (line, stage) entries whose blocks actually differ. The unbounded
    sentinel (99999) passes through as a plain large capacity.
    """
    dense = _ia_dense(dadger, calendar)
    op_date = start_date.isoformat()

    pairs = sorted(
        dense,
        key=lambda p: (id_map.bus_id_by_name(p[0]), id_map.bus_id_by_name(p[1])),
    )

    lines: list[dict] = []
    bounds_line_ids: list[int] = []
    bounds_stage_ids: list[int] = []
    bounds_direct: list[float] = []
    bounds_reverse: list[float] = []
    factor_entries: list[dict] = []

    for line_id, pair in enumerate(pairs):
        per_stage = dense[pair]
        base_first = per_stage[0]
        lines.append(
            {
                "id": line_id,
                "name": f"{pair[0]}-{pair[1]}",
                "operational_start_date": op_date,
                "source_bus_id": id_map.bus_id_by_name(pair[0]),
                "target_bus_id": id_map.bus_id_by_name(pair[1]),
                "capacity": {
                    "direct_mw": max(base_first["de_para"]),
                    "reverse_mw": max(base_first["para_de"]),
                },
            }
        )
        for stage in calendar:
            limits = per_stage[stage.index]
            direct_base = max(limits["de_para"])
            reverse_base = max(limits["para_de"])
            bounds_line_ids.append(line_id)
            bounds_stage_ids.append(stage.index)
            bounds_direct.append(direct_base)
            bounds_reverse.append(reverse_base)

            block_factors = []
            uniform = True
            for b, (d, r) in enumerate(
                zip(limits["de_para"], limits["para_de"], strict=True)
            ):
                d_factor = d / direct_base if direct_base > 0 else 1.0
                r_factor = r / reverse_base if reverse_base > 0 else 1.0
                if d_factor <= 0.0 or r_factor <= 0.0:
                    raise ValueError(
                        f"line {pair[0]}-{pair[1]} stage {stage.index}: zero "
                        f"block limit (block {b}) has no factor encoding"
                    )
                if d_factor != 1.0 or r_factor != 1.0:
                    uniform = False
                block_factors.append(
                    {
                        "block_id": b,
                        "direct_factor": d_factor,
                        "reverse_factor": r_factor,
                    }
                )
            if not uniform:
                factor_entries.append(
                    {
                        "line_id": line_id,
                        "stage_id": stage.index,
                        "block_factors": block_factors,
                    }
                )

    bounds = pa.table(
        {
            "line_id": pa.array(bounds_line_ids, type=pa.int32()),
            "stage_id": pa.array(bounds_stage_ids, type=pa.int32()),
            "direct_mw": pa.array(bounds_direct, type=pa.float64()),
            "reverse_mw": pa.array(bounds_reverse, type=pa.float64()),
        }
    )
    return (
        {"$schema": _LINES_SCHEMA_URL, "lines": lines},
        bounds,
        {
            "$schema": _EXCHANGE_FACTORS_SCHEMA_URL,
            "exchange_factors": factor_entries,
        },
    )


def convert_pumping_stations(
    dadger: Dadger,
    id_map: DecompIdMap,
    start_date: date,
) -> dict:
    """Convert the ``UE`` pumping stations (1:1).

    Water is lifted from the downstream plant to the upstream one; both
    must be operated plants.
    """
    ue = dadger.ue(df=True)
    if ue is None or ue.empty:
        return {"$schema": _PUMPING_SCHEMA_URL, "pumping_stations": []}

    op_date = start_date.isoformat()
    stations: list[dict] = []
    for _, row in ue.sort_values("codigo_usina").iterrows():
        name = str(row["nome_usina"]).strip()
        source = int(row["codigo_usina_jusante"])
        destination = int(row["codigo_usina_montante"])
        stations.append(
            {
                "id": len(stations),
                "name": name,
                "operational_start_date": op_date,
                "bus_id": id_map.bus_id(int(row["codigo_submercado"])),
                "source_hydro_id": id_map.hydro_id(source),
                "destination_hydro_id": id_map.hydro_id(destination),
                "consumption_mw_per_m3s": float(row["taxa_consumo"]),
                "flow": {
                    "min_m3s": float(row["vazao_minima_bombeavel"]),
                    "max_m3s": float(row["vazao_maxima_bombeavel"]),
                },
            }
        )
    return {"$schema": _PUMPING_SCHEMA_URL, "pumping_stations": stations}
