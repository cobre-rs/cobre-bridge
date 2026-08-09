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
        bus_id = id_map.bus_id(code)
        entry: dict = {
            "id": bus_id,
            "name": id_map.bus_name(bus_id),
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
    logging.getLogger(__name__).warning(
        "exchange network deferred (IA accessor fix upstream): emitting an "
        "empty lines.json — subsystems are unconnected and self-balance"
    )
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


_LINE_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("line_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("direct_mw", pa.float64(), nullable=False),
        pa.field("reverse_mw", pa.float64(), nullable=False),
    ]
)


def convert_lines(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    start_date: date,
) -> tuple[dict, pa.Table]:
    """Convert the ``IA`` exchange network.

    Returns ``(lines.json dict, line_bounds table)``: one line per declared
    pair, plus a ``line_bounds`` table carrying a stage-level base row
    (``block_id = None``) whose direct/reverse capacity is the maximum block
    limit of each direction, and — only for (line, stage) entries whose
    blocks actually differ from that base — one per-block override row
    (``block_id = 0..n-1``) per declared block. Base and block rows coexist
    (cobre rule 36). Block rows carry the ``IA`` record's own per-block
    limit as an absolute MW value, read directly with no factor round-trip.
    The unbounded sentinel (99999) passes through as a plain large capacity.
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
    bounds_block_ids: list[int | None] = []

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
            bounds_block_ids.append(None)

            # Per-block rows carry the IA record's own de_para/para_de
            # values directly, as absolute MW — no factor round-trip, so
            # no division and no reconstruction error. A block whose limit
            # is 0.0 is now an ordinary bound (direct_mw = 0.0 / reverse_mw
            # = 0.0): the previous factor encoding had no representation
            # for a zero limit and raised on it; under absolute MW that
            # case no longer exists, so the raise is gone and nothing
            # replaces it.
            blocks = list(zip(limits["de_para"], limits["para_de"], strict=True))
            uniform = all(d == direct_base and r == reverse_base for d, r in blocks)
            if not uniform:
                for b, (d, r) in enumerate(blocks):
                    bounds_line_ids.append(line_id)
                    bounds_stage_ids.append(stage.index)
                    bounds_direct.append(d)
                    bounds_reverse.append(r)
                    bounds_block_ids.append(b)

    bounds = pa.table(
        {
            "line_id": pa.array(bounds_line_ids, type=pa.int32()),
            "stage_id": pa.array(bounds_stage_ids, type=pa.int32()),
            "block_id": pa.array(bounds_block_ids, type=pa.int32()),
            "direct_mw": pa.array(bounds_direct, type=pa.float64()),
            "reverse_mw": pa.array(bounds_reverse, type=pa.float64()),
        },
        schema=_LINE_BOUNDS_SCHEMA,
    )
    return (
        {"$schema": _LINES_SCHEMA_URL, "lines": lines},
        bounds,
    )


def pumping_station_id_map(dadger: Dadger) -> dict[int, int]:
    """``{codigo_usina: 0-based pumping-station id}`` — the single authority
    for ``UE`` id assignment.

    The id is the row's position in ``codigo_usina``-sorted order — exactly
    what :func:`convert_pumping_stations` assigns via ``len(stations)``.
    ``single_term_bounds.single_term_bound_contributions``'s ``QBOM`` path
    resolves its pumping-station entity id through this same map (its
    ``pumping_station_ids`` argument), since the QBOM term's ``code`` is a
    pumping-station ``codigo_usina``, not a hydro one (epic-07,
    ticket-023).
    """
    ue = dadger.ue(df=True)
    if ue is None or ue.empty:
        return {}
    codes = sorted(int(c) for c in ue["codigo_usina"])
    return {code: station_id for station_id, code in enumerate(codes)}


def convert_pumping_stations(
    dadger: Dadger,
    id_map: DecompIdMap,
    start_date: date,
) -> dict:
    """Convert the ``UE`` pumping stations (1:1).

    Water is lifted from the downstream plant to the upstream one; both
    must be operated plants. Station ids come from
    :func:`pumping_station_id_map`, the single authority for ``UE`` id
    assignment.
    """
    ue = dadger.ue(df=True)
    if ue is None or ue.empty:
        return {"$schema": _PUMPING_SCHEMA_URL, "pumping_stations": []}

    station_ids = pumping_station_id_map(dadger)
    op_date = start_date.isoformat()
    stations: list[dict] = []
    for _, row in ue.sort_values("codigo_usina").iterrows():
        code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        source = int(row["codigo_usina_jusante"])
        destination = int(row["codigo_usina_montante"])
        stations.append(
            {
                "id": station_ids[code],
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
