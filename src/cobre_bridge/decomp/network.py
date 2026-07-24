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

from cobre_bridge.converters.network import _BUSES_SCHEMA_URL, _LINES_SCHEMA_URL

if TYPE_CHECKING:
    from datetime import date

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap

_COST_COLUMN = re.compile(r"^custo_(\d+)$")
_LIMIT_COLUMN = re.compile(r"^limite_superior_(\d+)$")
_FULL_DEPTH_PERCENT = 100.0


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
