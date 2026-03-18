"""Network entity converter: maps NEWAVE bus and line data to Cobre network JSON."""

from __future__ import annotations

import logging
from pathlib import Path

from inewave.newave import Sistema

from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

_BUSES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/buses.schema.json"
)
_LINES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/lines.schema.json"
)
_PENALTIES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/penalties.schema.json"
)

# Cobre standard defaults for hydro and line penalties.
_DEFAULT_SPILLAGE_COST = 0.01
_DEFAULT_FPHA_TURBINED_COST = 0.001
_DEFAULT_DIVERSION_COST = 0.001
_DEFAULT_STORAGE_VIOLATION_BELOW_COST = 1000.0
_DEFAULT_FILLING_TARGET_VIOLATION_COST = 1000.0
_DEFAULT_TURBINED_VIOLATION_BELOW_COST = 1000.0
_DEFAULT_OUTFLOW_VIOLATION_BELOW_COST = 1000.0
_DEFAULT_OUTFLOW_VIOLATION_ABOVE_COST = 1000.0
_DEFAULT_GENERATION_VIOLATION_BELOW_COST = 1000.0
_DEFAULT_EVAPORATION_VIOLATION_COST = 0.1
_DEFAULT_WATER_WITHDRAWAL_VIOLATION_COST = 1000.0
_DEFAULT_EXCESS_COST = 0.01
_DEFAULT_LINE_EXCHANGE_COST = 0.1
_DEFAULT_NCS_CURTAILMENT_COST = 0.001


def convert_buses(newave_dir: Path, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE subsystem data to a Cobre ``buses.json`` dict.

    Reads ``sistema.dat`` from *newave_dir*.  Each subsystem (including
    fictitious ones) becomes a bus.  Deficit segments are extracted from
    ``Sistema.custo_deficit``.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory.
    id_map:
        Pre-built ID mapping for bus IDs.

    Raises
    ------
    FileNotFoundError
        If ``sistema.dat`` is absent.
    """
    sistema_path = newave_dir / "sistema.dat"
    if not sistema_path.exists():
        raise FileNotFoundError(f"Required NEWAVE file not found: {sistema_path}")

    sistema = Sistema.read(str(sistema_path))
    deficit_df = sistema.custo_deficit

    if deficit_df is None:
        raise ValueError(
            "sistema.dat contains no deficit cost data (custo_deficit is None)"
        )

    # Build per-subsystem deficit segments.
    # Columns: codigo_submercado, nome_submercado, ficticio,
    # patamar_deficit, custo, corte
    buses_by_code: dict[int, dict] = {}

    for _, row in deficit_df.iterrows():
        code = int(row["codigo_submercado"])
        name = str(row["nome_submercado"]).strip()

        if code not in buses_by_code:
            buses_by_code[code] = {
                "newave_code": code,
                "name": name,
                "segments": [],
            }

        # corte is the MW depth of the patamar; last patamar has null depth.
        corte = row.get("corte")
        depth_mw: float | None = (
            float(corte) if corte is not None and not _is_na(corte) else None
        )
        buses_by_code[code]["segments"].append(
            {
                "patamar": int(row["patamar_deficit"]),
                "depth_mw": depth_mw,
                "cost": float(row["custo"]),
            }
        )

    buses: list[dict] = []
    for code, info in buses_by_code.items():
        # Sort segments by patamar ascending; make last segment unbounded.
        segs = sorted(info["segments"], key=lambda s: s["patamar"])
        deficit_segments: list[dict] = []
        for i, seg in enumerate(segs):
            is_last = i == len(segs) - 1
            deficit_segments.append(
                {
                    "depth_mw": None if is_last else seg["depth_mw"],
                    "cost": seg["cost"],
                }
            )

        bus_entry: dict = {
            "id": id_map.bus_id(code),
            "name": info["name"],
            "deficit_segments": deficit_segments,
        }
        buses.append(bus_entry)

    buses.sort(key=lambda b: b["id"])

    return {
        "$schema": _BUSES_SCHEMA_URL,
        "buses": buses,
    }


def convert_lines(newave_dir: Path, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE interchange limits to a Cobre ``lines.json`` dict.

    Reads ``sistema.dat`` from *newave_dir*.  Each directional interchange
    pair becomes a line using the first study month's limits as static
    capacities.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory.
    id_map:
        Pre-built ID mapping for bus IDs.

    Raises
    ------
    FileNotFoundError
        If ``sistema.dat`` is absent.
    """
    sistema_path = newave_dir / "sistema.dat"
    if not sistema_path.exists():
        raise FileNotFoundError(f"Required NEWAVE file not found: {sistema_path}")

    sistema = Sistema.read(str(sistema_path))
    limites_df = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {
            "$schema": _LINES_SCHEMA_URL,
            "lines": [],
        }

    # Columns: submercado_de, submercado_para, sentido, data, valor
    # Use the first date (earliest study month) for static capacity.
    first_date = limites_df["data"].min()
    first_month = limites_df[limites_df["data"] == first_date]

    # Build a (source, target) -> {direct_mw, reverse_mw} structure.
    # sentido == 1: direct (de -> para), sentido == 2: reverse (para -> de)
    # We normalise all pairs so source_code < target_code to deduplicate.
    pair_map: dict[tuple[int, int], dict[str, float]] = {}

    for _, row in first_month.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        valor = float(row["valor"])
        sentido = int(row["sentido"])

        # Canonical key: smaller ID first.
        src, tgt = (de, para) if de < para else (para, de)
        key = (src, tgt)

        if key not in pair_map:
            pair_map[key] = {"direct_mw": 0.0, "reverse_mw": 0.0}

        if de < para:
            # de -> para is the "direct" direction.
            if sentido == 1:
                pair_map[key]["direct_mw"] = valor
            else:
                pair_map[key]["reverse_mw"] = valor
        else:
            # de -> para is the "reverse" direction.
            if sentido == 1:
                pair_map[key]["reverse_mw"] = valor
            else:
                pair_map[key]["direct_mw"] = valor

    lines: list[dict] = []
    line_id = 0
    for (src, tgt), caps in sorted(pair_map.items()):
        src_bus = id_map.bus_id(src)
        tgt_bus = id_map.bus_id(tgt)
        src_name = _subsystem_name_from_id(src)
        tgt_name = _subsystem_name_from_id(tgt)
        line_entry: dict = {
            "id": line_id,
            "name": f"{src_name}_{tgt_name}",
            "source_bus_id": src_bus,
            "target_bus_id": tgt_bus,
            "capacity": {
                "direct_mw": caps["direct_mw"],
                "reverse_mw": caps["reverse_mw"],
            },
        }
        lines.append(line_entry)
        line_id += 1

    return {
        "$schema": _LINES_SCHEMA_URL,
        "lines": lines,
    }


def convert_penalties(newave_dir: Path) -> dict:
    """Generate a Cobre ``penalties.json`` dict from NEWAVE deficit data.

    Uses the first subsystem's first deficit tier cost as the primary
    deficit cost.  All other hydro and line penalty values use Cobre
    standard defaults.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory.

    Raises
    ------
    FileNotFoundError
        If ``sistema.dat`` is absent.
    """
    sistema_path = newave_dir / "sistema.dat"
    if not sistema_path.exists():
        raise FileNotFoundError(f"Required NEWAVE file not found: {sistema_path}")

    sistema = Sistema.read(str(sistema_path))
    deficit_df = sistema.custo_deficit

    # Primary deficit cost: first subsystem, first patamar.
    primary_deficit_cost = 0.0
    primary_depth_mw: float | None = None

    if deficit_df is not None and not deficit_df.empty:
        first_sub = deficit_df.sort_values(["codigo_submercado", "patamar_deficit"])
        first_row = first_sub.iloc[0]
        primary_deficit_cost = float(first_row["custo"])
        corte = first_row.get("corte")
        if corte is not None and not _is_na(corte):
            primary_depth_mw = float(corte)

    return {
        "$schema": _PENALTIES_SCHEMA_URL,
        "bus": {
            "deficit_segments": [
                {
                    "cost": primary_deficit_cost,
                    "depth_mw": primary_depth_mw,
                }
            ],
            "excess_cost": _DEFAULT_EXCESS_COST,
        },
        "hydro": {
            "spillage_cost": _DEFAULT_SPILLAGE_COST,
            "fpha_turbined_cost": _DEFAULT_FPHA_TURBINED_COST,
            "diversion_cost": _DEFAULT_DIVERSION_COST,
            "storage_violation_below_cost": _DEFAULT_STORAGE_VIOLATION_BELOW_COST,
            "filling_target_violation_cost": _DEFAULT_FILLING_TARGET_VIOLATION_COST,
            "turbined_violation_below_cost": _DEFAULT_TURBINED_VIOLATION_BELOW_COST,
            "outflow_violation_below_cost": _DEFAULT_OUTFLOW_VIOLATION_BELOW_COST,
            "outflow_violation_above_cost": _DEFAULT_OUTFLOW_VIOLATION_ABOVE_COST,
            "generation_violation_below_cost": _DEFAULT_GENERATION_VIOLATION_BELOW_COST,
            "evaporation_violation_cost": _DEFAULT_EVAPORATION_VIOLATION_COST,
            "water_withdrawal_violation_cost": _DEFAULT_WATER_WITHDRAWAL_VIOLATION_COST,
        },
        "line": {
            "exchange_cost": _DEFAULT_LINE_EXCHANGE_COST,
        },
        "non_controllable_source": {
            "curtailment_cost": _DEFAULT_NCS_CURTAILMENT_COST,
        },
    }


def _subsystem_name_from_id(subsystem_code: int) -> str:
    """Return a short name string for a subsystem code (fallback to str)."""
    return str(subsystem_code)


def _is_na(value: object) -> bool:
    """Return True if *value* is a pandas NA/NaN sentinel."""
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd

        return pd.isna(value)  # type: ignore[return-value]
    except (TypeError, ImportError):
        return False
