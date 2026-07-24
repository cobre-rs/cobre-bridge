"""Hydro conversion for DECOMP-like decks (registry ``hidr.dat`` + ``UH``).

The registry file is byte-identical across the two source families, so the
shared row-level physics helpers (rated/derated capability, cota
polynomial, hydraulic losses) are reused verbatim. Scope is the ratified
loop-closing milestone: faithful registry, cascade, capability and initial
storage — with everything whose faithful treatment is gated on later
features deferred **loudly** (one summary log warning each): registry
overrides (``AC``), travel time (``VI``), minimum-outflow joins
(``RQ`` × long-term means), per-stage availability (``FD``/``MP``),
FPHA/tailrace/evaporation models, and the two-frequency split of plant 66.

``UH`` rows without an initial volume (the coupling-only registrations)
are excluded from the operated set and reported.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
from inewave.newave import Hidr

from cobre_bridge.converters.hydro import (
    _PRODUCTION_MODELS_SCHEMA_URL,
    _SCHEMA_URL,
    _apply_hydraulic_loss,
    _compute_max_turbined_simple,
    _mean_cota_over_volume,
)

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)


def read_hidr(path: Path) -> pd.DataFrame:
    """Read the hydro registry file, indexed by plant code."""
    df = Hidr.read(str(path)).cadastro
    if df is None or df.empty:
        raise FileNotFoundError(f"{path} has no registry data")
    return df


def _operated_uh(dadger: Dadger) -> pd.DataFrame:
    """The ``UH`` rows carrying an initial volume (the operated plants).

    Rows without one (coupling-only registrations) are reported and
    excluded — their terminal-value treatment is the boundary importer's
    D3 territory, not the registry's.
    """
    uh = dadger.uh(df=True)
    if uh is None or uh.empty:
        raise ValueError("the deck has no UH records; cannot convert hydros")
    operated = uh[uh["volume_inicial"].notna()]
    excluded = uh[uh["volume_inicial"].isna()]
    if not excluded.empty:
        _LOG.warning(
            "%d UH registration(s) without an initial volume excluded from "
            "the operated set (coupling-only): codes %s",
            len(excluded),
            sorted(int(c) for c in excluded["codigo_usina"]),
        )
    return operated


def _downstream_operated(
    hidr: pd.DataFrame,
    code: int,
    operated: set[int],
) -> int | None:
    """Walk the registry cascade to the next *operated* plant downstream.

    Non-operated intermediates are skipped through (their routing is
    instantaneous absence — the water continues down the declared chain);
    code 0 is the sink.
    """
    visited = {code}
    current = int(hidr.loc[code, "codigo_usina_jusante"])
    while current != 0 and current not in operated:
        if current in visited or current not in hidr.index:
            _LOG.warning(
                "cascade walk from plant %d hit an invalid link at %d; "
                "treating as a sink",
                code,
                current,
            )
            return None
        visited.add(current)
        current = int(hidr.loc[current, "codigo_usina_jusante"])
    return current if current != 0 else None


def convert_hydros(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    start_date: date,
) -> dict:
    """Build ``hydros.json`` for the operated plants.

    Capability is the availability-derated sum of rated unit flows/powers
    (the shared simple formula); the production model is constant
    productivity, with the value emitted separately
    (:func:`convert_energy_productivity`). Deferred fidelity is logged
    once per family.
    """
    operated = _operated_uh(dadger)
    operated_codes = set(id_map.hydro_codes)
    min_outflow_by_code: dict[int, float] = {}
    for _, row in operated.iterrows():
        value = row.get("vazao_defluente_minima")
        if value is not None and not pd.isna(value):
            min_outflow_by_code[int(row["codigo_usina"])] = float(value)

    op_date = start_date.isoformat()
    hydros: list[dict] = []
    for code in id_map.hydro_codes:
        if code not in hidr.index:
            raise ValueError(f"UH plant {code} is not in the hydro registry")
        hreg = hidr.loc[code]
        name = str(hreg["nome_usina"]).strip()
        downstream = _downstream_operated(hidr, code, operated_codes)
        max_turbined, max_generation = _compute_max_turbined_simple(hreg, name)
        entry: dict = {
            "id": id_map.hydro_id(code),
            "name": name,
            "operational_start_date": op_date,
            "bus_id": id_map.bus_id(int(hreg["submercado"])),
            "downstream_id": (
                None if downstream is None else id_map.hydro_id(downstream)
            ),
            "reservoir": {
                "min_storage_hm3": float(hreg["volume_minimo"]),
                "max_storage_hm3": float(hreg["volume_maximo"]),
            },
            "outflow": {
                "min_outflow_m3s": min_outflow_by_code.get(code, 0.0),
                "max_outflow_m3s": None,
            },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": 0.0,
                "max_generation_mw": max_generation,
            },
        }
        hydros.append(entry)

    _LOG.warning(
        "deferred hydro fidelity (loop-closing milestone): AC registry "
        "overrides, VI travel time, RQ minimum-outflow joins, FD/MP "
        "availability, FPHA/tailrace/evaporation models, and the plant-66 "
        "frequency split are not applied yet"
    )
    return {"$schema": _SCHEMA_URL, "hydros": hydros}


def convert_initial_storage(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
) -> list[dict]:
    """Initial reservoir volumes from ``UH`` (% of useful → hm³)."""
    operated = _operated_uh(dadger)
    storage: list[dict] = []
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        hreg = hidr.loc[code]
        v_min = float(hreg["volume_minimo"])
        v_max = float(hreg["volume_maximo"])
        pct = float(row["volume_inicial"])
        value = v_min + (pct / 100.0) * (v_max - v_min)
        value = min(max(value, v_min), v_max)
        storage.append({"hydro_id": id_map.hydro_id(code), "value_hm3": value})
        dead = row.get("volume_morto_inicial")
        if dead is not None and not pd.isna(dead):
            _LOG.warning(
                "plant %d declares an initial dead volume (%s); "
                "dead-volume filling is not converted yet",
                code,
                dead,
            )
    storage.sort(key=lambda e: e["hydro_id"])
    return storage


def convert_energy_productivity(
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
) -> pa.Table:
    """Per-plant constant equivalent productivity (all stages).

    ``ρ_eq = ρ_esp · h_net`` with the gross head as the volume-averaged
    cota over the operating range minus the mean tailrace level, and the
    registry's hydraulic-loss model applied — the same construction the
    other converter family uses for its constant-productivity plants.
    """
    hydro_ids: list[int] = []
    values: list[float] = []
    for code in id_map.hydro_codes:
        hreg = hidr.loc[code]
        v_min = float(hreg["volume_minimo"])
        v_max = float(hreg["volume_maximo"])
        rho_esp = float(hreg.get("produtibilidade_especifica", 0.0) or 0.0)
        cf = float(hreg.get("canal_fuga_medio", 0.0) or 0.0)
        tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
        perdas = float(hreg.get("perdas", 0.0) or 0.0)
        h_gross = _mean_cota_over_volume(hreg, v_min, v_max) - cf
        h_net = max(_apply_hydraulic_loss(h_gross, tipo_perda, perdas), 0.0)
        hydro_ids.append(id_map.hydro_id(code))
        values.append(rho_esp * h_net)

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array([None] * len(hydro_ids), type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(values, type=pa.float64()),
            # Present-but-null: the reader requires the full column set.
            "reference_outflow_m3s": pa.array(
                [None] * len(hydro_ids), type=pa.float64()
            ),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                [None] * len(hydro_ids), type=pa.float64()
            ),
        }
    )


def convert_production_models(id_map: DecompIdMap) -> dict:
    """Constant-productivity production models for every operated plant."""
    return {
        "$schema": _PRODUCTION_MODELS_SCHEMA_URL,
        "production_models": [
            {
                "hydro_id": id_map.hydro_id(code),
                "selection_mode": "stage_ranges",
                "stage_ranges": [
                    {
                        "start_stage_id": 0,
                        "end_stage_id": None,
                        "model": "constant_productivity",
                    }
                ],
            }
            for code in id_map.hydro_codes
        ],
    }
