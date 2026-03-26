"""Entity alignment between NEWAVE codes and Cobre IDs.

Builds aligned entity pairs for hydros, thermals, and exchange lines
using the same NewaveIdMap that the converter produces, plus the
sintetizador reference Parquets for human-readable names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class HydroEntity:
    """Aligned hydro plant pair."""

    newave_code: int
    cobre_id: int
    name: str
    has_reservoir: bool


@dataclass(frozen=True)
class ThermalEntity:
    """Aligned thermal plant pair."""

    newave_code: int
    cobre_id: int
    name: str


@dataclass(frozen=True)
class LineEntity:
    """Aligned exchange line pair.

    NEWAVE models exchange as bidirectional flow between subsystem pairs.
    Cobre models each normalized pair as a single line where positive flow
    goes from source_bus to target_bus.

    The NEWAVE ``INT`` variable for pair (de, para) reports:
    - ``limite_superior``: max flow in the de->para direction
    - ``limite_inferior``: max flow in the para->de direction (negative)

    When the Cobre line's (source_bus, target_bus) matches (de, para),
    ``flow_max`` corresponds to ``limite_superior`` and ``flow_min`` to
    ``abs(limite_inferior)``.  When reversed, the mapping flips.
    """

    cobre_line_id: int
    name: str
    source_bus_id: int
    target_bus_id: int
    newave_de: int
    newave_para: int
    reversed: bool  # True if Cobre (src,tgt) = NEWAVE (para,de)


@dataclass
class EntityAlignment:
    """Complete entity alignment between NEWAVE and Cobre."""

    hydros: list[HydroEntity] = field(default_factory=list)
    thermals: list[ThermalEntity] = field(default_factory=list)
    lines: list[LineEntity] = field(default_factory=list)
    skipped_hydros: list[int] = field(default_factory=list)
    skipped_thermals: list[int] = field(default_factory=list)
    num_newave_stages: int = 0


def _read_reference_names(
    sintese_dir: Path,
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Read entity names from sintetizador reference Parquets.

    Returns (hydro_names, thermal_names, subsystem_names) dicts
    mapping NEWAVE codes to human-readable names.
    """
    hydro_names: dict[int, str] = {}
    thermal_names: dict[int, str] = {}
    subsystem_names: dict[int, str] = {}

    uhe_path = sintese_dir / "UHE.parquet"
    if uhe_path.exists():
        uhe = pl.read_parquet(uhe_path)
        for row in uhe.iter_rows(named=True):
            hydro_names[int(row["codigo_usina"])] = str(row["usina"]).strip()

    ute_path = sintese_dir / "UTE.parquet"
    if ute_path.exists():
        ute = pl.read_parquet(ute_path)
        for row in ute.iter_rows(named=True):
            thermal_names[int(row["codigo_usina"])] = str(row["usina"]).strip()

    sbm_path = sintese_dir / "SBM.parquet"
    if sbm_path.exists():
        sbm = pl.read_parquet(sbm_path)
        for row in sbm.iter_rows(named=True):
            subsystem_names[int(row["codigo_submercado"])] = str(
                row["submercado"]
            ).strip()

    return hydro_names, thermal_names, subsystem_names


def _detect_reservoir_plants(sintese_dir: Path) -> set[int]:
    """Return the set of NEWAVE hydro codes that have VARMF data (reservoirs)."""
    uhe_path = sintese_dir / "ESTATISTICAS_OPERACAO_UHE.parquet"
    if not uhe_path.exists():
        return set()

    uhe = pl.scan_parquet(uhe_path)
    varmf = (
        uhe.filter(pl.col("variavel") == "VARMF")
        .select("codigo_usina")
        .unique()
        .collect()
    )
    return set(varmf["codigo_usina"].to_list())


def _detect_newave_stages(sintese_dir: Path) -> int:
    """Return the number of stages reported by the sintetizador."""
    uhe_path = sintese_dir / "ESTATISTICAS_OPERACAO_UHE.parquet"
    if not uhe_path.exists():
        return 0

    uhe = pl.scan_parquet(uhe_path)
    max_stage = uhe.select(pl.col("estagio").max()).collect().item()
    return int(max_stage) if max_stage is not None else 0


def build_entity_alignment(
    id_map: NewaveIdMap,
    sintese_dir: Path,
    lines_json: list[dict],
) -> EntityAlignment:
    """Build entity alignment from the ID map and sintetizador data.

    Parameters
    ----------
    id_map:
        The same NewaveIdMap used by the converter.
    sintese_dir:
        Path to the sintetizador output directory containing reference
        Parquets (UHE.parquet, UTE.parquet, SBM.parquet).
    lines_json:
        The ``lines`` list from the converted Cobre ``lines.json``.
    """
    hydro_names, thermal_names, subsystem_names = _read_reference_names(sintese_dir)
    reservoir_codes = _detect_reservoir_plants(sintese_dir)
    num_stages = _detect_newave_stages(sintese_dir)

    alignment = EntityAlignment(num_newave_stages=num_stages)

    # --- Hydros ---
    for nw_code in id_map.all_hydro_codes:
        try:
            cobre_id = id_map.hydro_id(nw_code)
        except KeyError:
            alignment.skipped_hydros.append(nw_code)
            continue

        name = hydro_names.get(nw_code, f"code_{nw_code}")
        has_reservoir = nw_code in reservoir_codes
        alignment.hydros.append(
            HydroEntity(
                newave_code=nw_code,
                cobre_id=cobre_id,
                name=name,
                has_reservoir=has_reservoir,
            )
        )

    # --- Thermals ---
    for nw_code in id_map.all_thermal_codes:
        try:
            cobre_id = id_map.thermal_id(nw_code)
        except KeyError:
            alignment.skipped_thermals.append(nw_code)
            continue

        name = thermal_names.get(nw_code, f"code_{nw_code}")
        alignment.thermals.append(
            ThermalEntity(
                newave_code=nw_code,
                cobre_id=cobre_id,
                name=name,
            )
        )

    # --- Lines (exchange) ---
    # Build a reverse lookup: (source_bus_id, target_bus_id) -> line entry
    bus_pair_to_line: dict[tuple[int, int], dict] = {}
    for line in lines_json:
        key = (int(line["source_bus_id"]), int(line["target_bus_id"]))
        bus_pair_to_line[key] = line

    # Build reverse bus lookup: cobre_bus_id -> newave_subsystem_code
    bus_id_to_nw: dict[int, int] = {}
    for nw_code in id_map.all_bus_ids:
        bus_id_to_nw[id_map.bus_id(nw_code)] = nw_code

    # For each Cobre line, find the matching NEWAVE subsystem pair
    for line in lines_json:
        line_id = int(line["id"])
        src_bus = int(line["source_bus_id"])
        tgt_bus = int(line["target_bus_id"])
        line_name = str(line.get("name", f"line_{line_id}"))

        nw_de = bus_id_to_nw.get(src_bus)
        nw_para = bus_id_to_nw.get(tgt_bus)
        if nw_de is None or nw_para is None:
            _LOG.warning(
                "Line %d (%s) has unmapped buses; skipping.", line_id, line_name
            )
            continue

        alignment.lines.append(
            LineEntity(
                cobre_line_id=line_id,
                name=line_name,
                source_bus_id=src_bus,
                target_bus_id=tgt_bus,
                newave_de=nw_de,
                newave_para=nw_para,
                reversed=False,
            )
        )

    _LOG.debug(
        "Alignment: %d hydros (%d with reservoir), %d thermals, %d lines, %d stages",
        len(alignment.hydros),
        sum(1 for h in alignment.hydros if h.has_reservoir),
        len(alignment.thermals),
        len(alignment.lines),
        alignment.num_newave_stages,
    )

    return alignment
