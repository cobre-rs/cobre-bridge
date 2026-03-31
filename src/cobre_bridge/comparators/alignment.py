"""Entity alignment between NEWAVE codes and Cobre IDs.

Builds aligned entity pairs for hydros, thermals, and exchange lines
using the same NewaveIdMap that the converter produces, plus NEWAVE
input files (via ``NewaveFiles`` + inewave readers) for human-readable
names and reservoir detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

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
    nw_files: NewaveFiles,
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Read entity names from NEWAVE input files via inewave.

    Returns (hydro_names, thermal_names, subsystem_names) dicts
    mapping NEWAVE codes to human-readable names.
    """
    from inewave.newave import Confhd, Conft, Sistema

    hydro_names: dict[int, str] = {}
    thermal_names: dict[int, str] = {}
    subsystem_names: dict[int, str] = {}

    # Hydro names from confhd.dat
    confhd = Confhd.read(str(nw_files.confhd))
    for _, row in confhd.usinas.iterrows():
        hydro_names[int(row["codigo_usina"])] = str(row["nome_usina"]).strip()

    # Thermal names from conft.dat
    conft = Conft.read(str(nw_files.conft))
    for _, row in conft.usinas.iterrows():
        thermal_names[int(row["codigo_usina"])] = str(row["nome_usina"]).strip()

    # Subsystem names from sistema.dat (deduplicate from custo_deficit rows)
    sistema = Sistema.read(str(nw_files.sistema))
    deficit_df = sistema.custo_deficit
    if deficit_df is not None:
        seen: set[int] = set()
        for _, row in deficit_df.iterrows():
            code = int(row["codigo_submercado"])
            if code not in seen:
                subsystem_names[code] = str(row["nome_submercado"]).strip()
                seen.add(code)

    return hydro_names, thermal_names, subsystem_names


def _detect_reservoir_plants(nw_files: NewaveFiles) -> set[int]:
    """Return the set of NEWAVE hydro codes that have reservoirs.

    A plant has a reservoir when its ``volume_minimo != volume_maximo``
    in the HIDR cadastro (with permanent MODIF overrides applied).
    """
    from cobre_bridge.converters.hydro import read_cadastro

    cadastro = read_cadastro(nw_files)
    reservoir_codes: set[int] = set()
    for code, row in cadastro.iterrows():
        vol_min = float(row["volume_minimo"])
        vol_max = float(row["volume_maximo"])
        if vol_min != vol_max:
            reservoir_codes.add(int(code))  # type: ignore[arg-type]
    return reservoir_codes


def _detect_newave_stages(nw_files: NewaveFiles) -> int:
    """Compute total number of NEWAVE stages from DGER parameters.

    ``study_months = (13 - start_month) + (num_anos - 1) * 12``
    ``total_stages = study_months + num_anos_pos * 12``
    """
    from inewave.newave import Dger

    dger = Dger.read(str(nw_files.dger))
    start_month: int = dger.mes_inicio_estudo
    num_anos: int = dger.num_anos_estudo
    num_anos_pos: int = dger.num_anos_pos_estudo or 0

    if not num_anos:
        return 0

    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12
    return total_stages


def build_entity_alignment(
    id_map: NewaveIdMap,
    nw_files: NewaveFiles,
    lines_json: list[dict],
) -> EntityAlignment:
    """Build entity alignment from the ID map and NEWAVE input files.

    Parameters
    ----------
    id_map:
        The same NewaveIdMap used by the converter.
    nw_files:
        Resolved NEWAVE input file paths (replaces the former
        ``sintese_dir`` parameter).
    lines_json:
        The ``lines`` list from the converted Cobre ``lines.json``.
    """
    hydro_names, thermal_names, subsystem_names = _read_reference_names(nw_files)
    reservoir_codes = _detect_reservoir_plants(nw_files)
    num_stages = _detect_newave_stages(nw_files)

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
