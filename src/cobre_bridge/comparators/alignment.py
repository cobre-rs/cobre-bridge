"""Entity alignment between the source model codes and Cobre IDs.

Builds aligned entity pairs for hydros, thermals, and exchange lines
using the same NewaveIdMap that the converter produces, plus the parsed
``NewaveCase`` for human-readable names and reservoir detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from cobre_bridge.case import NewaveCase
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

    The source model models exchange as bidirectional flow between subsystem pairs.
    Cobre models each normalized pair as a single line where positive flow goes from
    source_bus to target_bus.

    ``newave_de`` / ``newave_para`` are the source model subsystem codes of the Cobre
    line's ``source_bus_id`` / ``target_bus_id``, so the Cobre orientation ``(source,
    target)`` always corresponds to the source model ``(de, para)`` by construction.
    The source model result files (NWLISTOP) may list a pair in either ``(de, para)`` or
    ``(para, de)`` order; that file-ordering is handled where the rows are read
    (sign-flipped on the reverse-ordered match), not here.
    """

    cobre_line_id: int
    name: str
    source_bus_id: int
    target_bus_id: int
    newave_de: int
    newave_para: int


@dataclass
class EntityAlignment:
    """Complete entity alignment between the source model and Cobre."""

    hydros: list[HydroEntity] = field(default_factory=list)
    thermals: list[ThermalEntity] = field(default_factory=list)
    lines: list[LineEntity] = field(default_factory=list)
    skipped_hydros: list[int] = field(default_factory=list)
    skipped_thermals: list[int] = field(default_factory=list)
    num_newave_stages: int = 0


def read_reference_names(
    case: NewaveCase,
) -> tuple[dict[int, str], dict[int, str], dict[int, str]]:
    """Read entity names from the source model input files via inewave.

    Returns (hydro_names, thermal_names, subsystem_names) dicts mapping the source model
    codes to human-readable names.
    """
    hydro_names: dict[int, str] = {}
    thermal_names: dict[int, str] = {}
    subsystem_names: dict[int, str] = {}

    # Hydro names from confhd.dat
    for _, row in case.confhd.usinas.iterrows():
        hydro_names[int(row["codigo_usina"])] = str(row["nome_usina"]).strip()

    # Thermal names from conft.dat
    for _, row in case.conft.usinas.iterrows():
        thermal_names[int(row["codigo_usina"])] = str(row["nome_usina"]).strip()

    # Subsystem names from sistema.dat (deduplicate from custo_deficit rows)
    deficit_df = case.sistema.custo_deficit
    if deficit_df is not None:
        seen: set[int] = set()
        for _, row in deficit_df.iterrows():
            code = int(row["codigo_submercado"])
            if code not in seen:
                subsystem_names[code] = str(row["nome_submercado"]).strip()
                seen.add(code)

    return hydro_names, thermal_names, subsystem_names


def _detect_reservoir_plants(case: NewaveCase) -> set[int]:
    """Return the set of the source model hydro codes that have reservoirs.

    A plant has a reservoir when its ``volume_minimo != volume_maximo``
    in the HIDR cadastro (with permanent MODIF overrides applied).
    """
    from cobre_bridge.converters.hydro import read_cadastro

    cadastro = read_cadastro(case)
    reservoir_codes: set[int] = set()
    for code, row in cadastro.iterrows():
        vol_min = float(row["volume_minimo"])
        vol_max = float(row["volume_maximo"])
        if vol_min != vol_max:
            reservoir_codes.add(int(code))  # type: ignore[arg-type]
    return reservoir_codes


def _detect_newave_stages(case: NewaveCase) -> int:
    """Compute total number of the source model stages from DGER parameters.

    Uses the case's cached :attr:`NewaveCase.horizon`; an empty study
    (``num_anos_estudo`` of 0/None) reports zero stages.
    """
    if not case.dger.num_anos_estudo:
        return 0

    return case.horizon.total_stages


def build_entity_alignment(
    id_map: NewaveIdMap,
    case: NewaveCase,
    lines_json: list[dict],
) -> EntityAlignment:
    """Build entity alignment from the ID map and the parsed the source model case.

    Parameters
    ----------
    id_map:
        The same NewaveIdMap used by the converter.
    case:
        Parsed the source model case.
    lines_json:
        The ``lines`` list from the converted Cobre ``lines.json``.
    """
    hydro_names, thermal_names, subsystem_names = read_reference_names(case)
    reservoir_codes = _detect_reservoir_plants(case)
    num_stages = _detect_newave_stages(case)

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

    # For each Cobre line, find the matching the source model subsystem pair
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
