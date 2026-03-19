"""Conversion pipeline: orchestrates entity and temporal/stochastic converters.

Reads a NEWAVE case directory and writes a complete Cobre case directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cobre_bridge.converters import hydro as hydro_conv
from cobre_bridge.converters import initial_conditions as ic_conv
from cobre_bridge.converters import network as network_conv
from cobre_bridge.converters import stochastic as stochastic_conv
from cobre_bridge.converters import temporal as temporal_conv
from cobre_bridge.converters import thermal as thermal_conv
from cobre_bridge.id_map import NewaveIdMap

logger = logging.getLogger(__name__)

REQUIRED_FILES = [
    "dger.dat",
    "hidr.dat",
    "confhd.dat",
    "sistema.dat",
    "conft.dat",
    "clast.dat",
    "term.dat",
    "ree.dat",
    "patamar.dat",
    "vazoes.dat",
]


@dataclass
class ConversionReport:
    """Summary of a completed NEWAVE-to-Cobre conversion."""

    hydro_count: int = 0
    thermal_count: int = 0
    bus_count: int = 0
    line_count: int = 0
    stage_count: int = 0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Converted: {self.hydro_count} hydros, "
            f"{self.thermal_count} thermals, "
            f"{self.bus_count} buses, "
            f"{self.line_count} lines, "
            f"{self.stage_count} stages"
        )


def _build_id_map(src: Path) -> NewaveIdMap:
    """Read Confhd, Conft, Sistema, and Ree to build the NewaveIdMap."""
    from inewave.newave import Confhd, Conft, Ree, Sistema

    confhd = Confhd.read(str(src / "confhd.dat"))
    conft = Conft.read(str(src / "conft.dat"))
    sistema = Sistema.read(str(src / "sistema.dat"))
    ree_file = Ree.read(str(src / "ree.dat"))

    # Hydro codes from confhd — existing, non-fictitious plants only.
    confhd_df = confhd.usinas
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]
    fict_names = existing.loc[
        existing["nome_usina"].str.strip().str.startswith("FICT."), "nome_usina"
    ].tolist()
    if fict_names:
        logger.warning(
            "Excluding %d fictitious plant(s) from id_map: %s",
            len(fict_names),
            fict_names,
        )
    hydro_codes = [int(r["codigo_usina"]) for _, r in non_fict.iterrows()]

    # Thermal codes from conft.
    conft_df = conft.usinas
    thermal_codes = [int(r["codigo_usina"]) for _, r in conft_df.iterrows()]

    # Subsystem codes from sistema deficit table.
    deficit_df = sistema.custo_deficit
    if deficit_df is not None:
        subsystem_ids = sorted(
            set(int(r["codigo_submercado"]) for _, r in deficit_df.iterrows())
        )
    else:
        subsystem_ids = []

    # Also include subsystem codes referenced in ree.dat (for completeness).
    ree_df = ree_file.rees
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            code = int(row["submercado"])
            if code not in subsystem_ids:
                subsystem_ids.append(code)

    return NewaveIdMap(
        subsystem_ids=subsystem_ids,
        hydro_codes=hydro_codes,
        thermal_codes=thermal_codes,
    )


def _convert_past_inflows_if_present(
    src: Path,
    id_map: NewaveIdMap,
    stages_dict: dict,  # noqa: ARG001 — accepted for API consistency
) -> pa.Table | None:
    """Call convert_past_inflows when Dger.num_anos_pre_estudo > 0.

    Extracts the study start date and pre-study month count from ``dger.dat``
    and delegates to ``stochastic_conv.convert_past_inflows``.  Returns
    ``None`` when the pre-study period is zero or ``vazpast.dat`` is absent.
    """
    from datetime import date as _date

    from inewave.newave import Dger as _Dger

    dger_path = src / "dger.dat"
    if not dger_path.exists():
        return None

    dger = _Dger.read(dger_path)
    num_anos_pre = dger.num_anos_pre_estudo or 0
    if num_anos_pre == 0:
        return None

    num_pre_months = num_anos_pre * 12
    study_start = _date(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 1)

    return stochastic_conv.convert_past_inflows(
        src, id_map, study_start, num_pre_months
    )


def convert_newave_case(src: Path, dst: Path) -> ConversionReport:
    """Convert a NEWAVE case directory to a Cobre case directory.

    Parameters
    ----------
    src:
        Path to the NEWAVE case directory.  Must exist and contain all
        required NEWAVE input files.
    dst:
        Path to the output Cobre case directory.  Must not exist or must be
        empty (call site is responsible for enforcing the --force contract
        before calling this function).

    Returns
    -------
    ConversionReport
        Summary of what was converted.

    Raises
    ------
    FileNotFoundError
        If *src* does not exist, is not a directory, or a required NEWAVE
        file is missing.
    """
    report = ConversionReport()

    # ------------------------------------------------------------------
    # 1. Validate source files.
    # ------------------------------------------------------------------
    for filename in REQUIRED_FILES:
        path = src / filename
        if not path.exists():
            raise FileNotFoundError(filename)

    # ------------------------------------------------------------------
    # 2. Build the entity ID map.
    # ------------------------------------------------------------------
    logger.debug("Building NewaveIdMap from %s", src)
    id_map = _build_id_map(src)

    # ------------------------------------------------------------------
    # 3. Call all converters.
    # ------------------------------------------------------------------
    logger.debug("Converting hydros")
    hydros_dict = hydro_conv.convert_hydros(src, id_map)

    logger.debug("Generating hydro geometry")
    cadastro = hydro_conv.read_cadastro(src)
    geometry_table = hydro_conv.generate_hydro_geometry(cadastro, id_map)

    logger.debug("Converting thermals")
    thermals_dict = thermal_conv.convert_thermals(src, id_map)

    logger.debug("Converting buses")
    buses_dict = network_conv.convert_buses(src, id_map)

    logger.debug("Converting lines")
    lines_dict = network_conv.convert_lines(src, id_map)

    logger.debug("Converting penalties")
    penalties_dict = network_conv.convert_penalties(src)

    logger.debug("Converting stages")
    stages_dict = temporal_conv.convert_stages(src, id_map)

    logger.debug("Converting config")
    config_dict = temporal_conv.convert_config(src)

    logger.debug("Converting initial conditions")
    ic_dict = ic_conv.convert_initial_conditions(src, id_map)

    logger.debug("Converting inflow stats")
    inflow_table = stochastic_conv.convert_inflow_stats(src, id_map)

    logger.debug("Converting load stats")
    load_table = stochastic_conv.convert_load_stats(src, id_map)

    logger.debug("Converting past inflows")
    past_inflow_table = _convert_past_inflows_if_present(src, id_map, stages_dict)

    # ------------------------------------------------------------------
    # 4. Create the output directory structure.
    # ------------------------------------------------------------------
    (dst / "system").mkdir(parents=True, exist_ok=True)
    (dst / "scenarios").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 5. Write JSON files.
    # ------------------------------------------------------------------
    def _write_json(path: Path, data: dict) -> None:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Wrote %s", path)

    _write_json(dst / "config.json", config_dict)
    _write_json(dst / "stages.json", stages_dict)
    _write_json(dst / "penalties.json", penalties_dict)
    _write_json(dst / "initial_conditions.json", ic_dict)
    _write_json(dst / "system" / "hydros.json", hydros_dict)
    _write_json(dst / "system" / "thermals.json", thermals_dict)
    _write_json(dst / "system" / "buses.json", buses_dict)
    _write_json(dst / "system" / "lines.json", lines_dict)

    # ------------------------------------------------------------------
    # 6. Write Parquet files.
    # ------------------------------------------------------------------
    geometry_path = dst / "system" / "hydro_geometry.parquet"
    pq.write_table(geometry_table, geometry_path)
    logger.debug("Wrote %s", geometry_path)

    inflow_path = dst / "scenarios" / "inflow_seasonal_stats.parquet"
    pq.write_table(inflow_table, inflow_path)
    logger.debug("Wrote %s", inflow_path)

    load_path = dst / "scenarios" / "load_seasonal_stats.parquet"
    pq.write_table(load_table, load_path)
    logger.debug("Wrote %s", load_path)

    if past_inflow_table is not None:
        history_path = dst / "scenarios" / "inflow_history.parquet"
        pq.write_table(past_inflow_table, history_path)
        logger.debug("Wrote %s", history_path)

    # ------------------------------------------------------------------
    # 7. Populate the report.
    # ------------------------------------------------------------------
    report.hydro_count = len(hydros_dict.get("hydros", []))
    report.thermal_count = len(thermals_dict.get("thermals", []))
    report.bus_count = len(buses_dict.get("buses", []))
    report.line_count = len(lines_dict.get("lines", []))
    report.stage_count = len(stages_dict.get("stages", []))

    return report


def _clear_dst_contents(dst: Path) -> None:
    """Remove the known output subdirectories and top-level JSON files from dst.

    Only the specific files/subdirectories produced by the pipeline are
    removed.  This avoids accidentally deleting unrelated files in the
    destination directory.
    """
    for subdir in ("system", "scenarios"):
        target = dst / subdir
        if target.exists():
            shutil.rmtree(target)

    for filename in (
        "config.json",
        "stages.json",
        "penalties.json",
        "initial_conditions.json",
    ):
        path = dst / filename
        if path.exists():
            path.unlink()
