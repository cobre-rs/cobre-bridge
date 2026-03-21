"""Conversion pipeline: orchestrates entity and temporal/stochastic converters.

Reads a NEWAVE case directory and writes a complete Cobre case directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from cobre_bridge.converters import constraints as constraints_conv
from cobre_bridge.converters import hydro as hydro_conv
from cobre_bridge.converters import initial_conditions as ic_conv
from cobre_bridge.converters import network as network_conv
from cobre_bridge.converters import stochastic as stochastic_conv
from cobre_bridge.converters import temporal as temporal_conv
from cobre_bridge.converters import thermal as thermal_conv
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

logger = logging.getLogger(__name__)


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


def _build_id_map(nw_files: NewaveFiles) -> NewaveIdMap:
    """Read Confhd, Conft, Sistema, and Ree to build the NewaveIdMap."""
    from inewave.newave import Confhd, Conft, Ree, Sistema

    confhd = Confhd.read(str(nw_files.confhd))
    conft = Conft.read(str(nw_files.conft))
    sistema = Sistema.read(str(nw_files.sistema))
    ree_file = Ree.read(str(nw_files.ree))

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
    # 1. Discover and validate all source files via caso.dat -> Arquivos.
    # ------------------------------------------------------------------
    logger.debug("Discovering NEWAVE files from %s", src)
    nw_files = NewaveFiles.from_directory(src)

    # ------------------------------------------------------------------
    # 2. Build the entity ID map.
    # ------------------------------------------------------------------
    logger.debug("Building NewaveIdMap from %s", src)
    id_map = _build_id_map(nw_files)

    # ------------------------------------------------------------------
    # 3. Call all converters.
    # ------------------------------------------------------------------
    logger.debug("Converting hydros")
    hydros_dict = hydro_conv.convert_hydros(nw_files, id_map)

    logger.debug("Generating hydro geometry")
    cadastro = hydro_conv.read_cadastro(nw_files)
    geometry_table = hydro_conv.generate_hydro_geometry(cadastro, id_map)

    logger.debug("Converting thermals")
    thermals_dict = thermal_conv.convert_thermals(nw_files, id_map)

    logger.debug("Converting buses")
    buses_dict = network_conv.convert_buses(nw_files, id_map)

    logger.debug("Converting lines")
    lines_dict = network_conv.convert_lines(nw_files, id_map)

    logger.debug("Converting penalties")
    penalties_dict = network_conv.convert_penalties(nw_files)

    logger.debug("Converting stages")
    stages_dict = temporal_conv.convert_stages(nw_files, id_map)

    logger.debug("Converting config")
    config_dict = temporal_conv.convert_config(nw_files)

    logger.debug("Converting initial conditions")
    ic_dict = ic_conv.convert_initial_conditions(nw_files, id_map)

    logger.debug("Extracting recent inflow lags from vazpast.dat")
    past_inflow_lags = stochastic_conv.convert_recent_inflow_lags(nw_files, id_map)
    if past_inflow_lags:
        ic_dict["past_inflows"] = past_inflow_lags

    logger.debug("Converting inflow stats")
    inflow_table = stochastic_conv.convert_inflow_stats(nw_files, id_map)

    logger.debug("Converting load stats")
    load_table = stochastic_conv.convert_load_stats(nw_files, id_map)

    logger.debug("Converting inflow history from vazoes.dat")
    inflow_history_table = stochastic_conv.convert_inflow_history(nw_files, id_map)

    logger.debug("Converting water withdrawal")
    withdrawal_table = hydro_conv.convert_water_withdrawal(nw_files, id_map)

    logger.debug("Converting VminOP constraints")
    vminop_result = constraints_conv.convert_vminop_constraints(nw_files, id_map)

    logger.debug("Converting electric constraints")
    vminop_count = (
        len(vminop_result[0].get("constraints", [])) if vminop_result is not None else 0
    )
    electric_result = constraints_conv.convert_electric_constraints(
        nw_files, id_map, start_id=vminop_count
    )

    logger.debug("Converting load factors")
    load_factors_dict = stochastic_conv.convert_load_factors(nw_files, id_map)

    logger.debug("Converting line bounds")
    line_bounds_table = network_conv.convert_line_bounds(nw_files, id_map)

    logger.debug("Converting non-controllable sources")
    ncs_dict = network_conv.convert_non_controllable_sources(nw_files, id_map)

    logger.debug("Converting exchange factors")
    exchange_factors_dict = network_conv.convert_exchange_factors(nw_files, id_map)

    logger.debug("Converting NCS block factors")
    ncs_factors_dict = network_conv.convert_ncs_factors(nw_files, id_map)

    logger.debug("Converting NCS stats")
    ncs_stats_table = network_conv.convert_ncs_stats(nw_files, id_map)

    logger.debug("Converting production models")
    production_models_dict = hydro_conv.convert_production_models(nw_files, id_map)

    # ------------------------------------------------------------------
    # 4. Create the output directory structure.
    # ------------------------------------------------------------------
    (dst / "system").mkdir(parents=True, exist_ok=True)
    (dst / "scenarios").mkdir(parents=True, exist_ok=True)
    (dst / "constraints").mkdir(parents=True, exist_ok=True)

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
    _write_json(dst / "system" / "non_controllable_sources.json", ncs_dict)
    _write_json(dst / "scenarios" / "load_factors.json", load_factors_dict)
    _write_json(dst / "constraints" / "exchange_factors.json", exchange_factors_dict)
    _write_json(dst / "scenarios" / "non_controllable_factors.json", ncs_factors_dict)

    if production_models_dict is not None:
        _write_json(
            dst / "system" / "hydro_production_models.json", production_models_dict
        )

    # ------------------------------------------------------------------
    # 6. Write Parquet files.
    # ------------------------------------------------------------------
    geometry_path = dst / "system" / "hydro_geometry.parquet"
    pq.write_table(geometry_table, geometry_path, compression="zstd")
    logger.debug("Wrote %s", geometry_path)

    inflow_path = dst / "scenarios" / "inflow_seasonal_stats.parquet"
    pq.write_table(inflow_table, inflow_path, compression="zstd")
    logger.debug("Wrote %s", inflow_path)

    load_path = dst / "scenarios" / "load_seasonal_stats.parquet"
    pq.write_table(load_table, load_path, compression="zstd")
    logger.debug("Wrote %s", load_path)

    history_path = dst / "scenarios" / "inflow_history.parquet"
    pq.write_table(inflow_history_table, history_path, compression="zstd")
    logger.debug("Wrote %s", history_path)

    constraints_dir = dst / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    line_bounds_path = constraints_dir / "line_bounds.parquet"
    pq.write_table(line_bounds_table, line_bounds_path, compression="zstd")
    logger.debug("Wrote %s", line_bounds_path)

    ncs_stats_path = dst / "scenarios" / "non_controllable_stats.parquet"
    pq.write_table(ncs_stats_table, ncs_stats_path, compression="zstd")
    logger.debug("Wrote %s", ncs_stats_path)

    if withdrawal_table is not None:
        hydro_bounds_path = constraints_dir / "hydro_bounds.parquet"
        pq.write_table(withdrawal_table, hydro_bounds_path, compression="zstd")
        logger.debug("Wrote %s", hydro_bounds_path)

    # Merge VminOP and electric constraints into a single output.
    import pyarrow as pa

    all_constraints: list[dict] = []
    bounds_tables: list[pa.Table] = []

    # Canonical column order for the merged bounds table.
    _BOUNDS_COLUMNS = ["constraint_id", "stage_id", "block_id", "bound"]

    if vminop_result is not None:
        vminop_dict, vminop_bounds = vminop_result
        all_constraints.extend(vminop_dict.get("constraints", []))
        # VminOP bounds table has no block_id column; add a null column and
        # reorder to match the canonical schema.
        n = len(vminop_bounds)
        vminop_bounds_extended = vminop_bounds.append_column(
            pa.field("block_id", pa.int32()),
            pa.array([None] * n, type=pa.int32()),
        ).select(_BOUNDS_COLUMNS)
        bounds_tables.append(vminop_bounds_extended)

    if electric_result is not None:
        elec_constraints, elec_bounds = electric_result
        all_constraints.extend(elec_constraints)
        bounds_tables.append(elec_bounds)

    if all_constraints:
        merged_dict = {
            "$schema": (
                "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
                "/book/src/schemas/generic_constraints.schema.json"
            ),
            "constraints": all_constraints,
        }
        _write_json(constraints_dir / "generic_constraints.json", merged_dict)

        if bounds_tables:
            merged_bounds = pa.concat_tables(bounds_tables)
            gc_bounds_path = constraints_dir / "generic_constraint_bounds.parquet"
            pq.write_table(merged_bounds, gc_bounds_path, compression="zstd")
            logger.debug("Wrote %s", gc_bounds_path)

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
    for subdir in ("system", "scenarios", "constraints"):
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
