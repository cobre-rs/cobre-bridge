"""Conversion orchestrator for DECOMP-like decks.

``convert_decomp_case(src, dst)`` discovers the deck (``caso.dat`` names
the revision extension; the ``rvN`` index file names the data files),
parses it, and writes a Cobre case directory. Scope is the ratified
loop-closing milestone: the deferred families (exchange network pending
the upstream accessor fix, renewables pending their reader, GNL
anticipation, boundary FCF, and the per-block/fidelity items) are logged
loudly by their emitters or here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from idecomp.decomp import Dadger, Vazoes

from cobre_bridge.decomp import bounds as bounds_conv
from cobre_bridge.decomp import config as config_conv
from cobre_bridge.decomp import hydro as hydro_conv
from cobre_bridge.decomp import load as load_conv
from cobre_bridge.decomp import ncs as ncs_conv
from cobre_bridge.decomp import network as network_conv
from cobre_bridge.decomp import scenarios as scenarios_conv
from cobre_bridge.decomp import temporal as temporal_conv
from cobre_bridge.decomp import thermal as thermal_conv
from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecompFiles:
    """Resolved input files of one deck revision."""

    revision: str
    dadger: Path
    vazoes: Path
    hidr: Path
    dadgnl: Path | None
    renovaveis: Path | None


def discover_decomp_files(src: Path) -> DecompFiles:
    """Resolve the deck files via ``caso.dat`` → the revision index file."""
    caso = src / "caso.dat"
    if not caso.is_file():
        raise FileNotFoundError(f"{caso} not found; not a deck directory")
    revision = caso.read_text(encoding="latin-1").split()[0].strip()

    names: list[str] = []
    index = src / revision
    if index.is_file():
        names = [
            line.strip()
            for line in index.read_text(encoding="latin-1").splitlines()
            if line.strip() and not line.strip().startswith("&")
        ]

    def find(prefix: str, required: bool) -> Path | None:
        for name in names:
            if name.lower().startswith(prefix):
                path = src / name
                if path.is_file():
                    return path
        matches = sorted(src.glob(f"{prefix}*"))
        if matches:
            return matches[0]
        if required:
            raise FileNotFoundError(f"no {prefix}* file found in {src}")
        return None

    dadger = find("dadger", required=True)
    vazoes = find("vazoes", required=True)
    hidr = find("hidr", required=True)
    dadgnl = find("dadgnl", required=False)
    renovaveis = find("renovaveis", required=False)
    assert dadger is not None and vazoes is not None and hidr is not None
    return DecompFiles(
        revision=revision,
        dadger=dadger,
        vazoes=vazoes,
        hidr=hidr,
        dadgnl=dadgnl,
        renovaveis=renovaveis,
    )


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _write_parquet(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # TRACKED COBRE-GAP WORKAROUND (C3): the solver's parquet reader is
    # built without snappy; zstd until the compression contract is settled.
    pq.write_table(table, path, compression="zstd")


def convert_decomp_case(src: Path, dst: Path, *, force: bool = False) -> None:
    """Convert one deck revision into a Cobre case directory."""
    if dst.exists() and any(dst.iterdir()) and not force:
        raise FileExistsError(f"{dst} already contains files; pass force to overwrite")

    files = discover_decomp_files(src)
    _LOG.info("converting %s (revision %s)", src, files.revision)

    dadger = Dadger.read(str(files.dadger))
    vazoes = Vazoes.read(str(files.vazoes))
    hidr = hydro_conv.read_hidr(files.hidr)
    renovaveis = None
    if files.renovaveis is not None:
        from idecomp.libs import Renovaveis

        renovaveis = Renovaveis.read(str(files.renovaveis))

    id_map = DecompIdMap.from_dadger(dadger)
    calendar = temporal_conv.operative_calendar_from_dadger(dadger)
    start_date = calendar[0].start_date
    probabilities = vazoes.probabilidades
    if probabilities is None or probabilities.empty:
        raise ValueError("the inflow file has no probability table")
    terminal_fan = int(probabilities.groupby("estagio")["cenario"].nunique().max())
    num_scenarios = [1] * (len(calendar) - 1) + [terminal_fan]

    tx = float(dadger.tx.taxa) / 100.0

    dst.mkdir(parents=True, exist_ok=True)

    _write_json(dst / "config.json", config_conv.convert_config(dadger, terminal_fan))
    _write_json(
        dst / "stages.json",
        temporal_conv.convert_stages(
            calendar, annual_discount_rate=tx, num_scenarios=num_scenarios
        ),
    )

    productivity = hydro_conv.convert_energy_productivity(hidr, id_map)
    deficit_costs = network_conv._bus_deficit_costs(dadger)
    deficit_cost = max(deficit_costs.values()) if deficit_costs else 0.0
    _write_json(
        dst / "penalties.json",
        config_conv.convert_penalties(
            deficit_cost,
            productivity["equivalent_productivity_mw_per_m3s"].to_pylist(),
        ),
    )
    _write_json(
        dst / "initial_conditions.json",
        {
            "storage": hydro_conv.convert_initial_storage(dadger, hidr, id_map),
            "filling_storage": [],
        },
    )

    system = dst / "system"
    _write_json(
        system / "buses.json", network_conv.convert_buses(dadger, id_map, start_date)
    )
    _write_json(
        system / "hydros.json",
        hydro_conv.convert_hydros(dadger, hidr, id_map, start_date),
    )
    lines_doc, line_bounds = network_conv.convert_lines(
        dadger, id_map, calendar, start_date
    )
    _write_json(system / "lines.json", lines_doc)
    _write_json(
        system / "pumping_stations.json",
        network_conv.convert_pumping_stations(dadger, id_map, start_date),
    )
    _write_json(
        system / "thermals.json",
        thermal_conv.convert_thermals(dadger, id_map, calendar, start_date),
    )
    _write_json(
        system / "hydro_production_models.json",
        hydro_conv.convert_production_models(id_map),
    )
    _write_parquet(system / "hydro_energy_productivity.parquet", productivity)
    ncs_registry = ncs_conv.convert_non_controllable_sources(
        dadger, id_map, calendar, start_date, renovaveis
    )
    _write_json(system / "non_controllable_sources.json", ncs_registry)

    scenarios = dst / "scenarios"
    _write_parquet(
        scenarios / "inflow_seasonal_stats.parquet",
        scenarios_conv.convert_inflow_stats_identity(id_map, calendar),
    )
    _write_parquet(
        scenarios / "external_inflow_scenarios.parquet",
        scenarios_conv.convert_external_inflows(vazoes, hidr, id_map, calendar),
    )
    _write_parquet(
        scenarios / "noise_openings.parquet",
        scenarios_conv.convert_noise_openings(
            vazoes,
            hidr,
            id_map,
            calendar,
            len(ncs_registry["non_controllable_sources"]),
        ),
    )
    _write_parquet(
        scenarios / "scenario_probabilities.parquet",
        scenarios_conv.convert_scenario_probabilities(vazoes, calendar),
    )
    _write_parquet(
        scenarios / "load_seasonal_stats.parquet",
        load_conv.convert_load_stats(dadger, id_map, calendar),
    )
    _write_json(
        scenarios / "load_factors.json",
        load_conv.convert_load_factors(dadger, id_map, calendar),
    )
    _write_parquet(
        scenarios / "non_controllable_stats.parquet",
        ncs_conv.convert_ncs_stats(dadger, id_map, calendar, renovaveis),
    )
    _write_json(
        scenarios / "non_controllable_factors.json",
        ncs_conv.convert_ncs_factors(dadger, id_map, calendar, renovaveis),
    )

    constraints = dst / "constraints"
    _write_parquet(
        constraints / "thermal_bounds.parquet",
        thermal_conv.convert_thermal_bounds(dadger, id_map, calendar),
    )
    _write_parquet(constraints / "line_bounds.parquet", line_bounds)
    hydro_bounds = bounds_conv.convert_hydro_bounds(dadger, hidr, id_map, calendar)
    if hydro_bounds.num_rows:
        _write_parquet(constraints / "hydro_bounds.parquet", hydro_bounds)

    _LOG.warning(
        "deferred at this milestone: GNL anticipation (dadgnl%s), boundary "
        "FCF (importer), windowed inflow inputs (solver 0.13), flow/volume/"
        "electrical constraint families (generic-constraints emitter)",
        " present" if files.dadgnl is not None else " absent",
    )
    _LOG.info(
        "converted %d buses, %d hydros, %d thermals, %d stages, terminal fan %d",
        id_map.n_buses,
        len(id_map.hydro_codes),
        len(id_map.thermal_codes),
        len(calendar),
        terminal_fan,
    )
