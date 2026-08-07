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

from cobre_bridge import diagnostics as dx
from cobre_bridge import emission_checks
from cobre_bridge.decomp import bounds as bounds_conv
from cobre_bridge.decomp import config as config_conv
from cobre_bridge.decomp import contracts as contracts_conv
from cobre_bridge.decomp import group_bounds as group_bounds_conv
from cobre_bridge.decomp import hydro as hydro_conv
from cobre_bridge.decomp import load as load_conv
from cobre_bridge.decomp import ncs as ncs_conv
from cobre_bridge.decomp import network as network_conv
from cobre_bridge.decomp import scenarios as scenarios_conv
from cobre_bridge.decomp import temporal as temporal_conv
from cobre_bridge.decomp import thermal as thermal_conv
from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)

#: Same logger name ``emission_checks`` itself logs to when no diagnostics
#: sink is active — used to re-emit each collected finding below so a
#: standalone call (no outer ``dx.collect()``) still logs under the check
#: module's own name, exactly as it did before this wiring existed.
_EMISSION_CHECKS_LOGGER = logging.getLogger("cobre_bridge.emission_checks")


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
    revision = caso.read_text(encoding="latin-1").split()[0]

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


def _describe_emission_check_errors(errors: list[dx.Diagnostic]) -> str:
    """Render *errors* (every entry ``ERROR`` severity) as a multi-line,
    human-readable block naming the failing rule and the offending entities —
    the message :func:`convert_decomp_case` raises when any is found. Each
    diagnostic's own detail table (hydro/thermal/line id, stage, column) is
    included, not just its one-line summary, so the exception is actionable
    on its own without re-running with a diagnostics sink attached.
    """
    lines: list[str] = []
    for diagnostic in errors:
        lines.append(f"- {diagnostic.code}: {diagnostic.summary}")
        if diagnostic.table is not None:
            for row in diagnostic.table.rows:
                lines.append("    " + ", ".join(str(value) for value in row))
    return "\n".join(lines)


def convert_decomp_case(src: Path, dst: Path, *, force: bool = False) -> None:
    """Convert one deck revision into a Cobre case directory.

    Raises
    ------
    ValueError
        If the post-emission self-checks (cobre 0.13 rules 43/41/45/38/36 and
        the ``block_id``-range rule; see :mod:`cobre_bridge.emission_checks`)
        find an ``ERROR``-severity violation in the converted artifacts. An
        ``INFO`` finding (e.g. rule 43's "not applicable" report, always the
        case for DECOMP) never raises.
    """
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
    fan_probabilities = scenarios_conv.terminal_fan_probabilities(vazoes, calendar)

    tx = float(dadger.tx.taxa) / 100.0

    dst.mkdir(parents=True, exist_ok=True)

    _write_json(
        dst / "config.json",
        config_conv.convert_config(dadger, len(fan_probabilities)),
    )
    stages_dict = temporal_conv.convert_stages(
        calendar, annual_discount_rate=tx, fan_probabilities=fan_probabilities
    )
    _write_json(dst / "stages.json", stages_dict)

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
    hydros_dict = hydro_conv.convert_hydros(dadger, hidr, id_map, start_date)
    _write_json(system / "hydros.json", hydros_dict)
    lines_doc, line_bounds = network_conv.convert_lines(
        dadger, id_map, calendar, start_date
    )
    _write_json(system / "lines.json", lines_doc)
    _write_json(
        system / "pumping_stations.json",
        network_conv.convert_pumping_stations(dadger, id_map, start_date),
    )
    thermals_dict = thermal_conv.convert_thermals(dadger, id_map, calendar, start_date)
    _write_json(system / "thermals.json", thermals_dict)
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
    load_stats = load_conv.convert_load_stats(dadger, id_map, calendar)
    _write_parquet(scenarios / "load_seasonal_stats.parquet", load_stats)
    _write_json(
        scenarios / "load_factors.json",
        load_conv.convert_load_factors(dadger, id_map, calendar),
    )
    ncs_stats = ncs_conv.convert_ncs_stats(dadger, id_map, calendar, renovaveis)
    _write_parquet(scenarios / "non_controllable_stats.parquet", ncs_stats)
    _write_json(
        scenarios / "non_controllable_factors.json",
        ncs_conv.convert_ncs_factors(dadger, id_map, calendar, renovaveis),
    )
    # Under the node-native explicit tree every stochastic class is sourced
    # externally: inflow (the tree), NCS (renewables), and load. cobre landed
    # scheme-aware load membership (`std > 0 OR load_scheme == External`), so a
    # deterministic (std = 0) external load class still occupies its noise slot
    # and standardizes to eta = 0 (value == mean) — no longer the in-sample-with-
    # null-std workaround. Each deterministic per-(entity, stage) mean fans out
    # unchanged across the same per-stage scenario columns as the inflow library
    # (1 on the trunk, the terminal fan width at the last stage) for joint
    # scenario_id coherence across the external classes.
    scenario_counts = [1] * (len(calendar) - 1) + [len(fan_probabilities)]
    _write_parquet(
        scenarios / "external_ncs_scenarios.parquet",
        scenarios_conv.deterministic_external_scenarios(
            ncs_stats,
            entity_column="ncs_id",
            value_in="mean",
            value_out="value",
            scenario_counts=scenario_counts,
        ),
    )
    _write_parquet(
        scenarios / "external_load_scenarios.parquet",
        scenarios_conv.deterministic_external_scenarios(
            load_stats,
            entity_column="bus_id",
            value_in="mean_mw",
            value_out="value_mw",
            scenario_counts=scenario_counts,
        ),
    )

    constraints = dst / "constraints"
    thermal_bounds_table = thermal_conv.convert_thermal_bounds(dadger, id_map, calendar)
    hydro_bounds = bounds_conv.convert_hydro_bounds(dadger, hidr, id_map, calendar)
    # ticket-026/027: B8 per-group per-stage availability (installed × MP ×
    # FD, capped by the ρ_eq·q_max hydraulic ceiling) for every plant,
    # single-group and Itaipu's own two per-frequency groups (code 66) alike.
    # Sparse by construction — a value only lands here where it lowers that
    # group's own declared envelope — so cobre rule 45 (a group-bound max_*
    # may not exceed that group's declared max) is now reachable; its mirror
    # is wired into the self-check block below.
    availability_values, availability_deltas = (
        hydro_conv.convert_hydro_group_availability(dadger, hidr, id_map, calendar)
    )
    group_bounds = group_bounds_conv.convert_hydro_unit_group_bounds(
        availability_values, calendar
    )

    # epic-01 (contracts): the CI/CE energy-contract model, read once and
    # shared by both emitters. Real decks carry no usable contract data
    # (rv0's single placeholder is skipped by D6, rv3 has none), so the
    # visible effect on a real conversion is: no contract files written, no
    # self-check finding, no regression.
    contracts = contracts_conv.read_contracts(dadger, calendar)
    contracts_conv.warn_nonnull_loss_factor(contracts)
    energy_contracts_doc = contracts_conv.convert_energy_contracts(
        contracts, id_map, calendar, start_date
    )
    contract_bounds_table = contracts_conv.convert_contract_bounds(contracts, calendar)

    # B8's attached measurement obligation: the hydraulic ceiling binding
    # below the installed×MP×FD availability is an *accepted cost*, not an
    # error — but it must be recorded, not assumed away. One INFO Diagnostic
    # per conversion (never raises), the ten worst offenders by how far under
    # the availability ceiling the hydraulic cap sits.
    if availability_deltas:
        worst = sorted(
            availability_deltas, key=lambda row: row.pct_under, reverse=True
        )[:10]
        dx.emit(
            dx.Diagnostic(
                code="hydro-availability-hydraulic-cap-binds",
                severity=dx.Severity.INFO,
                category="Hydro availability",
                title=(
                    "Hydraulic cap binds below availability "
                    f"({len(availability_deltas)} (hydro, stage) row(s))"
                ),
                summary=(
                    f"{len(availability_deltas)} single-group (hydro, stage) row(s) "
                    "have the ρ_eq·q_max hydraulic ceiling below installed×MP×FD; "
                    "the emitted max_generation_mw overlay is capped to the (lower) "
                    "hydraulic value on these rows, below the source model's "
                    "reported availability"
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Hydro",
                        "Code",
                        "Stage",
                        "Hydraulic MW",
                        "Availability MW",
                        "% under",
                    ],
                    rows=[
                        [
                            row.name,
                            row.code,
                            row.stage_id,
                            round(row.hydraulic_mw, 3),
                            round(row.availability_mw, 3),
                            round(row.pct_under, 2),
                        ]
                        for row in worst
                    ],
                    justify=["left", "right", "right", "right", "right", "right"],
                ),
            ),
            logger=_LOG,
        )

    # Post-emission self-checks (cobre 0.13 rules 43, 41, 45, 38, 36, and the
    # block_id-range rule) — a courtesy mirror of cheap cobre invariants over
    # the in-memory artifacts, run before the constraint tables are written.
    # DECOMP writes max_turbined_m3s/max_generation_mw only on the entity
    # (hydro_bounds carries min_outflow_m3s only), so the rule-43 check below
    # is always "not applicable" here — reported explicitly, not silently
    # skipped. See cobre_bridge.emission_checks for the rule scope.
    #
    # Unlike the NEWAVE pipeline (whose caller, convert_newave_case, opens one
    # dx.collect() around the whole conversion and returns every diagnostic on
    # the ConversionReport for the CLI to render and gate on), this simpler
    # DECOMP entry point has no report object a caller could inspect
    # afterwards — so an ERROR-severity finding here must fail the conversion
    # outright rather than degrade to a log line a caller could easily miss.
    # The local dx.collect() only exists so this function can inspect
    # severity; every collected finding is then re-emitted via dx.emit() so a
    # caller that already has its own sink open (e.g. a test wrapping this
    # call in dx.collect()) still observes it exactly as it would have
    # without this wiring, and a standalone call still logs it.
    bound_families = [
        emission_checks.BoundFamily("Hydro", "hydro_id", hydro_bounds),
        emission_checks.BoundFamily("Thermal", "thermal_id", thermal_bounds_table),
        emission_checks.BoundFamily("Line", "line_id", line_bounds),
        emission_checks.BoundFamily(
            "HydroUnitGroup",
            "hydro_id",
            group_bounds,
            group_column="hydro_unit_group_id",
        ),
        emission_checks.BoundFamily("Contract", "contract_id", contract_bounds_table),
    ]
    with dx.collect() as check_diagnostics:
        emission_checks.check_hydro_bounds_no_raising(hydros_dict, hydro_bounds)
        emission_checks.check_unit_group_envelope(hydros_dict)
        emission_checks.check_group_bound_envelope(hydros_dict, group_bounds)
        emission_checks.check_bound_row_uniqueness(bound_families)
        emission_checks.check_bound_block_id_range(stages_dict, bound_families)
        emission_checks.check_block_id_not_on_anticipated_thermal(
            thermals_dict, thermal_bounds_table
        )
    for diagnostic in check_diagnostics:
        dx.emit(diagnostic, logger=_EMISSION_CHECKS_LOGGER)
    check_errors = [d for d in check_diagnostics if d.severity is dx.Severity.ERROR]
    if check_errors:
        raise ValueError(
            f"DECOMP conversion failed {len(check_errors)} post-emission "
            "self-check error(s) (cobre 0.13 rules 43/41/45/38/36/"
            "block_id-range):\n"
            f"{_describe_emission_check_errors(check_errors)}"
        )

    _write_parquet(constraints / "thermal_bounds.parquet", thermal_bounds_table)
    _write_parquet(constraints / "line_bounds.parquet", line_bounds)
    if hydro_bounds.num_rows:
        _write_parquet(constraints / "hydro_bounds.parquet", hydro_bounds)
    if group_bounds.num_rows:
        _write_parquet(constraints / "hydro_unit_group_bounds.parquet", group_bounds)
    if contracts:
        _write_json(system / "energy_contracts.json", energy_contracts_doc)
    if contract_bounds_table.num_rows:
        _write_parquet(constraints / "contract_bounds.parquet", contract_bounds_table)

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
        len(fan_probabilities),
    )
