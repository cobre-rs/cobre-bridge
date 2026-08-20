"""Conversion pipeline: orchestrates entity and temporal/stochastic converters.

Reads a source-model case directory and writes a complete Cobre case directory.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cobre_bridge import diagnostics as dx
from cobre_bridge import emission_checks
from cobre_bridge.case import NewaveCase
from cobre_bridge.converters import constraints as constraints_conv
from cobre_bridge.converters import hydro as hydro_conv
from cobre_bridge.converters import inflow_windows
from cobre_bridge.converters import initial_conditions as ic_conv
from cobre_bridge.converters import network as network_conv
from cobre_bridge.converters import scalar_parameters as scalar_params_conv
from cobre_bridge.converters import stochastic as stochastic_conv
from cobre_bridge.converters import tailrace as tailrace_conv
from cobre_bridge.converters import temporal as temporal_conv
from cobre_bridge.converters import thermal as thermal_conv
from cobre_bridge.generic_constraint_format import GENERIC_BOUNDS_COLUMNS
from cobre_bridge.id_map import build_id_map

logger = logging.getLogger(__name__)

#: Coarse conversion phases reported to an optional ``on_phase`` callback (drives the
#: CLI progress bar). The order matches the boundaries in
#: :func:`_convert_newave_case_impl`; the count is the progress-bar total.
CONVERSION_PHASE_LABELS: tuple[str, ...] = (
    "Discovering files",
    "Converting entities",
    "Converting temporal & stochastic",
    "Converting constraints",
    "Writing JSON",
    "Writing Parquet",
)


@dataclass
class ConversionReport:
    """Summary of a completed the source-model-to-Cobre conversion."""

    hydro_count: int = 0
    thermal_count: int = 0
    bus_count: int = 0
    line_count: int = 0
    stage_count: int = 0
    #: Structured findings (rich tables, severities, remediation) for the CLI to
    #: render. Populated by :func:`convert_newave_case`.
    diagnostics: list[dx.Diagnostic] = field(default_factory=list)
    #: Flat WARNING-severity summary strings, kept for backward-compatible consumers
    #: (derived from :attr:`diagnostics`).
    warnings: list[str] = field(default_factory=list)
    #: Absolute output paths the conversion produced (real run) or would produce
    #: (dry run), in write order. Optional Parquet tables that were skipped are
    #: absent. Populated by :func:`_convert_newave_case_impl`.
    would_write_paths: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Converted: {self.hydro_count} hydros, "
            f"{self.thermal_count} thermals, "
            f"{self.bus_count} buses, "
            f"{self.line_count} lines, "
            f"{self.stage_count} stages"
        )


class _WarningCollector(logging.Handler):
    """Capture ``WARNING``+ records emitted under ``cobre_bridge`` during a run.

    Converters log every degraded-input substitution (vazpast → empty, c_adic →
    no load, EXPT/RE skipped, REE.DAT cutoff fallback, …) at ``WARNING`` level.
    Attaching this to the package logger for the duration of a conversion lets
    :func:`convert_newave_case` report those degradations through
    :attr:`ConversionReport.warnings` instead of letting them pass silently.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.messages.append(record.getMessage())


class _ConstraintIdAllocator:
    """Hands out contiguous, non-overlapping generic-constraint ID ranges.

    VminOP, electric, and AGRINT constraints share one 0-based ID space. Each
    converter is given the next free start ID; after it returns, ``advance(n)``
    reserves the ``n`` IDs it produced so the next converter starts past them.
    This replaces the by-hand offset arithmetic
    (``start_id = vminop_count + electric_count``) that silently corrupts IDs if
    a term is missed.
    """

    def __init__(self) -> None:
        self.next_id = 0

    def advance(self, count: int) -> None:
        self.next_id += count


def _compute_prod_media_sin_safe(case: NewaveCase) -> float | None:
    """Return ``PROD_MEDIA_SIN`` (mean PRODT), or ``None`` on failure.

    The source model converts the PENALID R$/MWh penalties (VAZMIN, TURBMN/TURBMX, …)
    with the mean **PRODT** — the equivalent productivity from vol_min to vol_max — over
    all existing plants including zeros. See :func:`hydro.compute_prodt_sin_mean`;
    validated against pmo.dat's applied penalty (0.6299 ↔ VAZMIN 821.78). Falls back to
    ``None`` (caller uses its legacy point-reference mean) when the source model files
    cannot be read — e.g. in unit tests that mock the pipeline.
    """
    try:
        return hydro_conv.compute_prodt_sin_mean(case)
    except (OSError, ValueError, AttributeError, TypeError, KeyError):
        return None


def _compute_per_stage_sin_productivities(
    case: NewaveCase,
) -> tuple[list[float], list[float]] | None:
    """Return ``(PROD_MEDIA_SIN[s], MAX_PRODTACUM_SIN[s])`` per stage, or None.

    These SIN-aggregate productivity constants are what the source model uses to convert
    its flow-domain hydro penalties (manual §3.24 p.87).

    - ``PROD_MEDIA_SIN[s]`` = mean **PRODT** (equivalent ρ vol_min→vol_max) over
      the existing plants. PRODT is **structural** — it does *not* track the
      seasonal reference volume — so this is essentially **flat** across the
      horizon. pmo.dat confirms it: the applied VAZMIN/TURBMN/TURBMX penalties
      vary only 0.15% over the whole study (= the per-config PRODT-mean spread), not the
      ~2% that the old seasonal point-reference ρ implied. We therefore emit a constant
      ``compute_prodt_sin_mean`` for every stage; the per-stage penalty override then
      collapses to the base for the ρ_avg-scaled columns (correct — the source model
      does not seasonalise these penalties). The genuinely seasonal per-plant ρ still
      ships in ``hydro_energy_productivity.parquet`` for the *generation* model — that
      is a separate concern from the penalty.
    - ``MAX_PRODTACUM_SIN`` = max accumulated cascade ρ at **altura máxima**,
      **constant** over the horizon. It governs the DESVIO ("outros usos") and
      evaporation penalties, which pmo.dat reports as *fixed* — the max is
      ITAIPU's cascade, which carries no CFUGA/CMONT override, so it never moves.
      (See :func:`constraints.compute_max_prodtacum_sin`.)

    Both lists span the full horizon (``_total_study_stages``, incl. post-study). Falls
    back to ``None`` when the source model files can't be read (mocked tests).
    """
    try:
        rho_avg = hydro_conv.compute_per_stage_prodt_sin_mean(case)
        if not rho_avg:
            return None
        max_prodtacum = constraints_conv.compute_max_prodtacum_sin(case)
        if max_prodtacum is None:
            return None
        rho_max_acum = [max_prodtacum] * len(rho_avg)
        return rho_avg, rho_max_acum
    except (
        OSError,
        ValueError,
        AttributeError,
        TypeError,
        KeyError,
        StopIteration,
        ZeroDivisionError,
    ):
        return None


# Backward-compatible alias for importers still referencing this private name;
# the builder itself lives in ``cobre_bridge.id_map.build_id_map``.
_build_id_map = build_id_map


def _merge_hydro_bounds(
    withdrawal: pa.Table | None,
    storage: pa.Table | None,
) -> pa.Table | None:
    """Merge water withdrawal and storage bounds into one table.

    Both inputs are optional.  When both are present, they are outer-joined
    on ``(hydro_id, stage_id)`` so that rows from either source appear in the
    result with nulls for missing columns.
    """
    if withdrawal is None and storage is None:
        return None
    if withdrawal is None:
        return storage
    if storage is None:
        return withdrawal

    import polars as pl

    w = pl.from_arrow(withdrawal)
    s = pl.from_arrow(storage)
    merged = w.join(s, on=["hydro_id", "stage_id"], how="full", coalesce=True)
    merged = merged.sort("hydro_id", "stage_id")
    return merged.to_arrow()


def _fold_head_turbined_bounds(
    base: pa.Table | None,
    head_turbined: pa.Table | None,
) -> pa.Table | None:
    """Fold per-stage head-corrected ``max_turbined_m3s`` into the hydro bounds.

    ``head_turbined`` carries the per-(hydro, stage) engolimento cap derived from
    the per-stage operating head. An explicit TURBMAXT override already present in
    ``base`` (from ``convert_storage_bounds``) takes precedence — it is a deliberate
    operational limit — so the head-derived value only fills cells TURBMAXT leaves
    empty (the common case: plants with CFUGA/CMONT but no TURBMAXT).
    """
    if head_turbined is None:
        return base

    import polars as pl

    h = pl.from_arrow(head_turbined)
    if base is None:
        return h.sort("hydro_id", "stage_id").to_arrow()

    b = pl.from_arrow(base)
    merged = b.join(h, on=["hydro_id", "stage_id"], how="full", coalesce=True)
    if "max_turbined_m3s_right" in merged.columns:
        # Explicit TURBMAXT (left) wins; head-derived (right) fills the rest.
        merged = merged.with_columns(
            pl.coalesce(["max_turbined_m3s", "max_turbined_m3s_right"]).alias(
                "max_turbined_m3s"
            )
        ).drop("max_turbined_m3s_right")
    merged = merged.sort("hydro_id", "stage_id")
    return merged.to_arrow()


def convert_newave_case(
    src: Path,
    dst: Path,
    *,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> ConversionReport:
    """Convert a source-model case directory to a Cobre case directory.

    Parameters
    ----------
    src:
        Path to the source model case directory.  Must exist and contain all required
        the source model input files.
    dst:
        Path to the output Cobre case directory.  Must not exist or must be
        empty (call site is responsible for enforcing the --force contract
        before calling this function).
    dry_run:
        When ``True``, run the full in-memory conversion but write nothing to
        *dst* (no files, no subdirectories). The would-write paths are still
        recorded in :attr:`ConversionReport.would_write_paths`.

    Returns
    -------
    ConversionReport
        Summary of what was converted, including a ``warnings`` list of every
        degraded-input substitution that occurred (empty when the conversion
        ran clean).

    Raises
    ------
    FileNotFoundError
        If *src* does not exist, is not a directory, or a required the source model file
        is missing.
    """
    collector = _WarningCollector()
    pkg_logger = logging.getLogger("cobre_bridge")
    pkg_logger.addHandler(collector)
    try:
        # Structured diagnostics emitted by converters land in ``collected``; any
        # remaining ``logger.warning`` strings (sites not yet migrated) are picked
        # up by ``collector`` and bridged below, so every warning still surfaces.
        with dx.collect() as collected:
            report = _convert_newave_case_impl(src, dst, on_phase, dry_run=dry_run)
    except BaseException:
        # The write phase is a sequence of independent file writes with no
        # rollback, so a failure partway through (disk full, a converter
        # raising mid-write, an interrupt) can leave a subset of the output
        # files behind. Remove the known pipeline outputs so a half-written
        # case is never mistaken for a complete one and a plain (no --force)
        # re-run is not refused as "destination not empty". Re-raise so the
        # CLI still reports the original failure. Skipped under a dry run:
        # nothing was written, and dst may be a pre-existing populated
        # directory the user never asked to clear.
        if not dry_run:
            _clear_dst_contents(dst, NEWAVE_CLEARED_ARTIFACTS)
        raise
    finally:
        pkg_logger.removeHandler(collector)
    report.diagnostics = _finalize_diagnostics(collected, collector.messages)
    # Backward-compatible flat WARNING strings derived from the diagnostics.
    report.warnings = [
        d.summary for d in report.diagnostics if d.severity is dx.Severity.WARNING
    ]
    return report


def _finalize_diagnostics(
    collected: list[dx.Diagnostic], legacy_messages: list[str]
) -> list[dx.Diagnostic]:
    """Combine structured diagnostics with bridged legacy ``logger.warning`` strings.

    Structured diagnostics are de-duplicated by ``(code, summary)`` (a converter run
    more than once should surface a finding only once). Each remaining captured
    warning string — emitted by a site not yet migrated to :func:`diagnostics.emit` —
    is wrapped in a generic WARNING diagnostic so it still renders through the same
    pipeline, with first-occurrence order preserved.
    """
    result: list[dx.Diagnostic] = []
    seen: set[tuple[str, str]] = set()
    for diag in collected:
        key = (diag.code, diag.summary)
        if key in seen:
            continue
        seen.add(key)
        result.append(diag)
    for message in dict.fromkeys(legacy_messages):
        result.append(
            dx.Diagnostic(
                code="legacy-warning",
                severity=dx.Severity.WARNING,
                category="Other warnings",
                title="Conversion warning",
                summary=message,
            )
        )
    return result


def _convert_newave_case_impl(
    src: Path,
    dst: Path,
    on_phase: Callable[[str], None] | None = None,
    *,
    dry_run: bool = False,
) -> ConversionReport:
    """Run the conversion pipeline (warning capture handled by the wrapper).

    ``on_phase`` (when given) is called once at each boundary in
    :data:`CONVERSION_PHASE_LABELS`, so the CLI can advance a progress bar without
    the pipeline knowing anything about rendering.

    When ``dry_run`` is ``True``, the build phases run exactly as for a real run,
    but the write phase creates no directories and writes no files; the paths that
    would have been written are still accumulated and recorded on the report.
    """
    report = ConversionReport()
    step = on_phase if on_phase is not None else (lambda _label: None)

    step("Discovering files")
    logger.debug("Discovering NEWAVE files from %s", src)
    # Build the parsed-case object once; every converter reads its parsed inputs from
    # ``case`` (each source-model file parsed once and cached).
    case = NewaveCase.from_directory(src)

    logger.debug("Building NewaveIdMap from %s", src)
    id_map = case.id_map

    step("Converting entities")
    logger.debug("Converting hydros")
    hydros_dict = hydro_conv.convert_hydros(case, id_map)

    logger.debug("Computing base hydro productivities")
    base_productivities = hydro_conv.compute_base_productivities(case, id_map)

    logger.debug("Generating hydro geometry")
    cadastro = hydro_conv.read_cadastro(case)
    geometry_table = hydro_conv.generate_hydro_geometry(cadastro, id_map)

    logger.debug("Converting thermals")
    thermals_dict = thermal_conv.convert_thermals(case, id_map)

    logger.debug("Converting buses")
    buses_dict = network_conv.convert_buses(case, id_map)

    logger.debug("Converting lines")
    lines_dict = network_conv.convert_lines(case, id_map)

    logger.debug("Converting penalties")
    # DESVIO ("outros usos") + evaporation use MAX_PRODTACUM_SIN (max accumulated ρ at
    # altura máxima, constant); VAZMIN/TURBMN/TURBMX use PROD_MEDIA_SIN (mean PRODT,
    # drifting per CFUGA/CMONT config). Both come from the source model inputs and
    # return None when files are absent (mocked-pipeline tests), so convert_penalties
    # falls back to its own legacy approximation.
    max_prodtacum_sin = constraints_conv.compute_max_prodtacum_sin(case)
    prod_media_sin = _compute_prod_media_sin_safe(case)
    penalties_dict = network_conv.convert_penalties(
        case,
        hydros_dict,
        productivities=base_productivities,
        max_accumulated_productivity=max_prodtacum_sin,
        prod_media_sin=prod_media_sin,
    )

    step("Converting temporal & stochastic")
    logger.debug("Converting stages")
    stages_dict = temporal_conv.convert_stages(case, id_map)

    logger.debug("Converting config")
    config_dict = temporal_conv.convert_config(case)

    logger.debug("Converting initial conditions")
    ic_dict = ic_conv.convert_initial_conditions(case, id_map)

    logger.debug("Converting the hydrological tendency to conditioning windows")
    observation_windows = inflow_windows.convert_recent_observation_windows(
        case, id_map
    )
    if observation_windows:
        ic_dict["recent_observations"] = observation_windows

    logger.debug("Converting inflow stats")
    inflow_table = stochastic_conv.convert_inflow_stats(case, id_map)

    logger.debug("Converting load stats")
    load_table = stochastic_conv.convert_load_stats(case, id_map)

    logger.debug("Converting inflow history from vazoes.dat")
    inflow_history_table = inflow_windows.convert_inflow_history_windows(case, id_map)

    logger.debug("Converting water withdrawal")
    withdrawal_table = hydro_conv.convert_water_withdrawal(case, id_map)

    logger.debug("Converting storage bounds from VMAXT/VMINT")
    storage_bounds_table = hydro_conv.convert_storage_bounds(case, id_map)

    logger.debug("Converting per-stage head-corrected turbined caps")
    head_turbined_table = hydro_conv.convert_turbined_bounds_head_corrected(
        case, id_map
    )

    # Generic constraints (VminOP, electric, AGRINT) share one ID space; the
    # allocator hands each converter the next free start ID so the pipeline no
    # longer threads `start_id = vminop_count + electric_count` by hand.
    constraint_ids = _ConstraintIdAllocator()

    step("Converting constraints")
    logger.debug("Converting VminOP constraints")
    vminop_result = constraints_conv.convert_vminop_constraints(case, id_map)
    vminop_referenced_ids: list[int] = []
    rho_acum_overrides: dict[int, list[float]] = {}
    if vminop_result is not None:
        vminop_referenced_ids = list(vminop_result.referenced_hydro_ids)
        rho_acum_overrides = vminop_result.rho_acum_overrides
        constraint_ids.advance(
            len(vminop_result.constraints_dict.get("constraints", []))
        )

    logger.debug("Converting electric constraints")
    electric_result = constraints_conv.convert_electric_constraints(
        case, id_map, start_id=constraint_ids.next_id
    )
    if electric_result is not None:
        constraint_ids.advance(len(electric_result.constraints))

    logger.debug("Converting AGRINT group constraints")
    agrint_result = constraints_conv.convert_agrint_constraints(
        case, id_map, start_id=constraint_ids.next_id
    )

    logger.debug("Converting load factors")
    load_factors_dict = stochastic_conv.convert_load_factors(case, id_map)

    logger.debug("Converting line bounds (folds in per-block exchange factors)")
    line_bounds_table = network_conv.convert_line_bounds(case, id_map)

    logger.debug("Converting non-controllable sources")
    ncs_dict = network_conv.convert_non_controllable_sources(case, id_map)

    logger.debug("Converting NCS block factors")
    ncs_factors_dict = network_conv.convert_ncs_factors(case, id_map)

    logger.debug("Converting NCS stats")
    ncs_stats_table = network_conv.convert_ncs_stats(case, id_map)

    logger.debug("Converting production models")
    production_models_dict = hydro_conv.convert_production_models(case, id_map)

    logger.debug("Converting hydro energy productivity overrides")
    hydro_energy_productivity_table = hydro_conv.convert_hydro_energy_productivity(
        case, id_map
    )

    logger.debug("Converting tailrace curves from polinjus")
    tailrace_table = tailrace_conv.convert_tailrace_curves(case, id_map)

    logger.debug("Converting thermal bounds from expt.dat and manutt.dat")
    thermal_bounds_table = thermal_conv.convert_thermal_bounds(case, id_map)

    # Merge hydro_bounds now (hoisted from the write phase below) so the
    # post-emission self-checks see the exact table that will be written.
    hydro_bounds_table = _merge_hydro_bounds(withdrawal_table, storage_bounds_table)
    hydro_bounds_table = _fold_head_turbined_bounds(
        hydro_bounds_table, head_turbined_table
    )

    # Clamp any per-stage MAX bound above the plant's declared envelope back to
    # the declared value (WARNING, not silent), so the rule-43 self-check below
    # passes on the result.
    hydro_bounds_table = emission_checks.clamp_hydro_bounds_to_declared(
        hydros_dict, hydro_bounds_table
    )

    # Post-emission self-checks: mirror cheap cobre load invariants
    # (rules 43, 41, 36, and the block_id-range rule) over the in-memory
    # artifacts before anything is written. See cobre_bridge.emission_checks
    # for the rule scope.
    bound_families = [
        emission_checks.BoundFamily("Hydro", "hydro_id", hydro_bounds_table),
        emission_checks.BoundFamily("Thermal", "thermal_id", thermal_bounds_table),
        emission_checks.BoundFamily("Line", "line_id", line_bounds_table),
    ]

    def _run_checks() -> None:
        emission_checks.check_hydro_bounds_no_raising(hydros_dict, hydro_bounds_table)
        emission_checks.check_unit_group_envelope(hydros_dict)
        emission_checks.check_bound_row_uniqueness(bound_families)
        emission_checks.check_bound_block_id_range(stages_dict, bound_families)

    emission_checks.run_and_gate(_run_checks)

    if not dry_run:
        (dst / "system").mkdir(parents=True, exist_ok=True)
        (dst / "scenarios").mkdir(parents=True, exist_ok=True)
        (dst / "constraints").mkdir(parents=True, exist_ok=True)

    # Every output path is routed through ``_write_json`` / ``_write_parquet`` so
    # the would-write listing and the dry-run gate live in exactly one place each,
    # rather than guarding ~30 individual write sites.
    would_write: list[Path] = []

    def _write_json(path: Path, data: dict) -> None:
        would_write.append(path)
        if dry_run:
            logger.debug("Would write %s", path)
            return
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug("Wrote %s", path)

    def _write_parquet(table: pa.Table, path: Path) -> None:
        would_write.append(path)
        if dry_run:
            logger.debug("Would write %s", path)
            return
        pq.write_table(table, path, compression="zstd")
        logger.debug("Wrote %s", path)

    step("Writing JSON")
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
    _write_json(dst / "scenarios" / "non_controllable_factors.json", ncs_factors_dict)

    _write_json(dst / "system" / "hydro_production_models.json", production_models_dict)

    # Declare per-hydro @rho_eq_h{id} / @rho_acum_h{id} computed parameters for
    # every hydro in the case. cobre rejects any @name token that has not been
    # declared, so we always emit the file so handwritten or generated
    # constraint expressions can reference any per-hydro productivity. The
    # ids set is the union of the VminOP-referenced ids (covers the values we
    # currently emit) and every hydro that has a production-model entry
    # (covers future references).
    all_hydro_ids = sorted(
        set(vminop_referenced_ids)
        | {
            int(m["hydro_id"])
            for m in production_models_dict.get("production_models", [])
        }
    )
    scalar_parameters_dict = scalar_params_conv.build_scalar_parameters(
        all_hydro_ids,
        rho_acum_per_stage_overrides=rho_acum_overrides or None,
    )
    _write_json(dst / "constraints" / "generic_parameters.json", scalar_parameters_dict)

    step("Writing Parquet")
    geometry_path = dst / "system" / "hydro_geometry.parquet"
    _write_parquet(geometry_table, geometry_path)

    if hydro_energy_productivity_table.num_rows > 0:
        hep_path = dst / "system" / "hydro_energy_productivity.parquet"
        _write_parquet(hydro_energy_productivity_table, hep_path)

    # Optional tailrace curves (polinjus) — only written when the case ships them.
    if tailrace_table is not None and tailrace_table.num_rows > 0:
        tailrace_path = dst / "system" / "tailrace_curves.parquet"
        _write_parquet(tailrace_table, tailrace_path)

    inflow_path = dst / "scenarios" / "inflow_seasonal_stats.parquet"
    _write_parquet(inflow_table, inflow_path)

    load_path = dst / "scenarios" / "load_seasonal_stats.parquet"
    _write_parquet(load_table, load_path)

    history_path = dst / "scenarios" / "inflow_history.parquet"
    _write_parquet(inflow_history_table, history_path)

    constraints_dir = dst / "constraints"
    line_bounds_path = constraints_dir / "line_bounds.parquet"
    _write_parquet(line_bounds_table, line_bounds_path)

    ncs_stats_path = dst / "scenarios" / "non_controllable_stats.parquet"
    _write_parquet(ncs_stats_table, ncs_stats_path)

    if hydro_bounds_table is not None:
        hydro_bounds_path = constraints_dir / "hydro_bounds.parquet"
        _write_parquet(hydro_bounds_table, hydro_bounds_path)

    if thermal_bounds_table is not None:
        thermal_bounds_path = constraints_dir / "thermal_bounds.parquet"
        _write_parquet(thermal_bounds_table, thermal_bounds_path)

    # Per-bus excess-cost override: forbid energy excess at fictitious
    # submarkets (pure transshipment nodes) by pricing it prohibitively.
    bus_penalty_table = network_conv.convert_bus_penalty_overrides(case, id_map)
    if bus_penalty_table is not None:
        bus_penalty_path = constraints_dir / "penalty_overrides_bus.parquet"
        _write_parquet(bus_penalty_table, bus_penalty_path)

    # Per-stage hydro penalty override: The source model's PROD_MEDIA_SIN /
    # MAX_PRODTACUM_SIN shift with seasonal (VOLREF_SAZ) and temporal (CFUGA/CMONT)
    # productivity changes, so the ρ-scaled flow-domain hydro penalties are
    # stage-varying. Emitted sparsely (only stages/columns that differ from
    # penalties.json), keeping the penalty conversion coherent with the per-stage ρ
    # already shipped in system/hydro_energy_productivity.parquet.
    per_stage_sin = _compute_per_stage_sin_productivities(case)
    if per_stage_sin is not None:
        per_stage_rho_avg, per_stage_rho_max_acum = per_stage_sin
        hydro_ids = [int(h["id"]) for h in hydros_dict.get("hydros", []) if "id" in h]
        hydro_penalty_table = network_conv.convert_hydro_penalty_overrides(
            case,
            hydro_ids,
            penalties_dict["hydro"],
            per_stage_rho_avg,
            per_stage_rho_max_acum,
        )
        if hydro_penalty_table is not None:
            hydro_penalty_path = constraints_dir / "penalty_overrides_hydro.parquet"
            _write_parquet(hydro_penalty_table, hydro_penalty_path)

    # Merge VminOP and electric constraints into a single output.
    all_constraints: list[dict] = []
    bounds_tables: list[pa.Table] = []

    if vminop_result is not None:
        all_constraints.extend(vminop_result.constraints_dict.get("constraints", []))
        # VminOP bounds table has no block_id column; add a null column and
        # reorder to match the canonical F3 schema.
        vminop_bounds = vminop_result.bounds
        vminop_bounds_extended = vminop_bounds.append_column(
            pa.field("block_id", pa.int32()),
            pa.array([None] * len(vminop_bounds), type=pa.int32()),
        ).select(GENERIC_BOUNDS_COLUMNS)
        bounds_tables.append(vminop_bounds_extended)

    if electric_result is not None:
        all_constraints.extend(electric_result.constraints)
        bounds_tables.append(electric_result.bounds)

    if agrint_result is not None:
        all_constraints.extend(agrint_result.constraints)
        bounds_tables.append(agrint_result.bounds)

    if all_constraints:
        merged_dict = {
            "$schema": (
                "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
                "/schemas/generic_constraints.schema.json"
            ),
            "constraints": all_constraints,
        }
        _write_json(constraints_dir / "generic_constraints.json", merged_dict)

        if bounds_tables:
            merged_bounds = pa.concat_tables(bounds_tables)
            gc_bounds_path = constraints_dir / "generic_constraint_bounds.parquet"
            _write_parquet(merged_bounds, gc_bounds_path)

    report.hydro_count = len(hydros_dict.get("hydros", []))
    report.thermal_count = len(thermals_dict.get("thermals", []))
    report.bus_count = len(buses_dict.get("buses", []))
    report.line_count = len(lines_dict.get("lines", []))
    report.stage_count = len(stages_dict.get("stages", []))
    report.would_write_paths = [str(p) for p in would_write]

    return report


@dataclass(frozen=True)
class ClearedArtifacts:
    """A conversion track's on-disk output set, for --force pre-clear and
    failure rollback: subdirectories removed as a tree, files unlinked."""

    subdirs: tuple[str, ...]
    files: tuple[str, ...]


NEWAVE_CLEARED_ARTIFACTS = ClearedArtifacts(
    subdirs=("system", "scenarios", "constraints"),
    files=(
        "config.json",
        "stages.json",
        "penalties.json",
        "initial_conditions.json",
        "conversion_manifest.json",
    ),
)


def _clear_dst_contents(dst: Path, artifacts: ClearedArtifacts) -> None:
    """Remove *artifacts*' known output subdirectories and top-level files from dst.

    Only the specific files/subdirectories named by *artifacts* are removed.
    This avoids accidentally deleting unrelated files in the destination
    directory.
    """
    for subdir in artifacts.subdirs:
        target = dst / subdir
        if target.exists():
            shutil.rmtree(target)

    for filename in artifacts.files:
        path = dst / filename
        if path.exists():
            path.unlink()
