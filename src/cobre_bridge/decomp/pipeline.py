"""Conversion orchestrator for DECOMP-like decks.

``convert_decomp_case(src, dst)`` discovers the deck (``caso.dat`` names
the revision extension; the ``rvN`` index file names the data files),
parses it, and writes a Cobre case directory. Scope is the ratified
loop-closing milestone: the deferred families (exchange network pending
the upstream accessor fix, renewables pending their reader, boundary FCF,
and the per-block/fidelity items) are logged loudly by their emitters or
here. GNL anticipation is now emitted (the anticipated ring + both temporal
boundaries), no longer deferred.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from idecomp.decomp import Dadger, Dadgnl, Vazoes

from cobre_bridge import diagnostics as dx
from cobre_bridge import emission_checks
from cobre_bridge.converters.constraints import (
    _SCHEMA_URL as _GENERIC_CONSTRAINTS_SCHEMA_URL,
)
from cobre_bridge.decomp import anticipated as anticipated_conv
from cobre_bridge.decomp import bounds as bounds_conv
from cobre_bridge.decomp import (
    bounds_accumulator,
    constraint_registers,
    libs_electrical_emit,
    single_term_bounds,
)
from cobre_bridge.decomp import cadastro as cadastro_conv
from cobre_bridge.decomp import config as config_conv
from cobre_bridge.decomp import constraints as constraints_conv
from cobre_bridge.decomp import contracts as contracts_conv
from cobre_bridge.decomp import fpha as fpha_conv
from cobre_bridge.decomp import group_bounds as group_bounds_conv
from cobre_bridge.decomp import hydro as hydro_conv
from cobre_bridge.decomp import libs_electrical as libs_electrical_conv
from cobre_bridge.decomp import load as load_conv
from cobre_bridge.decomp import ncs as ncs_conv
from cobre_bridge.decomp import network as network_conv
from cobre_bridge.decomp import scenarios as scenarios_conv
from cobre_bridge.decomp import temporal as temporal_conv
from cobre_bridge.decomp import thermal as thermal_conv
from cobre_bridge.decomp import travel_time as travel_time_conv
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.scalar_parameters import (
    build_decomp_scalar_parameters,
    write_scalar_parameters,
)
from cobre_bridge.pipeline import (
    ConversionReport,
    _clear_dst_contents,
    _finalize_diagnostics,
    _WarningCollector,
)

_LOG = logging.getLogger(__name__)

#: Coarse conversion phases reported to an optional ``on_phase`` callback (drives the
#: CLI progress bar). The order matches the boundaries in
#: :func:`_convert_decomp_case_impl`; the count is the progress-bar total.
DECOMP_CONVERSION_PHASE_LABELS: tuple[str, ...] = (
    "Discovering deck",
    "Converting entities",
    "Converting scenarios",
    "Resolving bounds",
    "Converting constraints",
    "Writing outputs",
)

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
    polinjus: Path | None
    #: The deck's LIBs-era electrical-constraint file (ticket-013), resolved
    #: via :func:`constraint_registers.resolve_libs_electrical_path`; ``None``
    #: when the deck carries none. Defaults to ``None`` so every pre-existing
    #: ``DecompFiles(...)`` call site keeps constructing without it.
    libs_restricao_eletrica: Path | None = None
    #: The deck's boundary-FCF header file (``cortesh.dat``), resolved via
    #: :func:`_resolve_fc_record_path` (the deck's ``FC NEWV21`` record) or,
    #: failing that, the ``cortesh*`` glob idiom; ``None`` when the deck
    #: carries no boundary FCF (ticket-007). Defaults to ``None`` for the
    #: same back-compat reason as ``libs_restricao_eletrica`` above.
    cortesh: Path | None = None
    #: The deck's boundary-FCF cut-record file: a single-stage partition
    #: export (``cortes-<estagio>.dat``, preferred) or the consolidated
    #: archive (``cortes.dat``), resolved via :func:`_resolve_fc_record_path`
    #: (the deck's ``FC NEWCUT`` record) or the glob idiom; ``None`` when
    #: absent (ticket-007).
    cortes: Path | None = None


#: ``FC`` register ``tipo`` mnemonics (``idecomp.decomp.Dadger.fc()``) naming
#: the boundary-FCF header and cut-record files respectively.
_FC_TIPO_CORTESH = "NEWV21"
_FC_TIPO_CORTES = "NEWCUT"


def _resolve_fc_record_path(dadger: Path, deck_dir: Path, *, tipo: str) -> Path | None:
    """Resolve one boundary-FCF file named by the deck's ``FC`` register.

    A lightweight fixed-width text scan of *dadger* — mirroring
    :func:`~cobre_bridge.decomp.constraint_registers.resolve_libs_electrical_path`'s
    own text-scan idiom for a deck-relative file named by an index entry —
    rather than a full :class:`idecomp.decomp.Dadger` parse (the caller
    re-parses *dadger* structurally right after discovery returns;
    duplicating that heavier, structured parse here would be wasted work and
    would need to guard against whatever exception surface a malformed
    dadger raises through it). The ``FC`` register
    (``idecomp.decomp.modelos.dadger.FC``) is fixed-width: identifier at
    columns 0:4, ``tipo`` mnemonic at columns 4:10, ``caminho`` at columns
    14:214 — confirmed against ``example/decomp-mar-26-rv2/dadger.rv2``'s own
    ``FC  NEWV21    cortesh.dat`` / ``FC  NEWCUT    cortes-004.dat`` lines.
    Resolves ``caminho`` (which may be a relative path, including
    parent-directory references, e.g. ``../../cortesh.dat``) against
    *deck_dir*.

    Never raises: returns ``None`` — falling through to the glob idiom —
    when *dadger* is unreadable, carries no ``FC`` record for *tipo*, or the
    named path does not resolve to an existing file. Plain string slicing
    cannot itself raise, so no malformed-content exception handling is
    needed beyond the ``OSError`` guard on the read itself.
    """
    try:
        text = dadger.read_text(encoding="latin-1")
    except OSError:
        return None
    for line in text.splitlines():
        if line[:4] != "FC  " or line[4:10].strip().upper() != tipo:
            continue
        caminho = line[14:214].strip()
        if not caminho:
            continue
        # `.resolve()` collapses any `..` in `caminho` (e.g. `../../cortesh.dat`)
        # so the returned path is a plain, normalized filesystem path rather
        # than one carrying the FC record's own relative-path spelling.
        candidate = (deck_dir / caminho).resolve()
        if candidate.is_file():
            return candidate
    return None


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

    def find(prefix: str, required: bool, *, exclude: str | None = None) -> Path | None:
        def is_candidate(name: str) -> bool:
            lname = name.lower()
            return lname.startswith(prefix) and (
                exclude is None or not lname.startswith(exclude)
            )

        for name in names:
            if is_candidate(name):
                path = src / name
                if path.is_file():
                    return path
        matches = sorted(p for p in src.glob(f"{prefix}*") if is_candidate(p.name))
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
    polinjus = find("polinjus", required=False)
    assert dadger is not None and vazoes is not None and hidr is not None
    # cortesh/cortes (ticket-007): prefer the deck's own FC record over the
    # glob idiom (its caminho may point outside `src`, e.g. `../../cortesh.dat`,
    # which the glob fallback below could never find); the `cortes-` prefix is
    # checked before the broader `cortes` prefix so a single-stage partition
    # export (`cortes-<estagio>.dat`) always wins over the consolidated
    # `cortes.dat` archive when both are present, matching `fcf/cortes.py`'s
    # own trailer-based shape detection. `exclude="cortesh"` keeps the broader
    # `cortes` glob from mistaking the header file for the record file.
    cortesh = _resolve_fc_record_path(dadger, src, tipo=_FC_TIPO_CORTESH) or find(
        "cortesh", required=False
    )
    cortes = (
        _resolve_fc_record_path(dadger, src, tipo=_FC_TIPO_CORTES)
        or find("cortes-", required=False)
        or find("cortes", required=False, exclude="cortesh")
    )
    return DecompFiles(
        revision=revision,
        dadger=dadger,
        vazoes=vazoes,
        hidr=hidr,
        dadgnl=dadgnl,
        renovaveis=renovaveis,
        polinjus=polinjus,
        libs_restricao_eletrica=constraint_registers.resolve_libs_electrical_path(src),
        cortesh=cortesh,
        cortes=cortes,
    )


def _rejoin_thermal_cost(thermal_bounds: pa.Table, cost_table: pa.Table) -> pa.Table:
    """Fold *cost_table* (``thermal.py``'s ``cost_per_mwh`` side-table) onto
    *thermal_bounds* (the accumulator's resolved ``THERMAL_BOUNDS_SCHEMA``
    table), restoring the ``cost_per_mwh`` column ``convert_thermal_bounds``
    carries alongside rather than through its bound contributions.

    A full outer merge on ``(thermal_id, stage_id, block_id)`` — not a
    left-join keyed off *thermal_bounds* — because a ``(thermal, stage)``
    whose generation bound resolved entirely to per-block contributions
    (every block non-uniform, so ``resolve()`` never materializes a
    ``block_id = None`` row for it, per the module's replace-vs-intersect
    discipline) still needs a row to carry that stage's cost: a strict left
    join would have nothing in *thermal_bounds* to attach it to and silently
    drop the value. The outer merge instead adds a ``block_id = None`` row
    with every bound column ``null`` for exactly that case. Per-block rows
    always get ``null`` cost, since the cost table never carries a
    non-``None`` ``block_id``.
    """
    key = ["thermal_id", "stage_id", "block_id"]
    bounds_df = thermal_bounds.to_pandas()
    cost_df = cost_table.to_pandas()
    for frame in (bounds_df, cost_df):
        frame["block_id"] = frame["block_id"].astype("Int64")
    merged = bounds_df.merge(cost_df, on=key, how="outer")

    schema = pa.schema(
        [
            *bounds_accumulator.THERMAL_BOUNDS_SCHEMA,
            pa.field("cost_per_mwh", pa.float64(), nullable=True),
        ]
    )
    return pa.table(
        {field.name: pa.array(merged[field.name], type=field.type) for field in schema},
        schema=schema,
    )


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


def _topology_relink_diagnostic(
    effective: cadastro_conv.EffectiveCadastro,
    id_map: DecompIdMap,
) -> dx.Diagnostic | None:
    """The ``cadastro-topology-relinked`` INFO diagnostic (ticket-014, OQ-4),
    or ``None`` when no operated plant's effective downstream link/inflow
    gauge differs from the base registry (the byte-identical no-override
    case).

    Lists every operated plant whose stage-0 effective downstream link
    (``AC NUMJUS``) or inflow gauge (``AC NUMPOS``) differs from the base
    ``hidr`` value — the direct, human-readable relink; rv3's own 5
    ``AC NUMJUS`` rows all take this form (34/43 -> 45, 127/181 -> 129,
    117 -> 108).

    As a defensive backstop — the "precise relink check" the ticket calls
    for — also walks the incremental-inflow cascade
    (:func:`~cobre_bridge.decomp.scenarios._incremental_context`) under
    both *effective* and a base-only cadastro (same registry, no override
    maps) and compares the resolved ``parents`` maps: an ``AC NUMJUS`` on a
    non-operated intermediate can change which operated plant receives an
    upstream plant's water without any operated plant's own link differing,
    so the direct per-code diff above would miss it. No rv3 row exercises
    that case, so it is not itemized by name here, but it is never silently
    dropped either — a tracked-gap warning fires instead of a
    diagnostic-less, byte-different conversion.
    """
    base = cadastro_conv.EffectiveCadastro(
        base=effective.base, n_stages=effective.n_stages, stage_varying={}
    )
    _, base_parents = scenarios_conv._incremental_context(base, id_map)
    _, effective_parents = scenarios_conv._incremental_context(effective, id_map)

    rows: list[list[object]] = []
    affected_codes: set[int] = set()
    for code in id_map.hydro_codes:
        name = str(effective.base.loc[code, "nome_usina"]).strip()
        base_downstream = int(effective.base.loc[code, "codigo_usina_jusante"])
        eff_downstream = effective.downstream_plant(code, 0)
        if eff_downstream != base_downstream:
            rows.append([name, code, "downstream", base_downstream, eff_downstream])
            affected_codes.add(code)
        base_gauge = int(effective.base.loc[code, "posto"])
        eff_gauge = effective.inflow_gauge(code, 0)
        if eff_gauge != base_gauge:
            rows.append([name, code, "gauge", base_gauge, eff_gauge])
            affected_codes.add(code)

    if not rows:
        if base_parents != effective_parents:
            _LOG.warning(
                "the effective and base incremental-inflow cascades differ, "
                "but no operated plant's own downstream link or inflow "
                "gauge changed directly; likely an AC NUMJUS relink on a "
                "non-operated intermediate -- not itemized by the "
                "cadastro-topology-relinked diagnostic"
            )
        return None

    return dx.Diagnostic(
        code="cadastro-topology-relinked",
        severity=dx.Severity.INFO,
        category="Cadastro overrides",
        title=f"Topology/gauge relinked for {len(affected_codes)} plant(s)",
        summary=(
            f"{len(affected_codes)} operated plant(s) carry an AC NUMJUS/"
            "NUMPOS override that changes their effective downstream link "
            "or inflow gauge from the base registry"
        ),
        table=dx.DiagnosticTable(
            columns=["Plant", "Code", "Kind", "Base", "Effective"],
            rows=rows,
            justify=["left", "right", "left", "right", "right"],
        ),
    )


def _diversion_channels(
    hydro_bounds: pa.Table,
    id_map: DecompIdMap,
    effective: cadastro_conv.EffectiveCadastro,
) -> tuple[dict[int, dict], list[int]]:
    """Cobre ``diversion`` channels for hydros carrying a positive diversion floor.

    A single-term QDES limit lowers to a ``min_diversion_m3s`` row; cobre couples
    a positive floor to a declared diversion channel (without one, diversion is
    pinned ``[0, 0]`` and any positive floor is infeasible). The source deck
    already declares the channel — read into ``effective.diversions`` — so this
    returns, keyed by cobre hydro id, the ``{downstream_id, max_flow_m3s}`` entry
    for every floored hydro whose channel resolves (a numeric limit and a
    downstream plant that survives into the case). Its second element lists the
    cobre ids of any floored hydro whose channel does NOT resolve — a genuine
    gap the caller surfaces rather than emitting an invalid case silently.

    Only floored hydros get a channel: an unconstrained diverter keeps
    ``diversion = null`` (its diversion stays pinned ``[0, 0]``, unchanged), so
    no plant's LP feasible region moves except the ones the floor already binds.
    """
    if "min_diversion_m3s" not in hydro_bounds.column_names:
        return {}, []
    columns = hydro_bounds.select(["hydro_id", "min_diversion_m3s"]).to_pydict()
    floored_ids = {
        int(hydro_id)
        for hydro_id, floor in zip(
            columns["hydro_id"], columns["min_diversion_m3s"], strict=True
        )
        if floor is not None and floor > 0.0
    }
    if not floored_ids:
        return {}, []
    code_by_id = {id_map.hydro_id(code): code for code in id_map.hydro_codes}
    operated_codes = set(id_map.hydro_codes)
    channels: dict[int, dict] = {}
    unresolved: list[int] = []
    for hydro_id in sorted(floored_ids):
        code = code_by_id.get(hydro_id)
        channel = None
        if code is not None:
            for stage in range(effective.n_stages):
                channel = effective.diversion(code, stage)
                if channel is not None:
                    break
        if (
            channel is None
            or channel.limit is None
            or channel.downstream not in operated_codes
        ):
            unresolved.append(hydro_id)
            continue
        channels[hydro_id] = {
            "downstream_id": id_map.hydro_id(channel.downstream),
            "max_flow_m3s": float(channel.limit),
        }
    return channels, unresolved


def _base_diversion_channels(
    effective: cadastro_conv.EffectiveCadastro,
    id_map: DecompIdMap,
    hydro_capacities: Mapping[int, single_term_bounds.HydroCapacities],
    already_bounded_ids: set[int],
    stage_ids: Sequence[int],
) -> tuple[dict[int, dict], list[bounds_accumulator.BoundContribution]]:
    """Diversion channels + ``max_diversion`` bounds for BASE ``desvio`` diverters.

    A base ``desvio`` (the ``hidr`` field) with neither an ``AC DESVIO`` limit
    nor a QDES single-term bound leaves cobre's diversion column pinned
    ``[0, 0]`` (``max_diversion_m3s`` defaults to 0), stranding any downstream
    plant fed *solely* by that diversion -- e.g. MOXOTO's spill-side split to
    P.AFONSO 4, which otherwise receives no water and generates 0 MW. This
    declares the channel and opens the diversion column up to the receiving
    plant's turbine capacity (the flow it can actually pass to generation), so
    the diverted water reaches the downstream plant.

    Skips any diverter whose diversion axis is already bounded
    (``already_bounded_ids`` -- a QDES/``AC DESVIO`` floor already handled by
    :func:`_diversion_channels`), so this only *adds* the previously-unmodelled
    base channels and never overrides an explicit source limit. Returns the
    ``{downstream_id, max_flow_m3s}`` channel per source cobre id and the
    per-stage ``diversion`` bound contributions to fold into the accumulator.
    """
    operated = set(id_map.hydro_codes)
    channels: dict[int, dict] = {}
    contribs: list[bounds_accumulator.BoundContribution] = []
    for code in sorted(id_map.hydro_codes):
        if not effective.has_diversion(code):
            continue
        source_id = id_map.hydro_id(code)
        if source_id in already_bounded_ids:
            continue
        channel = None
        for stage in range(effective.n_stages):
            channel = effective.diversion(code, stage)
            if channel is not None:
                break
        if channel is None or channel.downstream not in operated:
            continue
        downstream_id = id_map.hydro_id(channel.downstream)
        # A base `desvio` carries no explicit flow limit; cap the channel at the
        # receiving plant's turbine capacity (what it can turn into generation).
        cap = channel.limit
        if cap is None:
            cap = hydro_capacities[downstream_id].max_turbined_m3s
        if cap is None or cap <= 0.0:
            continue
        cap = float(cap)
        channels[source_id] = {"downstream_id": downstream_id, "max_flow_m3s": cap}
        contribs.extend(
            bounds_accumulator.BoundContribution(
                family="hydro",
                entity_id=source_id,
                stage_id=stage_id,
                block_id=None,
                axis="diversion",
                lower=0.0,
                upper=cap,
                contributor="base-desvio",
            )
            for stage_id in stage_ids
        )
    return channels, contribs


def _libs_electrical_census_diagnostic(
    result: libs_electrical_emit.LibsElectricalResult,
) -> dx.Diagnostic:
    """The ``decomp-libs-electrical-converted`` INFO diagnostic (ticket-013):
    the authoritative per-restriction census for the deck's LIBs-era
    long-form electrical file -- how many restrictions converted to a cobre
    generic constraint, and how many were dropped, broken down by
    :class:`~cobre_bridge.decomp.libs_electrical_emit.LibsElectricalResult`'s
    four drop reasons (``inactive``/``unrecognized-token``/
    ``unresolved-bucket-bc``/``unresolved-bucket-a``, each already carrying
    its own per-restriction WARNING/INFO diagnostic).

    This supersedes the flat ``decomp-libs-electrical-present`` warning for
    the converted subset (:func:`_convert_decomp_case_impl` suppresses that
    warning once a model converts) -- this is the one place a caller reads
    how much of the entry actually converted, so no "not converted" claim
    survives over it.
    """
    n_converted = len(result.converted_codes)
    n_dropped = sum(len(codes) for codes in result.deferred.values())
    rows: list[list[object]] = [["Converted", n_converted]]
    rows.extend(
        [reason, len(codes)] for reason, codes in sorted(result.deferred.items())
    )
    return dx.Diagnostic(
        code="decomp-libs-electrical-converted",
        severity=dx.Severity.INFO,
        category="Special constraints",
        title=(
            f"LIBs electrical constraints converted "
            f"({n_converted}/{n_converted + n_dropped})"
        ),
        summary=(
            f"{n_converted} of {n_converted + n_dropped} LIBs-era long-form "
            "electrical restriction(s) converted to a cobre generic "
            "constraint; the rest were dropped (see table for the "
            "per-reason breakdown)."
        ),
        table=dx.DiagnosticTable(
            columns=["Outcome", "Count"],
            rows=rows,
            justify=["left", "right"],
        ),
    )


def convert_decomp_case(
    src: Path,
    dst: Path,
    *,
    force: bool = False,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> ConversionReport:
    """Convert one deck revision into a Cobre case directory.

    Mirrors the NEWAVE twin ``convert_newave_case``'s return contract: wraps
    the conversion in a top-level :func:`cobre_bridge.diagnostics.collect`
    sink and a package-logger ``_WarningCollector`` so every structured
    ``dx.emit`` finding *and* every residual ``logger.warning`` string is
    captured, then returns them as one de-duplicated
    :class:`~cobre_bridge.pipeline.ConversionReport`.

    Parameters
    ----------
    dry_run:
        When ``True``, run the full in-memory conversion but write nothing to
        *dst* (no files, no subdirectories). The would-write paths are still
        recorded in :attr:`~cobre_bridge.pipeline.ConversionReport.would_write_paths`.

    Returns
    -------
    ConversionReport
        Summary of what was converted: the five entity/stage counts, a
        ``diagnostics`` list of every structured finding, and a ``warnings``
        list of the WARNING-severity summaries.

    Raises
    ------
    ValueError
        If the post-emission self-checks (cobre 0.13 rules 43/41/45/38/36 and
        the ``block_id``-range rule; see :mod:`cobre_bridge.emission_checks`)
        find an ``ERROR``-severity violation in the converted artifacts. An
        ``INFO`` finding (e.g. rule 43's "not applicable" report, emitted when
        no hydro-bounds capacity column is populated) never raises. No report
        is returned on this path.
    FileExistsError
        If *dst* already exists, is non-empty, and ``force`` is not set.
    """
    # This refusal must run before the clearing try/except below: it is a
    # "do not touch dst" guard, not a mid-write failure, so it must never
    # reach ``_clear_dst_contents`` and delete a pre-existing, unrelated case
    # (mirrors the source model's own NEWAVE-twin ordering, which performs
    # this check ahead of its own write/clear try block).
    if dst.exists() and any(dst.iterdir()) and not force:
        raise FileExistsError(f"{dst} already contains files; pass force to overwrite")

    collector = _WarningCollector()
    pkg_logger = logging.getLogger("cobre_bridge")
    pkg_logger.addHandler(collector)
    try:
        # Structured diagnostics emitted by converters (and the inner
        # emission-self-check re-emit loop below) land in ``collected``; any
        # remaining ``logger.warning`` strings (the deferral warning, the
        # topology non-itemized warning) are picked up by ``collector`` and
        # bridged below, so every warning still surfaces.
        with dx.collect() as collected:
            report = _convert_decomp_case_impl(
                src, dst, force=force, on_phase=on_phase, dry_run=dry_run
            )
    except BaseException:
        # The write phase is a sequence of independent file writes with no
        # rollback, so a failure partway through can leave a subset of the
        # output files behind. Remove the known pipeline outputs so a
        # half-written case is never mistaken for a complete one and a plain
        # (no --force) re-run is not refused as "destination not empty".
        # Skipped under a dry run: nothing was written, and dst may be a
        # pre-existing populated directory the user never asked to clear.
        if not dry_run:
            _clear_dst_contents(dst)
        raise
    finally:
        pkg_logger.removeHandler(collector)
    report.diagnostics = _finalize_diagnostics(collected, collector.messages)
    # Backward-compatible flat WARNING strings derived from the diagnostics.
    report.warnings = [
        d.summary for d in report.diagnostics if d.severity is dx.Severity.WARNING
    ]
    return report


def _convert_decomp_case_impl(
    src: Path,
    dst: Path,
    *,
    force: bool = False,
    on_phase: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> ConversionReport:
    """Run the DECOMP conversion pipeline (warning capture handled by the wrapper).

    ``on_phase`` (when given) is called once at each boundary in
    :data:`DECOMP_CONVERSION_PHASE_LABELS`, so the CLI can advance a progress bar
    without the pipeline knowing anything about rendering.

    When ``dry_run`` is ``True``, the build phases run exactly as for a real run,
    but the write phase creates no directories and writes no files; the paths that
    would have been written are still accumulated and recorded on the report.
    """
    step = on_phase if on_phase is not None else (lambda _label: None)
    step("Discovering deck")
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
    # ticket-007: Itaipu's split-plant topology (ticket-006) needs a
    # synthesized SE<->IV line and, only when the deck carries the data,
    # the ANDE load on the IV bus -- both driven by this single
    # Itaipu(66)-operated check, computed once and reused by both wiring
    # sites below.
    itaipu_operated = hydro_conv._ITAIPU_CODE in id_map.hydro_codes
    calendar = temporal_conv.operative_calendar_from_dadger(dadger)
    # epic-02/epic-03 (cadastro overrides): the per-stage-effective view of
    # the registry, folding in any temporal AC VOLMIN/VOLMAX/VAZMIN override —
    # read once and threaded to every consumer below (initial storage, the
    # entity reservoir envelope, the per-stage storage and minimum-outflow
    # bounds) so they all agree on the same effective values.
    effective, cadastro_report = cadastro_conv.build_effective_cadastro(
        dadger, hidr, calendar
    )
    if cadastro_report.applied or cadastro_report.out_of_horizon:
        applied_desc = ", ".join(
            f"{count} {param}"
            for param, count in sorted(cadastro_report.applied.items())
        )
        summary = f"cadastro overrides applied: {applied_desc or 'none'}"
        if cadastro_report.out_of_horizon:
            summary += (
                f"; {len(cadastro_report.out_of_horizon)} override(s) fall "
                "outside the study horizon and were not applied"
            )
        dx.emit(
            dx.Diagnostic(
                code="cadastro-overrides-applied",
                severity=dx.Severity.INFO,
                category="Cadastro overrides",
                title="Cadastro overrides applied",
                summary=summary,
            ),
            logger=_LOG,
        )
    # ticket-014 (E5, OQ-4): the topology/gauge relink is its own diagnostic
    # (not folded into the summary above) so an automated consumer can key
    # on "cadastro-topology-relinked" specifically — the resolved
    # downstream/gauge links, not just an override count.
    relink_diagnostic = _topology_relink_diagnostic(effective, id_map)
    if relink_diagnostic is not None:
        dx.emit(relink_diagnostic, logger=_LOG)
    start_date = calendar[0].start_date
    fan_probabilities = scenarios_conv.terminal_fan_probabilities(vazoes, calendar)

    tx = float(dadger.tx.taxa) / 100.0

    if not dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    # Every output path is routed through ``_write_json`` / ``_write_parquet`` so
    # the would-write listing and the dry-run gate live in exactly one place each,
    # rather than guarding ~24 individual write sites. Local closures (not module-
    # level helpers) so each carries this call's own *dry_run* and *would_write*
    # without threading them through every call site.
    would_write: list[Path] = []

    def _write_json(path: Path, data: dict) -> None:
        would_write.append(path)
        if dry_run:
            _LOG.debug("would write %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")

    def _write_parquet(path: Path, table: pa.Table) -> None:
        would_write.append(path)
        if dry_run:
            _LOG.debug("would write %s", path)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # TRACKED COBRE-GAP WORKAROUND (C3): the solver's parquet reader is
        # built without snappy; zstd until the compression contract is settled.
        pq.write_table(table, path, compression="zstd")

    step("Converting entities")
    _write_json(
        dst / "config.json",
        config_conv.convert_config(dadger),
    )
    stages_dict = temporal_conv.convert_stages(
        calendar, annual_discount_rate=tx, fan_probabilities=fan_probabilities
    )
    _write_json(dst / "stages.json", stages_dict)

    # FPHA-anchor fidelity: the source model fits each plant's hydro production
    # function around its initial reservoir volume. FPHA-eligible reservoirs get
    # cobre's computed-FPHA model (geometry + tailrace families, fit over a
    # ±window around the initial volume); the rest keep constant productivity,
    # whose ρ_eq is likewise anchored at the initial volume (not the full-range
    # mean) — see hydro._equivalent_productivity_mw_per_m3s and decomp/fpha.py.
    initial_volumes = hydro_conv._operated_initial_volumes(dadger, effective)
    fpha_codes = fpha_conv.fpha_eligible_codes(effective, id_map)
    fpha_configs: dict[int, dict] = {}
    reference_volumes: dict[int, dict] = {}
    for code in fpha_codes:
        vmin, vmax = cadastro_conv.effective_storage_range(effective, code, 0)
        v_init = initial_volumes.get(code, vmin)
        fpha_configs[code] = {
            "source": "computed",
            "fitting_window": fpha_conv.fitting_window(v_init, vmin, vmax),
        }
        reference_volumes[code] = {"volume_hm3": v_init}

    # penalties.json takes the full per-plant ρ_eq list (system ρ_avg/ρ_max);
    # the parquet omits FPHA plants (cobre computes their ρ_eq, so a parquet row
    # would double-supply it).
    productivity = hydro_conv.convert_energy_productivity(
        effective, id_map, initial_volumes
    )
    productivity_parquet = hydro_conv.convert_energy_productivity(
        effective, id_map, initial_volumes, exclude_codes=fpha_codes
    )
    deficit_costs = network_conv._bus_deficit_costs(dadger)
    deficit_cost = max(deficit_costs.values()) if deficit_costs else 0.0
    _write_json(
        dst / "penalties.json",
        config_conv.convert_penalties(
            deficit_cost,
            productivity["equivalent_productivity_mw_per_m3s"].to_pylist(),
        ),
    )
    # TRACKED COBRE-GAP WORKAROUND (VI water travel time): the converter
    # (decomp/travel_time.py) is complete and `cobre validate` accepts the
    # emitted travel_time_hours + past_defluences, but cobre's transit-bucket
    # cut generation currently produces an invalid cut that blows the lower
    # bound past the upper bound (a large negative gap that cobre clamps to
    # zero and mistakes for convergence, stopping on a corrupted policy). So VI
    # is DEFERRED alongside GNL and the boundary FCF: detected for the deferral
    # warning below, but not emitted. Restore by wiring convert_travel_time back
    # into hydros.json + initial_conditions.past_defluences once the cobre bug
    # is fixed. Removal condition tracked in
    # ~/git/cobre/plans/conversion-found-improvements.md (C-travel-time).
    has_travel_time = bool(travel_time_conv.read_travel_times(dadger))
    # Built here but written below (after GNL): an anticipated GNL fleet extends
    # it with the left/right temporal-boundary arrays, and those need the thermal
    # ids assigned when thermals.json is built.
    initial_conditions_doc: dict = {
        "storage": hydro_conv.convert_initial_storage(dadger, hidr, id_map, effective),
        "filling_storage": [],
    }

    system = dst / "system"
    buses_doc = network_conv.convert_buses(dadger, id_map, start_date)
    _write_json(system / "buses.json", buses_doc)
    hydros_dict = hydro_conv.convert_hydros(
        dadger, hidr, id_map, start_date, effective, fpha_codes=fpha_codes
    )
    # hydros.json is written after the bound tables are resolved (below): a
    # plant carrying a positive QDES diversion floor (min_diversion_m3s > 0)
    # must also declare its diversion channel, or cobre rejects the floor as
    # infeasible against the channel-less [0, 0] pin. That coupling can only be
    # applied once the resolved hydro_bounds are in hand.
    lines_doc, line_bounds = network_conv.convert_lines(
        dadger, id_map, calendar, start_date
    )
    if itaipu_operated:
        # When Itaipu is operated (ticket-006 relocates its 50 Hz group to the
        # IV bus), the IV bus needs a connection to SE. append_iv_se_line
        # synthesizes an SE<->IV line only when the deck's own IA register does
        # NOT already wire IV<->SE; when it does (as on most Itaipu decks), the
        # helper reuses that line -- its real operational capacity -- and adding
        # a second one would make cobre reject the duplicate.
        itaipu_submercado = int(hidr.loc[hydro_conv._ITAIPU_CODE, "submercado"])
        lines_doc, line_bounds = network_conv.append_iv_se_line(
            lines_doc,
            line_bounds,
            calendar,
            start_date,
            source_bus_id=id_map.transhipment_bus_id,
            target_bus_id=id_map.bus_id(itaipu_submercado),
            capacity_mw=network_conv._itaipu_50hz_capacity_mw(dadger),
        )
    _write_json(system / "lines.json", lines_doc)
    _write_json(
        system / "pumping_stations.json",
        network_conv.convert_pumping_stations(dadger, id_map, start_date),
    )
    thermals_dict = thermal_conv.convert_thermals(dadger, id_map, calendar, start_date)
    # GNL (fuel-constrained) thermals are declared in dadgnl, absent from CT, so
    # the thermal converter never sees them. Read the commitment model and emit
    # the created thermals plus their anticipated ring: the mandatory left
    # boundary (past_anticipated_commitments) and any post-horizon deliveries as
    # the right boundary (future_anticipated_deliveries + post_study_stages.json).
    gnl_model = (
        anticipated_conv.read_gnl_model(Dadgnl.read(str(files.dadgnl)))
        if files.dadgnl is not None
        else None
    )
    if gnl_model is not None:
        gnl = anticipated_conv.convert_gnl(
            gnl_model,
            first_thermal_id=max(t["id"] for t in thermals_dict["thermals"]) + 1,
            bus_id_of=id_map.bus_id,
            stages=stages_dict["stages"],
        )
        thermals_dict["thermals"].extend(gnl.thermals)
        thermals_dict["thermals"].sort(key=lambda t: t["id"])
        if gnl.past_anticipated_commitments:
            initial_conditions_doc["past_anticipated_commitments"] = (
                gnl.past_anticipated_commitments
            )
        if gnl.future_anticipated_deliveries:
            initial_conditions_doc["future_anticipated_deliveries"] = (
                gnl.future_anticipated_deliveries
            )
        if gnl.post_study_stages is not None:
            _write_json(dst / "post_study_stages.json", gnl.post_study_stages)
        _LOG.info(
            "emitted %d GNL anticipated thermal(s) from dadgnl; %d future "
            "anticipated deliver(y/ies)",
            len(gnl.thermals),
            len(gnl.future_anticipated_deliveries),
        )
    _write_json(system / "thermals.json", thermals_dict)
    _write_json(dst / "initial_conditions.json", initial_conditions_doc)
    _write_json(
        system / "hydro_production_models.json",
        hydro_conv.convert_production_models(id_map, fpha_configs, reference_volumes),
    )
    _write_parquet(system / "hydro_energy_productivity.parquet", productivity_parquet)
    # FPHA geometry + tailrace inputs cobre fits the production function from.
    _write_parquet(
        system / "hydro_geometry.parquet",
        fpha_conv.convert_hydro_geometry(effective, id_map),
    )
    tailrace_table = fpha_conv.convert_tailrace_curves(
        fpha_conv.read_polinjus(files.polinjus) if files.polinjus is not None else None,
        id_map,
    )
    if tailrace_table is not None:
        _write_parquet(system / "tailrace_curves.parquet", tailrace_table)
    ncs_registry = ncs_conv.convert_non_controllable_sources(
        dadger, id_map, calendar, start_date, renovaveis
    )
    _write_json(system / "non_controllable_sources.json", ncs_registry)

    step("Converting scenarios")
    scenarios = dst / "scenarios"
    _write_parquet(
        scenarios / "inflow_seasonal_stats.parquet",
        scenarios_conv.convert_inflow_stats_identity(id_map, calendar),
    )
    _write_parquet(
        scenarios / "external_inflow_scenarios.parquet",
        scenarios_conv.convert_external_inflows(vazoes, effective, id_map, calendar),
    )
    # ticket-007: the IV bus's carga_ande load is gated on the dadger RI
    # register being present -- independent of (and additional to) the
    # unconditional Itaipu-operated line above. An Itaipu deck with no RI
    # register (read_carga_ande returns {}) converts gracefully: the line
    # exists, but the IV bus stays at zero load.
    extra_bus_loads: dict[tuple[int, int], list[float]] | None = None
    if itaipu_operated:
        carga_ande = libs_electrical_conv.read_carga_ande(dadger, calendar)
        if carga_ande:
            iv_bus = id_map.transhipment_bus_id
            extra_bus_loads = {
                (iv_bus, stage_index): values
                for stage_index, values in carga_ande.items()
            }
    load_stats = load_conv.convert_load_stats(
        dadger, id_map, calendar, extra_bus_loads=extra_bus_loads
    )
    _write_parquet(scenarios / "load_seasonal_stats.parquet", load_stats)
    _write_json(
        scenarios / "load_factors.json",
        load_conv.convert_load_factors(
            dadger, id_map, calendar, extra_bus_loads=extra_bus_loads
        ),
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
            # cobre's clean break (0.14) removed the legacy ``value`` alias for
            # the NCS availability column; the sole accepted spelling is now
            # ``availability_factor`` (external inflow/load keep value_m3s/value_mw).
            value_out="availability_factor",
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
    # epic-07 (special-constraints pipeline wiring, ticket-023): every
    # per-entity bound — the legacy RQ/UH minimum-outflow, the per-stage
    # storage envelope, the CT thermal generation bounds, and the RE/RHQ/RHV
    # single-term special-constraint bounds — is collected as
    # bounds_accumulator.BoundContribution objects and resolved through
    # exactly ONE resolve() + build_bound_tables() pass, so a new special
    # constraint colliding with a legacy bound on the same (entity, stage,
    # block) cell correctly intersects instead of producing the
    # two-rows-same-column parquet cobre rejects. read_constraints/
    # pumping_station_id_map are read once here and reused by ticket-023b
    # (generic constraints).
    step("Resolving bounds")
    census = constraint_registers.read_constraints(dadger)
    pumping_ids = network_conv.pumping_station_id_map(dadger)
    thermal_generation_contribs, thermal_cost_table = (
        thermal_conv.convert_thermal_bounds(dadger, id_map, calendar)
    )
    # ticket-023c/boundary-review-fix-1: the RE `FU` single-hydro-generation
    # and the RHQ `QTUR`/turbined bound producers each clamp their emitted
    # upper to the plant's own declared max_generation_mw/max_turbined_m3s,
    # so a looser source-declared ceiling on either axis (a real
    # cross-source mismatch — see BELO MONTE's RE ceiling on both real
    # decks) never trips cobre rule 43
    # (emission_checks.check_hydro_bounds_no_raising, self-checked below).
    # Read the same way that check reads hydros_dict: hydro id -> its
    # generation.max_generation_mw/max_turbined_m3s.
    hydro_capacities: dict[int, single_term_bounds.HydroCapacities] = {
        hydro["id"]: single_term_bounds.HydroCapacities(
            max_generation_mw=hydro["generation"]["max_generation_mw"],
            max_turbined_m3s=hydro["generation"]["max_turbined_m3s"],
        )
        for hydro in hydros_dict["hydros"]
    }
    contribs = [
        *bounds_conv.convert_hydro_bounds(dadger, id_map, calendar, effective),
        *bounds_conv.convert_storage_bounds(effective, id_map, calendar),
        *thermal_generation_contribs,
        *single_term_bounds.single_term_bound_contributions(
            census, id_map, pumping_ids, calendar, effective, hydro_capacities
        ),
    ]
    # Base `desvio` diverters carry no QDES flow bound, so cobre pins their
    # diversion column to [0, 0] and strands any plant fed solely by the
    # diversion (e.g. MOXOTO -> P.AFONSO 4, which otherwise generates 0 MW).
    # Give the still-unbounded ones a max_diversion bound + a routed channel.
    _diversion_bounded_ids = {
        contrib.entity_id
        for contrib in contribs
        if contrib.family == "hydro" and contrib.axis == "diversion"
    }
    base_diversion_channels, base_diversion_contribs = _base_diversion_channels(
        effective,
        id_map,
        hydro_capacities,
        _diversion_bounded_ids,
        [stage.index for stage in calendar],
    )
    contribs.extend(base_diversion_contribs)
    block_counts = {stage.index: len(stage.block_hours) for stage in calendar}
    bound_tables = bounds_accumulator.build_bound_tables(
        bounds_accumulator.resolve(contribs, block_counts)
    )
    _bound_sort_keys = [
        ("stage_id", "ascending"),
        ("block_id", "ascending"),
    ]
    hydro_bounds = bound_tables.hydro.sort_by(
        [("hydro_id", "ascending"), *_bound_sort_keys]
    )
    # Attach the diversion channel to every hydro that carries a positive
    # diversion floor, then write hydros.json (its write was deferred from the
    # system-file block above for exactly this). cobre couples the two: a
    # `min_diversion_m3s > 0` row is infeasible unless the hydro also declares a
    # `diversion` channel. The source deck already carries the channel (read
    # into `effective.diversions`); this is the only place both the resolved
    # floor and the channel are in hand.
    diversion_channels, diversion_floor_no_channel = _diversion_channels(
        hydro_bounds, id_map, effective
    )
    for hydro in hydros_dict["hydros"]:
        channel = diversion_channels.get(hydro["id"]) or base_diversion_channels.get(
            hydro["id"]
        )
        if channel is not None:
            hydro["diversion"] = channel
    if base_diversion_channels:
        dx.emit(
            dx.Diagnostic(
                code="decomp-base-diversion-modeled",
                severity=dx.Severity.INFO,
                category="Diversion",
                title="Base diversion channels modeled",
                summary=(
                    f"{len(base_diversion_channels)} hydro(s) "
                    f"({sorted(base_diversion_channels)}) carry a base diversion "
                    "channel with no explicit flow limit; each was given a "
                    "diversion channel capped at the receiving plant's turbine "
                    "capacity so the diverted water reaches the downstream plant "
                    "(otherwise that plant is stranded with no inflow)."
                ),
            ),
            logger=_LOG,
        )
    if diversion_floor_no_channel:
        dx.emit(
            dx.Diagnostic(
                code="decomp-diversion-floor-no-channel",
                severity=dx.Severity.WARNING,
                category="Diversion",
                title="Diversion floor without a resolvable channel",
                summary=(
                    f"{len(diversion_floor_no_channel)} hydro(s) "
                    f"({sorted(diversion_floor_no_channel)}) carry a positive "
                    "min_diversion_m3s floor but no resolvable diversion channel "
                    "(absent/limitless channel or an unmapped downstream plant); "
                    "cobre rejects the case until the channel is supplied."
                ),
            ),
            logger=_LOG,
        )
    _write_json(system / "hydros.json", hydros_dict)
    thermal_bounds_table = _rejoin_thermal_cost(
        bound_tables.thermal, thermal_cost_table
    ).sort_by([("thermal_id", "ascending"), *_bound_sort_keys])
    pumping_bounds_table = bound_tables.pumping.sort_by(
        [("pumping_station_id", "ascending"), *_bound_sort_keys]
    )
    # ticket-026/027: B8 per-group per-stage availability (installed × MP ×
    # FD, capped by the ρ_eq·q_max hydraulic ceiling) for every plant,
    # single-group and Itaipu's own two per-frequency groups (code 66) alike.
    # Sparse by construction — a value only lands here where it lowers that
    # group's own declared envelope — so cobre rule 45 (a group-bound max_*
    # may not exceed that group's declared max) is now reachable; its mirror
    # is wired into the self-check block below.
    step("Converting constraints")
    availability_values = hydro_conv.convert_hydro_group_availability(
        dadger, hidr, id_map, calendar, effective
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

    # Post-emission self-checks (cobre 0.13 rules 43, 41, 45, 38, 36, and the
    # block_id-range rule) — a courtesy mirror of cheap cobre invariants over
    # the in-memory artifacts, run before the constraint tables are written.
    # hydro_bounds now carries max_turbined_m3s/max_generation_mw whenever a
    # single-term special constraint (e.g. an RE FU generation ceiling) lowers
    # to one, so the rule-43 check below is genuinely reachable — it raises
    # when such a bound exceeds the entity's own declared capacity, and only
    # reports "not applicable" when no capacity column is populated at all.
    # See cobre_bridge.emission_checks for the rule scope.
    #
    # An ERROR-severity finding mirrors a cobre load invariant the converted
    # case would fail, so it fails the conversion outright (the raise below)
    # rather than surfacing only as a diagnostic on the returned report. The
    # local dx.collect() exists so this function can inspect severity; every
    # collected finding is then re-emitted via dx.emit() so the wrapper's
    # outer sink (and thus the returned ConversionReport) still records it,
    # and a standalone call with no sink still logs it.
    step("Writing outputs")
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
        emission_checks.BoundFamily(
            "Pumping", "pumping_station_id", pumping_bounds_table
        ),
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
    if pumping_bounds_table.num_rows:
        _write_parquet(constraints / "pumping_bounds.parquet", pumping_bounds_table)
    if group_bounds.num_rows:
        _write_parquet(constraints / "hydro_unit_group_bounds.parquet", group_bounds)
    if contracts:
        _write_json(system / "energy_contracts.json", energy_contracts_doc)
    if contract_bounds_table.num_rows:
        _write_parquet(constraints / "contract_bounds.parquet", contract_bounds_table)

    # epic-07 (special-constraints pipeline wiring, ticket-023b): emit every
    # RE/RHQ/RHV/RHE special constraint that did NOT lower to an entity bound
    # (`census.to_generic`, read once by ticket-023 above) as a Cobre generic
    # constraint, over one shared 0-based constraint-id allocator so ids
    # never collide across emitters regardless of which family produced
    # them. The three emitters run in this fixed order (E7); `big_m`,
    # `line_map`, and `hydro_to_ree` are shared inputs every emitter needs
    # but none of them reads the deck to build for itself.
    big_m = constraints_conv.big_m_penalty(deficit_cost)
    line_map = constraints_conv.build_fi_line_map(lines_doc["lines"])
    uh = dadger.uh(df=True)
    operated_uh = uh[uh["volume_inicial"].notna()]
    hydro_to_ree = {
        int(row["codigo_usina"]): int(row["codigo_ree"])
        for _, row in operated_uh.iterrows()
    }

    # ticket-013: read the deck's LIBs-era electrical-constraint file (the
    # indices.csv RESTRICAO-ELETRICA-ESPECIAL entry, resolved by
    # discover_decomp_files) exactly once here -- both the narrowed
    # detect_libs_electrical decision and the 4th generic-constraint emitter
    # link below thread this same `libs_electrical_model`, so neither can
    # disagree about whether the deck's long-form subset converted.
    # `read_libs_electrical` itself returns `None` for a short-form-only
    # (RE/RE-* or date-indexed) file -- that fallback stays on
    # detect_libs_electrical's flat warning (OQ-4).
    libs_electrical_model: libs_electrical_conv.LibsElectricalModel | None = None
    if files.libs_restricao_eletrica is not None:
        from idecomp.libs.restricoes import Restricoes

        restricoes = Restricoes.read(str(files.libs_restricao_eletrica))
        libs_electrical_model = libs_electrical_conv.read_libs_electrical(
            restricoes, calendar
        )

    generic_constraints: list[dict] = []
    generic_bound_tables: list[pa.Table] = []
    next_generic_id = 0

    re_generics = constraints_conv.emit_re_generics(
        census, id_map, line_map, big_m, calendar, next_generic_id
    )
    if re_generics is not None:
        generic_constraints.extend(re_generics.constraints)
        generic_bound_tables.append(re_generics.bounds)
        next_generic_id += len(re_generics.constraints)

    rhq_rhv_generics = constraints_conv.emit_rhq_rhv_generics(
        census, id_map, pumping_ids, effective, big_m, calendar, next_generic_id
    )
    if rhq_rhv_generics is not None:
        generic_constraints.extend(rhq_rhv_generics.constraints)
        generic_bound_tables.append(rhq_rhv_generics.bounds)
        next_generic_id += len(rhq_rhv_generics.constraints)

    rhe_generics = constraints_conv.emit_rhe_generics(
        census, id_map, effective, hydro_to_ree, calendar, next_generic_id
    )
    if rhe_generics.result is not None:
        generic_constraints.extend(rhe_generics.result.constraints)
        generic_bound_tables.append(rhe_generics.result.bounds)
        next_generic_id += len(rhe_generics.result.constraints)

    # ticket-013 (E9 wiring): the 4th generic-constraint emitter link -- every
    # LIBs-era long-form electrical restriction that resolved
    # (`libs_electrical_model`, read once above), spliced after RHE so the
    # unconditional write block below picks it up automatically, identical in
    # shape to the three links above. Built only when the deck actually
    # carries a long-form model, since none of these inputs are otherwise
    # needed.
    libs_electrical_result: libs_electrical_emit.LibsElectricalResult | None = None
    if libs_electrical_model is not None:
        context_factory = libs_electrical_conv.build_data_context(
            libs_electrical_model, dadger, id_map, calendar
        )
        a_h = libs_electrical_conv.build_available_power(
            availability_values, hidr, id_map, effective
        )
        ncs_id_by_pee_code = ncs_conv.build_pee_ncs_id_map(
            dadger, id_map, calendar, renovaveis
        )
        conjh_bus_by_code_group = {
            (id_map.hydro_codes[entry["id"]], group_index): group["bus_id"]
            for entry in hydros_dict["hydros"]
            for group_index, group in enumerate(entry["unit_groups"], start=1)
        }
        libs_electrical_result = libs_electrical_emit.emit_libs_electrical_generics(
            libs_electrical_model,
            id_map,
            context_factory,
            a_h,
            calendar,
            ncs_id_by_pee_code,
            conjh_bus_by_code_group,
            line_map,
            big_m,
            next_generic_id,
        )
        if libs_electrical_result.generic is not None:
            generic_constraints.extend(libs_electrical_result.generic.constraints)
            generic_bound_tables.append(libs_electrical_result.generic.bounds)
            next_generic_id += len(libs_electrical_result.generic.constraints)

    if generic_constraints:
        _write_json(
            constraints / "generic_constraints.json",
            {
                "$schema": _GENERIC_CONSTRAINTS_SCHEMA_URL,
                "constraints": generic_constraints,
            },
        )
        _write_parquet(
            constraints / "generic_constraint_bounds.parquet",
            pa.concat_tables(generic_bound_tables),
        )

    # E5 (ticket-017/018): every `@rho_acum_h{id}` sigil a surviving RHE
    # expression references must resolve, or cobre fails to load the case --
    # write generic_parameters.json whenever any RHE constraint survives,
    # never only when another generic family also happens to be present.
    if rhe_generics.rho_acum_overrides:
        would_write.append(
            write_scalar_parameters(
                dst,
                build_decomp_scalar_parameters(
                    [id_map.hydro_id(code) for code in id_map.hydro_codes],
                    rhe_generics.rho_acum_overrides,
                ),
                dry_run=dry_run,
            )
        )

    # E1: the FE/RHA/LIBs-electrical detection diagnostics each surface
    # once, through the structured sink, alongside the flat deferral warning
    # below (ticket-024 dedups the matching clause out of that warning).
    for detection in constraint_registers.detect_unreadable_electrical(files.dadger):
        dx.emit(detection, logger=_LOG)
    # ticket-013: the flat presence warning is narrowed to the subset
    # `libs_electrical_model` (read once above) did NOT convert -- a
    # short-form-only (RE/RE-*, date-indexed) file, or no long-form card at
    # all despite the indices.csv entry (OQ-4). Once the long-form subset
    # converts, the census INFO below is authoritative and this flat warning
    # would otherwise be a stale "not converted" claim over that subset.
    libs_detection = constraint_registers.detect_libs_electrical(src)
    if libs_detection is not None and libs_electrical_model is None:
        dx.emit(libs_detection, logger=_LOG)
    if libs_electrical_result is not None:
        dx.emit(_libs_electrical_census_diagnostic(libs_electrical_result), logger=_LOG)

    # Milestone-deferral warning. Boundary FCF is imported by default now (the
    # CLI runs the importer after this converter returns), reservoir evaporation
    # is converted (cobre >= 0.14's C11 fix), and windowed inflow inputs are a
    # source-model concept that does not apply to the external explicit tree the
    # DECOMP path emits — so none of those are deferred. Only VI water travel
    # time remains deferred (a tracked cobre-gap; see the note above where it is
    # detected), and only when the deck actually carries it.
    if has_travel_time:
        _LOG.warning("deferred at this milestone: water travel time (VI present)")
    _LOG.info(
        "converted %d buses, %d hydros, %d thermals, %d stages, terminal fan %d",
        id_map.n_buses,
        len(id_map.hydro_codes),
        len(id_map.thermal_codes),
        len(calendar),
        len(fan_probabilities),
    )

    return ConversionReport(
        hydro_count=len(hydros_dict["hydros"]),
        thermal_count=len(thermals_dict["thermals"]),
        bus_count=len(buses_doc["buses"]),
        line_count=len(lines_doc["lines"]),
        stage_count=len(calendar),
        would_write_paths=[str(p) for p in would_write],
    )
