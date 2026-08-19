"""Boundary FCF importer: reads the source model's cut files and authors a
cobre policy checkpoint.

:func:`import_boundary_fcf` is the epic's single entry point: it composes the
cut reader (``fcf/cortes.py``, epic 1), the terminal-manifest bootstrap
(``fcf/bootstrap.py``), the manifest-to-manifest mapper (``fcf/mapper.py``),
and the checkpoint writer (``fcf/writer.py``) in order, then patches the
converted case's ``config.json`` so cobre loads the authored ``boundary/``
checkpoint at its terminal stage. It is thin orchestration only — every
algorithm it calls already exists in one of the four modules above; this
module adds no new cut-mapping or checkpoint-authoring logic of its own.

The importer is a **post-conversion** step: it runs against an already
converted case directory (``convert_decomp_case``'s output), never inside the
conversion itself, because the bootstrap stage needs a real ``cobre run`` on
the converted case to read back its terminal state-vector layout.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from idecomp.decomp import Dadger, Dadgnl, Mlt, Vazoes
from inewave.newave import Cortesh

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.anticipated import read_gnl_model
from cobre_bridge.decomp.cadastro import build_effective_cadastro
from cobre_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)
from cobre_bridge.decomp.fcf.cortes import (
    read_cortes,
    summarize_cut_families,
)
from cobre_bridge.decomp.fcf.mapper import (
    GnlRingPlan,
    GnlThermalTarget,
    map_boundary_cuts,
)
from cobre_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)
from cobre_bridge.decomp.hydro import read_hidr
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.inflow_mlt import build_incremental_mlt, coupling_lag_means
from cobre_bridge.decomp.pipeline import DecompFiles, discover_decomp_files
from cobre_bridge.decomp.scenarios import convert_recent_observation_windows
from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from cobre_bridge.decomp.anticipated import GnlCommitmentModel
    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
    from cobre_bridge.decomp.fcf.mapper import MappingResult
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: (ticket-013 Requirement C.3 / Finding 3) a GNL deviation group whose
#: carried |Σ| is below this FRACTION of the panel's own max |Σ| is treated
#: as numerically vanished relative to the group that actually carries
#: weight — its relative spread `(max_p c_p - min_p c_p) / |Σ_p c_p|` would
#: otherwise inflate into noise (observed: a Σ≈-4e-05 group reported spread
#: 0.25 while the weight-carrying Σ=-4412 group's spread was only ~0.09; an
#: earlier fixed-magnitude floor of `1e-6` was ~40x too small to exclude
#: that 4e-05 group, since `4e-05 >= 1e-6` — an absolute floor cannot track
#: how "small" a Σ is without knowing the panel's own scale). Excluded from
#: the `boundary-fcf-gnl-anticipated-deviation` diagnostic's relative
#: `max_spread` HEADLINE only; its per-row table values are unaffected.
_GNL_DEVIATION_REL_FLOOR = 1e-3


def _gnl_targets_from(
    model: GnlCommitmentModel, thermals_doc: Mapping[str, object]
) -> GnlRingPlan | None:
    """Build the submercado -> GNL-thermal ring plan from the deck + case.

    Joins ``model.thermals`` (ascending by ``code``, read from ``dadgnl``'s
    ``tg`` registry by :func:`~cobre_bridge.decomp.anticipated.read_gnl_model`
    — reconciled with, never re-derived) onto the converted case's GNL
    thermal ids: ``thermals_doc["thermals"]`` entries carrying
    ``anticipated_config``, sorted ascending, are exactly
    ``convert_decomp_case``'s ``first_thermal_id + i`` assignment (ascending
    by code — see ``decomp/pipeline.py``/``decomp/anticipated.py::convert_gnl``).
    Zipping the two ascending-by-code sequences positionally reproduces that
    assignment exactly, without re-deriving it.

    A plant absent from ``model.nl_lag_months`` has no dispatch-anticipation
    lag to key its ring slot(s) on (no ``pi_gnl`` lag axis), so it is
    *skipped* — its ring stays at coefficient ``0.0`` via the mapper's own
    D3-like drop path — with a single INFO log line naming every skipped
    plant, never raised on. Returns ``None`` when every plant is skipped (no
    live target at all).

    Raises
    ------
    ValueError
        If the converted case's GNL-fleet count (``thermals_doc["thermals"]``
        entries carrying ``anticipated_config``) disagrees with the
        ``dadgnl`` registry's thermal count — a converter/reader
        inconsistency that would otherwise silently mis-zip ids onto the
        wrong codes.
    """
    thermals_entries = thermals_doc["thermals"]
    if not isinstance(thermals_entries, list):
        raise TypeError(
            f"thermals.json 'thermals' is {type(thermals_entries).__name__}, not a list"
        )
    gnl_ids = sorted(
        int(entry["id"])
        for entry in thermals_entries
        if isinstance(entry, dict) and "anticipated_config" in entry
    )
    codes = sorted(thermal.code for thermal in model.thermals)
    if len(gnl_ids) != len(codes):
        raise ValueError(
            f"thermals.json carries {len(gnl_ids)} GNL thermal(s) "
            "(anticipated_config), but the dadgnl registry declares "
            f"{len(codes)} thermal(s); the GNL fleet must match exactly"
        )
    id_of = dict(zip(codes, gnl_ids, strict=True))

    targets: list[GnlThermalTarget] = []
    skipped: list[str] = []
    for thermal in model.thermals:
        lag = model.nl_lag_months.get(thermal.code)
        if lag is None:
            skipped.append(thermal.name)
            continue
        targets.append(
            GnlThermalTarget(
                thermal_id=id_of[thermal.code],
                submercado=thermal.submarket_code,
                nl_lag=lag,
            )
        )

    if skipped:
        _LOG.info(
            "%d GNL plant(s) declare no nl dispatch-anticipation lag, so "
            "their AnticipatedThermalState ring slot(s) stay at coefficient "
            "0.0 (no pi_gnl lag axis to key on): %s",
            len(skipped),
            ", ".join(skipped),
        )

    return GnlRingPlan(tuple(targets)) if targets else None


def _post_horizon_start(case_dir: Path) -> int | None:
    """The earliest post-study stage start, as a ``YYYYMMDD`` int, or ``None``.

    Reads ``case_dir / "post_study_stages.json"`` (the
    ``decomp/anticipated.py::GnlEmission.post_study_stages`` payload,
    written by ``decomp/pipeline.py`` only when some GNL delivery lands
    post-horizon). Returns ``None`` when the file is absent or its
    ``stages`` list is empty — a case shape as legitimate as "no post-study
    horizon at all"; the mapper's ``GnlRingPlan.post_horizon_start=None``
    disables the covered-lane filter entirely for that case (ticket-013).

    A malformed ``start_date`` string is NOT swallowed here: the
    ``ValueError`` from ``int(...)`` propagates verbatim rather than
    silently disabling the filter — a corrupt horizon is a real problem,
    never a "no horizon" case.
    """
    path = case_dir / "post_study_stages.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    stages = doc.get("stages", [])
    if not stages:
        return None
    return min(int(stage["start_date"].replace("-", "")) for stage in stages)


def _final_stage_block_hours(case_dir: Path) -> list[float]:
    """The coupling (terminal) stage's per-block hours, in file order, from
    ``stages.json``.

    The boundary FCF attaches at the end of the last modelled stage, so the
    coupling period is the final ``stages.json`` stage. Returns its per-block
    ``hours`` values in file order — patamar order, matching the source
    model's own cut axis (a weekly deck's ~168 h split across blocks, or the
    coupling month's post-exclusion hours — e.g. 648 h for a 27-day April) —
    so :func:`~cobre_bridge.decomp.fcf.mapper.map_boundary_cuts`'s GNL branch
    can weight each ``pi_gnl`` patamar column by its own coupling block's
    hours (ticket-001), rather than the stage's total.
    :func:`_coupling_stage_hours` sums this same vector for the
    intercept/storage/inflow-lag terms.

    Raises
    ------
    ValueError
        If ``stages.json`` is missing, carries no stages, or the final stage
        has no positive total block-hours — a boundary import cannot scale
        the cut coefficients to a cost without the coupling stage's duration.
    """
    path = case_dir / "stages.json"
    if not path.is_file():
        raise ValueError(f"stages.json not found at {path}; cannot scale boundary FCF")
    with path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    stages = doc.get("stages", [])
    if not stages:
        raise ValueError(f"{path} carries no stages; cannot scale boundary FCF")
    hours = [float(block["hours"]) for block in stages[-1].get("blocks", [])]
    total = math.fsum(hours)
    if total <= 0.0:
        raise ValueError(
            f"{path} final stage has non-positive total block hours ({total}); "
            "cannot scale boundary FCF"
        )
    return hours


def _coupling_stage_hours(case_dir: Path) -> float:
    """The coupling (terminal) stage's total duration in hours, from ``stages.json``.

    The source model's FCF coefficients are a *per-hour* cost rate (see
    ``fcf/mapper.py``'s header): the source integrates that rate over its
    coupling period's actual hours to obtain the future-cost value, so the
    mapper needs those same hours (``cost_unit_hours``) to reproduce it in
    cobre's plain-$ objective. Using the actual stage hours (not a fixed 730-h
    month) reproduces the source's ``E(CF)`` to ~2-3 %, where a fixed month
    overshoots by ~15 % on a short coupling period.

    A thin ``math.fsum`` wrapper over :func:`_final_stage_block_hours` — see
    that function for the per-block vector and the validation the two share.

    Raises
    ------
    ValueError
        Propagated verbatim from :func:`_final_stage_block_hours`.
    """
    return math.fsum(_final_stage_block_hours(case_dir))


#: A ``CX`` register line: ``CX <complexo_code> <component_code>``. idecomp
#: exposes no typed accessor for ``CX``, so it is parsed directly.
_CX_LINE = re.compile(r"^CX\s+(\d+)\s+(\d+)")


def _read_complexo_map(dadger_path: Path) -> dict[int, list[int]]:
    """Parse the deck's ``CX`` register into ``{complexo_code: [component_codes]}``.

    The ``CX`` register (*acoplamento de usinas que representam complexos no
    NEWAVE com o DECOMP*) maps a NEWAVE **complexo** — an aggregated plant
    carrying a single set of future-cost coefficients in the boundary cuts — to
    the individual DECOMP plants that share it. The boundary cut header prices
    the complexo (which is absent from the DECOMP model), so
    :func:`~cobre_bridge.decomp.fcf.mapper.map_boundary_cuts` replicates its
    coefficients onto these components. Each ``CX <complexo> <component>`` line
    appends ``component`` to ``complexo``'s list; returns an empty map when the
    deck carries no ``CX`` register or the file is absent (the importer's own
    ``Dadger.read`` validates the deck's presence upstream — this raw pass only
    reads the ``CX`` lines idecomp does not model).
    """
    complexo_map: dict[int, list[int]] = {}
    if not dadger_path.is_file():
        return complexo_map
    with dadger_path.open(encoding="latin-1") as handle:
        for line in handle:
            match = _CX_LINE.match(line)
            if match is not None:
                complexo_map.setdefault(int(match.group(1)), []).append(
                    int(match.group(2))
                )
    return complexo_map


def _find_mlt(deck_dir: Path) -> Path | None:
    """Locate the deck's ``mlt.dat`` (média de longo termo), case-insensitively.

    ``mlt.dat`` is not one of :class:`~cobre_bridge.decomp.pipeline.DecompFiles`'
    resolved inputs (it feeds only the boundary FCF's inflow-lag mean fold, not
    the conversion), so it is discovered here directly. Returns ``None`` when the
    deck carries none.
    """
    for path in deck_dir.iterdir():
        if path.is_file() and path.name.lower() == "mlt.dat":
            return path
    return None


def _boundary_inflow_context(
    deck_files: DecompFiles,
    dadger: Dadger,
    id_map: DecompIdMap,
    *,
    coupling_month: int,
) -> (
    tuple[EffectiveCadastro, list[OperativeStage], dict[int, tuple[float, ...]]] | None
):
    """The effective cadastro, calendar, and per-plant inflow-lag mean fold.

    Reads the deck's ``mlt.dat`` and builds the seasonal-mean fold vector
    (:func:`~cobre_bridge.decomp.inflow_mlt.build_incremental_mlt` incrementalises
    the natural MLT, :func:`~cobre_bridge.decomp.inflow_mlt.coupling_lag_means`
    aligns it to the cut's lag-depth axis at ``coupling_month``), returning it
    alongside the ``effective``/``calendar`` the recent-observation seed also
    needs — the two ship together (mean fold + raw seed), sharing this one
    cadastro build.

    Returns ``None`` when the deck carries no ``mlt.dat``: without the seasonal
    means the deviation fold cannot be applied, so the boundary FCF stays on the
    unseeded lags-0 approximation (a logged tracked gap) rather than a raw seed
    the RHS would fail to offset — over-draining the reservoirs.
    """
    mlt_path = _find_mlt(deck_files.dadger.parent)
    if mlt_path is None:
        _LOG.warning(
            "no mlt.dat in %s; the boundary FCF's inflow-lag mean fold and the "
            "recent-observation seed are both skipped (the lag state stays at 0 "
            "-- a near-mean approximation, tracked gap), since seeding raw "
            "inflows without the seasonal-mean fold over-drains the reservoirs",
            deck_files.dadger.parent,
        )
        return None
    hidr = read_hidr(deck_files.hidr)
    calendar = operative_calendar_from_dadger(dadger)
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    incremental_mlt = build_incremental_mlt(
        Mlt.read(str(mlt_path)).valores, effective, id_map
    )
    means = coupling_lag_means(incremental_mlt, coupling_month)
    return effective, calendar, means


def _seed_recent_observations(
    case_dir: Path,
    deck_files: DecompFiles,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> int:
    """Add ``recent_observations`` to the case's ``initial_conditions.json``.

    Seeds cobre's PAR inflow-lag accumulator with the deck's pre-study observed
    (already-incremental) inflows
    (:func:`~cobre_bridge.decomp.scenarios.convert_recent_observation_windows`),
    so the boundary cut's inflow-lag terms are evaluated against the real recent
    inflows rather than 0. Written here — inside the importer, patched onto the
    already-converted case exactly like ``config.json``'s policy boundary —
    rather than in the conversion, so the raw seed and the RHS mean fold (which
    offsets it) are authored together and never ship apart. Returns the number of
    observation windows written (0 when the deck carries no observation tables).
    """
    windows = convert_recent_observation_windows(
        Vazoes.read(str(deck_files.vazoes)), effective, id_map, calendar
    )
    ic_path = case_dir / "initial_conditions.json"
    with ic_path.open(encoding="utf-8") as handle:
        doc = json.load(handle)
    doc["recent_observations"] = windows
    with ic_path.open("w", encoding="utf-8") as handle:
        json.dump(doc, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return len(windows)


def _build_gnl_ring_plan(case_dir: Path, deck_files: DecompFiles) -> GnlRingPlan | None:
    """Read the deck's ``dadgnl`` and build the GNL ring plan, or ``None``.

    Deck-reading wrapper around :func:`_gnl_targets_from`: returns ``None``
    when the deck carries no ``dadgnl`` file at all, or when
    :func:`~cobre_bridge.decomp.anticipated.read_gnl_model` reports the deck
    is GNL-off (no committed dispatch, the G6 gate) — reconciled with that
    reader's own gate, never re-derived here. Otherwise threads
    :func:`_post_horizon_start` into the resolved plan so
    ``fcf/mapper.py::_resolve_gnl_targets`` restricts placement to covered
    post-horizon lanes (ticket-013) without itself reading any deck file.
    """
    if deck_files.dadgnl is None:
        return None
    model = read_gnl_model(Dadgnl.read(str(deck_files.dadgnl)))
    if model is None:
        return None
    thermals_path = case_dir / "system" / "thermals.json"
    with thermals_path.open(encoding="utf-8") as handle:
        thermals_doc = json.load(handle)
    plan = _gnl_targets_from(model, thermals_doc)
    if plan is None:
        return None
    return GnlRingPlan(plan.targets, post_horizon_start=_post_horizon_start(case_dir))


def _gnl_deviation_rows(
    cuts: BoundaryCuts, gnl_plan: GnlRingPlan
) -> list[tuple[int, int, float, float, float]]:
    """The pre-fan-out patamar spread carried into each live GNL ring group.

    Mirrors ``mapper.py::_resolve_gnl_targets``'s ``col(s, p, l)`` flat-column
    formula exactly — never the placement/drop logic itself, which stays the
    mapper's job — to recompute, from the raw ``pi_gnl`` coefficients, how
    far each active cut's per-patamar sensitivities deviate from the uniform
    rate cobre's hours-weighted fan-out reconstructs. For each unique
    ``(submercado, lag)`` pair named by ``gnl_plan.targets``, returns
    ``(submercado, lag, carried_sum, relative_spread, absolute_spread)`` from
    whichever active record maximises the relative spread
    ``(max_p c_p - min_p c_p) / |sum_p c_p|`` (guarded to ``0.0`` when
    ``|sum_p c_p| <= 1e-9``) — a worst-case snapshot, not an aggregate across
    records; ``absolute_spread`` (``max_p c_p - min_p c_p``, ticket-013
    Requirement C.3) is that same selected record's un-normalised spread, so
    a caller can report it alongside the relative figure without the
    near-zero-Σ inflation the ratio alone is prone to. Skips a group whose
    ``(submercado, lag)`` falls outside ``cuts``' own ``pi_gnl`` shape (the
    mapper already records that as a target-side ``GnlDroppedTerm``).

    Returns rows sorted ascending by ``(submercado, lag)``; empty when
    ``cuts`` has no active records, or the boundary carries no GNL block
    (``lag_maximo_gnl == 0`` or an empty/misshapen ``pi_gnl``).
    """
    n_patamares = cuts.header.n_patamares
    lag_maximo_gnl = cuts.header.lag_maximo_gnl
    active_records = tuple(record for record in cuts.records if record.is_active)
    if not active_records or lag_maximo_gnl == 0:
        return []
    width = len(active_records[0].pi_gnl)
    block = n_patamares * lag_maximo_gnl
    if width == 0 or block == 0 or width % block != 0:
        return []
    n_submercados = width // block

    def col(submercado: int, patamar: int, lag: int) -> int:
        """Flat pi_gnl column for (submercado, patamar, lag), 1-based axes."""
        return ((submercado - 1) * n_patamares + (patamar - 1)) * lag_maximo_gnl + (
            lag - 1
        )

    groups = sorted({(target.submercado, target.nl_lag) for target in gnl_plan.targets})
    rows: list[tuple[int, int, float, float, float]] = []
    for submercado, lag in groups:
        if not (1 <= submercado <= n_submercados) or not (1 <= lag <= lag_maximo_gnl):
            continue
        cols = tuple(
            col(submercado, patamar, lag) for patamar in range(1, n_patamares + 1)
        )
        best_sum = 0.0
        best_spread = 0.0
        best_abs_spread = 0.0
        for record in active_records:
            values = tuple(record.pi_gnl[c] for c in cols)
            total = math.fsum(values)
            abs_spread = max(values) - min(values)
            spread = abs_spread / abs(total) if abs(total) > 1e-9 else 0.0
            if spread >= best_spread:
                best_spread = spread
                best_sum = total
                best_abs_spread = abs_spread
        rows.append((submercado, lag, best_sum, best_spread, best_abs_spread))
    return rows


def _emit_import_diagnostics(
    cuts: BoundaryCuts,
    mapping: MappingResult,
    gnl_plan: GnlRingPlan | None = None,
) -> None:
    """Surface the importer's documented, accepted approximations.

    Always emits ``boundary-fcf-cut-family-summary`` — the importer always
    authors cuts, so the family triage (built from
    :func:`~cobre_bridge.decomp.fcf.cortes.summarize_cut_families`, never
    re-implemented here) is always informative. Additionally emits
    ``boundary-fcf-source-only-plants-dropped`` when ``mapping.dropped`` is
    non-empty — a source-only plant has no target ``HydroStorage`` slot, so
    its storage/lag terms are omitted (D3: dropped, never folded). When
    ``gnl_plan`` is given and the boundary carries a GNL block
    (``cuts.header.lag_maximo_gnl > 0``), additionally emits
    ``boundary-fcf-gnl-anticipated-deviation`` — the per-``(submercado, lag)``
    pre-fan-out patamar spread the mapper's chain-rule sum collapses (see
    :func:`_gnl_deviation_rows`), headlined as a relative/absolute spread
    pair (the relative figure excludes a near-zero-Σ group per
    ``_GNL_DEVIATION_REL_FLOOR``, ticket-013 Requirement C.3) plus a
    dropped-coverage count read verbatim from ``mapping.gnl_dropped``
    (source-submercado drops plus, since ticket-013, non-covered
    dated-slot drops); ``gnl_plan=None`` (the default) gates it off
    entirely, so pre-ticket-010 2-arg callers are unchanged.

    Pure side effect via :func:`cobre_bridge.diagnostics.emit`: reads
    ``cuts``/``mapping``/``gnl_plan`` but does not alter any of them, so it
    can run before the checkpoint is written without changing the checkpoint
    bytes or the importer's return value. Mirrors ``decomp/pipeline.py``'s
    gated-INFO-``Diagnostic`` idiom; relies on the ambient-sink/log-fallback
    contract of ``diagnostics.emit`` rather than opening its own
    ``dx.collect()`` sink.
    """
    summary = summarize_cut_families(cuts)
    dx.emit(
        dx.Diagnostic(
            code="boundary-fcf-cut-family-summary",
            severity=dx.Severity.INFO,
            category="Boundary FCF",
            title=f"Boundary FCF authors {summary.n_active_cuts} cut(s)",
            # (Finding 1, ticket-013 Requirement C.1) `lag_nonzero_by_depth` is
            # the one fact not already in this string; every other figure a
            # `notes` bullet used to restate (n_active_cuts, storage_nonzero_
            # plants, rhs_min/max) is already here, so `notes` is dropped
            # rather than duplicating them. Full-precision RHS belongs in
            # `--diagnostics-json`, not a duplicate human bullet.
            summary=(
                f"{summary.n_active_cuts} active cut(s) authored from the "
                f"source model's boundary cuts; {summary.storage_nonzero_plants} "
                "plant(s) carry a nonzero storage coefficient; nonzero "
                f"inflow-lag plants by depth {summary.lag_nonzero_by_depth}; "
                f"RHS range [{summary.rhs_min:.6g}, {summary.rhs_max:.6g}]"
            ),
        ),
        logger=_LOG,
    )

    if mapping.dropped:
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-source-only-plants-dropped",
                severity=dx.Severity.INFO,
                category="Boundary FCF",
                title=(
                    f"{len(mapping.dropped)} source-only plant(s) dropped "
                    "from the boundary FCF"
                ),
                summary=(
                    f"{len(mapping.dropped)} plant(s) present in the source "
                    "model's boundary cuts have no target HydroStorage slot "
                    "in the converted case; their storage and inflow-lag "
                    "terms are omitted from every authored cut, never "
                    "folded into a neighbouring plant"
                ),
                table=dx.DiagnosticTable(
                    columns=["Plant code", "β (pi_varm)"],
                    rows=[
                        [term.plant_code, round(term.beta, 6)]
                        for term in mapping.dropped
                    ],
                    justify=["right", "right"],
                ),
            ),
            logger=_LOG,
        )

    if gnl_plan is not None and cuts.header.lag_maximo_gnl > 0:
        rows = _gnl_deviation_rows(cuts, gnl_plan)
        # "Dropped" is every GNL term that reached no *covered* target: a
        # source submercado with no live thermal at all (`thermal_id is
        # None`), or (ticket-013) a target's dated slot dropped for falling
        # before the post-study horizon (reason names it) — read straight
        # from `mapping.gnl_dropped`, never recomputing the covered/
        # uncovered split independently here.
        dropped_coverage_terms = [
            term
            for term in mapping.gnl_dropped
            if term.thermal_id is None or "post-study horizon" in term.reason
        ]
        # See `_GNL_DEVIATION_REL_FLOOR`'s docstring: a group carrying less
        # than that fraction of the panel's own max |Σ| is excluded from the
        # relative headline only (its row still renders below); the absolute
        # spread has no such denominator and is reported unfiltered. Guard
        # the degenerate case where every group's |Σ| is itself ~0 (no
        # group carries any real weight at all) — no group qualifies for a
        # meaningful relative headline, which then reports "n/a" rather than
        # a misleading 0.
        max_abs_sum = max((abs(row[2]) for row in rows), default=0.0)
        if max_abs_sum <= 1e-12:
            headline_rows: list[tuple[int, int, float, float, float]] = []
        else:
            headline_rows = [
                row
                for row in rows
                if abs(row[2]) >= _GNL_DEVIATION_REL_FLOOR * max_abs_sum
            ]
        relative_headline = (
            f"{max(row[3] for row in headline_rows):.4g}" if headline_rows else "n/a"
        )
        max_absolute_spread = max((row[4] for row in rows), default=0.0)
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-gnl-anticipated-deviation",
                severity=dx.Severity.INFO,
                category="Boundary FCF",
                title="GNL anticipated ring carries a per-patamar sum",
                summary=(
                    f"max pre-fan-out patamar spread {relative_headline} "
                    f"relative / {max_absolute_spread:.4g} absolute across "
                    f"{len(rows)} live GNL ring target group(s) (a group "
                    f"carrying less than {_GNL_DEVIATION_REL_FLOOR:g} of the "
                    "panel's max |Σ| is excluded from the relative headline); "
                    f"{len(dropped_coverage_terms)} GNL term(s) dropped for "
                    "no covered target (no live thermal in that submercado, "
                    "or delivery before the post-study horizon)"
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Submercado",
                        "Lag",
                        "Σ pi_gnl (carried)",
                        "Patamar spread",
                    ],
                    rows=[
                        [submercado, lag, round(carried_sum, 6), round(spread, 6)]
                        for submercado, lag, carried_sum, spread, _abs_spread in rows
                    ],
                    justify=["right", "right", "right", "right"],
                ),
            ),
            logger=_LOG,
        )


def _patch_policy_boundary(config_path: Path, *, source_stage: int) -> None:
    """Set ``["policy"]["boundary"]`` in ``config_path``, preserving the rest.

    Reads the whole ``config.json`` (``state_space``/``training``/
    ``simulation`` included), creates ``["policy"]`` if the case predates any
    policy section, and rewrites only ``["policy"]["boundary"]`` — mirrors
    ``decomp/pipeline.py``'s ``_write_json`` formatting (``indent=2``,
    ``ensure_ascii=False``, trailing newline) so the patched file matches the
    rest of the case's JSON output byte-for-byte in style.
    """
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    policy = config.setdefault("policy", {})
    # TRACKED COBRE-GAP WORKAROUND (C8, ~/git/cobre/plans/
    # conversion-found-improvements.md): cobre resolves this path against the
    # run's --output directory, not case_dir (run/policy.rs:209), while every
    # other case input resolves against case_dir. A default `cobre run
    # <case>` will not find `case_dir/boundary` — callers must run with
    # `--output <case_dir>` until cobre resolves policy.boundary.path
    # relative to case_dir.
    policy["boundary"] = {"path": "boundary", "source_stage": source_stage}
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def import_boundary_fcf(
    case_dir: Path,
    cortesh_path: Path | None,
    cortes_path: Path | None,
    *,
    work_dir: Path,
    cost_scale_factor: float,
) -> Path | None:
    """Import the source model's boundary FCF into the converted case at
    ``case_dir``.

    Gated on cut-file presence: if either ``cortesh_path`` or ``cortes_path``
    is ``None``, this is a no-op — no ``boundary/`` directory is written, and
    ``config.json`` is left untouched. Otherwise:

    1. Rebuilds the :class:`~cobre_bridge.decomp.id_map.DecompIdMap` from the
       deck at ``cortesh_path.parent`` (same-study: the boundary cut source
       is the very deck whose ``dadger`` produced ``case_dir``, so no
       separate deck path is needed).
    2. Reads the header (``cortesh_path``) and the boundary-stage cut records
       (``cortes_path``), deriving the boundary stage from the cut file's own
       trailer when it is a single-stage partition export.
    3. Checks the writer binding, then runs a 1-iteration in-process
       ``cobre.run.run`` pass on ``case_dir`` (checkpoint under ``work_dir``,
       the case is never mutated) to read back its terminal state-vector
       layout.
    4. Builds the deck's GNL ring plan (:func:`_build_gnl_ring_plan`) and the
       inflow-lag mean fold (:func:`_boundary_inflow_context`: the
       incrementalised ``mlt.dat`` aligned to the cut's lag-depth axis at the
       coupling month), then maps every boundary cut onto that layout: storage
       terms by plant code, inflow-lag terms by calendar-month lag depth (with
       each cut's RHS reduced by ``Σ lag_coef · mu`` so the loaded cut prices
       the raw lag state as the *deviation* from the seasonal mean —
       ``fcf/mapper.py``'s ``inflow_lag_means``), and ``AnticipatedThermalState``
       GNL-ring terms via the plan's per-block hours-weighted patamar sum
       (ticket-001); ``HydroTransitBucket`` slots are left at coefficient 0
       regardless (epic 5's job, not this one's).
    5. Surfaces the mapping's documented approximations as ``Diagnostic``s
       (:func:`_emit_import_diagnostics`): the always-on cut-family triage,
       the D3-dropped source-only plants (gated on non-empty), and — when the
       deck carries a GNL block — the per-``(submercado, lag)`` deviation the
       ring's chain-rule sum collapses.
    6. Assembles and writes ``case_dir/boundary/{metadata.json,
       cuts/stage_NNN.bin, basis/}``, patches ``case_dir/config.json``'s
       ``["policy"]["boundary"]`` to point at it, and — when the deck carries an
       ``mlt.dat`` (so the mean fold above was applied) — patches
       ``case_dir/initial_conditions.json`` with the pre-study
       ``recent_observations`` seed (:func:`_seed_recent_observations`), the raw
       inflow-lag values the folded RHS is built to offset. Fold and seed ship
       together or not at all.
    7. Logs the TRACKED COBRE-GAP WORKAROUND (C8) usage constraint this patch
       implies: until cobre resolves ``policy.boundary.path`` relative to
       ``case_dir`` rather than the run's ``--output`` directory, the case
       must be run with ``--output <case_dir>`` (see
       ``_patch_policy_boundary``'s code comment and
       ``~/git/cobre/plans/conversion-found-improvements.md``).

    Returns the ``case_dir/boundary`` path, or ``None`` on the no-cut-files
    no-op.

    Raises
    ------
    RuntimeError
        Propagated verbatim from ``fcf.bootstrap``'s ``ensure_writer_binding``
        (writer binding missing) or ``bootstrap_terminal_manifest`` (the
        in-process ``cobre.run.run`` failed or its checkpoint was malformed).
    ValueError
        Propagated verbatim from the cut reader (``fcf/cortes.py``, e.g. a
        non-individualized deck or a nonzero SAR coefficient), the mapper
        (``fcf/mapper.py``, e.g. no ``HydroStorage`` slots in the target
        manifest), or the writer (``fcf/writer.py``, e.g. a mapped
        coefficient vector length mismatch).
    """
    if cortesh_path is None or cortes_path is None:
        _LOG.info("boundary FCF skipped — no cut files")
        return None

    deck_files = discover_decomp_files(cortesh_path.parent)
    dadger = Dadger.read(str(deck_files.dadger))
    id_map = DecompIdMap.from_dadger(dadger)

    cortesh = Cortesh.read(str(cortesh_path))
    cuts = read_cortes(cortes_path, cortesh, boundary_stage=None)
    # `BoundaryCuts.boundary_stage` is typed `int`, but a single-stage export's
    # derived value inherits `numpy.int32` from `cortesh.ano_inicio_estudo`'s
    # own numpy dtype (confirmed against this deck) — narrow to a plain `int`
    # here, at the boundary between the numpy-sourced reader and the
    # JSON/cobre-FFI payloads this function builds below.
    boundary_stage = int(cuts.boundary_stage)

    ensure_writer_binding()
    import cobre

    # No explicit inflow-lag depth is supplied to the bootstrap: cobre sizes the
    # HydroInflowLag state block from the case's own PAR(p) model order (the same
    # autoregressive structure the boundary cuts' pi_qafl terms derive from), so
    # the terminal manifest already carries the slots the mapper places those
    # terms onto. The former state_space.inflow_lag_depth override was redundant
    # with that sizing and is rejected by cobre >= 0.14.
    manifest = bootstrap_terminal_manifest(case_dir, work_dir=work_dir)
    gnl_plan = _build_gnl_ring_plan(case_dir, deck_files)
    # Inflow-lag mean fold + recent-observation seed (built together, shipped
    # together — see `_boundary_inflow_context`). The coupling month is the cut
    # stage's own calendar month (`boundary_stage` is calendar-anchored as
    # `(Y - Y0)*12 + M`, so `(boundary_stage - 1) % 12 + 1 == M`); the fold
    # aligns each `pi_qafl` lag depth to `coupling_month - depth`. `None` (no
    # mlt.dat) leaves the lag state at 0 — no fold, no seed.
    coupling_month = ((boundary_stage - 1) % 12) + 1
    inflow_context = _boundary_inflow_context(
        deck_files, dadger, id_map, coupling_month=coupling_month
    )
    inflow_lag_means = inflow_context[2] if inflow_context is not None else None
    # Read stages.json's per-block hours once; cost_unit_hours (the
    # intercept/storage/inflow-lag scale) is this same vector's sum, never a
    # second stages.json read (ticket-001).
    coupling_block_hours = _final_stage_block_hours(case_dir)
    cost_unit_hours = math.fsum(coupling_block_hours)
    # CX register (complexo -> DECOMP components): the header prices a NEWAVE
    # complexo absent from the DECOMP model, so its coefficients replicate onto
    # the individual plants that share it (see map_boundary_cuts) rather than
    # being dropped.
    complexo_components = _read_complexo_map(deck_files.dadger)
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=cost_unit_hours,
        gnl_plan=gnl_plan,
        coupling_block_hours=coupling_block_hours,
        inflow_lag_means=inflow_lag_means,
        complexo_components=complexo_components,
    )
    _LOG.info(
        "scaling boundary FCF coefficients to cobre cost units over the "
        "coupling stage's %.0f h (the source's per-hour cut rate integrated "
        "over the coupling period; inflow-lag additionally x C_M3S2HM3 for "
        "cobre's m3/s lag state)",
        cost_unit_hours,
    )
    _emit_import_diagnostics(cuts, mapping, gnl_plan)

    stage_cuts_payload = build_stage_cuts_payload(
        mapping, manifest, stage_id=boundary_stage
    )
    completed_iterations = max((cut.iteration for cut in mapping.cuts), default=0)
    metadata = build_metadata(
        num_stages=1,
        cost_scale_factor=cost_scale_factor,
        completed_iterations=completed_iterations,
        final_lower_bound=0.0,
        max_iterations=completed_iterations,
        forward_passes=0,
        warm_start_cuts=0,
        rng_seed=0,
        created_at=datetime.now(tz=UTC).isoformat(),
        cobre_version=cobre.__version__,
    )

    boundary_dir = case_dir / "boundary"
    write_boundary_checkpoint(boundary_dir, stage_cuts_payload, metadata)

    _patch_policy_boundary(case_dir / "config.json", source_stage=boundary_stage)

    # Seed the pre-study inflow-lag state and record the mean fold — both gated
    # on the same mlt.dat presence as the fold above, so the raw seed never
    # ships without the RHS mean that offsets it (`_boundary_inflow_context`).
    if inflow_context is not None:
        effective, calendar, _ = inflow_context
        n_windows = _seed_recent_observations(
            case_dir, deck_files, effective, id_map, calendar
        )
        _LOG.info(
            "boundary FCF inflow-lag coupling: folded the seasonal-mean (MLT) "
            "deviation into %d cut RHS(es) and seeded %d recent-observation "
            "window(s) at coupling month %d, so the loaded cut prices the raw "
            "inflow-lag state as the deviation Q - mu",
            len(mapping.cuts),
            n_windows,
            coupling_month,
        )

    # TRACKED COBRE-GAP WORKAROUND (C8): cobre resolves policy.boundary.path
    # against the run's --output directory rather than the case directory the
    # checkpoint was authored into, so this case must be run with
    # --output=<case_dir>. Removal condition tracked in
    # ~/git/cobre/plans/conversion-found-improvements.md. The message below is
    # end-user-facing (no repo-internal references).
    _LOG.warning(
        "This case must be run with `cobre run %s --output %s`: the boundary "
        "cost-to-go checkpoint at %s is resolved relative to the run's output "
        "directory, so a plain `cobre run %s` (with output elsewhere) will not "
        "find it and will stop before the first iteration.",
        case_dir,
        case_dir,
        boundary_dir,
        case_dir,
    )

    return boundary_dir
