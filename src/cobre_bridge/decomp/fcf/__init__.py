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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from idecomp.decomp import Dadger, Dadgnl
from inewave.newave import Cortesh

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.anticipated import read_gnl_model
from cobre_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)
from cobre_bridge.decomp.fcf.cortes import (
    read_cortes,
    required_inflow_lag_depth,
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
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.pipeline import DecompFiles, discover_decomp_files

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cobre_bridge.decomp.anticipated import GnlCommitmentModel
    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
    from cobre_bridge.decomp.fcf.mapper import MappingResult

_LOG = logging.getLogger(__name__)


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


def _build_gnl_ring_plan(case_dir: Path, deck_files: DecompFiles) -> GnlRingPlan | None:
    """Read the deck's ``dadgnl`` and build the GNL ring plan, or ``None``.

    Deck-reading wrapper around :func:`_gnl_targets_from`: returns ``None``
    when the deck carries no ``dadgnl`` file at all, or when
    :func:`~cobre_bridge.decomp.anticipated.read_gnl_model` reports the deck
    is GNL-off (no committed dispatch, the G6 gate) — reconciled with that
    reader's own gate, never re-derived here.
    """
    if deck_files.dadgnl is None:
        return None
    model = read_gnl_model(Dadgnl.read(str(deck_files.dadgnl)))
    if model is None:
        return None
    thermals_path = case_dir / "system" / "thermals.json"
    with thermals_path.open(encoding="utf-8") as handle:
        thermals_doc = json.load(handle)
    return _gnl_targets_from(model, thermals_doc)


def _gnl_deviation_rows(
    cuts: BoundaryCuts, gnl_plan: GnlRingPlan
) -> list[tuple[int, int, float, float]]:
    """The pre-fan-out patamar spread carried into each live GNL ring group.

    Mirrors ``mapper.py::_resolve_gnl_targets``'s ``col(s, p, l)`` flat-column
    formula exactly — never the placement/drop logic itself, which stays the
    mapper's job — to recompute, from the raw ``pi_gnl`` coefficients, how
    far each active cut's per-patamar sensitivities deviate from the uniform
    rate cobre's hours-weighted fan-out reconstructs. For each unique
    ``(submercado, lag)`` pair named by ``gnl_plan.targets``, returns
    ``(submercado, lag, carried_sum, spread)`` from whichever active record
    maximises the spread ``(max_p c_p - min_p c_p) / |sum_p c_p|`` (guarded
    to ``0.0`` when ``|sum_p c_p| <= 1e-9``) — a worst-case snapshot, not an
    aggregate across records. Skips a group whose ``(submercado, lag)`` falls
    outside ``cuts``' own ``pi_gnl`` shape (the mapper already records that as
    a target-side ``GnlDroppedTerm``).

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
    rows: list[tuple[int, int, float, float]] = []
    for submercado, lag in groups:
        if not (1 <= submercado <= n_submercados) or not (1 <= lag <= lag_maximo_gnl):
            continue
        cols = tuple(
            col(submercado, patamar, lag) for patamar in range(1, n_patamares + 1)
        )
        best_sum = 0.0
        best_spread = 0.0
        for record in active_records:
            values = tuple(record.pi_gnl[c] for c in cols)
            total = math.fsum(values)
            spread = (
                (max(values) - min(values)) / abs(total) if abs(total) > 1e-9 else 0.0
            )
            if spread >= best_spread:
                best_spread = spread
                best_sum = total
        rows.append((submercado, lag, best_sum, best_spread))
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
    :func:`_gnl_deviation_rows`); ``gnl_plan=None`` (the default) gates it off
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
            summary=(
                f"{summary.n_active_cuts} active cut(s) authored from the "
                f"source model's boundary cuts; {summary.storage_nonzero_plants} "
                "plant(s) carry a nonzero storage coefficient, RHS range "
                f"[{summary.rhs_min:.6g}, {summary.rhs_max:.6g}]"
            ),
            notes=[
                f"n_active_cuts={summary.n_active_cuts}",
                f"storage_nonzero_plants={summary.storage_nonzero_plants}",
                f"lag_nonzero_by_depth={summary.lag_nonzero_by_depth}",
                f"rhs_min={summary.rhs_min!r}",
                f"rhs_max={summary.rhs_max!r}",
            ],
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
        dropped_source_terms = [
            term for term in mapping.gnl_dropped if term.thermal_id is None
        ]
        max_spread = max((row[3] for row in rows), default=0.0)
        dx.emit(
            dx.Diagnostic(
                code="boundary-fcf-gnl-anticipated-deviation",
                severity=dx.Severity.INFO,
                category="Boundary FCF",
                title="GNL anticipated ring carries a per-patamar sum",
                summary=(
                    f"max pre-fan-out patamar spread {max_spread:.4g} across "
                    f"{len(rows)} live GNL ring target group(s); "
                    f"{len(dropped_source_terms)} source submercado GNL "
                    "term(s) dropped (no live thermal in that submercado)"
                ),
                table=dx.DiagnosticTable(
                    columns=[
                        "Submercado",
                        "Lag",
                        "Σ pi_gnl (carried)",
                        "Patamar spread",
                        "Dropped",
                    ],
                    rows=[
                        [
                            submercado,
                            lag,
                            round(carried_sum, 6),
                            round(spread, 6),
                            len(dropped_source_terms),
                        ]
                        for submercado, lag, carried_sum, spread in rows
                    ],
                    justify=["right", "right", "right", "right", "right"],
                ),
                remediation=(
                    "The bridge carries a single per-patamar sum into each "
                    "anticipated ring slot; cobre re-derives per-block "
                    "coefficients via hours-weighted fan-out — an accepted, "
                    "documented approximation (TRACKED COBRE-GAP C12, "
                    "~/git/cobre/plans/conversion-found-improvements.md)"
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


def _patch_inflow_lag_depth(config_path: Path, *, depth: int) -> None:
    """Set ``["state_space"]["inflow_lag_depth"]`` in ``config_path``.

    ``decomp/config.py`` emits no ``state_space`` block: with the boundary FCF
    deferred, the external inflow model needs no lag state and cobre resolves a
    zero depth. The inflow-lag depth is a property of the *boundary policy*, so
    it is reserved here — and only here — when a boundary FCF is imported, to
    exactly the depth the loaded cuts reference (``depth`` =
    :func:`~cobre_bridge.decomp.fcf.cortes.required_inflow_lag_depth`). Sizing
    the reserved lag state to the cuts lets the bootstrap manifest (and the
    final run's terminal cut) hold the boundary policy's conditioning history.

    Must run before :func:`bootstrap_terminal_manifest`, whose 1-iteration pass
    resolves the terminal manifest the cuts are mapped onto. cobre is slated to
    infer this depth from the checkpoint itself, retiring this patch (see
    ``~/git/cobre/plans/state-space-inflow-lag-depth-inference-spec.md``).

    Mirrors :func:`_patch_policy_boundary`'s read-modify-write and JSON
    formatting so the patched file matches the rest of the case byte-for-byte.
    """
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    config.setdefault("state_space", {})["inflow_lag_depth"] = depth
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def import_boundary_fcf(
    case_dir: Path,
    cortesh_path: Path | None,
    cortes_path: Path | None,
    *,
    cobre_bin: Path,
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
    3. Checks the writer binding, then runs a 1-iteration ``cobre_bin`` pass
       on a copy of ``case_dir`` (under ``work_dir``) to read back its
       terminal state-vector layout.
    4. Builds the deck's GNL ring plan (:func:`_build_gnl_ring_plan`) and maps
       every boundary cut onto that layout: storage terms by plant code,
       inflow-lag terms by calendar-month lag depth, and
       ``AnticipatedThermalState`` GNL-ring terms via the plan's chain-rule
       patamar sum; ``HydroTransitBucket`` slots are left at coefficient 0
       regardless (epic 5's job, not this one's).
    5. Surfaces the mapping's documented approximations as ``Diagnostic``s
       (:func:`_emit_import_diagnostics`): the always-on cut-family triage,
       the D3-dropped source-only plants (gated on non-empty), and — when the
       deck carries a GNL block — the per-``(submercado, lag)`` deviation the
       ring's chain-rule sum collapses.
    6. Assembles and writes ``case_dir/boundary/{metadata.json,
       cuts/stage_NNN.bin, basis/}``, then patches ``case_dir/config.json``'s
       ``["policy"]["boundary"]`` to point at it.
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
        bootstrap ``cobre_bin run`` failed or its checkpoint was malformed).
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

    # Reserve inflow-lag state to the depth the boundary cuts actually
    # reference, BEFORE the bootstrap pass resolves the terminal manifest the
    # cuts are mapped onto (`convert_config` emits no `state_space`, so cobre
    # would otherwise size zero lag slots and the mapping would have nowhere to
    # land the pi_qafl terms). A boundary that prices no lag state (depth 0)
    # needs no reservation — and cobre rejects an explicit 0.
    lag_depth = required_inflow_lag_depth(summarize_cut_families(cuts))
    if lag_depth >= 1:
        _patch_inflow_lag_depth(case_dir / "config.json", depth=lag_depth)
        _LOG.info(
            "boundary FCF references inflow-lag state to depth %d; reserving "
            "state_space.inflow_lag_depth=%d (derived from the source cuts, not "
            "a fixed constant)",
            lag_depth,
            lag_depth,
        )

    ensure_writer_binding()
    import cobre

    manifest = bootstrap_terminal_manifest(case_dir, cobre_bin, work_dir=work_dir)
    gnl_plan = _build_gnl_ring_plan(case_dir, deck_files)
    mapping = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=gnl_plan)
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
    _LOG.warning(
        "TRACKED COBRE-GAP WORKAROUND (C8): cobre resolves "
        "policy.boundary.path against the run's --output directory, not "
        "case_dir (~/git/cobre/plans/conversion-found-improvements.md); "
        "until cobre is fixed, run this case with `cobre run %s --output "
        "%s` so output_dir == case_dir and %s resolves — a default `cobre "
        "run <case>` (no --output, or --output pointed elsewhere) will "
        "NOT find the boundary checkpoint and aborts before any iteration",
        case_dir,
        case_dir,
        boundary_dir,
    )

    return boundary_dir
