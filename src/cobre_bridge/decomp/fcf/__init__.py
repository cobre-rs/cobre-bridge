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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import cobre
from idecomp.decomp import Dadger
from inewave.newave import Cortesh

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)
from cobre_bridge.decomp.fcf.cortes import read_cortes, summarize_cut_families
from cobre_bridge.decomp.fcf.mapper import map_boundary_cuts
from cobre_bridge.decomp.fcf.writer import (
    build_metadata,
    build_stage_cuts_payload,
    write_boundary_checkpoint,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.pipeline import discover_decomp_files

if TYPE_CHECKING:
    from pathlib import Path

    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
    from cobre_bridge.decomp.fcf.mapper import MappingResult

_LOG = logging.getLogger(__name__)


def _emit_import_diagnostics(cuts: BoundaryCuts, mapping: MappingResult) -> None:
    """Surface the importer's two documented, accepted approximations.

    Always emits ``boundary-fcf-cut-family-summary`` — the importer always
    authors cuts, so the family triage (built from
    :func:`~cobre_bridge.decomp.fcf.cortes.summarize_cut_families`, never
    re-implemented here) is always informative. Additionally emits
    ``boundary-fcf-source-only-plants-dropped`` when ``mapping.dropped`` is
    non-empty — a source-only plant has no target ``HydroStorage`` slot, so
    its storage/lag terms are omitted (D3: dropped, never folded).

    Pure side effect via :func:`cobre_bridge.diagnostics.emit`: reads
    ``cuts``/``mapping`` but does not alter either, so it can run before the
    checkpoint is written without changing the checkpoint bytes or the
    importer's return value. Mirrors ``decomp/pipeline.py``'s gated-INFO-
    ``Diagnostic`` idiom; relies on the ambient-sink/log-fallback contract of
    ``diagnostics.emit`` rather than opening its own ``dx.collect()`` sink.
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
    4. Maps every boundary cut onto that layout (storage terms by plant code,
       inflow-lag terms by calendar-month lag depth; the
       ``AnticipatedThermalState`` GNL ring and ``HydroTransitBucket`` slots
       are left at coefficient 0 — epic 3's job, not this one's).
    5. Surfaces the mapping's two documented approximations as ``Diagnostic``s
       (:func:`_emit_import_diagnostics`): the always-on cut-family triage,
       and — gated on non-empty — the D3-dropped source-only plants.
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

    ensure_writer_binding()
    manifest = bootstrap_terminal_manifest(case_dir, cobre_bin, work_dir=work_dir)
    mapping = map_boundary_cuts(cuts, manifest, id_map)
    _emit_import_diagnostics(cuts, mapping)

    stage_cuts_payload = build_stage_cuts_payload(
        mapping, manifest, stage_id=boundary_stage
    )
    completed_iterations = max((cut.iteration for cut in mapping.cuts), default=0)
    metadata = build_metadata(
        state_dimension=manifest.state_dimension,
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
