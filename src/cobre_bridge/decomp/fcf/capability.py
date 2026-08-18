"""Capability probe gating boundary-FCF import on a real CBVF round trip.

The branch wheel (``feat/cobre-gnl-boundary-pricing``, worktree
``~/git/cobre-gnlbp``) self-reports version ``0.13.0`` yet writes the
CBVF/``delivery_date`` checkpoint format that the released ``0.13.0`` wheel
does not understand (spec Sec. 2, D1). A version-string check would
therefore pass a wrong (released) wheel and then fail deep inside a
``cobre run`` with an opaque error. This module gates the ``--boundary-fcf``
path on a real write -> load round trip instead: it authors a minimal
synthetic checkpoint via ``cobre.write_policy_checkpoint``, reloads it via
``cobre.results.load_policy``, and asserts the reloaded terminal
``entity_manifest`` slot carries the CBVF-format ``delivery_date`` key — the
schema break that distinguishes the branch wheel from the released
``0.13.0`` wheel. Never a version comparison.

Mirrors ``fcf/bootstrap.py``'s ``ensure_writer_binding`` convention of a
lazy, function-body-only ``import cobre`` so this module stays importable
in a cobre-free (tier-1) environment.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
from cobre_bridge.decomp.fcf.mapper import MappedCut, MappingResult
from cobre_bridge.decomp.fcf.writer import build_metadata, build_stage_cuts_payload

#: The probe's single synthetic terminal-manifest slot. `delivery_date` is
#: deliberately omitted — cobre's `write_policy_checkpoint` treats it as
#: optional (defaulting to the "not applicable" sentinel) — this probe only
#: cares whether the *reloaded* slot carries the key at all, never what it
#: writes.
_PROBE_SLOT: dict[str, object] = {
    "entity_type": 0,
    "entity_id": 0,
    "subindex": 0,
    "was_active": True,
}
_PROBE_MANIFEST = TerminalManifest(entity_manifest=(_PROBE_SLOT,), state_dimension=1)
_PROBE_MAPPING = MappingResult(
    cuts=(
        MappedCut(
            intercept=0.0,
            coefficients=(0.0,),
            cut_id=0,
            iteration=0,
            forward_pass_index=0,
            is_active=True,
        ),
    ),
    dropped=(),
)
_PROBE_STAGE_ID = 0
_PROBE_CREATED_AT = "1970-01-01T00:00:00Z"

#: Remediation text raised on any probe failure. Kept as a module-level
#: constant so tests assert against it directly.
REMEDIATION = (
    "boundary-FCF capability probe failed: the installed cobre wheel does "
    "not write and load the CBVF delivery_date checkpoint format that "
    "--boundary-fcf requires; rebuild cobre from "
    "feat/cobre-gnl-boundary-pricing (worktree ~/git/cobre-gnlbp) — see "
    "docs/decomp-boundary-fcf-build.md"
)

#: Every exception type the CBVF round trip can fail with: `ModuleNotFoundError`
#: (cobre absent), `AttributeError` (missing `write_policy_checkpoint`/`results`
#: binding), `ValueError`/`OSError`/`RuntimeError` (every `cobre.errors.CobreError`
#: leaf subclasses one of those three builtins, per `cobre-python/src/errors.rs`),
#: and `TypeError`/`KeyError` from this probe's own access into a malformed
#: reloaded policy dict. Never a bare `except:`.
_PROBE_FAILURE_TYPES: tuple[type[Exception], ...] = (
    ModuleNotFoundError,
    AttributeError,
    ValueError,
    OSError,
    RuntimeError,
    TypeError,
    KeyError,
)


def ensure_boundary_fcf_capability() -> None:
    """Raise unless the installed cobre wheel writes+loads the CBVF format.

    Writes a minimal one-slot synthetic checkpoint into a
    :class:`tempfile.TemporaryDirectory`, reloads it via
    ``cobre.results.load_policy``, and requires the reloaded terminal
    ``entity_manifest`` slot to carry a ``delivery_date`` key — the schema
    break the released ``0.13.0`` wheel lacks. Leaves no artifacts on disk.

    Raises
    ------
    RuntimeError
        Naming ``feat/cobre-gnl-boundary-pricing``, ``~/git/cobre-gnlbp``,
        and ``docs/decomp-boundary-fcf-build.md`` — on any failure: cobre
        absent, no writer binding, the write/load call itself raising, or a
        reloaded slot lacking ``delivery_date``.
    """
    try:
        _probe_cbvf_roundtrip()
    except _PROBE_FAILURE_TYPES as err:
        raise RuntimeError(REMEDIATION) from err


def _probe_cbvf_roundtrip() -> None:
    """Write, reload, and format-check the synthetic checkpoint.

    Raises
    ------
    RuntimeError
        If the reloaded terminal ``entity_manifest`` slot lacks
        ``delivery_date`` — caught and re-wrapped by
        :func:`ensure_boundary_fcf_capability`.
    """
    import cobre

    payload = build_stage_cuts_payload(
        _PROBE_MAPPING, _PROBE_MANIFEST, stage_id=_PROBE_STAGE_ID
    )
    metadata = build_metadata(
        num_stages=1,
        cost_scale_factor=1.0,
        completed_iterations=0,
        final_lower_bound=0.0,
        max_iterations=0,
        forward_passes=0,
        warm_start_cuts=0,
        rng_seed=0,
        created_at=_PROBE_CREATED_AT,
        cobre_version="0.0.0",
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        boundary_dir = Path(tmp_dir) / "boundary"
        cobre.write_policy_checkpoint(boundary_dir, [payload], metadata)
        policy = cobre.results.load_policy(
            boundary_dir.parent, policy_subdir=boundary_dir.name
        )
        terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])
        entity_manifest = terminal["entity_manifest"]
        if not entity_manifest or "delivery_date" not in entity_manifest[0]:
            raise RuntimeError(
                "reloaded terminal entity_manifest slot lacks the CBVF "
                "delivery_date key"
            )
