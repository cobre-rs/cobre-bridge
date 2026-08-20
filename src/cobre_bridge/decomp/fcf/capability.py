"""Capability probe gating boundary-FCF import on a real CBVF round trip.

``convert decomp`` imports the deck's boundary FCF by default, which needs a
``cobre-python`` that can write and reload the CBVF/``delivery_date``
checkpoint format (introduced in cobre 0.14.x; ``cobre-python`` is a required
bridge dependency). This module gates the ``--boundary-fcf`` path on a real
write -> load round trip rather than a version-string check: a round trip also
catches a broken, partial, or ABI-mismatched wheel that reports a satisfying
version yet cannot actually read back what it wrote. It authors a minimal
synthetic checkpoint via ``cobre.write_policy_checkpoint``, reloads it via
``cobre.results.load_policy``, and asserts the reloaded terminal
``entity_manifest`` slot carries the CBVF-format ``delivery_date`` key.

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
    "The boundary cost-to-go function could not be imported: the installed "
    "cobre package cannot write and read back the policy checkpoint format it "
    "requires. cobre-python is a required dependency of cobre-bridge — reinstall "
    "or upgrade it (for example: pip install --upgrade cobre-python), or "
    "reinstall cobre-bridge, then try again. To convert without the boundary "
    "cost-to-go function, re-run with --no-fcf."
)

#: Every exception type the CBVF round trip can fail with: `ModuleNotFoundError`
#: (cobre absent), `AttributeError` (missing `write_policy_checkpoint`/`results`
#: binding), `ValueError`/`OSError`/`RuntimeError` (every `cobre.errors.CobreError`
#: leaf subclasses one of those three builtins),
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
        Carrying :data:`REMEDIATION` — a self-contained, end-user-facing
        message (the cobre-python install/upgrade fix plus the ``--no-fcf``
        escape hatch, with no repo-internal paths) — on any failure: cobre
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
