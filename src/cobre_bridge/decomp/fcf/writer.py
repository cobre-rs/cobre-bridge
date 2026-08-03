"""Assemble and write the source model's boundary-cut checkpoint.

The mapper (``fcf/mapper.py``, ticket-005) produces a :class:`MappingResult`
of :class:`~cobre_bridge.decomp.fcf.mapper.MappedCut`, each already aligned
to the terminal manifest's state-vector layout (``fcf/bootstrap.py``,
ticket-004). This module assembles that pair into the plain-dict payload and
metadata shapes ``cobre.write_policy_checkpoint`` expects, then calls it to
produce ``boundary/{metadata.json, cuts/stage_NNN.bin, basis/}`` — a raw,
one-stage checkpoint the target case's ``config.json -> policy.boundary``
loads.

``cost_scale_factor`` is the single most dangerous field in ``metadata``: if
absent (``None``), cobre treats the checkpoint as legacy and silently scales
every value by 10⁶ (design §2.1). :func:`build_metadata` therefore refuses to
build a metadata dict without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cobre

from cobre_bridge.decomp.fcf.bootstrap import ensure_writer_binding

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
    from cobre_bridge.decomp.fcf.mapper import MappingResult


def build_stage_cuts_payload(
    mapping: MappingResult,
    manifest: TerminalManifest,
    *,
    stage_id: int,
) -> dict[str, Any]:
    """Assemble one ``stage_cuts`` entry from `mapping` over `manifest`.

    Each ``cuts[]`` entry carries the source ``MappedCut``'s real provenance
    verbatim — ``cut_id``, ``iteration``, ``forward_pass_index``, and
    ``is_active`` are never re-synthesized — with ``slot_index`` set to the
    cut's dense 0-based position in the pool (its `enumerate` index here, not
    the source ``cut_id``). ``entity_manifest`` is `manifest.entity_manifest`
    copied verbatim (never re-derived) so a future cobre layout change breaks
    loudly at load, not silently. ``active_cut_indices`` lists the pool
    positions whose `MappedCut.is_active` is `True`; ``populated_count`` is
    `len(cuts)`.

    Raises
    ------
    ValueError
        If any `MappedCut.coefficients` length disagrees with
        `manifest.state_dimension` — checked before any cobre call.
    """
    cuts: list[dict[str, Any]] = []
    active_cut_indices: list[int] = []
    for slot_index, mapped in enumerate(mapping.cuts):
        if len(mapped.coefficients) != manifest.state_dimension:
            raise ValueError(
                f"cut {mapped.cut_id}: coefficients has "
                f"{len(mapped.coefficients)} entries, expected "
                f"state_dimension={manifest.state_dimension}"
            )
        cuts.append(
            {
                "cut_id": mapped.cut_id,
                "slot_index": slot_index,
                "iteration": mapped.iteration,
                "forward_pass_index": mapped.forward_pass_index,
                "intercept": mapped.intercept,
                "coefficients": list(mapped.coefficients),
                "is_active": mapped.is_active,
            }
        )
        if mapped.is_active:
            active_cut_indices.append(slot_index)

    return {
        "stage_id": stage_id,
        "state_dimension": manifest.state_dimension,
        "capacity": len(cuts),
        "cuts": cuts,
        "active_cut_indices": active_cut_indices,
        "populated_count": len(cuts),
        "entity_manifest": list(manifest.entity_manifest),
    }


def build_metadata(
    *,
    state_dimension: int,
    num_stages: int,
    cost_scale_factor: float | None,
    completed_iterations: int,
    final_lower_bound: float,
    max_iterations: int,
    forward_passes: int,
    warm_start_cuts: int,
    rng_seed: int,
    created_at: str,
    cobre_version: str,
) -> dict[str, Any]:
    """Build the checkpoint `metadata` dict, refusing an unset `cost_scale_factor`.

    `created_at` is accepted as a parameter rather than derived internally
    (this module never calls `datetime.now()`) — the caller (ticket-009's
    orchestration) is responsible for supplying an ISO 8601 timestamp.

    Raises
    ------
    RuntimeError
        If `cost_scale_factor` is `None` — an unset marker makes cobre treat
        the checkpoint as legacy and silently scale every value by 10⁶.
    """
    if cost_scale_factor is None:
        raise RuntimeError(
            "cost_scale_factor must not be None: an unset marker makes "
            "cobre treat this checkpoint as legacy and silently scale "
            "every value by 10⁶"
        )
    return {
        "cobre_version": cobre_version,
        "created_at": created_at,
        "completed_iterations": completed_iterations,
        "final_lower_bound": final_lower_bound,
        "state_dimension": state_dimension,
        "num_stages": num_stages,
        "max_iterations": max_iterations,
        "forward_passes": forward_passes,
        "warm_start_cuts": warm_start_cuts,
        "rng_seed": rng_seed,
        "cost_scale_factor": cost_scale_factor,
    }


def write_boundary_checkpoint(
    boundary_dir: Path,
    stage_cuts_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    """Write `stage_cuts_payload` + `metadata` to `boundary_dir` via cobre.

    Calls :func:`ensure_writer_binding` first (the environment gate), then
    `cobre.write_policy_checkpoint(boundary_dir, [stage_cuts_payload],
    metadata)` with `stage_bases`/`stage_states` left at their defaults — a
    raw-authored checkpoint carries neither, so `boundary_dir/basis/` is
    written empty and no `states/` directory is created.

    Raises
    ------
    RuntimeError
        Via `ensure_writer_binding`, if the installed `cobre` wheel lacks
        `write_policy_checkpoint`.
    """
    ensure_writer_binding()
    cobre.write_policy_checkpoint(boundary_dir, [stage_cuts_payload], metadata)
