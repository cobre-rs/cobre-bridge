"""Manifest-to-manifest mapper for the source model's boundary cuts.

The bootstrap stage (``fcf/bootstrap.py``, ticket-004) reads back cobre's
terminal ``entity_manifest`` — the target case's per-slot state-vector
layout. This module maps each of the source model's boundary cuts
(``fcf/cortes.py``'s :class:`~cobre_bridge.decomp.fcf.cortes.BoundaryCuts`,
ticket-002) onto that layout: storage terms join by plant code, inflow-lag
terms join 1:1 by calendar-month lag slot, and transit-bucket / GNL-
anticipated-ring slots are left at coefficient 0 (the GNL ring is epic 3's
job). A source plant with no match in the target manifest is dropped (D3),
never folded into a neighbour, and recorded in
:class:`MappingResult.dropped` for the diagnostics layer (epic 4) to render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
    from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
    from cobre_bridge.decomp.id_map import DecompIdMap

#: cobre `policy.fbs` entity_type codes (confirmed against `policy_export.rs`).
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3

#: `HydroStorage`'s `subindex` is always 0 (policy.fbs: one slot per plant).
_STORAGE_SUBINDEX = 0


@dataclass(frozen=True)
class MappedCut:
    """One source cut mapped onto the target's state-vector layout.

    ``coefficients`` is a full-length vector aligned to
    ``TerminalManifest.state_dimension`` — every target slot has an
    explicit coefficient, never merely unset. ``intercept`` is the source
    record's ``rhs`` carried verbatim (already the ``alpha - beta'xhat``
    form; never re-derived, per §2.1). ``cut_id``, ``iteration``,
    ``forward_pass_index``, and ``is_active`` are the source
    ``StageCutRecord``'s provenance fields, carried verbatim so the
    checkpoint writer (ticket-008) has every field it needs without
    re-reading the source.
    """

    intercept: float
    coefficients: tuple[float, ...]
    cut_id: int
    iteration: int
    forward_pass_index: int
    is_active: bool


@dataclass(frozen=True)
class DroppedTerm:
    """A source-only plant's storage term, dropped rather than folded (D3).

    ``beta`` is that plant's ``pi_varm`` coefficient in the source's first
    cut record — a representative value for the diagnostics layer (epic 4)
    to report; drop status itself is a per-plant, cut-invariant property
    (target availability does not vary cut to cut), so exactly one
    :class:`DroppedTerm` is recorded per unresolvable plant, not one per cut.
    """

    plant_code: int
    beta: float


@dataclass(frozen=True)
class MappingResult:
    """The mapped cuts plus every D3-dropped source-only plant term."""

    cuts: tuple[MappedCut, ...]
    dropped: tuple[DroppedTerm, ...]


def _default_lag_slot_of(depth: int) -> int:
    """Identity calendar-month lag mapping: source lag depth d -> subindex d-1."""
    return depth - 1


def _slot_int(slot: Mapping[str, object], field: str) -> int:
    """Extract `field` from a manifest slot dict, validating it is an `int`.

    cobre's own `load_policy` hands back each slot as an untyped
    `dict[str, object]`; this narrows the three positional-key fields
    (`entity_type`, `entity_id`, `subindex`) with an explicit runtime check
    rather than a bare `cast`, so a malformed slot fails loudly here instead
    of silently corrupting a downstream index lookup.
    """
    value = slot[field]
    if not isinstance(value, int):
        raise TypeError(
            f"manifest slot field {field!r} is {type(value).__name__}, not "
            f"int: {value!r}"
        )
    return value


def _index_manifest(manifest: TerminalManifest) -> dict[tuple[int, int, int], int]:
    """Index the target manifest by (entity_type, entity_id, subindex) -> position."""
    return {
        (
            _slot_int(slot, "entity_type"),
            _slot_int(slot, "entity_id"),
            _slot_int(slot, "subindex"),
        ): position
        for position, slot in enumerate(manifest.entity_manifest)
    }


def _lag_subindex_bound(slot_positions: Mapping[tuple[int, int, int], int]) -> int:
    """One past the max `HydroInflowLag` subindex present, or 0 if absent.

    ``0`` signals the family is entirely absent from the manifest — a
    legitimate case shape (a storage-only converted case, pre-ticket-006),
    not a read bug — and disables the `lag_slot_of` bounds check entirely.
    """
    lag_subindices = [
        subindex
        for entity_type, _entity_id, subindex in slot_positions
        if entity_type == _HYDRO_INFLOW_LAG
    ]
    return max(lag_subindices) + 1 if lag_subindices else 0


def _validated_lag_subindices(
    lag_slot_of: Callable[[int], int], lag_bound: int
) -> tuple[int, ...]:
    """Resolve `lag_slot_of(d)` for calendar-month depth `d` in 1..12.

    Returns a 12-tuple of target subindices (index 0 == lag depth 1),
    bound-checked against `lag_bound` up front — before any cut is
    processed — so a mis-injected `lag_slot_of` fails fast with a
    `ValueError` naming the offending depth, never an `IndexError` from a
    downstream indexing operation. The check is skipped when `lag_bound`
    is 0 (no `HydroInflowLag` family in the manifest at all).
    """
    subindices = tuple(lag_slot_of(depth) for depth in range(1, 13))
    if lag_bound == 0:
        return subindices
    for depth, subindex in enumerate(subindices, start=1):
        if not 0 <= subindex < lag_bound:
            raise ValueError(
                f"lag_slot_of({depth}) = {subindex} is out of range for the "
                f"manifest's {lag_bound} HydroInflowLag slot(s)"
            )
    return subindices


def _resolve_storage_targets(
    cuts: BoundaryCuts,
    id_map: DecompIdMap,
    slot_positions: Mapping[tuple[int, int, int], int],
) -> tuple[dict[int, tuple[int, int]], tuple[DroppedTerm, ...]]:
    """Resolve each source plant's `(hydro_id, storage position)` once.

    A plant is dropped (D3) when its code is unknown to `id_map` (a
    `KeyError` from `hydro_id`) or when the target manifest has no matching
    `HydroStorage` slot for the resolved id — either way, the plant's
    storage *and* inflow-lag terms are omitted from every mapped cut, never
    folded into a neighbour.
    """
    representative_varm = cuts.records[0].pi_varm if cuts.records else ()
    resolved: dict[int, tuple[int, int]] = {}
    dropped: list[DroppedTerm] = []
    for plant_index, plant_code in enumerate(cuts.header.plant_codes):
        try:
            hydro_id = id_map.hydro_id(plant_code)
        except KeyError:
            beta = representative_varm[plant_index] if representative_varm else 0.0
            dropped.append(DroppedTerm(plant_code=plant_code, beta=beta))
            continue
        position = slot_positions.get((_HYDRO_STORAGE, hydro_id, _STORAGE_SUBINDEX))
        if position is None:
            beta = representative_varm[plant_index] if representative_varm else 0.0
            dropped.append(DroppedTerm(plant_code=plant_code, beta=beta))
            continue
        resolved[plant_index] = (hydro_id, position)
    return resolved, tuple(dropped)


def map_boundary_cuts(
    cuts: BoundaryCuts,
    manifest: TerminalManifest,
    id_map: DecompIdMap,
    *,
    lag_slot_of: Callable[[int], int] = _default_lag_slot_of,
) -> MappingResult:
    """Map every cut in `cuts.records` onto `manifest`'s state-vector layout.

    Storage terms join by plant code (`HydroStorage`, D3-drop on a
    source-only plant); inflow-lag terms join 1:1 by `lag_slot_of` onto
    `HydroInflowLag`; `HydroTransitBucket` and `AnticipatedThermalState`
    (the GNL ring — epic 3's job, not this ticket's) slots are left at an
    explicit `0.0`. `intercept` is the source record's `rhs`, carried
    verbatim (never re-derived from alpha/x-hat). Produces one `MappedCut`
    per source record, active or not — active-frontier selection is
    ticket-008's writer concern, not the mapper's.

    Raises
    ------
    ValueError
        If the target manifest has no `HydroStorage` slots at all (a
        terminal-manifest read bug — never raised for a merely-absent
        `HydroInflowLag`/`HydroTransitBucket`/`AnticipatedThermalState`
        family, a legitimate case shape); if `lag_slot_of` returns a
        subindex out of range for the manifest's `HydroInflowLag` slots; or
        if a mapped coefficient vector's length disagrees with
        `manifest.state_dimension`.
    """
    slot_positions = _index_manifest(manifest)
    if not any(entity_type == _HYDRO_STORAGE for entity_type, _, _ in slot_positions):
        raise ValueError(
            "target manifest has no HydroStorage slots at all; a real "
            "terminal-manifest read must carry at least the storage family"
        )

    resolved_storage, dropped = _resolve_storage_targets(cuts, id_map, slot_positions)
    lag_bound = _lag_subindex_bound(slot_positions)
    lag_subindices = _validated_lag_subindices(lag_slot_of, lag_bound)

    mapped_cuts: list[MappedCut] = []
    for record in cuts.records:
        coefficients = [0.0] * manifest.state_dimension
        for plant_index, (hydro_id, storage_position) in resolved_storage.items():
            coefficients[storage_position] = record.pi_varm[plant_index]
            if lag_bound == 0:
                continue
            plant_lags = record.pi_qafl[plant_index]
            for depth_index, subindex in enumerate(lag_subindices):
                lag_position = slot_positions.get(
                    (_HYDRO_INFLOW_LAG, hydro_id, subindex)
                )
                if lag_position is not None:
                    coefficients[lag_position] = plant_lags[depth_index]

        if len(coefficients) != manifest.state_dimension:
            raise ValueError(
                f"mapped coefficient vector length {len(coefficients)} != "
                f"state_dimension {manifest.state_dimension}"
            )

        mapped_cuts.append(
            MappedCut(
                intercept=record.rhs,
                coefficients=tuple(coefficients),
                cut_id=record.cut_id,
                iteration=record.iteration,
                forward_pass_index=record.forward_pass_index,
                is_active=record.is_active,
            )
        )

    return MappingResult(cuts=tuple(mapped_cuts), dropped=dropped)
