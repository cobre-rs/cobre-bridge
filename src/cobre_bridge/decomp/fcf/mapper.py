"""Manifest-to-manifest mapper for the source model's boundary cuts.

The bootstrap stage (``fcf/bootstrap.py``, ticket-004) reads back cobre's
terminal ``entity_manifest`` — the target case's per-slot state-vector
layout. This module maps each of the source model's boundary cuts
(``fcf/cortes.py``'s :class:`~cobre_bridge.decomp.fcf.cortes.BoundaryCuts`,
ticket-002) onto that layout: storage terms join by plant code, inflow-lag
terms join 1:1 by calendar-month lag slot, and — when the caller supplies a
:class:`GnlRingPlan` — GNL-anticipated-ring terms join each target's dated
ring slot(s) via a chain-rule patamar sum over ``pi_gnl`` (ticket-009). A
source plant with no match in the target manifest is dropped (D3), never
folded into a neighbour, and recorded in :class:`MappingResult.dropped` for
the diagnostics layer (epic 4) to render; a GNL source/target term with no
live counterpart is dropped the same way into
:class:`MappingResult.gnl_dropped`. ``HydroTransitBucket`` slots, and any
``AnticipatedThermalState`` ring slot with no resolved target (including the
undated sentinel slot), are left at an explicit coefficient ``0.0``.
"""

from __future__ import annotations

import math
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

#: cobre's `i32::MIN` sentinel for "no delivery date" — the undated
#: `AnticipatedThermalState` slot (the in-study anticipation, already priced
#: by the converter's `past_anticipated_commitments`), never a GNL target.
_DELIVERY_DATE_SENTINEL = -2147483648


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
class GnlThermalTarget:
    """One GNL thermal's ring membership: which `pi_gnl` axes feed it.

    ``thermal_id`` is the cobre ring `entity_id`; ``submercado`` is the
    1-based source submercado index matching the `pi_gnl` sbm-major column
    layout (`cortes.py::_gnl_columns`); ``nl_lag`` is the plant's 1-based
    dispatch-anticipation lag — the `pi_gnl` lag-axis index whose coefficient
    lands on this thermal's ring slot(s) [ASSUMPTION B].
    """

    thermal_id: int
    submercado: int
    nl_lag: int


@dataclass(frozen=True)
class GnlRingPlan:
    """The resolved submercado -> GNL-thermal membership for one deck.

    Built by the importer (ticket-010) from the deck's own GNL declarations;
    this module never derives it — see the module docstring's deck-free
    contract.
    """

    targets: tuple[GnlThermalTarget, ...]


@dataclass(frozen=True)
class GnlDroppedTerm:
    """A GNL source/target term that reached no live dated ring slot.

    ``thermal_id`` is ``None`` for a source-submercado drop (no
    :class:`GnlThermalTarget` claims that submercado at all) and the
    resolved thermal id for a target-side drop (out-of-range
    lag/submercado, or a thermal with no dated ring slot). ``coefficient``
    is a representative value for the diagnostics layer (ticket-010) to
    report — the summed source coefficient for a source-submercado drop,
    ``0.0`` for a target-side drop (no source coefficient is attributable to
    a target that never resolves). Recorded once per unresolvable
    (submercado, lag) or target, never folded into a neighbour (D3-like).
    """

    thermal_id: int | None
    submercado: int
    nl_lag: int
    coefficient: float
    reason: str


@dataclass(frozen=True)
class MappingResult:
    """The mapped cuts plus every D3-dropped source-only term.

    ``dropped`` carries storage/lag source-only plants; ``gnl_dropped``
    (defaulted, so pre-ticket-009 constructions keep working) carries GNL
    source-submercado and target terms with no live dated ring slot.
    """

    cuts: tuple[MappedCut, ...]
    dropped: tuple[DroppedTerm, ...]
    gnl_dropped: tuple[GnlDroppedTerm, ...] = ()


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


def _index_gnl_ring(
    manifest: TerminalManifest,
) -> dict[int, tuple[tuple[int, int, int], ...]]:
    """Index the target manifest's `AnticipatedThermalState` ring by thermal id.

    Unlike `_index_manifest` (which discards `delivery_date` and would
    silently collapse a thermal's sentinel and dated slots onto the same
    key), this keeps every `(subindex, delivery_date, position)` triple per
    `entity_id` so the GNL placement can tell a dated slot from the undated
    sentinel. A separate index from `_index_manifest`'s
    `(entity_type, entity_id, subindex) -> position` contract, which the
    storage/lag path still depends on unchanged.
    """
    by_thermal: dict[int, list[tuple[int, int, int]]] = {}
    for position, slot in enumerate(manifest.entity_manifest):
        if _slot_int(slot, "entity_type") != _ANTICIPATED_THERMAL_STATE:
            continue
        thermal_id = _slot_int(slot, "entity_id")
        subindex = _slot_int(slot, "subindex")
        delivery_date = _slot_int(slot, "delivery_date")
        by_thermal.setdefault(thermal_id, []).append(
            (subindex, delivery_date, position)
        )
    return {thermal_id: tuple(slots) for thermal_id, slots in by_thermal.items()}


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


def _resolve_gnl_targets(
    cuts: BoundaryCuts,
    ring_index: Mapping[int, tuple[tuple[int, int, int], ...]],
    gnl_plan: GnlRingPlan | None,
) -> tuple[dict[int, tuple[int, ...]], tuple[GnlDroppedTerm, ...]]:
    """Resolve each GNL target's ring position(s) and `pi_gnl` columns once.

    Mirrors `_resolve_storage_targets`'s "resolve once, write per record"
    split: cut-invariant, computed exactly once regardless of how many
    records `cuts` carries. Returns `resolved: {ring position -> tuple of
    pi_gnl flat-column indices to sum}` plus every GNL drop.

    Returns `({}, ())` — no GNL mapping at all — when `gnl_plan` is `None`,
    `cuts` has no records, the source deck carries no GNL lag axis
    (`lag_maximo_gnl == 0`), or the boundary's `pi_gnl` is empty (a
    non-GNL deck).

    Raises
    ------
    ValueError
        If `len(cuts.records[0].pi_gnl)` is not a multiple of
        `n_patamares * lag_maximo_gnl` — a reader/plan layout
        inconsistency, named explicitly rather than silently truncated or
        padded.
    """
    n_patamares = cuts.header.n_patamares
    lag_maximo_gnl = cuts.header.lag_maximo_gnl
    if gnl_plan is None or not cuts.records or lag_maximo_gnl == 0:
        return {}, ()
    width = len(cuts.records[0].pi_gnl)
    if width == 0:
        return {}, ()

    block = n_patamares * lag_maximo_gnl
    if width % block != 0:
        raise ValueError(
            f"pi_gnl width {width} is not a multiple of n_patamares "
            f"({n_patamares}) * lag_maximo_gnl ({lag_maximo_gnl}) = {block}"
        )
    n_submercados = width // block

    def col(submercado: int, patamar: int, lag: int) -> int:
        """Flat pi_gnl column for (submercado, patamar, lag), 1-based axes."""
        return ((submercado - 1) * n_patamares + (patamar - 1)) * lag_maximo_gnl + (
            lag - 1
        )

    resolved: dict[int, tuple[int, ...]] = {}
    dropped: list[GnlDroppedTerm] = []
    targeted_submercados: set[int] = set()

    for target in gnl_plan.targets:
        if not (1 <= target.submercado <= n_submercados) or not (
            1 <= target.nl_lag <= lag_maximo_gnl
        ):
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=target.thermal_id,
                    submercado=target.submercado,
                    nl_lag=target.nl_lag,
                    coefficient=0.0,
                    reason="lag/submercado out of pi_gnl range",
                )
            )
            continue
        targeted_submercados.add(target.submercado)
        dated = tuple(
            position
            for _subindex, delivery_date, position in ring_index.get(
                target.thermal_id, ()
            )
            if delivery_date != _DELIVERY_DATE_SENTINEL
        )
        if not dated:
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=target.thermal_id,
                    submercado=target.submercado,
                    nl_lag=target.nl_lag,
                    coefficient=0.0,
                    reason="no dated ring slot for thermal",
                )
            )
            continue
        cols = tuple(
            col(target.submercado, patamar, target.nl_lag)
            for patamar in range(1, n_patamares + 1)
        )
        for position in dated:
            resolved[position] = cols

    # Source-submercado drops (D3-like): a submercado no target claims, but
    # whose pi_gnl carries a nonzero coefficient on some active record, has
    # a source term with no live GNL thermal to receive it.
    active_records = tuple(record for record in cuts.records if record.is_active)
    for submercado in range(1, n_submercados + 1):
        if submercado in targeted_submercados:
            continue
        for lag in range(1, lag_maximo_gnl + 1):
            lag_cols = tuple(
                col(submercado, patamar, lag) for patamar in range(1, n_patamares + 1)
            )
            first_nonzero = next(
                (
                    record
                    for record in active_records
                    if any(record.pi_gnl[c] != 0.0 for c in lag_cols)
                ),
                None,
            )
            if first_nonzero is None:
                continue
            dropped.append(
                GnlDroppedTerm(
                    thermal_id=None,
                    submercado=submercado,
                    nl_lag=lag,
                    coefficient=math.fsum(first_nonzero.pi_gnl[c] for c in lag_cols),
                    reason="no GNL thermal in submercado",
                )
            )

    return resolved, tuple(dropped)


def map_boundary_cuts(
    cuts: BoundaryCuts,
    manifest: TerminalManifest,
    id_map: DecompIdMap,
    *,
    lag_slot_of: Callable[[int], int] = _default_lag_slot_of,
    gnl_plan: GnlRingPlan | None = None,
) -> MappingResult:
    """Map every cut in `cuts.records` onto `manifest`'s state-vector layout.

    Storage terms join by plant code (`HydroStorage`, D3-drop on a
    source-only plant); inflow-lag terms join 1:1 by `lag_slot_of` onto
    `HydroInflowLag`. When `gnl_plan` is given, each `AnticipatedThermalState`
    ring slot named by one of its targets' dated slot(s) carries the
    chain-rule patamar sum `Σ_p pi_gnl[col(s,p,nl_lag)]` (`math.fsum`, order-
    independent); a target with no dated ring slot, or a source submercado
    with no matching target, is dropped and recorded in
    `MappingResult.gnl_dropped`, never folded onto a neighbour.
    `HydroTransitBucket` slots, the sentinel (undated) `AnticipatedThermalState`
    slot, and every ring slot with no resolved target are left at an
    explicit `0.0` regardless of `gnl_plan`; `gnl_plan=None` (the default)
    leaves the entire ring at `0.0`, byte-for-byte matching this function's
    pre-GNL behaviour. `intercept` is the source record's `rhs`, carried
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
        subindex out of range for the manifest's `HydroInflowLag` slots; if
        `gnl_plan` is given and `cuts`' `pi_gnl` width is not a multiple of
        `n_patamares * lag_maximo_gnl`; or if a mapped coefficient vector's
        length disagrees with `manifest.state_dimension`.
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

    gnl_ring_index = _index_gnl_ring(manifest)
    resolved_gnl, gnl_dropped = _resolve_gnl_targets(cuts, gnl_ring_index, gnl_plan)

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

        for gnl_position, gnl_cols in resolved_gnl.items():
            coefficients[gnl_position] = math.fsum(
                record.pi_gnl[column] for column in gnl_cols
            )

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

    return MappingResult(
        cuts=tuple(mapped_cuts), dropped=dropped, gnl_dropped=gnl_dropped
    )
