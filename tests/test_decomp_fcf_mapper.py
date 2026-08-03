"""Tests for the boundary-cut manifest-to-manifest mapper (``fcf/mapper.py``)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
from cobre_bridge.decomp.fcf.cortes import BoundaryCuts, CortesHeader, StageCutRecord
from cobre_bridge.decomp.fcf.mapper import DroppedTerm, map_boundary_cuts
from cobre_bridge.decomp.id_map import DecompIdMap

# cobre `policy.fbs` entity_type codes (see ticket-005's Current State) — a
# stable external contract, restated locally rather than importing the
# mapper module's private constants.
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3


def _slot(
    entity_type: int,
    entity_id: int,
    subindex: int,
    *,
    was_active: bool = True,
    delivery_anchor: int = 0,
) -> dict[str, object]:
    """One hand-authored terminal-manifest slot dict."""
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "subindex": subindex,
        "was_active": was_active,
        "delivery_anchor": delivery_anchor,
    }


def _manifest(slots: Sequence[dict[str, object]]) -> TerminalManifest:
    """A synthetic terminal manifest; `state_dimension` == slot count."""
    return TerminalManifest(entity_manifest=tuple(slots), state_dimension=len(slots))


def _header(plant_codes: tuple[int, ...]) -> CortesHeader:
    """A synthetic source header carrying only `plant_codes` meaningfully."""
    return CortesHeader(
        plant_codes=plant_codes,
        submercado_codes=(1,),
        n_patamares=3,
        lag_maximo_gnl=0,
        n_plants=len(plant_codes),
        individualized=True,
        record_size=32,
        last_cut_record_by_stage=(1,),
    )


def _record(
    *,
    pi_varm: tuple[float, ...],
    pi_qafl: tuple[tuple[float, ...], ...] | None = None,
    rhs: float = 0.0,
    cut_id: int = 1,
    iteration: int = 1,
    forward_pass_index: int = 1,
    is_active: bool = True,
) -> StageCutRecord:
    """A synthetic `StageCutRecord`; `pi_qafl` defaults to all-zero lags."""
    n_plants = len(pi_varm)
    return StageCutRecord(
        cut_id=cut_id,
        iteration=iteration,
        forward_pass_index=forward_pass_index,
        is_active=is_active,
        rhs=rhs,
        pi_varm=pi_varm,
        pi_qafl=(
            pi_qafl
            if pi_qafl is not None
            else tuple((0.0,) * 12 for _ in range(n_plants))
        ),
        pi_gnl=(),
    )


def _cuts(
    plant_codes: tuple[int, ...], records: tuple[StageCutRecord, ...]
) -> BoundaryCuts:
    """A synthetic `BoundaryCuts` over `plant_codes`, boundary_stage arbitrary."""
    return BoundaryCuts(header=_header(plant_codes), boundary_stage=5, records=records)


def _id_map(hydro_codes: tuple[int, ...]) -> DecompIdMap:
    """A minimal `DecompIdMap` carrying only the hydro-code join surface."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=hydro_codes)


def test_map_storage_places_pi_varm_at_hydro_slots() -> None:
    id_map = _id_map((10, 20))
    # Plant 10 (hydro_id 0) carries a full-width HydroInflowLag family
    # (subindices 0..11) so the default identity `lag_slot_of` (d -> d-1)
    # stays in range for every calendar depth 1..12 (all zero here, per
    # "lags zero" in the AC); plant 20 has no lag exposure in this fixture.
    manifest = _manifest(
        [
            _slot(_HYDRO_STORAGE, 0, 0),  # position 0: plant 10's storage
            _slot(_HYDRO_STORAGE, 1, 0),  # position 1: plant 20's storage
            *[
                _slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 2..13
                for lag in range(12)
            ],
            _slot(_HYDRO_TRANSIT_BUCKET, 1, 2),  # position 14
        ]
    )
    cuts = _cuts((10, 20), (_record(pi_varm=(3.0, 5.0), rhs=100.0),))

    result = map_boundary_cuts(cuts, manifest, id_map)

    assert result.dropped == ()
    assert len(result.cuts) == 1
    mapped = result.cuts[0]
    assert len(mapped.coefficients) == manifest.state_dimension
    assert mapped.coefficients[0] == 3.0
    assert mapped.coefficients[1] == 5.0
    assert mapped.coefficients[2:] == (0.0,) * 13
    assert mapped.intercept == 100.0


def test_map_drops_source_only_plant_with_diagnostic_record() -> None:
    # Plant 20 is present in the source but unknown to the target id_map.
    id_map = _id_map((10,))
    manifest = _manifest([_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = _cuts((10, 20), (_record(pi_varm=(3.0, 7.0)),))

    result = map_boundary_cuts(cuts, manifest, id_map)

    assert result.dropped == (DroppedTerm(plant_code=20, beta=7.0),)
    assert len(result.cuts) == 1
    assert result.cuts[0].coefficients == (3.0,)


def test_map_lags_one_to_one() -> None:
    id_map = _id_map((10,))
    # Full-width HydroInflowLag family (subindices 0..11) so identity
    # `lag_slot_of` stays in range for every calendar depth 1..12; only
    # depths 1 and 2 carry a nonzero source coefficient.
    manifest = _manifest(
        [
            _slot(_HYDRO_STORAGE, 0, 0),  # position 0
            *[
                _slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 1..12
                for lag in range(12)
            ],
        ]
    )
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = _cuts((10,), (_record(pi_varm=(0.0,), pi_qafl=(lags,)),))

    result = map_boundary_cuts(cuts, manifest, id_map, lag_slot_of=lambda d: d - 1)

    assert result.dropped == ()
    mapped = result.cuts[0]
    assert mapped.coefficients[1] == 1.0  # lag depth 1 -> subindex 0
    assert mapped.coefficients[2] == 2.0  # lag depth 2 -> subindex 1


def test_map_zeroes_buckets_and_gnl_ring() -> None:
    id_map = _id_map((10,))
    manifest = _manifest(
        [
            _slot(_HYDRO_STORAGE, 0, 0),  # position 0
            _slot(_HYDRO_TRANSIT_BUCKET, 0, 5),  # position 1
            _slot(_ANTICIPATED_THERMAL_STATE, 0, 0),  # position 2
        ]
    )
    cuts = _cuts((10,), (_record(pi_varm=(9.0,)),))

    result = map_boundary_cuts(cuts, manifest, id_map)

    for mapped in result.cuts:
        assert mapped.coefficients[1] == 0.0
        assert mapped.coefficients[2] == 0.0
    assert result.cuts[0].coefficients[0] == 9.0


def test_map_rejects_out_of_range_lag_slot() -> None:
    id_map = _id_map((10,))
    # Only one HydroInflowLag slot exists (subindex 0) -> valid range is {0}.
    manifest = _manifest(
        [
            _slot(_HYDRO_STORAGE, 0, 0),
            _slot(_HYDRO_INFLOW_LAG, 0, 0),
        ]
    )
    cuts = _cuts((10,), (_record(pi_varm=(1.0,)),))

    with pytest.raises(ValueError, match="out of range"):
        map_boundary_cuts(cuts, manifest, id_map, lag_slot_of=lambda d: d)
