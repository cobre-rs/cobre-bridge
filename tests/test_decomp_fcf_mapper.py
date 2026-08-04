"""Tests for the boundary-cut manifest-to-manifest mapper (``fcf/mapper.py``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from cobre_bridge.decomp.fcf.mapper import DroppedTerm, map_boundary_cuts
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import requires_cobre_python

# cobre `policy.fbs` entity_type codes (see ticket-005's Current State) — a
# stable external contract, restated locally rather than importing the
# mapper module's private constants.
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_ANTICIPATED_THERMAL_STATE = 2
_HYDRO_TRANSIT_BUCKET = 3


def test_map_storage_places_pi_varm_at_hydro_slots() -> None:
    id_map = make_id_map((10, 20))
    # Plant 10 (hydro_id 0) carries a full-width HydroInflowLag family
    # (subindices 0..11) so the default identity `lag_slot_of` (d -> d-1)
    # stays in range for every calendar depth 1..12 (all zero here, per
    # "lags zero" in the AC); plant 20 has no lag exposure in this fixture.
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0: plant 10's storage
            make_slot(_HYDRO_STORAGE, 1, 0),  # position 1: plant 20's storage
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 2..13
                for lag in range(12)
            ],
            make_slot(_HYDRO_TRANSIT_BUCKET, 1, 2),  # position 14
        ]
    )
    cuts = make_boundary_cuts(
        (10, 20), (make_cut_record(pi_varm=(3.0, 5.0), rhs=100.0),)
    )

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
    id_map = make_id_map((10,))
    manifest = make_manifest([make_slot(_HYDRO_STORAGE, 0, 0)])
    cuts = make_boundary_cuts((10, 20), (make_cut_record(pi_varm=(3.0, 7.0)),))

    result = map_boundary_cuts(cuts, manifest, id_map)

    assert result.dropped == (DroppedTerm(plant_code=20, beta=7.0),)
    assert len(result.cuts) == 1
    assert result.cuts[0].coefficients == (3.0,)


def test_map_lags_one_to_one() -> None:
    id_map = make_id_map((10,))
    # Full-width HydroInflowLag family (subindices 0..11) so identity
    # `lag_slot_of` stays in range for every calendar depth 1..12; only
    # depths 1 and 2 carry a nonzero source coefficient.
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 1..12
                for lag in range(12)
            ],
        ]
    )
    lags = (1.0, 2.0) + (0.0,) * 10
    cuts = make_boundary_cuts(
        (10,), (make_cut_record(pi_varm=(0.0,), pi_qafl=(lags,)),)
    )

    result = map_boundary_cuts(cuts, manifest, id_map, lag_slot_of=lambda d: d - 1)

    assert result.dropped == ()
    mapped = result.cuts[0]
    assert mapped.coefficients[1] == 1.0  # lag depth 1 -> subindex 0
    assert mapped.coefficients[2] == 2.0  # lag depth 2 -> subindex 1


def test_map_zeroes_buckets_and_gnl_ring() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            make_slot(_HYDRO_TRANSIT_BUCKET, 0, 5),  # position 1
            make_slot(_ANTICIPATED_THERMAL_STATE, 0, 0),  # position 2
        ]
    )
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(9.0,)),))

    result = map_boundary_cuts(cuts, manifest, id_map)

    for mapped in result.cuts:
        assert mapped.coefficients[1] == 0.0
        assert mapped.coefficients[2] == 0.0
    assert result.cuts[0].coefficients[0] == 9.0


def test_map_rejects_out_of_range_lag_slot() -> None:
    id_map = make_id_map((10,))
    # Only one HydroInflowLag slot exists (subindex 0) -> valid range is {0}.
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_HYDRO_INFLOW_LAG, 0, 0),
        ]
    )
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(1.0,)),))

    with pytest.raises(ValueError, match="out of range"):
        map_boundary_cuts(cuts, manifest, id_map, lag_slot_of=lambda d: d)


@requires_cobre_python
def test_synthetic_roundtrip_preserves_coeffs(tmp_path: Path) -> None:
    """The mapper's storage + lag-depth-1 placement survives a real
    map -> write -> load_policy round trip, with no deck and no cobre
    binary — the reloaded cut's intercept and leading coefficients match
    the synthetic source record verbatim.
    """
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 1..12
                for lag in range(12)
            ],
        ]
    )
    record = make_cut_record(
        pi_varm=(2.5,),
        pi_qafl=((0.75,) + (0.0,) * 11,),
        rhs=42.0,
    )
    cuts = make_boundary_cuts((10,), (record,))

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)

    reloaded_cut = reloaded["stage_cuts"][0]["cuts"][0]
    assert reloaded_cut["intercept"] == record.rhs
    assert reloaded_cut["coefficients"][0:3] == [2.5, 0.75, 0.0]
