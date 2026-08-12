"""Tests for the boundary-cut manifest-to-manifest mapper (``fcf/mapper.py``)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
from cobre_bridge.decomp.fcf.cortes import BoundaryCuts
from cobre_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    map_boundary_cuts,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cortes_header,
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


def _gnl_row(width: int, nonzero: Mapping[int, float]) -> tuple[float, ...]:
    """A `pi_gnl` flat vector of `width` zeros with `nonzero` columns set."""
    row = [0.0] * width
    for column, value in nonzero.items():
        row[column] = value
    return tuple(row)


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


def _make_gnl_ring_fixture(
    pi_gnl: tuple[float, ...],
    *,
    post_horizon_start: int | None = None,
) -> tuple[BoundaryCuts, TerminalManifest, DecompIdMap, GnlRingPlan]:
    """The READBACK-shaped ring (thermal 94: sentinel + dated; thermal 95:
    dated; thermal 96: dated but untargeted) over a `P=3, L=2, S=4` GNL
    block, plus the `GnlRingPlan` targeting 94 (submercado 1, lag 2) and 95
    (submercado 3, lag 1). No source hydro plants — the GNL placement is
    independent of the storage/lag families, only their manifest guard
    (`>= 1 HydroStorage slot`) must be satisfied.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0: unrelated dummy
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # position 1: sentinel
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 94, 1, delivery_date=20260501
            ),  # position 2: dated
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 95, 0, delivery_date=20260401
            ),  # position 3: dated
            make_slot(
                _ANTICIPATED_THERMAL_STATE, 96, 0, delivery_date=20260601
            ),  # position 4: dated but untargeted by the plan
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1, 2, 3, 4)
    )
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan(
        (GnlThermalTarget(94, 1, 2), GnlThermalTarget(95, 3, 1)),
        post_horizon_start=post_horizon_start,
    )
    return cuts, manifest, id_map, plan


def test_map_gnl_places_chain_rule_sum_on_dated_slots() -> None:
    # col(1,p,2) for p=1..3 -> flat indices 1, 3, 5; col(3,p,1) -> 12, 14, 16
    # (P=3, L=2): col(s,p,l) = ((s-1)*3 + (p-1))*2 + (l-1).
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6)  # thermal 94, dated
    assert mapped.coefficients[3] == pytest.approx(7.0)  # thermal 95, dated


def test_map_gnl_covered_lane_populated_uncovered_lane_dropped() -> None:
    """Ticket-013 AC 1 — the empirical `mar-26-rv2` READBACK shape: thermal
    94's `20260501` dated slot is covered (`>= post_horizon_start`) and
    keeps the chain-rule sum; thermal 95's `20260401` dated slot is
    *before* the post-study horizon (non-covered) and is dropped, staying
    at `0.0`, with a `GnlDroppedTerm` naming the post-study horizon —
    exactly the ring shape that made `cobre run` reject the boundary
    (thermal 95, delivery `20260401`, before horizon `20260501`).
    """
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(
        pi_gnl, post_horizon_start=20260501
    )

    result = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6)  # thermal 94, covered
    assert mapped.coefficients[3] == 0.0  # thermal 95, uncovered -> dropped

    matches = [
        term
        for term in result.gnl_dropped
        if term.thermal_id == 95
        and term.submercado == 3
        and term.nl_lag == 1
        and "post-study horizon" in term.reason
    ]
    assert len(matches) == 1
    assert matches[0].coefficient == pytest.approx(7.0)
    # The "all dated slots uncovered" case is the uncovered-drop above, not
    # the pre-existing "no dated ring slot for thermal" reason (that one is
    # reserved for a target with zero dated slots at all).
    assert not any(
        term.thermal_id == 95 and "no dated ring slot" in term.reason
        for term in result.gnl_dropped
    )


def test_map_gnl_post_horizon_start_none_is_old_behavior() -> None:
    """Ticket-013 AC 2 — `post_horizon_start=None` (the default) disables
    the covered-lane filter entirely: both dated slots populate exactly as
    ticket-009's pre-ticket-013 behavior, and no covered-lane drop fires.
    """
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(
        pi_gnl, post_horizon_start=None
    )

    result = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)

    mapped = result.cuts[0]
    assert mapped.coefficients[2] == pytest.approx(0.6)  # thermal 94, dated
    assert mapped.coefficients[3] == pytest.approx(7.0)  # thermal 95, dated
    assert not any("post-study horizon" in term.reason for term in result.gnl_dropped)


def test_map_gnl_sentinel_and_nontarget_slots_stay_zero() -> None:
    pi_gnl = _gnl_row(24, {1: 0.1, 3: 0.2, 5: 0.3, 12: 1.0, 14: 2.0, 16: 4.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)

    mapped = result.cuts[0]
    assert mapped.coefficients[1] == 0.0  # thermal 94, sentinel slot
    assert mapped.coefficients[4] == 0.0  # thermal 96, absent from the plan


def test_map_gnl_drops_submercado_without_thermal() -> None:
    # col(2,p,1) for p=1..3 -> flat indices 6, 8, 10 (submercado 2, lag 1);
    # neither target claims submercado 2, and submercados 1/3's own targeted
    # columns are all zero here.
    pi_gnl = _gnl_row(24, {6: 10.0, 8: 20.0, 10: 40.0})
    cuts, manifest, id_map, plan = _make_gnl_ring_fixture(pi_gnl)

    result = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)

    matches = [
        term
        for term in result.gnl_dropped
        if term.thermal_id is None
        and term.submercado == 2
        and term.nl_lag == 1
        and term.reason == "no GNL thermal in submercado"
    ]
    assert len(matches) == 1
    assert matches[0].coefficient == pytest.approx(70.0)
    # Neither targeted thermal's dated slot was altered by the drop.
    assert result.cuts[0].coefficients[2] == 0.0
    assert result.cuts[0].coefficients[3] == 0.0


def test_map_gnl_drops_target_with_no_dated_slot() -> None:
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0),  # sentinel only
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=1, n_patamares=1, submercado_codes=(1,)
    )
    record = make_cut_record(pi_varm=(), pi_gnl=(5.0,))
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan((GnlThermalTarget(94, 1, 1),))

    result = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)

    assert any(
        term.thermal_id == 94 and "no dated ring slot" in term.reason
        for term in result.gnl_dropped
    )
    assert result.cuts[0].coefficients[1] == 0.0  # sentinel slot untouched


def test_map_gnl_plan_none_is_noop() -> None:
    id_map = make_id_map((10,))
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, delivery_date=20260501),
        ]
    )
    cuts = make_boundary_cuts((10,), (make_cut_record(pi_varm=(9.0,)),))

    result = map_boundary_cuts(cuts, manifest, id_map)

    assert result.gnl_dropped == ()
    assert result.cuts[0].coefficients[0] == 9.0
    assert result.cuts[0].coefficients[1] == 0.0


def test_map_gnl_rejects_bad_pi_gnl_width() -> None:
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, delivery_date=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    # n_patamares * lag_maximo_gnl == 6; width 7 is not a multiple of it.
    record = make_cut_record(pi_varm=(), pi_gnl=(0.0,) * 7)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    plan = GnlRingPlan((GnlThermalTarget(94, 1, 1),))

    with pytest.raises(ValueError, match="pi_gnl width"):
        map_boundary_cuts(cuts, manifest, id_map, gnl_plan=plan)
