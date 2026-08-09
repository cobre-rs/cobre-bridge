"""Tier-1 tests for the bounds-assembly accumulator.

No deck (synthetic contributions only). Covers the axis registry (contents,
two-sided/stage-level flags — including the diversion/spillage axes
epic-06/ticket-021+022 widened/added — unknown-pair raise), the per-axis
``intersect`` (tightest pair, unbounded sentinel, empty-intersection and
upper-only raises), ``resolve`` (base vs per-block materialization + the
stage-level guard), and ``build_bound_tables`` (the pyarrow fan-out into the
hydro/thermal/pumping schemas, one row per cell — including a two-sided
diversion/spillage group landing both endpoints in one cell).
"""

from __future__ import annotations

import pytest

from cobre_bridge.decomp.bounds_accumulator import (
    AXES,
    AxisSpec,
    BoundContribution,
    ResolvedRow,
    axis_spec,
    build_bound_tables,
    intersect,
    resolve,
)


def test_storage_axis_is_stage_level() -> None:
    spec = axis_spec("hydro", "storage")
    assert spec.block_eligible is False
    assert spec.lower_column == "min_storage_hm3"
    assert spec.upper_column == "max_storage_hm3"


def test_diversion_axis_is_two_sided_block_eligible() -> None:
    """AC1 (ticket-021): diversion widened from upper-only to two-sided —
    cobre's generic-constraint-authoring epic-01 landed ``min_diversion_m3s``
    alongside the pre-existing ``max_diversion_m3s``."""
    spec = axis_spec("hydro", "diversion")
    assert spec.lower_column == "min_diversion_m3s"
    assert spec.upper_column == "max_diversion_m3s"
    assert spec.block_eligible is True


def test_spillage_axis_is_two_sided_block_eligible() -> None:
    """AC4 (ticket-021) / the axis ticket-022 routes into: a brand-new
    two-sided, block-eligible ``spillage`` axis."""
    spec = axis_spec("hydro", "spillage")
    assert spec.lower_column == "min_spillage_m3s"
    assert spec.upper_column == "max_spillage_m3s"
    assert spec.block_eligible is True


def test_thermal_and_pumping_axes_registered() -> None:
    spec = axis_spec("thermal", "generation")
    assert spec.family == "thermal"
    assert spec.block_eligible is True
    assert ("pumping", "flow") in AXES


def test_unknown_axis_raises() -> None:
    with pytest.raises(ValueError, match="hydro.*bogus"):
        axis_spec("hydro", "bogus")


def test_hydro_turbined_and_outflow_axes_are_two_sided_block_eligible() -> None:
    for axis in ("turbined", "outflow", "generation", "diversion", "spillage"):
        spec = axis_spec("hydro", axis)
        assert spec.lower_column is not None
        assert spec.upper_column is not None
        assert spec.block_eligible is True


def test_bound_contribution_is_frozen_plain_record() -> None:
    contribution = BoundContribution(
        family="hydro",
        entity_id=3,
        stage_id=1,
        block_id=None,
        axis="turbined",
        lower=0.0,
        upper=120.5,
        contributor="RQ",
    )
    assert contribution.family == "hydro"
    assert contribution.block_id is None
    with pytest.raises(AttributeError):
        contribution.upper = 130.0  # type: ignore[misc]


def test_axis_spec_is_frozen() -> None:
    spec = AxisSpec(
        family="hydro",
        lower_column="min_x",
        upper_column="max_x",
        block_eligible=True,
    )
    with pytest.raises(AttributeError):
        spec.block_eligible = False  # type: ignore[misc]


def _outflow_contribution(
    *, lower: float | None, upper: float | None, contributor: str
) -> BoundContribution:
    return BoundContribution(
        family="hydro",
        entity_id=7,
        stage_id=2,
        block_id=None,
        axis="outflow",
        lower=lower,
        upper=upper,
        contributor=contributor,
    )


def test_intersect_tightest_pair() -> None:
    spec = axis_spec("hydro", "outflow")
    contribs = [
        _outflow_contribution(lower=10.0, upper=100.0, contributor="RQ"),
        _outflow_contribution(lower=20.0, upper=80.0, contributor="RE_12"),
    ]
    assert intersect(contribs, spec) == (20.0, 80.0)


def test_intersect_ignores_unbounded_sentinel() -> None:
    spec = axis_spec("hydro", "outflow")
    contribs = [
        _outflow_contribution(lower=None, upper=1e21, contributor="RQ"),
        _outflow_contribution(lower=None, upper=50.0, contributor="HQ_164"),
    ]
    assert intersect(contribs, spec) == (None, 50.0)


def test_empty_intersection_raises() -> None:
    spec = axis_spec("hydro", "outflow")
    contribs = [
        _outflow_contribution(lower=90.0, upper=None, contributor="RQ"),
        _outflow_contribution(lower=None, upper=50.0, contributor="HQ_164"),
    ]
    with pytest.raises(ValueError, match="RQ.*HQ_164"):
        intersect(contribs, spec)


def test_lower_on_upper_only_axis_raises() -> None:
    """No axis in the live :data:`AXES` registry is upper-only any more
    (diversion widened in ticket-021), but ``intersect`` must still enforce
    the guard for a hypothetical one — built directly, not looked up via
    :func:`axis_spec`, mirroring ``test_axis_spec_is_frozen``'s approach."""
    spec = AxisSpec(
        family="hydro",
        lower_column=None,
        upper_column="max_x",
        block_eligible=True,
    )
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=1,
            block_id=None,
            axis="x",
            lower=5.0,
            upper=None,
            contributor="RD_9",
        )
    ]
    with pytest.raises(ValueError, match="upper-only"):
        intersect(contribs, spec)


def test_all_base_single_row() -> None:
    contribs = [
        _outflow_contribution(lower=10.0, upper=100.0, contributor="RQ"),
        _outflow_contribution(lower=20.0, upper=80.0, contributor="RE_12"),
    ]
    rows = resolve(contribs, {0: 3})
    assert len(rows) == 1
    row = rows[0]
    assert row.block_id is None
    assert row.family == "hydro"
    assert row.axis == "outflow"
    assert (row.lower, row.upper) == (20.0, 80.0)


def test_per_block_materializes_all_blocks() -> None:
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=7,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=10.0,
            upper=None,
            contributor="RQ",
        ),
        BoundContribution(
            family="hydro",
            entity_id=7,
            stage_id=0,
            block_id=1,
            axis="outflow",
            lower=20.0,
            upper=None,
            contributor="RQ_block1",
        ),
    ]
    rows = resolve(contribs, {0: 3})
    by_block = {row.block_id: row for row in rows}
    assert set(by_block) == {0, 1, 2}
    assert by_block[1].lower == 20.0
    assert by_block[0].lower == 10.0
    assert by_block[2].lower == 10.0
    assert None not in by_block


def test_block_id_on_stage_level_raises() -> None:
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=4,
            stage_id=0,
            block_id=1,
            axis="storage",
            lower=None,
            upper=500.0,
            contributor="VOLMAX",
        ),
    ]
    with pytest.raises(ValueError, match="storage.*entity_id=4"):
        resolve(contribs, {0: 3})


def test_out_of_range_block_id_raises() -> None:
    # block_id 5 is outside the stage's 3-block range; without a guard it would
    # match no materialization iteration and the bound would be silently lost.
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=7,
            stage_id=0,
            block_id=5,
            axis="outflow",
            lower=30.0,
            upper=None,
            contributor="HQ_99",
        ),
    ]
    with pytest.raises(ValueError, match=r"out of range.*entity_id=7"):
        resolve(contribs, {0: 3})


def test_empty_group_skipped() -> None:
    contribs = [
        _outflow_contribution(lower=None, upper=1e21, contributor="RQ"),
    ]
    assert resolve(contribs, {2: 3}) == []


def test_single_axis_row() -> None:
    row = ResolvedRow(
        family="hydro",
        entity_id=3,
        stage_id=0,
        block_id=None,
        axis="outflow",
        lower=10.0,
        upper=100.0,
    )
    tables = build_bound_tables([row])

    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["hydro_id"] == 3
    assert record["stage_id"] == 0
    assert record["block_id"] is None
    assert record["min_outflow_m3s"] == 10.0
    assert record["max_outflow_m3s"] == 100.0
    other_axis_columns = {
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_generation_mw",
        "max_generation_mw",
        "min_storage_hm3",
        "max_storage_hm3",
        "min_diversion_m3s",
        "max_diversion_m3s",
        "min_spillage_m3s",
        "max_spillage_m3s",
    }
    for column in other_axis_columns:
        assert record[column] is None


def test_two_axes_collapse_to_one_row() -> None:
    rows = [
        ResolvedRow(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=10.0,
            upper=100.0,
        ),
        ResolvedRow(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="turbined",
            lower=5.0,
            upper=None,
        ),
    ]
    tables = build_bound_tables(rows)

    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["min_outflow_m3s"] == 10.0
    assert record["max_outflow_m3s"] == 100.0
    assert record["min_turbined_m3s"] == 5.0
    assert record["max_turbined_m3s"] is None


def test_thermal_table_isolated() -> None:
    row = ResolvedRow(
        family="thermal",
        entity_id=2,
        stage_id=1,
        block_id=None,
        axis="generation",
        lower=50.0,
        upper=500.0,
    )
    tables = build_bound_tables([row])

    assert tables.thermal.num_rows == 1
    record = tables.thermal.to_pylist()[0]
    assert record["thermal_id"] == 2
    assert record["min_generation_mw"] == 50.0
    assert record["max_generation_mw"] == 500.0
    assert tables.hydro.num_rows == 0
    assert tables.pumping.num_rows == 0


def test_diversion_upper_only_column() -> None:
    """AC2 (ticket-021), second half: an upper-only diversion group still
    emits ``max_diversion_m3s`` alone — ``min_diversion_m3s`` is a real,
    nullable column in the widened schema, but stays ``null`` rather than
    picking up a spurious value."""
    row = ResolvedRow(
        family="hydro",
        entity_id=1,
        stage_id=0,
        block_id=None,
        axis="diversion",
        lower=None,
        upper=40.0,
    )
    tables = build_bound_tables([row])

    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["max_diversion_m3s"] == 40.0
    assert record["min_diversion_m3s"] is None


def test_diversion_two_sided_group_resolves_and_fans_out_both_columns() -> None:
    """AC2 (ticket-021), first half: a two-sided diversion group — one
    contributor supplying only the lower, another only the upper — resolves
    through ``resolve`` to one row and fans out through
    ``build_bound_tables`` into ``min_diversion_m3s``/``max_diversion_m3s``
    on the *same* hydro cell."""
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="diversion",
            lower=10.0,
            upper=None,
            contributor="HQ_1",
        ),
        BoundContribution(
            family="hydro",
            entity_id=3,
            stage_id=0,
            block_id=None,
            axis="diversion",
            lower=None,
            upper=40.0,
            contributor="HQ_2",
        ),
    ]
    rows = resolve(contribs, {0: 1})
    assert len(rows) == 1
    assert (rows[0].lower, rows[0].upper) == (10.0, 40.0)

    tables = build_bound_tables(rows)
    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["min_diversion_m3s"] == 10.0
    assert record["max_diversion_m3s"] == 40.0


def test_spillage_two_sided_group_resolves_and_fans_out_both_columns() -> None:
    """AC3 (ticket-022): the resolved spillage group lands
    ``min_spillage_m3s``/``max_spillage_m3s`` in the same hydro cell,
    reusing this ticket's spillage axis."""
    contribs = [
        BoundContribution(
            family="hydro",
            entity_id=2,
            stage_id=0,
            block_id=None,
            axis="spillage",
            lower=5.0,
            upper=None,
            contributor="HQ_9",
        ),
        BoundContribution(
            family="hydro",
            entity_id=2,
            stage_id=0,
            block_id=None,
            axis="spillage",
            lower=None,
            upper=60.0,
            contributor="HQ_10",
        ),
    ]
    rows = resolve(contribs, {0: 1})
    assert len(rows) == 1
    assert (rows[0].lower, rows[0].upper) == (5.0, 60.0)

    tables = build_bound_tables(rows)
    assert tables.hydro.num_rows == 1
    record = tables.hydro.to_pylist()[0]
    assert record["min_spillage_m3s"] == 5.0
    assert record["max_spillage_m3s"] == 60.0


def test_spillage_upper_only_row_leaves_min_null() -> None:
    row = ResolvedRow(
        family="hydro",
        entity_id=1,
        stage_id=0,
        block_id=None,
        axis="spillage",
        lower=None,
        upper=25.0,
    )
    tables = build_bound_tables([row])

    record = tables.hydro.to_pylist()[0]
    assert record["max_spillage_m3s"] == 25.0
    assert record["min_spillage_m3s"] is None
