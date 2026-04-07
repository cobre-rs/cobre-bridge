"""Unit tests for cobre_bridge.dashboard.tabs.constraints.

Covers module constants, can_render, _build_metrics_row, the new
_compute_violation_zones, _build_constraint_lhs_data, _build_lhs_section,
_add_type_filter_and_row_attrs helpers, and the full render() path using
MagicMock data with real polars/pandas objects for fields that get accessed
as LazyFrames/DataFrames.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pandas as pd
import polars as pl

import cobre_bridge.dashboard.tabs.constraints as tab_constraints
from cobre_bridge.dashboard.tabs.constraints import (
    _build_constraint_lhs_data,
    _build_lhs_section,
    _build_metrics_row,
    _compute_violation_zones,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_constraints(n: int = 2, ctype: str = "VminOP") -> list[dict]:
    """Return a list of minimal constraint dicts."""
    return [
        {
            "id": i,
            "name": f"{ctype}_constraint_{i}",
            "expression": f"hydro_storage({i})",
            "sense": ">=",
            "slack": {"enabled": False},
        }
        for i in range(n)
    ]


def _make_gc_bounds(constraint_ids: list[int], n_stages: int = 2) -> pd.DataFrame:
    """Return a minimal gc_bounds DataFrame."""
    rows = []
    for cid in constraint_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "constraint_id": cid,
                    "stage_id": stage_id,
                    "block_id": float("nan"),
                    "bound": 100.0,
                }
            )
    return pd.DataFrame(rows)


def _make_gc_violations(
    constraint_ids: list[int] | None = None,
    slack_value: float = 0.0,
) -> pd.DataFrame:
    """Return a minimal gc_violations DataFrame.

    When *constraint_ids* is None, returns an empty DataFrame.
    """
    if constraint_ids is None:
        return pd.DataFrame(columns=["constraint_id", "slack_value"])
    rows = [
        {"constraint_id": cid, "slack_value": slack_value} for cid in constraint_ids
    ]
    return pd.DataFrame(rows)


def _make_costs(
    n_scenarios: int = 1,
    n_stages: int = 2,
    include_violation_cost: bool = False,
    violation_cost: float = 0.0,
) -> pd.DataFrame:
    """Return a minimal costs DataFrame."""
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            row: dict = {
                "scenario_id": scenario_id,
                "stage_id": stage_id,
                "thermal_cost": 500.0,
            }
            if include_violation_cost:
                row["generic_violation_cost"] = violation_cost
            rows.append(row)
    return pd.DataFrame(rows)


def _make_lhs_df(
    constraint_ids: list[int],
    n_scenarios: int = 2,
    n_stages: int = 3,
    lhs_value: float = 120.0,
) -> pd.DataFrame:
    """Return a small LHS DataFrame with uniform lhs_value across all rows."""
    rows = []
    for cid in constraint_ids:
        for sid in range(n_scenarios):
            for stg in range(n_stages):
                rows.append(
                    {
                        "constraint_id": cid,
                        "scenario_id": sid,
                        "stage_id": stg,
                        "block_id": 0,
                        "lhs_value": lhs_value,
                    }
                )
    return pd.DataFrame(rows)


def _make_mock_data(
    *,
    constraints: list[dict] | None = None,
    gc_bounds: pd.DataFrame | None = None,
    gc_violations: pd.DataFrame | None = None,
    costs: pd.DataFrame | None = None,
    n_stages: int = 2,
) -> MagicMock:
    """Build a minimal MagicMock satisfying the DashboardData interface.

    Sets real polars LazyFrames for hydros_lf and exchanges_lf to prevent
    MagicMock auto-chaining from causing OOM.
    """
    data = MagicMock()
    data.gc_constraints = (
        constraints if constraints is not None else _make_constraints()
    )
    data.gc_bounds = (
        gc_bounds
        if gc_bounds is not None
        else _make_gc_bounds([c["id"] for c in data.gc_constraints], n_stages)
    )
    data.gc_violations = (
        gc_violations if gc_violations is not None else _make_gc_violations()
    )
    data.costs = costs if costs is not None else _make_costs(n_stages=n_stages)
    data.stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
    data.names = {}
    # Real empty LazyFrames prevent MagicMock auto-chaining OOM
    data.hydros_lf = pl.LazyFrame()
    data.exchanges_lf = pl.LazyFrame()
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match the ticket specification exactly."""
    assert tab_constraints.TAB_ID == "tab-constraints"
    assert tab_constraints.TAB_LABEL == "Constraints"
    assert tab_constraints.TAB_ORDER == 80


# ---------------------------------------------------------------------------
# test_can_render
# ---------------------------------------------------------------------------


def test_can_render_false_when_no_constraints() -> None:
    """can_render must return False when gc_constraints is empty."""
    data = _make_mock_data(constraints=[])
    assert can_render(data) is False


def test_can_render_true_when_constraints_present() -> None:
    """can_render must return True when gc_constraints contains 1+ dicts."""
    data = _make_mock_data(constraints=_make_constraints(n=1))
    assert can_render(data) is True


def test_can_render_true_with_multiple_constraints() -> None:
    """can_render must return True when gc_constraints contains multiple dicts."""
    data = _make_mock_data(constraints=_make_constraints(n=3))
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test__build_metrics_row
# ---------------------------------------------------------------------------


def test_build_metrics_row_produces_metrics_grid() -> None:
    """_build_metrics_row must produce HTML with 'metrics-grid' class."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "metrics-grid" in html


def test_build_metrics_row_produces_four_metric_cards() -> None:
    """_build_metrics_row must produce HTML with 'metric-card' at least 4 times."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 4


def test_build_metrics_row_shows_constraint_count() -> None:
    """_build_metrics_row must display the correct total constraint count."""
    constraints = _make_constraints(n=3)
    data = _make_mock_data(constraints=constraints)
    html = _build_metrics_row(data)
    assert ">3<" in html


def test_build_metrics_row_with_violations() -> None:
    """_build_metrics_row must show violated count > 0 when violations present.

    When one constraint has a nonzero slack_value, the violated count must be 1.
    """
    constraints = _make_constraints(n=2)
    gc_violations = _make_gc_violations(
        constraint_ids=[constraints[0]["id"]], slack_value=5.0
    )
    data = _make_mock_data(constraints=constraints, gc_violations=gc_violations)
    html = _build_metrics_row(data)
    # 1 violated constraint should appear as ">1<"
    assert ">1<" in html
    assert "Constraints with Violations" in html


def test_build_metrics_row_no_violation_cost_column() -> None:
    """_build_metrics_row must show R$ 0 when violation cost column is absent."""
    costs = _make_costs(include_violation_cost=False)
    data = _make_mock_data(costs=costs)
    html = _build_metrics_row(data)
    assert "R$ 0" in html
    assert "Total Violation Cost" in html


def test_build_metrics_row_with_violation_cost() -> None:
    """_build_metrics_row must display the summed violation cost when column present."""
    # 2 scenarios x 2 stages x 100.0 = 400.0 total
    costs = _make_costs(
        n_scenarios=2,
        n_stages=2,
        include_violation_cost=True,
        violation_cost=100.0,
    )
    data = _make_mock_data(costs=costs)
    html = _build_metrics_row(data)
    assert "R$ 400" in html


def test_build_metrics_row_shows_active_types() -> None:
    """_build_metrics_row must include unique type prefixes from constraint names."""
    constraints = [
        {
            "id": 0,
            "name": "VminOP_c1",
            "expression": "hydro_storage(0)",
            "sense": ">=",
            "slack": {},
        },
        {
            "id": 1,
            "name": "RE_c2",
            "expression": "hydro_storage(1)",
            "sense": "<=",
            "slack": {},
        },
    ]
    data = _make_mock_data(constraints=constraints)
    html = _build_metrics_row(data)
    assert "VminOP" in html
    assert "RE" in html
    assert "Active Types" in html


def test_build_metrics_row_empty_violations_shows_zero_violated() -> None:
    """_build_metrics_row must show 0 violated constraints when violations is empty."""
    data = _make_mock_data(gc_violations=_make_gc_violations(constraint_ids=None))
    html = _build_metrics_row(data)
    assert ">0<" in html


# ---------------------------------------------------------------------------
# test__compute_violation_zones
# ---------------------------------------------------------------------------


def test_compute_violation_zones_ge_single_dip() -> None:
    """>=: violation at stage index 1 where p10=80 < bound=85."""
    p10 = [100.0, 80.0, 90.0]
    p90 = [110.0, 95.0, 105.0]
    bound = [85.0, 85.0, 85.0]
    result = _compute_violation_zones(p10, p90, bound, ">=")
    assert result == [{"start": 1, "end": 1}]


def test_compute_violation_zones_le_single_spike() -> None:
    """<=: violation at stage index 1 where p90=120 > bound=110."""
    p10 = [90.0, 90.0, 90.0]
    p90 = [100.0, 120.0, 90.0]
    bound = [110.0, 110.0, 110.0]
    result = _compute_violation_zones(p10, p90, bound, "<=")
    assert result == [{"start": 1, "end": 1}]


def test_compute_violation_zones_no_violations() -> None:
    """No violations when band never crosses bound."""
    p10 = [100.0, 100.0, 100.0]
    p90 = [110.0, 110.0, 110.0]
    bound = [85.0, 85.0, 85.0]
    result = _compute_violation_zones(p10, p90, bound, ">=")
    assert result == []


def test_compute_violation_zones_nan_bound_skipped() -> None:
    """Stages with NaN bound must not be counted as violations."""
    p10 = [80.0, 80.0, 80.0]
    p90 = [90.0, 90.0, 90.0]
    bound = [85.0, float("nan"), 85.0]
    result = _compute_violation_zones(p10, p90, bound, ">=")
    # Stage 0 violated, NaN at 1 breaks the zone, stage 2 violated → two zones
    assert result == [{"start": 0, "end": 0}, {"start": 2, "end": 2}]


def test_compute_violation_zones_contiguous_range() -> None:
    """>=: contiguous violation across stages 0–2 produces a single interval."""
    p10 = [70.0, 70.0, 70.0]
    p90 = [80.0, 80.0, 80.0]
    bound = [85.0, 85.0, 85.0]
    result = _compute_violation_zones(p10, p90, bound, ">=")
    assert result == [{"start": 0, "end": 2}]


def test_compute_violation_zones_empty_inputs() -> None:
    """Empty input lists must return empty result without error."""
    result = _compute_violation_zones([], [], [], ">=")
    assert result == []


# ---------------------------------------------------------------------------
# test__build_constraint_lhs_data
# ---------------------------------------------------------------------------


def test_build_constraint_lhs_data_structure() -> None:
    """Returned dict must have stages, xlabels, and constraints with expected keys."""
    constraints = [
        {"id": 0, "name": "VminOP_c0", "sense": ">="},
        {"id": 1, "name": "RE_c1", "sense": "<="},
    ]
    lhs_df = _make_lhs_df([0, 1], n_scenarios=2, n_stages=3, lhs_value=120.0)
    gc_bounds = _make_gc_bounds([0, 1], n_stages=3)
    stage_labels = {0: "Jan", 1: "Feb", 2: "Mar"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)

    assert "stages" in result
    assert "xlabels" in result
    assert "constraints" in result
    assert result["stages"] == [0, 1, 2]
    assert result["xlabels"] == ["Jan", "Feb", "Mar"]

    for cid_str in ("0", "1"):
        entry = result["constraints"][cid_str]
        for key in (
            "name",
            "sense",
            "lhs_p10",
            "lhs_p50",
            "lhs_p90",
            "bound",
            "violations",
        ):
            assert key in entry, f"Missing key '{key}' in constraint {cid_str}"
        assert len(entry["lhs_p10"]) == 3
        assert len(entry["lhs_p50"]) == 3
        assert len(entry["lhs_p90"]) == 3
        assert len(entry["bound"]) == 3


def test_build_constraint_lhs_data_uniform_lhs_gives_equal_percentiles() -> None:
    """Uniform LHS across scenarios must yield p10 == p50 == p90 at every stage."""
    constraints = [{"id": 0, "name": "VminOP_c0", "sense": ">="}]
    lhs_df = _make_lhs_df([0], n_scenarios=3, n_stages=3, lhs_value=50.0)
    gc_bounds = _make_gc_bounds([0], n_stages=3)
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    for i in range(3):
        assert math.isclose(entry["lhs_p10"][i], 50.0, abs_tol=1e-3)
        assert math.isclose(entry["lhs_p50"][i], 50.0, abs_tol=1e-3)
        assert math.isclose(entry["lhs_p90"][i], 50.0, abs_tol=1e-3)


def test_build_constraint_lhs_data_violations_populated_for_ge() -> None:
    """>=: violations list is non-empty when p10 < bound at some stages."""
    constraints = [{"id": 0, "name": "VminOP_c0", "sense": ">="}]
    # lhs_value=80 < bound=100 → p10 will be 80, bound is 100 → violation everywhere
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=3, lhs_value=80.0)
    gc_bounds = _make_gc_bounds([0], n_stages=3)  # bound=100.0
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    violations = result["constraints"]["0"]["violations"]

    assert len(violations) > 0
    assert violations[0]["start"] == 0


def test_build_constraint_lhs_data_empty_lhs_gives_zeros() -> None:
    """Empty lhs_df must produce zero arrays for p10/p50/p90."""
    constraints = [{"id": 0, "name": "VminOP_c0", "sense": ">="}]
    lhs_df = pd.DataFrame(
        columns=["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
    )
    gc_bounds = _make_gc_bounds([0], n_stages=3)
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["0"]

    assert entry["lhs_p10"] == [0.0, 0.0, 0.0]
    assert entry["lhs_p50"] == [0.0, 0.0, 0.0]
    assert entry["lhs_p90"] == [0.0, 0.0, 0.0]


def test_build_constraint_lhs_data_missing_bounds_gives_none_bound() -> None:
    """No gc_bounds rows for a constraint → bound array is all None."""
    constraints = [{"id": 99, "name": "VminOP_c99", "sense": ">="}]
    lhs_df = _make_lhs_df([99], n_scenarios=2, n_stages=3)
    gc_bounds = pd.DataFrame(columns=["constraint_id", "stage_id", "block_id", "bound"])
    stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    result = _build_constraint_lhs_data(constraints, lhs_df, gc_bounds, stage_labels)
    entry = result["constraints"]["99"]

    assert all(v is None for v in entry["bound"])
    assert entry["violations"] == []


# ---------------------------------------------------------------------------
# test__build_lhs_section
# ---------------------------------------------------------------------------


def test_build_lhs_section_html_contains_required_elements() -> None:
    """_build_lhs_section must contain gc-constraint-sel, gc-lhs-chart, GC_LHS_DATA."""
    constraints = _make_constraints(n=2)
    gc_bounds = _make_gc_bounds([0, 1], n_stages=3)
    lhs_df = _make_lhs_df([0, 1], n_scenarios=2, n_stages=3)
    data = _make_mock_data(constraints=constraints, gc_bounds=gc_bounds, n_stages=3)
    data.gc_bounds = gc_bounds

    html = _build_lhs_section(data, lhs_df)

    assert "gc-constraint-sel" in html
    assert "gc-lhs-chart" in html
    assert "GC_LHS_DATA" in html


def test_build_lhs_section_html_has_one_option_per_constraint() -> None:
    """_build_lhs_section must emit one <option> per constraint."""
    constraints = _make_constraints(n=3)
    lhs_df = _make_lhs_df([0, 1, 2], n_scenarios=2, n_stages=2)
    data = _make_mock_data(constraints=constraints, n_stages=2)

    html = _build_lhs_section(data, lhs_df)

    assert html.count("<option") == 3


def test_build_lhs_section_json_blob_has_violations_key() -> None:
    """GC_LHS_DATA JSON embedded in the section must include 'violations' key."""
    constraints = _make_constraints(n=1)
    lhs_df = _make_lhs_df([0], n_scenarios=2, n_stages=2)
    data = _make_mock_data(constraints=constraints, n_stages=2)

    html = _build_lhs_section(data, lhs_df)

    assert '"violations"' in html


# ---------------------------------------------------------------------------
# test_type_filter_dropdown
# ---------------------------------------------------------------------------


def test_type_filter_dropdown_present_in_render() -> None:
    """render() HTML must contain gc-type-filter with 4 options."""
    data = _make_mock_data()

    with _render_with_stubs_ctx(data) as html:
        assert "gc-type-filter" in html
        assert (
            html.count("<option") >= 4
        )  # All, VminOP, RE, AGRINT + constraint options


def test_type_filter_has_all_four_options() -> None:
    """gc-type-filter must have options: All, VminOP, RE, AGRINT."""
    data = _make_mock_data()

    with _render_with_stubs_ctx(data) as html:
        # Find the filter select block
        assert 'value="All"' in html
        assert 'value="VminOP"' in html
        assert 'value="RE"' in html
        assert 'value="AGRINT"' in html


# ---------------------------------------------------------------------------
# test_no_three_separate_sections
# ---------------------------------------------------------------------------


def test_no_three_separate_old_section_titles() -> None:
    """render() must NOT contain the old three-section titles from Section C."""
    data = _make_mock_data()

    with _render_with_stubs_ctx(data) as html:
        assert "VminOP: Stored Energy vs Minimum" not in html
        assert "Electric Constraints (RE)" not in html
        assert "Exchange Group Constraints (AGRINT)" not in html


# ---------------------------------------------------------------------------
# test_render (existing + updated)
# ---------------------------------------------------------------------------

_PATCH_EVAL = "cobre_bridge.dashboard.tabs.constraints.evaluate_constraint_expressions"

_STUB_LHS_DF = pd.DataFrame(
    columns=["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
)


class _render_with_stubs_ctx:
    """Context manager that calls render() with LazyFrame-dependent functions patched."""

    def __init__(self, data: MagicMock) -> None:
        self._data = data
        self._html: str = ""

    def __enter__(self) -> str:
        from unittest.mock import patch

        with patch(_PATCH_EVAL, return_value=_STUB_LHS_DF):
            self._html = render(self._data)
        return self._html

    def __exit__(self, *_: object) -> None:
        pass


def _render_with_stubs(data: MagicMock) -> str:
    """Call render() with LazyFrame-dependent functions patched to stubs."""
    from unittest.mock import patch

    with patch(_PATCH_EVAL, return_value=_STUB_LHS_DF):
        return render(data)


def test_render_produces_expected_sections() -> None:
    """render() with minimal data must contain expected HTML substrings."""
    constraints = _make_constraints(n=2, ctype="VminOP")
    gc_bounds = _make_gc_bounds([c["id"] for c in constraints])
    gc_violations = _make_gc_violations(constraint_ids=None)
    costs = _make_costs(include_violation_cost=False)
    data = _make_mock_data(
        constraints=constraints,
        gc_bounds=gc_bounds,
        gc_violations=gc_violations,
        costs=costs,
    )
    html = _render_with_stubs(data)
    assert "Constraint Summary" in html
    assert "VminOP" in html
    assert "Violation" in html
    assert "metrics-grid" in html


def test_render_contains_section_titles() -> None:
    """render() must contain the current section titles (A, B, C only)."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "Constraint Summary" in html
    assert "LHS vs Bound" in html


def test_render_contains_chart_grid() -> None:
    """render() must contain chart-grid elements."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "chart-grid" in html


def test_render_contains_lhs_vs_bound_section() -> None:
    """render() must contain the new unified 'LHS vs Bound' section title."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "LHS vs Bound" in html


def test_render_contains_gc_lhs_chart_div() -> None:
    """render() must contain the gc-lhs-chart div for the JS chart."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "gc-lhs-chart" in html


def test_render_returns_string() -> None:
    """render() must return a str."""
    data = _make_mock_data()
    result = _render_with_stubs(data)
    assert isinstance(result, str)
