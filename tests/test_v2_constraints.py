"""Unit tests for cobre_bridge.dashboard.tabs.v2_constraints.

Covers module constants, can_render, _build_metrics_row, and the full
render() path using MagicMock data with real polars/pandas objects for
fields that get accessed as LazyFrames/DataFrames.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import polars as pl

import cobre_bridge.dashboard.tabs.v2_constraints as v2_constraints
from cobre_bridge.dashboard.tabs.v2_constraints import (
    _build_metrics_row,
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
    assert v2_constraints.TAB_ID == "tab-v2-constraints"
    assert v2_constraints.TAB_LABEL == "Constraints"
    assert v2_constraints.TAB_ORDER == 80


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
# test_render
#
# render() calls evaluate_constraint_expressions, chart_violation_summary,
# and chart_violation_heatmap — all of which operate on polars LazyFrames.
# The reused chart functions are already tested in constraints.py coverage, so
# we patch them here to return lightweight stub HTML, keeping these tests
# focused on the v2 wrapper structure rather than re-testing chart logic.
#
# The patch targets are in the v2_constraints module's namespace (where the
# names were imported into) to intercept the calls made by render().
# ---------------------------------------------------------------------------

_PATCH_EVAL = (
    "cobre_bridge.dashboard.tabs.v2_constraints.evaluate_constraint_expressions"
)
_PATCH_SUMMARY = "cobre_bridge.dashboard.tabs.v2_constraints.chart_violation_summary"
_PATCH_HEATMAP = "cobre_bridge.dashboard.tabs.v2_constraints.chart_violation_heatmap"

_STUB_LHS_DF = pd.DataFrame(
    columns=["constraint_id", "scenario_id", "stage_id", "block_id", "lhs_value"]
)


def _render_with_stubs(data: MagicMock) -> str:
    """Call render() with LazyFrame-dependent functions patched to stubs."""
    from unittest.mock import patch

    with (
        patch(_PATCH_EVAL, return_value=_STUB_LHS_DF),
        patch(_PATCH_SUMMARY, return_value="<p>violation summary stub</p>"),
        patch(_PATCH_HEATMAP, return_value="<p>violation heatmap stub</p>"),
    ):
        return render(data)


def test_render_produces_expected_sections() -> None:
    """render() with minimal data must contain expected HTML substrings.

    Acceptance criterion from ticket-019: the returned HTML contains
    'Constraint Summary', 'VminOP', 'Violation', and 'metrics-grid'.
    """
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
    """render() must contain all major section titles."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "Constraint Summary" in html
    assert "Constraint Bounds Timeline" in html
    assert "Violation Cost Timeline" in html
    assert "Violation Summary" in html


def test_render_contains_collapsible_sections() -> None:
    """render() must contain collapsible sections for lower-priority content."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "collapsible-section" in html


def test_render_bounds_timeline_is_collapsed() -> None:
    """render() must start the Bounds Timeline section collapsed."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    # The bounds timeline section title appears inside a default-collapsed section
    assert "default-collapsed" in html


def test_render_contains_chart_grid() -> None:
    """render() must contain chart-grid elements."""
    data = _make_mock_data()
    html = _render_with_stubs(data)
    assert "chart-grid" in html


def test_render_with_re_and_agrint_constraints() -> None:
    """render() with RE and AGRINT constraints must include those section titles."""
    constraints = [
        {
            "id": 0,
            "name": "RE_electric_1",
            "expression": "hydro_storage(0)",
            "sense": "<=",
            "slack": {},
        },
        {
            "id": 1,
            "name": "AGRINT_group_1",
            "expression": "hydro_storage(1)",
            "sense": "<=",
            "slack": {},
        },
    ]
    gc_bounds = _make_gc_bounds([0, 1])
    data = _make_mock_data(constraints=constraints, gc_bounds=gc_bounds)
    html = _render_with_stubs(data)
    assert "Electric Constraints (RE)" in html
    assert "Exchange Group Constraints (AGRINT)" in html


def test_render_returns_string() -> None:
    """render() must return a str."""
    data = _make_mock_data()
    result = _render_with_stubs(data)
    assert isinstance(result, str)
