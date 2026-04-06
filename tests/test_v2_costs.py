"""Unit tests for cobre_bridge.dashboard.tabs.v2_costs.

Covers module constants, can_render, _compute_npv_metric, _build_metrics_row,
_build_cost_table, _chart_cost_bar, and the full render() path including the
empty-costs degradation branch.

Ticket-015 additions cover: _render_cost_composition, _render_category_evolution,
_render_spot_price, _render_violations, and the extended render() output.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import plotly.graph_objects as go
import polars as pl

import cobre_bridge.dashboard.tabs.v2_costs as v2_costs
from cobre_bridge.dashboard.chart_helpers import compute_cost_summary
from cobre_bridge.dashboard.tabs.v2_costs import (
    _build_composition_data,
    _build_composition_section,
    _build_cost_table,
    _build_metrics_row,
    _chart_cost_bar,
    _chart_violation_timeline,
    _compute_npv_metric,
    _render_category_evolution,
    _render_cost_composition,
    _render_spot_price,
    _render_violations,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_costs_df(
    n_scenarios: int = 2,
    n_stages: int = 3,
    thermal_cost: float = 1000.0,
    deficit_cost: float = 50.0,
) -> pd.DataFrame:
    """Return a minimal costs DataFrame with ``n_scenarios`` and ``n_stages``.

    The ``discount_factor`` column is set to 1.0 for all rows so that tests
    with ``discount_rate=0.0`` can be verified without additional arithmetic.
    """
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "thermal_cost": thermal_cost,
                    "deficit_cost": deficit_cost,
                    "immediate_cost": thermal_cost + deficit_cost,
                    "total_cost": thermal_cost + deficit_cost,
                    "future_cost": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _make_mock_data(
    *,
    costs: pd.DataFrame | None = None,
    discount_rate: float = 0.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> MagicMock:
    """Build a minimal MagicMock that satisfies the DashboardData interface.

    Sets proper defaults for all fields accessed by ``render()`` (including
    ticket-015 sections) to avoid MagicMock auto-chaining on polars
    LazyFrames, which causes OOM.
    """
    data = MagicMock()
    data.costs = costs if costs is not None else _make_costs_df(n_scenarios, n_stages)
    data.discount_rate = discount_rate
    data.n_scenarios = n_scenarios
    data.n_stages = n_stages
    data.stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
    data.non_fictitious_bus_ids = [0, 1]
    data.bus_names = {0: "Bus A", 1: "Bus B"}
    # Provide real empty polars objects to prevent MagicMock auto-chaining OOM
    data.buses_lf = pl.LazyFrame()
    data.bh_df = pl.DataFrame(
        [{"stage_id": s, "block_id": 0, "_bh": 730.0} for s in range(n_stages)]
    )
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match the ticket specification exactly."""
    assert v2_costs.TAB_ID == "tab-v2-costs"
    assert v2_costs.TAB_LABEL == "Costs"
    assert v2_costs.TAB_ORDER == 40


# ---------------------------------------------------------------------------
# test_can_render
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render must return True unconditionally."""
    data = _make_mock_data()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test__compute_npv_metric
# ---------------------------------------------------------------------------


def test_compute_npv_metric_undiscounted_returns_mean_per_scenario() -> None:
    """With discount_rate=0.0 and thermal_cost summing to 6000 across 2 scenarios,
    _compute_npv_metric must return 3000.0 (mean across scenarios).

    Each scenario has 3 stages, each with thermal_cost=1000.0, so the per-scenario
    sum is 3000.0. With 2 scenarios the mean is also 3000.0.
    """
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data(costs=costs, discount_rate=0.0)
    result = _compute_npv_metric(data, "thermal_cost")
    assert abs(result - 3000.0) < 1e-6


def test_compute_npv_metric_discounted_less_than_undiscounted() -> None:
    """With discount_rate=0.12 the discounted NPV must be lower than the
    undiscounted sum, because later-stage costs are reduced by discount factors."""
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    undiscounted = _compute_npv_metric(
        _make_mock_data(costs=costs, discount_rate=0.0), "thermal_cost"
    )
    discounted = _compute_npv_metric(
        _make_mock_data(costs=costs, discount_rate=0.12), "thermal_cost"
    )
    assert discounted < undiscounted


def test_compute_npv_metric_missing_column_returns_zero() -> None:
    """When the requested column is absent, _compute_npv_metric must return 0.0."""
    costs = _make_costs_df()
    data = _make_mock_data(costs=costs)
    result = _compute_npv_metric(data, "nonexistent_column")
    assert result == 0.0


def test_compute_npv_metric_empty_dataframe_returns_zero() -> None:
    """When costs is empty, _compute_npv_metric must return 0.0."""
    data = _make_mock_data(costs=pd.DataFrame())
    result = _compute_npv_metric(data, "thermal_cost")
    assert result == 0.0


# ---------------------------------------------------------------------------
# test__build_metrics_row
# ---------------------------------------------------------------------------


def test_build_metrics_row_produces_four_metric_cards() -> None:
    """_build_metrics_row must produce HTML with 'metric-card' at least 4 times."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 4


def test_build_metrics_row_empty_costs_still_four_cards() -> None:
    """_build_metrics_row with empty costs must still produce 4 metric-card elements."""
    data = _make_mock_data(costs=pd.DataFrame())
    html = _build_metrics_row(data)
    assert html.count("metric-card") >= 4


def test_build_metrics_row_contains_expected_cost_label() -> None:
    """_build_metrics_row must include the 'Expected Cost (NPV)' label."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "Expected Cost (NPV)" in html


def test_build_metrics_row_contains_thermal_and_deficit_labels() -> None:
    """_build_metrics_row must include Thermal Cost and Deficit Cost labels."""
    data = _make_mock_data()
    html = _build_metrics_row(data)
    assert "Thermal Cost (NPV)" in html
    assert "Deficit Cost (NPV)" in html


# ---------------------------------------------------------------------------
# test__build_cost_table
# ---------------------------------------------------------------------------


def test_build_cost_table_contains_data_table_class() -> None:
    """_build_cost_table must return HTML with class 'data-table'."""
    costs = _make_costs_df()
    summary = compute_cost_summary(costs, 0.0)
    html = _build_cost_table(summary)
    assert 'class="data-table"' in html


def test_build_cost_table_has_tbody_with_rows() -> None:
    """_build_cost_table must include a <tbody> with at least one <tr>."""
    costs = _make_costs_df()
    summary = compute_cost_summary(costs, 0.0)
    html = _build_cost_table(summary)
    assert "<tbody>" in html
    assert "<tr>" in html


def test_build_cost_table_empty_df_returns_placeholder() -> None:
    """_build_cost_table on an empty DataFrame must return a fallback <p>."""
    html = _build_cost_table(pd.DataFrame())
    assert "<table" not in html
    assert "No cost data" in html


# ---------------------------------------------------------------------------
# test__chart_cost_bar
# ---------------------------------------------------------------------------


def test_chart_cost_bar_returns_figure() -> None:
    """_chart_cost_bar must return a plotly Figure."""
    costs = _make_costs_df()
    summary = compute_cost_summary(costs, 0.0)
    fig = _chart_cost_bar(summary)
    assert isinstance(fig, go.Figure)


def test_chart_cost_bar_has_horizontal_bar_traces() -> None:
    """_chart_cost_bar must produce at least one horizontal Bar trace."""
    costs = _make_costs_df()
    summary = compute_cost_summary(costs, 0.0)
    fig = _chart_cost_bar(summary)
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) >= 1
    for trace in bar_traces:
        assert trace.orientation == "h"


def test_chart_cost_bar_error_bars_p10_p90() -> None:
    """_chart_cost_bar must set error_x with p10–p90 range on each bar trace.

    Acceptance criterion from ticket-007: given p10=850, mean=1000, p90=1150,
    the trace must have error_x.array=[150] and error_x.arrayminus=[150].
    """
    summary = pd.DataFrame(
        {
            "group": ["Thermal"],
            "mean": [1000.0],
            "std": [100.0],
            "p5": [800.0],
            "p10": [850.0],
            "p90": [1150.0],
            "p95": [1200.0],
            "pct": [100.0],
        }
    )
    fig = _chart_cost_bar(summary)

    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) == 1
    trace = bar_traces[0]
    assert trace.error_x is not None
    assert trace.error_x.visible is True
    assert trace.error_x.array == (150.0,)
    assert trace.error_x.arrayminus == (150.0,)


def test_chart_cost_bar_error_bars_omitted_when_nan() -> None:
    """_chart_cost_bar must omit error_x when p10 or p90 is NaN."""
    import math

    summary = pd.DataFrame(
        {
            "group": ["Thermal"],
            "mean": [1000.0],
            "std": [0.0],
            "p5": [math.nan],
            "p10": [math.nan],
            "p90": [math.nan],
            "p95": [math.nan],
            "pct": [100.0],
        }
    )
    fig = _chart_cost_bar(summary)

    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) == 1
    assert bar_traces[0].error_x is None or bar_traces[0].error_x.visible is not True


# ---------------------------------------------------------------------------
# test_render
# ---------------------------------------------------------------------------


def test_render_with_valid_costs_contains_metric_card_four_times() -> None:
    """render() with valid costs must contain 'metric-card' at least 4 times."""
    data = _make_mock_data()
    html = render(data)
    assert html.count("metric-card") >= 4


def test_render_with_valid_costs_contains_data_table() -> None:
    """render() with valid costs must produce HTML with a data-table."""
    data = _make_mock_data()
    html = render(data)
    assert 'class="data-table"' in html


def test_render_with_valid_costs_has_table_and_tbody() -> None:
    """render() HTML must contain a <table with a <tbody> and at least one <tr>."""
    data = _make_mock_data()
    html = render(data)
    assert "<table" in html
    assert "<tbody>" in html
    assert "<tr>" in html


def test_render_with_empty_costs_contains_no_cost_data() -> None:
    """render() with empty costs must produce 'No cost data' fallback text."""
    data = _make_mock_data(costs=pd.DataFrame())
    html = render(data)
    assert "No cost data" in html


def test_render_contains_section_title() -> None:
    """render() must include the 'NPV Cost Analysis' section title."""
    data = _make_mock_data()
    html = render(data)
    assert "NPV Cost Analysis" in html


# ---------------------------------------------------------------------------
# Helpers for ticket-015 tests
# ---------------------------------------------------------------------------


def _make_costs_df_with_violations(
    n_scenarios: int = 2,
    n_stages: int = 3,
    thermal_cost: float = 1000.0,
    generic_violation_cost: float = 0.0,
) -> pd.DataFrame:
    """Return a costs DataFrame that includes a violation cost column."""
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "block_id": 0,
                    "thermal_cost": thermal_cost,
                    "deficit_cost": 50.0,
                    "immediate_cost": thermal_cost + 50.0,
                    "total_cost": thermal_cost + 50.0,
                    "future_cost": 0.0,
                    "generic_violation_cost": generic_violation_cost,
                }
            )
    return pd.DataFrame(rows)


def _make_mock_data_full(
    *,
    costs: pd.DataFrame | None = None,
    non_fictitious_bus_ids: list[int] | None = None,
    bus_names: dict[int, str] | None = None,
    buses_lf: pl.LazyFrame | None = None,
    bh_df: pl.DataFrame | None = None,
    discount_rate: float = 0.0,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> MagicMock:
    """Build a MagicMock satisfying the full DashboardData interface for ticket-015."""
    data = MagicMock()
    data.costs = costs if costs is not None else _make_costs_df(n_scenarios, n_stages)
    data.discount_rate = discount_rate
    data.n_scenarios = n_scenarios
    data.n_stages = n_stages
    data.stage_labels = {i: f"Stage {i}" for i in range(n_stages)}
    data.non_fictitious_bus_ids = (
        non_fictitious_bus_ids if non_fictitious_bus_ids is not None else [0, 1]
    )
    data.bus_names = bus_names if bus_names is not None else {0: "Bus A", 1: "Bus B"}

    if buses_lf is not None:
        data.buses_lf = buses_lf
    else:
        # Build a minimal buses_lf with spot_price
        rows_list = []
        for scen in range(n_scenarios):
            for stage in range(n_stages):
                for bus_id in data.non_fictitious_bus_ids or [0, 1]:
                    rows_list.append(
                        {
                            "scenario_id": scen,
                            "stage_id": stage,
                            "block_id": 0,
                            "bus_id": bus_id,
                            "spot_price": 100.0 + scen * 10 + stage * 5,
                        }
                    )
        data.buses_lf = pl.DataFrame(rows_list).lazy()

    if bh_df is not None:
        data.bh_df = bh_df
    else:
        # Build a minimal bh_df with one block per stage
        bh_rows = [
            {"stage_id": s, "block_id": 0, "_bh": 730.0} for s in range(n_stages)
        ]
        data.bh_df = pl.DataFrame(bh_rows)

    return data


# ---------------------------------------------------------------------------
# test__render_cost_composition (Section D)
# ---------------------------------------------------------------------------


def test_render_cost_composition_contains_collapsible_section() -> None:
    """_render_cost_composition must return HTML with collapsible-section class."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    assert "collapsible-section" in html


def test_render_cost_composition_contains_section_title() -> None:
    """_render_cost_composition must include the 'Cost Composition by Stage' title."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    assert "Cost Composition by Stage" in html


def test_render_cost_composition_with_empty_costs_returns_fallback() -> None:
    """_render_cost_composition with empty costs must return fallback text."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    html = _render_cost_composition(data)
    assert "No cost data" in html


def test_render_cost_composition_is_not_default_collapsed() -> None:
    """_render_cost_composition must start expanded (no 'default-collapsed' class)."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    # default_collapsed=False means the section_class is 'collapsible-section'
    # (not 'collapsible-section default-collapsed')
    assert "default-collapsed" not in html


def test_render_cost_composition_contains_chart_card() -> None:
    """_render_cost_composition must wrap the chart in a chart-card."""
    data = _make_mock_data_full()
    html = _render_cost_composition(data)
    assert "chart-card" in html


# ---------------------------------------------------------------------------
# test__render_category_evolution (Section E)
# ---------------------------------------------------------------------------


def test_render_category_evolution_contains_collapsible_section() -> None:
    """_render_category_evolution must return HTML with collapsible-section class."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "collapsible-section" in html


def test_render_category_evolution_contains_section_title() -> None:
    """_render_category_evolution must include the 'Cost Category Trends' title."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "Cost Category Trends" in html


def test_render_category_evolution_with_empty_costs_returns_fallback() -> None:
    """_render_category_evolution with empty costs must return fallback text."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    html = _render_category_evolution(data)
    assert "No cost data" in html


def test_render_category_evolution_is_default_collapsed() -> None:
    """_render_category_evolution must start collapsed."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "default-collapsed" in html


def test_render_category_evolution_contains_chart_card() -> None:
    """_render_category_evolution must wrap the chart in a chart-card."""
    data = _make_mock_data_full()
    html = _render_category_evolution(data)
    assert "chart-card" in html


# ---------------------------------------------------------------------------
# test__render_spot_price (Section F)
# ---------------------------------------------------------------------------


def test_render_spot_price_contains_collapsible_section() -> None:
    """_render_spot_price must return HTML with collapsible-section class."""
    data = _make_mock_data_full(non_fictitious_bus_ids=[0, 1, 2, 3])
    data.bus_names = {0: "Bus A", 1: "Bus B", 2: "Bus C", 3: "Bus D"}
    # Rebuild buses_lf for 4 buses
    rows_list = []
    for scen in range(2):
        for stage in range(3):
            for bus_id in [0, 1, 2, 3]:
                rows_list.append(
                    {
                        "scenario_id": scen,
                        "stage_id": stage,
                        "block_id": 0,
                        "bus_id": bus_id,
                        "spot_price": 100.0 + scen * 10,
                    }
                )
    data.buses_lf = pl.DataFrame(rows_list).lazy()
    data.stage_labels = {0: "Stage 0", 1: "Stage 1", 2: "Stage 2"}
    html = _render_spot_price(data)
    assert "collapsible-section" in html


def test_render_spot_price_subplot_titles_match_bus_count() -> None:
    """_render_spot_price with 4 non-fictitious buses must produce 4 subplot titles.

    Acceptance criterion from ticket-015: the number of subplot titles in the
    figure equals the number of non-fictitious buses.
    """
    bus_ids = [0, 1, 2, 3]
    bus_names = {0: "Alpha", 1: "Beta", 2: "Gamma", 3: "Delta"}
    rows_list = []
    for scen in range(2):
        for stage in range(3):
            for bus_id in bus_ids:
                rows_list.append(
                    {
                        "scenario_id": scen,
                        "stage_id": stage,
                        "block_id": 0,
                        "bus_id": bus_id,
                        "spot_price": 80.0 + scen * 5 + stage * 2,
                    }
                )
    buses_lf = pl.DataFrame(rows_list).lazy()
    bh_df = pl.DataFrame(
        [{"stage_id": s, "block_id": 0, "_bh": 730.0} for s in range(3)]
    )

    data = _make_mock_data_full(
        non_fictitious_bus_ids=bus_ids,
        bus_names=bus_names,
        buses_lf=buses_lf,
        bh_df=bh_df,
    )
    data.stage_labels = {0: "S0", 1: "S1", 2: "S2"}

    html = _render_spot_price(data)
    # Each bus name should appear in the HTML as a subplot title annotation
    for name in bus_names.values():
        assert name in html


def test_render_spot_price_with_empty_buses_lf_returns_fallback() -> None:
    """_render_spot_price with empty buses_lf must return 'No spot price data.'."""
    data = _make_mock_data_full(
        non_fictitious_bus_ids=[0, 1],
        buses_lf=pl.LazyFrame(),
    )
    html = _render_spot_price(data)
    assert "No spot price data" in html


def test_render_spot_price_with_no_bus_ids_returns_fallback() -> None:
    """_render_spot_price with empty non_fictitious_bus_ids returns fallback."""
    data = _make_mock_data_full(non_fictitious_bus_ids=[])
    html = _render_spot_price(data)
    assert "No spot price data" in html


def test_render_spot_price_is_default_collapsed() -> None:
    """_render_spot_price must start collapsed."""
    data = _make_mock_data_full()
    html = _render_spot_price(data)
    assert "default-collapsed" in html


# ---------------------------------------------------------------------------
# test__render_violations (Section G)
# ---------------------------------------------------------------------------


def test_render_violations_zero_costs_returns_no_violation_text() -> None:
    """_render_violations with all-zero violation costs must return 'No violation costs'.

    Acceptance criterion from ticket-015.
    """
    costs = _make_costs_df_with_violations(generic_violation_cost=0.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "No violation costs" in html


def test_render_violations_nonzero_costs_returns_collapsible_section() -> None:
    """_render_violations with non-zero generic_violation_cost returns collapsible-section.

    Acceptance criterion from ticket-015: the result contains 'collapsible-section'
    and a bar chart.
    """
    costs = _make_costs_df_with_violations(generic_violation_cost=500.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "collapsible-section" in html
    # Bar chart is embedded in the chart-card
    assert "chart-card" in html


def test_render_violations_nonzero_contains_section_title() -> None:
    """_render_violations with violations must include the 'Violation Costs' title."""
    costs = _make_costs_df_with_violations(generic_violation_cost=200.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "Violation Costs" in html


def test_render_violations_empty_costs_returns_fallback() -> None:
    """_render_violations with empty costs DataFrame must return fallback text."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    html = _render_violations(data)
    assert "No violation costs" in html


def test_render_violations_no_violation_columns_returns_fallback() -> None:
    """_render_violations when costs has no violation columns must return fallback."""
    # costs has only thermal_cost and deficit_cost — no violation columns
    costs = _make_costs_df(n_scenarios=2, n_stages=3)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "No violation costs" in html


def test_render_violations_is_default_collapsed() -> None:
    """_render_violations must start collapsed regardless of content."""
    costs = _make_costs_df_with_violations(generic_violation_cost=100.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "default-collapsed" in html


# ---------------------------------------------------------------------------
# test_render — extended (ticket-015)
# ---------------------------------------------------------------------------


def test_render_contains_cost_composition_section() -> None:
    """render() with non-zero thermal_cost across 3 stages must include
    'Cost Composition by Stage'.

    Acceptance criterion from ticket-015.
    """
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data_full(costs=costs)
    html = render(data)
    assert "Cost Composition by Stage" in html


def test_render_contains_at_least_three_collapsible_sections() -> None:
    """render() must contain at least 3 collapsible-section elements.

    Acceptance criterion from ticket-015: composition, category trends, spot
    price sections are always present (violation section may be absent).
    """
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data_full(costs=costs)
    html = render(data)
    assert html.count("collapsible-section") >= 3


def test_render_includes_npv_section() -> None:
    """render() must still include the NPV Cost Analysis section from ticket-014."""
    data = _make_mock_data_full()
    html = render(data)
    assert "NPV Cost Analysis" in html


def test_render_includes_all_four_temporal_sections() -> None:
    """render() with violation costs must include all four temporal sections."""
    costs = _make_costs_df_with_violations(
        n_scenarios=2, n_stages=3, thermal_cost=1000.0, generic_violation_cost=50.0
    )
    data = _make_mock_data_full(costs=costs)
    html = render(data)
    assert "Cost Composition by Stage" in html
    assert "Cost Category Trends" in html
    assert "Spot Price by Bus" in html
    assert "Violation Costs" in html


# ---------------------------------------------------------------------------
# test__chart_violation_timeline (ticket-009)
# ---------------------------------------------------------------------------


def _make_costs_df_with_storage_violation(
    n_scenarios: int = 2,
    n_stages: int = 3,
    storage_violation_cost: float = 10.0,
) -> pd.DataFrame:
    """Return a costs DataFrame with a storage_violation_cost column.

    Values are set to *storage_violation_cost* for all rows so that per-stage
    sums are always positive when the argument is nonzero.
    """
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "block_id": 0,
                    "thermal_cost": 1000.0,
                    "storage_violation_cost": storage_violation_cost,
                }
            )
    return pd.DataFrame(rows)


def test_chart_violation_timeline_returns_figure_with_nonzero_data() -> None:
    """_chart_violation_timeline must return a Figure when violation data is present."""
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=10.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is not None
    assert isinstance(fig, go.Figure)


def test_chart_violation_timeline_has_scatter_trace_for_storage_violation() -> None:
    """_chart_violation_timeline must include a Scatter trace for storage_violation.

    Acceptance criterion: trace name contains 'Storage Violation' (cleaned label).
    """
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=10.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is not None
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) >= 1
    trace_names = [t.name for t in scatter_traces]
    # Cleaned label: "_cost" removed, "_" -> " ", then title-cased
    assert any("Storage Violation" in name for name in trace_names)


def test_chart_violation_timeline_x_values_use_stage_labels() -> None:
    """_chart_violation_timeline x-axis values must use stage labels, not raw integers.

    Acceptance criterion from ticket-009: trace x-values are strings from
    stage_labels mapping, not bare stage_id integers.
    """
    costs = _make_costs_df_with_storage_violation(
        n_scenarios=2, n_stages=3, storage_violation_cost=20.0
    )
    data = _make_mock_data_full(costs=costs, n_stages=3)
    # stage_labels maps 0->"Stage 0", 1->"Stage 1", 2->"Stage 2"
    fig = _chart_violation_timeline(data)
    assert fig is not None
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) >= 1
    x_vals = list(scatter_traces[0].x)
    # Must contain at least one string label, not raw integers
    assert any(isinstance(v, str) for v in x_vals)
    assert "Stage 0" in x_vals or "Stage 1" in x_vals or "Stage 2" in x_vals


def test_chart_violation_timeline_returns_none_when_all_zero() -> None:
    """_chart_violation_timeline must return None when all violation costs are zero.

    Acceptance criterion from ticket-009.
    """
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=0.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is None


def test_chart_violation_timeline_returns_none_for_empty_costs() -> None:
    """_chart_violation_timeline must return None when costs DataFrame is empty."""
    data = _make_mock_data_full(costs=pd.DataFrame())
    fig = _chart_violation_timeline(data)
    assert fig is None


def test_chart_violation_timeline_returns_none_when_no_violation_columns() -> None:
    """_chart_violation_timeline must return None when no violation columns exist."""
    # costs only has thermal_cost (no violation column)
    costs = _make_costs_df(n_scenarios=2, n_stages=3, thermal_cost=1000.0)
    data = _make_mock_data_full(costs=costs)
    fig = _chart_violation_timeline(data)
    assert fig is None


# ---------------------------------------------------------------------------
# test__render_violations extended (ticket-009)
# ---------------------------------------------------------------------------


def test_render_violations_with_nonzero_data_contains_two_chart_cards() -> None:
    """_render_violations with nonzero violation data must contain two chart-card divs.

    Acceptance criterion from ticket-009: the Violations section now contains a
    2-column grid with bar chart (left) + timeline (right).
    """
    costs = _make_costs_df_with_storage_violation(
        n_scenarios=2, n_stages=3, storage_violation_cost=10.0
    )
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert html.count("chart-card") >= 2


def test_render_violations_timeline_absent_when_all_violation_zero() -> None:
    """_render_violations with all-zero violations must render only the bar fallback.

    When all violations are zero, _render_violations returns the 'No violation costs'
    message — the timeline chart is not rendered.
    """
    costs = _make_costs_df_with_storage_violation(storage_violation_cost=0.0)
    data = _make_mock_data_full(costs=costs)
    html = _render_violations(data)
    assert "No violation costs" in html
    assert "chart-card" not in html


# ---------------------------------------------------------------------------
# test__build_composition_data (ticket-011)
# ---------------------------------------------------------------------------


def _make_composition_costs_df(
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pd.DataFrame:
    """Return a small costs DataFrame with 3 cost columns for composition tests.

    Columns: thermal_generation_cost (non-zero), spillage_cost (non-zero),
    ncs_generation_cost (all zero).
    """
    rows = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "block_id": 0,
                    "thermal_generation_cost": 1000.0
                    + scenario_id * 100
                    + stage_id * 50,
                    "spillage_cost": 200.0 + scenario_id * 20 + stage_id * 10,
                    "ncs_generation_cost": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_composition_data_keys() -> None:
    """_build_composition_data must return a dict with required top-level keys.

    Acceptance criterion from ticket-011: keys are category, component, total,
    stages, colors.
    """
    costs = _make_composition_costs_df()
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    result = _build_composition_data(data)
    assert result is not None
    assert set(result.keys()) == {"category", "component", "total", "stages", "colors"}


def test_build_composition_data_total_stats() -> None:
    """total.mean, total.p10, total.p90 must all have length == number of stages
    and satisfy p10 <= mean <= p90 for each stage.

    Acceptance criterion from ticket-011.
    """
    costs = _make_composition_costs_df(n_scenarios=2, n_stages=3)
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    result = _build_composition_data(data)
    assert result is not None
    total = result["total"]
    n = len(result["stages"])
    assert len(total["mean"]) == n
    assert len(total["p10"]) == n
    assert len(total["p90"]) == n
    for i in range(n):
        assert total["p10"][i] <= total["mean"][i] + 1e-6
        assert total["mean"][i] <= total["p90"][i] + 1e-6


def test_build_composition_data_empty() -> None:
    """_build_composition_data must return None for an empty costs DataFrame.

    Acceptance criterion from ticket-011.
    """
    data = _make_mock_data_full(costs=pd.DataFrame())
    result = _build_composition_data(data)
    assert result is None


def test_build_composition_data_zero_cols_excluded() -> None:
    """Zero-valued columns must not appear in component.

    Acceptance criterion from ticket-011: ncs_generation_cost is all-zero
    and must be absent from component keys.
    """
    costs = _make_composition_costs_df()
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    result = _build_composition_data(data)
    assert result is not None
    # The cleaned label for "ncs_generation_cost" -> "Ncs Generation"
    assert "Ncs Generation" not in result["component"]
    # Non-zero columns must be present
    assert len(result["component"]) >= 1


def test_build_composition_section_html() -> None:
    """_build_composition_section must emit costs-group-sel, costs-comp,
    and COSTS_COMP_DATA in the HTML output.

    Acceptance criterion from ticket-011.
    """
    costs = _make_composition_costs_df()
    data = _make_mock_data_full(costs=costs, n_scenarios=2, n_stages=3)
    html = _build_composition_section(data)
    assert "costs-group-sel" in html
    assert "costs-comp" in html
    assert "COSTS_COMP_DATA" in html
