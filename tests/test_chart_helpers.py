"""Unit tests for cobre_bridge.dashboard.chart_helpers.

Covers compute_percentiles, add_mean_p50_band, add_bounds_overlay,
make_chart_card, compute_npv_costs, group_costs, and compute_cost_summary.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import polars as pl
import pytest

from cobre_bridge.dashboard.chart_helpers import (
    COST_GROUP_COLORS,
    COST_GROUPS,
    add_bounds_overlay,
    add_mean_p50_band,
    compute_cost_summary,
    compute_npv_costs,
    compute_percentiles,
    group_costs,
    make_chart_card,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def five_scenarios_three_stages() -> pd.DataFrame:
    """Return a pandas DataFrame with 5 scenarios x 3 stages."""
    rows = [
        {"stage_id": s, "scenario_id": sc, "value": float(s * 10 + sc)}
        for s in range(1, 4)
        for sc in range(1, 6)
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def percentile_df() -> pd.DataFrame:
    """Return a percentile DataFrame with 10 rows suitable for add_mean_p50_band."""
    return pd.DataFrame(
        {
            "stage_id": list(range(1, 11)),
            "mean": [float(i) for i in range(10)],
            "p10": [float(i) * 0.8 for i in range(10)],
            "p50": [float(i) * 0.95 for i in range(10)],
            "p90": [float(i) * 1.2 for i in range(10)],
        }
    )


@pytest.fixture()
def bounds_df() -> pd.DataFrame:
    """Return a bounds DataFrame with min_storage and max_storage columns."""
    return pd.DataFrame(
        {
            "stage_id": list(range(1, 6)),
            "min_storage": [10.0, 12.0, 11.0, 13.0, 14.0],
            "max_storage": [90.0, 88.0, 92.0, 85.0, 87.0],
        }
    )


# ---------------------------------------------------------------------------
# compute_percentiles — pandas input
# ---------------------------------------------------------------------------


def test_compute_percentiles_pandas_basic(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Five scenarios x three stages produces 3 rows with the expected columns."""
    result = compute_percentiles(five_scenarios_three_stages, ["stage_id"], "value")

    assert result.shape[0] == 3
    assert list(result.columns) == ["stage_id", "mean", "p10", "p50", "p90"]


def test_compute_percentiles_pandas_values(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Computed mean equals the arithmetic mean of the five scenarios per stage."""
    result = compute_percentiles(five_scenarios_three_stages, ["stage_id"], "value")
    # For stage 1: values are 11, 12, 13, 14, 15 -> mean = 13.0
    stage1 = result[result["stage_id"] == 1]
    assert stage1["mean"].iloc[0] == pytest.approx(13.0)


# ---------------------------------------------------------------------------
# compute_percentiles — polars input
# ---------------------------------------------------------------------------


def test_compute_percentiles_polars_input(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Polars DataFrame input is converted and returns correct pandas output."""
    polars_df = pl.from_pandas(five_scenarios_three_stages)

    result = compute_percentiles(polars_df, ["stage_id"], "value")

    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3
    assert list(result.columns) == ["stage_id", "mean", "p10", "p50", "p90"]
    # P50 for stage 2 using values: 21, 22, 23, 24, 25 -> median = 23.0
    stage2 = result[result["stage_id"] == 2]
    assert stage2["p50"].iloc[0] == pytest.approx(23.0)


# ---------------------------------------------------------------------------
# compute_percentiles — empty input
# ---------------------------------------------------------------------------


def test_compute_percentiles_empty() -> None:
    """Empty DataFrame returns empty result with the expected column names."""
    empty_df = pd.DataFrame(columns=["stage_id", "scenario_id", "value"])

    result = compute_percentiles(empty_df, ["stage_id"], "value")

    assert result.empty
    assert list(result.columns) == ["stage_id", "mean", "p10", "p50", "p90"]


# ---------------------------------------------------------------------------
# compute_percentiles — custom percentiles
# ---------------------------------------------------------------------------


def test_compute_percentiles_custom_percentiles(
    five_scenarios_three_stages: pd.DataFrame,
) -> None:
    """Custom percentile tuple produces correctly named columns."""
    result = compute_percentiles(
        five_scenarios_three_stages,
        ["stage_id"],
        "value",
        percentiles=(0.05, 0.5, 0.95),
    )

    assert list(result.columns) == ["stage_id", "mean", "p5", "p50", "p95"]
    assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# add_mean_p50_band
# ---------------------------------------------------------------------------


def test_add_mean_p50_band_traces(percentile_df: pd.DataFrame) -> None:
    """Three traces are added: mean (solid), p50 (dashed), and p10-p90 band."""
    fig = go.Figure()
    result = add_mean_p50_band(fig, percentile_df, "stage_id", "Hydro", "#3B82F6")

    # 1 mean + 1 p50 + 2 band traces (p10 lower + p90 upper)
    assert len(result.data) == 4

    mean_trace = result.data[0]
    assert mean_trace.mode == "lines"
    assert mean_trace.line.dash is None or mean_trace.line.dash == "solid"
    assert mean_trace.line.width == 2
    assert mean_trace.name == "Hydro"

    p50_trace = result.data[1]
    assert p50_trace.line.dash == "dash"
    assert p50_trace.line.width == 1.5
    assert p50_trace.opacity == pytest.approx(0.7)

    # Band: p10 lower (no fill) then p90 upper (fill="tonexty")
    p10_trace = result.data[2]
    assert p10_trace.line.width == 0

    p90_trace = result.data[3]
    assert p90_trace.fill == "tonexty"
    assert "rgba" in p90_trace.fillcolor


def test_add_mean_p50_band_returns_figure(percentile_df: pd.DataFrame) -> None:
    """The function returns the same figure object (chaining support)."""
    fig = go.Figure()
    returned = add_mean_p50_band(fig, percentile_df, "stage_id", "Hydro", "#3B82F6")
    assert returned is fig


def test_add_mean_p50_band_empty_df() -> None:
    """Empty DataFrame causes no traces to be added."""
    empty_df = pd.DataFrame(columns=["stage_id", "mean", "p10", "p50", "p90"])
    fig = go.Figure()
    result = add_mean_p50_band(fig, empty_df, "stage_id", "Hydro", "#3B82F6")

    assert len(result.data) == 0


def test_add_mean_p50_band_show_band_false(percentile_df: pd.DataFrame) -> None:
    """When show_band=False, only mean and p50 traces are added."""
    fig = go.Figure()
    add_mean_p50_band(
        fig, percentile_df, "stage_id", "Hydro", "#3B82F6", show_band=False
    )
    assert len(fig.data) == 2


def test_add_mean_p50_band_show_p50_false(percentile_df: pd.DataFrame) -> None:
    """When show_p50=False, only mean and band traces are added."""
    fig = go.Figure()
    add_mean_p50_band(
        fig, percentile_df, "stage_id", "Hydro", "#3B82F6", show_p50=False
    )
    # mean + p10 lower + p90 upper = 3
    assert len(fig.data) == 3


# ---------------------------------------------------------------------------
# add_bounds_overlay
# ---------------------------------------------------------------------------


def test_add_bounds_overlay_both(bounds_df: pd.DataFrame) -> None:
    """Two traces (min + max) are added when both columns are specified."""
    fig = go.Figure()
    result = add_bounds_overlay(
        fig,
        bounds_df,
        "stage_id",
        min_col="min_storage",
        max_col="max_storage",
    )

    assert len(result.data) == 2
    for trace in result.data:
        assert trace.line.dash == "dash"
        assert trace.line.color == "#6B7280"


def test_add_bounds_overlay_min_only(bounds_df: pd.DataFrame) -> None:
    """Only one trace is added when only min_col is specified."""
    fig = go.Figure()
    result = add_bounds_overlay(fig, bounds_df, "stage_id", min_col="min_storage")

    assert len(result.data) == 1
    assert "min_storage" in result.data[0].name


def test_add_bounds_overlay_max_only(bounds_df: pd.DataFrame) -> None:
    """Only one trace is added when only max_col is specified."""
    fig = go.Figure()
    result = add_bounds_overlay(fig, bounds_df, "stage_id", max_col="max_storage")

    assert len(result.data) == 1
    assert "max_storage" in result.data[0].name


def test_add_bounds_overlay_none(bounds_df: pd.DataFrame) -> None:
    """No-op when both min_col and max_col are None."""
    fig = go.Figure()
    result = add_bounds_overlay(fig, bounds_df, "stage_id")

    assert len(result.data) == 0


def test_add_bounds_overlay_returns_figure(bounds_df: pd.DataFrame) -> None:
    """The function returns the same figure object (chaining support)."""
    fig = go.Figure()
    returned = add_bounds_overlay(fig, bounds_df, "stage_id", min_col="min_storage")
    assert returned is fig


# ---------------------------------------------------------------------------
# make_chart_card
# ---------------------------------------------------------------------------


def test_make_chart_card_html() -> None:
    """Output contains the chart-card class and figure HTML."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))

    html = make_chart_card(fig, "Test Chart", "chart-test-001")

    assert "chart-card" in html
    assert "chart-test-001" in html
    # fig_to_html produces a div element
    assert "<div" in html


def test_make_chart_card_none_raises() -> None:
    """Passing None as fig raises ValueError."""
    with pytest.raises(ValueError, match="fig must not be None"):
        make_chart_card(None, "Title", "chart-id")


def test_make_chart_card_contains_expand_button() -> None:
    """The expand button SVG (from wrap_chart) is present in the output."""
    fig = go.Figure()
    html = make_chart_card(fig, "My Chart", "chart-abc", height=300)

    assert "expand-btn" in html


def test_make_chart_card_no_plotlyjs() -> None:
    """The output must not embed the plotly.js script tag."""
    fig = go.Figure()
    html = make_chart_card(fig, "Chart", "chart-xyz")

    assert "cdn.plot.ly" not in html
    assert "plotly.min.js" not in html
    assert "include_plotlyjs" not in html


# ---------------------------------------------------------------------------
# compute_npv_costs
# ---------------------------------------------------------------------------


def test_compute_npv_costs_basic() -> None:
    """Discount factors 1/(1.1)^stage are applied to cost columns per stage."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [0, 0, 0],
            "thermal_generation_cost": [100.0, 100.0, 100.0],
        }
    )
    result = compute_npv_costs(df, 0.10)

    assert result["thermal_generation_cost"].iloc[0] == pytest.approx(100.0)
    assert result["thermal_generation_cost"].iloc[1] == pytest.approx(
        100.0 / 1.1, rel=1e-4
    )
    assert result["thermal_generation_cost"].iloc[2] == pytest.approx(
        100.0 / (1.1**2), rel=1e-4
    )


def test_compute_npv_costs_zero_rate() -> None:
    """A discount rate of 0.0 leaves all cost values unchanged."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [0, 0, 0],
            "thermal_generation_cost": [50.0, 75.0, 100.0],
        }
    )
    result = compute_npv_costs(df, 0.0)

    pd.testing.assert_series_equal(
        result["thermal_generation_cost"],
        df["thermal_generation_cost"],
    )


def test_compute_npv_costs_empty() -> None:
    """Empty DataFrame input returns an empty DataFrame (no error)."""
    empty = pd.DataFrame(columns=["stage_id", "scenario_id", "thermal_generation_cost"])
    result = compute_npv_costs(empty, 0.10)

    assert result.empty
    assert list(result.columns) == list(empty.columns)


def test_compute_npv_costs_does_not_mutate() -> None:
    """Input DataFrame is not modified in place."""
    df = pd.DataFrame({"stage_id": [0, 1], "thermal_generation_cost": [100.0, 100.0]})
    original_values = df["thermal_generation_cost"].tolist()
    compute_npv_costs(df, 0.10)

    assert df["thermal_generation_cost"].tolist() == original_values


def test_compute_npv_costs_metadata_cols_unchanged() -> None:
    """scenario_id, stage_id, and block_id columns are never discounted."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1, 2],
            "scenario_id": [7, 7, 7],
            "block_id": [1, 1, 1],
            "thermal_generation_cost": [100.0, 100.0, 100.0],
        }
    )
    result = compute_npv_costs(df, 0.10)

    assert result["scenario_id"].tolist() == [7, 7, 7]
    assert result["stage_id"].tolist() == [0, 1, 2]
    assert result["block_id"].tolist() == [1, 1, 1]


# ---------------------------------------------------------------------------
# group_costs
# ---------------------------------------------------------------------------


def test_group_costs_known_groups() -> None:
    """Thermal and Deficit columns are summed into their respective groups."""
    df = pd.DataFrame(
        {
            "stage_id": [0],
            "scenario_id": [0],
            "thermal_generation_cost": [50.0],
            "thermal_startup_cost": [10.0],
            "deficit_cost_depth_1": [20.0],
        }
    )
    cost_cols = [
        "thermal_generation_cost",
        "thermal_startup_cost",
        "deficit_cost_depth_1",
    ]
    result = group_costs(df, cost_cols)

    assert result["Thermal"].iloc[0] == pytest.approx(60.0)
    assert result["Deficit"].iloc[0] == pytest.approx(20.0)
    # Component columns should be gone
    assert "thermal_generation_cost" not in result.columns
    assert "thermal_startup_cost" not in result.columns
    assert "deficit_cost_depth_1" not in result.columns


def test_group_costs_other_column() -> None:
    """Unrecognized cost columns are accumulated in the 'Other' group."""
    df = pd.DataFrame(
        {
            "stage_id": [0],
            "scenario_id": [0],
            "future_cost": [999.0],
        }
    )
    cost_cols = ["future_cost"]
    result = group_costs(df, cost_cols)

    assert result["Other"].iloc[0] == pytest.approx(999.0)
    assert "future_cost" not in result.columns


def test_group_costs_missing_columns() -> None:
    """Missing columns in COST_GROUPS components do not raise an error."""
    # Only thermal_generation_cost present; thermal_startup_cost is absent
    df = pd.DataFrame(
        {
            "stage_id": [0],
            "scenario_id": [0],
            "thermal_generation_cost": [80.0],
        }
    )
    cost_cols = ["thermal_generation_cost"]
    result = group_costs(df, cost_cols)

    # Thermal group contains only the present column
    assert result["Thermal"].iloc[0] == pytest.approx(80.0)
    # Other should be 0.0 since nothing is unassigned
    assert result["Other"].iloc[0] == pytest.approx(0.0)


def test_group_costs_empty() -> None:
    """Empty DataFrame returns an empty DataFrame without error."""
    empty = pd.DataFrame(columns=["stage_id", "scenario_id", "thermal_generation_cost"])
    result = group_costs(empty, ["thermal_generation_cost"])

    assert result.empty


def test_group_costs_preserves_metadata() -> None:
    """Non-cost columns (scenario_id, stage_id) are preserved in output."""
    df = pd.DataFrame(
        {
            "stage_id": [0, 1],
            "scenario_id": [3, 3],
            "spillage_cost": [5.0, 10.0],
        }
    )
    result = group_costs(df, ["spillage_cost"])

    assert "stage_id" in result.columns
    assert "scenario_id" in result.columns
    assert result["scenario_id"].tolist() == [3, 3]


# ---------------------------------------------------------------------------
# compute_cost_summary
# ---------------------------------------------------------------------------


def _make_costs_df(n_scenarios: int = 100, n_stages: int = 10) -> pd.DataFrame:
    """Helper: build a costs DataFrame with 100 scenarios x 10 stages."""
    rows = []
    for sc in range(n_scenarios):
        for st in range(n_stages):
            rows.append(
                {
                    "scenario_id": sc,
                    "stage_id": st,
                    "thermal_generation_cost": 50.0 + sc * 0.1,
                    "deficit_cost_depth_1": 20.0 + sc * 0.05,
                    "spillage_cost": 5.0,
                }
            )
    return pd.DataFrame(rows)


def test_cost_summary_shape() -> None:
    """Output has exactly the expected columns and one row per cost group."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    assert list(result.columns) == [
        "group",
        "mean",
        "std",
        "p5",
        "p10",
        "p90",
        "p95",
        "pct",
    ]
    # Groups present: Thermal, Deficit, Spillage, NCS, Violations, Other
    assert len(result) == len(COST_GROUPS)


def test_cost_summary_pct_sums_to_100() -> None:
    """The 'pct' column sums to approximately 100.0 across all groups."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    assert result["pct"].sum() == pytest.approx(100.0, rel=1e-4)


def test_cost_summary_sorted_descending() -> None:
    """Rows are sorted by mean in descending order."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    means = result["mean"].tolist()
    assert means == sorted(means, reverse=True)


def test_cost_summary_empty() -> None:
    """Empty input returns an empty DataFrame with the expected columns."""
    empty = pd.DataFrame(columns=["scenario_id", "stage_id", "thermal_generation_cost"])
    result = compute_cost_summary(empty, 0.12)

    assert result.empty
    assert list(result.columns) == [
        "group",
        "mean",
        "std",
        "p5",
        "p10",
        "p90",
        "p95",
        "pct",
    ]


# ---------------------------------------------------------------------------
# compute_cost_summary — ticket-007: p5 and p95 columns
# ---------------------------------------------------------------------------


def test_cost_summary_has_p5_and_p95_columns() -> None:
    """compute_cost_summary must return columns 'p5' and 'p95'."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    assert "p5" in result.columns
    assert "p95" in result.columns


def test_cost_summary_p5_le_p10_le_mean_le_p90_le_p95() -> None:
    """For every row: p5 <= p10 <= mean <= p90 <= p95."""
    df = _make_costs_df()
    result = compute_cost_summary(df, 0.12)

    # Only check rows with positive mean (zero-cost groups have equal percentiles)
    non_zero = result[result["mean"] > 0]
    for _, row in non_zero.iterrows():
        assert row["p5"] <= row["p10"] + 1e-9, f"p5 > p10 for group {row['group']}"
        assert row["p10"] <= row["mean"] + 1e-9, f"p10 > mean for group {row['group']}"
        assert row["mean"] <= row["p90"] + 1e-9, f"mean > p90 for group {row['group']}"
        assert row["p90"] <= row["p95"] + 1e-9, f"p90 > p95 for group {row['group']}"


# ---------------------------------------------------------------------------
# COST_GROUP_COLORS completeness
# ---------------------------------------------------------------------------


def test_cost_group_colors_keys() -> None:
    """Every key in COST_GROUPS has a corresponding colour in COST_GROUP_COLORS."""
    for group_name in COST_GROUPS:
        assert group_name in COST_GROUP_COLORS, (
            f"COST_GROUP_COLORS is missing an entry for group '{group_name}'"
        )
