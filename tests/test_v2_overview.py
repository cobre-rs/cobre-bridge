"""Unit tests for cobre_bridge.dashboard.tabs.v2_overview.

Covers module constants, can_render, helper functions, and the full
render() path including the empty-costs degradation branch.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import plotly.graph_objects as go
import polars as pl

import cobre_bridge.dashboard.tabs.v2_overview as v2_overview
from cobre_bridge.dashboard.chart_helpers import compute_cost_summary
from cobre_bridge.dashboard.tabs.v2_overview import (
    _build_cost_table,
    _chart_cost_bar,
    _chart_training_mini,
    _compute_gen_gwh,
    _run_identity_strip,
    _run_status_strip,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers / data factories
# ---------------------------------------------------------------------------


def _make_costs_df() -> pd.DataFrame:
    """Return a minimal costs DataFrame with two scenarios and three stages."""
    rows = []
    for scenario_id in range(2):
        for stage_id in range(3):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage_id": stage_id,
                    "thermal_generation_cost": float(1000 * (stage_id + 1)),
                    "deficit_cost_depth_1": float(50 * (stage_id + 1)),
                }
            )
    return pd.DataFrame(rows)


def _make_conv_df(n: int = 50) -> pd.DataFrame:
    """Return a minimal convergence DataFrame with *n* rows."""
    return pd.DataFrame(
        {
            "iteration": list(range(1, n + 1)),
            "lower_bound": [float(i * 100) for i in range(n)],
            "upper_bound_mean": [float(i * 110) for i in range(n)],
            "upper_bound_std": [float(5) for _ in range(n)],
            "gap_percent": [float(10 - i * 0.2) for i in range(n)],
        }
    )


def _make_mock_lf(gwh_value: float = 5000.0) -> pl.LazyFrame:
    """Return a LazyFrame with a single scenario/stage yielding *gwh_value* MWh."""
    return pl.DataFrame(
        {
            "scenario_id": [0],
            "stage_id": [0],
            "generation_mwh": [gwh_value],
        }
    ).lazy()


def _make_mock_data(
    *,
    costs: pd.DataFrame | None = None,
    conv: pd.DataFrame | None = None,
    training_manifest: dict | None = None,
    metadata: dict | None = None,
    case_name: str = "test-case",
    n_scenarios: int = 100,
    n_stages: int = 60,
    discount_rate: float = 0.12,
) -> MagicMock:
    """Build a minimal MagicMock that satisfies the DashboardData interface."""
    data = MagicMock()
    data.case_name = case_name
    data.n_scenarios = n_scenarios
    data.n_stages = n_stages
    data.discount_rate = discount_rate
    data.metadata = metadata if metadata is not None else {}
    data.training_manifest = training_manifest if training_manifest is not None else {}
    data.conv = conv if conv is not None else _make_conv_df()
    data.costs = costs if costs is not None else _make_costs_df()
    data.stage_labels = {0: "Jan 2024", 1: "Feb 2024", 2: "Mar 2024"}
    data.stage_hours = {0: 744.0, 1: 672.0, 2: 744.0}
    data.hydros_lf = _make_mock_lf(12_000.0)
    data.thermals_lf = _make_mock_lf(8_000.0)
    data.ncs_lf = _make_mock_lf(3_000.0)
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match the ticket specification exactly."""
    assert v2_overview.TAB_ID == "tab-v2-overview"
    assert v2_overview.TAB_LABEL == "Overview"
    assert v2_overview.TAB_ORDER == 0


# ---------------------------------------------------------------------------
# test_can_render_returns_true
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render must return True unconditionally."""
    data = _make_mock_data()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test_run_identity_strip
# ---------------------------------------------------------------------------


def test_run_identity_strip_contains_expected_values() -> None:
    """_run_identity_strip must embed case_name, scenario count, stage count,
    discount rate as percentage, and solver version."""
    data = _make_mock_data(
        case_name="test",
        n_scenarios=100,
        n_stages=60,
        discount_rate=0.12,
        metadata={"version": "1.0"},
    )
    html = _run_identity_strip(data)

    assert "test" in html
    assert "100" in html
    assert "60" in html
    # 0.12 -> 12.0%
    assert "12" in html
    assert "1.0" in html


def test_run_identity_strip_missing_version_shows_na() -> None:
    """When metadata has no 'version' key, 'N/A' must appear in the strip."""
    data = _make_mock_data(metadata={})
    html = _run_identity_strip(data)
    assert "N/A" in html


# ---------------------------------------------------------------------------
# test_run_status_strip
# ---------------------------------------------------------------------------


def test_run_status_strip_with_manifest() -> None:
    """When training_manifest contains termination_reason, both the reason
    and the iteration count must appear in the returned HTML."""
    conv = _make_conv_df(50)
    data = _make_mock_data(
        training_manifest={"termination_reason": "gap_tolerance"},
        conv=conv,
    )
    html = _run_status_strip(data)

    assert "gap_tolerance" in html
    assert "50" in html


def test_run_status_strip_empty_manifest() -> None:
    """When training_manifest is empty, the fallback message must appear."""
    data = _make_mock_data(training_manifest={})
    html = _run_status_strip(data)
    assert "No training manifest available" in html


# ---------------------------------------------------------------------------
# test_compute_gen_gwh
# ---------------------------------------------------------------------------


def test_compute_gen_gwh_returns_gwh() -> None:
    """_compute_gen_gwh must convert MWh sum to GWh (divide by 1e3)."""
    lf = pl.DataFrame(
        {
            "scenario_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mwh": [2000.0, 4000.0],
        }
    ).lazy()
    # Mean across scenarios: (2000 + 4000) / 2 = 3000 MWh -> 3.0 GWh
    result = _compute_gen_gwh(lf)
    assert abs(result - 3.0) < 0.001


def test_compute_gen_gwh_empty_lazyframe_returns_zero() -> None:
    """_compute_gen_gwh must return 0.0 for an empty LazyFrame."""
    lf = pl.LazyFrame({"scenario_id": [], "generation_mwh": []})
    result = _compute_gen_gwh(lf)
    assert result == 0.0


# ---------------------------------------------------------------------------
# test_build_cost_table
# ---------------------------------------------------------------------------


def test_build_cost_table_contains_table_and_thermal() -> None:
    """_build_cost_table must return HTML containing <table and group names."""
    costs = _make_costs_df()
    summary = compute_cost_summary(costs, 0.10)
    html = _build_cost_table(summary)

    assert "<table" in html
    assert "Thermal" in html


def test_build_cost_table_empty_df_returns_placeholder() -> None:
    """_build_cost_table on an empty DataFrame must return a <p> placeholder."""
    html = _build_cost_table(pd.DataFrame())
    assert "<table" not in html
    assert "No cost data" in html


# ---------------------------------------------------------------------------
# test_chart_cost_bar
# ---------------------------------------------------------------------------


def test_chart_cost_bar_produces_horizontal_bar_traces() -> None:
    """_chart_cost_bar must return a Figure with at least one horizontal bar."""
    costs = _make_costs_df()
    summary = compute_cost_summary(costs, 0.10)
    fig = _chart_cost_bar(summary)

    assert isinstance(fig, go.Figure)
    bar_traces = [t for t in fig.data if isinstance(t, go.Bar)]
    assert len(bar_traces) >= 1
    # All bars must be horizontal
    for trace in bar_traces:
        assert trace.orientation == "h"


# ---------------------------------------------------------------------------
# test_chart_training_mini
# ---------------------------------------------------------------------------


def test_chart_training_mini_returns_figure_for_valid_conv() -> None:
    """_chart_training_mini must return a Figure when conv has required columns."""
    conv = _make_conv_df(10)
    fig = _chart_training_mini(conv)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # lower_bound + upper_bound_mean


def test_chart_training_mini_returns_none_for_empty_conv() -> None:
    """_chart_training_mini must return None for an empty DataFrame."""
    result = _chart_training_mini(pd.DataFrame())
    assert result is None


def test_chart_training_mini_returns_none_for_missing_columns() -> None:
    """_chart_training_mini must return None when required columns are absent."""
    conv = pd.DataFrame({"iteration": [1, 2], "lower_bound": [100.0, 110.0]})
    result = _chart_training_mini(conv)
    assert result is None


# ---------------------------------------------------------------------------
# test_render_with_full_data
# ---------------------------------------------------------------------------


def test_render_with_full_data_contains_required_substrings() -> None:
    """render() on a fully-populated mock must contain all required substrings."""
    data = _make_mock_data()
    # Patch _stage_avg_mw so it doesn't try to execute the LazyFrame
    stage_mw = {0: 100.0, 1: 110.0, 2: 105.0}
    with patch(
        "cobre_bridge.dashboard.tabs.v2_overview._stage_avg_mw",
        return_value=stage_mw,
    ):
        html = render(data)

    assert "Expected Cost" in html
    assert "Hydro" in html
    assert "Thermal" in html
    assert "NCS" in html
    assert "Cost Breakdown" in html
    assert "Training Convergence" in html


def test_render_termination_reason_appears_in_output() -> None:
    """render() must surface the termination_reason from training_manifest."""
    data = _make_mock_data(
        training_manifest={"termination_reason": "gap_tolerance"},
    )
    with patch(
        "cobre_bridge.dashboard.tabs.v2_overview._stage_avg_mw",
        return_value={0: 100.0},
    ):
        html = render(data)

    assert "gap_tolerance" in html


# ---------------------------------------------------------------------------
# test_render_with_empty_costs
# ---------------------------------------------------------------------------


def test_render_with_empty_costs_does_not_raise_and_contains_placeholder() -> None:
    """render() with empty costs must not raise and must include placeholder text."""
    data = _make_mock_data(costs=pd.DataFrame())

    with patch(
        "cobre_bridge.dashboard.tabs.v2_overview._stage_avg_mw",
        return_value={0: 100.0},
    ):
        html = render(data)

    assert "No cost data" in html


def test_render_with_empty_conv_shows_no_convergence_placeholder() -> None:
    """render() with empty conv must emit the no-convergence placeholder."""
    data = _make_mock_data(conv=pd.DataFrame())

    with patch(
        "cobre_bridge.dashboard.tabs.v2_overview._stage_avg_mw",
        return_value={0: 100.0},
    ):
        html = render(data)

    assert "No convergence data" in html
