"""Unit tests for src/cobre_bridge/dashboard/tabs/v2_training.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from cobre_bridge.dashboard.tabs.v2_training import (
    TAB_ID,
    TAB_LABEL,
    TAB_ORDER,
    _build_metrics_row,
    _chart_convergence_hero,
    _chart_cut_activity_heatmap,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_conv(n: int = 10) -> pd.DataFrame:
    """Synthetic convergence DataFrame with *n* rows."""
    iterations = list(range(1, n + 1))
    lower_bound = [100.0 + i * 5.0 for i in range(n)]
    upper_bound_mean = [200.0 - i * 3.0 for i in range(n)]
    upper_bound_std = [10.0 - i * 0.5 for i in range(n)]
    gap_percent = [
        (ub - lb) / ub * 100 for lb, ub in zip(lower_bound, upper_bound_mean)
    ]
    cuts_active = [50 + i * 10 for i in range(n)]
    cuts_added = [5 + i % 3 for i in range(n)]
    return pd.DataFrame(
        {
            "iteration": iterations,
            "lower_bound": lower_bound,
            "upper_bound_mean": upper_bound_mean,
            "upper_bound_std": upper_bound_std,
            "gap_percent": gap_percent,
            "cuts_active": cuts_active,
            "cuts_added": cuts_added,
        }
    )


def _make_cut_selection(n_stages: int = 5, n_iters: int = 10) -> pd.DataFrame:
    """Synthetic cut_selection DataFrame."""
    rows = []
    for stage in range(1, n_stages + 1):
        for it in range(1, n_iters + 1):
            rows.append(
                {
                    "stage": stage,
                    "iteration": it,
                    "cuts_active_after": 50 + stage * 2,
                    "cuts_deactivated": stage % 3,
                }
            )
    return pd.DataFrame(rows)


def _make_timing(n: int = 10) -> pd.DataFrame:
    """Synthetic timing DataFrame."""
    iterations = list(range(1, n + 1))
    return pd.DataFrame(
        {
            "iteration": iterations,
            "forward_solve_ms": [100.0 + i for i in range(n)],
            "forward_sample_ms": [20.0 + i for i in range(n)],
            "backward_solve_ms": [150.0 + i for i in range(n)],
            "backward_cut_ms": [30.0 + i for i in range(n)],
            "cut_selection_ms": [40.0 + i for i in range(n)],
            "overhead_ms": [10.0 + i for i in range(n)],
        }
    )


def _make_mock_data(
    conv: pd.DataFrame | None = None,
    cut_selection: pd.DataFrame | None = None,
    timing: pd.DataFrame | None = None,
    training_manifest: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics DashboardData."""
    data = MagicMock()
    data.conv = conv if conv is not None else _make_conv()
    data.cut_selection = (
        cut_selection if cut_selection is not None else _make_cut_selection()
    )
    data.timing = timing if timing is not None else _make_timing()
    data.training_manifest = training_manifest if training_manifest is not None else {}
    data.stage_labels = {i: f"Stage {i}" for i in range(1, 20)}
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """TAB_ID, TAB_LABEL, and TAB_ORDER must match the spec."""
    assert TAB_ID == "tab-v2-training"
    assert TAB_LABEL == "Training"
    assert TAB_ORDER == 20


# ---------------------------------------------------------------------------
# test_can_render_returns_true
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render() must always return True regardless of data."""
    data = _make_mock_data()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test_build_metrics_row
# ---------------------------------------------------------------------------


def test_build_metrics_row() -> None:
    """_build_metrics_row() must include key labels and manifest termination reason."""
    data = _make_mock_data(
        conv=_make_conv(10),
        training_manifest={"termination_reason": "max_iterations"},
    )
    html = _build_metrics_row(data)
    assert "Lower Bound" in html
    assert "Gap" in html
    assert "max_iterations" in html
    assert "Upper Bound" in html
    assert "Total Iterations" in html
    assert "Termination" in html


def test_build_metrics_row_missing_manifest_shows_na() -> None:
    """When training_manifest is empty, termination shows 'N/A'."""
    data = _make_mock_data(training_manifest={})
    html = _build_metrics_row(data)
    assert "N/A" in html


# ---------------------------------------------------------------------------
# test_chart_convergence_hero_has_dropdown
# ---------------------------------------------------------------------------


def test_chart_convergence_hero_has_dropdown() -> None:
    """The hero convergence figure must have an updatemenus dropdown with 4 buttons."""
    conv = _make_conv(20)
    fig = _chart_convergence_hero(conv)
    assert fig.layout.updatemenus is not None
    assert len(fig.layout.updatemenus) > 0
    buttons = fig.layout.updatemenus[0].buttons
    assert len(buttons) == 4


def test_chart_convergence_hero_button_labels() -> None:
    """Dropdown buttons must be labeled: All, Last 200, Last 100, Last 50."""
    conv = _make_conv(20)
    fig = _chart_convergence_hero(conv)
    labels = [b.label for b in fig.layout.updatemenus[0].buttons]
    assert labels == ["All", "Last 200", "Last 100", "Last 50"]


# ---------------------------------------------------------------------------
# test_render_with_empty_cut_selection
# ---------------------------------------------------------------------------


def test_render_with_empty_cut_selection() -> None:
    """render() must not raise and must contain 'No cut selection data' when cut_selection is empty."""  # noqa: E501
    data = _make_mock_data(
        conv=_make_conv(10),
        cut_selection=pd.DataFrame(),
        timing=_make_timing(10),
    )
    html = render(data)
    assert "No cut selection data" in html


# ---------------------------------------------------------------------------
# test_render_with_empty_timing
# ---------------------------------------------------------------------------


def test_render_with_empty_timing() -> None:
    """render() must not raise and must contain 'No timing data' when timing is empty."""  # noqa: E501
    data = _make_mock_data(
        conv=_make_conv(10),
        cut_selection=_make_cut_selection(),
        timing=pd.DataFrame(),
    )
    html = render(data)
    assert "No timing data" in html


# ---------------------------------------------------------------------------
# test_chart_cut_activity_heatmap_empty
# ---------------------------------------------------------------------------


def test_chart_cut_activity_heatmap_empty() -> None:
    """_chart_cut_activity_heatmap() with empty DataFrame returns placeholder HTML."""
    html = _chart_cut_activity_heatmap(pd.DataFrame(), {})
    assert "No cut selection data" in html


# ---------------------------------------------------------------------------
# Additional render content checks
# ---------------------------------------------------------------------------


def test_render_contains_required_sections() -> None:
    """render() HTML must contain the substrings required by acceptance criteria."""
    data = _make_mock_data(conv=_make_conv(15))
    html = render(data)
    assert "Lower Bound" in html
    assert "Upper Bound" in html
    assert "Gap" in html
    assert "Convergence" in html
    assert "Cut Pool" in html


def test_render_empty_conv_returns_placeholder() -> None:
    """render() with empty conv must return a safe placeholder string."""
    data = _make_mock_data(conv=pd.DataFrame())
    html = render(data)
    assert "No convergence data" in html


def test_render_full_data_no_exception() -> None:
    """render() with all data present must complete without raising."""
    data = _make_mock_data(
        conv=_make_conv(20),
        cut_selection=_make_cut_selection(6, 20),
        timing=_make_timing(20),
        training_manifest={"termination_reason": "convergence"},
    )
    html = render(data)
    assert len(html) > 0


def test_render_gap_color_high() -> None:
    """When gap > 1.0, the gap metric card must use the red color."""
    conv = _make_conv(5)
    # Ensure gap is high
    conv["gap_percent"] = 10.0
    data = _make_mock_data(conv=conv)
    html = _build_metrics_row(data)
    assert "#DC4C4C" in html


def test_render_gap_color_low() -> None:
    """When gap <= 1.0, the gap metric card must use the green color."""
    conv = _make_conv(5)
    conv["gap_percent"] = 0.5
    data = _make_mock_data(conv=conv)
    html = _build_metrics_row(data)
    assert "#4A8B6F" in html


@pytest.mark.parametrize("n_iters", [50, 201, 400])
def test_chart_cut_activity_heatmap_sampling(n_iters: int) -> None:
    """Heatmap must succeed for both small and large iteration counts."""
    cs = _make_cut_selection(n_stages=4, n_iters=n_iters)
    html = _chart_cut_activity_heatmap(cs, {})
    # Should produce plotly HTML, not the placeholder
    assert "No cut selection data" not in html
    assert len(html) > 0
