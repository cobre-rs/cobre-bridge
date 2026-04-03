"""Unit tests for the cobre_bridge.ui package.

Covers all 5 modules: theme.py, plotly_helpers.py, css.py, js.py, html.py.
Also verifies backward-compatible re-exports from comparators.html_report.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pytest

from cobre_bridge.ui.css import comparison_css, dashboard_css
from cobre_bridge.ui.html import (
    build_html,
    chart_grid,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.js import TAB_SWITCH_JS
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    fig_to_html,
    plotly_div,
    stage_x_labels,
)
from cobre_bridge.ui.theme import (
    BUS_COLORS,
    CHART_PALETTES,
    COLORS,
    COMPARISON_COLORS,
    COPPER_ACCENT,
)


@pytest.fixture
def simple_figure() -> go.Figure:
    """Minimal Plotly figure with one scatter trace."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="test"))
    return fig


@pytest.fixture
def simple_tab_defs() -> list[tuple[str, str]]:
    return [("tab-a", "Tab A"), ("tab-b", "Tab B")]


@pytest.fixture
def simple_tab_contents() -> dict[str, str]:
    return {"tab-a": "<p>Content A</p>", "tab-b": "<p>Content B</p>"}


def test_theme_colors_complete() -> None:
    """COLORS must have exactly 11 keys covering every semantic role."""
    expected_keys = {
        "hydro",
        "thermal",
        "ncs",
        "load",
        "deficit",
        "spillage",
        "curtailment",
        "exchange",
        "lower_bound",
        "upper_bound",
        "future_cost",
    }
    assert set(COLORS.keys()) == expected_keys


def test_theme_colors_are_hex_strings() -> None:
    """Every value in COLORS must be a non-empty hex color string."""
    for key, value in COLORS.items():
        assert isinstance(value, str), f"COLORS[{key!r}] is not a string"
        assert value.startswith("#"), f"COLORS[{key!r}] does not start with '#'"


def test_theme_bus_colors_count() -> None:
    """BUS_COLORS must contain exactly 5 elements."""
    assert len(BUS_COLORS) == 5


def test_theme_bus_colors_are_hex_strings() -> None:
    """Every entry in BUS_COLORS must be a hex color string."""
    for i, color in enumerate(BUS_COLORS):
        assert isinstance(color, str), f"BUS_COLORS[{i}] is not a string"
        assert color.startswith("#"), f"BUS_COLORS[{i}] does not start with '#'"


def test_theme_copper_accent_value() -> None:
    """COPPER_ACCENT must be the documented copper tone."""
    assert COPPER_ACCENT == "#B87333"


def test_theme_comparison_colors_keys() -> None:
    """COMPARISON_COLORS must contain 'cobre', 'newave', 'diff', 'match'."""
    assert set(COMPARISON_COLORS.keys()) == {"cobre", "newave", "diff", "match"}


def test_theme_chart_palettes_default_mirrors_bus_colors() -> None:
    """CHART_PALETTES['default'] must be identical to BUS_COLORS."""
    assert "default" in CHART_PALETTES
    assert CHART_PALETTES["default"] is BUS_COLORS


def test_legend_defaults_orientation() -> None:
    """LEGEND_DEFAULTS must specify horizontal orientation."""
    assert LEGEND_DEFAULTS["orientation"] == "h"


def test_margin_defaults_keys() -> None:
    """MARGIN_DEFAULTS must provide l, r, t, b keys."""
    assert {"l", "r", "t", "b"} <= set(MARGIN_DEFAULTS.keys())


def test_fig_to_html_output(simple_figure: go.Figure) -> None:
    """fig_to_html must return a fragment with plotly-graph-div, no <html> tag."""
    result = fig_to_html(simple_figure)
    assert "plotly-graph-div" in result
    assert "<html" not in result


def test_fig_to_html_no_plotlyjs_script(simple_figure: go.Figure) -> None:
    """fig_to_html must not embed plotly.js — the caller loads it separately."""
    result = fig_to_html(simple_figure)
    assert "cdn.plot.ly" not in result
    assert "plotly.min.js" not in result


def test_plotly_div_output() -> None:
    """plotly_div must produce a div/script pair calling Plotly.newPlot."""
    traces = [{"type": "scatter", "x": [1, 2], "y": [3, 4]}]
    result = plotly_div(traces, layout={"title": "Test"})
    assert "Plotly.newPlot" in result
    assert "<div" in result
    assert "<script>" in result


def test_plotly_div_contains_chart_id_prefix() -> None:
    """plotly_div div id must match the 'chart-' prefix."""
    traces: list[dict] = []
    result = plotly_div(traces, layout={})
    assert 'id="chart-' in result


def test_stage_x_labels_mapped() -> None:
    """stage_x_labels must return mapped label when id is present."""
    labels = {1: "Jan/2024", 2: "Feb/2024"}
    result = stage_x_labels([1, 2], labels)
    assert result == ["Jan/2024", "Feb/2024"]


def test_stage_x_labels_fallback_to_string() -> None:
    """stage_x_labels must fall back to str(id) for unmapped stage ids."""
    labels: dict[int, str] = {}
    result = stage_x_labels([5, 10], labels)
    assert result == ["5", "10"]


def test_stage_x_labels_mixed() -> None:
    """stage_x_labels must map known ids and fall back for unknown ones."""
    labels = {1: "Jan"}
    result = stage_x_labels([1, 2], labels)
    assert result == ["Jan", "2"]


def test_dashboard_css_selectors() -> None:
    """dashboard_css() must contain all required class selectors."""
    output = dashboard_css()
    required = [".chart-card", ".metric-card", ".section-title", ".data-table"]
    for selector in required:
        assert selector in output, f"Missing selector: {selector!r}"


def test_comparison_css_selectors() -> None:
    """comparison_css() must contain all required class selectors."""
    output = comparison_css()
    required = [".chart-card", ".metric-card", ".section-title", ".plant-selector"]
    for selector in required:
        assert selector in output, f"Missing selector: {selector!r}"


def test_dashboard_css_excludes_plant_selector() -> None:
    """dashboard_css() must not include .plant-selector (comparator-only)."""
    assert ".plant-selector" not in dashboard_css()


def test_comparison_css_excludes_data_table() -> None:
    """comparison_css() must not include .data-table (dashboard-only)."""
    assert ".data-table" not in comparison_css()


def test_dashboard_css_returns_string() -> None:
    """dashboard_css() must return a non-empty string."""
    result = dashboard_css()
    assert isinstance(result, str)
    assert len(result) > 0


def test_tab_switch_js_contains_show_tab_function() -> None:
    """TAB_SWITCH_JS must define a showTab function."""
    assert "function showTab" in TAB_SWITCH_JS


def test_tab_switch_js_contains_resize_dispatch() -> None:
    """TAB_SWITCH_JS must dispatch a resize event."""
    assert "dispatchEvent" in TAB_SWITCH_JS
    assert "resize" in TAB_SWITCH_JS


def test_tab_switch_js_contains_tab_id_param() -> None:
    """showTab signature must accept tabId and btn parameters."""
    assert "tabId" in TAB_SWITCH_JS
    assert "btn" in TAB_SWITCH_JS


def test_tab_switch_js_is_non_empty_string() -> None:
    """TAB_SWITCH_JS must be a non-empty string."""
    assert isinstance(TAB_SWITCH_JS, str)
    assert len(TAB_SWITCH_JS.strip()) > 0


def test_wrap_chart_produces_chart_card() -> None:
    """wrap_chart must wrap content in a chart-card div."""
    result = wrap_chart("<p>Chart</p>")
    assert result == '<div class="chart-card"><p>Chart</p></div>'


def test_section_title_produces_section_title_div() -> None:
    """section_title must produce a div with class section-title."""
    result = section_title("Overview")
    assert 'class="section-title"' in result
    assert "Overview" in result


def test_metric_card_contains_value_and_label() -> None:
    """metric_card must contain the metric-value and metric-label divs."""
    result = metric_card("42 MW", "Total Load")
    assert "42 MW" in result
    assert "Total Load" in result
    assert 'class="metric-value"' in result
    assert 'class="metric-label"' in result


def test_metrics_grid_wraps_cards() -> None:
    """metrics_grid must wrap cards in a metrics-grid container."""
    cards = [metric_card("1", "A"), metric_card("2", "B")]
    result = metrics_grid(cards)
    assert 'class="metrics-grid"' in result
    assert "1" in result
    assert "2" in result


def test_chart_grid_default_uses_chart_grid_class() -> None:
    """chart_grid without single=True must use the chart-grid class."""
    result = chart_grid(["<div>A</div>"])
    assert 'class="chart-grid"' in result


def test_chart_grid_single_uses_single_class() -> None:
    """chart_grid with single=True must use the chart-grid-single class."""
    result = chart_grid(["<div>A</div>"], single=True)
    assert 'class="chart-grid-single"' in result


def test_build_html_structure(
    simple_tab_defs: list[tuple[str, str]],
    simple_tab_contents: dict[str, str],
) -> None:
    """build_html must produce a complete HTML document with required elements."""
    result = build_html(
        title="Test Report",
        tab_defs=simple_tab_defs,
        tab_contents=simple_tab_contents,
        css=dashboard_css(),
        js=TAB_SWITCH_JS,
    )
    assert "<!DOCTYPE html>" in result
    assert "plotly-2.35.2.min.js" in result
    assert "<header>" in result
    assert "<nav>" in result
    assert "<main>" in result
    assert "<script>" in result


def test_build_html_first_tab_active(
    simple_tab_defs: list[tuple[str, str]],
    simple_tab_contents: dict[str, str],
) -> None:
    """build_html must mark only the first tab as active."""
    result = build_html(
        title="Test",
        tab_defs=simple_tab_defs,
        tab_contents=simple_tab_contents,
        css="",
        js="",
    )
    # First nav button should be class="active"
    assert 'class="active"' in result
    # First section should contain 'active' in its class
    assert "tab-content active" in result


def test_build_html_missing_tab_content_renders_no_data(
    simple_tab_defs: list[tuple[str, str]],
) -> None:
    """build_html must render '<p>No data</p>' for tabs missing from tab_contents."""
    result = build_html(
        title="Test",
        tab_defs=simple_tab_defs,
        tab_contents={},  # Empty — both tabs missing
        css="",
        js="",
    )
    assert "<p>No data</p>" in result


def test_build_html_title_in_header(
    simple_tab_defs: list[tuple[str, str]],
    simple_tab_contents: dict[str, str],
) -> None:
    """build_html must include the title in both <title> and <header>."""
    result = build_html(
        title="My Report",
        tab_defs=simple_tab_defs,
        tab_contents=simple_tab_contents,
        css="",
        js="",
    )
    assert "<title>My Report</title>" in result
    assert "<header>My Report</header>" in result


def test_comparators_backward_compat() -> None:
    """All symbols re-exported from comparators.html_report must be importable."""
    from cobre_bridge.comparators.html_report import (  # noqa: PLC0415
        COLOR_COBRE,
        COLOR_DIFF,
        COLOR_MATCH,
        COLOR_NEWAVE,
        CSS,
        JS,
        build_comparison_html,
        chart_grid,
        metric_card,
        metrics_grid,
        section_title,
        wrap_chart,
    )

    assert isinstance(CSS, str) and len(CSS) > 0
    assert isinstance(JS, str) and len(JS) > 0
    assert COLOR_COBRE == COMPARISON_COLORS["cobre"]
    assert COLOR_NEWAVE == COMPARISON_COLORS["newave"]
    assert COLOR_DIFF == COMPARISON_COLORS["diff"]
    assert COLOR_MATCH == COMPARISON_COLORS["match"]
    assert callable(wrap_chart)
    assert callable(section_title)
    assert callable(chart_grid)
    assert callable(metrics_grid)
    assert callable(metric_card)
    assert callable(build_comparison_html)
