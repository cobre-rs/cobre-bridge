"""HTML comparison report builder for interactive results analysis."""

from __future__ import annotations

from cobre_bridge.ui.css import comparison_css
from cobre_bridge.ui.html import (
    build_html,
    chart_grid,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.js import TAB_SWITCH_JS
from cobre_bridge.ui.theme import COMPARISON_COLORS

CSS = comparison_css()
JS = TAB_SWITCH_JS

COLOR_COBRE = COMPARISON_COLORS["cobre"]
COLOR_NEWAVE = COMPARISON_COLORS["newave"]
COLOR_DIFF = COMPARISON_COLORS["diff"]
COLOR_MATCH = COMPARISON_COLORS["match"]

COMPARISON_TABS = [
    ("tab-overview", "Overview"),
    ("tab-system", "System"),
    ("tab-balance", "Energy Balance"),
    ("tab-hydro", "Hydro Operation"),
    ("tab-hydro-detail", "Plant Details"),
    ("tab-thermal", "Thermal Operation"),
    ("tab-thermal-detail", "Thermal Details"),
    ("tab-productivity", "Productivity"),
]


def build_comparison_html(
    title: str,
    tab_contents: dict[str, str],
) -> str:
    """Build a complete self-contained HTML comparison report."""
    return build_html(
        title=title,
        tab_defs=COMPARISON_TABS,
        tab_contents=tab_contents,
        css=CSS,
        js=JS,
    )


__all__ = [
    "CSS",
    "JS",
    "COLOR_COBRE",
    "COLOR_NEWAVE",
    "COLOR_DIFF",
    "COLOR_MATCH",
    "COMPARISON_TABS",
    "metric_card",
    "section_title",
    "wrap_chart",
    "chart_grid",
    "metrics_grid",
    "build_comparison_html",
]
