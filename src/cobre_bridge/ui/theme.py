"""Shared visual design tokens for the cobre-bridge UI.

Single source of truth for all color constants used across the dashboard
and comparator HTML reports. Do not define colors elsewhere.
"""

from __future__ import annotations

COLORS: dict[str, str] = {
    "hydro": "#4A90B8",
    "thermal": "#F5A623",
    "ncs": "#4A8B6F",
    "load": "#374151",
    "deficit": "#DC4C4C",
    "spillage": "#B87333",
    "curtailment": "#8B5E3C",
    "exchange": "#4A90B8",
    "lower_bound": "#4A8B6F",
    "upper_bound": "#DC4C4C",
    "future_cost": "#8B9298",
}

BUS_COLORS: list[str] = ["#4A90B8", "#F5A623", "#4A8B6F", "#DC4C4C", "#B87333"]

COMPARISON_COLORS: dict[str, str] = {
    "cobre": "#4A90B8",
    "newave": "#F5A623",
    "diff": "#DC4C4C",
    "match": "#4A8B6F",
}

COPPER_ACCENT: str = "#B87333"

CHART_PALETTES: dict[str, list[str]] = {
    "default": BUS_COLORS,
}

GENERATION_COLORS: dict[str, str] = {
    "hydro": "#3B82F6",
    "thermal": "#F59E0B",
    "ncs": "#10B981",
}

PERFORMANCE_PHASE_COLORS: dict[str, str] = {
    "forward": "#3B82F6",
    "backward": "#14B8A6",
    "lp_solve": "#B87333",
    "overhead": "#6B7280",
}

BOUND_LINE_COLOR: str = "#6B7280"

BAND_OPACITY: float = 0.15


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour string to an ``rgba(...)`` CSS value.

    Args:
        hex_color: A hex colour string such as ``"#3B82F6"`` (the leading
            ``#`` is optional).
        alpha: Opacity in the range 0.0–1.0.

    Returns:
        An ``rgba(r, g, b, alpha)`` string suitable for Plotly's
        ``fillcolor`` parameter.
    """
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
