"""Tests for the shared Plotly rendering helpers."""

from __future__ import annotations

import re

import plotly.graph_objects as go

from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    apply_standard_layout,
    fig_to_html,
    render_figure,
)


def _normalise_ids(s: str) -> str:
    """Replace Plotly's random div id (hex + hyphens) so structural HTML compares."""
    return re.sub(r"[0-9a-f]{4,}(-[0-9a-f]{2,})*", "ID", s)


def _bar_fig() -> go.Figure:
    return go.Figure([go.Bar(x=[1, 2], y=[3, 4], name="t")])


def test_render_figure_matches_manual_update_layout_pattern() -> None:
    """render_figure(...) must produce the exact HTML of the boilerplate it replaces.

    This is the contract the performance_charts conversion relied on:
    ``render_figure(fig, **kw)`` == ``fig.update_layout(legend=LEGEND_DEFAULTS,
    margin=MARGIN_DEFAULTS, **kw); fig_to_html(fig)``.
    """
    manual = _bar_fig()
    manual.update_layout(
        title="T",
        xaxis_title="X",
        yaxis_title="Y",
        barmode="stack",
        height=420,
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    manual_html = fig_to_html(manual)

    factory_html = render_figure(
        _bar_fig(),
        title="T",
        xaxis_title="X",
        yaxis_title="Y",
        barmode="stack",
        height=420,
    )

    assert _normalise_ids(manual_html) == _normalise_ids(factory_html)


def test_render_figure_forwards_unified_hover() -> None:
    manual = _bar_fig()
    manual.update_layout(title="A", legend=LEGEND_DEFAULTS, margin=MARGIN_DEFAULTS)
    manual_html = fig_to_html(manual, unified_hover=False)

    factory_html = render_figure(_bar_fig(), title="A", unified_hover=False)
    assert _normalise_ids(manual_html) == _normalise_ids(factory_html)


def test_apply_standard_layout_matches_manual_and_returns_fig() -> None:
    """apply_standard_layout(fig, **kw) mutates fig like the manual update_layout
    and returns it (for make_chart_card / return-fig call sites)."""
    manual = _bar_fig()
    manual.update_layout(
        title="T", xaxis_title="X", legend=LEGEND_DEFAULTS, margin=MARGIN_DEFAULTS
    )

    fig = _bar_fig()
    returned = apply_standard_layout(fig, title="T", xaxis_title="X")
    assert returned is fig  # mutates in place AND returns the same figure
    assert fig.to_dict()["layout"] == manual.to_dict()["layout"]


def test_render_figure_lets_caller_override_legend_and_margin() -> None:
    custom_margin = {"l": 10, "r": 10, "t": 10, "b": 10}
    fig = _bar_fig()
    render_figure(fig, title="C", margin=custom_margin)
    layout = fig.to_dict()["layout"]
    # caller's margin wins over the default
    assert layout["margin"] == custom_margin
    # legend defaulted to the shared constant
    assert layout["legend"]["orientation"] == LEGEND_DEFAULTS["orientation"]
