"""Plotly rendering helpers: layout defaults, HTML conversion, and stage labels."""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence

import plotly.graph_objects as go

LEGEND_DEFAULTS: dict = dict(
    orientation="h",
    yanchor="top",
    y=-0.15,
    xanchor="center",
    x=0.5,
    font=dict(size=11),
)

MARGIN_DEFAULTS: dict = dict(l=60, r=30, t=60, b=10)


def stage_x_labels(stage_ids: Sequence[int], labels: dict[int, str]) -> list[str]:
    """Map stage ids to human-readable labels."""
    return [labels.get(int(s), str(s)) for s in stage_ids]


def fig_to_html(fig: go.Figure, unified_hover: bool = True) -> str:
    """Convert a Plotly Figure to an HTML fragment.

    Returns an HTML string with no full document wrapper and no plotly.js
    script tag.  The caller is responsible for including plotly.js separately.
    """
    if unified_hover:
        fig.update_layout(hovermode="x unified")
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
    )


def plotly_div(
    traces: list[dict],
    layout: dict,
    height: int = 400,
) -> str:
    """Return a plotly div with inline data and layout.

    Generates a ``<div>`` + ``<script>`` pair that calls ``Plotly.newPlot``
    with the provided traces and layout.  Suitable for embedding in an HTML
    page that already loads plotly.js.
    """
    div_id = f"chart-{uuid.uuid4().hex[:8]}"
    layout.setdefault("height", height)
    layout.setdefault("margin", MARGIN_DEFAULTS)
    layout.setdefault("legend", LEGEND_DEFAULTS)
    layout.setdefault("template", "plotly_white")

    data_json = json.dumps(traces)
    layout_json = json.dumps(layout)

    return (
        f'<div id="{div_id}"></div>\n'
        "<script>"
        f"Plotly.newPlot('{div_id}', {data_json}, {layout_json}, "
        "{responsive: true});"
        "</script>"
    )
