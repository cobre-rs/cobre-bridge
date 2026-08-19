"""Plotly rendering helpers: layout defaults, HTML conversion, and stage labels."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass

import plotly.graph_objects as go

from cobre_bridge.ui.html import json_for_script

LEGEND_DEFAULTS: dict = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="center",
    x=0.5,
    font=dict(size=11),
)

MARGIN_DEFAULTS: dict = dict(l=60, r=30, t=60, b=50)


@dataclass(frozen=True, slots=True)
class FacetPanel:
    """An immutable subplot-domain record for one facet of a small-multiples grid.

    ``x_domain``/``y_domain`` are 2-element ``list[float]`` ``[lo, hi]`` Plotly
    axis-domain fractions, rounded to the helper's ``ndigits``. They are lists
    (not tuples) because ``list[float]`` matches the Plotly layout-dict
    expectation and preserves type identity with the legacy inline
    ``[round(x0, 3), round(x1, 3)]`` list literals fed into
    ``layout["xaxis"]["domain"]`` (this is a type-identity concern, NOT a
    JSON-serialization difference: ``json.dumps`` emits tuples and lists alike).
    """

    row: int
    col: int
    x_domain: list[float]
    y_domain: list[float]


def facet_grid(
    n: int,
    *,
    ncols: int = 2,
    row_gap: float = 0.06,
    col_gap: float = 0.05,
    min_row_h: float = 0.001,
    ndigits: int = 3,
) -> list[FacetPanel]:
    """Compute Plotly subplot domains for an ``n``-panel faceted grid.

    Reproduces the gap-based small-multiples layout arithmetic that the chart
    functions in ``comparators.charts`` hand-copy: ``nrows`` rows of ``ncols``
    columns, with ``row_gap``/``col_gap`` fractional gaps between panels and a
    ``min_row_h`` floor on row height. Returns one :class:`FacetPanel` per panel
    in index order (``0..n-1``); ``n == 0`` returns ``[]``.

    The arithmetic and rounding are byte-identical to the legacy inline code:
    intermediates are never rounded, only the final domain lists are rounded to
    ``ndigits``, and the single ``max(..., min_row_h)`` clamp covers both the
    clamped grids (``min_row_h=0.001``) and the unclamped vertical-stack chart
    (``min_row_h=0.0``). The ``-0.0`` that ``round`` can yield is preserved.
    """
    if n == 0:
        return []
    nrows = (n + ncols - 1) // ncols
    # Cap the effective row gap so nrows rows + gaps always fit within [0, 1].
    # A fixed row_gap times many rows otherwise sums past 1.0, marching the
    # y-domains negative (invisible/garbled bottom facets) for tall grids. For
    # grids where the requested gap already fits, eff_row_gap == row_gap — a
    # no-op, so existing small-multiples layouts (and their goldens) are
    # unchanged; the cap only engages once a grid would overflow.
    eff_row_gap = row_gap
    if nrows > 1:
        max_total_gap = max(1.0 - nrows * min_row_h, 0.0)
        eff_row_gap = min(row_gap, max_total_gap / (nrows - 1))
    row_h = max((1.0 - eff_row_gap * (nrows - 1)) / nrows, min_row_h)
    col_w = (1.0 - col_gap * (ncols - 1)) / ncols
    panels: list[FacetPanel] = []
    for idx in range(n):
        row_i = idx // ncols
        col_i = idx % ncols
        x0 = col_i * (col_w + col_gap)
        x1 = x0 + col_w
        y1 = 1.0 - row_i * (row_h + eff_row_gap)
        y0 = y1 - row_h
        panels.append(
            FacetPanel(
                row=row_i,
                col=col_i,
                x_domain=[round(x0, ndigits), round(x1, ndigits)],
                y_domain=[round(y0, ndigits), round(y1, ndigits)],
            )
        )
    return panels


def stage_x_labels(stage_ids: Sequence[int], labels: dict[int, str]) -> list[str]:
    """Map stage ids to human-readable labels."""
    return [labels.get(int(s), str(s)) for s in stage_ids]


def stage_x_dates(stage_ids: Sequence[int], stage_dates: dict[int, str]) -> list[str]:
    """Map stage ids to ISO ``YYYY-MM-DD`` x-positions (fallback: the id as text).

    Paired with :func:`stage_x_labels` (tick text) and :func:`apply_stage_date_axis`,
    these positions place each stage on a Plotly ``type="date"`` axis at its true
    calendar distance, so weekly stages sit closer than a following monthly one.
    """
    return [stage_dates.get(int(s), str(s)) for s in stage_ids]


def apply_stage_date_axis(
    fig: go.Figure, x_dates: Sequence[str], x_labels: Sequence[str]
) -> go.Figure:
    """Make *fig*'s x-axis a proportionally-spaced stage-date axis.

    Sets ``type="date"`` and pins the ticks to the stage dates (*x_dates*, the
    trace x-positions) rendered with the compact *x_labels* as tick text. Stages
    thus plot at their real calendar spacing while the axis keeps the readable
    date labels. Mutates *fig* in place and returns it.
    """
    fig.update_xaxes(
        type="date",
        tickmode="array",
        tickvals=list(x_dates),
        ticktext=list(x_labels),
    )
    return fig


def fig_to_html(fig: go.Figure, unified_hover: bool = True) -> str:
    """Convert a Plotly Figure to an HTML fragment.

    Returns an HTML string with no full document wrapper and no plotly.js
    script tag.  The caller is responsible for including plotly.js separately.
    """
    if unified_hover:
        fig.update_layout(hovermode="x unified")
    fig.update_layout(autosize=True, width=None)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        default_width="100%",
        config={"responsive": True},
    )


def apply_standard_layout(fig: go.Figure, **layout: object) -> go.Figure:
    """Apply the dashboard's standard ``legend``/``margin`` defaults to *fig*.

    Replaces the repeated ``fig.update_layout(..., legend=LEGEND_DEFAULTS,
    margin=MARGIN_DEFAULTS, ...)``. ``legend`` and ``margin`` default to the
    shared constants but can be overridden in ``**layout``; everything else
    (``title``, ``xaxis_title``, ``barmode``, ``height``, …) is forwarded to
    :meth:`plotly.graph_objects.Figure.update_layout`. Mutates *fig* in place and
    also returns it, so it fits both ``make_chart_card(apply_standard_layout(...))``
    and ``return apply_standard_layout(...)`` call sites.
    """
    layout.setdefault("legend", LEGEND_DEFAULTS)
    layout.setdefault("margin", MARGIN_DEFAULTS)
    fig.update_layout(**layout)
    return fig


def render_figure(
    fig: go.Figure,
    *,
    unified_hover: bool = True,
    **layout: object,
) -> str:
    """Apply the standard layout to *fig* and return an HTML fragment.

    Equivalent to :func:`apply_standard_layout` followed by :func:`fig_to_html`;
    collapses the ``fig.update_layout(..., legend=LEGEND_DEFAULTS,
    margin=MARGIN_DEFAULTS, ...)`` + ``fig_to_html(fig)`` boilerplate into one
    call for the tabs that render straight to HTML.
    """
    return fig_to_html(
        apply_standard_layout(fig, **layout), unified_hover=unified_hover
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
    layout.setdefault("hovermode", "x unified")

    data_json = json_for_script(traces)
    layout_json = json_for_script(layout)

    return (
        f'<div id="{div_id}"></div>\n'
        "<script>"
        f"Plotly.newPlot('{div_id}', {data_json}, {layout_json}, "
        "{responsive: true});"
        "</script>"
    )
