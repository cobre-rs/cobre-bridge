"""NCS tab module for the Cobre dashboard.

Displays NCS available vs generated and curtailment by source charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

from cobre_bridge.dashboard.data import entity_name
from cobre_bridge.ui.html import section_title, wrap_chart
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    fig_to_html,
    stage_x_labels,
)
from cobre_bridge.ui.theme import COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-ncs-thermal"
TAB_LABEL = "NCS"
TAB_ORDER = 90

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_ncs_available_vs_generated(
    ncs_lf: pl.LazyFrame,
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Stacked area of NCS available vs generated with curtailment gap."""
    data = (
        ncs_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "non_controllable_id"])
        .agg(
            (pl.col("available_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
            (pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
            (pl.col("curtailment_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
        )
        .group_by(["scenario_id", "stage_id"])
        .agg(
            pl.col("available_mw").sum(),
            pl.col("generation_mw").sum(),
            pl.col("curtailment_mw").sum(),
        )
        .group_by("stage_id")
        .agg(
            pl.col("available_mw").mean(),
            pl.col("generation_mw").mean(),
            pl.col("curtailment_mw").mean(),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = data["stage_id"].to_list()
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=data["available_mw"].to_list(),
            name="Available",
            line={"color": "#A5D6A7", "width": 2, "dash": "dash"},
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=data["generation_mw"].to_list(),
            name="Generated",
            stackgroup="ncs",
            fillcolor="rgba(76,175,80,0.7)",
            line={"color": COLORS["ncs"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=data["curtailment_mw"].to_list(),
            name="Curtailment",
            stackgroup="ncs",
            fillcolor="rgba(121,85,72,0.6)",
            line={"color": COLORS["curtailment"]},
        )
    )
    fig.update_layout(
        title="NCS Available vs Generated (block-hours weighted avg, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_ncs_curtailment_by_source(
    ncs_lf: pl.LazyFrame,
    names: dict[tuple[str, int], str],
    top_n: int = 20,
) -> str:
    """Bar chart of top curtailed NCS sources."""
    curt_by = (
        ncs_lf.group_by(["scenario_id", "non_controllable_id"])
        .agg(pl.col("curtailment_mwh").sum())
        .group_by("non_controllable_id")
        .agg(pl.col("curtailment_mwh").mean())
        .sort("curtailment_mwh", descending=True)
        .head(top_n)
        .collect(engine="streaming")
    )
    nids = curt_by["non_controllable_id"].to_list()
    ynames = [entity_name(names, "non_controllable_sources", int(nid)) for nid in nids]
    values = [v / 1e3 for v in curt_by["curtailment_mwh"].to_list()]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=ynames,
            orientation="h",
            marker_color=COLORS["curtailment"],
        )
    )
    fig.update_layout(
        title=f"Top {top_n} NCS Sources by Average Curtailment",
        xaxis_title="Avg Curtailment (GWh)",
        height=max(350, top_n * 22 + 100),
        margin=dict(l=100, r=30, t=60, b=50),
        showlegend=False,
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """NCS tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the NCS tab content area."""
    return (
        section_title("NCS Available vs Generated")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_ncs_available_vs_generated(data.ncs_lf, data.stage_labels, data.bh_df)
        )
        + "</div>"
        + section_title("NCS Curtailment by Source")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_ncs_curtailment_by_source(data.ncs_lf, data.names))
        + "</div>"
    )
