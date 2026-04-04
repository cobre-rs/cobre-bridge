"""Costs tab module for the Cobre dashboard.

Displays cost composition by stage and spot price by bus charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

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
from cobre_bridge.ui.theme import BUS_COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-costs"
TAB_LABEL = "Costs"
TAB_ORDER = 80

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_spot_price_by_bus(
    buses_lf: pl.LazyFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Spot price by bus by stage (block-hours weighted average)."""
    sp_data = (
        buses_lf.filter(pl.col("bus_id") <= 3)
        .join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg((pl.col("spot_price") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("spot_price").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    real_buses = sorted(sp_data["bus_id"].unique().to_list())
    stages = sorted(sp_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(real_buses):
        sub = sp_data.filter(pl.col("bus_id") == bus_id)
        sp_map = dict(zip(sub["stage_id"].to_list(), sub["spot_price"].to_list()))
        bname = bus_names.get(bus_id, str(bus_id))
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[sp_map.get(s, 0) for s in stages],
                name=bname,
                line={"color": BUS_COLORS[i % len(BUS_COLORS)], "width": 2},
            )
        )
    fig.update_layout(
        title="Spot Price by Bus by Stage (block-hours weighted avg, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Spot Price (R$/MWh)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_spot_price_by_bus_subplots(
    buses_lf: pl.LazyFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
) -> str:
    """2x2 subplots of weighted-average spot price (by block hours) per bus."""
    # Build block_hours as a Polars DataFrame for join
    bh_df = pl.DataFrame(
        {
            "stage_id": [k[0] for k in block_hours.keys()],
            "block_id": [k[1] for k in block_hours.keys()],
            "_bh": list(block_hours.values()),
        }
    )

    grp = (
        buses_lf.filter(pl.col("bus_id") <= 3)
        .join(bh_df.lazy(), on=["stage_id", "block_id"])
        .with_columns((pl.col("spot_price") * pl.col("_bh")).alias("_sp_x_bh"))
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg((pl.col("_sp_x_bh").sum() / pl.col("_bh").sum()).alias("w_spot"))
        .collect(engine="streaming")
    )

    real_bus_ids = sorted(grp["bus_id"].unique().to_list())
    n_buses = len(real_bus_ids)
    n_cols = 2
    n_rows = (n_buses + n_cols - 1) // n_cols
    subplot_titles = [bus_names.get(bid, str(bid)) for bid in real_bus_ids]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.18,
        horizontal_spacing=0.10,
    )

    stages = sorted(grp["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    for idx, bus_id in enumerate(real_bus_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = BUS_COLORS[idx % len(BUS_COLORS)]

        pcts = (
            grp.filter(pl.col("bus_id") == bus_id)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("w_spot").mean())
            .group_by("stage_id")
            .agg(
                pl.col("w_spot").quantile(0.1, interpolation="linear").alias("p10"),
                pl.col("w_spot").quantile(0.5, interpolation="linear").alias("p50"),
                pl.col("w_spot").quantile(0.9, interpolation="linear").alias("p90"),
            )
            .sort("stage_id")
        )
        pcts_map: dict[int, dict[str, float]] = {
            row_["stage_id"]: {
                "p10": row_["p10"],
                "p50": row_["p50"],
                "p90": row_["p90"],
            }
            for row_ in pcts.iter_rows(named=True)
        }
        p10 = [pcts_map.get(s, {}).get("p10", 0.0) for s in stages]
        p50 = [pcts_map.get(s, {}).get("p50", 0.0) for s in stages]
        p90 = [pcts_map.get(s, {}).get("p90", 0.0) for s in stages]

        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)"
                if color.startswith("#")
                else color,
                line={"color": "rgba(0,0,0,0)"},
                name=f"P10\u2013P90 {bus_names.get(bus_id, str(bus_id))}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p50,
                name=f"P50 {bus_names.get(bus_id, str(bus_id))}",
                line={"color": color, "width": 2},
                showlegend=True,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p10,
                name=f"P10 {bus_names.get(bus_id, str(bus_id))}",
                line={"color": color, "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p90,
                name=f"P90 {bus_names.get(bus_id, str(bus_id))}",
                line={"color": color, "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Weighted-Average Spot Price by Bus (block-hours weighted, p10/p50/p90)",
        height=350 * n_rows + 60,
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    for ax in fig.layout:
        if ax.startswith("yaxis"):
            fig.layout[ax].title = "R$/MWh"  # type: ignore[index]
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Costs tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Costs tab content area."""
    from cobre_bridge.dashboard.tabs.overview import chart_cost_by_stage

    return (
        section_title("Cost Composition by Stage")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_cost_by_stage(data.costs, data.stage_labels))
        + "</div>"
        + section_title("Spot Price by Bus (weighted avg across blocks)")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_spot_price_by_bus_subplots(
                data.buses_lf,
                data.bus_names,
                data.stage_labels,
                data.stage_hours,
                data.block_hours,
            )
        )
        + "</div>"
    )
