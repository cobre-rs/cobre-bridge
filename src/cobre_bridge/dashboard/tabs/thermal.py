"""Thermal Operations tab module for the Cobre dashboard.

Displays total thermal generation, thermal generation by bus, merit order,
generation by cost bracket, and cost vs generation scatter charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.graph_objects as go
import polars as pl

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
from cobre_bridge.ui.theme import BUS_COLORS, COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-thermal"
TAB_LABEL = "Thermal Operations"
TAB_ORDER = 50

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_thermal_generation_total(
    thermals_lf: pl.LazyFrame,
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Total thermal generation with p10/p50/p90 bands across all plants."""
    pcts = (
        thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "thermal_id"])
        .agg((pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["scenario_id", "stage_id"])
        .agg(pl.col("generation_mw").sum())
        .group_by("stage_id")
        .agg(
            pl.col("generation_mw").quantile(0.1, interpolation="linear").alias("p10"),
            pl.col("generation_mw").quantile(0.5, interpolation="linear").alias("p50"),
            pl.col("generation_mw").quantile(0.9, interpolation="linear").alias("p90"),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = pcts["stage_id"].to_list()
    xlabels = stage_x_labels(stages, stage_labels)
    p10 = pcts["p10"].to_list()
    p50 = pcts["p50"].to_list()
    p90 = pcts["p90"].to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels + xlabels[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(245,166,35,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10\u2013P90 range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p50,
            name="Median (P50)",
            line={"color": COLORS["thermal"], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p10,
            name="P10",
            line={"color": COLORS["thermal"], "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p90,
            name="P90",
            line={"color": COLORS["thermal"], "width": 1, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Total Thermal Generation (all plants, p10/p50/p90 across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_thermal_gen_by_bus(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Stacked area of thermal generation by bus (avg across scenarios)."""
    tbus_map = {k: v["bus_id"] for k, v in thermal_meta.items()}
    tbus_df = pl.DataFrame(
        {"thermal_id": list(tbus_map.keys()), "bus_id": list(tbus_map.values())}
    )
    gen_by_bus = (
        thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .join(tbus_df.lazy(), on="thermal_id")
        .group_by(["scenario_id", "stage_id", "bus_id", "thermal_id"])
        .agg((pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("generation_mw").sum())
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("generation_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    bus_ids = sorted(gen_by_bus["bus_id"].unique().to_list())
    stages = sorted(gen_by_bus["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        sub = gen_by_bus.filter(pl.col("bus_id") == bus_id)
        gm = dict(zip(sub["stage_id"].to_list(), sub["generation_mw"].to_list()))
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[gm.get(s, 0) for s in stages],
                name=bus_names.get(int(bus_id), str(bus_id)),
                stackgroup="buses",
                line={"color": BUS_COLORS[i % len(BUS_COLORS)]},
            )
        )
    fig.update_layout(
        title="Thermal Generation by Bus (block-hours weighted avg, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_thermal_merit_order(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    top_n: int = 30,
) -> str:
    """Horizontal bar: thermals sorted by cost, showing avg generation vs capacity."""
    avg_gen_df = (
        thermals_lf.group_by(["scenario_id", "thermal_id"])
        .agg(pl.col("generation_mwh").sum())
        .group_by("thermal_id")
        .agg(pl.col("generation_mwh").mean())
        .collect(engine="streaming")
    )
    avg_gen = dict(
        zip(avg_gen_df["thermal_id"].to_list(), avg_gen_df["generation_mwh"].to_list())
    )

    sorted_thermals = sorted(
        thermal_meta.items(),
        key=lambda x: x[1]["cost_per_mwh"],
    )[:top_n]

    names_list = []
    gen_vals = []
    cap_vals = []
    for tid, meta in sorted_thermals:
        names_list.append(meta["name"])
        gen_vals.append(avg_gen.get(tid, 0) / 1e3)
        cap_vals.append(meta["max_mw"] * 8760 / 1e3)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cap_vals,
            y=names_list,
            name="Installed Capacity (GWh/yr equivalent)",
            orientation="h",
            marker_color="rgba(255,152,0,0.3)",
            marker_line_color=COLORS["thermal"],
            marker_line_width=1,
        )
    )
    fig.add_trace(
        go.Bar(
            x=gen_vals,
            y=names_list,
            name="Avg Generation (GWh)",
            orientation="h",
            marker_color=COLORS["thermal"],
        )
    )
    fig.update_layout(
        title=f"Thermal Merit Order (top {top_n} by cost, sorted low\u2192high)",
        xaxis_title="GWh",
        barmode="overlay",
        legend=_LEGEND,
        height=max(440, len(names_list) * 22 + 100),
        margin=dict(l=120, r=30, t=90, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_thermal_by_cost_bracket(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Stacked area of thermal generation grouped by cost bracket."""
    bracket_colors = {
        "Zero cost": "#FFF9C4",
        "0\u2013100 R$/MWh": "#FFE082",
        "100\u2013500 R$/MWh": COLORS["thermal"],
        "500+ R$/MWh": "#E65100",
    }

    def assign_bracket(cost: float) -> str:
        if cost == 0:
            return "Zero cost"
        elif cost <= 100:
            return "0\u2013100 R$/MWh"
        elif cost <= 500:
            return "100\u2013500 R$/MWh"
        else:
            return "500+ R$/MWh"

    cost_map = {tid: meta["cost_per_mwh"] for tid, meta in thermal_meta.items()}
    bracket_map = {tid: assign_bracket(c) for tid, c in cost_map.items()}

    t0 = (
        thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "thermal_id"])
        .agg((pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["stage_id", "thermal_id"])
        .agg(pl.col("generation_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = sorted(t0["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for bracket_name, color in bracket_colors.items():
        tids = [tid for tid, b in bracket_map.items() if b == bracket_name]
        if not tids:
            continue
        sub = t0.filter(pl.col("thermal_id").is_in(tids))
        if sub.is_empty():
            continue
        gen_by_stage = (
            sub.group_by("stage_id").agg(pl.col("generation_mw").sum()).sort("stage_id")
        )
        gen_map = dict(
            zip(
                gen_by_stage["stage_id"].to_list(),
                gen_by_stage["generation_mw"].to_list(),
            )
        )
        vals = [gen_map.get(s, 0) for s in stages]
        if sum(vals) == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=vals,
                name=bracket_name,
                stackgroup="brackets",
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Thermal Generation by Cost Bracket (block-hours weighted avg, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_thermal_cost_vs_gen(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    bh_df: pl.DataFrame | None = None,
) -> str:
    """Scatter: avg generation vs cost_per_mwh, sized by max_mw, colored by bus."""
    if bh_df is not None:
        avg_gen_df = (
            thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
            .group_by(["scenario_id", "thermal_id"])
            .agg((pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum())
            .group_by("thermal_id")
            .agg(pl.col("generation_mw").mean())
            .collect(engine="streaming")
        )
    else:
        avg_gen_df = (
            thermals_lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "thermal_id"])
            .agg(pl.col("generation_mw").mean())
            .group_by("thermal_id")
            .agg(pl.col("generation_mw").mean())
            .collect(engine="streaming")
        )
    avg_gen = dict(
        zip(avg_gen_df["thermal_id"].to_list(), avg_gen_df["generation_mw"].to_list())
    )

    bus_to_plants: dict[int, list[dict]] = {}
    for tid, meta in thermal_meta.items():
        bus_id = meta["bus_id"]
        bus_to_plants.setdefault(bus_id, []).append(
            {
                "tid": tid,
                "name": meta["name"],
                "avg_gen": float(avg_gen.get(tid, 0)),
                "cost": meta["cost_per_mwh"],
                "max_mw": meta["max_mw"],
            }
        )

    fig = go.Figure()
    for i, (bus_id, plants) in enumerate(sorted(bus_to_plants.items())):
        color = BUS_COLORS[i % len(BUS_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=[p["avg_gen"] for p in plants],
                y=[p["cost"] for p in plants],
                mode="markers+text",
                marker={
                    "size": [max(6, min(40, p["max_mw"] / 20)) for p in plants],
                    "color": color,
                    "opacity": 0.75,
                    "line": {"color": "white", "width": 1},
                },
                text=[p["name"] for p in plants],
                textposition="top center",
                textfont={"size": 9},
                name=bus_names.get(bus_id, str(bus_id)),
                customdata=[[p["max_mw"], p["tid"]] for p in plants],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Avg gen: %{x:.1f} MW<br>"
                    "Cost: %{y:.1f} R$/MWh<br>"
                    "Capacity: %{customdata[0]:.1f} MW<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Thermal Plants: Avg Generation vs Cost (size = installed capacity)",
        xaxis_title="Avg Generation (MW)",
        yaxis_title="Cost (R$/MWh)",
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=90, b=60),
        height=480,
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Thermal Operations tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Thermal Operations tab content area."""
    return (
        section_title("Total Thermal Generation")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_thermal_generation_total(
                data.thermals_lf, data.stage_labels, data.bh_df
            )
        )
        + "</div>"
        + section_title("Thermal Generation by Bus")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_thermal_gen_by_bus(
                data.thermals_lf,
                data.thermal_meta,
                data.bus_names,
                data.stage_labels,
                data.bh_df,
            )
        )
        + "</div>"
    )
