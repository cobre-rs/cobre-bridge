"""Overview tab module for the Cobre dashboard.

Displays key metrics, training convergence, and cost breakdown charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from cobre_bridge.ui.html import metric_card, metrics_grid, section_title, wrap_chart
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    fig_to_html,
    stage_x_labels,
)
from cobre_bridge.ui.theme import COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-overview"
TAB_LABEL = "Overview"
TAB_ORDER = 0

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_convergence(conv: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["lower_bound"],
            name="Lower Bound",
            line={"color": COLORS["lower_bound"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["upper_bound_mean"],
            name="Upper Bound (mean)",
            line={"color": COLORS["upper_bound"], "width": 2},
        )
    )
    ub_upper = conv["upper_bound_mean"] + conv["upper_bound_std"]
    ub_lower = conv["upper_bound_mean"] - conv["upper_bound_std"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([conv["iteration"], conv["iteration"].iloc[::-1]]),
            y=pd.concat([ub_upper, ub_lower.iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(229,57,53,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="Upper Bound ± std",
            showlegend=True,
        )
    )
    last = conv.iloc[-1]
    fig.update_layout(
        title=f"Training Convergence (gap={last['gap_percent']:.2f}%, {int(last['iteration'])} iterations)",
        xaxis_title="Iteration",
        yaxis_title="Cost",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        height=420,
    )
    return fig_to_html(fig)


def chart_cost_breakdown(costs: pd.DataFrame) -> str:
    cost_cols = [
        "thermal_cost",
        "deficit_cost",
        "spillage_cost",
        "curtailment_cost",
        "exchange_cost",
        "excess_cost",
        "inflow_penalty_cost",
    ]
    values = []
    labels = []
    for col in cost_cols:
        if col in costs.columns:
            avg = costs.groupby("scenario_id")[col].sum().mean()
            if abs(avg) > 1.0:
                labels.append(col.replace("_cost", "").replace("_", " ").title())
                values.append(avg)
    color_map = {
        "Thermal": COLORS["thermal"],
        "Deficit": COLORS["deficit"],
        "Spillage": COLORS["spillage"],
        "Curtailment": COLORS["curtailment"],
        "Exchange": COLORS["exchange"],
        "Excess": "#FF5722",
        "Inflow Penalty": "#607D8B",
    }
    fig = go.Figure()
    for label, value in zip(labels, values):
        fig.add_trace(
            go.Bar(
                x=["Total"],
                y=[value],
                name=label,
                marker_color=color_map.get(label, "#90A4AE"),
                text=[f"{value:.2e}"],
                textposition="inside",
                insidetextanchor="middle",
            )
        )
    fig.update_layout(
        title="Average Cost Breakdown (sum over all stages)",
        yaxis_title="Cost (R$)",
        barmode="stack",
        height=420,
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        showlegend=True,
    )
    return fig_to_html(fig)


def build_key_metrics_html(
    hydros_lf: pl.LazyFrame,
    thermals_lf: pl.LazyFrame,
    ncs_lf: pl.LazyFrame,
    buses_lf: pl.LazyFrame,
    costs: pd.DataFrame,
) -> str:
    total_hydro_gwh = (
        hydros_lf.group_by("scenario_id")
        .agg(pl.col("generation_mwh").sum())
        .select(pl.col("generation_mwh").mean())
        .collect(engine="streaming")["generation_mwh"][0]
        / 1e3
    )
    total_thermal_gwh = (
        thermals_lf.group_by("scenario_id")
        .agg(pl.col("generation_mwh").sum())
        .select(pl.col("generation_mwh").mean())
        .collect(engine="streaming")["generation_mwh"][0]
        / 1e3
    )
    total_ncs_gwh = (
        ncs_lf.group_by("scenario_id")
        .agg(pl.col("generation_mwh").sum())
        .select(pl.col("generation_mwh").mean())
        .collect(engine="streaming")["generation_mwh"][0]
        / 1e3
    )
    avg_spot = (
        buses_lf.filter(pl.col("bus_id").is_in([0, 1, 2, 3]))
        .select(pl.col("spot_price").mean())
        .collect(engine="streaming")["spot_price"][0]
    )
    total_spillage = (
        hydros_lf.group_by("scenario_id")
        .agg(pl.col("spillage_m3s").sum())
        .select(pl.col("spillage_m3s").mean())
        .collect(engine="streaming")["spillage_m3s"][0]
    )
    ncs_stats = (
        ncs_lf.group_by("scenario_id")
        .agg(
            pl.col("generation_mwh").sum().alias("gen"),
            pl.col("curtailment_mwh").sum().alias("curt"),
        )
        .select(pl.col("gen").mean(), pl.col("curt").mean())
        .collect(engine="streaming")
    )
    ncs_gen = ncs_stats["gen"][0]
    ncs_curt = ncs_stats["curt"][0]
    curt_rate = ncs_curt / max(ncs_gen + ncs_curt, 1) * 100

    metrics = [
        ("Total Hydro Generation", f"{total_hydro_gwh:,.0f} GWh", COLORS["hydro"]),
        (
            "Total Thermal Generation",
            f"{total_thermal_gwh:,.0f} GWh",
            COLORS["thermal"],
        ),
        ("Total NCS Generation", f"{total_ncs_gwh:,.0f} GWh", COLORS["ncs"]),
        ("Average Spot Price", f"R$ {avg_spot:.2f}/MWh", "#607D8B"),
        ("Total Spillage", f"{total_spillage:,.0f} m³/s", COLORS["spillage"]),
        ("NCS Curtailment Rate", f"{curt_rate:.1f}%", COLORS["curtailment"]),
    ]
    cards = [metric_card(value, label, color=color) for label, value, color in metrics]
    return metrics_grid(cards)


def chart_cost_by_stage(
    costs: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Stacked area of cost components by stage."""
    avg = costs.groupby("stage_id").mean(numeric_only=True)
    stages = sorted(avg.index)
    xlabels = stage_x_labels(stages, stage_labels)

    cost_series = {
        "Thermal": ("thermal_cost", COLORS["thermal"]),
        "Deficit": ("deficit_cost", COLORS["deficit"]),
        "Spillage": ("spillage_cost", COLORS["spillage"]),
        "Curtailment": ("curtailment_cost", COLORS["curtailment"]),
        "Exchange": ("exchange_cost", COLORS["exchange"]),
        "Excess": ("excess_cost", "#FF5722"),
        "Inflow Penalty": ("inflow_penalty_cost", "#607D8B"),
    }

    fig = go.Figure()
    for label, (col, color) in cost_series.items():
        if col in avg.columns:
            vals = [avg[col].get(s, 0) for s in stages]
            if sum(abs(v) for v in vals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=xlabels,
                        y=vals,
                        name=label,
                        stackgroup="costs",
                        line={"color": color},
                        fillcolor=color.replace(")", ",0.7)")
                        .replace("rgb", "rgba")
                        .replace("#", "rgba(")
                        .replace("rgba(", "rgba(")
                        if "#" not in color
                        else color,
                    )
                )
    fig.update_layout(
        title="Cost Composition by Stage (avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Cost (R$)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
        height=420,
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Overview tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Overview tab content area."""
    metrics_html = build_key_metrics_html(
        data.hydros_lf, data.thermals_lf, data.ncs_lf, data.buses_lf, data.costs
    )
    return (
        section_title("Key Metrics")
        + metrics_html
        + section_title("Training Convergence")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_convergence(data.conv))
        + "</div>"
        + section_title("Cost Breakdown & Composition")
        + '<div class="chart-grid">'
        + wrap_chart(chart_cost_breakdown(data.costs))
        + wrap_chart(chart_cost_by_stage(data.costs, data.stage_labels))
        + "</div>"
    )
