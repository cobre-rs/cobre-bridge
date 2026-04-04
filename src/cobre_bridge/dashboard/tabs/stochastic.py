"""Stochastic Model tab module for the Cobre dashboard.

Displays AR model fitting, seasonal inflow statistics, AR coefficients,
and noise generation charts. Only rendered when stochastic output is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from cobre_bridge.ui.html import section_title, wrap_chart
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    fig_to_html,
)

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-stochastic"
TAB_LABEL = "Stochastic Model"
TAB_ORDER = 110

# ---------------------------------------------------------------------------
# Module-level constants (copied verbatim from dashboard.py)
# ---------------------------------------------------------------------------

_MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_ar_order_distribution(fitting_report: dict) -> str:
    """Bar chart of selected AR order distribution across hydros."""
    if not fitting_report:
        return "<p>No data.</p>"
    hydros_data = fitting_report.get("hydros", {})
    if not hydros_data:
        return "<p>No data.</p>"
    order_counts: dict[int, int] = {}
    for hydro_info in hydros_data.values():
        order = int(hydro_info.get("selected_order", 0))
        order_counts[order] = order_counts.get(order, 0) + 1
    max_order = max(order_counts) if order_counts else 0
    orders = list(range(max_order + 1))
    counts = [order_counts.get(o, 0) for o in orders]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=orders,
            y=counts,
            marker_color="#4A90B8",
            hovertemplate="Order %{x}: %{y} hydros<extra></extra>",
        )
    )
    fig.update_layout(
        title="AR Order Distribution",
        xaxis_title="Selected AR Order",
        yaxis_title="Number of Hydros",
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig, unified_hover=False)


def chart_seasonal_stats_heatmap(
    inflow_stats: pd.DataFrame,
    stage_labels: dict[int, str],
    hydro_meta: dict[int, dict],
) -> str:
    """Heatmap of coefficient of variation (std/mean) by stage and hydro."""
    if inflow_stats.empty:
        return "<p>No data.</p>"
    df = inflow_stats.copy()
    # Restrict to first 58 stages
    df = df[df["stage_id"] < 58]
    # Avoid division by zero
    df = df[df["mean_m3s"].abs() > 0].copy()
    df["cv"] = df["std_m3s"] / df["mean_m3s"].abs()
    # Compute average CV per hydro, pick top 30
    avg_cv = df.groupby("hydro_id")["cv"].mean()
    top_hydros = avg_cv.nlargest(30).index.tolist()
    df = df[df["hydro_id"].isin(top_hydros)]
    # Pivot: rows=hydro_id, cols=stage_id
    pivot = df.pivot_table(
        index="hydro_id", columns="stage_id", values="cv", aggfunc="mean"
    )
    # Sort rows by descending average CV
    row_order = avg_cv.loc[pivot.index].sort_values(ascending=False).index.tolist()
    pivot = pivot.loc[row_order]
    stage_ids = sorted(pivot.columns.tolist())
    x_labels = [stage_labels.get(int(s), str(s)) for s in stage_ids]
    y_labels = [
        hydro_meta[int(h)]["name"] if int(h) in hydro_meta else str(h)
        for h in pivot.index
    ]
    fig = go.Figure(
        go.Heatmap(
            z=pivot[stage_ids].values.tolist(),
            x=x_labels,
            y=y_labels,
            colorscale="YlOrRd",
            hovertemplate="Stage: %{x}<br>Hydro: %{y}<br>CV: %{z:.3f}<extra></extra>",
            colorbar={"title": "CV"},
        )
    )
    fig.update_layout(
        title="Seasonal Inflow CV (std/mean) — Top 30 Hydros",
        xaxis_title="Stage",
        yaxis_title="Hydro",
        margin=_MARGIN,
        height=600,
    )
    return fig_to_html(fig, unified_hover=False)


def chart_ar_coefficients_heatmap(
    ar_coefficients: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Heatmap of lag-1 AR coefficient by stage and hydro — top 40 by max |coef|."""
    if ar_coefficients.empty:
        return "<p>No data.</p>"
    df = ar_coefficients[ar_coefficients["lag"] == 1].copy()
    df = df[df["stage_id"] < 58]
    if df.empty:
        return "<p>No lag-1 AR coefficient data.</p>"
    # Pick top 40 hydros by max absolute coefficient
    max_abs = df.groupby("hydro_id")["coefficient"].apply(lambda s: s.abs().max())
    top_hydros = max_abs.nlargest(40).index.tolist()
    df = df[df["hydro_id"].isin(top_hydros)]
    pivot = df.pivot_table(
        index="hydro_id", columns="stage_id", values="coefficient", aggfunc="mean"
    )
    stage_ids = sorted(pivot.columns.tolist())
    x_labels = [stage_labels.get(int(s), str(s)) for s in stage_ids]
    y_labels = [str(h) for h in pivot.index]
    zvals = pivot[stage_ids].values.tolist()
    abs_max = float(max(abs(v) for row in zvals for v in row if v == v) or 1.0)
    fig = go.Figure(
        go.Heatmap(
            z=zvals,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            hovertemplate="Stage: %{x}<br>Hydro: %{y}<br>AR(1) coef: %{z:.4f}<extra></extra>",
            colorbar={"title": "AR(1)"},
        )
    )
    fig.update_layout(
        title="AR(1) Coefficient by Stage and Hydro — Top 40",
        xaxis_title="Stage",
        yaxis_title="Hydro ID",
        margin=_MARGIN,
        height=550,
    )
    return fig_to_html(fig, unified_hover=False)


def chart_residual_ratio_by_stage(
    ar_coefficients: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Line chart of average residual_std_ratio by stage with p10/p90 band."""
    if ar_coefficients.empty:
        return "<p>No data.</p>"
    # One row per (hydro_id, stage_id) — take unique residual_std_ratio per pair
    df = ar_coefficients.drop_duplicates(subset=["hydro_id", "stage_id"]).copy()
    grp = df.groupby("stage_id")["residual_std_ratio"]
    stats = grp.agg(
        mean_ratio="mean",
        p10=lambda s: s.quantile(0.1),
        p90=lambda s: s.quantile(0.9),
    ).reset_index()
    stats = stats.sort_values("stage_id")
    x_labels = [stage_labels.get(int(s), str(s)) for s in stats["stage_id"]]
    fig = go.Figure()
    # Shaded p10/p90 band
    fig.add_trace(
        go.Scatter(
            x=x_labels + x_labels[::-1],
            y=stats["p90"].tolist() + stats["p10"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(74,144,184,0.18)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10–P90",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=stats["mean_ratio"],
            name="Mean",
            line={"color": "#4A90B8", "width": 2},
            hovertemplate="Stage: %{x}<br>Mean residual ratio: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Residual Std Ratio by Stage (unexplained variance after AR fitting)",
        xaxis_title="Stage",
        yaxis_title="Residual Std Ratio",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_noise_distribution(noise_openings: pd.DataFrame) -> str:
    """Histogram of all noise values with N(0,1) reference curve overlay."""
    if noise_openings.empty:
        return "<p>No data.</p>"
    vals = noise_openings["value"].dropna().values
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            histnorm="probability density",
            nbinsx=60,
            name="Noise samples",
            marker_color="#4A90B8",
            opacity=0.7,
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        )
    )
    # Standard normal reference curve
    x_ref = np.linspace(float(vals.min()), float(vals.max()), 300)
    y_ref = np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi)
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            name="N(0,1) reference",
            line={"color": "#DC4C4C", "width": 2, "dash": "dash"},
            hovertemplate="x: %{x:.3f}<br>N(0,1): %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Noise Sample Distribution vs N(0,1)",
        xaxis_title="Value",
        yaxis_title="Density",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig, unified_hover=False)


def chart_noise_correlation_sample(noise_openings: pd.DataFrame) -> str:
    """Correlation matrix across entities from 20 openings of stage 0."""
    if noise_openings.empty:
        return "<p>No data.</p>"
    stage0 = noise_openings[
        noise_openings["stage_id"] == noise_openings["stage_id"].min()
    ]
    if stage0.empty:
        return "<p>No data for stage 0.</p>"
    # Pivot: rows=opening_index, cols=entity_index
    pivot = stage0.pivot_table(
        index="opening_index", columns="entity_index", values="value", aggfunc="mean"
    )
    if pivot.shape[0] < 2:
        return "<p>Insufficient openings to compute correlation.</p>"
    corr = pivot.corr()
    abs_max = (
        float(corr.values[~np.eye(len(corr), dtype=bool)].max())
        if len(corr) > 1
        else 1.0
    )
    abs_max = max(abs_max, 0.01)
    entity_ids = [str(e) for e in corr.columns.tolist()]
    fig = go.Figure(
        go.Heatmap(
            z=corr.values.tolist(),
            x=entity_ids,
            y=entity_ids,
            colorscale="RdBu",
            zmid=0,
            zmin=-abs_max,
            zmax=abs_max,
            hovertemplate="Entity %{x} vs %{y}: %{z:.4f}<extra></extra>",
            colorbar={"title": "Corr"},
        )
    )
    min_stage = int(noise_openings["stage_id"].min())
    fig.update_layout(
        title=f"Noise Spatial Correlation Matrix — Stage {min_stage}",
        xaxis_title="Entity Index",
        yaxis_title="Entity Index",
        margin=_MARGIN,
        height=550,
    )
    return fig_to_html(fig, unified_hover=False)


def chart_order_reduction_reasons(fitting_report: dict) -> str:
    """Stacked bar of AR order reduction reasons by season (month)."""
    if not fitting_report:
        return "<p>No data.</p>"
    hydros_data = fitting_report.get("hydros", {})
    if not hydros_data:
        return "<p>No data.</p>"
    # Aggregate reason counts per season across all hydros
    reason_season: dict[str, list[int]] = {}
    for hydro_info in hydros_data.values():
        contrib = hydro_info.get("contribution_reductions", [])
        # contrib is a list of 12 elements (one per season), each a list of reason strings
        for season_id, reasons in enumerate(contrib):
            if season_id >= 12:
                break
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                if reason not in reason_season:
                    reason_season[reason] = [0] * 12
                reason_season[reason][season_id] += 1
    if not reason_season:
        return "<p>No reduction reason data.</p>"
    reason_colors = [
        "#4A90B8",
        "#F5A623",
        "#DC4C4C",
        "#4A8B6F",
        "#B87333",
        "#8B5E3C",
        "#607D8B",
        "#8B9298",
    ]
    fig = go.Figure()
    for i, (reason, counts) in enumerate(sorted(reason_season.items())):
        fig.add_trace(
            go.Bar(
                x=_MONTH_NAMES,
                y=counts,
                name=reason,
                marker_color=reason_colors[i % len(reason_colors)],
                hovertemplate=f"{reason}<br>%{{x}}: %{{y}} reductions<extra></extra>",
            )
        )
    fig.update_layout(
        title="AR Order Reduction Reasons by Season",
        xaxis_title="Season (Month)",
        yaxis_title="Count of Reductions",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True when stochastic model output is available."""
    return data.stochastic_available


def render(data: DashboardData) -> str:
    """Return full HTML for the Stochastic Model tab."""
    return (
        section_title("AR Model Fitting")
        + '<div class="chart-grid">'
        + wrap_chart(chart_ar_order_distribution(data.fitting_report))
        + wrap_chart(chart_order_reduction_reasons(data.fitting_report))
        + "</div>"
        + section_title("Seasonal Inflow Statistics")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_seasonal_stats_heatmap(
                data.inflow_stats_stoch, data.stage_labels, data.hydro_meta
            )
        )
        + wrap_chart(
            chart_residual_ratio_by_stage(data.ar_coefficients, data.stage_labels)
        )
        + "</div>"
        + section_title("AR Coefficients")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_ar_coefficients_heatmap(data.ar_coefficients, data.stage_labels)
        )
        + "</div>"
        + section_title("Noise Generation")
        + '<div class="chart-grid">'
        + wrap_chart(chart_noise_distribution(data.noise_openings))
        + wrap_chart(chart_noise_correlation_sample(data.noise_openings))
        + "</div>"
    )
