"""Training Insights tab module for the Cobre dashboard.

Displays convergence gap evolution, cut management charts, and LP solver heatmaps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cobre_bridge.ui.html import section_title, wrap_chart
from cobre_bridge.ui.plotly_helpers import LEGEND_DEFAULTS, MARGIN_DEFAULTS, fig_to_html
from cobre_bridge.ui.theme import COLORS

from .overview import chart_convergence

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-training"
TAB_LABEL = "Training Insights"
TAB_ORDER = 10

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_gap_evolution(conv: pd.DataFrame) -> str:
    """Line chart of gap_percent by iteration with a zero reference line."""
    if conv.empty:
        return "<p>No convergence data available.</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["gap_percent"],
            name="Gap %",
            line={"color": "#DC4C4C", "width": 2},
            mode="lines+markers",
            marker={"size": 5},
        )
    )
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#8B9298",
        annotation_text="0%",
        annotation_position="right",
    )
    fig.update_layout(
        title="Convergence Gap (%) per Iteration",
        xaxis_title="Iteration",
        yaxis_title="Gap (%)",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        height=420,
    )
    return fig_to_html(fig)


def chart_cut_state_evolution(conv: pd.DataFrame) -> str:
    """Stacked area for cuts_active + bars for cuts_added per iteration."""
    if conv.empty:
        return "<p>No convergence data available.</p>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["cuts_active"],
            name="Cuts Active",
            fill="tozeroy",
            fillcolor="rgba(74,144,184,0.25)",
            line={"color": COLORS["hydro"], "width": 2},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=conv["iteration"],
            y=conv["cuts_added"],
            name="Cuts Added",
            marker_color="rgba(245,166,35,0.7)",
            opacity=0.8,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Cut Pool Evolution",
        xaxis_title="Iteration",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        height=420,
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Cuts Active", secondary_y=False)
    fig.update_yaxes(title_text="Cuts Added (per iter)", secondary_y=True)
    return fig_to_html(fig)


def chart_cut_activity_heatmap(
    cut_selection: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=iteration, y=stage, z=cuts_active_after. YlOrRd colorscale."""
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    cs = cut_selection[cut_selection["stage"] > 0]
    pivot = cs.pivot_table(
        index="stage", columns="iteration", values="cuts_active_after", aggfunc="sum"
    )
    stages = sorted(pivot.index.tolist())
    iters = sorted(pivot.columns.tolist())
    z = [
        [
            float(pivot.loc[s, it]) if s in pivot.index and it in pivot.columns else 0.0
            for it in iters
        ]
        for s in stages
    ]
    ytick_vals = [i for i, s in enumerate(stages) if s % 12 == 0]
    ytick_text = [stage_labels.get(stages[i], str(stages[i])) for i in ytick_vals]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(it) for it in iters],
            y=[stage_labels.get(s, str(s)) for s in stages],
            colorscale="YlOrRd",
            colorbar={"title": "Cuts Active"},
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Cuts Active: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Active Cuts per Stage after Cut Selection",
        xaxis_title="SDDP Iteration",
        yaxis_title="Stage",
        yaxis={
            "autorange": "reversed",
            "dtick": 12,
            "tickvals": ytick_vals,
            "ticktext": ytick_text,
        },
        height=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_cut_deactivation_heatmap(
    cut_selection: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=iteration, y=stage, z=cuts_deactivated. Blues colorscale."""
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    pivot = cut_selection.pivot_table(
        index="stage", columns="iteration", values="cuts_deactivated", aggfunc="sum"
    )
    stages = sorted(pivot.index.tolist())
    iters = sorted(pivot.columns.tolist())
    z = [
        [
            float(pivot.loc[s, it]) if s in pivot.index and it in pivot.columns else 0.0
            for it in iters
        ]
        for s in stages
    ]
    ytick_vals = [i for i, s in enumerate(stages) if s % 12 == 0]
    ytick_text = [stage_labels.get(stages[i], str(stages[i])) for i in ytick_vals]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(it) for it in iters],
            y=[stage_labels.get(s, str(s)) for s in stages],
            colorscale="Blues",
            colorbar={"title": "Cuts Deactivated"},
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Cuts Deactivated: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Cuts Deactivated per Stage (Cut Selection)",
        xaxis_title="SDDP Iteration",
        yaxis_title="Stage",
        yaxis={
            "autorange": "reversed",
            "dtick": 12,
            "tickvals": ytick_vals,
            "ticktext": ytick_text,
        },
        height=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_simplex_heatmap(
    solver_train: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=SDDP iteration, y=stage, z=simplex_iterations (backward phase)."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[
        (solver_train["phase"] == "backward") & (solver_train["stage"] >= 0)
    ]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    pivot = bwd.pivot_table(
        index="stage", columns="iteration", values="simplex_iterations", aggfunc="sum"
    )
    stages = sorted(pivot.index.tolist())
    iters = sorted(pivot.columns.tolist())
    z = [
        [
            float(pivot.loc[s, it]) if s in pivot.index and it in pivot.columns else 0.0
            for it in iters
        ]
        for s in stages
    ]
    ytick_vals = [i for i, s in enumerate(stages) if s % 12 == 0]
    ytick_text = [stage_labels.get(stages[i], str(stages[i])) for i in ytick_vals]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(it) for it in iters],
            y=[stage_labels.get(s, str(s)) for s in stages],
            colorscale="Viridis",
            colorbar={"title": "Simplex Iters"},
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Simplex Iterations: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Simplex Iterations per Stage (Backward Pass)",
        xaxis_title="SDDP Iteration",
        yaxis_title="Stage",
        yaxis={
            "autorange": "reversed",
            "dtick": 12,
            "tickvals": ytick_vals,
            "ticktext": ytick_text,
        },
        height=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_solve_time_heatmap(
    solver_train: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=SDDP iteration, y=stage, z=solve_time_ms (backward phase). Hot reversed."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[
        (solver_train["phase"] == "backward") & (solver_train["stage"] >= 0)
    ]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    pivot = bwd.pivot_table(
        index="stage", columns="iteration", values="solve_time_ms", aggfunc="sum"
    )
    stages = sorted(pivot.index.tolist())
    iters = sorted(pivot.columns.tolist())
    z = [
        [
            float(pivot.loc[s, it]) if s in pivot.index and it in pivot.columns else 0.0
            for it in iters
        ]
        for s in stages
    ]
    ytick_vals = [i for i, s in enumerate(stages) if s % 12 == 0]
    ytick_text = [stage_labels.get(stages[i], str(stages[i])) for i in ytick_vals]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(it) for it in iters],
            y=[stage_labels.get(s, str(s)) for s in stages],
            colorscale="Hot",
            reversescale=True,
            colorbar={"title": "ms"},
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Solve Time: %{z:.1f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        title="LP Solve Time per Stage (Backward Pass, ms)",
        xaxis_title="SDDP Iteration",
        yaxis_title="Stage",
        yaxis={
            "autorange": "reversed",
            "dtick": 12,
            "tickvals": ytick_vals,
            "ticktext": ytick_text,
        },
        height=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Training tab always renders (convergence.parquet is always present)."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Training Insights tab content area."""
    return (
        section_title("Convergence")
        + '<div class="chart-grid">'
        + wrap_chart(chart_convergence(data.conv))
        + wrap_chart(chart_gap_evolution(data.conv))
        + "</div>"
        + section_title("Cut Management")
        + '<div class="chart-grid">'
        + wrap_chart(chart_cut_state_evolution(data.conv))
        + wrap_chart(chart_cut_activity_heatmap(data.cut_selection, data.stage_labels))
        + "</div>"
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_cut_deactivation_heatmap(data.cut_selection, data.stage_labels)
        )
        + "</div>"
        + section_title("LP Solver Heatmaps")
        + '<div class="chart-grid">'
        + wrap_chart(chart_simplex_heatmap(data.solver_train, data.stage_labels))
        + wrap_chart(chart_solve_time_heatmap(data.solver_train, data.stage_labels))
        + "</div>"
    )
