"""v2 Training tab module for the Cobre dashboard.

Displays convergence metrics, bounds evolution, cut pool dynamics,
cut management heatmaps, and iteration timing breakdown.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.chart_helpers import make_chart_card
from cobre_bridge.ui.html import (
    chart_grid,
    collapsible_section,
    metric_card,
    metrics_grid,
    section_title,
)
from cobre_bridge.ui.plotly_helpers import LEGEND_DEFAULTS, MARGIN_DEFAULTS
from cobre_bridge.ui.theme import COLORS, COPPER_ACCENT, PERFORMANCE_PHASE_COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

TAB_ID = "tab-training"
TAB_LABEL = "Training"
TAB_ORDER = 20

_EXTRA_COLORS: list[str] = [
    "#8B5CF6",
    "#F59E0B",
    "#10B981",
    "#EC4899",
    "#06B6D4",
    "#F97316",
]


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 5-card metrics row from convergence and manifest data.

    Cards (in order): Iterations, Training Time, Final Lower Bound,
    Total Cuts, Active Cuts.
    """
    conv = data.conv
    last = conv.iloc[-1]

    total_iters = len(conv)

    elapsed = data.training_metadata.get("duration_seconds")
    if elapsed is not None:
        hours = int(elapsed) // 3600
        mins = (int(elapsed) % 3600) // 60
        time_str = f"{hours}h {mins}min"
    else:
        time_str = "N/A"

    lb_val = float(last["lower_bound"])
    total_cuts = int(conv["cuts_added"].sum())
    active_cuts = int(last["cuts_active"])

    cards = [
        metric_card(
            value=str(total_iters),
            label="Iterations",
            color=COPPER_ACCENT,
        ),
        metric_card(
            value=time_str,
            label="Training Time",
            color=COPPER_ACCENT,
        ),
        metric_card(
            value=f"{lb_val:,.2f}",
            label="Final Lower Bound",
            color=COLORS["lower_bound"],
        ),
        metric_card(
            value=f"{total_cuts:,}",
            label="Total Cuts",
            color=COLORS["hydro"],
        ),
        metric_card(
            value=f"{active_cuts:,}",
            label="Active Cuts",
            color=COLORS["hydro"],
        ),
    ]
    return metrics_grid(cards)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour string to an ``rgba(...)`` CSS value."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _chart_convergence_hero(conv: pd.DataFrame) -> go.Figure:
    """Full convergence chart: lower bound, upper bound mean, +/- std band, zoom.

    When *conv* contains a ``gap_percent`` column with at least one non-NaN
    value, a dashed copper line is added on the secondary (right) y-axis.
    """
    n = len(conv)
    iters = conv["iteration"].tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Upper bound std band (drawn first so it appears under lines)
    ub_upper = conv["upper_bound_mean"] + conv["upper_bound_std"]
    ub_lower = conv["upper_bound_mean"] - conv["upper_bound_std"]

    fig.add_trace(
        go.Scatter(
            x=iters,
            y=ub_lower.tolist(),
            name="UB -std",
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=ub_upper.tolist(),
            name="UB \u00b1std Band",
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=_hex_to_rgba(COLORS["upper_bound"], 0.15),
            showlegend=True,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )

    # Upper bound mean
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=conv["upper_bound_mean"].tolist(),
            name="Upper Bound",
            mode="lines",
            line={"color": COLORS["upper_bound"], "width": 2},
        ),
        secondary_y=False,
    )

    # Lower bound
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=conv["lower_bound"].tolist(),
            name="Lower Bound",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 2},
        ),
        secondary_y=False,
    )

    # Gap % on secondary axis (only when column present and has real values)
    has_gap = "gap_percent" in conv.columns and conv["gap_percent"].notna().any()
    if has_gap:
        fig.add_trace(
            go.Scatter(
                x=iters,
                y=conv["gap_percent"].tolist(),
                name="Gap %",
                mode="lines",
                line={"color": COPPER_ACCENT, "width": 1.5, "dash": "dash"},
            ),
            secondary_y=True,
        )

    # Zoom dropdown buttons
    buttons = [
        {"label": "All", "method": "relayout", "args": [{"xaxis.range": [0, n]}]},
        {
            "label": "Last 200",
            "method": "relayout",
            "args": [{"xaxis.range": [max(0, n - 200), n]}],
        },
        {
            "label": "Last 100",
            "method": "relayout",
            "args": [{"xaxis.range": [max(0, n - 100), n]}],
        },
        {
            "label": "Last 50",
            "method": "relayout",
            "args": [{"xaxis.range": [max(0, n - 50), n]}],
        },
    ]

    fig.update_layout(
        xaxis_title="Iteration",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        updatemenus=[
            {
                "type": "dropdown",
                "buttons": buttons,
                "x": 1.0,
                "y": 1.15,
                "xanchor": "right",
                "yanchor": "top",
                "showactive": True,
            }
        ],
    )
    fig.update_yaxes(title_text="Objective Value", secondary_y=False)
    fig.update_yaxes(title_text="Gap (%)", secondary_y=True)
    return fig


def _chart_lb_delta(conv: pd.DataFrame) -> go.Figure | None:
    """Dual-axis chart: LB delta % (left) + Gap % (right) per iteration.

    LB delta % = (LB[i] - LB[i-1]) / (LB[i-1] + eps) * 100
    """
    if conv.empty or len(conv) < 2:
        return None

    lb = conv["lower_bound"]
    eps = 1e-12
    delta_pct = ((lb.diff() / (lb.shift(1) + eps)) * 100.0).iloc[1:]
    iters = conv["iteration"].iloc[1:].tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=delta_pct.tolist(),
            name="LB Delta %",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 1.5},
        ),
        secondary_y=False,
    )

    has_gap = "gap_percent" in conv.columns and conv["gap_percent"].notna().any()
    if has_gap:
        fig.add_trace(
            go.Scatter(
                x=conv["iteration"].iloc[1:].tolist(),
                y=conv["gap_percent"].iloc[1:].tolist(),
                name="Gap %",
                mode="lines",
                line={"color": COPPER_ACCENT, "width": 1.5, "dash": "dash"},
            ),
            secondary_y=True,
        )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#8B9298",
        annotation_text="0",
        annotation_position="right",
    )
    fig.update_layout(
        xaxis_title="Iteration",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    fig.update_yaxes(title_text="LB Improvement (%)", secondary_y=False)
    fig.update_yaxes(title_text="Gap (%)", secondary_y=True)
    return fig


def _chart_gap_evolution(conv: pd.DataFrame) -> go.Figure | None:
    """Line chart of gap_percent by iteration with a zero reference line."""
    if conv.empty:
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"].tolist(),
            y=conv["gap_percent"].tolist(),
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
        xaxis_title="Iteration",
        yaxis_title="Gap (%)",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    return fig


def _chart_cut_pool(conv: pd.DataFrame) -> go.Figure:
    """Dual-axis chart: cuts_active area + cuts_added bars."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=conv["iteration"].tolist(),
            y=conv["cuts_active"].tolist(),
            name="Cuts Active",
            fill="tozeroy",
            fillcolor="rgba(74,144,184,0.25)",
            line={"color": COLORS["hydro"], "width": 2},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=conv["iteration"].tolist(),
            y=conv["cuts_added"].tolist(),
            name="Cuts Added",
            marker_color="rgba(245,166,35,0.7)",
            opacity=0.8,
        ),
        secondary_y=True,
    )

    fig.update_layout(
        xaxis_title="Iteration",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Cuts Active", secondary_y=False)
    fig.update_yaxes(title_text="Cuts Added (per iter)", secondary_y=True)
    return fig


def _cut_pool_summary(conv: pd.DataFrame) -> str:
    """Build a simple HTML summary of final cuts_active and total cuts_added."""
    final_active = int(conv.iloc[-1]["cuts_active"])
    total_added = int(conv["cuts_added"].sum())
    return (
        f'<div class="metric-card" style="border-top: 4px solid {COLORS["hydro"]};">'
        f'<div class="metric-value">{final_active:,}</div>'
        f'<div class="metric-label">Final Cuts Active</div>'
        f"</div>"
        f'<div class="metric-card" style="border-top: 4px solid {COPPER_ACCENT};">'
        f'<div class="metric-value">{total_added:,}</div>'
        f'<div class="metric-label">Total Cuts Added</div>'
        f"</div>"
    )


def _sample_pivot_columns(pivot: pd.DataFrame, step: int) -> pd.DataFrame:
    """Sample every *step*-th column of a pivot table (for large iteration counts)."""
    cols = sorted(pivot.columns.tolist())
    sampled_cols = [c for i, c in enumerate(cols) if i % step == 0]
    return pivot[sampled_cols]


def _chart_cut_activity_heatmap(
    cut_selection: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=iteration, y=stage, z=cuts_active_after. YlOrRd colorscale."""
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    cs = cut_selection[cut_selection["stage"] > 0]
    pivot = cs.pivot_table(
        index="stage", columns="iteration", values="cuts_active_after", aggfunc="sum"
    )

    # Sample every 2nd iteration for large iteration counts
    if len(pivot.columns) > 200:
        pivot = _sample_pivot_columns(pivot, 2)

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
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Cuts Active: %{z}<extra></extra>",  # noqa: E501
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
        margin={"l": 80, "r": 30, "t": 60, "b": 50},
    )
    from cobre_bridge.ui.plotly_helpers import fig_to_html

    return fig_to_html(fig, unified_hover=False)


def _chart_cut_deactivation_heatmap(
    cut_selection: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=iteration, y=stage, z=cuts_deactivated. Blues colorscale."""
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    pivot = cut_selection.pivot_table(
        index="stage", columns="iteration", values="cuts_deactivated", aggfunc="sum"
    )

    # Sample every 2nd iteration for large iteration counts
    if len(pivot.columns) > 200:
        pivot = _sample_pivot_columns(pivot, 2)

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
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Cuts Deactivated: %{z}<extra></extra>",  # noqa: E501
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
        margin={"l": 80, "r": 30, "t": 60, "b": 50},
    )
    from cobre_bridge.ui.plotly_helpers import fig_to_html

    return fig_to_html(fig, unified_hover=False)


_TIMING_COMPONENT_ORDER: list[str] = [
    "forward_solve_ms",
    "forward_sample_ms",
    "backward_solve_ms",
    "backward_cut_ms",
    "cut_selection_ms",
    "mpi_allreduce_ms",
    "mpi_broadcast_ms",
    "state_exchange_ms",
    "cut_batch_build_ms",
    "rayon_overhead_ms",
    "overhead_ms",
    "io_write_ms",
]

_TIMING_PHASE_MAP: dict[str, str] = {
    "forward_solve_ms": PERFORMANCE_PHASE_COLORS["forward"],
    "forward_sample_ms": PERFORMANCE_PHASE_COLORS["forward"],
    "backward_solve_ms": PERFORMANCE_PHASE_COLORS["backward"],
    "backward_cut_ms": PERFORMANCE_PHASE_COLORS["backward"],
    "cut_selection_ms": PERFORMANCE_PHASE_COLORS["lp_solve"],
    "overhead_ms": PERFORMANCE_PHASE_COLORS["overhead"],
    "rayon_overhead_ms": PERFORMANCE_PHASE_COLORS["overhead"],
}


def _timing_columns(timing: pd.DataFrame) -> list[str]:
    """Return ordered list of *_ms columns present in *timing*."""
    present_ordered = [c for c in _TIMING_COMPONENT_ORDER if c in timing.columns]
    extra = [
        c
        for c in timing.columns
        if c.endswith("_ms") and c not in _TIMING_COMPONENT_ORDER and c != "iteration"
    ]
    return present_ordered + extra


def _timing_color(col: str, idx: int) -> str:
    """Return a colour for a timing column, falling back to the extra palette."""
    if col in _TIMING_PHASE_MAP:
        return _TIMING_PHASE_MAP[col]
    return _EXTRA_COLORS[idx % len(_EXTRA_COLORS)]


def _chart_timing_stacked(timing: pd.DataFrame) -> go.Figure | None:
    """Stacked bar chart of timing components per iteration."""
    if timing.empty:
        return None

    cols = _timing_columns(timing)
    if not cols:
        return None

    iters = (
        timing["iteration"].tolist()
        if "iteration" in timing.columns
        else list(range(len(timing)))
    )

    fig = go.Figure()
    extra_idx = 0
    for col in cols:
        color = _timing_color(col, extra_idx)
        if col not in _TIMING_PHASE_MAP:
            extra_idx += 1
        label = col.replace("_ms", "").replace("_", " ").title()
        fig.add_trace(
            go.Bar(
                x=iters,
                y=timing[col].tolist(),
                name=label,
                marker_color=color,
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Iteration",
        yaxis_title="Time (ms)",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    return fig


def _chart_phase_distribution(timing: pd.DataFrame) -> go.Figure | None:
    """Single stacked horizontal bar showing percentage split of total time."""
    if timing.empty:
        return None

    cols = _timing_columns(timing)
    if not cols:
        return None

    totals: dict[str, float] = {c: float(timing[c].sum()) for c in cols}
    grand_total = sum(totals.values())
    if grand_total == 0.0:
        return None

    # Group into major phases: forward, backward, cut_selection, other
    phase_groups: dict[str, float] = {
        "Forward": totals.get("forward_solve_ms", 0.0)
        + totals.get("forward_sample_ms", 0.0),
        "Backward": totals.get("backward_solve_ms", 0.0)
        + totals.get("backward_cut_ms", 0.0),
        "Cut Selection": totals.get("cut_selection_ms", 0.0),
        "Other": sum(
            v
            for k, v in totals.items()
            if k
            not in {
                "forward_solve_ms",
                "forward_sample_ms",
                "backward_solve_ms",
                "backward_cut_ms",
                "cut_selection_ms",
            }
        ),
    }

    phase_colors = {
        "Forward": PERFORMANCE_PHASE_COLORS["forward"],
        "Backward": PERFORMANCE_PHASE_COLORS["backward"],
        "Cut Selection": PERFORMANCE_PHASE_COLORS["lp_solve"],
        "Other": PERFORMANCE_PHASE_COLORS["overhead"],
    }

    fig = go.Figure()
    for phase, total_ms in phase_groups.items():
        pct = total_ms / grand_total * 100.0
        fig.add_trace(
            go.Bar(
                x=[pct],
                y=["Time Split"],
                name=phase,
                orientation="h",
                marker_color=phase_colors[phase],
                text=[f"{pct:.1f}%"],
                textposition="inside",
                hovertemplate=f"{phase}: {total_ms / 1000:.1f}s ({pct:.1f}%)<extra></extra>",  # noqa: E501
            )
        )

    fig.update_layout(
        barmode="stack",
        xaxis_title="Percentage of Total Time (%)",
        xaxis={"range": [0, 100]},
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    return fig


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Training tab always renders (convergence.parquet is always present)."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Training tab content area."""
    conv = data.conv
    if conv.empty:
        return "<p>No convergence data.</p>"

    section_a = _build_metrics_row(data)

    hero_fig = _chart_convergence_hero(conv)
    hero_html = make_chart_card(
        hero_fig,
        title="Training Bounds",
        chart_id="v2-training-convergence-hero",
        height=500,
    )
    section_b = section_title("Training Bounds") + chart_grid([hero_html], single=True)

    # Row 2: LB Progress & Gap + Cut Pool side by side
    row2_parts: list[str] = []
    lb_delta_fig = _chart_lb_delta(conv)
    if lb_delta_fig is not None:
        row2_parts.append(
            make_chart_card(
                lb_delta_fig,
                title="Lower Bound Progress & Gap",
                chart_id="v2-training-lb-delta",
            )
        )
    cut_pool_fig = _chart_cut_pool(conv)
    row2_parts.append(
        make_chart_card(
            cut_pool_fig,
            title="Cut Pool Evolution",
            chart_id="v2-training-cut-pool",
        )
    )
    section_c = chart_grid(row2_parts)

    if data.cut_selection.empty:
        heatmap_content = "<p>No cut selection data available.</p>"
    else:
        activity_html = _chart_cut_activity_heatmap(
            data.cut_selection, data.stage_labels
        )
        deactivation_html = _chart_cut_deactivation_heatmap(
            data.cut_selection, data.stage_labels
        )
        heatmap_content = (
            f'<div class="chart-grid">'
            f'<div class="chart-card">{activity_html}</div>'
            f'<div class="chart-card">{deactivation_html}</div>'
            f"</div>"
        )
    section_e = collapsible_section(
        title="Cut Management Heatmaps",
        content=heatmap_content,
        default_collapsed=False,
    )

    if data.timing.empty:
        timing_content = "<p>No timing data available.</p>"
    else:
        timing_stacked = _chart_timing_stacked(data.timing)
        timing_dist = _chart_phase_distribution(data.timing)
        timing_parts: list[str] = []
        if timing_stacked is not None:
            timing_parts.append(
                make_chart_card(
                    timing_stacked,
                    title="Iteration Timing Breakdown (ms)",
                    chart_id="v2-training-timing-stacked",
                )
            )
        if timing_dist is not None:
            timing_parts.append(
                make_chart_card(
                    timing_dist,
                    title="Phase Time Distribution (%)",
                    chart_id="v2-training-phase-dist",
                )
            )
        timing_content = (
            chart_grid(timing_parts)
            if timing_parts
            else "<p>No timing data available.</p>"
        )
    section_f = collapsible_section(
        title="Iteration Timing",
        content=timing_content,
        default_collapsed=False,
    )

    return section_a + section_b + section_c + section_e + section_f
