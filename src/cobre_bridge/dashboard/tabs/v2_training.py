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

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-training"
TAB_LABEL = "Training"
TAB_ORDER = 10

# Fallback palette for timing components not in PERFORMANCE_PHASE_COLORS
_EXTRA_COLORS: list[str] = [
    "#8B5CF6",
    "#F59E0B",
    "#10B981",
    "#EC4899",
    "#06B6D4",
    "#F97316",
]

# ---------------------------------------------------------------------------
# Section A — Metrics Row
# ---------------------------------------------------------------------------


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 5-card metrics row from convergence and manifest data."""
    conv = data.conv
    last = conv.iloc[-1]

    lb_val = float(last["lower_bound"])
    ub_val = float(last["upper_bound_mean"])
    gap_val = float(last["gap_percent"])
    total_iters = len(conv)
    termination = data.training_manifest.get("termination_reason", "N/A")

    gap_color = "#DC4C4C" if gap_val > 1.0 else "#4A8B6F"

    cards = [
        metric_card(
            value=f"{lb_val:,.2f}",
            label="Final Lower Bound",
            color=COLORS["lower_bound"],
            sparkline_values=conv["lower_bound"].tolist(),
        ),
        metric_card(
            value=f"{ub_val:,.2f}",
            label="Final Upper Bound",
            color=COLORS["upper_bound"],
        ),
        metric_card(
            value=f"{gap_val:.2f}%",
            label="Final Gap",
            color=gap_color,
        ),
        metric_card(
            value=str(total_iters),
            label="Total Iterations",
            color=COPPER_ACCENT,
        ),
        metric_card(
            value=str(termination),
            label="Termination",
            color="#374151",
        ),
    ]
    return metrics_grid(cards)


# ---------------------------------------------------------------------------
# Section B — Convergence Hero Chart
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour string to an ``rgba(...)`` CSS value."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _chart_convergence_hero(conv: pd.DataFrame) -> go.Figure:
    """Full convergence chart: lower bound, upper bound mean, +/- std band, zoom dropdown."""  # noqa: E501
    n = len(conv)
    iters = conv["iteration"].tolist()

    fig = go.Figure()

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
        )
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
        )
    )

    # Upper bound mean
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=conv["upper_bound_mean"].tolist(),
            name="Upper Bound",
            mode="lines",
            line={"color": COLORS["upper_bound"], "width": 2},
        )
    )

    # Lower bound
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=conv["lower_bound"].tolist(),
            name="Lower Bound",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 2},
        )
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
        yaxis_title="Objective Value",
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
    return fig


# ---------------------------------------------------------------------------
# Section C — Lower Bound Progress & Gap
# ---------------------------------------------------------------------------


def _chart_lb_delta(conv: pd.DataFrame) -> go.Figure | None:
    """Line chart of lower bound improvement per iteration."""
    if conv.empty or len(conv) < 2:
        return None

    delta = conv["lower_bound"].diff().iloc[1:]
    iters = conv["iteration"].iloc[1:].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=delta.tolist(),
            name="LB Delta",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 1.5},
        )
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
        yaxis_title="\u0394 Lower Bound",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
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


# ---------------------------------------------------------------------------
# Section D — Cut Pool Evolution
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section E — Cut Management Heatmaps
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Section F — Iteration Timing
# ---------------------------------------------------------------------------

# Ordered list of primary timing components; the phase color map uses these
# names as keys where applicable.
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
    """Stacked area chart of timing components per iteration."""
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
            go.Scatter(
                x=iters,
                y=timing[col].tolist(),
                name=label,
                mode="lines",
                stackgroup="one",
                line={"width": 0.5, "color": color},
                fillcolor=color,
            )
        )

    fig.update_layout(
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


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Training tab always renders (convergence.parquet is always present)."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Training tab content area."""
    conv = data.conv
    if conv.empty:
        return "<p>No convergence data.</p>"

    # ------------------------------------------------------------------
    # Section A — Metrics Row
    # ------------------------------------------------------------------
    section_a = _build_metrics_row(data)

    # ------------------------------------------------------------------
    # Section B — Convergence Hero Chart
    # ------------------------------------------------------------------
    hero_fig = _chart_convergence_hero(conv)
    hero_html = make_chart_card(
        hero_fig,
        title="Convergence Bounds",
        chart_id="v2-training-convergence-hero",
        height=500,
    )
    section_b = section_title("Convergence") + chart_grid([hero_html], single=True)

    # ------------------------------------------------------------------
    # Section C — Lower Bound Progress & Gap
    # ------------------------------------------------------------------
    lb_delta_fig = _chart_lb_delta(conv)
    gap_fig = _chart_gap_evolution(conv)
    c_charts: list[str] = []
    if lb_delta_fig is not None:
        c_charts.append(
            make_chart_card(
                lb_delta_fig,
                title="Lower Bound Progress (delta per iteration)",
                chart_id="v2-training-lb-delta",
            )
        )
    if gap_fig is not None:
        c_charts.append(
            make_chart_card(
                gap_fig,
                title="Convergence Gap (%) per Iteration",
                chart_id="v2-training-gap",
            )
        )
    section_c = ""
    if c_charts:
        section_c = section_title("Lower Bound Progress & Gap") + chart_grid(c_charts)

    # ------------------------------------------------------------------
    # Section D — Cut Pool Evolution
    # ------------------------------------------------------------------
    cut_pool_fig = _chart_cut_pool(conv)
    cut_pool_html = make_chart_card(
        cut_pool_fig,
        title="Cut Pool Evolution",
        chart_id="v2-training-cut-pool",
    )
    summary_html = (
        f'<div class="chart-card">'
        f'<div style="padding: 1rem;">'
        f'<div class="metrics-grid">{_cut_pool_summary(conv)}</div>'
        f"</div></div>"
    )
    section_d = section_title("Cut Pool Evolution") + chart_grid(
        [cut_pool_html, summary_html]
    )

    # ------------------------------------------------------------------
    # Section E — Cut Management Heatmaps (collapsible, default collapsed)
    # ------------------------------------------------------------------
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
        default_collapsed=True,
    )

    # ------------------------------------------------------------------
    # Section F — Iteration Timing (collapsible, default collapsed)
    # ------------------------------------------------------------------
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
        default_collapsed=True,
    )

    return section_a + section_b + section_c + section_d + section_e + section_f
