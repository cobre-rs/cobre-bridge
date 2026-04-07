"""Performance chart functions shared by the performance tab.

Provides timing breakdowns, solver diagnostics, LP dimensions, and scaling
charts used by the main performance tab module.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    fig_to_html,
)
from cobre_bridge.ui.theme import COLORS

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Component display name mapping for the waterfall chart.
_TIMING_COMPONENT_LABELS: dict[str, str] = {
    "forward_solve_ms": "Forward Solve",
    "forward_sample_ms": "Forward Sample",
    "backward_solve_ms": "Backward Solve",
    "backward_cut_ms": "Backward Cut Add",
    "cut_selection_ms": "Cut Selection",
    "mpi_allreduce_ms": "MPI AllReduce",
    "mpi_broadcast_ms": "MPI Broadcast",
    "state_exchange_ms": "State Exchange",
    "cut_batch_build_ms": "Cut Batch Build",
    "rayon_overhead_ms": "Rayon Overhead",
    "overhead_ms": "Other Overhead",
    "io_write_ms": "IO Write",
}

_TIMING_COMPONENT_COLORS: list[str] = [
    "#4A90B8",
    "#A8D4F0",
    "#F5A623",
    "#F0D080",
    "#4A8B6F",
    "#8BC4A8",
    "#DC4C4C",
    "#F4A0A0",
    "#B87333",
    "#8B9298",
    "#607D8B",
    "#90A4AE",
]

# ---------------------------------------------------------------------------
# Chart functions
# ---------------------------------------------------------------------------


def chart_iteration_timing_breakdown(timing: pd.DataFrame) -> str:
    """Stacked bar per iteration: forward_solve, backward_solve, overhead."""
    if timing.empty:
        return "<p>No timing data available.</p>"

    overhead_cols = [
        c
        for c in timing.columns
        if c
        not in {
            "iteration",
            "forward_solve_ms",
            "backward_solve_ms",
        }
        and c.endswith("_ms")
    ]
    timing = timing.copy()
    timing["overhead_ms"] = timing[overhead_cols].sum(axis=1)

    iters = timing["iteration"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=iters,
            y=timing["forward_solve_ms"].tolist(),
            name="Forward Solve",
            marker_color=COLORS["hydro"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=iters,
            y=timing["backward_solve_ms"].tolist(),
            name="Backward Solve",
            marker_color=COLORS["thermal"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=iters,
            y=timing["overhead_ms"].tolist(),
            name="Overhead (other)",
            marker_color=COLORS["future_cost"],
        )
    )
    fig.update_layout(
        title="Iteration Timing Breakdown (ms per iteration)",
        xaxis_title="Iteration",
        yaxis_title="Time (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_backward_stage_heatmap(solver_train: pd.DataFrame) -> str:
    """Heatmap of solve_time_ms: x=stage, y=iteration (backward phase only)."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[solver_train["phase"] == "backward"]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    pivot = bwd.pivot_table(
        index="iteration", columns="stage", values="solve_time_ms", aggfunc="sum"
    )
    stages = sorted(pivot.columns.tolist())
    iters = sorted(pivot.index.tolist())
    z = [
        [
            float(pivot.loc[it, s]) if s in pivot.columns and it in pivot.index else 0.0
            for s in stages
        ]
        for it in iters
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(s) for s in stages],
            y=[str(i) for i in iters],
            colorscale="YlOrRd",
            colorbar={"title": "ms"},
        )
    )
    fig.update_layout(
        title="Backward LP Solve Time Heatmap (ms) — Stages vs Iterations",
        xaxis_title="Stage",
        yaxis_title="Iteration",
        height=max(400, len(iters) * 6 + 120),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_simplex_by_stage(solver_train: pd.DataFrame) -> str:
    """Bar chart of average simplex iterations per stage (backward phase)."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[solver_train["phase"] == "backward"]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    avg = bwd.groupby("stage")["simplex_iterations"].mean().sort_index()
    stages = [str(s) for s in avg.index.tolist()]
    values = avg.values.tolist()

    fig = go.Figure(
        go.Bar(
            x=stages,
            y=values,
            marker_color=COLORS["thermal"],
            name="Avg Simplex Iterations",
        )
    )
    fig.update_layout(
        title="Average Simplex Iterations per Stage (backward phase)",
        xaxis_title="Stage",
        yaxis_title="Simplex Iterations",
        legend=_LEGEND,
        margin=_MARGIN,
        height=400,
        showlegend=False,
    )
    return fig_to_html(fig)


def chart_lp_dimensions(scaling_report: dict) -> str:
    """Dual-axis bar chart: num_cols, num_rows, num_nz per stage."""
    stages_data = scaling_report.get("stages", [])
    if not stages_data:
        return "<p>No scaling report data available.</p>"

    stage_ids = [str(s["stage_id"]) for s in stages_data]
    num_cols = [s["dimensions"]["num_cols"] for s in stages_data]
    num_rows = [s["dimensions"]["num_rows"] for s in stages_data]
    num_nz = [s["dimensions"]["num_nz"] for s in stages_data]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=stage_ids,
            y=num_cols,
            name="Columns",
            marker_color=COLORS["hydro"],
            opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=stage_ids,
            y=num_rows,
            name="Rows",
            marker_color=COLORS["thermal"],
            opacity=0.8,
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=stage_ids,
            y=num_nz,
            name="Non-zeros",
            line={"color": COLORS["deficit"], "width": 2},
            mode="lines+markers",
        ),
        secondary_y=True,
    )
    fig.update_yaxes(title_text="Columns / Rows", secondary_y=False)
    fig.update_yaxes(title_text="Non-zeros", secondary_y=True)
    fig.update_layout(
        title="LP Dimensions by Stage",
        xaxis_title="Stage",
        barmode="group",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_scaling_quality(scaling_report: dict) -> str:
    """Line chart of coefficient ratio (pre/post scaling) per stage. Log y-axis."""
    stages_data = scaling_report.get("stages", [])
    if not stages_data:
        return "<p>No scaling report data available.</p>"

    stage_ids = []
    pre_ratios = []
    post_ratios = []
    for s in stages_data:
        pre = s.get("pre_scaling", {})
        post = s.get("post_scaling", {})
        pre_ratio = pre.get("matrix_coeff_ratio")
        post_ratio = post.get("matrix_coeff_ratio")
        if pre_ratio is not None and post_ratio is not None:
            stage_ids.append(str(s["stage_id"]))
            pre_ratios.append(float(pre_ratio))
            post_ratios.append(float(post_ratio))

    if not stage_ids:
        return "<p>No coefficient ratio data in scaling report.</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=stage_ids,
            y=pre_ratios,
            name="Pre-scaling ratio",
            line={"color": COLORS["deficit"], "width": 2},
            mode="lines+markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stage_ids,
            y=post_ratios,
            name="Post-scaling ratio",
            line={"color": COLORS["lower_bound"], "width": 2},
            mode="lines+markers",
        )
    )
    fig.update_layout(
        title="Matrix Coefficient Ratio by Stage (log scale — lower is better)",
        xaxis_title="Stage",
        yaxis_title="Coefficient Ratio",
        yaxis_type="log",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_simulation_scenario_times(solver_sim: pd.DataFrame) -> str:
    """Bar chart of solve_time_ms per simulation scenario."""
    if solver_sim.empty:
        return "<p>No simulation solver data available.</p>"

    agg = (
        solver_sim.groupby("iteration")["solve_time_ms"]
        .sum()
        .reset_index()
        .sort_values("iteration")
    )
    fig = go.Figure(
        go.Bar(
            x=agg["iteration"].astype(str).tolist(),
            y=agg["solve_time_ms"].tolist(),
            marker_color=COLORS["ncs"],
            name="Solve Time",
        )
    )
    fig.update_layout(
        title="Simulation Solve Time per Scenario (ms)",
        xaxis_title="Scenario",
        yaxis_title="Solve Time (ms)",
        margin=_MARGIN,
        height=400,
        showlegend=False,
    )
    return fig_to_html(fig)


def chart_basis_reuse(solver_train: pd.DataFrame) -> str:
    """Line chart of basis reuse rate per stage (backward phase, averaged over iterations)."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[solver_train["phase"] == "backward"].copy()
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    # Only use rows where basis was offered at least once
    offered = bwd[bwd["basis_offered"] > 0].copy()
    if offered.empty:
        return "<p>No basis warm-start data available (basis_offered=0 everywhere).</p>"

    offered["reuse_rate"] = 1.0 - offered["basis_rejections"] / offered["basis_offered"]
    avg_reuse = offered.groupby("stage")["reuse_rate"].mean().sort_index()
    stages = [str(s) for s in avg_reuse.index.tolist()]
    values = avg_reuse.values.tolist()

    fig = go.Figure(
        go.Scatter(
            x=stages,
            y=values,
            mode="lines+markers",
            line={"color": COLORS["hydro"], "width": 2},
            name="Basis Reuse Rate",
        )
    )
    fig.update_layout(
        title="Basis Warm-start Reuse Rate per Stage (backward phase, avg over iterations)",
        xaxis_title="Stage",
        yaxis_title="Reuse Rate (0-1)",
        yaxis={"range": [0, 1]},
        legend=_LEGEND,
        margin=_MARGIN,
        height=400,
    )
    return fig_to_html(fig)


def chart_solver_time_breakdown_by_phase(solver_train: pd.DataFrame) -> str:
    """Stacked bar: solve, load_model, add_rows, set_bounds per phase."""
    if solver_train.empty:
        return "<p>No solver data.</p>"
    components = [
        ("solve_time_ms", "LP Solve", COLORS["hydro"]),
        ("set_bounds_time_ms", "Set Bounds", COLORS["thermal"]),
        ("add_rows_time_ms", "Add Rows (cuts)", COLORS["ncs"]),
        ("load_model_time_ms", "Load Model", COLORS["future_cost"]),
    ]
    fig = go.Figure()
    for col, label, color in components:
        if col not in solver_train.columns:
            continue
        vals = solver_train.groupby("phase")[col].sum() / 1000.0  # seconds
        fig.add_trace(
            go.Bar(
                x=[p.title() for p in vals.index],
                y=vals.values,
                name=label,
                marker_color=color,
            )
        )
    fig.update_layout(
        title="Time Breakdown by Phase (seconds)",
        yaxis_title="Time (s)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=400,
    )
    return fig_to_html(fig)


def chart_solver_time_per_stage(solver_train: pd.DataFrame) -> str:
    """Stacked bar per stage: solve vs overhead (set_bounds+add_rows+load_model)."""
    if solver_train.empty:
        return "<p>No solver data.</p>"
    bw = solver_train[solver_train["phase"] == "backward"].copy()
    if bw.empty:
        return "<p>No backward phase data.</p>"
    overhead_cols = [
        c
        for c in ["load_model_time_ms", "add_rows_time_ms", "set_bounds_time_ms"]
        if c in bw.columns
    ]
    bw["overhead_ms"] = bw[overhead_cols].sum(axis=1) if overhead_cols else 0
    grouped = bw.groupby("stage")[["solve_time_ms", "overhead_ms"]].mean()
    stages = sorted(grouped.index)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stages,
            y=[grouped["solve_time_ms"].get(s, 0) for s in stages],
            name="LP Solve",
            marker_color=COLORS["hydro"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=stages,
            y=[grouped["overhead_ms"].get(s, 0) for s in stages],
            name="Overhead (bounds+rows+model)",
            marker_color=COLORS["thermal"],
        )
    )
    fig.update_layout(
        title="Backward Pass: Avg Solve vs Overhead per Stage (ms)",
        xaxis_title="Stage",
        yaxis_title="Time (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=400,
    )
    return fig_to_html(fig)


def chart_forward_vs_backward_per_iter(solver_train: pd.DataFrame) -> str:
    """Per-iteration: forward total solve time vs backward total solve time."""
    if solver_train.empty:
        return "<p>No solver data.</p>"
    by_iter_phase = (
        solver_train.groupby(["iteration", "phase"])["solve_time_ms"]
        .sum()
        .unstack(fill_value=0)
    )
    iters = sorted(by_iter_phase.index)
    fig = go.Figure()
    if "forward" in by_iter_phase.columns:
        fig.add_trace(
            go.Bar(
                x=iters,
                y=[by_iter_phase["forward"].get(i, 0) / 1000 for i in iters],
                name="Forward",
                marker_color=COLORS["hydro"],
            )
        )
    if "backward" in by_iter_phase.columns:
        fig.add_trace(
            go.Bar(
                x=iters,
                y=[by_iter_phase["backward"].get(i, 0) / 1000 for i in iters],
                name="Backward",
                marker_color=COLORS["thermal"],
            )
        )
    fig.update_layout(
        title="LP Solve Time per Iteration (seconds)",
        xaxis_title="Iteration",
        yaxis_title="Time (s)",
        barmode="group",
        legend=_LEGEND,
        margin=_MARGIN,
        height=400,
    )
    return fig_to_html(fig)


def chart_set_bounds_by_stage(solver_train: pd.DataFrame) -> str:
    """Per-stage avg set_bounds_time_ms for backward phase — often a hidden bottleneck."""
    if solver_train.empty or "set_bounds_time_ms" not in solver_train.columns:
        return "<p>No set_bounds data.</p>"
    bw = solver_train[solver_train["phase"] == "backward"]
    avg = bw.groupby("stage")["set_bounds_time_ms"].mean()
    stages = sorted(avg.index)
    fig = go.Figure(
        go.Bar(
            x=stages,
            y=[avg.get(s, 0) for s in stages],
            marker_color=COLORS["spillage"],
        )
    )
    fig.update_layout(
        title="Backward Pass: Avg set_bounds Time per Stage (ms)",
        xaxis_title="Stage",
        yaxis_title="Time (ms)",
        showlegend=False,
        margin=_MARGIN,
        height=350,
    )
    return fig_to_html(fig)


def chart_cost_per_simplex_iter(solver_train: pd.DataFrame) -> str:
    """Line chart: average microseconds per simplex iteration by stage (backward pass)."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[
        (solver_train["phase"] == "backward")
        & (solver_train["stage"] >= 0)
        & (solver_train["simplex_iterations"] > 0)
    ]
    if bwd.empty:
        return "<p>No backward solver data with simplex iterations available.</p>"

    avg = (
        bwd.assign(
            us_per_iter=bwd["solve_time_ms"] * 1000.0 / bwd["simplex_iterations"]
        )
        .groupby("stage")["us_per_iter"]
        .mean()
        .sort_index()
    )

    fig = go.Figure(
        go.Scatter(
            x=avg.index.tolist(),
            y=avg.values.tolist(),
            mode="lines+markers",
            line={"color": COLORS["thermal"], "width": 2},
            marker={"size": 4},
            name="us / simplex iter",
            hovertemplate="Stage: %{x}<br>Cost: %{y:.2f} us/iter<extra></extra>",
        )
    )
    fig.update_layout(
        title="Solver Cost per Simplex Iteration by Stage (Backward, averaged over SDDP iters)",
        xaxis_title="Stage",
        yaxis_title="Microseconds per Simplex Iteration",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
        showlegend=False,
    )
    return fig_to_html(fig)


def chart_timing_waterfall(timing: pd.DataFrame) -> str:
    """Stacked bar per iteration showing all non-zero timing components."""
    if timing.empty:
        return "<p>No timing data available.</p>"

    component_cols = [
        c for c in _TIMING_COMPONENT_LABELS if c in timing.columns and c != "iteration"
    ]
    # Drop components that are entirely zero or missing
    active_cols = [c for c in component_cols if timing[c].sum() > 0]
    if not active_cols:
        return "<p>No non-zero timing components found.</p>"

    iters = timing["iteration"].tolist()
    fig = go.Figure()
    for i, col in enumerate(active_cols):
        label = _TIMING_COMPONENT_LABELS.get(col, col)
        color = _TIMING_COMPONENT_COLORS[i % len(_TIMING_COMPONENT_COLORS)]
        fig.add_trace(
            go.Bar(
                x=iters,
                y=timing[col].tolist(),
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{y:.0f}} ms<extra></extra>",
            )
        )
    fig.update_layout(
        title="Full Timing Breakdown per Iteration",
        xaxis_title="Iteration",
        yaxis_title="Time (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)
