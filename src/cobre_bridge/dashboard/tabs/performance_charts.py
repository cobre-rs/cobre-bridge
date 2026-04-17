"""Performance chart functions shared by the performance tab.

Provides timing breakdowns, solver diagnostics, LP dimensions, and scaling
charts used by the main performance tab module.

Timing column hierarchy (verified against cobre crates/cobre-sddp/src/
{forward,backward,training_output}.rs):

- **Top-level phases** (mutually exclusive, sum ≈ iteration total):
  ``forward_wall_ms``, ``backward_wall_ms``, ``cut_selection_ms``,
  ``mpi_allreduce_ms``, ``lower_bound_ms``, ``overhead_ms``.

- **Sub-components of ``backward_wall_ms``** (already inside it —
  DO NOT stack alongside backward_wall_ms): ``state_exchange_ms``,
  ``cut_batch_build_ms``, ``cut_sync_ms``, ``bwd_load_imbalance_ms``,
  ``bwd_scheduling_overhead_ms``.

- **Sub-components of ``forward_wall_ms``**: ``fwd_load_imbalance_ms``,
  ``fwd_scheduling_overhead_ms``.

- **Aggregate CPU (NOT wall time, not stackable)**: ``fwd_setup_ms``,
  ``bwd_setup_ms`` — sum of per-worker non-solve CPU time. Useful as a
  parallel-efficiency diagnostic, not for wall-time budgets.
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
from cobre_bridge.ui.theme import COLORS, PERFORMANCE_PHASE_COLORS

# ---------------------------------------------------------------------------
# Timing column categorisation — single source of truth for all timing charts.
# ---------------------------------------------------------------------------

TOP_LEVEL_PHASE_COLUMNS: tuple[str, ...] = (
    "forward_wall_ms",
    "backward_wall_ms",
    "cut_selection_ms",
    "mpi_allreduce_ms",
    "lower_bound_ms",
    "overhead_ms",
)

# Wall-time sub-components inside backward_wall_ms.
BACKWARD_SUBCOMPONENT_COLUMNS: tuple[str, ...] = (
    "state_exchange_ms",
    "cut_batch_build_ms",
    "cut_sync_ms",
    "bwd_load_imbalance_ms",
    "bwd_scheduling_overhead_ms",
)

# Wall-time sub-components inside forward_wall_ms.
FORWARD_SUBCOMPONENT_COLUMNS: tuple[str, ...] = (
    "fwd_load_imbalance_ms",
    "fwd_scheduling_overhead_ms",
)

# Aggregate per-worker CPU time — NOT stackable with wall-time columns.
AGGREGATE_CPU_COLUMNS: tuple[str, ...] = (
    "fwd_setup_ms",
    "bwd_setup_ms",
)

# Legacy-schema rollups (v0.4.4 and earlier). When present, treat them as a
# single opaque sub-component of their respective wall phase.
LEGACY_FORWARD_OVERHEAD: str = "fwd_rayon_overhead_ms"
LEGACY_BACKWARD_OVERHEAD: str = "bwd_rayon_overhead_ms"

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Component display name mapping for the waterfall chart.
_TIMING_COMPONENT_LABELS: dict[str, str] = {
    # v0.4.3+ column names.
    "forward_wall_ms": "Forward Wall",
    "backward_wall_ms": "Backward Wall",
    "lower_bound_ms": "Lower Bound",
    "cut_selection_ms": "Cut Selection",
    "mpi_allreduce_ms": "MPI AllReduce",
    "cut_sync_ms": "Cut Sync",
    "state_exchange_ms": "State Exchange",
    "cut_batch_build_ms": "Cut Batch Build",
    "overhead_ms": "Other Overhead",
    # Post-v0.4.4 epic-01 parallel overhead decomposition.
    "fwd_setup_ms": "Fwd Worker Setup",
    "fwd_load_imbalance_ms": "Fwd Load Imbalance",
    "fwd_scheduling_overhead_ms": "Fwd Scheduling",
    "bwd_setup_ms": "Bwd Worker Setup",
    "bwd_load_imbalance_ms": "Bwd Load Imbalance",
    "bwd_scheduling_overhead_ms": "Bwd Scheduling",
    # v0.4.4 parallel overhead rollups (superseded by decomposition above).
    "fwd_rayon_overhead_ms": "Fwd Rayon Overhead",
    "bwd_rayon_overhead_ms": "Bwd Rayon Overhead",
    # Legacy names for pre-0.4.3 outputs.
    "forward_solve_ms": "Forward Solve",
    "forward_sample_ms": "Forward Sample",
    "backward_solve_ms": "Backward Solve",
    "backward_cut_ms": "Backward Cut Add",
    "mpi_broadcast_ms": "MPI Broadcast",
    "rayon_overhead_ms": "Rayon Overhead",
    "io_write_ms": "IO Write",
}

# Colors for the parallel-overhead decomposition chart. Three-tone palette
# distinguishes setup (cool) / load imbalance (warm) / scheduling (accent) and
# reuses phase hues for forward vs backward via alpha variations.
_OVERHEAD_DECOMPOSITION_COLORS: dict[str, str] = {
    "setup": "#8B5CF6",
    "load_imbalance": "#F59E0B",
    "scheduling": "#DC4C4C",
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


_TOP_LEVEL_PHASE_LABELS: dict[str, tuple[str, str]] = {
    "forward_wall_ms": ("Forward", PERFORMANCE_PHASE_COLORS["forward"]),
    "backward_wall_ms": ("Backward", PERFORMANCE_PHASE_COLORS["backward"]),
    "cut_selection_ms": ("Cut Selection", PERFORMANCE_PHASE_COLORS["lp_solve"]),
    "lower_bound_ms": ("Lower Bound Eval", "#14B8A6"),
    "mpi_allreduce_ms": ("MPI AllReduce", "#8B5CF6"),
    "overhead_ms": ("Other Overhead", PERFORMANCE_PHASE_COLORS["overhead"]),
}

_LEGACY_TOP_LEVEL_MAP: dict[str, tuple[str, str]] = {
    "forward_solve_ms": ("Forward (legacy)", PERFORMANCE_PHASE_COLORS["forward"]),
    "backward_solve_ms": ("Backward (legacy)", PERFORMANCE_PHASE_COLORS["backward"]),
}


def _detect_top_level_columns(timing: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return [(column, label, color)] for present top-level phases.

    Modern schema takes priority; falls back to pre-0.4.3 legacy columns
    (``forward_solve_ms`` / ``backward_solve_ms``) when the new names are
    absent. The returned list is emission-ordered for stacking.
    """
    if "forward_wall_ms" in timing.columns:
        order = TOP_LEVEL_PHASE_COLUMNS
        labels = _TOP_LEVEL_PHASE_LABELS
    else:
        order = (
            "forward_solve_ms",
            "backward_solve_ms",
            "cut_selection_ms",
            "mpi_broadcast_ms",
            "lower_bound_ms",
            "overhead_ms",
        )
        labels = {**_TOP_LEVEL_PHASE_LABELS, **_LEGACY_TOP_LEVEL_MAP}
    return [
        (col, labels[col][0], labels[col][1])
        for col in order
        if col in timing.columns and col in labels
    ]


def chart_iteration_timing_breakdown(timing: pd.DataFrame) -> str:
    """Stacked bar per iteration using ONLY top-level phases.

    Top-level phases are non-overlapping (they partition the iteration wall
    time). Sub-components like ``cut_sync_ms`` are intentionally excluded —
    they live inside ``backward_wall_ms`` and stacking them here would
    double-count.
    """
    if timing.empty:
        return "<p>No timing data available.</p>"

    present = _detect_top_level_columns(timing)
    if not present:
        return "<p>No recognised top-level timing columns.</p>"

    iters = timing["iteration"].tolist()
    fig = go.Figure()
    for col, label, color in present:
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
        title="Iteration Timing — Top-Level Phases (non-overlapping, sums to total)",
        xaxis_title="Iteration",
        yaxis_title="Time (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def _solve_time_heatmap(solver_train: pd.DataFrame, phase: str, title: str) -> str:
    """Render a per-(stage, iteration) heatmap of ``solve_time_ms`` for *phase*."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    sub = solver_train[solver_train["phase"] == phase]
    if sub.empty:
        return f"<p>No {phase} solver data available.</p>"

    pivot = sub.pivot_table(
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
        title=title,
        xaxis_title="Stage",
        yaxis_title="Iteration",
        height=max(400, len(iters) * 6 + 120),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_backward_stage_heatmap(solver_train: pd.DataFrame) -> str:
    """Heatmap of backward-phase solve_time_ms: x=stage, y=iteration."""
    return _solve_time_heatmap(
        solver_train,
        phase="backward",
        title="Backward LP Solve Time Heatmap (ms) — Stages vs Iterations",
    )


def chart_forward_stage_heatmap(solver_train: pd.DataFrame) -> str:
    """Heatmap of forward-phase solve_time_ms: x=stage, y=iteration."""
    return _solve_time_heatmap(
        solver_train,
        phase="forward",
        title="Forward LP Solve Time Heatmap (ms) — Stages vs Iterations",
    )


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


def _has_decomposition_columns(timing: pd.DataFrame) -> bool:
    """True when the per-iteration timing frame carries the epic-01 columns."""
    required = {
        "fwd_setup_ms",
        "fwd_load_imbalance_ms",
        "fwd_scheduling_overhead_ms",
        "bwd_setup_ms",
        "bwd_load_imbalance_ms",
        "bwd_scheduling_overhead_ms",
    }
    return not timing.empty and required.issubset(timing.columns)


def chart_parallel_overhead_decomposition(timing: pd.DataFrame) -> str:
    """Side-by-side stacked bars of forward vs backward parallel overhead.

    Decomposes the parallel overhead (time the root worker waits beyond LP
    solve) into three components per phase: per-worker setup, load imbalance,
    and residual scheduling overhead. This surfaces the epic-01 instrumentation
    added in the post-v0.4.4 cobre timing schema. Returns a fallback message
    when the decomposition columns are absent.
    """
    if not _has_decomposition_columns(timing):
        return (
            "<p>Parallel overhead decomposition requires cobre post-v0.4.4 "
            "timing output (fwd/bwd_setup_ms, _load_imbalance_ms, "
            "_scheduling_overhead_ms).</p>"
        )

    iters = timing["iteration"].tolist()
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Forward Parallel Overhead", "Backward Parallel Overhead"),
        shared_yaxes=True,
    )

    components: list[tuple[str, str, str, str]] = [
        ("setup", "Worker Setup", "fwd_setup_ms", "bwd_setup_ms"),
        (
            "load_imbalance",
            "Load Imbalance",
            "fwd_load_imbalance_ms",
            "bwd_load_imbalance_ms",
        ),
        (
            "scheduling",
            "Scheduling",
            "fwd_scheduling_overhead_ms",
            "bwd_scheduling_overhead_ms",
        ),
    ]
    for key, label, fwd_col, bwd_col in components:
        color = _OVERHEAD_DECOMPOSITION_COLORS[key]
        fig.add_trace(
            go.Bar(
                x=iters,
                y=timing[fwd_col].tolist(),
                name=label,
                marker_color=color,
                legendgroup=key,
                hovertemplate=f"{label} (fwd): %{{y:.0f}} ms<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=iters,
                y=timing[bwd_col].tolist(),
                name=label,
                marker_color=color,
                legendgroup=key,
                showlegend=False,
                hovertemplate=f"{label} (bwd): %{{y:.0f}} ms<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        title="Parallel Overhead Decomposition — per Iteration (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
    return fig_to_html(fig)


def chart_parallel_overhead_share(timing: pd.DataFrame) -> str:
    """Horizontal bar: share of total parallel overhead by component (fwd+bwd).

    Aggregates the six decomposition columns across all iterations and
    displays each component as a percentage of the summed overhead. Gives a
    one-shot view of which overhead category dominates in the run.
    """
    if not _has_decomposition_columns(timing):
        return "<p>Decomposition share requires cobre post-v0.4.4 timing output.</p>"

    totals: dict[str, float] = {
        "Fwd Worker Setup": float(timing["fwd_setup_ms"].sum()),
        "Fwd Load Imbalance": float(timing["fwd_load_imbalance_ms"].sum()),
        "Fwd Scheduling": float(timing["fwd_scheduling_overhead_ms"].sum()),
        "Bwd Worker Setup": float(timing["bwd_setup_ms"].sum()),
        "Bwd Load Imbalance": float(timing["bwd_load_imbalance_ms"].sum()),
        "Bwd Scheduling": float(timing["bwd_scheduling_overhead_ms"].sum()),
    }
    grand = sum(totals.values())
    if grand == 0.0:
        return "<p>All parallel overhead components are zero.</p>"

    colors_map = {
        "Fwd Worker Setup": _OVERHEAD_DECOMPOSITION_COLORS["setup"],
        "Fwd Load Imbalance": _OVERHEAD_DECOMPOSITION_COLORS["load_imbalance"],
        "Fwd Scheduling": _OVERHEAD_DECOMPOSITION_COLORS["scheduling"],
        "Bwd Worker Setup": _OVERHEAD_DECOMPOSITION_COLORS["setup"],
        "Bwd Load Imbalance": _OVERHEAD_DECOMPOSITION_COLORS["load_imbalance"],
        "Bwd Scheduling": _OVERHEAD_DECOMPOSITION_COLORS["scheduling"],
    }

    fig = go.Figure()
    for label, value in totals.items():
        pct = value / grand * 100.0
        fig.add_trace(
            go.Bar(
                x=[pct],
                y=["Overhead Share"],
                name=label,
                orientation="h",
                marker_color=colors_map[label],
                text=[f"{label}: {pct:.1f}%"],
                textposition="inside",
                hovertemplate=f"{label}: {value / 1000:.1f}s ({pct:.1f}%)<extra></extra>",
            )
        )
    fig.update_layout(
        title="Cumulative Parallel Overhead Share (fwd + bwd)",
        barmode="stack",
        xaxis={"title": "% of total parallel overhead", "range": [0, 100]},
        legend=_LEGEND,
        margin=_MARGIN,
        height=220,
        showlegend=True,
    )
    return fig_to_html(fig)


def chart_basis_padding_usage(solver_train: pd.DataFrame) -> str:
    """Stacked bar per iteration: tight vs slack basis padding counts.

    Requires the post-v0.4.4 solver schema (``basis_padding_tight`` and
    ``basis_padding_slack`` columns). Aggregates across (phase, stage) to a
    single per-iteration bar pair so the reader can spot trends in warm-start
    basis padding as training progresses.
    """
    tight_col = "basis_padding_tight"
    slack_col = "basis_padding_slack"
    if (
        solver_train.empty
        or tight_col not in solver_train.columns
        or slack_col not in solver_train.columns
    ):
        return (
            "<p>Basis padding columns not present (requires cobre post-v0.4.4 "
            "solver schema).</p>"
        )

    agg = solver_train.groupby("iteration")[[tight_col, slack_col]].sum().reset_index()
    if (agg[tight_col].sum() + agg[slack_col].sum()) == 0:
        return "<p>Basis padding disabled for this run (all counts zero).</p>"

    iters = agg["iteration"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=iters,
            y=agg[tight_col].tolist(),
            name="Tight (informed)",
            marker_color=COLORS["lower_bound"],
            hovertemplate="Tight: %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=iters,
            y=agg[slack_col].tolist(),
            name="Slack (default)",
            marker_color=COLORS["future_cost"],
            hovertemplate="Slack: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Basis Padding Status per Iteration (summed across stages)",
        xaxis_title="Iteration",
        yaxis_title="Cut rows",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=380,
    )
    return fig_to_html(fig)


def chart_timing_waterfall(timing: pd.DataFrame) -> str:
    """Cumulative total-time donut over top-level phases.

    One chart per run: each slice is the total ms of a non-overlapping
    top-level phase across all iterations. Replaces a prior per-iteration
    stacked bar that mixed top-level and sub-component columns (double-count).
    """
    if timing.empty:
        return "<p>No timing data available.</p>"

    present = _detect_top_level_columns(timing)
    if not present:
        return "<p>No recognised top-level timing columns.</p>"

    totals = {label: float(timing[col].sum()) for col, label, _ in present}
    grand = sum(totals.values())
    if grand == 0.0:
        return "<p>All top-level phase totals are zero.</p>"

    colors_by_label = {label: color for _, label, color in present}
    fig = go.Figure(
        go.Pie(
            labels=list(totals.keys()),
            values=list(totals.values()),
            hole=0.5,
            marker={"colors": [colors_by_label[lbl] for lbl in totals]},
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.0f} ms (%{percent})<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Cumulative Phase Totals — {grand / 1000:.1f}s total",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def _backward_breakdown_columns(timing: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Pick the available backward sub-component columns (modern or legacy).

    Returns (column, label, color) triples in stacking order.
    """
    palette = {
        "state_exchange_ms": ("State Exchange", "#6366F1"),
        "cut_batch_build_ms": ("Cut Batch Build", "#0EA5E9"),
        "cut_sync_ms": ("Cut Sync (MPI)", "#8B5CF6"),
        "bwd_load_imbalance_ms": ("Load Imbalance", "#F59E0B"),
        "bwd_scheduling_overhead_ms": ("Scheduling Overhead", "#DC4C4C"),
        LEGACY_BACKWARD_OVERHEAD: ("Rayon Overhead (legacy)", "#DC4C4C"),
    }
    order = list(BACKWARD_SUBCOMPONENT_COLUMNS) + [LEGACY_BACKWARD_OVERHEAD]
    return [
        (col, palette[col][0], palette[col][1])
        for col in order
        if col in timing.columns
    ]


def _forward_breakdown_columns(timing: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Pick the available forward sub-component columns (modern or legacy)."""
    palette = {
        "fwd_load_imbalance_ms": ("Load Imbalance", "#F59E0B"),
        "fwd_scheduling_overhead_ms": ("Scheduling Overhead", "#DC4C4C"),
        LEGACY_FORWARD_OVERHEAD: ("Rayon Overhead (legacy)", "#DC4C4C"),
    }
    order = list(FORWARD_SUBCOMPONENT_COLUMNS) + [LEGACY_FORWARD_OVERHEAD]
    return [
        (col, palette[col][0], palette[col][1])
        for col in order
        if col in timing.columns
    ]


def _wall_breakdown_chart(
    timing: pd.DataFrame,
    wall_col: str,
    subcomponents: list[tuple[str, str, str]],
    phase_label: str,
    useful_color: str,
) -> go.Figure | None:
    """Build a stacked-bar breakdown of a wall-time phase.

    Adds a derived ``parallel_useful`` component (wall - Σ sub-components)
    so the stack sums to exactly ``wall_col``.  Negative residuals are
    clamped to zero and reported in the hover.
    """
    if wall_col not in timing.columns or timing.empty:
        return None
    if not subcomponents:
        return None

    iters = timing["iteration"].tolist()
    subtotal = timing[[c for c, _, _ in subcomponents]].sum(axis=1)
    residual = (timing[wall_col] - subtotal).clip(lower=0)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=iters,
            y=residual.tolist(),
            name="Parallel Useful (derived)",
            marker_color=useful_color,
            hovertemplate="Parallel Useful: %{y:.0f} ms<extra></extra>",
        )
    )
    for col, label, color in subcomponents:
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
        title=f"{phase_label} Wall-Time Breakdown (stack sums to {wall_col})",
        xaxis_title="Iteration",
        yaxis_title="Time (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig


def chart_backward_wall_breakdown(timing: pd.DataFrame) -> str:
    """Per-iteration stack decomposing ``backward_wall_ms`` into wall components.

    Layers (bottom to top): parallel useful (derived residual), load
    imbalance, scheduling overhead, cut sync MPI, cut batch build, state
    exchange. Sums exactly to ``backward_wall_ms`` each iteration.
    """
    if "backward_wall_ms" not in timing.columns:
        return "<p>Backward wall-time breakdown requires the v0.4.3+ schema.</p>"
    fig = _wall_breakdown_chart(
        timing,
        "backward_wall_ms",
        _backward_breakdown_columns(timing),
        "Backward Pass",
        useful_color=PERFORMANCE_PHASE_COLORS["backward"],
    )
    return fig_to_html(fig) if fig is not None else "<p>No backward breakdown data.</p>"


def chart_forward_wall_breakdown(timing: pd.DataFrame) -> str:
    """Per-iteration stack decomposing ``forward_wall_ms`` into wall components."""
    if "forward_wall_ms" not in timing.columns:
        return "<p>Forward wall-time breakdown requires the v0.4.3+ schema.</p>"
    fig = _wall_breakdown_chart(
        timing,
        "forward_wall_ms",
        _forward_breakdown_columns(timing),
        "Forward Pass",
        useful_color=PERFORMANCE_PHASE_COLORS["forward"],
    )
    return fig_to_html(fig) if fig is not None else "<p>No forward breakdown data.</p>"


def chart_parallel_efficiency(timing: pd.DataFrame) -> str:
    """Compare aggregate per-worker CPU (``bwd_setup_ms``) with parallel wall.

    ``bwd_setup_ms`` is the SUM of setup CPU across all Rayon workers within
    the parallel region. Comparing it to ``backward_wall_ms`` reveals how
    much of the parallel region is non-solve overhead. A high ratio means
    workers spend disproportionate CPU on load_model/add_rows/set_bounds.
    """
    required = {"bwd_setup_ms", "backward_wall_ms", "fwd_setup_ms", "forward_wall_ms"}
    if timing.empty or not required.issubset(timing.columns):
        return (
            "<p>Parallel-efficiency chart requires post-v0.4.4 setup columns "
            "(fwd_setup_ms, bwd_setup_ms).</p>"
        )

    iters = timing["iteration"].tolist()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=iters,
            y=timing["bwd_setup_ms"].tolist(),
            name="Bwd Setup CPU (Σ workers)",
            marker_color="#6366F1",
            opacity=0.75,
            hovertemplate="Bwd Setup: %{y:.0f} ms (aggregate CPU)<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=iters,
            y=timing["fwd_setup_ms"].tolist(),
            name="Fwd Setup CPU (Σ workers)",
            marker_color="#F59E0B",
            opacity=0.75,
            hovertemplate="Fwd Setup: %{y:.0f} ms (aggregate CPU)<extra></extra>",
        )
    )
    total_wall = timing["backward_wall_ms"] + timing["forward_wall_ms"]
    ratio = (
        (timing["bwd_setup_ms"] + timing["fwd_setup_ms"]) / total_wall.replace(0, pd.NA)
    ).fillna(0)
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=ratio.tolist(),
            name="Setup CPU / Wall Ratio",
            mode="lines+markers",
            line={"color": COLORS["deficit"], "width": 2},
            hovertemplate="Ratio: %{y:.2f}×<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Worker Setup CPU vs Parallel Wall — higher ratio = more non-solve CPU",
        xaxis_title="Iteration",
        barmode="group",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    fig.update_yaxes(title_text="Setup CPU (ms, sum across workers)", secondary_y=False)
    fig.update_yaxes(
        title_text="Setup CPU ÷ (fwd+bwd wall)",
        secondary_y=True,
        rangemode="tozero",
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Solver progression charts — surface temporal evolution across SDDP iterations
# ---------------------------------------------------------------------------


def chart_solver_progression(solver_train: pd.DataFrame) -> str:
    """Totals per SDDP iteration: LP solves, simplex iters, failures, retries.

    Four traces on two axes. Left axis: LP solves and simplex iterations (log
    scale). Right axis: LP failures and retry attempts. Summed across phases
    and stages per iteration so the reader can track solver workload evolution
    as training progresses.
    """
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    cols = ["lp_solves", "simplex_iterations", "lp_failures", "retry_attempts"]
    missing = [c for c in cols if c not in solver_train.columns]
    if missing:
        return (
            f"<p>solver/iterations.parquet missing columns: {', '.join(missing)}.</p>"
        )

    agg = solver_train.groupby("iteration", as_index=False)[cols].sum()
    iters = agg["iteration"].tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=agg["lp_solves"].tolist(),
            name="LP Solves",
            mode="lines",
            line={"color": COLORS["hydro"], "width": 2},
            hovertemplate="LP Solves: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=agg["simplex_iterations"].tolist(),
            name="Simplex Iterations",
            mode="lines",
            line={"color": COLORS["thermal"], "width": 2, "dash": "dot"},
            hovertemplate="Simplex: %{y:,}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=iters,
            y=agg["lp_failures"].tolist(),
            name="LP Failures",
            marker_color=COLORS["deficit"],
            opacity=0.75,
            hovertemplate="Failures: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Bar(
            x=iters,
            y=agg["retry_attempts"].tolist(),
            name="Retry Attempts",
            marker_color=COLORS["spillage"],
            opacity=0.75,
            hovertemplate="Retries: %{y}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Solver Progression — per SDDP iteration (summed across phases/stages)",
        xaxis_title="Iteration",
        barmode="group",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    fig.update_yaxes(title_text="Count (log)", type="log", secondary_y=False)
    fig.update_yaxes(
        title_text="Failures / Retries", secondary_y=True, rangemode="tozero"
    )
    return fig_to_html(fig)


def chart_basis_reuse_over_iterations(solver_train: pd.DataFrame) -> str:
    """Rolling basis reuse rate per SDDP iteration, split by phase.

    Reuse rate = 1 − basis_rejections / basis_offered. When warm-starting is
    working well this trends upward as training stabilises. A persistent
    decline signals that newly-added cuts keep invalidating stored bases.
    """
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    required = {"basis_offered", "basis_rejections", "iteration", "phase"}
    if not required.issubset(solver_train.columns):
        return "<p>solver/iterations.parquet missing basis columns.</p>"

    offered = solver_train[solver_train["basis_offered"] > 0].copy()
    if offered.empty:
        return "<p>No basis warm-start offered during training.</p>"

    offered["reuse_rate"] = 1.0 - offered["basis_rejections"] / offered["basis_offered"]

    phase_style = {
        "forward": (COLORS["hydro"], "solid"),
        "backward": (COLORS["thermal"], "solid"),
    }
    fig = go.Figure()
    for phase, (color, dash) in phase_style.items():
        sub = offered[offered["phase"] == phase]
        if sub.empty:
            continue
        per_iter = sub.groupby("iteration")["reuse_rate"].mean().sort_index()
        fig.add_trace(
            go.Scatter(
                x=per_iter.index.tolist(),
                y=per_iter.values.tolist(),
                name=f"{phase.title()} reuse",
                mode="lines",
                line={"color": color, "width": 2, "dash": dash},
                hovertemplate=f"{phase.title()}: %{{y:.2%}}<extra></extra>",
            )
        )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#8B9298",
        annotation_text="perfect reuse",
        annotation_position="right",
    )
    fig.update_layout(
        title="Basis Warm-start Reuse — per SDDP iteration",
        xaxis_title="Iteration",
        yaxis_title="Reuse rate",
        yaxis={"range": [0, 1.05]},
        legend=_LEGEND,
        margin=_MARGIN,
        height=400,
    )
    return fig_to_html(fig)


def chart_solver_time_breakdown_over_iterations(solver_train: pd.DataFrame) -> str:
    """Per-iteration stacked bar of solver CPU time components (backward phase).

    Stacks ``solve_time_ms`` alongside non-solve time (``load_model_time_ms``,
    ``add_rows_time_ms``, ``set_bounds_time_ms``, ``basis_set_time_ms``). Shows
    how much solver CPU each iteration consumed and how the split between
    solve and setup changes over training.
    """
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[solver_train["phase"] == "backward"]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    components = [
        ("solve_time_ms", "LP Solve", COLORS["hydro"]),
        ("set_bounds_time_ms", "Set Bounds", COLORS["thermal"]),
        ("add_rows_time_ms", "Add Rows (cuts)", COLORS["ncs"]),
        ("load_model_time_ms", "Load Model", COLORS["future_cost"]),
        ("basis_set_time_ms", "Basis Set", COLORS["spillage"]),
    ]
    present = [(c, label, color) for c, label, color in components if c in bwd.columns]
    if not present:
        return "<p>Solver CPU component columns absent.</p>"

    agg = bwd.groupby("iteration", as_index=False)[[c for c, _, _ in present]].sum()
    iters = agg["iteration"].tolist()

    fig = go.Figure()
    for col, label, color in present:
        fig.add_trace(
            go.Bar(
                x=iters,
                y=agg[col].tolist(),
                name=label,
                marker_color=color,
                hovertemplate=f"{label}: %{{y:,.0f}} ms<extra></extra>",
            )
        )
    fig.update_layout(
        title="Backward Solver CPU Components — per SDDP iteration (aggregate across stages/workers)",
        xaxis_title="Iteration",
        yaxis_title="Time (ms)",
        barmode="stack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_retry_level_heatmap(retry_histogram: pd.DataFrame) -> str:
    """Heatmap of retry counts by (stage, retry_level), summed across iters+phases.

    Surfaces *where* the solver has to retry. A hot row = a stage with
    systematic trouble; a hot column = an attempt level that frequently wins.
    """
    if retry_histogram.empty:
        return "<p>No retry histogram data available.</p>"
    required = {"stage", "retry_level", "count"}
    if not required.issubset(retry_histogram.columns):
        return "<p>retry_histogram.parquet missing expected columns.</p>"

    pivot = retry_histogram.pivot_table(
        index="stage",
        columns="retry_level",
        values="count",
        aggfunc="sum",
        fill_value=0,
    )
    stages = sorted(pivot.index.tolist())
    levels = sorted(pivot.columns.tolist())
    if not stages or not levels:
        return "<p>No retry data.</p>"

    z = [[int(pivot.loc[s, lvl]) for lvl in levels] for s in stages]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(lv) for lv in levels],
            y=[str(s) for s in stages],
            colorscale="OrRd",
            colorbar={"title": "count"},
            hovertemplate="Stage %{y}, retry %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Retry Count Heatmap — stage × retry level (summed across iterations & phases)",
        xaxis_title="Retry level",
        yaxis_title="Stage",
        height=max(400, len(stages) * 6 + 120),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Cut-pool insight charts — surface cut_selection/iterations.parquet
# ---------------------------------------------------------------------------


def chart_cut_count_per_stage(
    cut_selection: pd.DataFrame, iteration: int | None = None
) -> str:
    """Stacked bar per stage: populated (background) vs active (foreground).

    Snapshot at the last iteration by default.  Quickly reveals stages where
    the active fraction is low (most cuts inactive → cut selection biting) or
    close to 100 % (cut selection disabled or budget wide open).  The ONS
    case showed stage 0 sitting at 98 % active → a key red flag.
    """
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    iter_to_use = (
        int(cut_selection["iteration"].max()) if iteration is None else int(iteration)
    )
    snap = cut_selection[cut_selection["iteration"] == iter_to_use].sort_values("stage")
    if snap.empty:
        return f"<p>No cut selection data for iteration {iter_to_use}.</p>"

    stages = snap["stage"].tolist()
    populated = snap["cuts_populated"].tolist()
    active = snap["cuts_active_after"].tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=stages,
            y=populated,
            name="Populated",
            marker_color=COLORS["future_cost"],
            opacity=0.55,
            hovertemplate="Stage %{x}: %{y} populated<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=stages,
            y=active,
            name="Active",
            marker_color=COLORS["hydro"],
            hovertemplate="Stage %{x}: %{y} active<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Cut Count per Stage (iteration {iter_to_use})",
        xaxis_title="Stage",
        yaxis_title="Cuts",
        barmode="overlay",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def _pick_highlight_stages(cut_selection: pd.DataFrame, n: int = 8) -> list[int]:
    """Pick the *n* stages with the largest terminal active-cut count."""
    if cut_selection.empty:
        return []
    last_iter = int(cut_selection["iteration"].max())
    snap = cut_selection[cut_selection["iteration"] == last_iter]
    top = snap.nlargest(n, "cuts_active_after")["stage"].tolist()
    return sorted(int(s) for s in top)


def chart_active_cuts_growth_by_stage(cut_selection: pd.DataFrame) -> str:
    """Line chart: active cuts per iteration, one line per highlighted stage.

    Highlights the eight stages that finish training with the largest active
    pool.  Fast diagnostic for runaway stages (e.g. the ONS case's stage 0
    blowing past 9 k active cuts while everything else hovers around a few
    hundred).
    """
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    highlight = _pick_highlight_stages(cut_selection, n=8)
    if not highlight:
        return "<p>No cut selection data to highlight.</p>"

    palette = [
        "#DC4C4C",
        "#F59E0B",
        "#4A8B6F",
        "#4A90B8",
        "#8B5CF6",
        "#B87333",
        "#0EA5E9",
        "#EC4899",
    ]

    fig = go.Figure()
    for i, stage in enumerate(highlight):
        sub = cut_selection[cut_selection["stage"] == stage].sort_values("iteration")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["iteration"].tolist(),
                y=sub["cuts_active_after"].tolist(),
                name=f"Stage {stage}",
                mode="lines",
                line={"color": palette[i % len(palette)], "width": 2},
                hovertemplate=(
                    f"Stage {stage}, iter %{{x}}: %{{y}} active<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Active Cuts Growth per Stage (top 8 stages by terminal active count)",
        xaxis_title="Iteration",
        yaxis_title="Active cuts",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# LP-difficulty-over-training charts — overlay early/mid/late iterations
# ---------------------------------------------------------------------------


def _pick_probe_iterations(iters: list[int]) -> list[tuple[int, str]]:
    """Pick up to three iterations (first, middle, last) as (iter, label) pairs.

    Returns fewer than three when the run is too short to meaningfully
    distinguish early/mid/late.  Output is always sorted by iteration.
    """
    if not iters:
        return []
    iters = sorted(set(int(i) for i in iters))
    if len(iters) == 1:
        return [(iters[0], f"Iter {iters[0]}")]
    if len(iters) == 2:
        return [
            (iters[0], f"Iter {iters[0]} (first)"),
            (iters[-1], f"Iter {iters[-1]} (last)"),
        ]
    mid = iters[len(iters) // 2]
    return [
        (iters[0], f"Iter {iters[0]} (first)"),
        (mid, f"Iter {mid} (mid)"),
        (iters[-1], f"Iter {iters[-1]} (last)"),
    ]


def _per_stage_metric_multi_iter(
    solver_train: pd.DataFrame, metric: str, agg: str, phase: str
) -> go.Figure | None:
    """Build a multi-line chart of a per-(iter, stage) solver metric.

    *metric* is one of ``simplex_iterations`` (normalised to per-LP) or
    ``solve_time_ms`` (per-LP average).  Returns a figure with one line per
    probe iteration and the stage axis on X.
    """
    if solver_train.empty:
        return None
    sub = solver_train[
        (solver_train["phase"] == phase) & (solver_train["stage"] >= 0)
    ].copy()
    if sub.empty:
        return None

    probes = _pick_probe_iterations(sub["iteration"].unique().tolist())
    if not probes:
        return None

    palette = ["#4A90B8", "#F59E0B", "#DC4C4C"]
    fig = go.Figure()
    for i, (it, label) in enumerate(probes):
        snap = sub[sub["iteration"] == it]
        if snap.empty:
            continue
        if metric == "simplex_iterations":
            snap = snap.assign(
                _per_lp=snap["simplex_iterations"] / snap["lp_solves"].replace(0, pd.NA)
            )
            ycol, hover = "_per_lp", "simplex/LP"
        elif metric == "solve_time_ms":
            snap = snap.assign(
                _per_lp=snap["solve_time_ms"] / snap["lp_solves"].replace(0, pd.NA)
            )
            ycol, hover = "_per_lp", "ms/LP"
        else:
            raise ValueError(f"unsupported metric {metric!r}")
        by_stage = (
            snap.groupby("stage")[ycol]
            .agg(agg)
            .reset_index()
            .dropna()
            .sort_values("stage")
        )
        fig.add_trace(
            go.Scatter(
                x=by_stage["stage"].tolist(),
                y=by_stage[ycol].tolist(),
                name=label,
                mode="lines+markers",
                line={"color": palette[i % len(palette)], "width": 2},
                marker={"size": 4},
                hovertemplate=(
                    f"{label} stage %{{x}}: %{{y:.2f}} {hover}<extra></extra>"
                ),
            )
        )
    return fig


def chart_lp_difficulty_multi_iter(solver_train: pd.DataFrame) -> str:
    """Avg simplex iterations per LP, per stage, overlaid for 3 iterations (backward)."""
    fig = _per_stage_metric_multi_iter(
        solver_train, "simplex_iterations", agg="mean", phase="backward"
    )
    if fig is None:
        return "<p>No backward solver data available.</p>"
    fig.update_layout(
        title="LP Difficulty per Stage (avg simplex iter / LP, backward phase)",
        xaxis_title="Stage",
        yaxis_title="Simplex iter / LP",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_lp_solve_time_multi_iter(solver_train: pd.DataFrame) -> str:
    """Avg solve ms/LP per stage for 3 iterations (backward, log y)."""
    fig = _per_stage_metric_multi_iter(
        solver_train, "solve_time_ms", agg="mean", phase="backward"
    )
    if fig is None:
        return "<p>No backward solver data available.</p>"
    fig.update_layout(
        title="LP Solve Time per Stage (ms/LP, backward phase, log scale)",
        xaxis_title="Stage",
        yaxis_title="ms / LP",
        yaxis_type="log",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Cross-correlation + forward-deceleration diagnostics
# ---------------------------------------------------------------------------


def chart_cuts_vs_solve_time_scatter(
    solver_train: pd.DataFrame, cut_selection: pd.DataFrame
) -> str:
    """Scatter of active cuts (x) vs avg solve ms/LP (y), coloured by stage.

    One point per (iteration, stage) pair, backward phase. Reveals the
    empirical cost-of-cut curve: cuts → LP size → solve time. Stages with
    many cuts that DON'T solve slowly (lower-left of their hue band) are
    well-warm-started; stages where cuts translate directly into LP time
    (steep positive slope) are the cut-management targets.
    """
    if solver_train.empty or cut_selection.empty:
        return "<p>Requires both solver and cut selection data.</p>"
    bwd = solver_train[
        (solver_train["phase"] == "backward") & (solver_train["stage"] >= 0)
    ].copy()
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    bwd["ms_per_lp"] = bwd["solve_time_ms"] / bwd["lp_solves"].replace(0, pd.NA)
    merged = bwd.merge(
        cut_selection[["iteration", "stage", "cuts_active_after"]],
        on=["iteration", "stage"],
        how="inner",
    ).dropna(subset=["ms_per_lp", "cuts_active_after"])
    if merged.empty:
        return "<p>No overlapping (iteration, stage) data.</p>"

    fig = go.Figure(
        go.Scatter(
            x=merged["cuts_active_after"].tolist(),
            y=merged["ms_per_lp"].tolist(),
            mode="markers",
            marker={
                "size": 5,
                "color": merged["stage"].tolist(),
                "colorscale": "Viridis",
                "colorbar": {"title": "Stage"},
                "opacity": 0.7,
                "line": {"width": 0},
            },
            text=[
                f"iter {it}, stage {st}, {cuts} cuts, {ms:.1f} ms/LP"
                for it, st, cuts, ms in zip(
                    merged["iteration"].tolist(),
                    merged["stage"].tolist(),
                    merged["cuts_active_after"].tolist(),
                    merged["ms_per_lp"].tolist(),
                )
            ],
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Active Cuts vs LP Solve Time — one point per (iteration, stage)",
        xaxis_title="Active cuts at stage",
        yaxis_title="Avg solve ms / LP",
        margin=_MARGIN,
        height=440,
        showlegend=False,
    )
    return fig_to_html(fig)


def chart_forward_basis_set_vs_stage0_cuts(
    solver_train: pd.DataFrame, cut_selection: pd.DataFrame
) -> str:
    """Dual-axis: total forward basis_set time vs stage-0 active cut count.

    The ONS production case's key finding — forward ``basis_set`` grew
    linearly with the stage-0 cut pool. When this chart shows two nearly
    parallel curves, forward pass deceleration is caused by cut growth at
    the unselected stage 0, not by LP difficulty. A flat purple line
    against a rising red dashed line means basis padding is free — a good
    sign that warm-start is absorbing cut growth.
    """
    if solver_train.empty or cut_selection.empty:
        return "<p>Requires both solver and cut selection data.</p>"
    fwd = solver_train[solver_train["phase"] == "forward"]
    if fwd.empty or "basis_set_time_ms" not in fwd.columns:
        return "<p>No forward basis_set data available.</p>"

    per_iter_basis = (
        fwd.groupby("iteration", as_index=False)["basis_set_time_ms"]
        .sum()
        .sort_values("iteration")
    )
    stage0 = cut_selection[cut_selection["stage"] == 0].sort_values("iteration")
    if stage0.empty:
        return "<p>No stage-0 cut selection data.</p>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=per_iter_basis["iteration"].tolist(),
            y=(per_iter_basis["basis_set_time_ms"] / 1000.0).tolist(),
            name="Forward basis_set (s)",
            mode="lines+markers",
            line={"color": "#6D28D9", "width": 2},
            hovertemplate="Iter %{x}: %{y:.2f} s basis_set<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=stage0["iteration"].tolist(),
            y=stage0["cuts_active_after"].tolist(),
            name="Stage 0 active cuts",
            mode="lines",
            line={"color": COLORS["deficit"], "width": 2, "dash": "dash"},
            hovertemplate="Iter %{x}: %{y} stage-0 cuts<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Forward Deceleration — basis_set Tracks Stage-0 Cut Growth",
        xaxis_title="Iteration",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    fig.update_yaxes(title_text="Forward basis_set time (s)", secondary_y=False)
    fig.update_yaxes(
        title_text="Stage 0 active cuts", secondary_y=True, rangemode="tozero"
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Convergence against wall-clock (rather than iteration)
# ---------------------------------------------------------------------------


def chart_convergence_vs_wall_time(conv: pd.DataFrame, timing: pd.DataFrame) -> str:
    """Lower & upper bound plotted against cumulative wall time.

    Complements the iteration-axis convergence chart.  Reveals whether the
    algorithm is making efficient use of clock time — two runs can look
    identical vs iteration yet differ by hours in wall time because of
    per-iteration cost growth.  Uses ``time_total_ms`` from the convergence
    table when present, falling back to the sum of top-level phases in the
    timing table.
    """
    if conv.empty:
        return "<p>No convergence data available.</p>"

    if "time_total_ms" in conv.columns and conv["time_total_ms"].sum() > 0:
        per_iter_ms = conv["time_total_ms"].astype(float)
    elif not timing.empty:
        cols = [c for c in TOP_LEVEL_PHASE_COLUMNS if c in timing.columns]
        if not cols:
            return "<p>Cannot derive per-iteration wall time.</p>"
        per_iter_ms = (
            timing[["iteration", *cols]]
            .set_index("iteration")
            .reindex(conv["iteration"])
            .fillna(0)
            .sum(axis=1)
            .astype(float)
            .values
        )
    else:
        return "<p>Cannot derive per-iteration wall time.</p>"

    cum_s = (pd.Series(per_iter_ms).cumsum() / 1000.0).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=cum_s,
            y=conv["lower_bound"].tolist(),
            name="Lower bound",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 2},
            hovertemplate="%{x:.1f} s wall: LB %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=cum_s,
            y=conv["upper_bound_mean"].tolist(),
            name="Upper bound (mean)",
            mode="lines",
            line={"color": COLORS["upper_bound"], "width": 2},
            hovertemplate="%{x:.1f} s wall: UB %{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Convergence vs Wall Time",
        xaxis_title="Cumulative wall time (s)",
        yaxis_title="Objective value",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)
