"""v2 Performance tab module for the Cobre dashboard.

Displays run summary metrics (training time, simulation time, LP solve stats),
training iteration breakdown charts, the full timing waterfall, and LP solver
diagnostics (heatmaps, simplex by stage, basis reuse, scaling, retries).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go

from cobre_bridge.dashboard.tabs.performance import (
    chart_backward_stage_heatmap,
    chart_basis_reuse,
    chart_cost_per_simplex_iter,
    chart_forward_vs_backward_per_iter,
    chart_iteration_timing_breakdown,
    chart_lp_dimensions,
    chart_scaling_quality,
    chart_set_bounds_by_stage,
    chart_simplex_by_stage,
    chart_simulation_scenario_times,
    chart_solver_time_breakdown_by_phase,
    chart_solver_time_per_stage,
    chart_timing_waterfall,
)
from cobre_bridge.ui.html import (
    chart_grid,
    collapsible_section,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.plotly_helpers import fig_to_html
from cobre_bridge.ui.theme import COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-performance"
TAB_LABEL = "Performance"
TAB_ORDER = 90


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_time(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string.

    Returns ``"X.X s"`` below 60 s, ``"X.X min"`` below 3600 s, and
    ``"X.XX h"`` for anything longer.
    """
    if seconds >= 3600.0:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 60.0:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 6-card metrics row from timing and solver DataFrames.

    Card order matches the v1 ``build_performance_metrics_html`` layout:
    1. Total Training Time
    2. Simulation Time (wall-clock or CPU)
    3. Avg LP Solve (training)
    4. Avg LP Solve (simulation)
    5. Total LP Solves
    6. Total Simplex Iterations
    """
    # ------------------------------------------------------------------
    # Training time: prefer wall-clock from metadata, fall back to timing
    # ------------------------------------------------------------------
    meta_duration = data.metadata.get("run_info", {}).get("duration_seconds")
    if meta_duration and float(meta_duration) > 0:
        total_train_s = float(meta_duration)
    else:
        timing: pd.DataFrame = data.timing
        total_train_ms = 0.0
        if not timing.empty:
            time_cols = [
                c for c in timing.columns if c.endswith("_ms") and c != "iteration"
            ]
            if time_cols:
                total_train_ms = float(timing[time_cols].sum().sum())
        total_train_s = total_train_ms / 1000.0

    train_str = _format_time(total_train_s)

    # ------------------------------------------------------------------
    # Simulation time: prefer wall-clock from simulation_manifest
    # ------------------------------------------------------------------
    sim_manifest_duration = data.simulation_manifest.get("duration_seconds")
    if sim_manifest_duration and float(sim_manifest_duration) > 0:
        total_sim_s = float(sim_manifest_duration)
        sim_is_wallclock = True
    else:
        solver_sim: pd.DataFrame = data.solver_sim
        total_sim_ms = (
            float(solver_sim["solve_time_ms"].sum()) if not solver_sim.empty else 0.0
        )
        total_sim_s = total_sim_ms / 1000.0
        sim_is_wallclock = False

    sim_str = _format_time(total_sim_s)
    sim_label = "Total Simulation Time" if sim_is_wallclock else "Simulation CPU Time"

    # ------------------------------------------------------------------
    # Avg LP solve time — training
    # ------------------------------------------------------------------
    solver_train: pd.DataFrame = data.solver_train
    if not solver_train.empty:
        train_lp_solves = solver_train["lp_solves"].sum()
        train_lp_time = solver_train["solve_time_ms"].sum()
        avg_lp_train_ms = (
            float(train_lp_time) / float(train_lp_solves)
            if train_lp_solves > 0
            else 0.0
        )
    else:
        train_lp_solves = 0
        avg_lp_train_ms = 0.0

    # ------------------------------------------------------------------
    # Avg LP solve time — simulation
    # ------------------------------------------------------------------
    solver_sim = data.solver_sim
    if not solver_sim.empty:
        sim_lp_solves = solver_sim["lp_solves"].sum()
        sim_lp_time = solver_sim["solve_time_ms"].sum()
        avg_lp_sim_ms = (
            float(sim_lp_time) / float(sim_lp_solves) if sim_lp_solves > 0 else 0.0
        )
    else:
        sim_lp_solves = 0
        avg_lp_sim_ms = 0.0

    # ------------------------------------------------------------------
    # Total LP solves and simplex iterations (training + simulation)
    # ------------------------------------------------------------------
    all_lp_solves = int(train_lp_solves) + int(sim_lp_solves)

    train_simplex = (
        int(solver_train["simplex_iterations"].sum()) if not solver_train.empty else 0
    )
    sim_simplex = (
        int(solver_sim["simplex_iterations"].sum()) if not solver_sim.empty else 0
    )
    total_simplex = train_simplex + sim_simplex

    # ------------------------------------------------------------------
    # Build cards
    # ------------------------------------------------------------------
    cards = [
        metric_card(
            value=train_str,
            label="Total Training Time",
            color=COLORS["lower_bound"],
        ),
        metric_card(
            value=sim_str,
            label=sim_label,
            color=COLORS["ncs"],
        ),
        metric_card(
            value=f"{avg_lp_train_ms:.2f} ms",
            label="Avg LP Solve (training)",
            color=COLORS["hydro"],
        ),
        metric_card(
            value=f"{avg_lp_sim_ms:.2f} ms",
            label="Avg LP Solve (simulation)",
            color=COLORS["spillage"],
        ),
        metric_card(
            value=f"{all_lp_solves:,}",
            label="Total LP Solves",
            color=COLORS["thermal"],
        ),
        metric_card(
            value=f"{total_simplex:,}",
            label="Total Simplex Iterations",
            color=COLORS["spillage"],
        ),
    ]
    return metrics_grid(cards)


def _chart_retry_histogram(retry_histogram: pd.DataFrame) -> str:
    """Render a bar chart of solver retry counts.

    Args:
        retry_histogram: DataFrame with columns ``retry_count`` (int) and
            ``frequency`` (int).

    Returns:
        HTML string containing a Plotly bar chart, or a fallback ``<p>``
        when the DataFrame is empty.
    """
    if retry_histogram.empty:
        return "<p>No retry data available.</p>"

    # Aggregate raw per-solve retry data into a histogram.
    # Actual schema: iteration, phase, stage, retry_level, count.
    agg = retry_histogram.groupby("retry_level", as_index=False)["count"].sum()
    retry_counts = agg["retry_level"].tolist()
    frequencies = agg["count"].tolist()

    colors = []
    for rc in retry_counts:
        if rc == 0:
            colors.append("#4A8B6F")
        elif rc <= 2:
            colors.append("#F5A623")
        else:
            colors.append("#DC4C4C")

    fig = go.Figure(
        go.Bar(
            x=retry_counts,
            y=frequencies,
            marker_color=colors,
            name="Retry Frequency",
        )
    )
    fig.update_layout(
        title="Solver Retry Distribution",
        xaxis_title="Retry Count",
        yaxis_title="Frequency",
        showlegend=False,
        height=400,
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Return True — performance tab is always shown."""
    return True


def render(data: DashboardData) -> str:
    """Return full HTML for the v2 Performance tab.

    Sections:
    - Run Summary: 6 summary metric cards
    - Training Iteration Breakdown (expanded): iteration timing + forward vs backward
    - Timing Waterfall (expanded): full per-component stacked bar
    - Solver Time Breakdown (expanded): phase breakdown + per-stage solve vs overhead
    - LP Solver Detail (expanded): backward heatmap + simplex by stage
    - Solver Overhead (collapsed): set_bounds + basis reuse
    - LP Dimensions & Scaling (collapsed): LP dimensions + scaling quality
    - Solver Efficiency (collapsed): cost per simplex iteration
    - Solver Retries (collapsed): retry histogram
    - Simulation (collapsed): per-scenario solve times
    """
    metrics_html = _build_metrics_row(data)

    breakdown_content = chart_grid(
        [
            wrap_chart(chart_iteration_timing_breakdown(data.timing)),
            wrap_chart(chart_forward_vs_backward_per_iter(data.solver_train)),
        ]
    )

    waterfall_content = chart_grid(
        [wrap_chart(chart_timing_waterfall(data.timing))],
        single=True,
    )

    solver_breakdown_content = chart_grid(
        [
            wrap_chart(chart_solver_time_breakdown_by_phase(data.solver_train)),
            wrap_chart(chart_solver_time_per_stage(data.solver_train)),
        ]
    )

    lp_detail_content = chart_grid(
        [
            wrap_chart(chart_backward_stage_heatmap(data.solver_train)),
            wrap_chart(chart_simplex_by_stage(data.solver_train)),
        ]
    )

    overhead_content = chart_grid(
        [
            wrap_chart(chart_set_bounds_by_stage(data.solver_train)),
            wrap_chart(chart_basis_reuse(data.solver_train)),
        ]
    )

    dimensions_content = chart_grid(
        [
            wrap_chart(chart_lp_dimensions(data.scaling_report)),
            wrap_chart(chart_scaling_quality(data.scaling_report)),
        ]
    )

    efficiency_content = chart_grid(
        [wrap_chart(chart_cost_per_simplex_iter(data.solver_train))],
        single=True,
    )

    retry_content = _chart_retry_histogram(data.retry_histogram)

    simulation_content = chart_grid(
        [wrap_chart(chart_simulation_scenario_times(data.solver_sim))],
        single=True,
    )

    return (
        section_title("Run Summary")
        + metrics_html
        + collapsible_section(
            title="Training Iteration Breakdown",
            content=breakdown_content,
            default_collapsed=False,
        )
        + collapsible_section(
            title="Timing Waterfall",
            content=waterfall_content,
            default_collapsed=False,
        )
        + collapsible_section(
            title="Solver Time Breakdown",
            content=solver_breakdown_content,
            default_collapsed=False,
        )
        + collapsible_section(
            title="LP Solver Detail",
            content=lp_detail_content,
            default_collapsed=False,
        )
        + collapsible_section(
            title="Solver Overhead",
            content=overhead_content,
            default_collapsed=True,
        )
        + collapsible_section(
            title="LP Dimensions & Scaling",
            content=dimensions_content,
            default_collapsed=True,
        )
        + collapsible_section(
            title="Solver Efficiency",
            content=efficiency_content,
            default_collapsed=True,
        )
        + collapsible_section(
            title="Solver Retries",
            content=retry_content,
            default_collapsed=True,
        )
        + collapsible_section(
            title="Simulation",
            content=simulation_content,
            default_collapsed=True,
        )
    )
