"""v2 Overview tab — executive summary landing page for the Cobre dashboard.

Displays run identity, run status, key metric cards, cost breakdown
(horizontal stacked bar + summary table), and two quick-look mini charts
(training convergence, generation mix).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from cobre_bridge.dashboard.chart_helpers import (
    COST_GROUP_COLORS,
    compute_cost_summary,
    make_chart_card,
)
from cobre_bridge.dashboard.data import _stage_avg_mw
from cobre_bridge.ui.html import (
    chart_grid,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    stage_x_labels,
)
from cobre_bridge.ui.theme import COLORS, GENERATION_COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

TAB_ID = "tab-v2-overview"
TAB_LABEL = "Overview"
TAB_ORDER = 0


def _format_duration(elapsed_seconds: object) -> str:
    """Format elapsed seconds as a human-readable duration string.

    Args:
        elapsed_seconds: Numeric seconds value, or ``None``.

    Returns:
        A string like ``"4h 12min"``, or ``"N/A"`` when *elapsed_seconds* is
        ``None`` or cannot be converted to an integer.
    """
    if elapsed_seconds is None:
        return "N/A"
    try:
        total = int(elapsed_seconds)
    except (ValueError, TypeError):
        return "N/A"
    hours = total // 3600
    mins = (total % 3600) // 60
    return f"{hours}h {mins}min"


def _run_identity_strip(data: DashboardData) -> str:
    """Build the run identity row HTML."""
    version = data.metadata.get("version", "N/A")
    discount_pct = data.discount_rate * 100.0

    run_date = (
        data.training_manifest.get("start_time", "N/A")
        if data.training_manifest
        else "N/A"
    )
    elapsed = (
        data.training_manifest.get("elapsed_seconds")
        if data.training_manifest
        else None
    )
    duration_str = _format_duration(elapsed)

    n_hydros = len(data.hydro_meta)
    n_thermals = len(data.thermal_meta)
    n_buses = len(data.non_fictitious_bus_ids)
    n_lines = len(data.line_meta)

    return (
        '<div class="run-identity-strip" style="'
        "display:flex;gap:2rem;flex-wrap:wrap;padding:0.75rem 1rem;"
        "background:var(--surface,#F8FAFC);border-radius:6px;"
        'margin-bottom:1rem;font-size:0.875rem;">'
        f"<span><strong>Case:</strong> {data.case_name}</span>"
        f"<span><strong>Scenarios:</strong> {data.n_scenarios}</span>"
        f"<span><strong>Stages:</strong> {data.n_stages}</span>"
        f"<span><strong>Discount Rate:</strong> {discount_pct:.1f}%</span>"
        f"<span><strong>Solver Version:</strong> {version}</span>"
        f"<span><strong>Run Date:</strong> {run_date}</span>"
        f"<span><strong>Duration:</strong> {duration_str}</span>"
        f"<span><strong>Hydros:</strong> {n_hydros}</span>"
        f"<span><strong>Thermals:</strong> {n_thermals}</span>"
        f"<span><strong>Buses:</strong> {n_buses}</span>"
        f"<span><strong>Lines:</strong> {n_lines}</span>"
        "</div>"
    )


def _run_status_strip(data: DashboardData) -> str:
    """Build the run status row HTML."""
    if not data.training_manifest:
        return (
            '<div class="run-status-strip" style="'
            "padding:0.5rem 1rem;background:#FEF3C7;border-radius:6px;"
            'margin-bottom:1rem;font-size:0.875rem;">'
            "No training manifest available"
            "</div>"
        )
    termination = data.training_manifest.get("termination_reason", "N/A")
    n_iterations = len(data.conv)
    policy_states = data.training_manifest.get("policy_states", "N/A")

    if not data.conv.empty:
        lb_str = f"{float(data.conv.iloc[-1]['lower_bound']):,.2f}"
        active_cuts = int(data.conv.iloc[-1]["cuts_active"])
        total_cuts = int(data.conv["cuts_added"].sum())
        cuts_fields = (
            f"<span><strong>Lower bound:</strong> {lb_str}</span>"
            f"<span><strong>Total cuts:</strong> {total_cuts}</span>"
            f"<span><strong>Active cuts:</strong> {active_cuts}</span>"
        )
    else:
        cuts_fields = ""

    return (
        '<div class="run-status-strip" style="'
        "display:flex;gap:2rem;flex-wrap:wrap;padding:0.5rem 1rem;"
        "background:var(--surface,#F0FDF4);border-radius:6px;"
        'margin-bottom:1rem;font-size:0.875rem;">'
        f"<span><strong>Termination:</strong> {termination}</span>"
        f"<span><strong>Training Iterations:</strong> {n_iterations}</span>"
        + cuts_fields
        + f"<span><strong>Policy states:</strong> {policy_states}</span>"
        "</div>"
    )


def _compute_gen_gwh(lf: pl.LazyFrame) -> float:
    """Extract mean total generation GWh from a simulation LazyFrame.

    Groups by scenario_id, sums generation_mwh, then averages across
    scenarios.  Returns 0.0 when the LazyFrame is empty or the column
    is absent.
    """
    try:
        result = (
            lf.group_by("scenario_id")
            .agg(pl.col("generation_mwh").sum())
            .select(pl.col("generation_mwh").mean())
            .collect(engine="streaming")
        )
        if result.height == 0:
            return 0.0
        value = result["generation_mwh"][0]
        return float(value) / 1e3 if value is not None else 0.0
    except (ValueError, TypeError, KeyError):
        return 0.0


def _build_cost_table(summary_df: pd.DataFrame) -> str:
    """Return a ``<table class="data-table">`` HTML string from a cost summary.

    Args:
        summary_df: DataFrame with columns
            ``["group", "mean", "std", "p10", "p90", "pct"]`` as returned
            by :func:`~cobre_bridge.dashboard.chart_helpers.compute_cost_summary`.

    Returns:
        An HTML string containing a complete ``<table>`` element.
    """
    if summary_df.empty:
        return "<p>No cost data available.</p>"

    headers = ("Group", "Mean", "Std", "P10", "P90", "% of Total")
    header_cells = "".join(f"<th>{h}</th>" for h in headers)
    rows = [
        f"<tr>"
        f"<td>{row['group']}</td>"
        f"<td>{row['mean']:,.0f}</td>"
        f"<td>{row['std']:,.0f}</td>"
        f"<td>{row['p10']:,.0f}</td>"
        f"<td>{row['p90']:,.0f}</td>"
        f"<td>{row['pct']:.1f}%</td>"
        f"</tr>"
        for _, row in summary_df.iterrows()
    ]

    return (
        '<table class="data-table" style="width:100%;border-collapse:collapse;">'
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _chart_cost_bar(summary_df: pd.DataFrame) -> go.Figure:
    """Build a horizontal stacked bar figure from a cost summary DataFrame.

    One trace per cost group, stacked horizontally.

    Args:
        summary_df: DataFrame with columns
            ``["group", "mean", ...]`` as returned by
            :func:`~cobre_bridge.dashboard.chart_helpers.compute_cost_summary`.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()
    for _, row in summary_df.iterrows():
        group = str(row["group"])
        color = COST_GROUP_COLORS.get(group, "#6B7280")
        fig.add_trace(
            go.Bar(
                x=[float(row["mean"])],
                y=["NPV Cost"],
                name=group,
                orientation="h",
                marker_color=color,
            )
        )
    fig.update_layout(barmode="stack")
    return fig


def _chart_training_mini(conv: pd.DataFrame) -> go.Figure | None:
    """Build a simplified convergence mini-chart with lower and upper bound lines.

    Args:
        conv: Convergence DataFrame with columns ``iteration``,
            ``lower_bound``, ``upper_bound_mean``.

    Returns:
        A :class:`plotly.graph_objects.Figure`, or ``None`` when *conv* is
        empty or the required columns are absent.
    """
    required = {"iteration", "lower_bound", "upper_bound_mean"}
    if conv.empty or not required.issubset(conv.columns):
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["lower_bound"],
            name="Lower Bound",
            mode="lines",
            line={"color": COLORS["lower_bound"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["upper_bound_mean"],
            name="Upper Bound",
            mode="lines",
            line={"color": COLORS["upper_bound"], "width": 2},
        )
    )
    fig.update_layout(
        xaxis_title="Iteration",
        yaxis_title="Cost",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    return fig


def _chart_gen_mix(data: DashboardData) -> go.Figure | None:
    """Build a stacked area chart of mean generation MW per stage.

    Computes stage-average MW for hydro, thermal, and NCS using
    :func:`~cobre_bridge.dashboard.data._stage_avg_mw` and renders them
    as stacked area traces.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A :class:`plotly.graph_objects.Figure`, or ``None`` when all three
        generation sources yield no data.
    """
    if not data.stage_hours:
        return None

    sources: list[tuple[str, pl.LazyFrame, str]] = [
        ("Hydro", data.hydros_lf, GENERATION_COLORS["hydro"]),
        ("Thermal", data.thermals_lf, GENERATION_COLORS["thermal"]),
        ("NCS", data.ncs_lf, GENERATION_COLORS["ncs"]),
    ]

    has_data = False
    fig = go.Figure()

    for label, lf, color in sources:
        try:
            stage_mw: dict[int, float] | object = _stage_avg_mw(
                lf, "generation_mwh", data.stage_hours, []
            )
        except (ValueError, TypeError, KeyError):
            continue
        if not isinstance(stage_mw, dict) or not stage_mw:
            continue

        stages = sorted(stage_mw.keys())
        xlabels = stage_x_labels(stages, data.stage_labels)
        yvals = [stage_mw[s] for s in stages]
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=yvals,
                name=label,
                stackgroup="gen",
                mode="lines",
                line={"color": color},
            )
        )
        has_data = True

    if not has_data:
        return None

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Average MW",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
    )
    return fig


def _render_section_c(data: DashboardData) -> str:
    """Render Section C — four metric cards."""
    # Expected cost (NPV)
    if data.costs.empty:
        cost_value_str = "N/A"
        sparkline: list[float] | None = None
    else:
        summary = compute_cost_summary(data.costs, data.discount_rate)
        if summary.empty:
            cost_value_str = "N/A"
            sparkline = None
        else:
            total_npv = float(summary["mean"].sum())
            cost_value_str = f"{total_npv:,.0f}"
            # Per-stage mean total cost as sparkline data
            try:
                sparkline = (
                    data.costs.groupby("stage_id")
                    .sum(numeric_only=True)
                    .sum(axis=1)
                    .sort_index()
                    .tolist()
                )
            except (ValueError, TypeError, KeyError):
                sparkline = None

    hydro_gwh = _compute_gen_gwh(data.hydros_lf)
    thermal_gwh = _compute_gen_gwh(data.thermals_lf)
    ncs_gwh = _compute_gen_gwh(data.ncs_lf)

    cards = [
        metric_card(
            cost_value_str,
            "Expected Cost (NPV)",
            color=COST_GROUP_COLORS["Thermal"],
            sparkline_values=sparkline,
        ),
        metric_card(
            f"{hydro_gwh:,.0f} GWh",
            "Total Hydro GWh",
            color=GENERATION_COLORS["hydro"],
        ),
        metric_card(
            f"{thermal_gwh:,.0f} GWh",
            "Total Thermal GWh",
            color=GENERATION_COLORS["thermal"],
        ),
        metric_card(
            f"{ncs_gwh:,.0f} GWh",
            "Total NCS GWh",
            color=GENERATION_COLORS["ncs"],
        ),
    ]
    return section_title("Key Metrics") + metrics_grid(cards)


def _render_section_d(data: DashboardData) -> str:
    """Render Section D — cost breakdown bar chart and summary table."""
    if data.costs.empty:
        return section_title("Cost Breakdown") + "<p>No cost data available.</p>"

    summary = compute_cost_summary(data.costs, data.discount_rate)
    if summary.empty:
        return section_title("Cost Breakdown") + "<p>No cost data available.</p>"

    bar_fig = _chart_cost_bar(summary)
    bar_html = make_chart_card(bar_fig, "NPV Cost by Group", "v2-cost-bar-chart")
    table_html = wrap_chart(_build_cost_table(summary))
    return section_title("Cost Breakdown") + chart_grid([bar_html, table_html])


def _render_section_e(data: DashboardData) -> str:
    """Render Section E — quick-look mini charts."""
    # Training Bounds mini
    training_fig = _chart_training_mini(data.conv)
    if training_fig is None:
        training_html = wrap_chart("<p>No convergence data.</p>")
    else:
        training_html = make_chart_card(
            training_fig,
            "Training Convergence",
            "v2-training-mini-chart",
            height=300,
        )
        deep_link = (
            '<p style="font-size:0.8rem;margin-top:0.25rem;">'
            '<a href="#" onclick="showTab(\'tab-v2-training\','
            " document.querySelector('nav button:nth-child(2)'));"
            'return false;">Full training analysis \u2192</a>'
            "</p>"
        )
        training_html = training_html + deep_link

    # Generation Mix mini
    gen_fig = _chart_gen_mix(data)
    if gen_fig is None:
        gen_html = wrap_chart("<p>No generation data.</p>")
    else:
        gen_html = make_chart_card(
            gen_fig,
            "Generation Mix",
            "v2-gen-mix-chart",
            height=300,
        )
        gen_deep_link = (
            '<p style="font-size:0.8rem;margin-top:0.25rem;">'
            '<a href="#" onclick="showTab(\'tab-v2-energy-balance\','
            " document.querySelector('nav button:nth-child(4)'));"
            'return false;">Energy balance details \u2192</a>'
            "</p>"
        )
        gen_html = gen_html + gen_deep_link

    return section_title("Quick Look") + chart_grid([training_html, gen_html])


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Overview tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Overview tab content area."""
    return (
        _run_identity_strip(data)
        + _run_status_strip(data)
        + _render_section_c(data)
        + _render_section_d(data)
        + _render_section_e(data)
    )
