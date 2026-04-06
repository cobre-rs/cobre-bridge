"""v2 Costs tab — NPV financial analysis and temporal evolution of simulation costs.

Displays a 4-card NPV metrics row (total expected cost, immediate cost,
thermal cost, deficit cost), a horizontal stacked cost breakdown bar chart,
and a companion summary table with component/mean/std/p10/p90/percentage
columns.

Ticket-015 extends this module with four temporal evolution sections:
  D. Cost Composition by Stage — stacked area, undiscounted.
  E. Cost Category Trends — line chart with p10-p90 bands per group.
  F. Spot Price by Bus — faceted mean+p50+band, block-hours weighted.
  G. Violation Costs — conditional horizontal bar chart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.chart_helpers import (
    COST_GROUP_COLORS,
    COST_GROUPS,
    add_mean_p50_band,
    compute_cost_summary,
    compute_npv_costs,
    compute_percentiles,
    group_costs,
    make_chart_card,
)
from cobre_bridge.ui.html import (
    chart_grid,
    collapsible_section,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    stage_x_labels,
)
from cobre_bridge.ui.theme import BUS_COLORS, COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

TAB_ID = "tab-v2-costs"
TAB_LABEL = "Costs"
TAB_ORDER = 40

# Non-cost metadata columns that are excluded from cost aggregations.
_NON_COST_COLS: frozenset[str] = frozenset({"scenario_id", "stage_id", "block_id"})

# All known violation cost column names from the costs DataFrame spec.
_VIOLATION_COLS: frozenset[str] = frozenset(
    {
        "storage_violation_cost",
        "filling_target_cost",
        "hydro_violation_cost",
        "outflow_violation_below_cost",
        "outflow_violation_above_cost",
        "turbined_violation_cost",
        "generation_violation_cost",
        "evaporation_violation_cost",
        "withdrawal_violation_cost",
        "inflow_penalty_cost",
        "generic_violation_cost",
        "fpha_turbined_cost",
        "generic_constraint_violation_cost",
    }
)


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Costs tab always renders."""
    return True


def _compute_npv_metric(data: DashboardData, column_name: str) -> float:
    """Compute the mean NPV of a single cost column across scenarios.

    Applies discount factors via :func:`compute_npv_costs`, groups by
    ``scenario_id``, sums *column_name* per scenario, then returns the mean
    across scenarios.

    Args:
        data: Dashboard data instance containing ``costs`` and
            ``discount_rate``.
        column_name: Name of the cost column to aggregate.

    Returns:
        Mean NPV value as a float.  Returns ``0.0`` when the column is
        absent, the DataFrame is empty, or any aggregation fails.
    """
    if data.costs.empty:
        return 0.0
    if column_name not in data.costs.columns:
        return 0.0

    try:
        discounted = compute_npv_costs(data.costs, data.discount_rate)
        if "scenario_id" not in discounted.columns:
            total = float(discounted[column_name].sum())
            return total
        per_scenario = discounted.groupby("scenario_id")[column_name].sum()
        return float(per_scenario.mean())
    except (ValueError, TypeError, KeyError):
        return 0.0


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 4-card NPV metrics row HTML.

    Cards (in order):
    1. Expected Cost (NPV) — total NPV from ``compute_cost_summary``.
    2. Immediate Cost (NPV) — NPV of ``immediate_cost`` column.
    3. Thermal Cost (NPV) — NPV of ``thermal_cost`` column.
    4. Deficit Cost (NPV) — NPV of ``deficit_cost`` column.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string containing a ``<div class="metrics-grid">`` with 4 cards.
    """
    if data.costs.empty:
        na_card = metric_card(
            "N/A", "Expected Cost (NPV)", color=COST_GROUP_COLORS["Thermal"]
        )
        return metrics_grid([na_card] * 4)

    # Card 1: Total expected cost NPV via compute_cost_summary
    try:
        summary = compute_cost_summary(data.costs, data.discount_rate)
        total_npv_str = (
            f"{float(summary['mean'].sum()):,.0f}" if not summary.empty else "N/A"
        )
    except (ValueError, TypeError, KeyError):
        total_npv_str = "N/A"

    # Cards 2-4: per-column NPV metrics
    def _fmt(col: str) -> str:
        val = _compute_npv_metric(data, col)
        return f"{val:,.0f}" if col in data.costs.columns else "N/A"

    cards = [
        metric_card(
            total_npv_str,
            "Expected Cost (NPV)",
            color=COST_GROUP_COLORS["Thermal"],
        ),
        metric_card(
            _fmt("immediate_cost"),
            "Immediate Cost (NPV)",
            color=COLORS["future_cost"],
        ),
        metric_card(
            _fmt("thermal_cost"),
            "Thermal Cost (NPV)",
            color=COLORS["thermal"],
        ),
        metric_card(
            _fmt("deficit_cost"),
            "Deficit Cost (NPV)",
            color=COLORS["deficit"],
        ),
    ]
    return metrics_grid(cards)


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


def _build_cost_table(summary_df: pd.DataFrame) -> str:
    """Return a ``<table class="data-table">`` HTML string from a cost summary.

    Args:
        summary_df: DataFrame with columns
            ``["group", "mean", "std", "p10", "p90", "pct"]`` as returned
            by :func:`~cobre_bridge.dashboard.chart_helpers.compute_cost_summary`.

    Returns:
        An HTML string containing a complete ``<table>`` element, or a
        fallback ``<p>`` when *summary_df* is empty.
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


def _render_npv_section(data: DashboardData) -> str:
    """Render the full NPV perspective section.

    Computes the cost summary, then renders:
    - Metrics row (4 cards)
    - Horizontal stacked bar chart card
    - Summary table card

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string for the NPV section body (metrics grid + chart grid).
    """
    metrics_html = _build_metrics_row(data)

    if data.costs.empty:
        chart_html = wrap_chart("<p>No cost data available.</p>")
        table_html = wrap_chart("<p>No cost data available.</p>")
        return metrics_html + chart_grid([chart_html, table_html])

    summary = compute_cost_summary(data.costs, data.discount_rate)
    if summary.empty:
        chart_html = wrap_chart("<p>No cost data available.</p>")
        table_html = wrap_chart("<p>No cost data available.</p>")
        return metrics_html + chart_grid([chart_html, table_html])

    bar_fig = _chart_cost_bar(summary)
    bar_html = make_chart_card(bar_fig, "NPV Cost by Group", "v2-costs-bar-chart")
    table_html = wrap_chart(_build_cost_table(summary))
    return metrics_html + chart_grid([bar_html, table_html])


# ---------------------------------------------------------------------------
# Section D — Cost Composition by Stage (stacked area, undiscounted)
# ---------------------------------------------------------------------------


def _render_cost_composition(data: DashboardData) -> str:
    """Render stacked area chart of undiscounted cost composition per stage.

    Groups individual cost component columns into logical categories using
    :func:`~cobre_bridge.dashboard.chart_helpers.group_costs`, computes the
    mean across scenarios per stage (summing blocks first), then builds a
    stacked area chart with one area per group.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart card.
    """
    if data.costs.empty:
        return collapsible_section(
            "Cost Composition by Stage",
            "<p>No cost data available.</p>",
            section_id="v2-costs-composition-section",
            default_collapsed=False,
        )

    cost_cols = [c for c in data.costs.columns if c not in _NON_COST_COLS]
    if not cost_cols:
        return collapsible_section(
            "Cost Composition by Stage",
            "<p>No cost data available.</p>",
            section_id="v2-costs-composition-section",
            default_collapsed=False,
        )

    # Sum blocks per (scenario_id, stage_id), then group by cost categories
    by_stage_scen = (
        data.costs.groupby(["scenario_id", "stage_id"])[cost_cols].sum().reset_index()
    )
    grouped = group_costs(by_stage_scen, cost_cols)
    group_cols = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]

    # Mean across scenarios per stage
    mean_by_stage = grouped.groupby("stage_id")[group_cols].mean().reset_index()
    stages = sorted(mean_by_stage["stage_id"].tolist())
    xlabels = stage_x_labels(stages, data.stage_labels)
    mean_by_stage["_x"] = mean_by_stage["stage_id"].map(dict(zip(stages, xlabels)))

    fig = go.Figure()
    for group_name in COST_GROUPS:
        if group_name not in mean_by_stage.columns:
            continue
        color = COST_GROUP_COLORS.get(group_name, "#6B7280")
        fig.add_trace(
            go.Scatter(
                x=mean_by_stage["_x"],
                y=mean_by_stage[group_name],
                name=group_name,
                stackgroup="costs",
                mode="lines",
                line=dict(color=color, width=0.5),
                fillcolor=color,
            )
        )
    # Include "Other" if it exists and has non-zero values
    if "Other" in mean_by_stage.columns and mean_by_stage["Other"].sum() > 0:
        if "Other" not in [t.name for t in fig.data]:
            fig.add_trace(
                go.Scatter(
                    x=mean_by_stage["_x"],
                    y=mean_by_stage["Other"],
                    name="Other",
                    stackgroup="costs",
                    mode="lines",
                    line=dict(color=COST_GROUP_COLORS["Other"], width=0.5),
                    fillcolor=COST_GROUP_COLORS["Other"],
                )
            )

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Cost (R$)",
        legend=_LEGEND,
        margin=_MARGIN,
    )

    chart_html = make_chart_card(
        fig,
        "Cost Composition by Stage (undiscounted mean across scenarios)",
        "v2-costs-composition-chart",
        height=420,
    )
    return collapsible_section(
        "Cost Composition by Stage",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-composition-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section E — Cost Category Trends (line chart with p10-p90 bands)
# ---------------------------------------------------------------------------


def _render_category_evolution(data: DashboardData) -> str:
    """Render line chart of cost category trends with p10-p90 bands per stage.

    For each cost group, computes per-(scenario_id, stage_id) sum (across
    blocks), then uses :func:`compute_percentiles` and
    :func:`add_mean_p50_band` to show the distribution across scenarios.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart card.
    """
    if data.costs.empty:
        return collapsible_section(
            "Cost Category Trends",
            "<p>No cost data available.</p>",
            section_id="v2-costs-trends-section",
            default_collapsed=True,
        )

    cost_cols = [c for c in data.costs.columns if c not in _NON_COST_COLS]
    if not cost_cols:
        return collapsible_section(
            "Cost Category Trends",
            "<p>No cost data available.</p>",
            section_id="v2-costs-trends-section",
            default_collapsed=True,
        )

    # Sum blocks per (scenario_id, stage_id), then group by cost categories
    by_stage_scen = (
        data.costs.groupby(["scenario_id", "stage_id"])[cost_cols].sum().reset_index()
    )
    grouped = group_costs(by_stage_scen, cost_cols)
    group_cols = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]

    stages = sorted(grouped["stage_id"].unique().tolist())
    xlabels = stage_x_labels(stages, data.stage_labels)
    stage_to_x = dict(zip(stages, xlabels))

    fig = go.Figure()
    for group_name in group_cols:
        color = COST_GROUP_COLORS.get(group_name, "#6B7280")
        sub = grouped[["scenario_id", "stage_id", group_name]].copy()
        pct = compute_percentiles(sub, ["stage_id"], group_name)
        if pct.empty:
            continue
        pct["_x"] = pct["stage_id"].map(stage_to_x)
        add_mean_p50_band(fig, pct, "_x", group_name, color)

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Cost (R$)",
        legend=_LEGEND,
        margin=_MARGIN,
    )

    chart_html = make_chart_card(
        fig,
        "Cost Category Trends (undiscounted, p10/mean/p50/p90 across scenarios)",
        "v2-costs-trends-chart",
        height=420,
    )
    return collapsible_section(
        "Cost Category Trends",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-trends-section",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Section F — Spot Price by Bus (faceted, block-hours weighted)
# ---------------------------------------------------------------------------


def _render_spot_price(data: DashboardData) -> str:
    """Render faceted spot price chart: one subplot per non-fictitious bus.

    Spot price is weighted by block-hours before computing percentiles:
    ``sum(spot_price * _bh) / sum(_bh)`` per ``(scenario_id, stage_id, bus_id)``.
    Then :func:`compute_percentiles` + :func:`add_mean_p50_band` per bus subplot.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart card.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)

    # Guard: no buses or no buses_lf data
    if not bus_ids:
        return collapsible_section(
            "Spot Price by Bus",
            "<p>No spot price data.</p>",
            section_id="v2-costs-spot-price-section",
            default_collapsed=True,
        )

    try:
        weighted = (
            data.buses_lf.filter(pl.col("bus_id").is_in(bus_ids))
            .join(data.bh_df.lazy(), on=["stage_id", "block_id"])
            .with_columns((pl.col("spot_price") * pl.col("_bh")).alias("_sp_bh"))
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg((pl.col("_sp_bh").sum() / pl.col("_bh").sum()).alias("w_spot"))
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError, pl.exceptions.SchemaError):
        return collapsible_section(
            "Spot Price by Bus",
            "<p>No spot price data.</p>",
            section_id="v2-costs-spot-price-section",
            default_collapsed=True,
        )

    if weighted.is_empty():
        return collapsible_section(
            "Spot Price by Bus",
            "<p>No spot price data.</p>",
            section_id="v2-costs-spot-price-section",
            default_collapsed=True,
        )

    stages = sorted(weighted["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, data.stage_labels)
    stage_to_x = dict(zip(stages, xlabels))

    n_buses = len(bus_ids)
    n_cols = 2
    n_rows = (n_buses + n_cols - 1) // n_cols
    subplot_titles = [data.bus_names.get(bid, str(bid)) for bid in bus_ids]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=max(0.06, 0.35 / max(n_rows, 1)),
        horizontal_spacing=0.10,
    )

    for idx, bus_id in enumerate(bus_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = BUS_COLORS[idx % len(BUS_COLORS)]
        bus_name = data.bus_names.get(bus_id, str(bus_id))

        sub_pl = weighted.filter(pl.col("bus_id") == bus_id).select(
            ["scenario_id", "stage_id", "w_spot"]
        )
        if sub_pl.is_empty():
            continue
        sub = sub_pl.to_pandas()
        pct = compute_percentiles(sub, ["stage_id"], "w_spot")
        if pct.empty:
            continue
        pct["_x"] = pct["stage_id"].map(stage_to_x)
        add_mean_p50_band(fig, pct, "_x", bus_name, color, row=row, col=col)

    fig.update_layout(
        height=350 * n_rows + 60,
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    for ax in fig.layout:
        if str(ax).startswith("yaxis"):
            fig.layout[ax].title = "R$/MWh"  # type: ignore[index]

    chart_html = make_chart_card(
        fig,
        "Spot Price by Bus (block-hours weighted, p10/mean/p50/p90)",
        "v2-costs-spot-price-chart",
        height=350 * n_rows + 60,
    )
    return collapsible_section(
        "Spot Price by Bus",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-spot-price-section",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Section G — Violation Costs (conditional)
# ---------------------------------------------------------------------------


def _render_violations(data: DashboardData) -> str:
    """Render a horizontal bar chart of violation costs per category.

    Only rendered when at least one known violation cost column has non-zero
    values.  When all violation costs are zero, returns a ``<p>`` fallback.

    Violation cost columns are identified by intersecting ``data.costs.columns``
    with the known set :data:`_VIOLATION_COLS` plus any column whose name ends
    in ``"_violation_cost"``.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string: a ``collapsible_section`` wrapping the chart (or a
        ``<p>`` fallback) when data is missing or all violations are zero.
    """
    if data.costs.empty:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Identify violation columns present in the DataFrame
    viol_cols = [
        c
        for c in data.costs.columns
        if c in _VIOLATION_COLS or c.endswith("_violation_cost")
    ]

    if not viol_cols:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Check whether any violation column has non-zero values
    subset = data.costs[viol_cols]
    if subset.values.sum() == 0:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Compute mean total violation cost per category across all stages/blocks,
    # averaged across scenarios.
    total_per_scen: dict[str, float] = {}
    for col in viol_cols:
        if "scenario_id" in data.costs.columns:
            per_scen = data.costs.groupby("scenario_id")[col].sum()
            total_per_scen[col] = float(per_scen.mean())
        else:
            total_per_scen[col] = float(data.costs[col].sum())

    # Remove zero-valued categories
    total_per_scen = {k: v for k, v in total_per_scen.items() if v > 0}
    if not total_per_scen:
        return collapsible_section(
            "Violation Costs",
            "<p>No violation costs recorded.</p>",
            section_id="v2-costs-violations-section",
            default_collapsed=True,
        )

    # Sort descending
    sorted_items = sorted(total_per_scen.items(), key=lambda kv: kv[1], reverse=True)
    categories = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=categories,
            orientation="h",
            marker_color=COST_GROUP_COLORS["Violations"],
        )
    )
    fig.update_layout(
        xaxis_title="Mean Cost (R$)",
        yaxis=dict(autorange="reversed"),
        legend=_LEGEND,
        margin=_MARGIN,
    )

    chart_html = make_chart_card(
        fig,
        "Violation Costs by Category (mean across scenarios)",
        "v2-costs-violations-chart",
        height=max(300, 40 * len(categories) + 120),
    )
    return collapsible_section(
        "Violation Costs",
        chart_grid([chart_html], single=True),
        section_id="v2-costs-violations-section",
        default_collapsed=True,
    )


# ---------------------------------------------------------------------------
# Public render
# ---------------------------------------------------------------------------


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Costs tab content area.

    Renders five sections in order:
    - NPV Cost Analysis (metrics + bar chart + table) — from ticket-014.
    - Cost Composition by Stage (stacked area, undiscounted) — Section D.
    - Cost Category Trends (line + bands per group) — Section E.
    - Spot Price by Bus (faceted subplots, block-hours weighted) — Section F.
    - Violation Costs (conditional horizontal bar) — Section G.

    Args:
        data: Dashboard data instance.

    Returns:
        HTML string for the entire tab content.
    """
    return (
        section_title("NPV Cost Analysis")
        + _render_npv_section(data)
        + _render_cost_composition(data)
        + _render_category_evolution(data)
        + _render_spot_price(data)
        + _render_violations(data)
    )
