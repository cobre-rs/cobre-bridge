"""v2 Constraints tab module for the Cobre dashboard.

Displays generic constraint summary, LHS vs bound charts, bounds timeline,
violation cost timeline, and violation summary/heatmap. Only rendered when
generic constraint definitions are present in the case.

Reuses chart functions from constraints.py and wraps them in the v2 UI
framework (collapsible sections, make_chart_card, chart_grid, metric_card,
metrics_grid).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.dashboard.tabs.constraints import (
    build_constraints_summary_table,
    chart_constraint_bounds_timeline,
    chart_constraint_lhs_vs_bound,
    chart_violation_cost_timeline,
    chart_violation_heatmap,
    chart_violation_summary,
    evaluate_constraint_expressions,
)
from cobre_bridge.ui.html import (
    chart_grid,
    collapsible_section,
    metric_card,
    metrics_grid,
    section_title,
    wrap_chart,
)

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-constraints"
TAB_LABEL = "Constraints"
TAB_ORDER = 80

# ---------------------------------------------------------------------------
# Metric colors
# ---------------------------------------------------------------------------

_COLOR_TOTAL = "#4A90B8"
_COLOR_VIOLATED = "#DC4C4C"
_COLOR_COST = "#F5A623"
_COLOR_TYPES = "#4A8B6F"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_metrics_row(data: DashboardData) -> str:
    """Build the 4-card metrics row for the Constraints tab.

    Cards:
    1. Total constraints count
    2. Constraints with violations count
    3. Total violation cost (sum of generic_violation_cost across all scenarios/stages)
    4. Active constraint types (comma-joined unique type prefixes)
    """
    total_constraints = len(data.gc_constraints)

    # Count constraints that have at least one nonzero slack_value in gc_violations
    violated_count = 0
    violations_df: pd.DataFrame = data.gc_violations
    if not violations_df.empty and "constraint_id" in violations_df.columns:
        viol_ids = set(
            violations_df[violations_df["slack_value"].abs() > 1e-6][
                "constraint_id"
            ].unique()
        )
        violated_count = len(viol_ids)

    # Total violation cost
    costs_df: pd.DataFrame = data.costs
    if "generic_violation_cost" in costs_df.columns:
        total_viol_cost = float(costs_df["generic_violation_cost"].sum())
    else:
        total_viol_cost = 0.0
    cost_str = f"R$ {total_viol_cost:,.0f}"

    # Active constraint types: unique first element after splitting name on "_"
    type_prefixes: list[str] = []
    seen: set[str] = set()
    for c in data.gc_constraints:
        prefix = c["name"].split("_")[0]
        if prefix not in seen:
            seen.add(prefix)
            type_prefixes.append(prefix)
    active_types_str = ", ".join(type_prefixes) if type_prefixes else "—"

    cards = [
        metric_card(
            value=str(total_constraints),
            label="Total Constraints",
            color=_COLOR_TOTAL,
        ),
        metric_card(
            value=str(violated_count),
            label="Constraints with Violations",
            color=_COLOR_VIOLATED,
        ),
        metric_card(
            value=cost_str,
            label="Total Violation Cost",
            color=_COLOR_COST,
        ),
        metric_card(
            value=active_types_str,
            label="Active Types",
            color=_COLOR_TYPES,
        ),
    ]
    return metrics_grid(cards)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True when generic constraint definitions are present."""
    return len(data.gc_constraints) > 0


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Constraints tab content area."""
    # Evaluate LHS expressions once; passed to LHS vs Bound charts
    lhs_df = evaluate_constraint_expressions(
        data.gc_constraints, data.hydros_lf, data.exchanges_lf
    )

    # --- Section A: metrics row ---
    section_a = _build_metrics_row(data)

    # --- Section B: Constraint Summary Table ---
    summary_table = build_constraints_summary_table(
        data.gc_constraints, data.gc_bounds, data.gc_violations
    )
    section_b = section_title("Constraint Summary") + summary_table

    # --- Section C: LHS vs Bound charts (VminOP, RE, AGRINT) ---
    vminop_html = chart_constraint_lhs_vs_bound(
        data.gc_constraints,
        lhs_df,
        data.gc_bounds,
        data.stage_labels,
        ctype_filter="VminOP",
    )
    re_html = chart_constraint_lhs_vs_bound(
        data.gc_constraints,
        lhs_df,
        data.gc_bounds,
        data.stage_labels,
        ctype_filter="RE",
    )
    agrint_html = chart_constraint_lhs_vs_bound(
        data.gc_constraints,
        lhs_df,
        data.gc_bounds,
        data.stage_labels,
        ctype_filter="AGRINT",
    )
    section_c = (
        section_title("VminOP: Stored Energy vs Minimum")
        + chart_grid([wrap_chart(vminop_html)], single=True)
        + section_title("Electric Constraints (RE)")
        + chart_grid([wrap_chart(re_html)], single=True)
        + section_title("Exchange Group Constraints (AGRINT)")
        + chart_grid([wrap_chart(agrint_html)], single=True)
    )

    # --- Section D: Bounds Timeline (collapsible, default collapsed) ---
    bounds_html = chart_constraint_bounds_timeline(
        data.gc_constraints, data.gc_bounds, data.stage_labels
    )
    section_d = collapsible_section(
        title="Constraint Bounds Timeline",
        content=chart_grid([wrap_chart(bounds_html)], single=True),
        default_collapsed=True,
    )

    # --- Section E: Violation Cost Timeline ---
    viol_cost_html = chart_violation_cost_timeline(data.costs, data.stage_labels)
    section_e = collapsible_section(
        title="Violation Cost Timeline",
        content=chart_grid([wrap_chart(viol_cost_html)], single=True),
        default_collapsed=False,
    )

    # --- Section F: Violation Summary & Heatmap (collapsible, default collapsed) ---
    viol_summary_html = chart_violation_summary(data.hydros_lf, data.stage_labels)
    viol_heatmap_html = chart_violation_heatmap(
        data.hydros_lf, data.names, data.stage_labels
    )
    viol_content = chart_grid(
        [wrap_chart(viol_summary_html), wrap_chart(viol_heatmap_html)]
    )
    section_f = collapsible_section(
        title="Violation Summary & Heatmap",
        content=viol_content,
        default_collapsed=True,
    )

    return section_a + section_b + section_c + section_d + section_e + section_f
