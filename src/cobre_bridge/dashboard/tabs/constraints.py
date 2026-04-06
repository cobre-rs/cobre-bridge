"""Constraints tab module for the Cobre dashboard.

Displays generic constraint summary, LHS vs bound charts, and violation charts.
Only rendered when generic constraint definitions are present in the case.
"""

from __future__ import annotations

import re as _re
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.ui.html import section_title, wrap_chart
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    fig_to_html,
    stage_x_labels,
)
from cobre_bridge.ui.theme import COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-constraints"
TAB_LABEL = "Constraints"
TAB_ORDER = 100

# ---------------------------------------------------------------------------
# Expression parser (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------

# Matches terms like: [+/-] [coeff *] variable_type(id)
# Examples: "5.68 * hydro_storage(78)", "hydro_generation(145)", "- line_exchange(4)"
_TERM_RE = _re.compile(
    r"([+-]?\s*\d*\.?\d*)\s*\*?\s*(hydro_storage|hydro_generation|line_exchange)\((\d+)\)"
)


def _parse_expression(expr: str) -> list[tuple[float, str, int]]:
    """Parse a constraint LHS expression into (coefficient, variable_type, entity_id) terms.

    Handles:
    - Leading ``-`` with no explicit coefficient → -1.0
    - No coefficient → 1.0
    - ``0.5 * hydro_generation(47)`` → 0.5
    """
    terms: list[tuple[float, str, int]] = []
    for m in _TERM_RE.finditer(expr):
        raw_coeff = m.group(1).replace(" ", "")
        var_type = m.group(2)
        entity_id = int(m.group(3))
        if raw_coeff in ("", "+"):
            coeff = 1.0
        elif raw_coeff == "-":
            coeff = -1.0
        else:
            coeff = float(raw_coeff)
        terms.append((coeff, var_type, entity_id))
    return terms


# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def evaluate_constraint_expressions(
    constraints: list[dict],
    hydros_lf: pl.LazyFrame,
    exchanges_lf: pl.LazyFrame,
) -> pd.DataFrame:
    """Evaluate LHS of all generic constraints from simulation output.

    Variable lookups:
    - ``hydro_storage(id)``    → ``storage_final_hm3`` where hydro_id=id, block_id=0
    - ``hydro_generation(id)`` → ``generation_mw``     where hydro_id=id  (per block)
    - ``line_exchange(id)``    → ``net_flow_mw``        where line_id=id   (per block)

    Storage-only constraints produce one row per (scenario, stage) with block_id=0.
    Mixed / generation / exchange constraints produce one row per (scenario, stage, block).

    Accepts LazyFrames and only collects the specific entity IDs referenced
    in constraint expressions, keeping memory usage minimal.

    Returns DataFrame with columns:
        constraint_id, scenario_id, stage_id, block_id, lhs_value
    """
    # Parse ALL constraints to find which entity IDs are referenced
    hydro_ids_needed: set[int] = set()
    line_ids_needed: set[int] = set()
    for c in constraints:
        for _coeff, vtype, eid in _parse_expression(c["expression"]):
            if vtype.startswith("hydro"):
                hydro_ids_needed.add(eid)
            elif vtype == "line_exchange":
                line_ids_needed.add(eid)

    # Collect only the referenced entities (tiny subset of full data)
    h0_pd = (
        hydros_lf.filter(
            (pl.col("block_id") == 0) & pl.col("hydro_id").is_in(list(hydro_ids_needed))
        )
        .select(["scenario_id", "stage_id", "hydro_id", "storage_final_hm3"])
        .collect(engine="streaming")
        .to_pandas()
    )
    hg_pd = (
        hydros_lf.filter(pl.col("hydro_id").is_in(list(hydro_ids_needed)))
        .select(["scenario_id", "stage_id", "block_id", "hydro_id", "generation_mw"])
        .collect(engine="streaming")
        .to_pandas()
    )
    ex_pd = (
        (
            exchanges_lf.filter(pl.col("line_id").is_in(list(line_ids_needed)))
            .select(["scenario_id", "stage_id", "block_id", "line_id", "net_flow_mw"])
            .collect(engine="streaming")
            .to_pandas()
        )
        if line_ids_needed
        else pd.DataFrame(
            columns=["scenario_id", "stage_id", "block_id", "line_id", "net_flow_mw"]
        )
    )

    all_results: list[pd.DataFrame] = []

    for c in constraints:
        cid = c["id"]
        expr = c["expression"]
        terms = _parse_expression(expr)
        if not terms:
            continue

        var_types = {t[1] for t in terms}
        storage_only = var_types == {"hydro_storage"}

        if storage_only:
            # One LHS value per (scenario, stage) — use block_id 0
            # Start with a base frame of all (scenario, stage) combos from h0
            base = h0_pd[["scenario_id", "stage_id"]].drop_duplicates().copy()
            base["_lhs"] = 0.0
            for coeff, vtype, eid in terms:
                sub = h0_pd[h0_pd["hydro_id"] == eid][
                    ["scenario_id", "stage_id", "storage_final_hm3"]
                ].rename(columns={"storage_final_hm3": "_val"})
                merged = base.merge(sub, on=["scenario_id", "stage_id"], how="left")
                merged["_val"] = merged["_val"].fillna(0.0)
                base["_lhs"] = base["_lhs"].values + coeff * merged["_val"].values
            base["constraint_id"] = cid
            base["block_id"] = 0
            base = base.rename(columns={"_lhs": "lhs_value"})
            all_results.append(
                base[
                    [
                        "constraint_id",
                        "scenario_id",
                        "stage_id",
                        "block_id",
                        "lhs_value",
                    ]
                ]
            )
        else:
            # Per (scenario, stage, block) — need consistent block grid
            # Build base from hydro_generation block grid (always present)
            base = (
                hg_pd[["scenario_id", "stage_id", "block_id"]].drop_duplicates().copy()
            )
            base["_lhs"] = 0.0

            for coeff, vtype, eid in terms:
                if vtype == "hydro_storage":
                    # Storage is stage-level: broadcast across all blocks
                    sub = h0_pd[h0_pd["hydro_id"] == eid][
                        ["scenario_id", "stage_id", "storage_final_hm3"]
                    ].rename(columns={"storage_final_hm3": "_val"})
                    merged = base.merge(sub, on=["scenario_id", "stage_id"], how="left")
                elif vtype == "hydro_generation":
                    sub = hg_pd[hg_pd["hydro_id"] == eid][
                        ["scenario_id", "stage_id", "block_id", "generation_mw"]
                    ].rename(columns={"generation_mw": "_val"})
                    merged = base.merge(
                        sub, on=["scenario_id", "stage_id", "block_id"], how="left"
                    )
                else:  # line_exchange
                    sub = ex_pd[ex_pd["line_id"] == eid][
                        ["scenario_id", "stage_id", "block_id", "net_flow_mw"]
                    ].rename(columns={"net_flow_mw": "_val"})
                    merged = base.merge(
                        sub, on=["scenario_id", "stage_id", "block_id"], how="left"
                    )
                merged["_val"] = merged["_val"].fillna(0.0)
                base["_lhs"] = base["_lhs"].values + coeff * merged["_val"].values

            base["constraint_id"] = cid
            base = base.rename(columns={"_lhs": "lhs_value"})
            all_results.append(
                base[
                    [
                        "constraint_id",
                        "scenario_id",
                        "stage_id",
                        "block_id",
                        "lhs_value",
                    ]
                ]
            )

    if not all_results:
        return pd.DataFrame(
            columns=[
                "constraint_id",
                "scenario_id",
                "stage_id",
                "block_id",
                "lhs_value",
            ]
        )
    return pd.concat(all_results, ignore_index=True)


def build_constraints_summary_table(
    constraints: list[dict],
    gc_bounds: pd.DataFrame,
    violations_df: pd.DataFrame,
) -> str:
    """HTML summary table of all generic constraints.

    Columns: Name, Type, Sense, Active Stages, Bound Range, Slack, Penalty, Has Violations
    Rows are colour-coded by constraint type.
    """
    type_colors = {
        "VminOP": "#EEF4FB",
        "RE": "#F0FAF4",
        "AGRINT": "#FFF8EE",
    }
    rows_html: list[str] = []
    for c in constraints:
        cid = c["id"]
        name = c["name"]
        ctype = name.split("_")[0]
        sense = c["sense"]
        slack = c["slack"]
        slack_enabled = "Yes" if slack.get("enabled") else "No"
        penalty = (
            f"{slack['penalty']:,.0f}"
            if slack.get("enabled") and "penalty" in slack
            else "—"
        )
        bounds_rows = gc_bounds[gc_bounds["constraint_id"] == cid]
        active_stages = (
            int(bounds_rows["stage_id"].nunique()) if not bounds_rows.empty else 0
        )
        bmin = bounds_rows["bound"].min() if not bounds_rows.empty else 0.0
        bmax = bounds_rows["bound"].max() if not bounds_rows.empty else 0.0
        if abs(bmax - bmin) < 1e-6:
            bound_range = f"{bmin:,.1f}"
        else:
            bound_range = f"{bmin:,.1f} – {bmax:,.1f}"
        has_viol = "No"
        if not violations_df.empty and "constraint_id" in violations_df.columns:
            viol_sub = violations_df[violations_df["constraint_id"] == cid]
            if not viol_sub.empty and viol_sub["slack_value"].abs().sum() > 1e-6:
                has_viol = "Yes"
        bg = type_colors.get(ctype, "#FFFFFF")
        viol_style = (
            ' style="color:#DC4C4C;font-weight:600;"' if has_viol == "Yes" else ""
        )
        rows_html.append(
            f'<tr style="background:{bg};">'
            f"<td>{name}</td>"
            f"<td>{ctype}</td>"
            f"<td><code>{sense}</code></td>"
            f"<td style='text-align:center;'>{active_stages}</td>"
            f"<td style='text-align:right;'>{bound_range}</td>"
            f"<td style='text-align:center;'>{slack_enabled}</td>"
            f"<td style='text-align:right;'>{penalty}</td>"
            f"<td style='text-align:center;'{viol_style}>{has_viol}</td>"
            "</tr>"
        )

    legend_html = (
        '<div style="margin-top:8px;font-size:0.8rem;color:#666;">'
        '<span style="background:#EEF4FB;padding:2px 8px;margin-right:8px;">VminOP — Minimum stored energy</span>'
        '<span style="background:#F0FAF4;padding:2px 8px;margin-right:8px;">RE — Electric constraint</span>'
        '<span style="background:#FFF8EE;padding:2px 8px;">AGRINT — Exchange group constraint</span>'
        "</div>"
    )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        "<th>Name</th><th>Type</th><th>Sense</th>"
        "<th>Active Stages</th><th>Bound Range</th>"
        "<th>Slack</th><th>Penalty (R$/unit)</th><th>Has Violations</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>" + legend_html
    )


def chart_constraint_lhs_vs_bound(
    constraints: list[dict],
    lhs_df: pd.DataFrame,
    gc_bounds: pd.DataFrame,
    stage_labels: dict[int, str],
    ctype_filter: str = "VminOP",
) -> str:
    """Subplots — one per constraint of given type.

    Shows LHS p50 median across scenarios with p10-p90 band vs RHS bound.
    For VminOP (storage energy): y-axis label is MWh equivalent.
    For RE (generation): MW.
    """
    target = [c for c in constraints if c["name"].startswith(ctype_filter + "_")]
    if not target or lhs_df.empty:
        return f"<p>No {ctype_filter} constraints or no LHS data available.</p>"

    n = len(target)
    cols = min(n, 2)
    rows_count = (n + cols - 1) // cols

    subtitles = [c["name"] for c in target]
    fig = make_subplots(
        rows=rows_count,
        cols=cols,
        subplot_titles=subtitles,
        vertical_spacing=max(0.06, 0.35 / max(rows_count, 1)),
        horizontal_spacing=0.1,
    )

    palette_lhs = "#4A90B8"
    palette_bound = "#DC4C4C"

    for idx, c in enumerate(target):
        row = idx // cols + 1
        col = idx % cols + 1
        show_legend = idx == 0
        cid = c["id"]

        sub = lhs_df[lhs_df["constraint_id"] == cid].copy()
        if sub.empty:
            continue

        # For storage constraints, LHS is stage-level (block_id=0). For others average
        # over blocks to get a single stage value for the chart.
        pcts = (
            sub.groupby(["scenario_id", "stage_id"])["lhs_value"]
            .mean()
            .reset_index()
            .groupby("stage_id")["lhs_value"]
            .quantile([0.1, 0.5, 0.9])
            .unstack(level=-1)
            .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
        )
        stages = sorted(pcts.index)
        xlabels = stage_x_labels(stages, stage_labels)

        p10 = [pcts["p10"].get(s, 0) for s in stages]
        p50 = [pcts["p50"].get(s, 0) for s in stages]
        p90 = [pcts["p90"].get(s, 0) for s in stages]

        # Bound: for storage it is stage-level (no block). For RE/AGRINT take block 0.
        bounds_c = gc_bounds[gc_bounds["constraint_id"] == cid]
        if not bounds_c.empty:
            if bounds_c["block_id"].isna().all():
                b_by_stage = bounds_c.set_index("stage_id")["bound"]
            else:
                b_by_stage = bounds_c[bounds_c["block_id"] == 0.0].set_index(
                    "stage_id"
                )["bound"]
            bound_vals = [float(b_by_stage.get(s, float("nan"))) for s in stages]
        else:
            bound_vals = [float("nan")] * len(stages)

        # P10-P90 band
        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(74,144,184,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                name="P10–P90",
                legendgroup="band",
                showlegend=show_legend,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p50,
                name="LHS Median (P50)",
                line={"color": palette_lhs, "width": 2},
                legendgroup="lhs",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=bound_vals,
                name="Bound (RHS)",
                line={"color": palette_bound, "width": 1.5, "dash": "dash"},
                legendgroup="bound",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

    type_labels = {
        "VminOP": "Energy Equivalent (MWh)",
        "RE": "Generation (MW)",
        "AGRINT": "Flow (MW)",
    }
    type_labels.get(ctype_filter, "Value")
    sense_desc = {
        "VminOP": "LHS ≥ Bound (minimum energy)",
        "RE": "LHS ≤ Bound (upper limit)",
        "AGRINT": "LHS ≤ Bound (exchange limit)",
    }
    title = (
        f"{ctype_filter} Constraints — {sense_desc.get(ctype_filter, 'LHS vs Bound')}"
    )

    fig.update_layout(
        title=title,
        height=max(360, rows_count * 320),
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig_to_html(fig)


def chart_constraint_bounds_timeline(
    constraints: list[dict],
    gc_bounds: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Line chart of RHS bound evolution over stages for constraints with varying bounds.

    Only plots constraints where the bound varies across stages or blocks.
    """
    varying = []
    for c in constraints:
        cid = c["id"]
        rows = gc_bounds[gc_bounds["constraint_id"] == cid]
        if rows.empty:
            continue
        if rows["bound"].std() > 0.01:
            varying.append(c)

    if not varying:
        return "<p>All constraint bounds are constant across stages.</p>"

    fig = go.Figure()
    palette = [
        "#4A90B8",
        "#F5A623",
        "#4A8B6F",
        "#DC4C4C",
        "#B87333",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
        "#607D8B",
        "#795548",
    ]

    # Use a common x-axis covering ALL stages so lines don't jump across gaps
    all_stages = sorted(gc_bounds["stage_id"].unique())
    all_xlabels = stage_x_labels(all_stages, stage_labels)

    for i, c in enumerate(varying):
        cid = c["id"]
        color = palette[i % len(palette)]
        rows = gc_bounds[gc_bounds["constraint_id"] == cid]

        # Use block 0 bounds if multiple blocks present; otherwise stage-level
        if not rows["block_id"].isna().all():
            rows = rows[rows["block_id"] == 0.0]
        b_by_stage = rows.groupby("stage_id")["bound"].mean()
        # Map to the common x-axis, using None for missing stages
        xlabels = all_xlabels

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[float(b_by_stage.get(s, float("nan"))) for s in all_stages],
                name=c["name"],
                line={"color": color, "width": 2},
                connectgaps=False,
            )
        )

    fig.update_layout(
        title="Constraint Bounds Timeline (constraints with varying bounds, block 0)",
        xaxis_title="Stage",
        yaxis_title="Bound Value",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_violation_cost_timeline(
    costs: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Total generic_violation_cost from simulation costs, aggregated by stage.

    Shows p10/p50/p90 across scenarios. If all zeros, still renders the chart
    with a zero line as infrastructure for future cases.
    """
    if "generic_violation_cost" not in costs.columns:
        return "<p>generic_violation_cost column not present in costs output.</p>"

    # costs is stage-level (block_id is NaN)
    pcts = (
        costs.groupby(["scenario_id", "stage_id"])["generic_violation_cost"]
        .sum()
        .reset_index()
        .groupby("stage_id")["generic_violation_cost"]
        .quantile([0.1, 0.5, 0.9])
        .unstack(level=-1)
        .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
    )
    stages = sorted(pcts.index)
    xlabels = stage_x_labels(stages, stage_labels)

    all_zero = pcts["p50"].abs().max() < 1e-6

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels + xlabels[::-1],
            y=list(pcts["p90"].values) + list(pcts["p10"].values[::-1]),
            fill="toself",
            fillcolor="rgba(220,76,76,0.12)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10–P90",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p50"].values,
            name="Median (P50)",
            line={"color": COLORS["deficit"], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p10"].values,
            name="P10",
            line={"color": COLORS["deficit"], "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p90"].values,
            name="P90",
            line={"color": COLORS["deficit"], "width": 1, "dash": "dot"},
        )
    )
    suffix = " — No violations in this run" if all_zero else ""
    fig.update_layout(
        title=f"Generic Constraint Violation Cost by Stage (p10/p50/p90){suffix}",
        xaxis_title="Stage",
        yaxis_title="Violation Cost (R$)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_violation_summary(
    hydros_lf: pl.LazyFrame, stage_labels: dict[int, str]
) -> str:
    """Aggregate constraint violations across all hydros by stage."""
    violation_cols = {
        "storage_violation_below_hm3": ("Storage Violation", "#F44336"),
        "filling_target_violation_hm3": ("Filling Target", "#E91E63"),
        "evaporation_violation_m3s": ("Evaporation Violation", "#9C27B0"),
        "inflow_nonnegativity_slack_m3s": ("Inflow Slack", "#607D8B"),
        "water_withdrawal_violation_m3s": ("Water Withdrawal", "#795548"),
        "turbined_slack_m3s": ("Turbined Slack", "#FF9800"),
        "outflow_slack_below_m3s": ("Outflow Slack Below", "#2196F3"),
        "outflow_slack_above_m3s": ("Outflow Slack Above", "#00BCD4"),
        "generation_slack_mw": ("Generation Slack", "#4CAF50"),
    }

    schema = hydros_lf.collect_schema()
    existing_cols = {c: v for c, v in violation_cols.items() if c in schema}
    if not existing_cols:
        return "<p>No violation data available.</p>"

    viol_data = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .group_by(["scenario_id", "stage_id"])
        .agg([pl.col(c).sum() for c in existing_cols])
        .group_by("stage_id")
        .agg([pl.col(c).mean() for c in existing_cols])
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = viol_data["stage_id"].to_list()
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for col, (label, color) in existing_cols.items():
        if col not in viol_data.columns:
            continue
        vals = viol_data[col].to_list()
        if sum(abs(v) for v in vals) < 1e-6:
            continue
        fig.add_trace(
            go.Scatter(x=xlabels, y=vals, name=label, line={"color": color, "width": 2})
        )

    fig.update_layout(
        title="Aggregate Constraint Violations & Slacks by Stage (all hydros, avg scenarios)",
        xaxis_title="Stage",
        yaxis_title="Total violation / slack",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_violation_heatmap(
    hydros_lf: pl.LazyFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    top_n: int = 20,
) -> str:
    """Heatmap of violations by plant and stage for plants with most violations."""
    violation_cols = [
        "storage_violation_below_hm3",
        "filling_target_violation_hm3",
        "evaporation_violation_m3s",
        "inflow_nonnegativity_slack_m3s",
        "water_withdrawal_violation_m3s",
    ]
    schema = hydros_lf.collect_schema()
    existing = [c for c in violation_cols if c in schema]
    if not existing:
        return "<p>No violation data available.</p>"

    viol_data = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .with_columns(
            pl.sum_horizontal([pl.col(c).abs() for c in existing]).alias("_total_viol")
        )
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg(pl.col("_total_viol").sum())
        .group_by(["stage_id", "hydro_id"])
        .agg(pl.col("_total_viol").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )

    # Top plants
    top_plants = (
        viol_data.group_by("hydro_id")
        .agg(pl.col("_total_viol").sum())
        .sort("_total_viol", descending=True)
        .head(top_n)
        .filter(pl.col("_total_viol") > 1e-6)["hydro_id"]
        .to_list()
    )
    if not top_plants:
        return "<p>No significant violations detected.</p>"

    from cobre_bridge.dashboard.data import entity_name

    stages = sorted(viol_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)
    ynames = [entity_name(names, "hydros", hid) for hid in top_plants]

    z = []
    for hid in top_plants:
        sub = viol_data.filter(pl.col("hydro_id") == hid)
        vmap = dict(zip(sub["stage_id"].to_list(), sub["_total_viol"].to_list()))
        z.append([vmap.get(s, 0) for s in stages])

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=xlabels,
            y=ynames,
            colorscale="YlOrRd",
            colorbar={"title": "Violation"},
        )
    )
    fig.update_layout(
        title=f"Constraint Violations Heatmap (top {len(top_plants)} plants)",
        height=max(350, len(top_plants) * 25 + 120),
        margin=dict(l=120, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True when generic constraint definitions are present."""
    return len(data.gc_constraints) > 0


def render(data: DashboardData) -> str:
    """Return full HTML for the Constraints tab."""
    _lhs_df = evaluate_constraint_expressions(
        data.gc_constraints, data.hydros_lf, data.exchanges_lf
    )
    _summary_table = build_constraints_summary_table(
        data.gc_constraints, data.gc_bounds, data.gc_violations
    )
    _vminop_chart = chart_constraint_lhs_vs_bound(
        data.gc_constraints,
        _lhs_df,
        data.gc_bounds,
        data.stage_labels,
        ctype_filter="VminOP",
    )
    _re_chart = chart_constraint_lhs_vs_bound(
        data.gc_constraints,
        _lhs_df,
        data.gc_bounds,
        data.stage_labels,
        ctype_filter="RE",
    )
    _agrint_chart = chart_constraint_lhs_vs_bound(
        data.gc_constraints,
        _lhs_df,
        data.gc_bounds,
        data.stage_labels,
        ctype_filter="AGRINT",
    )
    return (
        section_title("Constraint Summary")
        + _summary_table
        + section_title("VminOP: Stored Energy vs Minimum")
        + '<div class="chart-grid-single">'
        + wrap_chart(_vminop_chart)
        + "</div>"
        + section_title("Electric Constraints (RE)")
        + '<div class="chart-grid-single">'
        + wrap_chart(_re_chart)
        + "</div>"
        + section_title("Exchange Group Constraints (AGRINT)")
        + '<div class="chart-grid-single">'
        + wrap_chart(_agrint_chart)
        + "</div>"
    )
