#!/usr/bin/env python3
"""Generate a self-contained interactive HTML dashboard from cobre simulation results.

Usage:
    uv run python scripts/dashboard.py example/convertido/ -o example/convertido/dashboard.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyarrow.parquet as pq
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    "hydro": "#2196F3",
    "thermal": "#FF9800",
    "ncs": "#4CAF50",
    "load": "#333333",
    "deficit": "#F44336",
    "spillage": "#9C27B0",
    "curtailment": "#795548",
    "exchange": "#00BCD4",
    "lower_bound": "#1565C0",
    "upper_bound": "#E53935",
    "future_cost": "#546E7A",
}

BUS_COLORS = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]

# Shared legend style: horizontal, positioned above the chart area.
_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="left",
    x=0,
    font=dict(size=11),
)

# Standard bottom margin to leave room for x-axis labels without legend overlap.
_MARGIN = dict(l=60, r=30, t=60, b=50)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_all_scenarios(case_dir: Path, entity: str) -> pd.DataFrame:
    """Load all scenario parquet files for an entity, concatenated."""
    sim_dir = case_dir / "output" / "simulation" / entity
    frames = []
    for scenario_dir in sorted(sim_dir.iterdir()):
        if scenario_dir.is_dir() and scenario_dir.name.startswith("scenario_id="):
            t = pq.read_table(scenario_dir / "data.parquet")
            frames.append(t.to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_names(case_dir: Path) -> dict[tuple[str, int], str]:
    """Load entity name mappings from system JSON files."""
    names: dict[tuple[str, int], str] = {}
    for entity, key in [
        ("hydros", "hydros"),
        ("buses", "buses"),
        ("thermals", "thermals"),
        ("lines", "lines"),
        ("non_controllable_sources", "non_controllable_sources"),
    ]:
        path = case_dir / "system" / f"{entity}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for item in data.get(key, []):
                names[(entity, item["id"])] = item.get("name", str(item["id"]))
    return names


def entity_name(names: dict[tuple[str, int], str], entity: str, eid: int) -> str:
    return names.get((entity, eid), str(eid))


def load_stage_labels(case_dir: Path) -> dict[int, str]:
    """Return stage_id -> "Mon YYYY" label from stages.json."""
    path = case_dir / "stages.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    labels: dict[int, str] = {}
    for stage in data.get("stages", []):
        sid = stage["id"]
        start = stage.get("start_date", "")
        if start:
            try:
                dt = pd.to_datetime(start)
                labels[sid] = dt.strftime("%b %Y")
            except Exception:
                labels[sid] = str(sid)
        else:
            labels[sid] = str(sid)
    return labels


def load_hydro_bus_map(case_dir: Path) -> dict[int, int]:
    path = case_dir / "system" / "hydros.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {h["id"]: h["bus_id"] for h in d["hydros"]}


def load_thermal_metadata(case_dir: Path) -> dict[int, dict]:
    """Return thermal_id -> {bus_id, max_mw, cost_per_mwh} metadata."""
    path = case_dir / "system" / "thermals.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    result = {}
    for t in d["thermals"]:
        segments = t.get("cost_segments", [])
        cost = segments[0]["cost_per_mwh"] if segments else 0.0
        cap = (
            sum(s["capacity_mw"] for s in segments)
            if segments
            else t.get("generation", {}).get("max_mw", 0.0)
        )
        result[t["id"]] = {
            "bus_id": t["bus_id"],
            "max_mw": cap,
            "cost_per_mwh": cost,
            "name": t.get("name", str(t["id"])),
        }
    return result


def load_ncs_bus_map(case_dir: Path) -> dict[int, int]:
    path = case_dir / "system" / "non_controllable_sources.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {n["id"]: n["bus_id"] for n in d["non_controllable_sources"]}


def stage_x_labels(
    stage_ids: "pd.Index | list[int]", labels: dict[int, str]
) -> list[str]:
    """Map stage ids to human-readable labels."""
    return [labels.get(int(s), str(s)) for s in stage_ids]


def scenario_percentiles(
    df: pd.DataFrame, group_col: str, value_col: str
) -> pd.DataFrame:
    """Compute p10/p50/p90 of value_col grouped by group_col across scenarios."""
    grp = df.groupby(["scenario_id", group_col])[value_col].sum().reset_index()
    result = (
        grp.groupby(group_col)[value_col]
        .quantile([0.1, 0.5, 0.9])
        .unstack(level=-1)
        .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
    )
    return result


def fig_to_html(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# Tab 1: Overview charts
# ---------------------------------------------------------------------------


def chart_convergence(conv: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["lower_bound"],
            name="Lower Bound",
            line={"color": COLORS["lower_bound"], "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["upper_bound_mean"],
            name="Upper Bound (mean)",
            line={"color": COLORS["upper_bound"], "width": 2},
        )
    )
    # Shaded band for upper_bound_std
    ub_upper = conv["upper_bound_mean"] + conv["upper_bound_std"]
    ub_lower = conv["upper_bound_mean"] - conv["upper_bound_std"]
    fig.add_trace(
        go.Scatter(
            x=pd.concat([conv["iteration"], conv["iteration"].iloc[::-1]]),
            y=pd.concat([ub_upper, ub_lower.iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(229,57,53,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="Upper Bound ± std",
            showlegend=True,
        )
    )
    last = conv.iloc[-1]
    fig.update_layout(
        title=f"Training Convergence (gap={last['gap_percent']:.2f}%, {int(last['iteration'])} iterations)",
        xaxis_title="Iteration",
        yaxis_title="Cost",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_cost_breakdown(costs: pd.DataFrame) -> str:
    cost_cols = [
        "thermal_cost",
        "deficit_cost",
        "spillage_cost",
        "curtailment_cost",
        "exchange_cost",
        "excess_cost",
        "inflow_penalty_cost",
    ]
    values = []
    labels = []
    for col in cost_cols:
        if col in costs.columns:
            avg = costs.groupby("scenario_id")[col].sum().mean()
            if abs(avg) > 1.0:
                labels.append(col.replace("_cost", "").replace("_", " ").title())
                values.append(avg)
    color_map = {
        "Thermal": COLORS["thermal"],
        "Deficit": COLORS["deficit"],
        "Spillage": COLORS["spillage"],
        "Curtailment": COLORS["curtailment"],
        "Exchange": COLORS["exchange"],
        "Excess": "#FF5722",
        "Inflow Penalty": "#607D8B",
    }
    colors = [color_map.get(l, "#90A4AE") for l in labels]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2e}" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Average Cost Breakdown (sum over all stages)",
        xaxis_title="Cost (R$)",
        height=420,
        margin=dict(l=120, r=30, t=60, b=50),
        showlegend=False,
    )
    return fig_to_html(fig)


def build_key_metrics_html(
    hydros: pd.DataFrame,
    thermals: pd.DataFrame,
    ncs: pd.DataFrame,
    buses: pd.DataFrame,
    costs: pd.DataFrame,
) -> str:
    total_hydro_gwh = hydros.groupby("scenario_id")["generation_mwh"].sum().mean() / 1e3
    total_thermal_gwh = (
        thermals.groupby("scenario_id")["generation_mwh"].sum().mean() / 1e3
    )
    total_ncs_gwh = ncs.groupby("scenario_id")["generation_mwh"].sum().mean() / 1e3
    avg_spot = buses[buses["bus_id"].isin([0, 1, 2, 3])]["spot_price"].mean()
    total_spillage = hydros.groupby("scenario_id")["spillage_m3s"].sum().mean()
    ncs_gen = ncs.groupby("scenario_id")["generation_mwh"].sum().mean()
    ncs_curt = ncs.groupby("scenario_id")["curtailment_mwh"].sum().mean()
    curt_rate = ncs_curt / max(ncs_gen + ncs_curt, 1) * 100

    metrics = [
        ("Total Hydro Generation", f"{total_hydro_gwh:,.0f} GWh", COLORS["hydro"]),
        (
            "Total Thermal Generation",
            f"{total_thermal_gwh:,.0f} GWh",
            COLORS["thermal"],
        ),
        ("Total NCS Generation", f"{total_ncs_gwh:,.0f} GWh", COLORS["ncs"]),
        ("Average Spot Price", f"R$ {avg_spot:.2f}/MWh", "#607D8B"),
        ("Total Spillage", f"{total_spillage:,.0f} m³/s", COLORS["spillage"]),
        ("NCS Curtailment Rate", f"{curt_rate:.1f}%", COLORS["curtailment"]),
    ]
    cards = []
    for label, value, color in metrics:
        cards.append(
            f'<div class="metric-card" style="border-top: 4px solid {color};">'
            f'<div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f"</div>"
        )
    return '<div class="metrics-grid">' + "".join(cards) + "</div>"


# ---------------------------------------------------------------------------
# Tab 2: Energy Balance charts
# ---------------------------------------------------------------------------


def _stage_avg_mw(
    df: pd.DataFrame,
    mwh_col: str,
    stage_hours: dict[int, float],
    group_cols: list[str],
) -> pd.Series:
    """Compute stage-average MW from MWh summed across all blocks.

    Groups by ``["scenario_id", "stage_id"] + group_cols``, sums
    ``mwh_col`` over all blocks, divides by total stage hours, then
    averages across scenarios.
    """
    scen_stage = (
        df.groupby(["scenario_id", "stage_id"] + group_cols)[mwh_col]
        .sum()
        .reset_index()
    )
    scen_stage["_hours"] = scen_stage["stage_id"].map(stage_hours)
    scen_stage["_avg_mw"] = scen_stage[mwh_col] / scen_stage["_hours"]
    return scen_stage.groupby(["stage_id"] + group_cols)["_avg_mw"].mean()


def _compute_lp_load(
    load_stats: pd.DataFrame,
    load_factors: list[dict],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
    bus_filter: list[int] | None = None,
) -> pd.Series:
    """Compute LP load balance RHS from input files as stage-average MW.

    The LP RHS is ``mean_mw * load_factor`` per (bus, stage, block).
    Computed directly from input data to avoid ambiguity in simulation output.
    """
    factor_lk: dict[tuple[int, int, int], float] = {}
    for entry in load_factors:
        bid, sid = entry["bus_id"], entry["stage_id"]
        for bf in entry["block_factors"]:
            factor_lk[(bid, sid, bf["block_id"])] = bf["factor"]

    rows: list[dict] = []
    for _, r in load_stats.iterrows():
        bid, sid = int(r["bus_id"]), int(r["stage_id"])
        if bus_filter is not None and bid not in bus_filter:
            continue
        mean = float(r["mean_mw"])
        for blk_id in range(20):
            bh = block_hours.get((sid, blk_id))
            if bh is None:
                break
            factor = factor_lk.get((bid, sid, blk_id), 1.0)
            rows.append(
                {"scenario_id": 0, "stage_id": sid, "_lp_mwh": mean * factor * bh}
            )

    if not rows:
        return pd.Series(dtype=float)
    return _stage_avg_mw(pd.DataFrame(rows), "_lp_mwh", stage_hours, [])


def chart_generation_mix(
    hydros: pd.DataFrame,
    thermals: pd.DataFrame,
    ncs: pd.DataFrame,
    load_stats: pd.DataFrame,
    load_factors_list: list[dict],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
) -> str:
    """Stacked area: hydro/thermal/NCS vs LP load per stage (stage-avg MW)."""
    h_gen = _stage_avg_mw(hydros, "generation_mwh", stage_hours, [])
    t_gen = _stage_avg_mw(thermals, "generation_mwh", stage_hours, [])
    n_gen = _stage_avg_mw(ncs, "generation_mwh", stage_hours, [])

    load_ser = _compute_lp_load(
        load_stats,
        load_factors_list,
        stage_hours,
        block_hours,
        bus_filter=[0, 1, 2, 3],
    )

    stages = sorted(h_gen.index)
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[h_gen.get(s, 0) for s in stages],
            name="Hydro",
            stackgroup="gen",
            fillcolor="rgba(33,150,243,0.7)",
            line={"color": COLORS["hydro"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[t_gen.get(s, 0) for s in stages],
            name="Thermal",
            stackgroup="gen",
            fillcolor="rgba(255,152,0,0.7)",
            line={"color": COLORS["thermal"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[n_gen.get(s, 0) for s in stages],
            name="NCS",
            stackgroup="gen",
            fillcolor="rgba(76,175,80,0.7)",
            line={"color": COLORS["ncs"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[load_ser.get(s, 0) for s in stages],
            name="LP Load",
            line={"color": COLORS["load"], "width": 2.5, "dash": "dash"},
            mode="lines",
        )
    )
    fig.update_layout(
        title="System-Wide Generation Mix vs LP Load (stage-avg MW)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_generation_by_bus(
    hydros: pd.DataFrame,
    thermals: pd.DataFrame,
    ncs: pd.DataFrame,
    buses: pd.DataFrame,
    exchanges: pd.DataFrame,
    hydro_bus_map: dict[int, int],
    thermal_meta: dict[int, dict],
    ncs_bus_map: dict[int, int],
    line_meta: list[dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    load_stats: pd.DataFrame,
    load_factors_list: list[dict],
    block_hours: dict[tuple[int, int], float],
) -> str:
    """Subplots per bus: hydro+thermal+NCS+net_import vs LP load (stage-avg MW)."""
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    n_buses = len(bus_ids)
    fig = make_subplots(
        rows=n_buses,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=0.06,
    )

    hydros_c = hydros.copy()
    hydros_c["bus_id"] = hydros_c["hydro_id"].map(hydro_bus_map)
    thermals_c = thermals.copy()
    thermals_c["bus_id"] = thermals_c["thermal_id"].map(
        {k: v["bus_id"] for k, v in thermal_meta.items()}
    )
    ncs_c = ncs.copy()
    ncs_c["bus_id"] = ncs_c["non_controllable_id"].map(ncs_bus_map)

    # Net exchange import per bus (stage-average MW)
    ex_import: dict[int, pd.Series] = {}
    for bus_id in bus_ids:
        parts = []
        for ln in line_meta:
            lid, src, tgt = ln["id"], ln["source_bus_id"], ln["target_bus_id"]
            if src != bus_id and tgt != bus_id:
                continue
            le = exchanges[exchanges["line_id"] == lid].copy()
            if tgt == bus_id:
                le["_imp_mwh"] = le["net_flow_mwh"]
            else:
                le["_imp_mwh"] = -le["net_flow_mwh"]
            parts.append(le.groupby(["scenario_id", "stage_id"])["_imp_mwh"].sum())
        if parts:
            c = pd.concat(parts, axis=1).sum(axis=1).reset_index()
            c.columns = ["scenario_id", "stage_id", "_imp_mwh"]
            c["_h"] = c["stage_id"].map(stage_hours)
            c["_mw"] = c["_imp_mwh"] / c["_h"]
            ex_import[bus_id] = c.groupby("stage_id")["_mw"].mean()
        else:
            ex_import[bus_id] = pd.Series(dtype=float)

    stages_all = sorted(hydros_c["stage_id"].unique())
    xlabels = stage_x_labels(stages_all, stage_labels)

    for row_idx, bus_id in enumerate(bus_ids, start=1):
        show_legend = row_idx == 1
        h_gen = _stage_avg_mw(
            hydros_c[hydros_c["bus_id"] == bus_id], "generation_mwh", stage_hours, []
        )
        t_gen = _stage_avg_mw(
            thermals_c[thermals_c["bus_id"] == bus_id],
            "generation_mwh",
            stage_hours,
            [],
        )
        n_gen = _stage_avg_mw(
            ncs_c[ncs_c["bus_id"] == bus_id], "generation_mwh", stage_hours, []
        )
        load_s = _compute_lp_load(
            load_stats,
            load_factors_list,
            stage_hours,
            block_hours,
            bus_filter=[bus_id],
        )
        net_imp = ex_import.get(bus_id, pd.Series(dtype=float))

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[h_gen.get(s, 0) for s in stages_all],
                name="Hydro",
                stackgroup=f"g{bus_id}",
                fillcolor="rgba(33,150,243,0.6)",
                line={"color": COLORS["hydro"]},
                legendgroup="hydro",
                showlegend=show_legend,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[t_gen.get(s, 0) for s in stages_all],
                name="Thermal",
                stackgroup=f"g{bus_id}",
                fillcolor="rgba(255,152,0,0.6)",
                line={"color": COLORS["thermal"]},
                legendgroup="thermal",
                showlegend=show_legend,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[n_gen.get(s, 0) for s in stages_all],
                name="NCS",
                stackgroup=f"g{bus_id}",
                fillcolor="rgba(76,175,80,0.6)",
                line={"color": COLORS["ncs"]},
                legendgroup="ncs",
                showlegend=show_legend,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[net_imp.get(s, 0) for s in stages_all],
                name="Net Import",
                stackgroup=f"g{bus_id}",
                fillcolor="rgba(0,188,212,0.5)",
                line={"color": COLORS["exchange"]},
                legendgroup="import",
                showlegend=show_legend,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[load_s.get(s, 0) for s in stages_all],
                name="LP Load",
                mode="lines",
                line={"color": COLORS["load"], "width": 2, "dash": "dash"},
                legendgroup="load",
                showlegend=show_legend,
            ),
            row=row_idx,
            col=1,
        )

    fig.update_layout(
        title="Generation + Net Import vs LP Load by Bus (stage-avg MW)",
        height=280 * n_buses,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig_to_html(fig)


def chart_generation_share_pie(
    hydros: pd.DataFrame,
    thermals: pd.DataFrame,
    ncs: pd.DataFrame,
    buses: pd.DataFrame,
) -> str:
    """Pie chart of average generation shares including deficit."""
    h_total = hydros.groupby("scenario_id")["generation_mwh"].sum().mean()
    t_total = thermals.groupby("scenario_id")["generation_mwh"].sum().mean()
    n_total = ncs.groupby("scenario_id")["generation_mwh"].sum().mean()
    d_total = (
        buses[buses["bus_id"].isin([0, 1, 2, 3])]
        .groupby("scenario_id")["deficit_mwh"]
        .sum()
        .mean()
    )

    labels = ["Hydro", "Thermal", "NCS", "Deficit"]
    values = [h_total, t_total, n_total, d_total]
    colors = [COLORS["hydro"], COLORS["thermal"], COLORS["ncs"], COLORS["deficit"]]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            textinfo="label+percent",
            hole=0.35,
        )
    )
    fig.update_layout(
        title="Average Generation Share (GWh)", height=440, margin=_MARGIN
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Tab 3: Hydro Operations charts
# ---------------------------------------------------------------------------


def chart_hydro_storage(hydros: pd.DataFrame, stage_labels: dict[int, str]) -> str:
    """Total storage with p10/p50/p90 bands."""
    # storage_final_hm3: sum across all hydros per stage per scenario
    # Use block_id=0 since storage is per stage, not block
    h0 = hydros[hydros["block_id"] == 0]
    total_stor = (
        h0.groupby(["scenario_id", "stage_id"])["storage_final_hm3"].sum().reset_index()
    )
    pcts = (
        total_stor.groupby("stage_id")["storage_final_hm3"]
        .quantile([0.1, 0.5, 0.9])
        .unstack(level=-1)
        .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
    )
    stages = sorted(pcts.index)
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    # P10-P90 band
    fig.add_trace(
        go.Scatter(
            x=xlabels + xlabels[::-1],
            y=list(pcts["p90"].values) + list(pcts["p10"].values[::-1]),
            fill="toself",
            fillcolor="rgba(33,150,243,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10–P90 range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p50"].values,
            name="Median (P50)",
            line={"color": COLORS["hydro"], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p10"].values,
            name="P10",
            line={"color": COLORS["hydro"], "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p90"].values,
            name="P90",
            line={"color": COLORS["hydro"], "width": 1, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Aggregate Reservoir Storage (all hydros, p10/p50/p90 across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Storage (hm³)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_hydro_gen_by_bus(
    hydros: pd.DataFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Stacked area of hydro generation by bus."""
    h0 = hydros[hydros["block_id"] == 0].copy()
    h0["bus_id"] = h0["hydro_id"].map(hydro_bus_map)
    bus_ids = sorted(h0["bus_id"].dropna().unique())
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        bus_id_int = int(bus_id)
        b_gen = (
            h0[h0["bus_id"] == bus_id_int]
            .groupby(["scenario_id", "stage_id"])["generation_mw"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        color = BUS_COLORS[i % len(BUS_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[b_gen.get(s, 0) for s in stages],
                name=bus_names.get(bus_id_int, str(bus_id_int)),
                stackgroup="buses",
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Hydro Generation by Bus (Block 0, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_spillage_by_stage(
    hydros: pd.DataFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    top_n: int = 5,
) -> str:
    """Total spillage with top contributing hydros highlighted."""
    h0 = hydros[hydros["block_id"] == 0]
    # Top hydros by average spillage
    top_hydros = (
        h0.groupby(["scenario_id", "hydro_id"])["spillage_m3s"]
        .sum()
        .groupby("hydro_id")
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    total_spill = (
        h0.groupby(["scenario_id", "stage_id"])["spillage_m3s"]
        .sum()
        .groupby("stage_id")
        .mean()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[total_spill.get(s, 0) for s in stages],
            name="Total Spillage",
            line={"color": COLORS["spillage"], "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(156,39,176,0.15)",
        )
    )
    palette = ["#AB47BC", "#7B1FA2", "#BA68C8", "#CE93D8", "#E1BEE7"]
    for i, hid in enumerate(top_hydros):
        hname = entity_name(names, "hydros", hid)
        spill = (
            h0[h0["hydro_id"] == hid]
            .groupby(["scenario_id", "stage_id"])["spillage_m3s"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[spill.get(s, 0) for s in stages],
                name=hname,
                line={"color": palette[i % len(palette)], "width": 1.5, "dash": "dash"},
                mode="lines",
            )
        )
    fig.update_layout(
        title=f"Spillage by Stage (total + top {top_n} hydros, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Spillage (m³/s)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_inflow_slack(
    hydros: pd.DataFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    top_n: int = 5,
) -> str:
    """Inflow nonnegativity slack by stage."""
    h0 = hydros[hydros["block_id"] == 0]
    top_hydros = (
        h0.groupby(["scenario_id", "hydro_id"])["inflow_nonnegativity_slack_m3s"]
        .sum()
        .groupby("hydro_id")
        .mean()
        .nlargest(top_n)
        .index.tolist()
    )
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)
    total_slack = (
        h0.groupby(["scenario_id", "stage_id"])["inflow_nonnegativity_slack_m3s"]
        .sum()
        .groupby("stage_id")
        .mean()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[total_slack.get(s, 0) for s in stages],
            name="Total Slack",
            line={"color": "#607D8B", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(96,125,139,0.15)",
        )
    )
    palette = ["#78909C", "#546E7A", "#90A4AE", "#B0BEC5", "#455A64"]
    for i, hid in enumerate(top_hydros):
        hname = entity_name(names, "hydros", hid)
        slack = (
            h0[h0["hydro_id"] == hid]
            .groupby(["scenario_id", "stage_id"])["inflow_nonnegativity_slack_m3s"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        if slack.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=xlabels,
                    y=[slack.get(s, 0) for s in stages],
                    name=hname,
                    line={
                        "color": palette[i % len(palette)],
                        "width": 1.5,
                        "dash": "dash",
                    },
                    mode="lines",
                )
            )
    fig.update_layout(
        title=f"Inflow Nonnegativity Slack (total + top {top_n} hydros, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Slack (m³/s)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_water_value_distribution(
    hydros: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Box plot of water values across hydros by stage (sample stages for readability)."""
    h0 = hydros[hydros["block_id"] == 0]
    # Average across scenarios first, then keep per-hydro variation
    avg_by_stage_hydro = (
        h0.groupby(["stage_id", "hydro_id"])["water_value_per_hm3"].mean().reset_index()
    )
    stages = sorted(avg_by_stage_hydro["stage_id"].unique())
    # Sample at most 24 stages for readability
    step = max(1, len(stages) // 24)
    sampled_stages = stages[::step]
    xlabels = stage_x_labels(sampled_stages, stage_labels)

    fig = go.Figure()
    for s, lbl in zip(sampled_stages, xlabels):
        vals = avg_by_stage_hydro[avg_by_stage_hydro["stage_id"] == s][
            "water_value_per_hm3"
        ].values
        fig.add_trace(
            go.Box(
                y=vals,
                name=lbl,
                marker_color=COLORS["hydro"],
                showlegend=False,
                boxpoints=False,
            )
        )
    fig.update_layout(
        title="Water Value Distribution across Hydros by Stage",
        xaxis_title="Stage",
        yaxis_title="Water Value (R$/hm³)",
        height=440,
        legend=_LEGEND,
        margin=_MARGIN,
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Tab 4: Exchanges charts
# ---------------------------------------------------------------------------


def chart_net_flow_by_line(
    exchanges: pd.DataFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
) -> str:
    """Net flow by line by stage."""
    ex0 = exchanges[exchanges["block_id"] == 0]
    stages = sorted(ex0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)
    line_ids = sorted(ex0["line_id"].unique())

    fig = go.Figure()
    palette = [
        "#00BCD4",
        "#0097A7",
        "#006064",
        "#4DD0E1",
        "#80DEEA",
        "#B2EBF2",
        "#00838F",
    ]
    for i, lid in enumerate(line_ids):
        lname = entity_name(names, "lines", lid)
        flow = (
            ex0[ex0["line_id"] == lid]
            .groupby(["scenario_id", "stage_id"])["net_flow_mw"]
            .mean()
            .groupby("stage_id")
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[flow.get(s, 0) for s in stages],
                name=lname,
                line={"color": palette[i % len(palette)], "width": 2},
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Net Flow by Line by Stage (avg across scenarios, positive = direct direction)",
        xaxis_title="Stage",
        yaxis_title="Net Flow (MW)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_capacity_utilization_heatmap(
    exchanges: pd.DataFrame,
    line_bounds: pd.DataFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
) -> str:
    """Heatmap of capacity utilization: lines vs stages."""
    ex0 = exchanges[exchanges["block_id"] == 0]
    line_ids = sorted(ex0["line_id"].unique())
    stages = sorted(ex0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)
    ynames = [entity_name(names, "lines", lid) for lid in line_ids]

    # Direct utilization
    z_direct = []
    z_reverse = []
    for lid in line_ids:
        ex_line = (
            ex0[ex0["line_id"] == lid]
            .groupby(["scenario_id", "stage_id"])
            .agg(
                direct=("direct_flow_mw", "mean"),
                reverse=("reverse_flow_mw", "mean"),
            )
            .groupby("stage_id")
            .mean()
        )

        lb_line = line_bounds[line_bounds["line_id"] == lid].set_index("stage_id")
        row_d = []
        row_r = []
        for s in stages:
            d_flow = ex_line["direct"].get(s, 0) if s in ex_line.index else 0
            r_flow = ex_line["reverse"].get(s, 0) if s in ex_line.index else 0
            d_cap = lb_line["direct_mw"].get(s, 1) if s in lb_line.index else 1
            r_cap = lb_line["reverse_mw"].get(s, 1) if s in lb_line.index else 1
            row_d.append(min(d_flow / max(d_cap, 0.1) * 100, 100))
            row_r.append(min(r_flow / max(r_cap, 0.1) * 100, 100))
        z_direct.append(row_d)
        z_reverse.append(row_r)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Direct Utilization (%)", "Reverse Utilization (%)"],
        horizontal_spacing=0.12,
    )
    for col, z, title in [(1, z_direct, "Direct"), (2, z_reverse, "Reverse")]:
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=xlabels,
                y=ynames,
                colorscale="RdYlGn_r",
                zmin=0,
                zmax=100,
                colorbar={"x": 0.45 if col == 1 else 1.0, "len": 0.9, "title": "%"},
                showscale=True,
                name=title,
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        title="Capacity Utilization Heatmap (avg across scenarios)",
        height=max(300, len(line_ids) * 60 + 120),
        margin=_MARGIN,
    )
    return fig_to_html(fig)


def chart_flow_direction_summary(
    exchanges: pd.DataFrame,
    names: dict[tuple[str, int], str],
) -> str:
    """Average direct and reverse flow per line (horizontal grouped bar)."""
    ex0 = exchanges[exchanges["block_id"] == 0]
    line_ids = sorted(ex0["line_id"].unique())
    ynames = [entity_name(names, "lines", lid) for lid in line_ids]
    avg_direct = []
    avg_reverse = []
    for lid in line_ids:
        ex_line = ex0[ex0["line_id"] == lid]
        avg_direct.append(
            ex_line.groupby("scenario_id")["direct_flow_mw"].mean().mean()
        )
        avg_reverse.append(
            ex_line.groupby("scenario_id")["reverse_flow_mw"].mean().mean()
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=avg_direct,
            y=ynames,
            name="Direct Flow",
            orientation="h",
            marker_color=COLORS["exchange"],
        )
    )
    fig.add_trace(
        go.Bar(
            x=[-v for v in avg_reverse],
            y=ynames,
            name="Reverse Flow (negative)",
            orientation="h",
            marker_color="#0097A7",
        )
    )
    fig.update_layout(
        title="Average Direct vs Reverse Flow per Line",
        xaxis_title="Flow (MW)",
        barmode="relative",
        legend=_LEGEND,
        height=max(300, len(line_ids) * 50 + 100),
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Tab 5: Costs charts
# ---------------------------------------------------------------------------


def chart_cost_by_stage(
    costs: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Stacked area of cost components by stage."""
    avg = costs.groupby("stage_id").mean(numeric_only=True)
    stages = sorted(avg.index)
    xlabels = stage_x_labels(stages, stage_labels)

    cost_series = {
        "Thermal": ("thermal_cost", COLORS["thermal"]),
        "Deficit": ("deficit_cost", COLORS["deficit"]),
        "Spillage": ("spillage_cost", COLORS["spillage"]),
        "Curtailment": ("curtailment_cost", COLORS["curtailment"]),
        "Exchange": ("exchange_cost", COLORS["exchange"]),
        "Excess": ("excess_cost", "#FF5722"),
        "Inflow Penalty": ("inflow_penalty_cost", "#607D8B"),
    }

    fig = go.Figure()
    for label, (col, color) in cost_series.items():
        if col in avg.columns:
            vals = [avg[col].get(s, 0) for s in stages]
            if sum(abs(v) for v in vals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=xlabels,
                        y=vals,
                        name=label,
                        stackgroup="costs",
                        line={"color": color},
                        fillcolor=color.replace(")", ",0.7)")
                        .replace("rgb", "rgba")
                        .replace("#", "rgba(")
                        .replace("rgba(", "rgba(")
                        if "#" not in color
                        else color,
                    )
                )
    fig.update_layout(
        title="Cost Composition by Stage (avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Cost (R$)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_spot_price_by_bus(
    buses: pd.DataFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Spot price by bus by stage."""
    b0 = buses[buses["block_id"] == 0]
    real_buses = [bid for bid in sorted(b0["bus_id"].unique()) if bid <= 3]
    stages = sorted(b0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(real_buses):
        bname = bus_names.get(bus_id, str(bus_id))
        sp = (
            b0[b0["bus_id"] == bus_id]
            .groupby(["scenario_id", "stage_id"])["spot_price"]
            .mean()
            .groupby("stage_id")
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[sp.get(s, 0) for s in stages],
                name=bname,
                line={"color": BUS_COLORS[i % len(BUS_COLORS)], "width": 2},
            )
        )
    fig.update_layout(
        title="Spot Price by Bus by Stage (Block 0, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Spot Price (R$/MWh)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_thermal_merit_order(
    thermals: pd.DataFrame,
    thermal_meta: dict[int, dict],
    top_n: int = 30,
) -> str:
    """Horizontal bar: thermals sorted by cost, showing avg generation vs capacity."""
    avg_gen = (
        thermals.groupby(["scenario_id", "thermal_id"])["generation_mwh"]
        .sum()
        .groupby("thermal_id")
        .mean()
    )
    # Sort by cost
    sorted_thermals = sorted(
        thermal_meta.items(),
        key=lambda x: x[1]["cost_per_mwh"],
    )[:top_n]

    names_list = []
    gen_vals = []
    cap_vals = []
    cost_vals = []
    for tid, meta in sorted_thermals:
        names_list.append(meta["name"])
        gen_vals.append(avg_gen.get(tid, 0) / 1e3)  # GWh
        cap_vals.append(meta["max_mw"] * 8760 / 1e3)  # theoretical max GWh/year
        cost_vals.append(meta["cost_per_mwh"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=cap_vals,
            y=names_list,
            name="Installed Capacity (GWh/yr equivalent)",
            orientation="h",
            marker_color="rgba(255,152,0,0.3)",
            marker_line_color=COLORS["thermal"],
            marker_line_width=1,
        )
    )
    fig.add_trace(
        go.Bar(
            x=gen_vals,
            y=names_list,
            name="Avg Generation (GWh)",
            orientation="h",
            marker_color=COLORS["thermal"],
        )
    )
    fig.update_layout(
        title=f"Thermal Merit Order (top {top_n} by cost, sorted low→high)",
        xaxis_title="GWh",
        barmode="overlay",
        legend=_LEGEND,
        height=max(440, len(names_list) * 22 + 100),
        margin=dict(l=120, r=30, t=60, b=50),
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# Tab 6: NCS & Thermals charts
# ---------------------------------------------------------------------------


def chart_ncs_available_vs_generated(
    ncs: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Stacked area of NCS available vs generated with curtailment gap."""
    n0 = ncs[ncs["block_id"] == 0]
    stages = sorted(n0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    avail = (
        n0.groupby(["scenario_id", "stage_id"])["available_mw"]
        .sum()
        .groupby("stage_id")
        .mean()
    )
    gen = (
        n0.groupby(["scenario_id", "stage_id"])["generation_mw"]
        .sum()
        .groupby("stage_id")
        .mean()
    )
    curtail = (
        n0.groupby(["scenario_id", "stage_id"])["curtailment_mw"]
        .sum()
        .groupby("stage_id")
        .mean()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[avail.get(s, 0) for s in stages],
            name="Available",
            line={"color": "#A5D6A7", "width": 2, "dash": "dash"},
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[gen.get(s, 0) for s in stages],
            name="Generated",
            stackgroup="ncs",
            fillcolor="rgba(76,175,80,0.7)",
            line={"color": COLORS["ncs"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[curtail.get(s, 0) for s in stages],
            name="Curtailment",
            stackgroup="ncs",
            fillcolor="rgba(121,85,72,0.6)",
            line={"color": COLORS["curtailment"]},
        )
    )
    fig.update_layout(
        title="NCS Available vs Generated (Block 0, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_ncs_curtailment_by_source(
    ncs: pd.DataFrame,
    names: dict[tuple[str, int], str],
    top_n: int = 20,
) -> str:
    """Bar chart of top curtailed NCS sources."""
    curt_by_ncs = (
        ncs.groupby(["scenario_id", "non_controllable_id"])["curtailment_mwh"]
        .sum()
        .groupby("non_controllable_id")
        .mean()
        .nlargest(top_n)
    )
    ynames = [
        entity_name(names, "non_controllable_sources", int(nid))
        for nid in curt_by_ncs.index
    ]
    values = [v / 1e3 for v in curt_by_ncs.values]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=ynames,
            orientation="h",
            marker_color=COLORS["curtailment"],
        )
    )
    fig.update_layout(
        title=f"Top {top_n} NCS Sources by Average Curtailment",
        xaxis_title="Avg Curtailment (GWh)",
        height=max(350, top_n * 22 + 100),
        margin=dict(l=100, r=30, t=60, b=50),
        showlegend=False,
    )
    return fig_to_html(fig)


def chart_thermal_by_cost_bracket(
    thermals: pd.DataFrame,
    thermal_meta: dict[int, dict],
    stage_labels: dict[int, str],
) -> str:
    """Stacked area of thermal generation grouped by cost bracket."""
    # Assign cost bracket to each thermal
    brackets = {
        "Zero cost": (0, 0),
        "0–100 R$/MWh": (0, 100),
        "100–500 R$/MWh": (100, 500),
        "500+ R$/MWh": (500, float("inf")),
    }
    bracket_colors = {
        "Zero cost": "#FFF9C4",
        "0–100 R$/MWh": "#FFE082",
        "100–500 R$/MWh": COLORS["thermal"],
        "500+ R$/MWh": "#E65100",
    }

    t0 = thermals[thermals["block_id"] == 0].copy()
    t0["cost_per_mwh"] = t0["thermal_id"].map(
        {k: v["cost_per_mwh"] for k, v in thermal_meta.items()}
    )

    def assign_bracket(cost: float) -> str:
        if cost == 0:
            return "Zero cost"
        elif cost <= 100:
            return "0–100 R$/MWh"
        elif cost <= 500:
            return "100–500 R$/MWh"
        else:
            return "500+ R$/MWh"

    t0["bracket"] = t0["cost_per_mwh"].map(assign_bracket)
    stages = sorted(t0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for bracket_name, color in bracket_colors.items():
        bt = t0[t0["bracket"] == bracket_name]
        if bt.empty:
            continue
        gen = (
            bt.groupby(["scenario_id", "stage_id"])["generation_mw"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        if gen.sum() == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[gen.get(s, 0) for s in stages],
                name=bracket_name,
                stackgroup="brackets",
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Thermal Generation by Cost Bracket (Block 0, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5; color: #212121; }

header {
    background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
    color: white;
    padding: 16px 32px;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

nav {
    background: #1e2a3a;
    padding: 0 24px;
    display: flex;
    gap: 4px;
    overflow-x: auto;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

nav button {
    background: none;
    border: none;
    color: #90A4AE;
    padding: 14px 20px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: color 0.2s, border-color 0.2s;
    letter-spacing: 0.3px;
}

nav button:hover { color: #E3F2FD; border-bottom-color: #42A5F5; }
nav button.active { color: #90CAF9; border-bottom-color: #2196F3; }

main { padding: 24px 32px; max-width: 1400px; margin: 0 auto; }

.tab-content { display: none; }
.tab-content.active { display: block; }

.chart-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(520px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}

.chart-grid-single {
    display: grid;
    grid-template-columns: 1fr;
    gap: 20px;
    margin-bottom: 20px;
}

.chart-card {
    background: white;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    overflow: hidden;
}

.chart-card .plotly-graph-div { width: 100% !important; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    border-radius: 8px;
    padding: 20px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #212121;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.8rem;
    color: #757575;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #37474F;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #E0E0E0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
"""

JS = """
function showTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
    // Trigger Plotly resize for charts in newly visible tab
    window.dispatchEvent(new Event('resize'));
}
"""

TAB_DEFS = [
    ("tab-overview", "Overview"),
    ("tab-energy", "Energy Balance"),
    ("tab-hydro", "Hydro Operations"),
    ("tab-exchanges", "Exchanges"),
    ("tab-costs", "Costs"),
    ("tab-ncs-thermal", "NCS & Thermals"),
]


def build_html(
    case_name: str,
    tab_contents: dict[str, str],
) -> str:
    nav_buttons = []
    for i, (tab_id, tab_label) in enumerate(TAB_DEFS):
        active_cls = ' class="active"' if i == 0 else ""
        nav_buttons.append(
            f"<button{active_cls} onclick=\"showTab('{tab_id}', this)\">{tab_label}</button>"
        )

    tab_sections = []
    for i, (tab_id, _) in enumerate(TAB_DEFS):
        active_cls = " active" if i == 0 else ""
        content = tab_contents.get(tab_id, "<p>No data</p>")
        tab_sections.append(
            f'<section id="{tab_id}" class="tab-content{active_cls}">\n{content}\n</section>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cobre Dashboard — {case_name}</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
{CSS}
    </style>
</head>
<body>
    <header>Cobre Simulation Dashboard — {case_name}</header>
    <nav>
        {"".join(nav_buttons)}
    </nav>
    <main>
        {"".join(tab_sections)}
    </main>
    <script>
{JS}
    </script>
</body>
</html>"""


def wrap_chart(html: str) -> str:
    return f'<div class="chart-card">{html}</div>'


def section_title(text: str) -> str:
    return f'<h2 class="section-title">{text}</h2>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_dashboard(case_dir: Path, output_path: Path) -> None:
    print(f"Loading data from {case_dir} ...")

    # Load all data
    conv = pq.read_table(
        case_dir / "output" / "training" / "convergence.parquet"
    ).to_pandas()
    hydros = load_all_scenarios(case_dir, "hydros")
    thermals = load_all_scenarios(case_dir, "thermals")
    ncs = load_all_scenarios(case_dir, "non_controllables")
    buses = load_all_scenarios(case_dir, "buses")
    costs = load_all_scenarios(case_dir, "costs")
    exchanges = load_all_scenarios(case_dir, "exchanges")

    lb_path = case_dir / "constraints" / "line_bounds.parquet"
    line_bounds = (
        pq.read_table(lb_path).to_pandas() if lb_path.exists() else pd.DataFrame()
    )

    names = load_names(case_dir)
    stage_labels = load_stage_labels(case_dir)
    hydro_bus_map = load_hydro_bus_map(case_dir)
    thermal_meta = load_thermal_metadata(case_dir)
    ncs_bus_map = load_ncs_bus_map(case_dir)

    # Build bus_names dict: id -> name from names dict
    bus_names = {eid: nm for (entity, eid), nm in names.items() if entity == "buses"}

    # Stage hours: total hours per stage (sum of block hours)
    stages_json_path = case_dir / "stages.json"
    with stages_json_path.open() as f:
        stages_data = json.load(f)
    stage_hours: dict[int, float] = {}
    for s in stages_data["stages"]:
        stage_hours[s["id"]] = sum(b["hours"] for b in s["blocks"])

    # Line metadata for exchange calculations
    lines_path = case_dir / "system" / "lines.json"
    with lines_path.open() as f:
        line_meta: list[dict] = json.load(f)["lines"]

    # Load input data for LP load computation
    load_stats = pq.read_table(
        case_dir / "scenarios" / "load_seasonal_stats.parquet"
    ).to_pandas()
    lf_path = case_dir / "scenarios" / "load_factors.json"
    with lf_path.open() as f:
        load_factors_list: list[dict] = json.load(f)["load_factors"]
    block_hours: dict[tuple[int, int], float] = {}
    for s in stages_data["stages"]:
        for b in s["blocks"]:
            block_hours[(s["id"], b["id"])] = b["hours"]

    case_name = case_dir.resolve().name
    n_scenarios = costs["scenario_id"].nunique()
    n_stages = costs["stage_id"].nunique()
    print(
        f"  {n_scenarios} scenarios, {n_stages} stages, {len(stage_labels)} stage labels"
    )

    tab_contents: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Tab 1: Overview
    # ------------------------------------------------------------------
    print("Building Tab 1: Overview ...")
    metrics_html = build_key_metrics_html(hydros, thermals, ncs, buses, costs)
    tab_contents["tab-overview"] = (
        section_title("Key Metrics")
        + metrics_html
        + section_title("Training Convergence & Cost Breakdown")
        + '<div class="chart-grid">'
        + wrap_chart(chart_convergence(conv))
        + wrap_chart(chart_cost_breakdown(costs))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 2: Energy Balance
    # ------------------------------------------------------------------
    print("Building Tab 2: Energy Balance ...")
    tab_contents["tab-energy"] = (
        section_title("System-Wide Generation Mix")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_generation_mix(
                hydros,
                thermals,
                ncs,
                load_stats,
                load_factors_list,
                stage_labels,
                stage_hours,
                block_hours,
            )
        )
        + "</div>"
        + section_title("Generation Share")
        + '<div class="chart-grid">'
        + wrap_chart(chart_generation_share_pie(hydros, thermals, ncs, buses))
        + "</div>"
        + section_title("Generation by Bus")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_generation_by_bus(
                hydros,
                thermals,
                ncs,
                buses,
                exchanges,
                hydro_bus_map,
                thermal_meta,
                ncs_bus_map,
                line_meta,
                bus_names,
                stage_labels,
                stage_hours,
                load_stats,
                load_factors_list,
                block_hours,
            )
        )
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 3: Hydro Operations
    # ------------------------------------------------------------------
    print("Building Tab 3: Hydro Operations ...")
    tab_contents["tab-hydro"] = (
        section_title("Aggregate Reservoir Storage")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_hydro_storage(hydros, stage_labels))
        + "</div>"
        + section_title("Hydro Generation & Spillage")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_hydro_gen_by_bus(hydros, hydro_bus_map, bus_names, stage_labels)
        )
        + wrap_chart(chart_spillage_by_stage(hydros, names, stage_labels))
        + "</div>"
        + section_title("Inflow Slack & Water Values")
        + '<div class="chart-grid">'
        + wrap_chart(chart_inflow_slack(hydros, names, stage_labels))
        + wrap_chart(chart_water_value_distribution(hydros, stage_labels))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 4: Exchanges
    # ------------------------------------------------------------------
    print("Building Tab 4: Exchanges ...")
    tab_contents["tab-exchanges"] = (
        section_title("Net Flow by Line")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_net_flow_by_line(exchanges, names, stage_labels))
        + "</div>"
        + section_title("Capacity Utilization")
        + '<div class="chart-grid-single">'
        + (
            wrap_chart(
                chart_capacity_utilization_heatmap(
                    exchanges, line_bounds, names, stage_labels
                )
            )
            if not line_bounds.empty
            else "<p>No line bounds data.</p>"
        )
        + "</div>"
        + section_title("Flow Direction Summary")
        + '<div class="chart-grid">'
        + wrap_chart(chart_flow_direction_summary(exchanges, names))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 5: Costs
    # ------------------------------------------------------------------
    print("Building Tab 5: Costs ...")
    tab_contents["tab-costs"] = (
        section_title("Cost Composition by Stage")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_cost_by_stage(costs, stage_labels))
        + "</div>"
        + section_title("Spot Price & Merit Order")
        + '<div class="chart-grid">'
        + wrap_chart(chart_spot_price_by_bus(buses, bus_names, stage_labels))
        + wrap_chart(chart_thermal_merit_order(thermals, thermal_meta))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 6: NCS & Thermals
    # ------------------------------------------------------------------
    print("Building Tab 6: NCS & Thermals ...")
    tab_contents["tab-ncs-thermal"] = (
        section_title("NCS Available vs Generated")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_ncs_available_vs_generated(ncs, stage_labels))
        + "</div>"
        + section_title("NCS Curtailment & Thermal by Cost Bracket")
        + '<div class="chart-grid">'
        + wrap_chart(chart_ncs_curtailment_by_source(ncs, names))
        + wrap_chart(
            chart_thermal_by_cost_bracket(thermals, thermal_meta, stage_labels)
        )
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Write HTML
    # ------------------------------------------------------------------
    print(f"Writing dashboard to {output_path} ...")
    html = build_html(case_name, tab_contents)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    size_kb = output_path.stat().st_size / 1024
    print(f"Done. File size: {size_kb:.0f} KB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate an interactive HTML dashboard from cobre simulation results.",
    )
    parser.add_argument("case_dir", type=Path, help="Path to the cobre case directory.")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output HTML file path (default: <case_dir>/dashboard.html).",
    )
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    if not (case_dir / "output" / "simulation").exists():
        print(f"Error: no simulation output found in {case_dir}", file=sys.stderr)
        sys.exit(1)

    output_path = args.output or (case_dir / "dashboard.html")
    build_dashboard(case_dir, output_path)


if __name__ == "__main__":
    main()
