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
    "hydro": "#4A90B8",
    "thermal": "#F5A623",
    "ncs": "#4A8B6F",
    "load": "#374151",
    "deficit": "#DC4C4C",
    "spillage": "#B87333",
    "curtailment": "#8B5E3C",
    "exchange": "#4A90B8",
    "lower_bound": "#4A8B6F",
    "upper_bound": "#DC4C4C",
    "future_cost": "#8B9298",
}

BUS_COLORS = ["#4A90B8", "#F5A623", "#4A8B6F", "#DC4C4C", "#B87333"]

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


def load_hydro_metadata(case_dir: Path) -> dict[int, dict]:
    """Return hydro_id -> {bus_id, name, vol_max, vol_min, max_gen_mw, max_turbined}."""
    path = case_dir / "system" / "hydros.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    result = {}
    for h in d["hydros"]:
        gen = h.get("generation", {})
        res = h.get("reservoir", {})
        result[h["id"]] = {
            "bus_id": h["bus_id"],
            "name": h.get("name", str(h["id"])),
            "vol_max": res.get("max_storage_hm3", 0),
            "vol_min": res.get("min_storage_hm3", 0),
            "max_gen_mw": gen.get("max_generation_mw", 0),
            "max_gen_physical": gen.get("productivity_mw_per_m3s", 0)
            * gen.get("max_turbined_m3s", 0),
            "max_turbined": gen.get("max_turbined_m3s", 0),
        }
    return result


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


def fig_to_html(fig: go.Figure, unified_hover: bool = True) -> str:
    if unified_hover:
        fig.update_layout(hovermode="x unified")
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True},
    )


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
    fig = go.Figure()
    for label, value in zip(labels, values):
        fig.add_trace(
            go.Bar(
                x=["Total"],
                y=[value],
                name=label,
                marker_color=color_map.get(label, "#90A4AE"),
                text=[f"{value:.2e}"],
                textposition="inside",
                insidetextanchor="middle",
            )
        )
    fig.update_layout(
        title="Average Cost Breakdown (sum over all stages)",
        yaxis_title="Cost (R$)",
        barmode="stack",
        height=420,
        legend=_LEGEND,
        margin=_MARGIN,
        showlegend=True,
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
        shared_xaxes=False,
        subplot_titles=[bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=0.12,
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
        height=350 * n_buses,
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
    return fig_to_html(fig, unified_hover=False)


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
        legend={**_LEGEND, "y": 1.08},
        margin={**_MARGIN, "t": 90},
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
        legend={**_LEGEND, "y": 1.08},
        margin={**_MARGIN, "t": 90},
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
    return fig_to_html(fig, unified_hover=False)


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
    return fig_to_html(fig, unified_hover=False)


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
    return fig_to_html(fig, unified_hover=False)


def build_interactive_exchange_detail(
    exchanges: pd.DataFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
) -> str:
    """Build HTML with embedded per-line p10/p50/p90 data and JS dropdown."""
    ex0 = exchanges[exchanges["block_id"] == 0]
    stages = sorted(ex0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)
    line_ids = sorted(ex0["line_id"].unique())

    line_data: dict[str, dict] = {}
    for lid in line_ids:
        lname = entity_name(names, "lines", lid)
        ldf = ex0[ex0["line_id"] == lid]
        entry: dict = {"name": lname}
        for col, prefix in [
            ("net_flow_mw", "net"),
            ("direct_flow_mw", "direct"),
            ("reverse_flow_mw", "reverse"),
        ]:
            if col not in ldf.columns:
                for sfx in ["p10", "p50", "p90"]:
                    entry[f"{prefix}_{sfx}"] = [0.0] * len(stages)
                continue
            grp = ldf.groupby(["scenario_id", "stage_id"])[col].mean()
            pcts = (
                grp.groupby("stage_id")
                .quantile([0.1, 0.5, 0.9])
                .unstack(level=-1)
                .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
            )
            for sfx in ["p10", "p50", "p90"]:
                entry[f"{prefix}_{sfx}"] = [
                    round(float(pcts[sfx].get(s, 0)), 2) for s in stages
                ]
        line_data[str(lid)] = entry

    options_html = "\n".join(
        f'<option value="{lid}">{d["name"]} (id={lid})</option>'
        for lid, d in sorted(line_data.items(), key=lambda x: x[1]["name"])
    )
    data_json = json.dumps(line_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    chart_rows = (
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="ex-net" style="width:100%;height:350px;"></div></div>'
        "</div>"
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="ex-dir" style="width:100%;height:350px;"></div></div>'
        "</div>"
    )

    return (
        '<div style="margin-bottom:16px;">'
        '<label for="ex-select" style="font-weight:600;margin-right:8px;">Select Line:</label>'
        '<select id="ex-select" onchange="updateExchangeDetail()" '
        'style="padding:8px 12px;font-size:0.9rem;border-radius:4px;border:1px solid #ccc;min-width:280px;">'
        + options_html
        + "</select>"
        + "</div>"
        + chart_rows
        + "<script>\n"
        + "const EX_DATA = "
        + data_json
        + ";\n"
        + "const EX_LABELS = "
        + labels_json
        + ";\n"
        + r"""
function _ex_band(lbl, p10, p90, color) {
  return {x: EX_LABELS.concat(EX_LABELS.slice().reverse()),
          y: p90.concat(p10.slice().reverse()),
          fill:'toself', fillcolor:color, line:{color:'rgba(0,0,0,0)'},
          name:lbl, showlegend:true, hoverinfo:'skip'};
}
function _ex_line(nm, y, c, w, dash) {
  return {x:EX_LABELS, y:y, name:nm, line:{color:c, width:w||2, dash:dash||'solid'}};
}
var _EX_L = {hovermode:'x unified', margin:{l:60,r:20,t:50,b:60},
             legend:{orientation:'h',y:1.12,x:0,font:{size:11}}};
var _EX_C = {responsive:true};
function _ex_lo(extra){return Object.assign({},_EX_L,extra);}

function updateExchangeDetail() {
  var lid = document.getElementById('ex-select').value;
  var d = EX_DATA[lid]; if(!d) return;

  var zeroLine = Array(EX_LABELS.length).fill(0);

  Plotly.react('ex-net', [
    _ex_band('P10\u2013P90', d.net_p10, d.net_p90, 'rgba(74,144,184,0.15)'),
    _ex_line('P50', d.net_p50, '#4A90B8'),
    _ex_line('P10', d.net_p10, '#4A90B8', 1, 'dot'),
    _ex_line('P90', d.net_p90, '#4A90B8', 1, 'dot'),
    {x:EX_LABELS, y:zeroLine, name:'Zero', line:{color:'gray',width:1,dash:'dot'}, showlegend:false},
  ], _ex_lo({title:d.name+' \u2014 Net Flow (MW)', yaxis:{title:'Net Flow (MW)'}}), _EX_C);

  var rev_p10_neg = d.reverse_p10.map(function(v){return -v;});
  var rev_p50_neg = d.reverse_p50.map(function(v){return -v;});
  var rev_p90_neg = d.reverse_p90.map(function(v){return -v;});
  Plotly.react('ex-dir', [
    _ex_band('Direct P10\u2013P90', d.direct_p10, d.direct_p90, 'rgba(74,139,111,0.18)'),
    _ex_line('Direct P50', d.direct_p50, '#4A8B6F'),
    _ex_line('Direct P10', d.direct_p10, '#4A8B6F', 1, 'dot'),
    _ex_line('Direct P90', d.direct_p90, '#4A8B6F', 1, 'dot'),
    _ex_band('Reverse P10\u2013P90 (neg)', rev_p10_neg, rev_p90_neg, 'rgba(220,76,76,0.15)'),
    _ex_line('Reverse P50 (neg)', rev_p50_neg, '#DC4C4C'),
    _ex_line('Reverse P10 (neg)', rev_p10_neg, '#DC4C4C', 1, 'dot'),
    _ex_line('Reverse P90 (neg)', rev_p90_neg, '#DC4C4C', 1, 'dot'),
    {x:EX_LABELS, y:zeroLine, name:'Zero', line:{color:'gray',width:1,dash:'dot'}, showlegend:false},
  ], _ex_lo({title:d.name+' \u2014 Direct / Reverse Flow (MW)', yaxis:{title:'Flow (MW)'}}), _EX_C);
}
document.addEventListener('DOMContentLoaded', function(){setTimeout(updateExchangeDetail,100);});
"""
        + "</script>"
    )


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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
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


def chart_spot_price_by_bus_subplots(
    buses: pd.DataFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
) -> str:
    """2x2 subplots of weighted-average spot price (by block hours) per bus."""
    real_bus_ids = [bid for bid in sorted(buses["bus_id"].unique()) if bid <= 3]

    # Attach block hours as a column for vectorised weighting
    buses_w = buses.copy()
    buses_w["_bh"] = buses_w.apply(
        lambda r: block_hours.get((int(r["stage_id"]), int(r["block_id"])), 0.0),
        axis=1,
    )
    buses_w["_sp_x_bh"] = buses_w["spot_price"] * buses_w["_bh"]

    # Weighted average per (scenario, stage, bus)
    grp = (
        buses_w.groupby(["scenario_id", "stage_id", "bus_id"])
        .apply(
            lambda g: (
                g["_sp_x_bh"].sum() / g["_bh"].sum() if g["_bh"].sum() > 0 else 0.0
            ),
            include_groups=False,
        )
        .rename("w_spot")
        .reset_index()
    )

    n_buses = len(real_bus_ids)
    n_cols = 2
    n_rows = (n_buses + n_cols - 1) // n_cols
    subplot_titles = [bus_names.get(bid, str(bid)) for bid in real_bus_ids]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=subplot_titles,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    stages = sorted(grp["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    for idx, bus_id in enumerate(real_bus_ids):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        color = BUS_COLORS[idx % len(BUS_COLORS)]

        b_grp = grp[grp["bus_id"] == bus_id]
        pcts = (
            b_grp.groupby(["scenario_id", "stage_id"])["w_spot"]
            .mean()
            .groupby("stage_id")
            .quantile([0.1, 0.5, 0.9])
            .unstack(level=-1)
            .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
        )
        p10 = [float(pcts["p10"].get(s, 0)) for s in stages]
        p50 = [float(pcts["p50"].get(s, 0)) for s in stages]
        p90 = [float(pcts["p90"].get(s, 0)) for s in stages]

        # Shaded band
        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.15)"
                if color.startswith("#")
                else color,
                line={"color": "rgba(0,0,0,0)"},
                name=f"P10\u2013P90 {bus_names.get(bus_id, str(bus_id))}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p50,
                name=f"P50 {bus_names.get(bus_id, str(bus_id))}",
                line={"color": color, "width": 2},
                showlegend=True,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p10,
                name=f"P10 {bus_names.get(bus_id, str(bus_id))}",
                line={"color": color, "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p90,
                name=f"P90 {bus_names.get(bus_id, str(bus_id))}",
                line={"color": color, "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title="Weighted-Average Spot Price by Bus (block-hours weighted, p10/p50/p90)",
        height=350 * n_rows + 60,
        legend={**_LEGEND, "y": 1.02, "yanchor": "bottom"},
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    for ax in fig.layout:
        if ax.startswith("yaxis"):
            fig.layout[ax].title = "R$/MWh"  # type: ignore[index]
    return fig_to_html(fig, unified_hover=False)


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
        legend={**_LEGEND, "y": 1.08},
        height=max(440, len(names_list) * 22 + 100),
        margin=dict(l=120, r=30, t=90, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


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
    return fig_to_html(fig, unified_hover=False)


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
# Tab 7: Plant Details charts
# ---------------------------------------------------------------------------


def chart_storage_by_bus(
    hydros: pd.DataFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Storage by bus with p50 line and p10-p90 band in subplots."""
    h0 = hydros[hydros["block_id"] == 0].copy()
    h0["bus_id"] = h0["hydro_id"].map(hydro_bus_map)
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    n = len(bus_ids)
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=0.12,
    )

    for row, bus_id in enumerate(bus_ids, 1):
        bus_data = h0[h0["bus_id"] == bus_id]
        total = (
            bus_data.groupby(["scenario_id", "stage_id"])["storage_final_hm3"]
            .sum()
            .reset_index()
        )
        pcts = (
            total.groupby("stage_id")["storage_final_hm3"]
            .quantile([0.1, 0.5, 0.9])
            .unstack(level=-1)
            .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
        )
        show_legend = row == 1
        p90 = [pcts["p90"].get(s, 0) for s in stages]
        p10 = [pcts["p10"].get(s, 0) for s in stages]
        p50 = [pcts["p50"].get(s, 0) for s in stages]

        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(33,150,243,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                name="P10-P90",
                legendgroup="band",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p50,
                name="Median",
                line={"color": COLORS["hydro"], "width": 2},
                legendgroup="median",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="hm³", row=row, col=1)

    fig.update_layout(
        title="Reservoir Storage by Bus (p10/p50/p90 across scenarios)",
        height=320 * n + 60,
        legend={**_LEGEND, "y": 1.02, "yanchor": "bottom"},
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig_to_html(fig)


def chart_spillage_by_bus(
    hydros: pd.DataFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Total spillage by bus by stage."""
    h0 = hydros[hydros["block_id"] == 0].copy()
    h0["bus_id"] = h0["hydro_id"].map(hydro_bus_map)
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        spill = (
            h0[h0["bus_id"] == bus_id]
            .groupby(["scenario_id", "stage_id"])["spillage_m3s"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[spill.get(s, 0) for s in stages],
                name=bus_names.get(bus_id, str(bus_id)),
                line={"color": BUS_COLORS[i % len(BUS_COLORS)], "width": 2},
            )
        )
    fig.update_layout(
        title="Total Spillage by Bus (avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Spillage (m³/s)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_water_value_by_bus(
    hydros: pd.DataFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Average water value by bus by stage."""
    h0 = hydros[hydros["block_id"] == 0].copy()
    h0["bus_id"] = h0["hydro_id"].map(hydro_bus_map)
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        wv = (
            h0[h0["bus_id"] == bus_id]
            .groupby(["scenario_id", "stage_id"])["water_value_per_hm3"]
            .mean()
            .groupby("stage_id")
            .mean()
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[wv.get(s, 0) for s in stages],
                name=bus_names.get(bus_id, str(bus_id)),
                line={"color": BUS_COLORS[i % len(BUS_COLORS)], "width": 2},
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Average Water Value by Bus (avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Water Value (R$/hm³)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_deficit_by_bus(
    buses: pd.DataFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
) -> str:
    """Deficit by bus by stage (stage-avg MW)."""
    real_buses = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(buses["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(real_buses):
        bdata = buses[buses["bus_id"] == bus_id]
        def_mw = _stage_avg_mw(bdata, "deficit_mwh", stage_hours, [])
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[def_mw.get(s, 0) for s in stages],
                name=bus_names.get(bus_id, str(bus_id)),
                line={"color": BUS_COLORS[i % len(BUS_COLORS)], "width": 2},
            )
        )
    fig.update_layout(
        title="Deficit by Bus (stage-avg MW, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Deficit (MW)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_excess_by_bus(
    buses: pd.DataFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
) -> str:
    """Excess by bus by stage (stage-avg MW)."""
    real_buses = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(buses["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(real_buses):
        bdata = buses[buses["bus_id"] == bus_id]
        exc_mw = _stage_avg_mw(bdata, "excess_mwh", stage_hours, [])
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[exc_mw.get(s, 0) for s in stages],
                name=bus_names.get(bus_id, str(bus_id)),
                line={"color": BUS_COLORS[i % len(BUS_COLORS)], "width": 2},
            )
        )
    fig.update_layout(
        title="Excess by Bus (stage-avg MW)",
        xaxis_title="Stage",
        yaxis_title="Excess (MW)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_top_hydros_detail(
    hydros: pd.DataFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    top_n: int = 8,
) -> str:
    """Generation, storage, and spillage timeseries for top hydro plants."""
    ranked = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]
    hids = [hid for hid, _ in ranked]

    h0 = hydros[hydros["block_id"] == 0]
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Generation (MW)", "Storage (hm³)", "Spillage (m³/s)"],
        vertical_spacing=0.10,
    )

    palette = [
        "#2196F3",
        "#FF9800",
        "#4CAF50",
        "#F44336",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
        "#607D8B",
    ]

    for i, hid in enumerate(hids):
        meta = hydro_meta[hid]
        name = meta["name"]
        color = palette[i % len(palette)]
        hdata = h0[h0["hydro_id"] == hid]

        gen = (
            hdata.groupby(["scenario_id", "stage_id"])["generation_mw"]
            .mean()
            .groupby("stage_id")
            .mean()
        )
        stor = (
            hdata.groupby(["scenario_id", "stage_id"])["storage_final_hm3"]
            .mean()
            .groupby("stage_id")
            .mean()
        )
        spill = (
            hdata.groupby(["scenario_id", "stage_id"])["spillage_m3s"]
            .mean()
            .groupby("stage_id")
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[gen.get(s, 0) for s in stages],
                name=name,
                legendgroup=name,
                line={"color": color, "width": 1.5},
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[stor.get(s, 0) for s in stages],
                name=name,
                legendgroup=name,
                line={"color": color, "width": 1.5},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[spill.get(s, 0) for s in stages],
                name=name,
                legendgroup=name,
                line={"color": color, "width": 1.5},
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=f"Top {top_n} Hydro Plants by Installed Capacity",
        height=900,
        legend={**_LEGEND, "y": 1.04},
        margin=dict(l=60, r=30, t=100, b=50),
    )
    return fig_to_html(fig)


def build_top_hydros_table(
    hydros: pd.DataFrame,
    hydro_meta: dict[int, dict],
    bus_names: dict[int, str],
    top_n: int = 20,
) -> str:
    """HTML table of top hydro plants with key simulation metrics."""
    ranked = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]
    h0 = hydros[hydros["block_id"] == 0]

    rows_html = []
    for hid, meta in ranked:
        hdata = h0[h0["hydro_id"] == hid]
        avg_gen = (
            hdata.groupby("scenario_id")["generation_mw"].mean().mean()
            if not hdata.empty
            else 0
        )
        avg_spill = (
            hdata.groupby("scenario_id")["spillage_m3s"].mean().mean()
            if not hdata.empty
            else 0
        )
        avg_wv = hdata["water_value_per_hm3"].mean() if not hdata.empty else 0
        avg_stor = hdata["storage_final_hm3"].mean() if not hdata.empty else 0
        bus = bus_names.get(meta["bus_id"], str(meta["bus_id"]))

        rows_html.append(
            f"<tr><td>{meta['name']}</td><td>{bus}</td>"
            f"<td>{meta['max_gen_mw']:.0f}</td>"
            f"<td>{meta['vol_max']:.0f}</td>"
            f"<td>{avg_gen:.0f}</td>"
            f"<td>{avg_stor:.0f}</td>"
            f"<td>{avg_spill:.0f}</td>"
            f"<td>{avg_wv:,.0f}</td></tr>"
        )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        "<th>Plant</th><th>Bus</th><th>Max Gen (MW)</th><th>Vol Max (hm³)</th>"
        "<th>Avg Gen (MW)</th><th>Avg Storage (hm³)</th>"
        "<th>Avg Spillage (m³/s)</th><th>Avg Water Value (R$/hm³)</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


# ---------------------------------------------------------------------------
# Tab 8: Investigation charts
# ---------------------------------------------------------------------------

M3S_TO_HM3_PER_HOUR = 3600 / 1e6


def chart_per_block_balance(
    hydros: pd.DataFrame,
    thermals: pd.DataFrame,
    ncs: pd.DataFrame,
    buses: pd.DataFrame,
    stage_labels: dict[int, str],
    block_hours: dict[tuple[int, int], float],
) -> str:
    """Generation vs Load per block across stages (avg across scenarios)."""
    blocks = sorted({b for _, b in block_hours.keys()})
    block_names = {0: "Heavy", 1: "Medium", 2: "Light"}

    fig = make_subplots(
        rows=len(blocks),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[block_names.get(b, f"Block {b}") for b in blocks],
        vertical_spacing=0.10,
    )

    stages = sorted({s for s, _ in block_hours.keys()})
    xlabels = stage_x_labels(stages, stage_labels)

    for row, blk in enumerate(blocks, 1):
        show_legend = row == 1

        hb = hydros[hydros["block_id"] == blk]
        h_gen = (
            hb.groupby(["scenario_id", "stage_id"])["generation_mw"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        tb = thermals[thermals["block_id"] == blk]
        t_gen = (
            tb.groupby(["scenario_id", "stage_id"])["generation_mw"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        nb = ncs[ncs["block_id"] == blk]
        n_gen = (
            nb.groupby(["scenario_id", "stage_id"])["generation_mw"]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        bb = buses[(buses["block_id"] == blk) & (buses["bus_id"].isin([0, 1, 2, 3]))]
        load = (
            bb.groupby(["scenario_id", "stage_id"])["load_mw"]
            .sum()
            .groupby("stage_id")
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[h_gen.get(s, 0) for s in stages],
                name="Hydro",
                stackgroup=f"g{blk}",
                fillcolor="rgba(33,150,243,0.6)",
                line={"color": COLORS["hydro"]},
                legendgroup="hydro",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[t_gen.get(s, 0) for s in stages],
                name="Thermal",
                stackgroup=f"g{blk}",
                fillcolor="rgba(255,152,0,0.6)",
                line={"color": COLORS["thermal"]},
                legendgroup="thermal",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[n_gen.get(s, 0) for s in stages],
                name="NCS",
                stackgroup=f"g{blk}",
                fillcolor="rgba(76,175,80,0.6)",
                line={"color": COLORS["ncs"]},
                legendgroup="ncs",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[load.get(s, 0) for s in stages],
                name="Load",
                mode="lines",
                line={"color": COLORS["load"], "width": 2, "dash": "dash"},
                legendgroup="load",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="MW", row=row, col=1)

    fig.update_layout(
        title="Generation vs Load by Block (avg across scenarios)",
        height=300 * len(blocks),
        legend={**_LEGEND, "y": 1.05},
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig_to_html(fig)


def chart_inflow_comparison(
    hydros: pd.DataFrame,
    inflow_stats: pd.DataFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    top_n: int = 6,
) -> str:
    """Compare realized inflow (p10/p50/p90) with historical mean +/- std."""
    ranked = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]

    h0 = hydros[hydros["block_id"] == 0]
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = make_subplots(
        rows=top_n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[meta["name"] for _, meta in ranked],
        vertical_spacing=0.10,
    )

    for row, (hid, meta) in enumerate(ranked, 1):
        show_legend = row == 1
        hdata = h0[h0["hydro_id"] == hid]

        realized = hdata.groupby(["scenario_id", "stage_id"])["inflow_m3s"].mean()
        pcts = (
            realized.groupby("stage_id")
            .quantile([0.1, 0.5, 0.9])
            .unstack(level=-1)
            .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
        )

        hist = inflow_stats[inflow_stats["hydro_id"] == hid].set_index("stage_id")

        p90 = [pcts["p90"].get(s, 0) for s in stages]
        p10 = [pcts["p10"].get(s, 0) for s in stages]
        p50 = [pcts["p50"].get(s, 0) for s in stages]
        hist_mean = [
            hist["mean_m3s"].get(s, 0) if s in hist.index else 0 for s in stages
        ]
        hist_upper = [
            (hist["mean_m3s"].get(s, 0) + hist["std_m3s"].get(s, 0))
            if s in hist.index
            else 0
            for s in stages
        ]
        hist_lower = [
            max(0, hist["mean_m3s"].get(s, 0) - hist["std_m3s"].get(s, 0))
            if s in hist.index
            else 0
            for s in stages
        ]

        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=hist_upper + hist_lower[::-1],
                fill="toself",
                fillcolor="rgba(255,152,0,0.12)",
                line={"color": "rgba(255,255,255,0)"},
                name="Historical ±1\u03c3",
                legendgroup="hist_band",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=hist_mean,
                name="Historical Mean",
                mode="lines",
                line={"color": COLORS["thermal"], "width": 1.5, "dash": "dash"},
                legendgroup="hist_mean",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(33,150,243,0.12)",
                line={"color": "rgba(255,255,255,0)"},
                name="Realized P10-P90",
                legendgroup="real_band",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p50,
                name="Realized Median",
                mode="lines",
                line={"color": COLORS["hydro"], "width": 2},
                legendgroup="real_med",
                showlegend=show_legend,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="m³/s", row=row, col=1)

    fig.update_layout(
        title=f"Realized Inflow vs Historical Statistics (top {top_n} hydros)",
        height=220 * top_n,
        legend={**_LEGEND, "y": 1.04},
        margin=dict(l=60, r=30, t=100, b=50),
    )
    return fig_to_html(fig)


def chart_plant_water_balance(
    hydros: pd.DataFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
    top_n: int = 6,
) -> str:
    """Water balance components for top hydro plants.

    Shows storage trajectory, flow rates, and balance residual.
    """
    # Pick plants with most spillage as they are the most suspicious
    h0 = hydros[hydros["block_id"] == 0]
    avg_spill = (
        h0.groupby(["scenario_id", "hydro_id"])["spillage_m3s"]
        .mean()
        .groupby("hydro_id")
        .mean()
    )
    # Also pick top by capacity, merge both lists
    by_cap = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]
    by_spill = avg_spill.nlargest(top_n).index.tolist()
    hids = list(dict.fromkeys([h for h, _ in by_cap] + by_spill))[:top_n]

    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Storage (hm³)",
            "Inflow / Outflow (m³/s)",
            "Water Balance Residual (hm³)",
        ],
        vertical_spacing=0.10,
    )

    palette = [
        "#2196F3",
        "#FF9800",
        "#4CAF50",
        "#F44336",
        "#9C27B0",
        "#00BCD4",
        "#FF5722",
        "#607D8B",
    ]

    for i, hid in enumerate(hids):
        name = hydro_meta.get(hid, {}).get("name", str(hid))
        color = palette[i % len(palette)]
        hdata = hydros[hydros["hydro_id"] == hid]

        # Storage: from block 0
        hb0 = hdata[hdata["block_id"] == 0]
        stor = (
            hb0.groupby(["scenario_id", "stage_id"])["storage_final_hm3"]
            .mean()
            .groupby("stage_id")
            .mean()
        )

        # Inflow (stage-level, same for all blocks)
        inflow = (
            hb0.groupby(["scenario_id", "stage_id"])["inflow_m3s"]
            .mean()
            .groupby("stage_id")
            .mean()
        )

        # Total outflow: sum across blocks of (turbined + spillage) weighted by tau
        outflow_vals: dict[int, float] = {}
        residual_vals: dict[int, float] = {}
        for s in stages:
            s_data = hdata[hdata["stage_id"] == s]
            if s_data.empty:
                outflow_vals[s] = 0
                residual_vals[s] = 0
                continue

            # Average across scenarios
            scen_out = []
            scen_res = []
            for scen_id in s_data["scenario_id"].unique():
                ss = s_data[s_data["scenario_id"] == scen_id]
                v_in = ss["storage_initial_hm3"].iloc[0]
                v_out = ss["storage_final_hm3"].iloc[0]
                zeta = stage_hours.get(s, 744) * M3S_TO_HM3_PER_HOUR
                inf = ss["inflow_m3s"].iloc[0]
                evap = ss["evaporation_m3s"].iloc[0]

                total_out_vol = 0.0
                for _, row in ss.iterrows():
                    blk = int(row["block_id"])
                    tau = block_hours.get((s, blk), 0) * M3S_TO_HM3_PER_HOUR
                    total_out_vol += tau * (row["turbined_m3s"] + row["spillage_m3s"])

                out_m3s = total_out_vol / max(zeta, 1e-9)
                scen_out.append(out_m3s)
                res = (v_out - v_in) - zeta * (inf - evap) + total_out_vol
                scen_res.append(res)

            outflow_vals[s] = float(np.mean(scen_out))
            residual_vals[s] = float(np.mean(scen_res))

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[stor.get(s, 0) for s in stages],
                name=name,
                legendgroup=name,
                line={"color": color, "width": 1.5},
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[inflow.get(s, 0) for s in stages],
                name=f"{name} (inflow)",
                legendgroup=name,
                line={"color": color, "width": 1.5},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[outflow_vals.get(s, 0) for s in stages],
                name=f"{name} (outflow)",
                legendgroup=name,
                line={"color": color, "width": 1.5, "dash": "dash"},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[residual_vals.get(s, 0) for s in stages],
                name=f"{name} (res)",
                legendgroup=name,
                line={"color": color, "width": 1.5},
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5, row=3, col=1)
    fig.update_yaxes(title_text="hm³", row=1, col=1)
    fig.update_yaxes(title_text="m³/s", row=2, col=1)
    fig.update_yaxes(title_text="hm³", row=3, col=1)

    fig.update_layout(
        title=f"Water Balance Detail (top {top_n} plants, avg across scenarios)",
        height=900,
        legend={**_LEGEND, "y": 1.04},
        margin=dict(l=60, r=30, t=100, b=50),
    )
    return fig_to_html(fig)


def chart_violation_summary(
    hydros: pd.DataFrame,
    stage_labels: dict[int, str],
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

    h0 = hydros[hydros["block_id"] == 0]
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for col, (label, color) in violation_cols.items():
        if col not in h0.columns:
            continue
        total = (
            h0.groupby(["scenario_id", "stage_id"])[col]
            .sum()
            .groupby("stage_id")
            .mean()
        )
        if total.sum() < 1e-6:
            continue
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[total.get(s, 0) for s in stages],
                name=label,
                line={"color": color, "width": 2},
            )
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
    hydros: pd.DataFrame,
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
    existing = [c for c in violation_cols if c in hydros.columns]
    if not existing:
        return "<p>No violation data available.</p>"

    h0 = hydros[hydros["block_id"] == 0].copy()
    h0["_total_viol"] = h0[existing].abs().sum(axis=1)

    # Top plants by total violations
    top_hydros = (
        h0.groupby("hydro_id")["_total_viol"].sum().nlargest(top_n).index.tolist()
    )
    top_hydros = [
        h for h in top_hydros if h0[h0["hydro_id"] == h]["_total_viol"].sum() > 1e-6
    ]
    if not top_hydros:
        return "<p>No significant violations detected.</p>"

    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)
    ynames = [entity_name(names, "hydros", hid) for hid in top_hydros]

    z = []
    for hid in top_hydros:
        hdata = h0[h0["hydro_id"] == hid]
        row = []
        for s in stages:
            sdata = hdata[hdata["stage_id"] == s]
            val = sdata["_total_viol"].mean() if not sdata.empty else 0
            row.append(val)
        z.append(row)

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
        title=f"Constraint Violations Heatmap (top {len(top_hydros)} plants)",
        height=max(350, len(top_hydros) * 25 + 120),
        margin=dict(l=120, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab: Training Insights charts
# ---------------------------------------------------------------------------


def chart_gap_evolution(conv: pd.DataFrame) -> str:
    """Line chart of gap_percent by iteration with a zero reference line."""
    if conv.empty:
        return "<p>No convergence data available.</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["gap_percent"],
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
        title="Convergence Gap (%) per Iteration",
        xaxis_title="Iteration",
        yaxis_title="Gap (%)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
    )
    return fig_to_html(fig)


def chart_cut_state_evolution(conv: pd.DataFrame) -> str:
    """Stacked area for cuts_active + bars for cuts_added per iteration."""
    if conv.empty:
        return "<p>No convergence data available.</p>"

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=conv["iteration"],
            y=conv["cuts_active"],
            name="Cuts Active",
            fill="tozeroy",
            fillcolor="rgba(74,144,184,0.25)",
            line={"color": COLORS["hydro"], "width": 2},
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(
            x=conv["iteration"],
            y=conv["cuts_added"],
            name="Cuts Added",
            marker_color="rgba(245,166,35,0.7)",
            opacity=0.8,
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Cut Pool Evolution",
        xaxis_title="Iteration",
        legend=_LEGEND,
        margin=_MARGIN,
        height=420,
        barmode="overlay",
    )
    fig.update_yaxes(title_text="Cuts Active", secondary_y=False)
    fig.update_yaxes(title_text="Cuts Added (per iter)", secondary_y=True)
    return fig_to_html(fig)


def chart_cut_activity_heatmap(
    cut_selection: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=iteration, y=stage, z=cuts_active_after. YlOrRd colorscale."""
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    cs = cut_selection[cut_selection["stage"] > 0]
    pivot = cs.pivot_table(
        index="stage", columns="iteration", values="cuts_active_after", aggfunc="sum"
    )
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
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Cuts Active: %{z}<extra></extra>",
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
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_cut_deactivation_heatmap(
    cut_selection: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=iteration, y=stage, z=cuts_deactivated. Blues colorscale."""
    if cut_selection.empty:
        return "<p>No cut selection data available.</p>"

    pivot = cut_selection.pivot_table(
        index="stage", columns="iteration", values="cuts_deactivated", aggfunc="sum"
    )
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
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Cuts Deactivated: %{z}<extra></extra>",
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
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_simplex_heatmap(
    solver_train: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=SDDP iteration, y=stage, z=simplex_iterations (backward phase)."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[
        (solver_train["phase"] == "backward") & (solver_train["stage"] >= 0)
    ]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    pivot = bwd.pivot_table(
        index="stage", columns="iteration", values="simplex_iterations", aggfunc="sum"
    )
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
            colorscale="Viridis",
            colorbar={"title": "Simplex Iters"},
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Simplex Iterations: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Simplex Iterations per Stage (Backward Pass)",
        xaxis_title="SDDP Iteration",
        yaxis_title="Stage",
        yaxis={
            "autorange": "reversed",
            "dtick": 12,
            "tickvals": ytick_vals,
            "ticktext": ytick_text,
        },
        height=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


def chart_solve_time_heatmap(
    solver_train: pd.DataFrame, stage_labels: dict[int, str]
) -> str:
    """Heatmap x=SDDP iteration, y=stage, z=solve_time_ms (backward phase). Hot reversed."""
    if solver_train.empty:
        return "<p>No solver data available.</p>"

    bwd = solver_train[
        (solver_train["phase"] == "backward") & (solver_train["stage"] >= 0)
    ]
    if bwd.empty:
        return "<p>No backward solver data available.</p>"

    pivot = bwd.pivot_table(
        index="stage", columns="iteration", values="solve_time_ms", aggfunc="sum"
    )
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
            colorscale="Hot",
            reversescale=True,
            colorbar={"title": "ms"},
            hovertemplate="Iteration: %{x}<br>Stage: %{y}<br>Solve Time: %{z:.1f} ms<extra></extra>",
        )
    )
    fig.update_layout(
        title="LP Solve Time per Stage (Backward Pass, ms)",
        xaxis_title="SDDP Iteration",
        yaxis_title="Stage",
        yaxis={
            "autorange": "reversed",
            "dtick": 12,
            "tickvals": ytick_vals,
            "ticktext": ytick_text,
        },
        height=500,
        margin=dict(l=80, r=30, t=60, b=50),
    )
    return fig_to_html(fig, unified_hover=False)


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


# ---------------------------------------------------------------------------
# Tab 9: Performance charts
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


def build_performance_metrics_html(
    conv: pd.DataFrame,
    timing: pd.DataFrame,
    solver_train: pd.DataFrame,
    solver_sim: pd.DataFrame,
    scaling_report: dict,
    metadata: dict,
) -> str:
    """Summary metric cards for the Performance tab."""
    # Total training time
    # Prefer wall-clock from metadata.json; fall back to summing iteration times.
    meta_duration = metadata.get("run_info", {}).get("duration_seconds")
    if meta_duration and meta_duration > 0:
        total_train_s = float(meta_duration)
    else:
        total_train_ms = (
            conv["time_total_ms"].sum() if "time_total_ms" in conv.columns else 0.0
        )
        if total_train_ms == 0 and not timing.empty:
            time_cols = [
                c for c in timing.columns if c.endswith("_ms") and c != "iteration"
            ]
            total_train_ms = timing[time_cols].sum().sum()
        total_train_s = total_train_ms / 1000.0

    # Total simulation time — prefer wall-clock from manifest, fall back to CPU sum.
    sim_manifest_duration = metadata.get("_sim_manifest", {}).get("duration_seconds")
    if sim_manifest_duration and sim_manifest_duration > 0:
        total_sim_s = float(sim_manifest_duration)
        total_sim_ms = total_sim_s * 1000.0
        sim_is_wallclock = True
    else:
        total_sim_ms = (
            solver_sim["solve_time_ms"].sum() if not solver_sim.empty else 0.0
        )
        total_sim_s = total_sim_ms / 1000.0
        sim_is_wallclock = False

    # Avg LP solve time — training
    if not solver_train.empty:
        total_lp_solves = solver_train["lp_solves"].sum()
        total_lp_time = solver_train["solve_time_ms"].sum()
        avg_lp_train_ms = (
            total_lp_time / total_lp_solves if total_lp_solves > 0 else 0.0
        )
    else:
        total_lp_solves = 0
        avg_lp_train_ms = 0.0

    # Avg LP solve time — simulation
    if not solver_sim.empty:
        sim_lp_solves = solver_sim["lp_solves"].sum()
        sim_lp_time = solver_sim["solve_time_ms"].sum()
        avg_lp_sim_ms = sim_lp_time / sim_lp_solves if sim_lp_solves > 0 else 0.0
    else:
        avg_lp_sim_ms = 0.0

    # Total simplex iterations (training + simulation)
    train_simplex = (
        int(solver_train["simplex_iterations"].sum()) if not solver_train.empty else 0
    )
    sim_simplex = (
        int(solver_sim["simplex_iterations"].sum()) if not solver_sim.empty else 0
    )
    total_simplex = train_simplex + sim_simplex

    # LP dimensions from scaling report
    stages_sr = scaling_report.get("stages", [])
    max_nz = max((s["dimensions"]["num_nz"] for s in stages_sr), default=0)

    # Format training time
    if total_train_s >= 3600:
        train_str = f"{total_train_s / 3600:.2f} h"
    elif total_train_s >= 60:
        train_str = f"{total_train_s / 60:.1f} min"
    else:
        train_str = f"{total_train_s:.1f} s"

    if total_sim_s >= 3600:
        sim_str = f"{total_sim_s / 3600:.2f} h"
    elif total_sim_s >= 60:
        sim_str = f"{total_sim_s / 60:.1f} min"
    else:
        sim_str = f"{total_sim_s:.1f} s"
    sim_label = "Total Simulation Time" if sim_is_wallclock else "Simulation CPU Time"

    # Total LP solves across everything
    all_lp_solves = (
        int(solver_train["lp_solves"].sum()) if not solver_train.empty else 0
    ) + (int(solver_sim["lp_solves"].sum()) if not solver_sim.empty else 0)

    metrics = [
        ("Total Training Time", train_str, COLORS["lower_bound"]),
        (sim_label, sim_str, COLORS["ncs"]),
        ("Avg LP Solve (training)", f"{avg_lp_train_ms:.2f} ms", COLORS["hydro"]),
        ("Avg LP Solve (simulation)", f"{avg_lp_sim_ms:.2f} ms", COLORS["spillage"]),
        ("Total LP Solves", f"{all_lp_solves:,}", COLORS["thermal"]),
        ("Total Simplex Iterations", f"{total_simplex:,}", COLORS["spillage"]),
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
# Tab 10: Generic Constraints — expression parser + charts
# ---------------------------------------------------------------------------

import re as _re

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


def evaluate_constraint_expressions(
    constraints: list[dict],
    hydros_df: pd.DataFrame,
    exchanges_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate LHS of all generic constraints from simulation output.

    Variable lookups:
    - ``hydro_storage(id)``    → ``storage_final_hm3`` where hydro_id=id, block_id=0
    - ``hydro_generation(id)`` → ``generation_mw``     where hydro_id=id  (per block)
    - ``line_exchange(id)``    → ``net_flow_mw``        where line_id=id   (per block)

    Storage-only constraints produce one row per (scenario, stage) with block_id=0.
    Mixed / generation / exchange constraints produce one row per (scenario, stage, block).

    Returns DataFrame with columns:
        constraint_id, scenario_id, stage_id, block_id, lhs_value
    """
    # Convert to polars for fast grouped operations
    try:
        import polars as pl
    except ImportError:
        pl = None  # fall through to pandas path

    h0_pd = hydros_df[hydros_df["block_id"] == 0][
        ["scenario_id", "stage_id", "hydro_id", "storage_final_hm3"]
    ].copy()
    hg_pd = hydros_df[
        ["scenario_id", "stage_id", "block_id", "hydro_id", "generation_mw"]
    ].copy()
    ex_pd = exchanges_df[
        ["scenario_id", "stage_id", "block_id", "line_id", "net_flow_mw"]
    ].copy()

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
        vertical_spacing=0.12,
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
    yaxis_title = type_labels.get(ctype_filter, "Value")
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
        legend={**_LEGEND, "y": 1.02},
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


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'IBM Plex Sans', 'Inter', 'Segoe UI', system-ui, sans-serif; background: #F0EDE8; color: #374151; }

header {
    background: linear-gradient(135deg, #0F1419 0%, #1A2028 100%);
    border-top: 3px solid #B87333;
    color: white;
    padding: 16px 32px;
    font-size: 1.3rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

nav {
    background: #1A2028;
    padding: 0 24px;
    display: flex;
    gap: 4px;
    overflow-x: auto;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

nav button {
    background: none;
    border: none;
    color: #8B9298;
    padding: 14px 20px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    border-bottom: 3px solid transparent;
    white-space: nowrap;
    transition: color 0.2s, border-color 0.2s;
    letter-spacing: 0.3px;
}

nav button:hover { color: #E8E6E3; border-bottom-color: #B87333; }
nav button.active { color: #B87333; border-bottom-color: #B87333; }

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
    background: #FAFAF8;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    overflow: hidden;
    min-width: 0;
}

.chart-card .plotly-graph-div { width: 100% !important; }
.chart-card .js-plotly-plot { width: 100% !important; }
.chart-card .plot-container { width: 100% !important; }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
}

.metric-card {
    background: #FAFAF8;
    border-radius: 8px;
    padding: 20px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #374151;
    margin-bottom: 6px;
}

.metric-label {
    font-size: 0.8rem;
    color: #8B9298;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.section-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    margin: 24px 0 12px;
    padding-left: 10px;
    border-left: 4px solid #B87333;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
    background: #FAFAF8;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.data-table th, .data-table td {
    padding: 10px 12px;
    text-align: right;
    border-bottom: 1px solid #E0E0E0;
}
.data-table th {
    background: #374151;
    color: white;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
.data-table td:first-child, .data-table th:first-child { text-align: left; }
.data-table td:nth-child(2), .data-table th:nth-child(2) { text-align: left; }
.data-table tr:hover td { background: #F0EDE8; }
"""

JS = """
function showTab(tabId, btn) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');
    window.dispatchEvent(new Event('resize'));
}
// Plotly charts in the initial active tab render before layout settles.
// Fire a deferred resize so they recalculate to the correct container width.
window.addEventListener('load', function() { setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 50); });
"""

TAB_DEFS = [
    ("tab-overview", "Overview"),
    ("tab-training", "Training Insights"),
    ("tab-energy", "Energy Balance"),
    ("tab-hydro", "Hydro Operations"),
    ("tab-plants", "Plant Details"),
    ("tab-thermal", "Thermal Operations"),
    ("tab-thermal-plants", "Thermal Plant Details"),
    ("tab-exchanges", "Exchanges"),
    ("tab-costs", "Costs"),
    ("tab-ncs-thermal", "NCS"),
    ("tab-constraints", "Constraints"),
    ("tab-perf", "Performance"),
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
# Plant Details Interactive Component
# ---------------------------------------------------------------------------


def build_interactive_plant_details(
    hydros: pd.DataFrame,
    hydro_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    lp_bounds: pd.DataFrame | None = None,
) -> str:
    """Build HTML with embedded per-hydro p10/p50/p90 data, LP bounds, and JS dropdown."""
    h0 = hydros[hydros["block_id"] == 0]
    stages = sorted(h0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    metrics = [
        "generation_mw",
        "storage_final_hm3",
        "spillage_m3s",
        "inflow_m3s",
        "turbined_m3s",
        "water_value_per_hm3",
        "evaporation_m3s",
        "outflow_m3s",
    ]
    short = {
        "generation_mw": "gen",
        "storage_final_hm3": "stor",
        "spillage_m3s": "spill",
        "inflow_m3s": "inflow",
        "turbined_m3s": "turb",
        "water_value_per_hm3": "wv",
        "evaporation_m3s": "evap",
        "outflow_m3s": "outflow",
    }

    hydro_data: dict[str, dict] = {}
    for hid, meta in sorted(hydro_meta.items()):
        hdata = h0[h0["hydro_id"] == hid]
        if hdata.empty:
            continue
        entry: dict = {
            "name": meta["name"],
            "bus": bus_names.get(meta["bus_id"], str(meta["bus_id"])),
            "vol_max": round(meta["vol_max"], 1),
            "vol_min": round(meta["vol_min"], 1),
            "max_gen": round(meta["max_gen_mw"], 1),
            "max_gen_phys": round(meta.get("max_gen_physical", meta["max_gen_mw"]), 1),
            "max_turb": round(meta["max_turbined"], 1),
        }
        for col in metrics:
            k = short[col]
            if col not in hdata.columns:
                for sfx in ["p10", "p50", "p90"]:
                    entry[f"{k}_{sfx}"] = [0.0] * len(stages)
                continue
            grp = hdata.groupby(["scenario_id", "stage_id"])[col].mean()
            pcts = (
                grp.groupby("stage_id")
                .quantile([0.1, 0.5, 0.9])
                .unstack(level=-1)
                .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
            )
            for sfx in ["p10", "p50", "p90"]:
                entry[f"{k}_{sfx}"] = [
                    round(float(pcts[sfx].get(s, 0)), 2) for s in stages
                ]
        hydro_data[str(hid)] = entry

    # Embed per-hydro per-stage LP bounds (from solver output)
    # bound_type: 0=storage_min, 1=storage_max, 2=turbined_min, 3=turbined_max,
    #             4=outflow_min, 6=generation_min, 7=generation_max
    if lp_bounds is not None and not lp_bounds.empty:
        hb = lp_bounds[lp_bounds["entity_type_code"] == 0]  # hydro
        bound_keys = {
            0: "stor_min",
            1: "stor_max",
            2: "turb_min",
            3: "turb_max",
            4: "outflow_min",
            6: "gen_min",
            7: "gen_max",
        }
        for hid_str, entry in hydro_data.items():
            hid_int = int(hid_str)
            hb_plant = hb[hb["entity_id"] == hid_int]
            for bt_code, key in bound_keys.items():
                bt_rows = hb_plant[hb_plant["bound_type_code"] == bt_code]
                if bt_rows.empty:
                    entry[key] = [0.0] * len(stages)
                else:
                    by_stage = bt_rows.set_index("stage_id")["bound_value"]
                    entry[key] = [round(float(by_stage.get(s, 0)), 2) for s in stages]

    options = sorted(hydro_data.items(), key=lambda x: x[1]["name"])
    options_html = "\n".join(
        f'<option value="{hid}">{d["name"]} (id={hid})</option>' for hid, d in options
    )
    data_json = json.dumps(hydro_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    chart_rows = (
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="hd-gen" style="width:100%;height:350px;"></div></div>'
        "</div>"
        '<div class="chart-grid">'
        '<div class="chart-card"><div id="hd-stor" style="width:100%;height:350px;"></div></div>'
        '<div class="chart-card"><div id="hd-inflow" style="width:100%;height:350px;"></div></div>'
        "</div>"
        '<div class="chart-grid">'
        '<div class="chart-card"><div id="hd-turb" style="width:100%;height:350px;"></div></div>'
        '<div class="chart-card"><div id="hd-spill" style="width:100%;height:350px;"></div></div>'
        "</div>"
        '<div class="chart-grid">'
        '<div class="chart-card"><div id="hd-wv" style="width:100%;height:350px;"></div></div>'
        '<div class="chart-card"><div id="hd-misc" style="width:100%;height:350px;"></div></div>'
        "</div>"
    )

    return (
        '<div style="margin-bottom:16px;">'
        '<label for="hydro-select" style="font-weight:600;margin-right:8px;">Select Hydro Plant:</label>'
        '<select id="hydro-select" onchange="updateHydroDetail()" '
        'style="padding:8px 12px;font-size:0.9rem;border-radius:4px;border:1px solid #ccc;min-width:300px;">'
        + options_html
        + "</select>"
        + '<span id="hydro-info" style="margin-left:16px;color:#666;font-size:0.85rem;"></span>'
        + "</div>"
        + chart_rows
        + "<script>\n"
        + "const HD = "
        + data_json
        + ";\n"
        + "const HD_LABELS = "
        + labels_json
        + ";\n"
        + r"""
function _band(lbl, p10, p90, color) {
  return {x: HD_LABELS.concat(HD_LABELS.slice().reverse()),
          y: p90.concat(p10.slice().reverse()),
          fill:'toself', fillcolor:color, line:{color:'rgba(0,0,0,0)'},
          name:lbl, showlegend:true, hoverinfo:'skip'};
}
function _line(nm, y, c, w, dash) {
  return {x:HD_LABELS, y:y, name:nm, line:{color:c, width:w||2, dash:dash||'solid'}};
}
function _ref(nm, val, c) {
  return {x:HD_LABELS, y:Array(HD_LABELS.length).fill(val), name:nm,
          line:{color:c, width:1, dash:'dot'}};
}
var _L = {hovermode:'x unified', margin:{l:60,r:20,t:50,b:60},
           legend:{orientation:'h',y:1.12,x:0,font:{size:11}}};
var _C = {responsive:true};
function _lo(extra){return Object.assign({},_L,extra);}

function updateHydroDetail() {
  var hid = document.getElementById('hydro-select').value;
  var d = HD[hid]; if(!d) return;
  document.getElementById('hydro-info').textContent =
    d.bus+' | Gen: '+d.max_gen_phys.toFixed(0)+' MW | Turb: '+d.max_turb.toFixed(0)+
    ' m\u00b3/s | Vol: '+d.vol_min.toFixed(0)+'\u2013'+d.vol_max.toFixed(0)+' hm\u00b3';

  Plotly.react('hd-gen', [
    _band('P10-P90', d.gen_p10, d.gen_p90, 'rgba(74,144,184,0.15)'),
    _line('P50', d.gen_p50, '#4A90B8'),
    _line('P10', d.gen_p10, '#4A90B8', 1, 'dot'),
    _line('P90', d.gen_p90, '#4A90B8', 1, 'dot'),
    _line('Effective Capacity', d.gen_max || Array(HD_LABELS.length).fill(d.max_gen_phys), '#DC4C4C', 1, 'dash'),
  ], _lo({title:d.name+' \u2014 Generation (MW)', yaxis:{title:'MW'}}), _C);

  Plotly.react('hd-stor', [
    _band('P10-P90', d.stor_p10, d.stor_p90, 'rgba(74,144,184,0.15)'),
    _line('P50', d.stor_p50, '#4A90B8'),
    _line('P10', d.stor_p10, '#4A90B8', 1, 'dot'),
    _line('P90', d.stor_p90, '#4A90B8', 1, 'dot'),
    _ref('Vol Max', d.vol_max, '#DC4C4C'),
    _ref('Vol Min', d.vol_min, '#4A8B6F'),
  ], _lo({title:'Storage (hm\u00b3)', yaxis:{title:'hm\u00b3'}}), _C);

  Plotly.react('hd-inflow', [
    _band('P10-P90', d.inflow_p10, d.inflow_p90, 'rgba(74,139,111,0.15)'),
    _line('P50', d.inflow_p50, '#4A8B6F'),
    _line('P10', d.inflow_p10, '#4A8B6F', 1, 'dot'),
    _line('P90', d.inflow_p90, '#4A8B6F', 1, 'dot'),
  ], _lo({title:'Inflow (m\u00b3/s)', yaxis:{title:'m\u00b3/s'}}), _C);

  Plotly.react('hd-turb', [
    _band('P10-P90', d.turb_p10, d.turb_p90, 'rgba(245,166,35,0.15)'),
    _line('P50', d.turb_p50, '#F5A623'),
    _line('P10', d.turb_p10, '#F5A623', 1, 'dot'),
    _line('P90', d.turb_p90, '#F5A623', 1, 'dot'),
    _line('Turb Max (LP)', d.turb_max || Array(HD_LABELS.length).fill(d.max_turb), '#DC4C4C', 1, 'dash'),
  ], _lo({title:'Turbined (m\u00b3/s)', yaxis:{title:'m\u00b3/s'}}), _C);

  Plotly.react('hd-spill', [
    _band('P10-P90', d.spill_p10, d.spill_p90, 'rgba(184,115,51,0.15)'),
    _line('P50', d.spill_p50, '#B87333'),
    _line('P10', d.spill_p10, '#B87333', 1, 'dot'),
    _line('P90', d.spill_p90, '#B87333', 1, 'dot'),
  ], _lo({title:'Spillage (m\u00b3/s)', yaxis:{title:'m\u00b3/s'}}), _C);

  Plotly.react('hd-wv', [
    _band('P10-P90', d.wv_p10, d.wv_p90, 'rgba(139,146,152,0.15)'),
    _line('P50', d.wv_p50, '#8B9298'),
    _line('P10', d.wv_p10, '#8B9298', 1, 'dot'),
    _line('P90', d.wv_p90, '#8B9298', 1, 'dot'),
  ], _lo({title:'Water Value (R$/hm\u00b3)', yaxis:{title:'R$/hm\u00b3'}}), _C);

  Plotly.react('hd-misc', [
    _line('Evaporation P50', d.evap_p50, '#8B5E3C'),
    _line('Outflow P50', d.outflow_p50, '#4A90B8'),
  ], _lo({title:'Evaporation & Outflow (m\u00b3/s)', yaxis:{title:'m\u00b3/s'}}), _C);
}
document.addEventListener('DOMContentLoaded', function(){setTimeout(updateHydroDetail,100);});
"""
        + "</script>"
    )


# ---------------------------------------------------------------------------
# Tab: Thermal Operations charts
# ---------------------------------------------------------------------------


def chart_thermal_generation_total(
    thermals: pd.DataFrame,
    stage_labels: dict[int, str],
) -> str:
    """Total thermal generation with p10/p50/p90 bands across all plants."""
    t0 = thermals[thermals["block_id"] == 0]
    total_gen = (
        t0.groupby(["scenario_id", "stage_id"])["generation_mw"].sum().reset_index()
    )
    pcts = (
        total_gen.groupby("stage_id")["generation_mw"]
        .quantile([0.1, 0.5, 0.9])
        .unstack(level=-1)
        .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
    )
    stages = sorted(pcts.index)
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels + xlabels[::-1],
            y=list(pcts["p90"].values) + list(pcts["p10"].values[::-1]),
            fill="toself",
            fillcolor="rgba(245,166,35,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10\u2013P90 range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p50"].values,
            name="Median (P50)",
            line={"color": COLORS["thermal"], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p10"].values,
            name="P10",
            line={"color": COLORS["thermal"], "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=pcts["p90"].values,
            name="P90",
            line={"color": COLORS["thermal"], "width": 1, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Total Thermal Generation (all plants, p10/p50/p90 across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_thermal_gen_by_bus(
    thermals: pd.DataFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Stacked area of thermal generation by bus (avg across scenarios)."""
    t0 = thermals[thermals["block_id"] == 0].copy()
    t0["bus_id"] = t0["thermal_id"].map(
        {k: v["bus_id"] for k, v in thermal_meta.items()}
    )
    bus_ids = sorted(t0["bus_id"].dropna().unique())
    stages = sorted(t0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        bus_id_int = int(bus_id)
        b_gen = (
            t0[t0["bus_id"] == bus_id_int]
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
        title="Thermal Generation by Bus (Block 0, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_thermal_cost_vs_gen(
    thermals: pd.DataFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
) -> str:
    """Scatter: avg generation vs cost_per_mwh, sized by max_mw, colored by bus."""
    avg_gen = (
        thermals[thermals["block_id"] == 0]
        .groupby(["scenario_id", "thermal_id"])["generation_mw"]
        .mean()
        .groupby("thermal_id")
        .mean()
    )

    # Group traces by bus so legend entries are per-bus
    bus_to_plants: dict[int, list[dict]] = {}
    for tid, meta in thermal_meta.items():
        bus_id = meta["bus_id"]
        bus_to_plants.setdefault(bus_id, []).append(
            {
                "tid": tid,
                "name": meta["name"],
                "avg_gen": float(avg_gen.get(tid, 0)),
                "cost": meta["cost_per_mwh"],
                "max_mw": meta["max_mw"],
            }
        )

    fig = go.Figure()
    for i, (bus_id, plants) in enumerate(sorted(bus_to_plants.items())):
        color = BUS_COLORS[i % len(BUS_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=[p["avg_gen"] for p in plants],
                y=[p["cost"] for p in plants],
                mode="markers+text",
                marker={
                    "size": [max(6, min(40, p["max_mw"] / 20)) for p in plants],
                    "color": color,
                    "opacity": 0.75,
                    "line": {"color": "white", "width": 1},
                },
                text=[p["name"] for p in plants],
                textposition="top center",
                textfont={"size": 9},
                name=bus_names.get(bus_id, str(bus_id)),
                customdata=[[p["max_mw"], p["tid"]] for p in plants],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Avg gen: %{x:.1f} MW<br>"
                    "Cost: %{y:.1f} R$/MWh<br>"
                    "Capacity: %{customdata[0]:.1f} MW<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title="Thermal Plants: Avg Generation vs Cost (size = installed capacity)",
        xaxis_title="Avg Generation (MW)",
        yaxis_title="Cost (R$/MWh)",
        legend={**_LEGEND, "y": 1.08},
        margin=dict(l=60, r=30, t=90, b=60),
        height=480,
    )
    return fig_to_html(fig, unified_hover=False)


# ---------------------------------------------------------------------------
# Tab: Thermal Plant Details
# ---------------------------------------------------------------------------


def build_interactive_thermal_details(
    thermals: pd.DataFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    lp_bounds: pd.DataFrame | None = None,
) -> str:
    """Build HTML with embedded per-thermal p10/p50/p90 data, LP bounds, and JS dropdown."""
    t0 = thermals[thermals["block_id"] == 0]
    stages = sorted(t0["stage_id"].unique())
    xlabels = stage_x_labels(stages, stage_labels)

    metrics = [
        "generation_mw",
        "generation_cost",
        "generation_mwh",
    ]
    short = {
        "generation_mw": "gen",
        "generation_cost": "cost",
        "generation_mwh": "energy",
    }

    thermal_data: dict[str, dict] = {}
    for tid, meta in sorted(thermal_meta.items()):
        tdata = t0[t0["thermal_id"] == tid]
        if tdata.empty:
            continue
        entry: dict = {
            "name": meta["name"],
            "bus": bus_names.get(meta["bus_id"], str(meta["bus_id"])),
            "max_mw": round(meta["max_mw"], 1),
            "cost_per_mwh": round(meta["cost_per_mwh"], 2),
        }
        for col in metrics:
            k = short[col]
            if col not in tdata.columns:
                for sfx in ["p10", "p50", "p90"]:
                    entry[f"{k}_{sfx}"] = [0.0] * len(stages)
                continue
            grp = tdata.groupby(["scenario_id", "stage_id"])[col].mean()
            pcts = (
                grp.groupby("stage_id")
                .quantile([0.1, 0.5, 0.9])
                .unstack(level=-1)
                .rename(columns={0.1: "p10", 0.5: "p50", 0.9: "p90"})
            )
            for sfx in ["p10", "p50", "p90"]:
                entry[f"{k}_{sfx}"] = [
                    round(float(pcts[sfx].get(s, 0)), 2) for s in stages
                ]
        thermal_data[str(tid)] = entry

    # Embed per-thermal LP bounds (entity_type_code=1): 6=gen_min, 7=gen_max
    if lp_bounds is not None and not lp_bounds.empty:
        tb = lp_bounds[lp_bounds["entity_type_code"] == 1]
        bound_keys = {6: "gen_min", 7: "gen_max"}
        for tid_str, entry in thermal_data.items():
            tid_int = int(tid_str)
            tb_plant = tb[tb["entity_id"] == tid_int]
            for bt_code, key in bound_keys.items():
                bt_rows = tb_plant[tb_plant["bound_type_code"] == bt_code]
                if bt_rows.empty:
                    entry[key] = [0.0] * len(stages)
                else:
                    by_stage = bt_rows.set_index("stage_id")["bound_value"]
                    entry[key] = [round(float(by_stage.get(s, 0)), 2) for s in stages]

    options = sorted(thermal_data.items(), key=lambda x: x[1]["name"])
    options_html = "\n".join(
        f'<option value="{tid}">{d["name"]} (id={tid})</option>' for tid, d in options
    )
    data_json = json.dumps(thermal_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    chart_rows = (
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="td-gen" style="width:100%;height:380px;"></div></div>'
        "</div>"
        '<div class="chart-grid">'
        '<div class="chart-card"><div id="td-cost" style="width:100%;height:320px;"></div></div>'
        '<div class="chart-card"><div id="td-energy" style="width:100%;height:320px;"></div></div>'
        "</div>"
    )

    return (
        '<div style="margin-bottom:16px;">'
        '<label for="thermal-select" style="font-weight:600;margin-right:8px;">Select Thermal Plant:</label>'
        '<select id="thermal-select" onchange="updateThermalDetail()" '
        'style="padding:8px 12px;font-size:0.9rem;border-radius:4px;border:1px solid #ccc;min-width:300px;">'
        + options_html
        + "</select>"
        + '<span id="thermal-info" style="margin-left:16px;color:#666;font-size:0.85rem;"></span>'
        + "</div>"
        + chart_rows
        + "<script>\n"
        + "const TD = "
        + data_json
        + ";\n"
        + "const TD_LABELS = "
        + labels_json
        + ";\n"
        + r"""
function _td_band(lbl, p10, p90, color) {
  return {x: TD_LABELS.concat(TD_LABELS.slice().reverse()),
          y: p90.concat(p10.slice().reverse()),
          fill:'toself', fillcolor:color, line:{color:'rgba(0,0,0,0)'},
          name:lbl, showlegend:true, hoverinfo:'skip'};
}
function _td_line(nm, y, c, w, dash) {
  return {x:TD_LABELS, y:y, name:nm, line:{color:c, width:w||2, dash:dash||'solid'}};
}
function _td_ref(nm, vals, c) {
  return {x:TD_LABELS, y:vals, name:nm, line:{color:c, width:1, dash:'dash'}};
}
var _TL = {hovermode:'x unified', margin:{l:60,r:20,t:50,b:60},
            legend:{orientation:'h',y:1.12,x:0,font:{size:11}}};
var _TC = {responsive:true};
function _tlo(extra){return Object.assign({},_TL,extra);}

function updateThermalDetail() {
  var tid = document.getElementById('thermal-select').value;
  var d = TD[tid]; if(!d) return;
  document.getElementById('thermal-info').textContent =
    d.bus+' | Capacity: '+d.max_mw.toFixed(0)+' MW | Cost: '+d.cost_per_mwh.toFixed(2)+' R$/MWh';

  var genTraces = [
    _td_band('P10\u2013P90', d.gen_p10, d.gen_p90, 'rgba(245,166,35,0.15)'),
    _td_line('P50', d.gen_p50, '#F5A623'),
    _td_line('P10', d.gen_p10, '#F5A623', 1, 'dot'),
    _td_line('P90', d.gen_p90, '#F5A623', 1, 'dot'),
  ];
  if(d.gen_max && d.gen_max.some(function(v){return v>0;})) {
    genTraces.push(_td_ref('Gen Max (LP)', d.gen_max, '#DC4C4C'));
  }
  if(d.gen_min && d.gen_min.some(function(v){return v>0;})) {
    genTraces.push(_td_ref('Gen Min (LP)', d.gen_min, '#4A8B6F'));
  }
  Plotly.react('td-gen', genTraces,
    _tlo({title:d.name+' \u2014 Generation (MW)', yaxis:{title:'MW'}}), _TC);

  Plotly.react('td-cost', [
    _td_band('P10\u2013P90', d.cost_p10, d.cost_p90, 'rgba(220,76,76,0.12)'),
    _td_line('P50', d.cost_p50, '#DC4C4C'),
    _td_line('P10', d.cost_p10, '#DC4C4C', 1, 'dot'),
    _td_line('P90', d.cost_p90, '#DC4C4C', 1, 'dot'),
  ], _tlo({title:'Generation Cost (R$)', yaxis:{title:'R$'}}), _TC);

  Plotly.react('td-energy', [
    _td_band('P10\u2013P90', d.energy_p10, d.energy_p90, 'rgba(74,139,111,0.15)'),
    _td_line('P50', d.energy_p50, '#4A8B6F'),
    _td_line('P10', d.energy_p10, '#4A8B6F', 1, 'dot'),
    _td_line('P90', d.energy_p90, '#4A8B6F', 1, 'dot'),
  ], _tlo({title:'Generation Energy (MWh)', yaxis:{title:'MWh'}}), _TC);
}
document.addEventListener('DOMContentLoaded', function(){setTimeout(updateThermalDetail,100);});
"""
        + "</script>"
    )


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
    hydro_meta = load_hydro_metadata(case_dir)

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

    # Performance / solver data (optional — graceful fallback to empty frames)
    timing_path = case_dir / "output" / "training" / "timing" / "iterations.parquet"
    timing = (
        pq.read_table(timing_path).to_pandas()
        if timing_path.exists()
        else pd.DataFrame()
    )
    solver_train_path = (
        case_dir / "output" / "training" / "solver" / "iterations.parquet"
    )
    solver_train = (
        pq.read_table(solver_train_path).to_pandas()
        if solver_train_path.exists()
        else pd.DataFrame()
    )
    sim_solver_path = (
        case_dir / "output" / "simulation" / "solver" / "iterations.parquet"
    )
    solver_sim = (
        pq.read_table(sim_solver_path).to_pandas()
        if sim_solver_path.exists()
        else pd.DataFrame()
    )
    scaling_path = case_dir / "output" / "training" / "scaling_report.json"
    scaling_report: dict = (
        json.load(scaling_path.open()) if scaling_path.exists() else {}
    )
    cs_path = case_dir / "output" / "training" / "cut_selection" / "iterations.parquet"
    cut_selection = (
        pq.read_table(cs_path).to_pandas() if cs_path.exists() else pd.DataFrame()
    )
    metadata_path = case_dir / "output" / "training" / "metadata.json"
    metadata: dict = json.load(metadata_path.open()) if metadata_path.exists() else {}

    # Load resolved LP bounds dictionary (actual bounds used by the solver)
    bounds_path = case_dir / "output" / "training" / "dictionaries" / "bounds.parquet"
    lp_bounds = (
        pq.read_table(bounds_path).to_pandas()
        if bounds_path.exists()
        else pd.DataFrame()
    )
    codes_path = case_dir / "output" / "training" / "dictionaries" / "codes.json"
    bounds_codes: dict = json.load(codes_path.open()) if codes_path.exists() else {}

    # Generic constraints data (optional — graceful fallback when absent)
    gc_path = case_dir / "constraints" / "generic_constraints.json"
    gc_constraints: list[dict] = []
    if gc_path.exists():
        with gc_path.open() as _f:
            _gc_data = json.load(_f)
        gc_constraints = _gc_data.get("constraints", [])

    gc_bounds_path = case_dir / "constraints" / "generic_constraint_bounds.parquet"
    gc_bounds = (
        pq.read_table(gc_bounds_path).to_pandas()
        if gc_bounds_path.exists()
        else pd.DataFrame()
    )

    gc_violations = load_all_scenarios(case_dir, "violations/generic")

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
        + section_title("Training Convergence")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_convergence(conv))
        + "</div>"
        + section_title("Cost Breakdown & Composition")
        + '<div class="chart-grid">'
        + wrap_chart(chart_cost_breakdown(costs))
        + wrap_chart(chart_cost_by_stage(costs, stage_labels))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab: Training Insights
    # ------------------------------------------------------------------
    print("Building Tab: Training Insights ...")
    tab_contents["tab-training"] = (
        section_title("Convergence")
        + '<div class="chart-grid">'
        + wrap_chart(chart_convergence(conv))
        + wrap_chart(chart_gap_evolution(conv))
        + "</div>"
        + section_title("Cut Management")
        + '<div class="chart-grid">'
        + wrap_chart(chart_cut_state_evolution(conv))
        + wrap_chart(chart_cut_activity_heatmap(cut_selection, stage_labels))
        + "</div>"
        + '<div class="chart-grid">'
        + wrap_chart(chart_cut_deactivation_heatmap(cut_selection, stage_labels))
        + "</div>"
        + section_title("LP Solver Heatmaps")
        + '<div class="chart-grid">'
        + wrap_chart(chart_simplex_heatmap(solver_train, stage_labels))
        + wrap_chart(chart_solve_time_heatmap(solver_train, stage_labels))
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
        + '<p style="color:#888;font-size:0.82rem;margin:-8px 0 16px 12px;">Note: Generation sum may exceed load due to exchange losses, NCS curtailment, and excess energy.</p>'
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
        + section_title("Deficit & Excess by Bus")
        + '<div class="chart-grid">'
        + wrap_chart(chart_deficit_by_bus(buses, bus_names, stage_labels, stage_hours))
        + wrap_chart(chart_excess_by_bus(buses, bus_names, stage_labels, stage_hours))
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
        + section_title("Storage by Bus")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_storage_by_bus(hydros, hydro_bus_map, bus_names, stage_labels)
        )
        + "</div>"
        + section_title("Hydro Generation & Spillage by Bus")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_hydro_gen_by_bus(hydros, hydro_bus_map, bus_names, stage_labels)
        )
        + wrap_chart(
            chart_spillage_by_bus(hydros, hydro_bus_map, bus_names, stage_labels)
        )
        + "</div>"
        + section_title("Water Values & Inflow Slack")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_water_value_by_bus(hydros, hydro_bus_map, bus_names, stage_labels)
        )
        + wrap_chart(chart_water_value_distribution(hydros, stage_labels))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 4: Plant Details
    # ------------------------------------------------------------------
    print("Building Tab 4: Plant Details ...")
    tab_contents["tab-plants"] = section_title(
        "Plant Explorer"
    ) + build_interactive_plant_details(
        hydros, hydro_meta, bus_names, stage_labels, lp_bounds
    )

    # ------------------------------------------------------------------
    # Tab 5: Exchanges
    # ------------------------------------------------------------------
    print("Building Tab 5: Exchanges ...")
    tab_contents["tab-exchanges"] = (
        section_title("Line Explorer")
        + build_interactive_exchange_detail(exchanges, names, stage_labels)
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
    print("Building Tab 6: Costs ...")
    tab_contents["tab-costs"] = (
        section_title("Cost Composition by Stage")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_cost_by_stage(costs, stage_labels))
        + "</div>"
        + section_title("Spot Price by Bus (weighted avg across blocks)")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_spot_price_by_bus_subplots(
                buses, bus_names, stage_labels, stage_hours, block_hours
            )
        )
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 6: NCS
    # ------------------------------------------------------------------
    print("Building Tab 7: NCS ...")
    tab_contents["tab-ncs-thermal"] = (
        section_title("NCS Available vs Generated")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_ncs_available_vs_generated(ncs, stage_labels))
        + "</div>"
        + section_title("NCS Curtailment by Source")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_ncs_curtailment_by_source(ncs, names))
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 8: Thermal Operations
    # ------------------------------------------------------------------
    print("Building Tab 8: Thermal Operations ...")
    tab_contents["tab-thermal"] = (
        section_title("Total Thermal Generation")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_thermal_generation_total(thermals, stage_labels))
        + "</div>"
        + section_title("Thermal Generation by Bus")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_thermal_gen_by_bus(thermals, thermal_meta, bus_names, stage_labels)
        )
        + "</div>"
    )

    # ------------------------------------------------------------------
    # Tab 9: Thermal Plant Details
    # ------------------------------------------------------------------
    print("Building Tab 9: Thermal Plant Details ...")
    tab_contents["tab-thermal-plants"] = section_title(
        "Thermal Plant Explorer"
    ) + build_interactive_thermal_details(
        thermals, thermal_meta, bus_names, stage_labels, lp_bounds
    )

    # ------------------------------------------------------------------
    # Tab 10: Constraints
    # ------------------------------------------------------------------
    print("Building Tab 10: Constraints ...")
    if gc_constraints:
        _lhs_df = evaluate_constraint_expressions(gc_constraints, hydros, exchanges)
        _summary_table = build_constraints_summary_table(
            gc_constraints, gc_bounds, gc_violations
        )
        _vminop_chart = chart_constraint_lhs_vs_bound(
            gc_constraints, _lhs_df, gc_bounds, stage_labels, ctype_filter="VminOP"
        )
        _re_chart = chart_constraint_lhs_vs_bound(
            gc_constraints, _lhs_df, gc_bounds, stage_labels, ctype_filter="RE"
        )
        _agrint_chart = chart_constraint_lhs_vs_bound(
            gc_constraints, _lhs_df, gc_bounds, stage_labels, ctype_filter="AGRINT"
        )
        tab_contents["tab-constraints"] = (
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
    else:
        tab_contents["tab-constraints"] = (
            f"<p>No generic constraint definitions found in {gc_path}.</p>"
        )

    # ------------------------------------------------------------------
    # Tab 9: Performance
    # ------------------------------------------------------------------
    print("Building Tab 11: Performance ...")
    # Filter simulation solver to actual scenario count from manifest.
    sim_manifest_path = case_dir / "output" / "simulation" / "_manifest.json"
    if sim_manifest_path.exists():
        sim_manifest = json.load(sim_manifest_path.open())
        actual_sim_scenarios = sim_manifest.get("scenarios", {}).get(
            "completed", n_scenarios
        )
        # Pass sim wall-clock to metrics via metadata
        metadata["_sim_manifest"] = sim_manifest
    else:
        actual_sim_scenarios = n_scenarios
    if not solver_sim.empty:
        solver_sim = solver_sim.head(actual_sim_scenarios)
    perf_metrics_html = build_performance_metrics_html(
        conv, timing, solver_train, solver_sim, scaling_report, metadata
    )
    tab_contents["tab-perf"] = (
        section_title("Run Summary")
        + perf_metrics_html
        + section_title("Training Iteration Breakdown")
        + '<div class="chart-grid">'
        + wrap_chart(chart_iteration_timing_breakdown(timing))
        + wrap_chart(chart_forward_vs_backward_per_iter(solver_train))
        + "</div>"
        + section_title("Solver Time Breakdown")
        + '<div class="chart-grid">'
        + wrap_chart(chart_solver_time_breakdown_by_phase(solver_train))
        + wrap_chart(chart_solver_time_per_stage(solver_train))
        + "</div>"
        + section_title("LP Solver Detail")
        + '<div class="chart-grid">'
        + wrap_chart(chart_backward_stage_heatmap(solver_train))
        + wrap_chart(chart_simplex_by_stage(solver_train))
        + "</div>"
        + section_title("Solver Overhead Detail")
        + '<div class="chart-grid">'
        + wrap_chart(chart_set_bounds_by_stage(solver_train))
        + wrap_chart(chart_basis_reuse(solver_train))
        + "</div>"
        + section_title("LP Dimensions & Scaling")
        + '<div class="chart-grid">'
        + wrap_chart(chart_lp_dimensions(scaling_report))
        + wrap_chart(chart_scaling_quality(scaling_report))
        + "</div>"
        + section_title("Solver Efficiency")
        + '<div class="chart-grid">'
        + wrap_chart(chart_cost_per_simplex_iter(solver_train))
        + wrap_chart(chart_timing_waterfall(timing))
        + "</div>"
        + section_title("Simulation")
        + '<div class="chart-grid-single">'
        + wrap_chart(chart_simulation_scenario_times(solver_sim))
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
