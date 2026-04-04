"""Plant Details tab module for the Cobre dashboard.

Displays the interactive hydro plant explorer with per-plant p10/p50/p90
statistics, LP bounds, and a JavaScript master-detail layout powered by
Plotly.react().
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.ui.html import (
    _sparkline_svg,
    chart_grid,
    collapsible_section,
    plant_explorer_table,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.js import PLANT_EXPLORER_JS
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
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

TAB_ID = "tab-plants"
TAB_LABEL = "Plant Details"
TAB_ORDER = 40

# ---------------------------------------------------------------------------
# Private helpers (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------

M3S_TO_HM3_PER_HOUR = 3600 / 1e6


def chart_top_hydros_detail(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
    top_n: int = 8,
) -> str:
    """Generation, storage, and spillage timeseries for top hydro plants."""
    ranked = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]
    hids = [hid for hid, _ in ranked]

    # Flow variables (generation, spillage): block-hours weighted average
    flow_data = (
        hydros_lf.filter(pl.col("hydro_id").is_in(hids))
        .join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg(
            (pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
            (pl.col("spillage_m3s") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
        )
        .group_by(["stage_id", "hydro_id"])
        .agg(
            pl.col("generation_mw").mean(),
            pl.col("spillage_m3s").mean(),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    # Stage-level variable (storage): block_id==0 only
    stor_data = (
        hydros_lf.filter((pl.col("block_id") == 0) & pl.col("hydro_id").is_in(hids))
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg(pl.col("storage_final_hm3").mean())
        .group_by(["stage_id", "hydro_id"])
        .agg(pl.col("storage_final_hm3").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = sorted(flow_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Generation (MW)", "Storage (hm³)", "Spillage (m³/s)"],
        vertical_spacing=0.18,
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
        sub_f = flow_data.filter(pl.col("hydro_id") == hid)
        sub_s = stor_data.filter(pl.col("hydro_id") == hid)
        gen_map = dict(
            zip(sub_f["stage_id"].to_list(), sub_f["generation_mw"].to_list())
        )
        stor_map = dict(
            zip(sub_s["stage_id"].to_list(), sub_s["storage_final_hm3"].to_list())
        )
        spill_map = dict(
            zip(sub_f["stage_id"].to_list(), sub_f["spillage_m3s"].to_list())
        )

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[gen_map.get(s, 0) for s in stages],
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
                y=[stor_map.get(s, 0) for s in stages],
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
                y=[spill_map.get(s, 0) for s in stages],
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
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=100, b=50),
    )
    return fig_to_html(fig)


def build_top_hydros_table(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    bus_names: dict[int, str],
    bh_df: pl.DataFrame,
    top_n: int = 20,
) -> str:
    """HTML table of top hydro plants with key simulation metrics."""
    ranked = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]
    hids = [hid for hid, _ in ranked]

    # Flow variables (generation, spillage): block-hours weighted average
    flow_stats = (
        hydros_lf.filter(pl.col("hydro_id").is_in(hids))
        .join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg(
            (pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
            (pl.col("spillage_m3s") * pl.col("_bh")).sum() / pl.col("_bh").sum(),
        )
        .group_by(["scenario_id", "hydro_id"])
        .agg(pl.col("generation_mw").mean(), pl.col("spillage_m3s").mean())
        .group_by("hydro_id")
        .agg(pl.col("generation_mw").mean(), pl.col("spillage_m3s").mean())
        .collect(engine="streaming")
    )
    # Stage-level variables (water_value, storage): block_id==0 only
    stage_stats = (
        hydros_lf.filter((pl.col("block_id") == 0) & pl.col("hydro_id").is_in(hids))
        .group_by(["scenario_id", "hydro_id"])
        .agg(
            pl.col("water_value_per_hm3").mean(),
            pl.col("storage_final_hm3").mean(),
        )
        .group_by("hydro_id")
        .agg(pl.col("water_value_per_hm3").mean(), pl.col("storage_final_hm3").mean())
        .collect(engine="streaming")
    )
    flow_map = {r["hydro_id"]: r for r in flow_stats.iter_rows(named=True)}
    stage_map = {r["hydro_id"]: r for r in stage_stats.iter_rows(named=True)}

    # Merge into unified stats_map
    stats_map: dict[int, dict] = {}
    for hid in hids:
        entry: dict = {}
        entry.update(flow_map.get(hid, {}))
        entry.update(stage_map.get(hid, {}))
        stats_map[hid] = entry

    rows_html = []
    for hid, meta in ranked:
        r = stats_map.get(hid, {})
        avg_gen = r.get("generation_mw", 0)
        avg_spill = r.get("spillage_m3s", 0)
        avg_wv = r.get("water_value_per_hm3", 0)
        avg_stor = r.get("storage_final_hm3", 0)
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


def chart_per_block_balance(
    hydros_lf: pl.LazyFrame,
    thermals_lf: pl.LazyFrame,
    ncs_lf: pl.LazyFrame,
    buses_lf: pl.LazyFrame,
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
        vertical_spacing=0.18,
    )

    stages = sorted({s for s, _ in block_hours.keys()})
    xlabels = stage_x_labels(stages, stage_labels)

    for row, blk in enumerate(blocks, 1):
        show_legend = row == 1

        h_gen_df = (
            hydros_lf.filter(pl.col("block_id") == blk)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("generation_mw").sum())
            .group_by("stage_id")
            .agg(pl.col("generation_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
        t_gen_df = (
            thermals_lf.filter(pl.col("block_id") == blk)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("generation_mw").sum())
            .group_by("stage_id")
            .agg(pl.col("generation_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
        n_gen_df = (
            ncs_lf.filter(pl.col("block_id") == blk)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("generation_mw").sum())
            .group_by("stage_id")
            .agg(pl.col("generation_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
        load_df = (
            buses_lf.filter(
                (pl.col("block_id") == blk) & pl.col("bus_id").is_in([0, 1, 2, 3])
            )
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("load_mw").sum())
            .group_by("stage_id")
            .agg(pl.col("load_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )

        h_map = dict(
            zip(h_gen_df["stage_id"].to_list(), h_gen_df["generation_mw"].to_list())
        )
        t_map = dict(
            zip(t_gen_df["stage_id"].to_list(), t_gen_df["generation_mw"].to_list())
        )
        n_map = dict(
            zip(n_gen_df["stage_id"].to_list(), n_gen_df["generation_mw"].to_list())
        )
        l_map = dict(zip(load_df["stage_id"].to_list(), load_df["load_mw"].to_list()))

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[h_map.get(s, 0) for s in stages],
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
                y=[t_map.get(s, 0) for s in stages],
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
                y=[n_map.get(s, 0) for s in stages],
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
                y=[l_map.get(s, 0) for s in stages],
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
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig_to_html(fig)


def chart_inflow_comparison(
    hydros_lf: pl.LazyFrame,
    inflow_stats: pd.DataFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    top_n: int = 6,
) -> str:
    """Compare realized inflow (p10/p50/p90) with historical mean +/- std."""
    ranked = sorted(hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True)[
        :top_n
    ]
    hids = [hid for hid, _ in ranked]

    inflow_data = (
        hydros_lf.filter((pl.col("block_id") == 0) & pl.col("hydro_id").is_in(hids))
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg(pl.col("inflow_m3s").mean())
        .group_by(["stage_id", "hydro_id"])
        .agg(
            pl.col("inflow_m3s").quantile(0.1, interpolation="linear").alias("p10"),
            pl.col("inflow_m3s").quantile(0.5, interpolation="linear").alias("p50"),
            pl.col("inflow_m3s").quantile(0.9, interpolation="linear").alias("p90"),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages_all = sorted(inflow_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages_all, stage_labels)

    fig = make_subplots(
        rows=top_n,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[meta["name"] for _, meta in ranked],
        vertical_spacing=0.18,
    )

    for row, (hid, meta) in enumerate(ranked, 1):
        show_legend = row == 1
        sub = inflow_data.filter(pl.col("hydro_id") == hid)
        pmap = {r["stage_id"]: r for r in sub.iter_rows(named=True)}
        hist = inflow_stats[inflow_stats["hydro_id"] == hid].set_index("stage_id")

        p90 = [pmap.get(s, {}).get("p90", 0) for s in stages_all]
        p10 = [pmap.get(s, {}).get("p10", 0) for s in stages_all]
        p50 = [pmap.get(s, {}).get("p50", 0) for s in stages_all]
        hist_mean = [
            hist["mean_m3s"].get(s, 0) if s in hist.index else 0 for s in stages_all
        ]
        hist_upper = [
            (hist["mean_m3s"].get(s, 0) + hist["std_m3s"].get(s, 0))
            if s in hist.index
            else 0
            for s in stages_all
        ]
        hist_lower = [
            max(0, hist["mean_m3s"].get(s, 0) - hist["std_m3s"].get(s, 0))
            if s in hist.index
            else 0
            for s in stages_all
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
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=100, b=50),
    )
    return fig_to_html(fig)


def chart_plant_water_balance(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
    top_n: int = 6,
) -> str:
    """Water balance components for top hydro plants."""
    # Pick top by spillage and capacity
    avg_spill_df = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .group_by(["scenario_id", "hydro_id"])
        .agg(pl.col("spillage_m3s").mean())
        .group_by("hydro_id")
        .agg(pl.col("spillage_m3s").mean())
        .sort("spillage_m3s", descending=True)
        .head(top_n)
        .collect(engine="streaming")
    )
    by_cap = [
        hid
        for hid, _ in sorted(
            hydro_meta.items(), key=lambda x: x[1]["max_gen_mw"], reverse=True
        )[:top_n]
    ]
    by_spill = avg_spill_df["hydro_id"].to_list()
    hids = list(dict.fromkeys(by_cap + by_spill))[:top_n]

    # Collect all data for these plants (all blocks needed for outflow calc)
    plant_data = (
        hydros_lf.filter(pl.col("hydro_id").is_in(hids))
        .select(
            "scenario_id",
            "stage_id",
            "block_id",
            "hydro_id",
            "storage_initial_hm3",
            "storage_final_hm3",
            "inflow_m3s",
            "evaporation_m3s",
            "turbined_m3s",
            "spillage_m3s",
        )
        .collect(engine="streaming")
    )
    plant_pd = plant_data.to_pandas()

    plant_pd[plant_pd["block_id"] == 0]
    stages = sorted(plant_pd["stage_id"].unique().tolist())
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
        vertical_spacing=0.18,
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
        hdata = plant_pd[plant_pd["hydro_id"] == hid]
        hb0 = hdata[hdata["block_id"] == 0]

        stor = (
            hb0.groupby(["scenario_id", "stage_id"])["storage_final_hm3"]
            .mean()
            .groupby("stage_id")
            .mean()
        )
        inflow = (
            hb0.groupby(["scenario_id", "stage_id"])["inflow_m3s"]
            .mean()
            .groupby("stage_id")
            .mean()
        )

        outflow_vals: dict[int, float] = {}
        residual_vals: dict[int, float] = {}
        for s in stages:
            s_data = hdata[hdata["stage_id"] == s]
            if s_data.empty:
                outflow_vals[s] = 0
                residual_vals[s] = 0
                continue
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
                for _, row_ in ss.iterrows():
                    blk = int(row_["block_id"])
                    tau = block_hours.get((s, blk), 0) * M3S_TO_HM3_PER_HOUR
                    total_out_vol += tau * (row_["turbined_m3s"] + row_["spillage_m3s"])
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
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=100, b=50),
    )
    return fig_to_html(fig)


def build_interactive_plant_details(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
    lp_bounds: pd.DataFrame | None = None,
) -> str:
    """Build HTML with embedded per-hydro p10/p50/p90 data, LP bounds, and JS dropdown."""
    # Flow variables: block-hours weighted average across blocks
    flow_metrics = [
        "generation_mw",
        "spillage_m3s",
        "turbined_m3s",
        "outflow_m3s",
    ]
    # Stage-level variables: constant across blocks, use block_id==0
    stage_metrics = [
        "storage_final_hm3",
        "inflow_m3s",
        "water_value_per_hm3",
        "evaporation_m3s",
        "water_withdrawal_violation_m3s",
    ]
    all_metrics = flow_metrics + stage_metrics
    short = {
        "generation_mw": "gen",
        "storage_final_hm3": "stor",
        "spillage_m3s": "spill",
        "inflow_m3s": "inflow",
        "turbined_m3s": "turb",
        "water_value_per_hm3": "wv",
        "evaporation_m3s": "evap",
        "outflow_m3s": "outflow",
        "water_withdrawal_violation_m3s": "ww",
    }

    schema = hydros_lf.collect_schema()
    avail_flow = [m for m in flow_metrics if m in schema]
    avail_stage = [m for m in stage_metrics if m in schema]
    available_metrics = avail_flow + avail_stage

    # Weighted average for flow metrics
    flow_pcts = (
        hydros_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg(
            *[
                ((pl.col(m) * pl.col("_bh")).sum() / pl.col("_bh").sum()).alias(m)
                for m in avail_flow
            ]
        )
        .group_by(["stage_id", "hydro_id"])
        .agg(
            [
                expr
                for m in avail_flow
                for expr in [
                    pl.col(m).quantile(0.1, interpolation="linear").alias(f"{m}_p10"),
                    pl.col(m).quantile(0.5, interpolation="linear").alias(f"{m}_p50"),
                    pl.col(m).quantile(0.9, interpolation="linear").alias(f"{m}_p90"),
                ]
            ]
        )
        .sort(["hydro_id", "stage_id"])
        .collect(engine="streaming")
        if avail_flow
        else pl.DataFrame()
    )

    # Simple mean for stage-level metrics (block_id==0)
    stage_pcts = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .group_by(["scenario_id", "stage_id", "hydro_id"])
        .agg([pl.col(m).mean() for m in avail_stage])
        .group_by(["stage_id", "hydro_id"])
        .agg(
            [
                expr
                for m in avail_stage
                for expr in [
                    pl.col(m).quantile(0.1, interpolation="linear").alias(f"{m}_p10"),
                    pl.col(m).quantile(0.5, interpolation="linear").alias(f"{m}_p50"),
                    pl.col(m).quantile(0.9, interpolation="linear").alias(f"{m}_p90"),
                ]
            ]
        )
        .sort(["hydro_id", "stage_id"])
        .collect(engine="streaming")
        if avail_stage
        else pl.DataFrame()
    )

    # Join flow and stage percentile frames on [stage_id, hydro_id]
    if not flow_pcts.is_empty() and not stage_pcts.is_empty():
        all_pcts = flow_pcts.join(
            stage_pcts, on=["stage_id", "hydro_id"], how="full", coalesce=True
        )
    elif not flow_pcts.is_empty():
        all_pcts = flow_pcts
    else:
        all_pcts = stage_pcts

    stages = sorted(all_pcts["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    hydro_data: dict[str, dict] = {}
    for hid, meta in sorted(hydro_meta.items()):
        sub = all_pcts.filter(pl.col("hydro_id") == hid)
        if sub.is_empty():
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
        sub_map: dict[int, dict] = {r["stage_id"]: r for r in sub.iter_rows(named=True)}
        for m in all_metrics:
            k = short[m]
            if m not in available_metrics:
                for sfx in ["p10", "p50", "p90"]:
                    entry[f"{k}_{sfx}"] = [0.0] * len(stages)
                continue
            for sfx in ["p10", "p50", "p90"]:
                entry[f"{k}_{sfx}"] = [
                    round(float(sub_map.get(s, {}).get(f"{m}_{sfx}", 0) or 0), 2)
                    for s in stages
                ]
        hydro_data[str(hid)] = entry

    # LP bounds
    if lp_bounds is not None and not lp_bounds.empty:
        hb = lp_bounds[lp_bounds["entity_type_code"] == 0]
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

    if not options:
        return "<p>No hydro plant data available.</p>"

    # Build table rows (sorted alphabetically by name, each row has data-index for
    # selectRow() lookup in HD, and data-name for filterTable() search).
    table_rows: list[str] = []
    for hid, d in options:
        stor_p50: list[float] = d.get("stor_p50", [])
        gen_p50: list[float] = d.get("gen_p50", [])
        stor_spark = (
            _sparkline_svg(stor_p50, "#4A90B8")
            if len(stor_p50) >= 2 and any(v != 0 for v in stor_p50)
            else ""
        )
        gen_spark = (
            _sparkline_svg(gen_p50, "#4A8B6F")
            if len(gen_p50) >= 2 and any(v != 0 for v in gen_p50)
            else ""
        )
        max_gen = d.get("max_gen", 0)
        vol_max = d.get("vol_max", 0)
        bus = d.get("bus", "")
        name = d.get("name", "")
        table_rows.append(
            f'<tr data-name="{name.lower()}" data-index="{hid}">'
            f'<td><input type="checkbox" class="compare-checkbox" data-id="{hid}"></td>'
            f"<td>{name}</td>"
            f"<td>{bus}</td>"
            f'<td data-sort-value="{max_gen:.0f}">{max_gen:.0f}</td>'
            f'<td data-sort-value="{vol_max:.0f}">{vol_max:.0f}</td>'
            f"<td>{stor_spark}</td>"
            f"<td>{gen_spark}</td>"
            f"</tr>"
        )

    rows_html = "".join(table_rows)
    columns: list[tuple[str, str]] = [
        ("Cmp", "none"),
        ("Name", "string"),
        ("Bus", "string"),
        ("MW", "number"),
        ("Vol", "number"),
        ("Storage", "none"),
        ("Gen", "none"),
    ]
    table_pane = (
        '<div class="explorer-table-pane">'
        + plant_explorer_table(
            "hd-tbody",
            "hd-search",
            columns,
            rows_html,
        )
        + "</div>"
    )

    # Detail pane: 3 collapsible sections with chart divs
    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" style="width:100%;height:350px;"></div>'

    water_balance_section = collapsible_section(
        "Water Balance",
        chart_grid(
            [
                wrap_chart(_chart_div("hd-stor")),
                wrap_chart(_chart_div("hd-inflow")),
                wrap_chart(_chart_div("hd-spill")),
                wrap_chart(_chart_div("hd-evap")),
            ]
        ),
    )
    generation_section = collapsible_section(
        "Generation",
        chart_grid([wrap_chart(_chart_div("hd-gen"))], single=True)
        + chart_grid(
            [
                wrap_chart(_chart_div("hd-turb")),
                wrap_chart(_chart_div("hd-outflow")),
            ]
        ),
    )
    economics_section = collapsible_section(
        "Economics",
        chart_grid(
            [
                wrap_chart(_chart_div("hd-wv")),
                wrap_chart(_chart_div("hd-ww")),
            ]
        ),
    )

    detail_pane = (
        '<div class="explorer-detail-pane">'
        + water_balance_section
        + generation_section
        + economics_section
        + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    data_json = json.dumps(hydro_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    inline_js = (
        PLANT_EXPLORER_JS
        + "\nconst HD = "
        + data_json
        + ";\n"
        + "const HD_LABELS = "
        + labels_json
        + ";\n"
        + """
var _C = {responsive: true};

function renderHydroDetail(containerId, d) {
  if (!d) { return; }
  var lbl = HD_LABELS;

  Plotly.react('hd-gen', [
    plotlyBand(lbl, d.gen_p10, d.gen_p90, 'rgba(74,144,184,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.gen_p50, '#4A90B8', 'P50'),
    plotlyLine(lbl, d.gen_p10, '#4A90B8', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.gen_p90, '#4A90B8', 'P90', 1, 'dot'),
    plotlyLine(lbl, d.gen_max || lbl.map(function() { return d.max_gen_phys; }),
               '#DC4C4C', 'Effective Capacity', 1, 'dash'),
  ], plotlyLayout({title: d.name + ' \u2014 Generation (MW)', yaxis: {title: 'MW'}}), _C);

  Plotly.react('hd-stor', [
    plotlyBand(lbl, d.stor_p10, d.stor_p90, 'rgba(74,144,184,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.stor_p50, '#4A90B8', 'P50'),
    plotlyLine(lbl, d.stor_p10, '#4A90B8', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.stor_p90, '#4A90B8', 'P90', 1, 'dot'),
    plotlyRef(lbl, d.vol_max, '#DC4C4C', 'Vol Max'),
    plotlyRef(lbl, d.vol_min, '#4A8B6F', 'Vol Min'),
  ], plotlyLayout({title: 'Storage (hm\u00b3)', yaxis: {title: 'hm\u00b3'}}), _C);

  Plotly.react('hd-inflow', [
    plotlyBand(lbl, d.inflow_p10, d.inflow_p90, 'rgba(74,139,111,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.inflow_p50, '#4A8B6F', 'P50'),
    plotlyLine(lbl, d.inflow_p10, '#4A8B6F', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.inflow_p90, '#4A8B6F', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Inflow (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _C);

  var turbTraces = [
    plotlyBand(lbl, d.turb_p10, d.turb_p90, 'rgba(245,166,35,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.turb_p50, '#F5A623', 'P50'),
    plotlyLine(lbl, d.turb_p10, '#F5A623', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.turb_p90, '#F5A623', 'P90', 1, 'dot'),
    plotlyLine(lbl, d.turb_max || lbl.map(function() { return d.max_turb; }),
               '#DC4C4C', 'Turb Max (LP)', 1, 'dash'),
  ];
  if (d.turb_min && d.turb_min.some(function(v) { return v > 0; })) {
    turbTraces.push(plotlyLine(lbl, d.turb_min, '#4A8B6F', 'Turb Min (LP)', 1, 'dash'));
  }
  Plotly.react('hd-turb', turbTraces,
               plotlyLayout({title: 'Turbined (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _C);

  Plotly.react('hd-spill', [
    plotlyBand(lbl, d.spill_p10, d.spill_p90, 'rgba(184,115,51,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.spill_p50, '#B87333', 'P50'),
    plotlyLine(lbl, d.spill_p10, '#B87333', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.spill_p90, '#B87333', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Spillage (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _C);

  Plotly.react('hd-wv', [
    plotlyBand(lbl, d.wv_p10, d.wv_p90, 'rgba(139,146,152,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.wv_p50, '#8B9298', 'P50'),
    plotlyLine(lbl, d.wv_p10, '#8B9298', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.wv_p90, '#8B9298', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Water Value (R$/hm\u00b3)', yaxis: {title: 'R$/hm\u00b3'}}), _C);

  var outflowTraces = [
    plotlyBand(lbl, d.outflow_p10, d.outflow_p90, 'rgba(74,144,184,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.outflow_p50, '#4A90B8', 'P50'),
    plotlyLine(lbl, d.outflow_p10, '#4A90B8', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.outflow_p90, '#4A90B8', 'P90', 1, 'dot'),
  ];
  if (d.outflow_min && d.outflow_min.some(function(v) { return v > 0; })) {
    outflowTraces.push(
      plotlyLine(lbl, d.outflow_min, '#4A8B6F', 'Outflow Min (LP)', 1, 'dash')
    );
  }
  Plotly.react('hd-outflow', outflowTraces,
               plotlyLayout({title: 'Outflow (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _C);

  Plotly.react('hd-evap', [
    plotlyBand(lbl, d.evap_p10, d.evap_p90, 'rgba(139,94,60,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.evap_p50, '#8B5E3C', 'P50'),
    plotlyLine(lbl, d.evap_p10, '#8B5E3C', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.evap_p90, '#8B5E3C', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Evaporation (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _C);

  Plotly.react('hd-ww', [
    plotlyBand(lbl, d.ww_p10, d.ww_p90, 'rgba(121,85,72,0.15)', 'P10-P90'),
    plotlyLine(lbl, d.ww_p50, '#795548', 'P50'),
    plotlyLine(lbl, d.ww_p10, '#795548', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.ww_p90, '#795548', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Water Withdrawal Violation (m\u00b3/s)',
                   yaxis: {title: 'm\u00b3/s'}}), _C);
}

var _HD_PALETTE = ['#2196F3', '#FF9800', '#4CAF50'];

function renderHydroComparison(entries, labels) {
  if (!entries || entries.length === 0) { return; }
  var lbl = labels || HD_LABELS;

  function _compTraces(metric_p50, colorIdx) {
    var color = _HD_PALETTE[colorIdx % _HD_PALETTE.length];
    return entries.map(function(d, i) {
      var c = _HD_PALETTE[i % _HD_PALETTE.length];
      return plotlyLine(lbl, d[metric_p50] || [], c, d.name || ('Plant ' + i));
    });
  }

  function _buildTraces(p50key) {
    return entries.map(function(d, i) {
      var c = _HD_PALETTE[i % _HD_PALETTE.length];
      return plotlyLine(lbl, d[p50key] || [], c, d.name || ('Plant ' + i));
    });
  }

  Plotly.react('hd-gen', _buildTraces('gen_p50'),
    plotlyLayout({title: 'Generation (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _C);
  Plotly.react('hd-stor', _buildTraces('stor_p50'),
    plotlyLayout({title: 'Storage (hm\u00b3) \u2014 Comparison', yaxis: {title: 'hm\u00b3'}}), _C);
  Plotly.react('hd-inflow', _buildTraces('inflow_p50'),
    plotlyLayout({title: 'Inflow (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _C);
  Plotly.react('hd-turb', _buildTraces('turb_p50'),
    plotlyLayout({title: 'Turbined (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _C);
  Plotly.react('hd-spill', _buildTraces('spill_p50'),
    plotlyLayout({title: 'Spillage (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _C);
  Plotly.react('hd-wv', _buildTraces('wv_p50'),
    plotlyLayout({title: 'Water Value (R$/hm\u00b3) \u2014 Comparison', yaxis: {title: 'R$/hm\u00b3'}}), _C);
  Plotly.react('hd-outflow', _buildTraces('outflow_p50'),
    plotlyLayout({title: 'Outflow (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _C);
  Plotly.react('hd-evap', _buildTraces('evap_p50'),
    plotlyLayout({title: 'Evaporation (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _C);
  Plotly.react('hd-ww', _buildTraces('ww_p50'),
    plotlyLayout({title: 'Water Withdrawal Violation (m\u00b3/s) \u2014 Comparison',
                  yaxis: {title: 'm\u00b3/s'}}), _C);
}

initPlantExplorer({
  tableId: 'hd-tbody',
  searchInputId: 'hd-search',
  detailContainerId: 'hd-detail',
  dataVar: 'HD',
  renderDetail: renderHydroDetail
});

initComparisonMode({
  tableId: 'hd-tbody',
  dataVar: 'HD',
  labelsVar: 'HD_LABELS',
  chartIds: ['hd-gen','hd-stor','hd-inflow','hd-turb','hd-spill','hd-wv','hd-outflow','hd-evap','hd-ww'],
  renderComparison: renderHydroComparison,
  renderDetail: renderHydroDetail,
  maxCompare: 3
});

syncHover(['hd-gen','hd-stor','hd-inflow','hd-turb','hd-spill','hd-wv','hd-outflow','hd-evap','hd-ww']);
"""
    )

    return explorer_html + "<script>\n" + inline_js + "\n</script>"


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Plant Details tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Plant Details tab content area."""
    return section_title("Plant Explorer") + build_interactive_plant_details(
        data.hydros_lf,
        data.hydro_meta,
        data.bus_names,
        data.stage_labels,
        data.bh_df,
        data.lp_bounds,
    )
