"""Hydro Operations tab module for the Cobre dashboard.

Displays reservoir storage, stored energy, hydro generation, spillage,
and water value charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
from cobre_bridge.ui.theme import BUS_COLORS, COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-hydro"
TAB_LABEL = "Hydro Operations"
TAB_ORDER = 30

# ---------------------------------------------------------------------------
# Private helpers (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def _build_accumulated_productivity(
    hydro_meta: dict[int, dict],
) -> dict[int, float]:
    """Compute accumulated cascade productivity for each hydro plant.

    Uses memoized DAG traversal: each plant's accumulated productivity is
    its own productivity plus the accumulated productivity of its downstream
    plant.  Each node is visited exactly once (O(n) total).
    """
    acc: dict[int, float] = {}

    def _accumulate(hid: int) -> float:
        if hid in acc:
            return acc[hid]
        meta = hydro_meta.get(hid)
        if meta is None:
            return 0.0
        ds = meta.get("downstream_id")
        ds_acc = _accumulate(ds) if ds is not None else 0.0
        acc[hid] = meta.get("productivity", 0.0) + ds_acc
        return acc[hid]

    for hid in hydro_meta:
        _accumulate(hid)
    return acc


def _add_stored_energy_column(
    lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
) -> pl.LazyFrame:
    """Add ``stored_energy_mwmonth`` column to a hydro LazyFrame.

    Formula: (storage_final_hm3 - vol_min) × accumulated_productivity / 2.628
    The divisor converts hm³ to the m³/s-equivalent over one month
    (1 hm³ = 10⁶ m³; 1 month ≈ 2.628 × 10⁶ s).
    """
    acc_prod = _build_accumulated_productivity(hydro_meta)

    # Build a small DataFrame: hydro_id -> (vol_min, acc_prod)
    rows = []
    for hid, meta in hydro_meta.items():
        rows.append(
            {
                "hydro_id": hid,
                "_vol_min": meta.get("vol_min", 0.0),
                "_acc_prod": acc_prod.get(hid, 0.0),
            }
        )
    param_df = pl.DataFrame(rows).cast(
        {"hydro_id": pl.Int32, "_vol_min": pl.Float64, "_acc_prod": pl.Float64}
    )

    return (
        lf.join(param_df.lazy(), on="hydro_id")
        .with_columns(
            (
                (pl.col("storage_final_hm3") - pl.col("_vol_min")).clip(lower_bound=0.0)
                * pl.col("_acc_prod")
                / 2.628
            ).alias("stored_energy_mwmonth")
        )
        .drop("_vol_min", "_acc_prod")
    )


# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_hydro_storage(hydros_lf: pl.LazyFrame, stage_labels: dict[int, str]) -> str:
    """Total storage with p10/p50/p90 bands."""
    pcts = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .group_by(["scenario_id", "stage_id"])
        .agg(pl.col("storage_final_hm3").sum())
        .group_by("stage_id")
        .agg(
            pl.col("storage_final_hm3")
            .quantile(0.1, interpolation="linear")
            .alias("p10"),
            pl.col("storage_final_hm3")
            .quantile(0.5, interpolation="linear")
            .alias("p50"),
            pl.col("storage_final_hm3")
            .quantile(0.9, interpolation="linear")
            .alias("p90"),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = pcts["stage_id"].to_list()
    xlabels = stage_x_labels(stages, stage_labels)
    p10 = pcts["p10"].to_list()
    p50 = pcts["p50"].to_list()
    p90 = pcts["p90"].to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels + xlabels[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(33,150,243,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10–P90 range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p50,
            name="Median (P50)",
            line={"color": COLORS["hydro"], "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p10,
            name="P10",
            line={"color": COLORS["hydro"], "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p90,
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


def chart_stored_energy(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
) -> str:
    """System-total stored energy with p10/p50/p90 bands."""
    lf = _add_stored_energy_column(
        hydros_lf.filter(pl.col("block_id") == 0), hydro_meta
    )
    pcts = (
        lf.group_by(["scenario_id", "stage_id"])
        .agg(pl.col("stored_energy_mwmonth").sum())
        .group_by("stage_id")
        .agg(
            pl.col("stored_energy_mwmonth")
            .quantile(0.1, interpolation="linear")
            .alias("p10"),
            pl.col("stored_energy_mwmonth")
            .quantile(0.5, interpolation="linear")
            .alias("p50"),
            pl.col("stored_energy_mwmonth")
            .quantile(0.9, interpolation="linear")
            .alias("p90"),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = pcts["stage_id"].to_list()
    xlabels = stage_x_labels(stages, stage_labels)
    p10, p50, p90 = pcts["p10"].to_list(), pcts["p50"].to_list(), pcts["p90"].to_list()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels + xlabels[::-1],
            y=p90 + p10[::-1],
            fill="toself",
            fillcolor="rgba(76,175,80,0.15)",
            line={"color": "rgba(255,255,255,0)"},
            name="P10–P90 range",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p50,
            name="Median (P50)",
            line={"color": "#4CAF50", "width": 2.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p10,
            name="P10",
            line={"color": "#4CAF50", "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=p90,
            name="P90",
            line={"color": "#4CAF50", "width": 1, "dash": "dot"},
        )
    )
    fig.update_layout(
        title="Aggregate Stored Energy (all hydros, p10/p50/p90 across scenarios)",
        xaxis_title="Stage",
        yaxis_title="Stored Energy (MWmonth)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_storage_by_bus(
    hydros_lf: pl.LazyFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Storage by bus with p50 line and p10-p90 band in subplots."""
    hbus_df = pl.DataFrame(
        {"hydro_id": list(hydro_bus_map.keys()), "bus_id": list(hydro_bus_map.values())}
    )
    stor_data = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .join(hbus_df.lazy(), on="hydro_id")
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("storage_final_hm3").sum())
        .group_by(["stage_id", "bus_id"])
        .agg(
            pl.col("storage_final_hm3")
            .quantile(0.1, interpolation="linear")
            .alias("p10"),
            pl.col("storage_final_hm3")
            .quantile(0.5, interpolation="linear")
            .alias("p50"),
            pl.col("storage_final_hm3")
            .quantile(0.9, interpolation="linear")
            .alias("p90"),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(stor_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)
    n = len(bus_ids)

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=0.18,
    )

    for row, bus_id in enumerate(bus_ids, 1):
        sub = stor_data.filter(pl.col("bus_id") == bus_id)
        pmap: dict[int, dict] = {
            r["stage_id"]: {"p10": r["p10"], "p50": r["p50"], "p90": r["p90"]}
            for r in sub.iter_rows(named=True)
        }
        show_legend = row == 1
        p90 = [pmap.get(s, {}).get("p90", 0) for s in stages]
        p10 = [pmap.get(s, {}).get("p10", 0) for s in stages]
        p50 = [pmap.get(s, {}).get("p50", 0) for s in stages]

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
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig_to_html(fig)


def chart_stored_energy_by_bus(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Stored energy by bus with p50 line and p10-p90 band in subplots."""
    lf = _add_stored_energy_column(
        hydros_lf.filter(pl.col("block_id") == 0), hydro_meta
    )
    hbus_df = pl.DataFrame(
        {"hydro_id": list(hydro_bus_map.keys()), "bus_id": list(hydro_bus_map.values())}
    )
    data = (
        lf.join(hbus_df.lazy(), on="hydro_id")
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("stored_energy_mwmonth").sum())
        .group_by(["stage_id", "bus_id"])
        .agg(
            pl.col("stored_energy_mwmonth")
            .quantile(0.1, interpolation="linear")
            .alias("p10"),
            pl.col("stored_energy_mwmonth")
            .quantile(0.5, interpolation="linear")
            .alias("p50"),
            pl.col("stored_energy_mwmonth")
            .quantile(0.9, interpolation="linear")
            .alias("p90"),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = make_subplots(
        rows=len(bus_ids),
        cols=1,
        shared_xaxes=False,
        subplot_titles=[bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=0.18,
    )
    for row, bus_id in enumerate(bus_ids, 1):
        sub = data.filter(pl.col("bus_id") == bus_id)
        pmap = {r["stage_id"]: r for r in sub.iter_rows(named=True)}
        show = row == 1
        p90 = [pmap.get(s, {}).get("p90", 0) for s in stages]
        p10 = [pmap.get(s, {}).get("p10", 0) for s in stages]
        p50 = [pmap.get(s, {}).get("p50", 0) for s in stages]
        fig.add_trace(
            go.Scatter(
                x=xlabels + xlabels[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(76,175,80,0.15)",
                line={"color": "rgba(255,255,255,0)"},
                name="P10-P90",
                legendgroup="band",
                showlegend=show,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=p50,
                name="Median",
                line={"color": "#4CAF50", "width": 2},
                legendgroup="median",
                showlegend=show,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="MWmonth", row=row, col=1)
    fig.update_layout(
        title="Stored Energy by Bus (p10/p50/p90)",
        legend=_LEGEND,
        margin=_MARGIN,
        height=300 * len(bus_ids),
    )
    return fig_to_html(fig)


def chart_hydro_gen_by_bus(
    hydros_lf: pl.LazyFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Stacked area of hydro generation by bus."""
    hbus_df = pl.DataFrame(
        {"hydro_id": list(hydro_bus_map.keys()), "bus_id": list(hydro_bus_map.values())}
    )
    gen_by_bus = (
        hydros_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .join(hbus_df.lazy(), on="hydro_id")
        .group_by(["scenario_id", "stage_id", "bus_id", "hydro_id"])
        .agg((pl.col("generation_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("generation_mw").sum())
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("generation_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    bus_ids = sorted(gen_by_bus["bus_id"].unique().to_list())
    stages = sorted(gen_by_bus["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        sub = gen_by_bus.filter(pl.col("bus_id") == bus_id)
        b_map = dict(zip(sub["stage_id"].to_list(), sub["generation_mw"].to_list()))
        color = BUS_COLORS[i % len(BUS_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[b_map.get(s, 0) for s in stages],
                name=bus_names.get(int(bus_id), str(bus_id)),
                stackgroup="buses",
                line={"color": color},
            )
        )
    fig.update_layout(
        title="Hydro Generation by Bus (block-hours weighted avg, avg across scenarios)",
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
        height=440,
    )
    return fig_to_html(fig)


def chart_spillage_by_bus(
    hydros_lf: pl.LazyFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Total spillage by bus by stage."""
    hbus_df = pl.DataFrame(
        {"hydro_id": list(hydro_bus_map.keys()), "bus_id": list(hydro_bus_map.values())}
    )
    spill_data = (
        hydros_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .join(hbus_df.lazy(), on="hydro_id")
        .group_by(["scenario_id", "stage_id", "bus_id", "hydro_id"])
        .agg((pl.col("spillage_m3s") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("spillage_m3s").sum())
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("spillage_m3s").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(spill_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        sub = spill_data.filter(pl.col("bus_id") == bus_id)
        spill_map = dict(zip(sub["stage_id"].to_list(), sub["spillage_m3s"].to_list()))
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[spill_map.get(s, 0) for s in stages],
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
    hydros_lf: pl.LazyFrame,
    hydro_bus_map: dict[int, int],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
) -> str:
    """Average water value by bus by stage."""
    hbus_df = pl.DataFrame(
        {"hydro_id": list(hydro_bus_map.keys()), "bus_id": list(hydro_bus_map.values())}
    )
    wv_data = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .join(hbus_df.lazy(), on="hydro_id")
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("water_value_per_hm3").mean())
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("water_value_per_hm3").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    bus_ids = sorted([bid for bid in bus_names if bid <= 3])
    stages = sorted(wv_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(bus_ids):
        sub = wv_data.filter(pl.col("bus_id") == bus_id)
        wv_map = dict(
            zip(sub["stage_id"].to_list(), sub["water_value_per_hm3"].to_list())
        )
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[wv_map.get(s, 0) for s in stages],
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


def chart_water_value_distribution(
    hydros_lf: pl.LazyFrame, stage_labels: dict[int, str]
) -> str:
    """Box plot of water values across hydros by stage (sample stages for readability)."""
    avg_by = (
        hydros_lf.filter(pl.col("block_id") == 0)
        .group_by(["stage_id", "hydro_id"])
        .agg(pl.col("water_value_per_hm3").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = sorted(avg_by["stage_id"].unique().to_list())
    step = max(1, len(stages) // 24)
    sampled_stages = stages[::step]
    xlabels = stage_x_labels(sampled_stages, stage_labels)

    fig = go.Figure()
    for s, lbl in zip(sampled_stages, xlabels):
        vals = avg_by.filter(pl.col("stage_id") == s)["water_value_per_hm3"].to_list()
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
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Hydro Operations tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Hydro Operations tab content area."""
    return (
        section_title("Aggregate Reservoir Storage")
        + '<div class="chart-grid">'
        + wrap_chart(chart_hydro_storage(data.hydros_lf, data.stage_labels))
        + wrap_chart(
            chart_stored_energy(data.hydros_lf, data.hydro_meta, data.stage_labels)
        )
        + "</div>"
        + section_title("Storage & Stored Energy by Bus")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_storage_by_bus(
                data.hydros_lf, data.hydro_bus_map, data.bus_names, data.stage_labels
            )
        )
        + wrap_chart(
            chart_stored_energy_by_bus(
                data.hydros_lf,
                data.hydro_meta,
                data.hydro_bus_map,
                data.bus_names,
                data.stage_labels,
            )
        )
        + "</div>"
        + section_title("Hydro Generation & Spillage by Bus")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_hydro_gen_by_bus(
                data.hydros_lf,
                data.hydro_bus_map,
                data.bus_names,
                data.stage_labels,
                data.bh_df,
            )
        )
        + wrap_chart(
            chart_spillage_by_bus(
                data.hydros_lf,
                data.hydro_bus_map,
                data.bus_names,
                data.stage_labels,
                data.bh_df,
            )
        )
        + "</div>"
        + section_title("Water Values & Inflow Slack")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_water_value_by_bus(
                data.hydros_lf,
                data.hydro_bus_map,
                data.bus_names,
                data.stage_labels,
            )
        )
        + wrap_chart(chart_water_value_distribution(data.hydros_lf, data.stage_labels))
        + "</div>"
    )
