"""Energy Balance tab module for the Cobre dashboard.

Displays system-wide generation mix, generation share, generation by bus,
and deficit/excess by bus charts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.data import _compute_lp_load, _stage_avg_mw
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

TAB_ID = "tab-energy"
TAB_LABEL = "Energy Balance"
TAB_ORDER = 20

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_generation_mix(
    hydros_lf: pl.LazyFrame,
    thermals_lf: pl.LazyFrame,
    ncs_lf: pl.LazyFrame,
    load_stats: pd.DataFrame,
    load_factors_list: list[dict],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
) -> str:
    """Stacked area: hydro/thermal/NCS vs LP load per stage (stage-avg MW)."""
    h_gen = _stage_avg_mw(hydros_lf, "generation_mwh", stage_hours, [])
    t_gen = _stage_avg_mw(thermals_lf, "generation_mwh", stage_hours, [])
    n_gen = _stage_avg_mw(ncs_lf, "generation_mwh", stage_hours, [])
    assert isinstance(h_gen, dict)
    assert isinstance(t_gen, dict)
    assert isinstance(n_gen, dict)

    load_ser = _compute_lp_load(
        load_stats,
        load_factors_list,
        stage_hours,
        block_hours,
        bus_filter=[0, 1, 2, 3],
    )

    stages = sorted(h_gen.keys())
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
    hydros_lf: pl.LazyFrame,
    thermals_lf: pl.LazyFrame,
    ncs_lf: pl.LazyFrame,
    buses_lf: pl.LazyFrame,
    exchanges_lf: pl.LazyFrame,
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

    # Build mapping DataFrames for joins
    hbus_df = pl.DataFrame(
        {"hydro_id": list(hydro_bus_map.keys()), "bus_id": list(hydro_bus_map.values())}
    )
    tbus_map = {k: v["bus_id"] for k, v in thermal_meta.items()}
    tbus_df = pl.DataFrame(
        {"thermal_id": list(tbus_map.keys()), "bus_id": list(tbus_map.values())}
    )
    nbus_df = pl.DataFrame(
        {
            "non_controllable_id": list(ncs_bus_map.keys()),
            "bus_id": list(ncs_bus_map.values()),
        }
    )

    hours_df = pl.DataFrame(
        {"stage_id": list(stage_hours.keys()), "_hours": list(stage_hours.values())}
    )

    # Compute stage-average MW per bus for each entity type
    def _bus_stage_avg(
        lf: pl.LazyFrame, mwh_col: str, id_col: str, mapping_df: pl.DataFrame
    ) -> pl.DataFrame:
        return (
            lf.join(mapping_df.lazy(), on=id_col)
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg(pl.col(mwh_col).sum())
            .join(hours_df.lazy(), on="stage_id")
            .with_columns((pl.col(mwh_col) / pl.col("_hours")).alias("_avg_mw"))
            .group_by(["stage_id", "bus_id"])
            .agg(pl.col("_avg_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )

    h_by_bus = _bus_stage_avg(hydros_lf, "generation_mwh", "hydro_id", hbus_df)
    t_by_bus = _bus_stage_avg(thermals_lf, "generation_mwh", "thermal_id", tbus_df)
    n_by_bus = _bus_stage_avg(ncs_lf, "generation_mwh", "non_controllable_id", nbus_df)

    stages_all = sorted(h_by_bus["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages_all, stage_labels)

    def _to_map(df: pl.DataFrame, bus_id: int) -> dict[int, float]:
        sub = df.filter(pl.col("bus_id") == bus_id)
        return dict(zip(sub["stage_id"].to_list(), sub["_avg_mw"].to_list()))

    # Net exchange import per bus
    ex_import: dict[int, dict[int, float]] = {}
    for bus_id in bus_ids:
        parts: list[pl.DataFrame] = []
        for ln in line_meta:
            lid, src, tgt = ln["id"], ln["source_bus_id"], ln["target_bus_id"]
            if src != bus_id and tgt != bus_id:
                continue
            sign = 1.0 if tgt == bus_id else -1.0
            part = (
                exchanges_lf.filter(pl.col("line_id") == lid)
                .group_by(["scenario_id", "stage_id"])
                .agg((pl.col("net_flow_mwh") * sign).sum().alias("_imp_mwh"))
                .join(hours_df.lazy(), on="stage_id")
                .with_columns((pl.col("_imp_mwh") / pl.col("_hours")).alias("_mw"))
                .group_by("stage_id")
                .agg(pl.col("_mw").mean())
                .collect(engine="streaming")
            )
            parts.append(part)
        if parts:
            combined = pl.concat(parts)
            agg = (
                combined.group_by("stage_id").agg(pl.col("_mw").sum()).sort("stage_id")
            )
            ex_import[bus_id] = dict(
                zip(agg["stage_id"].to_list(), agg["_mw"].to_list())
            )
        else:
            ex_import[bus_id] = {}

    fig = make_subplots(
        rows=n_buses,
        cols=1,
        shared_xaxes=False,
        subplot_titles=[bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=0.18,
    )

    for row_idx, bus_id in enumerate(bus_ids, start=1):
        show_legend = row_idx == 1
        h_gen = _to_map(h_by_bus, bus_id)
        t_gen = _to_map(t_by_bus, bus_id)
        n_gen = _to_map(n_by_bus, bus_id)
        load_s = _compute_lp_load(
            load_stats,
            load_factors_list,
            stage_hours,
            block_hours,
            bus_filter=[bus_id],
        )
        net_imp = ex_import.get(bus_id, {})

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
    hydros_lf: pl.LazyFrame,
    thermals_lf: pl.LazyFrame,
    ncs_lf: pl.LazyFrame,
    buses_lf: pl.LazyFrame,
) -> str:
    """Pie chart of average generation shares including deficit."""
    h_total = (
        hydros_lf.group_by("scenario_id")
        .agg(pl.col("generation_mwh").sum())
        .select(pl.col("generation_mwh").mean())
        .collect(engine="streaming")["generation_mwh"][0]
    )
    t_total = (
        thermals_lf.group_by("scenario_id")
        .agg(pl.col("generation_mwh").sum())
        .select(pl.col("generation_mwh").mean())
        .collect(engine="streaming")["generation_mwh"][0]
    )
    n_total = (
        ncs_lf.group_by("scenario_id")
        .agg(pl.col("generation_mwh").sum())
        .select(pl.col("generation_mwh").mean())
        .collect(engine="streaming")["generation_mwh"][0]
    )
    d_total = (
        buses_lf.filter(pl.col("bus_id").is_in([0, 1, 2, 3]))
        .group_by("scenario_id")
        .agg(pl.col("deficit_mwh").sum())
        .select(pl.col("deficit_mwh").mean())
        .collect(engine="streaming")["deficit_mwh"][0]
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


def chart_deficit_by_bus(
    buses_lf: pl.LazyFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
) -> str:
    """Deficit by bus by stage (stage-avg MW)."""
    real_buses = sorted([bid for bid in bus_names if bid <= 3])
    hours_df = pl.DataFrame(
        {"stage_id": list(stage_hours.keys()), "_hours": list(stage_hours.values())}
    )
    def_data = (
        buses_lf.filter(pl.col("bus_id").is_in(real_buses))
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("deficit_mwh").sum())
        .join(hours_df.lazy(), on="stage_id")
        .with_columns((pl.col("deficit_mwh") / pl.col("_hours")).alias("_avg_mw"))
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("_avg_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = sorted(def_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(real_buses):
        sub = def_data.filter(pl.col("bus_id") == bus_id)
        dm = dict(zip(sub["stage_id"].to_list(), sub["_avg_mw"].to_list()))
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[dm.get(s, 0) for s in stages],
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
    buses_lf: pl.LazyFrame,
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    stage_hours: dict[int, float],
) -> str:
    """Excess by bus by stage (stage-avg MW)."""
    real_buses = sorted([bid for bid in bus_names if bid <= 3])
    hours_df = pl.DataFrame(
        {"stage_id": list(stage_hours.keys()), "_hours": list(stage_hours.values())}
    )
    exc_data = (
        buses_lf.filter(pl.col("bus_id").is_in(real_buses))
        .group_by(["scenario_id", "stage_id", "bus_id"])
        .agg(pl.col("excess_mwh").sum())
        .join(hours_df.lazy(), on="stage_id")
        .with_columns((pl.col("excess_mwh") / pl.col("_hours")).alias("_avg_mw"))
        .group_by(["stage_id", "bus_id"])
        .agg(pl.col("_avg_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    stages = sorted(exc_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    fig = go.Figure()
    for i, bus_id in enumerate(real_buses):
        sub = exc_data.filter(pl.col("bus_id") == bus_id)
        em = dict(zip(sub["stage_id"].to_list(), sub["_avg_mw"].to_list()))
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[em.get(s, 0) for s in stages],
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


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Energy Balance tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Energy Balance tab content area."""
    return (
        section_title("System-Wide Generation Mix")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_generation_mix(
                data.hydros_lf,
                data.thermals_lf,
                data.ncs_lf,
                data.load_stats,
                data.load_factors_list,
                data.stage_labels,
                data.stage_hours,
                data.block_hours,
            )
        )
        + "</div>"
        + '<p style="color:#888;font-size:0.82rem;margin:-8px 0 16px 12px;">Note: Generation sum may exceed load due to exchange losses, NCS curtailment, and excess energy.</p>'
        + section_title("Generation Share")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_generation_share_pie(
                data.hydros_lf, data.thermals_lf, data.ncs_lf, data.buses_lf
            )
        )
        + "</div>"
        + section_title("Generation by Bus")
        + '<div class="chart-grid-single">'
        + wrap_chart(
            chart_generation_by_bus(
                data.hydros_lf,
                data.thermals_lf,
                data.ncs_lf,
                data.buses_lf,
                data.exchanges_lf,
                data.hydro_bus_map,
                data.thermal_meta,
                data.ncs_bus_map,
                data.line_meta,
                data.bus_names,
                data.stage_labels,
                data.stage_hours,
                data.load_stats,
                data.load_factors_list,
                data.block_hours,
            )
        )
        + "</div>"
        + section_title("Deficit & Excess by Bus")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_deficit_by_bus(
                data.buses_lf, data.bus_names, data.stage_labels, data.stage_hours
            )
        )
        + wrap_chart(
            chart_excess_by_bus(
                data.buses_lf, data.bus_names, data.stage_labels, data.stage_hours
            )
        )
        + "</div>"
    )
