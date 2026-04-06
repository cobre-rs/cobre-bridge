"""v2 Energy Balance tab — system-wide generation mix and per-bus breakdown.

Implements six sections of the Energy Balance tab (Tab 4):
  A. Metrics row — 6 cards (hydro, thermal, NCS GWh; deficit GWh;
     spillage m3/s; curtailment GWh).
  B. System generation mix hero chart — stacked area of hydro/thermal/NCS
     vs LP load (dashed overlay), stage-average MW.
  C. Generation by bus — collapsible facet with one subplot per
     non-fictitious bus.
  D. Deficit & Excess — two side-by-side charts, one line per bus,
     with mean+p50+p10-p90 band.
  E. Reservoir Storage — system aggregate trajectory with vol_min/vol_max
     reference lines, plus per-bus storage as % useful volume.
  F. NCS & Curtailment — NCS generation vs available capacity (left),
     curtailment by stage bar chart (right).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.chart_helpers import (
    add_bounds_overlay,
    add_mean_p50_band,
    compute_percentiles,
    make_chart_card,
)
from cobre_bridge.dashboard.data import _compute_lp_load, _stage_avg_mw
from cobre_bridge.ui.html import (
    chart_grid,
    collapsible_section,
    metric_card,
    metrics_grid,
    section_title,
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
from cobre_bridge.ui.theme import BUS_COLORS, COLORS, GENERATION_COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-energy-balance"
TAB_LABEL = "Energy Balance"
TAB_ORDER = 30

# ---------------------------------------------------------------------------
# Private metric helpers
# ---------------------------------------------------------------------------


def _compute_total_gwh(lf: pl.LazyFrame, mwh_col: str) -> float:
    """Return mean total GWh across scenarios for *mwh_col*.

    Sums *mwh_col* per scenario, averages across scenarios, then converts
    MWh to GWh (divide by 1000).  Returns ``0.0`` when the LazyFrame is
    empty, *mwh_col* is absent, or any computation error occurs.

    Args:
        lf: Simulation LazyFrame containing ``scenario_id`` and *mwh_col*.
        mwh_col: Column name holding energy values in MWh.

    Returns:
        Mean total GWh as a float, or ``0.0`` on any failure.
    """
    try:
        result = (
            lf.group_by("scenario_id")
            .agg(pl.col(mwh_col).sum())
            .select(pl.col(mwh_col).mean())
            .collect(engine="streaming")
        )
        if result.height == 0:
            return 0.0
        value = result[mwh_col][0]
        return float(value) / 1_000.0 if value is not None else 0.0
    except (ValueError, TypeError, KeyError, pl.exceptions.ColumnNotFoundError):
        return 0.0


def _compute_total_avg(
    lf: pl.LazyFrame,
    col: str,
) -> float:
    """Return mean total of *col* across scenarios.

    Groups by scenario, sums *col*, then averages across scenarios.  Used
    for non-MWh metrics (e.g. spillage in m3/s averaged over all scenarios
    and entities).  Returns ``0.0`` on any failure.

    Args:
        lf: Simulation LazyFrame containing ``scenario_id`` and *col*.
        col: Column name to aggregate.

    Returns:
        Mean total as a float, or ``0.0`` on any failure.
    """
    try:
        result = (
            lf.group_by("scenario_id")
            .agg(pl.col(col).sum())
            .select(pl.col(col).mean())
            .collect(engine="streaming")
        )
        if result.height == 0:
            return 0.0
        value = result[col][0]
        return float(value) if value is not None else 0.0
    except (ValueError, TypeError, KeyError, pl.exceptions.ColumnNotFoundError):
        return 0.0


# ---------------------------------------------------------------------------
# Section A — Metrics row
# ---------------------------------------------------------------------------


def _build_metrics_row(data: DashboardData) -> str:
    """Build a 6-card metrics grid for the Energy Balance tab.

    Cards: Hydro GWh, Thermal GWh, NCS GWh, Deficit GWh, Avg Spillage
    (m3/s), Curtailment GWh.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string containing the section title and metrics grid.
    """
    hydro_gwh = _compute_total_gwh(data.hydros_lf, "generation_mwh")
    thermal_gwh = _compute_total_gwh(data.thermals_lf, "generation_mwh")
    ncs_gwh = _compute_total_gwh(data.ncs_lf, "generation_mwh")

    # Deficit: filter to non-fictitious buses only
    try:
        deficit_gwh = _compute_total_gwh(
            data.buses_lf.filter(pl.col("bus_id").is_in(data.non_fictitious_bus_ids)),
            "deficit_mwh",
        )
    except (ValueError, TypeError, KeyError, pl.exceptions.ColumnNotFoundError):
        deficit_gwh = 0.0

    # Spillage: expressed as average m3/s (not GWh — no productivity data here)
    spillage_avg = _compute_total_avg(data.hydros_lf, "spillage_m3s")

    # Curtailment: expressed in GWh
    curtailment_gwh = _compute_total_gwh(data.ncs_lf, "curtailment_mwh")

    cards = [
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
        metric_card(
            f"{deficit_gwh:,.0f} GWh",
            "Total Deficit GWh",
            color=COLORS["deficit"],
        ),
        metric_card(
            f"{spillage_avg:,.1f} m\u00b3/s",
            "Avg Spillage (m\u00b3/s)",
            color=COLORS["spillage"],
        ),
        metric_card(
            f"{curtailment_gwh:,.0f} GWh",
            "Total Curtailment GWh",
            color=COLORS["curtailment"],
        ),
    ]
    return section_title("Energy Metrics") + metrics_grid(cards)


# ---------------------------------------------------------------------------
# Section B — System generation mix hero chart
# ---------------------------------------------------------------------------


def _chart_gen_mix_hero(data: DashboardData) -> go.Figure:
    """Build the system-wide generation mix stacked area hero chart.

    Shows stage-average MW for hydro, thermal, and NCS (stacked areas) plus
    LP load as a dashed overlay.  Uses ``data.non_fictitious_bus_ids`` for
    the LP load filter instead of a hardcoded bus list.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    h_gen = _stage_avg_mw(data.hydros_lf, "generation_mwh", data.stage_hours, [])
    t_gen = _stage_avg_mw(data.thermals_lf, "generation_mwh", data.stage_hours, [])
    n_gen = _stage_avg_mw(data.ncs_lf, "generation_mwh", data.stage_hours, [])

    assert isinstance(h_gen, dict)
    assert isinstance(t_gen, dict)
    assert isinstance(n_gen, dict)

    load_ser = _compute_lp_load(
        data.load_stats,
        data.load_factors_list,
        data.stage_hours,
        data.block_hours,
        bus_filter=data.non_fictitious_bus_ids,
    )

    stages = sorted(h_gen.keys())
    xlabels = stage_x_labels(stages, data.stage_labels)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[h_gen.get(s, 0) for s in stages],
            name="Hydro",
            stackgroup="gen",
            mode="lines",
            line={"color": GENERATION_COLORS["hydro"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[t_gen.get(s, 0) for s in stages],
            name="Thermal",
            stackgroup="gen",
            mode="lines",
            line={"color": GENERATION_COLORS["thermal"]},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xlabels,
            y=[n_gen.get(s, 0) for s in stages],
            name="NCS",
            stackgroup="gen",
            mode="lines",
            line={"color": GENERATION_COLORS["ncs"]},
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
        xaxis_title="Stage",
        yaxis_title="MW",
        legend=_LEGEND,
        margin=_MARGIN,
    )
    return fig


# ---------------------------------------------------------------------------
# Section C — Generation by bus (collapsible facet)
# ---------------------------------------------------------------------------


def _chart_gen_by_bus(data: DashboardData) -> go.Figure:
    """Build subplots per non-fictitious bus: hydro+thermal+NCS+net import vs LP load.

    One row per bus in ``data.non_fictitious_bus_ids``.  Each subplot shows
    stacked area traces for hydro, thermal, and NCS generation plus a net
    import area trace and a dashed LP load overlay.  Y-axis scales are
    independent across buses.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one subplot row per bus.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)
    n_buses = len(bus_ids)

    if n_buses == 0:
        fig = go.Figure()
        fig.update_layout(
            title="No non-fictitious buses found",
            legend=_LEGEND,
            margin=_MARGIN,
        )
        return fig

    # Build mapping DataFrames for LazyFrame joins
    hbus_df = pl.DataFrame(
        {
            "hydro_id": list(data.hydro_bus_map.keys()),
            "bus_id": list(data.hydro_bus_map.values()),
        }
    )
    tbus_map = {k: v["bus_id"] for k, v in data.thermal_meta.items()}
    tbus_df = pl.DataFrame(
        {
            "thermal_id": list(tbus_map.keys()),
            "bus_id": list(tbus_map.values()),
        }
    )
    nbus_df = pl.DataFrame(
        {
            "non_controllable_id": list(data.ncs_bus_map.keys()),
            "bus_id": list(data.ncs_bus_map.values()),
        }
    )
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )

    def _bus_stage_avg(
        lf: pl.LazyFrame,
        mwh_col: str,
        id_col: str,
        mapping_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Join lf to mapping_df, compute per-bus stage-average MW."""
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

    h_by_bus = _bus_stage_avg(data.hydros_lf, "generation_mwh", "hydro_id", hbus_df)
    t_by_bus = _bus_stage_avg(data.thermals_lf, "generation_mwh", "thermal_id", tbus_df)
    n_by_bus = _bus_stage_avg(
        data.ncs_lf, "generation_mwh", "non_controllable_id", nbus_df
    )

    # Collect the union of stage IDs across all entity types
    stages_all = sorted(
        set(h_by_bus["stage_id"].to_list())
        | set(t_by_bus["stage_id"].to_list())
        | set(n_by_bus["stage_id"].to_list())
    )
    if not stages_all:
        stages_all = sorted(data.stage_hours.keys())
    xlabels = stage_x_labels(stages_all, data.stage_labels)

    def _to_map(df: pl.DataFrame, bus_id: int) -> dict[int, float]:
        sub = df.filter(pl.col("bus_id") == bus_id)
        return dict(zip(sub["stage_id"].to_list(), sub["_avg_mw"].to_list()))

    # Compute net exchange import per bus
    ex_import: dict[int, dict[int, float]] = {}
    for bus_id in bus_ids:
        parts: list[pl.DataFrame] = []
        for ln in data.line_meta:
            lid, src, tgt = ln["id"], ln["source_bus_id"], ln["target_bus_id"]
            if src != bus_id and tgt != bus_id:
                continue
            sign = 1.0 if tgt == bus_id else -1.0
            part = (
                data.exchanges_lf.filter(pl.col("line_id") == lid)
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
        subplot_titles=[data.bus_names.get(b, str(b)) for b in bus_ids],
        vertical_spacing=max(0.06, 0.35 / max(n_buses, 1)),
    )

    for row_idx, bus_id in enumerate(bus_ids, start=1):
        show_legend = row_idx == 1
        h_gen = _to_map(h_by_bus, bus_id)
        t_gen = _to_map(t_by_bus, bus_id)
        n_gen = _to_map(n_by_bus, bus_id)
        load_s = _compute_lp_load(
            data.load_stats,
            data.load_factors_list,
            data.stage_hours,
            data.block_hours,
            bus_filter=[bus_id],
        )
        net_imp = ex_import.get(bus_id, {})

        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[h_gen.get(s, 0) for s in stages_all],
                name="Hydro",
                stackgroup=f"g{bus_id}",
                mode="lines",
                line={"color": GENERATION_COLORS["hydro"]},
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
                mode="lines",
                line={"color": GENERATION_COLORS["thermal"]},
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
                mode="lines",
                line={"color": GENERATION_COLORS["ncs"]},
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
                mode="lines",
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
        height=350 * n_buses,
        legend=_LEGEND,
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig


# ---------------------------------------------------------------------------
# Section D — Deficit & Excess by bus (collapsible)
# ---------------------------------------------------------------------------


def _render_deficit_excess(data: DashboardData) -> str:
    """Build two side-by-side charts: deficit by bus (left) and excess by bus (right).

    Each chart shows one mean+p50+band line per non-fictitious bus.  Data
    source is ``buses_lf`` with columns ``deficit_mwh`` and ``excess_mwh``,
    converted to stage-average MW by dividing by stage hours.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string: a ``collapsible_section`` wrapping a ``chart_grid``
        with two chart cards.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )

    def _bus_mw_collected(col: str) -> pl.DataFrame:
        """Return per-(scenario, stage, bus) MW for *col* from buses_lf."""
        try:
            return (
                data.buses_lf.filter(pl.col("bus_id").is_in(bus_ids))
                .group_by(["scenario_id", "stage_id", "bus_id"])
                .agg(pl.col(col).sum())
                .join(hours_df.lazy(), on="stage_id")
                .with_columns((pl.col(col) / pl.col("_hours")).alias("_mw"))
                .select(["scenario_id", "stage_id", "bus_id", "_mw"])
                .sort("stage_id")
                .collect(engine="streaming")
            )
        except (pl.exceptions.ColumnNotFoundError, KeyError):
            return pl.DataFrame()

    def _bus_chart(col: str, title: str, y_label: str) -> go.Figure:
        """Build a go.Figure with one mean+p50+band line per bus."""
        raw = _bus_mw_collected(col)
        fig = go.Figure()
        if raw.is_empty():
            fig.add_annotation(
                text="No data.",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )
            fig.update_layout(
                xaxis_title="Stage", yaxis_title=y_label, legend=_LEGEND, margin=_MARGIN
            )
            return fig

        stages = sorted(raw["stage_id"].unique().to_list())
        xlabels = stage_x_labels(stages, data.stage_labels)

        for i, bus_id in enumerate(bus_ids):
            sub = (
                raw.filter(pl.col("bus_id") == bus_id)
                .select(["scenario_id", "stage_id", "_mw"])
                .to_pandas()
            )
            if sub.empty:
                continue
            pct = compute_percentiles(sub, ["stage_id"], "_mw")
            # Map stage_id -> x-label for display
            pct["_x"] = pct["stage_id"].map(dict(zip(stages, xlabels)))
            color = BUS_COLORS[i % len(BUS_COLORS)]
            bus_name = data.bus_names.get(bus_id, str(bus_id))
            add_mean_p50_band(fig, pct, "_x", bus_name, color)

        fig.update_layout(
            xaxis_title="Stage", yaxis_title=y_label, legend=_LEGEND, margin=_MARGIN
        )
        return fig

    deficit_fig = _bus_chart("deficit_mwh", "Deficit by Bus", "Deficit (MW)")
    excess_fig = _bus_chart("excess_mwh", "Excess by Bus", "Excess (MW)")

    deficit_card = make_chart_card(
        deficit_fig,
        "Deficit by Bus (stage-avg MW)",
        "v2-energy-deficit-by-bus",
        height=420,
    )
    excess_card = make_chart_card(
        excess_fig,
        "Excess by Bus (stage-avg MW)",
        "v2-energy-excess-by-bus",
        height=420,
    )

    return collapsible_section(
        "Deficit & Excess",
        chart_grid([deficit_card, excess_card]),
        section_id="v2-energy-deficit-excess-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section E — Reservoir Storage (collapsible, collapsed by default)
# ---------------------------------------------------------------------------


def _render_reservoir_storage(data: DashboardData) -> str:
    """Build system aggregate storage trajectory and per-bus storage % charts.

    Left chart: sum of ``storage_final_hm3`` across all hydros per stage,
    with mean+p50+p10-p90 band and horizontal reference lines for total
    system ``vol_min`` / ``vol_max`` from ``data.hydro_meta``.

    Right chart: per-bus storage as percentage of useful volume
    (``vol_max - vol_min`` for hydros in that bus), one subplot per
    non-fictitious bus that has hydros.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string: a ``collapsible_section`` wrapping a ``chart_grid``
        with two chart cards.
    """
    bus_ids = sorted(data.non_fictitious_bus_ids)

    # ------------------------------------------------------------------
    # System aggregate storage
    # ------------------------------------------------------------------
    try:
        sys_stor = (
            data.hydros_lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "stage_id"])
            .agg(pl.col("storage_final_hm3").sum())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        sys_stor = pl.DataFrame()

    sys_fig = go.Figure()
    if not sys_stor.is_empty():
        pct = compute_percentiles(
            sys_stor.to_pandas(), ["stage_id"], "storage_final_hm3"
        )
        stages = sorted(pct["stage_id"].tolist())
        xlabels = stage_x_labels(stages, data.stage_labels)
        pct["_x"] = pct["stage_id"].map(dict(zip(stages, xlabels)))
        add_mean_p50_band(sys_fig, pct, "_x", "System Storage", COLORS["hydro"])

        # Reference lines for total vol_min / vol_max
        total_vol_max = sum(m.get("vol_max", 0) for m in data.hydro_meta.values())
        total_vol_min = sum(m.get("vol_min", 0) for m in data.hydro_meta.values())

        bounds_df = pd.DataFrame(
            {
                "_x": xlabels,
                "vol_min": [total_vol_min] * len(xlabels),
                "vol_max": [total_vol_max] * len(xlabels),
            }
        )
        add_bounds_overlay(
            sys_fig, bounds_df, "_x", min_col="vol_min", max_col="vol_max"
        )
    else:
        sys_fig.add_annotation(
            text="No data.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )

    sys_fig.update_layout(
        xaxis_title="Stage", yaxis_title="Storage (hm³)", legend=_LEGEND, margin=_MARGIN
    )

    # ------------------------------------------------------------------
    # By-bus storage as % of useful volume
    # ------------------------------------------------------------------
    # Build hydro-to-bus mapping and per-bus useful volume
    hydro_to_bus: dict[int, int] = data.hydro_bus_map
    bus_useful_vol: dict[int, float] = {}
    for hid, bus_id in hydro_to_bus.items():
        if bus_id not in bus_useful_vol:
            bus_useful_vol[bus_id] = 0.0
        meta = data.hydro_meta.get(hid, {})
        vol_max = float(meta.get("vol_max", 0))
        vol_min = float(meta.get("vol_min", 0))
        bus_useful_vol[bus_id] += vol_max - vol_min

    # Filter to buses that have hydros and are non-fictitious
    active_bus_ids = [b for b in bus_ids if b in bus_useful_vol]

    try:
        hbus_df = pl.DataFrame(
            {
                "hydro_id": list(hydro_to_bus.keys()),
                "bus_id": list(hydro_to_bus.values()),
            }
        )
        bus_stor = (
            data.hydros_lf.filter(pl.col("block_id") == 0)
            .join(hbus_df.lazy(), on="hydro_id")
            .filter(pl.col("bus_id").is_in(active_bus_ids))
            .group_by(["scenario_id", "stage_id", "bus_id"])
            .agg(pl.col("storage_final_hm3").sum())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        bus_stor = pl.DataFrame()

    n_buses = len(active_bus_ids)
    if n_buses > 0 and not bus_stor.is_empty():
        bus_fig = make_subplots(
            rows=n_buses,
            cols=1,
            shared_xaxes=False,
            subplot_titles=[data.bus_names.get(b, str(b)) for b in active_bus_ids],
            vertical_spacing=max(0.06, 0.35 / max(n_buses, 1)),
        )
        all_stages = sorted(bus_stor["stage_id"].unique().to_list())
        xlabels = stage_x_labels(all_stages, data.stage_labels)

        for row_idx, bus_id in enumerate(active_bus_ids, start=1):
            sub = bus_stor.filter(pl.col("bus_id") == bus_id).to_pandas()
            useful_vol = bus_useful_vol.get(bus_id, 0.0)
            if useful_vol > 0.0:
                sub = sub.copy()
                sub["_pct"] = sub["storage_final_hm3"] / useful_vol * 100.0
            else:
                sub = sub.copy()
                sub["_pct"] = 0.0

            pct = compute_percentiles(sub, ["stage_id"], "_pct")
            pct["_x"] = pct["stage_id"].map(dict(zip(all_stages, xlabels)))
            color = BUS_COLORS[row_idx - 1 % len(BUS_COLORS)]
            bus_name = data.bus_names.get(bus_id, str(bus_id))
            add_mean_p50_band(bus_fig, pct, "_x", bus_name, color, row=row_idx, col=1)
            bus_fig.update_yaxes(title_text="%", row=row_idx, col=1)

        bus_fig.update_layout(
            height=320 * n_buses + 60,
            legend=_LEGEND,
            margin=dict(l=60, r=30, t=80, b=50),
        )
    else:
        bus_fig = go.Figure()
        bus_fig.add_annotation(
            text="No data.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )
        bus_fig.update_layout(legend=_LEGEND, margin=_MARGIN)

    sys_card = make_chart_card(
        sys_fig,
        "System Aggregate Reservoir Storage (hm³)",
        "v2-energy-storage-system",
        height=420,
    )
    bus_card = make_chart_card(
        bus_fig,
        "Reservoir Storage by Bus (% useful volume)",
        "v2-energy-storage-by-bus",
        height=max(320 * n_buses + 60, 420),
    )

    return collapsible_section(
        "Reservoir Storage",
        chart_grid([sys_card, bus_card]),
        section_id="v2-energy-storage-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Section F — NCS & Curtailment (collapsible, collapsed by default)
# ---------------------------------------------------------------------------


def _render_ncs_curtailment(data: DashboardData) -> str:
    """Build NCS generation vs available capacity and curtailment bar charts.

    Left chart: stacked area of NCS generation (mean per stage, stage-avg MW)
    with an overlay line for available capacity (block-hours-weighted mean
    ``available_mw`` per stage).

    Right chart: mean curtailment MWh per stage as a bar chart.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string: a ``collapsible_section`` wrapping a ``chart_grid``
        with two chart cards.
    """
    hours_df = pl.DataFrame(
        {
            "stage_id": list(data.stage_hours.keys()),
            "_hours": list(data.stage_hours.values()),
        }
    )
    # Build block-hours lookup LazyFrame for available_mw weighting
    bh_lazy = data.bh_df.lazy()

    # ------------------------------------------------------------------
    # NCS generation (stage-avg MW, mean across scenarios)
    # ------------------------------------------------------------------
    try:
        ncs_gen_raw = (
            data.ncs_lf.group_by(["scenario_id", "stage_id"])
            .agg(pl.col("generation_mwh").sum())
            .join(hours_df.lazy(), on="stage_id")
            .with_columns(
                (pl.col("generation_mwh") / pl.col("_hours")).alias("_gen_mw")
            )
            .group_by("stage_id")
            .agg(pl.col("_gen_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        ncs_gen_raw = pl.DataFrame()

    # ------------------------------------------------------------------
    # NCS available capacity (block-hours-weighted mean per stage,
    # then mean across scenarios)
    # ------------------------------------------------------------------
    try:
        ncs_avail_raw = (
            data.ncs_lf.join(bh_lazy, on=["stage_id", "block_id"])
            .group_by(["scenario_id", "stage_id"])
            .agg(
                (pl.col("available_mw") * pl.col("_bh")).sum().alias("_wmw"),
                pl.col("_bh").sum().alias("_bh_sum"),
            )
            .with_columns((pl.col("_wmw") / pl.col("_bh_sum")).alias("_avail_mw"))
            .group_by("stage_id")
            .agg(pl.col("_avail_mw").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        ncs_avail_raw = pl.DataFrame()

    # ------------------------------------------------------------------
    # NCS curtailment (mean MWh per stage, across scenarios)
    # ------------------------------------------------------------------
    try:
        ncs_curtail_raw = (
            data.ncs_lf.group_by(["scenario_id", "stage_id"])
            .agg(pl.col("curtailment_mwh").sum())
            .group_by("stage_id")
            .agg(pl.col("curtailment_mwh").mean())
            .sort("stage_id")
            .collect(engine="streaming")
        )
    except (pl.exceptions.ColumnNotFoundError, KeyError):
        ncs_curtail_raw = pl.DataFrame()

    # ------------------------------------------------------------------
    # Build left chart: generation stacked area + available capacity overlay
    # ------------------------------------------------------------------
    gen_fig = go.Figure()

    if not ncs_gen_raw.is_empty():
        stages = sorted(ncs_gen_raw["stage_id"].to_list())
        xlabels = stage_x_labels(stages, data.stage_labels)
        gen_map = dict(
            zip(ncs_gen_raw["stage_id"].to_list(), ncs_gen_raw["_gen_mw"].to_list())
        )

        gen_fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[gen_map.get(s, 0) for s in stages],
                name="NCS Generation",
                stackgroup="ncs_gen",
                mode="lines",
                line={"color": GENERATION_COLORS["ncs"]},
            )
        )

        if not ncs_avail_raw.is_empty():
            avail_map = dict(
                zip(
                    ncs_avail_raw["stage_id"].to_list(),
                    ncs_avail_raw["_avail_mw"].to_list(),
                )
            )
            gen_fig.add_trace(
                go.Scatter(
                    x=xlabels,
                    y=[avail_map.get(s, 0) for s in stages],
                    name="Available Capacity",
                    mode="lines",
                    line={"color": COLORS["curtailment"], "width": 2, "dash": "dash"},
                )
            )
    else:
        gen_fig.add_annotation(
            text="No data.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )

    gen_fig.update_layout(
        xaxis_title="Stage", yaxis_title="MW", legend=_LEGEND, margin=_MARGIN
    )

    # ------------------------------------------------------------------
    # Build right chart: curtailment bar chart
    # ------------------------------------------------------------------
    curtail_fig = go.Figure()

    if not ncs_curtail_raw.is_empty():
        stages_c = sorted(ncs_curtail_raw["stage_id"].to_list())
        xlabels_c = stage_x_labels(stages_c, data.stage_labels)
        curtail_map = dict(
            zip(
                ncs_curtail_raw["stage_id"].to_list(),
                ncs_curtail_raw["curtailment_mwh"].to_list(),
            )
        )

        curtail_fig.add_trace(
            go.Bar(
                x=xlabels_c,
                y=[curtail_map.get(s, 0) for s in stages_c],
                name="Curtailment",
                marker_color=COLORS["curtailment"],
            )
        )
    else:
        curtail_fig.add_annotation(
            text="No data.", showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )

    curtail_fig.update_layout(
        xaxis_title="Stage", yaxis_title="MWh", legend=_LEGEND, margin=_MARGIN
    )

    gen_card = make_chart_card(
        gen_fig,
        "NCS Generation vs Available Capacity (stage-avg MW)",
        "v2-energy-ncs-gen-avail",
        height=420,
    )
    curtail_card = make_chart_card(
        curtail_fig,
        "NCS Curtailment by Stage (mean MWh)",
        "v2-energy-ncs-curtailment",
        height=420,
    )

    return collapsible_section(
        "NCS & Curtailment",
        chart_grid([gen_card, curtail_card]),
        section_id="v2-energy-ncs-section",
        default_collapsed=False,
    )


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Energy Balance v2 tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Energy Balance tab content area.

    Sections:
      A. 6-card metrics row.
      B. System generation mix hero chart (stacked area + LP load).
      C. Generation by bus — collapsible facet, one subplot per
         non-fictitious bus.
      D. Deficit & Excess — two side-by-side per-bus charts with
         mean+p50+band, expanded by default.
      E. Reservoir Storage — system aggregate trajectory + per-bus %,
         collapsed by default.
      F. NCS & Curtailment — generation vs available + curtailment bar,
         collapsed by default.

    Args:
        data: Full :class:`~cobre_bridge.dashboard.data.DashboardData` instance.

    Returns:
        An HTML string ready to be inserted into the tab content area.
    """
    # Section A — metrics row
    metrics_html = _build_metrics_row(data)

    # Section B — hero chart
    hero_fig = _chart_gen_mix_hero(data)
    hero_html = chart_grid(
        [
            make_chart_card(
                hero_fig,
                "System-Wide Generation Mix vs LP Load (stage-avg MW)",
                "v2-energy-gen-mix-hero",
                height=420,
            )
        ],
        single=True,
    )

    # Section C — generation by bus (collapsible, default expanded)
    by_bus_fig = _chart_gen_by_bus(data)
    n_buses = len(data.non_fictitious_bus_ids)
    by_bus_html = collapsible_section(
        "Generation by Bus",
        chart_grid(
            [
                make_chart_card(
                    by_bus_fig,
                    "Generation + Net Import vs LP Load by Bus (stage-avg MW)",
                    "v2-energy-gen-by-bus",
                    height=max(350 * n_buses, 400),
                )
            ],
            single=True,
        ),
        section_id="v2-energy-gen-by-bus-section",
        default_collapsed=False,
    )

    # Section D — deficit & excess (collapsible, default expanded)
    deficit_excess_html = _render_deficit_excess(data)

    # Section E — reservoir storage (collapsible, default collapsed)
    storage_html = _render_reservoir_storage(data)

    # Section F — NCS & curtailment (collapsible, default collapsed)
    ncs_html = _render_ncs_curtailment(data)

    return (
        metrics_html
        + hero_html
        + by_bus_html
        + deficit_excess_html
        + storage_html
        + ncs_html
    )
