"""v2 Network tab — line explorer, capacity utilisation heatmap, and bus balance.

Implements three sections of the Network tab (Tab 7):
  A. Line Explorer -- split-pane dropdown + per-line net/direct/reverse flow
     charts with p10/p50/p90 bands, following the exchanges.py JSON+JS pattern.
     Includes a third per-line capacity utilisation chart (|net flow| / capacity %).
  B. Capacity Utilisation Heatmap -- single net utilisation heatmap using
     go.Figure, with RdYlGn_r colorscale.
  C. Bus Balance -- horizontal grouped bar chart showing mean net import/export
     per bus across all stages (all buses included, no fictitious filtering).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl

from cobre_bridge.dashboard.chart_helpers import make_chart_card
from cobre_bridge.dashboard.data import entity_name
from cobre_bridge.ui.html import chart_grid, section_title, wrap_chart
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    fig_to_html,
    stage_x_labels,
)

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-network"
TAB_LABEL = "Network"
TAB_ORDER = 60

# ---------------------------------------------------------------------------
# Section A: Line Explorer
# ---------------------------------------------------------------------------


def build_line_explorer(
    exchanges_lf: pl.LazyFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
    line_bounds: pd.DataFrame | None = None,
    line_meta: list[dict] | None = None,
) -> str:
    """Build HTML with embedded per-line p10/p50/p90 data and JS dropdown.

    Follows the same JSON-embed + Plotly.react() pattern as
    ``exchanges.py`` ``build_interactive_exchange_detail()``.  Renders
    three charts per line: net flow (id: ``nw-net``), direct/reverse
    flow (id: ``nw-dir``), and capacity utilisation (id: ``nw-util``).

    Args:
        exchanges_lf: LazyFrame with columns ``scenario_id``, ``stage_id``,
            ``block_id``, ``line_id``, ``net_flow_mw``, ``direct_flow_mw``,
            ``reverse_flow_mw``.
        names: Entity name mapping ``("lines", line_id) -> name``.
        stage_labels: Stage id to human-readable label mapping.
        bh_df: Block-hours DataFrame with columns ``stage_id``, ``block_id``,
            ``_bh``.
        line_bounds: pandas DataFrame with columns ``line_id``, ``stage_id``,
            ``direct_mw``, ``reverse_mw``.  May be None or empty.
        line_meta: List of line metadata dicts, each with keys ``id``,
            ``direct_capacity_mw``, ``reverse_capacity_mw``.  May be None.

    Returns:
        HTML string with ``<select id="nw-select">``, chart divs
        ``<div id="nw-net">``, ``<div id="nw-dir">``, ``<div id="nw-util">``,
        and inline JS.
    """
    flow_cols = ["net_flow_mw", "direct_flow_mw", "reverse_flow_mw"]
    schema = exchanges_lf.collect_schema()
    avail_flow_cols = [c for c in flow_cols if c in schema]

    # Check if there is any data at all
    schema_names = list(schema.names())
    if not avail_flow_cols or "line_id" not in schema_names:
        return "<p>No line flow data available.</p>"

    ex0 = (
        exchanges_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "line_id"])
        .agg(
            *[
                ((pl.col(c) * pl.col("_bh")).sum() / pl.col("_bh").sum()).alias(c)
                for c in avail_flow_cols
            ]
        )
        .collect(engine="streaming")
    )

    if ex0.height == 0:
        return "<p>No line flow data available.</p>"

    stages = sorted(ex0["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)
    line_ids = sorted(ex0["line_id"].unique().to_list())

    # Build capacity lookups for utilisation computation
    static_direct_cap: dict[int, float] = {}
    static_reverse_cap: dict[int, float] = {}
    if line_meta:
        for lm in line_meta:
            lid_m = lm["id"]
            static_direct_cap[lid_m] = float(lm.get("direct_capacity_mw", 1.0))
            static_reverse_cap[lid_m] = float(lm.get("reverse_capacity_mw", 1.0))

    bounds_lookup_le: dict[tuple[int, int], tuple[float, float]] = {}
    if line_bounds is not None and not line_bounds.empty:
        lb_indexed = line_bounds.set_index(["line_id", "stage_id"])
        for (lid_b, sid_b), row_b in lb_indexed.iterrows():
            bounds_lookup_le[(int(lid_b), int(sid_b))] = (
                float(row_b.get("direct_mw", 1.0)),
                float(row_b.get("reverse_mw", 1.0)),
            )

    line_data: dict[str, dict] = {}
    for lid in line_ids:
        lname = entity_name(names, "lines", lid)
        ldf = ex0.filter(pl.col("line_id") == lid)
        entry: dict = {"name": lname}
        net_pcts_map: dict[int, dict[str, float]] = {}
        for col, prefix in [
            ("net_flow_mw", "net"),
            ("direct_flow_mw", "direct"),
            ("reverse_flow_mw", "reverse"),
        ]:
            if col not in ldf.columns:
                for sfx in ["p10", "p50", "p90"]:
                    entry[f"{prefix}_{sfx}"] = [0.0] * len(stages)
                continue
            pcts = (
                ldf.group_by("stage_id")
                .agg(
                    pl.col(col).quantile(0.1, interpolation="linear").alias("p10"),
                    pl.col(col).quantile(0.5, interpolation="linear").alias("p50"),
                    pl.col(col).quantile(0.9, interpolation="linear").alias("p90"),
                )
                .sort("stage_id")
            )
            pcts_map: dict[int, dict[str, float]] = {}
            for row in pcts.iter_rows(named=True):
                pcts_map[row["stage_id"]] = {
                    "p10": row["p10"],
                    "p50": row["p50"],
                    "p90": row["p90"],
                }
            for sfx in ["p10", "p50", "p90"]:
                entry[f"{prefix}_{sfx}"] = [
                    round(pcts_map.get(s, {}).get(sfx, 0.0), 2) for s in stages
                ]
            if prefix == "net":
                net_pcts_map = pcts_map

        # Compute capacity utilisation: |net_flow| / max(d_cap, r_cap) * 100
        for sfx in ["p10", "p50", "p90"]:
            util_vals: list[float] = []
            for s in stages:
                if bounds_lookup_le:
                    d_cap, r_cap = bounds_lookup_le.get(
                        (lid, s),
                        (
                            static_direct_cap.get(lid, 1.0),
                            static_reverse_cap.get(lid, 1.0),
                        ),
                    )
                else:
                    d_cap = static_direct_cap.get(lid, 1.0)
                    r_cap = static_reverse_cap.get(lid, 1.0)
                cap = max(d_cap, r_cap, 0.1)
                net_val = net_pcts_map.get(s, {}).get(sfx, 0.0)
                util = min(abs(net_val) / cap * 100.0, 100.0)
                util_vals.append(round(util, 2))
            entry[f"util_{sfx}"] = util_vals

        line_data[str(lid)] = entry

    options_html = "\n".join(
        f'<option value="{lid}">{d["name"]} (id={lid})</option>'
        for lid, d in sorted(line_data.items(), key=lambda x: x[1]["name"])
    )
    data_json = json.dumps(line_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    chart_rows = (
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="nw-net" style="width:100%;height:350px;"></div></div>'
        "</div>"
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="nw-dir" style="width:100%;height:350px;"></div></div>'
        "</div>"
        '<div class="chart-grid-single">'
        '<div class="chart-card"><div id="nw-util" style="width:100%;height:350px;"></div></div>'
        "</div>"
    )

    return (
        '<div style="margin-bottom:16px;">'
        '<label for="nw-select" style="font-weight:600;margin-right:8px;">Select Line:</label>'
        '<select id="nw-select" onchange="updateNetworkDetail()" '
        'style="padding:8px 12px;font-size:0.9rem;border-radius:4px;border:1px solid #ccc;min-width:280px;">'
        + options_html
        + "</select>"
        + "</div>"
        + chart_rows
        + "<script>\n"
        + "const NW_DATA = "
        + data_json
        + ";\n"
        + "const NW_LABELS = "
        + labels_json
        + ";\n"
        + r"""
function _nw_band(lbl, p10, p90, color) {
  return {x: NW_LABELS.concat(NW_LABELS.slice().reverse()),
          y: p90.concat(p10.slice().reverse()),
          fill:'toself', fillcolor:color, line:{color:'rgba(0,0,0,0)'},
          name:lbl, showlegend:true, hoverinfo:'skip'};
}
function _nw_line(nm, y, c, w, dash) {
  return {x:NW_LABELS, y:y, name:nm, line:{color:c, width:w||2, dash:dash||'solid'}};
}
var _NW_L = {hovermode:'x unified', margin:{l:60,r:20,t:60,b:10},
             legend:{orientation:'h',yanchor:'top',y:-0.15,xanchor:'center',x:0.5,font:{size:11}}};
var _NW_C = {responsive:true};
function _nw_lo(extra){return Object.assign({},_NW_L,extra);}

function updateNetworkDetail() {
  var lid = document.getElementById('nw-select').value;
  var d = NW_DATA[lid]; if(!d) return;

  var zeroLine = Array(NW_LABELS.length).fill(0);

  Plotly.react('nw-net', [
    _nw_band('P10\u2013P90', d.net_p10, d.net_p90, 'rgba(74,144,184,0.15)'),
    _nw_line('P50', d.net_p50, '#4A90B8'),
    _nw_line('P10', d.net_p10, '#4A90B8', 1, 'dot'),
    _nw_line('P90', d.net_p90, '#4A90B8', 1, 'dot'),
    {x:NW_LABELS, y:zeroLine, name:'Zero', line:{color:'gray',width:1,dash:'dot'}, showlegend:false},
  ], _nw_lo({title:d.name+' \u2014 Net Flow (MW)', yaxis:{title:'Net Flow (MW)'}}), _NW_C);

  var rev_p10_neg = d.reverse_p10.map(function(v){return -v;});
  var rev_p50_neg = d.reverse_p50.map(function(v){return -v;});
  var rev_p90_neg = d.reverse_p90.map(function(v){return -v;});
  Plotly.react('nw-dir', [
    _nw_band('Direct P10\u2013P90', d.direct_p10, d.direct_p90, 'rgba(74,139,111,0.18)'),
    _nw_line('Direct P50', d.direct_p50, '#4A8B6F'),
    _nw_line('Direct P10', d.direct_p10, '#4A8B6F', 1, 'dot'),
    _nw_line('Direct P90', d.direct_p90, '#4A8B6F', 1, 'dot'),
    _nw_band('Reverse P10\u2013P90 (neg)', rev_p10_neg, rev_p90_neg, 'rgba(220,76,76,0.15)'),
    _nw_line('Reverse P50 (neg)', rev_p50_neg, '#DC4C4C'),
    _nw_line('Reverse P10 (neg)', rev_p10_neg, '#DC4C4C', 1, 'dot'),
    _nw_line('Reverse P90 (neg)', rev_p90_neg, '#DC4C4C', 1, 'dot'),
    {x:NW_LABELS, y:zeroLine, name:'Zero', line:{color:'gray',width:1,dash:'dot'}, showlegend:false},
  ], _nw_lo({title:d.name+' \u2014 Direct / Reverse Flow (MW)', yaxis:{title:'Flow (MW)'}}), _NW_C);

  var ref100 = Array(NW_LABELS.length).fill(100);
  Plotly.react('nw-util', [
    _nw_band('Util P10\u2013P90', d.util_p10, d.util_p90, 'rgba(245,166,35,0.18)'),
    _nw_line('Util P50', d.util_p50, '#F5A623'),
    {x:NW_LABELS, y:ref100, name:'100%', line:{color:'#DC4C4C',width:1.5,dash:'dash'}, showlegend:true},
  ], _nw_lo({title:d.name+' \u2014 Capacity Utilisation (%)', yaxis:{title:'% Capacity', range:[0, 110]}}), _NW_C);
}
document.addEventListener('DOMContentLoaded', function(){setTimeout(updateNetworkDetail,100);});
"""
        + "</script>"
    )


# ---------------------------------------------------------------------------
# Section B: Capacity Utilisation Heatmap
# ---------------------------------------------------------------------------


def build_heatmap(
    exchanges_lf: pl.LazyFrame,
    line_bounds: pd.DataFrame,
    line_meta: list[dict],
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Build a single net capacity utilisation heatmap.

    Computes block-hours weighted average net flow per (scenario, stage, line),
    then averages across scenarios.  Capacity is taken from ``line_bounds``
    per stage when available, falling back to the static capacities in
    ``line_meta``.  Heatmap value = ``|mean_net_flow| / max(d_cap, r_cap) * 100``,
    clamped to [0, 100].

    Args:
        exchanges_lf: Line flow LazyFrame.
        line_bounds: pandas DataFrame with columns ``line_id``, ``stage_id``,
            ``direct_mw``, ``reverse_mw``.  May be empty.
        line_meta: List of line metadata dicts, each with keys ``id``,
            ``direct_capacity_mw``, ``reverse_capacity_mw``.
        names: Entity name mapping.
        stage_labels: Stage id to label mapping.
        bh_df: Block-hours DataFrame.

    Returns:
        HTML string containing the Plotly heatmap figure.
    """
    schema = exchanges_lf.collect_schema()
    schema_names = list(schema.names())
    net_col_present = "net_flow_mw" in schema_names
    direct_col_present = "direct_flow_mw" in schema_names
    reverse_col_present = "reverse_flow_mw" in schema_names

    if not net_col_present and not (direct_col_present or reverse_col_present):
        return "<p>No capacity data available.</p>"

    # Build net flow agg expression: prefer net_flow_mw; else compute direct - reverse
    if net_col_present:
        net_agg = (
            (pl.col("net_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
        ).alias("net")
    elif direct_col_present and reverse_col_present:
        net_agg = (
            (
                (pl.col("direct_flow_mw") - pl.col("reverse_flow_mw")) * pl.col("_bh")
            ).sum()
            / pl.col("_bh").sum()
        ).alias("net")
    elif direct_col_present:
        net_agg = (
            (pl.col("direct_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
        ).alias("net")
    else:
        net_agg = (
            (pl.col("reverse_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
        ).alias("net")

    flow_data = (
        exchanges_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "line_id"])
        .agg(net_agg)
        .group_by(["stage_id", "line_id"])
        .agg(pl.col("net").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )

    if flow_data.height == 0:
        return "<p>No capacity utilisation data available.</p>"

    # Build static capacity lookup from line_meta
    static_direct: dict[int, float] = {}
    static_reverse: dict[int, float] = {}
    for lm in line_meta:
        lid = lm["id"]
        static_direct[lid] = float(lm.get("direct_capacity_mw", 1.0))
        static_reverse[lid] = float(lm.get("reverse_capacity_mw", 1.0))

    # Build per-(line_id, stage_id) capacity lookup from line_bounds
    bounds_lookup: dict[tuple[int, int], tuple[float, float]] = {}
    if not line_bounds.empty:
        lb_indexed = line_bounds.set_index(["line_id", "stage_id"])
        for (lid, sid), row in lb_indexed.iterrows():
            bounds_lookup[(int(lid), int(sid))] = (
                float(row.get("direct_mw", 1.0)),
                float(row.get("reverse_mw", 1.0)),
            )

    line_ids = sorted(flow_data["line_id"].unique().to_list())
    stages = sorted(flow_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)
    ynames = [entity_name(names, "lines", lid) for lid in line_ids]

    z_net: list[list[float]] = []
    for lid in line_ids:
        sub = flow_data.filter(pl.col("line_id") == lid)
        net_map: dict[int, float] = dict(
            zip(sub["stage_id"].to_list(), sub["net"].to_list())
        )
        row_net: list[float] = []
        for s in stages:
            if bounds_lookup:
                d_cap, r_cap = bounds_lookup.get(
                    (lid, s),
                    (static_direct.get(lid, 1.0), static_reverse.get(lid, 1.0)),
                )
            else:
                d_cap = static_direct.get(lid, 1.0)
                r_cap = static_reverse.get(lid, 1.0)
            cap = max(d_cap, r_cap, 0.1)
            net_flow = net_map.get(s, 0.0)
            row_net.append(min(abs(net_flow) / cap * 100.0, 100.0))
        z_net.append(row_net)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z_net,
            x=xlabels,
            y=ynames,
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=100,
            colorbar={"title": "%", "len": 0.9},
            showscale=True,
            name="Net Utilisation",
        )
    )
    fig.update_layout(
        title="Net Capacity Utilisation Heatmap (%)",
        height=max(300, len(line_ids) * 60 + 120),
        margin=_MARGIN,
    )
    return wrap_chart(fig_to_html(fig, unified_hover=False))


# ---------------------------------------------------------------------------
# Section C: Bus Balance
# ---------------------------------------------------------------------------


def build_bus_balance(
    exchanges_lf: pl.LazyFrame,
    line_meta: list[dict],
    bus_names: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Build a horizontal bar chart of mean net import/export per bus.

    Net import for a bus = sum of ``net_flow_mw`` for all lines where the
    bus is the **target** minus sum where the bus is the **source**.
    Positive = net importer, negative = net exporter.  All buses in
    ``line_meta`` are included (no fictitious bus filtering per epic-01
    design decision).

    Args:
        exchanges_lf: Line flow LazyFrame with ``net_flow_mw`` column.
        line_meta: List of line metadata dicts with ``id``,
            ``source_bus_id``, ``target_bus_id``.
        bus_names: Mapping from bus_id to display name.
        bh_df: Block-hours DataFrame for weighted averaging.

    Returns:
        HTML string containing the Plotly bar chart.
    """
    import math

    schema = exchanges_lf.collect_schema()
    schema_names = list(schema.names())
    if "net_flow_mw" not in schema_names or not line_meta:
        return "<p>No bus balance data available.</p>"

    # Compute block-hours weighted avg per (scenario, stage, line), then
    # average across stages to get a per-(scenario, line) mean flow.
    # Keeping scenario_id allows p10/p90 computation across the scenario
    # distribution for bus balance error bars.
    per_scen_flow_df = (
        exchanges_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "line_id"])
        .agg(
            ((pl.col("net_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()).alias(
                "net_flow_mw"
            )
        )
        .group_by(["scenario_id", "line_id"])
        .agg(pl.col("net_flow_mw").mean())
        .collect(engine="streaming")
    )

    if per_scen_flow_df.height == 0:
        return "<p>No bus balance data available.</p>"

    # Build per-(scenario, bus) balance by accumulating line flows.
    # target buses receive positive flow, source buses receive negative flow.
    scenario_ids = per_scen_flow_df["scenario_id"].unique().to_list()
    line_ids_in_data = set(per_scen_flow_df["line_id"].unique().to_list())

    # Build lookup: (scenario_id, line_id) -> mean_flow
    flow_lookup: dict[tuple[int, int], float] = {}
    for row in per_scen_flow_df.iter_rows(named=True):
        flow_lookup[(int(row["scenario_id"]), int(row["line_id"]))] = float(
            row["net_flow_mw"]
        )

    # For each scenario compute per-bus balance
    bus_ids_in_meta: set[int] = set()
    for lm in line_meta:
        src = lm.get("source_bus_id")
        tgt = lm.get("target_bus_id")
        if src is not None:
            bus_ids_in_meta.add(int(src))
        if tgt is not None:
            bus_ids_in_meta.add(int(tgt))

    if not bus_ids_in_meta:
        return "<p>No bus balance data available.</p>"

    # per_scenario_bus_balance[bus_id] = [balance_scen0, balance_scen1, ...]
    per_scenario_bus_balance: dict[int, list[float]] = {
        bid: [] for bid in bus_ids_in_meta
    }
    for scen in sorted(scenario_ids):
        bus_bal_scen: dict[int, float] = {bid: 0.0 for bid in bus_ids_in_meta}
        for lm in line_meta:
            lid = int(lm["id"])
            if lid not in line_ids_in_data:
                continue
            src_bus = lm.get("source_bus_id")
            tgt_bus = lm.get("target_bus_id")
            flow = flow_lookup.get((int(scen), lid), 0.0)
            if tgt_bus is not None:
                bus_bal_scen[int(tgt_bus)] = bus_bal_scen.get(int(tgt_bus), 0.0) + flow
            if src_bus is not None:
                bus_bal_scen[int(src_bus)] = bus_bal_scen.get(int(src_bus), 0.0) - flow
        for bid in bus_ids_in_meta:
            per_scenario_bus_balance[bid].append(bus_bal_scen[bid])

    # Derive mean, p10, p90 per bus from scenario distribution
    bus_stats: dict[int, tuple[float, float, float]] = {}
    for bid, values in per_scenario_bus_balance.items():
        if not values:
            continue
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        mean_val = sum(sorted_vals) / n

        # Linear interpolation quantile
        def _quantile(vs: list[float], q: float) -> float:
            if len(vs) == 1:
                return vs[0]
            idx = q * (len(vs) - 1)
            lo = int(idx)
            hi = min(lo + 1, len(vs) - 1)
            frac = idx - lo
            return vs[lo] * (1.0 - frac) + vs[hi] * frac

        p10_val = _quantile(sorted_vals, 0.1)
        p90_val = _quantile(sorted_vals, 0.9)
        bus_stats[bid] = (mean_val, p10_val, p90_val)

    if not bus_stats:
        return "<p>No bus balance data available.</p>"

    sorted_bus_ids = sorted(bus_stats.keys())
    ynames = [bus_names.get(bid, str(bid)) for bid in sorted_bus_ids]
    x_values = [bus_stats[bid][0] for bid in sorted_bus_ids]
    p10_values = [bus_stats[bid][1] for bid in sorted_bus_ids]
    p90_values = [bus_stats[bid][2] for bid in sorted_bus_ids]

    bar_colors = ["#4A90B8" if v >= 0 else "#DC4C4C" for v in x_values]

    # Build error_x arrays: upper = p90 - mean, lower = mean - p10
    error_array = [p90_values[i] - x_values[i] for i in range(len(x_values))]
    error_arrayminus = [x_values[i] - p10_values[i] for i in range(len(x_values))]

    # Only emit error_x when at least one non-NaN, non-zero pair exists
    has_valid_errors = any(
        not (math.isnan(error_array[i]) or math.isnan(error_arrayminus[i]))
        for i in range(len(error_array))
    )
    error_x: dict | None = None
    if has_valid_errors:
        error_x = dict(
            type="data",
            array=error_array,
            arrayminus=error_arrayminus,
            visible=True,
        )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_values,
            y=ynames,
            orientation="h",
            marker_color=bar_colors,
            name="Net Balance",
            text=[f"{v:+.1f} MW" for v in x_values],
            textposition="auto",
            error_x=error_x,
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Net Flow (MW, positive = import)",
        yaxis_title="Bus",
        height=max(300, len(sorted_bus_ids) * 50 + 100),
        margin=dict(l=120, r=30, t=60, b=50),
        showlegend=False,
    )
    return make_chart_card(
        fig,
        title="Bus Balance — Mean Net Import / Export",
        chart_id="nw-bus-balance",
        height=max(300, len(sorted_bus_ids) * 50 + 100),
    )


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True — network tab is always shown when line_meta is populated."""
    return True


def render(data: DashboardData) -> str:
    """Return full HTML for the Network tab.

    Sections:
      A. Line Explorer (interactive dropdown + JS charts)
      B. Capacity Utilisation Heatmap (server-side Plotly)
      C. Bus Balance (server-side Plotly bar chart)

    Returns ``"<p>No network data available.</p>"`` when ``line_meta`` is
    empty.

    Args:
        data: Fully populated ``DashboardData`` instance.

    Returns:
        HTML string for the tab body.
    """
    if not data.line_meta:
        return "<p>No network data available.</p>"

    return (
        section_title("Line Explorer")
        + build_line_explorer(
            data.exchanges_lf,
            data.names,
            data.stage_labels,
            data.bh_df,
            data.line_bounds,
            data.line_meta,
        )
        + section_title("Capacity Utilisation")
        + chart_grid(
            [
                build_heatmap(
                    data.exchanges_lf,
                    data.line_bounds,
                    data.line_meta,
                    data.names,
                    data.stage_labels,
                    data.bh_df,
                )
            ],
            single=True,
        )
        + section_title("Bus Balance")
        + chart_grid(
            [
                build_bus_balance(
                    data.exchanges_lf,
                    data.line_meta,
                    data.bus_names,
                    data.bh_df,
                )
            ],
            single=True,
        )
    )
