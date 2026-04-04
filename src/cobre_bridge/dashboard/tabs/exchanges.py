"""Exchanges tab module for the Cobre dashboard.

Displays line explorer, capacity utilisation heatmap, and flow direction summary.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from cobre_bridge.dashboard.data import entity_name
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

TAB_ID = "tab-exchanges"
TAB_LABEL = "Exchanges"
TAB_ORDER = 70

# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def chart_net_flow_by_line(
    exchanges_lf: pl.LazyFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Net flow by line by stage."""
    flow_data = (
        exchanges_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "line_id"])
        .agg((pl.col("net_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum())
        .group_by(["stage_id", "line_id"])
        .agg(pl.col("net_flow_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    line_ids = sorted(flow_data["line_id"].unique().to_list())
    stages = sorted(flow_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

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
        sub = flow_data.filter(pl.col("line_id") == lid)
        flow_map = dict(zip(sub["stage_id"].to_list(), sub["net_flow_mw"].to_list()))
        fig.add_trace(
            go.Scatter(
                x=xlabels,
                y=[flow_map.get(s, 0) for s in stages],
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
    exchanges_lf: pl.LazyFrame,
    line_bounds: pd.DataFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Heatmap of capacity utilization: lines vs stages."""
    flow_data = (
        exchanges_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "line_id"])
        .agg(
            (
                (pl.col("direct_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
            ).alias("direct"),
            (
                (pl.col("reverse_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
            ).alias("reverse"),
        )
        .group_by(["stage_id", "line_id"])
        .agg(
            pl.col("direct").mean(),
            pl.col("reverse").mean(),
        )
        .sort("stage_id")
        .collect(engine="streaming")
    )
    line_ids = sorted(flow_data["line_id"].unique().to_list())
    stages = sorted(flow_data["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)
    ynames = [entity_name(names, "lines", lid) for lid in line_ids]

    z_direct = []
    z_reverse = []
    for lid in line_ids:
        sub = flow_data.filter(pl.col("line_id") == lid)
        d_map = dict(zip(sub["stage_id"].to_list(), sub["direct"].to_list()))
        r_map = dict(zip(sub["stage_id"].to_list(), sub["reverse"].to_list()))
        lb_line = line_bounds[line_bounds["line_id"] == lid].set_index("stage_id")
        row_d = []
        row_r = []
        for s in stages:
            d_flow = d_map.get(s, 0)
            r_flow = r_map.get(s, 0)
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
    exchanges_lf: pl.LazyFrame,
    names: dict[tuple[str, int], str],
    bh_df: pl.DataFrame,
) -> str:
    """Average direct and reverse flow per line (horizontal grouped bar)."""
    flow_data = (
        exchanges_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "line_id"])
        .agg(
            (
                (pl.col("direct_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
            ).alias("direct"),
            (
                (pl.col("reverse_flow_mw") * pl.col("_bh")).sum() / pl.col("_bh").sum()
            ).alias("reverse"),
        )
        .group_by(["scenario_id", "line_id"])
        .agg(
            pl.col("direct").mean(),
            pl.col("reverse").mean(),
        )
        .group_by("line_id")
        .agg(
            pl.col("direct").mean(),
            pl.col("reverse").mean(),
        )
        .sort("line_id")
        .collect(engine="streaming")
    )
    line_ids = flow_data["line_id"].to_list()
    ynames = [entity_name(names, "lines", lid) for lid in line_ids]
    avg_direct = flow_data["direct"].to_list()
    avg_reverse = flow_data["reverse"].to_list()

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
    exchanges_lf: pl.LazyFrame,
    names: dict[tuple[str, int], str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
) -> str:
    """Build HTML with embedded per-line p10/p50/p90 data and JS dropdown."""
    # Collect block-hours-weighted average per scenario/stage/line
    flow_cols = ["net_flow_mw", "direct_flow_mw", "reverse_flow_mw"]
    schema = exchanges_lf.collect_schema()
    avail_flow_cols = [c for c in flow_cols if c in schema]
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
    stages = sorted(ex0["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)
    line_ids = sorted(ex0["line_id"].unique().to_list())

    line_data: dict[str, dict] = {}
    for lid in line_ids:
        lname = entity_name(names, "lines", lid)
        ldf = ex0.filter(pl.col("line_id") == lid)
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
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True — exchanges tab is always shown."""
    return True


def render(data: DashboardData) -> str:
    """Return full HTML for the Exchanges tab."""
    return (
        section_title("Line Explorer")
        + build_interactive_exchange_detail(
            data.exchanges_lf, data.names, data.stage_labels, data.bh_df
        )
        + section_title("Capacity Utilization")
        + '<div class="chart-grid-single">'
        + (
            wrap_chart(
                chart_capacity_utilization_heatmap(
                    data.exchanges_lf,
                    data.line_bounds,
                    data.names,
                    data.stage_labels,
                    data.bh_df,
                )
            )
            if not data.line_bounds.empty
            else "<p>No line bounds data.</p>"
        )
        + "</div>"
        + section_title("Flow Direction Summary")
        + '<div class="chart-grid">'
        + wrap_chart(
            chart_flow_direction_summary(data.exchanges_lf, data.names, data.bh_df)
        )
        + "</div>"
    )
