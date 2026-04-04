"""Thermal Plant Details tab module for the Cobre dashboard.

Displays an interactive per-plant explorer with p10/p50/p90 generation,
cost, and energy charts driven by an embedded JavaScript dropdown.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from cobre_bridge.ui.html import section_title
from cobre_bridge.ui.plotly_helpers import stage_x_labels

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-thermal-plants"
TAB_LABEL = "Thermal Plant Details"
TAB_ORDER = 60

# ---------------------------------------------------------------------------
# Interactive plant details (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


def build_interactive_thermal_details(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    lp_bounds: pd.DataFrame | None = None,
    bh_df: pl.DataFrame | None = None,
) -> str:
    """Build HTML with embedded per-thermal p10/p50/p90 data, LP bounds, and JS dropdown."""
    metrics = ["generation_mw", "generation_cost", "generation_mwh"]
    short = {
        "generation_mw": "gen",
        "generation_cost": "cost",
        "generation_mwh": "energy",
    }

    schema = thermals_lf.collect_schema()
    available_metrics = [m for m in metrics if m in schema]

    # All thermal metrics are flow variables (generation_mw, generation_cost,
    # generation_mwh) — use block-hours weighted average.
    if bh_df is not None:
        all_pcts = (
            thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
            .group_by(["scenario_id", "stage_id", "thermal_id"])
            .agg(
                [
                    (pl.col(m) * pl.col("_bh")).sum() / pl.col("_bh").sum()
                    for m in available_metrics
                ]
            )
            .group_by(["stage_id", "thermal_id"])
            .agg(
                [
                    expr
                    for m in available_metrics
                    for expr in [
                        pl.col(m)
                        .quantile(0.1, interpolation="linear")
                        .alias(f"{m}_p10"),
                        pl.col(m)
                        .quantile(0.5, interpolation="linear")
                        .alias(f"{m}_p50"),
                        pl.col(m)
                        .quantile(0.9, interpolation="linear")
                        .alias(f"{m}_p90"),
                    ]
                ]
            )
            .sort(["thermal_id", "stage_id"])
            .collect(engine="streaming")
        )
    else:
        all_pcts = (
            thermals_lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "stage_id", "thermal_id"])
            .agg([pl.col(m).mean() for m in available_metrics])
            .group_by(["stage_id", "thermal_id"])
            .agg(
                [
                    expr
                    for m in available_metrics
                    for expr in [
                        pl.col(m)
                        .quantile(0.1, interpolation="linear")
                        .alias(f"{m}_p10"),
                        pl.col(m)
                        .quantile(0.5, interpolation="linear")
                        .alias(f"{m}_p50"),
                        pl.col(m)
                        .quantile(0.9, interpolation="linear")
                        .alias(f"{m}_p90"),
                    ]
                ]
            )
            .sort(["thermal_id", "stage_id"])
            .collect(engine="streaming")
        )

    stages = sorted(all_pcts["stage_id"].unique().to_list())
    xlabels = stage_x_labels(stages, stage_labels)

    thermal_data: dict[str, dict] = {}
    for tid, meta in sorted(thermal_meta.items()):
        sub = all_pcts.filter(pl.col("thermal_id") == tid)
        if sub.is_empty():
            continue
        entry: dict = {
            "name": meta["name"],
            "bus": bus_names.get(meta["bus_id"], str(meta["bus_id"])),
            "max_mw": round(meta["max_mw"], 1),
            "cost_per_mwh": round(meta["cost_per_mwh"], 2),
        }
        sub_map: dict[int, dict] = {r["stage_id"]: r for r in sub.iter_rows(named=True)}
        for m in metrics:
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
        thermal_data[str(tid)] = entry

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
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Thermal Plant Details tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Thermal Plant Details tab content area."""
    return section_title("Thermal Plant Explorer") + build_interactive_thermal_details(
        data.thermals_lf,
        data.thermal_meta,
        data.bus_names,
        data.stage_labels,
        data.lp_bounds,
        data.bh_df,
    )
