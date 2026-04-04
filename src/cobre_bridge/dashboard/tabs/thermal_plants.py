"""Thermal Plant Details tab module for the Cobre dashboard.

Displays an interactive per-plant explorer with p10/p50/p90 generation,
cost, and energy charts driven by a JavaScript master-detail split-pane layout.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from cobre_bridge.ui.html import (
    _sparkline_svg,
    chart_grid,
    collapsible_section,
    plant_explorer_table,
    section_title,
    wrap_chart,
)
from cobre_bridge.ui.js import PLANT_EXPLORER_JS
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
# Interactive plant details
# ---------------------------------------------------------------------------


def build_interactive_thermal_details(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    lp_bounds: pd.DataFrame | None = None,
    bh_df: pl.DataFrame | None = None,
) -> str:
    """Build HTML with embedded per-thermal p10/p50/p90 data, LP bounds, and JS master-detail layout."""
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

    if not options:
        return "<p>No thermal plant data available.</p>"

    # Build table rows (sorted alphabetically by name, each row has data-index for
    # selectRow() lookup in TD, and data-name for filterTable() search).
    table_rows: list[str] = []
    for tid, d in options:
        name: str = d.get("name", "")
        bus: str = d.get("bus", "")
        max_mw: float = d.get("max_mw", 0)
        cost_per_mwh: float = d.get("cost_per_mwh", 0)
        gen_p50: list[float] = d.get("gen_p50", [])
        gen_spark = (
            _sparkline_svg(gen_p50, "#F5A623")
            if len(gen_p50) >= 2 and any(v != 0 for v in gen_p50)
            else ""
        )
        table_rows.append(
            f'<tr data-name="{name.lower()}" data-index="{tid}">'
            f'<td><input type="checkbox" class="compare-checkbox" data-id="{tid}"></td>'
            f"<td>{name}</td>"
            f"<td>{bus}</td>"
            f'<td data-sort-value="{max_mw:.0f}">{max_mw:.0f}</td>'
            f'<td data-sort-value="{cost_per_mwh:.2f}">{cost_per_mwh:.2f} R$/MWh</td>'
            f"<td>{gen_spark}</td>"
            f"</tr>"
        )

    rows_html = "".join(table_rows)
    columns: list[tuple[str, str]] = [
        ("Cmp", "none"),
        ("Name", "string"),
        ("Bus", "string"),
        ("MW", "number"),
        ("Cost", "number"),
        ("Gen", "none"),
    ]
    table_pane = (
        '<div class="explorer-table-pane">'
        + plant_explorer_table(
            "td-tbody",
            "td-search",
            columns,
            rows_html,
        )
        + "</div>"
    )

    # Detail pane: 2 collapsible sections with chart divs
    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" style="width:100%;height:350px;"></div>'

    generation_section = collapsible_section(
        "Generation",
        chart_grid([wrap_chart(_chart_div("td-gen"))], single=True),
    )
    economics_section = collapsible_section(
        "Economics",
        chart_grid(
            [
                wrap_chart(_chart_div("td-cost")),
                wrap_chart(_chart_div("td-energy")),
            ]
        ),
    )

    detail_pane = (
        '<div class="explorer-detail-pane">'
        + generation_section
        + economics_section
        + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    data_json = json.dumps(thermal_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    inline_js = (
        PLANT_EXPLORER_JS
        + "\nconst TD = "
        + data_json
        + ";\n"
        + "const TD_LABELS = "
        + labels_json
        + ";\n"
        + """
var _TC = {responsive: true};

function renderThermalDetail(containerId, d) {
  if (!d) { return; }
  var lbl = TD_LABELS;

  var genTraces = [
    plotlyBand(lbl, d.gen_p10, d.gen_p90, 'rgba(245,166,35,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.gen_p50, '#F5A623', 'P50'),
    plotlyLine(lbl, d.gen_p10, '#F5A623', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.gen_p90, '#F5A623', 'P90', 1, 'dot'),
  ];
  if (d.gen_max && d.gen_max.some(function(v) { return v > 0; })) {
    genTraces.push(plotlyRef(lbl, d.gen_max, '#DC4C4C', 'Gen Max (LP)'));
  }
  if (d.gen_min && d.gen_min.some(function(v) { return v > 0; })) {
    genTraces.push(plotlyRef(lbl, d.gen_min, '#4A8B6F', 'Gen Min (LP)'));
  }
  Plotly.react('td-gen', genTraces,
    plotlyLayout({title: d.name + ' \u2014 Generation (MW)', yaxis: {title: 'MW'}}), _TC);

  Plotly.react('td-cost', [
    plotlyBand(lbl, d.cost_p10, d.cost_p90, 'rgba(220,76,76,0.12)', 'P10\u2013P90'),
    plotlyLine(lbl, d.cost_p50, '#DC4C4C', 'P50'),
    plotlyLine(lbl, d.cost_p10, '#DC4C4C', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.cost_p90, '#DC4C4C', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Generation Cost (R$)', yaxis: {title: 'R$'}}), _TC);

  Plotly.react('td-energy', [
    plotlyBand(lbl, d.energy_p10, d.energy_p90, 'rgba(74,139,111,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.energy_p50, '#4A8B6F', 'P50'),
    plotlyLine(lbl, d.energy_p10, '#4A8B6F', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.energy_p90, '#4A8B6F', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Generation Energy (MWh)', yaxis: {title: 'MWh'}}), _TC);
}

var _TD_PALETTE = ['#2196F3', '#FF9800', '#4CAF50'];

function renderThermalComparison(entries, labels) {
  if (!entries || entries.length === 0) { return; }
  var lbl = labels || TD_LABELS;

  function _buildTraces(p50key) {
    return entries.map(function(d, i) {
      var c = _TD_PALETTE[i % _TD_PALETTE.length];
      return plotlyLine(lbl, d[p50key] || [], c, d.name || ('Plant ' + i));
    });
  }

  Plotly.react('td-gen', _buildTraces('gen_p50'),
    plotlyLayout({title: 'Generation (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _TC);
  Plotly.react('td-cost', _buildTraces('cost_p50'),
    plotlyLayout({title: 'Generation Cost (R$) \u2014 Comparison', yaxis: {title: 'R$'}}), _TC);
  Plotly.react('td-energy', _buildTraces('energy_p50'),
    plotlyLayout({title: 'Generation Energy (MWh) \u2014 Comparison', yaxis: {title: 'MWh'}}), _TC);
}

initPlantExplorer({
  tableId: 'td-tbody',
  searchInputId: 'td-search',
  detailContainerId: 'td-detail',
  dataVar: 'TD',
  renderDetail: renderThermalDetail
});

initComparisonMode({
  tableId: 'td-tbody',
  dataVar: 'TD',
  labelsVar: 'TD_LABELS',
  chartIds: ['td-gen','td-cost','td-energy'],
  renderComparison: renderThermalComparison,
  renderDetail: renderThermalDetail,
  maxCompare: 3
});

syncHover(['td-gen','td-cost','td-energy']);
"""
    )

    return explorer_html + "<script>\n" + inline_js + "\n</script>"


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
