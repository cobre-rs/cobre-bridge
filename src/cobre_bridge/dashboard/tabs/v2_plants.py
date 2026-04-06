"""v2 Plant Explorer tab — hydro and thermal sub-tabs.

Provides a master-detail split-pane layout for both hydro and thermal plants.
Left pane: searchable/sortable table.  Right pane: detail charts rendered
client-side via Plotly.react() using precomputed per-plant p10/p50/p90
percentile bands.  Sub-tab switching is handled by SUB_TAB_JS.
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
from cobre_bridge.ui.js import PLANT_EXPLORER_JS, SUB_TAB_JS
from cobre_bridge.ui.plotly_helpers import stage_x_labels

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-plants"
TAB_LABEL = "Plant Explorer"
TAB_ORDER = 50

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

# Flow variables: block-hours weighted average
_FLOW_VARS: list[str] = [
    "generation_mw",
    "turbined_m3s",
    "spillage_m3s",
    "evaporation_m3s",
]

# Stage-level variables: filter block_id == 0
_STAGE_VARS: list[str] = [
    "storage_final_hm3",
    "inflow_m3s",
    "water_value_per_hm3",
]


def _compute_hydro_percentiles(
    hydros_lf: pl.LazyFrame,
    bh_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-plant p10/p50/p90 percentiles for all 7 variables.

    Flow variables (generation_mw, turbined_m3s, spillage_m3s, evaporation_m3s)
    are aggregated via block-hours weighted average before computing percentiles
    across scenarios.

    Stage-level variables (storage_final_hm3, inflow_m3s, water_value_per_hm3)
    are filtered to block_id == 0 before computing percentiles.

    Returns a single DataFrame with columns:
        hydro_id, stage_id, <var>_p10, <var>_p50, <var>_p90 for each var.
    """
    schema = hydros_lf.collect_schema()

    # --- Flow variables -------------------------------------------------------
    available_flow = [v for v in _FLOW_VARS if v in schema]
    flow_pcts: pl.DataFrame | None = None
    if available_flow:
        flow_pcts = (
            hydros_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
            .group_by(["scenario_id", "stage_id", "hydro_id"])
            .agg(
                [
                    (pl.col(m) * pl.col("_bh")).sum() / pl.col("_bh").sum()
                    for m in available_flow
                ]
            )
            .group_by(["stage_id", "hydro_id"])
            .agg(
                [
                    expr
                    for m in available_flow
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
            .sort(["hydro_id", "stage_id"])
            .collect(engine="streaming")
        )

    # --- Stage-level variables ------------------------------------------------
    available_stage = [v for v in _STAGE_VARS if v in schema]
    stage_pcts: pl.DataFrame | None = None
    if available_stage:
        stage_pcts = (
            hydros_lf.filter(pl.col("block_id") == 0)
            .group_by(["scenario_id", "stage_id", "hydro_id"])
            .agg([pl.col(m).mean() for m in available_stage])
            .group_by(["stage_id", "hydro_id"])
            .agg(
                [
                    expr
                    for m in available_stage
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
            .sort(["hydro_id", "stage_id"])
            .collect(engine="streaming")
        )

    # --- Merge ---------------------------------------------------------------
    if flow_pcts is not None and stage_pcts is not None:
        return flow_pcts.join(
            stage_pcts, on=["stage_id", "hydro_id"], how="full", coalesce=True
        )
    if flow_pcts is not None:
        return flow_pcts
    if stage_pcts is not None:
        return stage_pcts
    # Neither set of variables is available — return empty
    return pl.DataFrame(
        {
            "hydro_id": pl.Series([], dtype=pl.Int32),
            "stage_id": pl.Series([], dtype=pl.Int32),
        }
    )


def _build_hydro_json(
    hydro_meta: dict[int, dict],
    percentiles: pl.DataFrame,
    stages: list[int],
    stage_labels: dict[int, str],
    bus_names: dict[int, str],
    hydro_bounds: pd.DataFrame,
    lp_bounds: pd.DataFrame,
) -> tuple[dict[str, dict], list[str]]:
    """Build per-plant data dict for JS embedding.

    Returns ``(hydro_data_dict, xlabels_list)`` where ``hydro_data_dict``
    maps string plant-id keys to dicts containing percentile arrays and
    optional bounds arrays.
    """
    xlabels = stage_x_labels(stages, stage_labels)
    hydro_data: dict[str, dict] = {}

    # Pre-index percentiles by (hydro_id, stage_id)
    pct_index: dict[tuple[int, int], dict] = {}
    for row in percentiles.iter_rows(named=True):
        pct_index[(row["hydro_id"], row["stage_id"])] = row

    # Pre-process hydro_bounds once: {hydro_id: {stage_id: row}}
    hb_index: dict[int, dict[int, dict]] = {}
    if not hydro_bounds.empty:
        for _, row in hydro_bounds.iterrows():
            hid = int(row["hydro_id"])
            sid = int(row["stage_id"])
            hb_index.setdefault(hid, {})[sid] = row.to_dict()

    # Pre-process lp_bounds for hydro generation (entity_type_code == 0)
    lp_index: dict[
        int, dict[int, dict[int, float]]
    ] = {}  # hid -> {stage_id -> {bt_code -> value}}
    if not lp_bounds.empty:
        lp_hydro = lp_bounds[lp_bounds["entity_type_code"] == 0]
        for _, row in lp_hydro.iterrows():
            hid = int(row["entity_id"])
            sid = int(row["stage_id"])
            btc = int(row["bound_type_code"])
            lp_index.setdefault(hid, {}).setdefault(sid, {})[btc] = float(
                row["bound_value"]
            )

    all_vars = _FLOW_VARS + _STAGE_VARS

    for hid, meta in sorted(hydro_meta.items()):
        name: str = meta.get("name", str(hid))
        bus_id: int = meta.get("bus_id", -1)
        bus: str = bus_names.get(bus_id, str(bus_id))
        max_gen_mw: float = float(meta.get("max_gen_mw", 0) or 0)
        vol_max: float = float(meta.get("vol_max", 0) or 0)

        entry: dict = {
            "name": name,
            "bus": bus,
            "max_gen_mw": round(max_gen_mw, 1),
            "vol_max": round(vol_max, 1),
        }

        # Percentile arrays for each variable
        for var in all_vars:
            short = _var_short(var)
            for sfx in ["p10", "p50", "p90"]:
                col = f"{var}_{sfx}"
                entry[f"{short}_{sfx}"] = [
                    round(float(pct_index.get((hid, s), {}).get(col, 0) or 0), 4)
                    for s in stages
                ]

        # Derived outflow = turbined + spillage (sum of p50 arrays for display)
        turb_p50 = entry.get("turb_p50", [])
        spill_p50 = entry.get("spill_p50", [])
        entry["outflow_p50"] = [
            round(float(t + sp), 4) for t, sp in zip(turb_p50, spill_p50)
        ]
        turb_p10 = entry.get("turb_p10", [])
        spill_p10 = entry.get("spill_p10", [])
        entry["outflow_p10"] = [
            round(float(t + sp), 4) for t, sp in zip(turb_p10, spill_p10)
        ]
        turb_p90 = entry.get("turb_p90", [])
        spill_p90 = entry.get("spill_p90", [])
        entry["outflow_p90"] = [
            round(float(t + sp), 4) for t, sp in zip(turb_p90, spill_p90)
        ]

        # hydro_bounds: storage and turbined/spillage/outflow min/max
        if hid in hb_index:
            hb_stages = hb_index[hid]
            bound_cols = [
                ("min_storage_hm3", "stor_min"),
                ("max_storage_hm3", "stor_max"),
                ("min_turbined_m3s", "turb_min"),
                ("max_turbined_m3s", "turb_max"),
                ("min_spillage_m3s", "spill_min"),
                ("max_spillage_m3s", "spill_max"),
                ("min_outflow_m3s", "outflow_min"),
                ("max_outflow_m3s", "outflow_max"),
            ]
            for col, key in bound_cols:
                entry[key] = [
                    round(float(hb_stages.get(s, {}).get(col, 0) or 0), 4)
                    for s in stages
                ]
        else:
            for key in [
                "stor_min",
                "stor_max",
                "turb_min",
                "turb_max",
                "spill_min",
                "spill_max",
                "outflow_min",
                "outflow_max",
            ]:
                entry[key] = []

        # lp_bounds: generation min/max (bound_type_code 6=min, 7=max)
        if hid in lp_index:
            lp_stages = lp_index[hid]
            entry["gen_min"] = [
                round(lp_stages.get(s, {}).get(6, 0.0), 4) for s in stages
            ]
            entry["gen_max"] = [
                round(lp_stages.get(s, {}).get(7, 0.0), 4) for s in stages
            ]
        else:
            entry["gen_min"] = []
            entry["gen_max"] = []

        hydro_data[str(hid)] = entry

    return hydro_data, xlabels


def _var_short(var: str) -> str:
    """Map a column name to its short JSON key prefix."""
    _MAP: dict[str, str] = {
        "generation_mw": "gen",
        "turbined_m3s": "turb",
        "spillage_m3s": "spill",
        "evaporation_m3s": "evap",
        "storage_final_hm3": "stor",
        "inflow_m3s": "inflow",
        "water_value_per_hm3": "wv",
    }
    return _MAP.get(var, var)


# ---------------------------------------------------------------------------
# Thermal helpers
# ---------------------------------------------------------------------------

_THERMAL_FLOW_VARS: list[str] = [
    "generation_mw",
    "generation_cost",
    "generation_mwh",
]

_THERMAL_VAR_SHORT: dict[str, str] = {
    "generation_mw": "gen",
    "generation_cost": "cost",
    "generation_mwh": "energy",
}


def _compute_thermal_percentiles(
    thermals_lf: pl.LazyFrame,
    bh_df: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-plant p10/p50/p90 percentiles for all 3 thermal variables.

    All thermal metrics are flow variables (generation_mw, generation_cost,
    generation_mwh) — aggregate via block-hours weighted average before
    computing percentiles across scenarios.

    Returns a DataFrame with columns:
        thermal_id, stage_id, <var>_p10, <var>_p50, <var>_p90 for each var.
    """
    schema = thermals_lf.collect_schema()
    available = [m for m in _THERMAL_FLOW_VARS if m in schema]

    if not available:
        return pl.DataFrame(
            {
                "thermal_id": pl.Series([], dtype=pl.Int32),
                "stage_id": pl.Series([], dtype=pl.Int32),
            }
        )

    return (
        thermals_lf.join(bh_df.lazy(), on=["stage_id", "block_id"])
        .group_by(["scenario_id", "stage_id", "thermal_id"])
        .agg(
            [(pl.col(m) * pl.col("_bh")).sum() / pl.col("_bh").sum() for m in available]
        )
        .group_by(["stage_id", "thermal_id"])
        .agg(
            [
                expr
                for m in available
                for expr in [
                    pl.col(m).quantile(0.1, interpolation="linear").alias(f"{m}_p10"),
                    pl.col(m).quantile(0.5, interpolation="linear").alias(f"{m}_p50"),
                    pl.col(m).quantile(0.9, interpolation="linear").alias(f"{m}_p90"),
                ]
            ]
        )
        .sort(["thermal_id", "stage_id"])
        .collect(engine="streaming")
    )


def _build_thermal_json(
    thermal_meta: dict[int, dict],
    percentiles: pl.DataFrame,
    stages: list[int],
    stage_labels: dict[int, str],
    bus_names: dict[int, str],
    lp_bounds: pd.DataFrame,
) -> tuple[dict[str, dict], list[str]]:
    """Build per-plant data dict for JS embedding (thermal).

    Returns ``(thermal_data_dict, xlabels_list)`` where ``thermal_data_dict``
    maps string plant-id keys to dicts containing percentile arrays and
    optional LP bounds arrays.
    """
    xlabels = stage_x_labels(stages, stage_labels)
    thermal_data: dict[str, dict] = {}

    # Pre-index percentiles by (thermal_id, stage_id)
    pct_index: dict[tuple[int, int], dict] = {}
    for row in percentiles.iter_rows(named=True):
        pct_index[(row["thermal_id"], row["stage_id"])] = row

    # Pre-process lp_bounds for thermals (entity_type_code == 1)
    lp_index: dict[int, dict[int, dict[int, float]]] = {}
    if not lp_bounds.empty:
        tb = lp_bounds[lp_bounds["entity_type_code"] == 1]
        for _, row in tb.iterrows():
            tid = int(row["entity_id"])
            sid = int(row["stage_id"])
            btc = int(row["bound_type_code"])
            lp_index.setdefault(tid, {}).setdefault(sid, {})[btc] = float(
                row["bound_value"]
            )

    for tid, meta in sorted(thermal_meta.items()):
        name: str = meta.get("name", str(tid))
        bus_id: int = meta.get("bus_id", -1)
        bus: str = bus_names.get(bus_id, str(bus_id))
        max_mw: float = float(meta.get("max_mw", 0) or 0)
        cost_per_mwh: float = float(meta.get("cost_per_mwh", 0) or 0)

        entry: dict = {
            "name": name,
            "bus": bus,
            "max_mw": round(max_mw, 1),
            "cost_per_mwh": round(cost_per_mwh, 2),
        }

        # Percentile arrays for each variable
        for var in _THERMAL_FLOW_VARS:
            k = _THERMAL_VAR_SHORT[var]
            for sfx in ["p10", "p50", "p90"]:
                col = f"{var}_{sfx}"
                entry[f"{k}_{sfx}"] = [
                    round(float(pct_index.get((tid, s), {}).get(col, 0) or 0), 4)
                    for s in stages
                ]

        # lp_bounds: generation min/max (bound_type_code 6=min, 7=max)
        if tid in lp_index:
            lp_stages = lp_index[tid]
            entry["gen_min"] = [
                round(lp_stages.get(s, {}).get(6, 0.0), 4) for s in stages
            ]
            entry["gen_max"] = [
                round(lp_stages.get(s, {}).get(7, 0.0), 4) for s in stages
            ]
        else:
            entry["gen_min"] = []
            entry["gen_max"] = []

        thermal_data[str(tid)] = entry

    return thermal_data, xlabels


def build_thermal_explorer(
    thermals_lf: pl.LazyFrame,
    thermal_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    lp_bounds: pd.DataFrame,
    bh_df: pl.DataFrame,
) -> str:
    """Build the HTML for the thermal plant explorer sub-tab (data + JS only).

    Returns an HTML string with the explorer layout and embedded data/JS.
    Returns a no-data paragraph if ``thermal_meta`` is empty.

    Note: PLANT_EXPLORER_JS must NOT be emitted here; it is emitted once
    in ``render()`` to avoid duplicate JS function definitions.
    """
    if not thermal_meta:
        return "<p>No thermal plant data available.</p>"

    percentiles = _compute_thermal_percentiles(thermals_lf, bh_df)

    stages: list[int] = []
    if not percentiles.is_empty() and "stage_id" in percentiles.columns:
        stages = sorted(percentiles["stage_id"].unique().to_list())

    thermal_data, xlabels = _build_thermal_json(
        thermal_meta=thermal_meta,
        percentiles=percentiles,
        stages=stages,
        stage_labels=stage_labels,
        bus_names=bus_names,
        lp_bounds=lp_bounds if lp_bounds is not None else pd.DataFrame(),
    )

    options = sorted(thermal_data.items(), key=lambda x: x[1]["name"])
    if not options:
        return "<p>No thermal plant data available.</p>"

    # --- Table rows -----------------------------------------------------------
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
        ("Capacity (MW)", "number"),
        ("Cost (R$/MWh)", "number"),
        ("Gen", "none"),
    ]
    table_pane = (
        '<div class="explorer-table-pane" style="width:480px;min-width:320px;flex-shrink:0;">'
        + plant_explorer_table(
            "tt-tbody",
            "tt-search",
            columns,
            rows_html,
        )
        + "</div>"
    )

    # --- Detail pane ----------------------------------------------------------
    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" style="width:100%;height:350px;"></div>'

    generation_section = collapsible_section(
        "Generation",
        chart_grid([wrap_chart(_chart_div("tt-gen"))], single=True),
    )
    economics_section = collapsible_section(
        "Economics",
        chart_grid(
            [
                wrap_chart(_chart_div("tt-cost")),
                wrap_chart(_chart_div("tt-energy")),
            ]
        ),
    )

    detail_pane = (
        '<div id="tt-detail" class="explorer-detail-pane">'
        + generation_section
        + economics_section
        + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    # --- Embedded JS (data + render functions only — PLANT_EXPLORER_JS emitted
    #     once in render()) ---------------------------------------------------
    data_json = json.dumps(thermal_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    inline_js = (
        "\nvar TT = "
        + data_json
        + ";\n"
        + "var TT_LABELS = "
        + labels_json
        + ";\n"
        + """
var _TC = {responsive: true};

function renderThermalDetail(containerId, d) {
  if (!d) { return; }
  var lbl = TT_LABELS;

  var genTraces = [
    plotlyBand(lbl, d.gen_p10, d.gen_p90, 'rgba(245,166,35,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.gen_p50, '#F5A623', 'P50'),
    plotlyLine(lbl, d.gen_p10, '#F5A623', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.gen_p90, '#F5A623', 'P90', 1, 'dot'),
  ];
  if (d.gen_max && d.gen_max.length > 0 && d.gen_max.some(function(v) { return v > 0; })) {
    genTraces.push(plotlyRef(lbl, d.gen_max, '#DC4C4C', 'Gen Max (LP)'));
  }
  if (d.gen_min && d.gen_min.length > 0 && d.gen_min.some(function(v) { return v > 0; })) {
    genTraces.push(plotlyRef(lbl, d.gen_min, '#4A8B6F', 'Gen Min (LP)'));
  }
  Plotly.react('tt-gen', genTraces,
    plotlyLayout({title: d.name + ' \u2014 Generation (MW)', yaxis: {title: 'MW'}}), _TC);

  Plotly.react('tt-cost', [
    plotlyBand(lbl, d.cost_p10, d.cost_p90, 'rgba(220,76,76,0.12)', 'P10\u2013P90'),
    plotlyLine(lbl, d.cost_p50, '#DC4C4C', 'P50'),
    plotlyLine(lbl, d.cost_p10, '#DC4C4C', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.cost_p90, '#DC4C4C', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Dispatch Cost (R$)', yaxis: {title: 'R$'}}), _TC);

  Plotly.react('tt-energy', [
    plotlyBand(lbl, d.energy_p10, d.energy_p90, 'rgba(74,139,111,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.energy_p50, '#4A8B6F', 'P50'),
    plotlyLine(lbl, d.energy_p10, '#4A8B6F', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.energy_p90, '#4A8B6F', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Generation Energy (MWh)', yaxis: {title: 'MWh'}}), _TC);
}

var _TT_PALETTE = ['#2196F3', '#FF9800', '#4CAF50'];

function renderThermalComparison(entries, labels) {
  if (!entries || entries.length === 0) { return; }
  var lbl = labels || TT_LABELS;

  function _buildTraces(p50key) {
    return entries.map(function(d, i) {
      var c = _TT_PALETTE[i % _TT_PALETTE.length];
      return plotlyLine(lbl, d[p50key] || [], c, d.name || ('Plant ' + i));
    });
  }

  Plotly.react('tt-gen', _buildTraces('gen_p50'),
    plotlyLayout({title: 'Generation (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _TC);
  Plotly.react('tt-cost', _buildTraces('cost_p50'),
    plotlyLayout({title: 'Dispatch Cost (R$) \u2014 Comparison', yaxis: {title: 'R$'}}), _TC);
  Plotly.react('tt-energy', _buildTraces('energy_p50'),
    plotlyLayout({title: 'Generation Energy (MWh) \u2014 Comparison', yaxis: {title: 'MWh'}}), _TC);
}

initPlantExplorer({
  tableId: 'tt-tbody',
  searchInputId: 'tt-search',
  detailContainerId: 'tt-detail',
  dataVar: 'TT',
  labelsVar: 'TT_LABELS',
  renderDetail: renderThermalDetail
});

initComparisonMode({
  tableId: 'tt-tbody',
  dataVar: 'TT',
  labelsVar: 'TT_LABELS',
  chartIds: ['tt-gen','tt-cost','tt-energy'],
  renderComparison: renderThermalComparison,
  renderDetail: renderThermalDetail,
  maxCompare: 3
});

syncHover(['tt-gen','tt-cost','tt-energy']);
"""
    )

    return explorer_html + "<script>\n" + inline_js + "\n</script>"


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def build_hydro_explorer(
    hydros_lf: pl.LazyFrame,
    hydro_meta: dict[int, dict],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bh_df: pl.DataFrame,
    hydro_bounds: pd.DataFrame,
    lp_bounds: pd.DataFrame,
) -> str:
    """Build the HTML for the hydro plant explorer sub-tab.

    Returns the complete HTML string (explorer layout + embedded JS + data).
    Returns a no-data paragraph if ``hydro_meta`` is empty.
    """
    if not hydro_meta:
        return "<p>No hydro plant data available.</p>"

    percentiles = _compute_hydro_percentiles(hydros_lf, bh_df)

    stages: list[int] = []
    if not percentiles.is_empty() and "stage_id" in percentiles.columns:
        stages = sorted(percentiles["stage_id"].unique().to_list())

    hydro_data, xlabels = _build_hydro_json(
        hydro_meta=hydro_meta,
        percentiles=percentiles,
        stages=stages,
        stage_labels=stage_labels,
        bus_names=bus_names,
        hydro_bounds=hydro_bounds if hydro_bounds is not None else pd.DataFrame(),
        lp_bounds=lp_bounds if lp_bounds is not None else pd.DataFrame(),
    )

    options = sorted(hydro_data.items(), key=lambda x: x[1]["name"])
    if not options:
        return "<p>No hydro plant data available.</p>"

    # --- Table rows -----------------------------------------------------------
    table_rows: list[str] = []
    for hid, d in options:
        name: str = d.get("name", "")
        bus: str = d.get("bus", "")
        max_gen_mw: float = d.get("max_gen_mw", 0)
        vol_max: float = d.get("vol_max", 0)
        gen_p50: list[float] = d.get("gen_p50", [])
        gen_spark = (
            _sparkline_svg(gen_p50, "#2196F3")
            if len(gen_p50) >= 2 and any(v != 0 for v in gen_p50)
            else ""
        )
        table_rows.append(
            f'<tr data-name="{name.lower()}" data-index="{hid}">'
            f'<td><input type="checkbox" class="compare-checkbox" data-id="{hid}"></td>'
            f"<td>{name}</td>"
            f"<td>{bus}</td>"
            f'<td data-sort-value="{max_gen_mw:.0f}">{max_gen_mw:.0f}</td>'
            f'<td data-sort-value="{vol_max:.0f}">{vol_max:.0f}</td>'
            f"<td>{gen_spark}</td>"
            f"</tr>"
        )

    rows_html = "".join(table_rows)
    columns: list[tuple[str, str]] = [
        ("Cmp", "none"),
        ("Name", "string"),
        ("Bus", "string"),
        ("Max Gen (MW)", "number"),
        ("Vol Max (hm3)", "number"),
        ("Gen", "none"),
    ]
    table_pane = (
        '<div class="explorer-table-pane" style="width:480px;min-width:320px;flex-shrink:0;">'
        + plant_explorer_table(
            "hp-tbody",
            "hp-search",
            columns,
            rows_html,
        )
        + "</div>"
    )

    # --- Detail pane ----------------------------------------------------------
    def _chart_div(div_id: str) -> str:
        return f'<div id="{div_id}" style="width:100%;height:350px;"></div>'

    water_balance_section = collapsible_section(
        "Water Balance",
        chart_grid(
            [
                wrap_chart(_chart_div("hp-stor")),
                wrap_chart(_chart_div("hp-inflow")),
                wrap_chart(_chart_div("hp-spill")),
                wrap_chart(_chart_div("hp-evap")),
            ]
        ),
    )
    generation_section = collapsible_section(
        "Generation",
        chart_grid(
            [
                wrap_chart(_chart_div("hp-gen")),
                wrap_chart(_chart_div("hp-turb")),
                wrap_chart(_chart_div("hp-outflow")),
                wrap_chart(_chart_div("hp-wv")),
            ]
        ),
    )

    detail_pane = (
        '<div id="hp-detail" class="explorer-detail-pane">'
        + water_balance_section
        + generation_section
        + "</div>"
    )

    explorer_html = (
        '<div class="explorer-container">' + table_pane + detail_pane + "</div>"
    )

    # --- Embedded JS (data + render functions only — PLANT_EXPLORER_JS emitted
    #     once in render()) ---------------------------------------------------
    data_json = json.dumps(hydro_data, separators=(",", ":"))
    labels_json = json.dumps(xlabels)

    inline_js = (
        "\nvar HP = "
        + data_json
        + ";\n"
        + "var HP_LABELS = "
        + labels_json
        + ";\n"
        + """
var _HC = {responsive: true};

function renderHydroDetail(containerId, d) {
  if (!d) { return; }
  var lbl = HP_LABELS;

  // Storage (hm3) — stage-level
  var storTraces = [
    plotlyBand(lbl, d.stor_p10, d.stor_p90, 'rgba(33,150,243,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.stor_p50, '#2196F3', 'P50'),
    plotlyLine(lbl, d.stor_p10, '#2196F3', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.stor_p90, '#2196F3', 'P90', 1, 'dot'),
  ];
  if (d.stor_min && d.stor_min.length > 0 && d.stor_min.some(function(v){return v>0;})) {
    storTraces.push(plotlyRef(lbl, d.stor_min, '#4A8B6F', 'Min Storage'));
  }
  if (d.stor_max && d.stor_max.length > 0 && d.stor_max.some(function(v){return v>0;})) {
    storTraces.push(plotlyRef(lbl, d.stor_max, '#DC4C4C', 'Max Storage'));
  }
  Plotly.react('hp-stor', storTraces,
    plotlyLayout({title: d.name + ' \u2014 Storage (hm\u00b3)', yaxis: {title: 'hm\u00b3'}}), _HC);

  // Inflow (m3/s)
  Plotly.react('hp-inflow', [
    plotlyBand(lbl, d.inflow_p10, d.inflow_p90, 'rgba(76,175,80,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.inflow_p50, '#4CAF50', 'P50'),
    plotlyLine(lbl, d.inflow_p10, '#4CAF50', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.inflow_p90, '#4CAF50', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Inflow (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Spillage (m3/s)
  var spillTraces = [
    plotlyBand(lbl, d.spill_p10, d.spill_p90, 'rgba(255,152,0,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.spill_p50, '#FF9800', 'P50'),
    plotlyLine(lbl, d.spill_p10, '#FF9800', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.spill_p90, '#FF9800', 'P90', 1, 'dot'),
  ];
  if (d.spill_min && d.spill_min.length > 0 && d.spill_min.some(function(v){return v>0;})) {
    spillTraces.push(plotlyRef(lbl, d.spill_min, '#4A8B6F', 'Min Spillage'));
  }
  if (d.spill_max && d.spill_max.length > 0 && d.spill_max.some(function(v){return v>0;})) {
    spillTraces.push(plotlyRef(lbl, d.spill_max, '#DC4C4C', 'Max Spillage'));
  }
  Plotly.react('hp-spill', spillTraces,
    plotlyLayout({title: 'Spillage (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Evaporation (m3/s)
  Plotly.react('hp-evap', [
    plotlyBand(lbl, d.evap_p10, d.evap_p90, 'rgba(156,39,176,0.12)', 'P10\u2013P90'),
    plotlyLine(lbl, d.evap_p50, '#9C27B0', 'P50'),
    plotlyLine(lbl, d.evap_p10, '#9C27B0', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.evap_p90, '#9C27B0', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Evaporation (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Generation (MW) — flow variable
  var genTraces = [
    plotlyBand(lbl, d.gen_p10, d.gen_p90, 'rgba(245,166,35,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.gen_p50, '#F5A623', 'P50'),
    plotlyLine(lbl, d.gen_p10, '#F5A623', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.gen_p90, '#F5A623', 'P90', 1, 'dot'),
  ];
  if (d.gen_max && d.gen_max.length > 0 && d.gen_max.some(function(v){return v>0;})) {
    genTraces.push(plotlyRef(lbl, d.gen_max, '#DC4C4C', 'Gen Max (LP)'));
  }
  if (d.gen_min && d.gen_min.length > 0 && d.gen_min.some(function(v){return v>0;})) {
    genTraces.push(plotlyRef(lbl, d.gen_min, '#4A8B6F', 'Gen Min (LP)'));
  }
  Plotly.react('hp-gen', genTraces,
    plotlyLayout({title: d.name + ' \u2014 Generation (MW)', yaxis: {title: 'MW'}}), _HC);

  // Turbined (m3/s)
  var turbTraces = [
    plotlyBand(lbl, d.turb_p10, d.turb_p90, 'rgba(3,169,244,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.turb_p50, '#03A9F4', 'P50'),
    plotlyLine(lbl, d.turb_p10, '#03A9F4', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.turb_p90, '#03A9F4', 'P90', 1, 'dot'),
  ];
  if (d.turb_min && d.turb_min.length > 0 && d.turb_min.some(function(v){return v>0;})) {
    turbTraces.push(plotlyRef(lbl, d.turb_min, '#4A8B6F', 'Min Turbined'));
  }
  if (d.turb_max && d.turb_max.length > 0 && d.turb_max.some(function(v){return v>0;})) {
    turbTraces.push(plotlyRef(lbl, d.turb_max, '#DC4C4C', 'Max Turbined'));
  }
  Plotly.react('hp-turb', turbTraces,
    plotlyLayout({title: 'Turbined (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Outflow (m3/s) — derived turbined + spillage
  var outflowTraces = [
    plotlyBand(lbl, d.outflow_p10, d.outflow_p90, 'rgba(0,188,212,0.15)', 'P10\u2013P90'),
    plotlyLine(lbl, d.outflow_p50, '#00BCD4', 'P50'),
    plotlyLine(lbl, d.outflow_p10, '#00BCD4', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.outflow_p90, '#00BCD4', 'P90', 1, 'dot'),
  ];
  if (d.outflow_min && d.outflow_min.length > 0 && d.outflow_min.some(function(v){return v>0;})) {
    outflowTraces.push(plotlyRef(lbl, d.outflow_min, '#4A8B6F', 'Min Outflow'));
  }
  if (d.outflow_max && d.outflow_max.length > 0 && d.outflow_max.some(function(v){return v>0;})) {
    outflowTraces.push(plotlyRef(lbl, d.outflow_max, '#DC4C4C', 'Max Outflow'));
  }
  Plotly.react('hp-outflow', outflowTraces,
    plotlyLayout({title: 'Outflow (m\u00b3/s)', yaxis: {title: 'm\u00b3/s'}}), _HC);

  // Water Value (R$/hm3) — stage-level
  Plotly.react('hp-wv', [
    plotlyBand(lbl, d.wv_p10, d.wv_p90, 'rgba(233,30,99,0.12)', 'P10\u2013P90'),
    plotlyLine(lbl, d.wv_p50, '#E91E63', 'P50'),
    plotlyLine(lbl, d.wv_p10, '#E91E63', 'P10', 1, 'dot'),
    plotlyLine(lbl, d.wv_p90, '#E91E63', 'P90', 1, 'dot'),
  ], plotlyLayout({title: 'Water Value (R$/hm\u00b3)', yaxis: {title: 'R$/hm\u00b3'}}), _HC);
}

var _HP_PALETTE = ['#2196F3', '#FF9800', '#4CAF50'];

function renderHydroComparison(entries, labels) {
  if (!entries || entries.length === 0) { return; }
  var lbl = labels || HP_LABELS;

  function _buildTraces(p50key) {
    return entries.map(function(d, i) {
      var c = _HP_PALETTE[i % _HP_PALETTE.length];
      return plotlyLine(lbl, d[p50key] || [], c, d.name || ('Plant ' + i));
    });
  }

  Plotly.react('hp-stor', _buildTraces('stor_p50'),
    plotlyLayout({title: 'Storage (hm\u00b3) \u2014 Comparison', yaxis: {title: 'hm\u00b3'}}), _HC);
  Plotly.react('hp-inflow', _buildTraces('inflow_p50'),
    plotlyLayout({title: 'Inflow (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-spill', _buildTraces('spill_p50'),
    plotlyLayout({title: 'Spillage (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-evap', _buildTraces('evap_p50'),
    plotlyLayout({title: 'Evaporation (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-gen', _buildTraces('gen_p50'),
    plotlyLayout({title: 'Generation (MW) \u2014 Comparison', yaxis: {title: 'MW'}}), _HC);
  Plotly.react('hp-turb', _buildTraces('turb_p50'),
    plotlyLayout({title: 'Turbined (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-outflow', _buildTraces('outflow_p50'),
    plotlyLayout({title: 'Outflow (m\u00b3/s) \u2014 Comparison', yaxis: {title: 'm\u00b3/s'}}), _HC);
  Plotly.react('hp-wv', _buildTraces('wv_p50'),
    plotlyLayout({title: 'Water Value (R$/hm\u00b3) \u2014 Comparison', yaxis: {title: 'R$/hm\u00b3'}}), _HC);
}

initPlantExplorer({
  tableId: 'hp-tbody',
  searchInputId: 'hp-search',
  detailContainerId: 'hp-detail',
  dataVar: 'HP',
  labelsVar: 'HP_LABELS',
  renderDetail: renderHydroDetail
});

initComparisonMode({
  tableId: 'hp-tbody',
  dataVar: 'HP',
  labelsVar: 'HP_LABELS',
  chartIds: ['hp-stor','hp-inflow','hp-spill','hp-evap','hp-gen','hp-turb','hp-outflow','hp-wv'],
  renderComparison: renderHydroComparison,
  renderDetail: renderHydroDetail,
  maxCompare: 3
});

syncHover(['hp-stor','hp-inflow','hp-spill','hp-evap','hp-gen','hp-turb','hp-outflow','hp-wv']);
"""
    )

    return explorer_html + "<script>\n" + inline_js + "\n</script>"


# ---------------------------------------------------------------------------
# Tab interface
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:  # noqa: ARG001
    """Plant Explorer tab always renders."""
    return True


def render(data: DashboardData) -> str:
    """Return the full HTML string for the Plant Explorer tab content area.

    Wraps the hydro and thermal sub-tabs in a sub-tab bar with switching.
    Emits PLANT_EXPLORER_JS and SUB_TAB_JS once each before both panels.
    """
    hydro_content = build_hydro_explorer(
        hydros_lf=data.hydros_lf,
        hydro_meta=data.hydro_meta,
        bus_names=data.bus_names,
        stage_labels=data.stage_labels,
        bh_df=data.bh_df,
        hydro_bounds=data.hydro_bounds,
        lp_bounds=data.lp_bounds,
    )
    thermal_content = build_thermal_explorer(
        thermals_lf=data.thermals_lf,
        thermal_meta=data.thermal_meta,
        bus_names=data.bus_names,
        stage_labels=data.stage_labels,
        lp_bounds=data.lp_bounds,
        bh_df=data.bh_df,
    )

    sub_tab_bar = (
        '<div class="sub-tab-bar">'
        "<button class=\"sub-tab-btn active\" onclick=\"switchSubTab('plants-hydro', 'plants-explorer')\">Hydro</button>"
        "<button class=\"sub-tab-btn\" onclick=\"switchSubTab('plants-thermal', 'plants-explorer')\">Thermal</button>"
        "</div>"
    )

    shared_js = "<script>\n" + SUB_TAB_JS + PLANT_EXPLORER_JS + "\n</script>"

    hydro_panel = (
        '<div id="plants-hydro" class="sub-tab-panel" style="display:block;">'
        + hydro_content
        + "</div>"
    )
    thermal_panel = (
        '<div id="plants-thermal" class="sub-tab-panel" style="display:none;">'
        + thermal_content
        + "</div>"
    )

    group = (
        '<div data-subtab-group="plants-explorer">'
        + sub_tab_bar
        + hydro_panel
        + thermal_panel
        + "</div>"
    )

    return section_title("Plant Explorer") + shared_js + group
