"""Chart implementations for the HTML comparison report.

All functions return HTML strings (plotly chart divs) that can be
embedded in the comparison report template.  Uses plotly.js via
inline JSON config (no python plotly server needed at report viewing
time).
"""

from __future__ import annotations

import json
import uuid

import polars as pl

from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.results import (
    ResultComparison,
    ResultsSummary,
)

# Cobre band fill color: blue at 15% opacity.
_BAND_FILL = "rgba(74,144,184,0.15)"
_BAND_LINE = "rgba(255,255,255,0)"

_LEGEND = dict(
    orientation="h",
    yanchor="top",
    y=-0.15,
    xanchor="center",
    x=0.5,
    font=dict(size=11),
)
_MARGIN = dict(l=60, r=30, t=60, b=10)


def _plotly_div(
    traces: list[dict],
    layout: dict,
    height: int = 400,
) -> str:
    """Return a plotly div with inline data and layout."""
    div_id = f"chart-{uuid.uuid4().hex[:8]}"
    layout.setdefault("height", height)
    layout.setdefault("margin", _MARGIN)
    layout.setdefault("legend", _LEGEND)
    layout.setdefault("template", "plotly_white")

    data_json = json.dumps(traces)
    layout_json = json.dumps(layout)

    return (
        f'<div id="{div_id}"></div>\n'
        "<script>"
        f"Plotly.newPlot('{div_id}', {data_json}, {layout_json}, "
        "{responsive: true});"
        "</script>"
    )


# -------------------------------------------------------------------
# Overview tab charts
# -------------------------------------------------------------------

# Mapping from a unified display label to (NEWAVE categories, Cobre categories).
# Each side is a list because one display row may aggregate multiple raw categories.
_COST_MAP: list[tuple[str, list[str], list[str]]] = [
    ("Thermal Generation", ["GERACAO TERMICA"], ["thermal_cost"]),
    ("Deficit", ["DEFICIT"], ["deficit_cost"]),
    ("Excess Energy", ["EXCESSO ENERGIA"], ["excess_cost"]),
    ("Spillage", ["VERTIMENTO", "VERTIMENTO UHE"], ["spillage_cost"]),
    ("Exchange", ["INTERCAMBIO"], ["exchange_cost"]),
    (
        "Min Outflow Violation",
        ["VIOLACAO VZMIN"],
        ["outflow_violation_below_cost"],
    ),
    (
        "Max Outflow Violation",
        ["VIOL. DEFL. MAXIMA"],
        ["outflow_violation_above_cost"],
    ),
    (
        "Turbined Violation",
        [
            "VIOL. TURB. MINIMO",
            "VIOL. TURB. MAXIMO",
            "TURBINAMENTO UHE",
        ],
        ["turbined_violation_cost"],
    ),
    (
        "Generation Violation",
        ["VIOLACAO GHMIN", "VIOLACAO GHMINU"],
        ["generation_violation_cost"],
    ),
    (
        "Storage Violation",
        ["VIOLACAO CAR", "VIOLACAO SAR", "VIOLACAO EVMIN"],
        ["storage_violation_cost", "filling_target_cost"],
    ),
    (
        "Water Withdrawal Violation",
        ["VIOLACAO RETIRADA"],
        ["withdrawal_violation_cost"],
    ),
    (
        "Evaporation Violation",
        ["VIOL. EVAP. UHE"],
        ["evaporation_violation_cost"],
    ),
    ("FPHA Violation", ["VIOLACAO FPHA"], ["fpha_turbined_cost"]),
    (
        "Curtailment",
        ["CORTE GER. EOLICA", "VERT. FIO N. TURB."],
        ["curtailment_cost"],
    ),
    (
        "Electric Constraint Violation",
        ["VIOL. RESTELETRICA", "VIOL. INTERC. MIN."],
        ["generic_violation_cost"],
    ),
    (
        "Inflow Penalty",
        [],
        ["inflow_penalty_cost"],
    ),
]


# Colors for each cost category — chosen to be visually distinct and
# semantically representative (warm = generation/operational, red = deficit,
# cool blues/grays = violations, green = exchange/hydro-related).
_COST_COLORS: dict[str, str] = {
    "Thermal Generation": "#E8913A",  # warm orange — thermal fuel
    "Deficit": "#C0392B",  # strong red — deficit is critical
    "Excess Energy": "#8E44AD",  # purple — surplus
    "Spillage": "#5DADE2",  # light blue — water spilled
    "Exchange": "#2ECC71",  # green — transmission exchange
    "Min Outflow Violation": "#1A5276",  # dark teal
    "Max Outflow Violation": "#1F618D",  # medium teal
    "Turbined Violation": "#6C7A89",  # steel gray
    "Generation Violation": "#7D6B57",  # brown-gray
    "Storage Violation": "#D4AC0D",  # gold — reservoir storage
    "Water Withdrawal Violation": "#A04000",  # burnt sienna
    "Evaporation Violation": "#76D7C4",  # mint — evaporation
    "FPHA Violation": "#AF7AC5",  # lavender
    "Curtailment": "#45B39D",  # teal-green — renewables
    "Electric Constraint Violation": "#5D6D7E",  # slate
    "Inflow Penalty": "#85929E",  # cool gray
}
_COST_COLOR_DEFAULT = "#95A5A6"  # fallback gray


def cost_breakdown_chart(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
) -> str:
    """Stacked vertical bar chart: one bar for NEWAVE, one for Cobre."""
    if not nw_costs and not cobre_costs:
        return "<p>No cost data available.</p>"

    # Build unified category list, summing raw keys per side.
    categories: list[tuple[str, float, float]] = []

    for display_label, nw_keys, cb_keys in _COST_MAP:
        nw_sum = sum(nw_costs.get(k, 0.0) for k in nw_keys)
        cb_sum = sum(cobre_costs.get(k, 0.0) for k in cb_keys)
        if abs(nw_sum) < 0.01 and abs(cb_sum) < 0.01:
            continue
        categories.append((display_label, nw_sum, cb_sum))

    # Unmapped NEWAVE categories.
    mapped_nw = {k for _, nw_keys, _ in _COST_MAP for k in nw_keys}
    for k, v in sorted(nw_costs.items()):
        if k not in mapped_nw and abs(v) > 0.01:
            categories.append((k.title(), v, 0.0))

    # Unmapped Cobre categories.
    mapped_cb = {k for _, _, cb_keys in _COST_MAP for k in cb_keys}
    for k, v in sorted(cobre_costs.items()):
        if k not in mapped_cb and abs(v) > 0.01:
            categories.append((k.replace("_", " ").title(), 0.0, v))

    if not categories:
        return "<p>No cost data available.</p>"

    # Sort so largest total cost is at the bottom of the stack (drawn first).
    categories.sort(key=lambda t: -(t[1] + t[2]))

    x_labels = ["NEWAVE", "Cobre"]
    traces: list[dict] = []

    for label, nw_v, cb_v in categories:
        color = _COST_COLORS.get(label, _COST_COLOR_DEFAULT)
        traces.append(
            {
                "x": x_labels,
                "y": [round(nw_v / 1e9, 3), round(cb_v / 1e9, 3)],
                "name": label,
                "type": "bar",
                "marker": {"color": color},
                "hovertemplate": (
                    f"%{{x}}<br>{label}: %{{y:.2f}} 10⁹ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": "Cost Breakdown (Present Value)",
        "yaxis": {"title": "Cost (10⁹ R$)"},
        "barmode": "stack",
        "bargap": 0.4,
    }

    return _plotly_div(traces, layout, height=550)


def convergence_chart(
    nw_conv: pl.DataFrame,
    cobre_conv: pl.DataFrame,
) -> str:
    """Convergence overlay: NEWAVE vs Cobre lower/upper bounds.

    Accepts raw convergence DataFrames directly so it can show NEWAVE
    data even when Cobre convergence is empty.
    """
    lb_nw: dict[int, float] = {}
    ub_nw: dict[int, float] = {}
    lb_cb: dict[int, float] = {}
    ub_cb: dict[int, float] = {}

    if not nw_conv.is_empty():
        for row in nw_conv.iter_rows(named=True):
            it = int(row["iteration"])
            lb_nw[it] = float(row["lower_bound"])
            ub_nw[it] = float(row["upper_bound_mean"])

    if not cobre_conv.is_empty():
        for row in cobre_conv.iter_rows(named=True):
            it = int(row["iteration"])
            lb_cb[it] = float(row["lower_bound"])
            ub_cb[it] = float(row["upper_bound_mean"])

    iters = sorted(set(lb_nw) | set(lb_cb))
    if not iters:
        return "<p>No convergence data available.</p>"

    traces: list[dict] = []

    if lb_nw:
        nw_iters = sorted(lb_nw)
        traces.append(
            {
                "x": nw_iters,
                "y": [lb_nw[i] for i in nw_iters],
                "name": "NEWAVE ZINF",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE},
            }
        )
        traces.append(
            {
                "x": nw_iters,
                "y": [ub_nw.get(i) for i in nw_iters],
                "name": "NEWAVE ZSUP",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "dash": "dash"},
            }
        )

    if lb_cb:
        cb_iters = sorted(lb_cb)
        traces.append(
            {
                "x": cb_iters,
                "y": [lb_cb[i] for i in cb_iters],
                "name": "Cobre Lower",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE},
            }
        )
        traces.append(
            {
                "x": cb_iters,
                "y": [ub_cb.get(i) for i in cb_iters],
                "name": "Cobre Upper",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "dash": "dash"},
            }
        )

    layout = {
        "title": "Convergence: NEWAVE vs Cobre",
        "xaxis": {"title": "Iteration"},
        "yaxis": {"title": "Cost (R$)", "type": "log"},
    }

    return _plotly_div(traces, layout)


# -------------------------------------------------------------------
# System tab charts
# -------------------------------------------------------------------


def system_comparison_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
) -> str:
    """Line chart comparing a system variable by stage with p10-p90 band."""
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]
    if not bus_data:
        return f"<p>No {variable} data available.</p>"

    nw_by_stage: dict[int, float] = {}
    cb_by_stage: dict[int, float] = {}
    for r in bus_data:
        nw_by_stage[r.stage] = nw_by_stage.get(r.stage, 0.0) + r.newave_value
        cb_by_stage[r.stage] = cb_by_stage.get(r.stage, 0.0) + r.cobre_value

    matched_ids = {r.cobre_id for r in bus_data}
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)
    traces.extend(
        [
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            },
            {
                "x": stages,
                "y": [cb_by_stage.get(s, 0) for s in stages],
                "name": "Cobre Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
            },
        ]
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": variable},
    }

    return _plotly_div(traces, layout)


# -------------------------------------------------------------------
# Hydro tab charts
# -------------------------------------------------------------------


def _aggregate_percentile_traces(
    pct_df: pl.DataFrame | None,
    variable: str,
    stages: list[int],
    entity_ids: set[int] | None = None,
) -> list[dict]:
    """Build p10-p90 band + p10/p90 line traces from aggregate percentiles.

    Sums percentiles across matched entities per stage (aggregate view).
    When *entity_ids* is provided, only those entities are included —
    this keeps the band consistent with the mean line which only covers
    entities matched between NEWAVE and Cobre.
    """
    if pct_df is None or pct_df.is_empty():
        return []

    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    if p10_col not in pct_df.columns or p90_col not in pct_df.columns:
        return []

    filtered = pct_df
    if entity_ids is not None:
        filtered = pct_df.filter(pl.col("entity_id").is_in(list(entity_ids)))

    # Sum across entities per stage.
    agg = filtered.group_by("stage_id").agg(
        pl.col(p10_col).sum(), pl.col(p90_col).sum()
    )
    lookup = {int(r["stage_id"]): r for r in agg.iter_rows(named=True)}

    p10 = [float(lookup.get(s, {}).get(p10_col, 0)) for s in stages]
    p90 = [float(lookup.get(s, {}).get(p90_col, 0)) for s in stages]

    return [
        {
            "x": stages + stages[::-1],
            "y": p90 + p10[::-1],
            "fill": "toself",
            "fillcolor": _BAND_FILL,
            "line": {"color": _BAND_LINE},
            "name": "Cobre P10–P90",
            "type": "scatter",
            "legendgroup": "band",
            "showlegend": True,
        },
        {
            "x": stages,
            "y": p10,
            "name": "Cobre P10",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 1, "dash": "dot"},
            "legendgroup": "band",
            "showlegend": False,
        },
        {
            "x": stages,
            "y": p90,
            "name": "Cobre P90",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 1, "dash": "dot"},
            "legendgroup": "band",
            "showlegend": False,
        },
    ]


def hydro_aggregate_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
) -> str:
    """Aggregate hydro comparison by stage with optional p10-p90 band."""
    hydro_data = [
        r for r in results if r.entity_type == "hydro" and r.variable == variable
    ]
    if not hydro_data:
        return f"<p>No hydro {variable} data.</p>"

    nw_by_stage: dict[int, float] = {}
    cb_by_stage: dict[int, float] = {}
    for r in hydro_data:
        nw_by_stage[r.stage] = nw_by_stage.get(r.stage, 0.0) + r.newave_value
        cb_by_stage[r.stage] = cb_by_stage.get(r.stage, 0.0) + r.cobre_value

    matched_ids = {r.cobre_id for r in hydro_data}
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))

    # Band traces first (rendered behind the lines).
    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)

    traces.extend(
        [
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            },
            {
                "x": stages,
                "y": [cb_by_stage.get(s, 0) for s in stages],
                "name": "Cobre Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
            },
        ]
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": variable},
    }

    return _plotly_div(traces, layout)


# -------------------------------------------------------------------
# Thermal tab charts
# -------------------------------------------------------------------


def thermal_generation_chart(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
) -> str:
    """Aggregate thermal generation comparison by stage."""
    thermal_data = [r for r in results if r.entity_type == "thermal"]
    if not thermal_data:
        return "<p>No thermal generation data.</p>"

    nw_by_stage: dict[int, float] = {}
    cb_by_stage: dict[int, float] = {}
    for r in thermal_data:
        nw_by_stage[r.stage] = nw_by_stage.get(r.stage, 0.0) + r.newave_value
        cb_by_stage[r.stage] = cb_by_stage.get(r.stage, 0.0) + r.cobre_value

    matched_ids = {r.cobre_id for r in thermal_data}
    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = _aggregate_percentile_traces(pct_df, "generation_mw", stages, matched_ids)
    traces.extend(
        [
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            },
            {
                "x": stages,
                "y": [cb_by_stage.get(s, 0) for s in stages],
                "name": "Cobre Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
            },
        ]
    )

    layout = {
        "title": "Thermal Generation",
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": "MW"},
    }

    return _plotly_div(traces, layout)


# -------------------------------------------------------------------
# Productivity tab charts
# -------------------------------------------------------------------


def productivity_scatter(
    results: list[ResultComparison],
) -> str:
    """Scatter plot of NEWAVE vs Cobre productivity."""
    prod = [r for r in results if r.entity_type == "productivity"]
    if not prod:
        return "<p>No productivity data available.</p>"

    nw_vals = [r.newave_value for r in prod]
    cb_vals = [r.cobre_value for r in prod]
    names = [r.entity_name for r in prod]

    min_val = min(min(nw_vals), min(cb_vals))
    max_val = max(max(nw_vals), max(cb_vals))

    traces = [
        {
            "x": nw_vals,
            "y": cb_vals,
            "text": names,
            "name": "Plants",
            "type": "scatter",
            "mode": "markers",
            "marker": {"color": COLOR_COBRE, "size": 8},
        },
        {
            "x": [min_val, max_val],
            "y": [min_val, max_val],
            "name": "Perfect match",
            "type": "scatter",
            "mode": "lines",
            "line": {
                "color": "#8B9298",
                "dash": "dash",
            },
            "showlegend": False,
        },
    ]

    layout = {
        "title": "Productivity: NEWAVE vs Cobre",
        "xaxis": {"title": "NEWAVE productivity"},
        "yaxis": {"title": "Cobre productivity"},
    }

    return _plotly_div(traces, layout)


# -------------------------------------------------------------------
# Summary metric cards
# -------------------------------------------------------------------


def overview_metrics(summary: ResultsSummary) -> str:
    """Generate metric card HTML for the overview tab."""
    from cobre_bridge.comparators.html_report import (
        metric_card,
        metrics_grid,
    )

    cards = [
        metric_card(str(summary.total), "Total Comparisons"),
        metric_card(
            str(len(summary.by_entity_type)),
            "Entity Types",
        ),
        metric_card(
            str(len(summary.by_variable)),
            "Variables",
        ),
    ]

    # Add top correlation.
    if summary.by_variable:
        best_var = max(
            summary.by_variable,
            key=lambda v: summary.by_variable[v].correlation,
        )
        best_corr = summary.by_variable[best_var].correlation
        cards.append(metric_card(f"{best_corr:.4f}", f"Best r ({best_var})"))

    # Add worst max relative diff.
    if summary.by_variable:
        worst_var = max(
            summary.by_variable,
            key=lambda v: summary.by_variable[v].max_rel_diff,
        )
        worst_pct = summary.by_variable[worst_var].max_rel_diff * 100
        cards.append(
            metric_card(
                f"{worst_pct:.1f}%",
                f"Max Rel Diff ({worst_var})",
            )
        )

    return metrics_grid(cards)


# -------------------------------------------------------------------
# Per-bus system charts
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Energy Balance tab charts
# -------------------------------------------------------------------

# NEWAVE MEDIAS-MERC variable → (display label, unit).
_BALANCE_VARS: list[tuple[str, str, str, str]] = [
    # (display_label, newave_var, cobre_var, unit)
    ("Hydro Generation", "GHTOT", "hydro_gen_mw", "MW"),
    ("Thermal Generation", "GTERM", "thermal_gen_mw", "MW"),
    ("NCS Generation", "GEOL", "ncs_gen_mw", "MW"),
    ("Load", "", "load_mw", "MW"),
    ("Deficit", "DEFT", "deficit_mw", "MW"),
    ("Excess", "EXCESSO", "excess_mw", "MW"),
]


def build_energy_balance_tab(
    nw_market: pl.DataFrame,
    bus_agg: pl.DataFrame,
    bus_meta: dict[int, dict],
    nw_bus_names: dict[int, str],
) -> str:
    """Build per-bus energy balance charts with p10/p90 bands.

    One 2x2 faceted chart per variable, with NEWAVE mean + Cobre p10/p50/p90.
    """
    if bus_agg.is_empty() and nw_market.is_empty():
        return "<p>No energy balance data available.</p>"

    nw_offset = 0
    if not nw_market.is_empty():
        nw_offset = int(nw_market["stage"].min())

    # Build Cobre bus_id → name and NEWAVE code → bus_id lookups.
    cobre_name_to_id: dict[str, int] = {
        m["name"].strip().upper(): eid for eid, m in bus_meta.items()
    }
    nw_code_to_name: dict[int, str] = {
        code: name.strip().upper() for code, name in nw_bus_names.items()
    }

    # Match NEWAVE bus codes to Cobre bus IDs by name.
    matched: dict[int, tuple[int, str]] = {}  # nw_code → (cobre_bus_id, name)
    for nw_code, nw_name in nw_code_to_name.items():
        cid = cobre_name_to_id.get(nw_name)
        if cid is not None:
            matched[nw_code] = (cid, nw_name)

    # Preferred bus order.
    bus_order = ["SUDESTE", "SUL", "NORDESTE", "NORTE"]
    # Exclude fictitious buses.
    _skip = {"NOFICT1", "NOFICT2", "NOFICT3"}
    ordered_buses = []
    for bname in bus_order:
        for nw_code, (cid, name) in matched.items():
            if name == bname:
                ordered_buses.append((nw_code, cid, name))
                break
    for nw_code, (cid, name) in sorted(matched.items()):
        if name in _skip:
            continue
        if not any(b[2] == name for b in ordered_buses):
            ordered_buses.append((nw_code, cid, name))

    if not ordered_buses:
        return "<p>No matching buses found.</p>"

    # Pre-index NEWAVE data: {(nw_code, var_upper): {stage_0based: value}}
    nw_lookup: dict[tuple[int, str], dict[int, float]] = {}
    for row in nw_market.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        stage = int(row["stage"]) - nw_offset
        var = str(row["variable"]).strip().upper()
        nw_lookup.setdefault((code, var), {})[stage] = float(row["value"])

    # Pre-index Cobre percentile data: {bus_id: {stage: row_dict}}
    cobre_lookup: dict[int, dict[int, dict]] = {}
    for row in bus_agg.iter_rows(named=True):
        bid = int(row["bus_id"])
        sid = int(row["stage_id"])
        cobre_lookup.setdefault(bid, {})[sid] = row

    from cobre_bridge.comparators.html_report import (
        chart_grid,
        section_title,
        wrap_chart,
    )

    parts: list[str] = []

    for display_label, nw_var, cb_var, unit in _BALANCE_VARS:
        p10_col = f"{cb_var}_p10"
        p50_col = f"{cb_var}_p50"
        p90_col = f"{cb_var}_p90"

        # Check if Cobre has this variable.
        has_cobre = not bus_agg.is_empty() and p50_col in bus_agg.columns
        has_newave = bool(nw_var)

        if not has_cobre and not has_newave:
            continue

        parts.append(section_title(display_label))
        charts: list[str] = []

        ncols = 2
        nrows = (len(ordered_buses) + 1) // ncols
        traces: list[dict] = []
        layout: dict = {"title": f"{display_label} ({unit})"}
        first = True

        for idx, (nw_code, cid, bname) in enumerate(ordered_buses):
            row_i = idx // ncols
            col_i = idx % ncols
            ax_idx = idx + 1

            xa = f"x{ax_idx}" if ax_idx > 1 else "x"
            ya = f"y{ax_idx}" if ax_idx > 1 else "y"

            x0 = col_i * 0.52
            x1 = x0 + 0.47
            y1 = 1.0 - row_i * 0.52
            y0 = y1 - 0.44

            xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
            ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
            layout[xa_key] = {
                "domain": [round(x0, 3), round(x1, 3)],
                "title": "Stage" if row_i == nrows - 1 else "",
                "anchor": ya,
            }
            layout[ya_key] = {
                "domain": [round(y0, 3), round(y1, 3)],
                "title": bname,
                "anchor": xa,
            }

            # Determine stage range from Cobre data.
            bus_pct = cobre_lookup.get(cid, {})
            nw_data = nw_lookup.get((nw_code, nw_var), {}) if nw_var else {}
            all_stages = sorted(set(bus_pct.keys()) | set(nw_data.keys()))
            if not all_stages:
                continue

            # Cobre P10-P90 band.
            if has_cobre and bus_pct:
                p10 = [
                    float(bus_pct.get(s, {}).get(p10_col, 0) or 0) for s in all_stages
                ]
                p90 = [
                    float(bus_pct.get(s, {}).get(p90_col, 0) or 0) for s in all_stages
                ]
                p50 = [
                    float(bus_pct.get(s, {}).get(p50_col, 0) or 0) for s in all_stages
                ]
                traces.append(
                    {
                        "x": all_stages + all_stages[::-1],
                        "y": p90 + p10[::-1],
                        "fill": "toself",
                        "fillcolor": _BAND_FILL,
                        "line": {"color": _BAND_LINE},
                        "name": "Cobre P10–P90",
                        "type": "scatter",
                        "xaxis": xa,
                        "yaxis": ya,
                        "legendgroup": "band",
                        "showlegend": first,
                    }
                )
                traces.append(
                    {
                        "x": all_stages,
                        "y": p50,
                        "name": "Cobre Median",
                        "type": "scatter",
                        "mode": "lines",
                        "line": {"color": COLOR_COBRE, "width": 2},
                        "xaxis": xa,
                        "yaxis": ya,
                        "legendgroup": "cb",
                        "showlegend": first,
                    }
                )

            # NEWAVE mean line.
            if has_newave and nw_data:
                nw_y = [nw_data.get(s, 0) for s in all_stages]
                traces.append(
                    {
                        "x": all_stages,
                        "y": nw_y,
                        "name": "NEWAVE",
                        "type": "scatter",
                        "mode": "lines",
                        "line": {"color": COLOR_NEWAVE, "width": 2},
                        "xaxis": xa,
                        "yaxis": ya,
                        "legendgroup": "nw",
                        "showlegend": first,
                    }
                )

            first = False

        if traces:
            charts.append(
                wrap_chart(_plotly_div(traces, layout, height=nrows * 300 + 80))
            )
            parts.append(chart_grid(charts, single=True))

    if not parts:
        return "<p>No energy balance data available.</p>"
    return "\n".join(parts)


_BUS_ORDER = ["SUDESTE", "SUL", "NORDESTE", "NORTE"]


def system_per_bus_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
) -> str:
    """2x2 faceted per-bus chart with p10-p90 bands.

    Buses are ordered: SUDESTE, SUL, NORDESTE, NORTE.
    """
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]
    if not bus_data:
        return f"<p>No {variable} data available.</p>"

    buses: dict[str, list[ResultComparison]] = {}
    for r in bus_data:
        buses.setdefault(r.entity_name.upper(), []).append(r)

    # Order buses: preferred order first, then any remaining.
    ordered = [b for b in _BUS_ORDER if b in buses]
    ordered += [b for b in sorted(buses) if b not in ordered]
    if not ordered:
        return f"<p>No {variable} data available.</p>"

    # Build per-bus p10/p90 lookups from percentile data.
    pct_by_eid: dict[int, dict[int, dict]] = {}
    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    if pct_df is not None and not pct_df.is_empty():
        if p10_col in pct_df.columns and p90_col in pct_df.columns:
            for r in pct_df.iter_rows(named=True):
                eid = int(r["entity_id"])
                sid = int(r["stage_id"])
                pct_by_eid.setdefault(eid, {})[sid] = r

    # 2x2 grid using plotly subplots via xaxis/yaxis domains.
    n = len(ordered)
    ncols = 2
    nrows = (n + 1) // ncols
    traces: list[dict] = []
    layout: dict = {"title": title}
    first = True

    for idx, bus_name in enumerate(ordered):
        rows_list = buses[bus_name]
        row_i = idx // ncols
        col_i = idx % ncols
        ax_idx = idx + 1

        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        x0 = col_i * 0.52
        x1 = x0 + 0.47
        y1 = 1.0 - row_i * 0.52
        y0 = y1 - 0.44

        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": [round(x0, 3), round(x1, 3)],
            "title": "Stage" if row_i == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": [round(y0, 3), round(y1, 3)],
            "title": bus_name,
            "anchor": xa,
        }

        rows_sorted = sorted(rows_list, key=lambda r: r.stage)
        stages = [r.stage for r in rows_sorted]
        nw = [r.newave_value for r in rows_sorted]
        cb = [r.cobre_value for r in rows_sorted]

        # P10-P90 band for this bus.
        cobre_id = rows_sorted[0].cobre_id if rows_sorted else None
        bus_pct = pct_by_eid.get(cobre_id, {}) if cobre_id is not None else {}
        if bus_pct:
            p10 = [float(bus_pct.get(s, {}).get(p10_col, 0) or 0) for s in stages]
            p90 = [float(bus_pct.get(s, {}).get(p90_col, 0) or 0) for s in stages]
            traces.append(
                {
                    "x": stages + stages[::-1],
                    "y": p90 + p10[::-1],
                    "fill": "toself",
                    "fillcolor": _BAND_FILL,
                    "line": {"color": _BAND_LINE},
                    "name": "Cobre P10–P90",
                    "type": "scatter",
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "band",
                    "showlegend": first,
                }
            )

        traces.append(
            {
                "x": stages,
                "y": nw,
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "nw",
                "showlegend": first,
            }
        )
        traces.append(
            {
                "x": stages,
                "y": cb,
                "name": "Cobre Mean",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "cb",
                "showlegend": first,
            }
        )
        first = False

    return _plotly_div(traces, layout, height=nrows * 300 + 80)


def _subplot_domains(n: int) -> list[tuple[float, float]]:
    """Compute non-overlapping y-axis domains for n subplots."""
    gap = 0.05
    h = (1.0 - gap * (n - 1)) / n
    domains = []
    for i in range(n):
        bottom = i * (h + gap)
        domains.append((round(bottom, 4), round(bottom + h, 4)))
    domains.reverse()
    return domains


# -------------------------------------------------------------------
# Interactive plant details
# -------------------------------------------------------------------

_HYDRO_VARIABLES = [
    ("storage_final_hm3", "Storage (hm³)"),
    ("generation_mw", "Generation (MW)"),
    ("turbined_m3s", "Turbined (m³/s)"),
    ("spillage_m3s", "Spillage (m³/s)"),
    ("inflow_m3s", "Inflow (m³/s)"),
    ("water_value_per_hm3", "Water Value (R$/hm³)"),
]


def _enrich_with_percentiles(
    js_plants: dict[str, dict],
    variables: list[tuple[str, str]],
    pct_df: pl.DataFrame | None,
    cobre_id_key: str = "cobre_id",
) -> None:
    """Add p10/p90 arrays to each plant entry from percentile data."""
    if pct_df is None or pct_df.is_empty():
        return

    for _pid, entry in js_plants.items():
        cid = entry.get(cobre_id_key)
        if cid is None:
            continue
        sub = pct_df.filter(pl.col("entity_id") == cid).sort("stage_id")
        if sub.is_empty():
            continue
        pct_map = {int(r["stage_id"]): r for r in sub.iter_rows(named=True)}
        for var_key, _ in variables:
            stages = entry.get(f"{var_key}_stages", [])
            p10_col = f"{var_key}_p10"
            p90_col = f"{var_key}_p90"
            if p10_col in sub.columns and p90_col in sub.columns:
                entry[f"{var_key}_p10"] = [
                    round(float(pct_map.get(s, {}).get(p10_col, 0) or 0), 2)
                    for s in stages
                ]
                entry[f"{var_key}_p90"] = [
                    round(float(pct_map.get(s, {}).get(p90_col, 0) or 0), 2)
                    for s in stages
                ]


def build_hydro_detail_tab(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
) -> str:
    """Build interactive per-plant hydro detail with JS dropdown."""
    hydro_data = [r for r in results if r.entity_type == "hydro"]
    if not hydro_data:
        return "<p>No hydro data available.</p>"

    plants: dict[tuple[str, int], dict[str, dict[int, tuple[float, float]]]] = {}
    cobre_ids: dict[tuple[str, int], int] = {}
    for r in hydro_data:
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {}).setdefault(r.variable, {})[r.stage] = (
            r.newave_value,
            r.cobre_value,
        )
        cobre_ids[key] = r.cobre_id

    if not plants:
        return "<p>No hydro data available.</p>"

    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        entry: dict = {
            "name": name,
            "code": nw_code,
            "cobre_id": cobre_ids.get((name, nw_code), -1),
        }
        for var_key, _var_label in _HYDRO_VARIABLES:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        js_plants[pid] = entry

    _enrich_with_percentiles(js_plants, _HYDRO_VARIABLES, pct_df)

    return _build_interactive_detail_html(
        js_plants,
        _HYDRO_VARIABLES,
        "hydro",
        "Hydro Plant",
    )


def build_thermal_detail_tab(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
) -> str:
    """Build interactive per-plant thermal detail with JS dropdown."""
    thermal_data = [r for r in results if r.entity_type == "thermal"]
    if not thermal_data:
        return "<p>No thermal data available.</p>"

    plants: dict[tuple[str, int], dict[str, dict[int, tuple[float, float]]]] = {}
    cobre_ids: dict[tuple[str, int], int] = {}
    for r in thermal_data:
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {}).setdefault(r.variable, {})[r.stage] = (
            r.newave_value,
            r.cobre_value,
        )
        cobre_ids[key] = r.cobre_id

    if not plants:
        return "<p>No thermal data available.</p>"

    thermal_vars = [("generation_mw", "Generation (MW)")]

    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        entry: dict = {
            "name": name,
            "code": nw_code,
            "cobre_id": cobre_ids.get((name, nw_code), -1),
        }
        for var_key, _var_label in thermal_vars:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        js_plants[pid] = entry

    _enrich_with_percentiles(js_plants, thermal_vars, pct_df)

    return _build_interactive_detail_html(
        js_plants,
        thermal_vars,
        "thermal",
        "Thermal Plant",
    )


def _build_interactive_detail_html(
    js_plants: dict[str, dict],
    variables: list[tuple[str, str]],
    prefix: str,
    label: str,
) -> str:
    """Build the HTML/JS for interactive per-plant detail charts."""
    import json as _json

    data_json = _json.dumps(js_plants)

    # Build chart divs.
    chart_divs: list[str] = []
    for var_key, var_label in variables:
        div_id = f"{prefix}-chart-{var_key.replace('_', '-')}"
        chart_divs.append(
            f'<div class="chart-card">'
            f'<div id="{div_id}" style="width:100%;height:350px;"></div>'
            f"</div>"
        )

    n_vars = len(variables)
    grid_class = "chart-grid" if n_vars > 1 else "chart-grid-single"
    charts_html = f'<div class="{grid_class}">{"".join(chart_divs)}</div>'

    # Build option list sorted by name.
    options: list[str] = []
    for pid, entry in sorted(js_plants.items(), key=lambda x: x[1]["name"]):
        name = entry["name"]
        code = entry["code"]
        options.append(f'<option value="{pid}">{name} ({code})</option>')

    # JS to update charts on selection (with optional p10/p90 bands).
    update_calls: list[str] = []
    for var_key, var_label in variables:
        div_id = f"{prefix}-chart-{var_key.replace('_', '-')}"
        update_calls.append(f"""
        (function() {{
            var s = d['{var_key}_stages'] || [];
            var nw = d['{var_key}_nw'] || [];
            var cb = d['{var_key}_cb'] || [];
            var p10 = d['{var_key}_p10'] || null;
            var p90 = d['{var_key}_p90'] || null;
            var traces = [];
            if (p10 && p90 && p10.length > 0) {{
                traces.push({{
                    x: s.concat(s.slice().reverse()),
                    y: p90.concat(p10.slice().reverse()),
                    fill: 'toself',
                    fillcolor: '{_BAND_FILL}',
                    line: {{color: '{_BAND_LINE}'}},
                    name: 'Cobre P10\u2013P90',
                    type: 'scatter',
                    legendgroup: 'band',
                    showlegend: true
                }});
                traces.push({{
                    x: s, y: p10,
                    name: 'P10', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
                traces.push({{
                    x: s, y: p90,
                    name: 'P90', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
            }}
            traces.push({{x: s, y: nw, name: 'NEWAVE', type: 'scatter',
                mode: 'lines', line: {{color: '{COLOR_NEWAVE}', width: 2}}}});
            traces.push({{x: s, y: cb, name: 'Cobre Mean', type: 'scatter',
                mode: 'lines', line: {{color: '{COLOR_COBRE}', width: 2}}}});
            Plotly.react('{div_id}', traces, {{
                title: d.name + ' \u2014 {var_label}',
                xaxis: {{title: 'Stage'}},
                yaxis: {{title: '{var_label}'}},
                legend: {_json.dumps(_LEGEND)},
                margin: {_json.dumps(_MARGIN)},
                template: 'plotly_white',
                height: 350
            }}, {{responsive: true}});
        }})();""")

    js = f"""
    var {prefix}Data = {data_json};
    function update{prefix.title()}Charts() {{
        var sel = document.getElementById('{prefix}-select');
        var pid = sel.value;
        var d = {prefix}Data[pid];
        if (!d) return;
        document.getElementById('{prefix}-info').innerHTML =
            '<span>Code: ' + d.code + '</span>';
        {"".join(update_calls)}
    }}
    document.addEventListener('DOMContentLoaded', function() {{
        var sel = document.getElementById('{prefix}-select');
        if (sel && sel.options.length > 0) {{
            update{prefix.title()}Charts();
        }}
    }});
    """

    return f"""
    <div class="plant-selector">
        <label for="{prefix}-select">{label}:</label>
        <select id="{prefix}-select"
                onchange="update{prefix.title()}Charts()">
            {"".join(options)}
        </select>
        <div class="plant-info" id="{prefix}-info"></div>
    </div>
    {charts_html}
    <script>{js}</script>
    """
