"""Chart implementations for the HTML comparison report.

All functions return HTML strings (plotly chart divs) that can be
embedded in the comparison report template.  Uses plotly.js via
inline JSON config (no python plotly server needed at report viewing
time).
"""

from __future__ import annotations

import json
import uuid

from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.results import (
    ResultComparison,
    ResultsSummary,
)

_LEGEND = dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="left",
    x=0,
    font=dict(size=11),
)
_MARGIN = dict(l=60, r=30, t=60, b=50)


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


def convergence_chart(
    results: list[ResultComparison],
) -> str:
    """Convergence overlay: NEWAVE vs Cobre lower/upper bounds."""
    conv = [r for r in results if r.entity_type == "convergence"]
    if not conv:
        return "<p>No convergence data available.</p>"

    lb_nw: dict[int, float] = {}
    lb_cb: dict[int, float] = {}
    ub_nw: dict[int, float] = {}
    ub_cb: dict[int, float] = {}

    for r in conv:
        if r.variable == "lower_bound":
            lb_nw[r.stage] = r.newave_value
            lb_cb[r.stage] = r.cobre_value
        elif r.variable == "upper_bound_mean":
            ub_nw[r.stage] = r.newave_value
            ub_cb[r.stage] = r.cobre_value

    iters = sorted(set(lb_nw) | set(ub_nw))
    if not iters:
        return "<p>No convergence data available.</p>"

    traces = [
        {
            "x": iters,
            "y": [lb_nw.get(i) for i in iters],
            "name": "NEWAVE ZINF",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_NEWAVE},
        },
        {
            "x": iters,
            "y": [lb_cb.get(i) for i in iters],
            "name": "Cobre Lower",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE},
        },
        {
            "x": iters,
            "y": [ub_nw.get(i) for i in iters],
            "name": "NEWAVE ZSUP",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_NEWAVE, "dash": "dash"},
        },
        {
            "x": iters,
            "y": [ub_cb.get(i) for i in iters],
            "name": "Cobre Upper",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "dash": "dash"},
        },
    ]

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
) -> str:
    """Line chart comparing a system variable by stage."""
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]
    if not bus_data:
        return f"<p>No {variable} data available.</p>"

    # Aggregate across all buses.
    nw_by_stage: dict[int, float] = {}
    cb_by_stage: dict[int, float] = {}
    for r in bus_data:
        nw_by_stage[r.stage] = nw_by_stage.get(r.stage, 0.0) + r.newave_value
        cb_by_stage[r.stage] = cb_by_stage.get(r.stage, 0.0) + r.cobre_value

    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = [
        {
            "x": stages,
            "y": [nw_by_stage.get(s, 0) for s in stages],
            "name": "NEWAVE",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_NEWAVE},
        },
        {
            "x": stages,
            "y": [cb_by_stage.get(s, 0) for s in stages],
            "name": "Cobre",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE},
        },
    ]

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": variable},
    }

    return _plotly_div(traces, layout)


# -------------------------------------------------------------------
# Hydro tab charts
# -------------------------------------------------------------------


def hydro_aggregate_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
) -> str:
    """Aggregate hydro comparison by stage."""
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

    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    traces = [
        {
            "x": stages,
            "y": [nw_by_stage.get(s, 0) for s in stages],
            "name": "NEWAVE",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_NEWAVE},
        },
        {
            "x": stages,
            "y": [cb_by_stage.get(s, 0) for s in stages],
            "name": "Cobre",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE},
        },
    ]

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
) -> str:
    """Aggregate thermal generation comparison by stage."""
    return hydro_aggregate_chart(results, "generation_mw", "Thermal Generation")


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


def system_per_bus_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
) -> str:
    """Subplot per bus for a system variable (one row per bus)."""
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]
    if not bus_data:
        return f"<p>No {variable} data available.</p>"

    buses: dict[str, list[ResultComparison]] = {}
    for r in bus_data:
        buses.setdefault(r.entity_name, []).append(r)

    n = len(buses)
    if n == 0:
        return f"<p>No {variable} data available.</p>"

    traces: list[dict] = []
    annotations: list[dict] = []
    y_domains = _subplot_domains(n)

    for idx, (bus_name, rows) in enumerate(sorted(buses.items())):
        rows_sorted = sorted(rows, key=lambda r: r.stage)
        stages = [r.stage for r in rows_sorted]
        nw = [r.newave_value for r in rows_sorted]
        cb = [r.cobre_value for r in rows_sorted]
        ya = f"y{idx + 1}" if idx > 0 else "y"
        show = idx == 0
        traces.append(
            {
                "x": stages,
                "y": nw,
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE},
                "yaxis": ya,
                "showlegend": show,
                "legendgroup": "nw",
            }
        )
        traces.append(
            {
                "x": stages,
                "y": cb,
                "name": "Cobre",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE},
                "yaxis": ya,
                "showlegend": show,
                "legendgroup": "cb",
            }
        )
        annotations.append(
            {
                "text": bus_name,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": y_domains[idx][1],
                "xanchor": "left",
                "yanchor": "bottom",
                "showarrow": False,
                "font": {"size": 11, "color": "#374151"},
            }
        )

    layout: dict = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "annotations": annotations,
    }
    for idx in range(n):
        key = f"yaxis{idx + 1}" if idx > 0 else "yaxis"
        layout[key] = {"domain": y_domains[idx]}
        if idx > 0:
            layout[f"xaxis{idx + 1}"] = {"anchor": f"y{idx + 1}"}
            for t in traces:
                if t.get("yaxis") == f"y{idx + 1}":
                    t["xaxis"] = f"x{idx + 1}"

    return _plotly_div(traces, layout, height=200 * n + 100)


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


def build_hydro_detail_tab(
    results: list[ResultComparison],
) -> str:
    """Build interactive per-plant hydro detail with JS dropdown.

    Embeds all plant data as JSON and uses JavaScript to update
    plotly charts when the user selects a different plant.
    """
    hydro_data = [r for r in results if r.entity_type == "hydro"]
    if not hydro_data:
        return "<p>No hydro data available.</p>"

    # Group: {(entity_name, newave_code): {variable: {stage: (nw, cb)}}}
    plants: dict[tuple[str, int], dict[str, dict[int, tuple[float, float]]]] = {}
    for r in hydro_data:
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {}).setdefault(r.variable, {})[r.stage] = (
            r.newave_value,
            r.cobre_value,
        )

    if not plants:
        return "<p>No hydro data available.</p>"

    # Build JSON data structure for JS.
    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        entry: dict = {"name": name, "code": nw_code}
        for var_key, _var_label in _HYDRO_VARIABLES:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        js_plants[pid] = entry

    return _build_interactive_detail_html(
        js_plants,
        _HYDRO_VARIABLES,
        "hydro",
        "Hydro Plant",
    )


def build_thermal_detail_tab(
    results: list[ResultComparison],
) -> str:
    """Build interactive per-plant thermal detail with JS dropdown."""
    thermal_data = [r for r in results if r.entity_type == "thermal"]
    if not thermal_data:
        return "<p>No thermal data available.</p>"

    plants: dict[tuple[str, int], dict[str, dict[int, tuple[float, float]]]] = {}
    for r in thermal_data:
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {}).setdefault(r.variable, {})[r.stage] = (
            r.newave_value,
            r.cobre_value,
        )

    if not plants:
        return "<p>No thermal data available.</p>"

    thermal_vars = [("generation_mw", "Generation (MW)")]

    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        entry: dict = {"name": name, "code": nw_code}
        for var_key, _var_label in thermal_vars:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        js_plants[pid] = entry

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

    # JS to update charts on selection.
    update_calls: list[str] = []
    for var_key, var_label in variables:
        div_id = f"{prefix}-chart-{var_key.replace('_', '-')}"
        update_calls.append(f"""
        var s = d['{var_key}_stages'] || [];
        var nw = d['{var_key}_nw'] || [];
        var cb = d['{var_key}_cb'] || [];
        Plotly.react('{div_id}', [
            {{x: s, y: nw, name: 'NEWAVE', type: 'scatter', mode: 'lines',
              line: {{color: '{COLOR_NEWAVE}'}}}},
            {{x: s, y: cb, name: 'Cobre', type: 'scatter', mode: 'lines',
              line: {{color: '{COLOR_COBRE}'}}}}
        ], {{
            title: d.name + ' — {var_label}',
            xaxis: {{title: 'Stage'}},
            yaxis: {{title: '{var_label}'}},
            legend: {_json.dumps(_LEGEND)},
            margin: {_json.dumps(_MARGIN)},
            template: 'plotly_white',
            height: 350
        }}, {{responsive: true}});""")

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
