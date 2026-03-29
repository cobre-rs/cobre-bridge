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
