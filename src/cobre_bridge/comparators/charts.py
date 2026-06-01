"""Chart implementations for the HTML comparison report."""

from __future__ import annotations

import pandas as pd
import polars as pl

from cobre_bridge.comparators.html_report import (
    COLOR_COBRE,
    COLOR_NEWAVE,
)
from cobre_bridge.comparators.results import (
    ResultComparison,
    ResultsSummary,
)
from cobre_bridge.horizon import is_effectively_infinite
from cobre_bridge.ui.html import escape_text, json_for_script
from cobre_bridge.ui.plotly_helpers import LEGEND_DEFAULTS as _LEGEND
from cobre_bridge.ui.plotly_helpers import MARGIN_DEFAULTS as _MARGIN
from cobre_bridge.ui.plotly_helpers import plotly_div as _plotly_div

_BAND_FILL = "rgba(74,144,184,0.15)"
_BAND_LINE = "rgba(255,255,255,0)"

# Per-category mapping between NEWAVE pmo.dat `custo_operacao_series_simuladas`
# `parcela` labels and cobre simulation cost-record columns.
#
# Tuple layout: (display_label, [newave_keys], [cobre_columns], hex_color).
#
# Ordering reflects logical grouping (generation, regularisation, hydro
# operational violations, generic violations) and drives the legend order in
# the stacked bar.
_COST_MAP: list[tuple[str, list[str], list[str], str]] = [
    # Operational / generation costs
    ("Thermal Generation", ["GERACAO TERMICA"], ["thermal_cost"], "#D97706"),
    ("Deficit", ["DEFICIT"], ["deficit_cost"], "#DC2626"),
    ("Energy Excess", ["EXCESSO ENERGIA"], ["excess_cost"], "#F59E0B"),
    ("Exchange", ["INTERCAMBIO"], ["exchange_cost"], "#7C3AED"),
    ("Pumping", [], ["pumping_cost"], "#0891B2"),
    # Regularisation costs (per-unit-flow charges, not violations)
    (
        "Spillage",
        ["VERTIMENTO", "VERTIMENTO UHE", "VERT. FIO N. TURB."],
        ["spillage_cost"],
        "#2563EB",
    ),
    ("Turbined Reg.", ["TURBINAMENTO UHE"], ["turbined_cost"], "#0EA5E9"),
    ("NCS Curtailment", ["CORTE GER. EOLICA"], ["curtailment_cost"], "#059669"),
    # Hydro operational-bound violations (slacks priced above regularisation)
    (
        "Outflow Min Viol.",
        ["VIOLACAO VZMIN"],
        ["outflow_violation_below_cost"],
        "#E11D48",
    ),
    (
        "Outflow Max Viol.",
        ["VIOL. DEFL. MAXIMA"],
        ["outflow_violation_above_cost"],
        "#F43F5E",
    ),
    (
        "Turbining Bounds Viol.",
        ["VIOL. TURB. MINIMO", "VIOL. TURB. MAXIMO"],
        ["turbined_violation_cost"],
        "#FB923C",
    ),
    (
        "Generation Bounds Viol.",
        ["VIOLACAO GHMIN", "VIOLACAO GHMINU"],
        ["generation_violation_cost"],
        "#F97316",
    ),
    (
        "Storage Bounds Viol.",
        [],
        ["storage_violation_cost"],
        "#C2410C",
    ),
    ("Filling Target Viol.", ["VIOLACAO EVMIN"], ["filling_target_cost"], "#A16207"),
    (
        "Water Withdrawal Viol.",
        ["VIOLACAO RETIRADA"],
        ["withdrawal_violation_cost"],
        "#0284C7",
    ),
    (
        "Evaporation Viol.",
        ["VIOL. EVAP. UHE"],
        ["evaporation_violation_cost"],
        "#9333EA",
    ),
    # FPHA folga slack (NEWAVE) — cobre has no direct analogue; left as
    # NEWAVE-only column so it shows up in the report rather than being hidden.
    ("FPHA Slack", ["VIOLACAO FPHA"], [], "#DB2777"),
    ("Inflow Non-Negativity", [], ["inflow_penalty_cost"], "#EA580C"),
    # Generic constraint violations: NEWAVE reports the risk-aversion curve and
    # surface (CAR/SAR), electric (RESTELETRICA), interchange-group
    # (INTERC. MIN.), hydraulic (RHQ/RHV) and piecewise-linear (RLPP) restriction
    # violations as separate parcelas, but cobre-bridge converts them all into
    # Cobre generic constraints, so Cobre aggregates their slacks into a single
    # `generic_violation_cost`. Sum the NEWAVE parcelas to compare like-for-like.
    (
        "Generic Constr. Viol.",
        [
            "VIOLACAO CAR",
            "VIOLACAO SAR",
            "VIOL. RESTELETRICA",
            "VIOL. INTERC. MIN.",
            "VIOLACAO RHQ",
            "VIOLACAO RHV",
            "VIOL.RLPP DEFLMAX",
            "VIOL.RLPP DEFLMAXU",
            "VIOL.RLPP TURBMAX",
            "VIOL.RLPP TURBMAXU",
        ],
        ["generic_violation_cost"],
        "#6D28D9",
    ),
]

_COST_COLOR_DEFAULT = "#6B7280"  # for any unmapped residual category

# Cobre cost-record fields that are aggregates / metadata and must not appear as
# their own category in the breakdown (they would double-count or pollute the
# chart). Mirrors the dashboard's `_NON_COST_COLS` exclusion set.
_COBRE_NON_COST_KEYS: frozenset[str] = frozenset(
    {
        "total_cost",
        "immediate_cost",
        "future_cost",
        "discount_factor",
        "hydro_violation_cost",  # aggregate of the 6 hydro-violation components
    }
)


def _resolve_cost_categories(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
) -> list[tuple[str, float, float, str]]:
    """Resolve NEWAVE/Cobre cost dicts into a sorted list of categories.

    Each entry is ``(display_label, newave_sum, cobre_sum, color)``. Categories
    with both sides ≤ 0.01 R$ are filtered out. Mapped entries from
    :data:`_COST_MAP` come first (preserving its logical ordering); unmapped
    NEWAVE/Cobre keys are appended at the end.
    """
    categories: list[tuple[str, float, float, str]] = []
    for display_label, nw_keys, cb_keys, color in _COST_MAP:
        nw_sum = sum(nw_costs.get(k, 0.0) for k in nw_keys)
        cb_sum = sum(cobre_costs.get(k, 0.0) for k in cb_keys)
        if abs(nw_sum) < 0.01 and abs(cb_sum) < 0.01:
            continue
        categories.append((display_label, nw_sum, cb_sum, color))

    mapped_nw = {k for _, nw_keys, _, _ in _COST_MAP for k in nw_keys}
    for k, v in sorted(nw_costs.items()):
        if k not in mapped_nw and abs(v) > 0.01:
            categories.append((k.title(), v, 0.0, _COST_COLOR_DEFAULT))

    mapped_cb = {k for _, _, cb_keys, _ in _COST_MAP for k in cb_keys}
    for k, v in sorted(cobre_costs.items()):
        if k in _COBRE_NON_COST_KEYS:
            continue
        if k not in mapped_cb and abs(v) > 0.01:
            categories.append(
                (k.replace("_", " ").title(), 0.0, v, _COST_COLOR_DEFAULT)
            )

    return categories


def cost_breakdown_chart(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
) -> str:
    """Stacked vertical bar comparing NEWAVE vs Cobre per cost category in NPV.

    Bars are stacked with the largest-magnitude category at the bottom for
    readability; each category has its own distinct color drawn from
    :data:`_COST_MAP`. Y-axis is 10⁹ R$ for legibility on Brazilian-scale
    cases.
    """
    if not nw_costs and not cobre_costs:
        return "<p>No cost data available.</p>"

    categories = _resolve_cost_categories(nw_costs, cobre_costs)
    if not categories:
        return "<p>No cost data available.</p>"

    # Sort so largest total cost is at the bottom of the stack (drawn first).
    categories = sorted(categories, key=lambda t: -(t[1] + t[2]))

    x_labels = ["NEWAVE", "Cobre"]
    traces: list[dict] = []

    for label, nw_v, cb_v, color in categories:
        traces.append(
            {
                "x": x_labels,
                "y": [round(nw_v / 1e9, 3), round(cb_v / 1e9, 3)],
                "name": label,
                "type": "bar",
                "marker": {"color": color},
                "hovertemplate": (
                    f"%{{x}}<br>{label}: %{{y:.3f}} 10⁹ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": "Cost Breakdown (Present Value)",
        "yaxis": {"title": "Cost (10⁹ R$)"},
        "barmode": "stack",
        "bargap": 0.4,
        # Vertical legend on the right — the cost map has 15+ categories
        # which wrap into 3+ rows when laid out horizontally above the
        # plot and collide with the chart title.
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 11},
        },
        "margin": {"l": 60, "r": 220, "t": 60, "b": 50},
    }

    return _plotly_div(traces, layout, height=600)


def cost_breakdown_table(
    nw_costs: dict[str, float],
    cobre_costs: dict[str, float],
) -> str:
    """Per-category NPV diff table — NEWAVE, Cobre, Δ, Δ% — sorted by |Δ|.

    Returns an HTML ``<table>`` styled to fit alongside
    :func:`cost_breakdown_chart` inside a 2-column ``chart_grid``. Color swatches
    in the first column mirror the colors used in the bar chart so the reader
    can cross-reference at a glance.
    """
    if not nw_costs and not cobre_costs:
        return "<p>No cost data available.</p>"

    categories = _resolve_cost_categories(nw_costs, cobre_costs)
    if not categories:
        return "<p>No cost data available.</p>"

    rows: list[tuple[str, float, float, float, float, str]] = []
    total_nw = 0.0
    total_cb = 0.0
    for label, nw_v, cb_v, color in categories:
        diff = cb_v - nw_v
        pct = (diff / nw_v * 100.0) if abs(nw_v) > 0.01 else float("nan")
        rows.append((label, nw_v, cb_v, diff, pct, color))
        total_nw += nw_v
        total_cb += cb_v
    # Sort by |Δ| descending so the largest disagreements surface first.
    rows.sort(key=lambda r: -abs(r[3]))

    def _fmt_money(v: float) -> str:
        return f"{v / 1e9:.3f}"

    def _fmt_pct(p: float) -> str:
        if p != p:  # NaN
            return "—"
        return f"{p:+.1f}%"

    def _diff_class(diff: float) -> str:
        if diff > 0.01 * 1e9:
            return "cb-num cb-diff-pos"
        if diff < -0.01 * 1e9:
            return "cb-num cb-diff-neg"
        return "cb-num"

    head = (
        "<thead><tr>"
        '<th class="cb-cat">Category</th>'
        '<th class="cb-num">NEWAVE</th>'
        '<th class="cb-num">Cobre</th>'
        '<th class="cb-num">Δ</th>'
        '<th class="cb-num">Δ%</th>'
        "</tr></thead>"
    )

    body_rows: list[str] = []
    for label, nw_v, cb_v, diff, pct, color in rows:
        swatch = f'<span class="cb-swatch" style="background:{color}"></span>'
        diff_cls = _diff_class(diff)
        body_rows.append(
            "<tr>"
            f'<td class="cb-cat">{swatch}{label}</td>'
            f'<td class="cb-num">{_fmt_money(nw_v)}</td>'
            f'<td class="cb-num">{_fmt_money(cb_v)}</td>'
            f'<td class="{diff_cls}">{_fmt_money(diff)}</td>'
            f'<td class="{diff_cls}">{_fmt_pct(pct)}</td>'
            "</tr>"
        )
    total_diff = total_cb - total_nw
    total_pct = (
        (total_diff / total_nw * 100.0) if abs(total_nw) > 0.01 else float("nan")
    )
    body_rows.append(
        "<tr>"
        '<td class="cb-cat">Total</td>'
        f'<td class="cb-num">{_fmt_money(total_nw)}</td>'
        f'<td class="cb-num">{_fmt_money(total_cb)}</td>'
        f'<td class="cb-num">{_fmt_money(total_diff)}</td>'
        f'<td class="cb-num">{_fmt_pct(total_pct)}</td>'
        "</tr>"
    )
    body = "<tbody>" + "".join(body_rows) + "</tbody>"
    caption = "<caption>NPV by Category (10⁹ R$)</caption>"
    return '<table class="cost-breakdown-table">' + caption + head + body + "</table>"


def _extract_stage_cost_series(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int,
    nw_variable: str,
    cb_column: str,
) -> tuple[list[int], dict[int, float], dict[int, float]]:
    """Pull aligned NEWAVE/Cobre per-stage cost series.

    Returns the sorted list of 0-based stages present on either side, plus
    ``{stage: R$}`` dicts for NEWAVE (from MEDIAS-SIN, converted from 10⁶ R$)
    and Cobre (from the simulation costs parquet).
    """
    nw_by_stage: dict[int, float] = {}
    if nw_sin is not None and not nw_sin.is_empty():
        sub = nw_sin.filter(pl.col("variable") == nw_variable)
        for row in sub.iter_rows(named=True):
            nw_by_stage[int(row["stage"]) - nw_offset] = float(row["value"]) * 1e6

    cb_by_stage: dict[int, float] = {}
    if cobre_stage_costs is not None and not cobre_stage_costs.is_empty():
        for row in cobre_stage_costs.iter_rows(named=True):
            v = row.get(cb_column)
            if v is None:
                continue
            cb_by_stage[int(row["stage_id"])] = float(v)

    stages = sorted(set(nw_by_stage) | set(cb_by_stage))
    return stages, nw_by_stage, cb_by_stage


def _stage_cost_subplot(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int,
    *,
    nw_variable: str,
    cb_column: str,
    title: str,
    nw_label: str,
    cb_label: str,
) -> str:
    """Render a single stage-cost line chart (one variable, both sides)."""
    stages, nw_by_stage, cb_by_stage = _extract_stage_cost_series(
        nw_sin, cobre_stage_costs, nw_offset, nw_variable, cb_column
    )
    if not stages:
        return f"<p>No {nw_variable} data available.</p>"

    def _series(by_stage: dict[int, float]) -> list[float | None]:
        return [round(by_stage[s] / 1e6, 4) if s in by_stage else None for s in stages]

    traces: list[dict] = []
    if nw_by_stage:
        traces.append(
            {
                "x": stages,
                "y": _series(nw_by_stage),
                "name": nw_label,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_NEWAVE},
                "hovertemplate": (
                    f"stage %{{x}}<br>{nw_label}: %{{y:.2f}} 10⁶ R$<extra></extra>"
                ),
            }
        )
    if cb_by_stage:
        traces.append(
            {
                "x": stages,
                "y": _series(cb_by_stage),
                "name": cb_label,
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_COBRE},
                "hovertemplate": (
                    f"stage %{{x}}<br>{cb_label}: %{{y:.2f}} 10⁶ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage (0-based)"},
        "yaxis": {"title": "Cost (10⁶ R$)"},
        "legend": {"orientation": "h", "y": -0.2},
    }
    return _plotly_div(traces, layout)


def immediate_cost_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
) -> str:
    """Per-stage *immediate* cost: NEWAVE ``COPER`` vs Cobre ``immediate_cost``.

    NEWAVE values come from MEDIAS-SIN in 10⁶ R$ (converted to R$ here by
    multiplying by 1e6).  Stage numbering on the NEWAVE side starts at the
    study's first calendar month — *nw_offset* (the minimum stage in
    MEDIAS-SIN) is subtracted to align with Cobre's 0-based ``stage_id``.
    """
    return _stage_cost_subplot(
        nw_sin,
        cobre_stage_costs,
        nw_offset,
        nw_variable="COPER",
        cb_column="immediate_cost",
        title="Immediate Cost — NEWAVE COPER vs Cobre",
        nw_label="NEWAVE COPER",
        cb_label="Cobre immediate_cost",
    )


def future_cost_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
) -> str:
    """Per-stage *future* cost: NEWAVE ``CUSTO_FUTURO`` vs Cobre ``future_cost``."""
    return _stage_cost_subplot(
        nw_sin,
        cobre_stage_costs,
        nw_offset,
        nw_variable="CUSTO_FUTURO",
        cb_column="future_cost",
        title="Future Cost — NEWAVE CUSTO_FUTURO vs Cobre",
        nw_label="NEWAVE CUSTO_FUTURO",
        cb_label="Cobre future_cost",
    )


def thermal_cost_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
) -> str:
    """Per-stage thermal cost: NEWAVE ``CTERM`` vs Cobre ``thermal_cost``.

    Unlike the immediate-cost chart (NEWAVE ``COPER``), CTERM is the live
    thermal generation cost on both sides, so this is an apples-to-apples
    comparison even in the post-study (where COPER is frozen — see
    :func:`other_costs_chart`).
    """
    return _stage_cost_subplot(
        nw_sin,
        cobre_stage_costs,
        nw_offset,
        nw_variable="CTERM",
        cb_column="thermal_cost",
        title="Thermal Cost — NEWAVE CTERM vs Cobre",
        nw_label="NEWAVE CTERM",
        cb_label="Cobre thermal_cost",
    )


def other_costs_chart(
    nw_sin: pl.DataFrame,
    cobre_stage_costs: pl.DataFrame,
    nw_offset: int = 0,
) -> str:
    """Per-stage non-thermal operation cost: ``COPER − CTERM`` per stage.

    NEWAVE: ``COPER − CTERM``. Cobre: ``immediate_cost − thermal_cost``. This
    isolates everything in the immediate cost that is *not* thermal generation
    (deficit, penalties, slacks). On the NEWAVE side it goes **negative** in the
    post-study because COPER is frozen at the last study value while CTERM
    tracks the live post-study thermal cost — so this chart surfaces that
    frozen-COPER gap explicitly.
    """
    _, nw_coper, cb_imm = _extract_stage_cost_series(
        nw_sin, cobre_stage_costs, nw_offset, "COPER", "immediate_cost"
    )
    _, nw_cterm, cb_therm = _extract_stage_cost_series(
        nw_sin, cobre_stage_costs, nw_offset, "CTERM", "thermal_cost"
    )

    nw_other = {s: nw_coper[s] - nw_cterm[s] for s in nw_coper if s in nw_cterm}
    cb_other = {s: cb_imm[s] - cb_therm[s] for s in cb_imm if s in cb_therm}

    stages = sorted(set(nw_other) | set(cb_other))
    if not stages:
        return "<p>No COPER/CTERM data available.</p>"

    def _series(by_stage: dict[int, float]) -> list[float | None]:
        return [round(by_stage[s] / 1e6, 4) if s in by_stage else None for s in stages]

    traces: list[dict] = []
    if nw_other:
        traces.append(
            {
                "x": stages,
                "y": _series(nw_other),
                "name": "NEWAVE COPER − CTERM",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_NEWAVE},
                "hovertemplate": (
                    "stage %{x}<br>NEWAVE COPER − CTERM: %{y:.2f} 10⁶ R$<extra></extra>"
                ),
            }
        )
    if cb_other:
        traces.append(
            {
                "x": stages,
                "y": _series(cb_other),
                "name": "Cobre immediate − thermal",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_COBRE},
                "hovertemplate": (
                    "stage %{x}<br>Cobre immediate − thermal: "
                    "%{y:.2f} 10⁶ R$<extra></extra>"
                ),
            }
        )

    layout = {
        "title": "Other Costs — COPER − CTERM (non-thermal operation)",
        "xaxis": {"title": "Stage (0-based)"},
        "yaxis": {"title": "Cost (10⁶ R$)"},
        "legend": {"orientation": "h", "y": -0.2},
    }
    return _plotly_div(traces, layout)


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
            # Wrap-around polygon — skip hover so the cursor's x-position
            # picks up the meaningful p10/p90 line traces instead of the
            # band's literal name.
            "hoverinfo": "skip",
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


def cobre_aggregate_chart(
    cobre_hydro: pl.DataFrame,
    variable: str,
    title: str,
    unit: str,
    pct_df: pl.DataFrame | None = None,
    *,
    nw_sin: pl.DataFrame | None = None,
    nw_variable: str | None = None,
    nw_factor: float = 1.0,
    nw_offset: int = 0,
    matched_ids: set[int] | None = None,
) -> str:
    """System-aggregate chart for a Cobre per-hydro variable.

    Sums *variable* across all (or matched) hydros per stage to produce a
    system-total mean line.  Adds a p10-p90 band from ``pct_df`` and an
    optional NEWAVE SIN-level line from ``nw_sin`` (multiplied by
    ``nw_factor`` for unit conversion).

    Parameters
    ----------
    cobre_hydro:
        Per-hydro Cobre means with ``entity_id``, ``stage_id``, and
        ``variable`` columns.
    variable:
        Column name in ``cobre_hydro`` and percentile prefix in ``pct_df``.
    title, unit:
        Chart title and y-axis unit label.
    pct_df:
        Per-hydro Cobre percentiles for the same variable.
    nw_sin:
        Long-format NEWAVE SIN DataFrame (``newave_code``, ``stage``,
        ``variable``, ``value``) — typically read by ``read_medias_sin``.
    nw_variable:
        Variable name to filter in ``nw_sin`` (e.g. ``"EARMF"``).
    nw_factor:
        Multiplicative factor applied to NEWAVE values for unit alignment
        (e.g. ``730`` to convert MWmes → MWh).
    nw_offset:
        Subtracted from NEWAVE ``stage`` to align with Cobre ``stage_id``
        (NEWAVE columns are numbered from the study start month).
    matched_ids:
        Optional subset of Cobre hydro IDs to include — keeps the
        aggregate consistent with comparisons that only cover matched
        plants.
    """
    if cobre_hydro.is_empty() or variable not in cobre_hydro.columns:
        return f"<p>No {variable} data available.</p>"

    df = cobre_hydro
    if matched_ids is not None:
        df = df.filter(pl.col("entity_id").is_in(list(matched_ids)))

    cobre_agg = (
        df.group_by("stage_id").agg(pl.col(variable).sum().alias("v")).sort("stage_id")
    )
    cobre_by_stage = {
        int(r["stage_id"]): float(r["v"]) for r in cobre_agg.iter_rows(named=True)
    }

    nw_by_stage: dict[int, float] = {}
    if nw_sin is not None and nw_variable is not None and not nw_sin.is_empty():
        sin_df = nw_sin.filter(
            pl.col("variable").str.strip_chars().str.to_uppercase() == nw_variable
        )
        for r in sin_df.iter_rows(named=True):
            stage_raw = r.get("stage")
            val = r.get("value")
            if stage_raw is None or val is None:
                continue
            s = int(stage_raw) - nw_offset
            nw_by_stage[s] = nw_by_stage.get(s, 0.0) + float(val) * nw_factor

    stages = sorted(set(cobre_by_stage) | set(nw_by_stage))

    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)
    if nw_by_stage:
        traces.append(
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": "NEWAVE SIN",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            }
        )
    traces.append(
        {
            "x": stages,
            "y": [cobre_by_stage.get(s, 0) for s in stages],
            "name": "Cobre Mean",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 2},
        }
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": f"{title} ({unit})"},
    }

    return _plotly_div(traces, layout)


_HYDRO_BUS_ORDER: list[str] = ["SUDESTE", "SUL", "NORDESTE", "NORTE"]


def hydro_per_bus_chart(
    results: list[ResultComparison],
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None,
    hydro_meta: dict[int, dict],
    bus_meta: dict[int, dict],
) -> str:
    """Per-bus faceted hydro comparison for *variable*.

    Aggregates hydro-plant ResultComparison rows by the plant's owning
    bus (taken from ``hydro_meta[cobre_id]["bus_id"]``), then renders a
    small-multiples grid (one panel per non-fictitious bus, same layout
    convention as ``line_summary_chart``) with NEWAVE + Cobre traces
    and an optional Cobre P10–P90 band summed across each bus's plants.

    Returns a short ``<p>`` fallback when the variable is absent on
    both sides or no plants can be mapped to buses.
    """
    hydro_data = [
        r for r in results if r.entity_type == "hydro" and r.variable == variable
    ]
    if not hydro_data:
        return f"<p>No hydro {variable} data.</p>"

    bus_id_to_name: dict[int, str] = {
        bid: meta.get("name", str(bid)) for bid, meta in bus_meta.items()
    }
    hydro_to_bus: dict[int, int] = {
        hid: meta["bus_id"]
        for hid, meta in hydro_meta.items()
        if meta.get("bus_id") is not None
    }

    # Aggregate per (bus_name, stage).
    per_bus_nw: dict[str, dict[int, float]] = {}
    per_bus_cb: dict[str, dict[int, float]] = {}
    per_bus_ids: dict[str, set[int]] = {}
    fictitious_skip = {"NOFICT1", "NOFICT2", "NOFICT3"}
    for r in hydro_data:
        bus_id = hydro_to_bus.get(r.cobre_id)
        if bus_id is None:
            continue
        bus_name = bus_id_to_name.get(bus_id, str(bus_id)).upper()
        if bus_name in fictitious_skip:
            continue
        per_bus_nw.setdefault(bus_name, {})
        per_bus_cb.setdefault(bus_name, {})
        per_bus_nw[bus_name][r.stage] = (
            per_bus_nw[bus_name].get(r.stage, 0.0) + r.newave_value
        )
        per_bus_cb[bus_name][r.stage] = (
            per_bus_cb[bus_name].get(r.stage, 0.0) + r.cobre_value
        )
        per_bus_ids.setdefault(bus_name, set()).add(r.cobre_id)

    if not per_bus_nw:
        return f"<p>No hydro {variable} data mapped to buses.</p>"

    # Preferred bus order with any extras appended.
    ordered = [b for b in _HYDRO_BUS_ORDER if b in per_bus_nw]
    ordered += [b for b in sorted(per_bus_nw) if b not in ordered]

    # Build aggregate-percentile lookups per bus (sum of plant p10/p90).
    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    per_bus_pct: dict[str, dict[int, tuple[float, float]]] = {}
    if (
        pct_df is not None
        and not pct_df.is_empty()
        and {p10_col, p90_col}.issubset(pct_df.columns)
    ):
        for bus_name, ids in per_bus_ids.items():
            agg = (
                pct_df.filter(pl.col("entity_id").is_in(list(ids)))
                .group_by("stage_id")
                .agg(pl.col(p10_col).sum(), pl.col(p90_col).sum())
            )
            per_bus_pct[bus_name] = {
                int(r["stage_id"]): (float(r[p10_col]), float(r[p90_col]))
                for r in agg.iter_rows(named=True)
            }

    ncols = 2
    nrows = (len(ordered) + ncols - 1) // ncols
    row_gap = 0.06
    row_h = max((1.0 - row_gap * (nrows - 1)) / nrows, 0.001)
    col_gap = 0.05
    col_w = (1.0 - col_gap * (ncols - 1)) / ncols

    traces: list[dict] = []
    layout: dict = {"title": title}
    first = True

    for idx, bus_name in enumerate(ordered):
        stages = sorted(set(per_bus_nw[bus_name]) | set(per_bus_cb[bus_name]))
        nw = [per_bus_nw[bus_name].get(s, 0.0) for s in stages]
        cb = [per_bus_cb[bus_name].get(s, 0.0) for s in stages]
        if not stages:
            continue

        row_i = idx // ncols
        col_i = idx % ncols
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        x0 = col_i * (col_w + col_gap)
        x1 = x0 + col_w
        y1 = 1.0 - row_i * (row_h + row_gap)
        y0 = y1 - row_h

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

        bus_pct = per_bus_pct.get(bus_name, {})
        if bus_pct:
            p10 = [bus_pct.get(s, (0.0, 0.0))[0] for s in stages]
            p90 = [bus_pct.get(s, (0.0, 0.0))[1] for s in stages]
            traces.append(
                {
                    "x": stages + stages[::-1],
                    "y": p90 + p10[::-1],
                    "fill": "toself",
                    "fillcolor": _BAND_FILL,
                    "line": {"color": _BAND_LINE},
                    "name": "Cobre P10–P90",
                    "hoverinfo": "skip",
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

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))


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


def _hydro_per_stage_sum(
    df: pl.DataFrame | None,
    variable: str,
    matched_ids: set[int] | None,
) -> dict[int, float]:
    """Sum *variable* across (matched) hydros per ``stage_id``.

    Returns an empty dict when the frame is missing/empty or the column is
    absent.  Used by the slack-aggregate / slack-per-bus chart helpers to
    collapse per-(entity_id, stage_id) frames into a per-stage SIN total.
    """
    if df is None or df.is_empty() or variable not in df.columns:
        return {}
    filtered = df
    if matched_ids is not None:
        filtered = df.filter(pl.col("entity_id").is_in(list(matched_ids)))
    agg = (
        filtered.group_by("stage_id")
        .agg(pl.col(variable).sum().alias("v"))
        .sort("stage_id")
    )
    return {int(r["stage_id"]): float(r["v"]) for r in agg.iter_rows(named=True)}


def hydro_slack_aggregate_chart(
    cobre_hydro: pl.DataFrame,
    nw_slacks: pl.DataFrame | None,
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None = None,
    matched_ids: set[int] | None = None,
    unit: str = "m³/s",
) -> str:
    """SIN-total slack chart from per-(entity_id, stage_id) frames.

    Mirrors :func:`hydro_aggregate_chart` but reads both sides from
    per-hydro frames instead of ``ResultComparison`` rows — the four hydro
    slacks (water-withdrawal pos/neg + evaporation pos/neg) plus the
    Cobre-only inflow non-negativity slack don't go through the comparison
    pipeline, so the chart machinery has to consume Cobre's
    ``cobre_hydro_means`` columns and the NEWAVE ``nw_hydro_slacks`` frame
    (or ``None`` for slacks without a NEWAVE counterpart) directly.
    """
    cobre_by_stage = _hydro_per_stage_sum(cobre_hydro, variable, matched_ids)
    nw_by_stage = _hydro_per_stage_sum(nw_slacks, variable, matched_ids)
    if not cobre_by_stage and not nw_by_stage:
        return f"<p>No {variable} data available.</p>"

    stages = sorted(set(cobre_by_stage) | set(nw_by_stage))
    traces = _aggregate_percentile_traces(pct_df, variable, stages, matched_ids)
    if nw_by_stage:
        traces.append(
            {
                "x": stages,
                "y": [nw_by_stage.get(s, 0) for s in stages],
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_NEWAVE, "width": 2},
            }
        )
    traces.append(
        {
            "x": stages,
            "y": [cobre_by_stage.get(s, 0) for s in stages],
            "name": "Cobre Mean",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": COLOR_COBRE, "width": 2},
        }
    )

    layout = {
        "title": title,
        "xaxis": {"title": "Stage"},
        "yaxis": {"title": f"{title} ({unit})"},
    }
    return _plotly_div(traces, layout)


def hydro_slack_per_bus_chart(
    cobre_hydro: pl.DataFrame,
    nw_slacks: pl.DataFrame | None,
    variable: str,
    title: str,
    pct_df: pl.DataFrame | None,
    hydro_meta: dict[int, dict],
    bus_meta: dict[int, dict],
    matched_ids: set[int] | None = None,
) -> str:
    """Per-bus faceted slack chart from per-(entity_id, stage_id) frames.

    Parallel to :func:`hydro_per_bus_chart` for the slack variables that
    aren't surfaced through ``ResultComparison`` (no Cobre/NEWAVE comparison
    row exists for them).  Plants are bucketed by their owning bus via
    ``hydro_meta[cobre_id]["bus_id"]``; fictitious buses (``NOFICT*``) are
    excluded, matching the existing per-bus charts.  When *nw_slacks* is
    ``None`` or lacks the column, the NEWAVE trace is omitted (used for
    ``inflow_nonnegativity_slack_m3s`` which has no NEWAVE counterpart).
    """
    if cobre_hydro.is_empty() or variable not in cobre_hydro.columns:
        return f"<p>No {variable} data available.</p>"

    bus_id_to_name: dict[int, str] = {
        bid: meta.get("name", str(bid)) for bid, meta in bus_meta.items()
    }
    hydro_to_bus: dict[int, int] = {
        hid: meta["bus_id"]
        for hid, meta in hydro_meta.items()
        if meta.get("bus_id") is not None
    }

    def _per_bus_from_frame(
        df: pl.DataFrame | None,
    ) -> tuple[dict[str, dict[int, float]], dict[str, set[int]]]:
        out: dict[str, dict[int, float]] = {}
        ids: dict[str, set[int]] = {}
        if df is None or df.is_empty() or variable not in df.columns:
            return out, ids
        frame = df
        if matched_ids is not None:
            frame = df.filter(pl.col("entity_id").is_in(list(matched_ids)))
        fictitious_skip = {"NOFICT1", "NOFICT2", "NOFICT3"}
        for row in frame.iter_rows(named=True):
            eid = int(row["entity_id"])
            bus_id = hydro_to_bus.get(eid)
            if bus_id is None:
                continue
            bus_name = bus_id_to_name.get(bus_id, str(bus_id)).upper()
            if bus_name in fictitious_skip:
                continue
            sid = int(row["stage_id"])
            val = row.get(variable)
            if val is None:
                continue
            out.setdefault(bus_name, {})
            out[bus_name][sid] = out[bus_name].get(sid, 0.0) + float(val)
            ids.setdefault(bus_name, set()).add(eid)
        return out, ids

    per_bus_cb, per_bus_ids = _per_bus_from_frame(cobre_hydro)
    per_bus_nw, _ = _per_bus_from_frame(nw_slacks)
    if not per_bus_cb and not per_bus_nw:
        return f"<p>No {variable} data mapped to buses.</p>"

    ordered = [b for b in _HYDRO_BUS_ORDER if b in per_bus_cb or b in per_bus_nw]
    ordered += [
        b for b in sorted(set(per_bus_cb) | set(per_bus_nw)) if b not in ordered
    ]

    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    per_bus_pct: dict[str, dict[int, tuple[float, float]]] = {}
    if (
        pct_df is not None
        and not pct_df.is_empty()
        and {p10_col, p90_col}.issubset(pct_df.columns)
    ):
        for bus_name, ids in per_bus_ids.items():
            if not ids:
                continue
            agg = (
                pct_df.filter(pl.col("entity_id").is_in(list(ids)))
                .group_by("stage_id")
                .agg(pl.col(p10_col).sum(), pl.col(p90_col).sum())
            )
            per_bus_pct[bus_name] = {
                int(r["stage_id"]): (float(r[p10_col]), float(r[p90_col]))
                for r in agg.iter_rows(named=True)
            }

    ncols = 2
    nrows = (len(ordered) + ncols - 1) // ncols
    row_gap = 0.06
    row_h = max((1.0 - row_gap * (nrows - 1)) / nrows, 0.001)
    col_gap = 0.05
    col_w = (1.0 - col_gap * (ncols - 1)) / ncols

    traces: list[dict] = []
    layout: dict = {"title": title}
    first = True

    for idx, bus_name in enumerate(ordered):
        cb_map = per_bus_cb.get(bus_name, {})
        nw_map = per_bus_nw.get(bus_name, {})
        stages = sorted(set(cb_map) | set(nw_map))
        if not stages:
            continue

        row_i = idx // ncols
        col_i = idx % ncols
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        x0 = col_i * (col_w + col_gap)
        x1 = x0 + col_w
        y1 = 1.0 - row_i * (row_h + row_gap)
        y0 = y1 - row_h

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

        bus_pct = per_bus_pct.get(bus_name, {})
        if bus_pct:
            p10 = [bus_pct.get(s, (0.0, 0.0))[0] for s in stages]
            p90 = [bus_pct.get(s, (0.0, 0.0))[1] for s in stages]
            traces.append(
                {
                    "x": stages + stages[::-1],
                    "y": p90 + p10[::-1],
                    "fill": "toself",
                    "fillcolor": _BAND_FILL,
                    "line": {"color": _BAND_LINE},
                    "name": "Cobre P10–P90",
                    "hoverinfo": "skip",
                    "type": "scatter",
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "band",
                    "showlegend": first,
                }
            )

        if nw_map:
            traces.append(
                {
                    "x": stages,
                    "y": [nw_map.get(s, 0.0) for s in stages],
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
                "y": [cb_map.get(s, 0.0) for s in stages],
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

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))


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
# Network (line interchange) tab charts
# -------------------------------------------------------------------


def line_summary_chart(
    results: list[ResultComparison],
    line_pct: pl.DataFrame | None,
    line_bounds: pd.DataFrame | None,
    line_meta: list[dict],
) -> str:
    """Per-line small-multiples comparing NEWAVE vs Cobre net flow.

    One panel per aligned line. Each panel shows the Cobre P10–P90
    band, Cobre median, and NEWAVE mean, plus dashed upper/lower
    capacity bounds (direct / −reverse).
    """
    line_data = [r for r in results if r.entity_type == "line"]
    if not line_data:
        return "<p>No line interchange data available.</p>"

    # Group by cobre_id.
    by_line: dict[int, list[ResultComparison]] = {}
    for r in line_data:
        by_line.setdefault(r.cobre_id, []).append(r)
    ordered_ids = sorted(by_line.keys())

    # Build per-line p10/p90 lookups from percentile data.
    pct_by_lid: dict[int, dict[int, dict]] = {}
    if line_pct is not None and not line_pct.is_empty():
        if {"net_flow_mw_p10", "net_flow_mw_p90"}.issubset(line_pct.columns):
            for r in line_pct.iter_rows(named=True):
                lid = int(r["entity_id"])
                sid = int(r["stage_id"])
                pct_by_lid.setdefault(lid, {})[sid] = r

    # Per-line stage-keyed capacity bounds.
    static_caps: dict[int, tuple[float, float]] = {}
    for lm in line_meta:
        cap = lm.get("capacity", {})
        static_caps[int(lm["id"])] = (
            float(cap.get("direct_mw", 0.0) or 0.0),
            float(cap.get("reverse_mw", 0.0) or 0.0),
        )
    stage_caps: dict[int, dict[int, tuple[float, float]]] = {}
    if line_bounds is not None and not line_bounds.empty:
        for _, row in line_bounds.iterrows():
            stage_caps.setdefault(int(row["line_id"]), {})[int(row["stage_id"])] = (
                float(row.get("direct_mw", 0.0) or 0.0),
                float(row.get("reverse_mw", 0.0) or 0.0),
            )

    ncols = 2
    nrows = (len(ordered_ids) + ncols - 1) // ncols
    # Distribute panels evenly across the [0, 1] y-domain with a
    # constant gap between rows. The previous fixed-stride formula
    # produced negative y-domains for nrows ≥ 3 (rendering the bottom
    # rows outside the chart area) which made bottom panels disappear.
    row_gap = 0.06
    row_h = max((1.0 - row_gap * (nrows - 1)) / nrows, 0.001)
    col_gap = 0.05
    col_w = (1.0 - col_gap * (ncols - 1)) / ncols

    traces: list[dict] = []
    layout: dict = {"title": "Net Line Flow (MW)"}
    first = True

    for idx, lid in enumerate(ordered_ids):
        rows_list = sorted(by_line[lid], key=lambda r: r.stage)
        row_i = idx // ncols
        col_i = idx % ncols
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        x0 = col_i * (col_w + col_gap)
        x1 = x0 + col_w
        y1 = 1.0 - row_i * (row_h + row_gap)
        y0 = y1 - row_h

        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": [round(x0, 3), round(x1, 3)],
            "title": "Stage" if row_i == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": [round(y0, 3), round(y1, 3)],
            "title": rows_list[0].entity_name if rows_list else f"line {lid}",
            "anchor": xa,
        }

        stages = [r.stage for r in rows_list]
        nw = [r.newave_value for r in rows_list]
        cb = [r.cobre_value for r in rows_list]

        # P10-P90 band.
        line_pct_map = pct_by_lid.get(lid, {})
        if line_pct_map:
            p10 = [
                float(line_pct_map.get(s, {}).get("net_flow_mw_p10", 0) or 0)
                for s in stages
            ]
            p90 = [
                float(line_pct_map.get(s, {}).get("net_flow_mw_p90", 0) or 0)
                for s in stages
            ]
            traces.append(
                {
                    "x": stages + stages[::-1],
                    "y": p90 + p10[::-1],
                    "fill": "toself",
                    "fillcolor": _BAND_FILL,
                    "line": {"color": _BAND_LINE},
                    "name": "Cobre P10–P90",
                    "hoverinfo": "skip",
                    "type": "scatter",
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "band",
                    "showlegend": first,
                }
            )

        # Capacity bound lines. Skip when both directions are
        # effectively infinite (NEWAVE big-M sentinel 99999 used for
        # fictitious connections) — those bounds dwarf real flows
        # and compress every other trace to a flat strip at zero.
        d_static, r_static = static_caps.get(lid, (0.0, 0.0))
        finite_upper: list[tuple[int, float]] = []
        finite_lower: list[tuple[int, float]] = []
        for s in stages:
            d_cap, r_cap = stage_caps.get(lid, {}).get(s, (d_static, r_static))
            if not is_effectively_infinite(d_cap):
                finite_upper.append((s, d_cap))
            if not is_effectively_infinite(r_cap):
                finite_lower.append((s, -r_cap))
        if finite_upper:
            traces.append(
                {
                    "x": [s for s, _ in finite_upper],
                    "y": [v for _, v in finite_upper],
                    "name": "Upper bound",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#DC4C4C", "width": 1.5, "dash": "dash"},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "bound",
                    "showlegend": first,
                }
            )
        if finite_lower:
            traces.append(
                {
                    "x": [s for s, _ in finite_lower],
                    "y": [v for _, v in finite_lower],
                    "name": "Lower bound",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#DC4C4C", "width": 1.5, "dash": "dash"},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "bound",
                    "showlegend": False,
                }
            )

        # NEWAVE + Cobre mean.
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

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))


# -------------------------------------------------------------------
# System spillage in MWmes
# -------------------------------------------------------------------


def system_spillage_energy_chart(
    results: list[ResultComparison],
    cobre_spill_energy: pl.DataFrame,
) -> str:
    """Three-panel chart of system spillage in MWmes.

    Each panel pairs a NEWAVE trace (``VERTOT`` / ``VERTcont`` /
    ``VERTfio``) against the matching Cobre aggregate (``total_mw`` /
    ``reservoir_mw`` / ``rorov_mw``).  Both axes are stage-average MW
    (MWmes).
    """
    nw_rows = [r for r in results if r.entity_type == "system_spillage"]
    if not nw_rows and cobre_spill_energy.is_empty():
        return "<p>No system spillage data available.</p>"

    nw_lookup: dict[str, dict[int, float]] = {}
    for r in nw_rows:
        nw_lookup.setdefault(r.variable, {})[r.stage] = r.newave_value

    cb_lookup: dict[str, dict[int, float]] = {}
    if not cobre_spill_energy.is_empty():
        for row in cobre_spill_energy.iter_rows(named=True):
            sid = int(row["stage_id"])
            cb_lookup.setdefault("spill_energy_total_mw", {})[sid] = float(
                row["total_mw"]
            )
            cb_lookup.setdefault("spill_energy_reservoir_mw", {})[sid] = float(
                row["reservoir_mw"]
            )
            cb_lookup.setdefault("spill_energy_rorov_mw", {})[sid] = float(
                row["rorov_mw"]
            )

    panels: list[tuple[str, str, str]] = [
        ("Total (VERTOT)", "spill_energy_total_mw", "VERTOT"),
        ("Reservoir cascades (VERTcont)", "spill_energy_reservoir_mw", "VERTcont"),
        ("Run-of-river (VERTfio)", "spill_energy_rorov_mw", "VERTfio"),
    ]

    traces: list[dict] = []
    layout: dict = {
        "title": "System Spillage Energy (MWmes)",
        "showlegend": True,
        "hovermode": "x unified",
        "legend": _LEGEND,
        "margin": _MARGIN,
    }

    for idx, (panel_title, var_key, nw_label) in enumerate(panels):
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"
        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"

        # Vertically stacked: y in [0,0.3], [0.35,0.65], [0.7,1.0]
        y1 = 1.0 - idx * 0.35
        y0 = y1 - 0.30
        layout[xa_key] = {
            "domain": [0.0, 1.0],
            "anchor": ya,
            "title": "Stage" if idx == len(panels) - 1 else "",
        }
        layout[ya_key] = {
            "domain": [round(y0, 3), round(y1, 3)],
            "anchor": xa,
            "title": panel_title,
        }

        all_stages = sorted(
            set(nw_lookup.get(var_key, {}).keys())
            | set(cb_lookup.get(var_key, {}).keys())
        )
        if not all_stages:
            continue

        nw_y = [nw_lookup.get(var_key, {}).get(s) for s in all_stages]
        cb_y = [cb_lookup.get(var_key, {}).get(s) for s in all_stages]

        first = idx == 0
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
        traces.append(
            {
                "x": all_stages,
                "y": cb_y,
                "name": "Cobre",
                "type": "scatter",
                "mode": "lines",
                "line": {"color": COLOR_COBRE, "width": 2},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "cb",
                "showlegend": first,
            }
        )

    if not traces:
        return "<p>No system spillage data available.</p>"

    return _plotly_div(traces, layout, height=720)


# -------------------------------------------------------------------
# Performance tab charts
# -------------------------------------------------------------------


def performance_metric_cards(
    nw_tim_stages: dict[str, float],
    cobre_training_seconds: float,
) -> str:
    """Headline timing KPIs: NEWAVE total / training, Cobre total, speedup."""
    from cobre_bridge.comparators.html_report import metric_card, metrics_grid
    from cobre_bridge.ui.theme import COMPARISON_COLORS

    nw_total = float(nw_tim_stages.get("Tempo Total", 0.0))
    nw_policy = float(nw_tim_stages.get("Calculo da Politica", 0.0))
    cb_total = float(cobre_training_seconds)
    speedup = (nw_policy / cb_total) if cb_total > 1e-6 else float("nan")

    def _fmt_dur(seconds: float) -> str:
        if seconds <= 0:
            return "—"
        if seconds < 60:
            return f"{seconds:.1f} s"
        if seconds < 3600:
            return f"{seconds / 60:.1f} min"
        return f"{seconds / 3600:.2f} h"

    def _fmt_x(v: float) -> str:
        return "—" if v != v else f"{v:.1f}×"

    cards = [
        metric_card(
            _fmt_dur(nw_total),
            "NEWAVE Total Wall-Clock",
            color=COMPARISON_COLORS.get("newave"),
        ),
        metric_card(
            _fmt_dur(nw_policy),
            "NEWAVE Policy Training",
            color=COMPARISON_COLORS.get("newave"),
        ),
        metric_card(
            _fmt_dur(cb_total),
            "Cobre Training",
            color=COMPARISON_COLORS.get("cobre"),
        ),
        metric_card(
            _fmt_x(speedup),
            "Speedup (NEWAVE policy ÷ Cobre training)",
            color=COMPARISON_COLORS.get("match"),
        ),
    ]
    return metrics_grid(cards)


def performance_iteration_chart(
    nw_tim_iterations: pl.DataFrame,
    cobre_convergence: pl.DataFrame,
) -> str:
    """Line chart of total seconds per training iteration.

    NEWAVE total times come from ``newave.tim`` (already in seconds);
    Cobre comes from ``training/convergence.parquet:time_total_ms``
    converted to seconds. Iteration 1 carries clock-init garbage on
    the NEWAVE side; we clip to a sensible max for the chart but show
    the raw value in the tooltip via a textual hover.
    """
    has_nw = not nw_tim_iterations.is_empty()
    has_cb = (
        not cobre_convergence.is_empty()
        and "iteration" in cobre_convergence.columns
        and "time_total_ms" in cobre_convergence.columns
    )
    if not has_nw and not has_cb:
        return "<p>No timing data available.</p>"

    traces: list[dict] = []
    all_seconds: list[float] = []

    if has_nw:
        it_col = nw_tim_iterations["iteration"].to_list()
        tot = nw_tim_iterations["total_seconds"].to_list()
        all_seconds.extend([v for v in tot if v < 1e5])  # exclude clock-init garbage
        traces.append(
            {
                "x": it_col,
                "y": tot,
                "name": "NEWAVE",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_NEWAVE, "width": 2},
                "marker": {"size": 5},
            }
        )

    if has_cb:
        df = (
            cobre_convergence.sort("iteration")
            if isinstance(cobre_convergence, pl.DataFrame)
            else cobre_convergence.sort_values("iteration")
        )
        it_col = (
            df["iteration"].to_list()
            if isinstance(df, pl.DataFrame)
            else df["iteration"].tolist()
        )
        ms = (
            df["time_total_ms"].to_list()
            if isinstance(df, pl.DataFrame)
            else df["time_total_ms"].tolist()
        )
        secs = [float(v) / 1000.0 for v in ms]
        all_seconds.extend(secs)
        traces.append(
            {
                "x": it_col,
                "y": secs,
                "name": "Cobre",
                "type": "scatter",
                "mode": "lines+markers",
                "line": {"color": COLOR_COBRE, "width": 2},
                "marker": {"size": 5},
            }
        )

    layout: dict = {
        "title": "Iteration Wall-Clock (seconds)",
        "xaxis": {"title": "Iteration"},
        "yaxis": {"title": "Total seconds"},
    }
    # Clip the y-axis to actual data so NEWAVE iter-1's clock-init garbage
    # doesn't compress every other iteration into a flat line at zero.
    if all_seconds:
        ymax = max(all_seconds) * 1.15
        layout["yaxis"]["range"] = [0, ymax]

    return _plotly_div(traces, layout, height=420)


def performance_fwd_bwd_split_chart(
    nw_tim_iterations: pl.DataFrame,
    cobre_convergence: pl.DataFrame,
) -> str:
    """Stacked forward / backward split per iteration, NEWAVE vs Cobre.

    Two panels stacked vertically: top panel = NEWAVE (backward +
    forward stacked bars in seconds), bottom panel = Cobre (same but
    converted from ms).
    """
    has_nw = not nw_tim_iterations.is_empty()
    has_cb = (
        not cobre_convergence.is_empty()
        and "iteration" in cobre_convergence.columns
        and {"time_forward_ms", "time_backward_ms"}.issubset(cobre_convergence.columns)
    )
    if not has_nw and not has_cb:
        return "<p>No forward/backward split available.</p>"

    panels: list[tuple[str, list[int], list[float], list[float]]] = []
    all_secs: list[float] = []
    if has_nw:
        it = nw_tim_iterations["iteration"].to_list()
        bw = nw_tim_iterations["backward_seconds"].to_list()
        fw = nw_tim_iterations["forward_seconds"].to_list()
        # Clip clock-init garbage on iter 1 so the chart auto-scales to the
        # real range. The raw value remains in the parsed DataFrame.
        bw_clipped = [v if v < 1e5 else 0.0 for v in bw]
        fw_clipped = [v if v < 1e5 else 0.0 for v in fw]
        panels.append(("NEWAVE", it, bw_clipped, fw_clipped))
        all_secs.extend([v for v in bw_clipped + fw_clipped if v > 0])
    if has_cb:
        df = cobre_convergence.sort("iteration")
        it = df["iteration"].to_list()
        bw = [float(v) / 1000.0 for v in df["time_backward_ms"].to_list()]
        fw = [float(v) / 1000.0 for v in df["time_forward_ms"].to_list()]
        panels.append(("Cobre", it, bw, fw))
        all_secs.extend(bw + fw)

    nrows = len(panels)
    row_gap = 0.10
    row_h = (1.0 - row_gap * (nrows - 1)) / nrows
    traces: list[dict] = []
    layout: dict = {"title": "Forward / Backward Split per Iteration (seconds)"}
    for idx, (label, it, bw, fw) in enumerate(panels):
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"
        y1 = 1.0 - idx * (row_h + row_gap)
        y0 = y1 - row_h
        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": [0.0, 1.0],
            "title": "Iteration" if idx == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": [round(y0, 3), round(y1, 3)],
            "title": f"{label} (s)",
            "anchor": xa,
        }
        first = idx == 0
        traces.append(
            {
                "x": it,
                "y": bw,
                "name": "Backward",
                "type": "bar",
                "marker": {"color": "#7C3AED"},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "bw",
                "showlegend": first,
            }
        )
        traces.append(
            {
                "x": it,
                "y": fw,
                "name": "Forward",
                "type": "bar",
                "marker": {"color": "#0EA5E9"},
                "xaxis": xa,
                "yaxis": ya,
                "legendgroup": "fw",
                "showlegend": first,
            }
        )
    layout["barmode"] = "stack"
    return _plotly_div(traces, layout, height=max(nrows * 320 + 80, 360))


# -------------------------------------------------------------------
# Productivity tab charts
# -------------------------------------------------------------------

# kind -> (pmo column, cobre-bridge column, pmo label, cobre-bridge label).
# Each productivity-comparison scatter is a *static* conversion-fidelity
# check: the NEWAVE pmo.dat productivity against the value cobre-bridge
# computes from the same HIDR cadastro inputs. Both sides live in the
# ``productivity_detail`` frame built in results.py and should land on y = x.
_PRODUCTIVITY_KINDS: dict[str, tuple[str, str, str, str]] = {
    "point": (
        "nw_altura_65",
        "cb_point",
        "produtibilidade_altura_65",
        "compute_productivity",
    ),
    "equivalent": (
        "nw_equivalent",
        "cb_equivalent",
        "produtibilidade_equivalente_volmin_volmax",
        "stored_energy_productivity",
    ),
    "accumulated": (
        "nw_accumulated_earm",
        "cb_accumulated",
        "produtibilidade_acumulada_calculo_earm",
        "accumulated_integrated_productivity",
    ),
}


def productivity_comparison_scatter(
    df: pl.DataFrame,
    kind: str,
    title: str | None = None,
) -> str:
    """Static conversion-fidelity scatter for one productivity *kind*.

    *kind* selects the (pmo, cobre-bridge) column pair from
    :data:`_PRODUCTIVITY_KINDS`: ``"point"`` (pmo ``produtibilidade_altura_65``
    vs ``compute_productivity``), ``"equivalent"`` (pmo
    ``produtibilidade_equivalente_volmin_volmax`` vs
    ``stored_energy_productivity``), ``"accumulated"`` (pmo
    ``produtibilidade_acumulada_calculo_earm`` vs the cascade
    accumulated-integrated value). Both sides are derived from the same
    NEWAVE inputs, so the points should land on the ``y = x`` reference line —
    this validates the conversion rather than comparing against the
    per-stage simulation output. NEWAVE pmo is on x, cobre-bridge on y; rows
    where either side is null are skipped. Annotated with mean & max relative
    error ``|cobre-bridge − pmo| / pmo`` and the number of plants compared.
    """
    if kind not in _PRODUCTIVITY_KINDS:
        raise ValueError(f"Unknown productivity kind: {kind!r}")
    nw_col, cb_col, nw_label, cb_label = _PRODUCTIVITY_KINDS[kind]
    if df.is_empty() or nw_col not in df.columns or cb_col not in df.columns:
        return "<p>No productivity data available.</p>"

    sub = df.select("plant_name", nw_col, cb_col).drop_nulls([nw_col, cb_col])
    if sub.is_empty():
        return "<p>No productivity data available.</p>"

    nw_vals = [float(v) for v in sub[nw_col].to_list()]
    cb_vals = [float(v) for v in sub[cb_col].to_list()]
    names = [escape_text(n) for n in sub["plant_name"].to_list()]

    rel_errs = [
        abs(cb - nw) / abs(nw) for nw, cb in zip(nw_vals, cb_vals) if abs(nw) > 1e-12
    ]
    mean_rel = sum(rel_errs) / len(rel_errs) if rel_errs else 0.0
    max_rel = max(rel_errs) if rel_errs else 0.0

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
            "hovertemplate": (
                "%{text}<br>pmo: %{x:.4f}<br>cobre-bridge: %{y:.4f}<extra></extra>"
            ),
        },
        {
            "x": [min_val, max_val],
            "y": [min_val, max_val],
            "name": "y = x",
            "type": "scatter",
            "mode": "lines",
            "line": {"color": "#8B9298", "dash": "dash"},
            "showlegend": False,
            "hoverinfo": "skip",
        },
    ]

    layout = {
        "title": title or f"Static productivity: {nw_label} vs {cb_label}",
        "xaxis": {"title": f"NEWAVE pmo {nw_label}"},
        "yaxis": {"title": f"cobre-bridge {cb_label}"},
        "annotations": [
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 0.98,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
                "bgcolor": "rgba(255,255,255,0.75)",
                "bordercolor": "#D1D5DB",
                "borderwidth": 1,
                "borderpad": 4,
                "font": {"size": 11},
                "text": (
                    f"N = {len(nw_vals)} plants<br>"
                    f"mean rel. err = {mean_rel * 100:.2f}%<br>"
                    f"max rel. err = {max_rel * 100:.2f}%"
                ),
            }
        ],
    }

    return _plotly_div(traces, layout)


def productivity_per_stage_chart(results: list[ResultComparison]) -> str:
    """Per-plant realized productivity (generation / turbined) across stages.

    Productivity is **constant within a stage but varies across stages** in
    both models, tracking the reservoir head reached each stage. Reuses the
    shared interactive per-plant widget (:func:`_build_interactive_detail_html`
    — the same JS ``<select>`` dropdown the hydro/thermal detail tabs use), so
    every reservoir is selectable one at a time (NEWAVE vs Cobre) rather than a
    hand-picked subset. Driven by the per-stage ``productivity_mw_per_m3s``
    hydro comparison rows already produced in results.py.
    """
    var_key = "productivity_mw_per_m3s"
    plants: dict[tuple[str, int], dict[int, tuple[float, float]]] = {}
    cobre_ids: dict[tuple[str, int], int] = {}
    for r in results:
        if r.entity_type != "hydro" or r.variable != var_key:
            continue
        key = (r.entity_name, r.newave_code)
        plants.setdefault(key, {})[r.stage] = (r.newave_value, r.cobre_value)
        cobre_ids[key] = r.cobre_id

    if not plants:
        return "<p>No per-stage productivity data available.</p>"

    js_plants: dict[str, dict] = {}
    for (name, code), stage_data in sorted(plants.items()):
        stages = sorted(stage_data)
        js_plants[f"{code}_{name}"] = {
            "name": name,
            "code": code,
            "cobre_id": cobre_ids[(name, code)],
            f"{var_key}_stages": stages,
            f"{var_key}_nw": [stage_data[s][0] for s in stages],
            f"{var_key}_cb": [stage_data[s][1] for s in stages],
        }

    variables = [(var_key, "Realized productivity — Gen / Turbined (MW per m³/s)")]
    return _build_interactive_detail_html(
        js_plants, variables, "prodstage", "Reservoir"
    )


def _prod_blocks_pct(nw: float | None, cb: float | None) -> float | None:
    """Relative diff (Cobre − NEWAVE)/NEWAVE in %, or None when NEWAVE ≈ 0."""
    if nw is None or cb is None or abs(nw) <= 1e-12:
        return None
    return (cb - nw) / nw * 100.0


def productivity_blocks_table(df: pl.DataFrame) -> str:
    """Grouped building-blocks table — per metric: NEWAVE | Cobre | Δ%.

    One row per aligned hydro. The columns are organised into metric groups
    (ρ_esp, tailwater, losses, vmin, vmax), each spanning three sub-columns —
    NEWAVE, Cobre, Δ% — via a two-level header (``colspan`` on the top row).
    Alternate metric groups get a subtle background tint across both header
    and body cells and a stronger left border, so the 2-by-2 (3-by-3 with Δ%)
    pairing is visually unmistakable. Δ% = (Cobre − NEWAVE)/NEWAVE (blank when
    NEWAVE ≈ 0); cells with ``|Δ%| > 1%`` are highlighted. Reuses the
    ``cost-breakdown-table`` styling.
    """
    if df.is_empty():
        return "<p>No productivity data available.</p>"

    # Column groups: (label, nw_col, cb_col, fmt_decimals).
    groups: list[tuple[str, str, str, int]] = [
        ("ρ_esp", "nw_specific_productivity", "cb_specific_productivity", 5),
        ("Tailwater (m)", "nw_tailwater_m", "cb_tailwater_m", 2),
        ("Losses (m)", "nw_losses_m", "cb_losses_m", 3),
        ("Vmin (hm³)", "nw_vmin_hm3", "cb_vmin_hm3", 1),
        ("Vmax (hm³)", "nw_vmax_hm3", "cb_vmax_hm3", 1),
    ]
    groups = [g for g in groups if g[1] in df.columns and g[2] in df.columns]
    if not groups:
        return "<p>No productivity data available.</p>"

    def _fmt(v: float | None, decimals: int) -> str:
        return "—" if v is None else f"{v:.{decimals}f}"

    def _fmt_pct(p: float | None) -> str:
        return "" if p is None else f"{p:+.1f}%"

    def _cls(idx: int, *, sub_index: int, extra: str = "") -> str:
        """Class string for a numeric cell in metric-group *idx*.

        Even groups (0-based) get ``cb-group-tint``; the first sub-column
        (``sub_index == 0``) of any group after the first gets
        ``cb-group-sep`` (the stronger vertical separator).
        """
        parts = ["cb-num"]
        if idx % 2 == 0:
            parts.append("cb-group-tint")
        if idx > 0 and sub_index == 0:
            parts.append("cb-group-sep")
        if extra:
            parts.append(extra)
        return " ".join(parts)

    # --- Two-level header ---
    top_cells = ['<th class="cb-cat" rowspan="2">Plant</th>']
    sub_cells: list[str] = []
    for idx, (label, _, _, _) in enumerate(groups):
        top_cells.append(
            f'<th class="{_cls(idx, sub_index=0)}" colspan="3">'
            f"{escape_text(label)}</th>"
        )
        for j, sub in enumerate(("NEWAVE", "Cobre", "Δ%")):
            sub_cells.append(f'<th class="{_cls(idx, sub_index=j)}">{sub}</th>')
    head = f"<thead><tr>{''.join(top_cells)}</tr><tr>{''.join(sub_cells)}</tr></thead>"

    # --- Body ---
    body_rows: list[str] = []
    for row in df.iter_rows(named=True):
        cells = [f'<td class="cb-cat">{escape_text(row["plant_name"])}</td>']
        for idx, (_, nw_col, cb_col, decimals) in enumerate(groups):
            nw_v = row.get(nw_col)
            cb_v = row.get(cb_col)
            pct = _prod_blocks_pct(nw_v, cb_v)
            highlight = "cb-diff-pos" if (pct is not None and abs(pct) > 1.0) else ""
            cells.append(
                f'<td class="{_cls(idx, sub_index=0)}">{_fmt(nw_v, decimals)}</td>'
            )
            cells.append(
                f'<td class="{_cls(idx, sub_index=1)}">{_fmt(cb_v, decimals)}</td>'
            )
            cells.append(
                f'<td class="{_cls(idx, sub_index=2, extra=highlight)}">'
                f"{_fmt_pct(pct)}</td>"
            )
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    body = "<tbody>" + "".join(body_rows) + "</tbody>"
    caption = (
        "<caption>Productivity Building Blocks "
        '<span class="cb-caption-note">— columns are grouped per metric: '
        "NEWAVE vs Cobre vs Δ%</span></caption>"
    )
    return (
        '<table class="cost-breakdown-table prod-blocks-table">'
        + caption
        + head
        + body
        + "</table>"
    )


# -------------------------------------------------------------------
# Summary metric cards
# -------------------------------------------------------------------


def overview_metrics(
    summary: ResultsSummary,
    nw_costs: dict[str, float] | None = None,
    cobre_costs: dict[str, float] | None = None,
) -> str:
    """Headline KPI cards for the overview tab.

    Four cards: total NPV cost on each side, the absolute Δ (10⁹ R$)
    with red/green coloring, and the relative Δ%. The previous
    Total Comparisons / Entity Types / Variables triple was meta about
    the report itself and offered no operational insight.
    """
    from cobre_bridge.comparators.html_report import (
        metric_card,
        metrics_grid,
    )
    from cobre_bridge.ui.theme import COMPARISON_COLORS

    # Pull thermal-generation cost only (single-category NPV). NEWAVE
    # parcela "GERACAO TERMICA" and Cobre "thermal_cost" are sourced
    # straight from ``_COST_MAP``.
    nw_thermal = (nw_costs or {}).get("GERACAO TERMICA", 0.0)
    cb_thermal = (cobre_costs or {}).get("thermal_cost", 0.0)
    diff = cb_thermal - nw_thermal
    pct = (diff / nw_thermal * 100.0) if abs(nw_thermal) > 1e-6 else float("nan")

    def _bn(v: float) -> str:
        return f"{v / 1e9:.3f}"

    def _pct(v: float) -> str:
        if v != v:  # NaN
            return "—"
        return f"{v:+.1f}%"

    # Color the Δ cards based on sign (Cobre overshoot = red, undershoot = green).
    diff_color = (
        COMPARISON_COLORS.get("diff", "#DC4C4C")
        if diff > 0
        else COMPARISON_COLORS.get("match", "#3F8E5F")
    )

    cards = [
        metric_card(
            _bn(nw_thermal),
            "NEWAVE Thermal Cost (10⁹ R$, NPV)",
            color=COMPARISON_COLORS.get("newave"),
        ),
        metric_card(
            _bn(cb_thermal),
            "Cobre Thermal Cost (10⁹ R$, NPV)",
            color=COMPARISON_COLORS.get("cobre"),
        ),
        metric_card(
            f"{diff / 1e9:+.3f}",
            "Δ Thermal Cost (Cobre − NEWAVE, 10⁹ R$)",
            color=diff_color,
        ),
        metric_card(
            _pct(pct),
            "Δ Thermal Cost (%)",
            color=diff_color,
        ),
    ]
    # Suppress the cards entirely when neither side reports cost data —
    # keeps the overview tidy on training-only outputs.
    if nw_thermal == 0 and cb_thermal == 0:
        if summary.total == 0:
            return ""
        cards = [
            metric_card(str(summary.total), "Comparisons"),
            metric_card(str(len(summary.by_entity_type)), "Entity Types"),
            metric_card(str(len(summary.by_variable)), "Variables"),
        ]
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
    ("Net Load", "NET_LOAD", "net_load_mw", "MW"),
    ("Deficit", "DEFT", "deficit_mw", "MW"),
    ("Excess", "EXCESSO", "excess_mw", "MW"),
]


def build_energy_balance_tab(
    nw_market: pl.DataFrame,
    bus_agg: pl.DataFrame,
    bus_meta: dict[int, dict],
    nw_bus_names: dict[int, str],
    *,
    nw_net_load: pl.DataFrame | None = None,
) -> str:
    """Build per-bus energy balance charts with p10/p90 bands.

    One 2x2 faceted chart per variable, with NEWAVE mean + Cobre p10/p50/p90.
    """
    if bus_agg.is_empty() and nw_market.is_empty():
        return "<p>No energy balance data available.</p>"

    nw_offset = 0
    if not nw_market.is_empty():
        nw_offset = int(nw_market["stage"].min())

    # Merge NEWAVE net load into nw_market if available.
    if nw_net_load is not None and not nw_net_load.is_empty():
        nw_market = pl.concat([nw_market, nw_net_load], how="diagonal_relaxed")
        if nw_offset == 0:
            nw_offset = int(nw_net_load["stage"].min())

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
                        "hoverinfo": "skip",
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
                    "hoverinfo": "skip",
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
    ("productivity_mw_per_m3s", "Productivity = Gen / Turbined (MW per m³/s)"),
    ("spillage_m3s", "Spillage (m³/s)"),
    ("outflow_m3s", "Total Outflow (m³/s)"),
    ("inflow_m3s", "Incremental Inflow (m³/s)"),
    ("total_inflow_m3s", "Total Inflow / QAFLUH (m³/s)"),
    ("evaporation_m3s", "Evaporation (m³/s)"),
    ("withdrawal_m3s", "Water Withdrawal (m³/s)"),
    ("water_value_per_hm3", "Water Value (R$/hm³)"),
]

# Cobre-only per-plant variables (NEWAVE has no per-plant equivalent).
# Appended to the dropdown after the comparison variables.
#
# Withdrawal-slack ``pos`` / ``neg`` labels follow NEWAVE's convention, which
# is the *inverse* of Cobre's column-name convention — Cobre's
# ``water_withdrawal_violation_pos_m3s`` is the physical equivalent of
# NEWAVE's ``VIOL_NEG_VRETIRUH`` and vice versa.  The
# ``_NW_HYDRO_SLACK_VARS`` mapping in ``results.py`` is correspondingly
# swapped so each panel pairs the right Cobre column with the right NEWAVE
# series under its NEWAVE-style label.  Evaporation slacks share NEWAVE's
# convention so no swap is needed there.
_HYDRO_COBRE_ONLY_VARIABLES = [
    ("stored_energy_initial_mwh", "Stored Energy Initial (MWh)"),
    ("stored_energy_final_mwh", "Stored Energy Final (MWh)"),
    ("incremental_inflow_energy_mw", "Natural Inflow Energy (MW)"),
    ("water_withdrawal_violation_neg_m3s", "Withdrawal Slack Pos (m³/s)"),
    ("water_withdrawal_violation_pos_m3s", "Withdrawal Slack Neg (m³/s)"),
    ("inflow_nonnegativity_slack_m3s", "Inflow Non-Negativity Slack (m³/s)"),
]

# Per-comparison-variable bound mapping: which static / per-stage bound
# columns to overlay as dashed reference lines.  Each entry is
# ``(static_meta_key, per_stage_bound_col)`` for the lower and upper
# bound respectively; either side may be ``None`` (e.g. "Outflow" has
# a min but typically no max in cobre).  The dashboard renders one
# dashed line per non-null bound; per-stage values shadow the static
# value when both are present at a given stage.
_HYDRO_BOUND_OVERLAY: dict[str, dict[str, tuple[str | None, str | None]]] = {
    "storage_final_hm3": {
        "min": ("min_storage_hm3", "min_storage_hm3"),
        "max": ("max_storage_hm3", "max_storage_hm3"),
    },
    "generation_mw": {
        "min": ("min_generation_mw", "min_generation_mw"),
        "max": ("max_generation_mw", None),
    },
    "turbined_m3s": {
        "min": ("min_turbined_m3s", "min_turbined_m3s"),
        "max": ("max_turbined_m3s", "max_turbined_m3s"),
    },
    "outflow_m3s": {
        "min": ("min_outflow_m3s", "min_outflow_m3s"),
        "max": ("max_outflow_m3s", None),
    },
}


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


def _plant_max_reldiff_table(
    results: list[ResultComparison],
    entity_type: str,
    variables: list[tuple[str, str]],
) -> str:
    """Per-plant max relative-difference summary table.

    Rows: plants (sorted by overall worst max-rel-diff across all
    variables, worst first). Columns: one per variable in *variables*
    (skipping variables for which no NEWAVE row exists).

    Cell value = ``max_stages |cobre − newave| / |newave| × 100``
    (NEWAVE-relative, per the report convention). Stages with
    ``|newave| ≈ 0`` are excluded — for any plant/variable that has no
    eligible stage the cell shows "—".

    Colour cues: ≤ 1 % green, ≤ 10 % amber, > 10 % red.
    """
    rows = [r for r in results if r.entity_type == entity_type]
    if not rows:
        return ""

    # (plant_key, variable) -> max(rel_diff) across stages.
    max_rd: dict[tuple[str, int, str], float] = {}
    for r in rows:
        if r.rel_diff is None:
            continue
        key = (r.entity_name, r.newave_code, r.variable)
        cur = max_rd.get(key)
        if cur is None or r.rel_diff > cur:
            max_rd[key] = r.rel_diff

    # Plant ordering: worst overall first (sum-of-max across variables
    # would be skewed by missing cells; use the per-row max for the
    # primary sort and fall back to plant name).
    plant_keys = sorted(
        {(name, code) for name, code, _ in max_rd},
        key=lambda k: (
            -max(
                (
                    max_rd[(k[0], k[1], v)]
                    for v, _ in variables
                    if (k[0], k[1], v) in max_rd
                ),
                default=0.0,
            ),
            k[0],
        ),
    )
    if not plant_keys:
        return ""

    def _cell(rd: float | None) -> str:
        if rd is None:
            return '<td class="cb-num">—</td>'
        pct = rd * 100.0
        if pct <= 1.0:
            cls = "cb-num cb-diff-neg"  # reuse green styling
        elif pct <= 10.0:
            cls = "cb-num"
        else:
            cls = "cb-num cb-diff-pos"  # reuse red styling
        return f'<td class="{cls}">{pct:.2f}%</td>'

    header_cells = '<th class="cb-cat">Plant</th>' + "".join(
        f'<th class="cb-num">{label}</th>' for _, label in variables
    )
    body_rows: list[str] = []
    for name, code in plant_keys:
        cells = [f'<td class="cb-cat">{escape_text(name)}</td>']
        for var_key, _ in variables:
            rd = max_rd.get((name, code, var_key))
            cells.append(_cell(rd))
        body_rows.append("<tr>" + "".join(cells) + "</tr>")

    # Footer: median max-rel-diff per variable across plants. The table
    # CSS bolds the last tbody row to call out a totals/summary line —
    # without this row the styling would land on whichever plant
    # happened to sort last, which is misleading.
    import statistics

    summary_cells = ['<td class="cb-cat">Median</td>']
    for var_key, _ in variables:
        col_values = [
            max_rd[(name, code, var_key)]
            for name, code in plant_keys
            if (name, code, var_key) in max_rd
        ]
        median = statistics.median(col_values) if col_values else None
        summary_cells.append(_cell(median))
    body_rows.append("<tr>" + "".join(summary_cells) + "</tr>")

    caption_label = "Hydro" if entity_type == "hydro" else "Thermal"
    return (
        '<table class="cost-breakdown-table">'
        f"<caption>{caption_label} per-plant max relative difference "
        "(|Cobre − NEWAVE| / |NEWAVE|, over stages)</caption>"
        f"<thead><tr>{header_cells}</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody>"
        "</table>"
    )


def build_hydro_detail_tab(
    results: list[ResultComparison],
    pct_df: pl.DataFrame | None = None,
    cobre_hydro: pl.DataFrame | None = None,
    cobre_hydro_meta: dict[int, dict] | None = None,
    cobre_hydro_per_stage_bounds: pl.DataFrame | None = None,
    nw_hydro_slacks: pl.DataFrame | None = None,
) -> str:
    """Build interactive per-plant hydro detail with JS dropdown.

    Comparison variables (NEWAVE + Cobre) are populated from ``results``.
    Cobre-only variables (EARM, ENA, plus the three operational slacks:
    withdrawal pos/neg and inflow non-negativity) are populated from
    ``cobre_hydro`` if provided — these display only the Cobre line and
    band.

    When ``cobre_hydro_meta`` is supplied, static reservoir / outflow /
    turbined / generation bounds are surfaced as dashed reference lines
    on the matching variable charts.  When
    ``cobre_hydro_per_stage_bounds`` is also supplied, any per-stage
    overrides from ``constraints/hydro_bounds.parquet`` replace the
    static value at the affected stages — matching what the LP
    actually saw.

    When ``nw_hydro_slacks`` is supplied, the NEWAVE
    ``VIOL_POS_VRETIRUH`` / ``VIOL_NEG_VRETIRUH`` series (converted to
    m³/s) are rendered as a NEWAVE line on the two withdrawal-slack
    panels alongside the existing Cobre Mean + p10/p90 band.  The
    inflow-non-negativity slack stays Cobre-only because NEWAVE has no
    direct counterpart.
    """
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

    # Build cobre_id -> {var: {stage: value}} for cobre-only variables.
    cobre_only_lookup: dict[int, dict[str, dict[int, float]]] = {}
    # Per-(cobre_id, stage_id) Cobre LP gen-max for the dashed overlay
    # trace on the generation_mw chart. The NEWAVE GHMAX_FPHC trace was
    # found to be unhelpful in practice and is intentionally not
    # surfaced — see report notes.
    gen_max_cb_lookup: dict[int, dict[int, float]] = {}
    cobre_only_vars = [v for v, _ in _HYDRO_COBRE_ONLY_VARIABLES]
    if cobre_hydro is not None and not cobre_hydro.is_empty():
        avail_vars = [v for v in cobre_only_vars if v in cobre_hydro.columns]
        has_cb_lp_max = "cobre_lp_gen_max_mw" in cobre_hydro.columns
        for row in cobre_hydro.iter_rows(named=True):
            eid = int(row["entity_id"])
            sid = int(row["stage_id"])
            entry = cobre_only_lookup.setdefault(eid, {})
            for v in avail_vars:
                val = row.get(v)
                if val is None:
                    continue
                entry.setdefault(v, {})[sid] = float(val)
            if has_cb_lp_max:
                val_cb = row.get("cobre_lp_gen_max_mw")
                if val_cb is not None:
                    gen_max_cb_lookup.setdefault(eid, {})[sid] = float(val_cb)

    # cobre_id -> bound_col -> {stage: value} for the per-stage overrides
    # supplied by ``hydro_bounds.parquet``.  Falls back to the empty dict
    # when the parquet is absent.
    per_stage_bounds_lookup: dict[int, dict[str, dict[int, float]]] = {}
    if (
        cobre_hydro_per_stage_bounds is not None
        and not cobre_hydro_per_stage_bounds.is_empty()
    ):
        for row in cobre_hydro_per_stage_bounds.iter_rows(named=True):
            eid = int(row["entity_id"])
            sid = int(row["stage_id"])
            entry_map = per_stage_bounds_lookup.setdefault(eid, {})
            for col, val in row.items():
                if col in ("entity_id", "stage_id") or val is None:
                    continue
                entry_map.setdefault(col, {})[sid] = float(val)

    static_meta = cobre_hydro_meta or {}

    # cobre_id -> {var: {stage_id: nw_value}} for the two withdrawal slacks.
    # Drives the NEWAVE line on the matching cobre-only chart panels.
    nw_slack_lookup: dict[int, dict[str, dict[int, float]]] = {}
    _NW_SLACK_VARS = (
        "water_withdrawal_violation_pos_m3s",
        "water_withdrawal_violation_neg_m3s",
    )
    if nw_hydro_slacks is not None and not nw_hydro_slacks.is_empty():
        avail_nw_slacks = [v for v in _NW_SLACK_VARS if v in nw_hydro_slacks.columns]
        for row in nw_hydro_slacks.iter_rows(named=True):
            eid = int(row["entity_id"])
            sid = int(row["stage_id"])
            entry_map = nw_slack_lookup.setdefault(eid, {})
            for v in avail_nw_slacks:
                val = row.get(v)
                if val is None:
                    continue
                entry_map.setdefault(v, {})[sid] = float(val)

    all_vars = _HYDRO_VARIABLES + _HYDRO_COBRE_ONLY_VARIABLES

    js_plants: dict[str, dict] = {}
    for (name, nw_code), var_data in sorted(plants.items()):
        pid = f"{nw_code}_{name}"
        cid = cobre_ids.get((name, nw_code), -1)
        entry: dict = {
            "name": name,
            "code": nw_code,
            "cobre_id": cid,
        }
        for var_key, _var_label in _HYDRO_VARIABLES:
            stage_data = var_data.get(var_key, {})
            stages = sorted(stage_data.keys())
            entry[f"{var_key}_stages"] = stages
            entry[f"{var_key}_nw"] = [stage_data[s][0] for s in stages]
            entry[f"{var_key}_cb"] = [stage_data[s][1] for s in stages]
        cobre_only = cobre_only_lookup.get(cid, {})
        nw_slacks_for_plant = nw_slack_lookup.get(cid, {})
        for var_key, _var_label in _HYDRO_COBRE_ONLY_VARIABLES:
            stage_data_co = cobre_only.get(var_key, {})
            stages = sorted(stage_data_co.keys())
            entry[f"{var_key}_stages"] = stages
            nw_stage_map = nw_slacks_for_plant.get(var_key)
            if nw_stage_map:
                entry[f"{var_key}_nw"] = [
                    round(nw_stage_map.get(s, 0.0), 2) for s in stages
                ]
            else:
                entry[f"{var_key}_nw"] = []
            entry[f"{var_key}_cb"] = [round(stage_data_co[s], 2) for s in stages]
        # Cobre LP gen_max overlay (dashed trace). Aligned to the
        # generation_mw stage grid populated above.
        gen_stages = entry.get("generation_mw_stages", [])
        cb_max_map = gen_max_cb_lookup.get(cid, {})
        if cb_max_map:
            entry["generation_mw_max_cb"] = [
                round(cb_max_map.get(s, 0.0), 2) for s in gen_stages
            ]

        # Bound overlays per variable.  Static values come from
        # ``hydros.json`` via cobre_hydro_meta; per-stage rows in
        # hydro_bounds.parquet shadow the static value at the matching
        # stages.  When the bound is structurally absent (e.g.
        # max_outflow), we skip emitting the array so the JS layer
        # doesn't draw a constant-zero dashed line.
        meta = static_meta.get(cid, {})
        per_stage_overrides = per_stage_bounds_lookup.get(cid, {})
        for var_key, sides in _HYDRO_BOUND_OVERLAY.items():
            var_stages = entry.get(f"{var_key}_stages", [])
            if not var_stages:
                continue
            for side, (static_key, ps_key) in sides.items():
                static_val = meta.get(static_key) if static_key is not None else None
                ps_overrides = per_stage_overrides.get(ps_key, {}) if ps_key else {}
                if static_val is None and not ps_overrides:
                    continue
                series = []
                any_value = False
                for s in var_stages:
                    v = ps_overrides.get(s)
                    if v is None and static_val is not None:
                        v = static_val
                    if v is None:
                        series.append(None)
                    else:
                        series.append(round(float(v), 4))
                        any_value = True
                if any_value:
                    entry[f"{var_key}_bound_{side}"] = series

        js_plants[pid] = entry

    _enrich_with_percentiles(js_plants, all_vars, pct_df)

    summary_table = _plant_max_reldiff_table(results, "hydro", _HYDRO_VARIABLES)
    detail_html = _build_interactive_detail_html(
        js_plants,
        all_vars,
        "hydro",
        "Hydro Plant",
    )
    if summary_table:
        summary_table = f'<div style="margin-bottom:32px">{summary_table}</div>'
    return summary_table + detail_html


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

    summary_table = _plant_max_reldiff_table(results, "thermal", thermal_vars)
    detail_html = _build_interactive_detail_html(
        js_plants,
        thermal_vars,
        "thermal",
        "Thermal Plant",
    )
    if summary_table:
        summary_table = f'<div style="margin-bottom:32px">{summary_table}</div>'
    return summary_table + detail_html


def _build_interactive_detail_html(
    js_plants: dict[str, dict],
    variables: list[tuple[str, str]],
    prefix: str,
    label: str,
) -> str:
    """Build the HTML/JS for interactive per-plant detail charts."""
    data_json = json_for_script(js_plants)

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
                    showlegend: true,
                    // Skip hover on the wrap-around polygon \u2014 its y values
                    // are the reversed-p10 closing edge and have no
                    // meaning at the cursor x.  Real values come from
                    // the visible p10 / p90 line traces below.
                    hoverinfo: 'skip'
                }});
                traces.push({{
                    x: s, y: p10,
                    name: 'Cobre P10', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
                traces.push({{
                    x: s, y: p90,
                    name: 'Cobre P90', type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1, dash: 'dot'}},
                    legendgroup: 'band', showlegend: false
                }});
            }}
            if (nw && nw.length > 0) {{
                traces.push({{x: s, y: nw, name: 'NEWAVE', type: 'scatter',
                    mode: 'lines', line: {{color: '{COLOR_NEWAVE}', width: 2}}}});
            }}
            traces.push({{x: s, y: cb, name: 'Cobre Mean', type: 'scatter',
                mode: 'lines', line: {{color: '{COLOR_COBRE}', width: 2}}}});
            var maxCb = d['{var_key}_max_cb'];
            if (maxCb && maxCb.length > 0) {{
                traces.push({{x: s, y: maxCb, name: 'Cobre LP gen_max',
                    type: 'scatter', mode: 'lines',
                    line: {{color: '{COLOR_COBRE}', width: 1.5, dash: 'dash'}}}});
            }}
            // Bound overlays (static from hydros.json, with per-stage
            // overrides from hydro_bounds.parquet shadowing where present).
            // Rendered in a muted grey so they sit behind the comparison
            // traces without competing for attention.  ``connectgaps:false``
            // ensures stages where the bound is structurally undefined
            // produce a gap in the dashed line.
            var bMin = d['{var_key}_bound_min'];
            if (bMin && bMin.length > 0) {{
                traces.push({{x: s, y: bMin, name: 'Lower bound',
                    type: 'scatter', mode: 'lines', connectgaps: false,
                    line: {{color: '#888', width: 1.2, dash: 'dash'}}}});
            }}
            var bMax = d['{var_key}_bound_max'];
            if (bMax && bMax.length > 0) {{
                traces.push({{x: s, y: bMax, name: 'Upper bound',
                    type: 'scatter', mode: 'lines', connectgaps: false,
                    line: {{color: '#888', width: 1.2, dash: 'dash'}}}});
            }}
            Plotly.react('{div_id}', traces, {{
                title: d.name + ' \u2014 {var_label}',
                xaxis: {{title: 'Stage'}},
                yaxis: {{title: '{var_label}'}},
                legend: {json_for_script(_LEGEND)},
                margin: {json_for_script(_MARGIN)},
                template: 'plotly_white',
                hovermode: 'x unified',
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


# -------------------------------------------------------------------
# Generic constraints (RE, AGRINT, VminOP) — NEWAVE vs Cobre LHS
# -------------------------------------------------------------------


def constraints_comparison_chart(
    constraints: list[dict],
    lhs_newave: pl.DataFrame,
    lhs_cobre: pl.DataFrame,
    bound_by_constraint: dict[int, dict[int, float]],
) -> str:
    """Per-constraint small-multiples comparing NEWAVE vs Cobre LHS vs bound.

    One panel per constraint. Each panel shows the per-stage NEWAVE LHS
    (mean evaluated from MEDIAS-USIH / int*.out outputs), the Cobre LHS
    (mean across scenarios and blocks from simulation parquet), and the
    constraint bound (dashed) overlaid as a horizontal-step series for
    every stage where the bound is defined. Constraints with no LHS
    data on either side are skipped silently.

    Parameters
    ----------
    constraints:
        Constraint dicts loaded from ``generic_constraints.json``.
    lhs_newave, lhs_cobre:
        DataFrames with columns ``constraint_id``, ``stage_id``,
        ``lhs_value``.
    bound_by_constraint:
        Output of
        :func:`cobre_bridge.comparators.constraints_compare.per_stage_bounds`
        — maps ``constraint_id`` to ``{stage_id: bound}``.

    Returns
    -------
    str
        HTML ``<div>`` with an embedded Plotly figure.  Returns a short
        fallback ``<p>`` when no constraints have any data.
    """
    if not constraints:
        return "<p>No generic constraints defined.</p>"

    # Index by id for fast lookup.
    nw_by_cid: dict[int, dict[int, float]] = {}
    cb_by_cid: dict[int, dict[int, float]] = {}
    if not lhs_newave.is_empty():
        for r in lhs_newave.iter_rows(named=True):
            nw_by_cid.setdefault(int(r["constraint_id"]), {})[int(r["stage_id"])] = (
                float(r["lhs_value"])
            )
    if not lhs_cobre.is_empty():
        for r in lhs_cobre.iter_rows(named=True):
            cb_by_cid.setdefault(int(r["constraint_id"]), {})[int(r["stage_id"])] = (
                float(r["lhs_value"])
            )

    # Keep only constraints that have at least one data point on either
    # side or at least one bound entry. A constraint with no LHS data
    # anywhere has nothing to show.
    renderable: list[dict] = []
    for c in constraints:
        cid = int(c["id"])
        if cid in nw_by_cid or cid in cb_by_cid or bound_by_constraint.get(cid):
            renderable.append(c)
    if not renderable:
        return "<p>No constraint data available to compare.</p>"

    ncols = 2
    nrows = (len(renderable) + ncols - 1) // ncols
    row_gap = 0.06
    row_h = max((1.0 - row_gap * (nrows - 1)) / nrows, 0.001)
    col_gap = 0.05
    col_w = (1.0 - col_gap * (ncols - 1)) / ncols

    traces: list[dict] = []
    layout: dict = {"title": "Generic Constraints — LHS vs Bound"}
    first = True

    for idx, c in enumerate(renderable):
        cid = int(c["id"])
        name = c["name"]
        sense = c.get("sense", "<=")
        row_i = idx // ncols
        col_i = idx % ncols
        ax_idx = idx + 1
        xa = f"x{ax_idx}" if ax_idx > 1 else "x"
        ya = f"y{ax_idx}" if ax_idx > 1 else "y"

        x0 = col_i * (col_w + col_gap)
        x1 = x0 + col_w
        y1 = 1.0 - row_i * (row_h + row_gap)
        y0 = y1 - row_h

        xa_key = f"xaxis{ax_idx}" if ax_idx > 1 else "xaxis"
        ya_key = f"yaxis{ax_idx}" if ax_idx > 1 else "yaxis"
        layout[xa_key] = {
            "domain": [round(x0, 3), round(x1, 3)],
            "title": "Stage" if row_i == nrows - 1 else "",
            "anchor": ya,
        }
        layout[ya_key] = {
            "domain": [round(y0, 3), round(y1, 3)],
            "title": f"{name} ({sense})",
            "anchor": xa,
        }

        # Union of stages with any data (NEWAVE LHS, Cobre LHS, or bound).
        stages = sorted(
            set(nw_by_cid.get(cid, {}).keys())
            | set(cb_by_cid.get(cid, {}).keys())
            | set(bound_by_constraint.get(cid, {}).keys())
        )
        if not stages:
            continue

        nw_y = [nw_by_cid.get(cid, {}).get(s) for s in stages]
        cb_y = [cb_by_cid.get(cid, {}).get(s) for s in stages]
        bound_y = [bound_by_constraint.get(cid, {}).get(s) for s in stages]

        # NEWAVE LHS line (only where defined).
        nw_x_present = [s for s, v in zip(stages, nw_y) if v is not None]
        nw_v_present = [v for v in nw_y if v is not None]
        if nw_x_present:
            traces.append(
                {
                    "x": nw_x_present,
                    "y": nw_v_present,
                    "name": "NEWAVE LHS",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": COLOR_NEWAVE, "width": 2},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "nw",
                    "showlegend": first,
                }
            )

        # Cobre LHS line.
        cb_x_present = [s for s, v in zip(stages, cb_y) if v is not None]
        cb_v_present = [v for v in cb_y if v is not None]
        if cb_x_present:
            traces.append(
                {
                    "x": cb_x_present,
                    "y": cb_v_present,
                    "name": "Cobre LHS (mean)",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": COLOR_COBRE, "width": 2},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "cb",
                    "showlegend": first,
                }
            )

        # Bound line (dashed red, only where defined). Use markers so
        # gaps where the bound is None remain visually obvious.
        bd_x_present = [s for s, v in zip(stages, bound_y) if v is not None]
        bd_v_present = [v for v in bound_y if v is not None]
        if bd_x_present:
            traces.append(
                {
                    "x": bd_x_present,
                    "y": bd_v_present,
                    "name": "Bound",
                    "type": "scatter",
                    "mode": "lines",
                    "line": {"color": "#DC4C4C", "width": 1.5, "dash": "dash"},
                    "xaxis": xa,
                    "yaxis": ya,
                    "legendgroup": "bound",
                    "showlegend": first,
                }
            )

        first = False

    return _plotly_div(traces, layout, height=max(nrows * 300 + 80, 360))
