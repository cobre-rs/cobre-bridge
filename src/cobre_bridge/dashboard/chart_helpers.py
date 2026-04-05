"""Shared chart builder helpers for the Cobre dashboard.

Provides reusable functions for building Plotly traces that follow the v2
dashboard convention: every line chart shows a mean solid line, a p50/median
dashed line, and a p10-p90 shaded band.  Bounds overlays and HTML card
wrapping are also provided here so that all tab modules share a single,
consistent implementation.

Also provides NPV cost computation and cost category grouping helpers used
by the Overview and Costs tabs.
"""

from __future__ import annotations

import plotly.graph_objects as go

from cobre_bridge.ui.html import wrap_chart
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS as _LEGEND,
)
from cobre_bridge.ui.plotly_helpers import (
    MARGIN_DEFAULTS as _MARGIN,
)
from cobre_bridge.ui.plotly_helpers import (
    fig_to_html,
)

try:
    import pandas as pd
except ImportError as _pd_err:  # pragma: no cover
    raise ImportError("pandas is required for chart_helpers") from _pd_err

try:
    import polars as pl
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

_BOUNDS_COLOR: str = "#6B7280"

#: Mapping from logical cost group name to known Cobre cost component columns.
#: The ``"Other"`` key is intentionally absent — it is computed dynamically in
#: :func:`group_costs` as all columns not claimed by the explicit groups.
COST_GROUPS: dict[str, list[str]] = {
    "Thermal": [
        "thermal_generation_cost",
        "thermal_startup_cost",
        "thermal_minimum_cost",
    ],
    "Deficit": [
        "deficit_cost_depth_1",
        "deficit_cost_depth_2",
        "deficit_cost_depth_3",
    ],
    "Spillage": ["spillage_cost"],
    "NCS": ["ncs_curtailment_cost", "ncs_generation_cost"],
    "Violations": ["generic_constraint_violation_cost"],
    "Other": [],  # filled dynamically by group_costs
}

COST_GROUP_COLORS: dict[str, str] = {
    "Thermal": "#F59E0B",
    "Deficit": "#EF4444",
    "Spillage": "#3B82F6",
    "NCS": "#10B981",
    "Violations": "#8B5CF6",
    "Other": "#6B7280",
}


def compute_percentiles(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    percentiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Group *df* and compute mean plus named percentile columns.

    Accepts either a :class:`pandas.DataFrame` or a :class:`polars.DataFrame`.
    Polars input is converted to pandas internally before aggregation.

    Args:
        df: Source data.  Must contain all columns in *group_cols* and
            *value_col*.
        group_cols: Column names to group by (e.g. ``["stage_id"]``).
        value_col: Name of the numeric column to aggregate.
        percentiles: Quantile levels.  Each value ``p`` produces a column
            named ``p{int(p * 100)}``, e.g. ``0.1`` -> ``"p10"``.

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``[*group_cols, "mean", "p10", "p50", "p90"]`` (or whichever
        percentile names correspond to *percentiles*).  One row per group.
        Returns an empty :class:`pandas.DataFrame` with the expected columns
        when *df* is empty.
    """
    # Convert polars to pandas if necessary
    if pl is not None and isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    pct_col_names = [f"p{int(p * 100)}" for p in percentiles]
    expected_cols = group_cols + ["mean"] + pct_col_names

    if df.empty:
        return pd.DataFrame(columns=expected_cols)

    grouped = df.groupby(group_cols, sort=True)[value_col]
    result = grouped.mean().rename("mean").to_frame()
    for p, col_name in zip(percentiles, pct_col_names):
        result[col_name] = grouped.quantile(p)
    return result.reset_index()


def add_mean_p50_band(
    fig: go.Figure,
    df: pd.DataFrame,
    x_col: str,
    name: str,
    color: str,
    row: int | None = None,
    col: int | None = None,
    show_band: bool = True,
    show_p50: bool = True,
) -> go.Figure:
    """Add mean, p50, and p10-p90 band traces to *fig*.

    The function adds up to three :class:`plotly.graph_objects.Scatter` traces:

    * **Mean** — solid line, width 2, full opacity.
    * **P50** — dashed line, width 1.5, 70% opacity (omitted when
      *show_p50* is ``False``).
    * **P10-P90 band** — two traces required by Plotly's ``fill="tonexty"``
      convention: the lower bound (p10) is plotted first as an invisible line,
      then the upper bound (p90) is plotted with ``fill="tonexty"`` at 15%
      opacity (omitted when *show_band* is ``False``).

    The function is a no-op when *df* is empty.

    Args:
        fig: The :class:`plotly.graph_objects.Figure` to mutate.
        df: Pre-computed percentile DataFrame as returned by
            :func:`compute_percentiles`.  Must contain ``x_col``, ``"mean"``,
            ``"p10"``, and ``"p90"`` columns (and ``"p50"`` when
            *show_p50* is ``True``).
        x_col: Name of the column used as the x-axis values.
        name: Display name for the trace group (used in the legend).
        color: Hex or CSS colour string for the traces.
        row: Subplot row (1-based) for :meth:`~plotly.graph_objects.Figure.add_trace`.
        col: Subplot column (1-based).
        show_band: When ``False``, the p10-p90 filled area is omitted.
        show_p50: When ``False``, the p50 dashed line is omitted.

    Returns:
        The same *fig* object (enables method chaining).
    """
    if df.empty:
        return fig

    subplot_kwargs: dict = {}
    if row is not None:
        subplot_kwargs["row"] = row
    if col is not None:
        subplot_kwargs["col"] = col

    x = df[x_col]

    # Mean (solid line)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df["mean"],
            name=name,
            legendgroup=name,
            mode="lines",
            line=dict(color=color, width=2),
        ),
        **subplot_kwargs,
    )

    # P50 (dashed line)
    if show_p50:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["p50"],
                name=f"{name} P50",
                legendgroup=name,
                showlegend=False,
                mode="lines",
                line=dict(color=color, width=1.5, dash="dash"),
                opacity=0.7,
            ),
            **subplot_kwargs,
        )

    # P10-P90 band
    if show_band:
        # Lower bound (invisible, reference for fill)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["p10"],
                name=f"{name} P10",
                legendgroup=name,
                showlegend=False,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
            ),
            **subplot_kwargs,
        )
        # Upper bound (p90) — fills down to the p10 trace
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["p90"],
                name=f"{name} Band",
                legendgroup=name,
                showlegend=False,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=_hex_to_rgba(color, 0.15),
                hoverinfo="skip",
            ),
            **subplot_kwargs,
        )

    return fig


def add_bounds_overlay(
    fig: go.Figure,
    bounds_df: pd.DataFrame,
    x_col: str,
    min_col: str | None = None,
    max_col: str | None = None,
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Overlay dashed grey reference lines for min/max bounds on *fig*.

    Either *min_col* or *max_col* (or both) may be specified.  When both are
    ``None`` the function is a no-op.

    Args:
        fig: The :class:`plotly.graph_objects.Figure` to mutate.
        bounds_df: DataFrame with one row per x value containing the bound
            columns.  Must include ``x_col`` and whichever of *min_col* /
            *max_col* are not ``None``.
        x_col: Name of the x-axis column in *bounds_df*.
        min_col: Column name for the lower bound; omitted when ``None``.
        max_col: Column name for the upper bound; omitted when ``None``.
        row: Subplot row (1-based).
        col: Subplot column (1-based).

    Returns:
        The same *fig* object (enables method chaining).
    """
    if min_col is None and max_col is None:
        return fig

    subplot_kwargs: dict = {}
    if row is not None:
        subplot_kwargs["row"] = row
    if col is not None:
        subplot_kwargs["col"] = col

    x = bounds_df[x_col]
    bound_line = dict(color=_BOUNDS_COLOR, width=1.5, dash="dash")

    if min_col is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bounds_df[min_col],
                name=f"Min ({min_col})",
                mode="lines",
                line=bound_line,
                showlegend=True,
            ),
            **subplot_kwargs,
        )

    if max_col is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bounds_df[max_col],
                name=f"Max ({max_col})",
                mode="lines",
                line=bound_line,
                showlegend=True,
            ),
            **subplot_kwargs,
        )

    return fig


def make_chart_card(
    fig: go.Figure | None,
    title: str,
    chart_id: str,
    height: int = 380,
) -> str:
    """Wrap a Plotly figure in a standard ``.chart-card`` HTML fragment.

    Applies default layout (template, margins, legend position) from
    :func:`~cobre_bridge.ui.plotly_helpers.fig_to_html`, sets the figure
    height, and embeds the result inside the ``.chart-card`` div with an
    expand button (via :func:`~cobre_bridge.ui.html.wrap_chart`).

    The output does **not** include a ``<script src="plotly.js">`` tag —
    callers are responsible for including Plotly exactly once in the outer
    HTML shell.

    Args:
        fig: The Plotly figure to embed.  Raises :class:`ValueError` when
            ``None``.
        title: Chart title applied to *fig* via
            :meth:`~plotly.graph_objects.Figure.update_layout`.
        chart_id: HTML ``id`` attribute placed on the inner chart container
            div for JavaScript targeting.
        height: Figure height in pixels (default 380).

    Returns:
        An HTML string: ``<div class="chart-card">...</div>``.

    Raises:
        ValueError: When *fig* is ``None``.
    """
    if fig is None:
        raise ValueError("make_chart_card: fig must not be None")

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.02, xanchor="left"),
        height=height,
        margin=_MARGIN,
        legend=_LEGEND,
        template="plotly_white",
    )

    inner_html = f'<div id="{chart_id}">{fig_to_html(fig)}</div>'
    return wrap_chart(inner_html)


# ---------------------------------------------------------------------------
# NPV and cost grouping helpers
# ---------------------------------------------------------------------------

#: Non-cost metadata columns that are never treated as cost components.
_NON_COST_COLS: frozenset[str] = frozenset({"scenario_id", "stage_id", "block_id"})


def compute_npv_costs(
    costs_df: pd.DataFrame,
    discount_rate: float,
    stage_start: int = 0,
) -> pd.DataFrame:
    """Apply per-stage NPV discount factors to cost component columns.

    The discount factor for stage *t* is ``1 / (1 + discount_rate) ^ (t -
    stage_start)``.  Stage ``stage_start`` (default 0) therefore has a factor
    of 1.0 — i.e. costs at the reference stage are undiscounted.

    Columns named ``"scenario_id"``, ``"stage_id"``, and ``"block_id"`` are
    treated as metadata and are never discounted.

    Args:
        costs_df: DataFrame containing a ``stage_id`` column (0-based) plus
            one or more numeric cost component columns.
        discount_rate: Annual (or per-stage) discount rate, e.g. ``0.10`` for
            10%.  A value of ``0.0`` leaves all values unchanged.
        stage_start: Reference stage index.  Defaults to ``0`` (first-stage
            perspective).

    Returns:
        A new :class:`pandas.DataFrame` with discounted cost values.  Input is
        never mutated.  Returns a copy of *costs_df* when it is empty.
    """
    if costs_df.empty:
        return costs_df.copy()

    result = costs_df.copy()
    cost_cols = [c for c in result.columns if c not in _NON_COST_COLS]

    if discount_rate > 0.0 and cost_cols:
        factors: pd.Series = 1.0 / (1.0 + discount_rate) ** (
            result["stage_id"] - stage_start
        )
        for col in cost_cols:
            result[col] = result[col] * factors

    return result


def group_costs(
    costs_df: pd.DataFrame,
    cost_columns: list[str],
) -> pd.DataFrame:
    """Aggregate individual cost component columns into logical groups.

    Groups are defined by :data:`COST_GROUPS`.  Any column present in
    *cost_columns* that is not claimed by a named group is placed in an
    ``"Other"`` group.  Missing columns (present in :data:`COST_GROUPS` but
    absent from *cost_columns*) are silently skipped.

    Args:
        costs_df: DataFrame containing the individual cost component columns
            listed in *cost_columns*.
        cost_columns: Complete list of cost column names in *costs_df*.  Used
            to determine which columns fall into ``"Other"``.

    Returns:
        A new :class:`pandas.DataFrame` where the individual component columns
        are replaced by one column per group.  Non-cost columns (``scenario_id``,
        ``stage_id``, ``block_id``) are preserved unchanged.  Input is never
        mutated.  Returns a copy of *costs_df* when it is empty.
    """
    if costs_df.empty:
        return costs_df.copy()

    result = costs_df.copy()
    assigned: set[str] = set()

    for group_name, components in COST_GROUPS.items():
        if group_name == "Other":
            continue
        present = [c for c in components if c in cost_columns]
        assigned.update(present)
        result[group_name] = result[present].sum(axis=1) if present else 0.0

    # Columns not claimed by any named group -> "Other"
    other_cols = [c for c in cost_columns if c not in assigned]
    result["Other"] = result[other_cols].sum(axis=1) if other_cols else 0.0

    # Drop the original component columns
    cols_to_drop = [c for c in cost_columns if c in result.columns]
    result = result.drop(columns=cols_to_drop)

    return result


def compute_cost_summary(
    costs_df: pd.DataFrame,
    discount_rate: float,
) -> pd.DataFrame:
    """Compute a grouped NPV cost summary with statistics across scenarios.

    Chains three operations:

    1. Apply NPV discount factors via :func:`compute_npv_costs`.
    2. Sum discounted costs across all stages per scenario.
    3. Group columns via :func:`group_costs`.
    4. Aggregate across scenarios: mean, std, p10, p90.
    5. Add a ``pct`` column (percentage of total mean cost).

    Args:
        costs_df: DataFrame with ``scenario_id``, ``stage_id``, and cost
            component columns.
        discount_rate: Per-stage discount rate (e.g. ``0.12`` for 12%).

    Returns:
        A :class:`pandas.DataFrame` with columns
        ``["group", "mean", "std", "p10", "p90", "pct"]``, one row per cost
        group, sorted descending by ``mean``.  Returns an empty DataFrame with
        those columns when *costs_df* is empty.
    """
    summary_cols = ["group", "mean", "std", "p10", "p90", "pct"]

    if costs_df.empty:
        return pd.DataFrame(columns=summary_cols)

    cost_columns = [c for c in costs_df.columns if c not in _NON_COST_COLS]

    # 1. Discount
    discounted = compute_npv_costs(costs_df, discount_rate)

    # 2. Sum across stages per scenario
    meta_cols_present = [c for c in ("scenario_id",) if c in discounted.columns]
    if meta_cols_present:
        per_scenario = discounted.groupby(meta_cols_present)[cost_columns].sum()
    else:
        per_scenario = discounted[cost_columns]

    # 3. Group columns
    grouped = group_costs(per_scenario.reset_index(), cost_columns)
    group_cols = [
        c for c in grouped.columns if c not in _NON_COST_COLS and c != "scenario_id"
    ]

    # 4. Aggregate across scenarios
    agg = grouped[group_cols].agg(
        ["mean", "std", lambda q: q.quantile(0.1), lambda q: q.quantile(0.9)]
    )
    agg.index = ["mean", "std", "p10", "p90"]  # type: ignore[assignment]
    agg = agg.T.reset_index().rename(columns={"index": "group"})

    # 5. Percentage of total mean
    total_mean = agg["mean"].sum()
    agg["pct"] = (agg["mean"] / total_mean * 100.0) if total_mean != 0.0 else 0.0

    # Sort descending by mean and return only the expected columns
    agg = agg.sort_values("mean", ascending=False).reset_index(drop=True)
    return agg[summary_cols]


# ---------------------------------------------------------------------------
# Private utilities
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour string to an ``rgba(...)`` CSS value.

    Args:
        hex_color: A hex colour string such as ``"#3B82F6"`` (the leading
            ``#`` is required).
        alpha: Opacity in the range 0.0–1.0.

    Returns:
        An ``rgba(r, g, b, alpha)`` string suitable for Plotly's
        ``fillcolor`` parameter.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
