"""v2 Stochastic Model tab — inflow comparison, correlation, noise, and AR sections.

Displays six collapsible sections:
  A. System-Wide Inflow — hero chart of historical vs synthetic mean+std band.
  B. Inflow by Bus — 2x2 faceted subplots for the first 4 non-fictitious buses.
  C. Per-Hydro Explorer — dropdown selector showing one hydro at a time.
  D. Spatial Correlation — fitted vs empirical correlation heatmaps.
  E. Noise Diagnostics — noise histogram and stage-grouped box plot.
  F. AR Model Summary — AR order distribution and reduction reasons.

Only rendered when stochastic output is available (data.stochastic_available).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as ps

from cobre_bridge.dashboard.chart_helpers import make_chart_card
from cobre_bridge.ui.html import chart_grid, collapsible_section
from cobre_bridge.ui.plotly_helpers import (
    LEGEND_DEFAULTS,
    MARGIN_DEFAULTS,
    stage_x_labels,
)
from cobre_bridge.ui.theme import COLORS

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

TAB_ID = "tab-v2-stochastic"
TAB_LABEL = "Stochastic Model"
TAB_ORDER = 20

_HIST_COLOR: str = COLORS["hydro"]  # #4A90B8 — historical traces
_SYNTH_COLOR: str = COLORS["thermal"]  # #F5A623 — synthetic traces

_NO_HIST = "<p>No historical inflow data.</p>"
_NO_DATA = "<p>No data.</p>"


# ---------------------------------------------------------------------------
# Helper: convert hex color to rgba string
# ---------------------------------------------------------------------------


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex colour string to an ``rgba(...)`` CSS value."""
    h = hex_color.lstrip("#")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _compute_historical_stats(data: DashboardData) -> pd.DataFrame:
    """Compute per-hydro per-stage mean and std from inflow_history.

    Maps each ``date`` in ``inflow_history`` to a stage_id using the stage
    start_date ranges from ``stages_data["stages"]``, then groups by
    ``(hydro_id, stage_id)`` to compute mean and std.

    Args:
        data: Full DashboardData instance.

    Returns:
        DataFrame with columns ``hydro_id``, ``stage_id``, ``mean_m3s``,
        ``std_m3s``.  Returns an empty DataFrame with those columns when
        ``inflow_history`` is empty or ``stages_data`` has no stages.
    """
    empty = pd.DataFrame(columns=["hydro_id", "stage_id", "mean_m3s", "std_m3s"])

    hist = data.inflow_history
    if hist.empty:
        return empty

    stages = data.stages_data.get("stages", [])
    if not stages:
        return empty

    # Sort stages by start_date
    stage_rows = sorted(
        [
            {"id": s["id"], "start_date": pd.to_datetime(s["start_date"])}
            for s in stages
            if "start_date" in s
        ],
        key=lambda x: x["start_date"],  # type: ignore[return-value]
    )
    if not stage_rows:
        return empty

    stage_ids = [r["id"] for r in stage_rows]
    stage_starts = [r["start_date"] for r in stage_rows]

    # Ensure date column is datetime
    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Assign stage_id via searchsorted: finds the rightmost position where
    # stage_starts[pos-1] <= date, so date belongs to stage at pos-1.
    starts_arr = np.array([ts.value for ts in stage_starts], dtype=np.int64)
    dates_arr = df["date"].values.astype("datetime64[ns]").astype(np.int64)
    idx = np.searchsorted(starts_arr, dates_arr, side="right") - 1
    # Clamp negatives (dates before first stage) to 0
    idx = np.clip(idx, 0, len(stage_ids) - 1)
    df["stage_id"] = [stage_ids[i] for i in idx]

    grp = df.groupby(["hydro_id", "stage_id"])["value_m3s"]
    result = grp.mean().rename("mean_m3s").to_frame()
    result["std_m3s"] = grp.std()
    result = result.reset_index()
    return result[["hydro_id", "stage_id", "mean_m3s", "std_m3s"]]


def _aggregate_by_bus(
    stats_df: pd.DataFrame,
    hydro_bus_map: dict[int, int],
) -> dict[int, pd.DataFrame]:
    """Aggregate per-hydro stats into per-bus stats.

    For each bus: mean_m3s is the sum of all hydro means; std_m3s is the
    root-sum-of-squares of all hydro stds (treating NaN as 0).

    Args:
        stats_df: DataFrame with columns ``hydro_id``, ``stage_id``,
            ``mean_m3s``, ``std_m3s``.
        hydro_bus_map: Mapping ``hydro_id -> bus_id``.

    Returns:
        Dict mapping ``bus_id`` to a DataFrame with columns ``stage_id``,
        ``mean_m3s``, ``std_m3s``.  Returns empty dict when *stats_df* is empty
        or *hydro_bus_map* is empty.
    """
    if stats_df.empty or not hydro_bus_map:
        return {}

    df = stats_df.copy()
    df["bus_id"] = df["hydro_id"].map(hydro_bus_map)
    df = df.dropna(subset=["bus_id"])
    if df.empty:
        return {}
    df["bus_id"] = df["bus_id"].astype(int)
    df["std_sq"] = df["std_m3s"].fillna(0.0) ** 2

    bus_ids = sorted(df["bus_id"].unique())
    result: dict[int, pd.DataFrame] = {}
    for bus_id in bus_ids:
        sub = df[df["bus_id"] == bus_id]
        grp = sub.groupby("stage_id")
        mean_ser = grp["mean_m3s"].sum().rename("mean_m3s")
        std_ser = grp["std_sq"].apply(np.nansum).apply(np.sqrt).rename("std_m3s")
        bus_df = mean_ser.to_frame().join(std_ser).reset_index()
        result[bus_id] = bus_df[["stage_id", "mean_m3s", "std_m3s"]]
    return result


def _aggregate_system(stats_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-hydro stats to a single system-wide series.

    Sum of means and root-sum-of-squares of stds per stage.

    Args:
        stats_df: DataFrame with columns ``hydro_id``, ``stage_id``,
            ``mean_m3s``, ``std_m3s``.

    Returns:
        DataFrame with columns ``stage_id``, ``mean_m3s``, ``std_m3s``.
        Returns an empty DataFrame with those columns when *stats_df* is empty.
    """
    empty = pd.DataFrame(columns=["stage_id", "mean_m3s", "std_m3s"])
    if stats_df.empty:
        return empty

    df = stats_df.copy()
    df["std_sq"] = df["std_m3s"].fillna(0.0) ** 2
    grp = df.groupby("stage_id")
    mean_ser = grp["mean_m3s"].sum().rename("mean_m3s")
    std_ser = grp["std_sq"].apply(np.nansum).apply(np.sqrt).rename("std_m3s")
    result = mean_ser.to_frame().join(std_ser).reset_index()
    return result[["stage_id", "mean_m3s", "std_m3s"]]


# ---------------------------------------------------------------------------
# Band trace helpers
# ---------------------------------------------------------------------------


def _add_mean_std_band(
    fig: go.Figure,
    stage_ids: list[int],
    mean_vals: list[float],
    std_vals: list[float],
    x_labels: list[str],
    name: str,
    color: str,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add mean line + mean±std shaded band traces to *fig*.

    Three traces are added:
    1. Lower bound ``mean - std`` (invisible, reference for fill).
    2. Upper bound ``mean + std`` filled to the lower bound.
    3. Mean as a solid line (plotted last so it sits on top of the band).

    Args:
        fig: Figure to mutate.
        stage_ids: Ordered stage IDs.
        mean_vals: Mean values aligned with *stage_ids*.
        std_vals: Std values aligned with *stage_ids*.
        x_labels: Human-readable labels aligned with *stage_ids*.
        name: Legend group name.
        color: Hex colour string.
        row: Subplot row (1-based), or ``None`` for non-subplot figures.
        col: Subplot column (1-based), or ``None`` for non-subplot figures.
    """
    subplot_kw: dict = {}
    if row is not None:
        subplot_kw["row"] = row
    if col is not None:
        subplot_kw["col"] = col

    lower = [m - s for m, s in zip(mean_vals, std_vals)]
    upper = [m + s for m, s in zip(mean_vals, std_vals)]
    fill_color = _hex_to_rgba(color, 0.18)

    # Lower bound (invisible anchor for fill="tonexty")
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=lower,
            name=f"{name} −σ",
            legendgroup=name,
            showlegend=False,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
        ),
        **subplot_kw,
    )
    # Upper bound with fill back to lower
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=upper,
            name=f"{name} ±σ band",
            legendgroup=name,
            showlegend=False,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=fill_color,
            hoverinfo="skip",
        ),
        **subplot_kw,
    )
    # Mean solid line
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=mean_vals,
            name=name,
            legendgroup=name,
            mode="lines",
            line={"color": color, "width": 2},
            hovertemplate=(
                f"{name}<br>Stage: %{{x}}<br>Mean: %{{y:,.0f}} m³/s<extra></extra>"
            ),
        ),
        **subplot_kw,
    )


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _chart_system_inflow(
    hist_system: pd.DataFrame,
    synth_system: pd.DataFrame,
    stage_labels: dict[int, str],
) -> go.Figure:
    """Build the system-wide historical vs synthetic inflow figure.

    Args:
        hist_system: DataFrame with columns ``stage_id``, ``mean_m3s``,
            ``std_m3s`` (historical aggregated across all hydros).
        synth_system: DataFrame with columns ``stage_id``, ``mean_m3s``,
            ``std_m3s`` (synthetic from inflow_stats_stoch).
        stage_labels: Mapping ``stage_id -> label string``.

    Returns:
        A :class:`plotly.graph_objects.Figure`.
    """
    fig = go.Figure()

    if not hist_system.empty:
        h = hist_system.sort_values("stage_id")
        x_labels = stage_x_labels(h["stage_id"].tolist(), stage_labels)
        _add_mean_std_band(
            fig,
            h["stage_id"].tolist(),
            h["mean_m3s"].tolist(),
            h["std_m3s"].fillna(0.0).tolist(),
            x_labels,
            "Historical",
            _HIST_COLOR,
        )

    if not synth_system.empty:
        s = synth_system.sort_values("stage_id")
        x_labels = stage_x_labels(s["stage_id"].tolist(), stage_labels)
        _add_mean_std_band(
            fig,
            s["stage_id"].tolist(),
            s["mean_m3s"].tolist(),
            s["std_m3s"].fillna(0.0).tolist(),
            x_labels,
            "Synthetic",
            _SYNTH_COLOR,
        )

    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Total Inflow (m³/s)",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


def _chart_bus_facet(
    hist_by_bus: dict[int, pd.DataFrame],
    synth_by_bus: dict[int, pd.DataFrame],
    bus_names: dict[int, str],
    stage_labels: dict[int, str],
    bus_ids: list[int],
) -> go.Figure:
    """Build a 2x2 faceted subplot for up to 4 buses.

    Args:
        hist_by_bus: Dict ``bus_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        synth_by_bus: Dict ``bus_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        bus_names: Dict ``bus_id -> name string``.
        stage_labels: Mapping ``stage_id -> label string``.
        bus_ids: Ordered list of bus IDs to display (up to 4 used).

    Returns:
        A :class:`plotly.graph_objects.Figure` with 4 subplots arranged 2x2.
    """
    selected = bus_ids[:4]
    subplot_titles = [bus_names.get(bid, str(bid)) for bid in selected]
    # Pad to 4 for make_subplots
    while len(subplot_titles) < 4:
        subplot_titles.append("")

    fig = ps.make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        shared_xaxes=False,
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for idx, bus_id in enumerate(selected):
        row, col = positions[idx]
        h = hist_by_bus.get(bus_id, pd.DataFrame())
        s = synth_by_bus.get(bus_id, pd.DataFrame())

        if not h.empty:
            h = h.sort_values("stage_id")
            x_labels = stage_x_labels(h["stage_id"].tolist(), stage_labels)
            _add_mean_std_band(
                fig,
                h["stage_id"].tolist(),
                h["mean_m3s"].tolist(),
                h["std_m3s"].fillna(0.0).tolist(),
                x_labels,
                "Historical",
                _HIST_COLOR,
                row=row,
                col=col,
            )

        if not s.empty:
            s = s.sort_values("stage_id")
            x_labels = stage_x_labels(s["stage_id"].tolist(), stage_labels)
            _add_mean_std_band(
                fig,
                s["stage_id"].tolist(),
                s["mean_m3s"].tolist(),
                s["std_m3s"].fillna(0.0).tolist(),
                x_labels,
                "Synthetic",
                _SYNTH_COLOR,
                row=row,
                col=col,
            )

    # Suppress legend for all traces except the first occurrence of each group
    seen: set[str] = set()
    for trace in fig.data:
        lg = getattr(trace, "legendgroup", None)
        if lg and trace.showlegend is not False:
            if lg in seen:
                trace.showlegend = False
            else:
                seen.add(lg)

    fig.update_layout(
        legend=LEGEND_DEFAULTS,
        margin=dict(l=50, r=20, t=60, b=10),
        template="plotly_white",
    )
    return fig


def _chart_hydro_explorer(
    hist_per_hydro: dict[int, pd.DataFrame],
    synth_per_hydro: dict[int, pd.DataFrame],
    hydro_meta: dict[int, dict],
    stage_labels: dict[int, str],
) -> go.Figure:
    """Build a per-hydro explorer figure with dropdown menu.

    All hydros' traces are added to the figure. A Plotly ``updatemenus``
    dropdown allows toggling visibility per hydro. The first hydro is visible
    by default; all others are hidden.

    Args:
        hist_per_hydro: Dict ``hydro_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        synth_per_hydro: Dict ``hydro_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
        hydro_meta: Dict ``hydro_id -> {name: str, ...}``.
        stage_labels: Mapping ``stage_id -> label string``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with updatemenus dropdown.
    """
    fig = go.Figure()

    # Ordered list of hydro_ids (sorted for deterministic order)
    hydro_ids = sorted(hydro_meta.keys())

    # Each hydro contributes up to 6 traces: 3 for hist (lower, upper, mean)
    # and 3 for synth.  Track the trace count per hydro to build visibility arrays.
    traces_per_hydro: list[int] = []

    for hydro_id in hydro_ids:
        trace_count_before = len(fig.data)

        h = hist_per_hydro.get(hydro_id, pd.DataFrame())
        if not h.empty:
            h = h.sort_values("stage_id")
            x_labels = stage_x_labels(h["stage_id"].tolist(), stage_labels)
            _add_mean_std_band(
                fig,
                h["stage_id"].tolist(),
                h["mean_m3s"].tolist(),
                h["std_m3s"].fillna(0.0).tolist(),
                x_labels,
                "Historical",
                _HIST_COLOR,
            )

        s = synth_per_hydro.get(hydro_id, pd.DataFrame())
        if not s.empty:
            s = s.sort_values("stage_id")
            x_labels = stage_x_labels(s["stage_id"].tolist(), stage_labels)
            _add_mean_std_band(
                fig,
                s["stage_id"].tolist(),
                s["mean_m3s"].tolist(),
                s["std_m3s"].fillna(0.0).tolist(),
                x_labels,
                "Synthetic",
                _SYNTH_COLOR,
            )

        trace_count_after = len(fig.data)
        traces_per_hydro.append(trace_count_after - trace_count_before)

    total_traces = len(fig.data)

    # Build visibility arrays per hydro and dropdown buttons
    buttons: list[dict] = []
    trace_offset = 0
    for i, hydro_id in enumerate(hydro_ids):
        n = traces_per_hydro[i]
        # Visibility array: True only for this hydro's traces
        visible: list[bool] = [False] * total_traces
        for j in range(n):
            visible[trace_offset + j] = True
        name = hydro_meta[hydro_id].get("name", str(hydro_id))
        buttons.append(
            {
                "method": "update",
                "label": name,
                "args": [
                    {"visible": visible},
                    {"title": f"Inflow — {name}"},
                ],
            }
        )
        trace_offset += n

    # Default: first hydro visible, rest hidden
    for trace_idx, trace in enumerate(fig.data):
        trace.visible = trace_idx < traces_per_hydro[0] if traces_per_hydro else False  # type: ignore[assignment]

    first_name = (
        hydro_meta[hydro_ids[0]].get("name", str(hydro_ids[0]))
        if hydro_ids
        else "No Data"
    )

    fig.update_layout(
        title=f"Inflow — {first_name}",
        xaxis_title="Stage",
        yaxis_title="Inflow (m³/s)",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "active": 0,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "type": "dropdown",
            }
        ],
    )
    return fig


# ---------------------------------------------------------------------------
# Per-hydro stats extraction from aggregated data
# ---------------------------------------------------------------------------


def _per_hydro_stats(
    stats_df: pd.DataFrame,
    hydro_ids: list[int],
) -> dict[int, pd.DataFrame]:
    """Split an aggregated stats DataFrame into per-hydro DataFrames.

    Args:
        stats_df: DataFrame with columns ``hydro_id``, ``stage_id``,
            ``mean_m3s``, ``std_m3s``.
        hydro_ids: List of hydro IDs to include.

    Returns:
        Dict ``hydro_id -> DataFrame(stage_id, mean_m3s, std_m3s)``.
    """
    if stats_df.empty:
        return {}
    result: dict[int, pd.DataFrame] = {}
    for hid in hydro_ids:
        sub = stats_df[stats_df["hydro_id"] == hid][["stage_id", "mean_m3s", "std_m3s"]]
        if not sub.empty:
            result[hid] = sub.reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Correlation and noise helpers
# ---------------------------------------------------------------------------

_MONTH_NAMES: list[str] = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]

_REDUCTION_COLORS: list[str] = [
    "#4A90B8",
    "#F5A623",
    "#DC4C4C",
    "#4A8B6F",
    "#B87333",
    "#8B5E3C",
    "#607D8B",
    "#8B9298",
]


def _extract_fitted_correlation(
    corr_dict: dict,
) -> tuple[list[int], np.ndarray] | None:
    """Extract hydro_ids and correlation matrix from a correlation dict.

    Navigates ``corr_dict["profiles"]["default"]["correlation_groups"][0]``
    and returns the entity IDs and matrix.

    Args:
        corr_dict: The loaded ``correlation.json`` dict.

    Returns:
        A tuple ``(hydro_ids, matrix)`` where ``hydro_ids`` is a list of
        integer IDs and ``matrix`` is a 2-D :class:`numpy.ndarray`, or
        ``None`` if any required key is missing or the matrix is empty.
    """
    try:
        group = corr_dict["profiles"]["default"]["correlation_groups"][0]
        entities: list[dict] = group["entities"]
        matrix_raw: list[list[float]] = group["matrix"]
    except (KeyError, IndexError, TypeError):
        return None

    if not entities or not matrix_raw:
        return None

    hydro_ids = [int(e["id"]) for e in entities]
    matrix = np.array(matrix_raw, dtype=np.float64)
    return hydro_ids, matrix


def _compute_empirical_correlation(
    inflow_history: pd.DataFrame,
    hydro_ids: list[int],
) -> np.ndarray | None:
    """Compute an averaged empirical correlation matrix from inflow history.

    Groups ``inflow_history`` by calendar month, computes a per-hydro
    correlation matrix for each month, then returns the element-wise mean
    across the 12 monthly matrices.

    Args:
        inflow_history: DataFrame with columns ``hydro_id``, ``date``,
            ``value_m3s``.
        hydro_ids: Ordered list of hydro IDs to include (determines row/column
            order of the returned matrix).

    Returns:
        A 2-D :class:`numpy.ndarray` of shape ``(N, N)`` where ``N =
        len(hydro_ids)``, or ``None`` when the history is empty or there are
        fewer than 2 hydros after filtering.
    """
    if inflow_history.empty or len(hydro_ids) < 2:
        return None

    df = inflow_history[inflow_history["hydro_id"].isin(hydro_ids)].copy()
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month

    n = len(hydro_ids)
    cumulative = np.zeros((n, n), dtype=np.float64)
    count = 0

    for _month in range(1, 13):
        sub = df[df["month"] == _month]
        if sub.empty:
            continue
        pivot = sub.pivot_table(
            index="date", columns="hydro_id", values="value_m3s", aggfunc="mean"
        )
        # Reindex to the canonical hydro_id order; missing hydros get NaN cols
        pivot = pivot.reindex(columns=hydro_ids)
        if pivot.shape[0] < 2:
            continue
        corr_m = pivot.corr(min_periods=10)
        corr_arr = corr_m.reindex(index=hydro_ids, columns=hydro_ids).values
        if not np.all(np.isnan(corr_arr)):
            cumulative += np.nan_to_num(corr_arr, nan=0.0)
            count += 1

    if count == 0:
        return None

    result = cumulative / count
    # Ensure diagonal is exactly 1.0 for non-NaN entries
    np.fill_diagonal(result, 1.0)
    return result


def _chart_correlation_heatmap(
    matrix: np.ndarray,
    hydro_ids: list[int],
    hydro_meta: dict[int, dict],
    title: str,
    max_display: int = 40,
) -> tuple[go.Figure, str]:
    """Build a correlation heatmap figure.

    Truncates to ``max_display x max_display`` and appends a truncation
    notice to the title when the matrix exceeds that size.

    Args:
        matrix: Square symmetric 2-D array.
        hydro_ids: Ordered list of hydro IDs matching the matrix rows/cols.
        hydro_meta: Mapping ``hydro_id -> {"name": str, ...}``.
        title: Chart title (may be extended with truncation note).
        max_display: Maximum number of rows/columns to display.

    Returns:
        A tuple ``(fig, resolved_title)`` where ``resolved_title`` includes
        the truncation note when the matrix was truncated.
    """
    total = len(hydro_ids)
    if total > max_display:
        display_ids = hydro_ids[:max_display]
        display_matrix = matrix[:max_display, :max_display]
        title = f"{title} (showing {max_display} of {total})"
    else:
        display_ids = hydro_ids
        display_matrix = matrix

    labels = [
        hydro_meta[hid]["name"] if hid in hydro_meta else str(hid)
        for hid in display_ids
    ]

    fig = go.Figure(
        go.Heatmap(
            z=display_matrix.tolist(),
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            zmin=-1.0,
            zmax=1.0,
            hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
            colorbar={"title": "Corr"},
        )
    )
    fig.update_layout(
        xaxis_title="Hydro",
        yaxis_title="Hydro",
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig, title


def _chart_noise_histogram(noise_openings: pd.DataFrame) -> go.Figure:
    """Build a noise histogram overlaid with the N(0,1) reference curve.

    Args:
        noise_openings: DataFrame with a ``value`` column of noise samples.

    Returns:
        A :class:`plotly.graph_objects.Figure` with two traces: histogram and
        N(0,1) reference line.
    """
    vals = noise_openings["value"].dropna().values
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=vals,
            histnorm="probability density",
            nbinsx=60,
            name="Noise samples",
            marker_color=COLORS["hydro"],
            opacity=0.7,
            hovertemplate="Value: %{x:.3f}<br>Density: %{y:.4f}<extra></extra>",
        )
    )
    x_ref = np.linspace(float(vals.min()), float(vals.max()), 300)
    y_ref = np.exp(-0.5 * x_ref**2) / np.sqrt(2 * np.pi)
    fig.add_trace(
        go.Scatter(
            x=x_ref,
            y=y_ref,
            name="N(0,1) reference",
            line={"color": COLORS["deficit"], "width": 2, "dash": "dash"},
            hovertemplate="x: %{x:.3f}<br>N(0,1): %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Value",
        yaxis_title="Density",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


def _chart_noise_boxplot_by_stage(
    noise_openings: pd.DataFrame,
    stage_labels: dict[int, str],
) -> go.Figure:
    """Build a box plot of noise values grouped by stage (first 12 stages).

    Args:
        noise_openings: DataFrame with columns ``stage_id`` and ``value``.
        stage_labels: Mapping ``stage_id -> label string``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one :class:`~plotly.graph_objects.Box`
        trace per stage (max 12).
    """
    all_stage_ids = sorted(noise_openings["stage_id"].unique().tolist())
    stage_ids = all_stage_ids[:12]

    fig = go.Figure()
    for sid in stage_ids:
        sub = noise_openings[noise_openings["stage_id"] == sid]["value"].dropna()
        label = stage_labels.get(int(sid), str(sid))
        fig.add_trace(
            go.Box(
                y=sub.values,
                name=label,
                marker_color=COLORS["hydro"],
                boxmean=False,
                showlegend=False,
                hovertemplate=f"Stage: {label}<br>Value: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Stage",
        yaxis_title="Noise Value",
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


def _chart_ar_order_distribution(fitting_report: dict) -> go.Figure:
    """Build a bar chart of selected AR order distribution across hydros.

    Args:
        fitting_report: The loaded ``fitting_report.json`` dict with a
            ``"hydros"`` key mapping hydro IDs to per-hydro fitting data.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one :class:`~plotly.graph_objects.Bar`
        trace.
    """
    hydros_data: dict = fitting_report.get("hydros", {})
    order_counts: dict[int, int] = {}
    for hydro_info in hydros_data.values():
        order = int(hydro_info.get("selected_order", 0))
        order_counts[order] = order_counts.get(order, 0) + 1

    max_order = max(order_counts) if order_counts else 0
    orders = list(range(max_order + 1))
    counts = [order_counts.get(o, 0) for o in orders]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=orders,
            y=counts,
            marker_color=COLORS["hydro"],
            hovertemplate="Order %{x}: %{y} hydros<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Selected AR Order",
        yaxis_title="Number of Hydros",
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


def _chart_order_reduction_reasons(fitting_report: dict) -> go.Figure:
    """Build a stacked bar chart of AR order reduction reasons by season.

    Args:
        fitting_report: The loaded ``fitting_report.json`` dict with a
            ``"hydros"`` key.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one stacked bar trace per
        reduction reason.
    """
    hydros_data: dict = fitting_report.get("hydros", {})
    reason_season: dict[str, list[int]] = {}
    for hydro_info in hydros_data.values():
        contrib = hydro_info.get("contribution_reductions", [])
        for season_id, reasons in enumerate(contrib):
            if season_id >= 12:
                break
            if not isinstance(reasons, list):
                continue
            for reason in reasons:
                if reason not in reason_season:
                    reason_season[reason] = [0] * 12
                reason_season[reason][season_id] += 1

    fig = go.Figure()
    for i, (reason, counts) in enumerate(sorted(reason_season.items())):
        fig.add_trace(
            go.Bar(
                x=_MONTH_NAMES,
                y=counts,
                name=reason,
                marker_color=_REDUCTION_COLORS[i % len(_REDUCTION_COLORS)],
                hovertemplate=f"{reason}<br>%{{x}}: %{{y}} reductions<extra></extra>",
            )
        )
    fig.update_layout(
        xaxis_title="Season (Month)",
        yaxis_title="Count of Reductions",
        barmode="stack",
        legend=LEGEND_DEFAULTS,
        margin=MARGIN_DEFAULTS,
        template="plotly_white",
    )
    return fig


# ---------------------------------------------------------------------------
# Section renderers for D, E, F
# ---------------------------------------------------------------------------


def _render_section_d(data: DashboardData) -> str:
    """Render Section D — Spatial Correlation heatmaps (fitted vs empirical)."""
    if not data.correlation:
        return collapsible_section(
            "Spatial Correlation",
            "<p>No correlation data available.</p>",
            section_id="v2-stoch-section-d",
        )

    extracted = _extract_fitted_correlation(data.correlation)
    if extracted is None:
        return collapsible_section(
            "Spatial Correlation",
            "<p>No correlation matrix in data.</p>",
            section_id="v2-stoch-section-d",
        )

    hydro_ids, fitted_matrix = extracted

    # Fitted heatmap
    fitted_fig, fitted_title = _chart_correlation_heatmap(
        fitted_matrix, hydro_ids, data.hydro_meta, "Fitted Correlation"
    )
    fitted_html = make_chart_card(
        fitted_fig,
        fitted_title,
        "v2-stoch-corr-fitted",
        height=500,
    )

    # Empirical heatmap
    if data.inflow_history.empty:
        empirical_html = '<div class="chart-card"><p>No historical data for empirical correlation.</p></div>'
    else:
        emp_matrix = _compute_empirical_correlation(data.inflow_history, hydro_ids)
        if emp_matrix is None:
            empirical_html = '<div class="chart-card"><p>No historical data for empirical correlation.</p></div>'
        else:
            emp_fig, emp_title = _chart_correlation_heatmap(
                emp_matrix, hydro_ids, data.hydro_meta, "Empirical Correlation"
            )
            empirical_html = make_chart_card(
                emp_fig,
                emp_title,
                "v2-stoch-corr-empirical",
                height=500,
            )

    content = chart_grid([fitted_html, empirical_html])
    return collapsible_section(
        "Spatial Correlation",
        content,
        section_id="v2-stoch-section-d",
    )


def _render_section_e(data: DashboardData) -> str:
    """Render Section E — Noise Diagnostics (histogram and stage box plot)."""
    if data.noise_openings.empty:
        return collapsible_section(
            "Noise Diagnostics",
            "<p>No noise data.</p>",
            section_id="v2-stoch-section-e",
        )

    hist_fig = _chart_noise_histogram(data.noise_openings)
    hist_html = make_chart_card(
        hist_fig,
        "Noise Sample Distribution vs N(0,1)",
        "v2-stoch-noise-histogram",
    )

    box_fig = _chart_noise_boxplot_by_stage(data.noise_openings, data.stage_labels)
    box_html = make_chart_card(
        box_fig,
        "Noise Distribution by Stage (first 12)",
        "v2-stoch-noise-boxplot",
    )

    content = chart_grid([hist_html, box_html])
    return collapsible_section(
        "Noise Diagnostics",
        content,
        section_id="v2-stoch-section-e",
    )


def _render_section_f(data: DashboardData) -> str:
    """Render Section F — AR Model Summary (order distribution and reduction reasons)."""
    if not data.fitting_report:
        return collapsible_section(
            "AR Model Summary",
            "<p>No fitting report available.</p>",
            section_id="v2-stoch-section-f",
        )

    order_fig = _chart_ar_order_distribution(data.fitting_report)
    order_html = make_chart_card(
        order_fig,
        "AR Order Distribution",
        "v2-stoch-ar-order",
    )

    reasons_fig = _chart_order_reduction_reasons(data.fitting_report)
    reasons_html = make_chart_card(
        reasons_fig,
        "AR Order Reduction Reasons by Season",
        "v2-stoch-ar-reasons",
    )

    content = chart_grid([order_html, reasons_html])
    return collapsible_section(
        "AR Model Summary",
        content,
        section_id="v2-stoch-section-f",
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


def _render_section_a(data: DashboardData) -> str:
    """Render Section A — System-Wide Inflow hero chart."""
    if data.inflow_history.empty:
        return collapsible_section(
            "System-Wide Inflow",
            _NO_HIST,
            section_id="v2-stoch-section-a",
        )

    hist_stats = _compute_historical_stats(data)
    hist_system = _aggregate_system(hist_stats)

    # Synthetic system-level aggregation
    if data.inflow_stats_stoch.empty:
        synth_system: pd.DataFrame = pd.DataFrame(
            columns=["stage_id", "mean_m3s", "std_m3s"]
        )
    else:
        synth_system = _aggregate_system(data.inflow_stats_stoch)

    fig = _chart_system_inflow(hist_system, synth_system, data.stage_labels)
    content = make_chart_card(
        fig, "System-Wide Inflow (Historical vs Synthetic)", "v2-stoch-system-inflow"
    )

    return collapsible_section(
        "System-Wide Inflow",
        content,
        section_id="v2-stoch-section-a",
    )


def _render_section_b(data: DashboardData) -> str:
    """Render Section B — Inflow by Bus (2x2 facet)."""
    if data.inflow_history.empty:
        return collapsible_section(
            "Inflow by Bus",
            _NO_HIST,
            section_id="v2-stoch-section-b",
        )

    if not data.hydro_bus_map:
        return collapsible_section(
            "Inflow by Bus",
            "<p>No bus mapping available.</p>",
            section_id="v2-stoch-section-b",
        )

    hist_stats = _compute_historical_stats(data)
    hist_by_bus = _aggregate_by_bus(hist_stats, data.hydro_bus_map)

    if not hist_by_bus:
        return collapsible_section(
            "Inflow by Bus",
            "<p>No bus mapping available.</p>",
            section_id="v2-stoch-section-b",
        )

    if data.inflow_stats_stoch.empty:
        synth_by_bus: dict[int, pd.DataFrame] = {}
    else:
        synth_by_bus = _aggregate_by_bus(data.inflow_stats_stoch, data.hydro_bus_map)

    # Use non_fictitious_bus_ids intersected with buses that have hydros
    hydro_bus_ids = set(data.hydro_bus_map.values())
    non_fict = [bid for bid in data.non_fictitious_bus_ids if bid in hydro_bus_ids]
    # Fallback: all buses with hydros (sorted) if non_fictitious_bus_ids is empty
    if not non_fict:
        non_fict = sorted(hydro_bus_ids)

    bus_ids_to_show = non_fict[:4]
    if not bus_ids_to_show:
        return collapsible_section(
            "Inflow by Bus",
            "<p>No bus mapping available.</p>",
            section_id="v2-stoch-section-b",
        )

    fig = _chart_bus_facet(
        hist_by_bus,
        synth_by_bus,
        data.bus_names,
        data.stage_labels,
        bus_ids_to_show,
    )
    content = make_chart_card(
        fig, "Inflow by Bus (Historical vs Synthetic)", "v2-stoch-bus-facet", height=600
    )

    return collapsible_section(
        "Inflow by Bus",
        content,
        section_id="v2-stoch-section-b",
    )


def _render_section_c(data: DashboardData) -> str:
    """Render Section C — Per-Hydro Inflow Explorer with dropdown."""
    if not data.hydro_meta:
        return collapsible_section(
            "Per-Hydro Inflow Explorer",
            "<p>No hydro data.</p>",
            section_id="v2-stoch-section-c",
        )

    if data.inflow_history.empty:
        return collapsible_section(
            "Per-Hydro Inflow Explorer",
            _NO_HIST,
            section_id="v2-stoch-section-c",
        )

    hist_stats = _compute_historical_stats(data)
    hydro_ids = sorted(data.hydro_meta.keys())
    hist_per_hydro = _per_hydro_stats(hist_stats, hydro_ids)

    if data.inflow_stats_stoch.empty:
        synth_per_hydro: dict[int, pd.DataFrame] = {}
    else:
        synth_per_hydro = _per_hydro_stats(data.inflow_stats_stoch, hydro_ids)

    fig = _chart_hydro_explorer(
        hist_per_hydro,
        synth_per_hydro,
        data.hydro_meta,
        data.stage_labels,
    )
    content = make_chart_card(
        fig, "Per-Hydro Inflow Explorer", "v2-stoch-hydro-explorer", height=460
    )

    return collapsible_section(
        "Per-Hydro Inflow Explorer",
        content,
        section_id="v2-stoch-section-c",
    )


# ---------------------------------------------------------------------------
# Tab interface (TabModule protocol)
# ---------------------------------------------------------------------------


def can_render(data: DashboardData) -> bool:
    """Return True when stochastic model output is available."""
    return data.stochastic_available


def render(data: DashboardData) -> str:
    """Return the full HTML string for the v2 Stochastic Model tab content area."""
    sections = [
        _render_section_a(data),
        _render_section_b(data),
        _render_section_c(data),
        _render_section_d(data),
        _render_section_e(data),
        _render_section_f(data),
    ]
    return "".join(sections)
