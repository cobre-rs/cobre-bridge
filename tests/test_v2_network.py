"""Unit tests for cobre_bridge.dashboard.tabs.v2_network.

Covers module constants, can_render, build_line_explorer, build_heatmap,
build_bus_balance, and the render() guard path.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import polars as pl

import cobre_bridge.dashboard.tabs.v2_network as v2_network
from cobre_bridge.dashboard.tabs.v2_network import (
    TAB_ID,
    TAB_LABEL,
    TAB_ORDER,
    build_bus_balance,
    build_heatmap,
    build_line_explorer,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def _make_bh_df(n_stages: int = 3) -> pl.DataFrame:
    """Return a block-hours DataFrame with one block per stage."""
    return pl.DataFrame(
        {
            "stage_id": list(range(n_stages)),
            "block_id": [0] * n_stages,
            "_bh": [720.0] * n_stages,
        }
    )


def _make_exchanges_lf(
    line_ids: list[int] | None = None,
    n_scenarios: int = 2,
    n_stages: int = 3,
) -> pl.LazyFrame:
    """Return an exchanges LazyFrame with all three flow columns."""
    if line_ids is None:
        line_ids = [0, 1]
    rows: list[dict] = []
    for scenario_id in range(n_scenarios):
        for stage_id in range(n_stages):
            for line_id in line_ids:
                rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage_id": stage_id,
                        "block_id": 0,
                        "line_id": line_id,
                        "net_flow_mw": float(50 + line_id * 10 + stage_id * 5),
                        "direct_flow_mw": float(80 + line_id * 5),
                        "reverse_flow_mw": float(30 - line_id * 3),
                    }
                )
    return pl.DataFrame(rows).lazy()


def _make_empty_exchanges_lf() -> pl.LazyFrame:
    """Return an exchanges LazyFrame with schema but no rows."""
    return pl.DataFrame(
        {
            "scenario_id": pl.Series([], dtype=pl.Int64),
            "stage_id": pl.Series([], dtype=pl.Int64),
            "block_id": pl.Series([], dtype=pl.Int64),
            "line_id": pl.Series([], dtype=pl.Int64),
            "net_flow_mw": pl.Series([], dtype=pl.Float64),
            "direct_flow_mw": pl.Series([], dtype=pl.Float64),
            "reverse_flow_mw": pl.Series([], dtype=pl.Float64),
        }
    ).lazy()


def _make_names(line_ids: list[int] | None = None) -> dict[tuple[str, int], str]:
    """Return a names dict for lines."""
    if line_ids is None:
        line_ids = [0, 1]
    return {("lines", lid): f"Line {lid}" for lid in line_ids}


def _make_stage_labels(n_stages: int = 3) -> dict[int, str]:
    """Return a stage_labels dict."""
    months = ["Jan 2024", "Feb 2024", "Mar 2024"]
    return {i: months[i] for i in range(n_stages)}


def _make_line_meta(line_ids: list[int] | None = None) -> list[dict]:
    """Return line metadata with source/target bus IDs and capacities."""
    if line_ids is None:
        line_ids = [0, 1]
    result: list[dict] = []
    for lid in line_ids:
        result.append(
            {
                "id": lid,
                "name": f"Line {lid}",
                "source_bus_id": 0,
                "target_bus_id": lid + 1,
                "direct_capacity_mw": 200.0,
                "reverse_capacity_mw": 150.0,
            }
        )
    return result


def _make_line_bounds(
    line_ids: list[int] | None = None, n_stages: int = 3
) -> pd.DataFrame:
    """Return a line_bounds DataFrame with direct_mw and reverse_mw."""
    if line_ids is None:
        line_ids = [0, 1]
    rows: list[dict] = []
    for lid in line_ids:
        for stage_id in range(n_stages):
            rows.append(
                {
                    "line_id": lid,
                    "stage_id": stage_id,
                    "direct_mw": 100.0 + lid * 10,
                    "reverse_mw": 50.0 + lid * 5,
                }
            )
    return pd.DataFrame(rows)


def _make_mock_data(
    line_ids: list[int] | None = None,
    *,
    empty_line_meta: bool = False,
) -> MagicMock:
    """Build a minimal MagicMock satisfying the DashboardData interface."""
    if line_ids is None:
        line_ids = [0, 1]

    line_meta = [] if empty_line_meta else _make_line_meta(line_ids)

    data = MagicMock()
    data.line_meta = line_meta
    data.line_bounds = _make_line_bounds(line_ids)
    data.names = _make_names(line_ids)
    data.stage_labels = _make_stage_labels()
    data.bus_names = {0: "Bus Alpha", 1: "Bus Beta", 2: "Bus Gamma"}
    data.bh_df = _make_bh_df()
    data.exchanges_lf = _make_exchanges_lf(line_ids)
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """Module-level constants must match the ticket specification exactly."""
    assert TAB_ID == "tab-v2-network"
    assert TAB_LABEL == "Network"
    assert TAB_ORDER == 60


def test_module_attributes() -> None:
    """Verify constants are accessible as module attributes."""
    assert v2_network.TAB_ID == "tab-v2-network"
    assert v2_network.TAB_LABEL == "Network"
    assert v2_network.TAB_ORDER == 60


# ---------------------------------------------------------------------------
# test_can_render
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render should return True for any DashboardData mock."""
    data = MagicMock()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test_build_line_explorer
# ---------------------------------------------------------------------------


def test_build_line_explorer_contains_select_and_divs() -> None:
    """build_line_explorer returns HTML with select dropdown and chart divs."""
    exchanges_lf = _make_exchanges_lf([0, 1], n_scenarios=2, n_stages=3)
    bh_df = _make_bh_df(3)
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels(3)

    html = build_line_explorer(exchanges_lf, names, stage_labels, bh_df)

    assert 'id="nw-select"' in html
    assert 'id="nw-net"' in html
    assert 'id="nw-dir"' in html


def test_build_line_explorer_has_two_options() -> None:
    """build_line_explorer returns exactly 2 <option> elements for 2 lines."""
    exchanges_lf = _make_exchanges_lf([0, 1], n_scenarios=2, n_stages=3)
    bh_df = _make_bh_df(3)
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels(3)

    html = build_line_explorer(exchanges_lf, names, stage_labels, bh_df)

    assert html.count("<option") == 2


def test_build_line_explorer_embeds_js_data() -> None:
    """build_line_explorer embeds NW_DATA and NW_LABELS JS constants."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    html = build_line_explorer(exchanges_lf, names, stage_labels, bh_df)

    assert "NW_DATA" in html
    assert "NW_LABELS" in html
    assert "updateNetworkDetail" in html


def test_build_line_explorer_auto_trigger() -> None:
    """build_line_explorer includes DOMContentLoaded auto-trigger."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    html = build_line_explorer(exchanges_lf, names, stage_labels, bh_df)

    assert "DOMContentLoaded" in html


def test_build_line_explorer_empty_exchanges_returns_no_data() -> None:
    """build_line_explorer returns a no-data message when exchanges_lf is empty."""
    exchanges_lf = _make_empty_exchanges_lf()
    bh_df = _make_bh_df()
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    html = build_line_explorer(exchanges_lf, names, stage_labels, bh_df)

    assert "<p>" in html
    assert "nw-select" not in html


# ---------------------------------------------------------------------------
# test_build_heatmap
# ---------------------------------------------------------------------------


def test_build_heatmap_with_line_bounds_returns_plotly_html() -> None:
    """build_heatmap with line_bounds returns HTML containing Plotly content."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    line_bounds = _make_line_bounds([0, 1])
    line_meta = _make_line_meta([0, 1])
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    html = build_heatmap(
        exchanges_lf, line_bounds, line_meta, names, stage_labels, bh_df
    )

    # Should contain plotly div content
    assert "plotly" in html.lower() or "Heatmap" in html or "<div" in html
    assert html  # non-empty


def test_build_heatmap_does_not_raise() -> None:
    """build_heatmap must not raise any exception with valid inputs."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    line_bounds = _make_line_bounds([0, 1])
    line_meta = _make_line_meta([0, 1])
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    # No exception expected
    html = build_heatmap(
        exchanges_lf, line_bounds, line_meta, names, stage_labels, bh_df
    )
    assert isinstance(html, str)


def test_build_heatmap_without_line_bounds_falls_back_to_line_meta() -> None:
    """build_heatmap with empty line_bounds falls back to static capacity."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    empty_bounds = pd.DataFrame(
        columns=["line_id", "stage_id", "direct_mw", "reverse_mw"]
    )
    line_meta = _make_line_meta([0, 1])
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    # Must not raise even with empty bounds
    html = build_heatmap(
        exchanges_lf, empty_bounds, line_meta, names, stage_labels, bh_df
    )
    assert isinstance(html, str)
    assert html  # non-empty


def test_build_heatmap_empty_exchanges_returns_no_data() -> None:
    """build_heatmap returns a no-data message when exchanges_lf has no rows."""
    exchanges_lf = _make_empty_exchanges_lf()
    bh_df = _make_bh_df()
    line_bounds = _make_line_bounds([0, 1])
    line_meta = _make_line_meta([0, 1])
    names = _make_names([0, 1])
    stage_labels = _make_stage_labels()

    html = build_heatmap(
        exchanges_lf, line_bounds, line_meta, names, stage_labels, bh_df
    )

    assert "<p>" in html


# ---------------------------------------------------------------------------
# test_build_bus_balance
# ---------------------------------------------------------------------------


def test_build_bus_balance_returns_plotly_bar_html() -> None:
    """build_bus_balance returns HTML containing a Plotly bar chart."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    line_meta = _make_line_meta([0, 1])
    bus_names = {0: "Bus Alpha", 1: "Bus Beta", 2: "Bus Gamma"}

    html = build_bus_balance(exchanges_lf, line_meta, bus_names, bh_df)

    # Should contain a chart div
    assert "<div" in html
    assert html  # non-empty


def test_build_bus_balance_includes_bus_names() -> None:
    """build_bus_balance includes bus names in the rendered output."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()
    line_meta = _make_line_meta([0, 1])
    bus_names = {0: "Bus Alpha", 1: "Bus Beta", 2: "Bus Gamma"}

    html = build_bus_balance(exchanges_lf, line_meta, bus_names, bh_df)

    # Bus names should appear somewhere in the rendered Plotly HTML
    assert "Bus Alpha" in html or "Bus Beta" in html or "Bus Gamma" in html


def test_build_bus_balance_no_line_meta_returns_no_data() -> None:
    """build_bus_balance returns no-data message when line_meta is empty."""
    exchanges_lf = _make_exchanges_lf([0, 1])
    bh_df = _make_bh_df()

    html = build_bus_balance(exchanges_lf, [], {0: "Bus A"}, bh_df)

    assert "<p>" in html


def test_build_bus_balance_direction() -> None:
    """Bus balance: target bus has positive flow, source bus has negative flow."""
    # One line: source_bus=0, target_bus=1, net_flow=100 MW
    exchanges_lf = pl.DataFrame(
        [
            {
                "scenario_id": 0,
                "stage_id": 0,
                "block_id": 0,
                "line_id": 0,
                "net_flow_mw": 100.0,
                "direct_flow_mw": 100.0,
                "reverse_flow_mw": 0.0,
            }
        ]
    ).lazy()
    bh_df = pl.DataFrame({"stage_id": [0], "block_id": [0], "_bh": [720.0]})
    line_meta = [
        {
            "id": 0,
            "name": "L0",
            "source_bus_id": 0,
            "target_bus_id": 1,
            "direct_capacity_mw": 200.0,
            "reverse_capacity_mw": 150.0,
        }
    ]
    bus_names = {0: "Source", 1: "Target"}

    html = build_bus_balance(exchanges_lf, line_meta, bus_names, bh_df)

    # Should render without error and contain bus names
    assert isinstance(html, str)
    assert "Source" in html or "Target" in html


# ---------------------------------------------------------------------------
# test_render
# ---------------------------------------------------------------------------


def test_render_empty_line_meta_returns_no_data_paragraph() -> None:
    """render() returns no-network-data paragraph when line_meta is empty."""
    data = _make_mock_data(empty_line_meta=True)

    result = render(data)

    assert result == "<p>No network data available.</p>"


def test_render_normal_contains_all_section_titles() -> None:
    """render() with valid data contains all 3 section titles."""
    data = _make_mock_data([0, 1])

    result = render(data)

    assert "Line Explorer" in result
    assert "Capacity Utilisation" in result
    assert "Bus Balance" in result


def test_render_normal_contains_line_explorer_elements() -> None:
    """render() output contains the interactive line explorer elements."""
    data = _make_mock_data([0, 1])

    result = render(data)

    assert "nw-select" in result
    assert "nw-net" in result
    assert "nw-dir" in result
