"""Unit tests for src/cobre_bridge/dashboard/tabs/v2_performance.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from cobre_bridge.dashboard.tabs.v2_performance import (
    TAB_ID,
    TAB_LABEL,
    TAB_ORDER,
    _build_metrics_row,
    _chart_retry_histogram,
    _format_time,
    can_render,
    render,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_solver_df(
    n_rows: int = 5,
    *,
    phase: str = "backward",
    lp_solves: int = 10,
    solve_time_ms: float = 50.0,
    simplex_iterations: int = 200,
) -> pd.DataFrame:
    """Synthetic solver DataFrame (solver_train or solver_sim).

    Includes all columns required by ticket-021 chart functions so that
    tests using the default ``_make_mock_data()`` do not fail on missing columns.
    """
    return pd.DataFrame(
        {
            "iteration": list(range(1, n_rows + 1)),
            "stage": list(range(0, n_rows)),
            "phase": [phase] * n_rows,
            "lp_solves": [lp_solves] * n_rows,
            "solve_time_ms": [solve_time_ms] * n_rows,
            "simplex_iterations": [simplex_iterations] * n_rows,
            "basis_offered": [8] * n_rows,
            "basis_rejections": [2] * n_rows,
            "set_bounds_time_ms": [5.0] * n_rows,
            "add_rows_time_ms": [3.0] * n_rows,
            "load_model_time_ms": [2.0] * n_rows,
        }
    )


def _make_timing(n: int = 5) -> pd.DataFrame:
    """Synthetic timing DataFrame."""
    return pd.DataFrame(
        {
            "iteration": list(range(1, n + 1)),
            "forward_solve_ms": [100.0 + i for i in range(n)],
            "backward_solve_ms": [150.0 + i for i in range(n)],
            "overhead_ms": [10.0 + i for i in range(n)],
        }
    )


def _make_solver_train(n_stages: int = 4, n_iters: int = 3) -> pd.DataFrame:
    """Synthetic solver_train DataFrame with all columns required by ticket-021.

    Includes both forward and backward rows so phase-filtered chart functions
    can find backward data.
    """
    rows = []
    for it in range(1, n_iters + 1):
        for stage in range(0, n_stages):
            for phase in ("forward", "backward"):
                rows.append(
                    {
                        "iteration": it,
                        "stage": stage,
                        "phase": phase,
                        "lp_solves": 10,
                        "solve_time_ms": 50.0 + stage * 2.0,
                        "simplex_iterations": 200 + stage * 5,
                        "basis_offered": 8,
                        "basis_rejections": 2,
                        "set_bounds_time_ms": 5.0 + stage * 0.5,
                        "add_rows_time_ms": 3.0,
                        "load_model_time_ms": 2.0,
                    }
                )
    return pd.DataFrame(rows)


def _make_retry_histogram(
    retry_levels: list[int] | None = None,
    counts: list[int] | None = None,
) -> pd.DataFrame:
    """Synthetic retry_histogram DataFrame matching actual parquet schema.

    Actual schema: iteration, phase, stage, retry_level, count.
    The chart function aggregates by retry_level, summing count.
    """
    if retry_levels is None:
        retry_levels = [0, 0, 1, 1, 3]
    if counts is None:
        counts = [50, 50, 10, 10, 5]
    n = len(retry_levels)
    return pd.DataFrame(
        {
            "iteration": list(range(n)),
            "phase": ["forward"] * n,
            "stage": [0] * n,
            "retry_level": retry_levels,
            "count": counts,
        }
    )


def _make_mock_data(
    solver_train: pd.DataFrame | None = None,
    solver_sim: pd.DataFrame | None = None,
    timing: pd.DataFrame | None = None,
    metadata: dict | None = None,
    simulation_manifest: dict | None = None,
    scaling_report: dict | None = None,
    retry_histogram: pd.DataFrame | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics DashboardData with real DataFrames."""
    data = MagicMock()
    data.solver_train = solver_train if solver_train is not None else _make_solver_df()
    data.solver_sim = solver_sim if solver_sim is not None else _make_solver_df()
    data.timing = timing if timing is not None else _make_timing()
    data.metadata = metadata if metadata is not None else {}
    data.simulation_manifest = (
        simulation_manifest if simulation_manifest is not None else {}
    )
    data.scaling_report = scaling_report if scaling_report is not None else {}
    data.retry_histogram = (
        retry_histogram if retry_histogram is not None else pd.DataFrame()
    )
    return data


# ---------------------------------------------------------------------------
# test_tab_constants
# ---------------------------------------------------------------------------


def test_tab_constants() -> None:
    """TAB_ID, TAB_LABEL, and TAB_ORDER must match the spec."""
    assert TAB_ID == "tab-v2-performance"
    assert TAB_LABEL == "Performance"
    assert TAB_ORDER == 90


# ---------------------------------------------------------------------------
# test_can_render_returns_true
# ---------------------------------------------------------------------------


def test_can_render_returns_true() -> None:
    """can_render() must always return True regardless of data."""
    data = _make_mock_data()
    assert can_render(data) is True


# ---------------------------------------------------------------------------
# test_format_time_*
# ---------------------------------------------------------------------------


def test_format_time_seconds() -> None:
    """_format_time(45.0) must return '45.0 s'."""
    assert _format_time(45.0) == "45.0 s"


def test_format_time_minutes() -> None:
    """_format_time(125.0) must return '2.1 min'."""
    assert _format_time(125.0) == "2.1 min"


def test_format_time_hours() -> None:
    """_format_time(7200.0) must return '2.00 h'."""
    assert _format_time(7200.0) == "2.00 h"


def test_format_time_boundary_60s() -> None:
    """_format_time(60.0) must switch to minutes at exactly 60 s."""
    result = _format_time(60.0)
    assert "min" in result


def test_format_time_boundary_3600s() -> None:
    """_format_time(3600.0) must switch to hours at exactly 3600 s."""
    result = _format_time(3600.0)
    assert "h" in result


# ---------------------------------------------------------------------------
# test_build_metrics_row_with_data
# ---------------------------------------------------------------------------


def test_build_metrics_row_with_data() -> None:
    """_build_metrics_row() with synthetic solver data contains expected labels."""
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=5, lp_solves=10, solve_time_ms=50.0),
        solver_sim=_make_solver_df(n_rows=5, lp_solves=10, solve_time_ms=50.0),
    )
    html = _build_metrics_row(data)
    assert "Avg LP Solve" in html
    assert "Total LP Solves" in html
    assert "Total Simplex Iterations" in html
    assert "metrics-grid" in html


def test_build_metrics_row_lp_solve_avg_value() -> None:
    """Avg LP Solve (training): solve_time / lp_solves across all rows."""
    # 5 rows * 50 ms total / (5 rows * 10 lp_solves) = 1.00 ms avg
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=5, lp_solves=10, solve_time_ms=50.0),
        solver_sim=pd.DataFrame(),
    )
    html = _build_metrics_row(data)
    # total_lp_time=250, total_lp_solves=50, avg=5.00 ms
    assert "5.00 ms" in html


def test_build_metrics_row_total_lp_solves_value() -> None:
    """Total LP Solves must be the sum from both solver_train and solver_sim."""
    train = _make_solver_df(n_rows=3, lp_solves=10)  # 30 total
    sim = _make_solver_df(n_rows=2, lp_solves=5)  # 10 total → 40 combined
    data = _make_mock_data(solver_train=train, solver_sim=sim)
    html = _build_metrics_row(data)
    assert "40" in html


# ---------------------------------------------------------------------------
# test_build_metrics_row_empty_solver
# ---------------------------------------------------------------------------


def test_build_metrics_row_empty_solver() -> None:
    """Empty solver DataFrames must not crash and must show zeros."""
    data = _make_mock_data(
        solver_train=pd.DataFrame(),
        solver_sim=pd.DataFrame(),
    )
    html = _build_metrics_row(data)
    # Both avg LP solve values should be 0.00 ms
    assert "0.00 ms" in html
    # Total LP Solves = 0
    assert ">0<" in html or "0,</div>" in html or ">0</div>" in html


def test_build_metrics_row_empty_solver_no_crash() -> None:
    """Empty solver DataFrames must produce a valid metrics-grid HTML string."""
    data = _make_mock_data(
        solver_train=pd.DataFrame(),
        solver_sim=pd.DataFrame(),
    )
    html = _build_metrics_row(data)
    assert "metrics-grid" in html
    assert len(html) > 0


# ---------------------------------------------------------------------------
# test_build_metrics_row_metadata_training_time
# ---------------------------------------------------------------------------


def test_build_metrics_row_uses_metadata_duration() -> None:
    """metadata duration_seconds takes priority over timing fallback."""
    data = _make_mock_data(
        metadata={"run_info": {"duration_seconds": 3661.0}},
        timing=pd.DataFrame(),  # fallback would give 0 s
    )
    html = _build_metrics_row(data)
    # 3661 s → "1.02 h"
    assert "1.02 h" in html


def test_build_metrics_row_uses_simulation_manifest_duration() -> None:
    """simulation_manifest duration_seconds causes 'Total Simulation Time' label."""
    data = _make_mock_data(
        simulation_manifest={"duration_seconds": 120.0},
        solver_sim=pd.DataFrame(),  # fallback would give 0 s
    )
    html = _build_metrics_row(data)
    assert "Total Simulation Time" in html
    assert "2.0 min" in html


def test_build_metrics_row_fallback_simulation_label() -> None:
    """When simulation_manifest has no duration, label says 'Simulation CPU Time'."""
    data = _make_mock_data(
        simulation_manifest={},
        solver_sim=_make_solver_df(n_rows=2, solve_time_ms=30000.0),
    )
    html = _build_metrics_row(data)
    assert "Simulation CPU Time" in html


# ---------------------------------------------------------------------------
# test_render_empty_timing
# ---------------------------------------------------------------------------


def test_render_empty_timing() -> None:
    """render() with empty timing DataFrame must include 'No timing data'."""
    data = _make_mock_data(timing=pd.DataFrame())
    html = render(data)
    assert "No timing data" in html


# ---------------------------------------------------------------------------
# test_render_with_data
# ---------------------------------------------------------------------------


def test_render_with_data() -> None:
    """render() with all synthetic data must contain required section strings."""
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=5),
        solver_sim=_make_solver_df(n_rows=5),
        timing=_make_timing(n=5),
    )
    html = render(data)
    assert "Training Iteration Breakdown" in html
    assert "Timing" in html
    assert "metrics-grid" in html


def test_render_contains_run_summary_section() -> None:
    """render() must produce a 'Run Summary' section title."""
    data = _make_mock_data()
    html = render(data)
    assert "Run Summary" in html


def test_render_no_exception_full_data() -> None:
    """render() with all fields populated must not raise."""
    data = _make_mock_data(
        solver_train=_make_solver_df(n_rows=10),
        solver_sim=_make_solver_df(n_rows=10),
        timing=_make_timing(n=10),
        metadata={"run_info": {"duration_seconds": 300.0}},
        simulation_manifest={"duration_seconds": 60.0},
    )
    html = render(data)
    assert len(html) > 0


# ---------------------------------------------------------------------------
# test_chart_retry_histogram_*
# ---------------------------------------------------------------------------


def test_chart_retry_histogram_with_data() -> None:
    """_chart_retry_histogram() with non-empty DataFrame returns HTML with 'plotly'."""
    df = _make_retry_histogram()
    html = _chart_retry_histogram(df)
    assert "plotly" in html.lower()


def test_chart_retry_histogram_empty() -> None:
    """_chart_retry_histogram() with empty DataFrame returns fallback message."""
    html = _chart_retry_histogram(pd.DataFrame())
    assert "No retry data" in html


def test_chart_retry_histogram_color_coding() -> None:
    """_chart_retry_histogram() with mixed retry counts (0, 1, 2, 5) does not raise."""
    df = _make_retry_histogram(
        retry_levels=[0, 1, 2, 5],
        counts=[80, 15, 10, 3],
    )
    html = _chart_retry_histogram(df)
    assert "plotly" in html.lower()


# ---------------------------------------------------------------------------
# test_render_full_sections (ticket-021)
# ---------------------------------------------------------------------------


def test_render_full_sections() -> None:
    """render() with all non-empty data must contain all section titles added by ticket-021."""
    scaling_report = {
        "stages": [
            {
                "stage_id": 0,
                "dimensions": {"num_cols": 100, "num_rows": 80, "num_nz": 500},
                "pre_scaling": {"matrix_coeff_ratio": 1000.0},
                "post_scaling": {"matrix_coeff_ratio": 10.0},
            }
        ]
    }
    data = _make_mock_data(
        solver_train=_make_solver_train(n_stages=3, n_iters=2),
        solver_sim=_make_solver_df(n_rows=3),
        timing=_make_timing(n=3),
        scaling_report=scaling_report,
        retry_histogram=_make_retry_histogram(),
    )
    html = render(data)
    assert "Solver Time Breakdown" in html
    assert "LP Solver Detail" in html
    assert "Solver Overhead" in html
    assert "LP Dimensions" in html
    assert "Solver Efficiency" in html
    assert "Solver Retries" in html
    assert "Simulation" in html
    # Sections from ticket-020 still present
    assert "Training Iteration Breakdown" in html
    assert "Timing Waterfall" in html
    assert "Run Summary" in html


# ---------------------------------------------------------------------------
# test_render_empty_solver (ticket-021)
# ---------------------------------------------------------------------------


def test_render_empty_solver() -> None:
    """render() with empty solver data must not raise and show degradation messages."""
    data = _make_mock_data(
        solver_train=pd.DataFrame(),
        solver_sim=pd.DataFrame(),
        timing=pd.DataFrame(),
        scaling_report={},
        retry_histogram=pd.DataFrame(),
    )
    html = render(data)
    assert len(html) > 0
    # Graceful degradation: at least one of the expected fallback strings appears
    assert (
        "No solver data" in html
        or "No scaling report" in html
        or "No retry data" in html
    )
