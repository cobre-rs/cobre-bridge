"""Unit tests for the unified ``--json`` verdict envelope (``cobre_bridge.verdict``)."""

from __future__ import annotations

import json

import polars as pl
import pytest

from cobre_bridge.comparators.decomp_results import DecompComparison
from cobre_bridge.comparators.verdict import CompareVerdict
from cobre_bridge.diagnostics import Diagnostic, Severity
from cobre_bridge.verdict import (
    SCHEMA_VERSION,
    build_verdict,
    check_summary,
    compare_summary,
    convert_summary,
    dashboard_summary,
    decomp_compare_summary,
)


def _fake_decomp_comparison() -> DecompComparison:
    """A tiny two-row comparison of the source model against the converted case."""
    rows = pl.DataFrame(
        {
            "level": ["hydro", "thermal"],
            "variable": ["generation", "generation"],
            "unit": ["MW", "MW"],
            "entity_id": [0, 1],
            "entity_name": ["A", "B"],
            "stage_id": [0, 1],
            "source": [100.0, 20.0],
            "cobre": [90.0, 20.0],
            "delta": [-10.0, 0.0],
            "delta_pct": [-10.0, 0.0],
            "smape_pct": [10.5, 0.0],
        }
    )
    summary = pl.DataFrame(
        {
            "level": ["hydro", "thermal"],
            "variable": ["generation", "generation"],
            "unit": ["MW", "MW"],
            "n": [1, 1],
            "source_total": [100.0, 20.0],
            "cobre_total": [90.0, 20.0],
            "delta_total": [-10.0, 0.0],
            "delta_total_pct": [-10.0, 0.0],
            "smape_pct": [10.5, 0.0],
            "worst_entity": ["A", "B"],
            "worst_delta": [-10.0, 0.0],
        }
    )
    return DecompComparison(
        rows=rows,
        summary=summary,
        convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
        unmapped={"hydro": [7], "thermal": [], "bus": []},
    )


def _empty_decomp_comparison() -> DecompComparison:
    """A comparison with no aligned rows (nothing on both sides mapped)."""
    empty_rows = pl.DataFrame(
        schema={
            "level": pl.Utf8,
            "variable": pl.Utf8,
            "unit": pl.Utf8,
            "entity_id": pl.Int64,
            "entity_name": pl.Utf8,
            "stage_id": pl.Int64,
            "source": pl.Float64,
            "cobre": pl.Float64,
            "delta": pl.Float64,
            "delta_pct": pl.Float64,
            "smape_pct": pl.Float64,
        }
    )
    empty_summary = pl.DataFrame(
        schema={
            "level": pl.Utf8,
            "variable": pl.Utf8,
            "unit": pl.Utf8,
            "n": pl.UInt32,
            "source_total": pl.Float64,
            "cobre_total": pl.Float64,
            "delta_total": pl.Float64,
            "delta_total_pct": pl.Float64,
            "smape_pct": pl.Float64,
            "worst_entity": pl.Utf8,
            "worst_delta": pl.Float64,
        }
    )
    return DecompComparison(
        rows=empty_rows,
        summary=empty_summary,
        convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
        unmapped={"hydro": [], "thermal": [], "bus": []},
    )


def _info_diagnostic() -> Diagnostic:
    """A minimal INFO diagnostic, mirroring the pattern in ``tests/test_cli.py``."""
    return Diagnostic(
        code="some-info",
        severity=Severity.INFO,
        category="Conversion",
        title="An info",
        summary="just so",
    )


class TestBuildVerdict:
    def test_build_verdict_key_order_and_schema_version(self) -> None:
        doc = build_verdict("convert newave", "ok", {"hydros": 1})

        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["schema_version"] == SCHEMA_VERSION
        assert doc["command"] == "convert newave"
        assert doc["status"] == "ok"
        assert doc["summary"] == {"hydros": 1}
        # Empty diagnostics (the default empty tuple) serializes to ``[]``.
        assert doc["diagnostics"] == []

    def test_build_verdict_serializes_diagnostics(self) -> None:
        diag = _info_diagnostic()

        doc = build_verdict("convert newave", "error", {}, [diag])

        assert doc["diagnostics"] == [diag.to_dict()]
        # The builder copies the passed mapping and injects no keys.
        assert doc["summary"] == {}

    def test_build_verdict_copies_summary(self) -> None:
        summary = convert_summary(1, 1, 1, 1, 1)

        doc = build_verdict("convert newave", "ok", summary)
        returned_summary = doc["summary"]
        assert isinstance(returned_summary, dict)
        returned_summary["hydros"] = 999

        # Mutating the returned summary must not touch the caller's mapping.
        assert summary["hydros"] == 1


class TestSummaryHelpers:
    def test_convert_summary_key_order(self) -> None:
        result = convert_summary(10, 5, 4, 3, 60)

        assert list(result.keys()) == [
            "hydros",
            "thermals",
            "buses",
            "lines",
            "stages",
        ]
        assert result == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
        }

    def test_check_summary_wraps_checks(self) -> None:
        rows = [
            {"label": "caso.dat present", "passed": True, "detail": "found"},
            {"label": "hidr.dat present", "passed": False, "detail": "missing"},
        ]

        result = check_summary(rows)

        assert list(result.keys()) == ["checks"]
        assert result["checks"] == rows
        # Each row is copied, not aliased: mutating the source leaves the result.
        rows[0]["passed"] = False
        wrapped = result["checks"]
        assert isinstance(wrapped, list)
        assert wrapped[0]["passed"] is True

    def test_dashboard_summary_shape(self) -> None:
        result = dashboard_summary("dashboard.html", 12.5)

        assert list(result.keys()) == ["output", "size_kb"]
        assert result["output"] == "dashboard.html"
        assert result["size_kb"] == 12.5
        assert isinstance(result["output"], str)
        assert isinstance(result["size_kb"], float)


class TestCompareSummary:
    def test_compare_summary_nulls_worst_when_all_within_tol(self) -> None:
        verdict = CompareVerdict(
            within_tol=2,
            total=2,
            worst_variable="storage",
            worst_smape=0.04,
            all_within_tol=True,
        )

        result = compare_summary(verdict)

        assert list(result.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
        ]
        assert result["all_within_tol"] is True
        assert result["worst_variable"] is None
        assert result["worst_smape"] == 0.0

    def test_compare_summary_passes_worst_when_divergent(self) -> None:
        verdict = CompareVerdict(
            within_tol=1,
            total=2,
            worst_variable="storage",
            worst_smape=0.04,
            all_within_tol=False,
        )

        result = compare_summary(verdict)

        assert result["all_within_tol"] is False
        assert result["worst_variable"] == "storage"
        assert result["worst_smape"] == pytest.approx(0.04)


class TestDecompCompareSummary:
    def test_two_rows_returns_key_order_and_values(self) -> None:
        comparison = _fake_decomp_comparison()

        result = decomp_compare_summary(comparison, tolerance=1e-2)

        assert list(result.keys()) == [
            "stages",
            "variables",
            "unmapped",
            "within_tol",
            "total",
            "all_within_tol",
        ]
        assert result["stages"] == comparison.stage_count
        assert result["variables"] == comparison.summary.to_dicts()
        assert result["unmapped"] == {"hydro": [7], "thermal": [], "bus": []}

    def test_two_rows_round_trips_through_json(self) -> None:
        comparison = _fake_decomp_comparison()
        result = decomp_compare_summary(comparison, tolerance=1e-2)

        round_tripped = json.loads(json.dumps(result))

        assert round_tripped == result

    def test_empty_comparison_returns_zero_stages_and_no_variables(self) -> None:
        comparison = _empty_decomp_comparison()

        result = decomp_compare_summary(comparison, tolerance=1e-2)

        assert result["stages"] == 0
        assert result["variables"] == []
        assert result["unmapped"] == {"hydro": [], "thermal": [], "bus": []}
        assert result["within_tol"] == 0
        assert result["total"] == 0
        assert result["all_within_tol"] is False
