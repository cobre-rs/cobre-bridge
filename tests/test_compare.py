"""Integration tests for the compare bounds and compare results pipelines.

Tests the full comparison flow using mocked inewave readers and fixture
data.  Verifies that the pipeline connects correctly from CLI through
alignment, computation, comparison, and report generation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import polars as pl
import pyarrow as pa
import pytest
from plotly.offline import get_plotlyjs_version

from cobre_bridge.comparators.bounds import (
    BoundComparison,
    _bounds_match,
    _is_effectively_infinite,
)
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    build_results_summary,
)
from tests.conftest import make_case

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.diagnostics import Diagnostic

# -------------------------------------------------------------------
# Bounds comparison unit tests
# -------------------------------------------------------------------


class TestBoundsHelpers:
    def test_is_effectively_infinite_big_m(self) -> None:
        assert _is_effectively_infinite(99999.0)
        assert _is_effectively_infinite(99990.0)
        assert not _is_effectively_infinite(99989.0)

    def test_is_effectively_infinite_ieee(self) -> None:
        assert _is_effectively_infinite(float("inf"))
        assert _is_effectively_infinite(float("-inf"))

    def test_is_effectively_infinite_normal(self) -> None:
        assert not _is_effectively_infinite(0.0)
        assert not _is_effectively_infinite(1000.0)

    def test_bounds_match_within_tolerance(self) -> None:
        assert _bounds_match(10.0, 10.0005, 1e-3)
        assert not _bounds_match(10.0, 10.002, 1e-3)

    def test_bounds_match_both_inf(self) -> None:
        assert _bounds_match(float("inf"), float("inf"), 1e-3)
        assert not _bounds_match(float("inf"), float("-inf"), 1e-3)

    def test_bounds_match_one_inf(self) -> None:
        assert not _bounds_match(float("inf"), 10.0, 1e-3)


# -------------------------------------------------------------------
# Results comparison unit tests
# -------------------------------------------------------------------


class TestResultsComparison:
    def test_build_results_summary_empty(self) -> None:
        summary = build_results_summary([])
        assert summary.total == 0
        assert summary.by_entity_type == {}
        assert summary.by_variable == {}

    def test_build_results_summary_counts(self) -> None:
        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="A",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=99.5,
                abs_diff=0.5,
                rel_diff=0.005,
            ),
            ResultComparison(
                entity_type="thermal",
                entity_name="B",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=50.0,
                cobre_value=50.1,
                abs_diff=0.1,
                rel_diff=0.002,
            ),
        ]
        summary = build_results_summary(results)
        assert summary.total == 2
        assert summary.by_entity_type["hydro"] == 1
        assert summary.by_entity_type["thermal"] == 1
        assert "generation_mw" in summary.by_variable
        stats = summary.by_variable["generation_mw"]
        assert stats.count == 2
        assert stats.max_abs_diff == 0.5


# -------------------------------------------------------------------
# Report formatting tests
# -------------------------------------------------------------------


class TestReportFormatting:
    def test_print_results_summary_no_crash(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report import (
            print_results_summary_from_dataset,
        )
        from cobre_bridge.comparators.results import PercentileData

        dataset = build_results_dataset([], PercentileData(), 1e-2)
        print_results_summary_from_dataset(
            dataset, Path("/fake/nw"), Path("/fake/cobre")
        )
        out = capsys.readouterr().out
        assert "Results Comparison" in out

    def test_print_results_summary_with_data(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report import (
            print_results_summary_from_dataset,
        )
        from cobre_bridge.comparators.results import (
            PercentileData,
            ResultComparison,
        )

        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=110.0,
                abs_diff=10.0,
                rel_diff=0.1,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="TUCURUI",
                newave_code=20,
                cobre_id=1,
                stage=1,
                variable="generation_mw",
                newave_value=50.0,
                cobre_value=40.0,
                abs_diff=10.0,
                rel_diff=0.2,
            ),
        ]
        dataset = build_results_dataset(results, PercentileData(), 1e-2)
        print_results_summary_from_dataset(dataset, Path("/nw"), Path("/cobre"))
        out = capsys.readouterr().out
        assert "generation_mw" in out
        assert "2" in out  # the per-variable Count cell


# -------------------------------------------------------------------
# Tolerance row-colouring tests (epic-03, ticket-010)
# -------------------------------------------------------------------


def _make_bounds_for_colour() -> list[BoundComparison]:
    """Four bound comparisons across two variables, mixing match/mismatch.

    ``storage_max`` and ``generation_max`` each carry one match and one mismatch,
    so both variable rows — and the Total — are expected red.
    """
    return [
        BoundComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            cobre_id=0,
            stage=0,
            variable="storage_max",
            newave_value=29000.0,
            cobre_value=29000.0,
            diff=0.0,
            match=True,
        ),
        BoundComparison(
            entity_type="hydro",
            entity_name="TUCURUI",
            newave_code=20,
            cobre_id=1,
            stage=0,
            variable="storage_max",
            newave_value=50000.0,
            cobre_value=49000.0,
            diff=1000.0,
            match=False,
        ),
        BoundComparison(
            entity_type="thermal",
            entity_name="ANGRA",
            newave_code=30,
            cobre_id=2,
            stage=0,
            variable="generation_max",
            newave_value=1350.0,
            cobre_value=1350.0,
            diff=0.0,
            match=True,
        ),
        BoundComparison(
            entity_type="thermal",
            entity_name="CUIABA",
            newave_code=40,
            cobre_id=3,
            stage=1,
            variable="generation_max",
            newave_value=500.0,
            cobre_value=450.0,
            diff=50.0,
            match=False,
        ),
    ]


class _MakeTableSpy:
    """Spy that records every ``make_table`` call and delegates to the real one.

    Patched in as ``make_table``'s ``side_effect`` so the printer still renders a
    genuine table (keeping the parity contract honest) while the test inspects the
    ``row_styles`` kwarg passed to each call.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        from cobre_bridge.ui.console import make_table

        self.records.append(dict(kwargs))
        return make_table(*args, **kwargs)  # type: ignore[arg-type]


class TestToleranceRowColouring:
    """Compare summary table rows are coloured green/red by tolerance."""

    def test_results_row_colour_green_for_within_tol_red_otherwise(self) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report import (
            print_results_summary_from_dataset,
        )
        from cobre_bridge.ui.console import compare_row_style

        # gen_within is fully within tol (rel_diff under 1e-2); gen_diverge is not.
        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="gen_within",
                newave_value=100.0,
                cobre_value=100.0,
                abs_diff=0.0,
                rel_diff=0.0,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="TUCURUI",
                newave_code=20,
                cobre_id=1,
                stage=0,
                variable="gen_diverge",
                newave_value=100.0,
                cobre_value=150.0,
                abs_diff=50.0,
                rel_diff=0.5,
            ),
        ]
        dataset = build_results_dataset(results, PercentileData(), 1e-2)

        spy = _MakeTableSpy()
        with patch("cobre_bridge.comparators.report.make_table", side_effect=spy):
            print_results_summary_from_dataset(dataset, Path("/nw"), Path("/cobre"))

        # Rows are sorted by variable; derive the expected colour FROM the dataset
        # summary (never recompute tolerance) so the source-of-truth stays single.
        summary = {row["variable"]: row for row in dataset.summary.to_dicts()}
        expected = [
            compare_row_style(within_tol=float(summary[var]["within_tol_rate"]) == 1.0)
            for var in sorted(summary)
        ]
        assert spy.records[0]["row_styles"] == expected
        # gen_diverge sorts before gen_within: red then green.
        assert expected[0] == compare_row_style(within_tol=False)  # gen_diverge
        assert expected[1] == compare_row_style(within_tol=True)  # gen_within

    def test_bounds_row_colour_total_row_red_on_mismatch(self) -> None:
        from cobre_bridge.comparators.analyze import build_bounds_dataset
        from cobre_bridge.comparators.report import (
            print_bounds_summary_from_dataset,
        )
        from cobre_bridge.ui.console import compare_row_style

        dataset = build_bounds_dataset(_make_bounds_for_colour())

        spy = _MakeTableSpy()
        with patch("cobre_bridge.comparators.report.make_table", side_effect=spy):
            print_bounds_summary_from_dataset(
                dataset, Path("/nw"), Path("/cobre"), 1e-3
            )

        # Two make_table calls: by-entity-type (with Total) then by-variable.
        type_styles = spy.records[0]["row_styles"]
        var_styles = spy.records[1]["row_styles"]

        counts = dataset.metadata["summary_counts"]
        assert isinstance(counts, dict)
        mismatches_all = counts["mismatches"]
        by_entity_type = counts["by_entity_type"]
        by_variable = counts["by_variable"]
        assert isinstance(by_entity_type, dict)
        assert isinstance(by_variable, dict)

        # By-type styles: one per type (sorted) + the trailing Total row.
        expected_type = [
            compare_row_style(within_tol=by_entity_type[etype][1] == 0)
            for etype in sorted(by_entity_type)
        ]
        expected_type.append(compare_row_style(within_tol=mismatches_all == 0))
        assert type_styles == expected_type

        # The Total row (last entry) is red: the fixture has mismatches.
        assert type_styles[-1] == compare_row_style(within_tol=False)
        assert mismatches_all != 0

        # By-variable styles derived from summary_counts mismatch flags.
        expected_var = [
            compare_row_style(within_tol=by_variable[var][1] == 0)
            for var in sorted(by_variable)
        ]
        assert var_styles == expected_var
        # Both fixture variables have a mismatch -> both red.
        assert all(s == compare_row_style(within_tol=False) for s in var_styles)


# -------------------------------------------------------------------
# Compare verdict line tests (epic-03, ticket-009)
# -------------------------------------------------------------------


def _first_nonempty_line(text: str) -> str:
    """Return the first non-blank line of *text*, stripped."""
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            return line
    raise AssertionError("no non-empty output line")


class TestCompareVerdictLine:
    """The verdict headline leads both compare summary printers (stdout)."""

    def test_verdict_line_leads_bounds_with_worst_clause(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.comparators.analyze import build_bounds_dataset
        from cobre_bridge.comparators.bounds import BoundComparison
        from cobre_bridge.comparators.report import (
            print_bounds_summary_from_dataset,
        )

        # Two variables, both with at least one mismatch -> 0/2 within tol.
        results = [
            BoundComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="storage_max",
                newave_value=29000.0,
                cobre_value=29000.0,
                diff=0.0,
                match=True,
            ),
            BoundComparison(
                entity_type="hydro",
                entity_name="TUCURUI",
                newave_code=20,
                cobre_id=1,
                stage=0,
                variable="storage_max",
                newave_value=50000.0,
                cobre_value=49000.0,
                diff=1000.0,
                match=False,
            ),
            BoundComparison(
                entity_type="thermal",
                entity_name="ANGRA",
                newave_code=30,
                cobre_id=2,
                stage=0,
                variable="generation_max",
                newave_value=1350.0,
                cobre_value=1350.0,
                diff=0.0,
                match=True,
            ),
            BoundComparison(
                entity_type="thermal",
                entity_name="CUIABA",
                newave_code=40,
                cobre_id=3,
                stage=1,
                variable="generation_max",
                newave_value=500.0,
                cobre_value=450.0,
                diff=50.0,
                match=False,
            ),
        ]
        dataset = build_bounds_dataset(results)

        print_bounds_summary_from_dataset(
            dataset, Path("/fake/nw"), Path("/fake/cobre"), 1e-3
        )

        first = _first_nonempty_line(capsys.readouterr().out)
        assert first.startswith("⚠ ")
        assert "0/2 variables within tol" in first
        assert "worst:" in first

    def test_verdict_line_all_within_tol_no_worst_clause(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.comparators.analyze import build_bounds_dataset
        from cobre_bridge.comparators.bounds import BoundComparison
        from cobre_bridge.comparators.report import (
            print_bounds_summary_from_dataset,
        )

        # Every comparison matches -> all variables within tol, no worst clause.
        results = [
            BoundComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="storage_max",
                newave_value=29000.0,
                cobre_value=29000.0,
                diff=0.0,
                match=True,
            ),
            BoundComparison(
                entity_type="thermal",
                entity_name="ANGRA",
                newave_code=30,
                cobre_id=1,
                stage=0,
                variable="generation_max",
                newave_value=1350.0,
                cobre_value=1350.0,
                diff=0.0,
                match=True,
            ),
        ]
        dataset = build_bounds_dataset(results)
        variable_count = dataset.summary.height

        print_bounds_summary_from_dataset(
            dataset, Path("/fake/nw"), Path("/fake/cobre"), 1e-3
        )

        first = _first_nonempty_line(capsys.readouterr().out)
        assert first.startswith("✓ ")
        assert first == f"✓ {variable_count}/{variable_count} variables within tol"
        assert "worst:" not in first

    def test_render_compare_verdict_empty_dataset(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.comparators.verdict import CompareVerdict
        from cobre_bridge.ui.console import render_compare_verdict

        render_compare_verdict(
            CompareVerdict(
                within_tol=0,
                total=0,
                worst_variable=None,
                worst_smape=0.0,
                all_within_tol=False,
            )
        )

        assert (
            _first_nonempty_line(capsys.readouterr().out) == "⚠ no variables compared"
        )


class TestCompareVerdictExitCodes:
    """The verdict line does not change the compare exit-code contract."""

    @staticmethod
    def _invoke_main(
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        import io
        import sys

        from cobre_bridge import cli

        monkeypatch.setattr(sys, "argv", ["cobre-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _patch_compare_context(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cobre_bridge.case.NewaveCase.from_directory",
            classmethod(lambda cls, _dir: MagicMock(id_map=MagicMock())),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.alignment.build_entity_alignment",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "cobre_bridge.cli._load_lines_json",
            lambda _dir: [],
        )

    def test_verdict_results_exit_zero_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset

        self._patch_compare_context(monkeypatch)
        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=110.0,
                abs_diff=10.0,
                rel_diff=0.1,
            ),
        ]
        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            lambda **k: build_results_dataset(results, PercentileData(), 1e-2),
        )
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        # compare results is informational -> always exits 0.
        assert code == 0
        # The verdict line is a primary result: on stdout, not stderr.
        verdict_line = _first_nonempty_line(stdout)
        assert verdict_line.startswith(("✓ ", "⚠ "))
        assert "variables within tol" in verdict_line
        assert "variables within tol" not in stderr


# -------------------------------------------------------------------
# Compare diagnostics wiring (dx.collect() sink -> render + verdict)
# -------------------------------------------------------------------


def _fake_results_dataset() -> ComparisonDataset:
    """One-row dataset built through the shared dataset-assembly kernel."""
    from cobre_bridge.comparators.analyze import build_results_dataset

    results = [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            cobre_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            cobre_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
    ]
    return build_results_dataset(results, PercentileData(), 1e-2)


def _fake_decomp_dataset() -> ComparisonDataset:
    """Adds the ``metadata["unmapped"]`` key ``decomp_dataset_summary`` reads."""
    dataset = _fake_results_dataset()
    dataset.metadata["unmapped"] = {"hydro": [], "thermal": [], "bus": []}
    return dataset


def _dropped_plant_diagnostic() -> Diagnostic:
    from cobre_bridge.diagnostics import Diagnostic, Severity

    return Diagnostic(
        code="test-dropped-plant",
        severity=Severity.WARNING,
        category="Test",
        title="Dropped test plant",
        summary="one plant was dropped for diagnostics-wiring coverage",
    )


class TestCompareDiagnosticsWiring:
    """A ``Severity.WARNING`` emit() during dataset building must reach both
    the ``--json`` envelope's ``diagnostics`` array and the non-``--json``
    stderr render, on both compare tracks."""

    @staticmethod
    def _invoke(argv: list[str]) -> Any:
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        return CliRunner().invoke(app, argv)

    def test_compare_newave_json_diagnostics_array_is_non_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge import diagnostics as dx

        TestCompareVerdictExitCodes._patch_compare_context(monkeypatch)
        diagnostic = _dropped_plant_diagnostic()

        def _compare_results_with_diagnostic(**_kwargs: object) -> ComparisonDataset:
            dx.emit(diagnostic)
            return _fake_results_dataset()

        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            _compare_results_with_diagnostic,
        )
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        result = self._invoke(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir), "--json"]
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["diagnostics"]
        assert payload["diagnostics"][0]["code"] == diagnostic.code

    def test_compare_decomp_json_diagnostics_array_is_non_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge import diagnostics as dx

        diagnostic = _dropped_plant_diagnostic()

        def _build_decomp_dataset_with_diagnostic(
            *_args: object, **_kwargs: object
        ) -> ComparisonDataset:
            dx.emit(diagnostic)
            return _fake_decomp_dataset()

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.build_decomp_dataset",
            _build_decomp_dataset_with_diagnostic,
        )

        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"]
        )

        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["diagnostics"]
        assert payload["diagnostics"][0]["code"] == diagnostic.code

    def test_compare_newave_non_json_renders_diagnostic_title_on_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge import diagnostics as dx

        TestCompareVerdictExitCodes._patch_compare_context(monkeypatch)
        diagnostic = _dropped_plant_diagnostic()

        def _compare_results_with_diagnostic(**_kwargs: object) -> ComparisonDataset:
            dx.emit(diagnostic)
            return _fake_results_dataset()

        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            _compare_results_with_diagnostic,
        )
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        result = self._invoke(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)]
        )

        assert result.exit_code == 0
        assert diagnostic.title in result.stderr

    def test_compare_decomp_non_json_renders_diagnostic_title_on_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge import diagnostics as dx

        diagnostic = _dropped_plant_diagnostic()

        def _build_decomp_dataset_with_diagnostic(
            *_args: object, **_kwargs: object
        ) -> ComparisonDataset:
            dx.emit(diagnostic)
            return _fake_decomp_dataset()

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.build_decomp_dataset",
            _build_decomp_dataset_with_diagnostic,
        )

        result = self._invoke(["compare", "decomp", str(tmp_path), str(tmp_path)])

        assert result.exit_code == 0
        assert diagnostic.title in result.stderr


# -------------------------------------------------------------------
# HTML report tests
# -------------------------------------------------------------------


class TestHtmlReport:
    def test_build_comparison_html_produces_valid_html(self) -> None:
        from cobre_bridge.comparators.html_report import (
            build_comparison_html,
        )

        html = build_comparison_html(
            title="Test Report",
            tab_contents={"tab-overview": "<p>Hello</p>"},
        )
        assert "<!DOCTYPE html>" in html
        assert "Test Report" in html
        assert f"plotly-{get_plotlyjs_version()}.min.js" in html
        assert "<p>Hello</p>" in html

    def test_build_comparison_report_no_crash(self) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )
        from cobre_bridge.comparators.results import PercentileData

        pct = PercentileData()
        dataset = build_results_dataset([], pct, 0.05)
        html = build_comparison_report(dataset)
        assert "<!DOCTYPE html>" in html
        assert "Cobre vs NEWAVE" in html

    def test_build_comparison_report_with_data(self) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )
        from cobre_bridge.comparators.results import PercentileData

        results = [
            ResultComparison(
                entity_type="convergence",
                entity_name="iter_1",
                newave_code=1,
                cobre_id=1,
                stage=1,
                variable="lower_bound",
                newave_value=1000.0,
                cobre_value=1001.0,
                abs_diff=1.0,
                rel_diff=0.001,
            ),
        ]
        pct = PercentileData()
        dataset = build_results_dataset(results, pct, 0.05)
        html = build_comparison_report(dataset)
        assert "<!DOCTYPE html>" in html
        assert "Convergence" in html


# -------------------------------------------------------------------
# Bounds from inputs tests
# -------------------------------------------------------------------


class TestBoundsFromInputs:
    """The bounds 'expected' side now delegates to the converters (ARCH-01).

    These verify the delegation + reshape into the comparison dict, not an
    independent re-derivation (which drifted and caused false positives).
    """

    def test_compute_hydro_bounds_reshapes_converter_table(
        self, tmp_path: Path
    ) -> None:
        from cobre_bridge.comparators.bounds_from_inputs import compute_hydro_bounds

        table = pa.table(
            {
                "hydro_id": pa.array([0, 0], pa.int32()),
                "stage_id": pa.array([0, 1], pa.int32()),
                "min_storage_hm3": pa.array([10.0, None], pa.float64()),
                "max_storage_hm3": pa.array([100.0, 110.0], pa.float64()),
                "min_turbined_m3s": pa.array([None, None], pa.float64()),
                "max_turbined_m3s": pa.array([None, None], pa.float64()),
                "min_outflow_m3s": pa.array([5.0, None], pa.float64()),
                "min_generation_mw": pa.array([7.0, 7.0], pa.float64()),
            }
        )
        case = make_case(tmp_path)
        with patch(
            "cobre_bridge.converters.hydro.convert_storage_bounds",
            return_value=table,
        ):
            result = compute_hydro_bounds(case, MagicMock())

        assert result[(0, 0, "storage_min")] == 10.0
        assert result[(0, 0, "storage_max")] == 100.0
        assert result[(0, 1, "storage_max")] == 110.0
        assert result[(0, 0, "outflow_min")] == 5.0
        # None cells are skipped; hydro generation (GHMIN) is intentionally
        # not part of the bounds comparison.
        assert (0, 1, "storage_min") not in result
        assert not any(name == "generation_min" for (_, _, name) in result)

    def test_compute_hydro_bounds_empty_when_converter_returns_none(
        self, tmp_path: Path
    ) -> None:
        from cobre_bridge.comparators.bounds_from_inputs import compute_hydro_bounds

        case = make_case(tmp_path)
        with patch(
            "cobre_bridge.converters.hydro.convert_storage_bounds", return_value=None
        ):
            assert compute_hydro_bounds(case, MagicMock()) == {}

    def test_compute_thermal_bounds_no_expt_no_manutt(self, tmp_path: Path) -> None:
        """Empty dict when neither expt.dat nor manutt.dat present (path guard)."""
        from cobre_bridge.comparators.bounds_from_inputs import compute_thermal_bounds

        case = make_case(tmp_path)  # expt/manutt default to None
        assert compute_thermal_bounds(case, MagicMock()) == {}

    def test_compute_line_bounds_reshapes_converter_table(self, tmp_path: Path) -> None:
        from cobre_bridge.comparators.bounds_from_inputs import compute_line_bounds

        table = pa.table(
            {
                "line_id": pa.array([0], pa.int32()),
                "stage_id": pa.array([0], pa.int32()),
                "direct_mw": pa.array([500.0], pa.float64()),
                "reverse_mw": pa.array([300.0], pa.float64()),
                "block_id": pa.array([None], pa.int32()),
            }
        )
        case = make_case(tmp_path)
        with patch(
            "cobre_bridge.converters.network.convert_line_bounds",
            return_value=table,
        ):
            result = compute_line_bounds(case, MagicMock())

        assert result[(0, 0, "direct_flow_max")] == 500.0
        assert result[(0, 0, "reverse_flow_max")] == 300.0

    def test_compute_line_bounds_ignores_per_block_override_rows(
        self, tmp_path: Path
    ) -> None:
        """A per-block override row must not shadow the stage-level base row.

        Regression test: before the ``block_id is None`` filter was added,
        building the result dict by iterating every row (base row then block
        rows) let the last-seen row for a given (line_id, stage_id) silently
        overwrite the base capacity — collapsing to a block's absolute MW
        instead of the line's declared capacity.
        """
        from cobre_bridge.comparators.bounds_from_inputs import compute_line_bounds

        table = pa.table(
            {
                "line_id": pa.array([0, 0], pa.int32()),
                "stage_id": pa.array([0, 0], pa.int32()),
                "direct_mw": pa.array([500.0, 450.0], pa.float64()),
                "reverse_mw": pa.array([300.0, 270.0], pa.float64()),
                "block_id": pa.array([None, 0], pa.int32()),
            }
        )
        case = make_case(tmp_path)
        with patch(
            "cobre_bridge.converters.network.convert_line_bounds",
            return_value=table,
        ):
            result = compute_line_bounds(case, MagicMock())

        # The base row's capacity (500.0 / 300.0), not the block row's
        # (450.0 / 270.0), must survive.
        assert result[(0, 0, "direct_flow_max")] == 500.0
        assert result[(0, 0, "reverse_flow_max")] == 300.0
        assert len(result) == 2


class TestCompareHydrosProductivity:
    """Derived operational productivity = generation / turbined (m³/s)."""

    @staticmethod
    def _run():
        import polars as pl

        from cobre_bridge.comparators.results import _compare_hydros

        # stage column min = 9 → offset 9 → stages map to 0 (turb>0) and 1
        # (turb==0, must be filtered out of the productivity comparison).
        nw_hydro = pl.DataFrame(
            {
                "newave_code": [1, 1, 1, 1],
                "stage": [9, 9, 10, 10],
                "variable": ["GHIDUH", "QTURUH", "GHIDUH", "QTURUH"],
                "value": [30.0, 100.0, 0.0, 0.0],
            }
        )
        cobre_hydro = pl.DataFrame(
            {
                "entity_id": [0, 0],
                "stage_id": [0, 1],
                "generation_mw": [33.0, 0.0],
                "turbined_m3s": [100.0, 0.0],
            }
        )
        nw_names = {1: "TEST"}
        cobre_meta = {0: {"name": "TEST", "min_storage_hm3": 0.0}}
        return _compare_hydros(nw_hydro, cobre_hydro, nw_names, cobre_meta)

    def test_productivity_emitted_with_ratio_value(self) -> None:
        prod = [r for r in self._run() if r.variable == "productivity_mw_per_m3s"]
        assert len(prod) == 1  # only the turbined>0 stage
        r = prod[0]
        assert r.entity_type == "hydro"
        assert r.stage == 0
        assert r.newave_value == pytest.approx(0.3)  # 30 / 100
        assert r.cobre_value == pytest.approx(0.33)  # 33 / 100

    def test_zero_turbined_stage_filtered_out(self) -> None:
        prod = [
            r
            for r in self._run()
            if r.variable == "productivity_mw_per_m3s" and r.stage == 1
        ]
        assert prod == []  # turbined == 0 on both sides → no productivity row


class TestReconstructedCost:
    """Reconstruct the source model live immediate cost from MEDIAS × our penalties."""

    def test_read_converted_penalties(self, tmp_path: Path) -> None:
        from cobre_bridge.comparators.cobre_readers import read_converted_penalties

        (tmp_path / "penalties.json").write_text('{"hydro": {"spillage_cost": 0.5}}')
        out = tmp_path / "output"
        out.mkdir()
        # Found at the case root (parent of output).
        assert read_converted_penalties(out)["hydro"]["spillage_cost"] == 0.5
        # Absent (neither dir nor its parent has penalties.json) → empty dict.
        isolated = tmp_path / "sub"
        isolated.mkdir()
        assert read_converted_penalties(isolated / "output") == {}


class TestOverviewCostCharts:
    """Overview thermal-cost (CTERM) and other-costs (COPER − CTERM) charts."""

    @staticmethod
    def _data():
        import polars as pl

        # nw_offset will be 0 (min stage == 0). Distinctive values so the
        # substring assertions can't false-match elsewhere in the plotly JSON.
        nw_sin = pl.DataFrame(
            {
                "newave_code": [0, 0, 0, 0],
                "stage": [0, 0, 1, 1],
                "variable": ["COPER", "CTERM", "COPER", "CTERM"],
                "value": [137.0, 100.0, 70.0, 95.0],  # 10⁶ R$
            }
        )
        cobre = pl.DataFrame(
            {
                "stage_id": [0, 1],
                "immediate_cost": [150.0e6, 50.0e6],
                "future_cost": [0.0, 0.0],
                "thermal_cost": [110.0e6, 90.0e6],
                "anticipated_thermal_cost": [0.0, 0.0],
                "thermal_cost_total": [110.0e6, 90.0e6],
            }
        )
        return nw_sin, cobre

    def test_thermal_cost_chart_plots_cterm(self) -> None:
        from cobre_bridge.comparators.charts import thermal_cost_chart

        html = thermal_cost_chart(*self._data(), nw_offset=0)
        assert "No CTERM" not in html
        assert "NEWAVE CTERM" in html
        assert "100.0" in html and "95.0" in html  # CTERM values
        assert "110.0" in html  # Cobre thermal_cost / 1e6

    def test_other_costs_chart_is_coper_minus_cterm(self) -> None:
        from cobre_bridge.comparators.charts import other_costs_chart

        html = other_costs_chart(*self._data(), nw_offset=0)
        assert "No COPER" not in html
        # The source model COPER − CTERM: 137−100 = 37, 70−95 = −25 (negative, like the
        # frozen-COPER post-study gap).
        assert "37.0" in html and "-25.0" in html
        # Cobre immediate − thermal: 150−110 = 40, 50−90 = −40.
        assert "40.0" in html and "-40.0" in html

    def test_stage_costs_reader_includes_thermal_cost(self, tmp_path: Path) -> None:
        import polars as pl

        from cobre_bridge.comparators.cobre_readers import read_cobre_stage_costs

        d = tmp_path / "simulation" / "costs" / "scenario_id=0000"
        d.mkdir(parents=True)
        pl.DataFrame(
            {
                "scenario_id": [0, 0],
                "stage_id": [0, 0],
                "block_id": [0, 1],
                "immediate_cost": [10.0, 20.0],
                "future_cost": [5.0, 5.0],
                "thermal_cost": [8.0, 12.0],
            }
        ).write_parquet(d / "data.parquet")

        df = read_cobre_stage_costs(tmp_path)
        assert "thermal_cost" in df.columns
        row = df.filter(pl.col("stage_id") == 0).row(0, named=True)
        assert row["thermal_cost"] == pytest.approx(20.0)  # block sum 8 + 12
        assert row["immediate_cost"] == pytest.approx(30.0)  # 10 + 20
        # Pre-anticipation run (no anticipated_thermal_cost column): the derived
        # thermal_cost_total falls back to thermal_cost (anticipated read as 0).
        assert "thermal_cost_total" in df.columns
        assert row["thermal_cost_total"] == pytest.approx(20.0)

    def test_stage_costs_reader_folds_anticipated_into_thermal_total(
        self, tmp_path: Path
    ) -> None:
        import polars as pl

        from cobre_bridge.comparators.cobre_readers import read_cobre_stage_costs

        d = tmp_path / "simulation" / "costs" / "scenario_id=0000"
        d.mkdir(parents=True)
        pl.DataFrame(
            {
                "scenario_id": [0],
                "stage_id": [0],
                "block_id": [0],
                "immediate_cost": [30.0],
                "future_cost": [5.0],
                "thermal_cost": [8.0],
                "anticipated_thermal_cost": [7.0],
            }
        ).write_parquet(d / "data.parquet")

        row = read_cobre_stage_costs(tmp_path).row(0, named=True)
        assert row["thermal_cost"] == pytest.approx(8.0)
        assert row["anticipated_thermal_cost"] == pytest.approx(7.0)
        # The source-model-comparable thermal total = live + anticipated GNL fuel.
        assert row["thermal_cost_total"] == pytest.approx(15.0)

    def test_thermal_cost_chart_includes_anticipated_in_cobre_series(self) -> None:
        import polars as pl

        from cobre_bridge.comparators.charts import thermal_cost_chart

        nw_sin = pl.DataFrame(
            {
                "newave_code": [0],
                "stage": [0],
                "variable": ["CTERM"],
                "value": [100.0],
            }
        )
        cobre = pl.DataFrame(
            {
                "stage_id": [0],
                "immediate_cost": [150.0e6],
                "future_cost": [0.0],
                "thermal_cost": [90.0e6],
                "anticipated_thermal_cost": [10.0e6],
                "thermal_cost_total": [100.0e6],
            }
        )
        html = thermal_cost_chart(nw_sin, cobre, nw_offset=0)
        # Cobre series plots thermal_cost_total (90 + 10) = 100, not 90.
        assert "100.0" in html
        assert "anticipated" in html  # legend label

    def test_generic_violation_groups_all_source_model_restriction_parcelas(
        self,
    ) -> None:
        from cobre_bridge.comparators.charts import _resolve_cost_categories

        # cobre-bridge converts the risk-aversion curve/surface (CAR/SAR), electric
        # (RESTELETRICA), interchange (INTERC. MIN.), hydraulic (RHQ/RHV) and
        # piecewise-linear (RLPP) restrictions ALL into generic constraints, so the
        # source model's separate parcelas must sum into the single "Generic Constr.
        # Viol." row to compare against Cobre's aggregated generic_violation_cost.
        nw_costs = {
            "VIOLACAO CAR": 1.0e7,
            "VIOLACAO SAR": 2.0e7,
            "VIOL. RESTELETRICA": 3.0e7,
            "VIOL. INTERC. MIN.": 4.0e7,
            "VIOLACAO RHQ": 5.0e7,
            "VIOLACAO RHV": 6.0e7,
            "VIOL.RLPP DEFLMAX": 7.0e7,
            "VIOL.RLPP DEFLMAXU": 8.0e7,
            "VIOL.RLPP TURBMAX": 9.0e7,
            "VIOL.RLPP TURBMAXU": 1.0e8,
        }
        cobre_costs = {
            "generic_violation_cost": 5.5e8,
            "storage_violation_cost": 1.0e7,
        }

        cats = {
            label: (nw, cb)
            for label, nw, cb, _ in _resolve_cost_categories(nw_costs, cobre_costs)
        }
        # All 10 the source model parcelas land in one row, equal to Cobre's aggregate.
        assert cats["Generic Constr. Viol."] == pytest.approx((5.5e8, 5.5e8))
        # CAR/SAR no longer pollute the storage-bounds row (the source model side empty;
        # the row is now Cobre-only, carrying the individual-reservoir slack).
        assert cats["Storage Bounds Viol."][0] == pytest.approx(0.0)
        assert cats["Storage Bounds Viol."][1] == pytest.approx(1.0e7)


# -------------------------------------------------------------------
# Edge cases and error handling
# -------------------------------------------------------------------


class TestEdgeCases:
    """Test graceful handling of missing/empty data."""

    def test_newave_readers_missing_dir(self, tmp_path: Path) -> None:
        """The source model readers return empty DataFrames when dir missing."""
        from cobre_bridge.comparators.newave_readers import (
            read_medias_hydro,
            read_medias_system,
            read_medias_thermal,
            read_pmo_convergence,
            read_pmo_productivity_detail,
        )

        fake_dir = tmp_path / "nonexistent"
        assert read_medias_hydro(fake_dir).is_empty()
        assert read_medias_thermal(fake_dir).is_empty()
        assert read_medias_system(fake_dir).is_empty()
        assert read_pmo_convergence(fake_dir).is_empty()
        assert read_pmo_productivity_detail(fake_dir).is_empty()

    def test_cobre_readers_missing_dir(self, tmp_path: Path) -> None:
        """Cobre readers return empty DataFrames when dir missing."""
        from cobre_bridge.comparators.cobre_readers import (
            read_cobre_bus_means,
            read_cobre_convergence,
            read_cobre_hydro_means,
            read_cobre_hydro_metadata,
            read_cobre_thermal_means,
        )

        fake_dir = tmp_path / "nonexistent"
        assert read_cobre_hydro_means(fake_dir).is_empty()
        assert read_cobre_thermal_means(fake_dir).is_empty()
        assert read_cobre_bus_means(fake_dir).is_empty()
        assert read_cobre_convergence(fake_dir).is_empty()
        assert read_cobre_hydro_metadata(fake_dir) == {}

    def test_cobre_convergence_reads_014_upper_bound(self, tmp_path: Path) -> None:
        """cobre 0.14 renamed ``upper_bound_mean`` -> ``upper_bound`` (with a
        sibling ``upper_bound_std``/``upper_bound_kind``).  The reader must map
        ``upper_bound`` into the canonical ``upper_bound_mean`` slot and must
        NOT let the prefix-sharing ``_std``/``_kind`` columns hijack it — the
        failure mode is a silently empty convergence frame."""
        import polars as pl

        from cobre_bridge.comparators.cobre_readers import read_cobre_convergence

        training = tmp_path / "training"
        training.mkdir()
        pl.DataFrame(
            {
                "iteration": [1, 2],
                "lower_bound": [10.0, 20.0],
                "upper_bound_std": [1.0, 0.5],
                "upper_bound_kind": ["statistical", "statistical"],
                "upper_bound": [15.0, 22.0],
            }
        ).write_parquet(training / "convergence.parquet")

        df = read_cobre_convergence(tmp_path)
        assert not df.is_empty()
        assert df["iteration"].to_list() == [1, 2]
        assert df["lower_bound"].to_list() == [10.0, 20.0]
        assert df["upper_bound_mean"].to_list() == [15.0, 22.0]

    def test_cobre_convergence_reads_legacy_upper_bound_mean(
        self, tmp_path: Path
    ) -> None:
        """The legacy ``upper_bound_mean`` spelling still reads (back-compat)."""
        import polars as pl

        from cobre_bridge.comparators.cobre_readers import read_cobre_convergence

        training = tmp_path / "training"
        training.mkdir()
        pl.DataFrame(
            {
                "iteration": [1],
                "lower_bound": [10.0],
                "upper_bound_mean": [15.0],
            }
        ).write_parquet(training / "convergence.parquet")

        df = read_cobre_convergence(tmp_path)
        assert df["upper_bound_mean"].to_list() == [15.0]

    def test_empty_alignment_produces_no_results(self) -> None:
        """Alignment with no entities produces empty comparison."""
        summary = build_results_summary([])
        assert summary.total == 0

    def test_html_report_with_empty_results(self) -> None:
        """HTML report renders without error on empty results."""
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )
        from cobre_bridge.comparators.results import PercentileData

        pct = PercentileData()
        dataset = build_results_dataset([], pct, 0.05)
        html = build_comparison_report(dataset)
        assert "<!DOCTYPE html>" in html

    def test_metric_card_html(self) -> None:
        """metric_card produces expected HTML structure."""
        from cobre_bridge.comparators.html_report import metric_card

        html = metric_card("42", "Total")
        assert "42" in html
        assert "Total" in html
        assert "metric-card" in html


# -------------------------------------------------------------------
# Integration tests — full report pipeline
# -------------------------------------------------------------------


class TestComparisonReportIntegration:
    """Integration tests for build_comparison_report() with realistic data.

    Exercises the full pipeline: ResultComparison entries across multiple
    entity types, PercentileData with non-empty DataFrames, and the HTML
    assembly that calls all 8 tab builders.
    """

    @staticmethod
    def _make_results() -> list[ResultComparison]:
        """Build a representative set of comparison results across entity types."""
        return [
            ResultComparison(
                entity_type="hydro",
                entity_name="PLANT_A",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=1200.0,
                cobre_value=1195.0,
                abs_diff=5.0,
                rel_diff=0.004,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name="PLANT_A",
                newave_code=1,
                cobre_id=0,
                stage=1,
                variable="storage_final_hm3",
                newave_value=4500.0,
                cobre_value=4510.0,
                abs_diff=10.0,
                rel_diff=0.002,
            ),
            ResultComparison(
                entity_type="thermal",
                entity_name="GAS_A",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=300.0,
                cobre_value=298.0,
                abs_diff=2.0,
                rel_diff=0.007,
            ),
            ResultComparison(
                entity_type="bus",
                entity_name="SE",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="spot_price",
                newave_value=150.0,
                cobre_value=152.0,
                abs_diff=2.0,
                rel_diff=0.013,
            ),
            ResultComparison(
                entity_type="convergence",
                entity_name="iteration_1",
                newave_code=1,
                cobre_id=1,
                stage=1,
                variable="lower_bound",
                newave_value=50000.0,
                cobre_value=50100.0,
                abs_diff=100.0,
                rel_diff=0.002,
            ),
        ]

    @staticmethod
    def _make_percentile_data() -> PercentileData:
        """Build a PercentileData with minimal non-empty polars DataFrames."""
        import polars as pl

        hydro_df = pl.DataFrame(
            {
                "entity_id": [0, 0, 0],
                "stage_id": [0, 1, 2],
                "generation_mw_p10": [1000.0, 1050.0, 1100.0],
                "generation_mw_p50": [1200.0, 1250.0, 1300.0],
                "generation_mw_p90": [1400.0, 1450.0, 1500.0],
                "storage_final_hm3_p10": [4000.0, 4100.0, 4200.0],
                "storage_final_hm3_p50": [4500.0, 4550.0, 4600.0],
                "storage_final_hm3_p90": [5000.0, 5050.0, 5100.0],
            }
        )
        thermal_df = pl.DataFrame(
            {
                "entity_id": [0, 0, 0],
                "stage_id": [0, 1, 2],
                "generation_mw_p10": [250.0, 260.0, 270.0],
                "generation_mw_p50": [300.0, 310.0, 320.0],
                "generation_mw_p90": [350.0, 360.0, 370.0],
            }
        )
        return PercentileData(hydro=hydro_df, thermal=thermal_df)

    def test_comparison_report_full_pipeline(self) -> None:
        """Full pipeline: multi-entity data renders all 8 tab sections."""
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.html_report import COMPARISON_TABS
        from cobre_bridge.comparators.report_builder import build_comparison_report

        results = self._make_results()
        pctiles = self._make_percentile_data()

        dataset = build_results_dataset(results, pctiles, 0.05)
        html = build_comparison_report(dataset)

        assert "<!DOCTYPE html>" in html
        for tab_id, _label in COMPARISON_TABS:
            assert tab_id in html, f"Tab section '{tab_id}' missing from HTML output"

    def test_comparison_report_contains_plotly_chart(self) -> None:
        """Full pipeline with real data produces at least one Plotly chart."""
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.report_builder import build_comparison_report

        results = self._make_results()
        pctiles = self._make_percentile_data()

        dataset = build_results_dataset(results, pctiles, 0.05)
        html = build_comparison_report(dataset)

        assert "Plotly.newPlot" in html


class TestCompareResultsReturnsDataset:
    """``compare_results`` returns a validated ``ComparisonDataset`` (ticket-022).

    Drives the REAL ``compare_results`` with every reader patched to empty so the
    return-type contract is exercised end-to-end without any case files: every
    Cobre and source-model reader (MEDIAS / nwlistop / pmo included) returns an
    empty frame/dict, and the generic-constraint loaders are empty.
    """

    @staticmethod
    def _patch_all_readers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import polars as pl

        cr = "cobre_bridge.comparators.cobre_readers."
        nr = "cobre_bridge.comparators.newave_readers."
        empty_pl = pl.DataFrame
        empty_pd = lambda *a, **k: __import__("pandas").DataFrame()  # noqa: E731

        # The source-model MEDIAS / nwlistop readers return empty (no result
        # files in the case dir); the MEDIAS block still runs (no saidas/ gate).
        for name in (
            "read_medias_hydro",
            "read_medias_thermal",
            "read_medias_system",
            "read_medias_market",
            "read_medias_sin",
            "read_nwlistop_intercambio",
        ):
            monkeypatch.setattr(nr + name, lambda *a, **k: empty_pl())

        # Frame-returning Cobre readers.
        for name in (
            "read_cobre_hydro_means",
            "read_cobre_thermal_means",
            "read_cobre_bus_means",
            "read_cobre_hydro_percentiles",
            "read_cobre_thermal_percentiles",
            "read_cobre_bus_percentiles",
            "read_cobre_line_means",
            "read_cobre_line_percentiles",
            "read_cobre_lp_max_generation",
            "read_cobre_hydro_total_flows",
            "read_cobre_hydro_withdrawal",
            "read_cobre_hydro_per_stage_bounds",
            "read_cobre_spillage_energy",
            "read_cobre_stage_costs",
            "read_cobre_bus_aggregates",
            "read_cobre_convergence",
            "read_cobre_iteration_timing",
            "read_cobre_productivity_detail",
        ):
            monkeypatch.setattr(cr + name, lambda *a, **k: empty_pl())
        # Dict / scalar Cobre readers.
        monkeypatch.setattr(cr + "read_cobre_hydro_metadata", lambda *a, **k: {})
        monkeypatch.setattr(cr + "read_cobre_hydro_bus_labels", lambda *a, **k: {})
        monkeypatch.setattr(cr + "read_cobre_thermal_metadata", lambda *a, **k: {})
        monkeypatch.setattr(cr + "read_cobre_bus_metadata", lambda *a, **k: {})
        monkeypatch.setattr(cr + "read_cobre_cost_breakdown", lambda *a, **k: {})
        monkeypatch.setattr(cr + "read_cobre_training_duration", lambda *a, **k: 0.0)

        # The remaining source-model readers (pmo / net-load / tim).
        monkeypatch.setattr(nr + "read_pmo_convergence", lambda *a, **k: empty_pl())
        monkeypatch.setattr(
            nr + "read_pmo_productivity_detail", lambda *a, **k: empty_pl()
        )
        monkeypatch.setattr(nr + "read_newave_net_load", lambda *a, **k: empty_pl())
        monkeypatch.setattr(
            nr + "read_newave_tim_iterations", lambda *a, **k: empty_pl()
        )
        monkeypatch.setattr(nr + "read_pmo_cost_breakdown", lambda *a, **k: {})
        monkeypatch.setattr(nr + "read_newave_tim_stages", lambda *a, **k: {})

        # Names / cadastro.
        monkeypatch.setattr(
            "cobre_bridge.comparators.alignment.read_reference_names",
            lambda _case: ({}, {}, {}),
        )
        monkeypatch.setattr(
            "cobre_bridge.converters.hydro.read_cadastro", lambda _case: empty_pd()
        )

        # Generic-constraint loaders (case dir resolves under tmp_path).
        monkeypatch.setattr("cobre_bridge.cobre_io.case_dir_for", lambda _d: tmp_path)
        monkeypatch.setattr(
            "cobre_bridge.comparators.constraints_compare._load_generic_constraints",
            lambda _d: [],
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.constraints_compare."
            "_load_generic_constraint_bounds",
            lambda _d: empty_pl(),
        )

    def test_compare_results_returns_validated_dataset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.comparators.dataset import ComparisonDataset
        from cobre_bridge.comparators.results import compare_results

        self._patch_all_readers(monkeypatch, tmp_path)

        case = MagicMock()
        case.files.directory = tmp_path

        out = compare_results(
            case=case,
            id_map=MagicMock(),
            alignment=MagicMock(),
            cobre_output_dir=tmp_path,
            tolerance=0.05,
        )

        assert isinstance(out, ComparisonDataset)
        out.validate()
        # The render-only carry-overs are present in-memory but absent from
        # the serialized artifact.
        assert "results" in out.metadata
        paths = out.to_dir(tmp_path / "artifacts")
        import json

        written = json.loads(paths[2].read_text(encoding="utf-8"))
        assert "results" not in written
        assert "top_divergences" in written


class TestProductivityDetail:
    """Productivity-tab readers, assembly, and charts.

    The Productivity tab is a *static* conversion-fidelity check: The source model pmo
    productivities vs what cobre-bridge computes from the same HIDR cadastro, plus a
    per-stage realized-productivity line chart and a grouped building-blocks table.
    """

    @staticmethod
    def _write_hydros_json(tmp_path: Path) -> None:
        """Write a tiny ``system/hydros.json`` (building blocks only)."""
        import json

        system = tmp_path / "system"
        system.mkdir(parents=True, exist_ok=True)
        (system / "hydros.json").write_text(
            json.dumps(
                {
                    "hydros": [
                        {
                            "id": 0,
                            "name": "ALPHA",
                            "reservoir": {
                                "min_storage_hm3": 100.0,
                                "max_storage_hm3": 500.0,
                            },
                            "specific_productivity_mw_per_m3s_per_m": 0.009,
                            "tailrace": {
                                "type": "polynomial",
                                "coefficients": [672.0],
                            },
                            "hydraulic_losses": {"type": "constant", "value_m": 0.8},
                        },
                        {
                            "id": 1,
                            "name": "BETA",
                            "reservoir": {
                                "min_storage_hm3": 0.0,
                                "max_storage_hm3": 0.0,
                            },
                            "specific_productivity_mw_per_m3s_per_m": 0.01,
                            "tailrace": {
                                "type": "polynomial",
                                "coefficients": [400.0],
                            },
                            "hydraulic_losses": {"type": "constant", "value_m": 0.0},
                        },
                    ]
                }
            )
        )

    def test_cobre_productivity_detail_reader(self, tmp_path: Path) -> None:
        """Cobre reader surfaces the converted building blocks per hydro."""
        from cobre_bridge.comparators.cobre_readers import (
            read_cobre_productivity_detail,
        )

        out = tmp_path / "output"
        out.mkdir()
        self._write_hydros_json(tmp_path)  # hydros.json resolves from case_dir

        detail = read_cobre_productivity_detail(out)
        assert set(detail) == {0, 1}
        alpha = detail[0]
        assert alpha["name"] == "ALPHA"
        assert alpha["specific_productivity"] == pytest.approx(0.009)
        assert alpha["tailwater_m"] == pytest.approx(672.0)
        assert alpha["losses_m"] == pytest.approx(0.8)
        assert alpha["vmin_hm3"] == pytest.approx(100.0)
        assert alpha["vmax_hm3"] == pytest.approx(500.0)
        # The reader no longer reads point/equivalent/accumulated from Cobre.
        assert "point" not in alpha
        assert "equivalent" not in alpha
        assert "accumulated" not in alpha

    def test_cobre_productivity_detail_missing_dir(self, tmp_path: Path) -> None:
        from cobre_bridge.comparators.cobre_readers import (
            read_cobre_productivity_detail,
        )

        assert read_cobre_productivity_detail(tmp_path / "nope") == {}

    def test_pmo_productivity_detail_missing_pmo(self, tmp_path: Path) -> None:
        from cobre_bridge.comparators.newave_readers import (
            read_pmo_productivity_detail,
        )

        df = read_pmo_productivity_detail(tmp_path / "nope")
        assert df.is_empty()
        assert df.columns == [
            "plant_name",
            "altura_min",
            "altura_65",
            "altura_max",
            "equivalent",
            "accumulated_earm",
        ]

    @staticmethod
    def _detail_df():
        """A small productivity_detail frame for the chart/table smoke tests.

        ``cb_point`` / ``cb_equivalent`` / ``cb_accumulated`` are the
        cobre-bridge *static* values (here close to the pmo side, so the
        scatters cluster on y = x).
        """
        import polars as pl

        from cobre_bridge.comparators.analyze import _PRODUCTIVITY_DETAIL_SCHEMA

        return pl.DataFrame(
            {
                "plant_name": ["ALPHA", "BETA"],
                "newave_code": [1, 2],
                "cobre_id": [0, 1],
                "nw_altura_min": [0.69, None],
                "nw_altura_65": [0.81, 0.40],
                "nw_altura_max": [0.85, None],
                "nw_equivalent": [0.7865, 0.50],
                "nw_accumulated_earm": [5.35, 0.50],
                "nw_specific_productivity": [0.009, 0.0102],
                "nw_tailwater_m": [672.0, 400.0],
                "nw_losses_m": [0.8, 0.0],
                "nw_vmin_hm3": [100.0, 0.0],
                "nw_vmax_hm3": [500.0, 0.0],
                "cb_point": [0.811, 0.401],
                "cb_equivalent": [0.7860, 0.50],
                "cb_accumulated": [5.349, 0.50],
                "cb_specific_productivity": [0.009, 0.0098],
                "cb_tailwater_m": [672.0, 405.0],
                "cb_losses_m": [0.8, 0.0],
                "cb_vmin_hm3": [100.0, 0.0],
                "cb_vmax_hm3": [500.0, 0.0],
            },
            schema=_PRODUCTIVITY_DETAIL_SCHEMA,
        )

    def test_build_productivity_detail_computes_cobre_bridge_side(self) -> None:
        """cb_point/equivalent/accumulated come from the converter, not Cobre."""
        import pandas as pd

        from cobre_bridge.comparators.alignment import (
            EntityAlignment,
            HydroEntity,
        )
        from cobre_bridge.comparators.analyze import build_productivity_detail
        from cobre_bridge.productivity import compute_productivity

        alignment = EntityAlignment(
            hydros=[
                HydroEntity(
                    newave_code=6, cobre_id=69, name="ALPHA", has_reservoir=True
                )
            ]
        )
        nw_detail = __import__("polars").DataFrame(
            {
                "plant_name": ["ALPHA"],
                "altura_min": [0.6926],
                "altura_65": [0.813],
                "altura_max": [0.8545],
                "equivalent": [0.7865],
                "accumulated_earm": [5.3517],
            }
        )
        # Monthly-regulated plant with a simple linear volume→cota polynomial:
        #   h(v) = 600 + 0.2 v ;  v_65 = vmin + 0.65 (vmax − vmin) = 360 ;
        #   h(360) = 672 ; net_drop = 672 − 200 = 472 ; additive losses 0.8 ;
        #   ρ_point = 0.009 × (472 − 0.8) = 4.2408
        cadastro = pd.DataFrame(
            {
                "produtibilidade_especifica": [0.009],
                "canal_fuga_medio": [200.0],
                "perdas": [0.8],
                "tipo_perda": [2],
                "tipo_regulacao": ["M"],
                "volume_minimo": [100.0],
                "volume_maximo": [500.0],
                "volume_referencia": [360.0],
                "a0_volume_cota": [600.0],
                "a1_volume_cota": [0.2],
                "a2_volume_cota": [0.0],
                "a3_volume_cota": [0.0],
                "a4_volume_cota": [0.0],
            },
            index=pd.Index([6], name="codigo_usina"),
        )
        cobre_detail = {
            69: {
                "name": "ALPHA",
                "specific_productivity": 0.009,
                "tailwater_m": 200.0,
                "losses_m": 0.8,
                "vmin_hm3": 100.0,
                "vmax_hm3": 500.0,
            }
        }
        cb_accumulated = {6: 12.34}
        df = build_productivity_detail(
            alignment, nw_detail, cadastro, cobre_detail, cb_accumulated
        )
        assert df.height == 1
        row = df.row(0, named=True)
        # The source model pmo side carried through.
        assert row["nw_altura_65"] == pytest.approx(0.813)
        assert row["nw_equivalent"] == pytest.approx(0.7865)
        assert row["nw_accumulated_earm"] == pytest.approx(5.3517)
        # cobre-bridge side computed from the cadastro / cascade map.
        expected_point = compute_productivity(cadastro.loc[6])
        assert row["cb_point"] == pytest.approx(expected_point)
        assert row["cb_point"] == pytest.approx(4.2408)
        # Monthly plant → stored_energy_productivity integrates; just assert
        # it is populated and finite.
        assert row["cb_equivalent"] is not None
        assert row["cb_accumulated"] == pytest.approx(12.34)
        # Building blocks from system/hydros.json.
        assert row["cb_tailwater_m"] == pytest.approx(200.0)

    def test_build_detail_run_of_river_uses_volume_referencia(self) -> None:
        """Daily-regulation ('D') plants compare against volume_referencia, not
        the dead-storage volume_minimo/maximo the converter freezes them off."""
        import pandas as pd
        import polars as pl

        from cobre_bridge.comparators.alignment import (
            EntityAlignment,
            HydroEntity,
        )
        from cobre_bridge.comparators.analyze import build_productivity_detail

        alignment = EntityAlignment(
            hydros=[
                HydroEntity(newave_code=4, cobre_id=7, name="ROR", has_reservoir=False)
            ]
        )
        cadastro = pd.DataFrame(
            {
                "tipo_regulacao": ["D"],
                "volume_minimo": [304.0],
                "volume_maximo": [304.0],
                "volume_referencia": [265.9],
            },
            index=pd.Index([4], name="codigo_usina"),
        )
        cobre_detail = {7: {"name": "ROR", "vmin_hm3": 265.9, "vmax_hm3": 265.9}}
        df = build_productivity_detail(
            alignment, pl.DataFrame({"plant_name": ["ROR"]}), cadastro, cobre_detail, {}
        )
        row = df.row(0, named=True)
        # The source model side uses volume_referencia (265.9), matching Cobre — no
        # spurious delta vs the cadastro volume_minimo/maximo (304).
        assert row["nw_vmin_hm3"] == pytest.approx(265.9)
        assert row["nw_vmax_hm3"] == pytest.approx(265.9)
        assert row["cb_vmin_hm3"] == pytest.approx(265.9)

    def test_comparison_scatter_renders_series_and_stats(self) -> None:
        from cobre_bridge.comparators.charts import (
            productivity_comparison_scatter,
        )

        df = self._detail_df()
        for kind, nw_label, cb_label in (
            ("point", "produtibilidade_altura_65", "compute_productivity"),
            (
                "equivalent",
                "produtibilidade_equivalente_volmin_volmax",
                "stored_energy_productivity",
            ),
            (
                "accumulated",
                "produtibilidade_acumulada_calculo_earm",
                "accumulated_integrated_productivity",
            ),
        ):
            html = productivity_comparison_scatter(df, kind)
            assert "Plotly.newPlot" in html
            assert nw_label in html
            assert cb_label in html
            assert "rel. err" in html
            assert "y = x" in html

    def test_comparison_scatter_empty_and_bad_kind(self) -> None:
        import polars as pl

        from cobre_bridge.comparators.charts import (
            productivity_comparison_scatter,
        )

        assert "No productivity data" in productivity_comparison_scatter(
            pl.DataFrame(), "point"
        )
        with pytest.raises(ValueError, match="Unknown productivity kind"):
            productivity_comparison_scatter(self._detail_df(), "bogus")

    @staticmethod
    def _per_stage_results():
        """Per-stage productivity_mw_per_m3s ResultComparison rows, 2 plants."""
        from cobre_bridge.comparators.results import ResultComparison

        rows: list[ResultComparison] = []
        for name, code, cid, base in (("ALPHA", 1, 0, 0.78), ("BETA", 2, 1, 0.40)):
            for stage in range(3):
                nw = base + 0.01 * stage
                cb = base + 0.012 * stage
                rows.append(
                    ResultComparison(
                        entity_type="hydro",
                        entity_name=name,
                        newave_code=code,
                        cobre_id=cid,
                        stage=stage,
                        variable="productivity_mw_per_m3s",
                        newave_value=nw,
                        cobre_value=cb,
                        abs_diff=abs(nw - cb),
                        rel_diff=abs(nw - cb) / nw,
                    )
                )
        return rows

    def test_per_stage_chart_reuses_shared_per_plant_dropdown(self) -> None:
        from cobre_bridge.comparators.analyze import productivity_per_stage_frame
        from cobre_bridge.comparators.charts import productivity_per_stage_chart

        html = productivity_per_stage_chart(
            productivity_per_stage_frame(self._per_stage_results())
        )
        # Reuses the shared interactive per-plant <select> dropdown widget
        # (same as the hydro/thermal detail tabs) — every plant is selectable.
        assert "<select" in html
        assert "ALPHA (1)" in html and "BETA (2)" in html
        assert "prodstage-chart-productivity-mw-per-m3s" in html
        assert "Realized productivity" in html
        # Per-stage the source model + Cobre arrays embedded for the JS to plot.
        assert "productivity_mw_per_m3s_nw" in html
        assert "productivity_mw_per_m3s_cb" in html

    def test_per_stage_chart_no_rows(self) -> None:
        from cobre_bridge.comparators.analyze import productivity_per_stage_frame
        from cobre_bridge.comparators.charts import productivity_per_stage_chart

        empty = productivity_per_stage_frame([])
        assert "No per-stage productivity data" in productivity_per_stage_chart(empty)

    def test_blocks_table_grouped_header_and_highlight(self) -> None:
        from cobre_bridge.comparators.charts import productivity_blocks_table

        html = productivity_blocks_table(self._detail_df())
        assert "cost-breakdown-table" in html
        assert "prod-blocks-table" in html
        assert "Productivity Building Blocks" in html
        # Two-level grouped header: metric label spans 3 sub-columns.
        assert 'colspan="3"' in html
        assert "ρ_esp" in html
        assert "Tailwater" in html
        # Per-group sub-columns and group tint/separator cues.
        assert ">Δ%<" in html
        assert "cb-group-tint" in html
        assert "cb-group-sep" in html
        # BETA: ρ_esp 0.0102 vs 0.0098 (~−3.9%) → Δ% cell highlighted.
        assert "cb-diff-pos" in html
        assert "ALPHA" in html and "BETA" in html

    def test_blocks_table_empty(self) -> None:
        import polars as pl

        from cobre_bridge.comparators.charts import productivity_blocks_table

        assert "No productivity data" in productivity_blocks_table(pl.DataFrame())


class TestEvaluateLhsCobre:
    """Regression cover for evaluate_lhs_cobre's simulation-scan paths.

    The real-LazyFrame path was untested and regressed once: `lf or
    pl.LazyFrame()` evaluated bool(lf), which polars rejects.
    """

    @staticmethod
    def _storage_constraint() -> list[dict]:
        return [
            {
                "id": 0,
                "name": "VminOP_0",
                "expression": "hydro_storage(0)",
                "sense": ">=",
                "slack": {"enabled": False},
            }
        ]

    def test_evaluates_lhs_from_a_real_simulation_lazyframe(self) -> None:
        """A present simulation (LazyFrame, not None) must evaluate, not raise."""
        import polars as pl

        from cobre_bridge.comparators.constraints_compare import evaluate_lhs_cobre

        hydros = pl.DataFrame(
            {
                "scenario_id": [0, 0],
                "stage_id": [0, 1],
                "block_id": [0, 0],
                "hydro_id": [0, 0],
                "storage_final_hm3": [100.0, 200.0],
                "generation_mw": [10.0, 20.0],
            }
        ).lazy()

        def fake_scan(_output_dir, entity):
            return hydros if entity == "hydros" else None

        with patch(
            "cobre_bridge.comparators.constraints_compare._scan_simulation_entity",
            side_effect=fake_scan,
        ):
            result = evaluate_lhs_cobre(self._storage_constraint(), Path("/out"))

        rows = {
            (r["constraint_id"], r["stage_id"]): r["lhs_value"]
            for r in result.iter_rows(named=True)
        }
        assert rows[(0, 0)] == pytest.approx(100.0)
        assert rows[(0, 1)] == pytest.approx(200.0)

    def test_missing_simulation_returns_empty(self) -> None:
        """Both entities absent (None) → empty frame, no error."""
        from cobre_bridge.comparators.constraints_compare import evaluate_lhs_cobre

        with patch(
            "cobre_bridge.comparators.constraints_compare._scan_simulation_entity",
            return_value=None,
        ):
            result = evaluate_lhs_cobre(self._storage_constraint(), Path("/out"))
        assert result.is_empty()

    def test_no_constraints_returns_empty_without_scanning(self) -> None:
        from cobre_bridge.comparators.constraints_compare import evaluate_lhs_cobre

        result = evaluate_lhs_cobre([], Path("/out"))
        assert result.is_empty()


# -------------------------------------------------------------------
# ticket-028: comparator readers -> F3 (derive shape from the interval)
# -------------------------------------------------------------------


class TestShapeFromBounds:
    """AC2: unit tests for the F3 inverse-mapping helper."""

    def test_lower_only_is_ge(self) -> None:
        from cobre_bridge.generic_constraint_format import shape_from_bounds

        assert shape_from_bounds(10.0, None) == ">="

    def test_upper_only_is_le(self) -> None:
        from cobre_bridge.generic_constraint_format import shape_from_bounds

        assert shape_from_bounds(None, 10.0) == "<="

    def test_both_equal_is_eq(self) -> None:
        from cobre_bridge.generic_constraint_format import shape_from_bounds

        assert shape_from_bounds(7.0, 7.0) == "=="

    def test_both_unequal_is_range(self) -> None:
        from cobre_bridge.generic_constraint_format import shape_from_bounds

        assert shape_from_bounds(1.0, 10.0) == "range"

    def test_both_none_raises_value_error(self) -> None:
        from cobre_bridge.generic_constraint_format import shape_from_bounds

        with pytest.raises(ValueError, match="at least one endpoint"):
            shape_from_bounds(None, None)


class TestGenericConstraintF3Loaders:
    """AC1/AC4: the loaders consume F3's sense-free JSON + interval parquet."""

    def test_bounds_loader_missing_file_has_no_bound_column(
        self, tmp_path: Path
    ) -> None:
        from cobre_bridge.comparators.constraints_compare import (
            _load_generic_constraint_bounds,
        )

        df = _load_generic_constraint_bounds(tmp_path)
        assert "bound" not in df.columns
        assert {"bound_lower", "bound_upper"}.issubset(set(df.columns))

    def test_bounds_loader_reads_f3_parquet_endpoints(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        from cobre_bridge.comparators.constraints_compare import (
            _load_generic_constraint_bounds,
        )

        constraints_dir = tmp_path / "constraints"
        constraints_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "constraint_id": pa.array([0], type=pa.int32()),
                    "stage_id": pa.array([0], type=pa.int32()),
                    "block_id": pa.array([0], type=pa.int32()),
                    "bound_lower": pa.array([10.0], type=pa.float64()),
                    "bound_upper": pa.array([None], type=pa.float64()),
                }
            ),
            constraints_dir / "generic_constraint_bounds.parquet",
        )

        df = _load_generic_constraint_bounds(tmp_path)
        assert "bound" not in df.columns
        assert df["bound_lower"].to_list() == [10.0]
        assert df["bound_upper"].to_list() == [None]

    def test_constraints_loader_parses_sense_free_json(self, tmp_path: Path) -> None:
        import json

        from cobre_bridge.comparators.constraints_compare import (
            _load_generic_constraints,
        )

        constraints_dir = tmp_path / "constraints"
        constraints_dir.mkdir()
        (constraints_dir / "generic_constraints.json").write_text(
            json.dumps(
                {
                    "constraints": [
                        {
                            "id": 0,
                            "name": "RE_1",
                            "expression": "hydro_generation(0)",
                            "description": "Electric restriction",
                        }
                    ]
                }
            )
        )

        constraints = _load_generic_constraints(tmp_path)
        assert len(constraints) == 1
        assert "sense" not in constraints[0]
        assert constraints[0]["name"] == "RE_1"


class TestPerStageBoundsResolution:
    """AC3 (unit half): per_stage_bounds resolves the endpoint the pre-F3
    single ``bound`` column used to hold, for every constraint direction."""

    @staticmethod
    def _f3_bounds() -> pl.DataFrame:
        return pl.DataFrame(
            {
                "constraint_id": [0, 0, 1, 1, 2, 2],
                "stage_id": [0, 1, 0, 1, 0, 1],
                "block_id": [0, 0, 0, 0, 0, 0],
                # cid 0: >= (VminOP-style)  -> lower-only
                # cid 1: <= (AGRINT-style)  -> upper-only
                # cid 2: == (equality)      -> both endpoints equal
                "bound_lower": [500.0, 520.0, None, None, 300.0, 300.0],
                "bound_upper": [None, None, 200.0, 210.0, 300.0, 300.0],
            }
        )

    def test_resolves_ge_from_lower_endpoint(self) -> None:
        from cobre_bridge.comparators.constraints_compare import per_stage_bounds

        resolved = per_stage_bounds(self._f3_bounds())
        assert resolved[0][0].value == 500.0
        assert resolved[0][0].shape == ">="
        assert resolved[0][1].value == 520.0

    def test_resolves_le_from_upper_endpoint(self) -> None:
        from cobre_bridge.comparators.constraints_compare import per_stage_bounds

        resolved = per_stage_bounds(self._f3_bounds())
        assert resolved[1][0].value == 200.0
        assert resolved[1][0].shape == "<="
        assert resolved[1][1].value == 210.0

    def test_resolves_eq_from_either_endpoint(self) -> None:
        from cobre_bridge.comparators.constraints_compare import per_stage_bounds

        resolved = per_stage_bounds(self._f3_bounds())
        assert resolved[2][0].value == 300.0
        assert resolved[2][0].shape == "=="


class TestAC3NumericRegressionAcrossF3Migration:
    """AC3 (integration half): an F3 case whose constraints round-trip
    unchanged from the pre-F3 ``(sense, bound)`` content produces identical
    per-stage evaluated limits and pass/fail verdicts to the pre-F3
    comparison — same numbers, new column layout."""

    def test_f3_limits_and_verdicts_match_pre_f3_content(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        from cobre_bridge.comparators.constraints_compare import (
            _load_generic_constraint_bounds,
            per_stage_bounds,
        )
        from cobre_bridge.generic_constraint_format import sense_to_interval

        # Pre-F3 ground truth this case is migrated from: constraint 0 is a
        # VminOP-style `>=` security-curve floor; constraint 1 is an
        # AGRINT-style `<=` ceiling. These are the exact (sense, value) pairs
        # a pre-F3 case would have carried in its single `bound` column.
        pre_f3_bound = {
            (0, 0): (">=", 500.0),
            (0, 1): (">=", 520.0),
            (1, 0): ("<=", 200.0),
            (1, 1): ("<=", 200.0),
        }
        lhs = {(0, 0): 510.0, (0, 1): 515.0, (1, 0): 190.0, (1, 1): 205.0}

        constraints_dir = tmp_path / "constraints"
        constraints_dir.mkdir()
        cids: list[int] = []
        stages: list[int] = []
        blocks: list[int] = []
        lowers: list[float | None] = []
        uppers: list[float | None] = []
        for (cid, stage), (sense, value) in pre_f3_bound.items():
            lo, up = sense_to_interval(sense, value)
            cids.append(cid)
            stages.append(stage)
            blocks.append(0)
            lowers.append(lo)
            uppers.append(up)
        pq.write_table(
            pa.table(
                {
                    "constraint_id": pa.array(cids, type=pa.int32()),
                    "stage_id": pa.array(stages, type=pa.int32()),
                    "block_id": pa.array(blocks, type=pa.int32()),
                    "bound_lower": pa.array(lowers, type=pa.float64()),
                    "bound_upper": pa.array(uppers, type=pa.float64()),
                }
            ),
            constraints_dir / "generic_constraint_bounds.parquet",
        )

        gc_bounds = _load_generic_constraint_bounds(tmp_path)
        assert "bound" not in gc_bounds.columns
        resolved = per_stage_bounds(gc_bounds)

        for (cid, stage), (sense, value) in pre_f3_bound.items():
            rb = resolved[cid][stage]
            # Identical per-stage evaluated limit to the pre-F3 single
            # `bound` column value.
            assert rb.value == value
            assert rb.shape == sense
            # Identical pass/fail verdict to what a pre-F3 comparison
            # (LHS vs. the single (sense, bound) pair) would have reached.
            lhs_value = lhs[(cid, stage)]
            pre_f3_pass = lhs_value >= value if sense == ">=" else lhs_value <= value
            f3_pass = (
                lhs_value >= rb.value if rb.shape == ">=" else lhs_value <= rb.value
            )
            assert f3_pass == pre_f3_pass


class TestConstraintsChartShapeLabelFromBounds:
    """AC4 (chart half): the chart title derives the shape label from the
    resolved bounds via `shape_from_bounds`, not from a removed `sense`
    field on the (now sense-free) constraint dict."""

    def test_chart_title_uses_shape_derived_from_bounds(self) -> None:
        import polars as pl

        from cobre_bridge.comparators.charts import constraints_comparison_chart
        from cobre_bridge.comparators.constraints_compare import ResolvedBound

        constraints = [{"id": 0, "name": "VminOP_X"}]  # sense-free (F3)
        lhs_newave = pl.DataFrame(
            {"constraint_id": [0], "stage_id": [0], "lhs_value": [510.0]}
        )
        lhs_cobre = pl.DataFrame(
            {"constraint_id": [0], "stage_id": [0], "lhs_value": [508.0]}
        )
        bound_by_constraint = {0: {0: ResolvedBound(500.0, ">=")}}

        html = constraints_comparison_chart(
            constraints, lhs_newave, lhs_cobre, bound_by_constraint
        )
        # json_for_script escapes `>` as `>` before embedding in <script>.
        assert "VminOP_X (\\u003e=)" in html

    def test_chart_title_defaults_when_no_bound_data(self) -> None:
        import polars as pl

        from cobre_bridge.comparators.charts import constraints_comparison_chart

        constraints = [{"id": 0, "name": "RE_1"}]
        lhs_newave = pl.DataFrame(
            {"constraint_id": [0], "stage_id": [0], "lhs_value": [10.0]}
        )
        lhs_cobre = pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

        # No bound entries for cid 0 -> the "<=" default (matching the old
        # `c.get("sense", "<=")` fallback) is used.
        html = constraints_comparison_chart(constraints, lhs_newave, lhs_cobre, {})
        assert "RE_1 (\\u003c=)" in html


_BOUND_ACCESS_RE = re.compile(
    r'\.col\(\s*["\']bound["\']\s*\)|\[\s*["\']bound["\']\s*\]|\.get\(\s*["\']bound["\']'
)
_SENSE_ACCESS_RE = re.compile(
    r'\.col\(\s*["\']sense["\']\s*\)|\[\s*["\']sense["\']\s*\]|\.get\(\s*["\']sense["\']'
)


class TestNoSenseOrSingleBoundColumnRemainsInComparators:
    """AC5 grep guard: no comparator reads a removed `sense` key or a single
    `bound` column. Matches only genuine column/dict *access* patterns
    (``.col("bound")``, ``row["bound"]``, ``.get("bound"``) — column *names*
    that merely contain "bound" as a substring (``bound_lower``,
    ``bound_upper``) and unrelated string literals (e.g. charts.py's
    ``"legendgroup": "bound"`` trace label) are not matches.
    """

    @pytest.mark.parametrize(
        "relative_path",
        [
            "src/cobre_bridge/comparators/constraints_compare.py",
            "src/cobre_bridge/comparators/charts.py",
            "src/cobre_bridge/comparators/results.py",
        ],
    )
    def test_module_has_no_sense_or_bound_column_access(
        self, relative_path: str
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        text = (repo_root / relative_path).read_text(encoding="utf-8")
        assert not _BOUND_ACCESS_RE.search(text), (
            f"{relative_path} still accesses a single `bound` column"
        )
        assert not _SENSE_ACCESS_RE.search(text), (
            f"{relative_path} still accesses a `sense` key"
        )
