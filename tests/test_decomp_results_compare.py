"""Tests for the DECOMP-vs-Cobre results comparison slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from cobre_bridge.comparators.decomp_html_report import build_decomp_comparison_report
from cobre_bridge.comparators.decomp_results import (
    _HYDRO_VARIABLES,
    DecompComparison,
    _map_entities,
    _scenario_mean,
    _stage_rows,
    _summarize,
    _tidy,
)
from cobre_bridge.verdict import decomp_compare_summary


def _source_frame() -> pl.DataFrame:
    """Two stages of one plant: per-block rows plus the stage-aggregate row."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 2, 2, 2],
            "no": [1, 1, 1, 2, 2, 2],
            "patamar": [1.0, 2.0, None, 1.0, 2.0, None],
            "duracao": [24.0, 144.0, None, 24.0, 144.0, None],
            "codigo_usina": [10, 10, 10, 10, 10, 10],
            "geracao_MW": [120.0, 60.0, 68.57, 100.0, 50.0, 57.14],
        }
    )


class TestStageRows:
    def test_prefers_the_aggregate_row(self) -> None:
        rows = _stage_rows(_source_frame())
        assert len(rows) == 2
        assert "patamar" not in rows.columns
        assert rows["geracao_MW"].to_list() == [68.57, 57.14]

    def test_falls_back_to_block_rows_when_absent(self) -> None:
        frame = _source_frame().filter(pl.col("patamar").is_not_null())
        rows = _stage_rows(frame)
        assert len(rows) == 4


class TestScenarioMean:
    def test_averages_over_nodes(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3, 3],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [10.0, 20.0, 60.0],
            }
        )
        out = _scenario_mean(
            frame, "estagio", ["geracao_MW"], entity_column="codigo_usina"
        )
        assert out["geracao_MW"].to_list() == [30.0]


class TestMapEntities:
    def test_maps_codes_and_rebases_stages(self) -> None:
        frame = pl.DataFrame({"estagio": [1, 2], "codigo_usina": [10, 10]})
        mapped, unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["entity_id"].to_list() == [4, 4]
        assert mapped["stage_id"].to_list() == [0, 1]
        assert unmapped == []

    def test_reports_unmapped_codes_instead_of_dropping_silently(self) -> None:
        frame = pl.DataFrame({"estagio": [1, 1], "codigo_usina": [10, 99]})
        mapped, unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["entity_id"].to_list() == [4]
        assert unmapped == [99]


class TestTidyAndSummary:
    def _pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        source = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "geracao_MW": [100.0, 50.0],
            }
        )
        cobre = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw": [90.0, 50.0],
            }
        )
        return source, cobre

    def test_tidy_rows_carry_both_sides_and_the_difference(self) -> None:
        source, cobre = self._pair()
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={0: "A", 1: "B"})
        generation = rows.filter(pl.col("variable") == "generation").sort("entity_id")
        assert generation["source"].to_list() == [100.0, 50.0]
        assert generation["cobre"].to_list() == [90.0, 50.0]
        assert generation["delta"].to_list() == [-10.0, 0.0]
        assert generation["delta_pct"].to_list() == [pytest.approx(-10.0), 0.0]
        assert generation["entity_name"].to_list() == ["A", "B"]

    def test_variables_missing_on_either_side_are_skipped(self) -> None:
        source, cobre = self._pair()
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={})
        # Only generation is present in both frames.
        assert set(rows["variable"]) == {"generation"}

    def test_summary_totals_and_worst_entity(self) -> None:
        source, cobre = self._pair()
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={0: "A", 1: "B"})
        summary = _summarize(rows)
        assert len(summary) == 1
        row = summary.to_dicts()[0]
        assert row["n"] == 2
        assert row["source_total"] == 150.0
        assert row["cobre_total"] == 140.0
        assert row["delta_total"] == -10.0
        assert row["delta_total_pct"] == pytest.approx(-100.0 / 15.0)
        assert row["worst_entity"] == "A"

    def test_zero_versus_zero_reads_as_agreement(self) -> None:
        source = pl.DataFrame({"entity_id": [0], "stage_id": [0], "geracao_MW": [0.0]})
        cobre = pl.DataFrame(
            {"entity_id": [0], "stage_id": [0], "generation_mw": [0.0]}
        )
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={})
        assert rows["smape_pct"].to_list() == [0.0]
        assert rows["delta_pct"].to_list() == [None]

    def test_empty_join_yields_an_empty_but_typed_frame(self) -> None:
        source = pl.DataFrame({"entity_id": [0], "stage_id": [0], "geracao_MW": [1.0]})
        cobre = pl.DataFrame(
            {"entity_id": [9], "stage_id": [9], "generation_mw": [1.0]}
        )
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={})
        assert rows.is_empty()
        assert "smape_pct" in rows.columns
        assert _summarize(rows).is_empty()


class TestComparisonRecord:
    def test_stage_count_counts_distinct_stages(self) -> None:
        rows = pl.DataFrame(
            {
                "level": ["hydro"] * 3,
                "variable": ["generation"] * 3,
                "unit": ["MW"] * 3,
                "entity_id": [0, 0, 1],
                "entity_name": ["A", "A", "B"],
                "stage_id": [0, 1, 1],
                "source": [1.0, 2.0, 3.0],
                "cobre": [1.0, 2.0, 3.0],
                "delta": [0.0, 0.0, 0.0],
                "delta_pct": [0.0, 0.0, 0.0],
                "smape_pct": [0.0, 0.0, 0.0],
            }
        )
        comparison = DecompComparison(
            rows=rows,
            summary=_summarize(rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        assert comparison.stage_count == 2


def _fake_comparison() -> DecompComparison:
    rows = pl.DataFrame(
        {
            "level": ["hydro", "thermal"],
            "variable": ["generation", "generation"],
            "unit": ["MW", "MW"],
            "entity_id": [0, 0],
            "entity_name": ["A", "T"],
            "stage_id": [0, 0],
            "source": [100.0, 20.0],
            "cobre": [90.0, 20.0],
            "delta": [-10.0, 0.0],
            "delta_pct": [-10.0, 0.0],
            "smape_pct": [10.5, 0.0],
        }
    )
    return DecompComparison(
        rows=rows,
        summary=_summarize(rows),
        convergence=pl.DataFrame(
            {
                "iteration": [1, 2],
                "source_lower": [1.0, 2.0],
                "source_upper": [3.0, 3.0],
                "cobre_lower": [10.0, None],
                "cobre_upper": [30.0, None],
            }
        ),
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


class TestBuildDecompComparisonReport:
    """``build_decomp_comparison_report`` renders the same frames as the console."""

    def test_populated_comparison_renders_all_three_frames(self) -> None:
        comparison = _fake_comparison()

        report = build_decomp_comparison_report(comparison)

        assert "<!DOCTYPE html>" in report
        assert "Operation comparison" in report
        assert "generation" in report
        assert "Final bounds" in report

    def test_unmapped_codes_appear_in_the_report(self) -> None:
        """``unmapped={"thermal": [86, 224]}`` in the fixture must surface."""
        report = build_decomp_comparison_report(_fake_comparison())

        assert "86" in report
        assert "224" in report

    def test_empty_comparison_short_circuits_without_raising(self) -> None:
        empty_rows = _fake_comparison().rows.clear()
        comparison = DecompComparison(
            rows=empty_rows,
            summary=_summarize(empty_rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )

        report = build_decomp_comparison_report(comparison)

        assert "no comparable rows" in report
        assert "<!DOCTYPE html>" in report


class TestDecompCompareSummaryTolerance:
    """The within-tolerance verdict keys ``decomp_compare_summary`` appends."""

    def test_mixed_fixture_has_one_variable_exceeding_tolerance(self) -> None:
        """Hydro's ``smape_pct == 10.5`` exceeds 1e-2; thermal's ``0.0`` is within."""
        comparison = _fake_comparison()

        summary = decomp_compare_summary(comparison, tolerance=1e-2)

        assert list(summary.keys()) == [
            "stages",
            "variables",
            "unmapped",
            "within_tol",
            "total",
            "all_within_tol",
        ]
        assert summary["total"] == 2
        assert summary["within_tol"] == 1
        assert summary["all_within_tol"] is False

    def test_all_rows_within_tolerance_report_all_within_tol_true(self) -> None:
        rows = pl.DataFrame(
            {
                "level": ["hydro", "thermal"],
                "variable": ["generation", "generation"],
                "unit": ["MW", "MW"],
                "entity_id": [0, 0],
                "entity_name": ["A", "T"],
                "stage_id": [0, 0],
                "source": [100.0, 20.0],
                "cobre": [100.0, 20.0],
                "delta": [0.0, 0.0],
                "delta_pct": [0.0, 0.0],
                "smape_pct": [0.0, 0.0],
            }
        )
        comparison = DecompComparison(
            rows=rows,
            summary=_summarize(rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )

        summary = decomp_compare_summary(comparison, tolerance=1e-2)

        assert summary["total"] == 2
        assert summary["within_tol"] == 2
        assert summary["all_within_tol"] is True

    def test_empty_comparison_reports_zero_totals(self) -> None:
        empty_rows = _fake_comparison().rows.clear()
        comparison = DecompComparison(
            rows=empty_rows,
            summary=_summarize(empty_rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )

        summary = decomp_compare_summary(comparison, tolerance=1e-2)

        assert summary["within_tol"] == 0
        assert summary["total"] == 0
        assert summary["all_within_tol"] is False


class TestCompareDecompCommand:
    """The ``compare decomp`` subcommand, with the comparison itself stubbed."""

    @staticmethod
    def _invoke(
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        comparison: DecompComparison | None = None,
    ) -> Any:
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        resolved = comparison if comparison is not None else _fake_comparison()
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results",
            lambda *_args, **_kwargs: resolved,
        )
        return CliRunner().invoke(app, argv)

    def test_renders_tables_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)], monkeypatch
        )
        assert result.exit_code == 0
        assert "Operation comparison" in result.stdout
        assert "Final bounds" in result.stdout

    def test_json_carries_the_summary_and_unmapped_codes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """At the default tolerance (1e-2), hydro's ``smape_pct == 10.5`` exceeds
        it while thermal's ``0.0`` does not, so the fixture reports a mismatch."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"], monkeypatch
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert list(payload.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert payload["command"] == "compare decomp"
        assert payload["status"] == "mismatch"
        summary = payload["summary"]
        assert list(summary.keys()) == [
            "stages",
            "variables",
            "unmapped",
            "within_tol",
            "total",
            "all_within_tol",
        ]
        assert summary["stages"] == 1
        assert {row["level"] for row in summary["variables"]} == {"hydro", "thermal"}
        assert summary["unmapped"]["thermal"] == [86, 224]
        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["all_within_tol"] is False

    def test_json_status_is_ok_under_a_loose_tolerance(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A looser ``--tolerance`` flips the same fixture's mismatch to ok."""
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--tolerance",
                "0.5",
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"
        assert payload["summary"]["all_within_tol"] is True

    def test_json_reports_no_comparable_rows_when_comparison_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        empty_rows = _fake_comparison().rows.clear()
        empty_comparison = DecompComparison(
            rows=empty_rows,
            summary=_summarize(empty_rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
            comparison=empty_comparison,
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "no-comparable-rows"
        assert payload["summary"]["variables"] == []
        assert payload["summary"]["within_tol"] == 0
        assert payload["summary"]["total"] == 0
        assert payload["summary"]["all_within_tol"] is False

    def test_unreadable_output_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> DecompComparison:
            raise FileNotFoundError("dec_oper_sist.csv not found")

        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        assert result.exit_code == 2

    def test_writes_artifacts_to_the_default_out_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cobre_output_dir = tmp_path / "cobre"
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(cobre_output_dir)], monkeypatch
        )
        assert result.exit_code == 0
        artifacts = cobre_output_dir / "comparison_artifacts"
        assert (artifacts / "comparison.parquet").exists()
        assert (artifacts / "comparison.json").exists()

    def test_format_and_out_dir_flags_with_json_keep_stdout_pure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cobre_output_dir = tmp_path / "cobre"
        other = tmp_path / "other"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(cobre_output_dir),
                "--format",
                "json",
                "--out-dir",
                str(other),
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["command"] == "compare decomp"
        assert (other / "summary.json").exists()
        assert "Artifacts written to" not in result.stdout

    def test_format_html_writes_a_self_contained_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "artifacts"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--format",
                "html",
                "--out-dir",
                str(out_dir),
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        report_path = out_dir / "report.html"
        assert report_path.exists()
        assert "Operation comparison" in report_path.read_text(encoding="utf-8")

    def test_format_html_advisory_routes_to_stderr_under_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "artifacts"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--format",
                "html",
                "--out-dir",
                str(out_dir),
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        assert (out_dir / "report.html").exists()
        json.loads(result.stdout)  # stdout carries only the JSON verdict
        assert "HTML report written to" not in result.stdout

    def test_partition_missing_output_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """FINDING-1 regression: CobrePartitionMissingError extends
        BridgeError, a hierarchy disjoint from CobreReadError (RuntimeError)
        and FileNotFoundError/ValueError. The compare decomp CLI handler
        must catch it too -- a clean ERROR line + exit 2, not an unhandled
        traceback -- mirroring the compare newave fix and this class's own
        CobreReadError-analogue test above."""
        from cobre_bridge.errors import CobrePartitionMissingError

        sim_dir = tmp_path / "cobre" / "simulation" / "hydro_bus_generation"

        def _boom(*_args: object, **_kwargs: object) -> DecompComparison:
            raise CobrePartitionMissingError(
                f"Cobre output partition not found: {sim_dir}. The "
                "hydro_bus_generation partition is produced by cobre "
                ">= 0.13.0; this output directory may predate that cobre "
                "version.",
                path=str(sim_dir),
            )

        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        # exit_code == 2 (not 1) proves this is the clean typer.Exit(code=2)
        # path, not an unhandled exception caught by CliRunner's default
        # catch_exceptions=True (which would report exit_code == 1).
        assert result.exit_code == 2
        assert "ERROR:" in result.stderr
        assert str(sim_dir) in result.stderr
        assert "0.13.0" in result.stderr
