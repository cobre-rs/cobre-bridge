"""Integration tests for the compare bounds and compare results pipelines.

Tests the full comparison flow using mocked inewave readers and fixture
data.  Verifies that the pipeline connects correctly from CLI through
alignment, computation, comparison, and report generation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cobre_bridge.comparators.alignment import (
    EntityAlignment,
    HydroEntity,
    ThermalEntity,
)
from cobre_bridge.comparators.bounds import (
    _bounds_match,
    _is_effectively_infinite,
)
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    ResultsSummary,
    build_results_summary,
)
from cobre_bridge.id_map import NewaveIdMap

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
    @staticmethod
    def _make_alignment() -> EntityAlignment:
        return EntityAlignment(
            hydros=[
                HydroEntity(
                    newave_code=1,
                    cobre_id=0,
                    name="PLANT_A",
                    has_reservoir=True,
                ),
            ],
            thermals=[
                ThermalEntity(
                    newave_code=10,
                    cobre_id=0,
                    name="THERMAL_A",
                ),
            ],
            lines=[],
            num_newave_stages=3,
        )

    @staticmethod
    def _make_id_map() -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1],
            thermal_codes=[10],
        )

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
        from cobre_bridge.comparators.report import (
            print_results_summary,
        )

        summary = ResultsSummary(total=0)
        print_results_summary(summary, Path("/fake/nw"), Path("/fake/cobre"))
        out = capsys.readouterr().out
        assert "Results Comparison" in out

    def test_print_results_summary_with_data(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge.comparators.report import (
            print_results_summary,
        )
        from cobre_bridge.comparators.results import (
            ResultVariableStats,
        )

        summary = ResultsSummary(
            total=100,
            by_entity_type={"hydro": 80, "thermal": 20},
            by_variable={
                "generation_mw": ResultVariableStats(
                    count=100,
                    mean_abs_diff=0.5,
                    max_abs_diff=3.0,
                    mean_rel_diff=0.01,
                    max_rel_diff=0.05,
                    correlation=0.9998,
                ),
            },
        )
        print_results_summary(summary, Path("/nw"), Path("/cobre"))
        out = capsys.readouterr().out
        assert "generation_mw" in out
        assert "100" in out


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
        assert "plotly-2.35.2.min.js" in html
        assert "<p>Hello</p>" in html

    def test_build_comparison_report_no_crash(self) -> None:
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )

        html = build_comparison_report([])
        assert "<!DOCTYPE html>" in html
        assert "Cobre vs NEWAVE" in html

    def test_build_comparison_report_with_data(self) -> None:
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )

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
        html = build_comparison_report(results)
        assert "<!DOCTYPE html>" in html
        assert "Convergence" in html


# -------------------------------------------------------------------
# Bounds from inputs tests
# -------------------------------------------------------------------


class TestBoundsFromInputs:
    def test_compute_hydro_bounds_no_modif(self) -> None:
        """Empty dict when MODIF is absent."""
        from cobre_bridge.comparators.bounds_from_inputs import (
            compute_hydro_bounds,
        )

        nw_files = MagicMock()
        nw_files.modif = None
        id_map = MagicMock()

        result = compute_hydro_bounds(nw_files, id_map)
        assert result == {}

    def test_compute_thermal_bounds_no_expt_no_manutt(
        self,
    ) -> None:
        """Empty dict when neither expt.dat nor manutt.dat present."""
        from cobre_bridge.comparators.bounds_from_inputs import (
            compute_thermal_bounds,
        )

        nw_files = MagicMock()
        nw_files.expt = None
        nw_files.manutt = None
        id_map = MagicMock()

        result = compute_thermal_bounds(nw_files, id_map)
        assert result == {}

    def test_compute_line_bounds_no_limits(self) -> None:
        """Empty dict when sistema has no interchange limits."""
        from cobre_bridge.comparators.bounds_from_inputs import (
            compute_line_bounds,
        )

        nw_files = MagicMock()
        id_map = MagicMock()

        with patch("inewave.newave.Sistema") as mock_sis:
            mock_inst = MagicMock()
            mock_inst.limites_intercambio = None
            mock_sis.read.return_value = mock_inst
            result = compute_line_bounds(nw_files, id_map)

        assert result == {}


# -------------------------------------------------------------------
# Edge cases and error handling
# -------------------------------------------------------------------


class TestEdgeCases:
    """Test graceful handling of missing/empty data."""

    def test_newave_readers_missing_dir(self, tmp_path: Path) -> None:
        """NEWAVE readers return empty DataFrames when dir missing."""
        from cobre_bridge.comparators.newave_readers import (
            read_medias_hydro,
            read_medias_system,
            read_medias_thermal,
            read_pmo_convergence,
            read_pmo_productivity,
        )

        fake_dir = tmp_path / "nonexistent"
        assert read_medias_hydro(fake_dir).is_empty()
        assert read_medias_thermal(fake_dir).is_empty()
        assert read_medias_system(fake_dir).is_empty()
        assert read_pmo_convergence(fake_dir).is_empty()
        assert read_pmo_productivity(fake_dir).is_empty()

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

    def test_empty_alignment_produces_no_results(self) -> None:
        """Alignment with no entities produces empty comparison."""
        summary = build_results_summary([])
        assert summary.total == 0

    def test_html_report_with_empty_results(self) -> None:
        """HTML report renders without error on empty results."""
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )

        html = build_comparison_report([])
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
            # Hydro entries — two stages
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
            # Thermal entry
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
            # Bus entry
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
            # Convergence entry
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
        from cobre_bridge.comparators.html_report import COMPARISON_TABS
        from cobre_bridge.comparators.report_builder import build_comparison_report

        results = self._make_results()
        pctiles = self._make_percentile_data()

        html = build_comparison_report(results, pctiles)

        assert "<!DOCTYPE html>" in html
        for tab_id, _label in COMPARISON_TABS:
            assert tab_id in html, f"Tab section '{tab_id}' missing from HTML output"

    def test_comparison_report_contains_plotly_chart(self) -> None:
        """Full pipeline with real data produces at least one Plotly chart."""
        from cobre_bridge.comparators.report_builder import build_comparison_report

        results = self._make_results()
        pctiles = self._make_percentile_data()

        html = build_comparison_report(results, pctiles)

        assert "Plotly.newPlot" in html
