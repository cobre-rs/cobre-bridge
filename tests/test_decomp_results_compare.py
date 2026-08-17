"""Tests for the DECOMP-vs-Cobre results comparison slice."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.decomp_html_report import build_decomp_comparison_report
from cobre_bridge.comparators.decomp_results import (
    _BUS_VARIABLES,
    _CANONICAL_VARIABLE,
    _HYDRO_VARIABLES,
    _THERMAL_VARIABLES,
    DecompComparison,
    _AlignedDecompFrames,
    _map_entities,
    _result_comparisons,
    _scenario_mean,
    _stage_rows,
    _summarize,
    _tidy,
    build_decomp_dataset,
    compare_decomp_results,
)
from cobre_bridge.verdict import decomp_compare_summary, decomp_dataset_summary


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

    def test_keeps_the_original_code_as_newave_code(self) -> None:
        """``build_decomp_dataset`` needs the reference code back to fill
        ``ResultComparison.newave_code`` -- ``_map_entities`` must not drop it
        once it has been used to derive ``entity_id``."""
        frame = pl.DataFrame({"estagio": [1, 2], "codigo_usina": [10, 10]})
        mapped, _unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["newave_code"].to_list() == [10, 10]


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


class TestResultComparisons:
    """``_result_comparisons`` is the ``ResultComparison`` counterpart of
    ``_tidy``, feeding ``build_decomp_dataset`` instead of ``DecompComparison``."""

    def _pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        source = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "newave_code": [10, 11],
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

    def test_emits_one_result_comparison_per_row_with_the_canonical_variable(
        self,
    ) -> None:
        source, cobre = self._pair()
        results = _result_comparisons(
            source, cobre, _HYDRO_VARIABLES, names={0: "A", 1: "B"}
        )
        assert {r.variable for r in results} == {"generation_mw"}
        by_id = {r.cobre_id: r for r in results}
        assert by_id[0].newave_code == 10
        assert by_id[0].entity_name == "A"
        assert by_id[0].entity_type == "hydro"
        assert by_id[0].stage == 0
        assert by_id[0].newave_value == 100.0
        assert by_id[0].cobre_value == 90.0
        assert by_id[0].abs_diff == pytest.approx(10.0)
        assert by_id[0].rel_diff == pytest.approx(0.1)
        assert by_id[1].newave_code == 11
        assert by_id[1].abs_diff == pytest.approx(0.0)

    def test_variables_missing_on_either_side_are_skipped(self) -> None:
        source, cobre = self._pair()
        results = _result_comparisons(source, cobre, _HYDRO_VARIABLES, names={})
        # Only generation's columns are present in both frames.
        assert {r.variable for r in results} == {"generation_mw"}

    def test_empty_join_yields_no_results(self) -> None:
        source = pl.DataFrame(
            {
                "entity_id": [0],
                "newave_code": [10],
                "stage_id": [0],
                "geracao_MW": [1.0],
            }
        )
        cobre = pl.DataFrame(
            {"entity_id": [9], "stage_id": [9], "generation_mw": [1.0]}
        )
        assert _result_comparisons(source, cobre, _HYDRO_VARIABLES, names={}) == []

    def test_canonical_variable_covers_all_eight_today_variables(self) -> None:
        """D-SOURCE-TOKEN-adjacent guard: every ``_Variable`` spec this module
        ships must resolve to a canonical chart name -- a spec with no entry
        would raise a ``KeyError`` deep inside ``_result_comparisons``."""
        all_vars = _HYDRO_VARIABLES + _THERMAL_VARIABLES + _BUS_VARIABLES
        assert len(all_vars) == 8
        canonical_names = {_CANONICAL_VARIABLE[(v.level, v.name)] for v in all_vars}
        assert canonical_names == {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
        }


def _aligned_fixture() -> _AlignedDecompFrames:
    """One hydro plant, one thermal plant, one bus -- already aligned to Cobre
    ids/stages, matching the shape :func:`_read_aligned_frames` returns."""
    source_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "newave_code": [10, 20],
            "stage_id": [0, 0],
            "geracao_MW": [120.0, 60.0],
            "vazao_turbinada_m3s": [80.0, 40.0],
            "vazao_vertida_m3s": [0.0, 0.0],
            "vazao_defluente_m3s": [80.0, 40.0],
            "volume_util_final_hm3": [500.0, 300.0],
        }
    )
    cobre_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw": [110.0, 60.0],
            "turbined_m3s": [78.0, 40.0],
            "spillage_m3s": [0.0, 0.0],
            "outflow_m3s": [78.0, 40.0],
            "useful_storage_hm3": [480.0, 300.0],
        }
    )
    source_thermal = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [5],
            "stage_id": [0],
            "geracao_MW": [30.0],
        }
    )
    cobre_thermal = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "generation_mw": [28.0]}
    )
    source_bus = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [1],
            "stage_id": [0],
            "deficit_MW": [0.0],
            "cmo": [45.0],
        }
    )
    cobre_bus = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "deficit_mw": [0.0], "spot_price": [44.0]}
    )
    return _AlignedDecompFrames(
        source_hydro=source_hydro,
        source_thermal=source_thermal,
        source_bus=source_bus,
        cobre_hydro=cobre_hydro,
        cobre_thermal=cobre_thermal,
        cobre_bus=cobre_bus,
        hydro_names={0: "A", 1: "B"},
        thermal_names={0: "T"},
        bus_names={0: "SE"},
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


def _patch_aligned_frames(
    monkeypatch: pytest.MonkeyPatch, aligned: _AlignedDecompFrames
) -> None:
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._read_aligned_frames",
        lambda *_args, **_kwargs: aligned,
    )
    # ``compare_decomp_results`` also reads the convergence report directly
    # (outside ``_read_aligned_frames``); stub it so the parity fixture never
    # has to touch a real deck/output directory.
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._convergence",
        lambda *_args, **_kwargs: pl.DataFrame(schema={"iteration": pl.Int64}),
    )


class TestBuildDecompDataset:
    """``build_decomp_dataset`` assembles the canonical dataset for the
    current 8 DECOMP variables via the shared ``_read_aligned_frames`` seam."""

    def test_dataset_validates_with_the_eight_canonical_variables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset.validate()
        assert set(dataset.summary["variable"].to_list()) == {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
        }

    def test_tidy_sources_are_newave_and_cobre_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert set(dataset.tidy["source"].unique().to_list()) == {"newave", "cobre"}

    def test_hydro_storage_rows_compare_useful_volume(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The cobre-side value must be ``useful_storage_hm3`` (already
        ``storage_final_hm3 - min_storage_hm3`` upstream in ``_cobre_hydro``),
        not the raw absolute ``storage_final_hm3``."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        cobre_storage = (
            dataset.tidy.filter(
                (pl.col("variable") == "storage_final_hm3")
                & (pl.col("source") == "cobre")
            )
            .sort("entity_id")["value"]
            .to_list()
        )
        assert cobre_storage == [480.0, 300.0]

    def test_unmapped_codes_surface_in_metadata_and_are_excluded_from_tidy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [86, 224],
            "bus": [],
        }
        thermal_codes = {
            r.newave_code
            for r in dataset.metadata["results"]
            if r.entity_type == "thermal"
        }
        assert 86 not in thermal_codes
        assert 224 not in thermal_codes

    def test_calls_the_shared_stat_kernel_not_a_local_reimplementation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``top_divergences``/``footer_counts`` are only populated by
        ``analyze.build_results_dataset``, so their presence is proof that
        function ran instead of a locally re-derived summary."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert "top_divergences" in dataset.metadata
        assert "footer_counts" in dataset.metadata

    def test_empty_comparison_returns_a_schema_valid_empty_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fixture = _aligned_fixture()
        empty = _AlignedDecompFrames(
            source_hydro=fixture.source_hydro.clear(),
            source_thermal=fixture.source_thermal.clear(),
            source_bus=fixture.source_bus.clear(),
            cobre_hydro=fixture.cobre_hydro.clear(),
            cobre_thermal=fixture.cobre_thermal.clear(),
            cobre_bus=fixture.cobre_bus.clear(),
            hydro_names={},
            thermal_names={},
            bus_names={},
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        _patch_aligned_frames(monkeypatch, empty)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset.validate()
        assert dataset.tidy.is_empty()
        assert dataset.summary.is_empty()
        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [],
            "bus": [],
        }


class TestBuildDecompDatasetParityWithLegacyComparison:
    """No drift from the migration: on the same aligned fixture,
    ``build_decomp_dataset`` and the legacy ``compare_decomp_results`` must
    report the same per-(entity, stage) newave/cobre values."""

    def test_every_legacy_row_has_a_matching_dataset_pair(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        legacy = compare_decomp_results(tmp_path, tmp_path)
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert legacy.rows.height > 0  # the fixture must exercise real rows
        for row in legacy.rows.iter_rows(named=True):
            canonical = _CANONICAL_VARIABLE[(row["level"], row["variable"])]
            match = dataset.tidy.filter(
                (pl.col("entity_type") == row["level"])
                & (pl.col("variable") == canonical)
                & (pl.col("entity_id") == row["entity_id"])
                & (pl.col("stage") == row["stage_id"])
            )
            newave_value = match.filter(pl.col("source") == "newave")["value"].to_list()
            cobre_value = match.filter(pl.col("source") == "cobre")["value"].to_list()
            assert newave_value == pytest.approx([row["source"]])
            assert cobre_value == pytest.approx([row["cobre"]])

    def test_unmapped_dict_is_identical_between_entry_points(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        legacy = compare_decomp_results(tmp_path, tmp_path)
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["unmapped"] == legacy.unmapped


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


def _fake_dataset(*, all_within_tol: bool = False) -> ComparisonDataset:
    """The canonical-dataset counterpart of ``_fake_comparison``: two stages,
    two variables. ``generation_mw`` always matches; ``turbined_m3s`` diverges
    unless *all_within_tol* asks for a fully-passing dataset instead."""
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "hydro", "hydro", "hydro"],
            "entity_id": [0, 0, 0, 0],
            "entity_name": ["A", "A", "A", "A"],
            "bus": [-1, -1, -1, -1],
            "stage": [0, 0, 1, 1],
            "block": [-1, -1, -1, -1],
            "variable": [
                "generation_mw",
                "generation_mw",
                "turbined_m3s",
                "turbined_m3s",
            ],
            "source": ["newave", "cobre", "newave", "cobre"],
            "value": [100.0, 100.0, 100.0, 100.0 if all_within_tol else 90.0],
        },
        schema=TIDY_SCHEMA,
    )
    turbined_within_tol_rate = 1.0 if all_within_tol else 0.0
    turbined_smape = 0.0 if all_within_tol else 0.12
    summary = pl.DataFrame(
        {
            "variable": ["generation_mw", "turbined_m3s"],
            "count": [1, 1],
            "mean_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "max_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "mean_smape": [0.0, turbined_smape],
            "max_smape": [0.0, turbined_smape],
            "within_tol_rate": [1.0, turbined_within_tol_rate],
            "correlation": [1.0, 0.9],
        },
        schema=SUMMARY_SCHEMA,
    )
    return ComparisonDataset(
        tidy=tidy,
        summary=summary,
        metadata={"unmapped": {"hydro": [], "thermal": [86, 224], "bus": []}},
    )


def _empty_fake_dataset() -> ComparisonDataset:
    return ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )


class TestDecompDatasetSummary:
    """``decomp_dataset_summary`` -- the superset ``--json`` payload builder
    that supersedes ``decomp_compare_summary`` at the ``compare decomp``
    call site (the legacy summary itself is untouched, per D-STRANGLER)."""

    def test_returns_the_superset_shape_in_the_documented_key_order(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert list(summary.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
            "stages",
            "variables",
            "unmapped",
        ]

    def test_headline_fields_match_build_compare_verdict(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["worst_variable"] == "turbined_m3s"
        assert summary["worst_smape"] == pytest.approx(0.12)
        assert summary["all_within_tol"] is False

    def test_stages_counts_distinct_tidy_stage_values(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["stages"] == 2

    def test_variables_is_the_dataset_summary_verbatim(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["variables"] == dataset.summary.to_dicts()

    def test_unmapped_is_sourced_from_dataset_metadata_with_int_codes(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["unmapped"] == {"hydro": [], "thermal": [86, 224], "bus": []}
        assert all(
            isinstance(code, int)
            for codes in summary["unmapped"].values()
            for code in codes
        )

    def test_all_within_tol_dataset_reports_no_worst_clause(self) -> None:
        dataset = _fake_dataset(all_within_tol=True)

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 2
        assert summary["total"] == 2
        assert summary["all_within_tol"] is True
        assert summary["worst_variable"] is None
        assert summary["worst_smape"] == 0.0

    def test_empty_dataset_reports_zero_totals_and_no_stages(self) -> None:
        dataset = _empty_fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 0
        assert summary["total"] == 0
        assert summary["all_within_tol"] is False
        assert summary["stages"] == 0
        assert summary["variables"] == []
        assert summary["unmapped"] == {"hydro": [], "thermal": [], "bus": []}


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
        dataset: ComparisonDataset | None = None,
    ) -> Any:
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        resolved_comparison = (
            comparison if comparison is not None else _fake_comparison()
        )
        resolved_dataset = dataset if dataset is not None else _fake_dataset()
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results",
            lambda *_args, **_kwargs: resolved_comparison,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.build_decomp_dataset",
            lambda *_args, **_kwargs: resolved_dataset,
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

    def test_headline_leads_stdout_on_a_diverging_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The shared ``build_compare_verdict`` headline, from ``_fake_dataset``'s
        mismatch (1/2 within tol, worst ``turbined_m3s`` at 12% sMAPE), leads
        stdout ahead of the legacy per-variable table."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)], monkeypatch
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "⚠ 1/2 variables within tol — worst: turbined_m3s sMAPE 12%"
        assert "Operation comparison" in result.stdout
        assert result.stdout.index("Operation comparison") > result.stdout.index(
            lines[0]
        )

    def test_headline_leads_stdout_when_all_within_tol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)],
            monkeypatch,
            dataset=_fake_dataset(all_within_tol=True),
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "✓ 2/2 variables within tol"
        assert "Operation comparison" in result.stdout

    def test_json_carries_the_summary_and_unmapped_codes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_fake_dataset``'s ``turbined_m3s`` row (``within_tol_rate=0.0``)
        diverges while ``generation_mw`` (``within_tol_rate=1.0``) does not, so
        the default fixture reports a mismatch."""
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
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
            "stages",
            "variables",
            "unmapped",
        ]
        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["worst_variable"] == "turbined_m3s"
        assert summary["worst_smape"] == pytest.approx(0.12)
        assert summary["all_within_tol"] is False
        assert summary["stages"] == 2
        assert {row["variable"] for row in summary["variables"]} == {
            "generation_mw",
            "turbined_m3s",
        }
        assert summary["unmapped"]["thermal"] == [86, 224]

    def test_json_status_is_ok_when_dataset_reports_all_within_tol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the dataset's ``within_tol_rate`` is 1.0 for every variable
        (the tolerance was already applied upstream when the caller built the
        dataset), the CLI reports ``ok``."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
            dataset=_fake_dataset(all_within_tol=True),
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
            dataset=_empty_fake_dataset(),
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "no-comparable-rows"
        assert payload["summary"]["stages"] == 0
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
