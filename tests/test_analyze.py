"""Unit tests for the ANALYZE-layer adapters (epic-02 tickets 004-006, 008)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import polars as pl

from cobre_bridge.comparators.analyze import (
    bounds_mismatch_listing,
    bounds_summary_counts,
    build_bounds_dataset,
    build_results_dataset,
    results_footer_counts,
    summary_frame_from_bounds,
    summary_frame_from_results,
    tidy_from_bounds,
    tidy_from_results,
    tidy_percentiles_from_percentile_data,
    tidy_results_dataset,
    top_divergences_from_results,
)
from cobre_bridge.comparators.bounds import BoundComparison
from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.report import (
    build_summary,
    print_bounds_mismatches_from_dataset,
    print_bounds_summary_from_dataset,
    print_mismatches,
    print_results_summary,
    print_results_summary_from_dataset,
    print_summary,
)
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    build_results_summary,
)


def _make_results() -> list[ResultComparison]:
    """Three result comparisons spanning two entity types."""
    return [
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
        ResultComparison(
            entity_type="thermal",
            entity_name="ANGRA",
            newave_code=30,
            cobre_id=2,
            stage=0,
            variable="generation_mw",
            newave_value=0.0,
            cobre_value=5.0,
            abs_diff=5.0,
            rel_diff=None,
        ),
    ]


def _empty_summary() -> pl.DataFrame:
    """An empty summary frame for ``ComparisonDataset`` validation."""
    return ComparisonDataset.empty().summary


def test_tidy_from_results_row_count_and_sources() -> None:
    results = _make_results()

    out = tidy_from_results(results)

    assert out.height == 2 * len(results)
    assert set(out["source"].unique().to_list()) == {"newave", "cobre"}


def test_tidy_from_results_schema_conforms() -> None:
    out = tidy_from_results(_make_results())

    assert list(out.columns) == list(TIDY_SCHEMA)
    expected = {name: dtype() for name, dtype in TIDY_SCHEMA.items()}
    assert dict(out.schema) == expected


def test_tidy_from_results_values_and_sentinels() -> None:
    out = tidy_from_results(_make_results())

    newave_row = out.filter((pl.col("entity_id") == 0) & (pl.col("source") == "newave"))
    assert newave_row["value"].to_list() == [100.0]
    cobre_row = out.filter((pl.col("entity_id") == 0) & (pl.col("source") == "cobre"))
    assert cobre_row["value"].to_list() == [110.0]
    assert out["bus"].unique().to_list() == [-1]
    assert out["block"].unique().to_list() == [-1]


def test_tidy_from_results_empty_returns_schema_frame() -> None:
    out = tidy_from_results([])

    assert out.height == 0
    assert list(out.columns) == list(TIDY_SCHEMA)


def test_tidy_percentiles_unpivots_triplets() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw_p10": [90.0, 45.0],
                "generation_mw_p50": [100.0, 50.0],
                "generation_mw_p90": [110.0, 55.0],
            }
        )
    )

    out = tidy_percentiles_from_percentile_data(pct)

    assert out.height == 6
    assert set(out["source"].unique().to_list()) == {"p10", "p50", "p90"}
    assert out["variable"].unique().to_list() == ["generation_mw"]
    assert out["entity_type"].unique().to_list() == ["hydro"]
    assert out["entity_name"].unique().to_list() == [""]
    assert out["bus"].unique().to_list() == [-1]
    assert out["block"].unique().to_list() == [-1]
    p50 = out.filter((pl.col("entity_id") == 0) & (pl.col("source") == "p50"))
    assert p50["value"].to_list() == [100.0]


def test_tidy_percentiles_multiple_entity_types() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "generation_mw_p10": [1.0],
                "generation_mw_p50": [2.0],
                "generation_mw_p90": [3.0],
            }
        ),
        thermal=pl.DataFrame(
            {
                "entity_id": [5],
                "stage_id": [2],
                "generation_mw_p10": [10.0],
                "generation_mw_p50": [20.0],
                "generation_mw_p90": [30.0],
            }
        ),
    )

    out = tidy_percentiles_from_percentile_data(pct)

    assert set(out["entity_type"].unique().to_list()) == {"hydro", "thermal"}
    assert out.height == 6


def test_tidy_percentiles_empty_returns_schema_frame() -> None:
    out = tidy_percentiles_from_percentile_data(PercentileData())

    assert out.height == 0
    assert list(out.columns) == list(TIDY_SCHEMA)
    expected = {name: dtype() for name, dtype in TIDY_SCHEMA.items()}
    assert dict(out.schema) == expected


def test_tidy_percentiles_missing_identifier_skipped() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0],
                "generation_mw_p10": [1.0],
                "generation_mw_p50": [2.0],
                "generation_mw_p90": [3.0],
            }
        )
    )

    out = tidy_percentiles_from_percentile_data(pct)

    assert out.height == 0
    assert list(out.columns) == list(TIDY_SCHEMA)


def test_tidy_results_dataset_validates() -> None:
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw_p10": [90.0, 45.0],
                "generation_mw_p50": [100.0, 50.0],
                "generation_mw_p90": [110.0, 55.0],
            }
        )
    )

    tidy = tidy_results_dataset(_make_results(), pct)
    dataset = ComparisonDataset(tidy=tidy, summary=_empty_summary(), metadata={})

    dataset.validate()
    assert tidy.height == 2 * len(_make_results()) + 6


def test_tidy_results_dataset_empty_inputs_validate() -> None:
    tidy = tidy_results_dataset([], PercentileData())
    dataset = ComparisonDataset(tidy=tidy, summary=_empty_summary(), metadata={})

    dataset.validate()
    assert tidy.height == 0


def _make_many_results(n: int) -> list[ResultComparison]:
    """``n`` result comparisons with monotonically increasing ``abs_diff``."""
    return [
        ResultComparison(
            entity_type="hydro",
            entity_name=f"PLANT_{i:03d}",
            newave_code=i,
            cobre_id=i,
            stage=i % 4,
            variable="generation_mw",
            newave_value=float(i),
            cobre_value=float(i) + float(i),
            abs_diff=float(i),
            rel_diff=None if i == 0 else 1.0,
        )
        for i in range(n)
    ]


def test_summary_frame_matches_build_results_summary() -> None:
    results = _make_results()

    frame = summary_frame_from_results(results, 1e-2)
    summary = build_results_summary(results, 1e-2)

    assert list(frame.columns) == list(SUMMARY_SCHEMA)
    expected = {name: dtype() for name, dtype in SUMMARY_SCHEMA.items()}
    assert dict(frame.schema) == expected
    assert frame.height == len(summary.by_variable)

    for row in frame.iter_rows(named=True):
        stats = summary.by_variable[row["variable"]]
        assert row["count"] == stats.count
        assert row["mean_abs_diff"] == stats.mean_abs_diff
        assert row["max_abs_diff"] == stats.max_abs_diff
        assert row["mean_smape"] == stats.mean_smape
        assert row["max_smape"] == stats.max_smape
        assert row["within_tol_rate"] == stats.within_tol_rate
        assert row["correlation"] == stats.correlation


def test_top_divergences_sorted_and_truncated() -> None:
    results = _make_many_results(50)

    top = top_divergences_from_results(results, 5)

    assert len(top) == 5
    abs_diffs = [d["abs_diff"] for d in top]
    assert abs_diffs == sorted(abs_diffs, reverse=True)
    assert top[0]["abs_diff"] == max(r.abs_diff for r in results)
    assert set(top[0].keys()) == {
        "entity_type",
        "entity_name",
        "cobre_id",
        "stage",
        "variable",
        "newave_value",
        "cobre_value",
        "abs_diff",
        "rel_diff",
    }


def test_top_divergences_tie_break_deterministic() -> None:
    results = _make_results()  # two rows share abs_diff == 10.0

    top = top_divergences_from_results(results, 3)

    # abs_diff 10.0 ties resolved by entity_name (ITAIPU before TUCURUI),
    # the 5.0 row (ANGRA) comes last despite the earlier name.
    assert [d["entity_name"] for d in top] == ["ITAIPU", "TUCURUI", "ANGRA"]


def test_summary_and_top_divergences_empty_inputs() -> None:
    frame = summary_frame_from_results([], 1e-2)
    top = top_divergences_from_results([])

    assert frame.height == 0
    assert list(frame.columns) == list(SUMMARY_SCHEMA)
    expected = {name: dtype() for name, dtype in SUMMARY_SCHEMA.items()}
    assert dict(frame.schema) == expected
    assert top == []


def test_build_results_dataset_validates_and_carries_metadata() -> None:
    results = _make_results()
    pct = PercentileData(
        hydro=pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw_p10": [90.0, 45.0],
                "generation_mw_p50": [100.0, 50.0],
                "generation_mw_p90": [110.0, 55.0],
            }
        ),
        nw_costs={"deficit": 1.0},
        cobre_costs={"deficit": 2.0},
        nw_bus_names={0: "SUDESTE"},
        nw_hydro_names={0: "ITAIPU", 1: "TUCURUI"},
    )

    dataset = build_results_dataset(results, pct, 1e-2)

    dataset.validate()
    assert len(dataset.metadata["top_divergences"]) == len(results)
    assert dataset.metadata["nw_costs"] == {"deficit": 1.0}
    assert dataset.metadata["cobre_costs"] == {"deficit": 2.0}
    assert dataset.metadata["nw_bus_names"] == {0: "SUDESTE"}
    assert dataset.metadata["nw_hydro_names"] == {0: "ITAIPU", 1: "TUCURUI"}
    assert dataset.tidy.height == 2 * len(results) + 6


def test_build_results_dataset_empty_inputs_validate() -> None:
    dataset = build_results_dataset([], PercentileData(), 1e-2)

    dataset.validate()
    assert dataset.tidy.height == 0
    assert dataset.summary.height == 0


def _make_bounds() -> list[BoundComparison]:
    """Four bound comparisons across two variables, mixing match/mismatch."""
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


def test_tidy_from_bounds_row_count_and_sources() -> None:
    results = _make_bounds()

    out = tidy_from_bounds(results)

    assert out.height == 2 * len(results)
    assert list(out.columns) == list(TIDY_SCHEMA)
    assert set(out["source"].unique().to_list()) == {"newave", "cobre"}
    assert out["bus"].unique().to_list() == [-1]
    assert out["block"].unique().to_list() == [-1]


def test_summary_from_bounds_within_tol_matches_build_summary() -> None:
    results = _make_bounds()

    out = summary_frame_from_bounds(results)
    expected = build_summary(results)

    assert list(out.columns) == list(SUMMARY_SCHEMA)
    expected_schema = {name: dtype() for name, dtype in SUMMARY_SCHEMA.items()}
    assert dict(out.schema) == expected_schema

    for row in out.iter_rows(named=True):
        matches, mismatches = expected.by_variable[row["variable"]]
        assert row["within_tol_rate"] == matches / (matches + mismatches)
        assert row["count"] == matches + mismatches
        assert row["correlation"] is None


def test_bounds_dataset_top_divergences_only_mismatches() -> None:
    results = _make_bounds()

    dataset = build_bounds_dataset(results)

    dataset.validate()
    divergences = dataset.metadata["top_divergences"]
    assert isinstance(divergences, list)
    assert all(d["match"] is False for d in divergences)
    # Two mismatched rows in the fixture, largest abs(diff) first.
    assert [d["entity_name"] for d in divergences] == ["TUCURUI", "CUIABA"]
    assert dataset.tidy.height == 2 * len(results)


def test_bounds_adapters_empty_inputs() -> None:
    tidy = tidy_from_bounds([])
    summary = summary_frame_from_bounds([])
    dataset = build_bounds_dataset([])

    assert tidy.height == 0
    assert list(tidy.columns) == list(TIDY_SCHEMA)
    assert summary.height == 0
    assert list(summary.columns) == list(SUMMARY_SCHEMA)
    dataset.validate()
    assert dataset.metadata["top_divergences"] == []
    assert dataset.metadata["top_divergences"] == []


# -------------------------------------------------------------------
# ticket-008: footer/summary count metadata
# -------------------------------------------------------------------


def test_results_footer_counts_matches_legacy_summary() -> None:
    results = _make_results()

    footer = results_footer_counts(results)
    legacy = build_results_summary(results, 1e-2)

    assert footer["total"] == legacy.total
    assert footer["by_entity_type"] == legacy.by_entity_type


def test_build_results_dataset_carries_footer_counts() -> None:
    results = _make_results()

    dataset = build_results_dataset(results, PercentileData(), 1e-2)

    legacy = build_results_summary(results, 1e-2)
    footer = dataset.metadata["footer_counts"]
    assert isinstance(footer, dict)
    assert footer["total"] == legacy.total
    assert footer["by_entity_type"] == legacy.by_entity_type


def test_bounds_summary_counts_matches_build_summary() -> None:
    results = _make_bounds()

    counts = bounds_summary_counts(results)
    legacy = build_summary(results)

    assert counts["total"] == legacy.total
    assert counts["matches"] == legacy.matches
    assert counts["mismatches"] == legacy.mismatches
    # The metadata pairs are [match, mismatch] lists; legacy uses tuples.
    by_type = counts["by_entity_type"]
    assert isinstance(by_type, dict)
    assert {k: tuple(v) for k, v in by_type.items()} == legacy.by_entity_type
    by_var = counts["by_variable"]
    assert isinstance(by_var, dict)
    assert {k: tuple(v) for k, v in by_var.items()} == legacy.by_variable


def test_build_bounds_dataset_carries_summary_and_mismatch_metadata() -> None:
    results = _make_bounds()

    dataset = build_bounds_dataset(results)

    assert "summary_counts" in dataset.metadata
    assert "mismatch_listing" in dataset.metadata
    listing = dataset.metadata["mismatch_listing"]
    assert isinstance(listing, dict)
    # Two mismatches in the fixture; sorted by raw diff descending.
    assert listing["total"] == 2
    rows = listing["rows"]
    assert isinstance(rows, list)
    assert [r["entity_name"] for r in rows] == ["TUCURUI", "CUIABA"]
    assert rows[0]["newave_code"] == 20


def test_bounds_mismatch_listing_sorts_by_raw_diff_descending() -> None:
    results = _make_bounds()

    listing = bounds_mismatch_listing(results, max_rows=50)

    diffs = [r["diff"] for r in listing["rows"]]
    assert diffs == sorted(diffs, reverse=True)


def test_bounds_mismatch_listing_respects_max_rows_but_keeps_total() -> None:
    results = _make_bounds()

    listing = bounds_mismatch_listing(results, max_rows=1)

    assert listing["total"] == 2
    rows = listing["rows"]
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["entity_name"] == "TUCURUI"


# -------------------------------------------------------------------
# ticket-008: byte-identical legacy-vs-dataset console printers
# -------------------------------------------------------------------


def _capture(func: object, *args: object) -> str:
    """Run a stdout-writing printer and return the captured text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args)  # type: ignore[operator]
    return buffer.getvalue()


def test_print_results_summary_from_dataset_matches_legacy() -> None:
    results = _make_results()
    nw, cobre = Path("/fake/nw"), Path("/fake/cobre")

    legacy_summary = build_results_summary(results, 1e-2)
    dataset = build_results_dataset(results, PercentileData(), 1e-2)

    legacy_out = _capture(print_results_summary, legacy_summary, nw, cobre)
    dataset_out = _capture(print_results_summary_from_dataset, dataset, nw, cobre)

    assert dataset_out == legacy_out


def test_print_results_summary_from_dataset_matches_legacy_empty() -> None:
    nw, cobre = Path("/fake/nw"), Path("/fake/cobre")

    legacy_summary = build_results_summary([], 1e-2)
    dataset = build_results_dataset([], PercentileData(), 1e-2)

    legacy_out = _capture(print_results_summary, legacy_summary, nw, cobre)
    dataset_out = _capture(print_results_summary_from_dataset, dataset, nw, cobre)

    assert dataset_out == legacy_out


def test_print_bounds_summary_from_dataset_matches_legacy() -> None:
    results = _make_bounds()
    nw, cobre = Path("/fake/nw"), Path("/fake/cobre")
    tol = 1e-3

    legacy_summary = build_summary(results)
    dataset = build_bounds_dataset(results)

    legacy_out = _capture(print_summary, legacy_summary, nw, cobre, tol)
    dataset_out = _capture(print_bounds_summary_from_dataset, dataset, nw, cobre, tol)

    assert dataset_out == legacy_out


def test_print_bounds_summary_from_dataset_matches_legacy_empty() -> None:
    nw, cobre = Path("/fake/nw"), Path("/fake/cobre")
    tol = 1e-3

    legacy_summary = build_summary([])
    dataset = build_bounds_dataset([])

    legacy_out = _capture(print_summary, legacy_summary, nw, cobre, tol)
    dataset_out = _capture(print_bounds_summary_from_dataset, dataset, nw, cobre, tol)

    assert dataset_out == legacy_out


def test_print_bounds_mismatches_from_dataset_matches_legacy() -> None:
    results = _make_bounds()
    dataset = build_bounds_dataset(results)

    legacy_out = _capture(print_mismatches, results)
    dataset_out = _capture(print_bounds_mismatches_from_dataset, dataset)

    assert dataset_out == legacy_out


def test_print_bounds_mismatches_from_dataset_matches_legacy_no_mismatches() -> None:
    results = [r for r in _make_bounds() if r.match]
    dataset = build_bounds_dataset(results)

    legacy_out = _capture(print_mismatches, results)
    dataset_out = _capture(print_bounds_mismatches_from_dataset, dataset)

    assert dataset_out == legacy_out
    assert dataset_out == "No mismatches found.\n"
