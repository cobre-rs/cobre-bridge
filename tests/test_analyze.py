"""Unit tests for the ANALYZE-layer adapters (epic-02 tickets 004-006, 008)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import cast

import pandas as pd
import polars as pl

from cobre_bridge import diagnostics as dx
from cobre_bridge.comparators.analyze import (
    aggregate_percentile_band,
    bounds_mismatch_listing,
    bounds_summary_counts,
    build_bounds_dataset,
    build_results_dataset,
    bus_groups_and_pct,
    cobre_sum_and_newave_sin,
    per_bus_band_from_pct,
    per_bus_sums_from_frame,
    per_bus_sums_from_results,
    per_stage_sum_from_frame,
    per_stage_sum_from_results,
    plant_percentile_arrays,
    results_footer_counts,
    spillage_lookups,
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
    print_mismatches,
)
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    build_results_summary,
)
from cobre_bridge.comparators.verdict import (
    CompareVerdict,
    build_compare_verdict,
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


def test_build_results_dataset_carries_render_only_keys_in_memory() -> None:
    """The render-only keys (results + drained pct frames) live in metadata."""
    results = _make_results()
    pct = PercentileData(
        thermal=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        nw_max_stage=12,
        cobre_training_seconds=3.5,
    )

    dataset = build_results_dataset(results, pct, 1e-2)

    assert dataset.metadata["results"] == list(results)
    assert dataset.metadata["nw_max_stage"] == 12
    assert dataset.metadata["cobre_training_seconds"] == 3.5
    assert isinstance(dataset.metadata["thermal"], pl.DataFrame)
    assert isinstance(dataset.metadata["gc_constraints"], list)


def test_metadata_json_excludes_render_only_keys(tmp_path: Path) -> None:
    """``to_dir`` writes a small metadata.json without the render-only inputs.

    The bulky render inputs (``results`` plus the drained ``PercentileData``
    frames) are in :data:`RENDER_ONLY_METADATA_KEYS`, so ``to_dir`` must NOT
    serialize them — yet the genuine provenance key ``top_divergences`` is kept.
    """
    import json

    from cobre_bridge.comparators.dataset import RENDER_ONLY_METADATA_KEYS

    results = _make_results()
    pct = PercentileData(
        hydro=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        thermal=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        nw_costs={"deficit": 1.0},
        nw_max_stage=5,
    )
    dataset = build_results_dataset(results, pct, 1e-2)

    paths = dataset.to_dir(tmp_path)
    metadata_path = paths[2]
    written = json.loads(metadata_path.read_text(encoding="utf-8"))

    # No render-only key leaked into the artifact.
    for key in RENDER_ONLY_METADATA_KEYS:
        assert key not in written, f"render-only key {key!r} leaked into metadata.json"
    assert "results" not in written
    assert "hydro" not in written
    # Genuine provenance / JSON-native keys survive.
    assert "top_divergences" in written
    assert written["nw_costs"] == {"deficit": 1.0}


def test_render_only_covers_all_frame_metadata_keys(tmp_path: Path) -> None:
    """Every frame-valued metadata key must be listed in RENDER_ONLY_METADATA_KEYS.

    Guards the sync invariant between :func:`build_results_dataset` and
    :data:`RENDER_ONLY_METADATA_KEYS`: a contributor who adds a new frame-valued
    metadata key but forgets to register it here would otherwise only discover it
    as a runtime ``TypeError`` from :meth:`ComparisonDataset.to_dir` (which cannot
    serialize a bare ``pl``/``pd`` frame unless it is skipped as render-only).
    This asserts the containment up front, and round-trips through ``to_dir`` to
    confirm no frame key leaks into ``metadata.json``.
    """
    import json

    from cobre_bridge.comparators.dataset import RENDER_ONLY_METADATA_KEYS

    results = _make_results()
    pct = PercentileData(
        hydro=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        thermal=pl.DataFrame({"entity_id": [0], "stage_id": [0]}),
        line_bounds=pd.DataFrame({"line_id": [0], "value": [1.0]}),
        nw_costs={"deficit": 1.0},
    )

    dataset = build_results_dataset(results, pct, 1e-2)

    frame_keys = {
        key
        for key, value in dataset.metadata.items()
        if isinstance(value, (pl.DataFrame, pd.DataFrame))
    }
    assert frame_keys, "expected at least one frame-valued metadata key in the fixture"
    missing = frame_keys - RENDER_ONLY_METADATA_KEYS
    assert not missing, (
        f"frame-valued metadata keys {sorted(missing)} are not listed in "
        f"RENDER_ONLY_METADATA_KEYS; add them or to_dir will raise a TypeError"
    )

    paths = dataset.to_dir(tmp_path)
    written = json.loads(paths[2].read_text(encoding="utf-8"))
    for key in frame_keys:
        assert key not in written, (
            f"frame metadata key {key!r} leaked into metadata.json"
        )


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
    summary_counts = dataset.metadata["summary_counts"]
    assert isinstance(summary_counts, dict)
    assert summary_counts["total"] == 0
    mismatch_listing = dataset.metadata["mismatch_listing"]
    assert isinstance(mismatch_listing, dict)
    assert mismatch_listing["total"] == 0


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


# ---------------------------------------------------------------------------
# ticket-009: chart-aggregation primitives
# ---------------------------------------------------------------------------


def _band_result(stage: int, cobre_id: int, var: str) -> ResultComparison:
    return ResultComparison(
        entity_type="hydro",
        entity_name=f"H{cobre_id}",
        newave_code=cobre_id + 10,
        cobre_id=cobre_id,
        stage=stage,
        variable=var,
        newave_value=float(stage * 100 + cobre_id),
        cobre_value=float(stage * 100 + cobre_id) - 5.0,
        abs_diff=5.0,
        rel_diff=None,
    )


def test_aggregate_percentile_band_sums_across_entities() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "storage_final_hm3_p10": [80.0, 160.0, 85.0, 165.0],
            "storage_final_hm3_p90": [100.0, 200.0, 105.0, 205.0],
        }
    )
    p10, p90 = aggregate_percentile_band(pct, "storage_final_hm3", [1, 2], None)
    assert p10 == [240.0, 250.0]
    assert p90 == [300.0, 310.0]


def test_aggregate_percentile_band_missing_column_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            # only the _p90 column present -> _p10 missing
            "storage_final_hm3_p90": [100.0, 200.0],
        }
    )
    assert aggregate_percentile_band(pct, "storage_final_hm3", [1], None) == ([], [])


def test_aggregate_percentile_band_none_returns_empty() -> None:
    assert aggregate_percentile_band(None, "storage_final_hm3", [1, 2], None) == (
        [],
        [],
    )
    empty = pl.DataFrame()
    assert aggregate_percentile_band(empty, "storage_final_hm3", [1], None) == ([], [])


def test_aggregate_percentile_band_missing_stage_defaults_to_zero() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            "v_p10": [10.0],
            "v_p90": [20.0],
        }
    )
    # stage 3 is absent from the frame -> 0.0, not a KeyError.
    p10, p90 = aggregate_percentile_band(pct, "v", [1, 3], None)
    assert p10 == [10.0, 0.0]
    assert p90 == [20.0, 0.0]


def test_aggregate_percentile_band_filters_entity_ids() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            "v_p10": [80.0, 160.0],
            "v_p90": [100.0, 200.0],
        }
    )
    # keep only entity 1.
    p10, p90 = aggregate_percentile_band(pct, "v", [1], {1})
    assert p10 == [160.0]
    assert p90 == [200.0]


def test_per_stage_sum_from_results_filters_entity_type_and_variable() -> None:
    results = [
        _band_result(stage=1, cobre_id=0, var="storage_final_hm3"),
        _band_result(stage=1, cobre_id=1, var="storage_final_hm3"),
        _band_result(stage=2, cobre_id=0, var="storage_final_hm3"),
        # different variable / type -> excluded.
        _band_result(stage=1, cobre_id=9, var="other"),
        ResultComparison(
            entity_type="thermal",
            entity_name="T",
            newave_code=99,
            cobre_id=99,
            stage=1,
            variable="storage_final_hm3",
            newave_value=1000.0,
            cobre_value=1.0,
            abs_diff=999.0,
            rel_diff=None,
        ),
    ]
    nw, cb, matched = per_stage_sum_from_results(results, "hydro", "storage_final_hm3")
    # newave_value = stage*100 + cobre_id; cobre_value = that - 5.
    # stage 1: entities 0,1 -> 100 + 101 = 201; stage 2: entity 0 -> 200.
    assert nw == {1: 100.0 + 101.0, 2: 200.0}
    assert cb == {1: 95.0 + 96.0, 2: 195.0}
    assert matched == {0, 1}


def test_per_stage_sum_from_results_empty_variable_matches_all() -> None:
    results = [
        ResultComparison(
            entity_type="thermal",
            entity_name="T1",
            newave_code=21,
            cobre_id=5,
            stage=1,
            variable="generation_mw",
            newave_value=50.0,
            cobre_value=45.0,
            abs_diff=5.0,
            rel_diff=None,
        ),
        ResultComparison(
            entity_type="thermal",
            entity_name="T2",
            newave_code=22,
            cobre_id=6,
            stage=1,
            variable="commitment",  # different variable, same type -> still kept
            newave_value=1.0,
            cobre_value=1.0,
            abs_diff=0.0,
            rel_diff=None,
        ),
    ]
    nw, cb, matched = per_stage_sum_from_results(results, "thermal", "")
    assert nw == {1: 51.0}
    assert cb == {1: 46.0}
    assert matched == {5, 6}


def test_per_stage_sum_from_results_no_match_returns_empty() -> None:
    results = [_band_result(stage=1, cobre_id=0, var="v")]
    nw, cb, matched = per_stage_sum_from_results(results, "bus", "v")
    assert nw == {}
    assert cb == {}
    assert matched == set()


def test_per_stage_sum_from_frame_matches_legacy() -> None:
    df = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [2, 2, 1, 1],
            "turbined_m3s": [10.0, 20.0, 30.0, 40.0],
        }
    )
    out = per_stage_sum_from_frame(df, "turbined_m3s", None)
    # sorted by stage_id.
    assert list(out.keys()) == [1, 2]
    assert out == {1: 70.0, 2: 30.0}


def test_per_stage_sum_from_frame_filters_matched_ids() -> None:
    df = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            "turbined_m3s": [30.0, 40.0],
        }
    )
    assert per_stage_sum_from_frame(df, "turbined_m3s", {1}) == {1: 40.0}


def test_per_stage_sum_from_frame_empty_returns_empty() -> None:
    assert per_stage_sum_from_frame(None, "turbined_m3s", None) == {}
    assert per_stage_sum_from_frame(pl.DataFrame(), "turbined_m3s", None) == {}
    # frame present but column absent.
    df = pl.DataFrame({"entity_id": [0], "stage_id": [1], "other": [1.0]})
    assert per_stage_sum_from_frame(df, "turbined_m3s", None) == {}


# ---------------------------------------------------------------------------
# ticket-010: per-bus roll-up + per-plant percentile extraction
# ---------------------------------------------------------------------------


def _hydro_row(
    cobre_id: int,
    stage: int,
    variable: str,
    nw: float,
    cb: float,
    *,
    entity_type: str = "hydro",
) -> ResultComparison:
    return ResultComparison(
        entity_type=entity_type,
        entity_name=f"H{cobre_id}",
        newave_code=cobre_id + 10,
        cobre_id=cobre_id,
        stage=stage,
        variable=variable,
        newave_value=nw,
        cobre_value=cb,
        abs_diff=abs(nw - cb),
        rel_diff=None,
    )


def test_per_bus_sums_from_results_buckets_and_skips_fictitious() -> None:
    # Four hydro rows: ids 0,1 -> SUDESTE; id 2 -> SUL; id 3 -> NOFICT1 (dropped).
    #
    # ticket-011: the bus label is re-sourced from the hydro_bus_generation
    # partition's distinct (hydro_id, bus_id) pairs (merged onto hydro_meta as
    # "bus_ids" by the results-comparison orchestrator) -- read_cobre_hydro_metadata
    # itself no longer carries a "bus_id" key.
    results = [
        _hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0),
        _hydro_row(1, 1, "storage_final_hm3", 200.0, 180.0),
        _hydro_row(2, 1, "storage_final_hm3", 50.0, 48.0),
        _hydro_row(3, 1, "storage_final_hm3", 999.0, 999.0),
        # a non-matching variable / type must be ignored.
        _hydro_row(0, 1, "generation_mw", 1.0, 1.0),
        _hydro_row(5, 1, "storage_final_hm3", 7.0, 7.0, entity_type="thermal"),
    ]
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        1: {"bus_ids": {100}},
        2: {"bus_ids": {101}},
        3: {"bus_ids": {199}},
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
        199: {"name": "NOFICT1"},
    }

    out = per_bus_sums_from_results(results, "storage_final_hm3", hydro_meta, bus_meta)

    assert set(out) == {"SUDESTE", "SUL"}  # NOFICT1 excluded.
    assert out["SUDESTE"]["nw"] == {1: 300.0}
    assert out["SUDESTE"]["cb"] == {1: 270.0}
    assert out["SUDESTE"]["ids"] == {0, 1}
    assert out["SUL"]["nw"] == {1: 50.0}
    assert out["SUL"]["cb"] == {1: 48.0}
    assert out["SUL"]["ids"] == {2}


def test_per_bus_sums_from_results_skips_plants_without_bus_ids() -> None:
    results = [
        _hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0),
        _hydro_row(1, 1, "storage_final_hm3", 200.0, 180.0),
    ]
    # id 1 carries no "bus_ids" label at all (e.g. it never appears in the
    # hydro_bus_generation partition) -> skipped entirely, same as the legacy
    # bus_id-is-None skip.
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        1: {},
    }
    bus_meta: dict[int, dict[str, object]] = {100: {"name": "SUDESTE"}}

    out = per_bus_sums_from_results(results, "storage_final_hm3", hydro_meta, bus_meta)

    assert set(out) == {"SUDESTE"}
    assert out["SUDESTE"]["ids"] == {0}
    assert out["SUDESTE"]["nw"] == {1: 100.0}


def test_per_bus_sums_from_results_excludes_multi_bus_plant() -> None:
    """ticket-011 AC5: a plant recorded at two buses is neither collapsed onto

    one of them nor added to both (which would double-count its aggregate
    value); it is dropped from every bus's roll-up and a Diagnostic is raised.
    """
    results = [
        _hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0),  # single-bus, kept.
        _hydro_row(9, 1, "storage_final_hm3", 500.0, 480.0),  # two-bus, excluded.
    ]
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        # synthetic two-bus plant (epic 08 territory); named so FINDING-4's
        # fix (include the plant name, not just its numeric id) is checked.
        9: {"bus_ids": {100, 101}, "name": "AMBIGUOUS_PLANT"},
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
    }

    with dx.collect() as collected:
        out = per_bus_sums_from_results(
            results, "storage_final_hm3", hydro_meta, bus_meta
        )

    # Plant 9's value appears in NEITHER bus's bucket -- not collapsed onto
    # SUDESTE or SUL, and not double-counted into both.
    assert out["SUDESTE"]["ids"] == {0}
    assert out["SUDESTE"]["nw"] == {1: 100.0}
    assert "SUL" not in out

    ambiguous = [d for d in collected if d.code == "hydro-multi-bus-ambiguous"]
    assert len(ambiguous) == 1
    assert "9" in ambiguous[0].summary
    assert "AMBIGUOUS_PLANT" in ambiguous[0].summary


def test_per_bus_sums_from_frame_matches_legacy_closure() -> None:
    df = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 2],
            "stage_id": [1, 1, 2, 1],
            "water_withdrawal_violation_pos_m3s": [5.0, 3.0, 6.0, 9.0],
        }
    )
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        1: {"bus_ids": {100}},
        2: {"bus_ids": {199}},  # NOFICT -> dropped.
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        199: {"name": "NOFICT1"},
    }

    out = per_bus_sums_from_frame(
        df, "water_withdrawal_violation_pos_m3s", {0, 1, 2}, hydro_meta, bus_meta
    )

    assert set(out) == {"SUDESTE"}
    assert out["SUDESTE"]["sum"] == {1: 8.0, 2: 6.0}
    assert out["SUDESTE"]["ids"] == {0, 1}


def test_per_bus_sums_from_frame_excludes_multi_bus_plant() -> None:
    """ticket-011 AC5, frame-sourced variant of the two-bus exclusion."""
    df = pl.DataFrame(
        {
            "entity_id": [0, 9],
            "stage_id": [1, 1],
            "water_withdrawal_violation_pos_m3s": [5.0, 50.0],
        }
    )
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        9: {"bus_ids": {100, 101}},
    }
    bus_meta: dict[int, dict[str, object]] = {
        100: {"name": "SUDESTE"},
        101: {"name": "SUL"},
    }

    with dx.collect() as collected:
        out = per_bus_sums_from_frame(
            df, "water_withdrawal_violation_pos_m3s", {0, 9}, hydro_meta, bus_meta
        )

    assert out["SUDESTE"]["sum"] == {1: 5.0}
    assert out["SUDESTE"]["ids"] == {0}
    assert "SUL" not in out
    assert any(d.code == "hydro-multi-bus-ambiguous" for d in collected)


def test_per_bus_sums_from_frame_empty_or_missing_column_returns_empty() -> None:
    hydro_meta: dict[int, dict[str, object]] = {0: {"bus_ids": {100}}}
    bus_meta: dict[int, dict[str, object]] = {100: {"name": "SUDESTE"}}
    assert per_bus_sums_from_frame(None, "v", None, hydro_meta, bus_meta) == {}
    df = pl.DataFrame({"entity_id": [0], "stage_id": [1], "other": [1.0]})
    assert per_bus_sums_from_frame(df, "v", None, hydro_meta, bus_meta) == {}


def test_per_bus_sums_from_results_empty_hydro_meta_diagnoses_map_empty() -> None:
    """ticket-011 AC4: a non-empty hydro_meta that resolves to zero usable

    bus labels must not silently produce an empty map -- a Diagnostic names
    the failure instead.
    """
    results = [_hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0)]
    # Every plant lacks a "bus_ids" label (e.g. the merge step never ran).
    hydro_meta: dict[int, dict[str, object]] = {0: {}}
    bus_meta: dict[int, dict[str, object]] = {}

    with dx.collect() as collected:
        out = per_bus_sums_from_results(
            results, "storage_final_hm3", hydro_meta, bus_meta
        )

    assert out == {}
    assert any(d.code == "hydro-to-bus-map-empty" for d in collected)


def test_per_bus_sums_from_results_dedupes_repeated_ambiguity_diagnostic() -> None:
    """FINDING-3 regression: report_builder.py threads the SAME hydro_meta/
    bus_meta object pair into ~11 per-bus chart calls per comparison run
    (one per chart variable); the ambiguous-plant diagnostic must fire once
    per run, not once per call -- compare has no diagnostics de-dup sink, so
    the un-memoized helper would log the identical warning ~11x."""
    results = [_hydro_row(0, 1, "storage_final_hm3", 100.0, 90.0)]
    hydro_meta: dict[int, dict[str, object]] = {
        0: {"bus_ids": {100}},
        9: {"bus_ids": {100, 101}},
    }
    bus_meta: dict[int, dict[str, object]] = {100: {"name": "SUDESTE"}}

    with dx.collect() as collected:
        for _ in range(11):
            per_bus_sums_from_results(
                results, "storage_final_hm3", hydro_meta, bus_meta
            )

    ambiguous = [d for d in collected if d.code == "hydro-multi-bus-ambiguous"]
    assert len(ambiguous) == 1

    # A genuinely different hydro_meta/bus_meta pair (a later, distinct
    # comparison run) is unaffected by the memo -- the diagnostic is
    # per-pair, not globally suppressed after its first ever emission.
    other_hydro_meta: dict[int, dict[str, object]] = {9: {"bus_ids": {100, 101}}}
    with dx.collect() as collected_other:
        per_bus_sums_from_results(
            results, "storage_final_hm3", other_hydro_meta, bus_meta
        )
    assert any(d.code == "hydro-multi-bus-ambiguous" for d in collected_other)


def test_per_bus_band_from_pct_sums_across_bus_ids() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "generation_mw_p10": [80.0, 160.0, 85.0, 165.0],
            "generation_mw_p90": [100.0, 200.0, 105.0, 205.0],
        }
    )
    per_bus_ids = {"SUDESTE": {0, 1}}

    out = per_bus_band_from_pct(pct, "generation_mw", per_bus_ids)

    assert out["SUDESTE"][1] == (240.0, 300.0)
    assert out["SUDESTE"][2] == (250.0, 310.0)


def test_per_bus_band_from_pct_missing_column_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [1, 1],
            # only _p90 present -> _p10 missing.
            "generation_mw_p90": [100.0, 200.0],
        }
    )
    assert per_bus_band_from_pct(pct, "generation_mw", {"SUDESTE": {0, 1}}) == {}
    assert per_bus_band_from_pct(None, "generation_mw", {"SUDESTE": {0}}) == {}


def test_per_bus_band_from_pct_skips_empty_id_set() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            "v_p10": [10.0],
            "v_p90": [20.0],
        }
    )
    out = per_bus_band_from_pct(pct, "v", {"SUDESTE": set(), "SUL": {0}})
    assert "SUDESTE" not in out
    assert out["SUL"][1] == (10.0, 20.0)


def test_plant_percentile_arrays_rounds_and_defaults() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0, 0],
            "stage_id": [1, 2],
            "storage_final_hm3_p10": [80.123, 85.0],
            "storage_final_hm3_p90": [100.456, 105.0],
        }
    )
    # stage 3 is absent for this plant -> defaults to 0.0.
    out = plant_percentile_arrays(pct, [("storage_final_hm3", "Storage", [1, 2, 3])], 0)
    assert out["storage_final_hm3_p10"] == [80.12, 85.0, 0.0]
    assert out["storage_final_hm3_p90"] == [100.46, 105.0, 0.0]


def test_plant_percentile_arrays_missing_column_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            # neither storage_final_hm3 _p10 nor _p90 present.
            "other_p10": [1.0],
            "other_p90": [2.0],
        }
    )
    assert plant_percentile_arrays(pct, [("storage_final_hm3", "", [1, 2])], 0) == {}


def test_plant_percentile_arrays_unknown_plant_returns_empty() -> None:
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            "v_p10": [1.0],
            "v_p90": [2.0],
        }
    )
    # plant 99 has no rows.
    assert plant_percentile_arrays(pct, [("v", "", [1])], 99) == {}
    assert plant_percentile_arrays(None, [("v", "", [1])], 0) == {}


def test_plant_percentile_arrays_per_variable_stage_axes() -> None:
    """Each variable aligns to its OWN stage axis from the single filtered sub."""
    pct = pl.DataFrame(
        {
            "entity_id": [0, 0],
            "stage_id": [1, 2],
            "a_p10": [10.0, 11.0],
            "a_p90": [20.0, 21.0],
            "b_p10": [30.0, 31.0],
            "b_p90": [40.0, 41.0],
        }
    )
    # Variable "a" spans stages [1, 2]; "b" spans only [2] (its own axis).
    out = plant_percentile_arrays(pct, [("a", "A", [1, 2]), ("b", "B", [2])], 0)
    assert out["a_p10"] == [10.0, 11.0]
    assert out["a_p90"] == [20.0, 21.0]
    assert out["b_p10"] == [31.0]
    assert out["b_p90"] == [41.0]


def test_plant_percentile_arrays_filters_once_per_plant() -> None:
    """The per-plant ``entity_id`` filter runs exactly once regardless of var count."""

    class _CountingFrame:
        """Minimal pl.DataFrame proxy counting ``filter`` invocations."""

        def __init__(self, inner: pl.DataFrame) -> None:
            self._inner = inner
            self.filter_calls = 0

        def is_empty(self) -> bool:
            return self._inner.is_empty()

        def filter(self, *args: object, **kwargs: object) -> pl.DataFrame:
            self.filter_calls += 1
            return self._inner.filter(*args, **kwargs)

    pct = pl.DataFrame(
        {
            "entity_id": [0, 0],
            "stage_id": [1, 2],
            "a_p10": [10.0, 11.0],
            "a_p90": [20.0, 21.0],
            "b_p10": [30.0, 31.0],
            "b_p90": [40.0, 41.0],
        }
    )
    counting = _CountingFrame(pct)

    # Three variables, yet only ONE filter call for the single plant.
    out = plant_percentile_arrays(
        cast("pl.DataFrame", counting),
        [("a", "A", [1, 2]), ("b", "B", [1, 2]), ("missing", "M", [1, 2])],
        0,
    )
    assert counting.filter_calls == 1
    assert set(out) == {"a_p10", "a_p90", "b_p10", "b_p90"}


# ---------------------------------------------------------------------------
# ticket-011: system/network draw-only primitives
# ---------------------------------------------------------------------------


def _bus_rc(name: str, cobre_id: int, stage: int, variable: str) -> ResultComparison:
    return ResultComparison(
        entity_type="bus",
        entity_name=name,
        newave_code=cobre_id + 10,
        cobre_id=cobre_id,
        stage=stage,
        variable=variable,
        newave_value=float(stage * 10 + cobre_id),
        cobre_value=float(stage * 10 + cobre_id) - 1.0,
        abs_diff=1.0,
        rel_diff=None,
    )


def _spill_rc(stage: int, variable: str, nw: float) -> ResultComparison:
    return ResultComparison(
        entity_type="system_spillage",
        entity_name="SIN",
        newave_code=0,
        cobre_id=0,
        stage=stage,
        variable=variable,
        newave_value=nw,
        cobre_value=nw - 1.0,
        abs_diff=1.0,
        rel_diff=None,
    )


def test_cobre_sum_and_newave_sin_sums_both_sides() -> None:
    cobre_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1, 0, 1],
            "stage_id": [1, 1, 2, 2],
            "stored_energy_final_mwh": [900.0, 1900.0, 1000.0, 2000.0],
        }
    )
    nw_sin = pl.DataFrame(
        {
            "newave_code": [0, 0],
            "stage": [2, 3],
            "variable": ["EARMF", "EARMF"],
            "value": [3.0, 4.0],
        }
    )

    cobre_by_stage, nw_by_stage = cobre_sum_and_newave_sin(
        cobre_hydro, "stored_energy_final_mwh", nw_sin, "EARMF", 730.0, 1, None
    )

    assert cobre_by_stage == {1: 2800.0, 2: 3000.0}
    # stage 2 -> 2-1=1, stage 3 -> 3-1=2; value * 730.
    assert nw_by_stage == {1: 3.0 * 730.0, 2: 4.0 * 730.0}


def test_cobre_sum_and_newave_sin_applies_factor_and_offset() -> None:
    cobre_hydro = pl.DataFrame({"entity_id": [0], "stage_id": [5], "v": [10.0]})
    nw_sin = pl.DataFrame(
        {
            "stage": [10, 11, None, 12],
            "variable": [" earmf ", "EARMF", "EARMF", "OTHER"],
            "value": [1.0, None, 5.0, 9.0],
        }
    )

    cobre_by_stage, nw_by_stage = cobre_sum_and_newave_sin(
        cobre_hydro, "v", nw_sin, "EARMF", 2.0, 3, None
    )

    assert cobre_by_stage == {5: 10.0}
    # Row 1: " earmf " strips/uppercases to EARMF -> stage 10-3=7, 1.0*2=2.0.
    # Row 2: value None -> skipped. Row 3: stage None -> skipped.
    # Row 4: variable OTHER -> filtered out.
    assert nw_by_stage == {7: 2.0}


def test_cobre_sum_and_newave_sin_empty_returns_empty() -> None:
    empty = pl.DataFrame(schema={"entity_id": pl.Int64, "stage_id": pl.Int64})
    assert cobre_sum_and_newave_sin(empty, "v", None, None, 1.0, 0, None) == ({}, {})

    cobre_hydro = pl.DataFrame({"entity_id": [0], "stage_id": [1], "v": [5.0]})
    # Missing variable column -> ({}, {}).
    assert cobre_sum_and_newave_sin(
        cobre_hydro, "absent", None, None, 1.0, 0, None
    ) == ({}, {})
    # No nw_sin -> only Cobre side populated.
    cb, nw = cobre_sum_and_newave_sin(cobre_hydro, "v", None, "EARMF", 1.0, 0, None)
    assert cb == {1: 5.0}
    assert nw == {}


def test_bus_groups_and_pct_groups_by_name() -> None:
    results = [
        _bus_rc("Sudeste", 0, 1, "deficit_mw"),
        _bus_rc("sul", 1, 1, "deficit_mw"),
        _bus_rc("NORTE", 2, 1, "deficit_mw"),
        _bus_rc("Sudeste", 0, 2, "deficit_mw"),
        _bus_rc("Sudeste", 0, 1, "marginal_cost"),  # other variable, excluded
    ]
    pct = pl.DataFrame(
        {
            "entity_id": [0, 1, 2],
            "stage_id": [1, 1, 1],
            "deficit_mw_p10": [1.0, 2.0, 3.0],
            "deficit_mw_p90": [4.0, 5.0, 6.0],
        }
    )

    buses, pct_by_eid = bus_groups_and_pct(results, "deficit_mw", pct)

    assert set(buses) == {"SUDESTE", "SUL", "NORTE"}
    assert len(buses["SUDESTE"]) == 2
    assert pct_by_eid[0][1]["deficit_mw_p10"] == 1.0
    assert pct_by_eid[2][1]["deficit_mw_p90"] == 6.0


def test_bus_groups_and_pct_missing_column_empty_lookup() -> None:
    results = [_bus_rc("SUDESTE", 0, 1, "deficit_mw")]
    pct = pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [1],
            # only _p10 present, _p90 absent -> empty lookup.
            "deficit_mw_p10": [1.0],
        }
    )

    buses, pct_by_eid = bus_groups_and_pct(results, "deficit_mw", pct)

    assert set(buses) == {"SUDESTE"}
    assert pct_by_eid == {}
    # None frame also yields empty lookup.
    _, pct_none = bus_groups_and_pct(results, "deficit_mw", None)
    assert pct_none == {}


def test_spillage_lookups_builds_nw_and_cb() -> None:
    results = [
        _spill_rc(1, "VERTOT", 100.0),
        _spill_rc(2, "VERTOT", 110.0),
        _spill_rc(1, "VERTcont", 60.0),
    ]
    cobre_spill_energy = pl.DataFrame(
        {
            "stage_id": [1, 2],
            "total_mw": [96.0, 106.0],
            "reservoir_mw": [58.0, 63.0],
            "rorov_mw": [38.0, 43.0],
        }
    )

    nw_lookup, cb_lookup = spillage_lookups(results, cobre_spill_energy)

    assert nw_lookup["VERTOT"] == {1: 100.0, 2: 110.0}
    assert nw_lookup["VERTcont"] == {1: 60.0}
    assert cb_lookup["spill_energy_total_mw"] == {1: 96.0, 2: 106.0}
    assert cb_lookup["spill_energy_reservoir_mw"] == {1: 58.0, 2: 63.0}
    assert cb_lookup["spill_energy_rorov_mw"] == {1: 38.0, 2: 43.0}


def test_spillage_lookups_empty_returns_empty() -> None:
    empty = pl.DataFrame(
        schema={
            "stage_id": pl.Int64,
            "total_mw": pl.Float64,
            "reservoir_mw": pl.Float64,
            "rorov_mw": pl.Float64,
        }
    )
    nw_lookup, cb_lookup = spillage_lookups([], empty)
    assert nw_lookup == {}
    assert cb_lookup == {}


# -------------------------------------------------------------------
# ticket-008 (epic-03): compare verdict builder
# -------------------------------------------------------------------


def _summary_dataset(rows: list[dict[str, object]]) -> ComparisonDataset:
    """A dataset with an empty tidy frame and a hand-built summary frame.

    The summary frame conforms to ``SUMMARY_SCHEMA``; ``tidy`` is empty so the
    dataset validates while exercising only the summary-driven verdict path.
    """
    summary = pl.DataFrame(rows, schema=SUMMARY_SCHEMA)
    dataset = ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=summary,
    )
    dataset.validate()
    return dataset


class TestCompareVerdict:
    """Tests for ``build_compare_verdict`` reading only ``dataset.summary``."""

    def test_verdict_bounds_fixture_counts_variables_and_worst(self) -> None:
        results = _make_bounds()
        dataset = build_bounds_dataset(results)

        verdict = build_compare_verdict(dataset)

        rows = dataset.summary.to_dicts()
        assert verdict.total == 2
        assert verdict.within_tol == 0
        assert verdict.all_within_tol is False
        # Derive the expected winner FROM the summary, never recompute sMAPE.
        expected = min(rows, key=lambda r: (-float(r["max_smape"]), str(r["variable"])))
        assert verdict.worst_variable == expected["variable"]
        assert verdict.worst_smape == expected["max_smape"]

    def test_verdict_empty_dataset_returns_zeroed_verdict(self) -> None:
        dataset = build_results_dataset([], PercentileData(), 1e-2)

        verdict = build_compare_verdict(dataset)

        assert verdict == CompareVerdict(
            within_tol=0,
            total=0,
            worst_variable=None,
            worst_smape=0.0,
            all_within_tol=False,
        )

    def test_verdict_all_within_tol_true_when_every_variable_perfect(self) -> None:
        dataset = _summary_dataset(
            [
                {
                    "variable": "storage_max",
                    "count": 3,
                    "mean_abs_diff": 0.0,
                    "max_abs_diff": 0.0,
                    "mean_smape": 0.0,
                    "max_smape": 0.0,
                    "within_tol_rate": 1.0,
                    "correlation": None,
                },
                {
                    "variable": "generation_max",
                    "count": 2,
                    "mean_abs_diff": 0.0,
                    "max_abs_diff": 0.0,
                    "mean_smape": 0.01,
                    "max_smape": 0.02,
                    "within_tol_rate": 1.0,
                    "correlation": None,
                },
            ]
        )

        verdict = build_compare_verdict(dataset)

        assert verdict.within_tol == 2
        assert verdict.total == 2
        assert verdict.all_within_tol is True

    def test_verdict_tie_break_picks_lexicographically_first_variable(self) -> None:
        dataset = _summary_dataset(
            [
                {
                    "variable": "b_var",
                    "count": 1,
                    "mean_abs_diff": 1.0,
                    "max_abs_diff": 1.0,
                    "mean_smape": 0.1,
                    "max_smape": 0.5,
                    "within_tol_rate": 0.0,
                    "correlation": None,
                },
                {
                    "variable": "a_var",
                    "count": 1,
                    "mean_abs_diff": 1.0,
                    "max_abs_diff": 1.0,
                    "mean_smape": 0.1,
                    "max_smape": 0.5,
                    "within_tol_rate": 0.0,
                    "correlation": None,
                },
            ]
        )

        verdict = build_compare_verdict(dataset)

        assert verdict.worst_variable == "a_var"
        assert verdict.worst_smape == 0.5
