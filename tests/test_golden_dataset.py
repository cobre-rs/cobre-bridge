"""Golden-file regression tests pinning the canonical dataset numbers.

These tests snapshot the canonical ``ComparisonDataset`` numbers for the
``compare results`` subcommand, built from synthetic in-memory fixtures (NO
real the source model case, NO ``inewave`` I/O). They pin:

- the canonical tidy frame (``dataset.tidy``),
- the per-variable summary frame (``dataset.summary``),
- the on-disk ``summary.json`` written by :func:`export.write_artifacts`.

Comparison is EXACT: frames via ``polars.testing.assert_frame_equal(...,
check_exact=True, check_dtypes=True)`` and the JSON records via plain ``==``.
This is correct because the math is deterministic and the JSON float round-trip
was verified to preserve the float ``repr`` byte-for-byte.

Regeneration recipe (only when an intentional, reviewed output change is made)::

    COBRE_UPDATE_GOLDEN=1 .venv/bin/pytest tests/test_golden_dataset.py

When ``COBRE_UPDATE_GOLDEN=1`` the tests WRITE the golden files and pass;
otherwise they READ the goldens and assert equality. Goldens are NEVER silently
overwritten on mismatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from cobre_bridge.comparators import export
from cobre_bridge.comparators.analyze import build_results_dataset
from cobre_bridge.comparators.results import PercentileData, ResultComparison

_GOLDEN_DIR = Path(__file__).parent / "golden"


def _update_golden() -> bool:
    return os.environ.get("COBRE_UPDATE_GOLDEN") == "1"


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


def _make_percentiles() -> PercentileData:
    """One hydro percentile frame over two entities at stage 0."""
    return PercentileData(
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


def _assert_frame_golden(actual: pl.DataFrame, name: str) -> None:
    """Snapshot ``actual`` against the golden file ``name``.

    ``pl.read_json`` infers wider/narrower dtypes than ``actual``'s schema, so
    the golden is cast to that schema before comparing.
    """
    path = _GOLDEN_DIR / name
    if _update_golden():
        actual.write_json(path)
        return
    golden = pl.read_json(path).cast(actual.schema)
    assert_frame_equal(actual, golden, check_exact=True, check_dtypes=True)


def _assert_json_golden(records: list[dict[str, object]], name: str) -> None:
    """Snapshot ``records`` against the golden JSON file ``name``."""
    path = _GOLDEN_DIR / name
    if _update_golden():
        path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return
    golden = json.loads(path.read_text(encoding="utf-8"))
    assert records == golden


_TIDY_SORT = ["entity_type", "entity_id", "stage", "variable", "source"]


def test_golden_results_tidy() -> None:
    dataset = build_results_dataset(_make_results(), _make_percentiles(), 1e-2)

    actual = dataset.tidy.sort(_TIDY_SORT)

    _assert_frame_golden(actual, "dataset_results_tidy.json")


def test_golden_results_summary() -> None:
    dataset = build_results_dataset(_make_results(), _make_percentiles(), 1e-2)

    actual = dataset.summary.sort(["variable"])

    _assert_frame_golden(actual, "dataset_results_summary.json")


def test_golden_results_summary_json_on_disk(tmp_path: Path) -> None:
    dataset = build_results_dataset(_make_results(), _make_percentiles(), 1e-2)

    export.write_artifacts(
        dataset,
        command="compare newave",
        source_dir=Path("/fake/nw"),
        cobre_output_dir=Path("/fake/cobre"),
        tolerance=1e-2,
        out_dir=tmp_path,
        formats=["json"],
    )
    records = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))

    _assert_json_golden(records, "dataset_results_summary_json.json")
