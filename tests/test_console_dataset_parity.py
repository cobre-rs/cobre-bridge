"""Console==dataset parity regression tests (epic-07, ticket-020).

The compare-data-layer migration's central invariant is single-source-of-truth:
every number on the console must trace back to ONE ``ComparisonDataset`` analysis
(``dataset.summary`` rows / ``dataset.metadata`` counts), never to a second,
independently computed path. The historical bug this guards against was a
two-path divergence where console and HTML aggregated separately and drifted.

This module renders the dataset-driven console printers, parses the numbers back
out of the rendered text, and asserts those parsed cells equal the cells produced
by formatting the matching ``dataset.summary`` / ``dataset.metadata`` values with
the printer's OWN helpers (``report._fmt_metric`` and the percent/correlation
format strings). The ONLY computation the test performs is parsing printed text:
every expected cell is derived FROM the dataset, never recomputed from raw
``ResultComparison`` / ``BoundComparison`` rows. If a future edit makes a printer
recompute a statistic instead of reading the dataset, the parsed cells diverge
from the dataset-derived expected cells and the test fails.

A third group of byte-identical legacy-vs-dataset guards (mirroring the ticket-008
tests in ``tests/test_analyze.py``) lives here too, so the parity fence is enforced
in one place: it fails if anyone reintroduces a second computation path that drifts
from the legacy printer.

Fixtures are hermetic synthetic in-memory data (NO real the source model case, NO
``inewave`` I/O), copied from ``tests/test_analyze.py`` and
``tests/test_golden_dataset.py``.
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import polars as pl

from cobre_bridge.comparators.analyze import (
    build_bounds_dataset,
    build_results_dataset,
)
from cobre_bridge.comparators.bounds import BoundComparison
from cobre_bridge.comparators.report import (
    _fmt_metric,
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

_NW = Path("/fake/nw")
_COBRE = Path("/fake/cobre")
_RESULTS_TOL = 1e-2
_BOUNDS_TOL = 1e-3

# Number of value columns in the per-variable results table, after the variable
# name: Count, Mean|D|, Max|D|, WithinTol, sMAPE, r.
_RESULTS_VALUE_COLS = 6
# Bounds table rows are: <name> Compared Match Mismatch Rate% (the variable name
# is space-free in the fixtures, so a whitespace split yields exactly 5 tokens).
_BOUNDS_ROW_TOKENS = 5


# ---------------------------------------------------------------------------
# Synthetic fixtures (copied verbatim from tests/test_analyze.py)
# ---------------------------------------------------------------------------


def _make_single_point_results() -> list[ResultComparison]:
    """One comparison for a variable that appears exactly once.

    A single-data-point variable group has ``len(nw_vals) <= 1``, so
    ``build_results_summary`` leaves ``correlation`` at its ``None`` default
    (Pearson is undefined for fewer than two points). This drives the dataset
    summary's ``correlation`` cell to ``None`` and exercises the printer's
    ``"N/A"`` rendering branch.
    """
    return [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            cobre_id=0,
            stage=0,
            variable="lonely_var",
            newave_value=100.0,
            cobre_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
    ]


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


def _one_hydro_pct() -> PercentileData:
    """The one-hydro-frame percentile data (mirrors ticket-019)."""
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


def _capture(func: object, *args: object) -> str:
    """Run a stdout-writing printer and return the captured text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args)  # type: ignore[operator]
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Parsers — the ONLY computation the test does is parsing printed text.
# ---------------------------------------------------------------------------


def _parse_results_table(text: str) -> dict[str, list[str]]:
    """Parse the per-variable results table into ``{variable: [cells]}``.

    Skips the title block, the ``=`` / ``-`` rules, the column header, the blank
    lines, and the ``Summary:`` footer. Each surviving data row is split on runs
    of whitespace (the fixtures' variable names are space-free); the first token
    is the variable and the remaining tokens are the value cells. A clear
    ``AssertionError`` is raised if a data row's token count is unexpected.
    """
    table: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("Cobre vs", "NEWAVE case:", "Cobre output:")):
            continue
        if line.startswith("Summary:"):
            continue
        if set(line) <= {"=", "-"}:
            continue
        tokens = line.split()
        if tokens[0] == "Variable":
            continue
        assert len(tokens) == 1 + _RESULTS_VALUE_COLS, (
            f"unexpected results data row token count {len(tokens)} "
            f"(expected {1 + _RESULTS_VALUE_COLS}) in line: {raw!r}"
        )
        table[tokens[0]] = tokens[1:]
    return table


def _parse_results_footer(text: str) -> tuple[int, dict[str, int]]:
    """Parse the ``Summary: N comparisons across K entity types (...)`` footer.

    Returns the total comparison count and the per-entity-type count map parsed
    from the parenthetical ``(c0 t0, c1 t1, ...)`` list.
    """
    footer = next(
        line for line in text.splitlines() if line.strip().startswith("Summary:")
    ).strip()
    after = footer[len("Summary:") :].strip()
    total = int(after.split(maxsplit=1)[0])

    open_paren = footer.index("(")
    close_paren = footer.rindex(")")
    inner = footer[open_paren + 1 : close_paren]

    by_entity_type: dict[str, int] = {}
    for part in inner.split(","):
        count_str, etype = part.split()
        by_entity_type[etype] = int(count_str)
    return total, by_entity_type


def _parse_bounds_section(
    lines: list[str], header_token: str
) -> tuple[dict[str, tuple[int, int]], tuple[int, int, int] | None]:
    """Parse one bounds table (by-type or by-variable) plus its ``Total`` row.

    ``header_token`` is the first column header (``Type`` or ``Variable``).
    Returns a ``{key: (match, mismatch)}`` map and the ``Total`` triple
    ``(compared, match, mismatch)`` when present in this section, else ``None``.
    """
    rows: dict[str, tuple[int, int]] = {}
    total_triple: tuple[int, int, int] | None = None
    in_section = False
    for raw in lines:
        line = raw.strip()
        if not line:
            if in_section:
                break
            continue
        if set(line) <= {"=", "-"}:
            continue
        tokens = line.split()
        if tokens[0] == header_token:
            in_section = True
            continue
        if not in_section:
            continue
        # Rate column carries a trailing '%'; name is space-free in fixtures.
        assert len(tokens) == _BOUNDS_ROW_TOKENS, (
            f"unexpected bounds data row token count {len(tokens)} "
            f"(expected {_BOUNDS_ROW_TOKENS}) in line: {raw!r}"
        )
        name = tokens[0]
        compared = int(tokens[1].replace(",", ""))
        match = int(tokens[2].replace(",", ""))
        mismatch = int(tokens[3].replace(",", ""))
        if name == "Total":
            total_triple = (compared, match, mismatch)
            continue
        rows[name.lower()] = (match, mismatch)
    return rows, total_triple


def _parse_mismatch_header(text: str) -> tuple[int, int]:
    """Parse ``Top N mismatches (of M total):`` -> ``(N, M)``."""
    header = next(
        line for line in text.splitlines() if line.strip().startswith("Top ")
    ).strip()
    tokens = header.split()
    # "Top N mismatches (of M total):"
    shown = int(tokens[1])
    total = int(tokens[4])
    return shown, total


def _count_mismatch_detail_lines(text: str) -> int:
    """Count the printed mismatch detail lines (indented, the source model=...)."""
    return sum(
        1
        for line in text.splitlines()
        if line.startswith("  ") and "NEWAVE=" in line and "Cobre=" in line
    )


# ---------------------------------------------------------------------------
# Group 1: parsed-console-equals-dataset (results)
# ---------------------------------------------------------------------------


def _expected_results_cells(row: dict[str, object]) -> list[str]:
    """Format a ``dataset.summary`` row with the printer's OWN helpers.

    The cell order matches the printed table:
    ``[Count, Mean|D|, Max|D|, WithinTol, sMAPE, r]``. Every cell is derived from
    the dataset row — nothing is recomputed from raw comparison rows.
    """
    correlation = row["correlation"]
    corr = f"{float(correlation):.4f}" if correlation is not None else "N/A"
    return [
        str(int(row["count"])),  # type: ignore[arg-type]
        _fmt_metric(float(row["mean_abs_diff"])),  # type: ignore[arg-type]
        _fmt_metric(float(row["max_abs_diff"])),  # type: ignore[arg-type]
        f"{float(row['within_tol_rate']) * 100:.1f}%",  # type: ignore[arg-type]
        f"{float(row['mean_smape']) * 100:.1f}%",  # type: ignore[arg-type]
        corr,
    ]


def test_results_table_cells_equal_dataset_summary() -> None:
    dataset = build_results_dataset(_make_results(), _one_hydro_pct(), _RESULTS_TOL)
    text = _capture(print_results_summary_from_dataset, dataset, _NW, _COBRE)

    parsed = _parse_results_table(text)
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}

    # Every dataset variable is printed and every printed variable is in the dataset.
    assert set(parsed) == set(summary_rows)

    for variable, row in summary_rows.items():
        expected_cells = _expected_results_cells(row)
        assert parsed[variable] == expected_cells, (
            f"console cells for {variable!r} diverged from dataset.summary"
        )


def test_results_footer_equals_dataset_footer_counts() -> None:
    dataset = build_results_dataset(_make_results(), _one_hydro_pct(), _RESULTS_TOL)
    text = _capture(print_results_summary_from_dataset, dataset, _NW, _COBRE)

    total, by_entity_type = _parse_results_footer(text)
    footer_counts = dataset.metadata["footer_counts"]
    assert isinstance(footer_counts, dict)

    assert total == footer_counts["total"]
    assert by_entity_type == footer_counts["by_entity_type"]


def test_results_correlation_none_renders_na_from_dataset() -> None:
    # Arrange: a single-data-point variable -> dataset.summary.correlation is None.
    dataset = build_results_dataset(
        _make_single_point_results(), PercentileData(), _RESULTS_TOL
    )
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}
    row = summary_rows["lonely_var"]
    # The N/A branch must be genuinely reached, not asserted blindly.
    assert row["correlation"] is None, (
        "fixture must yield correlation=None to exercise the N/A branch; "
        f"got {row['correlation']!r}"
    )

    # Act: render the dataset-driven printer and parse the variable's row.
    text = _capture(print_results_summary_from_dataset, dataset, _NW, _COBRE)
    parsed = _parse_results_table(text)

    # Assert: the printed ``r`` cell is "N/A", and "N/A" is exactly what the
    # dataset-derived expected cells produce from correlation=None — proving the
    # printed N/A traces to dataset.summary.correlation, not a recompute.
    expected_cells = _expected_results_cells(row)
    assert (
        expected_cells[-1] == "N/A"
    )  # derived FROM dataset.summary (correlation None)
    assert parsed["lonely_var"][-1] == "N/A"  # the ``r`` column is the last cell
    assert parsed["lonely_var"] == expected_cells


# ---------------------------------------------------------------------------
# Group 2: parsed-console-equals-dataset (bounds)
# ---------------------------------------------------------------------------


def test_bounds_tables_equal_dataset_summary_counts() -> None:
    dataset = build_bounds_dataset(_make_bounds())
    text = _capture(
        print_bounds_summary_from_dataset, dataset, _NW, _COBRE, _BOUNDS_TOL
    )
    lines = text.splitlines()

    summary_counts = dataset.metadata["summary_counts"]
    assert isinstance(summary_counts, dict)

    # By-entity-type table carries the Total row; by-variable table does not.
    by_type_parsed, total_triple = _parse_bounds_section(lines, "Type")
    by_var_parsed, _ = _parse_bounds_section(lines, "Variable")

    assert total_triple is not None, "Total row missing from bounds summary"
    parsed_compared, parsed_matches, parsed_mismatches = total_triple
    assert parsed_compared == summary_counts["total"]
    assert parsed_matches == summary_counts["matches"]
    assert parsed_mismatches == summary_counts["mismatches"]

    expected_by_type = {
        etype: (pair[0], pair[1])
        for etype, pair in summary_counts["by_entity_type"].items()
    }
    assert by_type_parsed == expected_by_type

    expected_by_var = {
        var: (pair[0], pair[1]) for var, pair in summary_counts["by_variable"].items()
    }
    assert by_var_parsed == expected_by_var


def test_bounds_mismatch_header_equals_listing_metadata() -> None:
    dataset = build_bounds_dataset(_make_bounds())
    text = _capture(print_bounds_mismatches_from_dataset, dataset)

    shown, total = _parse_mismatch_header(text)
    listing = dataset.metadata["mismatch_listing"]
    assert isinstance(listing, dict)

    assert total == listing["total"]
    assert shown == _count_mismatch_detail_lines(text)


def test_bounds_mismatch_truncation_footer_from_listing() -> None:
    # Arrange: the bounds fixture carries >=2 mismatches; cap the listing at 1
    # so the printer must truncate and emit its "... and N more." footer.
    dataset = build_bounds_dataset(_make_bounds())
    listing = dataset.metadata["mismatch_listing"]
    assert isinstance(listing, dict)
    total = listing["total"]
    assert total >= 2, "fixture must have >=2 mismatches to exercise truncation"

    # Act: render with max_rows=1 (positional, since _capture forwards *args).
    text = _capture(print_bounds_mismatches_from_dataset, dataset, 1)

    # Assert: header shows N==1 of M==total; exactly one detail line is printed;
    # and the exact truncation footer the printer emits is present. M/total are
    # derived from dataset.metadata, never recomputed from raw comparisons.
    shown, header_total = _parse_mismatch_header(text)
    assert shown == 1
    assert header_total == total
    assert _count_mismatch_detail_lines(text) == 1
    assert f"... and {total - 1} more." in text


# ---------------------------------------------------------------------------
# Group 3: legacy-vs-dataset byte-identical guard (regression fence)
# ---------------------------------------------------------------------------


def test_results_summary_byte_identical_to_legacy() -> None:
    results = _make_results()
    legacy_summary = build_results_summary(results, _RESULTS_TOL)
    dataset = build_results_dataset(results, PercentileData(), _RESULTS_TOL)

    legacy_out = _capture(print_results_summary, legacy_summary, _NW, _COBRE)
    dataset_out = _capture(print_results_summary_from_dataset, dataset, _NW, _COBRE)

    assert dataset_out == legacy_out


def test_bounds_summary_byte_identical_to_legacy() -> None:
    results = _make_bounds()
    legacy_summary = build_summary(results)
    dataset = build_bounds_dataset(results)

    legacy_out = _capture(print_summary, legacy_summary, _NW, _COBRE, _BOUNDS_TOL)
    dataset_out = _capture(
        print_bounds_summary_from_dataset, dataset, _NW, _COBRE, _BOUNDS_TOL
    )

    assert dataset_out == legacy_out


def test_bounds_mismatches_byte_identical_to_legacy() -> None:
    results = _make_bounds()
    dataset = build_bounds_dataset(results)

    legacy_out = _capture(print_mismatches, results)
    dataset_out = _capture(print_bounds_mismatches_from_dataset, dataset)

    assert dataset_out == legacy_out
