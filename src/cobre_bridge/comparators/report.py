"""Output formatting for bounds and results comparison.

Provides terminal summary, mismatch detail listing, Parquet report export,
and results comparison summary formatting.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from cobre_bridge.comparators.bounds import BoundComparison
from cobre_bridge.ui.console import get_console, make_table

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset


@dataclass
class ComparisonSummary:
    """Aggregate comparison statistics."""

    total: int = 0
    matches: int = 0
    mismatches: int = 0
    by_entity_type: dict[str, tuple[int, int]] = field(default_factory=dict)
    by_variable: dict[str, tuple[int, int]] = field(default_factory=dict)


def build_summary(results: list[BoundComparison]) -> ComparisonSummary:
    """Compute aggregate statistics from comparison results."""
    summary = ComparisonSummary(total=len(results))

    type_matches: dict[str, int] = defaultdict(int)
    type_mismatches: dict[str, int] = defaultdict(int)
    var_matches: dict[str, int] = defaultdict(int)
    var_mismatches: dict[str, int] = defaultdict(int)

    for r in results:
        if r.match:
            summary.matches += 1
            type_matches[r.entity_type] += 1
            var_matches[r.variable] += 1
        else:
            summary.mismatches += 1
            type_mismatches[r.entity_type] += 1
            var_mismatches[r.variable] += 1

    all_types = sorted(set(type_matches) | set(type_mismatches))
    for t in all_types:
        summary.by_entity_type[t] = (type_matches.get(t, 0), type_mismatches.get(t, 0))

    all_vars = sorted(set(var_matches) | set(var_mismatches))
    for v in all_vars:
        summary.by_variable[v] = (var_matches.get(v, 0), var_mismatches.get(v, 0))

    return summary


def print_mismatches(
    results: list[BoundComparison],
    max_rows: int = 50,
) -> None:
    """Print the top mismatches sorted by descending absolute difference."""
    mismatches = [r for r in results if not r.match]
    if not mismatches:
        sys.stdout.write("No mismatches found.\n")
        return

    mismatches.sort(key=lambda r: r.diff, reverse=True)
    shown = mismatches[:max_rows]

    sys.stdout.write(f"Top {len(shown)} mismatches (of {len(mismatches)} total):\n\n")

    for r in shown:
        sys.stdout.write(
            f"  {r.entity_type.capitalize():<8} "
            f'"{r.entity_name}" '
            f"(code={r.newave_code}, id={r.cobre_id}) "
            f"stage={r.stage} "
            f"{r.variable}: "
            f"NEWAVE={r.newave_value:.4f} "
            f"Cobre={r.cobre_value:.4f} "
            f"(d={r.diff:.4f})\n"
        )

    if len(mismatches) > max_rows:
        sys.stdout.write(f"\n  ... and {len(mismatches) - max_rows} more.\n")

    sys.stdout.write("\n")


def write_report_parquet(
    results: list[BoundComparison],
    path: Path,
) -> None:
    """Write the full comparison results as a Parquet file."""
    if not results:
        return

    df = pl.DataFrame(
        {
            "entity_type": [r.entity_type for r in results],
            "entity_name": [r.entity_name for r in results],
            "newave_code": [r.newave_code for r in results],
            "cobre_id": [r.cobre_id for r in results],
            "stage": [r.stage for r in results],
            "variable": [r.variable for r in results],
            "newave_value": [r.newave_value for r in results],
            "cobre_value": [r.cobre_value for r in results],
            "diff": [r.diff for r in results],
            "match": [r.match for r in results],
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    sys.stdout.write(f"Report written to {path} ({len(results)} rows)\n")


# -------------------------------------------------------------------
# Results comparison formatting
# -------------------------------------------------------------------


def _fmt_metric(x: float) -> str:
    """Format a metric value for the summary table.

    Scientific notation for very large or very small magnitudes (so columns
    like ``lower_bound`` stay readable instead of printing a 10-digit float),
    thousands-grouped fixed-point otherwise.
    """
    ax = abs(x)
    if ax >= 1e6 or (0 < ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:,.3f}"


# -------------------------------------------------------------------
# Dataset-driven formatting: the console renders off the canonical
# ComparisonDataset. Numbers are single-sourced from dataset.summary rows and
# dataset.metadata so the console and the file artifacts derive from ONE
# analysis; the Rich tables only restyle those same numbers.
# -------------------------------------------------------------------


def print_results_summary_from_dataset(
    dataset: ComparisonDataset,
    newave_dir: Path,
    cobre_output_dir: Path,
) -> None:
    """Print the results comparison summary (Rich table) from the canonical dataset.

    The per-variable table is read from ``dataset.summary`` rows and the footer
    from ``dataset.metadata["footer_counts"]`` (``total`` / ``by_entity_type``,
    populated by ``analyze.build_results_dataset``), so every number traces back to
    a single analysis.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset for the results subcommand.
    newave_dir:
        Path to the source model case directory.
    cobre_output_dir:
        Path to the Cobre output directory.
    """
    out = sys.stdout

    out.write("\nCobre vs NEWAVE Results Comparison\n")
    out.write("=" * 88 + "\n")
    out.write(f"NEWAVE case:  {newave_dir}\n")
    out.write(f"Cobre output: {cobre_output_dir}\n")

    # Per-variable table. WithinTol = share within the (relative) tolerance; sMAPE =
    # mean symmetric error (robust to near-zero source-model references). Cells are
    # formatted here from dataset.summary so the console and artifacts share ONE
    # analysis; the Rich table only restyles those same numbers.
    summary_rows = {row["variable"]: row for row in dataset.summary.to_dicts()}
    rows: list[list[str]] = []
    for var in sorted(summary_rows):
        stats = summary_rows[var]
        correlation = stats["correlation"]
        corr = f"{float(correlation):.4f}" if correlation is not None else "N/A"
        rows.append(
            [
                var,
                str(int(stats["count"])),
                _fmt_metric(float(stats["mean_abs_diff"])),
                _fmt_metric(float(stats["max_abs_diff"])),
                f"{float(stats['within_tol_rate']) * 100:.1f}%",
                f"{float(stats['mean_smape']) * 100:.1f}%",
                corr,
            ]
        )

    get_console().print(
        make_table(
            ["Variable", "Count", "Mean|D|", "Max|D|", "WithinTol", "sMAPE", "r"],
            rows,
            justify=["left", "right", "right", "right", "right", "right", "right"],
        )
    )

    total, by_entity_type = _footer_counts(dataset)

    entity_parts = []
    for etype, count in sorted(by_entity_type.items()):
        entity_parts.append(f"{count} {etype}")
    entity_str = ", ".join(entity_parts) if entity_parts else "none"

    out.write(
        f"\nSummary: {total} comparisons across "
        f"{len(by_entity_type)} entity types ({entity_str})\n\n"
    )


def print_bounds_summary_from_dataset(
    dataset: ComparisonDataset,
    newave_dir: Path,
    cobre_output_dir: Path,
    tolerance: float,
) -> None:
    """Print the bounds comparison summary (Rich tables) from the canonical dataset.

    The by-entity-type and by-variable tables and the totals row are read from the
    exact integer counts in ``dataset.metadata["summary_counts"]`` (populated by
    ``analyze.build_bounds_dataset``), so every number traces back to a single
    analysis.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset for the bounds subcommand.
    newave_dir:
        Path to the source model case directory.
    cobre_output_dir:
        Path to the Cobre output directory.
    tolerance:
        Absolute tolerance used for the comparison (printed verbatim).
    """
    out = sys.stdout

    (
        total_all,
        matches_all,
        mismatches_all,
        by_entity_type,
        by_variable,
    ) = _bounds_summary_counts(dataset)

    out.write("\nCobre vs NEWAVE Bound Comparison\n")
    out.write("=" * 64 + "\n")
    out.write(f"NEWAVE case:  {newave_dir}\n")
    out.write(f"Cobre output: {cobre_output_dir}\n")
    out.write(f"Tolerance:    {tolerance}\n")

    console = get_console()

    # --- By entity type (with a Total row) ---
    type_rows: list[list[str]] = []
    for etype, (m, mm) in sorted(by_entity_type.items()):
        total = m + mm
        rate = m / total * 100 if total > 0 else 0.0
        type_rows.append(
            [etype.capitalize(), f"{total:,}", f"{m:,}", f"{mm:,}", f"{rate:.2f}%"]
        )
    rate_all = matches_all / total_all * 100 if total_all > 0 else 0.0
    type_rows.append(
        [
            "Total",
            f"{total_all:,}",
            f"{matches_all:,}",
            f"{mismatches_all:,}",
            f"{rate_all:.2f}%",
        ]
    )
    console.print(
        make_table(
            ["Type", "Compared", "Match", "Mismatch", "Rate"],
            type_rows,
            justify=["left", "right", "right", "right", "right"],
        )
    )

    out.write("\n")

    # --- By variable ---
    var_rows: list[list[str]] = []
    for var, (m, mm) in sorted(by_variable.items()):
        total_v = m + mm
        rate_v = m / total_v * 100 if total_v > 0 else 0.0
        var_rows.append([var, f"{total_v:,}", f"{m:,}", f"{mm:,}", f"{rate_v:.2f}%"])
    console.print(
        make_table(
            ["Variable", "Compared", "Match", "Mismatch", "Rate"],
            var_rows,
            justify=["left", "right", "right", "right", "right"],
        )
    )

    out.write("\n")


def print_bounds_mismatches_from_dataset(
    dataset: ComparisonDataset,
    max_rows: int = 50,
) -> None:
    """Print the bounds mismatch listing from the canonical dataset.

    Byte-identical to :func:`print_mismatches`: it renders the raw-diff-sorted
    rows carried in ``dataset.metadata["mismatch_listing"]`` (built by
    ``analyze.build_bounds_dataset`` mirroring the legacy sort/cap) and uses its
    ``total`` count for the header and the ``... and M more`` footer.

    The stored rows are already capped at 50 (the CLI default); when ``max_rows``
    is smaller, the rows are re-sliced and the "more" line recomputed from
    ``total`` so the output stays correct.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset for the bounds subcommand.
    max_rows:
        The maximum number of mismatch rows to print.
    """
    total, all_rows = _bounds_mismatch_listing(dataset)
    if total == 0:
        sys.stdout.write("No mismatches found.\n")
        return

    rows = all_rows[:max_rows]

    sys.stdout.write(f"Top {len(rows)} mismatches (of {total} total):\n\n")

    for r in rows:
        entity_type = str(r["entity_type"])
        sys.stdout.write(
            f"  {entity_type.capitalize():<8} "
            f'"{r["entity_name"]}" '
            f"(code={r['newave_code']}, id={r['cobre_id']}) "
            f"stage={r['stage']} "
            f"{r['variable']}: "
            f"NEWAVE={_as_float(r['newave_value']):.4f} "
            f"Cobre={_as_float(r['cobre_value']):.4f} "
            f"(d={_as_float(r['diff']):.4f})\n"
        )

    if total > max_rows:
        sys.stdout.write(f"\n  ... and {total - max_rows} more.\n")

    sys.stdout.write("\n")


# -------------------------------------------------------------------
# Metadata accessors for the dataset-driven printers
# -------------------------------------------------------------------


def _footer_counts(dataset: ComparisonDataset) -> tuple[int, dict[str, int]]:
    """Return the results footer ``(total, by_entity_type)`` from metadata.

    Args:
        dataset: The results dataset built by ``build_results_dataset``.

    Returns:
        A pair of the total comparison count and the per-entity-type count map.
        Missing/ill-typed metadata yields ``(0, {})``.
    """
    raw = dataset.metadata.get("footer_counts")
    if not isinstance(raw, dict):
        return 0, {}
    total = raw.get("total", 0)
    return (
        int(total) if isinstance(total, int) else 0,
        _as_int_counts(raw.get("by_entity_type")),
    )


def _bounds_summary_counts(
    dataset: ComparisonDataset,
) -> tuple[int, int, int, dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
    """Return the bounds summary counts from metadata.

    Args:
        dataset: The bounds dataset built by ``build_bounds_dataset``.

    Returns:
        ``(total, matches, mismatches, by_entity_type, by_variable)`` where the
        last two are ``[match, mismatch]`` pair maps. Missing/ill-typed metadata
        yields all-empty counts.
    """
    raw = dataset.metadata.get("summary_counts")
    if not isinstance(raw, dict):
        return 0, 0, 0, {}, {}
    total = raw.get("total", 0)
    matches = raw.get("matches", 0)
    mismatches = raw.get("mismatches", 0)
    return (
        int(total) if isinstance(total, int) else 0,
        int(matches) if isinstance(matches, int) else 0,
        int(mismatches) if isinstance(mismatches, int) else 0,
        _as_count_pairs(raw.get("by_entity_type")),
        _as_count_pairs(raw.get("by_variable")),
    )


def _bounds_mismatch_listing(
    dataset: ComparisonDataset,
) -> tuple[int, list[dict[str, object]]]:
    """Return the bounds mismatch ``(total, rows)`` from metadata.

    Args:
        dataset: The bounds dataset built by ``build_bounds_dataset``.

    Returns:
        The full mismatch count and the (already raw-diff-sorted, ≤50) row
        dicts. Missing/ill-typed metadata yields ``(0, [])``.
    """
    raw = dataset.metadata.get("mismatch_listing")
    if not isinstance(raw, dict):
        return 0, []
    total = raw.get("total", 0)
    return (
        int(total) if isinstance(total, int) else 0,
        _as_dict_rows(raw.get("rows")),
    )


def _as_float(value: object) -> float:
    """Coerce a metadata scalar into ``float`` (the listing stores real numbers).

    Args:
        value: A row-dict value that ``bounds_mismatch_listing`` populated with a
            ``BoundComparison`` float field.

    Returns:
        The value as ``float``.

    Raises:
        TypeError: If ``value`` is not numeric (a corrupted metadata payload).
    """
    if isinstance(value, (int, float)):
        return float(value)
    msg = f"expected a numeric mismatch value, got {type(value).__name__}"
    raise TypeError(msg)


def _as_int_counts(value: object) -> dict[str, int]:
    """Coerce a metadata mapping into a ``dict[str, int]`` (empty on mismatch)."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, count in value.items():
        if isinstance(key, str) and isinstance(count, int):
            result[key] = count
    return result


def _as_count_pairs(value: object) -> dict[str, tuple[int, int]]:
    """Coerce a metadata mapping into a ``dict[str, (match, mismatch)]`` map."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, tuple[int, int]] = {}
    for key, pair in value.items():
        if (
            isinstance(key, str)
            and isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(x, int) for x in pair)
        ):
            result[key] = (int(pair[0]), int(pair[1]))
    return result


def _as_dict_rows(value: object) -> list[dict[str, object]]:
    """Coerce a metadata value into a ``list[dict[str, object]]`` of rows."""
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]
