"""Output formatting for bounds comparison results.

Provides terminal summary, mismatch detail listing, and Parquet report export.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from cobre_bridge.comparators.bounds import BoundComparison


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


def print_summary(
    summary: ComparisonSummary,
    newave_dir: Path,
    cobre_output_dir: Path,
    tolerance: float,
) -> None:
    """Print the terminal summary table."""
    out = sys.stdout

    out.write("\nCobre vs NEWAVE Bound Comparison\n")
    out.write("=" * 64 + "\n")
    out.write(f"NEWAVE case:  {newave_dir}\n")
    out.write(f"Cobre output: {cobre_output_dir}\n")
    out.write(f"Tolerance:    {tolerance}\n\n")

    # --- By entity type ---
    _W = 62
    out.write(
        f"{'Type':<12} {'Compared':>9} {'Match':>9} {'Mismatch':>9} {'Rate':>9}\n"
    )
    out.write("-" * _W + "\n")

    for etype, (m, mm) in sorted(summary.by_entity_type.items()):
        total = m + mm
        rate = m / total * 100 if total > 0 else 0.0
        out.write(
            f"{etype.capitalize():<12} {total:>9,} {m:>9,} {mm:>9,} {rate:>8.2f}%\n"
        )

    total = summary.total
    rate = summary.matches / total * 100 if total > 0 else 0.0
    out.write("-" * _W + "\n")
    out.write(
        f"{'Total':<12} {total:>9,} {summary.matches:>9,}"
        f" {summary.mismatches:>9,} {rate:>8.2f}%\n"
    )

    # --- By variable ---
    out.write("\n")
    out.write(
        f"{'Variable':<18} {'Compared':>9} {'Match':>9} {'Mismatch':>9} {'Rate':>9}\n"
    )
    out.write("-" * _W + "\n")

    for var, (m, mm) in sorted(summary.by_variable.items()):
        total_v = m + mm
        rate_v = m / total_v * 100 if total_v > 0 else 0.0
        out.write(f"{var:<18} {total_v:>9,} {m:>9,} {mm:>9,} {rate_v:>8.2f}%\n")

    out.write("\n")


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
