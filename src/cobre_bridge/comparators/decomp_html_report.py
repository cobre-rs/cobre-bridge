"""Self-contained HTML report for a source-model-vs-Cobre operation comparison.

`compare decomp` has no :class:`~cobre_bridge.comparators.dataset.ComparisonDataset`
to hand to the heavier, Plotly-driven
:func:`cobre_bridge.comparators.report_builder.build_comparison_report` that
`compare newave` uses — a
:class:`~cobre_bridge.comparators.decomp_results.DecompComparison` carries three
flat frames instead. This module is a dedicated, lightweight renderer over those
frames: the same operation-summary table, final-bounds table, and
unmapped-entities note that
:func:`cobre_bridge.ui.console.render_decomp_comparison` prints to the terminal,
as one self-contained HTML document (inline CSS only, no external
scripts/stylesheets/fonts).

Pure function of a :class:`DecompComparison` — no I/O, no console.
"""

from __future__ import annotations

import html
from collections.abc import Iterable
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from cobre_bridge.comparators.decomp_results import DecompComparison

_OPERATION_COLUMNS: tuple[str, ...] = (
    "level",
    "variable",
    "unit",
    "n",
    "source",
    "cobre",
    "Δ",
    "Δ%",
    "sMAPE%",
    "worst entity",
)
_BOUNDS_COLUMNS: tuple[str, ...] = ("side", "lower bound", "upper bound", "iterations")

_STYLE = """
body { font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
       margin: 2rem; color: #1c1c1c; background: #ffffff; }
h1 { font-size: 1.3rem; margin-bottom: 0.75rem; }
h2 { font-size: 1.05rem; margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 0.5rem; }
caption { text-align: left; font-size: 0.8rem; color: #666666;
          margin-bottom: 0.4rem; caption-side: top; }
th, td { border: 1px solid #d0d0d0; padding: 0.3rem 0.6rem; font-size: 0.9rem;
         text-align: right; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2),
th:nth-child(3), td:nth-child(3), th:last-child, td:last-child { text-align: left; }
th { background: #f2f2f2; }
.empty-note { font-style: italic; color: #444444; }
ul { margin-top: 0.25rem; }
"""


def build_decomp_comparison_report(comparison: DecompComparison) -> str:
    """Render *comparison* as a self-contained HTML document.

    Mirrors :func:`cobre_bridge.ui.console.render_decomp_comparison`: an
    operation-summary table, a final-bounds table (shown only when
    ``comparison.convergence`` is non-empty), and an unmapped-entities section.
    An empty ``comparison.rows`` short-circuits to a minimal document containing
    the ``no comparable rows`` message, mirroring the console renderer's own
    short-circuit. Never raises.
    """
    if comparison.rows.is_empty():
        body = (
            '<p class="empty-note">no comparable rows — check that both runs '
            "cover the same stages</p>"
        )
        return _document("Operation comparison", body)

    sections = [
        f"<h1>Operation comparison ({comparison.stage_count} stages)</h1>",
        _operation_table(comparison),
    ]
    if not comparison.convergence.is_empty():
        sections.append(_bounds_table(comparison.convergence))
    sections.append(_unmapped_section(comparison.unmapped))
    return _document("Operation comparison", "\n".join(sections))


def _document(title: str, body: str) -> str:
    """Wrap *body* in a minimal, self-contained HTML5 document."""
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


def _cell(value: object) -> str:
    """Render one table cell as escaped text (``None`` shows as an empty cell)."""
    if value is None:
        return ""
    if isinstance(value, float):
        return html.escape(f"{value:g}")
    return html.escape(str(value))


def _row(cells: Iterable[object]) -> str:
    return "<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in cells) + "</tr>"


def _table(columns: tuple[str, ...], rows: list[str], *, caption: str) -> str:
    header = "".join(f"<th>{html.escape(c)}</th>" for c in columns)
    return (
        "<table>"
        f"<caption>{html.escape(caption)}</caption>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )


def _operation_table(comparison: DecompComparison) -> str:
    """Build the per-variable summary table (mirrors the console's table)."""
    rows = [
        _row(
            [
                summary_row["level"],
                summary_row["variable"],
                summary_row["unit"],
                summary_row["n"],
                summary_row["source_total"],
                summary_row["cobre_total"],
                summary_row["delta_total"],
                None
                if summary_row["delta_total_pct"] is None
                else f"{summary_row['delta_total_pct']:+.1f}",
                f"{summary_row['smape_pct']:.1f}",
                summary_row["worst_entity"],
            ]
        )
        for summary_row in comparison.summary.iter_rows(named=True)
    ]
    return _table(
        _OPERATION_COLUMNS,
        rows,
        caption="stage sums of scenario means; Δ = cobre − source",
    )


def _final(convergence: pl.DataFrame, column: str) -> object:
    """The last non-null value of *column*, or ``None`` when absent/empty."""
    if column not in convergence.columns:
        return None
    values = convergence[column].drop_nulls()
    return None if values.is_empty() else values[-1]


def _iteration_count(convergence: pl.DataFrame, column: str) -> int | None:
    """How many iterations one side reported, or ``None`` when it has no data."""
    if column not in convergence.columns:
        return None
    count = int(convergence[column].drop_nulls().len())
    return count or None


def _bounds_table(convergence: pl.DataFrame) -> str:
    """Build the final-bounds table (mirrors the console's "Final bounds")."""
    rows = [
        _row(
            [
                "source",
                _final(convergence, "source_lower"),
                _final(convergence, "source_upper"),
                _iteration_count(convergence, "source_lower"),
            ]
        ),
        _row(
            [
                "cobre",
                _final(convergence, "cobre_lower"),
                _final(convergence, "cobre_upper"),
                _iteration_count(convergence, "cobre_lower"),
            ]
        ),
    ]
    table = _table(
        _BOUNDS_COLUMNS,
        rows,
        caption="bounds are reported in each product's own units",
    )
    return f"<h2>Final bounds</h2>\n{table}"


def _unmapped_section(unmapped: dict[str, list[int]]) -> str:
    """Build the unmapped-entities note, or an empty string when nothing is missing."""
    missing = {level: codes for level, codes in unmapped.items() if codes}
    if not missing:
        return ""
    items = "".join(
        f"<li>{html.escape(level)}: "
        + ", ".join(html.escape(str(code)) for code in codes)
        + "</li>"
        for level, codes in missing.items()
    )
    return (
        "<h2>Unmapped entities</h2>\n"
        "<p>entities in the run outputs with no converted counterpart:</p>\n"
        f"<ul>{items}</ul>"
    )
