"""Tests for the reference-label indirection (ticket-021, D-LABEL Option A).

``build_comparison_report`` and ``print_results_summary_from_dataset`` hard-coded
the reference series' display label as the literal string ``"NEWAVE"`` across
~70 sites in ``charts.py`` / ``report_builder.py`` / ``report.py``. This ticket
threads an additive ``reference_label: str = "NEWAVE"`` keyword through every one
of those sites (explicit parameter passing, no module-level render state, no
post-render string rewrite) so a future DECOMP report can pass
``reference_label="DECOMP"`` instead of mislabeling the DECOMP series as
"NEWAVE".

This module:

* GOLDEN-guards that the default (no ``reference_label`` argument) renders
  byte-identical HTML / terminal output to the pre-ticket behaviour, by reusing
  the exact fixture/golden pair ``tests/test_chart_helpers.py`` already uses for
  ``build_comparison_report``'s dataset-seam golden test
  (``tests/golden/build_comparison_report_full.html``) — the ground truth
  captured before this ticket's edit.
* Proves every one of the ~70 sites was threaded (no site missed) by asserting
  that ``reference_label="DECOMP"`` leaves zero "NEWAVE" substrings in the
  rendered HTML / printed terminal text, while the Cobre-side labels are
  unchanged.

No real NEWAVE/DECOMP case data is read (Tier 1 — CI, no ``example/``); the
dataset is built from the same hermetic in-memory fixtures
``test_chart_helpers.py`` uses for its own golden test.
"""

from __future__ import annotations

import contextlib
import io
from collections.abc import Callable
from pathlib import Path

from cobre_bridge.comparators.analyze import build_results_dataset
from cobre_bridge.comparators.dataset import ComparisonDataset
from cobre_bridge.comparators.report import print_results_summary_from_dataset
from cobre_bridge.comparators.report_builder import build_comparison_report
from tests.test_chart_helpers import (
    _GOLDEN_DIR,
    _report_fixture_pct,
    _report_fixture_results,
    _strip_chart_id,
)

_NW_DIR = Path("/fake/nw")
_COBRE_DIR = Path("/fake/cobre")


def _capture(func: Callable[..., None], *args: object, **kwargs: object) -> str:
    """Run a stdout-writing printer and return the captured text."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func(*args, **kwargs)
    return buffer.getvalue()


def _build_dataset() -> ComparisonDataset:
    """Build the same hermetic dataset used by test_chart_helpers's own golden test."""
    return build_results_dataset(_report_fixture_results(), _report_fixture_pct(), 0.05)


# ---------------------------------------------------------------------------
# GOLDEN: build_comparison_report(dataset) at the default label is
# byte-identical to the pre-ticket output.
# ---------------------------------------------------------------------------


def test_build_comparison_report_default_label_matches_golden() -> None:
    """No ``reference_label`` argument renders byte-identical to the pre-ticket HTML.

    Reuses the exact golden file
    ``tests/golden/build_comparison_report_full.html`` that
    ``tests.test_chart_helpers.test_build_comparison_report_dataset_golden``
    already guards — captured from the pre-ticket ``charts.py`` /
    ``report_builder.py`` — to prove the additive ``reference_label`` parameter
    changes nothing at its ``"NEWAVE"`` default.
    """
    html = build_comparison_report(_build_dataset())

    golden = (_GOLDEN_DIR / "build_comparison_report_full.html").read_text(
        encoding="utf-8"
    )
    assert _strip_chart_id(html) == _strip_chart_id(golden)


def test_build_comparison_report_explicit_newave_label_matches_golden() -> None:
    """Explicitly passing ``reference_label="NEWAVE"`` matches the default."""
    html = build_comparison_report(_build_dataset(), reference_label="NEWAVE")

    golden = (_GOLDEN_DIR / "build_comparison_report_full.html").read_text(
        encoding="utf-8"
    )
    assert _strip_chart_id(html) == _strip_chart_id(golden)


# ---------------------------------------------------------------------------
# DECOMP label: every reference-series trace name / chart title / report
# title / prose string reads "DECOMP" and zero "NEWAVE" substrings remain,
# proving no site was missed. The Cobre-side labels are unchanged.
# ---------------------------------------------------------------------------


def test_build_comparison_report_decomp_label_has_no_newave_substring() -> None:
    """``reference_label="DECOMP"`` relabels every site; none was left behind.

    A single missed site among the ~70 (a trace name, chart title, hover
    template, or prose string still hard-coding "NEWAVE") would surface here as
    a leftover "NEWAVE" substring in the rendered report.
    """
    html = build_comparison_report(_build_dataset(), reference_label="DECOMP")

    assert "NEWAVE" not in html
    assert "DECOMP" in html


def test_build_comparison_report_decomp_label_cobre_side_unchanged() -> None:
    """The Cobre-side series labels are untouched by the reference_label swap."""
    html_default = build_comparison_report(_build_dataset())
    html_decomp = build_comparison_report(_build_dataset(), reference_label="DECOMP")

    assert "Cobre" in html_default
    assert "Cobre" in html_decomp
    # "Cobre Mean" is the most common reference-series-adjacent trace name;
    # confirm it is present verbatim (untouched) in both renders.
    assert "Cobre Mean" in html_default
    assert "Cobre Mean" in html_decomp


def test_build_comparison_report_decomp_label_report_title() -> None:
    """The document title and report header read the DECOMP label."""
    html = build_comparison_report(_build_dataset(), reference_label="DECOMP")

    assert "Cobre vs DECOMP Results Comparison" in html
    assert "Cobre vs NEWAVE Results Comparison" not in html


def test_build_comparison_report_default_label_report_title() -> None:
    """The document title and report header keep the NEWAVE label by default."""
    html = build_comparison_report(_build_dataset())

    assert "Cobre vs NEWAVE Results Comparison" in html


# ---------------------------------------------------------------------------
# print_results_summary_from_dataset: default unchanged; "DECOMP" relabels.
# ---------------------------------------------------------------------------


def test_print_results_summary_default_label_unchanged() -> None:
    """No ``reference_label`` argument prints the pre-ticket "NEWAVE" header/labels."""
    dataset = _build_dataset()

    text = _capture(print_results_summary_from_dataset, dataset, _NW_DIR, _COBRE_DIR)

    assert "Cobre vs NEWAVE Results Comparison" in text
    assert f"NEWAVE case:  {_NW_DIR}" in text
    assert f"Cobre output: {_COBRE_DIR}" in text


def test_print_results_summary_explicit_newave_label_matches_default() -> None:
    """Explicitly passing ``reference_label="NEWAVE"`` matches the default text."""
    dataset = _build_dataset()

    text_default = _capture(
        print_results_summary_from_dataset, dataset, _NW_DIR, _COBRE_DIR
    )
    text_explicit = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW_DIR,
        _COBRE_DIR,
        "NEWAVE",
    )

    assert text_default == text_explicit


def test_print_results_summary_decomp_label_relabels_header() -> None:
    """``reference_label="DECOMP"`` relabels the header/labels; no "NEWAVE" remains."""
    dataset = _build_dataset()

    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        _NW_DIR,
        _COBRE_DIR,
        "DECOMP",
    )

    assert "Cobre vs DECOMP Results Comparison" in text
    assert f"DECOMP case:  {_NW_DIR}" in text
    assert f"Cobre output: {_COBRE_DIR}" in text
    assert "NEWAVE" not in text


def test_print_results_summary_decomp_label_keeps_newave_dir_param_name() -> None:
    """``newave_dir`` stays the parameter name; only its printed LABEL changes.

    Calling with the ``newave_dir=`` keyword must still work post-ticket (the
    parameter was NOT renamed to ``reference_dir`` or similar) — only the
    printed prefix in front of the path derives from ``reference_label``.
    """
    dataset = _build_dataset()

    text = _capture(
        print_results_summary_from_dataset,
        dataset,
        newave_dir=_NW_DIR,
        cobre_output_dir=_COBRE_DIR,
        reference_label="DECOMP",
    )

    assert f"DECOMP case:  {_NW_DIR}" in text
