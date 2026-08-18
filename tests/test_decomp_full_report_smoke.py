"""Final dead-code sweep + Tier-3 full multi-tab report smoke (ticket-024).

Two independent checks share this module:

* **Tier 1** (:func:`test_no_retired_decomp_symbols_remain`) -- a plain text
  scan of ``src/`` and ``tests/`` asserting that none of the symbols/modules
  retired across the ``compare decomp`` render/export/verdict unification
  (ticket-022 HTML unification, ticket-023 export + ``--json`` verdict
  unification) survive anywhere in the tree. No ``example/`` read; runs on
  every CI job (3.12/3.13/3.14).
* **Tier 3** (:func:`test_full_decomp_report_contains_every_parity_tab`) --
  exercises the *complete* multi-tab HTML report end to end against the real
  reduced deck (``example/decomp-mar-26-rv2-reduced``) and its converted
  Cobre output (``example/cobre-mar-26-rv2-reduced/output``), the same
  directory pair ``tests/test_decomp_results_compare.py``'s own
  ``TestBuildDecompDatasetEnergyBalanceE2E``/``...NetworkE2E`` classes use.
  Both directories are gitignored dev fixtures, so this never runs in CI --
  it is a dev-only smoke that catches gaps a synthetic-fixture golden test
  cannot see (e.g. a report code path that is only ever populated by real
  per-stage productivity data).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cobre_bridge.comparators.decomp_results import build_decomp_dataset
from cobre_bridge.comparators.report_builder import build_comparison_report

# ---------------------------------------------------------------------------
# Tier 1 -- dead-code guard (no example/ read; runs in CI)
# ---------------------------------------------------------------------------

# Symbols/modules retired across the compare-decomp unification epic:
# DecompComparison/compare_decomp_results/decomp_compare_summary (ticket-023,
# folded onto build_decomp_dataset + the shared verdict/export path),
# decomp_export/decomp_html_report (ticket-022/023, replaced by the shared
# report_builder/export modules), render_decomp_comparison (ticket-023,
# retired from ui/console.py under the ticket's gate-G7 scope expansion).
_RETIRED_SYMBOLS = (
    "DecompComparison",
    "compare_decomp_results",
    "decomp_export",
    "decomp_html_report",
    "decomp_compare_summary",
    "render_decomp_comparison",
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCAN_DIRS = (_REPO_ROOT / "src", _REPO_ROOT / "tests")


def test_no_retired_decomp_symbols_remain() -> None:
    """None of the retired compare-decomp symbols/modules survive in `src/`/`tests/`.

    This guard's own source line naming the retired symbols above is excluded
    from the scan (it is documentation, not a survivor), so the check stays
    meaningful without ever self-matching.
    """
    this_file = Path(__file__).resolve()
    offenders: list[str] = []
    for scan_dir in _SCAN_DIRS:
        for path in sorted(scan_dir.rglob("*.py")):
            if "__pycache__" in path.parts or path.resolve() == this_file:
                continue
            text = path.read_text(encoding="utf-8")
            for symbol in _RETIRED_SYMBOLS:
                if symbol in text:
                    offenders.append(f"{path}: {symbol}")
    assert not offenders, "retired symbol(s) still present:\n" + "\n".join(offenders)


# ---------------------------------------------------------------------------
# Tier 3 -- full multi-tab report smoke (dev-only; never runs in CI)
# ---------------------------------------------------------------------------

_REDUCED_DECK = Path("example/decomp-mar-26-rv2-reduced")
_REDUCED_COBRE_OUTPUT = Path("example/cobre-mar-26-rv2-reduced/output")

# The literal nav-tab labels from html_report.COMPARISON_TABS, plus "FPHA"
# which is a section title inside the Productivity tab rather than a
# separate nav tab (there is no dedicated FPHA tab -- fitted production
# functions render inline in "Productivity" when both sides fitted FPHA
# hyperplanes). "Hydro Details"/"Thermal Details" are the actual rendered
# nav labels for what the master plan calls the Hydro/Thermal Plant Details
# tabs (``tab-hydro-detail``/``tab-thermal-detail``).
_PARITY_TAB_MARKERS = (
    "Overview",
    "System",
    "Energy Balance",
    "Network",
    "Constraints",
    "Hydro Operation",
    "Hydro Details",
    "Thermal Operation",
    "Thermal Details",
    "Productivity",
    "Performance",
    "FPHA",
)


@pytest.mark.skipif(not _REDUCED_DECK.is_dir(), reason="reduced deck not present")
def test_full_decomp_report_contains_every_parity_tab() -> None:
    """The full multi-tab report renders end to end on the real reduced deck.

    Builds the canonical dataset from the reduced deck + its converted Cobre
    output (skipping cleanly if the latter hasn't been produced yet) and
    renders the complete HTML report with ``reference_label="DECOMP"``,
    asserting every parity tab marker is present and that no "NEWAVE"
    reference label leaked through.
    """
    if not _REDUCED_COBRE_OUTPUT.is_dir():
        pytest.skip("converted cobre output not present")

    dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)
    html = build_comparison_report(dataset, reference_label="DECOMP")

    missing = [marker for marker in _PARITY_TAB_MARKERS if marker not in html]
    assert not missing, f"missing parity tab marker(s): {missing}"
    assert "NEWAVE" not in html, 'reference_label="DECOMP" left a "NEWAVE" literal'
