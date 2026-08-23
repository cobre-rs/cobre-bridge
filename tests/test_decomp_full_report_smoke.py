"""Final dead-code sweep + Tier-3 full multi-tab report smoke (ticket-024).

Two independent checks share this module:

* **Tier 1** (:func:`test_no_retired_decomp_symbols_remain`) -- a plain text
  scan of ``src/`` and ``tests/`` asserting that none of the symbols/modules
  retired across the ``compare decomp`` render/export/verdict unification
  (ticket-022 HTML unification, ticket-023 export + ``--json`` verdict
  unification) survive anywhere in the tree. No ``example/`` read; runs on
  every CI job (3.12/3.13/3.14).
* **Tier 3** (:func:`test_full_decomp_report_contains_every_parity_tab`, plus
  the per-feature ``TestBuildDecompDataset*E2E`` classes below) -- exercise
  the *complete* multi-tab HTML report end to end against the real reduced
  deck (``example/decomp-mar-26-rv2-reduced``) and its converted Cobre output
  (``example/cobre-mar-26-rv2-reduced/output``). Both directories are
  gitignored dev fixtures, so none of this ever runs in CI -- it is dev-only
  smoke that catches gaps a synthetic-fixture golden test cannot see (e.g. a
  report code path that is only ever populated by real per-stage
  productivity data).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cobre_bridge.comparators.decomp_results import build_decomp_dataset
from cobre_bridge.comparators.report_builder import build_comparison_report
from tests.conftest import _extract_tab_content

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


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetEnergyBalanceE2E:
    """Tier 3 (dev-only smoke): the reduced deck's real Energy Balance tab
    renders end to end. Both directories are gitignored, so this never runs
    in CI."""

    def test_energy_balance_tab_is_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert "No energy balance data available." not in html
        assert "Plotly.newPlot" in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetNetworkE2E:
    """Tier 3 (dev-only smoke, ticket-008): the reduced deck's real Network
    tab renders end to end. Both directories are gitignored, so this never
    runs in CI."""

    def test_network_tab_is_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert "Line Net Flow" in html
        assert "Plotly.newPlot" in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetCostsE2E:
    """Tier 3 (dev-only smoke, ticket-010): the reduced deck's real Overview
    cost sections render end to end. Both directories are gitignored, so
    this never runs in CI."""

    def test_overview_cost_sections_are_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert "No cost data available." not in html
        assert "Plotly.newPlot" in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetPerformanceE2E:
    """Tier 3 (dev-only smoke, ticket-013): the reduced deck's real
    Performance tab renders end to end. Both directories are gitignored, so
    this never runs in CI."""

    def test_performance_tab_is_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert dataset.render.nw_tim_stages
        assert not dataset.render.nw_tim_iterations.is_empty()
        assert "Time per Iteration" in html
        assert "Forward / Backward Split" in html
        assert "Plotly.newPlot" in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetFphaE2E:
    """Tier 3 (dev-only smoke, ticket-017): the reduced deck's real FPHA
    section renders end to end, and the full report renders without
    exception across every E1-E6 tab. Both directories are gitignored, so
    this never runs in CI."""

    def test_report_renders_every_tab_without_exception(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)  # must not raise

        for tab_id in (
            "tab-overview",
            "tab-system",
            "tab-balance",
            "tab-network",
            "tab-constraints",
            "tab-hydro",
            "tab-hydro-detail",
            "tab-thermal",
            "tab-thermal-detail",
            "tab-productivity",
            "tab-performance",
        ):
            assert f'id="{tab_id}"' in html

    def test_fpha_section_renders_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        fpha_metrics = dataset.metadata.get("fpha_metrics")
        assert fpha_metrics is not None
        assert not fpha_metrics.is_empty()
        for column in ("cobre_id", "plant_name", "nmae", "bias"):
            assert column in fpha_metrics.columns

        html = build_comparison_report(dataset)
        assert "Fitted production functions (FPHA)" in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetReeE2E:
    """Tier 3 (dev-only smoke, ticket-018): the reduced deck's real REE
    section renders end to end. Both directories are gitignored, so this
    never runs in CI."""

    def test_ree_section_renders_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert not ree_rows.is_empty()
        assert set(ree_rows["variable"].unique().to_list()) == {
            "ena_mwmes",
            "earm_final_mwmes",
        }

        html = build_comparison_report(dataset)
        assert "REE Energy" in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetEvaporationE2E:
    """Tier 3 (dev-only smoke, ticket-020): the reduced deck's real
    evaporation comparison renders end to end. Both directories are
    gitignored, so this never runs in CI."""

    def test_evaporation_rows_render_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        evap_rows = dataset.tidy.filter(
            (pl.col("entity_type") == "hydro")
            & (pl.col("variable") == "evaporation_m3s")
        )
        assert not evap_rows.is_empty()
        # The real deck's Cobre run also carries a p10/p50/p90 percentile
        # band for evaporation_m3s (generic per-variable percentile
        # unpivoting, not ticket-020-specific) -- assert the two E1-shaped
        # sources are present rather than an exact source set.
        assert {"newave", "cobre"} <= set(evap_rows["source"].unique().to_list())

        html = build_comparison_report(dataset)
        assert 'id="tab-hydro-detail"' in html


@pytest.mark.skipif(
    not _REDUCED_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetConstraintsE2E:
    """Tier 3 (dev-only smoke, ticket-019): the reduced deck's real
    Constraints tab renders end to end. Both directories are gitignored, so
    this never runs in CI."""

    def test_constraints_tab_renders_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECK, _REDUCED_COBRE_OUTPUT)

        assert dataset.render.gc_constraints
        assert not dataset.render.gc_lhs_newave.is_empty()

        html = build_comparison_report(dataset)
        constraints_tab = _extract_tab_content(html, "tab-constraints")
        assert "Generic Constraints — LHS vs Bound" in constraints_tab
        assert "Plotly.newPlot" in constraints_tab
