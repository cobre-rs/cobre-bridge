"""Tests for ``check decomp`` — deck validation without conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from idecomp.decomp.modelos.dadger import (
    ACALTEFE,
    ACCOTVAZ,
    ACNCHAVE,
    ACTIPERH,
    ACVERTJU,
    ACVOLMIN,
)

from cobre_bridge.decomp.cadastro import (
    APPLIED_AC_CLASSES,
    UNINGESTABLE_AC_CLASSES,
    CadastroResolutionReport,
    OutOfHorizon,
    _SCALAR_AC_SPECS,
)
from cobre_bridge.decomp.preflight import (
    _ALL_AC_CLASSES,
    _ac_coverage,
    run_decomp_preflight,
)
from cobre_bridge.diagnostics import Severity
from cobre_bridge.preflight import PreflightVerdict
from tests.test_decomp_cadastro import _FakeDadger

_DECKS = (
    Path("example/decomp-jul-26-rv3"),
    Path("example/decomp-set-24-rv0"),
)
_needs_decks = pytest.mark.skipif(
    not all((deck / "caso.dat").exists() for deck in _DECKS),
    reason="production decks not present",
)


def _checks(result: Any) -> dict[str, Any]:
    return {check.label: check for check in result.checks}


class TestDiscoveryFailure:
    def test_a_directory_that_is_not_a_deck_blocks(self, tmp_path: Path) -> None:
        result = run_decomp_preflight(tmp_path)
        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        assert result.checks[0].passed is False
        assert result.diagnostics, "a blocking verdict must carry a diagnostic"

    def test_nothing_is_written(self, tmp_path: Path) -> None:
        run_decomp_preflight(tmp_path)
        assert list(tmp_path.iterdir()) == []


@_needs_decks
class TestProductionDecks:
    @pytest.mark.parametrize("deck", _DECKS, ids=lambda p: p.name)
    def test_deck_passes_every_structural_check(self, deck: Path) -> None:
        result = run_decomp_preflight(deck)
        failed = [check.label for check in result.checks if not check.passed]
        assert not failed, f"{deck.name}: {failed}"

    @pytest.mark.parametrize("deck", _DECKS, ids=lambda p: p.name)
    def test_calendar_tree_and_load_checks_all_ran(self, deck: Path) -> None:
        checks = _checks(run_decomp_preflight(deck))
        for label in (
            "Operative calendar (weekly walk, month-boundary close)",
            "Load block factors reproduce the stage span",
            "Scenario probabilities sum to 1 per stage",
            "Tree shape is a trunk with one terminal fan",
        ):
            assert label in checks, f"{deck.name} missing check: {label}"

    def test_tree_shape_reports_the_node_counts(self) -> None:
        checks = _checks(run_decomp_preflight(_DECKS[0]))
        detail = checks["Tree shape is a trunk with one terminal fan"].detail
        # A trunk that branches only at the end: every stage but the last has
        # a single node.
        assert detail is not None and detail.startswith("nodes per stage: [1, 1,")

    @pytest.mark.parametrize("deck", _DECKS, ids=lambda p: p.name)
    def test_deferred_features_are_named_not_silent(self, deck: Path) -> None:
        result = run_decomp_preflight(deck)
        codes = {diag.code for diag in result.diagnostics}
        assert "decomp-anticipation-deferred" in codes
        assert "decomp-availability-deferred" in codes
        assert result.verdict is PreflightVerdict.WARNINGS


@_needs_decks
class TestCheckDecompCommand:
    @staticmethod
    def _invoke(argv: list[str]) -> Any:
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        return CliRunner().invoke(app, argv)

    def test_warnings_verdict_exits_one(self) -> None:
        result = self._invoke(["check", "decomp", str(_DECKS[0])])
        assert result.exit_code == 1
        assert "Deck discovery" in result.stdout

    def test_non_deck_exits_two(self, tmp_path: Path) -> None:
        result = self._invoke(["check", "decomp", str(tmp_path)])
        assert result.exit_code == 2

    def test_json_envelope_carries_checks_and_diagnostics(self) -> None:
        result = self._invoke(["check", "decomp", str(_DECKS[0]), "--json"])
        payload = json.loads(result.stdout)
        assert payload["command"] == "check decomp"
        assert payload["status"] == "warnings"
        assert len(payload["summary"]["checks"]) >= 6
        assert {d["code"] for d in payload["diagnostics"]} >= {
            "decomp-anticipation-deferred",
            "decomp-availability-deferred",
        }


class TestAcCoverageRegistry:
    """ticket-015: the resolver's ``APPLIED_AC_CLASSES``/
    ``UNINGESTABLE_AC_CLASSES`` registries and the reflected idecomp ``AC``
    universe (``_ALL_AC_CLASSES``) stay consistent with each other, and the
    derived deferred bucket is populated by enumerate-and-diff rather than a
    hand-maintained list — proven by three families
    (``NCHAVE``/``TIPERH``/``VERTJU``) the old blanket warning never named.
    """

    def test_coverage_registry_applied_classes_count_is_16(self) -> None:
        assert len(APPLIED_AC_CLASSES) == 16

    def test_coverage_registry_uningestable_is_altefe_only(self) -> None:
        assert UNINGESTABLE_AC_CLASSES == frozenset({ACALTEFE})

    def test_coverage_registry_applied_and_uningestable_disjoint(self) -> None:
        assert not (APPLIED_AC_CLASSES & UNINGESTABLE_AC_CLASSES)

    def test_coverage_registry_scalar_specs_subset_of_applied(self) -> None:
        scalar_classes = {spec.ac_class for spec in _SCALAR_AC_SPECS}
        assert scalar_classes <= APPLIED_AC_CLASSES

    def test_coverage_registry_applied_subset_of_all_ac_classes(self) -> None:
        assert APPLIED_AC_CLASSES <= _ALL_AC_CLASSES

    def test_coverage_registry_uningestable_subset_of_all_ac_classes(self) -> None:
        assert UNINGESTABLE_AC_CLASSES <= _ALL_AC_CLASSES

    def test_coverage_registry_all_ac_classes_has_at_least_25_members(self) -> None:
        # idecomp 1.13.0 exposes exactly 25; ``>=`` stays robust to a future add.
        assert len(_ALL_AC_CLASSES) >= 25

    def test_coverage_registry_deferred_bucket_contains_uncovered_families(
        self,
    ) -> None:
        deferred = _ALL_AC_CLASSES - APPLIED_AC_CLASSES - UNINGESTABLE_AC_CLASSES
        assert {ACNCHAVE, ACTIPERH, ACVERTJU} <= deferred


class TestAcCoverage:
    """ticket-015: ``_ac_coverage``'s three-bucket classification and its
    always-passing summary ``CheckItem`` plus per-bucket diagnostics.
    """

    def test_ac_coverage_mixed_presence_emits_all_four_diagnostics(self) -> None:
        present = pd.DataFrame({"codigo_usina": [1]})
        dadger = _FakeDadger({ACVOLMIN: present, ACCOTVAZ: present, ACALTEFE: present})
        report = CadastroResolutionReport(
            applied={"volume_minimo": 1},
            out_of_horizon=(
                OutOfHorizon(code=7, param="volume_minimo", mes=12, ano=2030),
            ),
        )

        checks, diagnostics = _ac_coverage(dadger, report)

        assert len(checks) == 1
        assert checks[0].label == "AC cadastro-override coverage"
        assert checks[0].passed is True

        by_code = {d.code: d for d in diagnostics}
        assert set(by_code) == {
            "decomp-ac-overrides-applied",
            "decomp-ac-overrides-deferred",
            "decomp-ac-altefe-uningestable",
            "decomp-ac-out-of-horizon",
        }
        assert by_code["decomp-ac-overrides-applied"].severity is Severity.INFO
        assert by_code["decomp-ac-overrides-deferred"].severity is Severity.WARNING
        assert by_code["decomp-ac-altefe-uningestable"].severity is Severity.WARNING
        assert by_code["decomp-ac-out-of-horizon"].severity is Severity.WARNING

        deferred_summary = by_code["decomp-ac-overrides-deferred"].summary
        assert "COTVAZ" in deferred_summary
        assert "ALTEFE" not in deferred_summary

        table = by_code["decomp-ac-out-of-horizon"].table
        assert table is not None
        assert table.columns == ["plant", "param", "month", "year"]
        assert table.rows == [[7, "volume_minimo", 12, 2030]]

    def test_ac_coverage_applied_only_emits_only_info_diagnostic(self) -> None:
        present = pd.DataFrame({"codigo_usina": [1]})
        dadger = _FakeDadger({ACVOLMIN: present})
        report = CadastroResolutionReport(
            applied={"volume_minimo": 1}, out_of_horizon=()
        )

        checks, diagnostics = _ac_coverage(dadger, report)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert len(diagnostics) == 1
        assert diagnostics[0].code == "decomp-ac-overrides-applied"
        assert diagnostics[0].severity is Severity.INFO

    def test_ac_coverage_empty_deck_emits_only_the_passing_summary(self) -> None:
        dadger = _FakeDadger({})
        report = CadastroResolutionReport(applied={}, out_of_horizon=())

        checks, diagnostics = _ac_coverage(dadger, report)

        assert len(checks) == 1
        assert checks[0].passed is True
        assert diagnostics == []
