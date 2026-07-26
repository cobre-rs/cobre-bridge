"""Tests for ``check decomp`` — deck validation without conversion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cobre_bridge.decomp.preflight import run_decomp_preflight
from cobre_bridge.preflight import PreflightVerdict

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
