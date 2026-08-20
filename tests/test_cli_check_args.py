"""Unit tests for the check-family CLI args migration (ticket-004).

Tier 1 — pure Python, imports no cobre. Locks in the migration of
``_run_check``/``_run_decomp_check`` from ``SimpleNamespace`` to the typed
``CheckArgs``, and the ``--no-color``-aware console split between the
checklist (stdout) and diagnostics (stderr) it now threads through
:func:`cobre_bridge.ui.console.render_checklist`.
"""

from __future__ import annotations

import inspect
import json
import typing
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner, Result

from cobre_bridge.cli import _run_check, _run_decomp_check, app
from cobre_bridge.cli_args import CheckArgs
from cobre_bridge.diagnostics import Diagnostic, Severity
from cobre_bridge.preflight import CheckItem, PreflightResult, PreflightVerdict


def _invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(app, argv)


def _warnings_result() -> PreflightResult:
    """A WARNINGS verdict carrying one diagnostic, so both the checklist and
    the diagnostics panel render — exercising both consoles at once."""
    return PreflightResult(
        verdict=PreflightVerdict.WARNINGS,
        diagnostics=[
            Diagnostic(
                code="optional-file-absent",
                severity=Severity.WARNING,
                category="Preflight",
                title="Optional input absent",
                summary="Optional input 'modif' was not found.",
            )
        ],
        checks=[
            CheckItem(label="Required files present", passed=True),
            CheckItem(
                label="Optional: modif",
                passed=True,
                detail="absent (will use defaults)",
            ),
        ],
    )


class TestCheckHandlerSignatures:
    """Both check handlers are typed on ``CheckArgs``, not ``SimpleNamespace``."""

    @pytest.mark.parametrize("func", [_run_check, _run_decomp_check])
    def test_args_parameter_is_annotated_check_args(
        self, func: typing.Callable[..., None]
    ) -> None:
        hints = typing.get_type_hints(func)
        assert hints["args"] is CheckArgs

    @pytest.mark.parametrize("func", [_run_check, _run_decomp_check])
    def test_does_not_reference_simple_namespace(
        self, func: typing.Callable[..., None]
    ) -> None:
        assert "SimpleNamespace" not in inspect.getsource(func)


class TestCheckNewaveJsonShape:
    def test_json_emits_expected_command_status_and_checks_shape(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "cobre_bridge.preflight.run_preflight", return_value=_warnings_result()
        ):
            result = _invoke(["check", "newave", str(tmp_path / "case"), "--json"])

        assert result.exit_code == 1  # WARNINGS -> 1
        document = json.loads(result.stdout)
        assert document["command"] == "check newave"
        assert document["status"] == "warnings"
        assert document["summary"]["checks"] == [
            {"label": "Required files present", "passed": True, "detail": None},
            {
                "label": "Optional: modif",
                "passed": True,
                "detail": "absent (will use defaults)",
            },
        ]
        assert document["diagnostics"][0]["code"] == "optional-file-absent"
        # No Rich checklist leaked onto either stream under --json.
        assert "Ready with warnings" not in result.stdout
        assert result.stderr == ""


class TestCheckNewaveNoColor:
    def test_no_color_produces_ansi_free_output_on_both_streams(
        self, tmp_path: Path, dumb_terminal: None
    ) -> None:
        with patch(
            "cobre_bridge.preflight.run_preflight", return_value=_warnings_result()
        ):
            result = _invoke(["check", "newave", str(tmp_path / "case"), "--no-color"])

        assert (
            result.exit_code == 1
        )  # the verdict's mapped code, unaffected by --no-color
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr
        # Rendering still happened on both streams (headline on stdout,
        # diagnostics delegated to stderr), just without ANSI escapes.
        assert "Ready with warnings" in result.stdout
        assert "Optional input absent" in result.stderr


class TestCheckDecompJsonShape:
    """Mirrors TestCheckNewaveJsonShape for ``_run_decomp_check`` — the two
    handlers build the identical verdict envelope off the same PreflightResult
    shape, so the twin-track contract (``.claude/rules/bridge.md`` §1) applies."""

    def test_json_emits_expected_command_status_and_checks_shape(
        self, tmp_path: Path
    ) -> None:
        with patch(
            "cobre_bridge.decomp.preflight.run_decomp_preflight",
            return_value=_warnings_result(),
        ):
            result = _invoke(["check", "decomp", str(tmp_path / "case"), "--json"])

        assert result.exit_code == 1  # WARNINGS -> 1
        document = json.loads(result.stdout)
        assert document["command"] == "check decomp"
        assert document["status"] == "warnings"
        assert document["summary"]["checks"] == [
            {"label": "Required files present", "passed": True, "detail": None},
            {
                "label": "Optional: modif",
                "passed": True,
                "detail": "absent (will use defaults)",
            },
        ]
        assert document["diagnostics"][0]["code"] == "optional-file-absent"
        # No Rich checklist leaked onto either stream under --json.
        assert "Ready with warnings" not in result.stdout
        assert result.stderr == ""


class TestCheckDecompNoColor:
    """Mirrors TestCheckNewaveNoColor for ``_run_decomp_check``."""

    def test_no_color_produces_ansi_free_output_on_both_streams(
        self, tmp_path: Path, dumb_terminal: None
    ) -> None:
        with patch(
            "cobre_bridge.decomp.preflight.run_decomp_preflight",
            return_value=_warnings_result(),
        ):
            result = _invoke(["check", "decomp", str(tmp_path / "case"), "--no-color"])

        assert (
            result.exit_code == 1
        )  # the verdict's mapped code, unaffected by --no-color
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr
        # Rendering still happened on both streams (headline on stdout,
        # diagnostics delegated to stderr), just without ANSI escapes.
        assert "Ready with warnings" in result.stdout
        assert "Optional input absent" in result.stderr
