"""Tier-1 CLI-UX tests for ``dashboard`` (ticket-005).

Locks in the ``dashboard`` migration from ``SimpleNamespace`` to the typed
``DashboardArgs``: the no-simulation-output early exit now routes through
``_fail`` (CLI-02 — a ``--json`` error envelope instead of empty stdout),
the two human status lines are suppressed under ``--quiet`` (CLI-06) while
the dashboard file is still written, and ``--no-color`` reaches every
console ``_run_dashboard`` builds. Imports no cobre; the dashboard build
itself is stubbed so no real case data is required.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import typing
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from cobre_bridge.cli import _run_dashboard, app
from cobre_bridge.cli_args import DashboardArgs


def _invoke(argv: list[str]) -> Result:
    return CliRunner().invoke(app, argv)


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the cobre-bridge entry point as a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "cobre_bridge.cli", *args],
        capture_output=True,
        text=True,
    )


def _make_case_dir(tmp_path: Path) -> Path:
    """Create a case dir whose ``output/simulation`` path exists."""
    case_dir = tmp_path / "case"
    (case_dir / "output" / "simulation").mkdir(parents=True)
    return case_dir


def _stub_build_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``build_dashboard`` to write a tiny file at the output path.

    ``_run_dashboard`` imports ``build_dashboard`` lazily inside its own body,
    so the patch target is the defining module, not ``cobre_bridge.cli``.
    """

    def _fake_build(_case_dir: Path, output_path: Path) -> None:
        output_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr("cobre_bridge.dashboard.build_dashboard", _fake_build)


class TestDashboardHandlerSignature:
    """``_run_dashboard`` is typed on ``DashboardArgs``, not ``SimpleNamespace``."""

    def test_args_parameter_is_annotated_dashboard_args(self) -> None:
        hints = typing.get_type_hints(_run_dashboard)
        assert hints["args"] is DashboardArgs

    def test_does_not_reference_simple_namespace(self) -> None:
        assert "SimpleNamespace" not in inspect.getsource(_run_dashboard)


class TestDashboardNoSimulationJson:
    def test_json_emits_one_error_envelope_and_exits_1(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        case_dir.mkdir()

        result = _run_cli_subprocess("dashboard", str(case_dir), "--json")

        assert result.returncode == 1
        document = json.loads(result.stdout)
        assert list(document.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert document["command"] == "dashboard"
        assert document["status"] == "error"
        assert len(document["diagnostics"]) == 1
        assert "no simulation output found" in document["diagnostics"][0]["summary"]


class TestDashboardNoSimulationPlain:
    def test_without_json_stdout_stays_empty_and_exits_1(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        case_dir.mkdir()

        result = _run_cli_subprocess("dashboard", str(case_dir))

        assert result.returncode == 1
        assert result.stdout == ""
        assert "no simulation output found" in result.stderr


class TestDashboardQuiet:
    def test_quiet_suppresses_status_lines_but_still_writes_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = _make_case_dir(tmp_path)
        _stub_build_dashboard(monkeypatch)
        output_path = case_dir / "dashboard.html"

        result = _invoke(["dashboard", str(case_dir), "--quiet"])

        assert result.exit_code == 0
        assert "Building dashboard from" not in result.stdout
        assert "Dashboard written to" not in result.stdout
        assert output_path.read_text(encoding="utf-8") == "x"


class TestDashboardNoColor:
    def test_no_color_yields_ansi_free_status_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        dumb_terminal: None,
    ) -> None:
        case_dir = _make_case_dir(tmp_path)
        _stub_build_dashboard(monkeypatch)

        result = _invoke(["dashboard", str(case_dir), "--no-color"])

        assert result.exit_code == 0
        assert "Building dashboard from" in result.stdout
        assert "Dashboard written to" in result.stdout
        assert "\x1b[" not in result.stdout
        assert "\x1b[" not in result.stderr
