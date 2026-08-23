"""Tests for cli.py: the Typer app surface."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from tests.conftest import _make_fake_newave_dir


class TestTyperApp:
    """Typer-app behaviours via the idiomatic CliRunner invocation path."""

    @staticmethod
    def _invoke(argv: list[str]):
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        return CliRunner().invoke(app, argv)

    def test_version_exit_zero(self) -> None:
        result = self._invoke(["--version"])
        assert result.exit_code == 0
        assert "cobre-bridge" in result.stdout

    def test_help_lists_subcommands(self, dumb_terminal: None) -> None:
        result = self._invoke(["--help"])
        assert result.exit_code == 0
        assert "convert" in result.stdout
        assert "compare" in result.stdout
        assert "dashboard" in result.stdout

    def test_help_exposes_shell_completion(self, dumb_terminal: None) -> None:
        result = self._invoke(["--help"])
        assert "install-completion" in result.stdout

    def test_convert_missing_subcommand_exits_two(self) -> None:
        assert self._invoke(["convert"]).exit_code == 2

    def test_compare_missing_subcommand_exits_two(self) -> None:
        assert self._invoke(["compare"]).exit_code == 2

    def test_convert_newave_happy_path(self, tmp_path: Path) -> None:
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "out"
        report = ConversionReport(hydro_count=7, stage_count=12)
        with patch("cobre_bridge.pipeline.convert_newave_case", return_value=report):
            result = self._invoke(["convert", "newave", str(src), str(dst)])
        assert result.exit_code == 0
        assert "7 hydros" in result.stdout
