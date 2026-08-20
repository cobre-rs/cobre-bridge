"""Subprocess tests for the ``convert newave`` early-exit failure paths.

Tier 1 — pure Python, imports no cobre. Locks in the migration of
``_run_newave_conversion``/``_run_decomp_conversion`` from ``SimpleNamespace``
to the typed ``ConvertArgs``, and the routing of the two NEWAVE early-exit
paths (source-missing, dest-not-empty) through ``_fail`` (CLI-02): a
``--json`` run on either condition must now emit a valid error envelope
instead of empty stdout.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import typing
from pathlib import Path

import pytest

from cobre_bridge.cli import _run_decomp_conversion, _run_newave_conversion
from cobre_bridge.cli_args import ConvertArgs

_ZEROED_CONVERT_SUMMARY = {
    "hydros": 0,
    "thermals": 0,
    "buses": 0,
    "lines": 0,
    "stages": 0,
}


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the cobre-bridge entry point as a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "cobre_bridge.cli", *args],
        capture_output=True,
        text=True,
    )


class TestConvertNewaveSourceMissingFailpath:
    def test_json_emits_one_error_envelope_with_zeroed_summary(
        self, tmp_path: Path
    ) -> None:
        dst = tmp_path / "dst"

        result = _run_cli_subprocess(
            "convert",
            "newave",
            str(tmp_path / "nonexistent"),
            str(dst),
            "--json",
        )

        assert result.returncode == 1
        document = json.loads(result.stdout)
        assert document["command"] == "convert newave"
        assert document["status"] == "error"
        assert document["summary"] == _ZEROED_CONVERT_SUMMARY
        assert len(document["diagnostics"]) == 1

    def test_without_json_renders_diagnostic_on_stderr_with_empty_stdout(
        self, tmp_path: Path
    ) -> None:
        dst = tmp_path / "dst"

        result = _run_cli_subprocess(
            "convert", "newave", str(tmp_path / "nonexistent"), str(dst)
        )

        assert result.returncode == 1
        assert result.stdout == ""
        assert "does not exist" in result.stderr


class TestConvertNewaveDestNotEmptyFailpath:
    def test_json_emits_one_error_envelope_with_use_force_substring(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst), "--json")

        assert result.returncode == 1
        document = json.loads(result.stdout)
        assert document["command"] == "convert newave"
        assert document["status"] == "error"
        assert len(document["diagnostics"]) == 1
        assert "Use --force" in document["diagnostics"][0]["summary"]
        # The refusal fires before anything touches the destination.
        assert [p.name for p in dst.iterdir()] == ["existing.txt"]

    def test_without_json_renders_diagnostic_on_stderr_with_empty_stdout(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))

        assert result.returncode == 1
        assert result.stdout == ""
        assert "Use --force" in result.stderr


class TestConvertHandlerSignatures:
    """Both handlers are typed on ``ConvertArgs``, not ``SimpleNamespace``."""

    @pytest.mark.parametrize("func", [_run_newave_conversion, _run_decomp_conversion])
    def test_args_parameter_is_annotated_convert_args(
        self, func: typing.Callable[..., None]
    ) -> None:
        hints = typing.get_type_hints(func)
        assert hints["args"] is ConvertArgs

    @pytest.mark.parametrize("func", [_run_newave_conversion, _run_decomp_conversion])
    def test_does_not_reference_simple_namespace(
        self, func: typing.Callable[..., None]
    ) -> None:
        assert "SimpleNamespace" not in inspect.getsource(func)
