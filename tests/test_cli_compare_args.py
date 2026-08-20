"""Tier-1 tests for the compare-family typed-args migration (ticket-006).

Locks in two things: `_compare_newave` / `_compare_decomp` build a typed
`CompareArgs` instead of a duck-typed `SimpleNamespace`, and
`_resolve_compare_settings` returns a resolved copy instead of mutating its
(frozen) input. Drives the precedence logic directly with a patched
`load_config`, so nothing reads real config/env state. Imports no cobre.
"""

from __future__ import annotations

import ast
import inspect
import typing
from pathlib import Path
from unittest.mock import patch

from cobre_bridge import cli
from cobre_bridge.cli_args import CompareArgs
from cobre_bridge.config_resolution import RESULTS_TOLERANCE_DEFAULT, BridgeConfig


def _make_args(
    *,
    tolerance: float | None = None,
    fmt: list[str] | None = None,
    out_dir: Path | None = None,
) -> CompareArgs:
    return CompareArgs(
        source_dir=Path("source"),
        cobre_output_dir=Path("cobre"),
        tolerance=tolerance,
        format=fmt,
        out_dir=out_dir,
        json_output=False,
        verbose=0,
        log_file=None,
        no_color=False,
        quiet=False,
    )


class TestResolveCompareSettingsPrecedence:
    """`_resolve_compare_settings` returns a resolved copy; the input is untouched."""

    def test_config_fills_when_flag_and_env_are_none(self) -> None:
        args = _make_args(tolerance=None, fmt=None, out_dir=None)
        config = BridgeConfig(
            results_tolerance=5e-4,
            formats=("json", "csv"),
            out_dir=Path("art"),
        )

        with patch.object(cli, "load_config", return_value=config):
            resolved = cli._resolve_compare_settings(args)

        assert resolved is not args
        assert resolved.tolerance == 5e-4
        assert resolved.format == ["json", "csv"]
        assert resolved.out_dir == Path("art")

        assert args.tolerance is None
        assert args.format is None
        assert args.out_dir is None

    def test_flag_wins_over_config(self) -> None:
        args = _make_args(tolerance=0.2, fmt=["json"], out_dir=Path("from_flag"))
        config = BridgeConfig(
            results_tolerance=5e-4,
            formats=("csv",),
            out_dir=Path("from_config"),
        )

        with patch.object(cli, "load_config", return_value=config):
            resolved = cli._resolve_compare_settings(args)

        assert resolved is not args
        assert resolved.tolerance == 0.2
        assert resolved.format == ["json"]
        assert resolved.out_dir == Path("from_flag")

        assert args.tolerance == 0.2
        assert args.format == ["json"]
        assert args.out_dir == Path("from_flag")

    def test_builtin_default_when_config_empty(self) -> None:
        args = _make_args(tolerance=None, fmt=None, out_dir=None)

        with patch.object(cli, "load_config", return_value=BridgeConfig()):
            resolved = cli._resolve_compare_settings(args)

        assert resolved.tolerance == RESULTS_TOLERANCE_DEFAULT
        # Format/out-dir stay None so the downstream defaults (_parse_formats /
        # derived out-dir) still apply.
        assert resolved.format is None
        assert resolved.out_dir is None


class TestResolveCompareSettingsPurity:
    """The body performs no in-place `args.<attr> = ...` mutation."""

    def test_body_contains_no_attribute_assignment_on_args(self) -> None:
        source = inspect.getsource(cli._resolve_compare_settings)
        func_def = ast.parse(source).body[0]
        assert isinstance(func_def, ast.FunctionDef)

        offending: list[str] = []
        for node in ast.walk(func_def):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AugAssign):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "args"
                ):
                    offending.append(ast.dump(target))

        assert offending == []

    def test_returns_a_compare_args_instance(self) -> None:
        args = _make_args()
        with patch.object(cli, "load_config", return_value=BridgeConfig()):
            resolved = cli._resolve_compare_settings(args)
        assert isinstance(resolved, CompareArgs)


class TestCompareHandlerSignatures:
    """`_run_newave_comparison` / `_run_decomp_comparison` take `CompareArgs`."""

    def test_newave_handler_annotated_compare_args(self) -> None:
        hints = typing.get_type_hints(cli._run_newave_comparison)
        assert hints["args"] is CompareArgs

    def test_decomp_handler_annotated_compare_args(self) -> None:
        hints = typing.get_type_hints(cli._run_decomp_comparison)
        assert hints["args"] is CompareArgs

    def test_newave_handler_does_not_reference_simple_namespace(self) -> None:
        assert "SimpleNamespace" not in inspect.getsource(cli._run_newave_comparison)

    def test_decomp_handler_does_not_reference_simple_namespace(self) -> None:
        assert "SimpleNamespace" not in inspect.getsource(cli._run_decomp_comparison)
