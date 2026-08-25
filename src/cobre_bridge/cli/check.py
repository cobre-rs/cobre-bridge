"""Check-command handlers for ``check newave`` / ``check decomp``."""

from __future__ import annotations

from pathlib import Path

import typer

from cobre_bridge.cli.args import CheckArgs
from cobre_bridge.cli.failure import _emit_convert_json
from cobre_bridge.cli.verdict import build_verdict, check_summary
from cobre_bridge.core.preflight import PreflightVerdict
from cobre_bridge.ui.console import render_checklist

#: Preflight verdict → process exit code: ``OK`` is clean (0), ``WARNINGS`` is
#: advisory (1), and ``WILL_NOT_CONVERT`` is the most severe (2). Kept as data
#: so the mapping is directly unit-testable and impossible to drift from in the
#: handler.
_VERDICT_EXIT_CODE: dict[PreflightVerdict, int] = {
    PreflightVerdict.OK: 0,
    PreflightVerdict.WARNINGS: 1,
    PreflightVerdict.WILL_NOT_CONVERT: 2,
}


def _run_decomp_check(args: CheckArgs) -> None:
    """Execute the check decomp subcommand.

    Same contract as ``check newave`` — the preflight captures every failure
    as a verdict rather than raising, so this is rendering plus the exit code
    (``OK`` → 0, ``WARNINGS`` → 1, ``WILL_NOT_CONVERT`` → 2). Writes no files
    and never calls the conversion pipeline.
    """
    from cobre_bridge.decomp.preflight import run_decomp_preflight

    result = run_decomp_preflight(args.src)

    if args.json_output:
        summary = check_summary(
            [
                {"label": check.label, "passed": check.passed, "detail": check.detail}
                for check in result.checks
            ]
        )
        _emit_convert_json(
            build_verdict(
                "check decomp", result.verdict.value, summary, result.diagnostics
            )
        )
    else:
        render_checklist(
            result,
            console=args.out_console(),
            diagnostics_console=args.err_console(),
            quiet=args.quiet,
        )

    exit_code = _VERDICT_EXIT_CODE[result.verdict]
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _run_check(args: CheckArgs) -> None:
    """Execute the check newave subcommand.

    Runs :func:`run_preflight` (which already captures every discovery/parse
    failure as a ``WILL_NOT_CONVERT`` verdict, so no ``try/except`` is needed),
    renders the ✓/✗ checklist (or, with ``--json``, emits one machine-readable
    verdict to stdout instead of any Rich output), and exits per the verdict:
    ``OK`` → 0, ``WARNINGS`` → 1, ``WILL_NOT_CONVERT`` → 2. Writes no files and
    never calls the conversion pipeline.
    """
    from cobre_bridge.newave.preflight import run_preflight

    src: Path = args.src
    result = run_preflight(src)

    if args.json_output:
        # --json: one machine-readable verdict to stdout, no Rich on either stream.
        # The command-specific payload (the checklist) lives under ``summary`` per
        # the unified verdict envelope; ``status`` passes through the preflight
        # verdict verbatim and never recomputes it.
        summary = check_summary(
            [
                {"label": check.label, "passed": check.passed, "detail": check.detail}
                for check in result.checks
            ]
        )
        _emit_convert_json(
            build_verdict(
                "check newave", result.verdict.value, summary, result.diagnostics
            )
        )
    else:
        # The ✓/✗ checklist goes to stdout while diagnostics go to stderr, both
        # via the --no-color-aware consoles the typed args build.
        render_checklist(
            result,
            console=args.out_console(),
            diagnostics_console=args.err_console(),
            quiet=args.quiet,
        )

    exit_code = _VERDICT_EXIT_CODE[result.verdict]
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
