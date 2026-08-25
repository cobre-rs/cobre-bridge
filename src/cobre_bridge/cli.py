"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import tempfile
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from cobre_bridge import __version__
from cobre_bridge.cli_args import (
    CheckArgs,
    CompareArgs,
    ConvertArgs,
    DashboardArgs,
    _parse_formats,
)
from cobre_bridge.cobre.compat import MIN_COBRE_VERSION as MIN_COBRE_VERSION
from cobre_bridge.cobre_validation import (
    _partition_validation_warnings as _partition_validation_warnings,
)
from cobre_bridge.cobre_validation import _run_cobre_validation
from cobre_bridge.config_resolution import (
    RESULTS_TOLERANCE_DEFAULT,
    load_config,
)
from cobre_bridge.conversion_manifest import _write_conversion_manifest
from cobre_bridge.core.diagnostics import _write_diagnostics_json
from cobre_bridge.core.errors import (
    BridgeError,
    CobreOutputError,
    SourceFileError,
    diagnostic_from_exception,
)

# noqa: F401 below -- re-exported so `cli._NULL_HANDLER` keeps resolving for the
# ``test_configure_logging_levels`` import + the ``cli._configure_logging`` spy sites.
from cobre_bridge.logging_config import NULL_HANDLER as _NULL_HANDLER  # noqa: F401
from cobre_bridge.logging_config import configure_logging as _configure_logging
from cobre_bridge.logging_config import restore_log_file_handler
from cobre_bridge.preflight import PreflightVerdict
from cobre_bridge.ui.console import (
    conversion_progress,
    get_console,
    make_table,
    print_status,
    render_checklist,
    render_conversion_summary,
    render_diagnostics,
    spinner,
)
from cobre_bridge.verdict import (
    _convert_status,
    _convert_verdict_summary,
    build_verdict,
    check_summary,
    compare_summary,
    dashboard_summary,
    decomp_dataset_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.console import Console

    from cobre_bridge.case import NewaveCase
    from cobre_bridge.cli_args import CommonArgs
    from cobre_bridge.comparators.alignment import EntityAlignment
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.core.diagnostics import Diagnostic
    from cobre_bridge.id_map import NewaveIdMap
    from cobre_bridge.pipeline import ConversionReport


def _load_compare_context(
    newave_dir: Path,
    cobre_output_dir: Path,
    *,
    command: str,
    args: CompareArgs,
) -> tuple[NewaveCase, NewaveIdMap, EntityAlignment, list[dict[str, object]]]:
    """Load the source model case, id-map, entity alignment, and lines.json.

    Setup for `compare newave`. Builds the parsed
    case once (so the id-map reuses its cached readers), loads lines.json, and
    builds the entity alignment.

    Exits the process with code 1 via :func:`_fail` (a rendered diagnostic, or
    under ``--json`` a verdict envelope) if the source model case directory is
    missing (``FileNotFoundError`` from ``NewaveCase.from_directory``).
    """
    from cobre_bridge.case import NewaveCase
    from cobre_bridge.cobre.readers import read_cobre_lines
    from cobre_bridge.comparators.alignment import build_entity_alignment

    try:
        case = NewaveCase.from_directory(newave_dir)
    except FileNotFoundError as exc:
        _fail(command, args, exc, 1)

    id_map = case.id_map
    lines_json = read_cobre_lines(cobre_output_dir)
    alignment = build_entity_alignment(id_map, case, lines_json)
    return case, id_map, alignment, lines_json


def _export_compare_artifacts(
    dataset: ComparisonDataset,
    *,
    command: str,
    args: CompareArgs,
    raw_formats: list[str] | None,
    source_dir: Path,
    cobre_output_dir: Path,
    tolerance: float,
    out_dir_arg: Path | None,
    input_files: list[dict[str, object]] | None = None,
    diagnostics: list[dict[str, object]] | None = None,
    quiet_status: bool = False,
) -> tuple[set[str], Path]:
    """Resolve ``--format`` and write the machine-readable comparison artifacts.

    Shared by `compare newave` and `compare decomp` (``source_dir`` names the
    source-case directory generically — the manifest field itself is
    reused across both callers). Returns the requested formats and the
    resolved out_dir so the handler can run its own HTML branch and
    exit-code logic.

    An invalid ``--format`` token exits 2 via :func:`_fail`. A write failure
    must NOT change the comparison exit code, so an ``OSError`` is warned and
    swallowed.

    *quiet_status* (set by ``--json``) gates ONLY the ``Artifacts written to …``
    stdout status line so stdout stays pure JSON; the file export still runs and
    the ``OSError`` write-failure warning still reaches stderr.
    """
    from cobre_bridge.comparators.export import write_artifacts

    try:
        formats = _parse_formats(raw_formats)
    except ValueError as exc:
        _fail(command, args, exc, 2)

    out_dir: Path = out_dir_arg or (cobre_output_dir / "comparison_artifacts")
    export_formats = formats & {"csv", "parquet", "json"}

    try:
        write_artifacts(
            dataset,
            command=command,
            source_dir=source_dir,
            cobre_output_dir=cobre_output_dir,
            tolerance=tolerance,
            out_dir=out_dir,
            formats=sorted(export_formats),
            input_files=input_files,
            diagnostics=diagnostics,
        )
        if not quiet_status:
            print_status(f"Artifacts written to {out_dir}")
    except OSError as exc:
        print_status(
            f"Warning: failed to write artifacts: {exc}",
            console=args.err_console(),
            style="#F5A623",
        )

    return formats, out_dir


def _write_html_compare_report(
    dataset: ComparisonDataset,
    out_dir: Path,
    args: CompareArgs,
    *,
    reference_label: str,
) -> None:
    """Build and write the opt-in HTML comparison report (``--format html``/``all``).

    Shared by `compare newave` and `compare decomp`. The file is still written
    under ``--json`` (it is a ``--format`` artifact); only its stdout advisory
    is routed to stderr so stdout stays pure JSON.
    """
    from cobre_bridge.comparators.report_builder import build_comparison_report

    html = build_comparison_report(dataset, reference_label=reference_label)
    report_path = out_dir / "report.html"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    print_status(
        f"HTML report written to {report_path}",
        console=args.err_console() if args.json_output else None,
    )


def _resolve_compare_settings(args: CompareArgs) -> CompareArgs:
    """Return *args* with tolerance/format/out_dir filled in from config.

    Implements the bottom two rungs of the precedence chain
    **CLI flag > env var > config file > built-in default**: Typer has already
    resolved flag-or-env into ``args`` (a non-``None`` value means the flag or
    env var supplied it), so this fills in the config-file value, then the
    built-in default, for whatever is still ``None``. ``args`` is frozen, so
    the resolved settings come back as a new instance; the input is untouched.

    Loads the config once via :func:`load_config` (discovery from cwd) and emits
    one stderr WARNING note per load warning (e.g. a malformed config file). It
    never raises and never changes the exit code; config provenance stays off
    stdout so a ``--json``-style payload remains deterministic.
    """
    cfg = load_config()

    # Tolerance: flag/env, else the config value, else the built-in default.
    if args.tolerance is not None:
        resolved_tolerance = args.tolerance
    elif cfg.results_tolerance is not None:
        resolved_tolerance = cfg.results_tolerance
    else:
        resolved_tolerance = RESULTS_TOLERANCE_DEFAULT

    # Format: only consult config when neither flag nor env supplied it. Leave
    # ``None`` (rather than pre-expanding) so ``_parse_formats`` stays the single
    # validator/expander downstream.
    resolved_format = args.format
    if resolved_format is None and cfg.formats is not None:
        resolved_format = list(cfg.formats)

    # Out-dir: only consult config when neither flag nor env supplied it. The
    # config value may itself be ``None``, which keeps the derived
    # ``<cobre_output_dir>/comparison_artifacts`` default downstream.
    resolved_out_dir = args.out_dir if args.out_dir is not None else cfg.out_dir

    # Surface any config-load warnings on stderr only (never stdout), matching
    # the side-file advisory pattern used by ``_export_compare_artifacts``.
    for warning in cfg.warnings:
        print_status(
            warning,
            console=args.err_console(),
            style="#F5A623",
        )

    return dataclasses.replace(
        args,
        tolerance=resolved_tolerance,
        format=resolved_format,
        out_dir=resolved_out_dir,
    )


def _run_newave_comparison(args: CompareArgs) -> None:
    """Execute the compare newave subcommand.

    Intentionally always exits 0: ``compare newave`` is informational (a
    descriptive NEWAVE-vs-Cobre divergence report), so it never signals a
    failure on divergence.
    """
    args = _resolve_compare_settings(args)

    from cobre_bridge.cobre.readers import CobreReadError
    from cobre_bridge.comparators.report import print_results_summary_from_dataset
    from cobre_bridge.comparators.results import compare_results
    from cobre_bridge.comparators.verdict import build_compare_verdict, compare_status
    from cobre_bridge.core import diagnostics as dx
    from cobre_bridge.core.errors import CobrePartitionMissingError
    from cobre_bridge.core.provenance import hash_input_files

    newave_dir: Path = args.source_dir
    cobre_output_dir: Path = args.cobre_output_dir
    tolerance: float = args.tolerance

    case, id_map, alignment, _lines_json = _load_compare_context(
        newave_dir, cobre_output_dir, command="compare newave", args=args
    )

    # CobrePartitionMissingError (a BridgeError; output predates a partition
    # this compare needs) and CobreReadError (a RuntimeError; a malformed
    # output file) are disjoint hierarchies — both must stay in the ``except``,
    # or the dropped one crashes with a bare traceback instead of exit 2.
    with dx.collect() as compare_diagnostics:
        try:
            with spinner(
                "Comparing results…",
                verbose=args.verbose > 0,
                quiet=args.quiet,
                no_color=args.no_color,
            ):
                dataset = compare_results(
                    case=case,
                    id_map=id_map,
                    alignment=alignment,
                    cobre_output_dir=cobre_output_dir,
                    tolerance=tolerance,
                )
        except (CobreReadError, CobrePartitionMissingError) as exc:
            _fail("compare newave", args, exc, 2)

    # Print text summary (sourced from the dataset). Under --json the Rich tables
    # are suppressed in favour of a single machine-readable verdict on stdout;
    # under --quiet the summary is suppressed too, but diagnostics still render.
    if not args.json_output:
        if not args.quiet:
            print_results_summary_from_dataset(
                dataset, newave_dir, cobre_output_dir, console=args.out_console()
            )
        render_diagnostics(
            compare_diagnostics, console=args.err_console(), quiet=args.quiet
        )

    formats, out_dir = _export_compare_artifacts(
        dataset,
        command="compare newave",
        args=args,
        raw_formats=args.format,
        source_dir=newave_dir,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
        out_dir_arg=args.out_dir,
        input_files=hash_input_files(case.files),
        diagnostics=[d.to_dict() for d in compare_diagnostics],
        quiet_status=args.json_output,
    )

    if "html" in formats:
        _write_html_compare_report(dataset, out_dir, args, reference_label="NEWAVE")

    if args.json_output:
        # ``status`` is DECOUPLED from the exit code (this command always
        # exits 0) and uses the shared ``compare_status`` vocabulary, so an
        # empty dataset reports "no-comparable-rows", not "mismatch".
        verdict = build_compare_verdict(dataset)
        status = compare_status(dataset)
        _emit_convert_json(
            build_verdict(
                "compare newave",
                status,
                compare_summary(verdict),
                compare_diagnostics,
            )
        )


def _run_dashboard(args: DashboardArgs) -> None:
    """Execute the dashboard subcommand."""
    from cobre_bridge.dashboard import build_dashboard

    case_dir: Path = args.case_dir.resolve()
    if not (case_dir / "output" / "simulation").exists():
        _fail(
            "dashboard",
            args,
            CobreOutputError(f"no simulation output found in {case_dir}"),
            1,
        )

    from cobre_bridge.core import diagnostics as dx

    output_path: Path = args.output or (case_dir / "dashboard.html")
    if not args.json_output and not args.quiet:
        print_status(
            f"Building dashboard from {case_dir} ...", console=args.out_console()
        )
    with dx.collect() as dash_diags:
        with spinner(
            "Building dashboard…",
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ):
            build_dashboard(case_dir, output_path)
    size_kb = output_path.stat().st_size / 1024
    if not args.json_output:
        if not args.quiet:
            print_status(
                f"Dashboard written to {output_path} ({size_kb:.0f} KB)",
                console=args.out_console(),
            )
        render_diagnostics(
            dash_diags, console=get_console(stderr=True), quiet=args.quiet
        )

    if args.json_output:
        # --json: one machine-readable verdict to stdout. Emitted BEFORE the
        # --open block so the stdout JSON flushes regardless of browser outcome;
        # a built dashboard is always a success (the only failure path exits 1
        # before the build). ``size_kb`` keeps the precise float — only the
        # human line above rounds it with ``:.0f``.
        _emit_convert_json(
            build_verdict(
                "dashboard",
                "ok",
                dashboard_summary(str(output_path), size_kb),
                dash_diags,
            )
        )

    # The --open advisory stays off the stdout --json payload; it already goes to
    # stderr (err_console below), so the two compose cleanly.
    if args.open_browser:
        err_console = args.err_console()
        try:
            opened = webbrowser.open(output_path.resolve().as_uri())
        except (webbrowser.Error, OSError):
            opened = False
        if not opened:
            print_status(
                f"Note: could not open a browser for {output_path}; open it manually.",
                console=err_console,
                style="#F5A623",
            )


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
    from cobre_bridge.preflight import run_preflight

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


def _handle_conversion_pipeline_failure(
    exc: Exception, args: ConvertArgs, *, command: str, err_console: Console
) -> NoReturn:
    """Render/emit a pipeline failure verdict and exit 1 (shared by both tracks).

    ``report`` is None on this path, so counts are zeroed and the dry-run
    would-write listing is empty; ``status`` is "error" because
    ``diagnostic_from_exception`` yields an ERROR-severity diagnostic.
    """
    diag = diagnostic_from_exception(exc, context="Conversion")
    if args.json_output:
        diagnostics = [diag]
        summary = _convert_verdict_summary(None)
        if args.dry_run:
            summary["would_write"] = []
            status = _convert_status(diagnostics, success="dry-run")
        else:
            status = _convert_status(diagnostics, success="ok")
        _emit_convert_json(build_verdict(command, status, summary, diagnostics))
    else:
        render_diagnostics([diag], console=err_console, quiet=args.quiet)
    raise typer.Exit(code=1)


def _run_newave_conversion(args: ConvertArgs) -> None:
    """Execute the convert newave subcommand."""
    from cobre_bridge.newave_files import NewaveFiles
    from cobre_bridge.pipeline import (
        CONVERSION_PHASE_LABELS,
        NEWAVE_CLEARED_ARTIFACTS,
        clear_dst_contents,
        convert_newave_case,
    )

    src: Path = args.src
    dst: Path = args.dst

    out_console = args.out_console()
    err_console = args.err_console()

    if not src.exists() or not src.is_dir():
        _fail(
            "convert newave",
            args,
            SourceFileError(f"source directory '{src}' does not exist"),
            1,
            summary=_convert_verdict_summary(None),
        )

    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            _fail(
                "convert newave",
                args,
                BridgeError(
                    f"destination directory '{dst}' is not empty."
                    " Use --force to overwrite."
                ),
                1,
                summary=_convert_verdict_summary(None),
            )
        # --force: remove previous pipeline outputs before converting. A dry run
        # never mutates the destination, so the clear is skipped even with --force.
        if not args.dry_run:
            clear_dst_contents(dst, NEWAVE_CLEARED_ARTIFACTS)

    # A dry run creates no destination directory; the pipeline writes nothing.
    if not args.dry_run:
        dst.mkdir(parents=True, exist_ok=True)

    try:
        with conversion_progress(
            len(CONVERSION_PHASE_LABELS),
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ) as step:
            report: ConversionReport = convert_newave_case(
                src, dst, on_phase=step, dry_run=args.dry_run
            )
    except Exception as exc:  # noqa: BLE001
        _handle_conversion_pipeline_failure(
            exc, args, command="convert newave", err_console=err_console
        )

    if args.dry_run:
        # Dry run: report the would-write listing only; touch nothing on disk
        # (no diagnostics-json sidecar, no manifest, no validation).
        if args.json_output:
            # The would-write listing moves UNDER ``summary`` (dst-relative,
            # forward-slash, sorted) so the only top-level keys are the five
            # envelope keys.
            summary = _convert_verdict_summary(report)
            summary["would_write"] = sorted(
                Path(p).relative_to(dst).as_posix() for p in report.would_write_paths
            )
            status = _convert_status(report.diagnostics, success="dry-run")
            _emit_convert_json(
                build_verdict("convert newave", status, summary, report.diagnostics)
            )
        else:
            if not args.quiet:
                _render_dry_run_summary(report, console=out_console)
            render_diagnostics(
                report.diagnostics, console=err_console, quiet=args.quiet
            )
        if args.validate:
            print_status(
                "Note: --validate is ignored under --dry-run"
                " (nothing was written to validate).",
                console=err_console,
                style="#F5A623",
            )
        return

    # Build the convert ``summary`` + ``status`` up front; ``--validate`` may
    # later append a ``summary["validation"]`` sub-object (under --json), and the
    # verdict is emitted to stdout only after validation has run so that block is
    # populated. ``status`` is diagnostics-only and is NOT touched by validation.
    summary = _convert_verdict_summary(report)
    status = _convert_status(report.diagnostics, success="ok")

    if not args.json_output:
        if not args.quiet:
            render_conversion_summary(report, console=out_console)
        render_diagnostics(report.diagnostics, console=err_console, quiet=args.quiet)

    if args.diagnostics_json is not None:
        # The --diagnostics-json sidecar coexists with --json (both can be set).
        _write_diagnostics_json(report, args.diagnostics_json, console=err_console)

    # Provenance manifest, always written on a successful conversion. Notes go
    # to err_console (stderr) so the --json stdout verdict stays byte-deterministic.
    _write_conversion_manifest(
        report,
        src,
        dst,
        command="convert newave",
        discover=NewaveFiles.from_directory,
        console=err_console,
    )

    # ``--validate`` failure flips the exit code, never the status (verdict
    # exit-code contract). The verdict is emitted only AFTER this block so the
    # ``validation`` sub-object it folds under ``summary`` is populated first.
    validation_failed = False
    if args.validate:
        validation_failed = _run_cobre_validation(
            dst,
            command="convert newave",
            summary=summary,
            json_output=args.json_output,
            err_console=err_console,
            whitelist_substrings=(),
        )

    # Emit the --json verdict now (after validation has populated ``summary``).
    if args.json_output:
        _emit_convert_json(
            build_verdict("convert newave", status, summary, report.diagnostics)
        )

    _gate_convert_exit(status, validation_failed=validation_failed)


def _gate_convert_exit(status: str, *, validation_failed: bool) -> None:
    """Convert exit-code gate: validation failure (2) over error status (1)."""
    if validation_failed:
        raise typer.Exit(code=2)
    if status == "error":
        raise typer.Exit(code=1)


def _render_dry_run_summary(
    report: ConversionReport, *, console: Console | None = None
) -> None:
    """Render the dry-run would-write listing to stdout as a primary result.

    Prints a clear "Dry run — no files written" banner, the entity summary, and a
    table of every path the conversion would have written. Diagnostics are
    rendered separately on stderr.
    """
    target = console or get_console()
    print_status(
        "Dry run — no files written.",
        console=target,
        style="bold #F5A623",
    )
    render_conversion_summary(report, console=target)

    # Human output lists the ABSOLUTE paths (easy to copy-paste / inspect on
    # disk); the ``--dry-run --json`` document instead emits dst-relative sorted
    # paths for byte-stable, location-independent automation. The divergence is
    # intentional — keep the two representations separate.
    rows: list[list[object]] = [[path] for path in report.would_write_paths]
    table = make_table(
        ["Would write"],
        rows,
        title=f"{len(rows)} output files",
    )
    target.print(table)


def _emit_convert_json(document: dict[str, object]) -> None:
    """Write the ``--json`` verdict *document* to stdout as one JSON object.

    Writes directly to ``sys.stdout`` (NOT through the Rich console, which may
    inject styling/wrapping), with a trailing newline. A fixed insertion order is
    preserved (``sort_keys=False``) so the output is byte-stable.
    """
    json.dump(document, sys.stdout, indent=2, ensure_ascii=False, sort_keys=False)
    sys.stdout.write("\n")


def _fail(
    command: str,
    args: CommonArgs,
    exc: Exception,
    code: int,
    *,
    summary: Mapping[str, object] | None = None,
) -> NoReturn:
    """Render + (under --json) emit one failure verdict, then Exit(code)."""
    diag = diagnostic_from_exception(exc, context=command)
    if args.json_output:
        _emit_convert_json(build_verdict(command, "error", summary or {}, [diag]))
    else:
        render_diagnostics([diag], console=args.err_console(), quiet=args.quiet)
    raise typer.Exit(code=code)


# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

# Shared flags reused across the leaf commands.
_VerboseOpt = Annotated[
    int,
    typer.Option(
        "--verbose",
        "-v",
        count=True,
        help="Increase console log verbosity (-v INFO, -vv DEBUG).",
    ),
]
_LogFileOpt = Annotated[
    Path | None,
    typer.Option(
        "--log-file",
        metavar="PATH",
        help="Write the full DEBUG log to PATH (the console verbosity is unaffected).",
    ),
]
_NoColorOpt = Annotated[
    bool,
    typer.Option(
        "--no-color",
        help="Disable coloured output (also honoured via the NO_COLOR env var).",
    ),
]
_QuietOpt = Annotated[
    bool,
    typer.Option(
        "--quiet",
        help="Suppress the summary and info notes; warnings/errors still show.",
    ),
]
_FormatOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--format",
        metavar="FORMAT",
        envvar="COBRE_BRIDGE_FORMAT",
        help=(
            "Output format(s): console,html,csv,parquet,json,all. "
            "Comma-separated and/or repeatable. Overridable via "
            "COBRE_BRIDGE_FORMAT or cobre-bridge.toml. "
            "(default: console,parquet,json)"
        ),
    ),
]
_OutDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--out-dir",
        envvar="COBRE_BRIDGE_OUT_DIR",
        help=(
            "Directory for file artifacts. Overridable via COBRE_BRIDGE_OUT_DIR "
            "or cobre-bridge.toml. "
            "(default: <cobre_output_dir>/comparison_artifacts)."
        ),
    ),
]
_JsonOpt = Annotated[
    bool,
    typer.Option(
        "--json",
        help=(
            "Emit a single machine-readable JSON verdict to stdout and "
            "suppress the human-readable (Rich) output."
        ),
    ),
]
_ToleranceOpt = Annotated[
    float | None,
    typer.Option(
        envvar="COBRE_BRIDGE_RESULTS_TOLERANCE",
        help=(
            "Relative tolerance for results comparison (default 1e-2; "
            "overridable via COBRE_BRIDGE_RESULTS_TOLERANCE or cobre-bridge.toml)."
        ),
    ),
]

app = typer.Typer(
    name="cobre-bridge",
    help="Convert power system data to Cobre input format.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=True,
)
convert_app = typer.Typer(help="Convert data from a source format to Cobre JSON.")
compare_app = typer.Typer(help="Compare source model inputs/results against Cobre.")
check_app = typer.Typer(help="Validate source-model inputs without converting.")
app.add_typer(convert_app, name="convert")
app.add_typer(compare_app, name="compare")
app.add_typer(check_app, name="check")


def _version_callback(value: bool) -> None:
    if value:
        print_status(f"cobre-bridge {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Convert power system data to Cobre input format."""


@convert_app.command("newave")
def _convert_newave(
    src: Annotated[Path, typer.Argument(help="Path to the NEWAVE case directory.")],
    dst: Annotated[
        Path, typer.Argument(help="Path to the output Cobre case directory.")
    ],
    validate: Annotated[
        bool,
        typer.Option(
            "--validate",
            help="After conversion, validate the output with the cobre package.",
        ),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite destination directory if it already contains files.",
        ),
    ] = False,
    diagnostics_json: Annotated[
        Path | None,
        typer.Option(
            "--diagnostics-json",
            metavar="PATH",
            help="Also write the conversion diagnostics (counts + findings) as JSON.",
        ),
    ] = None,
    json_output: _JsonOpt = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Run the full conversion in memory and report what would be "
                "written, without creating or modifying the destination directory."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Convert a NEWAVE case directory to a Cobre case directory."""
    _configure_logging(verbose, log_file)
    _run_newave_conversion(
        ConvertArgs(
            src=src,
            dst=dst,
            validate=validate,
            force=force,
            diagnostics_json=diagnostics_json,
            json_output=json_output,
            dry_run=dry_run,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


#: Warning substring that marks the lag-blind stage shape
#: (``state_variables.inflow_lags = false`` on every stage, alongside the
#: positive inflow-lag depth cobre infers from the imported boundary policy) as
#: deliberate external-solver interoperability, not a misconfiguration. Only
#: fires once a boundary FCF is imported. Matched against cobre's stable
#: substring, never the volatile message prefix, which would drift.
_DECOMP_VALIDATION_WHITELIST: tuple[str, ...] = ("external-solver interoperability",)


def _run_decomp_conversion(args: ConvertArgs) -> None:
    """Execute the convert decomp subcommand.

    Structurally mirrors ``_run_newave_conversion``: a TTY phase bar over the
    DECOMP phases, the ``✓ Converted …`` summary, grouped diagnostic panels, a
    ``--dry-run`` branch, a provenance-manifest write, and a unified ``--json``
    verdict.

    A broad ``except Exception`` (rather than a fixed exception tuple) also
    covers an ERROR-severity post-emission self-check finding (cobre rules
    43/41/45/38/36 + the block_id-range rule) — ``convert_decomp_case`` raises a
    ``ValueError`` naming the failing rule(s) and entities, mapped to exit 1
    like every other conversion failure.

    Unless ``--no-fcf`` is set, the boundary-FCF importer runs after the
    manifest write and BEFORE ``--validate`` (so validation sees the patched
    ``config.json``) whenever the deck declares its cut files: the capability
    probe, then ``import_boundary_fcf``. A deck with no cut files simply
    converts without a boundary FCF (an INFO note, not an error); a capability
    or importer failure exits 1 via the same ``diagnostic_from_exception``
    mapping as a conversion failure. A
    successful import surfaces the C8 ``cobre run ... --output <case_dir>``
    recipe and a ``summary["boundary_fcf"]`` sub-object.

    ``--validate`` runs after a successful conversion (and boundary-FCF
    import) via the shared ``_run_cobre_validation`` helper (mirroring
    ``convert newave``), with the DECOMP external-solver-interop whitelist so
    the deliberate ``inflow_lags=false`` shape never surfaces as a scary
    warning; a failed validation exits 2, giving ``convert decomp`` the same
    0/1/2 exit-code set as ``convert newave``.
    """
    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.pipeline import (
        DECOMP_CONVERSION_PHASE_LABELS,
        FcfInputs,
        convert_decomp_case,
        discover_decomp_files,
    )

    out_console = args.out_console()
    err_console = args.err_console()
    # Threaded to `import_boundary_fcf` below, so it reuses the pipeline's own
    # in-memory `config`/`initial_conditions` dicts instead of re-reading
    # either file off disk.
    fcf_inputs = FcfInputs()

    try:
        with conversion_progress(
            len(DECOMP_CONVERSION_PHASE_LABELS),
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ) as step:
            report: ConversionReport = convert_decomp_case(
                args.src,
                args.dst,
                force=args.force,
                on_phase=step,
                dry_run=args.dry_run,
                fcf_inputs_out=fcf_inputs,
            )
    except Exception as exc:  # noqa: BLE001
        _handle_conversion_pipeline_failure(
            exc, args, command="convert decomp", err_console=err_console
        )

    if args.dry_run:
        # Dry run: report the would-write listing only; touch nothing on disk
        # (no diagnostics-json sidecar, no validation).
        if args.json_output:
            summary = _convert_verdict_summary(report)
            summary["would_write"] = sorted(
                Path(p).relative_to(args.dst).as_posix()
                for p in report.would_write_paths
            )
            status = _convert_status(report.diagnostics, success="dry-run")
            _emit_convert_json(
                build_verdict("convert decomp", status, summary, report.diagnostics)
            )
        else:
            if not args.quiet:
                _render_dry_run_summary(report, console=out_console)
            render_diagnostics(
                report.diagnostics, console=err_console, quiet=args.quiet
            )
        if args.validate:
            print_status(
                "Note: --validate is ignored under --dry-run"
                " (nothing was written to validate).",
                console=err_console,
                style="#F5A623",
            )
        if not args.no_fcf:
            print_status(
                "Note: boundary FCF import is skipped under --dry-run"
                " (nothing was written to import into).",
                console=err_console,
                style="#F5A623",
            )
        return

    # Build the convert ``summary`` up front; ``--validate`` may later append a
    # ``summary["validation"]`` sub-object (under --json), and the verdict is
    # emitted to stdout only after validation has run so that block is
    # populated. ``status`` is computed at emission time (below) from the
    # merged converter + boundary-FCF diagnostics, since the latter are not
    # known until the boundary-FCF block below has run.
    summary = _convert_verdict_summary(report)

    if not args.json_output:
        if not args.quiet:
            render_conversion_summary(report, console=out_console)
        render_diagnostics(report.diagnostics, console=err_console, quiet=args.quiet)

    # The --diagnostics-json sidecar write is deferred until AFTER the
    # boundary-FCF block below (both on its success and its failure path) so
    # it can serialize the combined converter + boundary-FCF diagnostic set —
    # mirroring the --json verdict's merge — instead of the converter-only
    # snapshot a write at this point would capture. When --boundary-fcf is
    # off (or never reaches the sink), ``boundary_diagnostics`` stays empty
    # and the sidecar content is unchanged from a converter-only write.

    # Provenance manifest, always written on a successful conversion. Notes go
    # to err_console (stderr) so the --json stdout verdict stays byte-deterministic.
    _write_conversion_manifest(
        report,
        args.src,
        args.dst,
        command="convert decomp",
        discover=discover_decomp_files,
        console=err_console,
    )

    # Runs after the manifest write and BEFORE ``--validate`` so validation
    # sees the patched ``config.json`` (``policy.boundary``); cobre infers the
    # inflow-lag depth from the boundary, so no ``state_space`` is written.
    # Cut-files-absent, the capability probe, and the importer all funnel
    # through this one broad ``except`` mapped to exit 1 (a conversion-step
    # failure), not the ``--validate`` exit-2 idiom — this is a conversion step,
    # not a validation gate. The call runs inside a ``dx.collect()`` sink so its
    # ``Diagnostic``s reach the Rich panels and the ``--json`` verdict instead
    # of degrading to invisible log records.
    boundary_diagnostics: list[Diagnostic] = []
    # The boundary FCF is imported by default; ``--no-fcf`` skips the whole
    # step (and its deck discovery). With it on, the deck's own FC records
    # (or the cortes* glob) locate the cut files; a deck that declares none
    # simply converts without a boundary FCF (an INFO note, not an error).
    # The single `DecompCase` built here is handed to the importer below, so
    # the deck is discovered (and, on first attribute access, parsed) once
    # for the FCF step rather than the importer re-discovering it itself.
    case = DecompCase.from_directory(args.src) if not args.no_fcf else None
    fcf_cut_files_present = (
        case is not None
        and case.files.cortesh is not None
        and case.files.cortes is not None
    )
    if not args.no_fcf and not fcf_cut_files_present:
        print_status(
            "Note: the deck declares no cortes/cortesh files; converting "
            "without a boundary FCF.",
            console=err_console,
            style="#F5A623",
        )
    if fcf_cut_files_present:
        assert case is not None  # narrowed by fcf_cut_files_present
        from cobre_bridge.core import diagnostics as dx

        fcf_diags: list[Diagnostic] = []
        try:
            from cobre_bridge.decomp.fcf.capability import (
                ensure_boundary_fcf_capability,
            )
            from cobre_bridge.decomp.fcf.importer import import_boundary_fcf

            ensure_boundary_fcf_capability()

            with dx.collect() as fcf_diags, tempfile.TemporaryDirectory() as work_dir:
                import_boundary_fcf(
                    args.dst,
                    case,
                    work_dir=Path(work_dir),
                    # Never None: a None cost_scale_factor triggers cobre's
                    # legacy 1e6 scaling — the source cuts are authored in
                    # cobre's native scale already.
                    cost_scale_factor=1.0,
                    config=fcf_inputs.config,
                    initial_conditions=fcf_inputs.initial_conditions,
                )
        except Exception as exc:  # noqa: BLE001
            diag = diagnostic_from_exception(exc, context="Boundary FCF import")
            failure_diagnostics = [*report.diagnostics, *fcf_diags, diag]
            if args.diagnostics_json is not None:
                # The sidecar carries the same merged set as the failure
                # verdict below: the converter's own findings, whatever the
                # importer's sink captured before it raised, and the failure
                # diagnostic itself.
                _write_diagnostics_json(
                    report,
                    args.diagnostics_json,
                    diagnostics=failure_diagnostics,
                    console=err_console,
                )
            if args.json_output:
                _emit_convert_json(
                    build_verdict(
                        "convert decomp",
                        _convert_status(failure_diagnostics, success="ok"),
                        summary,
                        failure_diagnostics,
                    )
                )
            else:
                render_diagnostics(
                    [*fcf_diags, diag], console=err_console, quiet=args.quiet
                )
            raise typer.Exit(code=1)
        else:
            boundary_diagnostics = list(fcf_diags)
            # C8 surfacing (D7, TRACKED COBRE-GAP WORKAROUND — see
            # ``fcf/importer.py::_patch_policy_boundary`` and the cobre
            # repository's conversion-found-improvements registry): until cobre
            # resolves ``policy.boundary.path`` relative to case_dir rather than
            # the run's --output directory, this case must be run with
            # ``--output <case_dir>``.
            run_constraint = f"--output={args.dst}"
            print_status(
                f"Boundary FCF imported. Run this case with: "
                f"cobre run {args.dst} {run_constraint}",
                console=err_console,
            )
            summary["boundary_fcf"] = {
                "imported": True,
                "path": "boundary",
                "run_constraint": run_constraint,
            }
            if not args.json_output:
                # boundary_diagnostics only: ``report.diagnostics`` was
                # already rendered above (the converter's own panel), so
                # this renders solely the importer's captured diagnostics —
                # never a double-render of the same findings.
                render_diagnostics(
                    boundary_diagnostics, console=err_console, quiet=args.quiet
                )

    # The merge of the converter's own findings and the boundary-FCF
    # importer's (empty when ``--boundary-fcf`` was not requested, or the
    # sink never captured anything). Computed once here so both the
    # ``--diagnostics-json`` sidecar and the ``--json`` verdict below emit the
    # identical merged set.
    combined_diagnostics = [*report.diagnostics, *boundary_diagnostics]

    if args.diagnostics_json is not None:
        # The --diagnostics-json sidecar coexists with --json (both can be
        # set); it always carries the same merged diagnostics as the
        # eventual verdict, on the success path handled here.
        _write_diagnostics_json(
            report,
            args.diagnostics_json,
            diagnostics=combined_diagnostics,
            console=err_console,
        )

    validation_failed = False
    if args.validate:
        validation_failed = _run_cobre_validation(
            args.dst,
            command="convert decomp",
            summary=summary,
            json_output=args.json_output,
            err_console=err_console,
            whitelist_substrings=_DECOMP_VALIDATION_WHITELIST,
        )

    # ``status`` is derived from ``combined_diagnostics`` (importer diagnostics
    # are INFO-only, so this stays "ok" whenever the converter's own
    # diagnostics allow it) and feeds both the --json verdict below and the
    # exit-code gate, so it is computed unconditionally.
    status = _convert_status(combined_diagnostics, success="ok")

    # Emit the --json verdict now (after validation has populated ``summary``).
    if args.json_output:
        _emit_convert_json(
            build_verdict(
                "convert decomp",
                status,
                summary,
                combined_diagnostics,
            )
        )

    _gate_convert_exit(status, validation_failed=validation_failed)


@convert_app.command("decomp")
def _convert_decomp(
    src: Annotated[Path, typer.Argument(help="Path to the DECOMP deck directory.")],
    dst: Annotated[
        Path, typer.Argument(help="Path to the output Cobre case directory.")
    ],
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Overwrite destination directory if it already contains files.",
        ),
    ] = False,
    validate: Annotated[
        bool,
        typer.Option(
            "--validate",
            help="After conversion, validate the output with the cobre package.",
        ),
    ] = False,
    diagnostics_json: Annotated[
        Path | None,
        typer.Option(
            "--diagnostics-json",
            metavar="PATH",
            help="Also write the conversion diagnostics (counts + findings) as JSON.",
        ),
    ] = None,
    json_output: _JsonOpt = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Run the full conversion in memory and report what would be "
                "written, without creating or modifying the destination directory."
            ),
        ),
    ] = False,
    no_fcf: Annotated[
        bool,
        typer.Option(
            "--no-fcf",
            help=(
                "Skip importing the deck's boundary FCF. By default, when the "
                "deck declares cortes/cortesh files (its FC records), they are "
                "imported as a terminal-stage cobre policy checkpoint via an "
                "in-process 1-iteration cobre pass (slow; requires cobre-python). "
                "Pass this for a quick conversion without the terminal FCF. "
                "The FCF is always skipped under --dry-run."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Convert a DECOMP deck revision to a Cobre case directory.

    Loop-closing subset: the exchange network, renewables card file, and GNL
    anticipation are deferred and reported as warnings. The boundary FCF is
    imported by default whenever the deck declares its cut files; ``--no-fcf``
    skips it.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_conversion(
        ConvertArgs(
            src=src,
            dst=dst,
            force=force,
            validate=validate,
            diagnostics_json=diagnostics_json,
            json_output=json_output,
            dry_run=dry_run,
            no_fcf=no_fcf,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


def _run_decomp_comparison(args: CompareArgs) -> None:
    """Execute the compare decomp subcommand.

    Informational like ``compare newave``: it always exits 0 and describes the
    divergence rather than judging it. An unreadable *existing* output file is
    the one failure (exit 2) — reporting a zero-vs-zero match on data we could
    not read would be worse than stopping.
    """
    from cobre_bridge.cobre.readers import CobreReadError
    from cobre_bridge.comparators.decomp_results import build_decomp_dataset
    from cobre_bridge.comparators.report import print_results_summary_from_dataset
    from cobre_bridge.comparators.verdict import compare_status
    from cobre_bridge.core import diagnostics as dx
    from cobre_bridge.core.errors import CobrePartitionMissingError, FieldParseError
    from cobre_bridge.core.provenance import hash_input_files
    from cobre_bridge.decomp.case import DecompCase

    # Resolved before the read (unlike the pre-dataset ordering) so
    # ``build_decomp_dataset`` below gets a concrete tolerance rather than the
    # raw, possibly-``None`` CLI value — mirrors ``_run_newave_comparison``.
    args = _resolve_compare_settings(args)

    with dx.collect() as compare_diagnostics:
        try:
            with spinner(
                "Comparing results…",
                verbose=args.verbose > 0,
                quiet=args.quiet,
                no_color=args.no_color,
            ):
                dataset = build_decomp_dataset(
                    args.source_dir, args.cobre_output_dir, tolerance=args.tolerance
                )
        except (
            CobreReadError,
            CobrePartitionMissingError,
            FieldParseError,
            FileNotFoundError,
            ValueError,
        ) as exc:
            _fail("compare decomp", args, exc, 2)

    if not args.json_output:
        if not args.quiet:
            print_results_summary_from_dataset(
                dataset,
                args.source_dir,
                args.cobre_output_dir,
                reference_label="DECOMP",
                console=args.out_console(),
            )
        render_diagnostics(
            compare_diagnostics, console=args.err_console(), quiet=args.quiet
        )

    decomp_case = DecompCase.from_directory(args.source_dir)

    formats, out_dir = _export_compare_artifacts(
        dataset,
        command="compare decomp",
        args=args,
        raw_formats=args.format,
        source_dir=args.source_dir,
        cobre_output_dir=args.cobre_output_dir,
        tolerance=args.tolerance,
        out_dir_arg=args.out_dir,
        input_files=hash_input_files(decomp_case.files),
        diagnostics=[d.to_dict() for d in compare_diagnostics],
        quiet_status=args.json_output,
    )

    if "html" in formats:
        _write_html_compare_report(dataset, out_dir, args, reference_label="DECOMP")

    if args.json_output:
        # ``status`` is DECOUPLED from the exit code (this command always
        # exits 0, mirroring ``compare newave``) and uses the shared
        # ``compare_status`` vocabulary.
        summary = decomp_dataset_summary(dataset, args.tolerance)
        status = compare_status(dataset)
        _emit_convert_json(
            build_verdict("compare decomp", status, summary, compare_diagnostics)
        )


@compare_app.command("decomp")
def _compare_decomp(
    decomp_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the DECOMP deck directory (deck + dec_oper_*.csv "
            "result files, all directly in it)."
        ),
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    tolerance: _ToleranceOpt = None,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare a DECOMP run's published operation against Cobre's simulation.

    Informational: always exits 0, reporting divergences without failing.

    Two caveats apply to the generated report. First, the Overview tab's NPV
    cost cards compare DECOMP's undiscounted-nominal costs against Cobre's
    time-discounted costs: DECOMP's own cost report carries no per-stage
    discount factor, so none is fabricated on that side, and the two totals
    are not on the same time-value footing. Second, percentile bands are
    omitted (or labelled low-N where one would otherwise appear) for a
    deterministic tree with too few scenarios to report a spread without
    synthesizing it.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_comparison(
        CompareArgs(
            source_dir=decomp_dir,
            cobre_output_dir=cobre_output_dir,
            format=fmt,
            out_dir=out_dir,
            tolerance=tolerance,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@compare_app.command("newave")
def _compare_newave(
    newave_dir: Annotated[
        Path,
        typer.Argument(
            help="Path to the NEWAVE case directory (case + MEDIAS-*.CSV "
            "result files, all directly in it)."
        ),
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    tolerance: _ToleranceOpt = None,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare NEWAVE published results against Cobre simulation output.

    Informational: always exits 0, reporting divergences without failing.
    """
    _configure_logging(verbose, log_file)
    _run_newave_comparison(
        CompareArgs(
            source_dir=newave_dir,
            cobre_output_dir=cobre_output_dir,
            tolerance=tolerance,
            format=fmt,
            out_dir=out_dir,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@check_app.command("newave")
def _check_newave(
    src: Annotated[Path, typer.Argument(help="Path to the NEWAVE case directory.")],
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Validate a NEWAVE case directory without converting or writing any files."""
    _configure_logging(verbose, log_file)
    _run_check(
        CheckArgs(
            src=src,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@check_app.command("decomp")
def _check_decomp(
    src: Annotated[Path, typer.Argument(help="Path to the DECOMP deck directory.")],
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Validate a DECOMP deck revision without converting or writing any files.

    Also reports what the conversion will leave behind, so a deferred feature
    is never a silent omission.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_check(
        CheckArgs(
            src=src,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


@app.command("dashboard")
def _dashboard(
    case_dir: Annotated[Path, typer.Argument(help="Path to the Cobre case directory.")],
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Output HTML file path (default: <case_dir>/dashboard.html).",
        ),
    ] = None,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open",
            help=(
                "Open the generated dashboard in the default web browser "
                "after writing it."
            ),
        ),
    ] = False,
    json_output: _JsonOpt = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Generate an interactive HTML dashboard from Cobre simulation results."""
    _configure_logging(verbose, log_file)
    _run_dashboard(
        DashboardArgs(
            case_dir=case_dir,
            output=output,
            open_browser=open_browser,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


def main() -> None:
    """Console entry point: run the Typer app, restoring the logger afterwards.

    The thin wrapper restores the ``cobre_bridge`` logger ``propagate`` flag that
    ``_configure_logging`` flips, and removes + closes any ``--log-file``
    ``FileHandler`` it attached, so a real CLI run never leaks logging state.
    """
    pkg_logger = logging.getLogger("cobre_bridge")
    prior_propagate = pkg_logger.propagate
    try:
        app()
    finally:
        pkg_logger.propagate = prior_propagate
        restore_log_file_handler(pkg_logger)


if __name__ == "__main__":
    main()
