"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import dataclasses
import logging
import webbrowser
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from cobre_bridge import __version__
from cobre_bridge.cli.args import (
    CheckArgs,
    CompareArgs,
    ConvertArgs,
    DashboardArgs,
    _parse_formats,
)
from cobre_bridge.cli.config import (
    RESULTS_TOLERANCE_DEFAULT,
    load_config,
)
from cobre_bridge.cli.convert import _run_decomp_conversion, _run_newave_conversion
from cobre_bridge.cli.failure import _emit_convert_json, _fail

# noqa: F401 below -- re-exported so `cli._NULL_HANDLER` keeps resolving for the
# ``test_configure_logging_levels`` import + the ``cli._configure_logging`` spy sites.
from cobre_bridge.cli.logging_config import NULL_HANDLER as _NULL_HANDLER  # noqa: F401
from cobre_bridge.cli.logging_config import configure_logging as _configure_logging
from cobre_bridge.cli.logging_config import restore_log_file_handler
from cobre_bridge.cli.validate import (
    _partition_validation_warnings as _partition_validation_warnings,
)
from cobre_bridge.cli.verdict import (
    build_verdict,
    check_summary,
    compare_summary,
    dashboard_summary,
    decomp_dataset_summary,
)
from cobre_bridge.cobre.compat import MIN_COBRE_VERSION as MIN_COBRE_VERSION
from cobre_bridge.core.errors import CobreOutputError
from cobre_bridge.core.preflight import PreflightVerdict
from cobre_bridge.ui.console import (
    get_console,
    print_status,
    render_checklist,
    render_diagnostics,
    spinner,
)

if TYPE_CHECKING:
    from cobre_bridge.comparators.alignment import EntityAlignment
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.newave.case import NewaveCase
    from cobre_bridge.newave.id_map import NewaveIdMap


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
    from cobre_bridge.cobre.readers import read_cobre_lines
    from cobre_bridge.comparators.alignment import build_entity_alignment
    from cobre_bridge.newave.case import NewaveCase

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
