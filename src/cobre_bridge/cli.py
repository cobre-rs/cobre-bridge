"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated

import typer

from cobre_bridge import __version__
from cobre_bridge.cobre_io import case_dir_for
from cobre_bridge.diagnostics import Severity
from cobre_bridge.errors import diagnostic_from_exception
from cobre_bridge.preflight import PreflightVerdict
from cobre_bridge.ui.console import (
    conversion_progress,
    get_console,
    print_status,
    render_checklist,
    render_conversion_summary,
    render_diagnostics,
    render_error,
    spinner,
)

if TYPE_CHECKING:
    from rich.console import Console

    from cobre_bridge.case import NewaveCase
    from cobre_bridge.comparators.alignment import EntityAlignment
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.diagnostics import Diagnostic
    from cobre_bridge.id_map import NewaveIdMap
    from cobre_bridge.pipeline import ConversionReport
    from cobre_bridge.preflight import PreflightResult


def _load_lines_json(cobre_output_dir: Path) -> list[dict]:
    """Load lines.json from the Cobre case directory.

    Searches for ``system/lines.json`` near the output directory.
    Returns an empty list if not found.
    """
    cobre_case_dir = case_dir_for(cobre_output_dir)
    lines_path = cobre_case_dir / "system" / "lines.json"
    if not lines_path.exists():
        for candidate in [cobre_output_dir, cobre_output_dir.parent]:
            p = candidate / "system" / "lines.json"
            if p.exists():
                lines_path = p
                break

    if not lines_path.exists():
        return []

    with lines_path.open() as f:
        lines_data = json.load(f)
    return lines_data.get("lines", [])


#: The ``--format`` tokens the compare subcommands accept on the CLI.
_VALID_CLI_FORMATS: frozenset[str] = frozenset(
    {"console", "html", "csv", "parquet", "json", "all"}
)


def _parse_formats(raw: list[str] | None) -> set[str]:
    """Parse ``--format`` tokens (comma-separated and/or repeatable) into a set.

    Defaults to ``{"console", "parquet", "json"}`` when none given, so a plain
    ``compare`` run still writes the queryable data artifacts (the always-on
    behavior agents rely on). Expands ``"all"`` to every concrete format.
    Raises ``ValueError`` naming the offending token on any unknown value.
    """
    if raw is None:
        return {"console", "parquet", "json"}

    formats: set[str] = set()
    for element in raw:
        for token in element.split(","):
            cleaned = token.strip().lower()
            if not cleaned:
                continue
            if cleaned not in _VALID_CLI_FORMATS:
                msg = (
                    f"unknown format '{cleaned}'; allowed formats are "
                    f"{sorted(_VALID_CLI_FORMATS)}"
                )
                raise ValueError(msg)
            if cleaned == "all":
                formats |= {"console", "html", "csv", "parquet", "json"}
            else:
                formats.add(cleaned)
    return formats


def _load_compare_context(
    newave_dir: Path,
    cobre_output_dir: Path,
) -> tuple[NewaveCase, NewaveIdMap, EntityAlignment, list[dict[str, object]]]:
    """Load the source model case, id-map, entity alignment, and lines.json.

    Shared setup for `compare bounds` and `compare results`. Builds the parsed
    case once (so the id-map reuses its cached readers), loads lines.json, and
    builds the entity alignment.

    Exits the process with code 1 (clean stderr message) if the source model case
    directory is missing (FileNotFoundError from NewaveCase.from_directory).
    """
    from cobre_bridge.case import NewaveCase
    from cobre_bridge.comparators.alignment import build_entity_alignment

    # Build the parsed case once; the id-map reuses its cached readers.
    try:
        case = NewaveCase.from_directory(newave_dir)
    except FileNotFoundError as exc:
        render_error(str(exc))
        raise typer.Exit(code=1)

    id_map = case.id_map
    lines_json = _load_lines_json(cobre_output_dir)
    alignment = build_entity_alignment(id_map, case, lines_json)
    return case, id_map, alignment, lines_json


def _export_compare_artifacts(
    dataset: ComparisonDataset,
    *,
    command: str,
    raw_formats: list[str] | None,
    newave_dir: Path,
    cobre_output_dir: Path,
    tolerance: float,
    out_dir_arg: Path | None,
) -> tuple[set[str], Path]:
    """Resolve ``--format`` and write the machine-readable comparison artifacts.

    Shared by `compare bounds` and `compare results`. Returns the requested
    formats and the resolved out_dir so each handler can run its own HTML branch
    and exit-code logic.

    An invalid ``--format`` token exits 2 (clean stderr). A write failure must
    NOT change the comparison exit code, so an ``OSError`` is warned and
    swallowed.
    """
    from cobre_bridge.comparators.export import write_artifacts

    try:
        formats = _parse_formats(raw_formats)
    except ValueError as exc:
        render_error(str(exc))
        raise typer.Exit(code=2)

    out_dir: Path = out_dir_arg or (cobre_output_dir / "comparison_artifacts")
    export_formats = formats & {"csv", "parquet", "json"}

    try:
        write_artifacts(
            dataset,
            command=command,
            newave_dir=newave_dir,
            cobre_output_dir=cobre_output_dir,
            tolerance=tolerance,
            out_dir=out_dir,
            formats=sorted(export_formats),
        )
        print_status(f"Artifacts written to {out_dir}")
    except OSError as exc:
        print_status(
            f"Warning: failed to write artifacts: {exc}",
            console=get_console(stderr=True),
            style="#F5A623",
        )

    return formats, out_dir


def _run_bounds_comparison(args: SimpleNamespace) -> None:
    """Execute the compare bounds subcommand."""
    from cobre_bridge.comparators.analyze import build_bounds_dataset
    from cobre_bridge.comparators.bounds import compare_bounds
    from cobre_bridge.comparators.cobre_readers import CobreReadError
    from cobre_bridge.comparators.report import (
        print_bounds_mismatches_from_dataset,
        print_bounds_summary_from_dataset,
    )

    newave_dir: Path = args.newave_dir
    cobre_output_dir: Path = args.cobre_output_dir
    tolerance: float = args.tolerance

    # Validate paths.
    bounds_path = cobre_output_dir / "training" / "dictionaries" / "bounds.parquet"
    if not bounds_path.exists():
        render_error(
            f"bounds.parquet not found at {bounds_path}. "
            "Run cobre with --output first.",
        )
        raise typer.Exit(code=1)

    case, id_map, alignment, _lines_json = _load_compare_context(
        newave_dir, cobre_output_dir
    )

    variables: set[str] | None = None
    if args.variables:
        variables = {v.strip() for v in args.variables.split(",")}

    # Run comparison.  A CobreReadError means an *existing* Cobre output file
    # was unreadable/malformed — fail loudly (exit 2) rather than report a
    # false "no divergence" on data we could not actually read.
    try:
        with spinner(
            "Comparing bounds…",
            verbose=args.verbose,
            quiet=args.quiet,
            no_color=args.no_color,
        ):
            results = compare_bounds(
                alignment=alignment,
                case=case,
                id_map=id_map,
                cobre_output_dir=cobre_output_dir,
                tolerance=tolerance,
                variables=variables,
            )
    except CobreReadError as exc:
        print_status(
            f"ERROR: {exc}", console=get_console(stderr=True), style="bold #DC4C4C"
        )
        raise typer.Exit(code=2)

    # Build the canonical dataset once; console + artifacts derive from it.
    dataset = build_bounds_dataset(results)

    # Output (sourced from the dataset).
    print_bounds_summary_from_dataset(dataset, newave_dir, cobre_output_dir, tolerance)

    if not args.summary:
        print_bounds_mismatches_from_dataset(dataset)

    formats, _out_dir = _export_compare_artifacts(
        dataset,
        command="compare bounds",
        raw_formats=args.format,
        newave_dir=newave_dir,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
        out_dir_arg=args.out_dir,
    )

    # Bounds has no HTML report; honor --format html with an ignore-warning.
    if "html" in formats:
        print_status(
            "Warning: --format html is not supported for 'compare bounds' "
            "(no HTML report); ignoring.",
            console=get_console(stderr=True),
            style="#F5A623",
        )

    mismatches = sum(1 for r in results if not r.match)
    if mismatches:
        raise typer.Exit(code=1)


def _run_results_comparison(args: SimpleNamespace) -> None:
    """Execute the compare results subcommand.

    Intentionally always exits 0: ``compare results`` is informational (a
    descriptive the source-model-vs-Cobre divergence report), so it never signals a
    failure on divergence. This is asymmetric with ``compare bounds``, which exits 1 on
    any mismatch — bounds are a strict equivalence check.
    """
    from cobre_bridge.comparators.cobre_readers import CobreReadError
    from cobre_bridge.comparators.report import print_results_summary_from_dataset
    from cobre_bridge.comparators.results import compare_results

    newave_dir: Path = args.newave_dir
    cobre_output_dir: Path = args.cobre_output_dir
    tolerance: float = args.tolerance

    case, id_map, alignment, _lines_json = _load_compare_context(
        newave_dir, cobre_output_dir
    )

    # Run comparison.  A CobreReadError means an *existing* Cobre output file
    # was unreadable/malformed — fail loudly (exit 2) rather than report a
    # false "no divergence" on data we could not actually read.
    try:
        with spinner(
            "Comparing results…",
            verbose=args.verbose,
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
    except CobreReadError as exc:
        print_status(
            f"ERROR: {exc}", console=get_console(stderr=True), style="bold #DC4C4C"
        )
        raise typer.Exit(code=2)

    # Print text summary (sourced from the dataset).
    print_results_summary_from_dataset(dataset, newave_dir, cobre_output_dir)

    formats, out_dir = _export_compare_artifacts(
        dataset,
        command="compare results",
        raw_formats=args.format,
        newave_dir=newave_dir,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
        out_dir_arg=args.out_dir,
    )

    # HTML report (opt-in via --format html / all).
    if "html" in formats:
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )

        html = build_comparison_report(dataset)
        report_path = out_dir / "report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html, encoding="utf-8")
        print_status(f"HTML report written to {report_path}")

    return


def _run_dashboard(args: SimpleNamespace) -> None:
    """Execute the dashboard subcommand."""
    from cobre_bridge.dashboard import build_dashboard

    case_dir: Path = args.case_dir.resolve()
    if not (case_dir / "output" / "simulation").exists():
        render_error(f"no simulation output found in {case_dir}")
        raise typer.Exit(code=1)

    output_path: Path = args.output or (case_dir / "dashboard.html")
    print_status(f"Building dashboard from {case_dir} ...")
    with spinner(
        "Building dashboard…",
        verbose=args.verbose,
        quiet=args.quiet,
        no_color=args.no_color,
    ):
        build_dashboard(case_dir, output_path)
    size_kb = output_path.stat().st_size / 1024
    print_status(f"Dashboard written to {output_path} ({size_kb:.0f} KB)")
    return


#: Preflight verdict → process exit code. The contract is fixed by the epic
#: overview: ``OK`` is clean (0), ``WARNINGS`` is advisory (1), and
#: ``WILL_NOT_CONVERT`` is the most severe (2). Kept as data so the mapping is
#: directly unit-testable and impossible to drift from in the handler.
_VERDICT_EXIT_CODE: dict[PreflightVerdict, int] = {
    PreflightVerdict.OK: 0,
    PreflightVerdict.WARNINGS: 1,
    PreflightVerdict.WILL_NOT_CONVERT: 2,
}


def _check_json_document(result: PreflightResult) -> dict[str, object]:
    """Build the ``--json`` verdict document for ``check newave``.

    Mirrors the ``_convert_json_document`` shape from the convert stub so Epic 07
    can later generalize both: a fixed-order ``{"command", "status", "checks",
    "diagnostics"}`` mapping. ``status`` is taken verbatim from the preflight
    ``verdict.value`` (it is never recomputed here), each :class:`CheckItem`
    becomes a ``{"label", "passed", "detail"}`` object, and each diagnostic is
    serialized via its own ``to_dict``.

    The document carries no timestamps and no paths beyond what the diagnostics
    already hold, so the serialized form is deterministic across runs.
    """
    return {
        "command": "check newave",
        "status": result.verdict.value,
        "checks": [
            {"label": check.label, "passed": check.passed, "detail": check.detail}
            for check in result.checks
        ],
        "diagnostics": [d.to_dict() for d in result.diagnostics],
    }


def _run_check(args: SimpleNamespace) -> None:
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
        _emit_convert_json(_check_json_document(result))
    else:
        # Default consoles: the ✓/✗ checklist goes to stdout while render_checklist
        # delegates its diagnostics block to render_diagnostics (stderr default),
        # preserving the stdout(results)/stderr(diagnostics) split.
        render_checklist(result, quiet=args.quiet)

    exit_code = _VERDICT_EXIT_CODE[result.verdict]
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
    return


def _run_newave_conversion(args: SimpleNamespace) -> None:
    """Execute the convert newave subcommand."""
    # Import here so the module-level import of pipeline is deferred.
    from cobre_bridge.pipeline import (
        CONVERSION_PHASE_LABELS,
        _clear_dst_contents,
        convert_newave_case,
    )

    src: Path = args.src
    dst: Path = args.dst

    out_console = get_console(no_color=args.no_color)
    err_console = get_console(stderr=True, no_color=args.no_color)

    # ------------------------------------------------------------------
    # Source validation.
    # ------------------------------------------------------------------
    if not src.exists() or not src.is_dir():
        render_error(f"source directory '{src}' does not exist", console=err_console)
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------
    # Destination validation.
    # ------------------------------------------------------------------
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            render_error(
                f"destination directory '{dst}' is not empty."
                " Use --force to overwrite.",
                console=err_console,
            )
            raise typer.Exit(code=1)
        # --force: remove previous pipeline outputs before converting.
        _clear_dst_contents(dst)

    dst.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run conversion pipeline.
    # ------------------------------------------------------------------
    try:
        with conversion_progress(
            len(CONVERSION_PHASE_LABELS),
            verbose=args.verbose,
            quiet=args.quiet,
            no_color=args.no_color,
        ) as step:
            report: ConversionReport = convert_newave_case(src, dst, on_phase=step)
    except Exception as exc:  # noqa: BLE001
        diag = diagnostic_from_exception(exc, context="Conversion")
        if args.json_output:
            _emit_convert_json(_convert_json_document(None, [diag]))
        else:
            render_diagnostics([diag], console=err_console, quiet=args.quiet)
        raise typer.Exit(code=1)

    if args.json_output:
        # --json: one machine-readable verdict to stdout, no Rich on either stream.
        _emit_convert_json(_convert_json_document(report, report.diagnostics))
    else:
        if not args.quiet:
            render_conversion_summary(report, console=out_console)
        render_diagnostics(report.diagnostics, console=err_console, quiet=args.quiet)

    if args.diagnostics_json is not None:
        # The --diagnostics-json sidecar coexists with --json (both can be set).
        _write_diagnostics_json(report, args.diagnostics_json, console=err_console)

    # Provenance manifest, always written on a successful conversion. Notes go
    # to err_console (stderr) so the --json stdout verdict stays byte-deterministic.
    _write_conversion_manifest(report, src, dst, console=err_console)

    # ------------------------------------------------------------------
    # Optional post-conversion validation.
    # ------------------------------------------------------------------
    # NOTE: --validate is out of scope for the --json stub in this ticket. When
    # --validate and --json are combined, validation messages render as today
    # (Rich on stderr, exit 2 on failure); Epic 7 unifies validation into the
    # JSON verdict.
    if args.validate:
        try:
            import cobre.io  # type: ignore[import-untyped]
        except ImportError:
            print_status(
                "Warning: cobre package not installed, skipping validation",
                console=err_console,
                style="#F5A623",
            )
            return

        try:
            # cobre v0.6.x: cobre.io.validate is a function returning a
            # report dict; it never raises (errors are surfaced as data).
            result = cobre.io.validate(str(dst))
        except Exception as exc:  # noqa: BLE001
            render_error(f"Validation error: {exc}", console=err_console)
            raise typer.Exit(code=2)

        def _msg(item: object) -> object:
            return item.get("message", item) if isinstance(item, dict) else item

        for warning in result.get("warnings", []):
            print_status(
                f"Validation warning: {_msg(warning)}",
                console=err_console,
                style="#F5A623",
            )
        if not result.get("valid", False):
            for err in result.get("errors", []):
                print_status(
                    f"Validation error: {_msg(err)}",
                    console=err_console,
                    style="bold #DC4C4C",
                )
            print_status(
                "Validation failed.", console=err_console, style="bold #DC4C4C"
            )
            raise typer.Exit(code=2)

    return


def _convert_json_document(
    report: ConversionReport | None,
    diagnostics: list[Diagnostic],
) -> dict[str, object]:
    """Build the ``--json`` verdict document for ``convert newave``.

    Reuses the ``{"summary": {...}, "diagnostics": [...]}`` payload shape from
    :func:`_write_diagnostics_json`, adding the ``command`` and ``status`` keys
    that make it a self-contained machine-readable verdict. ``report`` is ``None``
    on the failure path, where the summary counts are all zero. ``status`` is
    ``"error"`` when any diagnostic has ``ERROR`` severity, else ``"ok"``.

    The returned dict has a fixed insertion order and carries no timestamps or
    paths beyond what the diagnostics already hold, so the serialized form is
    deterministic across runs.
    """
    status = "error" if any(d.severity is Severity.ERROR for d in diagnostics) else "ok"
    return {
        "command": "convert newave",
        "status": status,
        "summary": {
            "hydros": report.hydro_count if report is not None else 0,
            "thermals": report.thermal_count if report is not None else 0,
            "buses": report.bus_count if report is not None else 0,
            "lines": report.line_count if report is not None else 0,
            "stages": report.stage_count if report is not None else 0,
        },
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def _emit_convert_json(document: dict[str, object]) -> None:
    """Write the ``--json`` verdict *document* to stdout as one JSON object.

    Writes directly to ``sys.stdout`` (NOT through the Rich console, which may
    inject styling/wrapping), with a trailing newline. A fixed insertion order is
    preserved (``sort_keys=False``) so the output is byte-stable.
    """
    json.dump(document, sys.stdout, indent=2, ensure_ascii=False, sort_keys=False)
    sys.stdout.write("\n")


def _write_diagnostics_json(
    report: ConversionReport, path: Path, *, console: Console
) -> None:
    """Write the conversion counts + diagnostics to *path* as JSON.

    A write failure is reported but does not change the exit code — the conversion
    itself already succeeded.
    """
    payload = {
        "summary": {
            "hydros": report.hydro_count,
            "thermals": report.thermal_count,
            "buses": report.bus_count,
            "lines": report.line_count,
            "stages": report.stage_count,
        },
        "diagnostics": [d.to_dict() for d in report.diagnostics],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print_status(
            f"Warning: failed to write diagnostics JSON: {exc}",
            console=console,
            style="#F5A623",
        )
    else:
        print_status(f"Diagnostics written to {path}", console=console)


def _write_conversion_manifest(
    report: ConversionReport, src: Path, dst: Path, *, console: Console
) -> None:
    """Write the conversion provenance manifest into ``dst`` as JSON.

    Rediscovers the source-model input files to hash, builds a
    :class:`ConversionManifest` from the bridge version/git SHA, the entity
    counts in *report*, and its diagnostics, then writes it to
    ``dst / "conversion_manifest.json"``.

    Both a discovery failure and a write failure are reported as warnings and
    swallowed — the conversion itself already succeeded, so neither changes the
    exit code.
    """
    from cobre_bridge.conversion_manifest import (
        ConversionManifest,
        hash_input_files,
        summarize_diagnostics,
    )
    from cobre_bridge.newave_files import NewaveFiles

    try:
        files = NewaveFiles.from_directory(src)
    except OSError as exc:
        print_status(
            f"Warning: failed to discover source files for conversion manifest: {exc}",
            console=console,
            style="#F5A623",
        )
        return

    entity_counts = {
        "hydros": report.hydro_count,
        "thermals": report.thermal_count,
        "buses": report.bus_count,
        "lines": report.line_count,
        "stages": report.stage_count,
    }
    manifest = ConversionManifest.create(
        "convert newave",
        src,
        dst,
        entity_counts=entity_counts,
        input_files=hash_input_files(files),
        diagnostics_summary=summarize_diagnostics(report.diagnostics),
        diagnostics=[d.to_dict() for d in report.diagnostics],
    )

    path = dst / "conversion_manifest.json"
    try:
        manifest.to_json(path)
    except OSError as exc:
        print_status(
            f"Warning: failed to write conversion manifest: {exc}",
            console=console,
            style="#F5A623",
        )
    else:
        print_status(f"Conversion manifest written to {path}", console=console)


#: A no-op handler parked on the package logger when warnings are suppressed, so a
#: suppressed record does not fall through to ``logging.lastResort`` (which would
#: otherwise echo it to stderr, defeating the suppression).
_NULL_HANDLER = logging.NullHandler()


def _configure_logging(verbose: bool) -> None:
    """Configure logging for a CLI run.

    With ``--verbose``, everything down to DEBUG is logged live (power-user mode).
    Without it, ``cobre_bridge`` warnings are still recorded — the diagnostics
    collector and ``--diagnostics-json`` rely on them — but kept off the live
    console so the Rich diagnostics block is the single user-facing surface, and so
    warnings are not printed twice. ``main`` restores ``propagate`` afterwards.
    """
    pkg = logging.getLogger("cobre_bridge")
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(name)s: %(message)s",
        )
        pkg.setLevel(logging.DEBUG)
        pkg.propagate = True
    else:
        # Leave the package logger level untouched (root's default WARNING already
        # records warnings for the collector); just keep them off the live console.
        if _NULL_HANDLER not in pkg.handlers:
            pkg.addHandler(_NULL_HANDLER)
        pkg.propagate = False


# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

# Shared flags reused across the leaf commands (kept *after* the subcommand,
# matching the previous argparse UX).
_VerboseOpt = Annotated[
    bool, typer.Option("--verbose", help="Enable detailed logging output.")
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
        help=(
            "Output format(s): console,html,csv,parquet,json,all. "
            "Comma-separated and/or repeatable. (default: console,parquet,json)"
        ),
    ),
]
_OutDirOpt = Annotated[
    Path | None,
    typer.Option(
        "--out-dir",
        help=(
            "Directory for file artifacts "
            "(default: <cobre_output_dir>/comparison_artifacts)."
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
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a single machine-readable JSON verdict to stdout and "
                "suppress the human (Rich) rendering."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = False,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Convert a NEWAVE case directory to a Cobre case directory."""
    _configure_logging(verbose)
    _run_newave_conversion(
        SimpleNamespace(
            src=src,
            dst=dst,
            validate=validate,
            force=force,
            diagnostics_json=diagnostics_json,
            json_output=json_output,
            verbose=verbose,
            no_color=no_color,
            quiet=quiet,
        )
    )


@compare_app.command("bounds")
def _compare_bounds(
    newave_dir: Annotated[
        Path, typer.Argument(help="Path to the NEWAVE case directory.")
    ],
    cobre_output_dir: Annotated[
        Path,
        typer.Argument(help="Path to the Cobre output directory (has bounds.parquet)."),
    ],
    tolerance: Annotated[
        float, typer.Option(help="Absolute tolerance for bound comparison.")
    ] = 1e-3,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    summary: Annotated[
        bool,
        typer.Option(
            "--summary", help="Print only summary counts, not individual mismatches."
        ),
    ] = False,
    variables: Annotated[
        str | None,
        typer.Option(
            "--variables",
            help="Comma-separated variables to compare (e.g. storage_min,turbined).",
        ),
    ] = None,
    verbose: _VerboseOpt = False,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare LP bounds computed from NEWAVE inputs against Cobre bounds."""
    _configure_logging(verbose)
    _run_bounds_comparison(
        SimpleNamespace(
            newave_dir=newave_dir,
            cobre_output_dir=cobre_output_dir,
            tolerance=tolerance,
            format=fmt,
            out_dir=out_dir,
            summary=summary,
            variables=variables,
            verbose=verbose,
            no_color=no_color,
            quiet=quiet,
        )
    )


@compare_app.command("results")
def _compare_results(
    newave_dir: Annotated[
        Path, typer.Argument(help="Path to the NEWAVE case directory (has saidas/).")
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    tolerance: Annotated[
        float, typer.Option(help="Relative tolerance for results comparison.")
    ] = 1e-2,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    verbose: _VerboseOpt = False,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare NEWAVE published results against Cobre simulation output.

    Informational: always exits 0, whereas 'compare bounds' exits 1 on any mismatch.
    """
    _configure_logging(verbose)
    _run_results_comparison(
        SimpleNamespace(
            newave_dir=newave_dir,
            cobre_output_dir=cobre_output_dir,
            tolerance=tolerance,
            format=fmt,
            out_dir=out_dir,
            verbose=verbose,
            no_color=no_color,
            quiet=quiet,
        )
    )


@check_app.command("newave")
def _check_newave(
    src: Annotated[Path, typer.Argument(help="Path to the NEWAVE case directory.")],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a single machine-readable JSON verdict to stdout and "
                "suppress the human (Rich) checklist."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = False,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Validate a NEWAVE case directory without converting or writing any files."""
    _configure_logging(verbose)
    _run_check(
        SimpleNamespace(
            src=src,
            json_output=json_output,
            verbose=verbose,
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
    verbose: _VerboseOpt = False,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Generate an interactive HTML dashboard from Cobre simulation results."""
    _configure_logging(verbose)
    _run_dashboard(
        SimpleNamespace(
            case_dir=case_dir,
            output=output,
            verbose=verbose,
            no_color=no_color,
            quiet=quiet,
        )
    )


def main() -> None:
    """Console entry point: run the Typer app, restoring the logger afterwards.

    The thin wrapper restores the ``cobre_bridge`` logger ``propagate`` flag that
    ``_configure_logging`` flips, so a real CLI run never leaks logging state.
    """
    pkg_logger = logging.getLogger("cobre_bridge")
    prior_propagate = pkg_logger.propagate
    try:
        app()
    finally:
        pkg_logger.propagate = prior_propagate


if __name__ == "__main__":
    main()
