"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated

import typer

from cobre_bridge import __version__
from cobre_bridge.cobre_io import case_dir_for
from cobre_bridge.ui.console import (
    conversion_progress,
    get_console,
    print_status,
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
    from cobre_bridge.id_map import NewaveIdMap
    from cobre_bridge.pipeline import ConversionReport


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
    except FileNotFoundError as exc:
        missing = str(exc)
        render_error(
            f"required file '{missing}' not found in {src}", console=err_console
        )
        raise typer.Exit(code=1)
    except Exception as exc:  # noqa: BLE001
        render_error(f"conversion failed: {exc}", console=err_console)
        raise typer.Exit(code=1)

    if not args.quiet:
        render_conversion_summary(report, console=out_console)
    render_diagnostics(report.diagnostics, console=err_console, quiet=args.quiet)

    if args.diagnostics_json is not None:
        _write_diagnostics_json(report, args.diagnostics_json, console=err_console)

    # ------------------------------------------------------------------
    # Optional post-conversion validation.
    # ------------------------------------------------------------------
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
app.add_typer(convert_app, name="convert")
app.add_typer(compare_app, name="compare")


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
