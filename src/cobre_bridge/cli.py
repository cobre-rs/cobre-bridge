"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from cobre_bridge import __version__
from cobre_bridge.cobre_io import case_dir_for
from cobre_bridge.ui.console import (
    get_console,
    print_status,
    render_conversion_summary,
    render_diagnostics,
    render_error,
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
        sys.exit(1)

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
        sys.exit(2)

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


def _run_bounds_comparison(args: argparse.Namespace) -> None:
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
        sys.exit(1)

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
        sys.exit(2)

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
    sys.exit(0 if mismatches == 0 else 1)


def _run_results_comparison(args: argparse.Namespace) -> None:
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
        sys.exit(2)

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

    sys.exit(0)


def _run_dashboard(args: argparse.Namespace) -> None:
    """Execute the dashboard subcommand."""
    from cobre_bridge.dashboard import build_dashboard

    case_dir: Path = args.case_dir.resolve()
    if not (case_dir / "output" / "simulation").exists():
        render_error(f"no simulation output found in {case_dir}")
        sys.exit(1)

    output_path: Path = args.output or (case_dir / "dashboard.html")
    print_status(f"Building dashboard from {case_dir} ...")
    build_dashboard(case_dir, output_path)
    size_kb = output_path.stat().st_size / 1024
    print_status(f"Dashboard written to {output_path} ({size_kb:.0f} KB)")
    sys.exit(0)


def _run_newave_conversion(args: argparse.Namespace) -> None:
    """Execute the convert newave subcommand."""
    # Import here so the module-level import of pipeline is deferred.
    from cobre_bridge.pipeline import (
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
        sys.exit(1)

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
            sys.exit(1)
        # --force: remove previous pipeline outputs before converting.
        _clear_dst_contents(dst)

    dst.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Run conversion pipeline.
    # ------------------------------------------------------------------
    try:
        report: ConversionReport = convert_newave_case(src, dst)
    except FileNotFoundError as exc:
        missing = str(exc)
        render_error(
            f"required file '{missing}' not found in {src}", console=err_console
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        render_error(f"conversion failed: {exc}", console=err_console)
        sys.exit(1)

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
            sys.exit(0)

        try:
            # cobre v0.6.x: cobre.io.validate is a function returning a
            # report dict; it never raises (errors are surfaced as data).
            result = cobre.io.validate(str(dst))
        except Exception as exc:  # noqa: BLE001
            render_error(f"Validation error: {exc}", console=err_console)
            sys.exit(2)

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
            sys.exit(2)

    sys.exit(0)


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


def main() -> None:
    """Entry point for the cobre-bridge CLI."""
    parser = argparse.ArgumentParser(
        prog="cobre-bridge",
        description="Convert power system data to Cobre input format.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Flags shared by every leaf subcommand, attached via ``parents=``.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )
    common.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured output (also honoured via the NO_COLOR env var).",
    )
    common.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the summary and info notes; warnings/errors still show.",
    )

    # convert subcommand
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert data from a source format to Cobre JSON.",
    )
    convert_subparsers = convert_parser.add_subparsers(
        dest="source",
        metavar="SOURCE",
        required=True,
    )

    # convert newave sub-subcommand
    newave_parser = convert_subparsers.add_parser(
        "newave",
        parents=[common],
        help="Convert a NEWAVE case directory to a Cobre case directory.",
    )
    newave_parser.add_argument(
        "src",
        metavar="SRC",
        type=Path,
        help="Path to the NEWAVE case directory.",
    )
    newave_parser.add_argument(
        "dst",
        metavar="DST",
        type=Path,
        help="Path to the output Cobre case directory.",
    )
    newave_parser.add_argument(
        "--validate",
        action="store_true",
        help="After conversion, validate the output with the cobre package.",
    )
    newave_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination directory if it already contains files.",
    )
    newave_parser.add_argument(
        "--diagnostics-json",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also write the conversion diagnostics (counts + findings) as JSON.",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare LP bounds between source model and Cobre output.",
    )
    compare_subparsers = compare_parser.add_subparsers(
        dest="compare_source",
        metavar="SOURCE",
        required=True,
    )

    # compare bounds sub-subcommand
    compare_nw = compare_subparsers.add_parser(
        "bounds",
        parents=[common],
        help="Compare LP bounds computed from NEWAVE inputs against Cobre bounds.",
    )
    compare_nw.add_argument(
        "newave_dir",
        type=Path,
        help="Path to the NEWAVE case directory.",
    )
    compare_nw.add_argument(
        "cobre_output_dir",
        type=Path,
        help="Path to the Cobre output directory (contains bounds.parquet).",
    )
    compare_nw.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Absolute tolerance for bound comparison (default: 1e-3).",
    )
    compare_nw.add_argument(
        "--format",
        action="append",
        default=None,
        metavar="FORMAT",
        help=(
            "Output format(s): console,html,csv,parquet,json,all. "
            "Comma-separated and/or repeatable. (default: console,parquet,json)"
        ),
    )
    compare_nw.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for file artifacts "
            "(default: <cobre_output_dir>/comparison_artifacts)."
        ),
    )
    compare_nw.add_argument(
        "--summary",
        action="store_true",
        help="Print only summary counts, not individual mismatches.",
    )
    compare_nw.add_argument(
        "--variables",
        type=str,
        default=None,
        help="Comma-separated variables to compare (e.g., storage_min,turbined_max).",
    )

    # compare results sub-subcommand
    compare_res = compare_subparsers.add_parser(
        "results",
        parents=[common],
        help="Compare NEWAVE published results against Cobre simulation output.",
        epilog=(
            "compare results is informational and always exits 0; "
            "compare bounds exits 1 on any mismatch."
        ),
    )
    compare_res.add_argument(
        "newave_dir",
        type=Path,
        help="Path to the NEWAVE case directory (must contain saidas/).",
    )
    compare_res.add_argument(
        "cobre_output_dir",
        type=Path,
        help="Path to the Cobre output directory.",
    )
    compare_res.add_argument(
        "--format",
        action="append",
        default=None,
        metavar="FORMAT",
        help=(
            "Output format(s): console,html,csv,parquet,json,all. "
            "Comma-separated and/or repeatable. (default: console,parquet,json)"
        ),
    )
    compare_res.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Directory for file artifacts "
            "(default: <cobre_output_dir>/comparison_artifacts)."
        ),
    )
    compare_res.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Relative tolerance for results comparison (default: 1e-2).",
    )

    # dashboard subcommand
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        parents=[common],
        help="Generate an interactive HTML dashboard from Cobre simulation results.",
    )
    dashboard_parser.add_argument(
        "case_dir",
        type=Path,
        help="Path to the Cobre case directory.",
    )
    dashboard_parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output HTML file path (default: <case_dir>/dashboard.html).",
    )

    args = parser.parse_args()

    pkg_logger = logging.getLogger("cobre_bridge")
    prior_propagate = pkg_logger.propagate
    _configure_logging(getattr(args, "verbose", False))
    try:
        if args.command == "convert" and args.source == "newave":
            _run_newave_conversion(args)
            return

        if args.command == "compare" and args.compare_source == "bounds":
            _run_bounds_comparison(args)
            return

        if args.command == "compare" and args.compare_source == "results":
            _run_results_comparison(args)
            return

        if args.command == "dashboard":
            _run_dashboard(args)
            return
    finally:
        # Restore propagation so the package logger is left as we found it — the
        # handlers (e.g. SystemExit from a subcommand) run through here, and this
        # keeps an in-process CLI call from leaking ``propagate=False`` into a
        # later caplog-based test in the same interpreter.
        pkg_logger.propagate = prior_propagate

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
