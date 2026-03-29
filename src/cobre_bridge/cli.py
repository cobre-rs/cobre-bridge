"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cobre_bridge import __version__


def _load_lines_json(cobre_output_dir: Path) -> list[dict]:
    """Load lines.json from the Cobre case directory.

    Searches for ``system/lines.json`` near the output directory.
    Returns an empty list if not found.
    """
    cobre_case_dir = cobre_output_dir.parent
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


def _run_bounds_comparison(args: argparse.Namespace) -> None:
    """Execute the compare bounds subcommand."""
    from cobre_bridge.comparators.alignment import build_entity_alignment
    from cobre_bridge.comparators.bounds import compare_bounds
    from cobre_bridge.comparators.report import (
        build_summary,
        print_mismatches,
        print_summary,
        write_report_parquet,
    )
    from cobre_bridge.newave_files import NewaveFiles
    from cobre_bridge.pipeline import _build_id_map

    newave_dir: Path = args.newave_dir
    cobre_output_dir: Path = args.cobre_output_dir
    tolerance: float = args.tolerance

    # Validate paths.
    bounds_path = cobre_output_dir / "training" / "dictionaries" / "bounds.parquet"
    if not bounds_path.exists():
        print(
            f"Error: bounds.parquet not found at {bounds_path}. "
            "Run cobre with --output first.",
            file=sys.stderr,
        )
        sys.exit(1)

    lines_json = _load_lines_json(cobre_output_dir)

    # Build alignment.
    nw_files = NewaveFiles.from_directory(newave_dir)
    id_map = _build_id_map(nw_files)

    variables: set[str] | None = None
    if args.variables:
        variables = {v.strip() for v in args.variables.split(",")}

    alignment = build_entity_alignment(id_map, nw_files, lines_json)

    # Run comparison.
    results = compare_bounds(
        alignment=alignment,
        nw_files=nw_files,
        id_map=id_map,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
        variables=variables,
    )

    # Output.
    summary = build_summary(results)
    print_summary(summary, newave_dir, cobre_output_dir, tolerance)

    if not args.summary:
        print_mismatches(results)

    if args.output:
        write_report_parquet(results, args.output)

    sys.exit(0 if summary.mismatches == 0 else 1)


def _run_results_comparison(args: argparse.Namespace) -> None:
    """Execute the compare results subcommand."""
    from cobre_bridge.comparators.alignment import build_entity_alignment
    from cobre_bridge.comparators.report import print_results_summary
    from cobre_bridge.comparators.results import build_results_summary, compare_results
    from cobre_bridge.newave_files import NewaveFiles
    from cobre_bridge.pipeline import _build_id_map

    newave_dir: Path = args.newave_dir
    cobre_output_dir: Path = args.cobre_output_dir
    tolerance: float = args.tolerance

    # Build alignment.
    try:
        nw_files = NewaveFiles.from_directory(newave_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    id_map = _build_id_map(nw_files)
    lines_json = _load_lines_json(cobre_output_dir)
    alignment = build_entity_alignment(id_map, nw_files, lines_json)

    # Run comparison.
    results = compare_results(
        nw_files=nw_files,
        id_map=id_map,
        alignment=alignment,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
    )

    # Print text summary.
    summary = build_results_summary(results)
    print_results_summary(summary, newave_dir, cobre_output_dir)

    # HTML report.
    if args.output:
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )

        html = build_comparison_report(results)
        output_path: Path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        print(f"HTML report written to {output_path}")

    sys.exit(0)


def _run_newave_conversion(args: argparse.Namespace) -> None:
    """Execute the convert newave subcommand."""
    # Import here so the module-level import of pipeline is deferred.
    from cobre_bridge.pipeline import (
        ConversionReport,
        _clear_dst_contents,
        convert_newave_case,
    )

    src: Path = args.src
    dst: Path = args.dst

    # ------------------------------------------------------------------
    # Source validation.
    # ------------------------------------------------------------------
    if not src.exists() or not src.is_dir():
        print(
            f"Error: source directory '{src}' does not exist",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Destination validation.
    # ------------------------------------------------------------------
    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            print(
                f"Error: destination directory '{dst}' is not empty."
                " Use --force to overwrite.",
                file=sys.stderr,
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
        print(
            f"Error: required file '{missing}' not found in {src}",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: conversion failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(str(report))

    if report.warnings:
        for warning in report.warnings:
            print(f"Warning: {warning}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Optional post-conversion validation.
    # ------------------------------------------------------------------
    if args.validate:
        try:
            import cobre.io  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            print(
                "Warning: cobre package not installed, skipping validation",
                file=sys.stderr,
            )
            sys.exit(0)

        try:
            import cobre.io.validate as cobre_validate  # type: ignore[import-untyped]

            result = cobre_validate.validate(dst)
            if result is not None and not result:
                print("Validation failed.", file=sys.stderr)
                sys.exit(2)
        except Exception as exc:  # noqa: BLE001
            print(f"Validation error: {exc}", file=sys.stderr)
            sys.exit(2)

    sys.exit(0)


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

    # convert subcommand
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert data from a source format to Cobre JSON.",
    )
    convert_subparsers = convert_parser.add_subparsers(
        dest="source",
        metavar="SOURCE",
    )

    # convert newave sub-subcommand
    newave_parser = convert_subparsers.add_parser(
        "newave",
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
        "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    # compare subcommand
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare LP bounds between source model and Cobre output.",
    )
    compare_subparsers = compare_parser.add_subparsers(
        dest="compare_source",
        metavar="SOURCE",
    )

    # compare bounds sub-subcommand
    compare_nw = compare_subparsers.add_parser(
        "bounds",
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
        "--output",
        type=Path,
        default=None,
        help="Write detailed diff report as a Parquet file.",
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
    compare_nw.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    # compare results sub-subcommand
    compare_res = compare_subparsers.add_parser(
        "results",
        help="Compare NEWAVE published results against Cobre simulation output.",
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
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path for HTML comparison report.",
    )
    compare_res.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Relative tolerance for results comparison (default: 1e-2).",
    )
    compare_res.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging output.",
    )

    args = parser.parse_args()

    # Configure logging based on --verbose.
    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.command == "convert" and args.source == "newave":
        _run_newave_conversion(args)
        return

    if args.command == "compare" and args.compare_source == "bounds":
        _run_bounds_comparison(args)
        return

    if args.command == "compare" and args.compare_source == "results":
        _run_results_comparison(args)
        return

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
