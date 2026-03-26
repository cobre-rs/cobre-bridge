"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cobre_bridge import __version__


def _run_newave_comparison(args: argparse.Namespace) -> None:
    """Execute the compare newave subcommand."""
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
    sintese_dir = newave_dir / "sintese"
    if not sintese_dir.is_dir():
        print(
            f"Error: sintese/ directory not found in {newave_dir}. "
            "Run sintetizador-newave first.",
            file=sys.stderr,
        )
        sys.exit(1)

    bounds_path = cobre_output_dir / "training" / "dictionaries" / "bounds.parquet"
    if not bounds_path.exists():
        print(
            f"Error: bounds.parquet not found at {bounds_path}. "
            "Run cobre with --output first.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Find the converted Cobre case directory (parent of output dir).
    # lines.json lives in the case dir, not the output dir.
    cobre_case_dir = cobre_output_dir.parent
    lines_path = cobre_case_dir / "system" / "lines.json"
    if not lines_path.exists():
        # Try output dir's parent's parent (in case output is nested).
        for candidate in [cobre_output_dir, cobre_output_dir.parent]:
            p = candidate / "system" / "lines.json"
            if p.exists():
                lines_path = p
                break

    lines_json: list[dict] = []
    if lines_path.exists():
        with lines_path.open() as f:
            lines_data = json.load(f)
        lines_json = lines_data.get("lines", [])

    # Build alignment.
    nw_files = NewaveFiles.from_directory(newave_dir)
    id_map = _build_id_map(nw_files)

    variables: set[str] | None = None
    if args.variables:
        variables = {v.strip() for v in args.variables.split(",")}

    alignment = build_entity_alignment(id_map, sintese_dir, lines_json)

    # Run comparison.
    results = compare_bounds(
        alignment=alignment,
        sintese_dir=sintese_dir,
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

    # compare newave sub-subcommand
    compare_nw = compare_subparsers.add_parser(
        "newave",
        help="Compare NEWAVE sintetizador bounds against Cobre bounds.",
    )
    compare_nw.add_argument(
        "newave_dir",
        type=Path,
        help="Path to the NEWAVE case directory (must contain sintese/).",
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

    if args.command == "compare" and args.compare_source == "newave":
        _run_newave_comparison(args)
        return

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
