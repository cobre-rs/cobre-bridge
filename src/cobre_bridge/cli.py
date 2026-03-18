"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from cobre_bridge import __version__


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

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
