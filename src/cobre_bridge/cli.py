"""Command-line interface entry point for cobre-bridge."""

import argparse
import sys

from cobre_bridge import __version__


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
        help="Path to the NEWAVE case directory.",
    )
    newave_parser.add_argument(
        "dst",
        metavar="DST",
        help="Path to the output Cobre case directory.",
    )

    args = parser.parse_args()

    if args.command == "convert" and args.source == "newave":
        print("Not implemented yet")
        sys.exit(0)

    parser.print_help()
    sys.exit(0)


if __name__ == "__main__":
    main()
