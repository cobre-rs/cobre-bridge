"""Command-line interface entry point for cobre-bridge."""

from __future__ import annotations

import json
import logging
import sys
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated

import typer

from cobre_bridge import __version__
from cobre_bridge.cobre_io import case_dir_for
from cobre_bridge.config_resolution import (
    RESULTS_TOLERANCE_DEFAULT,
    load_config,
)
from cobre_bridge.diagnostics import Severity
from cobre_bridge.errors import diagnostic_from_exception
from cobre_bridge.preflight import PreflightVerdict
from cobre_bridge.ui.console import (
    conversion_progress,
    get_console,
    make_table,
    print_status,
    render_checklist,
    render_conversion_summary,
    render_decomp_comparison,
    render_diagnostics,
    render_error,
    spinner,
)
from cobre_bridge.verdict import (
    build_verdict,
    check_summary,
    compare_summary,
    convert_summary,
    dashboard_summary,
    decomp_compare_summary,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import Console

    from cobre_bridge.case import NewaveCase
    from cobre_bridge.comparators.alignment import EntityAlignment
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.diagnostics import Diagnostic
    from cobre_bridge.id_map import NewaveIdMap
    from cobre_bridge.pipeline import ConversionReport


#: Minimum cobre / cobre-python version that can load the bridge's converted
#: output. Since cobre 0.12.0 the converted ``config.json`` carries a
#: ``training.parallelism.backward_scheduler`` block (an opening-block scheduler,
#: new in 0.12.0); under the config's strict unknown-key rejection an older cobre
#: rejects it outright. (Since 0.10.0 every ``system/*.json`` entity also carries
#: a required ``operational_start_date``.) Since cobre 0.13.0 every hydro also
#: carries a mandatory ``unit_groups`` array and no longer accepts the removed
#: top-level ``bus_id`` (cobre decisions 13/14), and the windowed
#: ``inflow_history`` the bridge now emits also targets 0.13. So *all* converted
#: cases require cobre >= this version. The manifest records it (single source
#: of truth) and the ``--validate`` gate uses it to decide whether the installed
#: cobre-python is new enough to validate the output.
#:
#: PENDING BUMP: the bridge now targets three cobre input-contract changes that
#: all ride cobre's unreleased ``feat/generic-constraint-authoring`` branch (the
#: next tagged release, expected 0.14):
#:   1. **F3** sense-free generic constraints — ``generic_constraints.json`` +
#:      ``generic_constraint_bounds.parquet`` carry no ``sense`` key and use
#:      ``bound_lower``/``bound_upper`` in place of a single ``bound`` column
#:      (see ``cobre_bridge.generic_constraint_format``).
#:   2. the scalar-parameters input moved from ``system/scalar_parameters.json``
#:      to ``constraints/generic_parameters.json`` (a clean break — cobre rejects
#:      the old path); the JSON shape is unchanged.
#:   3. the NCS availability column is now ``availability_factor`` only (cobre's
#:      clean break dropped the legacy ``value`` alias).
#: None of these has been tagged/released yet, so there is no released version
#: number to pin to; the installed editable ``cobre-python`` still reports
#: ``0.13.0``, which is why this constant is unchanged for now. Bump
#: ``MIN_COBRE_VERSION`` (and the ``cobre-python`` pin in ``pyproject.toml``) to
#: the real release version as soon as cobre tags one — do not guess a number in
#: the meantime.
MIN_COBRE_VERSION = "0.13.0"


def _installed_cobre_python_version() -> str | None:
    """Return the installed ``cobre-python`` distribution version, or ``None``.

    The package imports as ``cobre`` but is distributed as ``cobre-python``;
    this reads the distribution metadata. Returns ``None`` when it is not
    installed, so the caller falls through to the generic "not installed" skip.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version("cobre-python")
    except PackageNotFoundError:
        return None


def _cobre_python_supports_output(installed: str) -> bool:
    """Whether an installed cobre-python *version* can load the bridge's output.

    ``True`` when *installed* is at least :data:`MIN_COBRE_VERSION` by a numeric
    release-segment comparison (so ``"0.10.0"`` and ``"0.11.2"`` qualify,
    ``"0.9.1"`` does not). A non-numeric pre-release suffix on a segment is
    ignored (``"0.10.0rc1"`` reads as ``0.10.0``); the gate only guards against an
    obviously-older install, so the leniency is deliberate.
    """

    def _release(value: str) -> tuple[int, ...]:
        parts: list[int] = []
        for segment in value.split("."):
            digits = ""
            for char in segment:
                if not char.isdigit():
                    break
                digits += char
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    return _release(installed) >= _release(MIN_COBRE_VERSION)


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

    Setup for `compare newave`. Builds the parsed
    case once (so the id-map reuses its cached readers), loads lines.json, and
    builds the entity alignment.

    Exits the process with code 1 (clean stderr message) if the source model case
    directory is missing (FileNotFoundError from NewaveCase.from_directory).
    """
    from cobre_bridge.case import NewaveCase
    from cobre_bridge.comparators.alignment import build_entity_alignment

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
    quiet_status: bool = False,
) -> tuple[set[str], Path]:
    """Resolve ``--format`` and write the machine-readable comparison artifacts.

    Used by `compare newave`. Returns the requested
    formats and the resolved out_dir so the handler can run its own HTML branch
    and exit-code logic.

    An invalid ``--format`` token exits 2 (clean stderr). A write failure must
    NOT change the comparison exit code, so an ``OSError`` is warned and
    swallowed.

    *quiet_status* (set by ``--json``) gates ONLY the ``Artifacts written to …``
    stdout status line so stdout stays pure JSON; the file export still runs and
    the ``OSError`` write-failure warning still reaches stderr.
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
        if not quiet_status:
            print_status(f"Artifacts written to {out_dir}")
    except OSError as exc:
        print_status(
            f"Warning: failed to write artifacts: {exc}",
            console=get_console(stderr=True),
            style="#F5A623",
        )

    return formats, out_dir


def _resolve_compare_settings(args: SimpleNamespace) -> None:
    """Fill tolerance/format/out_dir from config when flag+env left them ``None``.

    Implements the bottom two rungs of the precedence chain
    **CLI flag > env var > config file > built-in default**: Typer has already
    resolved flag-or-env into ``args`` (a non-``None`` value means the flag or
    env var supplied it), so this fills in the config-file value, then the
    built-in default, for whatever is still ``None``. Mutates *args* in place.

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
    args.tolerance = resolved_tolerance

    # Format: only consult config when neither flag nor env supplied it. Leave
    # ``None`` (rather than pre-expanding) so ``_parse_formats`` stays the single
    # validator/expander downstream.
    if args.format is None and cfg.formats is not None:
        args.format = list(cfg.formats)

    # Out-dir: only consult config when neither flag nor env supplied it. The
    # config value may itself be ``None``, which keeps the derived
    # ``<cobre_output_dir>/comparison_artifacts`` default downstream.
    if args.out_dir is None:
        args.out_dir = cfg.out_dir

    # Surface any config-load warnings on stderr only (never stdout), matching
    # the side-file advisory pattern used by ``_export_compare_artifacts``.
    for warning in cfg.warnings:
        print_status(
            warning,
            console=get_console(stderr=True),
            style="#F5A623",
        )


def _run_newave_comparison(args: SimpleNamespace) -> None:
    """Execute the compare newave subcommand.

    Intentionally always exits 0: ``compare newave`` is informational (a
    descriptive NEWAVE-vs-Cobre divergence report), so it never signals a
    failure on divergence.
    """
    _resolve_compare_settings(args)

    from cobre_bridge.comparators.cobre_readers import CobreReadError
    from cobre_bridge.comparators.report import print_results_summary_from_dataset
    from cobre_bridge.comparators.results import compare_results
    from cobre_bridge.comparators.verdict import build_compare_verdict
    from cobre_bridge.errors import CobrePartitionMissingError

    newave_dir: Path = args.newave_dir
    cobre_output_dir: Path = args.cobre_output_dir
    tolerance: float = args.tolerance

    case, id_map, alignment, _lines_json = _load_compare_context(
        newave_dir, cobre_output_dir
    )

    # Run comparison. A CobreReadError means an *existing* Cobre output file
    # was unreadable/malformed; a CobrePartitionMissingError means the output
    # predates the cobre version that introduced a partition this compare
    # relies on (e.g. simulation/hydro_bus_generation/, cobre >= 0.13.0).
    # CobrePartitionMissingError extends BridgeError, a hierarchy disjoint
    # from CobreReadError (RuntimeError) — both must be caught here, or the
    # latter crashes with an unhandled traceback instead of failing loudly
    # (exit 2) like the former.
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
        print_status(
            f"ERROR: {exc}", console=get_console(stderr=True), style="bold #DC4C4C"
        )
        raise typer.Exit(code=2)

    # Print text summary (sourced from the dataset). Under --json the Rich tables
    # are suppressed in favour of a single machine-readable verdict on stdout.
    if not args.json_output:
        print_results_summary_from_dataset(dataset, newave_dir, cobre_output_dir)

    formats, out_dir = _export_compare_artifacts(
        dataset,
        command="compare newave",
        raw_formats=args.format,
        newave_dir=newave_dir,
        cobre_output_dir=cobre_output_dir,
        tolerance=tolerance,
        out_dir_arg=args.out_dir,
        quiet_status=args.json_output,
    )

    # HTML report (opt-in via --format html / all). The file is still written
    # under --json (it is a --format artifact); only its stdout advisory is
    # routed to stderr so stdout stays pure JSON.
    if "html" in formats:
        from cobre_bridge.comparators.report_builder import (
            build_comparison_report,
        )

        html = build_comparison_report(dataset)
        report_path = out_dir / "report.html"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(html, encoding="utf-8")
        print_status(
            f"HTML report written to {report_path}",
            console=get_console(stderr=True) if args.json_output else None,
        )

    if args.json_output:
        # ``status`` REFLECTS divergence (so it is self-consistent with
        # ``summary``), but the results exit
        # code is DECOUPLED from it — this command always exits 0. An empty
        # dataset → ``all_within_tol`` False → ``status`` "mismatch".
        verdict = build_compare_verdict(dataset)
        status = "mismatch" if not verdict.all_within_tol else "ok"
        _emit_convert_json(
            build_verdict("compare newave", status, compare_summary(verdict))
        )


def _run_dashboard(args: SimpleNamespace) -> None:
    """Execute the dashboard subcommand."""
    from cobre_bridge.dashboard import build_dashboard

    case_dir: Path = args.case_dir.resolve()
    if not (case_dir / "output" / "simulation").exists():
        render_error(f"no simulation output found in {case_dir}")
        raise typer.Exit(code=1)

    output_path: Path = args.output or (case_dir / "dashboard.html")
    if not args.json_output:
        print_status(f"Building dashboard from {case_dir} ...")
    with spinner(
        "Building dashboard…",
        verbose=args.verbose > 0,
        quiet=args.quiet,
        no_color=args.no_color,
    ):
        build_dashboard(case_dir, output_path)
    size_kb = output_path.stat().st_size / 1024
    if not args.json_output:
        print_status(f"Dashboard written to {output_path} ({size_kb:.0f} KB)")

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
            )
        )

    # The --open advisory stays off the stdout --json payload; it already goes to
    # stderr (err_console below), so the two compose cleanly.
    if args.open_browser:
        err_console = get_console(stderr=True, no_color=args.no_color)
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


#: Preflight verdict → process exit code. The contract is fixed by the epic
#: overview: ``OK`` is clean (0), ``WARNINGS`` is advisory (1), and
#: ``WILL_NOT_CONVERT`` is the most severe (2). Kept as data so the mapping is
#: directly unit-testable and impossible to drift from in the handler.
_VERDICT_EXIT_CODE: dict[PreflightVerdict, int] = {
    PreflightVerdict.OK: 0,
    PreflightVerdict.WARNINGS: 1,
    PreflightVerdict.WILL_NOT_CONVERT: 2,
}


def _run_decomp_check(args: SimpleNamespace) -> None:
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
        render_checklist(result, quiet=args.quiet)

    exit_code = _VERDICT_EXIT_CODE[result.verdict]
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


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
        # Default consoles: the ✓/✗ checklist goes to stdout while render_checklist
        # delegates its diagnostics block to render_diagnostics (stderr default),
        # preserving the stdout(results)/stderr(diagnostics) split.
        render_checklist(result, quiet=args.quiet)

    exit_code = _VERDICT_EXIT_CODE[result.verdict]
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _validation_message(item: object) -> str:
    """Extract the display text from a ``cobre.io.validate`` warning/error item.

    Each item is either a plain string or a ``{"message": ...}`` dict; both
    forms render and partition on the same text.
    """
    text = item.get("message", item) if isinstance(item, dict) else item
    return str(text)


def _partition_validation_warnings(
    warnings: Sequence[object], whitelist_substrings: Sequence[str] = ()
) -> tuple[list[object], list[object]]:
    """Split ``cobre.io.validate`` ``warnings`` into ``(rendered, whitelisted)``.

    A warning whose message (:func:`_validation_message`) contains any of
    *whitelist_substrings* is whitelisted — not rendered as a "Validation
    warning" and not counted in the rendered/blocking warning count; every
    other warning goes to *rendered* exactly as before. An empty
    *whitelist_substrings* — what ``convert newave`` passes — is the identity
    partition: ``rendered == warnings`` and ``whitelisted == []``. Pure; never
    touches ``errors``, and never suppresses a non-matching warning.
    """
    if not whitelist_substrings:
        return list(warnings), []

    rendered: list[object] = []
    whitelisted: list[object] = []
    for warning in warnings:
        message = _validation_message(warning)
        if any(substring in message for substring in whitelist_substrings):
            whitelisted.append(warning)
        else:
            rendered.append(warning)
    return rendered, whitelisted


def _run_cobre_validation(
    dst: Path,
    *,
    command: str,
    summary: dict[str, object],
    json_output: bool,
    err_console: Console,
    whitelist_substrings: Sequence[str] = (),
) -> bool:
    """Validate *dst* with the installed cobre-python and render the outcome.

    Shared by every ``convert *`` command's ``--validate`` gate: the
    :data:`MIN_COBRE_VERSION` skip, the ``cobre.io.validate`` call,
    warning/error rendering (warnings are first partitioned through
    :func:`_partition_validation_warnings` against *whitelist_substrings* —
    ``convert newave`` passes an empty tuple, the identity case, so its
    rendering stays byte-identical), and the machine-readable
    ``summary["validation"]`` sub-object, populated only when *json_output*
    is set. *command* names the caller in the whitelisted-note message;
    *summary* is mutated in place.

    Returns whether validation FAILED (``valid`` came back ``False``, or
    ``cobre.io.validate`` itself raised) so the caller can flip its exit code
    to 2 — a skipped validation (old/absent cobre-python) is never a failure.
    Does not emit the enclosing ``--json`` verdict or raise ``typer.Exit``;
    that stays the caller's job, run immediately after this returns.
    """
    installed = _installed_cobre_python_version()
    if installed is not None and not _cobre_python_supports_output(installed):
        print_status(
            f"Note: converted output requires cobre-python >= "
            f"{MIN_COBRE_VERSION} (installed cobre-python {installed} is "
            f"older); skipping cobre-python validation.",
            console=err_console,
            style="#F5A623",
        )
        if json_output:
            summary["validation"] = {
                "ran": False,
                "valid": None,
                "warnings": 0,
                "errors": 0,
                "skipped_reason": "cobre-python-too-old",
            }
        return False

    try:
        import cobre.io  # type: ignore[import-untyped]
    except ImportError:
        print_status(
            "Warning: cobre package not installed, skipping validation",
            console=err_console,
            style="#F5A623",
        )
        if json_output:
            # Validation was requested but could not run; record that it was
            # skipped so the absence of a real outcome is explicit.
            summary["validation"] = {
                "ran": False,
                "valid": None,
                "warnings": 0,
                "errors": 0,
            }
        return False

    try:
        # cobre v0.6.x: cobre.io.validate is a function returning a
        # report dict; it never raises (errors are surfaced as data).
        result = cobre.io.validate(str(dst))
    except Exception as exc:  # noqa: BLE001
        render_error(f"Validation error: {exc}", console=err_console)
        if json_output:
            # Validation raised unexpectedly; still emit one JSON object so
            # the --json contract (exactly one verdict on stdout) holds on
            # this exit-2 path too. The conversion itself succeeded, so the
            # summary is intact; only the validation outcome is an error.
            summary["validation"] = {
                "ran": False,
                "valid": None,
                "warnings": 0,
                "errors": 1,
            }
        return True

    raw_warnings = result.get("warnings", [])
    errors = result.get("errors", [])
    valid = bool(result.get("valid", False))

    rendered_warnings, whitelisted_warnings = _partition_validation_warnings(
        raw_warnings, whitelist_substrings
    )

    for warning in rendered_warnings:
        print_status(
            f"Validation warning: {_validation_message(warning)}",
            console=err_console,
            style="#F5A623",
        )
    if whitelisted_warnings:
        # Whitelisted-but-present is worth one INFO note, never a WARNING —
        # this is what tells the whitelist apart from a real suppression.
        print_status(
            f"Note: {len(whitelisted_warnings)} validation warning(s) matched "
            "the expected external-solver-interop configuration for "
            f"{command}; not rendered.",
            console=err_console,
        )

    validation_failed = False
    if not valid:
        for err in errors:
            print_status(
                f"Validation error: {_validation_message(err)}",
                console=err_console,
                style="bold #DC4C4C",
            )
        print_status("Validation failed.", console=err_console, style="bold #DC4C4C")
        validation_failed = True

    if json_output:
        # The machine-readable outcome under ``summary``; ``status`` stays
        # derived from diagnostics only (validation never flips it).
        summary["validation"] = {
            "ran": True,
            "valid": valid,
            "warnings": len(rendered_warnings),
            "errors": len(errors),
        }

    return validation_failed


def _run_newave_conversion(args: SimpleNamespace) -> None:
    """Execute the convert newave subcommand."""
    from cobre_bridge.pipeline import (
        CONVERSION_PHASE_LABELS,
        _clear_dst_contents,
        convert_newave_case,
    )

    src: Path = args.src
    dst: Path = args.dst

    out_console = get_console(no_color=args.no_color)
    err_console = get_console(stderr=True, no_color=args.no_color)

    if not src.exists() or not src.is_dir():
        render_error(f"source directory '{src}' does not exist", console=err_console)
        raise typer.Exit(code=1)

    if dst.exists() and any(dst.iterdir()):
        if not args.force:
            render_error(
                f"destination directory '{dst}' is not empty."
                " Use --force to overwrite.",
                console=err_console,
            )
            raise typer.Exit(code=1)
        # --force: remove previous pipeline outputs before converting. A dry run
        # never mutates the destination, so the clear is skipped even with --force.
        if not args.dry_run:
            _clear_dst_contents(dst)

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
        diag = diagnostic_from_exception(exc, context="Conversion")
        if args.json_output:
            # Pipeline failure: ``report`` is None, so counts are zeroed and the
            # dry-run path has an empty would-write listing. ``status`` is "error"
            # because ``diagnostic_from_exception`` yields an ERROR-severity diag.
            diagnostics = [diag]
            summary = _convert_verdict_summary(None)
            if args.dry_run:
                summary["would_write"] = []
                status = _convert_status(diagnostics, success="dry-run")
            else:
                status = _convert_status(diagnostics, success="ok")
            _emit_convert_json(
                build_verdict("convert newave", status, summary, diagnostics)
            )
        else:
            render_diagnostics([diag], console=err_console, quiet=args.quiet)
        raise typer.Exit(code=1)

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
    _write_conversion_manifest(report, src, dst, console=err_console)

    # When --validate and --json are combined, the human validation messages stay
    # on err_console (stderr) exactly as without --json, and the machine-readable
    # outcome is folded UNDER ``summary`` as a ``validation`` sub-object (mutated
    # in place by the shared helper below). The verdict is emitted (below) only
    # AFTER this block so ``validation`` is populated; validation failure flips
    # the exit code, never the status. ``convert newave`` passes an empty
    # whitelist, so ``_run_cobre_validation`` is the identity case here and its
    # rendering/``--json`` shape stay byte-identical to before the extraction.
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

    if validation_failed:
        raise typer.Exit(code=2)


def _convert_verdict_summary(report: ConversionReport | None) -> dict[str, object]:
    """The convert ``summary`` block — entity counts, zeroed when *report* is None.

    A thin wrapper over :func:`cobre_bridge.verdict.convert_summary` that supplies
    the five counts from a :class:`ConversionReport` (or all zeros on the failure
    path, where ``report`` is ``None``). Keeping the count plumbing here lets the
    real-run, failure, and dry-run call sites share one source of truth while the
    key order itself stays owned by ``verdict.convert_summary``.
    """
    if report is None:
        return convert_summary(0, 0, 0, 0, 0)
    return convert_summary(
        report.hydro_count,
        report.thermal_count,
        report.bus_count,
        report.line_count,
        report.stage_count,
    )


def _convert_status(diagnostics: Sequence[Diagnostic], *, success: str) -> str:
    """Derive the convert verdict ``status`` from diagnostic severity ONLY.

    Returns ``"error"`` when any diagnostic has ``ERROR`` severity, otherwise the
    caller's *success* token (``"ok"`` for a real run, ``"dry-run"`` for a dry
    run). Validation outcome never enters here — it lands in ``summary.validation``
    and the exit code, keeping ``status`` a pure diagnostics signal.
    """
    if any(d.severity is Severity.ERROR for d in diagnostics):
        return "error"
    return success


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
    # Record the minimum cobre version the output requires. Every converted case
    # now emits a ``training.parallelism.backward_scheduler`` block (cobre 0.12.0+),
    # ``operational_start_date`` on all system entities (cobre 0.10.0+), and a
    # mandatory hydro ``unit_groups`` array with the top-level ``bus_id`` removed
    # (cobre 0.13.0+), so the output is only loadable by cobre >= MIN_COBRE_VERSION.
    manifest = ConversionManifest.create(
        "convert newave",
        src,
        dst,
        entity_counts=entity_counts,
        input_files=hash_input_files(files),
        diagnostics_summary=summarize_diagnostics(report.diagnostics),
        diagnostics=[d.to_dict() for d in report.diagnostics],
        min_cobre_version=MIN_COBRE_VERSION,
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

#: The ``--log-file`` DEBUG ``FileHandler`` attached to the package logger for the
#: duration of a run, or ``None`` when no ``--log-file`` was given. ``main`` removes
#: and closes it in its ``finally`` so it never leaks across in-process invocations
#: (the autouse test fixture restores ``propagate``/level but NOT handlers).
_LOG_FILE_HANDLER: logging.FileHandler | None = None


def _configure_logging(verbose: int, log_file: Path | None) -> None:
    """Configure logging for a CLI run.

    *verbose* is a graduated count selecting the live console level:
    ``0`` keeps ``cobre_bridge`` warnings recorded — the diagnostics collector and
    ``--diagnostics-json`` rely on them — but off the live console (the Rich
    diagnostics block is the single user-facing surface, and warnings are not
    printed twice); ``1`` (``-v`` / ``--verbose``) raises the console to INFO; and
    ``2`` or more (``-vv``) raises it to DEBUG.

    This is a behavior change from the previous boolean ``--verbose``: a bare
    ``--verbose`` used to mean DEBUG and now means INFO; ``-vv`` is required for the
    full DEBUG firehose. ``--log-file`` (below) gives the complete DEBUG trace to
    anyone who needs it regardless of the console level.

    When *log_file* is not ``None``, a DEBUG ``FileHandler`` is attached to the
    ``cobre_bridge`` logger and the package logger level is lowered to DEBUG, so the
    file always captures the full trace even at console verbose ``0`` (the console
    output stays at the ladder level — it is driven by ``basicConfig``/root, while
    ``_NULL_HANDLER`` keeps suppressed records off ``logging.lastResort``). The
    created handler is stored in the module-level ``_LOG_FILE_HANDLER`` so ``main``
    can remove and close it; ``main`` also restores ``propagate`` afterwards.
    """
    global _LOG_FILE_HANDLER

    pkg = logging.getLogger("cobre_bridge")
    if verbose >= 1:
        level = logging.DEBUG if verbose >= 2 else logging.INFO
        logging.basicConfig(
            level=level,
            format="%(levelname)s %(name)s: %(message)s",
        )
        pkg.setLevel(level)
        pkg.propagate = True
        # Symmetric with the suppress branch below: drop the null handler so a
        # verbose run after a suppressed one in the same process leaves the
        # package logger in a clean state (removeHandler is a no-op if absent).
        pkg.removeHandler(_NULL_HANDLER)
    else:
        # Leave the package logger level untouched (root's default WARNING already
        # records warnings for the collector); just keep them off the live console.
        if _NULL_HANDLER not in pkg.handlers:
            pkg.addHandler(_NULL_HANDLER)
        pkg.propagate = False

    if log_file is not None:
        # An unwritable path raises OSError here; it is deliberately not swallowed —
        # an unwritable --log-file is a user error that should fail loudly through
        # the per-command CLI boundary.
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        pkg.addHandler(handler)
        # Lower the logger threshold so DEBUG records reach the file handler even
        # when the console ladder leaves it at verbose 0.
        pkg.setLevel(logging.DEBUG)
        _LOG_FILE_HANDLER = handler


# ---------------------------------------------------------------------------
# Typer application
# ---------------------------------------------------------------------------

# Shared flags reused across the leaf commands (kept *after* the subcommand,
# matching the previous argparse UX).
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
        SimpleNamespace(
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


#: Warning substring that marks the P3 lag-blind stage shape
#: (``state_variables.inflow_lags = false`` on every stage, alongside a
#: positive ``inflow_lag_depth`` — see ``decomp/temporal.py::stage_records``)
#: as deliberate external-solver interoperability, not a misconfiguration.
#: Matched via :func:`_partition_validation_warnings` against cobre's stable
#: substring (never the volatile message prefix); see
#: ``cobre-io/src/validation/semantic/stages.rs``.
_DECOMP_VALIDATION_WHITELIST: tuple[str, ...] = ("external-solver interoperability",)


def _run_decomp_conversion(args: SimpleNamespace) -> None:
    """Execute the convert decomp subcommand.

    Structurally mirrors ``_run_newave_conversion``, minus the dry-run and
    provenance-manifest blocks (out of scope for this ticket; see
    epic-03/ticket-006): a TTY phase bar over the DECOMP phases, the
    ``✓ Converted …`` summary, grouped diagnostic panels, and a unified
    ``--json`` verdict.

    A broad ``except Exception`` (rather than a fixed exception tuple) also
    covers an ERROR-severity post-emission self-check finding (cobre rules
    43/41/45/38/36 + the block_id-range rule) — ``convert_decomp_case`` raises a
    ``ValueError`` naming the failing rule(s) and entities, mapped to exit 1
    like every other conversion failure.

    ``--validate`` runs after a successful conversion via the shared
    ``_run_cobre_validation`` helper (mirroring ``convert newave``), with the
    DECOMP external-solver-interop whitelist so the deliberate
    ``inflow_lags=false`` shape never surfaces as a scary warning; a failed
    validation exits 2, giving ``convert decomp`` the same 0/1/2 exit-code
    set as ``convert newave``.
    """
    from cobre_bridge.decomp.pipeline import (
        DECOMP_CONVERSION_PHASE_LABELS,
        convert_decomp_case,
    )

    out_console = get_console(no_color=args.no_color)
    err_console = get_console(stderr=True, no_color=args.no_color)

    try:
        with conversion_progress(
            len(DECOMP_CONVERSION_PHASE_LABELS),
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ) as step:
            report: ConversionReport = convert_decomp_case(
                args.src, args.dst, force=args.force, on_phase=step
            )
    except Exception as exc:  # noqa: BLE001
        diag = diagnostic_from_exception(exc, context="Conversion")
        if args.json_output:
            _emit_convert_json(
                build_verdict(
                    "convert decomp",
                    _convert_status([diag], success="ok"),
                    _convert_verdict_summary(None),
                    [diag],
                )
            )
        else:
            render_diagnostics([diag], console=err_console, quiet=args.quiet)
        raise typer.Exit(code=1)

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

    # Emit the --json verdict now (after validation has populated ``summary``).
    if args.json_output:
        _emit_convert_json(
            build_verdict("convert decomp", status, summary, report.diagnostics)
        )

    if validation_failed:
        raise typer.Exit(code=2)


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
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Convert a DECOMP deck revision to a Cobre case directory.

    Loop-closing subset: the exchange network, renewables card file, GNL
    anticipation and boundary FCF are deferred and reported as warnings.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_conversion(
        SimpleNamespace(
            src=src,
            dst=dst,
            force=force,
            validate=validate,
            diagnostics_json=diagnostics_json,
            json_output=json_output,
            verbose=verbose,
            log_file=log_file,
            no_color=no_color,
            quiet=quiet,
        )
    )


def _run_decomp_comparison(args: SimpleNamespace) -> None:
    """Execute the compare decomp subcommand.

    Informational like ``compare newave``: it always exits 0 and describes the
    divergence rather than judging it. An unreadable *existing* output file is
    the one failure (exit 2) — reporting a zero-vs-zero match on data we could
    not read would be worse than stopping.
    """
    from cobre_bridge.comparators.cobre_readers import CobreReadError
    from cobre_bridge.comparators.decomp_results import compare_decomp_results
    from cobre_bridge.errors import CobrePartitionMissingError

    try:
        with spinner(
            "Comparing results…",
            verbose=args.verbose > 0,
            quiet=args.quiet,
            no_color=args.no_color,
        ):
            comparison = compare_decomp_results(args.decomp_dir, args.cobre_output_dir)
    except (
        CobreReadError,
        CobrePartitionMissingError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        print_status(
            f"ERROR: {exc}", console=get_console(stderr=True), style="bold #DC4C4C"
        )
        raise typer.Exit(code=2) from exc

    if args.json_output:
        status = "ok" if not comparison.rows.is_empty() else "no-comparable-rows"
        _emit_convert_json(
            build_verdict(
                "compare decomp",
                status,
                decomp_compare_summary(comparison),
            )
        )
        return

    render_decomp_comparison(comparison)


@compare_app.command("decomp")
def _compare_decomp(
    decomp_dir: Annotated[
        Path, typer.Argument(help="Path to the DECOMP deck directory (has saidas/).")
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a single machine-readable JSON verdict to stdout and "
                "suppress the human (Rich) tables."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Compare a DECOMP run's published operation against Cobre's simulation.

    Informational: always exits 0, reporting divergences without failing.
    """
    _configure_logging(verbose, log_file)
    _run_decomp_comparison(
        SimpleNamespace(
            decomp_dir=decomp_dir,
            cobre_output_dir=cobre_output_dir,
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
        Path, typer.Argument(help="Path to the NEWAVE case directory (has saidas/).")
    ],
    cobre_output_dir: Annotated[
        Path, typer.Argument(help="Path to the Cobre output directory.")
    ],
    tolerance: Annotated[
        float | None,
        typer.Option(
            envvar="COBRE_BRIDGE_RESULTS_TOLERANCE",
            help=(
                "Relative tolerance for results comparison (default 1e-2; "
                "overridable via COBRE_BRIDGE_RESULTS_TOLERANCE or cobre-bridge.toml)."
            ),
        ),
    ] = None,
    fmt: _FormatOpt = None,
    out_dir: _OutDirOpt = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a single machine-readable JSON verdict to stdout and "
                "suppress the human (Rich) tables."
            ),
        ),
    ] = False,
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
        SimpleNamespace(
            newave_dir=newave_dir,
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
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Validate a NEWAVE case directory without converting or writing any files."""
    _configure_logging(verbose, log_file)
    _run_check(
        SimpleNamespace(
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
        SimpleNamespace(
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
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help=(
                "Emit a single machine-readable JSON verdict to stdout and "
                "suppress the human (Rich) status lines."
            ),
        ),
    ] = False,
    verbose: _VerboseOpt = 0,
    log_file: _LogFileOpt = None,
    no_color: _NoColorOpt = False,
    quiet: _QuietOpt = False,
) -> None:
    """Generate an interactive HTML dashboard from Cobre simulation results."""
    _configure_logging(verbose, log_file)
    _run_dashboard(
        SimpleNamespace(
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
    global _LOG_FILE_HANDLER

    pkg_logger = logging.getLogger("cobre_bridge")
    prior_propagate = pkg_logger.propagate
    try:
        app()
    finally:
        pkg_logger.propagate = prior_propagate
        if _LOG_FILE_HANDLER is not None:
            pkg_logger.removeHandler(_LOG_FILE_HANDLER)
            _LOG_FILE_HANDLER.close()
            _LOG_FILE_HANDLER = None


if __name__ == "__main__":
    main()
