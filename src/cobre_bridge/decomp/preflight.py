"""``check decomp`` — validate a deck revision without converting it.

Answers one question cheaply: *would this deck convert, and what will the
conversion silently leave behind?* The checks are the subset of the
cross-input validation matrix that can be decided from the deck alone, so a
malformed calendar or an unusable scenario tree fails in milliseconds with
deck-side context instead of surfacing later as a solver diagnostic.

The deferred-feature inventory is deliberately part of the report. A deck
carrying anticipation records or a plant with per-group availability converts
fine today — but not *completely*, and the difference is invisible unless the
preflight names it.

The solver remains the authority on the converted case; this is a courtesy
mirror, never a substitute.
"""

from __future__ import annotations

from pathlib import Path

from cobre_bridge.diagnostics import Diagnostic, Severity
from cobre_bridge.errors import diagnostic_from_exception
from cobre_bridge.preflight import CheckItem, PreflightResult, PreflightVerdict

_CONTEXT = "Preflight"

#: Per-stage scenario probabilities must sum to 1 within this absolute
#: tolerance — the deck writes them rounded to four decimals.
_PROBABILITY_ATOL = 1e-4

#: Relative tolerance on the block-factor identity ``Σ_b f_b·h_b = H``.
_FACTOR_RTOL = 1e-9


def _verdict(
    checks: list[CheckItem], diagnostics: list[Diagnostic]
) -> PreflightVerdict:
    if any(not check.passed for check in checks):
        return PreflightVerdict.WILL_NOT_CONVERT
    if any(diag.severity is Severity.WARNING for diag in diagnostics):
        return PreflightVerdict.WARNINGS
    return PreflightVerdict.OK


def _deferred(code: str, title: str, summary: str, remediation: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=Severity.WARNING,
        category=_CONTEXT,
        title=title,
        summary=summary,
        remediation=remediation,
    )


def _calendar_check(dadger: object) -> tuple[CheckItem, list, str | None]:
    """Build the operative calendar, turning its validation into one check."""
    from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

    try:
        calendar = operative_calendar_from_dadger(dadger)  # type: ignore[arg-type]
    except (ValueError, AttributeError, TypeError) as exc:
        return (
            CheckItem(
                label="Operative calendar (weekly walk, month-boundary close)",
                passed=False,
                detail=str(exc),
            ),
            [],
            str(exc),
        )
    span = f"{calendar[0].start_date} → {calendar[-1].end_date}"
    return (
        CheckItem(
            label="Operative calendar (weekly walk, month-boundary close)",
            passed=True,
            detail=f"{len(calendar)} stages, {span}",
        ),
        calendar,
        None,
    )


def _tree_checks(vazoes: object, calendar: list) -> list[CheckItem]:
    """Per-stage probability mass and the trunk-plus-terminal-fan shape gate."""
    from cobre_bridge.decomp.scenarios import convert_scenario_probabilities

    try:
        table = convert_scenario_probabilities(vazoes, calendar).to_pydict()  # type: ignore[arg-type]
    except (ValueError, KeyError, AttributeError) as exc:
        return [
            CheckItem(
                label="Scenario tree probabilities",
                passed=False,
                detail=str(exc),
            )
        ]

    per_stage: dict[int, float] = {}
    counts: dict[int, int] = {}
    for stage_id, probability in zip(
        table["stage_id"], table["probability"], strict=True
    ):
        per_stage[stage_id] = per_stage.get(stage_id, 0.0) + probability
        counts[stage_id] = counts.get(stage_id, 0) + 1

    worst = max(
        (abs(total - 1.0) for total in per_stage.values()),
        default=0.0,
    )
    checks = [
        CheckItem(
            label="Scenario probabilities sum to 1 per stage",
            passed=worst <= _PROBABILITY_ATOL,
            detail=f"worst |Σp − 1| = {worst:.2e}",
        )
    ]

    # The training path enumerates a trunk that branches once, at the end. A
    # deck that fans earlier is a node-graph study and would be mis-modelled
    # silently as a wider terminal fan.
    ordered = [counts[stage] for stage in sorted(counts)]
    branching = [i for i, n in enumerate(ordered) if n > 1]
    fan_is_terminal = not branching or branching == [len(ordered) - 1]
    checks.append(
        CheckItem(
            label="Tree shape is a trunk with one terminal fan",
            passed=fan_is_terminal,
            detail=f"nodes per stage: {ordered}",
        )
    )
    return checks


def _load_factor_check(dadger: object, id_map: object, calendar: list) -> CheckItem:
    """The per-(bus, stage) identity ``Σ_b f_b·h_b = H`` (matrix row 17)."""
    from cobre_bridge.decomp.load import convert_load_factors

    try:
        document = convert_load_factors(dadger, id_map, calendar)  # type: ignore[arg-type]
    except (ValueError, KeyError, AttributeError) as exc:
        return CheckItem(label="Load block factors", passed=False, detail=str(exc))

    hours = {stage.index: list(stage.block_hours) for stage in calendar}
    worst = 0.0
    for entry in document.get("load_factors", []):
        stage_hours = hours.get(entry["stage_id"])
        if stage_hours is None:
            continue
        total = sum(
            factor["factor"] * hour
            for factor, hour in zip(entry["block_factors"], stage_hours, strict=False)
        )
        span = sum(stage_hours)
        worst = max(worst, abs(total - span) / span)
    return CheckItem(
        label="Load block factors reproduce the stage span",
        passed=worst <= _FACTOR_RTOL,
        detail=f"worst relative deviation {worst:.2e}",
    )


def _deferred_inventory(dadger: object, files: object) -> list[Diagnostic]:
    """Name what the conversion will not carry, so it is never a surprise."""
    found: list[Diagnostic] = []

    if getattr(files, "dadgnl", None) is not None:
        found.append(
            _deferred(
                "decomp-anticipation-deferred",
                "Anticipated thermal generation not converted",
                "The deck declares anticipated (lead-time) thermal generation; "
                "those plants are absent from the converted case, so their "
                "generation and commitment are not modelled.",
                "Compare against the reference run will report them as entities "
                "with no converted counterpart.",
            )
        )

    for register, code, title, summary in (
        (
            "ez",
            "decomp-ez-ignored",
            "Coupling volume limit ignored",
            "The deck carries maximum-useful-volume records for the boundary "
            "coupling; they are read and ignored by decision.",
        ),
        (
            "mp",
            "decomp-availability-deferred",
            "Per-stage availability not applied",
            "The deck declares per-stage maintenance and availability factors; "
            "the converted capacity is static, so a stage under maintenance "
            "carries more capacity than the reference allows.",
        ),
    ):
        try:
            frame = getattr(dadger, register)(df=True)
        except (AttributeError, TypeError, ValueError):
            continue
        if frame is None or frame.empty:
            continue
        found.append(
            _deferred(
                code,
                title,
                f"{summary} ({len(frame)} record(s)).",
                "Tracked in the conversion roadmap; no action needed to convert.",
            )
        )

    return found


def run_decomp_preflight(src: Path) -> PreflightResult:
    """Validate the deck at *src*, writing nothing and raising nothing.

    A discovery failure is terminal and returns immediately; everything after
    it needs the resolved files. Later checks degrade independently — a
    malformed calendar blocks the tree and load checks (both need it) but
    still reports alongside the inventory.
    """
    from idecomp.decomp import Dadger, Vazoes

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.pipeline import discover_decomp_files

    try:
        files = discover_decomp_files(src)
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=[diagnostic_from_exception(exc, context=_CONTEXT)],
            checks=[
                CheckItem(
                    label="Deck discovery (caso.dat → revision index)",
                    passed=False,
                    detail=str(exc),
                )
            ],
        )

    checks: list[CheckItem] = [
        CheckItem(
            label="Deck discovery (caso.dat → revision index)",
            passed=True,
            detail=f"revision {files.revision}",
        )
    ]
    diagnostics: list[Diagnostic] = []

    for name in ("dadgnl", "renovaveis"):
        if getattr(files, name) is None:
            checks.append(
                CheckItem(
                    label=f"Optional: {name}",
                    passed=True,
                    detail="absent (nothing to convert from it)",
                )
            )

    try:
        dadger = Dadger.read(str(files.dadger))
    except Exception as exc:  # noqa: BLE001
        checks.append(
            CheckItem(label="Deck registers readable", passed=False, detail=str(exc))
        )
        diagnostics.append(diagnostic_from_exception(exc, context=_CONTEXT))
        return PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=diagnostics,
            checks=checks,
        )

    try:
        id_map = DecompIdMap.from_dadger(dadger)
    except ValueError as exc:
        checks.append(CheckItem(label="Entity id map", passed=False, detail=str(exc)))
        return PreflightResult(
            verdict=PreflightVerdict.WILL_NOT_CONVERT,
            diagnostics=diagnostics,
            checks=checks,
        )

    checks.append(
        CheckItem(
            label="Entity id map",
            passed=True,
            detail=(
                f"{len(id_map.bus_codes)} subsystems, "
                f"{len(id_map.hydro_codes)} hydros, "
                f"{len(id_map.thermal_codes)} thermals"
            ),
        )
    )

    calendar_check, calendar, failure = _calendar_check(dadger)
    checks.append(calendar_check)

    if failure is None:
        checks.append(_load_factor_check(dadger, id_map, calendar))
        try:
            vazoes = Vazoes.read(str(files.vazoes))
        except Exception as exc:  # noqa: BLE001
            checks.append(
                CheckItem(label="Scenario tree readable", passed=False, detail=str(exc))
            )
        else:
            checks.extend(_tree_checks(vazoes, calendar))

    diagnostics.extend(_deferred_inventory(dadger, files))

    return PreflightResult(
        verdict=_verdict(checks, diagnostics),
        diagnostics=diagnostics,
        checks=checks,
    )
