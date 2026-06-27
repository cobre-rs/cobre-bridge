"""Pure pre-conversion validation of a source-model case directory.

The ``check`` command must answer "would this case convert?" *without* running
the conversion or writing any output. This module is that validation engine: it
resolves files via :meth:`NewaveFiles.from_directory`, confirms the required
inputs are present, raises a per-file advisory for each absent optional input,
runs a couple of structural sanity checks, and returns a structured
:class:`PreflightResult` — a ``✓/✗`` :class:`CheckItem` list plus the
:class:`Diagnostic` list and an overall :class:`PreflightVerdict`.

It contains no Typer code and writes no file, so it is unit-testable with mocks
and reusable by both the CLI command and, later, the ``--dry-run`` path. It does
no deep field parsing: binary inputs (``hidr.dat``, ``vazoes.dat``) are checked
only at the discovery/structural level — their contents are never read here.

The verdict is ``WILL_NOT_CONVERT`` if any check failed, else ``WARNINGS`` if any
``WARNING`` diagnostic was emitted, else ``OK``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from pathlib import Path

from cobre_bridge.diagnostics import Diagnostic, Severity
from cobre_bridge.errors import diagnostic_from_exception
from cobre_bridge.newave_files import NewaveFiles

_DISCOVERY_LABEL = "File discovery (caso.dat → arquivos.dat)"
_PREFLIGHT_CONTEXT = "Preflight"


class PreflightVerdict(Enum):
    """Overall outcome of a preflight run, in increasing order of severity."""

    OK = "ok"
    WARNINGS = "warnings"
    WILL_NOT_CONVERT = "will-not-convert"


@dataclass(frozen=True)
class CheckItem:
    """One ``✓/✗`` line in the preflight report.

    Attributes
    ----------
    label:
        Short human description of what was checked.
    passed:
        ``True`` for a ``✓`` line, ``False`` for a ``✗`` (blocking) line.
    detail:
        Optional context shown after the label (e.g. why it failed, or the
        absent-optional note).
    """

    label: str
    passed: bool
    detail: str | None = None


@dataclass(frozen=True)
class PreflightResult:
    """Structured outcome of :func:`run_preflight`.

    Attributes
    ----------
    verdict:
        The overall :class:`PreflightVerdict`.
    diagnostics:
        The :class:`Diagnostic` list — ``ERROR`` for a blocking discovery
        failure, ``WARNING`` for each absent optional file.
    checks:
        The ``✓/✗`` :class:`CheckItem` list the renderer consumes.
    """

    verdict: PreflightVerdict
    diagnostics: list[Diagnostic] = field(default_factory=list)
    checks: list[CheckItem] = field(default_factory=list)


def _optional_file_fields() -> list[str]:
    """Return the optional (``Path | None``) field names of :class:`NewaveFiles`.

    Derived from the dataclass annotations (the ``| None`` ones) rather than a
    duplicated literal list, so the advisory stays in sync if the discovery
    dataclass grows new optional inputs. ``directory`` is excluded as it is not a
    discovered input file.
    """
    return [
        f.name
        for f in fields(NewaveFiles)
        if f.name != "directory" and "None" in str(f.type)
    ]


def _optional_advisory(
    files: NewaveFiles,
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Build the absent-optional checks and ``WARNING`` diagnostics for *files*.

    For each optional input that resolved to ``None`` (absent), emit a *passing*
    :class:`CheckItem` (it does not block conversion) and a ``WARNING``
    :class:`Diagnostic` noting defaults will be used. Present optionals produce
    nothing.
    """
    checks: list[CheckItem] = []
    diagnostics: list[Diagnostic] = []
    for name in _optional_file_fields():
        if getattr(files, name) is not None:
            continue
        checks.append(
            CheckItem(
                label=f"Optional: {name}",
                passed=True,
                detail="absent (will use defaults)",
            )
        )
        diagnostics.append(
            Diagnostic(
                code="optional-file-absent",
                severity=Severity.WARNING,
                category="Preflight",
                title="Optional input absent",
                summary=(
                    f"Optional input '{name}' was not found; "
                    "the converter will fall back to its defaults."
                ),
                notes=[f"field: {name}"],
                remediation=(
                    "→ Provide the file if this case relies on it; "
                    "otherwise this is informational."
                ),
            )
        )
    return checks, diagnostics


def _structural_checks(src: Path, files: NewaveFiles) -> list[CheckItem]:
    """Discovery-level structural sanity checks that read no file contents.

    Confirms *src* is a directory and that ``caso.dat`` resolved (``files`` only
    exists once discovery succeeded, so its directory entry point is present).
    Deep field parsing is out of scope.
    """
    return [
        CheckItem(
            label="Source path is a directory",
            passed=src.is_dir(),
            detail=str(src),
        ),
        CheckItem(
            label="caso.dat resolved",
            passed=True,
            detail=str(files.directory),
        ),
    ]


def _verdict_from(
    checks: list[CheckItem], diagnostics: list[Diagnostic]
) -> PreflightVerdict:
    """Compute the verdict: a failed check blocks, else a warning warns, else OK."""
    if any(not check.passed for check in checks):
        return PreflightVerdict.WILL_NOT_CONVERT
    if any(diag.severity is Severity.WARNING for diag in diagnostics):
        return PreflightVerdict.WARNINGS
    return PreflightVerdict.OK


def run_preflight(src: Path) -> PreflightResult:
    """Validate the source-model case at *src* without converting or writing.

    Resolves the case files via :meth:`NewaveFiles.from_directory`; on a
    discovery failure (a missing required input, or any unexpected parse error)
    the failure is captured as a failed :class:`CheckItem` plus an ``ERROR``
    :class:`Diagnostic` and the result is returned immediately with verdict
    ``WILL_NOT_CONVERT`` — nothing propagates. On successful discovery it adds the
    required-files-present check, a per-absent-optional advisory (``WARNING``),
    and structural sanity checks, then computes the verdict.

    This function writes no file and raises nothing for a missing input.
    """
    try:
        files = NewaveFiles.from_directory(src)
    except Exception as exc:  # noqa: BLE001
        # Any discovery failure — a missing required input (SourceFileError, a
        # FileNotFoundError subclass) or an unexpected parse error — is non-
        # recoverable for preflight; surface it as a blocking verdict.
        return _discovery_failure(exc)

    checks: list[CheckItem] = [
        CheckItem(
            label="Required files present",
            passed=True,
            detail="none missing",
        )
    ]
    diagnostics: list[Diagnostic] = []

    optional_checks, optional_diags = _optional_advisory(files)
    checks.extend(optional_checks)
    diagnostics.extend(optional_diags)

    checks.extend(_structural_checks(src, files))

    return PreflightResult(
        verdict=_verdict_from(checks, diagnostics),
        diagnostics=diagnostics,
        checks=checks,
    )


def _discovery_failure(exc: Exception) -> PreflightResult:
    """Build the blocking result for a failed file-discovery step."""
    return PreflightResult(
        verdict=PreflightVerdict.WILL_NOT_CONVERT,
        diagnostics=[diagnostic_from_exception(exc, context=_PREFLIGHT_CONTEXT)],
        checks=[
            CheckItem(label=_DISCOVERY_LABEL, passed=False, detail=str(exc)),
        ],
    )
