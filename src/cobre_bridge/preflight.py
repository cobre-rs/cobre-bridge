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
from typing import TYPE_CHECKING, TypeVar

from cobre_bridge.diagnostics import Diagnostic, Severity
from cobre_bridge.errors import diagnostic_from_exception
from cobre_bridge.newave_files import NewaveFiles

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

_DISCOVERY_LABEL = "File discovery (caso.dat → arquivos.dat)"
_PREFLIGHT_CONTEXT = "Preflight"
_OPTIONAL_ABSENT_DETAIL = "absent (optional; conversion proceeds)"

_FilesT = TypeVar("_FilesT", bound="DataclassInstance")


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
        failure, ``INFO`` for each absent optional input.
    checks:
        The ``✓/✗`` :class:`CheckItem` list the renderer consumes.
    """

    verdict: PreflightVerdict
    diagnostics: list[Diagnostic] = field(default_factory=list)
    checks: list[CheckItem] = field(default_factory=list)


def _optional_file_fields(files: DataclassInstance) -> list[str]:
    """Optional (``Path | None``) field names of the passed files dataclass.

    Derived from the dataclass annotations (the ``| None`` ones) rather than a
    duplicated literal list, so the advisory stays in sync if the discovery
    dataclass grows new optional inputs. ``directory`` is excluded as it is not a
    discovered input file.
    """
    return [
        f.name for f in fields(files) if f.name != "directory" and "None" in str(f.type)
    ]


# ruff's PEP 695 rewrite of this signature is unsafe/incomplete (--unsafe-fixes
# renames the type parameter but leaves the body's `_FilesT` annotation as-is).
def optional_input_advisory(  # noqa: UP047
    files: _FilesT,
) -> tuple[list[CheckItem], list[Diagnostic]]:
    """Passing check + INFO advisory per absent optional input of *files*.

    Reflects over the ``Path | None`` fields of *files*'s dataclass (any
    files dataclass — the NEWAVE and DECOMP engines share this helper). Each
    that resolved to ``None`` yields a passing :class:`CheckItem` plus a
    ``Severity.INFO`` :class:`Diagnostic` (``code="optional-file-absent"``):
    an absent optional never blocks conversion and, being INFO not WARNING,
    never drives the ``WARNINGS`` verdict. Present optionals produce nothing.
    """
    checks: list[CheckItem] = []
    diagnostics: list[Diagnostic] = []
    for name in _optional_file_fields(files):
        if getattr(files, name) is not None:
            continue
        checks.append(
            CheckItem(
                label=f"Optional: {name}",
                passed=True,
                detail=_OPTIONAL_ABSENT_DETAIL,
            )
        )
        diagnostics.append(
            Diagnostic(
                code="optional-file-absent",
                severity=Severity.INFO,
                category="Preflight",
                title="Optional input absent",
                summary=(
                    f"Optional input '{name}' was not found; "
                    "the conversion proceeds without it."
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
    required-files-present check, a per-absent-optional advisory (``INFO``),
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

    optional_checks, optional_diags = optional_input_advisory(files)
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
