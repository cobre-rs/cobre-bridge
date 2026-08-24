"""Unit tests for the pure preflight validation engine (``cobre_bridge.preflight``)."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from cobre_bridge import preflight
from cobre_bridge.diagnostics import Severity
from cobre_bridge.newave_files import NewaveFiles
from cobre_bridge.preflight import PreflightVerdict, run_preflight
from tests.conftest import make_nw_files

# Optional field names derived the same way the module does, so the "all present"
# fixture and the assertions stay in sync with the discovery dataclass.
_OPTIONAL_FIELDS = [
    f.name
    for f in fields(NewaveFiles)
    if f.name != "directory" and "None" in str(f.type)
]


def _all_optionals_present(tmp_path: Path) -> dict[str, Path]:
    """Every optional input resolved to a path under *tmp_path* (none absent)."""
    return {name: tmp_path / f"{name}.dat" for name in _OPTIONAL_FIELDS}


class TestRunPreflight:
    def test_missing_caso_yields_will_not_convert(self, tmp_path: Path) -> None:
        # An empty directory has no caso.dat: real discovery raises a
        # SourceFileError before any binary is parsed.
        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.WILL_NOT_CONVERT
        error_codes = [
            diag.code for diag in result.diagnostics if diag.severity is Severity.ERROR
        ]
        assert "source-file-missing" in error_codes
        assert any(not check.passed for check in result.checks)

    def test_absent_optional_yields_ok_with_info_advisory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All optionals present except cvar (left None): discovery succeeds, the
        # single absent optional is advisory-only and does not block or warn.
        present = _all_optionals_present(tmp_path)
        present["cvar"] = None  # type: ignore[assignment]
        files = make_nw_files(tmp_path, **present)
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.OK
        info_codes = [
            diag.code for diag in result.diagnostics if diag.severity is Severity.INFO
        ]
        warning_codes = [
            diag.code
            for diag in result.diagnostics
            if diag.severity is Severity.WARNING
        ]
        assert "optional-file-absent" in info_codes
        assert "optional-file-absent" not in warning_codes
        assert all(check.passed for check in result.checks)

    def test_complete_case_yields_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)

        result = run_preflight(tmp_path)

        assert result.verdict is PreflightVerdict.OK
        assert result.diagnostics == []
        assert all(check.passed for check in result.checks)

    def test_preflight_writes_no_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))
        monkeypatch.setattr(preflight.NewaveFiles, "from_directory", lambda _src: files)
        before = sorted(p.name for p in tmp_path.iterdir())

        run_preflight(tmp_path)

        after = sorted(p.name for p in tmp_path.iterdir())
        assert after == before

    def test_missing_caso_writes_no_files(self, tmp_path: Path) -> None:
        before = sorted(p.name for p in tmp_path.iterdir())

        run_preflight(tmp_path)

        after = sorted(p.name for p in tmp_path.iterdir())
        assert after == before


class TestOptionalInputAdvisory:
    def test_optional_input_advisory_reports_every_absent_optional(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path)  # every optional field left None

        checks, diagnostics = preflight.optional_input_advisory(files)

        assert len(checks) == len(_OPTIONAL_FIELDS)
        assert len(diagnostics) == len(_OPTIONAL_FIELDS)
        assert all(diag.severity is Severity.INFO for diag in diagnostics)
        assert all(diag.code == "optional-file-absent" for diag in diagnostics)
        assert all(
            check.detail == "absent (optional; conversion proceeds)" for check in checks
        )

    def test_optional_input_advisory_skips_present_optionals(
        self, tmp_path: Path
    ) -> None:
        files = make_nw_files(tmp_path, **_all_optionals_present(tmp_path))

        assert preflight.optional_input_advisory(files) == ([], [])
