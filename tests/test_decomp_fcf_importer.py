"""End-to-end tests for the boundary FCF importer orchestration
(``fcf/__init__.py::import_boundary_fcf``).

Option C (same-study, 2026-08-03): ``example/decomp-set-24-rv0`` is both the
convertible deck (``dadger.rv0`` -> the target case) and the boundary cut
source (its own ``cortesh.dat``/``cortes-010.dat``, a single-stage export
whose trailer derives boundary stage 10). The deck carries GNL, but the
epic-2 mapper leaves every ``AnticipatedThermalState`` slot at coefficient 0
(``pi_gnl`` is epic 3's job) — this converted case's terminal manifest in
fact carries *no* ``AnticipatedThermalState``/``HydroTransitBucket`` slots at
all yet (GNL-anticipation emission is deferred at this DECOMP milestone), so
there is nothing to zero here; the mapper's docstring names this an
explicitly legitimate case shape, not an error.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cobre
import pytest

from cobre_bridge.decomp.fcf import import_boundary_fcf
from cobre_bridge.decomp.pipeline import convert_decomp_case

# Real, gitignored deck + local cobre build (see example/README.md and
# tests/test_decomp_fcf_bootstrap.py's identical constants) — CI has
# neither, so the heavy end-to-end tests (AC 1-3) are skipif-guarded on both
# plus the writer binding; the no-cut-files no-op (AC 4) needs none of them
# and runs unconditionally.
_DECK = Path("example/decomp-set-24-rv0")
_CORTESH = _DECK / "cortesh.dat"
_CORTES = _DECK / "cortes-010.dat"
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"
_HAS_WRITER_BINDING = hasattr(cobre, "write_policy_checkpoint")

_HAS_E2E_DEPS = _COBRE_BIN.exists() and _DECK.exists() and _HAS_WRITER_BINDING
_skip_e2e = pytest.mark.skipif(
    not _HAS_E2E_DEPS,
    reason=(
        f"requires the local cobre binary ({_COBRE_BIN}), the "
        f"decomp-set-24-rv0 deck ({_DECK}), and the write_policy_checkpoint "
        "writer binding"
    ),
)


@dataclass(frozen=True)
class _ImportedCase:
    """One converted-and-boundary-imported case, shared across the AC1-3 tests."""

    case_dir: Path
    boundary_dir: Path


@pytest.fixture(scope="module")
def imported_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> _ImportedCase:
    """Convert ``decomp-set-24-rv0`` and import its own boundary FCF, once.

    Module-scoped and shared across the three AC1-3 tests below: the deck's
    ``cortes-010.dat`` is ~176 MB and the bootstrap stage runs a real
    ``cobre run``, so this ~30s+ path executes exactly once regardless of how
    many assertions exercise its result. Guarded by the same skip condition
    as the consuming tests (belt-and-suspenders — a skipped test never
    reaches its fixtures in this suite's other epic-2 modules, but the
    explicit ``pytest.skip`` here keeps this fixture safe even if invoked on
    its own).
    """
    if not _HAS_E2E_DEPS:
        pytest.skip(
            f"requires the local cobre binary ({_COBRE_BIN}), the "
            f"decomp-set-24-rv0 deck ({_DECK}), and the "
            "write_policy_checkpoint writer binding"
        )

    root = tmp_path_factory.mktemp("fcf_importer_e2e")
    case_dir = root / "converted"
    convert_decomp_case(_DECK, case_dir, force=True)

    boundary_dir = import_boundary_fcf(
        case_dir,
        _CORTESH,
        _CORTES,
        cobre_bin=_COBRE_BIN,
        work_dir=root / "work",
        cost_scale_factor=1.0,
    )
    assert boundary_dir is not None
    return _ImportedCase(case_dir=case_dir, boundary_dir=boundary_dir)


@_skip_e2e
def test_import_boundary_fcf_nongnl_writes_checkpoint(
    imported_case: _ImportedCase,
) -> None:
    """AC 1 — the boundary checkpoint's files exist at the derived stage 10."""
    boundary_dir = imported_case.boundary_dir

    assert boundary_dir == imported_case.case_dir / "boundary"
    assert (boundary_dir / "metadata.json").is_file()
    assert (boundary_dir / "cuts" / "stage_010.bin").is_file()
    assert (boundary_dir / "basis").is_dir()

    metadata = json.loads((boundary_dir / "metadata.json").read_text())
    assert metadata["cost_scale_factor"] == 1.0
    assert metadata["num_stages"] == 1


@_skip_e2e
def test_import_boundary_fcf_patches_policy_boundary(
    imported_case: _ImportedCase,
) -> None:
    """AC 2 — config.json's policy.boundary is wired at the derived stage 10."""
    config = json.loads((imported_case.case_dir / "config.json").read_text())

    assert config["policy"]["boundary"] == {
        "path": "boundary",
        "source_stage": 10,
    }
    # the rest of the case's config sections must survive the patch untouched
    assert "state_space" in config
    assert "training" in config
    assert "simulation" in config


@_skip_e2e
def test_import_boundary_fcf_case_validates(imported_case: _ImportedCase) -> None:
    """AC 3 — ``cobre validate`` accepts the boundary-injected case.

    Only the exit code gates this test — a non-fatal external-interop
    warning (e.g. ``inflow_lags``) may legitimately appear on stdout/stderr
    without flipping it.
    """
    completed = subprocess.run(
        [str(_COBRE_BIN), "validate", str(imported_case.case_dir)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        f"cobre validate failed (exit {completed.returncode}):\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


def test_import_boundary_fcf_no_cut_files_is_noop(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """AC 4 — ``cortesh_path``/``cortes_path`` both ``None`` is an
    unconditional no-op: no ``boundary/`` directory, no binary invoked."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with caplog.at_level(logging.INFO):
        result = import_boundary_fcf(
            case_dir,
            None,
            None,
            cobre_bin=Path("/nonexistent/cobre"),
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
        )

    assert result is None
    assert not (case_dir / "boundary").exists()
    assert "boundary FCF skipped" in caplog.text
