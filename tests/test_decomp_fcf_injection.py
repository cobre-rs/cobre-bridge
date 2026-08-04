"""Injection-validate + short-train + manifest-identity guard for the
boundary FCF importer (ticket-014, epic 04).

Ticket-013 (elsewhere) proves the authored boundary cuts are
*value-faithful*. This module proves the authored checkpoint is
*operationally loadable*: cobre reads the boundary at the terminal stage,
passes its own on-load manifest slot-identity check
(``validate_policy_load``), and drives a solve. It reuses the same
same-study Option C fixture shape as ``test_decomp_fcf_importer.py``:
``example/decomp-set-24-rv0-small`` is both the convertible deck and the
boundary cut source (its own ``cortesh.dat``/``cortes-010.dat``) — a reduced
GNL deck (12 cuts, ~228 KB) that keeps boundary stage 10 while making the
short-train fast.

**TRACKED COBRE-GAP C8** (see
``~/git/cobre/plans/conversion-found-improvements.md`` and the code comment
at ``fcf/__init__.py::_patch_policy_boundary``): cobre resolves
``policy.boundary.path`` against the run's ``--output`` directory, not
``case_dir``, so every ``cobre run`` invocation here that touches the
boundary passes ``--output`` equal to the case dir it runs against — never
the default and never a distinct scratch output dir. cobre itself is not
touched; this is a workaround, tracked until C8 is fixed upstream.
"""

from __future__ import annotations

import inspect
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cobre
import pytest

from cobre_bridge.decomp.fcf import import_boundary_fcf
from cobre_bridge.decomp.fcf.bootstrap import (
    TerminalManifest,
    bootstrap_terminal_manifest,
)
from cobre_bridge.decomp.fcf.cortes import BoundaryCuts, CortesHeader, StageCutRecord
from cobre_bridge.decomp.fcf.mapper import MappingResult
from cobre_bridge.decomp.pipeline import convert_decomp_case

# Real, gitignored deck + local cobre build (see example/README.md and
# tests/test_decomp_fcf_importer.py's identical constants) — CI has neither,
# so the three e2e assertions below (validate / short-train / manifest-guard)
# are skipif-guarded on the binary, the deck, and the writer binding. The
# Requirement-0 C8-warning test needs none of the three (see its own
# docstring) and runs unconditionally.
_DECK = Path("example/decomp-set-24-rv0-small")
_CORTESH = _DECK / "cortesh.dat"
_CORTES = _DECK / "cortes-010.dat"
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"
_HAS_WRITER_BINDING = hasattr(cobre, "write_policy_checkpoint")

_HAS_E2E_DEPS = _COBRE_BIN.exists() and _DECK.exists() and _HAS_WRITER_BINDING
_skip_e2e = pytest.mark.skipif(
    not _HAS_E2E_DEPS,
    reason=(
        f"requires the local cobre binary ({_COBRE_BIN}), the "
        f"decomp-set-24-rv0-small deck ({_DECK}), and the "
        "write_policy_checkpoint writer binding"
    ),
)


@dataclass(frozen=True)
class _ImportedCase:
    """One converted-and-boundary-imported case, shared across this module's
    e2e tests."""

    case_dir: Path
    boundary_dir: Path


@pytest.fixture(scope="module")
def imported_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> _ImportedCase:
    """Convert ``decomp-set-24-rv0-small`` and import its own boundary FCF,
    once.

    Module-scoped and shared across this module's e2e tests: the bootstrap
    stage runs a real ``cobre run``, so this path executes exactly once
    regardless of how many assertions exercise its result. Guarded by the
    same skip condition as the consuming tests (belt-and-suspenders — a
    skipped test never reaches its fixtures in this suite's other epic-2/
    epic-4 modules, but the explicit ``pytest.skip`` here keeps this fixture
    safe even if invoked on its own).
    """
    if not _HAS_E2E_DEPS:
        pytest.skip(
            f"requires the local cobre binary ({_COBRE_BIN}), the "
            f"decomp-set-24-rv0-small deck ({_DECK}), and the "
            "write_policy_checkpoint writer binding"
        )

    root = tmp_path_factory.mktemp("fcf_injection_e2e")
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


def _short_train(case_dir: Path, root: Path) -> Path:
    """Run a fast, boundary-loading train on a scratch copy of ``case_dir``.

    Copies ``case_dir`` into ``root`` (never mutating the input case — the
    manifest-identity guard elsewhere in this module reads the untouched
    authored boundary checkpoint), then overrides the copy's ``config.json``
    stopping rule to a 2-iteration limit — leaving ``policy.boundary`` and
    ``policy.mode`` untouched, since boundary injection is mode-independent.

    ``--output`` is set to the scratch case dir itself — the TRACKED
    COBRE-GAP WORKAROUND (C8) recipe: cobre resolves ``policy.boundary.path``
    against ``--output``, not ``case_dir``, so ``--output`` must equal the
    scratch case dir for ``<scratch>/boundary`` to resolve. A default or
    differently-targeted ``--output`` hits C8 and ``cobre run`` aborts
    before any iteration.

    Asserts exit 0 (surfacing the stdout/stderr tail on failure) and returns
    the scratch case dir — also where the trained policy lands, at
    ``<scratch>/policy``.

    Mirrors ``fcf/bootstrap.py::bootstrap_terminal_manifest``'s scratch-copy
    + config-edit + run recipe, but does not reuse its private
    ``_set_iteration_limit_one`` (that helper forces a 1-iteration limit; the
    short train here needs 2, to run a forward pass that actually reaches
    the boundary-loaded terminal stage).
    """
    scratch_case = root / "scratch"
    shutil.copytree(case_dir, scratch_case)

    config_path = scratch_case / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 2}]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            str(_COBRE_BIN),
            "run",
            str(scratch_case),
            "--output",
            str(scratch_case),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"cobre run failed (exit {completed.returncode}) on {scratch_case}:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return scratch_case


def test_import_boundary_fcf_logs_c8_workaround(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Requirement 0 / AC 1 — the C8 cobre-gap workaround is logged, not
    silently absorbed.

    Needs no cobre binary and no real deck: every seam that would touch
    either (``bootstrap_terminal_manifest``, the deck readers, the
    checkpoint writer) is monkeypatched to a minimal stand-in — a
    "monkeypatched-shape unit path" — isolating the one thing under test:
    that ``import_boundary_fcf`` emits a ``WARNING`` naming the C8
    workaround right after patching ``policy.boundary``, and that the
    source carries the matching code comment at the patch site.
    """
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.discover_decomp_files",
        lambda _deck_dir: SimpleNamespace(dadger=Path("dadger.rv0")),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.Dadger",
        SimpleNamespace(read=lambda _path: object()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.DecompIdMap",
        SimpleNamespace(from_dadger=lambda _dadger: object()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.Cortesh",
        SimpleNamespace(read=lambda _path: object()),
    )
    # A real (if minimal) `BoundaryCuts`/`MappingResult` pair, not a bare
    # `SimpleNamespace` — ticket-015's `_emit_import_diagnostics` (called
    # between `map_boundary_cuts` and `write_boundary_checkpoint`) reads
    # `cuts.records` (via `summarize_cut_families`) and `mapping.dropped`, so
    # the fakes must carry those attributes with a shape that survives the
    # real triage pass, not just the fields this test's own assertions read.
    fake_cuts = BoundaryCuts(
        header=CortesHeader(
            plant_codes=(1,),
            submercado_codes=(1,),
            n_patamares=1,
            lag_maximo_gnl=0,
            n_plants=1,
            individualized=True,
            record_size=112,
            last_cut_record_by_stage=(1,),
        ),
        boundary_stage=10,
        records=(
            StageCutRecord(
                cut_id=1,
                iteration=1,
                forward_pass_index=0,
                is_active=True,
                rhs=10.0,
                pi_varm=(1.5,),
                pi_qafl=((0.0,) * 12,),
                pi_gnl=(),
            ),
        ),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.read_cortes",
        lambda *_args, **_kwargs: fake_cuts,
    )
    monkeypatch.setattr("cobre_bridge.decomp.fcf.ensure_writer_binding", lambda: None)
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: TerminalManifest(
            entity_manifest=({"entity_type": 0, "entity_id": 0, "subindex": 0},),
            state_dimension=1,
        ),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(cuts=(), dropped=()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.build_stage_cuts_payload",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    with caplog.at_level(logging.WARNING):
        boundary_dir = import_boundary_fcf(
            case_dir,
            tmp_path / "deck" / "cortesh.dat",
            tmp_path / "deck" / "cortes-010.dat",
            cobre_bin=Path("unused-cobre-bin"),
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
        )

    assert boundary_dir == case_dir / "boundary"

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("TRACKED COBRE-GAP WORKAROUND (C8)" in r.message for r in warnings), (
        f"no C8 WARNING found in captured records: {caplog.text}"
    )
    assert any("--output" in r.message for r in warnings), (
        f"C8 warning is missing --output guidance: {caplog.text}"
    )

    source_path = inspect.getsourcefile(import_boundary_fcf)
    assert source_path is not None
    source_text = Path(source_path).read_text(encoding="utf-8")
    assert "# TRACKED COBRE-GAP WORKAROUND (C8" in source_text


@_skip_e2e
def test_injection_validate_accepts_boundary_case(
    imported_case: _ImportedCase,
) -> None:
    """``cobre validate`` accepts the boundary-injected case (exit 0).

    Only the exit code gates this test — a non-fatal external-interop
    warning may legitimately appear on stdout/stderr without flipping it
    (same convention as ``test_decomp_fcf_importer.py::
    test_import_boundary_fcf_case_validates``). ``cobre validate`` does not
    load the boundary checkpoint, so it is unaffected by C8.
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


@_skip_e2e
def test_short_train_injects_boundary_at_terminal_stage(
    imported_case: _ImportedCase,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A 2-iteration train (run with the C8 ``--output==case`` workaround)
    loads and solves the injected boundary.

    Proves the boundary rows were actually injected at the terminal stage
    (not merely that the config validated): after the short train, the
    terminal stage's (max ``stage_id``) ``warm_start_count`` is nonzero.
    """
    root = tmp_path_factory.mktemp("short_train")
    output_dir = _short_train(imported_case.case_dir, root)

    policy = cobre.results.load_policy(output_dir, policy_subdir="policy")
    terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])

    assert terminal["warm_start_count"] > 0, (
        "terminal stage warm_start_count is 0 (or absent) after the short "
        f"train — the boundary was not injected (stage_id={terminal['stage_id']})"
    )


@_skip_e2e
def test_manifest_identity_guard_matches_fresh_bootstrap(
    imported_case: _ImportedCase,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """A fresh, boundary-free bootstrap agrees with the authored manifest.

    Re-derives the terminal manifest from a *second*, boundary-free
    conversion of the deck — never the imported case, which already carries
    ``policy.boundary`` and would hit C8 if bootstrapped again (the
    bootstrap's own scratch-copy + ``--output`` split lands outside the
    scratch case, exactly the C8 layout) — and asserts it is identical, same
    ``state_dimension`` and same ordered ``(entity_type, entity_id,
    subindex)`` triples, to the authored checkpoint's own
    ``entity_manifest``. The re-derivation is independent of the authored
    bytes, so it can detect cobre-side terminal-layout drift.
    """
    root = tmp_path_factory.mktemp("manifest_guard")
    boundary_free_case = root / "converted"
    convert_decomp_case(_DECK, boundary_free_case, force=True)

    fresh = bootstrap_terminal_manifest(
        boundary_free_case, _COBRE_BIN, work_dir=root / "work"
    )

    authored_policy = cobre.results.load_policy(
        imported_case.case_dir, policy_subdir="boundary"
    )
    authored_stage = authored_policy["stage_cuts"][0]

    assert fresh.state_dimension == authored_stage["state_dimension"]

    fresh_triples = [
        (slot["entity_type"], slot["entity_id"], slot["subindex"])
        for slot in fresh.entity_manifest
    ]
    authored_triples = [
        (slot["entity_type"], slot["entity_id"], slot["subindex"])
        for slot in authored_stage["entity_manifest"]
    ]
    assert fresh_triples == authored_triples
