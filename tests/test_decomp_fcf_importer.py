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

import importlib.util
import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest
from inewave.newave import Cortesh

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.anticipated import GnlCommitmentModel, GnlThermal
from cobre_bridge.decomp.fcf import (
    _emit_import_diagnostics,
    _gnl_targets_from,
    import_boundary_fcf,
)
from cobre_bridge.decomp.fcf.cortes import (
    BoundaryCuts,
    read_cortes,
    summarize_cut_families,
)
from cobre_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    MappingResult,
    map_boundary_cuts,
)
from cobre_bridge.decomp.pipeline import convert_decomp_case
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cortes_header,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_mapped_cut,
    make_slot,
)

# Real, gitignored deck + local cobre build (see example/README.md and
# tests/test_decomp_fcf_bootstrap.py's identical constants) — CI has
# neither, so the heavy end-to-end tests (AC 1-3) are skipif-guarded on both
# plus the writer binding; the no-cut-files no-op (AC 4) needs none of them
# and runs unconditionally.
_DECK = Path("example/decomp-set-24-rv0")
_CORTESH = _DECK / "cortesh.dat"
_CORTES = _DECK / "cortes-010.dat"
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"


def _has_writer_binding() -> bool:
    """Whether an installed cobre wheel exposes ``write_policy_checkpoint``.

    Checked via ``importlib.util.find_spec`` before any import, so this
    module's collection never requires cobre — the real ``import cobre``
    below only executes once the package's presence is already confirmed
    (mirrors ``test_decomp_fcf_injection.py``'s identical helper).
    """
    if importlib.util.find_spec("cobre") is None:
        return False
    import cobre

    return hasattr(cobre, "write_policy_checkpoint")


_HAS_WRITER_BINDING = _has_writer_binding()

_HAS_E2E_DEPS = _COBRE_BIN.exists() and _DECK.exists() and _HAS_WRITER_BINDING
_SKIP_REASON = (
    f"requires the local cobre binary ({_COBRE_BIN}), the "
    f"decomp-set-24-rv0 deck ({_DECK}), and the write_policy_checkpoint "
    "writer binding"
)
_skip_e2e = pytest.mark.skipif(not _HAS_E2E_DEPS, reason=_SKIP_REASON)


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
        pytest.skip(_SKIP_REASON)

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
    # cobre 0.14 keys the cut file by pool id (stage 10 -> "010.bin").
    assert (boundary_dir / "cuts" / "010.bin").is_file()
    assert (boundary_dir / "basis").is_dir()

    metadata = json.loads((boundary_dir / "metadata.json").read_text())
    # 0.14 nests the algorithm provenance under a "producer" block.
    assert metadata["producer"]["cost_scale_factor"] == 1.0
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
    # The importer reserves the cut-derived inflow-lag depth (convert_config
    # emits no state_space); this individualized deck's boundary cuts carry
    # pi_qafl terms, so a positive depth is patched in.
    assert config["state_space"]["inflow_lag_depth"] >= 1
    # the convert_config sections must survive the policy.boundary patch untouched
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


@_skip_e2e
def test_import_boundary_fcf_emits_diagnostics(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """AC 1/AC 2 (ticket-015) — the importer's two INFO ``Diagnostic``s.

    A fresh convert+import, wrapped in its own ``dx.collect()`` block —
    ``imported_case``'s call above already ran with no sink installed, and
    ``diagnostics.emit`` cannot retroactively populate a sink that wasn't
    active at call time, so this test performs its own run rather than
    reusing that fixture's result.

    Asserts the dropped-plant table names exactly the two D3-dropped plants
    (codes 132, 176 — pinned from the epic-2 learnings) and that the
    cut-family-summary diagnostic's reported figures equal a *direct*
    ``summarize_cut_families`` call on an independently re-read
    ``BoundaryCuts`` — proving the diagnostic never re-implements the triage.
    """
    root = tmp_path_factory.mktemp("fcf_importer_diagnostics_e2e")
    case_dir = root / "converted"
    convert_decomp_case(_DECK, case_dir, force=True)

    with dx.collect() as sink:
        boundary_dir = import_boundary_fcf(
            case_dir,
            _CORTESH,
            _CORTES,
            cobre_bin=_COBRE_BIN,
            work_dir=root / "work",
            cost_scale_factor=1.0,
        )
    assert boundary_dir is not None

    assert len(sink) == 2
    by_code = {diagnostic.code: diagnostic for diagnostic in sink}
    assert set(by_code) == {
        "boundary-fcf-cut-family-summary",
        "boundary-fcf-source-only-plants-dropped",
    }

    dropped_diagnostic = by_code["boundary-fcf-source-only-plants-dropped"]
    assert dropped_diagnostic.severity is dx.Severity.INFO
    assert dropped_diagnostic.table is not None
    assert {row[0] for row in dropped_diagnostic.table.rows} == {132, 176}

    summary_diagnostic = by_code["boundary-fcf-cut-family-summary"]
    assert summary_diagnostic.severity is dx.Severity.INFO

    cortesh = Cortesh.read(str(_CORTESH))
    cuts = read_cortes(_CORTES, cortesh, boundary_stage=None)
    summary = summarize_cut_families(cuts)
    assert f"n_active_cuts={summary.n_active_cuts}" in summary_diagnostic.notes
    assert (
        f"storage_nonzero_plants={summary.storage_nonzero_plants}"
        in summary_diagnostic.notes
    )
    assert (
        f"lag_nonzero_by_depth={summary.lag_nonzero_by_depth}"
        in summary_diagnostic.notes
    )


def test_emit_import_diagnostics_ac1_ac2_from_synthetic() -> None:
    """AC 1/AC 2 (ticket-015) — both diagnostics fire against fully synthetic
    inputs: no deck, no cobre binary. ``dropped`` uses codes 20/30 rather
    than the real deck's 132/176 (see
    ``test_import_boundary_fcf_emits_diagnostics`` above), so this test
    carries no hidden dependency on ``example/``.
    """
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),),
        dropped=(
            DroppedTerm(plant_code=20, beta=0.4),
            DroppedTerm(plant_code=30, beta=0.7),
        ),
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    assert {diagnostic.code for diagnostic in sink} == {
        "boundary-fcf-cut-family-summary",
        "boundary-fcf-source-only-plants-dropped",
    }
    assert all(diagnostic.severity is dx.Severity.INFO for diagnostic in sink)

    by_code = {diagnostic.code: diagnostic for diagnostic in sink}
    dropped_diagnostic = by_code["boundary-fcf-source-only-plants-dropped"]
    assert dropped_diagnostic.table is not None
    assert {row[0] for row in dropped_diagnostic.table.rows} == {20, 30}

    summary_diagnostic = by_code["boundary-fcf-cut-family-summary"]
    summary = summarize_cut_families(cuts)
    assert f"n_active_cuts={summary.n_active_cuts}" in summary_diagnostic.notes
    assert (
        f"storage_nonzero_plants={summary.storage_nonzero_plants}"
        in summary_diagnostic.notes
    )
    assert (
        f"lag_nonzero_by_depth={summary.lag_nonzero_by_depth}"
        in summary_diagnostic.notes
    )


def test_emit_import_diagnostics_no_dropped_gates_dropped_diagnostic_off() -> None:
    """AC 3 (ticket-015) — ``mapping.dropped == ()`` gates off the
    dropped-plant diagnostic while the cut-family-summary diagnostic still
    fires. No cobre binary or real deck needed: both payloads are hand-built
    from the shared ``tests/_fcf_fixtures.py`` builders (ticket-003).
    """
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    assert [diagnostic.code for diagnostic in sink] == [
        "boundary-fcf-cut-family-summary"
    ]
    assert sink[0].severity is dx.Severity.INFO


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


# ---------------------------------------------------------------------------
# ticket-010: GnlRingPlan build (`_gnl_targets_from`) + the per-cut GNL
# deviation diagnostic. All tier-1: pure Python, no deck, no cobre binary.
# The entity_type code is restated locally rather than importing the
# mapper module's private constant — mirrors ``test_decomp_fcf_mapper.py``'s
# identical convention.
# ---------------------------------------------------------------------------

_HYDRO_STORAGE = 0
_ANTICIPATED_THERMAL_STATE = 2


def _gnl_row(width: int, nonzero: dict[int, float]) -> tuple[float, ...]:
    """A `pi_gnl` flat vector of `width` zeros with `nonzero` columns set."""
    row = [0.0] * width
    for column, value in nonzero.items():
        row[column] = value
    return tuple(row)


def _make_gnl_model(
    thermals: tuple[GnlThermal, ...], nl_lag_months: dict[int, int]
) -> GnlCommitmentModel:
    """A minimal `GnlCommitmentModel` carrying only the ring-plan join surface."""
    return GnlCommitmentModel(
        thermals=thermals,
        commitments={},
        weeks_per_month={},
        nl_lag_months=nl_lag_months,
    )


def _make_gnl_thermal(code: int, submarket_code: int) -> GnlThermal:
    """A minimal `GnlThermal`; cost/bounds are irrelevant to the ring join."""
    return GnlThermal(
        code=code,
        name=f"GNL-{code}",
        submarket_code=submarket_code,
        cost_per_mwh=0.0,
        min_mw=0.0,
        max_mw=0.0,
    )


def test_gnl_targets_from_maps_codes_to_ids() -> None:
    """AC — the READBACK join: code 86 -> id 94 (SE, lag 2), code 224 -> id 95
    (NE, lag 1), ascending by code."""
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)),
        {86: 2, 224: 1},
    )
    thermals_doc = {
        "thermals": [
            {"id": 1, "name": "not-gnl"},
            {"id": 94, "name": "GNL-86", "anticipated_config": {}},
            {"id": 95, "name": "GNL-224", "anticipated_config": {}},
        ]
    }

    plan = _gnl_targets_from(model, thermals_doc)

    assert plan == GnlRingPlan(
        (
            GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),
            GnlThermalTarget(thermal_id=95, submercado=3, nl_lag=1),
        )
    )


def test_gnl_targets_from_count_mismatch_raises() -> None:
    """AC — a thermals.json GNL-fleet count that disagrees with the dadgnl
    registry count raises loudly, naming both counts, rather than silently
    mis-zipping ids onto the wrong codes."""
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)),
        {86: 2, 224: 1},
    )
    thermals_doc = {
        "thermals": [
            {"id": 94, "anticipated_config": {}},
            {"id": 95, "anticipated_config": {}},
            {"id": 96, "anticipated_config": {}},
        ]
    }

    with pytest.raises(ValueError, match="GNL fleet must match exactly"):
        _gnl_targets_from(model, thermals_doc)


def test_gnl_targets_from_skips_plant_without_nl_lag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AC — a plant absent from ``nl_lag_months`` is skipped (its ring stays
    at coefficient 0.0 via the mapper's own drop path), never raised on, with
    a single INFO note naming it."""
    model = _make_gnl_model(
        (_make_gnl_thermal(86, 1), _make_gnl_thermal(224, 3)),
        {224: 1},  # 86 declares no nl dispatch-anticipation lag
    )
    thermals_doc = {
        "thermals": [
            {"id": 94, "anticipated_config": {}},
            {"id": 95, "anticipated_config": {}},
        ]
    }

    with caplog.at_level(logging.INFO):
        plan = _gnl_targets_from(model, thermals_doc)

    assert plan == GnlRingPlan(
        (GnlThermalTarget(thermal_id=95, submercado=3, nl_lag=1),)
    )
    assert "GNL-86" in caplog.text


def test_emit_import_diagnostics_gnl_deviation_fires() -> None:
    """AC — for (submercado 1, lag 2) with col(1,p,2) = (0.1, 0.2, 0.3), the
    deviation diagnostic's row reports carried sum 0.6 and patamar spread
    (0.3 - 0.1) / 0.6. ``mapping`` is produced by the real
    ``map_boundary_cuts`` (ticket-009's mapper), never hand-built, so the
    diagnostic is exercised against the same placement it reports on.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, delivery_date=20260501),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    # col(1,p,2) for p=1..3 -> flat indices 1, 3, 5 (P=3, L=2):
    # col(s,p,l) = ((s-1)*3 + (p-1))*2 + (l-1).
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan((GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),))

    mapping = map_boundary_cuts(cuts, manifest, id_map, gnl_plan=gnl_plan)

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    codes = [diagnostic.code for diagnostic in sink]
    assert codes == [
        "boundary-fcf-cut-family-summary",
        "boundary-fcf-gnl-anticipated-deviation",
    ]
    deviation = sink[1]
    assert deviation.severity is dx.Severity.INFO
    assert deviation.table is not None
    row = next(r for r in deviation.table.rows if r[0] == 1 and r[1] == 2)
    assert row[2] == pytest.approx(0.6)
    assert row[3] == pytest.approx((0.3 - 0.1) / 0.6)


def test_emit_import_diagnostics_gnl_deviation_gated_off_without_plan() -> None:
    """AC — ``gnl_plan=None`` (the default) gates the deviation diagnostic
    off entirely: the sink carries only the pre-ticket-010 diagnostic(s),
    unchanged."""
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    assert [diagnostic.code for diagnostic in sink] == [
        "boundary-fcf-cut-family-summary"
    ]
