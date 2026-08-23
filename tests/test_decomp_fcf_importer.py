"""Tests for the boundary FCF importer orchestration
(``fcf/importer.py::import_boundary_fcf``).

The GNL-ring and diagnostics tests below are tier-1 (synthetic fixtures, no
deck, no cobre binary); the ``decomp-mar-26-rv2`` block further down is the
real-deck, real-cobre-binary end-to-end round trip.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.converters.network import MONTH_HOURS
from cobre_bridge.decomp.anticipated import GnlCommitmentModel, GnlThermal
from cobre_bridge.decomp.case import DecompCase
from cobre_bridge.decomp.fcf.cortes import BoundaryCuts, summarize_cut_families
from cobre_bridge.decomp.fcf.importer import (
    _coupling_stage_hours,
    _emit_import_diagnostics,
    _gnl_targets_from,
    _post_horizon_start,
    import_boundary_fcf,
)
from cobre_bridge.decomp.fcf.mapper import (
    DroppedTerm,
    GnlRingPlan,
    GnlThermalTarget,
    MappingResult,
    map_boundary_cuts,
)
from cobre_bridge.decomp.pipeline import FcfInputs, convert_decomp_case
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cortes_header,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_mapped_cut,
    make_slot,
)
from tests.conftest import make_decomp_case, make_decomp_files


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


def test_emit_import_diagnostics_ac1_ac2_from_synthetic() -> None:
    """AC 1/AC 2 (ticket-015) — both diagnostics fire against fully synthetic
    inputs: no deck, no cobre binary. ``dropped`` uses codes 20/30 (not a
    real deck's plant codes), so this test carries no hidden dependency on
    ``example/``.
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
    # ticket-013 Requirement C.1: the figures live in `summary` now, not a
    # separate restating `notes` bullet (dropped as duplicate bloat).
    assert summary_diagnostic.notes == []
    assert str(summary.n_active_cuts) in summary_diagnostic.summary
    assert str(summary.storage_nonzero_plants) in summary_diagnostic.summary
    assert str(summary.lag_nonzero_by_depth) in summary_diagnostic.summary


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
    """AC 4 — ``case.files.cortesh``/``case.files.cortes`` both ``None`` is an
    unconditional no-op: no ``boundary/`` directory, no binary invoked."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    case = make_decomp_case(tmp_path)

    with caplog.at_level(logging.INFO):
        result = import_boundary_fcf(
            case_dir,
            case,
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
        )

    assert result is None
    assert not (case_dir / "boundary").exists()
    assert "boundary FCF skipped" in caplog.text


# ---------------------------------------------------------------------------
# ticket-003: thread node_id/graph_stage_id from the bootstrap manifest into
# `build_stage_cuts_payload`, and the R3 graph_stage_id/boundary_stage
# mismatch escalation. Every cut-reader/cobre-import seam is monkeypatched
# (mirrors `test_decomp_fcf_injection.py`'s seam-stubbing convention, kept
# local per the one-home-per-source-module test convention) — no real deck,
# cobre binary, or installed cobre wheel needed.
# ---------------------------------------------------------------------------


def _stub_import_seams(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_cuts: BoundaryCuts
) -> DecompCase:
    """Monkeypatch the deck/cut-reader/cobre-import seams shared by the
    node_id/graph_stage_id threading tests below, returning the
    ``DecompCase`` the importer reads instead of re-parsing a deck."""
    monkeypatch.setitem(sys.modules, "cobre", SimpleNamespace(__version__="0.13.0"))
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.Cortesh",
        SimpleNamespace(read=lambda _path: object()),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.read_cortes",
        lambda *_args, **_kwargs: fake_cuts,
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.ensure_writer_binding", lambda: None
    )

    files = make_decomp_files(
        tmp_path,
        cortesh=tmp_path / "deck" / "cortesh.dat",
        cortes=tmp_path / "deck" / "cortes-010.dat",
    )
    return make_decomp_case(
        files,
        dadger=object(),
        id_map=make_id_map(()),
        hidr=object(),
        calendar=[],
    )


def test_import_boundary_fcf_threads_node_and_graph_stage_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 1 — given a bootstrap manifest with node_id=0/graph_stage_id=4 and
    a cuts file whose boundary_stage=4, `build_stage_cuts_payload` is called
    with `cost_scale_factor=1.0`, `node_id=0`, `graph_stage_id=4`."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=4,
    )
    case = _stub_import_seams(monkeypatch, tmp_path, fake_cuts)
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer._final_stage_block_hours",
        lambda _case_dir: [648.0],
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest(
            [make_slot(0, 0, 0)], node_id=0, graph_stage_id=4
        ),
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.map_boundary_cuts",
        lambda *_args, **_kwargs: MappingResult(
            cuts=(make_mapped_cut(coefficients=(1.5,)),), dropped=()
        ),
    )
    calls: list[dict[str, object]] = []

    def _spy_build_stage_cuts_payload(
        *_args: object, **kwargs: object
    ) -> dict[str, object]:
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.build_stage_cuts_payload",
        _spy_build_stage_cuts_payload,
    )
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.write_boundary_checkpoint",
        lambda *_args, **_kwargs: None,
    )

    import_boundary_fcf(
        case_dir,
        case,
        work_dir=tmp_path / "work",
        cost_scale_factor=1.0,
        config={},
        initial_conditions={},
    )

    assert len(calls) == 1
    assert calls[0] == {
        "stage_id": 4,
        "cost_scale_factor": 1.0,
        "node_id": 0,
        "graph_stage_id": 4,
    }


def test_import_boundary_fcf_graph_stage_id_mismatch_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 2 (the R3 escalation) — a bootstrap manifest whose graph_stage_id
    disagrees with the cut file's own boundary_stage is a real
    inconsistency, never resolved by preferring one value: `import_
    boundary_fcf` raises `ValueError` naming both."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    fake_cuts = make_boundary_cuts(
        plant_codes=(1,),
        records=(make_cut_record(pi_varm=(1.5,), rhs=10.0, forward_pass_index=0),),
        boundary_stage=4,
    )
    case = _stub_import_seams(monkeypatch, tmp_path, fake_cuts)
    monkeypatch.setattr(
        "cobre_bridge.decomp.fcf.importer.bootstrap_terminal_manifest",
        lambda *_args, **_kwargs: make_manifest(
            [make_slot(0, 0, 0)], node_id=0, graph_stage_id=3
        ),
    )

    with pytest.raises(ValueError, match="graph_stage_id") as exc_info:
        import_boundary_fcf(
            case_dir,
            case,
            work_dir=tmp_path / "work",
            cost_scale_factor=1.0,
            config={},
            initial_conditions={},
        )

    assert "3" in str(exc_info.value)
    assert "4" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ticket-010: GnlRingPlan build (`_gnl_targets_from`) + the per-cut GNL
# deviation diagnostic. All tier-1: pure Python, no deck, no cobre binary.
# The entity_type code is restated locally rather than importing the
# mapper module's private constant — mirrors ``test_decomp_fcf_mapper.py``'s
# identical convention.
# ---------------------------------------------------------------------------

_HYDRO_STORAGE = 0
_ANTICIPATED_THERMAL_STATE = 2

#: Every GNL fixture below builds an `n_patamares=3` header; a uniform split
#: of `MONTH_HOURS` across those 3 coupling blocks is enough to satisfy
#: `map_boundary_cuts`'s `coupling_block_hours` guard (ticket-001) — none of
#: these tests assert on the placed GNL coefficient's magnitude, only on the
#: deviation diagnostic, which recomputes straight from the raw `pi_gnl`.
_UNIFORM_GNL_BLOCK_HOURS = (MONTH_HOURS / 3, MONTH_HOURS / 3, MONTH_HOURS / 3)


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

    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

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


# ---------------------------------------------------------------------------
# ticket-013: covered-lane filter. `_post_horizon_start` (tier-1, no deck, no
# cobre binary) + the deviation diagnostic's dropped-coverage count reading
# the new uncovered-lane drops from `mapping.gnl_dropped`; plus one tier-3
# `@skipif` e2e boundary-load smoke against the real `decomp-mar-26-rv2`
# deck + a CBVF-capable `cobre` binary (cobre >= 0.14.1).
# ---------------------------------------------------------------------------


def test_post_horizon_start_returns_earliest_stage(tmp_path: Path) -> None:
    """AC 3 — the earliest ``stages[i].start_date`` becomes a ``YYYYMMDD``
    int, regardless of the list's own order."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps(
            {
                "stages": [
                    {"start_date": "2026-05-09", "duration_hours": 168.0},
                    {"start_date": "2026-05-01", "duration_hours": 192.0},
                ],
                "thermal_bounds": [],
            }
        ),
        encoding="utf-8",
    )

    assert _post_horizon_start(case_dir) == 20260501


def test_post_horizon_start_none_when_file_absent(tmp_path: Path) -> None:
    """AC 3 — no ``post_study_stages.json`` at all returns ``None``."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    assert _post_horizon_start(case_dir) is None


def test_post_horizon_start_none_when_stages_empty(tmp_path: Path) -> None:
    """AC 3 — an empty ``stages`` list is the same "no horizon" case as an
    absent file."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps({"stages": [], "thermal_bounds": []}), encoding="utf-8"
    )

    assert _post_horizon_start(case_dir) is None


def test_post_horizon_start_malformed_start_date_propagates(tmp_path: Path) -> None:
    """Error Handling — a corrupt ``start_date`` is NOT silently downgraded
    to "no horizon"; the ``ValueError`` from ``int(...)`` propagates
    verbatim."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "post_study_stages.json").write_text(
        json.dumps({"stages": [{"start_date": "not-a-date"}]}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="invalid literal for int"):
        _post_horizon_start(case_dir)


def _write_stages(case_dir: Path, stages: list[dict[str, object]]) -> None:
    """Write a minimal ``stages.json`` (only the ``blocks[].hours`` the
    coupling-hours helper reads) into ``case_dir``."""
    (case_dir / "stages.json").write_text(
        json.dumps({"stages": stages}), encoding="utf-8"
    )


def test_coupling_stage_hours_sums_last_stage_blocks(tmp_path: Path) -> None:
    """The coupling-hours factor is the FINAL stage's summed block hours — the
    648 h April coupling period here, not an earlier weekly stage or a fixed
    730 h month."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_stages(
        case_dir,
        [
            {"blocks": [{"hours": 30.0}, {"hours": 74.0}, {"hours": 64.0}]},  # 168
            {"blocks": [{"hours": 108.0}, {"hours": 261.0}, {"hours": 279.0}]},  # 648
        ],
    )

    assert _coupling_stage_hours(case_dir) == pytest.approx(648.0)


def test_coupling_stage_hours_missing_file_raises(tmp_path: Path) -> None:
    """A boundary import cannot scale coefficients without the coupling stage's
    duration — an absent ``stages.json`` is a hard error, never a silent 0/730."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()

    with pytest.raises(ValueError, match="stages.json not found"):
        _coupling_stage_hours(case_dir)


def test_coupling_stage_hours_empty_stages_raises(tmp_path: Path) -> None:
    """An empty ``stages`` list has no coupling stage to read hours from."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_stages(case_dir, [])

    with pytest.raises(ValueError, match="carries no stages"):
        _coupling_stage_hours(case_dir)


def test_coupling_stage_hours_nonpositive_raises(tmp_path: Path) -> None:
    """A final stage with no positive block hours (e.g. an empty ``blocks``)
    cannot scale the cut coefficients — a hard error, not a divide-by-zero."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_stages(case_dir, [{"blocks": []}])

    with pytest.raises(ValueError, match="non-positive total block hours"):
        _coupling_stage_hours(case_dir)


def test_emit_import_diagnostics_gnl_deviation_dropped_count_includes_uncovered_lane() -> (
    None
):
    """AC 4 — the ``boundary-fcf-gnl-anticipated-deviation`` diagnostic's
    dropped-coverage count includes a covered-lane drop produced by a
    ``post_horizon_start`` that makes a target's only dated slot uncovered
    -- read straight from ``mapping.gnl_dropped``, never recomputed
    independently in the diagnostic.
    """
    id_map = make_id_map(())
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # unrelated dummy, satisfies the guard
            make_slot(_ANTICIPATED_THERMAL_STATE, 94, 0, delivery_date=20260401),
        ]
    )
    header = make_cortes_header(
        (), lag_maximo_gnl=2, n_patamares=3, submercado_codes=(1,)
    )
    # col(1,p,2) for p=1..3 -> flat indices 1, 3, 5 (P=3, L=2).
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    # thermal 94's only dated slot (20260401) is before this post_horizon_start
    # (20260501) -- non-covered, so the mapper drops it rather than placing it.
    gnl_plan = GnlRingPlan(
        (GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),),
        post_horizon_start=20260501,
    )

    mapping = map_boundary_cuts(
        cuts, manifest, id_map, cost_unit_hours=MONTH_HOURS, gnl_plan=gnl_plan
    )
    assert any(
        term.thermal_id == 94 and "post-study horizon" in term.reason
        for term in mapping.gnl_dropped
    )
    assert mapping.cuts[0].coefficients[1] == 0.0  # dropped, stays at 0.0

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    assert "1 GNL term(s) dropped" in deviation.summary
    # ticket-013 Requirement C.2: no per-row `Dropped` column; the count
    # above is the only place it is reported.
    assert deviation.table.columns == [
        "Submercado",
        "Lag",
        "Σ pi_gnl (carried)",
        "Patamar spread",
    ]
    assert all(len(row) == 4 for row in deviation.table.rows)


def test_emit_import_diagnostics_c1_panel1_notes_deduped() -> None:
    """Ticket-013 AC C.1 — Panel 1's `notes` no longer restate `summary`.

    `lag_nonzero_by_depth` is the one fact `notes` used to carry that
    `summary` did not already state; folded into `summary`, `notes` is now
    empty rather than restating `n_active_cuts`/`storage_nonzero_plants`/
    `rhs_min`/`rhs_max`.
    """
    cuts = make_boundary_cuts((1,), (make_cut_record(pi_varm=(1.5,), rhs=10.0),))
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(1.5,), intercept=10.0),), dropped=()
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping)

    summary_diagnostic = next(
        d for d in sink if d.code == "boundary-fcf-cut-family-summary"
    )
    summary = summarize_cut_families(cuts)
    assert summary_diagnostic.notes == []
    assert str(summary.lag_nonzero_by_depth) in summary_diagnostic.summary
    for stale_bullet in (
        f"n_active_cuts={summary.n_active_cuts}",
        f"storage_nonzero_plants={summary.storage_nonzero_plants}",
        f"rhs_min={summary.rhs_min!r}",
        f"rhs_max={summary.rhs_max!r}",
    ):
        assert stale_bullet not in summary_diagnostic.notes


def test_emit_import_diagnostics_c2_panel3_no_dropped_column() -> None:
    """Ticket-013 AC C.2 — Panel 3's table drops the constant `Dropped`
    column; the dropped count stays in `summary` only.
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
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan((GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),))
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    assert deviation.table.columns == [
        "Submercado",
        "Lag",
        "Σ pi_gnl (carried)",
        "Patamar spread",
    ]
    assert deviation.table.justify == ["right", "right", "right", "right"]
    assert "0 GNL term(s) dropped" in deviation.summary


def test_emit_import_diagnostics_c3_headline_excludes_near_zero_sum_group() -> None:
    """Ticket-013 AC C.3 — a noisy group's inflated relative spread does not
    dominate the `max_spread` HEADLINE; the weight-carrying group's spread
    does, and an absolute spread is reported alongside.

    Uses the REAL-DECK magnitudes cited by the epic-04 review that exposed
    the earlier fixed-magnitude floor's bug (`1e-6` failed to exclude a
    `4e-05` group, since `4e-05 >= 1e-6`): group `(submercado 1, lag 1)`
    carries the real weight (`Σ=-4412.0`, relative spread `392/4412 ≈
    0.0888`, matching the observed `~0.09`); group `(submercado 2, lag 1)`
    is the noisy one (`Σ=-4e-05`, relative spread `0.25`, matching the
    observed `0.25` exactly). `_GNL_DEVIATION_REL_FLOOR` (`1e-3`, a
    *fraction of the panel's own max |Σ|*) excludes the noisy group from
    the headline (`4e-05 < 1e-3 * 4412.0`) even though its own relative
    spread is far larger than the weight-carrying group's; the per-row
    table still carries both groups' own values unfiltered.
    """
    header = make_cortes_header(
        (), lag_maximo_gnl=1, n_patamares=3, submercado_codes=(1, 2)
    )
    # col(s,p,1) = (s-1)*3 + (p-1): submercado 1 -> cols 0,1,2; submercado 2
    # -> cols 3,4,5 (lag_maximo_gnl=1 collapses the lag axis to a single
    # column per patamar).
    # Group 1 (weight-carrying): sum -1340-1340-1732 = -4412.0; abs spread
    # -1340-(-1732) = 392.0; relative 392/4412 ≈ 0.0888 (observed ~0.09).
    # Group 2 (noisy): sum -1e-05-1e-05-2e-05 = -4e-05; abs spread
    # -1e-05-(-2e-05) = 1e-05; relative 1e-05/4e-05 = 0.25 (observed exactly).
    pi_gnl = (-1340.0, -1340.0, -1732.0, -1e-05, -1e-05, -2e-05)
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=0.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan(
        (
            GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=1),
            GnlThermalTarget(thermal_id=95, submercado=2, nl_lag=1),
        )
    )
    mapping = MappingResult(
        cuts=(make_mapped_cut(coefficients=(0.0,), intercept=0.0),),
        dropped=(),
        gnl_dropped=(),
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.table is not None
    rows_by_group = {(row[0], row[1]): row for row in deviation.table.rows}
    # `round(..., 6)` in the table rendering can perturb the raw ratio by up
    # to 5e-7, so use an absolute tolerance rather than the (much tighter)
    # relative default.
    assert rows_by_group[(1, 1)][3] == pytest.approx(392.0 / 4412.0, abs=1e-6)
    assert rows_by_group[(2, 1)][3] == pytest.approx(0.25, abs=1e-6)

    # The headline is the weight-carrying group's relative spread (~0.0888)
    # and the panel's overall absolute spread (392, from the same group) —
    # never the noisy group's inflated 0.25 relative figure, even though
    # 0.25 > 0.0888 on its own.
    expected_relative = f"{392.0 / 4412.0:.4g}"
    expected_absolute = f"{392.0:.4g}"
    assert (
        f"spread {expected_relative} relative / {expected_absolute} absolute"
        in deviation.summary
    )


def test_emit_import_diagnostics_c4_no_remediation_footer() -> None:
    """Ticket-013 AC C.4 — Panel 3 no longer carries a `remediation`
    footer; the C12 ledger row (`~/git/cobre/plans/conversion-found-
    improvements.md`) is the record now, not a runtime paragraph.
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
    pi_gnl = _gnl_row(6, {1: 0.1, 3: 0.2, 5: 0.3})
    record = make_cut_record(pi_varm=(), pi_gnl=pi_gnl, rhs=5.0)
    cuts = BoundaryCuts(header=header, boundary_stage=10, records=(record,))
    gnl_plan = GnlRingPlan((GnlThermalTarget(thermal_id=94, submercado=1, nl_lag=2),))
    mapping = map_boundary_cuts(
        cuts,
        manifest,
        id_map,
        cost_unit_hours=MONTH_HOURS,
        gnl_plan=gnl_plan,
        coupling_block_hours=_UNIFORM_GNL_BLOCK_HOURS,
    )

    with dx.collect() as sink:
        _emit_import_diagnostics(cuts, mapping, gnl_plan)

    deviation = next(
        d for d in sink if d.code == "boundary-fcf-gnl-anticipated-deviation"
    )
    assert deviation.remediation is None


# The real, gitignored `decomp-mar-26-rv2` deck (the empirical READBACK deck) +
# the local cobre **develop** build. The boundary-pricing / anticipated-
# reconciliation work has landed in develop (commits `abf73bf1` reconcile-by-
# dated-fan-out, `7faed7a0` exempt post-horizon anticipated deliveries from the
# lead-horizon cap), so the develop build is the correct target. Under the
# mirror-shift GNL emission the anticipation lead is capped strictly below
# the horizon (TRACKED COBRE-GAP WORKAROUND C13,
# `decomp/anticipated.py`), so this deck loads with no K=0 / no dropped deliveries.
# CI has neither the deck nor the binary, so this stays a dev-only tier-3 smoke.
_MAR26_DECK = Path("example/decomp-mar-26-rv2")
_MAR26_CORTESH = _MAR26_DECK / "cortesh.dat"
_MAR26_CORTES = _MAR26_DECK / "cortes-004.dat"
_MAR26_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"
_HAS_MAR26_E2E_DEPS = (
    _MAR26_COBRE_BIN.exists() and _MAR26_DECK.exists() and _HAS_WRITER_BINDING
)
_MAR26_SKIP_REASON = (
    f"requires the local cobre develop build ({_MAR26_COBRE_BIN}), the "
    f"decomp-mar-26-rv2 deck ({_MAR26_DECK}), and the write_policy_checkpoint "
    "writer binding"
)
_skip_mar26_e2e = pytest.mark.skipif(not _HAS_MAR26_E2E_DEPS, reason=_MAR26_SKIP_REASON)


@dataclass(frozen=True)
class _Mar26ImportedCase:
    """The mar-26-rv2 case, boundary-imported once, shared by the tests below."""

    case_dir: Path
    boundary_dir: Path


@pytest.fixture(scope="module")
def mar26rv2_imported_case(
    tmp_path_factory: pytest.TempPathFactory,
) -> _Mar26ImportedCase:
    """Convert ``decomp-mar-26-rv2`` and import its own boundary FCF, once.

    Module-scoped and shared by both tests below: the ~20 s convert +
    bootstrap pass runs exactly once regardless of how many assertions
    exercise its result. Neither consuming test
    mutates this fixture's ``case_dir`` in place -- the run-load test copies
    it into its own scratch directory first -- so test order never matters.
    """
    if not _HAS_MAR26_E2E_DEPS:
        pytest.skip(_MAR26_SKIP_REASON)

    root = tmp_path_factory.mktemp("fcf_importer_mar26rv2_e2e")
    case_dir = root / "converted"
    fcf_inputs = FcfInputs()
    convert_decomp_case(_MAR26_DECK, case_dir, force=True, fcf_inputs_out=fcf_inputs)

    # Pin the explicitly-named `cortes-004.dat` partition this fixture has
    # always imported, overriding whatever cortes partition discovery
    # resolves for this deck (`DecompFiles` is frozen, so `replace` rebuilds
    # it rather than mutating in place).
    case = DecompCase.from_directory(_MAR26_DECK)
    case.files = replace(case.files, cortesh=_MAR26_CORTESH, cortes=_MAR26_CORTES)

    boundary_dir = import_boundary_fcf(
        case_dir,
        case,
        work_dir=root / "work",
        cost_scale_factor=1.0,
        config=fcf_inputs.config,
        initial_conditions=fcf_inputs.initial_conditions,
    )
    assert boundary_dir is not None
    return _Mar26ImportedCase(case_dir=case_dir, boundary_dir=boundary_dir)


def _as_text(raw: str | bytes | None) -> str:
    """Decode a subprocess capture (``TimeoutExpired`` yields bytes even under
    ``text=True``) to ``str``; ``None`` → ``""``."""
    if raw is None:
        return ""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw


def _dated_ring_position(
    entity_manifest: list[dict[str, object]],
    thermal_id: int,
    *,
    covered: bool,
    post_horizon_start: int,
) -> int:
    """The position of `thermal_id`'s covered/non-covered dated ring slot.

    Scans a reloaded checkpoint's `entity_manifest` (never hardcoding a
    position, since that shifts whenever an unrelated converter output
    changes the case's state dimension) for the `AnticipatedThermalState`
    slot matching `thermal_id` whose `delivery_date` is dated (not the
    sentinel) and on the requested side of `post_horizon_start`.
    """
    for position, slot in enumerate(entity_manifest):
        delivery_date = slot["delivery_date"]
        if not isinstance(delivery_date, int):
            continue
        if (
            slot["entity_type"] == _ANTICIPATED_THERMAL_STATE
            and slot["entity_id"] == thermal_id
            and delivery_date != -2147483648
            and (delivery_date >= post_horizon_start) == covered
        ):
            return position
    raise AssertionError(
        f"no {'covered' if covered else 'non-covered'} dated ring slot for "
        f"thermal {thermal_id} in the reloaded entity_manifest"
    )


@_skip_mar26_e2e
def test_import_boundary_fcf_mar26rv2_covered_lane_and_case_validates(
    mar26rv2_imported_case: _Mar26ImportedCase,
) -> None:
    """Post-mirror-shift covered-lane + calendar + validate facts (ticket-006
    AC4/AC5/AC7).

    Under the mirror-shift GNL emission every anticipated plant gets one free
    forward delivery per study stage on the (C13-capped, H=1008) mirror calendar,
    so BOTH thermal 94 and thermal 95 now have covered dated ring slots and
    NEITHER has a non-covered dated slot -- superseding the old thermal-95
    non-covered `20260401` / `== 0.0` assertion (that pre-horizon April slot no
    longer exists). thermal 94's covered lane is priced (nonzero, finite); the
    exact GAP-1 hours-weighted collapse `Σ_p pi_gnl[c]·h_p` is pinned by the
    tier-1 mapper tests, so here the real-deck fact is that the lane carries a
    real coefficient. The post-study calendar is the mirror shape. `cobre
    validate`'s CASE-structural check (the first `Validation: N errors` line it
    prints) reports `0`; its aggregate exit is C8-gated (a second pass looks for
    the checkpoint at `case_dir/output/boundary`), so this asserts on the
    structural line's own count, never the process exit code.
    """
    case_dir = mar26rv2_imported_case.case_dir
    import cobre

    horizon_start = _post_horizon_start(case_dir)
    assert horizon_start is not None

    policy = cobre.results.load_policy(case_dir, policy_subdir="boundary")
    terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])
    entity_manifest = terminal["entity_manifest"]
    first_cut = terminal["cuts"][0]

    # AC4/AC5: BOTH plants now have a covered (post-horizon May) dated ring slot
    # priced nonzero -- the core fix: PSERGIPE I (95) was non-covered / 0.0
    # pre-mirror and now carries a real coefficient, exactly like thermal 94. The
    # exact GAP-1 hours-weighted collapse is pinned by the tier-1 mapper tests;
    # the real-deck fact asserted here is that the free-only lane is priced. The
    # residual non-covered slot each plant still carries (the reconciliation's
    # in-study April ring slot, `_post_horizon_start` = 20260501) stays 0.0.
    for tid in (94, 95):
        covered = _dated_ring_position(
            entity_manifest, tid, covered=True, post_horizon_start=horizon_start
        )
        coeff = first_cut["coefficients"][covered]
        assert coeff != 0.0, f"thermal {tid}'s covered ring lane is unpriced"
        assert math.isfinite(coeff)

        non_covered = _dated_ring_position(
            entity_manifest, tid, covered=False, post_horizon_start=horizon_start
        )
        assert first_cut["coefficients"][non_covered] == 0.0

    # AC7: the C13-capped H=1008 mirror calendar (study stages shifted below the
    # horizon): a 24 h Fri->Sat stub, two Saturday weekly stages, one monthly.
    post_study = json.loads(
        (case_dir / "post_study_stages.json").read_text(encoding="utf-8")
    )
    assert [s["start_date"] for s in post_study["stages"]] == [
        "2026-05-01",
        "2026-05-02",
        "2026-05-09",
        "2026-05-16",
    ]
    assert [s["duration_hours"] for s in post_study["stages"]] == [
        24.0,
        168.0,
        168.0,
        648.0,
    ]

    completed = subprocess.run(
        [str(_MAR26_COBRE_BIN), "validate", str(case_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    case_validation = re.search(r"Validation: (\d+) errors", completed.stdout)
    assert case_validation is not None, (
        f"no 'Validation: N errors' line in stdout:\n{completed.stdout}"
    )
    assert case_validation.group(1) == "0", (
        f"case-structural validation reported {case_validation.group(1)} "
        f"error(s):\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


@_skip_mar26_e2e
def test_import_boundary_fcf_mar26rv2_run_loads_boundary(
    mar26rv2_imported_case: _Mar26ImportedCase, tmp_path: Path
) -> None:
    """The bounded real ``cobre run`` LOADS the terminal boundary (ticket-006
    AC3, RESOLVED under the mirror-shift emission).

    Pre-mirror this xfailed: the per-plant emission left an anticipated ring
    slot at a K=0 sub-stage lead that cobre's boundary-load rejected. The
    mirror-shift emission (single global lead, C13-capped strictly below the
    horizon) makes every delivery's decider land in-study, so a fresh
    ``cobre run <dst> --output <dst>`` (the C8 recipe, 1-iteration) loads the
    boundary cleanly: no LP-builder panic, no "no resolved delivery interval",
    no K=0 warning, no dropped ``future_anticipated_deliveries``.

    Everything asserted here is printed at setup / boundary-load, which is fast;
    the subsequent SDDP iteration over the terminal leaf is expensive, so the
    run is given a wall budget and, if it is still iterating when that elapses,
    the output captured so far is what the assertions inspect -- reaching
    iteration with the boundary loaded and no panic/rejection IS the proof. If
    it finishes inside the budget, its exit is additionally required to be 0.

    Copies the shared fixture's case into a private scratch directory first
    (never mutating ``mar26rv2_imported_case.case_dir`` in place).
    """
    scratch_case = tmp_path / "case"
    shutil.copytree(mar26rv2_imported_case.case_dir, scratch_case)

    config_path = scratch_case / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 1}]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    argv = [
        str(_MAR26_COBRE_BIN),
        "run",
        str(scratch_case),
        "--output",
        str(scratch_case),
    ]
    timed_out = False
    returncode: int | None
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=60
        )
        combined = completed.stdout + completed.stderr
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        # Setup + boundary-load finish well before the slow iteration; a timeout
        # kill means we reached iteration -- inspect the setup output captured.
        # (TimeoutExpired carries raw bytes even under text=True, so decode.)
        timed_out = True
        combined = _as_text(exc.stdout) + _as_text(exc.stderr)
        returncode = None

    assert "Cuts loaded" in combined, (
        f"cobre run did not report a boundary load (exit {returncode}):\n{combined}"
    )
    for forbidden in (
        "panic",
        "attempt to calculate the remainder with a divisor of zero",
        "no resolved delivery interval",
        "K=0 sub-stage lead",
        "dropped — out of the lead's reach",
    ):
        assert forbidden not in combined, (
            f"cobre run boundary-load produced {forbidden!r} (exit {returncode}):"
            f"\n{combined}"
        )
    if not timed_out:
        assert returncode == 0, f"cobre run failed (exit {returncode}):\n{combined}"


@_skip_mar26_e2e
def test_convert_decomp_boundary_fcf_cli_mar26rv2_authors_populated_boundary(
    tmp_path: Path,
) -> None:
    """Ticket-012 -- the real operator CLI command, driven as a subprocess.

    Neither ticket-008 (CLI wiring, ``import_boundary_fcf`` mocked) nor
    ticket-013 (the two tests above, which call the real importer as a
    *library* function) exercises the actual, documented
    ``cobre-bridge convert decomp`` command (boundary FCF on by default) end to end.
    This test closes that seam: it drives the command as a subprocess
    (``sys.executable -m cobre_bridge.cli``, hermetic against PATH) on the
    real mar-26-rv2 deck with the local cobre develop binary, and proves the
    CLI-surface facts the library-level tests cannot -- exit 0, the authored
    ``boundary/`` + ``config.json`` wiring, the ``--json`` verdict's
    ``summary["boundary_fcf"]``, and, via a coarse reload of the authored
    checkpoint, that the CLI path produced a *populated* GNL ring rather than
    all zeros. It deliberately does not re-run ``cobre validate`` or
    ``cobre run``, and does not re-assert the per-thermal covered/uncovered
    placement identity -- both stay owned by the two ticket-013 tests above.
    """
    dst = tmp_path / "converted"

    # The boundary FCF is imported by default now (the deck declares its cut
    # files), so no flag is needed; the bootstrap runs in-process via
    # cobre-python, so there is no --cobre-bin to pass.
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cobre_bridge.cli",
            "convert",
            "decomp",
            str(_MAR26_DECK),
            str(dst),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"convert decomp (default boundary FCF) exited {completed.returncode}:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )

    assert (dst / "boundary" / "manifest.bin").is_file()
    config = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    assert config["policy"]["boundary"] == {"path": "boundary", "source_stage": 4}

    doc = json.loads(completed.stdout)
    assert doc["summary"]["boundary_fcf"] == {
        "imported": True,
        "path": "boundary",
        "run_constraint": f"--output={dst}",
    }

    import cobre

    policy = cobre.results.load_policy(dst, policy_subdir="boundary")
    terminal = max(policy["stage_cuts"], key=lambda stage: stage["stage_id"])
    entity_manifest = terminal["entity_manifest"]
    first_cut = terminal["cuts"][0]
    assert any(
        slot["entity_type"] == _ANTICIPATED_THERMAL_STATE
        and first_cut["coefficients"][position] != 0.0
        for position, slot in enumerate(entity_manifest)
    ), "no nonzero AnticipatedThermalState coefficient in the terminal cut"
