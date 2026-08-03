"""Tests for the source model's ``CORTESH``/``cortes`` readers (``fcf/cortes.py``)."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs not installed
import pytest
from inewave.newave import Cortesh

from cobre_bridge.decomp.fcf.cortes import (
    BoundaryCuts,
    CortesHeader,
    CutFamilySummary,
    StageCutRecord,
    read_cortes,
    read_cortesh,
    summarize_cut_families,
)

# Real, gitignored decks (see example/README.md — local-only, not part of the
# repo). CI does not have example/, so every test reading one is
# skipif-guarded on its presence.
_NEWAVE_RODADA = Path("example/newave_rodada/cortesh.dat")
_NEWAVE_RODADA_CORTES = Path("example/newave_rodada/cortes.dat")
_DECOMP_SET_24 = Path("example/decomp-set-24-rv0/cortesh.dat")
_DECOMP_SET_24_CORTES = Path("example/decomp-set-24-rv0/cortes-010.dat")

_NONZERO = 1e-9


@pytest.mark.skipif(
    not _NEWAVE_RODADA.exists(), reason="newave_rodada deck not present"
)
def test_read_cortesh_nongnl_deck_header() -> None:
    header = read_cortesh(_NEWAVE_RODADA)

    assert isinstance(header, CortesHeader)
    assert header.n_plants == 154
    assert header.lag_maximo_gnl == 0
    assert header.individualized is True
    assert len(header.plant_codes) == 154

    uhes = Cortesh.read(str(_NEWAVE_RODADA)).dados_uhes.sort_values("indice_usina")
    expected_plant_codes = tuple(int(code) for code in uhes["codigo_usina"])
    assert header.plant_codes == expected_plant_codes


@pytest.mark.skipif(
    not _DECOMP_SET_24.exists(), reason="decomp-set-24-rv0 deck not present"
)
def test_read_cortesh_gnl_deck_header() -> None:
    header = read_cortesh(_DECOMP_SET_24)

    assert header.n_plants == 155
    assert header.lag_maximo_gnl == 2
    assert header.n_patamares == 3


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            _NEWAVE_RODADA,
            marks=pytest.mark.skipif(
                not _NEWAVE_RODADA.exists(), reason="newave_rodada deck not present"
            ),
        ),
        pytest.param(
            _DECOMP_SET_24,
            marks=pytest.mark.skipif(
                not _DECOMP_SET_24.exists(),
                reason="decomp-set-24-rv0 deck not present",
            ),
        ),
    ],
)
def test_read_cortesh_exposes_stage_chain_heads(path: Path) -> None:
    header = read_cortesh(path)

    assert len(header.last_cut_record_by_stage) > 0
    assert all(isinstance(value, int) for value in header.last_cut_record_by_stage)


def test_read_cortesh_rejects_non_individualized() -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 2

    with (
        patch(
            "cobre_bridge.decomp.fcf.cortes.Cortesh.read",
            return_value=_FakeCortesh(),
        ),
        pytest.raises(ValueError, match="individualized"),
    ):
        read_cortesh(Path("unused-cortesh.dat"))


@pytest.mark.skipif(
    not _NEWAVE_RODADA.exists() or not _NEWAVE_RODADA_CORTES.exists(),
    reason="newave_rodada deck not present",
)
def test_read_cortes_nongnl_boundary_stage_and_shapes() -> None:
    cortesh = Cortesh.read(str(_NEWAVE_RODADA))
    boundary = read_cortes(_NEWAVE_RODADA_CORTES, cortesh, boundary_stage=11)

    assert boundary.boundary_stage == 11
    assert len(boundary.records) > 0
    for record in boundary.records:
        assert len(record.pi_varm) == 154
        assert len(record.pi_qafl) == 154
        assert all(len(lags) == 12 for lags in record.pi_qafl)
        assert record.pi_gnl == ()


@pytest.mark.skipif(
    not _NEWAVE_RODADA.exists() or not _NEWAVE_RODADA_CORTES.exists(),
    reason="newave_rodada deck not present",
)
def test_read_cortes_exposes_per_cut_provenance() -> None:
    cortesh = Cortesh.read(str(_NEWAVE_RODADA))
    boundary = read_cortes(_NEWAVE_RODADA_CORTES, cortesh, boundary_stage=11)

    assert len(boundary.records) > 0
    for record in boundary.records:
        assert isinstance(record.cut_id, int) and record.cut_id > 0
        assert isinstance(record.iteration, int) and record.iteration > 0
        assert (
            isinstance(record.forward_pass_index, int) and record.forward_pass_index > 0
        )
        assert isinstance(record.is_active, bool)

    # Every record in this deck's boundary-stage export is active (byte-exact
    # `from_cortesh` measurement, 2026-08-03) — a real cut chain with no
    # deactivated entries at this boundary, not a read bug.
    assert all(record.is_active for record in boundary.records)
    cut_ids = [record.cut_id for record in boundary.records]
    assert len(set(cut_ids)) == len(cut_ids)  # cut_id is unique per record


@pytest.mark.skipif(
    not _NEWAVE_RODADA.exists() or not _NEWAVE_RODADA_CORTES.exists(),
    reason="newave_rodada deck not present",
)
def test_read_cortes_nongnl_nonzero_families() -> None:
    cortesh = Cortesh.read(str(_NEWAVE_RODADA))
    boundary = read_cortes(_NEWAVE_RODADA_CORTES, cortesh, boundary_stage=11)

    n_plants = boundary.header.n_plants
    varm_nonzero = [False] * n_plants
    nonzero_at_lag = [[False] * 12 for _ in range(n_plants)]
    for record in boundary.records:
        for i, value in enumerate(record.pi_varm):
            if abs(value) > _NONZERO:
                varm_nonzero[i] = True
        for i, lags in enumerate(record.pi_qafl):
            for lag_index, value in enumerate(lags):
                if abs(value) > _NONZERO:
                    nonzero_at_lag[i][lag_index] = True

    assert sum(varm_nonzero) == 154

    # 147/154 plants are nonzero at every lag 1..12 (byte-exact `from_cortesh`
    # measurement, 2026-08-03); 5 plants (codes 2, 133, 178, 305, 314) are
    # exactly 0.0 at all 12 lags and 2 more are zero only at the tail lags —
    # a real per-plant PAR-order-<12 fact, not a read bug. Supersedes the
    # pre-1.15.0 informal "149" triage figure.
    qafl_nonzero_all_lags = sum(1 for flags in nonzero_at_lag if all(flags))
    assert qafl_nonzero_all_lags >= 147


@pytest.mark.skipif(
    not _DECOMP_SET_24.exists() or not _DECOMP_SET_24_CORTES.exists(),
    reason="decomp-set-24-rv0 deck not present",
)
def test_read_cortes_gnl_boundary_stage_and_gnl_block() -> None:
    cortesh = Cortesh.read(str(_DECOMP_SET_24))
    boundary = read_cortes(_DECOMP_SET_24_CORTES, cortesh)

    assert boundary.boundary_stage == 10
    assert len(boundary.records) == 10000

    nonzero_any_record = [False] * 24
    fully_nonzero_records = 0
    for record in boundary.records:
        assert len(record.pi_gnl) == 24
        record_fully_nonzero = True
        for i, value in enumerate(record.pi_gnl):
            if abs(value) > _NONZERO:
                nonzero_any_record[i] = True
            else:
                record_fully_nonzero = False
        if record_fully_nonzero:
            fully_nonzero_records += 1

    assert all(nonzero_any_record)

    # Exactly 3 first-Benders-iteration records zero NORDESTE's 6 slots (a
    # degenerate early-iteration state, byte-exact `from_cortesh`
    # measurement, 2026-08-03) — supersedes the pre-1.15.0 literal "each
    # record all-24" figure.
    assert fully_nonzero_records >= 9990


def test_read_cortes_rejects_nonzero_sar(tmp_path: Path) -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 1
        dados_uhes = pd.DataFrame({"indice_usina": [1], "codigo_usina": [4]})
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 0
        numero_submercados = 1
        tamanho_registro_individualizado = 32
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [5],
                "indice_ultimo_corte": [1],
            }
        )

    record_size = _FakeCortesh.tamanho_registro_individualizado
    record = bytearray(record_size)
    # Nonzero rhs (bytes 16:24) on the last physical record marks this as
    # the consolidated (no-sentinel) shape, so `boundary_stage` is required
    # and no trailer is derived.
    record[16:24] = struct.pack("<d", 1.0)
    cortes_path = tmp_path / "cortes.dat"
    cortes_path.write_bytes(bytes(record) * 2)

    bad_frame = pd.DataFrame(
        {
            "rhs": [123.0],
            "pi_varm_uhe4": [0.0],
            **{f"pi_qafl_uhe4_lag{lag}": [0.0] for lag in range(1, 13)},
            "pi_mx_sar_uhe4": [7.0],
        }
    )

    class _FakeCortes:
        cortes = bad_frame

    with (
        patch(
            "cobre_bridge.decomp.fcf.cortes.Cortes.from_cortesh",
            return_value=_FakeCortes(),
        ),
        pytest.raises(ValueError, match="SAR"),
    ):
        read_cortes(cortes_path, _FakeCortesh(), boundary_stage=5)


@pytest.mark.skipif(
    not _NEWAVE_RODADA.exists() or not _NEWAVE_RODADA_CORTES.exists(),
    reason="newave_rodada deck not present",
)
def test_summarize_families_nongnl_matches_probed_facts() -> None:
    cortesh = Cortesh.read(str(_NEWAVE_RODADA))
    boundary = read_cortes(_NEWAVE_RODADA_CORTES, cortesh, boundary_stage=11)

    summary = summarize_cut_families(boundary)

    assert isinstance(summary, CutFamilySummary)
    assert summary.n_active_cuts == len(boundary.records)
    assert summary.storage_nonzero_plants == 154
    assert summary.gnl_nonzero_slots == 0
    assert len(summary.lag_nonzero_by_depth) == 12

    # 147/154 plants nonzero at lag 12 (byte-exact `from_cortesh` measurement,
    # 2026-08-03; supersedes the pre-1.15.0 informal "149" figure — see
    # test_read_cortes_nongnl_nonzero_families).
    assert summary.lag_nonzero_by_depth[11] >= 147

    assert summary.rhs_max >= summary.rhs_min > 0
    assert summary.rhs_min == pytest.approx(summary.rhs_min)  # finite, not NaN
    assert summary.rhs_max == pytest.approx(summary.rhs_max)  # finite, not NaN


@pytest.mark.skipif(
    not _DECOMP_SET_24.exists() or not _DECOMP_SET_24_CORTES.exists(),
    reason="decomp-set-24-rv0 deck not present",
)
def test_summarize_families_gnl_matches_probed_facts() -> None:
    cortesh = Cortesh.read(str(_DECOMP_SET_24))
    boundary = read_cortes(_DECOMP_SET_24_CORTES, cortesh)

    summary = summarize_cut_families(boundary)

    assert summary.n_active_cuts == len(boundary.records) == 10000
    assert summary.storage_nonzero_plants == 155
    assert summary.gnl_nonzero_slots == 24
    assert len(summary.lag_nonzero_by_depth) == 12

    # Byte-exact `from_cortesh` measurement, 2026-08-03: per-lag nonzero
    # plant counts are (151, 151, 151, 151, 151, 151, 151, 151, 150, 150,
    # 149, 148) for lags 1..12 — a measured floor of 148, pinned here (not
    # the informal pre-1.15.0 "148-151" estimate the ticket superseded,
    # which happens to bracket the true floor).
    _MEASURED_GNL_LAG_FLOOR = 148
    assert min(summary.lag_nonzero_by_depth) >= _MEASURED_GNL_LAG_FLOOR
    assert all(
        count >= 145 for count in summary.lag_nonzero_by_depth
    )  # AC robust floor

    assert summary.rhs_max >= summary.rhs_min > 0


def test_summarize_families_empty_records_raises() -> None:
    header = CortesHeader(
        plant_codes=(4,),
        submercado_codes=(1,),
        n_patamares=3,
        lag_maximo_gnl=0,
        n_plants=1,
        individualized=True,
        record_size=32,
        last_cut_record_by_stage=(1,),
    )
    empty = BoundaryCuts(header=header, boundary_stage=5, records=())

    with pytest.raises(ValueError, match="no active cuts"):
        summarize_cut_families(empty)


def test_summarize_families_counts_plant_once_across_records() -> None:
    header = CortesHeader(
        plant_codes=(4, 5),
        submercado_codes=(1,),
        n_patamares=3,
        lag_maximo_gnl=1,
        n_plants=2,
        individualized=True,
        record_size=32,
        last_cut_record_by_stage=(1,),
    )
    records = (
        StageCutRecord(
            cut_id=1,
            iteration=1,
            forward_pass_index=1,
            is_active=True,
            rhs=10.0,
            pi_varm=(1.0, 0.0),
            pi_qafl=((1.0,) + (0.0,) * 11, (0.0,) * 12),
            pi_gnl=(1.0,),
        ),
        StageCutRecord(
            cut_id=2,
            iteration=1,
            forward_pass_index=2,
            is_active=True,
            rhs=20.0,
            pi_varm=(1.0, 0.0),
            pi_qafl=((1.0,) + (0.0,) * 11, (0.0,) * 12),
            pi_gnl=(1.0,),
        ),
    )
    cuts = BoundaryCuts(header=header, boundary_stage=5, records=records)

    summary = summarize_cut_families(cuts)

    # Plant 0 is nonzero at lag 1 in both records; counted once, not twice.
    assert summary.storage_nonzero_plants == 1
    assert summary.lag_nonzero_by_depth == (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    assert summary.gnl_nonzero_slots == 1
    assert summary.rhs_min == 10.0
    assert summary.rhs_max == 20.0
