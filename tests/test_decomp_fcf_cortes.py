"""Tests for the source model's ``CORTESH``/``cortes`` readers (``fcf/cortes.py``).

Per H5 (deck-independent unit coverage), every *unit* test in this module
reads no real deck: the header-parse (``read_cortesh``), record-assembly
(``read_cortes``), and trailer-detection (``_read_trailer``) paths are
exercised against synthetic ``_FakeCortesh``/``_FakeCortes`` stand-ins and
tiny in-code ``struct.pack`` blobs. The ``@pytest.mark.skipif(...exists())``
tests below (pinning byte-exact facts from the real, gitignored decks under
``example/``) are **dev smoke only** — they exercise the full ~176 MB export
and never run in CI; they skip cleanly in a CI environment.
"""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import patch

import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs not installed
import pytest
from inewave.newave import Cortes, Cortesh

from cobre_bridge.decomp.fcf.cortes import (
    BoundaryCuts,
    CortesHeader,
    CutFamilySummary,
    StageCutRecord,
    _build_header,
    _read_trailer,
    read_cortes,
    read_cortesh,
    required_inflow_lag_depth,
    summarize_cut_families,
)

# Real, gitignored decks (see example/README.md — local-only, not part of the
# repo). CI does not have example/, so every test reading one is
# skipif-guarded on its presence.
_NEWAVE_RODADA = Path("example/newave_rodada/cortesh.dat")
_NEWAVE_RODADA_CORTES = Path("example/newave_rodada/cortes.dat")
_DECOMP_SET_24 = Path("example/decomp-set-24-rv0/cortesh.dat")
_DECOMP_SET_24_CORTES = Path("example/decomp-set-24-rv0/cortes-010.dat")
_DECOMP_MAR26 = Path("example/decomp-mar-26-rv2/cortesh.dat")
_DECOMP_MAR26_CORTES = Path("example/decomp-mar-26-rv2/cortes-004.dat")

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


@pytest.mark.skipif(
    not _DECOMP_MAR26.exists(), reason="decomp-mar-26-rv2 deck not present"
)
def test_read_cortesh_hybrid_deck_header() -> None:
    header = read_cortesh(_DECOMP_MAR26)

    # No `tipo_agregacao_caso`/contiguity raise: `read_cortesh` accepted this
    # hybrid deck (aggregated stages up to the individualized band, then
    # plant-space cuts) and excluded its fictitious plants.
    assert header.n_plants == 159
    assert len(header.plant_codes) == 159

    cortesh = Cortesh.read(str(_DECOMP_MAR26))
    expected_gnl_width = (
        cortesh.numero_submercados * header.n_patamares * header.lag_maximo_gnl
    )
    assert expected_gnl_width == 24


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


def test_build_header_rejects_boundary_stage_outside_individualized_band() -> None:
    """The per-stage gate names the individualized band it rejected against.

    Only the band attributes are needed on the fake: the gate check runs
    before any ``dados_uhes``/plant access, so the raise fires without ever
    touching the rest of the (deliberately absent) ``Cortesh`` surface.
    """

    class _FakeCortesh:
        estagio_individualizado_inicial = 3
        estagio_individualizado_final = 14

    with pytest.raises(ValueError, match=r"\[3, 14\]"):
        _build_header(_FakeCortesh(), boundary_stage=2)  # type: ignore[arg-type]


def _hybrid_fake_cortesh_169_plants_10_ficticia() -> object:
    """A synthetic hybrid ``Cortesh`` (mirrors production ``mar-26-rv2`` shape).

    169 plants total, 10 of them fictitious (indices 5, 12, 20, 33, 47, 58,
    69, 81, 95, 110 — deliberately including 5 and 12 so the non-fictitious
    subset has holes at both), leaving 159 non-fictitious. ``tipo_agregacao_caso
    == 0`` (hybrid) with an individualized band of ``[3, 14]``.
    """
    all_indices = list(range(1, 170))
    ficticia_indices = {5, 12, 20, 33, 47, 58, 69, 81, 95, 110}
    assert len(ficticia_indices) == 10

    class _FakeCortesh:
        tipo_agregacao_caso = 0
        numero_maximo_uhes = 169
        dados_uhes = pd.DataFrame(
            {
                "indice_usina": all_indices,
                "codigo_usina": [index * 1000 for index in all_indices],
                "ficticia": [
                    1 if index in ficticia_indices else 0 for index in all_indices
                ],
            }
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1, 2]})
        numero_patamares = 3
        lag_maximo_gnl = 2
        numero_submercados = 2
        tamanho_registro_individualizado = 1500
        estagio_individualizado_inicial = 3
        estagio_individualizado_final = 14
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [4],
                "indice_ultimo_corte": [1],
            }
        )

    return _FakeCortesh()


def test_build_header_hybrid_excludes_fictitious_plants() -> None:
    header = _build_header(
        _hybrid_fake_cortesh_169_plants_10_ficticia(),  # type: ignore[arg-type]
        boundary_stage=4,
    )

    assert header.n_plants == 159
    assert len(header.plant_codes) == 159
    assert header.individualized is True


def test_build_header_hybrid_tolerates_indice_usina_holes() -> None:
    # Fictitious plants sit at indice_usina 5 and 12 (among others); the
    # non-fictitious subset skips them without raising the old
    # "indice_usina is not a contiguous 1..N range" error.
    header = _build_header(
        _hybrid_fake_cortesh_169_plants_10_ficticia(),  # type: ignore[arg-type]
        boundary_stage=4,
    )

    # codigo_usina == indice_usina * 1000, so the excluded fictitious slots'
    # codes (5000, 12000) must be absent from the retained plant_codes, and
    # the retained codes stay in ascending indice_usina slot order.
    assert 5000 not in header.plant_codes
    assert 12000 not in header.plant_codes
    assert list(header.plant_codes) == sorted(header.plant_codes)


def test_read_cortesh_synthetic_preserves_slot_order() -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 3
        dados_uhes = pd.DataFrame(
            {
                "indice_usina": [1, 2, 3],
                "codigo_usina": [30, 10, 20],
                "ficticia": [0, 0, 0],
            }
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 2
        numero_submercados = 1
        tamanho_registro_individualizado = 112
        estagio_individualizado_inicial = 1
        estagio_individualizado_final = 12
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [10],
                "indice_ultimo_corte": [1],
            }
        )

    with patch(
        "cobre_bridge.decomp.fcf.cortes.Cortesh.read",
        return_value=_FakeCortesh(),
    ):
        header = read_cortesh(Path("unused-cortesh.dat"))

    assert isinstance(header, CortesHeader)
    # Slot order (indice_usina 1,2,3), not sorted by codigo_usina (30,10,20).
    assert header.plant_codes == (30, 10, 20)
    # No fictitious plants in this deck: n_plants/plant_codes equal the full
    # plant set — the fully-individualized, no-fictitious case is
    # regression-safe under the fictitious-exclusion + band-gate rework.
    assert header.n_plants == 3
    assert header.lag_maximo_gnl == 2
    assert header.individualized is True


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


@pytest.mark.skipif(
    not _DECOMP_MAR26.exists() or not _DECOMP_MAR26_CORTES.exists(),
    reason="decomp-mar-26-rv2 deck not present",
)
def test_read_cortes_hybrid_boundary_stage_and_shapes() -> None:
    cortesh = Cortesh.read(str(_DECOMP_MAR26))
    # Trailer-derived boundary stage (the single-stage export path, matching
    # the importer's own call), not a caller-supplied one.
    boundary = read_cortes(_DECOMP_MAR26_CORTES, cortesh, boundary_stage=None)

    assert boundary.boundary_stage == 4
    assert len(boundary.records) == 10000

    # `Cortes.from_cortesh` names the header's plant-space columns plus 4
    # per-cut provenance columns: 1 (rhs) + 159 (pi_varm) + 159*12
    # (pi_qafl) + 159 (pi_mx_sar) + 24 (pi_gnl) + 4 provenance == 2255.
    raw = Cortes.from_cortesh(str(_DECOMP_MAR26_CORTES), cortesh, por_estagio=True)
    assert raw.cortes is not None
    assert raw.cortes.shape == (10000, 2255)

    sar_cols = [f"pi_mx_sar_uhe{code}" for code in boundary.header.plant_codes]
    assert raw.cortes[sar_cols].abs().to_numpy().max() < _NONZERO


def test_read_cortes_rejects_nonzero_sar(tmp_path: Path) -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 1
        dados_uhes = pd.DataFrame(
            {"indice_usina": [1], "codigo_usina": [4], "ficticia": [0]}
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 0
        numero_submercados = 1
        tamanho_registro_individualizado = 32
        estagio_individualizado_inicial = 1
        estagio_individualizado_final = 12
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


def test_read_cortes_synthetic_records_and_trailer_stage(tmp_path: Path) -> None:
    class _FakeCortesh:
        tipo_agregacao_caso = 1
        numero_maximo_uhes = 1
        dados_uhes = pd.DataFrame(
            {"indice_usina": [1], "codigo_usina": [4], "ficticia": [0]}
        )
        dados_submercados = pd.DataFrame({"codigo_submercado": [1]})
        numero_patamares = 3
        lag_maximo_gnl = 0
        numero_submercados = 1
        tamanho_registro_individualizado = 32
        estagio_individualizado_inicial = 1
        estagio_individualizado_final = 12
        ultimo_registro_cortes_estagio = pd.DataFrame(
            {
                "tipo_estagio": ["estudo"],
                "estagio": [10],
                "indice_ultimo_corte": [1],
            }
        )
        ano_inicio_estudo = 2024

    record_size = _FakeCortesh.tamanho_registro_individualizado
    # The single physical record is the sentinel: leading int32[4] trailer
    # (study_start_month, study_start_year, cut_stage_month, cut_stage_year),
    # then rhs left at 0.0 (bytes [16:24] default-zeroed) so `_read_trailer`
    # recognizes it as the sentinel, not a real cut.
    sentinel = bytearray(record_size)
    sentinel[0:16] = struct.pack("<4i", 9, 2024, 10, 2024)
    cortes_path = tmp_path / "cortes-010.dat"
    cortes_path.write_bytes(bytes(sentinel))

    qafl_lags = tuple(float(lag) for lag in range(1, 13))
    frame = pd.DataFrame(
        {
            "rhs": [100.0],
            "indice_corte": [7],
            "iteracao_construcao": [3],
            "indice_forward": [2],
            "iteracao_desativacao": [0],
            "pi_varm_uhe4": [0.5],
            **{
                f"pi_qafl_uhe4_lag{lag}": [value]
                for lag, value in enumerate(qafl_lags, start=1)
            },
            "pi_mx_sar_uhe4": [0.0],
        }
    )

    class _FakeCortes:
        cortes = frame

    with patch(
        "cobre_bridge.decomp.fcf.cortes.Cortes.from_cortesh",
        return_value=_FakeCortes(),
    ):
        boundary = read_cortes(cortes_path, _FakeCortesh(), boundary_stage=None)

    # Trailer-derived: (cut_stage_year=2024 - ano_inicio_estudo=2024) * 12
    # + cut_stage_month=10 == 10.
    assert boundary.boundary_stage == 10
    assert len(boundary.records) == 1
    record = boundary.records[0]
    assert record.pi_varm == (0.5,)
    assert record.pi_qafl == (qafl_lags,)
    assert record.cut_id == 7
    assert record.iteration == 3
    assert record.forward_pass_index == 2
    assert record.is_active is True


def test_read_trailer_sentinel_vs_nonzero_rhs(tmp_path: Path) -> None:
    # A tiny record: int32[4] (16 bytes) + rhs float64 (8 bytes), no padding.
    record_size = 24

    sentinel_path = tmp_path / "sentinel.dat"
    sentinel_path.write_bytes(struct.pack("<4i", 1, 2, 3, 4) + struct.pack("<d", 0.0))

    nonzero_path = tmp_path / "nonzero.dat"
    nonzero_path.write_bytes(struct.pack("<4i", 1, 2, 3, 4) + struct.pack("<d", 5.0))

    assert _read_trailer(sentinel_path, record_size) == (1, 2, 3, 4)
    assert _read_trailer(nonzero_path, record_size) is None


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


@pytest.mark.skipif(
    not _DECOMP_MAR26.exists() or not _DECOMP_MAR26_CORTES.exists(),
    reason="decomp-mar-26-rv2 deck not present",
)
def test_summarize_families_hybrid_matches_probed_facts() -> None:
    cortesh = Cortesh.read(str(_DECOMP_MAR26))
    boundary = read_cortes(_DECOMP_MAR26_CORTES, cortesh, boundary_stage=None)

    summary = summarize_cut_families(boundary)

    assert summary.n_active_cuts == len(boundary.records) == 10000
    assert summary.storage_nonzero_plants == 159
    assert summary.gnl_nonzero_slots == 24
    assert len(summary.lag_nonzero_by_depth) == 12

    # RHS scale is on the order of 1e9-1e10 (byte-exact `from_cortesh`
    # measurement, 2026-08-12: rhs_max ~= 3.73e9) — an order-of-magnitude
    # tolerance so a benign scale nuance does not make the smoke brittle.
    assert summary.rhs_max >= summary.rhs_min > 0
    assert 1e9 <= summary.rhs_max <= 1e10


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


def _summary_with_lag_depths(lag_nonzero_by_depth: tuple[int, ...]) -> CutFamilySummary:
    """A ``CutFamilySummary`` whose only varying field is the lag-depth census."""
    return CutFamilySummary(
        n_active_cuts=1,
        storage_nonzero_plants=0,
        lag_nonzero_by_depth=lag_nonzero_by_depth,
        gnl_nonzero_slots=0,
        rhs_min=0.0,
        rhs_max=0.0,
    )


def test_required_inflow_lag_depth_returns_deepest_nonzero_depth() -> None:
    # Nonzero at calendar-month depths 1 and 3 (0-based indices 0 and 2): the
    # deepest referenced depth is 3, so 3 lag slots must be reserved.
    summary = _summary_with_lag_depths((2, 0, 1) + (0,) * 9)

    assert required_inflow_lag_depth(summary) == 3


def test_required_inflow_lag_depth_full_annual_depth() -> None:
    summary = _summary_with_lag_depths((1,) * 12)

    assert required_inflow_lag_depth(summary) == 12


def test_required_inflow_lag_depth_zero_when_no_lag_terms() -> None:
    # No cut carries a lag coefficient: no slots to reserve, no inflow_lag_depth
    # to declare (cobre rejects 0 and resolves a zero depth from an absent field).
    summary = _summary_with_lag_depths((0,) * 12)

    assert required_inflow_lag_depth(summary) == 0
