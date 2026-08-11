"""Unit tests for the source model's GNL reader (``decomp/anticipated.py``).

CI-tier and deck-independent: every test drives a synthetic ``_StubDadgnl`` with
hand-authored ``gl``/``tg``/``gs`` data (mirroring the real register shapes), so
the suite needs no deck and no cobre binary. ``gl`` is a list of register stubs
(the real ``gl(df=True)`` is unusable on ragged weekly blocks); ``tg``/``gs`` are
DataFrames (their fixed shapes make ``df=True`` well-formed).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cobre_bridge.decomp.anticipated import (
    GnlCommitment,
    GnlCommitmentModel,
    GnlStageCommitment,
    GnlThermal,
    convert_gnl,
    is_gnl_enabled,
    read_gnl_model,
)


class _GlReg:
    """A stand-in for one ``gl`` register (attribute access, like idecomp's)."""

    def __init__(
        self,
        codigo_usina: int,
        estagio: int,
        data_inicio: str | int,
        duracao: list[float],
        geracao: list[float],
        codigo_submercado: int = 1,
    ) -> None:
        self.codigo_usina = codigo_usina
        self.estagio = estagio
        self.data_inicio = data_inicio
        self.duracao = duracao
        self.geracao = geracao
        self.codigo_submercado = codigo_submercado


class _StubDadgnl:
    """Minimal ``Dadgnl`` stub: ``gl()`` → register list, ``tg``/``gs`` → frames."""

    def __init__(
        self,
        gl_registers: list[_GlReg],
        tg_rows: list[dict],
        gs_rows: list[dict],
    ) -> None:
        self._gl = gl_registers
        self._tg = pd.DataFrame(tg_rows)
        self._gs = pd.DataFrame(gs_rows)

    def gl(self, df: bool = False) -> list[_GlReg]:
        return list(self._gl)

    def tg(self, df: bool = False) -> pd.DataFrame:
        return self._tg

    def gs(self, df: bool = False) -> pd.DataFrame:
        return self._gs


def _tg_row(
    code: int,
    name: str,
    *,
    sub: int = 1,
    cvu: float = 199.22,
    disp: float = 500.0,
    inflex: float = 0.0,
) -> dict:
    """A ``tg`` registry row with equal block values (the common cadastro shape)."""
    return {
        "codigo_submercado": sub,
        "codigo_usina": code,
        "estagio": 1,
        "nome": name,
        "cvu_1": cvu,
        "cvu_2": cvu,
        "cvu_3": cvu,
        "disponibilidade_1": disp,
        "disponibilidade_2": disp,
        "disponibilidade_3": disp,
        "inflexibilidade_1": inflex,
        "inflexibilidade_2": inflex,
        "inflexibilidade_3": inflex,
    }


def test_committed_mw_is_block_duration_weighted() -> None:
    gl = [_GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [107.0, 131.0, 97.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    got = model.commitments[86].stages[0].committed_mw
    expected = (40 * 107 + 48 * 131 + 80 * 97) / 168
    assert abs(got - expected) < 1e-9
    assert model.commitments[86].stages[0].hours == 168.0  # sum of duracao


def test_gate_off_when_gl_absent() -> None:
    stub = _StubDadgnl([], [_tg_row(86, "SANTA CRUZ")], [])

    assert is_gnl_enabled(stub) is False
    assert read_gnl_model(stub) is None


def test_gate_off_when_all_geracao_zero() -> None:
    gl = [_GlReg(86, 1, "14032026", [40.0, 48.0, 80.0], [0.0, 0.0, 0.0])]
    stub = _StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], [])

    assert is_gnl_enabled(stub) is False
    assert read_gnl_model(stub) is None


def test_gate_off_for_none() -> None:
    assert is_gnl_enabled(None) is False


def test_thermals_ascending_by_code() -> None:
    # One nonzero gl register turns the gate on; the registry lists all tg plants.
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    tg = [
        _tg_row(224, "PSERGIPE I", sub=3),
        _tg_row(15, "X"),
        _tg_row(86, "SANTA CRUZ"),
    ]
    model = read_gnl_model(_StubDadgnl(gl, tg, []))

    assert model is not None
    assert [t.code for t in model.thermals] == [15, 86, 224]


def test_gl_code_absent_from_tg_raises_naming_code() -> None:
    gl = [_GlReg(999, 1, "14032026", [168.0], [100.0])]
    with pytest.raises(ValueError, match="999"):
        read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))


def test_registry_only_plant_has_empty_commitment() -> None:
    # code 86 carries the (enabling) gl dispatch; 224 is registry-only.
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    tg = [_tg_row(86, "SANTA CRUZ"), _tg_row(224, "PSERGIPE I", sub=3)]
    model = read_gnl_model(_StubDadgnl(gl, tg, []))

    assert model is not None
    assert model.commitments[224].stages == ()
    assert model.commitments[86].stages != ()


def test_post_horizon_delivery_date_and_mw_captured() -> None:
    gl = [_GlReg(86, 9, "09052026", [500.0, 500.0, 500.0], [500.0, 500.0, 500.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    stage = model.commitments[86].stages[0]
    assert stage.estagio == 9
    assert stage.start_date == date(2026, 5, 9)
    assert stage.committed_mw == 500.0


def test_data_inicio_integer_with_dropped_leading_zero() -> None:
    # A ddmmyyyy stored as an int loses its leading-zero day (04.. -> 4042026).
    gl = [_GlReg(86, 4, 4042026, [30.0, 69.0, 69.0], [10.0, 10.0, 10.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    assert model.commitments[86].stages[0].start_date == date(2026, 4, 4)


def test_registry_cost_and_bounds_from_tg() -> None:
    gl = [_GlReg(86, 1, "14032026", [30.0, 74.0, 64.0], [100.0, 100.0, 100.0])]
    tg = [_tg_row(86, "SANTA CRUZ", cvu=199.22, disp=500.0, inflex=0.0)]
    model = read_gnl_model(_StubDadgnl(gl, tg, []))

    assert model is not None
    thermal = model.thermals[0]
    assert thermal.name == "SANTA CRUZ"
    assert thermal.submarket_code == 1
    assert thermal.cost_per_mwh == pytest.approx(199.22)
    assert thermal.max_mw == pytest.approx(500.0)
    assert thermal.min_mw == 0.0


def test_weeks_per_month_from_gs() -> None:
    gl = [_GlReg(86, 1, "14032026", [168.0], [100.0])]
    gs = [{"mes": 1, "semanas": 3}, {"mes": 2, "semanas": 4}, {"mes": 3, "semanas": 5}]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], gs))

    assert model is not None
    assert model.weeks_per_month == {1: 3, 2: 4, 3: 5}
    assert isinstance(model, GnlCommitmentModel)


# --------------------------------------------------------------------------
# Emission (convert_gnl)
# --------------------------------------------------------------------------

# Two study stages, horizon ends 2026-05-01 (mirrors decomp-mar-26-rv2's shape).
_EMIT_STAGES = [
    {
        "start_date": "2026-03-14",
        "end_date": "2026-03-21",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-03-21",
        "end_date": "2026-05-01",
        "blocks": [{"hours": 984.0}],
    },
]
_BUS_OF = {1: 0, 3: 2}.get


def _emit_model() -> GnlCommitmentModel:
    """SANTA CRUZ (86): zero in-horizon + 500 MW post-horizon (2026-05-09);
    PSERGIPE I (224): registry-only, no committed delivery."""
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    pserg = GnlThermal(224, "PSERGIPE I", 3, 321.26, 0.0, 1593.0)
    return GnlCommitmentModel(
        thermals=(santa, pserg),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),
                    # Post-horizon weekly estágios; the last register's duracao is
                    # empty on real decks (hours=0), so the window comes from the
                    # 7-day estágio cadence (estágio 8 -> 9).
                    GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 500.0, 0.0),
                ),
            ),
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            ),
        },
        weeks_per_month={},
    )


def test_convert_gnl_creates_thermals_with_ids_and_anticipated_config() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    assert [(t["id"], t["name"]) for t in e.thermals] == [
        (94, "SANTA CRUZ"),
        (95, "PSERGIPE I"),
    ]
    santa = e.thermals[0]
    assert santa["bus_id"] == 0
    assert santa["cost_per_mwh"] == 199.22
    assert santa["generation"] == {"min_mw": 0.0, "max_mw": 500.0}
    # All in-horizon geracao is 0 -> K=1 -> lead = study stage 0 hours.
    assert santa["anticipated_config"] == {"lead_time_hours": 168.0}


def test_convert_gnl_left_boundary_mandatory_and_tiles_k_stages() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    # Both anticipated thermals get a commitment window (K=1 -> study stage 0),
    # zero MW since in-horizon geracao is 0.
    assert e.past_anticipated_commitments == [
        {
            "thermal_id": 94,
            "start_date": "2026-03-14",
            "end_date": "2026-03-21",
            "value_mw": 0.0,
        },
        {
            "thermal_id": 95,
            "start_date": "2026-03-14",
            "end_date": "2026-03-21",
            "value_mw": 0.0,
        },
    ]


def test_convert_gnl_right_boundary_delivery_and_post_study() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    # Only SANTA CRUZ's post-horizon 500 MW delivery; window is [start, start+7d).
    assert e.future_anticipated_deliveries == [
        {
            "thermal_id": 94,
            "delivery_start": "2026-05-09",
            "delivery_end": "2026-05-16",
            "min_mw": 500.0,
            "max_mw": 500.0,
        }
    ]
    pss = e.post_study_stages
    assert pss is not None
    # Break at horizon end + delivery boundaries: 05-01 | 05-09 | 05-16.
    assert [s["start_date"] for s in pss["stages"]] == ["2026-05-01", "2026-05-09"]
    # The delivery covers whole stage index 1; bound carries cvu + capability.
    assert pss["thermal_bounds"] == [
        {
            "thermal_id": 94,
            "post_study_stage_index": 1,
            "cost_per_mwh": 199.22,
            "min_mw": 0.0,
            "max_mw": 500.0,
        }
    ]


def test_convert_gnl_no_post_horizon_delivery_yields_no_post_study() -> None:
    # PSERGIPE-only model: registry, zero in-horizon, nothing post-horizon.
    pserg = GnlThermal(224, "PSERGIPE I", 3, 321.26, 0.0, 1593.0)
    model = GnlCommitmentModel(
        thermals=(pserg,),
        commitments={
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            )
        },
        weeks_per_month={},
    )
    e = convert_gnl(model, first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES)

    assert e.future_anticipated_deliveries == []
    assert e.post_study_stages is None
    assert len(e.past_anticipated_commitments) == 1  # left boundary still mandatory
