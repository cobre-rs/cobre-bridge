"""Unit tests for the source model's GNL reader (``decomp/anticipated.py``).

CI-tier and deck-independent: every test drives a synthetic ``_StubDadgnl`` with
hand-authored ``gl``/``tg``/``gs`` data (mirroring the real register shapes), so
the suite needs no deck and no cobre binary. ``gl`` is a list of register stubs
(the real ``gl(df=True)`` is unusable on ragged weekly blocks); ``tg``/``gs`` are
DataFrames (their fixed shapes make ``df=True`` well-formed).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import pytest

from cobre_bridge.decomp.anticipated import (
    GnlCommitment,
    GnlCommitmentModel,
    GnlStageCommitment,
    GnlThermal,
    _build_post_study_calendar,
    _calendar_stage_span,
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
        nl_rows: list[dict] | None = None,
    ) -> None:
        self._gl = gl_registers
        self._tg = pd.DataFrame(tg_rows)
        self._gs = pd.DataFrame(gs_rows)
        self._nl = pd.DataFrame(nl_rows or [])

    def gl(self, df: bool = False) -> list[_GlReg]:
        return list(self._gl)

    def tg(self, df: bool = False) -> pd.DataFrame:
        return self._tg

    def gs(self, df: bool = False) -> pd.DataFrame:
        return self._gs

    def nl(self, df: bool = False) -> pd.DataFrame:
        return self._nl


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


def test_nl_lags_read_per_plant() -> None:
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    tg = [_tg_row(86, "SANTA CRUZ"), _tg_row(224, "PSERGIPE I", sub=3)]
    nl = [
        {"codigo_usina": 86, "codigo_submercado": 1, "lag": 2},
        {"codigo_usina": 224, "codigo_submercado": 3, "lag": 2},
    ]
    model = read_gnl_model(_StubDadgnl(gl, tg, [], nl))

    assert model is not None
    assert model.nl_lag_months == {86: 2, 224: 2}


def test_nl_lags_empty_when_block_absent() -> None:
    gl = [_GlReg(86, 9, "09052026", [500.0], [500.0])]
    model = read_gnl_model(_StubDadgnl(gl, [_tg_row(86, "SANTA CRUZ")], []))

    assert model is not None
    assert model.nl_lag_months == {}


# --------------------------------------------------------------------------
# Post-study calendar (_build_post_study_calendar)
# --------------------------------------------------------------------------


def test_build_post_study_calendar_saturday_cadence() -> None:
    # 2026-05-01 is a Friday; GS trailing month (3) declares 5 weeks.
    stages = _build_post_study_calendar(date(2026, 5, 1), {1: 3, 2: 4, 3: 5})

    assert [s["start_date"] for s in stages] == [
        "2026-05-01",
        "2026-05-09",
        "2026-05-16",
        "2026-05-23",
        "2026-05-30",
        "2026-06-06",
    ]
    assert stages[0]["duration_hours"] == 192.0  # stub-absorbing first week
    assert [s["duration_hours"] for s in stages[1:5]] == [168.0] * 4
    assert stages[5]["duration_hours"] == 600.0  # trailing month: 06-06 -> 07-01


def test_build_post_study_calendar_is_contiguous() -> None:
    stages = _build_post_study_calendar(date(2026, 5, 1), {1: 3, 2: 4, 3: 5})

    for cur, nxt in zip(stages, stages[1:]):
        start = date.fromisoformat(cur["start_date"])
        end = start + timedelta(hours=cur["duration_hours"])
        assert end == date.fromisoformat(nxt["start_date"])
    # Every weekly stage after the stub-absorbing first starts on a Saturday.
    for stage in stages[1:5]:
        assert date.fromisoformat(stage["start_date"]).weekday() == 5


def test_build_post_study_calendar_no_stub_when_horizon_end_is_saturday() -> None:
    saturday = date(2026, 5, 2)
    assert saturday.weekday() == 5

    stages = _build_post_study_calendar(saturday, {1: 3})

    assert stages[0]["start_date"] == "2026-05-02"
    assert stages[0]["duration_hours"] == 168.0  # no stub: already Saturday-aligned


def test_build_post_study_calendar_empty_gs_returns_empty() -> None:
    assert _build_post_study_calendar(date(2026, 5, 1), {}) == []


# --------------------------------------------------------------------------
# Emission (convert_gnl)
# --------------------------------------------------------------------------

# decomp-mar-26-rv2's operative calendar: 3 weekly March stages (168 h each) + 1
# monthly April stage (648 h), horizon ends 2026-05-01 (cumulative 1152 h).
_EMIT_STAGES = [
    {
        "start_date": "2026-03-14",
        "end_date": "2026-03-21",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-03-21",
        "end_date": "2026-03-28",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-03-28",
        "end_date": "2026-04-04",
        "blocks": [{"hours": 168.0}],
    },
    {
        "start_date": "2026-04-04",
        "end_date": "2026-05-01",
        "blocks": [{"hours": 648.0}],
    },
]
_BUS_OF = {1: 0, 3: 2}.get


# The GS post-study calendar shared by the emission tests below: horizon_end
# 2026-05-01 (a Friday) -> a stub-absorbing first weekly stage, 4 more full
# weekly stages, then one trailing monthly stage (mirrors the standalone
# _build_post_study_calendar fixture, so both are cross-checked together).
_EMIT_WEEKS_PER_MONTH = {1: 3, 2: 4, 3: 5}


def _emit_model() -> GnlCommitmentModel:
    """SANTA CRUZ (86): zero in-horizon + 500 MW post-horizon (2026-05-09), a
    2-month dispatch-anticipation lag; PSERGIPE I (224): registry-only, no
    committed delivery, same 2-month lag (inert but still anticipated -> still
    gets free forward decisions synthesised onto the GS calendar)."""
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
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2, 224: 2},
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
    # SANTA CRUZ delivers post-horizon (2026-05-09) with a 2-month nl lag, so its
    # physical lead is derived from that lag against the calendar: the delivery
    # window ends 2026-05-16 (1512 h from the study start) and the decision lands
    # in the first (early-March) stage [0, 168 h) -> H = 1512 - 168 = 1344 h. This
    # is the fix: NOT the old committed-footprint 168 h that dropped the delivery.
    assert santa["anticipated_config"] == {"lead_time_hours": 1344.0}
    # PSERGIPE has no post-horizon delivery -> horizon-bounded footprint lead.
    assert e.thermals[1]["anticipated_config"] == {"lead_time_hours": 168.0}


def test_convert_gnl_left_boundary_tiles_the_h_derived_leading_stages() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    santa = [p for p in e.past_anticipated_commitments if p["thermal_id"] == 94]
    pserg = [p for p in e.past_anticipated_commitments if p["thermal_id"] == 95]
    # SANTA CRUZ's lead (1344 h) reaches past the horizon, so cobre treats every
    # study stage as pre-study-committed: all 4 stages tiled (0 MW, in-horizon
    # geracao is 0), matching lead_delivery_stage_count(1344) = 4.
    assert [p["start_date"] for p in santa] == [
        "2026-03-14",
        "2026-03-21",
        "2026-03-28",
        "2026-04-04",
    ]
    assert all(p["value_mw"] == 0.0 for p in santa)
    # PSERGIPE's footprint lead (168 h) tiles only the leading stage.
    assert len(pserg) == 1
    assert pserg[0]["start_date"] == "2026-03-14"


def test_convert_gnl_right_boundary_delivery_and_post_study() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    pss = e.post_study_stages
    assert pss is not None
    # post_study_stages is the GS calendar verbatim (calendar-driven, not a
    # delivery-breakpoint one) — cross-checked against the standalone builder.
    assert pss["stages"] == _build_post_study_calendar(
        date(2026, 5, 1), _EMIT_WEEKS_PER_MONTH
    )

    # SANTA CRUZ's prior-revision 500 MW delivery (2026-05-09 -> 05-16) stays
    # pinned; every other delivery (both plants, every other stage) is free.
    pinned = [
        d
        for d in e.future_anticipated_deliveries
        if d["thermal_id"] == 94
        and d["delivery_start"] == "2026-05-09"
        and d["delivery_end"] == "2026-05-16"
    ]
    assert pinned == [
        {
            "thermal_id": 94,
            "delivery_start": "2026-05-09",
            "delivery_end": "2026-05-16",
            "min_mw": 500.0,
            "max_mw": 500.0,
        }
    ]
    free = [d for d in e.future_anticipated_deliveries if d not in pinned]
    assert free  # the whole point of this ticket: free decisions now exist
    assert all(d["min_mw"] < d["max_mw"] for d in free)


def test_convert_gnl_no_post_horizon_delivery_yields_no_post_study() -> None:
    # PSERGIPE-only model with no GS calendar declared: registry, zero
    # in-horizon, nothing post-horizon, and an empty weeks_per_month -> no
    # calendar to place a free forward decision on either.
    pserg = GnlThermal(224, "PSERGIPE I", 3, 321.26, 0.0, 1593.0)
    model = GnlCommitmentModel(
        thermals=(pserg,),
        commitments={
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            )
        },
        weeks_per_month={},
        nl_lag_months={224: 2},
    )
    e = convert_gnl(model, first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES)

    assert e.future_anticipated_deliveries == []
    assert e.post_study_stages is None
    assert len(e.past_anticipated_commitments) == 1  # left boundary still mandatory


def test_convert_gnl_synthesises_free_forward_delivery() -> None:
    """A study stage maps forward (via its nl lead) to a free post-study week."""
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    model = GnlCommitmentModel(
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            )
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2},
    )
    e = convert_gnl(model, first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES)

    calendar = _build_post_study_calendar(date(2026, 5, 1), _EMIT_WEEKS_PER_MONTH)
    # Study stage 0 is March's first weekly stage (offset 0); +2 months lands
    # on May's first calendar stage (also offset 0) -> calendar[0].
    expected_start = calendar[0]["start_date"]
    matches = [
        d
        for d in e.future_anticipated_deliveries
        if d["delivery_start"] == expected_start
    ]
    assert len(matches) == 1
    delivery = matches[0]
    assert delivery["min_mw"] < delivery["max_mw"]
    assert delivery["min_mw"] == santa.min_mw
    assert delivery["max_mw"] == santa.max_mw


def test_convert_gnl_prior_revision_delivery_stays_pinned() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    santa_pinned = [
        d
        for d in e.future_anticipated_deliveries
        if d["thermal_id"] == 94 and d["delivery_start"] == "2026-05-09"
    ]
    assert len(santa_pinned) == 1
    assert santa_pinned[0]["min_mw"] == santa_pinned[0]["max_mw"] == 500.0

    everything_else = [
        d for d in e.future_anticipated_deliveries if d not in santa_pinned
    ]
    assert everything_else
    assert all(d["min_mw"] < d["max_mw"] for d in everything_else)


def test_convert_gnl_deliveries_tile_whole_post_study_stages() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    pss = e.post_study_stages
    assert pss is not None
    stage_spans = {_calendar_stage_span(s) for s in pss["stages"]}

    assert e.future_anticipated_deliveries  # non-trivial coverage check
    for d in e.future_anticipated_deliveries:
        window = (
            date.fromisoformat(d["delivery_start"]),
            date.fromisoformat(d["delivery_end"]),
        )
        assert window in stage_spans  # exactly one whole calendar stage


def test_convert_gnl_thermal_bounds_intersect_delivery() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    pss = e.post_study_stages
    assert pss is not None
    cost_of = {94: 199.22, 95: 321.26}
    stage_index_of_span = {
        _calendar_stage_span(s): idx for idx, s in enumerate(pss["stages"])
    }
    bound_of = {
        (b["thermal_id"], b["post_study_stage_index"]): b for b in pss["thermal_bounds"]
    }

    assert e.future_anticipated_deliveries
    for d in e.future_anticipated_deliveries:
        window = (
            date.fromisoformat(d["delivery_start"]),
            date.fromisoformat(d["delivery_end"]),
        )
        idx = stage_index_of_span[window]
        bound = bound_of[(d["thermal_id"], idx)]
        assert bound["cost_per_mwh"] == cost_of[d["thermal_id"]]
        assert bound["min_mw"] <= d["max_mw"]
        assert d["min_mw"] <= bound["max_mw"]


def test_convert_gnl_clamps_committed_above_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # gl geracao and tg disponibilidade are independent; a commitment above the
    # plant's max_mw is clamped into [min_mw, max_mw] (both boundaries) + warned,
    # so cobre never rejects an out-of-bounds pin.
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    model = GnlCommitmentModel(
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 900.0, 168.0),
                    GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 900.0, 0.0),
                ),
            )
        },
        weeks_per_month={},
        nl_lag_months={86: 2},
    )
    with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.anticipated"):
        e = convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
        )

    assert e.past_anticipated_commitments[0]["value_mw"] == 500.0  # 900 -> max 500
    assert e.future_anticipated_deliveries[0]["min_mw"] == 500.0
    assert e.future_anticipated_deliveries[0]["max_mw"] == 500.0
    assert any("clamped" in r.message for r in caplog.records)


def _post_horizon_model(lag_months: int) -> GnlCommitmentModel:
    """SANTA CRUZ delivering 500 MW post-horizon (2026-05-09) with a given lag."""
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    return GnlCommitmentModel(
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(8, date(2026, 5, 2), 0.0, 0.0),
                    GnlStageCommitment(9, date(2026, 5, 9), 500.0, 0.0),
                ),
            )
        },
        weeks_per_month={},
        nl_lag_months={86: lag_months},
    )


@pytest.mark.parametrize(
    ("lag_months", "expected_lead"),
    [
        # delivery window ends 2026-05-16 = 1512 h from the study start; the lead
        # is 1512 h minus the cumulative hours through the decision stage (the
        # operative stage `lag` months before 2026-05-16):
        (1, 1512.0 - 1152.0),  # 2026-04-16 -> April stage [504, 1152) -> H = 360
        (2, 1512.0 - 168.0),  # 2026-03-16 -> stage 0 [0, 168)      -> H = 1344
    ],
)
def test_convert_gnl_nl_lag_sets_physical_lead_time(
    lag_months: int, expected_lead: float
) -> None:
    """The nl dispatch-anticipation lag (in months), resolved against the deck
    calendar, sets lead_time_hours so cobre's end-anchored decider lands on the
    operative stage `lag` months before the post-horizon delivery."""
    e = convert_gnl(
        _post_horizon_model(lag_months),
        first_thermal_id=94,
        bus_id_of=_BUS_OF,
        stages=_EMIT_STAGES,
    )
    assert e.thermals[0]["anticipated_config"]["lead_time_hours"] == expected_lead


def test_convert_gnl_pre_study_decision_delivery_skipped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A post-horizon delivery whose nl-implied decision predates the study is
    not emitted as an in-study-decided future delivery (cobre would drop it);
    it warns and falls back to the horizon-bounded footprint lead."""
    # lag 6 months: 2026-05-16 - 6 months = 2025-11-16, well before the study.
    with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.anticipated"):
        e = convert_gnl(
            _post_horizon_model(6),
            first_thermal_id=94,
            bus_id_of=_BUS_OF,
            stages=_EMIT_STAGES,
        )

    assert e.future_anticipated_deliveries == []
    assert e.post_study_stages is None
    # Footprint fallback stays within the horizon (never > 1152 h).
    assert e.thermals[0]["anticipated_config"]["lead_time_hours"] <= 1152.0
    assert any("decided before the study horizon" in r.message for r in caplog.records)
