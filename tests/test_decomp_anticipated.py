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
    _cobre_safe_lead_hours,
    _study_lead_hours,
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
# Study lead H + mirror-shift post-study calendar
# --------------------------------------------------------------------------

# decomp-mar-26-rv2's operative study calendar: 3 weekly March stages (168 h
# each) + 1 monthly April stage (648 h); horizon ends 2026-05-01 (Friday),
# cumulative 1152 h. All weekly stage boundaries are Saturdays.
_STUDY_SPANS = [
    (date(2026, 3, 14), date(2026, 3, 21)),
    (date(2026, 3, 21), date(2026, 3, 28)),
    (date(2026, 3, 28), date(2026, 4, 4)),
    (date(2026, 4, 4), date(2026, 5, 1)),
]
# Ideal mirror lead H = (GS over the study's 2 months) x 168 = (3 + 4) x 168 =
# 1176 h (49 days) — the faithful full-week calendar (used post-cobre-fix).
_LEAD_H = 1176.0
# Cobre-safe emitted lead: the ideal is >= the 1152 h horizon, so it is capped to
# the largest operative-week multiple strictly below it = 6 x 168 = 1008 h
# (TRACKED COBRE-GAP WORKAROUND C13). This is what convert_gnl actually emits.
_CAPPED_H = 1008.0
_HORIZON_H = 1152.0


def test_study_lead_hours_sums_study_months() -> None:
    # The study spans March + April (2 distinct months); GS[1]+GS[2] = 3+4 = 7.
    assert _study_lead_hours(_STUDY_SPANS, {1: 3, 2: 4, 3: 5}) == 1176.0
    # Empty GS -> no lead (caller emits no post-study calendar).
    assert _study_lead_hours(_STUDY_SPANS, {}) == 0.0


def test_cobre_safe_lead_hours_caps_at_horizon(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # TRACKED COBRE-GAP WORKAROUND C13 (dev-facing only): an ideal lead >= the
    # horizon is capped to the largest operative-week multiple strictly below it
    # (1176 -> 1008), with a neutral INFO note (no cobre-limitation exposed); a
    # lead already below the horizon passes through untouched.
    with caplog.at_level(logging.INFO, logger="cobre_bridge.decomp.anticipated"):
        assert _cobre_safe_lead_hours(_LEAD_H, _HORIZON_H) == _CAPPED_H
    assert any("operative weeks" in r.message for r in caplog.records)
    # The neutral note must not leak the cobre limitation into user-facing logs.
    assert not any(
        "k_max" in r.message
        or "panic" in r.message
        or "workaround" in r.message.lower()
        for r in caplog.records
    )
    assert _cobre_safe_lead_hours(1008.0, _HORIZON_H) == 1008.0
    assert _cobre_safe_lead_hours(0.0, _HORIZON_H) == 0.0


def test_build_post_study_calendar_mirror_shift() -> None:
    # The post-study calendar is the study stages shifted forward by H = 1176 h:
    # post-study stage m ends at study_stage_end[m] + 49 days.
    stages = _build_post_study_calendar(_STUDY_SPANS, _LEAD_H)

    assert [s["start_date"] for s in stages] == [
        "2026-05-01",  # study horizon end (Friday); stub-absorbing first stage
        "2026-05-09",  # 03-21 + 49 d (Saturday)
        "2026-05-16",  # 03-28 + 49 d (Saturday)
        "2026-05-23",  # 04-04 + 49 d (Saturday)
    ]
    assert [s["duration_hours"] for s in stages] == [192.0, 168.0, 168.0, 648.0]
    # The trailing monthly stage mirrors the study monthly stage: 05-01 + 49 d.
    last = stages[-1]
    last_end = date.fromisoformat(last["start_date"]) + timedelta(
        hours=last["duration_hours"]
    )
    assert last_end == date(2026, 6, 19)
    # One post-study stage per study stage.
    assert len(stages) == len(_STUDY_SPANS)


def test_build_post_study_calendar_is_contiguous() -> None:
    stages = _build_post_study_calendar(_STUDY_SPANS, _LEAD_H)

    assert stages[0]["start_date"] == "2026-05-01"  # the study horizon end
    for cur, nxt in zip(stages, stages[1:]):
        start = date.fromisoformat(cur["start_date"])
        end = start + timedelta(hours=cur["duration_hours"])
        assert end == date.fromisoformat(nxt["start_date"])
    # Every stage after the stub-absorbing first starts on a Saturday — inherited
    # from the (Saturday-aligned) study weekly-stage ends + a week-multiple shift.
    for stage in stages[1:]:
        assert date.fromisoformat(stage["start_date"]).weekday() == 5


def test_build_post_study_calendar_no_stub_for_all_weekly_study() -> None:
    # An all-weekly study (last stage a full 168 h week ending Saturday) shifted by
    # a whole number of weeks -> the mirror's first stage is a clean 168 h week
    # (no stub). The monthly last stage is what offsets the real deck onto a stub.
    spans = [
        (date(2026, 3, 14), date(2026, 3, 21)),
        (date(2026, 3, 21), date(2026, 3, 28)),
    ]
    assert spans[-1][1].weekday() == 5  # 2026-03-28 is a Saturday
    stages = _build_post_study_calendar(spans, 336.0)  # shift by 2 whole weeks

    assert stages[0]["start_date"] == "2026-03-28"  # study horizon end (Saturday)
    assert [s["duration_hours"] for s in stages] == [168.0, 168.0]


def test_build_post_study_calendar_empty_gs_returns_empty() -> None:
    assert _build_post_study_calendar(_STUDY_SPANS, 0.0) == []


# --------------------------------------------------------------------------
# Emission (convert_gnl) — single global lead, free-only, index-direct
# --------------------------------------------------------------------------

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
_EMIT_WEEKS_PER_MONTH = {1: 3, 2: 4, 3: 5}
_BUS_OF = {1: 0, 3: 2}.get


def _emit_model() -> GnlCommitmentModel:
    """SANTA CRUZ (86): a prior-revision 500 MW post-horizon commitment
    (2026-05-09) that the mirror-shift emission does NOT re-emit (accepted loss),
    plus PSERGIPE I (224): registry-only, no committed delivery. Both share the
    2-month anticipation lag and both get one free forward decision per study
    stage synthesised onto the mirror calendar."""
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    pserg = GnlThermal(224, "PSERGIPE I", 3, 321.26, 0.0, 1593.0)
    return GnlCommitmentModel(
        thermals=(santa, pserg),
        commitments={
            86: GnlCommitment(
                86,
                (
                    GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),
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


def test_convert_gnl_single_global_lead() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    assert [(t["id"], t["name"]) for t in e.thermals] == [
        (94, "SANTA CRUZ"),
        (95, "PSERGIPE I"),
    ]
    # A single global lead for EVERY plant — the fix replacing the per-plant leads
    # (1344 / 168) that dropped windows + K=0'd. Emitted value is the cobre-safe
    # cap (1008 h, C13), not the ideal 1176 h that panics cobre.
    assert all(
        t["anticipated_config"] == {"lead_time_hours": _CAPPED_H} for t in e.thermals
    )
    santa = e.thermals[0]
    assert santa["bus_id"] == 0
    assert santa["cost_per_mwh"] == 199.22
    assert santa["generation"] == {"min_mw": 0.0, "max_mw": 500.0}


def test_convert_gnl_left_boundary_tiles_the_h_derived_leading_stages() -> None:
    # The capped lead 1008 h < horizon (1152), so lead_delivery_stage_count = the
    # leading 3 study stages (168+336+504 <= 1008 < 1152) — cobre's
    # check_commitment_coverage requires exactly these. Both plants tile 3 (0 MW:
    # no in-horizon gl commitment folds in).
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    for tid in (94, 95):
        past = [p for p in e.past_anticipated_commitments if p["thermal_id"] == tid]
        assert [p["start_date"] for p in past] == [
            "2026-03-14",
            "2026-03-21",
            "2026-03-28",
        ]
        assert all(p["value_mw"] == 0.0 for p in past)


def test_convert_gnl_free_only_deliveries_on_mirror_calendar() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    pss = e.post_study_stages
    assert pss is not None
    # post_study_stages carries the mirror calendar (at the cobre-safe capped
    # lead) verbatim (cross-checked).
    assert pss["stages"] == _build_post_study_calendar(_STUDY_SPANS, _CAPPED_H)
    # Every delivery is FREE (min < max for a non-degenerate plant); NONE pinned.
    assert e.future_anticipated_deliveries
    assert all(d["min_mw"] < d["max_mw"] for d in e.future_anticipated_deliveries)
    # One free delivery per plant per calendar stage, at the plant's capability.
    for tid, mx in ((94, 500.0), (95, 1593.0)):
        mine = [d for d in e.future_anticipated_deliveries if d["thermal_id"] == tid]
        assert len(mine) == len(pss["stages"])
        assert all(d["min_mw"] == 0.0 and d["max_mw"] == mx for d in mine)
    # The dropped prior-revision commitment leaves no pinned (min == max) delivery.
    assert not any(
        d["thermal_id"] == 94
        and d["delivery_start"] == "2026-05-09"
        and d["min_mw"] == d["max_mw"]
        for d in e.future_anticipated_deliveries
    )


def test_convert_gnl_synthesises_free_forward_delivery() -> None:
    # Study stage m -> post-study stage m (index-direct): stage 0 -> calendar[0].
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    calendar = _build_post_study_calendar(_STUDY_SPANS, _CAPPED_H)
    expected_start = calendar[0]["start_date"]
    santa = [
        d
        for d in e.future_anticipated_deliveries
        if d["thermal_id"] == 94 and d["delivery_start"] == expected_start
    ]
    assert len(santa) == 1
    assert santa[0]["min_mw"] == 0.0
    assert santa[0]["max_mw"] == 500.0
    assert santa[0]["min_mw"] < santa[0]["max_mw"]


def test_convert_gnl_free_deliveries_resolve_in_study() -> None:
    # Each free delivery's decider (window_end_hours - H, cobre's rule) lands on a
    # DISTINCT in-study stage 0..n-1; none is dropped (target in (0, horizon_end]).
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    cum = [0.0]
    for s in _EMIT_STAGES:
        cum.append(cum[-1] + sum(b["hours"] for b in s["blocks"]))
    horizon_end_h = cum[-1]
    start0 = date.fromisoformat(_EMIT_STAGES[0]["start_date"])

    deciders = []
    for d in e.future_anticipated_deliveries:
        if d["thermal_id"] != 94:
            continue
        window_end_h = (date.fromisoformat(d["delivery_end"]) - start0).days * 24.0
        target = window_end_h - _CAPPED_H
        assert 0.0 < target <= horizon_end_h  # in-study, not dropped
        deciders.append(sum(1 for b in cum if b < target) - 1)
    assert deciders == [0, 1, 2, 3]  # distinct, one per study stage


def test_convert_gnl_deliveries_tile_whole_post_study_stages() -> None:
    e = convert_gnl(
        _emit_model(), first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
    )
    pss = e.post_study_stages
    assert pss is not None
    stage_spans = {_calendar_stage_span(s) for s in pss["stages"]}

    assert e.future_anticipated_deliveries
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


def test_convert_gnl_warns_on_nonuniform_nl_lag(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A deck mixing NL lags is surfaced (the single global H assumes a uniform
    # lag); H is still emitted globally.
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    pserg = GnlThermal(224, "PSERGIPE I", 3, 321.26, 0.0, 1593.0)
    model = GnlCommitmentModel(
        thermals=(santa, pserg),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            ),
            224: GnlCommitment(
                224, (GnlStageCommitment(1, date(2026, 3, 14), 0.0, 168.0),)
            ),
        },
        weeks_per_month=_EMIT_WEEKS_PER_MONTH,
        nl_lag_months={86: 2, 224: 3},  # differing lags
    )
    with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.anticipated"):
        e = convert_gnl(
            model, first_thermal_id=94, bus_id_of=_BUS_OF, stages=_EMIT_STAGES
        )

    assert any("differing anticipation lags" in r.message for r in caplog.records)
    assert all(
        t["anticipated_config"] == {"lead_time_hours": _CAPPED_H} for t in e.thermals
    )


def test_convert_gnl_empty_gs_yields_no_post_study() -> None:
    # No GS calendar (degenerate deck): no post-study horizon to anticipate into,
    # so no free deliveries and no post_study_stages — but the plant stays a valid
    # anticipated thermal with the mandatory single leading commitment.
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
    assert e.thermals[0]["anticipated_config"] == {"lead_time_hours": 168.0}


def test_convert_gnl_clamps_past_commitment_above_capability(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # gl geracao and tg disponibilidade are independent; an in-study commitment
    # above the plant's max_mw is clamped into [min_mw, max_mw] (+ warned) so
    # cobre's semantic validator (value_mw in [min, max]) never rejects it.
    santa = GnlThermal(86, "SANTA CRUZ", 1, 199.22, 0.0, 500.0)
    model = GnlCommitmentModel(
        thermals=(santa,),
        commitments={
            86: GnlCommitment(
                86, (GnlStageCommitment(1, date(2026, 3, 14), 900.0, 168.0),)
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
    assert any("clamped" in r.message for r in caplog.records)
