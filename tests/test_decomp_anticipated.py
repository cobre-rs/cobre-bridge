"""Tests for the source model's GNL commitment reader (``decomp/anticipated.py``).

Deck-independent, CI-tier: every fixture below is a hand-authored
``_StubDadgnl`` exposing ``gl``/``tg``/``gs`` DataFrames that mirror the
verified column shape (no real ``Dadgnl.read`` call, no cobre import).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cobre_bridge.decomp.anticipated import is_gnl_enabled, read_gnl_model
from cobre_bridge.decomp.temporal import OperativeStage, build_operative_calendar

# rv0-shaped block hours, verbatim from tests/test_decomp_temporal.py's
# `_RV0_HOURS`: five operative weeks then the aggregated month (the shortest
# start date/duration combination that satisfies `build_operative_calendar`'s
# month-boundary invariant).
_WEEK_HOURS = [40.0, 48.0, 80.0]
_MONTH_HOURS = [152.0, 184.0, 312.0]


def _calendar() -> list[OperativeStage]:
    hours = [_WEEK_HOURS] * 5 + [_MONTH_HOURS]
    return build_operative_calendar(date(2024, 8, 31), hours)


class _StubDadgnl:
    """A plain fake exposing ``gl``/``tg``/``gs(df=True)`` — no real I/O."""

    def __init__(
        self,
        *,
        gl: pd.DataFrame | None = None,
        tg: pd.DataFrame | None = None,
        gs: pd.DataFrame | None = None,
    ) -> None:
        self._gl, self._tg, self._gs = gl, tg, gs

    def gl(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._gl

    def tg(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._tg

    def gs(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._gs


def _tg_row(
    code: int,
    name: str,
    *,
    submarket: int = 1,
    estagio: int = 1,
    cvu: tuple[float, float, float] = (10.0, 10.0, 10.0),
    disponibilidade: tuple[float, float, float] = (100.0, 100.0, 100.0),
    inflexibilidade: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    return {
        "codigo_submercado": submarket,
        "codigo_usina": code,
        "nome": name,
        "estagio": estagio,
        "cvu_1": cvu[0],
        "cvu_2": cvu[1],
        "cvu_3": cvu[2],
        "disponibilidade_1": disponibilidade[0],
        "disponibilidade_2": disponibilidade[1],
        "disponibilidade_3": disponibilidade[2],
        "inflexibilidade_1": inflexibilidade[0],
        "inflexibilidade_2": inflexibilidade[1],
        "inflexibilidade_3": inflexibilidade[2],
    }


def _gl_row(
    code: int,
    *,
    submarket: int = 1,
    estagio: int = 1,
    duracao: tuple[float, float, float] = (40.0, 48.0, 80.0),
    geracao: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    return {
        "codigo_submercado": submarket,
        "codigo_usina": code,
        "estagio": estagio,
        "data_inicio": "31082024",
        "duracao_1": duracao[0],
        "duracao_2": duracao[1],
        "duracao_3": duracao[2],
        "geracao_1": geracao[0],
        "geracao_2": geracao[1],
        "geracao_3": geracao[2],
    }


def _gs_frame() -> pd.DataFrame:
    return pd.DataFrame([{"mes": 1, "semanas": 4}, {"mes": 2, "semanas": 5}])


class TestIsGnlEnabled:
    def test_false_when_dadgnl_absent(self) -> None:
        assert is_gnl_enabled(None) is False

    def test_false_when_gl_empty(self) -> None:
        assert is_gnl_enabled(_StubDadgnl(gl=pd.DataFrame())) is False

    def test_false_when_gl_all_geracao_zero(self) -> None:
        gl = pd.DataFrame([_gl_row(86, geracao=(0.0, 0.0, 0.0))])
        assert is_gnl_enabled(_StubDadgnl(gl=gl)) is False

    def test_true_when_any_geracao_nonzero(self) -> None:
        gl = pd.DataFrame([_gl_row(86, geracao=(107.0, 131.0, 97.0))])
        assert is_gnl_enabled(_StubDadgnl(gl=gl)) is True


class TestReadGnlModel:
    def test_gate_off_returns_none_for_empty_gl(self) -> None:
        stub = _StubDadgnl(gl=pd.DataFrame(), tg=pd.DataFrame([_tg_row(86, "GNL 86")]))
        assert read_gnl_model(stub, _calendar()) is None

    def test_gate_off_returns_none_for_all_zero_gl(self) -> None:
        gl = pd.DataFrame([_gl_row(86, geracao=(0.0, 0.0, 0.0))])
        tg = pd.DataFrame([_tg_row(86, "GNL 86")])
        assert read_gnl_model(_StubDadgnl(gl=gl, tg=tg), _calendar()) is None

    def test_block_weighted_committed_mw(self) -> None:
        tg = pd.DataFrame([_tg_row(86, "GNL 86")])
        gl = pd.DataFrame(
            [_gl_row(86, duracao=(40.0, 48.0, 80.0), geracao=(107.0, 131.0, 97.0))]
        )
        stub = _StubDadgnl(gl=gl, tg=tg, gs=_gs_frame())

        model = read_gnl_model(stub, _calendar())

        assert model is not None
        expected = (40.0 * 107.0 + 48.0 * 131.0 + 80.0 * 97.0) / 168.0
        actual = model.commitments[86].committed_mw_by_stage[1]
        assert abs(actual - expected) < 1e-9
        assert model.weeks_per_month == {1: 4, 2: 5}

    def test_thermals_ascending_by_code(self) -> None:
        # Declared out of order to prove the ascending invariant isn't an
        # accident of insertion order.
        tg = pd.DataFrame([_tg_row(224, "C"), _tg_row(15, "A"), _tg_row(86, "B")])
        gl = pd.DataFrame([_gl_row(86, geracao=(10.0, 10.0, 10.0))])
        stub = _StubDadgnl(gl=gl, tg=tg)

        model = read_gnl_model(stub, _calendar())

        assert model is not None
        assert len(model.thermals) == 3
        assert [t.code for t in model.thermals] == [15, 86, 224]

    def test_missing_registry_code_raises(self) -> None:
        tg = pd.DataFrame([_tg_row(86, "GNL 86")])
        gl = pd.DataFrame(
            [
                _gl_row(86, geracao=(10.0, 10.0, 10.0)),
                _gl_row(999, geracao=(5.0, 5.0, 5.0)),
            ]
        )
        stub = _StubDadgnl(gl=gl, tg=tg)

        with pytest.raises(ValueError, match="999"):
            read_gnl_model(stub, _calendar())

    def test_registry_only_plant_has_empty_committed_mw_by_stage(self) -> None:
        tg = pd.DataFrame([_tg_row(15, "A"), _tg_row(86, "B")])
        gl = pd.DataFrame([_gl_row(86, geracao=(10.0, 10.0, 10.0))])
        stub = _StubDadgnl(gl=gl, tg=tg)

        model = read_gnl_model(stub, _calendar())

        assert model is not None
        assert model.commitments[15].committed_mw_by_stage == {}
        assert model.commitments[86].committed_mw_by_stage != {}
