"""Tests for the DECOMP PQ → NCS converters."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.ncs import (
    convert_ncs_factors,
    convert_ncs_stats,
    convert_non_controllable_sources,
)
from cobre_bridge.decomp.temporal import build_operative_calendar

_RV0_DECK = Path("example/decomp-set-24-rv0/dadger.rv0")
_RV3_DECK = Path("example/decomp-jul-26-rv3/dadger.rv3")

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


class _StubDadger:
    def __init__(self, pq: pd.DataFrame | None) -> None:
        self._pq = pq

    def pq(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._pq


def _pq_frame() -> pd.DataFrame:
    rows = [
        # Series declaring stages 1 and 3 (stage 2 inherits stage 1).
        {
            "codigo_submercado": 1,
            "estagio": 1,
            "nome": "SECO_PCH",
            "geracao_1": 1475.0,
            "geracao_2": 1424.0,
            "geracao_3": 1392.0,
        },
        {
            "codigo_submercado": 1,
            "estagio": 3,
            "nome": "SECO_PCH",
            "geracao_1": 1319.0,
            "geracao_2": 1270.0,
            "geracao_3": 1233.0,
        },
        # Series declaring only stage 1 (everything inherits).
        {
            "codigo_submercado": 2,
            "estagio": 1,
            "nome": "S_PCT",
            "geracao_1": 100.0,
            "geracao_2": 200.0,
            "geracao_3": 300.0,
        },
        # Zero-generation series: stats mean 0, no factors entries.
        {
            "codigo_submercado": 4,
            "estagio": 1,
            "nome": "N_EOL",
            "geracao_1": 0.0,
            "geracao_2": 0.0,
            "geracao_3": 0.0,
        },
    ]
    return pd.DataFrame(rows)


class TestRegistry:
    def test_entries_sorted_and_must_run(self) -> None:
        doc = convert_non_controllable_sources(
            _StubDadger(_pq_frame()), _ID_MAP, _calendar(), date(2026, 7, 18)
        )
        entries = doc["non_controllable_sources"]
        assert [e["id"] for e in entries] == [0, 1, 2]
        assert [e["name"] for e in entries] == ["SECO_PCH_1", "S_PCT_2", "N_EOL_4"]
        assert [e["bus_id"] for e in entries] == [0, 1, 3]
        assert all(e["allow_curtailment"] is False for e in entries)
        assert entries[0]["max_generation_mw"] == 1475.0
        assert entries[0]["operational_start_date"] == "2026-07-18"

    def test_empty_pq_gives_empty_registry(self) -> None:
        doc = convert_non_controllable_sources(
            _StubDadger(None), _ID_MAP, _calendar(), date(2026, 7, 18)
        )
        assert doc["non_controllable_sources"] == []

    def test_missing_stage_one_raises(self) -> None:
        pq = _pq_frame()
        pq.loc[pq["nome"] == "S_PCT", "estagio"] = 2
        with pytest.raises(ValueError, match="stage 1"):
            convert_non_controllable_sources(
                _StubDadger(pq), _ID_MAP, _calendar(), date(2026, 7, 18)
            )

    def test_stage_outside_calendar_raises(self) -> None:
        pq = _pq_frame()
        pq.loc[0, "estagio"] = 9
        with pytest.raises(ValueError, match="outside the calendar"):
            convert_non_controllable_sources(
                _StubDadger(pq), _ID_MAP, _calendar(), date(2026, 7, 18)
            )


class TestStats:
    def test_availability_fraction_with_inheritance(self) -> None:
        calendar = _calendar()
        stats = convert_ncs_stats(
            _StubDadger(_pq_frame()), _ID_MAP, calendar
        ).to_pandas()
        assert len(stats) == 3 * 3
        assert set(stats["std"]) == {0.0}

        pch = stats[stats["ncs_id"] == 0].set_index("stage_id")["mean"]
        stage0_mean = (1475.0 * 15 + 1424.0 * 64 + 1392.0 * 89) / 168.0
        stage2_mean = (1319.0 * 63 + 1270.0 * 280 + 1233.0 * 401) / 744.0
        assert pch[0] == pytest.approx(stage0_mean / 1475.0)
        assert pch[1] == pytest.approx(stage0_mean / 1475.0)  # inherited
        assert pch[2] == pytest.approx(stage2_mean / 1475.0)

        assert set(stats[stats["ncs_id"] == 2]["mean"]) == {0.0}


class TestFactors:
    def test_invariant_and_zero_mean_omission(self) -> None:
        calendar = _calendar()
        doc = convert_ncs_factors(_StubDadger(_pq_frame()), _ID_MAP, calendar)
        entries = doc["non_controllable_factors"]
        assert {e["ncs_id"] for e in entries} == {0, 1}
        for entry in entries:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-12)


class TestRealDecks:
    @pytest.mark.skipif(not _RV0_DECK.exists(), reason="rv0 deck not present")
    def test_rv0_series_count(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV0_DECK))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        doc = convert_non_controllable_sources(
            dadger, id_map, calendar, calendar[0].start_date
        )
        entries = doc["non_controllable_sources"]
        assert len(entries) == 32
        assert all(e["allow_curtailment"] is False for e in entries)

        stats = convert_ncs_stats(dadger, id_map, calendar).to_pandas()
        assert len(stats) == 32 * len(calendar)

    @pytest.mark.skipif(not _RV3_DECK.exists(), reason="rv3 deck not present")
    def test_rv3_end_to_end(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV3_DECK))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        doc = convert_non_controllable_sources(
            dadger, id_map, calendar, calendar[0].start_date
        )
        assert len(doc["non_controllable_sources"]) == 24

        factors = convert_ncs_factors(dadger, id_map, calendar)
        for entry in factors["non_controllable_factors"]:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-9)
