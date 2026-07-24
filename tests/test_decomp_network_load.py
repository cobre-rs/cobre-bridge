"""Tests for the DECOMP bus and load converters."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.load import convert_load_factors, convert_load_stats
from cobre_bridge.decomp.network import convert_buses
from cobre_bridge.decomp.temporal import build_operative_calendar

_RV0_DECK = Path("example/decomp-set-24-rv0/dadger.rv0")
_RV3_DECK = Path("example/decomp-jul-26-rv3/dadger.rv3")

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
)


def _calendar_rv3():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


class _StubDadger:
    """Minimal Dadger stand-in carrying sb/cd/dp DataFrames."""

    def __init__(
        self,
        sb: pd.DataFrame | None = None,
        cd: pd.DataFrame | None = None,
        dp: pd.DataFrame | None = None,
    ) -> None:
        self._sb, self._cd, self._dp = sb, cd, dp

    def sb(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._sb

    def cd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._cd

    def dp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._dp


def _cd_frame(
    cost: float = 7810.62,
    limit: float = 100.0,
    codes: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_curva": 1,
                "codigo_submercado": code,
                "estagio": 1,
                "nome_curva": "1PDEF",
                "custo_1": cost,
                "custo_2": cost,
                "custo_3": cost,
                "limite_superior_1": limit,
                "limite_superior_2": limit,
                "limite_superior_3": limit,
            }
            for code in codes
        ]
    )


def _dp_frame(calendar_stages: int = 3) -> pd.DataFrame:
    rows = []
    weekly = {"duracao_1": 15.0, "duracao_2": 64.0, "duracao_3": 89.0}
    monthly = {"duracao_1": 63.0, "duracao_2": 280.0, "duracao_3": 401.0}
    for estagio in range(1, calendar_stages + 1):
        durations = monthly if estagio == calendar_stages else weekly
        for sbm, base in ((1, 40000.0), (2, 15000.0), (11, None)):
            rows.append(
                {
                    "codigo_submercado": sbm,
                    "estagio": estagio,
                    "numero_patamares": 3,
                    "carga_1": None if base is None else base * 1.2,
                    "carga_2": None if base is None else base,
                    "carga_3": None if base is None else base * 0.8,
                    **durations,
                }
            )
    return pd.DataFrame(rows)


class TestDecompIdMap:
    def test_deterministic_ids_and_transhipment(self) -> None:
        assert _ID_MAP.bus_id(1) == 0
        assert _ID_MAP.bus_id(11) == 4
        assert _ID_MAP.transhipment_bus_id == 5
        assert _ID_MAP.n_buses == 6
        assert _ID_MAP.bus_id_by_name("SE") == 0
        assert _ID_MAP.bus_id_by_name("IV") == 5
        assert _ID_MAP.bus_name(5) == "IV"

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(KeyError, match="99"):
            _ID_MAP.bus_id(99)

    def test_reserved_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "IV"))

    @pytest.mark.skipif(not _RV0_DECK.exists(), reason="rv0 deck not present")
    def test_from_dadger_rv0(self) -> None:
        from idecomp.decomp import Dadger

        id_map = DecompIdMap.from_dadger(Dadger.read(str(_RV0_DECK)))
        assert id_map.bus_codes == (1, 2, 3, 4, 11)
        assert id_map.bus_names == ("SE", "S", "NE", "N", "FC")


class TestConvertBuses:
    def test_buses_with_and_without_deficit(self) -> None:
        dadger = _StubDadger(cd=_cd_frame())
        doc = convert_buses(dadger, _ID_MAP, date(2024, 8, 31))
        buses = doc["buses"]
        assert [b["id"] for b in buses] == [0, 1, 2, 3, 4, 5]
        assert buses[0]["deficit_segments"] == [{"depth_mw": None, "cost": 7810.62}]
        assert "deficit_segments" not in buses[4]  # FC
        assert buses[5]["name"] == "IV"
        assert "deficit_segments" not in buses[5]
        assert buses[0]["operational_start_date"] == "2024-08-31"

    def test_rejects_non_full_depth(self) -> None:
        dadger = _StubDadger(cd=_cd_frame(limit=95.0))
        with pytest.raises(ValueError, match="depth"):
            convert_buses(dadger, _ID_MAP, date(2024, 8, 31))

    def test_rejects_block_varying_cost(self) -> None:
        cd = _cd_frame()
        cd.loc[0, "custo_2"] = 9000.0
        with pytest.raises(ValueError, match="uniform"):
            convert_buses(_StubDadger(cd=cd), _ID_MAP, date(2024, 8, 31))

    def test_rejects_multi_segment_curves(self) -> None:
        cd = pd.concat([_cd_frame(), _cd_frame(cost=9000.0)], ignore_index=True)
        cd.loc[cd.index[-4:], "codigo_curva"] = 2
        with pytest.raises(ValueError, match="multi-segment"):
            convert_buses(_StubDadger(cd=cd), _ID_MAP, date(2024, 8, 31))


class TestConvertLoad:
    def test_stats_cover_every_bus_and_stage(self) -> None:
        table = convert_load_stats(
            _StubDadger(dp=_dp_frame()), _ID_MAP, _calendar_rv3()
        )
        df = table.to_pandas()
        assert len(df) == 6 * 3
        assert set(df["std_mw"]) == {0.0}
        # SE stage 0: (48000·15 + 40000·64 + 32000·89)/168
        se = df[(df["bus_id"] == 0) & (df["stage_id"] == 0)]["mean_mw"].iloc[0]
        assert se == pytest.approx((48000 * 15 + 40000 * 64 + 32000 * 89) / 168)
        # FC (NaN cargas) and IV (absent from DP) carry zero load.
        assert set(df[df["bus_id"].isin([4, 5])]["mean_mw"]) == {0.0}

    def test_factors_invariant_and_zero_mean_omission(self) -> None:
        calendar = _calendar_rv3()
        doc = convert_load_factors(_StubDadger(dp=_dp_frame()), _ID_MAP, calendar)
        entries = doc["load_factors"]
        assert {e["bus_id"] for e in entries} == {0, 1}
        for entry in entries:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-12)

    def test_rejects_block_count_mismatch(self) -> None:
        dp = _dp_frame()
        dp.loc[0, "numero_patamares"] = 2
        with pytest.raises(ValueError, match="blocks"):
            convert_load_stats(_StubDadger(dp=dp), _ID_MAP, _calendar_rv3())

    def test_rejects_stage_outside_calendar(self) -> None:
        dp = _dp_frame(calendar_stages=4)
        with pytest.raises(ValueError, match="outside the calendar"):
            convert_load_stats(_StubDadger(dp=dp), _ID_MAP, _calendar_rv3())


class TestRealDecks:
    @pytest.mark.skipif(not _RV0_DECK.exists(), reason="rv0 deck not present")
    def test_rv0_end_to_end(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV0_DECK))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        buses = convert_buses(dadger, id_map, calendar[0].start_date)["buses"]
        assert buses[0]["deficit_segments"][0]["cost"] == 7810.62

        stats = convert_load_stats(dadger, id_map, calendar).to_pandas()
        assert len(stats) == id_map.n_buses * len(calendar)
        assert (stats[stats["bus_id"] == 0]["mean_mw"] > 0).all()

        factors = convert_load_factors(dadger, id_map, calendar)["load_factors"]
        assert {e["bus_id"] for e in factors} == {0, 1, 2, 3}

    @pytest.mark.skipif(not _RV3_DECK.exists(), reason="rv3 deck not present")
    def test_rv3_end_to_end(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV3_DECK))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        buses = convert_buses(dadger, id_map, calendar[0].start_date)["buses"]
        assert buses[0]["deficit_segments"][0]["cost"] == 8291.25

        stats = convert_load_stats(dadger, id_map, calendar).to_pandas()
        se_stage0 = stats[(stats["bus_id"] == 0) & (stats["stage_id"] == 0)]
        expected = (47751.0 * 15 + 45562.0 * 64 + 38982.0 * 89) / 168.0
        assert se_stage0["mean_mw"].iloc[0] == pytest.approx(expected)
