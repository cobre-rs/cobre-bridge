"""Tests for the converters unblocked by idecomp 1.12 (IA/UE/RQ/renewables)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge.decomp.bounds import convert_hydro_bounds
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.ncs import convert_non_controllable_sources
from cobre_bridge.decomp.network import convert_lines, convert_pumping_stations
from cobre_bridge.decomp.temporal import build_operative_calendar

_RV3_DECK = Path("example/decomp-jul-26-rv3")
_needs_deck = pytest.mark.skipif(
    not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
)

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
    hydro_codes=(1, 2, 5),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


class _StubDadger:
    def __init__(self, **frames: pd.DataFrame | None) -> None:
        self._frames = frames

    def __getattr__(self, name: str):
        if name in self._frames:
            frame = self._frames[name]
            return lambda df=False, **kwargs: frame  # noqa: ARG005
        raise AttributeError(name)


def _ia_frame() -> pd.DataFrame:
    def row(estagio, de, para, dp, pd_):
        entry = {
            "estagio": estagio,
            "nome_submercado_de": de,
            "nome_submercado_para": para,
        }
        for k in range(1, 6):
            entry[f"limite_de_para_{k}"] = dp[k - 1] if k <= len(dp) else None
            entry[f"limite_para_de_{k}"] = pd_[k - 1] if k <= len(pd_) else None
        return entry

    return pd.DataFrame(
        [
            # SE-IV: per-block variation on the reverse direction.
            row(1, "SE", "IV", (6500.0, 6500.0, 6500.0), (9800.0, 9800.0, 11275.0)),
            # S-IV declares stages 1 and 3 (stage 2 inherits).
            row(1, "S", "IV", (5000.0, 5000.0, 5000.0), (6000.0, 6000.0, 6000.0)),
            row(3, "S", "IV", (5500.0, 5500.0, 5500.0), (6000.0, 6000.0, 6000.0)),
        ]
    )


class TestConvertLines:
    def test_lines_and_base_bounds(self) -> None:
        calendar = _calendar()
        lines_doc, bounds = convert_lines(
            _StubDadger(ia=_ia_frame()), _ID_MAP, calendar, date(2026, 7, 18)
        )
        lines = lines_doc["lines"]
        assert len(lines) == 2
        se_iv = next(line for line in lines if line["name"] == "SE-IV")
        assert se_iv["source_bus_id"] == 0
        assert se_iv["target_bus_id"] == 5  # the transhipment bus
        assert se_iv["capacity"]["reverse_mw"] == 11275.0

        table = bounds.to_pandas()
        base_rows = table[table["block_id"].isna()]
        assert len(base_rows) == 2 * 3  # one base row per (line, stage)
        s_iv_id = next(line["id"] for line in lines if line["name"] == "S-IV")
        s_iv = base_rows[base_rows["line_id"] == s_iv_id].set_index("stage_id")
        assert s_iv.loc[1, "direct_mw"] == 5000.0  # inherited
        assert s_iv.loc[2, "direct_mw"] == 5500.0

    def test_block_bounds_are_absolute_mw_no_factor(self) -> None:
        calendar = _calendar()
        lines_doc, bounds = convert_lines(
            _StubDadger(ia=_ia_frame()), _ID_MAP, calendar, date(2026, 7, 18)
        )
        lines = lines_doc["lines"]
        se_iv_id = next(line["id"] for line in lines if line["name"] == "SE-IV")
        s_iv_id = next(line["id"] for line in lines if line["name"] == "S-IV")

        table = bounds.to_pandas()

        # S-IV's blocks never differ from the base: no block rows at all.
        assert s_iv_id not in set(table[table["block_id"].notna()]["line_id"])

        # SE-IV's reverse direction varies per block; the absolute MW value
        # comes straight from the IA record's own per-block limit columns,
        # not from dividing by the base to build a factor.
        stage0 = table[(table["line_id"] == se_iv_id) & (table["stage_id"] == 0)]

        def _reverse_mw(block_id: int) -> float:
            row = stage0[stage0["block_id"] == block_id]
            assert len(row) == 1
            return float(row.iloc[0]["reverse_mw"])

        assert _reverse_mw(0) == 9800.0
        assert _reverse_mw(1) == 9800.0
        assert _reverse_mw(2) == 11275.0
        block_rows = stage0[stage0["block_id"].notna()]
        assert set(block_rows["direct_mw"]) == {6500.0}

    def test_missing_stage_one_raises(self) -> None:
        ia = _ia_frame()
        ia.loc[ia["nome_submercado_de"] == "SE", "estagio"] = 2
        with pytest.raises(ValueError, match="stage 1"):
            convert_lines(_StubDadger(ia=ia), _ID_MAP, _calendar(), date(2026, 7, 18))


class TestConvertPumping:
    def test_stations_mapped(self) -> None:
        ue = pd.DataFrame(
            [
                {
                    "codigo_submercado": 1,
                    "codigo_usina": 1,
                    "codigo_usina_jusante": 2,
                    "codigo_usina_montante": 5,
                    "nome_usina": "Elevatoria",
                    "taxa_consumo": 0.2,
                    "vazao_maxima_bombeavel": 160.0,
                    "vazao_minima_bombeavel": 0.0,
                }
            ]
        )
        doc = convert_pumping_stations(_StubDadger(ue=ue), _ID_MAP, date(2026, 7, 18))
        station = doc["pumping_stations"][0]
        assert station["source_hydro_id"] == 1
        assert station["destination_hydro_id"] == 2
        assert station["consumption_mw_per_m3s"] == 0.2
        assert station["flow"]["max_m3s"] == 160.0


class TestConvertHydroBounds:
    def _dadger(
        self,
        rq: pd.DataFrame | None = None,
        ac: pd.DataFrame | None = None,
        cq: pd.DataFrame | None = None,
    ) -> _StubDadger:
        uh = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "codigo_ree": 1,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": None,
                },
                {
                    "codigo_usina": 2,
                    "codigo_ree": 1,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": 25.0,
                },
                {
                    "codigo_usina": 5,
                    "codigo_ree": 2,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": None,
                },
            ]
        )
        stub = _StubDadger(
            uh=uh,
            rq=rq
            if rq is not None
            else pd.DataFrame(
                [{"codigo_ree": 1, "vazao_1": 100.0, "vazao_2": 100.0, "vazao_3": 0.0}]
            ),
            cq=cq if cq is not None else pd.DataFrame(),
        )
        overrides = ac if ac is not None else pd.DataFrame()
        stub.ac = lambda codigo_usina=None, modificacao=None, df=False: overrides  # type: ignore[attr-defined]
        return stub

    def _hidr(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                1: {"vazao_minima_historica": 40.0},
                2: {"vazao_minima_historica": 80.0},
                5: {"vazao_minima_historica": 10.0},
            }
        ).T
        df.index.name = "codigo_usina"
        return df

    def test_rq_percentages_and_uh_priority(self) -> None:
        calendar = _calendar()
        table = convert_hydro_bounds(
            self._dadger(), self._hidr(), _ID_MAP, calendar
        ).to_pandas()
        # Base row only (block_id = None) — the backward-compatible
        # hours-weighted per-stage value; the RQ percentages (100, 100, 0)
        # are non-uniform, so per-block override rows are also emitted
        # (covered by tests/test_decomp_rq_bounds.py).
        plant1 = table[(table["hydro_id"] == 0) & (table["block_id"].isna())].set_index(
            "stage_id"
        )
        # Weekly stages: (100·15 + 100·64 + 0·89)/168 = 47.02 % of 40 m³/s.
        weekly_pct = (100.0 * 15 + 100.0 * 64) / 168.0
        assert plant1.loc[0, "min_outflow_m3s"] == pytest.approx(
            weekly_pct / 100.0 * 40.0
        )
        # UH-declared value has priority, fixed for all stages.
        plant2 = table[table["hydro_id"] == 1]
        assert set(plant2["min_outflow_m3s"]) == {25.0}
        # REE 2 has no RQ record: no rows for plant 5.
        assert 2 not in set(table["hydro_id"])

    def test_ac_vazmin_override_applies(self) -> None:
        ac = pd.DataFrame([{"codigo_usina": 1, "vazao": 0.0}])
        table = convert_hydro_bounds(
            self._dadger(ac=ac), self._hidr(), _ID_MAP, _calendar()
        ).to_pandas()
        # Plant 1's historical minimum overridden to zero: no rows.
        assert 0 not in set(table["hydro_id"])

    def test_qdef_plants_skipped(self) -> None:
        cq = pd.DataFrame(
            [
                {
                    "codigo_restricao": 9,
                    "codigo_usina": 1,
                    "coeficiente": 1.0,
                    "estagio": 1,
                    "tipo": "QDEF",
                }
            ]
        )
        table = convert_hydro_bounds(
            self._dadger(cq=cq), self._hidr(), _ID_MAP, _calendar()
        ).to_pandas()
        assert 0 not in set(table["hydro_id"])


class TestRenovaveis:
    @_needs_deck
    def test_parks_append_after_pq(self) -> None:
        from idecomp.decomp import Dadger
        from idecomp.libs import Renovaveis

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV3_DECK / "dadger.rv3"))
        renovaveis = Renovaveis.read(str(_RV3_DECK / "renovaveis.csv"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        doc = convert_non_controllable_sources(
            dadger, id_map, calendar, calendar[0].start_date, renovaveis
        )
        entries = doc["non_controllable_sources"]
        assert len(entries) == 24 + 8
        park_names = [e["name"] for e in entries[24:]]
        assert any("Eolica" in name for name in park_names)
        assert [e["id"] for e in entries] == list(range(32))
        assert all(e["allow_curtailment"] is False for e in entries)
