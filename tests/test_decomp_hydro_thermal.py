"""Tests for the DECOMP hydro and thermal converters."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge.decomp.hydro import (
    convert_energy_productivity,
    convert_hydros,
    convert_initial_storage,
    convert_production_models,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import build_operative_calendar
from cobre_bridge.decomp.thermal import convert_thermal_bounds, convert_thermals

_RV3_DECK = Path("example/decomp-jul-26-rv3")

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
    hydro_codes=(1, 2, 5),
    thermal_codes=(10, 20),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _hidr_frame() -> pd.DataFrame:
    def plant(
        name: str,
        sub: int,
        jusante: int,
        vmin: float,
        vmax: float,
        cota: float = 100.0,
        cf: float = 20.0,
    ) -> dict:
        return {
            "nome_usina": name,
            "submercado": sub,
            "codigo_usina_jusante": jusante,
            "volume_minimo": vmin,
            "volume_maximo": vmax,
            "numero_conjuntos_maquinas": 1,
            "maquinas_conjunto_1": 2,
            "vazao_nominal_conjunto_1": 100.0,
            "potencia_nominal_conjunto_1": 50.0,
            "teif": 0.0,
            "ip": 0.0,
            "a0_volume_cota": cota,
            "a1_volume_cota": 0.0,
            "a2_volume_cota": 0.0,
            "a3_volume_cota": 0.0,
            "a4_volume_cota": 0.0,
            "canal_fuga_medio": cf,
            "produtibilidade_especifica": 0.009,
            "tipo_perda": 0,
            "perdas": 0.0,
        }

    df = pd.DataFrame(
        {
            1: plant("UP_RES", 1, 3, 100.0, 500.0),
            # Plant 3 exists in the registry but is not operated (skipped through).
            3: plant("SKIPPED", 1, 2, 0.0, 0.0),
            2: plant("MID_RES", 2, 0, 50.0, 250.0),
            5: plant("LEAF", 4, 0, 10.0, 10.0),
        }
    ).T
    df.index.name = "codigo_usina"
    return df


class _StubDadger:
    def __init__(
        self,
        uh: pd.DataFrame | None = None,
        ct: pd.DataFrame | None = None,
    ) -> None:
        self._uh, self._ct = uh, ct

    def uh(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._uh

    def ct(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ct

    def ac(  # noqa: ARG002
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = False,
    ) -> pd.DataFrame:
        # No AC machine-configuration overrides on this synthetic stub — every
        # plant here has teif=ip=0.0 anyway, so the AC-adjusted rated capacity
        # (decomp/hydro.py::_compute_max_turbined_rated_ac_adjusted) reduces
        # to the plain registry rated sum regardless.
        return pd.DataFrame()


def _uh_frame() -> pd.DataFrame:
    rows = [
        {
            "codigo_usina": 1,
            "volume_inicial": 50.0,
            "vazao_defluente_minima": 30.0,
            "volume_morto_inicial": None,
        },
        {
            "codigo_usina": 2,
            "volume_inicial": 100.0,
            "vazao_defluente_minima": None,
            "volume_morto_inicial": None,
        },
        {
            "codigo_usina": 5,
            "volume_inicial": 0.0,
            "vazao_defluente_minima": None,
            "volume_morto_inicial": None,
        },
        # Coupling-only registration: no initial volume.
        {
            "codigo_usina": 99,
            "volume_inicial": None,
            "vazao_defluente_minima": None,
            "volume_morto_inicial": None,
        },
    ]
    return pd.DataFrame(rows)


class TestConvertHydros:
    def test_registry_entries_and_cascade_skip(self, caplog) -> None:
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.hydro"):
            doc = convert_hydros(
                _StubDadger(uh=_uh_frame()), _hidr_frame(), _ID_MAP, date(2026, 7, 18)
            )
        hydros = doc["hydros"]
        assert [h["id"] for h in hydros] == [0, 1, 2]

        up = hydros[0]
        assert up["name"] == "UP_RES"
        assert "bus_id" not in up
        assert up["unit_groups"][0]["bus_id"] == 0
        # Plant 1 → 3 (not operated, skipped) → 2 (operated).
        assert up["downstream_id"] == 1
        assert up["reservoir"] == {
            "min_storage_hm3": 100.0,
            "max_storage_hm3": 500.0,
        }
        assert up["outflow"]["min_outflow_m3s"] == 30.0
        # 2 machines × 100 m³/s and × 50 MW, no derating.
        assert up["generation"]["max_turbined_m3s"] == 200.0
        assert up["generation"]["max_generation_mw"] == 100.0

        assert hydros[1]["downstream_id"] is None
        assert "coupling-only" in caplog.text
        assert "99" in caplog.text

    def test_unit_groups_present_and_mirror_generation(self, caplog) -> None:
        """Every hydro carries exactly one mirror unit group (cobre rule 41)
        and no top-level ``bus_id`` (removed field)."""
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.hydro"):
            doc = convert_hydros(
                _StubDadger(uh=_uh_frame()), _hidr_frame(), _ID_MAP, date(2026, 7, 18)
            )
        for h in doc["hydros"]:
            assert "bus_id" not in h
            assert len(h["unit_groups"]) == 1
            group = h["unit_groups"][0]
            assert group.keys() == {
                "id",
                "name",
                "bus_id",
                "min_generation_mw",
                "max_generation_mw",
                "min_turbined_m3s",
                "max_turbined_m3s",
            }
            gen = h["generation"]
            assert group["min_generation_mw"] == pytest.approx(gen["min_generation_mw"])
            assert group["max_generation_mw"] == pytest.approx(gen["max_generation_mw"])
            assert group["min_turbined_m3s"] == pytest.approx(gen["min_turbined_m3s"])
            assert group["max_turbined_m3s"] == pytest.approx(gen["max_turbined_m3s"])

    def test_initial_storage_formula(self) -> None:
        storage = convert_initial_storage(
            _StubDadger(uh=_uh_frame()), _hidr_frame(), _ID_MAP
        )
        by_id = {e["hydro_id"]: e["value_hm3"] for e in storage}
        assert by_id[0] == pytest.approx(100.0 + 0.5 * 400.0)
        assert by_id[1] == pytest.approx(250.0)  # 100 %
        assert by_id[2] == pytest.approx(10.0)  # run-of-river, vmin == vmax

    def test_energy_productivity_head(self) -> None:
        table = convert_energy_productivity(_hidr_frame(), _ID_MAP).to_pandas()
        # Flat cota 100, tailrace 20 → head 80; ρ = 0.009 × 80.
        assert table["equivalent_productivity_mw_per_m3s"].iloc[0] == pytest.approx(
            0.009 * 80.0
        )
        assert table["stage_id"].isna().all()

    def test_production_models_constant(self) -> None:
        doc = convert_production_models(_ID_MAP)
        models = doc["production_models"]
        assert [m["hydro_id"] for m in models] == [0, 1, 2]
        assert all(
            m["stage_ranges"][0]["model"] == "constant_productivity" for m in models
        )


def _ct_frame() -> pd.DataFrame:
    def row(code, name, estagio, cvu, disp, inflex):
        return {
            "codigo_submercado": 1,
            "codigo_usina": code,
            "estagio": estagio,
            "nome_usina": name,
            "cvu_1": cvu,
            "cvu_2": cvu,
            "cvu_3": cvu,
            "disponibilidade_1": disp[0],
            "disponibilidade_2": disp[1],
            "disponibilidade_3": disp[2],
            "inflexibilidade_1": inflex,
            "inflexibilidade_2": inflex,
            "inflexibilidade_3": inflex,
        }

    return pd.DataFrame(
        [
            # Plant 10: per-block spread on availability; declares stages 1 and 3.
            row(10, "SPREAD", 1, 100.0, (400.0, 300.0, 200.0), 0.0),
            row(10, "SPREAD", 3, 120.0, (400.0, 300.0, 200.0), 0.0),
            # Plant 20: flat, stage 1 only (stages 2-3 inherit).
            row(20, "FLAT", 1, 50.0, (640.0, 640.0, 640.0), 640.0),
        ]
    )


class TestConvertThermals:
    def test_registry_and_bounds(self) -> None:
        calendar = _calendar()
        doc = convert_thermals(
            _StubDadger(ct=_ct_frame()), _ID_MAP, calendar, date(2026, 7, 18)
        )
        thermals = doc["thermals"]
        assert [t["id"] for t in thermals] == [0, 1]
        spread_expected = (400.0 * 15 + 300.0 * 64 + 200.0 * 89) / 168.0
        assert thermals[0]["generation"]["max_mw"] == pytest.approx(spread_expected)
        assert thermals[1]["cost_per_mwh"] == pytest.approx(50.0)
        assert thermals[1]["generation"]["min_mw"] == pytest.approx(640.0)

        bounds = convert_thermal_bounds(
            _StubDadger(ct=_ct_frame()), _ID_MAP, calendar
        ).to_pandas()
        base = bounds[bounds["block_id"].isna()]
        overrides = bounds[bounds["block_id"].notna()]
        assert len(base) == 2 * 3  # one base row per (thermal, stage)

        flat = base[base["thermal_id"] == 1]
        assert set(flat["max_generation_mw"]) == {640.0}
        assert bounds[bounds["thermal_id"] == 1]["block_id"].isna().all()  # no spread

        spread = base[base["thermal_id"] == 0].set_index("stage_id")
        assert spread.loc[1, "cost_per_mwh"] == pytest.approx(100.0)  # inherited
        assert spread.loc[2, "cost_per_mwh"] == pytest.approx(120.0)

        # Plant 10's availability varies by block on every stage (its own
        # declarations at stage 1 and 3, plus stage 2 inheriting stage 1) —
        # one override row per (stage, block), inflexibilidade flat at 0.0.
        spread_overrides = overrides[overrides["thermal_id"] == 0]
        assert len(spread_overrides) == 3 * 3
        assert set(spread_overrides["max_generation_mw"]) == {400.0, 300.0, 200.0}
        assert set(spread_overrides["min_generation_mw"]) == {0.0}
        assert spread_overrides["cost_per_mwh"].isna().all()

    def test_missing_stage_one_raises(self) -> None:
        ct = _ct_frame()
        ct.loc[ct["codigo_usina"] == 20, "estagio"] = 2
        with pytest.raises(ValueError, match="stage 1"):
            convert_thermals(
                _StubDadger(ct=ct), _ID_MAP, _calendar(), date(2026, 7, 18)
            )


class TestRealDecks:
    @pytest.mark.skipif(
        not (_RV3_DECK / "dadger.rv3").exists(), reason="rv3 deck not present"
    )
    def test_rv3_end_to_end(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.hydro import read_hidr
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV3_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        hidr = read_hidr(_RV3_DECK / "hidr.dat")

        assert len(id_map.hydro_codes) > 150
        assert len(id_map.thermal_codes) == 97

        doc = convert_hydros(dadger, hidr, id_map, calendar[0].start_date)
        hydros = doc["hydros"]
        assert len(hydros) == len(id_map.hydro_codes)
        itaipu = next(h for h in hydros if h["name"] == "ITAIPU")
        assert "bus_id" not in itaipu
        # Two per-frequency groups (ticket-027, code 66): unique ids {0, 1},
        # both on bus SE (id_map.bus_id(1) == 0), each 7000 MW / 6620 m3/s —
        # id-addressed, not array-order (cobre sorts by group id on load).
        groups = itaipu["unit_groups"]
        assert {g["id"] for g in groups} == {0, 1}
        for group in groups:
            assert group["bus_id"] == 0  # SE
            assert group["max_generation_mw"] == pytest.approx(7000.0)
            assert group["max_turbined_m3s"] == pytest.approx(6620.0)
        assert itaipu["downstream_id"] is None

        storage = convert_initial_storage(dadger, hidr, id_map)
        assert len(storage) == len(id_map.hydro_codes)
        for entry, hydro in zip(storage, hydros, strict=True):
            reservoir = hydro["reservoir"]
            assert (
                reservoir["min_storage_hm3"]
                <= entry["value_hm3"]
                <= reservoir["max_storage_hm3"]
            )

        thermals = convert_thermals(dadger, id_map, calendar, calendar[0].start_date)[
            "thermals"
        ]
        assert len(thermals) == 97
        cuiaba = next(t for t in thermals if t["name"] == "CUIABA CC")
        assert cuiaba["generation"]["max_mw"] == pytest.approx(490.0)

        bounds = convert_thermal_bounds(dadger, id_map, calendar).to_pandas()
        base = bounds[bounds["block_id"].isna()]
        assert len(base) == 97 * len(calendar)  # one base row per (thermal, stage)
        assert (bounds["block_id"].notna() & bounds["cost_per_mwh"].notna()).sum() == 0
