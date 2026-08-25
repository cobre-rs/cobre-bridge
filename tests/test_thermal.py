"""Unit tests for the source model thermal converter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from cobre_bridge.newave.id_map import NewaveIdMap
from tests.conftest import (
    _make_term_df,
    _thermal_readers,
    make_case,
    make_nw_files,
)

# ---------------------------------------------------------------------------
# Thermal conversion
# ---------------------------------------------------------------------------


class TestConvertThermals:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2],
            hydro_codes=[],
            thermal_codes=[10, 20, 30],
        )

    def test_returns_thermals_key(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        assert "thermals" in result

    def test_thermal_count(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        assert len(result["thermals"]) == 3

    def test_thermal_ids_are_zero_based_sorted(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        ids = [t["id"] for t in result["thermals"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    def test_cost_per_mwh_scalar(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        for t in result["thermals"]:
            assert "cost_per_mwh" in t
            assert isinstance(t["cost_per_mwh"], float)
            assert "cost_segments" not in t
            assert "generation" in t
            assert "min_mw" in t["generation"]
            assert "max_mw" in t["generation"]

    def test_bus_id_assignment(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        # TERMO_A (code 10) and TERMO_B (code 20) are in submercado 1 -> bus 0.
        # TERMO_C (code 30) is in submercado 2 -> bus 1.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        termo_c = next(t for t in result["thermals"] if t["name"] == "TERMO_C")
        assert termo_a["bus_id"] == 0
        assert termo_c["bus_id"] == 1

    def test_capacity_uses_factor(self, tmp_path) -> None:
        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(case, self._make_id_map())
        # TERMO_A: potencia=100, factor=0.9 -> max_mw=90.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        assert termo_a["generation"]["max_mw"] == pytest.approx(90.0)


class TestConvertThermalBoundsClastModificacoes:
    """Per-stage cost overrides from the modificacoes block in clast.dat."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2],
            hydro_codes=[],
            thermal_codes=[10, 20, 30],
        )

    def _make_dger(self) -> MagicMock:
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        dger.num_anos_estudo = 1
        dger.num_anos_pos_estudo = 0
        dger.num_anos_manutencao_utes = 0
        return dger

    def test_modificacao_overrides_year_indexed_cost_inside_window(
        self, tmp_path
    ) -> None:
        import datetime

        conft, clast, term = _thermal_readers()

        # Override TERMO_A (code 10) cost from 50.0 -> 77.0 for stages 2-4
        # of a 12-stage 2023 horizon (March-May). Other stages keep 50.0.
        modif_df = pd.DataFrame(
            {
                "codigo_usina": [10],
                "nome_usina": ["TERMO_A"],
                "data_inicio": [datetime.datetime(2023, 3, 1)],
                "data_fim": [datetime.datetime(2023, 5, 1)],
                "custo": [77.0],
            }
        )
        clast.modificacoes = modif_df
        case = make_case(
            tmp_path,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
        )

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None

        df = table.to_pandas()
        termo_a_id = self._make_id_map().thermal_id(10)
        a_rows = df[df["thermal_id"] == termo_a_id].sort_values("stage_id")
        # 12 stages emitted for the cost-varying plant.
        assert len(a_rows) == 12
        # Inside the modification window (stages 2, 3, 4 -> Mar, Apr, May).
        assert a_rows.iloc[2]["cost_per_mwh"] == pytest.approx(77.0)
        assert a_rows.iloc[3]["cost_per_mwh"] == pytest.approx(77.0)
        assert a_rows.iloc[4]["cost_per_mwh"] == pytest.approx(77.0)
        # Outside the window the year-1 base cost is restored.
        assert a_rows.iloc[0]["cost_per_mwh"] == pytest.approx(50.0)
        assert a_rows.iloc[5]["cost_per_mwh"] == pytest.approx(50.0)
        assert a_rows.iloc[11]["cost_per_mwh"] == pytest.approx(50.0)
        # Plants without a modificacao (and uniform year cost) emit no
        # per-stage cost override — cost_per_mwh is left null.
        termo_b_id = self._make_id_map().thermal_id(20)
        b_rows = df[df["thermal_id"] == termo_b_id]
        assert b_rows["cost_per_mwh"].isna().all()

    def test_chained_potef_finite_then_open_keeps_plant_alive(self, tmp_path) -> None:
        """Regression: two consecutive POTEF windows (finite then open-ended)
        must keep the plant alive across both, matching the source model.  Prior to the
        fix, the first window's data_fim was treated as a decommission date, zeroing
        capacity for every later stage even though a follow-up POTEF re-activated the
        plant."""
        import datetime

        conft, clast, term = _thermal_readers()

        # POTEF window 1: stages 0-3 (Jan-Apr 2023) at 100 MW.
        # POTEF window 2: stage 4 onwards (May 2023+) at 200 MW.
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10],
                "tipo": ["POTEF", "POTEF"],
                "modificacao": [100.0, 200.0],
                "data_inicio": [
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 5, 1),
                ],
                "data_fim": [datetime.datetime(2023, 4, 1), pd.NaT],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        # Use a real expt file path so the optional source is wired in.
        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
            expt=expt_obj,
        )

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        df = table.to_pandas()
        termo_a_id = self._make_id_map().thermal_id(10)
        a_rows = df[df["thermal_id"] == termo_a_id].sort_values("stage_id")

        # FCMAX=90, TEIF=0.05% (fixture stores TEIF in percent units, applied
        # as (100-teif)/100), IP zeroed by step 1.
        # Window 1: max = 100 * 0.9 * (1 - 0.0005) = 89.955
        # Window 2: max = 200 * 0.9 * (1 - 0.0005) = 179.910
        assert a_rows.iloc[0]["max_generation_mw"] == pytest.approx(89.955)
        assert a_rows.iloc[3]["max_generation_mw"] == pytest.approx(89.955)
        # The fix: stages from May 2023 onwards stay alive at the second
        # POTEF capacity, instead of being zeroed by the old step 4b logic.
        assert a_rows.iloc[4]["max_generation_mw"] == pytest.approx(179.910)
        assert a_rows.iloc[11]["max_generation_mw"] == pytest.approx(179.910)

    def test_potef_window_gap_decommissions_plant(self, tmp_path) -> None:
        """A gap between two finite POTEF windows truly decommissions the
        plant — capacity goes to zero for stages outside any window."""
        import datetime

        conft, clast, term = _thermal_readers()

        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10],
                "tipo": ["POTEF", "POTEF"],
                "modificacao": [100.0, 200.0],
                "data_inicio": [
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 8, 1),
                ],
                "data_fim": [
                    datetime.datetime(2023, 3, 1),
                    datetime.datetime(2023, 10, 1),
                ],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
            expt=expt_obj,
        )

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        df = table.to_pandas()
        termo_a_id = self._make_id_map().thermal_id(10)
        a_rows = df[df["thermal_id"] == termo_a_id].sort_values("stage_id")
        # Stage 3 (Apr) and Stage 6 (Jul) sit in the gap → zeroed.
        assert a_rows.iloc[3]["max_generation_mw"] == pytest.approx(0.0)
        assert a_rows.iloc[6]["max_generation_mw"] == pytest.approx(0.0)
        # Stage 0 (Jan) in window 1 → 89.955; stage 7 (Aug) in window 2 → 179.910.
        assert a_rows.iloc[0]["max_generation_mw"] == pytest.approx(89.955)
        assert a_rows.iloc[7]["max_generation_mw"] == pytest.approx(179.910)
        # Stage 11 (Dec) past window 2 → zeroed.
        assert a_rows.iloc[11]["max_generation_mw"] == pytest.approx(0.0)

    def test_modificacao_with_open_end_extends_to_horizon(self, tmp_path) -> None:
        import datetime

        conft, clast, term = _thermal_readers()

        modif_df = pd.DataFrame(
            {
                "codigo_usina": [20],
                "nome_usina": ["TERMO_B"],
                "data_inicio": [datetime.datetime(2023, 7, 1)],
                "data_fim": [pd.NaT],
                "custo": [120.0],
            }
        )
        clast.modificacoes = modif_df
        case = make_case(
            tmp_path,
            conft=conft,
            clast=clast,
            term=term,
            dger=self._make_dger(),
        )

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None

        df = table.to_pandas().sort_values(["thermal_id", "stage_id"])
        termo_b_id = self._make_id_map().thermal_id(20)
        b_rows = df[df["thermal_id"] == termo_b_id].sort_values("stage_id")
        # Stage 5 = Jun 2023 (outside the window) keeps the base 80.0.
        # Stage 6 = Jul 2023 onwards picks up the open-ended override.
        assert b_rows.iloc[5]["cost_per_mwh"] == pytest.approx(80.0)
        assert b_rows.iloc[6]["cost_per_mwh"] == pytest.approx(120.0)
        assert b_rows.iloc[11]["cost_per_mwh"] == pytest.approx(120.0)

    def test_gtmin_availability_freezes_post_study_tail(self, tmp_path) -> None:
        """The "período estático final" freezes thermal min generation at December
        of the last study year (manual p.32-33). A seasonal GTMIN window must NOT
        keep cycling its on/off pattern through the post-study tail."""
        import datetime

        conft, clast, term = _thermal_readers()

        # POTEF gives capacity across the whole horizon. GTMIN is active only
        # Jan-Apr and Sep-Dec 2023 (zero May-Aug). December — the freeze point —
        # is active, so the entire post-study tail must hold GTMIN rather than
        # re-dropping it in the post-study May-Aug months.
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10, 10],
                "tipo": ["POTEF", "GTMIN", "GTMIN"],
                "modificacao": [100.0, 30.0, 30.0],
                "data_inicio": [
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 1, 1),
                    datetime.datetime(2023, 9, 1),
                ],
                "data_fim": [
                    pd.NaT,
                    datetime.datetime(2023, 4, 1),
                    datetime.datetime(2023, 12, 1),
                ],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        dger = self._make_dger()
        dger.num_anos_pos_estudo = 1  # 12 study + 12 post-study stages

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw, conft=conft, clast=clast, term=term, dger=dger, expt=expt_obj
        )

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a_rows = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
            .reset_index(drop=True)
        )
        assert len(a_rows) == 24  # 12 study + 12 post-study

        # In-study: active Jan-Apr (3) and Sep-Dec (11); zero May-Aug (6).
        assert a_rows.loc[3, "min_generation_mw"] == pytest.approx(30.0)
        assert a_rows.loc[6, "min_generation_mw"] == pytest.approx(0.0)
        assert a_rows.loc[11, "min_generation_mw"] == pytest.approx(30.0)
        # Post-study (12-23): frozen at December → 30 every month, INCLUDING the
        # May (16) and Aug (19) 2024 stages the old actual-stage-date logic zeroed.
        assert a_rows.loc[16, "min_generation_mw"] == pytest.approx(30.0)
        assert a_rows.loc[19, "min_generation_mw"] == pytest.approx(30.0)
        assert a_rows.loc[12:23, "min_generation_mw"].tolist() == pytest.approx(
            [30.0] * 12
        )

    def test_cost_modificacao_does_not_leak_into_post_study(self, tmp_path) -> None:
        """A clast cost change dated inside the post-study tail must not apply —
        the static final period freezes cost at December of the last study year."""
        import datetime

        conft, clast, term = _thermal_readers()

        # Future cost change starting Mar 2024 (a post-study stage). Frozen at
        # December 2023, it must never take effect.
        clast.modificacoes = pd.DataFrame(
            {
                "codigo_usina": [10],
                "nome_usina": ["TERMO_A"],
                "data_inicio": [datetime.datetime(2024, 3, 1)],
                "data_fim": [pd.NaT],
                "custo": [999.0],
            }
        )
        dger = self._make_dger()
        dger.num_anos_pos_estudo = 1

        case = make_case(tmp_path, conft=conft, clast=clast, term=term, dger=dger)

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a_rows = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
            .reset_index(drop=True)
        )
        # The year-1 base cost (50.0) holds through the whole post-study tail;
        # the future 999.0 modification is frozen out.
        assert a_rows.loc[11, "cost_per_mwh"] == pytest.approx(50.0)  # Dec 2023
        assert a_rows.loc[14, "cost_per_mwh"] == pytest.approx(50.0)  # Mar 2024
        assert a_rows.loc[12:23, "cost_per_mwh"].tolist() == pytest.approx([50.0] * 12)

    def test_post_study_expansion_freezes_at_online_value(self, tmp_path) -> None:
        """A plant that comes online only in the post-study (POTEF dated in the first
        post-study month) freezes at its *online* terminal December value, not at the
        last study stage (where it does not yet exist) and not at its seasonal profile —
        mirroring the source model's AZULAO II/IV and MANAUS I."""
        import datetime

        conft, clast, term = _thermal_readers()

        # Plant exists only from 2024 (the post-study). GTMIN: a closed seasonal window
        # Jan-May (30) plus an open-ended tail from Jun (80). The source model freezes
        # the whole tail at the terminal December value (the open tail, 80).
        expt_df = pd.DataFrame(
            {
                "codigo_usina": [10, 10, 10],
                "tipo": ["POTEF", "GTMIN", "GTMIN"],
                "modificacao": [100.0, 30.0, 80.0],
                "data_inicio": [
                    datetime.datetime(2024, 1, 1),
                    datetime.datetime(2024, 1, 1),
                    datetime.datetime(2024, 6, 1),
                ],
                "data_fim": [pd.NaT, datetime.datetime(2024, 5, 1), pd.NaT],
            }
        )
        expt_obj = MagicMock()
        expt_obj.expansoes = expt_df

        dger = self._make_dger()
        dger.num_anos_pos_estudo = 1  # study Jan-Dec 2023, post Jan-Dec 2024

        nw = make_nw_files(tmp_path, expt=tmp_path / "expt.dat")
        case = make_case(
            nw, conft=conft, clast=clast, term=term, dger=dger, expt=expt_obj
        )

        from cobre_bridge.converters.thermal import convert_thermal_bounds

        table = convert_thermal_bounds(case, self._make_id_map())
        assert table is not None
        a = (
            table.to_pandas()
            .query("thermal_id == @self._make_id_map().thermal_id(10)")
            .sort_values("stage_id")
            .reset_index(drop=True)
        )
        assert len(a) == 24
        # Study (0-11): plant offline → zero capacity and zero must-run.
        assert a.loc[0, "max_generation_mw"] == pytest.approx(0.0)
        assert a.loc[11, "min_generation_mw"] == pytest.approx(0.0)
        # Post-study (12-23): frozen at the terminal open-ended GTMIN (80) and
        # online — NOT the last-study-stage value (0) and NOT the seasonal Jan
        # value (30). The whole tail is flat at 80.
        assert a.loc[12, "min_generation_mw"] == pytest.approx(80.0)  # Jan 2024
        assert a.loc[12, "max_generation_mw"] > 0.0  # online
        assert a.loc[12:23, "min_generation_mw"].tolist() == pytest.approx([80.0] * 12)


# ---------------------------------------------------------------------------
# Thermal-bound per-stage steps (extracted from the convert_thermal_bounds loop)
# ---------------------------------------------------------------------------


class TestThermalBoundStageSteps:
    """Each of the 6 per-stage steps is now an isolated, testable helper."""

    @staticmethod
    def _state(**overrides: float):
        from cobre_bridge.converters.thermal import _StageInputs

        defaults = {
            "potencia": 100.0,
            "fcmax": 100.0,
            "teif": 0.0,
            "ip": 0.0,
            "gen_min": 0.0,
        }
        defaults.update(overrides)
        return _StageInputs(**defaults)

    def test_step1_zeroes_ip_before_maintenance_end(self) -> None:
        from cobre_bridge.converters.thermal import (
            _step1_zero_ip_before_maintenance,
        )

        state = self._state(ip=8.0)
        _step1_zero_ip_before_maintenance(state, stage_idx=2, maint_end_stage=5)
        assert state.ip == 0.0
        # At/after the maintenance end IP is left untouched.
        state2 = self._state(ip=8.0)
        _step1_zero_ip_before_maintenance(state2, stage_idx=5, maint_end_stage=5)
        assert state2.ip == 8.0

    def test_step2_nulls_potencia_only_for_potef_after_maint_end(self) -> None:
        from cobre_bridge.converters.thermal import _step2_null_potencia_for_potef

        state = self._state(potencia=100.0)
        _step2_null_potencia_for_potef(state, 5, 5, has_potef=True)
        assert state.potencia == 0.0
        # No POTEF → untouched; before maint end → untouched.
        s_no_potef = self._state(potencia=100.0)
        _step2_null_potencia_for_potef(s_no_potef, 5, 5, has_potef=False)
        assert s_no_potef.potencia == 100.0
        s_before = self._state(potencia=100.0)
        _step2_null_potencia_for_potef(s_before, 4, 5, has_potef=True)
        assert s_before.potencia == 100.0

    def test_step3_nulls_gen_min_only_for_gtmin_after_maint_end(self) -> None:
        from cobre_bridge.converters.thermal import _step3_null_gen_min_for_gtmin

        state = self._state(gen_min=50.0)
        _step3_null_gen_min_for_gtmin(state, 5, 5, has_gtmin=True)
        assert state.gen_min == 0.0
        s_no = self._state(gen_min=50.0)
        _step3_null_gen_min_for_gtmin(s_no, 5, 5, has_gtmin=False)
        assert s_no.gen_min == 50.0

    def test_step4_applies_in_file_order_for_closed_window(self) -> None:
        from datetime import date

        from cobre_bridge.converters.thermal import _step4_apply_expt_overrides

        state = self._state()
        overrides = [
            {
                "tipo": "FCMAX",
                "modificacao": 73.38,
                "data_inicio": "2024-01-01",
                "data_fim": "2024-12-01",
            },
            {
                "tipo": "GTMIN",
                "modificacao": 469.62,
                "data_inicio": "2024-01-01",
                "data_fim": "2024-12-01",
            },
        ]
        _step4_apply_expt_overrides(
            state,
            overrides,
            ref_date=date(2024, 6, 1),
            is_post_study=False,
            last_stage_date=date(2030, 12, 1),
        )
        assert state.fcmax == pytest.approx(73.38)
        assert state.gen_min == pytest.approx(469.62)

    def test_step4_skips_window_not_covering_ref_date(self) -> None:
        from datetime import date

        from cobre_bridge.converters.thermal import _step4_apply_expt_overrides

        state = self._state(fcmax=100.0)
        overrides = [
            {
                "tipo": "FCMAX",
                "modificacao": 50.0,
                "data_inicio": "2024-01-01",
                "data_fim": "2024-03-01",
            }
        ]
        _step4_apply_expt_overrides(
            state,
            overrides,
            ref_date=date(2024, 6, 1),  # outside the window
            is_post_study=False,
            last_stage_date=date(2030, 12, 1),
        )
        assert state.fcmax == 100.0

    def test_step4_open_ended_override_blankets_post_study_tail(self) -> None:
        from datetime import date

        from cobre_bridge.converters.thermal import _step4_apply_expt_overrides

        state = self._state(potencia=100.0)
        overrides = [
            {
                "tipo": "POTEF",
                "modificacao": 250.0,
                "data_inicio": "2024-01-01",
                "data_fim": float("nan"),  # open-ended
            }
        ]
        _step4_apply_expt_overrides(
            state,
            overrides,
            ref_date=date(2026, 12, 1),
            is_post_study=True,
            last_stage_date=date(2030, 12, 1),
        )
        assert state.potencia == pytest.approx(250.0)

    def test_step4b_zeroes_out_of_window_stage(self) -> None:
        from datetime import date

        from cobre_bridge.converters.thermal import (
            _step4b_apply_potef_availability,
        )

        state = self._state(potencia=100.0, gen_min=30.0)
        windows = [(date(2024, 1, 1), date(2024, 6, 1))]
        _step4b_apply_potef_availability(state, windows, stage_date=date(2024, 9, 1))
        assert state.potencia == 0.0
        assert state.gen_min == 0.0
        # Inside a window → untouched.
        s_in = self._state(potencia=100.0, gen_min=30.0)
        _step4b_apply_potef_availability(s_in, windows, stage_date=date(2024, 3, 1))
        assert s_in.potencia == 100.0

    def test_step4b_zeroes_expt_plant_without_potef(self) -> None:
        """EXPT plant with modifier-only entries (no POTEF) is not installed.

        The source model reports GERACAO MAXIMA = 0 for such a plant (e.g. LINHARES,
        which carries only a TEIFT entry); without the flag it would fall back to its
        TERM.DAT registry capacity.
        """
        from datetime import date

        from cobre_bridge.converters.thermal import (
            _step4b_apply_potef_availability,
        )

        # No POTEF window + flagged as EXPT-without-POTEF → held out of service.
        state = self._state(potencia=204.0, gen_min=0.0)
        _step4b_apply_potef_availability(
            state, None, stage_date=date(2024, 9, 1), expt_without_potef=True
        )
        assert state.potencia == 0.0
        assert state.gen_min == 0.0

        # No POTEF window + NOT flagged (purely TERM.DAT plant) → untouched.
        s_keep = self._state(potencia=204.0, gen_min=0.0)
        _step4b_apply_potef_availability(
            s_keep, None, stage_date=date(2024, 9, 1), expt_without_potef=False
        )
        assert s_keep.potencia == 204.0

    def test_step4c_drops_gtmin_outside_window(self) -> None:
        """GTMIN applies only inside EXPT windows; outside it is 0 (capacity kept).

        The source model ignores the TERM.DAT GTMIN outside the EXPT GTMIN windows (e.g.
        DO_ATLANTICO: window Sep-Oct, TERM.DAT 201.5 in Nov/Dec, but the source model
        GERACAO MINIMA = 0 there).
        """
        from datetime import date

        from cobre_bridge.converters.thermal import (
            _step4c_apply_gtmin_availability,
        )

        windows = [(date(2024, 9, 1), date(2024, 10, 1))]
        # Inside the window → minimum kept; capacity untouched.
        s_in = self._state(potencia=235.0, gen_min=218.68)
        _step4c_apply_gtmin_availability(s_in, windows, stage_date=date(2024, 9, 1))
        assert s_in.gen_min == 218.68
        assert s_in.potencia == 235.0
        # Outside the window → minimum dropped to 0; capacity untouched.
        s_out = self._state(potencia=235.0, gen_min=201.5)
        _step4c_apply_gtmin_availability(s_out, windows, stage_date=date(2024, 11, 1))
        assert s_out.gen_min == 0.0
        assert s_out.potencia == 235.0

    def test_step4c_drops_gtmin_for_expt_plant_without_gtmin(self) -> None:
        """EXPT plant with no GTMIN entry has no minimum (TERM.DAT GTMIN ignored).

        E.g. JARAQUI / MARLIM AZUL: in EXPT (POTEF/FCMAX) with a nonzero TERM.DAT GTMIN
        but no GTMIN entry → the source model GERACAO MINIMA = 0.
        """
        from datetime import date

        from cobre_bridge.converters.thermal import (
            _step4c_apply_gtmin_availability,
        )

        # No GTMIN window + flagged → minimum dropped, capacity untouched.
        s = self._state(potencia=75.0, gen_min=62.99)
        _step4c_apply_gtmin_availability(
            s, None, stage_date=date(2024, 9, 1), expt_without_gtmin=True
        )
        assert s.gen_min == 0.0
        assert s.potencia == 75.0
        # No GTMIN window + NOT flagged (purely TERM.DAT plant) → untouched.
        s_keep = self._state(potencia=75.0, gen_min=62.99)
        _step4c_apply_gtmin_availability(
            s_keep, None, stage_date=date(2024, 9, 1), expt_without_gtmin=False
        )
        assert s_keep.gen_min == 62.99

    def test_step5_subtracts_maint_reduction_before_maint_end(self) -> None:
        import numpy as np

        from cobre_bridge.converters.thermal import _step5_apply_maint_reduction

        state = self._state(potencia=100.0)
        reduction = np.array([10.0, 20.0, 30.0])
        _step5_apply_maint_reduction(state, reduction, stage_idx=1, maint_end_stage=3)
        assert state.potencia == pytest.approx(80.0)
        # At/after maint end → no reduction.
        s2 = self._state(potencia=100.0)
        _step5_apply_maint_reduction(s2, reduction, stage_idx=3, maint_end_stage=3)
        assert s2.potencia == 100.0

    def test_step6_normal_case(self) -> None:
        from cobre_bridge.converters.thermal import _step6_evaluate_bounds

        state = self._state(potencia=200.0, fcmax=100.0, ip=0.0, teif=0.0, gen_min=50.0)
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        assert max_mw == pytest.approx(200.0)
        assert min_mw == pytest.approx(50.0)
        assert exceeded is False

    def test_step6_honors_gtmin_above_capacity(self) -> None:
        """GTMIN (the inflexible minimum) is honored even when it exceeds the
        FCMAX-derived capacity; the cap is lifted to keep the bound feasible.

        Per source-model, FCMAX and GTMIN are independent and the source model rejects
        min > max. Cobre formerly clamped min DOWN to max, forcing the plant below
        GTMIN; now it honors GTMIN. (ANGRA-1-like numbers: capacity 420.88 < GTMIN
        469.62 → bound [469.62, 469.62], not [420.88, 420.88].)
        """
        from cobre_bridge.converters.thermal import _step6_evaluate_bounds

        state = self._state(
            potencia=420.88, fcmax=100.0, ip=0.0, teif=0.0, gen_min=469.62
        )
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        assert min_mw == pytest.approx(469.62)  # GTMIN honored, not clamped down
        assert max_mw == pytest.approx(469.62)  # cap lifted to GTMIN for feasibility
        assert exceeded is True

    def test_step6_clamps_negative_potencia_to_zero(self) -> None:
        from cobre_bridge.converters.thermal import _step6_evaluate_bounds

        state = self._state(potencia=-5.0, fcmax=100.0, gen_min=10.0)
        min_mw, max_mw, exceeded = _step6_evaluate_bounds(state)
        # gen_min 10 > capacity 0 → honor GTMIN, lift cap.
        assert min_mw == pytest.approx(10.0)
        assert max_mw == pytest.approx(10.0)
        assert exceeded is True


class TestThermalGenerationBounds:
    """``thermal_generation_bounds`` returns the static ``[min_mw, max_mw]``."""

    def test_bounds_from_term_month1(self, tmp_path) -> None:
        from cobre_bridge.converters.thermal import thermal_generation_bounds

        term = MagicMock()
        term.usinas = _make_term_df()
        case = make_case(tmp_path, term=term)

        bounds = thermal_generation_bounds(case)
        # max_mw = potencia_instalada * fator_capacidade_maximo / 100;
        # min_mw = geracao_minima.
        assert bounds[10] == pytest.approx((10.0, 90.0))
        assert bounds[20] == pytest.approx((0.0, 200.0))
        assert bounds[30] == pytest.approx((5.0, 40.0))

    def test_no_usinas_returns_empty(self, tmp_path) -> None:
        from cobre_bridge.converters.thermal import thermal_generation_bounds

        term = MagicMock()
        term.usinas = None
        case = make_case(tmp_path, term=term)

        assert thermal_generation_bounds(case) == {}
