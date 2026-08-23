"""Tests for FPHA emission in the hydro converters.

FPHA is gated on the source model's ``dger`` ``funcao_producao_uhe == 0``
(:attr:`NewaveCase.fpha_enabled`). When on, reservoir plants with storage swing are
emitted as ``model: "fpha"`` so cobre fits the production function; when off, every
plant stays on the constant-productivity path (regression-preserving).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cobre_bridge.converters.hydro import (
    _fpha_computed_config,
    _is_fpha_eligible,
    _parse_fpha_plane_reduction,
    convert_hydro_energy_productivity,
    convert_hydros,
    convert_production_models,
    fpha_eligible_codes,
)
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.productivity import fpha_efficiency
from tests.conftest import (
    _make_confhd_df,
    _make_hidr_cadastro,
    _make_prod_model_dger_mock,
    _make_ree_df,
    make_case,
    make_nw_files,
)

# cobre's phi = K * eta * q * h_net with K = g/1000.
_K = 9.81e-3
# A physically realistic specific productivity -> eta in (0, 1].
_REALISTIC_RHO_ESP = 0.00892


def _reservoir_row(
    *,
    vmin: float = 100.0,
    vmax: float = 1000.0,
    rho_esp: float = _REALISTIC_RHO_ESP,
    a1: float = 0.1,
) -> pd.Series:
    return pd.Series(
        {
            "volume_minimo": vmin,
            "volume_maximo": vmax,
            "a0_volume_cota": 300.0,
            "a1_volume_cota": a1,
            "a2_volume_cota": 0.0,
            "a3_volume_cota": 0.0,
            "a4_volume_cota": 0.0,
            "produtibilidade_especifica": rho_esp,
        }
    )


class TestIsFphaEligible:
    def test_reservoir_with_storage_swing_is_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row()) is True

    def test_run_of_river_zero_storage_is_eligible(self) -> None:
        # Single-volume plant (vmax == vmin): cobre fits it via the single-volume
        # FPHA path (γ_V = 0), so it IS eligible given a poly + ρ_esp.
        assert _is_fpha_eligible(_reservoir_row(vmin=50.0, vmax=50.0)) is True

    def test_degenerate_polynomial_not_eligible(self) -> None:
        # All volume_cota coefficients zero -> no forebay curve to fit.
        row = _reservoir_row()
        for i in range(5):
            row[f"a{i}_volume_cota"] = 0.0
        assert _is_fpha_eligible(row) is False

    def test_zero_rho_esp_not_eligible(self) -> None:
        assert _is_fpha_eligible(_reservoir_row(rho_esp=0.0)) is False


class TestFphaEfficiency:
    def test_realistic_rho_esp_maps_to_fraction(self) -> None:
        eta = fpha_efficiency(_REALISTIC_RHO_ESP, "USINA")
        assert eta == pytest.approx(_REALISTIC_RHO_ESP / _K)
        assert 0.0 < eta <= 1.0

    def test_unphysical_rho_esp_clamped_to_one(self) -> None:
        # rho_esp >> K would imply eta > 1; cobre requires (0, 1].
        assert fpha_efficiency(0.9, "USINA") == 1.0


class TestFphaComputedConfig:
    def _row(self, reg: str) -> pd.Series:
        return pd.Series(
            {
                "tipo_regulacao": reg,
                "volume_minimo": 430.0,
                "volume_maximo": 1781.0,
                "volume_referencia": 900.0,
            }
        )

    def test_monthly_reservoir_uses_full_storage_range(self) -> None:
        cfg = _fpha_computed_config(self._row("M"))
        assert cfg == {
            "source": "computed",
            "fitting_window": {"volume_min_hm3": 430.0, "volume_max_hm3": 1781.0},
        }

    def test_daily_plant_is_single_volume_at_vref(self) -> None:
        cfg = _fpha_computed_config(self._row("D"))
        assert cfg["fitting_window"] == {
            "volume_min_hm3": 900.0,
            "volume_max_hm3": 900.0,
        }

    def test_run_of_river_plant_is_single_volume_at_vref(self) -> None:
        cfg = _fpha_computed_config(self._row("S"))
        assert cfg["fitting_window"] == {
            "volume_min_hm3": 900.0,
            "volume_max_hm3": 900.0,
        }


class TestParseFphaPlaneReduction:
    def _case(self, tmp_path: Path, text: str | None):
        overrides: dict = {}
        if text is not None:
            path = tmp_path / "tratamento-fpha.csv"
            path.write_text(text, encoding="utf-8")
            overrides["tratamento_fpha"] = path
        return make_case(make_nw_files(tmp_path, **overrides))

    def test_angle_method(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path, "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n"
        )
        assert _parse_fpha_plane_reduction(case) == {
            "method": "angle",
            "tolerance_deg": 1.0,
        }

    def test_distance_method(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path,
            "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002\n",
        )
        result = _parse_fpha_plane_reduction(case)
        assert result is not None
        assert result["method"] == "distance"
        assert result["tolerance_pct"] == 0.002
        assert result["n_samples"] >= 1

    def test_commented_lines_ignored(self, tmp_path: Path) -> None:
        case = self._case(
            tmp_path,
            "&HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002\n"
            "HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n",
        )
        assert _parse_fpha_plane_reduction(case) == {
            "method": "angle",
            "tolerance_deg": 1.0,
        }

    def test_no_file_returns_none(self, tmp_path: Path) -> None:
        assert _parse_fpha_plane_reduction(self._case(tmp_path, None)) is None


class TestFphaConverters:
    """Plant 1 = reservoir (FPHA-eligible); plant 2 = non-FPHA.

    Plant 2 has its volume→cota polynomial zeroed (no forebay curve), so it
    stays on the constant-productivity path — the remaining way a plant is
    non-FPHA now that run-of-river plants are eligible.
    """

    def _id_map(self) -> NewaveIdMap:
        return NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])

    def _case(
        self,
        tmp_path: Path,
        *,
        fpha: bool = True,
        tratamento: str | None = None,
        volref: dict[int, dict[int, float]] | None = None,
    ):
        cadastro = _make_hidr_cadastro().copy()
        # Plant 1: realistic specific productivity -> FPHA.
        cadastro.loc[1, "produtibilidade_especifica"] = _REALISTIC_RHO_ESP
        # Plant 2: zero the volume->cota polynomial (no forebay) -> non-FPHA.
        for i in range(5):
            cadastro.loc[2, f"a{i}_volume_cota"] = 0.0
        cadastro.loc[2, "produtibilidade_especifica"] = 0.0088

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        dger = _make_prod_model_dger_mock()
        # dger.dat: =0 FPHA, =1 LINEAR (constant productivity).
        dger.funcao_producao_uhe = 0 if fpha else 1

        overrides: dict = {}
        if tratamento is not None:
            path = tmp_path / "tratamento-fpha.csv"
            path.write_text(tratamento, encoding="utf-8")
            overrides["tratamento_fpha"] = path

        parsed: dict = {
            "hidr": mock_hidr,
            "confhd": mock_confhd,
            "ree": mock_ree,
            "dger": dger,
        }
        if volref is not None:
            rows = [
                {"codigo_usina": code, "mes": month, "valor": value}
                for code, months in volref.items()
                for month, value in months.items()
            ]
            mock_vs = MagicMock()
            mock_vs.volumes = pd.DataFrame(rows)
            parsed["volref_saz"] = mock_vs

        return make_case(make_nw_files(tmp_path, **overrides), **parsed)

    def test_eligible_codes_only_reservoir(self, tmp_path: Path) -> None:
        assert fpha_eligible_codes(self._case(tmp_path)) == {1}

    def test_eligible_codes_empty_when_fpha_off(self, tmp_path: Path) -> None:
        assert fpha_eligible_codes(self._case(tmp_path, fpha=False)) == set()

    def test_hydros_reservoir_is_fpha_with_efficiency(self, tmp_path: Path) -> None:
        hydros = convert_hydros(self._case(tmp_path), self._id_map())["hydros"]
        reservoir = next(h for h in hydros if h["id"] == 0)
        assert reservoir["generation"]["model"] == "fpha"
        assert reservoir["efficiency"] == {
            "type": "constant",
            "value": pytest.approx(_REALISTIC_RHO_ESP / _K),
        }
        # The fpha path also emits the mandatory mirror unit group (ticket 002).
        assert len(reservoir["unit_groups"]) == 1

    def test_hydros_non_fpha_plant_stays_constant(self, tmp_path: Path) -> None:
        hydros = convert_hydros(self._case(tmp_path), self._id_map())["hydros"]
        non_fpha = next(h for h in hydros if h["id"] == 1)
        assert non_fpha["generation"]["model"] == "constant_productivity"
        assert non_fpha["efficiency"] is None
        # The constant-productivity path also emits the mirror unit group.
        assert len(non_fpha["unit_groups"]) == 1

    def test_hydros_fpha_off_all_constant(self, tmp_path: Path) -> None:
        hydros = convert_hydros(self._case(tmp_path, fpha=False), self._id_map())[
            "hydros"
        ]
        assert all(h["generation"]["model"] == "constant_productivity" for h in hydros)
        assert all(h["efficiency"] is None for h in hydros)

    def test_production_models_reservoir_fpha(self, tmp_path: Path) -> None:
        result = convert_production_models(self._case(tmp_path), self._id_map())
        models = {
            m["hydro_id"]: m["stage_ranges"][0] for m in result["production_models"]
        }
        assert models[0]["model"] == "fpha"
        # Plant 1 is tipo_regulacao "M" (vmin=100, vmax=1000) -> multi-volume window.
        assert models[0]["fpha_config"] == {
            "source": "computed",
            "fitting_window": {"volume_min_hm3": 100.0, "volume_max_hm3": 1000.0},
        }
        assert models[0]["reference_volume"] == {"percentile": 0.65}
        assert "productivity_mw_per_m3s" not in models[0]
        assert models[1]["model"] == "constant_productivity"

    def test_production_models_plane_reduction_emitted(self, tmp_path: Path) -> None:
        result = convert_production_models(
            self._case(
                tmp_path,
                tratamento="HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n",
            ),
            self._id_map(),
        )
        assert result["fpha_plane_reduction"] == {
            "method": "angle",
            "tolerance_deg": 1.0,
        }

    def test_production_models_no_plane_reduction_when_fpha_off(
        self, tmp_path: Path
    ) -> None:
        result = convert_production_models(
            self._case(
                tmp_path,
                fpha=False,
                tratamento="HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0\n",
            ),
            self._id_map(),
        )
        assert "fpha_plane_reduction" not in result
        assert all(
            m["stage_ranges"][0]["model"] == "constant_productivity"
            for m in result["production_models"]
        )

    def test_energy_productivity_includes_fpha_plants(self, tmp_path: Path) -> None:
        # FPHA plants must keep an equivalent_productivity row: cobre's
        # energy-conversion build resolves their ρ_eq from this parquet override
        # (build_energy_and_templates feeds the VHA-geometry derivation an empty
        # map, so the parquet is the only working source). Excluding them makes
        # cobre fail at load with "cannot derive ρ_eq".
        table = convert_hydro_energy_productivity(self._case(tmp_path), self._id_map())
        rows = {r["hydro_id"]: r for r in table.to_pylist()}
        assert 0 in rows  # FPHA reservoir is NOT excluded
        assert rows[0]["equivalent_productivity_mw_per_m3s"] is not None
        assert rows[1]["equivalent_productivity_mw_per_m3s"] is not None

    def test_seasonal_reference_volume_from_volref_saz(self, tmp_path: Path) -> None:
        # Plant 1: vmin=100, vmax=1000. volref stores USEFUL storage above vmin.
        case = self._case(
            tmp_path,
            volref={1: {1: 200.0, 6: 850.0, 7: 5000.0}},  # Jan, Jun, Jul (Jul > band)
        )
        models = {
            m["hydro_id"]: m
            for m in convert_production_models(case, self._id_map())[
                "production_models"
            ]
        }
        entry = models[0]
        assert entry["selection_mode"] == "seasonal"
        assert entry["default_model"] == "constant_productivity"
        seasons = {s["season_id"]: s for s in entry["seasons"]}
        assert sorted(seasons) == list(range(12))
        # season_id = month - 1; absolute = clamp(vmin + useful, vmin, vmax).
        assert seasons[0]["reference_volume"] == {"volume_hm3": 300.0}  # Jan: 100+200
        assert seasons[5]["reference_volume"] == {"volume_hm3": 950.0}  # Jun: 100+850
        assert seasons[6]["reference_volume"] == {"volume_hm3": 1000.0}  # Jul clamped
        assert seasons[2]["reference_volume"] == {"volume_hm3": 100.0}  # unset -> vmin
        # FPHA plant: every season carries model fpha + fpha_config (with window).
        assert all(s["model"] == "fpha" for s in entry["seasons"])
        assert all(
            s["fpha_config"]
            == {
                "source": "computed",
                "fitting_window": {"volume_min_hm3": 100.0, "volume_max_hm3": 1000.0},
            }
            for s in entry["seasons"]
        )

    def test_nonfpha_downstream_plant_with_volref_is_seasonal_constant(
        self, tmp_path: Path
    ) -> None:
        # Plant 2 is non-FPHA but still gets seasonal reference volumes so it is
        # a correct backwater reference for an upstream FPHA plant.
        # (At least one non-zero month, else volref_saz treats the row as absent.)
        case = self._case(tmp_path, volref={2: {1: 10.0}})
        models = {
            m["hydro_id"]: m
            for m in convert_production_models(case, self._id_map())[
                "production_models"
            ]
        }
        entry = models[1]
        assert entry["selection_mode"] == "seasonal"
        assert all(s["model"] == "constant_productivity" for s in entry["seasons"])
        assert all("fpha_config" not in s for s in entry["seasons"])
        assert all("reference_volume" in s for s in entry["seasons"])

    def test_volref_not_consulted_when_fpha_off(self, tmp_path: Path) -> None:
        case = self._case(tmp_path, fpha=False, volref={1: {1: 200.0}})
        models = convert_production_models(case, self._id_map())["production_models"]
        # Linear case: volref ignored, every plant stays a single constant range.
        assert all(m["selection_mode"] == "stage_ranges" for m in models)
        assert all(
            m["stage_ranges"][0]["model"] == "constant_productivity" for m in models
        )
