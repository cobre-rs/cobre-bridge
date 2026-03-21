"""Unit tests for temporal and stochastic conversion functions.

All inewave I/O is mocked via ``unittest.mock.patch`` so no real NEWAVE
files are required.  Synthetic DataFrames exercise the core logic of each
converter.
"""

from __future__ import annotations

import calendar
import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_id_map_hydros(hydro_codes: list[int]) -> NewaveIdMap:
    return NewaveIdMap(subsystem_ids=[], hydro_codes=hydro_codes, thermal_codes=[])


def _make_id_map_buses(subsystem_ids: list[int]) -> NewaveIdMap:
    return NewaveIdMap(subsystem_ids=subsystem_ids, hydro_codes=[], thermal_codes=[])


def _make_nw_files(
    tmp_path,
    *,
    vazpast: Path | None = None,
    dsvagua: Path | None = None,
) -> NewaveFiles:
    """Construct a NewaveFiles instance pointing into tmp_path.

    Does not access the filesystem — passes paths directly to the constructor.
    """
    return NewaveFiles(
        directory=tmp_path,
        dger=tmp_path / "dger.dat",
        confhd=tmp_path / "confhd.dat",
        conft=tmp_path / "conft.dat",
        sistema=tmp_path / "sistema.dat",
        clast=tmp_path / "clast.dat",
        term=tmp_path / "term.dat",
        ree=tmp_path / "ree.dat",
        patamar=tmp_path / "patamar.dat",
        hidr=tmp_path / "hidr.dat",
        vazoes=tmp_path / "vazoes.dat",
        modif=None,
        ghmin=None,
        penalid=None,
        vazpast=vazpast,
        dsvagua=dsvagua,
        curva=None,
    )


def _make_dger_mock(
    *,
    mes_inicio: int = 1,
    ano_inicio: int = 2020,
    num_anos: int = 5,
    num_anos_pre: int = 0,
    num_anos_pos: int = 0,
    num_forwards: int = 20,
    num_max_iteracoes: int = 200,
    num_series: int = 500,
    taxa_de_desconto: float = 12.0,
    num_aberturas: int = 10,
) -> MagicMock:
    dger = MagicMock()
    dger.mes_inicio_estudo = mes_inicio
    dger.ano_inicio_estudo = ano_inicio
    dger.num_anos_estudo = num_anos
    dger.num_anos_pre_estudo = num_anos_pre
    dger.num_anos_pos_estudo = num_anos_pos
    dger.num_forwards = num_forwards
    dger.num_max_iteracoes = num_max_iteracoes
    dger.num_series_sinteticas = num_series
    dger.taxa_de_desconto = taxa_de_desconto
    dger.num_aberturas = num_aberturas
    return dger


def _make_patamar_mock_single() -> MagicMock:
    """Single block: fraction=1.0 for every calendar month."""
    rows = []
    for month in range(1, 13):
        rows.append(
            {
                "data": datetime.datetime(2020, month, 1),
                "patamar": 1,
                "valor": 1.0,
            }
        )
    df = pd.DataFrame(rows)
    pat = MagicMock()
    pat.duracao_mensal_patamares = df
    pat.numero_patamares = 1
    return pat


def _make_patamar_mock_three_blocks() -> MagicMock:
    """Three blocks: fractions 0.3, 0.4, 0.3 for every calendar month."""
    fractions = [0.3, 0.4, 0.3]
    rows = []
    for month in range(1, 13):
        for pat_idx, frac in enumerate(fractions, start=1):
            rows.append(
                {
                    "data": datetime.datetime(2020, month, 1),
                    "patamar": pat_idx,
                    "valor": frac,
                }
            )
    df = pd.DataFrame(rows)
    pat = MagicMock()
    pat.duracao_mensal_patamares = df
    pat.numero_patamares = 3
    return pat


def _setup_dger_and_patamar(mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar):
    """Create dummy files and configure mock read() returns."""
    (tmp_path / "dger.dat").touch()
    (tmp_path / "patamar.dat").touch()
    mock_dger_cls.read.return_value = dger
    mock_patamar_cls.read.return_value = patamar


# ---------------------------------------------------------------------------
# Tests: convert_stages
# ---------------------------------------------------------------------------


class TestConvertStagesSingleBlock:
    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_stage_count_five_years(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos=5)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        assert len(result["stages"]) == 60

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_stage_ids_are_sequential_zero_based(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos=5)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        ids = [s["id"] for s in result["stages"]]
        assert ids == list(range(60))

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_first_stage_dates_jan_2020(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=5)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        stage0 = result["stages"][0]
        assert stage0["start_date"] == "2020-01-01"
        assert stage0["end_date"] == "2020-02-01"

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_last_stage_end_date(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        # 5 years starting Jan 2020 -> last stage is Dec 2024, end = 2025-01-01.
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=5)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        last = result["stages"][-1]
        assert last["end_date"] == "2025-01-01"

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_block_hours_january(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=1)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        stage0 = result["stages"][0]  # January 2020
        assert len(stage0["blocks"]) == 1
        block = stage0["blocks"][0]
        # January 2020 has 31 days * 24 h = 744 h.
        assert block["hours"] == pytest.approx(744.0)
        assert block["name"] == "SINGLE"
        assert block["id"] == 0

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_num_scenarios_from_dger(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_aberturas=50)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert stage["num_scenarios"] == 50

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_discount_rate_percent_to_decimal(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(taxa_de_desconto=12.0)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        assert result["policy_graph"]["annual_discount_rate"] == pytest.approx(0.12)

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_policy_graph_is_finite_horizon(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock()
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        assert result["policy_graph"]["type"] == "finite_horizon"

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_transitions_are_linear(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos=2)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        transitions = result["policy_graph"]["transitions"]
        # 24 stages -> 23 transitions.
        assert len(transitions) == 23
        # Each transition goes from i to i+1 with probability 1.0.
        for i, t in enumerate(transitions):
            assert t["source_id"] == i
            assert t["target_id"] == i + 1
            assert t["probability"] == pytest.approx(1.0)

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_risk_measure_is_expectation(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock()
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert stage["risk_measure"] == "expectation"

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_zero_study_years_raises_value_error(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos=0)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        with pytest.raises(ValueError, match="zero study years"):
            convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))


class TestConvertStagesThreeBlocks:
    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_three_blocks_present(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos=1)
        patamar = _make_patamar_mock_three_blocks()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        for stage in result["stages"]:
            assert len(stage["blocks"]) == 3

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_three_blocks_names(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos=1)
        patamar = _make_patamar_mock_three_blocks()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        stage0 = result["stages"][0]
        names = [b["name"] for b in stage0["blocks"]]
        assert names == ["HEAVY", "MEDIUM", "LIGHT"]

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_block_hours_sum_to_month_hours(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(mes_inicio=1, ano_inicio=2020, num_anos=1)
        patamar = _make_patamar_mock_three_blocks()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        for i, stage in enumerate(result["stages"]):
            year = 2020
            month = i + 1
            expected_total = float(calendar.monthrange(year, month)[1] * 24)
            total_hours = sum(b["hours"] for b in stage["blocks"])
            assert total_hours == pytest.approx(expected_total)


class TestConvertStagesPreStudy:
    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_pre_study_stages_generated(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(
            mes_inicio=1, ano_inicio=2020, num_anos=5, num_anos_pre=1
        )
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        pre = result.get("pre_study_stages", [])
        assert len(pre) == 12

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_pre_study_ids_are_negative(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(
            mes_inicio=1, ano_inicio=2020, num_anos=5, num_anos_pre=1
        )
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        pre = result["pre_study_stages"]
        assert all(s["id"] < 0 for s in pre)
        ids = [s["id"] for s in pre]
        assert ids == list(range(-12, 0))

    @patch("cobre_bridge.converters.temporal.Patamar")
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_no_pre_study_key_when_zero(
        self, mock_dger_cls, mock_patamar_cls, tmp_path
    ) -> None:
        dger = _make_dger_mock(num_anos_pre=0)
        patamar = _make_patamar_mock_single()
        _setup_dger_and_patamar(
            mock_dger_cls, mock_patamar_cls, tmp_path, dger, patamar
        )

        from cobre_bridge.converters.temporal import convert_stages

        result = convert_stages(_make_nw_files(tmp_path), _make_id_map_hydros([]))
        assert "pre_study_stages" not in result


# ---------------------------------------------------------------------------
# Tests: convert_config
# ---------------------------------------------------------------------------


class TestConvertConfig:
    @patch("cobre_bridge.converters.temporal.Dger")
    def test_forward_passes(self, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "dger.dat").touch()
        dger = _make_dger_mock(num_forwards=20)
        mock_dger_cls.read.return_value = dger

        from cobre_bridge.converters.temporal import convert_config

        result = convert_config(_make_nw_files(tmp_path))
        assert result["training"]["forward_passes"] == 20

    @patch("cobre_bridge.converters.temporal.Dger")
    def test_iteration_limit(self, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "dger.dat").touch()
        dger = _make_dger_mock(num_max_iteracoes=200)
        mock_dger_cls.read.return_value = dger

        from cobre_bridge.converters.temporal import convert_config

        result = convert_config(_make_nw_files(tmp_path))
        rules = result["training"]["stopping_rules"]
        assert len(rules) == 1
        assert rules[0]["type"] == "iteration_limit"
        assert rules[0]["limit"] == 200

    @patch("cobre_bridge.converters.temporal.Dger")
    def test_simulation_enabled(self, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "dger.dat").touch()
        dger = _make_dger_mock()
        mock_dger_cls.read.return_value = dger

        from cobre_bridge.converters.temporal import convert_config

        result = convert_config(_make_nw_files(tmp_path))
        assert result["simulation"]["enabled"] is True

    @patch("cobre_bridge.converters.temporal.Dger")
    def test_num_scenarios_from_num_series(self, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "dger.dat").touch()
        dger = _make_dger_mock(num_series=500)
        mock_dger_cls.read.return_value = dger

        from cobre_bridge.converters.temporal import convert_config

        result = convert_config(_make_nw_files(tmp_path))
        assert result["simulation"]["num_scenarios"] == 500


# ---------------------------------------------------------------------------
# Tests: convert_inflow_stats
# ---------------------------------------------------------------------------


def _make_vazoes_mock(
    num_years: int = 10,
    postos: list[int] | None = None,
    start_year: int = 1931,
) -> MagicMock:
    """Build a synthetic Vazoes mock with ``num_years`` of monthly data.

    Parameters
    ----------
    num_years:
        Number of years in the historical record.
    postos:
        List of gauging station codes (columns in the DataFrame).
    start_year:
        First year of the historical record.
    """
    if postos is None:
        postos = [1, 2]

    rows = []
    rng = np.random.default_rng(42)
    for year in range(start_year, start_year + num_years):
        for month in range(1, 13):
            row = {"data": datetime.datetime(year, month, 1)}
            for posto in postos:
                row[posto] = float(rng.uniform(50.0, 500.0))
            rows.append(row)

    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.vazoes = df
    return mock


def _make_confhd_mock(hydro_to_posto: dict[int, int]) -> MagicMock:
    """Build a synthetic Confhd mock mapping hydro codes to postos."""
    rows = [
        {
            "codigo_usina": code,
            "posto": posto,
            "nome_usina": f"PLANT_{code}",
            "usina_existente": "EX",
            "codigo_usina_jusante": 0,
        }
        for code, posto in hydro_to_posto.items()
    ]
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.usinas = df
    return mock


def _make_dger_inflow_mock(
    mes_inicio_estudo: int = 1, num_anos_estudo: int = 10
) -> MagicMock:
    """Build a minimal Dger mock for inflow stats tests."""
    mock = MagicMock()
    mock.mes_inicio_estudo = mes_inicio_estudo
    mock.num_anos_estudo = num_anos_estudo
    mock.num_anos_pos_estudo = 0
    return mock


class TestConvertInflowStats:
    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    @patch("cobre_bridge.converters.stochastic.Vazoes")
    def test_returns_pyarrow_table(
        self, mock_vazoes_cls, mock_confhd_cls, mock_dger_cls, tmp_path
    ) -> None:
        (tmp_path / "vazoes.dat").touch()
        (tmp_path / "confhd.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(
            num_years=10, postos=[1, 2]
        )
        mock_confhd_cls.read.return_value = _make_confhd_mock({1: 1, 2: 2})
        mock_dger_cls.read.return_value = _make_dger_inflow_mock()
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(_make_nw_files(tmp_path), id_map)
        assert isinstance(result, pa.Table)

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    @patch("cobre_bridge.converters.stochastic.Vazoes")
    def test_schema_columns(
        self, mock_vazoes_cls, mock_confhd_cls, mock_dger_cls, tmp_path
    ) -> None:
        (tmp_path / "vazoes.dat").touch()
        (tmp_path / "confhd.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(
            num_years=10, postos=[1, 2]
        )
        mock_confhd_cls.read.return_value = _make_confhd_mock({1: 1, 2: 2})
        mock_dger_cls.read.return_value = _make_dger_inflow_mock()
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(_make_nw_files(tmp_path), id_map)
        assert result.column_names == ["hydro_id", "stage_id", "mean_m3s", "std_m3s"]

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    @patch("cobre_bridge.converters.stochastic.Vazoes")
    def test_column_types(
        self, mock_vazoes_cls, mock_confhd_cls, mock_dger_cls, tmp_path
    ) -> None:
        (tmp_path / "vazoes.dat").touch()
        (tmp_path / "confhd.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(
            num_years=10, postos=[1, 2]
        )
        mock_confhd_cls.read.return_value = _make_confhd_mock({1: 1, 2: 2})
        mock_dger_cls.read.return_value = _make_dger_inflow_mock()
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(_make_nw_files(tmp_path), id_map)
        assert result.schema.field("hydro_id").type == pa.int32()
        assert result.schema.field("stage_id").type == pa.int32()
        assert result.schema.field("mean_m3s").type == pa.float64()
        assert result.schema.field("std_m3s").type == pa.float64()

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    @patch("cobre_bridge.converters.stochastic.Vazoes")
    def test_january_mean_correct(
        self, mock_vazoes_cls, mock_confhd_cls, mock_dger_cls, tmp_path
    ) -> None:
        """mean_m3s for January stages must equal the mean of all January values."""
        (tmp_path / "vazoes.dat").touch()
        (tmp_path / "confhd.dat").touch()

        num_years = 10
        rows = []
        jan_vals = []
        rng = np.random.default_rng(0)
        for year in range(1931, 1931 + num_years):
            for month in range(1, 13):
                v = float(rng.uniform(100.0, 400.0))
                rows.append({"data": datetime.datetime(year, month, 1), 1: v})
                if month == 1:
                    jan_vals.append(v)
        df = pd.DataFrame(rows)
        mock_vazoes = MagicMock()
        mock_vazoes.vazoes = df
        mock_vazoes_cls.read.return_value = mock_vazoes
        mock_confhd_cls.read.return_value = _make_confhd_mock({1: 1})
        mock_dger_cls.read.return_value = _make_dger_inflow_mock(
            mes_inicio_estudo=1, num_anos_estudo=num_years
        )

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(_make_nw_files(tmp_path), id_map)
        df_result = result.to_pydict()

        expected_jan_mean = float(np.mean(jan_vals))
        jan_stage_ids = [
            sid
            for sid, mean in zip(df_result["stage_id"], df_result["mean_m3s"])
            if abs(mean - expected_jan_mean) < 1e-9
        ]
        assert len(jan_stage_ids) > 0, "No January stage found with expected mean"

    @patch("cobre_bridge.converters.stochastic.Confhd")
    @patch("cobre_bridge.converters.stochastic.Vazoes")
    def test_empty_vazoes_raises_file_not_found(
        self, mock_vazoes_cls, mock_confhd_cls, tmp_path
    ) -> None:
        (tmp_path / "vazoes.dat").touch()
        (tmp_path / "confhd.dat").touch()
        mock_obj = MagicMock()
        mock_obj.vazoes = pd.DataFrame()
        mock_vazoes_cls.read.return_value = mock_obj
        mock_confhd_cls.read.return_value = _make_confhd_mock({})

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_inflow_stats

        with pytest.raises(FileNotFoundError, match="vazoes.dat not found or empty"):
            convert_inflow_stats(_make_nw_files(tmp_path), id_map)

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    @patch("cobre_bridge.converters.stochastic.Vazoes")
    def test_missing_posto_column_produces_zero(
        self, mock_vazoes_cls, mock_confhd_cls, mock_dger_cls, tmp_path
    ) -> None:
        """Hydro whose posto column is absent in vazoes -> mean/std = 0.0."""
        (tmp_path / "vazoes.dat").touch()
        (tmp_path / "confhd.dat").touch()
        mock_vazoes_cls.read.return_value = _make_vazoes_mock(num_years=5, postos=[1])
        mock_confhd_cls.read.return_value = _make_confhd_mock({1: 1, 2: 99})
        mock_dger_cls.read.return_value = _make_dger_inflow_mock(num_anos_estudo=5)
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_inflow_stats

        result = convert_inflow_stats(_make_nw_files(tmp_path), id_map)
        df = result.to_pydict()
        hydro2_means = [m for hid, m in zip(df["hydro_id"], df["mean_m3s"]) if hid == 1]
        assert all(m == 0.0 for m in hydro2_means)


# ---------------------------------------------------------------------------
# Tests: convert_load_stats
# ---------------------------------------------------------------------------


def _make_sistema_mock(
    subsystem_codes: list[int],
    num_months: int = 12,
    start_year: int = 2020,
    start_month: int = 1,
) -> MagicMock:
    """Build a synthetic Sistema mock with load data."""
    rows = []
    rng = np.random.default_rng(7)
    year, month = start_year, start_month
    for _ in range(num_months):
        for code in subsystem_codes:
            rows.append(
                {
                    "codigo_submercado": code,
                    "data": datetime.datetime(year, month, 1),
                    "valor": float(rng.uniform(1000.0, 5000.0)),
                }
            )
        month += 1
        if month > 12:
            month = 1
            year += 1
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.mercado_energia = df
    return mock


def _make_load_stats_dger_mock(
    *,
    mes_inicio: int = 1,
    ano_inicio: int = 2020,
    num_anos: int = 5,
    num_anos_pos: int = 0,
) -> MagicMock:
    """Minimal Dger mock for convert_load_stats tests."""
    dger = MagicMock()
    dger.mes_inicio_estudo = mes_inicio
    dger.ano_inicio_estudo = ano_inicio
    dger.num_anos_estudo = num_anos
    dger.num_anos_pos_estudo = num_anos_pos
    return dger


class TestConvertLoadStats:
    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_returns_pyarrow_table(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        mock_sistema_cls.read.return_value = _make_sistema_mock(
            subsystem_codes=[1, 2, 3, 4], num_months=60
        )
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=5)
        id_map = _make_id_map_buses([1, 2, 3, 4])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        assert isinstance(result, pa.Table)

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_schema_columns(self, mock_sistema_cls, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        mock_sistema_cls.read.return_value = _make_sistema_mock(
            subsystem_codes=[1, 2], num_months=12
        )
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=1)
        id_map = _make_id_map_buses([1, 2])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        assert result.column_names == ["bus_id", "stage_id", "mean_mw", "std_mw"]

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_column_types(self, mock_sistema_cls, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        mock_sistema_cls.read.return_value = _make_sistema_mock(
            subsystem_codes=[1, 2], num_months=12
        )
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=1)
        id_map = _make_id_map_buses([1, 2])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        assert result.schema.field("bus_id").type == pa.int32()
        assert result.schema.field("stage_id").type == pa.int32()
        assert result.schema.field("mean_mw").type == pa.float64()
        assert result.schema.field("std_mw").type == pa.float64()

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_row_count_four_subsystems_60_months(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        mock_sistema_cls.read.return_value = _make_sistema_mock(
            subsystem_codes=[1, 2, 3, 4], num_months=60
        )
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=5)
        id_map = _make_id_map_buses([1, 2, 3, 4])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        assert result.num_rows == 4 * 60

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_std_mw_all_zeros(self, mock_sistema_cls, mock_dger_cls, tmp_path) -> None:
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        mock_sistema_cls.read.return_value = _make_sistema_mock(
            subsystem_codes=[1, 2, 3, 4], num_months=60
        )
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=5)
        id_map = _make_id_map_buses([1, 2, 3, 4])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        std_vals = result.column("std_mw").to_pylist()
        assert all(v == 0.0 for v in std_vals)

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_stage_ids_per_bus_sequential(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        mock_sistema_cls.read.return_value = _make_sistema_mock(
            subsystem_codes=[1, 2], num_months=12
        )
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=1)
        id_map = _make_id_map_buses([1, 2])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        df = result.to_pydict()
        for bus_id in [0, 1]:
            stages = [
                sid for bid, sid in zip(df["bus_id"], df["stage_id"]) if bid == bus_id
            ]
            assert stages == list(range(12))

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Sistema")
    def test_mean_mw_values_match_source(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        """mean_mw values must equal the raw 'valor' from mercado_energia."""
        (tmp_path / "sistema.dat").touch()
        (tmp_path / "dger.dat").touch()
        # Deterministic load: 2 subsystems, 2 months.
        rows = [
            {
                "codigo_submercado": 1,
                "data": datetime.datetime(2020, 1, 1),
                "valor": 3000.0,
            },
            {
                "codigo_submercado": 1,
                "data": datetime.datetime(2020, 2, 1),
                "valor": 2800.0,
            },
            {
                "codigo_submercado": 2,
                "data": datetime.datetime(2020, 1, 1),
                "valor": 1500.0,
            },
            {
                "codigo_submercado": 2,
                "data": datetime.datetime(2020, 2, 1),
                "valor": 1600.0,
            },
        ]
        mock = MagicMock()
        mock.mercado_energia = pd.DataFrame(rows)
        mock_sistema_cls.read.return_value = mock
        # 1 year study so stage 0=Jan, stage 1=Feb.
        mock_dger_cls.read.return_value = _make_load_stats_dger_mock(num_anos=1)
        id_map = _make_id_map_buses([1, 2])

        from cobre_bridge.converters.stochastic import convert_load_stats

        result = convert_load_stats(_make_nw_files(tmp_path), id_map)
        df = result.to_pydict()

        # Bus 0 (subsystem 1), stage 0 -> 3000.0; stage 1 -> 2800.0.
        bus0_means = [m for bid, m in zip(df["bus_id"], df["mean_mw"]) if bid == 0]
        assert bus0_means[0] == pytest.approx(3000.0)
        assert bus0_means[1] == pytest.approx(2800.0)


# ---------------------------------------------------------------------------
# Helpers for convert_past_inflows tests
# ---------------------------------------------------------------------------


def _make_vazpast_mock(
    postos: list[int],
    num_months: int = 12,
    end_year: int = 2019,
    end_month: int = 12,
) -> MagicMock:
    """Build a synthetic Vazpast mock with monthly tendency data.

    The ``tendencia`` DataFrame has columns ``codigo_usina``, ``nome_usina``,
    ``mes`` (1-12), and ``valor``, with one row per (posto, calendar month).
    This matches the format expected by ``convert_past_inflows`` after the
    production code was updated to read ``vazpast_obj.tendencia`` instead of
    ``vazpast_obj.vazoes``.

    ``end_year`` and ``end_month`` are accepted for API compatibility but
    the tendencia format is calendar-month based, not date based.
    ``num_months`` is capped at 12 (the tendencia has at most one row per
    calendar month per plant).
    """
    rng = np.random.default_rng(99)
    rows = []
    n_months = min(num_months, 12)
    for posto in postos:
        for mes in range(1, n_months + 1):
            rows.append(
                {
                    "codigo_usina": posto,
                    "nome_usina": f"PLANT_{posto}",
                    "mes": mes,
                    "valor": float(rng.uniform(50.0, 500.0)),
                }
            )
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.tendencia = df
    return mock


def _make_confhd_posto_mock(posto_to_code: dict[int, int]) -> MagicMock:
    """Build a Confhd mock mapping postos to hydro codes.

    Includes all columns needed by _build_upstream_postos (cascade logic).
    All plants are headwater (no upstream) unless overridden.
    """
    rows = [
        {
            "posto": p,
            "codigo_usina": c,
            "nome_usina": f"PLANT_{c}",
            "usina_existente": "EX",
            "codigo_usina_jusante": 0,
        }
        for p, c in posto_to_code.items()
    ]
    df = pd.DataFrame(rows)
    mock = MagicMock()
    mock.usinas = df
    return mock


# ---------------------------------------------------------------------------
# Tests: convert_recent_inflow_lags (vazpast.dat → initial_conditions)
# ---------------------------------------------------------------------------


class TestConvertRecentInflowLagsNoFile:
    def test_returns_empty_when_vazpast_absent(self, tmp_path: Path) -> None:
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_recent_inflow_lags

        result = convert_recent_inflow_lags(_make_nw_files(tmp_path), id_map)
        assert result == []


class TestConvertRecentInflowLagsWithFile:
    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    def test_returns_12_lags_per_hydro(
        self, mock_confhd_cls, mock_dger_cls, tmp_path: Path
    ) -> None:
        from unittest.mock import patch as _patch

        (tmp_path / "vazpast.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        mock_confhd_cls.read.return_value = _make_confhd_posto_mock({1: 1, 2: 2})
        dger_mock = MagicMock()
        dger_mock.mes_inicio_estudo = 3  # March
        mock_dger_cls.read.return_value = dger_mock

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1, 2], thermal_codes=[])
        vazpast_mock = _make_vazpast_mock(postos=[1, 2], num_months=12)

        from cobre_bridge.converters.stochastic import convert_recent_inflow_lags

        with _patch("inewave.newave.Vazpast", create=True) as mock_vp_cls:
            mock_vp_cls.read.return_value = vazpast_mock
            result = convert_recent_inflow_lags(
                _make_nw_files(tmp_path, vazpast=tmp_path / "vazpast.dat"),
                id_map,
            )

        assert len(result) == 2
        assert all(len(e["values_m3s"]) == 12 for e in result)

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    def test_lag_order_march_start(
        self, mock_confhd_cls, mock_dger_cls, tmp_path: Path
    ) -> None:
        """Study starts March: lag1=Feb, lag2=Jan, lag3=Dec, ..., lag12=Mar."""
        from unittest.mock import patch as _patch

        (tmp_path / "vazpast.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        mock_confhd_cls.read.return_value = _make_confhd_posto_mock({1: 1})
        dger_mock = MagicMock()
        dger_mock.mes_inicio_estudo = 3
        mock_dger_cls.read.return_value = dger_mock

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1], thermal_codes=[])

        # Build vazpast where each month has a distinct value = month number
        rows = [
            {"codigo_usina": 1, "nome_usina": "P1", "mes": m, "valor": float(m * 100)}
            for m in range(1, 13)
        ]
        vp_mock = MagicMock()
        vp_mock.tendencia = pd.DataFrame(rows)

        from cobre_bridge.converters.stochastic import convert_recent_inflow_lags

        with _patch("inewave.newave.Vazpast", create=True) as mock_vp_cls:
            mock_vp_cls.read.return_value = vp_mock
            result = convert_recent_inflow_lags(
                _make_nw_files(tmp_path, vazpast=tmp_path / "vazpast.dat"),
                id_map,
            )

        lags = result[0]["values_m3s"]
        # lag1=Feb(200), lag2=Jan(100), lag3=Dec(1200), lag4=Nov(1100), ...
        assert lags[0] == 200.0  # February
        assert lags[1] == 100.0  # January
        assert lags[2] == 1200.0  # December
        assert lags[11] == 300.0  # March (12 months back)

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    def test_unknown_plant_skipped(
        self, mock_confhd_cls, mock_dger_cls, tmp_path: Path
    ) -> None:
        from unittest.mock import patch as _patch

        (tmp_path / "vazpast.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        mock_confhd_cls.read.return_value = _make_confhd_posto_mock({1: 1})
        dger_mock = MagicMock()
        dger_mock.mes_inicio_estudo = 1
        mock_dger_cls.read.return_value = dger_mock

        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1], thermal_codes=[])
        # vazpast has plant 1 (in id_map) and plant 99 (not in id_map)
        vazpast_mock = _make_vazpast_mock(postos=[1, 99], num_months=12)

        from cobre_bridge.converters.stochastic import convert_recent_inflow_lags

        with _patch("inewave.newave.Vazpast", create=True) as mock_vp_cls:
            mock_vp_cls.read.return_value = vazpast_mock
            result = convert_recent_inflow_lags(
                _make_nw_files(tmp_path, vazpast=tmp_path / "vazpast.dat"),
                id_map,
            )

        assert len(result) == 1
        assert result[0]["hydro_id"] == 0

    @patch("cobre_bridge.converters.stochastic.Dger")
    @patch("cobre_bridge.converters.stochastic.Confhd")
    def test_parse_failure_returns_empty(
        self, mock_confhd_cls, mock_dger_cls, tmp_path: Path
    ) -> None:
        """If Vazpast.read() raises, return empty list gracefully."""
        from unittest.mock import patch as _patch

        (tmp_path / "vazpast.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        mock_confhd_cls.read.return_value = _make_confhd_posto_mock({1: 1})
        dger_mock = MagicMock()
        dger_mock.mes_inicio_estudo = 1
        mock_dger_cls.read.return_value = dger_mock
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[1], thermal_codes=[])

        from cobre_bridge.converters.stochastic import convert_recent_inflow_lags

        with _patch("inewave.newave.Vazpast", create=True) as mock_vp_cls:
            mock_vp_cls.read.side_effect = RuntimeError("bad file")
            result = convert_recent_inflow_lags(
                _make_nw_files(tmp_path, vazpast=tmp_path / "vazpast.dat"),
                id_map,
            )

        assert result == []
