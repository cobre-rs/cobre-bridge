"""Unit tests for NEWAVE entity conversion functions.

All inewave I/O is mocked via ``unittest.mock.patch`` so no real NEWAVE
files are required.  The tests use small synthetic DataFrames that cover
the logic exercised by each converter.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cobre_bridge.id_map import NewaveIdMap


class TestNewaveIdMap:
    def test_bus_id_remapping(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[3, 1, 4, 2],
            hydro_codes=[],
            thermal_codes=[],
        )
        assert id_map.bus_id(1) == 0
        assert id_map.bus_id(2) == 1
        assert id_map.bus_id(3) == 2
        assert id_map.bus_id(4) == 3

    def test_hydro_id_remapping(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[],
            hydro_codes=[10, 5, 20],
            thermal_codes=[],
        )
        assert id_map.hydro_id(5) == 0
        assert id_map.hydro_id(10) == 1
        assert id_map.hydro_id(20) == 2

    def test_thermal_id_remapping(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[],
            hydro_codes=[],
            thermal_codes=[7, 3, 15],
        )
        assert id_map.thermal_id(3) == 0
        assert id_map.thermal_id(7) == 1
        assert id_map.thermal_id(15) == 2

    def test_unknown_key_raises_key_error(self) -> None:
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[1])
        with pytest.raises(KeyError):
            id_map.bus_id(99)
        with pytest.raises(KeyError):
            id_map.hydro_id(99)
        with pytest.raises(KeyError):
            id_map.thermal_id(99)

    def test_all_hydro_codes_sorted(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[], hydro_codes=[30, 10, 20], thermal_codes=[]
        )
        assert id_map.all_hydro_codes == [10, 20, 30]

    def test_deterministic_regardless_of_input_order(self) -> None:
        a = NewaveIdMap(subsystem_ids=[2, 1, 3], hydro_codes=[], thermal_codes=[])
        b = NewaveIdMap(subsystem_ids=[3, 1, 2], hydro_codes=[], thermal_codes=[])
        assert a.bus_id(1) == b.bus_id(1)
        assert a.bus_id(2) == b.bus_id(2)
        assert a.bus_id(3) == b.bus_id(3)


def _make_confhd_df() -> pd.DataFrame:
    """Two hydros: plant 1 upstream of plant 2, in REE 1 (subsystem 1)."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2],
            "nome_usina": ["USINA_A", "USINA_B"],
            "posto": [1, 2],
            "codigo_usina_jusante": [pd.NA, 1],
            "ree": [1, 1],
            "volume_inicial_percentual": [50.0, 75.0],
            "usina_existente": ["EX", "EX"],
            "usina_modificada": [0, 0],
        }
    )


def _make_hidr_cadastro() -> pd.DataFrame:
    """Synthetic Hidr.cadastro for two plants.

    Both plants use ``tipo_regulacao="M"`` with a simple linear polynomial
    ``h(v) = 300 + 0.1*v`` (volume_cota_0=300, volume_cota_1=0.1, rest
    zero) and ``canal_fuga_medio=50.0``.  With ``tipo_perda=1`` and
    ``perdas=0.0`` the loss model leaves the net drop unchanged.

    USINA_A: [volume_minimo=100, volume_maximo=1000]
    - F(v) = 300*v + 0.05*v^2
    - avg_height = (F(1000)-F(100)) / 900 = (350000-30500)/900 = 355.0
    - net_drop = 355.0 - 50.0 = 305.0
    - productivity_A = 0.9 * 305.0 = 274.5

    USINA_B: [volume_minimo=50, volume_maximo=500]
    - avg_height = (F(500)-F(50)) / 450 = (162500-15125)/450 = 327.5
    - net_drop = 327.5 - 50.0 = 277.5
    - productivity_B = 0.85 * 277.5 = 235.875

    Both productivities differ from their raw ``produtibilidade_especifica``
    values (0.9 and 0.85) because ``canal_fuga_medio`` is nonzero.
    """
    months = [
        "JAN",
        "FEV",
        "MAR",
        "ABR",
        "MAI",
        "JUN",
        "JUL",
        "AGO",
        "SET",
        "OUT",
        "NOV",
        "DEZ",
    ]
    base: dict[str, list] = {
        "nome_usina": ["USINA_A", "USINA_B"],
        "posto": [1, 2],
        "submercado": [1, 1],
        "empresa": [1, 1],
        "codigo_usina_jusante": [pd.NA, 1],
        "desvio": [pd.NA, pd.NA],
        "volume_minimo": [100.0, 50.0],
        "volume_maximo": [1000.0, 500.0],
        "volume_referencia": [550.0, 275.0],
        "canal_fuga_medio": [50.0, 50.0],
        "tipo_regulacao": ["M", "M"],
        "tipo_perda": [1, 1],
        "perdas": [0.0, 0.0],
        "volume_cota_0": [300.0, 300.0],
        "volume_cota_1": [0.1, 0.1],
        "volume_cota_2": [0.0, 0.0],
        "volume_cota_3": [0.0, 0.0],
        "volume_cota_4": [0.0, 0.0],
        "produtibilidade_especifica": [0.9, 0.85],
        "numero_conjuntos_maquinas": [1, 2],
        "maquinas_conjunto_1": [4, 3],
        "maquinas_conjunto_2": [0, 2],
        "maquinas_conjunto_3": [0, 0],
        "maquinas_conjunto_4": [0, 0],
        "maquinas_conjunto_5": [0, 0],
        "potencia_nominal_conjunto_1": [200.0, 150.0],
        "potencia_nominal_conjunto_2": [0.0, 120.0],
        "potencia_nominal_conjunto_3": [0.0, 0.0],
        "potencia_nominal_conjunto_4": [0.0, 0.0],
        "potencia_nominal_conjunto_5": [0.0, 0.0],
        "vazao_nominal_conjunto_1": [222.2, 176.5],
        "vazao_nominal_conjunto_2": [0.0, 141.2],
        "vazao_nominal_conjunto_3": [0.0, 0.0],
        "vazao_nominal_conjunto_4": [0.0, 0.0],
        "vazao_nominal_conjunto_5": [0.0, 0.0],
        "vazao_minima_historica": [0, 0],
        "teif": [0.0, 0.0],
        "ip": [0.0, 0.0],
        "fator_carga_maximo": [1.0, 1.0],
        "fator_carga_minimo": [0.0, 0.0],
    }
    for m in months:
        base[f"evaporacao_{m}"] = [1.5, 2.0]

    df = pd.DataFrame(base, index=pd.Index([1, 2], name="codigo_usina"))
    return df


def _make_ree_df() -> pd.DataFrame:
    return pd.DataFrame({"codigo": [1], "nome": ["SE"], "submercado": [1]})


def _make_conft_df() -> pd.DataFrame:
    """Three thermals: 2 in subsystem 1, 1 in subsystem 2."""
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "submercado": [1, 1, 2],
            "usina_existente": ["EX", "EX", "EX"],
            "classe": [1, 1, 2],
        }
    )


def _make_clast_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "tipo_combustivel": ["GAS", "GAS", "OLEO"],
            "indice_ano_estudo": [1, 1, 1],
            "valor": [50.0, 80.0, 200.0],
        }
    )


def _make_term_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "potencia_instalada": [100.0, 200.0, 50.0],
            "fator_capacidade_maximo": [0.9, 1.0, 0.8],
            "teif": [0.05, 0.02, 0.10],
            "indisponibilidade_programada": [0.0, 0.0, 0.0],
            "mes": [1, 1, 1],
            "geracao_minima": [10.0, 0.0, 5.0],
        }
    )


def _make_deficit_df(n_patamares: int = 2) -> pd.DataFrame:
    """Deficit costs for subsystems 1 and 2 (non-fictitious) plus fictitious 99."""
    rows = []
    for sub, name, fict in [(1, "SE", 0), (2, "S", 0), (99, "FICT", 1)]:
        for pat in range(1, n_patamares + 1):
            rows.append(
                {
                    "codigo_submercado": sub,
                    "nome_submercado": name,
                    "ficticio": fict,
                    "patamar_deficit": pat,
                    "custo": 500.0 * pat,
                    "corte": 1000.0 if pat < n_patamares else None,
                }
            )
    return pd.DataFrame(rows)


def _make_intercambio_df() -> pd.DataFrame:
    """Three interchange pairs for subsystems 1, 2, 99."""
    import datetime

    d = datetime.datetime(2023, 1, 1)
    rows = [
        # 1 -> 2 direct
        {
            "submercado_de": 1,
            "submercado_para": 2,
            "sentido": 1,
            "data": d,
            "valor": 3000.0,
        },
        # 2 -> 1 reverse
        {
            "submercado_de": 2,
            "submercado_para": 1,
            "sentido": 1,
            "data": d,
            "valor": 2500.0,
        },
        # 1 -> 99 direct
        {
            "submercado_de": 1,
            "submercado_para": 99,
            "sentido": 1,
            "data": d,
            "valor": 4000.0,
        },
        # 99 -> 1 reverse
        {
            "submercado_de": 99,
            "submercado_para": 1,
            "sentido": 1,
            "data": d,
            "valor": 2000.0,
        },
        # 2 -> 99 direct
        {
            "submercado_de": 2,
            "submercado_para": 99,
            "sentido": 1,
            "data": d,
            "valor": 1500.0,
        },
        # 99 -> 2 reverse
        {
            "submercado_de": 99,
            "submercado_para": 2,
            "sentido": 1,
            "data": d,
            "valor": 1200.0,
        },
    ]
    return pd.DataFrame(rows)


class TestConvertHydros:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_returns_hydros_key(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        assert "hydros" in result

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydro_count_matches_existing_plants(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        assert len(result["hydros"]) == 2

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydro_ids_are_zero_based_and_sorted(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        ids = [h["id"] for h in result["hydros"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydro_has_required_fields(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        for h in result["hydros"]:
            assert "id" in h
            assert "name" in h
            assert "bus_id" in h
            assert "reservoir" in h
            assert "min_storage_hm3" in h["reservoir"]
            assert "max_storage_hm3" in h["reservoir"]
            assert "outflow" in h
            assert "generation" in h
            assert h["generation"]["model"] == "constant_productivity"

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_cascade_downstream_linkage(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """Plant 2 (code=2) is downstream of plant 1 (code=1)."""
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        # USINA_A (code=1, cobre id=0) has no downstream.
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["downstream_id"] is None
        # USINA_B (code=2, cobre id=1) is downstream of USINA_A (cobre id=0).
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")
        assert hydro_b["downstream_id"] == 0

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_bus_id_matches_ree_subsystem(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        for h in result["hydros"]:
            # Both plants are in REE 1 -> subsystem 1 -> bus 0.
            assert h["bus_id"] == 0

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_generation_values_match_machine_sets(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        gen = hydro_a["generation"]
        # USINA_A: 1 set, 4 machines, 200 MW each, flow 222.2 each.
        assert gen["max_generation_mw"] == pytest.approx(4 * 200.0)
        assert gen["max_turbined_m3s"] == pytest.approx(4 * 222.2)
        # Productivity from new formula (see _make_hidr_cadastro docstring):
        # avg_height = 300 + 0.1 * (100+1000)/2 = 355.0
        # net_drop = 355.0 - 50.0 = 305.0; adjusted (no loss) = 305.0
        # productivity_A = 0.9 * 305.0 = 274.5
        assert gen["productivity_mw_per_m3s"] == pytest.approx(0.9 * 305.0)
        # Verify it differs from raw produtibilidade_especifica (0.9).
        assert gen["productivity_mw_per_m3s"] != pytest.approx(0.9)

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_missing_hidr_raises_file_not_found(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        # Create confhd.dat and ree.dat but NOT hidr.dat.
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "ree.dat").touch()
        from cobre_bridge.converters.hydro import convert_hydros

        with pytest.raises(FileNotFoundError, match="hidr.dat"):
            convert_hydros(tmp_path, self._make_id_map())

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_schema_key_present(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        assert "$schema" in result
        assert "hydros.schema.json" in result["$schema"]

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydro_code_absent_in_hidr_raises_value_error(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        # Set up mocks but make the cadastro empty (no plants).
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        mock_hidr = MagicMock()
        mock_hidr.cadastro = pd.DataFrame()  # empty — no plants
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        with pytest.raises(ValueError, match="not found in hidr.dat"):
            convert_hydros(tmp_path, id_map)

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydraulic_losses_factor(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """tipo_perda=1 and perdas=0.05 -> hydraulic_losses factor dict."""
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 1
        cadastro["perdas"] = 0.05

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] == {
                "type": "factor",
                "value": pytest.approx(0.05),
            }

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydraulic_losses_constant(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """tipo_perda=2 and perdas=3.5 -> hydraulic_losses constant dict."""
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 2
        cadastro["perdas"] = 3.5

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] == {
                "type": "constant",
                "value_m": pytest.approx(3.5),
            }

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydraulic_losses_none_when_zero(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """perdas=0.0 produces hydraulic_losses=None regardless of tipo_perda."""
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 1
        cadastro["perdas"] = 0.0

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] is None

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_teif_ip_derating_reduces_max_generation(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """TEIF=5%, IP=3% reduces max_generation by factor 0.95 * 0.97 = 0.9215."""
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        cadastro = _make_hidr_cadastro().copy()
        # Override only USINA_A (code=1) with non-zero TEIF/IP.
        cadastro.loc[1, "teif"] = 5.0
        cadastro.loc[1, "ip"] = 3.0

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # USINA_A nominal: 4 machines * 200 MW = 800 MW
        # Derating: 800 * 0.95 * 0.97 = 737.2
        expected = 800.0 * 0.95 * 0.97
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(expected)
        # max_turbined_m3s must NOT be derated
        assert hydro_a["generation"]["max_turbined_m3s"] == pytest.approx(4 * 222.2)
        # min_generation_mw must NOT be derated (it is zero here)
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_zero_teif_ip_no_derating(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """TEIF=0% and IP=0% leaves max_generation_mw unchanged."""
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # teif=0, ip=0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_nan_teif_treated_as_zero(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """NaN teif is treated as 0.0 — no derating, no error."""
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        cadastro = _make_hidr_cadastro().copy()
        cadastro.loc[1, "teif"] = float("nan")
        cadastro.loc[1, "ip"] = float("nan")

        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(tmp_path, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # NaN treated as 0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)


def _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path):
    """Create dummy files and wire mock read() returns."""
    for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
        (tmp_path / fname).touch()

    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro()
    mock_hidr_cls.read.return_value = mock_hidr

    mock_confhd = MagicMock()
    mock_confhd.usinas = _make_confhd_df()
    mock_confhd_cls.read.return_value = mock_confhd

    mock_ree = MagicMock()
    mock_ree.rees = _make_ree_df()
    mock_ree_cls.read.return_value = mock_ree


def _make_hreg(overrides: dict) -> pd.Series:
    """Build a minimal plant cadastro row (pd.Series) for unit tests.

    Provides sensible defaults for all columns consumed by
    ``_compute_productivity``.  Pass ``overrides`` to customise
    individual fields.
    """
    defaults: dict = {
        "nome_usina": "TEST",
        "produtibilidade_especifica": 0.009,
        "volume_minimo": 100.0,
        "volume_maximo": 1000.0,
        "volume_referencia": 500.0,
        "canal_fuga_medio": 250.0,
        "tipo_regulacao": "M",
        "tipo_perda": 1,
        "perdas": 0.05,
        "volume_cota_0": 300.0,
        "volume_cota_1": 0.1,
        "volume_cota_2": 0.0,
        "volume_cota_3": 0.0,
        "volume_cota_4": 0.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


# ---------------------------------------------------------------------------
# _compute_productivity unit tests
# ---------------------------------------------------------------------------


class TestComputeProductivity:
    """Unit tests for the ``_compute_productivity`` helper."""

    def test_monthly_regulated_linear_polynomial(self) -> None:
        """tipo_regulacao='M': uses integral average of poly over [vmin, vmax]."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "volume_cota_0": 300.0,
                "volume_cota_1": 0.1,
                "volume_cota_2": 0.0,
                "volume_cota_3": 0.0,
                "volume_cota_4": 0.0,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.05,
                "produtibilidade_especifica": 0.009,
            }
        )
        # Integral average of (300 + 0.1*v) over [100, 1000]:
        #   F(v) = 300*v + 0.1*v^2/2
        #   avg = (F(1000) - F(100)) / (1000 - 100)
        #       = (300*1000 + 0.05*1000^2 - 300*100 - 0.05*100^2) / 900
        #       = (300000 + 50000 - 30000 - 500) / 900
        #       = 319500 / 900 = 355.0
        # net_drop = 355.0 - 250.0 = 105.0
        # adjusted_drop = 105.0 * (1 - 0.05) = 99.75
        # result = 0.009 * 99.75 = 0.89775
        avg_height = 355.0
        expected = 0.009 * (1.0 - 0.05) * (avg_height - 250.0)
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_run_of_river_point_evaluation(self) -> None:
        """tipo_regulacao='D': evaluates poly at volume_referencia."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "volume_cota_0": 300.0,
                "volume_cota_1": 0.1,
                "volume_cota_2": 0.0,
                "volume_cota_3": 0.0,
                "volume_cota_4": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.05,
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 0.1*500 = 350.0
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted_drop = 100.0 * (1 - 0.05) = 95.0
        # result = 0.009 * 95.0 = 0.855
        expected = 0.009 * (1.0 - 0.05) * (350.0 - 250.0)
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_multiplicative_loss(self) -> None:
        """tipo_perda=1: adjusted_drop = net_drop * (1 - perdas)."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "volume_cota_0": 300.0,
                "volume_cota_1": 0.1,
                "volume_cota_2": 0.0,
                "volume_cota_3": 0.0,
                "volume_cota_4": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.10,
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = (300 + 50) - 250 = 100.0
        # adjusted = 100.0 * (1 - 0.10) = 90.0
        expected = 0.009 * 90.0
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_additive_loss(self) -> None:
        """tipo_perda=2: adjusted_drop = net_drop - perdas."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "volume_cota_0": 300.0,
                "volume_cota_1": 0.1,
                "volume_cota_2": 0.0,
                "volume_cota_3": 0.0,
                "volume_cota_4": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 2,
                "perdas": 3.5,
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted = 100.0 - 3.5 = 96.5
        expected = 0.009 * 96.5
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_no_loss(self) -> None:
        """tipo_perda=0 (or unknown): no loss applied, adjusted_drop = net_drop."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "volume_cota_0": 300.0,
                "volume_cota_1": 0.1,
                "volume_cota_2": 0.0,
                "volume_cota_3": 0.0,
                "volume_cota_4": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 99.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # tipo_perda=0 -> no loss applied, perdas value ignored
        # net_drop = 350.0 - 250.0 = 100.0
        expected = 0.009 * 100.0
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_equal_volumes_fallback(self) -> None:
        """tipo_regulacao='M' with volume_minimo==volume_maximo: falls back to point eval."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "volume_minimo": 500.0,
                "volume_maximo": 500.0,
                "volume_cota_0": 300.0,
                "volume_cota_1": 0.1,
                "volume_cota_2": 0.0,
                "volume_cota_3": 0.0,
                "volume_cota_4": 0.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # Falls back to poly(500) = 300 + 0.1*500 = 350.0
        # net_drop = 350.0 - 250.0 = 100.0; no loss
        # result = 0.009 * 100.0
        expected = 0.009 * 100.0
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)


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

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_returns_thermals_key(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(tmp_path, self._make_id_map())
        assert "thermals" in result

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_thermal_count(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(tmp_path, self._make_id_map())
        assert len(result["thermals"]) == 3

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_thermal_ids_are_zero_based_sorted(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(tmp_path, self._make_id_map())
        ids = [t["id"] for t in result["thermals"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_cost_segments_structure(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(tmp_path, self._make_id_map())
        for t in result["thermals"]:
            assert "cost_segments" in t
            assert len(t["cost_segments"]) == 1
            seg = t["cost_segments"][0]
            assert "capacity_mw" in seg
            assert "cost_per_mwh" in seg
            assert "generation" in t
            assert "min_mw" in t["generation"]
            assert "max_mw" in t["generation"]

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_bus_id_assignment(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(tmp_path, self._make_id_map())
        # TERMO_A (code 10) and TERMO_B (code 20) are in submercado 1 -> bus 0.
        # TERMO_C (code 30) is in submercado 2 -> bus 1.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        termo_c = next(t for t in result["thermals"] if t["name"] == "TERMO_C")
        assert termo_a["bus_id"] == 0
        assert termo_c["bus_id"] == 1

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_capacity_uses_factor(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(tmp_path, self._make_id_map())
        # TERMO_A: potencia=100, factor=0.9 -> max_mw=90.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        assert termo_a["generation"]["max_mw"] == pytest.approx(90.0)

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_missing_conft_raises_file_not_found(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        (tmp_path / "clast.dat").touch()
        (tmp_path / "term.dat").touch()
        from cobre_bridge.converters.thermal import convert_thermals

        with pytest.raises(FileNotFoundError, match="conft.dat"):
            convert_thermals(tmp_path, self._make_id_map())


def _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path):
    for fname in ("conft.dat", "clast.dat", "term.dat"):
        (tmp_path / fname).touch()

    mock_conft = MagicMock()
    mock_conft.usinas = _make_conft_df()
    mock_conft_cls.read.return_value = mock_conft

    mock_clast = MagicMock()
    mock_clast.usinas = _make_clast_df()
    mock_clast_cls.read.return_value = mock_clast

    mock_term = MagicMock()
    mock_term.usinas = _make_term_df()
    mock_term_cls.read.return_value = mock_term


# ---------------------------------------------------------------------------
# Bus conversion
# ---------------------------------------------------------------------------


class TestConvertBuses:
    def _make_id_map(self) -> NewaveIdMap:
        # Subsystems: 1, 2, 99 (fictitious)
        return NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )

    @patch("cobre_bridge.converters.network.Sistema")
    def test_returns_buses_key(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(tmp_path, self._make_id_map())
        assert "buses" in result

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_count_includes_fictitious(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(tmp_path, self._make_id_map())
        # 3 subsystems total: 1, 2, 99.
        assert len(result["buses"]) == 3

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_ids_are_zero_based_sorted(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(tmp_path, self._make_id_map())
        ids = [b["id"] for b in result["buses"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_has_deficit_segments(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(tmp_path, self._make_id_map())
        for b in result["buses"]:
            assert "deficit_segments" in b
            assert isinstance(b["deficit_segments"], list)
            assert len(b["deficit_segments"]) == 2  # 2 patamares

    @patch("cobre_bridge.converters.network.Sistema")
    def test_last_deficit_segment_depth_is_null(
        self, mock_sistema_cls, tmp_path
    ) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(tmp_path, self._make_id_map())
        for b in result["buses"]:
            last_seg = b["deficit_segments"][-1]
            assert last_seg["depth_mw"] is None

    @patch("cobre_bridge.converters.network.Sistema")
    def test_missing_sistema_raises_file_not_found(
        self, mock_sistema_cls, tmp_path
    ) -> None:
        from cobre_bridge.converters.network import convert_buses

        with pytest.raises(FileNotFoundError, match="sistema.dat"):
            convert_buses(tmp_path, self._make_id_map())


# ---------------------------------------------------------------------------
# Line conversion
# ---------------------------------------------------------------------------


class TestConvertLines:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )

    @patch("cobre_bridge.converters.network.Sistema")
    def test_returns_lines_key(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(tmp_path, self._make_id_map())
        assert "lines" in result

    @patch("cobre_bridge.converters.network.Sistema")
    def test_line_count_three_pairs(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(tmp_path, self._make_id_map())
        # 3 unique pairs: (1,2), (1,99), (2,99).
        assert len(result["lines"]) == 3

    @patch("cobre_bridge.converters.network.Sistema")
    def test_line_capacity_structure(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(tmp_path, self._make_id_map())
        for line in result["lines"]:
            assert "capacity" in line
            assert "direct_mw" in line["capacity"]
            assert "reverse_mw" in line["capacity"]
            assert "source_bus_id" in line
            assert "target_bus_id" in line

    @patch("cobre_bridge.converters.network.Sistema")
    def test_line_ids_sequential(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(tmp_path, self._make_id_map())
        ids = [ln["id"] for ln in result["lines"]]
        assert ids == list(range(len(ids)))

    @patch("cobre_bridge.converters.network.Sistema")
    def test_first_month_used_for_capacity(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(tmp_path, self._make_id_map())
        # Find the (1, 2) pair — subsystems 1<2, bus_ids 0 and 1.
        line_12 = next(
            ln
            for ln in result["lines"]
            if ln["source_bus_id"] == 0 and ln["target_bus_id"] == 1
        )
        assert line_12["capacity"]["direct_mw"] == pytest.approx(3000.0)
        assert line_12["capacity"]["reverse_mw"] == pytest.approx(2500.0)


def _setup_sistema_mocks(mock_sistema_cls, tmp_path):
    (tmp_path / "sistema.dat").touch()
    mock_sistema = MagicMock()
    mock_sistema.custo_deficit = _make_deficit_df(n_patamares=2)
    mock_sistema.limites_intercambio = _make_intercambio_df()
    mock_sistema_cls.read.return_value = mock_sistema


# ---------------------------------------------------------------------------
# Penalties conversion
# ---------------------------------------------------------------------------


class TestConvertPenalties:
    @patch("cobre_bridge.converters.network.Sistema")
    def test_returns_required_keys(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_penalties

        result = convert_penalties(tmp_path)
        for key in ("bus", "hydro", "line", "non_controllable_source"):
            assert key in result

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_deficit_uses_first_subsystem_first_tier(
        self, mock_sistema_cls, tmp_path
    ) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_penalties

        result = convert_penalties(tmp_path)
        # First subsystem=1, patamar=1: custo = 500.0*1 = 500.0
        seg = result["bus"]["deficit_segments"][0]
        assert seg["cost"] == pytest.approx(500.0)

    @patch("cobre_bridge.converters.network.Sistema")
    def test_hydro_has_all_penalty_fields(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_penalties

        result = convert_penalties(tmp_path)
        required = {
            "spillage_cost",
            "fpha_turbined_cost",
            "diversion_cost",
            "storage_violation_below_cost",
            "filling_target_violation_cost",
            "turbined_violation_below_cost",
            "outflow_violation_below_cost",
            "outflow_violation_above_cost",
            "generation_violation_below_cost",
            "evaporation_violation_cost",
            "water_withdrawal_violation_cost",
        }
        assert required == set(result["hydro"].keys())


# ---------------------------------------------------------------------------
# Initial conditions conversion
# ---------------------------------------------------------------------------


class TestConvertInitialConditions:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_returns_storage_and_filling_storage(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(tmp_path, self._make_id_map())
        assert "storage" in result
        assert "filling_storage" in result

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_storage_values_converted_from_percentage(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(tmp_path, self._make_id_map())
        # USINA_A: pct=50%, vol_max=1000 -> 500 hm3.
        # USINA_B: pct=75%, vol_max=500  -> 375 hm3.
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        assert storage[0] == pytest.approx(500.0)
        assert storage[1] == pytest.approx(375.0)

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_storage_sorted_by_hydro_id(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(tmp_path, self._make_id_map())
        ids = [s["hydro_id"] for s in result["storage"]]
        assert ids == sorted(ids)

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_out_of_range_percentage_clamped(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path, pct_b=120.0)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # Should not raise; pct is clamped to 100.
        result = convert_initial_conditions(tmp_path, self._make_id_map())
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # pct clamped to 100 -> vol_max=500 -> 500.0 hm3.
        assert storage[1] == pytest.approx(500.0)

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_missing_hidr_raises_file_not_found(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        (tmp_path / "confhd.dat").touch()
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        with pytest.raises(FileNotFoundError, match="hidr.dat"):
            convert_initial_conditions(tmp_path, self._make_id_map())

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_filling_storage_is_empty(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(tmp_path, self._make_id_map())
        assert result["filling_storage"] == []


def _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path, pct_b: float = 75.0):
    for fname in ("hidr.dat", "confhd.dat"):
        (tmp_path / fname).touch()

    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro()
    mock_hidr_cls.read.return_value = mock_hidr

    df = _make_confhd_df().copy()
    df.loc[df["codigo_usina"] == 2, "volume_inicial_percentual"] = pct_b
    mock_confhd = MagicMock()
    mock_confhd.usinas = df
    mock_confhd_cls.read.return_value = mock_confhd


# ---------------------------------------------------------------------------
# Cross-reference consistency
# ---------------------------------------------------------------------------


class TestCrossReferenceConsistency:
    """Verify bus_id values in hydros and thermals match the buses output."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1, 2],
            hydro_codes=[1, 2],
            thermal_codes=[10, 20, 30],
        )

    @patch("cobre_bridge.converters.network.Sistema")
    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_all_bus_ids_are_valid(
        self,
        mock_hidr_cls,
        mock_confhd_cls,
        mock_ree_cls,
        mock_conft_cls,
        mock_clast_cls,
        mock_term_cls,
        mock_sistema_cls,
        tmp_path,
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)

        from cobre_bridge.converters.hydro import convert_hydros
        from cobre_bridge.converters.network import convert_buses
        from cobre_bridge.converters.thermal import convert_thermals

        # Use a shared id_map that covers both subsystems and all plants.
        id_map = NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[1, 2],
            thermal_codes=[10, 20, 30],
        )

        buses_result = convert_buses(tmp_path, id_map)
        hydros_result = convert_hydros(tmp_path, id_map)
        thermals_result = convert_thermals(tmp_path, id_map)

        valid_bus_ids = {b["id"] for b in buses_result["buses"]}

        for h in hydros_result["hydros"]:
            assert h["bus_id"] in valid_bus_ids, (
                f"Hydro '{h['name']}' has bus_id={h['bus_id']} not in buses"
            )

        for t in thermals_result["thermals"]:
            assert t["bus_id"] in valid_bus_ids, (
                f"Thermal '{t['name']}' has bus_id={t['bus_id']} not in buses"
            )

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_downstream_ids_are_valid(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(tmp_path, id_map)
        valid_hydro_ids = {h["id"] for h in result["hydros"]}

        for h in result["hydros"]:
            ds = h.get("downstream_id")
            if ds is not None:
                assert ds in valid_hydro_ids, (
                    f"Hydro '{h['name']}' has downstream_id={ds} not in hydros"
                )
