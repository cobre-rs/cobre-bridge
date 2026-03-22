"""Unit tests for NEWAVE entity conversion functions.

All inewave I/O is mocked via ``unittest.mock.patch`` so no real NEWAVE
files are required.  The tests use small synthetic DataFrames that cover
the logic exercised by each converter.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles


def _make_nw_files(
    tmp_path: Path,
    *,
    modif: Path | None = None,
    ghmin: Path | None = None,
    penalid: Path | None = None,
    vazpast: Path | None = None,
    dsvagua: Path | None = None,
    curva: Path | None = None,
    expt: Path | None = None,
    manutt: Path | None = None,
    c_adic: Path | None = None,
    cvar: Path | None = None,
    agrint: Path | None = None,
) -> NewaveFiles:
    """Construct a ``NewaveFiles`` with sentinel paths under *tmp_path*.

    All required file paths point to ``tmp_path / "<name>.dat"`` regardless
    of whether the files actually exist on disk — the inewave I/O is mocked
    in the tests that use this helper.  Optional paths are passed through as-is
    (default ``None``).
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
        modif=modif,
        ghmin=ghmin,
        penalid=penalid,
        vazpast=vazpast,
        dsvagua=dsvagua,
        curva=curva,
        expt=expt,
        manutt=manutt,
        c_adic=c_adic,
        cvar=cvar,
        agrint=agrint,
    )


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
    ``h(v) = 300 + 0.1*v`` (a0_volume_cota=300, a1_volume_cota=0.1, rest
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
        "a0_volume_cota": [300.0, 300.0],
        "a1_volume_cota": [0.1, 0.1],
        "a2_volume_cota": [0.0, 0.0],
        "a3_volume_cota": [0.0, 0.0],
        "a4_volume_cota": [0.0, 0.0],
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
            "fator_capacidade_maximo": [90.0, 100.0, 80.0],
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
        assert "hydros" in result

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydro_count_matches_existing_plants(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
        assert len(result["hydros"]) == 2

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_hydro_ids_are_zero_based_and_sorted(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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
    def test_schema_key_present(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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
            convert_hydros(_make_nw_files(tmp_path), id_map)

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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # NaN treated as 0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)


def _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path):
    """Wire mock read() returns for the three hydro-required files."""
    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro()
    mock_hidr_cls.read.return_value = mock_hidr

    mock_confhd = MagicMock()
    mock_confhd.usinas = _make_confhd_df()
    mock_confhd_cls.read.return_value = mock_confhd

    mock_ree = MagicMock()
    mock_ree.rees = _make_ree_df()
    mock_ree_cls.read.return_value = mock_ree


# ---------------------------------------------------------------------------
# _apply_permanent_overrides unit tests  (ticket-004)
# ---------------------------------------------------------------------------


class TestApplyPermanentOverrides:
    """Unit tests for ``_apply_permanent_overrides``."""

    def _base_cadastro(self) -> pd.DataFrame:
        return _make_hidr_cadastro()

    def test_missing_modif_returns_unchanged(self, tmp_path) -> None:
        """No MODIF.DAT -> cadastro returned unchanged."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        cadastro = self._base_cadastro()
        result = _apply_permanent_overrides(
            cadastro, _make_nw_files(tmp_path, modif=None)
        )
        pd.testing.assert_frame_equal(result, cadastro)

    def test_volmax_override(self, tmp_path) -> None:
        """VOLMAX record updates volume_maximo for the target plant."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        (tmp_path / "modif.dat").touch()

        # Build MODIF mock: plant 1 gets VOLMAX=2000.
        volmax_rec = MagicMock()
        volmax_rec.__class__.__name__ = "VOLMAX"
        type(volmax_rec).__name__ = "VOLMAX"
        volmax_rec.volume = 2000.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volmax_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _apply_permanent_overrides(
                self._base_cadastro(),
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            )

        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(2000.0)
        # Plant 2 must be unchanged.
        assert float(result.loc[2, "volume_maximo"]) == pytest.approx(500.0)

    def test_vazmin_override(self, tmp_path) -> None:
        """VAZMIN record updates vazao_minima_historica for the target plant."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        (tmp_path / "modif.dat").touch()

        vazmin_rec = MagicMock()
        type(vazmin_rec).__name__ = "VAZMIN"
        vazmin_rec.vazao = 75.5

        usina_rec = MagicMock()
        usina_rec.codigo = 2

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmin_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _apply_permanent_overrides(
                self._base_cadastro(),
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            )

        assert float(result.loc[2, "vazao_minima_historica"]) == pytest.approx(75.5)
        # Plant 1 must be unchanged (was 0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)

    def test_numcnj_nummaq_override(self, tmp_path) -> None:
        """NUMCNJ + NUMMAQ records update machine set counts."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        (tmp_path / "modif.dat").touch()

        numcnj_rec = MagicMock()
        type(numcnj_rec).__name__ = "NUMCNJ"
        numcnj_rec.numero = 2

        nummaq_rec = MagicMock()
        type(nummaq_rec).__name__ = "NUMMAQ"
        nummaq_rec.conjunto = 2
        nummaq_rec.numero_maquinas = 3

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [numcnj_rec, nummaq_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _apply_permanent_overrides(
                self._base_cadastro(),
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            )

        assert int(result.loc[1, "numero_conjuntos_maquinas"]) == 2
        assert int(result.loc[1, "maquinas_conjunto_2"]) == 3

    def test_volcota_override_warns_and_skips(self, tmp_path, caplog) -> None:
        """VOLCOTA records produce a warning and are skipped gracefully."""
        import logging

        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        (tmp_path / "modif.dat").touch()

        volcota_rec = MagicMock()
        type(volcota_rec).__name__ = "VOLCOTA"

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volcota_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            with caplog.at_level(
                logging.WARNING, logger="cobre_bridge.converters.hydro"
            ):
                result = _apply_permanent_overrides(
                    self._base_cadastro(),
                    _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
                )

        # Values must be unchanged (dtype may differ due to float cast for safety).
        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert any("VOLCOTA" in msg for msg in caplog.messages)

    def test_unknown_plant_code_skipped(self, tmp_path, caplog) -> None:
        """Plant code not in cadastro: warning logged, no crash."""
        import logging

        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        (tmp_path / "modif.dat").touch()

        usina_rec = MagicMock()
        usina_rec.codigo = 999  # not in cadastro

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = []

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            with caplog.at_level(
                logging.WARNING, logger="cobre_bridge.converters.hydro"
            ):
                result = _apply_permanent_overrides(
                    self._base_cadastro(),
                    _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
                )

        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert any("999" in msg for msg in caplog.messages)

    def test_temporal_records_skipped_in_permanent_pass(self, tmp_path) -> None:
        """Temporal override types are ignored in _apply_permanent_overrides."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        (tmp_path / "modif.dat").touch()

        import datetime

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 999.0  # large value that should NOT be applied

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _apply_permanent_overrides(
                self._base_cadastro(),
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            )

        # vazao_minima_historica must stay at the base value (0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _extract_temporal_overrides unit tests  (ticket-005)
# ---------------------------------------------------------------------------


class TestExtractTemporalOverrides:
    """Unit tests for ``_extract_temporal_overrides``."""

    def test_missing_modif_returns_empty(self, tmp_path) -> None:
        """No MODIF.DAT -> empty dict returned, no error."""
        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        result = _extract_temporal_overrides(
            _make_nw_files(tmp_path, modif=None), [1, 2]
        )
        assert result == {}

    def test_extracts_vazmint_records(self, tmp_path) -> None:
        """VAZMINT record is extracted with correct month, year, value."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        (tmp_path / "modif.dat").touch()

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 50.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _extract_temporal_overrides(
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"), [1, 2]
            )

        assert 1 in result
        assert result[1] == [
            {"type": "VAZMINT", "month": 1, "year": 2025, "value": 50.0}
        ]

    def test_filters_by_confhd_codes(self, tmp_path) -> None:
        """Plants not in confhd_codes are excluded from the result."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        (tmp_path / "modif.dat").touch()

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 3, 1)
        vazmint_rec.vazao = 40.0

        # Plant 99 is NOT in confhd_codes [1, 2].
        usina_rec = MagicMock()
        usina_rec.codigo = 99

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _extract_temporal_overrides(
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"), [1, 2]
            )

        assert result == {}

    def test_preserves_file_order(self, tmp_path) -> None:
        """Multiple records for the same plant are returned in file order."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        (tmp_path / "modif.dat").touch()

        def _vazmint(month: int, vazao: float) -> MagicMock:
            r = MagicMock()
            type(r).__name__ = "VAZMINT"
            r.data_inicio = datetime.datetime(2025, month, 1)
            r.vazao = vazao
            return r

        recs = [_vazmint(1, 50.0), _vazmint(6, 60.0), _vazmint(3, 55.0)]

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = recs

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _extract_temporal_overrides(
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"), [1]
            )

        assert len(result[1]) == 3
        assert result[1][0]["value"] == pytest.approx(50.0)
        assert result[1][1]["value"] == pytest.approx(60.0)
        assert result[1][2]["value"] == pytest.approx(55.0)

    def test_extracts_cfuga_records(self, tmp_path) -> None:
        """CFUGA record extracted with correct level value."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        (tmp_path / "modif.dat").touch()

        cfuga_rec = MagicMock()
        type(cfuga_rec).__name__ = "CFUGA"
        cfuga_rec.data_inicio = datetime.datetime(2025, 6, 1)
        cfuga_rec.nivel = 75.4

        usina_rec = MagicMock()
        usina_rec.codigo = 2

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _extract_temporal_overrides(
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"), [2]
            )

        assert result[2] == [
            {"type": "CFUGA", "month": 6, "year": 2025, "value": pytest.approx(75.4)}
        ]

    def test_extracts_turbmint_turbmaxt_records(self, tmp_path) -> None:
        """TURBMINT and TURBMAXT records use turbinamento field."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        (tmp_path / "modif.dat").touch()

        turbmint_rec = MagicMock()
        type(turbmint_rec).__name__ = "TURBMINT"
        turbmint_rec.data_inicio = datetime.datetime(2025, 11, 1)
        turbmint_rec.turbinamento = 330.0

        turbmaxt_rec = MagicMock()
        type(turbmaxt_rec).__name__ = "TURBMAXT"
        turbmaxt_rec.data_inicio = datetime.datetime(2025, 3, 1)
        turbmaxt_rec.turbinamento = 322.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [turbmint_rec, turbmaxt_rec]

        with patch("cobre_bridge.converters.hydro.Modif") as mock_modif_cls:
            mock_modif_cls.read.return_value = mock_modif
            result = _extract_temporal_overrides(
                _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"), [1]
            )

        assert result[1][0] == {
            "type": "TURBMINT",
            "month": 11,
            "year": 2025,
            "value": pytest.approx(330.0),
        }
        assert result[1][1] == {
            "type": "TURBMAXT",
            "month": 3,
            "year": 2025,
            "value": pytest.approx(322.0),
        }


# ---------------------------------------------------------------------------
# _read_ghmin unit tests  (ticket-006)
# ---------------------------------------------------------------------------


class TestReadGhmin:
    """Unit tests for ``_read_ghmin``."""

    def test_missing_ghmin_returns_empty(self, tmp_path) -> None:
        """No GHMIN.DAT -> empty dict, no error."""
        from cobre_bridge.converters.hydro import _read_ghmin

        result = _read_ghmin(_make_nw_files(tmp_path, ghmin=None))
        assert result == {}

    def test_reads_plant_min_generation(self, tmp_path) -> None:
        """Single plant entry returned with correct MW value."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin

        (tmp_path / "ghmin.dat").touch()

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1],
                "data": [datetime.datetime(2025, 1, 1)],
                "patamar": [0],
                "geracao": [50.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        with patch("cobre_bridge.converters.hydro.Ghmin") as mock_ghmin_cls:
            mock_ghmin_cls.read.return_value = mock_ghmin
            result = _read_ghmin(_make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"))

        assert result == {1: pytest.approx(50.0)}

    def test_multiple_plants(self, tmp_path) -> None:
        """Multiple plants returned with correct MW values."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin

        (tmp_path / "ghmin.dat").touch()

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2],
                "data": [
                    datetime.datetime(2025, 1, 1),
                    datetime.datetime(2025, 1, 1),
                ],
                "patamar": [0, 0],
                "geracao": [50.0, 120.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        with patch("cobre_bridge.converters.hydro.Ghmin") as mock_ghmin_cls:
            mock_ghmin_cls.read.return_value = mock_ghmin
            result = _read_ghmin(_make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"))

        assert result[1] == pytest.approx(50.0)
        assert result[2] == pytest.approx(120.0)

    def test_multiple_periods_uses_earliest(self, tmp_path) -> None:
        """When multiple time periods exist, earliest date is used."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin

        (tmp_path / "ghmin.dat").touch()

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2025, 3, 1),
                    datetime.datetime(2025, 1, 1),
                ],
                "patamar": [0, 0],
                "geracao": [999.0, 50.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        with patch("cobre_bridge.converters.hydro.Ghmin") as mock_ghmin_cls:
            mock_ghmin_cls.read.return_value = mock_ghmin
            result = _read_ghmin(_make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"))

        # Earliest date (Jan) has geracao=50.0.
        assert result[1] == pytest.approx(50.0)

    def test_patamar_nonzero_excluded(self, tmp_path) -> None:
        """Rows with patamar != 0 are excluded."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin

        (tmp_path / "ghmin.dat").touch()

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2025, 1, 1),
                    datetime.datetime(2025, 1, 1),
                ],
                "patamar": [1, 2],  # no patamar=0 rows
                "geracao": [50.0, 60.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        with patch("cobre_bridge.converters.hydro.Ghmin") as mock_ghmin_cls:
            mock_ghmin_cls.read.return_value = mock_ghmin
            result = _read_ghmin(_make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"))

        assert result == {}


# ---------------------------------------------------------------------------
# convert_hydros integration tests for ticket-006
# ---------------------------------------------------------------------------


class TestConvertHydrosGhmin:
    """Integration tests verifying GHMIN.DAT override in convert_hydros."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_ghmin_overrides_approximation(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """When GHMIN has an entry for a plant, min_generation_mw uses that value."""
        import datetime

        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1],
                "data": [datetime.datetime(2025, 1, 1)],
                "patamar": [0],
                "geracao": [99.9],
            }
        )
        mock_ghmin_obj = MagicMock()
        mock_ghmin_obj.geracoes = ghmin_df

        from cobre_bridge.converters.hydro import convert_hydros

        with patch("cobre_bridge.converters.hydro.Ghmin") as mock_ghmin_cls:
            mock_ghmin_cls.read.return_value = mock_ghmin_obj
            result = convert_hydros(
                _make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"),
                self._make_id_map(),
            )

        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(99.9)

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_no_ghmin_uses_fallback(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """When no GHMIN.DAT, min_generation_mw uses the approximation."""
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(_make_nw_files(tmp_path), self._make_id_map())
        # USINA_A: vazao_minima_historica=0 -> min_outflow=0 -> min_generation=0
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)


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
        "a0_volume_cota": 300.0,
        "a1_volume_cota": 0.1,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
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
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
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
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
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
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
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
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
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
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
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
        """tipo_regulacao='M' with equal volumes: falls back to point evaluation."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "volume_minimo": 500.0,
                "volume_maximo": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
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
# _compute_productivity with override parameters
# ---------------------------------------------------------------------------


class TestComputeProductivityOverrides:
    """Unit tests for ``_compute_productivity`` with canal_fuga/cmont overrides."""

    def test_canal_fuga_override_replaces_base(self) -> None:
        """canal_fuga_override replaces canal_fuga_medio in the net drop calc."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 50 = 350
        # With override canal_fuga=260: net_drop = 350 - 260 = 90
        base = _compute_productivity(hreg)  # uses 250 -> drop 100
        overridden = _compute_productivity(hreg, canal_fuga_override=260.0)
        assert base == pytest.approx(0.009 * 100.0)
        assert overridden == pytest.approx(0.009 * 90.0)

    def test_cmont_override_replaces_polynomial_height(self) -> None:
        """cmont_override bypasses the polynomial and uses the supplied height."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # cmont=380 overrides polynomial; net_drop = 380 - 250 = 130
        result = _compute_productivity(hreg, cmont_override=380.0)
        assert result == pytest.approx(0.009 * 130.0)

    def test_both_overrides_together(self) -> None:
        """canal_fuga_override and cmont_override can both be active."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "D",
                "volume_referencia": 500.0,
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 0,
                "perdas": 0.0,
                "produtibilidade_especifica": 0.009,
            }
        )
        # cmont=400, canal_fuga=260 -> net_drop = 400 - 260 = 140
        result = _compute_productivity(
            hreg, canal_fuga_override=260.0, cmont_override=400.0
        )
        assert result == pytest.approx(0.009 * 140.0)

    def test_no_overrides_matches_original_behaviour(self) -> None:
        """With no overrides, refactored function gives same result as before."""
        from cobre_bridge.converters.hydro import _compute_productivity

        hreg = _make_hreg(
            {
                "tipo_regulacao": "M",
                "a0_volume_cota": 300.0,
                "a1_volume_cota": 0.1,
                "volume_minimo": 100.0,
                "volume_maximo": 1000.0,
                "canal_fuga_medio": 250.0,
                "tipo_perda": 1,
                "perdas": 0.05,
                "produtibilidade_especifica": 0.009,
            }
        )
        # avg_height = 355.0 (see TestComputeProductivity)
        expected = 0.009 * (1.0 - 0.05) * (355.0 - 250.0)
        assert _compute_productivity(hreg) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# convert_production_models unit tests
# ---------------------------------------------------------------------------


def _make_prod_model_dger_mock(
    *,
    ano_inicio: int = 2025,
    mes_inicio: int = 1,
    num_anos: int = 5,
    num_anos_pos: int = 0,
) -> MagicMock:
    """Return a mock Dger object for use in production model tests."""
    m = MagicMock()
    m.ano_inicio_estudo = ano_inicio
    m.mes_inicio_estudo = mes_inicio
    m.num_anos_estudo = num_anos
    m.num_anos_pos_estudo = num_anos_pos
    return m


def _make_cfuga_rec(month: int, year: int, nivel: float) -> MagicMock:
    import datetime

    r = MagicMock()
    type(r).__name__ = "CFUGA"
    r.data_inicio = datetime.datetime(year, month, 1)
    r.nivel = nivel
    return r


def _make_cmont_rec(month: int, year: int, nivel: float) -> MagicMock:
    import datetime

    r = MagicMock()
    type(r).__name__ = "CMONT"
    r.data_inicio = datetime.datetime(year, month, 1)
    r.nivel = nivel
    return r


class TestConvertProductionModels:
    """Unit tests for ``convert_production_models``."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def _setup_base_mocks(
        self,
        mock_hidr_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
        *,
        ano_inicio: int = 2025,
        mes_inicio: int = 1,
        num_anos: int = 5,
    ) -> None:
        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_dger_cls.read.return_value = _make_prod_model_dger_mock(
            ano_inicio=ano_inicio,
            mes_inicio=mes_inicio,
            num_anos=num_anos,
            num_anos_pos=0,
        )

    @patch("cobre_bridge.converters.hydro.Dger")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_returns_none_when_no_modif(
        self,
        mock_hidr_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """No MODIF.DAT -> None returned, no error."""
        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_dger_cls.read.return_value = _make_prod_model_dger_mock()

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(
            _make_nw_files(tmp_path, modif=None), self._make_id_map()
        )
        assert result is None

    @patch("cobre_bridge.converters.hydro.Dger")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Modif")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_returns_none_when_no_cfuga_cmont(
        self,
        mock_hidr_cls: MagicMock,
        mock_modif_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """MODIF.DAT present but only VAZMINT overrides -> None returned."""
        import datetime

        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()
        mock_hidr_cls.read.return_value = mock_hidr

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_dger_cls.read.return_value = _make_prod_model_dger_mock()

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 3, 1)
        vazmint_rec.vazao = 50.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]
        mock_modif_cls.read.return_value = mock_modif

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(
            _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            self._make_id_map(),
        )
        assert result is None

    @patch("cobre_bridge.converters.hydro.Dger")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Modif")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_single_cfuga_override_two_ranges(
        self,
        mock_hidr_cls: MagicMock,
        mock_modif_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """One CFUGA override at stage 3 -> two stage_ranges (base then overridden)."""
        # Study: start Jan 2025, 5 years -> 60 stages total.
        self._setup_base_mocks(
            mock_hidr_cls,
            mock_confhd_cls,
            mock_dger_cls,
            tmp_path,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=5,
        )

        cfuga_rec = _make_cfuga_rec(month=4, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]
        mock_modif_cls.read.return_value = mock_modif

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(
            _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            self._make_id_map(),
        )

        assert result is not None
        assert "production_models" in result
        models = result["production_models"]
        assert len(models) == 1

        model = models[0]
        assert model["hydro_id"] == 0  # code=1 -> sorted position 0
        assert model["selection_mode"] == "stage_ranges"
        ranges = model["stage_ranges"]

        # CFUGA at April 2025 = stage (2025-2025)*12 + (4-1) = 3.
        # So: [0..2] base, [3..None] overridden.
        assert len(ranges) == 2
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] == 2
        assert ranges[0]["model"] == "constant_productivity"
        assert ranges[1]["start_stage_id"] == 3
        assert ranges[1]["end_stage_id"] is None
        assert ranges[1]["model"] == "constant_productivity"

        # The overridden productivity uses canal_fuga=60 instead of 50.
        # USINA_A: tipo_regulacao='M', avg_height=355, canal_fuga_base=50 -> drop=305
        # With cfuga=60: drop = 355 - 60 = 295 -> prod = 0.9 * 295 = 265.5
        base_prod = 0.9 * (355.0 - 50.0)
        overridden_prod = 0.9 * (355.0 - 60.0)
        assert ranges[0]["productivity_override"] == pytest.approx(base_prod)
        assert ranges[1]["productivity_override"] == pytest.approx(overridden_prod)

    @patch("cobre_bridge.converters.hydro.Dger")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Modif")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_cmont_override_bypasses_polynomial(
        self,
        mock_hidr_cls: MagicMock,
        mock_modif_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """CMONT override at stage 0 -> single stage_range using cmont as height."""
        self._setup_base_mocks(
            mock_hidr_cls,
            mock_confhd_cls,
            mock_dger_cls,
            tmp_path,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=5,
        )

        cmont_rec = _make_cmont_rec(month=1, year=2025, nivel=400.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cmont_rec]
        mock_modif_cls.read.return_value = mock_modif

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(
            _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            self._make_id_map(),
        )

        assert result is not None
        models = result["production_models"]
        assert len(models) == 1

        ranges = models[0]["stage_ranges"]
        # CMONT at Jan 2025 = stage 0, so no base range before it.
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        # net_drop = 400 - 50 (canal_fuga_medio) = 350 -> prod = 0.9 * 350 = 315.0
        assert ranges[0]["productivity_override"] == pytest.approx(0.9 * 350.0)

    @patch("cobre_bridge.converters.hydro.Dger")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Modif")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_multiple_overrides_three_ranges(
        self,
        mock_hidr_cls: MagicMock,
        mock_modif_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Two CFUGA overrides -> three stage_ranges."""
        self._setup_base_mocks(
            mock_hidr_cls,
            mock_confhd_cls,
            mock_dger_cls,
            tmp_path,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=5,
        )

        recs = [
            _make_cfuga_rec(month=6, year=2025, nivel=55.0),  # stage 5
            _make_cfuga_rec(month=1, year=2026, nivel=65.0),  # stage 12
        ]
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = recs
        mock_modif_cls.read.return_value = mock_modif

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(
            _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            self._make_id_map(),
        )

        assert result is not None
        ranges = result["production_models"][0]["stage_ranges"]
        # stage 5: June 2025 -> (2025-2025)*12 + (6-1) = 5
        # stage 12: Jan 2026 -> (2026-2025)*12 + (1-1) = 12
        assert len(ranges) == 3
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] == 4
        assert ranges[1]["start_stage_id"] == 5
        assert ranges[1]["end_stage_id"] == 11
        assert ranges[2]["start_stage_id"] == 12
        assert ranges[2]["end_stage_id"] is None

    @patch("cobre_bridge.converters.hydro.Dger")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Modif")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_output_sorted_by_hydro_id(
        self,
        mock_hidr_cls: MagicMock,
        mock_modif_cls: MagicMock,
        mock_confhd_cls: MagicMock,
        mock_dger_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        """production_models list is sorted ascending by hydro_id."""
        self._setup_base_mocks(
            mock_hidr_cls,
            mock_confhd_cls,
            mock_dger_cls,
            tmp_path,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=5,
        )

        # Both plants have CFUGA overrides; plant codes 1 and 2 -> ids 0 and 1.
        usina_rec1 = MagicMock()
        usina_rec1.codigo = 1
        usina_rec2 = MagicMock()
        usina_rec2.codigo = 2

        def _mods(code: int) -> list:
            if code == 1:
                return [_make_cfuga_rec(month=3, year=2025, nivel=55.0)]
            return [_make_cfuga_rec(month=6, year=2025, nivel=55.0)]

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec2, usina_rec1]  # reversed order
        mock_modif.modificacoes_usina.side_effect = _mods
        mock_modif_cls.read.return_value = mock_modif

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(
            _make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            self._make_id_map(),
        )

        assert result is not None
        ids = [m["hydro_id"] for m in result["production_models"]]
        assert ids == sorted(ids)


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

        result = convert_thermals(_make_nw_files(tmp_path), self._make_id_map())
        assert "thermals" in result

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_thermal_count(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(_make_nw_files(tmp_path), self._make_id_map())
        assert len(result["thermals"]) == 3

    @patch("cobre_bridge.converters.thermal.Term")
    @patch("cobre_bridge.converters.thermal.Clast")
    @patch("cobre_bridge.converters.thermal.Conft")
    def test_thermal_ids_are_zero_based_sorted(
        self, mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path
    ) -> None:
        _setup_thermal_mocks(mock_conft_cls, mock_clast_cls, mock_term_cls, tmp_path)
        from cobre_bridge.converters.thermal import convert_thermals

        result = convert_thermals(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_thermals(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_thermals(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_thermals(_make_nw_files(tmp_path), self._make_id_map())
        # TERMO_A: potencia=100, factor=0.9 -> max_mw=90.
        termo_a = next(t for t in result["thermals"] if t["name"] == "TERMO_A")
        assert termo_a["generation"]["max_mw"] == pytest.approx(90.0)


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

        result = convert_buses(_make_nw_files(tmp_path), self._make_id_map())
        assert "buses" in result

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_count_includes_fictitious(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(_make_nw_files(tmp_path), self._make_id_map())
        # 3 subsystems total: 1, 2, 99.
        assert len(result["buses"]) == 3

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_ids_are_zero_based_sorted(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(_make_nw_files(tmp_path), self._make_id_map())
        ids = [b["id"] for b in result["buses"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_has_deficit_segments(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_buses

        result = convert_buses(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_buses(_make_nw_files(tmp_path), self._make_id_map())
        for b in result["buses"]:
            last_seg = b["deficit_segments"][-1]
            assert last_seg["depth_mw"] is None


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

    def _setup(self, mock_sistema_cls, mock_dger_cls, tmp_path):
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        mock_dger_cls.read.return_value = dger

    @patch("cobre_bridge.converters.network.Dger")
    @patch("cobre_bridge.converters.network.Sistema")
    def test_returns_lines_key(self, mock_sistema_cls, mock_dger_cls, tmp_path) -> None:
        self._setup(mock_sistema_cls, mock_dger_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(_make_nw_files(tmp_path), self._make_id_map())
        assert "lines" in result

    @patch("cobre_bridge.converters.network.Dger")
    @patch("cobre_bridge.converters.network.Sistema")
    def test_line_count_three_pairs(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        self._setup(mock_sistema_cls, mock_dger_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(_make_nw_files(tmp_path), self._make_id_map())
        assert len(result["lines"]) == 3

    @patch("cobre_bridge.converters.network.Dger")
    @patch("cobre_bridge.converters.network.Sistema")
    def test_line_capacity_structure(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        self._setup(mock_sistema_cls, mock_dger_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(_make_nw_files(tmp_path), self._make_id_map())
        for line in result["lines"]:
            assert "capacity" in line
            assert "direct_mw" in line["capacity"]
            assert "reverse_mw" in line["capacity"]
            assert "source_bus_id" in line
            assert "target_bus_id" in line

    @patch("cobre_bridge.converters.network.Dger")
    @patch("cobre_bridge.converters.network.Sistema")
    def test_line_ids_sequential(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        self._setup(mock_sistema_cls, mock_dger_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(_make_nw_files(tmp_path), self._make_id_map())
        ids = [ln["id"] for ln in result["lines"]]
        assert ids == list(range(len(ids)))

    @patch("cobre_bridge.converters.network.Dger")
    @patch("cobre_bridge.converters.network.Sistema")
    def test_first_month_used_for_capacity(
        self, mock_sistema_cls, mock_dger_cls, tmp_path
    ) -> None:
        self._setup(mock_sistema_cls, mock_dger_cls, tmp_path)
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(_make_nw_files(tmp_path), self._make_id_map())
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

        result = convert_penalties(
            _make_nw_files(tmp_path),
            {"hydros": [{"generation": {"productivity_mw_per_m3s": 1.0}}]},
        )
        for key in ("bus", "hydro", "line", "non_controllable_source"):
            assert key in result

    @patch("cobre_bridge.converters.network.Sistema")
    def test_bus_deficit_uses_first_subsystem_first_tier(
        self, mock_sistema_cls, tmp_path
    ) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_penalties

        result = convert_penalties(
            _make_nw_files(tmp_path),
            {"hydros": [{"generation": {"productivity_mw_per_m3s": 1.0}}]},
        )
        # First subsystem=1, patamar=1: custo = 500.0*1 = 500.0
        seg = result["bus"]["deficit_segments"][0]
        assert seg["cost"] == pytest.approx(500.0)

    @patch("cobre_bridge.converters.network.Sistema")
    def test_hydro_has_all_penalty_fields(self, mock_sistema_cls, tmp_path) -> None:
        _setup_sistema_mocks(mock_sistema_cls, tmp_path)
        from cobre_bridge.converters.network import convert_penalties

        result = convert_penalties(
            _make_nw_files(tmp_path),
            {"hydros": [{"generation": {"productivity_mw_per_m3s": 1.0}}]},
        )
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

        result = convert_initial_conditions(
            _make_nw_files(tmp_path), self._make_id_map()
        )
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

        result = convert_initial_conditions(
            _make_nw_files(tmp_path), self._make_id_map()
        )
        # New formula: (pct / 100) * (vol_max - vol_min) + vol_min
        # USINA_A: pct=50%, vol_min=100, vol_max=1000
        #   -> (0.50) * (1000 - 100) + 100 = 450 + 100 = 550 hm3.
        # USINA_B: pct=75%, vol_min=50, vol_max=500
        #   -> (0.75) * (500 - 50) + 50 = 337.5 + 50 = 387.5 hm3.
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        assert storage[0] == pytest.approx(550.0)
        assert storage[1] == pytest.approx(387.5)

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_storage_sorted_by_hydro_id(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(
            _make_nw_files(tmp_path), self._make_id_map()
        )
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
        result = convert_initial_conditions(
            _make_nw_files(tmp_path), self._make_id_map()
        )
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # pct clamped to 100 -> vol_max=500 -> 500.0 hm3.
        assert storage[1] == pytest.approx(500.0)

    @patch("cobre_bridge.converters.initial_conditions.Confhd")
    @patch("cobre_bridge.converters.initial_conditions.Hidr")
    def test_filling_storage_is_empty(
        self, mock_hidr_cls, mock_confhd_cls, tmp_path
    ) -> None:
        _setup_ic_mocks(mock_hidr_cls, mock_confhd_cls, tmp_path)
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(
            _make_nw_files(tmp_path), self._make_id_map()
        )
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

        buses_result = convert_buses(_make_nw_files(tmp_path), id_map)
        hydros_result = convert_hydros(_make_nw_files(tmp_path), id_map)
        thermals_result = convert_thermals(_make_nw_files(tmp_path), id_map)

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
        result = convert_hydros(_make_nw_files(tmp_path), id_map)
        valid_hydro_ids = {h["id"] for h in result["hydros"]}

        for h in result["hydros"]:
            ds = h.get("downstream_id")
            if ds is not None:
                assert ds in valid_hydro_ids, (
                    f"Hydro '{h['name']}' has downstream_id={ds} not in hydros"
                )


# ---------------------------------------------------------------------------
# _build_id_map fictitious plant filtering  (ticket-009)
# ---------------------------------------------------------------------------


def _make_confhd_df_with_fict() -> pd.DataFrame:
    """Four plants: two real, two fictitious (names start with 'FICT.')."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 3, 4],
            "nome_usina": ["USINA_A", "FICT.SERRA M", "USINA_B", "FICT.CAMPO G"],
            "posto": [1, 2, 3, 4],
            "codigo_usina_jusante": [pd.NA, pd.NA, 1, 2],
            "ree": [1, 1, 1, 1],
            "volume_inicial_percentual": [50.0, 60.0, 70.0, 80.0],
            "usina_existente": ["EX", "EX", "EX", "EX"],
            "usina_modificada": [0, 0, 0, 0],
        }
    )


class TestBuildIdMap:
    """Unit tests for ``pipeline._build_id_map`` fictitious-plant filtering."""

    @patch("inewave.newave.Ree")
    @patch("inewave.newave.Conft")
    @patch("inewave.newave.Sistema")
    @patch("inewave.newave.Confhd")
    def test_excludes_fictitious_plants(
        self,
        mock_confhd_cls,
        mock_sistema_cls,
        mock_conft_cls,
        mock_ree_cls,
        tmp_path,
    ) -> None:
        """FICT. plants must be absent from id_map.all_hydro_codes."""
        for fname in ("confhd.dat", "conft.dat", "sistema.dat", "ree.dat"):
            (tmp_path / fname).touch()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df_with_fict()
        mock_confhd_cls.read.return_value = mock_confhd

        mock_conft = MagicMock()
        mock_conft.usinas = pd.DataFrame({"codigo_usina": []})
        mock_conft_cls.read.return_value = mock_conft

        mock_sistema = MagicMock()
        mock_sistema.custo_deficit = None
        mock_sistema_cls.read.return_value = mock_sistema

        mock_ree = MagicMock()
        mock_ree.rees = None
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.pipeline import _build_id_map

        id_map = _build_id_map(_make_nw_files(tmp_path))

        # Only the two non-fictitious plants must appear.
        assert 1 in id_map.all_hydro_codes
        assert 3 in id_map.all_hydro_codes
        assert 2 not in id_map.all_hydro_codes, "FICT.SERRA M must be excluded"
        assert 4 not in id_map.all_hydro_codes, "FICT.CAMPO G must be excluded"
        assert len(id_map.all_hydro_codes) == 2

    @patch("inewave.newave.Ree")
    @patch("inewave.newave.Conft")
    @patch("inewave.newave.Sistema")
    @patch("inewave.newave.Confhd")
    def test_count_excludes_fict_plants(
        self,
        mock_confhd_cls,
        mock_sistema_cls,
        mock_conft_cls,
        mock_ree_cls,
        tmp_path,
    ) -> None:
        """15 FICT plants among 160 existing -> 145 hydro codes in id_map."""
        for fname in ("confhd.dat", "conft.dat", "sistema.dat", "ree.dat"):
            (tmp_path / fname).touch()

        n_real, n_fict = 145, 15
        rows = []
        for i in range(1, n_real + n_fict + 1):
            name = f"FICT.PLANT_{i}" if i > n_real else f"PLANT_{i}"
            rows.append(
                {
                    "codigo_usina": i,
                    "nome_usina": name,
                    "posto": i,
                    "codigo_usina_jusante": pd.NA,
                    "ree": 1,
                    "volume_inicial_percentual": 50.0,
                    "usina_existente": "EX",
                }
            )
        confhd_df = pd.DataFrame(rows)

        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_confhd_cls.read.return_value = mock_confhd

        mock_conft = MagicMock()
        mock_conft.usinas = pd.DataFrame({"codigo_usina": []})
        mock_conft_cls.read.return_value = mock_conft

        mock_sistema = MagicMock()
        mock_sistema.custo_deficit = None
        mock_sistema_cls.read.return_value = mock_sistema

        mock_ree = MagicMock()
        mock_ree.rees = None
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.pipeline import _build_id_map

        id_map = _build_id_map(_make_nw_files(tmp_path))
        assert len(id_map.all_hydro_codes) == n_real

    @patch("inewave.newave.Ree")
    @patch("inewave.newave.Conft")
    @patch("inewave.newave.Sistema")
    @patch("inewave.newave.Confhd")
    def test_no_fictitious_plants_proceeds_normally(
        self,
        mock_confhd_cls,
        mock_sistema_cls,
        mock_conft_cls,
        mock_ree_cls,
        tmp_path,
    ) -> None:
        """When no FICT. plants exist, all existing plants are included."""
        for fname in ("confhd.dat", "conft.dat", "sistema.dat", "ree.dat"):
            (tmp_path / fname).touch()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()  # standard two-plant fixture, no FICT.
        mock_confhd_cls.read.return_value = mock_confhd

        mock_conft = MagicMock()
        mock_conft.usinas = pd.DataFrame({"codigo_usina": []})
        mock_conft_cls.read.return_value = mock_conft

        mock_sistema = MagicMock()
        mock_sistema.custo_deficit = None
        mock_sistema_cls.read.return_value = mock_sistema

        mock_ree = MagicMock()
        mock_ree.rees = None
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.pipeline import _build_id_map

        id_map = _build_id_map(_make_nw_files(tmp_path))

        assert 1 in id_map.all_hydro_codes
        assert 2 in id_map.all_hydro_codes


class TestConvertHydrosDownstreamFict:
    """Downstream reference to a fictitious plant must produce downstream_id=None."""

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_downstream_to_fict_is_none(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """Plant with a fictitious downstream gets downstream_id=None.

        USINA_A (code=1) has codigo_usina_jusante=2, which is FICT.SERRA M.
        Because FICT.SERRA M is absent from id_map, the KeyError catch in
        hydro.py must produce downstream_id=None for USINA_A.
        """
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        # Build a confhd DataFrame where plant 1 points downstream to a
        # fictitious plant (code=2) that is NOT present in the id_map.
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1],
                "nome_usina": ["USINA_A"],
                "posto": [1],
                "codigo_usina_jusante": [2],  # points to the absent fict. plant
                "ree": [1],
                "volume_inicial_percentual": [50.0],
                "usina_existente": ["EX"],
                "usina_modificada": [0],
            }
        )
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_confhd_cls.read.return_value = mock_confhd

        # Hidr.cadastro for plant 1 only.
        cadastro = _make_hidr_cadastro().iloc[:1].copy()
        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_hidr_cls.read.return_value = mock_hidr

        mock_ree = MagicMock()
        mock_ree.rees = _make_ree_df()
        mock_ree_cls.read.return_value = mock_ree

        from cobre_bridge.converters.hydro import convert_hydros

        # id_map has only plant 1; plant 2 (fictitious) is absent.
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        result = convert_hydros(_make_nw_files(tmp_path), id_map)

        assert len(result["hydros"]) == 1
        assert result["hydros"][0]["downstream_id"] is None


def _make_geometry_cadastro() -> pd.DataFrame:
    """Synthetic Hidr.cadastro for generate_hydro_geometry tests.

    Two plants using real inewave column names (a0_volume_cota, a0_cota_area):
    - Plant 1: reservoir plant with vol_min=100, vol_max=1000
      volume_cota: h(v) = 300 + 0.1*v  (a0=300, a1=0.1, rest zero)
      cota_area:   A(h) = 0.5*h         (a0=0, a1=0.5, rest zero)
    - Plant 2: run-of-river with vol_min == vol_max == 50
    """
    return pd.DataFrame(
        {
            "volume_minimo": [100.0, 50.0],
            "volume_maximo": [1000.0, 50.0],
            "a0_volume_cota": [300.0, 300.0],
            "a1_volume_cota": [0.1, 0.1],
            "a2_volume_cota": [0.0, 0.0],
            "a3_volume_cota": [0.0, 0.0],
            "a4_volume_cota": [0.0, 0.0],
            "a0_cota_area": [0.0, 0.0],
            "a1_cota_area": [0.5, 0.5],
            "a2_cota_area": [0.0, 0.0],
            "a3_cota_area": [0.0, 0.0],
            "a4_cota_area": [0.0, 0.0],
        },
        index=pd.Index([1, 2], name="codigo_usina"),
    )


class TestGenerateHydroGeometry:
    """Tests for hydro.generate_hydro_geometry."""

    def test_produces_100_rows_per_plant(self) -> None:
        """A reservoir plant yields exactly 100 rows in the output table."""
        import pyarrow as pa

        from cobre_bridge.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        # Plant 1 has vol range → 100 rows. Plant 2 is run-of-river → 1 row.
        assert isinstance(table, pa.Table)
        assert len(table) == 101

        cobre_id_1 = id_map.hydro_id(1)
        cobre_id_2 = id_map.hydro_id(2)
        ids = table.column("hydro_id").to_pylist()
        assert ids.count(cobre_id_1) == 100
        assert ids.count(cobre_id_2) == 1

    def test_run_of_river_emits_single_point(self) -> None:
        """Plant with vol_min == vol_max produces one geometry row."""
        from cobre_bridge.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[2], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        assert len(table) == 1

    def test_correct_schema(self) -> None:
        """Output table has the required schema with correct column types."""
        import pyarrow as pa

        from cobre_bridge.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        assert table.schema.field("hydro_id").type == pa.int32()
        assert table.schema.field("volume_hm3").type == pa.float64()
        assert table.schema.field("height_m").type == pa.float64()
        assert table.schema.field("area_km2").type == pa.float64()

    def test_correct_schema_roundtrip_parquet(self, tmp_path) -> None:
        """Schema is preserved when written and read back as Parquet."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from cobre_bridge.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        out = tmp_path / "hydro_geometry.parquet"
        pq.write_table(table, out)
        reloaded = pq.read_table(out)

        assert reloaded.schema.field("hydro_id").type == pa.int32()
        assert reloaded.schema.field("volume_hm3").type == pa.float64()
        assert reloaded.schema.field("height_m").type == pa.float64()
        assert reloaded.schema.field("area_km2").type == pa.float64()
        assert len(reloaded) == 100

    def test_volumes_are_uniformly_spaced(self) -> None:
        """The 100 volume points are uniformly distributed on [vol_min, vol_max]."""
        import numpy as np

        from cobre_bridge.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        vols = table.column("volume_hm3").to_pylist()
        expected = np.linspace(100.0, 1000.0, 100).tolist()
        assert vols == pytest.approx(expected, rel=1e-9)

    def test_polynomial_evaluation_correctness(self) -> None:
        """Heights and areas match the expected polynomial values."""
        import numpy as np

        from cobre_bridge.converters.hydro import generate_hydro_geometry

        cadastro = _make_geometry_cadastro()
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        vols = np.array(table.column("volume_hm3").to_pylist())
        heights = np.array(table.column("height_m").to_pylist())
        areas = np.array(table.column("area_km2").to_pylist())

        # h(v) = 300 + 0.1*v
        expected_heights = 300.0 + 0.1 * vols
        np.testing.assert_allclose(heights, expected_heights, rtol=1e-9)

        # A(h) = 0.5 * h
        expected_areas = 0.5 * expected_heights
        np.testing.assert_allclose(areas, expected_areas, rtol=1e-9)

    def test_skips_all_zero_volume_cota(self) -> None:
        """Plant with all-zero volume_cota coefficients is skipped (no rows emitted)."""
        from cobre_bridge.converters.hydro import generate_hydro_geometry

        # Build a cadastro with all-zero volume_cota for plant 1.
        cadastro = _make_geometry_cadastro().copy()
        for i in range(5):
            cadastro.loc[1, f"a{i}_volume_cota"] = 0.0

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        # Should not raise; plant is silently skipped after logging a warning.
        table = generate_hydro_geometry(cadastro, id_map)
        assert len(table) == 0

    def test_negative_values_clamped_to_zero(self) -> None:
        """Negative polynomial outputs are clamped to 0.0."""
        from cobre_bridge.converters.hydro import generate_hydro_geometry

        # volume_cota: h(v) = -1000 + v  (negative at low volumes)
        # cota_area:   A(h) = -1000 + h  (negative at low heights)
        cadastro = _make_geometry_cadastro().copy()
        cadastro.loc[1, "a0_volume_cota"] = -1000.0
        cadastro.loc[1, "a1_volume_cota"] = 1.0
        cadastro.loc[1, "a2_volume_cota"] = 0.0
        cadastro.loc[1, "a3_volume_cota"] = 0.0
        cadastro.loc[1, "a4_volume_cota"] = 0.0
        cadastro.loc[1, "a0_cota_area"] = -1000.0
        cadastro.loc[1, "a1_cota_area"] = 1.0
        cadastro.loc[1, "a2_cota_area"] = 0.0
        cadastro.loc[1, "a3_cota_area"] = 0.0
        cadastro.loc[1, "a4_cota_area"] = 0.0

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        table = generate_hydro_geometry(cadastro, id_map)

        heights = table.column("height_m").to_pylist()
        areas = table.column("area_km2").to_pylist()
        assert all(h >= 0.0 for h in heights), "Heights must be >= 0"
        assert all(a >= 0.0 for a in areas), "Areas must be >= 0"


# ---------------------------------------------------------------------------
# _read_penalid unit tests  (ticket-007)
# ---------------------------------------------------------------------------


def _make_penalid_df() -> pd.DataFrame:
    """Synthetic PENALID.DAT penalties for two REEs and several variables.

    REE 1 has DESVIO=8300.0, VAZMIN=3179.35, GHMIN=4500.0 at patamar 1.
    REE 2 has DESVIO=9100.0, VAZMIN=2800.0 at patamar 1.
    Both REEs have patamar 2 rows with NaN values (unbounded tier).
    TURBMX is included to verify the "no mapping" skip path.
    """
    import math

    return pd.DataFrame(
        {
            "variavel": [
                "DESVIO",
                "DESVIO",
                "VAZMIN",
                "VAZMIN",
                "GHMIN",
                "GHMIN",
                "TURBMX",
                "TURBMX",
                "DESVIO",
                "DESVIO",
                "VAZMIN",
                "VAZMIN",
            ],
            "codigo_ree_submercado": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "patamar_penalidade": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "patamar_carga": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "valor_R$_MWh": [
                8300.0,
                math.nan,
                3179.35,
                math.nan,
                4500.0,
                math.nan,
                999.0,  # TURBMX — should be skipped (no mapping)
                math.nan,
                9100.0,
                math.nan,
                2800.0,
                math.nan,
            ],
            "valor_R$_hm3": [0.0] * 12,
        }
    )


class TestReadPenalid:
    """Unit tests for ``_read_penalid``."""

    def test_reads_penalties_by_ree(self, tmp_path) -> None:
        """Correct Cobre field names and values are returned per REE."""
        from cobre_bridge.converters.hydro import _read_penalid

        (tmp_path / "penalid.dat").touch()

        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()

        with patch("cobre_bridge.converters.hydro.Penalid") as mock_cls:
            mock_cls.read.return_value = mock_penalid
            result = _read_penalid(
                _make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat")
            )

        # REE 1 checks.
        assert 1 in result
        assert result[1]["spillage_cost"] == pytest.approx(8300.0)
        assert result[1]["outflow_violation_below_cost"] == pytest.approx(3179.35)
        assert result[1]["generation_violation_below_cost"] == pytest.approx(4500.0)
        # TURBMX must not appear (no Cobre mapping).
        assert "turbined_violation_below_cost" not in result[1]

        # REE 2 checks.
        assert 2 in result
        assert result[2]["spillage_cost"] == pytest.approx(9100.0)
        assert result[2]["outflow_violation_below_cost"] == pytest.approx(2800.0)

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        """Absent PENALID.DAT returns an empty dict without raising."""
        from cobre_bridge.converters.hydro import _read_penalid

        # No penalid.dat — pass penalid=None.
        result = _read_penalid(_make_nw_files(tmp_path, penalid=None))

        assert result == {}

    def test_nan_values_are_skipped(self, tmp_path) -> None:
        """NaN cost values at patamar 1 do not appear in the output dict."""
        import math

        from cobre_bridge.converters.hydro import _read_penalid

        (tmp_path / "penalid.dat").touch()

        df = pd.DataFrame(
            {
                "variavel": ["DESVIO", "VAZMIN"],
                "codigo_ree_submercado": [1, 1],
                "patamar_penalidade": [1, 1],
                "patamar_carga": [1, 1],
                "valor_R$_MWh": [math.nan, 5000.0],
                "valor_R$_hm3": [0.0, 0.0],
            }
        )

        mock_penalid = MagicMock()
        mock_penalid.penalidades = df

        with patch("cobre_bridge.converters.hydro.Penalid") as mock_cls:
            mock_cls.read.return_value = mock_penalid
            result = _read_penalid(
                _make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat")
            )

        assert 1 in result
        # DESVIO had NaN — must be absent.
        assert "spillage_cost" not in result[1]
        # VAZMIN had 5000.0 — must be present.
        assert result[1]["outflow_violation_below_cost"] == pytest.approx(5000.0)

    def test_patamar2_rows_ignored(self, tmp_path) -> None:
        """Tier-2 patamar rows are excluded even when they have numeric values."""
        from cobre_bridge.converters.hydro import _read_penalid

        (tmp_path / "penalid.dat").touch()

        df = pd.DataFrame(
            {
                "variavel": ["DESVIO", "DESVIO"],
                "codigo_ree_submercado": [1, 1],
                "patamar_penalidade": [2, 2],  # only tier-2 rows — should be skipped
                "patamar_carga": [1, 1],
                "valor_R$_MWh": [8300.0, 8300.0],
                "valor_R$_hm3": [0.0, 0.0],
            }
        )

        mock_penalid = MagicMock()
        mock_penalid.penalidades = df

        with patch("cobre_bridge.converters.hydro.Penalid") as mock_cls:
            mock_cls.read.return_value = mock_penalid
            result = _read_penalid(
                _make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat")
            )

        assert result == {}


class TestConvertHydrosPenalid:
    """Integration tests for PENALID.DAT -> hydro penalties in convert_hydros."""

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_penalties_from_penalid(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """Plants get penalties populated from PENALID.DAT via their REE code."""
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        (tmp_path / "penalid.dat").touch()

        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        with patch("cobre_bridge.converters.hydro.Penalid") as mock_cls:
            mock_cls.read.return_value = mock_penalid
            result = convert_hydros(
                _make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat"), id_map
            )

        # Both plants are in REE 1 (see _make_confhd_df).
        # All plants should share REE 1 penalties.
        for hydro in result["hydros"]:
            assert hydro["penalties"] is not None, (
                f"Plant '{hydro['name']}' should have non-None penalties"
            )
            assert hydro["penalties"]["spillage_cost"] == pytest.approx(8300.0)
            assert hydro["penalties"]["outflow_violation_below_cost"] == pytest.approx(
                3179.35
            )
            assert hydro["penalties"][
                "generation_violation_below_cost"
            ] == pytest.approx(4500.0)

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_missing_penalid_leaves_penalties_none(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """When PENALID.DAT is absent, every hydro entry has penalties=None."""
        _setup_hydro_mocks(mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path)
        # Deliberately do NOT create penalid.dat.

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(_make_nw_files(tmp_path), id_map)

        for hydro in result["hydros"]:
            assert hydro["penalties"] is None, (
                f"Plant '{hydro['name']}' should have penalties=None "
                "when PENALID.DAT is absent"
            )

    @patch("cobre_bridge.converters.hydro.Ree")
    @patch("cobre_bridge.converters.hydro.Confhd")
    @patch("cobre_bridge.converters.hydro.Hidr")
    def test_different_rees_get_different_penalties(
        self, mock_hidr_cls, mock_confhd_cls, mock_ree_cls, tmp_path
    ) -> None:
        """Plants in different REEs receive penalty values for their own REE."""
        for fname in ("hidr.dat", "confhd.dat", "ree.dat"):
            (tmp_path / fname).touch()

        # Two plants in different REEs: plant 1 in REE 1, plant 2 in REE 2.
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2],
                "nome_usina": ["USINA_A", "USINA_B"],
                "posto": [1, 2],
                "codigo_usina_jusante": [pd.NA, pd.NA],
                "ree": [1, 2],
                "volume_inicial_percentual": [50.0, 75.0],
                "usina_existente": ["EX", "EX"],
                "usina_modificada": [0, 0],
            }
        )
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_confhd_cls.read.return_value = mock_confhd

        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()
        mock_hidr_cls.read.return_value = mock_hidr

        # REE table: REE 1 -> subsystem 1, REE 2 -> subsystem 1.
        ree_df = pd.DataFrame(
            {"codigo": [1, 2], "nome": ["SE", "S"], "submercado": [1, 1]}
        )
        mock_ree = MagicMock()
        mock_ree.rees = ree_df
        mock_ree_cls.read.return_value = mock_ree

        (tmp_path / "penalid.dat").touch()
        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        with patch("cobre_bridge.converters.hydro.Penalid") as mock_cls:
            mock_cls.read.return_value = mock_penalid
            result = convert_hydros(
                _make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat"), id_map
            )

        hydros_by_name = {h["name"]: h for h in result["hydros"]}

        # USINA_A is in REE 1.
        pen_a = hydros_by_name["USINA_A"]["penalties"]
        assert pen_a is not None
        assert pen_a["spillage_cost"] == pytest.approx(8300.0)

        # USINA_B is in REE 2.
        pen_b = hydros_by_name["USINA_B"]["penalties"]
        assert pen_b is not None
        assert pen_b["spillage_cost"] == pytest.approx(9100.0)
        assert pen_b["outflow_violation_below_cost"] == pytest.approx(2800.0)


# ---------------------------------------------------------------------------
# Helper builders for water-withdrawal tests.
# ---------------------------------------------------------------------------


def _make_dsvagua_df(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic dsvagua desvios DataFrame from explicit rows."""
    return pd.DataFrame(rows)


def _make_withdrawal_confhd_df(
    postos: list[tuple[int, int, int]],
) -> pd.DataFrame:
    """Build a minimal confhd DataFrame mapping posto -> hydro code.

    Parameters
    ----------
    postos:
        List of ``(posto, codigo_usina, ree)`` tuples.
    """
    return pd.DataFrame(
        {
            "posto": [p[0] for p in postos],
            "codigo_usina": [p[1] for p in postos],
            "nome_usina": [f"USINA_{p[0]}" for p in postos],
            "ree": [p[2] for p in postos],
            "usina_existente": ["EX"] * len(postos),
            "codigo_usina_jusante": [pd.NA] * len(postos),
            "volume_inicial_percentual": [50.0] * len(postos),
            "usina_modificada": [0] * len(postos),
        }
    )


def _make_dger_mock(start_year: int, start_month: int, num_anos: int) -> MagicMock:
    """Build a MagicMock mimicking the Dger object."""
    mock = MagicMock()
    mock.ano_inicio_estudo = start_year
    mock.mes_inicio_estudo = start_month
    mock.num_anos_estudo = num_anos
    return mock


class TestWaterWithdrawalConversion:
    """Unit tests for ``convert_water_withdrawal`` in ``hydro.py``."""

    def _make_id_map(self) -> NewaveIdMap:
        """Two hydros: NEWAVE codes 10 and 20 -> Cobre IDs 0 and 1."""
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[10, 20],
            thermal_codes=[],
        )

    def test_basic_returns_correct_schema(self, tmp_path: Path) -> None:
        """Two postos, three dates each: table has the three expected columns."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {
                "codigo_usina": 1,
                "data": datetime.datetime(2020, 1, 1),
                "valor": -2.0,
            },
            {
                "codigo_usina": 1,
                "data": datetime.datetime(2020, 2, 1),
                "valor": -3.0,
            },
            {
                "codigo_usina": 2,
                "data": datetime.datetime(2020, 1, 1),
                "valor": -1.0,
            },
        ]
        confhd_df = _make_withdrawal_confhd_df([(1, 10, 1), (2, 20, 1)])
        dger_mock = _make_dger_mock(2020, 1, 5)

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = dger_mock.ano_inicio_estudo
        mock_dger.mes_inicio_estudo = dger_mock.mes_inicio_estudo
        mock_dger.num_anos_estudo = dger_mock.num_anos_estudo

        with (
            patch(
                "inewave.newave.Dsvagua.read",
                return_value=mock_dsvagua,
            ),
            patch(
                "inewave.newave.Confhd.read",
                return_value=mock_confhd,
            ),
            patch(
                "inewave.newave.Dger.read",
                return_value=mock_dger,
            ),
        ):
            result = convert_water_withdrawal(
                _make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"),
                self._make_id_map(),
            )

        assert result is not None
        assert result.schema.names == ["hydro_id", "stage_id", "water_withdrawal_m3s"]
        import pyarrow as pa

        assert result.schema.field("hydro_id").type == pa.int32()
        assert result.schema.field("stage_id").type == pa.int32()
        assert result.schema.field("water_withdrawal_m3s").type == pa.float64()

    def test_sign_negation_and_stage_mapping(self, tmp_path: Path) -> None:
        """valor=-5.0 at 2020-02 -> water_withdrawal_m3s=5.0, stage_id=1."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 1, "data": datetime.datetime(2020, 2, 1), "valor": -5.0}
        ]
        confhd_df = _make_withdrawal_confhd_df([(1, 10, 1)])
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        with (
            patch(
                "inewave.newave.Dsvagua.read",
                return_value=mock_dsvagua,
            ),
            patch(
                "inewave.newave.Confhd.read",
                return_value=mock_confhd,
            ),
            patch(
                "inewave.newave.Dger.read",
                return_value=mock_dger,
            ),
        ):
            result = convert_water_withdrawal(
                _make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"), id_map
            )

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        assert row["hydro_id"][0] == id_map.hydro_id(10)
        assert row["stage_id"][0] == 1
        assert row["water_withdrawal_m3s"][0] == pytest.approx(5.0)

    def test_groupby_sum_same_posto_same_date(self, tmp_path: Path) -> None:
        """Two rows with the same posto/date are summed then negated."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 1, "data": datetime.datetime(2020, 1, 1), "valor": -3.0},
            {"codigo_usina": 1, "data": datetime.datetime(2020, 1, 1), "valor": -7.0},
        ]
        confhd_df = _make_withdrawal_confhd_df([(1, 10, 1)])
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        with (
            patch(
                "inewave.newave.Dsvagua.read",
                return_value=mock_dsvagua,
            ),
            patch(
                "inewave.newave.Confhd.read",
                return_value=mock_confhd,
            ),
            patch(
                "inewave.newave.Dger.read",
                return_value=mock_dger,
            ),
        ):
            result = convert_water_withdrawal(
                _make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"), id_map
            )

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        # -3.0 + -7.0 = -10.0; negated -> 10.0
        assert row["water_withdrawal_m3s"][0] == pytest.approx(10.0)
        assert row["stage_id"][0] == 0

    def test_missing_dsvagua_file_returns_none(self, tmp_path: Path) -> None:
        """When dsvagua.dat is absent the converter returns None without error."""
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        # Do NOT create dsvagua.dat — only create the other required files.
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        result = convert_water_withdrawal(
            _make_nw_files(tmp_path, dsvagua=None), self._make_id_map()
        )
        assert result is None

    def test_empty_desvios_returns_none(self, tmp_path: Path) -> None:
        """When desvios is None the converter returns None."""
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = None

        with patch(
            "inewave.newave.Dsvagua.read",
            return_value=mock_dsvagua,
        ):
            result = convert_water_withdrawal(
                _make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"),
                self._make_id_map(),
            )

        assert result is None

    def test_unknown_posto_skipped_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A posto not present in confhd.dat is skipped and a warning is logged."""
        import datetime
        import logging

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "confhd.dat").touch()
        (tmp_path / "dger.dat").touch()

        # posto 99 is not in confhd (which maps only posto 1 -> code 10).
        rows = [
            {"codigo_usina": 1, "data": datetime.datetime(2020, 1, 1), "valor": -4.0},
            {"codigo_usina": 99, "data": datetime.datetime(2020, 1, 1), "valor": -2.0},
        ]
        confhd_df = _make_withdrawal_confhd_df([(1, 10, 1)])
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        with (
            patch(
                "inewave.newave.Dsvagua.read",
                return_value=mock_dsvagua,
            ),
            patch(
                "inewave.newave.Confhd.read",
                return_value=mock_confhd,
            ),
            patch(
                "inewave.newave.Dger.read",
                return_value=mock_dger,
            ),
            caplog.at_level(logging.WARNING, logger="cobre_bridge.converters.hydro"),
        ):
            result = convert_water_withdrawal(
                _make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"), id_map
            )

        # The known posto 1 produces one valid row; posto 99 is skipped.
        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        assert row["water_withdrawal_m3s"][0] == pytest.approx(4.0)
        # A warning must have been logged for the unknown posto.
        assert any("99" in record.message for record in caplog.records)
