"""Unit tests for the source model entity conversion functions.

All inewave I/O is mocked via ``unittest.mock.patch`` so no real the source model files
are required.  The tests use small synthetic DataFrames that cover the logic exercised
by each converter.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles
from tests.conftest import make_case, make_nw_files


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
    volref_saz: Path | None = None,
    shist: Path | None = None,
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
        exph=None,
        manutt=manutt,
        c_adic=c_adic,
        cvar=cvar,
        agrint=agrint,
        re_dat=None,
        volref_saz=volref_saz,
        shist=shist,
        adterm=None,
        polinjus=None,
        tratamento_fpha=None,
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

    def test_hydro_id_remapping_preserves_confhd_order(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[],
            hydro_codes=[10, 5, 20],
            thermal_codes=[],
        )
        assert id_map.hydro_id(10) == 0
        assert id_map.hydro_id(5) == 1
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

    def test_all_hydro_codes_in_cobre_id_order(self) -> None:
        id_map = NewaveIdMap(
            subsystem_ids=[], hydro_codes=[30, 10, 20], thermal_codes=[]
        )
        assert id_map.all_hydro_codes == [30, 10, 20]

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

    For monthly-regulated plants the height is evaluated at 65% of useful storage
    (``v_65 = vmin + 0.65 * (vmax - vmin)``), matching the source model's
    ``produtibilidade_altura_65`` convention.

    USINA_A: [volume_minimo=100, volume_maximo=1000]
    - v_65 = 100 + 0.65 * 900 = 685.0
    - h(v_65) = 300 + 0.1 * 685.0 = 368.5
    - net_drop = 368.5 - 50.0 = 318.5
    - productivity_A = 0.9 * 318.5 = 286.65

    USINA_B: [volume_minimo=50, volume_maximo=500]
    - v_65 = 50 + 0.65 * 450 = 342.5
    - h(v_65) = 300 + 0.1 * 342.5 = 334.25
    - net_drop = 334.25 - 50.0 = 284.25
    - productivity_B = 0.85 * 284.25 = 241.6125

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
        # 1 -> 2 direct (sentido=0 means de->para, i.e. 1->2)
        {
            "submercado_de": 1,
            "submercado_para": 2,
            "sentido": 0,
            "data": d,
            "valor": 3000.0,
        },
        # 2 -> 1 reverse (sentido=0 means de->para, i.e. 2->1)
        {
            "submercado_de": 2,
            "submercado_para": 1,
            "sentido": 0,
            "data": d,
            "valor": 2500.0,
        },
        # 1 -> 99 direct
        {
            "submercado_de": 1,
            "submercado_para": 99,
            "sentido": 0,
            "data": d,
            "valor": 4000.0,
        },
        # 99 -> 1 reverse
        {
            "submercado_de": 99,
            "submercado_para": 1,
            "sentido": 0,
            "data": d,
            "valor": 2000.0,
        },
        # 2 -> 99 direct
        {
            "submercado_de": 2,
            "submercado_para": 99,
            "sentido": 0,
            "data": d,
            "valor": 1500.0,
        },
        # 99 -> 2 reverse
        {
            "submercado_de": 99,
            "submercado_para": 2,
            "sentido": 0,
            "data": d,
            "valor": 1200.0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NE-with-filling fixtures (ticket-009): a JURUENA-shaped run-of-river ('S')
# plant (code 309) admitted into the active set by its exph dead-volume row.
# ---------------------------------------------------------------------------


def _make_ne_confhd_df() -> pd.DataFrame:
    """Two EX plants (1, 2) plus one NE filling plant (309 JURUENA)."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 309],
            "nome_usina": ["USINA_A", "USINA_B", "JURUENA"],
            "posto": [1, 2, 226],
            "codigo_usina_jusante": [pd.NA, 1, pd.NA],
            "ree": [1, 1, 1],
            "volume_inicial_percentual": [50.0, 75.0, 0.0],
            "usina_existente": ["EX", "EX", "NE"],
            "usina_modificada": [0, 0, 0],
        }
    )


def _make_ne_cadastro() -> pd.DataFrame:
    """Two-plant synthetic cadastro extended with JURUENA (code 309).

    JURUENA is run-of-river (``tipo_regulacao='S'``) with
    ``volume_minimo == volume_maximo == 2.93`` so its reservoir block already
    collapses to the single point 2.93 hm³ via the existing 'S' path.
    """
    base = _make_hidr_cadastro()
    juruena = base.loc[[1]].copy()
    juruena.index = pd.Index([309], name="codigo_usina")
    juruena["nome_usina"] = "JURUENA"
    juruena["posto"] = 226
    juruena["codigo_usina_jusante"] = pd.NA
    juruena["volume_minimo"] = 2.93
    juruena["volume_maximo"] = 2.93
    juruena["volume_referencia"] = 2.93
    juruena["tipo_regulacao"] = "S"
    return pd.concat([base, juruena])


def _make_ne_exph_mock(*, duracao: int = 1, volume_morto: float = 0.0) -> MagicMock:
    """An ``exph`` reader mock whose ``expansoes`` carries JURUENA's filling row.

    The schedule row (non-null ``data_inicio_enchimento``) is what
    ``filling_hydro_codes`` selects; a trailing unit row with ``NaT`` mirrors the
    real exph layout (one schedule row + one row per generating unit).
    """
    expansoes = pd.DataFrame(
        {
            "codigo_usina": [309, 309],
            "nome_usina": ["JURUENA", "JURUENA"],
            "data_inicio_enchimento": [
                pd.Timestamp("2024-10-01"),
                pd.NaT,
            ],
            "duracao_enchimento": [duracao, 0],
            "volume_morto": [volume_morto, 0.0],
            "data_entrada_operacao": [
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2024-11-01"),
            ],
        }
    )
    exph = MagicMock()
    exph.expansoes = expansoes
    return exph


def _ne_filling_case(tmp_path, *, duracao: int = 1, volume_morto: float = 0.0):
    """A ``NewaveCase`` with JURUENA (NE+filling) under a Sep-2024 3-year horizon.

    Study start Sep 2024 ⇒ stage 0 = Sep, stage 1 = Oct, stage 2 = Nov. JURUENA's
    Oct-2024 filling start maps to ``start_sid == 1``; with ``duracao == 1`` the
    entry is ``entry_sid == 2`` (design §5).
    """
    return _hydro_case(
        tmp_path,
        cadastro=_make_ne_cadastro(),
        confhd=_make_ne_confhd_df(),
        dger=_make_hydro_dger_mock(
            start_year=2024, start_month=9, num_anos=3, num_anos_pos=0
        ),
        exph=_make_ne_exph_mock(duracao=duracao, volume_morto=volume_morto),
    )


def _ne_filling_id_map() -> NewaveIdMap:
    """Id-map including JURUENA (309) so ``hydro_id(309)`` resolves."""
    return NewaveIdMap(
        subsystem_ids=[1],
        hydro_codes=[1, 2, 309],
        thermal_codes=[],
    )


class TestConvertHydros:
    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def test_returns_hydros_key(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        assert "hydros" in result

    def test_hydro_count_matches_existing_plants(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        assert len(result["hydros"]) == 2

    def test_hydro_ids_are_zero_based_and_sorted(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        ids = [h["id"] for h in result["hydros"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    def test_hydro_has_required_fields(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
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

    def test_run_of_river_S_storage_collapsed_to_vmin(self, tmp_path) -> None:
        """``tipo_regulacao='S'`` (fio-d'água) collapses storage to Vmin.

        The source model treats 'S' plants as run-of-river with no usable buffer
        (ITAIPU, the only 'S' plant, sits at VARMPUH 0% = Vmin every stage, spilling the
        turbine-excess inflow).  The converter must pin min==max==Vmin so cobre doesn't
        store and shift that surplus across stages.
        """
        from cobre_bridge.converters.hydro import convert_hydros

        cadastro = _make_hidr_cadastro()
        cadastro.loc[1, "tipo_regulacao"] = "S"  # USINA_A: Vmin 100, Vmax 1000
        case = _hydro_case(tmp_path, cadastro=cadastro)

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["reservoir"]["min_storage_hm3"] == 100.0
        assert hydro_a["reservoir"]["max_storage_hm3"] == 100.0
        # 'M' plant unchanged (keeps its full range).
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")
        assert hydro_b["reservoir"]["min_storage_hm3"] == 50.0
        assert hydro_b["reservoir"]["max_storage_hm3"] == 500.0

    def test_cascade_downstream_linkage(self, tmp_path) -> None:
        """Plant 2 (code=2) is downstream of plant 1 (code=1)."""
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        # USINA_A (code=1, cobre id=0) has no downstream.
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["downstream_id"] is None
        # USINA_B (code=2, cobre id=1) is downstream of USINA_A (cobre id=0).
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")
        assert hydro_b["downstream_id"] == 0

    def test_bus_id_matches_ree_subsystem(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            # Both plants are in REE 1 -> subsystem 1 -> bus 0.
            assert h["bus_id"] == 0

    def test_generation_values_match_machine_sets(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        gen = hydro_a["generation"]
        # USINA_A: 1 set, 4 machines, 200 MW each, flow 222.2 each.
        assert gen["max_generation_mw"] == pytest.approx(4 * 200.0)
        assert gen["max_turbined_m3s"] == pytest.approx(4 * 222.2)
        # On cobre HEAD productivity lives in hydro_production_models.json,
        # not in the hydros.json generation block. ρ_esp surfaces as a
        # top-level optional field for cobre's energy-conversion pipeline.
        assert "productivity_mw_per_m3s" not in gen
        assert hydro_a["specific_productivity_mw_per_m3s_per_m"] == pytest.approx(0.9)

    def test_schema_key_present(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        assert "$schema" in result
        assert "hydros.schema.json" in result["$schema"]

    def test_hydro_code_absent_in_hidr_raises_value_error(self, tmp_path) -> None:
        # Set up mocks but make the cadastro empty (no plants).
        case = _hydro_case(tmp_path, cadastro=pd.DataFrame())

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        with pytest.raises(ValueError, match="not found in hidr.dat"):
            convert_hydros(case, id_map)

    def test_hydraulic_losses_factor(self, tmp_path) -> None:
        """tipo_perda=1 and perdas=5.0 (%) -> hydraulic_losses factor dict."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 1
        cadastro["perdas"] = 5.0  # 5% — stored as percentage in hidr.dat

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] == {
                "type": "factor",
                "value": pytest.approx(0.05),  # 5% / 100 = 0.05
            }

    def test_hydraulic_losses_constant(self, tmp_path) -> None:
        """tipo_perda=2 and perdas=3.5 -> hydraulic_losses constant dict."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 2
        cadastro["perdas"] = 3.5

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] == {
                "type": "constant",
                "value_m": pytest.approx(3.5),
            }

    def test_hydraulic_losses_none_when_zero(self, tmp_path) -> None:
        """perdas=0.0 produces hydraulic_losses=None regardless of tipo_perda."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro["tipo_perda"] = 1
        cadastro["perdas"] = 0.0

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["hydraulic_losses"] is None

    def test_evaporation_reference_volumes_from_volref_saz(self, tmp_path) -> None:
        """Plant with seasonal volref → reference_volumes_hm3 emitted as
        ``vmin + volref_saz[m]`` per calendar month."""
        # Only USINA_A (code=1) gets a non-zero seasonal row.
        # vmin_A=100, vmax_A=1000 → useful volumes 50..600 all inside the range.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": [50.0 * m for m in range(1, 13)],
            }
        )
        case = _hydro_case(
            tmp_path,
            volref_volumes=volref_df,
            volref_saz=tmp_path / "volref_saz.dat",
        )

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")

        # USINA_A: reference_volumes_hm3 = vmin_A + useful → 150, 200, ..., 700.
        assert hydro_a["evaporation"] is not None
        assert hydro_a["evaporation"]["coefficients_mm"] == [1.5] * 12
        assert hydro_a["evaporation"]["reference_volumes_hm3"] == [
            100.0 + 50.0 * m for m in range(1, 13)
        ]
        # USINA_B has no row in volref_saz → reference_volumes_hm3 omitted.
        assert "reference_volumes_hm3" not in hydro_b["evaporation"]

    def test_evaporation_reference_volumes_absent_for_all_zero_row(
        self, tmp_path
    ) -> None:
        """All-zero volref_saz row is the source model's sentinel; cobre falls back to
        its mid-storage default, so reference_volumes_hm3 is NOT emitted."""
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12 + [2] * 12,
                "nome_usina": ["USINA_A"] * 12 + ["USINA_B"] * 12,
                "mes": list(range(1, 13)) * 2,
                "valor": [0.0] * 24,
            }
        )
        case = _hydro_case(
            tmp_path,
            volref_volumes=volref_df,
            volref_saz=tmp_path / "volref_saz.dat",
        )

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        for h in result["hydros"]:
            assert h["evaporation"] is not None
            assert "reference_volumes_hm3" not in h["evaporation"]

    def test_evaporation_reference_volumes_clamp_into_reservoir_range(
        self, tmp_path
    ) -> None:
        """Useful volumes larger than (vmax-vmin) get clamped to vmax — the
        cobre schema requires every reference volume in [min_storage,
        max_storage], so we never emit a value outside the reservoir
        bounds even when volref_saz has out-of-range data."""
        # USINA_A has useful=[100,200,...,1200]. Useful range is 900 so
        # values 1000, 1100, 1200 exceed vmax (1000). Expect clamping to vmax.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": [100.0 * m for m in range(1, 13)],
            }
        )
        case = _hydro_case(
            tmp_path,
            volref_volumes=volref_df,
            volref_saz=tmp_path / "volref_saz.dat",
        )

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        ref_volumes = hydro_a["evaporation"]["reference_volumes_hm3"]
        assert len(ref_volumes) == 12
        # m=1..9 → vmin + 100*m = 200..1000, all <= vmax=1000.
        # m=10..12 → would be 1100, 1200, 1300 → clamped to 1000.
        for v in ref_volumes:
            assert 100.0 <= v <= 1000.0
        # Last three months hit the cap.
        assert ref_volumes[-3:] == [1000.0, 1000.0, 1000.0]

    def test_teif_ip_derates_turbined_not_generation(self, tmp_path) -> None:
        """TEIF/IP availability derates ``max_turbined`` but NOT ``max_generation``.

        ``max_generation`` is the rated installed power ``Σ n·p_nom`` — the source
        model's FPHA ``GHmax`` (verified TUCURUI 7445, QUEBRA QUEIX 120), independent of
        the production function and *not* availability-derated. ``max_turbined`` is the
        head-corrected engolimento, which an unavailable unit cannot pass, so it carries
        the ``0.95 * 0.97`` availability factor (the source model's operational cap —
        verified QUEBRA QUEIX 113.01 m³/s == head-corrected engolimento).
        """
        cadastro = _make_hidr_cadastro().copy()
        # Override only USINA_A (code=1) with non-zero TEIF/IP.
        cadastro.loc[1, "teif"] = 5.0
        cadastro.loc[1, "ip"] = 3.0

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # max_generation = rated nameplate (4 machines * 200 MW = 800 MW), NOT
        # availability-derated: The source model's GHmax is the installed capacity.
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)
        # max_turbined IS derated — an unavailable unit can't pass water either,
        # so the head-corrected engolimento carries the availability factor.
        assert hydro_a["generation"]["max_turbined_m3s"] == pytest.approx(
            4 * 222.2 * 0.95 * 0.97
        )
        # min_generation_mw must NOT be derated (it is zero here)
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)

    def test_zero_teif_ip_no_derating(self, tmp_path) -> None:
        """TEIF=0% and IP=0% leaves max_generation_mw unchanged."""
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # teif=0, ip=0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)

    def test_nan_teif_treated_as_zero(self, tmp_path) -> None:
        """NaN teif is treated as 0.0 — no derating, no error."""
        cadastro = _make_hidr_cadastro().copy()
        cadastro.loc[1, "teif"] = float("nan")
        cadastro.loc[1, "ip"] = float("nan")

        case = _hydro_case(tmp_path, cadastro=cadastro)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        # NaN treated as 0 -> factor = 1.0 -> no change from nominal 800 MW
        assert hydro_a["generation"]["max_generation_mw"] == pytest.approx(800.0)

    # --- FILLING phase emission (ticket-009) -------------------------------

    def test_ne_plant_emits_filling_block(self, tmp_path) -> None:
        """An admitted NE plant emits its entry stage + filling contract.

        JURUENA (code 309) fills Oct 2024 (start_sid=1) and enters Nov 2024
        (entry_sid=2); the single-stage rate is ``2.93 / ζ_Oct`` with
        ``ζ_Oct = 744 * 3600 / 1e6 = 2.6784`` (design §5).
        """
        case = _ne_filling_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, _ne_filling_id_map())
        juruena = next(h for h in result["hydros"] if h["name"] == "JURUENA")
        assert juruena["entry_stage_id"] == 2
        assert juruena["exit_stage_id"] is None
        assert juruena["filling"] == {
            "start_stage_id": 1,
            "filling_min_rate_m3s": pytest.approx(2.93 / 2.6784, rel=1e-6),
        }
        # Reservoir still collapses to the 'S' single point (no new branch).
        assert juruena["reservoir"]["min_storage_hm3"] == pytest.approx(2.93)
        assert juruena["reservoir"]["max_storage_hm3"] == pytest.approx(2.93)

    def test_ex_plants_keep_none_filling(self, tmp_path) -> None:
        """EX plants in the same case keep entry/exit/filling all None."""
        case = _ne_filling_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, _ne_filling_id_map())
        for name in ("USINA_A", "USINA_B"):
            hydro = next(h for h in result["hydros"] if h["name"] == name)
            assert hydro["entry_stage_id"] is None
            assert hydro["exit_stage_id"] is None
            assert hydro["filling"] is None

    def test_ne_plant_zero_duration_omits_filling(self, tmp_path) -> None:
        """``duracao_enchimento == 0`` ⇒ entry == start, no filling block.

        cobre rejects ``start_stage_id >= entry_stage_id``, so the empty window
        emits ``entry_stage_id`` only and keeps ``filling`` None (design §8).
        """
        case = _ne_filling_case(tmp_path, duracao=0)
        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, _ne_filling_id_map())
        juruena = next(h for h in result["hydros"] if h["name"] == "JURUENA")
        # Oct-2024 start under Sep-2024 horizon ⇒ start_sid == 1; duracao 0 ⇒
        # entry_sid == start_sid == 1.
        assert juruena["entry_stage_id"] == 1
        assert juruena["filling"] is None

    def test_ne_plant_filling_past_horizon_no_crash(self, tmp_path) -> None:
        """An NE plant whose filling completes past the horizon converts cleanly.

        Under a SHORT 3-stage horizon (Oct–Dec 2024) JURUENA's Oct-2024 filling
        start maps to ``start_sid == 0`` and a 6-month ``duracao`` pushes the
        entry to ``entry_sid == 6 > total_stages (3)`` — a valid case (design §8):
        the plant fills but never operates within the study, yet is still emitted.
        The rate is summed only over the in-horizon stages (the clamp in the
        caller), so ``convert_hydros`` must NOT raise ``IndexError`` indexing the
        per-stage date list, and the true unclamped ``entry_stage_id`` is emitted.
        """
        case = _hydro_case(
            tmp_path,
            cadastro=_make_ne_cadastro(),
            confhd=_make_ne_confhd_df(),
            # 3-stage horizon: Oct 2024 start, 1 study year, 0 post ⇒
            # study_months = (13 - 10) = 3; stages 0=Oct, 1=Nov, 2=Dec 2024.
            dger=_make_hydro_dger_mock(
                start_year=2024, start_month=10, num_anos=1, num_anos_pos=0
            ),
            # duracao 6: Oct-2024 start ⇒ start_sid == 0, entry_sid == 6 > 3.
            exph=_make_ne_exph_mock(duracao=6, volume_morto=0.0),
        )
        from cobre_bridge.converters.hydro import convert_hydros

        # Must not raise (the IndexError this fix guards against).
        result = convert_hydros(case, _ne_filling_id_map())
        juruena = next(h for h in result["hydros"] if h["name"] == "JURUENA")

        # True, unclamped filling-complete stage is emitted (≥ total_stages == 3).
        assert juruena["entry_stage_id"] == 6
        assert juruena["entry_stage_id"] >= case.horizon.total_stages

        # entry_sid (6) > start_sid (0) ⇒ a filling block is emitted; its rate is
        # finite and non-negative (summed only over the in-horizon stages 0..2).
        filling = juruena["filling"]
        assert filling is not None
        rate = filling["filling_min_rate_m3s"]
        assert math.isfinite(rate)
        assert rate >= 0.0


def _make_hydro_dger_mock(
    *,
    start_year: int = 2024,
    start_month: int = 1,
    num_anos: int = 1,
    num_anos_pos: int = 0,
) -> MagicMock:
    """A ``dger`` mock that yields a concrete study horizon for ``convert_hydros``.

    ``convert_hydros`` builds the per-stage date list from ``case.horizon``
    (which reads these four ``dger`` fields) once, unconditionally, before the
    plant loop — so every ``_hydro_case`` needs a ``dger`` that resolves all four
    horizon fields (the module's other ``_make_dger_mock`` leaves
    ``num_anos_pos_estudo`` unset). ``funcao_producao_uhe = 1`` selects the linear
    (constant-productivity) generation model, matching the synthetic plants.
    """
    dger = MagicMock()
    dger.ano_inicio_estudo = start_year
    dger.mes_inicio_estudo = start_month
    dger.num_anos_estudo = num_anos
    dger.num_anos_pos_estudo = num_anos_pos
    dger.funcao_producao_uhe = 1
    return dger


def _hydro_case(
    tmp_path,
    *,
    cadastro: pd.DataFrame | None = None,
    confhd: pd.DataFrame | None = None,
    rees: pd.DataFrame | None = None,
    modif=None,
    volref_volumes: pd.DataFrame | None = None,
    ghmin=None,
    penalid=None,
    dsvagua=None,
    dger=None,
    exph=None,
    **file_overrides,
):
    """Build a ``NewaveCase`` with mock hydro readers pre-cached.

    The three required hydro files (hidr/confhd/ree) default to the shared
    synthetic fixtures; pass ``cadastro`` / ``confhd`` / ``rees`` DataFrames to
    override. Optional readers (modif, volref_saz, ghmin, penalid, dsvagua) are
    passed as already-built mock reader objects; for those guarded behind a
    ``case.files.X`` path check, set the matching path via ``file_overrides``
    (e.g. ``volref_saz=tmp_path / "volref_saz.dat"``).

    ``dger`` defaults to :func:`_make_hydro_dger_mock` (a Jan-2024 one-year horizon) so
    ``case.horizon`` always resolves; pass a custom ``dger`` mock to drive a
    different horizon. ``exph`` is the dead-volume filling reader mock (``None``
    by default — EX-only cases admit no filling plant).
    """
    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro() if cadastro is None else cadastro

    mock_confhd = MagicMock()
    mock_confhd.usinas = _make_confhd_df() if confhd is None else confhd

    mock_ree = MagicMock()
    mock_ree.rees = _make_ree_df() if rees is None else rees

    parsed: dict = {
        "hidr": mock_hidr,
        "confhd": mock_confhd,
        "ree": mock_ree,
        "dger": _make_hydro_dger_mock() if dger is None else dger,
        "exph": exph,
    }

    if volref_volumes is not None:
        mock_volref = MagicMock()
        mock_volref.volumes = volref_volumes
        parsed["volref_saz"] = mock_volref
    if modif is not None:
        parsed["modif"] = modif
    if ghmin is not None:
        parsed["ghmin"] = ghmin
    if penalid is not None:
        parsed["penalid"] = penalid
    if dsvagua is not None:
        parsed["dsvagua"] = dsvagua

    files = make_nw_files(tmp_path, **file_overrides)
    return make_case(files, **parsed)


# ---------------------------------------------------------------------------
# _apply_permanent_overrides unit tests  (ticket-004)
# ---------------------------------------------------------------------------


class TestApplyPermanentOverrides:
    """Unit tests for ``_apply_permanent_overrides``."""

    def _base_cadastro(self) -> pd.DataFrame:
        return _make_hidr_cadastro()

    def _modif_case(self, tmp_path, mock_modif):
        """Build a case whose MODIF reader is *mock_modif* (path set)."""
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_missing_modif_returns_unchanged(self, tmp_path) -> None:
        """No MODIF.DAT -> cadastro returned unchanged."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        cadastro = self._base_cadastro()
        result = _apply_permanent_overrides(cadastro, make_case(tmp_path, modif=None))
        pd.testing.assert_frame_equal(result, cadastro)

    def test_volmax_override(self, tmp_path) -> None:
        """VOLMAX record updates volume_maximo for the target plant."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

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

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(2000.0)
        # Plant 2 must be unchanged.
        assert float(result.loc[2, "volume_maximo"]) == pytest.approx(500.0)

    def test_vazmin_override(self, tmp_path) -> None:
        """VAZMIN record updates vazao_minima_historica for the target plant."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        vazmin_rec = MagicMock()
        type(vazmin_rec).__name__ = "VAZMIN"
        vazmin_rec.vazao = 75.5

        usina_rec = MagicMock()
        usina_rec.codigo = 2

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmin_rec]

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert float(result.loc[2, "vazao_minima_historica"]) == pytest.approx(75.5)
        # Plant 1 must be unchanged (was 0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)

    def test_numcnj_nummaq_override(self, tmp_path) -> None:
        """NUMCNJ + NUMMAQ records update machine set counts."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

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

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert int(result.loc[1, "numero_conjuntos_maquinas"]) == 2
        assert int(result.loc[1, "maquinas_conjunto_2"]) == 3

    def test_volcota_override_warns_and_skips(self, tmp_path, caplog) -> None:
        """VOLCOTA records produce a warning and are skipped gracefully."""
        import logging

        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        volcota_rec = MagicMock()
        type(volcota_rec).__name__ = "VOLCOTA"

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volcota_rec]

        with caplog.at_level(logging.WARNING, logger="cobre_bridge.converters.hydro"):
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        # Values must be unchanged (dtype may differ due to float cast for safety).
        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert any("VOLCOTA" in msg for msg in caplog.messages)

    def test_unknown_plant_code_skipped(self, tmp_path, caplog) -> None:
        """Plant code not in cadastro: warning logged, no crash."""
        import logging

        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        usina_rec = MagicMock()
        usina_rec.codigo = 999  # not in cadastro

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = []

        with caplog.at_level(logging.WARNING, logger="cobre_bridge.converters.hydro"):
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert any("999" in msg for msg in caplog.messages)

    def test_temporal_records_skipped_in_permanent_pass(self, tmp_path) -> None:
        """Temporal override types are ignored in _apply_permanent_overrides."""
        import datetime

        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 999.0  # large value that should NOT be applied

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        # vazao_minima_historica must stay at the base value (0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _extract_temporal_overrides unit tests  (ticket-005)
# ---------------------------------------------------------------------------


class TestExtractTemporalOverrides:
    """Unit tests for ``_extract_temporal_overrides``."""

    def _modif_case(self, tmp_path, mock_modif):
        """Build a case whose MODIF reader is *mock_modif* (path set)."""
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_missing_modif_returns_empty(self, tmp_path) -> None:
        """No MODIF.DAT -> empty dict returned, no error."""
        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        result = _extract_temporal_overrides(make_case(tmp_path, modif=None), [1, 2])
        assert result == {}

    def test_extracts_vazmint_records(self, tmp_path) -> None:
        """VAZMINT record is extracted with correct month, year, value."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 50.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1, 2]
        )

        assert 1 in result
        assert result[1] == [
            {"type": "VAZMINT", "month": 1, "year": 2025, "value": 50.0}
        ]

    def test_filters_by_confhd_codes(self, tmp_path) -> None:
        """Plants not in confhd_codes are excluded from the result."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

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

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1, 2]
        )

        assert result == {}

    def test_preserves_file_order(self, tmp_path) -> None:
        """Multiple records for the same plant are returned in file order."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

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

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
        )

        assert len(result[1]) == 3
        assert result[1][0]["value"] == pytest.approx(50.0)
        assert result[1][1]["value"] == pytest.approx(60.0)
        assert result[1][2]["value"] == pytest.approx(55.0)

    def test_extracts_cfuga_records(self, tmp_path) -> None:
        """CFUGA record extracted with correct level value."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        cfuga_rec = MagicMock()
        type(cfuga_rec).__name__ = "CFUGA"
        cfuga_rec.data_inicio = datetime.datetime(2025, 6, 1)
        cfuga_rec.nivel = 75.4

        usina_rec = MagicMock()
        usina_rec.codigo = 2

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [2]
        )

        assert result[2] == [
            {"type": "CFUGA", "month": 6, "year": 2025, "value": pytest.approx(75.4)}
        ]

    def test_extracts_turbmint_turbmaxt_records(self, tmp_path) -> None:
        """TURBMINT and TURBMAXT records use turbinamento field."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

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

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
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
# _read_ghmin_per_stage unit tests
# ---------------------------------------------------------------------------


class TestReadGhminPerStage:
    """Unit tests for ``_read_ghmin_per_stage``.

    GHMIN values are time-varying and now live in
    ``hydro_bounds.parquet:min_generation_mw`` rather than the static
    ``hydros.json:generation.min_generation_mw``.  This helper expands
    each (plant, month, year) record into a per-(plant, stage_0based)
    mapping with step-function semantics and seasonal post-study
    repetition.
    """

    def _ghmin_case(self, tmp_path, mock_ghmin):
        """Build a case whose GHMIN reader is *mock_ghmin* (path set)."""
        return make_case(
            make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"),
            ghmin=mock_ghmin,
        )

    def test_missing_ghmin_returns_empty(self, tmp_path) -> None:
        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        result = _read_ghmin_per_stage(
            make_case(tmp_path, ghmin=None),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=24,
        )
        assert result == {}

    def test_step_function_persists_until_next_entry(self, tmp_path) -> None:
        """Sparse entries persist the last applied value forward."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        # Plant 1 at Sep 2024 = 100 MW, Dec 2024 = 80 MW.
        # Stages 0 (Sep) and 1 (Oct) and 2 (Nov) should all be 100.
        # Stage 3 (Dec) and onwards should be 80 within the study.
        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),
                    datetime.datetime(2024, 12, 1),
                ],
                "patamar": [0, 0],
                "geracao": [100.0, 80.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=12,
        )

        per_stage = result[1]
        assert per_stage[0] == pytest.approx(100.0)
        assert per_stage[1] == pytest.approx(100.0)
        assert per_stage[2] == pytest.approx(100.0)
        assert per_stage[3] == pytest.approx(80.0)
        assert per_stage[4] == pytest.approx(80.0)

    def test_post_study_uses_pos_seasonal_pattern(self, tmp_path) -> None:
        """POS year=9999 entries supply per-calendar-month values."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),  # study Sep 2024
                    datetime.datetime(9999, 9, 1),  # POS Sep
                    datetime.datetime(9999, 12, 1),  # POS Dec
                ],
                "patamar": [0, 0, 0],
                "geracao": [100.0, 150.0, 200.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,  # study ends Aug 2025
            total_stages=24,  # post-study: Sep 2025 – Aug 2026
        )

        per_stage = result[1]
        # Stage 12 = Sep 2025 → POS Sep = 150.
        assert per_stage[12] == pytest.approx(150.0)
        # Stage 15 = Dec 2025 → POS Dec = 200.
        assert per_stage[15] == pytest.approx(200.0)

    def test_patamar_nonzero_excluded(self, tmp_path) -> None:
        """Rows with patamar != 0 are excluded — only the all-blocks mean
        is meaningful at hydro_bounds' stage granularity."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),
                    datetime.datetime(2024, 9, 1),
                ],
                "patamar": [1, 2],
                "geracao": [50.0, 60.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=12,
        )

        assert result == {}


class TestConvertStorageBoundsPostStudy:
    """Per-quantity post-study extrapolation in convert_storage_bounds.

    VMINT/VMAXT repeat the last study year's seasonal pattern only when their
    dger ``sazonaliza_*`` flag is set; outflow (VAZMINT) and turbined
    (TURBMINT/TURBMAXT) have no flag and freeze the last study stage value.
    """

    def _run(self, tmp_path, overrides, *, vmaxt_flag=1, vmint_flag=1):
        from cobre_bridge.converters.hydro import convert_storage_bounds

        # start_month=1, 1 study year → study_months=12 (Jan–Dec); 1 post-study
        # year → stages 12–23 (Jan–Dec again).
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2024
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 1
        mock_dger.num_anos_pos_estudo = 1
        mock_dger.sazonaliza_vmaxt = vmaxt_flag
        mock_dger.sazonaliza_vmint = vmint_flag

        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [10],
                "usina_existente": ["EX"],
                "nome_usina": ["TEST"],
            }
        )
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        cadastro = pd.DataFrame(
            {"volume_minimo": [0.0], "volume_maximo": [100.0]}, index=[10]
        )
        id_map = MagicMock()
        id_map.hydro_id = lambda c: 0

        case = make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            dger=mock_dger,
            confhd=mock_confhd,
        )
        with (
            patch("cobre_bridge.converters.hydro.read_cadastro", return_value=cadastro),
            patch(
                "cobre_bridge.converters.hydro._extract_temporal_overrides",
                return_value={10: overrides},
            ),
            patch(
                "cobre_bridge.converters.hydro._read_ghmin_per_stage",
                return_value={},
            ),
        ):
            tbl = convert_storage_bounds(case, id_map)
        assert tbl is not None
        return tbl.to_pandas().set_index("stage_id")

    def test_outflow_freezes_post_study(self, tmp_path) -> None:
        """VAZMINT (no flag) freezes the post-study tail at last study Dec."""
        overrides = [
            {"type": "VAZMINT", "year": 2024, "month": 1, "value": 10.0},
            {"type": "VAZMINT", "year": 2024, "month": 12, "value": 120.0},
        ]
        df = self._run(tmp_path, overrides)
        # Study: Jan–Nov step-carry 10, Dec=120.
        assert df.loc[0, "min_outflow_m3s"] == pytest.approx(10.0)
        assert df.loc[11, "min_outflow_m3s"] == pytest.approx(120.0)
        # Post-study (12–23): all frozen at Dec=120, NOT the seasonal Jan=10.
        for s in range(12, 24):
            assert df.loc[s, "min_outflow_m3s"] == pytest.approx(120.0)

    def test_turbined_min_freezes_post_study(self, tmp_path) -> None:
        """TURBMINT (no flag) freezes the post-study tail."""
        overrides = [
            {"type": "TURBMINT", "year": 2024, "month": 1, "value": 5.0},
            {"type": "TURBMINT", "year": 2024, "month": 12, "value": 50.0},
        ]
        df = self._run(tmp_path, overrides)
        assert df.loc[11, "min_turbined_m3s"] == pytest.approx(50.0)
        for s in range(12, 24):
            assert df.loc[s, "min_turbined_m3s"] == pytest.approx(50.0)

    def test_vmaxt_seasonalizes_when_flag_set(self, tmp_path) -> None:
        """VMAXT with sazonaliza_vmaxt=1 repeats the seasonal pattern."""
        overrides = [
            {"type": "VMAXT", "year": 2024, "month": 1, "value": 50.0},
            {"type": "VMAXT", "year": 2024, "month": 12, "value": 80.0},
        ]
        df = self._run(tmp_path, overrides, vmaxt_flag=1)
        # vol_min=0, useful=100 → pct == hm3. Study Jan=50, Dec=80.
        assert df.loc[0, "max_storage_hm3"] == pytest.approx(50.0)
        assert df.loc[11, "max_storage_hm3"] == pytest.approx(80.0)
        # Post-study seasonal: stage 12 (Jan) keeps 50, stage 23 (Dec) keeps 80.
        assert df.loc[12, "max_storage_hm3"] == pytest.approx(50.0)
        assert df.loc[23, "max_storage_hm3"] == pytest.approx(80.0)

    def test_vmaxt_freezes_when_flag_clear(self, tmp_path) -> None:
        """VMAXT with sazonaliza_vmaxt=0 freezes the post-study tail."""
        overrides = [
            {"type": "VMAXT", "year": 2024, "month": 1, "value": 50.0},
            {"type": "VMAXT", "year": 2024, "month": 12, "value": 80.0},
        ]
        df = self._run(tmp_path, overrides, vmaxt_flag=0)
        # Post-study frozen at Dec=80, NOT seasonal Jan=50.
        assert df.loc[12, "max_storage_hm3"] == pytest.approx(80.0)
        assert df.loc[23, "max_storage_hm3"] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# convert_hydros integration tests for ticket-006
# ---------------------------------------------------------------------------


class TestConvertHydrosGhmin:
    """Integration tests for GHMIN handling.

    GHMIN values are now emitted per-stage in ``hydro_bounds.parquet``
    and the static ``hydros.json:generation.min_generation_mw`` is
    always 0.0.  These tests pin both halves of that contract.
    """

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def test_static_min_generation_is_zero_when_ghmin_present(self, tmp_path) -> None:
        """The static field is always zero — GHMIN goes elsewhere."""
        import datetime

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

        # convert_hydros never reads GHMIN (the per-stage values live in
        # hydro_bounds.parquet); the GHMIN mock is supplied to mirror a case
        # where the file is present, and the static field must still be 0.
        case = _hydro_case(tmp_path, ghmin=mock_ghmin_obj)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())

        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)

    def test_static_min_generation_is_zero_when_ghmin_absent(self, tmp_path) -> None:
        """With no GHMIN.DAT, static min_generation_mw is still 0."""
        case = _hydro_case(tmp_path)

        from cobre_bridge.converters.hydro import convert_hydros

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["min_generation_mw"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _per_stage_productivities: sazonaliza_cfuga_cmont behaviour
# ---------------------------------------------------------------------------


class TestPerStageProductivitiesSazonalCfugaCmont:
    """Verify that CFUGA/CMONT step-function carries vs. seasonal cycling.

    The source model's Dger ``sazonaliza_cfuga_cmont`` flag changes how CFUGA/CMONT
    overrides are extended beyond their last explicit
    entry: when ``0`` the last applied value carries forward
    indefinitely (pure step function); when ``1`` each calendar
    month picks up the value from the latest year that defined it
    and that seasonal pattern repeats month-by-month thereafter.
    """

    def _hreg(self) -> pd.Series:
        # Linear cota polynomial so we can read head off the coefficients.
        return _make_hreg(
            {
                "tipo_regulacao": "M",
                "volume_minimo": 0.0,
                "volume_maximo": 1000.0,
                "a0_volume_cota": 0.0,
                "a1_volume_cota": 1.0,
                "a2_volume_cota": 0.0,
                "a3_volume_cota": 0.0,
                "a4_volume_cota": 0.0,
                "canal_fuga_medio": 0.0,
                "tipo_perda": 1,
                "perdas": 0.0,
                "produtibilidade_especifica": 1.0,
                "volume_referencia": 500.0,
            }
        )

    def _dger_case(self, tmp_path, sazonaliza: int, num_anos_estudo: int = 3):
        """Build a case whose ``dger`` carries a controllable
        ``sazonaliza_cfuga_cmont``.

        ``num_anos_estudo`` defaults to 3 → study_months = 4 + 2*12 = 28 (start
        month 9), placing the seasonal-cycle assertions inside the study period.
        Lower it to push the post-study freeze boundary earlier.
        """
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2024
        mock_dger.mes_inicio_estudo = 9
        mock_dger.num_anos_estudo = num_anos_estudo
        mock_dger.sazonaliza_cfuga_cmont = sazonaliza
        return make_case(tmp_path, dger=mock_dger)

    def test_step_function_carries_when_sazonaliza_zero(self, tmp_path) -> None:
        from cobre_bridge.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2024, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=0),
            total_stages=24,
        )

        # tipo_regulacao = "M": v_65 = vmin + 0.65*(vmax-vmin) = 650.
        # cota(650) = 0 + 1*650 = 650. perdas=0 → prod = 1 * (650 - cfuga).
        # Stage 0 = Sep 2024 → CFUGA 5.0 → 645.
        # Stage 1 = Oct 2024 → CFUGA 10.0 → 640.
        # Stage 12 = Sep 2025 → no event → step-function carries 10.0 →
        # head = 640 (step function, NOT seasonal).
        assert vals[0] == pytest.approx(645.0)
        assert vals[1] == pytest.approx(640.0)
        assert vals[12] == pytest.approx(640.0)

    def test_seasonal_cycle_after_last_event_when_sazonaliza_one(
        self, tmp_path
    ) -> None:
        from cobre_bridge.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2024, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=1),
            total_stages=24,
        )

        # See test_step_function_carries_when_sazonaliza_zero for the
        # head computation: prod = 650 - cfuga at every stage.
        # Stage 0 = Sep 2024 = explicit 5.0 → 645.
        # Stage 1 = Oct 2024 = explicit 10.0 → 640.
        # Stage 12 = Sep 2025 → AFTER last_event_stage (1, Oct 2024) →
        # seasonal cfuga[9] = 5.0 → 645.
        # Stage 13 = Oct 2025 → seasonal cfuga[10] = 10.0 → 640.
        assert vals[0] == pytest.approx(645.0)
        assert vals[1] == pytest.approx(640.0)
        assert vals[12] == pytest.approx(645.0)
        assert vals[13] == pytest.approx(640.0)

    def test_seasonal_picks_latest_year_value_per_month(self, tmp_path) -> None:
        """When the same calendar month appears in multiple years, the
        latest year's value becomes the seasonal value."""
        from cobre_bridge.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2025, "month": 9, "value": 7.0},  # newer
            {"type": "CFUGA", "year": 2025, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=1),
            total_stages=36,
        )

        # Stage 24 = Sep 2026 → AFTER last_event_stage → seasonal cfuga[9]
        # = 7.0 (Sep 2025 won over Sep 2024) → prod = 650 - 7 = 643.
        assert vals[24] == pytest.approx(643.0)

    def test_no_overrides_returns_base(self, tmp_path) -> None:
        """Without any CFUGA/CMONT overrides the base value is returned
        unchanged at every stage."""
        from cobre_bridge.converters.hydro import _per_stage_productivities

        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=42.0,
            drop_overrides=[],
            case=make_case(tmp_path),
            total_stages=12,
        )
        assert vals == [42.0] * 12

    def test_post_study_continues_seasonal_cycle(self, tmp_path) -> None:
        """Post-study continues the seasonal CFUGA/CMONT cycle (no freeze).

        VOLREF_SAZ / CFUGA-CMONT seasonal patterns are re-applied every year,
        including post-study, when ``sazonaliza_cfuga_cmont == 1`` — only the
        no-flag bounds (outflow / turbined) freeze. With study_months = 4,
        stages 4+ are post-study and must keep cycling Sep=645 / Oct=640.
        """
        from cobre_bridge.converters.hydro import _per_stage_productivities

        overrides = [
            {"type": "CFUGA", "year": 2024, "month": 9, "value": 5.0},
            {"type": "CFUGA", "year": 2024, "month": 10, "value": 10.0},
        ]
        vals = _per_stage_productivities(
            self._hreg(),
            base_productivity=0.0,
            drop_overrides=overrides,
            case=self._dger_case(tmp_path, sazonaliza=1, num_anos_estudo=1),
            total_stages=24,
        )

        # Post-study Sep (stage 12) keeps the seasonal Sep value (645), and
        # post-study Oct (stage 13) keeps Oct (640) — NOT frozen at Dec's 640.
        assert vals[12] == pytest.approx(645.0)
        assert vals[13] == pytest.approx(640.0)


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
        "perdas": 5.0,  # percentage — divided by 100 in _compute_productivity
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
        """tipo_regulacao='M': poly evaluated at 65% useful storage (the source model
        ``produtibilidade_altura_65`` convention)."""
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
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # 65% of useful storage: v_65 = 100 + 0.65 * (1000 - 100) = 685.0
        # poly(685) = 300 + 0.1 * 685 = 368.5
        # net_drop = 368.5 - 250.0 = 118.5
        # adjusted_drop = 118.5 * (1 - 5.0/100) = 112.575
        # result = 0.009 * 112.575 = 1.013175
        v_65_height = 300.0 + 0.1 * (100.0 + 0.65 * (1000.0 - 100.0))
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (v_65_height - 250.0)
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
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # poly(500) = 300 + 0.1*500 = 350.0
        # net_drop = 350.0 - 250.0 = 100.0
        # adjusted_drop = 100.0 * (1 - 5.0/100) = 95.0
        # result = 0.009 * 95.0 = 0.855
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (350.0 - 250.0)
        result = _compute_productivity(hreg)
        assert result == pytest.approx(expected)

    def test_multiplicative_loss(self) -> None:
        """tipo_perda=1: adjusted_drop = net_drop * (1 - perdas/100)."""
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
                "perdas": 10.0,  # 10%
                "produtibilidade_especifica": 0.009,
            }
        )
        # net_drop = (300 + 50) - 250 = 100.0
        # adjusted = 100.0 * (1 - 10.0/100) = 90.0
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
        """tipo_regulacao='M' with vmin == vmax: v_65 collapses to that point."""
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
        # vmin == vmax: v_65 = 500.0; poly(500) = 350.0; net_drop = 100.0
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
        """With no overrides, M-plant ρ comes from poly evaluated at 65% storage."""
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
                "perdas": 5.0,  # 5%
                "produtibilidade_especifica": 0.009,
            }
        )
        # v_65 = 100 + 0.65 * 900 = 685; poly(685) = 368.5; net_drop = 118.5
        v_65_height = 300.0 + 0.1 * (100.0 + 0.65 * (1000.0 - 100.0))
        expected = 0.009 * (1.0 - 5.0 / 100.0) * (v_65_height - 250.0)
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


class TestEquivalentProductivity:
    """Unit tests for ``_equivalent_productivity`` (the source model PRODT)."""

    def test_linear_polynomial_uses_mean_head(self) -> None:
        """h(v)=a0+a1·v → mean head over [vmin,vmax] = a0 + a1·(vmin+vmax)/2."""
        from cobre_bridge.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({})  # a0=300, a1=0.1, vmin=100, vmax=1000, cfuga=250
        # mean head = 300 + 0.1·550 = 355 ; net = 105 ; ·0.95 loss ; ·0.009 pesp
        assert _equivalent_productivity(hreg) == pytest.approx(
            0.009 * (355.0 - 250.0) * 0.95
        )

    def test_differs_from_point_reference(self) -> None:
        """PRODT (mean over range) ≠ the 65%-volume point productivity."""
        from cobre_bridge.converters.hydro import (
            _compute_productivity,
            _equivalent_productivity,
        )

        # Non-linear forebay curve so the average over [vmin,vmax] differs from
        # the value at the 65% reference point.
        hreg = _make_hreg({"a2_volume_cota": 1e-4})
        assert _equivalent_productivity(hreg) != pytest.approx(
            _compute_productivity(hreg)
        )

    def test_run_of_river_uses_point_head(self) -> None:
        """Vmax == Vmin → head evaluated at Vmin (no integral)."""
        from cobre_bridge.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({"volume_minimo": 500.0, "volume_maximo": 500.0})
        # h(500) = 300 + 0.1·500 = 350 ; net 100 ; ·0.95 ; ·0.009
        assert _equivalent_productivity(hreg) == pytest.approx(
            0.009 * (350.0 - 250.0) * 0.95
        )

    def test_canal_fuga_override(self) -> None:
        from cobre_bridge.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({})
        assert _equivalent_productivity(
            hreg, canal_fuga_override=300.0
        ) == pytest.approx(0.009 * (355.0 - 300.0) * 0.95)

    def test_cmont_override_pins_forebay(self) -> None:
        """CMONT pins the upstream level → head = cmont − cfuga (no integral)."""
        from cobre_bridge.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({})
        assert _equivalent_productivity(hreg, cmont_override=400.0) == pytest.approx(
            0.009 * (400.0 - 250.0) * 0.95
        )

    def test_zero_polynomial_returns_zero(self) -> None:
        from cobre_bridge.converters.hydro import _equivalent_productivity

        hreg = _make_hreg({f"a{i}_volume_cota": 0.0 for i in range(5)})
        assert _equivalent_productivity(hreg) == 0.0


class TestProductivitySinMeans:
    """SIN-mean productivity aggregation over synthetic plant sets.

    These exercise the EX / FICT / out-of-cadastro filtering and the averaging
    wiring of the ``PROD_MEDIA_SIN`` helpers without depending on the
    (git-ignored) example case. Every expected value is *derived* from the same
    synthetic cadastro the function reads — never a hard-coded example snapshot.
    The per-plant productivity math itself is covered by
    :class:`TestEquivalentProductivity` / :class:`TestComputeProductivity`.
    """

    @staticmethod
    def _cadastro(rows: dict[int, dict]) -> pd.DataFrame:
        """Build a ``Hidr.cadastro``-shaped frame indexed by plant code."""
        return pd.DataFrame({code: _make_hreg(ov) for code, ov in rows.items()}).T

    @staticmethod
    def _confhd(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows)

    def _case(self, tmp_path, cadastro: pd.DataFrame, confhd: pd.DataFrame):
        """Build a case with hidr/confhd mocks. The ``_apply_permanent_overrides``
        identity patch is returned separately so callers can ``with`` on it."""
        hidr_obj = MagicMock()
        hidr_obj.cadastro = cadastro
        confhd_obj = MagicMock()
        confhd_obj.usinas = confhd
        case = make_case(tmp_path, hidr=hidr_obj, confhd=confhd_obj)
        overrides = patch(
            "cobre_bridge.converters.hydro._apply_permanent_overrides",
            new=lambda cadastro, case: cadastro,
        )
        return case, overrides

    def test_prodt_sin_mean_averages_existing_nonfict_in_cadastro(
        self, tmp_path: Path
    ) -> None:
        """Mean over EX, non-FICT plants present in cadastro; others excluded."""
        from cobre_bridge.converters.hydro import (
            _equivalent_productivity,
            compute_prodt_sin_mean,
        )

        cadastro = self._cadastro(
            {
                1: {"produtibilidade_especifica": 0.009, "canal_fuga_medio": 250.0},
                2: {"produtibilidade_especifica": 0.010, "canal_fuga_medio": 300.0},
                # code 3: ρ=0 sharing plant 1's posto → fictitious → excluded
                3: {"produtibilidade_especifica": 0.0},
                # code 4 is EX/non-fict but absent from cadastro → skipped
            }
        )
        confhd = self._confhd(
            [
                {
                    "codigo_usina": 1,
                    "nome_usina": "PLANT A",
                    "posto": 1,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 2,
                    "nome_usina": "PLANT B",
                    "posto": 2,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 3,
                    "nome_usina": "FICT TWIN",
                    "posto": 1,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 4,
                    "nome_usina": "PLANT D",
                    "posto": 4,
                    "usina_existente": "EX",
                },
                {
                    "codigo_usina": 5,
                    "nome_usina": "PLANT E",
                    "posto": 5,
                    "usina_existente": "NE",
                },
            ]
        )
        case, overrides = self._case(tmp_path, cadastro, confhd)
        with overrides:
            result = compute_prodt_sin_mean(case)

        expected = (
            _equivalent_productivity(cadastro.loc[1])
            + _equivalent_productivity(cadastro.loc[2])
        ) / 2
        assert result == pytest.approx(expected)

    def test_prodt_sin_mean_no_eligible_plants_returns_unit_fallback(
        self, tmp_path: Path
    ) -> None:
        """No EX/non-FICT plant in cadastro → fall back to 1.0, not divide-by-zero."""
        from cobre_bridge.converters.hydro import compute_prodt_sin_mean

        cadastro = self._cadastro({1: {}})
        confhd = self._confhd(
            [
                {
                    "codigo_usina": 9,
                    "nome_usina": "FICT. ONLY",
                    "usina_existente": "EX",
                },
                {"codigo_usina": 8, "nome_usina": "GONE", "usina_existente": "NE"},
            ]
        )
        case, overrides = self._case(tmp_path, cadastro, confhd)
        with overrides:
            assert compute_prodt_sin_mean(case) == 1.0

    def test_per_stage_prodt_flat_without_temporal_overrides(
        self, tmp_path: Path
    ) -> None:
        """No CFUGA/CMONT override → every stage equals the constant SIN mean."""
        from cobre_bridge.converters.hydro import (
            _equivalent_productivity,
            compute_per_stage_prodt_sin_mean,
        )

        cadastro = self._cadastro(
            {
                1: {"produtibilidade_especifica": 0.009},
                2: {"produtibilidade_especifica": 0.011, "canal_fuga_medio": 280.0},
            }
        )
        confhd = self._confhd(
            [
                {"codigo_usina": 1, "nome_usina": "PLANT A", "usina_existente": "EX"},
                {"codigo_usina": 2, "nome_usina": "PLANT B", "usina_existente": "EX"},
            ]
        )
        case, overrides = self._case(tmp_path, cadastro, confhd)
        with (
            overrides,
            patch("cobre_bridge.converters.hydro._total_study_stages", return_value=4),
            patch(
                "cobre_bridge.converters.hydro._extract_temporal_overrides",
                return_value={},
            ),
        ):
            per_stage = compute_per_stage_prodt_sin_mean(case)

        base = (
            _equivalent_productivity(cadastro.loc[1])
            + _equivalent_productivity(cadastro.loc[2])
        ) / 2
        assert len(per_stage) == 4
        assert all(v == pytest.approx(base) for v in per_stage)

    def test_per_stage_prodt_routes_overrides_and_averages_per_stage(
        self, tmp_path: Path
    ) -> None:
        """A plant carrying a CFUGA override drifts; the SIN mean tracks it per
        stage."""
        from cobre_bridge.converters.hydro import (
            _equivalent_productivity,
            compute_per_stage_prodt_sin_mean,
        )

        cadastro = self._cadastro({1: {}, 2: {"produtibilidade_especifica": 0.011}})
        confhd = self._confhd(
            [
                {"codigo_usina": 1, "nome_usina": "PLANT A", "usina_existente": "EX"},
                {"codigo_usina": 2, "nome_usina": "PLANT B", "usina_existente": "EX"},
            ]
        )

        def fake_series(hreg, base, drops, case, total_stages):
            # Plants with a routed CFUGA/CMONT override drift per stage; others flat.
            if drops:
                return [base, base * 1.02, base * 0.98][:total_stages]
            return [base] * total_stages

        case, overrides = self._case(tmp_path, cadastro, confhd)
        with (
            overrides,
            patch("cobre_bridge.converters.hydro._total_study_stages", return_value=3),
            patch(
                "cobre_bridge.converters.hydro._extract_temporal_overrides",
                return_value={1: [{"type": "CFUGA"}]},
            ),
            patch(
                "cobre_bridge.converters.hydro._per_stage_equivalent_productivities",
                side_effect=fake_series,
            ),
        ):
            per_stage = compute_per_stage_prodt_sin_mean(case)

        b1 = _equivalent_productivity(cadastro.loc[1])
        b2 = _equivalent_productivity(cadastro.loc[2])
        expected = [
            (b1 + b2) / 2,
            (b1 * 1.02 + b2) / 2,
            (b1 * 0.98 + b2) / 2,
        ]
        assert per_stage == pytest.approx(expected)

    def test_max_prodtacum_sin_picks_cascade_max(self, tmp_path: Path) -> None:
        """Accumulated productivity peaks at the head of the longest cascade."""
        from cobre_bridge.converters.constraints import compute_max_prodtacum_sin
        from cobre_bridge.converters.hydro import _compute_productivity

        # Cascade A(1) → B(2) → terminal; C(3) standalone.
        cadastro = self._cadastro({1: {}, 2: {}, 3: {}})
        confhd = self._confhd(
            [
                {
                    "codigo_usina": 1,
                    "nome_usina": "A",
                    "usina_existente": "EX",
                    "codigo_usina_jusante": 2,
                },
                {
                    "codigo_usina": 2,
                    "nome_usina": "B",
                    "usina_existente": "EX",
                    "codigo_usina_jusante": 0,
                },
                {
                    "codigo_usina": 3,
                    "nome_usina": "C",
                    "usina_existente": "EX",
                    "codigo_usina_jusante": 0,
                },
            ]
        )
        hidr_obj = MagicMock()
        hidr_obj.cadastro = cadastro
        confhd_obj = MagicMock()
        confhd_obj.usinas = confhd
        case = make_case(tmp_path, hidr=hidr_obj, confhd=confhd_obj)
        with patch(
            "cobre_bridge.converters.constraints._apply_permanent_overrides",
            new=lambda cadastro, case: cadastro,
        ):
            result = compute_max_prodtacum_sin(case)

        def own(code: int) -> float:
            hreg = cadastro.loc[code]
            useful = float(hreg["volume_maximo"]) - float(hreg["volume_minimo"])
            return _compute_productivity(hreg, useful_volume_override=useful)

        acc_a = own(1) + own(2)  # A accumulates B downstream
        assert result == pytest.approx(max(acc_a, own(2), own(3)))

    def test_max_prodtacum_sin_returns_none_on_read_error(self, tmp_path: Path) -> None:
        """Unreadable the source model inputs → None (soft fallback for mocked
        pipelines)."""
        from cobre_bridge.converters.constraints import compute_max_prodtacum_sin

        case = make_case(tmp_path)
        # Drop the conftest default hidr so accessing ``case.hidr`` triggers the
        # patched ``Hidr.read`` (this test exercises the read-error fallback).
        case.__dict__.pop("hidr", None)
        with patch("cobre_bridge.case.Hidr") as mh:
            mh.read.side_effect = OSError("no file")
            assert compute_max_prodtacum_sin(case) is None


class TestConvertProductionModels:
    """Unit tests for ``convert_production_models``."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def _base_case(
        self,
        tmp_path: Path,
        *,
        modif=None,
        ano_inicio: int = 2025,
        mes_inicio: int = 1,
        num_anos: int = 5,
    ):
        """Build a case with hidr/confhd/dger mocks (and optional modif).

        Pass *modif* as a mock reader object; the matching path is set so the
        ``case.files.modif`` guard sees a present file.
        """
        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()

        mock_dger = _make_prod_model_dger_mock(
            ano_inicio=ano_inicio,
            mes_inicio=mes_inicio,
            num_anos=num_anos,
            num_anos_pos=0,
        )

        parsed: dict = {"hidr": mock_hidr, "confhd": mock_confhd, "dger": mock_dger}
        if modif is not None:
            parsed["modif"] = modif
            files = make_nw_files(tmp_path, modif=tmp_path / "modif.dat")
        else:
            files = make_nw_files(tmp_path)
        return make_case(files, **parsed)

    def test_returns_all_hydros_when_no_modif(self, tmp_path: Path) -> None:
        """No MODIF.DAT: every hydro still gets a single-range entry.

        Cobre HEAD requires the productivity for every hydro to live in
        ``hydro_production_models.json`` (it was removed from hydros.json
        generation block), so we always emit an entry per plant.
        """
        case = self._base_case(tmp_path)

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())
        assert result is not None
        models = result["production_models"]
        assert len(models) == 2  # USINA_A and USINA_B
        for model in models:
            assert model["selection_mode"] == "stage_ranges"
            ranges = model["stage_ranges"]
            assert len(ranges) == 1
            assert ranges[0]["start_stage_id"] == 0
            assert ranges[0]["end_stage_id"] is None
            # Productivity now lives in hydro_energy_productivity.parquet,
            # not in the JSON stage_range entries.
            assert "productivity_mw_per_m3s" not in ranges[0]

    def test_returns_all_hydros_when_no_cfuga_cmont(self, tmp_path: Path) -> None:
        """MODIF.DAT present but only VAZMINT overrides -> per-hydro entries with single
        range."""
        import datetime

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 3, 1)
        vazmint_rec.vazao = 50.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        case = self._base_case(tmp_path, modif=mock_modif)

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())
        assert result is not None
        models = result["production_models"]
        # Both hydros still get an entry, each with one stage range
        # covering the whole horizon (no CFUGA/CMONT temporal overrides).
        assert len(models) == 2
        for model in models:
            ranges = model["stage_ranges"]
            assert len(ranges) == 1
            assert ranges[0]["end_stage_id"] is None

    def test_single_cfuga_override_two_ranges(self, tmp_path: Path) -> None:
        """One CFUGA override at stage 3 -> two stage_ranges (base then overridden)."""
        cfuga_rec = _make_cfuga_rec(month=4, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        # Study: start Jan 2025, 5 years -> 60 stages total.
        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        assert "production_models" in result
        models = result["production_models"]
        # JSON now carries only model selection — productivity moved to
        # hydro_energy_productivity.parquet. Both USINA_A and USINA_B emit
        # one model-only stage_range entry; per-stage variation for USINA_A's
        # CFUGA override is asserted via TestConvertHydroEnergyProductivity.
        assert len(models) == 2

        model_a = next(m for m in models if m["hydro_id"] == 0)
        assert model_a["selection_mode"] == "stage_ranges"
        ranges = model_a["stage_ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        assert ranges[0]["model"] == "constant_productivity"
        assert "productivity_mw_per_m3s" not in ranges[0]

    def test_cmont_override_bypasses_polynomial(self, tmp_path: Path) -> None:
        """CMONT override at stage 0 -> single stage_range using cmont as height."""
        cmont_rec = _make_cmont_rec(month=1, year=2025, nivel=400.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cmont_rec]

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        models = result["production_models"]
        # JSON has model-only entries — productivity moved to the parquet.
        assert len(models) == 2
        model_a = next(m for m in models if m["hydro_id"] == 0)
        ranges = model_a["stage_ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        assert "productivity_mw_per_m3s" not in ranges[0]

    def test_multiple_overrides_three_ranges(self, tmp_path: Path) -> None:
        """Two CFUGA overrides -> three stage_ranges."""
        recs = [
            _make_cfuga_rec(month=6, year=2025, nivel=55.0),  # stage 5
            _make_cfuga_rec(month=1, year=2026, nivel=65.0),  # stage 12
        ]
        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = recs

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        # JSON now has a single model-only stage_range per hydro; the multiple
        # CFUGA breakpoints surface in hydro_energy_productivity.parquet.
        ranges = result["production_models"][0]["stage_ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start_stage_id"] == 0
        assert ranges[0]["end_stage_id"] is None
        assert "productivity_mw_per_m3s" not in ranges[0]

    def test_output_sorted_by_hydro_id(self, tmp_path: Path) -> None:
        """production_models list is sorted ascending by hydro_id."""
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

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from cobre_bridge.converters.hydro import convert_production_models

        result = convert_production_models(case, self._make_id_map())

        assert result is not None
        ids = [m["hydro_id"] for m in result["production_models"]]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Hydro energy productivity (parquet) conversion
# ---------------------------------------------------------------------------


class TestConvertHydroEnergyProductivity:
    """Per-(hydro, stage) ρ_eq override parquet for the cobre productivity-resolution
    contract."""

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[],
        )

    def _base_case(
        self,
        tmp_path: Path,
        *,
        modif=None,
        volref_volumes: pd.DataFrame | None = None,
        ano_inicio: int = 2025,
        mes_inicio: int = 1,
        num_anos: int = 5,
    ):
        """Build a case with hidr/confhd/dger mocks plus optional modif/volref.

        *modif* is a mock reader object; *volref_volumes* is a DataFrame for the
        VolrefSaz reader. Matching file paths are set so the ``case.files.X``
        guards see present files.
        """
        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()

        mock_dger = _make_prod_model_dger_mock(
            ano_inicio=ano_inicio,
            mes_inicio=mes_inicio,
            num_anos=num_anos,
            num_anos_pos=0,
        )

        parsed: dict = {"hidr": mock_hidr, "confhd": mock_confhd, "dger": mock_dger}
        file_overrides: dict = {}
        if modif is not None:
            parsed["modif"] = modif
            file_overrides["modif"] = tmp_path / "modif.dat"
        if volref_volumes is not None:
            mock_volref = MagicMock()
            mock_volref.volumes = volref_volumes
            parsed["volref_saz"] = mock_volref
            file_overrides["volref_saz"] = tmp_path / "volref_saz.dat"
        return make_case(make_nw_files(tmp_path, **file_overrides), **parsed)

    def test_null_stage_row_per_hydro_when_no_overrides(self, tmp_path: Path) -> None:
        """Without CFUGA/CMONT: one NULL-stage_id row per hydro with base
        productivity."""
        case = self._base_case(tmp_path)

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())

        assert table.num_rows == 2
        assert table.column_names[:2] == ["hydro_id", "stage_id"]
        stage_ids = table["stage_id"].to_pylist()
        assert stage_ids == [None, None]
        prods = table["equivalent_productivity_mw_per_m3s"].to_pylist()
        # USINA_A: v_65=685, poly(685)=368.5, net_drop=318.5, ρ=0.9 * 318.5
        assert prods[0] == pytest.approx(0.9 * 318.5)

    def test_per_stage_rows_for_cfuga_override(self, tmp_path: Path) -> None:
        """CFUGA at stage 3 → per-stage rows for the full horizon; stages [0..2] use
        base."""
        cfuga_rec = _make_cfuga_rec(month=4, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        case = self._base_case(
            tmp_path, modif=mock_modif, ano_inicio=2025, mes_inicio=1, num_anos=5
        )

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())

        # USINA_A (hydro_id=0): 60 per-stage rows. USINA_B (hydro_id=1): 1 NULL row.
        rows = table.to_pylist()
        a_rows = [r for r in rows if r["hydro_id"] == 0]
        b_rows = [r for r in rows if r["hydro_id"] == 1]
        assert len(a_rows) == 60  # 5 years * 12 months
        assert len(b_rows) == 1
        assert b_rows[0]["stage_id"] is None

        # USINA_A is tipo_regulacao="M", so _compute_productivity evaluates the
        # cota polynomial at v_65 = vmin + 0.65·useful = 100 + 0.65·900 = 685.
        # h(685) = 300 + 0.1·685 = 368.5.  CFUGA overrides canal_fuga from
        # cadastro's 50.0 to 60.0 starting at stage 3.
        base = 0.9 * (368.5 - 50.0)
        override = 0.9 * (368.5 - 60.0)
        # Stages 0..2 = base, stages 3..59 = override.
        a_by_stage = {
            r["stage_id"]: r["equivalent_productivity_mw_per_m3s"] for r in a_rows
        }
        assert a_by_stage[0] == pytest.approx(base)
        assert a_by_stage[2] == pytest.approx(base)
        assert a_by_stage[3] == pytest.approx(override)
        assert a_by_stage[59] == pytest.approx(override)

    def test_seasonal_volref_emits_per_stage_rows(self, tmp_path: Path) -> None:
        """volref_saz row with non-zero values → per-stage ρ computed at
        ``vol_min + volref[month]`` for that calendar month."""
        # USINA_A: seasonal row (every month has its own useful volume).
        # USINA_B: not present in the file → falls back to altura_65 default.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                # Useful volumes (hm³ above vol_min=100): 100, 200, ..., 1200
                "valor": [float(100 * m) for m in range(1, 13)],
            }
        )

        case = self._base_case(
            tmp_path,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = table.to_pylist()
        a_rows = [r for r in rows if r["hydro_id"] == 0]
        b_rows = [r for r in rows if r["hydro_id"] == 1]

        # USINA_A: 12 per-stage rows (1 year * 12 months); USINA_B: 1 null row.
        assert len(a_rows) == 12
        assert len(b_rows) == 1
        assert b_rows[0]["stage_id"] is None

        # USINA_A stage 0 = calendar month 1, useful=100, V=200, h(V)=320, drop=270,
        # ρ=243.
        by_stage = {
            r["stage_id"]: r["equivalent_productivity_mw_per_m3s"] for r in a_rows
        }
        # ρ_esp=0.9, cf=50, h(v)=300+0.1v
        for stage in range(12):
            useful = 100.0 * (stage + 1)
            expected = 0.9 * ((300.0 + 0.1 * (100.0 + useful)) - 50.0)
            assert by_stage[stage] == pytest.approx(expected)

    def test_seasonal_volref_all_zero_row_falls_back_to_legacy(
        self, tmp_path: Path
    ) -> None:
        """All-zero volref_saz row is the source model's "no seasonal reference"
        sentinel:
        emit a single null-stage row with the legacy altura_65 productivity."""
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12 + [2] * 12,
                "nome_usina": ["USINA_A"] * 12 + ["USINA_B"] * 12,
                "mes": list(range(1, 13)) * 2,
                # USINA_A all zeros (sentinel); USINA_B all zeros (sentinel).
                "valor": [0.0] * 24,
            }
        )

        case = self._base_case(
            tmp_path,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = table.to_pylist()
        assert len(rows) == 2
        for r in rows:
            assert r["stage_id"] is None
        # Both fall back to altura_65 legacy default.
        a_row = next(r for r in rows if r["hydro_id"] == 0)
        assert a_row["equivalent_productivity_mw_per_m3s"] == pytest.approx(0.9 * 318.5)

    def test_seasonal_volref_zero_month_inside_nonzero_row_uses_vmin(
        self, tmp_path: Path
    ) -> None:
        """A zero entry within a row that has some non-zero months means
        "operate at vol_min" for that month — distinct from the legacy
        fallback used for an entirely all-zero row."""
        # USINA_A: month 1 = 0 (V=vmin); month 2 = 500 (V=vmin+500); rest = 0.
        # Plant is kept because at least one month is non-zero.
        valor = [0.0] * 12
        valor[1] = 500.0  # month 2
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": valor,
            }
        )

        case = self._base_case(
            tmp_path,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = table.to_pylist()
        a_rows = sorted(
            (r for r in rows if r["hydro_id"] == 0),
            key=lambda r: r["stage_id"],
        )
        # Stage 0 = month 1 (volref=0) → V=vmin=100, h=310, drop=260, ρ=234.
        assert a_rows[0]["equivalent_productivity_mw_per_m3s"] == pytest.approx(
            0.9 * (300.0 + 0.1 * 100.0 - 50.0)
        )
        # Stage 1 = month 2 (volref=500) → V=600, h=360, drop=310, ρ=279.
        assert a_rows[1]["equivalent_productivity_mw_per_m3s"] == pytest.approx(
            0.9 * (300.0 + 0.1 * 600.0 - 50.0)
        )

    def test_seasonal_volref_combined_with_cfuga_override(self, tmp_path: Path) -> None:
        """Seasonal volref and CFUGA must compose: each stage uses the
        month's reference volume AND the active canal_fuga override."""
        # CFUGA effective from month 6 (stage 5) onward — canal_fuga 50 → 60.
        cfuga_rec = _make_cfuga_rec(month=6, year=2025, nivel=60.0)
        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        # Seasonal volref: every month useful=200 → V=300, h=330.
        volref_df = pd.DataFrame(
            {
                "codigo_usina": [1] * 12,
                "nome_usina": ["USINA_A"] * 12,
                "mes": list(range(1, 13)),
                "valor": [200.0] * 12,
            }
        )

        case = self._base_case(
            tmp_path,
            modif=mock_modif,
            volref_volumes=volref_df,
            ano_inicio=2025,
            mes_inicio=1,
            num_anos=1,
        )

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())
        rows = [r for r in table.to_pylist() if r["hydro_id"] == 0]
        by_stage = {
            r["stage_id"]: r["equivalent_productivity_mw_per_m3s"] for r in rows
        }
        # Stages 0..4: cf=50, V=300, h=330, drop=280, ρ=252.
        pre = 0.9 * (300.0 + 0.1 * 300.0 - 50.0)
        # Stages 5..11: cf=60, drop=270, ρ=243.
        post = 0.9 * (300.0 + 0.1 * 300.0 - 60.0)
        assert by_stage[0] == pytest.approx(pre)
        assert by_stage[4] == pytest.approx(pre)
        assert by_stage[5] == pytest.approx(post)
        assert by_stage[11] == pytest.approx(post)

    def test_other_override_columns_are_null(self, tmp_path: Path) -> None:
        """reference_outflow_m3s / ρ_esp columns remain NULL.

        The retired ``reference_volume_hm3`` column must no longer be emitted;
        V_ref now lives in ``hydro_production_models.json``.
        """
        case = self._base_case(tmp_path)

        from cobre_bridge.converters.hydro import convert_hydro_energy_productivity

        table = convert_hydro_energy_productivity(case, self._make_id_map())

        assert "reference_volume_hm3" not in table.column_names
        for col in (
            "reference_outflow_m3s",
            "specific_productivity_mw_per_m3s_per_m",
        ):
            assert all(v is None for v in table[col].to_pylist())


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


def _thermal_readers():
    conft = MagicMock()
    conft.usinas = _make_conft_df()
    clast = MagicMock()
    clast.usinas = _make_clast_df()
    clast.modificacoes = None
    term = MagicMock()
    term.usinas = _make_term_df()
    return conft, clast, term


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

    def test_returns_buses_key(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        assert "buses" in result

    def test_bus_count_includes_fictitious(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        # 3 subsystems total: 1, 2, 99.
        assert len(result["buses"]) == 3

    def test_bus_ids_are_zero_based_sorted(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        ids = [b["id"] for b in result["buses"]]
        assert ids == sorted(ids)
        assert ids[0] == 0

    def test_bus_has_deficit_segments(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
        for b in result["buses"]:
            assert "deficit_segments" in b
            assert isinstance(b["deficit_segments"], list)
            assert len(b["deficit_segments"]) == 2  # 2 patamares

    def test_last_deficit_segment_depth_is_null(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_buses

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_buses(case, self._make_id_map())
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

    def _make_case(self, tmp_path):
        dger = MagicMock()
        dger.mes_inicio_estudo = 1
        dger.ano_inicio_estudo = 2023
        return make_case(tmp_path, sistema=_make_sistema_mock(), dger=dger)

    def test_returns_lines_key(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        assert "lines" in result

    def test_line_count_three_pairs(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        assert len(result["lines"]) == 3

    def test_line_capacity_structure(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        for line in result["lines"]:
            assert "capacity" in line
            assert "direct_mw" in line["capacity"]
            assert "reverse_mw" in line["capacity"]
            assert "source_bus_id" in line
            assert "target_bus_id" in line

    def test_line_ids_sequential(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        ids = [ln["id"] for ln in result["lines"]]
        assert ids == list(range(len(ids)))

    def test_first_month_used_for_capacity(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_lines

        result = convert_lines(self._make_case(tmp_path), self._make_id_map())
        line_12 = next(
            ln
            for ln in result["lines"]
            if ln["source_bus_id"] == 0 and ln["target_bus_id"] == 1
        )
        assert line_12["capacity"]["direct_mw"] == pytest.approx(3000.0)
        assert line_12["capacity"]["reverse_mw"] == pytest.approx(2500.0)

    def test_fictitious_lines_get_half_exchange_cost(self, tmp_path) -> None:
        from cobre_bridge.converters.network import (
            _PINT,
            _PINT_FICTITIOUS_DISCOUNT,
            convert_lines,
        )

        id_map = NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[],
            thermal_codes=[],
        )
        result = convert_lines(self._make_case(tmp_path), id_map)

        fict_bus_id = id_map.bus_id(99)
        expected = _PINT * _PINT_FICTITIOUS_DISCOUNT
        for ln in result["lines"]:
            touches_fict = (
                ln["source_bus_id"] == fict_bus_id or ln["target_bus_id"] == fict_bus_id
            )
            if touches_fict:
                assert ln["exchange_cost"] == pytest.approx(expected)
            else:
                assert "exchange_cost" not in ln


def _make_sistema_mock() -> MagicMock:
    """Build the ``Sistema`` reader mock shared by the network tests."""
    mock_sistema = MagicMock()
    mock_sistema.custo_deficit = _make_deficit_df(n_patamares=2)
    mock_sistema.limites_intercambio = _make_intercambio_df()
    return mock_sistema


# ---------------------------------------------------------------------------
# Penalties conversion
# ---------------------------------------------------------------------------


class TestConvertPenalties:
    def test_returns_required_keys(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_penalties

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_penalties(
            case,
            {
                "hydros": [
                    {
                        "generation": {"productivity_mw_per_m3s": 1.0},
                        "reservoir": {
                            "max_storage_hm3": 1000.0,
                            "min_storage_hm3": 100.0,
                        },
                    }
                ]
            },
        )
        for key in ("bus", "hydro", "line", "non_controllable_source"):
            assert key in result

    def test_bus_deficit_uses_first_subsystem_first_tier(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_penalties

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_penalties(
            case,
            {
                "hydros": [
                    {
                        "generation": {"productivity_mw_per_m3s": 1.0},
                        "reservoir": {
                            "max_storage_hm3": 1000.0,
                            "min_storage_hm3": 100.0,
                        },
                    }
                ]
            },
        )
        # First subsystem=1, patamar=1: custo = 500.0*1 = 500.0
        seg = result["bus"]["deficit_segments"][0]
        assert seg["cost"] == pytest.approx(500.0)

    def test_hydro_has_all_penalty_fields(self, tmp_path) -> None:
        from cobre_bridge.converters.network import convert_penalties

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        result = convert_penalties(
            case,
            {
                "hydros": [
                    {
                        "generation": {"productivity_mw_per_m3s": 1.0},
                        "reservoir": {
                            "max_storage_hm3": 1000.0,
                            "min_storage_hm3": 100.0,
                        },
                    }
                ]
            },
        )
        required = {
            "spillage_cost",
            "turbined_cost",
            "diversion_cost",
            "storage_violation_below_cost",
            "filling_target_violation_cost",
            "turbined_violation_below_cost",
            "outflow_violation_below_cost",
            "outflow_violation_above_cost",
            "generation_violation_below_cost",
            "evaporation_violation_cost",
            "water_withdrawal_violation_cost",
            "inflow_nonnegativity_cost",
        }
        assert required == set(result["hydro"].keys())


class TestHydroPenaltyCosts:
    """The pure ρ-scaling helper shared by the base and per-stage paths."""

    def test_flow_penalties_scale_linearly_with_rho_avg(self) -> None:
        from cobre_bridge.converters.network import _PEVERT, _hydro_penalty_costs

        single = _hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=100.0
        )
        double = _hydro_penalty_costs(
            rho_avg=2.0, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=100.0
        )
        # spillage_cost = _PEVERT * rho_avg → doubles with rho_avg.
        assert single["spillage_cost"] == pytest.approx(_PEVERT * 1.0)
        assert double["spillage_cost"] == pytest.approx(2.0 * single["spillage_cost"])
        # water_withdrawal uses rho_max_acum (held fixed) → unchanged.
        assert double["water_withdrawal_violation_cost"] == pytest.approx(
            single["water_withdrawal_violation_cost"]
        )

    def test_water_withdrawal_scales_with_rho_max_acum(self) -> None:
        from cobre_bridge.converters.network import _hydro_penalty_costs

        low = _hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=1.0, penalid_costs={}, max_deficit_cost=100.0
        )
        high = _hydro_penalty_costs(
            rho_avg=1.0, rho_max_acum=3.0, penalid_costs={}, max_deficit_cost=100.0
        )
        assert high["water_withdrawal_violation_cost"] == pytest.approx(
            3.0 * low["water_withdrawal_violation_cost"]
        )
        # spillage (rho_avg only) is unaffected by rho_max_acum.
        assert high["spillage_cost"] == pytest.approx(low["spillage_cost"])


class TestConvertHydroPenaltyOverrides:
    """Per-stage, SIN-uniform hydro penalty override parquet."""

    @patch("cobre_bridge.converters.network._read_penalid_costs", return_value={})
    def test_sin_uniform_sparse_per_stage(self, _mock_penalid, tmp_path) -> None:
        from cobre_bridge.converters.network import (
            _PEVERT,
            _hydro_penalty_costs,
            convert_hydro_penalty_overrides,
        )

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        # Build the base via the same helper the override diffs against, with
        # the mocked max_deficit_cost (max custo = 500*2 = 1000). Stage 1 then
        # uses exactly the base (ρ_avg=0.6, ρ_max_acum=2.0) → no override.
        base = _hydro_penalty_costs(
            rho_avg=0.6, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=1000.0
        )
        table = convert_hydro_penalty_overrides(
            case,
            hydro_ids=[0, 1],
            base_hydro_penalties=base,
            per_stage_rho_avg=[0.5, 0.6, 0.55],
            per_stage_rho_max_acum=[2.0, 2.0, 2.0],
        )
        assert table is not None
        df = table.to_pandas()

        # Required key columns + only ρ-scaled columns that differ are present.
        assert {"hydro_id", "stage_id"}.issubset(df.columns)
        assert "generation_violation_below_cost" not in df.columns
        assert "filling_target_violation_cost" not in df.columns

        # Stage 1 matches the base exactly → no rows emitted for it (sparse).
        assert sorted(df["stage_id"].unique().tolist()) == [0, 2]

        # SIN-uniform: both hydros share one value per stage.
        s0 = df[df["stage_id"] == 0]
        assert s0["hydro_id"].tolist() == [0, 1]
        assert s0["spillage_cost"].nunique() == 1
        assert s0["spillage_cost"].iloc[0] == pytest.approx(_PEVERT * 0.5)

        # Output obeys the (hydro_id, stage_id) ordering contract.
        ordered = df.sort_values(["hydro_id", "stage_id"]).reset_index(drop=True)
        assert df.reset_index(drop=True).equals(ordered)

    @patch("cobre_bridge.converters.network._read_penalid_costs", return_value={})
    def test_returns_none_when_no_stage_differs(self, _mock_penalid, tmp_path) -> None:
        from cobre_bridge.converters.network import (
            _hydro_penalty_costs,
            convert_hydro_penalty_overrides,
        )

        case = make_case(tmp_path, sistema=_make_sistema_mock())
        # max_deficit_cost from the mocked deficit df (max custo = 500*2 = 1000).
        base = _hydro_penalty_costs(
            rho_avg=0.6, rho_max_acum=2.0, penalid_costs={}, max_deficit_cost=1000.0
        )
        # Every stage uses exactly the base ρ → fully sparse → None.
        table = convert_hydro_penalty_overrides(
            case,
            hydro_ids=[0, 1],
            base_hydro_penalties=base,
            per_stage_rho_avg=[0.6, 0.6],
            per_stage_rho_max_acum=[2.0, 2.0],
        )
        assert table is None


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

    def test_returns_storage_and_filling_storage(self, tmp_path) -> None:
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        assert "storage" in result
        assert "filling_storage" in result

    def test_storage_values_converted_from_percentage(self, tmp_path) -> None:
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        # New formula: (pct / 100) * (vol_max - vol_min) + vol_min
        # USINA_A: pct=50%, vol_min=100, vol_max=1000
        #   -> (0.50) * (1000 - 100) + 100 = 450 + 100 = 550 hm3.
        # USINA_B: pct=75%, vol_min=50, vol_max=500
        #   -> (0.75) * (500 - 50) + 50 = 337.5 + 50 = 387.5 hm3.
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        assert storage[0] == pytest.approx(550.0)
        assert storage[1] == pytest.approx(387.5)

    def test_storage_sorted_by_hydro_id(self, tmp_path) -> None:
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        ids = [s["hydro_id"] for s in result["storage"]]
        assert ids == sorted(ids)

    def test_out_of_range_percentage_clamped(self, tmp_path) -> None:
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # Should not raise; pct is clamped to 100.
        result = convert_initial_conditions(
            _ic_case(tmp_path, pct_b=120.0), self._make_id_map()
        )
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # pct clamped to 100 -> vol_max=500 -> 500.0 hm3.
        assert storage[1] == pytest.approx(500.0)

    def test_filling_storage_is_empty(self, tmp_path) -> None:
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        result = convert_initial_conditions(_ic_case(tmp_path), self._make_id_map())
        assert result["filling_storage"] == []

    def test_storage_uses_volmin_adjusted_min(self, tmp_path) -> None:
        """Initial-% is of the operational useful (vol_max − VOLMIN), not raw.

        A modif.dat VOLMIN override raises the operational minimum; the source model
        takes ``volume_inicial_percentual`` of the VOLMIN-adjusted useful range
        (verified against pmo.dat "VOLUME ARMAZENADO INICIAL" for I. SOLTEIRA). The
        initial storage must use the same min the bounds converter uses. Regression for
        the I. Solteira initial-storage bug.
        """
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # USINA_A (code 1): raw vol_min=100, vol_max=1000, pct=50%. A VOLMIN=400
        # override makes the operational useful 1000−400 = 600.
        volmin_rec = MagicMock()
        type(volmin_rec).__name__ = "VOLMIN"
        volmin_rec.volume = 400.0
        usina_rec = MagicMock()
        usina_rec.codigo = 1
        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volmin_rec]

        mock_hidr = MagicMock()
        mock_hidr.cadastro = _make_hidr_cadastro()
        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        case = make_case(tmp_path, hidr=mock_hidr, confhd=mock_confhd, modif=mock_modif)

        result = convert_initial_conditions(case, self._make_id_map())
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # operational base: 0.50 * (1000 − 400) + 400 = 700 (NOT the raw-min 550).
        assert storage[0] == pytest.approx(700.0)

    def test_run_of_river_S_initial_anchored_to_vmin(self, tmp_path) -> None:
        """``tipo_regulacao='S'`` initial storage is pinned to Vmin.

        The bounds converter collapses 'S' (fio-d'água) storage to Vmin; the initial
        condition must match so it stays inside the collapsed [min,max] range (the
        source model keeps ITAIPU at VARMPUH 0% = Vmin).  The
        ``volume_inicial_percentual`` (50% here) is ignored for 'S' plants.
        """
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        cadastro = _make_hidr_cadastro()
        cadastro.loc[1, "tipo_regulacao"] = "S"  # USINA_A: Vmin 100, Vmax 1000
        mock_hidr = MagicMock()
        mock_hidr.cadastro = cadastro
        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df()
        case = make_case(tmp_path, hidr=mock_hidr, confhd=mock_confhd)

        result = convert_initial_conditions(case, self._make_id_map())
        storage = {s["hydro_id"]: s["value_hm3"] for s in result["storage"]}
        # 'S' plant anchored to Vmin (100), NOT 0.50*(1000−100)+100 = 550.
        assert storage[0] == pytest.approx(100.0)


def _ic_case(tmp_path, pct_b: float = 75.0):
    """Build a NewaveCase with hidr/confhd readers pre-cached for IC tests."""
    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro()

    df = _make_confhd_df().copy()
    df.loc[df["codigo_usina"] == 2, "volume_inicial_percentual"] = pct_b
    mock_confhd = MagicMock()
    mock_confhd.usinas = df

    return make_case(tmp_path, hidr=mock_hidr, confhd=mock_confhd)


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


class TestAnticipatedCommitmentSeeding:
    """``convert_initial_conditions`` writes real adterm MW, clamped to bounds.

    Cobre (>= 0.7.0) honours non-zero pre-horizon seeds, so the committed MW is
    passed through (no longer zeroed); only out-of-bounds values are clamped.
    """

    def _id_map(self) -> NewaveIdMap:
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[1, 2],
            thermal_codes=[86],
        )

    @patch("cobre_bridge.converters.initial_conditions.thermal_generation_bounds")
    @patch("cobre_bridge.converters.initial_conditions.read_anticipated_dispatch")
    def test_in_range_values_pass_through(
        self, mock_read, mock_bounds, tmp_path, caplog
    ) -> None:
        from cobre_bridge.converters.anticipated import AnticipatedDispatch
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        mock_read.return_value = {
            86: AnticipatedDispatch(lead_stages=2, values_mw=[204.5647, 0.0])
        }
        mock_bounds.return_value = {86: (0.0, 481.27)}

        with caplog.at_level(
            logging.WARNING, logger="cobre_bridge.converters.initial_conditions"
        ):
            result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        assert result["past_anticipated_commitments"] == [
            {"thermal_id": 0, "values_mw": [204.5647, 0.0]}
        ]
        assert "clamping" not in caplog.text

    @patch("cobre_bridge.converters.initial_conditions.thermal_generation_bounds")
    @patch("cobre_bridge.converters.initial_conditions.read_anticipated_dispatch")
    def test_out_of_range_values_clamped_and_warned(
        self, mock_read, mock_bounds, tmp_path, caplog
    ) -> None:
        from cobre_bridge.converters.anticipated import AnticipatedDispatch
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        # 600 > max 481.27 -> clamp to 481.27; -5 < min 0 -> clamp to 0.
        mock_read.return_value = {
            86: AnticipatedDispatch(lead_stages=2, values_mw=[600.0, -5.0])
        }
        mock_bounds.return_value = {86: (0.0, 481.27)}

        with caplog.at_level(
            logging.WARNING, logger="cobre_bridge.converters.initial_conditions"
        ):
            result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        commitments = result["past_anticipated_commitments"]
        assert commitments[0]["values_mw"] == pytest.approx([481.27, 0.0])
        assert "code=86" in caplog.text
        assert "clamping" in caplog.text

    @patch("cobre_bridge.converters.initial_conditions.thermal_generation_bounds")
    @patch("cobre_bridge.converters.initial_conditions.read_anticipated_dispatch")
    def test_code_absent_from_id_map_skipped(
        self, mock_read, mock_bounds, tmp_path, caplog
    ) -> None:
        from cobre_bridge.converters.anticipated import AnticipatedDispatch
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        mock_read.return_value = {
            999: AnticipatedDispatch(lead_stages=1, values_mw=[100.0])
        }
        mock_bounds.return_value = {999: (0.0, 500.0)}

        with caplog.at_level(
            logging.WARNING, logger="cobre_bridge.converters.initial_conditions"
        ):
            result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        # Unknown thermal -> skipped, so no commitments key is emitted.
        assert "past_anticipated_commitments" not in result
        assert "absent from" in caplog.text

    @patch("cobre_bridge.converters.initial_conditions.thermal_generation_bounds")
    @patch("cobre_bridge.converters.initial_conditions.read_anticipated_dispatch")
    def test_non_gnl_case_skips_bounds_computation(
        self, mock_read, mock_bounds, tmp_path
    ) -> None:
        from cobre_bridge.converters.initial_conditions import (
            convert_initial_conditions,
        )

        mock_read.return_value = {}

        result = convert_initial_conditions(_ic_case(tmp_path), self._id_map())

        assert "past_anticipated_commitments" not in result
        mock_bounds.assert_not_called()


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

    def test_all_bus_ids_are_valid(self, tmp_path) -> None:
        hydro_case = _hydro_case(tmp_path)

        conft, clast, term = _thermal_readers()
        dger = MagicMock()
        dger.despacho_antecipado_gnl = 0
        nw_files = _make_nw_files(tmp_path)
        thermal_case = make_case(
            nw_files, conft=conft, clast=clast, term=term, dger=dger
        )

        from cobre_bridge.converters.hydro import convert_hydros
        from cobre_bridge.converters.network import convert_buses
        from cobre_bridge.converters.thermal import convert_thermals

        # Use a shared id_map that covers both subsystems and all plants.
        id_map = NewaveIdMap(
            subsystem_ids=[1, 2, 99],
            hydro_codes=[1, 2],
            thermal_codes=[10, 20, 30],
        )

        buses_result = convert_buses(
            make_case(nw_files, sistema=_make_sistema_mock()), id_map
        )
        hydros_result = convert_hydros(hydro_case, id_map)
        thermals_result = convert_thermals(thermal_case, id_map)

        valid_bus_ids = {b["id"] for b in buses_result["buses"]}

        for h in hydros_result["hydros"]:
            assert h["bus_id"] in valid_bus_ids, (
                f"Hydro '{h['name']}' has bus_id={h['bus_id']} not in buses"
            )

        for t in thermals_result["thermals"]:
            assert t["bus_id"] in valid_bus_ids, (
                f"Thermal '{t['name']}' has bus_id={t['bus_id']} not in buses"
            )

    def test_downstream_ids_are_valid(self, tmp_path) -> None:
        case = _hydro_case(tmp_path)
        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)
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
    """Four plants: two real, two fictitious accounting twins.

    Each fictitious plant (ρ=0) shares its real twin's posto — the structural
    fictitious signature: ``FICT.SERRA M`` on posto 1 with ``USINA_A`` (ρ>0),
    ``FICT.CAMPO G`` on posto 3 with ``USINA_B`` (ρ>0).
    """
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 3, 4],
            "nome_usina": ["USINA_A", "FICT.SERRA M", "USINA_B", "FICT.CAMPO G"],
            "posto": [1, 1, 3, 3],
            "codigo_usina_jusante": [pd.NA, pd.NA, 1, 2],
            "ree": [1, 1, 1, 1],
            "volume_inicial_percentual": [50.0, 60.0, 70.0, 80.0],
            "usina_existente": ["EX", "EX", "EX", "EX"],
            "usina_modificada": [0, 0, 0, 0],
        }
    )


def _fict_cadastro(rho: dict[int, float]) -> pd.DataFrame:
    """Minimal Hidr.cadastro: produtibilidade_especifica per code (for the
    structural fictitious test in id-map builds)."""
    return pd.DataFrame(
        [{"codigo_usina": c, "produtibilidade_especifica": r} for c, r in rho.items()]
    ).set_index("codigo_usina")


class TestBuildIdMap:
    """Unit tests for ``pipeline._build_id_map`` fictitious-plant filtering."""

    @patch("inewave.newave.Hidr")
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
        mock_hidr_cls,
        tmp_path,
    ) -> None:
        """FICT. plants must be absent from id_map.all_hydro_codes."""
        for fname in ("confhd.dat", "conft.dat", "sistema.dat", "ree.dat"):
            (tmp_path / fname).touch()

        mock_confhd = MagicMock()
        mock_confhd.usinas = _make_confhd_df_with_fict()
        mock_confhd_cls.read.return_value = mock_confhd

        # Real twins generate; FICT plants have ρ=0 and share their posto →
        # structurally fictitious, so codes 2 and 4 are excluded.
        mock_hidr_cls.read.return_value.cadastro = _fict_cadastro(
            {1: 0.01, 2: 0.0, 3: 0.01, 4: 0.0}
        )

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

    @patch("inewave.newave.Hidr")
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
        mock_hidr_cls,
        tmp_path,
    ) -> None:
        """15 fictitious twins among 160 existing -> 145 hydro codes in id_map.

        Each fictitious plant (ρ=0) shares a real generator's posto; the real
        plants get unique generating postos.
        """
        for fname in ("confhd.dat", "conft.dat", "sistema.dat", "ree.dat"):
            (tmp_path / fname).touch()

        n_real, n_fict = 145, 15
        rho: dict[int, float] = {}
        rows = []
        for i in range(1, n_real + n_fict + 1):
            if i > n_real:
                # Fictitious twin: ρ=0, shares the posto of real plant (i - n_real).
                name, posto, r = f"FICT.PLANT_{i}", i - n_real, 0.0
            else:
                name, posto, r = f"PLANT_{i}", i, 0.01
            rho[i] = r
            rows.append(
                {
                    "codigo_usina": i,
                    "nome_usina": name,
                    "posto": posto,
                    "codigo_usina_jusante": pd.NA,
                    "ree": 1,
                    "volume_inicial_percentual": 50.0,
                    "usina_existente": "EX",
                }
            )
        confhd_df = pd.DataFrame(rows)
        mock_hidr_cls.read.return_value.cadastro = _fict_cadastro(rho)

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

    @patch("inewave.newave.Hidr")
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
        mock_hidr_cls,
        tmp_path,
    ) -> None:
        """When no FICT. plants exist, all existing plants are included."""
        for fname in ("confhd.dat", "conft.dat", "sistema.dat", "ree.dat"):
            (tmp_path / fname).touch()
        mock_hidr_cls.read.return_value.cadastro = pd.DataFrame()

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

    def test_downstream_to_fict_is_none(self, tmp_path) -> None:
        """Plant with a fictitious downstream gets downstream_id=None.

        USINA_A (code=1) has codigo_usina_jusante=2, which is FICT.SERRA M.
        Because FICT.SERRA M is absent from id_map, the KeyError catch in
        hydro.py must produce downstream_id=None for USINA_A.
        """
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

        # Hidr.cadastro for plant 1 only.
        cadastro = _make_hidr_cadastro().iloc[:1].copy()

        case = _hydro_case(tmp_path, cadastro=cadastro, confhd=confhd_df)

        from cobre_bridge.converters.hydro import convert_hydros

        # id_map has only plant 1; plant 2 (fictitious) is absent.
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1], thermal_codes=[])
        result = convert_hydros(case, id_map)

        assert len(result["hydros"]) == 1
        assert result["hydros"][0]["downstream_id"] is None

    def test_terminal_plant_with_matching_fict_resolves_through_chain(
        self, tmp_path
    ) -> None:
        """A real plant with confhd jusante=0 and a name-matched FICT must
        wire to the next real plant via the FICT chain.

        Topology:
            USINA_A (code=1, jusante=0)          ← physically terminal in confhd
            FICT.USINA (code=2, jusante=3)       ← carries the energy cascade
            USINA_B (code=3, jusante=0)          ← real downstream

        After the FICT-cascade fix, USINA_A's downstream_id must point to
        USINA_B (cobre id=1), not None as in the pre-fix behavior.  The
        7-char name match is ``USINA A`` (after the FICT. prefix) matching
        ``USINA_A``'s first-7-char key — pure prefix equality.
        """
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2, 3],
                "nome_usina": ["USINA_A", "FICT.USINA_A", "USINA_B"],
                # FICT.USINA_A shares USINA_A's posto (1) — the structural
                # fictitious-twin signature that drives the posto-based Rule 3.
                "posto": [1, 1, 3],
                "codigo_usina_jusante": [0, 3, 0],
                "ree": [1, 1, 1],
                "volume_inicial_percentual": [50.0, 50.0, 50.0],
                "usina_existente": ["EX", "EX", "EX"],
                "usina_modificada": [0, 0, 0],
            }
        )

        cadastro = _make_hidr_cadastro().copy()
        # _make_hidr_cadastro has plants 1 and 2.  Make plant 2 the fictitious
        # twin (ρ=0, sharing plant 1's posto) and add plant 3 as a second real
        # plant cloned from plant 1.
        plant3 = cadastro.iloc[0:1].copy()
        plant3.index = [3]
        cadastro = pd.concat([cadastro, plant3])
        # Zero out the twin's specific productivity so it is classified
        # fictitious (ρ=0 sharing a generating posto).
        cadastro.loc[2, "produtibilidade_especifica"] = 0.0

        case = _hydro_case(tmp_path, cadastro=cadastro, confhd=confhd_df)

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 3], thermal_codes=[])
        result = convert_hydros(case, id_map)

        assert len(result["hydros"]) == 2
        by_code = {h["name"]: h for h in result["hydros"]}
        usina_a = by_code["USINA_A"]
        usina_b = by_code["USINA_B"]
        # USINA_A must wire to USINA_B via the FICT chain.
        assert usina_a["downstream_id"] == usina_b["id"], (
            f"Expected USINA_A.downstream_id == {usina_b['id']}, "
            f"got {usina_a['downstream_id']}"
        )
        # USINA_B remains terminal.
        assert usina_b["downstream_id"] is None


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

    def _penalid_case(self, tmp_path, mock_penalid):
        """Build a case whose PENALID reader is *mock_penalid* (path set)."""
        return make_case(
            make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat"),
            penalid=mock_penalid,
        )

    def test_reads_penalties_by_ree(self, tmp_path) -> None:
        """Correct Cobre field names and values are returned per REE."""
        from cobre_bridge.converters.hydro import _read_penalid

        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()

        result = _read_penalid(self._penalid_case(tmp_path, mock_penalid))

        # REE 1 checks.
        assert 1 in result
        assert result[1]["water_withdrawal_violation_cost"] == pytest.approx(8300.0)
        assert result[1]["outflow_violation_below_cost"] == pytest.approx(3179.35)
        assert result[1]["generation_violation_below_cost"] == pytest.approx(4500.0)
        # TURBMX must not appear (no Cobre mapping).
        assert "turbined_violation_below_cost" not in result[1]

        # REE 2 checks.
        assert 2 in result
        assert result[2]["water_withdrawal_violation_cost"] == pytest.approx(9100.0)
        assert result[2]["outflow_violation_below_cost"] == pytest.approx(2800.0)

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        """Absent PENALID.DAT returns an empty dict without raising."""
        from cobre_bridge.converters.hydro import _read_penalid

        # No penalid.dat — pass penalid=None.
        result = _read_penalid(make_case(tmp_path, penalid=None))

        assert result == {}

    def test_nan_values_are_skipped(self, tmp_path) -> None:
        """NaN cost values at patamar 1 do not appear in the output dict."""
        import math

        from cobre_bridge.converters.hydro import _read_penalid

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

        result = _read_penalid(self._penalid_case(tmp_path, mock_penalid))

        assert 1 in result
        # DESVIO had NaN — must be absent.
        assert "water_withdrawal_violation_cost" not in result[1]
        # VAZMIN had 5000.0 — must be present.
        assert result[1]["outflow_violation_below_cost"] == pytest.approx(5000.0)

    def test_patamar2_rows_ignored(self, tmp_path) -> None:
        """Tier-2 patamar rows are excluded even when they have numeric values."""
        from cobre_bridge.converters.hydro import _read_penalid

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

        result = _read_penalid(self._penalid_case(tmp_path, mock_penalid))

        assert result == {}


class TestConvertHydrosPenalid:
    """Integration tests for PENALID.DAT -> hydro penalties in convert_hydros."""

    def test_penalid_present_still_leaves_penalties_none(self, tmp_path) -> None:
        """Per-plant penalties are always None; PENALID is handled globally."""
        # convert_hydros does not read PENALID (per-plant overrides were removed;
        # the PENALID mock is supplied only to mirror a case where the file is
        # present). The case carries it for parity with production.
        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()
        case = _hydro_case(tmp_path, penalid=mock_penalid)

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)

        # Per-plant penalty overrides were removed: PENALID values are
        # converted once via system-average productivity in penalties.json.
        for hydro in result["hydros"]:
            assert hydro["penalties"] is None, (
                f"Plant '{hydro['name']}' should have penalties=None "
                "(per-plant overrides removed; handled globally)"
            )

    def test_missing_penalid_leaves_penalties_none(self, tmp_path) -> None:
        """When PENALID.DAT is absent, every hydro entry has penalties=None."""
        case = _hydro_case(tmp_path)

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)

        for hydro in result["hydros"]:
            assert hydro["penalties"] is None, (
                f"Plant '{hydro['name']}' should have penalties=None "
                "when PENALID.DAT is absent"
            )

    def test_different_rees_still_get_none_penalties(self, tmp_path) -> None:
        """Plants in different REEs still get penalties=None (global handling)."""
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

        # REE table: REE 1 -> subsystem 1, REE 2 -> subsystem 1.
        ree_df = pd.DataFrame(
            {"codigo": [1, 2], "nome": ["SE", "S"], "submercado": [1, 1]}
        )

        case = _hydro_case(tmp_path, confhd=confhd_df, rees=ree_df)

        from cobre_bridge.converters.hydro import convert_hydros

        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])
        result = convert_hydros(case, id_map)

        for hydro in result["hydros"]:
            assert hydro["penalties"] is None


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
        """Two hydros: The source model codes 10 and 20 -> Cobre IDs 0 and 1."""
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[10, 20],
            thermal_codes=[],
        )

    def _withdrawal_case(self, tmp_path, *, dsvagua, dger, confhd=None):
        """Build a case for ``convert_water_withdrawal`` with dsvagua/dger/confhd
        mocks pre-cached (dsvagua path set so the present-file guard passes)."""
        parsed: dict = {"dger": dger, "dsvagua": dsvagua}
        if confhd is not None:
            parsed["confhd"] = confhd
        return make_case(
            make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"), **parsed
        )

    def test_basic_returns_correct_schema(self, tmp_path: Path) -> None:
        """Two plants, three dates each: table has the three expected columns."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        rows = [
            {
                "codigo_usina": 10,
                "data": datetime.datetime(2020, 1, 1),
                "valor": -2.0,
            },
            {
                "codigo_usina": 10,
                "data": datetime.datetime(2020, 2, 1),
                "valor": -3.0,
            },
            {
                "codigo_usina": 20,
                "data": datetime.datetime(2020, 1, 1),
                "valor": -1.0,
            },
        ]
        dger_mock = _make_dger_mock(2020, 1, 5)

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = dger_mock.ano_inicio_estudo
        mock_dger.mes_inicio_estudo = dger_mock.mes_inicio_estudo
        mock_dger.num_anos_estudo = dger_mock.num_anos_estudo

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, self._make_id_map())

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
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 10, "data": datetime.datetime(2020, 2, 1), "valor": -5.0}
        ]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5
        mock_dger.num_anos_pos_estudo = 0

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, id_map)

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        assert row["hydro_id"][0] == id_map.hydro_id(10)
        assert row["stage_id"][0] == 1
        assert row["water_withdrawal_m3s"][0] == pytest.approx(5.0)

    def test_groupby_sum_same_plant_same_date(self, tmp_path: Path) -> None:
        """Two rows with the same plant/date are summed then negated."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 10, "data": datetime.datetime(2020, 1, 1), "valor": -3.0},
            {"codigo_usina": 10, "data": datetime.datetime(2020, 1, 1), "valor": -7.0},
        ]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5
        mock_dger.num_anos_pos_estudo = 0

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, id_map)

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        # -3.0 + -7.0 = -10.0; negated -> 10.0
        assert row["water_withdrawal_m3s"][0] == pytest.approx(10.0)
        assert row["stage_id"][0] == 0

    def test_missing_dsvagua_file_returns_none(self, tmp_path: Path) -> None:
        """When dsvagua.dat is absent the converter returns None without error."""
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        # dsvagua absent (path None) — the converter must return None before
        # touching dger.
        case = make_case(make_nw_files(tmp_path, dsvagua=None))
        result = convert_water_withdrawal(case, self._make_id_map())
        assert result is None

    def test_empty_desvios_returns_none(self, tmp_path: Path) -> None:
        """When desvios is None the converter returns None."""
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = None
        mock_dger = MagicMock()
        mock_dger.outros_usos_da_agua = 1
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        case = self._withdrawal_case(tmp_path, dsvagua=mock_dsvagua, dger=mock_dger)
        result = convert_water_withdrawal(case, self._make_id_map())

        assert result is None

    def test_dger_outros_usos_da_agua_zero_skips_dsvagua(self, tmp_path: Path) -> None:
        """``dger.outros_usos_da_agua == 0`` short-circuits the conversion.

        Mirrors the source model's own behaviour — when the dger switch is 0 the solver
        ignores ``dsvagua.dat`` regardless of its content, so the converter must not
        emit any water-withdrawal rows.
        """
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        mock_dger = MagicMock()
        mock_dger.outros_usos_da_agua = 0
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        # Do NOT pre-cache the dsvagua slot: the dger switch must short-circuit
        # before ``case.dsvagua`` is ever accessed. If the short-circuit
        # regressed, the cached_property would attempt to parse the (absent)
        # dsvagua.dat path and raise, failing this test loudly.
        case = make_case(
            make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"),
            dger=mock_dger,
        )
        result = convert_water_withdrawal(case, self._make_id_map())

        assert result is None
        # The short-circuit must happen before any dsvagua access.
        assert "dsvagua" not in case.__dict__

    def test_codes_outside_id_map_are_dropped(self, tmp_path: Path) -> None:
        """``codigo_usina`` codes the id_map doesn't know are silently dropped.

        ``dsvagua.dat`` frequently carries codes for non-dispatchable
        plants (fictitious nodes, RHEs, etc.) that are filtered out of
        the id_map; logging a warning for each would be noisy.
        """
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 10, "data": datetime.datetime(2020, 1, 1), "valor": -4.0},
            {"codigo_usina": 99, "data": datetime.datetime(2020, 1, 1), "valor": -2.0},
        ]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5
        mock_dger.num_anos_pos_estudo = 0

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, id_map)

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        assert row["water_withdrawal_m3s"][0] == pytest.approx(4.0)
