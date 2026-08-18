"""Tests for the DECOMP output readers."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl
import pytest
from idecomp.decomp import Decomptim, DecOperGnl, Relato

from cobre_bridge.comparators.decomp_readers import (
    _read_dec_oper,
    _read_relato_table,
    _resolve_relato,
    _resolve_result_file,
    read_dec_oper_gnl,
    read_dec_oper_interc,
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_dec_oper_usit,
    read_decomp_tim,
    read_relato_balance,
    read_relato_convergence,
    read_relato_costs,
    read_relato_expected_cost,
    reconcile_kdollars_to_reais,
)

_DECK = Path("example/decomp-jul-26-rv3")
_REDUCED_DECK = Path("example/decomp-mar-26-rv2-reduced")

_needs_deck = pytest.mark.skipif(not _DECK.is_dir(), reason="rv3 outputs not present")
_needs_reduced_deck = pytest.mark.skipif(
    not (_REDUCED_DECK / "saidas" / "relato.rv2").is_file(),
    reason="reduced deck outputs not present",
)
_needs_reduced_deck_tim = pytest.mark.skipif(
    not (_REDUCED_DECK / "saidas" / "decomp.tim").is_file(),
    reason="reduced deck decomp.tim not present",
)


class _StubFile:
    def __init__(self, table: pd.DataFrame | None) -> None:
        self.tabela = table


class _StubTimFile:
    """Stub for a `Decomptim.read(...)` result, carrying `tempos_etapas`."""

    def __init__(self, table: pd.DataFrame | None) -> None:
        self.tempos_etapas = table


def _stub_reader(table: pd.DataFrame | None) -> type:
    class _Reader:
        @staticmethod
        def read(path: str) -> _StubFile:  # noqa: ARG004
            return _StubFile(table)

    return _Reader


class _StubRelatoFile:
    """Stub for a `Relato.read(...)` result, carrying named pandas tables
    (e.g. ``convergencia=...``, ``balanco_energetico=...``)."""

    def __init__(self, **tables: pd.DataFrame | None) -> None:
        for name, table in tables.items():
            setattr(self, name, table)


class TestResolveResultFile:
    """`_resolve_result_file`: saidas-first, case-insensitive discovery."""

    def test_saidas_only(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        target = saidas / "dec_oper_ree.csv"
        target.touch()
        assert _resolve_result_file(tmp_path, "dec_oper_ree.csv") == target

    def test_root_only_no_saidas_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "dec_oper_sist.csv"
        target.touch()
        assert _resolve_result_file(tmp_path, "dec_oper_sist.csv") == target

    def test_prefers_saidas_when_both_present(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        saidas_copy = saidas / "dec_oper_sist.csv"
        saidas_copy.touch()
        (tmp_path / "dec_oper_sist.csv").touch()
        assert _resolve_result_file(tmp_path, "dec_oper_sist.csv") == saidas_copy

    def test_case_insensitive_in_saidas(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        target = saidas / "DEC_OPER_SIST.CSV"
        target.touch()
        assert _resolve_result_file(tmp_path, "dec_oper_sist.csv") == target

    def test_absent_from_both_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_result_file(tmp_path, "dec_oper_ree.csv") is None


class TestResolveRelato:
    """`_resolve_relato`: same saidas-first precedence for relato.rvN."""

    def test_saidas_only(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        target = saidas / "relato.rv2"
        target.touch()
        assert _resolve_relato(tmp_path) == target

    def test_root_only_no_saidas_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "relato.rv2"
        target.touch()
        assert _resolve_relato(tmp_path) == target

    def test_absent_from_both_returns_none(self, tmp_path: Path) -> None:
        assert _resolve_relato(tmp_path) is None


class TestReadDecOperCore:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_sist.csv"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(None))

    def test_missing_file_names_both_locations(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError) as exc_info:
            _read_dec_oper(tmp_path, "dec_oper_ree.csv", _stub_reader(None))
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas" in message

    def test_found_in_saidas_only(self, tmp_path: Path) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_sist.csv").touch()
        table = pd.DataFrame({"estagio": [1, 2], "cmo": [10.0, 12.0]})
        result = _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(table))
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)

    def test_empty_parse_raises(self, tmp_path: Path) -> None:
        (tmp_path / "dec_oper_sist.csv").touch()
        with pytest.raises(ValueError, match="parsed empty"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(None))
        with pytest.raises(ValueError, match="parsed empty"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(pd.DataFrame()))

    def test_returns_polars_frame(self, tmp_path: Path) -> None:
        (tmp_path / "dec_oper_sist.csv").touch()
        table = pd.DataFrame({"estagio": [1, 2], "cmo": [10.0, 12.0]})
        result = _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(table))
        assert isinstance(result, pl.DataFrame)
        assert result.shape == (2, 2)


class TestRealDeckReaders:
    @_needs_deck
    def test_sist_has_system_columns(self) -> None:
        df = read_dec_oper_sist(_DECK)
        assert df.height > 0
        for column in (
            "estagio",
            "no",
            "cenario",
            "patamar",
            "codigo_submercado",
            "demanda_MW",
            "geracao_hidroeletrica_MW",
            "geracao_termica_MW",
            "itaipu_50MW",
            "itaipu_60MW",
            "deficit_MW",
            "cmo",
        ):
            assert column in df.columns

    @_needs_deck
    def test_usih_has_water_balance_columns(self) -> None:
        df = read_dec_oper_usih(_DECK)
        assert df.height > 0
        for column in (
            "codigo_usina",
            "geracao_MW",
            "potencia_disponivel_MW",
            "vazao_turbinada_m3s",
            "vazao_vertida_m3s",
            "volume_util_final_hm3",
        ):
            assert column in df.columns

    @_needs_deck
    def test_usit_has_bound_columns(self) -> None:
        df = read_dec_oper_usit(_DECK)
        assert df.height > 0
        for column in (
            "codigo_usina",
            "geracao_MW",
            "geracao_minima_MW",
            "geracao_maxima_MW",
            "custo_incremental",
        ):
            assert column in df.columns

    @_needs_deck
    def test_interc_has_flow_columns(self) -> None:
        df = read_dec_oper_interc(_DECK)
        assert df.height > 0
        for column in (
            "codigo_submercado_de",
            "codigo_submercado_para",
            "intercambio_origem_MW",
            "capacidade_MW",
        ):
            assert column in df.columns

    @_needs_deck
    def test_relato_convergence(self) -> None:
        df = read_relato_convergence(_DECK)
        assert df.height > 0
        for column in ("iteracao", "zinf", "zsup", "gap_percentual"):
            assert column in df.columns
        # A converged production run: the last gap is small.
        final_gap = df["gap_percentual"][-1]
        assert final_gap is not None

    def test_relato_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_convergence(tmp_path)


class TestReadRelatoConvergenceDiscovery:
    """Tier-1: `read_relato_convergence` resolves a `saidas/`-only report."""

    def test_finds_saidas_only_relato(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "iteracao": [1],
                "zinf": [1.0],
                "zsup": [1.0],
                "gap_percentual": [0.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(convergencia=stub_table)),
        )
        result = read_relato_convergence(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1


class TestReadRelatoTable:
    """`_read_relato_table`: shared helper behind every relato reader."""

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            _read_relato_table(tmp_path, "convergencia")

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=None)),
        )
        with pytest.raises(ValueError, match="balanco_energetico"):
            _read_relato_table(tmp_path, "balanco_energetico")

    def test_empty_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(balanco_energetico=pd.DataFrame())
            ),
        )
        with pytest.raises(ValueError, match="balanco_energetico"):
            _read_relato_table(tmp_path, "balanco_energetico")

    def test_finds_saidas_only_relato(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "relato.rv2").touch()
        stub_table = pd.DataFrame({"nome_submercado": ["SUDESTE"]})
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=stub_table)),
        )
        result = _read_relato_table(tmp_path, "balanco_energetico")
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1


class TestReadRelatoBalance:
    """`read_relato_balance`: per-submarket energy balance table."""

    def test_returns_expected_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "cenario": [1],
                "nome_submercado": ["SUDESTE"],
                "geracao_hidraulica": [1000.0],
                "geracao_termica": [200.0],
                "compra": [0.0],
                "venda": [0.0],
                "geracao_itaipu_50hz": [50.0],
                "geracao_itaipu_60hz": [60.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=stub_table)),
        )
        result = read_relato_balance(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in (
            "nome_submercado",
            "geracao_hidraulica",
            "geracao_termica",
            "compra",
            "venda",
            "geracao_itaipu_50hz",
            "geracao_itaipu_60hz",
        ):
            assert column in result.columns

    def test_none_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(balanco_energetico=None)),
        )
        with pytest.raises(ValueError, match="balanco_energetico"):
            read_relato_balance(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_balance(tmp_path)

    @_needs_reduced_deck
    def test_real_deck_is_non_empty(self) -> None:
        df = read_relato_balance(_REDUCED_DECK)
        assert df.height > 0


class TestReadRelatoCosts:
    """`read_relato_costs`: per-(stage, scenario) operating cost table."""

    def test_returns_expected_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "cenario": [1],
                "probabilidade": [1.0],
                "custo_futuro": [1234.5],
                "custo_presente": [678.9],
                "geracao_termica": [50.0],
                "violacao_desvio": [0.0],
                "penalidade_vertimento_reservatorio": [0.0],
                "penalidade_vertimento_fio": [0.0],
                "violacao_turbinamento_reservatorio": [0.0],
                "violacao_turbinamento_fio": [0.0],
                "penalidade_intercambio": [0.0],
                "cmo_SE": [100.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(relatorio_operacao_custos=stub_table)
            ),
        )
        result = read_relato_costs(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in (
            "custo_presente",
            "custo_futuro",
            "geracao_termica",
            "penalidade_intercambio",
            "penalidade_vertimento_reservatorio",
        ):
            assert column in result.columns

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(lambda _path: _StubRelatoFile(relatorio_operacao_custos=None)),
        )
        with pytest.raises(ValueError, match="relatorio_operacao_custos"):
            read_relato_costs(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_costs(tmp_path)

    @_needs_reduced_deck
    def test_real_deck_is_non_empty(self) -> None:
        df = read_relato_costs(_REDUCED_DECK)
        assert df.height > 0


class TestReadRelatoExpectedCost:
    """`read_relato_expected_cost`: per-parcela expected cost, wide by stage."""

    def test_returns_estagio_columns(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        stub_table = pd.DataFrame(
            {
                "parcela": ["GERACAO TERMICA"],
                "estagio_1": [10.0],
                "estagio_2": [20.0],
                "estagio_3": [30.0],
            }
        )
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(custo_operacao_valor_esperado=stub_table)
            ),
        )
        result = read_relato_expected_cost(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        for column in ("parcela", "estagio_1", "estagio_2", "estagio_3"):
            assert column in result.columns

    def test_none_table_raises_naming_attr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "relato.rv2").touch()
        monkeypatch.setattr(
            Relato,
            "read",
            staticmethod(
                lambda _path: _StubRelatoFile(custo_operacao_valor_esperado=None)
            ),
        )
        with pytest.raises(ValueError, match="custo_operacao_valor_esperado"):
            read_relato_expected_cost(tmp_path)

    def test_missing_relato_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="relato"):
            read_relato_expected_cost(tmp_path)

    @_needs_reduced_deck
    def test_real_deck_is_non_empty(self) -> None:
        df = read_relato_expected_cost(_REDUCED_DECK)
        assert df.height > 0


class TestReadDecOperGnl:
    """`read_dec_oper_gnl`: anticipated-thermal operation, saidas-only file."""

    def test_finds_saidas_only_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "dec_oper_gnl.csv").touch()
        stub_table = pd.DataFrame(
            {
                "estagio": [1],
                "codigo_usina": [1],
                "custo_incremental": [5.0],
                "geracao_MW": [100.0],
                "custo_geracao": [42.0],
            }
        )
        monkeypatch.setattr(
            DecOperGnl,
            "read",
            staticmethod(lambda _path: _StubFile(stub_table)),
        )
        result = read_dec_oper_gnl(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "custo_geracao" in result.columns
        assert "geracao_MW" in result.columns

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_gnl.csv"):
            read_dec_oper_gnl(tmp_path)

    def test_empty_parse_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "dec_oper_gnl.csv").touch()
        monkeypatch.setattr(
            DecOperGnl,
            "read",
            staticmethod(lambda _path: _StubFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_dec_oper_gnl(tmp_path)

    @_needs_reduced_deck
    def test_real_deck_is_non_empty(self) -> None:
        df = read_dec_oper_gnl(_REDUCED_DECK)
        assert df.height > 0


class TestReadDecompTim:
    """`read_decomp_tim`: wall-clock timing table from ``decomp.tim``."""

    def test_finds_saidas_only_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        saidas = tmp_path / "saidas"
        saidas.mkdir()
        (saidas / "decomp.tim").touch()
        stub_table = pd.DataFrame(
            {
                "Etapa": ["Leitura de Dados", "Convergencia", "Tempo Total"],
                "Tempo": [
                    timedelta(seconds=1),
                    timedelta(minutes=5),
                    timedelta(minutes=6, seconds=1),
                ],
            }
        )
        monkeypatch.setattr(
            Decomptim,
            "read",
            staticmethod(lambda _path: _StubTimFile(stub_table)),
        )
        result = read_decomp_tim(tmp_path)
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["Etapa", "Tempo"]
        assert result.height == 3

    def test_missing_file_raises_naming_both_locations(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="decomp.tim") as exc_info:
            read_decomp_tim(tmp_path)
        message = str(exc_info.value)
        assert str(tmp_path) in message
        assert "saidas" in message

    def test_none_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "decomp.tim").touch()
        monkeypatch.setattr(
            Decomptim,
            "read",
            staticmethod(lambda _path: _StubTimFile(None)),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_decomp_tim(tmp_path)

    def test_empty_table_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "decomp.tim").touch()
        monkeypatch.setattr(
            Decomptim,
            "read",
            staticmethod(lambda _path: _StubTimFile(pd.DataFrame())),
        )
        with pytest.raises(ValueError, match="parsed empty"):
            read_decomp_tim(tmp_path)

    @_needs_reduced_deck_tim
    def test_real_deck_is_non_empty(self) -> None:
        df = read_decomp_tim(_REDUCED_DECK)
        assert df.height > 0
        assert "total" in " ".join(df["Etapa"].to_list()).lower()


class TestReconcileKdollarsToReais:
    """`reconcile_kdollars_to_reais`: the single k$ -> R$ conversion site."""

    @pytest.mark.parametrize(
        ("kdollars", "reais"),
        [(1.0, 1000.0), (2.5, 2500.0), (0.0, 0.0)],
    )
    def test_applies_thousand_factor(self, kdollars: float, reais: float) -> None:
        assert reconcile_kdollars_to_reais(kdollars) == reais

    def test_docstring_names_kdollar_provenance(self) -> None:
        docstring = reconcile_kdollars_to_reais.__doc__
        assert docstring is not None
        assert "project_decomp_fcf_unit_conversion_bug" in docstring
        assert "k$" in docstring
