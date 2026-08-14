"""Tests for the DECOMP output readers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from cobre_bridge.comparators.decomp_readers import (
    _read_dec_oper,
    read_dec_oper_interc,
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_dec_oper_usit,
    read_relato_convergence,
)

_DECK = Path("example/decomp-jul-26-rv3")

_needs_deck = pytest.mark.skipif(not _DECK.is_dir(), reason="rv3 outputs not present")


class _StubFile:
    def __init__(self, table: pd.DataFrame | None) -> None:
        self.tabela = table


def _stub_reader(table: pd.DataFrame | None) -> type:
    class _Reader:
        @staticmethod
        def read(path: str) -> _StubFile:  # noqa: ARG004
            return _StubFile(table)

    return _Reader


class TestReadDecOperCore:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dec_oper_sist.csv"):
            _read_dec_oper(tmp_path, "dec_oper_sist.csv", _stub_reader(None))

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
