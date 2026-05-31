"""Tests for the canonical active-hydro filter helpers."""

from __future__ import annotations

import pandas as pd

from cobre_bridge.plants import (
    active_hydro_codes,
    active_hydros,
    existing_hydros,
    fictitious_existing_names,
)


def _confhd(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


_CASE = _confhd(
    [
        {"codigo_usina": 10, "nome_usina": "PLANT A", "usina_existente": "EX"},
        {"codigo_usina": 3, "nome_usina": "FICT. ACCT", "usina_existente": "EX"},
        {"codigo_usina": 7, "nome_usina": "PLANT B", "usina_existente": "EX"},
        {"codigo_usina": 99, "nome_usina": "FUTURE", "usina_existente": "NE"},
        {"codigo_usina": 42, "nome_usina": "  FICT. PADDED ", "usina_existente": "EX"},
    ]
)


def test_existing_hydros_keeps_all_ex_including_fict() -> None:
    existing = existing_hydros(_CASE)
    assert sorted(existing["codigo_usina"]) == [3, 7, 10, 42]  # NE (99) dropped


def test_active_hydros_drops_fict_and_non_existing() -> None:
    active = active_hydros(_CASE)
    assert list(active["codigo_usina"]) == [10, 7]  # FICT (3, 42) and NE (99) gone


def test_active_hydros_preserves_declaration_order() -> None:
    # 10 is declared before 7; the id-map relies on this ordering.
    assert active_hydro_codes(_CASE) == [10, 7]


def test_active_hydros_strips_before_matching_fict_prefix() -> None:
    # "  FICT. PADDED " (code 42) must be treated as fictitious despite padding.
    assert 42 not in active_hydro_codes(_CASE)


def test_fictitious_existing_names_lists_excluded() -> None:
    names = fictitious_existing_names(_CASE)
    assert [n.strip() for n in names] == ["FICT. ACCT", "FICT. PADDED"]


def test_active_hydro_codes_returns_python_ints() -> None:
    assert all(isinstance(code, int) for code in active_hydro_codes(_CASE))


def test_empty_confhd_yields_empty_active_set() -> None:
    empty = _confhd(
        [{"codigo_usina": 1, "nome_usina": "GONE", "usina_existente": "NE"}]
    )
    assert active_hydro_codes(empty) == []
    assert fictitious_existing_names(empty) == []
