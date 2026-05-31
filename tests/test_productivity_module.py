"""Tests for the extracted productivity domain module."""

from __future__ import annotations

import pandas as pd
import pytest

from cobre_bridge.productivity import (
    compute_productivity,
    equivalent_productivity,
    integrated_productivity,
)


def _hreg(**overrides: object) -> pd.Series:
    defaults: dict = {
        "nome_usina": "TEST",
        "produtibilidade_especifica": 0.009,
        "volume_minimo": 100.0,
        "volume_maximo": 1000.0,
        "volume_referencia": 500.0,
        "canal_fuga_medio": 250.0,
        "tipo_regulacao": "M",
        "tipo_perda": 1,
        "perdas": 5.0,
        "a0_volume_cota": 300.0,
        "a1_volume_cota": 0.1,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def test_hydro_aliases_point_to_public_functions() -> None:
    # The whole point of the extraction: hydro.py's private names are now just
    # aliases of the public domain functions, so every existing test that uses
    # the private names also validates these.
    from cobre_bridge.converters.hydro import (
        _compute_integrated_productivity,
        _compute_productivity,
        _equivalent_productivity,
    )

    assert _compute_productivity is compute_productivity
    assert _equivalent_productivity is equivalent_productivity
    assert _compute_integrated_productivity is integrated_productivity


def test_constraints_and_fict_cascade_import_public_names() -> None:
    # Cross-boundary callers must no longer reach into hydro privates.
    import cobre_bridge.converters.constraints as c
    import cobre_bridge.converters.fict_cascade as fc

    assert c.compute_productivity is compute_productivity
    assert c.integrated_productivity is integrated_productivity
    assert fc.compute_productivity is compute_productivity


def test_cmont_override_pins_forebay() -> None:
    # head = cmont - cfuga = 400 - 250 = 150; ·0.95 loss; ·0.009 ρ_esp.
    hreg = _hreg()
    assert compute_productivity(hreg, cmont_override=400.0) == pytest.approx(
        0.009 * (400.0 - 250.0) * 0.95
    )


def test_equivalent_vs_integrated_agree_on_linear_polynomial() -> None:
    # For a purely linear h(V) the mean head equals the value at the midpoint;
    # equivalent_productivity and integrated_productivity use the same integral.
    hreg = _hreg(tipo_perda=0)
    assert equivalent_productivity(hreg) == pytest.approx(integrated_productivity(hreg))
