"""Tests for the DECOMP-side per-stage storage-bounds emitter.

Pins ``storage_envelope`` (the outer per-stage operating-range helper both
this emitter and the entity ``reservoir`` block share) and
``convert_storage_bounds`` (the sparse per-stage override emitter) against a
synthetic ``EffectiveCadastro`` — no real deck, no dadger.

Rule A (user-confirmed 2026-08-07): a stage's effective ``(volume_minimo,
volume_maximo)`` is compared to the plant's outer envelope
(``storage_envelope``), not to its pre-override base scalar. A stage equal
to the envelope emits no row and simply inherits it from the entity
``reservoir`` block; a stage that differs from the envelope — even by being
*narrower* than it — emits an override row.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from cobre_bridge.decomp.bounds import _HYDRO_BOUNDS_SCHEMA, convert_storage_bounds
from cobre_bridge.decomp.cadastro import EffectiveCadastro, storage_envelope
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import OperativeStage


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame({1: {"volume_minimo": 20.0, "volume_maximo": 100.0}}).T
    df.index.name = "codigo_usina"
    return df


@pytest.fixture
def effective() -> EffectiveCadastro:
    """Plant 1: flat ``volume_minimo``; ``volume_maximo`` raised at stage 2."""
    return EffectiveCadastro(
        base=_hidr_frame(),
        n_stages=3,
        stage_varying={(1, "volume_maximo"): (100.0, 100.0, 250.0)},
    )


@pytest.fixture
def id_map() -> DecompIdMap:
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))


@pytest.fixture
def calendar() -> list[OperativeStage]:
    """Three placeholder stages — only their count is used by the emitter."""
    return [
        OperativeStage(
            index=i,
            start_date=date(2026, 7, 4),
            end_date=date(2026, 7, 11),
            season_id=6,
            block_hours=(168.0,),
        )
        for i in range(3)
    ]


def test_storage_envelope_returns_min_minimo_max_maximo(
    effective: EffectiveCadastro,
) -> None:
    assert storage_envelope(effective, 1) == (20.0, 250.0)


def test_convert_storage_bounds_emits_one_row_per_stage_that_differs_from_envelope(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: list[OperativeStage],
) -> None:
    """Stages 0-1 (ceiling 100.0) differ from the envelope (20.0, 250.0);
    stage 2 equals the envelope exactly and emits no row.
    """
    table = convert_storage_bounds(effective, id_map, calendar)

    assert table["hydro_id"].to_pylist() == [0, 0]
    assert table["stage_id"].to_pylist() == [0, 1]
    assert table["block_id"].to_pylist() == [None, None]
    assert table["min_outflow_m3s"].to_pylist() == [None, None]
    assert table["min_storage_hm3"].to_pylist() == [20.0, 20.0]
    assert table["max_storage_hm3"].to_pylist() == [100.0, 100.0]


def test_convert_storage_bounds_no_stage_varying_volumes_emits_empty_table(
    id_map: DecompIdMap,
    calendar: list[OperativeStage],
) -> None:
    no_override = EffectiveCadastro(base=_hidr_frame(), n_stages=3, stage_varying={})
    table = convert_storage_bounds(no_override, id_map, calendar)
    assert table.num_rows == 0
    assert table.schema == _HYDRO_BOUNDS_SCHEMA


def test_convert_storage_bounds_schema_names_and_null_columns(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: list[OperativeStage],
) -> None:
    table = convert_storage_bounds(effective, id_map, calendar)
    assert table.schema.names == _HYDRO_BOUNDS_SCHEMA.names
    assert all(v is None for v in table["block_id"].to_pylist())
    assert all(v is None for v in table["min_outflow_m3s"].to_pylist())
