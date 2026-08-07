"""Column order of the DECOMP ``constraints/`` bounds parquets.

``block_id`` is an *indexing* column (which patamar a row applies to), so it
belongs with the other index columns — ``entity_id``, ``stage_id`` — ahead of
the value columns it indexes, matching the ``(constraint_id, stage_id,
block_id)`` order the generic-constraint bounds already use. These tests pin
that layout so it cannot silently drift back to a trailing ``block_id``.

The schema constant governs the written column order (each writer passes it as
``pa.table(..., schema=...)``), so asserting on the schema names asserts on the
parquet layout.
"""

from __future__ import annotations

from cobre_bridge.decomp.bounds import _HYDRO_BOUNDS_SCHEMA
from cobre_bridge.decomp.group_bounds import _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA
from cobre_bridge.decomp.network import _LINE_BOUNDS_SCHEMA
from cobre_bridge.decomp.thermal import _THERMAL_BOUNDS_SCHEMA


def test_hydro_bounds_index_columns_precede_values() -> None:
    assert _HYDRO_BOUNDS_SCHEMA.names == [
        "hydro_id",
        "stage_id",
        "block_id",
        "min_outflow_m3s",
        "min_storage_hm3",
        "max_storage_hm3",
    ]


def test_thermal_bounds_index_columns_precede_values() -> None:
    assert _THERMAL_BOUNDS_SCHEMA.names == [
        "thermal_id",
        "stage_id",
        "block_id",
        "min_generation_mw",
        "max_generation_mw",
        "cost_per_mwh",
    ]


def test_line_bounds_index_columns_precede_values() -> None:
    assert _LINE_BOUNDS_SCHEMA.names == [
        "line_id",
        "stage_id",
        "block_id",
        "direct_mw",
        "reverse_mw",
    ]


def test_group_bounds_index_columns_precede_values() -> None:
    assert _HYDRO_UNIT_GROUP_BOUNDS_SCHEMA.names == [
        "hydro_id",
        "hydro_unit_group_id",
        "stage_id",
        "block_id",
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_generation_mw",
        "max_generation_mw",
    ]
