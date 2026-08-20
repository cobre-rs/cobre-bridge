"""The source-model-derived bounds for the ``compare bounds`` engine — via the
converters.

Each function here delegates to the matching converter (the single definition of how a
bound is built from the source model inputs) and reshapes its ``pa.Table`` output into
the flat ``dict[tuple[int, int, str], float]`` lookup keyed by ``(cobre_entity_id,
stage_id, bound_name)`` that :mod:`comparators.bounds` consumes.

These used to be hand-maintained re-derivations of the converter logic. They
drifted and produced false positives (the ``sentido`` line-direction bug and the
POTEF thermal bug — see the ``bounds-comparator-reimplements-conversion`` memory),
so they now delegate. As a consequence ``compare bounds`` is a **round-trip
consistency check** (converter output vs the Cobre solver's ``bounds.parquet`` for
hydro/thermal; converter recompute vs its on-disk ``line_bounds.parquet`` for
lines), not an independent re-derivation — divergence detection is the job of
``compare results``.
"""

from __future__ import annotations

from cobre_bridge.case import NewaveCase
from cobre_bridge.id_map import NewaveIdMap

# convert_storage_bounds column -> comparison bound_name. (Hydro generation
# (GHMIN) is emitted by the converter but was never part of the hydro bounds
# comparison, so it is intentionally not mapped here.)
_HYDRO_COLUMN_TO_BOUND: dict[str, str] = {
    "min_storage_hm3": "storage_min",
    "max_storage_hm3": "storage_max",
    "min_turbined_m3s": "turbined_min",
    "max_turbined_m3s": "turbined_max",
    "min_outflow_m3s": "outflow_min",
}


def compute_hydro_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Per-stage hydro bounds ``{(hydro_id, stage_id, bound_name): value}``.

    Delegates to :func:`cobre_bridge.converters.hydro.convert_storage_bounds` and
    reshapes its ``min/max_storage``/``turbined``/``outflow`` columns. ``None``
    cells (no override for that plant/stage) are skipped.
    """
    from cobre_bridge.converters.hydro import convert_storage_bounds

    table = convert_storage_bounds(case, id_map)
    if table is None:
        return {}

    cols = table.to_pydict()
    hydro_ids = cols["hydro_id"]
    stage_ids = cols["stage_id"]
    result: dict[tuple[int, int, str], float] = {}
    for column, bound_name in _HYDRO_COLUMN_TO_BOUND.items():
        values = cols[column]
        for hydro_id, stage_id, value in zip(hydro_ids, stage_ids, values):
            if value is None:
                continue
            result[(int(hydro_id), int(stage_id), bound_name)] = float(value)
    return result


def compute_thermal_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Per-stage thermal generation bounds ``{(thermal_id, stage_id, name): v}``.

    Delegates to :func:`cobre_bridge.converters.thermal.convert_thermal_bounds`.
    Returns an empty dict when neither ``expt.dat`` nor ``manutt.dat`` is present
    (no override-driven thermal bounds to check).
    """
    if case.files.expt is None and case.files.manutt is None:
        return {}

    from cobre_bridge.converters.thermal import convert_thermal_bounds

    table = convert_thermal_bounds(case, id_map)
    if table is None:
        return {}

    cols = table.to_pydict()
    result: dict[tuple[int, int, str], float] = {}
    for thermal_id, stage_id, min_mw, max_mw in zip(
        cols["thermal_id"],
        cols["stage_id"],
        cols["min_generation_mw"],
        cols["max_generation_mw"],
    ):
        result[(int(thermal_id), int(stage_id), "generation_min")] = float(min_mw)
        result[(int(thermal_id), int(stage_id), "generation_max")] = float(max_mw)
    return result


def compute_line_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Per-stage line flow bounds ``{(line_id, stage_id, bound_name): value}``.

    Delegates to :func:`cobre_bridge.converters.network.convert_line_bounds`
    (``direct_mw`` -> ``direct_flow_max``, ``reverse_mw`` -> ``reverse_flow_max``).
    Since :mod:`comparators.bounds` also reads the converter's
    ``line_bounds.parquet`` as the "actual" side, this is a recompute-vs-on-disk
    consistency check.

    ``convert_line_bounds`` now also emits per-block override rows
    (``block_id`` set) alongside the stage-level base row (``block_id is
    None``). Only the base rows are used
    here: this dict is keyed by ``(line_id, stage_id)`` with no block
    dimension, so folding block rows in would let a later row silently
    overwrite the base capacity for any stage with a differing block, rather
    than raising or comparing per block. Block-level fidelity is not yet this
    check's job.
    """
    from cobre_bridge.converters.network import convert_line_bounds

    table = convert_line_bounds(case, id_map)
    base_rows = table.filter(table["block_id"].is_null())
    cols = base_rows.to_pydict()
    result: dict[tuple[int, int, str], float] = {}
    for line_id, stage_id, direct_mw, reverse_mw in zip(
        cols["line_id"],
        cols["stage_id"],
        cols["direct_mw"],
        cols["reverse_mw"],
    ):
        result[(int(line_id), int(stage_id), "direct_flow_max")] = float(direct_mw)
        result[(int(line_id), int(stage_id), "reverse_flow_max")] = float(reverse_mw)
    return result
