"""Core bounds comparison between NEWAVE input-derived bounds and Cobre bounds.parquet.

Computes NEWAVE bounds from input files via ``bounds_from_inputs`` and
compares them against Cobre's ``bounds.parquet``, producing a list of
BoundComparison results for every (entity, stage, variable) triple.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cobre_bridge.comparators.alignment import (
    EntityAlignment,
    HydroEntity,
    LineEntity,
    ThermalEntity,
)
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundComparison:
    """Single bound comparison result."""

    entity_type: str
    entity_name: str
    newave_code: int
    cobre_id: int
    stage: int
    variable: str
    newave_value: float
    cobre_value: float
    diff: float
    match: bool


# Mapping from computed bound_name to Cobre bound_type_code.
_BOUND_NAME_TO_CODE: dict[str, int] = {
    "storage_min": 0,
    "storage_max": 1,
    "turbined_min": 2,
    "turbined_max": 3,
    "outflow_min": 4,
    "generation_min": 6,
    "generation_max": 7,
    "reverse_flow_max": 8,
    "direct_flow_max": 9,
}

# Cobre entity_type_code for each entity class.
_ENTITY_TYPE_HYDRO = 0
_ENTITY_TYPE_THERMAL = 1
_ENTITY_TYPE_LINE = 3

# NEWAVE uses 99999 as a "big M" sentinel meaning "no limit".
_NEWAVE_BIG_M = 99990.0


def _is_effectively_infinite(value: float) -> bool:
    """Return True if the value represents an unbounded variable.

    Catches both IEEE inf and NEWAVE's 99999 sentinel.
    """
    return math.isinf(value) or abs(value) >= _NEWAVE_BIG_M


def _bounds_match(a: float, b: float, tolerance: float) -> bool:
    """Check if two bound values match within tolerance.

    Both-infinite with same sign counts as a match.
    One finite and one infinite is always a mismatch.
    """
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= tolerance


def _read_cobre_bounds(
    cobre_output_dir: Path,
) -> dict[tuple[int, int, int, int], float]:
    """Read Cobre bounds.parquet.

    Returns {(entity_type, entity_id, stage_id, bound_type): value}.

    Only reads rows with block_id IS NULL (stage-level bounds).
    """
    path = cobre_output_dir / "training" / "dictionaries" / "bounds.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cobre bounds.parquet not found at {path}. Run cobre with --output first."
        )

    df = pl.read_parquet(path)
    df = df.filter(pl.col("block_id").is_null())

    result: dict[tuple[int, int, int, int], float] = {}
    for row in df.iter_rows(named=True):
        key = (
            int(row["entity_type_code"]),
            int(row["entity_id"]),
            int(row["stage_id"]),
            int(row["bound_type_code"]),
        )
        result[key] = float(row["bound_value"])

    return result


def _make_comparison(
    entity_type: str,
    entity_name: str,
    newave_code: int,
    cobre_id: int,
    stage: int,
    variable: str,
    nw_value: float,
    cobre_value: float,
    tolerance: float,
) -> BoundComparison:
    """Build a BoundComparison with computed diff and match."""
    if math.isinf(nw_value) or math.isinf(cobre_value):
        diff = float("inf")
    else:
        diff = abs(nw_value - cobre_value)
    matched = _bounds_match(nw_value, cobre_value, tolerance)
    return BoundComparison(
        entity_type=entity_type,
        entity_name=entity_name,
        newave_code=newave_code,
        cobre_id=cobre_id,
        stage=stage,
        variable=variable,
        newave_value=nw_value,
        cobre_value=cobre_value,
        diff=diff,
        match=matched,
    )


def _compare_hydros(
    computed: dict[tuple[int, int, str], float],
    cobre_bounds: dict[tuple[int, int, int, int], float],
    hydro_lookup: dict[int, HydroEntity],
    tolerance: float,
    variables: set[str] | None,
) -> list[BoundComparison]:
    """Compare hydro bounds using computed NEWAVE bounds."""
    results: list[BoundComparison] = []

    for (hydro_id, stage_id, bound_name), nw_value in sorted(computed.items()):
        if variables is not None and bound_name not in variables:
            continue
        if _is_effectively_infinite(nw_value):
            continue

        cobre_bt = _BOUND_NAME_TO_CODE.get(bound_name)
        if cobre_bt is None:
            continue

        cobre_key = (_ENTITY_TYPE_HYDRO, hydro_id, stage_id, cobre_bt)
        cobre_value = cobre_bounds.get(cobre_key)
        if cobre_value is None:
            continue

        entity = hydro_lookup.get(hydro_id)
        name = entity.name if entity else f"hydro_{hydro_id}"
        nw_code = entity.newave_code if entity else 0

        results.append(
            _make_comparison(
                "hydro",
                name,
                nw_code,
                hydro_id,
                stage_id,
                bound_name,
                nw_value,
                cobre_value,
                tolerance,
            )
        )

    return results


def _compare_thermals(
    computed: dict[tuple[int, int, str], float],
    cobre_bounds: dict[tuple[int, int, int, int], float],
    thermal_lookup: dict[int, ThermalEntity],
    tolerance: float,
    variables: set[str] | None,
) -> list[BoundComparison]:
    """Compare thermal bounds using computed NEWAVE bounds."""
    results: list[BoundComparison] = []

    for (thermal_id, stage_id, bound_name), nw_value in sorted(computed.items()):
        if variables is not None and bound_name not in variables:
            continue
        if _is_effectively_infinite(nw_value):
            continue

        cobre_bt = _BOUND_NAME_TO_CODE.get(bound_name)
        if cobre_bt is None:
            continue

        cobre_key = (_ENTITY_TYPE_THERMAL, thermal_id, stage_id, cobre_bt)
        cobre_value = cobre_bounds.get(cobre_key)
        if cobre_value is None:
            continue

        entity = thermal_lookup.get(thermal_id)
        name = entity.name if entity else f"thermal_{thermal_id}"
        nw_code = entity.newave_code if entity else 0

        results.append(
            _make_comparison(
                "thermal",
                name,
                nw_code,
                thermal_id,
                stage_id,
                bound_name,
                nw_value,
                cobre_value,
                tolerance,
            )
        )

    return results


def _compare_lines(
    computed: dict[tuple[int, int, str], float],
    cobre_bounds: dict[tuple[int, int, int, int], float],
    line_lookup: dict[int, LineEntity],
    tolerance: float,
    variables: set[str] | None,
) -> list[BoundComparison]:
    """Compare line flow bounds using computed NEWAVE bounds."""
    results: list[BoundComparison] = []

    for (line_id, stage_id, bound_name), nw_value in sorted(computed.items()):
        if variables is not None and bound_name not in variables:
            continue
        if _is_effectively_infinite(nw_value):
            continue

        cobre_bt = _BOUND_NAME_TO_CODE.get(bound_name)
        if cobre_bt is None:
            continue

        cobre_key = (_ENTITY_TYPE_LINE, line_id, stage_id, cobre_bt)
        cobre_value = cobre_bounds.get(cobre_key)
        if cobre_value is None:
            continue

        entity = line_lookup.get(line_id)
        name = entity.name if entity else f"line_{line_id}"
        nw_code = entity.newave_de if entity else 0

        results.append(
            _make_comparison(
                "line",
                name,
                nw_code,
                line_id,
                stage_id,
                bound_name,
                nw_value,
                cobre_value,
                tolerance,
            )
        )

    return results


def compare_bounds(
    alignment: EntityAlignment,
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
    cobre_output_dir: Path,
    tolerance: float = 1e-3,
    variables: set[str] | None = None,
) -> list[BoundComparison]:
    """Compare LP variable bounds between NEWAVE and Cobre.

    Computes NEWAVE bounds from input files (via ``bounds_from_inputs``)
    and compares them against Cobre's ``bounds.parquet``.

    Parameters
    ----------
    alignment:
        Pre-built entity alignment.
    nw_files:
        Resolved NEWAVE input file paths.
    id_map:
        Entity ID mapping (NEWAVE codes to Cobre IDs).
    cobre_output_dir:
        Path to Cobre output directory.
    tolerance:
        Absolute tolerance for bound comparison.
    variables:
        Optional set of variable names to include. None means all.

    Returns
    -------
    list[BoundComparison]
        All comparison results, including matches and mismatches.
    """
    from cobre_bridge.comparators.bounds_from_inputs import (
        compute_hydro_bounds,
        compute_line_bounds,
        compute_thermal_bounds,
    )

    _LOG.info(
        "Comparing bounds: %d hydros, %d thermals, %d lines, %d stages, tol=%g",
        len(alignment.hydros),
        len(alignment.thermals),
        len(alignment.lines),
        alignment.num_newave_stages,
        tolerance,
    )

    # Compute NEWAVE bounds from input files.
    _LOG.info("Computing NEWAVE bounds from input files...")
    computed_hydro = compute_hydro_bounds(nw_files, id_map)
    computed_thermal = compute_thermal_bounds(nw_files, id_map)
    computed_line = compute_line_bounds(nw_files, id_map)
    _LOG.info(
        "Computed: %d hydro entries, %d thermal entries, %d line entries",
        len(computed_hydro),
        len(computed_thermal),
        len(computed_line),
    )

    # Read Cobre bounds.
    _LOG.info("Reading Cobre bounds...")
    cobre_bounds = _read_cobre_bounds(cobre_output_dir)
    _LOG.info("Cobre: %d bound entries", len(cobre_bounds))

    # Build reverse lookups: cobre_id -> entity.
    hydro_lookup: dict[int, HydroEntity] = {h.cobre_id: h for h in alignment.hydros}
    thermal_lookup: dict[int, ThermalEntity] = {
        t.cobre_id: t for t in alignment.thermals
    }
    line_lookup: dict[int, LineEntity] = {
        ln.cobre_line_id: ln for ln in alignment.lines
    }

    results: list[BoundComparison] = []

    _LOG.info("Comparing hydro bounds...")
    results.extend(
        _compare_hydros(
            computed_hydro, cobre_bounds, hydro_lookup, tolerance, variables
        )
    )

    _LOG.info("Comparing thermal bounds...")
    results.extend(
        _compare_thermals(
            computed_thermal, cobre_bounds, thermal_lookup, tolerance, variables
        )
    )

    _LOG.info("Comparing line bounds...")
    results.extend(
        _compare_lines(computed_line, cobre_bounds, line_lookup, tolerance, variables)
    )

    return results
