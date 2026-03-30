"""Results comparison engine: aligns NEWAVE output with Cobre simulation means.

Reads NEWAVE MEDIAS / pmo.dat output and Cobre simulation parquets,
aligns entities via ``EntityAlignment``, and computes per-variable
absolute and relative differences.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from cobre_bridge.comparators.alignment import EntityAlignment
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResultComparison:
    """Single variable comparison result for results comparison."""

    entity_type: str  # "hydro", "thermal", "bus", "convergence", "productivity"
    entity_name: str
    newave_code: int
    cobre_id: int
    stage: int  # 0-indexed (Cobre convention)
    variable: str
    newave_value: float
    cobre_value: float
    abs_diff: float
    rel_diff: float | None  # None when newave_value == 0


@dataclass
class ResultVariableStats:
    """Per-variable comparison statistics."""

    count: int = 0
    mean_abs_diff: float = 0.0
    max_abs_diff: float = 0.0
    mean_rel_diff: float = 0.0
    max_rel_diff: float = 0.0
    correlation: float = 0.0


@dataclass
class ResultsSummary:
    """Aggregate results comparison statistics."""

    total: int = 0
    by_entity_type: dict[str, int] = field(default_factory=dict)
    by_variable: dict[str, ResultVariableStats] = field(default_factory=dict)


def _compute_diff(nw_value: float, cobre_value: float) -> tuple[float, float | None]:
    """Compute absolute and relative diff."""
    abs_diff = abs(nw_value - cobre_value)
    rel_diff: float | None = None
    if abs(nw_value) > 1e-10:
        rel_diff = abs_diff / abs(nw_value)
    return abs_diff, rel_diff


def _make_result(
    entity_type: str,
    entity_name: str,
    newave_code: int,
    cobre_id: int,
    stage: int,
    variable: str,
    nw_value: float,
    cobre_value: float,
) -> ResultComparison:
    """Build a ResultComparison with computed diffs."""
    abs_diff, rel_diff = _compute_diff(nw_value, cobre_value)
    return ResultComparison(
        entity_type=entity_type,
        entity_name=entity_name,
        newave_code=newave_code,
        cobre_id=cobre_id,
        stage=stage,
        variable=variable,
        newave_value=nw_value,
        cobre_value=cobre_value,
        abs_diff=abs_diff,
        rel_diff=rel_diff,
    )


# MEDIAS variable name -> our standard variable name.
_HYDRO_VAR_MAP: dict[str, str] = {
    "VARMUH": "storage_final_hm3",
    "GHIDUH": "generation_mw",
    "QTURUH": "turbined_m3s",
    "QVERTUH": "spillage_m3s",
    "QAFLUH": "inflow_m3s",
    "PIVARM": "water_value_per_hm3",
}

_SYSTEM_VAR_MAP: dict[str, str] = {
    "CMO": "spot_price",
    "DEFT": "deficit_mw",
}


def _compare_hydros(
    alignment: EntityAlignment,
    nw_hydro: pl.DataFrame,
    cobre_hydro: pl.DataFrame,
) -> list[ResultComparison]:
    """Compare hydro results."""
    results: list[ResultComparison] = []

    # Build lookup: (newave_code, stage_0based, variable) -> nw_value
    nw_lookup: dict[tuple[int, int, str], float] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        stage = int(row["stage"]) - 1  # Convert 1-indexed to 0-indexed.
        var = str(row["variable"]).strip().upper()
        mapped = _HYDRO_VAR_MAP.get(var, var.lower())
        nw_lookup[(code, stage, mapped)] = float(row["value"])

    # Build lookup: (entity_id, stage_id, variable) -> cobre_value
    cobre_lookup: dict[tuple[int, int, str], float] = {}
    hydro_value_cols = [
        "storage_final_hm3",
        "generation_mw",
        "turbined_m3s",
        "spillage_m3s",
        "inflow_m3s",
        "water_value_per_hm3",
    ]
    for row in cobre_hydro.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        for col in hydro_value_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                cobre_lookup[(eid, sid, col)] = round(float(val), 2)

    # Compare.
    for hydro in alignment.hydros:
        for (code, stage, var), nw_val in sorted(nw_lookup.items()):
            if code != hydro.newave_code:
                continue
            cobre_key = (hydro.cobre_id, stage, var)
            cobre_val = cobre_lookup.get(cobre_key)
            if cobre_val is None:
                continue
            results.append(
                _make_result(
                    "hydro",
                    hydro.name,
                    hydro.newave_code,
                    hydro.cobre_id,
                    stage,
                    var,
                    nw_val,
                    cobre_val,
                )
            )

    return results


def _compare_thermals(
    alignment: EntityAlignment,
    nw_thermal: pl.DataFrame,
    cobre_thermal: pl.DataFrame,
) -> list[ResultComparison]:
    """Compare thermal results."""
    results: list[ResultComparison] = []

    nw_lookup: dict[tuple[int, int], float] = {}
    for row in nw_thermal.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        stage = int(row["stage"]) - 1
        nw_lookup[(code, stage)] = float(row["value"])

    cobre_lookup: dict[tuple[int, int], float] = {}
    for row in cobre_thermal.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        val = row.get("generation_mw")
        if val is not None:
            cobre_lookup[(eid, sid)] = round(float(val), 2)

    for thermal in alignment.thermals:
        for (code, stage), nw_val in sorted(nw_lookup.items()):
            if code != thermal.newave_code:
                continue
            cobre_val = cobre_lookup.get((thermal.cobre_id, stage))
            if cobre_val is None:
                continue
            results.append(
                _make_result(
                    "thermal",
                    thermal.name,
                    thermal.newave_code,
                    thermal.cobre_id,
                    stage,
                    "generation_mw",
                    nw_val,
                    cobre_val,
                )
            )

    return results


def _compare_buses(
    alignment: EntityAlignment,
    nw_system: pl.DataFrame,
    cobre_bus: pl.DataFrame,
    id_map: NewaveIdMap,
) -> list[ResultComparison]:
    """Compare bus/subsystem results."""
    results: list[ResultComparison] = []

    # Build lookup: (newave_code, stage, variable) -> value
    nw_lookup: dict[tuple[int, int, str], float] = {}
    for row in nw_system.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        stage = int(row["stage"]) - 1
        var = str(row["variable"]).strip().upper()
        mapped = _SYSTEM_VAR_MAP.get(var, var.lower())
        nw_lookup[(code, stage, mapped)] = float(row["value"])

    cobre_lookup: dict[tuple[int, int, str], float] = {}
    for row in cobre_bus.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        for col in ("spot_price", "deficit_mw"):
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                cobre_lookup[(eid, sid, col)] = round(float(val), 2)

    # Build subsystem code -> bus_id mapping.
    for nw_code in id_map.all_bus_ids:
        bus_id = id_map.bus_id(nw_code)
        bus_name = f"bus_{bus_id}"

        for (code, stage, var), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            cobre_val = cobre_lookup.get((bus_id, stage, var))
            if cobre_val is None:
                continue
            results.append(
                _make_result(
                    "bus",
                    bus_name,
                    nw_code,
                    bus_id,
                    stage,
                    var,
                    nw_val,
                    cobre_val,
                )
            )

    return results


def _compare_convergence(
    nw_conv: pl.DataFrame,
    cobre_conv: pl.DataFrame,
) -> list[ResultComparison]:
    """Compare convergence data."""
    results: list[ResultComparison] = []

    nw_lookup: dict[int, dict[str, float]] = {}
    for row in nw_conv.iter_rows(named=True):
        it = int(row["iteration"])
        nw_lookup[it] = {
            "lower_bound": float(row["lower_bound"]),
            "upper_bound_mean": float(row["upper_bound_mean"]),
        }

    cobre_lookup: dict[int, dict[str, float]] = {}
    for row in cobre_conv.iter_rows(named=True):
        it = int(row["iteration"])
        cobre_lookup[it] = {
            "lower_bound": float(row["lower_bound"]),
            "upper_bound_mean": float(row["upper_bound_mean"]),
        }

    for it in sorted(set(nw_lookup) & set(cobre_lookup)):
        for var in ("lower_bound", "upper_bound_mean"):
            nw_val = nw_lookup[it][var]
            cobre_val = cobre_lookup[it][var]
            results.append(
                _make_result(
                    "convergence",
                    f"iteration_{it}",
                    it,
                    it,
                    it,
                    var,
                    nw_val,
                    cobre_val,
                )
            )

    return results


def _compare_productivity(
    alignment: EntityAlignment,
    nw_prod: pl.DataFrame,
    cobre_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare hydro productivity values."""
    results: list[ResultComparison] = []

    # nw_prod has plant_name (str) and productivity (float).
    # Build lookup by uppercased name for fuzzy matching.
    nw_lookup: dict[str, float] = {}
    for row in nw_prod.iter_rows(named=True):
        name = str(row["plant_name"]).strip().upper()
        nw_lookup[name] = float(row["productivity"])

    for hydro in alignment.hydros:
        # Try matching by name (uppercased).
        nw_val = nw_lookup.get(hydro.name.strip().upper())
        if nw_val is None:
            continue
        meta = cobre_meta.get(hydro.cobre_id)
        if meta is None:
            continue
        cobre_val = meta.get("productivity_mw_per_m3s")
        if cobre_val is None:
            continue
        results.append(
            _make_result(
                "productivity",
                hydro.name,
                hydro.newave_code,
                hydro.cobre_id,
                0,
                "productivity",
                nw_val,
                float(cobre_val),
            )
        )

    return results


def compare_results(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
    alignment: EntityAlignment,
    cobre_output_dir: Path,
    tolerance: float = 1e-2,
) -> list[ResultComparison]:
    """Compare NEWAVE output results against Cobre simulation means.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE input file paths (for locating pmo.dat).
    id_map:
        Entity ID mapping.
    alignment:
        Pre-built entity alignment.
    cobre_output_dir:
        Path to Cobre output directory.
    tolerance:
        Relative tolerance for results comparison (informational).

    Returns
    -------
    list[ResultComparison]
        All comparison results across hydro, thermal, bus, convergence,
        and productivity dimensions.
    """
    from cobre_bridge.comparators.cobre_readers import (
        read_cobre_bus_means,
        read_cobre_convergence,
        read_cobre_hydro_means,
        read_cobre_hydro_metadata,
        read_cobre_thermal_means,
    )
    from cobre_bridge.comparators.newave_readers import (
        _find_saidas_dir,
        read_medias_hydro,
        read_medias_system,
        read_medias_thermal,
        read_pmo_convergence,
        read_pmo_productivity,
    )

    results: list[ResultComparison] = []

    # Locate NEWAVE saidas directory.
    saidas_dir = _find_saidas_dir(nw_files.directory)

    # --- Hydro comparison ---
    if saidas_dir is not None:
        nw_hydro = read_medias_hydro(saidas_dir)
        cobre_hydro = read_cobre_hydro_means(cobre_output_dir)
        if not nw_hydro.is_empty() and not cobre_hydro.is_empty():
            _LOG.info("Comparing hydro results...")
            results.extend(_compare_hydros(alignment, nw_hydro, cobre_hydro))

        # --- Thermal comparison ---
        nw_thermal = read_medias_thermal(saidas_dir)
        cobre_thermal = read_cobre_thermal_means(cobre_output_dir)
        if not nw_thermal.is_empty() and not cobre_thermal.is_empty():
            _LOG.info("Comparing thermal results...")
            results.extend(_compare_thermals(alignment, nw_thermal, cobre_thermal))

        # --- Bus/system comparison ---
        nw_system = read_medias_system(saidas_dir)
        cobre_bus = read_cobre_bus_means(cobre_output_dir)
        if not nw_system.is_empty() and not cobre_bus.is_empty():
            _LOG.info("Comparing bus results...")
            results.extend(_compare_buses(alignment, nw_system, cobre_bus, id_map))
    else:
        _LOG.warning("NEWAVE saidas/ directory not found; skipping MEDIAS comparison.")

    # --- Convergence comparison ---
    nw_conv = read_pmo_convergence(nw_files.directory)
    cobre_conv = read_cobre_convergence(cobre_output_dir)
    if not nw_conv.is_empty() and not cobre_conv.is_empty():
        _LOG.info("Comparing convergence data...")
        results.extend(_compare_convergence(nw_conv, cobre_conv))

    # --- Productivity comparison ---
    nw_prod = read_pmo_productivity(nw_files.directory)
    cobre_meta = read_cobre_hydro_metadata(cobre_output_dir)
    if not nw_prod.is_empty() and cobre_meta:
        _LOG.info("Comparing productivity data...")
        results.extend(_compare_productivity(alignment, nw_prod, cobre_meta))

    _LOG.info("Results comparison: %d total comparisons", len(results))
    return results


def build_results_summary(results: list[ResultComparison]) -> ResultsSummary:
    """Compute aggregate statistics from comparison results."""
    summary = ResultsSummary(total=len(results))

    # Group by entity type.
    for r in results:
        summary.by_entity_type[r.entity_type] = (
            summary.by_entity_type.get(r.entity_type, 0) + 1
        )

    # Group by variable for stats.
    var_groups: dict[str, list[ResultComparison]] = {}
    for r in results:
        var_groups.setdefault(r.variable, []).append(r)

    for var, group in var_groups.items():
        stats = ResultVariableStats(count=len(group))

        abs_diffs = [r.abs_diff for r in group]
        rel_diffs = [r.rel_diff for r in group if r.rel_diff is not None]

        stats.mean_abs_diff = sum(abs_diffs) / len(abs_diffs) if abs_diffs else 0.0
        stats.max_abs_diff = max(abs_diffs) if abs_diffs else 0.0
        stats.mean_rel_diff = sum(rel_diffs) / len(rel_diffs) if rel_diffs else 0.0
        stats.max_rel_diff = max(rel_diffs) if rel_diffs else 0.0

        # Pearson correlation.
        nw_vals = [r.newave_value for r in group]
        cb_vals = [r.cobre_value for r in group]
        if len(nw_vals) > 1:
            stats.correlation = _pearson(nw_vals, cb_vals)

        summary.by_variable[var] = stats

    return summary


def _pearson(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-15:
        return 0.0
    return cov / denom
