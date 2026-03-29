"""Cobre simulation output readers for results comparison.

Reads Cobre simulation parquets using Polars lazy scanning and
streaming aggregation to compute scenario means matching the NEWAVE
MEDIAS aggregation level.  Also reads convergence data and hydro
metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

_LOG = logging.getLogger(__name__)


def _scan_simulation_entity(
    cobre_output_dir: Path,
    entity: str,
) -> pl.LazyFrame | None:
    """Scan hive-partitioned simulation parquets for *entity*.

    Returns a LazyFrame or None if the directory does not exist.

    Parameters
    ----------
    cobre_output_dir:
        Path to the Cobre ``output/`` directory.
    entity:
        Entity subdirectory name (e.g., ``"hydros"``, ``"thermals"``,
        ``"buses"``).
    """
    sim_dir = cobre_output_dir / "simulation" / entity
    if not sim_dir.is_dir():
        _LOG.warning("Simulation directory not found: %s", sim_dir)
        return None

    pattern = sim_dir / "**/*.parquet"
    try:
        lf = pl.scan_parquet(pattern, hive_partitioning=True)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to scan parquets in %s", sim_dir)
        return None

    return lf


def read_cobre_hydro_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre hydro simulation means per (entity_id, stage_id).

    Scans ``output/simulation/hydros/`` with Polars streaming, filters
    for ``block_id == 0``, and computes scenario means.

    Returns DataFrame with columns: ``entity_id``, ``stage_id``,
    ``storage_final_hm3``, ``generation_mw``, ``turbined_m3s``,
    ``spillage_m3s``, ``inflow_m3s``, ``water_value_per_hm3``.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "storage_final_hm3": pl.Float64,
            "generation_mw": pl.Float64,
            "turbined_m3s": pl.Float64,
            "spillage_m3s": pl.Float64,
            "inflow_m3s": pl.Float64,
            "water_value_per_hm3": pl.Float64,
        }
    )

    lf = _scan_simulation_entity(cobre_output_dir, "hydros")
    if lf is None:
        return empty

    value_cols = [
        "storage_final_hm3",
        "generation_mw",
        "turbined_m3s",
        "spillage_m3s",
        "inflow_m3s",
        "water_value_per_hm3",
    ]

    # Filter for available columns only.
    available = set(lf.collect_schema().names())
    agg_cols = [c for c in value_cols if c in available]

    if not agg_cols:
        _LOG.warning("No recognized value columns in hydros simulation")
        return empty

    try:
        result = (
            lf.filter(pl.col("block_id") == 0)
            .group_by("entity_id", "stage_id")
            .agg([pl.col(c).mean() for c in agg_cols])
            .sort("entity_id", "stage_id")
            .collect(engine="streaming")
        )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to aggregate hydro simulation data")
        return empty

    # Add missing columns with null.
    for col in value_cols:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result


def read_cobre_thermal_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre thermal simulation means per (entity_id, stage_id).

    Returns DataFrame with columns: ``entity_id``, ``stage_id``,
    ``generation_mw``.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "generation_mw": pl.Float64,
        }
    )

    lf = _scan_simulation_entity(cobre_output_dir, "thermals")
    if lf is None:
        return empty

    available = set(lf.collect_schema().names())
    if "generation_mw" not in available:
        _LOG.warning("generation_mw column not found in thermals simulation")
        return empty

    try:
        result = (
            lf.filter(pl.col("block_id") == 0)
            .group_by("entity_id", "stage_id")
            .agg(pl.col("generation_mw").mean())
            .sort("entity_id", "stage_id")
            .collect(engine="streaming")
        )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to aggregate thermal simulation data")
        return empty

    return result


def read_cobre_bus_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre bus simulation means per (entity_id, stage_id).

    Returns DataFrame with columns: ``entity_id``, ``stage_id``,
    ``spot_price``, ``deficit_mw``.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "spot_price": pl.Float64,
            "deficit_mw": pl.Float64,
        }
    )

    lf = _scan_simulation_entity(cobre_output_dir, "buses")
    if lf is None:
        return empty

    available = set(lf.collect_schema().names())
    value_cols = [c for c in ("spot_price", "deficit_mw") if c in available]

    if not value_cols:
        _LOG.warning("No recognized value columns in buses simulation")
        return empty

    try:
        result = (
            lf.filter(pl.col("block_id") == 0)
            .group_by("entity_id", "stage_id")
            .agg([pl.col(c).mean() for c in value_cols])
            .sort("entity_id", "stage_id")
            .collect(engine="streaming")
        )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to aggregate bus simulation data")
        return empty

    for col in ("spot_price", "deficit_mw"):
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result


def read_cobre_convergence(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre convergence data from training output.

    Returns DataFrame with columns: ``iteration`` (Int64),
    ``lower_bound`` (Float64), ``upper_bound_mean`` (Float64).
    """
    empty = pl.DataFrame(
        schema={
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        }
    )

    conv_path = cobre_output_dir / "training" / "convergence.parquet"
    if not conv_path.exists():
        _LOG.warning("convergence.parquet not found at %s", conv_path)
        return empty

    try:
        df = pl.read_parquet(conv_path)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read convergence.parquet")
        return empty

    # Map columns to standard names.
    col_map: dict[str, str] = {}
    for col in df.columns:
        lower = col.lower()
        if "iteration" in lower or "iter" in lower:
            col_map[col] = "iteration"
        elif "lower" in lower or "zinf" in lower:
            col_map[col] = "lower_bound"
        elif "upper" in lower or "zsup" in lower:
            col_map[col] = "upper_bound_mean"

    if "iteration" not in col_map.values():
        # Use row index as iteration.
        df = df.with_row_index("iteration")
        col_map["iteration"] = "iteration"

    if (
        "lower_bound" not in col_map.values()
        or "upper_bound_mean" not in col_map.values()
    ):
        _LOG.warning(
            "Cannot identify bound columns in convergence.parquet: %s",
            df.columns,
        )
        return empty

    inv_map = {v: k for k, v in col_map.items()}
    result = df.select(
        pl.col(inv_map["iteration"]).cast(pl.Int64).alias("iteration"),
        pl.col(inv_map["lower_bound"]).cast(pl.Float64).alias("lower_bound"),
        pl.col(inv_map["upper_bound_mean"]).cast(pl.Float64).alias("upper_bound_mean"),
    )

    return result


def read_cobre_hydro_metadata(cobre_output_dir: Path) -> dict[int, dict]:
    """Read hydro metadata from Cobre hydros.json.

    Looks for ``system/hydros.json`` in the Cobre case directory
    (parent of output dir).

    Returns ``{entity_id: {"name": str, "productivity_mw_per_m3s": float}}``.
    """
    # The case dir is the parent of the output dir.
    case_dir = cobre_output_dir.parent
    hydros_path = case_dir / "system" / "hydros.json"

    if not hydros_path.exists():
        # Try the output dir's parent's parent.
        for candidate in [cobre_output_dir, case_dir.parent]:
            p = candidate / "system" / "hydros.json"
            if p.exists():
                hydros_path = p
                break

    if not hydros_path.exists():
        _LOG.warning("hydros.json not found near %s", cobre_output_dir)
        return {}

    try:
        with hydros_path.open() as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to parse hydros.json")
        return {}

    result: dict[int, dict] = {}
    for hydro in data.get("hydros", []):
        entity_id = int(hydro["id"])
        name = str(hydro.get("name", f"hydro_{entity_id}"))

        # Productivity may be in generation.productivity_mw_per_m3s or
        # a top-level field. Check common locations.
        prod = None
        gen = hydro.get("generation", {})
        if isinstance(gen, dict):
            prod = gen.get("productivity_mw_per_m3s")
        if prod is None:
            prod = hydro.get("productivity_mw_per_m3s")

        result[entity_id] = {
            "name": name,
            "productivity_mw_per_m3s": float(prod) if prod is not None else None,
        }

    return result
