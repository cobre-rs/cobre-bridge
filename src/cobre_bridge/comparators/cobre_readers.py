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


def _load_block_hours(cobre_output_dir: Path) -> pl.DataFrame | None:
    """Load block hours from stages.json as a Polars DataFrame.

    Returns DataFrame with columns ``stage_id``, ``block_id``, ``hours``
    or None if stages.json cannot be found or parsed.
    """
    case_dir = cobre_output_dir.parent
    for candidate in [case_dir, cobre_output_dir]:
        p = candidate / "stages.json"
        if p.exists():
            try:
                with p.open() as f:
                    data = json.load(f)
                rows: list[dict] = []
                for stage in data.get("stages", []):
                    for block in stage.get("blocks", []):
                        rows.append(
                            {
                                "stage_id": stage["id"],
                                "block_id": block["id"],
                                "hours": block["hours"],
                            }
                        )
                if rows:
                    return pl.DataFrame(rows).cast(
                        {
                            "stage_id": pl.Int32,
                            "block_id": pl.Int32,
                            "hours": pl.Float64,
                        }
                    )
            except Exception:  # noqa: BLE001
                pass
    return None


def _weighted_stage_mean(
    lf: pl.LazyFrame,
    id_col: str,
    value_cols: list[str],
    block_hours: pl.DataFrame,
    stage_level_cols: list[str] | None = None,
) -> pl.LazyFrame:
    """Compute block-hours-weighted stage mean for flow variables.

    For columns in *value_cols*, computes:
        stage_avg = Σ(value × hours) / Σ(hours)
    averaged across scenarios.

    For columns in *stage_level_cols* (same across all blocks), takes the
    value from any block (first) and averages across scenarios.
    """
    bh = block_hours.lazy()
    stage_level_cols = stage_level_cols or []

    weighted_aggs = [
        (pl.col(c) * pl.col("hours")).sum().alias(f"_{c}_wsum") for c in value_cols
    ]
    hour_sum = pl.col("hours").sum().alias("_total_hours")
    stage_aggs = [pl.col(c).first().alias(c) for c in stage_level_cols]

    # Per scenario+entity+stage: weighted sum across blocks.
    per_scenario = (
        lf.join(bh, on=["stage_id", "block_id"])
        .group_by(["scenario_id", id_col, "stage_id"])
        .agg(weighted_aggs + [hour_sum] + stage_aggs)
    )

    # Compute weighted mean per scenario, then mean across scenarios.
    with_means = per_scenario
    for c in value_cols:
        with_means = with_means.with_columns(
            (pl.col(f"_{c}_wsum") / pl.col("_total_hours")).alias(c)
        )

    drop_cols = [f"_{c}_wsum" for c in value_cols] + ["_total_hours"]
    final_aggs = [pl.col(c).mean() for c in value_cols + stage_level_cols]

    return with_means.drop(drop_cols).group_by(id_col, "stage_id").agg(final_aggs)


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

    Scans ``output/simulation/hydros/`` with Polars streaming and computes
    block-hours-weighted scenario means for flow variables (generation,
    turbined, spillage) and plain scenario means for stage-level variables
    (storage, inflow, water value).

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

    flow_cols = ["generation_mw", "turbined_m3s", "spillage_m3s"]
    stage_cols = ["storage_final_hm3", "inflow_m3s", "water_value_per_hm3"]

    available = set(lf.collect_schema().names())
    id_col = "hydro_id" if "hydro_id" in available else "entity_id"

    avail_flow = [c for c in flow_cols if c in available]
    avail_stage = [c for c in stage_cols if c in available]

    if not avail_flow and not avail_stage:
        _LOG.warning("No recognized value columns in hydros simulation")
        return empty

    block_hours = _load_block_hours(cobre_output_dir)

    try:
        if block_hours is not None and avail_flow:
            result = (
                _weighted_stage_mean(lf, id_col, avail_flow, block_hours, avail_stage)
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
        else:
            result = (
                lf.filter(pl.col("block_id") == 0)
                .group_by(id_col, "stage_id")
                .agg([pl.col(c).mean() for c in avail_flow + avail_stage])
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to aggregate hydro simulation data")
        return empty

    for col in flow_cols + stage_cols:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result


def read_cobre_thermal_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre thermal simulation means per (entity_id, stage_id).

    Computes block-hours-weighted scenario means for generation_mw.

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

    id_col = "thermal_id" if "thermal_id" in available else "entity_id"
    block_hours = _load_block_hours(cobre_output_dir)

    try:
        if block_hours is not None:
            result = (
                _weighted_stage_mean(lf, id_col, ["generation_mw"], block_hours)
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
        else:
            result = (
                lf.filter(pl.col("block_id") == 0)
                .group_by(id_col, "stage_id")
                .agg(pl.col("generation_mw").mean())
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to aggregate thermal simulation data")
        return empty

    return result


def read_cobre_bus_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre bus simulation means per (entity_id, stage_id).

    Computes block-hours-weighted scenario means for spot_price and
    deficit_mw.

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

    id_col = "bus_id" if "bus_id" in available else "entity_id"
    block_hours = _load_block_hours(cobre_output_dir)

    try:
        if block_hours is not None:
            result = (
                _weighted_stage_mean(lf, id_col, value_cols, block_hours)
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
        else:
            result = (
                lf.filter(pl.col("block_id") == 0)
                .group_by(id_col, "stage_id")
                .agg([pl.col(c).mean() for c in value_cols])
                .rename({id_col: "entity_id"})
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


def _weighted_scenario_values(
    lf: pl.LazyFrame,
    id_col: str,
    flow_cols: list[str],
    stage_cols: list[str],
    block_hours: pl.DataFrame | None,
) -> pl.LazyFrame:
    """Compute per-scenario weighted-mean values (one row per scenario+entity+stage).

    Returns LazyFrame with columns: scenario_id, entity_id, stage_id, and
    one column per variable in *flow_cols* + *stage_cols*.
    """
    if block_hours is not None and flow_cols:
        bh = block_hours.lazy()
        w_aggs = [
            (pl.col(c) * pl.col("hours")).sum().alias(f"_{c}_w") for c in flow_cols
        ]
        h_sum = pl.col("hours").sum().alias("_h")
        s_aggs = [pl.col(c).first().alias(c) for c in stage_cols]

        per_sc = (
            lf.join(bh, on=["stage_id", "block_id"])
            .group_by(["scenario_id", id_col, "stage_id"])
            .agg(w_aggs + [h_sum] + s_aggs)
        )
        for c in flow_cols:
            per_sc = per_sc.with_columns((pl.col(f"_{c}_w") / pl.col("_h")).alias(c))
        return per_sc.drop([f"_{c}_w" for c in flow_cols] + ["_h"]).rename(
            {id_col: "entity_id"}
        )

    all_cols = flow_cols + stage_cols
    return (
        lf.filter(pl.col("block_id") == 0)
        .group_by(["scenario_id", id_col, "stage_id"])
        .agg([pl.col(c).mean() for c in all_cols])
        .rename({id_col: "entity_id"})
    )


def _compute_percentiles(
    per_scenario: pl.LazyFrame,
    value_cols: list[str],
) -> pl.DataFrame:
    """Aggregate per-scenario values into p10/p50/p90 per (entity, stage).

    Returns DataFrame with columns: entity_id, stage_id, and for each var
    in *value_cols*: ``{var}_p10``, ``{var}_p50``, ``{var}_p90``.
    """
    aggs: list[pl.Expr] = []
    for c in value_cols:
        aggs.extend(
            [
                pl.col(c).quantile(0.1, interpolation="linear").alias(f"{c}_p10"),
                pl.col(c).quantile(0.5, interpolation="linear").alias(f"{c}_p50"),
                pl.col(c).quantile(0.9, interpolation="linear").alias(f"{c}_p90"),
            ]
        )
    return (
        per_scenario.group_by("entity_id", "stage_id")
        .agg(aggs)
        .sort("entity_id", "stage_id")
        .collect(engine="streaming")
    )


def read_cobre_hydro_percentiles(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre hydro p10/p50/p90 per (entity_id, stage_id).

    Returns DataFrame with columns: entity_id, stage_id, and for each
    hydro variable: ``{var}_p10``, ``{var}_p50``, ``{var}_p90``.
    """
    lf = _scan_simulation_entity(cobre_output_dir, "hydros")
    if lf is None:
        return pl.DataFrame()

    flow_cols = ["generation_mw", "turbined_m3s", "spillage_m3s"]
    stage_cols = ["storage_final_hm3", "inflow_m3s", "water_value_per_hm3"]
    available = set(lf.collect_schema().names())
    id_col = "hydro_id" if "hydro_id" in available else "entity_id"
    avail_flow = [c for c in flow_cols if c in available]
    avail_stage = [c for c in stage_cols if c in available]
    all_vars = avail_flow + avail_stage
    if not all_vars:
        return pl.DataFrame()

    block_hours = _load_block_hours(cobre_output_dir)
    try:
        per_sc = _weighted_scenario_values(
            lf, id_col, avail_flow, avail_stage, block_hours
        )
        return _compute_percentiles(per_sc, all_vars)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to compute hydro percentiles")
        return pl.DataFrame()


def read_cobre_thermal_percentiles(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre thermal p10/p50/p90 per (entity_id, stage_id)."""
    lf = _scan_simulation_entity(cobre_output_dir, "thermals")
    if lf is None:
        return pl.DataFrame()

    available = set(lf.collect_schema().names())
    if "generation_mw" not in available:
        return pl.DataFrame()

    id_col = "thermal_id" if "thermal_id" in available else "entity_id"
    block_hours = _load_block_hours(cobre_output_dir)
    try:
        per_sc = _weighted_scenario_values(
            lf, id_col, ["generation_mw"], [], block_hours
        )
        return _compute_percentiles(per_sc, ["generation_mw"])
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to compute thermal percentiles")
        return pl.DataFrame()


def read_cobre_bus_percentiles(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre bus p10/p50/p90 per (entity_id, stage_id)."""
    lf = _scan_simulation_entity(cobre_output_dir, "buses")
    if lf is None:
        return pl.DataFrame()

    available = set(lf.collect_schema().names())
    id_col = "bus_id" if "bus_id" in available else "entity_id"
    flow_cols = [c for c in ["spot_price", "deficit_mw"] if c in available]
    if not flow_cols:
        return pl.DataFrame()

    block_hours = _load_block_hours(cobre_output_dir)
    try:
        per_sc = _weighted_scenario_values(lf, id_col, flow_cols, [], block_hours)
        return _compute_percentiles(per_sc, flow_cols)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to compute bus percentiles")
        return pl.DataFrame()


def _load_entity_bus_map(
    cobre_output_dir: Path,
    entity: str,
    id_field: str,
) -> dict[int, int]:
    """Load entity_id → bus_id mapping from system JSON."""
    path = _find_system_json(cobre_output_dir, f"{entity}.json")
    if path is None:
        return {}
    try:
        with path.open() as f:
            data = json.load(f)
        return {
            int(e["id"]): int(e["bus_id"])
            for e in data.get(entity, [])
            if "bus_id" in e
        }
    except Exception:  # noqa: BLE001
        return {}


def read_cobre_bus_aggregates(
    cobre_output_dir: Path,
) -> pl.DataFrame:
    """Compute per-bus aggregated simulation percentiles.

    Aggregates hydro generation, thermal generation, NCS generation,
    load, deficit, and excess by bus, then computes p10/p50/p90 across
    scenarios.

    Returns DataFrame with columns: bus_id, stage_id, and for each
    variable: ``{var}_p10``, ``{var}_p50``, ``{var}_p90``.
    """
    block_hours = _load_block_hours(cobre_output_dir)

    # --- Bus-level variables (load, deficit, excess) ---
    bus_lf = _scan_simulation_entity(cobre_output_dir, "buses")
    bus_vars = ["load_mw", "deficit_mw", "excess_mw"]

    # --- Hydro generation aggregated by bus ---
    hydro_bus_map = _load_entity_bus_map(cobre_output_dir, "hydros", "hydro_id")
    hydro_lf = _scan_simulation_entity(cobre_output_dir, "hydros")

    # --- Thermal generation aggregated by bus ---
    thermal_bus_map = _load_entity_bus_map(cobre_output_dir, "thermals", "thermal_id")
    thermal_lf = _scan_simulation_entity(cobre_output_dir, "thermals")

    # --- NCS generation aggregated by bus ---
    ncs_bus_map = _load_entity_bus_map(
        cobre_output_dir, "non_controllable_sources", "non_controllable_id"
    )
    ncs_lf = _scan_simulation_entity(cobre_output_dir, "non_controllables")

    def _agg_entity_by_bus(
        lf: pl.LazyFrame | None,
        id_col: str,
        value_col: str,
        bus_map: dict[int, int],
        out_col: str,
    ) -> pl.DataFrame | None:
        """Aggregate an entity variable by bus with block-hours weighting."""
        if lf is None or not bus_map:
            return None

        available = set(lf.collect_schema().names())
        if value_col not in available:
            return None

        # Map entity to bus.
        mapping = pl.DataFrame(
            {id_col: list(bus_map.keys()), "bus_id": list(bus_map.values())}
        )
        joined = lf.join(mapping.lazy(), on=id_col)

        # Sum generation across entities per bus within each
        # (scenario, stage, block) first, then block-hours weight.
        bus_totals = joined.group_by(
            ["scenario_id", "bus_id", "stage_id", "block_id"]
        ).agg(pl.col(value_col).sum())

        if block_hours is not None:
            bh = block_hours.lazy()
            per_sc = (
                bus_totals.join(bh, on=["stage_id", "block_id"])
                .group_by(["scenario_id", "bus_id", "stage_id"])
                .agg(
                    (pl.col(value_col) * pl.col("hours")).sum().alias("_w"),
                    pl.col("hours").sum().alias("_h"),
                )
                .with_columns((pl.col("_w") / pl.col("_h")).alias(out_col))
                .drop("_w", "_h")
            )
        else:
            per_sc = (
                bus_totals.filter(pl.col("block_id") == 0)
                .group_by(["scenario_id", "bus_id", "stage_id"])
                .agg(pl.col(value_col).sum().alias(out_col))
            )
        return per_sc.collect(engine="streaming")

    # Collect per-scenario frames.
    frames: list[pl.DataFrame] = []
    all_vars: list[str] = []

    # Bus-level variables.
    if bus_lf is not None:
        avail = set(bus_lf.collect_schema().names())
        bus_avail = [c for c in bus_vars if c in avail]
        bid_col = "bus_id" if "bus_id" in avail else "entity_id"
        if bus_avail and block_hours is not None:
            bh = block_hours.lazy()
            bus_sc = (
                bus_lf.join(bh, on=["stage_id", "block_id"])
                .group_by(["scenario_id", bid_col, "stage_id"])
                .agg(
                    [
                        (
                            (pl.col(c) * pl.col("hours")).sum() / pl.col("hours").sum()
                        ).alias(c)
                        for c in bus_avail
                    ]
                )
                .rename({bid_col: "bus_id"})
                .collect(engine="streaming")
            )
        elif bus_avail:
            bus_sc = (
                bus_lf.filter(pl.col("block_id") == 0)
                .group_by(["scenario_id", bid_col, "stage_id"])
                .agg([pl.col(c).mean() for c in bus_avail])
                .rename({bid_col: "bus_id"})
                .collect(engine="streaming")
            )
        else:
            bus_sc = None
        if bus_sc is not None:
            frames.append(bus_sc)
            all_vars.extend(bus_avail)

    # Entity aggregations.
    for lf, id_col, val_col, bmap, out in [
        (hydro_lf, "hydro_id", "generation_mw", hydro_bus_map, "hydro_gen_mw"),
        (
            thermal_lf,
            "thermal_id",
            "generation_mw",
            thermal_bus_map,
            "thermal_gen_mw",
        ),
        (
            ncs_lf,
            "non_controllable_id",
            "generation_mw",
            ncs_bus_map,
            "ncs_gen_mw",
        ),
    ]:
        result = _agg_entity_by_bus(lf, id_col, val_col, bmap, out)
        if result is not None:
            frames.append(result)
            all_vars.append(out)

    if not frames:
        return pl.DataFrame()

    # Join all frames on (scenario_id, bus_id, stage_id).
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(
            f,
            on=["scenario_id", "bus_id", "stage_id"],
            how="full",
            coalesce=True,
        )

    # Compute net load = load - NCS per scenario.
    if "load_mw" in merged.columns and "ncs_gen_mw" in merged.columns:
        merged = merged.with_columns(
            (
                pl.col("load_mw").fill_null(0.0)
                - pl.col("ncs_gen_mw").fill_null(0.0)
            ).alias("net_load_mw")
        )
        all_vars.append("net_load_mw")

    # Compute percentiles across scenarios.
    aggs: list[pl.Expr] = []
    for c in all_vars:
        if c not in merged.columns:
            continue
        aggs.extend(
            [
                pl.col(c).quantile(0.1, interpolation="linear").alias(f"{c}_p10"),
                pl.col(c).quantile(0.5, interpolation="linear").alias(f"{c}_p50"),
                pl.col(c).quantile(0.9, interpolation="linear").alias(f"{c}_p90"),
            ]
        )

    if not aggs:
        return pl.DataFrame()

    return merged.group_by("bus_id", "stage_id").agg(aggs).sort("bus_id", "stage_id")


def read_cobre_cost_breakdown(cobre_output_dir: Path) -> dict[str, float]:
    """Read cost breakdown from Cobre simulation costs entity.

    Returns ``{category: mean_total_R$}`` averaged across scenarios,
    summed across all stages and blocks.  Zero-cost categories are excluded.
    """
    lf = _scan_simulation_entity(cobre_output_dir, "costs")
    if lf is None:
        return {}

    cost_cols = [
        "thermal_cost",
        "deficit_cost",
        "excess_cost",
        "storage_violation_cost",
        "filling_target_cost",
        "hydro_violation_cost",
        "outflow_violation_below_cost",
        "outflow_violation_above_cost",
        "turbined_violation_cost",
        "generation_violation_cost",
        "evaporation_violation_cost",
        "withdrawal_violation_cost",
        "inflow_penalty_cost",
        "generic_violation_cost",
        "spillage_cost",
        "fpha_turbined_cost",
        "curtailment_cost",
        "exchange_cost",
        "pumping_cost",
    ]

    available = set(lf.collect_schema().names())
    cols = [c for c in cost_cols if c in available]
    if not cols:
        return {}

    has_discount = "discount_factor" in available

    try:
        # Discount costs to present value, then sum per scenario.
        if has_discount:
            disc_exprs = [
                (pl.col(c) * pl.col("discount_factor")).sum().alias(c) for c in cols
            ]
        else:
            disc_exprs = [pl.col(c).sum() for c in cols]

        per_sc = lf.group_by("scenario_id").agg(disc_exprs)
        means = per_sc.select([pl.col(c).mean() for c in cols]).collect()
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read Cobre cost breakdown")
        return {}

    result: dict[str, float] = {}
    for c in cols:
        v = float(means[c][0] or 0)
        if abs(v) > 0.01:
            result[c] = v

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

    # Map columns to standard names.  Prefer exact matches first to avoid
    # collisions (e.g. "upper_bound_std" overwriting "upper_bound_mean").
    col_map: dict[str, str] = {}
    cols_lower = {col: col.lower() for col in df.columns}

    # Exact matches (highest priority).
    for col, lower in cols_lower.items():
        if lower == "iteration":
            col_map[col] = "iteration"
        elif lower == "lower_bound":
            col_map[col] = "lower_bound"
        elif lower == "upper_bound_mean":
            col_map[col] = "upper_bound_mean"

    # Fuzzy fallback for columns not yet mapped.
    for col, lower in cols_lower.items():
        if col in col_map:
            continue
        if "iteration" not in col_map.values() and (
            "iter" in lower
        ):
            col_map[col] = "iteration"
        elif "lower_bound" not in col_map.values() and (
            "lower" in lower or "zinf" in lower
        ):
            col_map[col] = "lower_bound"
        elif "upper_bound_mean" not in col_map.values() and (
            "upper_mean" in lower or "zsup" in lower
        ):
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

        # Reservoir min/max storage for offset correction.
        reservoir = hydro.get("reservoir", {})
        min_storage = reservoir.get("min_storage_hm3", 0.0) if reservoir else 0.0

        result[entity_id] = {
            "name": name,
            "productivity_mw_per_m3s": float(prod) if prod is not None else None,
            "min_storage_hm3": float(min_storage),
        }

    return result


def _find_system_json(cobre_output_dir: Path, filename: str) -> Path | None:
    """Locate a system JSON file near the Cobre output directory."""
    case_dir = cobre_output_dir.parent
    for candidate in [case_dir, cobre_output_dir, case_dir.parent]:
        p = candidate / "system" / filename
        if p.exists():
            return p
    return None


def read_cobre_thermal_metadata(cobre_output_dir: Path) -> dict[int, dict]:
    """Read thermal metadata from Cobre thermals.json.

    Returns ``{entity_id: {"name": str}}``.
    """
    path = _find_system_json(cobre_output_dir, "thermals.json")
    if path is None:
        _LOG.warning("thermals.json not found near %s", cobre_output_dir)
        return {}

    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to parse thermals.json")
        return {}

    return {
        int(t["id"]): {"name": str(t.get("name", f"thermal_{t['id']}"))}
        for t in data.get("thermals", [])
    }


def read_cobre_bus_metadata(cobre_output_dir: Path) -> dict[int, dict]:
    """Read bus metadata from Cobre buses.json.

    Returns ``{entity_id: {"name": str}}``.
    """
    path = _find_system_json(cobre_output_dir, "buses.json")
    if path is None:
        _LOG.warning("buses.json not found near %s", cobre_output_dir)
        return {}

    try:
        with path.open() as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to parse buses.json")
        return {}

    return {
        int(b["id"]): {"name": str(b.get("name", f"bus_{b['id']}"))}
        for b in data.get("buses", [])
    }
