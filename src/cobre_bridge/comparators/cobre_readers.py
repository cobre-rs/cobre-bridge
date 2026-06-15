"""Cobre simulation output readers for results comparison.

Reads Cobre simulation parquets using Polars lazy scanning and streaming aggregation to
compute scenario means matching the source model MEDIAS aggregation level.  Also reads
convergence data and hydro metadata.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import polars as pl

from cobre_bridge.cobre_io import (
    case_dir_for,
    resolve_hydro_productivities,
)
from cobre_bridge.cost_categories import COBRE_COST_COMPONENT_COLUMNS

_LOG = logging.getLogger(__name__)


class CobreReadError(RuntimeError):
    """A Cobre output file/dir existed but could not be read or parsed.

    Raised when a reader fails while reading an *already-confirmed-present*
    Cobre output (parquet/dir/JSON) that feeds the bounds/results comparison.
    A genuinely **absent** optional output must still yield an empty frame
    (never this error) — only a real read/schema/dtype failure on existing
    data raises, so the comparison engine never silently treats unreadable
    output as "no divergence".
    """


def _load_block_hours(cobre_output_dir: Path) -> pl.DataFrame | None:
    """Load block hours from stages.json as a Polars DataFrame.

    Returns DataFrame with columns ``stage_id``, ``block_id``, ``hours``
    or None if stages.json cannot be found or parsed.
    """
    case_dir = case_dir_for(cobre_output_dir)
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
        # ``scan_parquet`` is lazy and will not surface a malformed/corrupt
        # file until the schema or data is actually touched.  Probe the
        # schema eagerly so a present-but-unreadable parquet raises here —
        # the single choke point every simulation reader shares — instead of
        # leaking a raw polars error past each reader's narrower try/except.
        lf.collect_schema()
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(f"Failed to scan parquets in {sim_dir}") from exc

    return lf


def read_cobre_hydro_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre hydro simulation means per (entity_id, stage_id).

    Scans ``output/simulation/hydros/`` with Polars streaming and computes
    block-hours-weighted scenario means for flow variables (generation,
    turbined, spillage, evaporation, outflow) and plain scenario means
    for stage-level variables (storage, inflow, water value, energy).

    Returns DataFrame with columns: ``entity_id``, ``stage_id``,
    ``storage_final_hm3``, ``generation_mw``, ``turbined_m3s``,
    ``spillage_m3s``, ``inflow_m3s``, ``water_value_per_hm3``,
    ``stored_energy_initial_mwh``, ``stored_energy_final_mwh``,
    ``incremental_inflow_energy_mw``, ``evaporation_m3s``,
    ``outflow_m3s``, ``incremental_inflow_m3s``.
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
            "stored_energy_initial_mwh": pl.Float64,
            "stored_energy_final_mwh": pl.Float64,
            "incremental_inflow_energy_mw": pl.Float64,
            "evaporation_m3s": pl.Float64,
            "outflow_m3s": pl.Float64,
            "incremental_inflow_m3s": pl.Float64,
            "evaporation_violation_pos_m3s": pl.Float64,
            "evaporation_violation_neg_m3s": pl.Float64,
            "water_withdrawal_violation_pos_m3s": pl.Float64,
            "water_withdrawal_violation_neg_m3s": pl.Float64,
            "inflow_nonnegativity_slack_m3s": pl.Float64,
        }
    )

    lf = _scan_simulation_entity(cobre_output_dir, "hydros")
    if lf is None:
        return empty

    flow_cols = [
        "generation_mw",
        "turbined_m3s",
        "spillage_m3s",
        "evaporation_m3s",
        "outflow_m3s",
        "evaporation_violation_pos_m3s",
        "evaporation_violation_neg_m3s",
        "water_withdrawal_violation_pos_m3s",
        "water_withdrawal_violation_neg_m3s",
        "inflow_nonnegativity_slack_m3s",
    ]
    stage_cols = [
        "storage_final_hm3",
        "inflow_m3s",
        "incremental_inflow_m3s",
        "water_value_per_hm3",
        "stored_energy_initial_mwh",
        "stored_energy_final_mwh",
        "incremental_inflow_energy_mw",
    ]

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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to aggregate hydro simulation data: "
            f"{cobre_output_dir / 'simulation' / 'hydros'}"
        ) from exc

    for col in flow_cols + stage_cols:
        if col not in result.columns:
            result = result.with_columns(pl.lit(None).cast(pl.Float64).alias(col))

    return result


def read_cobre_hydro_total_flows(
    cobre_output_dir: Path,
    cobre_hydro_means: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-(entity_id, stage_id) the source-model-equivalent total inflow.

    ``QAFLUH`` in the source model is the *total* inflow arriving at a hydro
    plant: the local incremental inflow plus the outflow (turbined +
    spilled) of every immediate upstream plant. Cobre emits the
    components separately; this helper sums them via the
    ``downstream_id`` topology from ``system/hydros.json``.

    The total *outflow* is already in the Cobre simulation parquet
    (``outflow_m3s``) and is exposed by :func:`read_cobre_hydro_means`,
    so this function only computes the inflow side.

    Parameters
    ----------
    cobre_output_dir:
        Path to ``<case_dir>/output``.
    cobre_hydro_means:
        Output of :func:`read_cobre_hydro_means` containing
        ``entity_id``, ``stage_id``, ``incremental_inflow_m3s`` and
        ``outflow_m3s`` columns.

    Returns
    -------
    pl.DataFrame
        Columns ``entity_id``, ``stage_id``, ``total_inflow_m3s``.
        Empty when ``hydros.json`` is missing or the source frame lacks
        either component column.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "total_inflow_m3s": pl.Float64,
        }
    )
    if cobre_hydro_means.is_empty():
        return empty
    required = {"entity_id", "stage_id", "incremental_inflow_m3s", "outflow_m3s"}
    if not required.issubset(cobre_hydro_means.columns):
        return empty

    case_dir = case_dir_for(cobre_output_dir)
    hydros_path = _find_system_json(cobre_output_dir, "hydros.json")
    if hydros_path is None:
        hydros_path = case_dir / "system" / "hydros.json"
    if not hydros_path.exists():
        return empty
    try:
        with hydros_path.open() as f:
            hydros_data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            f"Failed to parse hydros.json for total-inflow topology: {hydros_path}"
        ) from exc

    # parents[child_id] = list of hydro_ids whose downstream_id == child_id.
    parents: dict[int, list[int]] = {}
    for h in hydros_data.get("hydros", []):
        ds = h.get("downstream_id")
        if ds is None:
            continue
        parents.setdefault(int(ds), []).append(int(h["id"]))

    # Per-(entity_id, stage_id) outflow lookup for the upstream sum.
    means = cobre_hydro_means.select(
        pl.col("entity_id").cast(pl.Int64),
        pl.col("stage_id").cast(pl.Int64),
        pl.col("incremental_inflow_m3s").cast(pl.Float64),
        pl.col("outflow_m3s").cast(pl.Float64),
    )
    outflow_lookup: dict[tuple[int, int], float] = {}
    for row in means.iter_rows(named=True):
        if row["outflow_m3s"] is None:
            continue
        outflow_lookup[(int(row["entity_id"]), int(row["stage_id"]))] = float(
            row["outflow_m3s"]
        )

    rows: list[dict[str, float | int]] = []
    for row in means.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        incr = row["incremental_inflow_m3s"]
        if incr is None:
            continue
        upstream_sum = 0.0
        for pid in parents.get(eid, []):
            upstream_sum += outflow_lookup.get((pid, sid), 0.0)
        rows.append(
            {
                "entity_id": eid,
                "stage_id": sid,
                "total_inflow_m3s": float(incr) + upstream_sum,
            }
        )
    if not rows:
        return empty
    return pl.DataFrame(rows).cast(
        {"entity_id": pl.Int64, "stage_id": pl.Int64, "total_inflow_m3s": pl.Float64}
    )


def _load_hydro_reservoir_flag(cobre_output_dir: Path) -> dict[int, bool]:
    """Return ``{hydro_id: max_storage_hm3 > 0}`` from system/hydros.json.

    The discriminator mirrors the dashboard convention (reservoir = any
    plant with positive max storage; run-of-river otherwise).
    """
    case_dir = case_dir_for(cobre_output_dir)
    hydros_path = _find_system_json(cobre_output_dir, "hydros.json")
    if hydros_path is None:
        hydros_path = case_dir / "system" / "hydros.json"
    if not hydros_path.exists():
        return {}
    try:
        with hydros_path.open() as f:
            data = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    out: dict[int, bool] = {}
    for h in data.get("hydros", []):
        max_stor = h.get("reservoir", {}).get("max_storage_hm3", 0) or 0
        out[int(h["id"])] = max_stor > 0
    return out


def read_cobre_spillage_energy(cobre_output_dir: Path) -> pl.DataFrame:
    """Compute system spillage in MWmes (stage-average MW).

    For each (scenario, stage, block) and each hydro the per-row energy
    contribution is ``spillage_m3s × equivalent_productivity_mw_per_m3s``
    (MW). The system aggregation is:

    1. Sum across hydros within each (scenario, stage, block, category)
       to get the system MW per block per category.
    2. Block-hours-weighted average over blocks → stage-mean MW.
    3. Mean across scenarios.

    The reservoir / run-of-river split uses ``max_storage_hm3 > 0`` from
    ``system/hydros.json`` (matches the dashboard rule).

    Returns columns ``stage_id``, ``total_mw``, ``reservoir_mw``,
    ``rorov_mw``. Empty when required columns or system data are
    missing.
    """
    empty = pl.DataFrame(
        schema={
            "stage_id": pl.Int64,
            "total_mw": pl.Float64,
            "reservoir_mw": pl.Float64,
            "rorov_mw": pl.Float64,
        }
    )
    lf = _scan_simulation_entity(cobre_output_dir, "hydros")
    if lf is None:
        return empty

    available = set(lf.collect_schema().names())
    required = {
        "spillage_m3s",
        "equivalent_productivity_mw_per_m3s",
        "stage_id",
        "block_id",
        "scenario_id",
    }
    if not required.issubset(available):
        return empty

    id_col = "hydro_id" if "hydro_id" in available else "entity_id"
    if id_col not in available:
        return empty

    is_reservoir = _load_hydro_reservoir_flag(cobre_output_dir)
    if not is_reservoir:
        return empty

    block_hours = _load_block_hours(cobre_output_dir)
    if block_hours is None:
        return empty

    flag_df = pl.DataFrame(
        {
            id_col: list(is_reservoir.keys()),
            "category": ["reservoir" if v else "rorov" for v in is_reservoir.values()],
        }
    )

    try:
        per_stage = (
            lf.with_columns(
                (
                    pl.col("spillage_m3s")
                    * pl.col("equivalent_productivity_mw_per_m3s")
                ).alias("spill_mw")
            )
            .join(flag_df.lazy(), on=id_col)
            .group_by(["scenario_id", "stage_id", "block_id", "category"])
            .agg(pl.col("spill_mw").sum())
            .join(block_hours.lazy(), on=["stage_id", "block_id"])
            .group_by(["scenario_id", "stage_id", "category"])
            .agg(
                (
                    (pl.col("spill_mw") * pl.col("hours")).sum() / pl.col("hours").sum()
                ).alias("mw_mean")
            )
            .group_by(["stage_id", "category"])
            .agg(pl.col("mw_mean").mean())
            .collect(engine="streaming")
        )
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to aggregate Cobre spillage-energy: "
            f"{cobre_output_dir / 'simulation' / 'hydros'}"
        ) from exc

    if per_stage.is_empty():
        return empty

    df = per_stage.pivot(index="stage_id", on="category", values="mw_mean")
    if "reservoir" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("reservoir"))
    if "rorov" not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias("rorov"))
    df = df.with_columns(
        pl.col("reservoir").fill_null(0.0).alias("reservoir"),
        pl.col("rorov").fill_null(0.0).alias("rorov"),
    )
    df = df.with_columns(
        (pl.col("reservoir") + pl.col("rorov")).alias("total_mw"),
    ).rename({"reservoir": "reservoir_mw", "rorov": "rorov_mw"})
    return df.select(
        pl.col("stage_id").cast(pl.Int64),
        pl.col("total_mw").cast(pl.Float64),
        pl.col("reservoir_mw").cast(pl.Float64),
        pl.col("rorov_mw").cast(pl.Float64),
    ).sort("stage_id")


def read_cobre_iteration_timing(cobre_output_dir: Path) -> pl.DataFrame:
    """Return per-iteration wall-clock from ``training/convergence.parquet``.

    Columns: ``iteration`` (Int64), ``time_forward_ms``,
    ``time_backward_ms``, ``time_total_ms`` (Float64). Missing timing columns are filled
    with nulls; empty DataFrame when the parquet is absent. Separate from
    :func:`read_cobre_convergence` which only surfaces the bound columns the source
    model pmo.dat can be compared against.
    """
    empty = pl.DataFrame(
        schema={
            "iteration": pl.Int64,
            "time_forward_ms": pl.Float64,
            "time_backward_ms": pl.Float64,
            "time_total_ms": pl.Float64,
        }
    )
    conv_path = cobre_output_dir / "training" / "convergence.parquet"
    if not conv_path.exists():
        return empty
    try:
        df = pl.read_parquet(conv_path)
    except Exception:  # noqa: BLE001
        _LOG.warning("Failed to read convergence.parquet for timings")
        return empty
    keep = ["iteration", "time_forward_ms", "time_backward_ms", "time_total_ms"]
    avail = [c for c in keep if c in df.columns]
    if "iteration" not in avail:
        return empty
    out = df.select(avail).sort("iteration")
    for col in ("time_forward_ms", "time_backward_ms", "time_total_ms"):
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).cast(pl.Float64).alias(col))
    return out.cast(
        {
            "iteration": pl.Int64,
            "time_forward_ms": pl.Float64,
            "time_backward_ms": pl.Float64,
            "time_total_ms": pl.Float64,
        }
    )


def read_cobre_training_duration(cobre_output_dir: Path) -> float:
    """Return total training duration in seconds.

    Sourced from ``training/metadata.json:duration_seconds``. Returns
    ``0.0`` when the metadata file is missing or malformed.
    """
    meta_path = cobre_output_dir / "training" / "metadata.json"
    if not meta_path.exists():
        return 0.0
    try:
        with meta_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        _LOG.warning("Failed to parse training/metadata.json")
        return 0.0
    return float(data.get("duration_seconds", 0.0) or 0.0)


def read_cobre_line_means(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre line simulation means per (line_id, stage_id).

    Computes block-hours-weighted scenario means for ``net_flow_mw``.
    Returns columns ``entity_id``, ``stage_id``, ``net_flow_mw``.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "net_flow_mw": pl.Float64,
        }
    )
    lf = _scan_simulation_entity(cobre_output_dir, "exchanges")
    if lf is None:
        return empty
    available = set(lf.collect_schema().names())
    if "net_flow_mw" not in available:
        return empty
    id_col = "line_id" if "line_id" in available else "entity_id"
    block_hours = _load_block_hours(cobre_output_dir)
    try:
        if block_hours is not None:
            result = (
                _weighted_stage_mean(lf, id_col, ["net_flow_mw"], block_hours)
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
        else:
            result = (
                lf.filter(pl.col("block_id") == 0)
                .group_by(id_col, "stage_id")
                .agg(pl.col("net_flow_mw").mean())
                .rename({id_col: "entity_id"})
                .sort("entity_id", "stage_id")
                .collect(engine="streaming")
            )
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to aggregate exchange simulation data: "
            f"{cobre_output_dir / 'simulation' / 'exchanges'}"
        ) from exc
    return result


def read_cobre_line_percentiles(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre line p10/p50/p90 per (entity_id, stage_id).

    Returns columns ``entity_id``, ``stage_id``, ``net_flow_mw_p10``,
    ``net_flow_mw_p50``, ``net_flow_mw_p90``.
    """
    lf = _scan_simulation_entity(cobre_output_dir, "exchanges")
    if lf is None:
        return pl.DataFrame()
    available = set(lf.collect_schema().names())
    if "net_flow_mw" not in available:
        return pl.DataFrame()
    id_col = "line_id" if "line_id" in available else "entity_id"
    block_hours = _load_block_hours(cobre_output_dir)
    try:
        per_sc = _weighted_scenario_values(lf, id_col, ["net_flow_mw"], [], block_hours)
        return _compute_percentiles(per_sc, ["net_flow_mw"])
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to compute line percentiles: "
            f"{cobre_output_dir / 'simulation' / 'exchanges'}"
        ) from exc


def read_cobre_lp_max_generation(cobre_output_dir: Path) -> pl.DataFrame:
    """Return per-(entity_id, stage_id) hydro LP max-generation bound.

    Reads ``output/training/dictionaries/bounds.parquet`` and filters to
    hydro generation upper bounds (``entity_type_code == 0``,
    ``bound_type_code == 7``). Takes block 0 as the stage representative
    — Cobre currently emits a stage-constant max in practice, so this
    matches the dashboard's plant-explorer behaviour.

    Returns columns: ``entity_id``, ``stage_id``,
    ``cobre_lp_gen_max_mw``. Empty when the parquet is missing.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "cobre_lp_gen_max_mw": pl.Float64,
        }
    )
    bounds_path = cobre_output_dir / "training" / "dictionaries" / "bounds.parquet"
    if not bounds_path.exists():
        return empty
    try:
        df = pl.read_parquet(bounds_path)
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(f"Failed to read bounds.parquet: {bounds_path}") from exc
    required = {
        "entity_type_code",
        "entity_id",
        "stage_id",
        "bound_type_code",
        "bound_value",
    }
    if not required.issubset(df.columns):
        return empty
    filt = df.filter(
        (pl.col("entity_type_code") == 0) & (pl.col("bound_type_code") == 7)
    )
    if "block_id" in filt.columns:
        filt = filt.filter(pl.col("block_id") == 0)
    return filt.select(
        pl.col("entity_id").cast(pl.Int64),
        pl.col("stage_id").cast(pl.Int64),
        pl.col("bound_value").cast(pl.Float64).alias("cobre_lp_gen_max_mw"),
    ).sort("entity_id", "stage_id")


def read_cobre_hydro_withdrawal(cobre_output_dir: Path) -> pl.DataFrame:
    """Return per-(hydro_id, stage_id) input water-withdrawal target.

    Cobre does not emit realized water withdrawal as a per-stage simulation result;
    instead the target lives in ``constraints/hydro_bounds.parquet`` as
    ``water_withdrawal_m3s`` (one value per hydro-stage). Comparison against the source
    model ``VRETIRUH`` therefore matches the *input* target — discrepancies beyond the
    post-study horizon are expected (see ``hydro.py:1083`` converter note).

    Returns columns: ``entity_id``, ``stage_id``, ``withdrawal_m3s``.
    Empty frame if the parquet is missing or lacks the column.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "withdrawal_m3s": pl.Float64,
        }
    )
    case_dir = case_dir_for(cobre_output_dir)
    bounds_path = case_dir / "constraints" / "hydro_bounds.parquet"
    if not bounds_path.exists():
        return empty
    try:
        df = pl.read_parquet(bounds_path)
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            f"Failed to read hydro_bounds.parquet: {bounds_path}"
        ) from exc
    if "water_withdrawal_m3s" not in df.columns:
        return empty
    return df.select(
        pl.col("hydro_id").cast(pl.Int64).alias("entity_id"),
        pl.col("stage_id").cast(pl.Int64),
        pl.col("water_withdrawal_m3s").cast(pl.Float64).alias("withdrawal_m3s"),
    ).sort("entity_id", "stage_id")


def read_cobre_hydro_per_stage_bounds(cobre_output_dir: Path) -> pl.DataFrame:
    """Per-(hydro_id, stage_id) operational bounds from
    ``constraints/hydro_bounds.parquet``.

    Surfaces every bound column the dashboard's plant-detail panel
    overlays as dashed lines: ``min_storage_hm3``, ``max_storage_hm3``,
    ``min_turbined_m3s``, ``max_turbined_m3s``, ``min_outflow_m3s``,
    ``min_generation_mw``.  Columns absent from the parquet are
    silently dropped from the result; callers should treat missing
    columns as "no per-stage override — use the static value from
    ``hydros.json``".

    Output columns: ``entity_id`` (Int64), ``stage_id`` (Int64),
    plus any of the bound columns that exist in the parquet, all
    cast to ``Float64``.  Returns an empty frame when the parquet
    is missing.
    """
    case_dir = case_dir_for(cobre_output_dir)
    bounds_path = case_dir / "constraints" / "hydro_bounds.parquet"
    bound_cols = [
        "min_storage_hm3",
        "max_storage_hm3",
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_outflow_m3s",
        "min_generation_mw",
    ]
    empty_schema: dict[str, pl.DataType] = {
        "entity_id": pl.Int64,
        "stage_id": pl.Int64,
    }
    for c in bound_cols:
        empty_schema[c] = pl.Float64
    if not bounds_path.exists():
        return pl.DataFrame(schema=empty_schema)
    try:
        df = pl.read_parquet(bounds_path)
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            f"Failed to read hydro_bounds.parquet for dashboard bounds: {bounds_path}"
        ) from exc
    available = [c for c in bound_cols if c in df.columns]
    if not available:
        return pl.DataFrame(schema=empty_schema)
    select_exprs: list[pl.Expr] = [
        pl.col("hydro_id").cast(pl.Int64).alias("entity_id"),
        pl.col("stage_id").cast(pl.Int64),
    ]
    select_exprs.extend(pl.col(c).cast(pl.Float64) for c in available)
    return df.select(select_exprs).sort("entity_id", "stage_id")


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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to aggregate thermal simulation data: "
            f"{cobre_output_dir / 'simulation' / 'thermals'}"
        ) from exc

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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to aggregate bus simulation data: "
            f"{cobre_output_dir / 'simulation' / 'buses'}"
        ) from exc

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

    flow_cols = [
        "generation_mw",
        "turbined_m3s",
        "spillage_m3s",
        "evaporation_m3s",
        "outflow_m3s",
        # Operational slacks — surfaced as cobre-only series on the
        # plant-detail tab.  Percentiles are needed so the band+P10/P90
        # tooltip works the same as for the rest of the flow variables;
        # without them only the Cobre Mean line is plotted and the
        # unified-x hover has nothing to lock onto.
        "water_withdrawal_violation_pos_m3s",
        "water_withdrawal_violation_neg_m3s",
        "inflow_nonnegativity_slack_m3s",
    ]
    stage_cols = [
        "storage_final_hm3",
        "inflow_m3s",
        "incremental_inflow_m3s",
        "water_value_per_hm3",
        "stored_energy_initial_mwh",
        "stored_energy_final_mwh",
        "incremental_inflow_energy_mw",
    ]
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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to compute hydro percentiles: "
            f"{cobre_output_dir / 'simulation' / 'hydros'}"
        ) from exc


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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to compute thermal percentiles: "
            f"{cobre_output_dir / 'simulation' / 'thermals'}"
        ) from exc


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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to compute bus percentiles: "
            f"{cobre_output_dir / 'simulation' / 'buses'}"
        ) from exc


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
                pl.col("load_mw").fill_null(0.0) - pl.col("ncs_gen_mw").fill_null(0.0)
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


def read_cobre_cost_breakdown(
    cobre_output_dir: Path,
    max_stage_id: int | None = None,
) -> dict[str, float]:
    """Read cost breakdown from Cobre simulation costs entity.

    Returns ``{category: mean_total_R$}`` averaged across scenarios,
    summed across all stages and blocks.  Zero-cost categories are excluded.

    Parameters
    ----------
    cobre_output_dir:
        Path to the Cobre ``output/`` directory.
    max_stage_id:
        If provided, only include stages with ``stage_id <= max_stage_id``. Used to make
        the cost breakdown comparable to the source model, which usually reports a
        shorter horizon than Cobre.
    """
    lf = _scan_simulation_entity(cobre_output_dir, "costs")
    if lf is None:
        return {}

    # Sum every individual cost component (the canonical set, so no column is
    # silently dropped — contract_cost used to be missing here). The aggregate
    # roll-ups (hydro_violation_cost, total/immediate/future) are excluded.
    cost_cols = list(COBRE_COST_COMPONENT_COLUMNS)

    available = set(lf.collect_schema().names())
    cols = [c for c in cost_cols if c in available]
    if not cols:
        return {}

    has_discount = "discount_factor" in available

    if max_stage_id is not None and "stage_id" in available:
        lf = lf.filter(pl.col("stage_id") <= max_stage_id)

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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to read Cobre cost breakdown: "
            f"{cobre_output_dir / 'simulation' / 'costs'}"
        ) from exc

    result: dict[str, float] = {}
    for c in cols:
        v = float(means[c][0] or 0)
        if abs(v) > 0.01:
            result[c] = v

    return result


def read_cobre_stage_costs(cobre_output_dir: Path) -> pl.DataFrame:
    """Read Cobre per-stage immediate/future/thermal cost (mean across scenarios).

    Returns a DataFrame with columns ``stage_id`` (Int64), ``immediate_cost`` (Float64,
    R$), ``future_cost`` (Float64, R$), ``thermal_cost`` (Float64, R$),
    ``anticipated_thermal_cost`` (Float64, R$) and the derived ``thermal_cost_total`` (=
    ``thermal_cost`` + ``anticipated_thermal_cost``). All are raw, *undiscounted* stage
    values — the counterparts of the source model's MEDIAS-SIN ``COPER`` /
    ``CUSTO_FUTURO`` / ``CTERM`` (after the 10⁶ R$ unit conversion on the source model
    side).

    ``anticipated_thermal_cost`` is the GNL forward-committed thermal fuel that Cobre
    books on the decision-stage commitment column (part of ``immediate_cost`` but
    excluded from ``thermal_cost``); it was added to Cobre's costs schema after 0.8.0.
    ``thermal_cost_total`` is the the source-model-comparable thermal generation cost
    (CTERM books GNL at delivery). Pre-anticipation runs lack the column → it reads as 0
    and ``thermal_cost_total == thermal_cost``.

    Cobre's costs table is one row per ``(scenario_id, stage_id, block_id)``;
    we sum block-level immediate_cost / thermal_cost / anticipated_thermal_cost
    within each (scenario, stage) and keep the (scenario, stage) value of
    future_cost, then average across scenarios.  ``future_cost`` is identical
    across blocks of the same stage so a ``max`` (= any) collapse is safe.
    """
    _SUM_COLS = ("immediate_cost", "thermal_cost", "anticipated_thermal_cost")
    _ALL_COLS = (
        "immediate_cost",
        "future_cost",
        "thermal_cost",
        "anticipated_thermal_cost",
    )
    # Output adds the derived the source-model-comparable thermal total.
    _OUT_COLS = (*_ALL_COLS, "thermal_cost_total")
    empty = pl.DataFrame(
        schema={"stage_id": pl.Int64, **{c: pl.Float64 for c in _OUT_COLS}}
    )

    lf = _scan_simulation_entity(cobre_output_dir, "costs")
    if lf is None:
        return empty

    available = set(lf.collect_schema().names())
    if "stage_id" not in available:
        return empty
    if not any(c in available for c in _ALL_COLS):
        return empty

    agg_exprs: list[pl.Expr] = [
        pl.col(c).sum().alias(c) for c in _SUM_COLS if c in available
    ]
    if "future_cost" in available:
        # future_cost is a per-stage quantity replicated across blocks;
        # ``max`` collapses without double-counting.
        agg_exprs.append(pl.col("future_cost").max().alias("future_cost"))

    try:
        per_sc = lf.group_by(["scenario_id", "stage_id"]).agg(agg_exprs)
        mean_cols = [pl.col(c).mean() for c in _ALL_COLS if c in available]
        df = per_sc.group_by("stage_id").agg(mean_cols).sort("stage_id").collect()
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            "Failed to read Cobre per-stage costs: "
            f"{cobre_output_dir / 'simulation' / 'costs'}"
        ) from exc

    # Ensure all columns are present even if one was missing in the schema
    # (e.g. anticipated_thermal_cost on pre-anticipation Cobre runs).
    for c in _ALL_COLS:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias(c))
    # Derived the source-model-comparable thermal total: live generation + anticipated
    # (GNL forward-committed) fuel. Null anticipated (old runs) counts as 0.
    df = df.with_columns(
        (
            pl.col("thermal_cost").fill_null(0.0)
            + pl.col("anticipated_thermal_cost").fill_null(0.0)
        ).alias("thermal_cost_total")
    )
    return df.select(["stage_id", *_OUT_COLS]).cast(
        {"stage_id": pl.Int64, **{c: pl.Float64 for c in _OUT_COLS}}
    )


def read_converted_penalties(cobre_output_dir: Path) -> dict:
    """Load the converted ``penalties.json`` for the Cobre case.

    ``penalties.json`` lives at the Cobre *case* root (the parent of the
    ``output`` directory the comparator is pointed at), so we look there first,
    then in the output dir itself. Returns the parsed dict, or ``{}`` when not
    found — callers should treat an empty dict as "penalties unavailable".
    """
    for candidate in (
        case_dir_for(cobre_output_dir) / "penalties.json",
        cobre_output_dir / "penalties.json",
    ):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                _LOG.warning("Failed to parse %s", candidate)
                return {}
    return {}


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
    except Exception as exc:  # noqa: BLE001
        raise CobreReadError(
            f"Failed to read convergence.parquet: {conv_path}"
        ) from exc

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
        if "iteration" not in col_map.values() and ("iter" in lower):
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
    """Read hydro metadata from Cobre system JSON files.

    Reads ``hydros.json`` and ``hydro_production_models.json``.
    Productivity now lives per-(hydro, stage) in
    ``hydro_production_models.json`` on cobre HEAD. We surface the
    first ``stage_ranges`` entry's value as ``productivity_mw_per_m3s`` for
    backward compatibility with comparator/dashboard callers.

    Returns ``{entity_id: {"name": str, "productivity_mw_per_m3s": float | None,
    "min_storage_hm3": float, "bus_id": int | None}}``.
    """
    hydros_path = _find_system_json(cobre_output_dir, "hydros.json")
    if hydros_path is None:
        _LOG.warning("hydros.json not found near %s", cobre_output_dir)
        return {}

    try:
        with hydros_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        _LOG.warning("Failed to parse hydros.json")
        return {}

    case_dir = hydros_path.parent.parent
    # Productivity resolution is shared with the dashboard via the single
    # canonical cascade in cobre_io so the two products never report a
    # different ρ for the same plant (see resolve_hydro_productivities).
    productivities = resolve_hydro_productivities(case_dir, data.get("hydros", []))

    result: dict[int, dict] = {}
    for hydro in data.get("hydros", []):
        entity_id = int(hydro["id"])
        name = str(hydro.get("name", f"hydro_{entity_id}"))

        prod = productivities.get(entity_id)

        reservoir = hydro.get("reservoir", {}) or {}
        outflow = hydro.get("outflow", {}) or {}
        generation = hydro.get("generation", {}) or {}

        bus_id = hydro.get("bus_id")
        result[entity_id] = {
            "name": name,
            "productivity_mw_per_m3s": float(prod) if prod is not None else None,
            "min_storage_hm3": float(reservoir.get("min_storage_hm3", 0.0) or 0.0),
            "max_storage_hm3": float(reservoir.get("max_storage_hm3", 0.0) or 0.0),
            "min_outflow_m3s": float(outflow.get("min_outflow_m3s", 0.0) or 0.0),
            # ``max_outflow_m3s`` is typically null in cobre cases — leave as
            # None so the dashboard can skip the dashed line.
            "max_outflow_m3s": (
                float(outflow["max_outflow_m3s"])
                if outflow.get("max_outflow_m3s") is not None
                else None
            ),
            "min_turbined_m3s": float(generation.get("min_turbined_m3s", 0.0) or 0.0),
            "max_turbined_m3s": float(generation.get("max_turbined_m3s", 0.0) or 0.0),
            "min_generation_mw": float(generation.get("min_generation_mw", 0.0) or 0.0),
            "max_generation_mw": float(generation.get("max_generation_mw", 0.0) or 0.0),
            "bus_id": int(bus_id) if bus_id is not None else None,
        }

    return result


def read_cobre_productivity_detail(cobre_output_dir: Path) -> dict[int, dict]:
    """Read the per-hydro converted building blocks for the Productivity tab.

    Surfaces the productivity building blocks cobre-bridge wrote into
    ``system/hydros.json`` so the Building-Blocks table can show them next to
    the source model HIDR cadastro values: ``specific_productivity``
    (``specific_productivity_mw_per_m3s_per_m``), ``tailwater_m`` (constant
    ``tailrace.coefficients[0]``), ``losses_m`` (constant
    ``hydraulic_losses.value_m``), ``vmin_hm3`` / ``vmax_hm3``.

    Returns ``{hydro_id: {"name", "specific_productivity", "tailwater_m", "losses_m",
    "vmin_hm3", "vmax_hm3"}}``; per-field values are ``None`` when absent.  Returns an
    empty dict when ``hydros.json`` cannot be located.  (The
    point/equivalent/accumulated productivities are *not* read from Cobre here — the
    Productivity-tab scatters validate the conversion against what cobre-bridge computes
    from the source model inputs, and the realized per-stage productivity comes from the
    simulation generation/turbined comparison rows.)
    """
    hydros_path = _find_system_json(cobre_output_dir, "hydros.json")
    if hydros_path is None:
        _LOG.warning("hydros.json not found near %s", cobre_output_dir)
        return {}
    try:
        with hydros_path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        _LOG.warning("Failed to parse hydros.json for productivity detail")
        return {}

    result: dict[int, dict] = {}
    for hydro in data.get("hydros", []):
        hid = int(hydro["id"])

        tailrace = hydro.get("tailrace") or {}
        coeffs = tailrace.get("coefficients") or []
        tailwater_m = float(coeffs[0]) if coeffs else None

        losses = hydro.get("hydraulic_losses") or {}
        losses_m = (
            float(losses["value_m"])
            if losses.get("type") == "constant" and losses.get("value_m") is not None
            else None
        )

        reservoir = hydro.get("reservoir") or {}
        spec = hydro.get("specific_productivity_mw_per_m3s_per_m")

        result[hid] = {
            "name": str(hydro.get("name", f"hydro_{hid}")),
            "specific_productivity": float(spec) if spec is not None else None,
            "tailwater_m": tailwater_m,
            "losses_m": losses_m,
            "vmin_hm3": float(reservoir.get("min_storage_hm3", 0.0) or 0.0),
            "vmax_hm3": float(reservoir.get("max_storage_hm3", 0.0) or 0.0),
        }

    return result


def _find_system_json(cobre_output_dir: Path, filename: str) -> Path | None:
    """Locate a system JSON file near the Cobre output directory."""
    case_dir = case_dir_for(cobre_output_dir)
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
