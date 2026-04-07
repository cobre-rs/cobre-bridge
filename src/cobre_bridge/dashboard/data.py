"""Dashboard data layer — loading helpers and DashboardData dataclass.

All data loading previously performed inside ``build_dashboard()`` is
consolidated here into ``DashboardData.load(case_dir)``.  Chart functions
receive a single ``DashboardData`` instance instead of a long list of
individual arguments.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pandas as pd
import polars as pl
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def scan_entity(case_dir: Path, entity: str) -> pl.LazyFrame:
    """Return a LazyFrame scanning all hive-partitioned parquet files for entity."""
    sim_dir = case_dir / "output" / "simulation" / entity
    return pl.scan_parquet(str(sim_dir / "**" / "*.parquet"), hive_partitioning=True)


def load_names(case_dir: Path) -> dict[tuple[str, int], str]:
    """Load entity name mappings from system JSON files."""
    names: dict[tuple[str, int], str] = {}
    for entity, key in [
        ("hydros", "hydros"),
        ("buses", "buses"),
        ("thermals", "thermals"),
        ("lines", "lines"),
        ("non_controllable_sources", "non_controllable_sources"),
    ]:
        path = case_dir / "system" / f"{entity}.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            for item in data.get(key, []):
                names[(entity, item["id"])] = item.get("name", str(item["id"]))
    return names


def entity_name(names: dict[tuple[str, int], str], entity: str, eid: int) -> str:
    return names.get((entity, eid), str(eid))


def load_stage_labels(case_dir: Path) -> dict[int, str]:
    """Return stage_id -> "Mon YYYY" label from stages.json."""
    path = case_dir / "stages.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    labels: dict[int, str] = {}
    for stage in data.get("stages", []):
        sid = stage["id"]
        start = stage.get("start_date", "")
        if start:
            try:
                labels[sid] = pd.to_datetime(start).strftime("%b %Y")
            except Exception:
                labels[sid] = str(sid)
        else:
            labels[sid] = str(sid)
    return labels


def load_hydro_bus_map(case_dir: Path) -> dict[int, int]:
    path = case_dir / "system" / "hydros.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {h["id"]: h["bus_id"] for h in d["hydros"]}


def load_thermal_metadata(case_dir: Path) -> dict[int, dict]:
    """Return thermal_id -> {bus_id, max_mw, cost_per_mwh} metadata."""
    path = case_dir / "system" / "thermals.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    result = {}
    for t in d["thermals"]:
        segments = t.get("cost_segments", [])
        cost = segments[0]["cost_per_mwh"] if segments else 0.0
        cap = (
            sum(s["capacity_mw"] for s in segments)
            if segments
            else t.get("generation", {}).get("max_mw", 0.0)
        )
        result[t["id"]] = {
            "bus_id": t["bus_id"],
            "max_mw": cap,
            "cost_per_mwh": cost,
            "name": t.get("name", str(t["id"])),
        }
    return result


def load_ncs_bus_map(case_dir: Path) -> dict[int, int]:
    path = case_dir / "system" / "non_controllable_sources.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    return {n["id"]: n["bus_id"] for n in d["non_controllable_sources"]}


def load_hydro_metadata(case_dir: Path) -> dict[int, dict]:
    """Return hydro_id -> {bus_id, name, vol_max, vol_min, max_gen_mw, max_turbined}."""
    path = case_dir / "system" / "hydros.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    result = {}
    for h in d["hydros"]:
        gen = h.get("generation", {})
        res = h.get("reservoir", {})
        result[h["id"]] = {
            "bus_id": h["bus_id"],
            "name": h.get("name", str(h["id"])),
            "vol_max": res.get("max_storage_hm3", 0),
            "vol_min": res.get("min_storage_hm3", 0),
            "max_gen_mw": gen.get("max_generation_mw", 0),
            "max_gen_physical": gen.get("productivity_mw_per_m3s", 0)
            * gen.get("max_turbined_m3s", 0),
            "max_turbined": gen.get("max_turbined_m3s", 0),
            "productivity": gen.get("productivity_mw_per_m3s", 0),
            "downstream_id": h.get("downstream_id"),
        }
    return result


# ---------------------------------------------------------------------------
# Polars helpers
# ---------------------------------------------------------------------------


def _stage_avg_mw(
    lf: pl.LazyFrame,
    mwh_col: str,
    stage_hours: dict[int, float],
    group_cols: list[str],
) -> dict[int, float] | pl.DataFrame:
    """Compute stage-average MW from MWh summed across all blocks via LazyFrame.

    Scans all blocks (no block_id filter), sums mwh_col per
    (scenario, stage [+ group_cols]), divides by total stage hours, then
    averages across scenarios.

    Returns dict[int, float] mapping stage_id -> avg_mw when group_cols is empty,
    or a Polars DataFrame with columns [stage_id, *group_cols, _avg_mw] otherwise.
    """
    hours_df = pl.DataFrame(
        {"stage_id": list(stage_hours.keys()), "_hours": list(stage_hours.values())}
    )
    result = (
        lf.group_by(["scenario_id", "stage_id"] + group_cols)
        .agg(pl.col(mwh_col).sum())
        .join(hours_df.lazy(), on="stage_id")
        .with_columns((pl.col(mwh_col) / pl.col("_hours")).alias("_avg_mw"))
        .group_by(["stage_id"] + group_cols)
        .agg(pl.col("_avg_mw").mean())
        .sort("stage_id")
        .collect(engine="streaming")
    )
    if group_cols:
        return result
    return dict(zip(result["stage_id"].to_list(), result["_avg_mw"].to_list()))


def _compute_lp_load(
    load_stats: pd.DataFrame,
    load_factors: list[dict],
    stage_hours: dict[int, float],
    block_hours: dict[tuple[int, int], float],
    bus_filter: list[int] | None = None,
) -> dict[int, float]:
    """Compute LP load balance RHS from input files as stage-average MW."""
    factor_lk: dict[tuple[int, int, int], float] = {}
    for entry in load_factors:
        bid, sid = entry["bus_id"], entry["stage_id"]
        for bf in entry["block_factors"]:
            factor_lk[(bid, sid, bf["block_id"])] = bf["factor"]

    rows: list[dict] = []
    for _, r in load_stats.iterrows():
        bid, sid = int(r["bus_id"]), int(r["stage_id"])
        if bus_filter is not None and bid not in bus_filter:
            continue
        mean = float(r["mean_mw"])
        for blk_id in range(20):
            bh = block_hours.get((sid, blk_id))
            if bh is None:
                break
            factor = factor_lk.get((bid, sid, blk_id), 1.0)
            rows.append(
                {"scenario_id": 0, "stage_id": sid, "_lp_mwh": mean * factor * bh}
            )

    if not rows:
        return {}
    df = pd.DataFrame(rows)
    hours_map = {
        sid: stage_hours[sid] for sid in df["stage_id"].unique() if sid in stage_hours
    }
    scen_stage = df.groupby(["scenario_id", "stage_id"])["_lp_mwh"].sum().reset_index()
    scen_stage["_hours"] = scen_stage["stage_id"].map(hours_map)
    scen_stage["_avg_mw"] = scen_stage["_lp_mwh"] / scen_stage["_hours"]
    ser = scen_stage.groupby("stage_id")["_avg_mw"].mean()
    return dict(zip(ser.index.tolist(), ser.values.tolist()))


def compute_non_fictitious_bus_ids(load_stats: pd.DataFrame) -> list[int]:
    """Return sorted bus IDs that have mean_mw > 0 in at least one stage.

    A fictitious bus is defined as one whose mean load is zero across all
    stages in the input load seasonal stats.  Tabs that show per-bus facets
    should filter to this list so that accounting-only buses (e.g. NOFICT1)
    are excluded.

    Args:
        load_stats: DataFrame with columns ``bus_id``, ``stage_id``, ``mean_mw``.

    Returns:
        Sorted list of integer bus IDs with at least one stage where
        ``mean_mw > 0``.  Returns an empty list when ``load_stats`` is empty
        or the ``mean_mw`` column is absent.
    """
    if load_stats.empty or "mean_mw" not in load_stats.columns:
        return []
    positive = load_stats[load_stats["mean_mw"] > 0]
    return sorted(positive["bus_id"].unique().tolist())


# ---------------------------------------------------------------------------
# DashboardData dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class DashboardData:
    """All data sources required to render the Cobre dashboard.

    Populated via ``DashboardData.load(case_dir)``.  Fields match the local
    variables previously declared inside ``build_dashboard()``.
    """

    case_dir: Path
    case_name: str

    # Training — small single-file parquets loaded as pandas
    conv: pd.DataFrame

    # Simulation entity data — large hive-partitioned parquets kept as LazyFrames
    hydros_lf: pl.LazyFrame
    thermals_lf: pl.LazyFrame
    ncs_lf: pl.LazyFrame
    buses_lf: pl.LazyFrame
    exchanges_lf: pl.LazyFrame

    # Costs (~236 K rows) collected to pandas for chart_cost_* functions
    costs: pd.DataFrame

    # Optional line bounds
    line_bounds: pd.DataFrame

    # Entity name/metadata dictionaries
    names: dict[tuple[str, int], str]
    stage_labels: dict[int, str]
    hydro_bus_map: dict[int, int]
    thermal_meta: dict[int, dict]
    ncs_bus_map: dict[int, int]
    hydro_meta: dict[int, dict]
    bus_names: dict[int, str]
    non_fictitious_bus_ids: list[int]

    # Temporal resolution
    stage_hours: dict[int, float]
    block_hours: dict[tuple[int, int], float]

    # Block-hours DataFrame for weighted-average joins in chart functions
    # Columns: ["stage_id", "block_id", "_bh"]
    bh_df: pl.DataFrame

    # Line metadata for exchange calculations
    line_meta: list[dict]

    # LP load input data
    load_stats: pd.DataFrame
    load_factors_list: list[dict]

    # Performance / solver data (optional — empty DataFrame when absent)
    timing: pd.DataFrame
    solver_train: pd.DataFrame
    solver_sim: pd.DataFrame
    scaling_report: dict
    cut_selection: pd.DataFrame

    # Stochastic model output (optional)
    stochastic_available: bool
    inflow_stats_stoch: pd.DataFrame
    ar_coefficients: pd.DataFrame
    noise_openings: pd.DataFrame
    fitting_report: dict
    inflow_history: pd.DataFrame
    correlation: dict
    inflow_lags_lf: pl.LazyFrame

    # Output metadata (from metadata.json in each output subdirectory)
    training_metadata: dict
    simulation_metadata: dict
    policy_metadata: dict

    # New v2 fields: config
    config: dict
    discount_rate: float
    stages_data: dict

    # Resolved LP bounds (optional)
    lp_bounds: pd.DataFrame

    # Generic constraints (optional)
    gc_constraints: list[dict]
    gc_bounds: pd.DataFrame
    gc_violations: pd.DataFrame

    # Input constraint bounds (optional — ticket-002)
    hydro_bounds: pd.DataFrame
    thermal_bounds: pd.DataFrame
    ncs_stats: pd.DataFrame
    exchange_factors: list[dict]
    retry_histogram: pd.DataFrame

    # Summary counts
    n_scenarios: int
    n_stages: int

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------

    @classmethod
    def load(cls, case_dir: Path) -> DashboardData:
        """Load all dashboard data from a Cobre case directory.

        Replicates the data loading logic from ``build_dashboard()`` lines
        6137-6323 and the sim_manifest filtering at lines 6655-6665.

        Raises ``FileNotFoundError`` if required files (convergence.parquet,
        stages.json) are missing.  Optional files fall back to empty
        DataFrames / empty dicts.
        """
        print(f"Loading data from {case_dir} ...")

        # ------------------------------------------------------------------
        # Training data — small single-file parquets, load as pandas
        # ------------------------------------------------------------------
        conv = pq.read_table(
            case_dir / "output" / "training" / "convergence.parquet"
        ).to_pandas()

        # ------------------------------------------------------------------
        # Simulation entity data — large hive-partitioned parquets, LazyFrames
        # ------------------------------------------------------------------
        hydros_lf = scan_entity(case_dir, "hydros")
        thermals_lf = scan_entity(case_dir, "thermals")
        ncs_lf = scan_entity(case_dir, "non_controllables")
        buses_lf = scan_entity(case_dir, "buses")
        exchanges_lf = scan_entity(case_dir, "exchanges")

        inflow_lags_dir = case_dir / "output" / "simulation" / "inflow_lags"
        if inflow_lags_dir.exists():
            inflow_lags_lf = pl.scan_parquet(
                str(inflow_lags_dir / "**" / "*.parquet"), hive_partitioning=True
            )
        else:
            inflow_lags_lf = pl.LazyFrame()

        # Costs is stage-level only (~236 K rows total), collect to pandas
        # for chart_cost_* functions
        costs = (
            pl.scan_parquet(
                str(case_dir / "output" / "simulation" / "costs" / "**" / "*.parquet"),
                hive_partitioning=True,
            )
            .collect(engine="streaming")
            .to_pandas()
        )

        lb_path = case_dir / "constraints" / "line_bounds.parquet"
        line_bounds = (
            pq.read_table(lb_path).to_pandas() if lb_path.exists() else pd.DataFrame()
        )

        hb_path = case_dir / "constraints" / "hydro_bounds.parquet"
        hydro_bounds = (
            pq.read_table(hb_path).to_pandas() if hb_path.exists() else pd.DataFrame()
        )

        tb_path = case_dir / "constraints" / "thermal_bounds.parquet"
        thermal_bounds = (
            pq.read_table(tb_path).to_pandas() if tb_path.exists() else pd.DataFrame()
        )

        names = load_names(case_dir)
        stage_labels = load_stage_labels(case_dir)
        hydro_bus_map = load_hydro_bus_map(case_dir)
        thermal_meta = load_thermal_metadata(case_dir)
        ncs_bus_map = load_ncs_bus_map(case_dir)
        hydro_meta = load_hydro_metadata(case_dir)

        # Build bus_names dict: id -> name from names dict
        bus_names = {
            eid: nm for (entity, eid), nm in names.items() if entity == "buses"
        }

        # Config (optional — v2 tabs need run configuration)
        config_path = case_dir / "config.json"
        config: dict = {}
        if config_path.exists():
            try:
                with config_path.open() as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Failed to parse config.json; using empty dict")
        discount_rate = float(config.get("discount_rate", 0.0))

        # Stage hours: total hours per stage (sum of block hours)
        stages_json_path = case_dir / "stages.json"
        with stages_json_path.open() as f:
            stages_data = json.load(f)
        stage_hours: dict[int, float] = {}
        for s in stages_data["stages"]:
            stage_hours[s["id"]] = sum(b["hours"] for b in s["blocks"])

        # Line metadata for exchange calculations
        lines_path = case_dir / "system" / "lines.json"
        with lines_path.open() as f:
            line_meta: list[dict] = json.load(f)["lines"]

        # Load input data for LP load computation
        load_stats = pq.read_table(
            case_dir / "scenarios" / "load_seasonal_stats.parquet"
        ).to_pandas()
        lf_path = case_dir / "scenarios" / "load_factors.json"
        with lf_path.open() as f:
            load_factors_list: list[dict] = json.load(f)["load_factors"]

        non_fictitious_bus_ids = compute_non_fictitious_bus_ids(load_stats)

        ncs_stats_path = case_dir / "scenarios" / "non_controllable_stats.parquet"
        ncs_stats = (
            pq.read_table(ncs_stats_path).to_pandas()
            if ncs_stats_path.exists()
            else pd.DataFrame()
        )

        ih_path = case_dir / "scenarios" / "inflow_history.parquet"
        inflow_history = (
            pq.read_table(ih_path).to_pandas() if ih_path.exists() else pd.DataFrame()
        )

        ef_path = case_dir / "constraints" / "exchange_factors.json"
        exchange_factors: list[dict] = []
        if ef_path.exists():
            with ef_path.open() as f:
                exchange_factors = json.load(f).get("exchange_factors", [])

        block_hours: dict[tuple[int, int], float] = {}
        for s in stages_data["stages"]:
            for b in s["blocks"]:
                block_hours[(s["id"], b["id"])] = b["hours"]

        # Build block-hours DataFrame once for weighted-average joins
        # across all chart functions
        bh_keys = list(block_hours.keys())
        bh_df = pl.DataFrame(
            {
                "stage_id": [k[0] for k in bh_keys],
                "block_id": [k[1] for k in bh_keys],
                "_bh": [block_hours[k] for k in bh_keys],
            }
        )

        # Performance / solver data (optional — graceful fallback to empty frames)
        timing_path = case_dir / "output" / "training" / "timing" / "iterations.parquet"
        timing = (
            pq.read_table(timing_path).to_pandas()
            if timing_path.exists()
            else pd.DataFrame()
        )
        solver_train_path = (
            case_dir / "output" / "training" / "solver" / "iterations.parquet"
        )
        solver_train = (
            pq.read_table(solver_train_path).to_pandas()
            if solver_train_path.exists()
            else pd.DataFrame()
        )
        sim_solver_path = (
            case_dir / "output" / "simulation" / "solver" / "iterations.parquet"
        )
        solver_sim = (
            pq.read_table(sim_solver_path).to_pandas()
            if sim_solver_path.exists()
            else pd.DataFrame()
        )
        scaling_path = case_dir / "output" / "training" / "scaling_report.json"
        if scaling_path.exists():
            with scaling_path.open() as f:
                scaling_report: dict = json.load(f)
        else:
            scaling_report = {}
        cs_path = (
            case_dir / "output" / "training" / "cut_selection" / "iterations.parquet"
        )
        cut_selection = (
            pq.read_table(cs_path).to_pandas() if cs_path.exists() else pd.DataFrame()
        )
        retry_path = (
            case_dir / "output" / "training" / "solver" / "retry_histogram.parquet"
        )
        retry_histogram = (
            pq.read_table(retry_path).to_pandas()
            if retry_path.exists()
            else pd.DataFrame()
        )
        stochastic_dir = case_dir / "output" / "stochastic"
        stochastic_available = stochastic_dir.exists()
        if stochastic_available:
            inflow_stats_stoch = pq.read_table(
                stochastic_dir / "inflow_seasonal_stats.parquet"
            ).to_pandas()
            ar_coefficients = pq.read_table(
                stochastic_dir / "inflow_ar_coefficients.parquet"
            ).to_pandas()
            noise_openings = pq.read_table(
                stochastic_dir / "noise_openings.parquet"
            ).to_pandas()
            fitting_report: dict = json.load(
                (stochastic_dir / "fitting_report.json").open()
            )
            corr_path = stochastic_dir / "correlation.json"
            if corr_path.exists():
                correlation: dict = json.load(corr_path.open())
            else:
                logger.warning(
                    "output/stochastic/correlation.json missing; using empty dict"
                )
                correlation = {}
        else:
            inflow_stats_stoch = pd.DataFrame()
            ar_coefficients = pd.DataFrame()
            noise_openings = pd.DataFrame()
            fitting_report = {}
            correlation = {}

        # Output metadata (from metadata.json in each output subdirectory)
        def _load_metadata(subdir: str) -> dict:
            meta_path = case_dir / "output" / subdir / "metadata.json"
            if not meta_path.exists():
                return {}
            try:
                with meta_path.open() as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Failed to parse output/%s/metadata.json", subdir)
                return {}

        training_metadata = _load_metadata("training")
        simulation_metadata = _load_metadata("simulation")
        policy_metadata = _load_metadata("policy")

        # Load resolved LP bounds dictionary (actual bounds used by the solver)
        bounds_path = (
            case_dir / "output" / "training" / "dictionaries" / "bounds.parquet"
        )
        lp_bounds = (
            pq.read_table(bounds_path).to_pandas()
            if bounds_path.exists()
            else pd.DataFrame()
        )

        # Generic constraints data (optional — graceful fallback when absent)
        gc_path = case_dir / "constraints" / "generic_constraints.json"
        gc_constraints: list[dict] = []
        if gc_path.exists():
            with gc_path.open() as f:
                gc_data = json.load(f)
            gc_constraints = gc_data.get("constraints", [])

        gc_bounds_path = case_dir / "constraints" / "generic_constraint_bounds.parquet"
        gc_bounds = (
            pq.read_table(gc_bounds_path).to_pandas()
            if gc_bounds_path.exists()
            else pd.DataFrame()
        )

        # violations/generic: collect to pandas once (needed by
        # build_constraints_summary_table)
        gc_viol_dir = case_dir / "output" / "simulation" / "violations" / "generic"
        if gc_viol_dir.exists():
            gc_violations = (
                pl.scan_parquet(
                    str(gc_viol_dir / "**" / "*.parquet"),
                    hive_partitioning=True,
                )
                .collect(engine="streaming")
                .to_pandas()
            )
        else:
            gc_violations = pd.DataFrame()

        case_name = case_dir.resolve().name
        n_scenarios = costs["scenario_id"].nunique()
        n_stages = costs["stage_id"].nunique()
        print(
            f"  {n_scenarios} scenarios, {n_stages} stages,"
            f" {len(stage_labels)} stage labels"
        )

        # Filter simulation solver to actual scenario count from metadata
        actual_sim_scenarios = simulation_metadata.get("scenarios", {}).get(
            "completed", n_scenarios
        )
        if not solver_sim.empty:
            solver_sim = solver_sim.head(actual_sim_scenarios)

        return cls(
            case_dir=case_dir,
            case_name=case_name,
            conv=conv,
            hydros_lf=hydros_lf,
            thermals_lf=thermals_lf,
            ncs_lf=ncs_lf,
            buses_lf=buses_lf,
            exchanges_lf=exchanges_lf,
            costs=costs,
            line_bounds=line_bounds,
            names=names,
            stage_labels=stage_labels,
            hydro_bus_map=hydro_bus_map,
            thermal_meta=thermal_meta,
            ncs_bus_map=ncs_bus_map,
            hydro_meta=hydro_meta,
            bus_names=bus_names,
            non_fictitious_bus_ids=non_fictitious_bus_ids,
            stage_hours=stage_hours,
            block_hours=block_hours,
            bh_df=bh_df,
            line_meta=line_meta,
            load_stats=load_stats,
            load_factors_list=load_factors_list,
            timing=timing,
            solver_train=solver_train,
            solver_sim=solver_sim,
            scaling_report=scaling_report,
            cut_selection=cut_selection,
            stochastic_available=stochastic_available,
            inflow_stats_stoch=inflow_stats_stoch,
            ar_coefficients=ar_coefficients,
            noise_openings=noise_openings,
            fitting_report=fitting_report,
            inflow_history=inflow_history,
            correlation=correlation,
            inflow_lags_lf=inflow_lags_lf,
            training_metadata=training_metadata,
            simulation_metadata=simulation_metadata,
            policy_metadata=policy_metadata,
            config=config,
            discount_rate=discount_rate,
            stages_data=stages_data,
            lp_bounds=lp_bounds,
            gc_constraints=gc_constraints,
            gc_bounds=gc_bounds,
            gc_violations=gc_violations,
            hydro_bounds=hydro_bounds,
            thermal_bounds=thermal_bounds,
            ncs_stats=ncs_stats,
            exchange_factors=exchange_factors,
            retry_histogram=retry_histogram,
            n_scenarios=n_scenarios,
            n_stages=n_stages,
        )
