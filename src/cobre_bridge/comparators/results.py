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

import pandas as pd
import polars as pl

from cobre_bridge.comparators.alignment import EntityAlignment
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)


@dataclass
class PercentileData:
    """Cobre simulation percentile statistics for the report.

    Each DataFrame has columns: entity_id, stage_id, and for each
    variable: ``{var}_p10``, ``{var}_p50``, ``{var}_p90``.
    """

    hydro: pl.DataFrame = field(default_factory=pl.DataFrame)
    thermal: pl.DataFrame = field(default_factory=pl.DataFrame)
    bus: pl.DataFrame = field(default_factory=pl.DataFrame)
    bus_aggregates: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_market: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_net_load: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_sin: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_hydro_means: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_bus_meta: dict[int, dict] = field(default_factory=dict)
    cobre_hydro_meta: dict[int, dict] = field(default_factory=dict)
    nw_bus_names: dict[int, str] = field(default_factory=dict)
    nw_hydro_names: dict[int, str] = field(default_factory=dict)
    nw_convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    cobre_convergence: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_costs: dict[str, float] = field(default_factory=dict)
    cobre_costs: dict[str, float] = field(default_factory=dict)
    # Per-stage immediate/future cost on the Cobre side — mean across
    # scenarios, in R$.  Joined with MEDIAS-SIN's COPER/CUSTO_FUTURO
    # (after the 10⁶ R$ unit conversion) by the overview-tab chart.
    cobre_stage_costs: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_offset: int = 0
    nw_max_stage: int | None = None

    # --- Epic 1: line interchange ---
    # Cobre per-(line_id, stage_id) p10/p50/p90 of net_flow_mw across
    # scenarios; bound table from constraints/line_bounds.parquet; metadata
    # list straight from system/lines.json["lines"]. NEWAVE side: mean
    # interchange per (line_id, stage_0based) read from int*.out NWLISTOP
    # files and aligned via EntityAlignment.lines.
    line: pl.DataFrame = field(default_factory=pl.DataFrame)
    line_bounds: pd.DataFrame = field(default_factory=pd.DataFrame)
    line_meta: list[dict] = field(default_factory=list)
    nw_line_means: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Epic 2: per-hydro derived flow variables ---
    # Cobre per-(hydro_id, stage_id) total_inflow_m3s (incremental +
    # upstream turbined+spilled) and total_outflow_m3s (turbined + spilled).
    # Outflow comes directly from the parquet ``outflow_m3s`` column when
    # present.
    hydro_total_flows: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Per-stage hydro operational bounds (dashboard overlay) ---
    # Per-(entity_id, stage_id) optional bound columns from
    # ``constraints/hydro_bounds.parquet`` — read by
    # ``read_cobre_hydro_per_stage_bounds``.  Columns are nullable per row:
    # NULL means "no per-stage override, fall back to the static value
    # from ``hydros.json``" (carried by ``cobre_hydro_meta``).
    cobre_hydro_per_stage_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- NEWAVE hydro slacks (VIOL_POS/NEG_VRETIRUH and VIOL_POS/NEG_EVAP) ---
    # Per-(entity_id, stage_id) flow-domain values for the four NEWAVE hydro
    # slack columns (water-withdrawal pos/neg and evaporation pos/neg),
    # aligned to Cobre IDs and stage 0-base.  Used by the plant-detail tab
    # to overlay NEWAVE on Cobre-only withdrawal slack panels and by the
    # Hydro Operation tab to drive per-bus and SIN-total slack aggregates.
    # Built by :func:`_compute_nw_hydro_slacks`.
    nw_hydro_slacks: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Epic 3: system spillage in MWmes ---
    # Per-stage stage-mean MW of system spillage ``spillage_m3s × ρ_eq``,
    # split into total / reservoir / run-of-river via the
    # ``max_storage_hm3 > 0`` discriminator.
    cobre_spillage_energy: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Performance / wall-clock timings ---
    # NEWAVE per-iteration ``backward_seconds`` / ``forward_seconds`` /
    # ``total_seconds`` from ``newave.tim``. NEWAVE stage labels (e.g.
    # ``"Tempo Total"``, ``"Calculo da Politica"``) → seconds. Cobre
    # total training duration in seconds from
    # ``output/training/metadata.json``.
    nw_tim_iterations: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_tim_stages: dict[str, float] = field(default_factory=dict)
    cobre_training_seconds: float = 0.0
    cobre_iteration_timing: pl.DataFrame = field(default_factory=pl.DataFrame)

    # --- Generic constraints (RE, AGRINT, VminOP) — LHS comparison ---
    # Constraint definitions and per-(stage, block) bound table read
    # straight from the converted Cobre case. ``lhs_newave`` evaluates
    # each constraint's LHS against NEWAVE outputs (MEDIAS-USIH GHIDUH +
    # int*.out interchanges) at stage_0based granularity; ``lhs_cobre``
    # does the same against Cobre simulation parquets, collapsed to
    # mean across scenarios and blocks.
    gc_constraints: list[dict] = field(default_factory=list)
    gc_bounds: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_lhs_newave: pl.DataFrame = field(default_factory=pl.DataFrame)
    gc_lhs_cobre: pl.DataFrame = field(default_factory=pl.DataFrame)


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
    within_tol_rate: float = 0.0  # fraction in [0, 1] within --tolerance
    mean_smape: float = 0.0  # symmetric MAPE in [0, 2]
    max_smape: float = 0.0
    correlation: float | None = None  # None when undefined (constant series)


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


def _smape(nw_value: float, cobre_value: float) -> float:
    """Symmetric mean absolute percentage error for one pair, in [0, 2].

    Robust to a near-zero reference (unlike ``|d| / |nw|``): returns 0 when
    both values are effectively zero (perfect agreement).
    """
    denom = (abs(nw_value) + abs(cobre_value)) / 2.0
    if denom <= 1e-12:
        return 0.0
    return abs(nw_value - cobre_value) / denom


def _within_tolerance(nw_value: float, cobre_value: float, tolerance: float) -> bool:
    """True if Cobre is within ``tolerance`` (relative) of the NEWAVE reference.

    When the reference is ~0, counts as a match only if Cobre is also ~0 (both
    effectively zero); otherwise uses ``|nw - cobre| <= tolerance * |nw|``.
    The 1e-9 / 1e-6 floors are absolute and tuned for physical magnitudes
    (MW, m³/s, hm³); adjust if comparing very small-scale quantities.
    """
    denom = abs(nw_value)
    if denom <= 1e-9:
        return abs(cobre_value) <= 1e-6
    return abs(nw_value - cobre_value) <= tolerance * denom


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
    "QINCRUH": "inflow_m3s",
    "PIVARM": "water_value_per_hm3",
    "VEVAPUH": "evaporation_m3s",
    # Cobre emits water withdrawal as an input constant (target) in
    # ``constraints/hydro_bounds.parquet`` rather than as a per-stage
    # simulation result; ``read_cobre_hydro_withdrawal`` joins it onto
    # ``cobre_hydro`` so this map entry still resolves.
    "VRETIRUH": "withdrawal_m3s",
    # QAFLUH = total inflow at the plant: incremental + Σ upstream
    # (turbined + spilled). Computed Cobre-side by
    # ``read_cobre_hydro_total_flows``; merged into ``cobre_hydro``.
    "QAFLUH": "total_inflow_m3s",
}

_SYSTEM_VAR_MAP: dict[str, str] = {
    "CMO": "spot_price",
    "DEFT": "deficit_mw",
}


def _nw_stage_offset(nw_df: pl.DataFrame) -> int:
    """Return the minimum stage number in a NEWAVE MEDIAS DataFrame.

    NEWAVE v29+ MEDIAS columns are numbered from the study start month
    (e.g. 3 for March).  We subtract this offset to convert to 0-based
    stage indices matching Cobre convention.
    """
    stages = nw_df["stage"].drop_nulls()
    if stages.is_empty():
        return 1
    return int(stages.min())


def _build_gen_max_overlay(
    nw_hydro: pl.DataFrame,
    cobre_lp_gen_max: pl.DataFrame,
    nw_names: dict[int, str],
    cobre_meta: dict[int, dict],
    nw_offset: int,
) -> pl.DataFrame:
    """Build per-(entity_id, stage_id) generation upper-bound overlay.

    Returns columns ``entity_id``, ``stage_id``,
    ``nw_ghmax_fphc_mw`` (NEWAVE constant-FPH max generation, MWmes
    from MEDIAS-USIH), and ``cobre_lp_gen_max_mw`` (Cobre LP upper
    bound at block 0). Either column may be null when the source side
    is missing a value.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "nw_ghmax_fphc_mw": pl.Float64,
            "cobre_lp_gen_max_mw": pl.Float64,
        }
    )
    if cobre_lp_gen_max.is_empty() and nw_hydro.is_empty():
        return empty

    # NEWAVE GHMAX_FPHC → (cobre_id, stage_0based) lookup via name match.
    cobre_by_name: dict[str, int] = {
        meta["name"].strip().upper(): eid for eid, meta in cobre_meta.items()
    }
    nw_code_to_cobre_id: dict[int, int] = {}
    for nw_code, nw_name in nw_names.items():
        cid = cobre_by_name.get(nw_name.strip().upper())
        if cid is not None:
            nw_code_to_cobre_id[nw_code] = cid

    nw_rows: list[dict[str, float | int]] = []
    if not nw_hydro.is_empty():
        for row in nw_hydro.iter_rows(named=True):
            if row["value"] is None:
                continue
            var = str(row["variable"]).strip().upper()
            if var != "GHMAX_FPHC":
                continue
            code = int(row["newave_code"])
            cid = nw_code_to_cobre_id.get(code)
            if cid is None:
                continue
            stage = int(row["stage"]) - nw_offset
            nw_rows.append(
                {
                    "entity_id": cid,
                    "stage_id": stage,
                    "nw_ghmax_fphc_mw": float(row["value"]),
                }
            )

    nw_df = (
        pl.DataFrame(nw_rows).cast(
            {
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "nw_ghmax_fphc_mw": pl.Float64,
            }
        )
        if nw_rows
        else pl.DataFrame(
            schema={
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "nw_ghmax_fphc_mw": pl.Float64,
            }
        )
    )

    if cobre_lp_gen_max.is_empty():
        return nw_df

    return nw_df.join(
        cobre_lp_gen_max, on=["entity_id", "stage_id"], how="full", coalesce=True
    )


# MEDIAS reports VIOL_POS_VRETIRUH / VIOL_NEG_VRETIRUH and VIOL_POS_EVAP /
# VIOL_NEG_EVAP as monthly volumes in hm³ (same convention as VRETIRUH /
# VEVAPUH).  Cobre emits the matching slacks in m³/s.  NEWAVE's internal
# reports round the month-length factor to 2.63 (≈ 730 h × 3600 s / 1e6), so
# dividing by 2.63 yields apples-to-apples flows matching what
# ``_compare_hydros`` already does for the VEVAPUH / VRETIRUH realized series.
#
# WITHDRAWAL pos/neg are SWAPPED on purpose: Cobre's sign convention for the
# withdrawal slack is the inverse of NEWAVE's, so the column NEWAVE calls
# ``VIOL_POS_VRETIRUH`` lines up physically with Cobre's
# ``water_withdrawal_violation_neg_m3s`` (and vice versa).  Mapping them this
# way keeps the comparison HTML's "Withdrawal Slack Pos / Neg" panels
# self-consistent under NEWAVE's labelling — the display labels in
# ``_HYDRO_COBRE_ONLY_VARIABLES`` / ``report_builder.slack_specs`` are
# correspondingly swapped so each panel shows the column that matches its
# header.  Evaporation pos/neg share NEWAVE's convention and don't need the
# swap.
_NW_HYDRO_SLACK_VARS: dict[str, str] = {
    "VIOL_POS_VRETIRUH": "water_withdrawal_violation_neg_m3s",
    "VIOL_NEG_VRETIRUH": "water_withdrawal_violation_pos_m3s",
    "VIOL_POS_EVAP": "evaporation_violation_pos_m3s",
    "VIOL_NEG_EVAP": "evaporation_violation_neg_m3s",
}
_NW_HYDRO_SLACK_COLUMNS: tuple[str, ...] = tuple(_NW_HYDRO_SLACK_VARS.values())


def _compute_nw_hydro_slacks(
    nw_hydro: pl.DataFrame,
    nw_names: dict[int, str],
    cobre_meta: dict[int, dict],
) -> pl.DataFrame:
    """Build a per-(entity_id, stage_id) frame of NEWAVE hydro slacks.

    Surfaces ``VIOL_POS_VRETIRUH`` / ``VIOL_NEG_VRETIRUH`` (water withdrawal)
    and ``VIOL_POS_EVAP`` / ``VIOL_NEG_EVAP`` (evaporation) from MEDIAS-USIH
    aligned to Cobre's ``entity_id`` and ``stage_id`` (0-based) conventions
    and converted from hm³/month to m³/s with the same /2.63 factor used by
    :func:`_compare_hydros` for the realized withdrawal / evaporation series.
    The chart layer joins this against the matching Cobre slack columns so
    the Hydro Operation tab can render NEWAVE alongside Cobre on per-bus and
    SIN-total slack panels.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "water_withdrawal_violation_pos_m3s": pl.Float64,
            "water_withdrawal_violation_neg_m3s": pl.Float64,
            "evaporation_violation_pos_m3s": pl.Float64,
            "evaporation_violation_neg_m3s": pl.Float64,
        }
    )
    if nw_hydro.is_empty() or not nw_names or not cobre_meta:
        return empty

    cobre_by_name: dict[str, int] = {
        meta["name"].strip().upper(): eid for eid, meta in cobre_meta.items()
    }
    nw_code_to_cobre_id: dict[int, int] = {}
    for nw_code, nw_name in nw_names.items():
        cid = cobre_by_name.get(nw_name.strip().upper())
        if cid is not None:
            nw_code_to_cobre_id[nw_code] = cid
    if not nw_code_to_cobre_id:
        return empty

    offset = _nw_stage_offset(nw_hydro)
    rows: dict[tuple[int, int], dict[str, float]] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        var_raw = str(row["variable"]).strip().upper()
        col = _NW_HYDRO_SLACK_VARS.get(var_raw)
        if col is None:
            continue
        code = int(row["newave_code"])
        cid = nw_code_to_cobre_id.get(code)
        if cid is None:
            continue
        stage = int(row["stage"]) - offset
        rows.setdefault((cid, stage), {})[col] = float(row["value"]) / 2.63

    if not rows:
        return empty

    return (
        pl.DataFrame(
            [
                {
                    "entity_id": eid,
                    "stage_id": sid,
                    **{col: cols.get(col, 0.0) for col in _NW_HYDRO_SLACK_COLUMNS},
                }
                for (eid, sid), cols in rows.items()
            ]
        )
        .cast(
            {
                "entity_id": pl.Int64,
                "stage_id": pl.Int64,
                "water_withdrawal_violation_pos_m3s": pl.Float64,
                "water_withdrawal_violation_neg_m3s": pl.Float64,
                "evaporation_violation_pos_m3s": pl.Float64,
                "evaporation_violation_neg_m3s": pl.Float64,
            }
        )
        .sort("entity_id", "stage_id")
    )


def _compare_hydros(
    nw_hydro: pl.DataFrame,
    cobre_hydro: pl.DataFrame,
    nw_names: dict[int, str],
    cobre_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare hydro results by matching plant names."""
    results: list[ResultComparison] = []

    offset = _nw_stage_offset(nw_hydro)

    # Build Cobre name→(id, min_storage) lookup.
    cobre_by_name: dict[str, tuple[int, float]] = {}
    for eid, meta in cobre_meta.items():
        name_upper = meta["name"].strip().upper()
        min_stor = meta.get("min_storage_hm3", 0.0)
        cobre_by_name[name_upper] = (eid, min_stor)

    # Match NEWAVE codes to Cobre IDs by name.
    matched: dict[
        int, tuple[int, str, float]
    ] = {}  # nw_code→(cobre_id, name, min_stor)
    for nw_code, nw_name in nw_names.items():
        name_upper = nw_name.strip().upper()
        hit = cobre_by_name.get(name_upper)
        if hit is not None:
            matched[nw_code] = (hit[0], nw_name.strip(), hit[1])

    # Build NEWAVE lookup: (nw_code, stage, variable) -> value
    nw_lookup: dict[tuple[int, int, str], float] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        var = str(row["variable"]).strip().upper()
        mapped = _HYDRO_VAR_MAP.get(var)
        if mapped is None:
            continue
        nw_lookup[(code, stage, mapped)] = float(row["value"])

    # Reconstruct realized evaporation and withdrawal from the NEWAVE LP
    # slack columns. VEVAPUH/VRETIRUH are reported as the *scheduled*
    # values; VIOL_POS_* / VIOL_NEG_* track over- and under-application
    # respectively. Realized = scheduled + POS − NEG (same convention as
    # Cobre's matrix.rs water-balance row). Reading the slacks directly
    # from the long-format ``nw_hydro`` frame avoids adding them to
    # ``_HYDRO_VAR_MAP`` (which would spuriously emit ResultComparison
    # rows for the slacks themselves).
    nw_slacks: dict[tuple[int, int, str], float] = {}
    for row in nw_hydro.iter_rows(named=True):
        if row["value"] is None:
            continue
        var_raw = str(row["variable"]).strip().upper()
        if var_raw not in (
            "VIOL_POS_EVAP",
            "VIOL_NEG_EVAP",
            "VIOL_POS_VRETIRUH",
            "VIOL_NEG_VRETIRUH",
        ):
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        nw_slacks[(code, stage, var_raw)] = float(row["value"])

    _SLACK_PAIRS = {
        "evaporation_m3s": ("VIOL_POS_EVAP", "VIOL_NEG_EVAP"),
        "withdrawal_m3s": ("VIOL_POS_VRETIRUH", "VIOL_NEG_VRETIRUH"),
    }
    for key in list(nw_lookup.keys()):
        code, stage, mapped = key
        slack_vars = _SLACK_PAIRS.get(mapped)
        if slack_vars is None:
            continue
        pos = nw_slacks.get((code, stage, slack_vars[0]), 0.0)
        neg = nw_slacks.get((code, stage, slack_vars[1]), 0.0)
        nw_lookup[key] = nw_lookup[key] + pos - neg

    # Derive NEWAVE total outflow = turbined + spillage (no MEDIAS code).
    derived: dict[tuple[int, int, str], float] = {}
    for (code, stage, var), val in nw_lookup.items():
        if var != "turbined_m3s":
            continue
        spill = nw_lookup.get((code, stage, "spillage_m3s"))
        if spill is None:
            continue
        derived[(code, stage, "outflow_m3s")] = val + spill
    nw_lookup.update(derived)

    # Build Cobre lookup: (entity_id, stage, variable) -> value
    cobre_lookup: dict[tuple[int, int, str], float] = {}
    hydro_value_cols = list(_HYDRO_VAR_MAP.values()) + ["outflow_m3s"]
    for row in cobre_hydro.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        for col in hydro_value_cols:
            val = row.get(col)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                cobre_lookup[(eid, sid, col)] = round(float(val), 2)

    # Compare matched pairs.
    for nw_code, (cobre_id, name, min_stor) in sorted(matched.items()):
        for (code, stage, var), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            cobre_val = cobre_lookup.get((cobre_id, stage, var))
            if cobre_val is None:
                continue

            # NEWAVE reports useful storage (storage - vol_min).
            # Add min_storage to align with Cobre absolute storage.
            if var == "storage_final_hm3":
                nw_val = nw_val + min_stor
            elif var in ("evaporation_m3s", "withdrawal_m3s"):
                # MEDIAS-USIH VEVAPUH and VRETIRUH are reported as
                # monthly *volume* in hm³, not flow. Cobre emits m³/s.
                # NEWAVE rounds the month-length factor to 2.63 in its
                # internal output reporting (≈ 730 h × 3600 s / 10⁶);
                # using the same rounded constant here makes the
                # comparison apples-to-apples. The converter side keeps
                # the exact 2.628 (see network.py:C_M3S2HM3) because the
                # input-data conversion has no analogous rounding.
                # Handled per-variable rather than via a V*-prefix
                # heuristic to avoid mis-scaling future MEDIAS columns
                # that happen to start with V.
                nw_val = nw_val / 2.63

            results.append(
                _make_result(
                    "hydro",
                    name,
                    nw_code,
                    cobre_id,
                    stage,
                    var,
                    nw_val,
                    cobre_val,
                )
            )

    return results


def _compare_thermals(
    nw_thermal: pl.DataFrame,
    cobre_thermal: pl.DataFrame,
    nw_names: dict[int, str],
    cobre_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare thermal results by matching plant names."""
    results: list[ResultComparison] = []

    offset = _nw_stage_offset(nw_thermal)

    # Name-based matching.
    cobre_by_name: dict[str, int] = {}
    for eid, meta in cobre_meta.items():
        cobre_by_name[meta["name"].strip().upper()] = eid

    matched: dict[int, tuple[int, str]] = {}  # nw_code→(cobre_id, name)
    for nw_code, nw_name in nw_names.items():
        hit = cobre_by_name.get(nw_name.strip().upper())
        if hit is not None:
            matched[nw_code] = (hit, nw_name.strip())

    nw_lookup: dict[tuple[int, int], float] = {}
    for row in nw_thermal.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
        nw_lookup[(code, stage)] = float(row["value"])

    cobre_lookup: dict[tuple[int, int], float] = {}
    for row in cobre_thermal.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        val = row.get("generation_mw")
        if val is not None:
            cobre_lookup[(eid, sid)] = round(float(val), 2)

    for nw_code, (cobre_id, name) in sorted(matched.items()):
        for (code, stage), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            cobre_val = cobre_lookup.get((cobre_id, stage))
            if cobre_val is None:
                continue
            results.append(
                _make_result(
                    "thermal",
                    name,
                    nw_code,
                    cobre_id,
                    stage,
                    "generation_mw",
                    nw_val,
                    cobre_val,
                )
            )

    return results


def _compare_buses(
    nw_system: pl.DataFrame,
    cobre_bus: pl.DataFrame,
    nw_names: dict[int, str],
    cobre_meta: dict[int, dict],
) -> list[ResultComparison]:
    """Compare bus/subsystem results by matching names."""
    results: list[ResultComparison] = []

    offset = _nw_stage_offset(nw_system)

    # Name-based matching.
    cobre_by_name: dict[str, int] = {}
    for eid, meta in cobre_meta.items():
        cobre_by_name[meta["name"].strip().upper()] = eid

    matched: dict[int, tuple[int, str]] = {}  # nw_code→(cobre_id, name)
    for nw_code, nw_name in nw_names.items():
        hit = cobre_by_name.get(nw_name.strip().upper())
        if hit is not None:
            matched[nw_code] = (hit, nw_name.strip())

    nw_lookup: dict[tuple[int, int, str], float] = {}
    for row in nw_system.iter_rows(named=True):
        if row["value"] is None:
            continue
        code = int(row["newave_code"])
        if code not in matched:
            continue
        stage = int(row["stage"]) - offset
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

    for nw_code, (cobre_id, name) in sorted(matched.items()):
        for (code, stage, var), nw_val in sorted(nw_lookup.items()):
            if code != nw_code:
                continue
            cobre_val = cobre_lookup.get((cobre_id, stage, var))
            if cobre_val is None:
                continue
            results.append(
                _make_result(
                    "bus",
                    name,
                    nw_code,
                    cobre_id,
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

    def _load(df: pl.DataFrame) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for row in df.iter_rows(named=True):
            lb = row["lower_bound"]
            ub = row["upper_bound_mean"]
            # Skip iterations where either bound is missing — robust against
            # pmo.dat layouts that inewave parses partially (NaNs in tail rows).
            if lb is None or ub is None:
                continue
            out[int(row["iteration"])] = {
                "lower_bound": float(lb),
                "upper_bound_mean": float(ub),
            }
        return out

    nw_lookup = _load(nw_conv)
    cobre_lookup = _load(cobre_conv)

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


_LINE_VAR_MAP: dict[str, str] = {
    "INTERC": "net_flow_mw",
}


def _compare_lines(
    nw_intercambio: pl.DataFrame,
    cobre_line: pl.DataFrame,
    alignment: EntityAlignment,
    nw_offset: int,
) -> list[ResultComparison]:
    """Compare NEWAVE intercâmbio to Cobre net line flow.

    NEWAVE NWLISTOP ``int*.out`` files report directional flow (mean MW
    over the month) per (from, to) submarket pair. The ``alignment``
    contains ``LineEntity`` objects already matched by submarket-pair to
    Cobre line IDs (see ``build_entity_alignment``).

    Pre-study int*.out stages (NEWAVE writes all absolute calendar
    months, including those before the study horizon) are filtered out
    by passing the MEDIAS-derived ``nw_offset`` — these rows would
    otherwise misalign with Cobre's ``stage_id`` numbering, which
    starts at the first study month.
    """
    results: list[ResultComparison] = []
    if nw_intercambio.is_empty() or cobre_line.is_empty():
        return results

    offset = nw_offset

    # (nw_de, nw_para) → (cobre_line_id, name)
    matched: dict[tuple[int, int], tuple[int, str]] = {}
    for ln in alignment.lines:
        matched[(ln.newave_de, ln.newave_para)] = (ln.cobre_line_id, ln.name)

    # NEWAVE lookup: (cobre_line_id, stage_0based) -> mean MW.
    nw_lookup: dict[tuple[int, int], float] = {}
    for row in nw_intercambio.iter_rows(named=True):
        if row["value"] is None:
            continue
        key = (int(row["from_submarket_code"]), int(row["to_submarket_code"]))
        match = matched.get(key)
        if match is None:
            continue
        cobre_line_id, _name = match
        stage_0based = int(row["stage"]) - offset
        if stage_0based < 0:
            continue
        nw_lookup[(cobre_line_id, stage_0based)] = float(row["value"])

    # Cobre lookup: (entity_id, stage_id) -> net_flow_mw.
    cobre_lookup: dict[tuple[int, int], float] = {}
    for row in cobre_line.iter_rows(named=True):
        eid = int(row["entity_id"])
        sid = int(row["stage_id"])
        val = row.get("net_flow_mw")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        cobre_lookup[(eid, sid)] = float(val)

    # Emit comparison rows in (line, stage) order.
    by_line: dict[int, str] = {ln.cobre_line_id: ln.name for ln in alignment.lines}
    code_by_line: dict[int, tuple[int, int]] = {
        ln.cobre_line_id: (ln.newave_de, ln.newave_para) for ln in alignment.lines
    }
    for cobre_line_id in sorted(by_line.keys()):
        for (eid, sid), nw_val in sorted(nw_lookup.items()):
            if eid != cobre_line_id:
                continue
            cobre_val = cobre_lookup.get((eid, sid))
            if cobre_val is None:
                continue
            de, _para = code_by_line.get(cobre_line_id, (0, 0))
            results.append(
                _make_result(
                    "line",
                    by_line[cobre_line_id],
                    de,
                    cobre_line_id,
                    sid,
                    "net_flow_mw",
                    nw_val,
                    round(cobre_val, 2),
                )
            )
    return results


_SYSTEM_SPILLAGE_VAR_MAP: dict[str, str] = {
    "VERTOT": "spill_energy_total_mw",
    "VERTCONT": "spill_energy_reservoir_mw",
    "VERTFIO": "spill_energy_rorov_mw",
}


def _compare_system_spillage(
    nw_sin: pl.DataFrame,
    cobre_spill_energy: pl.DataFrame,
) -> list[ResultComparison]:
    """Compare system spillage in MWmes between NEWAVE SIN and Cobre.

    NEWAVE MEDIAS-SIN reports ``VERTOT`` (total), ``VERTcont``
    (reservoir cascades), and ``VERTfio`` (run-of-river) — all in
    MWmes (stage-mean MW). Cobre values come from
    :func:`read_cobre_spillage_energy` which already produces
    stage-mean MW per category.
    """
    results: list[ResultComparison] = []
    if nw_sin.is_empty() or cobre_spill_energy.is_empty():
        return results

    offset = _nw_stage_offset(nw_sin)

    # NEWAVE side: (stage_0based, mapped_var) -> value.
    nw_lookup: dict[tuple[int, str], float] = {}
    for row in nw_sin.iter_rows(named=True):
        if row["value"] is None:
            continue
        var = str(row["variable"]).strip().upper()
        mapped = _SYSTEM_SPILLAGE_VAR_MAP.get(var)
        if mapped is None:
            continue
        stage = int(row["stage"]) - offset
        nw_lookup[(stage, mapped)] = float(row["value"])

    cobre_cols = {
        "spill_energy_total_mw": "total_mw",
        "spill_energy_reservoir_mw": "reservoir_mw",
        "spill_energy_rorov_mw": "rorov_mw",
    }
    cobre_lookup: dict[tuple[int, str], float] = {}
    for row in cobre_spill_energy.iter_rows(named=True):
        sid = int(row["stage_id"])
        for mapped, col in cobre_cols.items():
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue
            cobre_lookup[(sid, mapped)] = float(val)

    for (stage, mapped), nw_val in sorted(nw_lookup.items()):
        cobre_val = cobre_lookup.get((stage, mapped))
        if cobre_val is None:
            continue
        results.append(
            _make_result(
                "system_spillage",
                "SIN",
                0,
                0,
                stage,
                mapped,
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
) -> tuple[list[ResultComparison], PercentileData]:
    """Compare NEWAVE output results against Cobre simulation means.

    Entities are matched by **name** (case-insensitive) rather than by
    converter-assigned IDs, so the comparison works even when the Cobre
    case was built by a different tool.

    Returns
    -------
    tuple[list[ResultComparison], PercentileData]
        Comparison results and Cobre simulation percentile statistics.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE input file paths (for locating pmo.dat).
    id_map:
        Entity ID mapping (used only for productivity fallback).
    alignment:
        Pre-built entity alignment (used only for productivity).
    cobre_output_dir:
        Path to Cobre output directory.
    tolerance:
        Relative tolerance for results comparison (informational).

    """
    from cobre_bridge.comparators.alignment import _read_reference_names
    from cobre_bridge.comparators.cobre_readers import (
        read_cobre_bus_aggregates,
        read_cobre_bus_means,
        read_cobre_bus_metadata,
        read_cobre_bus_percentiles,
        read_cobre_convergence,
        read_cobre_cost_breakdown,
        read_cobre_hydro_means,
        read_cobre_hydro_metadata,
        read_cobre_hydro_per_stage_bounds,
        read_cobre_hydro_percentiles,
        read_cobre_hydro_total_flows,
        read_cobre_hydro_withdrawal,
        read_cobre_iteration_timing,
        read_cobre_line_means,
        read_cobre_line_percentiles,
        read_cobre_lp_max_generation,
        read_cobre_spillage_energy,
        read_cobre_stage_costs,
        read_cobre_thermal_means,
        read_cobre_thermal_metadata,
        read_cobre_thermal_percentiles,
        read_cobre_training_duration,
    )
    from cobre_bridge.comparators.newave_readers import (
        _find_saidas_dir,
        read_medias_hydro,
        read_medias_market,
        read_medias_sin,
        read_medias_system,
        read_medias_thermal,
        read_newave_net_load,
        read_newave_tim_iterations,
        read_newave_tim_stages,
        read_nwlistop_intercambio,
        read_pmo_convergence,
        read_pmo_cost_breakdown,
        read_pmo_productivity,
    )

    results: list[ResultComparison] = []

    # Read entity names from both sides.
    nw_hydro_names, nw_thermal_names, nw_bus_names = _read_reference_names(nw_files)
    cobre_hydro_meta = read_cobre_hydro_metadata(cobre_output_dir)
    cobre_thermal_meta = read_cobre_thermal_metadata(cobre_output_dir)
    cobre_bus_meta = read_cobre_bus_metadata(cobre_output_dir)

    # Locate NEWAVE saidas directory.
    saidas_dir = _find_saidas_dir(nw_files.directory)

    nw_offset = 0
    nw_max_stage_1based: int | None = None
    cobre_hydro = pl.DataFrame()
    nw_hydro_slacks = pl.DataFrame()

    # --- Hydro comparison ---
    if saidas_dir is not None:
        nw_hydro = read_medias_hydro(saidas_dir)
        cobre_hydro = read_cobre_hydro_means(cobre_output_dir)
        if not cobre_hydro.is_empty():
            # Merge derived per-(hydro, stage) quantities:
            # - total_inflow_m3s (≡ QAFLUH): incremental + Σ upstream outflow
            # - withdrawal_m3s (≡ VRETIRUH): input target from hydro_bounds
            total_flows = read_cobre_hydro_total_flows(cobre_output_dir, cobre_hydro)
            if not total_flows.is_empty():
                cobre_hydro = cobre_hydro.join(
                    total_flows, on=["entity_id", "stage_id"], how="left"
                )
            withdrawal = read_cobre_hydro_withdrawal(cobre_output_dir)
            if not withdrawal.is_empty():
                cobre_hydro = cobre_hydro.join(
                    withdrawal, on=["entity_id", "stage_id"], how="left"
                )
            # Reconstruct *realized* evaporation and withdrawal flows from
            # the LP slacks (Cobre's matrix.rs water-balance row uses
            #    +ζ × pos_slack − ζ × neg_slack = ζ × (base − scheduled)
            # so realized = scheduled + pos − neg).
            # ``evaporation_m3s`` from the simulation parquet is already
            # the LP-output (realized) flow — adding pos/neg slacks
            # represents the same correction applied symmetrically on
            # both sides so the comparison is apples-to-apples.
            recon_cols = {
                "evaporation_m3s": (
                    "evaporation_violation_pos_m3s",
                    "evaporation_violation_neg_m3s",
                ),
                "withdrawal_m3s": (
                    "water_withdrawal_violation_pos_m3s",
                    "water_withdrawal_violation_neg_m3s",
                ),
            }
            adjustments: list[pl.Expr] = []
            for base, (pos, neg) in recon_cols.items():
                if (
                    base in cobre_hydro.columns
                    and pos in cobre_hydro.columns
                    and neg in cobre_hydro.columns
                ):
                    adjustments.append(
                        (
                            pl.col(base).fill_null(0.0)
                            + pl.col(pos).fill_null(0.0)
                            - pl.col(neg).fill_null(0.0)
                        ).alias(base)
                    )
            if adjustments:
                cobre_hydro = cobre_hydro.with_columns(adjustments)
        if not nw_hydro.is_empty():
            nw_offset = _nw_stage_offset(nw_hydro)
            stages_col = nw_hydro["stage"].drop_nulls()
            if not stages_col.is_empty():
                max_val = stages_col.max()
                if max_val is not None:
                    nw_max_stage_1based = int(max_val)  # type: ignore[arg-type]
        if not cobre_hydro.is_empty():
            # Generation upper-bound overlay (NEWAVE GHMAX_FPHC vs Cobre LP).
            gen_max_overlay = _build_gen_max_overlay(
                nw_hydro,
                read_cobre_lp_max_generation(cobre_output_dir),
                nw_hydro_names,
                cobre_hydro_meta,
                nw_offset,
            )
            if not gen_max_overlay.is_empty():
                cobre_hydro = cobre_hydro.join(
                    gen_max_overlay, on=["entity_id", "stage_id"], how="left"
                )
        if not nw_hydro.is_empty() and not cobre_hydro.is_empty():
            _LOG.info("Comparing hydro results...")
            results.extend(
                _compare_hydros(nw_hydro, cobre_hydro, nw_hydro_names, cobre_hydro_meta)
            )
        nw_hydro_slacks = _compute_nw_hydro_slacks(
            nw_hydro, nw_hydro_names, cobre_hydro_meta
        )

        # --- Thermal comparison ---
        nw_thermal = read_medias_thermal(saidas_dir)
        cobre_thermal = read_cobre_thermal_means(cobre_output_dir)
        if not nw_thermal.is_empty() and not cobre_thermal.is_empty():
            _LOG.info("Comparing thermal results...")
            results.extend(
                _compare_thermals(
                    nw_thermal, cobre_thermal, nw_thermal_names, cobre_thermal_meta
                )
            )

        # --- Bus/system comparison ---
        nw_system = read_medias_system(saidas_dir)
        cobre_bus = read_cobre_bus_means(cobre_output_dir)
        if not nw_system.is_empty() and not cobre_bus.is_empty():
            _LOG.info("Comparing bus results...")
            results.extend(
                _compare_buses(nw_system, cobre_bus, nw_bus_names, cobre_bus_meta)
            )

        # --- Line interchange comparison (NWLISTOP int*.out) ---
        # nw_offset is the MEDIAS-derived study-start offset (e.g., 9 for
        # a September-start study). int*.out files emit absolute NEWAVE
        # stages including pre-study calendar months with all-zero
        # values, so we reuse this offset to filter them out and align
        # the remainder with Cobre's 0-based stage_id.
        nw_intercambio = read_nwlistop_intercambio(saidas_dir)
        cobre_line = read_cobre_line_means(cobre_output_dir)
        if not nw_intercambio.is_empty() and not cobre_line.is_empty():
            _LOG.info("Comparing line interchange...")
            results.extend(
                _compare_lines(nw_intercambio, cobre_line, alignment, nw_offset)
            )
    else:
        _LOG.warning("NEWAVE saidas/ directory not found; skipping MEDIAS comparison.")
        nw_intercambio = pl.DataFrame()
        cobre_line = pl.DataFrame()

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

    # --- Cost breakdown ---
    # NEWAVE typically runs a shorter horizon than Cobre.  Restrict
    # Cobre's cost sum to NEWAVE's stage range so the totals compare like-
    # for-like.  ``nw_max_stage_1based`` is the largest stage label
    # appearing in MEDIAS files; convert to Cobre's 0-based stage_id by
    # subtracting the NEWAVE start-month offset.
    nw_max_stage_0based: int | None = None
    if nw_max_stage_1based is not None:
        nw_max_stage_0based = nw_max_stage_1based - nw_offset

    _LOG.info(
        "Reading cost breakdowns (NEWAVE max stage_0based=%s)...", nw_max_stage_0based
    )
    nw_costs = read_pmo_cost_breakdown(nw_files.directory)
    cobre_costs = read_cobre_cost_breakdown(
        cobre_output_dir, max_stage_id=nw_max_stage_0based
    )
    cobre_stage_costs = read_cobre_stage_costs(cobre_output_dir)
    if nw_max_stage_0based is not None and not cobre_stage_costs.is_empty():
        cobre_stage_costs = cobre_stage_costs.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )

    cobre_hydro_per_stage_bounds = read_cobre_hydro_per_stage_bounds(cobre_output_dir)
    if nw_max_stage_0based is not None and not cobre_hydro_per_stage_bounds.is_empty():
        cobre_hydro_per_stage_bounds = cobre_hydro_per_stage_bounds.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )
    if nw_max_stage_0based is not None and not nw_hydro_slacks.is_empty():
        nw_hydro_slacks = nw_hydro_slacks.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )

    # --- Bus-level energy balance ---
    _LOG.info("Computing bus-level aggregates...")
    bus_aggregates = read_cobre_bus_aggregates(cobre_output_dir)
    nw_market = pl.DataFrame()
    nw_sin = pl.DataFrame()
    if saidas_dir is not None:
        nw_market = read_medias_market(saidas_dir)
        nw_sin = read_medias_sin(saidas_dir)

    # --- NEWAVE deterministic net load (load - NCS from sistema.dat) ---
    nw_net_load = read_newave_net_load(nw_files.directory)

    # --- Percentile statistics ---
    _LOG.info("Computing Cobre percentile statistics...")
    hydro_pct = read_cobre_hydro_percentiles(cobre_output_dir)
    thermal_pct = read_cobre_thermal_percentiles(cobre_output_dir)
    bus_pct = read_cobre_bus_percentiles(cobre_output_dir)
    line_pct = read_cobre_line_percentiles(cobre_output_dir)

    # --- Line bounds (per stage) and line metadata for the Network tab ---
    line_bounds_path = cobre_output_dir.parent / "constraints" / "line_bounds.parquet"
    line_bounds = (
        pd.read_parquet(line_bounds_path)
        if line_bounds_path.exists()
        else pd.DataFrame()
    )
    lines_json_path = cobre_output_dir.parent / "system" / "lines.json"
    line_meta: list[dict] = []
    if lines_json_path.exists():
        try:
            import json as _json

            line_meta = _json.loads(lines_json_path.read_text()).get("lines", [])
        except Exception:  # noqa: BLE001
            _LOG.warning("Failed to read lines.json")

    # NEWAVE typically reports a shorter horizon than Cobre.  Truncate all
    # Cobre-side per-stage DataFrames to NEWAVE's max stage so the report's
    # charts compare like-for-like across tabs (energy balance, hydro
    # operation, plant details, etc.).  Truncation is a no-op when NEWAVE
    # stage data was unavailable.
    if nw_max_stage_0based is not None:

        def _truncate(df: pl.DataFrame) -> pl.DataFrame:
            if df.is_empty() or "stage_id" not in df.columns:
                return df
            return df.filter(pl.col("stage_id") <= nw_max_stage_0based)

        cobre_hydro = _truncate(cobre_hydro)
        hydro_pct = _truncate(hydro_pct)
        thermal_pct = _truncate(thermal_pct)
        bus_pct = _truncate(bus_pct)
        bus_aggregates = _truncate(bus_aggregates)
        line_pct = _truncate(line_pct)
        cobre_line = _truncate(cobre_line)

    # --- Performance timings ---
    nw_tim_iters = read_newave_tim_iterations(nw_files.directory)
    nw_tim_stages = read_newave_tim_stages(nw_files.directory)
    cobre_training_seconds = read_cobre_training_duration(cobre_output_dir)
    cobre_iter_timing = read_cobre_iteration_timing(cobre_output_dir)

    # --- Generic constraints LHS comparison (RE / AGRINT / VminOP) ---
    # The Cobre case directory is one level above the simulation output dir
    # (cobre_output_dir ends in ``output/``).  Both the constraint
    # definitions and the per-stage bound parquet live in the case's
    # ``constraints/`` subdirectory.
    from cobre_bridge.comparators.constraints_compare import (
        _load_generic_constraint_bounds,
        _load_generic_constraints,
        evaluate_lhs_cobre,
        evaluate_lhs_newave,
    )

    cobre_case_dir = cobre_output_dir.parent
    gc_constraints = _load_generic_constraints(cobre_case_dir)
    gc_bounds_df = _load_generic_constraint_bounds(cobre_case_dir)
    if gc_constraints and saidas_dir is not None:
        _LOG.info(
            "Comparing %d generic constraints (RE/AGRINT/VminOP)...",
            len(gc_constraints),
        )
        gc_lhs_nw = evaluate_lhs_newave(
            gc_constraints,
            nw_hydro,
            nw_intercambio,
            alignment,
            id_map,
            nw_offset,
        )
        gc_lhs_cb = evaluate_lhs_cobre(gc_constraints, cobre_output_dir)
    else:
        gc_lhs_nw = pl.DataFrame()
        gc_lhs_cb = pl.DataFrame()

    # --- System spillage in MWmes ---
    cobre_spill_energy = read_cobre_spillage_energy(cobre_output_dir)
    if nw_max_stage_0based is not None and not cobre_spill_energy.is_empty():
        cobre_spill_energy = cobre_spill_energy.filter(
            pl.col("stage_id") <= nw_max_stage_0based
        )
    if not nw_sin.is_empty() and not cobre_spill_energy.is_empty():
        _LOG.info("Comparing system spillage energy...")
        results.extend(_compare_system_spillage(nw_sin, cobre_spill_energy))

    pctiles = PercentileData(
        hydro=hydro_pct,
        thermal=thermal_pct,
        bus=bus_pct,
        bus_aggregates=bus_aggregates,
        nw_convergence=nw_conv,
        cobre_convergence=cobre_conv,
        nw_market=nw_market,
        nw_net_load=nw_net_load,
        nw_sin=nw_sin,
        cobre_hydro_means=cobre_hydro,
        cobre_bus_meta=cobre_bus_meta,
        cobre_hydro_meta=cobre_hydro_meta,
        nw_bus_names=nw_bus_names,
        nw_hydro_names=nw_hydro_names,
        nw_costs=nw_costs,
        cobre_costs=cobre_costs,
        cobre_stage_costs=cobre_stage_costs,
        cobre_hydro_per_stage_bounds=cobre_hydro_per_stage_bounds,
        nw_hydro_slacks=nw_hydro_slacks,
        nw_offset=nw_offset,
        nw_max_stage=nw_max_stage_0based,
        cobre_spillage_energy=cobre_spill_energy,
        line=line_pct,
        line_bounds=line_bounds,
        line_meta=line_meta,
        nw_line_means=nw_intercambio,
        nw_tim_iterations=nw_tim_iters,
        nw_tim_stages=nw_tim_stages,
        cobre_training_seconds=cobre_training_seconds,
        cobre_iteration_timing=cobre_iter_timing,
        gc_constraints=gc_constraints,
        gc_bounds=gc_bounds_df,
        gc_lhs_newave=gc_lhs_nw,
        gc_lhs_cobre=gc_lhs_cb,
    )

    _LOG.info("Results comparison: %d total comparisons", len(results))
    return results, pctiles


def build_results_summary(
    results: list[ResultComparison], tolerance: float = 1e-2
) -> ResultsSummary:
    """Compute aggregate statistics from comparison results.

    ``tolerance`` is the relative tolerance used for the per-variable
    within-tolerance match rate (mirrors ``compare bounds``).
    """
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

        # Bounded symmetric error (robust to near-zero references) and the
        # within-tolerance match rate.
        smapes = [_smape(r.newave_value, r.cobre_value) for r in group]
        stats.mean_smape = sum(smapes) / len(smapes) if smapes else 0.0
        stats.max_smape = max(smapes) if smapes else 0.0
        n_within = sum(
            1
            for r in group
            if _within_tolerance(r.newave_value, r.cobre_value, tolerance)
        )
        stats.within_tol_rate = n_within / len(group) if group else 0.0

        # Pearson correlation (None when undefined, e.g. a constant series).
        nw_vals = [r.newave_value for r in group]
        cb_vals = [r.cobre_value for r in group]
        if len(nw_vals) > 1:
            stats.correlation = _pearson(nw_vals, cb_vals)

        summary.by_variable[var] = stats

    return summary


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient; ``None`` when undefined.

    Returns ``None`` for fewer than two points or a constant (zero-variance)
    series, so callers can distinguish "uncomputable" from a genuine 0.0.
    """
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom < 1e-15:
        return None
    return cov / denom
