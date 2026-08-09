"""Generic constraint LHS evaluation from the source model and Cobre simulation outputs.

Reads the converted Cobre case's ``constraints/generic_constraints.json``
and ``constraints/generic_constraint_bounds.parquet`` and evaluates each
constraint's LHS using both:

- the source model outputs — ``MEDIAS-USIH.CSV`` (``GHIDUH``, ``VARMUH`` per plant
  per stage) and ``int*.out`` (per-line per-stage interchange).
- Cobre simulation outputs — per-(scenario, stage, block) hydro and
  exchange data, collapsed to one value per (constraint, stage) by
  averaging across scenarios and blocks.

The output is consumed by the compare-results report's Constraints tab, mirroring the
dashboard's per-constraint LHS-vs-Bound visualisation with a the source model overlay so
the user can see whether a binding constraint matches across the two solvers.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import NamedTuple

import pandas as pd
import polars as pl

from cobre_bridge.comparators.alignment import EntityAlignment
from cobre_bridge.comparators.cobre_readers import _scan_simulation_entity
from cobre_bridge.constraint_expr import (
    evaluate_constraint_expressions,
    parse_expression,
    resolve_param_to_column,
)
from cobre_bridge.generic_constraint_format import shape_from_bounds
from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)


def _load_generic_constraints(cobre_input_dir: Path) -> list[dict]:
    """Load constraint definitions from ``constraints/generic_constraints.json``.

    The F3 objects are sense-free (no ``sense`` key); direction is derived
    from the companion bounds table's endpoints when a label is needed (see
    :func:`per_stage_bounds` / :func:`cobre_bridge.generic_constraint_format.
    shape_from_bounds`). Returns an empty list when the file is missing or
    malformed.
    """
    path = cobre_input_dir / "constraints" / "generic_constraints.json"
    if not path.exists():
        return []
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("generic_constraints.json could not be parsed: %s", exc)
        return []
    return list(data.get("constraints", []))


def _load_generic_constraint_bounds(cobre_input_dir: Path) -> pl.DataFrame:
    """Load bound table from ``constraints/generic_constraint_bounds.parquet``.

    F3 shape: nullable ``bound_lower``/``bound_upper`` endpoints, no single
    ``bound`` column — even in the missing-file fallback schema below.
    """
    path = cobre_input_dir / "constraints" / "generic_constraint_bounds.parquet"
    if not path.exists():
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "block_id": pl.Int32,
                "bound_lower": pl.Float64,
                "bound_upper": pl.Float64,
            }
        )
    return pl.read_parquet(path)


def evaluate_lhs_newave(
    constraints: list[dict],
    nw_hydro_df: pl.DataFrame,
    nw_line_means: pl.DataFrame,
    alignment: EntityAlignment,
    id_map: NewaveIdMap,
    nw_offset: int,
) -> pl.DataFrame:
    """Evaluate each constraint's LHS from the source model simulation outputs.

    The constraint expressions reference Cobre entity IDs. We translate via ``id_map``
    (hydro Cobre id → the source model code) and ``alignment.lines`` (Cobre line id →
    the source model submarket pair) before looking values up in the source model data.

    The source model outputs are stage-level (block-collapsed) — MEDIAS values are
    monthly means and ``int*.out`` ``TOTAL`` rows already average over blocks weighted
    by their duration. The result therefore has one value per (constraint, stage_0based)
    pair.

    Parameters
    ----------
    constraints:
        Constraint dicts from ``generic_constraints.json``.
    nw_hydro_df:
        Output of :func:`cobre_bridge.comparators.newave_readers.read_medias_hydro`.
        Columns ``newave_code`` (Int64), ``stage`` (Int64, 1-based as in
        MEDIAS — Sep-start studies use 9 for the first stage),
        ``variable`` (Utf8), ``value`` (Float64).
    nw_line_means:
        Output of
        :func:`cobre_bridge.comparators.newave_readers.read_nwlistop_intercambio`.
        One row per directional pair × stage.  We map each Cobre line to
        the matching directional row via the alignment's ``newave_de`` /
        ``newave_para`` fields (respecting the ``reversed`` flag for
        sign).
    alignment:
        Pre-built entity alignment.  Provides Cobre-line → (newave_de,
        newave_para) mappings and the ``reversed`` flag.
    id_map:
        Used to translate Cobre hydro IDs back to the source model plant codes.
    nw_offset:
        MEDIAS stage offset (e.g. 9 for a September-start study).  Used to convert the
        source model 1-based MEDIAS stages to Cobre 0-based.

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32, 0-based),
        ``lhs_value`` (Float64).  One row per (constraint, stage) pair that has at least
        one resolvable term.  Stages with missing variables produce no row (rather than
        a zero); this avoids misrepresenting "no source-model data" as "LHS = 0".
    """
    if not constraints:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # --- Build hydro generation lookup: (cobre_hydro_id, stage_0based) -> MW
    hydro_gen: dict[tuple[int, int], float] = {}
    if not nw_hydro_df.is_empty():
        ghiduh = nw_hydro_df.filter(pl.col("variable") == "GHIDUH")
        for row in ghiduh.iter_rows(named=True):
            nw_code = int(row["newave_code"])
            try:
                cobre_id = id_map.hydro_id(nw_code)
            except KeyError:
                # Skip plants that are not in the LP (e.g. FICT, NE/NC).
                continue
            stage_0based = int(row["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            hydro_gen[(cobre_id, stage_0based)] = float(row["value"])

    # --- Build hydro storage lookup: (cobre_hydro_id, stage_0based) -> hm3
    hydro_storage: dict[tuple[int, int], float] = {}
    if not nw_hydro_df.is_empty():
        varmuh = nw_hydro_df.filter(pl.col("variable") == "VARMUH")
        for row in varmuh.iter_rows(named=True):
            nw_code = int(row["newave_code"])
            try:
                cobre_id = id_map.hydro_id(nw_code)
            except KeyError:
                continue
            stage_0based = int(row["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            hydro_storage[(cobre_id, stage_0based)] = float(row["value"])

    # --- Build line exchange lookup: (cobre_line_id, stage_0based) -> MW Aligned via
    # EntityAlignment.lines: each Cobre line records the source model directional pair
    # (newave_de → newave_para), which matches the Cobre (source, target) orientation by
    # construction.  NWLISTOP rows for the opposite (para, de) ordering carry the
    # opposite sign, so they are negated.
    line_flow: dict[tuple[int, int], float] = {}
    if not nw_line_means.is_empty() and alignment.lines:
        # Build (de, para, stage_0based) -> value for both directions.
        # NWLISTOP files are per (de, para) ordered pair; if our alignment
        # uses the opposite ordering we negate.
        df = nw_line_means.with_columns(
            (pl.col("stage") - nw_offset).cast(pl.Int64).alias("stage_0based")
        ).filter(pl.col("stage_0based") >= 0)
        nw_by_pair: dict[tuple[int, int, int], float] = {}
        for row in df.iter_rows(named=True):
            key = (
                int(row["from_submarket_code"]),
                int(row["to_submarket_code"]),
                int(row["stage_0based"]),
            )
            nw_by_pair[key] = float(row["value"])
        for line in alignment.lines:
            de = line.newave_de
            para = line.newave_para
            if de is None or para is None:
                continue
            for (de_k, para_k, s), val in nw_by_pair.items():
                if de_k == de and para_k == para:
                    line_flow[(line.cobre_line_id, s)] = val
                elif de_k == para and para_k == de:
                    # Reversed-ordering NWLISTOP row supplies the opposite sign
                    # of what our alignment expects.  Only fill if the
                    # canonical-direction row hasn't already populated this slot.
                    line_flow.setdefault((line.cobre_line_id, s), -val)

    # --- Per-constraint LHS evaluation ---
    rows: list[dict] = []
    for c in constraints:
        cid = int(c["id"])
        terms = parse_expression(c["expression"])
        if not terms:
            continue
        # Determine the set of stages we can evaluate for this constraint: a stage is
        # evaluable iff every referenced variable has a source-model value at that
        # stage.
        per_stage: dict[int, float] = {}
        all_stages: set[int] = set()
        for _, _, vtype, eid in terms:
            if vtype == "hydro_storage":
                for h, s in hydro_storage:
                    if h == eid:
                        all_stages.add(s)
            elif vtype == "hydro_generation":
                for h, s in hydro_gen:
                    if h == eid:
                        all_stages.add(s)
            elif vtype in ("line_exchange", "line_direct", "line_reverse"):
                for line_id, s in line_flow:
                    if line_id == eid:
                        all_stages.add(s)

        for stage in all_stages:
            lhs = 0.0
            stage_complete = True
            for coeff, param_name, vtype, eid in terms:
                if vtype == "hydro_storage":
                    val = hydro_storage.get((eid, stage))
                elif vtype == "hydro_generation":
                    val = hydro_gen.get((eid, stage))
                elif vtype == "line_direct":
                    # ``line_direct`` is the non-negative flow in the canonical
                    # (src<tgt) direction. The source model int*.out gives a single
                    # signed value per (stage, pair); we take its positive part as the
                    # stage-mean approximation. Per-block decomposition would require
                    # parsing the per-patamar rows separately.
                    signed = line_flow.get((eid, stage))
                    val = max(0.0, signed) if signed is not None else None
                elif vtype == "line_reverse":
                    signed = line_flow.get((eid, stage))
                    val = max(0.0, -signed) if signed is not None else None
                else:  # line_exchange (signed net flow)
                    val = line_flow.get((eid, stage))
                if val is None:
                    stage_complete = False
                    break
                # @rho_eq / @rho_acum parameters scale the coefficient at solve time in
                # Cobre.  We don't have the source-model-side productivity per (hydro,
                # stage) handy here, so skip constraints with such parameters by
                # treating them as missing on this side — the chart will simply lack a
                # The source model trace.  This affects VminOP only; RE/AGRINT never
                # reference @-parameters.
                if param_name is not None:
                    resolved = resolve_param_to_column(param_name)
                    if resolved is not None:
                        stage_complete = False
                        break
                lhs += coeff * val
            if stage_complete:
                per_stage[stage] = lhs

        for stage, lhs in sorted(per_stage.items()):
            rows.append(
                {
                    "constraint_id": cid,
                    "stage_id": stage,
                    "lhs_value": lhs,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


def evaluate_lhs_cobre(
    constraints: list[dict],
    cobre_output_dir: Path,
) -> pl.DataFrame:
    """Evaluate each constraint's LHS from Cobre simulation outputs.

    Uses the shared
    :func:`cobre_bridge.constraint_expr.evaluate_constraint_expressions`
    (which returns one row per (constraint, scenario, stage, block)) and
    collapses to mean across scenarios and blocks per (constraint, stage).

    Returns
    -------
    polars.DataFrame
        Columns: ``constraint_id`` (Int32), ``stage_id`` (Int32),
        ``lhs_value`` (Float64).  Empty when no simulation data is
        available or no constraints reference the simulation entities.
    """
    if not constraints:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # Scan with the comparator's own simulation reader (which already takes the
    # ``output/`` directory directly), instead of reaching into the dashboard's
    # case-dir-based scanner via a synthetic ``cobre_output_dir.parent``. A
    # present-but-corrupt parquet raises CobreReadError.
    # NB: ``lf or pl.LazyFrame()`` would evaluate ``bool(lf)``, which polars
    # rejects ("truth value of a LazyFrame is ambiguous") — use explicit None
    # checks.
    empty = pl.DataFrame(
        schema={
            "constraint_id": pl.Int32,
            "stage_id": pl.Int32,
            "lhs_value": pl.Float64,
        }
    )

    hydros_lf = _scan_simulation_entity(cobre_output_dir, "hydros")
    if hydros_lf is None:
        # No hydro simulation → no operation data to evaluate the LHS against.
        return empty
    exchanges_lf = _scan_simulation_entity(cobre_output_dir, "exchanges")
    if exchanges_lf is None:
        exchanges_lf = pl.LazyFrame()

    lhs_pd: pd.DataFrame = evaluate_constraint_expressions(
        constraints, hydros_lf, exchanges_lf
    )
    if lhs_pd.empty:
        return pl.DataFrame(
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "lhs_value": pl.Float64,
            }
        )

    # Collapse to mean across scenarios and blocks per (constraint, stage).
    grouped = lhs_pd.groupby(["constraint_id", "stage_id"], as_index=False)
    means = pd.DataFrame(grouped["lhs_value"].mean())
    return pl.from_pandas(means).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


class ResolvedBound(NamedTuple):
    """A per-(constraint, stage) F3 bound reduced to one comparison-ready limit.

    ``value`` is the endpoint appropriate to the row's evaluated direction —
    the number the numeric comparison and the chart plot, identical to what
    the pre-F3 single ``bound`` column held. ``shape`` is the direction label
    (``">="``/``"<="``/``"=="``/``"range"``) from
    :func:`~cobre_bridge.generic_constraint_format.shape_from_bounds`, for
    display only.
    """

    value: float
    shape: str


def _resolve_bound(lower: float | None, upper: float | None) -> ResolvedBound:
    """Resolve one F3 bound row to a single comparison-ready limit + shape.

    ``>=`` (lower-only) reads ``lower``; ``<=`` (upper-only) reads ``upper``;
    ``==`` (both equal) reads either (they agree) — this matches the pre-F3
    single ``bound`` column value exactly for every constraint family this
    comparator evaluates today (RE/AGRINT/VminOP are all single-sided). A
    genuine two-sided ``range`` has no producer feeding this comparator; the
    upper (ceiling) endpoint is used as the documented fallback so a future
    band constraint degrades to one visualized limit rather than raising.
    """
    shape = shape_from_bounds(lower, upper)
    if shape in ("<=", "range"):
        if upper is None:  # pragma: no cover - guaranteed by shape_from_bounds
            raise AssertionError(f"shape {shape!r} implies a non-null upper bound")
        return ResolvedBound(upper, shape)
    if lower is None:  # pragma: no cover - guaranteed by shape_from_bounds
        raise AssertionError(f"shape {shape!r} implies a non-null lower bound")
    return ResolvedBound(lower, shape)


def per_stage_bounds(
    gc_bounds: pl.DataFrame,
    max_stage: int | None = None,
) -> dict[int, dict[int, ResolvedBound]]:
    """Reduce per-(stage, block) bounds to one value per (constraint, stage).

    When per-block bounds disagree (rare; AGRINT/RE bounds are usually
    block-invariant), the block_id=0 value is preferred.  Returns
    ``{constraint_id: {stage_id: ResolvedBound}}``; stages beyond
    ``max_stage`` are dropped when ``max_stage`` is supplied.
    """
    out: dict[int, dict[int, ResolvedBound]] = {}
    if gc_bounds.is_empty():
        return out
    df = gc_bounds
    if max_stage is not None:
        df = df.filter(pl.col("stage_id") <= max_stage)
    for cid, group in df.group_by("constraint_id"):
        cid_int = int(cid[0])
        per_stage: dict[int, ResolvedBound] = {}
        if "block_id" in group.columns:
            preferred = group.filter(pl.col("block_id") == 0)
            if preferred.is_empty():
                preferred = group
        else:
            preferred = group
        for row in preferred.iter_rows(named=True):
            stage = int(row["stage_id"])
            if stage not in per_stage:
                per_stage[stage] = _resolve_bound(
                    row["bound_lower"], row["bound_upper"]
                )
        out[cid_int] = per_stage
    return out


# --- VminOP useful-energy rewrite -------------------------------------------
#
# VminOP (security-curve) constraints bound *stored energy*: their expression is ``Σ
# @rho_acum_h{id} * hydro_storage(id) >= bound``.  The generic LHS evaluator above
# resolves ``@rho_acum`` to cobre's *default* point productivity
# ``accumulated_productivity_mw_per_m3s`` (MW/(m³/s)), which is the energy-per-volume
# coefficient over-scaled by the hm³↔(m³/s)·month factor (≈ 2.628) relative to the
# *override* the LP actually uses (energy-scaled, MWmonth/hm³).  That makes the raw
# VminOP LHS incomparable to its own bound and to the source model.  This rewrite
# re-expresses VminOP rows as **useful stored energy
# in MWmonth** so they line up with the source model's per-REE ``EARMF`` (MEDIAS-REE):
#
# cobre LHS  = Σ override_ρ_acum(stage) · (storage_final − Vmin)   [useful] The source
# model LHS = EARMF for the constraint's REE                       [useful] bound      =
# stored-bound − dead-energy (= pct · useful EARMX)    [useful]
#
# where dead-energy = Σ override_ρ_acum(stage) · Vmin removes the absolute-vs-
# relative-to-minimum offset (cobre stores absolute volume; the source model EARM is
# relative to the minimum operative volume).  RE / AGRINT rows are untouched.
_GC_SCHEMA = {
    "constraint_id": pl.Int32,
    "stage_id": pl.Int32,
    "lhs_value": pl.Float64,
}


def _is_vminop(constraint: dict) -> bool:
    """True when the constraint scales ``hydro_storage`` by an ``@rho_acum``.

    RE / AGRINT constraints never reference ``@rho_acum_h{id}``; only VminOP
    (security-curve) constraints do, so this cleanly partitions the set.
    """
    for _, param_name, vtype, _ in parse_expression(constraint.get("expression", "")):
        if (
            vtype == "hydro_storage"
            and param_name is not None
            and param_name.startswith("rho_acum_h")
        ):
            return True
    return False


def _load_rho_acum_overrides(cobre_case_dir: Path) -> dict[int, dict[int, float]]:
    """Load per-stage ρ_acum overrides from ``constraints/generic_parameters.json``.

    Returns ``{hydro_id: {stage_id: ρ_acum}}`` where ρ_acum is the energy-scaled
    coefficient (MWmonth/hm³) the VminOP LP uses, so ρ·storage[hm³] is MWmonth.
    """
    path = cobre_case_dir / "constraints" / "generic_parameters.json"
    out: dict[int, dict[int, float]] = {}
    if not path.exists():
        return out
    try:
        with path.open() as f:
            params = json.load(f).get("scalar_parameters", [])
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("generic_parameters.json could not be parsed: %s", exc)
        return out
    for entry in params:
        m = re.fullmatch(r"rho_acum_h(\d+)", str(entry.get("name", "")))
        if m is None or entry.get("kind") != "per_stage":
            continue
        out[int(m.group(1))] = {int(s): float(v) for s, v in entry.get("values", [])}
    return out


def _load_hydro_min_storage(cobre_case_dir: Path) -> dict[int, float]:
    """Load minimum operative storage (hm³) per hydro id from ``hydros.json``."""
    path = cobre_case_dir / "system" / "hydros.json"
    out: dict[int, float] = {}
    if not path.exists():
        return out
    try:
        with path.open() as f:
            hydros = json.load(f).get("hydros", [])
    except (OSError, json.JSONDecodeError) as exc:
        _LOG.warning("hydros.json could not be parsed: %s", exc)
        return out
    for h in hydros:
        vmin = (h.get("reservoir") or {}).get("min_storage_hm3")
        if vmin is not None:
            out[int(h["id"])] = float(vmin)
    return out


def apply_vminop_useful_energy(
    constraints: list[dict],
    gc_bounds: pl.DataFrame,
    gc_lhs_nw: pl.DataFrame,
    gc_lhs_cb: pl.DataFrame,
    cobre_case_dir: Path,
    cobre_output_dir: Path,
    nw_hydro: pl.DataFrame,
    id_map: NewaveIdMap,
    nw_offset: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Re-express VminOP rows as *useful* stored energy (MWmonth).

    Rewrites the VminOP entries of ``(gc_bounds, gc_lhs_nw, gc_lhs_cb)`` so the
    Constraints tab compares like-for-like useful stored energy:

    - cobre LHS  = Σ override ρ_acum(stage) · (storage_final − Vmin)
    - the source model LHS = Σ override ρ_acum(stage) · VARMUH(plant, stage)
    - bound      = original stored-bound − dead-volume energy (= pct · useful)

    The source model LHS uses the per-plant ``VARMUH`` (useful stored volume above the
    minimum, MEDIAS-USIH) weighted by the *same* per-stage ρ_acum override as the cobre
    LHS and the bound — i.e. the **linear** stored energy that source-model's
    security-curve constraint actually binds on.  This is deliberately **not** the
    per-REE ``EARMF`` (MEDIAS-REE), which the source model reports as the *nonlinear*
    physical stored energy (∫ρ dv, head-dependent, up to ~4–5 % lower at mid storage).
    Comparing the nonlinear ``EARMF`` against the linear bound made the source model
    appear to sit below the curve with no penalty; the linear reconstruction here
    reproduces the source model's published ``VIOL_CAR`` to ~0.1 % and lands exactly on
    the curve where the source model holds it.

    RE / AGRINT rows pass through unchanged.  If any required input is missing
    (no scalar parameters, no min-storage, no simulation), the inputs are
    returned untouched so the caller degrades gracefully to the prior behaviour.

    Returns the updated ``(gc_bounds, gc_lhs_nw, gc_lhs_cb)``.
    """
    vminop = [c for c in constraints if _is_vminop(c)]
    if not vminop:
        return gc_bounds, gc_lhs_nw, gc_lhs_cb

    rho = _load_rho_acum_overrides(cobre_case_dir)
    vmin = _load_hydro_min_storage(cobre_case_dir)
    hydros_lf = _scan_simulation_entity(cobre_output_dir, "hydros")
    if not rho or not vmin or hydros_lf is None:
        _LOG.warning(
            "VminOP useful-energy rewrite skipped (missing ρ_acum/Vmin/sim data)."
        )
        return gc_bounds, gc_lhs_nw, gc_lhs_cb

    storage = (
        hydros_lf.select("hydro_id", "stage_id", "storage_final_hm3")
        .group_by(["hydro_id", "stage_id"])
        .agg(pl.col("storage_final_hm3").mean().alias("sf"))
        .collect()
    )
    sf_lookup = {
        (int(r["hydro_id"]), int(r["stage_id"])): float(r["sf"])
        for r in storage.iter_rows(named=True)
    }

    # The source model per-plant useful stored volume (VARMUH, hm³ above Vmin), keyed by
    # cobre hydro id and 0-based stage — the linear-energy counterpart of the cobre
    # ``storage_final − Vmin`` term.
    varmuh: dict[tuple[int, int], float] = {}
    if not nw_hydro.is_empty():
        for r in nw_hydro.filter(pl.col("variable") == "VARMUH").iter_rows(named=True):
            try:
                cobre_id = id_map.hydro_id(int(r["newave_code"]))
            except KeyError:
                continue
            stage_0based = int(r["stage"]) - nw_offset
            if stage_0based < 0:
                continue
            varmuh[(cobre_id, stage_0based)] = float(r["value"])

    bounds_by_cs = per_stage_bounds(gc_bounds)
    cb_rows: list[dict] = []
    nw_rows: list[dict] = []
    dead_by_cs: dict[tuple[int, int], float] = {}
    for c in vminop:
        cid = int(c["id"])
        hydro_ids = [
            eid
            for _, _, vtype, eid in parse_expression(c["expression"])
            if vtype == "hydro_storage"
        ]
        for stage in sorted(bounds_by_cs.get(cid, {})):
            cb_useful = 0.0
            dead = 0.0
            nw_useful = 0.0
            complete = True
            nw_complete = True
            for hid in hydro_ids:
                rho_s = rho.get(hid, {}).get(stage)
                vm = vmin.get(hid)
                sf = sf_lookup.get((hid, stage))
                if rho_s is None or vm is None or sf is None:
                    complete = False
                    break
                cb_useful += rho_s * (sf - vm)
                dead += rho_s * vm
                vu = varmuh.get((hid, stage))
                if vu is None:
                    nw_complete = False
                else:
                    nw_useful += rho_s * vu
            if not complete:
                continue
            cb_rows.append(
                {"constraint_id": cid, "stage_id": stage, "lhs_value": cb_useful}
            )
            dead_by_cs[(cid, stage)] = dead
            # The source model LHS = Σ ρ_acum(stage) · VARMUH — the linear stored energy
            # the security curve binds on (NOT the nonlinear MEDIAS-REE EARMF).
            if nw_complete:
                nw_rows.append(
                    {"constraint_id": cid, "stage_id": stage, "lhs_value": nw_useful}
                )

    vminop_ids = [int(c["id"]) for c in vminop]

    # Replace cobre VminOP LHS rows with the useful-energy values.
    cb_keep = (
        gc_lhs_cb.filter(~pl.col("constraint_id").is_in(vminop_ids))
        if not gc_lhs_cb.is_empty()
        else gc_lhs_cb
    )
    cb_new = pl.DataFrame(cb_rows, schema=_GC_SCHEMA)
    gc_lhs_cb_out = pl.concat([cb_keep, cb_new]) if cb_rows else gc_lhs_cb

    # Add the source model VminOP LHS rows (none existed before — the generic evaluator
    # skips @rho_acum constraints on the source model side).
    nw_new = pl.DataFrame(nw_rows, schema=_GC_SCHEMA)
    gc_lhs_nw_out = pl.concat([gc_lhs_nw, nw_new]) if nw_rows else gc_lhs_nw

    # Subtract dead-volume energy from VminOP bounds → useful (pct·EARMX) bound.
    # VminOP rows are always ``>=`` (see `convert_vminop_constraints`), so the
    # writer's F3 endpoint pair always carries the stored value in
    # ``bound_lower`` with ``bound_upper`` null — subtracting only from
    # ``bound_lower`` is therefore exactly the pre-F3 ``bound - dead``
    # semantics. Non-VminOP rows get a filled `dead` of 0.0, so their
    # (possibly-null) ``bound_lower`` passes through unchanged.
    if dead_by_cs and not gc_bounds.is_empty():
        dead_df = pl.DataFrame(
            [
                {"constraint_id": cid, "stage_id": s, "dead": d}
                for (cid, s), d in dead_by_cs.items()
            ],
            schema={
                "constraint_id": pl.Int32,
                "stage_id": pl.Int32,
                "dead": pl.Float64,
            },
        )
        gc_bounds_out = (
            gc_bounds.join(dead_df, on=["constraint_id", "stage_id"], how="left")
            .with_columns(
                (pl.col("bound_lower") - pl.col("dead").fill_null(0.0)).alias(
                    "bound_lower"
                )
            )
            .drop("dead")
        )
    else:
        gc_bounds_out = gc_bounds

    return gc_bounds_out, gc_lhs_nw_out, gc_lhs_cb_out
