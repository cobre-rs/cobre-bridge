"""Generic-constraint expression parsing and LHS evaluation (presentation-free).

Cobre encodes each generic constraint's left-hand side as a small textual
expression over LP variables, e.g.::

    "5.68 * hydro_storage(78)"
    "@rho_acum_h78 * hydro_storage(78) - line_exchange(4)"

This module owns the *domain* logic for those expressions — the term parser,
the scalar-parameter resolver, and the routine that evaluates the LHS from a
Cobre simulation's parquet output. It has **no** presentation dependency, so it
is the single shared home for both the dashboard (which renders the result) and
the comparator (which checks it against NEWAVE). It deliberately lives at the
package top level so neither consumer has to import across the comparator↔UI
boundary.
"""

from __future__ import annotations

import re

import pandas as pd
import polars as pl

# Matches terms like: [+/-] [coeff *] [@name *] variable_type(id)
# Examples:
#   "5.68 * hydro_storage(78)"             — literal coefficient
#   "hydro_generation(145)"                — implicit 1.0
#   "- line_exchange(4)"                   — implicit -1.0
#   "@rho_acum_h78 * hydro_storage(78)"    — @name coefficient (cobre HEAD sigil)
#   "0.5 * @rho_eq_h47 * hydro_generation(47)"  — literal × @name scale
_TERM_RE = re.compile(
    r"([+-]?\s*\d*\.?\d*)\s*\*?\s*(?:@([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*)?"
    r"(hydro_storage|hydro_generation|line_exchange|line_direct|line_reverse)"
    r"\((\d+)\)"
)

# A scalar-parameter reference resolved against simulation columns:
#   rho_eq_h{id}    → equivalent_productivity_mw_per_m3s for that hydro
#   rho_acum_h{id}  → accumulated_productivity_mw_per_m3s for that hydro
_RHO_EQ_RE = re.compile(r"^rho_eq_h(\d+)$")
_RHO_ACUM_RE = re.compile(r"^rho_acum_h(\d+)$")


def resolve_param_to_column(name: str) -> tuple[str, int] | None:
    """Map a parameter ``name`` to the (column, hydro_id) used to look it up.

    Returns ``(simulation_column, hydro_id)`` for the two computed parameters
    cobre-bridge declares (``rho_eq_h{id}``, ``rho_acum_h{id}``), or ``None``
    when the name is unrecognised. Callers treat unrecognised parameters as if
    they had value 0 to avoid a hard error.
    """
    m = _RHO_ACUM_RE.match(name)
    if m is not None:
        return "accumulated_productivity_mw_per_m3s", int(m.group(1))
    m = _RHO_EQ_RE.match(name)
    if m is not None:
        return "equivalent_productivity_mw_per_m3s", int(m.group(1))
    return None


def parse_expression(expr: str) -> list[tuple[float, str | None, str, int]]:
    """Parse a constraint LHS expression into term tuples.

    Returns a list of ``(literal_coeff, param_name_or_None, variable_type,
    entity_id)`` tuples. The effective coefficient on the variable is
    ``literal_coeff × value_of(param_name)`` when a parameter is present,
    or ``literal_coeff`` otherwise.

    Handles:
    - Leading ``-`` with no explicit coefficient → -1.0
    - No coefficient → 1.0
    - ``0.5 * hydro_generation(47)`` → 0.5
    - ``@rho_acum_h78 * hydro_storage(78)``
      → (1.0, "rho_acum_h78", "hydro_storage", 78)
    - ``- @rho_eq_h12 * hydro_generation(12)``
      → (-1.0, "rho_eq_h12", "hydro_generation", 12)
    """
    terms: list[tuple[float, str | None, str, int]] = []
    for m in _TERM_RE.finditer(expr):
        raw_coeff = m.group(1).replace(" ", "")
        param_name = m.group(2)
        var_type = m.group(3)
        entity_id = int(m.group(4))
        if raw_coeff in ("", "+"):
            coeff = 1.0
        elif raw_coeff == "-":
            coeff = -1.0
        else:
            coeff = float(raw_coeff)
        terms.append((coeff, param_name, var_type, entity_id))
    return terms


def evaluate_constraint_expressions(
    constraints: list[dict],
    hydros_lf: pl.LazyFrame,
    exchanges_lf: pl.LazyFrame,
) -> pd.DataFrame:
    """Evaluate LHS of all generic constraints from simulation output.

    Variable lookups:
    - ``hydro_storage(id)``    → ``storage_final_hm3`` where hydro_id=id, block_id=0
    - ``hydro_generation(id)`` → ``generation_mw``     where hydro_id=id  (per block)
    - ``line_exchange(id)``    → ``net_flow_mw``        where line_id=id   (per block)
    - ``line_direct(id)``      → ``direct_flow_mw``     where line_id=id   (per block)
    - ``line_reverse(id)``     → ``reverse_flow_mw``    where line_id=id   (per block)

    Storage-only constraints produce one row per (scenario, stage) with
    block_id=0. Mixed / generation / exchange constraints produce one row per
    (scenario, stage, block).

    Accepts LazyFrames and only collects the specific entity IDs referenced
    in constraint expressions, keeping memory usage minimal.

    Returns DataFrame with columns:
        constraint_id, scenario_id, stage_id, block_id, lhs_value
    """
    # Parse ALL constraints to find which entity IDs / parameters are referenced.
    hydro_ids_needed: set[int] = set()
    line_ids_needed: set[int] = set()
    needs_rho_eq = False
    needs_rho_acum = False
    for c in constraints:
        for _coeff, param_name, vtype, eid in parse_expression(c["expression"]):
            if vtype.startswith("hydro"):
                hydro_ids_needed.add(eid)
            elif vtype in ("line_exchange", "line_direct", "line_reverse"):
                line_ids_needed.add(eid)
            if param_name is not None:
                resolved = resolve_param_to_column(param_name)
                if resolved is not None:
                    col, _ = resolved
                    if col == "equivalent_productivity_mw_per_m3s":
                        needs_rho_eq = True
                    elif col == "accumulated_productivity_mw_per_m3s":
                        needs_rho_acum = True

    # Collect only the referenced entities (tiny subset of full data). We pull
    # productivity columns when any @name reference depends on them so the
    # LHS evaluator can multiply the literal coefficient by the resolved
    # productivity at solve time (mirrors cobre's @name resolution).
    schema = hydros_lf.collect_schema()
    h0_cols = ["scenario_id", "stage_id", "hydro_id", "storage_final_hm3"]
    if needs_rho_eq and "equivalent_productivity_mw_per_m3s" in schema:
        h0_cols.append("equivalent_productivity_mw_per_m3s")
    if needs_rho_acum and "accumulated_productivity_mw_per_m3s" in schema:
        h0_cols.append("accumulated_productivity_mw_per_m3s")
    h0_pd = (
        hydros_lf.filter(
            (pl.col("block_id") == 0) & pl.col("hydro_id").is_in(list(hydro_ids_needed))
        )
        .select(h0_cols)
        .collect(engine="streaming")
        .to_pandas()
    )
    hg_cols = [
        "scenario_id",
        "stage_id",
        "block_id",
        "hydro_id",
        "generation_mw",
    ]
    if needs_rho_eq and "equivalent_productivity_mw_per_m3s" in schema:
        hg_cols.append("equivalent_productivity_mw_per_m3s")
    if needs_rho_acum and "accumulated_productivity_mw_per_m3s" in schema:
        hg_cols.append("accumulated_productivity_mw_per_m3s")
    hg_pd = (
        hydros_lf.filter(pl.col("hydro_id").is_in(list(hydro_ids_needed)))
        .select(hg_cols)
        .collect(engine="streaming")
        .to_pandas()
    )
    # Pull all three flow columns when line-touching terms are referenced;
    # the per-term branch below picks whichever column matches the
    # variable type.  ``direct_flow_mw`` and ``reverse_flow_mw`` are
    # the non-negative LP primitives behind cobre's ``line_direct`` /
    # ``line_reverse`` variables; ``net_flow_mw = direct - reverse`` is
    # the signed shorthand referenced by ``line_exchange``.
    ex_cols = [
        "scenario_id",
        "stage_id",
        "block_id",
        "line_id",
        "net_flow_mw",
        "direct_flow_mw",
        "reverse_flow_mw",
    ]
    ex_pd = (
        (
            exchanges_lf.filter(pl.col("line_id").is_in(list(line_ids_needed)))
            .select(ex_cols)
            .collect(engine="streaming")
            .to_pandas()
        )
        if line_ids_needed
        else pd.DataFrame(columns=ex_cols)
    )

    all_results: list[pd.DataFrame] = []

    for c in constraints:
        cid = c["id"]
        expr = c["expression"]
        terms = parse_expression(expr)
        if not terms:
            continue

        var_types = {t[2] for t in terms}
        storage_only = var_types == {"hydro_storage"}

        if storage_only:
            # One LHS value per (scenario, stage) — use block_id 0
            base = h0_pd[["scenario_id", "stage_id"]].drop_duplicates().copy()
            base["_lhs"] = 0.0
            for coeff, param_name, _vtype, eid in terms:
                sub = h0_pd[h0_pd["hydro_id"] == eid].copy()
                if sub.empty:
                    continue
                sub["_val"] = sub["storage_final_hm3"]
                if param_name is not None:
                    resolved = resolve_param_to_column(param_name)
                    if resolved is not None and resolved[0] in sub.columns:
                        sub["_val"] = sub["_val"] * sub[resolved[0]].fillna(0.0)
                    else:
                        # Unknown / unresolved parameter contributes zero so we
                        # don't pollute LHS with stale storage values.
                        sub["_val"] = 0.0
                sub = sub[["scenario_id", "stage_id", "_val"]]
                merged = base.merge(sub, on=["scenario_id", "stage_id"], how="left")
                merged["_val"] = merged["_val"].fillna(0.0)
                base["_lhs"] = base["_lhs"].values + coeff * merged["_val"].values
            base["constraint_id"] = cid
            base["block_id"] = 0
            base = base.rename(columns={"_lhs": "lhs_value"})
            all_results.append(
                base[
                    [
                        "constraint_id",
                        "scenario_id",
                        "stage_id",
                        "block_id",
                        "lhs_value",
                    ]
                ]
            )
        else:
            # Per (scenario, stage, block) — need consistent block grid.
            base = (
                hg_pd[["scenario_id", "stage_id", "block_id"]].drop_duplicates().copy()
            )
            base["_lhs"] = 0.0

            for coeff, param_name, vtype, eid in terms:
                if vtype == "hydro_storage":
                    sub = h0_pd[h0_pd["hydro_id"] == eid].copy()
                    if sub.empty:
                        continue
                    sub["_val"] = sub["storage_final_hm3"]
                    join_cols = ["scenario_id", "stage_id"]
                elif vtype == "hydro_generation":
                    sub = hg_pd[hg_pd["hydro_id"] == eid].copy()
                    if sub.empty:
                        continue
                    sub["_val"] = sub["generation_mw"]
                    join_cols = ["scenario_id", "stage_id", "block_id"]
                else:  # line_exchange / line_direct / line_reverse
                    sub = ex_pd[ex_pd["line_id"] == eid].copy()
                    if sub.empty:
                        continue
                    if vtype == "line_direct":
                        sub["_val"] = sub["direct_flow_mw"]
                    elif vtype == "line_reverse":
                        sub["_val"] = sub["reverse_flow_mw"]
                    else:
                        sub["_val"] = sub["net_flow_mw"]
                    join_cols = ["scenario_id", "stage_id", "block_id"]

                if param_name is not None:
                    resolved = resolve_param_to_column(param_name)
                    if resolved is not None and resolved[0] in sub.columns:
                        sub["_val"] = sub["_val"] * sub[resolved[0]].fillna(0.0)
                    else:
                        sub["_val"] = 0.0

                sub = sub[join_cols + ["_val"]]
                merged = base.merge(sub, on=join_cols, how="left")
                merged["_val"] = merged["_val"].fillna(0.0)
                base["_lhs"] = base["_lhs"].values + coeff * merged["_val"].values

            base["constraint_id"] = cid
            base = base.rename(columns={"_lhs": "lhs_value"})
            all_results.append(
                base[
                    [
                        "constraint_id",
                        "scenario_id",
                        "stage_id",
                        "block_id",
                        "lhs_value",
                    ]
                ]
            )

    if not all_results:
        return pd.DataFrame(
            columns=[
                "constraint_id",
                "scenario_id",
                "stage_id",
                "block_id",
                "lhs_value",
            ]
        )
    return pd.concat(all_results, ignore_index=True)
