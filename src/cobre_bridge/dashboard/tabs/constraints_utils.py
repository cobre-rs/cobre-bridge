"""Constraint utility functions shared by the constraints tab.

Provides expression parsing, LHS evaluation, and summary table generation.
"""

from __future__ import annotations

import re as _re

import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Expression parser (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------

# Matches terms like: [+/-] [coeff *] variable_type(id)
# Examples: "5.68 * hydro_storage(78)", "hydro_generation(145)", "- line_exchange(4)"
_TERM_RE = _re.compile(
    r"([+-]?\s*\d*\.?\d*)\s*\*?\s*(hydro_storage|hydro_generation|line_exchange)\((\d+)\)"
)


def _parse_expression(expr: str) -> list[tuple[float, str, int]]:
    """Parse a constraint LHS expression into (coefficient, variable_type, entity_id) terms.

    Handles:
    - Leading ``-`` with no explicit coefficient → -1.0
    - No coefficient → 1.0
    - ``0.5 * hydro_generation(47)`` → 0.5
    """
    terms: list[tuple[float, str, int]] = []
    for m in _TERM_RE.finditer(expr):
        raw_coeff = m.group(1).replace(" ", "")
        var_type = m.group(2)
        entity_id = int(m.group(3))
        if raw_coeff in ("", "+"):
            coeff = 1.0
        elif raw_coeff == "-":
            coeff = -1.0
        else:
            coeff = float(raw_coeff)
        terms.append((coeff, var_type, entity_id))
    return terms


# ---------------------------------------------------------------------------
# Chart functions (copied verbatim from dashboard.py, imports updated)
# ---------------------------------------------------------------------------


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

    Storage-only constraints produce one row per (scenario, stage) with block_id=0.
    Mixed / generation / exchange constraints produce one row per (scenario, stage, block).

    Accepts LazyFrames and only collects the specific entity IDs referenced
    in constraint expressions, keeping memory usage minimal.

    Returns DataFrame with columns:
        constraint_id, scenario_id, stage_id, block_id, lhs_value
    """
    # Parse ALL constraints to find which entity IDs are referenced
    hydro_ids_needed: set[int] = set()
    line_ids_needed: set[int] = set()
    for c in constraints:
        for _coeff, vtype, eid in _parse_expression(c["expression"]):
            if vtype.startswith("hydro"):
                hydro_ids_needed.add(eid)
            elif vtype == "line_exchange":
                line_ids_needed.add(eid)

    # Collect only the referenced entities (tiny subset of full data)
    h0_pd = (
        hydros_lf.filter(
            (pl.col("block_id") == 0) & pl.col("hydro_id").is_in(list(hydro_ids_needed))
        )
        .select(["scenario_id", "stage_id", "hydro_id", "storage_final_hm3"])
        .collect(engine="streaming")
        .to_pandas()
    )
    hg_pd = (
        hydros_lf.filter(pl.col("hydro_id").is_in(list(hydro_ids_needed)))
        .select(["scenario_id", "stage_id", "block_id", "hydro_id", "generation_mw"])
        .collect(engine="streaming")
        .to_pandas()
    )
    ex_pd = (
        (
            exchanges_lf.filter(pl.col("line_id").is_in(list(line_ids_needed)))
            .select(["scenario_id", "stage_id", "block_id", "line_id", "net_flow_mw"])
            .collect(engine="streaming")
            .to_pandas()
        )
        if line_ids_needed
        else pd.DataFrame(
            columns=["scenario_id", "stage_id", "block_id", "line_id", "net_flow_mw"]
        )
    )

    all_results: list[pd.DataFrame] = []

    for c in constraints:
        cid = c["id"]
        expr = c["expression"]
        terms = _parse_expression(expr)
        if not terms:
            continue

        var_types = {t[1] for t in terms}
        storage_only = var_types == {"hydro_storage"}

        if storage_only:
            # One LHS value per (scenario, stage) — use block_id 0
            # Start with a base frame of all (scenario, stage) combos from h0
            base = h0_pd[["scenario_id", "stage_id"]].drop_duplicates().copy()
            base["_lhs"] = 0.0
            for coeff, vtype, eid in terms:
                sub = h0_pd[h0_pd["hydro_id"] == eid][
                    ["scenario_id", "stage_id", "storage_final_hm3"]
                ].rename(columns={"storage_final_hm3": "_val"})
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
            # Per (scenario, stage, block) — need consistent block grid
            # Build base from hydro_generation block grid (always present)
            base = (
                hg_pd[["scenario_id", "stage_id", "block_id"]].drop_duplicates().copy()
            )
            base["_lhs"] = 0.0

            for coeff, vtype, eid in terms:
                if vtype == "hydro_storage":
                    # Storage is stage-level: broadcast across all blocks
                    sub = h0_pd[h0_pd["hydro_id"] == eid][
                        ["scenario_id", "stage_id", "storage_final_hm3"]
                    ].rename(columns={"storage_final_hm3": "_val"})
                    merged = base.merge(sub, on=["scenario_id", "stage_id"], how="left")
                elif vtype == "hydro_generation":
                    sub = hg_pd[hg_pd["hydro_id"] == eid][
                        ["scenario_id", "stage_id", "block_id", "generation_mw"]
                    ].rename(columns={"generation_mw": "_val"})
                    merged = base.merge(
                        sub, on=["scenario_id", "stage_id", "block_id"], how="left"
                    )
                else:  # line_exchange
                    sub = ex_pd[ex_pd["line_id"] == eid][
                        ["scenario_id", "stage_id", "block_id", "net_flow_mw"]
                    ].rename(columns={"net_flow_mw": "_val"})
                    merged = base.merge(
                        sub, on=["scenario_id", "stage_id", "block_id"], how="left"
                    )
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


def build_constraints_summary_table(
    constraints: list[dict],
    gc_bounds: pd.DataFrame,
    violations_df: pd.DataFrame,
) -> str:
    """HTML summary table of all generic constraints.

    Columns: Name, Type, Sense, Active Stages, Bound Range, Slack, Penalty, Has Violations
    Rows are colour-coded by constraint type.
    """
    type_colors = {
        "VminOP": "#EEF4FB",
        "RE": "#F0FAF4",
        "AGRINT": "#FFF8EE",
    }
    rows_html: list[str] = []
    for c in constraints:
        cid = c["id"]
        name = c["name"]
        ctype = name.split("_")[0]
        sense = c["sense"]
        slack = c["slack"]
        slack_enabled = "Yes" if slack.get("enabled") else "No"
        penalty = (
            f"{slack['penalty']:,.0f}"
            if slack.get("enabled") and "penalty" in slack
            else "—"
        )
        bounds_rows = gc_bounds[gc_bounds["constraint_id"] == cid]
        active_stages = (
            int(bounds_rows["stage_id"].nunique()) if not bounds_rows.empty else 0
        )
        bmin = bounds_rows["bound"].min() if not bounds_rows.empty else 0.0
        bmax = bounds_rows["bound"].max() if not bounds_rows.empty else 0.0
        if abs(bmax - bmin) < 1e-6:
            bound_range = f"{bmin:,.1f}"
        else:
            bound_range = f"{bmin:,.1f} – {bmax:,.1f}"
        has_viol = "No"
        if not violations_df.empty and "constraint_id" in violations_df.columns:
            viol_sub = violations_df[violations_df["constraint_id"] == cid]
            if not viol_sub.empty and viol_sub["slack_value"].abs().sum() > 1e-6:
                has_viol = "Yes"
        bg = type_colors.get(ctype, "#FFFFFF")
        viol_style = (
            ' style="color:#DC4C4C;font-weight:600;"' if has_viol == "Yes" else ""
        )
        rows_html.append(
            f'<tr style="background:{bg};">'
            f"<td>{name}</td>"
            f"<td>{ctype}</td>"
            f"<td><code>{sense}</code></td>"
            f"<td style='text-align:center;'>{active_stages}</td>"
            f"<td style='text-align:right;'>{bound_range}</td>"
            f"<td style='text-align:center;'>{slack_enabled}</td>"
            f"<td style='text-align:right;'>{penalty}</td>"
            f"<td style='text-align:center;'{viol_style}>{has_viol}</td>"
            "</tr>"
        )

    legend_html = (
        '<div style="margin-top:8px;font-size:0.8rem;color:#666;">'
        '<span style="background:#EEF4FB;padding:2px 8px;margin-right:8px;">VminOP — Minimum stored energy</span>'
        '<span style="background:#F0FAF4;padding:2px 8px;margin-right:8px;">RE — Electric constraint</span>'
        '<span style="background:#FFF8EE;padding:2px 8px;">AGRINT — Exchange group constraint</span>'
        "</div>"
    )

    return (
        '<table class="data-table">'
        "<thead><tr>"
        "<th>Name</th><th>Type</th><th>Sense</th>"
        "<th>Active Stages</th><th>Bound Range</th>"
        "<th>Slack</th><th>Penalty (R$/unit)</th><th>Has Violations</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>" + legend_html
    )
