"""Presentation helpers for the constraints tab.

The constraint-expression *domain* logic (parser, parameter resolver, LHS
evaluator) now lives in :mod:`cobre_bridge.constraint_expr` so the dashboard and
the comparator share one definition. This module keeps only the HTML rendering.
"""

from __future__ import annotations

import pandas as pd

from cobre_bridge.ui.html import escape_text


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
            f"<td>{escape_text(name)}</td>"
            f"<td>{escape_text(ctype)}</td>"
            f"<td><code>{escape_text(sense)}</code></td>"
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
