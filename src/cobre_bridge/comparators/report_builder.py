"""Assemble the full HTML comparison report from results data.

Combines chart implementations and the HTML template into a complete
self-contained report.
"""

from __future__ import annotations

from cobre_bridge.comparators.charts import (
    convergence_chart,
    hydro_aggregate_chart,
    overview_metrics,
    productivity_scatter,
    system_comparison_chart,
    thermal_generation_chart,
)
from cobre_bridge.comparators.html_report import (
    build_comparison_html,
    chart_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.comparators.results import (
    ResultComparison,
    build_results_summary,
)


def build_comparison_report(
    results: list[ResultComparison],
) -> str:
    """Build a complete HTML comparison report.

    Parameters
    ----------
    results:
        List of all comparison results from ``compare_results``.

    Returns
    -------
    str
        Complete HTML document string.
    """
    summary = build_results_summary(results)

    tab_contents: dict[str, str] = {}

    # --- Overview tab ---
    overview_parts: list[str] = []
    overview_parts.append(overview_metrics(summary))
    overview_parts.append(section_title("Convergence"))
    overview_parts.append(
        chart_grid(
            [wrap_chart(convergence_chart(results))],
            single=True,
        )
    )
    tab_contents["tab-overview"] = "\n".join(overview_parts)

    # --- System tab ---
    system_parts: list[str] = []
    system_parts.append(section_title("System Comparison"))
    system_charts: list[str] = []
    for var, title in [
        ("spot_price", "Spot Price (CMO)"),
        ("deficit_mw", "Deficit"),
    ]:
        system_charts.append(wrap_chart(system_comparison_chart(results, var, title)))
    system_parts.append(chart_grid(system_charts))
    tab_contents["tab-system"] = "\n".join(system_parts)

    # --- Hydro tab ---
    hydro_parts: list[str] = []
    hydro_parts.append(section_title("Aggregate Hydro Comparison"))
    hydro_charts: list[str] = []
    for var, title in [
        ("storage_final_hm3", "Total Storage"),
        ("generation_mw", "Hydro Generation"),
        ("spillage_m3s", "Total Spillage"),
    ]:
        hydro_charts.append(wrap_chart(hydro_aggregate_chart(results, var, title)))
    hydro_parts.append(chart_grid(hydro_charts))
    tab_contents["tab-hydro"] = "\n".join(hydro_parts)

    # --- Thermal tab ---
    thermal_parts: list[str] = []
    thermal_parts.append(section_title("Thermal Generation Comparison"))
    thermal_parts.append(
        chart_grid(
            [wrap_chart(thermal_generation_chart(results))],
            single=True,
        )
    )
    tab_contents["tab-thermal"] = "\n".join(thermal_parts)

    # --- Productivity tab ---
    prod_parts: list[str] = []
    prod_parts.append(section_title("Productivity Comparison"))
    prod_parts.append(
        chart_grid(
            [wrap_chart(productivity_scatter(results))],
            single=True,
        )
    )
    tab_contents["tab-productivity"] = "\n".join(prod_parts)

    return build_comparison_html(
        title="Cobre vs NEWAVE Results Comparison",
        tab_contents=tab_contents,
    )
