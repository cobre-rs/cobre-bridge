"""Assemble the full HTML comparison report from results data.

Combines chart implementations and the HTML template into a complete
self-contained report.
"""

from __future__ import annotations

import polars as pl

from cobre_bridge.comparators.charts import (
    build_energy_balance_tab,
    build_hydro_detail_tab,
    build_thermal_detail_tab,
    convergence_chart,
    cost_breakdown_chart,
    hydro_aggregate_chart,
    overview_metrics,
    productivity_scatter,
    system_comparison_chart,
    system_per_bus_chart,
    thermal_generation_chart,
)
from cobre_bridge.comparators.html_report import (
    build_comparison_html,
    chart_grid,
    section_title,
    wrap_chart,
)
from cobre_bridge.comparators.results import (
    PercentileData,
    ResultComparison,
    build_results_summary,
)


def build_comparison_report(
    results: list[ResultComparison],
    pctiles: PercentileData | None = None,
) -> str:
    """Build a complete HTML comparison report.

    Parameters
    ----------
    results:
        List of all comparison results from ``compare_results``.
    pctiles:
        Cobre simulation percentile statistics (p10/p50/p90).

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
    overview_parts.append(section_title("Cost Breakdown"))
    nw_costs = pctiles.nw_costs if pctiles else {}
    cobre_costs = pctiles.cobre_costs if pctiles else {}
    overview_parts.append(
        chart_grid(
            [wrap_chart(cost_breakdown_chart(nw_costs, cobre_costs))],
            single=True,
        )
    )
    overview_parts.append(section_title("Convergence"))
    nw_conv = pctiles.nw_convergence if pctiles else pl.DataFrame()
    cb_conv = pctiles.cobre_convergence if pctiles else pl.DataFrame()
    overview_parts.append(
        chart_grid(
            [wrap_chart(convergence_chart(nw_conv, cb_conv))],
            single=True,
        )
    )
    tab_contents["tab-overview"] = "\n".join(overview_parts)

    # --- System tab ---
    bus_pct = pctiles.bus if pctiles else None
    system_parts: list[str] = []
    system_parts.append(section_title("Spot Price by Bus"))
    system_parts.append(
        chart_grid(
            [
                wrap_chart(
                    system_per_bus_chart(results, "spot_price", "CMO by Bus", bus_pct)
                )
            ],
            single=True,
        )
    )
    system_parts.append(section_title("Deficit"))
    system_parts.append(
        chart_grid(
            [
                wrap_chart(
                    system_comparison_chart(results, "deficit_mw", "Deficit", bus_pct)
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-system"] = "\n".join(system_parts)

    # --- Energy Balance tab ---
    tab_contents["tab-balance"] = build_energy_balance_tab(
        pctiles.nw_market if pctiles else pl.DataFrame(),
        pctiles.bus_aggregates if pctiles else pl.DataFrame(),
        pctiles.cobre_bus_meta if pctiles else {},
        pctiles.nw_bus_names if pctiles else {},
        nw_net_load=pctiles.nw_net_load if pctiles else pl.DataFrame(),
    )

    # --- Hydro Operation tab ---
    hydro_pct = pctiles.hydro if pctiles else None
    hydro_parts: list[str] = []
    hydro_parts.append(section_title("Aggregate Hydro Comparison"))
    hydro_charts: list[str] = []
    for var, title in [
        ("storage_final_hm3", "Total Storage (hm³)"),
        ("generation_mw", "Hydro Generation (MW)"),
        ("spillage_m3s", "Total Spillage (m³/s)"),
        ("turbined_m3s", "Total Turbined (m³/s)"),
        ("inflow_m3s", "Total Inflow (m³/s)"),
        ("water_value_per_hm3", "Water Value (R$/hm³)"),
    ]:
        hydro_charts.append(
            wrap_chart(hydro_aggregate_chart(results, var, title, hydro_pct))
        )
    hydro_parts.append(chart_grid(hydro_charts))
    tab_contents["tab-hydro"] = "\n".join(hydro_parts)

    # --- Hydro Plant Details tab ---
    tab_contents["tab-hydro-detail"] = build_hydro_detail_tab(results, hydro_pct)

    # --- Thermal Operation tab ---
    thermal_pct = pctiles.thermal if pctiles else None
    thermal_parts: list[str] = []
    thermal_parts.append(section_title("Thermal Generation Comparison"))
    thermal_parts.append(
        chart_grid(
            [wrap_chart(thermal_generation_chart(results, thermal_pct))],
            single=True,
        )
    )
    tab_contents["tab-thermal"] = "\n".join(thermal_parts)

    # --- Thermal Plant Details tab ---
    tab_contents["tab-thermal-detail"] = build_thermal_detail_tab(results, thermal_pct)

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
