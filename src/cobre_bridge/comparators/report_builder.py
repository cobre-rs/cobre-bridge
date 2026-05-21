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
    cobre_aggregate_chart,
    convergence_chart,
    cost_breakdown_chart,
    cost_breakdown_table,
    hydro_aggregate_chart,
    hydro_per_bus_chart,
    line_summary_chart,
    overview_metrics,
    performance_fwd_bwd_split_chart,
    performance_iteration_chart,
    performance_metric_cards,
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
    nw_costs = pctiles.nw_costs if pctiles else {}
    cobre_costs = pctiles.cobre_costs if pctiles else {}
    overview_parts.append(overview_metrics(summary, nw_costs, cobre_costs))
    overview_parts.append(section_title("Cost Breakdown"))
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(cost_breakdown_chart(nw_costs, cobre_costs)),
                wrap_chart(cost_breakdown_table(nw_costs, cobre_costs)),
            ],
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
    balance_html = build_energy_balance_tab(
        pctiles.nw_market if pctiles else pl.DataFrame(),
        pctiles.bus_aggregates if pctiles else pl.DataFrame(),
        pctiles.cobre_bus_meta if pctiles else {},
        pctiles.nw_bus_names if pctiles else {},
        nw_net_load=pctiles.nw_net_load if pctiles else pl.DataFrame(),
    )
    energy_balance_extra: list[str] = []
    if pctiles is not None and not pctiles.cobre_hydro_means.is_empty():
        energy_balance_extra.append(section_title("System Energy (EARM / ENA)"))
        energy_balance_extra.append(
            chart_grid(
                [
                    wrap_chart(
                        cobre_aggregate_chart(
                            pctiles.cobre_hydro_means,
                            "stored_energy_final_mwh",
                            "System Stored Energy (EARM)",
                            "MWh",
                            pctiles.hydro,
                            nw_sin=pctiles.nw_sin,
                            nw_variable="EARMF",
                            nw_factor=730.0,
                            nw_offset=pctiles.nw_offset,
                        )
                    ),
                    wrap_chart(
                        cobre_aggregate_chart(
                            pctiles.cobre_hydro_means,
                            "incremental_inflow_energy_mw",
                            "System Natural Inflow Energy (ENA)",
                            "MW",
                            pctiles.hydro,
                            nw_sin=pctiles.nw_sin,
                            nw_variable="ENA",
                            nw_factor=1.0,
                            nw_offset=pctiles.nw_offset,
                        )
                    ),
                ]
            )
        )
    tab_contents["tab-balance"] = balance_html + "\n" + "\n".join(energy_balance_extra)

    # --- Network tab ---
    line_pct = pctiles.line if pctiles else None
    line_bounds = pctiles.line_bounds if pctiles else None
    line_meta = pctiles.line_meta if pctiles else []
    network_parts: list[str] = []
    network_parts.append(section_title("Line Net Flow"))
    network_parts.append(
        chart_grid(
            [wrap_chart(line_summary_chart(results, line_pct, line_bounds, line_meta))],
            single=True,
        )
    )
    tab_contents["tab-network"] = "\n".join(network_parts)

    # --- Hydro Operation tab ---
    hydro_pct = pctiles.hydro if pctiles else None
    cobre_hydro_means = pctiles.cobre_hydro_means if pctiles else pl.DataFrame()
    nw_sin = pctiles.nw_sin if pctiles else pl.DataFrame()
    nw_offset = pctiles.nw_offset if pctiles else 0
    matched_hydro_ids = {r.cobre_id for r in results if r.entity_type == "hydro"}

    hydro_meta = pctiles.cobre_hydro_meta if pctiles else {}
    bus_meta = pctiles.cobre_bus_meta if pctiles else {}
    hydro_parts: list[str] = []
    for var, title in [
        ("storage_final_hm3", "Storage by Bus (hm³)"),
        ("generation_mw", "Hydro Generation by Bus (MW)"),
        ("spillage_m3s", "Spillage by Bus (m³/s)"),
        ("turbined_m3s", "Turbined by Bus (m³/s)"),
        ("inflow_m3s", "Inflow by Bus (m³/s)"),
        ("water_value_per_hm3", "Water Value by Bus (R$/hm³)"),
    ]:
        hydro_parts.append(section_title(title))
        hydro_parts.append(
            chart_grid(
                [
                    wrap_chart(
                        hydro_per_bus_chart(
                            results, var, title, hydro_pct, hydro_meta, bus_meta
                        )
                    )
                ],
                single=True,
            )
        )

    # System-level EARM and ENA (Cobre per-hydro aggregate vs NEWAVE SIN).
    # NEWAVE EARMF is in MWmes (mean MW over a month); convert to MWh via
    # the canonical 730 h/month factor used by NEWAVE.  ENA is already in
    # MW (mean power) on both sides.
    hydro_parts.append(section_title("Aggregate Energy Variables"))
    energy_charts = [
        wrap_chart(
            cobre_aggregate_chart(
                cobre_hydro_means,
                "stored_energy_final_mwh",
                "Stored Energy (EARM)",
                "MWh",
                hydro_pct,
                nw_sin=nw_sin,
                nw_variable="EARMF",
                nw_factor=730.0,
                nw_offset=nw_offset,
                matched_ids=matched_hydro_ids or None,
            )
        ),
        wrap_chart(
            cobre_aggregate_chart(
                cobre_hydro_means,
                "incremental_inflow_energy_mw",
                "Natural Inflow Energy (ENA)",
                "MW",
                hydro_pct,
                nw_sin=nw_sin,
                nw_variable="ENA",
                nw_factor=1.0,
                nw_offset=nw_offset,
                matched_ids=matched_hydro_ids or None,
            )
        ),
    ]
    hydro_parts.append(chart_grid(energy_charts))

    # System-aggregate (SIN) totals for each hydro variable. Sums Cobre
    # plant values per stage and overlays the NEWAVE total. Mirrors the
    # per-bus facet section but collapses across buses — useful as a
    # one-glance global view alongside the per-bus disaggregation.
    hydro_parts.append(section_title("System Totals (SIN)"))
    aggregate_charts: list[str] = []
    for var, title in [
        ("storage_final_hm3", "Total Storage (hm³)"),
        ("generation_mw", "Hydro Generation (MW)"),
        ("spillage_m3s", "Total Spillage (m³/s)"),
        ("turbined_m3s", "Total Turbined (m³/s)"),
        ("inflow_m3s", "Total Inflow (m³/s)"),
        ("water_value_per_hm3", "Water Value (R$/hm³)"),
    ]:
        aggregate_charts.append(
            wrap_chart(hydro_aggregate_chart(results, var, title, hydro_pct))
        )
    hydro_parts.append(chart_grid(aggregate_charts))

    tab_contents["tab-hydro"] = "\n".join(hydro_parts)

    # --- Hydro Plant Details tab ---
    tab_contents["tab-hydro-detail"] = build_hydro_detail_tab(
        results,
        hydro_pct,
        cobre_hydro_means,
    )

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

    # --- Performance tab ---
    nw_tim_iters = pctiles.nw_tim_iterations if pctiles else pl.DataFrame()
    nw_tim_stages = pctiles.nw_tim_stages if pctiles else {}
    cb_train_secs = pctiles.cobre_training_seconds if pctiles else 0.0
    cb_conv_perf = pctiles.cobre_iteration_timing if pctiles else pl.DataFrame()
    perf_parts: list[str] = []
    perf_parts.append(performance_metric_cards(nw_tim_stages, cb_train_secs))
    perf_parts.append(section_title("Time per Iteration"))
    perf_parts.append(
        chart_grid(
            [wrap_chart(performance_iteration_chart(nw_tim_iters, cb_conv_perf))],
            single=True,
        )
    )
    perf_parts.append(section_title("Forward / Backward Split"))
    perf_parts.append(
        chart_grid(
            [wrap_chart(performance_fwd_bwd_split_chart(nw_tim_iters, cb_conv_perf))],
            single=True,
        )
    )
    tab_contents["tab-performance"] = "\n".join(perf_parts)

    return build_comparison_html(
        title="Cobre vs NEWAVE Results Comparison",
        tab_contents=tab_contents,
    )
