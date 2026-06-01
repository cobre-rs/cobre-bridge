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
    constraints_comparison_chart,
    convergence_chart,
    cost_breakdown_chart,
    cost_breakdown_table,
    future_cost_chart,
    hydro_aggregate_chart,
    hydro_per_bus_chart,
    hydro_slack_aggregate_chart,
    hydro_slack_per_bus_chart,
    immediate_cost_chart,
    line_summary_chart,
    other_costs_chart,
    overview_metrics,
    performance_fwd_bwd_split_chart,
    performance_iteration_chart,
    performance_metric_cards,
    productivity_blocks_table,
    productivity_comparison_scatter,
    productivity_per_stage_chart,
    system_comparison_chart,
    system_per_bus_chart,
    thermal_cost_chart,
    thermal_generation_chart,
)
from cobre_bridge.comparators.constraints_compare import per_stage_bounds
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
    overview_parts.append(section_title("Per-Stage Cost"))
    nw_sin = pctiles.nw_sin if pctiles else pl.DataFrame()
    cobre_stage_costs = pctiles.cobre_stage_costs if pctiles else pl.DataFrame()
    nw_offset = pctiles.nw_offset if pctiles else 0
    # Two side-by-side charts — immediate and future cost have very
    # different scales (one is per-stage operating cost, the other is a
    # cumulative future expectation), so we don't share an axis.
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(immediate_cost_chart(nw_sin, cobre_stage_costs, nw_offset)),
                wrap_chart(future_cost_chart(nw_sin, cobre_stage_costs, nw_offset)),
            ],
        )
    )
    # Thermal-only (CTERM, live on both sides) and the non-thermal remainder
    # (COPER − CTERM) — the latter goes negative for NEWAVE in the post-study
    # because COPER is frozen at the last study value while CTERM stays live.
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(thermal_cost_chart(nw_sin, cobre_stage_costs, nw_offset)),
                wrap_chart(other_costs_chart(nw_sin, cobre_stage_costs, nw_offset)),
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

    # --- Constraints tab ---
    # Per-constraint LHS comparison: NEWAVE-side LHS evaluated against
    # MEDIAS-USIH / int*.out output, Cobre-side LHS as the mean across
    # scenarios and blocks from the simulation parquet. Bounds are taken
    # from constraints/generic_constraint_bounds.parquet (block 0
    # preferred when blocks disagree).
    gc_constraints = pctiles.gc_constraints if pctiles else []
    gc_bounds_df = pctiles.gc_bounds if pctiles else pl.DataFrame()
    gc_lhs_nw = pctiles.gc_lhs_newave if pctiles else pl.DataFrame()
    gc_lhs_cb = pctiles.gc_lhs_cobre if pctiles else pl.DataFrame()
    gc_max_stage = pctiles.nw_max_stage if pctiles else None
    bound_lookup = per_stage_bounds(gc_bounds_df, max_stage=gc_max_stage)
    if gc_max_stage is not None:
        gc_lhs_nw = (
            gc_lhs_nw.filter(pl.col("stage_id") <= gc_max_stage)
            if not gc_lhs_nw.is_empty()
            else gc_lhs_nw
        )
        gc_lhs_cb = (
            gc_lhs_cb.filter(pl.col("stage_id") <= gc_max_stage)
            if not gc_lhs_cb.is_empty()
            else gc_lhs_cb
        )
    constraints_parts: list[str] = []
    constraints_parts.append(section_title("Generic Constraints — LHS vs Bound"))
    constraints_parts.append(
        chart_grid(
            [
                wrap_chart(
                    constraints_comparison_chart(
                        gc_constraints, gc_lhs_nw, gc_lhs_cb, bound_lookup
                    )
                )
            ],
            single=True,
        )
    )
    tab_contents["tab-constraints"] = "\n".join(constraints_parts)

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

    # Slack variables: same per-bus + SIN-total treatment as the operational
    # variables above, but driven by the per-(entity_id, stage_id) Cobre
    # frame and the NEWAVE slack frame (no ResultComparison rows exist for
    # slacks).  The inflow non-negativity slack has no NEWAVE counterpart,
    # so its NEWAVE source is passed as None — the chart still renders the
    # Cobre Mean + p10/p90 band, just without an overlaid NEWAVE line.
    nw_hydro_slacks = pctiles.nw_hydro_slacks if pctiles else pl.DataFrame()
    # Withdrawal pos/neg are SWAPPED to follow NEWAVE's sign convention; the
    # ``_NW_HYDRO_SLACK_VARS`` mapping in ``results.py`` is correspondingly
    # swapped so each panel pairs the right Cobre column with the right
    # NEWAVE series.  Evaporation pos/neg already share NEWAVE's convention.
    slack_specs: list[tuple[str, str, bool]] = [
        ("water_withdrawal_violation_neg_m3s", "Withdrawal Slack Pos (m³/s)", True),
        ("water_withdrawal_violation_pos_m3s", "Withdrawal Slack Neg (m³/s)", True),
        ("evaporation_violation_pos_m3s", "Evaporation Slack Pos (m³/s)", True),
        ("evaporation_violation_neg_m3s", "Evaporation Slack Neg (m³/s)", True),
        ("inflow_nonnegativity_slack_m3s", "Inflow Non-Negativity Slack (m³/s)", False),
    ]
    for var, slack_title, has_newave in slack_specs:
        hydro_parts.append(section_title(slack_title + " by Bus"))
        hydro_parts.append(
            chart_grid(
                [
                    wrap_chart(
                        hydro_slack_per_bus_chart(
                            cobre_hydro_means,
                            nw_hydro_slacks if has_newave else None,
                            var,
                            slack_title + " by Bus",
                            hydro_pct,
                            hydro_meta,
                            bus_meta,
                            matched_ids=matched_hydro_ids or None,
                        )
                    )
                ],
                single=True,
            )
        )

    hydro_parts.append(section_title("Hydro Slacks (SIN)"))
    slack_sin_charts = [
        wrap_chart(
            hydro_slack_aggregate_chart(
                cobre_hydro_means,
                nw_hydro_slacks if has_newave else None,
                var,
                slack_title,
                hydro_pct,
                matched_ids=matched_hydro_ids or None,
            )
        )
        for var, slack_title, has_newave in slack_specs
    ]
    hydro_parts.append(chart_grid(slack_sin_charts))

    tab_contents["tab-hydro"] = "\n".join(hydro_parts)

    # --- Hydro Plant Details tab ---
    tab_contents["tab-hydro-detail"] = build_hydro_detail_tab(
        results,
        hydro_pct,
        cobre_hydro_means,
        cobre_hydro_meta=pctiles.cobre_hydro_meta if pctiles else {},
        cobre_hydro_per_stage_bounds=(
            pctiles.cobre_hydro_per_stage_bounds if pctiles else pl.DataFrame()
        ),
        nw_hydro_slacks=(pctiles.nw_hydro_slacks if pctiles else pl.DataFrame()),
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
    prod_df = pctiles.productivity_detail if pctiles else pl.DataFrame()
    prod_parts: list[str] = []
    static_title = (
        "Static productivity — pmo vs cobre-bridge conversion "
        "(point / equivalent / accumulated)"
    )
    if prod_df.is_empty():
        prod_parts.append(section_title(static_title))
        prod_parts.append("<p>No productivity data available.</p>")
    else:
        prod_parts.append(section_title(static_title))
        prod_parts.append(
            chart_grid(
                [
                    wrap_chart(
                        productivity_comparison_scatter(
                            prod_df,
                            "point",
                            "Point — pmo altura_65 vs compute_productivity",
                        )
                    ),
                    wrap_chart(
                        productivity_comparison_scatter(
                            prod_df,
                            "equivalent",
                            "Equivalent — pmo vs stored_energy_productivity",
                        )
                    ),
                    wrap_chart(
                        productivity_comparison_scatter(
                            prod_df,
                            "accumulated",
                            "Accumulated — pmo vs cobre-bridge cascade",
                        )
                    ),
                ]
            )
        )
        prod_parts.append(section_title("Realized productivity across stages"))
        prod_parts.append(
            '<p style="color:#64748B;margin:-8px 0 12px">Productivity is constant'
            " within a stage but varies across stages, tracking the reservoir"
            " head reached each stage — pick a reservoir to compare NEWAVE vs"
            " Cobre.</p>"
        )
        # Reuses the shared per-plant dropdown widget (same as the hydro/thermal
        # detail tabs), so every reservoir is selectable — not a fixed subset.
        prod_parts.append(productivity_per_stage_chart(results))
        prod_parts.append(section_title("Productivity Building Blocks"))
        prod_parts.append(
            chart_grid(
                [wrap_chart(productivity_blocks_table(prod_df))],
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
