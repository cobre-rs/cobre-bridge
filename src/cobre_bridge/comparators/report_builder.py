"""Assemble the full HTML comparison report from results data.

Combines chart implementations and the HTML template into a complete
self-contained report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd  # type: ignore[import-untyped]  # pandas-stubs not installed
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
    fpha_detail_chart,
    fpha_metrics_table,
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
from cobre_bridge.comparators.report import _footer_counts
from cobre_bridge.comparators.results import (
    ResultComparison,
    ResultsSummary,
    ResultVariableStats,
)

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset


# -------------------------------------------------------------------
# Typed metadata accessors for the migrated tab blocks (ticket-013)
#
# Each isinstance-guards its named key and returns a safe default when the key
# is absent or ill-typed, reproducing the legacy ``pct.<field> if pctiles else
# <default>`` semantics for the empty ``PercentileData()`` case. Never raises.
# -------------------------------------------------------------------


def _meta_frame(metadata: dict[str, object], key: str) -> pl.DataFrame:
    """Return ``metadata[key]`` as a ``pl.DataFrame`` (empty frame on miss)."""
    value = metadata.get(key)
    if isinstance(value, pl.DataFrame):
        return value
    return pl.DataFrame()


def _meta_pd_frame(metadata: dict[str, object], key: str) -> pd.DataFrame:
    """Return ``metadata[key]`` as a ``pd.DataFrame`` (empty frame on miss)."""
    value = metadata.get(key)
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()


def _meta_int(metadata: dict[str, object], key: str) -> int:
    """Return ``metadata[key]`` as an ``int`` (``0`` on miss/ill-typed).

    ``bool`` is rejected so a stray boolean never masquerades as ``0``/``1``.
    """
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return 0


def _meta_opt_int(metadata: dict[str, object], key: str) -> int | None:
    """Return ``metadata[key]`` as an ``int | None`` (``None`` on miss/ill-typed).

    Mirrors :func:`_meta_int` but the safe default is ``None`` so the legacy
    ``pct.nw_max_stage if pctiles else None`` semantics are reproduced — keeping
    the downstream ``if gc_max_stage is not None:`` filter intact. ``bool`` is
    rejected so a stray boolean never masquerades as an ``int``.
    """
    value = metadata.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _meta_float(metadata: dict[str, object], key: str) -> float:
    """Return ``metadata[key]`` as a ``float`` (``0.0`` on miss/ill-typed).

    Reproduces the legacy ``pct.cobre_training_seconds if pctiles else 0.0``
    semantics. ``bool`` is rejected; ``int`` is accepted and widened to float.
    """
    value = metadata.get(key)
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _meta_dict(metadata: dict[str, object], key: str) -> dict[object, object]:
    """Return ``metadata[key]`` as a ``dict`` (empty dict on miss/ill-typed)."""
    value = metadata.get(key)
    if isinstance(value, dict):
        return value
    return {}


def _meta_list(metadata: dict[str, object], key: str) -> list[object]:
    """Return ``metadata[key]`` as a ``list`` (empty list on miss/ill-typed)."""
    value = metadata.get(key)
    if isinstance(value, list):
        return value
    return []


def _results_summary_from_dataset(dataset: ComparisonDataset) -> ResultsSummary:
    """Reconstruct a :class:`ResultsSummary` from the canonical dataset.

    Rebuilds the summary object that ``overview_metrics`` consumes straight from
    the already-computed ``dataset.summary`` rows (one
    :class:`ResultVariableStats` per variable) and the ``footer_counts`` metadata
    (``total`` + ``by_entity_type``), so no per-row statistic is recomputed. The
    per-variable ``mean_rel_diff`` / ``max_rel_diff`` fields aren't carried in
    ``SUMMARY_SCHEMA`` (and aren't read by ``overview_metrics``); they keep their
    dataclass defaults of ``0.0``.
    """
    total, by_entity_type = _footer_counts(dataset)

    by_variable: dict[str, ResultVariableStats] = {}
    for row in dataset.summary.to_dicts():
        correlation = row["correlation"]
        by_variable[str(row["variable"])] = ResultVariableStats(
            count=int(row["count"]),
            mean_abs_diff=float(row["mean_abs_diff"]),
            max_abs_diff=float(row["max_abs_diff"]),
            within_tol_rate=float(row["within_tol_rate"]),
            mean_smape=float(row["mean_smape"]),
            max_smape=float(row["max_smape"]),
            correlation=float(correlation) if correlation is not None else None,
        )

    return ResultsSummary(
        total=total,
        by_entity_type=by_entity_type,
        by_variable=by_variable,
    )


def build_comparison_report(dataset: ComparisonDataset) -> str:
    """Build a complete HTML comparison report.

    Every tab sources its inputs from ``dataset.metadata``: the migrated tabs
    read their named frame/dict/list/int keys via the typed accessors, and the
    chart functions that still take ``list[ResultComparison]`` directly read the
    raw rows from the render-only ``metadata["results"]`` key.

    Parameters
    ----------
    dataset:
        The canonical comparison dataset. Its ``metadata`` carries every render
        input (named per-tab keys plus the ``results`` list); these are
        in-memory render-only carry-overs, excluded from the serialized
        artifact (see ``RENDER_ONLY_METADATA_KEYS``).

    Returns
    -------
    str
        Complete HTML document string.
    """
    results = [
        r
        for r in _meta_list(dataset.metadata, "results")
        if isinstance(r, ResultComparison)
    ]
    # Reconstruct the overview summary from the dataset's already-computed
    # summary rows + footer counts rather than recomputing it from the raw
    # ``results`` list — the chart functions below still consume ``results``
    # directly, but ``overview_metrics`` reads only the reconstructed summary.
    summary = _results_summary_from_dataset(dataset)

    tab_contents: dict[str, str] = {}

    # --- Overview tab ---
    overview_parts: list[str] = []
    nw_costs = cast("dict[str, float]", _meta_dict(dataset.metadata, "nw_costs"))
    cobre_costs = cast("dict[str, float]", _meta_dict(dataset.metadata, "cobre_costs"))
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
    nw_sin = _meta_frame(dataset.metadata, "nw_sin")
    cobre_stage_costs = _meta_frame(dataset.metadata, "cobre_stage_costs")
    nw_offset = _meta_int(dataset.metadata, "nw_offset")
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
    # Thermal-only (CTERM, live on both sides) and the non-thermal remainder (COPER −
    # CTERM) — the latter goes negative for the source model in the post-study because
    # COPER is frozen at the last study value while CTERM stays live.
    overview_parts.append(
        chart_grid(
            [
                wrap_chart(thermal_cost_chart(nw_sin, cobre_stage_costs, nw_offset)),
                wrap_chart(other_costs_chart(nw_sin, cobre_stage_costs, nw_offset)),
            ],
        )
    )

    overview_parts.append(section_title("Convergence"))
    nw_conv = _meta_frame(dataset.metadata, "nw_convergence")
    cb_conv = _meta_frame(dataset.metadata, "cobre_convergence")
    overview_parts.append(
        chart_grid(
            [wrap_chart(convergence_chart(nw_conv, cb_conv))],
            single=True,
        )
    )
    tab_contents["tab-overview"] = "\n".join(overview_parts)

    # --- System tab ---
    bus_pct = _meta_frame(dataset.metadata, "bus")
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
        _meta_frame(dataset.metadata, "nw_market"),
        _meta_frame(dataset.metadata, "bus_aggregates"),
        cast(
            "dict[int, dict[object, object]]",
            _meta_dict(dataset.metadata, "cobre_bus_meta"),
        ),
        cast("dict[int, str]", _meta_dict(dataset.metadata, "nw_bus_names")),
        nw_net_load=_meta_frame(dataset.metadata, "nw_net_load"),
    )
    balance_cobre_hydro_means = _meta_frame(dataset.metadata, "cobre_hydro_means")
    balance_hydro = _meta_frame(dataset.metadata, "hydro")
    balance_nw_sin = _meta_frame(dataset.metadata, "nw_sin")
    balance_nw_offset = _meta_int(dataset.metadata, "nw_offset")
    energy_balance_extra: list[str] = []
    if not balance_cobre_hydro_means.is_empty():
        energy_balance_extra.append(section_title("System Energy (EARM / ENA)"))
        energy_balance_extra.append(
            chart_grid(
                [
                    wrap_chart(
                        cobre_aggregate_chart(
                            balance_cobre_hydro_means,
                            "stored_energy_final_mwh",
                            "System Stored Energy (EARM)",
                            "MWh",
                            balance_hydro,
                            nw_sin=balance_nw_sin,
                            nw_variable="EARMF",
                            nw_factor=730.0,
                            nw_offset=balance_nw_offset,
                        )
                    ),
                    wrap_chart(
                        cobre_aggregate_chart(
                            balance_cobre_hydro_means,
                            "incremental_inflow_energy_mw",
                            "System Natural Inflow Energy (ENA)",
                            "MW",
                            balance_hydro,
                            nw_sin=balance_nw_sin,
                            nw_variable="ENA",
                            nw_factor=1.0,
                            nw_offset=balance_nw_offset,
                        )
                    ),
                ]
            )
        )
    tab_contents["tab-balance"] = balance_html + "\n" + "\n".join(energy_balance_extra)

    # --- Network tab ---
    line_pct = _meta_frame(dataset.metadata, "line")
    line_bounds = _meta_pd_frame(dataset.metadata, "line_bounds")
    line_meta = cast(
        "list[dict[object, object]]", _meta_list(dataset.metadata, "line_meta")
    )
    network_parts: list[str] = []
    network_parts.append(section_title("Line Net Flow"))
    network_parts.append(
        chart_grid(
            [wrap_chart(line_summary_chart(results, line_pct, line_bounds, line_meta))],
            single=True,
        )
    )
    tab_contents["tab-network"] = "\n".join(network_parts)

    # --- Constraints tab --- Per-constraint LHS comparison: The source-model-side LHS
    # evaluated against MEDIAS-USIH / int*.out output, Cobre-side LHS as the mean across
    # scenarios and blocks from the simulation parquet. Bounds are taken from
    # constraints/generic_constraint_bounds.parquet (block 0 preferred when blocks
    # disagree).
    gc_constraints = cast(
        "list[dict[object, object]]", _meta_list(dataset.metadata, "gc_constraints")
    )
    gc_bounds_df = _meta_frame(dataset.metadata, "gc_bounds")
    gc_lhs_nw = _meta_frame(dataset.metadata, "gc_lhs_newave")
    gc_lhs_cb = _meta_frame(dataset.metadata, "gc_lhs_cobre")
    gc_max_stage = _meta_opt_int(dataset.metadata, "nw_max_stage")
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
    hydro_pct = _meta_frame(dataset.metadata, "hydro")
    cobre_hydro_means = _meta_frame(dataset.metadata, "cobre_hydro_means")
    nw_sin = _meta_frame(dataset.metadata, "nw_sin")
    nw_offset = _meta_int(dataset.metadata, "nw_offset")
    matched_hydro_ids = {r.cobre_id for r in results if r.entity_type == "hydro"}

    hydro_meta = cast(
        "dict[int, dict[object, object]]",
        _meta_dict(dataset.metadata, "cobre_hydro_meta"),
    )
    bus_meta = cast(
        "dict[int, dict[object, object]]",
        _meta_dict(dataset.metadata, "cobre_bus_meta"),
    )
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

    # System-level EARM and ENA (Cobre per-hydro aggregate vs the source model SIN). The
    # source model EARMF is in MWmes (mean MW over a month); convert to MWh via the
    # canonical 730 h/month factor used by the source model.  ENA is already in MW (mean
    # power) on both sides.
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

    # System-aggregate (SIN) totals for each hydro variable. Sums Cobre plant values per
    # stage and overlays the source model total. Mirrors the per-bus facet section but
    # collapses across buses — useful as a one-glance global view alongside the per-bus
    # disaggregation.
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
    # variables above, but driven by the per-(entity_id, stage_id) Cobre frame and the
    # source model slack frame (no ResultComparison rows exist for slacks).  The inflow
    # non-negativity slack has no source-model counterpart, so its the source model
    # source is passed as None — the chart still renders the Cobre Mean + p10/p90 band,
    # just without an overlaid the source model line.
    nw_hydro_slacks = _meta_frame(dataset.metadata, "nw_hydro_slacks")
    # Withdrawal pos/neg are SWAPPED to follow the source model's sign convention; the
    # ``_NW_HYDRO_SLACK_VARS`` mapping in ``results.py`` is correspondingly swapped so
    # each panel pairs the right Cobre column with the right The source model series.
    # Evaporation pos/neg already share the source model's convention.
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
        cobre_hydro_meta=cast(
            "dict[int, dict[object, object]]",
            _meta_dict(dataset.metadata, "cobre_hydro_meta"),
        ),
        cobre_hydro_per_stage_bounds=_meta_frame(
            dataset.metadata, "cobre_hydro_per_stage_bounds"
        ),
        nw_hydro_slacks=_meta_frame(dataset.metadata, "nw_hydro_slacks"),
    )

    # --- Thermal Operation tab ---
    thermal_pct = _meta_frame(dataset.metadata, "thermal")
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
    prod_df = _meta_frame(dataset.metadata, "productivity_detail")
    per_stage_df = _meta_frame(dataset.metadata, "productivity_per_stage")
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
        prod_parts.append(productivity_per_stage_chart(per_stage_df))
        prod_parts.append(section_title("Productivity Building Blocks"))
        prod_parts.append(
            chart_grid(
                [wrap_chart(productivity_blocks_table(prod_df))],
                single=True,
            )
        )

    # --- Fitted production functions (FPHA) --- Present only when both sides
    # fitted FPHA hyperplanes; compares the production surfaces GH(V, Q) the two
    # solvers fit, on a shared grid (run-of-river plants reduce to a Q-curve).
    fpha_metrics = _meta_frame(dataset.metadata, "fpha_metrics")
    if not fpha_metrics.is_empty():
        fpha_surface = _meta_frame(dataset.metadata, "fpha_surface")
        fpha_spill = _meta_frame(dataset.metadata, "fpha_spill")
        prod_parts.append(section_title("Fitted production functions (FPHA)"))
        prod_parts.append(
            '<p style="color:#64748B;margin:-8px 0 12px">Both solvers fit the'
            " hydro production surface GH(V, Q, S) as a set of hyperplanes; this"
            " compares the resulting surfaces at the fitting-grid nodes. Use the"
            " NEWAVE / Cobre / Both / Difference buttons to isolate each surface"
            " or their difference (Cobre − NEWAVE, MW); they nearly coincide at"
            " S = 0. Spillage (S) is shown separately at the max V/Q corner."
            " Pick a plant and stage.</p>"
        )
        prod_parts.append(fpha_detail_chart(fpha_surface, fpha_spill))
        prod_parts.append(section_title("FPHA surface fidelity by plant"))
        prod_parts.append(
            chart_grid([wrap_chart(fpha_metrics_table(fpha_metrics))], single=True)
        )

    tab_contents["tab-productivity"] = "\n".join(prod_parts)

    # --- Performance tab ---
    nw_tim_iters = _meta_frame(dataset.metadata, "nw_tim_iterations")
    nw_tim_stages = cast(
        "dict[str, float]", _meta_dict(dataset.metadata, "nw_tim_stages")
    )
    cb_train_secs = _meta_float(dataset.metadata, "cobre_training_seconds")
    cb_conv_perf = _meta_frame(dataset.metadata, "cobre_iteration_timing")
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
