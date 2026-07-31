"""Adapters mapping the comparison-engine output into the tidy dataset.

These are the ANALYZE-layer adapters (epic-02). They turn the existing
``list[ResultComparison]`` + ``PercentileData`` value frames produced by
``results.py`` AND the ``list[BoundComparison]`` produced by ``bounds.py`` into
the canonical tidy/long frame defined by
:data:`cobre_bridge.comparators.dataset.TIDY_SCHEMA`.

It runs BEHIND the existing report flow (strangler-fig): it calls no readers and
recomputes no diffs — it consumes the passed-in objects verbatim and never
mutates them. ``bus`` enrichment, constraint/convergence/productivity rows are
out of scope here and are added when their tabs migrate in later epics; this
adapter emits the sentinel ``bus = -1`` / ``block = -1`` for every row.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np
import polars as pl

from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.results import build_results_summary, smape
from cobre_bridge.diagnostics import Diagnostic, Severity, emit

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pandas as pd

    from cobre_bridge.comparators.alignment import EntityAlignment, HydroEntity
    from cobre_bridge.comparators.bounds import BoundComparison
    from cobre_bridge.comparators.results import PercentileData, ResultComparison

_LOG = logging.getLogger(__name__)

#: Sentinel for tidy integer columns that do not apply to results rows.
_SENTINEL: int = -1

#: Percentile column suffix -> tidy ``source`` literal.
_PCT_SUFFIX_TO_SOURCE: dict[str, str] = {
    "_p10": "p10",
    "_p50": "p50",
    "_p90": "p90",
}

#: Identifier columns expected on every percentile frame.
_PCT_ENTITY_COL: str = "entity_id"
_PCT_STAGE_COL: str = "stage_id"


def tidy_from_results(results: Sequence[ResultComparison]) -> pl.DataFrame:
    """Map ``ResultComparison`` rows into tidy ``newave``/``cobre`` value rows.

    Each input row yields exactly two tidy rows: one ``source="newave"`` with
    ``value = r.newave_value`` and one ``source="cobre"`` with
    ``value = r.cobre_value``. Both carry ``entity_id = r.cobre_id``,
    ``entity_name = r.entity_name``, ``stage = r.stage``,
    ``variable = r.variable``, and the sentinels ``bus = -1`` / ``block = -1``.

    Args:
        results: The result comparisons to map; consumed verbatim.

    Returns:
        A frame conforming exactly (columns, order, dtypes) to
        :data:`TIDY_SCHEMA`, with ``2 * len(results)`` rows.
    """
    return _tidy_newave_cobre_pair(results)


def tidy_percentiles_from_percentile_data(pct: PercentileData) -> pl.DataFrame:
    """Map ``PercentileData`` percentile frames into tidy ``p10``/``p50``/``p90`` rows.

    Scans ``pct.hydro``, ``pct.thermal``, ``pct.bus`` and ``pct.line``. For each
    non-empty frame it detects ``{var}_p10``/``{var}_p50``/``{var}_p90`` column
    triplets and unpivots them into long rows, mapping the suffix to the tidy
    ``source`` (``p10``/``p50``/``p90``), ``entity_id`` from the frame's
    ``entity_id`` column, ``stage`` from ``stage_id``, ``variable`` from the
    stripped column stem, ``entity_name = ""``, ``bus = -1``, ``block = -1`` and
    ``entity_type`` per source frame.

    A frame missing the ``entity_id`` or ``stage_id`` column contributes no rows
    (it is logged at DEBUG and skipped — never raises). Empty frames are skipped.

    Args:
        pct: The percentile data; its frames are read but not mutated.

    Returns:
        A frame conforming exactly to :data:`TIDY_SCHEMA` (empty when no
        percentile triplet is found).
    """
    frames_by_type: list[tuple[str, pl.DataFrame]] = [
        ("hydro", pct.hydro),
        ("thermal", pct.thermal),
        ("bus", pct.bus),
        ("line", pct.line),
    ]

    parts: list[pl.DataFrame] = []
    for entity_type, frame in frames_by_type:
        part = _tidy_one_percentile_frame(entity_type, frame)
        if part is not None and part.height > 0:
            parts.append(part)

    if not parts:
        return pl.DataFrame(schema=TIDY_SCHEMA)

    return _conform(pl.concat(parts, how="vertical"))


def tidy_results_dataset(
    results: Sequence[ResultComparison],
    pct: PercentileData,
) -> pl.DataFrame:
    """Build the full tidy frame for the results subcommand.

    Vertically concatenates :func:`tidy_from_results` and
    :func:`tidy_percentiles_from_percentile_data` into a single frame conforming
    exactly to :data:`TIDY_SCHEMA`.

    Args:
        results: The result comparisons (``newave``/``cobre`` rows).
        pct: The percentile data (``p10``/``p50``/``p90`` rows).

    Returns:
        A frame conforming exactly to :data:`TIDY_SCHEMA`.
    """
    result_rows = tidy_from_results(results)
    pct_rows = tidy_percentiles_from_percentile_data(pct)
    return _conform(pl.concat([result_rows, pct_rows], how="vertical"))


def summary_frame_from_results(
    results: Sequence[ResultComparison],
    tolerance: float,
) -> pl.DataFrame:
    """Project the existing per-variable summary into a ``SUMMARY_SCHEMA`` frame.

    Calls the canonical :func:`cobre_bridge.comparators.results.build_results_summary`
    and turns its ``by_variable`` mapping into one row per variable. The numbers
    are carried verbatim from that summary — no statistic is recomputed here, so
    console and HTML aggregates derive from the same analysis.

    Args:
        results: The result comparisons to summarize; consumed verbatim.
        tolerance: Relative tolerance forwarded to ``build_results_summary`` for
            the per-variable within-tolerance match rate.

    Returns:
        A frame conforming exactly (columns, order, dtypes) to
        :data:`SUMMARY_SCHEMA`, one row per variable. Empty ``results`` yields a
        0-row frame.
    """
    if not results:
        return pl.DataFrame(schema=SUMMARY_SCHEMA)

    summary = build_results_summary(list(results), tolerance)

    variables: list[str] = []
    counts: list[int] = []
    mean_abs_diff: list[float] = []
    max_abs_diff: list[float] = []
    mean_smape: list[float] = []
    max_smape: list[float] = []
    within_tol_rate: list[float] = []
    correlation: list[float | None] = []

    for variable, stats in summary.by_variable.items():
        variables.append(variable)
        counts.append(stats.count)
        mean_abs_diff.append(stats.mean_abs_diff)
        max_abs_diff.append(stats.max_abs_diff)
        mean_smape.append(stats.mean_smape)
        max_smape.append(stats.max_smape)
        within_tol_rate.append(stats.within_tol_rate)
        correlation.append(stats.correlation)

    frame = pl.DataFrame(
        {
            "variable": variables,
            "count": counts,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "mean_smape": mean_smape,
            "max_smape": max_smape,
            "within_tol_rate": within_tol_rate,
            "correlation": correlation,
        },
        schema=SUMMARY_SCHEMA,
    )

    return _conform_summary(frame)


def top_divergences_from_results(
    results: Sequence[ResultComparison],
    n: int = 20,
) -> list[dict[str, object]]:
    """Return the ``n`` comparisons with the largest ``abs_diff`` as dicts.

    Sorted by descending ``abs_diff``, with ties broken by ``entity_name`` then
    ``variable`` so the order is deterministic. Each row is projected into an
    explicit dict literal (``ResultComparison`` is frozen, so attributes are read
    directly) carrying only the divergence-relevant fields.

    Args:
        results: The result comparisons to rank; consumed verbatim.
        n: The maximum number of rows to return.

    Returns:
        A list of at most ``n`` dicts, each with keys ``entity_type``,
        ``entity_name``, ``cobre_id``, ``stage``, ``variable``, ``newave_value``,
        ``cobre_value``, ``abs_diff``, ``rel_diff``. Empty ``results`` yields
        ``[]``.
    """
    if not results:
        return []

    ordered = sorted(
        results,
        key=lambda r: (-r.abs_diff, r.entity_name, r.variable),
    )

    return [
        {
            "entity_type": r.entity_type,
            "entity_name": r.entity_name,
            "cobre_id": r.cobre_id,
            "stage": r.stage,
            "variable": r.variable,
            "newave_value": r.newave_value,
            "cobre_value": r.cobre_value,
            "abs_diff": r.abs_diff,
            "rel_diff": r.rel_diff,
        }
        for r in ordered[:n]
    ]


def results_footer_counts(
    results: Sequence[ResultComparison],
) -> dict[str, object]:
    """Compute the results-summary footer counts as JSON-native metadata.

    Mirrors the footer of
    :func:`cobre_bridge.comparators.report.print_results_summary`, which prints
    ``summary.total`` and the per-entity-type comparison counts. The counts are
    derived here inline (one increment per comparison row) so the console footer
    can be rendered from the dataset metadata without reaching back into
    ``results.py`` and without recomputing any statistic.

    Args:
        results: The result comparisons; consumed verbatim (one count each).

    Returns:
        A JSON-native dict with keys ``"total"`` (the comparison count) and
        ``"by_entity_type"`` (a ``dict[str, int]`` mapping ``entity_type`` to the
        number of comparisons of that type).
    """
    by_entity_type: dict[str, int] = {}
    for r in results:
        by_entity_type[r.entity_type] = by_entity_type.get(r.entity_type, 0) + 1

    return {
        "total": len(results),
        "by_entity_type": by_entity_type,
    }


def build_results_dataset(
    results: Sequence[ResultComparison],
    pct: PercentileData,
    tolerance: float,
) -> ComparisonDataset:
    """Assemble the full canonical dataset for the results subcommand.

    Combines the tidy frame from :func:`tidy_results_dataset`, the summary frame
    from :func:`summary_frame_from_results`, and a metadata side-table carrying
    the top divergences plus the non-tidy artifacts (cost dicts and entity-name
    dicts from ``pct``) verbatim. The raw ``results`` list and the per-tab render
    inputs drained from ``pct`` (overview through constraints/performance) are
    stored in-memory under their named keys but are excluded from the serialized
    artifact via
    :data:`cobre_bridge.comparators.dataset.RENDER_ONLY_METADATA_KEYS`.
    Validates before returning.

    Args:
        results: The result comparisons (``newave``/``cobre`` rows + summary).
        pct: The percentile data (``p10``/``p50``/``p90`` rows + carry-over).
        tolerance: Relative tolerance forwarded to the summary builder.

    Returns:
        A validated :class:`ComparisonDataset` whose ``metadata`` holds
        ``top_divergences``, ``footer_counts`` (the ``total`` /
        ``by_entity_type`` needed to render the console footer byte-identically),
        ``nw_costs``, ``cobre_costs``, ``nw_bus_names`` and ``nw_hydro_names``.

    Raises:
        SchemaError: If the assembled dataset fails :meth:`validate`.
    """
    tidy = tidy_results_dataset(results, pct)
    summary = summary_frame_from_results(results, tolerance)
    metadata: dict[str, object] = {
        "top_divergences": top_divergences_from_results(results),
        "footer_counts": results_footer_counts(results),
        # Raw comparison rows, threaded in-memory for the chart functions that
        # still take ``list[ResultComparison]`` directly. Render-only: excluded
        # from the serialized artifact (see RENDER_ONLY_METADATA_KEYS).
        "results": list(results),
        "nw_costs": pct.nw_costs,
        "cobre_costs": pct.cobre_costs,
        "nw_bus_names": pct.nw_bus_names,
        "nw_hydro_names": pct.nw_hydro_names,
        # --- ticket-013: Overview/System/Energy-Balance/Network tab inputs ---
        # The four migrated tabs read these named keys (via report_builder's
        # typed metadata accessors) instead of the monolithic ``pct`` object.
        # Stored as the live objects: frames as pl/pd DataFrames, the dict/list/
        # int carry-overs as-is. The seven un-migrated tabs still read ``pct``.
        "nw_sin": pct.nw_sin,
        "cobre_stage_costs": pct.cobre_stage_costs,
        "nw_offset": pct.nw_offset,
        "nw_convergence": pct.nw_convergence,
        "cobre_convergence": pct.cobre_convergence,
        "bus": pct.bus,
        "nw_market": pct.nw_market,
        "bus_aggregates": pct.bus_aggregates,
        "cobre_bus_meta": pct.cobre_bus_meta,
        "nw_net_load": pct.nw_net_load,
        "cobre_hydro_means": pct.cobre_hydro_means,
        "hydro": pct.hydro,
        "line": pct.line,
        "line_bounds": pct.line_bounds,
        "line_meta": pct.line_meta,
        # --- ticket-014: Hydro Operation / Hydro Details tab inputs ---
        # The two hydro tabs read these named keys (via report_builder's typed
        # metadata accessors). ``cobre_hydro_meta`` is a dict[int, dict];
        # ``nw_hydro_slacks`` and ``cobre_hydro_per_stage_bounds`` are frames.
        # The remaining un-migrated tabs still read ``pct``.
        "cobre_hydro_meta": pct.cobre_hydro_meta,
        "nw_hydro_slacks": pct.nw_hydro_slacks,
        "cobre_hydro_per_stage_bounds": pct.cobre_hydro_per_stage_bounds,
        # --- ticket-021: Thermal Operation / Thermal Details / Productivity ---
        # The thermal tabs read ``thermal``; the Productivity tab reads
        # ``productivity_detail``. Both are live ``pl.DataFrame`` objects.
        "thermal": pct.thermal,
        "productivity_detail": pct.productivity_detail,
        # Per-(plant, stage) realized productivity, derived from the result rows
        # here in the analyze layer so the chart consumes a frame, not raw rows.
        "productivity_per_stage": productivity_per_stage_frame(results),
        # --- Production-function (FPHA) comparison --- The Productivity tab's
        # FPHA section reads these. ``fpha_metrics`` is the per-(plant, stage)
        # fidelity conclusion; ``fpha_surface``/``fpha_spill`` are the dense
        # render substrate. All three are ``None`` for constant-productivity
        # cases. Render-only (excluded from the serialized artifact).
        "fpha_metrics": pct.fpha_metrics,
        "fpha_surface": pct.fpha_surface,
        "fpha_spill": pct.fpha_spill,
        # --- ticket-022: Constraints / Performance tab inputs ---
        # The final two tabs read these named keys (via report_builder's typed
        # metadata accessors). ``gc_constraints`` is a ``list[dict]``;
        # ``nw_max_stage`` is ``int | None``; ``cobre_training_seconds`` is a
        # ``float``; ``nw_tim_stages`` is a ``dict[str, float]``; the rest are
        # frames. All are render-only carry-overs (excluded from the artifact).
        "gc_constraints": pct.gc_constraints,
        "gc_bounds": pct.gc_bounds,
        "gc_lhs_newave": pct.gc_lhs_newave,
        "gc_lhs_cobre": pct.gc_lhs_cobre,
        "nw_max_stage": pct.nw_max_stage,
        "nw_tim_iterations": pct.nw_tim_iterations,
        "nw_tim_stages": pct.nw_tim_stages,
        "cobre_training_seconds": pct.cobre_training_seconds,
        "cobre_iteration_timing": pct.cobre_iteration_timing,
    }

    dataset = ComparisonDataset(tidy=tidy, summary=summary, metadata=metadata)
    dataset.validate()
    return dataset


def tidy_from_bounds(results: Sequence[BoundComparison]) -> pl.DataFrame:
    """Map ``BoundComparison`` rows into tidy ``newave``/``cobre`` value rows.

    Mirrors :func:`tidy_from_results` for the bounds engine. Each input row
    yields exactly two tidy rows: one ``source="newave"`` with
    ``value = r.newave_value`` and one ``source="cobre"`` with
    ``value = r.cobre_value``. Both carry ``entity_id = r.cobre_id``,
    ``entity_type = r.entity_type``, ``entity_name = r.entity_name``,
    ``stage = r.stage``, ``variable = r.variable``, and the sentinels
    ``bus = -1`` / ``block = -1``.

    Args:
        results: The bound comparisons to map; consumed verbatim.

    Returns:
        A frame conforming exactly (columns, order, dtypes) to
        :data:`TIDY_SCHEMA`, with ``2 * len(results)`` rows. Empty ``results``
        yields a 0-row frame.
    """
    return _tidy_newave_cobre_pair(results)


def summary_frame_from_bounds(results: Sequence[BoundComparison]) -> pl.DataFrame:
    """Project the bounds comparisons into a per-variable ``SUMMARY_SCHEMA`` frame.

    Bounds use absolute tolerance, so the shared schema is filled with bounds
    semantics: ``count`` is the number of comparisons for the variable;
    ``mean_abs_diff`` / ``max_abs_diff`` are the mean / max of ``abs(r.diff)``;
    ``within_tol_rate`` is the fraction of rows whose ``match`` is ``True``;
    ``mean_smape`` / ``max_smape`` are computed via the shared
    :func:`cobre_bridge.comparators.results.smape` over
    ``(r.newave_value, r.cobre_value)``; ``correlation`` is always null.

    Args:
        results: The bound comparisons to summarize; consumed verbatim.

    Returns:
        A frame conforming exactly (columns, order, dtypes) to
        :data:`SUMMARY_SCHEMA`, one row per variable. Empty ``results`` yields a
        0-row frame.
    """
    if not results:
        return pl.DataFrame(schema=SUMMARY_SCHEMA)

    groups: dict[str, list[BoundComparison]] = {}
    for r in results:
        groups.setdefault(r.variable, []).append(r)

    variables: list[str] = []
    counts: list[int] = []
    mean_abs_diff: list[float] = []
    max_abs_diff: list[float] = []
    mean_smape: list[float] = []
    max_smape: list[float] = []
    within_tol_rate: list[float] = []
    correlation: list[float | None] = []

    for variable, group in groups.items():
        abs_diffs = [abs(r.diff) for r in group]
        smapes = [smape(r.newave_value, r.cobre_value) for r in group]
        n_match = sum(1 for r in group if r.match)

        variables.append(variable)
        counts.append(len(group))
        mean_abs_diff.append(sum(abs_diffs) / len(group))
        max_abs_diff.append(max(abs_diffs))
        mean_smape.append(sum(smapes) / len(group))
        max_smape.append(max(smapes))
        within_tol_rate.append(n_match / len(group))
        correlation.append(None)

    frame = pl.DataFrame(
        {
            "variable": variables,
            "count": counts,
            "mean_abs_diff": mean_abs_diff,
            "max_abs_diff": max_abs_diff,
            "mean_smape": mean_smape,
            "max_smape": max_smape,
            "within_tol_rate": within_tol_rate,
            "correlation": correlation,
        },
        schema=SUMMARY_SCHEMA,
    )

    return _conform_summary(frame)


def top_divergences_from_bounds(
    results: Sequence[BoundComparison],
    n: int = 20,
) -> list[dict[str, object]]:
    """Return the ``n`` largest-``abs(diff)`` mismatched comparisons as dicts.

    Only rows whose ``match`` is ``False`` are eligible. Sorted by descending
    ``abs(diff)``, with ties broken by ``entity_name`` then ``variable`` so the
    order is deterministic. Each row is projected into an explicit dict literal
    (``BoundComparison`` is frozen, so attributes are read directly).

    Args:
        results: The bound comparisons to rank; consumed verbatim.
        n: The maximum number of rows to return.

    Returns:
        A list of at most ``n`` dicts, each with keys ``entity_type``,
        ``entity_name``, ``cobre_id``, ``stage``, ``variable``, ``newave_value``,
        ``cobre_value``, ``diff``, ``match``. Empty/all-matching ``results``
        yields ``[]``.
    """
    mismatches = [r for r in results if not r.match]
    if not mismatches:
        return []

    ordered = sorted(
        mismatches,
        key=lambda r: (-abs(r.diff), r.entity_name, r.variable),
    )

    return [
        {
            "entity_type": r.entity_type,
            "entity_name": r.entity_name,
            "cobre_id": r.cobre_id,
            "stage": r.stage,
            "variable": r.variable,
            "newave_value": r.newave_value,
            "cobre_value": r.cobre_value,
            "diff": r.diff,
            "match": r.match,
        }
        for r in ordered[:n]
    ]


def bounds_summary_counts(
    results: Sequence[BoundComparison],
) -> dict[str, object]:
    """Compute the bounds-summary match/mismatch counts as JSON-native metadata.

    Mirrors the counting logic of
    :func:`cobre_bridge.comparators.report.build_summary` (a row whose ``match``
    is ``True`` increments the match count for its entity type and variable, else
    the mismatch count) so the bounds console tables can be rendered from the
    dataset metadata byte-identically — without importing ``report.py`` here and
    without recomputing any diff. The integer counts are exact (not derived from
    a fractional rate), so the printed ``Compared``/``Match``/``Mismatch``/``Rate``
    columns reproduce the legacy output exactly.

    Args:
        results: The bound comparisons; consumed verbatim (one count each).

    Returns:
        A JSON-native dict with keys ``"total"``, ``"matches"``, ``"mismatches"``
        (overall ints), and ``"by_entity_type"`` / ``"by_variable"`` — each a
        ``dict[str, list[int]]`` mapping the key to a ``[match_count,
        mismatch_count]`` pair. The mappings carry every entity type / variable
        seen, matching ``build_summary``'s ``sorted`` union of matched and
        mismatched keys.
    """
    type_counts: dict[str, list[int]] = {}
    var_counts: dict[str, list[int]] = {}
    matches = 0
    mismatches = 0

    for r in results:
        type_pair = type_counts.setdefault(r.entity_type, [0, 0])
        var_pair = var_counts.setdefault(r.variable, [0, 0])
        if r.match:
            matches += 1
            type_pair[0] += 1
            var_pair[0] += 1
        else:
            mismatches += 1
            type_pair[1] += 1
            var_pair[1] += 1

    return {
        "total": len(results),
        "matches": matches,
        "mismatches": mismatches,
        "by_entity_type": type_counts,
        "by_variable": var_counts,
    }


def bounds_mismatch_listing(
    results: Sequence[BoundComparison],
    max_rows: int = 50,
) -> dict[str, object]:
    """Project the bounds mismatches into the console-listing metadata.

    Mirrors :func:`cobre_bridge.comparators.report.print_mismatches` EXACTLY so
    the listing can be rendered from the dataset byte-identically: it filters to
    ``not r.match``, sorts by **raw** ``diff`` descending (``reverse=True``, no
    abs, no tie-break — the legacy ordering verbatim), and keeps the first
    ``max_rows`` rows. The full mismatch count (before the cap) is carried so the
    printer can reproduce the ``(of N total)`` header and the ``... and M more``
    footer.

    This is intentionally distinct from :func:`top_divergences_from_bounds`
    (which is abs-sorted, tie-broken, capped at 20, and feeds the export
    manifest) — the console listing has different ordering and cap semantics.

    Args:
        results: The bound comparisons; consumed verbatim.
        max_rows: The number of rows to retain (the CLI fixes this at 50).

    Returns:
        A JSON-native dict with keys ``"total"`` (the full mismatch count) and
        ``"rows"`` (at most ``max_rows`` dicts, each carrying ``entity_type``,
        ``entity_name``, ``newave_code``, ``cobre_id``, ``stage``, ``variable``,
        ``newave_value``, ``cobre_value`` and ``diff``).
    """
    mismatches = [r for r in results if not r.match]
    mismatches.sort(key=lambda r: r.diff, reverse=True)
    shown = mismatches[:max_rows]

    rows: list[dict[str, object]] = [
        {
            "entity_type": r.entity_type,
            "entity_name": r.entity_name,
            "newave_code": r.newave_code,
            "cobre_id": r.cobre_id,
            "stage": r.stage,
            "variable": r.variable,
            "newave_value": r.newave_value,
            "cobre_value": r.cobre_value,
            "diff": r.diff,
        }
        for r in shown
    ]

    return {
        "total": len(mismatches),
        "rows": rows,
    }


def build_bounds_dataset(results: Sequence[BoundComparison]) -> ComparisonDataset:
    """Assemble the canonical dataset for the bounds subcommand.

    Combines the tidy frame from :func:`tidy_from_bounds`, the per-variable
    summary from :func:`summary_frame_from_bounds`, and a metadata side-table
    carrying the top mismatched divergences. Validates before returning.

    Args:
        results: The bound comparisons (``newave``/``cobre`` rows + summary).

    Returns:
        A validated :class:`ComparisonDataset` whose ``metadata`` holds
        ``top_divergences`` (at most 20 mismatched rows, largest ``abs(diff)``
        first), ``summary_counts`` (the exact match/mismatch integer counts
        needed to render the console summary tables byte-identically) and
        ``mismatch_listing`` (the raw-diff-sorted, 50-capped mismatch rows plus
        the full mismatch count for the console listing).

    Raises:
        SchemaError: If the assembled dataset fails :meth:`validate`.
    """
    tidy = tidy_from_bounds(results)
    summary = summary_frame_from_bounds(results)
    metadata: dict[str, object] = {
        "top_divergences": top_divergences_from_bounds(results),
        "summary_counts": bounds_summary_counts(results),
        "mismatch_listing": bounds_mismatch_listing(results),
    }

    dataset = ComparisonDataset(tidy=tidy, summary=summary, metadata=metadata)
    dataset.validate()
    return dataset


def _conform_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """Select and cast ``frame`` to exactly the :data:`SUMMARY_SCHEMA` contract.

    Mirrors :func:`_conform` for the summary frame: guarantees column presence,
    order, and dtypes (notably ``correlation`` as nullable ``Float64``) so the
    result passes :meth:`ComparisonDataset.validate`.

    Args:
        frame: A frame carrying (at least) the summary columns.

    Returns:
        The frame with exactly the :data:`SUMMARY_SCHEMA` columns in order and
        the schema dtypes.
    """
    return frame.select(list(SUMMARY_SCHEMA)).cast(
        {col: dtype() for col, dtype in SUMMARY_SCHEMA.items()}
    )


def aggregate_percentile_band(
    pct_df: pl.DataFrame | None,
    variable: str,
    stages: list[int],
    entity_ids: set[int] | None = None,
) -> tuple[list[float], list[float]]:
    """Sum ``{variable}_p10``/``{variable}_p90`` per stage into aligned lists.

    Pure numeric core of ``charts._aggregate_percentile_traces``: it produces the
    two per-stage envelope lists (p10, p90) the chart draws as a band, but emits
    no traces. Percentiles are summed across (optionally filtered) entities per
    ``stage_id``, then read back aligned to the explicit ``stages`` axis with the
    legacy default-to-zero semantics (a stage absent from the aggregate yields
    ``0.0``, never a ``KeyError``).

    Args:
        pct_df: A per-entity percentile frame with ``entity_id``, ``stage_id``
            and ``{var}_p10``/``{var}_p90`` columns, or ``None``.
        variable: The variable stem whose ``_p10``/``_p90`` columns are summed.
        stages: The explicit, caller-sorted stage axis the lists align to.
        entity_ids: Optional entity filter; when given, only rows whose
            ``entity_id`` is in this set contribute to the sums.

    Returns:
        A ``(p10, p90)`` tuple of per-stage float lists, each ``len(stages)``
        long. Returns ``([], [])`` when ``pct_df`` is ``None``/empty or either
        ``{variable}_p10``/``{variable}_p90`` column is absent.
    """
    if pct_df is None or pct_df.is_empty():
        return [], []

    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    if p10_col not in pct_df.columns or p90_col not in pct_df.columns:
        return [], []

    filtered = pct_df
    if entity_ids is not None:
        filtered = pct_df.filter(pl.col("entity_id").is_in(list(entity_ids)))

    agg = filtered.group_by("stage_id").agg(
        pl.col(p10_col).sum(), pl.col(p90_col).sum()
    )
    lookup = {int(r["stage_id"]): r for r in agg.iter_rows(named=True)}

    p10 = [float(lookup.get(s, {}).get(p10_col, 0)) for s in stages]
    p90 = [float(lookup.get(s, {}).get(p90_col, 0)) for s in stages]
    return p10, p90


def per_stage_sum_from_results(
    results: Sequence[ResultComparison],
    entity_type: str,
    variable: str,
) -> tuple[dict[int, float], dict[int, float], set[int]]:
    """Sum the source model/Cobre values per stage for one entity type (and variable).

    Pure numeric core of the inline per-stage accumulation loop repeated in
    ``charts.system_comparison_chart`` / ``hydro_aggregate_chart`` /
    ``thermal_generation_chart``. Rows are filtered to ``entity_type``; when
    ``variable`` is non-empty they must also match ``variable`` (the
    system/hydro case), and when ``variable == ""`` every variable of that
    entity type is included (the ``thermal_generation_chart`` case, which keys
    only on entity type). Each surviving row adds ``newave_value`` /
    ``cobre_value`` into its stage bucket (both starting at ``0.0``).

    Args:
        results: The comparison rows; consumed verbatim (read-only).
        entity_type: The ``entity_type`` to keep (e.g. ``"hydro"``,
            ``"thermal"``, ``"bus"``).
        variable: The ``variable`` to keep, or ``""`` to match every variable
            of the entity type.

    Returns:
        ``(nw_by_stage, cb_by_stage, matched_ids)`` where the first two map
        ``stage`` to the summed ``newave_value`` / ``cobre_value`` and
        ``matched_ids`` is the set of ``cobre_id`` over the filtered rows.
        Empty when no row matches.
    """
    filtered = [
        r
        for r in results
        if r.entity_type == entity_type and (variable == "" or r.variable == variable)
    ]

    nw_by_stage: dict[int, float] = {}
    cb_by_stage: dict[int, float] = {}
    for r in filtered:
        nw_by_stage[r.stage] = nw_by_stage.get(r.stage, 0.0) + r.newave_value
        cb_by_stage[r.stage] = cb_by_stage.get(r.stage, 0.0) + r.cobre_value

    matched_ids = {r.cobre_id for r in filtered}
    return nw_by_stage, cb_by_stage, matched_ids


def per_stage_sum_from_frame(
    df: pl.DataFrame | None,
    variable: str,
    matched_ids: set[int] | None = None,
) -> dict[int, float]:
    """Sum ``variable`` across (matched) entities per ``stage_id``.

    Pure numeric core of ``charts._hydro_per_stage_sum``: it collapses a
    per-``(entity_id, stage_id)`` frame into a per-stage SIN total. Returns an
    empty dict when the frame is missing/empty or the column is absent — never
    raises.

    Args:
        df: A per-entity frame with ``entity_id``, ``stage_id`` and the
            ``variable`` column, or ``None``.
        variable: The column to sum.
        matched_ids: Optional entity filter; when given, only rows whose
            ``entity_id`` is in this set contribute.

    Returns:
        A ``{stage_id: sum}`` dict (sorted by ``stage_id``). Empty when ``df``
        is ``None``/empty or ``variable`` is not a column of ``df``.
    """
    if df is None or df.is_empty() or variable not in df.columns:
        return {}
    filtered = df
    if matched_ids is not None:
        filtered = df.filter(pl.col("entity_id").is_in(list(matched_ids)))
    agg = (
        filtered.group_by("stage_id")
        .agg(pl.col(variable).sum().alias("v"))
        .sort("stage_id")
    )
    return {int(r["stage_id"]): float(r["v"]) for r in agg.iter_rows(named=True)}


#: Fictitious bus names skipped by the per-bus roll-up (mirrors charts.py).
_FICTITIOUS_BUSES: frozenset[str] = frozenset({"NOFICT1", "NOFICT2", "NOFICT3"})


class _BusNameLookupCache:
    """Single-slot memo for :func:`_bus_name_lookups`, keyed by object identity.

    ``report_builder.py`` threads the SAME ``hydro_meta``/``bus_meta`` pair
    into up to 11 per-bus chart calls per comparison run (one per chart
    variable: 6 ``hydro_per_bus_chart`` + 5 ``hydro_slack_per_bus_chart``
    calls); each resolves every plant's bus via :func:`_bus_name_lookups`,
    which -- for an ambiguous or empty-map condition -- emits a
    :class:`~cobre_bridge.diagnostics.Diagnostic`. Compare has no diagnostics
    de-dup sink, so without this cache the identical warning logs ~11x for
    one condition.

    The cache key is the actual ``(hydro_meta, bus_meta)`` object pair, not
    just their ``id()``: holding a strong reference keeps the cached objects
    alive, so a stale ``id()`` can never alias a different, later-allocated
    dict -- Python cannot reuse the memory address of an object this cache
    still references. A later call with a genuinely different pair (a new
    comparison run) simply evicts the single slot and recomputes+re-emits,
    so the diagnostic still fires once per run, not once ever.
    """

    def __init__(self) -> None:
        self.meta: (
            tuple[dict[int, dict[str, object]], dict[int, dict[str, object]]] | None
        ) = None
        self.result: tuple[dict[int, str], dict[int, int]] | None = None


_lookup_cache = _BusNameLookupCache()


def _bus_name_lookups(
    hydro_meta: dict[int, dict[str, object]],
    bus_meta: dict[int, dict[str, object]],
) -> tuple[dict[int, str], dict[int, int]]:
    """Build the ``bus_id -> name`` and ``plant -> bus_id`` lookups.

    Shared helper for the per-bus roll-up. ``bus_id_to_name`` maps each bus id
    to ``bus_meta[bid]["name"]`` (fallback ``str(bid)``).

    ``hydro_to_bus`` maps each plant id to its single owning bus, sourced from
    ``hydro_meta[hid]["bus_ids"]`` -- the distinct-``(hydro_id, bus_id)``-pair
    label the ``simulation/hydro_bus_generation`` partition carries per plant
    (see ``cobre_readers.read_cobre_hydro_bus_labels``). ``hydro_meta`` itself
    (``cobre_readers.read_cobre_hydro_metadata``) carries no bus information --
    it is plant physics only (decision B1) -- ``"bus_ids"`` is merged onto it
    by the results-comparison orchestrator before it reaches here.

    A plant with no recorded bus (``"bus_ids"`` absent or empty) is skipped,
    matching the legacy ``bus_id is None`` skip. A plant recorded at *more
    than one* bus (only possible once a case has genuine multi-bus hydros --
    epic 08) is **not** collapsed onto an arbitrary one of them and **not**
    added to every one of them either: either choice would misreport the
    per-bus roll-up (silently dropping one bus's share, or double-counting
    the plant's whole value at each bus it touches). It is instead excluded
    from ``hydro_to_bus`` and a :class:`~cobre_bridge.diagnostics.Diagnostic`
    is raised naming the plant (id and, when available, its
    ``hydro_meta[hid]["name"]``), so the ambiguity surfaces instead of
    quietly producing wrong numbers. If that leaves ``hydro_to_bus`` empty
    despite a non-empty ``hydro_meta``, a second diagnostic flags the map
    itself as unusable -- the failure mode this helper exists to make loud.

    Repeated calls with the identical ``(hydro_meta, bus_meta)`` object pair
    hit :data:`_lookup_cache` and do not re-emit (see that class's docstring).

    Args:
        hydro_meta: Per-plant metadata; read-only. The bus label, when
            present, lives under the ``"bus_ids"`` key as a set of bus ids.
        bus_meta: Per-bus metadata carrying ``name``; read-only.

    Returns:
        ``(bus_id_to_name, hydro_to_bus)``.
    """
    cached_meta = _lookup_cache.meta
    if (
        cached_meta is not None
        and cached_meta[0] is hydro_meta
        and cached_meta[1] is bus_meta
        and _lookup_cache.result is not None
    ):
        return _lookup_cache.result

    bus_id_to_name: dict[int, str] = {
        bid: cast("str", meta.get("name", str(bid))) for bid, meta in bus_meta.items()
    }

    hydro_to_bus: dict[int, int] = {}
    for hid, meta in hydro_meta.items():
        raw_bus_ids = meta.get("bus_ids")
        if not raw_bus_ids:
            continue
        bus_ids = {int(b) for b in cast("Iterable[int]", raw_bus_ids)}
        if len(bus_ids) > 1:
            name = cast("str", meta.get("name", str(hid)))
            emit(
                Diagnostic(
                    code="hydro-multi-bus-ambiguous",
                    severity=Severity.WARNING,
                    category="Compare data",
                    title="Hydro plant spans multiple buses",
                    summary=(
                        f"Hydro {hid} ({name}) is recorded at buses "
                        f"{sorted(bus_ids)} in the hydro_bus_generation "
                        "partition; a single plant-level value cannot be "
                        "attributed to one bus without collapsing or "
                        "double-counting, so it is excluded from the "
                        "per-bus hydro charts."
                    ),
                    notes=[f"hydro_id: {hid}", f"bus_ids: {sorted(bus_ids)}"],
                ),
                logger=_LOG,
            )
            continue
        hydro_to_bus[hid] = next(iter(bus_ids))

    if hydro_meta and not hydro_to_bus:
        emit(
            Diagnostic(
                code="hydro-to-bus-map-empty",
                severity=Severity.WARNING,
                category="Compare data",
                title="No hydro plant could be mapped to a bus",
                summary=(
                    "hydro_meta has entries but none carry a resolvable "
                    'single-bus "bus_ids" label; the per-bus hydro charts '
                    "will render empty. Check that the hydro_bus_generation "
                    "partition was read and merged for this output."
                ),
            ),
            logger=_LOG,
        )

    _lookup_cache.meta = (hydro_meta, bus_meta)
    _lookup_cache.result = (bus_id_to_name, hydro_to_bus)
    return bus_id_to_name, hydro_to_bus


def per_bus_sums_from_results(
    results: Sequence[ResultComparison],
    variable: str,
    hydro_meta: dict[int, dict[str, object]],
    bus_meta: dict[int, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Roll hydro ``ResultComparison`` rows up to per-(bus, stage) sums.

    Pure numeric core of the per-bus accumulation in
    ``charts.hydro_per_bus_chart`` (``charts.py:904-939``). Rows are filtered to
    ``entity_type == "hydro"`` and ``variable``; each plant is mapped to its
    owning bus via ``hydro_meta[cobre_id]["bus_ids"]`` (see
    :func:`_bus_name_lookups` -- plants with no resolvable single bus,
    including genuinely multi-bus plants, are skipped and diagnosed there),
    the bus name is resolved via ``bus_meta[bus_id]["name"]`` (fallback
    ``str(bus_id)``) and upper-cased, and the fictitious buses ``NOFICT1/2/3``
    are dropped. Surviving rows accumulate ``newave_value`` / ``cobre_value``
    into their ``(bus, stage)`` bucket and add ``cobre_id`` to the bus's id
    set.

    Args:
        results: The comparison rows; consumed verbatim (read-only).
        variable: The ``variable`` to keep.
        hydro_meta: Per-plant metadata; read-only. The bus label, when
            present, lives under the ``"bus_ids"`` key (see
            :func:`_bus_name_lookups`).
        bus_meta: Per-bus metadata carrying ``name``; read-only.

    Returns:
        A dict keyed by upper-cased ``bus_name``; each value is a dict with
        ``"nw"`` (``dict[int, float]``), ``"cb"`` (``dict[int, float]``) and
        ``"ids"`` (``set[int]``). Empty when no row maps to a non-fictitious bus.
    """
    bus_id_to_name, hydro_to_bus = _bus_name_lookups(hydro_meta, bus_meta)

    per_bus_nw: dict[str, dict[int, float]] = {}
    per_bus_cb: dict[str, dict[int, float]] = {}
    per_bus_ids: dict[str, set[int]] = {}
    for r in results:
        if r.entity_type != "hydro" or r.variable != variable:
            continue
        bus_id = hydro_to_bus.get(r.cobre_id)
        if bus_id is None:
            continue
        bus_name = bus_id_to_name.get(bus_id, str(bus_id)).upper()
        if bus_name in _FICTITIOUS_BUSES:
            continue
        per_bus_nw.setdefault(bus_name, {})
        per_bus_cb.setdefault(bus_name, {})
        per_bus_nw[bus_name][r.stage] = (
            per_bus_nw[bus_name].get(r.stage, 0.0) + r.newave_value
        )
        per_bus_cb[bus_name][r.stage] = (
            per_bus_cb[bus_name].get(r.stage, 0.0) + r.cobre_value
        )
        per_bus_ids.setdefault(bus_name, set()).add(r.cobre_id)

    return {
        bus_name: {
            "nw": per_bus_nw[bus_name],
            "cb": per_bus_cb[bus_name],
            "ids": per_bus_ids.get(bus_name, set()),
        }
        for bus_name in per_bus_nw
    }


def per_bus_sums_from_frame(
    df: pl.DataFrame | None,
    variable: str,
    matched_ids: set[int] | None,
    hydro_meta: dict[int, dict[str, object]],
    bus_meta: dict[int, dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Roll a per-(entity_id, stage_id) frame up to per-(bus, stage) sums.

    Pure numeric core of the ``_per_bus_from_frame`` closure in
    ``charts.hydro_slack_per_bus_chart`` (``charts.py:1207-1233``). Used for the
    slack variables that are sourced from per-entity frames rather than
    ``ResultComparison`` rows. Rows are optionally filtered to ``matched_ids``,
    each entity is mapped to its owning bus (same resolution / fictitious-skip
    as :func:`per_bus_sums_from_results`), and the ``variable`` value is summed
    per ``(bus, stage)``. Rows whose ``variable`` value is ``None`` contribute
    nothing.

    Args:
        df: A per-entity frame with ``entity_id``, ``stage_id`` and the
            ``variable`` column, or ``None``.
        variable: The column to sum.
        matched_ids: Optional entity filter; when given, only rows whose
            ``entity_id`` is in this set contribute.
        hydro_meta: Per-plant metadata; read-only. The bus label, when
            present, lives under the ``"bus_ids"`` key (see
            :func:`_bus_name_lookups`).
        bus_meta: Per-bus metadata carrying ``name``; read-only.

    Returns:
        A dict keyed by upper-cased ``bus_name``; each value is a dict with
        ``"sum"`` (``dict[int, float]`` per stage) and ``"ids"`` (``set[int]``).
        Empty when ``df`` is ``None``/empty, ``variable`` is absent, or no row
        maps to a non-fictitious bus.
    """
    if df is None or df.is_empty() or variable not in df.columns:
        return {}

    bus_id_to_name, hydro_to_bus = _bus_name_lookups(hydro_meta, bus_meta)

    frame = df
    if matched_ids is not None:
        frame = df.filter(pl.col("entity_id").is_in(list(matched_ids)))

    per_bus_sum: dict[str, dict[int, float]] = {}
    per_bus_ids: dict[str, set[int]] = {}
    for row in frame.iter_rows(named=True):
        eid = int(row["entity_id"])
        bus_id = hydro_to_bus.get(eid)
        if bus_id is None:
            continue
        bus_name = bus_id_to_name.get(bus_id, str(bus_id)).upper()
        if bus_name in _FICTITIOUS_BUSES:
            continue
        sid = int(row["stage_id"])
        val = row.get(variable)
        if val is None:
            continue
        per_bus_sum.setdefault(bus_name, {})
        per_bus_sum[bus_name][sid] = per_bus_sum[bus_name].get(sid, 0.0) + float(val)
        per_bus_ids.setdefault(bus_name, set()).add(eid)

    return {
        bus_name: {"sum": stage_map, "ids": per_bus_ids.get(bus_name, set())}
        for bus_name, stage_map in per_bus_sum.items()
    }


def per_bus_band_from_pct(
    pct_df: pl.DataFrame | None,
    variable: str,
    per_bus_ids: dict[str, set[int]],
) -> dict[str, dict[int, tuple[float, float]]]:
    """Sum ``{variable}_p10``/``{variable}_p90`` per stage for each bus's plants.

    Pure numeric core of the per-bus percentile-band roll-up in
    ``charts.hydro_per_bus_chart`` (``charts.py:948-966``) and the parallel block
    in ``charts.hydro_slack_per_bus_chart`` (``charts.py:1253-1264``). For each
    ``bus_name -> ids`` it sums the percentile columns across ``ids`` per
    ``stage_id`` and returns the per-stage ``(p10, p90)`` tuples. Buses with an
    empty id set contribute no entry (mirroring the slack-chart ``if not ids``
    skip; the result-sourced caller never passes an empty set).

    Args:
        pct_df: A per-entity percentile frame with ``entity_id``, ``stage_id``
            and ``{var}_p10``/``{var}_p90`` columns, or ``None``.
        variable: The variable stem whose ``_p10``/``_p90`` columns are summed.
        per_bus_ids: Mapping of ``bus_name`` to the set of plant ``entity_id``
            owned by that bus.

    Returns:
        A dict keyed by ``bus_name``; each value maps ``stage_id`` to the summed
        ``(p10, p90)`` tuple. Empty when ``pct_df`` is ``None``/empty or either
        ``{variable}_p10``/``{variable}_p90`` column is absent.
    """
    if pct_df is None or pct_df.is_empty():
        return {}

    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    if not {p10_col, p90_col}.issubset(pct_df.columns):
        return {}

    per_bus_pct: dict[str, dict[int, tuple[float, float]]] = {}
    for bus_name, ids in per_bus_ids.items():
        if not ids:
            continue
        agg = (
            pct_df.filter(pl.col("entity_id").is_in(list(ids)))
            .group_by("stage_id")
            .agg(pl.col(p10_col).sum(), pl.col(p90_col).sum())
        )
        per_bus_pct[bus_name] = {
            int(r["stage_id"]): (float(r[p10_col]), float(r[p90_col]))
            for r in agg.iter_rows(named=True)
        }
    return per_bus_pct


def plant_percentile_arrays(
    pct_df: pl.DataFrame | None,
    var_stages: Sequence[tuple[str, str, list[int]]],
    plant_cobre_id: int,
) -> dict[str, list[float]]:
    """Extract one plant's per-stage ``{var}_p10``/``{var}_p90`` arrays.

    Pure numeric core of the inner extraction in
    ``charts._enrich_with_percentiles`` (``charts.py:2786-2799``). For the single
    plant ``plant_cobre_id`` it filters ``pct_df`` to that entity **once**, then
    for each ``(var_key, _, stages)`` triple whose ``{var}_p10``/``{var}_p90``
    columns are present, reads the rounded per-stage values aligned to that
    variable's own ``stages`` axis with the legacy ``round(float(... or 0), 2)``
    semantics (a stage absent from the plant's frame yields ``0.0``). Each
    variable carries its OWN stage axis, so the single per-plant filter is
    reused across every variable. Variables whose columns are absent — and every
    variable when the plant has no rows — contribute no entry. The in-place
    ``js_plants`` mutation stays in ``charts.py``; this function only computes
    the arrays.

    Args:
        pct_df: A per-entity percentile frame with ``entity_id``, ``stage_id``
            and ``{var}_p10``/``{var}_p90`` columns, or ``None``.
        var_stages: ``(var_key, label, stages)`` triples; only ``var_key`` and
            ``stages`` (the per-variable stage axis the arrays align to) are
            used.
        plant_cobre_id: The ``entity_id`` to extract.

    Returns:
        A dict mapping ``{var}_p10``/``{var}_p90`` to the rounded per-stage float
        lists for the plant. Empty when ``pct_df`` is ``None``/empty, the plant
        has no rows, or no requested variable's columns are present.
    """
    if pct_df is None or pct_df.is_empty():
        return {}

    sub = pct_df.filter(pl.col("entity_id") == plant_cobre_id).sort("stage_id")
    if sub.is_empty():
        return {}

    pct_map = {int(r["stage_id"]): r for r in sub.iter_rows(named=True)}
    arrays: dict[str, list[float]] = {}
    for var_key, _, stages in var_stages:
        p10_col = f"{var_key}_p10"
        p90_col = f"{var_key}_p90"
        if p10_col in sub.columns and p90_col in sub.columns:
            arrays[p10_col] = [
                round(float(pct_map.get(s, {}).get(p10_col, 0) or 0), 2) for s in stages
            ]
            arrays[p90_col] = [
                round(float(pct_map.get(s, {}).get(p90_col, 0) or 0), 2) for s in stages
            ]
    return arrays


def cobre_sum_and_newave_sin(
    cobre_hydro: pl.DataFrame,
    variable: str,
    nw_sin: pl.DataFrame | None,
    nw_variable: str | None,
    nw_factor: float,
    nw_offset: int,
    matched_ids: set[int] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Roll a Cobre per-hydro variable and a source-model-SIN long frame to per-stage
    totals.

    Pure numeric core of ``charts.cobre_aggregate_chart`` (``charts.py:813-846``). The
    Cobre side sums ``variable`` across (optionally ``matched_ids``-filtered) plants per
    ``stage_id`` — delegated to :func:`per_stage_sum_from_frame` so the
    grouping/sort/filter semantics stay identical. The source model side folds the long
    ``nw_sin`` frame into a per-stage total: rows are filtered to
    ``variable.strip().upper() == nw_variable``, then each surviving row adds ``value *
    nw_factor`` into bucket ``stage - nw_offset``, skipping rows whose ``stage`` or
    ``value`` is ``None``. Never raises.

    Args:
        cobre_hydro: Per-hydro Cobre means with ``entity_id``, ``stage_id`` and
            the ``variable`` column.
        variable: The column to sum on the Cobre side.
        nw_sin: Long-format the source model SIN frame with ``stage``, ``variable`` and
            ``value`` columns, or ``None``.
        nw_variable: The (already upper-cased) variable to keep in ``nw_sin``, or
            ``None`` to skip the source model side entirely.
        nw_factor: Multiplicative factor applied to each source-model value (unit
            alignment).
        nw_offset: Subtracted from each source-model ``stage`` to align with the Cobre
            ``stage_id`` axis.
        matched_ids: Optional Cobre entity filter forwarded to the Cobre sum.

    Returns:
        ``(cobre_by_stage, nw_by_stage)``. ``cobre_by_stage`` is empty when
        ``cobre_hydro`` is empty or ``variable`` is absent; ``nw_by_stage`` is
        empty when ``nw_sin``/``nw_variable`` is missing or ``nw_sin`` is empty.
    """
    if cobre_hydro.is_empty() or variable not in cobre_hydro.columns:
        return {}, {}

    cobre_by_stage = per_stage_sum_from_frame(cobre_hydro, variable, matched_ids)

    nw_by_stage: dict[int, float] = {}
    if nw_sin is not None and nw_variable is not None and not nw_sin.is_empty():
        sin_df = nw_sin.filter(
            pl.col("variable").str.strip_chars().str.to_uppercase() == nw_variable
        )
        for r in sin_df.iter_rows(named=True):
            stage_raw = r.get("stage")
            val = r.get("value")
            if stage_raw is None or val is None:
                continue
            s = int(stage_raw) - nw_offset
            nw_by_stage[s] = nw_by_stage.get(s, 0.0) + float(val) * nw_factor

    return cobre_by_stage, nw_by_stage


def bus_groups_and_pct(
    results: Sequence[ResultComparison],
    variable: str,
    pct_df: pl.DataFrame | None,
) -> tuple[dict[str, list[ResultComparison]], dict[int, dict[int, dict[str, object]]]]:
    """Group ``bus`` rows by name and build the per-entity percentile lookup.

    Pure numeric core of ``charts.system_per_bus_chart`` (``charts.py:2500-2523``).
    Rows are filtered to ``entity_type == "bus"`` and ``variable`` and bucketed by
    upper-cased ``entity_name`` (insertion order preserved). The percentile lookup
    maps ``entity_id -> {stage_id -> row}`` over ``pct_df`` when both the
    ``{var}_p10`` and ``{var}_p90`` columns are present; otherwise it is empty.
    Never raises.

    Args:
        results: The comparison rows; consumed verbatim (read-only).
        variable: The ``variable`` to keep.
        pct_df: A per-entity percentile frame with ``entity_id``, ``stage_id`` and
            ``{var}_p10``/``{var}_p90`` columns, or ``None``.

    Returns:
        ``(buses, pct_by_eid)`` where ``buses`` maps each upper-cased bus name to
        its list of rows (in input order) and ``pct_by_eid`` maps
        ``entity_id -> {stage_id -> percentile row}``. ``buses`` is empty when no
        ``bus`` row matches; ``pct_by_eid`` is empty when ``pct_df`` is
        ``None``/empty or the percentile columns are absent.
    """
    bus_data = [r for r in results if r.entity_type == "bus" and r.variable == variable]

    buses: dict[str, list[ResultComparison]] = {}
    for r in bus_data:
        buses.setdefault(r.entity_name.upper(), []).append(r)

    pct_by_eid: dict[int, dict[int, dict[str, object]]] = {}
    p10_col = f"{variable}_p10"
    p90_col = f"{variable}_p90"
    if pct_df is not None and not pct_df.is_empty():
        if p10_col in pct_df.columns and p90_col in pct_df.columns:
            for row in pct_df.iter_rows(named=True):
                eid = int(row["entity_id"])
                sid = int(row["stage_id"])
                pct_by_eid.setdefault(eid, {})[sid] = row

    return buses, pct_by_eid


def spillage_lookups(
    results: Sequence[ResultComparison],
    cobre_spill_energy: pl.DataFrame,
) -> tuple[dict[str, dict[int, float]], dict[str, dict[int, float]]]:
    """Build the source model and Cobre per-variable, per-stage spillage lookups.

    Pure numeric core of ``charts.system_spillage_energy_chart``
    (``charts.py:1565-1581``). The source model lookup is keyed by each
    ``system_spillage`` row's ``variable`` (e.g. ``VERTOT``/``VERTcont``/ ``VERTfio``)
    then ``stage`` to ``newave_value``. The Cobre lookup maps the ``cobre_spill_energy``
    frame's ``total_mw``/``reservoir_mw``/``rorov_mw`` columns per ``stage_id`` under
    the keys ``spill_energy_total_mw`` / ``spill_energy_reservoir_mw`` /
    ``spill_energy_rorov_mw``. Never raises.

    Args:
        results: The comparison rows; only ``entity_type == "system_spillage"``
            rows contribute. Consumed verbatim (read-only).
        cobre_spill_energy: A per-stage frame with ``stage_id``, ``total_mw``,
            ``reservoir_mw`` and ``rorov_mw`` columns (may be empty).

    Returns:
        ``(nw_lookup, cb_lookup)``. ``nw_lookup`` is empty when no
        ``system_spillage`` row is present; ``cb_lookup`` is empty when
        ``cobre_spill_energy`` is empty.
    """
    nw_rows = [r for r in results if r.entity_type == "system_spillage"]

    nw_lookup: dict[str, dict[int, float]] = {}
    for r in nw_rows:
        nw_lookup.setdefault(r.variable, {})[r.stage] = r.newave_value

    cb_lookup: dict[str, dict[int, float]] = {}
    if not cobre_spill_energy.is_empty():
        for row in cobre_spill_energy.iter_rows(named=True):
            sid = int(row["stage_id"])
            cb_lookup.setdefault("spill_energy_total_mw", {})[sid] = float(
                row["total_mw"]
            )
            cb_lookup.setdefault("spill_energy_reservoir_mw", {})[sid] = float(
                row["reservoir_mw"]
            )
            cb_lookup.setdefault("spill_energy_rorov_mw", {})[sid] = float(
                row["rorov_mw"]
            )

    return nw_lookup, cb_lookup


def _tidy_one_percentile_frame(
    entity_type: str,
    frame: pl.DataFrame,
) -> pl.DataFrame | None:
    """Unpivot one percentile frame into tidy rows, or ``None`` if not usable.

    Args:
        entity_type: The tidy ``entity_type`` literal for every emitted row.
        frame: A ``PercentileData`` percentile frame.

    Returns:
        A :data:`TIDY_SCHEMA`-conforming frame, or ``None`` when the frame is
        empty, lacks the ``entity_id``/``stage_id`` identifier columns, or
        carries no ``{var}_p10/_p50/_p90`` triplet column.
    """
    if frame.is_empty():
        return None

    columns = set(frame.columns)
    if _PCT_ENTITY_COL not in columns or _PCT_STAGE_COL not in columns:
        _LOG.debug(
            "percentile frame %r missing identifier columns (have %s); skipping",
            entity_type,
            sorted(columns),
        )
        return None

    pct_columns = [
        col
        for col in frame.columns
        if any(col.endswith(suffix) for suffix in _PCT_SUFFIX_TO_SOURCE)
    ]
    if not pct_columns:
        _LOG.debug(
            "percentile frame %r has no p10/p50/p90 triplet columns; skipping",
            entity_type,
        )
        return None

    long = frame.select([_PCT_ENTITY_COL, _PCT_STAGE_COL, *pct_columns]).unpivot(
        on=pct_columns,
        index=[_PCT_ENTITY_COL, _PCT_STAGE_COL],
        variable_name="pct_column",
        value_name="value",
    )

    # Split each "{var}_pNN" column name into its variable stem and source.
    source_expr = pl.lit(None, dtype=pl.Utf8)
    variable_expr = pl.lit(None, dtype=pl.Utf8)
    for suffix, source in _PCT_SUFFIX_TO_SOURCE.items():
        is_suffix = pl.col("pct_column").str.ends_with(suffix)
        source_expr = pl.when(is_suffix).then(pl.lit(source)).otherwise(source_expr)
        variable_expr = (
            pl.when(is_suffix)
            .then(
                pl.col("pct_column").str.slice(
                    0, pl.col("pct_column").str.len_chars() - len(suffix)
                )
            )
            .otherwise(variable_expr)
        )

    tidy = long.select(
        pl.lit(entity_type).alias("entity_type"),
        pl.col(_PCT_ENTITY_COL).cast(pl.Int64).alias("entity_id"),
        pl.lit("").alias("entity_name"),
        pl.lit(_SENTINEL, dtype=pl.Int64).alias("bus"),
        pl.col(_PCT_STAGE_COL).cast(pl.Int64).alias("stage"),
        pl.lit(_SENTINEL, dtype=pl.Int64).alias("block"),
        variable_expr.alias("variable"),
        source_expr.alias("source"),
        pl.col("value").cast(pl.Float64).alias("value"),
    )

    return _conform(tidy)


def _tidy_newave_cobre_pair(
    results: Sequence[ResultComparison] | Sequence[BoundComparison],
) -> pl.DataFrame:
    """Emit the two ``newave``/``cobre`` tidy rows per comparison.

    Shared by :func:`tidy_from_results` and :func:`tidy_from_bounds`, which carry
    the same ``entity_type``/``cobre_id``/``entity_name``/``stage``/``variable``/
    ``newave_value``/``cobre_value`` fields. Returns a 0-row frame for empty input.
    """
    if not results:
        return pl.DataFrame(schema=TIDY_SCHEMA)

    n = len(results)
    sentinel = [_SENTINEL] * n
    shared = {
        "entity_type": [r.entity_type for r in results],
        "entity_id": [r.cobre_id for r in results],
        "entity_name": [r.entity_name for r in results],
        "bus": sentinel,
        "stage": [r.stage for r in results],
        "block": sentinel,
        "variable": [r.variable for r in results],
    }

    newave = pl.DataFrame(
        {
            **shared,
            "source": ["newave"] * n,
            "value": [r.newave_value for r in results],
        },
        schema=TIDY_SCHEMA,
    )
    cobre = pl.DataFrame(
        {**shared, "source": ["cobre"] * n, "value": [r.cobre_value for r in results]},
        schema=TIDY_SCHEMA,
    )

    return _conform(pl.concat([newave, cobre], how="vertical"))


def _conform(frame: pl.DataFrame) -> pl.DataFrame:
    """Select and cast ``frame`` to exactly the :data:`TIDY_SCHEMA` contract.

    Guarantees column presence, order, and dtypes regardless of inference,
    so the result passes :meth:`ComparisonDataset.validate`.

    Args:
        frame: A frame carrying (at least) the tidy columns.

    Returns:
        The frame with exactly the :data:`TIDY_SCHEMA` columns in order and the
        schema dtypes.
    """
    return frame.select(list(TIDY_SCHEMA)).cast(
        {col: dtype() for col, dtype in TIDY_SCHEMA.items()}
    )


# ---------------------------------------------------------------------------
# Production-function (FPHA) comparison
#
# Both solvers fit a piecewise-linear production surface GH(V, Q, S) as a set of
# hyperplanes whose lower envelope (the minimum over planes) is what the LP
# consumes. The two fits use different plane counts and philosophies, so they
# are NOT comparable plane-by-plane; instead both envelopes are evaluated on a
# shared (V, Q) grid (at S = 0) and the resulting surfaces are compared. A
# separate spillage slice (GH vs S at the max corner) covers the S dimension the
# (V, Q) grid holds fixed.
# ---------------------------------------------------------------------------

#: Output schema of the per-(plant, stage) FPHA fidelity metrics frame (the
#: comparison conclusion: how close Cobre's fitted surface is to the source
#: model's, normalized to the plant's max generation).
_FPHA_METRICS_SCHEMA: dict[str, type[pl.DataType]] = {
    "cobre_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "n_planes_newave": pl.Int64,
    "n_planes_cobre": pl.Int64,
    "n_v": pl.Int64,
    "nmae": pl.Float64,
    "bias": pl.Float64,
    "max_abs_dev": pl.Float64,
    "gh_max_ratio": pl.Float64,
}

#: Output schema of the dense (V, Q) production-surface frame (the render
#: substrate for the heatmaps; one row per grid point per source).
_FPHA_SURFACE_SCHEMA: dict[str, type[pl.DataType]] = {
    "cobre_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "v_hm3": pl.Float64,
    "q_m3s": pl.Float64,
    "source": pl.Utf8,
    "gh_mw": pl.Float64,
}

#: Output schema of the spillage-slice frame (GH vs spill at the max V/Q corner).
_FPHA_SPILL_SCHEMA: dict[str, type[pl.DataType]] = {
    "cobre_id": pl.Int64,
    "plant_name": pl.Utf8,
    "stage": pl.Int64,
    "s_m3s": pl.Float64,
    "source": pl.Utf8,
    "gh_mw": pl.Float64,
}


def _evaluate_fpha_envelope(
    gamma_0: np.ndarray,
    gamma_v: np.ndarray,
    gamma_q: np.ndarray,
    gamma_s: np.ndarray,
    multiplier: np.ndarray,
    v: np.ndarray,
    q: np.ndarray,
    s: np.ndarray,
    *,
    volume_offset: float = 0.0,
) -> np.ndarray:
    """Evaluate a min-over-planes FPHA envelope on a grid of operating points.

    Each plane contributes ``multiplier * (gamma_0 + gamma_v * (v -
    volume_offset) + gamma_q * q + gamma_s * s)``; the envelope is the minimum
    across planes — the value the operating model's LP consumes. ``v``, ``q``,
    ``s`` are broadcast together, so passing a meshgrid evaluates a whole
    surface in one call.

    ``volume_offset`` subtracts the plant minimum storage so a useful-volume
    coefficient is applied to absolute volume; pass ``0.0`` for a coefficient
    that already multiplies absolute volume. ``multiplier`` is the per-plane
    correction (the source model's ``fator_correcao``, Cobre's ``kappa``).

    Args:
        gamma_0: Per-plane constant term, shape ``(P,)``.
        gamma_v: Per-plane volume coefficient, shape ``(P,)``.
        gamma_q: Per-plane turbined-flow coefficient, shape ``(P,)``.
        gamma_s: Per-plane spilled-flow coefficient, shape ``(P,)``.
        multiplier: Per-plane scalar multiplier, shape ``(P,)``.
        v: Volume coordinate(s), broadcastable with ``q`` and ``s``.
        q: Turbined-flow coordinate(s), broadcastable with ``v`` and ``s``.
        s: Spilled-flow coordinate(s), broadcastable with ``v`` and ``q``.
        volume_offset: Storage subtracted from ``v`` before applying ``gamma_v``.

    Returns:
        The envelope generation, shaped like the broadcast of ``v``/``q``/``s``.
    """
    shape = np.broadcast_shapes(np.shape(v), np.shape(q), np.shape(s))
    vv = np.broadcast_to(v, shape).reshape(-1)
    qq = np.broadcast_to(q, shape).reshape(-1)
    ss = np.broadcast_to(s, shape).reshape(-1)
    useful_v = vv - volume_offset
    planes = multiplier[:, None] * (
        gamma_0[:, None]
        + gamma_v[:, None] * useful_v[None, :]
        + gamma_q[:, None] * qq[None, :]
        + gamma_s[:, None] * ss[None, :]
    )
    return planes.min(axis=0).reshape(shape)


def _plane_arrays(
    sub: pl.DataFrame, multiplier_col: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract ``(gamma_0, gamma_v, gamma_q, gamma_s, multiplier)`` as arrays."""
    return (
        sub.get_column("gamma_0").to_numpy(),
        sub.get_column("gamma_v").to_numpy(),
        sub.get_column("gamma_q").to_numpy(),
        sub.get_column("gamma_s").to_numpy(),
        sub.get_column(multiplier_col).to_numpy(),
    )


def build_fpha_comparison(
    nw_planes: pl.DataFrame | None,
    nw_grid: pl.DataFrame | None,
    cb_planes: pl.DataFrame | None,
    hydros: Sequence[HydroEntity],
    *,
    spill_n: int = 15,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Compare both solvers' fitted production functions on a shared grid.

    For every hydro fitted on BOTH sides, each (plant, stage) pair is evaluated at
    the source model's own ``(V, Q)`` fitting-grid nodes (``Npt_V × Npt_Q`` from
    ``fpha_eco``, at ``S = 0``); single-volume (run-of-river) plants collapse the
    ``V`` axis to one point, leaving a ``Q`` curve. Both envelopes are evaluated at
    the same nodes — the model's real resolution, not a synthetic densification.
    The source model's k-th period aligns to Cobre stage ``k``.

    Args:
        nw_planes: The source model's planes (from ``read_fpha_planes``), or
            ``None``.
        nw_grid: The source model's fitting grid (from ``read_fpha_grid``), or
            ``None`` — supplies the ``(V, Q)`` nodes and the useful-volume
            reference.
        cb_planes: Cobre's planes (from ``read_cobre_fpha_planes``), or ``None``.
        hydros: Aligned hydro entities pairing source codes to Cobre ids/names.
        spill_n: Sample count for the spillage slice (a synthetic ``S`` sweep,
            independent of the ``(V, Q)`` fitting grid).

    Returns:
        ``(metrics, surface, spill)`` frames conforming to
        :data:`_FPHA_METRICS_SCHEMA`, :data:`_FPHA_SURFACE_SCHEMA`, and
        :data:`_FPHA_SPILL_SCHEMA`. All three are empty (but typed) when either
        side lacks fitted planes or no plant is fitted on both sides.
    """
    empty = (
        pl.DataFrame(schema=_FPHA_METRICS_SCHEMA),
        pl.DataFrame(schema=_FPHA_SURFACE_SCHEMA),
        pl.DataFrame(schema=_FPHA_SPILL_SCHEMA),
    )
    if (
        nw_planes is None
        or nw_grid is None
        or cb_planes is None
        or nw_planes.is_empty()
        or nw_grid.is_empty()
        or cb_planes.is_empty()
    ):
        return empty

    code_to_cobre = {h.newave_code: h.cobre_id for h in hydros}
    name_of = {h.cobre_id: h.name for h in hydros}

    periods = sorted({int(p) for p in nw_planes.get_column("periodo").to_list()})
    stage_of = {p: i for i, p in enumerate(periods)}

    def _key2(
        frame: pl.DataFrame, a: str, b: str
    ) -> dict[tuple[int, int], pl.DataFrame]:
        parts = frame.partition_by([a, b], as_dict=True)
        return {(int(k[0]), int(k[1])): v for k, v in parts.items()}

    nw_lookup = _key2(nw_planes, "newave_code", "periodo")
    grid_lookup = _key2(nw_grid, "newave_code", "periodo")
    cb_lookup = _key2(cb_planes, "hydro_id", "stage_id")

    geo_cid: list[np.ndarray] = []
    geo_stage: list[np.ndarray] = []
    geo_v: list[np.ndarray] = []
    geo_q: list[np.ndarray] = []
    gh_nw: list[np.ndarray] = []
    gh_cb: list[np.ndarray] = []

    sp_cid: list[np.ndarray] = []
    sp_stage: list[np.ndarray] = []
    sp_s: list[np.ndarray] = []
    sp_gh_nw: list[np.ndarray] = []
    sp_gh_cb: list[np.ndarray] = []

    metric_rows: list[dict[str, object]] = []

    for (code, periodo), nw_sub in nw_lookup.items():
        cobre_id = code_to_cobre.get(code)
        stage = stage_of.get(periodo)
        if cobre_id is None or stage is None:
            continue
        cb_sub = cb_lookup.get((cobre_id, stage))
        grid_row = grid_lookup.get((code, periodo))
        if cb_sub is None or grid_row is None:
            continue

        g = grid_row.row(0, named=True)
        v_min = float(g["v_min_hm3"])
        v_max = float(g["v_max_hm3"])
        q_min = float(g["q_min_m3s"])
        q_max = float(g["q_max_m3s"])
        n_v = int(g["n_v"])

        # Sample at the source model's own fitting-grid nodes (Npt_V x Npt_Q).
        if n_v <= 1 or v_max <= v_min:
            v_axis = np.array([v_min], dtype=float)
        else:
            v_axis = np.linspace(v_min, v_max, n_v)
        q_axis = np.linspace(q_min, q_max, max(int(g["n_q"]), 2))
        vv, qq = np.meshgrid(v_axis, q_axis, indexing="ij")
        ss = np.zeros_like(vv)

        nw_arr = _plane_arrays(nw_sub, "fator_correcao")
        cb_arr = _plane_arrays(cb_sub, "kappa")
        nw_gh = _evaluate_fpha_envelope(*nw_arr, vv, qq, ss, volume_offset=v_min)
        cb_gh = _evaluate_fpha_envelope(*cb_arr, vv, qq, ss, volume_offset=0.0)

        flat_v = vv.reshape(-1)
        n = flat_v.size
        geo_cid.append(np.full(n, cobre_id, dtype=np.int64))
        geo_stage.append(np.full(n, stage, dtype=np.int64))
        geo_v.append(flat_v)
        geo_q.append(qq.reshape(-1))
        gh_nw.append(nw_gh.reshape(-1))
        gh_cb.append(cb_gh.reshape(-1))

        # Spillage slice at the max V/Q corner: GH vs spill up to ~2x q_max.
        s_axis = (
            np.linspace(0.0, 2.0 * q_max, max(spill_n, 2))
            if q_max > 0.0
            else np.array([0.0], dtype=float)
        )
        v_fix = np.full_like(s_axis, v_max)
        q_fix = np.full_like(s_axis, q_max)
        nw_sp = _evaluate_fpha_envelope(
            *nw_arr, v_fix, q_fix, s_axis, volume_offset=v_min
        )
        cb_sp = _evaluate_fpha_envelope(
            *cb_arr, v_fix, q_fix, s_axis, volume_offset=0.0
        )
        ns = s_axis.size
        sp_cid.append(np.full(ns, cobre_id, dtype=np.int64))
        sp_stage.append(np.full(ns, stage, dtype=np.int64))
        sp_s.append(s_axis)
        sp_gh_nw.append(nw_sp)
        sp_gh_cb.append(cb_sp)

        # Fidelity metrics, normalized to the plant's max generation (the corner
        # value v_max/q_max, which the grid includes as its last point).
        gh_max_nw = float(nw_gh[-1, -1])
        gh_max_cb = float(cb_gh[-1, -1])
        diff = cb_gh - nw_gh
        denom = abs(gh_max_nw)
        scaled = denom > 1e-9
        metric_rows.append(
            {
                "cobre_id": cobre_id,
                "plant_name": name_of.get(cobre_id, f"hydro_{cobre_id}"),
                "stage": stage,
                "n_planes_newave": nw_sub.height,
                "n_planes_cobre": cb_sub.height,
                "n_v": n_v,
                "nmae": float(np.mean(np.abs(diff)) / denom) if scaled else None,
                "bias": float(np.mean(diff) / denom) if scaled else None,
                "max_abs_dev": float(np.max(np.abs(diff))),
                "gh_max_ratio": (gh_max_cb / gh_max_nw) if scaled else None,
            }
        )

    if not geo_cid:
        return empty

    names = pl.DataFrame(
        {
            "cobre_id": list(name_of.keys()),
            "plant_name": list(name_of.values()),
        },
        schema={"cobre_id": pl.Int64, "plant_name": pl.Utf8},
    )

    def _stack_surface(gh: list[np.ndarray], source: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "cobre_id": np.concatenate(geo_cid),
                "stage": np.concatenate(geo_stage),
                "v_hm3": np.concatenate(geo_v),
                "q_m3s": np.concatenate(geo_q),
                "gh_mw": np.concatenate(gh),
            }
        ).with_columns(pl.lit(source).alias("source"))

    surface = (
        pl.concat([_stack_surface(gh_nw, "newave"), _stack_surface(gh_cb, "cobre")])
        .join(names, on="cobre_id", how="left")
        .select(list(_FPHA_SURFACE_SCHEMA))
        .cast({col: dtype() for col, dtype in _FPHA_SURFACE_SCHEMA.items()})
        .sort(["cobre_id", "stage", "source", "v_hm3", "q_m3s"])
    )

    def _stack_spill(gh: list[np.ndarray], source: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "cobre_id": np.concatenate(sp_cid),
                "stage": np.concatenate(sp_stage),
                "s_m3s": np.concatenate(sp_s),
                "gh_mw": np.concatenate(gh),
            }
        ).with_columns(pl.lit(source).alias("source"))

    spill = (
        pl.concat([_stack_spill(sp_gh_nw, "newave"), _stack_spill(sp_gh_cb, "cobre")])
        .join(names, on="cobre_id", how="left")
        .select(list(_FPHA_SPILL_SCHEMA))
        .cast({col: dtype() for col, dtype in _FPHA_SPILL_SCHEMA.items()})
        .sort(["cobre_id", "stage", "source", "s_m3s"])
    )

    metrics = pl.DataFrame(metric_rows, schema=_FPHA_METRICS_SCHEMA).sort(
        ["cobre_id", "stage"]
    )
    return metrics, surface, spill


# ---------------------------------------------------------------------------
# Productivity (single-value-per-plant/stage) analysis
# ---------------------------------------------------------------------------

_PRODUCTIVITY_DETAIL_SCHEMA: dict[str, type[pl.DataType]] = {
    "plant_name": pl.Utf8,
    "newave_code": pl.Int64,
    "cobre_id": pl.Int64,
    # The source model pmo.dat head-dependent productivities (FPHA).
    "nw_altura_min": pl.Float64,
    "nw_altura_65": pl.Float64,
    "nw_altura_max": pl.Float64,
    "nw_equivalent": pl.Float64,
    "nw_accumulated_earm": pl.Float64,
    # The source model HIDR cadastro building blocks.
    "nw_specific_productivity": pl.Float64,
    "nw_tailwater_m": pl.Float64,
    "nw_losses_m": pl.Float64,
    "nw_vmin_hm3": pl.Float64,
    "nw_vmax_hm3": pl.Float64,
    # cobre-bridge side: the *static* productivities the converter computes from the
    # source model inputs (HIDR cadastro + cascade), so the scatters are a
    # conversion-fidelity check against the matching pmo column rather than a comparison
    # against the per-stage simulation output.
    "cb_point": pl.Float64,
    "cb_equivalent": pl.Float64,
    "cb_accumulated": pl.Float64,
    # cobre-bridge converted building blocks (from system/hydros.json).
    "cb_specific_productivity": pl.Float64,
    "cb_tailwater_m": pl.Float64,
    "cb_losses_m": pl.Float64,
    "cb_vmin_hm3": pl.Float64,
    "cb_vmax_hm3": pl.Float64,
}


def build_productivity_detail(
    alignment: EntityAlignment,
    nw_prod_detail: pl.DataFrame,
    nw_cadastro: pd.DataFrame,
    cobre_prod_detail: dict[int, dict],
    cb_accumulated: dict[int, float],
) -> pl.DataFrame:
    """Assemble the per-plant static productivity comparison frame.

    One row per aligned hydro pair (``alignment.hydros``). The source model side carries
    the pmo.dat head-dependent productivities (matched by plant name) and the HIDR
    cadastro building blocks (matched by the source model code). The cobre-bridge side
    carries the *static* productivities the converter computes from those same inputs —
    ``cb_point`` from :func:`compute_productivity`, ``cb_equivalent`` from
    :func:`stored_energy_productivity`, ``cb_accumulated`` from the cascade accumulated
    map — plus the building blocks written into ``system/hydros.json``
    (``cobre_prod_detail``). Matching pmo and cobre-bridge columns should land on ``y =
    x`` (validating the conversion), since both are derived from the same the source
    model inputs.

    Returns an empty frame (with the full schema) when there are no aligned
    hydros.
    """
    from cobre_bridge.productivity import (
        compute_productivity,
        stored_energy_productivity,
    )

    nw_by_name: dict[str, dict] = {}
    for row in nw_prod_detail.iter_rows(named=True):
        nw_by_name[str(row["plant_name"]).strip().upper()] = row

    def _cad_float(code: int, column: str) -> float | None:
        if code not in nw_cadastro.index or column not in nw_cadastro.columns:
            return None
        value = nw_cadastro.loc[code, column]
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        return f if f == f else None  # drop NaN

    def _nw_reservoir_bounds(code: int) -> tuple[float | None, float | None]:
        """The source model reservoir bounds as cobre-bridge models them.

        Daily-regulation ('D') plants are frozen at ``volume_referencia`` by the
        converter (they can't store across stages), so compare like-for-like
        against Cobre's reservoir rather than the dead-storage
        ``volume_minimo``/``volume_maximo``. Otherwise every run-of-river plant
        shows a spurious Vmin/Vmax delta in the building-blocks table.
        """
        if code in nw_cadastro.index and "tipo_regulacao" in nw_cadastro.columns:
            reg = str(nw_cadastro.loc[code, "tipo_regulacao"]).strip()
            if reg == "D":
                vref = _cad_float(code, "volume_referencia")
                if vref is not None:
                    return vref, vref
        return (
            _cad_float(code, "volume_minimo"),
            _cad_float(code, "volume_maximo"),
        )

    def _cb_static(code: int) -> tuple[float | None, float | None, float | None]:
        """cobre-bridge (point, equivalent, accumulated) computed from inputs."""
        if code not in nw_cadastro.index:
            return None, None, None
        hreg = nw_cadastro.loc[code]
        try:
            point = float(compute_productivity(hreg))
            equiv = float(stored_energy_productivity(hreg))
        except (KeyError, ValueError, TypeError):
            point = equiv = None  # type: ignore[assignment]
        acc = cb_accumulated.get(code)
        return point, equiv, (float(acc) if acc is not None else None)

    rows: list[dict] = []
    for hydro in alignment.hydros:
        nw_prod = nw_by_name.get(hydro.name.strip().upper(), {})
        cb = cobre_prod_detail.get(hydro.cobre_id, {})
        cb_point, cb_equiv, cb_acc = _cb_static(hydro.newave_code)
        nw_vmin, nw_vmax = _nw_reservoir_bounds(hydro.newave_code)
        rows.append(
            {
                "plant_name": hydro.name,
                "newave_code": hydro.newave_code,
                "cobre_id": hydro.cobre_id,
                "nw_altura_min": nw_prod.get("altura_min"),
                "nw_altura_65": nw_prod.get("altura_65"),
                "nw_altura_max": nw_prod.get("altura_max"),
                "nw_equivalent": nw_prod.get("equivalent"),
                "nw_accumulated_earm": nw_prod.get("accumulated_earm"),
                "nw_specific_productivity": _cad_float(
                    hydro.newave_code, "produtibilidade_especifica"
                ),
                "nw_tailwater_m": _cad_float(hydro.newave_code, "canal_fuga_medio"),
                "nw_losses_m": _cad_float(hydro.newave_code, "perdas"),
                "nw_vmin_hm3": nw_vmin,
                "nw_vmax_hm3": nw_vmax,
                "cb_point": cb_point,
                "cb_equivalent": cb_equiv,
                "cb_accumulated": cb_acc,
                "cb_specific_productivity": cb.get("specific_productivity"),
                "cb_tailwater_m": cb.get("tailwater_m"),
                "cb_losses_m": cb.get("losses_m"),
                "cb_vmin_hm3": cb.get("vmin_hm3"),
                "cb_vmax_hm3": cb.get("vmax_hm3"),
            }
        )

    if not rows:
        return pl.DataFrame(schema=_PRODUCTIVITY_DETAIL_SCHEMA)
    return pl.DataFrame(rows, schema=_PRODUCTIVITY_DETAIL_SCHEMA)


#: Output schema of the per-(plant, stage) realized-productivity frame consumed by
#: the per-stage productivity chart.
_PRODUCTIVITY_PER_STAGE_SCHEMA: dict[str, type[pl.DataType]] = {
    "plant_name": pl.Utf8,
    "newave_code": pl.Int64,
    "cobre_id": pl.Int64,
    "stage": pl.Int64,
    "newave_value": pl.Float64,
    "cobre_value": pl.Float64,
}


def productivity_per_stage_frame(results: Sequence[ResultComparison]) -> pl.DataFrame:
    """Per-(plant, stage) realized productivity rows for the Productivity tab.

    Extracts the hydro ``productivity_mw_per_m3s`` comparisons (productivity is
    constant within a stage but varies across stages, tracking the reservoir head
    reached each stage) into a frame the per-stage chart renders directly — so the
    result-row filtering lives in the analyze layer, not the chart.

    Args:
        results: The result comparisons; only hydro ``productivity_mw_per_m3s``
            rows are kept.

    Returns:
        A frame conforming to :data:`_PRODUCTIVITY_PER_STAGE_SCHEMA`, one row per
        (plant, stage). Empty (but typed) when no such rows exist.
    """
    rows = [
        {
            "plant_name": r.entity_name,
            "newave_code": r.newave_code,
            "cobre_id": r.cobre_id,
            "stage": r.stage,
            "newave_value": r.newave_value,
            "cobre_value": r.cobre_value,
        }
        for r in results
        if r.entity_type == "hydro" and r.variable == "productivity_mw_per_m3s"
    ]
    if not rows:
        return pl.DataFrame(schema=_PRODUCTIVITY_PER_STAGE_SCHEMA)
    return pl.DataFrame(rows, schema=_PRODUCTIVITY_PER_STAGE_SCHEMA)
