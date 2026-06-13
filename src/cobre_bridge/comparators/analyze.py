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

import polars as pl

from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.results import build_results_summary, smape

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    """Sum NEWAVE/Cobre values per stage for one entity type (and variable).

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


def _bus_name_lookups(
    hydro_meta: dict[int, dict[str, object]],
    bus_meta: dict[int, dict[str, object]],
) -> tuple[dict[int, str], dict[int, int]]:
    """Build the ``bus_id -> name`` and ``plant -> bus_id`` lookups.

    Shared helper for the per-bus roll-up: reproduces the two dict
    comprehensions at ``charts.py:900-907`` / ``charts.py:1198-1205`` exactly.
    ``bus_id_to_name`` maps each bus id to ``bus_meta[bid]["name"]`` (fallback
    ``str(bid)``); ``hydro_to_bus`` maps each plant id to its ``bus_id``,
    skipping plants whose ``bus_id`` is ``None``.

    Args:
        hydro_meta: Per-plant metadata carrying ``bus_id``; read-only.
        bus_meta: Per-bus metadata carrying ``name``; read-only.

    Returns:
        ``(bus_id_to_name, hydro_to_bus)``.
    """
    bus_id_to_name: dict[int, str] = {
        bid: cast("str", meta.get("name", str(bid))) for bid, meta in bus_meta.items()
    }
    hydro_to_bus: dict[int, int] = {
        hid: cast("int", meta["bus_id"])
        for hid, meta in hydro_meta.items()
        if meta.get("bus_id") is not None
    }
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
    owning bus via ``hydro_meta[cobre_id]["bus_id"]`` (plants with no ``bus_id``
    are skipped), the bus name is resolved via ``bus_meta[bus_id]["name"]``
    (fallback ``str(bus_id)``) and upper-cased, and the fictitious buses
    ``NOFICT1/2/3`` are dropped. Surviving rows accumulate ``newave_value`` /
    ``cobre_value`` into their ``(bus, stage)`` bucket and add ``cobre_id`` to
    the bus's id set.

    Args:
        results: The comparison rows; consumed verbatim (read-only).
        variable: The ``variable`` to keep.
        hydro_meta: Per-plant metadata carrying ``bus_id``; read-only.
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
        hydro_meta: Per-plant metadata carrying ``bus_id``; read-only.
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
    """Roll a Cobre per-hydro variable and a NEWAVE-SIN long frame to per-stage totals.

    Pure numeric core of ``charts.cobre_aggregate_chart`` (``charts.py:813-846``).
    The Cobre side sums ``variable`` across (optionally ``matched_ids``-filtered)
    plants per ``stage_id`` — delegated to :func:`per_stage_sum_from_frame` so the
    grouping/sort/filter semantics stay identical. The NEWAVE side folds the long
    ``nw_sin`` frame into a per-stage total: rows are filtered to
    ``variable.strip().upper() == nw_variable``, then each surviving row adds
    ``value * nw_factor`` into bucket ``stage - nw_offset``, skipping rows whose
    ``stage`` or ``value`` is ``None``. Never raises.

    Args:
        cobre_hydro: Per-hydro Cobre means with ``entity_id``, ``stage_id`` and
            the ``variable`` column.
        variable: The column to sum on the Cobre side.
        nw_sin: Long-format NEWAVE SIN frame with ``stage``, ``variable`` and
            ``value`` columns, or ``None``.
        nw_variable: The (already upper-cased) variable to keep in ``nw_sin``, or
            ``None`` to skip the NEWAVE side entirely.
        nw_factor: Multiplicative factor applied to each NEWAVE value (unit
            alignment).
        nw_offset: Subtracted from each NEWAVE ``stage`` to align with the Cobre
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
    """Build the NEWAVE and Cobre per-variable, per-stage spillage lookups.

    Pure numeric core of ``charts.system_spillage_energy_chart``
    (``charts.py:1565-1581``). The NEWAVE lookup is keyed by each
    ``system_spillage`` row's ``variable`` (e.g. ``VERTOT``/``VERTcont``/
    ``VERTfio``) then ``stage`` to ``newave_value``. The Cobre lookup maps the
    ``cobre_spill_energy`` frame's ``total_mw``/``reservoir_mw``/``rorov_mw``
    columns per ``stage_id`` under the keys ``spill_energy_total_mw`` /
    ``spill_energy_reservoir_mw`` / ``spill_energy_rorov_mw``. Never raises.

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
