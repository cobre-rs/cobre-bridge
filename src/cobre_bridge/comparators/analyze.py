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
from typing import TYPE_CHECKING

import polars as pl

from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.results import _smape, build_results_summary

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
    dicts from ``pct``) verbatim. Validates before returning.

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
        "nw_costs": pct.nw_costs,
        "cobre_costs": pct.cobre_costs,
        "nw_bus_names": pct.nw_bus_names,
        "nw_hydro_names": pct.nw_hydro_names,
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
    :func:`cobre_bridge.comparators.results._smape` over
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
        smapes = [_smape(r.newave_value, r.cobre_value) for r in group]
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
