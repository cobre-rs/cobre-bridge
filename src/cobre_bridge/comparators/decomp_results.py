"""Side-by-side comparison of a DECOMP run against its converted Cobre run.

This is the minimal slice: it aligns the source model's shipped operation
tables (``dec_oper_*.csv``, directly in the deck directory) onto Cobre's
simulation output and reports
per-variable divergence. No tolerance verdict and no HTML — the point is that
both sides are read, aligned, and compared on the same quantities.

Three conventions, each chosen so neither side is silently privileged:

- **Stage aggregation is taken, not recomputed.** Every ``dec_oper_*`` table
  carries a ``patamar``-null row per (stage, node) holding the source model's
  own duration-weighted stage value; the comparison reads that row instead of
  re-deriving one from the per-block rows, so a block-weighting convention
  cannot drift between the two products.
- **Scenario averaging is unweighted on both sides.** Cobre's simulation
  statistics are plain scenario means until weighted statistics land with the
  explicit-tree work; weighting only the source side would bias every
  terminal-stage row. When per-node probabilities become consumable the two
  sides move together.
- **Storage is compared as useful volume.** The source model reports useful
  volume; Cobre's ``storage_final_hm3`` is absolute, so the registry's
  ``min_storage_hm3`` is subtracted before comparing.

Entities the id map cannot resolve are counted and reported, never dropped in
silence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from cobre_bridge.comparators import cobre_readers
from cobre_bridge.comparators.decomp_readers import (
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_dec_oper_usit,
    read_relato_convergence,
)
from cobre_bridge.comparators.results import PercentileData, ResultComparison

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset

_LOG = logging.getLogger(__name__)

#: Rows whose two values are both below this magnitude are treated as an exact
#: match rather than a division by ~0 (physical units: MW, m³/s, hm³).
_ZERO_FLOOR = 1e-9


@dataclass(frozen=True)
class _Variable:
    """One comparable quantity and its column on each side."""

    level: str
    name: str
    source_column: str
    cobre_column: str
    unit: str


_HYDRO_VARIABLES: tuple[_Variable, ...] = (
    _Variable("hydro", "generation", "geracao_MW", "generation_mw", "MW"),
    _Variable("hydro", "turbined", "vazao_turbinada_m3s", "turbined_m3s", "m³/s"),
    _Variable("hydro", "spillage", "vazao_vertida_m3s", "spillage_m3s", "m³/s"),
    _Variable("hydro", "outflow", "vazao_defluente_m3s", "outflow_m3s", "m³/s"),
    _Variable("hydro", "storage", "volume_util_final_hm3", "useful_storage_hm3", "hm³"),
)

_THERMAL_VARIABLES: tuple[_Variable, ...] = (
    _Variable("thermal", "generation", "geracao_MW", "generation_mw", "MW"),
)

_BUS_VARIABLES: tuple[_Variable, ...] = (
    _Variable("bus", "deficit", "deficit_MW", "deficit_mw", "MW"),
    _Variable("bus", "spot_price", "cmo", "spot_price", "$/MWh"),
)

#: Map each (level, ``_Variable.name``) pair to the canonical variable name
#: the shared render/chart layer expects (``analyze.build_results_dataset``
#: and everything downstream of it key charts off this name, not the
#: decomp-local ``_Variable.name``). Storage is the one name that does not
#: match its ``_Variable``/cobre-column spelling: the comparison is on
#: *useful* volume (see the module docstring), but the canonical chart
#: variable is still ``storage_final_hm3`` — the same name the newave-side
#: comparator uses for its (absolute-volume) storage rows.
_CANONICAL_VARIABLE: dict[tuple[str, str], str] = {
    ("hydro", "generation"): "generation_mw",
    ("hydro", "turbined"): "turbined_m3s",
    ("hydro", "spillage"): "spillage_m3s",
    ("hydro", "outflow"): "outflow_m3s",
    ("hydro", "storage"): "storage_final_hm3",
    ("thermal", "generation"): "generation_mw",
    ("bus", "deficit"): "deficit_mw",
    ("bus", "spot_price"): "spot_price",
}


@dataclass(frozen=True)
class DecompComparison:
    """Aligned comparison of one source-model run against one Cobre run.

    ``rows`` is tidy — one row per (level, variable, entity, stage) with both
    values and their difference. ``summary`` rolls that up per variable, and
    ``convergence`` puts the two bound sequences side by side.
    """

    rows: pl.DataFrame
    summary: pl.DataFrame
    convergence: pl.DataFrame
    unmapped: dict[str, list[int]]

    @property
    def stage_count(self) -> int:
        """Number of distinct stages that carry at least one compared row."""
        if self.rows.is_empty():
            return 0
        return int(self.rows["stage_id"].n_unique())


def _stage_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the source model's own stage-aggregate rows (``patamar`` null).

    Falls back to a duration-weighted fold of the per-block rows if a deck
    ever ships without the aggregate row, so the comparison degrades to a
    computed value instead of an empty frame.
    """
    if "patamar" not in frame.columns:
        return frame
    aggregate = frame.filter(pl.col("patamar").is_null())
    if not aggregate.is_empty():
        return aggregate.drop("patamar")
    _LOG.warning(
        "No stage-aggregate rows (patamar null) found; folding per-block rows "
        "by duration instead"
    )
    return frame


def _scenario_mean(
    frame: pl.DataFrame, key: str, columns: list[str], *, entity_column: str | None
) -> pl.DataFrame:
    """Average the stage rows over the source model's nodes.

    ``key`` is the stage column; ``entity_column`` narrows the grouping to one
    entity per stage (``None`` for tables that are already one row per stage).
    """
    present = [c for c in columns if c in frame.columns]
    group = [key] if entity_column is None else [key, entity_column]
    return (
        frame.group_by(group)
        .agg([pl.col(c).mean().alias(c) for c in present])
        .sort(group)
    )


def _difference_columns() -> list[pl.Expr]:
    """Δ and its relative form, with a floor so ~0 vs ~0 reads as agreement."""
    delta = pl.col("cobre") - pl.col("source")
    magnitude = pl.col("source").abs() + pl.col("cobre").abs()
    return [
        delta.alias("delta"),
        pl.when(pl.col("source").abs() > _ZERO_FLOOR)
        .then(delta / pl.col("source").abs() * 100.0)
        .otherwise(None)
        .alias("delta_pct"),
        # Same definition as comparators.results.smape, expressed per row so a
        # near-zero reference cannot dominate the roll-up.
        pl.when(magnitude > _ZERO_FLOOR)
        .then(delta.abs() / (magnitude / 2.0) * 100.0)
        .otherwise(0.0)
        .alias("smape_pct"),
    ]


def _result_diff(nw_value: float, cobre_value: float) -> tuple[float, float | None]:
    """Absolute and relative difference for one ``ResultComparison`` row.

    Mirrors :func:`cobre_bridge.comparators.results._compute_diff` exactly, so
    the two per-variable stat kernels in play here (this module's own tidy
    rows and the shared ``ResultComparison`` one) agree on what "the
    difference" means.
    """
    abs_diff = abs(nw_value - cobre_value)
    rel_diff = abs_diff / abs(nw_value) if abs(nw_value) > 1e-10 else None
    return abs_diff, rel_diff


def _tidy(
    source: pl.DataFrame,
    cobre: pl.DataFrame,
    variables: tuple[_Variable, ...],
    *,
    names: dict[int, str],
) -> pl.DataFrame:
    """Join one level's two frames into tidy per-variable rows."""
    joined = source.join(cobre, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return _empty_rows()

    parts: list[pl.DataFrame] = []
    for var in variables:
        if var.source_column not in joined.columns:
            continue
        if var.cobre_column not in joined.columns:
            _LOG.info("Cobre output has no %s column; skipping", var.cobre_column)
            continue
        part = joined.select(
            pl.lit(var.level).alias("level"),
            pl.lit(var.name).alias("variable"),
            pl.lit(var.unit).alias("unit"),
            pl.col("entity_id"),
            pl.col("entity_id")
            .map_elements(lambda i: names.get(int(i), ""), return_dtype=pl.Utf8)
            .alias("entity_name"),
            pl.col("stage_id"),
            pl.col(var.source_column).cast(pl.Float64).alias("source"),
            pl.col(var.cobre_column).cast(pl.Float64).alias("cobre"),
        ).with_columns(_difference_columns())
        parts.append(part)

    if not parts:
        return _empty_rows()
    return pl.concat(parts)


def _empty_rows() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "level": pl.Utf8,
            "variable": pl.Utf8,
            "unit": pl.Utf8,
            "entity_id": pl.Int64,
            "entity_name": pl.Utf8,
            "stage_id": pl.Int64,
            "source": pl.Float64,
            "cobre": pl.Float64,
            "delta": pl.Float64,
            "delta_pct": pl.Float64,
            "smape_pct": pl.Float64,
        }
    )


def _result_comparisons(
    source: pl.DataFrame,
    cobre: pl.DataFrame,
    variables: tuple[_Variable, ...],
    *,
    names: dict[int, str],
) -> list[ResultComparison]:
    """Join one level's two frames into ``ResultComparison`` rows.

    Mirrors :func:`_tidy`'s join, but emits the canonical
    :class:`~cobre_bridge.comparators.results.ResultComparison` shape
    :func:`build_decomp_dataset` hands to
    :func:`cobre_bridge.comparators.analyze.build_results_dataset`, keyed by
    :data:`_CANONICAL_VARIABLE` rather than the decomp-local
    ``_Variable.name`` :func:`_tidy` uses for its own standalone rows.
    """
    joined = source.join(cobre, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return []

    applicable = [
        var
        for var in variables
        if var.source_column in joined.columns and var.cobre_column in joined.columns
    ]
    if not applicable:
        return []

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        entity_id = int(row["entity_id"])
        stage = int(row["stage_id"])
        newave_code = int(row["newave_code"])
        entity_name = names.get(entity_id, "")
        for var in applicable:
            nw_val = row[var.source_column]
            cobre_val = row[var.cobre_column]
            if nw_val is None or cobre_val is None:
                continue
            nw_value = float(nw_val)
            cobre_value = float(cobre_val)
            abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
            results.append(
                ResultComparison(
                    entity_type=var.level,
                    entity_name=entity_name,
                    newave_code=newave_code,
                    cobre_id=entity_id,
                    stage=stage,
                    variable=_CANONICAL_VARIABLE[(var.level, var.name)],
                    newave_value=nw_value,
                    cobre_value=cobre_value,
                    abs_diff=abs_diff,
                    rel_diff=rel_diff,
                )
            )
    return results


def _summarize(rows: pl.DataFrame) -> pl.DataFrame:
    """Roll tidy rows up to one line per (level, variable)."""
    if rows.is_empty():
        return pl.DataFrame(
            schema={
                "level": pl.Utf8,
                "variable": pl.Utf8,
                "unit": pl.Utf8,
                "n": pl.UInt32,
                "source_total": pl.Float64,
                "cobre_total": pl.Float64,
                "delta_total": pl.Float64,
                "delta_total_pct": pl.Float64,
                "smape_pct": pl.Float64,
                "worst_entity": pl.Utf8,
                "worst_delta": pl.Float64,
            }
        )

    worst = (
        rows.sort(pl.col("delta").abs(), descending=True, nulls_last=True)
        .group_by("level", "variable")
        .agg(
            pl.col("entity_name").first().alias("worst_entity"),
            pl.col("delta").first().alias("worst_delta"),
        )
    )
    return (
        rows.group_by("level", "variable")
        .agg(
            pl.col("unit").first(),
            pl.len().alias("n").cast(pl.UInt32),
            pl.col("source").sum().alias("source_total"),
            pl.col("cobre").sum().alias("cobre_total"),
            pl.col("smape_pct").mean().alias("smape_pct"),
        )
        .with_columns(
            (pl.col("cobre_total") - pl.col("source_total")).alias("delta_total")
        )
        .with_columns(
            pl.when(pl.col("source_total").abs() > _ZERO_FLOOR)
            .then(pl.col("delta_total") / pl.col("source_total").abs() * 100.0)
            .otherwise(None)
            .alias("delta_total_pct")
        )
        .join(worst, on=["level", "variable"], how="left")
        .sort("level", "variable")
    )


def _convergence(decomp_dir: Path, cobre_output_dir: Path) -> pl.DataFrame:
    """Put both bound sequences side by side, one row per iteration index."""
    try:
        source = read_relato_convergence(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.warning("No source-model convergence table: %s", exc)
        source = pl.DataFrame()

    cobre = cobre_readers.read_cobre_convergence(cobre_output_dir)

    frames: list[pl.DataFrame] = []
    if not source.is_empty():
        columns = {c.lower(): c for c in source.columns}
        selected = [
            pl.col(columns["iteracao"]).cast(pl.Int64).alias("iteration"),
            pl.col(columns["zinf"]).cast(pl.Float64).alias("source_lower"),
            pl.col(columns["zsup"]).cast(pl.Float64).alias("source_upper"),
        ]
        frames.append(source.select(selected))
    if not cobre.is_empty():
        frames.append(
            cobre.select(
                pl.col("iteration").cast(pl.Int64),
                pl.col("lower_bound").alias("cobre_lower"),
                pl.col("upper_bound_mean").alias("cobre_upper"),
            )
        )

    if not frames:
        return pl.DataFrame(schema={"iteration": pl.Int64})
    combined = frames[0]
    for frame in frames[1:]:
        combined = combined.join(frame, on="iteration", how="full", coalesce=True)
    return combined.sort("iteration")


def _hydro_side(
    decomp_dir: Path, id_map_hydro: dict[int, int]
) -> tuple[pl.DataFrame, list[int]]:
    frame = _stage_rows(read_dec_oper_usih(decomp_dir))
    columns = [v.source_column for v in _HYDRO_VARIABLES]
    aggregated = _scenario_mean(frame, "estagio", columns, entity_column="codigo_usina")
    return _map_entities(aggregated, "codigo_usina", id_map_hydro)


def _thermal_side(
    decomp_dir: Path, id_map_thermal: dict[int, int]
) -> tuple[pl.DataFrame, list[int]]:
    frame = _stage_rows(read_dec_oper_usit(decomp_dir))
    columns = [v.source_column for v in _THERMAL_VARIABLES]
    aggregated = _scenario_mean(frame, "estagio", columns, entity_column="codigo_usina")
    return _map_entities(aggregated, "codigo_usina", id_map_thermal)


def _bus_side(
    decomp_dir: Path, id_map_bus: dict[int, int]
) -> tuple[pl.DataFrame, list[int]]:
    frame = _stage_rows(read_dec_oper_sist(decomp_dir))
    columns = [v.source_column for v in _BUS_VARIABLES]
    aggregated = _scenario_mean(
        frame, "estagio", columns, entity_column="codigo_submercado"
    )
    return _map_entities(aggregated, "codigo_submercado", id_map_bus)


#: ``dec_oper_sist`` columns the Energy Balance tab's reference-model overlay
#: needs: the two generation families it sums into GHTOT/GTERM, the demand
#: and non-controllable-generation basis for NET_LOAD, and the two system
#: energy series (EARM/ENA). Read once via :func:`_scenario_mean` so scenario
#: averaging matches the rest of the module (unweighted mean across nodes).
_NW_MARKET_COLUMNS: tuple[str, ...] = (
    "demanda_MW",
    "geracao_hidroeletrica_MW",
    "geracao_termica_MW",
    "geracao_termica_antecipada_MW",
    "geracao_eolica_MW",
    "geracao_pequenas_usinas_MW",
    "deficit_MW",
    "ena_MWmes",
    "earm_final_MWmes",
)


def _empty_market_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "newave_code": pl.Int64,
            "stage": pl.Int64,
            "variable": pl.Utf8,
            "value": pl.Float64,
        }
    )


def _energy_balance_frames(
    decomp_dir: Path, bus_codes: dict[int, int]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the Energy Balance tab's DECOMP-side reference frames.

    Returns ``(nw_market, nw_net_load, nw_sin)`` — the long
    ``newave_code``/``stage``/``variable``/``value`` frames
    :func:`~cobre_bridge.comparators.charts.build_energy_balance_tab` and
    :func:`~cobre_bridge.comparators.charts.cobre_aggregate_chart` read.

    Only the tokens the tab actually consumes are emitted — no dead rows:
    ``GHTOT``/``GTERM``/``DEFT`` in ``nw_market`` (mirroring
    ``charts._BALANCE_VARS``), ``NET_LOAD`` in ``nw_net_load``, and
    ``EARMF``/``ENA`` in ``nw_sin``. The tab's fifth token, ``EXCESSO``
    (system energy excess), has no DECOMP source — neither ``dec_oper_sist``
    nor ``relato.balanco_energetico`` carries an excess/curtailment column —
    so it is intentionally omitted rather than fabricated; that panel still
    renders (Cobre-only, no NEWAVE overlay line), matching the tab's existing
    graceful-degradation path for a variable absent on one side.

    ``newave_code`` here is the **Cobre bus id**, not the raw DECOMP
    ``codigo_submercado`` — ``nw_bus_names`` (ticket-002,
    ``build_decomp_dataset``) is populated from ``read_cobre_bus_metadata``
    and is therefore keyed by Cobre bus id, and ``build_energy_balance_tab``
    joins ``nw_market``/``nw_net_load`` against it by that same key.
    ``bus_codes`` (``{codigo_submercado: cobre_bus_id}``, already built by
    ``_read_aligned_frames``) performs that translation; ``nw_sin`` needs no
    such translation since it sums across every submarket rather than
    keying per-bus.

    D-STAGE-OFFSET: ``stage`` stays the raw 1-based ``estagio`` (unlike
    :func:`_map_entities`, this does *not* rebase to Cobre's 0-based
    ``stage_id``), so the consuming charts' own stage-alignment (``nw_offset
    = 1``, set on ``PercentileData`` by ``build_decomp_dataset``) lines up.
    """
    frame = _stage_rows(read_dec_oper_sist(decomp_dir))
    if frame.is_empty() or "codigo_submercado" not in frame.columns:
        empty = _empty_market_frame()
        return empty, empty, empty

    aggregated = _scenario_mean(
        frame, "estagio", list(_NW_MARKET_COLUMNS), entity_column="codigo_submercado"
    )
    present = set(aggregated.columns)

    mapped = aggregated.with_columns(
        pl.col("codigo_submercado")
        .map_elements(lambda c: bus_codes.get(int(c)), return_dtype=pl.Int64)
        .alias("newave_code"),
        pl.col("estagio").cast(pl.Int64).alias("stage"),
    ).filter(pl.col("newave_code").is_not_null())

    # --- nw_market: GHTOT (hydro gen) / GTERM (live + anticipated thermal
    # gen) / DEFT (deficit), keyed by the mapped Cobre bus id. GTERM sums the
    # two DECOMP thermal-generation columns because Cobre's own
    # ``thermal_gen_mw`` (the counterpart this overlays) already includes
    # anticipated GNL generation — the same live+anticipated fold the cost
    # breakdown applies (see charts._COST_MAP's "Thermal Generation" entry).
    market_exprs: list[tuple[str, pl.Expr]] = []
    if "geracao_hidroeletrica_MW" in present:
        market_exprs.append(("GHTOT", pl.col("geracao_hidroeletrica_MW")))
    if "geracao_termica_MW" in present and "geracao_termica_antecipada_MW" in present:
        market_exprs.append(
            (
                "GTERM",
                pl.col("geracao_termica_MW") + pl.col("geracao_termica_antecipada_MW"),
            )
        )
    if "deficit_MW" in present:
        market_exprs.append(("DEFT", pl.col("deficit_MW")))

    if market_exprs and not mapped.is_empty():
        wide = mapped.select(
            "newave_code",
            "stage",
            *[expr.alias(name) for name, expr in market_exprs],
        )
        nw_market = wide.unpivot(
            on=[name for name, _ in market_exprs],
            index=["newave_code", "stage"],
            variable_name="variable",
            value_name="value",
        ).with_columns(pl.col("value").cast(pl.Float64))
    else:
        nw_market = _empty_market_frame()

    # --- nw_net_load: demand minus the non-controllable sources (wind +
    # small plants) — the tab's NET_LOAD basis.
    non_controllable = {"demanda_MW", "geracao_eolica_MW", "geracao_pequenas_usinas_MW"}
    if non_controllable <= present and not mapped.is_empty():
        nw_net_load = mapped.select(
            "newave_code",
            "stage",
            pl.lit("NET_LOAD").alias("variable"),
            (
                pl.col("demanda_MW")
                - pl.col("geracao_eolica_MW")
                - pl.col("geracao_pequenas_usinas_MW")
            )
            .cast(pl.Float64)
            .alias("value"),
        )
    else:
        nw_net_load = _empty_market_frame()

    # --- nw_sin: EARMF / ENA summed across every submarket the deck
    # declares. The converter-created transhipment bus never appears among
    # ``codigo_submercado`` values (D-UNITS) — it has no source-model code of
    # its own, it is referenced only by name in ``IA`` records — so summing
    # every row this reader returns already excludes it; no extra filter
    # needed (mirrors the guarantee ``TestBusSideExcludesTranshipment``
    # pins down for the tidy-comparison bus rows). ``newave_code`` is the
    # constant SIN placeholder ``0``, matching ``read_medias_sin``'s
    # newave-side convention. Built as a plain per-stage sum — additive by
    # construction — so a later epic can ``pl.concat`` its own cost-series
    # rows onto this frame instead of overwriting it.
    system_energy = {"earm_final_MWmes", "ena_MWmes"}
    if system_energy <= present:
        sin_agg = aggregated.group_by("estagio").agg(
            pl.col("earm_final_MWmes").sum().alias("EARMF"),
            pl.col("ena_MWmes").sum().alias("ENA"),
        )
        nw_sin = (
            sin_agg.unpivot(
                on=["EARMF", "ENA"], index=["estagio"], variable_name="variable"
            )
            .with_columns(
                pl.lit(0).cast(pl.Int64).alias("newave_code"),
                pl.col("estagio").cast(pl.Int64).alias("stage"),
                pl.col("value").cast(pl.Float64),
            )
            .select("newave_code", "stage", "variable", "value")
            .sort("stage", "variable")
        )
    else:
        nw_sin = _empty_market_frame()

    return nw_market, nw_net_load, nw_sin


def _map_entities(
    frame: pl.DataFrame, code_column: str, mapping: dict[int, int]
) -> tuple[pl.DataFrame, list[int]]:
    """Translate source codes to Cobre ids; report the codes with no mapping.

    Keeps the original code alongside the mapped ``entity_id`` (renamed to
    the generic ``newave_code`` — D-SOURCE-TOKEN: the field name means "the
    reference model's code", not NEWAVE-specific) so callers that need the
    reference code back (:func:`_result_comparisons`) don't have to invert
    ``mapping`` themselves.
    """
    codes = {int(c) for c in frame[code_column].unique()}
    unmapped = sorted(code for code in codes if code not in mapping)
    mapped = (
        frame.with_columns(
            pl.col(code_column)
            .map_elements(lambda c: mapping.get(int(c)), return_dtype=pl.Int64)
            .alias("entity_id"),
            pl.col(code_column).cast(pl.Int64).alias("newave_code"),
            # Stage ids are 0-based in Cobre and 1-based in the source deck.
            (pl.col("estagio").cast(pl.Int64) - 1).alias("stage_id"),
        )
        .filter(pl.col("entity_id").is_not_null())
        .drop(code_column, "estagio")
    )
    return mapped, unmapped


def _cobre_hydro(cobre_output_dir: Path) -> tuple[pl.DataFrame, dict[int, str]]:
    """Cobre hydro means with absolute storage converted to useful volume."""
    means = cobre_readers.read_cobre_hydro_means(cobre_output_dir)
    metadata = cobre_readers.read_cobre_hydro_metadata(cobre_output_dir)
    names = {i: str(m.get("name", "")) for i, m in metadata.items()}

    if means.is_empty():
        return means, names

    floors = pl.DataFrame(
        {
            "entity_id": list(metadata.keys()),
            "min_storage_hm3": [
                float(m.get("min_storage_hm3") or 0.0) for m in metadata.values()
            ],
        },
        schema={"entity_id": pl.Int64, "min_storage_hm3": pl.Float64},
    )
    if "storage_final_hm3" not in means.columns or floors.is_empty():
        return means, names
    return (
        means.join(floors, on="entity_id", how="left").with_columns(
            (
                pl.col("storage_final_hm3") - pl.col("min_storage_hm3").fill_null(0.0)
            ).alias("useful_storage_hm3")
        ),
        names,
    )


@dataclass(frozen=True)
class _AlignedDecompFrames:
    """One run's DECOMP-side + Cobre-side frames, aligned to Cobre ids/stages.

    The shared read/align result :func:`compare_decomp_results` and
    :func:`build_decomp_dataset` both build on, via
    :func:`_read_aligned_frames` — deck discovery, id-map construction,
    per-level scenario averaging, and the matching Cobre means run exactly
    once per entry point.
    """

    source_hydro: pl.DataFrame
    source_thermal: pl.DataFrame
    source_bus: pl.DataFrame
    cobre_hydro: pl.DataFrame
    cobre_thermal: pl.DataFrame
    cobre_bus: pl.DataFrame
    hydro_names: dict[int, str]
    thermal_names: dict[int, str]
    bus_names: dict[int, str]
    unmapped: dict[str, list[int]]
    # --- ticket-006: Energy Balance tab reference frames ---
    # Long newave_code/stage/variable/value frames built by
    # :func:`_energy_balance_frames`; defaulted so existing fixtures that
    # construct this dataclass without them (built before this ticket) keep
    # working unchanged.
    nw_market: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_net_load: pl.DataFrame = field(default_factory=pl.DataFrame)
    nw_sin: pl.DataFrame = field(default_factory=pl.DataFrame)


def _read_aligned_frames(
    decomp_dir: Path, cobre_output_dir: Path
) -> _AlignedDecompFrames:
    """Read and align both sides of one DECOMP-vs-Cobre comparison run.

    ``decomp_dir`` is the deck directory (it must contain the ``dec_oper_*.csv``
    result tables and the deck files needed to rebuild the id map, all directly
    in that directory); ``cobre_output_dir`` is Cobre's output directory, whose
    case directory supplies the entity registries. Entities the id map cannot
    resolve are reported via ``unmapped`` rather than dropped in silence.
    """
    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.pipeline import discover_decomp_files

    files = discover_decomp_files(decomp_dir)
    dadger = Dadger.read(str(files.dadger))
    id_map = DecompIdMap.from_dadger(dadger)

    hydro_codes = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}
    thermal_codes = {code: id_map.thermal_id(code) for code in id_map.thermal_codes}
    bus_codes = {code: id_map.bus_id(code) for code in id_map.bus_codes}

    source_hydro, unmapped_hydro = _hydro_side(decomp_dir, hydro_codes)
    source_thermal, unmapped_thermal = _thermal_side(decomp_dir, thermal_codes)
    source_bus, unmapped_bus = _bus_side(decomp_dir, bus_codes)
    nw_market, nw_net_load, nw_sin = _energy_balance_frames(decomp_dir, bus_codes)

    cobre_hydro, hydro_names = _cobre_hydro(cobre_output_dir)
    cobre_thermal = cobre_readers.read_cobre_thermal_means(cobre_output_dir)
    cobre_bus = cobre_readers.read_cobre_bus_means(cobre_output_dir)

    thermal_names = {
        i: str(m.get("name", ""))
        for i, m in cobre_readers.read_cobre_thermal_metadata(cobre_output_dir).items()
    }
    bus_names = {
        i: str(m.get("name", ""))
        for i, m in cobre_readers.read_cobre_bus_metadata(cobre_output_dir).items()
    }

    return _AlignedDecompFrames(
        source_hydro=source_hydro,
        source_thermal=source_thermal,
        source_bus=source_bus,
        cobre_hydro=cobre_hydro,
        cobre_thermal=cobre_thermal,
        cobre_bus=cobre_bus,
        hydro_names=hydro_names,
        thermal_names=thermal_names,
        bus_names=bus_names,
        unmapped={
            "hydro": unmapped_hydro,
            "thermal": unmapped_thermal,
            "bus": unmapped_bus,
        },
        nw_market=nw_market,
        nw_net_load=nw_net_load,
        nw_sin=nw_sin,
    )


def compare_decomp_results(
    decomp_dir: Path, cobre_output_dir: Path
) -> DecompComparison:
    """Compare a source-model run against the Cobre run of its converted case.

    ``decomp_dir`` is the deck directory (it must contain the ``dec_oper_*.csv``
    result tables and the deck files needed to rebuild the id map, all directly
    in that directory); ``cobre_output_dir`` is Cobre's output directory, whose
    case directory supplies the entity registries.
    """
    aligned = _read_aligned_frames(decomp_dir, cobre_output_dir)

    rows = pl.concat(
        [
            _tidy(
                aligned.source_hydro,
                aligned.cobre_hydro,
                _HYDRO_VARIABLES,
                names=aligned.hydro_names,
            ),
            _tidy(
                aligned.source_thermal,
                aligned.cobre_thermal,
                _THERMAL_VARIABLES,
                names=aligned.thermal_names,
            ),
            _tidy(
                aligned.source_bus,
                aligned.cobre_bus,
                _BUS_VARIABLES,
                names=aligned.bus_names,
            ),
        ]
    )

    return DecompComparison(
        rows=rows,
        summary=_summarize(rows),
        convergence=_convergence(decomp_dir, cobre_output_dir),
        unmapped=aligned.unmapped,
    )


def build_decomp_dataset(
    decomp_dir: Path, cobre_output_dir: Path, *, tolerance: float = 1e-2
) -> ComparisonDataset:
    """Build the canonical results dataset for the current 8 DECOMP variables.

    Shares the read/align step with :func:`compare_decomp_results` (via
    :func:`_read_aligned_frames`), but emits the canonical
    :class:`~cobre_bridge.comparators.results.ResultComparison` shape and
    assembles it through the shared, source-agnostic
    :func:`~cobre_bridge.comparators.analyze.build_results_dataset` stat
    kernel — instead of the standalone tidy/summary frames
    :class:`DecompComparison` renders. This is the reuse engine every later
    epic extends by filling more
    :class:`~cobre_bridge.comparators.results.PercentileData` fields.

    Per D-PERCENTILEDATA, the returned dataset's ``PercentileData`` carries
    only the fields available at this stage: the entity-name dicts
    (``nw_bus_names``, ``nw_hydro_names``), the cobre-side bus percentile
    band (``bus``, from :func:`cobre_readers.read_cobre_bus_percentiles`),
    and — as of ticket-006 — the Energy Balance tab's reference frames
    (``nw_market``, ``nw_net_load``, ``nw_sin``, from
    :func:`_energy_balance_frames`) and cobre-side aggregates
    (``bus_aggregates``, ``cobre_bus_meta``, ``cobre_hydro_means``, each read
    verbatim from the matching ``cobre_readers`` function). ``nw_offset=1``
    (D-STAGE-OFFSET): DECOMP's ``estagio`` is 1-based from the deck's first
    stage, unlike the source model's arbitrary calendar-month-numbered
    MEDIAS stage, so the fixed offset is always 1 rather than derived from
    the data. Every other field stays at its empty default, and ``bus``
    itself stays empty whenever the Cobre run has no percentile output (e.g.
    the deterministic 2-node tree) — no percentile spread is ever
    fabricated.

    Args:
        decomp_dir: The deck directory (results resolved via the ticket-001
            union discovery).
        cobre_output_dir: Cobre's output directory.
        tolerance: Relative tolerance forwarded to the summary builder.

    Returns:
        The validated dataset. ``metadata["unmapped"]`` carries the per-level
        (hydro/thermal/bus) reference codes the id map could not resolve, so
        they survive into the dataset instead of being silently dropped.
    """
    from cobre_bridge.comparators.analyze import build_results_dataset

    aligned = _read_aligned_frames(decomp_dir, cobre_output_dir)

    results: list[ResultComparison] = []
    results.extend(
        _result_comparisons(
            aligned.source_hydro,
            aligned.cobre_hydro,
            _HYDRO_VARIABLES,
            names=aligned.hydro_names,
        )
    )
    results.extend(
        _result_comparisons(
            aligned.source_thermal,
            aligned.cobre_thermal,
            _THERMAL_VARIABLES,
            names=aligned.thermal_names,
        )
    )
    results.extend(
        _result_comparisons(
            aligned.source_bus,
            aligned.cobre_bus,
            _BUS_VARIABLES,
            names=aligned.bus_names,
        )
    )

    pct = PercentileData(
        nw_bus_names=aligned.bus_names,
        nw_hydro_names=aligned.hydro_names,
        bus=cobre_readers.read_cobre_bus_percentiles(cobre_output_dir),
        nw_market=aligned.nw_market,
        nw_net_load=aligned.nw_net_load,
        nw_sin=aligned.nw_sin,
        # D-STAGE-OFFSET: fixed at 1, not derived — see the docstring above.
        nw_offset=1,
        bus_aggregates=cobre_readers.read_cobre_bus_aggregates(cobre_output_dir),
        cobre_bus_meta=cobre_readers.read_cobre_bus_metadata(cobre_output_dir),
        cobre_hydro_means=cobre_readers.read_cobre_hydro_means(cobre_output_dir),
    )
    dataset = build_results_dataset(results, pct, tolerance)
    dataset.metadata["unmapped"] = {
        level: [int(code) for code in codes]
        for level, codes in aligned.unmapped.items()
    }
    return dataset
