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

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl

from cobre_bridge.cobre_io import case_dir_for
from cobre_bridge.comparators import cobre_readers
from cobre_bridge.comparators.decomp_readers import (
    read_dec_desvfpha,
    read_dec_estatfpha,
    read_dec_oper_evap,
    read_dec_oper_interc,
    read_dec_oper_ree,
    read_dec_oper_rhesoft,
    read_dec_oper_sist,
    read_dec_oper_usih,
    read_dec_oper_usit,
    read_decomp_tim,
    read_eco_fpha,
    read_relato_convergence,
    read_relato_costs,
    read_relato_membership,
    reconcile_kdollars_to_reais,
)
from cobre_bridge.comparators.results import PercentileData, ResultComparison
from cobre_bridge.diagnostics import Diagnostic, Severity, emit

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
        ConstraintTerm,
    )
    from cobre_bridge.decomp.id_map import DecompIdMap

_LOG = logging.getLogger(__name__)

#: Rows whose two values are both below this magnitude are treated as an exact
#: match rather than a division by ~0 (physical units: MW, m³/s, hm³).
_ZERO_FLOOR = 1e-9

#: Minimum turbined flow (m³/s) for the derived gen/turbined productivity
#: (ticket-016) to be meaningful -- mirrors
#: :data:`cobre_bridge.comparators.results._PRODUCTIVITY_TURB_EPS`: near-zero
#: turbining makes generation/turbined an undefined 0/0 on both sides.
_PRODUCTIVITY_TURBINED_EPS: float = 1.0e-6

#: ticket-018: cobre's ``stored_energy_final_mwh`` is MWh; the source model's
#: own REE ``earm_final_MWmes`` is MWmês (average MW sustained over a month).
#: 730 h/month is the same implicit hours-per-month convention
#: `report_builder`'s Energy Balance chart already applies to the SIN-total
#: EARM overlay (``cobre_aggregate_chart(..., nw_factor=730.0, ...)``) --
#: reused here rather than re-derived so the two EARM reconciliations never
#: drift apart.
_EARM_MWH_TO_MWMES: float = 730.0


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
    frame: pl.DataFrame,
    key: str,
    columns: list[str],
    *,
    entity_column: str | list[str] | None,
) -> pl.DataFrame:
    """Average the stage rows over the source model's nodes.

    ``key`` is the stage column; ``entity_column`` narrows the grouping to one
    entity per stage: ``None`` for tables that are already one row per stage,
    a single column name for a single-code entity, or a list of column names
    for a composite key -- e.g. an interchange corridor's
    ``(codigo_submercado_de, codigo_submercado_para)`` pair (ticket-007).
    """
    present = [c for c in columns if c in frame.columns]
    if entity_column is None:
        entity_columns: list[str] = []
    elif isinstance(entity_column, str):
        entity_columns = [entity_column]
    else:
        entity_columns = list(entity_column)
    group = [key, *entity_columns]
    return (
        frame.group_by(group)
        .agg([pl.col(c).mean().alias(c) for c in present])
        .sort(group)
    )


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


def _result_comparisons(
    source: pl.DataFrame,
    cobre: pl.DataFrame,
    variables: tuple[_Variable, ...],
    *,
    names: dict[int, str],
) -> list[ResultComparison]:
    """Join one level's two frames into ``ResultComparison`` rows.

    Emits the canonical
    :class:`~cobre_bridge.comparators.results.ResultComparison` shape
    :func:`build_decomp_dataset` hands to
    :func:`cobre_bridge.comparators.analyze.build_results_dataset`, keyed by
    :data:`_CANONICAL_VARIABLE`.
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


def _hydro_productivity_results(
    hydro_results: Sequence[ResultComparison],
) -> list[ResultComparison]:
    """Derive per-(plant, stage) realized productivity = generation / turbined.

    ticket-016: consumes the ``generation_mw``/``turbined_m3s`` hydro
    ``ResultComparison`` rows :func:`_result_comparisons` already produced
    (E1: the source values come from ``dec_oper_usih``'s stage-aggregate
    rows, the Cobre values from the cobre hydro means, joined on the
    id-map-resolved ``(entity_id, stage_id)`` pair) -- so this reuses that
    exact alignment/restriction instead of a separate lookup or filter.
    Mirrors :mod:`cobre_bridge.comparators.results`'s own ``_compare_hydros``
    productivity derivation. TRACKED zero-guard semantics: a plant/stage
    where either side's turbined flow is at/near zero -- or one side's row is
    simply absent -- is DROPPED, never null-kept, matching the NEWAVE-side
    behaviour for the exact same widget (gen/turbined is an undefined 0/0
    there).

    Returns a list meant to be appended onto the caller's main ``results``
    list, not held separately: there is no dedicated ``PercentileData``
    field for the per-stage productivity frame --
    :func:`cobre_bridge.comparators.analyze.build_results_dataset` derives
    ``dataset.metadata["productivity_per_stage"]`` by filtering the *same*
    ``results`` list it was handed for ``entity_type == "hydro"`` /
    ``variable == "productivity_mw_per_m3s"`` rows.
    """
    generation: dict[tuple[int, int], tuple[float, float, int, str]] = {}
    turbined: dict[tuple[int, int], tuple[float, float]] = {}
    for r in hydro_results:
        if r.entity_type != "hydro":
            continue
        key = (r.newave_code, r.stage)
        if r.variable == "generation_mw":
            generation[key] = (
                r.newave_value,
                r.cobre_value,
                r.cobre_id,
                r.entity_name,
            )
        elif r.variable == "turbined_m3s":
            turbined[key] = (r.newave_value, r.cobre_value)

    productivity: list[ResultComparison] = []
    for key, (nw_gen, cb_gen, cobre_id, name) in sorted(generation.items()):
        turb = turbined.get(key)
        if turb is None:
            continue
        nw_turb, cb_turb = turb
        # Zero-guard: drop (never null-keep) a plant/stage where either side
        # turbines at/near zero -- gen/turbined is an undefined 0/0 there.
        if (
            nw_turb <= _PRODUCTIVITY_TURBINED_EPS
            or cb_turb <= _PRODUCTIVITY_TURBINED_EPS
        ):
            continue
        nw_code, stage = key
        nw_value = nw_gen / nw_turb
        cobre_value = cb_gen / cb_turb
        abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
        productivity.append(
            ResultComparison(
                entity_type="hydro",
                entity_name=name,
                newave_code=nw_code,
                cobre_id=cobre_id,
                stage=stage,
                variable="productivity_mw_per_m3s",
                newave_value=nw_value,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return productivity


#: Canonical convergence-chart schema -- matches
#: :func:`cobre_readers.read_cobre_convergence`'s own return schema exactly,
#: so :func:`~cobre_bridge.comparators.charts.convergence_chart` can read
#: either side without a source-specific branch.
_CONVERGENCE_SCHEMA: dict[str, type[pl.DataType]] = {
    "iteration": pl.Int64,
    "lower_bound": pl.Float64,
    "upper_bound_mean": pl.Float64,
}


def _decomp_convergence_frame(decomp_dir: Path) -> pl.DataFrame:
    """DECOMP-side convergence bounds in the canonical chart schema.

    ticket-012: renames ``relato.convergencia``'s ``iteracao``/``zinf``/
    ``zsup`` onto the ``iteration``/``lower_bound``/``upper_bound_mean``
    columns :func:`cobre_readers.read_cobre_convergence` emits -- the shape
    :func:`~cobre_bridge.comparators.charts.convergence_chart` reads on each
    side separately (not the legacy :func:`_convergence` join shape). The
    iteration axis stays 1-based as reported, no offset -- the chart plots
    it directly.

    A missing/empty relato (``read_relato_convergence`` raising
    ``FileNotFoundError``/``ValueError``) degrades to an empty frame with
    this same canonical schema instead of aborting
    :func:`build_decomp_dataset` -- the chart then renders Cobre-only.
    """
    try:
        source = read_relato_convergence(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.warning("No source-model convergence table: %s", exc)
        return pl.DataFrame(schema=_CONVERGENCE_SCHEMA)

    columns = {c.lower(): c for c in source.columns}
    return source.select(
        pl.col(columns["iteracao"]).cast(pl.Int64).alias("iteration"),
        pl.col(columns["zinf"]).cast(pl.Float64).alias("lower_bound"),
        pl.col(columns["zsup"]).cast(pl.Float64).alias("upper_bound_mean"),
    )


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


# --- ticket-007: interchange corridor -> cobre line alignment ---
#
# DECOMP publishes exchange per submarket-pair corridor (``dec_oper_interc``);
# the converted Cobre case publishes exchange per directed line
# (``system/lines.json``, read into simulation results by
# ``cobre_readers.read_cobre_line_means`` as ``entity_id``/``stage_id``/
# ``net_flow_mw``). The two topologies usually coincide (one deck-declared
# exchange pair -> one cobre line), but the converter-created transhipment
# bus (``DecompIdMap.transhipment_bus_id``) can realize a corridor as a pair
# of legs instead of a single line -- see epic-03's "Modeling nuance"
# section. This block reconciles the two; ticket-008 turns the result into
# ``ResultComparison`` line rows and the Network tab metadata.
#
# Facts confirmed against the live code (not assumed from the ticket prose):
# - ``system/lines.json`` line entries key their endpoints
#   ``source_bus_id``/``target_bus_id`` (``decomp/network.py::convert_lines``,
#   ``converters/network.py::convert_lines``) -- not ``from_bus``/``to_bus``.
# - A cobre line's ``net_flow_mw`` is positive in its own
#   ``source_bus_id -> target_bus_id`` direction: ``comparators/alignment.py``'s
#   ``LineEntity`` docstring states it explicitly ("Cobre models each
#   normalized pair as a single line where positive flow goes from
#   source_bus to target_bus"), and ``dashboard/tabs/network.py``'s bus
#   balance credits the *target* bus and debits the *source* bus with the
#   same ``net_flow_mw`` value, which only holds under that convention.
# - ``dec_oper_interc``'s real columns (``idecomp`` 1.14,
#   ``idecomp.decomp.dec_oper_interc.DecOperInterc.tabela``) are
#   ``codigo_submercado_de``/``codigo_submercado_para`` (not
#   ``submercado_de``/``submercado_para``), plus ``intercambio_origem_MW``,
#   ``intercambio_destino_MW``, and ``perdas_MW``.


def _read_cobre_lines_index(cobre_output_dir: Path) -> dict[tuple[int, int], int]:
    """``{(source_bus_id, target_bus_id): line_id}`` from ``system/lines.json``.

    A missing ``system/lines.json`` -- e.g. a case predating the ``IA``
    exchange-network converter (``decomp/network.py::convert_lines_placeholder``)
    -- yields an empty index rather than raising: :func:`_corridor_line_alignment`
    then resolves nothing, and :func:`_interc_side` reports every corridor as
    unresolved instead of failing the comparison (the Network tab degrades to
    empty).

    [ASSUMPTION] (epic-03 Pending Decision, star topology): more than one
    line declared between the *same ordered* bus pair makes that leg
    ambiguous -- both are dropped from the index so any corridor that would
    resolve through it is reported unresolved rather than guessing which
    line applies.
    """
    lines_path = case_dir_for(cobre_output_dir) / "system" / "lines.json"
    if not lines_path.exists():
        return {}
    with lines_path.open() as f:
        data = json.load(f)

    index: dict[tuple[int, int], int] = {}
    ambiguous: set[tuple[int, int]] = set()
    for line in data.get("lines", []):
        key = (int(line["source_bus_id"]), int(line["target_bus_id"]))
        if key in index:
            ambiguous.add(key)
            continue
        index[key] = int(line["id"])
    for key in ambiguous:
        index.pop(key, None)
        _LOG.warning(
            "Multiple cobre lines declared for bus pair %s; excluding it from "
            "the corridor alignment instead of guessing which line applies",
            key,
        )
    return index


def _resolve_leg(
    lines_index: dict[tuple[int, int], int], bus_a: int, bus_para: int
) -> tuple[int, int] | None:
    """Return the ``(line_id, sign)`` leg realizing flow ``bus_a -> bus_para``.

    ``sign`` orients the cobre line's own ``source_bus_id -> target_bus_id``
    convention onto the requested direction: ``+1`` when a line already runs
    ``bus_a -> bus_para``, ``-1`` when the matching line runs the reverse.
    ``None`` when no cobre line connects the pair in either direction.
    """
    line_id = lines_index.get((bus_a, bus_para))
    if line_id is not None:
        return line_id, 1
    line_id = lines_index.get((bus_para, bus_a))
    if line_id is not None:
        return line_id, -1
    return None


def _corridor_line_alignment(
    cobre_output_dir: Path, id_map: DecompIdMap
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Map every resolvable DECOMP corridor onto its cobre line leg(s).

    Keyed by cobre bus-id pair ``(bus_de, bus_para)`` -- built over every
    ordered pair of *declared* subsystem buses (the transhipment bus is
    never a corridor endpoint in its own right, D-UNITS) -- to the ordered
    list of cobre ``(line_id, sign)`` legs realizing it:

    - a single leg when a direct cobre line connects the pair, in either
      direction (requirement 3's "direct" branch);
    - else the two-leg ``bus_de -> transhipment_bus_id -> bus_para`` path,
      when both legs exist (requirement 3's transhipment branch);
    - else the pair is simply absent from the returned mapping.

    :func:`_corridor_line_alignment` itself never raises and never reports:
    it is total in the sense of trying every declared bus pair, but a pair
    with no realizing line(s) is left out of the dict rather than tracked
    here. :func:`_interc_side` is the layer that turns "absent from this
    mapping" into a reported, never-silently-dropped unresolved corridor,
    scoped to the corridors ``dec_oper_interc`` actually carries.

    [ASSUMPTION] (record per epic-03 Pending Decision): the two-leg
    reconciliation assumes the converted case's exchange topology is a star
    through the transhipment bus, with at most one ``de<->IV<->para`` path
    per corridor -- guaranteed structurally here since ``DecompIdMap`` has
    exactly one transhipment bus and :func:`_read_cobre_lines_index` drops
    (rather than arbitrarily picks) any bus pair wired by more than one
    line. Alternatives considered: (a) direct submarket-pair match only --
    fails on the star topology, rejected; (b) transhipment-aware two-leg
    reconciliation -- chosen; (c) compare at the bus-injection level instead
    of per line -- defers to a future ticket, out of scope. A deck whose
    exchange graph is genuinely non-star (e.g. two different corridors that
    would need to share one leg) is not guessed at by this function -- each
    corridor is resolved independently, so a shared leg is assigned the same
    value by both corridors rather than an aggregate; that residual gap is a
    known limitation of the star assumption, not something this ticket
    reconciles further.
    """
    lines_index = _read_cobre_lines_index(cobre_output_dir)
    if not lines_index:
        return {}

    transhipment = id_map.transhipment_bus_id
    real_bus_ids = sorted({id_map.bus_id(code) for code in id_map.bus_codes})

    alignment: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for bus_de in real_bus_ids:
        for bus_para in real_bus_ids:
            if bus_de == bus_para:
                continue
            direct = _resolve_leg(lines_index, bus_de, bus_para)
            if direct is not None:
                alignment[(bus_de, bus_para)] = [direct]
                continue
            leg_in = _resolve_leg(lines_index, bus_de, transhipment)
            leg_out = _resolve_leg(lines_index, transhipment, bus_para)
            if leg_in is not None and leg_out is not None:
                alignment[(bus_de, bus_para)] = [leg_in, leg_out]
    return alignment


def _interc_side(
    decomp_dir: Path, cobre_output_dir: Path, id_map: DecompIdMap
) -> tuple[pl.DataFrame, list[tuple[int, int]]]:
    """Per-(cobre ``line_id``, 0-based stage) DECOMP interchange net-flow.

    Folds ``dec_oper_interc`` the way every other ``_*_side`` helper folds
    its table -- :func:`_stage_rows`'s patamar-null aggregate, then
    :func:`_scenario_mean` averaged over the source model's nodes, here keyed
    by the corridor's composite ``(codigo_submercado_de,
    codigo_submercado_para)`` pair rather than a single entity code -- aligns
    each corridor onto its cobre line leg(s) via
    :func:`_corridor_line_alignment`, and orients the net-flow value to each
    leg's ``sign``. ``estagio - 1`` rebases the 1-based DECOMP stage the same
    way :func:`_map_entities` does for every other level.

    The net-flow value is ``intercambio_origem_MW`` -- the flow measured at
    the corridor's origin, already in its own ``de -> para`` direction.
    ``perdas_MW`` (transmission losses) is read as part of the same fold but
    intentionally NOT subtracted from it: a cobre line has no loss model (its
    ``net_flow_mw`` is a single, lossless figure -- see
    ``decomp/network.py::convert_lines``), so the origin-side reading, not
    the post-loss ``intercambio_destino_MW``, is the one quantity comparable
    to it.

    A corridor whose endpoints the id map cannot resolve, or whose resolved
    bus pair has no entry in :func:`_corridor_line_alignment`'s mapping
    (including every corridor when ``system/lines.json`` is absent), is
    collected into the returned unresolved list as
    ``(codigo_submercado_de, codigo_submercado_para)`` code pairs --
    mirroring :func:`_map_entities`'s ``unmapped`` convention -- and excluded
    from the returned frame. It is reported, never silently dropped.

    The source model reports every physical interface as two corridor rows,
    one per direction (e.g. both ``SE -> S`` and ``S -> SE``); both align
    onto the very same cobre line leg with opposite ``sign``. The
    per-``(entity_id, stage_id)`` rows built above are therefore grouped and
    summed before being returned, collapsing each pair down to one row equal
    to ``flow(de -> para) - flow(para -> de)`` -- exactly the net figure a
    lossless cobre line's own ``net_flow_mw`` represents. The same grouping
    also nets more than one corridor's contribution onto a shared two-leg
    transhipment leg (the star-topology "shared leg" case) instead of
    colliding as duplicate ``(entity_id, stage_id)`` keys.
    """
    empty = pl.DataFrame(
        schema={
            "entity_id": pl.Int64,
            "stage_id": pl.Int64,
            "net_flow_mw": pl.Float64,
        }
    )
    frame = _stage_rows(read_dec_oper_interc(decomp_dir))
    aggregated = _scenario_mean(
        frame,
        "estagio",
        ["intercambio_origem_MW", "perdas_MW"],
        entity_column=["codigo_submercado_de", "codigo_submercado_para"],
    )
    if aggregated.is_empty():
        return empty, []

    alignment = _corridor_line_alignment(cobre_output_dir, id_map)

    unresolved: set[tuple[int, int]] = set()
    rows: list[dict[str, float | int]] = []
    for row in aggregated.iter_rows(named=True):
        de_raw = row["codigo_submercado_de"]
        para_raw = row["codigo_submercado_para"]
        if de_raw is None or para_raw is None:
            # The deck's own ``dec_oper_interc`` reports some corridors --
            # the Itaipu 50 Hz "IV" node, in particular -- by
            # ``nome_submercado_*`` only: "IV" carries no ``SB``-declared
            # numeric code, so ``codigo_submercado_de``/``_para`` comes back
            # null on that side. Such a corridor can never resolve through
            # ``_corridor_line_alignment`` regardless of how it is looked up
            # -- the transhipment bus is never an outer-loop endpoint there
            # (D-UNITS) -- so report it with the ``-1`` "no SB code" sentinel
            # instead of crashing on the ``int()`` cast below.
            unresolved.add(
                (
                    int(de_raw) if de_raw is not None else -1,
                    int(para_raw) if para_raw is not None else -1,
                )
            )
            continue
        de_code = int(de_raw)
        para_code = int(para_raw)
        try:
            bus_de = id_map.bus_id(de_code)
            bus_para = id_map.bus_id(para_code)
        except KeyError:
            unresolved.add((de_code, para_code))
            continue

        legs = alignment.get((bus_de, bus_para))
        if legs is None:
            unresolved.add((de_code, para_code))
            continue

        stage_id = int(row["estagio"]) - 1
        net_flow = float(row["intercambio_origem_MW"])
        for line_id, sign in legs:
            rows.append(
                {
                    "entity_id": line_id,
                    "stage_id": stage_id,
                    "net_flow_mw": sign * net_flow,
                }
            )

    if not rows:
        return empty, sorted(unresolved)
    result = (
        pl.DataFrame(rows)
        .group_by("entity_id", "stage_id")
        .agg(pl.col("net_flow_mw").sum())
        .cast({"entity_id": pl.Int64, "stage_id": pl.Int64, "net_flow_mw": pl.Float64})
        .sort("entity_id", "stage_id")
    )
    return result, sorted(unresolved)


# --- ticket-008: line ResultComparison rows + Network tab metadata ---
#
# Wires ticket-007's corridor alignment into ``build_decomp_dataset``: line
# ``ResultComparison`` rows (mirroring ``results._compare_lines``'s emitted
# shape) plus the three ``PercentileData`` fields the shared Network tab
# (``report_builder``, ``charts.line_summary_chart``) consumes.


def _line_entity_names(line_meta: list[dict], id_map: DecompIdMap) -> dict[int, str]:
    """Display name per cobre line id, straight from ``lines.json``'s own
    ``name`` field (built by ``decomp/network.py::convert_lines`` as
    ``f"{pair[0]}-{pair[1]}"``).

    Falls back to a bus-name pair label when an entry carries no usable
    ``name`` -- defensive; every line the converter emits does carry one --
    so a line's ``entity_name`` is never blank.
    """
    names: dict[int, str] = {}
    for entry in line_meta:
        line_id = int(entry["id"])
        name = str(entry.get("name") or "").strip()
        if not name:
            try:
                src_name = id_map.bus_name(int(entry["source_bus_id"]))
                tgt_name = id_map.bus_name(int(entry["target_bus_id"]))
                name = f"{src_name}<->{tgt_name}"
            except KeyError:
                name = f"line_{line_id}"
        names[line_id] = name
    return names


def _build_line_id_map(decomp_dir: Path) -> DecompIdMap | None:
    """Best-effort id map for the corridor -> line alignment.

    Mirrors :func:`_read_aligned_frames`'s own id-map construction (deck
    discovery -> ``Dadger.read`` -> :meth:`DecompIdMap.from_dadger`), but
    degrades to ``None`` on a missing/invalid deck instead of failing the
    whole dataset build: the Network tab is one optional section among many
    (ticket-008's Error Handling), and every fixture that exercises
    ``build_decomp_dataset`` against a bare directory (every other level's
    own tests) must keep working unchanged.
    """
    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.pipeline import discover_decomp_files

    try:
        files = discover_decomp_files(decomp_dir)
        dadger = Dadger.read(str(files.dadger))
        return DecompIdMap.from_dadger(dadger)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No deck available for line alignment: %s", exc)
        return None


def _line_result_comparisons(
    decomp_dir: Path,
    cobre_output_dir: Path,
    id_map: DecompIdMap | None,
    line_meta: list[dict],
) -> tuple[list[ResultComparison], list[list[int]]]:
    """Join ticket-007's aligned DECOMP line flow onto Cobre's per-line means.

    Mirrors :func:`_result_comparisons`'s join shape, but keyed by cobre line
    id rather than a source-model entity code: a single cobre line can be the
    shared leg of more than one corridor under the star topology (epic-03's
    two-leg transhipment path), so ``newave_code`` -- unlike every other
    level's rows -- carries no single canonical source-model code; it stays
    at the constant ``0`` placeholder, mirroring the SIN row's own
    convention (:func:`_energy_balance_frames`) rather than fabricating one.

    Returns ``([], [])`` when the deck could not be read (*id_map* is
    ``None``) or the source-model interchange table is unavailable, and
    ``([], unresolved)`` when corridors resolve but the join with Cobre's own
    per-line means produces no row -- matching :func:`_interc_side`'s own
    never-silently-drop convention for unresolved corridors.
    """
    if id_map is None:
        return [], []
    try:
        source_lines, unresolved = _interc_side(decomp_dir, cobre_output_dir, id_map)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model interchange table for line alignment: %s", exc)
        return [], []

    unresolved_lists = [list(pair) for pair in unresolved]
    cobre_lines = cobre_readers.read_cobre_line_means(cobre_output_dir)
    if source_lines.is_empty() or cobre_lines.is_empty():
        return [], unresolved_lists

    names = _line_entity_names(line_meta, id_map)
    joined = (
        source_lines.rename({"net_flow_mw": "source_net_flow_mw"})
        .join(cobre_lines, on=["entity_id", "stage_id"], how="inner")
        .sort("entity_id", "stage_id")
    )

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        line_id = int(row["entity_id"])
        nw_value = float(row["source_net_flow_mw"])
        cobre_value = float(row["net_flow_mw"])
        abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
        results.append(
            ResultComparison(
                entity_type="line",
                entity_name=names.get(line_id, f"line_{line_id}"),
                newave_code=0,
                cobre_id=line_id,
                stage=int(row["stage_id"]),
                variable="net_flow_mw",
                newave_value=nw_value,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return results, unresolved_lists


def _line_bounds_and_meta(cobre_output_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """Cobre-side line capacity bounds + metadata for the Network tab.

    Both are pure Cobre-case artifacts, not DECOMP-derived, so this reads
    them exactly the way ``results.compare_results`` does for the source
    model's own Network tab:

    - ``line_bounds``: ``constraints/line_bounds.parquet`` verbatim, via
      pandas (``charts.line_summary_chart`` indexes it with ``.iterrows()``
      / ``row["..."]``, so it must stay pandas, never polars -- handing it a
      polars frame makes the bounds overlay silently disappear). This is
      already per-stage: ``decomp/network.py::convert_lines`` writes one base
      row (``block_id`` null) per ``(line, stage)`` carrying the resolved
      max-of-blocks ``IA`` capacity for that stage, so a case whose capacity
      genuinely never changes across stages naturally produces identical
      per-stage rows here -- no separate static-capacity broadcast needs
      implementing on top.
    - ``line_meta``: ``system/lines.json``'s ``"lines"`` list verbatim --
      each entry's own nested ``capacity.direct_mw``/``capacity.reverse_mw``
      is exactly the shape ``line_summary_chart`` reads.

    A missing file degrades to an empty frame / empty list rather than
    raising -- the Network tab is optional (ticket-008's Error Handling).
    """
    case_dir = case_dir_for(cobre_output_dir)

    bounds_path = case_dir / "constraints" / "line_bounds.parquet"
    line_bounds = (
        pd.read_parquet(bounds_path) if bounds_path.exists() else pd.DataFrame()
    )

    lines_path = case_dir / "system" / "lines.json"
    line_meta: list[dict] = []
    if lines_path.exists():
        try:
            line_meta = json.loads(lines_path.read_text()).get("lines", [])
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("Failed to read %s: %s", lines_path, exc)
    return line_bounds, line_meta


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


# --- ticket-010: Overview cost metadata (nw_costs / nw_sin cost rows) ---
#
# Fills the Overview tab's cost sections: the NPV breakdown dict consumed by
# ``charts.overview_metrics``/``cost_breakdown_chart``/``_table`` (keyed by
# ``charts._COST_MAP``'s the source model label strings) and the per-stage
# ``nw_sin`` rows consumed by ``charts._extract_stage_cost_series``
# (``COPER``/``CUSTO_FUTURO``/``CTERM``). This is the single place the
# native k$ -> R$ reconciliation happens for both target unit conventions
# (see epic-04's "TOP RISK" table): the dict is R$ (``reconcile_kdollars_to_
# reais``, x1e3); ``nw_sin`` is 10^6 R$ (an additional /1e6 on top of that
# factor).

#: ``relatorio_operacao_custos`` numeric cost columns this ticket aggregates
#: -- every cost column the table carries except the per-submarket
#: ``cmo_<sbm>`` marginal-*price* columns (not a cost) and the
#: estagio/cenario/probabilidade bookkeeping columns.
_RELATO_COST_COLUMNS: tuple[str, ...] = (
    "custo_presente",
    "custo_futuro",
    "geracao_termica",
    "violacao_desvio",
    "penalidade_vertimento_reservatorio",
    "penalidade_vertimento_fio",
    "violacao_turbinamento_reservatorio",
    "violacao_turbinamento_fio",
    "penalidade_intercambio",
)

#: ``violacao_desvio`` has no ``charts._COST_MAP`` analogue -- surfaced under
#: its own label (matching the Portuguese-abbreviation style every other
#: ``_COST_MAP`` key uses) rather than being silently dropped;
#: ``charts._resolve_cost_categories`` appends it as an unmapped residual
#: category.
_DEVIATION_VIOLATION_LABEL = "VIOLACAO DESVIO"

#: ``(display_label, source_columns...)`` for the ``nw_costs`` NPV dict --
#: mirrors ``charts._COST_MAP``'s the source model label strings. VERTIMENTO
#: sums two source columns (reservoir + run-of-river spillage penalty); every
#: other category is a single column.
_NW_COST_LABELS: tuple[tuple[str, ...], ...] = (
    ("GERACAO TERMICA", "geracao_termica"),
    ("INTERCAMBIO", "penalidade_intercambio"),
    (
        "VERTIMENTO",
        "penalidade_vertimento_reservatorio",
        "penalidade_vertimento_fio",
    ),
    ("VIOL. TURB. MINIMO", "violacao_turbinamento_reservatorio"),
    ("VIOL. TURB. MAXIMO", "violacao_turbinamento_fio"),
    (_DEVIATION_VIOLATION_LABEL, "violacao_desvio"),
)

#: ``nw_sin`` variable token -> source column, for the per-stage cost rows
#: ``charts._extract_stage_cost_series`` reads. Matches MEDIAS-SIN's the
#: source model tokens on the newave side of ``compare newave``.
_NW_SIN_COST_TOKENS: tuple[tuple[str, str], ...] = (
    ("COPER", "custo_presente"),
    ("CUSTO_FUTURO", "custo_futuro"),
    ("CTERM", "geracao_termica"),
)


def _cost_frames(decomp_dir: Path) -> tuple[dict[str, float], pl.DataFrame]:
    """DECOMP-side Overview cost metadata: the NPV dict + ``nw_sin`` rows.

    Reads ``relatorio_operacao_custos`` (native k$, one row per (estagio,
    cenario)) and aggregates it in two passes, both built on the same
    per-stage mean:

    - **Per-stage mean across scenarios** -- :func:`_scenario_mean` with no
      entity column (a single SIN-wide row per stage), reusing the exact
      kernel every other ``_*_side``/``_energy_balance_frames`` helper in
      this module uses. This keeps the source and cobre sides on the same
      unweighted footing the module docstring already commits to ("Scenario
      averaging is unweighted on both sides") even though
      ``relatorio_operacao_custos`` carries an explicit ``probabilidade``
      column per row -- special-casing a probability-weighted mean here
      would make this one source diverge from every other level's
      convention. No stage discount is applied either: unlike cobre's own
      ``discount_factor`` column (:func:`cobre_readers.read_cobre_cost_
      breakdown`), ``relatorio_operacao_custos`` exposes no per-row discount
      factor at this granularity, so there is nothing to apply -- the
      per-stage values are DECOMP's own raw nominal $ costs at that node.
    - **``nw_costs``** (R$): the per-stage means summed across every stage,
      then :func:`reconcile_kdollars_to_reais` (x1e3). Categories below
      0.01 R$ are excluded, mirroring ``newave_readers.read_pmo_cost_
      breakdown`` and :func:`cobre_readers.read_cobre_cost_breakdown`'s own
      floor.
    - **``nw_sin`` rows** (10^6 R$): the same per-stage means, unpivoted to
      ``COPER``/``CUSTO_FUTURO``/``CTERM`` rows with the 1-based ``stage``
      column kept as-is (the consuming charts subtract ``nw_offset``).

    Raises whatever :func:`read_relato_costs` raises on a missing/empty
    parse -- no new swallowing (ticket-009's "no silent-empty" reader
    contract).
    """
    raw = read_relato_costs(decomp_dir)
    per_stage = _scenario_mean(
        raw, "estagio", list(_RELATO_COST_COLUMNS), entity_column=None
    )
    present = [c for c in _RELATO_COST_COLUMNS if c in per_stage.columns]

    totals: dict[str, float] = (
        per_stage.select([pl.col(c).sum().alias(c) for c in present]).row(0, named=True)
        if present
        else {}
    )

    nw_costs: dict[str, float] = {}
    for label, *columns in _NW_COST_LABELS:
        value = reconcile_kdollars_to_reais(sum(totals.get(c, 0.0) for c in columns))
        if abs(value) > 0.01:
            nw_costs[label] = value

    sin_tokens = [(var, col) for var, col in _NW_SIN_COST_TOKENS if col in present]
    if not sin_tokens or per_stage.is_empty():
        nw_sin_rows = _empty_market_frame()
    else:
        # A scalar factor derived from the single conversion site rather
        # than a re-derived literal: k$ -> R$ (x1e3) then R$ -> 10^6 R$
        # (/1e6), collapsing to x1e-3 -- but expressed via
        # reconcile_kdollars_to_reais so the x1e3 itself is never repeated.
        k_to_r6 = reconcile_kdollars_to_reais(1.0) / 1e6
        nw_sin_rows = (
            per_stage.select(
                "estagio",
                *[pl.col(col).alias(var) for var, col in sin_tokens],
            )
            .unpivot(
                on=[var for var, _ in sin_tokens],
                index=["estagio"],
                variable_name="variable",
                value_name="value",
            )
            .with_columns(
                pl.lit(0).cast(pl.Int64).alias("newave_code"),
                pl.col("estagio").cast(pl.Int64).alias("stage"),
                (pl.col("value") * k_to_r6).cast(pl.Float64).alias("value"),
            )
            .select("newave_code", "stage", "variable", "value")
            .sort("stage", "variable")
        )

    return nw_costs, nw_sin_rows


def _union_cost_rows(nw_sin: pl.DataFrame, cost_rows: pl.DataFrame) -> pl.DataFrame:
    """Union :func:`_cost_frames`'s cost rows onto the existing ``nw_sin``.

    ``nw_sin`` (from :func:`_energy_balance_frames`, ticket-006) already
    carries the EARM/ENA rows -- this must add the cost rows alongside them,
    never overwrite. A columnless frame on either side (the
    ``_AlignedDecompFrames``/``PercentileData`` dataclass default
    ``pl.DataFrame()``, which every fixture that predates ticket-006/010
    still carries) cannot be vertically concatenated against a typed frame,
    so it is treated as "nothing to union" instead of raising a
    schema-mismatch error.
    """
    if not nw_sin.columns:
        return cost_rows
    if not cost_rows.columns:
        return nw_sin
    return pl.concat([nw_sin, cost_rows], how="vertical", rechunk=True)


# --- ticket-013: Performance tab timing metadata ---
#
# Fills the Performance tab's DECOMP-side wall-clock metadata (Caveat #2):
# DECOMP is nested Benders over an explicit tree, not sampled SDDP -- it has
# no forward/backward pass structure, so ``nw_tim_iterations`` never carries
# ``forward_seconds``/``backward_seconds``. ``performance_fwd_bwd_split_chart``
# renders cobre-only whenever those columns are absent (charts.py's own
# ``has_nw`` guard checks for them, not just frame emptiness).

#: ``decomp.tim``'s ``Etapa`` phase name -> the two keys
#: ``charts.performance_metric_cards`` reads (``"Tempo Total"`` /
#: ``"Calculo da Politica"``), so the shared card renderer needs no
#: source-specific branch. Confirmed against the real reduced deck
#: (``example/decomp-mar-26-rv2-reduced/saidas/decomp.tim``): DECOMP emits
#: exactly four phases -- ``"Leitura de Dados"``, ``"Convergencia"``,
#: ``"Impressao"``, ``"Tempo Total"``. ``"Convergencia"`` is DECOMP's nested
#: Benders backward/forward loop -- the DECOMP analog of "Calculo da
#: Politica" (policy training) on the source model's own ``newave.tim``.
#: A deck whose ``decomp.tim`` doesn't carry one of these two ``Etapa`` rows
#: -- an incomplete/truncated run -- gets ``0.0`` for that key, explicitly,
#: from :func:`_decomp_tim_stages` itself (never KeyError, never omitted):
#: the two keys are guaranteed present whenever ``decomp.tim`` was read at
#: all, so the metric cards' own ``.get(key, 0.0)`` fallback is defense in
#: depth, not the sole guarantee.
_DECOMP_TIM_PHASE_MAP: dict[str, str] = {
    "Tempo Total": "Tempo Total",
    "Convergencia": "Calculo da Politica",
}


def _decomp_tim_stages(decomp_dir: Path) -> dict[str, float]:
    """``nw_tim_stages``: DECOMP phase totals mapped onto the shared card keys.

    ``read_decomp_tim``'s ``Tempo`` column is a ``datetime.timedelta`` (one
    row per ``iter_rows``); :meth:`~datetime.timedelta.total_seconds` gives
    the float seconds :func:`~cobre_bridge.comparators.charts.
    performance_metric_cards` expects. A missing/empty ``decomp.tim``
    (``read_decomp_tim`` raising ``FileNotFoundError``/``ValueError``)
    degrades to ``{}`` instead of failing :func:`build_decomp_dataset` --
    the metric cards then render with their ``0.0``/"—" fallback. When
    ``decomp.tim`` *is* read but one of the two mapped ``Etapa`` rows is
    absent from it, that key is still returned, explicitly set to ``0.0``
    (see :data:`_DECOMP_TIM_PHASE_MAP`).
    """
    try:
        table = read_decomp_tim(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model timing table (decomp.tim): %s", exc)
        return {}

    seconds_by_etapa: dict[str, float] = {}
    for row in table.iter_rows(named=True):
        tempo = row["Tempo"]
        seconds_by_etapa[str(row["Etapa"]).strip()] = (
            tempo.total_seconds() if tempo is not None else 0.0
        )
    return {
        key: seconds_by_etapa.get(etapa, 0.0)
        for etapa, key in _DECOMP_TIM_PHASE_MAP.items()
    }


#: Canonical ``nw_tim_iterations`` schema -- deliberately only ``iteration``/
#: ``total_seconds`` (Caveat #2): unlike the source model's own
#: ``newave.tim``, DECOMP has no forward/backward pass to split, so this
#: frame must never carry ``forward_seconds``/``backward_seconds`` --
#: fabricating either would misrepresent a source with no such structure.
_DECOMP_TIM_ITERATIONS_SCHEMA: dict[str, type[pl.DataType]] = {
    "iteration": pl.Int64,
    "total_seconds": pl.Float64,
}


def _decomp_tim_iterations(decomp_dir: Path) -> pl.DataFrame:
    """``nw_tim_iterations``: per-iteration wall-clock, DECOMP-shaped.

    ``relato.convergencia``'s ``iteracao``/``tempo`` rename straight onto
    ``iteration``/``total_seconds`` -- no forward/backward split is derived
    or fabricated (Caveat #2). Degrades to the empty canonical frame when
    ``read_relato_convergence`` raises ``FileNotFoundError``/``ValueError``,
    matching :func:`_decomp_convergence_frame`'s own degrade-to-empty
    contract for the same reader.
    """
    empty = pl.DataFrame(schema=_DECOMP_TIM_ITERATIONS_SCHEMA)
    try:
        source = read_relato_convergence(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model convergence table for timing: %s", exc)
        return empty

    columns = {c.lower(): c for c in source.columns}
    if "iteracao" not in columns or "tempo" not in columns:
        return empty
    return source.select(
        pl.col(columns["iteracao"]).cast(pl.Int64).alias("iteration"),
        pl.col(columns["tempo"]).cast(pl.Float64).alias("total_seconds"),
    )


def _decomp_max_stage(aligned: _AlignedDecompFrames) -> int | None:
    """Max DECOMP stage (0-based, Cobre convention) seen in this run.

    Neither of :func:`_decomp_tim_stages`'/:func:`_decomp_tim_iterations`'s
    two sources (``decomp.tim``, ``relato.convergencia``) carries a stage
    axis, so ``nw_max_stage`` is derived from the already-aligned per-level
    frames (:func:`_hydro_side`/:func:`_thermal_side`/:func:`_bus_side`, via
    :func:`_map_entities`) instead of an extra read -- their ``stage_id`` is
    already 0-based, matching every other Cobre-convention stage-axis field
    on :class:`~cobre_bridge.comparators.results.PercentileData`. ``None``
    when every level is empty, mirroring
    :func:`~cobre_bridge.comparators.results.compare_results`'s own
    ``nw_max_stage_1based is None`` case.
    """
    candidates: list[int] = []
    for frame in (aligned.source_hydro, aligned.source_thermal, aligned.source_bus):
        if frame.is_empty() or "stage_id" not in frame.columns:
            continue
        values = frame["stage_id"].drop_nulls()
        if not values.is_empty():
            candidates.append(int(values.max()))  # type: ignore[arg-type]
    return max(candidates) if candidates else None


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

    The shared read/align result :func:`build_decomp_dataset` builds on, via
    :func:`_read_aligned_frames` — deck discovery, id-map construction,
    per-level scenario averaging, and the matching Cobre means run exactly
    once.
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


# --- ticket-014: Hydro Operation + Plant Details tab metadata ---
#
# Fills the four remaining hydro ``PercentileData`` fields the shared Hydro
# Operation tab (``hydro_per_bus_chart``/``hydro_aggregate_chart``/
# ``hydro_slack_per_bus_chart``/``hydro_slack_aggregate_chart``) and the
# Hydro Plant Details tab (``build_hydro_detail_tab``) both read.


def _merge_hydro_bus_ids(
    meta: dict[int, dict], labels: dict[int, frozenset[int]]
) -> dict[int, dict]:
    """Copy each hydro's metadata dict and inject its cobre bus ids.

    ``cobre_readers.read_cobre_hydro_metadata`` carries plant physics only
    -- no bus information (decision B1, see that reader's docstring); the
    plant -> bus *label* is re-sourced from
    ``cobre_readers.read_cobre_hydro_bus_labels``'s
    ``hydro_bus_generation``-partition-derived map, exactly the way
    ``results.compare_results`` merges the two for the source model's own
    Hydro Operation/Details tabs.

    ``hydro_per_bus_chart``/``hydro_slack_per_bus_chart`` (via
    ``analyze._bus_name_lookups``) key off ``meta[hydro_id]["bus_ids"]``
    and KeyError without it -- every
    returned entry carries the key, an empty list when the plant has no
    label in the partition, rather than leaving it absent. Returns a new
    dict of new per-plant dicts; neither *meta* nor *labels* is mutated.
    """
    return {
        hydro_id: {**entry, "bus_ids": labels.get(hydro_id, [])}
        for hydro_id, entry in meta.items()
    }


# --- ticket-017: Productivity tab's "Fitted production functions (FPHA)"
# section (`fpha_metrics`/`fpha_surface`/`fpha_spill`) ---
#
# [ASSUMPTION] (epic-06 Pending Decision, Confidence Low): the newave-side
# comparison (`analyze.build_fpha_comparison`) reconstructs BOTH solvers'
# fitted production surfaces on a shared ``(V, Q)`` grid and renders a full
# 3D overlay -- option (a) in the ticket. That needs the source side's
# fitted PLANE coefficients (``gamma_0``/``gamma_v``/``gamma_q``/``gamma_s``
# + a multiplier), evaluated at that shared grid. None of this ticket's
# three declared readers carries them: `read_dec_estatfpha`'s table is a
# single DECK-WIDE variavel/valor summary (no per-hydro/stage key at all);
# `read_dec_desvfpha`'s table carries the REALIZED ``(V, Q, S)`` operating
# points from the forward pass, not a fitted plane; `read_eco_fpha`'s table
# gives only the fitting-grid *bounds*/node counts, no coefficients. The
# plane coefficients themselves live in a fourth file (``avl_cortesfpha.rvN``
# / idecomp's ``AvlCortesFpha``) that is explicitly NOT one of this ticket's
# three declared readers. Option (a) is therefore genuinely infeasible on
# the declared reader scope, so this ships the documented fallback (b):
# `fpha_metrics` only -- `fpha_surface`/`fpha_spill` always stay `None`, and
# the report's existing FPHA-section gate (a plain
# ``if not fpha_metrics.is_empty()`` in `report_builder`, unmodified by this
# ticket) already renders `fpha_detail_chart`'s "No production-function
# (FPHA) data available" placeholder for an empty surface/spill pair, so no
# exception follows from leaving them empty.
#
# Within (b), rather than reporting the source model's own fph-vs-fpha
# self-consistency in isolation (a number with no Cobre linkage at all),
# `_fpha_metrics` evaluates Cobre's OWN fitted envelope
# (`cobre_readers.read_cobre_fpha_planes`) AT the source model's realized
# operating points (`read_dec_desvfpha`'s ``volume_total_hm3``/
# ``vazao_turbinada_m3s``/``vazao_vertida_m3s``) and compares it to the
# source model's own realized ``geracao_hidraulica_fpha`` (the value its LP
# actually consumed at that point) -- a genuine, non-fabricated cross-solver
# comparison, just sampled at the source model's own trajectory instead of a
# shared dense grid (so it never claims the full-surface fidelity (a) would
# have shown). ``n_planes_newave`` stays null throughout -- no declared
# reader counts the source model's planes; ``n_planes_cobre``/``n_v`` are
# real counts from `read_cobre_fpha_planes`/`read_eco_fpha`.


def _log_decomp_fpha_deck_summary(decomp_dir: Path) -> None:
    """Log the source model's deck-wide FPHA deviation summary.

    `read_dec_estatfpha`'s table carries no per-hydro or per-stage key (a
    single deck-wide variavel/valor summary), so it cannot join into the
    per-(hydro, stage) `fpha_metrics` frame `_fpha_metrics` builds -- it is
    surfaced as log context only. Never raises: a missing/empty parse is
    logged and dropped, exactly like every other optional FPHA source here.
    """
    try:
        summary = read_dec_estatfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA deck-wide summary (dec_estatfpha): %s", exc)
        return
    _LOG.info(
        "Source-model deck-wide FPHA deviation summary: %s",
        dict(
            zip(
                summary["variavel"].to_list(),
                summary["valor"].to_list(),
                strict=True,
            )
        ),
    )


def _cobre_fpha_plane_counts(cb_planes: pl.DataFrame) -> pl.DataFrame:
    """One row per (cobre_id, stage) with Cobre's own fitted plane count."""
    return (
        cb_planes.rename({"hydro_id": "cobre_id", "stage_id": "stage"})
        .cast({"cobre_id": pl.Int64, "stage": pl.Int64})
        .group_by(["cobre_id", "stage"])
        .agg(pl.len().cast(pl.Int64).alias("n_planes_cobre"))
    )


def _decomp_fpha_grid_nv(
    decomp_dir: Path, code_to_cobre: dict[int, int]
) -> pl.DataFrame:
    """One row per (cobre_id, stage) with the source model's own volume
    fitting-grid node count (``numero_pontos_volume_armazenado``), from
    `read_eco_fpha`.

    Degrades to an empty frame -- never raises -- when the deck ships no
    ``eco_fpha`` table (confirmed absent on the reduced deck this ticket was
    developed against): ``n_v`` then stays null on every metrics row, honest
    about the value being genuinely unavailable rather than fabricating one.
    """
    empty = pl.DataFrame(
        schema={"cobre_id": pl.Int64, "stage": pl.Int64, "n_v": pl.Int64}
    )
    try:
        grid = read_eco_fpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA grid table (eco_fpha): %s", exc)
        return empty

    mapped = grid.with_columns(
        pl.col("codigo_usina")
        .map_elements(lambda c: code_to_cobre.get(int(c)), return_dtype=pl.Int64)
        .alias("cobre_id"),
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
    ).filter(pl.col("cobre_id").is_not_null())
    if mapped.is_empty():
        return empty
    return mapped.select(
        "cobre_id",
        "stage",
        pl.col("numero_pontos_volume_armazenado").cast(pl.Int64).alias("n_v"),
    )


def _evaluate_cobre_fpha_at_points(
    cb_planes: pl.DataFrame, points: pl.DataFrame
) -> pl.DataFrame:
    """Evaluate Cobre's min-over-planes FPHA envelope at the source model's
    own realized (hydro, stage, v, q, s) operating points.

    ``points`` carries one row per realized sample, with columns
    ``_point_id`` (a stable row identity), ``cobre_id``, ``stage``,
    ``v_hm3``, ``q_m3s``, ``s_m3s``. ``cb_planes`` is
    `cobre_readers.read_cobre_fpha_planes`'s own ``hydro_id``/``stage_id``/
    ``gamma_0``/``gamma_v``/``gamma_q``/``gamma_s``/``kappa`` frame (multiple
    plane rows per (hydro_id, stage_id)). Mirrors
    `cobre_bridge.comparators.analyze._evaluate_fpha_envelope`'s own
    ``kappa * (gamma_0 + gamma_v * v + gamma_q * q + gamma_s * s)``
    min-over-planes definition (``gamma_v`` already multiplies *absolute*
    volume for Cobre's own coefficients, so no volume offset applies here --
    the source model's own ``volume_total_hm3`` is likewise absolute, not
    useful, volume), expressed as a join + group-by rather than a numpy
    broadcast since this evaluates a scattered point cloud, not a dense grid.

    Returns one row per ``_point_id`` with the envelope value in
    ``cobre_gh_mw``; a point whose (hydro, stage) has no Cobre planes drops
    out of the join rather than null-keeping.
    """
    keyed_planes = cb_planes.rename({"hydro_id": "cobre_id", "stage_id": "stage"}).cast(
        {"cobre_id": pl.Int64, "stage": pl.Int64}
    )
    joined = points.join(keyed_planes, on=["cobre_id", "stage"], how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema={"_point_id": pl.Int64, "cobre_gh_mw": pl.Float64})
    return (
        joined.with_columns(
            (
                pl.col("kappa")
                * (
                    pl.col("gamma_0")
                    + pl.col("gamma_v") * pl.col("v_hm3")
                    + pl.col("gamma_q") * pl.col("q_m3s")
                    + pl.col("gamma_s") * pl.col("s_m3s")
                )
            ).alias("plane_value")
        )
        .group_by("_point_id")
        .agg(pl.col("plane_value").min().alias("cobre_gh_mw"))
    )


def _fpha_metrics(
    decomp_dir: Path,
    cobre_output_dir: Path,
    id_map: DecompIdMap | None,
    hydro_names: dict[int, str],
) -> pl.DataFrame | None:
    """Per-(hydro, stage) FPHA fit-fidelity metrics -- fallback (b), see the
    [ASSUMPTION] comment above this section.

    ``None`` whenever either side has no fitted planes to compare at all:
    Cobre fitted none (`cobre_readers.read_cobre_fpha_planes` returns
    ``None``), the deck has no id map to resolve hydro codes onto Cobre ids
    (`id_map` is ``None``), the source model's own deviation table is
    absent/empty, or no realized point resolves onto a Cobre hydro with
    fitted planes. `build_decomp_dataset` passes ``None`` straight through
    to `~cobre_bridge.comparators.results.PercentileData.fpha_metrics`; the
    report's FPHA section gate (`report_builder`, unmodified by this ticket)
    treats that identically to an empty frame via its own ``_meta_frame``
    helper, so the section is omitted, never a crash.
    """
    cb_planes = cobre_readers.read_cobre_fpha_planes(cobre_output_dir)
    if cb_planes is None or id_map is None:
        return None

    try:
        deviations = read_dec_desvfpha(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model FPHA deviation table: %s", exc)
        return None

    code_to_cobre = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}
    points = deviations.with_columns(
        pl.col("codigo_usina")
        .map_elements(lambda c: code_to_cobre.get(int(c)), return_dtype=pl.Int64)
        .alias("cobre_id"),
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage"),
    ).filter(pl.col("cobre_id").is_not_null())
    if points.is_empty():
        return None
    points = (
        points.select(
            "cobre_id",
            "stage",
            pl.col("volume_total_hm3").cast(pl.Float64).alias("v_hm3"),
            pl.col("vazao_turbinada_m3s").cast(pl.Float64).alias("q_m3s"),
            pl.col("vazao_vertida_m3s").cast(pl.Float64).alias("s_m3s"),
            pl.col("geracao_hidraulica_fpha").cast(pl.Float64).alias("decomp_gh_mw"),
        )
        .with_row_index("_point_id")
        .with_columns(pl.col("_point_id").cast(pl.Int64))
    )

    envelope = _evaluate_cobre_fpha_at_points(cb_planes, points)
    if envelope.is_empty():
        return None

    _log_decomp_fpha_deck_summary(decomp_dir)

    joined = points.join(envelope, on="_point_id", how="inner").with_columns(
        (pl.col("cobre_gh_mw") - pl.col("decomp_gh_mw")).alias("diff")
    )

    metrics = (
        joined.group_by(["cobre_id", "stage"])
        .agg(
            pl.col("decomp_gh_mw").max().abs().alias("_denom"),
            pl.col("diff").abs().mean().alias("_mean_abs_diff"),
            pl.col("diff").mean().alias("_mean_diff"),
            pl.col("diff").abs().max().alias("max_abs_dev"),
            pl.col("cobre_gh_mw").max().alias("_cobre_max"),
            pl.col("decomp_gh_mw").max().alias("_decomp_max"),
        )
        .with_columns((pl.col("_denom") > 1e-9).alias("_scaled"))
        .with_columns(
            pl.when(pl.col("_scaled"))
            .then(pl.col("_mean_abs_diff") / pl.col("_denom"))
            .otherwise(None)
            .alias("nmae"),
            pl.when(pl.col("_scaled"))
            .then(pl.col("_mean_diff") / pl.col("_denom"))
            .otherwise(None)
            .alias("bias"),
            pl.when(pl.col("_scaled"))
            .then(pl.col("_cobre_max") / pl.col("_decomp_max"))
            .otherwise(None)
            .alias("gh_max_ratio"),
        )
        .join(_cobre_fpha_plane_counts(cb_planes), on=["cobre_id", "stage"], how="left")
        .join(
            _decomp_fpha_grid_nv(decomp_dir, code_to_cobre),
            on=["cobre_id", "stage"],
            how="left",
        )
        .with_columns(
            pl.col("cobre_id")
            .map_elements(
                lambda c: hydro_names.get(int(c), f"hydro_{c}"), return_dtype=pl.Utf8
            )
            .alias("plant_name"),
            pl.lit(None).cast(pl.Int64).alias("n_planes_newave"),
        )
        .select(
            "cobre_id",
            "plant_name",
            "stage",
            "n_planes_newave",
            "n_planes_cobre",
            "n_v",
            "nmae",
            "bias",
            "max_abs_dev",
            "gh_max_ratio",
        )
        .sort(["cobre_id", "stage"])
    )
    return metrics if not metrics.is_empty() else None


# --- ticket-018: REE energy rollup via membership ---
#
# DECOMP reports energy at REE (reservoir-equivalent-energy) granularity
# (`read_dec_oper_ree`); Cobre has no REE entity and `DecompIdMap` carries no
# REE map, so the Cobre counterpart is built by rolling each REE's member
# plants' own energy up through the `relato.uhes_rees_submercados`
# membership table (`read_relato_membership`) instead. Reuses
# `build_decomp_dataset`'s own `line_id_map` (ticket-008) for the hydro
# code -> Cobre id half of that membership, the same way `_fpha_metrics`
# already does, rather than rebuilding a third `DecompIdMap`.


def _ree_membership_map(
    membership: pl.DataFrame, hydro_codes: dict[int, int]
) -> tuple[dict[int, int], list[int]]:
    """``{cobre_hydro_id: codigo_ree}`` via membership, restricted to the
    operated hydro codes this module already resolves
    (``{codigo_usina: cobre_id}``, from `DecompIdMap.hydro_id`).

    A hydro code with no row in *membership* cannot be attributed to any REE
    -- it is excluded from every REE sum and returned (sorted) as the second
    element instead of being silently dropped (ticket-018 requirement 4).
    """
    code_to_ree: dict[int, int] = {}
    if not membership.is_empty():
        for row in membership.iter_rows(named=True):
            code_to_ree[int(row["codigo_usina"])] = int(row["codigo_ree"])

    ree_by_cobre_id: dict[int, int] = {}
    unmapped: list[int] = []
    for code, cobre_id in hydro_codes.items():
        ree_code = code_to_ree.get(code)
        if ree_code is None:
            unmapped.append(code)
            continue
        ree_by_cobre_id[cobre_id] = ree_code
    return ree_by_cobre_id, sorted(unmapped)


_EMPTY_COBRE_REE_SUMS_SCHEMA: dict[str, type[pl.DataType]] = {
    "entity_id": pl.Int64,
    "stage_id": pl.Int64,
    "ena_mw": pl.Float64,
    "earm_mwh": pl.Float64,
}

#: Columns :func:`_cobre_ree_sums` reads off *cobre_hydro* besides
#: ``entity_id``/``stage_id``. A frame missing either -- e.g. some other
#: ticket's trimmed test fixture that only carries the E1 hydro-comparison
#: columns -- degrades to an empty sum instead of a Polars
#: ``ColumnNotFoundError``.
_REE_COBRE_ENERGY_COLUMNS: tuple[str, ...] = (
    "incremental_inflow_energy_mw",
    "stored_energy_final_mwh",
)


def _cobre_ree_sums(
    cobre_hydro: pl.DataFrame, ree_by_cobre_id: dict[int, int]
) -> pl.DataFrame:
    """Membership-weighted sum of Cobre hydro ENA/EARM per (codigo_ree, stage).

    *cobre_hydro* is `_AlignedDecompFrames.cobre_hydro` (ticket-002's own
    Cobre hydro means, reused rather than re-read); *ree_by_cobre_id* inverts
    :func:`_ree_membership_map`'s own map, one entry per member plant. A
    Cobre hydro id absent from it -- every plant this rollup could not
    attribute to a membership row -- contributes to no REE sum. Returned
    ``entity_id`` is the REE code (``codigo_ree``), matching the join key
    :func:`_decomp_ree_frame` emits; ``earm_mwh`` is still raw MWh here --
    the ÷730 MWmês reconciliation happens once, at the
    :class:`~cobre_bridge.comparators.results.ResultComparison` emission site.
    """
    empty = pl.DataFrame(schema=_EMPTY_COBRE_REE_SUMS_SCHEMA)
    if cobre_hydro.is_empty() or not ree_by_cobre_id:
        return empty
    if not set(_REE_COBRE_ENERGY_COLUMNS) <= set(cobre_hydro.columns):
        return empty
    mapped = cobre_hydro.with_columns(
        pl.col("entity_id")
        .map_elements(lambda i: ree_by_cobre_id.get(int(i)), return_dtype=pl.Int64)
        .alias("codigo_ree")
    ).filter(pl.col("codigo_ree").is_not_null())
    if mapped.is_empty():
        return empty
    return (
        mapped.group_by(["codigo_ree", "stage_id"])
        .agg(
            pl.col("incremental_inflow_energy_mw").sum().alias("ena_mw"),
            pl.col("stored_energy_final_mwh").sum().alias("earm_mwh"),
        )
        .rename({"codigo_ree": "entity_id"})
        .cast({"entity_id": pl.Int64, "stage_id": pl.Int64})
        .sort(["entity_id", "stage_id"])
    )


def _decomp_ree_frame(decomp_dir: Path) -> tuple[pl.DataFrame, dict[int, str]]:
    """DECOMP-side per-(codigo_ree, stage) ENA/EARM, scenario-averaged over
    the deck's own nodes, plus the REE display-name lookup.

    Mirrors every other ``_*_side`` helper's fold
    (:func:`_stage_rows`/:func:`_scenario_mean`), keyed by ``codigo_ree``
    (`read_dec_oper_ree` carries no ``patamar`` column, so :func:`_stage_rows`
    is a no-op here -- called anyway for the same fold shape every other level
    uses). Rebases ``estagio`` onto the 0-based ``stage_id`` the same way
    :func:`_map_entities` does. Raises whatever `read_dec_oper_ree` raises on
    a missing/empty parse -- the caller degrades that to "no REE section".
    """
    frame = _stage_rows(read_dec_oper_ree(decomp_dir))

    names: dict[int, str] = {}
    if "nome_ree" in frame.columns:
        for row in (
            frame.select("codigo_ree", "nome_ree").unique().iter_rows(named=True)
        ):
            names[int(row["codigo_ree"])] = str(row["nome_ree"])

    aggregated = _scenario_mean(
        frame, "estagio", ["ena_MWmes", "earm_final_MWmes"], entity_column="codigo_ree"
    )
    if aggregated.is_empty():
        return aggregated, names
    mapped = aggregated.with_columns(
        pl.col("codigo_ree").cast(pl.Int64).alias("entity_id"),
        (pl.col("estagio").cast(pl.Int64) - 1).alias("stage_id"),
    ).drop("codigo_ree", "estagio")
    return mapped, names


def _ree_result_comparisons(
    decomp_dir: Path,
    cobre_hydro: pl.DataFrame,
    id_map: DecompIdMap | None,
) -> tuple[list[ResultComparison], list[int]]:
    """REE-level ``ena_mwmes``/``earm_final_mwmes`` ``ResultComparison`` rows.

    The source-model side is `_decomp_ree_frame` (`read_dec_oper_ree`,
    scenario-averaged per REE/stage); the Cobre side is
    :func:`_cobre_ree_sums`'s membership-weighted plant rollup, with EARM
    converted MWh -> MWmês (:data:`_EARM_MWH_TO_MWMES`) at emission -- ENA is
    a plain sum, no unit conversion. ``newave_code``/``cobre_id`` are both the
    REE's own ``codigo_ree`` -- Cobre has no independent REE id to diverge
    from it.

    Returns ``([], [])`` when *id_map* is ``None`` (deck unreadable -- no
    hydro code -> Cobre id mapping exists to build the membership map with).
    A missing/empty membership table or `dec_oper_ree` table degrades to no
    REE rows (a genuinely unavailable section, logged at INFO) rather than
    raising -- matching every other optional field this module builds. A
    hydro code absent from the membership table is excluded from every REE
    sum and returned (sorted) as the second element, with a WARNING
    diagnostic recording the codes (ticket-018 requirement 4, never silently
    dropped).
    """
    if id_map is None:
        return [], []

    hydro_codes = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}

    try:
        membership = read_relato_membership(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model REE membership table (relato): %s", exc)
        return [], []

    ree_by_cobre_id, unmapped = _ree_membership_map(membership, hydro_codes)
    if unmapped:
        emit(
            Diagnostic(
                code="ree-membership-plant-unmapped",
                severity=Severity.WARNING,
                category="Compare data",
                title="Hydro plant absent from REE membership table",
                summary=(
                    f"{len(unmapped)} source-model hydro plant code(s) carry "
                    "no row in the relato uhes_rees_submercados membership "
                    "table; excluded from every REE energy sum rather than "
                    "silently dropped."
                ),
                notes=[f"codigo_usina: {unmapped}"],
            ),
            logger=_LOG,
        )

    try:
        source_ree, names = _decomp_ree_frame(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model REE table (dec_oper_ree): %s", exc)
        return [], unmapped

    cobre_sums = _cobre_ree_sums(cobre_hydro, ree_by_cobre_id)
    if source_ree.is_empty() or cobre_sums.is_empty():
        return [], unmapped

    joined = source_ree.join(cobre_sums, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return [], unmapped

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        ree_code = int(row["entity_id"])
        stage = int(row["stage_id"])
        name = names.get(ree_code, "")
        cobre_earm_mwmes = float(row["earm_mwh"]) / _EARM_MWH_TO_MWMES
        for variable, nw_raw, cobre_value in (
            ("ena_mwmes", row["ena_MWmes"], float(row["ena_mw"])),
            ("earm_final_mwmes", row["earm_final_MWmes"], cobre_earm_mwmes),
        ):
            if nw_raw is None:
                continue
            nw_value = float(nw_raw)
            abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
            results.append(
                ResultComparison(
                    entity_type="ree",
                    entity_name=name,
                    newave_code=ree_code,
                    cobre_id=ree_code,
                    stage=stage,
                    variable=variable,
                    newave_value=nw_value,
                    cobre_value=cobre_value,
                    abs_diff=abs_diff,
                    rel_diff=rel_diff,
                )
            )
    return results, unmapped


# --- ticket-020: evaporation comparison (hydro, "evaporation_m3s") ---
#
# The source model reports evaporation as a per-stage *volume* in hm³
# (`read_dec_oper_evap`'s `evaporacao_calculada_hm3` -- the run's own water
# balance, not the fitted `evaporacao_modelo_hm3` estimate); Cobre reports it
# as a mean *flow* in m³/s (`evaporation_m3s`, already carried by
# `_read_aligned_frames`'s `cobre_hydro`, see `cobre_readers.
# read_cobre_hydro_means`). Reconciling the two units needs the stage's own
# duration -- unlike the source model's *own* MEDIAS report (`results.py`'s
# `_compare_hydros`, which divides by the rounded fixed monthly constant
# 2.63), the source model's stages here are not always full calendar months
# (sub-monthly patamares), so the divisor must come from the stage's actual
# hours (`_cobre_stage_hours`, sourced from the Cobre case's own
# ``stages.json`` via `cobre_readers._load_block_hours` -- reused rather than
# re-parsed so the two block-hours readings can never drift apart).


def _cobre_stage_hours(cobre_output_dir: Path) -> dict[int, float]:
    """``{stage_id: total_hours}`` from the Cobre case's own ``stages.json``.

    Reuses `cobre_readers._load_block_hours` -- the project's one
    ``stages.json`` block-hours reader, already the source every
    hours-weighted `cobre_readers` aggregation goes through -- rather than
    re-parsing the file a second time; sums each stage's block hours into a
    flat per-stage lookup. Empty when ``stages.json`` cannot be found or
    parsed (`_load_block_hours` returns ``None``), which
    :func:`_evaporation_result_comparisons` treats as "no hours denominator
    available" and degrades to no evaporation rows, rather than fabricating
    one.
    """
    block_hours = cobre_readers._load_block_hours(cobre_output_dir)  # noqa: SLF001
    if block_hours is None or block_hours.is_empty():
        return {}
    totals = block_hours.group_by("stage_id").agg(pl.col("hours").sum().alias("hours"))
    return {
        int(row["stage_id"]): float(row["hours"])
        for row in totals.iter_rows(named=True)
    }


#: hm³ -> m³/s conversion factor per hour of stage duration -- the inverse of
#: `converters.network.C_M3S2HM3` (``hours * 3600 / 1e6``), generalized to the
#: stage's own ``hours`` instead of the fixed ``MONTH_HOURS`` (730) that
#: constant assumes.
_HM3_TO_M3S_HOUR_FACTOR: float = 3600.0 / 1e6


def _hm3_to_m3s(volume_hm3: float, stage_hours: float) -> float:
    """Convert a stage's evaporated volume (hm³) into a mean flow (m³/s).

    ``m3s = hm3 * 1e6 / (stage_hours * 3600)`` -- volume/time with the hm³ ->
    m³ and hour -> second unit changes folded into
    :data:`_HM3_TO_M3S_HOUR_FACTOR`, the same physical relationship
    `converters.network.C_M3S2HM3` expresses in the opposite direction.
    """
    return volume_hm3 / (stage_hours * _HM3_TO_M3S_HOUR_FACTOR)


def _evap_side(
    decomp_dir: Path, hydro_codes: dict[int, int]
) -> tuple[pl.DataFrame, list[int]]:
    """The source model's per-(hydro, stage) evaporated volume (hm³),
    scenario-averaged over the deck's own nodes.

    Mirrors :func:`_hydro_side`'s exact fold (:func:`_stage_rows` is a no-op
    here -- `read_dec_oper_evap` carries no ``patamar`` column, unlike
    `dec_oper_usih` -- called anyway for the same fold shape every other
    level uses). Raises whatever `read_dec_oper_evap` raises on a
    missing/empty parse; the caller degrades that to "no evaporation
    section".
    """
    frame = _stage_rows(read_dec_oper_evap(decomp_dir))
    aggregated = _scenario_mean(
        frame, "estagio", ["evaporacao_calculada_hm3"], entity_column="codigo_usina"
    )
    return _map_entities(aggregated, "codigo_usina", hydro_codes)


def _evaporation_result_comparisons(
    decomp_dir: Path,
    cobre_output_dir: Path,
    cobre_hydro: pl.DataFrame,
    id_map: DecompIdMap | None,
    names: dict[int, str],
) -> tuple[list[ResultComparison], list[int]]:
    """Per-(hydro, stage) ``evaporation_m3s`` ``ResultComparison`` rows.

    The source-model side is :func:`_evap_side` (`read_dec_oper_evap`,
    scenario-averaged, mapped onto Cobre ids via *id_map*); the Cobre side is
    *cobre_hydro*'s own ``evaporation_m3s`` column (already carried by
    `_read_aligned_frames`, not re-read here). A plant carrying an
    evaporation series on only one side -- the source model's
    ``dec_oper_evap`` but no Cobre ``evaporation_m3s`` for that plant, or the
    reverse -- is excluded from the paired join and returned (sorted, by
    Cobre id) as the second element instead of being silently dropped
    (ticket-020 requirement 5), with a WARNING diagnostic recording the
    count.

    TRACKED COBRE-GAP WORKAROUND (C11): Cobre's own evaporation model
    over-scales the volume it deposits on a sub-monthly stage -- it applies
    (pre-fix) a full calendar month's worth of evaporation regardless of the
    stage's actual duration (see `decomp.hydro.
    _evaporation_coefficients_mm`'s docstring and the spec at
    ``~/git/cobre/plans/evaporation-stage-duration-scaling-spec.md``). The
    reconciliation below ONLY rescales the source model's own hm³ volume into
    a directly comparable m³/s flow via :func:`_hm3_to_m3s`; Cobre's
    ``evaporation_m3s`` value is compared UNCHANGED. Any residual divergence
    this surfaces on a sub-monthly stage is that known Cobre gap (C11), not a
    conversion error -- it must not be "corrected away" by rescaling Cobre's
    side to match.

    Returns ``([], [])`` when *id_map* is ``None`` (deck unreadable). A
    missing/empty `dec_oper_evap` table, an absent ``evaporation_m3s`` column
    on the Cobre side, or no resolvable stage-hours denominator
    (:func:`_cobre_stage_hours`) each degrade to "no evaporation section"
    (logged at INFO) rather than raising, matching every other optional
    field this module builds.
    """
    if id_map is None:
        return [], []

    hydro_codes = {code: id_map.hydro_id(code) for code in id_map.hydro_codes}

    try:
        source_evap, _unmapped_hydro_codes = _evap_side(decomp_dir, hydro_codes)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No source-model evaporation table (dec_oper_evap): %s", exc)
        return [], []
    if source_evap.is_empty():
        return [], []

    if cobre_hydro.is_empty() or "evaporation_m3s" not in cobre_hydro.columns:
        _LOG.info("Cobre output has no evaporation_m3s column; skipping")
        return [], []
    cobre_evap = cobre_hydro.select(
        "entity_id", "stage_id", "evaporation_m3s"
    ).drop_nulls("evaporation_m3s")

    stage_hours = _cobre_stage_hours(cobre_output_dir)
    if not stage_hours:
        _LOG.info(
            "No stages.json block-hours available; skipping the evaporation "
            "hm3->m3s reconciliation"
        )
        return [], []

    decomp_ids = set(source_evap["entity_id"].unique().to_list())
    cobre_ids = set(cobre_evap["entity_id"].unique().to_list())
    one_sided = sorted(int(i) for i in decomp_ids.symmetric_difference(cobre_ids))
    if one_sided:
        emit(
            Diagnostic(
                code="evaporation-plant-one-sided",
                severity=Severity.WARNING,
                category="Compare data",
                title="Hydro plant has evaporation data on only one side",
                summary=(
                    f"{len(one_sided)} hydro plant(s) carry an evaporation "
                    "series on only one side (the source model's "
                    "dec_oper_evap or Cobre's evaporation_m3s, not both); "
                    "excluded from the paired evaporation comparison rather "
                    "than silently dropped."
                ),
                notes=[f"cobre_id: {one_sided}"],
            ),
            logger=_LOG,
        )

    joined = source_evap.join(cobre_evap, on=["entity_id", "stage_id"], how="inner")
    if joined.is_empty():
        return [], one_sided

    results: list[ResultComparison] = []
    for row in joined.iter_rows(named=True):
        entity_id = int(row["entity_id"])
        stage = int(row["stage_id"])
        hours = stage_hours.get(stage)
        if hours is None or hours <= 0:
            continue
        nw_value = _hm3_to_m3s(float(row["evaporacao_calculada_hm3"]), hours)
        cobre_value = float(row["evaporation_m3s"])
        abs_diff, rel_diff = _result_diff(nw_value, cobre_value)
        results.append(
            ResultComparison(
                entity_type="hydro",
                entity_name=names.get(entity_id, ""),
                newave_code=int(row["newave_code"]),
                cobre_id=entity_id,
                stage=stage,
                variable="evaporation_m3s",
                newave_value=nw_value,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )
    return results, one_sided


# --- ticket-019: Constraints tab (gc_* metadata, DECOMP-side LHS) ---
#
# `constraints_compare` already supplies the source-agnostic pieces verbatim:
# `_load_generic_constraints`/`_load_generic_constraint_bounds` (the
# converted case's own constraint/bounds tables) and `evaluate_lhs_cobre`
# (the Cobre-side LHS from simulation output). What is missing is the
# DECOMP-side LHS -- `evaluate_lhs_newave` is MEDIAS-USIH/int*.out +
# EntityAlignment/NewaveIdMap-coupled and cannot serve DECOMP (see the
# module's own docstring) -- so `_generic_constraint_lhs_decomp` below is
# its DECOMP sibling: it evaluates each constraint directly against
# `dec_oper_usih`/`dec_oper_usit`/`dec_oper_rhesoft`, using the
# special-constraint register (`decomp.constraint_registers.
# read_constraints`) to recover the same coefficients the conversion-time
# emitters (`decomp.constraints.emit_re_generics`/`emit_rhq_rhv_generics`/
# `emit_rhe_generics`) used to author each constraint's expression.


@dataclass(frozen=True)
class _DecompConstraintContext:
    """Deck-derived inputs `_generic_constraint_lhs_decomp` needs beyond the
    `dec_oper_*` tables: the special-constraint census (for RE/HQ/HV term
    coefficients) and the id map (to translate a plant code to its cobre
    hydro id for the HV storage floor)."""

    census: ConstraintCensus
    id_map: DecompIdMap


def _decomp_constraint_context(decomp_dir: Path) -> _DecompConstraintContext | None:
    """Best-effort census + id map for the DECOMP-side generic-constraint LHS.

    Mirrors `_build_line_id_map`'s degrade-gracefully pattern (deck
    discovery -> ``Dadger.read`` -> a from-dadger builder): a missing or
    invalid deck yields ``None`` (logged, not raised) instead of failing the
    whole dataset build -- the Constraints tab's DECOMP-side overlay is one
    optional section among many.
    """
    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.constraint_registers import read_constraints
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.pipeline import discover_decomp_files

    try:
        files = discover_decomp_files(decomp_dir)
        dadger = Dadger.read(str(files.dadger))
        return _DecompConstraintContext(
            census=read_constraints(dadger),
            id_map=DecompIdMap.from_dadger(dadger),
        )
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info(
            "No deck available for the DECOMP-side generic-constraint LHS "
            "derivation: %s",
            exc,
        )
        return None


def _dec_oper_hydro_stage_frame(decomp_dir: Path) -> pl.DataFrame:
    """One row per (source hydro code, stage): the stage-aggregate
    ``dec_oper_usih`` columns this ticket's constraint terms need, averaged
    over node/scenario.

    Reuses `_stage_rows` (keep the duration-weighted ``patamar``-null
    aggregate row) + `_scenario_mean` exactly like `_hydro_side`, but stops
    short of `_map_entities`'s translation to cobre ids -- the
    special-constraint register terms this feeds carry the source model's
    own plant codes, not cobre ids. Degrades to an empty frame (never
    raises) when the deck ships no ``dec_oper_usih.csv`` -- every caller
    already treats an empty frame as "nothing to look up".
    """
    try:
        frame = _stage_rows(read_dec_oper_usih(decomp_dir))
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No dec_oper_usih.csv for the Constraints tab LHS: %s", exc)
        return pl.DataFrame()
    columns = [
        "geracao_MW",
        "vazao_defluente_m3s",
        "vazao_turbinada_m3s",
        "vazao_desviada_m3s",
        "vazao_vertida_m3s",
        "volume_util_final_hm3",
    ]
    return _scenario_mean(frame, "estagio", columns, entity_column="codigo_usina")


def _dec_oper_thermal_stage_frame(decomp_dir: Path) -> pl.DataFrame:
    """One row per (source thermal code, stage): ``dec_oper_usit``'s
    generation, averaged over node/scenario. Mirrors
    `_dec_oper_hydro_stage_frame`, including the same degrade-to-empty."""
    try:
        frame = _stage_rows(read_dec_oper_usit(decomp_dir))
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No dec_oper_usit.csv for the Constraints tab LHS: %s", exc)
        return pl.DataFrame()
    return _scenario_mean(
        frame, "estagio", ["geracao_MW"], entity_column="codigo_usina"
    )


def _dec_oper_rhesoft_frame(decomp_dir: Path) -> pl.DataFrame:
    """Best-effort ``dec_oper_rhesoft.csv`` read.

    Empty when the deck carries no RHE (soft minimum-stored-energy)
    constraints -- the source model omits the file entirely in that case,
    which `read_dec_oper_rhesoft`'s own empty-is-error contract (for a
    genuinely broken/empty parse) does not distinguish on its own.
    """
    try:
        return read_dec_oper_rhesoft(decomp_dir)
    except (FileNotFoundError, ValueError) as exc:
        _LOG.info("No dec_oper_rhesoft.csv (no RHE constraints?): %s", exc)
        return pl.DataFrame()


def _stage_frame_to_lookup(
    frame: pl.DataFrame, value_col: str, *, code_col: str = "codigo_usina"
) -> dict[tuple[int, int], float]:
    """One stage-aggregate ``dec_oper_*`` frame column -> ``{(code,
    stage_0based): value}``.

    *frame* is already collapsed to one row per (code, stage) by
    `_scenario_mean`; this only reshapes it into a lookup and converts the
    source model's 1-based ``estagio`` to the 0-based ``stage_id`` every
    other ``gc_*`` frame uses (D-STAGE-OFFSET, offset=1).
    """
    if (
        frame.is_empty()
        or value_col not in frame.columns
        or code_col not in frame.columns
    ):
        return {}
    return {
        (int(row[code_col]), int(row["estagio"]) - 1): float(row[value_col])
        for row in frame.iter_rows(named=True)
        if row[value_col] is not None
    }


def _storage_lookup(
    hydro_frame: pl.DataFrame,
    id_map: DecompIdMap,
    min_storage_by_cobre_id: dict[int, float],
) -> dict[tuple[int, int], float]:
    """``{(source code, stage_0based): absolute storage hm3}`` = useful + Vmin.

    The source model's own operation export (``volume_util_final_hm3``)
    reports *useful* volume (above the plant's minimum operative storage);
    cobre's ``hydro_storage(id)`` -- what an ``HV``/``VARM`` generic
    constraint's expression is built from -- is absolute. This reuses the
    ``min_storage_hm3`` the converter already wrote to the case's own
    ``system/hydros.json`` (via `cobre_readers.read_cobre_hydro_metadata`) --
    the same floor `_cobre_hydro` subtracts in the other direction for the
    Hydro Operation tab's own storage row -- rather than rebuilding
    ``EffectiveCadastro`` from the deck a second time. A single static value
    per plant, not per-stage: the same simplification
    `constraints_compare._load_hydro_min_storage` already accepts on the
    source-model side.
    """
    useful = _stage_frame_to_lookup(hydro_frame, "volume_util_final_hm3")
    out: dict[tuple[int, int], float] = {}
    for (code, stage), value in useful.items():
        try:
            cobre_id = id_map.hydro_id(code)
        except KeyError:
            continue
        vmin = min_storage_by_cobre_id.get(cobre_id)
        if vmin is None:
            continue
        out[(code, stage)] = value + vmin
    return out


@dataclass(frozen=True)
class _DecompConstraintLookups:
    """Per-(source code, stage) means feeding
    `_generic_constraint_lhs_decomp`'s register-term resolution."""

    hydro_generation: dict[tuple[int, int], float] = field(default_factory=dict)
    thermal_generation: dict[tuple[int, int], float] = field(default_factory=dict)
    flow: dict[str, dict[tuple[int, int], float]] = field(default_factory=dict)
    storage: dict[tuple[int, int], float] = field(default_factory=dict)


#: Register term variables `_generic_constraint_lhs_decomp` cannot re-derive
#: from ``dec_oper_*`` output without conversion-time context this
#: comparator does not rebuild: ``interchange`` (RE ``FI``) needs the same
#: bus/line direction resolution `decomp.constraints.resolve_fi_term` does
#: at conversion time; ``QBOM`` (HQ pumping) names a pumping-station code
#: (not a hydro code), and no ``dec_oper_*`` table reports a pumping
#: station's own flow directly. A constraint carrying either is skipped
#: whole (skip-not-partial), never rendered with a partial sum.
_UNSUPPORTED_TERM_VARIABLES = frozenset({"interchange", "QBOM"})

#: Matches a cobre generic constraint's ``name`` field as authored by the
#: conversion-time emitters (``decomp.constraints.emit_re_generics`` ->
#: ``"RE_<id>"``, ``emit_rhq_rhv_generics`` -> ``"HQ_<id>"``/``"HV_<id>"``,
#: ``emit_rhe_generics`` -> ``"RHE_<id>"``), recovering the special-constraint
#: register's own family + ``codigo_restricao`` without re-running the
#: conversion-time id-assignment sequence.
_CONSTRAINT_NAME_RE = re.compile(r"^(RE|HQ|HV|RHE)_(\d+)$")


def _term_lookup_value(
    term: ConstraintTerm, lookups: _DecompConstraintLookups, stage: int
) -> float | None:
    """One register term's per-stage value, or ``None`` when unavailable."""
    if term.variable == "generation":
        return lookups.hydro_generation.get((term.code, stage))
    if term.variable == "thermal_generation":
        return lookups.thermal_generation.get((term.code, stage))
    if term.variable == "VARM":
        return lookups.storage.get((term.code, stage))
    flow_lookup = lookups.flow.get(term.variable)
    return flow_lookup.get((term.code, stage)) if flow_lookup is not None else None


def _rhe_lhs_lookup(dec_oper_rhesoft: pl.DataFrame) -> dict[int, dict[int, float]]:
    """``{codigo_restricao: {stage_0based: valor_MW}}``.

    RHE constraints report their achieved LHS directly in
    ``DecOperRheSoft``: ``valor_MW`` is the achieved Σρ_acum·storage in the
    operation -- the same raw physical quantity cobre's own (slack-free)
    expression evaluation produces, so the two overlaid series compare on
    the same basis. Read it directly rather than re-deriving the
    ρ_acum-weighted cascade sum ``decomp.constraints.emit_rhe_generics``
    computes at conversion time, which would duplicate a large, easily
    drifting piece of machinery.

    The file's ``violacao_absoluta_MW`` is the shortfall against the target
    ``Meta`` (``valor_MW + violacao_absoluta_MW == Meta``, the constraint's
    *bound*, not its LHS) -- adding it would collapse the reference trace
    onto the bound line during every violation episode, exactly the moments
    the LHS-vs-bound chart exists to surface. Verified against real
    ``dec_oper_rhesoft.csv`` output.
    """
    out: dict[int, dict[int, float]] = {}
    if dec_oper_rhesoft.is_empty():
        return out
    collapsed = _scenario_mean(
        dec_oper_rhesoft,
        "estagio",
        ["valor_MW"],
        entity_column="codigo_restricao",
    )
    for row in collapsed.iter_rows(named=True):
        valor = row["valor_MW"]
        if valor is None:
            continue
        cid = int(row["codigo_restricao"])
        stage0 = int(row["estagio"]) - 1
        out.setdefault(cid, {})[stage0] = float(valor)
    return out


_GC_LHS_SCHEMA = {
    "constraint_id": pl.Int32,
    "stage_id": pl.Int32,
    "lhs_value": pl.Float64,
}


def _generic_constraint_lhs_decomp(
    decomp_dir: Path,
    cobre_output_dir: Path,
    gc_constraints: list[dict],
) -> pl.DataFrame:
    """Evaluate each generic constraint's LHS from the source model's own
    operation output (ticket-019).

    The DECOMP sibling of ``constraints_compare.evaluate_lhs_newave``: that
    function reads MEDIAS-USIH/int*.out through an ``EntityAlignment``/
    ``NewaveIdMap`` this source has neither of, so this evaluates each
    constraint's participating entities directly against
    ``dec_oper_usih``/``dec_oper_usit``/``dec_oper_rhesoft``.

    Per-family strategy:

    - **RE** (hydro/thermal generation): ``generation``/
      ``thermal_generation`` register terms resolve against
      ``dec_oper_usih``/``dec_oper_usit``'s own stage-aggregate row.
    - **HQ** (flow): ``QDEF``/``QTUR``/``QDES``/``QVER`` register terms
      resolve against the matching ``dec_oper_usih`` flow column.
    - **HV** (storage): ``VARM`` register terms resolve against
      ``dec_oper_usih``'s *useful* stage-final volume, floor-adjusted to
      cobre's absolute basis (see `_storage_lookup`).
    - **RHE** (soft stored-energy): read directly from ``DecOperRheSoft``
      (see `_rhe_lhs_lookup`) -- no register/census lookup needed.

    Deferred (cobre-only render, logged, never fabricated): RE
    ``interchange`` (FI) and HQ ``QBOM`` (pumping) terms -- see
    :data:`_UNSUPPORTED_TERM_VARIABLES`. A constraint whose name does not
    match a known family/id (:data:`_CONSTRAINT_NAME_RE`), or whose id has
    no matching register record (deck unavailable, or the record was not
    found), is likewise skipped whole.
    """
    if not gc_constraints:
        return pl.DataFrame(schema=_GC_LHS_SCHEMA)

    context = _decomp_constraint_context(decomp_dir)
    census_by_key: dict[tuple[str, int], ConstraintRecord] = {}
    storage_lookup: dict[tuple[int, int], float] = {}
    hydro_frame = _dec_oper_hydro_stage_frame(decomp_dir)
    if context is not None:
        for records in context.census.by_family.values():
            for record in records:
                census_by_key[(record.family, record.constraint_id)] = record
        min_storage = {
            i: float(m.get("min_storage_hm3") or 0.0)
            for i, m in cobre_readers.read_cobre_hydro_metadata(
                cobre_output_dir
            ).items()
        }
        storage_lookup = _storage_lookup(hydro_frame, context.id_map, min_storage)

    thermal_frame = _dec_oper_thermal_stage_frame(decomp_dir)
    lookups = _DecompConstraintLookups(
        hydro_generation=_stage_frame_to_lookup(hydro_frame, "geracao_MW"),
        thermal_generation=_stage_frame_to_lookup(thermal_frame, "geracao_MW"),
        flow={
            "QDEF": _stage_frame_to_lookup(hydro_frame, "vazao_defluente_m3s"),
            "QTUR": _stage_frame_to_lookup(hydro_frame, "vazao_turbinada_m3s"),
            "QDES": _stage_frame_to_lookup(hydro_frame, "vazao_desviada_m3s"),
            "QVER": _stage_frame_to_lookup(hydro_frame, "vazao_vertida_m3s"),
        },
        storage=storage_lookup,
    )
    rhe_lhs = _rhe_lhs_lookup(_dec_oper_rhesoft_frame(decomp_dir))

    rows: list[dict[str, object]] = []
    n_skipped = 0
    for c in gc_constraints:
        name = str(c.get("name", ""))
        cobre_cid = int(c["id"])
        match = _CONSTRAINT_NAME_RE.match(name)
        if match is None:
            _LOG.info(
                "Generic constraint %d (%r) has no recognized source-model "
                "family/id in its name; rendering cobre-only.",
                cobre_cid,
                name,
            )
            n_skipped += 1
            continue
        prefix, raw_cid = match.group(1), int(match.group(2))

        if prefix == "RHE":
            stage_values = rhe_lhs.get(raw_cid)
            if not stage_values:
                _LOG.info(
                    "RHE constraint %d (%s) has no dec_oper_rhesoft rows; "
                    "rendering cobre-only.",
                    cobre_cid,
                    name,
                )
                n_skipped += 1
                continue
            rows.extend(
                {"constraint_id": cobre_cid, "stage_id": stage, "lhs_value": value}
                for stage, value in stage_values.items()
            )
            continue

        record = census_by_key.get((prefix, raw_cid))
        if record is None:
            _LOG.info(
                "Generic constraint %d (%s) has no matching source-model "
                "register record (deck unavailable or record not found); "
                "rendering cobre-only.",
                cobre_cid,
                name,
            )
            n_skipped += 1
            continue
        unsupported = sorted(
            {
                t.variable
                for t in record.terms
                if t.variable in _UNSUPPORTED_TERM_VARIABLES
            }
        )
        if unsupported:
            _LOG.info(
                "Generic constraint %d (%s) carries a term type this "
                "comparator cannot re-derive from dec_oper_* output (%s); "
                "rendering cobre-only.",
                cobre_cid,
                name,
                ", ".join(unsupported),
            )
            n_skipped += 1
            continue

        constraint_rows = 0
        for stage in range(record.stage_start, record.stage_end + 1):
            lhs = 0.0
            complete = True
            for term in record.terms:
                value = _term_lookup_value(term, lookups, stage)
                if value is None:
                    complete = False
                    break
                lhs += term.coefficient * value
            if complete:
                rows.append(
                    {"constraint_id": cobre_cid, "stage_id": stage, "lhs_value": lhs}
                )
                constraint_rows += 1
        if constraint_rows == 0:
            n_skipped += 1

    if n_skipped:
        _LOG.info(
            "DECOMP-side generic-constraint LHS: %d of %d constraint(s) "
            "rendered cobre-only (no source-model LHS derivable).",
            n_skipped,
            len(gc_constraints),
        )
    if not rows:
        return pl.DataFrame(schema=_GC_LHS_SCHEMA)
    return pl.DataFrame(rows).with_columns(
        pl.col("constraint_id").cast(pl.Int32),
        pl.col("stage_id").cast(pl.Int32),
        pl.col("lhs_value").cast(pl.Float64),
    )


def build_decomp_dataset(
    decomp_dir: Path, cobre_output_dir: Path, *, tolerance: float = 1e-2
) -> ComparisonDataset:
    """Build the canonical results dataset for the current 8 DECOMP variables.

    Reads and aligns both sides via :func:`_read_aligned_frames`, then emits
    the canonical :class:`~cobre_bridge.comparators.results.ResultComparison`
    shape and assembles it through the shared, source-agnostic
    :func:`~cobre_bridge.comparators.analyze.build_results_dataset` stat
    kernel. This is the reuse engine every later epic extends by filling more
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
    the data. As of ticket-008, ``line``/``line_bounds``/``line_meta`` carry
    the Network tab's cobre percentile band, per-stage capacity bounds, and
    line metadata (see :func:`_line_bounds_and_meta`). As of ticket-010, the
    Overview tab's cost sections are filled: ``nw_costs``/``cobre_costs``
    (the NPV breakdown dicts, from :func:`_cost_frames` and
    :func:`cobre_readers.read_cobre_cost_breakdown`) and
    ``cobre_stage_costs`` (from :func:`cobre_readers.read_cobre_stage_costs`);
    ``nw_sin`` additionally carries the ``COPER``/``CUSTO_FUTURO``/``CTERM``
    per-stage cost rows, unioned onto the ticket-006 EARM/ENA rows via
    :func:`_union_cost_rows` rather than overwriting them. As of ticket-012,
    ``nw_convergence``/``cobre_convergence`` carry the Overview tab's
    Convergence overlay (see :func:`_decomp_convergence_frame` and
    :func:`cobre_readers.read_cobre_convergence`). As of ticket-013, the
    Performance tab's timing
    metadata is filled: ``nw_tim_stages``/``nw_tim_iterations`` (see
    :func:`_decomp_tim_stages`/:func:`_decomp_tim_iterations`, sourced from
    ``decomp.tim``/``relato.convergencia``), ``nw_max_stage`` (see
    :func:`_decomp_max_stage`), and ``cobre_training_seconds``/
    ``cobre_iteration_timing`` (verbatim from the matching
    :mod:`cobre_readers` functions). Per Caveat #2, ``nw_tim_iterations``
    never carries a ``forward_seconds``/``backward_seconds`` split — DECOMP
    has no forward/backward pass structure to split. As of ticket-014, the
    Hydro Operation and Hydro Plant Details tabs are filled: ``hydro`` (the
    cobre percentile band, from
    :func:`cobre_readers.read_cobre_hydro_percentiles`),
    ``cobre_hydro_meta`` (:func:`cobre_readers.read_cobre_hydro_metadata`
    merged with the plant->bus label via :func:`_merge_hydro_bus_ids`, so
    every entry carries a ``"bus_ids"`` key), and
    ``cobre_hydro_per_stage_bounds`` (verbatim from
    :func:`cobre_readers.read_cobre_hydro_per_stage_bounds`).
    ``nw_hydro_slacks`` stays at its empty dataclass default — DECOMP has no
    withdrawal/evaporation-slack analog — so the slack charts render the
    cobre Mean + p10/p90 band only (``report_builder``'s documented
    ``has_newave=False`` path). As of ticket-015, ``thermal`` carries the
    Thermal Operation/Plant Details tabs' cobre percentile band (from
    :func:`cobre_readers.read_cobre_thermal_percentiles`) — the disjoint
    thermal counterpart to ticket-014's ``hydro`` field; the thermal
    ``ResultComparison`` rows themselves are unchanged (E1). As of ticket-016,
    the Productivity tab's realized per-stage half is filled: a
    ``productivity_mw_per_m3s`` hydro ``ResultComparison`` row is derived per
    (plant, stage) from the ``generation_mw``/``turbined_m3s`` rows E1 already
    produced (see :func:`_hydro_productivity_results`) and appended onto
    ``results``, so
    :func:`~cobre_bridge.comparators.analyze.build_results_dataset` fills
    ``dataset.metadata["productivity_per_stage"]`` the same way it does for
    the source model. ``productivity_detail`` (the pmo-derived static
    point/equivalent/accumulated scatter + building-blocks table) stays at its
    empty dataclass default — DECOMP ships no ``pmo.dat``, so that half has no
    DECOMP counterpart and the report's static section renders its "No
    productivity data available" path instead of a fabricated comparison.
    As of ticket-017, ``fpha_metrics`` carries the Productivity tab's
    "Fitted production functions (FPHA)" section fidelity table (see
    :func:`_fpha_metrics` and its preceding [ASSUMPTION] comment for why this
    is fallback (b) -- fit-fidelity statistics only, evaluated at the source
    model's own realized operating points against Cobre's fitted envelope --
    rather than the full grid-reconstructed surface overlay the source model
    path renders). ``fpha_surface``/``fpha_spill`` stay at their empty
    dataclass default; the section is entirely omitted (never a fabricated
    partial render) whenever either side has no fitted planes at all.
    As of ticket-018, ``results`` additionally carries ``entity_type="ree"``
    rows (``ena_mwmes``/``earm_final_mwmes``): the source model's own
    `read_dec_oper_ree` per-REE energy against a membership-weighted
    (`read_relato_membership`) sum of Cobre's per-plant
    ``incremental_inflow_energy_mw``/``stored_energy_final_mwh`` (see
    :func:`_ree_result_comparisons`; EARM is reconciled MWh -> MWmês via
    :data:`_EARM_MWH_TO_MWMES`). No dedicated ``PercentileData`` field
    exists for this -- unlike every earlier ticket's tab metadata, the REE
    section reads straight off the ``results`` rows the same way
    ``dataset.metadata["productivity_per_stage"]`` does.
    As of ticket-019, the Constraints tab's ``gc_constraints``/``gc_bounds``/
    ``gc_lhs_cobre`` are filled verbatim from ``constraints_compare``
    (source-agnostic: constraint/bounds tables and the Cobre-side simulation
    LHS); ``gc_lhs_newave`` is the DECOMP-side LHS, derived per constraint
    family from ``dec_oper_usih``/``dec_oper_usit``/``dec_oper_rhesoft`` and
    the special-constraint register (see
    :func:`_generic_constraint_lhs_decomp`'s docstring for the per-family
    strategy and its deliberately-deferred subset). ``nw_max_stage`` (already
    set by ticket-013) doubles as this section's stage clamp.
    As of ticket-020, ``results`` additionally carries a per-(hydro, stage)
    ``entity_type="hydro"``/``variable="evaporation_m3s"`` row: the source
    model's own ``dec_oper_evap`` volume (hm³) reconciled into a mean flow
    (m³/s) via the stage's actual hours and compared against Cobre's own
    ``evaporation_m3s`` (see :func:`_evaporation_result_comparisons`'s
    docstring, including the TRACKED COBRE-GAP WORKAROUND (C11) it
    surfaces rather than corrects away). No dedicated ``PercentileData``
    field exists for it — like ticket-018's REE rows, it flows straight
    through ``results``/``dataset.metadata["unmapped"]["evaporation"]``, and
    is rendered by the existing Hydro Plant Details tab's per-plant
    ``evaporation_m3s`` panel (``charts._HYDRO_VARIABLES``), not a new chart.
    Every other field stays at its empty default, and
    ``bus``/``hydro``/``thermal``/``line`` themselves stay empty whenever the
    Cobre run has no percentile output (e.g. the deterministic 2-node tree) —
    no percentile spread is ever fabricated.

    Args:
        decomp_dir: The deck directory (results resolved via the ticket-001
            union discovery).
        cobre_output_dir: Cobre's output directory.
        tolerance: Relative tolerance forwarded to the summary builder.

    Returns:
        The validated dataset. ``metadata["unmapped"]`` carries the per-level
        (hydro/thermal/bus/line/ree) reference codes/corridors the id map,
        the corridor alignment, or the REE membership table could not
        resolve, so they survive into the dataset instead of being silently
        dropped.
    """
    from cobre_bridge.comparators.analyze import build_results_dataset

    aligned = _read_aligned_frames(decomp_dir, cobre_output_dir)

    results: list[ResultComparison] = []
    # ticket-016: captured separately so the derived realized productivity
    # (generation / turbined, per plant/stage) can be appended right after
    # -- see _hydro_productivity_results.
    hydro_results = _result_comparisons(
        aligned.source_hydro,
        aligned.cobre_hydro,
        _HYDRO_VARIABLES,
        names=aligned.hydro_names,
    )
    results.extend(hydro_results)
    results.extend(_hydro_productivity_results(hydro_results))
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

    # --- ticket-008: Network tab (line rows + line/line_bounds/line_meta) ---
    line_bounds, line_meta = _line_bounds_and_meta(cobre_output_dir)
    line_id_map = _build_line_id_map(decomp_dir)
    line_results, unresolved_lines = _line_result_comparisons(
        decomp_dir, cobre_output_dir, line_id_map, line_meta
    )
    results.extend(line_results)

    # --- ticket-010: Overview cost metadata (nw_costs / cobre_costs /
    # nw_sin cost rows / cobre_stage_costs) ---
    nw_costs, nw_cost_rows = _cost_frames(decomp_dir)
    cobre_costs = cobre_readers.read_cobre_cost_breakdown(cobre_output_dir)
    cobre_stage_costs = cobre_readers.read_cobre_stage_costs(cobre_output_dir)
    nw_sin = _union_cost_rows(aligned.nw_sin, nw_cost_rows)

    # --- ticket-013: Performance tab timing metadata (Caveat #2: no
    # fabricated DECOMP forward/backward split) ---
    nw_tim_stages = _decomp_tim_stages(decomp_dir)
    nw_tim_iterations = _decomp_tim_iterations(decomp_dir)
    nw_max_stage = _decomp_max_stage(aligned)
    cobre_training_seconds = cobre_readers.read_cobre_training_duration(
        cobre_output_dir
    )
    cobre_iteration_timing = cobre_readers.read_cobre_iteration_timing(cobre_output_dir)

    # --- ticket-014: Hydro Operation + Plant Details tab metadata ---
    cobre_hydro_meta = _merge_hydro_bus_ids(
        cobre_readers.read_cobre_hydro_metadata(cobre_output_dir),
        cobre_readers.read_cobre_hydro_bus_labels(cobre_output_dir),
    )

    # --- ticket-017: Productivity tab's "Fitted production functions
    # (FPHA)" section. Reuses ``line_id_map`` (ticket-008) rather than
    # rebuilding it a third time -- it already carries every hydro code ->
    # Cobre id mapping this needs, and every fixture built before this
    # ticket that leaves it ``None`` (no deck to read) keeps working
    # unchanged, since `_fpha_metrics` treats ``id_map is None`` as "no FPHA
    # section" rather than raising.
    fpha_metrics = _fpha_metrics(
        decomp_dir, cobre_output_dir, line_id_map, aligned.hydro_names
    )

    # --- ticket-018: REE energy rollup via membership. Reuses ``line_id_map``
    # (ticket-008) exactly like ``_fpha_metrics`` above -- the same
    # hydro-code -> Cobre-id mapping, no third ``DecompIdMap`` rebuild.
    ree_results, unmapped_ree = _ree_result_comparisons(
        decomp_dir, aligned.cobre_hydro, line_id_map
    )
    results.extend(ree_results)

    # --- ticket-020: evaporation comparison (hydro, "evaporation_m3s").
    # Reuses ``line_id_map`` exactly like ``_fpha_metrics``/
    # ``_ree_result_comparisons`` above -- the same hydro-code -> Cobre-id
    # mapping, no fifth ``DecompIdMap`` rebuild. See
    # ``_evaporation_result_comparisons``'s docstring for the hm³ -> m³/s
    # reconciliation and the TRACKED COBRE-GAP WORKAROUND (C11) it surfaces.
    evaporation_results, unmapped_evaporation = _evaporation_result_comparisons(
        decomp_dir,
        cobre_output_dir,
        aligned.cobre_hydro,
        line_id_map,
        aligned.hydro_names,
    )
    results.extend(evaporation_results)

    # --- ticket-019: Constraints tab (gc_* metadata). The cobre-side pieces
    # (constraint/bounds tables, simulation LHS) are source-agnostic and
    # reused verbatim from `constraints_compare`; only the DECOMP-side LHS
    # (`_generic_constraint_lhs_decomp`) is new -- see that function's
    # docstring for the per-family derivation.
    from cobre_bridge.comparators.constraints_compare import (
        _load_generic_constraint_bounds,
        _load_generic_constraints,
        evaluate_lhs_cobre,
    )

    cobre_case_dir = case_dir_for(cobre_output_dir)
    gc_constraints = _load_generic_constraints(cobre_case_dir)
    gc_bounds_df = _load_generic_constraint_bounds(cobre_case_dir)
    if gc_constraints:
        gc_lhs_cb = evaluate_lhs_cobre(gc_constraints, cobre_output_dir)
        gc_lhs_nw = _generic_constraint_lhs_decomp(
            decomp_dir, cobre_output_dir, gc_constraints
        )
    else:
        gc_lhs_cb = pl.DataFrame()
        gc_lhs_nw = pl.DataFrame()

    pct = PercentileData(
        nw_bus_names=aligned.bus_names,
        nw_hydro_names=aligned.hydro_names,
        bus=cobre_readers.read_cobre_bus_percentiles(cobre_output_dir),
        nw_market=aligned.nw_market,
        nw_net_load=aligned.nw_net_load,
        nw_sin=nw_sin,
        # D-STAGE-OFFSET: fixed at 1, not derived — see the docstring above.
        nw_offset=1,
        # ticket-012: Overview tab's Convergence overlay.
        nw_convergence=_decomp_convergence_frame(decomp_dir),
        cobre_convergence=cobre_readers.read_cobre_convergence(cobre_output_dir),
        nw_costs=nw_costs,
        cobre_costs=cobre_costs,
        cobre_stage_costs=cobre_stage_costs,
        bus_aggregates=cobre_readers.read_cobre_bus_aggregates(cobre_output_dir),
        cobre_bus_meta=cobre_readers.read_cobre_bus_metadata(cobre_output_dir),
        cobre_hydro_means=cobre_readers.read_cobre_hydro_means(cobre_output_dir),
        line=cobre_readers.read_cobre_line_percentiles(cobre_output_dir),
        line_bounds=line_bounds,
        line_meta=line_meta,
        # ticket-013: Performance tab timing metadata.
        nw_tim_stages=nw_tim_stages,
        nw_tim_iterations=nw_tim_iterations,
        nw_max_stage=nw_max_stage,
        cobre_training_seconds=cobre_training_seconds,
        cobre_iteration_timing=cobre_iteration_timing,
        # ticket-014: Hydro Operation + Plant Details tab metadata. Never
        # fabricate a percentile spread when the Cobre run has no hydro
        # percentile output (e.g. the deterministic 2-node tree) --
        # ``read_cobre_hydro_percentiles`` already degrades to an empty
        # frame in that case, passed through verbatim.
        hydro=cobre_readers.read_cobre_hydro_percentiles(cobre_output_dir),
        cobre_hydro_meta=cobre_hydro_meta,
        cobre_hydro_per_stage_bounds=cobre_readers.read_cobre_hydro_per_stage_bounds(
            cobre_output_dir
        ),
        # DECOMP has no direct withdrawal/evaporation-slack analog --
        # stays at the dataclass empty-frame default (never set to a
        # DECOMP-derived source) so the slack charts render the cobre
        # Mean + p10/p90 band only (report_builder's has_newave=False path).
        nw_hydro_slacks=pl.DataFrame(),
        # ticket-015: Thermal Operation + Plant Details tab metadata.
        # Mirrors ticket-014's hydro band exactly -- never fabricate a
        # percentile spread when the Cobre run has no thermal percentile
        # output (e.g. the deterministic 2-node tree);
        # ``read_cobre_thermal_percentiles`` already degrades to an empty
        # frame in that case, passed through verbatim. The thermal
        # ``ResultComparison`` rows themselves are untouched -- E1 already
        # emits them above via ``_result_comparisons``.
        thermal=cobre_readers.read_cobre_thermal_percentiles(cobre_output_dir),
        # ticket-016: DECOMP ships no pmo.dat, so the static
        # point/equivalent/accumulated productivity scatter and the
        # building-blocks table have no DECOMP counterpart -- stays at the
        # dataclass empty-frame default (never fabricated) so the Productivity
        # tab's static section renders its "No productivity data available"
        # path. The *realized* per-stage productivity is still filled -- via
        # the ``productivity_mw_per_m3s`` hydro rows appended to ``results``
        # above (see _hydro_productivity_results), not through this field.
        productivity_detail=pl.DataFrame(),
        # ticket-017: Productivity tab's Fitted production functions (FPHA)
        # section -- fallback (b), see the [ASSUMPTION] comment above
        # `_fpha_metrics`. ``fpha_surface``/``fpha_spill`` stay at their
        # dataclass ``None`` default (never fabricated): the declared FPHA
        # readers carry no fitted plane coefficients to reconstruct a dense
        # surface from, only per-hydro/stage fit-fidelity statistics.
        fpha_metrics=fpha_metrics,
        # ticket-019: Constraints tab LHS-vs-bound metadata. ``nw_max_stage``
        # is unchanged (already set above by ticket-013's `_decomp_max_stage`)
        # and doubles as this section's stage clamp (D-STAGE-OFFSET).
        gc_constraints=gc_constraints,
        gc_bounds=gc_bounds_df,
        gc_lhs_newave=gc_lhs_nw,
        gc_lhs_cobre=gc_lhs_cb,
    )
    dataset = build_results_dataset(results, pct, tolerance)
    dataset.metadata["unmapped"] = {
        level: [int(code) for code in codes]
        for level, codes in aligned.unmapped.items()
    }
    dataset.metadata["unmapped"]["line"] = unresolved_lines
    dataset.metadata["unmapped"]["ree"] = unmapped_ree
    dataset.metadata["unmapped"]["evaporation"] = unmapped_evaporation
    return dataset
