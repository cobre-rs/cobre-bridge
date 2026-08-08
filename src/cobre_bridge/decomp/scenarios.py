"""Scenario conversion for DECOMP-like decks (the explicit inflow tree).

The inflow file carries the deterministic trunk (weekly forecasts, one
scenario per weekly stage) and the terminal fan (generated scenarios with
per-node probabilities). Values are natural flows per gauging station;
they become per-plant incrementals by subtracting the direct *operated*
upstream stations (water routed through non-operated intermediates is
attributed to the next operated plant, matching the registry cascade
walk).

The tree is emitted node-natively: every stage draws its openings from
``external_inflow_scenarios.parquet`` (trunk column 0, terminal fan
columns ``0..N-1``), and the per-scenario weights become the terminal
branch-edge probabilities on the ``policy_graph`` (see
``temporal.convert_stages``). Under the identity stochastic convention
(μ = 0, σ = 1, order 0) the standardized noise equals the natural value.
``terminal_fan_probabilities`` supplies those fan weights;
``convert_scenario_probabilities`` remains a validation view for
``check decomp`` and is no longer written to the case directory.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.decomp.hydro import _downstream_operated

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Vazoes

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

_PROBABILITY_ATOL = 1e-4


def _incremental_context(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Per-plant station column and direct operated-upstream stations, off
    the *effective* (post-``AC NUMJUS``/``NUMPOS``) topology (ticket-014).

    Stage-agnostic by design (one cascade for the whole horizon): both the
    station column and the downstream link are read at stage 0
    (:meth:`~cobre_bridge.decomp.cadastro.EffectiveCadastro.inflow_gauge`/
    :func:`~cobre_bridge.decomp.hydro._downstream_operated`'s own default).
    A plant whose effective gauge varies across stages (a temporal ``AC
    NUMPOS``) gets a tracked-gap warning here; the downstream sibling gap is
    logged inside ``_downstream_operated`` itself, which this function also
    walks per plant.
    """
    operated = set(id_map.hydro_codes)
    station_by_code: dict[int, str] = {}
    for code in id_map.hydro_codes:
        if effective.inflow_gauge_varies(code):
            _LOG.warning(
                "plant %d's inflow gauge (AC NUMPOS) varies across stages; "
                "per-stage gauge attribution is not modeled (deferred "
                "fidelity) -- using the stage-0 effective gauge for the "
                "whole horizon",
                code,
            )
        station_by_code[code] = str(effective.inflow_gauge(code, 0))
    parents: dict[int, list[int]] = {code: [] for code in id_map.hydro_codes}
    for code in id_map.hydro_codes:
        downstream = _downstream_operated(effective, code, operated)
        if downstream is not None:
            parents[downstream].append(code)
    return station_by_code, parents


def _incremental_values(
    row: pd.Series,
    id_map: DecompIdMap,
    station_by_code: dict[int, str],
    parents: dict[int, list[int]],
) -> list[float]:
    """Natural station flows → per-plant incrementals, in hydro-id order."""
    natural = {code: float(row[station]) for code, station in station_by_code.items()}
    return [
        natural[code] - sum(natural[u] for u in parents[code])
        for code in id_map.hydro_codes
    ]


def _tree_values(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> dict[tuple[int, int], list[float]]:
    """``{(stage_index, scenario_index): incrementals}`` for the whole tree."""
    station_by_code, parents = _incremental_context(effective, id_map)
    terminal = len(calendar) - 1

    previsoes = vazoes.previsoes
    if previsoes is None or previsoes.empty:
        raise ValueError("the inflow file has no trunk forecasts (previsoes)")
    if len(previsoes) != terminal:
        raise ValueError(
            f"{len(previsoes)} trunk forecast rows for {terminal} weekly stages"
        )

    values: dict[tuple[int, int], list[float]] = {}
    for _, row in previsoes.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < terminal:
            raise ValueError(
                f"trunk forecast stage {int(row['estagio'])} outside the "
                f"weekly range (1..{terminal})"
            )
        values[(stage_index, 0)] = _incremental_values(
            row, id_map, station_by_code, parents
        )

    cenarios = vazoes.cenarios_gerados
    if cenarios is None or cenarios.empty:
        raise ValueError("the inflow file has no terminal fan (cenarios_gerados)")
    for _, row in cenarios.iterrows():
        stage_index = int(row["estagio"]) - 1
        if stage_index != terminal:
            raise ValueError(
                f"fan scenario at stage {int(row['estagio'])}; expected only "
                f"the terminal stage {terminal + 1} — a pre-terminal branching "
                "deck needs the node-graph work"
            )
        values[(stage_index, int(row["cenario"]) - 1)] = _incremental_values(
            row, id_map, station_by_code, parents
        )
    return values


def convert_external_inflows(
    vazoes: Vazoes,
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """``external_inflow_scenarios.parquet``: the tree in natural units, off
    the *effective* (post-``AC NUMJUS``/``NUMPOS``) topology (ticket-014)."""
    values = _tree_values(vazoes, effective, id_map, calendar)

    stage_ids: list[int] = []
    scenario_ids: list[int] = []
    hydro_ids: list[int] = []
    flows: list[float] = []
    for (stage_index, scenario_index), incrementals in sorted(values.items()):
        for hydro_id, value in enumerate(incrementals):
            stage_ids.append(stage_index)
            scenario_ids.append(scenario_index)
            hydro_ids.append(hydro_id)
            flows.append(value)

    return pa.table(
        {
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "scenario_id": pa.array(scenario_ids, type=pa.int32()),
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "value_m3s": pa.array(flows, type=pa.float64()),
        }
    )


def terminal_fan_probabilities(
    vazoes: Vazoes,
    calendar: Sequence[OperativeStage],
) -> list[float]:
    """Terminal-stage per-scenario probabilities, ordered by scenario id.

    The DECOMP fan probabilities (``vazoes.probabilidades`` at the terminal
    stage) become the terminal branch-edge weights on the ``policy_graph``
    node graph; cobre re-normalizes each source's out-edges at load. The
    returned list is indexed by 0-based scenario id (``cenario - 1``), the
    same id the external inflow library binds via each fan node's
    ``scenario_id``. Validated to be a contiguous 0-based range summing to 1.
    """
    prob = vazoes.probabilidades
    if prob is None or prob.empty:
        raise ValueError("the inflow file has no probability table")

    terminal = len(calendar)  # 1-based id of the terminal (fan) stage
    rows = prob[prob["estagio"].astype(int) == terminal]
    if rows.empty:
        raise ValueError(
            f"the inflow probability table has no rows for the terminal "
            f"stage {terminal}"
        )

    by_scenario: dict[int, float] = {}
    for _, row in rows.iterrows():
        by_scenario[int(row["cenario"]) - 1] = float(row["probabilidade"])

    if set(by_scenario) != set(range(len(by_scenario))):
        raise ValueError(
            "terminal fan scenarios are not a contiguous 0-based range: "
            f"{sorted(by_scenario)}"
        )
    ordered = [by_scenario[k] for k in range(len(by_scenario))]

    total = sum(ordered)
    if abs(total - 1.0) > _PROBABILITY_ATOL:
        raise ValueError(f"terminal fan probabilities sum to {total}, expected 1.0")
    return ordered


def deterministic_external_scenarios(
    stats: pa.Table,
    *,
    entity_column: str,
    value_in: str,
    value_out: str,
    scenario_counts: Sequence[int],
) -> pa.Table:
    """Replicate a deterministic per-(entity, stage) value across external columns.

    ``stats`` carries one row per (``entity_column``, ``stage_id``) with the
    deterministic value in ``value_in`` (load base MW, or NCS availability
    fraction). Each row expands to ``scenario_counts[stage_id]`` external rows
    (``scenario_id`` ``0..n-1``) with the value repeated, so the library's
    per-stage column count matches the inflow library (1 on the deterministic
    trunk, the terminal fan width). This satisfies cobre's node-native rule
    that every non-empty class at an external-column node be external: the
    DECOMP load and NCS are deterministic, so a single value fans out unchanged.
    """
    entities = stats.column(entity_column).to_pylist()
    stages = stats.column("stage_id").to_pylist()
    values = stats.column(value_in).to_pylist()

    rows: list[tuple[int, int, int, float]] = []
    for entity, stage, value in zip(entities, stages, values, strict=True):
        for scenario_id in range(scenario_counts[int(stage)]):
            rows.append((int(stage), scenario_id, int(entity), float(value)))
    # Sort by (stage_id, scenario_id, entity) to match the inflow emitter and
    # cobre's canonical external-library order.
    rows.sort()

    return pa.table(
        {
            "stage_id": pa.array([r[0] for r in rows], type=pa.int32()),
            "scenario_id": pa.array([r[1] for r in rows], type=pa.int32()),
            entity_column: pa.array([r[2] for r in rows], type=pa.int32()),
            value_out: pa.array([r[3] for r in rows], type=pa.float64()),
        }
    )


def convert_inflow_stats_identity(
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """Identity stats (μ = 0, σ = 1) — the explicit-inflow convention.

    The stochastic model is plumbing here: with these values the
    standardized noise is the natural inflow itself, and no fan stage can
    hit the zero-σ pathology.
    """
    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    for hydro_id in range(len(id_map.hydro_codes)):
        for stage in calendar:
            hydro_ids.append(hydro_id)
            stage_ids.append(stage.index)
    n = len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "mean_m3s": pa.array([0.0] * n, type=pa.float64()),
            "std_m3s": pa.array([1.0] * n, type=pa.float64()),
        }
    )


def convert_scenario_probabilities(
    vazoes: Vazoes,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """Per-(stage, scenario) probability view for validation.

    Trunk rows carry probability 1.0; validated to sum to 1 per stage. No
    longer written to the case directory (the terminal weights now live on
    the ``policy_graph`` transitions); ``check decomp`` still uses this to
    validate the deck's probability table.
    """
    prob = vazoes.probabilidades
    if prob is None or prob.empty:
        raise ValueError("the inflow file has no probability table")

    stage_ids: list[int] = []
    scenario_ids: list[int] = []
    probabilities: list[float] = []
    for _, row in prob.iterrows():
        stage_index = int(row["estagio"]) - 1
        if not 0 <= stage_index < len(calendar):
            raise ValueError(
                f"probability row at stage {int(row['estagio'])} outside the "
                f"calendar (1..{len(calendar)})"
            )
        stage_ids.append(stage_index)
        scenario_ids.append(int(row["cenario"]) - 1)
        probabilities.append(float(row["probabilidade"]))

    sums: dict[int, float] = {}
    for stage_index, probability in zip(stage_ids, probabilities, strict=True):
        sums[stage_index] = sums.get(stage_index, 0.0) + probability
    for stage_index, total in sorted(sums.items()):
        if abs(total - 1.0) > _PROBABILITY_ATOL:
            raise ValueError(
                f"stage {stage_index}: scenario probabilities sum to {total}, "
                "expected 1.0"
            )

    return pa.table(
        {
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "scenario_id": pa.array(scenario_ids, type=pa.int32()),
            "probability": pa.array(probabilities, type=pa.float64()),
        }
    )
