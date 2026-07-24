"""Scenario conversion for DECOMP-like decks (the explicit inflow tree).

The inflow file carries the deterministic trunk (weekly forecasts, one
scenario per weekly stage) and the terminal fan (generated scenarios with
per-node probabilities). Values are natural flows per gauging station;
they become per-plant incrementals by subtracting the direct *operated*
upstream stations (water routed through non-operated intermediates is
attributed to the next operated plant, matching the registry cascade
walk).

Under the identity stochastic convention (μ = 0, σ = 1, order 0) the
standardized noise equals the natural value, so the same numbers feed
both the forward external scenarios and the backward opening tree
(``noise_openings``) — the two files describe one tree by construction.
``scenario_probabilities.parquet`` is emitted in the agreed shape for the
enumeration work; current solver versions ignore it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.decomp.hydro import _downstream_operated

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Vazoes

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_PROBABILITY_ATOL = 1e-4


def _incremental_context(
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
) -> tuple[dict[int, str], dict[int, list[int]]]:
    """Per-plant station column and direct operated-upstream stations."""
    operated = set(id_map.hydro_codes)
    station_by_code = {
        code: str(int(hidr.loc[code, "posto"])) for code in id_map.hydro_codes
    }
    parents: dict[int, list[int]] = {code: [] for code in id_map.hydro_codes}
    for code in id_map.hydro_codes:
        downstream = _downstream_operated(hidr, code, operated)
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
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> dict[tuple[int, int], list[float]]:
    """``{(stage_index, scenario_index): incrementals}`` for the whole tree."""
    station_by_code, parents = _incremental_context(hidr, id_map)
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
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """``external_inflow_scenarios.parquet``: the tree in natural units."""
    values = _tree_values(vazoes, hidr, id_map, calendar)

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


def convert_noise_openings(
    vazoes: Vazoes,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """``noise_openings.parquet``: the same tree as the backward openings.

    Under the identity convention the standardized noise equals the
    natural incremental value, and with deterministic load the noise
    vector is exactly the hydro block in id order — so
    ``entity_index = hydro_id`` and the two scenario files carry identical
    numbers by construction.
    """
    values = _tree_values(vazoes, hidr, id_map, calendar)

    stage_ids: list[int] = []
    opening_indices: list[int] = []
    entity_indices: list[int] = []
    noise: list[float] = []
    for (stage_index, scenario_index), incrementals in sorted(values.items()):
        for hydro_id, value in enumerate(incrementals):
            stage_ids.append(stage_index)
            opening_indices.append(scenario_index)
            entity_indices.append(hydro_id)
            noise.append(value)

    return pa.table(
        {
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "opening_index": pa.array(opening_indices, type=pa.uint32()),
            "entity_index": pa.array(entity_indices, type=pa.uint32()),
            "value": pa.array(noise, type=pa.float64()),
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
    """``scenario_probabilities.parquet``: per-(stage, scenario) weights.

    Emitted in the agreed enumeration-input shape (trunk rows carry
    probability 1.0); validated to sum to 1 per stage.
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
