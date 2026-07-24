"""Operative-calendar temporal conversion for DECOMP-like decks.

Builds the stage calendar (dates, blocks, seasons) from the deck's start
date (``DT``) and per-stage block durations (``DP``), per
``plans/decomp-converter-core.md`` §1.1 and the operative-calendar rules of
``plans/decomp-round-2-revision.md`` §2.2:

- the study starts on a Saturday; weekly stages are exactly 168 h and break
  on Saturdays; the single final stage aggregates the second operative
  month and ends at a calendar month boundary;
- every weekly stage carries the **first operative month's** season (the
  month containing its first week's Friday). A straddling head week's
  previous-month days carry no accumulation weight — that month is closed —
  and a straddling tail week's next-month days spill over into the second
  month's accumulator, which the final stage then completes;
- the final stage carries its own calendar month's season.

Season ids follow the shared 0-based convention (Jan=0 … Dec=11) via
:func:`cobre_bridge.converters.temporal.monthly_season_definitions`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from cobre_bridge.converters.temporal import _block_names, monthly_season_definitions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

_SATURDAY = 5
_WEEK_HOURS = 168.0
_HOURS_PER_DAY = 24.0
_STAGES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/stages.schema.json"
)


@dataclass(frozen=True)
class OperativeStage:
    """One stage of the operative calendar."""

    index: int
    start_date: date
    end_date: date  # exclusive
    season_id: int  # 0-based calendar month (Jan=0 … Dec=11)
    block_hours: tuple[float, ...]

    @property
    def total_hours(self) -> float:
        """Total stage duration in hours."""
        return sum(self.block_hours)


def build_operative_calendar(
    start_date: date,
    stage_block_hours: Sequence[Sequence[float]],
) -> list[OperativeStage]:
    """Derive the operative stage calendar from block durations.

    ``stage_block_hours[s][b]`` is the duration in hours of block *b* of
    stage *s*. All stages but the last must total exactly one operative
    week (168 h); the last stage aggregates the second operative month and
    must end at a calendar month boundary.

    Raises
    ------
    ValueError
        If the calendar violates the operative rules (non-Saturday start,
        non-weekly stage, fractional days, or a final stage that does not
        close the second operative month).
    """
    if len(stage_block_hours) < 2:
        raise ValueError(
            "an operative calendar has at least two stages "
            f"(N weekly + 1 monthly); got {len(stage_block_hours)}"
        )
    if start_date.weekday() != _SATURDAY:
        raise ValueError(
            f"operative weeks begin on Saturdays; start date {start_date} "
            f"is a {start_date:%A}"
        )

    first_month_date = start_date + timedelta(days=6)
    first_month = first_month_date.month

    stages: list[OperativeStage] = []
    cursor = start_date
    last_index = len(stage_block_hours) - 1
    for index, hours in enumerate(stage_block_hours):
        block_hours = tuple(float(h) for h in hours)
        total = sum(block_hours)
        days = total / _HOURS_PER_DAY
        if abs(days - round(days)) > 1e-9:
            raise ValueError(
                f"stage {index}: total duration {total} h is not a whole number of days"
            )
        if index < last_index and total != _WEEK_HOURS:
            raise ValueError(
                f"stage {index}: every stage before the last must be one "
                f"operative week (168 h); got {total} h"
            )
        end = cursor + timedelta(days=round(days))
        if index < last_index:
            season_id = first_month - 1
        else:
            if end.day != 1:
                raise ValueError(
                    f"final stage must end at a calendar month boundary; ends {end}"
                )
            second_month = first_month % 12 + 1
            if cursor.month != second_month:
                raise ValueError(
                    f"final stage must aggregate the second operative month "
                    f"(month {second_month}); starts {cursor}"
                )
            season_id = cursor.month - 1
        stages.append(
            OperativeStage(
                index=index,
                start_date=cursor,
                end_date=end,
                season_id=season_id,
                block_hours=block_hours,
            )
        )
        cursor = end

    return stages


def operative_calendar_from_dadger(dadger: Dadger) -> list[OperativeStage]:
    """Build the operative calendar from a deck's ``DT`` + ``DP`` records.

    Block durations must agree across subsystems for each stage (the deck
    defines one temporal grid); a mismatch is a hard error.
    """
    dt = dadger.dt
    start = date(int(dt.ano), int(dt.mes), int(dt.dia))

    dp = dadger.dp(df=True)
    if dp is None or dp.empty:
        raise ValueError("the deck has no DP records; cannot build the calendar")

    stage_block_hours: list[list[float]] = []
    expected = 1
    for estagio, group in dp.groupby("estagio", sort=True):
        if int(estagio) != expected:
            raise ValueError(
                f"DP stages must be contiguous from 1; expected {expected}, "
                f"got {int(estagio)}"
            )
        expected += 1

        n_blocks = int(group["numero_patamares"].iloc[0])
        columns = [f"duracao_{k}" for k in range(1, n_blocks + 1)]
        reference: list[float] | None = None
        for _, row in group.iterrows():
            if int(row["numero_patamares"]) != n_blocks:
                raise ValueError(
                    f"stage {int(estagio)}: numero_patamares differs across subsystems"
                )
            values = [float(row[c]) for c in columns]
            if reference is None:
                reference = values
            elif values != reference:
                raise ValueError(
                    f"stage {int(estagio)}: block durations differ across "
                    f"subsystems ({values} vs {reference}); the deck defines "
                    "one temporal grid"
                )
        assert reference is not None
        stage_block_hours.append(reference)

    return build_operative_calendar(start, stage_block_hours)


def stage_records(
    calendar: Sequence[OperativeStage],
    num_scenarios: Sequence[int],
) -> list[dict]:
    """Build the ``stages.json`` stage entries for an operative calendar.

    ``num_scenarios[s]`` is stage *s*'s opening branching factor (1 on the
    deterministic trunk, the fan width at the terminal stage). State
    variables follow the lag-blind convention: storage only, no inflow-lag
    state (only the boundary FCF prices lags).
    """
    if len(num_scenarios) != len(calendar):
        raise ValueError(
            f"num_scenarios has {len(num_scenarios)} entries for {len(calendar)} stages"
        )
    records: list[dict] = []
    for stage in calendar:
        names = _block_names(len(stage.block_hours))
        records.append(
            {
                "id": stage.index,
                "start_date": stage.start_date.isoformat(),
                "end_date": stage.end_date.isoformat(),
                "season_id": stage.season_id,
                "blocks": [
                    {"id": b, "name": names[b], "hours": hours}
                    for b, hours in enumerate(stage.block_hours)
                ],
                "num_scenarios": int(num_scenarios[stage.index]),
                "risk_measure": "expectation",
                "state_variables": {"storage": True, "inflow_lags": False},
            }
        )
    return records


def convert_stages(
    calendar: Sequence[OperativeStage],
    *,
    annual_discount_rate: float,
    num_scenarios: Sequence[int],
) -> dict:
    """Build the full ``stages.json`` dict for an operative calendar.

    A linear finite-horizon chain (transition probability 1.0 between
    consecutive stages) under the shared calendar-monthly season map. No
    pre-study stages: the inflow model is order-0 and pre-study inflows
    travel as dated windows.
    """
    stages = stage_records(calendar, num_scenarios)
    transitions = [
        {"source_id": s, "target_id": s + 1, "probability": 1.0}
        for s in range(len(stages) - 1)
    ]
    return {
        "$schema": _STAGES_SCHEMA_URL,
        "season_definitions": monthly_season_definitions(),
        "policy_graph": {
            "type": "finite_horizon",
            "annual_discount_rate": annual_discount_rate,
            "transitions": transitions,
        },
        "stages": stages,
    }
