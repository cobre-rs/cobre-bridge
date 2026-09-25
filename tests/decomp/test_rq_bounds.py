"""Tests the ``RQ``-derived and ``UH``-declared minimum-outflow bound
classification in ``convert_hydro_bounds``, via a synthetic mixed deck
(``TestSyntheticStageIndexedAndUhDeclared``).

The ``RQ`` register carries one percentage per study stage (``vazao_1`` →
stage 0, ``vazao_2`` → stage 1, …). For an ``RQ``-derived plant,
``convert_hydro_bounds`` contributes one stage-level (``block_id = None``)
``outflow`` floor per stage: the stage's percentage times the plant's
effective ``vazao_minima_historica`` for that stage. The last declared
percentage carries forward for any study stage beyond the register's own
columns (seasonal carry-forward). A stage whose resulting floor is
non-positive (or ``NaN``) emits nothing.

A ``UH``-declared plant's own value takes priority over ``RQ`` and is a
constant stage-level value every stage. A ``QDEF``-windowed plant still
contributes its ``RQ``/``UH`` value regardless of the window — that window's
own contribution comes separately from
``single_term_bounds.single_term_bound_contributions`` (RHQ), and the
accumulator intersects the two rather than one replacing the other.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from cobre_bridge.decomp.case import DecompCase
from cobre_bridge.decomp.converters.bounds import convert_hydro_bounds
from cobre_bridge.decomp.converters.cadastro import EffectiveCadastro
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import OperativeStage, build_operative_calendar
from tests.conftest import make_decomp_case


class _StubDadger:
    """Minimal ``Dadger``-shaped stub keying record accessors by dataframe."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):
        if name in self._frames:
            frame = self._frames[name]
            return lambda df=False, **kwargs: frame  # noqa: ARG005
        raise AttributeError(name)


class TestSyntheticStageIndexedAndUhDeclared:
    """Pins the per-stage ``RQ`` reading and ``UH`` priority on one mixed
    deck. The operative calendar has four weekly stages plus one aggregate
    stage (5 stages total), each with 3 blocks.

    Four plant classes share one stub deck: plant 1 is ``RQ``-derived with a
    fully-declared, stage-varying percentage vector; plant 2 is ``RQ``-derived
    with a zero stage (exercises the non-positive gate); plant 3 is
    ``UH``-declared; plant 4 is ``RQ``-derived and ``QDEF``-windowed (still
    contributes). Carry-forward past the last declared column is covered in
    ``test_bound_contributions.py`` (a single-REE frame, so no NaN column
    padding masks it).
    """

    _ID_MAP = DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=(1, 2, 3, 4),
    )

    def _calendar(self) -> list[OperativeStage]:
        # Four 168 h weekly stages (3 blocks each) starting Sat 2026-07-04
        # (first operative month = July), then one aggregate stage that
        # covers the second operative month (Aug 1 -> Sep 1, 31 days =
        # 744 h) and closes at a calendar month boundary. 3 blocks each.
        weekly = [[24.0, 64.0, 80.0]] * 4
        aggregate = [[144.0, 300.0, 300.0]]  # 744 h total
        return build_operative_calendar(date(2026, 7, 4), weekly + aggregate)

    def _dadger(self) -> _StubDadger:
        uh = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "codigo_ree": 1,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": None,
                },
                {
                    "codigo_usina": 2,
                    "codigo_ree": 2,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": None,
                },
                {
                    "codigo_usina": 3,
                    "codigo_ree": 1,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": 25.0,
                },
                {
                    "codigo_usina": 4,
                    "codigo_ree": 2,
                    "volume_inicial": 50.0,
                    "vazao_defluente_minima": None,
                },
            ]
        )
        rq = pd.DataFrame(
            [
                # REE 1 (plant 1): one percentage per stage, all five declared,
                # stage-varying to prove the axis is stage, not block.
                {
                    "codigo_ree": 1,
                    "vazao_1": 100.0,
                    "vazao_2": 80.0,
                    "vazao_3": 60.0,
                    "vazao_4": 40.0,
                    "vazao_5": 20.0,
                },
                # REE 2 (plants 2, 4): fewer columns than stages (carry-forward)
                # and a zero stage (non-positive gate). vazao_2 = 0 zeroes
                # stage 1; vazao_3 = 50 carries forward to stages 3 and 4.
                {
                    "codigo_ree": 2,
                    "vazao_1": 100.0,
                    "vazao_2": 0.0,
                    "vazao_3": 50.0,
                },
            ]
        )
        cq = pd.DataFrame(
            [
                {
                    "codigo_restricao": 1,
                    "codigo_usina": 4,
                    "coeficiente": 1.0,
                    "estagio": 1,
                    "tipo": "QDEF",
                }
            ]
        )
        return _StubDadger(uh=uh, rq=rq, cq=cq)

    def _hidr(self) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                1: {"vazao_minima_historica": 40.0},
                2: {"vazao_minima_historica": 40.0},
                3: {"vazao_minima_historica": 40.0},
                4: {"vazao_minima_historica": 40.0},
            }
        ).T
        df.index.name = "codigo_usina"
        return df

    def _case(self, calendar: Sequence[OperativeStage]) -> DecompCase:
        return make_decomp_case(
            Path("unused"), dadger=self._dadger(), calendar=calendar
        )

    def _effective(
        self,
        calendar: Sequence[OperativeStage],
        stage_varying: dict[tuple[int, str], tuple[float, ...]] | None = None,
    ) -> EffectiveCadastro:
        """Build an ``EffectiveCadastro`` directly over ``self._hidr()``,
        bypassing ``build_effective_cadastro``/``AC`` ingestion — this class
        pins the ``RQ``/``UH`` classification, not the resolver itself
        (covered by ``tests/decomp/test_cadastro.py``)."""
        return EffectiveCadastro(
            base=self._hidr(), n_stages=len(calendar), stage_varying=stage_varying or {}
        )

    def _by_stage(self, contributions, entity_id):
        rows = [c for c in contributions if c.entity_id == entity_id]
        # Every RQ/UH outflow floor is stage-level.
        assert all(c.block_id is None and c.axis == "outflow" for c in rows)
        return {c.stage_id: c.lower for c in rows}

    def test_rq_percentages_are_applied_per_stage(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        # Plant 1, REE 1: 100/80/60/40/20 % of 40 m3/s, one stage-level row
        # per stage.
        plant1_id = self._ID_MAP.hydro_id(1)
        assert self._by_stage(contributions, plant1_id) == {
            0: 40.0,
            1: 32.0,
            2: 24.0,
            3: 16.0,
            4: 8.0,
        }

    def test_nonpositive_and_blank_stages_emit_nothing(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        # Plant 2, REE 2: declared percentages 100, 0, 50; the frame is
        # padded to REE 1's five columns with NaN (read as 0.0) at vazao_4/5.
        # So per stage: 100, 0, 50, 0, 0 %. Only the strictly-positive stages
        # emit a row (stage 0 -> 40, stage 2 -> 20); the 0 % stages (1, 3, 4)
        # are gated out. (Carry-forward past the widest declared column is
        # covered separately in test_bound_contributions.py.)
        plant2_id = self._ID_MAP.hydro_id(2)
        by_stage = self._by_stage(contributions, plant2_id)
        assert by_stage == {0: 40.0, 2: 20.0}
        assert 1 not in by_stage and 3 not in by_stage and 4 not in by_stage

    def test_uh_declared_is_constant_stage_level(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        # Plant 3 is UH-declared (25.0): a constant stage-level value every
        # stage, its own value taking priority over REE 1's RQ percentages.
        plant3_id = self._ID_MAP.hydro_id(3)
        by_stage = self._by_stage(contributions, plant3_id)
        assert by_stage == {stage.index: 25.0 for stage in calendar}

    def test_qdef_plant_still_contributes_its_rq_value(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        # Plant 4 shares REE 2 with plant 2, so despite its QDEF window it
        # contributes the identical per-stage RQ floors (the window's own
        # bound comes separately from single_term_bounds).
        plant4_id = self._ID_MAP.hydro_id(4)
        plant2_id = self._ID_MAP.hydro_id(2)
        assert self._by_stage(contributions, plant4_id) == self._by_stage(
            contributions, plant2_id
        )

    def test_temporal_vazmin_override_gates_only_the_overridden_stage(self) -> None:
        """A temporal ``vazao_minima_historica`` override that zeroes plant
        1's effective minimum only at the final stage: the per-stage
        ``value <= 0.0`` gate drops that stage's contribution while the
        earlier stages (still at the 40.0 base) keep theirs — never a
        plant-level skip, which would incorrectly drop every stage."""
        calendar = self._calendar()
        effective = self._effective(
            calendar,
            stage_varying={
                (1, "vazao_minima_historica"): (40.0, 40.0, 40.0, 40.0, 0.0)
            },
        )
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=effective
        )

        plant1_id = self._ID_MAP.hydro_id(1)
        by_stage = self._by_stage(contributions, plant1_id)
        # Final stage's floor is 20% * 0.0 = 0.0 -> gated out; the rest hold.
        assert set(by_stage) == {0, 1, 2, 3}
        assert 4 not in by_stage

    def test_nan_historical_minimum_emits_no_contribution(self) -> None:
        # A NaN historical-minimum (a missing registry value) at every stage
        # must be treated like a non-positive one and contribute nothing, not
        # a NaN outflow bound.
        calendar = self._calendar()
        effective = self._effective(
            calendar,
            stage_varying={
                (1, "vazao_minima_historica"): (float("nan"),) * len(calendar)
            },
        )
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=effective
        )
        plant1_id = self._ID_MAP.hydro_id(1)
        assert [c for c in contributions if c.entity_id == plant1_id] == []
