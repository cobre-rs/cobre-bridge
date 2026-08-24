"""Tests the ``RQ``-derived and ``UH``-declared minimum-outflow bound
classification in ``convert_hydro_bounds``, via a synthetic mixed deck
(``TestSyntheticUniformAndUhDeclared``).

For an ``RQ``-derived plant, ``convert_hydro_bounds`` contributes, per stage,
**either** one stage-level (``block_id = None``) contribution — the
hours-weighted value — when the stage's per-block percentages are all equal,
**or** one contribution per block (``block_id = 0..n-1``, no base) when they
are not; never both (epic-07, ticket-023 — the accumulator does not
replicate cobre's replace-not-merge column semantics, so a base contribution
left alongside per-block ones would be double-counted into every block's
intersection). A ``UH``-declared plant's own value takes priority over
``RQ`` and is always base-only. A ``QDEF``-windowed plant still contributes
its ``RQ``/``UH`` value regardless of the window (ticket-023, AC7) — that
window's own contribution comes separately from
``single_term_bounds.single_term_bound_contributions`` (RHQ), and the
accumulator intersects the two rather than one replacing the other.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from cobre_bridge.decomp.bounds import convert_hydro_bounds
from cobre_bridge.decomp.cadastro import EffectiveCadastro
from cobre_bridge.decomp.case import DecompCase
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


class TestSyntheticUniformAndUhDeclared:
    """Branches the real deck's own data never exercises (see module
    docstring): a block-uniform ``RQ`` stage (criterion 3) and a
    ``UH``-declared plant (criterion 4), pinned deterministically here.

    All four plant classes live in one stub deck, mirroring criterion 4's
    "in the same deck" framing: plant 1 is ``RQ``-derived and block-uniform,
    plant 2 is ``RQ``-derived and non-uniform (for contrast), plant 3 is
    ``UH``-declared, plant 4 is ``RQ``-derived and ``QDEF``-windowed (still
    contributes, per AC7).
    """

    _ID_MAP = DecompIdMap(
        bus_codes=(1,),
        bus_names=("SE",),
        hydro_codes=(1, 2, 3, 4),
    )

    def _calendar(self):
        hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
        return build_operative_calendar(date(2026, 7, 18), hours)

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
                # REE 1: block-uniform (plant 1's own REE).
                {"codigo_ree": 1, "vazao_1": 50.0, "vazao_2": 50.0, "vazao_3": 50.0},
                # REE 2: non-uniform, light block fully relaxed.
                {"codigo_ree": 2, "vazao_1": 100.0, "vazao_2": 100.0, "vazao_3": 0.0},
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
        (covered by ``tests/test_decomp_cadastro.py``)."""
        return EffectiveCadastro(
            base=self._hidr(), n_stages=len(calendar), stage_varying=stage_varying or {}
        )

    def test_uniform_stage_emits_base_contribution_only(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        plant1_id = self._ID_MAP.hydro_id(1)
        plant1 = [c for c in contributions if c.entity_id == plant1_id]
        # Criterion 3 (uniform half): one base contribution per stage, no
        # per-block ones.
        assert len(plant1) == len(calendar)
        assert all(c.block_id is None for c in plant1)
        assert {c.lower for c in plant1} == {20.0}  # 50 % of 40 m3/s

    def test_nan_historical_minimum_emits_no_contribution(self) -> None:
        # A NaN historical-minimum (a missing registry value) must be treated
        # like a non-positive one and contribute nothing, not a NaN outflow
        # bound.
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

    def test_uh_declared_rows_carry_no_block_id(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        # Criterion 4 (UH half): a UH-declared plant's contributions are all
        # base (block_id = None), the constant declared value, unperturbed
        # by RQ per-block emission for the other plants in the same deck.
        plant3_id = self._ID_MAP.hydro_id(3)
        plant3 = [c for c in contributions if c.entity_id == plant3_id]
        assert len(plant3) == len(calendar)
        assert all(c.block_id is None for c in plant3)
        assert {c.lower for c in plant3} == {25.0}

    def test_qdef_plant_still_contributes_in_mixed_deck(self, caplog) -> None:
        calendar = self._calendar()
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.bounds"):
            contributions = convert_hydro_bounds(
                self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
            )

        # Criterion 4 (QDEF half, retired per AC7), same mixed deck as the
        # two tests above: plant 4 shares REE 2 with plant 2, so it
        # contributes the identical non-uniform per-block pattern, and no
        # warning fires (the skip-and-warn branch no longer exists).
        plant4_id = self._ID_MAP.hydro_id(4)
        plant2_id = self._ID_MAP.hydro_id(2)
        plant4 = [c for c in contributions if c.entity_id == plant4_id]
        plant2 = [c for c in contributions if c.entity_id == plant2_id]
        assert plant4
        assert {(c.stage_id, c.block_id, c.lower) for c in plant4} == {
            (c.stage_id, c.block_id, c.lower) for c in plant2
        }
        assert "QDEF" not in caplog.text

    def test_nonuniform_stage_still_gets_exact_per_block_contributions(self) -> None:
        # RQ per-block emission for the ordinary non-uniform plant (2) is
        # unperturbed by the UH/QDEF plants sharing the same deck.
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=self._effective(calendar)
        )

        plant2_id = self._ID_MAP.hydro_id(2)
        plant2 = [c for c in contributions if c.entity_id == plant2_id]
        base_contribs = [c for c in plant2 if c.block_id is None]
        override_contribs = [c for c in plant2 if c.block_id is not None]
        # Every stage is non-uniform (100, 100, 0): no base contribution at
        # all, only per-block ones (never both, per the replace-vs-intersect
        # discipline).
        assert base_contribs == []
        assert len(override_contribs) == len(calendar) * 3
        for stage in calendar:
            stage_overrides = [
                c for c in override_contribs if c.stage_id == stage.index
            ]
            values = {c.block_id: c.lower for c in stage_overrides}
            assert values == {0: 40.0, 1: 40.0, 2: 0.0}

    def test_temporal_vazmin_override_gates_only_the_overridden_stage(self) -> None:
        """A temporal ``vazao_minima_historica`` override that zeroes plant
        2's effective minimum only at the final stage: the stage-level
        ``value <= 0.0`` gate drops that stage's contributions while the
        earlier stages (still at the 40.0 base) keep theirs — never a
        plant-level skip, which would incorrectly drop every stage."""
        calendar = self._calendar()
        effective = self._effective(
            calendar,
            stage_varying={(2, "vazao_minima_historica"): (40.0, 40.0, 0.0)},
        )
        contributions = convert_hydro_bounds(
            self._case(calendar), self._ID_MAP, effective=effective
        )

        plant2_id = self._ID_MAP.hydro_id(2)
        plant2 = [c for c in contributions if c.entity_id == plant2_id]
        assert {c.stage_id for c in plant2} == {0, 1}
        assert [c for c in plant2 if c.stage_id == 2] == []
