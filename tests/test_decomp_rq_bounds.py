"""Grades the per-block ``RQ`` minimum-outflow bounds against their own
registers.

For an ``RQ``-derived plant, ``convert_hydro_bounds`` contributes, per stage,
**either** one stage-level (``block_id = None``) contribution — the
hours-weighted value, unchanged from the earlier interim fold — when the
stage's per-block percentages are all equal, **or** one contribution per
block (``block_id = 0..n-1``, no base) when they are not; never both (epic-07,
ticket-023 — the accumulator does not replicate cobre's replace-not-merge
column semantics, so a base contribution left alongside per-block ones would
be double-counted into every block's intersection). Three questions matter
and they are different:

1. **Does the reconstructed per-block bound match the registers?** Take the
   *effective* contribution for every ``RQ``-derived (hydro, stage, block)
   cell (the per-block contribution if one exists, else the base
   contribution) and compare it against ``pct[ree][block] / 100 * base``
   computed directly from the ``RQ`` and registry (post ``AC VAZMIN``)
   records — independently of the converter's own hours-weighted fold.
   Graded to a ``< 1e-6`` m3/s tolerance (criterion 1).
2. **Is the fold preserved on the base contribution?** Where a stage is
   block-uniform and does contribute one, it equals the hours-weighted mean
   of that same per-block list, so any stage-level consumer still sees the
   pre-change number (criterion 2).
3. **Is the sparse pattern exactly right?** A block-uniform stage
   contributes exactly one base contribution and no per-block ones; a
   non-uniform stage contributes exactly one per-block contribution per
   declared block (including a ``0.0``-valued one) and no base (criterion 3).

Measured on ``decomp-jul-26-rv3`` (2026-08-09, post ticket-023): a
``QDEF``-windowed plant is no longer excluded from the RQ/UH classification
here — that window's own contribution now comes from
``single_term_bounds.single_term_bound_contributions`` (RHQ), and the
accumulator intersects the two rather than one replacing the other (AC7) —
so this module's own independent reproduction (``_rq_derived_codes``) no
longer excludes them either: 169 RQ-derived plants (up from 86 pre-ticket-023,
since all 83 ``QDEF``-windowed plants also carry a valid RQ record via their
REE), of which 100 clear the (post ``AC VAZMIN``) positivity gate in at least
one stage — 300 (plant, stage) cells, all non-uniform (every ``RQ`` record on
this deck is ``(100, 100, 0)``, so the light block is *always* fully relaxed),
900 per-block contributions, zero base contributions. The block-uniform half
of criterion 3 and the ``UH``-declared half of criterion 4 therefore cannot
be exercised by this deck's own data and are pinned by a small synthetic
fixture instead (``TestSyntheticUniformAndUhDeclared`` below).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.bounds import convert_hydro_bounds
from cobre_bridge.decomp.bounds_accumulator import (
    BoundContribution,
    build_bound_tables,
    resolve,
)
from cobre_bridge.decomp.cadastro import EffectiveCadastro, build_effective_cadastro
from cobre_bridge.decomp.hydro import read_hidr
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import (
    OperativeStage,
    build_operative_calendar,
    operative_calendar_from_dadger,
)
from cobre_bridge.diagnostics import Severity
from cobre_bridge.emission_checks import (
    BoundFamily,
    check_bound_block_id_range,
    check_bound_row_uniqueness,
    check_hydro_bounds_no_raising,
)

_DECK = Path("example/decomp-jul-26-rv3")
_needs_deck = pytest.mark.skipif(
    not (_DECK / "saidas" / "dec_oper_usit.csv").exists(),
    reason="reference deck outputs not present",
)

#: The spec's own tolerance for the per-block reconstruction (criterion 1/2).
_TOL = 1e-6


def _load():
    """Parse the deck once: dadger, registry, id map, and calendar."""
    from idecomp.decomp import Dadger

    dadger = Dadger.read(str(_DECK / "dadger.rv3"))
    hidr = read_hidr(_DECK / "hidr.dat")
    id_map = DecompIdMap.from_dadger(dadger)
    calendar = operative_calendar_from_dadger(dadger)
    return dadger, hidr, id_map, calendar


def _rq_pct_blocks(dadger) -> dict[int, list[float]]:
    """``{ree: [pct_block_1, ...]}`` parsed directly from the ``RQ`` records."""
    rq = dadger.rq(df=True)
    pct_blocks: dict[int, list[float]] = {}
    for _, row in rq.iterrows():
        values: list[float] = []
        k = 1
        while f"vazao_{k}" in rq.columns:
            value = row[f"vazao_{k}"]
            values.append(0.0 if pd.isna(value) else float(value))
            k += 1
        pct_blocks[int(row["codigo_ree"])] = values
    return pct_blocks


def _vazmin(dadger, hidr, id_map: DecompIdMap) -> dict[int, float]:
    """Registry historical minimum flow, post ``AC VAZMIN`` overrides — the
    same source :func:`convert_hydro_bounds` reads, parsed independently
    here."""
    from idecomp.decomp.modelos.dadger import ACVAZMIN

    vazmin: dict[int, float] = {}
    for code in id_map.hydro_codes:
        if code in hidr.index:
            base = hidr.loc[code, "vazao_minima_historica"]
            vazmin[code] = 0.0 if pd.isna(base) else float(base)
    overrides = dadger.ac(codigo_usina=None, modificacao=ACVAZMIN, df=True)
    if isinstance(overrides, pd.DataFrame) and not overrides.empty:
        for _, row in overrides.iterrows():
            code = int(row["codigo_usina"])
            if code in vazmin:
                vazmin[code] = float(row["vazao"])
    return vazmin


def _ree_by_code(dadger) -> dict[int, int]:
    uh = dadger.uh(df=True)
    operated = uh[uh["volume_inicial"].notna()]
    return {
        int(row["codigo_usina"]): int(row["codigo_ree"])
        for _, row in operated.iterrows()
    }


def _uh_declared_codes(dadger) -> set[int]:
    uh = dadger.uh(df=True)
    operated = uh[uh["volume_inicial"].notna()]
    declared = set()
    for _, row in operated.iterrows():
        value = row.get("vazao_defluente_minima")
        if value is not None and not pd.isna(value):
            declared.add(int(row["codigo_usina"]))
    return declared


def _rq_derived_codes(dadger, id_map: DecompIdMap) -> dict[int, int]:
    """``{code: ree}`` for every operated hydro this deck governs through
    ``RQ`` (not ``UH``-declared, its REE has an ``RQ`` record) — matching
    :func:`convert_hydro_bounds`'s own classification, parsed independently
    here. A ``QDEF``-windowed plant is *not* excluded (ticket-023, AC7):
    ``convert_hydro_bounds`` contributes its RQ/UH value regardless, and the
    accumulator intersects it with that window's own contribution.
    """
    ree_by_code = _ree_by_code(dadger)
    uh_declared = _uh_declared_codes(dadger)
    pct_blocks = _rq_pct_blocks(dadger)
    return {
        code: ree_by_code[code]
        for code in id_map.hydro_codes
        if code in ree_by_code
        and code not in uh_declared
        and ree_by_code[code] in pct_blocks
    }


def _by_hydro_stage(
    contributions: Sequence[BoundContribution], hydro_id: int, stage_id: int
) -> list[BoundContribution]:
    return [
        c for c in contributions if c.entity_id == hydro_id and c.stage_id == stage_id
    ]


@_needs_deck
class TestRqPerBlockBounds:
    """Criteria 1-3: per-block exactness, fold preservation, sparse pattern."""

    def test_effective_per_block_matches_registers_and_fold(self) -> None:
        dadger, hidr, id_map, calendar = _load()
        effective, _ = build_effective_cadastro(dadger, hidr, calendar)
        contributions = convert_hydro_bounds(dadger, id_map, calendar, effective)

        pct_blocks = _rq_pct_blocks(dadger)
        vazmin = _vazmin(dadger, hidr, id_map)
        rq_codes = _rq_derived_codes(dadger, id_map)
        assert rq_codes, "deck must declare at least one RQ-derived plant"

        checked_cells = 0
        uniform_stages = 0
        nonuniform_stages = 0

        for code, ree in rq_codes.items():
            hydro_id = id_map.hydro_id(code)
            base = vazmin.get(code, 0.0)
            values = pct_blocks[ree]

            for stage in calendar:
                n_blocks = len(stage.block_hours)
                expected_blocks = [v / 100.0 * base for v in values[:n_blocks]]
                expected_stage = (
                    sum(
                        v * h
                        for v, h in zip(expected_blocks, stage.block_hours, strict=True)
                    )
                    / stage.total_hours
                )

                stage_contribs = _by_hydro_stage(contributions, hydro_id, stage.index)
                if expected_stage <= 0.0:
                    # Stage-level positivity gate: a zero (or AC-VAZMIN-zeroed)
                    # base contributes nothing at all, base or per-block.
                    assert stage_contribs == []
                    continue

                base_contribs = [c for c in stage_contribs if c.block_id is None]
                override_contribs = [
                    c for c in stage_contribs if c.block_id is not None
                ]
                override_by_block = {c.block_id: c.lower for c in override_contribs}
                uniform = all(v == expected_blocks[0] for v in expected_blocks)

                if uniform:
                    # Criterion 3: a uniform stage contributes exactly one
                    # base contribution and no per-block ones.
                    assert len(base_contribs) == 1
                    assert override_contribs == []
                    base_value = base_contribs[0].lower
                    # Criterion 2: the base contribution is the pre-change
                    # fold, exactly.
                    assert base_value == pytest.approx(expected_stage, abs=_TOL)
                    uniform_stages += 1
                else:
                    # Criterion 3: a non-uniform stage contributes exactly
                    # one per-block contribution per block, including a
                    # 0.0-valued one, and no base.
                    assert base_contribs == []
                    assert set(override_by_block) == set(range(n_blocks))
                    nonuniform_stages += 1

                # Criterion 1: the reconstructed effective value (per-block
                # contribution if present, else the base one) for every
                # block.
                for b, expected_value in enumerate(expected_blocks):
                    effective_value = override_by_block.get(
                        b, base_contribs[0].lower if base_contribs else None
                    )
                    assert effective_value == pytest.approx(expected_value, abs=_TOL)
                    checked_cells += 1

        # Pinned against the deck (see module docstring): every RQ record on
        # this deck is (100, 100, 0), so every positive-base stage is
        # non-uniform — the uniform branch never fires here by construction,
        # not because of a bug (covered synthetically below instead).
        assert uniform_stages == 0
        assert nonuniform_stages == 300
        assert checked_cells == 900


@_needs_deck
class TestQdefStillContributesRqDefault:
    """Criterion 4 (``QDEF`` half, retired per ticket-023 AC7): RQ per-block
    emission is unaffected by a plant's own ``QDEF`` flow window — that
    window no longer causes ``convert_hydro_bounds`` to skip the plant."""

    def test_qdef_plants_still_contribute_via_rq(self, caplog) -> None:
        dadger, hidr, id_map, calendar = _load()
        cq = dadger.cq(df=True)
        qdef = (
            {
                int(c)
                for c in cq[cq["tipo"].astype(str).str.strip() == "QDEF"][
                    "codigo_usina"
                ]
            }
            & set(id_map.hydro_codes)
            & set(_rq_derived_codes(dadger, id_map))
        )
        assert qdef, "deck must declare at least one QDEF-windowed RQ-derived plant"

        effective, _ = build_effective_cadastro(dadger, hidr, calendar)
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.bounds"):
            contributions = convert_hydro_bounds(dadger, id_map, calendar, effective)

        qdef_ids = {id_map.hydro_id(code) for code in qdef}
        contributed_ids = {c.entity_id for c in contributions}
        assert qdef_ids & contributed_ids
        # AC7: the skip-and-warn branch is retired; no such warning fires.
        assert "QDEF" not in caplog.text


@_needs_deck
class TestEmissionSelfChecks:
    """Criterion 5: the pipeline's post-emission self-checks raise nothing."""

    def test_no_error_severity_finding(self) -> None:
        dadger, hidr, id_map, calendar = _load()
        effective, _ = build_effective_cadastro(dadger, hidr, calendar)
        contributions = convert_hydro_bounds(dadger, id_map, calendar, effective)
        block_counts = {stage.index: len(stage.block_hours) for stage in calendar}
        table = build_bound_tables(resolve(contributions, block_counts)).hydro

        stages_doc = {
            "stages": [
                {"id": stage.index, "blocks": [{}] * len(stage.block_hours)}
                for stage in calendar
            ]
        }
        families = [BoundFamily("Hydro", "hydro_id", table)]

        with dx.collect() as diagnostics:
            check_hydro_bounds_no_raising({"hydros": []}, table)
            check_bound_row_uniqueness(families)
            check_bound_block_id_range(stages_doc, families)

        errors = [d for d in diagnostics if d.severity is Severity.ERROR]
        assert not errors, [d.summary for d in errors]


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
            self._dadger(), self._ID_MAP, calendar, self._effective(calendar)
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
            self._dadger(), self._ID_MAP, calendar, effective
        )
        plant1_id = self._ID_MAP.hydro_id(1)
        assert [c for c in contributions if c.entity_id == plant1_id] == []

    def test_uh_declared_rows_carry_no_block_id(self) -> None:
        calendar = self._calendar()
        contributions = convert_hydro_bounds(
            self._dadger(), self._ID_MAP, calendar, self._effective(calendar)
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
                self._dadger(), self._ID_MAP, calendar, self._effective(calendar)
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
            self._dadger(), self._ID_MAP, calendar, self._effective(calendar)
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
            self._dadger(), self._ID_MAP, calendar, effective
        )

        plant2_id = self._ID_MAP.hydro_id(2)
        plant2 = [c for c in contributions if c.entity_id == plant2_id]
        assert {c.stage_id for c in plant2} == {0, 1}
        assert [c for c in plant2 if c.stage_id == 2] == []
