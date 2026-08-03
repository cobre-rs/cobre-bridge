"""Grades the per-block ``RQ`` minimum-outflow bounds against their own
registers.

For an ``RQ``-derived plant, ``convert_hydro_bounds`` emits a stage-level
base row (``block_id = None``, the hours-weighted value — unchanged from the
earlier interim fold) plus sparse per-block override rows wherever a stage's
per-block percentages actually differ. Three questions matter and they are
different:

1. **Does the reconstructed per-block bound match the registers?** Take the
   emitted table's effective bound for every ``RQ``-derived (hydro, stage,
   block) cell (the override row if one exists, else the base row) and
   compare it against ``pct[ree][block] / 100 * base`` computed directly
   from the ``RQ`` and registry (post ``AC VAZMIN``) records — independently
   of the converter's own hours-weighted fold. Graded to a ``< 1e-6`` m3/s
   tolerance (criterion 1).
2. **Is the fold preserved as the base row?** The base row equals the
   hours-weighted mean of that same per-block list, so any stage-level
   consumer still sees the pre-change number (criterion 2).
3. **Is the sparse pattern exactly right?** A block-uniform stage gets no
   override rows; a non-uniform stage gets exactly one override row per
   declared block, including a ``0.0``-valued one (criterion 3).

Measured on ``decomp-jul-26-rv3`` (2026-08-02): 86 ``RQ``-derived plants, of
which 44 clear the (post ``AC VAZMIN``) positivity gate in at least one
stage; 132 (plant, stage) base rows; all 12 ``RQ`` records are ``(100, 100,
0)`` — the light block is *always* fully relaxed on this deck, so every one
of those 132 stages is non-uniform, and the 396 resulting override rows
account for the entire spread the old ``worst_spread`` warning used to
report as "typically 100 %"; 83 plants skipped as ``QDEF``-windowed; zero
plants are ``UH``-declared. The block-uniform half of criterion 3 and the
``UH``-declared half of criterion 4 therefore cannot be exercised by this
deck's own data and are pinned by a small synthetic fixture instead
(``TestSyntheticUniformAndUhDeclared`` below).
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.bounds import convert_hydro_bounds
from cobre_bridge.decomp.hydro import read_hidr
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import (
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


def _qdef_codes(dadger) -> set[int]:
    cq = dadger.cq(df=True)
    if cq is None or cq.empty or "tipo" not in cq.columns:
        return set()
    return {
        int(c) for c in cq[cq["tipo"].astype(str).str.strip() == "QDEF"]["codigo_usina"]
    }


def _rq_derived_codes(dadger, id_map: DecompIdMap) -> dict[int, int]:
    """``{code: ree}`` for every operated hydro this deck governs through
    ``RQ`` (not ``UH``-declared, not ``QDEF``-windowed, its REE has an ``RQ``
    record) — matching :func:`convert_hydro_bounds`'s own classification,
    parsed independently here.
    """
    ree_by_code = _ree_by_code(dadger)
    uh_declared = _uh_declared_codes(dadger)
    qdef = _qdef_codes(dadger)
    pct_blocks = _rq_pct_blocks(dadger)
    return {
        code: ree_by_code[code]
        for code in id_map.hydro_codes
        if code in ree_by_code
        and code not in uh_declared
        and code not in qdef
        and ree_by_code[code] in pct_blocks
    }


@_needs_deck
class TestRqPerBlockBounds:
    """Criteria 1-3: per-block exactness, fold preservation, sparse pattern."""

    def test_effective_per_block_matches_registers_and_fold(self) -> None:
        dadger, hidr, id_map, calendar = _load()
        table = convert_hydro_bounds(dadger, hidr, id_map, calendar).to_pandas()

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
            rows = table[table["hydro_id"] == hydro_id]

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

                stage_rows = rows[rows["stage_id"] == stage.index]
                if expected_stage <= 0.0:
                    # Stage-level positivity gate: a zero (or AC-VAZMIN-zeroed)
                    # base emits no row at all, base or override.
                    assert stage_rows.empty
                    continue

                base_rows = stage_rows[stage_rows["block_id"].isna()]
                assert len(base_rows) == 1
                base_value = float(base_rows.iloc[0]["min_outflow_m3s"])
                # Criterion 2: the base row is the pre-change fold, exactly.
                assert base_value == pytest.approx(expected_stage, abs=_TOL)

                override_rows = stage_rows[stage_rows["block_id"].notna()]
                override_by_block = {
                    int(r["block_id"]): float(r["min_outflow_m3s"])
                    for _, r in override_rows.iterrows()
                }
                uniform = all(v == expected_blocks[0] for v in expected_blocks)
                if uniform:
                    # Criterion 3: a uniform stage gets no override rows.
                    assert override_by_block == {}
                    uniform_stages += 1
                else:
                    # Criterion 3: a non-uniform stage gets exactly one
                    # override row per block, including a 0.0-valued one.
                    assert set(override_by_block) == set(range(n_blocks))
                    nonuniform_stages += 1

                # Criterion 1: the reconstructed effective value (override
                # row if present, else the base row) for every block.
                for b, expected_value in enumerate(expected_blocks):
                    effective = override_by_block.get(b, base_value)
                    assert effective == pytest.approx(expected_value, abs=_TOL)
                    checked_cells += 1

        # Pinned against the deck (see module docstring): every RQ record on
        # this deck is (100, 100, 0), so every positive-base stage is
        # non-uniform — the uniform branch never fires here by construction,
        # not because of a bug (covered synthetically below instead).
        assert uniform_stages == 0
        assert nonuniform_stages == 132
        assert checked_cells == 396


@_needs_deck
class TestQdefUnaffected:
    """Criterion 4 (``QDEF`` half): RQ per-block emission does not perturb it."""

    def test_qdef_plants_contribute_no_rows(self, caplog) -> None:
        dadger, hidr, id_map, calendar = _load()
        qdef = _qdef_codes(dadger) & set(id_map.hydro_codes)
        assert qdef, "deck must declare at least one QDEF-windowed plant"

        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.bounds"):
            table = convert_hydro_bounds(dadger, hidr, id_map, calendar).to_pandas()

        qdef_ids = {id_map.hydro_id(code) for code in qdef}
        assert not set(table["hydro_id"]) & qdef_ids
        assert f"{len(qdef)} plant(s) with explicit QDEF" in caplog.text


@_needs_deck
class TestEmissionSelfChecks:
    """Criterion 5: the pipeline's post-emission self-checks raise nothing."""

    def test_no_error_severity_finding(self) -> None:
        dadger, hidr, id_map, calendar = _load()
        table = convert_hydro_bounds(dadger, hidr, id_map, calendar)

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

        # Rule 43 stays "not applicable": hydro_bounds carries no
        # max_turbined_m3s / max_generation_mw column for this deck family.
        not_applicable = [
            d for d in diagnostics if d.code == "hydro-bounds-raising-not-applicable"
        ]
        assert len(not_applicable) == 1


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
    ``UH``-declared, plant 4 is ``QDEF``-windowed.
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
        stub = _StubDadger(uh=uh, rq=rq, cq=cq)
        stub.ac = lambda codigo_usina=None, modificacao=None, df=False: pd.DataFrame()  # type: ignore[attr-defined]
        return stub

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

    def test_uniform_stage_emits_base_row_only(self) -> None:
        calendar = self._calendar()
        table = convert_hydro_bounds(
            self._dadger(), self._hidr(), self._ID_MAP, calendar
        ).to_pandas()

        plant1 = table[table["hydro_id"] == self._ID_MAP.hydro_id(1)]
        # Criterion 3 (uniform half): one base row per stage, no overrides.
        assert len(plant1) == len(calendar)
        assert plant1["block_id"].isna().all()
        assert set(plant1["min_outflow_m3s"]) == {20.0}  # 50 % of 40 m3/s

    def test_uh_declared_rows_carry_no_block_id(self) -> None:
        calendar = self._calendar()
        table = convert_hydro_bounds(
            self._dadger(), self._hidr(), self._ID_MAP, calendar
        ).to_pandas()

        # Criterion 4 (UH half): a UH-declared plant's rows are all base
        # rows (block_id = None), the constant declared value, unperturbed
        # by RQ per-block emission for the other plants in the same deck.
        plant3 = table[table["hydro_id"] == self._ID_MAP.hydro_id(3)]
        assert len(plant3) == len(calendar)
        assert plant3["block_id"].isna().all()
        assert set(plant3["min_outflow_m3s"]) == {25.0}

    def test_qdef_plant_contributes_no_rows_in_mixed_deck(self, caplog) -> None:
        calendar = self._calendar()
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.bounds"):
            table = convert_hydro_bounds(
                self._dadger(), self._hidr(), self._ID_MAP, calendar
            ).to_pandas()

        # Criterion 4 (QDEF half), same mixed deck as the two tests above.
        assert self._ID_MAP.hydro_id(4) not in set(table["hydro_id"])
        assert "1 plant(s) with explicit QDEF" in caplog.text

    def test_nonuniform_stage_still_gets_exact_override_rows(self) -> None:
        # RQ per-block emission for the ordinary non-uniform plant (2) is
        # unperturbed by the UH/QDEF plants sharing the same deck.
        calendar = self._calendar()
        table = convert_hydro_bounds(
            self._dadger(), self._hidr(), self._ID_MAP, calendar
        ).to_pandas()

        plant2 = table[table["hydro_id"] == self._ID_MAP.hydro_id(2)]
        base_rows = plant2[plant2["block_id"].isna()]
        override_rows = plant2[plant2["block_id"].notna()]
        assert len(base_rows) == len(calendar)
        assert len(override_rows) == len(calendar) * 3
        for stage in calendar:
            stage_overrides = override_rows[override_rows["stage_id"] == stage.index]
            values = dict(
                zip(stage_overrides["block_id"], stage_overrides["min_outflow_m3s"])
            )
            assert values == {0: 40.0, 1: 40.0, 2: 0.0}
