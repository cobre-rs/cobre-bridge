"""Grades the per-block thermal bounds against the reference run.

The source model declares thermal bounds per block. ``convert_thermal_bounds``
contributes, per ``(thermal, stage)``, either a stage-level base contribution
(hours-weighted, unchanged from the earlier interim fold) when the stage's
blocks are uniform, or sparse per-block contributions (no base) when they are
not — never both (epic-07, ticket-023's replace-vs-intersect discipline), so
the resolution that fold used to throw away is now reproduced exactly. Every
test below resolves those contributions through the same
``bounds_accumulator.resolve``/``build_bound_tables`` + cost-rejoin pass
``pipeline.convert_decomp_case`` uses, so it grades exactly what lands in
``thermal_bounds.parquet``. Two questions matter and they are different:

1. **Does the reconstructed per-block bound match reality?** Take the emitted
   table's effective bound for every ``(plant, stage, block)`` cell (the
   override row if one exists, else the base row) and compare it against the
   reference's own per-block report. This is a hard invariant against
   independently-sourced data and is graded to a `< 1e-6` MW tolerance (the
   two sources are parsed from different files, so exact floating-point
   equality is not expected even when the underlying physical value is
   identical).
2. **Does the sparse override mechanism itself lose anything?** For every
   block whose stage the converter judged non-uniform, the override row's
   `min`/`max` is a direct, untransformed copy of the source `inflex`/`disp`
   entry the converter read — comparing the two, straight off the emitted
   table, is a bit-exact `== 0.0` invariant, not a tolerance.

Measured on ``decomp-jul-26-rv3`` (2026-08-02): 291 (plant, stage) rows, 38 of
them block-varying (5 in the minimum, 35 in the maximum), 114 override rows
emitted; every reconstructed per-block bound matches the reference to well
under `1e-6` MW (worst observed difference ~6e-14 MW, floating-point noise
from the reference's own duplicate-row averaging); the override mechanism's
own must-run misallocation and capacity over-allowance are both exactly zero.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest

from cobre_bridge.comparators.decomp_readers import read_dec_oper_usit
from cobre_bridge.decomp.bounds_accumulator import build_bound_tables, resolve
from cobre_bridge.decomp.pipeline import _rejoin_thermal_cost
from cobre_bridge.decomp.thermal import convert_thermal_bounds

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_DECK = Path("example/decomp-jul-26-rv3")
_needs_deck = pytest.mark.skipif(
    not (_DECK / "saidas" / "dec_oper_usit.csv").exists(),
    reason="reference deck outputs not present",
)


def _resolved_bounds_frame(
    dadger: Dadger, id_map: DecompIdMap, calendar: Sequence[OperativeStage]
) -> pd.DataFrame:
    """The exact ``thermal_bounds.parquet`` shape ``pipeline.py`` writes:
    ``convert_thermal_bounds``'s generation contributions resolved through
    the accumulator, with ``cost_per_mwh`` rejoined from its side-table."""
    bounds = convert_thermal_bounds(dadger, id_map, calendar)
    block_counts = {stage.index: len(stage.block_hours) for stage in calendar}
    resolved = build_bound_tables(resolve(bounds.generation, block_counts)).thermal
    return _rejoin_thermal_cost(resolved, bounds.cost).to_pandas()


def _reference_blocks() -> pl.DataFrame:
    """Per (plant, stage, block) effective bounds as the reference reports them."""
    usit = read_dec_oper_usit(_DECK / "saidas")
    return (
        usit.filter(pl.col("patamar").is_not_null())
        .group_by(["codigo_usina", "estagio", "patamar"])
        .agg(
            pl.col("duracao").first(),
            pl.col("geracao_minima_MW").mean().alias("min"),
            pl.col("geracao_maxima_MW").mean().alias("max"),
        )
    )


def _reference_fold(blocks: pl.DataFrame) -> pl.DataFrame:
    """Hours-weighted stage fold of the reference's per-block bounds."""
    return (
        blocks.with_columns(
            (pl.col("min") * pl.col("duracao")).alias("min_hours"),
            (pl.col("max") * pl.col("duracao")).alias("max_hours"),
        )
        .group_by(["codigo_usina", "estagio"])
        .agg(
            (pl.col("min_hours").sum() / pl.col("duracao").sum()).alias("min_folded"),
            (pl.col("max_hours").sum() / pl.col("duracao").sum()).alias("max_folded"),
        )
    )


@_needs_deck
class TestThermalFold:
    def test_converted_bounds_equal_the_reference_fold(self) -> None:
        """The emitted base rows are still the reference's own hours fold.

        Anything else means the converter and the reference disagree about
        the stage a block belongs to or the weight it carries. Scoped to the
        base (``block_id = None``) rows that actually carry a generation
        bound — a block-varying stage's own ``block_id = None`` row (if any)
        carries only ``cost_per_mwh`` (see the module docstring's rejoin
        note) and has no stage-level fold to compare here.
        """
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.id_map import DecompIdMap
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        table = _resolved_bounds_frame(dadger, id_map, calendar)
        ours = table[table["block_id"].isna() & table["max_generation_mw"].notna()]

        reference = _reference_fold(_reference_blocks()).to_pandas()
        reference["thermal_id"] = reference["codigo_usina"].map(
            lambda c: (
                id_map.thermal_id(int(c)) if int(c) in id_map.thermal_codes else None
            )
        )
        reference["stage_id"] = reference["estagio"] - 1
        reference = reference.dropna(subset=["thermal_id"])

        merged = ours.merge(reference, on=["thermal_id", "stage_id"], how="inner")
        assert len(merged) > 250, f"only {len(merged)} rows compared"
        for ours_col, ref_col in (
            ("min_generation_mw", "min_folded"),
            ("max_generation_mw", "max_folded"),
        ):
            worst = (merged[ours_col] - merged[ref_col]).abs().max()
            assert worst < 1e-6, f"{ours_col}: worst |Δ| {worst} MW"

    def test_effective_per_block_bounds_match_the_reference(self) -> None:
        """The emitted table's *effective* per-block bound — the override row
        if one exists for a ``(thermal, stage, block)``, else the base row —
        matches the reference's own per-block report to `< 1e-6` MW.

        This is the acceptance test for the sparse per-block convention: it
        reconstructs exactly what cobre reads for every block, not just the
        stage-level summary.
        """
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.id_map import DecompIdMap
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        table = _resolved_bounds_frame(dadger, id_map, calendar)

        base = table[table["block_id"].isna()][
            ["thermal_id", "stage_id", "min_generation_mw", "max_generation_mw"]
        ]
        overrides = table[table["block_id"].notna()][
            [
                "thermal_id",
                "stage_id",
                "block_id",
                "min_generation_mw",
                "max_generation_mw",
            ]
        ].copy()
        overrides["block_id"] = overrides["block_id"].astype(int)

        blocks = _reference_blocks().to_pandas()
        blocks["thermal_id"] = blocks["codigo_usina"].map(
            lambda c: (
                id_map.thermal_id(int(c)) if int(c) in id_map.thermal_codes else None
            )
        )
        blocks["stage_id"] = blocks["estagio"] - 1
        blocks["block_id"] = (blocks["patamar"] - 1).astype(int)
        blocks = blocks.dropna(subset=["thermal_id"])
        blocks["thermal_id"] = blocks["thermal_id"].astype(int)

        merged = blocks.merge(base, on=["thermal_id", "stage_id"], how="left")
        merged = merged.merge(
            overrides,
            on=["thermal_id", "stage_id", "block_id"],
            how="left",
            suffixes=("", "_override"),
        )
        merged["effective_min"] = merged["min_generation_mw_override"].combine_first(
            merged["min_generation_mw"]
        )
        merged["effective_max"] = merged["max_generation_mw_override"].combine_first(
            merged["max_generation_mw"]
        )

        assert len(merged) > 800, (
            f"only {len(merged)} (plant, stage, block) rows compared"
        )
        worst_min = (merged["min"] - merged["effective_min"]).abs().max()
        worst_max = (merged["max"] - merged["effective_max"]).abs().max()
        assert worst_min < 1e-6, f"min_generation_mw: worst |Δ| {worst_min} MW"
        assert worst_max < 1e-6, f"max_generation_mw: worst |Δ| {worst_max} MW"

    def test_override_rows_lose_no_resolution(self) -> None:
        """The per-block override mechanism itself is exact: for every block
        of a non-uniform stage, the emitted override row's `min`/`max` is a
        direct, untransformed copy of the source `inflex`/`disp` value —
        computed straight off the emitted table, must-run misallocation and
        capacity over-allowance are both exactly zero, replacing the
        previous interim `< 0.005` / `< 0.002` bounds on the old
        no-block-axis fold.

        Block-uniform stages are out of scope here: with no spread across
        blocks there is nothing to redistribute, so they carry no override
        row and no possible resolution loss by construction (criterion 2's
        base-row-fold test already grades their hours-weighted summary
        against the reference to `< 1e-6`).
        """
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.id_map import DecompIdMap
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger
        from cobre_bridge.decomp.thermal import _ct_dense

        dadger = Dadger.read(str(_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        plants = _ct_dense(dadger, calendar)
        table = _resolved_bounds_frame(dadger, id_map, calendar)

        overrides = table[table["block_id"].notna()]
        override_lookup = {
            (int(r.thermal_id), int(r.stage_id), int(r.block_id)): r
            for r in overrides.itertuples()
        }

        misallocated = 0.0
        over_allowed = 0.0
        checked_blocks = 0
        for code in id_map.thermal_codes:
            thermal_id = id_map.thermal_id(code)
            for stage in calendar:
                values = plants[code]["stages"][stage.index]
                disp, inflex = values["disp"], values["inflex"]
                uniform = all(d == disp[0] for d in disp) and all(
                    m == inflex[0] for m in inflex
                )
                if uniform:
                    continue
                for b, hours in enumerate(stage.block_hours):
                    row = override_lookup[(thermal_id, stage.index, b)]
                    misallocated += abs(inflex[b] - row.min_generation_mw) * hours
                    over_allowed += max(0.0, row.max_generation_mw - disp[b]) * hours
                    checked_blocks += 1

        assert checked_blocks > 0, "no block-varying stage found; nothing graded"
        assert misallocated == 0.0, f"must-run misallocation {misallocated} MWh"
        assert over_allowed == 0.0, f"capacity over-allowance {over_allowed} MWh"

    def test_no_override_row_for_block_uniform_stages(self) -> None:
        """A block-uniform ``(thermal, stage)`` gets only its base row —
        emitting per-block overrides there would inflate the table for no
        gain (AC #4)."""
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.id_map import DecompIdMap
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger
        from cobre_bridge.decomp.thermal import _ct_dense

        dadger = Dadger.read(str(_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        plants = _ct_dense(dadger, calendar)
        table = _resolved_bounds_frame(dadger, id_map, calendar)

        override_pairs = {
            (int(r.thermal_id), int(r.stage_id))
            for r in table[table["block_id"].notna()].itertuples()
        }

        for code in id_map.thermal_codes:
            thermal_id = id_map.thermal_id(code)
            for stage in calendar:
                values = plants[code]["stages"][stage.index]
                disp, inflex = values["disp"], values["inflex"]
                uniform = all(d == disp[0] for d in disp) and all(
                    m == inflex[0] for m in inflex
                )
                if uniform:
                    assert (thermal_id, stage.index) not in override_pairs

    def test_override_rows_carry_no_cost_and_are_fully_bounded(self) -> None:
        """Every per-block override row has `cost_per_mwh = None` (rule 37 —
        thermal cost is not block-eligible) and non-null `min`/`max` (AC #4)."""
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.id_map import DecompIdMap
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        table = _resolved_bounds_frame(dadger, id_map, calendar)

        overrides = table[table["block_id"].notna()]
        assert len(overrides) > 0, "no override rows found; nothing to grade"
        assert overrides["cost_per_mwh"].isna().all()
        assert overrides["min_generation_mw"].notna().all()
        assert overrides["max_generation_mw"].notna().all()

    def test_block_variation_is_rare_and_concentrated(self) -> None:
        """Few (plant, stage) rows vary across blocks at all.

        This is why the sparse override convention is cheap: only a handful
        of plants need per-block rows, not the fleet.
        """
        blocks = _reference_blocks()
        spread = blocks.group_by(["codigo_usina", "estagio"]).agg(
            (pl.col("min").max() - pl.col("min").min()).alias("min_spread"),
            (pl.col("max").max() - pl.col("max").min()).alias("max_spread"),
        )
        varying_min = spread.filter(pl.col("min_spread") > 1e-9)
        varying_max = spread.filter(pl.col("max_spread") > 1e-9)
        assert len(varying_min) < 0.1 * len(spread)
        assert len(varying_max) < 0.2 * len(spread)
