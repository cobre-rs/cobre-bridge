"""Grades the interim per-stage thermal fold against the reference run.

The source model declares thermal bounds per block; until the solver accepts
block-scoped bounds the converter folds them to one value per stage, weighted
by block hours. Two questions matter and they are different:

1. **Is the fold arithmetic right?** It must reproduce, exactly, the fold the
   reference's own per-block bounds imply. This is a hard invariant and the
   first test asserts it.
2. **What does the lost resolution cost?** Flattening preserves stage energy
   but moves it between blocks, so a peaking must-run is under-committed at
   peak and over-committed off-peak, and a flat cap can exceed the true cap
   in the block where it was lowest. The second test records the measured
   size of both effects, so an upgrade that changes them is visible.

Measured on ``decomp-jul-26-rv3`` (2026-07-25): the fold is exact on all 291
(plant, stage) rows; only 5 rows carry a block-varying minimum and 35 a
block-varying maximum; flattening misallocates 0.31 % of must-run energy —
72 % of it from a single peaking plant — and over-allows 0.078 % of capacity.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cobre_bridge.comparators.decomp_readers import read_dec_oper_usit

_DECK = Path("example/decomp-jul-26-rv3")
_needs_deck = pytest.mark.skipif(
    not (_DECK / "saidas" / "dec_oper_usit.csv").exists(),
    reason="reference deck outputs not present",
)


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
        """The emitted per-stage bounds are the reference's own hours fold.

        Anything else means the converter and the reference disagree about
        the stage a block belongs to or the weight it carries.
        """
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.id_map import DecompIdMap
        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger
        from cobre_bridge.decomp.thermal import convert_thermal_bounds

        dadger = Dadger.read(str(_DECK / "dadger.rv3"))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        ours = convert_thermal_bounds(dadger, id_map, calendar).to_pandas()

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

    def test_resolution_loss_stays_within_its_measured_size(self) -> None:
        """Records what the flattening costs, so an upgrade shows up here.

        Both figures are bounds on a known interim approximation, not
        tolerances on a correct result — they shrink to zero when block-scoped
        bounds land.
        """
        blocks = _reference_blocks()
        folded = _reference_fold(blocks)
        detail = blocks.join(folded, on=["codigo_usina", "estagio"])

        misallocated = (
            (detail["min"] - detail["min_folded"]).abs() * detail["duracao"]
        ).sum()
        must_run = (detail["min"] * detail["duracao"]).sum()
        assert misallocated / must_run < 0.005, (
            f"must-run misallocation grew to {100 * misallocated / must_run:.2f}%"
        )

        over = detail.filter(pl.col("max_folded") > pl.col("max"))
        over_allowed = ((over["max_folded"] - over["max"]) * over["duracao"]).sum()
        capacity = (detail["max"] * detail["duracao"]).sum()
        assert over_allowed / capacity < 0.002, (
            f"capacity over-allowance grew to {100 * over_allowed / capacity:.3f}%"
        )

    def test_block_variation_is_rare_and_concentrated(self) -> None:
        """Few (plant, stage) rows vary across blocks at all.

        This is why the interim fold is tolerable: the upgrade to block-scoped
        bounds matters for a handful of plants, not the fleet.
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
