"""Acceptance test pinning the availability rule against the reference run.

The source model's per-stage available capacity is a product of two
registers, and the discovery report flagged the combination as the highest
silent-wrong-answer risk in the conversion: get it wrong and every plant
carries a plausible but false cap. The rule is pinned here against the
reference run's own reported availability rather than against prose.

Verdict (deck ``decomp-jul-26-rv3``, 2026-07-25)::

    available_MW(plant, stage) = Σ_g installed_g × MP(plant, g, stage)
                                              × FD(plant, g, stage)

with one implicit group for an ordinary plant. The generating-unit group
axis is not decoration: the only plant carrying per-group register rows
(keyed by frequency) is the one whose two halves are maintained
independently, and the plant-level product misses it by up to 672 MW while
the group-wise sum lands on the reported value exactly.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from cobre_bridge.comparators.decomp_readers import read_dec_oper_usih

_DECK = Path("example/decomp-jul-26-rv3")
_needs_deck = pytest.mark.skipif(
    not (_DECK / "saidas" / "dec_oper_usih.csv").exists(),
    reason="reference deck outputs not present",
)

#: Tolerance in MW: the reference reports availability to the cent.
_TOL = 0.01

#: The plant whose maintenance and availability registers are declared per
#: generating-unit group, and the installed capacity of each group.
_SPLIT_PLANT = 66
_SPLIT_GROUPS_MW = (7000.0, 7000.0)


def _registers() -> tuple[dict, dict]:
    from idecomp.decomp import Dadger

    dadger = Dadger.read(str(_DECK / "dadger.rv3"))
    maintenance = dadger.mp(df=True)
    availability = dadger.fd(df=True)
    return maintenance, availability


def _reported() -> pl.DataFrame:
    stage_rows = read_dec_oper_usih(_DECK / "saidas").filter(
        pl.col("patamar").is_null()
    )
    return (
        stage_rows.group_by(["codigo_usina", "estagio"])
        .agg(
            pl.col("potencia_instalada_MW").mean().alias("installed"),
            pl.col("potencia_disponivel_MW").mean().alias("available"),
        )
        .sort(["codigo_usina", "estagio"])
    )


@_needs_deck
class TestAvailabilityRule:
    def test_product_of_both_registers_reproduces_single_group_plants(self) -> None:
        maintenance, availability = _registers()
        mp_by_plant = {
            int(r["codigo_usina"]): r
            for _, r in maintenance.drop_duplicates(
                "codigo_usina", keep=False
            ).iterrows()
        }
        fd_by_plant = {
            int(r["codigo_usina"]): r
            for _, r in availability.drop_duplicates(
                "codigo_usina", keep=False
            ).iterrows()
        }

        checked = 0
        for row in _reported().iter_rows(named=True):
            code, stage = int(row["codigo_usina"]), int(row["estagio"])
            if code == _SPLIT_PLANT:
                continue
            factor_m = (
                float(mp_by_plant[code][f"manutencao_{stage}"])
                if code in mp_by_plant
                else 1.0
            )
            factor_f = (
                float(fd_by_plant[code][f"fator_{stage}"])
                if code in fd_by_plant
                else 1.0
            )
            predicted = row["installed"] * factor_m * factor_f
            assert predicted == pytest.approx(row["available"], abs=_TOL), (
                f"plant {code} stage {stage}: predicted {predicted}, "
                f"reported {row['available']}"
            )
            checked += 1

        assert checked > 400, f"only {checked} (plant, stage) rows checked"

    def test_group_split_plant_needs_the_group_axis(self) -> None:
        """The split plant's availability is a per-group sum, not a plant product.

        Both halves carry their own maintenance and availability factors, so
        the plant-level product is wrong whenever the two differ — which is
        exactly what the generating-unit group representation exists for.
        """
        maintenance, availability = _registers()
        mp_rows = maintenance[maintenance["codigo_usina"] == _SPLIT_PLANT]
        fd_rows = availability[availability["codigo_usina"] == _SPLIT_PLANT]
        assert len(mp_rows) == len(_SPLIT_GROUPS_MW)
        assert set(mp_rows["frequencia"]) == {50.0, 60.0}

        reported = _reported().filter(pl.col("codigo_usina") == _SPLIT_PLANT)
        assert len(reported) > 0

        for row in reported.iter_rows(named=True):
            stage = int(row["estagio"])
            group_wise = 0.0
            for installed, (_, mp_row) in zip(
                _SPLIT_GROUPS_MW, mp_rows.iterrows(), strict=True
            ):
                frequency = mp_row["frequencia"]
                fd_row = fd_rows[fd_rows["frequencia"] == frequency].iloc[0]
                group_wise += (
                    installed
                    * float(mp_row[f"manutencao_{stage}"])
                    * float(fd_row[f"fator_{stage}"])
                )
            assert group_wise == pytest.approx(row["available"], abs=_TOL)

            plant_wise = (
                row["installed"]
                * float(mp_rows.iloc[0][f"manutencao_{stage}"])
                * float(fd_rows.iloc[0][f"fator_{stage}"])
            )
            if stage == 1:
                # Stage 1 is where the two halves differ most; the plant-level
                # product understates the fleet by the whole 60 Hz allowance.
                assert abs(plant_wise - row["available"]) > 100.0
