"""Tier-1 tests for the special-constraint register readers (M2.1 T1).

Synthetic-fixture only — a fake ``Dadger`` returns pandas DataFrames shaped like
``idecomp``'s ``df=True`` accessors, so these run in CI with no real deck.
"""

from __future__ import annotations

import pandas as pd

from cobre_bridge.decomp.constraint_registers import (
    ConstraintRecord,
    lowers_to_bound,
    read_constraints,
)


class _FakeDadger:
    """Return a preset DataFrame (or ``None``) for each register accessor."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):  # noqa: ANN204 - test double
        if name.startswith("_"):
            raise AttributeError(name)

        def accessor(df: bool = True) -> pd.DataFrame | None:
            return self._frames.get(name)

        return accessor


def _decl(*rows: tuple[int, int, int]) -> pd.DataFrame:
    """RE/HQ/HV declaration rows: (constraint_id, estagio_inicial, estagio_final)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "estagio_inicial", "estagio_final"]
    )


def _coeff(*rows: tuple, tipo: bool = False, freq: bool = False) -> pd.DataFrame:
    """FU/CQ/CV rows: (constraint_id, codigo_usina, coeficiente, estagio[, tipo|freq])."""
    cols = ["codigo_restricao", "codigo_usina", "coeficiente", "estagio"]
    if tipo:
        cols.append("tipo")
    if freq:
        cols.append("frequencia")
    return pd.DataFrame(rows, columns=cols)


def _lu(*rows: tuple) -> pd.DataFrame:
    """Per-block limit rows: (cid, estagio, li1, ls1, li2, ls2, li3, ls3)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio",
            "limite_inferior_1",
            "limite_superior_1",
            "limite_inferior_2",
            "limite_superior_2",
            "limite_inferior_3",
            "limite_superior_3",
        ],
    )


def _lv(*rows: tuple) -> pd.DataFrame:
    """Stage-level limit rows: (cid, estagio, limite_inferior, limite_superior)."""
    return pd.DataFrame(
        rows,
        columns=["codigo_restricao", "estagio", "limite_inferior", "limite_superior"],
    )


def _only(census_records: object, cid: int) -> ConstraintRecord:
    records = [r for r in census_records if r.constraint_id == cid]  # type: ignore[attr-defined]
    assert len(records) == 1, f"expected exactly one record for {cid}"
    return records[0]


def test_re_single_term_lowers_to_generation_bound() -> None:
    dadger = _FakeDadger(
        re=_decl((10, 1, 2)),
        fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
        lu=_lu((10, 1, 50.0, 212.0, 50.0, 212.0, 50.0, 212.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 10)
    assert rec.is_single_term
    assert rec.terms[0].variable == "generation"
    assert lowers_to_bound(rec)
    assert rec in census.to_bounds


def test_re_multi_term_lowers_to_generic() -> None:
    dadger = _FakeDadger(
        re=_decl((401, 1, 3)),
        fu=_coeff(
            (401, 285, 1.0, 1, float("nan")),
            (401, 287, 1.0, 1, float("nan")),
            freq=True,
        ),
        lu=_lu((401, 1, 0.0, 500.0, 0.0, 500.0, 0.0, 500.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 401)
    assert not rec.is_single_term
    assert not lowers_to_bound(rec)
    assert rec in census.to_generic


def test_hq_qdef_single_lowers_to_outflow_bound() -> None:
    dadger = _FakeDadger(
        hq=_decl((5, 1, 1)),
        cq=_coeff((5, 30, 1.0, 1, "QDEF"), tipo=True),
        lq=_lu((5, 1, 10.0, 100.0, 10.0, 100.0, 10.0, 100.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 5)
    assert rec.terms[0].variable == "QDEF"
    assert lowers_to_bound(rec)


def test_hq_qdes_single_stays_generic_no_bound_axis() -> None:
    """Spillage has no cobre bounds axis — a single QDES term is still generic."""
    dadger = _FakeDadger(
        hq=_decl((6, 1, 1)),
        cq=_coeff((6, 31, 1.0, 1, "QDES"), tipo=True),
        lq=_lu((6, 1, 0.0, 50.0, 0.0, 50.0, 0.0, 50.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 6)
    assert rec.is_single_term
    assert not lowers_to_bound(rec)
    assert rec in census.to_generic


def test_hq_mixed_flow_variables_one_plant_is_multi_term_generic() -> None:
    """QDEF + QDES on the same plant is two terms -> a generic constraint."""
    dadger = _FakeDadger(
        hq=_decl((164, 1, 1)),
        cq=_coeff(
            (164, 182, 1.0, 1, "QDEF"),
            (164, 182, 1.0, 1, "QDES"),
            tipo=True,
        ),
        lq=_lu((164, 1, 0.0, 900.0, 0.0, 900.0, 0.0, 900.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 164)
    assert rec.is_single_entity  # one plant …
    assert not rec.is_single_term  # … but two variables
    assert not lowers_to_bound(rec)


def test_hv_varm_single_lowers_to_storage_bound() -> None:
    dadger = _FakeDadger(
        hv=_decl((7, 1, 2)),
        cv=_coeff((7, 40, 1.0, 1, "VARM"), tipo=True),
        lv=_lv((7, 1, 100.0, 900.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HV"], 7)
    assert not rec.per_block
    assert rec.terms[0].variable == "VARM"
    assert lowers_to_bound(rec)
    # Stage-level bounds carry a single slot.
    assert rec.bounds[0].lower == (100.0,)
    assert rec.bounds[0].upper == (900.0,)


def test_he_energy_over_ree_is_always_generic() -> None:
    dadger = _FakeDadger(
        he=pd.DataFrame(
            [(100, 1, 20.0, 2, 3370.0)],
            columns=[
                "codigo_restricao",
                "estagio",
                "limite",
                "tipo_limite",
                "valor_penalidade",
            ],
        ),
        cm=pd.DataFrame(
            [(100, 3, 1.0)],
            columns=["codigo_restricao", "codigo_ree", "coeficiente"],
        ),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HE"], 100)
    assert rec.terms[0].variable == "energy"
    assert rec.is_single_term  # one REE …
    assert not lowers_to_bound(rec)  # … but energy has no bounds axis
    assert rec.tipo_limite == 2
    assert rec.penalty == 3370.0
    assert rec.bounds[0].lower == (20.0,)  # tipo_limite 2 => a floor


def test_sparse_limits_forward_fill_over_stage_range() -> None:
    """A limit declared only at the first stage densifies over the whole range."""
    dadger = _FakeDadger(
        re=_decl((10, 1, 3)),
        fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
        lu=_lu((10, 1, 5.0, 9.0, 5.0, 9.0, 5.0, 9.0)),  # declared at stage 1 only
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 10)
    assert sorted(rec.bounds) == [0, 1, 2]  # dense over the 0-based range
    assert all(rec.bounds[s].lower[0] == 5.0 for s in (0, 1, 2))


def test_declared_constraint_without_participation_is_skipped() -> None:
    dadger = _FakeDadger(
        re=_decl((10, 1, 1), (11, 1, 1)),  # 11 has no FU rows
        fu=_coeff((10, 21, 1.0, 1, float("nan")), freq=True),
        lu=_lu((10, 1, 5.0, 9.0, 5.0, 9.0, 5.0, 9.0)),
    )
    census = read_constraints(dadger)
    assert {r.constraint_id for r in census.by_family["RE"]} == {10}


def test_per_block_bounds_read_all_declared_slots() -> None:
    dadger = _FakeDadger(
        hq=_decl((5, 1, 1)),
        cq=_coeff((5, 30, 1.0, 1, "QTUR"), tipo=True),
        lq=_lu((5, 1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["HQ"], 5)
    assert rec.per_block
    assert rec.bounds[0].lower == (1.0, 3.0, 5.0)
    assert rec.bounds[0].upper == (2.0, 4.0, 6.0)


def test_frequency_split_plant_contributes_two_terms() -> None:
    """A plant declared at 50 and 60 Hz is two terms -> generic, not one bound."""
    dadger = _FakeDadger(
        re=_decl((20, 1, 1)),
        fu=_coeff(
            (20, 66, 1.0, 1, 50.0),
            (20, 66, 1.0, 1, 60.0),
            freq=True,
        ),
        lu=_lu((20, 1, 0.0, 14000.0, 0.0, 14000.0, 0.0, 14000.0)),
    )
    census = read_constraints(dadger)
    rec = _only(census.by_family["RE"], 20)
    assert rec.is_single_entity  # one plant code …
    assert not rec.is_single_term  # … split across two frequencies
    assert {t.frequency for t in rec.terms} == {50.0, 60.0}
    assert not lowers_to_bound(rec)
