"""The source model's GNL commitment reader (``dadgnl``: ``tg``/``gl``/``gs``).

A GNL (fuel-constrained) thermal is declared entirely in ``dadgnl``, never in
the ``CT`` registry ``decomp/thermal.py::convert_thermals`` reads — GNL codes
are simply absent from ``CT``. This module reads ``dadgnl``'s three tables
into a plant registry (:class:`GnlThermal`) and a per-stage committed
dispatch (:class:`GnlCommitment`), bundled as :class:`GnlCommitmentModel`.

It decides nothing about the lead declaration or ring placement (see the
epic's ticket-002) and performs no I/O beyond the accessor calls on an
already-loaded ``dadgnl``: the caller owns ``Dadgnl.read``, cobre emission,
and bounds clamping (ticket-003).

The three tables (verified via ``.tg/.gl/.gs(df=True)``):

* ``tg`` — the GNL plant registry: ``codigo_submercado, codigo_usina,
  estagio, nome, cvu_1..N, disponibilidade_1..N, inflexibilidade_1..N``.
  Identical block-field shape to ``CT`` (sparse by stage; stage 1
  mandatory). Only the stage-1 row is read here — :class:`GnlThermal`
  carries the plant's static registry data, not a per-stage bounds series.
* ``gl`` — the committed weekly dispatch: ``codigo_submercado,
  codigo_usina, data_inicio, estagio, duracao_1..N, geracao_1..N``. Every
  declared stage is kept (no forward-fill; a stage's absence means no
  commitment that week).
* ``gs`` — ``{mes: semanas}``, carried through verbatim.

Block weighting
---------------
The registry's ``cost_per_mwh``/``min_mw``/``max_mw`` are hours-weighted by
the deck's stage-1 block hours
(:func:`cobre_bridge.decomp.temporal.hours_weighted` — the same convention
``decomp/thermal.py`` uses for ``CT``). The commitment's per-stage committed
MW is instead self-normalising against ``gl``'s *own* ``duracao_b`` — not the
calendar's block hours — so a plant's committed MWh is preserved exactly
regardless of any block-count skew between ``gl`` and the calendar::

    MW_eq = Σ_b (duracao_b / Σ_b duracao_b) · geracao_b

Activation (gate G6)
---------------------
There is no dger-equivalent flag for GNL commitment in the source model
(``dadger`` has no ``.gl`` accessor); the gate is purely data-driven:
``dadgnl`` present, ``gl(df=True)`` non-empty, and at least one nonzero
``geracao_*`` value. See :func:`is_gnl_enabled`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from cobre_bridge.decomp.temporal import hours_weighted

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadgnl

    from cobre_bridge.decomp.temporal import OperativeStage


@dataclass(frozen=True)
class GnlThermal:
    """One GNL plant's static registry data (``tg``'s stage-1 row)."""

    code: int
    name: str
    submarket_code: int
    cost_per_mwh: float
    min_mw: float
    max_mw: float


@dataclass(frozen=True)
class GnlCommitment:
    """One GNL plant's committed weekly dispatch (``gl``).

    ``committed_mw_by_stage`` is keyed by 1-based ``estagio``; a plant with
    no ``gl`` rows (registry-only) carries an empty dict, never omitted.
    """

    code: int
    committed_mw_by_stage: dict[int, float]


@dataclass(frozen=True)
class GnlCommitmentModel:
    """The bundled GNL registry + commitments + weeks-per-month map."""

    thermals: tuple[GnlThermal, ...]
    commitments: dict[int, GnlCommitment]
    weeks_per_month: dict[int, int]


def is_gnl_enabled(dadgnl: Dadgnl | None) -> bool:
    """Gate G6: is the source model's GNL commitment data active?

    True when *dadgnl* is present, its ``gl`` table is non-empty, and it
    carries at least one nonzero ``geracao_*`` value. There is no
    dger-equivalent flag to check (``dadger`` has no ``.gl`` accessor); the
    gate is entirely data-driven.
    """
    if dadgnl is None:
        return False
    gl = dadgnl.gl(df=True)
    if gl is None or gl.empty:
        return False
    geracao_columns = [c for c in gl.columns if c.startswith("geracao_")]
    for column in geracao_columns:
        for value in gl[column]:
            if not pd.isna(value) and float(value) != 0.0:
                return True
    return False


def read_gnl_model(
    dadgnl: Dadgnl,
    calendar: Sequence[OperativeStage],
) -> GnlCommitmentModel | None:
    """Read ``dadgnl`` into a :class:`GnlCommitmentModel`, or ``None`` when
    the activation gate (G6, see :func:`is_gnl_enabled`) is off.

    ``calendar`` supplies the stage-1 block hours the ``tg`` registry's
    static values are weighted over; it plays no role in the ``gl``
    commitment weighting (self-normalising against ``gl``'s own
    ``duracao`` — see the module docstring).
    """
    if not is_gnl_enabled(dadgnl):
        return None

    thermals = _read_registry(dadgnl.tg(df=True), calendar)
    commitments = _read_commitments(dadgnl.gl(df=True), thermals)
    weeks_per_month = _read_weeks_per_month(dadgnl.gs(df=True))

    return GnlCommitmentModel(
        thermals=thermals,
        commitments=commitments,
        weeks_per_month=weeks_per_month,
    )


def _blocks(prefix: str, row: pd.Series, n: int) -> list[float]:
    """Read *n* 1-based ``{prefix}_k`` columns off *row*, ``NaN`` -> ``0.0``.

    Mirrors ``decomp/thermal.py::_ct_dense``'s ``_blocks`` closure — a
    missing patamar (fewer columns declared for this row than for the
    table's widest row) contributes a zero-duration, zero-value block,
    which is inert under both the hours-weighted mean and the
    self-normalising commitment mean below.
    """
    return [
        0.0 if pd.isna(row[f"{prefix}_{k}"]) else float(row[f"{prefix}_{k}"])
        for k in range(1, n + 1)
    ]


def _read_registry(
    tg: pd.DataFrame | None,
    calendar: Sequence[OperativeStage],
) -> tuple[GnlThermal, ...]:
    """Build the ascending-by-code GNL registry from ``tg``'s stage-1 rows."""
    if tg is None or tg.empty:
        return ()

    first_stage = calendar[0]
    n_blocks = len(first_stage.block_hours)

    thermals: list[GnlThermal] = []
    for code, group in tg.groupby("codigo_usina", sort=True):
        code_int = int(code)
        stage_one = group[group["estagio"] == 1]
        if stage_one.empty:
            raise ValueError(
                f"GNL plant {code_int} does not declare stage 1 in tg; "
                "sparse-stage inheritance has no base"
            )
        row = stage_one.iloc[0]
        thermals.append(
            GnlThermal(
                code=code_int,
                name=str(row["nome"]).strip(),
                submarket_code=int(row["codigo_submercado"]),
                cost_per_mwh=hours_weighted(_blocks("cvu", row, n_blocks), first_stage),
                min_mw=hours_weighted(
                    _blocks("inflexibilidade", row, n_blocks), first_stage
                ),
                max_mw=hours_weighted(
                    _blocks("disponibilidade", row, n_blocks), first_stage
                ),
            )
        )
    thermals.sort(key=lambda t: t.code)
    return tuple(thermals)


def _block_weighted_mw(geracao: Sequence[float], duracao: Sequence[float]) -> float:
    """Self-normalising block-weighted mean committed MW.

    ``Σ_b (duracao_b / Σ_b duracao_b) · geracao_b`` — weighted by *duracao*
    itself rather than the calendar's block hours, so a plant's committed
    MWh is preserved exactly regardless of any block-count skew.
    """
    total_duracao = sum(duracao)
    return sum(g * d for g, d in zip(geracao, duracao, strict=True)) / total_duracao


def _read_commitments(
    gl: pd.DataFrame | None,
    thermals: tuple[GnlThermal, ...],
) -> dict[int, GnlCommitment]:
    """Build one :class:`GnlCommitment` per registry code from ``gl``.

    Every code in *thermals* gets an entry — a registry-only code (no
    ``gl`` rows) still gets a :class:`GnlCommitment` with an empty
    ``committed_mw_by_stage``, never dropped, never fabricated.
    """
    registry_codes = {t.code for t in thermals}
    by_stage: dict[int, dict[int, float]] = {code: {} for code in registry_codes}

    if gl is not None and not gl.empty:
        n_blocks = sum(1 for c in gl.columns if c.startswith("duracao_"))
        for _, row in gl.iterrows():
            code = int(row["codigo_usina"])
            if code not in registry_codes:
                raise ValueError(
                    f"gl dispatches GNL plant {code}, which is absent from "
                    "the tg registry"
                )
            stage = int(row["estagio"])
            duracao = _blocks("duracao", row, n_blocks)
            geracao = _blocks("geracao", row, n_blocks)
            by_stage[code][stage] = _block_weighted_mw(geracao, duracao)

    return {
        code: GnlCommitment(code=code, committed_mw_by_stage=stages)
        for code, stages in by_stage.items()
    }


def _read_weeks_per_month(gs: pd.DataFrame | None) -> dict[int, int]:
    """Verbatim ``{mes: semanas}`` map from ``gs``."""
    if gs is None or gs.empty:
        return {}
    return {int(row["mes"]): int(row["semanas"]) for _, row in gs.iterrows()}
