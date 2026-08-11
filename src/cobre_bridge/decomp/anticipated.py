"""Read the source model's GNL (fuel-constrained) anticipated dispatch.

The source model declares its GNL thermals entirely in ``dadgnl`` — a separate
file from the ``CT`` thermal registry the main thermal converter reads — so
these plants are invisible to ``decomp/thermal.py`` and must be modelled here.
This module is the **pure read/model layer**: it turns ``dadgnl`` into a
structured commitment model and does nothing else — no ``cobre`` import, no
filesystem writes, no :class:`~cobre_bridge.diagnostics.Diagnostic`, no clamping
into bounds (a conversion-site policy) and no decision about lead declaration or
ring placement (the emission track's job). It returns data only.

``dadgnl`` has three register families:

* ``tg`` — the GNL thermal registry (one row per plant): ``codigo_usina``,
  ``codigo_submercado``, ``nome``, and per-block ``cvu`` (fuel cost, $/MWh),
  ``disponibilidade`` (max MW), ``inflexibilidade`` (min MW). Fixed 3-block shape,
  so ``tg(df=True)`` is well-formed.
* ``gl`` — the committed weekly dispatch: one register per ``(codigo_usina,
  estagio)`` carrying ``data_inicio`` (the delivery-stage start date, a
  ``ddmmyyyy`` string), a per-block ``duracao`` list, and a per-block ``geracao``
  (committed MW) list. Block counts vary across weekly stages, so ``gl(df=True)``
  is **not** usable (the ragged per-block lists raise "All arrays must be of the
  same length"); the registers are iterated directly instead.
* ``gs`` — the weeks-per-month calendar map (``mes`` → ``semanas``).

A commitment's delivery date decides its boundary: delivered in-study it is a
left-boundary ``past_anticipated_commitment``; delivered after the study horizon
it is a right-boundary ``future_anticipated_delivery`` (priced against
``post_study_stages``). This module records the parsed ``date`` per stage so the
emission track can make that split; it does not make it here.

Committed MW per stage is the block-duration-weighted mean of ``geracao`` over
that stage's own ``duracao`` blocks (``Σ_b duracao_b·geracao_b / Σ_b duracao_b``),
self-normalising so the committed MWh is preserved exactly regardless of block
count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from idecomp.decomp import Dadgnl

_NONZERO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class GnlThermal:
    """One GNL plant's registry data (from ``tg``), block-weighted to a scalar.

    ``cost_per_mwh``/``min_mw``/``max_mw`` come from the plant's ``cvu`` /
    ``inflexibilidade`` / ``disponibilidade`` block values, weighted by its
    stage-1 ``gl`` block durations (uniform when the plant has no ``gl`` stage-1
    register or the block counts disagree). No clamping — the emission site owns
    bounds policy.
    """

    code: int
    name: str
    submarket_code: int
    cost_per_mwh: float
    min_mw: float
    max_mw: float


@dataclass(frozen=True)
class GnlStageCommitment:
    """One ``gl`` register: a committed MW at one delivery stage, with its date.

    ``start_date`` is the parsed ``data_inicio`` (the delivery stage's start);
    the emission track compares it against the study horizon to route the
    commitment to the left or right temporal boundary. ``committed_mw`` is the
    block-duration-weighted mean of that register's ``geracao``.
    """

    estagio: int
    start_date: date
    committed_mw: float


@dataclass(frozen=True)
class GnlCommitment:
    """One plant's committed dispatch across every ``gl`` stage it declares.

    ``stages`` is ascending by ``estagio`` and may be empty for a plant present
    in ``tg`` but absent from ``gl`` (registry-only, no committed dispatch —
    never dropped, never fabricated).
    """

    code: int
    stages: tuple[GnlStageCommitment, ...]


@dataclass(frozen=True)
class GnlCommitmentModel:
    """The GNL registry, its committed dispatch, and the weeks-per-month map."""

    thermals: tuple[GnlThermal, ...]
    commitments: dict[int, GnlCommitment]
    weeks_per_month: dict[int, int]


def _as_floats(value: object) -> list[float]:
    """Coerce a register field to a list of floats (scalar → 1-list, None → [])."""
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [float(v) for v in value]
    return [float(value)]  # type: ignore[arg-type]


def _block_weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    """``Σ w·v / Σ w`` over aligned blocks; uniform mean when weights are unusable.

    Falls back to the plain mean of ``values`` when ``weights`` is empty, sums to
    zero, or has a different length than ``values`` — the value must survive a
    missing/degenerate block-hours vector rather than vanish.
    """
    if not values:
        return 0.0
    if len(weights) == len(values):
        total_w = sum(weights)
        if total_w > 0.0:
            return sum(v * w for v, w in zip(values, weights, strict=True)) / total_w
    return sum(values) / len(values)


def _parse_data_inicio(raw: object) -> date:
    """Parse a ``gl`` register's ``data_inicio`` into a :class:`datetime.date`.

    Accepts an already-parsed ``date``, or a ``ddmmyyyy`` string / integer (the
    on-file form, e.g. ``"14032026"`` → 2026-03-14). Integers are zero-padded to
    eight digits first, so a dropped leading-zero day (``4042026``) parses as
    ``2026-04-04``.
    """
    if isinstance(raw, date):
        return raw
    text = f"{int(raw):08d}" if isinstance(raw, int) else str(raw).strip()
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"unparseable gl data_inicio {raw!r} (expected ddmmyyyy)")
    day, month, year = int(text[0:2]), int(text[2:4]), int(text[4:8])
    return date(year, month, day)


def is_gnl_enabled(dadgnl: Dadgnl | None) -> bool:
    """Whether ``dadgnl`` declares any committed GNL dispatch (the G6 gate).

    ``True`` iff ``dadgnl`` is present and at least one ``gl`` register carries a
    nonzero ``geracao``. There is no source-model ``dger``-equivalent activation
    flag for GNL, so presence of real committed generation is the gate. A deck
    with only a ``tg`` registry (or all-zero ``gl``) is treated as GNL-off.
    """
    if dadgnl is None:
        return False
    registers = dadgnl.gl()
    if not registers:
        return False
    return any(
        abs(g) > _NONZERO_TOLERANCE
        for register in registers
        for g in _as_floats(register.geracao)
    )


def read_gnl_model(dadgnl: Dadgnl) -> GnlCommitmentModel | None:
    """Read ``dadgnl`` into a :class:`GnlCommitmentModel`, or ``None`` if GNL-off.

    Returns ``None`` when :func:`is_gnl_enabled` is ``False``. Otherwise builds
    the ``tg`` registry (ascending by code), the ``gl`` commitments (keyed by
    code, ascending by ``estagio``, each stage carrying its parsed delivery date
    and block-weighted committed MW), and the ``gs`` weeks-per-month map.

    Raises
    ------
    ValueError
        If a ``gl`` register names a ``codigo_usina`` with no ``tg`` registry
        entry (a committed dispatch for an unknown plant — a malformed deck).
    """
    if not is_gnl_enabled(dadgnl):
        return None

    registry = _read_tg_registry(dadgnl)
    stage1_weights = _stage1_block_hours(dadgnl)
    thermals = tuple(
        _build_gnl_thermal(row, stage1_weights.get(int(row["codigo_usina"])))
        for _, row in registry.sort_values("codigo_usina").iterrows()
    )
    known_codes = {t.code for t in thermals}

    commitments = _read_gl_commitments(dadgnl, known_codes)
    # Registry-only plants (in tg, absent from gl) still get an empty commitment.
    for thermal in thermals:
        commitments.setdefault(
            thermal.code, GnlCommitment(code=thermal.code, stages=())
        )

    return GnlCommitmentModel(
        thermals=thermals,
        commitments=commitments,
        weeks_per_month=_read_weeks_per_month(dadgnl),
    )


def _read_tg_registry(dadgnl: Dadgnl) -> pd.DataFrame:
    """The one-row-per-plant ``tg`` registry (stage-1 base), as a DataFrame."""
    frame = dadgnl.tg(df=True)
    # One registry row per plant: the earliest stage carries the base cadastro.
    return frame.sort_values(["codigo_usina", "estagio"]).drop_duplicates(
        "codigo_usina", keep="first"
    )


def _stage1_block_hours(dadgnl: Dadgnl) -> dict[int, list[float]]:
    """Each plant's stage-1 ``gl`` block durations, for weighting the registry."""
    weights: dict[int, list[float]] = {}
    for register in dadgnl.gl():
        if int(register.estagio) == 1:
            weights.setdefault(int(register.codigo_usina), _as_floats(register.duracao))
    return weights


def _build_gnl_thermal(row: pd.Series, block_hours: list[float] | None) -> GnlThermal:
    """Assemble one :class:`GnlThermal` from a ``tg`` row + stage-1 block hours."""
    weights = block_hours or []
    cvu = [float(row[f"cvu_{b}"]) for b in (1, 2, 3)]
    disp = [float(row[f"disponibilidade_{b}"]) for b in (1, 2, 3)]
    inflex = [float(row[f"inflexibilidade_{b}"]) for b in (1, 2, 3)]
    return GnlThermal(
        code=int(row["codigo_usina"]),
        name=str(row["nome"]).strip(),
        submarket_code=int(row["codigo_submercado"]),
        cost_per_mwh=_block_weighted_mean(cvu, weights),
        min_mw=_block_weighted_mean(inflex, weights),
        max_mw=_block_weighted_mean(disp, weights),
    )


def _read_gl_commitments(
    dadgnl: Dadgnl, known_codes: set[int]
) -> dict[int, GnlCommitment]:
    """Build ``{code: GnlCommitment}`` by iterating ``gl`` registers.

    Iterates registers (never ``gl(df=True)`` — the ragged per-block lists make
    it unusable). Each register contributes one :class:`GnlStageCommitment` with
    its parsed date and block-weighted committed MW.
    """
    by_code: dict[int, list[GnlStageCommitment]] = {}
    for register in dadgnl.gl():
        code = int(register.codigo_usina)
        if code not in known_codes:
            raise ValueError(
                f"gl declares a committed dispatch for code {code} with no tg "
                "registry entry (a dispatch for an unknown plant)"
            )
        geracao = _as_floats(register.geracao)
        duracao = _as_floats(register.duracao)
        by_code.setdefault(code, []).append(
            GnlStageCommitment(
                estagio=int(register.estagio),
                start_date=_parse_data_inicio(register.data_inicio),
                committed_mw=_block_weighted_mean(geracao, duracao),
            )
        )
    return {
        code: GnlCommitment(
            code=code, stages=tuple(sorted(stages, key=lambda s: s.estagio))
        )
        for code, stages in by_code.items()
    }


def _read_weeks_per_month(dadgnl: Dadgnl) -> dict[int, int]:
    """The ``gs`` weeks-per-month map ``{mes: semanas}`` (empty when absent)."""
    frame = dadgnl.gs(df=True)
    if frame is None or frame.empty:
        return {}
    return {int(row["mes"]): int(row["semanas"]) for _, row in frame.iterrows()}
