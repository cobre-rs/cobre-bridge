"""Read the source model's GNL (fuel-constrained) anticipated dispatch.

The source model declares its GNL thermals entirely in ``dadgnl`` — a separate
file from the ``CT`` thermal registry the main thermal converter reads — so
these plants are invisible to ``decomp/thermal.py`` and must be modelled here.
The **read/model layer** (:func:`read_gnl_model` and its helpers) is pure: it
turns ``dadgnl`` into a structured commitment model and does nothing else — no
``cobre`` import, no filesystem writes, no clamping, no decision about lead
declaration or ring placement; it returns the true committed values as data. The
**emission layer** (:func:`convert_gnl`) turns that model into cobre's
anticipated-dispatch inputs and owns the bounds policy the reader defers: it
clamps committed MW into each plant's ``[min_mw, max_mw]`` capability, warning
(via the module logger) on an out-of-range value, so the converted case never
pins a delivery cobre would reject.

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
* ``nl`` — the per-plant dispatch-anticipation lag in whole months
  (``codigo_usina`` → ``lag``): a plant's dispatch is decided ``lag`` months
  ahead of its delivery, and this is what sizes its physical ``lead_time_hours``.

A commitment's delivery date is recorded per stage so the emission track can
place it: an in-study ``gl`` commitment is folded onto the study stages as a
left-boundary ``past_anticipated_commitment``. The right boundary
(``future_anticipated_delivery``, priced against ``post_study_stages``) is
*synthesised* by the emission track as one free per-study-stage forward decision
on the mirror-shift calendar (the study stages shifted forward by the single
global anticipation lead ``H``) — it is not read from ``gl`` here.

Committed MW per stage is the block-duration-weighted mean of ``geracao`` over
that stage's own ``duracao`` blocks (``Σ_b duracao_b·geracao_b / Σ_b duracao_b``),
self-normalising so the committed MWh is preserved exactly regardless of block
count.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import pandas as pd
    from idecomp.decomp import Dadgnl

_LOG = logging.getLogger(__name__)

_NONZERO_TOLERANCE = 1e-9
_HOURS_PER_OPERATIVE_WEEK = 168  # 7 days x 24 h; the study/post-study grid step


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
    block-duration-weighted mean of that register's ``geracao``; ``hours`` is the
    stage span (sum of its ``duracao`` blocks), so the emission track can size a
    post-horizon delivery window ``[start_date, start_date + hours)``.
    """

    estagio: int
    start_date: date
    committed_mw: float
    hours: float


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
    """The GNL registry, its committed dispatch, and the weeks-per-month map.

    ``weeks_per_month`` is the ``gs`` block's ``{month_index: weeks}`` map (the
    number of operative weeks in each month, 1-based month index ascending).
    :func:`_study_lead_hours` sums it over the study's own months to size the
    single global anticipation lead ``H`` (a whole number of operative weeks),
    which shifts the study calendar forward into the mirror-shift post-study
    calendar the terminal FCF is priced against.

    ``nl_lag_months`` is the ``nl`` block's per-plant dispatch-anticipation lag
    (``{codigo_usina: months}``): the number of months by which a GNL plant's
    dispatch is decided ahead of its delivery (the LNG supply lead time). The
    mirror-shift emission uses one global lead for the whole fleet, so this drives
    only the uniform-lag consistency check (:func:`_warn_on_nonuniform_lag`) — a
    deck mixing lags is surfaced, not silently averaged.
    """

    thermals: tuple[GnlThermal, ...]
    commitments: dict[int, GnlCommitment]
    weeks_per_month: dict[int, int]
    nl_lag_months: dict[int, int]


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
        nl_lag_months=_read_nl_lags(dadgnl),
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
                hours=sum(duracao),
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


def _read_nl_lags(dadgnl: Dadgnl) -> dict[int, int]:
    """The ``nl`` dispatch-anticipation lags ``{codigo_usina: months}``.

    Each ``nl`` register carries ``codigo_usina``, ``codigo_submercado`` and
    ``lag`` (whole months of dispatch anticipation). Unlike ``gl``, ``nl`` is a
    fixed-shape register, so ``nl(df=True)`` is well-formed. Empty when the deck
    declares no ``nl`` block.
    """
    frame = dadgnl.nl(df=True)
    if frame is None or frame.empty:
        return {}
    return {int(row["codigo_usina"]): int(row["lag"]) for _, row in frame.iterrows()}


# ---------------------------------------------------------------------------
# Emission: GNL model -> cobre anticipated-dispatch inputs (both boundaries)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GnlEmission:
    """The cobre inputs an anticipated GNL fleet contributes to a converted case.

    ``thermals`` are created ``system/thermals.json`` entries (GNL plants are
    absent from ``CT``); ``past_anticipated_commitments`` and
    ``future_anticipated_deliveries`` extend ``initial_conditions.json`` (the
    left and right temporal boundary); ``post_study_stages`` is the standalone
    ``post_study_stages.json`` payload, ``None`` when the model declares no GS
    calendar (``weeks_per_month`` empty) and non-``None`` (its ``thermal_bounds``
    possibly empty) whenever a GS calendar exists.
    """

    thermals: list[dict]
    past_anticipated_commitments: list[dict]
    future_anticipated_deliveries: list[dict]
    post_study_stages: dict | None


def _study_lead_hours(
    stage_spans: Sequence[tuple[date, date]],
    weeks_per_month: Mapping[int, int],
) -> float:
    """The single global anticipation lead ``H``, a whole number of operative weeks.

    ``H = (Σ GS weeks over the study's months) × 168 h``. The study's month count
    is the number of distinct ``(year, month)`` among the study stage **start**
    dates; the first that many ``weeks_per_month`` values (ascending by month key)
    are the study months' operative-week counts. Using the operative-week count (a
    multiple of 168 h), not the trimmed study span, keeps the mirror calendar on
    the Saturday operative-week grid — a decision made in a Saturday-starting
    study week delivers in a Saturday-starting post-study week.

    Returns ``0.0`` when ``weeks_per_month`` is empty (no GS calendar to shift).
    """
    if not weeks_per_month:
        return 0.0
    study_month_count = len({(s.year, s.month) for s, _ in stage_spans})
    ordered_weeks = [weeks_per_month[k] for k in sorted(weeks_per_month)]
    return float(sum(ordered_weeks[:study_month_count]) * _HOURS_PER_OPERATIVE_WEEK)


def _cobre_safe_lead_hours(ideal_lead_hours: float, horizon_hours: float) -> float:
    """Cap the mirror lead strictly below the study horizon (a cobre workaround).

    TRACKED COBRE-GAP WORKAROUND (C13): cobre's LP builder panics with a
    divide-by-zero (``crates/cobre-sddp/src/lp/builder/entries.rs``, ``stage_idx
    % k_max``) when an anticipated ``LeadTime`` plant's lead reaches the full
    study horizon. cobre derives the in-study ring depth ``k_max`` from the lead
    alone: with a lead ``>=`` the horizon, every in-study delivery-stage decider
    is pre-study, so ``k_max`` collapses to ``0`` — yet the commitment-maturity
    ("fishing") rows still fire and index ``stage_idx % k_max``.

    The faithful mirror lead (:func:`_study_lead_hours`) is a whole number of
    study operative weeks, which on a horizon whose last stage does not end on
    the operative-week grid rounds *up* to ``>=`` the horizon — tripping the
    panic. Capping to the largest operative-week multiple STRICTLY below the
    horizon keeps ``k_max >= 1`` (one in-study decider survives). The mirror's
    decider mapping is lead-invariant (``window_end − H`` collapses to the study
    cumulative boundary for any ``H``), so every post-study delivery still lands
    on its own study stage; the only cost is the post-study calendar shifts one
    operative week less (study stage 0's delivery becomes the horizon-end stub
    window rather than a full week).

    Remove this cap (return ``ideal_lead_hours`` unchanged) once cobre handles
    ``k_max == 0`` by collapsing the in-study fishing rows. Tracked with its
    removal condition in ``~/git/cobre/plans/conversion-found-improvements.md``.
    """
    if ideal_lead_hours < horizon_hours:
        return ideal_lead_hours
    safe_weeks = int((horizon_hours - _NONZERO_TOLERANCE) // _HOURS_PER_OPERATIVE_WEEK)
    capped = float(safe_weeks * _HOURS_PER_OPERATIVE_WEEK)
    # Neutral, INFO-level note (out of the user's warning panel); the reason for
    # the cap is the dev-facing TRACKED COBRE-GAP WORKAROUND (C13) docstring above.
    _LOG.info(
        "GNL anticipation lead set to %.0f h (%d operative weeks); the post-study "
        "delivery calendar is shifted forward by that span",
        capped,
        safe_weeks,
    )
    return capped


def _warn_on_nonuniform_lag(nl_lag_months: Mapping[int, int]) -> None:
    """Warn when GNL plants declare differing anticipation lags (spec §7 A1).

    The single global lead ``H`` assumes every anticipated plant shares the
    study-span anticipation lag. A deck mixing ``nl`` lags would need per-plant
    windows — out of scope — so a non-uniform ``nl`` is surfaced (not silently
    averaged into one shift).
    """
    distinct = set(nl_lag_months.values())
    if len(distinct) > 1:
        _LOG.warning(
            "GNL plants declare differing anticipation lags %s; the single global "
            "lead assumes a uniform lag — the mirror-shift calendar approximates "
            "every plant at the study-span shift",
            sorted(distinct),
        )


def _clamp_committed(value: float, thermal: GnlThermal, context: str) -> float:
    """Clamp a committed MW into the plant's ``[min_mw, max_mw]`` capability.

    The emission site owns bounds policy (the reader returns the true committed
    values): the source model's ``gl`` geração and ``tg`` disponibilidade are
    independent fields, so a commitment can exceed capability, and a
    ``past_anticipated_commitment`` whose ``value_mw`` falls outside the plant's
    static generation bounds is rejected by cobre's semantic validator
    (``initial_conditions.rs`` requires every ``value_mw`` in ``[min, max]``). It
    is clamped into range with a warning instead — mirroring the sibling NEWAVE
    path (``converters/initial_conditions.py``).
    """
    lo, hi = thermal.min_mw, thermal.max_mw
    if lo > hi:
        _LOG.warning(
            "GNL %s: inflexibility %.4g > availability %.4g (degenerate bounds); "
            "clamping commitments to <= %.4g",
            thermal.name,
            lo,
            hi,
            hi,
        )
        lo = hi
    clamped = min(max(value, lo), hi)
    if abs(clamped - value) > _NONZERO_TOLERANCE:
        _LOG.warning(
            "GNL %s: committed %.4g MW (%s) outside [%.4g, %.4g]; clamped to %.4g",
            thermal.name,
            value,
            context,
            lo,
            hi,
            clamped,
        )
    return clamped


def _lead_delivery_stage_count(
    lead_hours: float, cumulative_hours: Sequence[float]
) -> int:
    """Leading study stages cobre treats as pre-study-committed for lead ``H``.

    Mirrors cobre-io's ``lead_delivery_stage_count`` for ``LeadTime``: the count
    of leading stages whose stage-end cumulative hours are ``<= H`` (tie-
    inclusive). The bridge tiles exactly these with
    ``past_anticipated_commitments`` so the left boundary matches the depth cobre
    derives from ``H`` — for a lead reaching past the horizon this is every study
    stage.
    """
    count = 0
    for boundary in cumulative_hours[1:]:
        if boundary > lead_hours:
            break
        count += 1
    return count


def _record_bound(
    bounds: list[dict],
    seen: set[tuple[int, int]],
    tid: int,
    idx: int,
    thermal: GnlThermal,
) -> None:
    """Append a ``(tid, idx)`` ``thermal_bounds`` row once (idempotent on ``seen``)."""
    if (tid, idx) in seen:
        return
    seen.add((tid, idx))
    bounds.append(
        {
            "thermal_id": tid,
            "post_study_stage_index": idx,
            "cost_per_mwh": thermal.cost_per_mwh,
            "min_mw": thermal.min_mw,
            "max_mw": thermal.max_mw,
        }
    )


def _calendar_stage_span(stage: Mapping) -> tuple[date, date]:
    """A calendar stage's ``[start, end)`` as parsed dates."""
    start = date.fromisoformat(stage["start_date"])
    return start, start + timedelta(hours=float(stage["duration_hours"]))


def _build_post_study_calendar(
    stage_spans: Sequence[tuple[date, date]], lead_hours: float
) -> list[dict]:
    """The mirror-shift post-study calendar: the study stages shifted forward by ``H``.

    Post-study stage ``m`` **ends** at ``study_stage_end[m] + lead_hours`` and
    **starts** at the previous post-study stage's end, with stage 0 starting at
    the study horizon end (``stage_spans[-1][1]``). So stage 0 spans
    ``[horizon_end, study_stage_end[0] + H)`` — absorbing the stub between the
    horizon end and the first shifted study-week end — while every later weekly
    stage is exactly ``168.0`` h and inherits the study calendar's Saturday
    alignment (``H`` is a whole number of weeks); the trailing stage mirrors the
    study monthly stage's own duration. One post-study stage per study stage.

    cobre resolves each delivery window's in-study decider end-anchored as the
    stage containing ``window_end_hours − H``; by construction a window that tiles
    post-study stage ``m`` has ``window_end_hours = study_cumulative[m + 1] + H``,
    so its decider collapses to study stage ``m`` — in range for every stage,
    nothing dropped.

    ``lead_hours`` is a multiple of 168 h (from :func:`_study_lead_hours`), so the
    forward shift is a whole number of days. Returns ``[]`` when ``lead_hours <=
    0`` (no GS calendar). Pure: no I/O, no ``cobre`` import.
    """
    if lead_hours <= 0.0:
        return []
    shift = timedelta(days=round(lead_hours / 24.0))
    horizon_end = stage_spans[-1][1]
    stages: list[dict] = []
    cursor = horizon_end
    for _, s_end in stage_spans:
        end = s_end + shift
        stages.append(
            {
                "start_date": cursor.isoformat(),
                "duration_hours": (end - cursor).days * 24.0,
            }
        )
        cursor = end
    return stages


def convert_gnl(
    model: GnlCommitmentModel,
    *,
    first_thermal_id: int,
    bus_id_of: Callable[[int], int],
    stages: Sequence[Mapping],
) -> GnlEmission:
    """Convert a :class:`GnlCommitmentModel` into cobre's anticipated-GNL inputs.

    Each GNL plant is *created* (absent from ``CT``) with a dense id assigned
    after the existing thermals (``first_thermal_id`` onward, ascending by code)
    and marked anticipated via ``anticipated_config = {"lead_time_hours": H}``,
    where ``H`` is the **single global** anticipation lead, shared by every GNL
    plant: the ideal is a whole number of study operative weeks
    (:func:`_study_lead_hours`), but the emitted value is that ideal **capped
    strictly below the study horizon by :func:`_cobre_safe_lead_hours` (TRACKED
    COBRE-GAP WORKAROUND C13)** — so on ``mar-26-rv2`` the emitted ``H`` is
    ``1008 h`` (6 weeks), not the ideal ``1176 h`` (7 weeks) that panics cobre's
    LP builder. The post-study calendar is the study stages
    shifted forward by ``H`` (:func:`_build_post_study_calendar`), so cobre's
    end-anchored decider (``window_end − H``) maps each post-study delivery back
    onto its own study stage (0→0, 1→1, …); nothing is dropped, and ``H ≥`` every
    study stage's duration, so no sub-stage (K=0) lead arises. Every anticipated
    thermal then gets:

    * ``past_anticipated_commitments`` tiling the leading
      ``_lead_delivery_stage_count(H)`` study stages (every study stage when
      ``H > horizon``) with the hours-weighted committed MW folded from the
      (weekly) ``gl`` deliveries onto each study stage (explicit ``0`` where
      none), each clamped into ``[min_mw, max_mw]`` — the mandatory left boundary;
    * ``future_anticipated_deliveries``, **free-only** (``min_mw ==
      thermal.min_mw`` / ``max_mw == thermal.max_mw`` — the decision cobre
      optimises within): one per study stage, placed index-direct onto its mirror
      post-study stage (study stage ``m`` → post-study stage ``m``) so each window
      tiles exactly one whole calendar stage (coverage 1.0). The source model's
      already-decided ``gl`` post-horizon commitments are *not* re-emitted as
      pinned deliveries — their fixed generation is an accepted modelling loss;
      the terminal FCF still prices every plant via its free per-stage lanes.

    Every referenced ``(thermal_id, post_study_stage_index)`` gets one
    ``thermal_bounds`` row carrying the plant's ``cvu`` (fuel-inclusive) as
    ``cost_per_mwh`` and its ``[min_mw, max_mw]`` capability, which contains the
    free delivery bound by construction. ``post_study_stages`` is ``None`` when
    the model declares no ``GS`` calendar (``model.weeks_per_month`` empty) — the
    deck's own signal that there is no post-study month to price.

    ``stages`` is the converted ``stages.json`` stage list (each a mapping with
    ``start_date``, ``end_date``, and ``blocks[].hours``).
    """
    horizon_start = date.fromisoformat(stages[0]["start_date"])
    stage_spans = [
        (date.fromisoformat(s["start_date"]), date.fromisoformat(s["end_date"]))
        for s in stages
    ]
    stage_hours = [sum(float(b["hours"]) for b in s["blocks"]) for s in stages]
    # Cumulative operative-stage boundaries S_0=0, S_1, .., S_n, matching cobre's
    # `cumulative_stage_boundaries(study_stage_durations)` — the clock the
    # anticipated-delivery decider is resolved against.
    cumulative_hours = [0.0]
    for h in stage_hours:
        cumulative_hours.append(cumulative_hours[-1] + h)

    # Single global anticipation lead H = (operative weeks in the study) x 168,
    # from GS; the post-study calendar is the study stages shifted forward by H.
    # The lead is capped strictly below the horizon to dodge a cobre k_max=0
    # LP-builder panic (TRACKED COBRE-GAP WORKAROUND C13, _cobre_safe_lead_hours).
    ideal_lead = _study_lead_hours(stage_spans, model.weeks_per_month)
    lead_hours = _cobre_safe_lead_hours(ideal_lead, cumulative_hours[-1])
    _warn_on_nonuniform_lag(model.nl_lag_months)
    calendar = _build_post_study_calendar(stage_spans, lead_hours)

    if calendar:
        # The left boundary tiles exactly the leading stages cobre derives from H
        # (every study stage, since H spans the whole study horizon).
        tile_k = _lead_delivery_stage_count(lead_hours, cumulative_hours)
    else:
        # No GS calendar (a real dadgnl always declares GS; this is a degenerate
        # deck): there is no post-study horizon to anticipate into, so the plant
        # keeps only the mandatory single leading commitment and a first-stage
        # physical lead — a valid anticipated thermal with no free deliveries.
        tile_k = 1
        lead_hours = stage_hours[0]

    gnl_id = {t.code: first_thermal_id + i for i, t in enumerate(model.thermals)}

    thermals: list[dict] = []
    past: list[dict] = []
    future: list[dict] = []
    bounds: list[dict] = []
    seen_bounds: set[tuple[int, int]] = set()

    for thermal in model.thermals:
        tid = gnl_id[thermal.code]
        commitment = model.commitments[thermal.code]

        # Fold weekly gl deliveries onto study stages (hours-weighted MW rate).
        folded: list[float] = []
        for s_start, s_end in stage_spans:
            windows = [
                (c.hours, c.committed_mw)
                for c in commitment.stages
                if s_start <= c.start_date < s_end
            ]
            total_h = sum(h for h, _ in windows)
            folded.append(
                sum(h * mw for h, mw in windows) / total_h if total_h > 0 else 0.0
            )

        thermals.append(
            {
                "id": tid,
                "name": thermal.name,
                "operational_start_date": horizon_start.isoformat(),
                "bus_id": bus_id_of(thermal.submarket_code),
                "cost_per_mwh": thermal.cost_per_mwh,
                "generation": {"min_mw": thermal.min_mw, "max_mw": thermal.max_mw},
                "anticipated_config": {"lead_time_hours": lead_hours},
                "entry_stage_id": None,
                "exit_stage_id": None,
            }
        )
        for j in range(tile_k):
            past.append(
                {
                    "thermal_id": tid,
                    "start_date": stages[j]["start_date"],
                    "end_date": stages[j]["end_date"],
                    "value_mw": _clamp_committed(
                        folded[j], thermal, f"in-horizon study stage {j}"
                    ),
                }
            )

        # Free forward decisions: one per study stage, placed index-direct onto
        # the mirror post-study stage that shares its index (study m -> post m).
        for m in range(len(calendar)):
            start, end = _calendar_stage_span(calendar[m])
            future.append(
                {
                    "thermal_id": tid,
                    "delivery_start": start.isoformat(),
                    "delivery_end": end.isoformat(),
                    "min_mw": thermal.min_mw,
                    "max_mw": thermal.max_mw,
                }
            )
            _record_bound(bounds, seen_bounds, tid, m, thermal)

    past.sort(key=lambda w: (w["thermal_id"], w["start_date"]))
    future.sort(key=lambda d: (d["thermal_id"], d["delivery_start"]))
    bounds.sort(key=lambda b: (b["thermal_id"], b["post_study_stage_index"]))
    post_study = {"stages": calendar, "thermal_bounds": bounds} if calendar else None

    return GnlEmission(
        thermals=thermals,
        past_anticipated_commitments=past,
        future_anticipated_deliveries=future,
        post_study_stages=post_study,
    )
