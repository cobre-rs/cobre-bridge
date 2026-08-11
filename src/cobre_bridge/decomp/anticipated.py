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

    ``nl_lag_months`` is the ``nl`` block's per-plant dispatch-anticipation lag
    (``{codigo_usina: months}``): the number of months by which a GNL plant's
    dispatch is decided ahead of its delivery (the LNG supply lead time). It is
    what sets each anticipated thermal's physical ``lead_time_hours`` — a plant
    absent from ``nl`` has no declared lead.
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
    ``post_study_stages.json`` payload, ``None`` when no delivery lands
    post-horizon.
    """

    thermals: list[dict]
    past_anticipated_commitments: list[dict]
    future_anticipated_deliveries: list[dict]
    post_study_stages: dict | None


def _lead_stage_count(committed_by_study_stage: Sequence[float]) -> int:
    """``K`` — leading study stages the plant is anticipated over.

    Through the last in-horizon study stage carrying a nonzero commitment, and
    at least ``1`` (cobre requires every anticipated thermal to tile >= 1 leading
    stage via ``past_anticipated_commitments``, even one that only delivers
    post-horizon or never — the source model's ``gl`` declares the plant, so it
    is anticipated regardless of its committed level).
    """
    last_nonzero = -1
    for j, mw in enumerate(committed_by_study_stage):
        if abs(mw) > _NONZERO_TOLERANCE:
            last_nonzero = j
    return max(last_nonzero + 1, 1)


def _clamp_committed(value: float, thermal: GnlThermal, context: str) -> float:
    """Clamp a committed MW into the plant's ``[min_mw, max_mw]`` capability.

    The emission site owns bounds policy (the reader returns the true committed
    values): the source model's ``gl`` geração and ``tg`` disponibilidade are
    independent fields, so a commitment can exceed capability, and a delivery
    pinned outside the plant's static generation bounds is rejected by cobre's
    semantic validator. It is clamped into range with a warning instead —
    mirroring the sibling NEWAVE path (``converters/initial_conditions.py``).
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


def _delivery_window_end(stages: Sequence[GnlStageCommitment], i: int) -> date:
    """End of ``stages[i]``'s delivery window, from the estágio cadence.

    The source model's ``gl`` deliveries are weekly, but the last register's
    ``duracao`` is empty on real decks, so the span cannot come from that
    register's own hours. It is taken from the spacing of consecutive estágios:
    the next estágio's start, or (for the last) the previous cadence extrapolated
    forward, falling back to 7 days when a plant has a single stage.
    """
    cur = stages[i].start_date
    if i + 1 < len(stages):
        return stages[i + 1].start_date
    if i > 0:
        return cur + (cur - stages[i - 1].start_date)
    return cur + timedelta(days=7)


def _subtract_months(d: date, months: int) -> date:
    """``d`` shifted back ``months`` whole calendar months, day-preserving.

    The day is clamped to the target month's length (e.g. 31 Mar − 1 month →
    28/29 Feb), so the result is always a valid date.
    """
    total = d.year * 12 + (d.month - 1) - months
    year, month = divmod(total, 12)
    month += 1
    first_of_next = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    last_day = (first_of_next - timedelta(days=1)).day
    return date(year, month, min(d.day, last_day))


def _lead_time_hours(
    anchor_end: date,
    lag_months: int,
    horizon_start: date,
    stage_spans: Sequence[tuple[date, date]],
    cumulative_hours: Sequence[float],
) -> tuple[float, bool]:
    """Physical ``lead_time_hours`` (``H``) for an anticipated GNL plant.

    cobre resolves an anticipated commitment's in-study decider **end-anchored**:
    ``decider`` = the operative stage containing ``window_end_hours − H`` on the
    cumulative operative-hours clock, with a boundary tie resolving to the
    earlier stage (``lead_time/mod.rs``). To land the decider on the source
    model's *decision stage* — the operative stage ``lag_months`` before the
    delivery — ``H`` is set so ``window_end_hours − H`` equals the **end**
    boundary of that stage (the tie then resolves onto it)::

        H = window_end_hours − cumulative_hours[decision_stage + 1]

    ``window_end_hours`` is the wall-clock hours from ``horizon_start`` to
    ``anchor_end`` (matching cobre's ``hours_between``) and ``cumulative_hours``
    are the cumulative operative-stage boundaries (matching cobre's
    ``study_stage_durations``), so ``window_end_hours − H`` reproduces cobre's
    boundary exactly. Returns ``(H, decided_pre_study)``; ``decided_pre_study``
    is ``True`` when the decision date precedes ``horizon_start`` (decided before
    the study — out of the in-study lead's reach), in which case ``H`` is
    anchored at the first stage so the value stays a valid physical lead.
    """
    window_end_hours = (anchor_end - horizon_start).days * 24.0
    decision_date = _subtract_months(anchor_end, lag_months)
    if decision_date < horizon_start:
        return window_end_hours - cumulative_hours[1], True
    decision_stage = len(stage_spans) - 1
    for m, (s_start, s_end) in enumerate(stage_spans):
        if s_start <= decision_date < s_end:
            decision_stage = m
            break
    return window_end_hours - cumulative_hours[decision_stage + 1], False


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


def _anticipation_lead_hours(
    thermal: GnlThermal,
    commitment: GnlCommitment,
    lag_months: int | None,
    footprint_stages: int,
    post_horizon: Sequence[int],
    horizon_start: date,
    stage_spans: Sequence[tuple[date, date]],
    stage_hours: Sequence[float],
    cumulative_hours: Sequence[float],
) -> float:
    """A GNL plant's ``anticipated_config.lead_time_hours`` (physical ``H``).

    A plant with a **post-horizon** committed delivery is the right-boundary
    case: its lead is the physical dispatch-anticipation span implied by the
    ``nl`` lag (:func:`_lead_time_hours`), anchored on the earliest post-horizon
    delivery whose ``nl``-implied decision still lands in-study — this lead may
    exceed the study horizon (the delivery is post-horizon), which is exactly
    what places the decider ``lag`` months back. (cobre-io's semantic validator
    must exempt a plant with ``future_anticipated_deliveries`` from its
    ``lead_time <= horizon`` check for such a case to validate — see the
    right-boundary spec §4.3; a purely-in-study lead below stays horizon-bounded.)

    A plant with **no** post-horizon delivery (purely in-study, or an inert
    all-zero registry plant) keeps the committed-footprint lead (the leading
    ``footprint_stages`` stages' cumulative hours) — a horizon-bounded value the
    in-study ring already validates. A plant carrying a post-horizon delivery but
    no ``nl`` lag (the source model normally declares one for every GNL plant),
    or whose every post-horizon decision predates the study, likewise falls back
    to the footprint lead.
    """
    footprint = sum(stage_hours[:footprint_stages])
    if not post_horizon:
        return footprint
    if lag_months is None:
        _LOG.warning(
            "GNL %s: post-horizon delivery but no nl dispatch-anticipation lag; "
            "falling back to the committed-footprint lead",
            thermal.name,
        )
        return footprint
    for i in post_horizon:
        lead_hours, decided_pre_study = _lead_time_hours(
            _delivery_window_end(commitment.stages, i),
            lag_months,
            horizon_start,
            stage_spans,
            cumulative_hours,
        )
        if not decided_pre_study:
            return lead_hours
    return footprint


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
    where ``H`` is the plant's **physical dispatch-anticipation lead** derived
    from its ``nl`` lag (:func:`_anticipation_lead_hours` /
    :func:`_lead_time_hours`) — the decision→delivery span that lands cobre's
    end-anchored in-study decider on the operative stage ``lag`` months before
    the delivery. (This single ``H`` drives *both* cobre roles: the in-study ring
    depth and each post-horizon delivery's decider.) Every anticipated thermal
    then gets:

    * ``past_anticipated_commitments`` tiling study stages ``[0, K)`` (``K`` from
      :func:`_lead_stage_count`) with the hours-weighted committed MW folded from
      the (weekly) ``gl`` deliveries onto each study stage (explicit ``0`` where
      none) — the mandatory left boundary;
    * ``future_anticipated_deliveries`` for each delivery landing on/after the
      study-horizon end, pinned ``min_mw == max_mw == committed_mw`` over
      ``[start, start + stage span)`` — the right boundary. A delivery whose
      ``nl``-implied decision predates the study is skipped with a warning (its
      pre-study left-boundary treatment is deferred), never emitted as a window
      cobre would silently drop.

    All post-horizon deliveries share one ``post_study_stages`` calendar, split
    at every delivery boundary so each delivery covers whole stages (cobre
    requires exact 1.0 coverage); each delivery's covered stage gets a
    ``thermal_bound`` carrying the plant's ``cvu`` as ``cost_per_mwh``
    (fuel-inclusive) and its ``[min_mw, max_mw]`` capability.

    ``stages`` is the converted ``stages.json`` stage list (each a mapping with
    ``start_date``, ``end_date``, and ``blocks[].hours``).
    """
    horizon_start = date.fromisoformat(stages[0]["start_date"])
    horizon_end = date.fromisoformat(stages[-1]["end_date"])
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

    gnl_id = {t.code: first_thermal_id + i for i, t in enumerate(model.thermals)}

    thermals: list[dict] = []
    past: list[dict] = []
    future: list[dict] = []

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

        footprint_stages = _lead_stage_count(folded)
        post_horizon = [
            i
            for i, c in enumerate(commitment.stages)
            if c.start_date >= horizon_end and abs(c.committed_mw) > _NONZERO_TOLERANCE
        ]
        lag_months = model.nl_lag_months.get(thermal.code)
        lead_hours = _anticipation_lead_hours(
            thermal,
            commitment,
            lag_months,
            footprint_stages,
            post_horizon,
            horizon_start,
            stage_spans,
            stage_hours,
            cumulative_hours,
        )
        # The left boundary tiles exactly the leading stages cobre derives from
        # ``H`` (:func:`_lead_delivery_stage_count`), so an NL-lag lead that
        # reaches past the horizon still lands a coherent past-commitment tiling.
        tile_k = _lead_delivery_stage_count(lead_hours, cumulative_hours)
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
        for i in post_horizon:
            c = commitment.stages[i]
            delivery_end = _delivery_window_end(commitment.stages, i)
            if lag_months is not None:
                _, decided_pre_study = _lead_time_hours(
                    delivery_end,
                    lag_months,
                    horizon_start,
                    stage_spans,
                    cumulative_hours,
                )
                if decided_pre_study:
                    _LOG.warning(
                        "GNL %s: post-horizon delivery %s was decided before the "
                        "study horizon (nl lag %d months); its pre-study "
                        "(left-boundary) treatment is deferred, so it is not "
                        "emitted as an in-study-decided future delivery",
                        thermal.name,
                        c.start_date.isoformat(),
                        lag_months,
                    )
                    continue
            committed = _clamp_committed(
                c.committed_mw, thermal, f"delivery {c.start_date.isoformat()}"
            )
            future.append(
                {
                    "thermal_id": tid,
                    "delivery_start": c.start_date.isoformat(),
                    "delivery_end": delivery_end.isoformat(),
                    "min_mw": committed,
                    "max_mw": committed,
                }
            )

    past.sort(key=lambda w: (w["thermal_id"], w["start_date"]))
    future.sort(key=lambda d: (d["thermal_id"], d["delivery_start"]))
    post_study = _build_post_study_stages(future, model, gnl_id, horizon_end)

    return GnlEmission(
        thermals=thermals,
        past_anticipated_commitments=past,
        future_anticipated_deliveries=future,
        post_study_stages=post_study,
    )


def _build_post_study_stages(
    future: Sequence[dict],
    model: GnlCommitmentModel,
    gnl_id: dict[int, int],
    horizon_end: date,
) -> dict | None:
    """Post-study calendar split at every delivery boundary + per-cell bounds.

    Returns ``None`` when no delivery lands post-horizon. The stages tile
    contiguously from ``horizon_end``; breaking at each delivery ``start``/``end``
    guarantees every ``future_anticipated_deliveries`` window covers whole stages
    (cobre's exact-coverage rule). Each delivery's covered stage gets a
    ``thermal_bound`` with the plant's ``cvu`` (fuel-inclusive) and capability.
    """
    if not future:
        return None
    code_of = {tid: code for code, tid in gnl_id.items()}
    thermal_of = {t.code: t for t in model.thermals}
    breakpoints = sorted(
        {horizon_end}
        | {date.fromisoformat(d["delivery_start"]) for d in future}
        | {date.fromisoformat(d["delivery_end"]) for d in future}
    )
    stages = [
        {
            "start_date": breakpoints[i].isoformat(),
            "duration_hours": (breakpoints[i + 1] - breakpoints[i]).days * 24.0,
        }
        for i in range(len(breakpoints) - 1)
    ]
    bounds: list[dict] = []
    for d in future:
        thermal = thermal_of[code_of[d["thermal_id"]]]
        ds = date.fromisoformat(d["delivery_start"])
        de = date.fromisoformat(d["delivery_end"])
        for idx in range(len(stages)):
            if breakpoints[idx] >= ds and breakpoints[idx + 1] <= de:
                bounds.append(
                    {
                        "thermal_id": d["thermal_id"],
                        "post_study_stage_index": idx,
                        "cost_per_mwh": thermal.cost_per_mwh,
                        "min_mw": thermal.min_mw,
                        "max_mw": thermal.max_mw,
                    }
                )
    return {"stages": stages, "thermal_bounds": bounds}
