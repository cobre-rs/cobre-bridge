"""Stage-math and ζ (zeta) pure helpers for the dead-volume filling mapping.

This foundation module holds the calendar/horizon arithmetic shared by the
filling converters (epics 3-4) and the fill-rate helpers (ticket-003): mapping a
NEWAVE filling date to a Cobre 0-based stage index (:func:`stage_id`) and the
per-month hm³-per-m³/s weight (:func:`zeta`).

All functions are pure (no logging, no I/O) so they can be reused and tested in
isolation. The module deliberately depends on nothing under
``cobre_bridge.converters`` — the calendar-hours basis is re-implemented locally
in :func:`month_hours` rather than imported, so a foundation module never pulls a
converter into its import graph.
"""

from __future__ import annotations

import calendar
from collections import Counter
from collections.abc import Callable, Sequence


def month_hours(year: int, month: int) -> float:
    """Total number of hours in the given calendar month.

    Uses the same calendar-hours basis as
    ``cobre_bridge.converters.temporal._month_hours`` (the basis of record):
    ``calendar.monthrange(year, month)[1] * 24``. Re-implemented here to keep
    this foundation module free of any converter dependency.

    A *month* outside 1-12 propagates ``calendar.monthrange``'s ``ValueError``.
    """
    days_in_month = calendar.monthrange(year, month)[1]
    return float(days_in_month * 24)


def zeta(year: int, month: int) -> float:
    """ζ for the given calendar month: hm³ stored per unit of m³/s inflow.

    Converts the month's calendar hours (:func:`month_hours`) into a volume
    weight via ``× 3600 / 1e6`` (seconds per hour over m³→hm³). For October with
    31 days this is ``744 * 3600 / 1e6 == 2.6784`` (design §5).
    """
    return month_hours(year, month) * 3600.0 / 1e6


def stage_id(year: int, month: int, start_year: int, start_month: int) -> int:
    """Cobre 0-based stage index for a calendar month within the study horizon.

    Computes ``(year - start_year) * 12 + (month - start_month)`` so the study
    start (``start_year``/``start_month``) maps to stage ``0``. The result may be
    negative for dates before the study start; clamping is a caller concern
    (design §8) handled in epic 3, not here.
    """
    return (year - start_year) * 12 + (month - start_month)


def filling_min_rate_m3s(
    min_storage_hm3: float,
    volume_morto_pct: float,
    start_stage_id: int,
    entry_stage_id: int,
    stage_year_month: Callable[[int], tuple[int, int]],
) -> float:
    """Constant per-stage minimum accumulation floor (m³/s) for the filling window.

    Fills the reservoir from its seeded ``filling_storage`` volume up to
    ``min_storage_hm3`` linearly across the half-open filling window
    ``[start_stage_id, entry_stage_id)``::

        rate = remaining / Σ_{t ∈ [start, entry)} ζ_t

    where ``remaining = min_storage_hm3 × (1 − volume_morto_pct / 100)`` and ζ_t
    comes from :func:`zeta` (design §4.1). ``stage_year_month`` maps a stage id to
    its ``(year, month)``; the converter supplies a closure over the horizon so
    this module stays horizon-agnostic.

    The window is **half-open**: the target reaches ``min_storage`` at
    ``entry_stage_id − 1`` (design §4.1), so the last filling stage is
    ``entry_stage_id − 1``. For JURUENA (single-stage window, Oct 2024):
    ``2.93 / 2.6784 ≈ 1.09`` (design §5).

    Returns ``0.0`` for an empty window (``start_stage_id >= entry_stage_id``)
    rather than raising ``ZeroDivisionError`` — this is the "no Filling phase"
    signal, and the caller omits ``filling`` in that case (design §8). This helper
    trusts its inputs: out-of-range ``volume_morto_pct`` is clamped upstream by the
    IC converter (epic 5), not here.
    """
    remaining = min_storage_hm3 * (1.0 - volume_morto_pct / 100.0)
    denom = sum(
        zeta(*stage_year_month(t)) for t in range(start_stage_id, entry_stage_id)
    )
    if denom > 0.0:
        return remaining / denom
    return 0.0


def online_machines(
    unit_rows: Sequence[tuple[int, int]],
    entry_stage_id: int,
    query_stage_id: int,
) -> dict[int, int]:
    """Count online machines per ``conjunto`` at the queried operating stage.

    ``unit_rows`` is a sequence of ``(conjunto, unit_entry_stage_id)`` pairs, one
    per entering machine (the converter pre-maps each ``data_entrada_operacao`` to
    a stage id via :func:`stage_id`). Each unit's effective online stage is
    ``max(unit_entry_stage_id, entry_stage_id)`` — the clamp ensures no unit is
    online before the reservoir is filled, since water cannot be turbined before
    the fill completes (design §4.2, §8).

    The clamp is applied **before** the ``<= query_stage_id`` comparison. Returns
    ``{conjunto: count}`` counting units whose effective online stage is
    ``<= query_stage_id``; groups with zero online machines are omitted.
    """
    counts: Counter[int] = Counter(
        conjunto
        for conjunto, unit_entry_stage_id in unit_rows
        if max(unit_entry_stage_id, entry_stage_id) <= query_stage_id
    )
    return dict(counts)
