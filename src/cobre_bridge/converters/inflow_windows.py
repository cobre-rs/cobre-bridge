"""Windowed inflow emission (Cobre >= 0.13 input shapes).

Builds the two windowed inflow inputs of the unified representation:

- the realized-inflow history Parquet
  (``{hydro_id, start_date, end_date, value_m3s}``, one row per hydro per
  calendar month of the historical record), and
- the conditioning windows for ``initial_conditions.recent_observations``
  (the source model's hydrological tendency, as the 12 calendar months
  preceding the study start).

Values are incremental m³/s, produced by the same posto-mapping and
upstream-subtraction helpers the point-dated emitters use. The pipeline
switches to these emitters when the Cobre dependency pin moves to the
windowed schema; until then the module is additive and unused by
``convert``.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import pyarrow as pa

from cobre_bridge.case import NewaveCase
from cobre_bridge.converters.stochastic import (
    _incremental_history,
    _vazpast_incremental,
)
from cobre_bridge.id_map import NewaveIdMap

# Parquet schema for the windowed past-inflow history (Cobre >= 0.13).
INFLOW_HISTORY_WINDOW_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("start_date", pa.date32()),
        pa.field("end_date", pa.date32()),
        pa.field("value_m3s", pa.float64()),
    ]
)


def month_window(year: int, month: int) -> tuple[date, date]:
    """Return the ``[start, end)`` calendar window of ``year``-``month``."""
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    return start, end


def previous_months(year: int, month: int, count: int) -> list[tuple[int, int]]:
    """Return the ``count`` calendar months before ``year``-``month``.

    Walks backward starting at the month immediately preceding the given
    one; the result is ordered oldest first.
    """
    months: list[tuple[int, int]] = []
    y, m = year, month
    for _ in range(count):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        months.append((y, m))
    months.reverse()
    return months


def format_observation_windows(
    rows: Iterable[tuple[int, date, date, float]],
) -> list[dict]:
    """Format ``(hydro_id, start, end, value_m3s)`` rows as JSON entries.

    Produces the ``recent_observations`` entry shape (ISO dates), preserving
    the caller's ordering.
    """
    return [
        {
            "hydro_id": hydro_id,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "value_m3s": float(value),
        }
        for hydro_id, start, end, value in rows
    ]


def convert_inflow_history_windows(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert the full historical inflow record to windowed Parquet rows.

    One row per (hydro, calendar month) from January of the record's first
    year through the month before the study start, with explicit
    ``[start_date, end_date)`` windows. Values are incremental m³/s.

    Raises
    ------
    FileNotFoundError
        If the historical-record DataFrame is absent or empty.
    """
    hist_start_year, n_rows, incremental = _incremental_history(case, id_map)

    rows_hydro_id: list[int] = []
    rows_start: list[date] = []
    rows_end: list[date] = []
    rows_value: list[float] = []

    for cobre_id in sorted(incremental):
        values = incremental[cobre_id]
        for i in range(n_rows):
            y = hist_start_year + (i // 12)
            m = (i % 12) + 1
            start, end = month_window(y, m)
            rows_hydro_id.append(cobre_id)
            rows_start.append(start)
            rows_end.append(end)
            rows_value.append(float(values[i]))

    return pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "start_date": pa.array(rows_start, type=pa.date32()),
            "end_date": pa.array(rows_end, type=pa.date32()),
            "value_m3s": pa.array(rows_value, type=pa.float64()),
        },
        schema=INFLOW_HISTORY_WINDOW_SCHEMA,
    )


def convert_recent_observation_windows(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> list[dict]:
    """Convert the hydrological tendency to conditioning windows.

    Emits, per hydro, the 12 calendar-month windows immediately preceding
    the study start (oldest first), shaped for the
    ``initial_conditions.recent_observations`` field. Values are
    incremental m³/s. A calendar month absent from the tendency data is
    omitted rather than zero-filled — downstream coverage validation
    surfaces genuine gaps, and a fabricated zero-inflow window would be
    worse than an absent one.

    Returns an empty list when the tendency file is absent or unreadable.
    """
    incremental = _vazpast_incremental(case, id_map)
    if not incremental:
        return []

    dger = case.dger
    months = previous_months(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 12)

    rows: list[tuple[int, date, date, float]] = []
    for hydro_id in sorted(incremental):
        month_values = incremental[hydro_id]
        for y, m in months:
            if m not in month_values:
                continue
            start, end = month_window(y, m)
            rows.append((hydro_id, start, end, month_values[m]))

    return format_observation_windows(rows)
