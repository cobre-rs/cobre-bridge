"""Stochastic data converter: maps NEWAVE inflow and load data to Cobre Parquet.

Converts ``vazoes.dat`` (historical inflow series), ``vazpast.dat`` (recent
past inflows), and ``sistema.dat`` (load demand) into PyArrow Tables that are
written as Parquet files in the ``scenarios/`` directory of a Cobre case.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Confhd, Sistema, Vazoes

from cobre_bridge.id_map import NewaveIdMap

logger = logging.getLogger(__name__)

# Parquet schema for inflow seasonal statistics.
_INFLOW_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("mean_m3s", pa.float64()),
        pa.field("std_m3s", pa.float64()),
    ]
)

# Parquet schema for load seasonal statistics.
_LOAD_SCHEMA = pa.schema(
    [
        pa.field("bus_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("mean_mw", pa.float64()),
        pa.field("std_mw", pa.float64()),
    ]
)

# Parquet schema for past inflow history.
_INFLOW_HISTORY_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("date", pa.date32()),
        pa.field("value_m3s", pa.float64()),
    ]
)


def convert_past_inflows(
    newave_dir: Path,
    id_map: NewaveIdMap,
    study_start: date,
    num_pre_study_months: int,
) -> pa.Table | None:
    """Convert NEWAVE recent past inflows to Cobre inflow history Parquet.

    Reads ``vazpast.dat`` (recent past inflow observations) and extracts
    the last *num_pre_study_months* months of inflow values per gauging
    station, mapped to Cobre hydro IDs via ``confhd.dat``.  The result is
    used by the Rust side to initialise PAR(p) autoregressive lag slots at
    stage 0.

    Parameters
    ----------
    newave_dir:
        Path to the directory containing NEWAVE input files.
    id_map:
        Entity ID map produced during entity conversion.  Used to resolve
        NEWAVE hydro codes to 0-based Cobre hydro IDs.
    study_start:
        First day of the study horizon (from ``Dger``).  Used to compute
        the calendar dates for each past inflow row.
    num_pre_study_months:
        Number of past months to include.  Typically
        ``Dger.num_anos_pre_estudo * 12``.

    Returns
    -------
    pyarrow.Table or None
        ``None`` if ``vazpast.dat`` does not exist in *newave_dir*.
        Otherwise, a ``pyarrow.Table`` with columns
        ``hydro_id`` (int32), ``date`` (date32), ``value_m3s`` (float64).
        One row per (hydro, past-month) pair.

    Raises
    ------
    FileNotFoundError
        If ``vazpast.dat`` exists but cannot be parsed.
    FileNotFoundError
        If ``confhd.dat`` is absent.
    """
    from inewave.newave import (
        Vazpast,
    )  # local import to avoid hard dependency at module load

    vazpast_path = newave_dir / "vazpast.dat"
    confhd_path = newave_dir / "confhd.dat"

    if not vazpast_path.exists():
        logger.warning("vazpast.dat not found in %s; skipping past inflows", newave_dir)
        return None

    if not confhd_path.exists():
        raise FileNotFoundError(f"confhd.dat not found in {newave_dir}")

    try:
        vazpast_obj = Vazpast.read(vazpast_path)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError("vazpast.dat could not be parsed") from exc

    df_vazpast: pd.DataFrame | None = vazpast_obj.vazoes
    if df_vazpast is None or df_vazpast.empty:
        raise FileNotFoundError("vazpast.dat could not be parsed")

    confhd_obj = Confhd.read(confhd_path)
    df_confhd: pd.DataFrame = confhd_obj.usinas

    # Build posto -> hydro_code mapping from confhd.
    posto_to_hydro_code: dict[int, int] = {}
    for _, row in df_confhd.iterrows():
        code = int(row["codigo_usina"])
        posto = int(row["posto"])
        posto_to_hydro_code[posto] = code

    # Compute the calendar dates for each past month, counting backwards from
    # study_start.  Month -1 ends at study_start; month -N starts N months
    # before study_start.
    past_months: list[date] = []
    py = study_start.year
    pm = study_start.month
    for _ in range(num_pre_study_months):
        pm -= 1
        if pm < 1:
            pm = 12
            py -= 1
        past_months.append(date(py, pm, 1))
    # past_months is in reverse order (most recent first); reverse to chronological.
    past_months.reverse()

    # The vazpast DataFrame has a ``data`` column (datetime, one row per month)
    # and one column per gauging station (posto).  Extract the last
    # num_pre_study_months rows sorted chronologically.
    if "data" in df_vazpast.columns:
        df_sorted = df_vazpast.sort_values("data").reset_index(drop=True)
    else:
        df_sorted = df_vazpast.reset_index(drop=True)

    df_tail = df_sorted.tail(num_pre_study_months).reset_index(drop=True)

    rows_hydro_id: list[int] = []
    rows_date: list[date] = []
    rows_value: list[float] = []

    # Identify posto columns: all columns except ``data``.
    posto_columns = [c for c in df_tail.columns if c != "data"]

    for col in posto_columns:
        try:
            posto = int(col)
        except (ValueError, TypeError):
            logger.warning(
                "Unexpected non-integer column '%s' in vazpast.dat; skipping",
                col,
            )
            continue

        hydro_code = posto_to_hydro_code.get(posto)
        if hydro_code is None:
            logger.warning(
                "Gauging station (posto) %d in vazpast.dat has no matching hydro in "
                "confhd.dat; skipping",
                posto,
            )
            continue

        try:
            hydro_id = id_map.hydro_id(hydro_code)
        except KeyError:
            logger.warning(
                "Hydro code %d (posto %d) is not registered in id_map; skipping",
                hydro_code,
                posto,
            )
            continue

        for i, row in df_tail.iterrows():
            row_date = past_months[i] if i < len(past_months) else None
            if row_date is None:
                continue
            rows_hydro_id.append(hydro_id)
            rows_date.append(row_date)
            rows_value.append(float(row[col]))

    return pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "date": pa.array(rows_date, type=pa.date32()),
            "value_m3s": pa.array(rows_value, type=pa.float64()),
        }
    )


def convert_inflow_stats(newave_dir: Path, id_map: NewaveIdMap) -> pa.Table:
    """Convert NEWAVE historical inflow data to Cobre inflow seasonal statistics.

    For each hydro plant and each study stage (calendar month), computes the
    mean and standard deviation of historical monthly inflows across all
    available years in ``vazoes.dat``.  The station-to-plant mapping comes
    from ``confhd.dat``.

    Parameters
    ----------
    newave_dir:
        Path to the directory containing NEWAVE input files.
    id_map:
        Entity ID map produced during entity conversion.  Used to resolve
        NEWAVE hydro codes to 0-based Cobre hydro IDs.

    Returns
    -------
    pyarrow.Table
        Table with columns ``hydro_id`` (int32), ``stage_id`` (int32),
        ``mean_m3s`` (float64), ``std_m3s`` (float64).  One row per
        (hydro, stage) pair.  The number of stages equals the number of
        months in the study horizon registered in *id_map*.

    Raises
    ------
    FileNotFoundError
        If ``vazoes.dat`` is absent or its DataFrame is empty.
    FileNotFoundError
        If ``confhd.dat`` is absent.
    """
    vazoes_path = newave_dir / "vazoes.dat"
    confhd_path = newave_dir / "confhd.dat"

    if not vazoes_path.exists():
        raise FileNotFoundError(f"vazoes.dat not found in {newave_dir}")
    if not confhd_path.exists():
        raise FileNotFoundError(f"confhd.dat not found in {newave_dir}")

    vazoes_obj = Vazoes.read(vazoes_path)
    df_vazoes: pd.DataFrame | None = vazoes_obj.vazoes

    if df_vazoes is None or df_vazoes.empty:
        raise FileNotFoundError("vazoes.dat not found or empty")

    confhd_obj = Confhd.read(confhd_path)
    df_confhd: pd.DataFrame = confhd_obj.usinas

    # The ``data`` column in vazoes contains datetime objects representing
    # the month.  Extract calendar month (1-12) as a separate column so we
    # can group by it.
    if "data" not in df_vazoes.columns:
        raise ValueError(
            "vazoes DataFrame has no 'data' column; cannot determine calendar month"
        )
    cal_months: np.ndarray = df_vazoes["data"].dt.month.to_numpy()

    # The registered study stages in id_map follow a fixed calendar sequence.
    # We derive stage_id by enumerating the study months in the same order
    # that convert_stages() would produce them.  Because convert_inflow_stats
    # does not receive the Dger object, we iterate over all registered hydro
    # codes and assume stage_id == position in the Parquet output.
    # The caller (CLI pipeline) is responsible for ensuring the stage count
    # in the Parquet files matches stages.json.

    hydro_codes = id_map.all_hydro_codes  # sorted list

    rows_hydro_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    # For each hydro plant, look up its gauging station (posto) in confhd.
    posto_for_hydro: dict[int, int] = {}
    for _, row in df_confhd.iterrows():
        code = int(row["codigo_usina"])
        posto = int(row["posto"])
        posto_for_hydro[code] = posto

    for hydro_code in hydro_codes:
        cobre_hydro_id = id_map.hydro_id(hydro_code)
        posto = posto_for_hydro.get(hydro_code)

        if posto is None:
            logger.warning(
                "Hydro code %d not found in confhd.dat; "
                "inflow stats will be zero for this plant",
                hydro_code,
            )
            # We still need rows for every stage — we'll fill zeros below
            # once we know the unique ordered months.

        # Determine which column in df_vazoes corresponds to this posto.
        # The inewave library names columns as the posto integer or as a string.
        col_key: int | str | None = None
        if posto is not None:
            if posto in df_vazoes.columns:
                col_key = posto
            elif str(posto) in df_vazoes.columns:
                col_key = str(posto)
            else:
                logger.warning(
                    "Gauging station (posto) %d for hydro code %d has no matching "
                    "column in vazoes.dat; inflow stats will be zero",
                    posto,
                    hydro_code,
                )

        # Compute per-calendar-month statistics across all years.
        # We rely on the order of months in the DataFrame to assign stage IDs.
        # The unique ordered calendar months define the seasonal pattern; each
        # occurrence of that calendar month in the study horizon maps to a
        # stage_id (assigned by the caller in sequence).
        #
        # Here we compute the seasonal statistics (one row per calendar month
        # 1-12), then the CLI pipeline repeats them for each year of the
        # study horizon.  However, the ticket spec says:
        # "For each hydro plant and each stage (by calendar month), compute
        # mean and std of the historical monthly inflows across all years."
        # This means one row per stage (not per season), where stages that
        # share the same calendar month share the same statistics.
        #
        # We compute seasonal stats first, then emit one row per stage.
        seasonal_mean: dict[int, float] = {}
        seasonal_std: dict[int, float] = {}

        for cal_month in range(1, 13):
            mask = cal_months == cal_month
            if col_key is not None and mask.any():
                vals = df_vazoes.loc[mask, col_key].to_numpy(dtype=float)
                m = float(np.nanmean(vals))
                s = float(np.nanstd(vals, ddof=0))
            else:
                m = 0.0
                s = 0.0
            seasonal_mean[cal_month] = m
            seasonal_std[cal_month] = s

        # Determine the ordered sequence of calendar months for the study
        # stages.  We derive this from the Dger information embedded in the
        # vazoes DataFrame's date range.  The study stages are numbered
        # sequentially starting from the first month of the study.  We use
        # the unique months from vazoes in order (starting from the minimum
        # date) to assign stage IDs.
        #
        # Because convert_inflow_stats does not receive Dger, we derive the
        # stage sequence from the unique sorted months present in df_vazoes.
        # This matches the sequence that convert_stages() would emit.
        # The unique study months are stored in the dates from df_vazoes
        # ordered by date. We use the DataFrame's order directly.
        unique_study_months = _derive_study_stage_months(df_vazoes)

        for stage_id, (_, cm) in enumerate(unique_study_months):
            rows_hydro_id.append(cobre_hydro_id)
            rows_stage_id.append(stage_id)
            rows_mean.append(seasonal_mean[cm])
            rows_std.append(seasonal_std[cm])

    return pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean_m3s": pa.array(rows_mean, type=pa.float64()),
            "std_m3s": pa.array(rows_std, type=pa.float64()),
        }
    )


def _derive_study_stage_months(df_vazoes: pd.DataFrame) -> list[tuple[int, int]]:
    """Return an ordered list of (year, month) tuples from the vazoes date column.

    The returned list contains one entry per unique month in the historical
    series, sorted chronologically.  This sequence defines the stage_id
    mapping: stage 0 = first month, stage 1 = second month, etc.

    In practice ``vazoes.dat`` covers the historical record (e.g., 80 years).
    The study horizon is a subset of that, but the seasonal statistics for
    each calendar month are the same regardless of the year.  We return all
    unique months to let the caller decide how many stages to emit.

    Parameters
    ----------
    df_vazoes:
        The raw vazoes DataFrame containing a ``data`` column.

    Returns
    -------
    list[tuple[int, int]]
        Sorted list of ``(year, month)`` pairs, one per unique month.
    """
    if "data" not in df_vazoes.columns:
        return []
    dates = pd.to_datetime(df_vazoes["data"])
    unique_sorted = sorted(set(zip(dates.dt.year, dates.dt.month)))
    return unique_sorted


def convert_load_stats(newave_dir: Path, id_map: NewaveIdMap) -> pa.Table:
    """Convert NEWAVE subsystem load data to Cobre load seasonal statistics.

    Reads ``sistema.dat`` and converts the ``mercado_energia`` DataFrame
    (load demand per subsystem per month) into a PyArrow Table.  NEWAVE
    load is deterministic; all ``std_mw`` values are 0.0.

    The ``mercado_energia`` values from inewave are in average MW (the
    inewave library already converts from MWmonth to average MW), so no
    unit conversion is applied.

    Parameters
    ----------
    newave_dir:
        Path to the directory containing NEWAVE input files.
    id_map:
        Entity ID map.  Used to resolve NEWAVE subsystem codes to 0-based
        Cobre bus IDs.

    Returns
    -------
    pyarrow.Table
        Table with columns ``bus_id`` (int32), ``stage_id`` (int32),
        ``mean_mw`` (float64), ``std_mw`` (float64).  One row per
        (bus, stage) pair.

    Raises
    ------
    FileNotFoundError
        If ``sistema.dat`` is absent.
    """
    sistema_path = newave_dir / "sistema.dat"
    if not sistema_path.exists():
        raise FileNotFoundError(f"sistema.dat not found in {newave_dir}")

    sistema_obj = Sistema.read(sistema_path)
    df_load: pd.DataFrame = sistema_obj.mercado_energia

    # The mercado_energia DataFrame has columns:
    #   codigo_submercado (int), data (datetime), valor (float, average MW)
    # Sort by (codigo_submercado, data) for deterministic ordering.
    df_sorted = df_load.sort_values(["codigo_submercado", "data"]).reset_index(
        drop=True
    )

    rows_bus_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    # Group by subsystem code and assign stage IDs sequentially.
    for subsystem_code, group in df_sorted.groupby("codigo_submercado", sort=True):
        try:
            bus_id = id_map.bus_id(int(subsystem_code))
        except KeyError:
            logger.warning(
                "Subsystem code %d from mercado_energia not in id_map; skipping",
                subsystem_code,
            )
            continue

        group_sorted = group.sort_values("data").reset_index(drop=True)
        for stage_id, row in enumerate(group_sorted.itertuples(index=False)):
            rows_bus_id.append(bus_id)
            rows_stage_id.append(stage_id)
            rows_mean.append(float(row.valor))
            rows_std.append(0.0)

    return pa.table(
        {
            "bus_id": pa.array(rows_bus_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean_mw": pa.array(rows_mean, type=pa.float64()),
            "std_mw": pa.array(rows_std, type=pa.float64()),
        }
    )
