"""Stochastic data converter: maps NEWAVE inflow and load data to Cobre Parquet.

Converts ``vazoes.dat`` (historical inflow series), ``vazpast.dat`` (recent
past inflows), ``sistema.dat`` (load demand), and ``patamar.dat`` (load block
factors) into PyArrow Tables and dicts that are written as Parquet files and
JSON files in the ``scenarios/`` directory of a Cobre case.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Confhd, Dger, Patamar, Sistema, Vazoes

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

logger = logging.getLogger(__name__)


def _build_upstream_postos(confhd_df: pd.DataFrame) -> dict[int, list[int]]:
    """Return {posto: [upstream_posto, ...]} for the hydro cascade.

    A plant's upstream plants are those whose ``codigo_usina_jusante``
    equals this plant's ``codigo_usina``.  The mapping is in posto space
    since inflow data (vazoes/vazpast) is indexed by posto.
    """
    from collections import defaultdict

    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]

    code_to_posto: dict[int, int] = {}
    for _, row in non_fict.iterrows():
        code_to_posto[int(row["codigo_usina"])] = int(row["posto"])

    upstream: dict[int, list[int]] = defaultdict(list)
    for _, row in non_fict.iterrows():
        code = int(row["codigo_usina"])
        ds_raw = row.get("codigo_usina_jusante")
        if ds_raw is not None and not pd.isna(ds_raw) and int(ds_raw) != 0:
            ds_code = int(ds_raw)
            ds_posto = code_to_posto.get(ds_code)
            if ds_posto is not None:
                upstream[ds_posto].append(code_to_posto[code])
    return dict(upstream)


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


def convert_recent_inflow_lags(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> list[dict]:
    """Extract 12 recent inflow lags from vazpast.dat for initial_conditions.json.

    Returns a list of ``{"hydro_id": int, "values_m3s": [lag1, ..., lag12]}``
    entries conforming to the ``past_inflows`` field of the Cobre
    ``initial_conditions.json`` schema.  ``values_m3s[0]`` is the most recent
    lag (month immediately before study start), ``values_m3s[11]`` is the
    oldest.

    Returns an empty list if ``vazpast.dat`` is absent.
    """
    from inewave.newave import Vazpast

    if nw_files.vazpast is None:
        logger.debug("vazpast.dat not found; no recent inflow lags.")
        return []

    try:
        vazpast_obj = Vazpast.read(nw_files.vazpast)
    except Exception:  # noqa: BLE001
        logger.warning("vazpast.dat could not be parsed; skipping recent lags.")
        return []

    df_tend: pd.DataFrame | None = vazpast_obj.tendencia
    if df_tend is None or df_tend.empty:
        return []

    dger = Dger.read(nw_files.dger)
    start_m = dger.mes_inicio_estudo

    # The vazpast "codigo_usina" column is actually the posto (gauging station),
    # same convention as vazoes.dat.  Map posto -> hydro_code -> cobre_id.
    confhd_obj = Confhd.read(nw_files.confhd)
    confhd_df = confhd_obj.usinas
    posto_to_cobre_id: dict[int, int] = {}
    for _, row in confhd_df.iterrows():
        code = int(row["codigo_usina"])
        posto = int(row["posto"])
        try:
            posto_to_cobre_id[posto] = id_map.hydro_id(code)
        except KeyError:
            pass

    # Lag order: lag 1 = month before study start, ..., lag 12 = 12 months back.
    lag_cal_months = [((start_m - 1 - i) % 12) + 1 for i in range(1, 13)]

    # Collect natural inflow values by posto: {posto: {cal_month: value}}.
    natural: dict[int, dict[int, float]] = {}
    for posto, plant_df in df_tend.groupby("codigo_usina"):
        posto = int(posto)
        if posto not in posto_to_cobre_id:
            continue
        month_vals: dict[int, float] = {}
        for _, row in plant_df.iterrows():
            month_vals[int(row["mes"])] = float(row["valor"])
        natural[posto] = month_vals

    # Convert natural → incremental: subtract each upstream posto's values.
    upstream_map = _build_upstream_postos(confhd_df)

    result: list[dict] = []
    for posto, nat_vals in natural.items():
        inc_vals: dict[int, float] = dict(nat_vals)
        for up_posto in upstream_map.get(posto, []):
            up_nat = natural.get(up_posto, {})
            for m in inc_vals:
                inc_vals[m] -= up_nat.get(m, 0.0)

        values_m3s = [max(0.0, inc_vals.get(m, 0.0)) for m in lag_cal_months]
        result.append({"hydro_id": posto_to_cobre_id[posto], "values_m3s": values_m3s})

    result.sort(key=lambda e: e["hydro_id"])
    return result


def convert_inflow_history(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert the full historical inflow series from vazoes.dat to Parquet.

    Reads ``vazoes.dat`` and produces one row per (hydro, month) pair,
    starting from January of ``ano_inicial_historico`` (from ``dger.dat``).

    Returns
    -------
    pa.Table
        Columns: ``hydro_id`` (INT32), ``date`` (DATE32), ``value_m3s`` (DOUBLE).
    """
    vazoes_obj = Vazoes.read(nw_files.vazoes)
    df_vazoes: pd.DataFrame | None = vazoes_obj.vazoes
    if df_vazoes is None or df_vazoes.empty:
        raise FileNotFoundError("vazoes.dat not found or empty")

    dger = Dger.read(nw_files.dger)
    hist_start_year: int = dger.ano_inicial_historico

    confhd_obj = Confhd.read(nw_files.confhd)
    confhd_df = confhd_obj.usinas

    posto_to_hydro: dict[int, int] = {}
    for _, row in confhd_df.iterrows():
        code = int(row["codigo_usina"])
        posto = int(row["posto"])
        try:
            posto_to_hydro[posto] = id_map.hydro_id(code)
        except KeyError:
            pass

    # Read natural inflow arrays by posto.
    n_rows = len(df_vazoes)
    natural_by_posto: dict[int, np.ndarray] = {}
    for col in df_vazoes.columns:
        if col == "data":
            continue
        try:
            posto = int(col)
        except (ValueError, TypeError):
            continue
        if posto in posto_to_hydro:
            natural_by_posto[posto] = df_vazoes[col].to_numpy(dtype=float)

    # Convert natural → incremental: subtract upstream postos' series.
    upstream_map = _build_upstream_postos(confhd_df)
    incremental_by_posto: dict[int, np.ndarray] = {}
    for posto, nat in natural_by_posto.items():
        inc = nat.copy()
        for up_posto in upstream_map.get(posto, []):
            up_nat = natural_by_posto.get(up_posto)
            if up_nat is not None:
                inc = inc - up_nat
        incremental_by_posto[posto] = inc

    rows_hydro_id: list[int] = []
    rows_date: list[date] = []
    rows_value: list[float] = []

    for posto, values in incremental_by_posto.items():
        cobre_id = posto_to_hydro[posto]
        for i in range(n_rows):
            y = hist_start_year + (i // 12)
            m = (i % 12) + 1
            rows_hydro_id.append(cobre_id)
            rows_date.append(date(y, m, 1))
            rows_value.append(float(values[i]))

    return pa.table(
        {
            "hydro_id": pa.array(rows_hydro_id, type=pa.int32()),
            "date": pa.array(rows_date, type=pa.date32()),
            "value_m3s": pa.array(rows_value, type=pa.float64()),
        }
    )


def convert_inflow_stats(nw_files: NewaveFiles, id_map: NewaveIdMap) -> pa.Table:
    """Convert NEWAVE historical inflow data to Cobre inflow seasonal statistics.

    For each hydro plant and each study stage (calendar month), computes the
    mean and standard deviation of historical monthly inflows across all
    available years in ``vazoes.dat``.  The station-to-plant mapping comes
    from ``confhd.dat``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
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
        If ``vazoes.dat`` DataFrame is empty.
    """
    vazoes_obj = Vazoes.read(nw_files.vazoes)
    df_vazoes: pd.DataFrame | None = vazoes_obj.vazoes

    if df_vazoes is None or df_vazoes.empty:
        raise FileNotFoundError("vazoes.dat not found or empty")

    confhd_obj = Confhd.read(nw_files.confhd)
    confhd_df: pd.DataFrame = confhd_obj.usinas

    dger = Dger.read(nw_files.dger)

    n_rows = len(df_vazoes)
    if "data" in df_vazoes.columns:
        cal_months: np.ndarray = df_vazoes["data"].dt.month.to_numpy()
    else:
        cal_months = np.array([(i % 12) + 1 for i in range(n_rows)])

    # Build posto -> hydro_code mapping and read natural series.
    posto_for_hydro: dict[int, int] = {}
    for _, row in confhd_df.iterrows():
        code = int(row["codigo_usina"])
        posto = int(row["posto"])
        posto_for_hydro[code] = posto

    natural_by_posto: dict[int, np.ndarray] = {}
    for col in df_vazoes.columns:
        if col == "data":
            continue
        try:
            posto = int(col)
        except (ValueError, TypeError):
            continue
        natural_by_posto[posto] = df_vazoes[col].to_numpy(dtype=float)

    # Convert natural → incremental.
    upstream_map = _build_upstream_postos(confhd_df)
    incremental_by_posto: dict[int, np.ndarray] = {}
    for posto, nat in natural_by_posto.items():
        inc = nat.copy()
        for up_posto in upstream_map.get(posto, []):
            up_nat = natural_by_posto.get(up_posto)
            if up_nat is not None:
                inc = inc - up_nat
        incremental_by_posto[posto] = inc

    hydro_codes = id_map.all_hydro_codes
    study_months = _derive_study_stage_months(dger)

    rows_hydro_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    for hydro_code in hydro_codes:
        cobre_hydro_id = id_map.hydro_id(hydro_code)
        posto = posto_for_hydro.get(hydro_code)
        inc_series = incremental_by_posto.get(posto) if posto is not None else None

        seasonal_mean: dict[int, float] = {}
        seasonal_std: dict[int, float] = {}

        for cal_month in range(1, 13):
            mask = cal_months == cal_month
            if inc_series is not None and mask.any():
                vals = inc_series[mask]
                seasonal_mean[cal_month] = float(np.nanmean(vals))
                seasonal_std[cal_month] = float(np.nanstd(vals, ddof=0))
            else:
                seasonal_mean[cal_month] = 0.0
                seasonal_std[cal_month] = 0.0

        for stage_id, cm in enumerate(study_months):
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


def convert_load_factors(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict:
    """Convert NEWAVE patamar load factors to a Cobre ``load_factors.json`` dict.

    Reads ``patamar.dat::carga_patamares`` and ``dger.dat`` to produce one
    entry per (bus, stage) pair containing a ``block_factors`` list with one
    entry per patamar.

    For post-study stages (beyond the study horizon), the last available
    year's factors are repeated seasonally (calendar-month match).

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map.  Used to resolve NEWAVE subsystem codes to 0-based
        Cobre bus IDs.

    Returns
    -------
    dict
        JSON-serializable dict with key ``"load_factors"`` containing a list
        of ``{"bus_id": int, "stage_id": int, "block_factors": [...]}``
        entries.
    """
    patamar = Patamar.read(nw_files.patamar)
    df_carga: pd.DataFrame | None = patamar.carga_patamares

    dger = Dger.read(nw_files.dger)
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    # Study end boundary: first month *after* the study horizon.
    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    if df_carga is None or df_carga.empty:
        logger.warning(
            "patamar.dat has no carga_patamares data; load_factors.json will be empty."
        )
        return {"load_factors": []}

    # Columns: codigo_submercado, data (datetime), patamar (1-based), valor (float)
    # Build lookup: {(subsystem_code, year, cal_month, patamar) -> factor}
    # Also track {(subsystem_code, cal_month, patamar) -> factor} for post-study.
    study_lookup: dict[tuple[int, int, int, int], float] = {}

    for _, row in df_carga.iterrows():
        sub = int(row["codigo_submercado"])
        dt = row["data"]
        cal_month = int(dt.month)
        yr = int(dt.year)
        pat = int(row["patamar"])
        val = float(row["valor"])
        study_lookup[(sub, yr, cal_month, pat)] = val

    # Build last-year lookup: for each (sub, cal_month, patamar), take the value
    # from the last study year that has data for that calendar month.
    # Group by (sub, cal_month, pat) and pick the max year.
    last_year_per_key: dict[tuple[int, int, int], tuple[int, float]] = {}
    for (sub, yr, cal_month, pat), val in study_lookup.items():
        key = (sub, cal_month, pat)
        existing = last_year_per_key.get(key)
        if existing is None or yr > existing[0]:
            last_year_per_key[key] = (yr, val)

    last_year_lookup = {k: v for k, (_, v) in last_year_per_key.items()}

    # Determine number of patamares and unique subsystem codes.
    num_patamares: int = patamar.numero_patamares or 1
    subsystem_codes = sorted(df_carga["codigo_submercado"].unique())

    load_factors: list[dict] = []

    for sub_code in subsystem_codes:
        try:
            bus_id = id_map.bus_id(int(sub_code))
        except KeyError:
            logger.warning(
                "Subsystem code %d from carga_patamares not in id_map; skipping",
                sub_code,
            )
            continue

        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            block_factors: list[dict] = []
            for pat_idx in range(1, num_patamares + 1):
                if is_post_study:
                    factor = last_year_lookup.get((int(sub_code), m, pat_idx))
                    if factor is None:
                        factor = 1.0
                else:
                    factor = study_lookup.get((int(sub_code), y, m, pat_idx))
                    if factor is None:
                        factor = 1.0

                block_factors.append({"block_id": pat_idx - 1, "factor": factor})

            load_factors.append(
                {
                    "bus_id": bus_id,
                    "stage_id": stage_id,
                    "block_factors": block_factors,
                }
            )

            m += 1
            if m > 12:
                m = 1
                y += 1

    return {"load_factors": load_factors}


def _derive_study_stage_months(dger: object) -> list[int]:
    """Return the ordered sequence of calendar months (1-12) for each study stage.

    Parameters
    ----------
    dger:
        A ``Dger`` object with ``mes_inicio_estudo``, ``ano_inicio_estudo``,
        and ``num_anos_estudo`` attributes.

    Returns
    -------
    list[int]
        Calendar month (1-12) for each stage, length = num_anos_estudo * 12.
    """
    start_month: int = dger.mes_inicio_estudo  # type: ignore[attr-defined]
    num_anos: int = dger.num_anos_estudo or 1  # type: ignore[attr-defined]
    num_anos_pos: int = dger.num_anos_pos_estudo or 0  # type: ignore[attr-defined]
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12
    return [((start_month - 1 + i) % 12) + 1 for i in range(total_stages)]


def convert_load_stats(nw_files: NewaveFiles, id_map: NewaveIdMap) -> pa.Table:
    """Convert NEWAVE subsystem load data to Cobre load seasonal statistics.

    Reads ``sistema.dat`` and converts the ``mercado_energia`` DataFrame
    (load demand per subsystem per month) into a PyArrow Table.  NEWAVE
    load is deterministic; all ``std_mw`` values are 0.0.

    The ``mercado_energia`` values from inewave are in average MW (the
    inewave library already converts from MWmonth to average MW), so no
    unit conversion is applied.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map.  Used to resolve NEWAVE subsystem codes to 0-based
        Cobre bus IDs.

    Returns
    -------
    pyarrow.Table
        Table with columns ``bus_id`` (int32), ``stage_id`` (int32),
        ``mean_mw`` (float64), ``std_mw`` (float64).  One row per
        (bus, stage) pair.
    """
    sistema_obj = Sistema.read(nw_files.sistema)
    df_load: pd.DataFrame = sistema_obj.mercado_energia

    dger = Dger.read(nw_files.dger)
    start_month = dger.mes_inicio_estudo
    start_year = dger.ano_inicio_estudo
    num_anos = dger.num_anos_estudo or 1
    num_anos_pos = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    rows_bus_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    for subsystem_code, group in df_load.groupby("codigo_submercado", sort=True):
        try:
            bus_id = id_map.bus_id(int(subsystem_code))
        except KeyError:
            logger.warning(
                "Subsystem code %d from mercado_energia not in id_map; skipping",
                subsystem_code,
            )
            continue

        # Build a month-keyed lookup: study-period months by (year, month),
        # post-study months (year=9999) by calendar month for seasonal repeat.
        study_values: dict[tuple[int, int], float] = {}
        pos_values: dict[int, float] = {}  # cal_month -> value
        for _, row in group.iterrows():
            dt = row["data"]
            val = row["valor"]
            if pd.isna(val):
                continue
            y, m = dt.year, dt.month
            if y == 9999:
                pos_values[m] = float(val)
            else:
                study_values[(y, m)] = float(val)

        # Emit one row per stage.
        y, m = start_year, start_month
        study_end_year = start_year + num_anos
        study_end_month = start_month  # first month after study
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )
            if is_post_study:
                val = pos_values.get(m)
                if val is None:
                    # Fall back to same calendar month from last study year
                    val = study_values.get((study_end_year - 1, m), 0.0)
            else:
                val = study_values.get((y, m), 0.0)

            rows_bus_id.append(bus_id)
            rows_stage_id.append(stage_id)
            rows_mean.append(val)
            rows_std.append(0.0)

            m += 1
            if m > 12:
                m = 1
                y += 1

    # Emit zero-load rows for buses that exist in id_map but have no
    # mercado_energia data (e.g., fictitious subsystems like NOFICT1).
    emitted_buses = set(rows_bus_id)
    for bus_code in id_map.all_bus_ids:
        bus_id = id_map.bus_id(bus_code)
        if bus_id not in emitted_buses:
            for stage_id in range(total_stages):
                rows_bus_id.append(bus_id)
                rows_stage_id.append(stage_id)
                rows_mean.append(0.0)
                rows_std.append(0.0)

    return pa.table(
        {
            "bus_id": pa.array(rows_bus_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean_mw": pa.array(rows_mean, type=pa.float64()),
            "std_mw": pa.array(rows_std, type=pa.float64()),
        }
    )
