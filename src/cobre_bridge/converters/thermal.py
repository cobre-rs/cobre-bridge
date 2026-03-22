"""Thermal entity converter: maps NEWAVE thermal plant data to Cobre thermal JSON.

Also provides ``convert_thermal_bounds`` which builds a per-stage
``thermal_bounds.parquet`` from ``expt.dat`` (temporal capacity/factor/TEIF/
GTMIN/IPTER overrides) and ``manutt.dat`` (scheduled maintenance windows).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Clast, Conft, Term

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

# Parquet schema for per-stage thermal generation bounds.
_THERMAL_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("thermal_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("min_generation_mw", pa.float64()),
        pa.field("max_generation_mw", pa.float64()),
    ]
)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/thermals.schema.json"
)


def convert_thermals(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE thermal plant data to a Cobre ``thermals.json`` dict.

    Reads ``conft.dat``, ``clast.dat``, and ``term.dat`` from *nw_files*.
    Returns a dict with a ``"thermals"`` key containing a list of thermal
    entries sorted by Cobre 0-based ID.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built ID mapping for bus cross-references.
    """
    conft = Conft.read(str(nw_files.conft))
    clast = Clast.read(str(nw_files.clast))
    term = Term.read(str(nw_files.term))

    conft_df = conft.usinas
    clast_df = clast.usinas
    term_df = term.usinas

    # Build cost lookup: codigo_usina -> cost for indice_ano_estudo == 1.
    cost_map: dict[int, float] = {}
    if clast_df is not None:
        first_year = clast_df[clast_df["indice_ano_estudo"] == 1]
        for _, row in first_year.iterrows():
            cost_map[int(row["codigo_usina"])] = float(row["valor"])

    # Build term lookup: codigo_usina -> (capacity, max_factor, min_gen_month1).
    term_map: dict[int, dict[str, float]] = {}
    if term_df is not None:
        # Use the first month's geracao_minima (mes == 1).
        month1 = term_df[term_df["mes"] == 1]
        for _, row in month1.iterrows():
            code = int(row["codigo_usina"])
            cap = float(row["potencia_instalada"])
            max_factor = float(row["fator_capacidade_maximo"])
            gen_min = float(row["geracao_minima"])
            term_map[code] = {
                "capacity": cap,
                "max_factor": max_factor,
                "gen_min": gen_min,
            }

        # For plants that appear in term but not in month1, use any row.
        for _, row in term_df.iterrows():
            code = int(row["codigo_usina"])
            if code not in term_map:
                cap = float(row["potencia_instalada"])
                max_factor = float(row["fator_capacidade_maximo"])
                term_map[code] = {
                    "capacity": cap,
                    "max_factor": max_factor,
                    "gen_min": 0.0,
                }

    thermals: list[dict] = []
    for _, row in conft_df.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        submercado = int(row["submercado"])

        bus_id = id_map.bus_id(submercado)

        term_info = term_map.get(
            newave_code, {"capacity": 0.0, "max_factor": 1.0, "gen_min": 0.0}
        )
        capacity = term_info["capacity"]
        max_factor = term_info["max_factor"]
        gen_min = term_info["gen_min"]

        max_mw = capacity * max_factor / 100.0
        cost = cost_map.get(newave_code, 0.0)

        thermal_entry: dict = {
            "id": id_map.thermal_id(newave_code),
            "name": name,
            "bus_id": bus_id,
            "cost_segments": [
                {
                    "capacity_mw": max_mw,
                    "cost_per_mwh": cost,
                }
            ],
            "generation": {
                "min_mw": gen_min,
                "max_mw": max_mw,
            },
            "gnl_config": None,
            "entry_stage_id": None,
            "exit_stage_id": None,
        }
        thermals.append(thermal_entry)

    thermals.sort(key=lambda t: t["id"])

    return {
        "$schema": _SCHEMA_URL,
        "thermals": thermals,
    }


def _build_stage_dates(
    start_year: int, start_month: int, total_stages: int
) -> list[date]:
    """Return a list of first-of-month dates for each study stage."""
    stages: list[date] = []
    y, m = start_year, start_month
    for _ in range(total_stages):
        stages.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return stages


def _month_date_to_stage_index(
    stage_dates: list[date], target_year: int, target_month: int
) -> int | None:
    """Return the 0-based stage index for (year, month), or None if not in range."""
    target = date(target_year, target_month, 1)
    for i, d in enumerate(stage_dates):
        if d == target:
            return i
    return None


def _apply_maint_to_capacity(
    base_capacity: float,
    maint_rows: pd.DataFrame,
    stage_dates: list[date],
) -> np.ndarray:
    """Compute monthly effective capacity after subtracting maintenance windows.

    For each stage (month), builds a daily-resolution view of the month and
    subtracts ``potencia`` (MW) for each maintenance unit whose window
    overlaps that month.  Multiple units (different ``codigo_unidade``) can be
    under maintenance simultaneously and are treated additively.

    Parameters
    ----------
    base_capacity:
        Installed capacity in MW (sum across all units).
    maint_rows:
        DataFrame slice for one thermal plant with columns
        ``data_inicio`` (datetime), ``duracao`` (int, days), ``potencia`` (float).
    stage_dates:
        First-of-month dates for every study stage.

    Returns
    -------
    np.ndarray
        Shape (total_stages,), dtype float64.  Each element is the monthly
        average effective capacity after maintenance.
    """
    total_stages = len(stage_dates)
    effective = np.full(total_stages, base_capacity, dtype=float)

    for _, row in maint_rows.iterrows():
        start_dt = pd.Timestamp(row["data_inicio"])
        duration_days = int(row["duracao"])
        unit_power = float(row["potencia"])
        end_dt = start_dt + timedelta(days=duration_days)

        for stage_idx, stage_start in enumerate(stage_dates):
            import calendar as _cal

            _, days_in_month = _cal.monthrange(stage_start.year, stage_start.month)
            # First day of the following month (exclusive upper bound).
            if stage_start.month == 12:
                stage_end = date(stage_start.year + 1, 1, 1)
            else:
                stage_end = date(stage_start.year, stage_start.month + 1, 1)

            maint_start_date = start_dt.date()
            maint_end_date = end_dt.date()

            # Overlap of [maint_start, maint_end) with [stage_start, stage_end).
            overlap_start = max(maint_start_date, stage_start)
            overlap_end = min(maint_end_date, stage_end)
            overlap_days = (overlap_end - overlap_start).days
            if overlap_days <= 0:
                continue

            # Fraction of the month under maintenance for this unit.
            fraction = overlap_days / days_in_month
            effective[stage_idx] -= unit_power * fraction

    return effective


def convert_thermal_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build per-stage thermal generation bounds from EXPT.DAT and MANUTT.DAT.

    For each thermal plant and each study stage, computes:

    ``max_generation_mw = potencia_instalada * (fcmax/100)
                          * ((100 - ip) / 100) * ((100 - teif) / 100)``

    Base values come from ``term.dat``.  Temporal overrides from ``expt.dat``
    update ``potencia_instalada`` (POTEF), ``fator_capacidade_maximo`` (FCMAX),
    ``teif`` (TEIFT), ``geracao_minima`` (GTMIN), and
    ``indisponibilidade_programada`` (IPTER) for specific date ranges.

    ``manutt.dat`` maintenance events further reduce ``potencia_instalada`` on
    a daily basis (averaged to monthly) for the maintenance window.  Following
    the NEWAVE convention, the ``indisponibilidade_programada`` is zeroed for
    stages that fall within the maintenance window of any unit for a plant,
    because the scheduled unavailability is already captured by the maintenance
    power reduction.

    Returns ``None`` if neither ``expt.dat`` nor ``manutt.dat`` is present.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths.
    id_map:
        Entity ID map for resolving NEWAVE thermal codes to 0-based Cobre IDs.

    Returns
    -------
    pa.Table | None
        Parquet table with columns ``thermal_id``, ``stage_id``,
        ``min_generation_mw``, ``max_generation_mw``; or ``None`` if no
        per-stage override data is available.
    """
    if nw_files.expt is None and nw_files.manutt is None:
        _LOG.debug("Neither expt.dat nor manutt.dat present; skipping thermal bounds.")
        return None

    from inewave.newave import Dger, Expt, Manutt

    dger = Dger.read(str(nw_files.dger))
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    stage_dates = _build_stage_dates(start_year, start_month, total_stages)

    # ------------------------------------------------------------------
    # 1. Build base values per (thermal_code, calendar_month) from term.dat.
    #    term.dat has 12 or 13 rows per plant (one per calendar month,
    #    month 13 = annual average used as default).
    # ------------------------------------------------------------------
    term = Term.read(str(nw_files.term))
    term_df = term.usinas

    # Base per-(code, month): potencia, fcmax, teif, ip, gen_min.
    # Use month=1..12; fall back to any row if specific month missing.
    BaseRow = dict[str, float]
    base_by_code_month: dict[tuple[int, int], BaseRow] = {}
    if term_df is not None:
        for _, row in term_df.iterrows():
            code = int(row["codigo_usina"])
            mes = int(row["mes"])
            if mes < 1 or mes > 12:
                continue
            base_by_code_month[(code, mes)] = {
                "potencia": float(row["potencia_instalada"]),
                "fcmax": float(row["fator_capacidade_maximo"]),
                "teif": float(row.get("teif", 0.0)),
                "ip": float(row.get("indisponibilidade_programada", 0.0)),
                "gen_min": float(row["geracao_minima"]),
            }

    # Also collect default base row per code (first available month).
    base_default: dict[int, BaseRow] = {}
    if term_df is not None:
        for _, row in term_df.iterrows():
            code = int(row["codigo_usina"])
            if code not in base_default:
                base_default[code] = {
                    "potencia": float(row["potencia_instalada"]),
                    "fcmax": float(row["fator_capacidade_maximo"]),
                    "teif": float(row.get("teif", 0.0)),
                    "ip": float(row.get("indisponibilidade_programada", 0.0)),
                    "gen_min": float(row["geracao_minima"]),
                }

    def _base(code: int, cal_month: int) -> BaseRow:
        row = base_by_code_month.get((code, cal_month))
        if row is not None:
            return dict(row)
        default = base_default.get(code)
        if default is not None:
            return dict(default)
        return {"potencia": 0.0, "fcmax": 100.0, "teif": 0.0, "ip": 0.0, "gen_min": 0.0}

    # ------------------------------------------------------------------
    # 2. Load EXPT overrides: {thermal_code: list of override dicts}.
    # ------------------------------------------------------------------
    expt_by_code: dict[int, list[dict]] = {}
    if nw_files.expt is not None:
        try:
            expt_obj = Expt.read(str(nw_files.expt))
            expt_df = expt_obj.expansoes
            for _, row in expt_df.iterrows():
                code = int(row["codigo_usina"])
                expt_by_code.setdefault(code, []).append(
                    {
                        "tipo": str(row["tipo"]),
                        "modificacao": float(row["modificacao"]),
                        "data_inicio": row["data_inicio"],
                        "data_fim": row["data_fim"],
                    }
                )
        except Exception:  # noqa: BLE001
            _LOG.warning("expt.dat could not be parsed; EXPT overrides skipped.")

    # ------------------------------------------------------------------
    # 3. Load MANUTT maintenance events: {thermal_code: DataFrame slice}.
    # ------------------------------------------------------------------
    manutt_by_code: dict[int, pd.DataFrame] = {}
    if nw_files.manutt is not None:
        try:
            manutt_obj = Manutt.read(str(nw_files.manutt))
            manutt_df = manutt_obj.manutencoes
            for code, grp in manutt_df.groupby("codigo_usina"):
                manutt_by_code[int(code)] = grp.reset_index(drop=True)
        except Exception:  # noqa: BLE001
            _LOG.warning("manutt.dat could not be parsed; maintenance skipped.")

    # Determine the set of thermal codes that have any per-stage override data.
    all_codes = (
        set(expt_by_code.keys()) | set(manutt_by_code.keys()) | set(base_default.keys())
    )

    rows_thermal_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_min: list[float] = []
    rows_max: list[float] = []

    for newave_code in sorted(all_codes):
        try:
            thermal_id = id_map.thermal_id(newave_code)
        except KeyError:
            _LOG.debug("Thermal code %d not in id_map; skipping bounds.", newave_code)
            continue

        overrides = expt_by_code.get(newave_code, [])
        maint_rows = manutt_by_code.get(newave_code)

        # Determine if any maintenance exists for this plant to zero out IP.
        has_maint = maint_rows is not None and not maint_rows.empty
        maint_stage_flags: set[int] = set()
        if has_maint:
            for _, mrow in maint_rows.iterrows():
                start_dt = pd.Timestamp(mrow["data_inicio"]).date()
                end_dt = (
                    pd.Timestamp(mrow["data_inicio"])
                    + timedelta(days=int(mrow["duracao"]))
                ).date()
                for idx, sd in enumerate(stage_dates):
                    # First day of the following month (exclusive upper bound).
                    if sd.month == 12:
                        stage_end = date(sd.year + 1, 1, 1)
                    else:
                        stage_end = date(sd.year, sd.month + 1, 1)
                    if start_dt < stage_end and end_dt > sd:
                        maint_stage_flags.add(idx)

        # Build per-stage capacity array from MANUTT (starts from base potencia).
        base_cap_for_maint = base_default.get(newave_code, {}).get("potencia", 0.0)
        if has_maint:
            maint_effective_cap = _apply_maint_to_capacity(
                base_cap_for_maint, maint_rows, stage_dates
            )
        else:
            maint_effective_cap = None

        for stage_idx, stage_date in enumerate(stage_dates):
            cal_month = stage_date.month

            # Start with base values for this calendar month.
            vals = _base(newave_code, cal_month)
            potencia = vals["potencia"]
            fcmax = vals["fcmax"]
            teif = vals["teif"]
            ip = vals["ip"]
            gen_min = vals["gen_min"]

            # Zero out IP for stages with active maintenance (NEWAVE convention).
            if stage_idx in maint_stage_flags:
                ip = 0.0

            # Apply EXPT overrides in file order for this stage.
            for override in overrides:
                ov_start = pd.Timestamp(override["data_inicio"]).date()
                ov_end_raw = override["data_fim"]
                if pd.isna(ov_end_raw):
                    # NaT means open-ended: override applies to end of horizon.
                    ov_end = stage_dates[-1]
                else:
                    ov_end = pd.Timestamp(ov_end_raw).date()

                if not (ov_start <= stage_date <= ov_end):
                    continue

                tipo = override["tipo"]
                value = override["modificacao"]
                if tipo == "POTEF":
                    potencia = value
                elif tipo == "FCMAX":
                    fcmax = value
                elif tipo == "TEIFT":
                    teif = value
                elif tipo == "GTMIN":
                    gen_min = value
                elif tipo == "IPTER":
                    # Only apply IPTER if no active maintenance zeroed it out.
                    if stage_idx not in maint_stage_flags:
                        ip = value

            # Apply MANUTT: replace potencia with daily-averaged effective capacity.
            if maint_effective_cap is not None:
                potencia = float(maint_effective_cap[stage_idx])

            # Clamp to zero (maintenance or overrides can push below 0).
            potencia = max(0.0, potencia)

            # Compute final bounds.
            max_mw = (
                potencia
                * (fcmax / 100.0)
                * ((100.0 - ip) / 100.0)
                * ((100.0 - teif) / 100.0)
            )
            max_mw = max(0.0, max_mw)
            min_mw = max(0.0, min(gen_min, max_mw))

            rows_thermal_id.append(thermal_id)
            rows_stage_id.append(stage_idx)
            rows_min.append(min_mw)
            rows_max.append(max_mw)

    if not rows_thermal_id:
        return None

    return pa.table(
        {
            "thermal_id": pa.array(rows_thermal_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "min_generation_mw": pa.array(rows_min, type=pa.float64()),
            "max_generation_mw": pa.array(rows_max, type=pa.float64()),
        },
        schema=_THERMAL_BOUNDS_SCHEMA,
    )
