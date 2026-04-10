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

    Follows the sintetizador-newave processing order:

    1. Zero IP for ALL plants in stages before ``maintenance_end_date``
       (= ``ano_inicio_estudo + num_anos_manutencao_utes``).
    2. For plants with EXPT POTEF: zero ``potencia`` for stages >=
       ``maintenance_end_date`` (to be restored by EXPT in step 3).
    3. For plants with EXPT GTMIN: zero ``gen_min`` for stages >=
       ``maintenance_end_date`` (to be restored by EXPT in step 3).
    4. Apply ALL EXPT overrides (POTEF, FCMAX, TEIFT, GTMIN, IPTER).
    5. Apply MANUTT capacity reductions (only stages < maintenance_end).
    6. Evaluate: ``pot * (fcmax/100) * ((100-ip)/100) * ((100-teif)/100)``

    Returns ``None`` if neither ``expt.dat`` nor ``manutt.dat`` is present.
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
    num_maint_years: int = dger.num_anos_manutencao_utes or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    # Maintenance end: stages before this index have IP=0 globally.
    # Maintenance years are counted as full calendar years from the study
    # start year.  For a March 2026 start with 1 maintenance year, the
    # period covers March-December 2026 (10 stages), not 12.
    maint_end_stage = num_maint_years * 12 + (1 - start_month)

    stage_dates = _build_stage_dates(start_year, start_month, total_stages)

    # ------------------------------------------------------------------
    # 1. Build base values per (thermal_code, calendar_month) from term.
    # ------------------------------------------------------------------
    term = Term.read(str(nw_files.term))
    term_df = term.usinas

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

    base_default: dict[int, BaseRow] = {}
    if term_df is not None:
        for _, row in term_df.iterrows():
            code = int(row["codigo_usina"])
            if code not in base_default:
                base_default[code] = {
                    "potencia": float(row["potencia_instalada"]),
                    "fcmax": float(row["fator_capacidade_maximo"]),
                    "teif": float(row.get("teif", 0.0)),
                    "ip": float(
                        row.get(
                            "indisponibilidade_programada",
                            0.0,
                        )
                    ),
                    "gen_min": float(row["geracao_minima"]),
                }

    def _base(code: int, cal_month: int) -> BaseRow:
        row = base_by_code_month.get((code, cal_month))
        if row is not None:
            return dict(row)
        default = base_default.get(code)
        if default is not None:
            return dict(default)
        return {
            "potencia": 0.0,
            "fcmax": 100.0,
            "teif": 0.0,
            "ip": 0.0,
            "gen_min": 0.0,
        }

    # ------------------------------------------------------------------
    # 2. Load EXPT overrides.
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

    # Pre-compute which codes have POTEF / GTMIN in EXPT.
    codes_with_potef: set[int] = set()
    codes_with_gtmin: set[int] = set()
    # Plants whose POTEF has a finite end date — they are only
    # available during the POTEF window (zero capacity outside).
    potef_finite_end: dict[int, date] = {}
    for code, overrides in expt_by_code.items():
        for o in overrides:
            if o["tipo"] == "POTEF":
                codes_with_potef.add(code)
                if not pd.isna(o["data_fim"]):
                    end = pd.Timestamp(o["data_fim"]).date()
                    prev = potef_finite_end.get(code)
                    if prev is None or end > prev:
                        potef_finite_end[code] = end
            elif o["tipo"] == "GTMIN":
                codes_with_gtmin.add(code)

    # ------------------------------------------------------------------
    # 3. Load MANUTT maintenance events.
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
            continue

        overrides = expt_by_code.get(newave_code, [])
        maint_rows = manutt_by_code.get(newave_code)
        has_maint = maint_rows is not None and not maint_rows.empty

        # Build per-stage MANUTT reduction (delta from base).
        # Applied to EXPT-modified potencia, matching sintetizador
        # which applies EXPT before MANUTT.
        base_cap = base_default.get(newave_code, {}).get("potencia", 0.0)
        maint_reduction: np.ndarray | None = None
        if has_maint:
            effective = _apply_maint_to_capacity(base_cap, maint_rows, stage_dates)
            maint_reduction = np.maximum(0.0, base_cap - effective)

        for stage_idx, stage_date in enumerate(stage_dates):
            cal_month = stage_date.month
            vals = _base(newave_code, cal_month)
            potencia = vals["potencia"]
            fcmax = vals["fcmax"]
            teif = vals["teif"]
            ip = vals["ip"]
            gen_min = vals["gen_min"]

            # Step 1: global IP zeroing before maintenance_end.
            if stage_idx < maint_end_stage:
                ip = 0.0

            # Step 2: null potencia for stages >= maint_end
            # if this plant has POTEF overrides (EXPT will restore).
            if stage_idx >= maint_end_stage and newave_code in codes_with_potef:
                potencia = 0.0

            # Step 3: null gen_min for stages >= maint_end
            # if this plant has GTMIN overrides (EXPT will restore).
            if stage_idx >= maint_end_stage and newave_code in codes_with_gtmin:
                gen_min = 0.0

            # Step 4: apply EXPT overrides in file order.
            for override in overrides:
                ov_start = pd.Timestamp(override["data_inicio"]).date()
                ov_end_raw = override["data_fim"]
                if pd.isna(ov_end_raw):
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
                    ip = value

            # Step 4b: POTEF with finite end defines availability window.
            # After the POTEF range expires the plant has zero capacity.
            potef_end = potef_finite_end.get(newave_code)
            if potef_end is not None and stage_date > potef_end:
                potencia = 0.0
                gen_min = 0.0

            # Step 5: MANUTT subtracts from potencia (only before maint_end).
            if maint_reduction is not None and stage_idx < maint_end_stage:
                potencia -= float(maint_reduction[stage_idx])

            potencia = max(0.0, potencia)

            # Step 6: evaluate formula.
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
