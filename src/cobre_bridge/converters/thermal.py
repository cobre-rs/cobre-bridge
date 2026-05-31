"""Thermal entity converter: maps NEWAVE thermal plant data to Cobre thermal JSON.

Also provides ``convert_thermal_bounds`` which builds a per-stage
``thermal_bounds.parquet`` from ``expt.dat`` (temporal capacity/factor/TEIF/
GTMIN/IPTER overrides) and ``manutt.dat`` (scheduled maintenance windows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Clast, Conft, Term

from cobre_bridge.converters.anticipated import read_anticipated_dispatch
from cobre_bridge.horizon import build_stage_dates, study_horizon
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
        pa.field("cost_per_mwh", pa.float64()),
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

    # Anticipated dispatch (NEWAVE GNL) — gated by dger.despacho_antecipado_gnl.
    # Returns an empty dict when the flag is off, so non-GNL cases incur
    # zero cost beyond the dger read inside the helper.
    anticipated_by_code = read_anticipated_dispatch(nw_files)

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

        anticipated = anticipated_by_code.get(newave_code)
        anticipated_config = (
            {"lead_stages": anticipated.lead_stages} if anticipated else None
        )

        thermal_entry: dict = {
            "id": id_map.thermal_id(newave_code),
            "name": name,
            "bus_id": bus_id,
            "cost_per_mwh": cost,
            "generation": {
                "min_mw": gen_min,
                "max_mw": max_mw,
            },
            "anticipated_config": anticipated_config,
            "entry_stage_id": None,
            "exit_stage_id": None,
        }
        thermals.append(thermal_entry)

    thermals.sort(key=lambda t: t["id"])

    return {
        "$schema": _SCHEMA_URL,
        "thermals": thermals,
    }


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


def _stage_to_study_year(
    stage_idx: int,
    first_year_stages: int,
    num_anos: int,
) -> int:
    """Map a 0-based stage index to a 1-based ``indice_ano_estudo``.

    The first study year covers ``first_year_stages`` months (``13 -
    start_month``).  Subsequent years cover 12 months each.  Post-study
    stages are clamped to the last study year.
    """
    if stage_idx < first_year_stages:
        return 1
    year = (stage_idx - first_year_stages) // 12 + 2
    return min(year, num_anos)


@dataclass
class _StageInputs:
    """Mutable per-stage thermal parameters threaded through the 6 bound steps.

    Each ``_step*`` helper transforms this state in place, mirroring NEWAVE's
    sintetizador processing order. Making the state explicit lets every step —
    including the FCMAX/GTMIN interaction in :func:`_step6_evaluate_bounds` — be
    unit-tested in isolation, which the former mutate-in-place monolith over
    shared loop locals could not.
    """

    potencia: float
    fcmax: float
    teif: float
    ip: float
    gen_min: float


def _step1_zero_ip_before_maintenance(
    state: _StageInputs, stage_idx: int, maint_end_stage: int
) -> None:
    """Step 1: zero IP for ALL plants in stages before the maintenance end."""
    if stage_idx < maint_end_stage:
        state.ip = 0.0


def _step2_null_potencia_for_potef(
    state: _StageInputs, stage_idx: int, maint_end_stage: int, has_potef: bool
) -> None:
    """Step 2: null ``potencia`` for stages >= maint end when EXPT POTEF exists.

    EXPT restores the real value in step 4; zeroing first means a plant with no
    POTEF window covering a stage stays at zero capacity there.
    """
    if stage_idx >= maint_end_stage and has_potef:
        state.potencia = 0.0


def _step3_null_gen_min_for_gtmin(
    state: _StageInputs, stage_idx: int, maint_end_stage: int, has_gtmin: bool
) -> None:
    """Step 3: null ``gen_min`` for stages >= maint end when EXPT GTMIN exists.

    EXPT restores the real value in step 4.
    """
    if stage_idx >= maint_end_stage and has_gtmin:
        state.gen_min = 0.0


def _step4_apply_expt_overrides(
    state: _StageInputs,
    overrides: list[dict],
    ref_date: date,
    is_post_study: bool,
    last_stage_date: date,
) -> None:
    """Step 4: apply EXPT overrides (POTEF/FCMAX/TEIFT/GTMIN/IPTER) in file order.

    Closed windows test against ``ref_date`` (frozen at the last study stage in
    the post-study tail); an open-ended override blankets the whole tail and,
    coming last in file order, wins over any per-month window for the stage.
    """
    for override in overrides:
        ov_start = pd.Timestamp(override["data_inicio"]).date()
        ov_end_raw = override["data_fim"]
        open_ended = pd.isna(ov_end_raw)
        ov_end = last_stage_date if open_ended else pd.Timestamp(ov_end_raw).date()
        if open_ended and is_post_study:
            applies = True
        else:
            applies = ov_start <= ref_date <= ov_end
        if not applies:
            continue

        tipo = override["tipo"]
        value = override["modificacao"]
        if tipo == "POTEF":
            state.potencia = value
        elif tipo == "FCMAX":
            state.fcmax = value
        elif tipo == "TEIFT":
            state.teif = value
        elif tipo == "GTMIN":
            state.gen_min = value
        elif tipo == "IPTER":
            state.ip = value


def _step4b_apply_potef_availability(
    state: _StageInputs,
    windows: list[tuple[date, date]] | None,
    stage_date: date,
) -> None:
    """Step 4b: a POTEF schedule defines the *only* periods the plant is available.

    Outside every window (tested against the ACTUAL stage date, not the frozen
    ``ref_date``) the plant is out of service for that stage.
    """
    if windows is not None and not any(ws <= stage_date <= we for ws, we in windows):
        state.potencia = 0.0
        state.gen_min = 0.0


def _step5_apply_maint_reduction(
    state: _StageInputs,
    maint_reduction: np.ndarray | None,
    stage_idx: int,
    maint_end_stage: int,
) -> None:
    """Step 5: MANUTT subtracts its capacity reduction from ``potencia``.

    Applied only in stages before the maintenance end, matching sintetizador,
    which applies EXPT (step 4) before MANUTT.
    """
    if maint_reduction is not None and stage_idx < maint_end_stage:
        state.potencia -= float(maint_reduction[stage_idx])


def _step6_evaluate_bounds(state: _StageInputs) -> tuple[float, float, bool]:
    """Step 6: evaluate ``(min_mw, max_mw, gtmin_above_capacity)``.

    Per NEWAVE, FCMAX sets the maximum generation and GTMIN the minimum, and the
    two are **independent**::

        capacity_max = potencia * (fcmax/100) * ((100-ip)/100) * ((100-teif)/100)
        min_mw       = gen_min   (GTMIN)

    NEWAVE treats ``min_mw > capacity_max`` as a data error (an inflexible plant
    whose minimum exceeds its available capacity). Cobre honors the inflexible
    GTMIN and lifts the upper bound to ``max(capacity_max, gen_min)`` to keep the
    LP feasible, returning ``gtmin_above_capacity`` so the caller can surface the
    (rare) condition. The former code instead clamped ``min_mw`` DOWN to
    ``capacity_max``, silently forcing the plant below its GTMIN.
    """
    potencia = max(0.0, state.potencia)
    capacity_max = max(
        0.0,
        potencia
        * (state.fcmax / 100.0)
        * ((100.0 - state.ip) / 100.0)
        * ((100.0 - state.teif) / 100.0),
    )
    min_mw = max(0.0, state.gen_min)
    gtmin_above_capacity = min_mw > capacity_max
    max_mw = max(capacity_max, min_mw)
    return min_mw, max_mw, gtmin_above_capacity


def convert_thermal_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build per-stage thermal generation bounds from EXPT.DAT and MANUTT.DAT.

    Also embeds per-stage ``cost_per_mwh`` overrides from ``clast.dat``
    when thermal costs vary across study years.

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

    Returns ``None`` if no bounds or cost overrides are needed.
    """
    from inewave.newave import Dger, Expt, Manutt

    dger = Dger.read(str(nw_files.dger))
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    num_anos = horizon.num_anos
    num_maint_years: int = dger.num_anos_manutencao_utes or 0
    study_months = horizon.study_months
    total_stages = horizon.total_stages
    first_year_stages = horizon.first_year_stages

    # ------------------------------------------------------------------
    # 0. Build per-stage cost lookup from CLAST.DAT.
    # ------------------------------------------------------------------
    clast = Clast.read(str(nw_files.clast))
    clast_df = clast.usinas
    clast_modif_df = clast.modificacoes

    # cost_by_code_year: (newave_code, indice_ano_estudo) -> cost
    cost_by_code_year: dict[tuple[int, int], float] = {}
    # Track which thermals have costs that vary across years.
    cost_varies: set[int] = set()
    if clast_df is not None:
        for _, row in clast_df.iterrows():
            code = int(row["codigo_usina"])
            year_idx = int(row["indice_ano_estudo"])
            cost_by_code_year[(code, year_idx)] = float(row["valor"])
        # Detect thermals with non-uniform costs.
        codes_in_clast = {c for c, _ in cost_by_code_year}
        for code in codes_in_clast:
            year_costs = [
                cost_by_code_year[(code, y)]
                for y in range(1, num_anos + 1)
                if (code, y) in cost_by_code_year
            ]
            if len(set(year_costs)) > 1:
                cost_varies.add(code)

    # Date-range cost overrides from the modificacoes block at the end of
    # clast.dat. Each entry overrides the year-indexed cost for stages
    # whose first-of-month date falls within [data_inicio, data_fim].
    # A plant with at least one modification is treated as cost-varying
    # even when its year-indexed costs are uniform.
    modif_by_code: dict[int, list[dict]] = {}
    if clast_modif_df is not None and not clast_modif_df.empty:
        for _, row in clast_modif_df.iterrows():
            code = int(row["codigo_usina"])
            modif_by_code.setdefault(code, []).append(
                {
                    "data_inicio": row["data_inicio"],
                    "data_fim": row["data_fim"],
                    "custo": float(row["custo"]),
                }
            )
        cost_varies.update(modif_by_code.keys())

    has_capacity_sources = nw_files.expt is not None or nw_files.manutt is not None

    # If no EXPT/MANUTT and no varying costs, nothing to emit.
    if not has_capacity_sources and not cost_varies:
        _LOG.debug("No EXPT/MANUTT/varying costs; skipping thermal bounds.")
        return None

    # Maintenance end: stages before this index have IP=0 globally.
    # Maintenance years are counted as full calendar years from the study
    # start year.  For a March 2026 start with 1 maintenance year, the
    # period covers March-December 2026 (10 stages), not 12.
    maint_end_stage = num_maint_years * 12 + (1 - start_month)

    stage_dates = build_stage_dates(start_year, start_month, total_stages)

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
    # Per-code union of POTEF availability windows.  A plant is considered
    # in service for any stage whose date falls inside at least one window.
    # Open-ended data_fim is treated as extending to the last stage date.
    # This correctly handles chained POTEF schedules (e.g. a finite window
    # followed by an open-ended one): NEWAVE applies them in sequence
    # rather than decommissioning the plant at the first window's end.
    potef_windows: dict[int, list[tuple[date, date]]] = {}
    for code, overrides in expt_by_code.items():
        for o in overrides:
            if o["tipo"] == "POTEF":
                codes_with_potef.add(code)
                ov_start = pd.Timestamp(o["data_inicio"]).date()
                end_raw = o["data_fim"]
                ov_end = (
                    stage_dates[-1]
                    if pd.isna(end_raw)
                    else pd.Timestamp(end_raw).date()
                )
                potef_windows.setdefault(code, []).append((ov_start, ov_end))
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
        set(expt_by_code.keys())
        | set(manutt_by_code.keys())
        | set(base_default.keys())
        | cost_varies
    )

    rows_thermal_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_min: list[float] = []
    rows_max: list[float] = []
    rows_cost: list[float | None] = []
    # Plants where GTMIN exceeded the FCMAX-derived capacity (warned once each).
    gtmin_above_capacity_codes: set[int] = set()

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

        # NEWAVE freezes the post-study tail at the LAST STUDY STAGE's
        # configuration: it re-uses neither the per-calendar-month base nor the
        # per-month EXPT windows dated inside the tail.
        # So for post-study stages we evaluate the
        # base and windowed overrides at the last-study-stage date (``ref_date``),
        # while open-ended overrides still apply across the tail and POTEF
        # availability (step 4b) uses the ACTUAL stage date.
        last_study_idx = study_months - 1
        for stage_idx, stage_date in enumerate(stage_dates):
            is_post_study = stage_idx >= study_months
            ref_date = stage_dates[last_study_idx] if is_post_study else stage_date

            cal_month = ref_date.month
            state = _StageInputs(**_base(newave_code, cal_month))

            _step1_zero_ip_before_maintenance(state, stage_idx, maint_end_stage)
            _step2_null_potencia_for_potef(
                state,
                stage_idx,
                maint_end_stage,
                newave_code in codes_with_potef,
            )
            _step3_null_gen_min_for_gtmin(
                state,
                stage_idx,
                maint_end_stage,
                newave_code in codes_with_gtmin,
            )
            _step4_apply_expt_overrides(
                state, overrides, ref_date, is_post_study, stage_dates[-1]
            )
            _step4b_apply_potef_availability(
                state, potef_windows.get(newave_code), stage_date
            )
            _step5_apply_maint_reduction(
                state, maint_reduction, stage_idx, maint_end_stage
            )
            min_mw, max_mw, gtmin_above_capacity = _step6_evaluate_bounds(state)
            if gtmin_above_capacity:
                gtmin_above_capacity_codes.add(newave_code)

            # Per-stage cost override from CLAST (only for varying-cost thermals).
            stage_cost: float | None = None
            if newave_code in cost_varies:
                year_idx = _stage_to_study_year(stage_idx, first_year_stages, num_anos)
                stage_cost = cost_by_code_year.get((newave_code, year_idx))
                # Apply clast.modificacoes overrides in file order; later
                # entries win when windows overlap, matching NEWAVE's
                # sequential application of the modification block.
                for modif in modif_by_code.get(newave_code, []):
                    mod_start = pd.Timestamp(modif["data_inicio"]).date()
                    mod_end_raw = modif["data_fim"]
                    if pd.isna(mod_end_raw):
                        mod_end = stage_dates[-1]
                    else:
                        mod_end = pd.Timestamp(mod_end_raw).date()
                    if mod_start <= stage_date <= mod_end:
                        stage_cost = modif["custo"]

            rows_thermal_id.append(thermal_id)
            rows_stage_id.append(stage_idx)
            rows_min.append(min_mw)
            rows_max.append(max_mw)
            rows_cost.append(stage_cost)

    if gtmin_above_capacity_codes:
        _LOG.warning(
            "GTMIN exceeds the FCMAX-derived capacity for %d thermal plant(s) "
            "%s in at least one stage; honoring GTMIN (NEWAVE rejects such "
            "min > max inputs). Check EXPT FCMAX/GTMIN and MANUTT for these "
            "plants.",
            len(gtmin_above_capacity_codes),
            sorted(gtmin_above_capacity_codes),
        )

    if not rows_thermal_id:
        return None

    return pa.table(
        {
            "thermal_id": pa.array(rows_thermal_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "min_generation_mw": pa.array(rows_min, type=pa.float64()),
            "max_generation_mw": pa.array(rows_max, type=pa.float64()),
            "cost_per_mwh": pa.array(rows_cost, type=pa.float64()),
        },
        schema=_THERMAL_BOUNDS_SCHEMA,
    )
