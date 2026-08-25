"""Thermal entity converter: maps the source model thermal plant data to Cobre thermal
JSON.

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

from cobre_bridge.case import NewaveCase
from cobre_bridge.cobre import schemas as cobre_schemas
from cobre_bridge.converters.anticipated import read_anticipated_dispatch
from cobre_bridge.core.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)
from cobre_bridge.horizon import build_stage_dates, historical_start_date
from cobre_bridge.id_map import NewaveIdMap

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


def thermal_generation_bounds(case: NewaveCase) -> dict[int, tuple[float, float]]:
    """Return ``{codigo_usina: (min_mw, max_mw)}`` static generation bounds.

    These are the plant-level (non-stage-varying) bounds written to
    ``thermals.json`` as ``generation.min_mw`` / ``generation.max_mw``, and the
    interval Cobre's semantic validator enforces on each
    ``past_anticipated_commitments.values_mw`` entry:

    * ``max_mw = potencia_instalada * fator_capacidade_maximo / 100`` and
    * ``min_mw = geracao_minima``,

    both from ``term.dat`` month 1 (falling back to any month for plants absent
    from month 1, with ``min_mw = 0``). Plants absent from ``term.dat`` map to
    ``(0.0, 0.0)``.
    """
    term_df = case.term.usinas
    bounds: dict[int, tuple[float, float]] = {}
    if term_df is None:
        return bounds

    month1 = term_df[term_df["mes"] == 1]
    for _, row in month1.iterrows():
        code = int(row["codigo_usina"])
        cap = float(row["potencia_instalada"])
        max_factor = float(row["fator_capacidade_maximo"])
        bounds[code] = (float(row["geracao_minima"]), cap * max_factor / 100.0)

    # Plants present in term.dat but not in month 1: use any row, min_mw = 0.
    for _, row in term_df.iterrows():
        code = int(row["codigo_usina"])
        if code not in bounds:
            cap = float(row["potencia_instalada"])
            max_factor = float(row["fator_capacidade_maximo"])
            bounds[code] = (0.0, cap * max_factor / 100.0)

    return bounds


def convert_thermals(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model thermal plant data to a Cobre ``thermals.json`` dict.

    Reads ``conft.dat``, ``clast.dat``, and ``term.dat`` from *case*.
    Returns a dict with a ``"thermals"`` key containing a list of thermal
    entries sorted by Cobre 0-based ID.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping for bus cross-references.
    """
    conft_df = case.conft.usinas
    clast_df = case.clast.usinas

    # Anticipated dispatch (the source model GNL) — gated by
    # dger.despacho_antecipado_gnl. Returns an empty dict when the flag is off, so
    # non-GNL cases incur zero cost beyond the dger read inside the helper.
    anticipated_by_code = read_anticipated_dispatch(case)

    # Build cost lookup: codigo_usina -> cost for indice_ano_estudo == 1.
    cost_map: dict[int, float] = {}
    if clast_df is not None:
        first_year = clast_df[clast_df["indice_ano_estudo"] == 1]
        for _, row in first_year.iterrows():
            cost_map[int(row["codigo_usina"])] = float(row["valor"])

    # Static (min_mw, max_mw) per plant — single-sourced so the anticipated
    # commitment seeding clamps against the very bounds written here.
    gen_bounds = thermal_generation_bounds(case)

    # The source model carries no per-thermal commissioning date; treat every
    # thermal as in service since the historical record (Cobre uses the date only
    # as a canonical-ordering key, tiebroken by id).
    op_date = historical_start_date(case.dger)

    thermals: list[dict] = []
    for _, row in conft_df.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()
        submercado = int(row["submercado"])

        bus_id = id_map.bus_id(submercado)

        gen_min, max_mw = gen_bounds.get(newave_code, (0.0, 0.0))
        cost = cost_map.get(newave_code, 0.0)

        anticipated = anticipated_by_code.get(newave_code)
        anticipated_config = (
            {"lead_stages": anticipated.lead_stages} if anticipated else None
        )

        thermal_entry: dict = {
            "id": id_map.thermal_id(newave_code),
            "name": name,
            "operational_start_date": op_date,
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
        "$schema": cobre_schemas.schema_url_for("system/thermals.json"),
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

    Each ``_step*`` helper transforms this state in place, mirroring the source model's
    sintetizador processing order. Making the state explicit lets every step — including
    the FCMAX/GTMIN interaction in :func:`_step6_evaluate_bounds` — be unit-tested in
    isolation, which the former mutate-in-place monolith over shared loop locals could
    not.
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
    *,
    expt_without_potef: bool = False,
) -> None:
    """Step 4b: a POTEF schedule defines the *only* periods the plant is available.

    Outside every window (tested against the caller-supplied ``stage_date`` — the
    actual stage date in-study, the frozen last-study-stage date in the post-study
    tail) the plant is out of service for that stage.

    A plant referenced in EXPT.DAT with modifier-only entries (TEIFT/FCMAX/ GTMIN/IPTER)
    but **no establishing POTEF** has no installed power: the modifiers have nothing to
    modify, so the source model reports ``GERACAO MAXIMA POR CLASSE TERMICA = 0`` for it
    every stage. ``expt_without_potef`` marks that case so the plant is held out of
    service across the whole horizon rather than falling back to its TERM.DAT registry
    capacity (e.g. LINHARES, which carries only a TEIFT entry in the validation deck).
    """
    if windows is None:
        if expt_without_potef:
            state.potencia = 0.0
            state.gen_min = 0.0
        return
    if not any(ws <= stage_date <= we for ws, we in windows):
        state.potencia = 0.0
        state.gen_min = 0.0


def _step4c_apply_gtmin_availability(
    state: _StageInputs,
    windows: list[tuple[date, date]] | None,
    stage_date: date,
    *,
    expt_without_gtmin: bool = False,
) -> None:
    """Step 4c: a GTMIN schedule defines the *only* periods with a minimum.

    The source model takes the minimum generation from EXPT GTMIN windows and uses **0**
    outside them — it ignores the TERM.DAT "GTMIN PARA O PRIMEIRO ANO" column. Outside
    every window (tested against the caller-supplied ``stage_date`` — the actual stage
    date in-study, the frozen last-study-stage date in the post-study tail) the plant's
    minimum is dropped to 0.

    A plant configured via EXPT but with **no GTMIN entry** has no minimum at
    all (``expt_without_gtmin``); its TERM.DAT GTMIN must not leak in as a
    spurious must-run (e.g. JARAQUI / MARLIM AZUL in the validation deck). This
    only drops the *lower* bound — capacity (step 4b) is unaffected.
    """
    if windows is None:
        if expt_without_gtmin:
            state.gen_min = 0.0
        return
    if not any(ws <= stage_date <= we for ws, we in windows):
        state.gen_min = 0.0


def _potef_online_at(
    windows: list[tuple[date, date]] | None,
    *,
    expt_without_potef: bool,
    when: date,
) -> bool:
    """Whether a plant has installed capacity at ``when`` per its POTEF schedule.

    A plant with no POTEF schedule is always online (its capacity comes from the
    TERM.DAT registry). A plant referenced in EXPT with no establishing POTEF has
    no installed power. Otherwise it is online iff some POTEF window covers
    ``when``. Used to pick the post-study freeze reference (see the loop).
    """
    if expt_without_potef:
        return False
    if not windows:
        return True
    return any(ws <= when <= we for ws, we in windows)


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

    Per source-model, FCMAX sets the maximum generation and GTMIN the minimum, and the
    two are **independent**::

        capacity_max = potencia * (fcmax/100) * ((100-ip)/100) * ((100-teif)/100)
        min_mw       = gen_min   (GTMIN)

    The source model treats ``min_mw > capacity_max`` as a data error (an inflexible
    plant whose minimum exceeds its available capacity). Cobre honors the inflexible
    GTMIN and lifts the upper bound to ``max(capacity_max, gen_min)`` to keep the LP
    feasible, returning ``gtmin_above_capacity`` so the caller can surface the (rare)
    condition. The former code instead clamped ``min_mw`` DOWN to ``capacity_max``,
    silently forcing the plant below its GTMIN.
    """
    capacity_max = _capacity_max(state)
    min_mw = max(0.0, state.gen_min)
    gtmin_above_capacity = min_mw > capacity_max
    max_mw = max(capacity_max, min_mw)
    return min_mw, max_mw, gtmin_above_capacity


def _capacity_max(state: _StageInputs) -> float:
    """Available maximum generation: ``potencia·(fcmax)·(100-ip)·(100-teif)`` (MW).

    Extracted from :func:`_step6_evaluate_bounds` so the GTMIN-above-capacity
    diagnostic can recover the capacity it was compared against, per stage, without
    changing that function's (unit-tested) return signature.
    """
    potencia = max(0.0, state.potencia)
    return max(
        0.0,
        potencia
        * (state.fcmax / 100.0)
        * ((100.0 - state.ip) / 100.0)
        * ((100.0 - state.teif) / 100.0),
    )


@dataclass
class _GtminRecord:
    """One stage where a thermal plant's GTMIN exceeded its available capacity."""

    code: int
    stage_id: int
    gtmin_mw: float
    capacity_mw: float


def _thermal_names(case: NewaveCase) -> dict[int, str]:
    """Map thermal codes to plant names from conft.dat (empty on any read failure)."""
    try:
        usinas = case.conft.usinas
    except (OSError, ValueError, AttributeError, KeyError, TypeError):
        return {}
    return {
        int(row["codigo_usina"]): str(row["nome_usina"]).strip()
        for _, row in usinas.iterrows()
    }


def _emit_gtmin_above_capacity(records: list[_GtminRecord], case: NewaveCase) -> None:
    """Emit the per-plant GTMIN-above-capacity diagnostic from the captured records.

    One row per plant: the stages it occurred on (collapsed to ranges) and the worst
    GTMIN vs the lowest available capacity across those stages — so the user sees the
    plant, where, and by how much, instead of a bare code list.
    """
    names = _thermal_names(case)
    by_code: dict[int, list[_GtminRecord]] = {}
    for record in records:
        by_code.setdefault(record.code, []).append(record)

    rows: list[list[object]] = []
    for code in sorted(by_code):
        recs = by_code[code]
        rows.append(
            [
                names.get(code, "?"),
                code,
                format_stage_ranges(r.stage_id for r in recs),
                round(max(r.gtmin_mw for r in recs), 1),
                round(min(r.capacity_mw for r in recs), 1),
            ]
        )

    plant_count = len(by_code)
    emit(
        Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title=f"GTMIN exceeds available capacity ({plant_count} plant(s))",
            summary=(
                f"GTMIN exceeds the FCMAX-derived capacity for {plant_count} thermal "
                "plant(s) in at least one stage; honoring GTMIN to keep the LP "
                "feasible."
            ),
            table=DiagnosticTable(
                columns=["Plant", "Code", "Stages", "GTMIN MW", "Cap MW"],
                rows=rows,
                justify=["left", "right", "left", "right", "right"],
                caption=(
                    "GTMIN = max, Cap = min available capacity across the listed stages"
                ),
            ),
            remediation="Check EXPT FCMAX/GTMIN and MANUTT for these plants.",
        ),
        logger=_LOG,
    )


def convert_thermal_bounds(
    case: NewaveCase,
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
    4b. POTEF availability: zero ``potencia`` outside the EXPT POTEF windows
       (and for plants in EXPT with no POTEF — not installed).
    4c. GTMIN availability: zero ``gen_min`` outside the EXPT GTMIN windows
       (and for plants in EXPT with no GTMIN). The source model takes the minimum only
       from EXPT GTMIN windows and ignores the TERM.DAT GTMIN outside them.
    5. Apply MANUTT capacity reductions (only stages < maintenance_end).
    6. Evaluate: ``pot * (fcmax/100) * ((100-ip)/100) * ((100-teif)/100)``

    Returns ``None`` if no bounds or cost overrides are needed.
    """
    dger = case.dger
    horizon = case.horizon
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
    clast = case.clast
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

    has_capacity_sources = case.files.expt is not None or case.files.manutt is not None

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
    term_df = case.term.usinas

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
    if case.files.expt is not None:
        try:
            expt_obj = case.expt
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
    # ── EXPT-authoritative-timeline principle ─────────────────────────────
    # The source model drives the thermal configuration from EXPT.DAT, not TERM.DAT.
    # TERM.DAT supplies *registry/reference* values; EXPT.DAT declares the operative
    # per-attribute timeline over date windows. Each attribute has a
    # DEFAULT it reverts to OUTSIDE its EXPT windows:
    #   • POTEF (installed capacity) -> default 0   (plant not motorised)
    #   • GTMIN (minimum generation) -> default 0   (no must-run)
    #   • FCMAX / TEIF / IP (modifiers) -> default = TERM.DAT first-year value
    # So a plant with no POTEF window has 0 capacity, and one with no GTMIN
    # window (or outside it) has 0 minimum — the TERM.DAT POT/GTMIN columns are
    # NOT operative defaults. The POTEF (step 4b) and GTMIN (step 4c) window
    # logic below are the SAME rule applied to two attributes, not two ad-hoc
    # exceptions: build each attribute's window union, then revert to its
    # default wherever no window covers the stage.
    #
    # A stage is covered if its date falls inside at least one window; open-ended
    # data_fim extends to the last stage date, and chained schedules (a finite
    # window followed by an open-ended one) apply in sequence, not ended at the
    # first window.
    potef_windows: dict[int, list[tuple[date, date]]] = {}
    gtmin_windows: dict[int, list[tuple[date, date]]] = {}

    def _window(o: dict) -> tuple[date, date]:
        start = pd.Timestamp(o["data_inicio"]).date()
        end_raw = o["data_fim"]
        end = stage_dates[-1] if pd.isna(end_raw) else pd.Timestamp(end_raw).date()
        return start, end

    for code, overrides in expt_by_code.items():
        for o in overrides:
            if o["tipo"] == "POTEF":
                codes_with_potef.add(code)
                potef_windows.setdefault(code, []).append(_window(o))
            elif o["tipo"] == "GTMIN":
                codes_with_gtmin.add(code)
                gtmin_windows.setdefault(code, []).append(_window(o))

    # Plants referenced in EXPT with modifier-only entries (TEIFT/FCMAX/GTMIN/ IPTER)
    # but no establishing POTEF have no installed power: The source model reports
    # ``GERACAO MAXIMA POR CLASSE TERMICA = 0`` for them every stage. Hold them out of
    # service (step 4b) rather than falling back to the TERM.DAT registry capacity.
    codes_expt_without_potef = set(expt_by_code) - codes_with_potef
    if codes_expt_without_potef:
        names = _thermal_names(case)
        emit(
            Diagnostic(
                code="thermal-expt-without-potef",
                severity=Severity.INFO,
                category="Thermal bounds",
                title=f"EXPT entries without POTEF ({len(codes_expt_without_potef)})",
                summary=(
                    f"{len(codes_expt_without_potef)} thermal plant(s) appear in "
                    "EXPT.DAT without a POTEF entry; treated as not installed "
                    "(max generation 0), matching NEWAVE."
                ),
                table=DiagnosticTable(
                    columns=["Plant", "Code"],
                    rows=[
                        [names.get(code, "?"), code]
                        for code in sorted(codes_expt_without_potef)
                    ],
                    justify=["left", "right"],
                ),
            ),
            logger=_LOG,
        )

    # The same authority applies to the minimum generation: The source model takes GTMIN
    # only from EXPT GTMIN windows and uses 0 outside them, ignoring the TERM.DAT "GTMIN
    # PARA O PRIMEIRO ANO" column. A plant configured via EXPT but with no GTMIN entry
    # therefore has no minimum (its TERM.DAT GTMIN must not leak in as a spurious
    # must-run). Handled in step 4c.
    codes_expt_without_gtmin = set(expt_by_code) - codes_with_gtmin

    # ------------------------------------------------------------------
    # 3. Load MANUTT maintenance events.
    # ------------------------------------------------------------------
    manutt_by_code: dict[int, pd.DataFrame] = {}
    if case.files.manutt is not None:
        try:
            manutt_obj = case.manutt
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
    # Per-stage records where GTMIN exceeded the FCMAX-derived capacity, kept so the
    # diagnostic can name the plant, the stages, and the GTMIN-vs-capacity values.
    gtmin_records: list[_GtminRecord] = []

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

        # The source model's "período estático final" freezes the post-study tail at a
        # single December snapshot (manual table, p.32-33): thermal min generation, max
        # generation and cost are ALL frozen there, and maintenance is "não
        # considerada". The freeze reference is December of the last STUDY year for a
        # plant already online then; but a plant that comes online ONLY in the
        # post-study (POTEF dated after the last study stage — e.g. AZULAO II/IV, MANAUS
        # I) does not yet exist in that December, so the source model instead freezes it
        # at its *online* terminal configuration. We mirror this by picking the freeze
        # reference per plant: the last study stage normally, or the last (terminal
        # December) stage when the plant only switches on in the post-study tail. Every
        # date-dependent input — base, windowed EXPT overrides, POTEF/GTMIN availability
        # (4b/4c) and clast cost modifications — is then evaluated at that single
        # reference date; using the ACTUAL stage date would re-apply the last year's
        # seasonal on/off pattern across the tail (a real bug observed as Feb–May GTMIN
        # dropouts repeating every post-study year).
        last_study_idx = study_months - 1
        comes_online_in_post_study = not _potef_online_at(
            potef_windows.get(newave_code),
            expt_without_potef=newave_code in codes_expt_without_potef,
            when=stage_dates[last_study_idx],
        ) and _potef_online_at(
            potef_windows.get(newave_code),
            expt_without_potef=newave_code in codes_expt_without_potef,
            when=stage_dates[-1],
        )
        freeze_idx = (
            len(stage_dates) - 1 if comes_online_in_post_study else last_study_idx
        )
        for stage_idx, stage_date in enumerate(stage_dates):
            is_post_study = stage_idx >= study_months
            ref_date = stage_dates[freeze_idx] if is_post_study else stage_date

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
                state,
                potef_windows.get(newave_code),
                ref_date,
                expt_without_potef=newave_code in codes_expt_without_potef,
            )
            _step4c_apply_gtmin_availability(
                state,
                gtmin_windows.get(newave_code),
                ref_date,
                expt_without_gtmin=newave_code in codes_expt_without_gtmin,
            )
            _step5_apply_maint_reduction(
                state, maint_reduction, stage_idx, maint_end_stage
            )
            min_mw, max_mw, gtmin_above_capacity = _step6_evaluate_bounds(state)
            if gtmin_above_capacity:
                gtmin_records.append(
                    _GtminRecord(
                        code=newave_code,
                        stage_id=stage_idx,
                        gtmin_mw=min_mw,
                        capacity_mw=_capacity_max(state),
                    )
                )

            # Per-stage cost override from CLAST (only for varying-cost thermals).
            stage_cost: float | None = None
            if newave_code in cost_varies:
                year_idx = _stage_to_study_year(stage_idx, first_year_stages, num_anos)
                stage_cost = cost_by_code_year.get((newave_code, year_idx))
                # Apply clast.modificacoes overrides in file order; later entries win
                # when windows overlap, matching the source model's sequential
                # application of the modification block. Tested against ``ref_date`` so
                # the post-study tail freezes at the last study stage's cost (December)
                # instead of letting a future-dated modification leak into the static
                # final period.
                for modif in modif_by_code.get(newave_code, []):
                    mod_start = pd.Timestamp(modif["data_inicio"]).date()
                    mod_end_raw = modif["data_fim"]
                    if pd.isna(mod_end_raw):
                        mod_end = stage_dates[-1]
                    else:
                        mod_end = pd.Timestamp(mod_end_raw).date()
                    if mod_start <= ref_date <= mod_end:
                        stage_cost = modif["custo"]

            rows_thermal_id.append(thermal_id)
            rows_stage_id.append(stage_idx)
            rows_min.append(min_mw)
            rows_max.append(max_mw)
            rows_cost.append(stage_cost)

    if gtmin_records:
        _emit_gtmin_above_capacity(gtmin_records, case)

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
