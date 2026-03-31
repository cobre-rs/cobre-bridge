"""Compute NEWAVE bounds from input files as flat dicts.

Standalone functions that extract bounds computation logic from the
converter modules (``hydro.py``, ``thermal.py``, ``network.py``) and
return simple ``dict[tuple[int, int, str], float]`` lookups keyed by
``(cobre_entity_id, stage_id, bound_name)``.

These functions avoid coupling to the conversion pipeline — they read
NEWAVE input files directly via inewave, except for ``read_cadastro``
which applies permanent MODIF overrides.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

# NEWAVE big-M sentinel: values >= this threshold mean "no limit".
_BIG_M = 99990.0


# -------------------------------------------------------------------
# Shared temporal helpers
# -------------------------------------------------------------------


def _build_stage_dates(
    start_year: int, start_month: int, total_stages: int
) -> list[date]:
    """Return first-of-month dates for each study stage."""
    stages: list[date] = []
    y, m = start_year, start_month
    for _ in range(total_stages):
        stages.append(date(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return stages


def _read_study_params(
    nw_files: NewaveFiles,
) -> tuple[int, int, int, int, int, int]:
    """Read DGER and return temporal parameters.

    Returns (start_year, start_month, num_anos, num_anos_pos,
    study_months, total_stages).
    """
    from inewave.newave import Dger

    dger = Dger.read(str(nw_files.dger))
    start_year: int = int(dger.ano_inicio_estudo)
    start_month: int = int(dger.mes_inicio_estudo)
    num_anos: int = int(dger.num_anos_estudo or 1)
    num_anos_pos: int = int(dger.num_anos_pos_estudo or 0)
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12
    return start_year, start_month, num_anos, num_anos_pos, study_months, total_stages


# -------------------------------------------------------------------
# Hydro bounds
# -------------------------------------------------------------------


def _extract_temporal_overrides(
    nw_files: NewaveFiles, confhd_codes: list[int]
) -> dict[int, list[dict]]:
    """Extract MODIF.DAT temporal overrides for plants in *confhd_codes*.

    Returns a dict keyed by plant code, each value a list of override
    dicts: ``{"type": str, "month": int, "year": int, "value": float}``.
    """
    from inewave.newave import Modif

    modif_path = nw_files.modif
    if modif_path is None:
        return {}

    temporal_types = frozenset(
        {"VAZMINT", "VMAXT", "VMINT", "CFUGA", "CMONT", "TURBMINT", "TURBMAXT"}
    )

    modif = Modif.read(str(modif_path))
    confhd_set = set(confhd_codes)
    result: dict[int, list[dict]] = {}

    usina_records = modif.usina()
    if not usina_records:
        return result

    for usina_rec in usina_records:
        code = int(usina_rec.codigo)
        if code not in confhd_set:
            continue

        plant_overrides: list[dict] = []
        for rec in modif.modificacoes_usina(code):
            type_name = type(rec).__name__
            if type_name not in temporal_types:
                continue

            data = rec.data_inicio
            month = int(data.month)
            year = int(data.year)

            if type_name == "VAZMINT":
                value = float(rec.vazao)
            elif type_name in ("VMAXT", "VMINT"):
                value = float(rec.volume)
            elif type_name in ("CFUGA", "CMONT"):
                value = float(rec.nivel)
            elif type_name in ("TURBMINT", "TURBMAXT"):
                value = float(rec.turbinamento)
            else:
                continue

            plant_overrides.append(
                {"type": type_name, "month": month, "year": year, "value": value}
            )

        if plant_overrides:
            result[code] = plant_overrides

    return result


def compute_hydro_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Compute per-stage hydro bounds from NEWAVE input files.

    Returns ``{(cobre_hydro_id, stage_id, bound_name): value}`` where
    ``bound_name`` is one of: ``storage_min``, ``storage_max``,
    ``turbined_min``, ``turbined_max``, ``outflow_min``.

    Values are in hm3 (storage) or m3/s (turbined/outflow).  Big-M
    sentinel values (>= 99990) are excluded (treated as "no limit").

    Returns an empty dict if MODIF.DAT is absent or contains no
    relevant temporal override records.
    """
    from inewave.newave import Confhd

    from cobre_bridge.converters.hydro import read_cadastro

    modif_path = nw_files.modif
    if modif_path is None:
        return {}

    (
        start_year,
        start_month,
        _num_anos,
        _num_anos_pos,
        study_months,
        total_stages,
    ) = _read_study_params(nw_files)

    cadastro = read_cadastro(nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]
    confhd_codes = [int(r["codigo_usina"]) for _, r in non_fict.iterrows()]

    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)

    def _build_step_function(
        recs: list[dict],
        transform: Callable[[float], float],
    ) -> dict[int, float]:
        """Build a step-function from override records.

        Each record sets the value from its stage onward until the next
        record overrides it.  Raw values >= _BIG_M mean "restore default"
        and clear the forward-fill.
        """
        changepoints: list[tuple[int, float]] = []
        for rec in recs:
            sid = (rec["year"] - start_year) * 12 + (rec["month"] - start_month)
            if sid < 0:
                sid = 0
            changepoints.append((sid, rec["value"]))
        changepoints.sort()

        if not changepoints:
            return {}

        result_inner: dict[int, float] = {}
        cp_idx = 0
        current: float | None = None
        first_stage = changepoints[0][0]

        for stage_id in range(first_stage, study_months):
            while cp_idx < len(changepoints) and changepoints[cp_idx][0] <= stage_id:
                raw = changepoints[cp_idx][1]
                current = None if raw >= _BIG_M else transform(raw)
                cp_idx += 1
            if current is not None:
                result_inner[stage_id] = current

        # Post-study seasonal repetition from last study year.
        seasonal: dict[int, float] = {}
        for stage_id in range(max(0, study_months - 12), study_months):
            if stage_id in result_inner:
                cal = ((start_month - 1 + stage_id) % 12) + 1
                seasonal[cal] = result_inner[stage_id]

        for stage_id in range(study_months, total_stages):
            cal = ((start_month - 1 + stage_id) % 12) + 1
            if cal in seasonal:
                current = seasonal[cal]
            if current is not None:
                result_inner[stage_id] = current

        return result_inner

    result: dict[tuple[int, int, str], float] = {}

    for newave_code in sorted(temporal_overrides):
        overrides = temporal_overrides[newave_code]
        vmaxt = [o for o in overrides if o["type"] == "VMAXT"]
        vmint = [o for o in overrides if o["type"] == "VMINT"]
        turbmaxt = [o for o in overrides if o["type"] == "TURBMAXT"]
        turbmint = [o for o in overrides if o["type"] == "TURBMINT"]
        vazmint = [o for o in overrides if o["type"] == "VAZMINT"]

        if not any((vmaxt, vmint, turbmaxt, turbmint, vazmint)):
            continue

        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue

        if newave_code not in cadastro.index:
            continue

        hreg = cadastro.loc[newave_code]
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])
        useful = vol_max - vol_min

        def _pct_to_hm3(
            pct: float,
            _u: float = useful,
            _vm: float = vol_min,
        ) -> float:
            return _vm + (pct / 100.0) * _u

        def _identity(val: float) -> float:
            return val

        # Storage bounds (percentage -> hm3).
        vmaxt_by_stage: dict[int, float] = {}
        vmint_by_stage: dict[int, float] = {}
        if useful > 0:
            vmaxt_by_stage = _build_step_function(vmaxt, _pct_to_hm3)
            vmint_by_stage = _build_step_function(vmint, _pct_to_hm3)

        # Turbined bounds (absolute m3/s).
        turbmaxt_by_stage = _build_step_function(turbmaxt, _identity)
        turbmint_by_stage = _build_step_function(turbmint, _identity)

        # Outflow bounds (absolute m3/s).
        vazmint_by_stage = _build_step_function(vazmint, _identity)

        all_stages = sorted(
            set(vmaxt_by_stage)
            | set(vmint_by_stage)
            | set(turbmaxt_by_stage)
            | set(turbmint_by_stage)
            | set(vazmint_by_stage)
        )
        for stage_id in all_stages:
            if stage_id in vmaxt_by_stage:
                result[(hydro_id, stage_id, "storage_max")] = vmaxt_by_stage[stage_id]
            if stage_id in vmint_by_stage:
                result[(hydro_id, stage_id, "storage_min")] = vmint_by_stage[stage_id]
            if stage_id in turbmaxt_by_stage:
                result[(hydro_id, stage_id, "turbined_max")] = turbmaxt_by_stage[
                    stage_id
                ]
            if stage_id in turbmint_by_stage:
                result[(hydro_id, stage_id, "turbined_min")] = turbmint_by_stage[
                    stage_id
                ]
            if stage_id in vazmint_by_stage:
                result[(hydro_id, stage_id, "outflow_min")] = vazmint_by_stage[stage_id]

    return result


# -------------------------------------------------------------------
# Thermal bounds
# -------------------------------------------------------------------


def _apply_maint_to_capacity(
    base_capacity: float,
    maint_rows: pd.DataFrame,
    stage_dates: list[date],
) -> np.ndarray:
    """Compute monthly effective capacity after subtracting maintenance windows.

    Returns an array of shape (total_stages,) with the monthly average
    effective capacity after maintenance deductions.
    """
    import calendar as _cal

    total_stages = len(stage_dates)
    effective = np.full(total_stages, base_capacity, dtype=float)

    for _, row in maint_rows.iterrows():
        start_dt = pd.Timestamp(row["data_inicio"])
        duration_days = int(row["duracao"])
        unit_power = float(row["potencia"])
        end_dt = start_dt + timedelta(days=duration_days)

        for stage_idx, stage_start in enumerate(stage_dates):
            _, days_in_month = _cal.monthrange(stage_start.year, stage_start.month)
            if stage_start.month == 12:
                stage_end = date(stage_start.year + 1, 1, 1)
            else:
                stage_end = date(stage_start.year, stage_start.month + 1, 1)

            maint_start_date = start_dt.date()
            maint_end_date = end_dt.date()

            overlap_start = max(maint_start_date, stage_start)
            overlap_end = min(maint_end_date, stage_end)
            overlap_days = (overlap_end - overlap_start).days
            if overlap_days <= 0:
                continue

            fraction = overlap_days / days_in_month
            effective[stage_idx] -= unit_power * fraction

    return effective


def compute_thermal_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Compute per-stage thermal generation bounds from NEWAVE input files.

    Returns ``{(cobre_thermal_id, stage_id, bound_name): value}`` where
    ``bound_name`` is one of: ``generation_min``, ``generation_max``.

    Values are in MW.  Follows the sintetizador processing order:
    IP zeroing, POTEF/GTMIN nulling, EXPT overrides, MANUTT reduction,
    formula evaluation.

    Returns an empty dict if neither ``expt.dat`` nor ``manutt.dat`` is
    present.
    """
    if nw_files.expt is None and nw_files.manutt is None:
        return {}

    from inewave.newave import Dger, Expt, Manutt, Term

    dger = Dger.read(str(nw_files.dger))
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    num_maint_years: int = dger.num_anos_manutencao_utes or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    maint_end_stage = (start_year + num_maint_years - start_year) * 12 + (
        1 - start_month
    )

    stage_dates = _build_stage_dates(start_year, start_month, total_stages)

    # Base values per (thermal_code, calendar_month) from TERM.
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
        return {
            "potencia": 0.0,
            "fcmax": 100.0,
            "teif": 0.0,
            "ip": 0.0,
            "gen_min": 0.0,
        }

    # EXPT overrides.
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

    codes_with_potef: set[int] = set()
    codes_with_gtmin: set[int] = set()
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

    # MANUTT maintenance events.
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

    result: dict[tuple[int, int, str], float] = {}

    for newave_code in sorted(all_codes):
        try:
            thermal_id = id_map.thermal_id(newave_code)
        except KeyError:
            continue

        overrides = expt_by_code.get(newave_code, [])
        maint_rows = manutt_by_code.get(newave_code)
        has_maint = maint_rows is not None and not maint_rows.empty

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

            if stage_idx < maint_end_stage:
                ip = 0.0

            if stage_idx >= maint_end_stage and newave_code in codes_with_potef:
                potencia = 0.0

            if stage_idx >= maint_end_stage and newave_code in codes_with_gtmin:
                gen_min = 0.0

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

            potef_end = potef_finite_end.get(newave_code)
            if potef_end is not None and stage_date > potef_end:
                potencia = 0.0
                gen_min = 0.0

            if maint_reduction is not None and stage_idx < maint_end_stage:
                potencia -= float(maint_reduction[stage_idx])

            potencia = max(0.0, potencia)

            max_mw = (
                potencia
                * (fcmax / 100.0)
                * ((100.0 - ip) / 100.0)
                * ((100.0 - teif) / 100.0)
            )
            max_mw = max(0.0, max_mw)
            min_mw = max(0.0, min(gen_min, max_mw))

            result[(thermal_id, stage_idx, "generation_min")] = min_mw
            result[(thermal_id, stage_idx, "generation_max")] = max_mw

    return result


# -------------------------------------------------------------------
# Line bounds
# -------------------------------------------------------------------


def compute_line_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Compute per-stage line flow bounds from NEWAVE input files.

    Returns ``{(cobre_line_id, stage_id, bound_name): value}`` where
    ``bound_name`` is one of: ``direct_flow_max``, ``reverse_flow_max``.

    Values are in MW.  For post-study stages, the last available study
    year's bounds are repeated seasonally.

    Returns an empty dict if ``sistema.dat`` has no interchange limits.
    """
    from inewave.newave import Dger, Sistema

    sistema = Sistema.read(str(nw_files.sistema))
    limites_df: pd.DataFrame | None = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {}

    dger = Dger.read(str(nw_files.dger))
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Canonical pair -> line_id mapping (same logic as convert_lines).
    all_pairs: set[tuple[int, int]] = set()
    for _, row in limites_df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        src, tgt = (de, para) if de < para else (para, de)
        all_pairs.add((src, tgt))

    pair_to_line_id: dict[tuple[int, int], int] = {
        pair: lid for lid, pair in enumerate(sorted(all_pairs))
    }

    # Per-date lookup: {(src, tgt, year, cal_month) -> {direct_mw, reverse_mw}}.
    date_lookup: dict[tuple[int, int, int, int], dict[str, float]] = {}

    for _, row in limites_df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        valor = float(row["valor"])
        sentido = int(row["sentido"])
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)

        src, tgt = (de, para) if de < para else (para, de)
        key = (src, tgt, yr, cal_month)

        if key not in date_lookup:
            date_lookup[key] = {"direct_mw": 0.0, "reverse_mw": 0.0}

        if de < para:
            if sentido == 1:
                date_lookup[key]["direct_mw"] = valor
            else:
                date_lookup[key]["reverse_mw"] = valor
        else:
            if sentido == 1:
                date_lookup[key]["reverse_mw"] = valor
            else:
                date_lookup[key]["direct_mw"] = valor

    # Last-year lookup for post-study repetition.
    last_year_per_key: dict[tuple[int, int, int], tuple[int, dict[str, float]]] = {}
    for (src, tgt, yr, cal_month), caps in date_lookup.items():
        key3 = (src, tgt, cal_month)
        existing = last_year_per_key.get(key3)
        if existing is None or yr > existing[0]:
            last_year_per_key[key3] = (yr, caps)
    last_year_lookup: dict[tuple[int, int, int], dict[str, float]] = {
        k: v for k, (_, v) in last_year_per_key.items()
    }

    result: dict[tuple[int, int, str], float] = {}

    for pair, line_id in sorted(pair_to_line_id.items(), key=lambda x: x[1]):
        src, tgt = pair
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            if is_post_study:
                caps = last_year_lookup.get(
                    (src, tgt, m), {"direct_mw": 0.0, "reverse_mw": 0.0}
                )
            else:
                caps = date_lookup.get((src, tgt, y, m))
                if caps is None:
                    caps = last_year_lookup.get(
                        (src, tgt, m), {"direct_mw": 0.0, "reverse_mw": 0.0}
                    )

            result[(line_id, stage_id, "direct_flow_max")] = caps["direct_mw"]
            result[(line_id, stage_id, "reverse_flow_max")] = caps["reverse_mw"]

            m += 1
            if m > 12:
                m = 1
                y += 1

    return result
