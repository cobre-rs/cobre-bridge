"""Temporal data converter: maps NEWAVE study horizon configuration to Cobre JSON.

Converts ``dger.dat`` and ``patamar.dat`` into the ``stages.json`` and
``config.json`` formats expected by the Cobre solver.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date

from inewave.newave import Cvar, Dger, Patamar

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

logger = logging.getLogger(__name__)

# Default block names by number of blocks.
_SINGLE_BLOCK_NAMES = ["SINGLE"]
_TWO_BLOCK_NAMES = ["HEAVY", "LIGHT"]
_THREE_BLOCK_NAMES = ["HEAVY", "MEDIUM", "LIGHT"]


def _block_names(n: int) -> list[str]:
    """Return a canonical list of block names for *n* blocks.

    Falls back to ``"BLOCK_0"``, ``"BLOCK_1"``, … for uncommon counts.
    """
    if n == 1:
        return _SINGLE_BLOCK_NAMES
    if n == 2:
        return _TWO_BLOCK_NAMES
    if n == 3:
        return _THREE_BLOCK_NAMES
    return [f"BLOCK_{i}" for i in range(n)]


def _month_start_date(year: int, month: int) -> date:
    """Return the first day of the given calendar month."""
    return date(year, month, 1)


def _month_end_date(year: int, month: int) -> date:
    """Return the first day of the *following* month (exclusive end date)."""
    if month == 12:
        return date(year + 1, 1, 1)
    return date(year, month + 1, 1)


def _month_hours(year: int, month: int) -> float:
    """Total number of hours in the given calendar month."""
    days_in_month = calendar.monthrange(year, month)[1]
    return float(days_in_month * 24)


def convert_stages(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:  # noqa: ARG001
    """Convert NEWAVE temporal configuration to a Cobre ``stages.json`` dict.

    Reads ``dger.dat`` and ``patamar.dat`` from *nw_files* and produces a
    dict that conforms to ``stages.schema.json``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map (not used for temporal conversion, accepted for API
        consistency with the other converters).

    Returns
    -------
    dict
        JSON-serializable dict conforming to ``stages.schema.json``.

    Raises
    ------
    ValueError
        If ``num_anos_estudo`` is 0 or None.
    """
    dger = Dger.read(nw_files.dger)
    patamar = Patamar.read(nw_files.patamar)

    num_anos = dger.num_anos_estudo
    if not num_anos:
        raise ValueError("NEWAVE case has zero study years")

    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_scenarios: int = dger.num_aberturas or 1
    taxa = dger.taxa_de_desconto or 0.0
    annual_discount_rate = float(taxa) / 100.0

    # ------------------------------------------------------------------
    # Read CVaR configuration from dger.dat and cvar.dat.
    # dger.cvar == 0 (or None): expectation
    # dger.cvar == 1: constant CVaR from cvar.dat::valores_constantes
    # dger.cvar == 2: temporal CVaR from cvar.dat::alfa_variavel / lambda_variavel
    # ------------------------------------------------------------------
    _raw_cvar = getattr(dger, "cvar", None)
    dger_cvar: int = int(_raw_cvar) if isinstance(_raw_cvar, int) else 0

    # Default: each stage uses "expectation"
    _cvar_by_stage: dict[
        int, dict
    ] = {}  # stage_id -> {"alpha": ..., "lambda": ...}

    if dger_cvar in (1, 2) and nw_files.cvar is not None:
        cvar_file = Cvar.read(str(nw_files.cvar))
        const_values: list[float] = cvar_file.valores_constantes or [0.0, 0.0]
        const_alpha = const_values[0] / 100.0
        const_lambda = const_values[1] / 100.0

        if dger_cvar == 1:
            # Same constant CVaR for ALL stages — populate lazily in loop below.
            _cvar_constant: dict | None = {
                "cvar": {"alpha": const_alpha, "lambda": const_lambda}
            }
        else:
            # dger_cvar == 2: build per-stage override maps from DataFrames.
            _cvar_constant = None
            df_alpha = cvar_file.alfa_variavel
            df_lambda = cvar_file.lambda_variavel

            # Build {(year, month): alpha_value_fraction} for study rows only.
            alpha_override: dict[tuple[int, int], float] = {}
            if df_alpha is not None and not df_alpha.empty:
                for _, row in df_alpha.iterrows():
                    y = int(row["data"].year)
                    m = int(row["data"].month)
                    if y < 9000:  # skip post-study sentinel rows (year 9999)
                        val = float(row["valor"])
                        alpha_override[(y, m)] = (
                            val / 100.0 if val != 0.0 else const_alpha
                        )

            lambda_override: dict[tuple[int, int], float] = {}
            if df_lambda is not None and not df_lambda.empty:
                for _, row in df_lambda.iterrows():
                    y = int(row["data"].year)
                    m = int(row["data"].month)
                    if y < 9000:
                        val = float(row["valor"])
                        lambda_override[(y, m)] = (
                            val / 100.0 if val != 0.0 else const_lambda
                        )
    elif dger_cvar in (1, 2) and nw_files.cvar is None:
        logger.warning(
            "dger.cvar=%d requires cvar.dat but file is missing; "
            "falling back to expectation for all stages.",
            dger_cvar,
        )
        dger_cvar = 0
        _cvar_constant = None
    else:
        _cvar_constant = None

    # Build block duration lookup: {(month, patamar) -> fraction}
    # Patamar.duracao_mensal_patamares columns: data (datetime), patamar (int),
    # valor (float, fraction of month).
    df_pat = patamar.duracao_mensal_patamares
    num_patamares: int = patamar.numero_patamares or 1

    # Index the patamar DataFrame by (year, calendar month, patamar index).
    # The ``data`` column is a datetime; we need year and calendar month.
    # Build a dict {(year, month_1_to_12, patamar_1_based) -> fraction}.
    pat_lookup: dict[tuple[int, int, int], float] = {}
    if df_pat is not None and not df_pat.empty:
        for _, row in df_pat.iterrows():
            cal_year = int(row["data"].year)
            cal_month = int(row["data"].month)
            pat_idx = int(row["patamar"])
            fraction = float(row["valor"])
            pat_lookup[(cal_year, cal_month, pat_idx)] = fraction

    names = _block_names(num_patamares)

    # ------------------------------------------------------------------
    # Build study stages (IDs 0 .. N-1).
    # ------------------------------------------------------------------
    stages: list[dict] = []
    transitions: list[dict] = []

    num_anos_pos = dger.num_anos_pos_estudo or 0
    # Study runs from mes_inicio to December of (ano_inicio + num_anos - 1).
    study_months = (13 - start_month) + (num_anos - 1) * 12
    # Post-study adds num_anos_pos full calendar years after that.
    pos_months = num_anos_pos * 12
    total_months = study_months + pos_months

    year = start_year
    month = start_month
    for stage_id in range(total_months):
        start_date = _month_start_date(year, month)
        end_date = _month_end_date(year, month)
        total_hours = _month_hours(year, month)

        blocks: list[dict] = []
        for pat_idx in range(1, num_patamares + 1):
            fraction = pat_lookup.get((year, month, pat_idx))
            if fraction is None:
                fraction = 1.0 / num_patamares
                logger.warning(
                    "No patamar duration for year %d, calendar month %d, "
                    "patamar %d; using equal fraction %.4f",
                    year,
                    month,
                    pat_idx,
                    fraction,
                )
            block_hours = fraction * total_hours
            blocks.append({
                "id": pat_idx - 1,
                "name": names[pat_idx - 1],
                "hours": block_hours,
            })

        # Determine risk_measure for this stage.
        if dger_cvar == 0 or _cvar_constant is None and dger_cvar != 2:
            risk_measure: str | dict = "expectation"
        elif dger_cvar == 1 and _cvar_constant is not None:
            risk_measure = _cvar_constant
        else:
            # dger_cvar == 2: use per-stage alpha/lambda, falling back to constant.
            a = alpha_override.get((year, month), const_alpha)
            lbd = lambda_override.get((year, month), const_lambda)
            risk_measure = {"cvar": {"alpha": a, "lambda": lbd}}

        stages.append({
            "id": stage_id,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "season_id": month - 1,  # 0-based: Jan=0, ..., Dec=11
            "blocks": blocks,
            "num_scenarios": num_scenarios,
            "risk_measure": risk_measure,
            "state_variables": {
                "storage": True,
                "inflow_lags": True,
            },
        })

        if stage_id < total_months - 1:
            transitions.append({
                "source_id": stage_id,
                "target_id": stage_id + 1,
                "probability": 1.0,
            })

        month += 1
        if month > 12:
            month = 1
            year += 1

    # ------------------------------------------------------------------
    # Build pre-study stages (IDs -N .. -1), if requested.
    # ------------------------------------------------------------------
    pre_study_stages: list[dict] = []
    num_anos_pre = dger.num_anos_pre_estudo or 0
    if num_anos_pre > 0:
        num_pre_months = num_anos_pre * 12
        pre_year = start_year
        pre_month = start_month
        # Walk backwards num_pre_months months from the study start.
        pre_dates: list[tuple[int, int]] = []
        for _ in range(num_pre_months):
            pre_month -= 1
            if pre_month < 1:
                pre_month = 12
                pre_year -= 1
            pre_dates.append((pre_year, pre_month))

        # pre_dates is in reverse order; reverse it so IDs go -N..-1.
        pre_dates.reverse()

        for offset, (py, pm) in enumerate(pre_dates):
            pre_id = -(num_pre_months - offset)
            pre_study_stages.append({
                "id": pre_id,
                "start_date": _month_start_date(py, pm).isoformat(),
                "end_date": _month_end_date(py, pm).isoformat(),
                "season_id": pm - 1,  # 0-based calendar month index
            })

    policy_graph: dict = {
        "type": "finite_horizon",
        "annual_discount_rate": annual_discount_rate,
        "transitions": transitions,
    }

    _MONTH_LABELS = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    season_definitions: dict = {
        "cycle_type": "monthly",
        "seasons": [
            {"id": i, "month_start": i + 1, "label": _MONTH_LABELS[i]}
            for i in range(12)
        ],
    }

    result: dict = {
        "$schema": (
            "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
            "/book/src/schemas/stages.schema.json"
        ),
        "season_definitions": season_definitions,
        "policy_graph": policy_graph,
        "stages": stages,
    }
    if pre_study_stages:
        result["pre_study_stages"] = pre_study_stages

    return result


def convert_config(nw_files: NewaveFiles) -> dict:
    """Convert NEWAVE training parameters to a Cobre ``config.json`` dict.

    Reads ``dger.dat`` from *nw_files* and produces a dict that conforms
    to ``config.schema.json``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.

    Returns
    -------
    dict
        JSON-serializable dict conforming to ``config.schema.json``.
    """
    dger = Dger.read(nw_files.dger)

    forward_passes: int = dger.num_forwards or 1
    max_iterations: int = dger.num_max_iteracoes or 200
    num_series: int = dger.num_series_sinteticas or 200
    max_order: int = dger.ordem_maxima_parp or 6

    tipo_execucao: int = (
        dger.tipo_execucao if dger.tipo_execucao is not None else 1
    )
    tipo_simulacao_final: int = (
        dger.tipo_simulacao_final
        if dger.tipo_simulacao_final is not None
        else 1
    )
    considera_reamostragem: int = (
        dger.considera_reamostragem_cenarios
        if dger.considera_reamostragem_cenarios is not None
        else 0
    )

    # tipo_execucao: 0 = simulation only, 1 = training (+ simulation).
    training_enabled: bool = tipo_execucao == 1

    # tipo_simulacao_final: 0 = disabled, 1 = out_of_sample, 2 = historical.
    # When tipo_execucao == 0 (simulation-only mode), simulation is always on.
    if tipo_execucao == 0:
        simulation_enabled = True
    else:
        simulation_enabled = tipo_simulacao_final != 0

    # -- Training scenario source --
    training_section: dict = {
        "forward_passes": forward_passes,
        "stopping_rules": [
            {"type": "iteration_limit", "limit": max_iterations},
        ],
        "cut_selection": {
            "check_frequency": 1,
            "cut_activity_tolerance": 1e-6,
            "enabled": True,
            "method": "domination",
            "threshold": 0,
        },
    }
    if not training_enabled:
        training_section["enabled"] = False
    if training_enabled and considera_reamostragem == 1:
        training_section["scenario_source"] = {
            "inflow": {"scheme": "out_of_sample"},
        }

    # -- Simulation scenario source --
    # Priority: tipo_simulacao_final=2 forces "historical"; otherwise
    # considera_reamostragem determines the scheme (1=out_of_sample, 0=in_sample).
    simulation_section: dict = {
        "enabled": simulation_enabled,
        "num_scenarios": num_series,
    }
    if simulation_enabled:
        if tipo_simulacao_final == 2:
            inflow_scheme = "historical"
        elif considera_reamostragem == 1:
            inflow_scheme = "out_of_sample"
        else:
            inflow_scheme = "in_sample"
        sim_source: dict = {"inflow": {"scheme": inflow_scheme}}
        if inflow_scheme == "historical":
            ano_ini_hist: int = dger.ano_inicial_historico or 1931
            ano_inicio: int = dger.ano_inicio_estudo or 2020
            sim_source["historical_years"] = {
                "from": ano_ini_hist + 1,
                "to": ano_inicio - 1,
            }
        simulation_section["scenario_source"] = sim_source

    config: dict = {
        "$schema": (
            "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
            "/book/src/schemas/config.schema.json"
        ),
        "estimation": {
            "max_order": max_order,
        },
        "training": training_section,
        "modeling": {
            "inflow_non_negativity": {
                "method": "penalty",
                "penalty_cost": 10000.0,
            },
        },
        "exports": {
            "stochastic": True,
        },
        "simulation": simulation_section,
    }

    return config
