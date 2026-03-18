"""Temporal data converter: maps NEWAVE study horizon configuration to Cobre JSON.

Converts ``dger.dat`` and ``patamar.dat`` into the ``stages.json`` and
``config.json`` formats expected by the Cobre solver.
"""

from __future__ import annotations

import calendar
import logging
from datetime import date
from pathlib import Path

from inewave.newave import Dger, Patamar

from cobre_bridge.id_map import NewaveIdMap

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


def convert_stages(newave_dir: Path, id_map: NewaveIdMap) -> dict:  # noqa: ARG001
    """Convert NEWAVE temporal configuration to a Cobre ``stages.json`` dict.

    Reads ``dger.dat`` and ``patamar.dat`` from *newave_dir* and produces a
    dict that conforms to ``stages.schema.json``.

    Parameters
    ----------
    newave_dir:
        Path to the directory containing NEWAVE input files.
    id_map:
        Entity ID map (not used for temporal conversion, accepted for API
        consistency with the other converters).

    Returns
    -------
    dict
        JSON-serializable dict conforming to ``stages.schema.json``.

    Raises
    ------
    FileNotFoundError
        If ``dger.dat`` or ``patamar.dat`` are absent.
    ValueError
        If ``num_anos_estudo`` is 0 or None.
    """
    dger_path = newave_dir / "dger.dat"
    patamar_path = newave_dir / "patamar.dat"

    if not dger_path.exists():
        raise FileNotFoundError(f"dger.dat not found in {newave_dir}")
    if not patamar_path.exists():
        raise FileNotFoundError(f"patamar.dat not found in {newave_dir}")

    dger = Dger.read(dger_path)
    patamar = Patamar.read(patamar_path)

    num_anos = dger.num_anos_estudo
    if not num_anos:
        raise ValueError("NEWAVE case has zero study years")

    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_scenarios: int = dger.num_aberturas or 1
    taxa = dger.taxa_de_desconto or 0.0
    annual_discount_rate = float(taxa) / 100.0

    # Build block duration lookup: {(month, patamar) -> fraction}
    # Patamar.duracao_mensal_patamares columns: data (datetime), patamar (int),
    # valor (float, fraction of month).
    df_pat = patamar.duracao_mensal_patamares
    num_patamares: int = patamar.numero_patamares or 1

    # Index the patamar DataFrame by (calendar month, patamar index).
    # The ``data`` column is a datetime; we need the calendar month (1-12).
    # Build a dict {(month_1_to_12, patamar_1_based) -> fraction}.
    pat_lookup: dict[tuple[int, int], float] = {}
    if df_pat is not None and not df_pat.empty:
        for _, row in df_pat.iterrows():
            cal_month = int(row["data"].month)
            pat_idx = int(row["patamar"])
            fraction = float(row["valor"])
            pat_lookup[(cal_month, pat_idx)] = fraction

    names = _block_names(num_patamares)

    # ------------------------------------------------------------------
    # Build study stages (IDs 0 .. N-1).
    # ------------------------------------------------------------------
    stages: list[dict] = []
    transitions: list[dict] = []

    year = start_year
    month = start_month
    for stage_id in range(num_anos * 12):
        start_date = _month_start_date(year, month)
        end_date = _month_end_date(year, month)
        total_hours = _month_hours(year, month)

        blocks: list[dict] = []
        for pat_idx in range(1, num_patamares + 1):
            fraction = pat_lookup.get((month, pat_idx))
            if fraction is None:
                # Fall back: distribute evenly across blocks.
                fraction = 1.0 / num_patamares
                logger.warning(
                    "No patamar duration for calendar month %d, patamar %d; "
                    "using equal fraction %.4f",
                    month,
                    pat_idx,
                    fraction,
                )
            block_hours = fraction * total_hours
            blocks.append(
                {
                    "id": pat_idx - 1,
                    "name": names[pat_idx - 1],
                    "hours": block_hours,
                }
            )

        stages.append(
            {
                "id": stage_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "blocks": blocks,
                "num_scenarios": num_scenarios,
                "risk_measure": "expectation",
            }
        )

        # Linear transition from this stage to the next.
        if stage_id < num_anos * 12 - 1:
            transitions.append(
                {
                    "source_id": stage_id,
                    "target_id": stage_id + 1,
                    "probability": 1.0,
                }
            )

        # Advance calendar month.
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
            pre_study_stages.append(
                {
                    "id": pre_id,
                    "start_date": _month_start_date(py, pm).isoformat(),
                    "end_date": _month_end_date(py, pm).isoformat(),
                    "season_id": pm - 1,  # 0-based calendar month index
                }
            )

    policy_graph: dict = {
        "type": "finite_horizon",
        "annual_discount_rate": annual_discount_rate,
        "transitions": transitions,
    }

    result: dict = {
        "$schema": (
            "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
            "/book/src/schemas/stages.schema.json"
        ),
        "policy_graph": policy_graph,
        "stages": stages,
    }
    if pre_study_stages:
        result["pre_study_stages"] = pre_study_stages

    return result


def convert_config(newave_dir: Path) -> dict:
    """Convert NEWAVE training parameters to a Cobre ``config.json`` dict.

    Reads ``dger.dat`` from *newave_dir* and produces a dict that conforms
    to ``config.schema.json``.

    Parameters
    ----------
    newave_dir:
        Path to the directory containing NEWAVE input files.

    Returns
    -------
    dict
        JSON-serializable dict conforming to ``config.schema.json``.

    Raises
    ------
    FileNotFoundError
        If ``dger.dat`` is absent.
    """
    dger_path = newave_dir / "dger.dat"
    if not dger_path.exists():
        raise FileNotFoundError(f"dger.dat not found in {newave_dir}")

    dger = Dger.read(dger_path)

    forward_passes: int = dger.num_forwards or 1
    max_iterations: int = dger.num_max_iteracoes or 200
    num_series: int = dger.num_series_sinteticas or 200

    return {
        "$schema": (
            "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
            "/book/src/schemas/config.schema.json"
        ),
        "training": {
            "forward_passes": forward_passes,
            "stopping_rules": [
                {"type": "iteration_limit", "limit": max_iterations},
            ],
        },
        "simulation": {
            "enabled": True,
            "num_scenarios": num_series,
        },
    }
