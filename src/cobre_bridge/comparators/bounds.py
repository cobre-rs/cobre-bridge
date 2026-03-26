"""Core bounds comparison between sintetizador Parquets and Cobre bounds.parquet.

Reads both data sources, aligns entities via EntityAlignment, and produces
a list of BoundComparison results for every (entity, stage, variable) triple.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cobre_bridge.comparators.alignment import (
    EntityAlignment,
)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundComparison:
    """Single bound comparison result."""

    entity_type: str
    entity_name: str
    newave_code: int
    cobre_id: int
    stage: int
    variable: str
    newave_value: float
    cobre_value: float
    diff: float
    match: bool


# --- Sintetizador variable -> Cobre bound_type_code mapping ---

# Hydro: (sintetizador_variable, "inferior"/"superior") -> cobre bound_type_code
_HYDRO_VARIABLE_MAP: list[tuple[str, str, int, str]] = [
    # (nw_variable, nw_bound_col, cobre_bound_type, comparison_name)
    ("VARMF", "limite_inferior", 0, "storage_min"),
    ("VARMF", "limite_superior", 1, "storage_max"),
    ("QTUR", "limite_inferior", 2, "turbined_min"),
    ("QTUR", "limite_superior", 3, "turbined_max"),
    ("QDEF", "limite_inferior", 4, "outflow_min"),
]

# Thermal: patamar=0 MWmed values
_THERMAL_VARIABLE_MAP: list[tuple[str, int, str]] = [
    # (nw_bound_col, cobre_bound_type, comparison_name)
    ("limite_inferior", 6, "generation_min"),
    ("limite_superior", 7, "generation_max"),
]

# Line bound type codes in Cobre
_LINE_FLOW_MIN = 8
_LINE_FLOW_MAX = 9


# NEWAVE uses 99999 as a "big M" sentinel meaning "no limit".
_NEWAVE_BIG_M = 99990.0


def _is_effectively_infinite(value: float) -> bool:
    """Return True if the value represents an unbounded variable.

    Catches both IEEE inf and NEWAVE's 99999 sentinel.
    """
    return math.isinf(value) or abs(value) >= _NEWAVE_BIG_M


def _bounds_match(a: float, b: float, tolerance: float) -> bool:
    """Check if two bound values match within tolerance.

    Both-infinite with same sign counts as a match.
    One finite and one infinite is always a mismatch.
    """
    if math.isinf(a) and math.isinf(b):
        return (a > 0) == (b > 0)
    if math.isinf(a) or math.isinf(b):
        return False
    return abs(a - b) <= tolerance


def _read_sintetizador_uhe(
    sintese_dir: Path, num_stages: int
) -> dict[tuple[int, int, str], tuple[float, float]]:
    """Read UHE sintetizador data.

    Returns {(code, stage_0based, variable): (lim_inf, lim_sup)}.

    Filters cenario=="mean", patamar==0, stages within range.
    """
    path = sintese_dir / "ESTATISTICAS_OPERACAO_UHE.parquet"
    if not path.exists():
        _LOG.warning("ESTATISTICAS_OPERACAO_UHE.parquet not found in %s", sintese_dir)
        return {}

    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("cenario") == "mean")
        & (pl.col("patamar") == 0)
        & (pl.col("estagio") >= 1)
        & (pl.col("estagio") <= num_stages)
    )

    result: dict[tuple[int, int, str], tuple[float, float]] = {}
    for row in df.iter_rows(named=True):
        code = int(row["codigo_usina"])
        stage_0 = int(row["estagio"]) - 1  # convert to 0-based
        var = str(row["variavel"])
        li = float(row["limite_inferior"])
        ls = float(row["limite_superior"])
        result[(code, stage_0, var)] = (li, ls)

    return result


def _read_sintetizador_ute(
    sintese_dir: Path, num_stages: int
) -> dict[tuple[int, int], tuple[float, float]]:
    """Read UTE sintetizador data, return {(code, stage_0based): (lim_inf, lim_sup)}.

    Only GTER variable, cenario=="mean", patamar==0.
    """
    path = sintese_dir / "ESTATISTICAS_OPERACAO_UTE.parquet"
    if not path.exists():
        _LOG.warning("ESTATISTICAS_OPERACAO_UTE.parquet not found in %s", sintese_dir)
        return {}

    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("cenario") == "mean")
        & (pl.col("patamar") == 0)
        & (pl.col("estagio") >= 1)
        & (pl.col("estagio") <= num_stages)
    )

    result: dict[tuple[int, int], tuple[float, float]] = {}
    for row in df.iter_rows(named=True):
        code = int(row["codigo_usina"])
        stage_0 = int(row["estagio"]) - 1
        li = float(row["limite_inferior"])
        ls = float(row["limite_superior"])
        result[(code, stage_0)] = (li, ls)

    return result


def _read_sintetizador_sbp(
    sintese_dir: Path, num_stages: int
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Read SBP sintetizador data.

    Returns {(de, para, stage_0based): (lim_inf, lim_sup)}.

    INT variable, cenario=="mean", patamar==0.
    """
    path = sintese_dir / "ESTATISTICAS_OPERACAO_SBP.parquet"
    if not path.exists():
        _LOG.warning("ESTATISTICAS_OPERACAO_SBP.parquet not found in %s", sintese_dir)
        return {}

    df = pl.read_parquet(path)
    df = df.filter(
        (pl.col("cenario") == "mean")
        & (pl.col("patamar") == 0)
        & (pl.col("estagio") >= 1)
        & (pl.col("estagio") <= num_stages)
    )

    result: dict[tuple[int, int, int], tuple[float, float]] = {}
    for row in df.iter_rows(named=True):
        de = int(row["codigo_submercado_de"])
        para = int(row["codigo_submercado_para"])
        stage_0 = int(row["estagio"]) - 1
        li = float(row["limite_inferior"])
        ls = float(row["limite_superior"])
        result[(de, para, stage_0)] = (li, ls)

    return result


def _read_cobre_bounds(
    cobre_output_dir: Path,
) -> dict[tuple[int, int, int, int], float]:
    """Read Cobre bounds.parquet.

    Returns {(entity_type, entity_id, stage_id, bound_type): value}.

    Only reads rows with block_id IS NULL (stage-level bounds).
    """
    path = cobre_output_dir / "training" / "dictionaries" / "bounds.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cobre bounds.parquet not found at {path}. Run cobre with --output first."
        )

    df = pl.read_parquet(path)
    df = df.filter(pl.col("block_id").is_null())

    result: dict[tuple[int, int, int, int], float] = {}
    for row in df.iter_rows(named=True):
        key = (
            int(row["entity_type_code"]),
            int(row["entity_id"]),
            int(row["stage_id"]),
            int(row["bound_type_code"]),
        )
        result[key] = float(row["bound_value"])

    return result


def _compare_hydros(
    alignment: EntityAlignment,
    nw_uhe: dict[tuple[int, int, str], tuple[float, float]],
    cobre_bounds: dict[tuple[int, int, int, int], float],
    tolerance: float,
    variables: set[str] | None,
) -> list[BoundComparison]:
    """Compare hydro bounds for all aligned plants and stages."""
    results: list[BoundComparison] = []

    for hydro in alignment.hydros:
        for stage in range(alignment.num_newave_stages):
            for nw_var, nw_col, cobre_bt, comp_name in _HYDRO_VARIABLE_MAP:
                if variables is not None and comp_name not in variables:
                    continue

                # Storage comparisons only for reservoir plants
                if nw_var == "VARMF" and not hydro.has_reservoir:
                    continue

                nw_key = (hydro.newave_code, stage, nw_var)
                nw_bounds = nw_uhe.get(nw_key)
                if nw_bounds is None:
                    continue

                nw_value = nw_bounds[0] if nw_col == "limite_inferior" else nw_bounds[1]

                # Skip effectively-infinite NEWAVE bounds (inf or 99999 sentinel)
                if _is_effectively_infinite(nw_value):
                    continue

                cobre_key = (0, hydro.cobre_id, stage, cobre_bt)
                cobre_value = cobre_bounds.get(cobre_key)
                if cobre_value is None:
                    continue

                matched = _bounds_match(nw_value, cobre_value, tolerance)
                diff = (
                    abs(nw_value - cobre_value)
                    if not (math.isinf(nw_value) or math.isinf(cobre_value))
                    else float("inf")
                )

                results.append(
                    BoundComparison(
                        entity_type="hydro",
                        entity_name=hydro.name,
                        newave_code=hydro.newave_code,
                        cobre_id=hydro.cobre_id,
                        stage=stage,
                        variable=comp_name,
                        newave_value=nw_value,
                        cobre_value=cobre_value,
                        diff=diff,
                        match=matched,
                    )
                )

    return results


def _compare_thermals(
    alignment: EntityAlignment,
    nw_ute: dict[tuple[int, int], tuple[float, float]],
    cobre_bounds: dict[tuple[int, int, int, int], float],
    tolerance: float,
    variables: set[str] | None,
) -> list[BoundComparison]:
    """Compare thermal generation bounds."""
    results: list[BoundComparison] = []

    for thermal in alignment.thermals:
        for stage in range(alignment.num_newave_stages):
            nw_key = (thermal.newave_code, stage)
            nw_bounds = nw_ute.get(nw_key)
            if nw_bounds is None:
                continue

            for nw_col, cobre_bt, comp_name in _THERMAL_VARIABLE_MAP:
                if variables is not None and comp_name not in variables:
                    continue

                nw_value = nw_bounds[0] if nw_col == "limite_inferior" else nw_bounds[1]

                if _is_effectively_infinite(nw_value):
                    continue

                cobre_key = (1, thermal.cobre_id, stage, cobre_bt)
                cobre_value = cobre_bounds.get(cobre_key)
                if cobre_value is None:
                    continue

                matched = _bounds_match(nw_value, cobre_value, tolerance)
                diff = (
                    abs(nw_value - cobre_value)
                    if not (math.isinf(nw_value) or math.isinf(cobre_value))
                    else float("inf")
                )

                results.append(
                    BoundComparison(
                        entity_type="thermal",
                        entity_name=thermal.name,
                        newave_code=thermal.newave_code,
                        cobre_id=thermal.cobre_id,
                        stage=stage,
                        variable=comp_name,
                        newave_value=nw_value,
                        cobre_value=cobre_value,
                        diff=diff,
                        match=matched,
                    )
                )

    return results


def _compare_lines(
    alignment: EntityAlignment,
    nw_sbp: dict[tuple[int, int, int], tuple[float, float]],
    cobre_bounds: dict[tuple[int, int, int, int], float],
    tolerance: float,
    variables: set[str] | None,
) -> list[BoundComparison]:
    """Compare exchange line bounds in all directions.

    For each Cobre line (src_bus -> tgt_bus) mapped to NEWAVE pair (de, para):

    NEWAVE INT(de, para):
      - limite_superior = max flow de->para (positive MW)
      - limite_inferior = max flow para->de (negative MW, so abs = capacity)

    Cobre line (src=de, tgt=para):
      - flow_max (bound 9) = max forward flow (src->tgt)
      - flow_min (bound 8) = min forward flow (usually 0 for unrestricted)

    We also check the reverse NEWAVE pair (para, de) if it exists,
    which reports the same exchange from the other subsystem's perspective.
    """
    results: list[BoundComparison] = []

    for line in alignment.lines:
        for stage in range(alignment.num_newave_stages):
            # Forward direction: NEWAVE (de, para)
            nw_key_fwd = (line.newave_de, line.newave_para, stage)
            nw_fwd = nw_sbp.get(nw_key_fwd)

            if nw_fwd is not None:
                nw_lim_inf, nw_lim_sup = nw_fwd

                # Compare forward flow_max: NEWAVE lim_sup vs Cobre flow_max
                if variables is None or "flow_max" in variables:
                    if not _is_effectively_infinite(nw_lim_sup):
                        cobre_key = (3, line.cobre_line_id, stage, _LINE_FLOW_MAX)
                        cobre_value = cobre_bounds.get(cobre_key)
                        if cobre_value is not None:
                            matched = _bounds_match(nw_lim_sup, cobre_value, tolerance)
                            results.append(
                                BoundComparison(
                                    entity_type="line",
                                    entity_name=f"{line.name} fwd",
                                    newave_code=line.newave_de,
                                    cobre_id=line.cobre_line_id,
                                    stage=stage,
                                    variable="flow_max",
                                    newave_value=nw_lim_sup,
                                    cobre_value=cobre_value,
                                    diff=abs(nw_lim_sup - cobre_value),
                                    match=matched,
                                )
                            )

                # Compare reverse capacity: NEWAVE abs(lim_inf) vs Cobre flow_min
                # NEWAVE lim_inf is negative (para->de direction).
                # Cobre flow_min is the lower bound on forward flow.
                # For a line modeled with flow >= 0, reverse capacity is
                # handled by a separate mechanism — but we compare abs(lim_inf)
                # against the Cobre flow_min to detect asymmetry.
                if variables is None or "flow_min" in variables:
                    if not _is_effectively_infinite(nw_lim_inf):
                        cobre_key = (3, line.cobre_line_id, stage, _LINE_FLOW_MIN)
                        cobre_value = cobre_bounds.get(cobre_key)
                        if cobre_value is not None:
                            # NEWAVE negative lim_inf means reverse capacity.
                            # Cobre flow_min=0 means no reverse flow on this line.
                            # The reverse capacity may be on a separate Cobre line
                            # or modeled differently. We report the raw comparison.
                            nw_reverse_cap = abs(nw_lim_inf)
                            matched = _bounds_match(
                                nw_reverse_cap, cobre_value, tolerance
                            )
                            results.append(
                                BoundComparison(
                                    entity_type="line",
                                    entity_name=f"{line.name} rev",
                                    newave_code=line.newave_para,
                                    cobre_id=line.cobre_line_id,
                                    stage=stage,
                                    variable="flow_min_reverse",
                                    newave_value=nw_reverse_cap,
                                    cobre_value=cobre_value,
                                    diff=abs(nw_reverse_cap - cobre_value),
                                    match=matched,
                                )
                            )

    return results


def compare_bounds(
    alignment: EntityAlignment,
    sintese_dir: Path,
    cobre_output_dir: Path,
    tolerance: float = 1e-3,
    variables: set[str] | None = None,
) -> list[BoundComparison]:
    """Compare LP variable bounds between NEWAVE and Cobre.

    Parameters
    ----------
    alignment:
        Pre-built entity alignment.
    sintese_dir:
        Path to sintetizador output directory.
    cobre_output_dir:
        Path to Cobre output directory.
    tolerance:
        Absolute tolerance for bound comparison.
    variables:
        Optional set of variable names to include. None means all.

    Returns
    -------
    list[BoundComparison]
        All comparison results, including matches and mismatches.
    """
    num_stages = alignment.num_newave_stages
    _LOG.info(
        "Comparing bounds: %d hydros, %d thermals, %d lines, %d stages, tol=%g",
        len(alignment.hydros),
        len(alignment.thermals),
        len(alignment.lines),
        num_stages,
        tolerance,
    )

    _LOG.info("Reading sintetizador data...")
    nw_uhe = _read_sintetizador_uhe(sintese_dir, num_stages)
    nw_ute = _read_sintetizador_ute(sintese_dir, num_stages)
    nw_sbp = _read_sintetizador_sbp(sintese_dir, num_stages)
    _LOG.info(
        "Sintetizador: %d UHE entries, %d UTE entries, %d SBP entries",
        len(nw_uhe),
        len(nw_ute),
        len(nw_sbp),
    )

    _LOG.info("Reading Cobre bounds...")
    cobre_bounds = _read_cobre_bounds(cobre_output_dir)
    _LOG.info("Cobre: %d bound entries", len(cobre_bounds))

    results: list[BoundComparison] = []

    _LOG.info("Comparing hydro bounds...")
    results.extend(
        _compare_hydros(alignment, nw_uhe, cobre_bounds, tolerance, variables)
    )

    _LOG.info("Comparing thermal bounds...")
    results.extend(
        _compare_thermals(alignment, nw_ute, cobre_bounds, tolerance, variables)
    )

    _LOG.info("Comparing line bounds...")
    results.extend(
        _compare_lines(alignment, nw_sbp, cobre_bounds, tolerance, variables)
    )

    return results
