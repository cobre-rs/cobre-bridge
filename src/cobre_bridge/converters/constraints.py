"""Generic constraints converter: VminOP (minimum stored energy) from curva.dat
and electric constraints from restricao-eletrica.csv.

Converts NEWAVE minimum stored energy constraints into Cobre generic constraints.
Each REE with entries in ``curva.dat`` becomes one generic constraint whose
expression is a weighted sum of ``hydro_storage`` variables, with weights equal
to the accumulated cascade productivities.

Electric constraints from ``restricao-eletrica.csv`` (discovered via
``indices.csv``) are converted into Cobre generic constraints with
``hydro_generation`` and ``line_exchange`` variables.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
from inewave.newave import Confhd, Curva, Dger, Hidr, Penalid, Ree, Sistema

from cobre_bridge.converters.hydro import (
    _apply_permanent_overrides,
    _compute_productivity,
)
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/generic_constraints.schema.json"
)


def _build_hydro_downstream_map(
    confhd_df: pd.DataFrame,
) -> dict[int, int | None]:
    """Return {plant_code: downstream_code} from confhd.

    A downstream_code of ``None`` means the plant discharges to the sea.
    Only existing, non-fictitious plants are included.
    """
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]

    result: dict[int, int | None] = {}
    for _, row in non_fict.iterrows():
        code = int(row["codigo_usina"])
        ds_raw = row.get("codigo_usina_jusante")
        if ds_raw is not None and not pd.isna(ds_raw) and int(ds_raw) != 0:
            ds_code = int(ds_raw)
            # Only reference valid study plants
            if ds_code in non_fict["codigo_usina"].values:
                result[code] = ds_code
            else:
                result[code] = None
        else:
            result[code] = None
    return result


def _build_hydro_to_ree(confhd_df: pd.DataFrame) -> dict[int, int]:
    """Return {plant_code: ree_code} for existing non-fictitious plants."""
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]
    return {int(r["codigo_usina"]): int(r["ree"]) for _, r in non_fict.iterrows()}


def compute_accumulated_productivities(
    cadastro: pd.DataFrame,
    confhd_df: pd.DataFrame,
) -> dict[int, float]:
    """Compute accumulated cascade productivity for each hydro plant.

    The accumulated productivity of a plant is its own productivity plus the
    accumulated productivity of its downstream plant.  This is computed by
    traversing the cascade DAG from downstream (sea-level sinks) to upstream.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame (with MODIF.DAT overrides applied).
    confhd_df:
        The ``Confhd.usinas`` DataFrame.

    Returns
    -------
    dict[int, float]
        Mapping from plant code to accumulated productivity in MW/(m³/s).
    """
    downstream_map = _build_hydro_downstream_map(confhd_df)
    plant_codes = list(downstream_map.keys())

    # Compute own productivity for each plant
    own_prod: dict[int, float] = {}
    for code in plant_codes:
        if code in cadastro.index:
            own_prod[code] = _compute_productivity(cadastro.loc[code])
        else:
            own_prod[code] = 0.0

    # Build children map (upstream plants for each plant)
    children: dict[int | None, list[int]] = defaultdict(list)
    for code, ds in downstream_map.items():
        children[ds].append(code)

    # BFS from sinks (downstream=None) upward
    acc_prod: dict[int, float] = {}

    def _accumulate(code: int) -> float:
        if code in acc_prod:
            return acc_prod[code]
        ds = downstream_map.get(code)
        ds_acc = _accumulate(ds) if ds is not None else 0.0
        acc_prod[code] = own_prod.get(code, 0.0) + ds_acc
        return acc_prod[code]

    for code in plant_codes:
        _accumulate(code)

    return acc_prod


def convert_vminop_constraints(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> tuple[dict, pa.Table] | None:
    """Convert curva.dat VminOP constraints to Cobre generic constraints.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths.
    id_map:
        Entity ID mapping.

    Returns
    -------
    tuple[dict, pa.Table] | None
        A ``(constraints_dict, bounds_table)`` pair, or ``None`` if
        ``curva.dat`` is absent.  ``constraints_dict`` conforms to
        ``generic_constraints.schema.json``; ``bounds_table`` has schema
        ``(constraint_id: INT32, stage_id: INT32, bound: DOUBLE)``.
    """
    if nw_files.curva is None:
        _LOG.debug("curva.dat not found; skipping VminOP constraints.")
        return None

    curva = Curva.read(str(nw_files.curva))
    curva_df = curva.curva_seguranca
    if curva_df is None or curva_df.empty:
        return None

    penalty_df = curva.custos_penalidades

    # Read supporting data
    dger = Dger.read(nw_files.dger)
    confhd = Confhd.read(str(nw_files.confhd))
    hidr = Hidr.read(str(nw_files.hidr))
    ree_file = Ree.read(str(nw_files.ree))

    cadastro = hidr.cadastro
    cadastro = _apply_permanent_overrides(cadastro, nw_files)
    confhd_df = confhd.usinas

    # Study horizon parameters
    start_month = dger.mes_inicio_estudo
    start_year = dger.ano_inicio_estudo
    num_anos = dger.num_anos_estudo or 1
    num_anos_pos = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    num_stages = study_months + num_anos_pos * 12

    # Compute accumulated productivities
    acc_prod = compute_accumulated_productivities(cadastro, confhd_df)

    # Map hydros to REEs
    hydro_to_ree = _build_hydro_to_ree(confhd_df)

    # Group hydros by REE
    ree_hydros: dict[int, list[int]] = defaultdict(list)
    for code, ree_code in hydro_to_ree.items():
        ree_hydros[ree_code].append(code)

    # Build penalty lookup: ree_code -> penalty value
    penalty_map: dict[int, float] = {}
    if penalty_df is not None and not penalty_df.empty:
        for _, row in penalty_df.iterrows():
            penalty_map[int(row["codigo_ree"])] = float(row["penalidade"])

    # REE names from ree.dat
    ree_names: dict[int, str] = {}
    ree_df = ree_file.rees
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_names[int(row["codigo"])] = str(row["nome"]).strip()

    # REEs that have constraints in curva.dat
    constraint_rees = sorted(curva_df["codigo_ree"].unique())

    constraints: list[dict] = []
    bound_constraint_ids: list[int] = []
    bound_stage_ids: list[int] = []
    bound_values: list[float] = []

    for constraint_id, ree_code in enumerate(constraint_rees):
        ree_code = int(ree_code)
        hydros_in_ree = ree_hydros.get(ree_code, [])

        if not hydros_in_ree:
            _LOG.warning(
                "REE %d has VminOP constraints but no hydro plants; skipping.",
                ree_code,
            )
            continue

        # Build expression: sum of acc_prod * hydro_storage(cobre_id)
        terms: list[str] = []
        useful_energy = 0.0
        dead_energy = 0.0

        for plant_code in sorted(hydros_in_ree):
            ap = acc_prod.get(plant_code, 0.0)
            if ap <= 0.0:
                continue

            try:
                cobre_id = id_map.hydro_id(plant_code)
            except KeyError:
                continue

            vol_min = float(cadastro.loc[plant_code, "volume_minimo"])
            vol_max = float(cadastro.loc[plant_code, "volume_maximo"])
            useful_energy += ap * (vol_max - vol_min)
            dead_energy += ap * vol_min
            terms.append(f"{ap} * hydro_storage({cobre_id})")

        max_energy = useful_energy + dead_energy
        if not terms or max_energy <= 0.0:
            _LOG.warning(
                "REE %d (%s): no valid terms for VminOP expression; skipping.",
                ree_code,
                ree_names.get(ree_code, "?"),
            )
            continue

        expression = " + ".join(terms)
        penalty = penalty_map.get(ree_code, 1000.0)
        ree_name = ree_names.get(ree_code, str(ree_code))

        constraints.append(
            {
                "id": constraint_id,
                "name": f"VminOP_{ree_name}",
                "description": (
                    f"Minimum stored energy for REE {ree_code} ({ree_name})"
                ),
                "expression": expression,
                "sense": ">=",
                "slack": {"enabled": True, "penalty": penalty},
            }
        )

        # Build per-stage bounds from curva_df.
        # curva.dat only covers the study period.  For post-study stages we
        # extrapolate seasonally using the last year's percentages (the same
        # approach used for RE constraints).
        ree_curva = curva_df[curva_df["codigo_ree"] == ree_code].sort_values("data")

        # First pass: collect bounds and build seasonal map for extrapolation
        seasonal_pct: dict[int, float] = {}  # calendar_month -> last percentage
        for _, crow in ree_curva.iterrows():
            dt: datetime = crow["data"]
            stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
            if stage_id < 0 or stage_id >= num_stages:
                continue

            percentage = float(crow["valor"])
            rhs = (percentage / 100.0) * useful_energy + dead_energy

            bound_constraint_ids.append(constraint_id)
            bound_stage_ids.append(stage_id)
            bound_values.append(rhs)

            # Track the percentage per calendar month (last value wins)
            seasonal_pct[dt.month] = percentage

        # Second pass: extrapolate to post-study stages using seasonal pattern
        if seasonal_pct:
            covered = {
                (dt.year - start_year) * 12 + (dt.month - start_month)
                for _, dt in ree_curva["data"].items()
                if 0
                <= (dt.year - start_year) * 12 + (dt.month - start_month)
                < num_stages
            }
            for stage_id in range(num_stages):
                if stage_id in covered:
                    continue
                cal_month = ((start_month - 1 + stage_id) % 12) + 1
                pct = seasonal_pct.get(cal_month)
                if pct is not None:
                    rhs = (pct / 100.0) * useful_energy + dead_energy
                    bound_constraint_ids.append(constraint_id)
                    bound_stage_ids.append(stage_id)
                    bound_values.append(rhs)

    if not constraints:
        return None

    constraints_dict = {
        "$schema": _SCHEMA_URL,
        "constraints": constraints,
    }

    bounds_table = pa.table(
        {
            "constraint_id": pa.array(bound_constraint_ids, type=pa.int32()),
            "stage_id": pa.array(bound_stage_ids, type=pa.int32()),
            "bound": pa.array(bound_values, type=pa.float64()),
        }
    )

    _LOG.info(
        "Generated %d VminOP constraints with %d stage bounds.",
        len(constraints),
        len(bound_values),
    )

    return constraints_dict, bounds_table


# ---------------------------------------------------------------------------
# Electric constraints from restricao-eletrica.csv
# ---------------------------------------------------------------------------

_UNBOUNDED_THRESHOLD = -1.0e29  # values below this are treated as -inf


def _find_restricao_eletrica(directory: Path) -> Path | None:
    """Locate ``restricao-eletrica.csv`` via ``indices.csv`` in *directory*.

    Parses ``indices.csv`` case-insensitively, looking for the
    ``RESTRICAO-ELETRICA-ESPECIAL`` key.  Returns the resolved path or
    ``None`` if either file is absent.
    """
    indices_path: Path | None = None
    for entry in directory.iterdir():
        if entry.is_file() and entry.name.lower() == "indices.csv":
            indices_path = entry
            break

    if indices_path is None:
        return None

    try:
        with indices_path.open(encoding="latin-1") as fh:
            for line in fh:
                parts = [p.strip() for p in line.split(";")]
                if (
                    len(parts) >= 3
                    and parts[0].upper() == "RESTRICAO-ELETRICA-ESPECIAL"
                ):
                    filename = parts[2].strip()
                    if not filename:
                        return None
                    candidate = directory / filename
                    if candidate.exists():
                        return candidate
                    # Case-insensitive fallback
                    lower = filename.lower()
                    for entry in directory.iterdir():
                        if entry.is_file() and entry.name.lower() == lower:
                            return entry
                    return None
    except OSError:
        return None

    return None


def _parse_restricao_eletrica(
    path: Path,
) -> tuple[
    dict[int, str],
    dict[int, tuple[str, str]],
    list[tuple[int, str, str, int, float, float]],
]:
    """Parse ``restricao-eletrica.csv`` into its three sections.

    Returns
    -------
    expressions : dict[int, str]
        Mapping from constraint code to raw formula string.
    horizons : dict[int, tuple[str, str]]
        Mapping from constraint code to (PerIni, PerFin) as "YYYY/MM" strings.
    bounds : list[tuple[int, str, str, int, float, float]]
        Each element is (cod_rest, PerIni, PerFin, pat, lim_inf, lim_sup).
    """
    expressions: dict[int, str] = {}
    horizons: dict[int, tuple[str, str]] = {}
    bounds: list[tuple[int, str, str, int, float, float]] = []

    with path.open(encoding="latin-1") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("&"):
                continue

            parts = [p.strip() for p in line.split(";")]

            if parts[0].upper() == "RE" and len(parts) >= 3:
                cod = int(parts[1])
                formula = parts[2]
                expressions[cod] = formula

            elif parts[0].upper() == "RE-HORIZ-PER" and len(parts) >= 4:
                cod = int(parts[1])
                horizons[cod] = (parts[2], parts[3])

            elif parts[0].upper() == "RE-LIM-FORM-PER-PAT" and len(parts) >= 7:
                cod = int(parts[1])
                per_ini = parts[2]
                per_fin = parts[3]
                pat = int(parts[4])
                lim_inf = float(parts[5])
                lim_sup = float(parts[6])
                bounds.append((cod, per_ini, per_fin, pat, lim_inf, lim_sup))

    return expressions, horizons, bounds


def _build_line_id_map(nw_files: NewaveFiles) -> dict[tuple[int, int], int]:
    """Build the canonical (src, tgt) -> line_id mapping.

    Replicates the exact logic from ``convert_lines`` in network.py so that
    line IDs are consistent across both converters.

    Returns
    -------
    dict[tuple[int, int], int]
        Maps canonical (smaller_subsystem, larger_subsystem) to 0-based
        line ID.  Only pairs from ``sistema.dat`` are included.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    limites_df = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {}

    from datetime import datetime as _dt

    dger = Dger.read(str(nw_files.dger))
    study_start_dt = _dt(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 1)
    first_month = limites_df[limites_df["data"] == study_start_dt]
    if first_month.empty:
        non_nan_df = limites_df.dropna(subset=["valor"])
        if not non_nan_df.empty:
            first_month = limites_df[limites_df["data"] == non_nan_df["data"].min()]

    canonical_pairs: set[tuple[int, int]] = set()
    for _, row in first_month.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        src, tgt = (de, para) if de < para else (para, de)
        canonical_pairs.add((src, tgt))

    return {pair: idx for idx, pair in enumerate(sorted(canonical_pairs))}


# Term regex: optional float coefficient, then function name and parenthesised args.
_TERM_RE = re.compile(
    r"(\d+\.?\d*)?(?P<fn>ger_usih|ener_interc)\((?P<args>\d+(?:,\s*\d+)*)\)"
)


def _parse_formula(
    formula: str,
    id_map: NewaveIdMap,
    line_id_map: dict[tuple[int, int], int],
) -> str | None:
    """Translate a NEWAVE RE formula into a Cobre expression string.

    Unknown plant codes or interchange pairs are skipped with a warning.
    Returns ``None`` if no valid terms remain after translation.

    Expression syntax rules:
    - Positive coefficient 1.0:  ``variable(id)``
    - Positive coefficient other: ``coeff * variable(id)``
    - Negative coefficient -1.0: ``- variable(id)``
    - Negative coefficient other: ``- abs(coeff) * variable(id)``
    - First term omits leading ``+``; subsequent positive terms use ``+ ``;
      negative terms use ``- `` as binary subtraction.
    """
    # Each element: (effective_coeff: float, variable_str: str)
    parsed_terms: list[tuple[float, str]] = []

    for match in _TERM_RE.finditer(formula):
        coeff_str = match.group(1)
        coeff = float(coeff_str) if coeff_str else 1.0
        fn = match.group("fn")
        args_str = match.group("args")
        args = [int(a.strip()) for a in args_str.split(",")]

        if fn == "ger_usih":
            plant_code = args[0]
            try:
                cobre_id = id_map.hydro_id(plant_code)
            except KeyError:
                _LOG.warning(
                    "Electric constraint: unknown hydro code %d; skipping term.",
                    plant_code,
                )
                continue
            parsed_terms.append((coeff, f"hydro_generation({cobre_id})"))

        elif fn == "ener_interc":
            if len(args) < 2:
                _LOG.warning(
                    "Electric constraint: ener_interc with fewer than 2 args: %s",
                    args_str,
                )
                continue
            from_sys, to_sys = args[0], args[1]
            src, tgt = (from_sys, to_sys) if from_sys < to_sys else (to_sys, from_sys)
            line_id = line_id_map.get((src, tgt))
            if line_id is None:
                _LOG.warning(
                    "Electric constraint: no line for interchange (%d,%d); skipping.",
                    from_sys,
                    to_sys,
                )
                continue

            # If the original direction is reversed relative to canonical, negate.
            effective_coeff = coeff if from_sys < to_sys else -coeff
            parsed_terms.append((effective_coeff, f"line_exchange({line_id})"))

    if not parsed_terms:
        return None

    parts: list[str] = []
    for i, (coeff, var) in enumerate(parsed_terms):
        abs_coeff = abs(coeff)
        is_negative = coeff < 0.0
        if abs_coeff == 1.0:
            term_body = var
        else:
            term_body = f"{abs_coeff} * {var}"

        if i == 0:
            # First term: prepend minus sign if negative, nothing if positive.
            parts.append(f"- {term_body}" if is_negative else term_body)
        else:
            # Subsequent terms: use binary +/- operator.
            parts.append(f"- {term_body}" if is_negative else f"+ {term_body}")

    return " ".join(parts)


def _parse_yyyymm(period_str: str) -> tuple[int, int]:
    """Parse "YYYY/MM" into (year, month)."""
    parts = period_str.strip().split("/")
    return int(parts[0]), int(parts[1])


def convert_electric_constraints(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
    start_id: int = 0,
) -> tuple[list[dict], pa.Table] | None:
    """Convert ``restricao-eletrica.csv`` electric constraints to Cobre format.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths.
    id_map:
        Entity ID mapping.
    start_id:
        First constraint ID to assign (so IDs do not collide with VminOP
        constraints).

    Returns
    -------
    tuple[list[dict], pa.Table] | None
        A ``(constraints_list, bounds_table)`` pair, or ``None`` if the
        file is absent or contains no valid constraints.
        ``bounds_table`` has schema
        ``(constraint_id: INT32, stage_id: INT32, block_id: INT32,
        bound: DOUBLE)``.
    """
    re_path = _find_restricao_eletrica(nw_files.directory)
    if re_path is None:
        _LOG.debug("restricao-eletrica.csv not found; skipping electric constraints.")
        return None

    expressions, horizons, bounds_rows = _parse_restricao_eletrica(re_path)
    if not expressions:
        return None

    # Study horizon
    dger = Dger.read(str(nw_files.dger))
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    num_stages = study_months + num_anos_pos * 12

    line_id_map = _build_line_id_map(nw_files)

    # Read ELETRI penalty from PENALID.DAT for slack costs.
    eletri_penalty: float | None = None
    if nw_files.penalid is not None:
        try:
            penalid = Penalid.read(str(nw_files.penalid))
            df_pen = penalid.penalidades
            if df_pen is not None and not df_pen.empty:
                eletri = df_pen[
                    (df_pen["variavel"] == "ELETRI")
                    & (df_pen["patamar_penalidade"] == 1)
                ]
                vals = eletri["valor_R$_MWh"].dropna()
                if not vals.empty:
                    eletri_penalty = float(vals.iloc[0])
        except Exception:  # noqa: BLE001
            _LOG.warning("Could not read ELETRI penalty from PENALID.DAT.")

    constraints: list[dict] = []
    bound_constraint_ids: list[int] = []
    bound_stage_ids: list[int] = []
    bound_block_ids: list[int | None] = []
    bound_values: list[float] = []

    # Index bounds by constraint code for quick lookup
    bounds_by_code: dict[int, list[tuple[str, str, int, float, float]]] = defaultdict(
        list
    )
    for cod, per_ini, per_fin, pat, lim_inf, lim_sup in bounds_rows:
        bounds_by_code[cod].append((per_ini, per_fin, pat, lim_inf, lim_sup))

    # Index horizons for validity check
    # Only emit bounds within the horizon window
    for code_idx, (cod, formula) in enumerate(sorted(expressions.items())):
        horizon = horizons.get(cod)
        if horizon is None:
            _LOG.warning(
                "Electric constraint %d has no RE-HORIZ-PER entry; skipping.", cod
            )
            continue

        cobre_expr = _parse_formula(formula, id_map, line_id_map)
        if cobre_expr is None:
            _LOG.warning(
                "Electric constraint %d: formula yielded no valid terms; skipping.",
                cod,
            )
            continue

        horiz_ini = _parse_yyyymm(horizon[0])
        horiz_fin = _parse_yyyymm(horizon[1])

        # Collect all (sense, bound_value) pairs for each (stage, block)
        # from the RE-LIM-FORM-PER-PAT rows
        entries_for_code = bounds_by_code.get(cod, [])
        if not entries_for_code:
            _LOG.warning(
                "Electric constraint %d has no RE-LIM-FORM-PER-PAT rows; skipping.",
                cod,
            )
            continue

        # Determine which senses are needed: scan all bound rows for this code
        # to decide whether we need a <= constraint, a >= constraint, or both.
        # entries_for_code elements: (per_ini, per_fin, pat, lim_inf, lim_sup)
        has_sup = any(
            lim_sup > _UNBOUNDED_THRESHOLD
            for _per_ini, _per_fin, _pat, _lim_inf, lim_sup in entries_for_code
        )
        has_inf = any(
            lim_inf > _UNBOUNDED_THRESHOLD
            for _per_ini, _per_fin, _pat, lim_inf, _lim_sup in entries_for_code
        )

        slack_config: dict = (
            {"enabled": True, "penalty": eletri_penalty}
            if eletri_penalty is not None
            else {"enabled": False}
        )

        def _add_constraint(sense: str, constraint_id: int) -> None:
            constraints.append(
                {
                    "id": constraint_id,
                    "name": f"RE_{cod}",
                    "description": f"Electric constraint {cod}",
                    "expression": cobre_expr,
                    "sense": sense,
                    "slack": slack_config,
                }
            )

        sup_id: int | None = None
        inf_id: int | None = None

        if has_sup:
            sup_id = start_id + len(constraints)
            _add_constraint("<=", sup_id)
        if has_inf:
            inf_id = start_id + len(constraints)
            _add_constraint(">=", inf_id)

        # Build a seasonal bound map from the original (first-year) data.
        # Key: (calendar_month, block_id) -> (lim_sup, lim_inf)
        # NEWAVE restricts restricao-eletrica.csv to the first 12 stages
        # (individual plant modeling horizon).  We extrapolate seasonally
        # across the full study horizon: each calendar month in every year
        # reuses the bound from the same month in the first year.
        seasonal_sup: dict[tuple[int, int], float] = {}
        seasonal_inf: dict[tuple[int, int], float] = {}

        for per_ini, per_fin, pat, lim_inf, lim_sup in entries_for_code:
            ini_year, ini_month = _parse_yyyymm(per_ini)
            fin_year, fin_month = _parse_yyyymm(per_fin)

            # Restrict to horizon window (for the initial data extraction)
            ini_year, ini_month = max(
                (ini_year, ini_month), horiz_ini, key=lambda t: t[0] * 12 + t[1]
            )
            fin_year, fin_month = min(
                (fin_year, fin_month), horiz_fin, key=lambda t: t[0] * 12 + t[1]
            )

            block_id = pat - 1  # 1-based Pat -> 0-based block_id

            y, m = ini_year, ini_month
            while (y, m) <= (fin_year, fin_month):
                key = (m, block_id)
                if lim_sup > _UNBOUNDED_THRESHOLD:
                    seasonal_sup[key] = lim_sup
                if lim_inf > _UNBOUNDED_THRESHOLD:
                    seasonal_inf[key] = lim_inf
                m += 1
                if m > 12:
                    m = 1
                    y += 1

        # Emit bound rows for the FULL study horizon using the seasonal map.
        for stage_id in range(num_stages):
            cal_month = ((start_month - 1 + stage_id) % 12) + 1
            for block_id in range(3):  # 3 patamares
                key = (cal_month, block_id)
                if sup_id is not None and key in seasonal_sup:
                    bound_constraint_ids.append(sup_id)
                    bound_stage_ids.append(stage_id)
                    bound_block_ids.append(block_id)
                    bound_values.append(seasonal_sup[key])
                if inf_id is not None and key in seasonal_inf:
                    bound_constraint_ids.append(inf_id)
                    bound_stage_ids.append(stage_id)
                    bound_block_ids.append(block_id)
                    bound_values.append(seasonal_inf[key])

    if not constraints:
        return None

    bounds_table = pa.table(
        {
            "constraint_id": pa.array(bound_constraint_ids, type=pa.int32()),
            "stage_id": pa.array(bound_stage_ids, type=pa.int32()),
            "block_id": pa.array(bound_block_ids, type=pa.int32()),
            "bound": pa.array(bound_values, type=pa.float64()),
        }
    )

    _LOG.info(
        "Generated %d electric constraints with %d bounds.",
        len(constraints),
        len(bound_values),
    )

    return constraints, bounds_table


# ---------------------------------------------------------------------------
# AGRINT.DAT — Exchange group constraints
# ---------------------------------------------------------------------------

_AGRINT_DEFAULT_PENALTY = 1000.0


def _parse_agrint(
    path: Path,
) -> tuple[
    dict[int, list[tuple[int, int, float]]],  # group_id -> [(A, B, coeff)]
    list[tuple[int, int, int, int | None, int | None, list[float]]],
    # (group_id, mi, anoi, mf|None, anof|None, [lim_p1, lim_p2, lim_p3])
]:
    """Parse AGRINT.DAT into group definitions and limit rows.

    Returns
    -------
    groups : dict[int, list[tuple[int, int, float]]]
        Mapping from group ID to a list of (A, B, coefficient) tuples.
    limits : list of 6-tuples
        Each element is
        (group_id, start_month, start_year, end_month|None, end_year|None,
         [lim_p1, lim_p2, lim_p3]).
    """
    groups: dict[int, list[tuple[int, int, float]]] = {}
    limits: list[tuple[int, int, int, int | None, int | None, list[float]]] = []

    in_groups_section = False
    in_limits_section = False

    with path.open(encoding="latin-1") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n")

            stripped = line.strip()
            if not stripped or stripped.startswith("&"):
                continue

            upper = stripped.upper()

            # Section headers
            if (
                "AGRUPAMENTOS" in upper
                and "INTERCÂMBIO" not in upper
                or ("AGRUPAMENTOS" in upper)
            ):
                if "LIMITES" not in upper:
                    in_groups_section = True
                    in_limits_section = False
                    continue

            if "LIMITES POR GRUPO" in upper:
                in_groups_section = False
                in_limits_section = True
                continue

            # Skip format/header comment lines
            if stripped.startswith("#") or stripped.startswith("X"):
                continue

            if in_groups_section:
                # Terminator
                if stripped.startswith("999"):
                    in_groups_section = False
                    continue
                # Data line: right-aligned fixed-width columns
                # Format: XXX XXX XXX XX.XXXX
                # columns: group(1-3), A(5-7), B(9-11), coeff(13-20)
                try:
                    parts = stripped.split()
                    if len(parts) < 4:
                        continue
                    group_id = int(parts[0])
                    a = int(parts[1])
                    b = int(parts[2])
                    coeff = float(parts[3])
                    groups.setdefault(group_id, []).append((a, b, coeff))
                except (ValueError, IndexError):
                    continue

            elif in_limits_section:
                if stripped.startswith("999"):
                    break
                # Format: #AG MI ANOI MF ANOF LIM_P1 LIM_P2 LIM_P3 [description]
                # MF/ANOF are optional (open-ended if blank)
                try:
                    parts = stripped.split()
                    if len(parts) < 4:
                        continue
                    group_id = int(parts[0])
                    mi = int(parts[1])
                    anoi = int(parts[2])

                    # Detect whether MF/ANOF are present by trying to parse
                    # parts[3] and parts[4] as int before float limits.
                    # Limits always end with '.' in the file, so they parse as float.
                    # MF/ANOF are small integers (1-12 / 4-digit year).
                    mf: int | None = None
                    anof: int | None = None
                    lim_start = 3

                    # parts[3]: could be MF (int 1-12) or first limit (float w/ '.')
                    if "." not in parts[3] and len(parts) > 5:
                        # Could still be a year — check parts[4] too
                        try:
                            candidate_mf = int(parts[3])
                            candidate_anof = int(parts[4])
                            # Sanity: MF in [1,12], ANOF is a 4-digit year
                            if (
                                1 <= candidate_mf <= 12
                                and 1900 <= candidate_anof <= 9999
                            ):
                                mf = candidate_mf
                                anof = candidate_anof
                                lim_start = 5
                        except (ValueError, IndexError):
                            pass

                    lim_parts = parts[lim_start : lim_start + 3]
                    if len(lim_parts) < 1:
                        continue
                    lims = [float(lp.rstrip(".")) for lp in lim_parts]
                    # Pad to 3 if fewer patamars declared
                    while len(lims) < 3:
                        lims.append(lims[-1])
                    limits.append((group_id, mi, anoi, mf, anof, lims))
                except (ValueError, IndexError):
                    continue

    return groups, limits


def convert_agrint_constraints(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
    start_id: int = 0,
) -> tuple[list[dict], pa.Table] | None:
    """Convert AGRINT.DAT exchange group constraints to Cobre generic constraints.

    Each group in AGRINT.DAT defines a weighted sum of directional interchange
    flows.  Each group becomes one ``<=`` generic constraint with a high-default
    slack penalty.  Bounds are stored per (constraint_id, stage_id, block_id).

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths.
    id_map:
        Entity ID mapping (used indirectly via ``_build_line_id_map``).
    start_id:
        First constraint ID to assign (must not collide with other constraints).

    Returns
    -------
    tuple[list[dict], pa.Table] | None
        A ``(constraints_list, bounds_table)`` pair, or ``None`` if
        ``agrint.dat`` is absent or contains no valid constraints.
        ``bounds_table`` has schema
        ``(constraint_id: INT32, stage_id: INT32, block_id: INT32,
        bound: DOUBLE)``.
    """
    if nw_files.agrint is None:
        _LOG.debug("agrint.dat not found; skipping AGRINT constraints.")
        return None

    groups, limits = _parse_agrint(nw_files.agrint)
    if not groups or not limits:
        return None

    # Study horizon
    dger = Dger.read(str(nw_files.dger))
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    num_stages = study_months + num_anos_pos * 12

    line_id_map = _build_line_id_map(nw_files)

    constraints: list[dict] = []
    bound_constraint_ids: list[int] = []
    bound_stage_ids: list[int] = []
    bound_block_ids: list[int] = []
    bound_values: list[float] = []

    for group_id in sorted(groups.keys()):
        terms_raw = groups[group_id]

        # Build expression: each (A, B, coeff) -> directional line_exchange term
        parsed_terms: list[tuple[float, str]] = []
        for a, b, coeff in terms_raw:
            src, tgt = (a, b) if a < b else (b, a)
            line_id = line_id_map.get((src, tgt))
            if line_id is None:
                _LOG.warning(
                    "AGRINT group %d: no line found for interchange (%d,%d); "
                    "skipping term.",
                    group_id,
                    a,
                    b,
                )
                continue
            # Canonical direction is src->tgt (src < tgt).
            # If flow is A->B and A < B: positive coeff.
            # If flow is A->B and A > B: reversed direction => negate.
            effective_coeff = coeff if a < b else -coeff
            parsed_terms.append((effective_coeff, f"line_exchange({line_id})"))

        if not parsed_terms:
            _LOG.warning(
                "AGRINT group %d: no valid terms after line lookup; skipping.",
                group_id,
            )
            continue

        # Render expression string
        parts: list[str] = []
        for i, (eff_coeff, var) in enumerate(parsed_terms):
            abs_coeff = abs(eff_coeff)
            is_neg = eff_coeff < 0.0
            term_body = var if abs_coeff == 1.0 else f"{abs_coeff} * {var}"
            if i == 0:
                parts.append(f"- {term_body}" if is_neg else term_body)
            else:
                parts.append(f"- {term_body}" if is_neg else f"+ {term_body}")
        expression = " ".join(parts)

        constraint_id = start_id + len(constraints)
        constraints.append(
            {
                "id": constraint_id,
                "name": f"AGRINT_{group_id}",
                "description": f"Exchange group constraint {group_id}",
                "expression": expression,
                "sense": "<=",
                "slack": {"enabled": False},
            }
        )

        # Collect limit rows for this group
        for grp, mi, anoi, mf, anof, lims in limits:
            if grp != group_id:
                continue

            # Determine effective end (open-ended => last study stage)
            if mf is None or anof is None:
                # Open-ended: run to the end of the study horizon
                end_y = start_year + (num_stages - 1 + start_month - 1) // 12
                end_m = (start_month - 1 + num_stages - 1) % 12 + 1
            else:
                end_y, end_m = anof, mf

            # Iterate month by month within [mi/anoi .. end_m/end_y]
            y, m = anoi, mi
            while (y, m) <= (end_y, end_m):
                stage_id = (y - start_year) * 12 + (m - start_month)
                if 0 <= stage_id < num_stages:
                    for block_idx, lim_val in enumerate(lims):
                        bound_constraint_ids.append(constraint_id)
                        bound_stage_ids.append(stage_id)
                        bound_block_ids.append(block_idx)
                        bound_values.append(lim_val)
                m += 1
                if m > 12:
                    m = 1
                    y += 1

    if not constraints:
        return None

    bounds_table = pa.table(
        {
            "constraint_id": pa.array(bound_constraint_ids, type=pa.int32()),
            "stage_id": pa.array(bound_stage_ids, type=pa.int32()),
            "block_id": pa.array(bound_block_ids, type=pa.int32()),
            "bound": pa.array(bound_values, type=pa.float64()),
        }
    )

    _LOG.info(
        "Generated %d AGRINT group constraints with %d bounds.",
        len(constraints),
        len(bound_values),
    )

    return constraints, bounds_table
