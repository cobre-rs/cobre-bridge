"""Network entity converter: maps NEWAVE bus and line data to Cobre network JSON."""

from __future__ import annotations

import logging

import pandas as pd
import pyarrow as pa
from inewave.newave import Dger, Sistema

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

_BUSES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/buses.schema.json"
)
_LINES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/lines.schema.json"
)
_PENALTIES_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/penalties.schema.json"
)

# Penalty reference value and multipliers.
# The spillage cost ($/m3s) is the reference. Other penalties are expressed
# as multipliers of the spillage cost, converted to the energy domain ($/MW)
# using the average hydro productivity.
#
# Ordering: exchange < spillage < fpha_turbined < curtailment < excess
# This ensures the optimizer prefers interchange over spilling or curtailing.
_SPILLAGE_REF = 0.001  # R$/(m3/s) — base reference in flow domain
_EXCHANGE_MULT = 0.9  # exchange < spillage
_FPHA_TURBINED_MULT = 1.1
_NCS_CURTAILMENT_MULT = 1.15
_EXCESS_MULT = 1.20

# Hard constraint violation penalties (high values, not affected by scaling).
_DEFAULT_STORAGE_VIOLATION_BELOW_COST = 10000.0
_DEFAULT_FILLING_TARGET_VIOLATION_COST = 10000.0
_DEFAULT_TURBINED_VIOLATION_BELOW_COST = 10000.0
_DEFAULT_OUTFLOW_VIOLATION_BELOW_COST = 10000.0
_DEFAULT_OUTFLOW_VIOLATION_ABOVE_COST = 10000.0
_DEFAULT_GENERATION_VIOLATION_BELOW_COST = 10000.0
_DEFAULT_EVAPORATION_VIOLATION_COST = (
    10000.0  # >> spillage_cost; prevents free-spillage via evap
)
_DEFAULT_WATER_WITHDRAWAL_VIOLATION_COST = 10000.0
_DEFAULT_DIVERSION_COST = 0.001


def _build_canonical_pair_to_line_id(
    nw_files: NewaveFiles,
) -> dict[tuple[int, int], int]:
    """Build the canonical (src, tgt) -> line_id mapping from sistema.dat.

    Scans ALL rows of ``sistema.limites_intercambio`` (all dates) to discover
    the full set of interchange pairs.  This is the single authoritative source
    used by ``convert_lines``, ``convert_line_bounds``, and
    ``convert_exchange_factors`` to guarantee consistent line IDs.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    limites_df = sistema.limites_intercambio
    if limites_df is None or limites_df.empty:
        return {}

    all_pairs: set[tuple[int, int]] = set()
    for _, row in limites_df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        src, tgt = (de, para) if de < para else (para, de)
        all_pairs.add((src, tgt))

    return {pair: lid for lid, pair in enumerate(sorted(all_pairs))}


def convert_buses(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE subsystem data to a Cobre ``buses.json`` dict.

    Reads ``sistema.dat`` from *nw_files*.  Each subsystem (including
    fictitious ones) becomes a bus.  Deficit segments are extracted from
    ``Sistema.custo_deficit``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built ID mapping for bus IDs.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    deficit_df = sistema.custo_deficit

    if deficit_df is None:
        raise ValueError(
            "sistema.dat contains no deficit cost data (custo_deficit is None)"
        )

    # Build per-subsystem deficit segments.
    # Columns: codigo_submercado, nome_submercado, ficticio,
    # patamar_deficit, custo, corte
    buses_by_code: dict[int, dict] = {}

    for _, row in deficit_df.iterrows():
        code = int(row["codigo_submercado"])
        name = str(row["nome_submercado"]).strip()

        if code not in buses_by_code:
            buses_by_code[code] = {
                "newave_code": code,
                "name": name,
                "segments": [],
            }

        corte = row.get("corte")
        depth_mw: float | None = (
            float(corte) if corte is not None and not _is_na(corte) else None
        )
        cost_raw = row["custo"]
        cost = float(cost_raw) if not _is_na(cost_raw) else None
        buses_by_code[code]["segments"].append(
            {
                "patamar": int(row["patamar_deficit"]),
                "depth_mw": depth_mw,
                "cost": cost,
            }
        )

    # Find the reference deficit cost (first non-NaN, non-zero cost across
    # all subsystems) to use as a fallback for fictitious subsystems.
    fallback_cost = 0.0
    for info in buses_by_code.values():
        for seg in info["segments"]:
            if seg["cost"] is not None and seg["cost"] > 0:
                fallback_cost = seg["cost"]
                break
        if fallback_cost > 0:
            break

    buses: list[dict] = []
    for code, info in buses_by_code.items():
        segs = sorted(info["segments"], key=lambda s: s["patamar"])
        active_segs = [s for s in segs if s["cost"] is not None and s["cost"] > 0]
        if not active_segs:
            active_segs = [{"cost": fallback_cost, "depth_mw": None}]

        deficit_segments: list[dict] = []
        for i, seg in enumerate(active_segs):
            is_last = i == len(active_segs) - 1
            deficit_segments.append(
                {
                    "depth_mw": None if is_last else seg["depth_mw"],
                    "cost": seg["cost"],
                }
            )

        bus_entry: dict = {
            "id": id_map.bus_id(code),
            "name": info["name"],
            "deficit_segments": deficit_segments,
        }
        buses.append(bus_entry)

    buses.sort(key=lambda b: b["id"])

    return {
        "$schema": _BUSES_SCHEMA_URL,
        "buses": buses,
    }


def convert_lines(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE interchange limits to a Cobre ``lines.json`` dict.

    Reads ``sistema.dat`` from *nw_files*.  Each directional interchange
    pair becomes a line using the first study month's limits as static
    capacities.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built ID mapping for bus IDs.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    limites_df = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {
            "$schema": _LINES_SCHEMA_URL,
            "lines": [],
        }

    # Use the study start month from dger.dat as the reference for static
    # capacities.  sistema.dat always contains full calendar years, so
    # pre-study months (before mes_inicio_estudo) may have NaN values.
    from datetime import datetime as _dt

    dger = Dger.read(nw_files.dger)
    study_start_dt = _dt(dger.ano_inicio_estudo, dger.mes_inicio_estudo, 1)
    first_month = limites_df[limites_df["data"] == study_start_dt]
    if first_month.empty:
        first_month = limites_df.dropna(subset=["valor"])
        if not first_month.empty:
            first_date = first_month["data"].min()
            first_month = limites_df[limites_df["data"] == first_date]

    # Build a (source, target) -> {direct_mw, reverse_mw} structure.
    # sentido == 1: direct (de -> para), sentido == 2: reverse (para -> de)
    # We normalise all pairs so source_code < target_code to deduplicate.
    pair_map: dict[tuple[int, int], dict[str, float]] = {}

    for _, row in first_month.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        valor = float(row["valor"])
        sentido = int(row["sentido"])

        # Canonical key: smaller ID first.
        src, tgt = (de, para) if de < para else (para, de)
        key = (src, tgt)

        if key not in pair_map:
            pair_map[key] = {"direct_mw": 0.0, "reverse_mw": 0.0}

        if de < para:
            # de -> para is the "direct" direction.
            if sentido == 1:
                pair_map[key]["direct_mw"] = valor
            else:
                pair_map[key]["reverse_mw"] = valor
        else:
            # de -> para is the "reverse" direction.
            if sentido == 1:
                pair_map[key]["reverse_mw"] = valor
            else:
                pair_map[key]["direct_mw"] = valor

    # Use the shared canonical mapping for consistent line IDs.
    canonical_map = _build_canonical_pair_to_line_id(nw_files)

    lines: list[dict] = []
    for (src, tgt), line_id in sorted(canonical_map.items(), key=lambda x: x[1]):
        caps = pair_map.get((src, tgt), {"direct_mw": 0.0, "reverse_mw": 0.0})
        src_bus = id_map.bus_id(src)
        tgt_bus = id_map.bus_id(tgt)
        src_name = _subsystem_name_from_id(src)
        tgt_name = _subsystem_name_from_id(tgt)
        line_entry: dict = {
            "id": line_id,
            "name": f"{src_name}_{tgt_name}",
            "source_bus_id": src_bus,
            "target_bus_id": tgt_bus,
            "capacity": {
                "direct_mw": caps["direct_mw"],
                "reverse_mw": caps["reverse_mw"],
            },
        }
        lines.append(line_entry)

    return {
        "$schema": _LINES_SCHEMA_URL,
        "lines": lines,
    }


def convert_penalties(nw_files: NewaveFiles, hydros_dict: dict) -> dict:
    """Generate a Cobre ``penalties.json`` dict from NEWAVE deficit data.

    Uses the first subsystem's first deficit tier cost as the primary
    deficit cost.  Operational penalties (spillage, exchange, curtailment,
    excess) are derived from a base spillage reference value with
    multipliers, converted to the energy domain using the average hydro
    productivity.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    hydros_dict:
        The already-converted ``hydros.json`` dict, used to compute the
        average productivity for flow-to-energy penalty conversion.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    deficit_df = sistema.custo_deficit

    # Primary deficit cost: first subsystem, first patamar.
    primary_deficit_cost = 0.0
    if deficit_df is not None and not deficit_df.empty:
        first_sub = deficit_df.sort_values(["codigo_submercado", "patamar_deficit"])
        first_row = first_sub.iloc[0]
        primary_deficit_cost = float(first_row["custo"])

    # Compute average productivity across all hydros for unit conversion.
    # Spillage is in m3/s, energy-domain penalties are in $/MW.
    # avg_prod converts: cost_per_MW = cost_per_m3s / avg_prod
    hydros = hydros_dict.get("hydros", [])
    productivities = [
        h["generation"]["productivity_mw_per_m3s"]
        for h in hydros
        if h["generation"].get("productivity_mw_per_m3s", 0) > 0
    ]
    avg_prod = sum(productivities) / len(productivities) if productivities else 1.0

    # Spillage cost is the reference (in flow domain: $/m3s).
    spillage_cost = _SPILLAGE_REF

    # Energy-domain penalties: convert the reference from flow to energy,
    # then apply multipliers.
    ref_energy = _SPILLAGE_REF / avg_prod
    exchange_cost = ref_energy * _EXCHANGE_MULT
    fpha_turbined_cost = _SPILLAGE_REF * _FPHA_TURBINED_MULT
    curtailment_cost = ref_energy * _NCS_CURTAILMENT_MULT
    excess_cost = ref_energy * _EXCESS_MULT

    return {
        "$schema": _PENALTIES_SCHEMA_URL,
        "bus": {
            "deficit_segments": [
                {
                    "cost": primary_deficit_cost,
                    "depth_mw": None,
                }
            ],
            "excess_cost": excess_cost,
        },
        "hydro": {
            "spillage_cost": spillage_cost,
            "fpha_turbined_cost": fpha_turbined_cost,
            "diversion_cost": _DEFAULT_DIVERSION_COST,
            "storage_violation_below_cost": _DEFAULT_STORAGE_VIOLATION_BELOW_COST,
            "filling_target_violation_cost": _DEFAULT_FILLING_TARGET_VIOLATION_COST,
            "turbined_violation_below_cost": _DEFAULT_TURBINED_VIOLATION_BELOW_COST,
            "outflow_violation_below_cost": _DEFAULT_OUTFLOW_VIOLATION_BELOW_COST,
            "outflow_violation_above_cost": _DEFAULT_OUTFLOW_VIOLATION_ABOVE_COST,
            "generation_violation_below_cost": _DEFAULT_GENERATION_VIOLATION_BELOW_COST,
            "evaporation_violation_cost": _DEFAULT_EVAPORATION_VIOLATION_COST,
            "water_withdrawal_violation_cost": _DEFAULT_WATER_WITHDRAWAL_VIOLATION_COST,
        },
        "line": {
            "exchange_cost": exchange_cost,
        },
        "non_controllable_source": {
            "curtailment_cost": curtailment_cost,
        },
    }


def convert_line_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert NEWAVE interchange limits to a Cobre ``line_bounds.parquet`` table.

    Reads ``sistema.dat::limites_intercambio`` and ``dger.dat`` to produce one
    row per (line, stage) pair with direct and reverse MW bounds.

    The canonical pair logic (``src < tgt``) and line ID assignment exactly
    mirror ``convert_lines`` so that line IDs are consistent.  For post-study
    stages, the last available study year's bounds are repeated seasonally.
    Per-block exchange bound factors from ``patamar.dat::intercambio_patamares``
    are emitted separately via ``convert_exchange_factors``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map.  Used to resolve subsystem codes to Cobre bus IDs
        (indirectly, via the same canonical-pair ordering used in
        ``convert_lines``).

    Returns
    -------
    pyarrow.Table
        Columns: ``line_id`` (INT32), ``stage_id`` (INT32),
        ``direct_mw`` (DOUBLE), ``reverse_mw`` (DOUBLE).
    """
    sistema = Sistema.read(str(nw_files.sistema))
    limites_df: pd.DataFrame | None = sistema.limites_intercambio

    _LINE_BOUNDS_SCHEMA = pa.schema(
        [
            pa.field("line_id", pa.int32()),
            pa.field("stage_id", pa.int32()),
            pa.field("direct_mw", pa.float64()),
            pa.field("reverse_mw", pa.float64()),
        ]
    )

    if limites_df is None or limites_df.empty:
        return pa.table(
            {
                "line_id": pa.array([], type=pa.int32()),
                "stage_id": pa.array([], type=pa.int32()),
                "direct_mw": pa.array([], type=pa.float64()),
                "reverse_mw": pa.array([], type=pa.float64()),
            },
            schema=_LINE_BOUNDS_SCHEMA,
        )

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

    pair_to_line_id = _build_canonical_pair_to_line_id(nw_files)

    # Build per-date lookup:
    # {(src, tgt, year, cal_month) -> {direct_mw, reverse_mw}}
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
            # de -> para is the "direct" direction.
            if sentido == 1:
                date_lookup[key]["direct_mw"] = valor
            else:
                date_lookup[key]["reverse_mw"] = valor
        else:
            # de -> para is the "reverse" direction.
            if sentido == 1:
                date_lookup[key]["reverse_mw"] = valor
            else:
                date_lookup[key]["direct_mw"] = valor

    # Build last-year lookup for post-study:
    # {(src, tgt, cal_month) -> {direct_mw, reverse_mw}} — use the latest year.
    last_year_per_key: dict[tuple[int, int, int], tuple[int, dict[str, float]]] = {}
    for (src, tgt, yr, cal_month), caps in date_lookup.items():
        key3 = (src, tgt, cal_month)
        existing = last_year_per_key.get(key3)
        if existing is None or yr > existing[0]:
            last_year_per_key[key3] = (yr, caps)
    last_year_lookup: dict[tuple[int, int, int], dict[str, float]] = {
        k: v for k, (_, v) in last_year_per_key.items()
    }

    rows_line_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_direct: list[float] = []
    rows_reverse: list[float] = []

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

            rows_line_id.append(line_id)
            rows_stage_id.append(stage_id)
            rows_direct.append(caps["direct_mw"])
            rows_reverse.append(caps["reverse_mw"])

            m += 1
            if m > 12:
                m = 1
                y += 1

    return pa.table(
        {
            "line_id": pa.array(rows_line_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "direct_mw": pa.array(rows_direct, type=pa.float64()),
            "reverse_mw": pa.array(rows_reverse, type=pa.float64()),
        },
        schema=_LINE_BOUNDS_SCHEMA,
    )


def _build_ncs_group_to_id(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int], int]:
    """Build the canonical (codigo_submercado, indice_bloco) -> ncs_id mapping.

    Applies the same horizon filtering and bus_id validation used by
    ``convert_non_controllable_sources``.  This is the single authoritative
    NCS group mapping shared by ``convert_ncs_factors`` and
    ``convert_ncs_stats``.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    df_ncs: pd.DataFrame | None = sistema.geracao_usinas_nao_simuladas
    if df_ncs is None or df_ncs.empty:
        return {}

    dger = Dger.read(nw_files.dger)
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    def _in_horizon(dt: object) -> bool:
        try:
            yr = int(dt.year)  # type: ignore[union-attr]
            mo = int(dt.month)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return False
        if yr == 9999:
            return True
        stage_id = (yr - start_year) * 12 + (mo - start_month)
        return 0 <= stage_id < total_stages

    df_filtered = df_ncs[df_ncs["data"].apply(_in_horizon)].copy()
    groups = df_filtered.groupby(
        ["codigo_submercado", "indice_bloco"], sort=True
    )

    result: dict[tuple[int, int], int] = {}
    ncs_id = 0
    for (sub_code, bloco), _group in groups:
        sub_code_int = int(sub_code)
        try:
            id_map.bus_id(sub_code_int)
        except KeyError:
            continue
        result[(sub_code_int, int(bloco))] = ncs_id
        ncs_id += 1

    return result


def convert_non_controllable_sources(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict:
    """Convert NEWAVE non-simulated generation to a Cobre NCS entity JSON dict.

    Reads ``sistema.dat::geracao_usinas_nao_simuladas``.  Each unique
    ``(codigo_submercado, indice_bloco)`` pair becomes one NCS entity.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map.  Used to resolve subsystem codes to 0-based Cobre bus
        IDs.

    Returns
    -------
    dict
        JSON-serializable dict with key ``"non_controllable_sources"``
        containing a list of NCS entity dicts.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    df_ncs: pd.DataFrame | None = sistema.geracao_usinas_nao_simuladas

    if df_ncs is None or df_ncs.empty:
        return {"non_controllable_sources": []}

    dger = Dger.read(nw_files.dger)
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    # Filter to study + post-study horizon only.
    # Rows with year == 9999 are post-study entries in inewave convention.
    def _in_horizon(dt: object) -> bool:
        try:
            yr = int(dt.year)  # type: ignore[union-attr]
            mo = int(dt.month)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return False
        if yr == 9999:
            return True
        stage_id = (yr - start_year) * 12 + (mo - start_month)
        return 0 <= stage_id < total_stages

    df_filtered = df_ncs[df_ncs["data"].apply(_in_horizon)].copy()

    # Columns: codigo_submercado, indice_bloco, fonte, data, valor
    # Group by (codigo_submercado, indice_bloco) — each unique pair is one NCS.
    ncs_list: list[dict] = []
    ncs_id = 0

    groups = df_filtered.groupby(["codigo_submercado", "indice_bloco"], sort=True)

    for (sub_code, bloco), group in groups:
        sub_code_int = int(sub_code)
        try:
            bus_id = id_map.bus_id(sub_code_int)
        except KeyError:
            _LOG.warning(
                "Subsystem code %d from geracao_usinas_nao_simuladas not in "
                "id_map; skipping NCS (indice_bloco=%s)",
                sub_code_int,
                bloco,
            )
            continue

        # fonte: use the first non-null value in the group.
        fonte_series = group["fonte"].dropna()
        fonte = str(fonte_series.iloc[0]).strip() if not fonte_series.empty else "NCS"

        # max_generation_mw: maximum non-NaN value across all rows in the group.
        valores = pd.to_numeric(group["valor"], errors="coerce")
        valid_vals = valores.dropna()
        max_gen = float(valid_vals.max()) if not valid_vals.empty else 0.0

        ncs_list.append(
            {
                "id": ncs_id,
                "name": f"{fonte}_{sub_code_int}",
                "bus_id": bus_id,
                "max_generation_mw": max_gen,
            }
        )
        ncs_id += 1

    return {"non_controllable_sources": ncs_list}


def convert_exchange_factors(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict:
    """Convert patamar.dat exchange factors to a Cobre ``exchange_factors.json`` dict.

    Reads ``patamar.dat::intercambio_patamares``.  For each (line_id,
    stage_id) pair, collects per-block direct and reverse factors from the
    (submercado_de, submercado_para) directional rows.

    The canonical pair logic and line ID assignment exactly mirror
    ``convert_lines`` and ``convert_line_bounds``:
    ``src, tgt = (de, para) if de < para else (para, de)`` and line IDs are
    assigned by ``enumerate(sorted(pairs))``.

    When ``de < para``, the row's factor applies to ``direct_factor``.
    When ``de > para`` (reversed pair), the factor applies to
    ``reverse_factor``.  Each (line_id, stage_id) entry combines factors
    from both directions into one ``block_factors`` array.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map (unused directly; kept for API consistency).

    Returns
    -------
    dict
        JSON-serializable dict with key ``"exchange_factors"``.
    """
    from inewave.newave import Dger, Patamar

    patamar = Patamar.read(str(nw_files.patamar))
    df: pd.DataFrame | None = patamar.intercambio_patamares

    if df is None or df.empty:
        return {"exchange_factors": []}

    dger = Dger.read(nw_files.dger)
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    pair_to_line_id = _build_canonical_pair_to_line_id(nw_files)
    if not pair_to_line_id:
        return {"exchange_factors": []}

    # Build per-(src, tgt, year, cal_month, patamar) factor lookup.
    # Key: (src, tgt, year, cal_month, block_id) -> (direct_factor, reverse_factor)
    # block_id is 0-based (patamar is 1-based in source).
    FactorKey = tuple  # (src, tgt, yr, cal_month, block_id)
    direct_map: dict[FactorKey, float] = {}
    reverse_map: dict[FactorKey, float] = {}

    for _, row in df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        val = float(row["valor"])
        block_id = int(row["patamar"]) - 1  # convert 1-based to 0-based
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)

        src, tgt = (de, para) if de < para else (para, de)
        key: FactorKey = (src, tgt, yr, cal_month, block_id)

        if de < para:
            direct_map[key] = val
        else:
            reverse_map[key] = val

    # Determine number of blocks from the data.
    all_blocks: set[int] = set()
    for _, row in df.iterrows():
        all_blocks.add(int(row["patamar"]) - 1)
    num_blocks = max(all_blocks) + 1 if all_blocks else 1

    # Build last-year seasonal lookups for post-study stages.
    # {(src, tgt, cal_month, block_id) -> factor}
    last_yr_direct: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    last_yr_reverse: dict[tuple[int, int, int, int], tuple[int, float]] = {}

    for (src, tgt, yr, cal_month, block_id), val in direct_map.items():
        k4 = (src, tgt, cal_month, block_id)
        existing = last_yr_direct.get(k4)
        if existing is None or yr > existing[0]:
            last_yr_direct[k4] = (yr, val)

    for (src, tgt, yr, cal_month, block_id), val in reverse_map.items():
        k4 = (src, tgt, cal_month, block_id)
        existing = last_yr_reverse.get(k4)
        if existing is None or yr > existing[0]:
            last_yr_reverse[k4] = (yr, val)

    last_direct: dict[tuple[int, int, int, int], float] = {
        k: v for k, (_, v) in last_yr_direct.items()
    }
    last_reverse: dict[tuple[int, int, int, int], float] = {
        k: v for k, (_, v) in last_yr_reverse.items()
    }

    results: list[dict] = []

    for pair, line_id in sorted(pair_to_line_id.items(), key=lambda x: x[1]):
        src, tgt = pair
        y, m = start_year, start_month

        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            block_factors: list[dict] = []
            for block_id in range(num_blocks):
                if is_post_study:
                    d_factor = last_direct.get((src, tgt, m, block_id), 1.0)
                    r_factor = last_reverse.get((src, tgt, m, block_id), 1.0)
                else:
                    key_lookup: FactorKey = (src, tgt, y, m, block_id)
                    d_factor = direct_map.get(
                        key_lookup,
                        last_direct.get((src, tgt, m, block_id), 1.0),
                    )
                    r_factor = reverse_map.get(
                        key_lookup,
                        last_reverse.get((src, tgt, m, block_id), 1.0),
                    )

                block_factors.append(
                    {
                        "block_id": block_id,
                        "direct_factor": d_factor,
                        "reverse_factor": r_factor,
                    }
                )

            results.append(
                {
                    "line_id": line_id,
                    "stage_id": stage_id,
                    "block_factors": block_factors,
                }
            )

            m += 1
            if m > 12:
                m = 1
                y += 1

    return {"exchange_factors": results}


def convert_ncs_factors(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict:
    """Convert patamar.dat NCS block factors to a Cobre non_controllable_factors dict.

    Reads ``patamar.dat::usinas_nao_simuladas``.  NCS entity IDs are assigned
    using the same ``(codigo_submercado, indice_bloco)`` sorted grouping as
    ``convert_non_controllable_sources`` in this module.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map.  Used for subsystem code validation.

    Returns
    -------
    dict
        JSON-serializable dict with key ``"non_controllable_factors"``.
    """
    from inewave.newave import Dger, Patamar

    patamar_file = Patamar.read(str(nw_files.patamar))
    df: pd.DataFrame | None = patamar_file.usinas_nao_simuladas

    if df is None or df.empty:
        return {"non_controllable_factors": []}

    dger = Dger.read(nw_files.dger)
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Filter to study + post-study horizon (year == 9999 is post-study).
    def _in_horizon(dt: object) -> bool:
        try:
            yr = int(dt.year)  # type: ignore[union-attr]
            mo = int(dt.month)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return False
        if yr == 9999:
            return True
        stage_id = (yr - start_year) * 12 + (mo - start_month)
        return 0 <= stage_id < total_stages

    df = df[df["data"].apply(_in_horizon)].copy()

    if df.empty:
        return {"non_controllable_factors": []}

    # Columns: codigo_submercado, indice_bloco, data, patamar, valor
    # Determine number of blocks from source.
    all_blocks_set: set[int] = set()
    for _, row in df.iterrows():
        all_blocks_set.add(int(row["patamar"]) - 1)
    num_blocks = max(all_blocks_set) + 1 if all_blocks_set else 1

    # Build per-(sub_code, bloco, yr, cal_month, block_id) -> factor lookup.
    NcsKey = tuple  # (sub_code, bloco, yr, cal_month, block_id)
    factor_map: dict[NcsKey, float] = {}

    for _, row in df.iterrows():
        sub_code = int(row["codigo_submercado"])
        bloco = int(row["indice_bloco"])
        block_id = int(row["patamar"]) - 1
        val = float(row["valor"])
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)
        factor_map[(sub_code, bloco, yr, cal_month, block_id)] = val

    # Last-year seasonal fallback.
    last_yr_map: dict[tuple[int, int, int, int], tuple[int, float]] = {}
    for (sub_code, bloco, yr, cal_month, block_id), val in factor_map.items():
        k4 = (sub_code, bloco, cal_month, block_id)
        existing = last_yr_map.get(k4)
        if existing is None or yr > existing[0]:
            last_yr_map[k4] = (yr, val)

    last_factor: dict[tuple[int, int, int, int], float] = {
        k: v for k, (_, v) in last_yr_map.items()
    }

    # Use the shared canonical NCS group -> ID mapping to guarantee consistency
    # with convert_non_controllable_sources and convert_ncs_stats.
    ncs_group_map = _build_ncs_group_to_id(nw_files, id_map)

    results: list[dict] = []

    for (sub_code, bloco), ncs_id in sorted(ncs_group_map.items(), key=lambda x: x[1]):

        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            block_factors: list[dict] = []
            for block_id in range(num_blocks):
                if is_post_study:
                    factor = last_factor.get((sub_code, bloco, m, block_id), 1.0)
                else:
                    factor = factor_map.get(
                        (sub_code, bloco, y, m, block_id),
                        last_factor.get((sub_code, bloco, m, block_id), 1.0),
                    )
                block_factors.append(
                    {"block_id": block_id, "factor": max(factor, 1e-6)}
                )

            results.append(
                {
                    "ncs_id": ncs_id,
                    "stage_id": stage_id,
                    "block_factors": block_factors,
                }
            )

            m += 1
            if m > 12:
                m = 1
                y += 1

    return {"non_controllable_factors": results}


def convert_ncs_stats(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert sistema.dat NCS generation to ``non_controllable_stats.parquet``.

    Produces the stochastic availability model for each NCS entity.  Since
    NEWAVE NCS generation is deterministic, ``std`` is always 0.0 and
    ``mean`` is the availability factor: ``available_mw / max_generation_mw``.

    NCS IDs are assigned using the same ``(codigo_submercado, indice_bloco)``
    sorted grouping as ``convert_non_controllable_sources``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Entity ID map (unused directly; kept for API consistency).

    Returns
    -------
    pyarrow.Table
        Columns: ``ncs_id`` (INT32), ``stage_id`` (INT32),
        ``mean`` (DOUBLE), ``std`` (DOUBLE).
    """
    from inewave.newave import Dger, Sistema

    _NCS_STATS_SCHEMA = pa.schema(
        [
            pa.field("ncs_id", pa.int32()),
            pa.field("stage_id", pa.int32()),
            pa.field("mean", pa.float64()),
            pa.field("std", pa.float64()),
        ]
    )

    sistema = Sistema.read(str(nw_files.sistema))
    df_raw: pd.DataFrame | None = sistema.geracao_usinas_nao_simuladas

    if df_raw is None or df_raw.empty:
        return pa.table(
            {
                "ncs_id": pa.array([], type=pa.int32()),
                "stage_id": pa.array([], type=pa.int32()),
                "mean": pa.array([], type=pa.float64()),
                "std": pa.array([], type=pa.float64()),
            },
            schema=_NCS_STATS_SCHEMA,
        )

    dger = Dger.read(nw_files.dger)
    start_month: int = dger.mes_inicio_estudo
    start_year: int = dger.ano_inicio_estudo
    num_anos: int = dger.num_anos_estudo or 1
    num_anos_pos: int = dger.num_anos_pos_estudo or 0
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Columns: codigo_submercado, indice_bloco, fonte, data, valor
    # Build per-(sub_code, bloco, yr, cal_month) -> valor lookup.
    # year == 9999 rows are stored with key yr=9999 for post-study seasonal
    # repeat logic handled below.
    BoundsKey = tuple  # (sub_code, bloco, yr, cal_month)
    bounds_map: dict[BoundsKey, float] = {}

    for _, row in df_raw.iterrows():
        val_raw = row["valor"]
        if _is_na(val_raw):
            continue
        val = float(val_raw)
        sub_code = int(row["codigo_submercado"])
        bloco = int(row["indice_bloco"])
        dt = row["data"]
        yr = int(dt.year)
        cal_month = int(dt.month)
        bounds_map[(sub_code, bloco, yr, cal_month)] = val

    # Build last-year seasonal fallback (for post-study and missing study stages).
    # Use year == 9999 rows preferentially as post-study entries; otherwise use
    # the highest real year available.
    last_yr_bounds: dict[tuple[int, int, int], tuple[int, float]] = {}
    for (sub_code, bloco, yr, cal_month), val in bounds_map.items():
        k3 = (sub_code, bloco, cal_month)
        existing = last_yr_bounds.get(k3)
        if existing is None or yr > existing[0]:
            last_yr_bounds[k3] = (yr, val)

    last_bounds: dict[tuple[int, int, int], float] = {
        k: v for k, (_, v) in last_yr_bounds.items()
    }

    # Use the shared canonical NCS group -> ID mapping.
    ncs_group_map = _build_ncs_group_to_id(nw_files, id_map)

    # Compute max_generation_mw per NCS entity.
    max_gen_per_ncs: dict[int, float] = {}
    for (sub_code, bloco), ncs_id in ncs_group_map.items():
        vals = [
            v
            for (sc, bl, _yr, _cm), v in bounds_map.items()
            if sc == sub_code and bl == bloco
        ]
        max_gen_per_ncs[ncs_id] = max(vals) if vals else 0.0

    rows_ncs_id: list[int] = []
    rows_stage_id: list[int] = []
    rows_mean: list[float] = []
    rows_std: list[float] = []

    for (sub_code, bloco), ncs_id in sorted(
        ncs_group_map.items(), key=lambda x: x[1]
    ):
        max_gen = max_gen_per_ncs[ncs_id]

        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            if is_post_study:
                gen_mw = last_bounds.get((sub_code, bloco, m))
            else:
                gen_mw = bounds_map.get((sub_code, bloco, y, m))
                if gen_mw is None:
                    gen_mw = last_bounds.get((sub_code, bloco, m))

            if gen_mw is not None:
                mean = gen_mw / max_gen if max_gen > 0 else 0.0
                mean = max(0.0, min(1.0, mean))
                rows_ncs_id.append(ncs_id)
                rows_stage_id.append(stage_id)
                rows_mean.append(mean)
                rows_std.append(0.0)

            m += 1
            if m > 12:
                m = 1
                y += 1

    return pa.table(
        {
            "ncs_id": pa.array(rows_ncs_id, type=pa.int32()),
            "stage_id": pa.array(rows_stage_id, type=pa.int32()),
            "mean": pa.array(rows_mean, type=pa.float64()),
            "std": pa.array(rows_std, type=pa.float64()),
        },
        schema=_NCS_STATS_SCHEMA,
    )


def _subsystem_name_from_id(subsystem_code: int) -> str:
    """Return a short name string for a subsystem code (fallback to str)."""
    return str(subsystem_code)


def _is_na(value: object) -> bool:
    """Return True if *value* is a pandas NA/NaN sentinel."""
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return True
    except (TypeError, ValueError):
        pass
    try:
        import pandas as pd

        return pd.isna(value)  # type: ignore[return-value]
    except (TypeError, ImportError):
        return False
