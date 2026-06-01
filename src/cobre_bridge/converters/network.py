"""Network entity converter: maps NEWAVE bus and line data to Cobre network JSON."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence

import pandas as pd
import pyarrow as pa
from inewave.newave import Dger, Sistema

from cobre_bridge.horizon import POST_STUDY_YEAR, study_horizon
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles
from cobre_bridge.pandas_utils import is_na

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
_NCS_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/non_controllable_sources.schema.json"
)
_EXCHANGE_FACTORS_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/exchange_factors.schema.json"
)
_NCS_FACTORS_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/non_controllable_factors.schema.json"
)

# --------------------------------------------------------------------------
# Penalty conversion constants
# --------------------------------------------------------------------------
#
# Source: NEWAVE User Manual v30 (CEPEL/ONS, 2023), section 3.24 "Penalidades
# (Ex.: Penalid.dat)" and the internal-default tables on pages 87–88.
#
# Time-aspect summary (re-derived from cobre/crates/cobre-sddp/src/lp_builder/
# matrix.rs in current HEAD):
#
# Every cobre penalty coefficient in penalties.json is multiplied by some
# hours quantity before entering the LP objective. The variable it sits on
# (MW, m³/s, or hm³) and the time multiplier (block_hours, total_stage_hours,
# or none) together determine the unit the user-facing value must be in.
# The pattern is "(penalty × hours) × variable_value = R$":
#
# Family A — Power columns (MW), per block. Variable carries MW for one block.
#   Cobre: `objective[col] = penalty × block_hours`
#   Cost  = (penalty × block_h) × MW = penalty × MWh → penalty unit R$/MWh.
#   Affected: bus.deficit_segments[].cost, bus.excess_cost, line.exchange_cost,
#   ncs.curtailment_cost, hydro.generation_violation_below_cost.
#   → Conversion from NEWAVE R$/MWh: **direct** (no productivity factor).
#
# Family B — Flow columns (m³/s), per block. Variable carries m³/s for one block.
#   Cobre: `objective[col] = penalty × block_hours`
#   Cost  = (penalty × block_h) × m³/s.  For this to equal R$ the penalty
#   must be R$/(m³/s · h). NEWAVE supplies R$/MWh; the per-flow-per-hour
#   form requires multiplying by ρ [MW/(m³/s)]:
#     R$/MWh × MW/(m³/s) = R$/(m³/s · h).
#   Affected: hydro.spillage_cost, hydro.turbined_cost, hydro.diversion_cost,
#   hydro.outflow_violation_(below|above)_cost, hydro.turbined_violation_below_cost.
#   → Conversion: **× ρ_avg** (`PROD_MEDIA_SIN`).
#
# Family C — Flow columns (m³/s), per stage. Same as B but cobre uses
# `total_stage_hours` instead of `block_hours`.
#   Affected: hydro.water_withdrawal_violation_(pos|neg)_cost,
#   hydro.evaporation_violation_(pos|neg)_cost,
#   hydro.inflow_nonnegativity_cost.
#   Cobre's docstring on evaporation_violation_cost says "$/mm" but the
#   actual LP column (matrix.rs:346-347) reads f_evap_plus/minus as flow rates
#   in m³/s — same unit as withdrawal. The "_m3s" suffix in the simulation
#   output `evaporation_violation_pos_m3s` confirms this.
#   → Conversion: **× ρ_max_acum** (`MAX_PRODTACUM_SIN`) for DESVIO and
#     evaporation (per NEWAVE manual p.87); **× ρ_avg** for the others.
#
# Family D — Volume columns (hm³), per stage. Variable carries hm³ once per stage.
#   Cobre: `objective[col] = penalty` (no time multiplier).
#   Cost  = penalty × hm³ = R$ → penalty unit R$/hm³.
#   Affected: hydro.storage_violation_below_cost, hydro.filling_target_violation_cost
#   (both currently NOT wired into the LP — slot is dormant).
#   Conversion: 1 hm³ × ρ → MWh of energy-equivalent is `(1e6 m³ / 3600 s/h) × ρ`
#   = 277.78 × ρ MWh (purely volumetric — 730h convention cancels). So
#     cobre_coef [R$/hm³] = NEWAVE_R$/MWh × ρ × HM3_TO_MWH_PER_RHO
#   with HM3_TO_MWH_PER_RHO = 1e6/3600 ≈ 277.78.
#
# NEWAVE's 730 h-per-month convention vs cobre's actual calendar block_hours
# (672–744 h) introduces only a ±2% numerical drift in absolute LP cost. It
# does not change merit order or LP decisions (all costs scale together).
# The HM3 → MWh conversion is pure volumetric and the 730 cancels out.
#
# Merit order from NEWAVE micro-penalty values ("NEWAVE individualizado"
# column, manual §3.24 p.88, v30 defaults):
#   p_INT (0.000273) < p_PFIO = p_EVERT (0.000300)
#   < p_TURB (0.000333) < p_CORTEOL (0.000344) < p_EXC (0.000355)

MONTH_HOURS: float = 730.0  # NEWAVE convention (manual §3.24, used in C_M3S2HM3)
C_M3S2HM3: float = MONTH_HOURS * 3600.0 / 1e6  # = 2.628 hm³ / (m³/s · month)
# HM3 × ρ_MW_per_m3s → MWh conversion (purely volumetric; 730 cancels here).
HM3_TO_MWH_PER_RHO: float = 1e6 / 3600.0  # ≈ 277.78

# --- NEWAVE micro-penalties (page 88, current v30) -------------------------
# Energy-domain (R$/MWh) — passed through to cobre without conversion.
# Flow-domain (multiplied by ρ_avg before emission) — see `_PEVERT` group.
#
# These are NEWAVE's v30 values verbatim: tiny (~1e-4 R$/MWh) regularization
# costs whose only role is to break LP ties in a fixed merit order
# (exchange < spillage < … < excess), well below any operational or deterrent
# cost. An earlier revision multiplied them by a uniform uplift factor to widen
# the HiGHS coefficient range; that factor was reverted to 1.0 (a no-op) and is
# now dropped — the bare NEWAVE values condition fine at cobre's case scale.
_PINT = 0.000273  # intercâmbio  → line.exchange_cost

# NEWAVE halves the intercâmbio penalty on lines that touch a fictitious
# submercado (e.g. NOFICT1). Rationale: a fictitious node is a routing-only
# hop with no demand or generation of its own, so a real → fict → real path
# would otherwise accumulate twice the penalty of an equivalent direct
# real → real link. The 0.5 discount restores cost-parity between the two
# topologies.  Emitted as the per-line `exchange_cost` override defined in
# lines.schema.json; absence falls back to the global `_PINT` value.
_PINT_FICTITIOUS_DISCOUNT = 0.5
_PCORTEOL = 0.000344  # corte geração eólica → ncs.curtailment_cost
_PEXC = 0.000355  # excesso de energia → bus.excess_cost

# Flow-domain (R$/MWh equivalent, multiplied by ρ_avg before emission).
# Cobre's `hydro.spillage_cost` covers ALL spillage (reservoir + run-of-river).
# In the *individualized* model (manual §3.24, p.88, "NEWAVE individualizado"
# column) BOTH controllable (pEVERT) and run-of-river (pPFIO) spillage use the
# same base 0.000300 — only the REE-aggregated ("NEWAVE equivalente") column
# raises pEVERT to 0.000327. Cobre cases are individualized, so anchor on 0.000300.
_PEVERT = 0.000300  # vertimento controlável → hydro.spillage_cost
_PTURB = 0.000333  # turbinamento → hydro.turbined_cost (applied to every hydro)
_PCDESV = 0.000300  # volume desviado → hydro.diversion_cost

# --- NEWAVE hard-coded internal defaults (no user input via PENALID) -------
# Page 87: evaporation and FPHA folga both derive from MAX_CUSTO_DEFICIT, and
# NEWAVE's manual prescribes a 10× multiplier — the evap/FPHA folga slack is
# ~10× more expensive per MWh-equivalent than the deficit cost, putting these
# physical-law constraints (water-cycle physics, water-supply requirements) at
# the top of the merit order so the LP violates them only as a last resort.
# We apply that 10× faithfully (it doubles as the PENALID fallback below, and
# `_ELETRI_HIGH_MULT` reuses the same magnitude). Earlier revisions trialled
# softer factors (1.1×, 2×) to tame HiGHS's coefficient range, but the bare
# 10× conditions acceptably at cobre's case scale and stays NEWAVE-faithful.
_EVAPORATION_MULT = 10.0

# NOTE: when PENALID supplies TURBMN, VAZMIN, TURBMX with the same R$/MWh value
# (typical NEWAVE convention), the resulting turbined/outflow-below/outflow-above
# slack costs share an LP coefficient (ρ_avg cancels nothing). An earlier
# revision multiplied each by a ~1 % "tie-break" factor to break that
# degeneracy; the factors were all reverted to 1.00 (no-op) and have been
# dropped. Reintroduce distinct spacing here if HiGHS degeneracy resurfaces.
# --- Cobre fields not yet wired into the LP --------------------------------
# Storage and filling-target violation costs are declared on cobre's schema
# but `lp_builder/matrix.rs` does NOT use them in the objective (all 0.0
# at build time). We still emit sensible values so the case is ready for
# the day cobre wires these in.
_DEFAULT_STORAGE_VIOLATION_BELOW_COST = 1.0e6  # R$/hm³ (dormant in cobre LP)
_DEFAULT_FILLING_TARGET_VIOLATION_COST = 1.0e7  # R$/hm³ (dormant in cobre LP)

# --- Soft fallback for ELETRI when PENALID is silent -----------------------
# NEWAVE's behaviour when ELETRI is absent: use the constraint only in final
# simulation, not in policy. Cobre can't represent that nuance, so we keep
# the slack enabled with a high penalty (10 × MAX_DEFICIT, matching
# NEWAVE's evaporation/FPHA default magnitude).
_ELETRI_HIGH_MULT = 10.0

# --- Inflow non-negativity penalty anchor ---------------------------------
# inflow_nonnegativity_cost is set to ``water_withdrawal_violation_cost + 1``
# (R$/m³/s) so that the LP can never use the inflow slack as a cheaper
# substitute for failing the deterministic withdrawal schedule.
#
# The previous "1 % above max(flow slacks)" rule scaled the slack to be a
# soft margin, which is the wrong design for this column: the inflow
# non-negativity slack physically *generates* water in the LP — every other
# flow slack just *absorbs* a constraint violation — so giving it a price
# even infinitesimally cheaper than some violation makes it exchangeable
# with that violation.  The +1 R$/m³/s constant offset is just enough to
# break the tie deterministically against withdrawal (the strictest
# slack with the largest spread on the case set we calibrate against).
#
# This anchor still doesn't eliminate the exploit (the slack remains a
# free water column in principle); ``modeling.inflow_non_negativity.method``
# = ``truncation`` is the real fix.  This setting is the conservative
# choice for the ``penalty`` method.
_INFLOW_NN_OFFSET_R_PER_M3S = 1.0


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
            float(corte) if corte is not None and not is_na(corte) else None
        )
        cost_raw = row["custo"]
        cost = float(cost_raw) if not is_na(cost_raw) else None
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


def convert_bus_penalty_overrides(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build ``constraints/penalty_overrides_bus.parquet`` for fictitious buses.

    NEWAVE fictitious submarkets (``custo_deficit.ficticio``) are pure
    transshipment nodes: no real load/generation, and NEWAVE forbids energy
    excess there.  Cobre has no hard per-bus "no excess" flag, so we override the
    per-bus ``excess_cost`` (sparse, per stage) to the deficit cost — symmetric
    with unserved energy — without which Cobre dumps surplus energy at the
    fictitious node for ~free.

    Returns ``None`` when the case has no fictitious submarkets (no file
    emitted, mirroring the sparse-override contract).

    Returns
    -------
    pyarrow.Table | None
        Columns: ``bus_id`` (INT32), ``stage_id`` (INT32),
        ``excess_cost`` (DOUBLE) — one row per (fictitious bus, stage).
    """
    sistema = Sistema.read(str(nw_files.sistema))
    deficit_df = sistema.custo_deficit

    if deficit_df is None or deficit_df.empty or "ficticio" not in deficit_df.columns:
        return None

    fic_mask = deficit_df["ficticio"].fillna(False).astype(bool)
    fictitious_codes = {
        int(code) for code in deficit_df.loc[fic_mask, "codigo_submercado"].unique()
    }
    if not fictitious_codes:
        return None

    # Excess at a fictitious node is penalised at the deficit cost — symmetric
    # with unserved energy, and equal to the deficit_segments cost convert_buses
    # assigns to these buses. Uses the reference (first non-null, positive)
    # deficit cost from custo_deficit.
    deficit_costs = [
        float(c)
        for c in deficit_df["custo"].tolist()
        if c is not None and not is_na(c) and float(c) > 0.0
    ]
    if not deficit_costs:
        return None
    excess_cost = deficit_costs[0]

    dger = Dger.read(nw_files.dger)
    total_stages = study_horizon(dger).total_stages

    bus_ids: list[int] = []
    stage_ids: list[int] = []
    costs: list[float] = []
    for code in sorted(fictitious_codes):
        try:
            bus_id = id_map.bus_id(code)
        except KeyError:
            continue
        for stage_id in range(total_stages):
            bus_ids.append(bus_id)
            stage_ids.append(stage_id)
            costs.append(excess_cost)

    if not bus_ids:
        return None

    schema = pa.schema(
        [
            pa.field("bus_id", pa.int32()),
            pa.field("stage_id", pa.int32()),
            pa.field("excess_cost", pa.float64()),
        ]
    )
    return pa.table(
        {
            "bus_id": pa.array(bus_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "excess_cost": pa.array(costs, type=pa.float64()),
        },
        schema=schema,
    )


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

    # Set of NEWAVE codes flagged as fictitious in sistema.custo_deficit.
    # Used below to halve the exchange penalty on lines that touch a
    # fictitious bus (see `_PINT_FICTITIOUS_DISCOUNT`).
    fictitious_codes: set[int] = set()
    deficit_df = sistema.custo_deficit
    if (
        deficit_df is not None
        and not deficit_df.empty
        and "ficticio" in deficit_df.columns
    ):
        fic_mask = deficit_df["ficticio"].fillna(False).astype(bool)
        fictitious_codes = {
            int(code) for code in deficit_df.loc[fic_mask, "codigo_submercado"].unique()
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
    # In inewave's parsed SISTEMA.DAT, sentido == 0 is the first block
    # (de -> para, i.e. A->B) and sentido == 1 is the second block
    # (para -> de, i.e. B->A).
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
            if sentido == 0:
                pair_map[key]["direct_mw"] = valor
            else:
                pair_map[key]["reverse_mw"] = valor
        else:
            # de -> para is the "reverse" direction.
            if sentido == 0:
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
        if src in fictitious_codes or tgt in fictitious_codes:
            line_entry["exchange_cost"] = _PINT * _PINT_FICTITIOUS_DISCOUNT
        lines.append(line_entry)

    return {
        "$schema": _LINES_SCHEMA_URL,
        "lines": lines,
    }


def _read_penalid_costs(nw_files: NewaveFiles) -> dict[str, float]:
    """Pull ``{variable_name: first non-null R$/MWh value}`` from PENALID.DAT.

    Falls back to an empty dict if the file is absent or unparseable. Each
    PENALID variable can have per-REE / per-patamar values; we pick the first
    non-null R$/MWh entry as the global default the same way NEWAVE does for
    REE-aggregated penalty handling.
    """
    from inewave.newave import Penalid

    if nw_files.penalid is None:
        return {}
    try:
        penalid = Penalid.read(str(nw_files.penalid))
    except (OSError, ValueError) as exc:
        _LOG.warning("penalid.dat could not be parsed (%s); using defaults.", exc)
        return {}

    pen_df = penalid.penalidades
    if pen_df is None or pen_df.empty:
        return {}

    out: dict[str, float] = {}
    for var in pen_df["variavel"].unique():
        rows = pen_df[(pen_df["variavel"] == var) & pen_df["valor_R$_MWh"].notna()]
        if not rows.empty:
            out[str(var).strip()] = float(rows.iloc[0]["valor_R$_MWh"])
    return out


def _own_productivities(
    hydros_dict: dict, productivities: dict[int, float]
) -> list[float]:
    """Return the list of per-hydro own productivities for averaging.

    Preferred source is the `productivities` map (new contract). Falls back
    to the legacy `hydros.json:generation.productivity_mw_per_m3s` field per
    entry so older test fixtures keep working.
    """
    out: list[float] = []
    for h in hydros_dict.get("hydros", []):
        prod = 0.0
        if "id" in h:
            prod = productivities.get(int(h["id"]), 0.0)
        if prod <= 0.0:
            legacy = h.get("generation", {}).get("productivity_mw_per_m3s")
            if legacy is not None:
                prod = float(legacy)
        if prod > 0:
            out.append(prod)
    return out


def _hydro_penalty_costs(
    *,
    rho_avg: float,
    rho_max_acum: float,
    penalid_costs: Mapping[str, float],
    max_deficit_cost: float,
) -> dict[str, float]:
    """Compute the ρ-scaled hydro penalty block for one (ρ_avg, ρ_max_acum).

    Pure function shared by :func:`convert_penalties` (global/base defaults in
    ``penalties.json``) and :func:`convert_hydro_penalty_overrides` (per-stage
    overrides in ``constraints/penalty_overrides_hydro.parquet``). Keeping a
    single formula site guarantees the two paths never drift: the override is
    exactly the base recomputed with a different productivity pair.

    Convention from NEWAVE manual p.87:

    - ``DESVIO`` → ``× MAX_PRODTACUM_SIN`` (water withdrawal).
    - ``VAZMIN`` / ``TURBMN`` / ``TURBMX`` / ``VOLMIN`` → ``× PROD_MEDIA_SIN``.
    - ``GHMIN`` → energy-domain slack, no productivity multiplier.
    - Micro-penalties ``pEVERT`` / ``pTURB`` / ``pCDESV`` → ``× PROD_MEDIA_SIN``.

    The returned dict preserves the exact key order of ``penalties.json:hydro``.
    """
    desvio_mwh = penalid_costs.get("DESVIO", _EVAPORATION_MULT * max_deficit_cost)
    vazmin_mwh = penalid_costs.get("VAZMIN", _EVAPORATION_MULT * max_deficit_cost)
    ghmin_mwh = penalid_costs.get("GHMIN", _EVAPORATION_MULT * max_deficit_cost)
    turbmn_mwh = penalid_costs.get("TURBMN", _EVAPORATION_MULT * max_deficit_cost)
    turbmx_mwh = penalid_costs.get("TURBMX", _EVAPORATION_MULT * max_deficit_cost)

    water_withdrawal_cost = desvio_mwh * rho_max_acum
    outflow_below_cost = vazmin_mwh * rho_avg
    outflow_above_cost = turbmx_mwh * rho_avg
    turbined_below_cost = turbmn_mwh * rho_avg
    generation_below_cost = ghmin_mwh  # energy-domain, no productivity factor

    # Storage / filling target: cobre slots are dormant in the LP today but
    # we still populate them. VOLMIN comes from PENALID when set; otherwise
    # use a high default.
    #
    # Conversion from NEWAVE R$/MWh to cobre R$/hm³ is purely volumetric:
    # 1 hm³ of stored water released through the cascade yields
    #   1e6 m³ × ρ MW/(m³/s) × 1/3600 s/h = (1e6/3600) × ρ MWh
    # so cobre_coef = P_R$_MWh × ρ × (1e6/3600). The 730h/month assumption
    # cancels out — this is dimensional energy-equivalence, not a per-hour
    # rate. (Earlier versions used × C_M3S2HM3 here which was wrong by
    # the factor MONTH_HOURS = 730.)
    volmin_mwh = penalid_costs.get("VOLMIN")
    storage_below_cost = (
        volmin_mwh * rho_avg * HM3_TO_MWH_PER_RHO
        if volmin_mwh is not None
        else _DEFAULT_STORAGE_VIOLATION_BELOW_COST
    )

    # Evaporation violation: no PENALID variable. NEWAVE manual p.87 prescribes
    # `(K × MAX_CUSTO_DEFICIT × MAX_PRODTACUM_SIN) / C_M3S2HM3` with K=10, which
    # we apply faithfully (K == `_EVAPORATION_MULT`) so the deterrent magnitude
    # sits above the PENALID-sourced operational slacks (typically
    # water_withdrawal_violation_cost). The /C_M3S2HM3 step is dropped (only
    # needed for NEWAVE's per-hm³ slot).
    evaporation_cost = _EVAPORATION_MULT * max_deficit_cost * rho_max_acum

    # Flow-domain micro-penalties (× ρ_avg per NEWAVE individualized conversion).
    spillage_cost = _PEVERT * rho_avg
    turbined_cost = _PTURB * rho_avg
    diversion_cost = _PCDESV * rho_avg

    # Inflow non-negativity: NEWAVE has no PENALID variable for this. Cobre's
    # default is 1000 R$/(m³/s · h), which is far below the operationally-
    # significant flow-domain slacks above (turbined / outflow / evaporation /
    # water-withdrawal). When the LP can choose between letting incremental
    # natural inflow go negative (a non-physical phenomenon — implies upstream
    # is somehow "absorbing water" the cascade can't account for) and
    # violating another flow constraint, we want it to *always* prefer fixing
    # the other constraint first. Anchor inflow non-negativity penalty to the
    # withdrawal slack plus a 1 R$/m³/s tie-breaker. The inflow_nn slack
    # physically *generates* water in the LP, so making it cheaper than the
    # withdrawal slack would let the LP buy free water to dodge a withdrawal
    # violation — an artefact with no real-world counterpart. Keeping it just
    # above withdrawal preserves a strict ordering against the strictest
    # operational flow slack without inflating the objective when this slack
    # genuinely needs to fire. Real fix: switch
    # ``modeling.inflow_non_negativity.method`` to ``truncation``.
    inflow_nonnegativity_cost = water_withdrawal_cost + _INFLOW_NN_OFFSET_R_PER_M3S

    return {
        "spillage_cost": spillage_cost,
        "turbined_cost": turbined_cost,
        "diversion_cost": diversion_cost,
        "storage_violation_below_cost": storage_below_cost,
        "filling_target_violation_cost": _DEFAULT_FILLING_TARGET_VIOLATION_COST,
        "turbined_violation_below_cost": turbined_below_cost,
        "outflow_violation_below_cost": outflow_below_cost,
        "outflow_violation_above_cost": outflow_above_cost,
        "generation_violation_below_cost": generation_below_cost,
        "evaporation_violation_cost": evaporation_cost,
        "water_withdrawal_violation_cost": water_withdrawal_cost,
        "inflow_nonnegativity_cost": inflow_nonnegativity_cost,
    }


# Hydro penalty columns whose value scales with productivity (ρ_avg or
# ρ_max_acum) and is therefore stage-varying. Energy-domain slots
# (``generation_violation_below_cost``) and ρ-independent defaults
# (``filling_target_violation_cost``, and ``storage_violation_below_cost`` when
# VOLMIN is unset) are intentionally excluded — they never differ per stage so
# the sparse override never emits them.
_RHO_SCALED_HYDRO_COLUMNS: tuple[str, ...] = (
    "spillage_cost",
    "turbined_cost",
    "diversion_cost",
    "storage_violation_below_cost",
    "turbined_violation_below_cost",
    "outflow_violation_below_cost",
    "outflow_violation_above_cost",
    "evaporation_violation_cost",
    "water_withdrawal_violation_cost",
    "inflow_nonnegativity_cost",
)


def convert_penalties(
    nw_files: NewaveFiles,
    hydros_dict: dict,
    productivities: dict[int, float] | None = None,
    *,
    max_accumulated_productivity: float | None = None,
    prod_media_sin: float | None = None,
) -> dict:
    """Generate a Cobre ``penalties.json`` dict from NEWAVE data.

    Faithful to NEWAVE User Manual v30 section 3.24:

    - Bus deficit segments come from ``sistema.custo_deficit`` directly
      (R$/MWh on both sides, no conversion).
    - PENALID-sourced flow-domain penalties are converted to cobre's
      coefficient slot via ``× ρ`` (where ρ is ``PROD_MEDIA_SIN`` or
      ``MAX_PRODTACUM_SIN`` per the NEWAVE conversion table on page 87).
    - The micro-penalties (``pINT``, ``pEVERT``, ``pTURB``, ``pCORTEOL``,
      ``pEXC``, ``pCDESV``) are NEWAVE's hard-coded internal defaults
      (page 88, current v30 values). They are written directly to cobre
      and preserve NEWAVE's merit order: exchange < spillage < FPHA <
      curtailment < excess.
    - Evaporation and storage-violation slots without a PENALID source
      fall back to ``10 × MAX_CUSTO_DEFICIT × ρ_max_acum`` (evaporation,
      per the manual) or ``1e6`` (storage/filling — dormant in cobre's LP
      today but emitted so the file is ready when cobre wires them in).

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    hydros_dict:
        The already-converted ``hydros.json`` dict (used for reservoir
        useful-volume weights and the productivity fallback).
    productivities:
        ``{hydro_id: own_productivity_mw_per_m3s}`` for each hydro. Required
        because productivity moved out of ``hydros.json:generation`` on
        cobre HEAD.
    max_accumulated_productivity:
        Optional ``MAX_PRODTACUM_SIN`` override. When omitted, defaults to
        ``max(productivities)`` — a coarse approximation; callers with
        access to the cascade DAG should pass the true accumulated max.
    """
    sistema = Sistema.read(str(nw_files.sistema))
    deficit_df = sistema.custo_deficit

    # Primary deficit cost: first subsystem, first patamar.
    primary_deficit_cost = 0.0
    max_deficit_cost = 0.0
    if deficit_df is not None and not deficit_df.empty:
        first_sub = deficit_df.sort_values(
            [
                "codigo_submercado",
                "patamar_deficit",
            ]
        )
        primary_deficit_cost = float(first_sub.iloc[0]["custo"])
        max_deficit_cost = float(deficit_df["custo"].max())

    penalid_costs = _read_penalid_costs(nw_files)
    productivities = productivities or {}

    # ρ_avg = PROD_MEDIA_SIN: NEWAVE converts the PENALID R$/MWh penalties
    # (VAZMIN, TURBMN, TURBMX, VOLMIN — manual table p.87) with the mean **PRODT**
    # (equivalent productivity vol_min→vol_max) over ALL existing plants including
    # zeros. Validated against pmo.dat's applied penalties (0.6299 ↔ penalty
    # 821.78). Pass it in via ``prod_media_sin`` (see ``hydro.compute_prodt_sin_mean``).
    # The legacy fallback — mean of the 65%-reference own ρ, ρ>0 only — is ~4%
    # high; kept only for callers/tests that don't supply ``prod_media_sin``.
    own_prods = _own_productivities(hydros_dict, productivities)
    rho_avg = (
        prod_media_sin
        if prod_media_sin is not None
        else (sum(own_prods) / len(own_prods) if own_prods else 1.0)
    )

    # ρ_max_acum = MAX_PRODTACUM_SIN: used by NEWAVE for DESVIO and the
    # evaporation default. When the caller doesn't supply the true cascade
    # accumulated max we approximate by `max(own_prods)`; the caller in
    # `pipeline.py` passes the real value computed from the cascade DAG.
    rho_max_acum = (
        max_accumulated_productivity
        if max_accumulated_productivity is not None
        else (max(own_prods) if own_prods else rho_avg)
    )

    # The ρ-scaled hydro penalty block is computed by a pure helper so the
    # global (base) penalties.json and the per-stage override parquet built by
    # ``convert_hydro_penalty_overrides`` apply byte-identical formulas — see
    # ``_hydro_penalty_costs``.
    hydro_costs = _hydro_penalty_costs(
        rho_avg=rho_avg,
        rho_max_acum=rho_max_acum,
        penalid_costs=penalid_costs,
        max_deficit_cost=max_deficit_cost,
    )

    # --------------------------------------------------------------------
    # NEWAVE micro-penalty defaults (page 88, current v30) — energy-domain
    # (R$/MWh), pass through directly (no productivity multiplier, hence not
    # stage-varying and not part of the hydro override).
    # --------------------------------------------------------------------
    excess_cost = _PEXC
    exchange_cost = _PINT
    curtailment_cost = _PCORTEOL

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
        "hydro": hydro_costs,
        "line": {
            "exchange_cost": exchange_cost,
        },
        "non_controllable_source": {
            "curtailment_cost": curtailment_cost,
        },
    }


def convert_hydro_penalty_overrides(
    nw_files: NewaveFiles,
    hydro_ids: Sequence[int],
    base_hydro_penalties: Mapping[str, float],
    per_stage_rho_avg: Sequence[float],
    per_stage_rho_max_acum: Sequence[float],
) -> pa.Table | None:
    """Build ``constraints/penalty_overrides_hydro.parquet`` (per-stage ρ).

    NEWAVE converts its flow-domain hydro penalties (spillage, turbined,
    diversion, the outflow/turbined/storage/withdrawal/evaporation slacks)
    using the **SIN-aggregate** productivity constants ``PROD_MEDIA_SIN`` and
    ``MAX_PRODTACUM_SIN``. Those constants are *not* fixed across the horizon:
    each plant's equivalent productivity tracks its seasonal reference volume
    (VOLREF_SAZ) and any CFUGA/CMONT tailrace/forebay overrides, so the SIN
    mean / accumulated-max shift stage to stage. cobre-bridge already ships the
    per-stage per-plant ρ in ``system/hydro_energy_productivity.parquet``; this
    override makes the **penalty** conversion use the same per-stage ρ, instead
    of a single static fleet mean — keeping the two coherent.

    The override is SIN-uniform (one value per stage, applied to every hydro,
    matching NEWAVE's use of a single SIN constant) and **sparse**: a row is
    emitted only for ``(hydro, stage)`` pairs and columns whose value actually
    differs from the global ``penalties.json`` default. ρ-independent slots
    (``generation_violation_below_cost``, ``filling_target_violation_cost``,
    and ``storage_violation_below_cost`` when VOLMIN is unset) never differ and
    are never emitted.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths (re-reads ``sistema`` for the max deficit
        cost and PENALID for the violation-slack base rates).
    hydro_ids:
        Every Cobre hydro id the SIN-uniform override must cover. Sorted
        ascending internally so output obeys the ``(hydro_id, stage_id)``
        ordering contract.
    base_hydro_penalties:
        The global ``penalties.json:hydro`` block — used as the diff baseline
        so only genuinely stage-varying values are emitted.
    per_stage_rho_avg:
        ``PROD_MEDIA_SIN[s]`` — mean own ρ over online plants at stage ``s``.
    per_stage_rho_max_acum:
        ``MAX_PRODTACUM_SIN[s]`` — max accumulated cascade ρ at stage ``s``.

    Returns
    -------
    pyarrow.Table | None
        Columns ``hydro_id`` (INT32), ``stage_id`` (INT32), and one DOUBLE
        column per ρ-scaled penalty that varies. ``None`` when nothing differs
        from the base (e.g. no seasonal/temporal productivity effects).
    """
    if not hydro_ids or not per_stage_rho_avg:
        return None
    n_stages = len(per_stage_rho_avg)
    if len(per_stage_rho_max_acum) != n_stages:
        raise ValueError(
            "per_stage_rho_avg and per_stage_rho_max_acum must be the same "
            f"length ({n_stages} vs {len(per_stage_rho_max_acum)})"
        )

    sistema = Sistema.read(str(nw_files.sistema))
    deficit_df = sistema.custo_deficit
    max_deficit_cost = (
        float(deficit_df["custo"].max())
        if deficit_df is not None and not deficit_df.empty
        else 0.0
    )
    penalid_costs = _read_penalid_costs(nw_files)

    # Recompute the full hydro penalty block per stage and keep only ρ-scaled
    # columns that differ from the global base (sparse-override contract).
    stage_overrides: list[tuple[int, dict[str, float]]] = []
    for s in range(n_stages):
        stage_costs = _hydro_penalty_costs(
            rho_avg=per_stage_rho_avg[s],
            rho_max_acum=per_stage_rho_max_acum[s],
            penalid_costs=penalid_costs,
            max_deficit_cost=max_deficit_cost,
        )
        diff: dict[str, float] = {}
        for col in _RHO_SCALED_HYDRO_COLUMNS:
            base_v = base_hydro_penalties.get(col)
            new_v = stage_costs.get(col)
            if base_v is None or new_v is None:
                continue
            if not math.isclose(new_v, base_v, rel_tol=1e-12, abs_tol=0.0):
                diff[col] = new_v
        if diff:
            stage_overrides.append((s, diff))

    if not stage_overrides:
        return None

    present_cols = [
        c for c in _RHO_SCALED_HYDRO_COLUMNS if any(c in d for _, d in stage_overrides)
    ]

    hydro_id_col: list[int] = []
    stage_id_col: list[int] = []
    value_cols: dict[str, list[float | None]] = {c: [] for c in present_cols}
    for hid in sorted(int(h) for h in hydro_ids):
        for stage_id, diff in stage_overrides:
            hydro_id_col.append(hid)
            stage_id_col.append(stage_id)
            for c in present_cols:
                value_cols[c].append(diff.get(c))

    schema = pa.schema(
        [pa.field("hydro_id", pa.int32()), pa.field("stage_id", pa.int32())]
        + [pa.field(c, pa.float64()) for c in present_cols]
    )
    data: dict[str, pa.Array] = {
        "hydro_id": pa.array(hydro_id_col, type=pa.int32()),
        "stage_id": pa.array(stage_id_col, type=pa.int32()),
    }
    for c in present_cols:
        data[c] = pa.array(value_cols[c], type=pa.float64())
    return pa.table(data, schema=schema)


def convert_line_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> pa.Table:
    """Convert NEWAVE interchange limits to a Cobre ``line_bounds.parquet`` table.

    Reads ``sistema.dat::limites_intercambio`` and ``dger.dat`` to produce one
    row per (line, stage) pair with direct and reverse MW bounds.

    The canonical pair logic (``src < tgt``) and line ID assignment exactly
    mirror ``convert_lines`` so that line IDs are consistent.  Interchange
    limits have no seasonalize flag, so post-study stages freeze at the last
    study stage's bounds.  Per-block exchange bound factors from
    ``patamar.dat::intercambio_patamares`` are emitted separately via
    ``convert_exchange_factors``.

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
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

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
            if sentido == 0:
                date_lookup[key]["direct_mw"] = valor
            else:
                date_lookup[key]["reverse_mw"] = valor
        else:
            # de -> para is the "reverse" direction.
            if sentido == 0:
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

    # Last study stage's (year, calendar month). Interchange limits have no
    # seasonalize flag, so the post-study tail freezes at this stage's value
    # rather than repeating the last study year's seasonal pattern.
    ls_y = start_year + (start_month - 1 + study_months - 1) // 12
    ls_m = ((start_month - 1 + study_months - 1) % 12) + 1

    for pair, line_id in sorted(pair_to_line_id.items(), key=lambda x: x[1]):
        src, tgt = pair
        freeze_caps = date_lookup.get((src, tgt, ls_y, ls_m)) or last_year_lookup.get(
            (src, tgt, ls_m), {"direct_mw": 0.0, "reverse_mw": 0.0}
        )
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            if is_post_study:
                caps = freeze_caps
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
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    total_stages = horizon.total_stages

    def _in_horizon(dt: object) -> bool:
        try:
            yr = int(dt.year)  # type: ignore[union-attr]
            mo = int(dt.month)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return False
        if yr == POST_STUDY_YEAR:
            return True
        stage_id = (yr - start_year) * 12 + (mo - start_month)
        return 0 <= stage_id < total_stages

    df_filtered = df_ncs[df_ncs["data"].apply(_in_horizon)].copy()
    groups = df_filtered.groupby(["codigo_submercado", "indice_bloco"], sort=True)

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
        return {"$schema": _NCS_SCHEMA_URL, "non_controllable_sources": []}

    dger = Dger.read(nw_files.dger)
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    total_stages = horizon.total_stages

    # Filter to study + post-study horizon only.
    # Rows with year == 9999 are post-study entries in inewave convention.
    def _in_horizon(dt: object) -> bool:
        try:
            yr = int(dt.year)  # type: ignore[union-attr]
            mo = int(dt.month)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return False
        if yr == POST_STUDY_YEAR:
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
                # NEWAVE pre-nets `geracao_usinas_nao_simuladas` from MERC
                # before the dispatch LP runs, so the aggregate is implicitly
                # must-run. Setting allow_curtailment=False instructs Cobre's
                # LP to pin dispatch to the realized availability for every
                # scenario; otherwise the LP discovers that curtailing NCS is
                # one of the cheapest slacks and produces a +15 % hydro /
                # -23 % spillage divergence vs NEWAVE on this case family.
                # See docs/findings/ncs-must-run-treatment.md.
                "allow_curtailment": False,
            }
        )
        ncs_id += 1

    return {"$schema": _NCS_SCHEMA_URL, "non_controllable_sources": ncs_list}


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
        return {"$schema": _EXCHANGE_FACTORS_SCHEMA_URL, "exchange_factors": []}

    dger = Dger.read(nw_files.dger)
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    pair_to_line_id = _build_canonical_pair_to_line_id(nw_files)
    if not pair_to_line_id:
        return {"$schema": _EXCHANGE_FACTORS_SCHEMA_URL, "exchange_factors": []}

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

    return {
        "$schema": _EXCHANGE_FACTORS_SCHEMA_URL,
        "exchange_factors": results,
    }


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
        return {
            "$schema": _NCS_FACTORS_SCHEMA_URL,
            "non_controllable_factors": [],
        }

    dger = Dger.read(nw_files.dger)
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Filter to study + post-study horizon (year == 9999 is post-study).
    def _in_horizon(dt: object) -> bool:
        try:
            yr = int(dt.year)  # type: ignore[union-attr]
            mo = int(dt.month)  # type: ignore[union-attr]
        except (AttributeError, TypeError, ValueError):
            return False
        if yr == POST_STUDY_YEAR:
            return True
        stage_id = (yr - start_year) * 12 + (mo - start_month)
        return 0 <= stage_id < total_stages

    df = df[df["data"].apply(_in_horizon)].copy()

    if df.empty:
        return {
            "$schema": _NCS_FACTORS_SCHEMA_URL,
            "non_controllable_factors": [],
        }

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
                    {
                        "block_id": block_id,
                        "factor": max(factor, 1e-6),
                    }
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

    return {
        "$schema": _NCS_FACTORS_SCHEMA_URL,
        "non_controllable_factors": results,
    }


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
    horizon = study_horizon(dger)
    start_month = horizon.start_month
    start_year = horizon.start_year
    study_months = horizon.study_months
    total_stages = horizon.total_stages

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
        if is_na(val_raw):
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

    for (sub_code, bloco), ncs_id in sorted(ncs_group_map.items(), key=lambda x: x[1]):
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
