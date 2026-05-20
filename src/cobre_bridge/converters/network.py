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
# Source: NEWAVE User Manual v29 (CEPEL/ONS, 2023), section 3.24 "Penalidades
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
# Merit order from NEWAVE micro-penalty values (current v29 defaults):
#   p_INT (0.000273) < p_PFIO (0.000300) < p_EVERT (0.000327)
#   < p_TURB (0.000333) < p_CORTEOL (0.000344) < p_EXC (0.000355)
# (manual page 88).

MONTH_HOURS: float = (
    730.0  # NEWAVE convention (manual §3.24, used in C_M3S2HM3)
)
C_M3S2HM3: float = MONTH_HOURS * 3600.0 / 1e6  # = 2.628 hm³ / (m³/s · month)
# HM3 × ρ_MW_per_m3s → MWh conversion (purely volumetric; 730 cancels here).
HM3_TO_MWH_PER_RHO: float = 1e6 / 3600.0  # ≈ 277.78

# --- NEWAVE micro-penalties (page 88, current v29) -------------------------
# Energy-domain (R$/MWh) — passed through to cobre without conversion.
# Flow-domain (multiplied by ρ_avg before emission) — see `_PEVERT` group.
#
# Uniform 100× uplift over NEWAVE's v29 values. NEWAVE chose values around
# 1e-4 R$/MWh for "absolute negligibility" in its own SPTcpp solver; with
# HiGHS and cobre's typical case scale, that floor stretches the LP
# coefficient range to ~1e9 against the operational/deterrent slacks at
# ~5e4. Multiplying every micro-penalty by the same factor preserves the
# internal ordering (exchange < spillage < ... < excess) and the merit
# tier (regularization ≪ violation cost) while compressing the LP
# coefficient range to ~1e7 — comfortably away from HiGHS's 1e10
# warning. Economic impact: at 100× uplift, 1 m³/s of spillage for an
# entire 730 h month adds ~15.5 R$ to the objective — about 1e-5 of a
# typical-stage total cost (~1e10 R$). Within NEWAVE-historical bounds
# (v29.2.1 values were already ~18× the current).
_MICRO_UPLIFT = 100.0

_PINT = 0.000273 * _MICRO_UPLIFT  # intercâmbio  → line.exchange_cost

# NEWAVE halves the intercâmbio penalty on lines that touch a fictitious
# submercado (e.g. NOFICT1). Rationale: a fictitious node is a routing-only
# hop with no demand or generation of its own, so a real → fict → real path
# would otherwise accumulate twice the penalty of an equivalent direct
# real → real link. The 0.5 discount restores cost-parity between the two
# topologies.  Emitted as the per-line `exchange_cost` override defined in
# lines.schema.json; absence falls back to the global `_PINT` value.
_PINT_FICTITIOUS_DISCOUNT = 0.5
_PCORTEOL = (
    0.000344 * _MICRO_UPLIFT
)  # corte geração eólica → ncs.curtailment_cost
_PEXC = 0.000355 * _MICRO_UPLIFT  # excesso de energia → bus.excess_cost

# Flow-domain (R$/MWh equivalent, multiplied by ρ_avg before emission).
# Cobre's `hydro.spillage_cost` covers ALL spillage (reservoir + run-of-river);
# we anchor on pEVERT (controllable reservoir spillage) since that's the
# dominant case in any NEWAVE-derived hydro fleet.
_PEVERT = (
    0.000327 * _MICRO_UPLIFT
)  # vertimento controlável → hydro.spillage_cost
_PTURB = (
    0.000333 * _MICRO_UPLIFT
)  # turbinamento → hydro.turbined_cost (applied to every hydro)
_PCDESV = 0.000300 * _MICRO_UPLIFT  # volume desviado → hydro.diversion_cost

# --- NEWAVE hard-coded internal defaults (no user input via PENALID) -------
# Page 87: evaporation and FPHA folga both derive from MAX_CUSTO_DEFICIT.
# NEWAVE's manual prescribes a 10× multiplier here, making the evap-folga
# slack roughly 10× more expensive per MWh-equivalent than the deficit
# cost. Faithfully applying that 10× in cobre's per-(m³/s · h) LP slot
# produces coefficients ~5e5 against a micro-penalty floor at ~2e-4 — the
# resulting ~1e10 coefficient range trips HiGHS's condition-number
# threshold and roughly doubles training time on the example case.
#
# We use 1.1× instead. Rationale: evaporation and water withdrawal are
# both *physical-law* constraints (water cycle physics; human water-supply
# requirements), so they belong at the top of the merit order — but their
# job is to be "violated only as a last resort", not "100× more painful
# than every other cost". A 10 % margin over water_withdrawal_violation_cost
# is enough that an LP would only pick the evap slack when no alternative
# exists, while keeping the deterrent tier numerically close to the
# operational tier and the LP well-conditioned.
_EVAPORATION_MULT = 1.1

# --- Tie-breaking factors for PENALID flow slacks that share ρ_avg ---------
# When PENALID supplies TURBMN, VAZMIN, TURBMX with the same R$/MWh value
# (typical NEWAVE convention), the three slacks would otherwise have
# identical LP coefficients — a degeneracy that hurts HiGHS performance.
# Apply small adjacent multipliers to give each a unique value while
# preserving a physically-defensible ordering:
#
#   turbined_violation_below (internal plant operation, can spill instead)
#     < outflow_violation_below (downstream environmental / navigation)
#     < outflow_violation_above (flood risk, public-safety regulation).
#
# 1 % spacing is small enough that LP decisions are unchanged in practice
# but large enough that HiGHS sees them as numerically distinct.
_TURBINED_BELOW_FACTOR = 0.99
_OUTFLOW_BELOW_FACTOR = 1.00
_OUTFLOW_ABOVE_FACTOR = 1.01

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

# --- Margin above max flow-domain slack for inflow non-negativity ----------
# inflow_nonnegativity_cost is set 1 % above the strictest other flow-domain
# slack so the LP never picks it over a "fixable" constraint while still
# allowing it to fire when no alternative exists. The margin is robust to
# floating-point noise yet small enough not to inflate the objective.
_INFLOW_NN_MARGIN = 1.01

_EXCESS_MULT = 10.0


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
        buses_by_code[code]["segments"].append({
            "patamar": int(row["patamar_deficit"]),
            "depth_mw": depth_mw,
            "cost": cost,
        })

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
        active_segs = [
            s for s in segs if s["cost"] is not None and s["cost"] > 0
        ]
        if not active_segs:
            active_segs = [{"cost": fallback_cost, "depth_mw": None}]

        deficit_segments: list[dict] = []
        for i, seg in enumerate(active_segs):
            is_last = i == len(active_segs) - 1
            deficit_segments.append({
                "depth_mw": None if is_last else seg["depth_mw"],
                "cost": seg["cost"],
            })

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
            int(code)
            for code in deficit_df.loc[fic_mask, "codigo_submercado"].unique()
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
    for (src, tgt), line_id in sorted(
        canonical_map.items(), key=lambda x: x[1]
    ):
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
        _LOG.warning(
            "penalid.dat could not be parsed (%s); using defaults.", exc
        )
        return {}

    pen_df = penalid.penalidades
    if pen_df is None or pen_df.empty:
        return {}

    out: dict[str, float] = {}
    for var in pen_df["variavel"].unique():
        rows = pen_df[
            (pen_df["variavel"] == var) & pen_df["valor_R$_MWh"].notna()
        ]
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


def convert_penalties(
    nw_files: NewaveFiles,
    hydros_dict: dict,
    productivities: dict[int, float] | None = None,
    *,
    max_accumulated_productivity: float | None = None,
) -> dict:
    """Generate a Cobre ``penalties.json`` dict from NEWAVE data.

    Faithful to NEWAVE User Manual v29 section 3.24:

    - Bus deficit segments come from ``sistema.custo_deficit`` directly
      (R$/MWh on both sides, no conversion).
    - PENALID-sourced flow-domain penalties are converted to cobre's
      coefficient slot via ``× ρ`` (where ρ is ``PROD_MEDIA_SIN`` or
      ``MAX_PRODTACUM_SIN`` per the NEWAVE conversion table on page 87).
    - The micro-penalties (``pINT``, ``pEVERT``, ``pTURB``, ``pCORTEOL``,
      ``pEXC``, ``pCDESV``) are NEWAVE's hard-coded internal defaults
      (page 88, current v29 values). They are written directly to cobre
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
        first_sub = deficit_df.sort_values([
            "codigo_submercado",
            "patamar_deficit",
        ])
        primary_deficit_cost = float(first_sub.iloc[0]["custo"])
        max_deficit_cost = float(deficit_df["custo"].max())

    penalid_costs = _read_penalid_costs(nw_files)
    productivities = productivities or {}

    # ρ_avg = PROD_MEDIA_SIN: arithmetic mean of own productivities (NEWAVE
    # convention for VAZMIN, TURBMN, TURBMX, VOLMIN — manual table p.87).
    own_prods = _own_productivities(hydros_dict, productivities)
    rho_avg = sum(own_prods) / len(own_prods) if own_prods else 1.0

    # ρ_max_acum = MAX_PRODTACUM_SIN: used by NEWAVE for DESVIO and the
    # evaporation default. When the caller doesn't supply the true cascade
    # accumulated max we approximate by `max(own_prods)`; the caller in
    # `pipeline.py` passes the real value computed from the cascade DAG.
    rho_max_acum = (
        max_accumulated_productivity
        if max_accumulated_productivity is not None
        else (max(own_prods) if own_prods else rho_avg)
    )

    # --------------------------------------------------------------------
    # PENALID-sourced violation costs.
    # --------------------------------------------------------------------
    # Convention from NEWAVE manual p.87:
    #   DESVIO    : × MAX_PRODTACUM_SIN
    #   VAZMIN    : × PROD_MEDIA_SIN  → outflow_violation_below_cost
    #   TURBMN    : × PROD_MEDIA_SIN
    #   TURBMX    : × PROD_MEDIA_SIN  (used for outflow_violation_above_cost)
    #   GHMIN     : direct (energy-domain slack, no productivity multiplier)
    #   VOLMIN    : × PROD_MEDIA_SIN × C_M3S2HM3 → storage_violation_below_cost
    desvio_mwh = penalid_costs.get(
        "DESVIO", _EVAPORATION_MULT * max_deficit_cost
    )
    vazmin_mwh = penalid_costs.get(
        "VAZMIN", _EVAPORATION_MULT * max_deficit_cost
    )
    ghmin_mwh = penalid_costs.get("GHMIN", _EVAPORATION_MULT * max_deficit_cost)
    turbmn_mwh = penalid_costs.get(
        "TURBMN", _EVAPORATION_MULT * max_deficit_cost
    )
    turbmx_mwh = penalid_costs.get(
        "TURBMX", _EVAPORATION_MULT * max_deficit_cost
    )

    water_withdrawal_cost = desvio_mwh * rho_max_acum
    # Apply tie-breaking factors so the three PENALID-flow slacks share an
    # ordering even when PENALID gives them identical R$/MWh values.
    outflow_below_cost = vazmin_mwh * rho_avg * _OUTFLOW_BELOW_FACTOR
    outflow_above_cost = turbmx_mwh * rho_avg * _OUTFLOW_ABOVE_FACTOR
    turbined_below_cost = turbmn_mwh * rho_avg * _TURBINED_BELOW_FACTOR
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
    # `(K × MAX_CUSTO_DEFICIT × MAX_PRODTACUM_SIN) / C_M3S2HM3` with K=10.
    # We use K=2 (see `_EVAPORATION_MULT` rationale) so the deterrent
    # magnitude stays above the PENALID-sourced operational slacks
    # (typically water_withdrawal_violation_cost) without stretching the
    # LP coefficient range past HiGHS's comfortable conditioning zone. The
    # /C_M3S2HM3 step is dropped (only needed for NEWAVE's per-hm³ slot).
    evaporation_cost = _EVAPORATION_MULT * max_deficit_cost * rho_max_acum

    # --------------------------------------------------------------------
    # NEWAVE micro-penalty defaults (page 88, current v29).
    # --------------------------------------------------------------------
    # Energy-domain (R$/MWh) passes through directly.
    excess_cost = _PEXC
    exchange_cost = _PINT
    curtailment_cost = _PCORTEOL
    # Flow-domain (multiplied by ρ_avg per NEWAVE individualized conversion).
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
    # the other constraint first. To enforce a strict preference, we set this
    # cost ~1 % above the maximum of the operationally-related flow-domain
    # costs — guarantees the LP never preferentially violates inflow non-
    # negativity unless there is no other available slack. The 1 % margin is
    # large enough to be numerically robust against floating-point noise yet
    # small enough that it doesn't dominate when this slack is genuinely the
    # only viable path.
    inflow_nonnegativity_cost = _INFLOW_NN_MARGIN * max(
        turbined_below_cost,
        outflow_below_cost,
        outflow_above_cost,
    )

    return {
        "$schema": _PENALTIES_SCHEMA_URL,
        "bus": {
            "deficit_segments": [
                {
                    "cost": primary_deficit_cost,
                    "depth_mw": None,
                }
            ],
            "excess_cost": excess_cost * _EXCESS_MULT,
        },
        "hydro": {
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

    _LINE_BOUNDS_SCHEMA = pa.schema([
        pa.field("line_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("direct_mw", pa.float64()),
        pa.field("reverse_mw", pa.float64()),
    ])

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
    last_year_per_key: dict[
        tuple[int, int, int], tuple[int, dict[str, float]]
    ] = {}
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
        return {"$schema": _NCS_SCHEMA_URL, "non_controllable_sources": []}

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

    groups = df_filtered.groupby(
        ["codigo_submercado", "indice_bloco"], sort=True
    )

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
        fonte = (
            str(fonte_series.iloc[0]).strip()
            if not fonte_series.empty
            else "NCS"
        )

        # max_generation_mw: maximum non-NaN value across all rows in the group.
        valores = pd.to_numeric(group["valor"], errors="coerce")
        valid_vals = valores.dropna()
        max_gen = float(valid_vals.max()) if not valid_vals.empty else 0.0

        ncs_list.append({
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
        })
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

                block_factors.append({
                    "block_id": block_id,
                    "direct_factor": d_factor,
                    "reverse_factor": r_factor,
                })

            results.append({
                "line_id": line_id,
                "stage_id": stage_id,
                "block_factors": block_factors,
            })

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

    for (sub_code, bloco), ncs_id in sorted(
        ncs_group_map.items(), key=lambda x: x[1]
    ):
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            block_factors: list[dict] = []
            for block_id in range(num_blocks):
                if is_post_study:
                    factor = last_factor.get(
                        (sub_code, bloco, m, block_id), 1.0
                    )
                else:
                    factor = factor_map.get(
                        (sub_code, bloco, y, m, block_id),
                        last_factor.get((sub_code, bloco, m, block_id), 1.0),
                    )
                block_factors.append({
                    "block_id": block_id,
                    "factor": max(factor, 1e-6),
                })

            results.append({
                "ncs_id": ncs_id,
                "stage_id": stage_id,
                "block_factors": block_factors,
            })

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

    _NCS_STATS_SCHEMA = pa.schema([
        pa.field("ncs_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("mean", pa.float64()),
        pa.field("std", pa.float64()),
    ])

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
