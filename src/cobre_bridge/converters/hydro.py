"""Hydro entity converter: maps the source model hydro plant data to Cobre hydro
JSON."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pyarrow as pa

from cobre_bridge.case import NewaveCase
from cobre_bridge.diagnostics import Diagnostic, DiagnosticTable, Severity, emit
from cobre_bridge.filling import (
    filling_completion_date,
    filling_min_rate_m3s,
    filling_schedule,
    online_machines,
)
from cobre_bridge.filling import stage_id as filling_stage_id
from cobre_bridge.horizon import (
    POST_STUDY_YEAR,
    build_stage_dates,
    historical_start_date,
    seasonal_step_function,
)
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.pandas_utils import is_na
from cobre_bridge.plants import fictitious_codes, filling_hydro_codes
from cobre_bridge.productivity import (
    apply_hydraulic_loss,
    compute_productivity,
    equivalent_productivity,
    evaluate_cota,
    integrated_productivity,
    mean_cota,
    stored_energy_productivity,
)

_LOG = logging.getLogger(__name__)

# The point / PRODT / integrated productivity functions now live in the
# presentation-free ``cobre_bridge.productivity`` domain module. These aliases
# keep hydro.py's many internal callers (and tests that import the private
# names) working; external modules import the public names directly.
_compute_productivity = compute_productivity
_equivalent_productivity = equivalent_productivity
_compute_integrated_productivity = integrated_productivity
_stored_energy_productivity = stored_energy_productivity

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/hydros.schema.json"
)
_PRODUCTION_MODELS_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/production_models.schema.json"
)

# --- FPHA (hydro production function) emission ---------------------------------
#
# When the source model evaluates generation via FPHA (``dger.dat`` line 96,
# ``funcao_producao_uhe == 0``; see :attr:`NewaveCase.fpha_enabled`), reservoir plants
# are emitted with cobre's ``model: "fpha"`` so cobre fits the production function from
# the plant geometry + tailrace families instead of the bridge pre-baking a single
# constant productivity.

# cobre's FPHA production function is ``phi = K · eta · q · h_net`` (MW), with ``K = g /
# 1000`` and ``eta`` the dimensionless turbine efficiency in (0, 1]. The source model
# instead carries the *specific* productivity ``rho_esp`` (MW/((m³/s)·m)), which already
# folds in ``K · eta``. So the efficiency cobre needs is ``eta = rho_esp / K`` — the
# value that makes cobre's ``phi`` reproduce the source model's ``rho_esp · q · h_liq``.
_GRAVITY_MW_FACTOR = 9.81e-3
# The source model's tratamento-fpha distance method carries only a tolerance; cobre
# also requires a sample count for the mean-squared-distance estimate. The source model
# does not specify one, so we supply a reasonable default.
_FPHA_DISTANCE_N_SAMPLES = 100


def _is_fpha_eligible(hreg: pd.Series) -> bool:
    """Whether a hydro plant can be fit by cobre's *computed* FPHA.

    Requires a non-degenerate volume→cota polynomial (the forebay curve) and a
    positive specific productivity ``rho_esp`` (needed to derive the
    dimensionless turbine efficiency). Storage swing is **not** required:
    run-of-river / zero-storage plants (``vmax == vmin``) emit a single VHA geometry row
    and cobre fits them through the single-volume FPHA path (γ_V = 0), matching the
    source model, which fits these plants with ``Npt_V = 1``.
    """
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
    if all(c == 0.0 for c in coeffs):
        return False
    rho_esp_raw = hreg.get("produtibilidade_especifica")
    if rho_esp_raw is None:
        return False
    rho_esp = float(rho_esp_raw)
    return not math.isnan(rho_esp) and rho_esp > 0.0


def fpha_eligible_codes(case: NewaveCase) -> set[int]:
    """The source model plant codes emitted as ``model: "fpha"`` for this case.

    Empty unless :attr:`NewaveCase.fpha_enabled`. The single source of truth for
    FPHA eligibility, shared by :func:`convert_hydros`,
    :func:`convert_production_models`, and :func:`convert_hydro_energy_productivity`
    so the three files agree on which plants are FPHA. Uses the same
    permanent-override cadastro the converters use, so eligibility is consistent.
    """
    if not case.fpha_enabled:
        return set()
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    eligible: set[int] = set()
    for _, row in case.active_hydros.iterrows():
        code = int(row["codigo_usina"])
        if code in cadastro.index and _is_fpha_eligible(cadastro.loc[code]):
            eligible.add(code)
    return eligible


def _fpha_efficiency(rho_esp: float, name: str) -> float:
    """Dimensionless turbine efficiency ``eta = rho_esp / K`` for cobre's FPHA.

    Clamped to ``1.0`` if the source ``rho_esp`` implies ``eta > 1`` (unphysical
    in the input data), with a warning.
    """
    eta = rho_esp / _GRAVITY_MW_FACTOR
    if eta > 1.0:
        _LOG.warning(
            "FPHA turbine efficiency for plant %s implied by rho_esp is %.4f "
            "(> 1.0); clamping to 1.0.",
            name,
            eta,
        )
        return 1.0
    return eta


def _parse_fpha_plane_reduction(case: NewaveCase) -> dict | None:
    """Parse ``tratamento-fpha`` into a cobre ``fpha_plane_reduction`` block.

    The source model's treatment file carries one active line (``&``-prefixed lines are
    comments) selecting either the angle or the distance plane-reduction method::

        HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-ANGULO-PADRAO; 1.0
        HIDRELETRICA-FPHA-METODO-REDUCAO-CORTES-DISTANCIA-PADRAO; 0.002

    Returns ``{"method": "angle", "tolerance_deg": v}`` or
    ``{"method": "distance", "tolerance_pct": v, "n_samples": N}``, or ``None``
    when no treatment file is present or no method line is active.
    """
    path = case.files.tratamento_fpha
    if path is None:
        return None

    methods: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("&"):
            continue
        parts = [part.strip() for part in line.split(";")]
        if len(parts) < 2:
            continue
        token = parts[0].upper()
        try:
            value = float(parts[1])
        except ValueError:
            continue
        if token.endswith("ANGULO-PADRAO"):
            methods.append({"method": "angle", "tolerance_deg": value})
        elif token.endswith("DISTANCIA-PADRAO"):
            methods.append(
                {
                    "method": "distance",
                    "tolerance_pct": value,
                    "n_samples": _FPHA_DISTANCE_N_SAMPLES,
                }
            )

    if not methods:
        return None
    if len(methods) > 1:
        _LOG.warning(
            "Multiple active FPHA plane-reduction methods in %s; using the first (%s).",
            path.name,
            methods[0]["method"],
        )
    return methods[0]


_EVAP_MONTHS = [
    "JAN",
    "FEV",
    "MAR",
    "ABR",
    "MAI",
    "JUN",
    "JUL",
    "AGO",
    "SET",
    "OUT",
    "NOV",
    "DEZ",
]

# Temporal override types extracted from MODIF.DAT.
_TEMPORAL_OVERRIDE_TYPES = frozenset(
    {"VAZMINT", "VMAXT", "VMINT", "CFUGA", "CMONT", "TURBMINT", "TURBMAXT"}
)


def _apply_permanent_overrides(
    cadastro: pd.DataFrame, case: NewaveCase
) -> pd.DataFrame:
    """Apply MODIF.DAT permanent overrides to the hidr.dat cadastro.

    Reads ``MODIF.DAT`` from *case* and
    applies permanent override records — VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
    NUMMAQ — to a *copy* of *cadastro*.  The original DataFrame is not
    mutated.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina``.
    case:
        Parsed the source model case.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with permanent overrides applied.
    """
    modif = case.modif
    if modif is None:
        _LOG.debug("MODIF.DAT not found; skipping permanent overrides.")
        return cadastro

    result = cadastro.copy()

    # Ensure float dtype for columns that permanent overrides may assign floats
    # into.  Without this, pandas 2.x raises TypeError when the column was
    # inferred as int64 (e.g. vazao_minima_historica=[0, 0]).
    _float_override_cols = (
        "vazao_minima_historica",
        "volume_maximo",
        "volume_minimo",
    )
    for _col in _float_override_cols:
        if _col in result.columns and result[_col].dtype.kind == "i":
            result[_col] = result[_col].astype(float)

    usina_records = modif.usina()
    if not usina_records:
        return result

    for usina_rec in usina_records:
        code = int(usina_rec.codigo)
        if code not in result.index:
            _LOG.warning(
                "MODIF.DAT references plant code %d which is not in hidr.dat;"
                " skipping.",
                code,
            )
            continue

        for rec in modif.modificacoes_usina(code):
            type_name = type(rec).__name__

            # Skip temporal override types — handled separately.
            if type_name in _TEMPORAL_OVERRIDE_TYPES:
                continue

            if type_name == "VAZMIN":
                result.loc[code, "vazao_minima_historica"] = float(rec.vazao)

            elif type_name == "VOLMAX":
                result.loc[code, "volume_maximo"] = float(rec.volume)

            elif type_name == "VOLMIN":
                result.loc[code, "volume_minimo"] = float(rec.volume)

            elif type_name == "NUMCNJ":
                result.loc[code, "numero_conjuntos_maquinas"] = int(rec.numero)

            elif type_name == "NUMMAQ":
                set_num = int(rec.conjunto)
                n_maq = int(rec.numero_maquinas)
                result.loc[code, f"maquinas_conjunto_{set_num}"] = n_maq

            elif type_name in ("VOLCOTA", "COTARE"):
                # VOLCOTA/COTARE are not present in the example case.
                # The spec mentions them but the inewave API does not expose
                # them as separate methods in the tested version.  Log a
                # warning if they appear so the operator knows to investigate.
                _LOG.warning(
                    "MODIF.DAT contains unsupported permanent override type"
                    " '%s' for plant %d; skipping.",
                    type_name,
                    code,
                )

            elif type_name == "DefaultRegister":
                # inewave emits DefaultRegister for records it does not model
                # (e.g. COTAREA). These are benign for the conversion, so log at
                # debug level only — no user-facing warning.
                _LOG.debug(
                    "MODIF.DAT contains an unmodeled record (DefaultRegister)"
                    " for plant %d; skipping.",
                    code,
                )

            else:
                _LOG.warning(
                    "MODIF.DAT contains unknown permanent override type '%s'"
                    " for plant %d; skipping.",
                    type_name,
                    code,
                )

    return result


def _extract_temporal_overrides(
    case: NewaveCase, confhd_codes: list[int]
) -> dict[int, list[dict]]:
    """Extract MODIF.DAT temporal overrides for plants in *confhd_codes*.

    Reads ``MODIF.DAT`` and returns a dict keyed by plant code.  Each value
    is a list of override dicts in file order::

        {"type": str, "month": int, "year": int, "value": float}

    For CFUGA/CMONT the ``"value"`` field is the level in metres.  For
    TURBMINT/TURBMAXT it is the turbined flow in m³/s.  For VAZMINT/VMAXT/
    VMINT it is the volume or flow as stored in the record.

    Parameters
    ----------
    case:
        Parsed the source model case.
    confhd_codes:
        List of plant codes present in the study (from confhd.dat).  Records
        for plants not in this list are excluded.

    Returns
    -------
    dict[int, list[dict]]
        Temporal override records per plant code.  Empty dict if MODIF.DAT is
        absent.
    """
    modif = case.modif
    if modif is None:
        _LOG.debug("MODIF.DAT not found; no temporal overrides extracted.")
        return {}

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
            if type_name not in _TEMPORAL_OVERRIDE_TYPES:
                continue

            data = rec.data_inicio
            month = int(data.month)
            year = int(data.year)

            if type_name in ("VAZMINT",):
                value = float(rec.vazao)
            elif type_name in ("VMAXT", "VMINT"):
                value = float(rec.volume)
            elif type_name in ("CFUGA", "CMONT"):
                value = float(rec.nivel)
            elif type_name in ("TURBMINT", "TURBMAXT"):
                value = float(rec.turbinamento)
            else:
                _LOG.warning(
                    "Unknown temporal override type '%s' for plant %d; skipping.",
                    type_name,
                    code,
                )
                continue

            plant_overrides.append(
                {"type": type_name, "month": month, "year": year, "value": value}
            )

        if plant_overrides:
            result[code] = plant_overrides

    return result


def _read_ghmin_per_stage(
    case: NewaveCase,
    start_year: int,
    start_month: int,
    study_months: int,
    total_stages: int,
) -> dict[int, dict[int, float]]:
    """Read GHMIN.DAT and expand into ``{plant_code: {stage_0based: min_gen_mw}}``.

    GHMIN values are time-varying minimum-generation requirements in MWmes that
    source-model enforces per plant per stage.  Each (plant, month, year) record sets
    the value from that stage forward until the next record overrides it (step
    function).  Records with ``year == 9999`` are post-study seasonal entries: each
    calendar month they appear for becomes the value used in every post-study stage with
    that calendar month, falling back to a seasonal repeat of the last study year for
    unspecified months.

    Only ``patamar == 0`` rows are used — they represent the all-blocks
    mean, which matches the per-stage granularity of
    ``hydro_bounds.parquet``.

    Returns an empty mapping when ``GHMIN.DAT`` is absent.

    Parameters
    ----------
    case:
        Parsed the source model case.
    start_year, start_month:
        Study start (Cobre stage 0 corresponds to this calendar month).
    study_months:
        Number of in-study stages.
    total_stages:
        Total number of stages (study + post-study).
    """
    ghmin = case.ghmin
    if ghmin is None:
        _LOG.debug("GHMIN.DAT not found; emitting no per-stage min_generation.")
        return {}

    df = ghmin.geracoes
    if df is None or df.empty:
        return {}

    patamar0 = df[df["patamar"] == 0]
    if patamar0.empty:
        return {}

    result: dict[int, dict[int, float]] = {}
    for code, group in patamar0.groupby("codigo_usina"):
        code_int = int(code)
        study_changes: list[tuple[int, float]] = []
        pos_by_month: dict[int, float] = {}
        for _, row in group.iterrows():
            dt = row["data"]
            yr = int(dt.year)
            mo = int(dt.month)
            value = float(row["geracao"])
            if yr == POST_STUDY_YEAR:
                pos_by_month[mo] = value
                continue
            sid = (yr - start_year) * 12 + (mo - start_month)
            if sid < 0:
                sid = 0
            study_changes.append((sid, value))
        study_changes.sort()

        per_stage: dict[int, float] = {}
        # Step function across the study period.
        if study_changes:
            first_stage = study_changes[0][0]
            cp_idx = 0
            current: float | None = None
            for stage_id in range(first_stage, study_months):
                while (
                    cp_idx < len(study_changes) and study_changes[cp_idx][0] <= stage_id
                ):
                    current = study_changes[cp_idx][1]
                    cp_idx += 1
                if current is not None:
                    per_stage[stage_id] = current

        # Seasonal pattern for post-study: prefer explicit POS entries
        # for each calendar month; fall back to the last study year's
        # value for that calendar month.
        if total_stages > study_months:
            last_year_seasonal: dict[int, float] = {}
            for stage_id in range(max(0, study_months - 12), study_months):
                if stage_id in per_stage:
                    cal = ((start_month - 1 + stage_id) % 12) + 1
                    last_year_seasonal[cal] = per_stage[stage_id]

            for stage_id in range(study_months, total_stages):
                cal = ((start_month - 1 + stage_id) % 12) + 1
                if cal in pos_by_month:
                    per_stage[stage_id] = pos_by_month[cal]
                elif cal in last_year_seasonal:
                    per_stage[stage_id] = last_year_seasonal[cal]

        if per_stage:
            result[code_int] = per_stage

    return result


# Mapping from PENALID.DAT variable names to Cobre penalty field names.
_PENALID_VAR_MAP: dict[str, str] = {
    "DESVIO": "water_withdrawal_violation_cost",
    "VAZMIN": "outflow_violation_below_cost",
    "VAZMAX": "outflow_violation_above_cost",
    "GHMIN": "generation_violation_below_cost",
    "TURBMN": "turbined_violation_below_cost",
    "TURBMX": "outflow_violation_above_cost",
}


def _read_penalid(case: NewaveCase) -> dict[int, dict[str, float]]:
    """Read PENALID.DAT and return per-REE penalty override mappings.

    If ``PENALID.DAT`` is absent (``case.penalid is None``), returns an
    empty dict.  Only the first patamar tier (``patamar_penalidade == 1``)
    is used — tier 2 has NaN costs (unbounded) and is skipped.  NaN values
    within tier 1 are also skipped.

    Parameters
    ----------
    case:
        Parsed the source model case.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from REE/subsystem code to a dict of Cobre penalty field
        names -> cost in R$/MWh.  Only fields with valid (non-NaN) values
        are included.  Returns an empty dict if the file is absent or
        contains no usable rows.
    """
    penalid = case.penalid
    if penalid is None:
        _LOG.debug("PENALID.DAT not found; leaving all plant penalties as None.")
        return {}

    df: pd.DataFrame | None = penalid.penalidades
    if df is None or df.empty:
        return {}

    # Keep only first-tier rows (patamar_penalidade == 1).
    tier1 = df[df["patamar_penalidade"] == 1]
    if tier1.empty:
        return {}

    result: dict[int, dict[str, float]] = {}
    for _, row in tier1.iterrows():
        variavel = str(row["variavel"]).strip()
        cobre_field = _PENALID_VAR_MAP.get(variavel)
        if cobre_field is None:
            # Variable not mapped (e.g. TURBMX, ELETRI) — skip silently.
            continue

        ree_code = int(row["codigo_ree_submercado"])
        valor = row["valor_R$_MWh"]

        if pd.isna(valor):
            continue

        cost = float(valor)
        result.setdefault(ree_code, {})[cobre_field] = cost

    return result


# Turbine type code -> kturb exponent used in the head-correction formula. The source
# model codes: 1 = Francis, 2 = Kaplan, 3 = Pelton.  Francis and Pelton share the kturb
# = 0.5 exponent (square-root flow/head response); Kaplan uses 0.2 (gentler response
# thanks to adjustable blades).  Code 0 (= not specified in hidr.dat) falls back to
# Francis.
_KTURB_BY_TIPO_TURBINA: dict[int, float] = {0: 0.5, 1: 0.5, 2: 0.2, 3: 0.5}


def _clamp_outage_pct(value: float, label: str, plant_name: str) -> float:
    """Clamp TEIF/IP percentages into ``[0, 100]`` and warn on overshoot."""
    if math.isnan(value) or value < 0.0:
        return 0.0
    if value > 100.0:
        _LOG.warning(
            "%s exceeds 100%% for plant %s (%s=%.2f); clamping to 100.",
            label,
            plant_name,
            label.lower(),
            value,
        )
        return 100.0
    return value


def _availability(teif: float, ip: float) -> float:
    """Joint TEIF/IP availability factor — both apply multiplicatively."""
    return ((100.0 - teif) / 100.0) * ((100.0 - ip) / 100.0)


def _compute_max_turbined_simple(hreg: pd.Series, name: str) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` using the simple sum-of-rated
    approach: ``Σ_c (n_c · q_nom_c)`` for flow and ``Σ_c (n_c · p_nom_c)`` for
    power, both derated by the combined TEIF/IP availability factor.

    This is **live code on two paths**: it is the fallback
    :func:`_compute_max_turbined_head_corrected` calls whenever a plant lacks the
    head data (``queda_nominal_conjunto_*`` / cota polynomial) the corrected
    formula needs, and it is the v0.6.x formulation the top-level caller can be
    switched back to if the head-corrected formula proves wrong.
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
        p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
        max_turbined += q_nominal * n_machines
        max_generation += p_nominal * n_machines

    teif = _clamp_outage_pct(float(hreg.get("teif", 0.0) or 0.0), "teif", name)
    ip = _clamp_outage_pct(float(hreg.get("ip", 0.0) or 0.0), "ip", name)
    availability = _availability(teif, ip)
    return max_turbined * availability, max_generation * availability


def _compute_max_turbined_rated(hreg: pd.Series) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` as the rated nameplate capacity:
    ``Σ_c (n_c · q_nom_c)`` for flow and ``Σ_c (n_c · p_nom_c)`` for power, with
    **no** TEIF/IP availability derating and no head correction.

    ``convert_hydros`` emits the power value ``[1]`` as every plant's ``max_generation``
    (independent of the production function): it equals the source model's
    installed-capacity ceiling / FPHA ``GHmax`` exactly (verified TUCURUI 7445, QUEBRA
    QUEIX 120). The flow value ``[0]`` (``Σ n·q_nom``) is the source model's
    fitting-grid ``Qmax``, **not** the operational turbined cap — the emitted
    ``max_turbined`` comes from :func:`_compute_max_turbined_head_corrected` instead
    (the head-corrected engolimento that actually binds in dispatch).
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        max_turbined += float(hreg[f"vazao_nominal_conjunto_{i}"]) * n_machines
        max_generation += float(hreg[f"potencia_nominal_conjunto_{i}"]) * n_machines
    return max_turbined, max_generation


def _compute_max_turbined_head_corrected(
    hreg: pd.Series, name: str, *, h_op_override: float | None = None
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` using the head-corrected the
    source-model-style cap.

    ``h_op_override`` supplies the per-stage operating head directly (in metres),
    bypassing the static base-data head derivation below. It is the head that
    produces the stage's equivalent productivity (``ρ_eq / ρ_esp``), so the
    engolimento and the ``p_inst/prodt_eq`` cap track the per-stage CFUGA/CMONT
    overrides in lockstep with :func:`convert_hydro_energy_productivity`. Used by
    :func:`convert_turbined_bounds_head_corrected` to emit a per-stage
    ``max_turbined`` instead of a single static cap.

    This is the source model's **operational turbined cap** — the maximum turbinable
    flow (engolimento) at the plant's operating head, which is what binds in the
    dispatch LP. Verified against the source model on QUEBRA QUEIX, whose cap binds:
    head-corrected 113.006 m³/s == the source model's realized 113.01 (the rated ``Σ
    n·q_nom`` = 117.0 overshoots it by 3.5%). ``convert_hydros`` emits the flow value
    ``[0]`` as ``max_turbined``; the ``[1]`` it returns is the availability-derated
    power and is no longer used for the emitted ``max_generation`` (that comes from
    :func:`_compute_max_turbined_rated`).

    For each machine set *c* with nominal head ``h_nom_c``, nominal flow
    ``q_nom_c`` and number of units ``n_c``, the effective rated flow at
    operating head ``h_op`` follows the affinity-law approximation::

        q_eff_c = (h_op / h_nom_c)^k_turb · q_nom_c

    where ``k_turb`` depends on the turbine family (0.5 for Francis/Pelton,
    0.2 for Kaplan).  The plant cap is then

        qtur_max = min(Σ_c n_c · q_eff_c , p_inst / prodt_eq) · availability

    with ``h_op`` being:

    * the volume-integrated net head from ``V_min`` to ``V_65``, i.e.
      ``mean_cota(V_min, V_65) - cota_jus - perdas``, for reservoir plants
      (``tipo_regulacao == 'M'``).  This is the source model ``h^{65%}`` symbol —
      despite the name, it is a *mean over the operating range* rather than the snapshot
      at V = V_65.  Reproduces the source model's actual cap exactly on the diagnostic
      case (M. DE MORAES) and matches the equivalent productivity ``prodt^M``
      denominator;
    * the machine-count-weighted average of nominal heads, for run-of-river /
      daily plants — they don't have a meaningful "operating volume".

    ``prodt_eq = ρ_esp · h_int`` integrates the head polynomial over
    ``[V_min, V_65]`` for M plants (so it equals ``ρ_esp · h_op``) and
    ``[V_min, V_max]`` for the others.

    Falls back to the simple Σ(n·q) when the inputs needed for the head
    correction are missing (``queda_nominal_conjunto_*`` columns absent,
    no polynomial coefficients, zero ρ_esp, etc.) — this keeps unit-test
    fixtures with partial schemas working.
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    available_cols = set(hreg.index)
    has_head_data = all(
        f"queda_nominal_conjunto_{i}" in available_cols for i in range(1, 6)
    ) and all(f"a{i}_volume_cota" in available_cols for i in range(5))
    if not has_head_data:
        return _compute_max_turbined_simple(hreg, name)

    sum_n_q = 0.0
    sum_n_p = 0.0
    sum_n_h = 0.0
    total_machines = 0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
        p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
        h_nominal = float(hreg[f"queda_nominal_conjunto_{i}"])
        sum_n_q += n_machines * q_nominal
        sum_n_p += n_machines * p_nominal
        sum_n_h += n_machines * h_nominal
        total_machines += n_machines

    teif = _clamp_outage_pct(float(hreg.get("teif", 0.0) or 0.0), "teif", name)
    ip = _clamp_outage_pct(float(hreg.get("ip", 0.0) or 0.0), "ip", name)
    availability = _availability(teif, ip)

    # Generation cap mirrors the legacy behavior — the source model caps power at rated
    # installed capacity, derated by availability.
    max_generation = sum_n_p * availability

    # Inputs needed for head correction.
    vol_min = float(hreg["volume_minimo"])
    vol_max = float(hreg["volume_maximo"])
    cf_raw = hreg.get("canal_fuga_medio")
    rho_esp_raw = hreg.get("produtibilidade_especifica")
    tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
    perdas = float(hreg.get("perdas", 0.0) or 0.0)
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()

    # If we lack the inputs to compute the head correction, fall back to the
    # simple sum.  This protects against malformed/incomplete hidr rows.
    if (
        total_machines == 0
        or cf_raw is None
        or is_na(cf_raw)
        or rho_esp_raw is None
        or is_na(rho_esp_raw)
        or float(rho_esp_raw) <= 0.0
    ):
        return sum_n_q * availability, max_generation

    cf = float(cf_raw)
    rho_esp = float(rho_esp_raw)
    kturb = _KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]

    if h_op_override is not None:
        # Per-stage caller supplies the operating head directly (= ρ_eq / ρ_esp for
        # the stage), so the cap tracks the per-stage CFUGA/CMONT head. h_int == h_op
        # holds in every branch below, so reuse it for the prodt_eq cap too.
        h_op = h_op_override
        h_int = h_op_override
    elif tipo_reg == "M":
        v65 = vol_min + 0.65 * (vol_max - vol_min)
        # The source model's ``h^{65%}`` is the *integrated* net head over [V_min,
        # V_65], not the snapshot at V = V_65.  Verified against M. DE MORAES (the
        # diagnostic case): with this interpretation the formula reproduces the observed
        # the source model cap of 1084.95 m³/s exactly.
        h_int_gross = mean_cota(coeffs, vol_min, v65) - cf
        h_op = apply_hydraulic_loss(h_int_gross, tipo_perda, perdas)
        # For M plants ``prodt^M`` integrates over the same range, so h_int
        # equals h_op — the same value is reused intentionally.
        h_int = h_op
    else:
        # Run-of-river ('S') and daily-regulated ('D') plants operate at a ~constant
        # reference volume (``volume_referencia``): operating head = forebay cota at
        # vol_ref minus tailrace and losses, used for BOTH the turbine affinity ratio
        # and prodt.  NOT the machine-weighted *nominal* head — that makes the affinity
        # ratio ``h_op / h_nominal`` ≈ 1, a silent no-op leaving ``max_turbined =
        # Σ(n·q)·availability`` uncorrected.  Matches the source model's operating head
        # ``(GHIDUH/QTURUH)/ρ_esp`` (ITAIPU 113.37 m vs nominal 117 m).
        vol_ref_raw = hreg.get("volume_referencia")
        if vol_ref_raw is None or is_na(vol_ref_raw) or float(vol_ref_raw) <= 0.0:
            h_op_gross = mean_cota(coeffs, vol_min, vol_max) - cf
        else:
            h_op_gross = evaluate_cota(coeffs, float(vol_ref_raw)) - cf
        h_op = apply_hydraulic_loss(h_op_gross, tipo_perda, perdas)
        h_int = h_op

    # Defensive: a negative or zero h_op means cota_jus is above the
    # forebay (data error or post-overhaul cota).  Fall back to the simple
    # Σ(n·q) to avoid emitting a meaningless bound.
    if h_op <= 0.0 or h_int <= 0.0:
        return sum_n_q * availability, max_generation

    # Affinity-law correction per machine set.
    sum_kt = 0.0
    for i in range(1, n_sets + 1):
        n_machines = int(hreg[f"maquinas_conjunto_{i}"])
        q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
        h_nominal = float(hreg[f"queda_nominal_conjunto_{i}"])
        if h_nominal <= 0.0:
            sum_kt += n_machines * q_nominal
            continue
        ratio = h_op / h_nominal
        # Use abs to guard against the rare case where h_op flips sign during
        # overhaul-only periods; the exponent < 1 keeps the result finite.
        sum_kt += n_machines * q_nominal * (ratio**kturb)

    # Installed-power-over-equivalent-productivity cap.
    prodt_eq = rho_esp * h_int
    cap_pinst = sum_n_p / prodt_eq if prodt_eq > 0.0 else math.inf

    max_turbined = min(sum_kt, cap_pinst) * availability
    return max_turbined, max_generation


def _reduced_caps(
    hreg: pd.Series, online: dict[int, int], name: str
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` over the online machine subset.

    Computes the head-corrected turbined cap and the rated generation cap for a
    filling plant at a ramp stage where only ``online`` of its machines are in
    service. Rather than threading an override count through the load-bearing
    shared helpers, it copies the cadastro ``hreg`` Series and rewrites each
    ``maquinas_conjunto_{c}`` (for ``c`` in ``1..numero_conjuntos_maquinas``) to
    the online count (``0`` for any group absent from the dict), then re-runs the
    exact full-capacity helpers — so the all-online stage equals the base
    ``hydros.json`` cap by construction. With an empty ``online`` dict every
    count is ``0`` and both caps are ``0.0`` (an explicit zero-capacity stage, not
    a skip).
    """
    n_sets = int(hreg["numero_conjuntos_maquinas"])
    hreg_copy = hreg.copy()
    for c in range(1, n_sets + 1):
        hreg_copy[f"maquinas_conjunto_{c}"] = int(online.get(c, 0))
    max_turbined = _compute_max_turbined_head_corrected(hreg_copy, name)[0]
    max_generation = _compute_max_turbined_rated(hreg_copy)[1]
    return max_turbined, max_generation


def build_mirror_unit_group(
    *,
    name: str,
    bus_id: int,
    min_generation_mw: float,
    max_generation_mw: float,
    min_turbined_m3s: float,
    max_turbined_m3s: float,
    group_id: int = 0,
) -> dict[str, object]:
    """Build one "mirror" unit group for a hydro plant.

    cobre requires every hydro to declare a non-empty ``unit_groups`` array
    (``RawUnitGroup``, all seven fields present). For the ordinary,
    single-group plant every caller in the bridge emits, the group's bounds
    *mirror* the plant's own generation envelope verbatim — no clamping, no
    zeroing of minima, no recomputation — and with a single group,
    ``sum(group maxima) == plant maximum`` holds trivially, which is exactly
    what cobre rule 41 checks, so the rule is satisfied by construction.

    A plant whose halves are maintained independently (e.g. a two-frequency
    split) instead calls this twice, once per physically separate group,
    passing each group's own conjunto-backed bounds (not the plant's) and a
    distinct ``group_id`` — the caller is responsible for the group maxima
    still summing to the plant envelope (rule 41) and for the ids being
    unique within the plant (cobre rule 39; the overlay is id-addressed, so
    array order is never load-bearing — see ``decomp/group_bounds.py``).

    Parameters
    ----------
    name:
        The plant name, used verbatim as the group name.
    bus_id:
        The group's bus id (the plant's own bus for every group, when every
        group sits on the same bus).
    min_generation_mw, max_generation_mw:
        The group's generation bounds in MW.
    min_turbined_m3s, max_turbined_m3s:
        The group's turbined-flow bounds in m^3/s.
    group_id:
        The group's ``id``. Defaults to ``0`` — unit-group ids are dense and
        0-based within a plant, and a single mirror group is the plant's
        only (hence first) group. Pass a distinct value for each group of a
        multi-group plant.

    Returns
    -------
    dict[str, object]
        Exactly the seven ``RawUnitGroup`` keys required by
        ``hydros.schema.json`` — no extras.
    """
    return {
        "id": group_id,
        "name": name,
        "bus_id": bus_id,
        "min_generation_mw": min_generation_mw,
        "max_generation_mw": max_generation_mw,
        "min_turbined_m3s": min_turbined_m3s,
        "max_turbined_m3s": max_turbined_m3s,
    }


def read_cadastro(case: NewaveCase) -> pd.DataFrame:
    """Read ``hidr.dat`` and apply permanent MODIF.DAT overrides.

    Parameters
    ----------
    case:
        Parsed the source model case.

    Returns
    -------
    pd.DataFrame
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina`` with all
        permanent MODIF.DAT overrides (VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
        NUMMAQ) already applied.
    """
    cadastro = case.hidr.cadastro
    return _apply_permanent_overrides(cadastro, case)


def _unit_ramp_summary(
    exph_df: pd.DataFrame,
    newave_code: int,
    start_year: int,
    start_month: int,
) -> str:
    """One-line summary of the units coming online for a filling plant.

    Reads the plant's exph unit rows — a unit row is defined by its
    ``data_entrada_operacao`` (the online date), with
    ``conjunto_maquina_entrada`` naming the machine group — maps each online
    date to a stage via :func:`filling_stage_id`, groups by ``conjunto``,
    counts the units, and renders ``"conj <c>: <n> unit(s) @ stage <s>"`` parts
    (``<s>`` = the stage at which the conjunto is fully online) joined by
    ``"; "``. Returns ``"—"`` when the plant has no parsable unit rows.

    The field-independence guards (``data_entrada_operacao.notna()`` +
    a ``pd.isna(conjunto)`` skip) keep a malformed exph row — a set conjunto
    with a ``NaT`` date, or the inverse — from crashing the cast.
    """
    rows = exph_df.loc[exph_df["codigo_usina"] == newave_code]
    unit_df = rows.loc[rows["data_entrada_operacao"].notna()]
    by_conjunto: dict[int, list[int]] = {}
    for _, ur in unit_df.iterrows():
        conjunto = ur["conjunto_maquina_entrada"]
        if pd.isna(conjunto):
            continue
        ud = ur["data_entrada_operacao"]
        usid = filling_stage_id(ud.year, ud.month, start_year, start_month)
        by_conjunto.setdefault(int(conjunto), []).append(usid)
    if not by_conjunto:
        return "—"
    parts = [
        f"conj {c}: {len(stages)} unit{'s' if len(stages) != 1 else ''} "
        f"@ stage {max(stages)}"
        for c, stages in sorted(by_conjunto.items())
    ]
    return "; ".join(parts)


def _per_stage_turbined_envelope(
    case: NewaveCase, id_map: NewaveIdMap
) -> dict[int, float]:
    """Return ``{cobre_hydro_id: max per-stage max_turbined_m3s}`` from the
    head-corrected per-stage table.

    Delegates to :func:`convert_turbined_bounds_head_corrected` — the exact
    function the pipeline uses to emit the per-(hydro, stage)
    ``hydro_bounds`` rows — instead of re-deriving the per-stage head. This is
    what lets :func:`convert_hydros` raise its declared
    ``generation.max_turbined_m3s`` to cover every emitted per-stage row
    (cobre rule 43) with zero risk of the two ever drifting apart: they are
    two views of the same table, not two formulas that happen to agree today.

    Returns an empty dict when no plant has a per-stage head (no CFUGA/CMONT
    temporal overrides and no seasonal ``VOLREF_SAZ`` row), so callers must
    treat a missing key as "no per-stage variation", not as zero.
    """
    table = convert_turbined_bounds_head_corrected(case, id_map)
    if table is None:
        return {}
    envelope: dict[int, float] = {}
    for hydro_id, value in zip(
        table["hydro_id"].to_pylist(),
        table["max_turbined_m3s"].to_pylist(),
        strict=True,
    ):
        current = envelope.get(hydro_id)
        if current is None or value > current:
            envelope[hydro_id] = value
    return envelope


def convert_hydros(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model hydro plant data to a Cobre ``hydros.json`` dict.

    Reads ``hidr.dat``, ``confhd.dat``, and ``ree.dat`` from *case*.
    Returns a dict with a ``"hydros"`` key containing a list of hydro
    entries sorted by Cobre 0-based ID.

    Also reads ``MODIF.DAT`` (if present) to apply permanent parameter
    overrides and extract temporal override metadata.  Reads ``GHMIN.DAT``
    (if present) to override computed minimum generation values.  Reads
    ``PENALID.DAT`` (if present) to populate per-plant penalty overrides.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping used for bus and downstream-hydro cross-
        references.

    Raises
    ------
    ValueError
        If a hydro in ``confhd.dat`` references a code not found in
        ``hidr.dat``.
    """
    hidr = case.hidr
    confhd = case.confhd
    ree_file = case.ree

    cadastro = hidr.cadastro  # DataFrame indexed by codigo_usina (1-based)
    confhd_df = confhd.usinas
    ree_df = ree_file.rees  # columns: codigo, nome, submercado, ...

    # Apply MODIF.DAT permanent overrides before the main conversion loop.
    cadastro = _apply_permanent_overrides(cadastro, case)

    # Seasonal reference volumes per plant — when present, fed back into the evaporation
    # block as ``reference_volumes_hm3`` so cobre's evaporation linearization matches
    # the per-month reference the source model itself uses.
    seasonal_volref = _read_volref_saz(case)

    # NE plants carrying an exph dead-volume filling row are admitted as real
    # downstream nodes (admission predicate, single source of truth). The
    # set is computed once here and threaded into the cascade walker so upstream
    # plants resolve to the filling plant; it also drives each filling plant's
    # FILLING contract below. Empty (byte-identical to the no-arg call) when the
    # case has no exph or no NE-with-filling plant.
    filling_codes = filling_hydro_codes(
        confhd_df, case.exph.expansoes if case.exph is not None else None
    )

    # Resolve the FICT-cascade for every real plant.  Provides the effective
    # next-real-plant downstream and the sum of any FICT-chain ρ_eq that must
    # be folded back into the upstream real plant's effective ρ_eq.  See
    # ``cobre_bridge.converters.fict_cascade`` for the resolution rules.
    from cobre_bridge.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro, filling_codes=filling_codes)

    # Per-hydro envelope over the head-corrected per-stage turbined caps that
    # convert_turbined_bounds_head_corrected emits into hydro_bounds.parquet.
    # cobre rule 43 forbids any hydro_bounds row from raising max_turbined_m3s
    # above the plant's own declared value, so the reference-head value below
    # must be raised to cover every emitted per-stage row. Empty
    # for a hydro with no per-stage head variation, which keeps its declared
    # value unchanged.
    turbined_envelope = _per_stage_turbined_envelope(case, id_map)

    existing = case.active_hydros

    # Per-stage (year, month) closure over the study horizon, built once and
    # passed into ``filling_min_rate_m3s`` for each filling plant.
    # The exph DataFrame is only consulted when ``filling_codes`` is non-empty,
    # which guarantees ``case.exph`` is not None there.
    horizon = case.horizon
    stage_dates = build_stage_dates(
        horizon.start_year, horizon.start_month, horizon.total_stages
    )

    def stage_year_month(t: int) -> tuple[int, int]:
        d = stage_dates[t]
        return d.year, d.month

    # Build REE-code -> subsystem-code mapping.
    ree_to_submercado: dict[int, int] = {}
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_to_submercado[int(row["codigo"])] = int(row["submercado"])

    # Plants emitted as FPHA (empty unless dger funcao_producao_uhe == 0).
    fpha_codes = fpha_eligible_codes(case)

    # Default operational_start_date for existing (EX) plants: in service since the
    # historical record. Filling (NE) plants override it below with the calendar
    # month they finish filling and enter operation.
    existing_op_date = historical_start_date(case.dger)

    hydros: list[dict] = []
    # One diagnostic row per admitted filling plant, accumulated in the loop and
    # emitted as a single INFO Diagnostic after it (empty for EX-only cases, so
    # nothing new is emitted there).
    filling_diag_rows: list[list[object]] = []
    for _, row in existing.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()

        if newave_code not in cadastro.index:
            raise ValueError(
                f"Hydro plant '{name}' (code {newave_code}) from confhd.dat"
                f" not found in hidr.dat"
            )

        hreg = cadastro.loc[newave_code]

        # Reservoir bounds.
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        # The source model treats Daily-regulation ('D') and run-of-river / fio-d'água
        # ('S') plants as fio-d'água — they can't accumulate water across stages, so the
        # useful volume is NOT a usable reservoir buffer. Collapse the active range to a
        # single point so Cobre's LP mirrors that; otherwise Cobre stores the inflow
        # excess in a phantom buffer and shifts it across stages, where the source
        # model simply spills it.
        #   * 'D' → frozen at ``volume_referencia``.
        #   * 'S' → pinned at ``volume_minimo`` (the source model holds ITAIPU at
        #     VARMPUH 0% = Vmin every stage, spilling the turbine-excess inflow).
        tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()
        if tipo_reg == "D":
            vol_ref_raw = hreg.get("volume_referencia")
            if vol_ref_raw is not None and not pd.isna(vol_ref_raw):
                vol_ref = float(vol_ref_raw)
                vol_min = vol_ref
                vol_max = vol_ref
        elif tipo_reg == "S":
            vol_max = vol_min

        # FILLING phase: an admitted NE plant fills its dead
        # volume from its seeded storage up to ``min_storage_hm3`` over the
        # half-open window ``[start_sid, entry_sid)``, then enters operation at
        # ``entry_sid``. EX plants never enter this branch (filling_codes is the
        # NE-with-exph set), so their entry/exit/filling stay None.
        entry_stage_id: int | None = None
        filling: dict | None = None
        operational_start_date = existing_op_date
        if newave_code in filling_codes:
            # filling_codes is non-empty here ⇒ case.exph is not None (the
            # admission predicate required exph), so .expansoes is safe.
            exph_df = case.exph.expansoes
            filling_row = exph_df.loc[
                (exph_df["codigo_usina"] == newave_code)
                & exph_df["data_inicio_enchimento"].notna()
            ].iloc[0]
            ts = filling_row["data_inicio_enchimento"]
            duracao = int(filling_row["duracao_enchimento"])
            volume_morto = float(filling_row["volume_morto"])
            start_sid, entry_sid = filling_schedule(
                ts.year, ts.month, duracao, horizon.start_year, horizon.start_month
            )
            entry_stage_id = entry_sid
            # Operational start = the month filling completes and the plant enters
            # service, computed from the raw exph schedule so it is truthful even
            # when completion falls outside the study horizon.
            operational_start_date = filling_completion_date(
                ts.year, ts.month, duracao
            ).isoformat()
            # One INFO-diagnostic row per filling plant (rendered after the loop):
            # its filling window, seeded dead volume, and unit-ramp summary. Built
            # for every filling plant, including the ``duracao == 0`` (no-filling-
            # block) case, so the user always sees the admitted plant.
            filling_diag_rows.append(
                [
                    name,
                    newave_code,
                    start_sid,
                    entry_sid,
                    f"{volume_morto:.1f}",
                    _unit_ramp_summary(
                        exph_df,
                        newave_code,
                        horizon.start_year,
                        horizon.start_month,
                    ),
                ]
            )
            if entry_sid > start_sid:
                # JURUENA's single-stage window is anchor-insensitive: the soft
                # target is pinned to ``min_storage_hm3`` regardless of the rate,
                # so the exact per-stage ζ anchor (ζ_t vs ζ_{t+1}) does not matter.
                # TODO(multi-stage-anchor): verify the multi-stage anchor against
                # cobre's LP layout builder before shipping a non-trivial
                # (multi-stage) filling rate.
                #
                # Rate clamp: a plant whose filling completes past the
                # study horizon (``entry_sid > horizon.total_stages``) is a VALID
                # case — it fills but never operates within the study, yet is still
                # emitted with its true ``entry_stage_id``. The rate is summed only
                # over in-horizon stages, so clamp the entry passed to
                # ``filling_min_rate_m3s`` to the horizon: post-study stages have no
                # ``stage_dates`` entry (and thus no ζ to sum), and cobre handles the
                # remaining out-of-horizon fill. The true (unclamped) ``entry_sid``
                # stays on the hydro record below. If ``start_sid >= total_stages``
                # too, ``rate_entry`` collapses to ``start_sid``, giving an empty
                # window → ``filling_min_rate_m3s`` returns 0.0 (no crash).
                rate_entry = min(entry_sid, horizon.total_stages)
                rate = filling_min_rate_m3s(
                    vol_min,
                    volume_morto,
                    start_sid,
                    rate_entry,
                    stage_year_month,
                )
                filling = {
                    "start_stage_id": start_sid,
                    "filling_min_rate_m3s": rate,
                }
            # entry_sid == start_sid (duracao 0): keep filling None but still set
            # entry_stage_id — cobre rejects start_stage_id >= entry_stage_id, so
            # no degenerate filling block is emitted.

        # Generation parameters. Productivity lives in
        # ``hydro_production_models.json`` on cobre HEAD; callers that need
        # the per-hydro base value call ``compute_base_productivities``.
        is_fpha = newave_code in fpha_codes
        # Turbined and generation caps are independent of the production
        # function:
        #   * turbined cap = head-corrected engolimento at the operating head
        #     (m³/s) — the operational dispatch cap; rated Σ n·q_nom overshoots it.
        #   * generation cap = rated installed power Σ n·p_nom = the FPHA GHmax.
        #     This loose ceiling never binds before the head-corrected turbined
        #     cap, so a plant reaches its turbined limit, not nameplate power.
        #
        # The declaration itself is the ENVELOPE over that reference-head value
        # and every per-stage head-corrected cap this hydro emits into
        # hydro_bounds: a plant with per-stage head variation
        # (MODIF.DAT CFUGA/CMONT or seasonal VOLREF_SAZ) can turbine more at a
        # higher-head stage than at the reference head, and cobre rule 43
        # forbids a hydro_bounds row from exceeding the plant's own declared
        # value. A hydro with no per-stage variation is absent from
        # ``turbined_envelope`` and keeps its reference-head value unchanged.
        cobre_hydro_id = id_map.hydro_id(newave_code)
        max_turbined_reference = _compute_max_turbined_head_corrected(hreg, name)[0]
        envelope_value = turbined_envelope.get(cobre_hydro_id)
        max_turbined = (
            max_turbined_reference
            if envelope_value is None
            else max(max_turbined_reference, envelope_value)
        )
        max_generation = _compute_max_turbined_rated(hreg)[1]

        # Minimum outflow from historical minimum (may have been overridden by MODIF).
        vazao_min_hist = hreg.get("vazao_minima_historica")
        vazao_min_hist_val = float(vazao_min_hist) if vazao_min_hist else 0.0
        min_outflow = vazao_min_hist_val if vazao_min_hist_val > 0 else 0.0

        # VAZMINT temporal overrides are now emitted as per-stage bounds in
        # hydro_bounds.parquet via convert_storage_bounds().  The static
        # min_outflow in hydros.json keeps the hidr.dat / VAZMIN base value
        # so that stages before the first VAZMINT record use the correct default.

        # CFUGA/CMONT temporal overrides are handled by convert_production_models;
        # no warning needed here as per-stage productivity is now supported.

        # Static field is always zero — per-stage GHMIN values are
        # emitted in ``hydro_bounds.parquet:min_generation_mw`` by
        # :func:`convert_storage_bounds` so the LP sees the correct
        # bound at every stage.
        min_generation = 0.0

        # Downstream cascade linkage.  Use the FICT-cascade resolver so that
        # plants whose physical downstream is a fictitious plant (or whose
        # confhd jusante is 0 but a name-matched FICT.<NAME> exists) end up
        # wired to the next real plant in the cascade, not silently
        # disconnected.  Fall back to the raw confhd link for plants the
        # resolver did not classify (defensive).
        downstream_id: int | None = None
        resolution = fict_cascade.get(newave_code)
        if resolution is not None and resolution.downstream_code is not None:
            try:
                downstream_id = id_map.hydro_id(resolution.downstream_code)
            except KeyError:
                downstream_id = None
        elif resolution is None:
            jusante_raw = row.get("codigo_usina_jusante")
            if (
                jusante_raw is not None
                and not is_na(jusante_raw)
                and int(jusante_raw) != 0
            ):
                try:
                    downstream_id = id_map.hydro_id(int(jusante_raw))
                except KeyError:
                    pass

        # Bus assignment via REE -> subsystem.
        ree_code = int(row["ree"])
        subsystem_code = ree_to_submercado.get(ree_code)
        if subsystem_code is None:
            raise ValueError(
                f"Hydro plant '{name}' (code {newave_code}) has REE {ree_code}"
                f" which is not present in ree.dat"
            )
        bus_id = id_map.bus_id(subsystem_code)

        evap_coeffs = [float(hreg[f"evaporacao_{m}"]) for m in _EVAP_MONTHS]
        has_evaporation = any(v != 0.0 for v in evap_coeffs)

        # Evaporation linearization points: when the plant has a seasonal row in
        # volref_saz.dat, emit one absolute hm³ value per calendar month (vmin +
        # useful_volume).  Missing months default to vmin (matching the source model's
        # "operate at vmin" semantics for zero entries). Clamped into [min_storage_hm3,
        # max_storage_hm3] so cobre's dimensional validator accepts every value even if
        # a permanent VOLMIN override raised vmin above what the file was written for.
        plant_seasonal_for_evap = seasonal_volref.get(newave_code)
        evap_reference_volumes: list[float] | None = None
        if has_evaporation and plant_seasonal_for_evap:
            evap_reference_volumes = [
                max(
                    vol_min,
                    min(
                        vol_max,
                        vol_min + plant_seasonal_for_evap.get(m, 0.0),
                    ),
                )
                for m in range(1, 13)
            ]

        # Hydraulic loss model derived from tipo_perda / perdas columns.
        tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
        perdas_val = float(hreg.get("perdas", 0.0) or 0.0)
        if tipo_perda == 1 and perdas_val > 0 and not math.isnan(perdas_val):
            hydraulic_losses: dict | None = {
                "type": "factor",
                "value": perdas_val / 100.0,
            }
        elif tipo_perda == 2 and perdas_val > 0 and not math.isnan(perdas_val):
            hydraulic_losses = {"type": "constant", "value_m": perdas_val}
        else:
            hydraulic_losses = None

        # Per-plant penalty overrides removed: newave converts PENALID R$/MWh
        # to rate units once using a system-average productivity and applies the
        # same value to all plants. The global defaults in penalties.json
        # (set by convert_penalties) already carry the converted values.
        penalties: dict | None = None

        # Specific productivity ρ_esp [MW / ((m³/s)·m)] — feeds cobre's energy
        # conversion pipeline (derives ρ_eq from VHA geometry).
        rho_esp_raw = hreg.get("produtibilidade_especifica")
        rho_esp: float | None = None
        if rho_esp_raw is not None:
            rho_esp_f = float(rho_esp_raw)
            if not math.isnan(rho_esp_f) and rho_esp_f > 0.0:
                rho_esp = rho_esp_f

        # FPHA reservoirs: cobre fits the production function from geometry +
        # tailrace, so hand it the dimensionless turbine efficiency
        # (eta = rho_esp / K) and select the "fpha" generation model. Eligibility
        # guarantees rho_esp is present. Non-eligible plants (run-of-river, or
        # non-FPHA cases) keep the constant-productivity path with no efficiency.
        # ``is_fpha`` was computed above for the capacity-cap branch.
        efficiency: dict | None = None
        if is_fpha and rho_esp is not None:
            efficiency = {
                "type": "constant",
                "value": _fpha_efficiency(rho_esp, name),
            }

        # Tailrace as a zero-order polynomial = canal_fuga_medio (constant). Cobre
        # subtracts the tailrace level from the upstream head when deriving ρ_eq;
        # without this source-model's productivity will not match.
        cf_raw = hreg.get("canal_fuga_medio")
        tailrace: dict | None = None
        if cf_raw is not None:
            cf_val = float(cf_raw)
            if not math.isnan(cf_val) and cf_val > 0.0:
                tailrace = {"type": "polynomial", "coefficients": [cf_val]}

        evaporation: dict | None = None
        if has_evaporation:
            evaporation = {"coefficients_mm": evap_coeffs}
            if evap_reference_volumes is not None:
                evaporation["reference_volumes_hm3"] = evap_reference_volumes

        hydro_entry: dict = {
            "id": cobre_hydro_id,
            "name": name,
            "operational_start_date": operational_start_date,
            "downstream_id": downstream_id,
            "reservoir": {
                "min_storage_hm3": vol_min,
                "max_storage_hm3": vol_max,
            },
            "outflow": {
                "min_outflow_m3s": min_outflow,
                "max_outflow_m3s": None,
            },
            "generation": {
                "model": "fpha" if is_fpha else "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": min_generation,
                "max_generation_mw": max_generation,
            },
            "unit_groups": [
                build_mirror_unit_group(
                    name=name,
                    bus_id=bus_id,
                    min_generation_mw=min_generation,
                    max_generation_mw=max_generation,
                    min_turbined_m3s=0.0,
                    max_turbined_m3s=max_turbined,
                )
            ],
            "specific_productivity_mw_per_m3s_per_m": rho_esp,
            "evaporation": evaporation,
            "tailrace": tailrace,
            "diversion": None,
            "filling": filling,
            "efficiency": efficiency,
            "hydraulic_losses": hydraulic_losses,
            "penalties": penalties,
            "entry_stage_id": entry_stage_id,
            "exit_stage_id": None,
        }
        hydros.append(hydro_entry)

    hydros.sort(key=lambda h: h["id"])

    # Surface the admitted dead-volume filling plants as a single INFO
    # Diagnostic with one table row per plant (the thermal-bounds diagnostic
    # shape). Emitted only when at least one filling plant was seen, so EX-only
    # cases add nothing. The de-dup in ``_finalize_diagnostics`` keys on
    # ``(code, summary)``, so the per-plant detail must ride on the table, not on
    # N separate diagnostics.
    if filling_diag_rows:
        emit(
            Diagnostic(
                code="ne-filling-plant",
                severity=Severity.INFO,
                category="Filling plants",
                title=(
                    f"Dead-volume filling plants admitted ({len(filling_diag_rows)})"
                ),
                summary=(
                    f"{len(filling_diag_rows)} NE plant(s) admitted with a "
                    "dead-volume filling schedule; each fills before operating "
                    "and ramps capacity as units come online."
                ),
                table=DiagnosticTable(
                    columns=[
                        "Plant",
                        "Code",
                        "Fill start",
                        "Operates from",
                        "Vol. morto %",
                        "Unit ramp",
                    ],
                    rows=filling_diag_rows,
                    justify=["left", "right", "right", "right", "right", "left"],
                    caption=(
                        "Stage ids are 0-based; capacity is 0 until 'Operates from'."
                    ),
                ),
            ),
            logger=_LOG,
        )

    return {
        "$schema": _SCHEMA_URL,
        "hydros": hydros,
    }


def compute_prodt_sin_mean(case: NewaveCase) -> float:
    """Return ``PROD_MEDIA_SIN`` = mean PRODT over existing plants (the source model).

    This is the system-average productivity the source model uses to convert the PENALID
    R$/MWh penalties to the flow/volume domain (pmo.dat "PENALIDADE POR VIOLACAO
    DE VAZAO MINIMA" etc.). Two conventions matter and are matched here:

    - the per-plant productivity is **PRODT** (:func:`_equivalent_productivity`),
      not the 65%-reference point value used for energy; and
    - the mean is over **all** existing non-fictitious plants **including** those
      with zero PRODT — no ``ρ > 0`` filter and no fictitious-cascade fold-in.

    On the example case this returns ≈ 0.6299, matching pmo's penalty-implied ρ
    (0.6294) to 0.06%; the legacy 65%-reference mean was 0.656 (~4% high).
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    existing = case.active_hydros

    prodt: list[float] = []
    for _, row in existing.iterrows():
        code = int(row["codigo_usina"])
        if code in cadastro.index:
            prodt.append(_equivalent_productivity(cadastro.loc[code]))
    return sum(prodt) / len(prodt) if prodt else 1.0


def compute_per_stage_prodt_sin_mean(case: NewaveCase) -> list[float]:
    """Return ``PROD_MEDIA_SIN[s]`` = mean PRODT per stage (CFUGA/CMONT-driven).

    Same plant set and convention as :func:`compute_prodt_sin_mean` (mean PRODT
    over all existing plants, zeros included, no FICT fold), but **per stage**:
    the handful of plants carrying MODIF.DAT CFUGA/CMONT temporal overrides make
    the SIN mean drift slightly per configuration — ≈0.15% on the example case,
    the decimal wiggle pmo.dat shows on the VAZMIN / TURBMN / TURBMX penalties
    (820.53…821.78). PRODT ignores VOLREF_SAZ, so only CFUGA/CMONT move it.

    Returns ``[]`` when the horizon can't be read (mocked-pipeline unit tests).
    """
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return []

    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    existing = case.active_hydros
    codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal = _extract_temporal_overrides(case, codes)
    drop_by_plant = {
        code: [o for o in ov if o["type"] in ("CFUGA", "CMONT")]
        for code, ov in temporal.items()
    }

    stage_sum = [0.0] * total_stages
    count = 0
    for code in codes:
        if code not in cadastro.index:
            continue
        hreg = cadastro.loc[code]
        series = _per_stage_equivalent_productivities(
            hreg,
            _equivalent_productivity(hreg),
            drop_by_plant.get(code, []),
            case,
            total_stages,
        )
        for s in range(total_stages):
            stage_sum[s] += series[s]
        count += 1
    return [v / count for v in stage_sum] if count else []


def _fpha_computed_config(hreg: pd.Series) -> dict:
    """``fpha_config`` for a computed-FPHA plant, with the source model's fitting
    window.

    The source model fits the FPHA volume axis over the *operative* storage range, which
    depends on the regulation type:

    - ``tipo_regulacao == "M"`` (monthly reservoir) → multi-volume over
      ``[volume_minimo, volume_maximo]``.
    - ``"D"`` / ``"S"`` (daily / run-of-river) → single-volume at
      ``volume_referencia`` (the reference operating volume).

    Passing this as ``fitting_window`` makes cobre fit the same volume grid. Without it
    cobre falls back to the full geometry span, which fits the "D" plants that carry a
    cadastro storage range (e.g. ITAIPU, JIRAU) as multi-volume even though the source
    model collapses them to a single volume.
    """
    reg = str(hreg["tipo_regulacao"]).strip()
    if reg == "M":
        vlo = float(hreg["volume_minimo"])
        vhi = float(hreg["volume_maximo"])
    else:
        vref = float(hreg["volume_referencia"])
        vlo = vhi = vref
    return {
        "source": "computed",
        "fitting_window": {"volume_min_hm3": vlo, "volume_max_hm3": vhi},
    }


def _seasonal_reference_volume(
    monthly_useful: dict[int, float],
    vmin: float,
    vmax: float,
    *,
    is_fpha: bool,
    fpha_config: dict | None = None,
) -> list[dict]:
    """Build 12 seasonal ``reference_volume`` entries from a volref_saz row.

    ``season_id = calendar_month - 1`` (matching the ``stages.json`` season map). The
    source model stores the seasonal reference as *useful* storage above ``vmin``, so
    the absolute reference volume is ``clamp(vmin + useful, vmin, vmax)`` (a missing or
    zero month means "operate at vmin"). FPHA seasons additionally carry ``fpha_config``
    (see :func:`_fpha_computed_config`); non-FPHA seasons declare only the reference
    volume (their productivity comes from ``hydro_energy_productivity.parquet``), so the
    value serves purely as the backwater reference for the upstream FPHA plant.
    """
    seasons: list[dict] = []
    for month in range(1, 13):
        useful = monthly_useful.get(month, 0.0)
        abs_vol = max(vmin, min(vmax, vmin + useful))
        season: dict = {
            "season_id": month - 1,
            "model": "fpha" if is_fpha else "constant_productivity",
            "reference_volume": {"volume_hm3": abs_vol},
        }
        if is_fpha and fpha_config is not None:
            season["fpha_config"] = fpha_config
        seasons.append(season)
    return seasons


def convert_production_models(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Build ``hydro_production_models.json`` with per-hydro model selection.

    - **Non-FPHA cases (and run-of-river plants in FPHA cases):** a single
      ``stage_ranges`` entry with ``model: "constant_productivity"`` and no
      numeric value; ``productivity_mw_per_m3s`` is supplied per-(hydro, stage)
      in ``hydro_energy_productivity.parquet`` (see
      :func:`convert_hydro_energy_productivity`). Cross-file validation in cobre
      rejects double-supply (JSON + parquet) and coverage gaps, so keeping
      productivity strictly in the parquet eliminates the conflict surface.
    - **FPHA reservoirs** (``dger`` ``funcao_producao_uhe == 0`` and the plant
      has storage swing — see :func:`fpha_eligible_codes`): ``model: "fpha"``
      with ``fpha_config: {source: "computed"}``; cobre fits the production
      function from geometry + tailrace families, so no parquet productivity is
      emitted for these plants.

    **Reference volume.** In FPHA cases, ``reference_volume`` (V_ref) sets the FPHA
    backwater / tailrace level and the ρ_eq reference. The source model's per-month
    seasonal reference comes from ``volref_saz.dat``: a plant with a seasonal row is
    emitted in ``seasonal`` mode with one absolute ``reference_volume`` per season
    (``clamp(vmin + useful, vmin, vmax)``); a plant without one falls back to
    ``percentile 0.65`` (= cobre's default, the source model's altura_65). This is
    emitted for FPHA reservoirs **and** for any non-FPHA plant that has a seasonal row,
    because cobre reads a plant's *downstream* ``reference_volume`` (via
    ``downstream_id``) to set that plant's backwater — so each plant must be a correct
    reference for its upstream FPHA neighbour.

    When FPHA is active, a file-level ``fpha_plane_reduction`` block parsed from
    ``tratamento-fpha`` is added (see :func:`_parse_fpha_plane_reduction`).

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built entity ID map used to translate the source model plant codes to
        0-based Cobre hydro IDs.

    Returns
    -------
    dict
        A dict with a ``"production_models"`` key ready to serialise as
        ``system/hydro_production_models.json``.
    """
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]
    fpha_codes = fpha_eligible_codes(case)

    # Seasonal reference volumes (volref_saz.dat) drive the FPHA reference volume V_ref
    # — only relevant in FPHA cases. V_ref feeds both a plant's own rho_eq AND, via
    # cobre's downstream lookup (`downstream_id`), the backwater / tailrace level of the
    # plant ABOVE it. So we emit reference_volume for every plant that has a volref_saz
    # row (FPHA or not), so each plant is a correct backwater reference for its upstream
    # FPHA neighbour. Plants without a row fall through to cobre's 0.65 default fraction
    # (= the source model's altura_65).
    seasonal_volref = _read_volref_saz(case) if case.fpha_enabled else {}
    # Load the cadastro whenever per-plant volume info is needed — for the FPHA
    # fitting window (every FPHA plant) or the seasonal reference volumes.
    cadastro = (
        _apply_permanent_overrides(case.hidr.cadastro, case)
        if (seasonal_volref or fpha_codes)
        else None
    )

    production_models: list[dict] = []
    for newave_code in confhd_codes:
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue

        is_fpha = newave_code in fpha_codes
        monthly_useful = seasonal_volref.get(newave_code)
        hreg = (
            cadastro.loc[newave_code]
            if cadastro is not None and newave_code in cadastro.index
            else None
        )
        fpha_config = (
            _fpha_computed_config(hreg) if (is_fpha and hreg is not None) else None
        )

        if monthly_useful and hreg is not None:
            production_models.append(
                {
                    "hydro_id": hydro_id,
                    "selection_mode": "seasonal",
                    # All 12 seasons are listed below, so the default is never
                    # consulted; a config-less model keeps it valid.
                    "default_model": "constant_productivity",
                    "seasons": _seasonal_reference_volume(
                        monthly_useful,
                        float(hreg["volume_minimo"]),
                        float(hreg["volume_maximo"]),
                        is_fpha=is_fpha,
                        fpha_config=fpha_config,
                    ),
                }
            )
        elif is_fpha:
            # FPHA reservoir with no seasonal reference: fall back to V_65 (percentile
            # 0.65 = the source model's altura_65, and cobre's own default).
            production_models.append(
                {
                    "hydro_id": hydro_id,
                    "selection_mode": "stage_ranges",
                    "stage_ranges": [
                        {
                            "start_stage_id": 0,
                            "end_stage_id": None,
                            "model": "fpha",
                            "fpha_config": fpha_config or {"source": "computed"},
                            "reference_volume": {"percentile": 0.65},
                        }
                    ],
                }
            )
        else:
            production_models.append(
                {
                    "hydro_id": hydro_id,
                    "selection_mode": "stage_ranges",
                    "stage_ranges": [
                        {
                            "start_stage_id": 0,
                            "end_stage_id": None,
                            "model": "constant_productivity",
                        }
                    ],
                }
            )

    production_models.sort(key=lambda m: m["hydro_id"])

    result: dict = {"$schema": _PRODUCTION_MODELS_SCHEMA_URL}
    # File-level FPHA plane reduction (tratamento-fpha), applied to every FPHA
    # plant. Only meaningful when there is at least one FPHA plant.
    if fpha_codes:
        plane_reduction = _parse_fpha_plane_reduction(case)
        if plane_reduction is not None:
            result["fpha_plane_reduction"] = plane_reduction
    result["production_models"] = production_models
    return result


def _total_study_stages(case: NewaveCase) -> int:
    """Return the total number of stages in the study (including post-study)."""
    return case.horizon.total_stages


def _per_stage_integrated_productivities(
    hreg: pd.Series,
    base_integrated: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[float]:
    """Per-stage integrated productivity with CFUGA/CMONT step-function awareness.

    Recomputes the *integrated* productivity (volmin_volmax average) at each
    stage where canal_fuga or cmont state changes; stages with no active override
    return *base_integrated*. The per-stage ``(CFUGA, CMONT)`` step-function sweep
    is shared with :func:`_per_stage_equivalent_productivities` via
    :func:`_per_stage_drop_overrides` (the integrated variant differs only in
    applying :func:`_compute_integrated_productivity` instead of the point ρ).
    """
    if not drop_overrides:
        return [base_integrated] * total_stages

    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    return [
        base_integrated
        if cfuga is None and cmont is None
        else _stored_energy_productivity(
            hreg, canal_fuga_override=cfuga, cmont_override=cmont
        )
        for cfuga, cmont in drops
    ]


def compute_per_stage_own_integrated_productivities(
    case: NewaveCase,
) -> dict[int, list[float]]:
    """Return ``{plant_code: [own integrated ρ per stage]}`` for every existing plant.

    Companion to :func:`compute_per_stage_own_productivities` but with the
    EARM convention: ρ is the volume-integrated productivity (matching
    the source model's ``produtibilidade_equivalente_volmin_volmax``), not the point
    productivity at v_65.  Used by VminOP to override the ``rho_acum_h{id}`` scalar
    parameter so the constraint coefficient matches the source model's stored-energy
    accounting rather than the LP's gen = ρ·Q point coefficient.

    CFUGA/CMONT temporal overrides shift the integrand at every stage from the
    override's effective stage forward; FICT-cascade contribution is folded into the
    upstream real plant's own value so cascade traversal in the source-model-code space
    matches the rewired ``downstream_id`` in ``hydros.json``.
    """
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return {}

    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    from cobre_bridge.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    result: dict[int, list[float]] = {}
    for plant_code in confhd_codes:
        if plant_code not in cadastro.index:
            continue
        hreg = cadastro.loc[plant_code]
        base = _stored_energy_productivity(hreg)
        resolution = fict_cascade.get(plant_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(plant_code, [])
        per_stage = _per_stage_integrated_productivities(
            hreg, base, overrides, case, total_stages
        )
        result[plant_code] = [v + fict_extra for v in per_stage]
    return result


def _read_volref_saz(case: NewaveCase) -> dict[int, dict[int, float]]:
    """Read ``volref_saz.dat`` into ``{plant_code: {calendar_month: useful_vol_hm3}}``.

    The source model uses two distinct conventions inside this file:

    - **Row of all-zeros** — sentinel meaning "no seasonal reference for this
      plant"; the source model falls back to its altura_65 / volume_referencia default.
      We mirror this by *excluding* the plant from the returned mapping.
    - **Row with at least one non-zero value** — real seasonal reference.
      Individual zero months in such a row mean "operate at exactly
      ``volume_minimo``" (useful = 0 above the dead-storage minimum), so we
      keep all twelve monthly entries including explicit zeros.

    Returns an empty dict when ``volref_saz.dat`` is absent.
    """
    vs = case.volref_saz
    if vs is None:
        _LOG.debug("volref_saz.dat not found; seasonal productivity disabled.")
        return {}

    df = vs.volumes
    if df is None or df.empty:
        return {}

    by_plant: dict[int, dict[int, float]] = {}
    for _, row in df.iterrows():
        code = int(row["codigo_usina"])
        month = int(row["mes"])
        by_plant.setdefault(code, {})[month] = float(row["valor"])

    return {
        code: months
        for code, months in by_plant.items()
        if any(v > 0.0 for v in months.values())
    }


def _per_stage_drop_overrides(
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[tuple[float | None, float | None]]:
    """Per-stage effective ``(CFUGA, CMONT)`` from MODIF.DAT step-functions.

    Evolves the CFUGA / CMONT temporal overrides into a per-stage
    ``(canal_fuga, cmont)`` state: each event applies from its stage of effect
    forward; events before the horizon fold in at stage 0; and when
    ``sazonaliza_cfuga_cmont == 1`` the calendar month's latest-year value
    repeats after the last explicit event. Shared by both
    :func:`_per_stage_productivities` (point ρ, combined with VOLREF_SAZ) and
    :func:`_per_stage_equivalent_productivities` (PRODT, which ignores VOLREF_SAZ).
    """
    if not drop_overrides:
        return [(None, None)] * total_stages

    dger = case.dger
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)
    seasonalize = int(getattr(dger, "sazonaliza_cfuga_cmont", 0) or 0) == 1

    events_by_stage: dict[int, list[tuple[float | None, float | None]]] = {}
    last_event_stage = -1
    for override in drop_overrides:
        stage_id = (override["year"] - start_year) * 12 + (
            override["month"] - start_month
        )
        last_event_stage = max(last_event_stage, stage_id)
        if override["type"] == "CFUGA":
            events_by_stage.setdefault(stage_id, []).append(
                (float(override["value"]), None)
            )
        else:  # CMONT
            events_by_stage.setdefault(stage_id, []).append(
                (None, float(override["value"]))
            )

    # sazonaliza_cfuga_cmont == 1: after the last explicit entry the source model
    # repeats the seasonal pattern from the latest year defining each calendar month.
    seasonal_cfuga: dict[int, float] = {}
    seasonal_cmont: dict[int, float] = {}
    if seasonalize:
        latest_cfuga: dict[int, int] = {}
        latest_cmont: dict[int, int] = {}
        for override in drop_overrides:
            year = int(override["year"])
            month = int(override["month"])
            value = float(override["value"])
            if override["type"] == "CFUGA":
                if month not in latest_cfuga or year > latest_cfuga[month]:
                    latest_cfuga[month] = year
                    seasonal_cfuga[month] = value
            elif month not in latest_cmont or year > latest_cmont[month]:
                latest_cmont[month] = year
                seasonal_cmont[month] = value

    drops: list[tuple[float | None, float | None]] = []
    active_cfuga: float | None = None
    active_cmont: float | None = None
    for stage_id in range(total_stages):
        if stage_id == 0:
            applicable_stages = sorted(s for s in events_by_stage if s <= 0)
        elif stage_id in events_by_stage:
            applicable_stages = [stage_id]
        else:
            applicable_stages = []
        for past_stage in applicable_stages:
            for cfuga_val, cmont_val in events_by_stage[past_stage]:
                if cfuga_val is not None:
                    active_cfuga = cfuga_val
                if cmont_val is not None:
                    active_cmont = cmont_val

        calendar_month = ((start_month - 1 + stage_id) % 12) + 1
        if seasonalize and stage_id > last_event_stage:
            if calendar_month in seasonal_cfuga:
                active_cfuga = seasonal_cfuga[calendar_month]
            if calendar_month in seasonal_cmont:
                active_cmont = seasonal_cmont[calendar_month]

        drops.append((active_cfuga, active_cmont))
    return drops


def _per_stage_equivalent_productivities(
    hreg: pd.Series,
    base_prodt: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[float]:
    """Per-stage PRODT, applying CFUGA/CMONT temporal overrides.

    PRODT (vol_min→vol_max equivalent) is independent of the operating volume,
    so — unlike :func:`_per_stage_productivities` — VOLREF_SAZ is *not* applied;
    the only per-stage driver is the CFUGA/CMONT tailrace/forebay. Plants with no
    such override get a flat list at ``base_prodt``.
    """
    if not drop_overrides:
        return [base_prodt] * total_stages
    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    return [
        base_prodt
        if cfuga is None and cmont is None
        else _equivalent_productivity(
            hreg, canal_fuga_override=cfuga, cmont_override=cmont
        )
        for cfuga, cmont in drops
    ]


def _per_stage_productivities(
    hreg: pd.Series,
    base_productivity: float,
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
    seasonal_volref_by_month: dict[int, float] | None = None,
) -> list[float]:
    """Build per-stage productivity values from seasonal volref + CFUGA/CMONT.

    For each stage *s* (0-based), the productivity is computed as:

    1. Determine calendar month ``m = ((start_month − 1 + s) mod 12) + 1``.
    2. Pick reference useful volume: ``seasonal_volref_by_month[m]`` when
       present and positive, else fall back to the source model's altura_65 /
       volume_referencia convention (i.e. use *base_productivity*).
    3. Apply any active CFUGA/CMONT temporal override (step-function from its
       stage of effect forward until the next event of the same type).

    Returns a list of length *total_stages*.  If neither seasonal nor temporal
    overrides apply for a stage, that stage's value equals *base_productivity*.
    """
    has_seasonal = bool(seasonal_volref_by_month)
    if not drop_overrides and not has_seasonal:
        return [base_productivity] * total_stages

    start_month = int(case.dger.mes_inicio_estudo)
    drops = _per_stage_drop_overrides(drop_overrides, case, total_stages)
    seasonal = seasonal_volref_by_month or {}

    values: list[float] = []
    for stage_id in range(total_stages):
        active_cfuga, active_cmont = drops[stage_id]
        calendar_month = ((start_month - 1 + stage_id) % 12) + 1
        vol_useful = seasonal.get(calendar_month)

        if vol_useful is None and active_cfuga is None and active_cmont is None:
            values.append(base_productivity)
        else:
            values.append(
                _compute_productivity(
                    hreg,
                    canal_fuga_override=active_cfuga,
                    cmont_override=active_cmont,
                    useful_volume_override=vol_useful,
                )
            )

    # Post-study tail continues the seasonal cycle: VOLREF_SAZ is an always- seasonal
    # monthly reference volume, and CFUGA/CMONT repeat seasonally when
    # ``sazonaliza_cfuga_cmont == 1`` (handled above). The source model re-applies these
    # seasonal patterns every year, including post-study — only the quantities without a
    # seasonalize flag (outflow / turbined bounds) freeze.
    return values


def convert_hydro_energy_productivity(
    case: NewaveCase, id_map: NewaveIdMap
) -> pa.Table:
    """Build the per-(hydro, stage) equivalent-productivity (ρ_eq) parquet table.

    **Every** plant gets an ``equivalent_productivity_mw_per_m3s`` value here,
    FPHA and non-FPHA alike. `convert_production_models` emits no numeric
    productivity, so this parquet is the single ρ_eq source. Plants without
    CFUGA/CMONT/seasonal overrides emit one ``stage_id = NULL`` default row;
    plants with overrides emit one row per study stage.

    FPHA plants are **not** excluded: cobre's energy-conversion build derives
    their ρ_eq from this parquet override, because ``build_energy_and_templates``
    feeds the alternative "VHA geometry + ρ_esp" derivation path an *empty*
    geometry map — so the parquet override is the only working source. (The FPHA
    production function φ itself is fit separately by cobre from geometry +
    tailrace + efficiency.) Excluding FPHA plants makes cobre fail at load with
    "FPHA hydro '…' cannot derive ρ_eq".

    ``reference_outflow_m3s`` and ``specific_productivity_mw_per_m3s_per_m`` are
    left NULL. The reference volume V_ref is declared in
    ``hydro_production_models.json`` — see :func:`convert_production_models`.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    seasonal_volref = _read_volref_saz(case)

    needs_per_stage = bool(plants_with_drop_overrides) or bool(seasonal_volref)
    total_stages = _total_study_stages(case) if needs_per_stage else 0

    # FICT-cascade: when a real plant's energy-cascade traverses fictitious plants,
    # fold those FICTs' ρ_eq into the upstream real plant's own ρ_eq so cobre's
    # per-plant cascade sum (from ``hydro_energy_productivity.parquet`` plus the
    # rewired ``downstream_id``) reproduces the source model's
    # ``produtibilidade_acumulada_calculo_earm``.  FICT plants have ρ_esp = 0 in the
    # bundled cases, so this is numerically a no-op there — a structural fix, robust
    # to non-zero FICT productivities.
    from cobre_bridge.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    hydro_ids: list[int] = []
    stage_ids: list[int | None] = []
    equiv_prods: list[float] = []

    for newave_code in sorted(confhd_codes):
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        hreg = cadastro.loc[newave_code]
        legacy_base = _compute_productivity(hreg)
        resolution = fict_cascade.get(newave_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(newave_code, [])
        plant_seasonal = seasonal_volref.get(newave_code)

        if not overrides and not plant_seasonal:
            hydro_ids.append(hydro_id)
            stage_ids.append(None)
            equiv_prods.append(legacy_base + fict_extra)
        else:
            per_stage = _per_stage_productivities(
                hreg,
                legacy_base,
                overrides,
                case,
                total_stages,
                seasonal_volref_by_month=plant_seasonal,
            )
            for stage_id, value in enumerate(per_stage):
                hydro_ids.append(hydro_id)
                stage_ids.append(stage_id)
                equiv_prods.append(value + fict_extra)

    nulls = [None] * len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(
                equiv_prods, type=pa.float64()
            ),
            "reference_outflow_m3s": pa.array(nulls, type=pa.float64()),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                nulls, type=pa.float64()
            ),
        }
    )


def convert_turbined_bounds_head_corrected(
    case: NewaveCase, id_map: NewaveIdMap
) -> pa.Table | None:
    """Per-stage ``max_turbined_m3s`` for plants whose operating head varies by stage.

    The engolimento (max turbinable flow) depends on the operating head — both via
    the turbine affinity ratio and the ``p_inst / prodt_eq`` installed-power cap. For
    plants carrying MODIF.DAT CFUGA/CMONT temporal overrides (or a seasonal V_ref),
    that head changes stage-to-stage, exactly as the per-stage equivalent
    productivity does (:func:`convert_hydro_energy_productivity`). The static
    per-plant cap in ``hydros.json`` is computed at a single reference head, so at
    high-flow/low-head stages it under-caps turbining and forces spill (lost hydro →
    extra thermal). This emits a per-(hydro, stage) ``max_turbined`` override using
    the SAME per-stage head (``h = ρ_eq / ρ_esp``) that drives productivity, so the
    two stay consistent.

    Returns a ``(hydro_id, stage_id, max_turbined_m3s)`` table for the affected
    plants/stages, or ``None`` when no plant has a per-stage head.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)
    confhd_codes = [int(r["codigo_usina"]) for _, r in case.active_hydros.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }
    seasonal_volref = _read_volref_saz(case)
    if not drop_overrides and not seasonal_volref:
        return None

    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return None

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    max_turbined_vals: list[float] = []

    for newave_code in sorted(confhd_codes):
        overrides = drop_overrides.get(newave_code, [])
        plant_seasonal = seasonal_volref.get(newave_code)
        if not overrides and not plant_seasonal:
            continue
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        hreg = cadastro.loc[newave_code]
        rho_esp_raw = hreg.get("produtibilidade_especifica")
        if rho_esp_raw is None or is_na(rho_esp_raw) or float(rho_esp_raw) <= 0.0:
            continue
        rho_esp = float(rho_esp_raw)
        name = str(hreg.get("nome_usina", newave_code))

        legacy_base = _compute_productivity(hreg)
        per_stage_prod = _per_stage_productivities(
            hreg,
            legacy_base,
            overrides,
            case,
            total_stages,
            seasonal_volref_by_month=plant_seasonal,
        )
        for stage_id, prod in enumerate(per_stage_prod):
            if prod <= 0.0:
                continue
            h_op = prod / rho_esp
            max_turbined = _compute_max_turbined_head_corrected(
                hreg, name, h_op_override=h_op
            )[0]
            hydro_ids.append(hydro_id)
            stage_ids.append(stage_id)
            max_turbined_vals.append(max_turbined)

    if not hydro_ids:
        return None

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "max_turbined_m3s": pa.array(max_turbined_vals, type=pa.float64()),
        }
    )


def compute_per_stage_own_productivities(
    case: NewaveCase,
) -> dict[int, list[float]]:
    """Return ``{plant_code: [own ρ_eq per stage]}`` for every existing plant.

    Per-stage own productivity reflects MODIF.DAT CFUGA / CMONT temporal
    overrides — for stages before any override the value is the base
    polynomial-integrated productivity; for later stages it picks up the
    effective tailrace / forebay overrides as they take effect. Plants
    without any temporal overrides return a flat list of length
    ``total_stages``.

    Used by the VminOP RHS calculation so that the absolute bound
    ``(pct/100) × useful + dead`` is computed with the **same** per-stage
    ρ_acum that cobre uses to evaluate the LHS at solve time — otherwise
    the constraint silently drifts at every stage where overrides apply
    or for any plant upstream of an overridden plant in the cascade.

    Keys are the source model plant codes (not Cobre ids) since cascade traversal in
    ``compute_accumulated_productivities`` works in the source-model-code space.
    """
    total_stages = _total_study_stages(case)
    if total_stages <= 0:
        return {}

    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(case, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    seasonal_volref = _read_volref_saz(case)

    # FICT-cascade fold-in: per-stage ρ_eq must already include any FICT
    # contribution so that the per-stage ρ_acum used by VminOP and EARM
    # accounting matches the topology rewired into ``hydros.json``.
    from cobre_bridge.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    result: dict[int, list[float]] = {}
    for plant_code in confhd_codes:
        if plant_code not in cadastro.index:
            continue
        hreg = cadastro.loc[plant_code]
        legacy_base = _compute_productivity(hreg)
        resolution = fict_cascade.get(plant_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(plant_code, [])
        plant_seasonal = seasonal_volref.get(plant_code)
        per_stage = _per_stage_productivities(
            hreg,
            legacy_base,
            overrides,
            case,
            total_stages,
            seasonal_volref_by_month=plant_seasonal,
        )
        result[plant_code] = [v + fict_extra for v in per_stage]
    return result


def compute_base_productivities(
    case: NewaveCase, id_map: NewaveIdMap
) -> dict[int, float]:
    """Return ``{hydro_id: base_productivity_mw_per_m3s}`` for every hydro.

    The base productivity is the value `_compute_productivity` returns with no
    CFUGA/CMONT overrides applied — i.e. the productivity used when the case
    has no temporal overrides for that plant. Consumers that previously read
    ``hydros_dict[i]["generation"]["productivity_mw_per_m3s"]`` should call
    this instead now that productivity has moved out of `hydros.json`.
    """
    cadastro = _apply_permanent_overrides(case.hidr.cadastro, case)

    confhd_df = case.confhd.usinas
    existing = case.active_hydros

    # FICT-cascade fold-in — keep this in lockstep with the other productivity
    # helpers so every downstream consumer sees the same effective ρ_eq.
    from cobre_bridge.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    result: dict[int, float] = {}
    for _, row in existing.iterrows():
        newave_code = int(row["codigo_usina"])
        if newave_code not in cadastro.index:
            continue
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue
        base = _compute_productivity(cadastro.loc[newave_code])
        resolution = fict_cascade.get(newave_code)
        if resolution is not None:
            base += resolution.fict_rho_sum
        result[hydro_id] = base
    return result


def generate_hydro_geometry(cadastro: pd.DataFrame, id_map: NewaveIdMap) -> pa.Table:
    """Generate a VHA curve table for all hydro plants in *id_map*.

    For each plant code in ``id_map.all_hydro_codes``, samples 100 uniformly
    spaced volume points on ``[volume_minimo, volume_maximo]``, evaluates the
    volume-to-height polynomial (``a0_volume_cota`` through
    ``a4_volume_cota``), then evaluates the height-to-area polynomial
    (``a0_cota_area`` through ``a4_cota_area``), and collects the results
    into a PyArrow Table.

    Plants where ``volume_minimo == volume_maximo`` (run-of-river with no
    reservoir) are skipped.  Plants whose volume_cota polynomial coefficients
    are all zero are logged as a warning and skipped.  Negative height or area
    values produced by the polynomials are clamped to 0.0.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame (indexed by ``codigo_usina``) with
        permanent MODIF.DAT overrides already applied.
    id_map:
        Pre-built ID mapping; ``id_map.all_hydro_codes`` determines which
        plants are processed.

    Returns
    -------
    pa.Table
        Schema: ``(hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE,
        area_km2: DOUBLE)``.  One row per sampled volume point across all
        eligible plants, ordered by plant then by volume.
    """
    _N_POINTS = 100

    hydro_ids: list[int] = []
    volumes: list[float] = []
    heights: list[float] = []
    areas: list[float] = []

    def _eval_poly(coeffs: list[float], x: np.ndarray) -> np.ndarray:
        """Evaluate a 4th-degree polynomial: c0 + c1*x + ... + c4*x^4."""
        return (
            coeffs[0]
            + coeffs[1] * x
            + coeffs[2] * x**2
            + coeffs[3] * x**3
            + coeffs[4] * x**4
        )

    for newave_code in id_map.all_hydro_codes:
        if newave_code not in cadastro.index:
            _LOG.warning(
                "Plant code %d in id_map is not present in cadastro; skipping.",
                newave_code,
            )
            continue

        hreg = cadastro.loc[newave_code]
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        # Polynomial coefficients for volume -> height (hm3 -> m).
        vc_coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
        if all(c == 0.0 for c in vc_coeffs):
            _LOG.warning(
                "All a0..a4_volume_cota coefficients are zero for plant %d;"
                " skipping geometry generation.",
                newave_code,
            )
            continue

        # Polynomial coefficients for height -> area (m -> km2).
        ca_coeffs = [float(hreg[f"a{i}_cota_area"]) for i in range(5)]

        cobre_id = id_map.hydro_id(newave_code)

        if vol_min == vol_max:
            # Run-of-river or fixed-level: emit a single geometry point
            # so evaporation can still use the surface area.
            v = np.array([vol_min])
            h = _eval_poly(vc_coeffs, v)
            h = np.maximum(h, 0.0)
            a = _eval_poly(ca_coeffs, h)
            a = np.maximum(a, 0.0)
            hydro_ids.append(cobre_id)
            volumes.append(float(v[0]))
            heights.append(float(h[0]))
            areas.append(float(a[0]))
            continue

        vol_grid: np.ndarray = np.linspace(vol_min, vol_max, _N_POINTS)
        height_arr: np.ndarray = _eval_poly(vc_coeffs, vol_grid)
        height_arr = np.maximum(height_arr, 0.0)

        area_arr: np.ndarray = _eval_poly(ca_coeffs, height_arr)
        area_arr = np.maximum(area_arr, 0.0)

        hydro_ids.extend([cobre_id] * _N_POINTS)
        volumes.extend(vol_grid.tolist())
        heights.extend(height_arr.tolist())
        areas.extend(area_arr.tolist())

    schema = pa.schema(
        [
            pa.field("hydro_id", pa.int32()),
            pa.field("volume_hm3", pa.float64()),
            pa.field("height_m", pa.float64()),
            pa.field("area_km2", pa.float64()),
        ]
    )

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "volume_hm3": pa.array(volumes, type=pa.float64()),
            "height_m": pa.array(heights, type=pa.float64()),
            "area_km2": pa.array(areas, type=pa.float64()),
        },
        schema=schema,
    )


def convert_water_withdrawal(case: NewaveCase, id_map: NewaveIdMap) -> pa.Table | None:
    """Convert the source model water withdrawal data to a hydro_bounds Parquet table.

    Reads ``dsvagua.dat`` (optional) from *case* and produces a
    ``pa.Table`` with columns ``(hydro_id: INT32, stage_id: INT32,
    water_withdrawal_m3s: DOUBLE)`` suitable for writing to
    ``constraints/hydro_bounds.parquet``.

    The ``codigo_usina`` field in ``dsvagua.dat`` is a source-model 1-based *plant* code
    (matching ``confhd``), not a posto. Each plant may contribute multiple rows per
    stage (one per consumptive-use or remaining-flow component) which are summed before
    the sign is negated to convert the source model's "withdrawal = negative valor"
    convention into Cobre's positive ``water_withdrawal_m3s``.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built entity ID map.

    Returns
    -------
    pa.Table | None
        Table with schema ``(hydro_id: INT32, stage_id: INT32,
        water_withdrawal_m3s: DOUBLE)`` sorted by ``(hydro_id, stage_id)``,
        or ``None`` when ``dsvagua.dat`` is absent, empty, or yields no
        valid rows after filtering.
    """
    if case.files.dsvagua is None:
        _LOG.debug("dsvagua.dat not found; no water withdrawal.")
        return None

    # Read dger upfront so the ``outros_usos_da_agua`` switch can short-circuit before
    # any dsvagua I/O. The source model treats 0 as "ignore dsvagua.dat" — mirror that
    # here so Cobre's hydro_bounds match The source model's actual run instead of the
    # file contents.
    dger = case.dger
    if int(getattr(dger, "outros_usos_da_agua", 1) or 0) == 0:
        _LOG.info("dger.outros_usos_da_agua == 0; skipping dsvagua.dat conversion.")
        return None

    dsvagua = case.dsvagua
    if dsvagua is None:
        return None
    df = dsvagua.desvios
    if df is None or df.empty:
        return None

    horizon = case.horizon
    start_year = horizon.start_year
    start_month = horizon.start_month
    num_study_stages = horizon.study_months
    num_post_study_stages = horizon.pos_months
    num_total_stages = horizon.total_stages

    # Build a cascade map so NC (Não Construída) plant dsvagua entries can be propagated
    # to the immediately downstream EX plant — the source model applies NC withdrawals
    # to the next real plant in the cascade, while FICT entries are not propagated and
    # are silently dropped.
    confhd_df = case.confhd.usinas
    fict_set = fictitious_codes(confhd_df, case.hidr.cadastro)
    plant_downstream: dict[int, int] = {}
    for _, r in confhd_df.iterrows():
        code = int(r["codigo_usina"])
        jus = r["codigo_usina_jusante"]
        plant_downstream[code] = int(jus) if jus is not None and not pd.isna(jus) else 0

    def _resolve_to_hydro_id(start_code: int) -> int | None:
        """Walk downstream from *start_code* until reaching a plant present
        in ``id_map``. Fictitious plants are never propagated. Returns the
        hydro_id of the first reachable EX plant, or ``None`` if the chain hits
        a terminus or a fictitious node first.
        """
        visited: set[int] = set()
        cur = start_code
        while cur and cur not in visited:
            visited.add(cur)
            if cur in fict_set:
                return None
            try:
                return id_map.hydro_id(cur)
            except KeyError:
                pass
            cur = plant_downstream.get(cur, 0)
        return None

    grouped = df.groupby(["codigo_usina", "data"], as_index=False)["valor"].sum()

    # Accumulate by (hydro_id, stage_id) so propagated NC entries merge
    # cleanly with the downstream plant's own dsvagua row.
    accumulated: dict[tuple[int, int], float] = {}

    for _, row in grouped.iterrows():
        hydro_code = int(row["codigo_usina"])
        hydro_id = _resolve_to_hydro_id(hydro_code)
        if hydro_id is None:
            continue

        dt = row["data"]
        stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
        if stage_id < 0 or stage_id >= num_study_stages:
            continue

        # Negate: The source model negative valor = withdrawal; Cobre positive =
        # withdrawal.
        withdrawal = -float(row["valor"])
        key = (hydro_id, stage_id)
        accumulated[key] = accumulated.get(key, 0.0) + withdrawal

    hydro_ids: list[int] = [k[0] for k in accumulated]
    stage_ids: list[int] = [k[1] for k in accumulated]
    values: list[float] = list(accumulated.values())

    if not hydro_ids:
        return None

    # Extrapolate to post-study period by cycling the last calendar year's
    # Jan–Dec pattern from dsvagua.  The last year of dsvagua always contains
    # a full Jan–Dec cycle.  We build a template keyed by calendar month
    # (1–12) and map each post-study stage to its calendar month.
    if num_post_study_stages > 0:
        # Build per-plant template: hydro_id -> {calendar_month: value}
        # using the last calendar year present in dsvagua per plant.
        plant_by_date: dict[int, dict[tuple[int, int], float]] = {}
        for hid, sid, val in zip(hydro_ids, stage_ids, values):
            # Recover calendar (year, month) from stage_id.
            total_month = start_month + sid  # 1-based month offset from year start
            cal_year = start_year + (total_month - 1) // 12
            cal_month = (total_month - 1) % 12 + 1
            plant_by_date.setdefault(hid, {})[(cal_year, cal_month)] = val

        # Build per-plant set of stages that already have data.
        existing_stages: dict[int, set[int]] = {}
        for hid, sid in zip(hydro_ids, stage_ids):
            existing_stages.setdefault(hid, set()).add(sid)

        for hid, date_map in plant_by_date.items():
            last_year = max(y for y, _m in date_map)
            template: dict[int, float] = {}
            for m in range(1, 13):
                if (last_year, m) in date_map:
                    template[m] = date_map[(last_year, m)]

            if not template:
                continue

            have = existing_stages.get(hid, set())
            for s in range(num_total_stages):
                if s in have:
                    continue
                total_month = start_month + s
                cal_month = (total_month - 1) % 12 + 1
                val = template.get(cal_month)
                if val is not None:
                    hydro_ids.append(hid)
                    stage_ids.append(s)
                    values.append(val)

        _LOG.info(
            "Extrapolated water withdrawal to %d post-study stages "
            "(%d -> %d total stages).",
            num_post_study_stages,
            num_study_stages,
            num_total_stages,
        )

    table = pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "water_withdrawal_m3s": pa.array(values, type=pa.float64()),
        }
    )
    # Sort by (hydro_id, stage_id) for deterministic output.
    return table.sort_by([("hydro_id", "ascending"), ("stage_id", "ascending")])


def convert_storage_bounds(
    case: NewaveCase,
    id_map: NewaveIdMap,
) -> pa.Table | None:
    """Build per-stage hydro bounds from MODIF.DAT temporal overrides.

    Handles four override types:

    - **VMAXT / VMINT**: storage volume overrides as percentage of useful
      volume (vol_max - vol_min).  Flood control and operational minimums.
    - **TURBMAXT / TURBMINT**: turbined flow overrides in absolute m³/s.
      Values of 99999 mean "no limit" (restore default).
    - **VAZMINT**: minimum outflow overrides in absolute m³/s.

    Each override acts as a step function: the value persists from its
    effective date until the next override for the same plant.

    Post-study extrapolation is per-quantity: VMINT/VMAXT repeat the last study
    year's seasonal pattern only when their dger ``sazonaliza_*`` flag is set
    (e.g. flood-control "volume de espera"); outflow (VAZMINT) and turbined
    (TURBMINT/TURBMAXT) have no seasonalize flag and freeze the last study
    stage value.

    Returns ``None`` if MODIF.DAT is absent or contains no relevant records.
    """
    dger = case.dger
    horizon = case.horizon
    start_year = horizon.start_year
    start_month = horizon.start_month
    study_months = horizon.study_months
    total_stages = horizon.total_stages

    # Post-study seasonalize flags (dger). VMINT/VMAXT repeat the last study
    # year's seasonal pattern only when their flag is set (e.g. flood-control
    # "volume de espera"); outflow (VAZMINT) and turbined (TURBMINT/TURBMAXT)
    # have no such flag and always freeze the last study stage value.
    sazonaliza_vmaxt = int(getattr(dger, "sazonaliza_vmaxt", 0) or 0) == 1
    sazonaliza_vmint = int(getattr(dger, "sazonaliza_vmint", 0) or 0) == 1

    # Read hidr.dat with permanent overrides for vol_min/vol_max.
    cadastro = read_cadastro(case)

    # Read confhd for the list of active plant codes.
    confhd_codes = case.active_hydro_codes

    # Determine whether the case has any NE-with-filling plant (admission
    # predicate). The max_generation_mw column is gated on this:
    # EX-only cases keep the byte-identical 8-column schema.
    exph_df = case.exph.expansoes if case.exph is not None else None
    filling_codes = filling_hydro_codes(case.confhd.usinas, exph_df)

    # Extract temporal overrides — empty dict when MODIF.DAT is absent,
    # which is fine because GHMIN.DAT alone can still produce per-stage
    # rows.
    if case.files.modif is None:
        temporal_overrides: dict[int, list[dict]] = {}
    else:
        temporal_overrides = _extract_temporal_overrides(case, confhd_codes)

    def _build_step_function(
        recs: list[dict],
        transform: Callable[[float], float],
        *,
        seasonalize: bool,
    ) -> dict[int, float]:
        """Thin adapter over :func:`cobre_bridge.horizon.seasonal_step_function`.

        Maps MODIF override dicts to ``(year, month, value)`` change-points. The
        forward-fill, big-M clearing, and seasonalize-vs-freeze post-study logic
        now live in the shared helper, so the bounds comparator
        (``bounds_from_inputs``) and this converter stay in lock-step by
        construction instead of via copied code.
        """
        return seasonal_step_function(
            [(int(r["year"]), int(r["month"]), float(r["value"])) for r in recs],
            transform,
            seasonalize=seasonalize,
            horizon=horizon,
        )

    # GHMIN.DAT per-stage minimums.  These are not MODIF.DAT overrides
    # but live alongside them at the per-(hydro, stage) granularity, so
    # they merge naturally into this parquet's row set.
    ghmin_by_plant_stage = _read_ghmin_per_stage(
        case, start_year, start_month, study_months, total_stages
    )

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    min_storage_vals: list[float | None] = []
    max_storage_vals: list[float | None] = []
    min_turbined_vals: list[float | None] = []
    max_turbined_vals: list[float | None] = []
    min_outflow_vals: list[float | None] = []
    min_generation_vals: list[float | None] = []
    max_generation_vals: list[float | None] = []
    # Per-row origin tag: True for a filling-plant ramp row, False for a
    # MODIF/GHMIN row. Drives the ramp-wins de-dup pass before the table build.
    is_ramp_vals: list[bool] = []

    def _identity(val: float) -> float:
        return val

    plant_codes_with_data = set(temporal_overrides) | set(ghmin_by_plant_stage)
    for newave_code in sorted(plant_codes_with_data):
        overrides = temporal_overrides.get(newave_code, [])
        vmaxt = [o for o in overrides if o["type"] == "VMAXT"]
        vmint = [o for o in overrides if o["type"] == "VMINT"]
        turbmaxt = [o for o in overrides if o["type"] == "TURBMAXT"]
        turbmint = [o for o in overrides if o["type"] == "TURBMINT"]
        vazmint = [o for o in overrides if o["type"] == "VAZMINT"]
        ghmin_by_stage = ghmin_by_plant_stage.get(newave_code, {})

        if not any((vmaxt, vmint, turbmaxt, turbmint, vazmint, ghmin_by_stage)):
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

        def _pct_to_hm3(pct: float) -> float:
            return vol_min + (pct / 100.0) * useful

        # Storage bounds (percentage -> hm³). Seasonal post-study iff the
        # corresponding dger flag is set; otherwise freeze.
        vmaxt_by_stage: dict[int, float] = {}
        vmint_by_stage: dict[int, float] = {}
        if useful > 0:
            vmaxt_by_stage = _build_step_function(
                vmaxt, _pct_to_hm3, seasonalize=sazonaliza_vmaxt
            )
            vmint_by_stage = _build_step_function(
                vmint, _pct_to_hm3, seasonalize=sazonaliza_vmint
            )

        # Turbined bounds (absolute m³/s) — no seasonalize flag → freeze.
        turbmaxt_by_stage = _build_step_function(turbmaxt, _identity, seasonalize=False)
        turbmint_by_stage = _build_step_function(turbmint, _identity, seasonalize=False)

        # Outflow bounds (absolute m³/s) — no seasonalize flag → freeze.
        vazmint_by_stage = _build_step_function(vazmint, _identity, seasonalize=False)

        all_stages = sorted(
            set(vmaxt_by_stage)
            | set(vmint_by_stage)
            | set(turbmaxt_by_stage)
            | set(turbmint_by_stage)
            | set(vazmint_by_stage)
            | set(ghmin_by_stage)
        )
        for stage_id in all_stages:
            hydro_ids.append(hydro_id)
            stage_ids.append(stage_id)
            max_storage_vals.append(vmaxt_by_stage.get(stage_id))
            min_storage_vals.append(vmint_by_stage.get(stage_id))
            max_turbined_vals.append(turbmaxt_by_stage.get(stage_id))
            min_turbined_vals.append(turbmint_by_stage.get(stage_id))
            min_outflow_vals.append(vazmint_by_stage.get(stage_id))
            min_generation_vals.append(ghmin_by_stage.get(stage_id))
            # Column present but unpopulated here; the filling-plant unit-ramp
            # branch below populates per-stage ramp caps. None mirrors
            # min_generation null handling.
            max_generation_vals.append(None)
            is_ramp_vals.append(False)

    # Filling-plant unit-ramp branch: a ``NE``-with-filling plant operates from
    # ``entry_sid`` but its turbine/generation capacity at each stage is whatever
    # generating units are online.  Export EXPLICIT 0/reduced caps over the FULL
    # pre-operating window ``[0, full_online_sid)`` (clamped to the in-study horizon)
    # so the parquet carries the true 0→full capacity profile.  ``online_machines``
    # clamps every unit's online stage up to ``entry_sid``, so ``[0, entry_sid)``
    # (PreFilling/Filling) gets ``(0, 0)`` caps and ``[entry_sid, full_online_sid)``
    # gets reduced caps; from ``full_online_sid`` the base ``hydros.json`` caps apply.
    # These rows are inert to cobre's own PreFilling/Filling forcing: its
    # ``hydro_bounds`` reader is a sparse override table with NO stage-window
    # validation, so a ``max=0`` row there leaves the simulation result UNCHANGED —
    # only the exported data gains the explicit 0-cap stages.  A ramp-window stage may
    # also carry a MODIF/GHMIN row; those append SEPARATE rows tagged ``is_ramp=True``,
    # and the de-dup pass below resolves any ``(hydro_id, stage_id)`` collision in
    # favour of the ramp row (the explicit 0-cap wins over a colliding MODIF/GHMIN
    # minimum), since cobre defers duplicate-pair handling.
    if filling_codes and case.exph is not None:
        exph_df = case.exph.expansoes
        for code in sorted(filling_codes):
            try:
                hydro_id = id_map.hydro_id(code)
            except KeyError:
                continue
            if code not in cadastro.index:
                continue
            hreg = cadastro.loc[code]
            name = str(hreg.get("nome_usina", "")).strip() or str(code)

            rows = exph_df.loc[exph_df["codigo_usina"] == code]
            fill_row = rows.loc[rows["data_inicio_enchimento"].notna()].iloc[0]
            ts = fill_row["data_inicio_enchimento"]
            duracao = int(fill_row["duracao_enchimento"])
            _start_sid, entry_sid = filling_schedule(
                ts.year,
                ts.month,
                duracao,
                start_year,
                start_month,
            )

            # inewave parses ``data_entrada_operacao`` (the unit's online date) and
            # ``conjunto_maquina_entrada`` (its machine group) independently, so a
            # unit row can carry a conjunto with a BLANK date (``NaT``) — which would
            # make ``ud.year``/``ud.month`` NaN and crash ``range(...)`` downstream.
            # Filter on the online date (the defining field) and skip a row missing
            # the conjunto: either alone cannot define an online stage.
            unit_rows: list[tuple[int, int]] = []
            unit_df = rows.loc[rows["data_entrada_operacao"].notna()]
            for _, ur in unit_df.iterrows():
                conjunto = ur["conjunto_maquina_entrada"]
                if pd.isna(conjunto):
                    continue
                ud = ur["data_entrada_operacao"]
                usid = filling_stage_id(ud.year, ud.month, start_year, start_month)
                unit_rows.append((int(conjunto), usid))
            if not unit_rows:
                continue

            # All-units-online stage: the exported pre-operating window is
            # [0, full_online_sid). Clamp the upper bound to total_stages so the
            # loop never indexes past the horizon. A
            # plant whose entry is at/after the horizon end still emits explicit
            # 0-cap rows for every in-study stage [0, total_stages) — it never
            # operates in-study, so every stage is pre-operating.
            full_online_sid = max(max(usid, entry_sid) for _c, usid in unit_rows)
            for s in range(min(full_online_sid, total_stages)):
                online = online_machines(unit_rows, entry_sid, s)
                mt, mg = _reduced_caps(hreg, online, name)
                hydro_ids.append(hydro_id)
                stage_ids.append(s)
                min_storage_vals.append(None)
                max_storage_vals.append(None)
                min_turbined_vals.append(None)
                max_turbined_vals.append(mt)
                min_outflow_vals.append(None)
                min_generation_vals.append(0.0)
                max_generation_vals.append(mg)
                is_ramp_vals.append(True)

    if not hydro_ids:
        return None

    # Resolve duplicate ``(hydro_id, stage_id)`` pairs: a ramp row wins over a
    # MODIF/GHMIN row at the same key (the explicit 0-cap during filling must not
    # be undercut by a MODIF/GHMIN minimum). EX-only cases produce no ramp rows
    # (``is_ramp_vals`` all False), so every key is unique and this is a no-op,
    # keeping the regression-guard output byte-identical.
    chosen: dict[tuple[int, int], int] = {}
    for i, (h, s, ramp) in enumerate(zip(hydro_ids, stage_ids, is_ramp_vals)):
        key = (h, s)
        if key not in chosen or ramp:  # ramp overrides an earlier MODIF/GHMIN row
            chosen[key] = i
    keep = sorted(chosen.values())
    if len(keep) != len(hydro_ids):
        hydro_ids = [hydro_ids[i] for i in keep]
        stage_ids = [stage_ids[i] for i in keep]
        min_storage_vals = [min_storage_vals[i] for i in keep]
        max_storage_vals = [max_storage_vals[i] for i in keep]
        min_turbined_vals = [min_turbined_vals[i] for i in keep]
        max_turbined_vals = [max_turbined_vals[i] for i in keep]
        min_outflow_vals = [min_outflow_vals[i] for i in keep]
        min_generation_vals = [min_generation_vals[i] for i in keep]
        max_generation_vals = [max_generation_vals[i] for i in keep]
        is_ramp_vals = [is_ramp_vals[i] for i in keep]

    columns = {
        "hydro_id": pa.array(hydro_ids, type=pa.int32()),
        "stage_id": pa.array(stage_ids, type=pa.int32()),
        "min_storage_hm3": pa.array(min_storage_vals, type=pa.float64()),
        "max_storage_hm3": pa.array(max_storage_vals, type=pa.float64()),
        "min_turbined_m3s": pa.array(min_turbined_vals, type=pa.float64()),
        "max_turbined_m3s": pa.array(max_turbined_vals, type=pa.float64()),
        "min_outflow_m3s": pa.array(min_outflow_vals, type=pa.float64()),
        "min_generation_mw": pa.array(min_generation_vals, type=pa.float64()),
    }
    # Gate the column on filling-plant presence: EX-only cases keep the
    # existing 8-column schema byte-identical (the regression guard depends
    # on this); cobre's parse_hydro_bounds tolerates the absent column.
    if filling_codes:
        columns["max_generation_mw"] = pa.array(max_generation_vals, type=pa.float64())
    return pa.table(columns).sort_by(
        [("hydro_id", "ascending"), ("stage_id", "ascending")]
    )
