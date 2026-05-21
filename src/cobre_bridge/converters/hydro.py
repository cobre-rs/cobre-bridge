"""Hydro entity converter: maps NEWAVE hydro plant data to Cobre hydro JSON."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import (
    Confhd,
    Dger,
    Ghmin,
    Hidr,
    Modif,
    Penalid,
    Ree,
    VolrefSaz,
)

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/hydros.schema.json"
)
_PRODUCTION_MODELS_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/production_models.schema.json"
)

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
    cadastro: pd.DataFrame, nw_files: NewaveFiles
) -> pd.DataFrame:
    """Apply MODIF.DAT permanent overrides to the hidr.dat cadastro.

    Reads ``MODIF.DAT`` from *nw_files* and
    applies permanent override records — VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
    NUMMAQ — to a *copy* of *cadastro*.  The original DataFrame is not
    mutated.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina``.
    nw_files:
        Resolved NEWAVE file paths for the case.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with permanent overrides applied.
    """
    modif_path = nw_files.modif
    if modif_path is None:
        _LOG.debug("MODIF.DAT not found; skipping permanent overrides.")
        return cadastro

    modif = Modif.read(str(modif_path))
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

            # Skip temporal override types — handled separately in ticket-005.
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
                # inewave uses DefaultRegister for unrecognised records.
                _LOG.warning(
                    "MODIF.DAT contains an unrecognised record (DefaultRegister)"
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
    nw_files: NewaveFiles, confhd_codes: list[int]
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
    nw_files:
        Resolved NEWAVE file paths for the case.
    confhd_codes:
        List of plant codes present in the study (from confhd.dat).  Records
        for plants not in this list are excluded.

    Returns
    -------
    dict[int, list[dict]]
        Temporal override records per plant code.  Empty dict if MODIF.DAT is
        absent.
    """
    modif_path = nw_files.modif
    if modif_path is None:
        _LOG.debug("MODIF.DAT not found; no temporal overrides extracted.")
        return {}

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


def _read_ghmin(nw_files: NewaveFiles) -> dict[int, float]:
    """Read GHMIN.DAT and return a mapping of plant code -> minimum generation MW.

    If ``GHMIN.DAT`` is absent (``nw_files.ghmin is None``), returns an
    empty dict.  Only ``patamar == 0`` rows (all load blocks) are used; if
    multiple time periods are present for the same plant, the first
    occurrence (earliest date) is taken.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.

    Returns
    -------
    dict[int, float]
        Plant code -> minimum generation in MW.  Empty dict if file absent.
    """
    ghmin_path = nw_files.ghmin
    if ghmin_path is None:
        _LOG.debug("GHMIN.DAT not found; using computed min_generation fallback.")
        return {}

    ghmin = Ghmin.read(str(ghmin_path))
    df = ghmin.geracoes
    if df is None or df.empty:
        return {}

    # Keep only patamar=0 (all-blocks) rows.
    patamar0 = df[df["patamar"] == 0]
    if patamar0.empty:
        return {}

    # Use the first (earliest) entry per plant.
    first_per_plant = (
        patamar0.sort_values("data").groupby("codigo_usina").first().reset_index()
    )

    return {
        int(row["codigo_usina"]): float(row["geracao"])
        for _, row in first_per_plant.iterrows()
    }


# Mapping from PENALID.DAT variable names to Cobre penalty field names.
_PENALID_VAR_MAP: dict[str, str] = {
    "DESVIO": "water_withdrawal_violation_cost",
    "VAZMIN": "outflow_violation_below_cost",
    "VAZMAX": "outflow_violation_above_cost",
    "GHMIN": "generation_violation_below_cost",
    "TURBMN": "turbined_violation_below_cost",
    "TURBMX": "outflow_violation_above_cost",
}


def _read_penalid(nw_files: NewaveFiles) -> dict[int, dict[str, float]]:
    """Read PENALID.DAT and return per-REE penalty override mappings.

    If ``PENALID.DAT`` is absent (``nw_files.penalid is None``), returns an
    empty dict.  Only the first patamar tier (``patamar_penalidade == 1``)
    is used — tier 2 has NaN costs (unbounded) and is skipped.  NaN values
    within tier 1 are also skipped.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from REE/subsystem code to a dict of Cobre penalty field
        names -> cost in R$/MWh.  Only fields with valid (non-NaN) values
        are included.  Returns an empty dict if the file is absent or
        contains no usable rows.
    """
    penalid_path = nw_files.penalid
    if penalid_path is None:
        _LOG.debug("PENALID.DAT not found; leaving all plant penalties as None.")
        return {}

    penalid = Penalid.read(str(penalid_path))
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

        # Skip NaN values.
        if pd.isna(valor):
            continue

        cost = float(valor)
        if ree_code not in result:
            result[ree_code] = {}
        result[ree_code][cobre_field] = cost

    return result


def read_cadastro(nw_files: NewaveFiles) -> pd.DataFrame:
    """Read ``hidr.dat`` and apply permanent MODIF.DAT overrides.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.

    Returns
    -------
    pd.DataFrame
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina`` with all
        permanent MODIF.DAT overrides (VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
        NUMMAQ) already applied.
    """
    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = hidr.cadastro
    return _apply_permanent_overrides(cadastro, nw_files)


def convert_hydros(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE hydro plant data to a Cobre ``hydros.json`` dict.

    Reads ``hidr.dat``, ``confhd.dat``, and ``ree.dat`` from *nw_files*.
    Returns a dict with a ``"hydros"`` key containing a list of hydro
    entries sorted by Cobre 0-based ID.

    Also reads ``MODIF.DAT`` (if present) to apply permanent parameter
    overrides and extract temporal override metadata.  Reads ``GHMIN.DAT``
    (if present) to override computed minimum generation values.  Reads
    ``PENALID.DAT`` (if present) to populate per-plant penalty overrides.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built ID mapping used for bus and downstream-hydro cross-
        references.

    Raises
    ------
    ValueError
        If a hydro in ``confhd.dat`` references a code not found in
        ``hidr.dat``.
    """
    hidr = Hidr.read(str(nw_files.hidr))
    confhd = Confhd.read(str(nw_files.confhd))
    ree_file = Ree.read(str(nw_files.ree))

    cadastro = hidr.cadastro  # DataFrame indexed by codigo_usina (1-based)
    confhd_df = confhd.usinas
    ree_df = ree_file.rees  # columns: codigo, nome, submercado, ...

    # Apply MODIF.DAT permanent overrides before the main conversion loop.
    cadastro = _apply_permanent_overrides(cadastro, nw_files)

    # Seasonal reference volumes per plant — when present, fed back into the
    # evaporation block as ``reference_volumes_hm3`` so cobre's evaporation
    # linearization matches the per-month reference NEWAVE itself uses.
    seasonal_volref = _read_volref_saz(nw_files)

    # Resolve the FICT-cascade for every real plant.  Provides the effective
    # next-real-plant downstream and the sum of any FICT-chain ρ_eq that must
    # be folded back into the upstream real plant's effective ρ_eq.  See
    # ``cobre_bridge.converters.fict_cascade`` for the resolution rules.
    from cobre_bridge.converters.fict_cascade import resolve_cascade

    fict_cascade = resolve_cascade(confhd_df, cadastro)

    # Collect study plant codes for temporal override extraction.
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    # Read GHMIN.DAT minimum generation map.
    ghmin_map = _read_ghmin(nw_files)

    # Build REE-code -> subsystem-code mapping.
    ree_to_submercado: dict[int, int] = {}
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_to_submercado[int(row["codigo"])] = int(row["submercado"])

    hydros: list[dict] = []
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

        # Generation parameters. Productivity lives in
        # ``hydro_production_models.json`` on cobre HEAD; callers that need
        # the per-hydro base value call ``compute_base_productivities``.
        n_sets = int(hreg["numero_conjuntos_maquinas"])

        max_turbined = 0.0
        max_generation = 0.0
        for i in range(1, n_sets + 1):
            n_machines = int(hreg[f"maquinas_conjunto_{i}"])
            q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
            p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
            max_turbined += q_nominal * n_machines
            max_generation += p_nominal * n_machines

        # Apply TEIF/IP availability derating to max_generation.
        teif = float(hreg.get("teif", 0.0) or 0.0)
        ip = float(hreg.get("ip", 0.0) or 0.0)
        if math.isnan(teif):
            teif = 0.0
        if math.isnan(ip):
            ip = 0.0

        def _clamp_to_100(value: float, label: str) -> float:
            if value > 100.0:
                _LOG.warning(
                    "%s exceeds 100%% for plant %s (%s=%.2f); clamping to 100.",
                    label,
                    name,
                    label.lower(),
                    value,
                )
                return 100.0
            return value

        teif = _clamp_to_100(teif, "teif")
        ip = _clamp_to_100(ip, "ip")
        max_generation *= ((100.0 - teif) / 100.0) * ((100.0 - ip) / 100.0)

        # Minimum outflow from historical minimum (may have been overridden by MODIF).
        vazao_min_hist = hreg.get("vazao_minima_historica")
        min_outflow = (
            float(vazao_min_hist)
            if vazao_min_hist and float(vazao_min_hist) > 0
            else 0.0
        )

        # VAZMINT temporal overrides are now emitted as per-stage bounds in
        # hydro_bounds.parquet via convert_storage_bounds().  The static
        # min_outflow in hydros.json keeps the hidr.dat / VAZMIN base value
        # so that stages before the first VAZMINT record use the correct default.

        # CFUGA/CMONT temporal overrides are handled by convert_production_models;
        # no warning needed here as per-stage productivity is now supported.

        # Min generation: use GHMIN.DAT if available, otherwise approximate.
        ghmin_value = ghmin_map.get(newave_code)
        if ghmin_value is not None:
            min_generation = ghmin_value
        else:
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
                and not _is_na(jusante_raw)
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

        # Evaporation linearization points: when the plant has a seasonal
        # row in volref_saz.dat, emit one absolute hm³ value per calendar
        # month (vmin + useful_volume).  Missing months default to vmin
        # (matching NEWAVE's "operate at vmin" semantics for zero entries).
        # Clamped into [min_storage_hm3, max_storage_hm3] so cobre's
        # dimensional validator accepts every value even if a permanent
        # VOLMIN override raised vmin above what the file was written for.
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

        # Tailrace as a zero-order polynomial = canal_fuga_medio (constant).
        # Cobre subtracts the tailrace level from the upstream head when
        # deriving ρ_eq; without this NEWAVE's productivity will not match.
        cf_raw = hreg.get("canal_fuga_medio")
        tailrace: dict | None = None
        if cf_raw is not None:
            cf_val = float(cf_raw)
            if not math.isnan(cf_val) and cf_val > 0.0:
                tailrace = {"type": "polynomial", "coefficients": [cf_val]}

        hydro_entry: dict = {
            "id": id_map.hydro_id(newave_code),
            "name": name,
            "bus_id": bus_id,
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
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": min_generation,
                "max_generation_mw": max_generation,
            },
            "specific_productivity_mw_per_m3s_per_m": rho_esp,
            "evaporation": (
                {
                    "coefficients_mm": evap_coeffs,
                    **(
                        {"reference_volumes_hm3": evap_reference_volumes}
                        if evap_reference_volumes is not None
                        else {}
                    ),
                }
                if has_evaporation
                else None
            ),
            "tailrace": tailrace,
            "diversion": _make_diversion(newave_code, id_map),
            "filling": None,
            "efficiency": None,
            "hydraulic_losses": hydraulic_losses,
            "penalties": penalties,
            "entry_stage_id": None,
            "exit_stage_id": None,
        }
        hydros.append(hydro_entry)

    hydros.sort(key=lambda h: h["id"])

    return {
        "$schema": _SCHEMA_URL,
        "hydros": hydros,
    }


def _compute_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
    useful_volume_override: float | None = None,
) -> float:
    """Compute constant productivity in MW/(m^3/s) for a hydro plant.

    Reads polynomial coefficients ``a0_volume_cota`` through
    ``a4_volume_cota`` from the plant's cadastro row to map storage volume
    (hm3) to upstream height (m).  Subtracts the tailrace level to obtain
    gross drop, applies the loss model defined by ``tipo_perda`` and
    ``perdas``, then multiplies by ``produtibilidade_especifica``.

    Reference-volume selection (in priority order):

    1. ``useful_volume_override`` — explicit useful volume (hm³ above
       ``volume_minimo``).  Used by the seasonal pathway driven by
       ``volref_saz.dat``: ``V = volume_minimo + useful_volume_override``.
    2. Monthly-regulated plants (``tipo_regulacao == "M"``) → 65% of useful
       storage (``V = vmin + 0.65 × (vmax − vmin)``); matches NEWAVE's
       ``produtibilidade_altura_65`` convention.
    3. All other plant types → ``volume_referencia``.

    ``cmont_override`` short-circuits the upstream polynomial entirely.
    """
    coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]

    canal_fuga = (
        canal_fuga_override
        if canal_fuga_override is not None
        else float(hreg["canal_fuga_medio"])
    )

    if cmont_override is not None:
        # CMONT supplies the upstream level directly.
        net_drop = cmont_override - canal_fuga
    else:
        if all(c == 0.0 for c in coeffs):
            _LOG.warning(
                "All volume_cota coefficients are zero for plant; "
                "returning zero productivity.",
                extra={"plant": hreg.get("nome_usina", "unknown")},
            )
            return 0.0

        def _poly(v: float) -> float:
            """Evaluate h(v) = c0 + c1*v + c2*v^2 + c3*v^3 + c4*v^4."""
            return (
                coeffs[0]
                + coeffs[1] * v
                + coeffs[2] * v**2
                + coeffs[3] * v**3
                + coeffs[4] * v**4
            )

        vol_min = float(hreg["volume_minimo"])

        if useful_volume_override is not None:
            net_drop = _poly(vol_min + useful_volume_override) - canal_fuga
        else:
            tipo_regulacao = str(hreg["tipo_regulacao"]).strip()
            vol_max = float(hreg["volume_maximo"])
            if tipo_regulacao == "M":
                v_65 = vol_min + 0.65 * (vol_max - vol_min)
                net_drop = _poly(v_65) - canal_fuga
            else:
                vol_ref = float(hreg["volume_referencia"])
                net_drop = _poly(vol_ref) - canal_fuga

    # Apply loss model.
    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    if tipo_perda == 1:
        # Multiplicative factor (perdas is a percentage, e.g. 2.35 = 2.35%).
        adjusted_drop = net_drop * (1.0 - perdas / 100.0)
    elif tipo_perda == 2:
        # Additive meters: adjusted_drop = net_drop - perdas
        adjusted_drop = net_drop - perdas
    else:
        adjusted_drop = net_drop

    produtibilidade = float(hreg["produtibilidade_especifica"])
    return produtibilidade * adjusted_drop


def convert_production_models(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Build ``hydro_production_models.json`` with model selection only.

    After the cobre productivity-resolution-rules plan, ``productivity_mw_per_m3s``
    is **optional** in this file: the value is supplied per-(hydro, stage) in
    ``hydro_energy_productivity.parquet`` (see
    :func:`convert_hydro_energy_productivity`). We therefore emit only the model
    selection here — one ``stage_ranges`` entry per hydro spanning the whole
    horizon, carrying ``model: "constant_productivity"`` with no numeric value.

    Cross-file validation in cobre rejects double-supply (JSON + parquet) and
    coverage gaps; keeping productivity strictly in the parquet eliminates the
    conflict surface.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built entity ID map used to translate NEWAVE plant codes to
        0-based Cobre hydro IDs.

    Returns
    -------
    dict
        A dict with a ``"production_models"`` key ready to serialise as
        ``system/hydro_production_models.json``.
    """
    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    production_models: list[dict] = []
    for newave_code in confhd_codes:
        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            continue

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
    return {
        "$schema": _PRODUCTION_MODELS_SCHEMA_URL,
        "production_models": production_models,
    }


def _total_study_stages(nw_files: NewaveFiles) -> int:
    """Return the total number of stages in the study (including post-study)."""
    dger = Dger.read(str(nw_files.dger))
    start_month: int = int(dger.mes_inicio_estudo)
    num_anos: int = int(dger.num_anos_estudo or 0)
    num_anos_pos: int = int(dger.num_anos_pos_estudo or 0)
    study_months = (13 - start_month) + (num_anos - 1) * 12
    return study_months + num_anos_pos * 12


def _compute_integrated_productivity(
    hreg: pd.Series,
    *,
    canal_fuga_override: float | None = None,
    cmont_override: float | None = None,
) -> float:
    """ρ_esp × ((1/useful) × ∫_vmin^vmax h(V) dV − cf − perdas).

    Mirrors NEWAVE's ``produtibilidade_equivalente_volmin_volmax``: the
    productivity averaged over the full useful storage range, used by
    NEWAVE to convert reservoir volume to stored energy (EARM) and to
    evaluate VminOP constraints.  This is different from the point
    productivity at v_65 that ``_compute_productivity`` returns and that
    the LP uses as the gen = ρ·Q coefficient.

    For a polynomial ``h(V) = a0 + a1·V + ... + a4·V⁴`` the integral has
    a closed form: ``F(V) = a0·V + a1·V²/2 + a2·V³/3 + a3·V⁴/4 + a4·V⁵/5``.
    With ``cmont_override`` the upstream level is held constant, so the
    integrated drop collapses to ``cmont − cf``.

    Run-of-river plants (vmax == vmin) evaluate the polynomial at the
    single operating point — equivalent to the point productivity.
    """
    if cmont_override is not None:
        cf = (
            canal_fuga_override
            if canal_fuga_override is not None
            else float(hreg["canal_fuga_medio"])
        )
        net_drop = cmont_override - cf
    else:
        coeffs = [float(hreg[f"a{i}_volume_cota"]) for i in range(5)]
        if all(c == 0.0 for c in coeffs):
            _LOG.warning(
                "All volume_cota coefficients are zero for plant; "
                "returning zero integrated productivity.",
                extra={"plant": hreg.get("nome_usina", "unknown")},
            )
            return 0.0

        vmin = float(hreg["volume_minimo"])
        vmax = float(hreg["volume_maximo"])
        cf = (
            canal_fuga_override
            if canal_fuga_override is not None
            else float(hreg["canal_fuga_medio"])
        )

        if vmax - vmin <= 0.0:
            # Run-of-river: integrate over the singleton {vmin}.
            avg_h = sum(coeffs[i] * vmin**i for i in range(5))
        else:

            def _antideriv(v: float) -> float:
                return (
                    coeffs[0] * v
                    + coeffs[1] * v**2 / 2.0
                    + coeffs[2] * v**3 / 3.0
                    + coeffs[3] * v**4 / 4.0
                    + coeffs[4] * v**5 / 5.0
                )

            avg_h = (_antideriv(vmax) - _antideriv(vmin)) / (vmax - vmin)

        net_drop = avg_h - cf

    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    if tipo_perda == 1:
        adjusted_drop = net_drop * (1.0 - perdas / 100.0)
    elif tipo_perda == 2:
        adjusted_drop = net_drop - perdas
    else:
        adjusted_drop = net_drop

    return float(hreg["produtibilidade_especifica"]) * adjusted_drop


def _per_stage_integrated_productivities(
    hreg: pd.Series,
    base_integrated: float,
    drop_overrides: list[dict],
    nw_files: NewaveFiles,
    total_stages: int,
) -> list[float]:
    """Per-stage integrated productivity with CFUGA/CMONT step-function awareness.

    Same forward-sweep shape as :func:`_per_stage_productivities` but
    recomputes the *integrated* productivity (volmin_volmax average) at
    each stage where canal_fuga or cmont state changes.  Stages with no
    active override return *base_integrated*.
    """
    if not drop_overrides:
        return [base_integrated] * total_stages

    dger = Dger.read(str(nw_files.dger))
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)

    events_by_stage: dict[int, list[tuple[float | None, float | None]]] = {}
    for override in drop_overrides:
        stage_id = (override["year"] - start_year) * 12 + (
            override["month"] - start_month
        )
        if override["type"] == "CFUGA":
            events_by_stage.setdefault(stage_id, []).append(
                (float(override["value"]), None)
            )
        else:  # CMONT
            events_by_stage.setdefault(stage_id, []).append(
                (None, float(override["value"]))
            )

    values: list[float] = []
    active_cfuga: float | None = None
    active_cmont: float | None = None
    for stage_id in range(total_stages):
        if stage_id == 0:
            for past_stage in sorted(s for s in events_by_stage if s <= 0):
                for cfuga_val, cmont_val in events_by_stage[past_stage]:
                    if cfuga_val is not None:
                        active_cfuga = cfuga_val
                    if cmont_val is not None:
                        active_cmont = cmont_val
        if stage_id in events_by_stage and stage_id > 0:
            for cfuga_val, cmont_val in events_by_stage[stage_id]:
                if cfuga_val is not None:
                    active_cfuga = cfuga_val
                if cmont_val is not None:
                    active_cmont = cmont_val

        if active_cfuga is None and active_cmont is None:
            values.append(base_integrated)
        else:
            values.append(
                _compute_integrated_productivity(
                    hreg,
                    canal_fuga_override=active_cfuga,
                    cmont_override=active_cmont,
                )
            )
    return values


def compute_per_stage_own_integrated_productivities(
    nw_files: NewaveFiles,
) -> dict[int, list[float]]:
    """Return ``{plant_code: [own integrated ρ per stage]}`` for every existing plant.

    Companion to :func:`compute_per_stage_own_productivities` but with the
    EARM convention: ρ is the volume-integrated productivity (matching
    NEWAVE's ``produtibilidade_equivalente_volmin_volmax``), not the
    point productivity at v_65.  Used by VminOP to override the
    ``rho_acum_h{id}`` scalar parameter so the constraint coefficient
    matches NEWAVE's stored-energy accounting rather than the LP's
    gen = ρ·Q point coefficient.

    CFUGA/CMONT temporal overrides shift the integrand at every stage
    from the override's effective stage forward; FICT-cascade contribution
    is folded into the upstream real plant's own value so cascade
    traversal in NEWAVE-code space matches the rewired ``downstream_id``
    in ``hydros.json``.
    """
    total_stages = _total_study_stages(nw_files)
    if total_stages <= 0:
        return {}

    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = _apply_permanent_overrides(hidr.cadastro, nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)
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
        base = _compute_integrated_productivity(hreg)
        resolution = fict_cascade.get(plant_code)
        fict_extra = resolution.fict_rho_sum if resolution is not None else 0.0
        overrides = plants_with_drop_overrides.get(plant_code, [])
        per_stage = _per_stage_integrated_productivities(
            hreg, base, overrides, nw_files, total_stages
        )
        result[plant_code] = [v + fict_extra for v in per_stage]
    return result


def _read_volref_saz(nw_files: NewaveFiles) -> dict[int, dict[int, float]]:
    """Read ``volref_saz.dat`` into ``{plant_code: {calendar_month: useful_vol_hm3}}``.

    NEWAVE uses two distinct conventions inside this file:

    - **Row of all-zeros** — sentinel meaning "no seasonal reference for this
      plant"; NEWAVE falls back to its altura_65 / volume_referencia default.
      We mirror this by *excluding* the plant from the returned mapping.
    - **Row with at least one non-zero value** — real seasonal reference.
      Individual zero months in such a row mean "operate at exactly
      ``volume_minimo``" (useful = 0 above the dead-storage minimum), so we
      keep all twelve monthly entries including explicit zeros.

    Returns an empty dict when ``volref_saz.dat`` is absent.
    """
    if nw_files.volref_saz is None:
        _LOG.debug("volref_saz.dat not found; seasonal productivity disabled.")
        return {}

    vs = VolrefSaz.read(str(nw_files.volref_saz))
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


def _per_stage_productivities(
    hreg: pd.Series,
    base_productivity: float,
    drop_overrides: list[dict],
    nw_files: NewaveFiles,
    total_stages: int,
    seasonal_volref_by_month: dict[int, float] | None = None,
) -> list[float]:
    """Build per-stage productivity values from seasonal volref + CFUGA/CMONT.

    For each stage *s* (0-based), the productivity is computed as:

    1. Determine calendar month ``m = ((start_month − 1 + s) mod 12) + 1``.
    2. Pick reference useful volume: ``seasonal_volref_by_month[m]`` when
       present and positive, else fall back to NEWAVE's altura_65 /
       volume_referencia convention (i.e. use *base_productivity*).
    3. Apply any active CFUGA/CMONT temporal override (step-function from its
       stage of effect forward until the next event of the same type).

    Returns a list of length *total_stages*.  If neither seasonal nor temporal
    overrides apply for a stage, that stage's value equals *base_productivity*.
    """
    has_seasonal = bool(seasonal_volref_by_month)
    if not drop_overrides and not has_seasonal:
        return [base_productivity] * total_stages

    dger = Dger.read(str(nw_files.dger))
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)

    # Group CFUGA/CMONT events by stage_id so per-stage state can be evolved
    # in a single forward sweep.
    events_by_stage: dict[int, list[tuple[float | None, float | None]]] = {}
    for override in drop_overrides:
        stage_id = (override["year"] - start_year) * 12 + (
            override["month"] - start_month
        )
        if override["type"] == "CFUGA":
            events_by_stage.setdefault(stage_id, []).append(
                (float(override["value"]), None)
            )
        else:  # CMONT
            events_by_stage.setdefault(stage_id, []).append(
                (None, float(override["value"]))
            )

    seasonal = seasonal_volref_by_month or {}

    values: list[float] = []
    active_cfuga: float | None = None
    active_cmont: float | None = None
    for stage_id in range(total_stages):
        # Apply events whose effective stage is exactly this stage; events with
        # negative stage_id (took effect before the study horizon) are folded
        # in by walking events with stage_id <= 0 at stage_id == 0.
        if stage_id == 0:
            for past_stage in sorted(s for s in events_by_stage if s <= 0):
                for cfuga_val, cmont_val in events_by_stage[past_stage]:
                    if cfuga_val is not None:
                        active_cfuga = cfuga_val
                    if cmont_val is not None:
                        active_cmont = cmont_val
        if stage_id in events_by_stage and stage_id > 0:
            for cfuga_val, cmont_val in events_by_stage[stage_id]:
                if cfuga_val is not None:
                    active_cfuga = cfuga_val
                if cmont_val is not None:
                    active_cmont = cmont_val

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
    return values


def convert_hydro_energy_productivity(
    nw_files: NewaveFiles, id_map: NewaveIdMap
) -> pa.Table:
    """Build the per-(hydro, stage) ρ_eq override parquet table.

    After cobre's productivity-resolution-rules plan, every non-FPHA
    ``(hydro, stage)`` pair must be supplied by exactly one source. Since
    `convert_production_models` no longer emits a numeric productivity, this
    parquet must cover **every** (hydro, stage).

    Layout:

    - Plants without CFUGA/CMONT overrides emit a single row with
      ``stage_id = NULL``: the per-hydro default. Resolution falls through to
      that default for every stage.
    - Plants with CFUGA/CMONT overrides emit one row per study stage so that
      cobre's per-stage lookup always finds an exact match (no ambiguity from
      mixing default + per-stage rows).

    Other override columns (``reference_volume_hm3``, ``reference_outflow_m3s``,
    ``specific_productivity_mw_per_m3s_per_m``) are left NULL — NEWAVE does not
    provide per-stage values for them.
    """
    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = _apply_permanent_overrides(hidr.cadastro, nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    seasonal_volref = _read_volref_saz(nw_files)

    needs_per_stage = bool(plants_with_drop_overrides) or bool(seasonal_volref)
    total_stages = _total_study_stages(nw_files) if needs_per_stage else 0

    # FICT-cascade: when a real plant's energy-cascade traverses fictitious
    # plants, fold those FICTs' ρ_eq into the upstream real plant's own ρ_eq
    # so that cobre's per-plant cascade sum (computed at solve time from
    # ``hydro_energy_productivity.parquet`` plus the rewired ``downstream_id``)
    # reproduces NEWAVE's ``produtibilidade_acumulada_calculo_earm``.  In
    # NEWAVE's bundled cases FICT plants have ρ_esp = 0 so this is a no-op
    # numerically; the fix is purely structural.  The helper is robust to
    # non-zero FICT productivities (uncommon but possible).
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
                nw_files,
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
            "reference_volume_hm3": pa.array(nulls, type=pa.float64()),
            "reference_outflow_m3s": pa.array(nulls, type=pa.float64()),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                nulls, type=pa.float64()
            ),
        }
    )


def compute_per_stage_own_productivities(
    nw_files: NewaveFiles,
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

    Keys are NEWAVE plant codes (not Cobre ids) since cascade traversal in
    ``compute_accumulated_productivities`` works in NEWAVE-code space.
    """
    total_stages = _total_study_stages(nw_files)
    if total_stages <= 0:
        return {}

    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = _apply_permanent_overrides(hidr.cadastro, nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    seasonal_volref = _read_volref_saz(nw_files)

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
            nw_files,
            total_stages,
            seasonal_volref_by_month=plant_seasonal,
        )
        result[plant_code] = [v + fict_extra for v in per_stage]
    return result


def compute_base_productivities(
    nw_files: NewaveFiles, id_map: NewaveIdMap
) -> dict[int, float]:
    """Return ``{hydro_id: base_productivity_mw_per_m3s}`` for every hydro.

    The base productivity is the value `_compute_productivity` returns with no
    CFUGA/CMONT overrides applied — i.e. the productivity used when the case
    has no temporal overrides for that plant. Consumers that previously read
    ``hydros_dict[i]["generation"]["productivity_mw_per_m3s"]`` should call
    this instead now that productivity has moved out of `hydros.json`.
    """
    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = _apply_permanent_overrides(hidr.cadastro, nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]

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

        def _eval_poly(coeffs: list[float], x: np.ndarray) -> np.ndarray:
            """Evaluate a 4th-degree polynomial: c0 + c1*x + ... + c4*x^4."""
            return (
                coeffs[0]
                + coeffs[1] * x
                + coeffs[2] * x**2
                + coeffs[3] * x**3
                + coeffs[4] * x**4
            )

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


def convert_water_withdrawal(
    nw_files: NewaveFiles, id_map: NewaveIdMap
) -> pa.Table | None:
    """Convert NEWAVE water withdrawal data to a hydro_bounds Parquet table.

    Reads ``dsvagua.dat`` (optional) from *nw_files* and produces a
    ``pa.Table`` with columns ``(hydro_id: INT32, stage_id: INT32,
    water_withdrawal_m3s: DOUBLE)`` suitable for writing to
    ``constraints/hydro_bounds.parquet``.

    The ``codigo_usina`` field in ``dsvagua.dat`` is a NEWAVE 1-based
    *plant* code (matching ``confhd``), not a posto. Each plant may
    contribute multiple rows per stage (one per consumptive-use or
    remaining-flow component) which are summed before the sign is
    negated to convert NEWAVE's "withdrawal = negative valor" convention
    into Cobre's positive ``water_withdrawal_m3s``.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
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
    from inewave.newave import (  # local import to avoid hard dependency at module load
        Dger as _Dger,
    )
    from inewave.newave import (
        Dsvagua as _Dsvagua,
    )

    dsvagua_path = nw_files.dsvagua
    if dsvagua_path is None:
        _LOG.debug("dsvagua.dat not found; no water withdrawal.")
        return None

    # Read dger upfront so the ``outros_usos_da_agua`` switch can
    # short-circuit before any dsvagua I/O. NEWAVE treats 0 as "ignore
    # dsvagua.dat" — mirror that here so Cobre's hydro_bounds match
    # NEWAVE's actual run instead of the file contents.
    dger = _Dger.read(str(nw_files.dger))
    if int(getattr(dger, "outros_usos_da_agua", 1) or 0) == 0:
        _LOG.info("dger.outros_usos_da_agua == 0; skipping dsvagua.dat conversion.")
        return None

    dsvagua = _Dsvagua.read(str(dsvagua_path))
    df = dsvagua.desvios
    if df is None or df.empty:
        return None

    start_year: int = int(dger.ano_inicio_estudo)
    start_month: int = int(dger.mes_inicio_estudo)
    num_anos: int = int(dger.num_anos_estudo or 1)
    num_study_stages: int = (13 - start_month) + (num_anos - 1) * 12
    _pos = dger.num_anos_pos_estudo
    num_post_study_stages: int = (
        int(_pos) * 12 if isinstance(_pos, (int, float)) and _pos else 0
    )
    num_total_stages: int = num_study_stages + num_post_study_stages

    # Group by (codigo_usina, data) and sum valor.
    grouped = df.groupby(["codigo_usina", "data"], as_index=False)["valor"].sum()

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    values: list[float] = []

    for _, row in grouped.iterrows():
        hydro_code = int(row["codigo_usina"])
        try:
            hydro_id = id_map.hydro_id(hydro_code)
        except KeyError:
            # Fictitious plants and any other entries the id_map filters
            # out are silently dropped — dsvagua frequently carries codes
            # outside the dispatchable hydro fleet.
            continue

        dt = row["data"]
        stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
        if stage_id < 0 or stage_id >= num_study_stages:
            continue

        # Negate: NEWAVE negative valor = withdrawal; Cobre positive = withdrawal.
        withdrawal = -float(row["valor"])
        hydro_ids.append(hydro_id)
        stage_ids.append(stage_id)
        values.append(withdrawal)

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
            for s in range(0, num_total_stages):
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
    nw_files: NewaveFiles,
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

    For post-study stages, the last study year's seasonal pattern is repeated.

    Returns ``None`` if MODIF.DAT is absent or contains no relevant records.
    """
    from inewave.newave import Confhd as _Confhd
    from inewave.newave import Dger as _Dger

    modif_path = nw_files.modif
    if modif_path is None:
        return None

    dger = _Dger.read(str(nw_files.dger))
    start_year: int = int(dger.ano_inicio_estudo)
    start_month: int = int(dger.mes_inicio_estudo)
    num_anos: int = int(dger.num_anos_estudo or 1)
    num_anos_pos: int = int(dger.num_anos_pos_estudo or 0)
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    # Read hidr.dat with permanent overrides for vol_min/vol_max.
    cadastro = read_cadastro(nw_files)

    # Read confhd for the list of active plant codes.
    confhd = _Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]
    confhd_codes = [int(r["codigo_usina"]) for _, r in non_fict.iterrows()]

    # Extract temporal overrides.
    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)

    # NEWAVE big-M sentinel: 99999 means "no limit" (restore default).
    _BIG_M = 99990.0

    def _build_step_function(
        recs: list[dict],
        transform: Callable[[float], float],
    ) -> dict[int, float]:
        """Build a step-function from override records.

        Each record sets the value from its stage onward until the next
        record overrides it.  Raw values >= 99990 (big-M) mean "restore
        default" and clear the forward-fill.
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

        result: dict[int, float] = {}
        cp_idx = 0
        current: float | None = None
        first_stage = changepoints[0][0]

        for stage_id in range(first_stage, study_months):
            while cp_idx < len(changepoints) and changepoints[cp_idx][0] <= stage_id:
                raw = changepoints[cp_idx][1]
                current = None if raw >= _BIG_M else transform(raw)
                cp_idx += 1
            if current is not None:
                result[stage_id] = current

        seasonal: dict[int, float] = {}
        for stage_id in range(max(0, study_months - 12), study_months):
            if stage_id in result:
                cal = ((start_month - 1 + stage_id) % 12) + 1
                seasonal[cal] = result[stage_id]

        for stage_id in range(study_months, total_stages):
            cal = ((start_month - 1 + stage_id) % 12) + 1
            if cal in seasonal:
                result[stage_id] = seasonal[cal]

        return result

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    min_storage_vals: list[float | None] = []
    max_storage_vals: list[float | None] = []
    min_turbined_vals: list[float | None] = []
    max_turbined_vals: list[float | None] = []
    min_outflow_vals: list[float | None] = []

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

        # Storage bounds (percentage -> hm³).
        vmaxt_by_stage: dict[int, float] = {}
        vmint_by_stage: dict[int, float] = {}
        if useful > 0:
            vmaxt_by_stage = _build_step_function(vmaxt, _pct_to_hm3)
            vmint_by_stage = _build_step_function(vmint, _pct_to_hm3)

        # Turbined bounds (absolute m³/s).
        turbmaxt_by_stage = _build_step_function(turbmaxt, _identity)
        turbmint_by_stage = _build_step_function(turbmint, _identity)

        # Outflow bounds (absolute m³/s).
        vazmint_by_stage = _build_step_function(vazmint, _identity)

        all_stages = sorted(
            set(vmaxt_by_stage)
            | set(vmint_by_stage)
            | set(turbmaxt_by_stage)
            | set(turbmint_by_stage)
            | set(vazmint_by_stage)
        )
        for stage_id in all_stages:
            hydro_ids.append(hydro_id)
            stage_ids.append(stage_id)
            max_storage_vals.append(vmaxt_by_stage.get(stage_id))
            min_storage_vals.append(vmint_by_stage.get(stage_id))
            max_turbined_vals.append(turbmaxt_by_stage.get(stage_id))
            min_turbined_vals.append(turbmint_by_stage.get(stage_id))
            min_outflow_vals.append(vazmint_by_stage.get(stage_id))

    if not hydro_ids:
        return None

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "min_storage_hm3": pa.array(min_storage_vals, type=pa.float64()),
            "max_storage_hm3": pa.array(max_storage_vals, type=pa.float64()),
            "min_turbined_m3s": pa.array(min_turbined_vals, type=pa.float64()),
            "max_turbined_m3s": pa.array(max_turbined_vals, type=pa.float64()),
            "min_outflow_m3s": pa.array(min_outflow_vals, type=pa.float64()),
        }
    ).sort_by([("hydro_id", "ascending"), ("stage_id", "ascending")])


# ---------------------------------------------------------------------------
# Hardcoded diversion: PIMENTAL → BELO MONTE
# ---------------------------------------------------------------------------
# The Belo Monte complex has two powerhouses sharing a reservoir. PIMENTAL
# (NEWAVE code 314) is the complementary powerhouse at the dam site; BELO
# MONTE (code 288) is the main powerhouse connected by a diversion canal.
# NEWAVE splits the Xingu river inflow between the two postos (302 / 292)
# but does not model an explicit diversion. We add it here so that cobre
# can route excess water from PIMENTAL to BELO MONTE instead of spilling.

_PIMENTAL_NEWAVE_CODE = 314
_BELO_MONTE_NEWAVE_CODE = 288
_PIMENTAL_DIVERSION_MAX_M3S = 13_900.0


def _make_diversion(newave_code: int, id_map: NewaveIdMap) -> dict | None:
    """Return a diversion dict for PIMENTAL, ``None`` for all other plants."""
    if newave_code != _PIMENTAL_NEWAVE_CODE:
        return None
    try:
        bm_id = id_map.hydro_id(_BELO_MONTE_NEWAVE_CODE)
    except KeyError:
        return None
    return {
        "downstream_id": bm_id,
        "max_flow_m3s": _PIMENTAL_DIVERSION_MAX_M3S,
    }


def _is_na(value: object) -> bool:
    """Return True if *value* is a pandas NA/NaN sentinel."""
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return pd.isna(value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return False
