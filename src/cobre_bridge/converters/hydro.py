"""Hydro entity converter: maps NEWAVE hydro plant data to Cobre hydro JSON."""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
import pyarrow as pa
from inewave.newave import Confhd, Dger, Ghmin, Hidr, Modif, Penalid, Ree

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/hydros.schema.json"
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
                result.loc[code, f"maquinas_conjunto_{set_num}"] = int(
                    rec.numero_maquinas
                )

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
    "DESVIO": "spillage_cost",
    "VAZMIN": "outflow_violation_below_cost",
    "VAZMAX": "outflow_violation_above_cost",
    "GHMIN": "generation_violation_below_cost",
    "TURBMN": "turbined_violation_below_cost",
    # TURBMX has no direct Cobre mapping — intentionally excluded.
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

    # Collect study plant codes for temporal override extraction.
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    # Extract temporal overrides (for reference / future use).
    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)

    # Read GHMIN.DAT minimum generation map.
    ghmin_map = _read_ghmin(nw_files)

    # Read PENALID.DAT per-REE penalty overrides.
    penalid_map = _read_penalid(nw_files)

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

        # Generation parameters.
        productivity = _compute_productivity(hreg)
        n_sets = int(hreg["numero_conjuntos_maquinas"])

        max_turbined = 0.0
        max_generation = 0.0
        for i in range(1, n_sets + 1):
            n_machines = int(hreg[f"maquinas_conjunto_{i}"])
            q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
            p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
            max_turbined += q_nominal * n_machines
            max_generation += p_nominal * n_machines

        # Apply TEIF/IP availability derating to max_generation only.
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

        # Apply VAZMINT temporal override: use the latest effective value at or
        # before study start (file order matters; last wins for the same date).
        plant_temps = temporal_overrides.get(newave_code, [])
        vazmint_overrides = [o for o in plant_temps if o["type"] == "VAZMINT"]
        if vazmint_overrides:
            # Use the value from the last VAZMINT record (highest effective date
            # at or before study start — we use last in file order as a proxy
            # since we cannot determine study start without additional context).
            min_outflow = vazmint_overrides[-1]["value"]

        # CFUGA/CMONT temporal overrides are handled by convert_production_models;
        # no warning needed here as per-stage productivity is now supported.

        # Min generation: use GHMIN.DAT if available, otherwise approximate.
        ghmin_value = ghmin_map.get(newave_code)
        if ghmin_value is not None:
            min_generation = ghmin_value
        else:
            min_generation = 0.0

        # Downstream cascade linkage.
        downstream_id: int | None = None
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

        # Hydraulic loss model derived from tipo_perda / perdas columns.
        tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
        perdas_val = float(hreg.get("perdas", 0.0) or 0.0)
        if tipo_perda == 1 and perdas_val > 0 and not math.isnan(perdas_val):
            hydraulic_losses: dict | None = {"type": "factor", "value": perdas_val}
        elif tipo_perda == 2 and perdas_val > 0 and not math.isnan(perdas_val):
            hydraulic_losses = {"type": "constant", "value_m": perdas_val}
        else:
            hydraulic_losses = None

        # Penalty overrides from PENALID.DAT — look up by the plant's REE code.
        ree_penalties = penalid_map.get(ree_code)
        penalties: dict | None = ree_penalties if ree_penalties else None

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
                "productivity_mw_per_m3s": productivity,
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": min_generation,
                "max_generation_mw": max_generation,
            },
            "evaporation": (
                {"coefficients_mm": evap_coeffs} if has_evaporation else None
            ),
            "tailrace": None,
            "diversion": None,
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
) -> float:
    """Compute average productivity in MW/(m^3/s) for a hydro plant.

    Reads polynomial coefficients ``a0_volume_cota`` through
    ``a4_volume_cota`` from the plant's cadastro row to map storage volume
    (hm3) to upstream height (m).  Subtracts the tailrace level to obtain
    gross drop, applies the loss model defined by ``tipo_perda`` and
    ``perdas``, then multiplies by ``produtibilidade_especifica``.

    For monthly-regulated plants (``tipo_regulacao == "M"``) the height
    is the integral average of the polynomial over
    ``[volume_minimo, volume_maximo]``.  For all other plant types the
    polynomial is evaluated at ``volume_referencia``.

    Parameters
    ----------
    hreg:
        One row of ``Hidr.cadastro``, indexed by column name.
    canal_fuga_override:
        If provided, replaces ``canal_fuga_medio`` as the tailrace level.
        Used when a CFUGA temporal override is active.
    cmont_override:
        If provided, replaces the polynomial-derived upstream height with
        this fixed value (in metres).  Used when a CMONT temporal override
        is active.

    Returns
    -------
    float
        Average productivity in MW/(m^3/s).  Returns zero if all
        polynomial coefficients are zero (no usable head).
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

        def _poly_antiderivative(v: float) -> float:
            """Evaluate the antiderivative F(v) = c0*v + c1*v^2/2 + ..."""
            return (
                coeffs[0] * v
                + coeffs[1] * v**2 / 2.0
                + coeffs[2] * v**3 / 3.0
                + coeffs[3] * v**4 / 4.0
                + coeffs[4] * v**5 / 5.0
            )

        tipo_regulacao = str(hreg["tipo_regulacao"]).strip()
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        if tipo_regulacao == "M":
            if vol_min == vol_max:
                # Degenerate interval: fall back to point evaluation.
                avg_height = _poly(vol_min)
            else:
                avg_height = (
                    _poly_antiderivative(vol_max) - _poly_antiderivative(vol_min)
                ) / (vol_max - vol_min)
            net_drop = avg_height - canal_fuga
        else:
            vol_ref = float(hreg["volume_referencia"])
            net_drop = _poly(vol_ref) - canal_fuga

    # Apply loss model.
    tipo_perda = int(hreg["tipo_perda"])
    perdas = float(hreg["perdas"])
    if tipo_perda == 1:
        # Multiplicative factor: adjusted_drop = net_drop * (1 - perdas)
        adjusted_drop = net_drop * (1.0 - perdas)
    elif tipo_perda == 2:
        # Additive meters: adjusted_drop = net_drop - perdas
        adjusted_drop = net_drop - perdas
    else:
        adjusted_drop = net_drop

    produtibilidade = float(hreg["produtibilidade_especifica"])
    return produtibilidade * adjusted_drop


def convert_production_models(
    nw_files: NewaveFiles, id_map: NewaveIdMap
) -> dict | None:
    """Build per-stage productivity overrides from MODIF.DAT CFUGA/CMONT records.

    For each hydro plant that has CFUGA or CMONT temporal overrides in
    ``MODIF.DAT``, computes the productivity at each change point and emits
    a ``stage_ranges`` entry in the Cobre ``hydro_production_models.json``
    format.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built entity ID map used to translate NEWAVE plant codes to
        0-based Cobre hydro IDs.

    Returns
    -------
    dict | None
        A dict with a ``"production_models"`` key ready to serialise as
        ``system/hydro_production_models.json``, or ``None`` when no plants
        have CFUGA/CMONT overrides.
    """
    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = hidr.cadastro
    # Apply permanent overrides so base productivity is consistent with
    # what convert_hydros computes.
    cadastro = _apply_permanent_overrides(cadastro, nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)

    # Filter to plants that actually have CFUGA or CMONT overrides.
    plants_with_drop_overrides = {
        code: [o for o in overrides if o["type"] in ("CFUGA", "CMONT")]
        for code, overrides in temporal_overrides.items()
        if any(o["type"] in ("CFUGA", "CMONT") for o in overrides)
    }

    if not plants_with_drop_overrides:
        return None

    # Read dger for study start date and total stage count.
    dger = Dger.read(str(nw_files.dger))
    start_year: int = int(dger.ano_inicio_estudo)
    start_month: int = int(dger.mes_inicio_estudo)
    num_anos: int = int(dger.num_anos_estudo or 0)
    num_anos_pos: int = int(dger.num_anos_pos_estudo or 0)
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12

    production_models: list[dict] = []

    for newave_code, drop_overrides in plants_with_drop_overrides.items():
        if newave_code not in cadastro.index:
            _LOG.warning(
                "Plant code %d has CFUGA/CMONT overrides but is not in hidr.dat;"
                " skipping production model.",
                newave_code,
            )
            continue

        try:
            hydro_id = id_map.hydro_id(newave_code)
        except KeyError:
            _LOG.warning(
                "Plant code %d has CFUGA/CMONT overrides but is not in id_map;"
                " skipping production model.",
                newave_code,
            )
            continue

        hreg = cadastro.loc[newave_code]
        base_productivity = _compute_productivity(hreg)

        # Build a sorted list of (stage_id, canal_fuga, cmont) change events.
        # Each event holds the *effective* values from that stage onwards.
        # Start with base values; events are processed in chronological order.
        events: list[tuple[int, float | None, float | None]] = []
        for override in drop_overrides:
            stage_id = (override["year"] - start_year) * 12 + (
                override["month"] - start_month
            )
            if override["type"] == "CFUGA":
                events.append((stage_id, override["value"], None))
            else:  # CMONT
                events.append((stage_id, None, override["value"]))

        # Sort by stage_id, then by CFUGA before CMONT within the same stage
        # so both can be applied when they coincide.
        events.sort(key=lambda e: (e[0], 0 if e[1] is not None else 1))

        # Merge events at the same stage: accumulate effective canal_fuga and
        # cmont by scanning in order.
        # Use sentinel None to mean "use base value from hreg".
        effective_cfuga: float | None = None
        effective_cmont: float | None = None

        # Build a sorted list of (stage_id, productivity) breakpoints by
        # replaying the events against the running effective state.
        breakpoints: list[tuple[int, float]] = []

        i = 0
        while i < len(events):
            stage_id = events[i][0]
            # Apply all events at this stage_id.
            while i < len(events) and events[i][0] == stage_id:
                _, cfuga_val, cmont_val = events[i]
                if cfuga_val is not None:
                    effective_cfuga = cfuga_val
                if cmont_val is not None:
                    effective_cmont = cmont_val
                i += 1

            prod = _compute_productivity(
                hreg,
                canal_fuga_override=effective_cfuga,
                cmont_override=effective_cmont,
            )
            breakpoints.append((stage_id, prod))

        # Build stage_ranges from breakpoints.
        # The implicit range before the first breakpoint uses base_productivity.
        stage_ranges: list[dict] = []

        first_stage = breakpoints[0][0]
        if first_stage > 0:
            stage_ranges.append(
                {
                    "start_stage_id": 0,
                    "end_stage_id": first_stage - 1,
                    "model": "constant_productivity",
                    "productivity_override": base_productivity,
                }
            )

        for idx, (bp_stage, bp_prod) in enumerate(breakpoints):
            if idx + 1 < len(breakpoints):
                next_stage = breakpoints[idx + 1][0]
                end_stage: int | None = next_stage - 1
            else:
                end_stage = None  # until end of study

            stage_ranges.append(
                {
                    "start_stage_id": bp_stage,
                    "end_stage_id": end_stage,
                    "model": "constant_productivity",
                    "productivity_override": bp_prod,
                }
            )

        _LOG.debug(
            "Plant code %d (hydro_id=%d): %d stage range(s) in production model,"
            " total_stages=%d",
            newave_code,
            hydro_id,
            len(stage_ranges),
            total_stages,
        )

        production_models.append(
            {
                "hydro_id": hydro_id,
                "selection_mode": "stage_ranges",
                "stage_ranges": stage_ranges,
            }
        )

    if not production_models:
        return None

    production_models.sort(key=lambda m: m["hydro_id"])
    return {"production_models": production_models}


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

    The ``codigo_usina`` field in ``dsvagua.dat`` is a **posto** (gauging
    station index), not a plant code.  This function reads ``confhd.dat`` to
    build the posto -> hydro_code mapping, then converts to 0-based Cobre IDs
    via *id_map*.

    NEWAVE stores withdrawal as a negative ``valor``; Cobre expects a positive
    ``water_withdrawal_m3s``.  The sign is negated during conversion.

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
        Confhd as _Confhd,
    )
    from inewave.newave import (
        Dger as _Dger,
    )
    from inewave.newave import (
        Dsvagua as _Dsvagua,
    )

    dsvagua_path = nw_files.dsvagua
    if dsvagua_path is None:
        _LOG.debug("dsvagua.dat not found; no water withdrawal.")
        return None

    dsvagua = _Dsvagua.read(str(dsvagua_path))
    df = dsvagua.desvios
    if df is None or df.empty:
        return None

    # Read confhd for posto -> hydro_code mapping.
    confhd = _Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    posto_to_code: dict[int, int] = {}
    for _, row in confhd_df.iterrows():
        posto_to_code[int(row["posto"])] = int(row["codigo_usina"])

    # Read dger for study start date and duration.
    dger = _Dger.read(str(nw_files.dger))
    start_year: int = int(dger.ano_inicio_estudo)
    start_month: int = int(dger.mes_inicio_estudo)
    num_stages: int = int(dger.num_anos_estudo or 0) * 12

    # Group by (codigo_usina, data) and sum valor.
    grouped = df.groupby(["codigo_usina", "data"], as_index=False)["valor"].sum()

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    values: list[float] = []

    for _, row in grouped.iterrows():
        posto = int(row["codigo_usina"])
        hydro_code = posto_to_code.get(posto)
        if hydro_code is None:
            _LOG.warning(
                "Posto %d in dsvagua.dat not found in confhd.dat; skipping.",
                posto,
            )
            continue

        try:
            hydro_id = id_map.hydro_id(hydro_code)
        except KeyError:
            _LOG.warning(
                "Hydro code %d (posto %d) not in id_map; skipping.",
                hydro_code,
                posto,
            )
            continue

        dt = row["data"]
        stage_id = (dt.year - start_year) * 12 + (dt.month - start_month)
        if stage_id < 0 or stage_id >= num_stages:
            _LOG.warning(
                "Stage %d for posto %d out of range [0, %d); skipping.",
                stage_id,
                posto,
                num_stages,
            )
            continue

        # Negate: NEWAVE negative valor = withdrawal; Cobre positive = withdrawal.
        withdrawal = -float(row["valor"])
        hydro_ids.append(hydro_id)
        stage_ids.append(stage_id)
        values.append(withdrawal)

    if not hydro_ids:
        return None

    table = pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "water_withdrawal_m3s": pa.array(values, type=pa.float64()),
        }
    )
    # Sort by (hydro_id, stage_id) for deterministic output.
    return table.sort_by([("hydro_id", "ascending"), ("stage_id", "ascending")])


def _is_na(value: object) -> bool:
    """Return True if *value* is a pandas NA/NaN sentinel."""
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return pd.isna(value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return False
