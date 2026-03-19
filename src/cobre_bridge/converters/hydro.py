"""Hydro entity converter: maps NEWAVE hydro plant data to Cobre hydro JSON."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd
from inewave.newave import Confhd, Ghmin, Hidr, Modif, Ree

from cobre_bridge.id_map import NewaveIdMap

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


def _find_file_case_insensitive(directory: Path, filename: str) -> Path | None:
    """Return the path for *filename* in *directory*, ignoring case.

    Scans *directory* for any file whose name matches *filename* case-
    insensitively.  Returns ``None`` if no match is found.  If multiple
    entries match (unusual on case-sensitive file systems with differently-
    cased variants), the first match in iteration order is returned.
    """
    lower_target = filename.lower()
    try:
        for entry in directory.iterdir():
            if entry.is_file() and entry.name.lower() == lower_target:
                return entry
    except OSError:
        pass
    return None


def _apply_permanent_overrides(
    cadastro: pd.DataFrame, newave_dir: Path
) -> pd.DataFrame:
    """Apply MODIF.DAT permanent overrides to the hidr.dat cadastro.

    Reads ``MODIF.DAT`` (case-insensitive lookup) from *newave_dir* and
    applies permanent override records — VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
    NUMMAQ — to a *copy* of *cadastro*.  The original DataFrame is not
    mutated.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina``.
    newave_dir:
        Path to the NEWAVE case directory.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with permanent overrides applied.
    """
    modif_path = _find_file_case_insensitive(newave_dir, "modif.dat")
    if modif_path is None:
        _LOG.debug(
            "MODIF.DAT not found in %s; skipping permanent overrides.", newave_dir
        )
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
    newave_dir: Path, confhd_codes: list[int]
) -> dict[int, list[dict]]:
    """Extract MODIF.DAT temporal overrides for plants in *confhd_codes*.

    Reads ``MODIF.DAT`` (case-insensitive) and returns a dict keyed by plant
    code.  Each value is a list of override dicts in file order::

        {"type": str, "month": int, "year": int, "value": float}

    For CFUGA/CMONT the ``"value"`` field is the level in metres.  For
    TURBMINT/TURBMAXT it is the turbined flow in m³/s.  For VAZMINT/VMAXT/
    VMINT it is the volume or flow as stored in the record.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory.
    confhd_codes:
        List of plant codes present in the study (from confhd.dat).  Records
        for plants not in this list are excluded.

    Returns
    -------
    dict[int, list[dict]]
        Temporal override records per plant code.  Empty dict if MODIF.DAT is
        absent.
    """
    modif_path = _find_file_case_insensitive(newave_dir, "modif.dat")
    if modif_path is None:
        _LOG.debug(
            "MODIF.DAT not found in %s; no temporal overrides extracted.", newave_dir
        )
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


def _read_ghmin(newave_dir: Path) -> dict[int, float]:
    """Read GHMIN.DAT and return a mapping of plant code -> minimum generation MW.

    Uses case-insensitive file lookup.  If ``GHMIN.DAT`` does not exist,
    returns an empty dict.  Only ``patamar == 0`` rows (all load blocks) are
    used; if multiple time periods are present for the same plant, the first
    occurrence (earliest date) is taken.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory.

    Returns
    -------
    dict[int, float]
        Plant code -> minimum generation in MW.  Empty dict if file absent.
    """
    ghmin_path = _find_file_case_insensitive(newave_dir, "ghmin.dat")
    if ghmin_path is None:
        _LOG.debug(
            "GHMIN.DAT not found in %s; using computed min_generation fallback.",
            newave_dir,
        )
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


def convert_hydros(newave_dir: Path, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE hydro plant data to a Cobre ``hydros.json`` dict.

    Reads ``hidr.dat``, ``confhd.dat``, and ``ree.dat`` from
    *newave_dir*.  Returns a dict with a ``"hydros"`` key containing a
    list of hydro entries sorted by Cobre 0-based ID.

    Also reads ``MODIF.DAT`` (if present) to apply permanent parameter
    overrides and extract temporal override metadata.  Reads ``GHMIN.DAT``
    (if present) to override computed minimum generation values.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory containing the required files.
    id_map:
        Pre-built ID mapping used for bus and downstream-hydro cross-
        references.

    Raises
    ------
    FileNotFoundError
        If ``hidr.dat``, ``confhd.dat``, or ``ree.dat`` is absent.
    ValueError
        If a hydro in ``confhd.dat`` references a code not found in
        ``hidr.dat``.
    """
    hidr_path = newave_dir / "hidr.dat"
    confhd_path = newave_dir / "confhd.dat"
    ree_path = newave_dir / "ree.dat"

    for p in (hidr_path, confhd_path, ree_path):
        if not p.exists():
            raise FileNotFoundError(f"Required NEWAVE file not found: {p}")

    hidr = Hidr.read(str(hidr_path))
    confhd = Confhd.read(str(confhd_path))
    ree_file = Ree.read(str(ree_path))

    cadastro = hidr.cadastro  # DataFrame indexed by codigo_usina (1-based)
    confhd_df = confhd.usinas
    ree_df = ree_file.rees  # columns: codigo, nome, submercado, ...

    # Apply MODIF.DAT permanent overrides before the main conversion loop.
    cadastro = _apply_permanent_overrides(cadastro, newave_dir)

    # Collect study plant codes for temporal override extraction.
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    confhd_codes = [int(r["codigo_usina"]) for _, r in existing.iterrows()]

    # Extract temporal overrides (for reference / future use).
    temporal_overrides = _extract_temporal_overrides(newave_dir, confhd_codes)

    # Read GHMIN.DAT minimum generation map.
    ghmin_map = _read_ghmin(newave_dir)

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

        # Warn about CFUGA/CMONT temporal overrides on this plant.
        drop_overrides = [o for o in plant_temps if o["type"] in ("CFUGA", "CMONT")]
        if drop_overrides:
            _LOG.warning(
                "Plant '%s' (code %d) has CFUGA/CMONT temporal overrides in"
                " MODIF.DAT, but per-stage productivity is not yet supported;"
                " overrides are ignored for productivity computation.",
                name,
                newave_code,
            )

        # Min generation: use GHMIN.DAT if available, otherwise approximate.
        ghmin_value = ghmin_map.get(newave_code)
        if ghmin_value is not None:
            min_generation = ghmin_value
        else:
            min_generation = min_outflow * productivity

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

        # Evaporation coefficients — 12 monthly values in mm/month.
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
            "penalties": None,
            "entry_stage_id": None,
            "exit_stage_id": None,
        }
        hydros.append(hydro_entry)

    hydros.sort(key=lambda h: h["id"])

    return {
        "$schema": _SCHEMA_URL,
        "hydros": hydros,
    }


def _compute_productivity(hreg: pd.Series) -> float:
    """Compute average productivity in MW/(m^3/s) for a hydro plant.

    Reads polynomial coefficients ``volume_cota_0`` through
    ``volume_cota_4`` from the plant's cadastro row to map storage volume
    (hm3) to upstream height (m).  Subtracts ``canal_fuga_medio`` to
    obtain gross drop, applies the loss model defined by ``tipo_perda``
    and ``perdas``, then multiplies by ``produtibilidade_especifica``.

    For monthly-regulated plants (``tipo_regulacao == "M"``) the height
    is the integral average of the polynomial over
    ``[volume_minimo, volume_maximo]``.  For all other plant types the
    polynomial is evaluated at ``volume_referencia``.

    Parameters
    ----------
    hreg:
        One row of ``Hidr.cadastro``, indexed by column name.

    Returns
    -------
    float
        Average productivity in MW/(m^3/s).  Returns zero if all
        polynomial coefficients are zero (no usable head).
    """
    coeffs = [float(hreg[f"volume_cota_{i}"]) for i in range(5)]

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

    canal_fuga = float(hreg["canal_fuga_medio"])
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


def _is_na(value: object) -> bool:
    """Return True if *value* is a pandas NA/NaN sentinel."""
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return pd.isna(value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return False
