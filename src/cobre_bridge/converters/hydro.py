"""Hydro entity converter: maps NEWAVE hydro plant data to Cobre hydro JSON."""

from __future__ import annotations

import logging
from pathlib import Path

from inewave.newave import Confhd, Hidr, Ree

from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/hydros.schema.json"
)

# Month abbreviations used by inewave in the Hidr cadastro DataFrame.
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


def convert_hydros(newave_dir: Path, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE hydro plant data to a Cobre ``hydros.json`` dict.

    Reads ``hidr.dat``, ``confhd.dat``, and ``ree.dat`` from
    *newave_dir*.  Returns a dict with a ``"hydros"`` key containing a
    list of hydro entries sorted by Cobre 0-based ID.

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

    # Build REE-code -> subsystem-code mapping.
    ree_to_submercado: dict[int, int] = {}
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            ree_to_submercado[int(row["codigo"])] = int(row["submercado"])

    # Filter to existing plants only.
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]

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
        productivity = float(hreg["produtibilidade_especifica"])
        n_sets = int(hreg["numero_conjuntos_maquinas"])

        max_turbined = 0.0
        max_generation = 0.0
        for i in range(1, n_sets + 1):
            n_machines = int(hreg[f"maquinas_conjunto_{i}"])
            q_nominal = float(hreg[f"vazao_nominal_conjunto_{i}"])
            p_nominal = float(hreg[f"potencia_nominal_conjunto_{i}"])
            max_turbined += q_nominal * n_machines
            max_generation += p_nominal * n_machines

        # Minimum turbined flow from historical minimum discharge.
        vazao_min_hist = hreg.get("vazao_minima_historica")
        min_turbined = (
            float(vazao_min_hist)
            if vazao_min_hist and float(vazao_min_hist) > 0
            else 0.0
        )
        min_generation = min_turbined * productivity

        # Downstream cascade linkage.
        jusante_raw = row.get("codigo_usina_jusante")
        if (
            jusante_raw is not None
            and not _is_na(jusante_raw)
            and int(jusante_raw) != 0
        ):
            try:
                downstream_id: int | None = id_map.hydro_id(int(jusante_raw))
            except KeyError:
                downstream_id = None
        else:
            downstream_id = None

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
                "min_outflow_m3s": 0.0,
                "max_outflow_m3s": None,
            },
            "generation": {
                "model": "constant_productivity",
                "productivity_mw_per_m3s": productivity,
                "min_turbined_m3s": min_turbined,
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
            "hydraulic_losses": None,
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
