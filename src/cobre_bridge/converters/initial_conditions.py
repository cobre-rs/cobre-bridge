"""Initial conditions converter: maps NEWAVE initial storage to Cobre JSON."""

from __future__ import annotations

import logging

from inewave.newave import Confhd, Hidr

from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/initial_conditions.schema.json"
)


def convert_initial_conditions(nw_files: NewaveFiles, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE initial reservoir storage to a Cobre initial_conditions dict.

    Reads ``hidr.dat`` and ``confhd.dat`` from *nw_files*.  Initial
    storage is derived from ``Confhd.usinas.volume_inicial_percentual``
    (a percentage of ``volume_maximo`` from Hidr).

    Values outside ``[0, 100]`` are clamped with a warning.

    Parameters
    ----------
    nw_files:
        Resolved NEWAVE file paths for the case.
    id_map:
        Pre-built ID mapping for hydro IDs.

    Raises
    ------
    ValueError
        If a hydro in ``confhd.dat`` references a code absent in
        ``hidr.dat``.
    """
    hidr = Hidr.read(str(nw_files.hidr))
    confhd = Confhd.read(str(nw_files.confhd))

    cadastro = hidr.cadastro
    confhd_df = confhd.usinas

    # Filter to existing, non-fictitious plants — same criterion as hydro.py.
    all_existing = confhd_df[confhd_df["usina_existente"] == "EX"]
    existing = all_existing[
        ~all_existing["nome_usina"].str.strip().str.startswith("FICT.")
    ]

    storage: list[dict] = []
    for _, row in existing.iterrows():
        newave_code = int(row["codigo_usina"])
        name = str(row["nome_usina"]).strip()

        if newave_code not in cadastro.index:
            raise ValueError(
                f"Hydro plant '{name}' (code {newave_code}) from confhd.dat"
                f" not found in hidr.dat"
            )

        hreg = cadastro.loc[newave_code]
        vol_min = float(hreg["volume_minimo"])
        vol_max = float(hreg["volume_maximo"])

        pct = float(row["volume_inicial_percentual"])
        if pct < 0.0 or pct > 100.0:
            _LOG.warning(
                "volume_inicial_percentual for plant '%s' (code %d) is %.2f"
                " — clamping to [0, 100]",
                name,
                newave_code,
                pct,
            )
            pct = max(0.0, min(100.0, pct))

        # Percentage is of useful volume (max - min), not of max.
        value_hm3 = (pct / 100.0) * (vol_max - vol_min) + vol_min

        storage.append(
            {
                "hydro_id": id_map.hydro_id(newave_code),
                "value_hm3": value_hm3,
            }
        )

    storage.sort(key=lambda s: s["hydro_id"])

    return {
        "$schema": _SCHEMA_URL,
        "storage": storage,
        "filling_storage": [],
    }
