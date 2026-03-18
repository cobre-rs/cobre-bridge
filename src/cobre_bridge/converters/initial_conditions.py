"""Initial conditions converter: maps NEWAVE initial storage to Cobre JSON."""

from __future__ import annotations

import logging
from pathlib import Path

from inewave.newave import Confhd, Hidr

from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/initial_conditions.schema.json"
)


def convert_initial_conditions(newave_dir: Path, id_map: NewaveIdMap) -> dict:
    """Convert NEWAVE initial reservoir storage to a Cobre initial_conditions dict.

    Reads ``hidr.dat`` and ``confhd.dat`` from *newave_dir*.  Initial
    storage is derived from ``Confhd.usinas.volume_inicial_percentual``
    (a percentage of ``volume_maximo`` from Hidr).

    Values outside ``[0, 100]`` are clamped with a warning.

    Parameters
    ----------
    newave_dir:
        Path to the NEWAVE case directory.
    id_map:
        Pre-built ID mapping for hydro IDs.

    Raises
    ------
    FileNotFoundError
        If ``hidr.dat`` or ``confhd.dat`` is absent.
    ValueError
        If a hydro in ``confhd.dat`` references a code absent in
        ``hidr.dat``.
    """
    hidr_path = newave_dir / "hidr.dat"
    confhd_path = newave_dir / "confhd.dat"

    for p in (hidr_path, confhd_path):
        if not p.exists():
            raise FileNotFoundError(f"Required NEWAVE file not found: {p}")

    hidr = Hidr.read(str(hidr_path))
    confhd = Confhd.read(str(confhd_path))

    cadastro = hidr.cadastro
    confhd_df = confhd.usinas

    # Filter to existing plants only — same criterion as hydro.py.
    existing = confhd_df[confhd_df["usina_existente"] == "EX"]

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

        value_hm3 = (pct / 100.0) * vol_max

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
