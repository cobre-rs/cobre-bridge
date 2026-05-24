"""Initial conditions converter: maps NEWAVE initial storage to Cobre JSON."""

from __future__ import annotations

import logging

import pandas as pd
from inewave.newave import Confhd, Hidr

from cobre_bridge.converters.anticipated import read_anticipated_dispatch
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

        # Daily-regulation ('D') plants are frozen at ``volume_referencia``
        # by NEWAVE — the reservoir doesn't accumulate water across stages.
        # Mirror that by anchoring initial storage to the reference volume
        # so it stays consistent with the (collapsed) min/max bounds set in
        # ``hydro.py``.
        tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()
        vol_ref_raw = hreg.get("volume_referencia")
        if tipo_reg == "D" and vol_ref_raw is not None and not pd.isna(vol_ref_raw):
            value_hm3 = float(vol_ref_raw)
            storage.append(
                {
                    "hydro_id": id_map.hydro_id(newave_code),
                    "value_hm3": value_hm3,
                }
            )
            continue

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

    # ── Past anticipated thermal commitments (from adterm.dat) ──────────
    # Empty for non-GNL cases (despacho_antecipado_gnl=0 in dger.dat).
    # Each entry maps a thermal's NEWAVE code to its cobre thermal_id.
    #
    # Cobre's anticipated-thermal LP currently REQUIRES every ``values_mw``
    # entry to be ``0.0`` — the fishing-constraint activation predicate is
    # FALSE at every stage before the first matured delivery and the ring
    # buffer overwrites slot 0 with the LP's own decision before any
    # constraint can read a seeded value (see plans/anticipated-thermals
    # AnticipatedCommitmentHistory docstring).  Non-zero entries are
    # rejected by ``cobre-io::validation::semantic::thermal``.
    #
    # We still compute the block-weighted NEWAVE MW so we can warn the
    # user about the pre-horizon dispatch being silently dropped, but we
    # emit zeros to satisfy cobre's current validator.
    past_anticipated_commitments: list[dict] = []
    for newave_code, dispatch in read_anticipated_dispatch(nw_files).items():
        try:
            thermal_id = id_map.thermal_id(newave_code)
        except KeyError:
            _LOG.warning(
                "adterm.dat references thermal code=%d that is absent from "
                "the cobre id map; skipping its anticipated commitment.",
                newave_code,
            )
            continue
        if any(abs(v) > 1e-9 for v in dispatch.values_mw):
            _LOG.warning(
                "adterm.dat thermal code=%d carries pre-horizon dispatch "
                "values [%s] MW that cobre's current LP cannot honour; "
                "writing zeros to satisfy the validator (NEWAVE-side "
                "dispatch over the next %d stages will not be enforced).",
                newave_code,
                ", ".join(f"{v:.4f}" for v in dispatch.values_mw),
                dispatch.lead_stages,
            )
        past_anticipated_commitments.append(
            {
                "thermal_id": thermal_id,
                "values_mw": [0.0] * dispatch.lead_stages,
            }
        )
    past_anticipated_commitments.sort(key=lambda c: c["thermal_id"])

    result: dict = {
        "$schema": _SCHEMA_URL,
        "storage": storage,
        "filling_storage": [],
    }
    if past_anticipated_commitments:
        result["past_anticipated_commitments"] = past_anticipated_commitments
    return result
