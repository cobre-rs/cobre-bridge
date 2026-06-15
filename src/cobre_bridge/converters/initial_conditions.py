"""Initial conditions converter: maps the source model initial storage to Cobre JSON."""

from __future__ import annotations

import logging

import pandas as pd

from cobre_bridge.case import NewaveCase
from cobre_bridge.converters.anticipated import read_anticipated_dispatch
from cobre_bridge.converters.hydro import read_cadastro
from cobre_bridge.converters.thermal import thermal_generation_bounds
from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/initial_conditions.schema.json"
)


def convert_initial_conditions(case: NewaveCase, id_map: NewaveIdMap) -> dict:
    """Convert the source model initial reservoir storage to a Cobre initial_conditions
    dict.

    Reads ``hidr.dat`` and ``confhd.dat`` from *case*.  Initial
    storage is derived from ``Confhd.usinas.volume_inicial_percentual``
    (a percentage of ``volume_maximo`` from Hidr).

    Values outside ``[0, 100]`` are clamped with a warning.

    Parameters
    ----------
    case:
        Parsed the source model case.
    id_map:
        Pre-built ID mapping for hydro IDs.

    Raises
    ------
    ValueError
        If a hydro in ``confhd.dat`` references a code absent in
        ``hidr.dat``.
    """
    # Apply permanent MODIF.DAT overrides (notably VOLMIN) so the initial-storage
    # percentage is taken of the *operational* useful volume — the same min the bounds
    # converter uses.  The source model applies ``volume_inicial_percentual`` to the
    # VOLMIN-adjusted useful range, not the raw hidr.dat one (verified against pmo.dat
    # "VOLUME ARMAZENADO INICIAL": I. SOLTEIRA V.INIC 3940.9 hm³ @ 71.70% is 71.70% of
    # (vmax − VOLMIN), not (vmax − raw_min)).
    cadastro = read_cadastro(case)

    existing = case.active_hydros

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

        # Fio-d'água plants don't accumulate water across stages, so the
        # bounds converter collapses their storage to a single point in
        # ``hydro.py``.  Anchor the initial storage to that same point so the
        # initial condition stays inside the (collapsed) [min, max] range:
        #   * 'D' (daily) → ``volume_referencia`` (legacy, validated).
        #   * 'S' (run-of-river) → ``volume_minimo`` (the source model pins ITAIPU at
        #     VARMPUH 0% = Vmin; matches the collapse in ``hydro.py``).
        tipo_reg = str(hreg.get("tipo_regulacao", "")).strip()
        vol_ref_raw = hreg.get("volume_referencia")
        anchored: float | None = None
        if tipo_reg == "D" and vol_ref_raw is not None and not pd.isna(vol_ref_raw):
            anchored = float(vol_ref_raw)
        elif tipo_reg == "S":
            anchored = vol_min
        if anchored is not None:
            storage.append(
                {
                    "hydro_id": id_map.hydro_id(newave_code),
                    "value_hm3": anchored,
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
    # Empty for non-GNL cases (despacho_antecipado_gnl=0 in dger.dat). Each entry maps a
    # thermal's the source model code to its cobre thermal_id.
    #
    # Cobre (>= 0.7.0) honours non-zero pre-horizon seeds: the always-active anticipated
    # "fishing" equality pins generation to the committed MW at each delivery stage,
    # faithfully reproducing the source model's pre-commitment. The committed value must
    # lie within the plant's static generation bounds ``[min_mw, max_mw]``
    # (``thermals.json`` / ``cobre-io`` semantic validator); an out-of-range seed makes
    # that stage's fishing equality infeasible, so we clamp into range and warn rather
    # than emit a case cobre would reject.
    anticipated = read_anticipated_dispatch(case)
    gen_bounds = thermal_generation_bounds(case) if anticipated else {}
    past_anticipated_commitments: list[dict] = []
    for newave_code, dispatch in anticipated.items():
        try:
            thermal_id = id_map.thermal_id(newave_code)
        except KeyError:
            _LOG.warning(
                "adterm.dat references thermal code=%d that is absent from "
                "the cobre id map; skipping its anticipated commitment.",
                newave_code,
            )
            continue
        lo, hi = gen_bounds.get(newave_code, (float("-inf"), float("inf")))
        seeded = [min(max(v, lo), hi) for v in dispatch.values_mw]
        if any(abs(s - v) > 1e-9 for s, v in zip(seeded, dispatch.values_mw)):
            _LOG.warning(
                "adterm.dat thermal code=%d committed dispatch [%s] MW falls "
                "outside the plant's generation bounds [%.4f, %.4f]; clamping to "
                "[%s] to keep cobre's anticipated fishing equality feasible.",
                newave_code,
                ", ".join(f"{v:.4f}" for v in dispatch.values_mw),
                lo,
                hi,
                ", ".join(f"{s:.4f}" for s in seeded),
            )
        past_anticipated_commitments.append(
            {
                "thermal_id": thermal_id,
                "values_mw": seeded,
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
