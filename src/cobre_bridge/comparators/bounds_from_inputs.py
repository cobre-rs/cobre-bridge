"""Compute NEWAVE bounds from input files as flat dicts.

Standalone functions that extract bounds computation logic from the
converter modules (``hydro.py``, ``thermal.py``, ``network.py``) and
return simple ``dict[tuple[int, int, str], float]`` lookups keyed by
``(cobre_entity_id, stage_id, bound_name)``.

These functions avoid coupling to the conversion pipeline — they read
NEWAVE input files directly via inewave, except for ``read_cadastro``
which applies permanent MODIF overrides.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from cobre_bridge.horizon import (
    seasonal_step_function,
    study_horizon,
)
from cobre_bridge.id_map import NewaveIdMap
from cobre_bridge.newave_files import NewaveFiles
from cobre_bridge.plants import active_hydro_codes

# -------------------------------------------------------------------
# Hydro bounds
# -------------------------------------------------------------------


def _extract_temporal_overrides(
    nw_files: NewaveFiles, confhd_codes: list[int]
) -> dict[int, list[dict]]:
    """Extract MODIF.DAT temporal overrides for plants in *confhd_codes*.

    Returns a dict keyed by plant code, each value a list of override
    dicts: ``{"type": str, "month": int, "year": int, "value": float}``.
    """
    from inewave.newave import Modif

    modif_path = nw_files.modif
    if modif_path is None:
        return {}

    temporal_types = frozenset(
        {"VAZMINT", "VMAXT", "VMINT", "CFUGA", "CMONT", "TURBMINT", "TURBMAXT"}
    )

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
            if type_name not in temporal_types:
                continue

            data = rec.data_inicio
            month = int(data.month)
            year = int(data.year)

            if type_name == "VAZMINT":
                value = float(rec.vazao)
            elif type_name in ("VMAXT", "VMINT"):
                value = float(rec.volume)
            elif type_name in ("CFUGA", "CMONT"):
                value = float(rec.nivel)
            elif type_name in ("TURBMINT", "TURBMAXT"):
                value = float(rec.turbinamento)
            else:
                continue

            plant_overrides.append(
                {"type": type_name, "month": month, "year": year, "value": value}
            )

        if plant_overrides:
            result[code] = plant_overrides

    return result


def compute_hydro_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Compute per-stage hydro bounds from NEWAVE input files.

    Returns ``{(cobre_hydro_id, stage_id, bound_name): value}`` where
    ``bound_name`` is one of: ``storage_min``, ``storage_max``,
    ``turbined_min``, ``turbined_max``, ``outflow_min``.

    Values are in hm3 (storage) or m3/s (turbined/outflow).  Big-M
    sentinel values (>= 99990) are excluded (treated as "no limit").

    Returns an empty dict if MODIF.DAT is absent or contains no
    relevant temporal override records.
    """
    from inewave.newave import Confhd

    from cobre_bridge.converters.hydro import read_cadastro

    modif_path = nw_files.modif
    if modif_path is None:
        return {}

    from inewave.newave import Dger

    dger = Dger.read(str(nw_files.dger))
    horizon = study_horizon(dger)

    # Post-study seasonalize flags (mirror convert_storage_bounds): VMINT/VMAXT
    # repeat seasonally only when their dger flag is set; outflow/turbined freeze.
    sazonaliza_vmaxt = int(getattr(dger, "sazonaliza_vmaxt", 0) or 0) == 1
    sazonaliza_vmint = int(getattr(dger, "sazonaliza_vmint", 0) or 0) == 1

    cadastro = read_cadastro(nw_files)

    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    confhd_codes = active_hydro_codes(confhd_df)

    temporal_overrides = _extract_temporal_overrides(nw_files, confhd_codes)

    def _build_step_function(
        recs: list[dict],
        transform: Callable[[float], float],
        *,
        seasonalize: bool,
    ) -> dict[int, float]:
        """Thin adapter over :func:`cobre_bridge.horizon.seasonal_step_function`.

        Shares the forward-fill + seasonalize-vs-freeze post-study logic with
        ``hydro.convert_storage_bounds`` so this comparator checks the converter
        against the *same* derivation, not a hand-maintained copy of it.
        """
        return seasonal_step_function(
            [(int(r["year"]), int(r["month"]), float(r["value"])) for r in recs],
            transform,
            seasonalize=seasonalize,
            horizon=horizon,
        )

    result: dict[tuple[int, int, str], float] = {}

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

        # Storage bounds (percentage -> hm3). Seasonal post-study iff flag set.
        vmaxt_by_stage: dict[int, float] = {}
        vmint_by_stage: dict[int, float] = {}
        if useful > 0:
            vmaxt_by_stage = _build_step_function(
                vmaxt, _pct_to_hm3, seasonalize=sazonaliza_vmaxt
            )
            vmint_by_stage = _build_step_function(
                vmint, _pct_to_hm3, seasonalize=sazonaliza_vmint
            )

        # Turbined bounds (absolute m3/s) — no seasonalize flag → freeze.
        turbmaxt_by_stage = _build_step_function(turbmaxt, _identity, seasonalize=False)
        turbmint_by_stage = _build_step_function(turbmint, _identity, seasonalize=False)

        # Outflow bounds (absolute m3/s) — no seasonalize flag → freeze.
        vazmint_by_stage = _build_step_function(vazmint, _identity, seasonalize=False)

        all_stages = sorted(
            set(vmaxt_by_stage)
            | set(vmint_by_stage)
            | set(turbmaxt_by_stage)
            | set(turbmint_by_stage)
            | set(vazmint_by_stage)
        )
        for stage_id in all_stages:
            if stage_id in vmaxt_by_stage:
                result[(hydro_id, stage_id, "storage_max")] = vmaxt_by_stage[stage_id]
            if stage_id in vmint_by_stage:
                result[(hydro_id, stage_id, "storage_min")] = vmint_by_stage[stage_id]
            if stage_id in turbmaxt_by_stage:
                result[(hydro_id, stage_id, "turbined_max")] = turbmaxt_by_stage[
                    stage_id
                ]
            if stage_id in turbmint_by_stage:
                result[(hydro_id, stage_id, "turbined_min")] = turbmint_by_stage[
                    stage_id
                ]
            if stage_id in vazmint_by_stage:
                result[(hydro_id, stage_id, "outflow_min")] = vazmint_by_stage[stage_id]

    return result


# -------------------------------------------------------------------
# Thermal bounds
# -------------------------------------------------------------------


def compute_thermal_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Return per-stage thermal generation bounds for the bounds comparison.

    ``{(cobre_thermal_id, stage_id, bound_name): value}`` where ``bound_name``
    is ``generation_min`` or ``generation_max`` (MW).

    Delegates to the converter's
    :func:`cobre_bridge.converters.thermal.convert_thermal_bounds` — the single
    definition of how thermal bounds are built from NEWAVE inputs — and reshapes
    its parquet output into the comparison lookup. This stops the comparator
    from silently drifting from the converter (it used to be a near-verbatim
    copy that had to be patched in lockstep, e.g. the FCMAX/GTMIN fix).

    Returns an empty dict when neither ``expt.dat`` nor ``manutt.dat`` is present
    (no override-driven thermal bounds to check). The converter's plant set
    (``expt | manutt | base | cost_varies``) matches the comparator's former
    ``expt | manutt | base`` because ``cost_varies`` is a subset of the base
    plants, so the comparison keys are unchanged.
    """
    if nw_files.expt is None and nw_files.manutt is None:
        return {}

    from cobre_bridge.case import NewaveCase
    from cobre_bridge.converters.thermal import convert_thermal_bounds

    # The converter now consumes a parsed NewaveCase; the comparator still works
    # from paths, so wrap them (parse-on-demand, behaviour-identical).
    table = convert_thermal_bounds(NewaveCase(files=nw_files), id_map)
    if table is None:
        return {}

    cols = table.to_pydict()
    result: dict[tuple[int, int, str], float] = {}
    for thermal_id, stage_id, min_mw, max_mw in zip(
        cols["thermal_id"],
        cols["stage_id"],
        cols["min_generation_mw"],
        cols["max_generation_mw"],
    ):
        result[(int(thermal_id), int(stage_id), "generation_min")] = float(min_mw)
        result[(int(thermal_id), int(stage_id), "generation_max")] = float(max_mw)
    return result


# -------------------------------------------------------------------
# Line bounds
# -------------------------------------------------------------------


def compute_line_bounds(
    nw_files: NewaveFiles,
    id_map: NewaveIdMap,
) -> dict[tuple[int, int, str], float]:
    """Compute per-stage line flow bounds from NEWAVE input files.

    Returns ``{(cobre_line_id, stage_id, bound_name): value}`` where
    ``bound_name`` is one of: ``direct_flow_max``, ``reverse_flow_max``.

    Values are in MW.  Interchange limits have no seasonalize flag, so
    post-study stages freeze at the last study stage's bounds (mirrors
    ``network.convert_line_bounds``).

    Returns an empty dict if ``sistema.dat`` has no interchange limits.
    """
    from inewave.newave import Dger, Sistema

    sistema = Sistema.read(str(nw_files.sistema))
    limites_df: pd.DataFrame | None = sistema.limites_intercambio

    if limites_df is None or limites_df.empty:
        return {}

    dger = Dger.read(str(nw_files.dger))
    h = study_horizon(dger)
    start_month, start_year = h.start_month, h.start_year
    study_months, total_stages = h.study_months, h.total_stages

    study_end_year = start_year + (start_month - 1 + study_months) // 12
    study_end_month = ((start_month - 1 + study_months) % 12) + 1

    # Canonical pair -> line_id mapping (same logic as convert_lines).
    all_pairs: set[tuple[int, int]] = set()
    for _, row in limites_df.iterrows():
        de = int(row["submercado_de"])
        para = int(row["submercado_para"])
        src, tgt = (de, para) if de < para else (para, de)
        all_pairs.add((src, tgt))

    pair_to_line_id: dict[tuple[int, int], int] = {
        pair: lid for lid, pair in enumerate(sorted(all_pairs))
    }

    # Per-date lookup: {(src, tgt, year, cal_month) -> {direct_mw, reverse_mw}}.
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

        # Direction convention MUST match convert_lines / convert_line_bounds
        # in converters/network.py: per inewave's SISTEMA.DAT parse order,
        # sentido == 0 is the first block (de -> para), sentido == 1 the
        # reverse. Comparing against a flipped convention here previously
        # reported every line as a (spurious) direct/reverse mismatch.
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

    # Last-year lookup for post-study repetition.
    last_year_per_key: dict[tuple[int, int, int], tuple[int, dict[str, float]]] = {}
    for (src, tgt, yr, cal_month), caps in date_lookup.items():
        key3 = (src, tgt, cal_month)
        existing = last_year_per_key.get(key3)
        if existing is None or yr > existing[0]:
            last_year_per_key[key3] = (yr, caps)
    last_year_lookup: dict[tuple[int, int, int], dict[str, float]] = {
        k: v for k, (_, v) in last_year_per_key.items()
    }

    result: dict[tuple[int, int, str], float] = {}

    # Interchange limits have no seasonalize flag → post-study freezes at the
    # last study stage's value (mirror convert_line_bounds in network.py).
    ls_y = start_year + (start_month - 1 + study_months - 1) // 12
    ls_m = ((start_month - 1 + study_months - 1) % 12) + 1

    for pair, line_id in sorted(pair_to_line_id.items(), key=lambda x: x[1]):
        src, tgt = pair
        freeze_caps = date_lookup.get((src, tgt, ls_y, ls_m)) or last_year_lookup.get(
            (src, tgt, ls_m), {"direct_mw": 0.0, "reverse_mw": 0.0}
        )
        y, m = start_year, start_month
        for stage_id in range(total_stages):
            is_post_study = (y > study_end_year) or (
                y == study_end_year and m >= study_end_month
            )

            if is_post_study:
                caps = freeze_caps
            else:
                caps = date_lookup.get((src, tgt, y, m))
                if caps is None:
                    caps = last_year_lookup.get(
                        (src, tgt, m), {"direct_mw": 0.0, "reverse_mw": 0.0}
                    )

            result[(line_id, stage_id, "direct_flow_max")] = caps["direct_mw"]
            result[(line_id, stage_id, "reverse_flow_max")] = caps["reverse_mw"]

            m += 1
            if m > 12:
                m = 1
                y += 1

    return result
