"""Hydro conversion for DECOMP-like decks (registry ``hidr.dat`` + ``UH``).

The registry file is byte-identical across the two source families, so the
shared row-level physics helpers (cota polynomial, hydraulic losses) are
reused verbatim. The rated-capability sum is *not* reused as-is, though:
:func:`_compute_max_turbined_rated_ac_adjusted` below is a DECOMP-only
counterpart that layers the ``NUMCON``/``NUMMAQ``/``POTEFE``/``VAZEFE``
machine-configuration overrides on top of the same per-conjunto registry
data the shared ``converters.hydro._compute_max_turbined_rated`` reads —
some plants' *true* in-service machine count differs from ``hidr.dat``'s
nameplate conjunto sum, and the shared helper (which the source-model side
keeps byte-identical) exposes only the pre-summed total, not a per-conjunto
override hook. Unlike the source-model side, these overrides are
**temporal**: :class:`~cobre_bridge.decomp.cadastro.EffectiveCadastro`'s
``machine_conjunto_count``/``machine_set`` accessors resolve them to a
per-stage-effective value (a plant's machine set can change mid-horizon), so
every rated-capacity read below takes a ``stage_index`` and a plant's or
group's *declared* capacity is the max-over-stages envelope of that
per-stage value (:func:`_rated_envelope`/:func:`_conjunto_rated_envelope`) —
the same outer-envelope construction :func:`~cobre_bridge.decomp.cadastro.
storage_envelope` uses for the reservoir block. A stage whose effective
machine set drops below that envelope gets a sparse per-stage overlay row
instead (:func:`convert_hydro_group_availability`), never a change to the
declared envelope itself.

**Tracked E5 fidelity gap (user-confirmed 2026-08-07):** the emitted
``max_turbined_m3s`` here is always the AC-adjusted **rated** (un-derated)
unit-flow sum, with no per-stage-head adjustment — unlike the source-model
side's ``converters.hydro._compute_max_turbined_head_corrected``, which
derates by the affinity law ``(h_op / h_nom) ** k_turb`` and a
``p_inst / (rho_esp * h)`` power cap. Porting that head-aware engolimento
correction to DECOMP (per-stage ``h = rho_eq / rho_esp``) is deliberately
**deferred to E5**, blocked on the ``ALTEFE`` (effective head) accessor gap
noted in ``decomp/cadastro.py``. This is a recorded, tracked gap, not a
silent omission — it composes cleanly with the per-stage machine-set
envelope here once landed, since the two adjustments are orthogonal (one
scales the rated flow by head, the other scales it by which conjuntos are
effective).

The one plant whose maintenance and availability registers are declared
*per generating-unit group* rather than per plant (Itaipu, code 66, carries
two ``MP``/``FD`` ``frequencia`` rows) gets two conjunto-backed unit groups
instead of the usual single mirror group — see
:func:`_build_split_unit_groups`. Scope is the ratified loop-closing
milestone: faithful registry, cascade, capability and initial storage —
with everything whose faithful treatment is gated on later features
deferred **loudly** (one summary log warning each): registry overrides
beyond ``VAZMIN``/``NUMCON``/``NUMMAQ``/``POTEFE``/``VAZEFE``, travel time
(``VI``), and FPHA/tailrace/evaporation models.

``UH`` rows without an initial volume (the coupling-only registrations)
are excluded from the operated set and reported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
from inewave.newave import Hidr

from cobre_bridge.converters.hydro import (
    _PRODUCTION_MODELS_SCHEMA_URL,
    _SCHEMA_URL,
    _apply_hydraulic_loss,
    _mean_cota_over_volume,
    build_mirror_unit_group,
)
from cobre_bridge.decomp.cadastro import storage_envelope
from cobre_bridge.decomp.group_bounds import GroupBoundEntry

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date
    from pathlib import Path

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: Itaipu's plant code — the one plant whose ``MP``/``FD`` maintenance and
#: availability registers are declared per generating-unit group (one row
#: per ``frequencia``, 50/60 Hz) rather than per plant. Both
#: :func:`convert_hydros` (two conjunto-backed unit groups instead of one
#: mirror group) and :func:`convert_hydro_group_availability` (a per-group,
#: not per-plant, availability overlay) special-case it; every other plant
#: takes the ordinary single-group path.
_ITAIPU_CODE = 66


def read_hidr(path: Path) -> pd.DataFrame:
    """Read the hydro registry file, indexed by plant code."""
    df = Hidr.read(str(path)).cadastro
    if df is None or df.empty:
        raise FileNotFoundError(f"{path} has no registry data")
    return df


def _operated_uh(dadger: Dadger) -> pd.DataFrame:
    """The ``UH`` rows carrying an initial volume (the operated plants).

    Rows without one (coupling-only registrations) are reported and
    excluded — their terminal-value treatment is the boundary importer's
    D3 territory, not the registry's.
    """
    uh = dadger.uh(df=True)
    if uh is None or uh.empty:
        raise ValueError("the deck has no UH records; cannot convert hydros")
    operated = uh[uh["volume_inicial"].notna()]
    excluded = uh[uh["volume_inicial"].isna()]
    if not excluded.empty:
        _LOG.warning(
            "%d UH registration(s) without an initial volume excluded from "
            "the operated set (coupling-only): codes %s",
            len(excluded),
            sorted(int(c) for c in excluded["codigo_usina"]),
        )
    return operated


def _downstream_operated(
    hidr: pd.DataFrame,
    code: int,
    operated: set[int],
) -> int | None:
    """Walk the registry cascade to the next *operated* plant downstream.

    Non-operated intermediates are skipped through (their routing is
    instantaneous absence — the water continues down the declared chain);
    code 0 is the sink.
    """
    visited = {code}
    current = int(hidr.loc[code, "codigo_usina_jusante"])
    while current != 0 and current not in operated:
        if current in visited or current not in hidr.index:
            _LOG.warning(
                "cascade walk from plant %d hit an invalid link at %d; "
                "treating as a sink",
                code,
                current,
            )
            return None
        visited.add(current)
        current = int(hidr.loc[current, "codigo_usina_jusante"])
    return current if current != 0 else None


def _conjunto_rated_ac_adjusted(
    hreg: pd.Series,
    code: int,
    conjunto_index: int,
    effective: EffectiveCadastro,
    stage_index: int,
) -> tuple[float, float]:
    """Return ``(q, p)`` — the AC-adjusted rated flow/power of one 1-based
    ``conjunto_index`` at *stage_index* — the per-conjunto building block
    :func:`_compute_max_turbined_rated_ac_adjusted` sums over every
    conjunto, and :func:`_build_split_unit_groups`/
    :func:`_conjunto_rated_envelope`/:func:`convert_hydro_group_availability`
    call directly for a single conjunto to get one per-frequency group's own
    bounds. Reads ``effective.machine_set(code, conjunto_index,
    stage_index)``: when present, its three fields already reflect
    :func:`~cobre_bridge.decomp.cadastro.build_effective_cadastro`'s
    independent per-field densification (a field with no override of its
    own is the ``hidr`` base, forward-filled flat); ``None`` means the pair
    carries no override at all, and every field falls back to the
    ``hidr.dat`` row directly — the same ``.get(..., base)`` fallback the
    pre-ticket date-blind reader used.
    """
    machine_set = effective.machine_set(code, conjunto_index, stage_index)
    if machine_set is None:
        n_machines = int(hreg[f"maquinas_conjunto_{conjunto_index}"])
        q_nom = float(hreg[f"vazao_nominal_conjunto_{conjunto_index}"])
        p_nom = float(hreg[f"potencia_nominal_conjunto_{conjunto_index}"])
    else:
        n_machines = machine_set.numero_maquinas
        q_nom = machine_set.vazao
        p_nom = machine_set.potencia
    return q_nom * n_machines, p_nom * n_machines


def _compute_max_turbined_rated_ac_adjusted(
    hreg: pd.Series,
    code: int,
    effective: EffectiveCadastro,
    stage_index: int,
) -> tuple[float, float]:
    """Return ``(max_turbined, max_generation)`` — the AC-adjusted rated
    nameplate capacity at *stage_index* — the DECOMP-only counterpart to the
    shared, un-derated, stage-invariant ``converters.hydro.
    _compute_max_turbined_rated``.

    Sums :func:`_conjunto_rated_ac_adjusted` over every conjunto effective
    at *stage_index* (``effective.machine_conjunto_count(code,
    stage_index)``, falling back to the ``hidr`` base
    ``numero_conjuntos_maquinas`` when that returns ``None``). Identical to
    the shared helper for a plant with no machine-set override at all —
    every lookup misses and falls back at every stage — so this changes
    nothing for the majority; it exists only because the override is
    per-conjunto, per-stage, and the shared helper exposes just the
    pre-summed, stage-invariant total. The shared helper itself is
    untouched. Callers needing the plant's *declared* (max-over-stages)
    capacity use :func:`_rated_envelope` instead of calling this per stage
    directly.
    """
    n_sets = effective.machine_conjunto_count(code, stage_index)
    if n_sets is None:
        n_sets = int(hreg["numero_conjuntos_maquinas"])
    max_turbined = 0.0
    max_generation = 0.0
    for i in range(1, n_sets + 1):
        q, p = _conjunto_rated_ac_adjusted(hreg, code, i, effective, stage_index)
        max_turbined += q
        max_generation += p
    return max_turbined, max_generation


def _rated_envelope(
    hreg: pd.Series, code: int, effective: EffectiveCadastro
) -> tuple[float, float]:
    """Max-over-stages AC-adjusted rated ``(max_turbined, max_generation)``
    for the whole plant.

    The entity ``generation``/mirror-``unit_groups`` envelope for every
    non-split plant (:func:`convert_hydros`) and the per-plant comparison
    base for the B8 availability overlay
    (:func:`convert_hydro_group_availability`) — mirrors
    :func:`~cobre_bridge.decomp.cadastro.storage_envelope`'s outer-bound
    construction: the widest each of :func:`_compute_max_turbined_rated_
    ac_adjusted`'s two components ever reaches over the horizon, taken
    independently (the two maxima need not land on the same stage). A
    constant machine set collapses this to the stage-0 value, matching the
    pre-ticket date-blind value exactly. Not used for the split plant
    (Itaipu): its entity envelope is instead the *sum* of the two groups'
    own :func:`_conjunto_rated_envelope`, computed in
    :func:`convert_hydros`, so cobre rule 41 holds by construction rather
    than by coincidence of the two groups peaking on the same stage.
    """
    per_stage = [
        _compute_max_turbined_rated_ac_adjusted(hreg, code, effective, stage)
        for stage in range(effective.n_stages)
    ]
    return (
        max(turbined for turbined, _ in per_stage),
        max(generation for _, generation in per_stage),
    )


def _conjunto_rated_envelope(
    hreg: pd.Series, code: int, conjunto_index: int, effective: EffectiveCadastro
) -> tuple[float, float]:
    """Max-over-stages AC-adjusted rated ``(q, p)`` for one conjunto.

    The split plant's own per-group declared envelope
    (:func:`_build_split_unit_groups`) and the per-group comparison base
    for its own B8 availability overlay
    (:func:`convert_hydro_group_availability`) — the per-conjunto mirror of
    :func:`_rated_envelope`.
    """
    per_stage = [
        _conjunto_rated_ac_adjusted(hreg, code, conjunto_index, effective, stage)
        for stage in range(effective.n_stages)
    ]
    return (
        max(q for q, _ in per_stage),
        max(p for _, p in per_stage),
    )


def _split_plant_frequencies(
    hreg: pd.Series,
    code: int,
    effective: EffectiveCadastro,
    mp: pd.DataFrame | None,
    fd: pd.DataFrame | None,
) -> list[float]:
    """Validate and return the split plant's ``frequencia`` values, sorted
    ascending — group id ``i`` (``i`` in ``0..len-1``) is then this list's
    ``i``-th entry, backed by hidr conjunto ``i + 1``
    (:func:`_build_split_unit_groups`,
    :func:`convert_hydro_group_availability`). The convention is
    oracle-invisible for a plant whose per-conjunto capacities are all
    identical (Itaipu's two conjuntos both being 7000 MW / 6620 m^3/s), so it
    is pinned for determinism, not derived from any frequency-to-conjunto
    label in the registry (``hidr.dat`` carries none).

    The conjunto-count/frequency-count agreement check below deliberately
    uses the plant's **stage-0** effective conjunto count
    (``effective.machine_conjunto_count(code, 0)``), never a per-stage
    value: no reference deck exercises a mid-horizon ``NUMCON`` change on
    the split plant specifically, and the ``MP``/``FD`` frequency rows
    themselves carry no per-stage cardinality to validate against. Tracked
    non-goal, not a silent branch (Itaipu carries no ``NUMCON`` override on
    rv3, so this is base-count = 2 there regardless).

    Raises
    ------
    ValueError
        Naming *code*, if the ``MP`` and ``FD`` registers disagree on the
        set of declared frequencies (or either is empty) — no silent
        single-group fallback for a mismatched registry — or if the
        stage-0 AC-adjusted conjunto count does not equal the number of
        declared frequencies.
    """

    def _freqs(table: pd.DataFrame | None) -> set[float]:
        if table is None or table.empty:
            return set()
        return {
            float(f) for f in table.loc[table["codigo_usina"] == code, "frequencia"]
        }

    mp_freqs, fd_freqs = _freqs(mp), _freqs(fd)
    if not mp_freqs or mp_freqs != fd_freqs:
        raise ValueError(
            f"plant {code}: MP frequencia rows {sorted(mp_freqs)} and FD "
            f"frequencia rows {sorted(fd_freqs)} must agree and be "
            "non-empty for the per-frequency split plant"
        )
    frequencies = sorted(mp_freqs)

    n_sets = effective.machine_conjunto_count(code, 0)
    if n_sets is None:
        n_sets = int(hreg["numero_conjuntos_maquinas"])
    if n_sets != len(frequencies):
        raise ValueError(
            f"plant {code}: numero_conjuntos_maquinas ({n_sets}) does not "
            f"match the number of MP/FD frequencia rows ({len(frequencies)})"
        )
    return frequencies


def _build_split_unit_groups(
    hreg: pd.Series,
    code: int,
    name: str,
    bus_id: int,
    frequencies: list[float],
    effective: EffectiveCadastro,
) -> tuple[list[dict[str, object]], float, float]:
    """Conjunto-backed unit groups for the per-frequency split plant.

    Group id ``i`` (``i`` in ``0..len(frequencies)-1``, ascending frequency,
    so id 0 = the lowest — 50 Hz for Itaipu) is backed by hidr conjunto
    ``i + 1``: its ``max_generation_mw``/``max_turbined_m3s`` is *that*
    conjunto's own max-over-stages AC-adjusted rated envelope
    (:func:`_conjunto_rated_envelope`), not the plant's. All groups sit on
    *bus_id*, the plant's own bus (no per-group bus on this deck).

    Returns the groups plus their summed ``(max_turbined_m3s,
    max_generation_mw)``: :func:`convert_hydros` declares the plant's own
    entity envelope as exactly this sum, rather than recomputing it
    independently, so cobre rule 41 holds by construction even though the
    two groups' own per-stage machine-set changes (if any) need not peak on
    the same stage.
    """
    groups: list[dict[str, object]] = []
    total_turbined = 0.0
    total_generation = 0.0
    for i in range(len(frequencies)):
        conjunto_index = i + 1
        q, p = _conjunto_rated_envelope(hreg, code, conjunto_index, effective)
        groups.append(
            build_mirror_unit_group(
                name=name,
                bus_id=bus_id,
                min_generation_mw=0.0,
                max_generation_mw=p,
                min_turbined_m3s=0.0,
                max_turbined_m3s=q,
                group_id=i,
            )
        )
        total_turbined += q
        total_generation += p
    return groups, total_turbined, total_generation


def _frequency_row(
    table: pd.DataFrame | None, code: int, frequency: float
) -> pd.Series:
    """The single ``(codigo_usina, frequencia)``-matched register row.

    Unlike the ordinary single-group case (a missing register defaults its
    factor to ``1.0``), the split plant's two groups each require exactly
    one ``MP`` row and one ``FD`` row at their own frequency — validated by
    :func:`_split_plant_frequencies` before this is called, so a missing or
    duplicated row here means the registers disagree with that validation
    (defensive; should not happen in practice).
    """
    if table is None:
        raise ValueError(f"plant {code}: no register table for frequencia {frequency}")
    rows = table.loc[
        (table["codigo_usina"] == code) & (table["frequencia"] == frequency)
    ]
    if len(rows) != 1:
        raise ValueError(
            f"plant {code}: expected exactly one register row at frequencia "
            f"{frequency}, found {len(rows)}"
        )
    return rows.iloc[0]


def convert_hydros(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    start_date: date,
    effective: EffectiveCadastro,
) -> dict:
    """Build ``hydros.json`` for the operated plants.

    Capability is the installed (un-derated, ``TEIF``/``IP``-free)
    max-over-stages envelope of the AC-adjusted rated unit-flow/power sum,
    sourced per stage from *effective*'s machine-set view
    (:func:`_rated_envelope`, mirroring :func:`storage_envelope`'s
    outer-bound construction for the reservoir block) — some plants' *true*
    in-service machine configuration differs from ``hidr.dat``'s nameplate
    conjunto sum, and/or changes mid-horizon; a stage whose effective
    capacity drops below that envelope gets a sparse per-stage overlay
    instead (:func:`convert_hydro_group_availability`), never a change to
    the declared envelope itself. The production model is constant
    productivity, with the value emitted separately
    (:func:`convert_energy_productivity`).

    **Tracked E5 fidelity gap (user-confirmed 2026-08-07):** this envelope
    is always the rated (un-derated) unit-flow sum with no per-stage-head
    adjustment — see the module docstring; the source-model side's
    head-aware engolimento correction is not ported here.

    Every plant declares one mirror unit group, except the per-frequency
    split plant (Itaipu, code 66), which declares two conjunto-backed groups
    instead (:func:`_build_split_unit_groups`), whose summed maxima *are*
    the entity envelope by construction — cobre rule 41 holds even though
    the two groups' own per-stage machine-set changes (if any) need not
    peak on the same stage. The entity ``reservoir`` block is the plant's
    outer per-stage storage envelope (:func:`storage_envelope`), so
    per-stage bound overrides (:func:`cobre_bridge.decomp.bounds.
    convert_storage_bounds`) always sit inside it. Deferred fidelity is
    logged once per family.
    """
    operated = _operated_uh(dadger)
    operated_codes = set(id_map.hydro_codes)
    min_outflow_by_code: dict[int, float] = {}
    for _, row in operated.iterrows():
        value = row.get("vazao_defluente_minima")
        if not pd.isna(value):
            min_outflow_by_code[int(row["codigo_usina"])] = float(value)

    # Read the MP/FD registers only if the split plant is actually present —
    # keeps every other (hand-built or real, non-Itaipu) caller from needing
    # a Dadger stub for methods this function otherwise never touches.
    mp = fd = None
    if _ITAIPU_CODE in operated_codes:
        mp = dadger.mp(df=True)
        fd = dadger.fd(df=True)

    op_date = start_date.isoformat()
    hydros: list[dict] = []
    for code in id_map.hydro_codes:
        if code not in hidr.index:
            raise ValueError(f"UH plant {code} is not in the hydro registry")
        hreg = hidr.loc[code]
        name = str(hreg["nome_usina"]).strip()
        downstream = _downstream_operated(hidr, code, operated_codes)
        min_storage_hm3, max_storage_hm3 = storage_envelope(effective, code)
        bus_id = id_map.bus_id(int(hreg["submercado"]))
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(hreg, code, effective, mp, fd)
            unit_groups, max_turbined, max_generation = _build_split_unit_groups(
                hreg, code, name, bus_id, frequencies, effective
            )
        else:
            max_turbined, max_generation = _rated_envelope(hreg, code, effective)
            unit_groups = [
                build_mirror_unit_group(
                    name=name,
                    bus_id=bus_id,
                    min_generation_mw=0.0,
                    max_generation_mw=max_generation,
                    min_turbined_m3s=0.0,
                    max_turbined_m3s=max_turbined,
                )
            ]
        entry: dict = {
            "id": id_map.hydro_id(code),
            "name": name,
            "operational_start_date": op_date,
            "downstream_id": (
                None if downstream is None else id_map.hydro_id(downstream)
            ),
            "reservoir": {
                "min_storage_hm3": min_storage_hm3,
                "max_storage_hm3": max_storage_hm3,
            },
            "outflow": {
                "min_outflow_m3s": min_outflow_by_code.get(code, 0.0),
                "max_outflow_m3s": None,
            },
            "generation": {
                "model": "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": 0.0,
                "max_generation_mw": max_generation,
            },
            "unit_groups": unit_groups,
        }
        hydros.append(entry)

    affected_codes = set(effective.machine_conjunto_counts) | {
        code for code, _ in effective.machine_sets
    }
    n_ac_affected = len(affected_codes & operated_codes)
    if n_ac_affected:
        _LOG.info(
            "applied AC NUMCON/NUMMAQ/POTEFE/VAZEFE machine-configuration "
            "overrides for %d plant(s)",
            n_ac_affected,
        )
    _LOG.warning(
        "deferred hydro fidelity (loop-closing milestone): AC registry "
        "overrides beyond VAZMIN/NUMCON/NUMMAQ/POTEFE/VAZEFE, VI travel "
        "time, and FPHA/tailrace/evaporation models are not applied yet"
    )
    return {"$schema": _SCHEMA_URL, "hydros": hydros}


def convert_initial_storage(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
) -> list[dict]:
    """Initial reservoir volumes from ``UH`` (% of useful → hm³).

    ``volume_inicial`` is a percentage of the *initial stage's* effective
    useful volume, not the plant's outer envelope, so the min/max here are
    read from :meth:`EffectiveCadastro.value` at stage ``0``.
    """
    operated = _operated_uh(dadger)
    storage: list[dict] = []
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        v_min = effective.value(code, "volume_minimo", 0)
        v_max = effective.value(code, "volume_maximo", 0)
        pct = float(row["volume_inicial"])
        value = v_min + (pct / 100.0) * (v_max - v_min)
        value = min(max(value, v_min), v_max)
        storage.append({"hydro_id": id_map.hydro_id(code), "value_hm3": value})
        dead = row.get("volume_morto_inicial")
        if not pd.isna(dead):
            _LOG.warning(
                "plant %d declares an initial dead volume (%s); "
                "dead-volume filling is not converted yet",
                code,
                dead,
            )
    storage.sort(key=lambda e: e["hydro_id"])
    return storage


def _equivalent_productivity_mw_per_m3s(hreg: pd.Series) -> float:
    """``ρ_eq = ρ_esp · h_net`` for one registry row.

    The gross head is the volume-averaged cota over the operating range minus
    the mean tailrace level, with the registry's hydraulic-loss model applied
    — the same construction the other converter family uses for its
    constant-productivity plants. Factored out so
    :func:`convert_energy_productivity` (the penalties input) and
    :func:`convert_hydro_group_availability` (the B8 hydraulic cap) always
    agree on the same plant's ρ_eq.
    """
    v_min = float(hreg["volume_minimo"])
    v_max = float(hreg["volume_maximo"])
    rho_esp = float(hreg.get("produtibilidade_especifica", 0.0) or 0.0)
    cf = float(hreg.get("canal_fuga_medio", 0.0) or 0.0)
    tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
    perdas = float(hreg.get("perdas", 0.0) or 0.0)
    h_gross = _mean_cota_over_volume(hreg, v_min, v_max) - cf
    h_net = max(_apply_hydraulic_loss(h_gross, tipo_perda, perdas), 0.0)
    return rho_esp * h_net


def convert_energy_productivity(
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
) -> pa.Table:
    """Per-plant constant equivalent productivity (all stages).

    ``ρ_eq = ρ_esp · h_net`` — see :func:`_equivalent_productivity_mw_per_m3s`.
    """
    hydro_ids: list[int] = []
    values: list[float] = []
    for code in id_map.hydro_codes:
        hreg = hidr.loc[code]
        hydro_ids.append(id_map.hydro_id(code))
        values.append(_equivalent_productivity_mw_per_m3s(hreg))

    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array([None] * len(hydro_ids), type=pa.int32()),
            "equivalent_productivity_mw_per_m3s": pa.array(values, type=pa.float64()),
            # Present-but-null: the reader requires the full column set.
            "reference_outflow_m3s": pa.array(
                [None] * len(hydro_ids), type=pa.float64()
            ),
            "specific_productivity_mw_per_m3s_per_m": pa.array(
                [None] * len(hydro_ids), type=pa.float64()
            ),
        }
    )


#: Mirrors ``emission_checks._ENVELOPE_TOLERANCE`` (cobre-io's relative
#: envelope tolerance) — used only to decide whether a computed availability
#: value sits below the declared envelope past float noise, not to gate a
#: cobre rule (that mirror lives in ``emission_checks.check_group_bound_envelope``).
_SPARSITY_TOLERANCE = 1e-9


def _below_envelope(value: float, envelope: float) -> bool:
    """Whether *value* is below *envelope* past relative float noise."""
    return value < envelope - _SPARSITY_TOLERANCE * max(abs(envelope), 1.0)


def _single_group_factor_rows(records: pd.DataFrame | None) -> dict[int, pd.Series]:
    """``{code: row}`` for a maintenance/availability register (``MP``/``FD``),
    keeping only single-group plants — mirrors the shipped rule test's own
    ``drop_duplicates("codigo_usina", keep=False)`` idiom, which also drops
    Itaipu's two per-frequency rows (handled separately, per group, by
    :func:`convert_hydro_group_availability`'s own split-plant branch).
    """
    if records is None or records.empty:
        return {}
    single = records.drop_duplicates("codigo_usina", keep=False)
    return {int(row["codigo_usina"]): row for _, row in single.iterrows()}


def _stage_register_factor(
    row: pd.Series, prefix: str, code: int, stage_index: int
) -> float:
    """``row[f"{prefix}_{stage_index + 1}"]`` (1-based register columns).

    Raises loudly if *row* is present but the declared stage's column is
    missing or blank — a malformed register is never silently defaulted;
    only a plant with *no* row at all defaults to factor ``1.0`` (the
    caller's job, not this helper's).
    """
    column = f"{prefix}_{stage_index + 1}"
    if column not in row.index or pd.isna(row[column]):
        raise ValueError(
            f"plant {code}: {prefix} register has no value for stage "
            f"{stage_index} (column {column!r} missing or blank)"
        )
    return float(row[column])


@dataclass(frozen=True)
class AvailabilityDeltaRow:
    """One single-group ``(hydro, stage)`` where the B8 hydraulic ceiling
    (``ρ_eq · q_max``) binds below the ``installed × MP × FD`` availability —
    the "accepted cost" B8 requires measuring and reporting, not assuming
    empty.
    """

    code: int
    name: str
    stage_id: int
    hydraulic_mw: float
    availability_mw: float

    @property
    def pct_under(self) -> float:
        """Percentage the hydraulic ceiling sits below the availability value."""
        return (self.availability_mw - self.hydraulic_mw) / self.availability_mw * 100.0


def _availability_bound_entry(
    q_g: float,
    hydraulic_mw: float,
    availability_mw: float,
    q_envelope: float,
    p_envelope: float,
) -> GroupBoundEntry | None:
    """One ``(hydro/group, stage)``'s B8 overlay entry, or ``None`` if
    neither bound column falls below that group's own declared envelope.

    ``max_generation_mw`` is ``min(hydraulic_mw, availability_mw)`` when
    that falls below *p_envelope* (the existing B8 behaviour);
    ``max_turbined_m3s`` is *q_g* — the per-stage AC-adjusted rated flow —
    when it falls below *q_envelope* (new: a mid-horizon machine-set shrink
    lowers the turbined-flow cap too, not just generation). Shared by the
    single-group and Itaipu per-conjunto-group emission loops in
    :func:`convert_hydro_group_availability`.
    """
    emitted = min(hydraulic_mw, availability_mw)
    max_generation_mw = emitted if _below_envelope(emitted, p_envelope) else None
    max_turbined_m3s = q_g if _below_envelope(q_g, q_envelope) else None
    if max_generation_mw is None and max_turbined_m3s is None:
        return None
    return GroupBoundEntry(
        max_generation_mw=max_generation_mw, max_turbined_m3s=max_turbined_m3s
    )


def convert_hydro_group_availability(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    effective: EffectiveCadastro,
) -> tuple[dict[tuple[int, int, int], GroupBoundEntry], list[AvailabilityDeltaRow]]:
    """B8 per-group per-stage available capacity.

    For the single-group majority (every plant but Itaipu):
    ``availability_mw(stage) = installed(stage) × MP(stage) × FD(stage)`` —
    ``installed(stage)`` is the per-stage AC-adjusted rated power
    (:func:`_compute_max_turbined_rated_ac_adjusted`, sourced from
    *effective*'s machine-set view), a missing ``MP``/``FD`` register
    defaults its factor to ``1.0`` — capped by the per-stage hydraulic
    ceiling ``ρ_eq · q(stage)`` (the same ρ_eq :func:`convert_energy_
    productivity` emits; ``q(stage)`` the per-stage AC-adjusted rated flow).
    Both the emitted ``max_generation_mw`` and the emitted
    ``max_turbined_m3s`` are sparse overlays against the group's own
    declared envelope (:func:`_rated_envelope`) — a mid-horizon machine-set
    shrink lowers *both* the availability ceiling and the turbined-flow cap
    (:func:`_availability_bound_entry`). Defensively, any *other* plant
    whose ``MP``/``FD`` register carries more than one row is also dropped
    from this path (today only Itaipu, code 66, does).

    **Tracked E5 fidelity gap (user-confirmed 2026-08-07):** ``installed``/
    the flow cap above is always the rated (un-derated) unit-flow sum, with
    no per-stage-head adjustment — see the module docstring; the
    source-model side's head-aware engolimento correction is not ported
    here.

    For Itaipu specifically, the same B8 formula is computed **per
    conjunto-backed group** rather than per plant —
    ``availability_mw(g, stage) = installed_g(stage) × MP_g(stage) ×
    FD_g(stage)``, ``g`` in ``{0, 1}`` (frequencies sorted ascending,
    matching :func:`_build_split_unit_groups`'s group ids), each register
    row frequency-matched via :func:`_frequency_row` (no defaulting to
    ``1.0`` — the split plant declares both rows) — capped by that same
    group's own per-stage hydraulic ceiling ``ρ_eq · q_g(stage)`` (``ρ_eq``
    is per-plant, ``q_g(stage)`` per-conjunto). Itaipu's binding rows are
    *not* folded into the returned ``deltas`` list (which stays
    single-group-only, preserving the shared B8 Diagnostic's pre-existing
    count) — its own accepted-cost delta is measured and pinned directly in
    ``tests/test_decomp_availability_rule.py`` per the ticket's own
    "reconstruct independently, never read the converter's own
    intermediate" discipline.

    Returns the ``(hydro_id, hydro_unit_group_id, stage_id) ->
    GroupBoundEntry`` mapping ticket-025's ``convert_hydro_unit_group_bounds``
    consumes unchanged — populated *sparsely*, only where an emitted value
    falls below that group's own declared envelope, the same "only where it
    differs" convention every sibling emitter uses — plus the list of
    single-group ``(hydro, stage)`` rows where the hydraulic ceiling actually
    bound below availability, B8's own attached measurement obligation. The
    caller reports that list as a :class:`~cobre_bridge.diagnostics.Diagnostic`.
    """
    mp = dadger.mp(df=True)
    fd = dadger.fd(df=True)
    mp_by_code = _single_group_factor_rows(mp)
    fd_by_code = _single_group_factor_rows(fd)

    values: dict[tuple[int, int, int], GroupBoundEntry] = {}
    deltas: list[AvailabilityDeltaRow] = []
    for code in id_map.hydro_codes:
        hreg = hidr.loc[code]
        hydro_id = id_map.hydro_id(code)
        rho_eq = _equivalent_productivity_mw_per_m3s(hreg)
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(hreg, code, effective, mp, fd)
            for i, frequency in enumerate(frequencies):
                conjunto_index = i + 1
                q_envelope, p_envelope = _conjunto_rated_envelope(
                    hreg, code, conjunto_index, effective
                )
                mp_row = _frequency_row(mp, code, frequency)
                fd_row = _frequency_row(fd, code, frequency)

                for stage in calendar:
                    q_g, p_g = _conjunto_rated_ac_adjusted(
                        hreg, code, conjunto_index, effective, stage.index
                    )
                    hydraulic_mw = rho_eq * q_g
                    mp_factor = _stage_register_factor(
                        mp_row, "manutencao", code, stage.index
                    )
                    fd_factor = _stage_register_factor(
                        fd_row, "fator", code, stage.index
                    )
                    availability_mw = p_g * mp_factor * fd_factor
                    entry = _availability_bound_entry(
                        q_g, hydraulic_mw, availability_mw, q_envelope, p_envelope
                    )
                    if entry is not None:
                        values[(hydro_id, i, stage.index)] = entry
            continue
        name = str(hreg["nome_usina"]).strip()
        q_envelope, p_envelope = _rated_envelope(hreg, code, effective)
        mp_row = mp_by_code.get(code)
        fd_row = fd_by_code.get(code)

        for stage in calendar:
            q_g, p_g = _compute_max_turbined_rated_ac_adjusted(
                hreg, code, effective, stage.index
            )
            hydraulic_mw = rho_eq * q_g
            mp_factor = (
                1.0
                if mp_row is None
                else _stage_register_factor(mp_row, "manutencao", code, stage.index)
            )
            fd_factor = (
                1.0
                if fd_row is None
                else _stage_register_factor(fd_row, "fator", code, stage.index)
            )
            availability_mw = p_g * mp_factor * fd_factor

            if hydraulic_mw < availability_mw:
                deltas.append(
                    AvailabilityDeltaRow(
                        code=code,
                        name=name,
                        stage_id=stage.index,
                        hydraulic_mw=hydraulic_mw,
                        availability_mw=availability_mw,
                    )
                )

            entry = _availability_bound_entry(
                q_g, hydraulic_mw, availability_mw, q_envelope, p_envelope
            )
            if entry is not None:
                values[(hydro_id, 0, stage.index)] = entry

    return values, deltas


def convert_production_models(id_map: DecompIdMap) -> dict:
    """Constant-productivity production models for every operated plant."""
    return {
        "$schema": _PRODUCTION_MODELS_SCHEMA_URL,
        "production_models": [
            {
                "hydro_id": id_map.hydro_id(code),
                "selection_mode": "stage_ranges",
                "stage_ranges": [
                    {
                        "start_stage_id": 0,
                        "end_stage_id": None,
                        "model": "constant_productivity",
                    }
                ],
            }
            for code in id_map.hydro_codes
        ],
    }
