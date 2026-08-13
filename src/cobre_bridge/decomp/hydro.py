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

**Head-aware ``max_turbined_m3s`` (ticket-017, E5 headline, validated
2026-08-08).** The emitted ``max_turbined_m3s`` is the head-corrected
engolimento, not the plain rated sum: :func:`_head_corrected_envelope`/
:func:`_conjunto_head_corrected_envelope` mirror the source-model side's
``converters.hydro._compute_max_turbined_head_corrected`` — derating the
rated flow by the turbine affinity law ``(h_op / h_nom) ** k_turb`` and
capping it at ``Σ n·p_nom / ρ_eq`` — using ticket-013's per-stage
``h_op = ρ_eq / ρ_esp``. Two DECOMP-specific corrections vs the shared
helper, both validated against ``relato.rv3``'s ``Qtur Maxima``: **no**
TEIF/IP availability derating (availability lives on the MP×FD/B8 path,
:func:`convert_hydro_group_availability`) and **no** ``tipo_regulacao``
head-branch (the full-range-mean ``h_op`` fits every class). Overall median
error ~1.3%; run-of-river ~0.4% (fixes the rated-sum overshoot); reservoir
plants carry a validated **~4.8% residual undershoot** (low impact —
reservoirs are energy-, not flow-, limited). ``max_generation_mw`` stays on
the rated (un-derated, TEIF/IP-free) envelope throughout — only
``max_turbined_m3s`` is head-aware, and it composes cleanly with the
per-stage machine-set envelope here, the two adjustments being orthogonal
(one scales the rated flow by head, the other scales it by which conjuntos
are effective). The source model's own effective-head override
(``AC ALTEFE``) is **not** consumed — idecomp exposes no value accessor for
it, only identifying/timing fields — so every plant's base ``hidr`` nominal
head is used regardless of an ``ALTEFE`` declaration; see the
``TRACKED COBRE-GAP WORKAROUND`` in :func:`convert_hydros`.

The one plant whose maintenance and availability registers are declared
*per generating-unit group* rather than per plant (Itaipu, code 66, carries
two ``MP``/``FD`` ``frequencia`` rows) gets two conjunto-backed unit groups
instead of the usual single mirror group — see
:func:`_build_split_unit_groups`. Scope is the ratified loop-closing
milestone: faithful registry, cascade, capability, initial storage, and (as
of tickets 013/017, E5) head/productivity, including head-aware
engolimento — with everything whose faithful treatment is still gated on
later features left unconverted: travel time (``VI`` — a separate dadger
register, not an ``AC`` class) and the evaporation/tailrace-polynomial
models (``COTVAZ``/``COTARE``/``COFEVA`` — ``AC`` classes with no DECOMP
consumer). The ``AC`` families' own per-deck coverage (which of them a
given deck actually declares) is reported by ``check decomp``
(:mod:`cobre_bridge.decomp.preflight`); ``VI``, not being an ``AC``
register, is documented only here.

**Effective (post-``AC``) cascade topology and inflow gauge (ticket-014,
E5, OQ-4 resolved).** The cascade walk (:func:`_downstream_operated`,
shared by this module's own ``downstream_id`` entity field and
``scenarios.py``'s incremental-inflow attribution) and the inflow-gauge
attribution both read :class:`~cobre_bridge.decomp.cadastro.
EffectiveCadastro`'s ``downstream_plant``/``inflow_gauge`` accessors — the
post-``AC NUMJUS``/``NUMPOS`` link/gauge — rather than the base ``hidr``
columns directly. Absent an override the effective value equals base, so
this is byte-identical to the pre-ticket behaviour for every plant with no
``NUMJUS``/``NUMPOS`` row. ``AC JUSENA`` (a downstream-**energy** coupling,
not a water-routing link) and ``AC NPOSNW`` (the other source family's own
inflow gauge) are deliberately **not** ingested — no DECOMP consumer; their
per-deck coverage is reported by ``check decomp`` alongside ``COTVAZ``/
``COTARE``/``COFEVA``.

``UH`` rows without an initial volume (the coupling-only registrations)
are excluded from the operated set and reported.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
from idecomp.decomp.modelos.dadger import ACALTEFE
from inewave.newave import Hidr

from cobre_bridge.converters.hydro import (
    _KTURB_BY_TIPO_TURBINA,
    _PRODUCTION_MODELS_SCHEMA_URL,
    _SCHEMA_URL,
    _apply_hydraulic_loss,
    _fpha_efficiency,
    build_mirror_unit_group,
)
from cobre_bridge.decomp.cadastro import effective_storage_range, storage_envelope
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

    Rows without one (coupling-only registrations) are excluded — their
    terminal-value treatment is the boundary importer's D3 territory, not the
    registry's. :func:`convert_hydros` reports the excluded set (by name) once.
    """
    uh = dadger.uh(df=True)
    if uh is None or uh.empty:
        raise ValueError("the deck has no UH records; cannot convert hydros")
    return uh[uh["volume_inicial"].notna()]


def _coupling_only_codes(dadger: Dadger) -> list[int]:
    """Operated-set-excluded ``UH`` codes — registrations with no initial volume."""
    uh = dadger.uh(df=True)
    if uh is None or uh.empty:
        return []
    excluded = uh[uh["volume_inicial"].isna()]
    return sorted(int(c) for c in excluded["codigo_usina"])


def _hydro_display_name(hidr: pd.DataFrame, code: int) -> str:
    """``'NAME (code)'`` for a plant, or just the code if the registry lacks it."""
    try:
        name = str(hidr.loc[code, "nome_usina"]).strip()
    except KeyError:
        return str(code)
    return f"{name} ({code})"


def _downstream_operated(
    effective: EffectiveCadastro,
    code: int,
    operated: set[int],
    *,
    stage_index: int = 0,
) -> int | None:
    """Walk the *effective* (post-``AC NUMJUS``) cascade to the next
    *operated* plant downstream, at *stage_index* (default the initial
    stage — the single stage-representative topology every caller uses,
    ticket-014).

    Non-operated intermediates are skipped through (their routing is
    instantaneous absence — the water continues down the declared chain);
    code 0 is the sink. If *code*'s own effective downstream link varies
    across stages (a temporal ``AC NUMJUS``), a tracked-gap warning is
    logged and the *stage_index* link is used for the whole horizon
    regardless — per-stage cascade topology is not modeled
    (:meth:`~cobre_bridge.decomp.cadastro.EffectiveCadastro.
    downstream_plant_varies`). A temporal override on a non-operated
    intermediate encountered mid-walk is not separately checked here (no
    rv3 row exercises it — see the module docstring); the pipeline's own
    relink diagnostic (:func:`cobre_bridge.decomp.pipeline.
    _topology_relink_diagnostic`) still surfaces a resulting cascade change
    via a fallback warning rather than dropping it silently.
    """
    if effective.downstream_plant_varies(code):
        _LOG.warning(
            "plant %d's downstream link (AC NUMJUS) varies across stages; "
            "per-stage cascade topology is not modeled (deferred fidelity) "
            "-- using the stage-%d effective link for the whole horizon",
            code,
            stage_index,
        )
    visited = {code}
    current = effective.downstream_plant(code, stage_index)
    while current != 0 and current not in operated:
        if current in visited or current not in effective.base.index:
            _LOG.warning(
                "cascade walk from plant %d hit an invalid link at %d; "
                "treating as a sink",
                code,
                current,
            )
            return None
        visited.add(current)
        current = effective.downstream_plant(current, stage_index)
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


def _conjunto_h_nom(hreg: pd.Series, conjunto_index: int) -> float:
    """The base ``hidr`` nominal head (``queda_nominal_conjunto_{i}``) for one
    1-based *conjunto_index*, or ``0.0`` when the column is absent or blank.

    ``0.0`` is the same sentinel :func:`_conjunto_head_corrected_ac_adjusted`/
    :func:`_head_corrected_max_turbined_ac_adjusted` treat as "no nominal
    head" (falls back to the rated flow for that conjunto, Requirement 1) —
    a genuine ``hidr.dat`` row always carries this column (the registry is
    byte-identical across the two source families), so a missing column only
    ever happens in a hand-built test fixture that doesn't exercise the head
    correction. The source model's own effective-head override
    (``AC ALTEFE``) is never consulted here — see the ``TRACKED COBRE-GAP
    WORKAROUND`` in :func:`convert_hydros`.
    """
    raw = hreg.get(f"queda_nominal_conjunto_{conjunto_index}")
    if raw is None or pd.isna(raw):
        return 0.0
    return float(raw)


def _operating_head(
    effective: EffectiveCadastro, code: int, stage_index: int, rho_eq: float
) -> float | None:
    """The per-stage operating head ``h_op = ρ_eq / ρ_esp``, or ``None`` when
    *rho_eq*, ``produtibilidade_especifica``, or the resulting ratio is
    non-positive.

    The single defensive gate every head-corrected envelope/overlay caller
    checks before trusting *h_op* — a ``None`` return means "fall back to
    the rated flow for this stage", never a meaningless or negative bound
    (Requirement/Error Handling). *rho_eq* is passed in rather than
    recomputed so a caller that already needs it for its own hydraulic-cap
    calculation (:func:`convert_hydro_group_availability`) calls
    :func:`_equivalent_productivity_mw_per_m3s` exactly once per stage.
    """
    if rho_eq <= 0.0:
        return None
    rho_esp = effective.value(code, "produtibilidade_especifica", stage_index)
    if rho_esp <= 0.0:
        return None
    h_op = rho_eq / rho_esp
    return h_op if h_op > 0.0 else None


def _conjunto_affinity_flow(
    n_q: float, h_nom: float, h_op: float, kturb: float
) -> float:
    """One conjunto's turbine-affinity-corrected flow ``n·q_nom·(h_op / h_nom)
    ** k_turb``, or the uncorrected ``n·q_nom`` when *h_nom* ``<= 0`` (no
    nominal head — falls back to the rated flow for that conjunto).

    The single place the affinity term is computed, shared by
    :func:`_conjunto_head_corrected_ac_adjusted` (per-conjunto power cap) and
    :func:`_head_corrected_max_turbined_ac_adjusted` (plant-wide power cap) so
    the two read as true mirrors — only the *capping* differs between them,
    not the affinity law itself.
    """
    if h_nom <= 0.0:
        return n_q
    return n_q * (h_op / h_nom) ** kturb


def _conjunto_head_corrected_ac_adjusted(
    hreg: pd.Series,
    code: int,
    conjunto_index: int,
    effective: EffectiveCadastro,
    stage_index: int,
    h_op: float,
    rho_eq: float,
) -> float:
    """The per-conjunto head-corrected flow at *stage_index*: ``min(n·q_nom·
    (h_op / h_nom) ** k_turb, n·p_nom / ρ_eq)``.

    ``(n·q_nom, n·p_nom)`` come from :func:`_conjunto_rated_ac_adjusted` (the
    same AC-effective building block the rated pair uses); ``h_nom`` is the
    base ``hidr`` ``queda_nominal_conjunto_{conjunto_index}``
    (:func:`_conjunto_h_nom`) — **not** the ``AC ALTEFE`` override, which has
    no idecomp value accessor — and ``k_turb`` is looked up from
    ``_KTURB_BY_TIPO_TURBINA`` by the plant's ``tipo_turbina`` (0.5 for
    Francis/Pelton, 0.2 for Kaplan). No TEIF/IP availability derating is
    applied (validated worse: 8.2% vs 1.3% median error vs ``Qtur Maxima`` —
    availability lives on the MP×FD/B8 path instead) and no
    ``tipo_regulacao`` head-branch (the full-range-mean *h_op* fits every
    class). ``h_nom ≤ 0`` skips the affinity correction for this conjunto
    (falls back to the rated ``n·q_nom``) but the power cap still applies;
    *h_op* and *rho_eq* are assumed already validated positive by the caller
    (:func:`_operating_head`). The per-conjunto (not plant-wide) power cap
    here is deliberate — used only for the split plant's own groups
    (:func:`_conjunto_head_corrected_envelope`,
    :func:`convert_hydro_group_availability`'s Itaipu branch), each of which
    only has its own conjunto's installed power to draw on.
    """
    n_q, n_p = _conjunto_rated_ac_adjusted(
        hreg, code, conjunto_index, effective, stage_index
    )
    h_nom = _conjunto_h_nom(hreg, conjunto_index)
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    kturb = _KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    affinity = _conjunto_affinity_flow(n_q, h_nom, h_op, kturb)
    return min(affinity, n_p / rho_eq)


def _head_corrected_max_turbined_ac_adjusted(
    hreg: pd.Series,
    code: int,
    effective: EffectiveCadastro,
    stage_index: int,
    h_op: float,
    rho_eq: float,
) -> float:
    """The plant-level head-corrected cap at *stage_index*: ``min(Σ_c
    n_c·q_eff_c, Σ_c n_c·p_nom_c / ρ_eq)`` — sum the per-conjunto affinity
    flow first, then apply **one** plant-wide power cap, over every conjunto
    effective at *stage_index* (``effective.machine_conjunto_count``,
    falling back to the ``hidr`` base ``numero_conjuntos_maquinas``).

    This is the non-split-plant counterpart of
    :func:`_conjunto_head_corrected_ac_adjusted`: it deliberately does
    **not** sum that function's own (per-conjunto-capped) return values —
    doing so would cap the power per conjunto, which only the split plant's
    own groups do. Instead it re-derives each conjunto's affinity flow the
    same way (``h_nom`` from :func:`_conjunto_h_nom`, ``k_turb`` from
    ``_KTURB_BY_TIPO_TURBINA``, no TEIF/IP, no ``tipo_regulacao`` branch —
    see :func:`_conjunto_head_corrected_ac_adjusted`'s docstring for the
    validated rationale) and sums both the affinity flow and the installed
    power across conjuntos before taking the single min. *h_op* and *rho_eq*
    are assumed already validated positive by the caller
    (:func:`_operating_head`).
    """
    n_sets = effective.machine_conjunto_count(code, stage_index)
    if n_sets is None:
        n_sets = int(hreg["numero_conjuntos_maquinas"])
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    kturb = _KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    sum_affinity = 0.0
    sum_p = 0.0
    for i in range(1, n_sets + 1):
        n_q, n_p = _conjunto_rated_ac_adjusted(hreg, code, i, effective, stage_index)
        h_nom = _conjunto_h_nom(hreg, i)
        sum_affinity += _conjunto_affinity_flow(n_q, h_nom, h_op, kturb)
        sum_p += n_p
    return min(sum_affinity, sum_p / rho_eq)


def _head_corrected_envelope(
    hreg: pd.Series, code: int, effective: EffectiveCadastro
) -> float:
    """Max-over-stages head-corrected ``max_turbined_m3s`` for the whole
    plant — the head-aware mirror of :func:`_rated_envelope`'s flow
    component (``max_generation_mw`` stays on the rated envelope; this
    function returns only the flow).

    Per stage, computes ``h_op = ρ_eq(stage) / ρ_esp(stage)`` via
    :func:`_operating_head` (``ρ_eq(stage)`` from
    :func:`_equivalent_productivity_mw_per_m3s`, the same per-stage value
    :func:`convert_hydro_group_availability`'s hydraulic ceiling uses) and,
    when that's positive, the head-corrected flow
    (:func:`_head_corrected_max_turbined_ac_adjusted`); a stage where
    *h_op*, ``ρ_esp``, or ``ρ_eq`` is non-positive falls back defensively to
    that stage's rated flow (:func:`_compute_max_turbined_rated_ac_adjusted`)
    rather than emit a meaningless bound.
    """
    per_stage: list[float] = []
    for stage in range(effective.n_stages):
        rho_eq = _equivalent_productivity_mw_per_m3s(effective, code, stage)
        h_op = _operating_head(effective, code, stage, rho_eq)
        if h_op is None:
            value = _compute_max_turbined_rated_ac_adjusted(
                hreg, code, effective, stage
            )[0]
        else:
            value = _head_corrected_max_turbined_ac_adjusted(
                hreg, code, effective, stage, h_op, rho_eq
            )
        per_stage.append(value)
    return max(per_stage)


def _conjunto_head_corrected_envelope(
    hreg: pd.Series, code: int, conjunto_index: int, effective: EffectiveCadastro
) -> float:
    """Max-over-stages head-corrected flow for one conjunto — the per-group
    mirror of :func:`_head_corrected_envelope`, used for the split plant's
    own groups (:func:`_build_split_unit_groups`) and their own B8 overlay
    (:func:`convert_hydro_group_availability`'s Itaipu branch).

    Same per-stage defensive fallback as :func:`_head_corrected_envelope`:
    a stage where :func:`_operating_head` returns ``None`` uses that
    conjunto's rated flow (:func:`_conjunto_rated_ac_adjusted`) instead.
    """
    per_stage: list[float] = []
    for stage in range(effective.n_stages):
        rho_eq = _equivalent_productivity_mw_per_m3s(effective, code, stage)
        h_op = _operating_head(effective, code, stage, rho_eq)
        if h_op is None:
            value = _conjunto_rated_ac_adjusted(
                hreg, code, conjunto_index, effective, stage
            )[0]
        else:
            value = _conjunto_head_corrected_ac_adjusted(
                hreg, code, conjunto_index, effective, stage, h_op, rho_eq
            )
        per_stage.append(value)
    return max(per_stage)


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
    group_bus_ids: Sequence[int],
    frequencies: list[float],
    effective: EffectiveCadastro,
) -> tuple[list[dict[str, object]], float, float]:
    """Conjunto-backed unit groups for the per-frequency split plant.

    Group id ``i`` (``i`` in ``0..len(frequencies)-1``, ascending frequency,
    so id 0 = the lowest — 50 Hz for Itaipu) is backed by hidr conjunto
    ``i + 1``: its ``max_generation_mw`` is *that* conjunto's own
    max-over-stages AC-adjusted rated envelope
    (:func:`_conjunto_rated_envelope`); its ``max_turbined_m3s`` is instead
    that conjunto's own head-corrected envelope
    (:func:`_conjunto_head_corrected_envelope`, ticket-017) — each group has
    only its own conjunto's installed power to draw on, so the power cap
    inside that function is per-group, never plant-wide. Group ``i`` sits on
    ``group_bus_ids[i]`` — a per-group bus, not necessarily the plant's own
    bus: :func:`convert_hydros` relocates Itaipu's 50 Hz group to the ``IV``
    transshipment bus while keeping its 60 Hz group on the plant's own bus.

    Raises
    ------
    ValueError
        Naming both counts, if *frequencies* and *group_bus_ids* disagree in
        length — a mis-wired split must fail loud, not silently colocate.

    Returns the groups plus their summed ``(max_turbined_m3s,
    max_generation_mw)``: :func:`convert_hydros` declares the plant's own
    entity envelope as exactly this sum, rather than recomputing it
    independently, so cobre rule 41 holds by construction even though the
    two groups' own per-stage machine-set changes (if any) need not peak on
    the same stage.
    """
    if len(frequencies) != len(group_bus_ids):
        raise ValueError(
            f"plant {code}: {len(frequencies)} split frequencies but "
            f"{len(group_bus_ids)} per-group buses were supplied"
        )
    groups: list[dict[str, object]] = []
    total_turbined = 0.0
    total_generation = 0.0
    for i in range(len(frequencies)):
        conjunto_index = i + 1
        _, p = _conjunto_rated_envelope(hreg, code, conjunto_index, effective)
        q = _conjunto_head_corrected_envelope(hreg, code, conjunto_index, effective)
        groups.append(
            build_mirror_unit_group(
                name=name,
                bus_id=group_bus_ids[i],
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


#: ``hidr.dat``'s 12 monthly reservoir-evaporation columns in calendar order
#: (Jan..Dec) — cobre's ``evaporation.coefficients_mm`` is index 0 = January.
_EVAPORATION_MONTH_COLUMNS = (
    "evaporacao_JAN",
    "evaporacao_FEV",
    "evaporacao_MAR",
    "evaporacao_ABR",
    "evaporacao_MAI",
    "evaporacao_JUN",
    "evaporacao_JUL",
    "evaporacao_AGO",
    "evaporacao_SET",
    "evaporacao_OUT",
    "evaporacao_NOV",
    "evaporacao_DEZ",
)


def _evaporation_flag_codes(dadger: Dadger) -> set[int]:
    """Plant codes whose ``UH`` register sets the reservoir-evaporation flag.

    DECOMP models a plant's evaporation only when its ``UH`` ``evaporacao``
    flag is set; a plant absent from this set is emitted with no evaporation
    regardless of the coefficients ``hidr.dat`` may carry.
    """
    uh = dadger.uh(df=True)
    if uh is None or "evaporacao" not in uh.columns or "codigo_usina" not in uh.columns:
        return set()
    return set(uh.loc[uh["evaporacao"].fillna(0) != 0, "codigo_usina"].astype(int))


def _evaporation_coefficients_mm(hidr: pd.DataFrame, code: int) -> list[float] | None:
    """The plant's 12 monthly evaporation coefficients [mm/month], Jan..Dec,
    from ``hidr.dat`` — or ``None`` when the plant is absent or every month is
    zero (nothing to model).

    cobre (>= 0.14) computes the evaporation-outflow model internally from
    these monthly rates and the reservoir area geometry
    (``hydro_geometry.parquet``, emitted by the FPHA path), scaling each stage
    to its ``stage_hours / month_hours`` share of the calendar month — the C11
    fix (pre-0.14 deposited a whole month's evaporation on every stage). The
    bridge supplies only the monthly mm rates.
    """
    if code not in hidr.index or not all(
        column in hidr.columns for column in _EVAPORATION_MONTH_COLUMNS
    ):
        return None
    row = hidr.loc[code]
    coefficients = [float(row[column]) for column in _EVAPORATION_MONTH_COLUMNS]
    if not any(coefficients):
        return None
    return coefficients


def convert_hydros(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    start_date: date,
    effective: EffectiveCadastro,
    travel_time_hours: dict[int, float] | None = None,
    fpha_codes: set[int] | None = None,
) -> dict:
    """Build ``hydros.json`` for the operated plants.

    *fpha_codes* (from :func:`cobre_bridge.decomp.fpha.fpha_eligible_codes`)
    selects the plants emitted with cobre's computed-FPHA generation model:
    their ``generation.model`` is ``"fpha"`` and they carry the turbine
    ``efficiency`` (η = ρ_esp / K), the ``specific_productivity_mw_per_m3s_per_m``
    cobre derives ρ_eq from, and an inline constant ``tailrace`` (the
    ``canal_fuga_medio`` fallback used when ``tailrace_curves.parquet`` carries
    no family for the plant). Plants absent from it — or the whole set being
    ``None`` — keep the constant-productivity model whose ρ_eq rides in
    ``hydro_energy_productivity.parquet``.

    *travel_time_hours* (``{plant code: hours}``, from
    :func:`cobre_bridge.decomp.travel_time.convert_travel_time`) stamps the
    ``VI`` water travel time onto each arc plant's entry; a plant absent from it
    — or the whole map being ``None`` — emits no ``travel_time_hours`` key
    (cobre defaults it to instantaneous). The key is emitted only when the plant
    also has a downstream arc for the delay to act on.

    ``max_generation_mw`` is the installed (un-derated, ``TEIF``/``IP``-free)
    max-over-stages envelope of the AC-adjusted rated unit-power sum,
    sourced per stage from *effective*'s machine-set view
    (:func:`_rated_envelope`, mirroring :func:`storage_envelope`'s
    outer-bound construction for the reservoir block) — some plants' *true*
    in-service machine configuration differs from ``hidr.dat``'s nameplate
    conjunto sum, and/or changes mid-horizon. ``max_turbined_m3s`` is
    instead the max-over-stages envelope of the **head-corrected** engolimento
    (:func:`_head_corrected_envelope`, ticket-017 — see the module docstring
    for the validated formula and accuracy). A stage whose effective
    capacity (installed power or head-corrected flow) drops below its own
    envelope gets a sparse per-stage overlay instead
    (:func:`convert_hydro_group_availability`), never a change to the
    declared envelope itself. The production model is constant
    productivity, with the value emitted separately
    (:func:`convert_energy_productivity`).

    **``ACALTEFE`` tracked gap:** a plant declaring an ``AC ALTEFE``
    effective-head override is logged (not consumed — see the
    ``TRACKED COBRE-GAP WORKAROUND`` below) and conversion proceeds with the
    base ``hidr`` nominal head.

    Every plant declares one mirror unit group, except the per-frequency
    split plant (Itaipu, code 66), which declares two conjunto-backed groups
    instead (:func:`_build_split_unit_groups`), whose summed
    ``max_turbined_m3s``/``max_generation_mw`` *are* the entity envelope by
    construction — cobre rule 41 holds even though the two groups' own
    per-stage machine-set changes (if any) need not peak on the same stage,
    nor their head-corrected flows peak on the same stage as their rated
    power. Itaipu's 50 Hz group (group id 0) is placed on
    ``id_map.transhipment_bus_id`` — the ``IV`` transshipment bus — rather
    than the plant's own submercado bus; the 60 Hz group (group id 1) stays
    on the plant's own bus. This relocation is unconditional whenever Itaipu
    is operated (ticket-006). The entity ``reservoir`` block is the plant's
    outer per-stage storage envelope (:func:`storage_envelope`), so
    per-stage bound overrides
    (:func:`cobre_bridge.decomp.bounds.convert_storage_bounds`)
    always sit inside it. Per-family ``AC`` coverage is reported by
    ``check decomp`` (:mod:`cobre_bridge.decomp.preflight`), not logged here.
    """
    operated = _operated_uh(dadger)
    coupling_only = _coupling_only_codes(dadger)
    if coupling_only:
        _LOG.warning(
            "%d UH registration(s) without an initial volume excluded from "
            "the operated set (coupling-only): %s",
            len(coupling_only),
            ", ".join(_hydro_display_name(hidr, c) for c in coupling_only),
        )
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
    evaporation_codes = _evaporation_flag_codes(dadger)
    n_runofriver_collapsed = 0
    for code in id_map.hydro_codes:
        if code not in hidr.index:
            raise ValueError(f"UH plant {code} is not in the hydro registry")
        hreg = hidr.loc[code]
        name = str(hreg["nome_usina"]).strip()
        is_fpha = fpha_codes is not None and code in fpha_codes
        downstream = _downstream_operated(effective, code, operated_codes)
        min_storage_hm3, max_storage_hm3 = storage_envelope(effective, code)
        if str(hreg.get("tipo_regulacao", "")).strip() == "D" and float(
            hreg["volume_minimo"]
        ) != float(hreg["volume_maximo"]):
            n_runofriver_collapsed += 1
        bus_id = id_map.bus_id(int(hreg["submercado"]))
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(hreg, code, effective, mp, fd)
            # Unconditional 50 Hz -> IV relocation (ticket-006): Itaipu's 50 Hz
            # unit group (group id 0, the lower of the two ascending
            # frequencies) is moved to the transshipment bus so cobre's
            # HydroGeneration{bus} selector can separate the two groups; the
            # 60 Hz group (group id 1) stays on the plant's own SE bus
            # (already computed above as `bus_id`).
            se_bus = bus_id
            iv_bus = id_map.transhipment_bus_id
            unit_groups, max_turbined, max_generation = _build_split_unit_groups(
                hreg, code, name, [iv_bus, se_bus], frequencies, effective
            )
        else:
            _, max_generation = _rated_envelope(hreg, code, effective)
            max_turbined = _head_corrected_envelope(hreg, code, effective)
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
                "model": "fpha" if is_fpha else "constant_productivity",
                "min_turbined_m3s": 0.0,
                "max_turbined_m3s": max_turbined,
                "min_generation_mw": 0.0,
                "max_generation_mw": max_generation,
            },
            "unit_groups": unit_groups,
        }
        if is_fpha:
            rho_esp = effective.value(code, "produtibilidade_especifica", 0)
            entry["specific_productivity_mw_per_m3s_per_m"] = rho_esp
            entry["efficiency"] = {
                "type": "constant",
                "value": _fpha_efficiency(rho_esp, name),
            }
            # Inline constant tailrace = mean canal de fuga: cobre's FPHA
            # fallback for a plant whose tailrace_curves.parquet family is
            # absent (all this deck's FPHA plants do carry a family, so it is
            # only a safety net).
            cf = effective.value(code, "canal_fuga_medio", 0)
            if cf > 0.0:
                entry["tailrace"] = {"type": "polynomial", "coefficients": [cf]}
            # Penstock hydraulic losses — a computed FPHA requires the field
            # (cobre rejects an FPHA plant without it). tipo_perda 1 = a % of
            # gross head (factor); 2 = constant metres; anything else / no loss
            # emits an explicit lossless factor so the required field is present.
            perdas = effective.value(code, "perdas", 0)
            tipo_perda = int(effective.base.loc[code, "tipo_perda"])
            if tipo_perda == 1 and perdas > 0.0:
                entry["hydraulic_losses"] = {"type": "factor", "value": perdas / 100.0}
            elif tipo_perda == 2 and perdas > 0.0:
                entry["hydraulic_losses"] = {"type": "constant", "value_m": perdas}
            else:
                entry["hydraulic_losses"] = {"type": "factor", "value": 0.0}
        if travel_time_hours and code in travel_time_hours and downstream is not None:
            entry["travel_time_hours"] = travel_time_hours[code]
        if code in evaporation_codes:
            evaporation_mm = _evaporation_coefficients_mm(hidr, code)
            if evaporation_mm is not None:
                entry["evaporation"] = {"coefficients_mm": evaporation_mm}
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

    if n_runofriver_collapsed:
        _LOG.info(
            "collapsed %d run-of-river ('D') plant(s) storage range to their "
            "reference volume (volume_referencia): the hidr registry's "
            "(volume_minimo, volume_maximo) band would otherwise emit a "
            "phantom weekly storage buffer the plant does not have",
            n_runofriver_collapsed,
        )

    # TRACKED COBRE-GAP WORKAROUND (idecomp limitation, not cobre's): the
    # source model's AC ALTEFE register overrides a conjunto's effective
    # head, but idecomp's typed accessor exposes only the identifying/timing
    # columns for it, no value column — so it cannot be folded into the
    # head-corrected max_turbined above. Every plant keeps its base hidr
    # queda_nominal_conjunto_* nominal head regardless of an ALTEFE
    # declaration. Remove this warning (and wire the override into
    # _conjunto_h_nom) once idecomp exposes an ALTEFE value accessor.
    altefe = dadger.ac(codigo_usina=None, modificacao=ACALTEFE, df=True)
    if isinstance(altefe, pd.DataFrame) and not altefe.empty:
        n_altefe = altefe["codigo_usina"].nunique()
        _LOG.warning(
            "%d plant(s) declare an AC ALTEFE effective-head override that "
            "cannot be consumed (idecomp exposes no value accessor for "
            "ALTEFE); head-corrected max_turbined uses the base hidr "
            "nominal head for these plants",
            n_altefe,
        )

    return {"$schema": _SCHEMA_URL, "hydros": hydros}


def _initial_volume_hm3(effective: EffectiveCadastro, code: int, pct: float) -> float:
    """The ``UH`` ``volume_inicial`` percentage resolved to hm³ at stage 0.

    ``pct`` is a percentage of the *initial stage's* effective useful volume,
    not the plant's outer envelope, so the range is read from
    :func:`~cobre_bridge.decomp.cadastro.effective_storage_range` at stage
    ``0``. A run-of-river (``D``) plant's stage-0 range is already the
    single-point collapse ``(vol_ref, vol_ref)`` (ticket-018), so its initial
    value is ``vol_ref`` regardless of *pct*. Shared by
    :func:`convert_initial_storage` (the initial condition) and
    :func:`_operated_initial_volumes` (the generation-productivity anchor) so
    both resolve a plant's initial volume identically.
    """
    v_min, v_max = effective_storage_range(effective, code, 0)
    return min(max(v_min + (pct / 100.0) * (v_max - v_min), v_min), v_max)


def _operated_initial_volumes(
    dadger: Dadger, effective: EffectiveCadastro
) -> dict[int, float]:
    """``{code: initial reservoir volume hm³}`` for every operated ``UH`` plant.

    The volume :func:`convert_energy_productivity` anchors each plant's
    generation productivity on — the same value :func:`convert_initial_storage`
    emits as the initial condition (both via :func:`_initial_volume_hm3`).
    """
    return {
        int(row["codigo_usina"]): _initial_volume_hm3(
            effective, int(row["codigo_usina"]), float(row["volume_inicial"])
        )
        for _, row in _operated_uh(dadger).iterrows()
    }


def convert_initial_storage(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
) -> list[dict]:
    """Initial reservoir volumes from ``UH`` (% of useful → hm³).

    The percentage is resolved by :func:`_initial_volume_hm3` against the
    stage-0 effective useful volume (run-of-river plants collapse to their
    reference volume — ticket-018).
    """
    operated = _operated_uh(dadger)
    storage: list[dict] = []
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        value = _initial_volume_hm3(effective, code, float(row["volume_inicial"]))
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


def _eval_cota_from_coeffs(coeffs: Sequence[float], volume_hm3: float) -> float:
    """Evaluate the upstream cota polynomial ``cota(V) = Σ a_i · V^i`` at *V*
    from an explicit 5-coefficient list, ordered ``ordem 0..4``.

    The coefficient-list counterpart of ``converters.hydro.
    _evaluate_cota_polynomial``, which reads the same five coefficients off a
    base ``hidr`` row directly — this variant exists so the per-stage
    *effective* polynomial (:meth:`~cobre_bridge.decomp.cadastro.
    EffectiveCadastro.cota_polynomial`, base or ``AC COTVOL``-overridden) can
    be evaluated without mutating a per-stage copy of the base row.
    """
    v = volume_hm3
    a = coeffs
    return a[0] + a[1] * v + a[2] * v * v + a[3] * v**3 + a[4] * v**4


def _mean_cota_from_coeffs(coeffs: Sequence[float], v_lo: float, v_hi: float) -> float:
    """Volume-averaged upstream cota over ``[v_lo, v_hi]`` from an explicit
    5-coefficient list.

    The same analytic quartic integral as ``converters.hydro.
    _mean_cota_over_volume``, taking *coeffs* directly rather than a ``hidr``
    row — see :func:`_eval_cota_from_coeffs` for why.
    """
    if v_hi <= v_lo:
        return _eval_cota_from_coeffs(coeffs, v_lo)
    a = coeffs

    def antideriv(v: float) -> float:
        return (
            a[0] * v
            + a[1] * v * v / 2.0
            + a[2] * v**3 / 3.0
            + a[3] * v**4 / 4.0
            + a[4] * v**5 / 5.0
        )

    return (antideriv(v_hi) - antideriv(v_lo)) / (v_hi - v_lo)


def _equivalent_productivity_mw_per_m3s(
    effective: EffectiveCadastro,
    code: int,
    stage_index: int,
    reference_volume_hm3: float | None = None,
) -> float:
    """``ρ_eq = ρ_esp · h_net`` for one plant at one stage.

    The gross head is an upstream cota minus the mean tailrace level, with the
    hydraulic-loss model applied. The cota anchor depends on
    *reference_volume_hm3*:

    * ``None`` (the engolimento anchor) — the volume-averaged cota over the
      **full** operating range (``volume_minimo``..``volume_maximo``). This is
      the value the head-aware ``max_turbined`` envelope and
      :func:`convert_hydro_group_availability`'s B8 hydraulic cap consume,
      validated against the source model's ``Qtur Maxima``; it stays on the
      full-range mean so that path is unchanged.
    * a volume (the **generation** anchor) — the cota at that single volume.
      :func:`convert_energy_productivity` passes the plant's initial reservoir
      volume because the source model anchors each plant's hydro production
      function (FPHA) at the initial volume ± a fit window (manual §3.4.6.4,
      ``FP`` fields 10-11), **not** the full-range mean. For a reservoir
      starting far from mid-range the two heads differ materially — validated
      against ``dec_oper_usih`` effective productivity, the initial-volume
      anchor cuts the reservoir median error 4.3% -> 2.6% and the worst case
      (BARRA BONITA, 87% full) 22.7% -> 5.9%.

    Every other input is read at *stage_index* through *effective*, so an
    ``AC PROESP``/``PERHID``/``JUSMED``/``COTVOL``/``VOLMIN``/``VOLMAX``
    override shifts this stage's ρ_eq; ``tipo_perda`` carries no ``AC`` register
    of its own and stays a base read.
    """
    v_min = effective.value(code, "volume_minimo", stage_index)
    v_max = effective.value(code, "volume_maximo", stage_index)
    rho_esp = effective.value(code, "produtibilidade_especifica", stage_index)
    cf = effective.value(code, "canal_fuga_medio", stage_index)
    perdas = effective.value(code, "perdas", stage_index)
    tipo_perda = int(effective.base.loc[code, "tipo_perda"])
    coeffs = effective.cota_polynomial(code, stage_index)
    if reference_volume_hm3 is None:
        cota = _mean_cota_from_coeffs(coeffs, v_min, v_max)
    else:
        cota = _eval_cota_from_coeffs(coeffs, reference_volume_hm3)
    h_gross = cota - cf
    h_net = max(_apply_hydraulic_loss(h_gross, tipo_perda, perdas), 0.0)
    return rho_esp * h_net


def convert_energy_productivity(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    initial_volumes: dict[int, float] | None = None,
    exclude_codes: set[int] | None = None,
) -> pa.Table:
    """Per-plant equivalent generation productivity, anchored at initial volume.

    ``ρ_eq = ρ_esp · h_net`` with the head taken at the plant's **initial
    reservoir volume** — the source model's FPHA fit anchor (see
    :func:`_equivalent_productivity_mw_per_m3s`). *initial_volumes*
    (``{code: hm³}``, from :func:`_operated_initial_volumes`) supplies that
    anchor; a plant absent from it — or the whole map being ``None`` — falls
    back to the full-range mean anchor (``reference_volume_hm3=None``), the
    pre-FPHA-fix behaviour.

    *exclude_codes* omits those plants from the emitted table — used for the
    ``hydro_energy_productivity.parquet`` write to drop the FPHA plants, which
    carry ``model: "fpha"`` and would be a double-supply (cobre rejects a plant
    that has both a computed FPHA and a parquet ρ_eq). Left ``None`` (every
    plant emitted) the full list still feeds :func:`convert_penalties`' system
    ρ_avg/ρ_max, so the penalty scale is unchanged by the FPHA split. The
    per-stage head variation instead feeds
    :func:`convert_hydro_group_availability`'s own hydraulic ceiling.
    """
    vols = initial_volumes or {}
    excluded = exclude_codes or set()
    hydro_ids: list[int] = []
    values: list[float] = []
    for code in id_map.hydro_codes:
        if code in excluded:
            continue
        hydro_ids.append(id_map.hydro_id(code))
        values.append(
            _equivalent_productivity_mw_per_m3s(
                effective, code, 0, reference_volume_hm3=vols.get(code)
            )
        )

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


def _availability_bound_entry(
    q_g: float,
    availability_mw: float,
    q_envelope: float,
    p_envelope: float,
) -> GroupBoundEntry | None:
    """One ``(hydro/group, stage)``'s B8 overlay entry, or ``None`` if
    neither bound column falls below that group's own declared envelope.

    ``max_generation_mw`` is *availability_mw* (installed × MP × FD — the
    source model's own reported available capacity, ``potencia_disponivel``)
    when it falls below *p_envelope*. The turbined-flow ceiling is *not*
    folded into this generation cap: ``max_turbined_m3s`` below already caps
    the flow at the per-stage head-corrected engolimento, and generation is
    ``productivity × turbined``, so the physical hydraulic limit binds through
    the flow cap — a separate ``ρ_eq·q_max`` generation cap only ever
    duplicated it (and understated any FPHA plant whose production curve beat
    the linear ``ρ_eq`` estimate).

    ``max_turbined_m3s`` is *q_g* — the per-stage **head-corrected** turbined
    flow (the head-aware engolimento supplied by the caller, not the rated
    unit-flow sum) — when it falls below *q_envelope* (a mid-horizon machine-set
    shrink or a per-stage operating-head change lowers the turbined-flow cap too,
    not just generation). Shared by the single-group and Itaipu
    per-conjunto-group emission loops in
    :func:`convert_hydro_group_availability`.
    """
    max_generation_mw = (
        availability_mw if _below_envelope(availability_mw, p_envelope) else None
    )
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
) -> dict[tuple[int, int, int], GroupBoundEntry]:
    """B8 per-group per-stage available capacity.

    For the single-group majority (every plant but Itaipu):
    ``availability_mw(stage) = installed(stage) × MP(stage) × FD(stage)`` —
    ``installed(stage)`` is the per-stage AC-adjusted rated power
    (:func:`_compute_max_turbined_rated_ac_adjusted`, sourced from
    *effective*'s machine-set view), a missing ``MP``/``FD`` register
    defaults its factor to ``1.0``. This *is* the source model's own reported
    available capacity (``potencia_disponivel``), so it is emitted directly as
    the ``max_generation_mw`` overlay, sparse against the group's own rated
    envelope (:func:`_rated_envelope`). The per-stage hydraulic engolimento
    limit is *not* imposed as a second generation cap: the emitted
    ``max_turbined_m3s`` overlay is the per-stage **head-corrected** flow
    (:func:`_head_corrected_max_turbined_ac_adjusted`, ticket-017), sparse
    against the group's own head-corrected envelope
    (:func:`_head_corrected_envelope`), and generation is
    ``productivity × turbined`` — so the physical hydraulic limit binds
    through the flow cap, not through a redundant ``ρ_eq·q_max`` generation
    ceiling (:func:`_availability_bound_entry`). A mid-horizon machine-set
    shrink or head drop lowers both ceilings independently. Defensively, any
    *other* plant whose ``MP``/``FD`` register carries more than one row is
    also dropped from this path (today only Itaipu, code 66, does).

    For Itaipu specifically, the same B8 formula is computed **per
    conjunto-backed group** rather than per plant —
    ``availability_mw(g, stage) = installed_g(stage) × MP_g(stage) ×
    FD_g(stage)``, ``g`` in ``{0, 1}`` (frequencies sorted ascending,
    matching :func:`_build_split_unit_groups`'s group ids), each register
    row frequency-matched via :func:`_frequency_row` (no defaulting to
    ``1.0`` — the split plant declares both rows); its ``max_turbined_m3s``
    overlay is the per-conjunto head-corrected flow
    (:func:`_conjunto_head_corrected_ac_adjusted`), sparse against that
    group's own head-corrected envelope
    (:func:`_conjunto_head_corrected_envelope`).

    Returns the ``(hydro_id, hydro_unit_group_id, stage_id) ->
    GroupBoundEntry`` mapping ticket-025's ``convert_hydro_unit_group_bounds``
    consumes unchanged — populated *sparsely*, only where an emitted value
    falls below that group's own declared envelope, the same "only where it
    differs" convention every sibling emitter uses.
    """
    mp = dadger.mp(df=True)
    fd = dadger.fd(df=True)
    mp_by_code = _single_group_factor_rows(mp)
    fd_by_code = _single_group_factor_rows(fd)

    values: dict[tuple[int, int, int], GroupBoundEntry] = {}
    for code in id_map.hydro_codes:
        hreg = hidr.loc[code]
        hydro_id = id_map.hydro_id(code)
        if code == _ITAIPU_CODE:
            frequencies = _split_plant_frequencies(hreg, code, effective, mp, fd)
            for i, frequency in enumerate(frequencies):
                conjunto_index = i + 1
                _, p_envelope = _conjunto_rated_envelope(
                    hreg, code, conjunto_index, effective
                )
                q_envelope = _conjunto_head_corrected_envelope(
                    hreg, code, conjunto_index, effective
                )
                mp_row = _frequency_row(mp, code, frequency)
                fd_row = _frequency_row(fd, code, frequency)

                for stage in calendar:
                    rho_eq = _equivalent_productivity_mw_per_m3s(
                        effective, code, stage.index
                    )
                    q_g, p_g = _conjunto_rated_ac_adjusted(
                        hreg, code, conjunto_index, effective, stage.index
                    )
                    mp_factor = _stage_register_factor(
                        mp_row, "manutencao", code, stage.index
                    )
                    fd_factor = _stage_register_factor(
                        fd_row, "fator", code, stage.index
                    )
                    availability_mw = p_g * mp_factor * fd_factor
                    h_op = _operating_head(effective, code, stage.index, rho_eq)
                    q_head = (
                        q_g
                        if h_op is None
                        else _conjunto_head_corrected_ac_adjusted(
                            hreg,
                            code,
                            conjunto_index,
                            effective,
                            stage.index,
                            h_op,
                            rho_eq,
                        )
                    )
                    entry = _availability_bound_entry(
                        q_head, availability_mw, q_envelope, p_envelope
                    )
                    if entry is not None:
                        values[(hydro_id, i, stage.index)] = entry
            continue
        _, p_envelope = _rated_envelope(hreg, code, effective)
        q_envelope = _head_corrected_envelope(hreg, code, effective)
        mp_row = mp_by_code.get(code)
        fd_row = fd_by_code.get(code)

        for stage in calendar:
            rho_eq = _equivalent_productivity_mw_per_m3s(effective, code, stage.index)
            q_g, p_g = _compute_max_turbined_rated_ac_adjusted(
                hreg, code, effective, stage.index
            )
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

            h_op = _operating_head(effective, code, stage.index, rho_eq)
            q_head = (
                q_g
                if h_op is None
                else _head_corrected_max_turbined_ac_adjusted(
                    hreg, code, effective, stage.index, h_op, rho_eq
                )
            )
            entry = _availability_bound_entry(
                q_head, availability_mw, q_envelope, p_envelope
            )
            if entry is not None:
                values[(hydro_id, 0, stage.index)] = entry

    return values


def convert_production_models(
    id_map: DecompIdMap,
    fpha_configs: dict[int, dict] | None = None,
    reference_volumes: dict[int, dict] | None = None,
) -> dict:
    """Per-plant production-model selection.

    A plant in *fpha_configs* (``{code: fpha_config}``, from the pipeline via
    :func:`cobre_bridge.decomp.fpha.fitting_window`) is emitted as ``model:
    "fpha"`` — cobre fits the production function from the plant geometry
    (``hydro_geometry.parquet``) + tailrace families (``tailrace_curves.parquet``)
    over the config's ``fitting_window`` — with its ``reference_volume`` (from
    *reference_volumes*, the initial-volume anchor) setting the FPHA reference /
    backwater level. Every other operated plant keeps ``constant_productivity``,
    its ρ_eq riding in ``hydro_energy_productivity.parquet``. *fpha_configs* and
    *reference_volumes* are pre-built by the pipeline so this module needs no
    import from :mod:`cobre_bridge.decomp.fpha` (which imports it).
    """
    fpha_configs = fpha_configs or {}
    reference_volumes = reference_volumes or {}
    models: list[dict] = []
    for code in id_map.hydro_codes:
        stage_range: dict = {"start_stage_id": 0, "end_stage_id": None}
        if code in fpha_configs:
            stage_range["model"] = "fpha"
            stage_range["fpha_config"] = fpha_configs[code]
            reference_volume = reference_volumes.get(code)
            if reference_volume is not None:
                stage_range["reference_volume"] = reference_volume
        else:
            stage_range["model"] = "constant_productivity"
        models.append(
            {
                "hydro_id": id_map.hydro_id(code),
                "selection_mode": "stage_ranges",
                "stage_ranges": [stage_range],
            }
        )
    return {"$schema": _PRODUCTION_MODELS_SCHEMA_URL, "production_models": models}
