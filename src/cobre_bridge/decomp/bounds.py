"""Minimum-outflow bounds for DECOMP-like decks (``RQ`` defaults).

Semantics pinned against the reference manual (§4.5.11) and verified on
the jul-26 deck's own outputs (2026-07-24):

- the default minimum defluence is zero;
- ``RQ`` supplies, per REE, **per-block percentages of the registry's
  historical minimum flow** (``vazao_minima_historica`` — *not* the
  long-term mean), after ``AC VAZMIN`` overrides patch that registry
  field (62 overrides in the jul-26 deck, many to zero);
- a ``UH``-declared minimum has priority over ``RQ`` and is fixed for all
  stages;
- a plant with an explicit defluence window in the flow-constraint family
  (``HQ``/``LQ``/``CQ`` on QDEF) **co-applies** that window with its
  ``RQ``/``UH`` default rather than either one replacing the other (user
  ruling, 2026-08-09): both constraints hold on the same ``(hydro, stage,
  block)`` cell, so this module still emits the ``RQ``/``UH`` contribution
  unconditionally, and the accumulator (``bounds_accumulator.resolve`` /
  :func:`~cobre_bridge.decomp.bounds_accumulator.intersect`) composes it
  with the RHQ ``QDEF``-derived ``outflow`` contribution
  (``single_term_bounds.single_term_bound_contributions``) via
  max-of-lowers/min-of-uppers — the tighter side of each source wins, never
  one source displacing the other. A prior version of this module instead
  *skipped* the ``RQ``/``UH`` contribution for any plant with a ``CQ``
  ``QDEF`` window — encoding the wrong "the window replaces the default"
  reading of the reference manual instead of the correct "both apply,
  tighter wins" one; that skip was correctly retired
  once the accumulator's ``intersect`` could express the co-apply
  composition directly instead of this module approximating it via a skip.

Both emitters here return :class:`~cobre_bridge.decomp.bounds_accumulator.
BoundContribution` lists — the accumulator, not this module, resolves
per-cell collisions and fans them into the cobre bound parquet rows.

Cobre's ``block_id`` axis is active for ``min_outflow_m3s``: an
``RQ``-derived plant contributes either one stage-level base contribution
(``block_id = None``, the hours-weighted value — unchanged from the earlier
interim fold) when a stage's per-block percentages are all equal, **or**
sparse per-block contributions (``block_id = 0..n-1``, no base) when they are
not — never both, since ``resolve()`` does not replicate cobre's
replace-not-merge column semantics and would otherwise double-count the
shadowed base into every block's intersection. ``UH``-declared plants are
unaffected — their bound never varies across blocks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.decomp.bounds_accumulator import BoundContribution
from cobre_bridge.decomp.cadastro import effective_storage_range, storage_envelope

if TYPE_CHECKING:
    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.case import DecompCase
    from cobre_bridge.decomp.id_map import DecompIdMap

#: Mirrors the ``_SPARSITY_TOLERANCE`` idiom in ``decomp/hydro.py`` — a
#: relative tolerance scaled by the reference magnitude, used only to decide
#: whether a stage's effective storage bounds differ from the plant's outer
#: envelope past float noise (not a cobre-side rule).
_SPARSITY_TOLERANCE = 1e-9


def _floats_differ(value: float, reference: float) -> bool:
    """Whether *value* differs from *reference* past relative float noise."""
    return abs(value - reference) > _SPARSITY_TOLERANCE * max(abs(reference), 1.0)


def convert_hydro_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """Minimum-outflow contributions from the ``RQ``/``UH`` defaults.

    A ``UH``-declared plant contributes a constant stage-level
    (``block_id = None``) value every stage. An ``RQ``-derived plant
    contributes, per stage, **either** that same base value (``block_id =
    None``) when the stage's per-block percentages are all equal, **or** one
    contribution per block (``block_id = 0..n-1``, no base) when they are
    not — see the module docstring's replace-vs-intersect note. A plant with
    an explicit ``QDEF`` flow window still contributes its RQ/UH value here;
    the accumulator co-applies it with that window's own contribution
    (``single_term_bounds``) via max-of-lowers/min-of-uppers rather than
    either one replacing the other.
    """
    calendar = case.calendar
    dadger = case.dadger
    rq = dadger.rq(df=True)
    if rq is None or rq.empty:
        return []

    uh = dadger.uh(df=True)
    operated = uh[uh["volume_inicial"].notna()]
    ree_by_code: dict[int, int] = {}
    uh_declared: dict[int, float] = {}
    for _, row in operated.iterrows():
        code = int(row["codigo_usina"])
        ree_by_code[code] = int(row["codigo_ree"])
        declared = row.get("vazao_defluente_minima")
        if declared is not None and not pd.isna(declared):
            uh_declared[code] = float(declared)

    pct_blocks: dict[int, list[float]] = {}
    for _, row in rq.iterrows():
        values = []
        k = 1
        while f"vazao_{k}" in rq.columns:
            value = row[f"vazao_{k}"]
            values.append(0.0 if pd.isna(value) else float(value))
            k += 1
        pct_blocks[int(row["codigo_ree"])] = values

    contributions: list[BoundContribution] = []
    for code in id_map.hydro_codes:
        # ``per_block_stage[stage.index]`` is the RQ-derived per-block
        # minimum (``pct[b] / 100 * base``, one entry per declared block) —
        # ``None`` for a UH-declared plant, which never varies across
        # blocks.
        per_block_stage: list[list[float]] | None
        if code in uh_declared:
            per_stage = [uh_declared[code]] * len(calendar)
            per_block_stage = None
            contributor = "UH"
        else:
            ree = ree_by_code.get(code)
            if ree is None or ree not in pct_blocks or code not in effective.base.index:
                continue
            values = pct_blocks[ree]
            per_stage = []
            per_block_stage = []
            for stage in calendar:
                base = effective.value(code, "vazao_minima_historica", stage.index)
                n_blocks = len(stage.block_hours)
                block_values = [pct / 100.0 * base for pct in values[:n_blocks]]
                weighted_pct = (
                    sum(
                        pct * hours
                        for pct, hours in zip(
                            values[:n_blocks], stage.block_hours, strict=True
                        )
                    )
                    / stage.total_hours
                )
                per_stage.append(weighted_pct / 100.0 * base)
                per_block_stage.append(block_values)
            contributor = "RQ"

        hydro_id = id_map.hydro_id(code)
        for stage in calendar:
            value = per_stage[stage.index]
            if pd.isna(value) or value <= 0.0:
                continue

            if per_block_stage is None:
                contributions.append(
                    BoundContribution(
                        family="hydro",
                        entity_id=hydro_id,
                        stage_id=stage.index,
                        block_id=None,
                        axis="outflow",
                        lower=value,
                        upper=None,
                        contributor=contributor,
                    )
                )
                continue

            block_values = per_block_stage[stage.index]
            uniform = all(v == block_values[0] for v in block_values)
            if uniform:
                contributions.append(
                    BoundContribution(
                        family="hydro",
                        entity_id=hydro_id,
                        stage_id=stage.index,
                        block_id=None,
                        axis="outflow",
                        lower=value,
                        upper=None,
                        contributor=contributor,
                    )
                )
            else:
                for b, block_value in enumerate(block_values):
                    contributions.append(
                        BoundContribution(
                            family="hydro",
                            entity_id=hydro_id,
                            stage_id=stage.index,
                            block_id=b,
                            axis="outflow",
                            lower=block_value,
                            upper=None,
                            contributor=contributor,
                        )
                    )

    return contributions


def convert_storage_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """Sparse per-stage storage contributions wherever a stage tightens the envelope.

    For each hydro *code*, the outer envelope is ``storage_envelope(effective,
    code)`` — the widest floor/ceiling the plant's per-stage volumes ever
    reach, and the default the entity ``reservoir`` block
    declares. A stage whose effective range (:func:`~cobre_bridge.decomp.
    cadastro.effective_storage_range`) differs from that envelope (past float
    noise) contributes a stage-level (``block_id = None``) override; a stage
    equal to the envelope contributes nothing and simply inherits it. A plant
    with no temporal ``VOLMIN``/``VOLMAX`` override never differs from its own
    envelope, so it contributes nothing at all — and neither does a
    run-of-river (``D``) plant, whose per-stage range is already the same
    single-point collapse as its envelope. Storage is a
    stage-level axis (``block_eligible=False``), so no ``block_id`` is ever
    emitted here.
    """
    calendar = case.calendar
    contributions: list[BoundContribution] = []
    for code in id_map.hydro_codes:
        env_min, env_max = storage_envelope(effective, code)
        hydro_id = id_map.hydro_id(code)
        for stage_index in range(len(calendar)):
            vmin, vmax = effective_storage_range(effective, code, stage_index)
            if not (_floats_differ(vmin, env_min) or _floats_differ(vmax, env_max)):
                continue
            contributions.append(
                BoundContribution(
                    family="hydro",
                    entity_id=hydro_id,
                    stage_id=stage_index,
                    block_id=None,
                    axis="storage",
                    lower=vmin,
                    upper=vmax,
                    contributor="storage-envelope",
                )
            )

    return contributions


def convert_volume_espera_bounds(
    case: DecompCase,
    id_map: DecompIdMap,
    *,
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """Per-stage max-storage contributions from the ``VE`` (volume de espera).

    The ``VE`` register (manual §3.4.6.15) declares a flood-control storage
    ceiling for a hydro *with reservoir*, as a percentage of the plant's
    **useful** volume for each of the study's ``N`` stages (``volume_k`` →
    stage ``k − 1``). It is a **hard** maximum-storage limit — the register
    carries no penalty field — so the reservoir may not fill above it, which
    forces releases during the flood season (verified: DECOMP pins ITAPARICA
    at its 55.1 % VE ceiling every flood-season stage). cobre has no other
    input for it, so it is emitted here as a per-stage ``max_storage_hm3``
    upper bound in absolute hm³ (``env_min + VE% · (env_max − env_min)``, the
    same ``volume útil`` base the source reports storage against), one-sided
    (no lower — the floor stays the plant's own), and composed by the
    accumulator's min-of-uppers so it only ever tightens the declared
    reservoir ceiling.

    Sparse by construction: a stage whose VE ceiling is not strictly below the
    plant's storage envelope max (a ``VE = 100 %`` no-op, or a blank field)
    contributes nothing, and neither does a plant absent from the register or
    without a useful-volume reservoir.
    """
    calendar = case.calendar
    ve = case.dadger.ve(df=True)
    if ve is None or ve.empty:
        return []

    stage_columns = [
        column
        for _, column in sorted(
            (int(column.split("_")[1]), column)
            for column in ve.columns
            if column.startswith("volume_") and column.split("_")[1].isdigit()
        )
    ]
    operated = set(id_map.hydro_codes)
    n_stages = len(calendar)

    contributions: list[BoundContribution] = []
    for _, row in ve.iterrows():
        code = int(row["codigo_usina"])
        if code not in operated:
            continue
        env_min, env_max = storage_envelope(effective, code)
        useful = env_max - env_min
        if useful <= 0.0:
            continue
        hydro_id = id_map.hydro_id(code)
        for stage_index in range(min(n_stages, len(stage_columns))):
            pct = row[stage_columns[stage_index]]
            if pd.isna(pct):
                continue
            ceiling = env_min + float(pct) / 100.0 * useful
            # Only tightens: a VE ≥ the envelope max is a no-op, and the
            # bound must never *raise* the declared ceiling.
            if ceiling >= env_max or not _floats_differ(ceiling, env_max):
                continue
            contributions.append(
                BoundContribution(
                    family="hydro",
                    entity_id=hydro_id,
                    stage_id=stage_index,
                    block_id=None,
                    axis="storage",
                    lower=None,
                    upper=ceiling,
                    contributor="VE",
                )
            )

    return contributions


#: cobre ``hydro_bounds`` column for a consumptive water withdrawal, in m³/s
#: (positive = water removed from the plant's balance). The DECOMP ``TI``
#: irrigation rate and the source model's ``dsvagua`` file both land here.
_WATER_WITHDRAWAL_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32()),
        pa.field("stage_id", pa.int32()),
        pa.field("water_withdrawal_m3s", pa.float64()),
    ]
)


def convert_irrigation_withdrawal(
    case: DecompCase,
    id_map: DecompIdMap,
) -> pa.Table | None:
    """Per-(hydro, stage) consumptive irrigation withdrawal from the ``TI`` register.

    The ``TI`` register (*taxas de irrigação por UHE*) declares the water a hydro
    loses to irrigation, one rate (m³/s) per study stage (``taxa_k`` → stage
    ``k − 1``). It is a **consumptive** withdrawal — the water leaves the river
    and is unavailable for generation downstream — so it maps 1:1 to cobre's
    ``hydro_bounds`` ``water_withdrawal_m3s`` column, the DECOMP counterpart of
    the source model's ``dsvagua`` water-withdrawal file
    (:func:`cobre_bridge.converters.hydro.convert_water_withdrawal`). Omitting it
    leaves that flow in the balance, so cobre turbines it and over-generates.

    The ``TI`` rate is already a positive withdrawal, matching cobre's positive
    ``water_withdrawal_m3s`` convention (no sign flip — unlike the source model's
    negative-``valor`` ``dsvagua`` convention). A stage beyond the register's own
    ``taxa`` columns repeats the last declared rate (seasonal carry-forward,
    matching the post-study extension the load/inflow converters use); a
    zero-withdrawal ``(hydro, stage)`` contributes no row.

    Returns the table sorted by ``(hydro_id, stage_id)`` with schema
    ``(hydro_id: int32, stage_id: int32, water_withdrawal_m3s: float64)``, or
    ``None`` when the deck carries no ``TI`` register or no operated plant
    withdraws (so the pipeline leaves ``hydro_bounds`` unchanged).
    """
    calendar = case.calendar
    ti = case.dadger.ti(df=True)
    if ti is None or ti.empty:
        return None

    taxa_columns = [
        column
        for _, column in sorted(
            (int(column.split("_")[1]), column)
            for column in ti.columns
            if column.startswith("taxa_") and column.split("_")[1].isdigit()
        )
    ]
    if not taxa_columns:
        return None

    operated = set(id_map.hydro_codes)
    n_stages = len(calendar)

    rows_hydro: list[int] = []
    rows_stage: list[int] = []
    rows_value: list[float] = []
    for _, row in ti.iterrows():
        code = int(row["codigo_usina"])
        if code not in operated:
            continue
        hydro_id = id_map.hydro_id(code)
        for stage_index in range(n_stages):
            # Carry the last declared rate forward for any stage past the
            # register's own columns (a no-op when they already align).
            column = taxa_columns[min(stage_index, len(taxa_columns) - 1)]
            value = row[column]
            if pd.isna(value) or float(value) == 0.0:
                continue
            rows_hydro.append(hydro_id)
            rows_stage.append(stage_index)
            rows_value.append(float(value))

    if not rows_hydro:
        return None

    order = sorted(range(len(rows_hydro)), key=lambda i: (rows_hydro[i], rows_stage[i]))
    return pa.table(
        {
            "hydro_id": pa.array([rows_hydro[i] for i in order], type=pa.int32()),
            "stage_id": pa.array([rows_stage[i] for i in order], type=pa.int32()),
            "water_withdrawal_m3s": pa.array(
                [rows_value[i] for i in order], type=pa.float64()
            ),
        },
        schema=_WATER_WITHDRAWAL_SCHEMA,
    )
