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
  tighter wins" one; that skip was correctly retired in epic-07/ticket-023,
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

from cobre_bridge.decomp.bounds_accumulator import BoundContribution
from cobre_bridge.decomp.cadastro import effective_storage_range, storage_envelope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

#: Mirrors the ``_SPARSITY_TOLERANCE`` idiom in ``decomp/hydro.py`` — a
#: relative tolerance scaled by the reference magnitude, used only to decide
#: whether a stage's effective storage bounds differ from the plant's outer
#: envelope past float noise (not a cobre-side rule).
_SPARSITY_TOLERANCE = 1e-9


def _floats_differ(value: float, reference: float) -> bool:
    """Whether *value* differs from *reference* past relative float noise."""
    return abs(value - reference) > _SPARSITY_TOLERANCE * max(abs(reference), 1.0)


def convert_hydro_bounds(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
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
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """Sparse per-stage storage contributions wherever a stage tightens the envelope.

    For each hydro *code*, the outer envelope is ``storage_envelope(effective,
    code)`` — the widest floor/ceiling the plant's per-stage volumes ever
    reach, and the default the entity ``reservoir`` block (ticket-007)
    declares. A stage whose effective range (:func:`~cobre_bridge.decomp.
    cadastro.effective_storage_range`) differs from that envelope (past float
    noise) contributes a stage-level (``block_id = None``) override; a stage
    equal to the envelope contributes nothing and simply inherits it. A plant
    with no temporal ``VOLMIN``/``VOLMAX`` override never differs from its own
    envelope, so it contributes nothing at all — and neither does a
    run-of-river (``D``) plant, whose per-stage range is already the same
    single-point collapse as its envelope (ticket-018). Storage is a
    stage-level axis (``block_eligible=False``), so no ``block_id`` is ever
    emitted here.
    """
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
