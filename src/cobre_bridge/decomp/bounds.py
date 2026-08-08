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
  (``HQ``/``LQ``/``CQ`` on QDEF) has its default **replaced** by it —
  those plants are skipped here and handled by the generic-constraints
  emitter.

Cobre's ``block_id`` axis is active for ``min_outflow_m3s``: an
``RQ``-derived plant emits a stage-level base row (``block_id = None``, the
hours-weighted value — unchanged from the earlier interim fold) plus, only
where a stage's per-block percentages actually differ, sparse per-block
override rows (``block_id = 0..n-1``) carrying the exact per-block minimum.
This mirrors the base-plus-sparse-override convention ``convert_lines`` and
``convert_thermal_bounds`` already use. ``UH``-declared and ``QDEF``-windowed
plants are unaffected — neither varies across blocks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa

from cobre_bridge.decomp.cadastro import storage_envelope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

#: Mirrors the ``_SPARSITY_TOLERANCE`` idiom in ``decomp/hydro.py`` — a
#: relative tolerance scaled by the reference magnitude, used only to decide
#: whether a stage's effective storage bounds differ from the plant's outer
#: envelope past float noise (not a cobre-side rule).
_SPARSITY_TOLERANCE = 1e-9


def _floats_differ(value: float, reference: float) -> bool:
    """Whether *value* differs from *reference* past relative float noise."""
    return abs(value - reference) > _SPARSITY_TOLERANCE * max(abs(reference), 1.0)


_HYDRO_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_outflow_m3s", pa.float64(), nullable=True),
        pa.field("min_storage_hm3", pa.float64(), nullable=True),
        pa.field("max_storage_hm3", pa.float64(), nullable=True),
    ]
)


def _empty() -> pa.Table:
    return pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
            "min_outflow_m3s": pa.array([], type=pa.float64()),
            "min_storage_hm3": pa.array([], type=pa.float64()),
            "max_storage_hm3": pa.array([], type=pa.float64()),
        },
        schema=_HYDRO_BOUNDS_SCHEMA,
    )


def convert_hydro_bounds(
    dadger: Dadger,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    effective: EffectiveCadastro,
) -> pa.Table:
    """Minimum-outflow bounds from the ``RQ`` defaults.

    ``UH``-declared and ``QDEF``-windowed plants each emit a single
    stage-level base row (``block_id = None``) per stage — a constant value
    for ``UH``, none at all for ``QDEF`` (skipped, deferred to the
    generic-constraints emitter). An ``RQ``-derived plant emits that same
    base row (the hours-weighted per-stage value, backward-compatible) plus,
    only where the stage's per-block percentages are not all equal, one
    override row per block (``block_id = 0..n-1``) carrying the exact
    ``pct[block] / 100 * base``.
    """
    rq = dadger.rq(df=True)
    if rq is None or rq.empty:
        return _empty()

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

    # Plants whose defluence is governed by an explicit flow-constraint
    # window are the generic-constraints emitter's territory.
    cq = dadger.cq(df=True)
    qdef_plants: set[int] = set()
    if cq is not None and not cq.empty and "tipo" in cq.columns:
        qdef_plants = {
            int(c)
            for c in cq[cq["tipo"].astype(str).str.strip() == "QDEF"]["codigo_usina"]
        }

    pct_blocks: dict[int, list[float]] = {}
    for _, row in rq.iterrows():
        values = []
        k = 1
        while f"vazao_{k}" in rq.columns:
            value = row[f"vazao_{k}"]
            values.append(0.0 if pd.isna(value) else float(value))
            k += 1
        pct_blocks[int(row["codigo_ree"])] = values

    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    minimums: list[float] = []
    block_ids: list[int | None] = []
    skipped_qdef = 0
    for code in id_map.hydro_codes:
        # ``per_block_stage[stage.index]`` is the RQ-derived per-block
        # minimum (``pct[b] / 100 * base``, one entry per declared block) —
        # ``None`` for a UH-declared plant, which never varies across
        # blocks.
        per_block_stage: list[list[float]] | None
        if code in uh_declared:
            per_stage = [uh_declared[code]] * len(calendar)
            per_block_stage = None
        elif code in qdef_plants:
            skipped_qdef += 1
            continue
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
        for stage in calendar:
            value = per_stage[stage.index]
            if pd.isna(value) or value <= 0.0:
                continue
            hydro_ids.append(id_map.hydro_id(code))
            stage_ids.append(stage.index)
            minimums.append(value)
            block_ids.append(None)

            if per_block_stage is None:
                continue
            block_values = per_block_stage[stage.index]
            uniform = all(v == block_values[0] for v in block_values)
            if uniform:
                continue
            for b, block_value in enumerate(block_values):
                hydro_ids.append(id_map.hydro_id(code))
                stage_ids.append(stage.index)
                minimums.append(block_value)
                block_ids.append(b)

    if skipped_qdef:
        _LOG.warning(
            "%d plant(s) with explicit QDEF flow windows skipped by the RQ "
            "default (the flow-constraint family replaces it; emitter "
            "pending)",
            skipped_qdef,
        )

    if not hydro_ids:
        return _empty()
    n = len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "block_id": pa.array(block_ids, type=pa.int32()),
            "min_outflow_m3s": pa.array(minimums, type=pa.float64()),
            "min_storage_hm3": pa.array([None] * n, type=pa.float64()),
            "max_storage_hm3": pa.array([None] * n, type=pa.float64()),
        },
        schema=_HYDRO_BOUNDS_SCHEMA,
    )


def convert_storage_bounds(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> pa.Table:
    """Sparse per-stage storage bounds wherever a stage tightens the envelope.

    For each hydro *code*, the outer envelope is ``storage_envelope(effective,
    code)`` — the widest floor/ceiling the plant's per-stage volumes ever
    reach, and the default the entity ``reservoir`` block (ticket-007)
    declares. A stage whose effective ``(volume_minimo, volume_maximo)``
    differs from that envelope (past float noise) emits an override row;
    a stage equal to the envelope emits none and simply inherits it. A plant
    with no temporal ``VOLMIN``/``VOLMAX`` override never differs from its own
    envelope, so it contributes no rows at all.
    """
    hydro_ids: list[int] = []
    stage_ids: list[int] = []
    min_storage_vals: list[float] = []
    max_storage_vals: list[float] = []
    for code in id_map.hydro_codes:
        env_min, env_max = storage_envelope(effective, code)
        for stage_index in range(len(calendar)):
            vmin = effective.value(code, "volume_minimo", stage_index)
            vmax = effective.value(code, "volume_maximo", stage_index)
            if not (_floats_differ(vmin, env_min) or _floats_differ(vmax, env_max)):
                continue
            hydro_ids.append(id_map.hydro_id(code))
            stage_ids.append(stage_index)
            min_storage_vals.append(vmin)
            max_storage_vals.append(vmax)

    if not hydro_ids:
        return _empty()
    n = len(hydro_ids)
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "block_id": pa.array([None] * n, type=pa.int32()),
            "min_outflow_m3s": pa.array([None] * n, type=pa.float64()),
            "min_storage_hm3": pa.array(min_storage_vals, type=pa.float64()),
            "max_storage_hm3": pa.array(max_storage_vals, type=pa.float64()),
        },
        schema=_HYDRO_BOUNDS_SCHEMA,
    )
