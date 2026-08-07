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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage

_LOG = logging.getLogger(__name__)

_HYDRO_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("min_outflow_m3s", pa.float64(), nullable=False),
    ]
)


def _empty() -> pa.Table:
    return pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
            "min_outflow_m3s": pa.array([], type=pa.float64()),
        },
        schema=_HYDRO_BOUNDS_SCHEMA,
    )


def convert_hydro_bounds(
    dadger: Dadger,
    hidr: pd.DataFrame,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
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

    # Registry historical minimum, post AC VAZMIN overrides.
    vazmin: dict[int, float] = {}
    for code in id_map.hydro_codes:
        if code in hidr.index:
            base = hidr.loc[code, "vazao_minima_historica"]
            vazmin[code] = 0.0 if pd.isna(base) else float(base)
    from idecomp.decomp.modelos.dadger import ACVAZMIN

    overrides = dadger.ac(codigo_usina=None, modificacao=ACVAZMIN, df=True)
    n_overrides = 0
    if isinstance(overrides, pd.DataFrame) and not overrides.empty:
        for _, row in overrides.iterrows():
            code = int(row["codigo_usina"])
            if code in vazmin:
                vazmin[code] = float(row["vazao"])
                n_overrides += 1

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
            base = vazmin.get(code, 0.0)
            if ree is None or ree not in pct_blocks or base <= 0.0:
                continue
            values = pct_blocks[ree]
            per_stage = []
            per_block_stage = []
            for stage in calendar:
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
            if value <= 0.0:
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

    if n_overrides:
        _LOG.info(
            "applied %d AC VAZMIN historical-minimum overrides "
            "(first AC subtype converted)",
            n_overrides,
        )
    if skipped_qdef:
        _LOG.warning(
            "%d plant(s) with explicit QDEF flow windows skipped by the RQ "
            "default (the flow-constraint family replaces it; emitter "
            "pending)",
            skipped_qdef,
        )

    if not hydro_ids:
        return _empty()
    return pa.table(
        {
            "hydro_id": pa.array(hydro_ids, type=pa.int32()),
            "stage_id": pa.array(stage_ids, type=pa.int32()),
            "block_id": pa.array(block_ids, type=pa.int32()),
            "min_outflow_m3s": pa.array(minimums, type=pa.float64()),
        },
        schema=_HYDRO_BOUNDS_SCHEMA,
    )
