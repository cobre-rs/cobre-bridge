"""Convert NEWAVE downstream-level curve families to ``tailrace_curves.parquet``.

NEWAVE's ``polinjus`` cadastro stores, per hydro plant, one or more *families* of
downstream (tailrace) level curves ``h_jus(q_jus)``. Each family is keyed by a
reference forebay level of the downstream plant (``HjusRef``) and is a piecewise
polynomial in the total downstream flow (turbined + spilled + lateral). cobre's
FPHA production model consumes these as ``system/tailrace_curves.parquet`` —
piecewise-quartic backwater families — to evaluate the exact net head.

The mapping is one-to-one with the ``inewave`` reader columns:

==========================================  =========================
``polinjus`` (UsinasHidreletricas) column   ``tailrace_curves`` column
==========================================  =========================
``codigo_usina``                            ``hydro_id`` (0-based, via id map)
``indice_familia``                          ``family_id``
``nivel_montante_referencia``               ``href_jus_m``
``indice_polinomio``                        ``segment_id``
``limite_inferior_vazao_jusante``           ``q_jus_inf_m3s``
``limite_superior_vazao_jusante``           ``q_jus_sup_m3s``
``coeficiente_a0`` .. ``coeficiente_a4``    ``a_cf0`` .. ``a_cf4``
==========================================  =========================

The family/segment indices are NEWAVE's 1-based sequential keys and are passed
through unchanged — cobre treats them as opaque per-plant grouping keys (the d31
reference case likewise uses 1-based ``family_id``/``segment_id``). Only the
plant code is remapped to cobre's dense 0-based id; plants absent from the id map
(filtered fictitious plants, etc.) are skipped.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import pandas as pd
import pyarrow as pa

if TYPE_CHECKING:
    from cobre_bridge.case import NewaveCase
    from cobre_bridge.id_map import NewaveIdMap

_LOG = logging.getLogger(__name__)

# Arrow schema required by cobre's `parse_tailrace_curves`: only `href_jus_m` is
# nullable; every other column is non-nullable Int32/Float64.
_TAILRACE_SCHEMA = pa.schema(
    [
        pa.field("hydro_id", pa.int32(), nullable=False),
        pa.field("family_id", pa.int32(), nullable=False),
        pa.field("href_jus_m", pa.float64(), nullable=True),
        pa.field("segment_id", pa.int32(), nullable=False),
        pa.field("q_jus_inf_m3s", pa.float64(), nullable=False),
        pa.field("q_jus_sup_m3s", pa.float64(), nullable=False),
        pa.field("a_cf0", pa.float64(), nullable=False),
        pa.field("a_cf1", pa.float64(), nullable=False),
        pa.field("a_cf2", pa.float64(), nullable=False),
        pa.field("a_cf3", pa.float64(), nullable=False),
        pa.field("a_cf4", pa.float64(), nullable=False),
    ]
)


def convert_tailrace_curves(case: NewaveCase, id_map: NewaveIdMap) -> pa.Table | None:
    """Build ``system/tailrace_curves.parquet`` from NEWAVE's ``polinjus`` families.

    Parameters
    ----------
    case:
        Parsed NEWAVE case. ``case.polinjus`` supplies the downstream-level
        curve families (``None`` when the case ships no ``polinjus`` file).
    id_map:
        Entity ID map translating NEWAVE plant codes to 0-based Cobre hydro IDs.

    Returns
    -------
    pyarrow.Table | None
        A table with the eleven ``tailrace_curves`` columns sorted by
        ``(hydro_id, family_id, segment_id)``, or ``None`` when the case has no
        ``polinjus`` file or no segment maps to a converted hydro. ``None`` means
        "do not write the optional file"; cobre's FPHA then falls back to the
        entity-level tailrace from ``hydros.json``.
    """
    uh = case.polinjus
    if uh is None:
        return None

    families = cast(pd.DataFrame, uh.hidreletrica_curvajusante(df=True))
    segments = cast(
        pd.DataFrame, uh.hidreletrica_curvajusante_polinomio_segmento(df=True)
    )
    if families is None or segments is None or families.empty or segments.empty:
        return None

    # Attach each segment's family reference forebay level (HjusRef), then remap
    # plant codes to dense 0-based hydro ids, dropping plants absent from the map
    # (filtered fictitious plants, etc.). A left merge keeps every segment row.
    merged = segments.merge(
        families[["codigo_usina", "indice_familia", "nivel_montante_referencia"]],
        on=["codigo_usina", "indice_familia"],
        how="left",
    )

    code_to_hydro: dict[int, int] = {}
    skipped: set[int] = set()
    for raw_code in merged["codigo_usina"].unique():
        code = int(raw_code)
        try:
            code_to_hydro[code] = id_map.hydro_id(code)
        except KeyError:
            skipped.add(code)
    if skipped:
        _LOG.debug(
            "Skipped %d polinjus plant code(s) absent from the id map: %s",
            len(skipped),
            sorted(skipped),
        )

    merged = merged[merged["codigo_usina"].isin(list(code_to_hydro))]
    if merged.empty:
        return None

    # Deterministic output order; cobre re-sorts by the same integer keys.
    hydro_ids = [code_to_hydro[int(code)] for code in merged["codigo_usina"].tolist()]
    merged = merged.assign(hydro_id=hydro_ids).sort_values(
        ["hydro_id", "indice_familia", "indice_polinomio"], kind="stable"
    )

    # href is nullable; coerce any NaN to None so cobre's finite-check passes.
    href = [
        None if pd.isna(v) else float(v)
        for v in merged["nivel_montante_referencia"].tolist()
    ]

    def _f64(column: str) -> pa.Array:
        return pa.array(merged[column].astype(float).tolist(), type=pa.float64())

    def _i32(column: str) -> pa.Array:
        return pa.array(merged[column].astype(int).tolist(), type=pa.int32())

    return pa.table(
        {
            "hydro_id": _i32("hydro_id"),
            "family_id": _i32("indice_familia"),
            "href_jus_m": pa.array(href, type=pa.float64()),
            "segment_id": _i32("indice_polinomio"),
            "q_jus_inf_m3s": _f64("limite_inferior_vazao_jusante"),
            "q_jus_sup_m3s": _f64("limite_superior_vazao_jusante"),
            "a_cf0": _f64("coeficiente_a0"),
            "a_cf1": _f64("coeficiente_a1"),
            "a_cf2": _f64("coeficiente_a2"),
            "a_cf3": _f64("coeficiente_a3"),
            "a_cf4": _f64("coeficiente_a4"),
        },
        schema=_TAILRACE_SCHEMA,
    )
