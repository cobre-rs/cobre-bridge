"""Hydro unit-group construction shared by both conversion tracks.

``build_mirror_unit_group`` is the single builder for cobre's mandatory
``unit_groups`` array entries — reused verbatim by the source-model track
(:mod:`cobre_bridge.converters.hydro.entity`) and the DECOMP track
(:mod:`cobre_bridge.decomp.hydro`).
"""

from __future__ import annotations


def build_mirror_unit_group(
    *,
    name: str,
    bus_id: int,
    min_generation_mw: float,
    max_generation_mw: float,
    min_turbined_m3s: float,
    max_turbined_m3s: float,
    group_id: int = 0,
) -> dict[str, object]:
    """Build one "mirror" unit group for a hydro plant.

    cobre requires every hydro to declare a non-empty ``unit_groups`` array
    (``RawUnitGroup``, all seven fields present). For the ordinary,
    single-group plant every caller in the bridge emits, the group's bounds
    *mirror* the plant's own generation envelope verbatim — no clamping, no
    zeroing of minima, no recomputation — and with a single group,
    ``sum(group maxima) == plant maximum`` holds trivially, which is exactly
    what cobre rule 41 checks, so the rule is satisfied by construction.

    A plant whose halves are maintained independently (e.g. a two-frequency
    split) instead calls this twice, once per physically separate group,
    passing each group's own conjunto-backed bounds (not the plant's) and a
    distinct ``group_id`` — the caller is responsible for the group maxima
    still summing to the plant envelope (rule 41) and for the ids being
    unique within the plant (cobre rule 39; the overlay is id-addressed, so
    array order is never load-bearing — see ``decomp/group_bounds.py``).

    Parameters
    ----------
    name:
        The plant name, used verbatim as the group name.
    bus_id:
        The group's bus id (the plant's own bus for every group, when every
        group sits on the same bus).
    min_generation_mw, max_generation_mw:
        The group's generation bounds in MW.
    min_turbined_m3s, max_turbined_m3s:
        The group's turbined-flow bounds in m^3/s.
    group_id:
        The group's ``id``. Defaults to ``0`` — unit-group ids are dense and
        0-based within a plant, and a single mirror group is the plant's
        only (hence first) group. Pass a distinct value for each group of a
        multi-group plant.

    Returns
    -------
    dict[str, object]
        Exactly the seven ``RawUnitGroup`` keys required by
        ``hydros.schema.json`` — no extras.
    """
    return {
        "id": group_id,
        "name": name,
        "bus_id": bus_id,
        "min_generation_mw": min_generation_mw,
        "max_generation_mw": max_generation_mw,
        "min_turbined_m3s": min_turbined_m3s,
        "max_turbined_m3s": max_turbined_m3s,
    }
