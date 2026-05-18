"""Scalar parameter declarations for ``system/scalar_parameters.json``.

Cobre HEAD added the ``@name`` sigil for `generic_constraints.json`
expressions and a parallel `scalar_parameters.json` file that declares
every parameter referenced via ``@name``. When the file is absent any
``@name`` token in a constraint expression causes a load-time error, so
cobre-bridge always emits it with the per-hydro computed parameters that
are most likely to appear in handwritten or generated constraints.

For every non-fictitious hydro in the converted case we declare two
``computed`` entries:

- ``rho_eq_h{id}`` — :math:`\\rho_{eq}` (equivalent productivity)
- ``rho_acum_h{id}`` — :math:`\\rho_{acum}` (accumulated cascade productivity)

These two are the canonical NEWAVE-style productivities used by every
hydro-storage / hydro-generation energy constraint we know of (VminOP,
minimum cascade energy, EARM-based bounds, ...). The values themselves
are derived by cobre from VHA geometry and ρ_esp at solve time — we only
declare the names.

See ``book/src/schemas/scalar_parameters.schema.json`` and
``book/src/guide/scalar-parameters.md`` in the cobre repo for the on-disk
shape and the seven valid ``computed_spec.tag`` variants.
"""

from __future__ import annotations

from collections.abc import Iterable

_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/book/src/schemas/scalar_parameters.schema.json"
)


def rho_acum_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the accumulated productivity of one hydro."""
    return f"rho_acum_h{hydro_id}"


def rho_eq_name(hydro_id: int) -> str:
    """Canonical ``@name`` for the equivalent productivity of one hydro."""
    return f"rho_eq_h{hydro_id}"


def build_scalar_parameters(hydro_ids: Iterable[int]) -> dict:
    """Return a ``scalar_parameters.json`` dict declaring per-hydro parameters.

    Two ``computed`` entries are emitted per ``hydro_id``: ``rho_eq_h{id}``
    (tag ``equivalent_productivity``) and ``rho_acum_h{id}`` (tag
    ``accumulated_productivity``). The file is always written so that
    handwritten or generated constraint expressions can reference any
    per-hydro productivity via ``@name`` without requiring a regeneration.
    """
    unique_ids = sorted({int(h) for h in hydro_ids})

    entries: list[dict] = []
    next_id = 0
    for hydro_id in unique_ids:
        entries.append(
            {
                "id": next_id,
                "name": rho_eq_name(hydro_id),
                "kind": "computed",
                "computed_spec": {
                    "tag": "equivalent_productivity",
                    "hydro_id": hydro_id,
                },
            }
        )
        next_id += 1
        entries.append(
            {
                "id": next_id,
                "name": rho_acum_name(hydro_id),
                "kind": "computed",
                "computed_spec": {
                    "tag": "accumulated_productivity",
                    "hydro_id": hydro_id,
                },
            }
        )
        next_id += 1

    return {
        "$schema": _SCHEMA_URL,
        "scalar_parameters": entries,
    }
