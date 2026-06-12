"""Canonical definition of "which hydro plants are in the Cobre case".

NEWAVE's ``confhd.dat`` lists every hydro the study knows about, tagged by
``usina_existente`` (``EX`` = existing/in-operation, ``NE``/``NC`` = not yet
built) and including *fictitious* accounting plants. Only the existing,
non-fictitious plants become LP variables in Cobre.

A plant is identified as **fictitious structurally**, not by the ``FICT.`` name
prefix: a fictitious accounting twin shares its inflow gauge (``posto``) with a
real generating plant and itself has zero productivity
(``produtibilidade_especifica == 0``). In NEWAVE's full decks each ``FICT.*``
plant pairs with its real twin on the same posto, so the structural rule flags
exactly the same plants as the prefix — but it is robust to the naming
convention and, crucially, a plant orphaned by a REE/subsystem reduction (no
generating posto-mate left) is **not** fictitious and stays in the LP as a ρ=0
routing/regulation reservoir. This unifies "is it fictitious" with "should we
keep an orphaned twin" into one rule.

These helpers are the one home for that filter (historically copy-pasted at ~13
sites). They are pure functions over an already-read ``Confhd.usinas`` DataFrame
plus the ``Hidr.cadastro`` (for the productivity test); they do **not** read
files.
"""

from __future__ import annotations

import pandas as pd

# Productivity at or below this is treated as "no generation" (NEWAVE FICT
# accounting plants carry produtibilidade_especifica == 0 exactly).
_RHO_COL = "produtibilidade_especifica"


def existing_hydros(confhd_df: pd.DataFrame) -> pd.DataFrame:
    """Return the ``usina_existente == "EX"`` rows (fictitious ones included)."""
    return confhd_df[confhd_df["usina_existente"] == "EX"]


def fictitious_codes(confhd_df: pd.DataFrame, cadastro: pd.DataFrame) -> set[int]:
    """Return codes of existing plants that are fictitious accounting nodes.

    Structural rule: a plant is fictitious iff it generates nothing
    (``produtibilidade_especifica == 0``) **and** shares its ``posto`` with
    another existing plant that does generate (ρ > 0). The generating twin owns
    the inflow; the ρ=0 twin is the accounting node.

    A ρ=0 plant whose posto carries no generator is **not** fictitious — it is a
    real regulation reservoir (NEWAVE has these, e.g. GUARAPIRANGA/BILLINGS) or a
    twin orphaned by a reduction (e.g. ``FICT.MAUA`` once ``MAUA`` is filtered
    out), and it stays in the LP.

    Returns an empty set when the productivity (``cadastro``) or ``posto`` data
    needed for the test is unavailable.
    """
    if (
        "usina_existente" not in confhd_df.columns
        or "posto" not in confhd_df.columns
        or cadastro.empty
        or _RHO_COL not in cadastro.columns
    ):
        return set()  # data needed for the structural test is unavailable
    existing = existing_hydros(confhd_df)
    rho: dict[int, float] = {}
    posto: dict[int, int] = {}
    for _, row in existing.iterrows():
        code = int(row["codigo_usina"])
        if code not in cadastro.index:
            continue  # no geometry → can't classify; treat as real
        rho[code] = float(cadastro.loc[code, _RHO_COL])
        posto[code] = int(row["posto"])
    generating_postos = {posto[c] for c, r in rho.items() if r > 0.0}
    return {c for c, r in rho.items() if r == 0.0 and posto[c] in generating_postos}


def active_hydros(confhd_df: pd.DataFrame, cadastro: pd.DataFrame) -> pd.DataFrame:
    """Return the existing, non-fictitious hydro rows that enter the Cobre LP.

    ``usina_existente == "EX"`` minus :func:`fictitious_codes`, in confhd
    declaration order (the id-map relies on this order to assign 0-based Cobre
    hydro IDs). *cadastro* (``Hidr.cadastro``) is required for the structural
    fictitious test; with the data unavailable, no plant is classified
    fictitious and all existing plants are returned.
    """
    existing = existing_hydros(confhd_df)
    fict = fictitious_codes(confhd_df, cadastro)
    if not fict:
        return existing
    return existing[~existing["codigo_usina"].isin(fict)]


def active_hydro_codes(confhd_df: pd.DataFrame, cadastro: pd.DataFrame) -> list[int]:
    """Return ``codigo_usina`` of the active hydros in declaration order."""
    return [int(code) for code in active_hydros(confhd_df, cadastro)["codigo_usina"]]
