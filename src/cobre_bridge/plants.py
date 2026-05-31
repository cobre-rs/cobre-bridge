"""Canonical definition of "which hydro plants are in the Cobre case".

NEWAVE's ``confhd.dat`` lists every hydro the study knows about, tagged by
``usina_existente`` (``EX`` = existing/in-operation, ``NE``/``NC`` = not yet
built) and including *fictitious* accounting plants whose ``nome_usina`` starts
with ``FICT.``. Only the existing, non-fictitious plants become LP variables in
Cobre.

That filter (``usina_existente == "EX"`` minus ``FICT.*``) was historically
copy-pasted at ~13 sites across the converters, comparators, and id-map, each
re-deriving "the plant set" independently. These helpers give it one home so
every caller agrees. They are pure functions over an already-read
``Confhd.usinas`` DataFrame — they do **not** read files, so they compose with
the existing per-converter ``Confhd.read`` mocking without changing it.
"""

from __future__ import annotations

import pandas as pd

_FICT_NAME_PREFIX = "FICT."


def _is_fictitious(names: pd.Series) -> pd.Series:
    """Boolean mask of names that are fictitious accounting plants."""
    return names.str.strip().str.startswith(_FICT_NAME_PREFIX)


def existing_hydros(confhd_df: pd.DataFrame) -> pd.DataFrame:
    """Return the ``usina_existente == "EX"`` rows (fictitious ones included)."""
    return confhd_df[confhd_df["usina_existente"] == "EX"]


def active_hydros(confhd_df: pd.DataFrame) -> pd.DataFrame:
    """Return the existing, non-fictitious hydro rows that enter the Cobre LP.

    This is the canonical "active hydro" set: ``usina_existente == "EX"`` minus
    the ``FICT.*`` accounting plants. Row order (confhd declaration order) is
    preserved, which the id-map relies on to assign 0-based Cobre hydro IDs.
    """
    existing = existing_hydros(confhd_df)
    return existing[~_is_fictitious(existing["nome_usina"])]


def active_hydro_codes(confhd_df: pd.DataFrame) -> list[int]:
    """Return ``codigo_usina`` of the active hydros in declaration order."""
    return [int(code) for code in active_hydros(confhd_df)["codigo_usina"]]


def fictitious_existing_names(confhd_df: pd.DataFrame) -> list[str]:
    """Return the names of existing plants excluded as fictitious (for warnings)."""
    existing = existing_hydros(confhd_df)
    return existing.loc[_is_fictitious(existing["nome_usina"]), "nome_usina"].tolist()
