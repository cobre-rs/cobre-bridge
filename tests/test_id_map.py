"""Tests for :mod:`cobre_bridge.id_map` — NE-with-filling admission (ticket-005).

``build_id_map_from_readers`` gains a keyword-only ``exph`` parameter. When the
case's ``Exph`` reader is threaded in, any ``NE`` plant carrying a dead-volume
filling row (e.g. JURUENA, code 309 in ``newave_rodada_2001``) is admitted into the
id-map at its confhd declaration position. With ``exph=None`` (the default, used by
the path-only ``build_id_map`` and the comparators) enumeration is byte-identical to
today's ``EX``-only set — no ``NE`` plant is admitted. This protects
``newave_rodada_2000_completo``, which has no ``NE`` plants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from inewave.newave import Confhd, Conft, Exph, Hidr, Ree, Sistema

from cobre_bridge.case import NewaveCase
from cobre_bridge.id_map import NewaveIdMap, build_id_map_from_readers

_RODADA_2001 = Path("example/newave_rodada_2001")
_RODADA_2000 = Path("example/newave_rodada_2000_completo")

# JURUENA (309) is the lone NE-with-filling plant in newave_rodada_2001; its first
# exph row carries data_inicio_enchimento and the unit rows carry NaT.
_JURUENA = 309


def _readers(base: Path) -> dict[str, object]:
    """Read the five id-map readers plus exph for *base* (real example case)."""
    return {
        "confhd": Confhd.read(str(base / "confhd.dat")),
        "conft": Conft.read(str(base / "conft.dat")),
        "sistema": Sistema.read(str(base / "sistema.dat")),
        "ree": Ree.read(str(base / "ree.dat")),
        "hidr": Hidr.read(str(base / "hidr.dat")),
        "exph": Exph.read(str(base / "exph.dat")),
    }


def _build(readers: dict[str, object], *, with_exph: bool) -> NewaveIdMap:
    return build_id_map_from_readers(
        readers["confhd"],
        readers["conft"],
        readers["sistema"],
        readers["ree"],
        readers["hidr"],
        exph=readers["exph"] if with_exph else None,
    )


# --- Unit: admission gated on the exph reader being passed --------------------


def test_id_map_admits_juruena_with_exph() -> None:
    # Threading the exph reader admits the lone NE-with-filling plant (309), and
    # its assigned id equals its 0-based position in the active-hydro list.
    if not (_RODADA_2001 / "confhd.dat").exists():
        pytest.skip("example/newave_rodada_2001 not available")
    id_map = _build(_readers(_RODADA_2001), with_exph=True)
    codes = id_map.all_hydro_codes
    assert _JURUENA in codes
    assert id_map.hydro_id(_JURUENA) == codes.index(_JURUENA)


def test_id_map_excludes_juruena_without_exph() -> None:
    # Admission is gated on exph being passed; with exph=None (the default for
    # path-only / compare callers) 309 is not registered and lookup raises KeyError.
    if not (_RODADA_2001 / "confhd.dat").exists():
        pytest.skip("example/newave_rodada_2001 not available")
    id_map = _build(_readers(_RODADA_2001), with_exph=False)
    assert _JURUENA not in id_map.all_hydro_codes
    with pytest.raises(KeyError):
        id_map.hydro_id(_JURUENA)


def test_id_map_2000_completo_unchanged_by_exph() -> None:
    # newave_rodada_2000_completo has no NE-with-filling rows, so passing exph must
    # not change the enumeration: the hydro-code lists are byte-identical. This is
    # the backward-compat guard for the established case.
    if not (_RODADA_2000 / "confhd.dat").exists():
        pytest.skip("example/newave_rodada_2000_completo not available")
    readers = _readers(_RODADA_2000)
    without = _build(readers, with_exph=False).all_hydro_codes
    with_exph = _build(readers, with_exph=True).all_hydro_codes
    assert with_exph == without


# --- Integration: the live case threads exph=self.exph -----------------------


def test_case_id_map_assigns_juruena_id() -> None:
    # The live NewaveCase threads exph=self.exph into build_id_map_from_readers, so
    # case.id_map admits 309 and assigns it the id equal to its 0-based position in
    # case.active_hydro_codes (confhd declaration order preserved).
    if not (_RODADA_2001 / "confhd.dat").exists():
        pytest.skip("example/newave_rodada_2001 not available")
    case = NewaveCase.from_directory(_RODADA_2001)
    id_map = case.id_map
    active = case.active_hydro_codes
    # The id equalling the 0-based active-list index proves declaration-order
    # placement (admission at the confhd position, not appended at the end).
    assert id_map.hydro_id(_JURUENA) == active.index(_JURUENA)
