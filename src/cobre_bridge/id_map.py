"""NewaveIdMap: deterministic 1-based-to-0-based entity ID remapping.

Accepts sorted lists of NEWAVE entity IDs and provides 0-based lookups
that are consistent across all entity files.  Tickets 010 and 011 import
this class to share the same mapping produced during entity conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cobre_bridge.plants import active_hydro_codes, fictitious_codes

if TYPE_CHECKING:
    from inewave.newave import Confhd, Conft, Hidr, Ree, Sistema

    from cobre_bridge.newave_files import NewaveFiles

_LOG = logging.getLogger(__name__)


class NewaveIdMap:
    """Bidirectional ID map from NEWAVE 1-based codes to Cobre 0-based IDs.

    Subsystems and thermals are remapped deterministically by sorting the
    input NEWAVE IDs ascending and assigning Cobre IDs 0, 1, 2, … in that
    order.  Hydros are remapped in the order the codes are passed in,
    which the pipeline supplies as the ``confhd.dat`` declaration order;
    this preserves the upstream-to-downstream layout authors typically
    encode in that file.

    Parameters
    ----------
    subsystem_ids:
        Iterable of NEWAVE subsystem (submercado) codes to register,
        including fictitious ones.  Each unique code maps to one bus ID.
    hydro_codes:
        NEWAVE hydro plant codes (``codigo_usina`` from ``confhd.dat``)
        in declaration order.  Cobre hydro IDs are assigned in this order.
    thermal_codes:
        Iterable of NEWAVE thermal plant codes (``codigo_usina`` from
        ``conft.dat``).
    """

    def __init__(
        self,
        subsystem_ids: list[int],
        hydro_codes: list[int],
        thermal_codes: list[int],
    ) -> None:
        self._bus: dict[int, int] = {
            newave_id: cobre_id
            for cobre_id, newave_id in enumerate(sorted(subsystem_ids))
        }
        self._hydro: dict[int, int] = {
            newave_id: cobre_id for cobre_id, newave_id in enumerate(hydro_codes)
        }
        self._thermal: dict[int, int] = {
            newave_id: cobre_id
            for cobre_id, newave_id in enumerate(sorted(thermal_codes))
        }

    def bus_id(self, newave_subsystem_id: int) -> int:
        """Return the 0-based Cobre bus ID for a NEWAVE subsystem code.

        Raises
        ------
        KeyError
            If ``newave_subsystem_id`` was not registered at construction.
        """
        return self._bus[newave_subsystem_id]

    def hydro_id(self, newave_hydro_code: int) -> int:
        """Return the 0-based Cobre hydro ID for a NEWAVE plant code.

        Raises
        ------
        KeyError
            If ``newave_hydro_code`` was not registered at construction.
        """
        return self._hydro[newave_hydro_code]

    def thermal_id(self, newave_thermal_code: int) -> int:
        """Return the 0-based Cobre thermal ID for a NEWAVE plant code.

        Raises
        ------
        KeyError
            If ``newave_thermal_code`` was not registered at construction.
        """
        return self._thermal[newave_thermal_code]

    @property
    def all_bus_ids(self) -> list[int]:
        """Sorted list of registered NEWAVE subsystem codes."""
        return sorted(self._bus)

    @property
    def all_hydro_codes(self) -> list[int]:
        """NEWAVE hydro codes in Cobre-ID order (confhd.dat declaration order)."""
        return list(self._hydro)

    @property
    def all_thermal_codes(self) -> list[int]:
        """Sorted list of registered NEWAVE thermal codes."""
        return sorted(self._thermal)


def build_id_map(nw_files: NewaveFiles) -> NewaveIdMap:
    """Build the canonical :class:`NewaveIdMap` from a case's NEWAVE inputs.

    Reads ``confhd.dat`` (hydros, existing non-fictitious only), ``conft.dat``
    (thermals), ``sistema.dat`` + ``ree.dat`` (subsystems). This is the single
    public entry point shared by the conversion pipeline and the comparators, so
    both derive the NEWAVE→Cobre mapping the same way (it used to live as a
    private ``pipeline._build_id_map`` that the comparators reached into).
    """
    from inewave.newave import Confhd, Conft, Hidr, Ree, Sistema

    confhd = Confhd.read(str(nw_files.confhd))
    conft = Conft.read(str(nw_files.conft))
    sistema = Sistema.read(str(nw_files.sistema))
    ree_file = Ree.read(str(nw_files.ree))
    hidr = Hidr.read(str(nw_files.hidr))
    return build_id_map_from_readers(confhd, conft, sistema, ree_file, hidr)


def build_id_map_from_readers(
    confhd: Confhd,
    conft: Conft,
    sistema: Sistema,
    ree_file: Ree,
    hidr: Hidr,
) -> NewaveIdMap:
    """Build the canonical :class:`NewaveIdMap` from already-parsed readers.

    The core of :func:`build_id_map`, factored out so :attr:`NewaveCase.id_map`
    can reuse the case's cached readers instead of re-parsing the files. ``hidr``
    supplies the productivity used to identify fictitious accounting plants
    structurally (see :func:`plants.fictitious_codes`).
    """
    # Hydro codes from confhd — existing plants minus the fictitious accounting
    # nodes (zero productivity sharing a generating plant's posto).
    confhd_df = confhd.usinas
    cadastro = hidr.cadastro
    fict = fictitious_codes(confhd_df, cadastro)
    if fict:
        names = [
            str(r["nome_usina"]).strip()
            for _, r in confhd_df.iterrows()
            if int(r["codigo_usina"]) in fict
        ]
        _LOG.warning(
            "Excluding %d fictitious plant(s) from id_map: %s",
            len(names),
            names,
        )
    hydro_codes = active_hydro_codes(confhd_df, cadastro)

    # Thermal codes from conft.
    conft_df = conft.usinas
    thermal_codes = [int(r["codigo_usina"]) for _, r in conft_df.iterrows()]

    # Subsystem codes from sistema deficit table.
    deficit_df = sistema.custo_deficit
    if deficit_df is not None:
        subsystem_ids = sorted(
            {int(r["codigo_submercado"]) for _, r in deficit_df.iterrows()}
        )
    else:
        subsystem_ids = []

    # Also include subsystem codes referenced in ree.dat (for completeness).
    ree_df = ree_file.rees
    if ree_df is not None:
        for _, row in ree_df.iterrows():
            code = int(row["submercado"])
            if code not in subsystem_ids:
                subsystem_ids.append(code)

    return NewaveIdMap(
        subsystem_ids=subsystem_ids,
        hydro_codes=hydro_codes,
        thermal_codes=thermal_codes,
    )
