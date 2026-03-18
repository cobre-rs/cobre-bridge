"""NewaveIdMap: deterministic 1-based-to-0-based entity ID remapping.

Accepts sorted lists of NEWAVE entity IDs and provides 0-based lookups
that are consistent across all entity files.  Tickets 010 and 011 import
this class to share the same mapping produced during entity conversion.
"""

from __future__ import annotations


class NewaveIdMap:
    """Bidirectional ID map from NEWAVE 1-based codes to Cobre 0-based IDs.

    The remapping is deterministic: sort the input NEWAVE IDs in ascending
    order, assign Cobre IDs 0, 1, 2, … in that order.  This guarantees
    declaration-order invariance — the output is the same regardless of
    the order in which callers pass the ID lists.

    Parameters
    ----------
    subsystem_ids:
        Iterable of NEWAVE subsystem (submercado) codes to register,
        including fictitious ones.  Each unique code maps to one bus ID.
    hydro_codes:
        Iterable of NEWAVE hydro plant codes (``codigo_usina`` from
        ``confhd.dat``).
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
            newave_id: cobre_id
            for cobre_id, newave_id in enumerate(sorted(hydro_codes))
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
        """Sorted list of registered NEWAVE hydro codes."""
        return sorted(self._hydro)

    @property
    def all_thermal_codes(self) -> list[int]:
        """Sorted list of registered NEWAVE thermal codes."""
        return sorted(self._thermal)
