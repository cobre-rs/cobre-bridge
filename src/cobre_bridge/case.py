"""Parsed NEWAVE case — read each input file once, cache the result.

``NewaveFiles`` resolves the *paths* of a NEWAVE case; this module adds the
parsing layer on top. Converters used to take ``NewaveFiles`` and each re-parse
the files they needed, so a single conversion re-read ``dger.dat`` ~26×,
``confhd.dat`` ~27×, ``hidr.dat`` ~18×, and so on.

:class:`NewaveCase` wraps a :class:`~cobre_bridge.newave_files.NewaveFiles` and
exposes one :func:`functools.cached_property` per input file. The first access
parses via the matching ``inewave`` reader; every later access is free. Required
files return the reader; optional files return ``Reader | None`` (``None`` when
the file is absent), mirroring the ``Path | None`` shape on ``NewaveFiles``.

The underlying paths stay reachable via :attr:`NewaveCase.files`. The large,
single-use ``vazoes.dat`` inflow file is deliberately *not* cached here — its one
consumer reads it straight off ``case.files.vazoes`` so the full inflow history
need not be held for the whole conversion.

In tests, build a case and pre-fill the cached slots so no I/O happens (see the
``make_case`` factory in ``tests/conftest.py``); ``cached_property`` reads from
the instance ``__dict__`` first, so a pre-set value shadows the parse. For this
reason ``NewaveCase`` must stay a plain (non-frozen, non-slotted) dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from inewave.newave import (
    Adterm,
    Cadic,
    Clast,
    Confhd,
    Conft,
    Curva,
    Cvar,
    Dger,
    Dsvagua,
    Expt,
    Ghmin,
    Hidr,
    Manutt,
    Modif,
    Patamar,
    Penalid,
    Re,
    Ree,
    Shist,
    Sistema,
    Term,
    Vazpast,
    VolrefSaz,
)

from cobre_bridge.newave_files import NewaveFiles

if TYPE_CHECKING:
    from cobre_bridge.id_map import NewaveIdMap


@dataclass
class NewaveCase:
    """A NEWAVE case with each input file parsed once and cached.

    Construct via :meth:`from_directory` in production code. Each property below
    parses the matching file on first access through its ``inewave`` reader and
    caches the reader object on the instance.
    """

    files: NewaveFiles

    @classmethod
    def from_directory(cls, directory: Path) -> NewaveCase:
        """Discover and bind the NEWAVE files in *directory* (no parsing yet)."""
        return cls(files=NewaveFiles.from_directory(directory))

    # --- Required files --------------------------------------------------------

    @cached_property
    def dger(self) -> Dger:
        return Dger.read(str(self.files.dger))

    @cached_property
    def confhd(self) -> Confhd:
        return Confhd.read(str(self.files.confhd))

    @cached_property
    def conft(self) -> Conft:
        return Conft.read(str(self.files.conft))

    @cached_property
    def sistema(self) -> Sistema:
        return Sistema.read(str(self.files.sistema))

    @cached_property
    def clast(self) -> Clast:
        return Clast.read(str(self.files.clast))

    @cached_property
    def term(self) -> Term:
        return Term.read(str(self.files.term))

    @cached_property
    def ree(self) -> Ree:
        return Ree.read(str(self.files.ree))

    @cached_property
    def patamar(self) -> Patamar:
        return Patamar.read(str(self.files.patamar))

    @cached_property
    def hidr(self) -> Hidr:
        return Hidr.read(str(self.files.hidr))

    # --- Optional files (None when absent) -------------------------------------

    @cached_property
    def modif(self) -> Modif | None:
        path = self.files.modif
        return Modif.read(str(path)) if path is not None else None

    @cached_property
    def ghmin(self) -> Ghmin | None:
        path = self.files.ghmin
        return Ghmin.read(str(path)) if path is not None else None

    @cached_property
    def penalid(self) -> Penalid | None:
        path = self.files.penalid
        return Penalid.read(str(path)) if path is not None else None

    @cached_property
    def vazpast(self) -> Vazpast | None:
        path = self.files.vazpast
        return Vazpast.read(str(path)) if path is not None else None

    @cached_property
    def dsvagua(self) -> Dsvagua | None:
        path = self.files.dsvagua
        return Dsvagua.read(str(path)) if path is not None else None

    @cached_property
    def curva(self) -> Curva | None:
        path = self.files.curva
        return Curva.read(str(path)) if path is not None else None

    @cached_property
    def expt(self) -> Expt | None:
        path = self.files.expt
        return Expt.read(str(path)) if path is not None else None

    @cached_property
    def manutt(self) -> Manutt | None:
        path = self.files.manutt
        return Manutt.read(str(path)) if path is not None else None

    @cached_property
    def c_adic(self) -> Cadic | None:
        path = self.files.c_adic
        return Cadic.read(str(path)) if path is not None else None

    @cached_property
    def cvar(self) -> Cvar | None:
        path = self.files.cvar
        return Cvar.read(str(path)) if path is not None else None

    @cached_property
    def re_dat(self) -> Re | None:
        path = self.files.re_dat
        return Re.read(str(path)) if path is not None else None

    @cached_property
    def volref_saz(self) -> VolrefSaz | None:
        path = self.files.volref_saz
        return VolrefSaz.read(str(path)) if path is not None else None

    @cached_property
    def shist(self) -> Shist | None:
        path = self.files.shist
        return Shist.read(str(path)) if path is not None else None

    @cached_property
    def adterm(self) -> Adterm | None:
        path = self.files.adterm
        return Adterm.read(str(path)) if path is not None else None

    # --- Derived state ---------------------------------------------------------

    @cached_property
    def id_map(self) -> NewaveIdMap:
        """The canonical NEWAVE→Cobre :class:`NewaveIdMap` for this case.

        Built from the cached ``confhd``/``conft``/``sistema``/``ree`` readers, so
        it reuses parses already done rather than re-reading the four files (the
        free ``id_map.build_id_map`` re-reads them for callers that only hold
        paths, e.g. the CLI compare commands).
        """
        from cobre_bridge.id_map import build_id_map_from_readers

        return build_id_map_from_readers(
            self.confhd, self.conft, self.sistema, self.ree
        )
