"""Shared test helpers for building the source model case inputs without real file I/O.

``make_nw_files`` constructs a :class:`~cobre_bridge.newave_files.NewaveFiles`
with paths under a tmp dir (no filesystem access). ``make_case`` wraps it in a
:class:`~cobre_bridge.case.NewaveCase` and pre-fills the requested cached reader
slots, so a converter under test reads the supplied mock objects instead of
parsing files. Import them with ``from tests.conftest import make_case``.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cobre_bridge.case import NewaveCase
from cobre_bridge.newave_files import NewaveFiles


@pytest.fixture(autouse=True)
def _restore_cobre_bridge_logger() -> Iterator[None]:
    """Snapshot/restore the ``cobre_bridge`` logger around every test.

    ``cli._configure_logging`` flips ``propagate``/level for a run, and the CLI's
    Typer ``CliRunner`` tests invoke the app directly (bypassing ``main``'s restore),
    so without this fixture that state would leak into a later ``caplog``-by-logger
    test in the same interpreter and silently swallow its records.
    """
    pkg = logging.getLogger("cobre_bridge")
    prior_propagate = pkg.propagate
    prior_level = pkg.level
    try:
        yield
    finally:
        pkg.propagate = prior_propagate
        pkg.setLevel(prior_level)


# Required NewaveFiles attributes get a default path under tmp_path; optional
# ones default to None. Callers override any of them via keyword.
_REQUIRED = (
    "dger",
    "confhd",
    "conft",
    "sistema",
    "clast",
    "term",
    "ree",
    "patamar",
    "hidr",
    "vazoes",
)
_OPTIONAL = (
    "modif",
    "ghmin",
    "penalid",
    "vazpast",
    "dsvagua",
    "curva",
    "expt",
    "exph",
    "manutt",
    "c_adic",
    "cvar",
    "agrint",
    "re_dat",
    "volref_saz",
    "shist",
    "adterm",
    "polinjus",
    "tratamento_fpha",
)


def make_nw_files(tmp_path: Path, **overrides: Path | None) -> NewaveFiles:
    """Build a ``NewaveFiles`` pointing into *tmp_path*; no I/O.

    Required files default to ``tmp_path / "<name>.dat"`` and optional files to
    ``None``. Pass any attribute as a keyword to override (e.g.
    ``make_nw_files(tmp_path, cvar=tmp_path / "cvar.dat")``).
    """
    fields: dict[str, Path | None] = {"directory": tmp_path}
    for name in _REQUIRED:
        fields[name] = tmp_path / f"{name}.dat"
    for name in _OPTIONAL:
        fields[name] = None
    fields.update(overrides)
    return NewaveFiles(**fields)  # type: ignore[arg-type]


def make_case(files_or_tmp: NewaveFiles | Path, **parsed: Any) -> NewaveCase:
    """Build a ``NewaveCase`` with the given parsed readers pre-cached.

    *files_or_tmp* is either a ``NewaveFiles`` or a tmp dir (then
    ``make_nw_files`` builds one). Each ``parsed`` keyword sets the matching
    cached-property slot directly, so accessing e.g. ``case.dger`` returns the
    supplied object without parsing. Pass ``cvar=None`` to force an optional
    reader to the absent state.
    """
    files = (
        files_or_tmp
        if isinstance(files_or_tmp, NewaveFiles)
        else make_nw_files(files_or_tmp)
    )
    case = NewaveCase(files=files)
    for name, value in parsed.items():
        case.__dict__[name] = value
    # ``case.active_hydros``/``case.id_map`` now consult the hidr cadastro to
    # decide which orphaned FICT reservoirs to keep. Tests that don't exercise
    # that path get a default empty-cadastro hidr so those accessors don't try
    # to read a real binary hidr.dat (an empty cadastro keeps the historical
    # "exclude all FICT" behavior). Tests needing real geometry pass ``hidr=``.
    if "hidr" not in parsed:
        default_hidr = MagicMock()
        default_hidr.cadastro = pd.DataFrame()
        case.__dict__["hidr"] = default_hidr
    return case
