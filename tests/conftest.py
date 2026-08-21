"""Shared test helpers for building the source model case inputs without real file I/O.

``make_nw_files`` constructs a :class:`~cobre_bridge.newave_files.NewaveFiles`
with paths under a tmp dir (no filesystem access). ``make_case`` wraps it in a
:class:`~cobre_bridge.case.NewaveCase` and pre-fills the requested cached reader
slots, so a converter under test reads the supplied mock objects instead of
parsing files. ``hydro_with_group`` builds a 0.13-shaped ``hydros.json`` hydro
dict (no top-level ``bus_id``, one mirror ``unit_groups`` entry) for tests that
hand-build a hydro fixture rather than calling a converter. ``_FakeDadger``
is a decomp-side test double returning preset DataFrames for register
accessors. Import them with ``from tests.conftest import make_case``.
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

from cobre_bridge.case import NewaveCase
from cobre_bridge.converters.hydro import build_mirror_unit_group
from cobre_bridge.newave_files import NewaveFiles

# Shared skip marker for tier-2 tests that need the optional cobre-python
# wheel (`import cobre`). CI installs it via the `validation`/`test-roundtrip`
# extras on the primary (3.13) job only; use ``find_spec`` (import-free) so
# this module stays importable in a cobre-free environment. ``condition`` is
# passed by keyword (rather than positionally) so ``.kwargs["condition"]`` is
# introspectable.
# Import via ``from tests.conftest import requires_cobre_python``.
requires_cobre_python = pytest.mark.skipif(
    condition=importlib.util.find_spec("cobre") is None,
    reason=(
        "requires the optional cobre-python wheel (validation / test-roundtrip extra)"
    ),
)


def has_writer_binding() -> bool:
    """Whether an installed cobre wheel exposes ``write_policy_checkpoint``.

    Checked via ``importlib.util.find_spec`` before any import — the same
    import-free-first convention as ``requires_cobre_python`` above — so
    calling this (or importing this module) never requires cobre to be
    installed. A cobre wheel older than the one that added the writer
    binding is import-able (``find_spec`` succeeds) but lacks the attribute;
    ``requires_writer_binding`` below turns that case into a clean SKIP
    instead of a runtime ``AttributeError``.
    """
    if importlib.util.find_spec("cobre") is None:
        return False
    import cobre

    return hasattr(cobre, "write_policy_checkpoint")


# Additional skip marker for tier-2 tests whose path calls an in-wheel
# binding newer than the mere import-ability ``requires_cobre_python``
# checks. Stack this alongside ``requires_cobre_python`` on any test that
# calls ``cobre.write_policy_checkpoint`` directly (or transitively via a
# helper that does), so an old-but-importable cobre wheel skips cleanly
# rather than failing at runtime with ``AttributeError``.
# Import via ``from tests.conftest import requires_writer_binding``.
requires_writer_binding = pytest.mark.skipif(
    condition=not has_writer_binding(),
    reason=(
        "requires a cobre-python wheel exposing the write_policy_checkpoint "
        "writer binding"
    ),
)


@pytest.fixture
def decomp_mini_deck() -> Path:
    """Path to the committed DECOMP mini-deck under ``tests/decks/decomp_mini/``.

    A tier-2 end-to-end test converts this deck with the real
    ``convert_decomp_case`` and validates the result with the real
    ``cobre.io.validate`` — see ``tests/test_convert_decomp_e2e.py``.
    """
    return Path(__file__).parent / "decks" / "decomp_mini"


@pytest.fixture
def newave_mini_deck() -> Path:
    """Path to the committed NEWAVE mini-deck under ``tests/decks/newave_mini/``.

    A tier-2 end-to-end test converts this deck with the real
    ``convert_newave_case`` and validates the result with the real
    ``cobre.io.validate`` — see ``tests/test_convert_newave_e2e.py``.
    """
    return Path(__file__).parent / "decks" / "newave_mini"


@pytest.fixture
def dumb_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force a plain, fixed-width terminal for Typer ``--help`` content tests.

    Typer's Rich help (``rich_markup_mode="rich"``) renders its options table
    from the detected terminal. On CI, GitHub Actions sets
    ``GITHUB_ACTIONS``/``FORCE_COLOR``, pushing Rich into "force terminal" mode;
    inside Typer's ``CliRunner`` (whose captured stream is not a real tty) that
    path renders an *empty* options box, so option-presence assertions pass
    locally but fail on CI. ``TERM=dumb`` makes Rich treat the captured output as
    a plain, fixed-width stream and emit the help content deterministically.

    Only the ``--help`` content tests opt in (via this fixture) so the fixed
    width does not perturb width-sensitive rendering in the rest of the suite.
    The real CLI, writing to a normal stdout, is unaffected.
    """
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLUMNS", "80")


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


def hydro_with_group(
    hydro_id: int,
    bus_id: int,
    *,
    name: str | None = None,
    min_generation_mw: float = 0.0,
    max_generation_mw: float = 50.0,
    min_turbined_m3s: float = 0.0,
    max_turbined_m3s: float = 100.0,
    **extra: Any,
) -> dict[str, Any]:
    """Build a 0.13-shaped ``hydros.json`` hydro dict for tests.

    Matches the shape every converter now emits (cobre decisions 13/14 →
    §7.6, §7.8, via
    :func:`cobre_bridge.converters.hydro.build_mirror_unit_group`): no
    top-level ``bus_id`` and a single seven-field mirror ``unit_groups``
    entry whose four bounds equal this dict's own ``generation`` envelope
    (cobre rule 41's mirror invariant).

    Use this wherever a test needs "some hydro on bus N" rather than a
    real converter output — it replaces a hand-rolled seven-field group
    literal duplicated across test modules.

    Parameters
    ----------
    hydro_id:
        The Cobre 0-based hydro id.
    bus_id:
        The plant's bus id (relocated into ``unit_groups[0].bus_id``; no
        longer emitted at the top level).
    name:
        The plant name. Defaults to ``f"HYDRO_{hydro_id}"``.
    min_generation_mw, max_generation_mw, min_turbined_m3s, max_turbined_m3s:
        The plant's generation envelope. Pass these where a test asserts
        specific bounds, so the mirror group tracks them.
    **extra:
        Any additional top-level keys a test needs (``reservoir``,
        ``outflow``, ``downstream_id``, etc.), merged in verbatim.
    """
    plant_name = name if name is not None else f"HYDRO_{hydro_id}"
    hydro: dict[str, Any] = {
        "id": hydro_id,
        "name": plant_name,
        "generation": {
            "min_generation_mw": min_generation_mw,
            "max_generation_mw": max_generation_mw,
            "min_turbined_m3s": min_turbined_m3s,
            "max_turbined_m3s": max_turbined_m3s,
        },
        "unit_groups": [
            build_mirror_unit_group(
                name=plant_name,
                bus_id=bus_id,
                min_generation_mw=min_generation_mw,
                max_generation_mw=max_generation_mw,
                min_turbined_m3s=min_turbined_m3s,
                max_turbined_m3s=max_turbined_m3s,
            )
        ],
    }
    hydro.update(extra)
    return hydro


class _FakeDadger:
    """Return a preset DataFrame (or ``None``) for each register accessor.

    Shared across the decomp constraint-register and results-compare test
    modules; import with ``from tests.conftest import _FakeDadger``.
    """

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):  # noqa: ANN204 - test double
        if name.startswith("_"):
            raise AttributeError(name)

        def accessor(df: bool = True) -> pd.DataFrame | None:
            return self._frames.get(name)

        return accessor
