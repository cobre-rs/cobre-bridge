"""Shared test helpers for building the source model case inputs without real file I/O.

``make_nw_files`` constructs a :class:`~cobre_bridge.newave_files.NewaveFiles`
with paths under a tmp dir (no filesystem access). ``make_case`` wraps it in a
:class:`~cobre_bridge.case.NewaveCase` and pre-fills the requested cached reader
slots, so a converter under test reads the supplied mock objects instead of
parsing files. ``make_decomp_files``/``make_decomp_case`` are the DECOMP-track
twins, wrapping a :class:`~cobre_bridge.decomp.case.DecompCase`.
``hydro_with_group`` builds a 0.13-shaped ``hydros.json`` hydro
dict (no top-level ``bus_id``, one mirror ``unit_groups`` entry) for tests that
hand-build a hydro fixture rather than calling a converter. ``_FakeDadger``
is a decomp-side test double returning preset DataFrames for register
accessors. ``_hydro_case``, ``_thermal_readers``, and ``_make_sistema_mock``
(with their synthetic confhd/hidr/conft/clast/term/ree DataFrame builders)
build the source-model entity-conversion fixtures shared across the hydro,
thermal, and network converter test modules. Import them with
``from tests.conftest import make_case``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import logging
import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from cobre_bridge.case import NewaveCase
from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.decomp_results import _AlignedDecompFrames
from cobre_bridge.converters.hydro import build_mirror_unit_group
from cobre_bridge.decomp.case import DecompCase
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.pipeline import DecompFiles
from cobre_bridge.id_map import NewaveIdMap
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


_DECOMP_REQUIRED = ("dadger", "vazoes", "hidr")
_DECOMP_OPTIONAL = (
    "dadgnl",
    "renovaveis",
    "polinjus",
    "libs_restricao_eletrica",
    "cortesh",
    "cortes",
)


def make_decomp_files(tmp_path: Path, **overrides: Any) -> DecompFiles:
    """Build a ``DecompFiles`` pointing into *tmp_path*; no I/O.

    Required paths default to ``tmp_path / "<name>"``, optional paths to
    ``None``, and ``revision`` to ``"rv0"``. Pass any attribute as a keyword
    to override (e.g. ``make_decomp_files(tmp_path, dadgnl=tmp_path /
    "dadgnl.rv0")``).
    """
    fields: dict[str, Any] = {"revision": "rv0"}
    for name in _DECOMP_REQUIRED:
        fields[name] = tmp_path / name
    for name in _DECOMP_OPTIONAL:
        fields[name] = None
    fields.update(overrides)
    return DecompFiles(**fields)


def make_decomp_case(files_or_tmp: DecompFiles | Path, **parsed: Any) -> DecompCase:
    """Build a ``DecompCase`` with the given parsed readers pre-cached.

    *files_or_tmp* is either a ``DecompFiles`` or a tmp dir (then
    ``make_decomp_files`` builds one). Each ``parsed`` keyword sets the
    matching cached-property slot directly, so accessing e.g. ``case.dadger``
    returns the supplied object without parsing.
    """
    files = (
        files_or_tmp
        if isinstance(files_or_tmp, DecompFiles)
        else make_decomp_files(files_or_tmp)
    )
    case = DecompCase(files=files)
    for name, value in parsed.items():
        case.__dict__[name] = value
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


def _patch_shared_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id_map: DecompIdMap,
    dadger: object | None = None,
) -> None:
    """Patch the shared ``DecompCase.from_directory`` build (ticket-020) so
    ``build_decomp_dataset``'s ``case.id_map``/``case.dadger`` resolve to
    *id_map*/*dadger* without touching the filesystem -- the case is now
    built unconditionally, before ``_read_aligned_frames`` runs, so every
    fixture exercising ``build_decomp_dataset`` against a bare ``tmp_path``
    needs this (mirrors ``_patch_aligned_frames``'s own "patch at the seam"
    convention; monkeypatch's last ``setattr`` wins, so a test needing a
    specific id map calls this again after ``_patch_aligned_frames``).
    """
    fake_dadger = _FakeDadger() if dadger is None else dadger
    monkeypatch.setattr(
        "cobre_bridge.decomp.case.DecompCase.from_directory",
        lambda directory: make_decomp_case(
            directory, dadger=fake_dadger, id_map=id_map
        ),
    )


def _patch_aligned_frames(
    monkeypatch: pytest.MonkeyPatch, aligned: _AlignedDecompFrames
) -> None:
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._read_aligned_frames",
        lambda *_args, **_kwargs: aligned,
    )
    # ticket-020: the shared ``DecompCase`` build now runs unconditionally at
    # the top of ``build_decomp_dataset`` (before ``_read_aligned_frames``,
    # which this stub bypasses) -- degenerate but valid, so a bare
    # ``tmp_path`` keeps working; tests needing a specific id map call
    # ``_patch_shared_case`` again afterwards (monkeypatch's last ``setattr``
    # wins).
    _patch_shared_case(monkeypatch, id_map=DecompIdMap(bus_codes=(), bus_names=()))
    # ticket-006: ``build_decomp_dataset`` also calls
    # ``read_cobre_bus_aggregates`` directly (outside ``_read_aligned_frames``).
    # Unlike the other cobre readers it does NOT degrade to empty on a missing
    # case -- it raises ``CobrePartitionMissingError`` for the pre-0.13
    # ``hydro_bus_generation`` partition, which a bare ``tmp_path`` always
    # trips. Stub it here too, so every fixture that does not care about
    # ticket-006's Energy Balance metadata (the vast majority) keeps working
    # against a bare ``tmp_path``; tests that DO care override this again
    # afterwards (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_bus_aggregates",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    # ticket-010: ``build_decomp_dataset`` also calls ``_cost_frames`` directly
    # (outside ``_read_aligned_frames``), which reads ``read_relato_costs`` --
    # unlike every other reader here, it RAISES on a missing/empty parse
    # (ticket-009's "no silent-empty" reader contract), which a bare
    # ``tmp_path`` always trips. Stub it here too, so every fixture that does
    # not care about ticket-010's cost metadata keeps working against a bare
    # ``tmp_path``; tests that DO care override this again afterwards
    # (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._cost_frames",
        lambda *_args, **_kwargs: ({}, pl.DataFrame()),
    )
    # ticket-014: ``build_decomp_dataset`` also calls
    # ``read_cobre_hydro_bus_labels`` directly (outside ``_read_aligned_frames``).
    # Like ``read_cobre_bus_aggregates`` above (ticket-006), it reads the
    # ``simulation/hydro_bus_generation/`` partition and RAISES
    # ``CobrePartitionMissingError`` on a bare ``tmp_path`` instead of
    # degrading to empty. Stub it here too, so every fixture that does not
    # care about ticket-014's hydro metadata keeps working against a bare
    # ``tmp_path``; tests that DO care override this again afterwards
    # (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_hydro_bus_labels",
        lambda *_args, **_kwargs: {},
    )


def _aligned_fixture() -> _AlignedDecompFrames:
    """One hydro plant, one thermal plant, one bus -- already aligned to Cobre
    ids/stages, matching the shape :func:`_read_aligned_frames` returns."""
    source_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "newave_code": [10, 20],
            "stage_id": [0, 0],
            "geracao_MW": [120.0, 60.0],
            "vazao_turbinada_m3s": [80.0, 40.0],
            "vazao_vertida_m3s": [0.0, 0.0],
            "vazao_defluente_m3s": [80.0, 40.0],
            "volume_util_final_hm3": [500.0, 300.0],
        }
    )
    cobre_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw": [110.0, 60.0],
            "turbined_m3s": [78.0, 40.0],
            "spillage_m3s": [0.0, 0.0],
            "outflow_m3s": [78.0, 40.0],
            "useful_storage_hm3": [480.0, 300.0],
        }
    )
    source_thermal = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [5],
            "stage_id": [0],
            "geracao_MW": [30.0],
        }
    )
    cobre_thermal = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "generation_mw": [28.0]}
    )
    source_bus = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [1],
            "stage_id": [0],
            "deficit_MW": [0.0],
            "cmo": [45.0],
        }
    )
    cobre_bus = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "deficit_mw": [0.0], "spot_price": [44.0]}
    )
    return _AlignedDecompFrames(
        source_hydro=source_hydro,
        source_thermal=source_thermal,
        source_bus=source_bus,
        cobre_hydro=cobre_hydro,
        cobre_thermal=cobre_thermal,
        cobre_bus=cobre_bus,
        hydro_names={0: "A", 1: "B"},
        thermal_names={0: "T"},
        bus_names={0: "SE"},
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


def _balance_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture`` extended with ticket-006's Energy Balance
    reference frames, keyed to the same bus (cobre id 0, name "SE")."""
    nw_market = pl.DataFrame(
        {
            "newave_code": [0, 0, 0],
            "stage": [1, 1, 1],
            "variable": ["GHTOT", "GTERM", "DEFT"],
            "value": [600.0, 250.0, 0.0],
        }
    )
    nw_net_load = pl.DataFrame(
        {
            "newave_code": [0],
            "stage": [1],
            "variable": ["NET_LOAD"],
            "value": [950.0],
        }
    )
    nw_sin = pl.DataFrame(
        {
            "newave_code": [0, 0],
            "stage": [1, 1],
            "variable": ["EARMF", "ENA"],
            "value": [7000.0, 1600.0],
        }
    )
    return dataclasses.replace(
        _aligned_fixture(), nw_market=nw_market, nw_net_load=nw_net_load, nw_sin=nw_sin
    )


def _ree_id_map() -> DecompIdMap:
    """Two hydro plants (codes 10, 20 -> cobre ids 0, 1) -- matches
    ``_aligned_fixture``'s own hydro codes/ids so the same
    ``_patch_aligned_frames`` fixture can back both the E1 result rows and
    the REE rollup in the same ``build_decomp_dataset`` test."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20))


def _ree_membership_fixture() -> pl.DataFrame:
    """Both fixture plants (codes 10, 20) belong to REE 100 ('SUDESTE')."""
    return pl.DataFrame(
        {
            "codigo_usina": [10, 20],
            "nome_usina": ["A", "B"],
            "codigo_ree": [100, 100],
            "nome_ree": ["SUDESTE", "SUDESTE"],
            "codigo_submercado": [1, 1],
            "nome_submercado": ["SE", "SE"],
            "nome_submercado_newave": ["SUDESTE", "SUDESTE"],
        }
    )


def _ree_dec_oper_ree_fixture() -> pl.DataFrame:
    """One REE (100), stage 1 (1-based), two nodes -- scenario-mean
    ``ena_MWmes=145.0``, ``earm_final_MWmes=1010.0``, deliberately offset from
    the Cobre-side fixture's ``150.0`` / ``1000.0`` (see
    :func:`_ree_cobre_hydro_fixture`) so the per-variable diff is
    hand-checkable rather than trivially zero."""
    return pl.DataFrame(
        {
            "estagio": [1, 1],
            "no": [1, 2],
            "cenario": [1, 1],
            "codigo_ree": [100, 100],
            "nome_ree": ["SUDESTE", "SUDESTE"],
            "codigo_submercado": [1, 1],
            "nome_submercado": ["SE", "SE"],
            "ena_MWmes": [140.0, 150.0],
            "earm_inicial_MWmes": [900.0, 900.0],
            "earm_inicial_percentual": [70.0, 70.0],
            "earm_final_MWmes": [1000.0, 1020.0],
            "earm_final_percentual": [72.0, 74.0],
            "earm_maximo_MWmes": [2000.0, 2000.0],
        }
    )


def _ree_aligned_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture()`` with its ``cobre_hydro`` extended to carry the
    ENA/EARM columns :func:`_cobre_ree_sums` reads -- the base fixture is
    trimmed to only the columns E1's ``_HYDRO_VARIABLES`` needs."""
    base = _aligned_fixture()
    return dataclasses.replace(
        base,
        cobre_hydro=base.cobre_hydro.with_columns(
            pl.Series("incremental_inflow_energy_mw", [90.0, 60.0]),
            pl.Series("stored_energy_final_mwh", [400000.0, 330000.0]),
        ),
    )


def _patch_ree_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire ``read_relato_membership``/``read_dec_oper_ree`` -- outside
    ``_read_aligned_frames`` -- to the fixtures above."""
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_relato_membership",
        lambda *_a, **_k: _ree_membership_fixture(),
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_dec_oper_ree",
        lambda *_a, **_k: _ree_dec_oper_ree_fixture(),
    )


def _extract_tab_content(html: str, tab_id: str) -> str:
    """Slice one ``id="tab-..."`` tab's content, up to the next ``id="tab-`` marker.

    Distinct from the same-named ``test_chart_helpers._extract_tab_content``,
    which is a ``<section>``-bounded matcher returning the inner group -- a
    different contract despite the shared name.
    """
    import re

    match = re.search(rf'id="{tab_id}".*?(?=id="tab-|\Z)', html, re.S)
    return match.group(0) if match else ""


def _fake_dataset(*, all_within_tol: bool = False) -> ComparisonDataset:
    """A canonical two-stage, two-variable dataset for the ``compare decomp``
    CLI tests. ``generation_mw`` always matches; ``turbined_m3s`` diverges
    unless *all_within_tol* asks for a fully-passing dataset instead."""
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "hydro", "hydro", "hydro"],
            "entity_id": [0, 0, 0, 0],
            "entity_name": ["A", "A", "A", "A"],
            "bus": [-1, -1, -1, -1],
            "stage": [0, 0, 1, 1],
            "block": [-1, -1, -1, -1],
            "variable": [
                "generation_mw",
                "generation_mw",
                "turbined_m3s",
                "turbined_m3s",
            ],
            "source": ["newave", "cobre", "newave", "cobre"],
            "value": [100.0, 100.0, 100.0, 100.0 if all_within_tol else 90.0],
        },
        schema=TIDY_SCHEMA,
    )
    turbined_within_tol_rate = 1.0 if all_within_tol else 0.0
    turbined_smape = 0.0 if all_within_tol else 0.12
    summary = pl.DataFrame(
        {
            "variable": ["generation_mw", "turbined_m3s"],
            "count": [1, 1],
            "mean_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "max_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "mean_smape": [0.0, turbined_smape],
            "max_smape": [0.0, turbined_smape],
            "within_tol_rate": [1.0, turbined_within_tol_rate],
            "correlation": [1.0, 0.9],
        },
        schema=SUMMARY_SCHEMA,
    )
    return ComparisonDataset(
        tidy=tidy,
        summary=summary,
        metadata={"unmapped": {"hydro": [], "thermal": [86, 224], "bus": []}},
    )


def _empty_fake_dataset() -> ComparisonDataset:
    return ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )


def _write_generic_constraints_case(
    case_dir: Path,
    constraints: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
) -> Path:
    """Write ``constraints/generic_constraints.json`` +
    ``constraints/generic_constraint_bounds.parquet`` under *case_dir* and
    return the Cobre output dir (``case_dir/output``) `case_dir_for`
    resolves back to *case_dir* from -- mirrors `_write_lines_json`."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    (constraints_dir / "generic_constraints.json").write_text(
        json.dumps({"constraints": constraints})
    )
    pd.DataFrame(
        bound_rows,
        columns=[
            "constraint_id",
            "stage_id",
            "block_id",
            "bound_lower",
            "bound_upper",
        ],
    ).to_parquet(constraints_dir / "generic_constraint_bounds.parquet")
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _usih_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    """A ``dec_oper_usih``-shaped frame: one stage-aggregate
    (``patamar=None``) row per (code, stage) -- the shape `_stage_rows`
    keeps."""
    base = {
        "no": 1,
        "cenario": 1,
        "patamar": None,
        "duracao": None,
        "vazao_defluente_m3s": 0.0,
        "vazao_turbinada_m3s": 0.0,
        "vazao_desviada_m3s": 0.0,
        "vazao_vertida_m3s": 0.0,
        "volume_util_final_hm3": 0.0,
        "geracao_MW": 0.0,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def _no_dec_oper(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """A ``read_dec_oper_usih``/``read_dec_oper_usit`` stub for "this deck has
    no such table": raises ``FileNotFoundError`` like the real reader would,
    so `_dec_oper_hydro_stage_frame`/`_dec_oper_thermal_stage_frame`'s own
    degrade-to-empty ``except`` path is exercised -- a bare ``pl.DataFrame()``
    (no columns at all) is not a shape the real reader ever returns (it
    raises on an empty parse) and trips `_scenario_mean`'s ``group_by``."""
    raise FileNotFoundError("dec_oper_*.csv not found")


def _make_confhd_df() -> pd.DataFrame:
    """Two hydros: plant 1 upstream of plant 2, in REE 1 (subsystem 1)."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2],
            "nome_usina": ["USINA_A", "USINA_B"],
            "posto": [1, 2],
            "codigo_usina_jusante": [pd.NA, 1],
            "ree": [1, 1],
            "volume_inicial_percentual": [50.0, 75.0],
            "usina_existente": ["EX", "EX"],
            "usina_modificada": [0, 0],
        }
    )


def _make_hidr_cadastro() -> pd.DataFrame:
    """Synthetic Hidr.cadastro for two plants.

    Both plants use ``tipo_regulacao="M"`` with a simple linear polynomial
    ``h(v) = 300 + 0.1*v`` (a0_volume_cota=300, a1_volume_cota=0.1, rest
    zero) and ``canal_fuga_medio=50.0``.  With ``tipo_perda=1`` and
    ``perdas=0.0`` the loss model leaves the net drop unchanged.

    For monthly-regulated plants the height is evaluated at 65% of useful storage
    (``v_65 = vmin + 0.65 * (vmax - vmin)``), matching the source model's
    ``produtibilidade_altura_65`` convention.

    USINA_A: [volume_minimo=100, volume_maximo=1000]
    - v_65 = 100 + 0.65 * 900 = 685.0
    - h(v_65) = 300 + 0.1 * 685.0 = 368.5
    - net_drop = 368.5 - 50.0 = 318.5
    - productivity_A = 0.9 * 318.5 = 286.65

    USINA_B: [volume_minimo=50, volume_maximo=500]
    - v_65 = 50 + 0.65 * 450 = 342.5
    - h(v_65) = 300 + 0.1 * 342.5 = 334.25
    - net_drop = 334.25 - 50.0 = 284.25
    - productivity_B = 0.85 * 284.25 = 241.6125

    Both productivities differ from their raw ``produtibilidade_especifica``
    values (0.9 and 0.85) because ``canal_fuga_medio`` is nonzero.
    """
    months = [
        "JAN",
        "FEV",
        "MAR",
        "ABR",
        "MAI",
        "JUN",
        "JUL",
        "AGO",
        "SET",
        "OUT",
        "NOV",
        "DEZ",
    ]
    base: dict[str, list] = {
        "nome_usina": ["USINA_A", "USINA_B"],
        "posto": [1, 2],
        "submercado": [1, 1],
        "empresa": [1, 1],
        "codigo_usina_jusante": [pd.NA, 1],
        "desvio": [pd.NA, pd.NA],
        "volume_minimo": [100.0, 50.0],
        "volume_maximo": [1000.0, 500.0],
        "volume_referencia": [550.0, 275.0],
        "canal_fuga_medio": [50.0, 50.0],
        "tipo_regulacao": ["M", "M"],
        "tipo_perda": [1, 1],
        "perdas": [0.0, 0.0],
        "a0_volume_cota": [300.0, 300.0],
        "a1_volume_cota": [0.1, 0.1],
        "a2_volume_cota": [0.0, 0.0],
        "a3_volume_cota": [0.0, 0.0],
        "a4_volume_cota": [0.0, 0.0],
        "produtibilidade_especifica": [0.9, 0.85],
        "numero_conjuntos_maquinas": [1, 2],
        "maquinas_conjunto_1": [4, 3],
        "maquinas_conjunto_2": [0, 2],
        "maquinas_conjunto_3": [0, 0],
        "maquinas_conjunto_4": [0, 0],
        "maquinas_conjunto_5": [0, 0],
        "potencia_nominal_conjunto_1": [200.0, 150.0],
        "potencia_nominal_conjunto_2": [0.0, 120.0],
        "potencia_nominal_conjunto_3": [0.0, 0.0],
        "potencia_nominal_conjunto_4": [0.0, 0.0],
        "potencia_nominal_conjunto_5": [0.0, 0.0],
        "vazao_nominal_conjunto_1": [222.2, 176.5],
        "vazao_nominal_conjunto_2": [0.0, 141.2],
        "vazao_nominal_conjunto_3": [0.0, 0.0],
        "vazao_nominal_conjunto_4": [0.0, 0.0],
        "vazao_nominal_conjunto_5": [0.0, 0.0],
        "vazao_minima_historica": [0, 0],
        "teif": [0.0, 0.0],
        "ip": [0.0, 0.0],
        "fator_carga_maximo": [1.0, 1.0],
        "fator_carga_minimo": [0.0, 0.0],
    }
    for m in months:
        base[f"evaporacao_{m}"] = [1.5, 2.0]

    df = pd.DataFrame(base, index=pd.Index([1, 2], name="codigo_usina"))
    return df


def _make_ree_df() -> pd.DataFrame:
    return pd.DataFrame({"codigo": [1], "nome": ["SE"], "submercado": [1]})


def _make_conft_df() -> pd.DataFrame:
    """Three thermals: 2 in subsystem 1, 1 in subsystem 2."""
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "submercado": [1, 1, 2],
            "usina_existente": ["EX", "EX", "EX"],
            "classe": [1, 1, 2],
        }
    )


def _make_clast_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "tipo_combustivel": ["GAS", "GAS", "OLEO"],
            "indice_ano_estudo": [1, 1, 1],
            "valor": [50.0, 80.0, 200.0],
        }
    )


def _make_term_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "codigo_usina": [10, 20, 30],
            "nome_usina": ["TERMO_A", "TERMO_B", "TERMO_C"],
            "potencia_instalada": [100.0, 200.0, 50.0],
            "fator_capacidade_maximo": [90.0, 100.0, 80.0],
            "teif": [0.05, 0.02, 0.10],
            "indisponibilidade_programada": [0.0, 0.0, 0.0],
            "mes": [1, 1, 1],
            "geracao_minima": [10.0, 0.0, 5.0],
        }
    )


def _make_deficit_df(n_patamares: int = 2) -> pd.DataFrame:
    """Deficit costs for subsystems 1 and 2 (non-fictitious) plus fictitious 99."""
    rows = []
    for sub, name, fict in [(1, "SE", 0), (2, "S", 0), (99, "FICT", 1)]:
        for pat in range(1, n_patamares + 1):
            rows.append(
                {
                    "codigo_submercado": sub,
                    "nome_submercado": name,
                    "ficticio": fict,
                    "patamar_deficit": pat,
                    "custo": 500.0 * pat,
                    "corte": 1000.0 if pat < n_patamares else None,
                }
            )
    return pd.DataFrame(rows)


def _make_intercambio_df() -> pd.DataFrame:
    """Three interchange pairs for subsystems 1, 2, 99."""
    import datetime

    d = datetime.datetime(2023, 1, 1)
    rows = [
        # 1 -> 2 direct (sentido=0 means de->para, i.e. 1->2)
        {
            "submercado_de": 1,
            "submercado_para": 2,
            "sentido": 0,
            "data": d,
            "valor": 3000.0,
        },
        # 2 -> 1 reverse (sentido=0 means de->para, i.e. 2->1)
        {
            "submercado_de": 2,
            "submercado_para": 1,
            "sentido": 0,
            "data": d,
            "valor": 2500.0,
        },
        # 1 -> 99 direct
        {
            "submercado_de": 1,
            "submercado_para": 99,
            "sentido": 0,
            "data": d,
            "valor": 4000.0,
        },
        # 99 -> 1 reverse
        {
            "submercado_de": 99,
            "submercado_para": 1,
            "sentido": 0,
            "data": d,
            "valor": 2000.0,
        },
        # 2 -> 99 direct
        {
            "submercado_de": 2,
            "submercado_para": 99,
            "sentido": 0,
            "data": d,
            "valor": 1500.0,
        },
        # 99 -> 2 reverse
        {
            "submercado_de": 99,
            "submercado_para": 2,
            "sentido": 0,
            "data": d,
            "valor": 1200.0,
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NE-with-filling fixtures (ticket-009): a JURUENA-shaped run-of-river ('S')
# plant (code 309) admitted into the active set by its exph dead-volume row.
# ---------------------------------------------------------------------------


def _make_ne_confhd_df() -> pd.DataFrame:
    """Two EX plants (1, 2) plus one NE filling plant (309 JURUENA)."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 309],
            "nome_usina": ["USINA_A", "USINA_B", "JURUENA"],
            "posto": [1, 2, 226],
            "codigo_usina_jusante": [pd.NA, 1, pd.NA],
            "ree": [1, 1, 1],
            "volume_inicial_percentual": [50.0, 75.0, 0.0],
            "usina_existente": ["EX", "EX", "NE"],
            "usina_modificada": [0, 0, 0],
        }
    )


def _make_ne_cadastro() -> pd.DataFrame:
    """Two-plant synthetic cadastro extended with JURUENA (code 309).

    JURUENA is run-of-river (``tipo_regulacao='S'``) with
    ``volume_minimo == volume_maximo == 2.93`` so its reservoir block already
    collapses to the single point 2.93 hm³ via the existing 'S' path.
    """
    base = _make_hidr_cadastro()
    juruena = base.loc[[1]].copy()
    juruena.index = pd.Index([309], name="codigo_usina")
    juruena["nome_usina"] = "JURUENA"
    juruena["posto"] = 226
    juruena["codigo_usina_jusante"] = pd.NA
    juruena["volume_minimo"] = 2.93
    juruena["volume_maximo"] = 2.93
    juruena["volume_referencia"] = 2.93
    juruena["tipo_regulacao"] = "S"
    return pd.concat([base, juruena])


def _make_ne_exph_mock(*, duracao: int = 1, volume_morto: float = 0.0) -> MagicMock:
    """An ``exph`` reader mock whose ``expansoes`` carries JURUENA's filling row.

    Mirrors the real ``exph.dat`` layout (verified on JURUENA): one schedule row
    (non-null ``data_inicio_enchimento``) — what ``filling_hydro_codes`` selects
    and the epic-03 ``convert_hydros`` filling tests read via ``.iloc[0]`` — then
    one row **per generating unit** with ``data_entrada_operacao`` /
    ``conjunto_maquina_entrada`` / ``maquina_entrada`` populated (epic-04). JURUENA
    has two machines, both entering Jan 2025 in machine group 1 (so under the
    Sep-2024 horizon their online stage is 4). The schedule row keeps a non-null
    ``data_entrada_operacao`` (per the epic-03 mock) but a NULL
    ``conjunto_maquina_entrada``, so the ramp branch — which filters unit rows on
    ``conjunto_maquina_entrada`` — never treats it as a generating unit.
    """
    expansoes = pd.DataFrame(
        {
            "codigo_usina": [309, 309, 309],
            "nome_usina": ["JURUENA", "JURUENA", "JURUENA"],
            "data_inicio_enchimento": [
                pd.Timestamp("2024-10-01"),
                pd.NaT,
                pd.NaT,
            ],
            "duracao_enchimento": [duracao, 0, 0],
            "volume_morto": [volume_morto, 0.0, 0.0],
            "data_entrada_operacao": [
                pd.Timestamp("2024-11-01"),
                pd.Timestamp("2025-01-01"),
                pd.Timestamp("2025-01-01"),
            ],
            "conjunto_maquina_entrada": [pd.NA, 1, 1],
            "maquina_entrada": [pd.NA, 1, 2],
        }
    )
    exph = MagicMock()
    exph.expansoes = expansoes
    return exph


def _ne_filling_case(tmp_path, *, duracao: int = 1, volume_morto: float = 0.0):
    """A ``NewaveCase`` with JURUENA (NE+filling) under a Sep-2024 3-year horizon.

    Study start Sep 2024 ⇒ stage 0 = Sep, stage 1 = Oct, stage 2 = Nov. JURUENA's
    Oct-2024 filling start maps to ``start_sid == 1``; with ``duracao == 1`` the
    entry is ``entry_sid == 2`` (design §5).
    """
    return _hydro_case(
        tmp_path,
        cadastro=_make_ne_cadastro(),
        confhd=_make_ne_confhd_df(),
        dger=_make_hydro_dger_mock(
            start_year=2024, start_month=9, num_anos=3, num_anos_pos=0
        ),
        exph=_make_ne_exph_mock(duracao=duracao, volume_morto=volume_morto),
    )


def _ne_filling_id_map() -> NewaveIdMap:
    """Id-map including JURUENA (309) so ``hydro_id(309)`` resolves."""
    return NewaveIdMap(
        subsystem_ids=[1],
        hydro_codes=[1, 2, 309],
        thermal_codes=[],
    )


def _make_hydro_dger_mock(
    *,
    start_year: int = 2024,
    start_month: int = 1,
    num_anos: int = 1,
    num_anos_pos: int = 0,
) -> MagicMock:
    """A ``dger`` mock that yields a concrete study horizon for ``convert_hydros``.

    ``convert_hydros`` builds the per-stage date list from ``case.horizon``
    (which reads these four ``dger`` fields) once, unconditionally, before the
    plant loop — so every ``_hydro_case`` needs a ``dger`` that resolves all four
    horizon fields (the module's other ``_make_dger_mock`` leaves
    ``num_anos_pos_estudo`` unset). ``funcao_producao_uhe = 1`` selects the linear
    (constant-productivity) generation model, matching the synthetic plants.
    """
    dger = MagicMock()
    dger.ano_inicio_estudo = start_year
    dger.mes_inicio_estudo = start_month
    dger.num_anos_estudo = num_anos
    dger.num_anos_pos_estudo = num_anos_pos
    # Historical-record start: drives the operational_start_date of always-in-service
    # (EX) plants. Set explicitly so it is a real int, not a MagicMock (whose int()
    # would otherwise resolve to 1).
    dger.ano_inicial_historico = 1931
    dger.funcao_producao_uhe = 1
    return dger


def _hydro_case(
    tmp_path,
    *,
    cadastro: pd.DataFrame | None = None,
    confhd: pd.DataFrame | None = None,
    rees: pd.DataFrame | None = None,
    modif=None,
    volref_volumes: pd.DataFrame | None = None,
    ghmin=None,
    penalid=None,
    dsvagua=None,
    dger=None,
    exph=None,
    **file_overrides,
):
    """Build a ``NewaveCase`` with mock hydro readers pre-cached.

    The three required hydro files (hidr/confhd/ree) default to the shared
    synthetic fixtures; pass ``cadastro`` / ``confhd`` / ``rees`` DataFrames to
    override. Optional readers (modif, volref_saz, ghmin, penalid, dsvagua) are
    passed as already-built mock reader objects; for those guarded behind a
    ``case.files.X`` path check, set the matching path via ``file_overrides``
    (e.g. ``volref_saz=tmp_path / "volref_saz.dat"``).

    ``dger`` defaults to :func:`_make_hydro_dger_mock` (a Jan-2024 one-year horizon) so
    ``case.horizon`` always resolves; pass a custom ``dger`` mock to drive a
    different horizon. ``exph`` is the dead-volume filling reader mock (``None``
    by default — EX-only cases admit no filling plant).
    """
    mock_hidr = MagicMock()
    mock_hidr.cadastro = _make_hidr_cadastro() if cadastro is None else cadastro

    mock_confhd = MagicMock()
    mock_confhd.usinas = _make_confhd_df() if confhd is None else confhd

    mock_ree = MagicMock()
    mock_ree.rees = _make_ree_df() if rees is None else rees

    parsed: dict = {
        "hidr": mock_hidr,
        "confhd": mock_confhd,
        "ree": mock_ree,
        "dger": _make_hydro_dger_mock() if dger is None else dger,
        "exph": exph,
    }

    if volref_volumes is not None:
        mock_volref = MagicMock()
        mock_volref.volumes = volref_volumes
        parsed["volref_saz"] = mock_volref
    if modif is not None:
        parsed["modif"] = modif
    if ghmin is not None:
        parsed["ghmin"] = ghmin
    if penalid is not None:
        parsed["penalid"] = penalid
    if dsvagua is not None:
        parsed["dsvagua"] = dsvagua

    files = make_nw_files(tmp_path, **file_overrides)
    return make_case(files, **parsed)


def _make_prod_model_dger_mock(
    *,
    ano_inicio: int = 2025,
    mes_inicio: int = 1,
    num_anos: int = 5,
    num_anos_pos: int = 0,
) -> MagicMock:
    """Return a mock Dger object for use in production model tests."""
    m = MagicMock()
    m.ano_inicio_estudo = ano_inicio
    m.mes_inicio_estudo = mes_inicio
    m.num_anos_estudo = num_anos
    m.num_anos_pos_estudo = num_anos_pos
    return m


def _make_cfuga_rec(month: int, year: int, nivel: float) -> MagicMock:
    import datetime

    r = MagicMock()
    type(r).__name__ = "CFUGA"
    r.data_inicio = datetime.datetime(year, month, 1)
    r.nivel = nivel
    return r


def _thermal_readers():
    conft = MagicMock()
    conft.usinas = _make_conft_df()
    clast = MagicMock()
    clast.usinas = _make_clast_df()
    clast.modificacoes = None
    term = MagicMock()
    term.usinas = _make_term_df()
    return conft, clast, term


def _make_sistema_mock() -> MagicMock:
    """Build the ``Sistema`` reader mock shared by the network tests."""
    mock_sistema = MagicMock()
    mock_sistema.custo_deficit = _make_deficit_df(n_patamares=2)
    mock_sistema.limites_intercambio = _make_intercambio_df()
    return mock_sistema


def _make_penalid_df() -> pd.DataFrame:
    """Synthetic PENALID.DAT penalties for two REEs and several variables.

    REE 1 has DESVIO=8300.0, VAZMIN=3179.35, GHMIN=4500.0 at patamar 1.
    REE 2 has DESVIO=9100.0, VAZMIN=2800.0 at patamar 1.
    Both REEs have patamar 2 rows with NaN values (unbounded tier).
    TURBMX is included to verify the "no mapping" skip path.
    """
    return pd.DataFrame(
        {
            "variavel": [
                "DESVIO",
                "DESVIO",
                "VAZMIN",
                "VAZMIN",
                "GHMIN",
                "GHMIN",
                "TURBMX",
                "TURBMX",
                "DESVIO",
                "DESVIO",
                "VAZMIN",
                "VAZMIN",
            ],
            "codigo_ree_submercado": [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
            "patamar_penalidade": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "patamar_carga": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "valor_R$_MWh": [
                8300.0,
                math.nan,
                3179.35,
                math.nan,
                4500.0,
                math.nan,
                999.0,  # TURBMX — should be skipped (no mapping)
                math.nan,
                9100.0,
                math.nan,
                2800.0,
                math.nan,
            ],
            "valor_R$_hm3": [0.0] * 12,
        }
    )


def _make_dger_mock(start_year: int, start_month: int, num_anos: int) -> MagicMock:
    """Build a MagicMock mimicking the Dger object."""
    mock = MagicMock()
    mock.ano_inicio_estudo = start_year
    mock.mes_inicio_estudo = start_month
    mock.num_anos_estudo = num_anos
    return mock


# ---------------------------------------------------------------------------
# CLI subprocess runner and pipeline-mock fixtures (shared by the CLI, schema
# registry, and convert-failpath tests).
# ---------------------------------------------------------------------------

# Fake data for converter functions.
_FAKE_LOAD_FACTORS: dict = {"load_factors": []}
_FAKE_LINE_BOUNDS_TABLE = pa.table(
    {
        "line_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "direct_mw": pa.array([], type=pa.float64()),
        "reverse_mw": pa.array([], type=pa.float64()),
        "block_id": pa.array([], type=pa.int32()),
    }
)
_FAKE_NCS: dict = {"non_controllable_sources": []}
_FAKE_NCS_FACTORS: dict = {"non_controllable_factors": []}
_FAKE_NCS_BOUNDS_TABLE = pa.table(
    {
        "ncs_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "available_generation_mw": pa.array([], type=pa.float64()),
    }
)


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the cobre-bridge entry point as a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "cobre_bridge.cli", *args],
        capture_output=True,
        text=True,
    )


_ARQUIVOS_DAT_CONTENT = """\
DADOS GERAIS                : dger.dat
DADOS DOS SUBSISTEMAS       : sistema.dat
CONFIGURACAO HIDRAULICA     : confhd.dat
ALTERACAO DADOS USINAS HIDRO: modif.dat
CONFIGURACAO TERMICA        : conft.dat
DADOS DAS USINAS TERMICAS   : term.dat
DADOS DAS CLASSES TERMICAS  : clast.dat
DADOS DE EXPANSAO HIDRAULICA: exph.dat
ARQUIVO DE EXPANSAO TERMICA : expt.dat
ARQUIVO DE PATAMARES MERCADO: patamar.dat
ARQUIVO DE CORTES DE BENDERS: cortes.dat
ARQUIVO DE CABECALHO CORTES : cortesh.dat
RELATORIO DE CONVERGENCIA   : pmo.dat
RELATORIO DE E. SINTETICAS  : parp.dat
RELATORIO DETALHADO FORWARD : forward.dat
ARQUIVO DE CABECALHO FORWARD: forwarh.dat
ARQUIVO DE S.HISTORICAS S.F.: shist.dat
ARQUIVO DE MANUT.PROG. UTE'S: manutt.dat
ARQUIVO P/DESPACHO HIDROTERM: newdesp.dat
ARQUIVO C/TEND. HIDROLOGICA : vazpast.dat
ARQUIVO C/DADOS DE ITAIPU   : itaipu.dat
ARQUIVO C/DEMAND S. BIDDING : bid.dat
ARQUIVO C/CARGAS ADICIONAIS : c_adic.dat
ARQUIVO C/FATORES DE PERDAS : loss.dat
ARQUIVO C/PATAMARES GTMIN   : gtminpat.dat
ARQUIVO ENSO 1              : elnino.dat
ARQUIVO ENSO 2              : ensoaux.dat
ARQUIVO DSVAGUA             : dsvagua.dat
ARQUIVO P/PENALID. POR DESV.: penalid.dat
ARQUIVO C.GUIA / PENAL.VMINT: curva.dat
ARQUIVO AGRUPAMENTO LIVRE   : agrint.dat
ARQUIVO DESP. ANTEC. GNL    : adterm.dat
ARQUIVO GER. HIDR. MIN      : ghmin.dat
ARQUIVO AVERSAO RISCO - SAR : sar.dat
ARQUIVO AVERSAO RISCO - CVAR: cvar.dat
DADOS DOS RESER.EQ.ENERGIA  : ree.dat
ARQUIVO RESTRICOES ELETRICAS: re.dat
ARQUIVO DE TECNOLOGIAS      : tecno.dat
DADOS DE ABERTURAS          : abertura.dat
ARQUIVO DE EMISSOES GEE     : gee.dat
ARQUIVO DE RESTRICAO DE GAS : clasgas.dat
ARQUIVO DE DADOS SIM. FINAL : simfinal.dat
ARQ. DE CORTES POS ESTUDO   : cortes-pos.dat
ARQ. DE CABECALHO CORTES POS: cortesh-pos.dat
ARQ. C/ VOLUME REF. SAZONAL : volref_saz.dat
"""

_REQUIRED_STUB_FILES = [
    "dger.dat",
    "confhd.dat",
    "conft.dat",
    "sistema.dat",
    "clast.dat",
    "term.dat",
    "ree.dat",
    "patamar.dat",
    "hidr.dat",
    "vazoes.dat",
]


def _make_fake_newave_dir(tmp_path: Path) -> Path:
    """Create a directory with caso.dat, arquivos.dat, and all required stub files."""
    newave_dir = tmp_path / "newave_case"
    newave_dir.mkdir()
    (newave_dir / "caso.dat").write_text("arquivos.dat\n")
    (newave_dir / "arquivos.dat").write_text(_ARQUIVOS_DAT_CONTENT)
    for filename in _REQUIRED_STUB_FILES:
        (newave_dir / filename).write_text("stub")
    return newave_dir


# Canned return values used across pipeline tests.
_FAKE_HYDROS = {"$schema": "http://example", "hydros": [{"id": 0}, {"id": 1}]}
_FAKE_THERMALS = {"$schema": "http://example", "thermals": [{"id": 0}]}
_FAKE_BUSES = {"$schema": "http://example", "buses": [{"id": 0}, {"id": 1}, {"id": 2}]}
_FAKE_LINES = {"$schema": "http://example", "lines": [{"id": 0}]}
_FAKE_PENALTIES = {"bus": {}, "hydro": {}, "line": {}, "non_controllable_source": {}}
_FAKE_STAGES = {
    "$schema": "http://example",
    "policy_graph": {"type": "finite_horizon"},
    "stages": [{"id": i} for i in range(12)],
}
_FAKE_CONFIG = {
    "$schema": "http://example",
    "training": {
        "selection": {"method": "sampled", "forward_passes": 5},
        "stopping_rules": [],
    },
    "simulation": {
        "enabled": True,
        "selection": {"method": "sampled", "num_scenarios": 200},
    },
}
_FAKE_IC = {"$schema": "http://example", "storage": [], "filling_storage": []}
_FAKE_INFLOW_TABLE = pa.table(
    {
        "hydro_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "mean_m3s": pa.array([100.0], type=pa.float64()),
        "std_m3s": pa.array([10.0], type=pa.float64()),
    }
)
_FAKE_LOAD_TABLE = pa.table(
    {
        "bus_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "mean_mw": pa.array([500.0], type=pa.float64()),
        "std_mw": pa.array([0.0], type=pa.float64()),
    }
)
_FAKE_HYDRO_ENERGY_PRODUCTIVITY_TABLE = pa.table(
    {
        "hydro_id": pa.array([0, 1], type=pa.int32()),
        "stage_id": pa.array([None, None], type=pa.int32()),
        "equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64()),
        "reference_outflow_m3s": pa.array([None, None], type=pa.float64()),
        "specific_productivity_mw_per_m3s_per_m": pa.array(
            [None, None], type=pa.float64()
        ),
    }
)


def _all_converter_patches(fake_id_map: MagicMock) -> list:  # type: ignore[type-arg]
    """Return patch context managers for all converter functions.

    The parsed case is mocked at ``NewaveCase.from_directory``; its ``id_map``
    is the supplied ``fake_id_map`` (the pipeline now reads ``case.id_map``).
    """
    fake_case = MagicMock()
    fake_case.id_map = fake_id_map
    return [
        patch(
            "cobre_bridge.pipeline.NewaveCase.from_directory",
            return_value=fake_case,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_hydros",
            return_value=_FAKE_HYDROS,
        ),
        patch(
            "cobre_bridge.pipeline.thermal_conv.convert_thermals",
            return_value=_FAKE_THERMALS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_buses",
            return_value=_FAKE_BUSES,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_lines",
            return_value=_FAKE_LINES,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_penalties",
            return_value=_FAKE_PENALTIES,
        ),
        patch(
            "cobre_bridge.pipeline.temporal_conv.convert_stages",
            return_value=_FAKE_STAGES,
        ),
        patch(
            "cobre_bridge.pipeline.temporal_conv.convert_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "cobre_bridge.pipeline.ic_conv.convert_initial_conditions",
            return_value=_FAKE_IC,
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_inflow_stats",
            return_value=_FAKE_INFLOW_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_load_stats",
            return_value=_FAKE_LOAD_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.inflow_windows.convert_recent_observation_windows",
            return_value=[],
        ),
        patch(
            "cobre_bridge.pipeline.inflow_windows.convert_inflow_history_windows",
            return_value=_FAKE_INFLOW_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.read_cadastro",
            return_value=MagicMock(),
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.generate_hydro_geometry",
            return_value=_FAKE_INFLOW_TABLE,  # reuse any small pa.Table
        ),
        patch(
            "cobre_bridge.pipeline.constraints_conv.convert_vminop_constraints",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.constraints_conv.convert_electric_constraints",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.constraints_conv.convert_agrint_constraints",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_load_factors",
            return_value=_FAKE_LOAD_FACTORS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_line_bounds",
            return_value=_FAKE_LINE_BOUNDS_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_non_controllable_sources",
            return_value=_FAKE_NCS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_ncs_factors",
            return_value=_FAKE_NCS_FACTORS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_ncs_stats",
            return_value=_FAKE_NCS_BOUNDS_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_production_models",
            return_value={"production_models": []},
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.compute_base_productivities",
            return_value={},
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_hydro_energy_productivity",
            return_value=_FAKE_HYDRO_ENERGY_PRODUCTIVITY_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.thermal_conv.convert_thermal_bounds",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_storage_bounds",
            return_value=None,
        ),
    ]


def _run_with_all_mocks(src: Path, dst: Path) -> object:
    """Run convert_newave_case with all converters replaced by canned fakes."""
    import contextlib

    from cobre_bridge.pipeline import convert_newave_case

    fake_id_map = MagicMock()
    with contextlib.ExitStack() as stack:
        for p in _all_converter_patches(fake_id_map):
            stack.enter_context(p)
        return convert_newave_case(src, dst)
