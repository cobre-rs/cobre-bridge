"""Cut-level round-trip test for the boundary FCF importer (epic 4 ticket-013).

Epic 2 (``fcf/__init__.py::import_boundary_fcf``) proved only that
``cobre validate`` exits 0 on the boundary-injected case — a config/structure
gate that never evaluates a single cut coefficient. This module is the
epic's primary **value-level** correctness gate: it reloads the authored
checkpoint (``cobre.results.load_policy(case_dir, policy_subdir="boundary")``)
and cross-evaluates its cut coefficients against a *direct, independent*
evaluation of the parsed ``cortesh.dat``/``cortes-010.dat`` cuts.

The oracle here never calls
:func:`cobre_bridge.decomp.fcf.mapper.map_boundary_cuts` — the mapper's
coefficient placement is the thing under test, so routing the oracle through
it would make this a circular, Python-vs-Python check. Instead this module
independently parses the source cuts (``Cortesh.read`` + ``read_cortes`` +
``DecompIdMap.from_dadger``) and builds its own physical<->slot join from the
*reloaded* ``entity_manifest`` — the ground truth for where the writer
actually placed each mapped coefficient — never from the mapper's own
bookkeeping.

Agreement to f64 tolerance (``policy.fbs`` stores ``coefficients`` as
``float64``, so authored values round-trip through cobre-io exactly) across
both a direct coefficient comparison and a deterministic evaluation sweep
catches a sign flip, a unit error, a lag off-by-one, or a dropped
``cost_scale_factor`` marker (which makes cobre silently scale every value
by 10⁶) in one test.

Alongside that tier-3 (real deck + local cobre binary) gate, this module also
carries a tier-2 synthetic round trip
(``test_synthetic_roundtrip_coefficient_identity``,
``test_synthetic_roundtrip_theta_sweep``) that proves the same coefficient
and evaluation identity against a hand-authored ``BoundaryCuts``/
``TerminalManifest``/``DecompIdMap`` — no deck, no cobre binary. Gated on
``@requires_cobre_python`` (cobre is import-able) stacked with
``@requires_writer_binding`` (the installed wheel actually exposes the
``write_policy_checkpoint`` binding this path calls — an older, importable
wheel lacking it would otherwise fail at runtime with ``AttributeError``
instead of skipping). It reuses this module's own
``_resolve``/``_theta_source``/``_theta_cobre`` oracle, keyed off the
synthetic inputs instead of a real deck parse.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# idecomp's own __init__ does not list Dadger in __all__, so mypy treats
# this as an implicit re-export under --strict (the same pre-existing gap
# id_map.py/pipeline.py carry unsuppressed for this identical import).
from idecomp.decomp import Dadger  # type: ignore[attr-defined]
from inewave.newave import Cortesh

from cobre_bridge.decomp.fcf import import_boundary_fcf
from cobre_bridge.decomp.fcf.bootstrap import TerminalManifest
from cobre_bridge.decomp.fcf.cortes import BoundaryCuts, StageCutRecord, read_cortes
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.pipeline import convert_decomp_case, discover_decomp_files
from tests._fcf_fixtures import (
    make_boundary_cuts,
    make_cut_record,
    make_id_map,
    make_manifest,
    make_slot,
    synthetic_roundtrip,
)
from tests.conftest import (
    has_writer_binding,
    requires_cobre_python,
    requires_writer_binding,
)

# Same real, gitignored deck + local cobre build as test_decomp_fcf_importer.py
# (identical constants, copied verbatim) — CI has neither, so every tier-3
# test below is skipif-guarded.
_DECK = Path("example/decomp-set-24-rv0")
_CORTESH = _DECK / "cortesh.dat"
_CORTES = _DECK / "cortes-010.dat"
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"


# ``has_writer_binding()`` (tests/conftest.py) is import-free-first (never a
# module-top ``import cobre``), so this module still collects in a
# cobre-free environment (the same convention
# ``fcf/bootstrap.py::ensure_writer_binding`` uses in the library code this
# test exercises).
_HAS_WRITER_BINDING = has_writer_binding()
_HAS_E2E_DEPS = _COBRE_BIN.exists() and _DECK.exists() and _HAS_WRITER_BINDING
_skip_e2e = pytest.mark.skipif(
    not _HAS_E2E_DEPS,
    reason=(
        f"requires the local cobre binary ({_COBRE_BIN}), the "
        f"decomp-set-24-rv0 deck ({_DECK}), and the write_policy_checkpoint "
        "writer binding"
    ),
)

#: cobre `policy.fbs` entity_type codes (mirrors `fcf/mapper.py`'s private
#: constants of the same name/value — re-declared here, never imported, so
#: this oracle's slot lookup never depends on the mapper under test).
_HYDRO_STORAGE = 0
_HYDRO_INFLOW_LAG = 1
_HYDRO_TRANSIT_BUCKET = 3
_STORAGE_SUBINDEX = 0

#: Random states in the AC 4 sweep beyond the all-zero and per-plant
#: unit-storage states — 2 + 8 = 10, comfortably above the ticket's >= 8
#: floor. Shared by the tier-2 synthetic sweep test below.
_N_RANDOM_STATES = 8

#: The tier-2 synthetic case's two resolved plant codes (joining hydro ids 0
#: and 1) and its one source-only, D3-dropped plant code.
_RESOLVED_PLANT_CODES = (10, 20)
_DROPPED_PLANT_CODE = 30


@dataclass(frozen=True)
class _RoundTripCase:
    """One converted-and-boundary-imported case, plus its independent oracle.

    ``reloaded`` is the authored checkpoint as read back by
    ``cobre.results.load_policy``; ``bcuts``/``id_map`` are a *second*,
    independent parse of the same source files, built without ever calling
    the mapper under test.
    """

    case_dir: Path
    reloaded: dict[str, Any]
    bcuts: BoundaryCuts
    id_map: DecompIdMap


@dataclass(frozen=True)
class _PhysicalState:
    """One synthetic terminal-state probe, keyed by source plant index.

    ``storage[plant_index]`` and ``inflow[(plant_index, depth)]`` are plain
    physical values — never a dense state-dimension vector until a caller
    projects them onto the manifest's slot positions (``_theta_cobre``'s
    ``x_cobre`` argument) or reads them off directly (``_theta_source``).
    A key absent from either mapping is implicitly 0.0.
    """

    storage: dict[int, float]
    inflow: dict[tuple[int, int], float]


def _slot_index(
    manifest: Sequence[Mapping[str, Any]],
    entity_type: int,
    entity_id: int,
    subindex: int,
) -> int | None:
    """Position of the manifest slot keyed ``(entity_type, entity_id, subindex)``.

    A linear scan over the *reloaded* ``entity_manifest`` — the ground truth
    for where the checkpoint writer actually placed each coefficient — never
    `fcf.mapper._index_manifest`, which is the mapper's own bookkeeping and
    would make this oracle circular.
    """
    for position, slot in enumerate(manifest):
        if (
            slot["entity_type"] == entity_type
            and slot["entity_id"] == entity_id
            and slot["subindex"] == subindex
        ):
            return position
    return None


def _resolve(
    plant_codes: Sequence[int],
    id_map: DecompIdMap,
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, tuple[int, int, dict[int, int]]], frozenset[int]]:
    """Resolve each source plant index to its target manifest slots.

    Returns ``(resolved, dropped)``. ``resolved[plant_index] = (hydro_id,
    storage_position, {lag_depth: lag_position})`` for a plant that both
    resolves via ``id_map.hydro_id`` and has a ``HydroStorage`` slot in
    ``manifest``; ``dropped`` is the set of plant indices excluded for
    either reason (D3 — codes 132 and 176 on ``decomp-set-24-rv0``).
    ``lag_positions`` only carries depths 1..12 whose
    ``(HydroInflowLag, hydro_id, depth-1)`` slot is actually present — a
    hydro-specific lag depth shorter than 12 is a legitimate case shape,
    not a read bug.
    """
    resolved: dict[int, tuple[int, int, dict[int, int]]] = {}
    dropped: set[int] = set()
    for plant_index, code in enumerate(plant_codes):
        try:
            hydro_id = id_map.hydro_id(code)
        except KeyError:
            dropped.add(plant_index)
            continue
        storage_position = _slot_index(
            manifest, _HYDRO_STORAGE, hydro_id, _STORAGE_SUBINDEX
        )
        if storage_position is None:
            dropped.add(plant_index)
            continue
        lag_positions: dict[int, int] = {}
        for depth in range(1, 13):
            position = _slot_index(manifest, _HYDRO_INFLOW_LAG, hydro_id, depth - 1)
            if position is not None:
                lag_positions[depth] = position
        resolved[plant_index] = (hydro_id, storage_position, lag_positions)
    return resolved, frozenset(dropped)


def _theta_source(
    records: Sequence[StageCutRecord],
    resolved: Mapping[int, tuple[int, int, dict[int, int]]],
    state: _PhysicalState,
) -> float:
    """``max_k(rhs_k + pi_varm_k . storage + pi_qafl_k . inflow)`` over active
    source records, restricted to resolved plants and present lag slots —
    the independent oracle AC 4 cross-evaluates against ``_theta_cobre``.
    """
    best = float("-inf")
    for record in records:
        if not record.is_active:
            continue
        value = record.rhs
        for plant_index, (
            _hydro_id,
            _storage_position,
            lag_positions,
        ) in resolved.items():
            value += record.pi_varm[plant_index] * state.storage.get(plant_index, 0.0)
            plant_lags = record.pi_qafl[plant_index]
            for depth in lag_positions:
                value += plant_lags[depth - 1] * state.inflow.get(
                    (plant_index, depth), 0.0
                )
        if value > best:
            best = value
    return best


def _theta_cobre(cuts: Sequence[Mapping[str, Any]], x_cobre: Sequence[float]) -> float:
    """``max_k(intercept_k + coefficients_k . x_cobre)`` over active reloaded
    cuts — cobre's own ``Policy.evaluate`` semantics, computed directly from
    the checkpoint the writer authored.
    """
    best = float("-inf")
    for cut in cuts:
        if not cut["is_active"]:
            continue
        value = float(cut["intercept"]) + sum(
            c * x for c, x in zip(cut["coefficients"], x_cobre, strict=True)
        )
        if value > best:
            best = value
    return best


# ---------------------------------------------------------------------------
# Tier 2 — in-wheel synthetic round trip (no deck, no cobre binary).
# ---------------------------------------------------------------------------


def _synthetic_two_plant_case() -> tuple[
    tuple[int, ...], DecompIdMap, TerminalManifest
]:
    """A synthetic two-resolved-plant, one-dropped-plant case.

    Plant codes 10 and 20 resolve to hydro ids 0 and 1 (each with a full
    12-slot ``HydroInflowLag`` family); plant code 30 is source-only —
    absent from ``id_map``, so ``map_boundary_cuts`` drops it (D3). One
    trailing ``HydroTransitBucket`` slot is never targeted by the mapper,
    giving the coefficient-identity test below an unmapped slot to check
    against 0.0.
    """
    plant_codes = (*_RESOLVED_PLANT_CODES, _DROPPED_PLANT_CODE)
    id_map = make_id_map(_RESOLVED_PLANT_CODES)
    manifest = make_manifest(
        [
            make_slot(_HYDRO_STORAGE, 0, 0),  # position 0: plant 10
            make_slot(_HYDRO_STORAGE, 1, 0),  # position 1: plant 20
            *[
                make_slot(_HYDRO_INFLOW_LAG, 0, lag)  # positions 2..13
                for lag in range(12)
            ],
            *[
                make_slot(_HYDRO_INFLOW_LAG, 1, lag)  # positions 14..25
                for lag in range(12)
            ],
            make_slot(_HYDRO_TRANSIT_BUCKET, 0, 0),  # position 26: unmapped
        ]
    )
    return plant_codes, id_map, manifest


@requires_cobre_python
@requires_writer_binding
def test_synthetic_roundtrip_coefficient_identity(tmp_path: Path) -> None:
    """AC 1/2 — coefficient identity, no deck and no cobre binary.

    Reuses this module's own ``_resolve`` oracle (never
    ``fcf.mapper.map_boundary_cuts``'s bookkeeping) against a hand-authored
    multi-cut ``BoundaryCuts`` with one active and one inactive record: for
    every active reloaded cut, each resolved plant's ``HydroStorage``
    coefficient equals the source ``pi_varm``, each present
    ``HydroInflowLag`` coefficient equals the source ``pi_qafl`` at that
    depth (nonzero at depths 1-2, proving no off-by-one), and every unmapped
    slot is exactly 0.0. Also checks the intercept and the
    ``cost_scale_factor == 1.0`` marker survive the round trip.
    """
    plant_codes, id_map, manifest = _synthetic_two_plant_case()
    active_record = make_cut_record(
        pi_varm=(3.0, 5.0, 7.0),
        pi_qafl=((1.5, 2.5) + (0.0,) * 10, (0.0,) * 12, (0.0,) * 12),
        rhs=100.0,
        cut_id=1,
        iteration=1,
        forward_pass_index=1,
        is_active=True,
    )
    inactive_record = make_cut_record(
        pi_varm=(30.0, 50.0, 70.0),
        rhs=999.0,
        cut_id=2,
        iteration=2,
        forward_pass_index=1,
        is_active=False,
    )
    cuts = make_boundary_cuts(plant_codes, (active_record, inactive_record))

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)
    assert reloaded["metadata"]["producer"]["cost_scale_factor"] == 1.0

    entry = reloaded["stage_cuts"][0]
    resolved, dropped = _resolve(plant_codes, id_map, entry["entity_manifest"])
    assert dropped == frozenset({2})
    assert len(resolved) == 2

    mapped_positions: set[int] = set()
    for _hydro_id, storage_position, lag_positions in resolved.values():
        mapped_positions.add(storage_position)
        mapped_positions.update(lag_positions.values())

    active_cuts = [cut for cut in entry["cuts"] if cut["is_active"]]
    assert len(active_cuts) == 1
    cut = active_cuts[0]
    coefficients = cut["coefficients"]
    assert cut["intercept"] == active_record.rhs

    for plant_index, (_hydro_id, storage_position, lag_positions) in resolved.items():
        source_storage = active_record.pi_varm[plant_index]
        assert math.isclose(
            coefficients[storage_position], source_storage, rel_tol=1e-9, abs_tol=1e-6
        ), f"plant_index={plant_index} storage slot {storage_position}"
        plant_lags = active_record.pi_qafl[plant_index]
        for depth, lag_position in lag_positions.items():
            assert math.isclose(
                coefficients[lag_position],
                plant_lags[depth - 1],
                rel_tol=1e-9,
                abs_tol=1e-6,
            ), f"plant_index={plant_index} lag depth={depth} slot {lag_position}"

    for position, value in enumerate(coefficients):
        if position not in mapped_positions:
            assert value == 0.0, f"unmapped slot {position} = {value!r}"


@requires_cobre_python
@requires_writer_binding
def test_synthetic_roundtrip_theta_sweep(tmp_path: Path) -> None:
    """AC 3 — ``theta_cobre(x) == theta_source(x)`` over >= 8 states, no deck
    and no cobre binary.

    Reuses this module's own ``_theta_source``/``_theta_cobre`` oracle
    against the same synthetic two-plant case, over the all-zero,
    per-plant-unit-storage, and ``rng(0)``-random states.
    """
    plant_codes, id_map, manifest = _synthetic_two_plant_case()
    records = (
        make_cut_record(
            pi_varm=(3.0, 5.0, 7.0),
            pi_qafl=((1.5, 2.5) + (0.0,) * 10, (0.4,) + (0.0,) * 11, (0.0,) * 12),
            rhs=100.0,
            cut_id=1,
            iteration=1,
            forward_pass_index=1,
            is_active=True,
        ),
        make_cut_record(
            pi_varm=(1.0, -2.0, 4.0),
            pi_qafl=((0.2,) * 12, (0.0,) * 12, (0.0,) * 12),
            rhs=-50.0,
            cut_id=2,
            iteration=2,
            forward_pass_index=1,
            is_active=True,
        ),
        make_cut_record(
            pi_varm=(30.0, 50.0, 70.0),
            rhs=999.0,
            cut_id=3,
            iteration=3,
            forward_pass_index=1,
            is_active=False,
        ),
    )
    cuts = make_boundary_cuts(plant_codes, records)

    reloaded = synthetic_roundtrip(tmp_path / "boundary", cuts, manifest, id_map)
    entry = reloaded["stage_cuts"][0]
    cuts_list = entry["cuts"]
    state_dimension = entry["state_dimension"]

    resolved, dropped = _resolve(plant_codes, id_map, entry["entity_manifest"])
    assert dropped == frozenset({2})

    rng = np.random.default_rng(0)
    states: list[_PhysicalState] = [
        _PhysicalState(storage={}, inflow={}),
        _PhysicalState(storage={i: 1.0 for i in resolved}, inflow={}),
    ]
    for _ in range(_N_RANDOM_STATES):
        storage = {i: float(rng.normal(scale=1000.0)) for i in resolved}
        inflow = {
            (i, depth): float(rng.normal(scale=1000.0))
            for i, (_hydro_id, _storage_position, lag_positions) in resolved.items()
            for depth in lag_positions
        }
        states.append(_PhysicalState(storage=storage, inflow=inflow))
    assert len(states) >= 8

    for index, state in enumerate(states):
        x_cobre = [0.0] * state_dimension
        for plant_index, (
            _hydro_id,
            storage_position,
            lag_positions,
        ) in resolved.items():
            x_cobre[storage_position] = state.storage.get(plant_index, 0.0)
            for depth, lag_position in lag_positions.items():
                x_cobre[lag_position] = state.inflow.get((plant_index, depth), 0.0)

        theta_cobre = _theta_cobre(cuts_list, x_cobre)
        theta_source = _theta_source(records, resolved, state)
        assert math.isclose(theta_cobre, theta_source, rel_tol=1e-9, abs_tol=1e-6), (
            f"state {index}: theta_cobre={theta_cobre!r} != "
            f"theta_source={theta_source!r}"
        )


# ---------------------------------------------------------------------------
# Tier 3 — real deck + local cobre binary (dev-only smoke; skips on CI).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def roundtrip_case(tmp_path_factory: pytest.TempPathFactory) -> _RoundTripCase:
    """Convert ``decomp-set-24-rv0``, import its own boundary FCF, reload the
    checkpoint, and independently re-parse the source cuts — once.

    Module-scoped, mirroring ``test_decomp_fcf_importer.py``'s
    ``imported_case``: the deck's ``cortes-010.dat`` is ~176 MB and the
    bootstrap stage runs a real ``cobre run``, so this ~30s+ path executes
    exactly once regardless of how many of the four AC tests below exercise
    its result.
    """
    if not _HAS_E2E_DEPS:
        pytest.skip(
            f"requires the local cobre binary ({_COBRE_BIN}), the "
            f"decomp-set-24-rv0 deck ({_DECK}), and the "
            "write_policy_checkpoint writer binding"
        )

    import cobre

    root = tmp_path_factory.mktemp("fcf_roundtrip_e2e")
    case_dir = root / "converted"
    convert_decomp_case(_DECK, case_dir, force=True)

    boundary_dir = import_boundary_fcf(
        case_dir,
        _CORTESH,
        _CORTES,
        cobre_bin=_COBRE_BIN,
        work_dir=root / "work",
        cost_scale_factor=1.0,
    )
    assert boundary_dir is not None

    reloaded = cobre.results.load_policy(case_dir, policy_subdir="boundary")

    # Independent oracle parse — deliberately re-reads the same source files
    # from scratch rather than reusing anything import_boundary_fcf computed,
    # and never calls map_boundary_cuts.
    deck_files = discover_decomp_files(_CORTESH.parent)
    dadger = Dadger.read(str(deck_files.dadger))
    # Dadger.read()'s stub return type is the base RegisterFile (same
    # pre-existing idecomp stub gap pipeline.py/fcf/__init__.py hit on this
    # identical from_dadger(dadger) call).
    id_map = DecompIdMap.from_dadger(dadger)  # type: ignore[arg-type]
    cortesh = Cortesh.read(str(_CORTESH))
    bcuts = read_cortes(_CORTES, cortesh, boundary_stage=None)

    cuts_list = reloaded["stage_cuts"][0]["cuts"]
    for k, (cut, record) in enumerate(zip(cuts_list, bcuts.records, strict=True)):
        assert cut["cut_id"] == record.cut_id, (
            f"pool position {k}: reloaded cut_id={cut['cut_id']!r} != "
            f"source cut_id={record.cut_id!r} — cuts[k] <-> bcuts.records[k] "
            "pairing is broken"
        )

    return _RoundTripCase(
        case_dir=case_dir, reloaded=reloaded, bcuts=bcuts, id_map=id_map
    )


@_skip_e2e
def test_reloaded_boundary_checkpoint_shape(roundtrip_case: _RoundTripCase) -> None:
    """AC 1 — the reloaded checkpoint has exactly one ``stage_cuts`` entry at
    ``stage_id == 10``, a non-empty ``cuts`` list, every coefficient vector
    matching ``state_dimension``, and ``cost_scale_factor == 1.0`` (the
    legacy 10⁶-scale marker is absent).
    """
    reloaded = roundtrip_case.reloaded
    stage_cuts = reloaded["stage_cuts"]
    assert len(stage_cuts) == 1

    entry = stage_cuts[0]
    assert entry["stage_id"] == 10
    assert entry["cuts"]

    state_dimension = entry["state_dimension"]
    for cut in entry["cuts"]:
        assert len(cut["coefficients"]) == state_dimension

    assert reloaded["metadata"]["producer"]["cost_scale_factor"] == 1.0


@_skip_e2e
def test_storage_coefficient_identity_and_unmapped_slots_zero(
    roundtrip_case: _RoundTripCase,
) -> None:
    """AC 2 — for every active cut and every resolved plant, the authored
    ``HydroStorage`` coefficient equals the source ``pi_varm``, and every
    manifest slot carrying no mapped source term (dropped-plant, transit,
    anticipated, or an unpopulated extra lag depth) is exactly ``0.0`` —
    proving sign, unit, and the 1.0 scale marker.
    """
    entry = roundtrip_case.reloaded["stage_cuts"][0]
    cuts_list = entry["cuts"]
    manifest = entry["entity_manifest"]
    plant_codes = roundtrip_case.bcuts.header.plant_codes

    resolved, dropped = _resolve(plant_codes, roundtrip_case.id_map, manifest)
    dropped_codes = {plant_codes[i] for i in dropped}
    assert dropped_codes == {132, 176}, dropped_codes
    assert len(resolved) == 153

    mapped_positions: set[int] = set()
    for _hydro_id, storage_pos, lag_positions in resolved.values():
        mapped_positions.add(storage_pos)
        mapped_positions.update(lag_positions.values())

    active_pairs = [
        (cut, record)
        for cut, record in zip(cuts_list, roundtrip_case.bcuts.records, strict=True)
        if cut["is_active"]
    ]
    assert active_pairs

    for cut, record in active_pairs:
        coefficients = cut["coefficients"]
        for plant_index, (_hydro_id, storage_position, _lags) in resolved.items():
            authored = coefficients[storage_position]
            source = record.pi_varm[plant_index]
            assert math.isclose(authored, source, rel_tol=1e-9, abs_tol=1e-6), (
                f"cut_id={cut['cut_id']} plant_index={plant_index} storage "
                f"slot {storage_position}: authored={authored!r} "
                f"source pi_varm={source!r}"
            )
        for position, value in enumerate(coefficients):
            if position not in mapped_positions:
                assert value == 0.0, (
                    f"cut_id={cut['cut_id']} unmapped slot {position} = "
                    f"{value!r}, expected exactly 0.0"
                )


@_skip_e2e
def test_lag_coefficient_identity(roundtrip_case: _RoundTripCase) -> None:
    """AC 3 — for every active cut, every resolved plant, and every lag
    depth 1..12 whose ``(HydroInflowLag, hydro_id, depth-1)`` slot exists,
    the authored coefficient equals the source ``pi_qafl`` — proving lag
    alignment (no off-by-one).
    """
    entry = roundtrip_case.reloaded["stage_cuts"][0]
    cuts_list = entry["cuts"]
    manifest = entry["entity_manifest"]

    resolved, _dropped = _resolve(
        roundtrip_case.bcuts.header.plant_codes, roundtrip_case.id_map, manifest
    )

    active_pairs = [
        (cut, record)
        for cut, record in zip(cuts_list, roundtrip_case.bcuts.records, strict=True)
        if cut["is_active"]
    ]
    assert active_pairs

    checked = 0
    for cut, record in active_pairs:
        coefficients = cut["coefficients"]
        for plant_index, (
            _hydro_id,
            _storage_position,
            lag_positions,
        ) in resolved.items():
            plant_lags = record.pi_qafl[plant_index]
            for depth, lag_position in lag_positions.items():
                checked += 1
                authored = coefficients[lag_position]
                source = plant_lags[depth - 1]
                assert math.isclose(authored, source, rel_tol=1e-9, abs_tol=1e-6), (
                    f"cut_id={cut['cut_id']} plant_index={plant_index} lag "
                    f"depth={depth} slot {lag_position}: authored="
                    f"{authored!r} source pi_qafl={source!r}"
                )
    assert checked > 0, "no HydroInflowLag slots were present to check"


@_skip_e2e
def test_evaluation_sweep_theta_identity(roundtrip_case: _RoundTripCase) -> None:
    """AC 4 — ``theta_cobre(x) == theta_source(x)`` over a deterministic
    ``rng(0)`` state sweep (all-zero, per-plant unit-storage, and several
    full random states), plus the optional ``cobre.Policy.evaluate``
    cross-check (skipped, not failed, if ``Study``/``Policy`` load raises).
    """
    import cobre

    entry = roundtrip_case.reloaded["stage_cuts"][0]
    cuts_list = entry["cuts"]
    manifest = entry["entity_manifest"]
    state_dimension = entry["state_dimension"]
    stage_id = entry["stage_id"]

    resolved, dropped = _resolve(
        roundtrip_case.bcuts.header.plant_codes, roundtrip_case.id_map, manifest
    )
    assert dropped

    rng = np.random.default_rng(0)
    states: list[_PhysicalState] = [
        _PhysicalState(storage={}, inflow={}),
        _PhysicalState(storage={i: 1.0 for i in resolved}, inflow={}),
    ]
    for _ in range(_N_RANDOM_STATES):
        storage = {i: float(rng.normal(scale=1000.0)) for i in resolved}
        inflow = {
            (i, depth): float(rng.normal(scale=1000.0))
            for i, (_hydro_id, _storage_position, lag_positions) in resolved.items()
            for depth in lag_positions
        }
        states.append(_PhysicalState(storage=storage, inflow=inflow))
    assert len(states) >= 8

    projected: list[tuple[list[float], float]] = []
    for index, state in enumerate(states):
        x_cobre = [0.0] * state_dimension
        for plant_index, (
            _hydro_id,
            storage_position,
            lag_positions,
        ) in resolved.items():
            x_cobre[storage_position] = state.storage.get(plant_index, 0.0)
            for depth, lag_position in lag_positions.items():
                x_cobre[lag_position] = state.inflow.get((plant_index, depth), 0.0)

        theta_cobre = _theta_cobre(cuts_list, x_cobre)
        theta_source = _theta_source(roundtrip_case.bcuts.records, resolved, state)
        assert math.isclose(theta_cobre, theta_source, rel_tol=1e-9, abs_tol=1e-6), (
            f"state {index}: theta_cobre={theta_cobre!r} != "
            f"theta_source={theta_source!r}"
        )
        projected.append((x_cobre, theta_cobre))

    try:
        policy = cobre.Study(roundtrip_case.case_dir).load_policy()
    except (RuntimeError, ValueError, TypeError) as exc:
        pytest.skip(f"Policy.evaluate cross-check unavailable: {exc}")

    for index, (x_cobre, theta_cobre) in enumerate(projected):
        theta_policy = policy.evaluate(stage_id, x_cobre)
        assert math.isclose(theta_policy, theta_cobre, rel_tol=1e-9, abs_tol=1e-6), (
            f"state {index}: Policy.evaluate={theta_policy!r} != "
            f"theta_cobre={theta_cobre!r}"
        )
