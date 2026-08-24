"""Tier-1 tests for the bucket-A -> cobre token map (TICKET-011) and the
E1-E7 ``GenericConstraintBuilder`` emit pipeline (TICKET-012).

Synthetic-fixture only, mirroring ``tests/test_decomp_libs_electrical.py``'s
own convention: no ``example/`` read, no ``import cobre`` at module scope.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.libs_electrical import (
    AssembledBound,
    AvailablePower,
    DataContext,
    ElectricalRestriction,
    LibsElectricalModel,
    ParsedTerm,
    UnrecognizedElectricalToken,
    ViolationTreatment,
    _UnresolvableBucketBTerm,
)
from cobre_bridge.decomp.libs_electrical_emit import (
    LibsElectricalResult,
    _cobre_token,
    _resolve_interc_bus,
    build_electrical_expression,
    emit_libs_electrical_generics,
)
from cobre_bridge.decomp.ncs import _pee_series, build_pee_ncs_id_map
from cobre_bridge.decomp.temporal import OperativeStage
from cobre_bridge.generic_constraint_builder import ConstraintIdAllocator
from tests.conftest import make_decomp_case

# ---------------------------------------------------------------------------
# _cobre_token — one bucket-A token kind at a time (spec §2)
# ---------------------------------------------------------------------------


def test_cobre_token_ger_usih_resolved() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    term = ParsedTerm(coefficient=1.0, token="ger_usih", args=(66,))
    assert _cobre_token(term, id_map, {}, {}, {}) == "hydro_generation(0)"


def test_cobre_token_ger_usit_resolved() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), thermal_codes=(5,))
    term = ParsedTerm(coefficient=1.0, token="ger_usit", args=(5,))
    assert _cobre_token(term, id_map, {}, {}, {}) == "thermal_generation(0)"


def test_cobre_token_ger_pee_resolved_via_map_not_code() -> None:
    # Pitfall guard: codigo_pee (11) is NOT the cobre ncs id -- the token
    # must go through ncs_id_by_pee_code, never straight to args[0].
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    term = ParsedTerm(coefficient=1.0, token="ger_pee", args=(11,))
    result = _cobre_token(term, id_map, {11: 4}, {}, {})
    assert result == "non_controllable_generation(4)"


def test_cobre_token_ger_conjh_frequency_split_bus() -> None:
    # Itaipu (66,1) -> IV bus, (66,2) -> SE bus (spec §4e), both hitting the
    # same underlying hydro id.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    conjh_bus_by_code_group = {(66, 1): 3, (66, 2): 0}
    iv_term = ParsedTerm(coefficient=1.0, token="ger_conjh", args=(66, 1))
    se_term = ParsedTerm(coefficient=1.0, token="ger_conjh", args=(66, 2))
    assert (
        _cobre_token(iv_term, id_map, {}, conjh_bus_by_code_group, {})
        == "hydro_generation(0, bus=3)"
    )
    assert (
        _cobre_token(se_term, id_map, {}, conjh_bus_by_code_group, {})
        == "hydro_generation(0, bus=0)"
    )


def test_cobre_token_ener_interc_direct_orientation() -> None:
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    line_map = {(0, 1): 2}
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(1, 2))
    assert _cobre_token(term, id_map, {}, {}, line_map) == "line_direct(2)"


def test_cobre_token_ener_interc_reverse_orientation() -> None:
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    line_map = {(1, 0): 5}
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(1, 2))
    assert _cobre_token(term, id_map, {}, {}, line_map) == "line_reverse(5)"


def test_cobre_token_ener_interc_no_line_warns_and_returns_none() -> None:
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(1, 2))
    with dx.collect() as sink:
        result = _cobre_token(term, id_map, {}, {}, {})
    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING


# ---------------------------------------------------------------------------
# _resolve_interc_bus + ener_interc's transshipment (IV) fallback (TICKET-018)
# ---------------------------------------------------------------------------


def test_resolve_interc_bus_declared_code_returns_bus_id() -> None:
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    assert _resolve_interc_bus(2, id_map) == 1


def test_resolve_interc_bus_unknown_code_returns_transhipment_bus_id() -> None:
    # No hardcoded code value: ANY code absent from bus_codes falls back to
    # the transshipment bus.
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    assert id_map.transhipment_bus_id == 2
    assert _resolve_interc_bus(6, id_map) == 2
    assert _resolve_interc_bus(-1, id_map) == 2


def test_cobre_token_ener_interc_transshipment_operand_resolves_direct() -> None:
    # AC1: code 6 is NOT a declared SB code; with line_map holding the
    # transshipment<->SE line, _cobre_token resolves it (not a drop).
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    line_map = {(id_map.transhipment_bus_id, 0): 7}
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(6, 1))
    assert _cobre_token(term, id_map, {}, {}, line_map) == "line_direct(7)"


def test_cobre_token_ener_interc_transshipment_operand_resolves_reverse() -> None:
    # AC2: the reverse-argument-order call resolves the SAME line, with the
    # opposite orientation token.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    line_map = {(id_map.transhipment_bus_id, 0): 7}
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(1, 6))
    assert _cobre_token(term, id_map, {}, {}, line_map) == "line_reverse(7)"


def test_cobre_token_ener_interc_declared_codes_fallback_not_taken() -> None:
    # AC3: both operands are declared SB codes -- the transshipment fallback
    # must never be consulted, even with a decoy line keyed at the
    # transshipment bus that would resolve to a DIFFERENT line id if the
    # fallback wrongly fired for a declared code.
    id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"))
    line_map = {(0, 1): 9, (id_map.transhipment_bus_id, 1): 99}
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(1, 2))
    assert _cobre_token(term, id_map, {}, {}, line_map) == "line_direct(9)"


def test_cobre_token_ener_interc_unknown_code_no_transshipment_line_drops() -> None:
    # AC4: a non-SB code resolves to the transshipment bus, but line_map has
    # no line for that pair -- the fallback never fabricates a line; the
    # existing skip-not-partial drop still applies.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    term = ParsedTerm(coefficient=1.0, token="ener_interc", args=(6, 1))
    with dx.collect() as sink:
        result = _cobre_token(term, id_map, {}, {}, {})
    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING


def test_cobre_token_ener_comerc_deferred_warns_and_returns_none() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    term = ParsedTerm(coefficient=1.0, token="ener_comerc", args=(1,))
    with dx.collect() as sink:
        result = _cobre_token(term, id_map, {}, {}, {})
    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING
    assert "ener_comerc" in sink[0].summary


def test_cobre_token_ger_usih_unresolved_warns_and_returns_none() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    term = ParsedTerm(coefficient=1.0, token="ger_usih", args=(999,))
    with dx.collect() as sink:
        result = _cobre_token(term, id_map, {}, {}, {})
    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING
    assert "ger_usih" in sink[0].summary
    assert "999" in sink[0].summary


def test_cobre_token_unrecognized_bucket_a_token_raises_value_error() -> None:
    # A caller-contract violation: AssembledBound.terms should never carry
    # anything but a bucket-A token.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    term = ParsedTerm(coefficient=1.0, token="disp_usih", args=(1,))
    with pytest.raises(ValueError, match="disp_usih"):
        _cobre_token(term, id_map, {}, {}, {})


# ---------------------------------------------------------------------------
# build_electrical_expression (spec §4)
# ---------------------------------------------------------------------------


def test_build_electrical_expression_all_token_kinds_exact_string() -> None:
    id_map = DecompIdMap(
        bus_codes=(1, 2, 3, 4),
        bus_names=("SE", "S", "NE", "N"),
        hydro_codes=(66,),
        thermal_codes=(5,),
    )
    ncs_id_by_pee_code = {11: 0}
    conjh_bus_by_code_group = {(66, 1): 3, (66, 2): 0}
    line_map = {(0, 1): 2}
    assembled = AssembledBound(
        terms=(
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(66,)),
            ParsedTerm(coefficient=1.0, token="ger_usit", args=(5,)),
            ParsedTerm(coefficient=1.0, token="ger_pee", args=(11,)),
            ParsedTerm(coefficient=1.0, token="ger_conjh", args=(66, 1)),
            ParsedTerm(coefficient=1.0, token="ger_conjh", args=(66, 2)),
            ParsedTerm(coefficient=1.0, token="ener_interc", args=(1, 2)),
        ),
        lower=None,
        upper=None,
    )
    result = build_electrical_expression(
        assembled, id_map, ncs_id_by_pee_code, conjh_bus_by_code_group, line_map
    )
    assert result == (
        "hydro_generation(0) + thermal_generation(0) + "
        "non_controllable_generation(0) + hydro_generation(0, bus=3) + "
        "hydro_generation(0, bus=0) + line_direct(2)"
    )


def test_build_electrical_expression_unresolved_token_drops_whole_expression() -> None:
    # AC5: ger_usih(999) is absent from id_map.hydro_codes -- the WHOLE
    # expression drops (None), not a partial one built from ger_usih(66)
    # alone (skip-not-partial).
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(66,))
    assembled = AssembledBound(
        terms=(
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(66,)),
            ParsedTerm(coefficient=1.0, token="ger_usih", args=(999,)),
        ),
        lower=None,
        upper=100.0,
    )
    with dx.collect() as sink:
        result = build_electrical_expression(assembled, id_map, {}, {}, {})
    assert result is None
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING
    assert "ger_usih" in sink[0].summary
    assert "999" in sink[0].summary


# ---------------------------------------------------------------------------
# ncs.build_pee_ncs_id_map (shared ordering with _pee_series)
# ---------------------------------------------------------------------------


def _emit_calendar(n_stages: int, *, n_blocks: int) -> list[OperativeStage]:
    return [
        OperativeStage(
            index=i,
            start_date=date(2026, 7, 4),
            end_date=date(2026, 7, 11),
            season_id=6,
            block_hours=tuple(24.0 for _ in range(n_blocks)),
        )
        for i in range(n_stages)
    ]


class _StubDadger:
    """Return a preset ``pq`` DataFrame (or ``None``) — mirrors ``_StubDadger``
    in ``tests/test_decomp_ncs.py``, duplicated here since that file is
    outside this ticket's declared scope."""

    def __init__(self, pq: pd.DataFrame | None) -> None:
        self._pq = pq

    def pq(self, df: bool = False) -> pd.DataFrame | None:
        return self._pq


class _StubRenovaveis:
    def __init__(
        self, cad: pd.DataFrame, subm: pd.DataFrame, ger: pd.DataFrame
    ) -> None:
        self._cad, self._subm, self._ger = cad, subm, ger

    def pee_cad(self, df: bool = False) -> pd.DataFrame:
        return self._cad

    def pee_subm(self, df: bool = False) -> pd.DataFrame:
        return self._subm

    def pee_ger_per_pat_cen(self, df: bool = False) -> pd.DataFrame:
        return self._ger


def _pq_frame(n_series: int, *, n_blocks: int) -> pd.DataFrame:
    rows = []
    for i in range(n_series):
        row: dict[str, object] = {
            "codigo_submercado": 1,
            "estagio": 1,
            "nome": f"S{i}",
        }
        for k in range(1, n_blocks + 1):
            row[f"geracao_{k}"] = 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_pee_ncs_id_map_sorted_offset_matches_pee_series() -> None:
    # AC6: parks {7, 3} declared after N PQ series -> {3: N, 7: N+1},
    # sorted-by-code, offset past the PQ series, matching _pee_series' ids.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _emit_calendar(1, n_blocks=1)
    dadger = _StubDadger(_pq_frame(n_series=2, n_blocks=1))
    cad = pd.DataFrame(
        [
            {"codigo_pee": 7, "nome_pee": "PARK7"},
            {"codigo_pee": 3, "nome_pee": "PARK3"},
        ]
    )
    subm = pd.DataFrame(
        [
            {"codigo_pee": 7, "codigo_submercado": 1},
            {"codigo_pee": 3, "codigo_submercado": 1},
        ]
    )
    ger = pd.DataFrame(
        [
            {
                "codigo_pee": code,
                "estagio": 1,
                "patamar": 1,
                "cenario": 1,
                "geracao": 5.0,
            }
            for code in (7, 3)
        ]
    )
    renovaveis = _StubRenovaveis(cad, subm, ger)

    case = make_decomp_case(
        Path("unused"), dadger=dadger, calendar=calendar, renovaveis=renovaveis
    )
    ncs_map = build_pee_ncs_id_map(case, id_map)
    assert ncs_map == {3: 2, 7: 3}

    series = _pee_series(renovaveis, id_map, calendar, first_ncs_id=2)
    assert [s.ncs_id for s in series] == [ncs_map[3], ncs_map[7]]
    assert series[0].name.startswith("PARK3")
    assert series[1].name.startswith("PARK7")


def test_build_pee_ncs_id_map_none_renovaveis_returns_empty() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _emit_calendar(1, n_blocks=1)
    dadger = _StubDadger(None)
    case = make_decomp_case(
        Path("unused"), dadger=dadger, calendar=calendar, renovaveis=None
    )
    assert build_pee_ncs_id_map(case, id_map) == {}


def test_build_pee_ncs_id_map_no_declared_parks_returns_empty() -> None:
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _emit_calendar(1, n_blocks=1)
    dadger = _StubDadger(None)
    renovaveis = _StubRenovaveis(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    case = make_decomp_case(
        Path("unused"), dadger=dadger, calendar=calendar, renovaveis=renovaveis
    )
    assert build_pee_ncs_id_map(case, id_map) == {}


# ---------------------------------------------------------------------------
# emit_libs_electrical_generics -- the E1-E7 _GenericBuilder emit pipeline
# (TICKET-012, spec §5)
# ---------------------------------------------------------------------------


def _dict_context(values: Mapping[str, float]) -> DataContext:
    """A trivial dict-backed ``DataContext`` test double (duplicated from
    ``tests/test_decomp_libs_electrical.py`` -- outside this ticket's
    declared scope)."""

    def resolve(term: ParsedTerm) -> float:
        if term.token == "__const__":
            return 1.0
        if term.token == "alias":
            assert term.alias_name is not None
            return values[term.alias_name]
        key = (
            f"{term.token}({','.join(str(arg) for arg in term.args)})"
            if term.args
            else term.token
        )
        return values[key]

    return resolve


def _demanda_sin_context_factory(
    values: Mapping[tuple[int, int], float],
) -> Callable[[int, int], DataContext]:
    """A per-cell ``DataContext`` factory over ``demanda_sin`` values keyed
    by ``(stage_index, block_index)``; an absent cell defaults to ``0.0``."""

    def factory(stage_index: int, block_index: int) -> DataContext:
        return _dict_context(
            {"demanda_sin": values.get((stage_index, block_index), 0.0)}
        )

    return factory


def _restriction(
    code: int,
    lhs: str,
    operator: str,
    rhs: str,
    *,
    stage_start: int | None = None,
    stage_end: int | None = None,
    habilita: int | None = None,
    violation: ViolationTreatment | None = None,
) -> ElectricalRestriction:
    """A minimal INEQUACAO restriction, covering every field
    ``emit_libs_electrical_generics`` reads."""
    return ElectricalRestriction(
        code=code,
        lhs=lhs,
        operator=operator,
        rhs=rhs,
        is_formula=False,
        limits={},
        overrides=(),
        stage_start=stage_start,
        stage_end=stage_end,
        habilita=habilita,
        violation=violation,
    )


def _reserve_restriction(
    code: int, plant_code: int, threshold: float
) -> ElectricalRestriction:
    """The documented ``disp_usih(h) - ger_usih(h) >= R`` reserve pattern
    (AC1/AC2's A1/A3 sign proof) for one plant, active across the whole
    (single-stage) calendar -- no ``habilita``, no horizon override."""
    return _restriction(
        code,
        f"disp_usih({plant_code}) - ger_usih({plant_code})",
        ">=",
        str(threshold),
    )


def test_libs_electrical_result_is_frozen_with_expected_fields() -> None:
    # Requirement 1: a frozen dataclass carrying the surviving
    # GenericConstraintResult plus the converted/deferred census.
    result = LibsElectricalResult(
        generic=None, converted_codes=(1, 2), deferred={"inactive": (3,)}
    )
    assert result.generic is None
    assert result.converted_codes == (1, 2)
    assert result.deferred == {"inactive": (3,)}
    with pytest.raises(AttributeError):
        result.converted_codes = (4,)  # type: ignore[misc]


def test_emit_reserve_a1_end_to_end_generation_cap() -> None:
    # AC1: disp_usih(1) - ger_usih(1) >= 300, A_h(1) = 1000 -> ger_usih(1)
    # <= 700, a POSITIVE bare generation-cap token -- never an inverted
    # lower bound (the A1 sign fix, proven end to end through the emitter).
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _reserve_restriction(407, plant_code=1, threshold=300.0)
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={407: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={(1, 0): 1000.0}, rated_envelope={})

    result = emit_libs_electrical_generics(
        model,
        id_map,
        lambda s, b: _dict_context({}),
        a_h,
        calendar,
        {},
        {},
        {},
        big_m=999.0,
    )

    assert result.converted_codes == (407,)
    assert result.deferred == {
        "inactive": (),
        "unresolved-bucket-bc": (),
        "unresolved-bucket-a": (),
        "unrecognized-token": (),
    }
    assert result.generic is not None
    [constraint] = result.generic.constraints
    assert constraint["name"] == "LIBS_ELEC_407"
    assert constraint["description"] == "LIBs electrical special constraint 407"
    assert constraint["expression"] == "hydro_generation(0)"
    rows = result.generic.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_upper"] == 700.0
    assert rows[0]["bound_lower"] is None


def test_emit_reserve_a3_negative_cap_stays_feasible_via_slack() -> None:
    # AC2: A_h(1) = 200 < R = 300 -> ger_usih(1) <= -100, a negative
    # generation cap that must stay feasible via slack (A3), never a hard
    # LP infeasibility.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _reserve_restriction(408, plant_code=1, threshold=300.0)
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={408: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={(1, 0): 200.0}, rated_envelope={})

    result = emit_libs_electrical_generics(
        model,
        id_map,
        lambda s, b: _dict_context({}),
        a_h,
        calendar,
        {},
        {},
        {},
        big_m=999.0,
    )

    assert result.generic is not None
    [constraint] = result.generic.constraints
    assert constraint["slack"] == {"enabled": True, "penalty": 999.0}
    rows = result.generic.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["bound_upper"] == -100.0


def test_emit_active_only_stage0_block0_single_bounds_row() -> None:
    # AC3: a restriction active only in stage 0 block 0 of a 2-block stage
    # emits exactly one bounds row for that cell -- activation via row
    # presence -- and none for block 1.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=2)
    restriction = _restriction(701, "ger_usih(1)", "<=", "500", habilita=10)
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={701: restriction},
        rules={10: "demanda_sin >= 50000"},
    )
    factory = _demanda_sin_context_factory({(0, 0): 60000.0})
    a_h = AvailablePower(overlay={}, rated_envelope={})

    result = emit_libs_electrical_generics(
        model, id_map, factory, a_h, calendar, {}, {}, {}, big_m=999.0
    )

    assert result.generic is not None
    rows = result.generic.bounds.to_pylist()
    assert len(rows) == 1
    assert rows[0]["stage_id"] == 0
    assert rows[0]["block_id"] == 0
    assert rows[0]["bound_upper"] == 500.0


def test_emit_violation_penalty_overrides_big_m() -> None:
    # AC4: an explicit TRATAMENTO-VIOLACAO penalty overrides the BIG-M
    # default.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(
        702,
        "ger_usih(1)",
        "<=",
        "500",
        violation=ViolationTreatment(enabled=True, penalty=500.0),
    )
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={702: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    result = emit_libs_electrical_generics(
        model,
        id_map,
        lambda s, b: _dict_context({}),
        a_h,
        calendar,
        {},
        {},
        {},
        big_m=999.0,
    )

    assert result.generic is not None
    [constraint] = result.generic.constraints
    assert constraint["slack"]["penalty"] == 500.0


def test_emit_inactive_everywhere_defers_without_duplicate_diagnostic() -> None:
    # AC5: active_cells already emits the one INFO for a restriction inactive
    # in every cell; the emitter must not add a second one.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(703, "ger_usih(1)", "<=", "500", habilita=10)
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={703: restriction},
        rules={10: "demanda_sin >= 50000"},
    )
    factory = _demanda_sin_context_factory({})  # every cell defaults to 0.0
    a_h = AvailablePower(overlay={}, rated_envelope={})

    with dx.collect() as sink:
        result = emit_libs_electrical_generics(
            model, id_map, factory, a_h, calendar, {}, {}, {}, big_m=999.0
        )

    assert result.generic is None
    assert result.deferred["inactive"] == (703,)
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.INFO


def test_emit_unresolved_bucket_a_token_defers_without_partial_constraint() -> None:
    # AC6: ger_usih(999) is absent from id_map -- the whole restriction is
    # dropped (skip-not-partial), never emitted with a partial expression.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(704, "ger_usih(999)", "<=", "500")
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={704: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    with dx.collect() as sink:
        result = emit_libs_electrical_generics(
            model,
            id_map,
            lambda s, b: _dict_context({}),
            a_h,
            calendar,
            {},
            {},
            {},
            big_m=999.0,
        )

    assert result.generic is None
    assert result.deferred["unresolved-bucket-a"] == (704,)
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING


def test_emit_unresolved_bucket_bc_term_defers_whole_restriction() -> None:
    # Requirement 2's second bullet: a single unresolvable active cell
    # (assemble_bound already WARNED) drops the WHOLE restriction -- no
    # partial constraint, and no duplicate diagnostic from this emitter.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(705, "carga_ande", "<=", "500")
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={705: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    def _unresolvable_carga_ande(term: ParsedTerm) -> float:
        if term.token == "carga_ande":
            raise _UnresolvableBucketBTerm("carga_ande has no value")
        return 1.0

    with dx.collect() as sink:
        result = emit_libs_electrical_generics(
            model,
            id_map,
            lambda s, b: _unresolvable_carga_ande,
            a_h,
            calendar,
            {},
            {},
            {},
            big_m=999.0,
        )

    assert result.generic is None
    assert result.deferred["unresolved-bucket-bc"] == (705,)
    assert len(sink) == 1
    assert sink[0].severity is dx.Severity.WARNING


def test_emit_cell_inconsistent_terms_raises_value_error_naming_restriction() -> None:
    # Error Handling: a resolver bug producing different AssembledBound.terms
    # across a restriction's active cells must fail loud, never silently
    # pick one cell's expression.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))
    calendar = _emit_calendar(1, n_blocks=2)
    restriction = _restriction(706, "ger_usih(1)", "<=", "500")
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={706: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    def _stub_assemble_bound(
        restriction: ElectricalRestriction,
        model: LibsElectricalModel,
        stage_index: int,
        block_index: int,
        ctx: DataContext,
        a_h: AvailablePower | None = None,
    ) -> AssembledBound:
        code = 1 if block_index == 0 else 2
        return AssembledBound(
            terms=(ParsedTerm(coefficient=1.0, token="ger_usih", args=(code,)),),
            lower=None,
            upper=100.0,
        )

    with patch(
        "cobre_bridge.decomp.libs_electrical_emit.assemble_bound",
        side_effect=_stub_assemble_bound,
    ):
        with pytest.raises(ValueError, match="706"):
            emit_libs_electrical_generics(
                model,
                id_map,
                lambda s, b: _dict_context({}),
                a_h,
                calendar,
                {},
                {},
                {},
                big_m=999.0,
            )


def test_emit_sorted_restriction_codes_deterministic_ids() -> None:
    # Requirement 2: iterates sorted(model.restrictions), so ids are
    # assigned in ascending restriction-code order regardless of dict
    # insertion order.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restrictions = {
        20: _restriction(20, "ger_usih(1)", "<=", "500"),
        10: _restriction(10, "ger_usih(1)", "<=", "600"),
    }
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions=restrictions, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    result = emit_libs_electrical_generics(
        model,
        id_map,
        lambda s, b: _dict_context({}),
        a_h,
        calendar,
        {},
        {},
        {},
        big_m=999.0,
        allocator=ConstraintIdAllocator(5),
    )

    assert result.converted_codes == (10, 20)
    assert result.generic is not None
    names = [c["name"] for c in result.generic.constraints]
    assert names == ["LIBS_ELEC_10", "LIBS_ELEC_20"]
    ids = [c["id"] for c in result.generic.constraints]
    assert ids == [5, 6]


# ---------------------------------------------------------------------------
# unrecognized-token skip-not-partial containment (TICKET-015)
# ---------------------------------------------------------------------------


def test_emit_unrecognized_token_defers_one_restriction_sibling_converts() -> None:
    # AC3 (ticket-015): a well-formed but undeclared identifier
    # (peq_N_PCHgd_N -- the MMGD family, deliberately unmodeled) in one
    # restriction's formula drops only that restriction; a fully resolvable
    # sibling still converts.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    unrecognized = _restriction(709, "peq_N_PCHgd_N", "<=", "500")
    resolvable = _restriction(710, "ger_usih(1)", "<=", "500")
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={709: unrecognized, 710: resolvable},
        rules={},
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    with dx.collect() as sink:
        result = emit_libs_electrical_generics(
            model,
            id_map,
            lambda s, b: _dict_context({}),
            a_h,
            calendar,
            {},
            {},
            {},
            big_m=999.0,
        )

    assert result.converted_codes == (710,)
    assert result.deferred["unrecognized-token"] == (709,)
    assert result.generic is not None
    names = [c["name"] for c in result.generic.constraints]
    assert names == ["LIBS_ELEC_710"]

    warnings = [d for d in sink if d.severity is dx.Severity.WARNING]
    assert len(warnings) == 1
    assert "709" in warnings[0].summary
    assert "peq_N_PCHgd_N" in warnings[0].summary


def test_emit_no_unrecognized_tokens_leaves_deferred_reason_empty() -> None:
    # AC4 (ticket-015): a model with no unrecognized tokens leaves
    # deferred["unrecognized-token"] an empty tuple and drops nothing
    # spuriously.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(711, "ger_usih(1)", "<=", "500")
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={711: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    result = emit_libs_electrical_generics(
        model,
        id_map,
        lambda s, b: _dict_context({}),
        a_h,
        calendar,
        {},
        {},
        {},
        big_m=999.0,
    )

    assert result.converted_codes == (711,)
    assert result.deferred["unrecognized-token"] == ()


def test_emit_propagates_plain_value_error_not_unrecognized_token() -> None:
    # Pitfall guard (ticket-015): the emitter catches ONLY
    # UnrecognizedElectricalToken -- a plain ValueError from a resolver bug
    # must still fail loud, never be masked as a routine skip-not-partial
    # drop.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(712, "ger_usih(1)", "<=", "500")
    model = LibsElectricalModel(
        expressions={}, aliases={}, restrictions={712: restriction}, rules={}
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    def _stub_assemble_bound(
        restriction: ElectricalRestriction,
        model: LibsElectricalModel,
        stage_index: int,
        block_index: int,
        ctx: DataContext,
        a_h: AvailablePower | None = None,
    ) -> AssembledBound:
        raise ValueError("a genuine resolver bug, not an unrecognized token")

    with patch(
        "cobre_bridge.decomp.libs_electrical_emit.assemble_bound",
        side_effect=_stub_assemble_bound,
    ):
        with pytest.raises(ValueError, match="resolver bug") as exc_info:
            emit_libs_electrical_generics(
                model,
                id_map,
                lambda s, b: _dict_context({}),
                a_h,
                calendar,
                {},
                {},
                {},
                big_m=999.0,
            )

    assert not isinstance(exc_info.value, UnrecognizedElectricalToken)


# ---------------------------------------------------------------------------
# unrecognized-token containment widened to active_cells (TICKET-017)
# ---------------------------------------------------------------------------


def test_emit_unrecognized_token_in_activation_rule_defers_one_restriction() -> None:
    # AC1 (ticket-017): an undeclared identifier in a restriction's
    # ACTIVATION RULE -- not its bound/expression -- must be contained the
    # same way: active_cells' own UnrecognizedElectricalToken now sits
    # inside the per-restriction guard, so it drops only that restriction; a
    # fully resolvable sibling (no habilita) still converts.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    rule_broken = _restriction(720, "ger_usih(1)", "<=", "500", habilita=20)
    resolvable = _restriction(721, "ger_usih(1)", "<=", "500")
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={720: rule_broken, 721: resolvable},
        rules={20: "peq_N_PCHgd_N >= 50000"},
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    with dx.collect() as sink:
        result = emit_libs_electrical_generics(
            model,
            id_map,
            lambda s, b: _dict_context({}),
            a_h,
            calendar,
            {},
            {},
            {},
            big_m=999.0,
        )

    assert result.converted_codes == (721,)
    assert result.deferred["unrecognized-token"] == (720,)
    assert result.generic is not None
    names = [c["name"] for c in result.generic.constraints]
    assert names == ["LIBS_ELEC_721"]

    warnings = [d for d in sink if d.severity is dx.Severity.WARNING]
    assert len(warnings) == 1
    assert "720" in warnings[0].summary
    assert "peq_N_PCHgd_N" in warnings[0].summary


def test_emit_malformed_activation_rule_propagates_value_error() -> None:
    # AC2 (ticket-017): a malformed-DSL activation rule (unbalanced parens,
    # not an undeclared identifier) must still fail loud, never be swallowed
    # by the widened UnrecognizedElectricalToken-specific guard.
    id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))
    calendar = _emit_calendar(1, n_blocks=1)
    restriction = _restriction(722, "ger_usih(1)", "<=", "500", habilita=30)
    model = LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={722: restriction},
        rules={30: "demanda(3 >= 5"},
    )
    a_h = AvailablePower(overlay={}, rated_envelope={})

    with pytest.raises(ValueError, match="unbalanced parentheses") as exc_info:
        emit_libs_electrical_generics(
            model,
            id_map,
            lambda s, b: _dict_context({}),
            a_h,
            calendar,
            {},
            {},
            {},
            big_m=999.0,
        )

    assert not isinstance(exc_info.value, UnrecognizedElectricalToken)


# ---------------------------------------------------------------------------
# Tier 3: the real deck + the local cobre binary, guarded exactly like
# tests/test_decomp_fcf_roundtrip.py.
# ---------------------------------------------------------------------------

_DECK = Path("example/decomp-abr-26-lpp")
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"
_HAS_E2E_DEPS = _COBRE_BIN.exists() and (_DECK / "caso.dat").exists()
_skip_e2e = pytest.mark.skipif(
    not _HAS_E2E_DEPS,
    reason=f"requires the local cobre binary ({_COBRE_BIN}) and the {_DECK} deck",
)

#: Pre-ticket-018 ``unresolved-bucket-a`` count on this deck -- measured by
#: stashing this ticket's ``_resolve_interc_bus`` fix (``git stash push --
#: src/cobre_bridge/decomp/libs_electrical_emit.py``) and reconverting: 20
#: (of which 10 are the IV-transshipment ``ener_interc`` restrictions this
#: ticket now resolves). Pinned here so the test proves the fix's DIRECTION
#: on the real deck without re-running the pre-fix code path inside the test
#: itself.
_PRE_TICKET_018_UNRESOLVED_BUCKET_A = 20


@_skip_e2e
def test_abr_26_lpp_ener_interc_transshipment_resolves_and_validates(
    tmp_path: Path,
) -> None:
    """AC5: converting the real deck now resolves the IV-transshipment
    ``ener_interc`` operands via ``_resolve_interc_bus``, so
    ``deferred["unresolved-bucket-a"]`` strictly decreases versus the
    pre-ticket-018 baseline, and ``cobre validate`` on the converted case
    still exits 0."""
    from cobre_bridge.decomp.pipeline import convert_decomp_case

    dst = tmp_path / "decomp-abr-26-lpp-converted"
    report = convert_decomp_case(_DECK, dst, force=True)

    census = [
        d for d in report.diagnostics if d.code == "decomp-libs-electrical-converted"
    ]
    assert len(census) == 1
    [diagnostic] = census
    assert diagnostic.table is not None
    counts = dict(diagnostic.table.rows)
    assert counts["unresolved-bucket-a"] < _PRE_TICKET_018_UNRESOLVED_BUCKET_A

    result = subprocess.run(
        [str(_COBRE_BIN), "validate", str(dst)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"cobre validate failed (exit {result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
