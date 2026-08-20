"""Effective-stage calendar resolution for cadastro override registers.

Cadastro override registers in the source model can carry an optional
``(mes, semana, ano)`` triple that makes an override effective from a given
stage forward; a blank triple means the override is effective from the
initial stage. This module provides the single pure function that resolves
that triple to a 0-based operative-calendar stage index, the same way the
source model's own temporal overrides resolve to per-stage effective values
(see ``converters/hydro.py``'s ``_TEMPORAL_OVERRIDE_TYPES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from idecomp.decomp.modelos.dadger import (
    ACALTEFE,
    ACCOTVOL,
    ACDESVIO,
    ACJUSMED,
    ACNUMCON,
    ACNUMJUS,
    ACNUMMAQ,
    ACNUMPOS,
    ACPERHID,
    ACPOTEFE,
    ACPROESP,
    ACVAZEFE,
    ACVAZMIN,
    ACVMDESV,
    ACVOLMAX,
    ACVOLMIN,
    ACVSVERT,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from idecomp.decomp import Dadger

    from cobre_bridge.decomp.temporal import OperativeStage


_MONTH_ABBR_TO_INT: dict[str, int] = {
    "JAN": 1,
    "FEV": 2,
    "MAR": 3,
    "ABR": 4,
    "MAI": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SET": 9,
    "OUT": 10,
    "NOV": 11,
    "DEZ": 12,
}


def _parse_month(mes: str | int | float | None) -> int | None:
    """Normalize an ``AC`` ``mes`` field to a 1..12 month, or ``None`` when blank.

    Blank (``None``, NaN, or an empty/whitespace-only string after
    ``strip()``) resolves to ``None``. An integer/float is coerced with
    ``int(...)``. A non-blank string is matched case-insensitively against
    the source model's 3-letter month abbreviations. A non-blank value that
    is neither a valid month int (1..12) nor a known abbreviation raises
    ``ValueError`` naming the value (a malformed deck is a hard error, never
    a silent default).
    """
    if mes is None:
        return None
    if isinstance(mes, str):
        stripped = mes.strip()
        if not stripped:
            return None
        month = _MONTH_ABBR_TO_INT.get(stripped.upper())
        if month is None:
            raise ValueError(f"unrecognized AC month value: {mes!r}")
        return month
    if pd.isna(mes):
        return None
    month = int(mes)
    if not 1 <= month <= 12:
        raise ValueError(f"unrecognized AC month value: {mes!r}")
    return month


def resolve_effective_stage(
    mes: str | int | float | None,
    semana: int | float | None,
    ano: int | float | None,
    calendar: Sequence[OperativeStage],
) -> int | None:
    """Resolve a ``(mes, semana, ano)`` triple to a 0-based stage index.

    A blank ``mes`` (per :func:`_parse_month`: ``None``, NaN, or an
    empty/whitespace-only string) means the override is effective from the
    initial stage (index ``0``). Otherwise ``(ano, mes)`` is resolved to an
    operative month against *calendar*: a blank ``ano`` defaults to the
    calendar's first stage's year, a month strictly before the calendar's
    horizon also resolves to the initial stage, and a month strictly after
    the horizon resolves to ``None`` for the caller to report as
    out-of-horizon. ``semana`` selects a 1-based operative week within that
    resolved month; a blank ``semana`` (``None``, NaN, or ``0``) resolves to
    the month's first stage, and any other value clamps to the month's last
    available stage (a month represented only by an aggregated stage maps
    every ``semana`` to that one stage).

    Raises
    ------
    ValueError
        If ``mes`` is a non-blank value that is neither a 1..12 int nor a
        known month abbreviation (see :func:`_parse_month`), or if
        ``semana``/``ano`` cannot be coerced to ``int`` (a malformed deck is
        a hard error, never a silent default).
    """
    month = _parse_month(mes)
    if month is None:
        return 0

    resolved_ano = (
        calendar[0].start_date.year if ano is None or pd.isna(ano) else int(ano)
    )

    target = resolved_ano * 12 + (month - 1)
    month_stages = [
        stage
        for stage in calendar
        if stage.season_id == month - 1 and stage.start_date.year == resolved_ano
    ]

    if not month_stages:
        first_ordinal = calendar[0].start_date.year * 12 + calendar[0].season_id
        last_ordinal = calendar[-1].start_date.year * 12 + calendar[-1].season_id
        if target < first_ordinal:
            return 0
        if target > last_ordinal:
            return None

    if semana is None or pd.isna(semana) or semana == 0:
        return month_stages[0].index

    week_index = int(semana) - 1
    return month_stages[min(week_index, len(month_stages) - 1)].index


@dataclass(frozen=True)
class _ScalarAcSpec:
    """One scalar single-value ``AC`` mnemonic to ingest.

    ``ac_class`` is the idecomp typed register class passed as
    ``modificacao=`` to ``Dadger.ac(...)``; ``value_column`` is the frame
    column carrying the new value; ``param`` is the target cadastro column
    the override applies to.
    """

    ac_class: type
    value_column: str
    param: str


#: The scalar single-value ``AC`` mnemonics ingested by
#: :func:`_read_scalar_overrides`. Module scope (rather than function-local)
#: so every consumer of the scalar-override machinery shares one registry.
#: The diversion-channel volume thresholds (`VMDESV`/`VSVERT`) are scalar and
#: belong here; the diversion channel itself (`DESVIO`) is not scalar and gets
#: its own reader, :func:`_read_diversion_overrides`. The three scalar head/
#: productivity mnemonics (`PROESP`/`PERHID`/`JUSMED`) belong here too — the
#: fourth, `COTVOL`, is a multi-row polynomial and gets its own reader instead,
#: :func:`_read_polynomial_overrides`. The topology/gauge pair
#: (`NUMJUS`/`NUMPOS`) goes through the shared :func:`_read_keyed_overrides`
#: (with :func:`_plant_code_key`): single-value and plant-keyed like this
#: tuple's own entries, but `int`-valued (a plant/gauge code), so it does
#: not fit this `float`-typed tuple either.
_SCALAR_AC_SPECS: tuple[_ScalarAcSpec, ...] = (
    _ScalarAcSpec(ACVOLMIN, "volume", "volume_minimo"),
    _ScalarAcSpec(ACVOLMAX, "volume", "volume_maximo"),
    _ScalarAcSpec(ACVAZMIN, "vazao", "vazao_minima_historica"),
    _ScalarAcSpec(ACVMDESV, "volume", "volume_desvio"),
    _ScalarAcSpec(ACVSVERT, "volume", "volume_vertedouro"),
    _ScalarAcSpec(ACPROESP, "produtibilidade", "produtibilidade_especifica"),
    _ScalarAcSpec(ACPERHID, "coeficiente", "perdas"),
    _ScalarAcSpec(ACJUSMED, "cota", "canal_fuga_medio"),
)


#: Every AC register the resolver ingests AND applies to a live consumer —
#: the single source of truth `check decomp` diffs the idecomp AC universe
#: against (see `cobre_bridge.decomp.preflight._ac_coverage`). The scalar
#: portion is derived from `_SCALAR_AC_SPECS`; any new NON-scalar reader
#: wired into `build_effective_cadastro` must add its class to this
#: frozenset too, or `check decomp` will misreport it as deferred.
APPLIED_AC_CLASSES: frozenset[type] = frozenset(
    spec.ac_class for spec in _SCALAR_AC_SPECS
) | frozenset(
    {
        ACDESVIO,
        ACNUMCON,
        ACNUMMAQ,
        ACPOTEFE,
        ACVAZEFE,
        ACCOTVOL,
        ACNUMJUS,
        ACNUMPOS,
    }
)

#: AC registers idecomp models but exposes NO value accessor for, so the
#: resolver cannot ingest them at all — reported distinctly from "deferred
#: (has a value but no consumer)". `ALTEFE` is the sole member (idecomp
#: 1.13.0); see the `TRACKED COBRE-GAP WORKAROUND` in
#: `cobre_bridge.decomp.hydro.convert_hydros`.
UNINGESTABLE_AC_CLASSES: frozenset[type] = frozenset({ACALTEFE})


@dataclass(frozen=True)
class OutOfHorizon:
    """An ``AC`` override whose effective date falls after the calendar horizon.

    Reported rather than dropped: the cadastro assembly surfaces these so a
    deck that overrides a plant's cadastro beyond the study horizon is never
    silently ignored.
    """

    code: int
    param: str
    mes: int
    ano: int


def _out_of_horizon_record(
    code: int,
    mes: str | int | float | None,
    ano: int | float | None,
    param: str,
    calendar: Sequence[OperativeStage],
) -> OutOfHorizon:
    """Build the :class:`OutOfHorizon` record for an override that resolved
    past the calendar horizon.

    Shared by :func:`_read_scalar_overrides` and
    :func:`_read_diversion_overrides`, which report an out-of-horizon override
    the same way, differing only in the *param* label.
    """
    resolved_ano = (
        calendar[0].start_date.year if ano is None or pd.isna(ano) else int(ano)
    )
    month = _parse_month(mes)
    assert month is not None  # eff is None only for a non-blank mes
    return OutOfHorizon(code, param, month, resolved_ano)


def _read_scalar_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> tuple[dict[tuple[int, str], list[tuple[int, float]]], list[OutOfHorizon]]:
    """Ingest the scalar single-value ``AC`` overrides via idecomp's typed accessor.

    For each registered :class:`_ScalarAcSpec`, reads
    ``dadger.ac(codigo_usina=None, modificacao=spec.ac_class, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. Rows that resolve within the horizon
    are grouped by ``(code, param)``; rows that resolve past the horizon are
    reported in the returned out-of-horizon list instead of being dropped.

    Raises
    ------
    KeyError
        If an accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[tuple[int, str], list[tuple[int, float]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    for spec in _SCALAR_AC_SPECS:
        table = dadger.ac(codigo_usina=None, modificacao=spec.ac_class, df=True)
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for _, row in table.iterrows():
            code = int(row["codigo_usina"])
            value = float(row[spec.value_column])
            mes = row["mes"]
            eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
            if eff is None:
                out_of_horizon.append(
                    _out_of_horizon_record(code, mes, row["ano"], spec.param, calendar)
                )
                continue
            records.setdefault((code, spec.param), []).append((eff, value))
    return records, out_of_horizon


@dataclass(frozen=True)
class DiversionChannel:
    """The diversion channel (canal de desvio) active for a plant at a stage.

    ``downstream`` is the ``codigo_usina`` of the plant that receives the
    diverted water; ``limit`` is the channel's flow limit in m³/s, or
    ``None`` for a base-declared diversion, which carries no explicit
    limit — only an ``AC DESVIO`` override supplies one.
    """

    downstream: int
    limit: float | None


def _read_diversion_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> tuple[dict[int, list[tuple[int, DiversionChannel]]], list[OutOfHorizon]]:
    """Ingest the ``AC DESVIO`` diversion-channel override via idecomp's typed accessor.

    Mirrors :func:`_read_scalar_overrides` one-for-one: reads
    ``dadger.ac(codigo_usina=None, modificacao=ACDESVIO, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. Rows that resolve within the horizon
    are grouped by the source plant *code* as ``(eff_stage, DiversionChannel(
    downstream, limit))``; rows that resolve past the horizon are reported
    (``param="diversion"``) in the returned out-of-horizon list instead of
    being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[int, list[tuple[int, DiversionChannel]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ACDESVIO, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return records, out_of_horizon
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        channel = DiversionChannel(
            int(row["codigo_usina_jusante"]), float(row["limite_vazao"])
        )
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], "diversion", calendar)
            )
            continue
        records.setdefault(code, []).append((eff, channel))
    return records, out_of_horizon


@dataclass(frozen=True)
class MachineSet:
    """One conjunto's effective machine configuration at a stage.

    ``numero_maquinas`` is the conjunto's machine-unit count; ``potencia``
    and ``vazao`` are its per-unit rated power (MW) and rated flow (m³/s) —
    the same three quantities the source model's ``NUMMAQ``/``POTEFE``/
    ``VAZEFE`` registers carry, densified per stage by
    :func:`build_effective_cadastro`.
    """

    numero_maquinas: int
    potencia: float
    vazao: float


def _plant_code_key(row: Any) -> int:
    """Group-key for the plant-keyed single-value ``AC`` mnemonics —
    ``codigo_usina`` alone (``NUMJUS``/``NUMPOS``)."""
    return int(row["codigo_usina"])


def _conjunto_key(row: Any) -> tuple[int, int]:
    """Group-key for the compound-keyed machine-set mnemonics —
    ``(codigo_usina, indice_conjunto)`` (``NUMMAQ``/``POTEFE``/``VAZEFE``)."""
    return int(row["codigo_usina"]), int(row["indice_conjunto"])


def _read_keyed_overrides[K, T](
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
    ac_class: type,
    value_column: str,
    param: str,
    value_caster: Callable[[Any], T],
    key: Callable[[Any], K],
) -> tuple[dict[K, list[tuple[int, T]]], list[OutOfHorizon]]:
    """Ingest one single-value ``AC`` mnemonic, grouped by *key*.

    The shared body for both single-value override shapes that differ only in
    their grouping key: the compound-key machine-set mnemonics
    (``NUMMAQ``/``POTEFE``/``VAZEFE``, grouped by ``(codigo_usina,
    indice_conjunto)`` via :func:`_conjunto_key`) and the plant-keyed
    topology/gauge mnemonics (``NUMJUS``/``NUMPOS``, grouped by
    ``codigo_usina`` via :func:`_plant_code_key`). Sharing one loop keeps
    these from repeating near-identical bodies, the same way
    :func:`_forward_fill_series` is reused across the ``int``/``float``
    per-stage series it densifies. ``NUMCON`` keeps its own plant-keyed loop
    in :func:`_read_machine_set_overrides`, and the multi-row polynomial shape
    (:func:`_read_polynomial_overrides`) is genuinely different — neither is
    folded in here. Does not reuse :func:`_read_scalar_overrides`'s
    ``_SCALAR_AC_SPECS`` tuple: that tuple is uniformly ``float``-typed, while
    these mnemonics span ``int`` (machine counts, plant/gauge codes) and
    ``float`` (rated power/flow) via *value_caster*.

    Reads ``dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. In-horizon rows are grouped by
    ``key(row)`` as ``(eff_stage, value_caster(row[value_column]))``; rows
    that resolve past the horizon are reported (with *param* as the label) in
    the returned out-of-horizon list instead of being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[K, list[tuple[int, T]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return records, out_of_horizon
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], param, calendar)
            )
            continue
        records.setdefault(key(row), []).append((eff, value_caster(row[value_column])))
    return records, out_of_horizon


def _read_machine_set_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
) -> tuple[
    dict[int, list[tuple[int, int]]],
    dict[tuple[int, int], list[tuple[int, int]]],
    dict[tuple[int, int], list[tuple[int, float]]],
    dict[tuple[int, int], list[tuple[int, float]]],
    list[OutOfHorizon],
]:
    """Ingest the machine-set ``AC`` overrides via idecomp's typed accessor.

    Covers the third distinct override shape (after the scalar single-value
    mnemonics and the non-scalar diversion channel): ``NUMCON`` is
    **plant-keyed** (one conjunto count per ``codigo_usina``), while
    ``NUMMAQ``/``POTEFE``/``VAZEFE`` are **compound-keyed** by
    ``(codigo_usina, indice_conjunto)`` — a conjunto's machine count, rated
    power, and rated flow. ``NUMCON`` is read by its own plant-keyed loop
    (its frame carries no ``indice_conjunto`` column); the other three are
    read through :func:`_read_keyed_overrides` with :func:`_conjunto_key`, one
    call per mnemonic.
    Every row's ``(mes, semana, ano)`` triple is resolved to an effective
    stage via :func:`resolve_effective_stage`; in-horizon rows are grouped by
    key, past-horizon rows are reported (param labels ``"numero_conjuntos"``,
    ``"numero_maquinas"``, ``"potencia"``, ``"vazao"``) in the returned
    out-of-horizon list instead of being dropped. Reads a frame only when
    ``isinstance(table, pd.DataFrame) and not table.empty``, mirroring
    :func:`_read_scalar_overrides`/:func:`_read_diversion_overrides`, so a
    ``None``/empty frame contributes nothing.

    A fifth machine-configuration mnemonic, ``ALTEFE`` (effective head), is
    deliberately **not** ingested here: the installed idecomp accessor
    exposes no value property on it, only the identifying/timing columns, so
    its ``df=True`` frame carries nothing to consume. Do not import or read
    it without a value accessor to back it.

    Raises
    ------
    KeyError
        If an accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    numero_conjuntos: dict[int, list[tuple[int, int]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ACNUMCON, df=True)
    if isinstance(table, pd.DataFrame) and not table.empty:
        for _, row in table.iterrows():
            code = int(row["codigo_usina"])
            mes = row["mes"]
            eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
            if eff is None:
                out_of_horizon.append(
                    _out_of_horizon_record(
                        code, mes, row["ano"], "numero_conjuntos", calendar
                    )
                )
                continue
            numero_conjuntos.setdefault(code, []).append(
                (eff, int(row["numero_conjuntos"]))
            )

    numero_maquinas, maquinas_out_of_horizon = _read_keyed_overrides(
        dadger,
        calendar,
        ACNUMMAQ,
        "numero_maquinas",
        "numero_maquinas",
        int,
        _conjunto_key,
    )
    potencia, potencia_out_of_horizon = _read_keyed_overrides(
        dadger, calendar, ACPOTEFE, "potencia", "potencia", float, _conjunto_key
    )
    vazao, vazao_out_of_horizon = _read_keyed_overrides(
        dadger, calendar, ACVAZEFE, "vazao", "vazao", float, _conjunto_key
    )
    out_of_horizon.extend(maquinas_out_of_horizon)
    out_of_horizon.extend(potencia_out_of_horizon)
    out_of_horizon.extend(vazao_out_of_horizon)

    return numero_conjuntos, numero_maquinas, potencia, vazao, out_of_horizon


def _read_polynomial_overrides(
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
    ac_class: type,
    param: str,
) -> tuple[dict[int, list[tuple[int, tuple[float, ...]]]], list[OutOfHorizon]]:
    """Ingest a multi-row-per-plant polynomial ``AC`` mnemonic (the fifth shape).

    Unlike the scalar, diversion, and compound-key shapes above, one
    *effective* override here is not a single row: ``AC COTVOL`` declares the
    plant's full 5-coefficient forebay-cota polynomial as up to five separate
    rows, one per **1-based** ``ordem`` (1..5, the coefficients a0..a4 — the
    reader normalises it to a 0-based tuple index), all sharing the same
    ``(codigo_usina, mes, semana, ano)`` triple. Reads
    ``dadger.ac(codigo_usina=None,
    modificacao=ac_class, df=True)`` and resolves **every row's own**
    ``(mes, semana, ano)`` triple to an effective stage via
    :func:`resolve_effective_stage`, then groups rows by ``(codigo_usina,
    effective_stage)`` rather than by the raw date fields themselves — a
    blank-date group's ``mes``/``semana``/``ano`` are typically distinct NaN
    objects that compare unequal to each other (``nan != nan``), which would
    silently fragment one group into several under a raw-field dict key;
    resolving first and grouping by the resolved stage sidesteps that
    entirely. Within a group, an ``ordem`` absent from the rows defaults its
    coefficient to ``0.0`` — the one documented default in this reader, never
    applied to an ``ordem`` that *is* present. Rows that resolve within the
    horizon are grouped by *codigo_usina* as ``(eff_stage, coefficients)``,
    *coefficients* ordered ``ordem 0..4``; rows whose date resolves past the
    horizon are reported (with *param* as the label) in the returned
    out-of-horizon list instead of being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    ValueError
        If a row's 1-based ``ordem`` is outside 1..5 — an out-of-range
        coefficient index is a malformed register, never silently dropped.
    """
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return {}, out_of_horizon

    coeffs_by_group: dict[tuple[int, int], dict[int, float]] = {}
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        # ``AC COTVOL`` numbers its coefficients 1..5 (a0..a4) and idecomp
        # surfaces that raw 1-based ``ordem`` verbatim. Normalise it to the
        # 0-based index the coefficient tuple below is read at: leaving the
        # 1-based value in place shifts every coefficient up one slot, which
        # silently zeroes the a0 constant term and drops a4 — for a
        # run-of-river plant whose override is a single constant forebay cota
        # (a0), that turns a fixed level of, say, 90 m into ``90·volume``.
        coeff_index = int(row["ordem"]) - 1
        if not 0 <= coeff_index < 5:
            raise ValueError(
                f"{param}: coefficient order {int(row['ordem'])} for plant "
                f"{code} is outside the expected 1..5 range"
            )
        coeficiente = float(row["coeficiente"])
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], param, calendar)
            )
            continue
        coeffs_by_group.setdefault((code, eff), {})[coeff_index] = coeficiente

    records: dict[int, list[tuple[int, tuple[float, ...]]]] = {}
    for (code, eff), coeffs_by_ordem in coeffs_by_group.items():
        coefficients = tuple(coeffs_by_ordem.get(i, 0.0) for i in range(5))
        records.setdefault(code, []).append((eff, coefficients))

    return records, out_of_horizon


def _forward_fill_series[T](
    base_value: T,
    records: Sequence[tuple[int, T]],
    n_stages: int,
) -> list[T]:
    """Densify a sparse set of effective-stage overrides into a dense series.

    *records* is a possibly-unordered, possibly-empty sequence of
    ``(effective_stage, value)`` pairs. The value at stage ``s`` is the value
    of the last record whose ``effective_stage <= s``, else *base_value*. A
    record at stage ``0`` therefore overwrites the base for every stage
    (permanent semantics); later records supersede earlier ones. *records*
    is not mutated. Generic over the value type ``T`` so both the scalar
    ``float`` overrides and the non-scalar ``DiversionChannel | None``
    overrides share this one densification helper.
    """
    ordered = sorted(records, key=lambda record: record[0])
    series: list[T] = []
    current = base_value
    next_index = 0
    for stage in range(n_stages):
        while next_index < len(ordered) and ordered[next_index][0] <= stage:
            current = ordered[next_index][1]
            next_index += 1
        series.append(current)
    return series


@dataclass(frozen=True)
class EffectiveCadastro:
    """Per-stage-effective view of the cadastro.

    Holds the base cadastro (the ``hidr`` DataFrame, indexed by plant code)
    alongside a sparse map of the ``(code, param)`` pairs that carry at
    least one override, each already densified to a per-stage tuple by
    :func:`_forward_fill_series`. A ``(code, param)`` pair absent from
    *stage_varying* has no override and falls through to *base* for every
    stage.
    """

    base: pd.DataFrame
    n_stages: int
    stage_varying: Mapping[tuple[int, str], tuple[float, ...]]
    diversions: Mapping[int, tuple[DiversionChannel | None, ...]] = field(
        default_factory=dict
    )
    machine_conjunto_counts: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    machine_sets: Mapping[tuple[int, int], tuple[MachineSet, ...]] = field(
        default_factory=dict
    )
    cota_polynomials: Mapping[int, tuple[tuple[float, ...], ...]] = field(
        default_factory=dict
    )
    downstream_links: Mapping[int, tuple[int, ...]] = field(default_factory=dict)
    inflow_gauges: Mapping[int, tuple[int, ...]] = field(default_factory=dict)

    def value(self, code: int, param: str, stage_index: int) -> float:
        """Effective value of *param* for plant *code* at *stage_index*."""
        key = (code, param)
        if key in self.stage_varying:
            return self.stage_varying[key][stage_index]
        return float(self.base.loc[code, param])

    def series(self, code: int, param: str) -> list[float]:
        """Dense per-stage series of *param* for plant *code*."""
        key = (code, param)
        if key in self.stage_varying:
            return list(self.stage_varying[key])
        return [float(self.base.loc[code, param])] * self.n_stages

    def is_stage_varying(self, code: int, param: str) -> bool:
        """Whether *param* for plant *code* carries at least one override."""
        return (code, param) in self.stage_varying

    def has_diversion(self, code: int) -> bool:
        """Whether plant *code* has a base or ``AC DESVIO`` diversion channel.

        ``True`` iff *code* is a key of :attr:`diversions`, i.e. it carries a
        non-zero base ``desvio`` code or an in-horizon ``AC DESVIO`` record in
        at least one stage.
        """
        return code in self.diversions

    def diversion(self, code: int, stage_index: int) -> DiversionChannel | None:
        """The diversion channel active for plant *code* at *stage_index*.

        ``None`` when plant *code* has no diversion at all (absent from
        :attr:`diversions`) or when it has a diversion in some stages but not
        at *stage_index*.
        """
        if code not in self.diversions:
            return None
        return self.diversions[code][stage_index]

    def machine_conjunto_count(self, code: int, stage_index: int) -> int | None:
        """The overridden per-stage ``AC NUMCON`` conjunto count for plant *code*.

        ``None`` when *code* carries no ``NUMCON`` override at all (absent
        from :attr:`machine_conjunto_counts`) — the caller then falls through
        to the ``hidr`` base ``numero_conjuntos_maquinas``.
        """
        if code not in self.machine_conjunto_counts:
            return None
        return self.machine_conjunto_counts[code][stage_index]

    def machine_set(
        self, code: int, indice_conjunto: int, stage_index: int
    ) -> MachineSet | None:
        """The overridden per-stage machine configuration for one conjunto.

        ``None`` when ``(code, indice_conjunto)`` carries no ``NUMMAQ``/
        ``POTEFE``/``VAZEFE`` override at all (absent from
        :attr:`machine_sets`) — the caller then falls through to the
        ``hidr`` base per-conjunto columns.
        """
        key = (code, indice_conjunto)
        if key not in self.machine_sets:
            return None
        return self.machine_sets[key][stage_index]

    def cota_polynomial(self, code: int, stage_index: int) -> tuple[float, ...]:
        """The effective 5-coefficient forebay-cota polynomial for plant
        *code* at *stage_index*, ordered ``ordem 0..4``.

        Falls through to the base ``a{0..4}_volume_cota`` columns when *code*
        carries no ``AC COTVOL`` override at all (absent from
        :attr:`cota_polynomials`) — the same "absent means base" convention
        every other accessor here follows.
        """
        if code not in self.cota_polynomials:
            return tuple(
                float(self.base.loc[code, f"a{i}_volume_cota"]) for i in range(5)
            )
        return self.cota_polynomials[code][stage_index]

    def downstream_plant(self, code: int, stage_index: int) -> int:
        """The effective downstream-plant code for *code* at *stage_index*.

        Falls through to the base ``codigo_usina_jusante`` when *code*
        carries no ``AC NUMJUS`` override at all (absent from
        :attr:`downstream_links`) — the same "absent means base" convention
        every other accessor here follows. ``0`` is a valid return (the
        sink); the cascade walk (:func:`~cobre_bridge.decomp.hydro.
        _downstream_operated`) treats it as such.
        """
        if code not in self.downstream_links:
            return int(self.base.loc[code, "codigo_usina_jusante"])
        return self.downstream_links[code][stage_index]

    def downstream_plant_varies(self, code: int) -> bool:
        """Whether *code*'s effective downstream link varies across stages
        (a temporal ``AC NUMJUS``) — the tracked-gap trigger:
        :func:`~cobre_bridge.decomp.hydro._downstream_operated` reads one
        stage-representative link for the whole horizon (stage 0 by
        default), so a caller checks this to warn rather than silently
        picking a stage. ``False`` for a plant with no override at all.
        """
        if code not in self.downstream_links:
            return False
        series = self.downstream_links[code]
        return any(value != series[0] for value in series)

    def inflow_gauge(self, code: int, stage_index: int) -> int:
        """The effective inflow-gauge (``posto``) for *code* at *stage_index*.

        Falls through to the base ``posto`` when *code* carries no ``AC
        NUMPOS`` override at all (absent from :attr:`inflow_gauges`).
        """
        if code not in self.inflow_gauges:
            return int(self.base.loc[code, "posto"])
        return self.inflow_gauges[code][stage_index]

    def inflow_gauge_varies(self, code: int) -> bool:
        """Whether *code*'s effective inflow gauge varies across stages (a
        temporal ``AC NUMPOS``) — the gauge sibling of
        :meth:`downstream_plant_varies`.
        """
        if code not in self.inflow_gauges:
            return False
        series = self.inflow_gauges[code]
        return any(value != series[0] for value in series)


def effective_storage_range(
    effective: EffectiveCadastro, code: int, stage_index: int
) -> tuple[float, float]:
    """Per-stage effective storage range for plant *code*, in hm³.

    A run-of-river plant (``tipo_regulacao == "D"``) cannot accumulate water
    across stages — the source model's own precedent (``converters.hydro``'s
    ``tipo_reg == "D"`` branch) freezes its operating range at the reference
    volume (``volume_referencia``) rather than the ``hidr`` registry's
    ``(volume_minimo, volume_maximo)`` band, so this collapses to
    ``(vol_ref, vol_ref)`` for a ``D`` plant. ``vol_ref`` falls back to the
    per-stage effective ``volume_minimo`` when ``volume_referencia`` is
    missing, ``NaN``, or ``<= 0`` — never a zero-width range at zero.
    ``tipo_regulacao`` is read defensively off the base row
    (``row.get(..., "")`` then ``str(...).strip()``, matching that same
    precedent), so a synthetic cadastro carrying no such column simply falls
    through to the per-stage read below.

    Every other regulation class — under the DECOMP predicate, a reservoir
    is ``tipo_regulacao in ("M", "S")`` — is unchanged: the per-stage
    ``(volume_minimo, volume_maximo)`` via :meth:`EffectiveCadastro.value`.
    This is the one place the ``D``-collapse predicate lives; every storage
    consumer (:func:`storage_envelope`, :func:`cobre_bridge.decomp.bounds.
    convert_storage_bounds`, :func:`cobre_bridge.decomp.hydro.
    convert_initial_storage`) routes through it. Productivity does **not** —
    :func:`cobre_bridge.decomp.hydro._equivalent_productivity_mw_per_m3s`
    keeps reading the full ``(volume_minimo, volume_maximo)`` range directly,
    validated independently of this collapse.
    """
    row = effective.base.loc[code]
    tipo_reg = str(row.get("tipo_regulacao", "")).strip()
    if tipo_reg == "D":
        vol_ref_raw = row.get("volume_referencia")
        if vol_ref_raw is not None and not pd.isna(vol_ref_raw) and vol_ref_raw > 0:
            vol_ref = float(vol_ref_raw)
        else:
            vol_ref = effective.value(code, "volume_minimo", stage_index)
        return (vol_ref, vol_ref)
    return (
        effective.value(code, "volume_minimo", stage_index),
        effective.value(code, "volume_maximo", stage_index),
    )


def storage_envelope(effective: EffectiveCadastro, code: int) -> tuple[float, float]:
    """Outer per-stage operating range for plant *code*, in hm³.

    ``(min over stages of the effective floor, max over stages of the
    effective ceiling)`` per :func:`effective_storage_range` — the widest
    floor/ceiling the plant's dense per-stage range ever reaches. For a
    plant with no override and no run-of-river collapse both reduce to the
    base scalar. A run-of-river (``D``) plant's per-stage range is already
    the single-point collapse ``(vol_ref, vol_ref)`` at every stage, so its
    envelope collapses to that same point; every ``M``/``S`` plant is
    unchanged. This is the envelope the entity ``reservoir`` block
    declares as its default storage bounds;
    :func:`cobre_bridge.decomp.bounds.convert_storage_bounds` emits a
    per-stage override wherever a stage's effective bounds differ from it.
    """
    ranges = [
        effective_storage_range(effective, code, stage_index)
        for stage_index in range(effective.n_stages)
    ]
    return (min(r[0] for r in ranges), max(r[1] for r in ranges))


@dataclass(frozen=True)
class CadastroResolutionReport:
    """Summary of the scalar ``AC`` overrides resolved by ``build_effective_cadastro``.

    ``applied`` maps each cadastro parameter to the count of distinct plant
    codes that received at least one in-horizon override; ``out_of_horizon``
    is the same tuple :func:`_read_scalar_overrides` returned, surfaced here
    for the caller instead of being silently dropped.
    """

    applied: Mapping[str, int]
    out_of_horizon: tuple[OutOfHorizon, ...]


def _require_cadastro_row(hidr: pd.DataFrame, code: int) -> None:
    """Raise ``ValueError`` if *code* has no row in the cadastro registry."""
    if code not in hidr.index:
        raise ValueError(
            f"AC override references plant code {code}, which is not in"
            " the cadastro registry"
        )


def build_effective_cadastro(
    dadger: Dadger,
    hidr: pd.DataFrame,
    calendar: Sequence[OperativeStage],
) -> tuple[EffectiveCadastro, CadastroResolutionReport]:
    """Assemble the per-stage-effective cadastro and its resolution report.

    Reads the scalar ``AC`` overrides from *dadger* via
    :func:`_read_scalar_overrides`, densifies each in-horizon ``(code,
    param)`` group against *hidr*'s base value with
    :func:`_forward_fill_series`, and reports how many distinct plants were
    touched per parameter. *hidr* is treated as an immutable base view and is
    never mutated; a ``(code, param)`` pair with no override is simply absent
    from the returned :class:`EffectiveCadastro`'s ``stage_varying`` map and
    falls through to *hidr* for every stage.

    Also builds the diversion-channel map from :func:`_read_diversion_overrides`
    for the union of {plants with a non-zero base ``desvio``} and {plants with
    an in-horizon ``AC DESVIO``}: the base ``desvio`` seeds the forward-fill
    (as ``DiversionChannel(downstream, limit=None)``, or ``None`` when
    ``desvio`` is zero), the resolved ``AC DESVIO`` records overlay it, and
    the result is stored densely per stage — sparsely, i.e. only for plants
    with a base or ``AC`` diversion. ``diversions`` is absent-by-default for
    every other plant, matching *stage_varying*'s sparsity discipline.

    Also builds the two machine-set maps from :func:`_read_machine_set_overrides`,
    each densified independently against *hidr*'s own per-conjunto columns:
    ``machine_conjunto_counts`` (sparse by plant code, seeded from
    ``numero_conjuntos_maquinas``) and ``machine_sets`` (sparse by
    ``(code, indice_conjunto)``, seeded from ``maquinas_conjunto_{k}``/
    ``potencia_nominal_conjunto_{k}``/``vazao_nominal_conjunto_{k}`` and
    zipped into a :class:`MachineSet` tuple). *hidr*'s per-conjunto columns
    are read only for the ``(code, k)`` pairs that actually carry an
    override — never scanned unconditionally — so a conjunto with, say, a
    ``NUMMAQ`` override but no ``POTEFE``/``VAZEFE`` override still gets its
    ``potencia``/``vazao`` from the ``hidr`` base at every stage.

    Also builds the ``cota_polynomials`` map from :func:`_read_polynomial_overrides`
    (the fifth, multi-row-per-plant shape): sparse by plant code, seeded from
    the base ``a{0..4}_volume_cota`` columns and densified the same way as
    every other per-stage series, one full 5-coefficient polynomial per
    stage rather than one scalar.

    Also builds ``downstream_links``/``inflow_gauges`` from
    :func:`_read_keyed_overrides`: sparse by plant
    code, seeded from the base ``codigo_usina_jusante``/``posto`` columns
    and densified the same way as every other per-stage series. Consumed by
    :meth:`EffectiveCadastro.downstream_plant`/``inflow_gauge`` — the
    cascade walk (:func:`~cobre_bridge.decomp.hydro._downstream_operated`)
    and the incremental-inflow gauge attribution
    (:func:`~cobre_bridge.decomp.scenarios._incremental_context`) read one
    stage-representative (stage 0) value off these, never per-stage, so a
    temporal ``NUMJUS``/``NUMPOS`` is a tracked gap
    (:meth:`EffectiveCadastro.downstream_plant_varies`/``inflow_gauge_varies``),
    not a silent per-stage cascade. ``AC JUSENA``/``AC NPOSNW`` are
    deliberately **not** ingested here — no DECOMP consumer; see the
    deferred-fidelity warning in :mod:`cobre_bridge.decomp.hydro`.

    Raises
    ------
    ValueError
        If an override references a plant *code* absent from ``hidr.index``
        — the registry has no cadastro row to override. Applies to the
        scalar overrides, the source plant of an ``AC DESVIO`` (the
        downstream plant it names is not validated here — that is M2.1's
        concern), and every machine-set override.
    """
    records, out_of_horizon = _read_scalar_overrides(dadger, calendar)
    diversion_records, diversion_out_of_horizon = _read_diversion_overrides(
        dadger, calendar
    )
    (
        numero_conjuntos_records,
        numero_maquinas_records,
        potencia_records,
        vazao_records,
        machine_out_of_horizon,
    ) = _read_machine_set_overrides(dadger, calendar)
    cota_records, cota_out_of_horizon = _read_polynomial_overrides(
        dadger, calendar, ACCOTVOL, "cota_volume"
    )
    topology_records, topology_out_of_horizon = _read_keyed_overrides(
        dadger,
        calendar,
        ACNUMJUS,
        "codigo_usina_jusante",
        "codigo_usina_jusante",
        int,
        _plant_code_key,
    )
    gauge_records, gauge_out_of_horizon = _read_keyed_overrides(
        dadger, calendar, ACNUMPOS, "codigo_posto", "posto", int, _plant_code_key
    )
    out_of_horizon = (
        out_of_horizon
        + diversion_out_of_horizon
        + machine_out_of_horizon
        + cota_out_of_horizon
        + topology_out_of_horizon
        + gauge_out_of_horizon
    )

    n_stages = len(calendar)
    stage_varying: dict[tuple[int, str], tuple[float, ...]] = {}
    applied: dict[str, int] = {}
    for (code, param), overrides in records.items():
        _require_cadastro_row(hidr, code)
        base_value = float(hidr.loc[code, param])
        stage_varying[(code, param)] = tuple(
            _forward_fill_series(base_value, overrides, n_stages)
        )
        applied[param] = applied.get(param, 0) + 1

    base_diversion_codes = {
        int(code)
        for code, desvio in zip(
            hidr.index.tolist(), hidr["desvio"].tolist(), strict=True
        )
        if float(desvio) != 0
    }
    diversions: dict[int, tuple[DiversionChannel | None, ...]] = {}
    for code in base_diversion_codes | set(diversion_records):
        _require_cadastro_row(hidr, code)
        base_desvio = float(hidr.loc[code, "desvio"])
        base_channel = (
            DiversionChannel(int(base_desvio), None) if base_desvio != 0 else None
        )
        diversion_overrides = diversion_records.get(code, [])
        diversions[code] = tuple(
            _forward_fill_series(base_channel, diversion_overrides, n_stages)
        )
    if diversion_records:
        applied["diversion"] = len(diversion_records)

    machine_conjunto_counts: dict[int, tuple[int, ...]] = {}
    for code, count_overrides in numero_conjuntos_records.items():
        _require_cadastro_row(hidr, code)
        base_count = int(hidr.loc[code, "numero_conjuntos_maquinas"])
        machine_conjunto_counts[code] = tuple(
            _forward_fill_series(base_count, count_overrides, n_stages)
        )
    if numero_conjuntos_records:
        applied["numero_conjuntos"] = len(numero_conjuntos_records)

    machine_sets: dict[tuple[int, int], tuple[MachineSet, ...]] = {}
    compound_keys = (
        set(numero_maquinas_records) | set(potencia_records) | set(vazao_records)
    )
    for code, conjunto in compound_keys:
        _require_cadastro_row(hidr, code)
        base_numero_maquinas = int(hidr.loc[code, f"maquinas_conjunto_{conjunto}"])
        base_potencia = float(hidr.loc[code, f"potencia_nominal_conjunto_{conjunto}"])
        base_vazao = float(hidr.loc[code, f"vazao_nominal_conjunto_{conjunto}"])
        numero_maquinas_series = _forward_fill_series(
            base_numero_maquinas,
            numero_maquinas_records.get((code, conjunto), []),
            n_stages,
        )
        potencia_series = _forward_fill_series(
            base_potencia, potencia_records.get((code, conjunto), []), n_stages
        )
        vazao_series = _forward_fill_series(
            base_vazao, vazao_records.get((code, conjunto), []), n_stages
        )
        machine_sets[(code, conjunto)] = tuple(
            MachineSet(numero_maquinas_series[s], potencia_series[s], vazao_series[s])
            for s in range(n_stages)
        )
    if numero_maquinas_records:
        applied["numero_maquinas"] = len(numero_maquinas_records)
    if potencia_records:
        applied["potencia"] = len(potencia_records)
    if vazao_records:
        applied["vazao"] = len(vazao_records)

    cota_polynomials: dict[int, tuple[tuple[float, ...], ...]] = {}
    for code, cota_overrides in cota_records.items():
        _require_cadastro_row(hidr, code)
        base_coefficients = tuple(
            float(hidr.loc[code, f"a{i}_volume_cota"]) for i in range(5)
        )
        cota_polynomials[code] = tuple(
            _forward_fill_series(base_coefficients, cota_overrides, n_stages)
        )
    if cota_records:
        applied["cota_volume"] = len(cota_records)

    downstream_links: dict[int, tuple[int, ...]] = {}
    for code, topology_overrides in topology_records.items():
        _require_cadastro_row(hidr, code)
        base_downstream = int(hidr.loc[code, "codigo_usina_jusante"])
        downstream_links[code] = tuple(
            _forward_fill_series(base_downstream, topology_overrides, n_stages)
        )
    if topology_records:
        applied["codigo_usina_jusante"] = len(topology_records)

    inflow_gauges: dict[int, tuple[int, ...]] = {}
    for code, gauge_overrides in gauge_records.items():
        _require_cadastro_row(hidr, code)
        base_gauge = int(hidr.loc[code, "posto"])
        inflow_gauges[code] = tuple(
            _forward_fill_series(base_gauge, gauge_overrides, n_stages)
        )
    if gauge_records:
        applied["posto"] = len(gauge_records)

    effective = EffectiveCadastro(
        base=hidr,
        n_stages=n_stages,
        stage_varying=stage_varying,
        diversions=diversions,
        machine_conjunto_counts=machine_conjunto_counts,
        machine_sets=machine_sets,
        cota_polynomials=cota_polynomials,
        downstream_links=downstream_links,
        inflow_gauges=inflow_gauges,
    )
    report = CadastroResolutionReport(
        applied=applied, out_of_horizon=tuple(out_of_horizon)
    )
    return effective, report
