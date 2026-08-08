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
    ACDESVIO,
    ACNUMCON,
    ACNUMMAQ,
    ACPOTEFE,
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
#: so every consumer of the scalar-override machinery shares one registry —
#: epic-03 extends this same tuple with the diversion-channel volume
#: thresholds (`VMDESV`/`VSVERT`); the diversion channel itself (`DESVIO`)
#: is not scalar and gets its own reader, :func:`_read_diversion_overrides`.
_SCALAR_AC_SPECS: tuple[_ScalarAcSpec, ...] = (
    _ScalarAcSpec(ACVOLMIN, "volume", "volume_minimo"),
    _ScalarAcSpec(ACVOLMAX, "volume", "volume_maximo"),
    _ScalarAcSpec(ACVAZMIN, "vazao", "vazao_minima_historica"),
    _ScalarAcSpec(ACVMDESV, "volume", "volume_desvio"),
    _ScalarAcSpec(ACVSVERT, "volume", "volume_vertedouro"),
)


@dataclass(frozen=True)
class OutOfHorizon:
    """An ``AC`` override whose effective date falls after the calendar horizon.

    Reported rather than dropped: ticket-004's assembly surfaces these so a
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


def _read_compound_key_overrides[T](
    dadger: Dadger,
    calendar: Sequence[OperativeStage],
    ac_class: type,
    value_column: str,
    param: str,
    value_caster: Callable[[Any], T],
) -> tuple[dict[tuple[int, int], list[tuple[int, T]]], list[OutOfHorizon]]:
    """Ingest one compound-key ``(codigo_usina, indice_conjunto)`` ``AC`` mnemonic.

    Shared body for the three compound-key machine-set mnemonics
    (``NUMMAQ``/``POTEFE``/``VAZEFE``) that :func:`_read_machine_set_overrides`
    calls once per mnemonic — this is what keeps that reader from repeating
    three near-identical loops, the same way :func:`_forward_fill_series` is
    reused for both the ``int`` and ``float`` per-stage series it densifies.
    Reads ``dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)`` and
    resolves every row's ``(mes, semana, ano)`` triple to an effective stage
    via :func:`resolve_effective_stage`. Rows that resolve within the horizon
    are grouped by ``(codigo_usina, indice_conjunto)`` as ``(eff_stage,
    value_caster(row[value_column]))``; rows that resolve past the horizon
    are reported (with *param* as the label) in the returned out-of-horizon
    list instead of being dropped.

    Raises
    ------
    KeyError
        If the accessor frame is missing an expected column (a malformed
        idecomp frame is a hard error, never a silent default).
    """
    records: dict[tuple[int, int], list[tuple[int, T]]] = {}
    out_of_horizon: list[OutOfHorizon] = []
    table = dadger.ac(codigo_usina=None, modificacao=ac_class, df=True)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return records, out_of_horizon
    for _, row in table.iterrows():
        code = int(row["codigo_usina"])
        conjunto = int(row["indice_conjunto"])
        mes = row["mes"]
        eff = resolve_effective_stage(mes, row["semana"], row["ano"], calendar)
        if eff is None:
            out_of_horizon.append(
                _out_of_horizon_record(code, mes, row["ano"], param, calendar)
            )
            continue
        value = value_caster(row[value_column])
        records.setdefault((code, conjunto), []).append((eff, value))
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
    read through :func:`_read_compound_key_overrides`, one call per mnemonic.
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
    its ``df=True`` frame carries nothing to consume — and effective head is
    out of this ticket's scope regardless (E5 head/productivity work). Do
    not import or read it without a value accessor to back it.

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

    numero_maquinas, maquinas_out_of_horizon = _read_compound_key_overrides(
        dadger, calendar, ACNUMMAQ, "numero_maquinas", "numero_maquinas", int
    )
    potencia, potencia_out_of_horizon = _read_compound_key_overrides(
        dadger, calendar, ACPOTEFE, "potencia", "potencia", float
    )
    vazao, vazao_out_of_horizon = _read_compound_key_overrides(
        dadger, calendar, ACVAZEFE, "vazao", "vazao", float
    )
    out_of_horizon.extend(maquinas_out_of_horizon)
    out_of_horizon.extend(potencia_out_of_horizon)
    out_of_horizon.extend(vazao_out_of_horizon)

    return numero_conjuntos, numero_maquinas, potencia, vazao, out_of_horizon


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


def storage_envelope(effective: EffectiveCadastro, code: int) -> tuple[float, float]:
    """Outer per-stage operating range for plant *code*, in hm³.

    ``(min over stages of volume_minimo, max over stages of volume_maximo)``
    — the widest floor/ceiling the plant's dense per-stage series ever
    reaches. For a plant with no override both reduce to the base scalar.
    This is the envelope the entity ``reservoir`` block (ticket-007) declares
    as its default storage bounds; :func:`cobre_bridge.decomp.bounds.
    convert_storage_bounds` emits a per-stage override wherever a stage's
    effective bounds differ from it.
    """
    return (
        min(effective.series(code, "volume_minimo")),
        max(effective.series(code, "volume_maximo")),
    )


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
    out_of_horizon = out_of_horizon + diversion_out_of_horizon + machine_out_of_horizon

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

    effective = EffectiveCadastro(
        base=hidr,
        n_stages=n_stages,
        stage_varying=stage_varying,
        diversions=diversions,
        machine_conjunto_counts=machine_conjunto_counts,
        machine_sets=machine_sets,
    )
    report = CadastroResolutionReport(
        applied=applied, out_of_horizon=tuple(out_of_horizon)
    )
    return effective, report
