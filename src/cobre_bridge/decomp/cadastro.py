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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    from idecomp.decomp.modelos.dadger import ACVOLMAX, ACVOLMIN

    _SCALAR_AC_SPECS: tuple[_ScalarAcSpec, ...] = (
        _ScalarAcSpec(ACVOLMIN, "volume", "volume_minimo"),
        _ScalarAcSpec(ACVOLMAX, "volume", "volume_maximo"),
    )

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
                ano = row["ano"]
                resolved_ano = (
                    calendar[0].start_date.year
                    if ano is None or pd.isna(ano)
                    else int(ano)
                )
                month = _parse_month(mes)
                assert month is not None  # eff is None only for a non-blank mes
                out_of_horizon.append(
                    OutOfHorizon(code, spec.param, month, resolved_ano)
                )
                continue
            records.setdefault((code, spec.param), []).append((eff, value))
    return records, out_of_horizon


def _forward_fill_series(
    base_value: float,
    records: Sequence[tuple[int, float]],
    n_stages: int,
) -> list[float]:
    """Densify a sparse set of effective-stage overrides into a dense series.

    *records* is a possibly-unordered, possibly-empty sequence of
    ``(effective_stage, value)`` pairs. The value at stage ``s`` is the value
    of the last record whose ``effective_stage <= s``, else *base_value*. A
    record at stage ``0`` therefore overwrites the base for every stage
    (permanent semantics); later records supersede earlier ones. *records*
    is not mutated.
    """
    ordered = sorted(records, key=lambda record: record[0])
    series: list[float] = []
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

    Raises
    ------
    ValueError
        If an override references a plant *code* absent from ``hidr.index``
        — the registry has no cadastro row to override.
    """
    records, out_of_horizon = _read_scalar_overrides(dadger, calendar)

    n_stages = len(calendar)
    stage_varying: dict[tuple[int, str], tuple[float, ...]] = {}
    applied: dict[str, int] = {}
    for (code, param), overrides in records.items():
        if code not in hidr.index:
            raise ValueError(
                f"AC override references plant code {code}, which is not in"
                " the cadastro registry"
            )
        base_value = float(hidr.loc[code, param])
        stage_varying[(code, param)] = tuple(
            _forward_fill_series(base_value, overrides, n_stages)
        )
        applied[param] = applied.get(param, 0) + 1

    effective = EffectiveCadastro(
        base=hidr, n_stages=n_stages, stage_varying=stage_varying
    )
    report = CadastroResolutionReport(
        applied=applied, out_of_horizon=tuple(out_of_horizon)
    )
    return effective, report
