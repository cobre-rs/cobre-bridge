"""Read the LIBs-era electrical special-constraint file into a structured model,
and parse its linear-expression DSL into flat terms.

A LIBs-era source-model deck moves its richer electrical special constraints
out of the classic ``dadger`` ``RE``/``LU``/``FU``/``FT``/``FI`` registers into
``lib_restricao-eletrica-especial.csv``, read by idecomp's
``idecomp.libs.restricoes.Restricoes``. This module reads the file's
period-keyed long-form cards — named expressions (``EXPRESSAO-ELETRICA``),
data aliases (``ALIAS-ELETRICO`` + its per-(período,patamar) values), and
restrictions in either the ``RESTRICAO-ELETRICA-FORMULA`` (two-sided
per-(período,patamar) limit) or ``RESTRICAO-ELETRICA-INEQUACAO`` (``lhs op
rhs``, with per-(período,patamar) overrides) shape, plus each restriction's
active horizon, activation-rule binding, and violation treatment — into an
immutable :class:`LibsElectricalModel`.

idecomp returns every ``formula``/``formula_limite``/``limite_inferior``/
``limite_superior`` DSL cell as an opaque string or number; the reader stores
it **verbatim** and does not evaluate it. :func:`parse_linear_expression` turns
one such string into a flat list of :class:`ParsedTerm`, expanding
``EXPRESSAO-ELETRICA`` named-expression references and ``re(R)`` restriction
references (multiplying through the enclosing coefficient) while keeping
``ALIAS-ELETRICO`` references and ``se(cond, X, Y)`` conditionals unresolved
for a later resolver; :func:`classify_term`/:func:`classify_terms` then sort
each parsed term into the source model's A/B/C bucket trichotomy (spec §0).
The activation-rule engine (:class:`ActivationRule`, :class:`Comparison`,
:func:`parse_activation_rule`, :func:`evaluate_rule`) builds on this model
and reuses this same parser for its comparison operands. This module only
reads the period-keyed long-form cards; the short-form ``RE``/``RE-*`` and
the date-indexed ``-HORIZONTE-DATA``/``-FORMULA-DATA-PATAMAR`` variants are
out of scope (kept on the detect+warn fallback).
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from idecomp.libs.restricoes import Restricoes

    from cobre_bridge.decomp.temporal import OperativeStage

#: DECOMP's unbounded-side sentinel on two-sided limit columns is exactly
#: ``-1E+31``/``+1E+31``; anything at or beyond this magnitude means "no bound
#: on this side" and is normalized to ``None``.
_UNBOUNDED_SENTINEL = 1e30


@dataclass(frozen=True)
class ElectricalExpression:
    """A named, reusable linear sub-expression (``EXPRESSAO-ELETRICA``).

    ``formula`` is the raw DSL string verbatim — expanding a reference to this
    expression inside another formula is the linear-expression parser's job,
    not this reader's.
    """

    code: int
    name: str
    formula: str


@dataclass(frozen=True)
class ElectricalAlias:
    """A data alias (``ALIAS-ELETRICO``) and its per-(stage,patamar) values.

    ``values`` is keyed by ``(stage_index, patamar)`` — the 0-based stage
    index (période mapped against the calendar) and the 1-based patamar, or
    ``patamar=None`` meaning "all blocks" (the source row's ``NA`` patamar).
    """

    code: int
    name: str
    values: Mapping[tuple[int, int | None], float]


@dataclass(frozen=True)
class PeriodPatamarOverride:
    """One ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` row.

    Overrides the parent restriction's constant ``lhs op rhs`` for its own
    ``[stage_start, stage_end]`` range and ``patamar`` (``None`` = all
    blocks). All three sides are the raw DSL strings verbatim; the source
    model's precedence rule (période-patamar overrides the constant) is a
    later resolution ticket's job.
    """

    stage_start: int
    stage_end: int
    patamar: int | None
    lhs: str
    operator: str
    rhs: str


@dataclass(frozen=True)
class ViolationTreatment:
    """One ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO`` row.

    Maps to cobre's generic-constraint ``slack{enabled, penalty}`` shape:
    ``enabled`` is ``True`` whenever the restriction carries an explicit
    treatment row (``penalty`` is that row's ``custo_violacao``, or ``None``
    when the row leaves it blank). Resolving the classic BIG-M default for a
    restriction with **no** treatment row at all is a later ticket's job —
    this reader leaves :attr:`ElectricalRestriction.violation` as ``None``
    in that case.
    """

    enabled: bool
    penalty: float | None


@dataclass(frozen=True)
class ElectricalRestriction:
    """One electrical special constraint, unifying the FORMULA and INEQUACAO
    long-form card shapes.

    ``lhs``/``operator``/``rhs`` are the raw DSL strings verbatim (never
    parsed here). A FORMULA-form restriction (``is_formula`` is ``True``) has
    no ``operator``/``rhs`` — its per-(stage,patamar) two-sided limit lives in
    ``limits`` instead, keyed by ``(stage_index, patamar)`` with
    ``patamar=None`` meaning "all blocks" and an absent side normalized from
    DECOMP's ``±1E+31`` sentinel to ``None``. An INEQUACAO-form restriction
    carries the constant ``lhs op rhs`` plus any per-(stage-range,patamar)
    ``overrides``. ``stage_start``/``stage_end`` come from
    ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` (``None`` when absent);
    ``habilita`` is the bound ``REGRA-ATIVACAO`` id — look its text up in
    :attr:`LibsElectricalModel.rules`; ``violation`` is the raw
    ``TRATAMENTO-VIOLACAO`` row, or ``None`` when the restriction carries
    none.
    """

    code: int
    lhs: str
    operator: str | None
    rhs: str | None
    is_formula: bool
    limits: Mapping[tuple[int, int | None], tuple[float | None, float | None]]
    overrides: tuple[PeriodPatamarOverride, ...]
    stage_start: int | None = None
    stage_end: int | None = None
    habilita: int | None = None
    violation: ViolationTreatment | None = None


@dataclass(frozen=True)
class LibsElectricalModel:
    """The full period-keyed long-form electrical-constraint model, read from
    one deck's ``Restricoes``.

    ``expressions``/``aliases`` are keyed by their string identifier
    (``identificador_expressao``/``identificador_alias``); ``restrictions``
    by ``codigo_restricao``; ``rules`` (the raw ``regra_ativacao`` text) by
    ``codigo_regra_ativacao``.
    """

    expressions: Mapping[str, ElectricalExpression]
    aliases: Mapping[str, ElectricalAlias]
    restrictions: Mapping[int, ElectricalRestriction]
    rules: Mapping[int, str]


def _df(source: Restricoes, attr: str) -> pd.DataFrame | None:
    """Read one accessor as a DataFrame, treating an absent card as empty."""
    frame = getattr(source, attr)(df=True)
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _opt_int(value: object) -> int | None:
    """A cell as ``int``, or ``None`` when blank (``NA``)."""
    return None if pd.isna(value) else int(value)  # type: ignore[call-overload]


def _opt_float(value: object) -> float | None:
    """A cell as ``float``, or ``None`` when blank (``NA``)."""
    return None if pd.isna(value) else float(value)  # type: ignore[arg-type]


def _opt_bound(value: object) -> float | None:
    """A two-sided limit cell as ``float``, or ``None`` when blank or at/beyond
    DECOMP's ``±1E+31`` unbounded sentinel."""
    if pd.isna(value):
        return None
    parsed = float(value)  # type: ignore[arg-type]
    return None if abs(parsed) >= _UNBOUNDED_SENTINEL else parsed


def _stage_index(estagio: int, n_stages: int, context: str) -> int:
    """Map a 1-based período to its 0-based stage index against the calendar.

    Mirrors ``network._ia_dense``'s bounds-check shape.
    """
    if not 1 <= estagio <= n_stages:
        raise ValueError(
            f"{context}: período {estagio} outside the calendar (1..{n_stages})"
        )
    return estagio - 1


def _read_expressions(restricoes: Restricoes) -> dict[str, ElectricalExpression]:
    """Read ``EXPRESSAO-ELETRICA`` into ``{name: ElectricalExpression}``."""
    frame = _df(restricoes, "expressao_eletrica")
    if frame is None:
        return {}
    expressions: dict[str, ElectricalExpression] = {}
    for _, row in frame.iterrows():
        name = str(row["identificador_expressao"]).strip()
        expressions[name] = ElectricalExpression(
            code=int(row["codigo_expressao"]),
            name=name,
            formula=str(row["formula"]),
        )
    return expressions


def _read_alias_values(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict[tuple[int, int | None], float]]:
    """Read ``ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR`` into ``{codigo_alias:
    {(stage_index, patamar): valor}}``, expanding each row's stage range."""
    frame = _df(restricoes, "alias_eletrico_valor_periodo_patamar")
    if frame is None:
        return {}
    n_stages = len(calendar)
    values_by_code: dict[int, dict[tuple[int, int | None], float]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_alias"])
        context = f"alias {code} value"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
        patamar = _opt_int(row["patamar"])
        value = float(row["valor"])
        cell = values_by_code.setdefault(code, {})
        for stage in range(start, end + 1):
            cell[(stage, patamar)] = value
    return values_by_code


def _read_aliases(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[str, ElectricalAlias]:
    """Read ``ALIAS-ELETRICO`` joined to its per-(stage,patamar) values."""
    declared = _df(restricoes, "alias_eletrico")
    if declared is None:
        return {}
    values_by_code = _read_alias_values(restricoes, calendar)
    aliases: dict[str, ElectricalAlias] = {}
    for _, row in declared.iterrows():
        code = int(row["codigo_alias"])
        name = str(row["identificador_alias"]).strip()
        aliases[name] = ElectricalAlias(
            code=code,
            name=name,
            values=values_by_code.get(code, {}),
        )
    return aliases


def _read_horizonte(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[int, tuple[int, int]]:
    """Read ``RESTRICAO-ELETRICA-HORIZONTE-PERIODO`` into ``{codigo_restricao:
    (stage_start, stage_end)}``."""
    frame = _df(restricoes, "restricao_eletrica_horizonte_periodo")
    if frame is None:
        return {}
    n_stages = len(calendar)
    horizonte: dict[int, tuple[int, int]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        context = f"restriction {code} horizonte"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
        horizonte[code] = (start, end)
    return horizonte


def _read_formula_limits(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[int, dict[tuple[int, int | None], tuple[float | None, float | None]]]:
    """Read ``RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR`` into
    ``{codigo_restricao: {(stage_index, patamar): (lower, upper)}}``,
    expanding each row's stage range and normalizing the ``±1E+31`` sentinel
    to ``None``."""
    frame = _df(restricoes, "restricao_eletrica_limite_formula_periodo_patamar")
    if frame is None:
        return {}
    n_stages = len(calendar)
    limits: dict[
        int, dict[tuple[int, int | None], tuple[float | None, float | None]]
    ] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        context = f"restriction {code} limits"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
        patamar = _opt_int(row["patamar"])
        bounds = (
            _opt_bound(row["limite_inferior"]),
            _opt_bound(row["limite_superior"]),
        )
        cell = limits.setdefault(code, {})
        for stage in range(start, end + 1):
            cell[(stage, patamar)] = bounds
    return limits


def _read_inequacao_overrides(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[int, tuple[PeriodPatamarOverride, ...]]:
    """Read ``RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR`` into
    ``{codigo_restricao: (PeriodPatamarOverride, ...)}``."""
    frame = _df(restricoes, "restricao_eletrica_inequacao_periodo_patamar")
    if frame is None:
        return {}
    n_stages = len(calendar)
    overrides: dict[int, list[PeriodPatamarOverride]] = {}
    for _, row in frame.iterrows():
        code = int(row["codigo_restricao"])
        context = f"restriction {code} inequacao override"
        start = _stage_index(int(row["estagio_inicio"]), n_stages, context)
        end = _stage_index(int(row["estagio_fim"]), n_stages, context)
        overrides.setdefault(code, []).append(
            PeriodPatamarOverride(
                stage_start=start,
                stage_end=end,
                patamar=_opt_int(row["patamar"]),
                lhs=str(row["formula"]),
                operator=str(row["operador"]),
                rhs=str(row["formula_limite"]),
            )
        )
    return {code: tuple(rows) for code, rows in overrides.items()}


def _read_habilita(restricoes: Restricoes) -> dict[int, int]:
    """Read ``RESTRICAO-ELETRICA-HABILITA`` into ``{codigo_restricao:
    codigo_regra_ativacao}``."""
    frame = _df(restricoes, "restricao_eletrica_habilita")
    if frame is None:
        return {}
    return {
        int(row["codigo_restricao"]): int(row["codigo_regra_ativacao"])
        for _, row in frame.iterrows()
    }


def _read_rules(restricoes: Restricoes) -> dict[int, str]:
    """Read ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` into
    ``{codigo_regra_ativacao: regra_ativacao}`` (the raw boolean-rule text)."""
    frame = _df(restricoes, "restricao_eletrica_regra_ativacao")
    if frame is None:
        return {}
    return {
        int(row["codigo_regra_ativacao"]): str(row["regra_ativacao"])
        for _, row in frame.iterrows()
    }


def _read_violations(restricoes: Restricoes) -> dict[int, ViolationTreatment]:
    """Read ``RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO`` into
    ``{codigo_restricao: ViolationTreatment}``. Absent entirely, or absent for
    a given restriction, is not an error — it simply yields no entry."""
    frame = _df(restricoes, "restricao_eletrica_tratamento_violacao")
    if frame is None:
        return {}
    return {
        int(row["codigo_restricao"]): ViolationTreatment(
            enabled=True,
            penalty=_opt_float(row["custo_violacao"]),
        )
        for _, row in frame.iterrows()
    }


def _read_restrictions(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> dict[int, ElectricalRestriction]:
    """Join the FORMULA and INEQUACAO restriction cards to their per-(period,
    patamar) limit/override rows, horizonte, habilita, and violation rows by
    ``codigo_restricao``."""
    formula = _df(restricoes, "restricao_eletrica_formula")
    inequacao = _df(restricoes, "restricao_eletrica_inequacao")
    if formula is None and inequacao is None:
        return {}

    horizonte = _read_horizonte(restricoes, calendar)
    limits = _read_formula_limits(restricoes, calendar)
    overrides = _read_inequacao_overrides(restricoes, calendar)
    habilita = _read_habilita(restricoes)
    violations = _read_violations(restricoes)

    restrictions: dict[int, ElectricalRestriction] = {}

    if formula is not None:
        for _, row in formula.iterrows():
            code = int(row["codigo_restricao"])
            stage_start, stage_end = horizonte.get(code, (None, None))
            restrictions[code] = ElectricalRestriction(
                code=code,
                lhs=str(row["formula"]),
                operator=None,
                rhs=None,
                is_formula=True,
                limits=limits.get(code, {}),
                overrides=(),
                stage_start=stage_start,
                stage_end=stage_end,
                habilita=habilita.get(code),
                violation=violations.get(code),
            )

    if inequacao is not None:
        for _, row in inequacao.iterrows():
            code = int(row["codigo_restricao"])
            stage_start, stage_end = horizonte.get(code, (None, None))
            restrictions[code] = ElectricalRestriction(
                code=code,
                lhs=str(row["formula"]),
                operator=str(row["operador"]),
                rhs=str(row["formula_limite"]),
                is_formula=False,
                limits={},
                overrides=overrides.get(code, ()),
                stage_start=stage_start,
                stage_end=stage_end,
                habilita=habilita.get(code),
                violation=violations.get(code),
            )

    return restrictions


def read_libs_electrical(
    restricoes: Restricoes,
    calendar: Sequence[OperativeStage],
) -> LibsElectricalModel | None:
    """Read every period-keyed long-form electrical card into a structured model.

    Joins the ``RESTRICAO-ELETRICA-FORMULA``/``-INEQUACAO`` restrictions to
    their per-(period,patamar) limit/override rows, horizonte, habilita, rule,
    and violation rows by ``codigo_restricao``, and maps every période (1-based)
    onto its 0-based stage index against *calendar*.

    Returns ``None`` when the deck carries no electrical long-form card at all
    (no expressions, aliases, restrictions, or rules), so callers can no-op
    cleanly instead of handling an empty-but-present model.
    """
    expressions = _read_expressions(restricoes)
    aliases = _read_aliases(restricoes, calendar)
    restrictions = _read_restrictions(restricoes, calendar)
    rules = _read_rules(restricoes)

    if not expressions and not aliases and not restrictions and not rules:
        return None

    return LibsElectricalModel(
        expressions=expressions,
        aliases=aliases,
        restrictions=restrictions,
        rules=rules,
    )


#: Built-in bare (argument-less) input-data tokens (spec §0/§3). A bare name not
#: declared by ``EXPRESSAO-ELETRICA`` or ``ALIAS-ELETRICO``, and not in this set,
#: is an unrecognized identifier.
_BUILTIN_BARE_TOKENS = frozenset({"demanda_sin", "val_demanda_sin", "carga_ande"})

#: Reserved token for a bare numeric constant; its value lives in
#: :attr:`ParsedTerm.coefficient`.
_CONST_TOKEN = "__const__"

#: Reserved token for an unresolved ``ALIAS-ELETRICO`` reference; the alias's
#: own name lives in :attr:`ParsedTerm.alias_name`.
_ALIAS_TOKEN = "alias"

#: Reserved token for a structural ``se(cond, X, Y)`` conditional.
_SE_TOKEN = "se"

_NUMBER_RE = re.compile(r"\d+\.?\d*")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class ParsedTerm:
    """One atomic, sign-and-coefficient-distributed term of a parsed linear
    expression (see :func:`parse_linear_expression`).

    ``token`` identifies the term's kind:

    - a function-call name (``ger_usih``, ``ener_interc``, ``disp_usih``, …)
      with its integer arguments in ``args``;
    - the reserved :data:`_CONST_TOKEN` (``"__const__"``) for a bare numeric
      constant — its value is in ``coefficient``, ``args`` is empty;
    - the reserved :data:`_ALIAS_TOKEN` (``"alias"``) for an unresolved
      ``ALIAS-ELETRICO`` reference — ``args`` is empty and the alias's own
      name is in ``alias_name``; its numeric value is resolved
      per-(stage,block) later (ticket-008), never here;
    - a built-in bare token (``demanda_sin`` / ``val_demanda_sin`` /
      ``carga_ande``), ``args`` empty;
    - the reserved :data:`_SE_TOKEN` (``"se"``) for a ``se(cond, X, Y)``
      conditional, kept structural: ``condition``, ``branch_true``, and
      ``branch_false`` carry the raw, unparsed sub-expression text (evaluated
      later, never here; ``cond`` is not evaluated by this parser).

    A named-expression (``EXPRESSAO-ELETRICA``) or ``re(R)`` reference never
    survives to a ``ParsedTerm`` — the parser inlines it into its own terms,
    multiplying the enclosing coefficient through.
    """

    coefficient: float
    token: str
    args: tuple[int, ...] = ()
    alias_name: str | None = None
    condition: str | None = None
    branch_true: str | None = None
    branch_false: str | None = None


class _ExpressionParser:
    """Character-position recursive-descent parser for one DSL expression string.

    Threads a running numeric ``scale`` — the product of every sign and scalar
    coefficient enclosing the current position — down to each atomic term, so a
    coefficient distributed over a parenthesized sum (``0.05*(a + b)``) lands on
    every term inside it, and a named-expression/``re(R)`` reference inlines its
    own terms multiplied by the scale in effect where the reference occurred.
    """

    def __init__(
        self, text: str, model: LibsElectricalModel, seen: frozenset[str]
    ) -> None:
        self._text = text
        self._model = model
        self._seen = seen
        self._pos = 0

    def parse(self) -> list[ParsedTerm]:
        terms = self._parse_expr(1.0)
        self._skip_ws()
        if self._pos != len(self._text):
            raise ValueError(
                f"unexpected trailing content {self._text[self._pos :]!r} in "
                f"electrical expression {self._text!r}"
            )
        return terms

    def _skip_ws(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos].isspace():
            self._pos += 1

    def _peek(self) -> str:
        self._skip_ws()
        return self._text[self._pos] if self._pos < len(self._text) else ""

    def _consume(self, char: str, context: str) -> None:
        self._skip_ws()
        if self._pos >= len(self._text) or self._text[self._pos] != char:
            raise ValueError(
                f"expected {char!r} {context} in electrical expression {self._text!r}"
            )
        self._pos += 1

    def _parse_expr(self, scale: float) -> list[ParsedTerm]:
        """``['+'|'-'] term (('+'|'-') term)*`` — a signed sum of terms."""
        terms: list[ParsedTerm] = []
        sign = 1.0
        if self._peek() in ("+", "-"):
            sign = -1.0 if self._text[self._pos] == "-" else 1.0
            self._pos += 1
        terms.extend(self._parse_term(scale * sign))
        while self._peek() in ("+", "-"):
            sign = -1.0 if self._text[self._pos] == "-" else 1.0
            self._pos += 1
            terms.extend(self._parse_term(scale * sign))
        return terms

    def _parse_term(self, scale: float) -> list[ParsedTerm]:
        """``[NUMBER '*'] factor`` — a bare number alone is a constant term."""
        self._skip_ws()
        number = _NUMBER_RE.match(self._text, self._pos)
        if number is None:
            return self._parse_factor(scale)
        probe = number.end()
        while probe < len(self._text) and self._text[probe].isspace():
            probe += 1
        if probe < len(self._text) and self._text[probe] == "*":
            self._pos = probe + 1
            return self._parse_factor(scale * float(number.group()))
        self._pos = number.end()
        return [
            ParsedTerm(coefficient=scale * float(number.group()), token=_CONST_TOKEN)
        ]

    def _parse_factor(self, scale: float) -> list[ParsedTerm]:
        """``'(' expr ')' | IDENT '(' args ')' | IDENT`` — a group or an atom."""
        self._skip_ws()
        if self._pos >= len(self._text):
            raise ValueError(
                "expected a term but reached the end of electrical expression "
                f"{self._text!r}"
            )
        if self._text[self._pos] == "(":
            self._pos += 1
            terms = self._parse_expr(scale)
            self._consume(")", "to close '('")
            return terms
        ident = _IDENT_RE.match(self._text, self._pos)
        if ident is None:
            raise ValueError(
                f"malformed token at {self._text[self._pos :]!r} in electrical "
                f"expression {self._text!r}"
            )
        name = ident.group()
        self._pos = ident.end()
        if self._peek() == "(":
            return self._parse_function_call(name, scale)
        return self._parse_bare_name(name, scale)

    def _parse_function_call(self, name: str, scale: float) -> list[ParsedTerm]:
        if name == _SE_TOKEN:
            return self._parse_se(scale)
        if name == "re":
            return self._parse_re(scale)
        args = self._parse_int_args(name)
        return [ParsedTerm(coefficient=scale, token=name, args=args)]

    def _parse_int_args(self, fn_name: str) -> tuple[int, ...]:
        self._consume("(", f"after {fn_name!r}")
        args: list[int] = []
        if self._peek() == ")":
            self._pos += 1
            return tuple(args)
        while True:
            self._skip_ws()
            number = _NUMBER_RE.match(self._text, self._pos)
            if number is None or "." in number.group():
                raise ValueError(
                    f"expected an integer argument for {fn_name!r}(...) at "
                    f"{self._text[self._pos :]!r} in electrical expression "
                    f"{self._text!r}"
                )
            args.append(int(number.group()))
            self._pos = number.end()
            if self._peek() == ",":
                self._pos += 1
                continue
            break
        self._consume(")", f"to close {fn_name!r}(...)")
        return tuple(args)

    def _parse_re(self, scale: float) -> list[ParsedTerm]:
        """``re(R)`` — a reference to restriction ``R``'s formula, inlined."""
        args = self._parse_int_args("re")
        if len(args) != 1:
            raise ValueError(
                f"re(...) expects exactly one restriction id, got {args!r} in "
                f"electrical expression {self._text!r}"
            )
        code = args[0]
        restriction = self._model.restrictions.get(code)
        if restriction is None:
            raise ValueError(
                f"re({code}) references unknown restriction {code} in "
                f"electrical expression {self._text!r}"
            )
        return self._expand_reference(f"re({code})", restriction.lhs, scale)

    def _parse_se(self, scale: float) -> list[ParsedTerm]:
        """``se(cond, X, Y)`` — one structural term; ``cond`` is never evaluated
        here (ticket-008), so its raw text (and both branches') is captured by a
        depth-aware scan for the top-level commas rather than by tokenizing it."""
        self._consume("(", "after 'se'")
        args_start = self._pos
        comma_positions: list[int] = []
        depth = 0
        pos = self._pos
        while True:
            if pos >= len(self._text):
                raise ValueError(
                    "unbalanced parentheses in se(...) in electrical expression "
                    f"{self._text!r}"
                )
            char = self._text[pos]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif char == "," and depth == 0:
                comma_positions.append(pos)
            pos += 1
        close_pos = pos
        if len(comma_positions) != 2:
            raise ValueError(
                "se(...) expects exactly 3 arguments (condition, true-branch, "
                f"false-branch) in electrical expression {self._text!r}"
            )
        first_comma, second_comma = comma_positions
        condition = self._text[args_start:first_comma].strip()
        branch_true = self._text[first_comma + 1 : second_comma].strip()
        branch_false = self._text[second_comma + 1 : close_pos].strip()
        self._pos = close_pos + 1
        return [
            ParsedTerm(
                coefficient=scale,
                token=_SE_TOKEN,
                condition=condition,
                branch_true=branch_true,
                branch_false=branch_false,
            )
        ]

    def _parse_bare_name(self, name: str, scale: float) -> list[ParsedTerm]:
        """Disambiguate a bare name per spec §3: a named expression, else a data
        alias, else a built-in bare token — else an unrecognized identifier."""
        if name in self._model.expressions:
            return self._expand_reference(
                name, self._model.expressions[name].formula, scale
            )
        if name in self._model.aliases:
            return [ParsedTerm(coefficient=scale, token=_ALIAS_TOKEN, alias_name=name)]
        if name in _BUILTIN_BARE_TOKENS:
            return [ParsedTerm(coefficient=scale, token=name)]
        raise ValueError(
            f"unrecognized identifier {name!r} in electrical expression "
            f"{self._text!r} (not a declared named expression, data alias, or "
            "built-in bare token)"
        )

    def _expand_reference(
        self, key: str, formula: str, scale: float
    ) -> list[ParsedTerm]:
        """Recursively parse *formula* (a named expression's or a restriction's
        LHS), multiplying every resulting term's coefficient by *scale*, and
        guard the recursion against a cycle through *key*."""
        if key in self._seen:
            cycle = " -> ".join(sorted({*self._seen, key}))
            raise ValueError(f"cyclic electrical expression reference: {cycle}")
        sub_terms = parse_linear_expression(
            formula, self._model, _seen=self._seen | {key}
        )
        return [
            ParsedTerm(
                coefficient=scale * term.coefficient,
                token=term.token,
                args=term.args,
                alias_name=term.alias_name,
                condition=term.condition,
                branch_true=term.branch_true,
                branch_false=term.branch_false,
            )
            for term in sub_terms
        ]


def parse_linear_expression(
    text: str,
    model: LibsElectricalModel,
    *,
    _seen: frozenset[str] = frozenset(),
) -> list[ParsedTerm]:
    """Parse one electrical linear-expression DSL string into flat terms.

    Distributes every sign and scalar coefficient onto each atomic term
    (including across a parenthesized sum) and recursively inlines
    ``EXPRESSAO-ELETRICA`` named-expression references and ``re(R)``
    restriction references, multiplying through the enclosing coefficient —
    ``_seen`` (the names/``re(R)`` keys expanded on the current recursion path)
    guards that inlining against a cycle, raising ``ValueError`` naming it. An
    ``ALIAS-ELETRICO`` reference is kept as a single unresolved
    :data:`_ALIAS_TOKEN` term (its value is resolved per-(stage,block) later,
    never here); a ``se(cond, X, Y)`` conditional is kept as a single
    structural term with its condition and both branches unevaluated.

    This is the source model's linear-expression grammar, not cobre's: the
    parser never normalizes the relation (no RHS→LHS move, no same-token
    merge) — it preserves the author's side and term multiplicity verbatim;
    cobre owns relational normalization.

    Raises
    ------
    ValueError
        On a cyclic named-expression/``re(R)`` reference, or a malformed token
        (unbalanced parentheses, a non-numeric coefficient/argument, an
        unrecognized identifier, or unexpected trailing content).
    """
    return _ExpressionParser(text, model, _seen).parse()


class Bucket(Enum):
    """Which of the source model's spec-§0 A/B/C trichotomy a parsed term
    belongs to.

    - **A** — a cobre decision term: stays a structural LP term.
    - **B** — an input-data term: the bridge evaluates it to a number and
      folds it into the bound.
    - **C** — ``disp_usih``: cobre has no available-power ``VariableRef``, so
      the bridge resolves it to a per-stage constant that rewrites the
      constraint form instead of emitting it as a structural LP term.
    """

    A = "A"
    B = "B"
    C = "C"


#: Bucket A (spec §0) — cobre decision tokens that stay structural LP terms.
#: ``re(X)`` is not listed here: the parser already inlines a restriction
#: reference into its own terms (multiplying through the enclosing
#: coefficient), so no :class:`ParsedTerm` ever carries a bare ``"re"`` token.
_BUCKET_A_TOKENS = frozenset(
    {"ger_usih", "ger_usit", "ger_pee", "ger_conjh", "ener_interc", "ener_comerc"}
)

#: Bucket B (spec §0) — input-data tokens the bridge evaluates to a number and
#: folds into the bound: the built-in demand/load tokens, plus the parser's
#: three reserved structural tokens (a bare numeric constant, an unresolved
#: ``ALIAS-ELETRICO`` reference, and a ``se(cond, X, Y)`` conditional).
_BUCKET_B_TOKENS = frozenset(
    {
        "demanda",
        "val_demanda",
        "demanda_sin",
        "val_demanda_sin",
        "carga_ande",
        _CONST_TOKEN,
        _ALIAS_TOKEN,
        _SE_TOKEN,
    }
)

#: Bucket C (spec §0) — ``disp_usih`` is the sole member (see :class:`Bucket`).
_BUCKET_C_TOKENS = frozenset({"disp_usih"})

#: The closed token -> :class:`Bucket` map (spec §0), the sole source of truth
#: for :func:`classify_term`. Mirrors
#: ``constraint_registers._BOUNDS_AXIS``'s module-level membership-map shape.
_TOKEN_BUCKET: dict[str, Bucket] = {
    **dict.fromkeys(_BUCKET_A_TOKENS, Bucket.A),
    **dict.fromkeys(_BUCKET_B_TOKENS, Bucket.B),
    **dict.fromkeys(_BUCKET_C_TOKENS, Bucket.C),
}


def classify_term(term: ParsedTerm) -> Bucket:
    """Classify one parsed term into its spec-§0 A/B/C bucket.

    Decides purely from ``term.token`` against the closed set in
    :data:`_TOKEN_BUCKET` — never a model lookup, since the parser already
    resolved every name to a concrete token (an alias to :data:`_ALIAS_TOKEN`,
    a bare numeric constant to :data:`_CONST_TOKEN`, a conditional to
    :data:`_SE_TOKEN`, a named-expression/``re(R)`` reference inlined away).

    Raises
    ------
    ValueError
        When ``term.token`` is not a recognized token — fail loud rather than
        silently drop or mis-bucket a term, which would corrupt either the LP
        structure (bucket A/C) or the bound (bucket B).
    """
    bucket = _TOKEN_BUCKET.get(term.token)
    if bucket is None:
        recognized = ", ".join(sorted(_TOKEN_BUCKET))
        raise ValueError(
            f"unrecognized electrical-expression token {term.token!r}; "
            f"recognized tokens are: {recognized}"
        )
    return bucket


def classify_terms(terms: Sequence[ParsedTerm]) -> dict[Bucket, list[ParsedTerm]]:
    """Partition a term sequence into its three spec-§0 buckets.

    Every :class:`Bucket` key is present (an empty list when no term
    classified into it), and each bucket's list preserves the input order —
    callers get a stable, total partition.

    Raises
    ------
    ValueError
        Propagated from :func:`classify_term` on the first unrecognized token.
    """
    partition: dict[Bucket, list[ParsedTerm]] = {bucket: [] for bucket in Bucket}
    for term in terms:
        partition[classify_term(term)].append(term)
    return partition


#: The activation-rule grammar's six comparison operators (spec §4a),
#: longest-match: the two-character operators must be tried before their
#: single-character prefix (``<=``/``>=`` before ``<``/``>``).
_COMPARISON_TWO_CHAR_OPS = frozenset({"<=", ">=", "==", "!="})
_COMPARISON_ONE_CHAR_OPS = frozenset({"<", ">"})
#: Every character that can start or continue a comparison operator — used to
#: recognize a malformed operator (e.g. ``<>``) fast, without first parsing
#: an operand.
_COMPARISON_OP_CHARS = frozenset("<>=!")

#: The closed six-operator -> predicate map that :func:`evaluate_rule` applies
#: to each :class:`Comparison`'s two evaluated sides.
_COMPARISON_FUNCS: Mapping[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


@dataclass(frozen=True)
class Comparison:
    """One parsed ``operand <op> operand`` comparison from an activation
    rule's ``&``-conjunction.

    ``left``/``right`` are the two operand sides, each a flat list of
    :class:`ParsedTerm` from :func:`parse_linear_expression`; every term on
    either side classifies as :attr:`Bucket.B` (:func:`parse_activation_rule`
    enforces this — an activation rule gates on input data, never a cobre
    decision term). ``op`` is one of the six recognized comparison operators
    (spec §4a), verbatim.
    """

    left: list[ParsedTerm]
    op: str
    right: list[ParsedTerm]


@dataclass(frozen=True)
class ActivationRule:
    """A parsed ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` boolean rule: an AND
    (the grammar's top-level ``&``) of :class:`Comparison`\\ s.

    An empty ``comparisons`` tuple models "always active" — see
    :func:`is_always_active` — for a restriction with no ``HABILITA``
    binding at all, which never has rule text to parse.
    """

    comparisons: tuple[Comparison, ...]


#: Resolves one bucket-B operand term to its numeric value for one
#: (stage, block) cell — a built-in bare token (``demanda_sin``), a
#: parameterized one (``demanda(3)``), an unresolved ``ALIAS-ELETRICO``
#: reference (``term.alias_name``), or a ``se(cond, X, Y)`` conditional.
#:
#: A bare numeric constant (``term.token == "__const__"``) is the one
#: exception: its value already lives in the term's own ``coefficient``
#: (see :class:`ParsedTerm`), so a resolver **must** return ``1.0`` for it —
#: :func:`evaluate_rule`'s ``coefficient * context(term)`` product would
#: otherwise double-apply the constant.
#:
#: This is the source model's per-(stage,block) demand/alias resolver's
#: interface, defined here so this engine stays pure and testable; the real
#: resolver driving it with the source model's demand/alias data over every
#: (stage, block) cell is a later ticket's job (ticket-008). Tests use a
#: trivial dict-backed double implementing this same interface.
type DataContext = Callable[[ParsedTerm], float]


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* on every top-level (parenthesis-depth-0) occurrence of
    *sep*.

    The observed activation-rule grammar never lets a parenthesized group
    span across a top-level ``&`` — this only affects which ``&`` occurrences
    are split points (one inside a parenthesized sub-expression is not); it
    is *not* violated by a normal nested ``&`` inside parentheses, which
    simply is not a split point.

    Raises
    ------
    ValueError
        When *text*'s parentheses are unbalanced (a stray closing paren, or
        one left open at the end) — this would mean some paren group is not
        self-contained the way the grammar requires.
    """
    parts: list[str] = []
    depth = 0
    start = 0
    for pos, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced parentheses in activation rule {text!r}")
        elif char == sep and depth == 0:
            parts.append(text[start:pos])
            start = pos + 1
    if depth != 0:
        raise ValueError(f"unbalanced parentheses in activation rule {text!r}")
    parts.append(text[start:])
    return parts


def _split_comparison(conjunct: str) -> tuple[str, str, str]:
    """Split one activation-rule conjunct into ``(left_text, op, right_text)``
    at its top-level comparison operator, longest-match (the two-character
    operators ``<=``/``>=``/``==``/``!=`` win over their single-character
    prefix ``<``/``>``).

    Scans outside of any parenthesized group (an operand may itself contain
    parentheses, e.g. ``demanda(3)+demanda(4)``) for the first character in
    :data:`_COMPARISON_OP_CHARS`.

    Raises
    ------
    ValueError
        When the conjunct carries no comparison operator at all, or an
        operator-like character sequence that is not one of the six
        recognized operators (e.g. ``<>``, a lone ``=``) — both name the
        conjunct verbatim.
    """
    depth = 0
    pos = 0
    length = len(conjunct)
    while pos < length:
        char = conjunct[pos]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif depth == 0 and char in _COMPARISON_OP_CHARS:
            two_char = conjunct[pos : pos + 2]
            if two_char in _COMPARISON_TWO_CHAR_OPS:
                return conjunct[:pos], two_char, conjunct[pos + 2 :]
            next_char = conjunct[pos + 1 : pos + 2]
            if next_char in _COMPARISON_OP_CHARS:
                raise ValueError(
                    f"unrecognized comparison operator {two_char!r} in "
                    f"activation-rule conjunct {conjunct!r}"
                )
            if char not in _COMPARISON_ONE_CHAR_OPS:
                raise ValueError(
                    f"unrecognized comparison operator {char!r} in "
                    f"activation-rule conjunct {conjunct!r}"
                )
            return conjunct[:pos], char, conjunct[pos + 1 :]
        pos += 1
    raise ValueError(
        f"no comparison operator found in activation-rule conjunct {conjunct!r} "
        "(recognized operators: <, <=, >, >=, ==, !=)"
    )


def _require_bucket_b(terms: Sequence[ParsedTerm], conjunct: str) -> None:
    """Raise ``ValueError`` naming the offending token when any *terms* is
    not bucket B — an activation-rule operand is always input data (spec
    §4a), never a cobre decision term (bucket A) or the available-power
    bucket-C term."""
    for term in terms:
        if classify_term(term) is not Bucket.B:
            raise ValueError(
                f"activation-rule operand {term.token!r} in conjunct "
                f"{conjunct!r} is not input data (bucket B); an activation "
                "rule must gate on input data, never a decision term"
            )


def parse_activation_rule(text: str, model: LibsElectricalModel) -> ActivationRule:
    """Parse one ``RESTRICAO-ELETRICA-REGRA-ATIVACAO`` boolean-rule string
    into an :class:`ActivationRule`.

    Splits *text* on its top-level ``&`` conjunctions (:func:`_split_top_level`),
    splits each conjunct on its comparison operator, longest-match
    (:func:`_split_comparison`), and parses each side with
    :func:`parse_linear_expression` — the comparison operands are the same
    input-data linear expressions that function already parses. Every
    resulting operand term must classify as :attr:`Bucket.B`
    (:func:`classify_term`); an activation rule gates on input data, never a
    cobre decision term.

    This is the closed six-operator + ``&`` grammar verified against every
    rule in the source model's decks (spec §4a) — it has no ``|``
    (disjunction) and no ``se`` literal at the rule-text level, so neither is
    supported here; a deck that needs either is a new ticket, not a silent
    extension.

    Raises
    ------
    ValueError
        Propagated from :func:`_split_top_level` on unbalanced parentheses,
        from :func:`_split_comparison` on a missing/unrecognized operator, or
        raised here when an operand term does not classify as bucket B
        (naming the offending token).
    """
    comparisons = tuple(
        _parse_comparison(conjunct, model) for conjunct in _split_top_level(text, "&")
    )
    return ActivationRule(comparisons=comparisons)


def _parse_comparison(conjunct: str, model: LibsElectricalModel) -> Comparison:
    """Parse one ``&``-delimited conjunct into a :class:`Comparison`,
    enforcing the bucket-B operand guard on both sides."""
    left_text, op, right_text = _split_comparison(conjunct)
    left = parse_linear_expression(left_text, model)
    right = parse_linear_expression(right_text, model)
    _require_bucket_b(left, conjunct)
    _require_bucket_b(right, conjunct)
    return Comparison(left=left, op=op, right=right)


def _evaluate_side(terms: Sequence[ParsedTerm], context: DataContext) -> float:
    """Sum one :class:`Comparison` side's ``coefficient * context(term)``
    over its operand terms."""
    return sum(term.coefficient * context(term) for term in terms)


def evaluate_rule(rule: ActivationRule, context: DataContext) -> bool:
    """Evaluate one parsed activation rule against a per-(stage,block)
    :data:`DataContext`.

    Returns the AND of every :class:`Comparison`: each side sums to
    ``Σ coefficient * context(term)`` over its operand terms
    (:func:`_evaluate_side`), and the six comparison operators map onto the
    matching predicate (:data:`_COMPARISON_FUNCS`). An empty
    ``rule.comparisons`` — :func:`is_always_active` — evaluates to ``True``,
    the vacuous AND.

    Pure and deterministic given *context*: this engine never resolves a
    bucket-B token's value itself, so the same rule can be evaluated against
    any (stage, block) cell just by supplying a different *context*.
    """
    return all(
        _COMPARISON_FUNCS[comparison.op](
            _evaluate_side(comparison.left, context),
            _evaluate_side(comparison.right, context),
        )
        for comparison in rule.comparisons
    )


def is_always_active(rule: ActivationRule) -> bool:
    """Whether *rule* is the vacuous always-active rule.

    A restriction with no ``HABILITA`` binding has no rule text to parse at
    all — model that as the empty-conjunction :class:`ActivationRule`
    (``comparisons=()``), which is always active, subject only to the
    restriction's own ``horizonte``.
    """
    return len(rule.comparisons) == 0
