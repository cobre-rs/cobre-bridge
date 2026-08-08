"""Foundation for the source model's generic-constraint emitter.

The special-constraint reader (``constraint_registers.py``) splits every
special constraint into ones that lower to a plain entity bound
(``single_term_bounds.py``) and everything else — multi-term participation,
an unbounded variable (spillage, diversion, pumping), a non-unit
coefficient, or a whole-cascade energy sum — which needs cobre's
**generic** constraint wire format instead: a flat ``expression`` string, a
``sense``/``slack{enabled, penalty}`` pair, and a companion per-
``(constraint_id, stage_id, block_id)`` bounds table.

This module stands up that shared scaffolding: the per-term expression-token
dispatch (``_variable_token``), the ``FI``-interchange line resolver
(``build_fi_line_map``/``resolve_fi_term``), the coefficient-string formatter
(``_format_expression``), the ``GenericConstraintResult`` shape and its
bounds schema, the ``BIG_M`` slack-penalty helper, and the
``_GenericBuilder`` two-sided assembler, plus the multi-term ``RE`` emitter
(``emit_re_generics``) and the ``RHQ``/``RHV`` emitter (
``emit_rhq_rhv_generics``) built on top of it. This module mirrors — but
never imports — the sibling ``converters/constraints.py`` emitter's
``GenericConstraintResult``/``_parse_formula`` patterns.

GNL pre-processing (feature spec section 2.1/G4 -- abating commanded
generation, aborting on uncommanded) is out of scope here: ``emit_re_generics``
simply skips a constraint whose ``thermal_generation`` term names a thermal
outside ``id_map.thermal_codes`` (an anticipation-track/GNL thermal), with a
``WARNING``, deferring coordination with the GNL-ring track.

``RHV``'s volume tipos (``VDEF``/``VDES``/``VBOM``) are deferred rather than
lowered: their hm³→flow conversion needs a per-stage coefficient (block
hours vary per stage) that a stage-invariant cobre expression cannot carry,
so ``emit_rhq_rhv_generics`` detects one and skips the constraint with a
``WARNING`` instead of guessing a coefficient (see the E5 scalar-parameter
plumbing).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, NamedTuple

import pyarrow as pa

from cobre_bridge.decomp.cadastro import effective_storage_range
from cobre_bridge.decomp.constraint_registers import StageBounds
from cobre_bridge.diagnostics import Diagnostic, Severity, emit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
        ConstraintTerm,
    )
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage


class GenericConstraintResult(NamedTuple):
    """Result of a generic-constraint emitter.

    ``constraints`` is the list of constraint dicts (``{"id", "name",
    "description", "expression", "sense", "slack"}``); ``bounds`` is the
    per-``(constraint_id, stage_id, block_id)`` bounds table honouring
    :data:`_GENERIC_BOUNDS_SCHEMA`.
    """

    constraints: list[dict]
    bounds: pa.Table


#: Schema for the generic-constraint bounds table. ``block_id`` is nullable:
#: ``None`` means "all blocks" (a stage-level constraint, or a per-block one
#: whose bound applies uniformly).
_GENERIC_BOUNDS_SCHEMA = pa.schema(
    [
        pa.field("constraint_id", pa.int32(), nullable=False),
        pa.field("stage_id", pa.int32(), nullable=False),
        pa.field("block_id", pa.int32(), nullable=True),
        pa.field("bound", pa.float64(), nullable=False),
    ]
)

#: The unbounded sentinel, mirroring ``decomp/bounds_accumulator._UNBOUNDED``:
#: a bound whose magnitude is at or past this value carries no real limit.
_UNBOUNDED = 1e21


def _is_bounded(value: float | None) -> bool:
    """True iff *value* is a real (non-``None``, non-sentinel) bound."""
    return value is not None and abs(value) < _UNBOUNDED


#: The source model's ``num_max_iteracoes``-style deficit multiplier: the
#: slack penalty on a generic constraint is this many times the worst bus
#: deficit cost, so violating the constraint is always far more expensive
#: than shedding load but never numerically unbounded.
_BIG_M_DEFICIT_FACTOR = 10.0


def big_m_penalty(max_deficit: float) -> float:
    """The slack penalty for a generic constraint: ``10 x`` the worst deficit cost.

    The pipeline computes ``max_deficit`` from the deck's per-bus deficit
    costs and passes the result here — this module never reads the deck
    itself.
    """
    return _BIG_M_DEFICIT_FACTOR * max_deficit


def _hydro_generation_token(hydro_id: int, bus_id: int | None = None) -> str:
    """The cobre ``hydro_generation`` token, optionally split by frequency bus.

    ``bus_id`` selects the v0.14 frequency-split generation selector
    (``hydro_generation(id, bus=bus_id)``); omitted, it is the plant's whole
    generation (``hydro_generation(id)``). Frequency-split resolution itself
    is deferred to the multi-term RE emitter (ticket-015).
    """
    if bus_id is not None:
        return f"hydro_generation({hydro_id}, bus={bus_id})"
    return f"hydro_generation({hydro_id})"


#: Flow/volume/generation variables that map straight through
#: ``id_map.hydro_id`` to a single-argument cobre hydro token. Kept explicit
#: and symmetric with the reader's ``constraint_registers._BOUNDS_AXIS`` and
#: ``single_term_bounds._HQ_AXIS_BY_VARIABLE`` variable taxonomies.
_HYDRO_TOKEN_FN: dict[str, str] = {
    "QDEF": "hydro_outflow",
    "QTUR": "hydro_turbined",
    "QVER": "hydro_spillage",
    "QDES": "hydro_diversion",
    "VARM": "hydro_storage",
}

#: Variables ``_variable_token`` refuses to resolve itself, mapped to the
#: mechanism (and owning ticket) that resolves them instead.
_DEFERRED_VARIABLES: dict[str, str] = {
    "interchange": "resolve_fi_term (ticket-014, FI line resolution)",
    "VDEF": "the hours-weighted flow-domain path (ticket-016)",
    "VDES": "the hours-weighted flow-domain path (ticket-016)",
    "VBOM": "the hours-weighted flow-domain path (ticket-016)",
}


def _variable_token(
    term: ConstraintTerm,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
) -> str:
    """The cobre expression token for one constraint term.

    Dispatches on ``term.variable``: ``generation``/``thermal_generation``
    map to the whole-plant generation tokens; ``QDEF``/``QTUR``/``QVER``/
    ``QDES``/``VARM`` map through :data:`_HYDRO_TOKEN_FN` and
    ``id_map.hydro_id``; ``QBOM`` maps through *pumping_station_ids*.

    Raises
    ------
    ValueError
        For ``interchange`` (resolved by ticket-014's ``resolve_fi_term``,
        which needs the line map), the volume tipos ``VDEF``/``VDES``/
        ``VBOM`` (resolved by ticket-016's hours-weighted flow-domain path),
        or any other unrecognised variable — fail-loud, never a silent skip
        that would build a malformed expression.
    KeyError
        Propagated from ``id_map.hydro_id``/``id_map.thermal_id`` for an
        unknown plant/thermal code, or from *pumping_station_ids* for an
        unknown ``QBOM`` code.
    """
    variable = term.variable
    if variable == "generation":
        return _hydro_generation_token(id_map.hydro_id(term.code))
    if variable == "thermal_generation":
        return f"thermal_generation({id_map.thermal_id(term.code)})"
    if variable == "QBOM":
        return f"pumping_flow({pumping_station_ids[term.code]})"
    token_fn = _HYDRO_TOKEN_FN.get(variable)
    if token_fn is not None:
        return f"{token_fn}({id_map.hydro_id(term.code)})"
    owner = _DEFERRED_VARIABLES.get(variable)
    if owner is not None:
        raise ValueError(
            f"_variable_token: {variable!r} is resolved by {owner}, not this dispatcher"
        )
    raise ValueError(f"_variable_token: unknown variable {variable!r}")


def _line_int_field(line: Mapping[str, object], field: str) -> int:
    """Extract *field* from a converted line dict, validating it is an ``int``.

    ``convert_lines``'s own ``lines.json`` list hands back untyped
    ``dict[str, object]`` entries; this narrows the three id fields with an
    explicit runtime check (mirroring ``fcf.mapper._slot_int``) rather than a
    bare ``int()`` cast, so a malformed line entry fails loudly here instead
    of silently corrupting the map.
    """
    value = line[field]
    if not isinstance(value, int):
        raise TypeError(
            f"line field {field!r} is {type(value).__name__}, not int: {value!r}"
        )
    return value


def build_fi_line_map(
    lines: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], int]:
    """Map each converted line's declared ``(source, target)`` bus pair to its id.

    ``lines`` is the ``convert_lines`` ``lines.json`` ``lines`` list — one
    entry per ``IA`` line, in its ``IA``-declared orientation (**not**
    canonicalized). Keying on that orientation, rather than a
    sorted/canonical pair, is what lets :func:`resolve_fi_term` tell an
    ``FI`` term's direct direction from its reverse via a two-probe lookup.
    """
    return {
        (
            _line_int_field(line, "source_bus_id"),
            _line_int_field(line, "target_bus_id"),
        ): _line_int_field(line, "id")
        for line in lines
    }


def _emit_fi_no_line(term: ConstraintTerm, *, reason: str) -> None:
    """Emit the shared ``decomp-fi-no-line`` WARNING for an unresolved FI term."""
    emit(
        Diagnostic(
            code="decomp-fi-no-line",
            severity=Severity.WARNING,
            category="Special constraints",
            title="FI interchange term has no matching line",
            summary=(
                f"RE constraint FI interchange term ({term.submarket_de!r} -> "
                f"{term.submarket_para!r}) could not be resolved to a cobre "
                f"line: {reason}."
            ),
            remediation=(
                "Check that the pair is declared as an IA line; the "
                "constraint carrying this FI term is skipped."
            ),
        )
    )


def resolve_fi_term(
    term: ConstraintTerm,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
) -> str | None:
    """Resolve one ``FI`` interchange term to a directional cobre line token.

    cobre has no submarket-pair variable, only per-line directional
    ``line_direct(id)``/``line_reverse(id)`` variables, so — mirroring the
    source model precedent's ``ener_interc`` resolution
    (``converters/constraints.py::_parse_formula``) — the term's direction is
    encoded by *which* directional variable is chosen, never by negating the
    coefficient (a negated coefficient would dilute the bound when the LP
    routes the non-canonical direction).

    Resolves both submarket names to bus ids via ``id_map.bus_id_by_name``
    (the reconciled ``SB`` name -> id map, which also handles the fictitious
    ``IV`` transhipment submarket) and probes *line_map* in both
    orientations, without canonicalizing the pair: ``(de, para)`` matching a
    line's declared ``source_bus_id``/``target_bus_id`` returns
    ``"line_direct(id)"`` (the term's orientation matches the line's); the
    reverse match returns ``"line_reverse(id)"``.

    Returns ``None`` — after emitting one ``Severity.WARNING``
    ``"decomp-fi-no-line"`` diagnostic — when either submarket name is
    unknown (a caught ``KeyError``) or no line joins the pair in either
    orientation. Raises ``ValueError`` if *term* is missing a submarket name
    (a caller contract violation: a genuine ``FI`` term, as produced by the
    reader, always carries both). Otherwise never raises: the caller drops
    the whole constraint on a ``None`` rather than build a partial
    expression.
    """
    if term.submarket_de is None or term.submarket_para is None:
        raise ValueError(
            "resolve_fi_term: an interchange term must carry "
            f"submarket_de/submarket_para, got {term!r}"
        )
    try:
        de = id_map.bus_id_by_name(term.submarket_de)
        para = id_map.bus_id_by_name(term.submarket_para)
    except KeyError as exc:
        reason = str(exc.args[0]) if exc.args else str(exc)
        _emit_fi_no_line(term, reason=reason)
        return None
    if (de, para) in line_map:
        return f"line_direct({line_map[(de, para)]})"
    if (para, de) in line_map:
        return f"line_reverse({line_map[(para, de)]})"
    _emit_fi_no_line(
        term,
        reason=f"no IA line joins {term.submarket_de!r} and {term.submarket_para!r}",
    )
    return None


def _format_expression(terms: Sequence[tuple[float, str]]) -> str | None:
    """Join ``(coefficient, token)`` pairs into a cobre expression string.

    Mirrors ``converters/constraints.py::_parse_formula``'s coefficient-
    formatting rules: a unit (``|coeff| == 1.0``) term is the bare token
    (or ``- token`` when negative); a fractional term is ``{abs_coeff} *
    {token}`` (or ``- {abs_coeff} * {token}`` when negative). The first term
    omits the leading ``+``; every later term uses ``+``/``-`` as a binary
    operator. Returns ``None`` for an empty term list.
    """
    if not terms:
        return None
    parts: list[str] = []
    for i, (coeff, token) in enumerate(terms):
        abs_coeff = abs(coeff)
        is_negative = coeff < 0.0
        body = token if abs_coeff == 1.0 else f"{abs_coeff} * {token}"
        if i == 0:
            parts.append(f"- {body}" if is_negative else body)
        else:
            parts.append(f"- {body}" if is_negative else f"+ {body}")
    return " ".join(parts)


class _GenericBuilder:
    """Assembles two-sided generic constraints sharing one 0-based id space.

    ``start_id`` is the first id this builder assigns; later emitters thread
    a running ``start_id`` across every builder so ids never collide (E7).
    Each :meth:`add_two_sided` call may append zero, one, or two constraint
    dicts plus their companion bounds rows; :meth:`result` packages
    everything added so far into a :class:`GenericConstraintResult`.
    """

    def __init__(self, start_id: int) -> None:
        self._start_id = start_id
        self._constraints: list[dict] = []
        self._bound_cids: list[int] = []
        self._bound_stages: list[int] = []
        self._bound_blocks: list[int | None] = []
        self._bound_values: list[float] = []

    def add_two_sided(
        self,
        *,
        name: str,
        description: str,
        expression: str,
        big_m: float,
        record: ConstraintRecord,
        calendar: Sequence[OperativeStage],
    ) -> None:
        """Emit the ``<=``/``>=`` pair for a two-sided limit ``L <= expr <= U``.

        A ``sense="<="`` constraint is added iff *record* carries any bounded
        upper limit; a separate ``sense=">="`` constraint is added iff it
        carries any bounded lower limit. Both share *expression* — the
        coefficient already lives there, so neither RHS is sign-flipped.
        Adds nothing (not even an id) when neither side is ever bounded.

        For a per-block record, bound rows are emitted for
        ``range(min(len(stage_bounds.lower), len(calendar[stage].block_hours)))``
        blocks per declared stage (mirroring
        ``single_term_bounds._per_block_hydro_contributions``'s clamp); for a
        stage-level record (``record.per_block is False``), one row per
        declared stage is emitted with ``block_id=None``.
        """
        # Enumerate the exact (stage, block) slots that will carry a bound row —
        # per-block slots clamped to each stage's real block count, or one
        # stage-level slot (block_id=None). The header (<=/>=) decision below
        # MUST be taken over these SAME clamped slots as the row emission: a
        # bounded slot beyond the block count is dropped, so scanning the raw
        # (up-to-5-wide) StageBounds would append a constraint dict with no
        # companion RHS row (an orphan cobre cannot bind).
        slots: list[tuple[int, int | None, float | None, float | None]] = []
        for stage_index, stage_bounds in record.bounds.items():
            if record.per_block:
                n_slots = min(
                    len(stage_bounds.lower), len(calendar[stage_index].block_hours)
                )
                slots.extend(
                    (stage_index, b, stage_bounds.lower[b], stage_bounds.upper[b])
                    for b in range(n_slots)
                )
            else:
                slots.append(
                    (stage_index, None, stage_bounds.lower[0], stage_bounds.upper[0])
                )

        any_upper = any(_is_bounded(u) for _, _, _, u in slots)
        any_lower = any(_is_bounded(lo) for _, _, lo, _ in slots)
        if not any_upper and not any_lower:
            return

        slack = {"enabled": True, "penalty": big_m}
        upper_id: int | None = None
        lower_id: int | None = None

        if any_upper:
            upper_id = self._start_id + len(self._constraints)
            self._constraints.append(
                {
                    "id": upper_id,
                    "name": name,
                    "description": description,
                    "expression": expression,
                    "sense": "<=",
                    "slack": slack,
                }
            )
        if any_lower:
            lower_id = self._start_id + len(self._constraints)
            self._constraints.append(
                {
                    "id": lower_id,
                    "name": name,
                    "description": description,
                    "expression": expression,
                    "sense": ">=",
                    "slack": slack,
                }
            )

        for stage_index, block_id, lower, upper in slots:
            self._add_bound_rows(
                upper_id=upper_id,
                lower_id=lower_id,
                stage_id=stage_index,
                block_id=block_id,
                lower=lower,
                upper=upper,
            )

    def _add_bound_rows(
        self,
        *,
        upper_id: int | None,
        lower_id: int | None,
        stage_id: int,
        block_id: int | None,
        lower: float | None,
        upper: float | None,
    ) -> None:
        """Append one bounds row per bounded side that has a matching id."""
        if upper_id is not None and upper is not None and _is_bounded(upper):
            self._bound_cids.append(upper_id)
            self._bound_stages.append(stage_id)
            self._bound_blocks.append(block_id)
            self._bound_values.append(upper)
        if lower_id is not None and lower is not None and _is_bounded(lower):
            self._bound_cids.append(lower_id)
            self._bound_stages.append(stage_id)
            self._bound_blocks.append(block_id)
            self._bound_values.append(lower)

    def result(self) -> GenericConstraintResult | None:
        """Package everything added so far, or ``None`` if nothing was added."""
        if not self._constraints:
            return None
        bounds = pa.table(
            {
                "constraint_id": pa.array(self._bound_cids, type=pa.int32()),
                "stage_id": pa.array(self._bound_stages, type=pa.int32()),
                "block_id": pa.array(self._bound_blocks, type=pa.int32()),
                "bound": pa.array(self._bound_values, type=pa.float64()),
            },
            schema=_GENERIC_BOUNDS_SCHEMA,
        )
        return GenericConstraintResult(
            constraints=list(self._constraints), bounds=bounds
        )


def _emit_re_frequency_split_deferred(term: ConstraintTerm) -> None:
    """Emit the shared ``decomp-re-frequency-split-deferred`` WARNING.

    Resolving a frequency-split ``FU`` term (Itaipu 50/60 Hz) needs a
    frequency-to-bus map this conversion does not build; the caller drops
    the whole constraint rather than guess the bus.
    """
    emit(
        Diagnostic(
            code="decomp-re-frequency-split-deferred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RE frequency-split generation term deferred",
            summary=(
                "RE generic constraint carries a hydro generation term "
                f"(plant code {term.code}) split by frequency "
                f"({term.frequency} Hz), which needs a frequency-to-bus map "
                "this conversion does not build; the constraint is skipped."
            ),
            remediation=(
                "Frequency-split (RI-to-bus) resolution is deferred; no "
                "action needed to convert."
            ),
        )
    )


def _emit_re_thermal_not_normal(term: ConstraintTerm) -> None:
    """Emit the shared ``decomp-re-thermal-not-normal`` WARNING, naming the code."""
    emit(
        Diagnostic(
            code="decomp-re-thermal-not-normal",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RE thermal generation term is not a normal thermal",
            summary=(
                "RE generic constraint carries a thermal generation term "
                f"for thermal code {term.code}, which is not declared as a "
                "normal thermal (an anticipation-track/GNL thermal); the "
                "constraint is skipped."
            ),
            remediation=(
                "GNL coordination for RE constraints is deferred; no action "
                "needed to convert."
            ),
        )
    )


def _resolve_re_terms(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
) -> list[tuple[float, str]] | None:
    """Resolve every term of one generic RE record to ``(coefficient, token)`` pairs.

    Dispatches on ``term.variable``:

    - ``generation`` with ``frequency is not None`` — a frequency-split
      term with no RI-to-bus map available — emits
      :func:`_emit_re_frequency_split_deferred` and drops the whole record.
    - ``generation`` with ``frequency is None`` — resolved via
      ``_variable_token``.
    - ``thermal_generation`` whose ``code`` is absent from
      ``id_map.thermal_codes`` — a GNL/anticipation-track thermal not yet a
      normal thermal entity — emits :func:`_emit_re_thermal_not_normal` and
      drops the whole record; the membership check runs *before*
      ``_variable_token`` so a GNL code never reaches ``id_map.thermal_id``
      (which would raise ``KeyError``). Otherwise resolved via
      ``_variable_token``.
    - ``interchange`` — resolved via ``resolve_fi_term``; a ``None`` result
      (ticket-014's resolver already warned) drops the whole record.

    Any single unresolved term drops the **whole** constraint
    (skip-not-partial) — never a partial expression. Returns the ordered
    ``(coefficient, token)`` list when every term resolves.
    """
    resolved: list[tuple[float, str]] = []
    for term in record.terms:
        if term.variable == "generation":
            if term.frequency is not None:
                _emit_re_frequency_split_deferred(term)
                return None
            resolved.append((term.coefficient, _variable_token(term, id_map, {})))
        elif term.variable == "thermal_generation":
            if term.code not in id_map.thermal_codes:
                _emit_re_thermal_not_normal(term)
                return None
            resolved.append((term.coefficient, _variable_token(term, id_map, {})))
        elif term.variable == "interchange":
            token = resolve_fi_term(term, id_map, line_map)
            if token is None:
                return None
            resolved.append((term.coefficient, token))
        else:
            raise ValueError(
                f"_resolve_re_terms: unexpected RE term variable {term.variable!r}"
            )
    return resolved


def emit_re_generics(
    census: ConstraintCensus,
    id_map: DecompIdMap,
    line_map: Mapping[tuple[int, int], int],
    big_m: float,
    calendar: Sequence[OperativeStage],
    start_id: int,
) -> GenericConstraintResult | None:
    """Emit every generic ``RE`` constraint (multi-term hydro/thermal/interchange).

    Iterates ``census.to_generic`` filtered to ``record.family == "RE"``,
    resolves each record's terms via :func:`_resolve_re_terms`, and feeds a
    :class:`_GenericBuilder`; a record whose terms cannot all be resolved is
    dropped entirely (skip-not-partial, one diagnostic per skipped
    constraint — see :func:`_resolve_re_terms`). Single-hydro-generation RE
    records never reach here — they lower to a plant bound in
    ``single_term_bounds.py`` (``lowers_to_bound``). ``HQ``/``HV`` records
    also present in ``census.to_generic`` are :func:`emit_rhq_rhv_generics`'s,
    not this function's; ``HE`` records are unhandled (deferred).

    Reads no ``dadger``; *big_m* and *line_map* arrive from the caller (E7).
    Returns ``builder.result()`` — ``None`` when no RE generic constraint
    survives.
    """
    builder = _GenericBuilder(start_id)
    for record in census.to_generic:
        if record.family != "RE":
            continue
        terms = _resolve_re_terms(record, id_map, line_map)
        if terms is None:
            continue
        expression = _format_expression(terms)
        if expression is None:
            continue
        builder.add_two_sided(
            name=f"RE_{record.constraint_id}",
            description=f"RE generic constraint {record.constraint_id}",
            expression=expression,
            big_m=big_m,
            record=record,
            calendar=calendar,
        )
    return builder.result()


#: RHV ``CV.tipo`` volume variables whose hm³→flow conversion needs a
#: per-stage coefficient (block hours vary per stage) that a stage-invariant
#: cobre expression cannot carry — deferred to the E5 scalar-parameter
#: plumbing rather than emitted with a guessed coefficient.
_RHV_VOLUME_TIPOS = frozenset({"VDEF", "VDES", "VBOM"})


def _emit_rhq_qbom_no_station(record: ConstraintRecord, code: int) -> None:
    """Emit the shared ``decomp-rhq-qbom-no-station`` WARNING, naming the code."""
    emit(
        Diagnostic(
            code="decomp-rhq-qbom-no-station",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHQ QBOM term has no matching pumping station",
            summary=(
                f"HQ constraint {record.constraint_id} carries a QBOM term "
                f"for plant code {code}, which has no matching pumping "
                "station; the constraint is skipped."
            ),
            remediation=(
                f"HQ constraint {record.constraint_id} is skipped; check "
                f"that plant code {code} is declared as a pumping station."
            ),
        )
    )


def _resolve_rhq_terms(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
) -> list[tuple[float, str]] | None:
    """Resolve every term of one generic RHQ record to ``(coefficient, token)`` pairs.

    Every flow variable resolves via :func:`_variable_token`. A ``QBOM`` term
    whose ``code`` has no matching entry in *pumping_station_ids* — it is a
    pumping-station code, never a hydro code, so it cannot fall through to
    ``id_map.hydro_id`` — emits :func:`_emit_rhq_qbom_no_station` and drops
    the whole record (skip-not-partial), checked *before* the
    ``_variable_token`` call so a missing station never raises a raw
    ``KeyError``. Returns the ordered ``(coefficient, token)`` list when
    every term resolves.
    """
    resolved: list[tuple[float, str]] = []
    for term in record.terms:
        if term.variable == "QBOM" and term.code not in pumping_station_ids:
            _emit_rhq_qbom_no_station(record, term.code)
            return None
        resolved.append(
            (term.coefficient, _variable_token(term, id_map, pumping_station_ids))
        )
    return resolved


def _emit_rhv_varm_uncadastred(record: ConstraintRecord, code: int) -> None:
    """Emit the shared ``decomp-rhv-varm-plant-uncadastred`` WARNING, naming the code.

    Mirrors ``single_term_bounds._hv_storage_contributions``'s single-``VARM``
    guard, reusing the same diagnostic code for the multi-``VARM`` case.
    """
    emit(
        Diagnostic(
            code="decomp-rhv-varm-plant-uncadastred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHV VARM constraint references an uncadastred plant",
            summary=(
                f"HV constraint {record.constraint_id} bounds the stored "
                f"volume of plant code {code}, which has no cadastro row; "
                "the constraint cannot be lowered to a generic constraint."
            ),
            remediation=(
                f"HV constraint {record.constraint_id} is skipped; check "
                f"that plant code {code} is cadastred and operated."
            ),
        )
    )


def _emit_rhv_volume_tipo_deferred(record: ConstraintRecord, tipo: str) -> None:
    """Emit the shared ``decomp-rhv-volume-tipo-deferred`` WARNING, naming the tipo."""
    emit(
        Diagnostic(
            code="decomp-rhv-volume-tipo-deferred",
            severity=Severity.WARNING,
            category="Special constraints",
            title="RHV volume tipo term deferred",
            summary=(
                f"HV constraint {record.constraint_id} carries a {tipo} "
                "volume term; its hm3-to-flow form needs a per-stage "
                "coefficient (block hours vary per stage) that a "
                "stage-invariant cobre expression cannot carry, so the "
                "constraint is skipped rather than emitted with a guessed "
                "coefficient."
            ),
            remediation=(
                "Volume-tipo RHV lowering needs the E5 per-stage "
                "scalar-parameter plumbing; no action needed to convert."
            ),
        )
    )


def _offset_if_bounded(value: float | None, offset: float) -> float | None:
    """Add *offset* to *value* iff it is a real bound; otherwise pass it through.

    Mirrors :func:`_is_bounded`'s sentinel check, phrased so mypy narrows
    *value* to ``float`` on the addition: a ``None`` or ``±1e21`` side never
    gets an offset added to it.
    """
    if value is None or not _is_bounded(value):
        return value
    return value + offset


def _resolve_hv_varm(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    effective: EffectiveCadastro,
    calendar: Sequence[OperativeStage],
) -> tuple[list[tuple[float, str]], Mapping[int, StageBounds]] | None:
    """Resolve one generic multi-``VARM`` RHV record's expression and floor-adjusted
    bounds.

    Every term resolves via :func:`_variable_token` to a ``hydro_storage``
    token; a term whose ``code`` is absent from ``effective.base.index``
    emits :func:`_emit_rhv_varm_uncadastred` and drops the whole record
    (skip-not-partial) before any bounds are adjusted.

    The source model's ``LV`` limits are relative to each plant's own useful
    volume, so the absolute RHS is the **per-term additive floor**: for each
    declared stage, ``offset = sum(cᵢ * effective_storage_range(effective,
    codeᵢ, stage)[0] for term i)``, added to whichever side of the record's
    stage-level ``StageBounds`` is actually bounded (:func:`_is_bounded`) —
    a ``±1e21``/``None`` side is left untouched, never offset. *calendar* is
    accepted for signature symmetry with the per-block resolvers (mirroring
    ``single_term_bounds._hv_storage_contributions``); ``VARM`` is
    stage-level, so it is not read here.

    Returns the ``(coefficient, token)`` list and the floor-adjusted
    ``{stage_index: StageBounds}`` map when every term resolves.
    """
    resolved: list[tuple[float, str]] = []
    coded_terms: list[tuple[float, int]] = []
    for term in record.terms:
        if term.code not in effective.base.index:
            _emit_rhv_varm_uncadastred(record, term.code)
            return None
        resolved.append((term.coefficient, _variable_token(term, id_map, {})))
        coded_terms.append((term.coefficient, term.code))

    adjusted: dict[int, StageBounds] = {}
    for stage_index, stage_bounds in record.bounds.items():
        offset = sum(
            coefficient * effective_storage_range(effective, code, stage_index)[0]
            for coefficient, code in coded_terms
        )
        adjusted[stage_index] = StageBounds(
            lower=(_offset_if_bounded(stage_bounds.lower[0], offset),),
            upper=(_offset_if_bounded(stage_bounds.upper[0], offset),),
        )
    return resolved, adjusted


def emit_rhq_rhv_generics(
    census: ConstraintCensus,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
    effective: EffectiveCadastro,
    big_m: float,
    calendar: Sequence[OperativeStage],
    start_id: int,
) -> GenericConstraintResult | None:
    """Emit every generic ``RHQ``/``RHV`` constraint (flow mixes and multi-``VARM``).

    Iterates ``census.to_generic`` filtered to ``record.family in ("HQ",
    "HV")``:

    - ``"HQ"`` (per-block flow terms) resolves via :func:`_resolve_rhq_terms`
      and feeds a two-sided ``HQ_<id>`` constraint.
    - ``"HV"`` carrying any volume-tipo term (``VDEF``/``VDES``/``VBOM``)
      emits :func:`_emit_rhv_volume_tipo_deferred` and is skipped (deferred
      to the E5 scalar-parameter plumbing — see the module docstring).
    - ``"HV"`` all-``VARM`` (stage-level) resolves via
      :func:`_resolve_hv_varm` and feeds a two-sided ``HV_<id>`` constraint
      on ``dataclasses.replace(record, bounds=adjusted)`` — the floor-
      adjusted bounds, never the reader's original record (a frozen
      dataclass is never mutated in place).

    Any record whose terms cannot all be resolved is dropped entirely
    (skip-not-partial, one diagnostic per skipped constraint). Single-term
    RHQ/RHV records on a bounded variable never reach here — they lower to
    an entity bound in ``single_term_bounds.py`` (``lowers_to_bound``).

    Reads no ``dadger``; *pumping_station_ids*, *effective*, *big_m*, and
    *start_id* arrive from the caller (E7). Returns ``builder.result()`` —
    ``None`` when no RHQ/RHV generic constraint survives.
    """
    builder = _GenericBuilder(start_id)
    for record in census.to_generic:
        if record.family not in ("HQ", "HV"):
            continue

        if record.family == "HQ":
            rhq_terms = _resolve_rhq_terms(record, id_map, pumping_station_ids)
            if rhq_terms is None:
                continue
            expression = _format_expression(rhq_terms)
            if expression is None:
                continue
            builder.add_two_sided(
                name=f"HQ_{record.constraint_id}",
                description=f"RHQ generic constraint {record.constraint_id}",
                expression=expression,
                big_m=big_m,
                record=record,
                calendar=calendar,
            )
            continue

        volume_tipo = next(
            (t.variable for t in record.terms if t.variable in _RHV_VOLUME_TIPOS),
            None,
        )
        if volume_tipo is not None:
            _emit_rhv_volume_tipo_deferred(record, volume_tipo)
            continue

        hv_resolved = _resolve_hv_varm(record, id_map, effective, calendar)
        if hv_resolved is None:
            continue
        hv_terms, adjusted_bounds = hv_resolved
        expression = _format_expression(hv_terms)
        if expression is None:
            continue
        builder.add_two_sided(
            name=f"HV_{record.constraint_id}",
            description=f"RHV generic constraint {record.constraint_id}",
            expression=expression,
            big_m=big_m,
            record=dataclasses.replace(record, bounds=adjusted_bounds),
            calendar=calendar,
        )
    return builder.result()
