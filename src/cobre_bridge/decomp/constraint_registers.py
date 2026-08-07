"""Read the special-constraint register families into joined, classified records.

The source model expresses special constraints as four register *triples* — a
declaration register (which stages the constraint spans), a per-stage limit
register, and a coefficient/participation register naming the entities:

    family  decl  limit        coeff  variable(s)               limit axis
    ----    ----  -----        -----  -----------               ----------
    RE      RE    LU (1..5)    FU     hydro generation (MW)     per-block
    HQ      HQ    LQ (1..5)    CQ     flow, per-term CQ.tipo    per-block
    HV      HV    LV           CV     hydro storage (hm3)       per-stage
    HE      HE    (inline)     CM     energy over a REE cascade per-stage

Participation (FU/CQ/CV) is declared at a base stage and inherited forward across
the constraint's ``[estagio_inicial, estagio_final]`` range; the limit registers
(LU/LQ/LV) are likewise sparse per stage and inherited forward — the same
sparse-stage convention the ``CT`` reader uses.

Each *term* carries its own variable: an ``HQ`` constraint may mix flow variables
(e.g. ``QDEF`` on one plant and ``QDES`` on another), so the variable is a
per-term property, not a per-constraint one. A constraint therefore lowers to an
entity **bound** only when it is a *single term* on a variable that has a cobre
bounds axis; every other shape (multiple terms, an unbounded variable such as
spillage/pumping, or the ``HE`` energy sum) becomes a **generic constraint**.

This module only reads, joins, and classifies. The lowering itself lives in the
bounds emitters and the generic-constraints emitter (M2.1 T3-T7). See
``plans/decomp-constraints-m21.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from idecomp.decomp import Dadger

_LOG = logging.getLogger(__name__)

#: The most block slots the source model's per-block limit rows carry
#: (``limite_{inferior,superior}_1`` … ``_5``).
_MAX_BLOCK_SLOTS = 5

#: Variables that have a cobre entity-bounds axis — a single-term constraint on
#: one of these lowers to a plant bound (`constraints/bounds.rs`). Everything else
#: (spillage, pumping, the HE energy sum) is always a generic constraint, because
#: there is no bound column for it.
_BOUNDS_AXIS: dict[str, str] = {
    "generation": "generation",  # RE  -> min/max_generation_mw
    "QDEF": "outflow",  # HQ defluencia -> min/max_outflow_m3s
    "QTUR": "turbined",  # HQ turbinado  -> min/max_turbined_m3s
    "VARM": "storage",  # HV volume     -> min/max_storage_hm3
}


@dataclass(frozen=True)
class ConstraintTerm:
    """One ``(entity, variable)`` participation in a constraint.

    ``code`` is the plant code (``codigo_usina``) for RE/HQ/HV or the REE code
    (``codigo_ree``) for HE. ``variable`` is what the term bounds — ``generation``
    (RE), the ``CQ.tipo`` flow (``QDEF``/``QTUR``/``QDES``/``QBOM``) for HQ,
    ``VARM`` for HV, or ``energy`` for HE. ``frequency`` is the 50/60 Hz tag
    carried only by RE electrical terms (``None`` elsewhere); a plant split across
    both frequencies contributes two distinct terms.
    """

    code: int
    coefficient: float
    variable: str
    frequency: float | None = None


@dataclass(frozen=True)
class StageBounds:
    """Per-slot lower/upper limits at one stage.

    One slot per block for the per-block families (RE/HQ, up to
    :data:`_MAX_BLOCK_SLOTS`), a single slot for the stage-level families
    (HV/HE). ``None`` marks an absent bound (a one-sided limit).
    """

    lower: tuple[float | None, ...]
    upper: tuple[float | None, ...]


@dataclass(frozen=True)
class ConstraintRecord:
    """One special constraint, joined across its register triple and made dense
    over its stage range."""

    family: str  # "RE" | "HQ" | "HV" | "HE"
    constraint_id: int
    stage_start: int  # 0-based, inclusive
    stage_end: int  # 0-based, inclusive
    terms: tuple[ConstraintTerm, ...]
    bounds: Mapping[int, StageBounds]  # 0-based stage index -> bounds, dense
    per_block: bool
    tipo_limite: int | None = None  # HE: 2 = lower limit
    penalty: float | None = None  # HE valor_penalidade

    @property
    def n_entities(self) -> int:
        """Distinct participating entities (plants, or REEs for HE)."""
        return len({t.code for t in self.terms})

    @property
    def is_single_entity(self) -> bool:
        """True when exactly one entity participates (may still be multi-term)."""
        return self.n_entities == 1

    @property
    def is_single_term(self) -> bool:
        """True when the constraint has exactly one ``(entity, variable)`` term."""
        return len(self.terms) == 1


def lowers_to_bound(record: ConstraintRecord) -> bool:
    """A constraint lowers to an entity bound iff it is a single ``(entity,
    variable)`` term on a variable that has a cobre bounds axis; otherwise it is a
    generic constraint.

    Spillage (``QDES``), pumping (``QBOM``) and the ``HE`` energy sum have no
    bound column, so they stay generic even as a single term.
    """
    return record.is_single_term and record.terms[0].variable in _BOUNDS_AXIS


@dataclass(frozen=True)
class _FamilyConfig:
    """Static description of a declaration/limit/coefficient register triple."""

    family: str
    decl_attr: str  # "re" / "hq" / "hv"
    limit_attr: str  # "lu" / "lq" / "lv"
    coeff_attr: str  # "fu" / "cq" / "cv"
    per_block: bool  # True for RE/HQ, False for HV
    #: Fixed per-term variable (RE = "generation"), or ``None`` to read it from
    #: the coefficient register's ``tipo`` column (HQ/HV).
    fixed_variable: str | None
    has_frequency: bool  # RE carries a 50/60 Hz tag on FU


_TRIPLE_FAMILIES: tuple[_FamilyConfig, ...] = (
    _FamilyConfig("RE", "re", "lu", "fu", True, "generation", True),
    _FamilyConfig("HQ", "hq", "lq", "cq", True, None, False),
    _FamilyConfig("HV", "hv", "lv", "cv", False, None, False),
)


def _df(dadger: Dadger, attr: str) -> pd.DataFrame | None:
    """Read one register as a DataFrame, treating an absent register as empty."""
    frame = getattr(dadger, attr)(df=True)
    if frame is None or getattr(frame, "empty", True):
        return None
    return frame


def _opt_float(value: object) -> float | None:
    """A cell as ``float``, or ``None`` when blank (NaN)."""
    return None if pd.isna(value) else float(value)  # type: ignore[arg-type]


def _read_stage_bounds(row: pd.Series, per_block: bool) -> StageBounds:
    """One limit row -> its per-slot lower/upper bounds.

    Per-block rows expose ``limite_{inferior,superior}_1`` … ``_N``; the
    stage-level rows expose bare ``limite_inferior`` / ``limite_superior``.
    """
    if not per_block:
        return StageBounds(
            lower=(_opt_float(row["limite_inferior"]),),
            upper=(_opt_float(row["limite_superior"]),),
        )
    lower: list[float | None] = []
    upper: list[float | None] = []
    for slot in range(1, _MAX_BLOCK_SLOTS + 1):
        lo_col, up_col = f"limite_inferior_{slot}", f"limite_superior_{slot}"
        if lo_col not in row.index:
            break
        lower.append(_opt_float(row[lo_col]))
        upper.append(_opt_float(row[up_col]))
    return StageBounds(lower=tuple(lower), upper=tuple(upper))


def _forward_fill_bounds(
    declared: Mapping[int, StageBounds],
    stage_start: int,
    stage_end: int,
) -> dict[int, StageBounds]:
    """Densify sparse per-stage limits over ``[stage_start, stage_end]``.

    A stage inherits the most recently declared limit at or before it; stages
    before the first declaration inherit that first declaration (mirroring the
    ``CT`` sparse-stage rule). Returns ``{}`` when nothing is declared.
    """
    if not declared:
        return {}
    first = declared[min(declared)]
    dense: dict[int, StageBounds] = {}
    current = first
    for stage in range(stage_start, stage_end + 1):
        if stage in declared:
            current = declared[stage]
        dense[stage] = current
    return dense


def _read_terms(
    coeff: pd.DataFrame,
    config: _FamilyConfig,
) -> dict[int, list[ConstraintTerm]]:
    """Group a coefficient register into ``{constraint_id: [terms]}``.

    Each ``(entity, variable, frequency)`` identity contributes one term; a
    repeated identity (declared at several stages) collapses to its first
    coefficient — the source coefficients are pure ``±1``/``1`` signs, constant
    across stages.
    """
    terms: dict[int, list[ConstraintTerm]] = {}
    seen: dict[int, set[tuple[int, str, float | None]]] = {}
    for _, row in coeff.iterrows():
        cid = int(row["codigo_restricao"])
        code = int(row["codigo_usina"])
        variable = (
            config.fixed_variable
            if config.fixed_variable is not None
            else str(row["tipo"]).strip()
        )
        frequency = (
            _opt_float(row["frequencia"])
            if config.has_frequency and "frequencia" in row.index
            else None
        )
        identity = (code, variable, frequency)
        if identity in seen.setdefault(cid, set()):
            continue
        seen[cid].add(identity)
        terms.setdefault(cid, []).append(
            ConstraintTerm(
                code=code,
                coefficient=float(row["coeficiente"]),
                variable=variable,
                frequency=frequency,
            )
        )
    return terms


def _read_triple_family(
    dadger: Dadger,
    config: _FamilyConfig,
) -> list[ConstraintRecord]:
    """Join one RE/HQ/HV register triple into classified constraint records."""
    decl = _df(dadger, config.decl_attr)
    coeff = _df(dadger, config.coeff_attr)
    if decl is None or coeff is None:
        return []
    limit = _df(dadger, config.limit_attr)

    terms_by_cid = _read_terms(coeff, config)

    limits_by_cid: dict[int, dict[int, StageBounds]] = {}
    if limit is not None:
        for _, row in limit.iterrows():
            cid = int(row["codigo_restricao"])
            stage0 = int(row["estagio"]) - 1
            limits_by_cid.setdefault(cid, {})[stage0] = _read_stage_bounds(
                row, config.per_block
            )

    records: list[ConstraintRecord] = []
    skipped_no_terms = 0
    for _, row in decl.iterrows():
        cid = int(row["codigo_restricao"])
        terms = terms_by_cid.get(cid)
        if not terms:
            skipped_no_terms += 1
            continue
        stage_start = int(row["estagio_inicial"]) - 1
        stage_end = int(row["estagio_final"]) - 1
        bounds = _forward_fill_bounds(
            limits_by_cid.get(cid, {}), stage_start, stage_end
        )
        records.append(
            ConstraintRecord(
                family=config.family,
                constraint_id=cid,
                stage_start=stage_start,
                stage_end=stage_end,
                terms=tuple(terms),
                bounds=bounds,
                per_block=config.per_block,
            )
        )
    if skipped_no_terms:
        _LOG.info(
            "%s: %d declared constraint(s) carry no participation rows and were "
            "skipped (inactive / defined elsewhere)",
            config.family,
            skipped_no_terms,
        )
    return records


def _read_energy_constraints(dadger: Dadger) -> list[ConstraintRecord]:
    """Join the ``HE``/``CM`` energy family (VminOP-style, over a REE cascade).

    ``HE`` rows carry the per-stage ``limite`` inline (with ``tipo_limite`` and
    the penalty); ``CM`` carries the per-REE productivity coefficients. Every HE
    is a generic constraint — the ``energy`` variable expands to
    ``Σ ρ_acum · storage`` over the REE's plants downstream (T7).
    """
    he = _df(dadger, "he")
    cm = _df(dadger, "cm")
    if he is None or cm is None:
        return []

    terms_by_cid: dict[int, list[ConstraintTerm]] = {}
    for _, row in cm.iterrows():
        cid = int(row["codigo_restricao"])
        terms_by_cid.setdefault(cid, []).append(
            ConstraintTerm(
                code=int(row["codigo_ree"]),
                coefficient=float(row["coeficiente"]),
                variable="energy",
            )
        )

    per_stage: dict[int, dict[int, StageBounds]] = {}
    meta: dict[int, tuple[int | None, float | None]] = {}
    stages_seen: dict[int, list[int]] = {}
    for _, row in he.iterrows():
        cid = int(row["codigo_restricao"])
        stage0 = int(row["estagio"]) - 1
        limit_value = _opt_float(row["limite"])
        tipo_limite = None if pd.isna(row["tipo_limite"]) else int(row["tipo_limite"])
        # tipo_limite 2 = lower limit; the value is a floor on the energy sum.
        bound = (
            StageBounds(lower=(limit_value,), upper=(None,))
            if tipo_limite == 2
            else StageBounds(lower=(None,), upper=(limit_value,))
        )
        per_stage.setdefault(cid, {})[stage0] = bound
        stages_seen.setdefault(cid, []).append(stage0)
        meta[cid] = (tipo_limite, _opt_float(row.get("valor_penalidade")))

    records: list[ConstraintRecord] = []
    for cid, terms in terms_by_cid.items():
        stages = stages_seen.get(cid)
        if not stages:
            continue
        stage_start, stage_end = min(stages), max(stages)
        bounds = _forward_fill_bounds(per_stage[cid], stage_start, stage_end)
        tipo_limite, penalty = meta[cid]
        records.append(
            ConstraintRecord(
                family="HE",
                constraint_id=cid,
                stage_start=stage_start,
                stage_end=stage_end,
                terms=tuple(terms),
                bounds=bounds,
                per_block=False,
                tipo_limite=tipo_limite,
                penalty=penalty,
            )
        )
    return records


@dataclass(frozen=True)
class ConstraintCensus:
    """The four families read and split by their lowering target."""

    #: Every record, keyed by family ("RE"/"HQ"/"HV"/"HE").
    by_family: Mapping[str, Sequence[ConstraintRecord]]
    #: Single-term records on a bounded variable — lower to entity bounds.
    to_bounds: Sequence[ConstraintRecord] = field(default_factory=tuple)
    #: Everything else — lower to generic constraints.
    to_generic: Sequence[ConstraintRecord] = field(default_factory=tuple)


def read_constraints(dadger: Dadger) -> ConstraintCensus:
    """Read all four special-constraint families and split them by lowering target.

    Returns a :class:`ConstraintCensus`: every record grouped by family, plus the
    two lowering buckets — ``to_bounds`` (single-term on a bounded variable) and
    ``to_generic`` (multi-term, unbounded-variable, or ``HE``).
    """
    by_family: dict[str, list[ConstraintRecord]] = {}
    for config in _TRIPLE_FAMILIES:
        by_family[config.family] = _read_triple_family(dadger, config)
    by_family["HE"] = _read_energy_constraints(dadger)

    to_bounds: list[ConstraintRecord] = []
    to_generic: list[ConstraintRecord] = []
    for records in by_family.values():
        for record in records:
            (to_bounds if lowers_to_bound(record) else to_generic).append(record)

    _LOG.info(
        "special constraints read: %s; %d -> bounds, %d -> generic",
        {fam: len(recs) for fam, recs in by_family.items()},
        len(to_bounds),
        len(to_generic),
    )
    return ConstraintCensus(
        by_family={fam: tuple(recs) for fam, recs in by_family.items()},
        to_bounds=tuple(to_bounds),
        to_generic=tuple(to_generic),
    )
