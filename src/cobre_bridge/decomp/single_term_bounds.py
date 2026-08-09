"""Lower single-term, unit-coefficient special constraints to entity bounds.

``constraint_registers.read_constraints`` splits every special constraint
into ``to_bounds`` (single-term, ``|coefficient|==1``, bounded-variable) and
``to_generic``. This module turns each ``to_bounds`` record into
:class:`~cobre_bridge.decomp.bounds_accumulator.BoundContribution`\\ s that
``bounds_accumulator.resolve``/``build_bound_tables`` can later fold into the
cobre bound parquet rows. It produces contributions only — it does not write
parquet, call ``resolve``/``build_bound_tables``, or touch the pipeline.

``single_term_bound_contributions`` is the stable public seam: it dispatches
each record on its ``family`` (``"RE"``, ``"HQ"``, and ``"HV"``) and raises
on anything else, so an unhandled family fails loud instead of silently
dropping a bound.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from cobre_bridge.decomp.bounds_accumulator import BoundContribution
from cobre_bridge.decomp.cadastro import effective_storage_range
from cobre_bridge.diagnostics import Diagnostic, Severity, emit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.constraint_registers import (
        ConstraintCensus,
        ConstraintRecord,
    )
    from cobre_bridge.decomp.id_map import DecompIdMap
    from cobre_bridge.decomp.temporal import OperativeStage


#: RHQ ``CQ.tipo`` flow -> cobre hydro bound axis, mirroring the reader's
#: ``constraint_registers._BOUNDS_AXIS`` QDEF/QTUR entries. Keep the two
#: mappings consistent for a *hydro* flow axis; a future hydro flow ``tipo``
#: gaining a bound axis updates both. ``QBOM`` is the one asymmetry: it
#: joined ``_BOUNDS_AXIS`` in M2 (ticket-020) but is a *pumping*-entity axis,
#: not a hydro one, so it is dispatched to
#: :func:`_qbom_pumping_contributions` instead of living in this mapping.
_HQ_AXIS_BY_VARIABLE: dict[str, str] = {"QDEF": "outflow", "QTUR": "turbined"}


def _sided_bounds(
    lower: float | None, upper: float | None, coefficient: float
) -> tuple[float | None, float | None]:
    """The coefficient-sign -> side map for a single term ``c*x in [lower, upper]``.

    ``coefficient > 0`` (``+1``) leaves the entity bound as-is:
    ``x in [lower, upper]``. ``coefficient < 0`` (``-1``) flips and swaps the
    sides: ``-x in [lower, upper] => x in [-upper, -lower]``. Only ``+-1``
    ever reaches here (``lowers_to_bound`` gates ``abs(coefficient) == 1``),
    so the two branches are exhaustive.

    The ``+-1e21`` unbounded sentinel is not re-derived here — a bound is
    passed through unchanged (negating it keeps its magnitude at the
    sentinel) and the accumulator's ``_effective`` is the single place that
    drops it. Only an actual ``None`` is treated as "no bound on that side".
    """
    if coefficient > 0:
        return lower, upper
    return (None if upper is None else -upper, None if lower is None else -lower)


def _per_block_contributions(
    record: ConstraintRecord,
    family: str,
    entity_id: int,
    axis: str,
    contributor: str,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """The shared per-block emission loop for a single-term entity bound.

    Iterates every declared stage in ``record.bounds`` and, within it, every
    block up to the stage's *real* block count
    (``len(calendar[stage_index].block_hours)``) — never the raw
    ``StageBounds`` slot count, which LU carries up to 5 slots wide
    regardless of the stage's actual block count. A ``block_id`` at or past
    ``len(stage_bounds.lower)`` is skipped (no declared slot for it), and a
    block whose sided bounds are both ``None`` emits no contribution.
    """
    contributions: list[BoundContribution] = []
    coefficient = record.terms[0].coefficient
    for stage_index, stage_bounds in record.bounds.items():
        n_blocks = len(calendar[stage_index].block_hours)
        for block_id in range(n_blocks):
            if block_id >= len(stage_bounds.lower):
                continue
            entity_lower, entity_upper = _sided_bounds(
                stage_bounds.lower[block_id],
                stage_bounds.upper[block_id],
                coefficient,
            )
            if entity_lower is None and entity_upper is None:
                continue
            contributions.append(
                BoundContribution(
                    family=family,
                    entity_id=entity_id,
                    stage_id=stage_index,
                    block_id=block_id,
                    axis=axis,
                    lower=entity_lower,
                    upper=entity_upper,
                    contributor=contributor,
                )
            )
    return contributions


def _re_generation_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RE single-hydro-generation constraint -> hydro ``generation`` bounds."""
    code = record.terms[0].code
    hydro_id = id_map.hydro_id(code)
    return _per_block_contributions(
        record, "hydro", hydro_id, "generation", f"RE_{record.constraint_id}", calendar
    )


def _ft_thermal_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RE single-thermal-generation constraint -> thermal ``generation`` bounds.

    Mirrors :func:`_re_generation_contributions`, but the term's ``code`` is a
    thermal ``codigo_usina`` (an ``FT`` term), so it resolves via
    ``id_map.thermal_id`` rather than ``id_map.hydro_id``.
    """
    thermal_id = id_map.thermal_id(record.terms[0].code)
    return _per_block_contributions(
        record,
        "thermal",
        thermal_id,
        "generation",
        f"RE_{record.constraint_id}",
        calendar,
    )


def _hq_flow_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RHQ single-flow constraint -> hydro ``outflow``/``turbined`` bounds.

    ``QDEF`` lowers to ``outflow``, ``QTUR`` to ``turbined`` — a QDEF and a
    QTUR constraint on the same plant land on different axes and both
    survive (the accumulator keys on axis).
    """
    term = record.terms[0]
    axis = _HQ_AXIS_BY_VARIABLE[term.variable]
    hydro_id = id_map.hydro_id(term.code)
    return _per_block_contributions(
        record, "hydro", hydro_id, axis, f"HQ_{record.constraint_id}", calendar
    )


def _qbom_pumping_contributions(
    record: ConstraintRecord,
    pumping_station_ids: Mapping[int, int],
    calendar: Sequence[OperativeStage],
) -> list[BoundContribution]:
    """One RHQ single ``QBOM`` constraint -> pumping ``flow`` bounds.

    A ``QBOM`` term's ``code`` is a pumping-station ``codigo_usina``, not a
    hydro code, so it resolves through *pumping_station_ids* rather than
    ``id_map.hydro_id``. A code absent from *pumping_station_ids* emits a
    ``WARNING`` diagnostic and is skipped entirely — skip-not-partial,
    mirroring ``constraints.py::_emit_rhq_qbom_no_station`` — rather than
    raising a raw ``KeyError`` or falling back to a generic constraint.
    """
    code = record.terms[0].code
    if code not in pumping_station_ids:
        emit(
            Diagnostic(
                code="decomp-rhq-qbom-no-station",
                severity=Severity.WARNING,
                category="Special constraints",
                title="RHQ QBOM term has no matching pumping station",
                summary=(
                    f"HQ constraint {record.constraint_id} carries a QBOM "
                    f"term for plant code {code}, which has no matching "
                    "pumping station; the constraint is skipped."
                ),
                remediation=(
                    f"HQ constraint {record.constraint_id} is skipped; check "
                    f"that plant code {code} is declared as a pumping "
                    "station."
                ),
            )
        )
        return []
    return _per_block_contributions(
        record,
        "pumping",
        pumping_station_ids[code],
        "flow",
        f"HQ_{record.constraint_id}",
        calendar,
    )


def _hv_storage_contributions(
    record: ConstraintRecord,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """One RHV single ``VARM`` constraint -> hydro ``storage`` bounds.

    Storage is a stage-level axis (unlike RE/RHQ's per-block axes), so every
    contribution carries ``block_id=None`` — the accumulator raises on a
    ``block_id`` reaching the ``storage`` axis, so this does not route through
    :func:`_per_block_contributions`.

    The source model's ``LV`` limits are relative to the plant's useful
    volume; cobre's ``min/max_storage_hm3`` are absolute. The sign map
    (:func:`_sided_bounds`) runs first, then each surviving side is added to
    the per-stage effective floor (:func:`~cobre_bridge.decomp.cadastro.
    effective_storage_range`'s floor element, which honours per-stage ``AC
    VOLMIN``/``VOLMAX`` overrides and the run-of-river ``D`` collapse) to
    reach the absolute bound.

    A ``code`` absent from the cadastro (``effective.base.index``) emits a
    ``WARNING`` diagnostic and is skipped entirely rather than crashing or
    silently dropping the constraint.
    """
    code = record.terms[0].code
    if code not in effective.base.index:
        emit(
            Diagnostic(
                code="decomp-rhv-varm-plant-uncadastred",
                severity=Severity.WARNING,
                category="Special constraints",
                title="RHV VARM constraint references an uncadastred plant",
                summary=(
                    f"HV constraint {record.constraint_id} bounds the stored "
                    f"volume of plant code {code}, which has no cadastro row; "
                    "the constraint cannot be lowered to a storage bound."
                ),
                remediation=(
                    f"HV constraint {record.constraint_id} is skipped; check "
                    f"that plant code {code} is cadastred and operated."
                ),
            )
        )
        return []

    hydro_id = id_map.hydro_id(code)
    coefficient = record.terms[0].coefficient
    contributions: list[BoundContribution] = []
    for stage_index, stage_bounds in record.bounds.items():
        rel_lower, rel_upper = _sided_bounds(
            stage_bounds.lower[0], stage_bounds.upper[0], coefficient
        )
        floor = effective_storage_range(effective, code, stage_index)[0]
        abs_lower = None if rel_lower is None else floor + rel_lower
        abs_upper = None if rel_upper is None else floor + rel_upper
        if abs_lower is None and abs_upper is None:
            continue
        contributions.append(
            BoundContribution(
                family="hydro",
                entity_id=hydro_id,
                stage_id=stage_index,
                block_id=None,
                axis="storage",
                lower=abs_lower,
                upper=abs_upper,
                contributor=f"HV_{record.constraint_id}",
            )
        )
    return contributions


def single_term_bound_contributions(
    census: ConstraintCensus,
    id_map: DecompIdMap,
    pumping_station_ids: Mapping[int, int],
    calendar: Sequence[OperativeStage],
    effective: EffectiveCadastro,
) -> list[BoundContribution]:
    """Lower every ``census.to_bounds`` record to its entity bound contributions.

    Dispatches on ``record.family``: ``"RE"`` lowers to a ``generation`` bound
    on either the hydro (``FU``/``generation``) or the thermal
    (``FT``/``thermal_generation``) entity named by its single term — any
    other bounded RE variable raises ``ValueError`` naming it. ``"HQ"``
    further dispatches on its single term's variable: ``QBOM`` lowers to a
    pumping ``flow`` bound via :func:`_qbom_pumping_contributions` (resolved
    through *pumping_station_ids*, never ``id_map.hydro_id`` — its ``code``
    is a pumping-station code, not a hydro one); everything else
    (``QDEF``/``QTUR``) lowers to hydro ``outflow``/``turbined`` bounds via
    :func:`_hq_flow_contributions`. ``"HV"`` lowers to hydro ``storage``
    bounds. Any other family raises ``ValueError`` naming it.
    *pumping_station_ids* feeds only the ``HQ`` ``QBOM`` path — required, not
    defaulted, so a forgotten wiring fails loud instead of silently dropping
    every QBOM bound; *effective* feeds only the ``HV`` handler's per-stage
    floor and cadastro guard. Neither the RE path nor the ``QDEF``/``QTUR``
    HQ path reads either.

    Raises
    ------
    KeyError
        Propagated from ``id_map.hydro_id``/``id_map.thermal_id`` when an
        RE/HQ/HV term names a code the id map does not know (a real
        reader/id-map mismatch).
    ValueError
        When a ``to_bounds`` record carries a family other than ``"RE"``/
        ``"HQ"``/``"HV"``, or an ``"RE"`` record whose single term's variable
        is neither ``"generation"`` nor ``"thermal_generation"``.
    """
    contributions: list[BoundContribution] = []
    for record in census.to_bounds:
        if record.family == "RE":
            variable = record.terms[0].variable
            if variable == "generation":
                contributions.extend(
                    _re_generation_contributions(record, id_map, calendar)
                )
            elif variable == "thermal_generation":
                contributions.extend(
                    _ft_thermal_contributions(record, id_map, calendar)
                )
            else:
                raise ValueError(
                    "single_term_bound_contributions: unhandled RE bounded "
                    f"variable {variable!r}"
                )
        elif record.family == "HQ":
            if record.terms[0].variable == "QBOM":
                contributions.extend(
                    _qbom_pumping_contributions(record, pumping_station_ids, calendar)
                )
            else:
                contributions.extend(_hq_flow_contributions(record, id_map, calendar))
        elif record.family == "HV":
            contributions.extend(
                _hv_storage_contributions(record, id_map, calendar, effective)
            )
        else:
            raise ValueError(
                f"single_term_bound_contributions: unhandled family {record.family!r}"
            )
    return contributions
