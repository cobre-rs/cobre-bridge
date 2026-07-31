"""Post-emission self-checks mirroring cobre 0.13's new hard validation rules.

Both pipelines build every artifact in memory before writing the Cobre case
directory. This module inspects those in-memory artifacts — the ``hydros.json``
document, the ``stages.json`` document, and the ``*_bounds`` Parquet tables —
for the four cobre 0.13 rules the bridge is most likely to violate silently,
and reports a match as an ``ERROR`` :class:`~cobre_bridge.diagnostics.Diagnostic`
through the sink the pipeline already runs converters inside. A failure here
therefore surfaces with bridge-side context (entity, stage, column, declared
vs. offending value) in milliseconds, instead of at ``cobre validate``/``cobre
run`` load time.

This is a **courtesy mirror**, not a substitute: cobre remains the authority.
Each rule is scoped to match cobre's own implementation exactly (see
``~/git/cobre/crates/cobre-io/src/validation/semantic/``):

- Rule 43 — :func:`check_hydro_bounds_no_raising`
  (``block_bounds.rs::check_bound_raises_declared_capacity``).
- Rule 41 — :func:`check_unit_group_envelope`
  (``hydro.rs::check_hydro_unit_groups``, ceiling half only; the floor half,
  rule 44, is out of scope here).
- Rule 36 — :func:`check_bound_row_uniqueness`
  (``block_bounds.rs::check_duplicate_bound_rows``).
- The ``block_id``-range rule — :func:`check_bound_block_id_range`
  (``block_bounds.rs::check_bound_block_id_range``).

This module must not import either pipeline — they import it — so it stays
unit-testable against hand-built artifacts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pyarrow as pa

from cobre_bridge.diagnostics import (
    Diagnostic,
    DiagnosticTable,
    Severity,
    emit,
    format_stage_ranges,
)

_LOG = logging.getLogger(__name__)

_CATEGORY = "Emission self-checks"

#: Mirrors cobre-io's
#: ``crates/cobre-io/src/validation/semantic/mod.rs::ENVELOPE_TOLERANCE``. A
#: relative tolerance, not an absolute one — an absolute epsilon would false-fire
#: on a plant declared at, say, 1e6 m^3/s and float-noise-pass a plant near zero.
_ENVELOPE_TOLERANCE = 1e-9

#: Both guarded columns for rule 43 / rule 41, checked independently.
_ENVELOPE_COLUMNS: tuple[str, ...] = ("max_turbined_m3s", "max_generation_mw")


def _tolerance(declared: float) -> float:
    """``ENVELOPE_TOLERANCE * max(|declared|, 1.0)`` — cobre's envelope tolerance."""
    return _ENVELOPE_TOLERANCE * max(abs(declared), 1.0)


@dataclass(frozen=True)
class BoundFamily:
    """One emitted bound-override table plus the metadata the row-level checks
    (:func:`check_bound_row_uniqueness`, :func:`check_bound_block_id_range`) need
    to name it in a finding.

    *table* is ``None`` when the pipeline did not build that family for this
    case (e.g. a NEWAVE case with no thermal capacity overrides); the row-level
    checks skip an absent family without a finding, exactly as they would skip
    an empty one.
    """

    label: str
    entity_column: str
    table: pa.Table | None


def _hydro_declared(hydros: Mapping[str, object]) -> dict[int, dict[str, float]]:
    """``{hydro_id: {column: declared_value}}`` from the ``hydros.json`` document.

    Only columns whose value is present and numeric are included, so a hydro
    entry with no ``generation`` block (or missing a column) simply has no
    entry for it — the callers treat that as "nothing to check", not a defect
    of this module.
    """
    declared: dict[int, dict[str, float]] = {}
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return declared
    for hydro in raw_hydros:
        if not isinstance(hydro, Mapping):
            continue
        hydro_id = hydro.get("id")
        if not isinstance(hydro_id, int):
            continue
        generation = hydro.get("generation")
        if not isinstance(generation, Mapping):
            continue
        entry: dict[str, float] = {}
        for column in _ENVELOPE_COLUMNS:
            value = generation.get(column)
            if isinstance(value, int | float):
                entry[column] = float(value)
        if entry:
            declared[hydro_id] = entry
    return declared


def check_hydro_bounds_no_raising(
    hydros: Mapping[str, object],
    hydro_bounds: pa.Table | None,
) -> None:
    """Cobre rule 43 mirror: no ``hydro_bounds`` row may raise ``max_turbined_m3s``
    or ``max_generation_mw`` above the hydro's own declared value in
    ``hydros.json`` (``generation.max_turbined_m3s`` /
    ``generation.max_generation_mw``), each column checked independently.
    Lowering is fine; every other column is untouched by this rule.

    Structurally inapplicable when *hydro_bounds* is ``None`` or carries
    neither guarded column — DECOMP writes only ``min_outflow_m3s`` there, so
    its exposure is nil by construction. That is reported as an explicit INFO
    finding, never a silent pass.
    """
    columns = (
        [c for c in _ENVELOPE_COLUMNS if c in hydro_bounds.column_names]
        if hydro_bounds is not None
        else []
    )

    if not columns:
        emit(
            Diagnostic(
                code="hydro-bounds-raising-not-applicable",
                severity=Severity.INFO,
                category=_CATEGORY,
                title="Hydro-bounds raising check (rule 43) not applicable",
                summary=(
                    "hydro_bounds carries neither max_turbined_m3s nor "
                    "max_generation_mw for this case, so cobre rule 43 (no row "
                    "may raise a hydro's declared capacity) has nothing to scan"
                ),
            ),
            logger=_LOG,
        )
        return

    declared = _hydro_declared(hydros)
    hydro_ids = hydro_bounds["hydro_id"].to_pylist()
    stage_ids = hydro_bounds["stage_id"].to_pylist()
    column_values = {c: hydro_bounds[c].to_pylist() for c in columns}

    violations: dict[tuple[int, str], list[tuple[int, float]]] = {}
    for i, hydro_id in enumerate(hydro_ids):
        entry = declared.get(hydro_id)
        if entry is None:
            continue
        stage_id = stage_ids[i]
        for column in columns:
            value = column_values[column][i]
            if value is None:
                continue
            declared_value = entry.get(column)
            if declared_value is None:
                continue
            if value > declared_value + _tolerance(declared_value):
                violations.setdefault((hydro_id, column), []).append((stage_id, value))

    if not violations:
        return

    rows: list[list[object]] = []
    for (hydro_id, column), occurrences in sorted(violations.items()):
        declared_value = declared[hydro_id][column]
        worst = max(value for _, value in occurrences)
        rows.append(
            [
                hydro_id,
                column,
                format_stage_ranges(stage for stage, _ in occurrences),
                round(declared_value, 6),
                round(worst, 6),
            ]
        )

    emit(
        Diagnostic(
            code="hydro-bounds-raises-declared-capacity",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"hydro_bounds raises declared capacity ({len(rows)} finding(s))",
            summary=(
                f"{len(rows)} (hydro, column) combination(s) have a hydro_bounds "
                "row that raises max_turbined_m3s or max_generation_mw above the "
                "plant's own declared value in system/hydros.json (cobre rule 43)"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Stages", "Declared", "Worst offending"],
                rows=rows,
                justify=["right", "left", "left", "right", "right"],
            ),
            remediation=(
                "Declare the plant at its final (post-uprate) value instead and "
                "add hydro_bounds rows tightening the earlier stages down to "
                "their true value."
            ),
        ),
        logger=_LOG,
    )


def check_unit_group_envelope(hydros: Mapping[str, object]) -> None:
    """Cobre rule 41 mirror: the sum of a hydro's unit-group maxima must not
    exceed the hydro's own declared value in ``hydros.json``, each of
    ``max_turbined_m3s`` / ``max_generation_mw`` checked independently, against
    the entity declaration only — never a per-stage ``hydro_bounds`` override.

    Always structurally applicable: both pipelines declare a non-empty
    ``unit_groups`` on every hydro (cobre requires it), so there is no
    "nothing to scan" case to report here.
    """
    raw_hydros = hydros.get("hydros")
    if not isinstance(raw_hydros, list):
        return

    rows: list[list[object]] = []
    for hydro in raw_hydros:
        if not isinstance(hydro, Mapping):
            continue
        hydro_id = hydro.get("id")
        generation = hydro.get("generation")
        groups = hydro.get("unit_groups")
        if not isinstance(hydro_id, int) or not isinstance(generation, Mapping):
            continue
        if not isinstance(groups, list):
            continue

        for column in _ENVELOPE_COLUMNS:
            declared_value = generation.get(column)
            if not isinstance(declared_value, int | float):
                continue
            declared_value = float(declared_value)

            group_sum = 0.0
            for group in groups:
                if not isinstance(group, Mapping):
                    continue
                group_value = group.get(column)
                if isinstance(group_value, int | float):
                    group_sum += float(group_value)

            if group_sum > declared_value + _tolerance(declared_value):
                rows.append(
                    [hydro_id, column, round(declared_value, 6), round(group_sum, 6)]
                )

    if not rows:
        return

    rows.sort(key=lambda r: (r[0], r[1]))
    emit(
        Diagnostic(
            code="hydro-unit-group-envelope-exceeded",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=(
                f"Unit-group maxima exceed the plant envelope ({len(rows)} finding(s))"
            ),
            summary=(
                f"{len(rows)} hydro/column combination(s) have unit groups whose "
                "max_turbined_m3s or max_generation_mw sums above the plant's "
                "own declared value in system/hydros.json (cobre rule 41)"
            ),
            table=DiagnosticTable(
                columns=["Hydro ID", "Column", "Declared", "Group sum"],
                rows=rows,
                justify=["right", "left", "right", "right"],
            ),
            remediation=(
                "Lower the unit groups' maxima so their sum does not exceed the "
                "plant's own declared value; the plant value is the envelope."
            ),
        ),
        logger=_LOG,
    )


def check_bound_row_uniqueness(families: Sequence[BoundFamily]) -> None:
    """Cobre rule 36 mirror: at most one row per ``(entity, stage, block,
    column)`` within a family, checked independently for every non-key column.

    ``block_id = None`` is a distinct key from ``Some(b)``: a stage-wide row
    and its per-block override rows are legal together — exactly what the
    ``line_bounds`` base-plus-per-block-override convention emits. Two rows
    setting *different* columns for the same ``(entity, stage, block)`` are
    also legal (sparse input), since the key is scoped per column.
    """
    findings: list[list[object]] = []
    for family in families:
        table = family.table
        if table is None or table.num_rows == 0:
            continue

        key_columns = {family.entity_column, "stage_id", "block_id"}
        value_columns = [c for c in table.column_names if c not in key_columns]
        if not value_columns:
            continue

        entity_ids = table[family.entity_column].to_pylist()
        stage_ids = table["stage_id"].to_pylist()
        block_ids = (
            table["block_id"].to_pylist()
            if "block_id" in table.column_names
            else [None] * table.num_rows
        )
        column_values = {c: table[c].to_pylist() for c in value_columns}

        seen: set[tuple[int, int, int | None, str]] = set()
        for i in range(table.num_rows):
            entity_id = entity_ids[i]
            stage_id = stage_ids[i]
            block_id = block_ids[i]
            for column in value_columns:
                value = column_values[column][i]
                if value is None:
                    continue
                key = (entity_id, stage_id, block_id, column)
                if key in seen:
                    findings.append(
                        [
                            family.label,
                            entity_id,
                            stage_id,
                            block_id if block_id is not None else "stage-wide",
                            column,
                        ]
                    )
                    continue
                seen.add(key)

    if not findings:
        return

    emit(
        Diagnostic(
            code="bound-row-duplicate",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"Duplicate bound rows ({len(findings)} finding(s))",
            summary=(
                f"{len(findings)} bound row(s) set the same column twice for the "
                "same (entity, stage, block) key (cobre rule 36)"
            ),
            table=DiagnosticTable(
                columns=["Family", "Entity ID", "Stage", "Block", "Column"],
                rows=findings,
            ),
            remediation=(
                "Remove the duplicate row or merge the overrides into a single row."
            ),
        ),
        logger=_LOG,
    )


def _stage_block_counts(stages: Mapping[str, object]) -> dict[int, int]:
    """``{stage_id: declared block count}`` from the ``stages.json`` document."""
    counts: dict[int, int] = {}
    raw_stages = stages.get("stages")
    if not isinstance(raw_stages, list):
        return counts
    for stage in raw_stages:
        if not isinstance(stage, Mapping):
            continue
        stage_id = stage.get("id")
        if not isinstance(stage_id, int):
            continue
        blocks = stage.get("blocks")
        counts[stage_id] = len(blocks) if isinstance(blocks, list) else 0
    return counts


def check_bound_block_id_range(
    stages: Mapping[str, object],
    families: Sequence[BoundFamily],
) -> None:
    """``block_id``-range mirror (matrix row 18): every non-null ``block_id`` in
    a bound-override row must index a block the row's stage actually declares
    in ``stages.json``. A row referencing a ``stage_id`` that is not a declared
    study stage is skipped without a finding — that is a stage-axis defect this
    rule does not own.
    """
    stage_block_counts = _stage_block_counts(stages)

    findings: list[list[object]] = []
    for family in families:
        table = family.table
        if table is None or table.num_rows == 0:
            continue
        if "block_id" not in table.column_names:
            continue

        entity_ids = table[family.entity_column].to_pylist()
        stage_ids = table["stage_id"].to_pylist()
        block_ids = table["block_id"].to_pylist()
        for entity_id, stage_id, block_id in zip(
            entity_ids, stage_ids, block_ids, strict=True
        ):
            if block_id is None:
                continue
            declared_count = stage_block_counts.get(stage_id)
            if declared_count is None:
                continue
            if block_id < 0 or block_id >= declared_count:
                findings.append(
                    [family.label, entity_id, stage_id, block_id, declared_count]
                )

    if not findings:
        return

    emit(
        Diagnostic(
            code="bound-block-id-out-of-range",
            severity=Severity.ERROR,
            category=_CATEGORY,
            title=f"block_id out of range ({len(findings)} finding(s))",
            summary=(
                f"{len(findings)} bound row(s) carry a block_id outside the "
                "range the row's own stage declares in stages.json"
            ),
            table=DiagnosticTable(
                columns=["Family", "Entity ID", "Stage", "block_id", "Declared blocks"],
                rows=findings,
                justify=["left", "right", "right", "right", "right"],
            ),
            remediation="Fix the block_id or the stage's declared block count.",
        ),
        logger=_LOG,
    )
