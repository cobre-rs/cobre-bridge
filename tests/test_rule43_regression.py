"""Regression guard for cobre 0.13 rule 43 on freshly converted decks
(epic-04, ticket-017; depends on ticket-016's ``emission_checks`` mirror).

Cobre rule 43 — ``check_bound_raises_declared_capacity`` in
``cobre-io/src/validation/semantic/block_bounds.rs`` — rejects any
``hydro_bounds`` row whose ``max_turbined_m3s`` or ``max_generation_mw``
exceeds the plant's own declared value in ``system/hydros.json``
(``generation.max_turbined_m3s`` / ``generation.max_generation_mw``).
Lowering a stage's cap below the plant's envelope is legal; a stage row
that raises it above the plant's own declared ceiling is not, and cobre
refuses to load the case at all.

This module exists because that rule was violated for real, not
hypothetically. Before ticket-015b, the NEWAVE converter declared
``generation.max_turbined_m3s`` at the single reference-head value while
``convert_turbined_bounds_head_corrected`` emitted per-stage rows at the
(higher) head-corrected cap for any plant carrying a CFUGA/CMONT/
VOLREF_SAZ override — a fresh conversion of ``example/newave_rodada`` had
2478 rule-43-raising rows, and cobre rejected the case outright. Ticket
015b fixed this by declaring the plant at ``max(reference, per-stage
envelope)``, so every per-stage row only ever tightens the bound
downward. ``emission_checks.check_hydro_bounds_no_raising`` mirrors the
rule as a diagnostic (non-fatal, courtesy-only) inside the pipeline; this
module is the executable pass/fail guard on real decks that ticket-015b's
fix must keep satisfying — if a future converter change reintroduces even
one raising row, this test fails.

The scan logic here is written independently of
``cobre_bridge.core.emission_checks`` (parses ``hydros.json``/
``hydro_bounds.parquet`` itself, does not import that module's private
helpers) so this test does not simply assert that the courtesy mirror
agrees with itself; it independently re-derives the same cobre-side
relative tolerance cobre's ``ENVELOPE_TOLERANCE`` uses (see
``emission_checks._tolerance`` for the mirrored form).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq
import pytest

#: Both columns cobre rule 43 guards, checked independently (matches
#: ``emission_checks._ENVELOPE_COLUMNS``).
_GUARDED_COLUMNS: tuple[str, ...] = ("max_turbined_m3s", "max_generation_mw")

#: Mirrors cobre-io's ``ENVELOPE_TOLERANCE`` (also independently mirrored by
#: ``cobre_bridge.core.emission_checks._tolerance``) — a *relative* tolerance, so
#: a plant declared near zero is not held to an absolute epsilon that would
#: swallow a real violation, while a plant declared at, say, 1e6 does not
#: false-fire on float noise.
_ENVELOPE_TOLERANCE = 1e-9

#: AC #2: the row count scanned must be non-trivial, or an empty/near-empty
#: scan could pass vacuously and this guard would silently stop guarding.
_MIN_EXPECTED_ROWS = 1000

#: AC #2, per-column strengthening: the aggregate row count above says
#: nothing about any one guarded column, which can be sparse (e.g.
#: ``max_generation_mw`` is only ever populated for a dead-volume filling
#: plant's per-stage override, a handful of rows even on a real deck) — a
#: column reporting ``present=True`` with (near-)zero non-null values would
#: otherwise satisfy ``raising_count == 0`` vacuously. This floor only rules
#: out that degenerate 0-or-1-value case; it deliberately stays far below
#: ``_MIN_EXPECTED_ROWS`` so it does not demand rich coverage from a column
#: that is intrinsically rare by construction.
_MIN_EXPECTED_NON_NULL_PER_COLUMN = 1

# Real, gitignored decks (see example/README.md — local-only, not part of
# the checked-in fixture set). Both dirs are confirmed present in this
# checkout; tests still skip cleanly (matching the convention already used
# by TestConvertLineBoundsRealDeckFidelity in test_entity_conversion.py) so
# that CI, which does not have example/, does not fail on their absence.
_NEWAVE_RODADA = Path("example/newave_rodada")

#: Per-deck expected set of guarded columns actually present in the
#: freshly converted ``hydro_bounds.parquet`` — pinned so that presence
#: itself is asserted explicitly (AC #1: "assert that explicitly rather
#: than skipping silently") instead of a bare ``if column in table:``
#: quietly doing nothing when a column happens to be absent for one deck.
_EXPECTED_PRESENT_COLUMNS: dict[Path, frozenset[str]] = {
    _NEWAVE_RODADA: frozenset({"max_turbined_m3s"}),
}


def _tolerance(declared: float) -> float:
    """``ENVELOPE_TOLERANCE * max(|declared|, 1.0)`` — cobre's relative
    envelope tolerance, mirrored independently from
    ``emission_checks._tolerance`` rather than imported from it."""
    return _ENVELOPE_TOLERANCE * max(abs(declared), 1.0)


def _declared_envelope(
    hydros_json: Mapping[str, object],
) -> dict[int, dict[str, float]]:
    """``{hydro_id: {column: declared value}}`` from a ``hydros.json`` document.

    Reads ``hydro["generation"][column]`` — the declared value lives there,
    never at the top level of the hydro entry.
    """
    declared: dict[int, dict[str, float]] = {}
    for hydro in hydros_json["hydros"]:
        generation = hydro.get("generation")
        if not isinstance(generation, Mapping):
            continue
        entry = {
            column: float(generation[column])
            for column in _GUARDED_COLUMNS
            if isinstance(generation.get(column), int | float)
        }
        if entry:
            declared[hydro["id"]] = entry
    return declared


@dataclass(frozen=True)
class ColumnScan:
    """Rule-43 scan result for one guarded column against one deck.

    ``raising_hydro_ids`` names *which* hydro ids raised the column (deduped
    across every stage/block row) — ``raising_count`` is the row count,
    ``len(raising_hydro_ids)`` the distinct-plant count, named in a raising
    assertion's failure message so a regression identifies its plant
    without a second debugging pass.
    """

    present: bool
    raising_count: int
    non_null_count: int
    raising_hydro_ids: frozenset[int] = frozenset()


def _scan_hydro_bounds(
    hydros_json: Mapping[str, object], hydro_bounds_path: Path
) -> tuple[int, dict[str, ColumnScan]]:
    """Scan ``hydro_bounds.parquet`` for rows that raise a guarded column
    above the owning hydro's declared value in *hydros_json*.

    Returns ``(total rows in the table, {column: ColumnScan})`` — one entry
    per column in :data:`_GUARDED_COLUMNS`, always present in the returned
    dict even when the column itself is absent from the table (``present``
    is ``False``, ``raising_count`` and ``non_null_count`` are both ``0`` in
    that case, since there is nothing to raise or to scan).
    """
    table = pq.read_table(hydro_bounds_path)
    declared = _declared_envelope(hydros_json)
    hydro_ids = table["hydro_id"].to_pylist()

    scans: dict[str, ColumnScan] = {}
    for column in _GUARDED_COLUMNS:
        if column not in table.column_names:
            scans[column] = ColumnScan(present=False, raising_count=0, non_null_count=0)
            continue

        values = table[column].to_pylist()
        non_null_count = table.num_rows - table[column].null_count
        raising_count = 0
        raising_hydro_ids: set[int] = set()
        for row_index, hydro_id in enumerate(hydro_ids):
            entry = declared.get(hydro_id)
            if entry is None:
                continue
            value = values[row_index]
            if value is None:
                continue
            declared_value = entry.get(column)
            if declared_value is None:
                continue
            if value > declared_value + _tolerance(declared_value):
                raising_count += 1
                raising_hydro_ids.add(hydro_id)
        scans[column] = ColumnScan(
            present=True,
            raising_count=raising_count,
            non_null_count=non_null_count,
            raising_hydro_ids=frozenset(raising_hydro_ids),
        )

    return table.num_rows, scans


class TestNewaveRule43NoRaising:
    """AC #1/#2/#4/#5: a fresh NEWAVE conversion has zero rule-43-raising
    ``hydro_bounds`` rows, on a non-trivial row count, both guarded columns
    checked independently at cobre's relative tolerance.
    """

    @pytest.mark.parametrize("deck", [_NEWAVE_RODADA], ids=lambda p: p.name)
    def test_fresh_conversion_has_zero_raising_rows(
        self, deck: Path, tmp_path: Path
    ) -> None:
        if not deck.exists():
            pytest.skip(f"{deck} not present (example/ is local-only, gitignored)")

        from cobre_bridge.pipeline import convert_newave_case

        dst = tmp_path / deck.name
        convert_newave_case(deck, dst)

        hydros_json = json.loads((dst / "system" / "hydros.json").read_text())
        rows_scanned, scans = _scan_hydro_bounds(
            hydros_json, dst / "constraints" / "hydro_bounds.parquet"
        )

        # AC #2: guard against a vacuous scan.
        assert rows_scanned > _MIN_EXPECTED_ROWS, (
            f"only {rows_scanned} hydro_bounds row(s) scanned for {deck} — "
            "too few to exercise rule 43; a near-empty scan would pass "
            "vacuously, which is worse than no test at all"
        )

        # AC #1 (explicit-not-silent arm): pin exactly which guarded columns
        # this deck's hydro_bounds carries. A column's absence is asserted,
        # not merely skipped — e.g. newave_rodada carries no
        # max_generation_mw at all, which is fine, but that fact is checked
        # here rather than passed over.
        present_columns = frozenset(
            column for column, scan in scans.items() if scan.present
        )
        assert present_columns == _EXPECTED_PRESENT_COLUMNS[deck], (
            f"{deck} now exposes guarded column(s) "
            f"{present_columns - _EXPECTED_PRESENT_COLUMNS[deck]} not previously "
            f"present, or dropped {_EXPECTED_PRESENT_COLUMNS[deck] - present_columns} "
            "that used to be present — update this test's expectation if the "
            "schema change is intentional, otherwise investigate a regression"
        )

        # AC #1 (the actual rule-43 assertion), each present column checked
        # independently.
        for column in present_columns:
            scan = scans[column]

            # AC #2, per-column strengthening: a column reporting present=True
            # with (near-)zero non-null values would satisfy raising_count==0
            # vacuously — assert real per-column coverage, not just a
            # non-trivial aggregate row count.
            assert scan.non_null_count > _MIN_EXPECTED_NON_NULL_PER_COLUMN, (
                f"only {scan.non_null_count} non-null {column} value(s) in "
                f"hydro_bounds for {deck} — too few to exercise rule 43 for "
                "this column specifically; a (near-)empty column would pass "
                "vacuously even though the aggregate row count is non-trivial"
            )

            assert scan.raising_count == 0, (
                f"{scan.raising_count} hydro_bounds row(s) raise {column} above "
                f"the plant's own declared value on {deck} (cobre rule 43) — "
                "this is exactly the regression ticket-015b fixed"
            )
