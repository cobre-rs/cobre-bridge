"""Tier-3 dev-only probe: measure the special-constraint census on the real
rv3 deck.

Guarded by ``example/decomp-jul-26-rv3`` presence — CI has neither the
gitignored deck nor the solver binary (see the project's 3-tier test
convention), so this module is collected and skipped everywhere else. No
``cobre`` import: this is a pure reader probe over ``idecomp``'s parsed
registers, needing no wheel or solver.

The exact census (per-family counts, the ``to_bounds``/``to_generic`` split)
is an expected order of magnitude, not a locked oracle yet — see the plan.
This probe asserts only the one property that needs no magic constant (no RE
constraint is silently dropped) plus a loose sanity floor, and prints the
measured numbers so they can be read off a dev run and locked later.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from idecomp.decomp import Dadger

from cobre_bridge.decomp.constraint_registers import _df, read_constraints
from cobre_bridge.decomp.pipeline import discover_decomp_files

_RV3 = Path("example/decomp-jul-26-rv3")
pytestmark = pytest.mark.skipif(
    not (_RV3 / "caso.dat").is_file(), reason="rv3 deck not present"
)


def _re_declared_ids(dadger: Dadger) -> set[int]:
    """Distinct RE ids named by at least one of ``fu``/``ft``/``fi``.

    Reads each register the same way ``constraint_registers`` itself does
    (via its ``_df`` accessor) and collects ``codigo_restricao`` — the
    no-silent-drop property this probe checks is that every id gathered here
    surfaces as one ``RE`` record.
    """
    ids: set[int] = set()
    for attr in ("fu", "ft", "fi"):
        frame = _df(dadger, attr)
        if frame is None:
            continue
        ids.update(int(cid) for cid in frame["codigo_restricao"])
    return ids


def test_read_constraints_re_family_no_silent_drop_on_rv3() -> None:
    """RE record count equals the declared-id count; census prints for the log."""
    files = discover_decomp_files(_RV3)
    dadger = Dadger.read(str(files.dadger))
    # Dadger.read()'s stub return type is the base RegisterFile (same
    # pre-existing idecomp stub gap pipeline.py/fcf/__init__.py hit on this
    # identical call).
    census = read_constraints(dadger)  # type: ignore[arg-type]

    declared_ids = _re_declared_ids(dadger)  # type: ignore[arg-type]
    assert len(census.by_family["RE"]) == len(declared_ids)

    per_family = {fam: len(recs) for fam, recs in census.by_family.items()}
    print(f"rv3 constraint census by family: {per_family}")
    print(
        f"rv3 lowering split: to_bounds={len(census.to_bounds)} "
        f"to_generic={len(census.to_generic)}"
    )

    # Loose sanity floor only — the exact split is locked into the plan from
    # this probe's printed output, not hard-coded here.
    assert len(census.by_family["RE"]) >= 90
    assert len(census.by_family["HE"]) > 0
