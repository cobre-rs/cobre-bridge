"""Byte-identical regression guard for the ``2000_completo`` example case.

``newave_rodada_2000_completo`` is an ``EX``-only case: it has **no**
``NE``-with-filling plants (``plants.filling_hydro_codes`` returns an empty
set), so every admission surface added across epic-02 must be *inert* for it.
This test pins that invariant the hard way — a **byte-identical** comparison of
the full conversion output against a committed hash manifest.

## How the baseline was produced

The committed manifest ``tests/golden/regression_2000_completo.sha256.json`` was
generated from the **pre-feature** code: the ``develop`` merge-base
(``f398ed6d``, the commit *before* epic-01), checked out into a ``git worktree``
and run against the same ``example/newave_rodada_2000_completo`` input via::

    BASE=$(git merge-base HEAD develop)
    git worktree add /tmp/ne-base "$BASE"
    PYTHONPATH=/tmp/ne-base/src .venv/bin/python -m cobre_bridge.cli \\
        convert newave example/newave_rodada_2000_completo /tmp/ne-base-out --force

Each emitted file's bytes were hashed with SHA-256 into a sorted
``{relative_path: sha256}`` mapping (excluding the volatile files below).

**Determinism was verified**: the base code was run twice and produced identical
per-file hashes for all hashed files (parquet footers included). If a future
output becomes non-deterministic run-to-run, this byte-identical strategy breaks
and must be revisited — do not silence the failure.

## Excluded (volatile provenance) files

``conversion_manifest.json`` embeds a git SHA, an ISO timestamp, and the
absolute output directory — all of which differ between the base run and any
later run — so it is **excluded** from the hashed set.  ``convert_newave_case``
(called here directly) does not even write that manifest; the exclusion is kept
explicit so the contract survives if the manifest (or any other
sha/timestamp/version-bearing side-file) is ever moved onto this code path.

The exclusion list is :data:`_EXCLUDED_FILES`.

## Regeneration recipe (only for an intentional, reviewed output change)

Re-run the base-vs-feature procedure above; once the *intended* diff is
confirmed, regenerate the manifest from the new feature output and review every
changed hash in the PR.  The manifest is NEVER silently overwritten on mismatch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cobre_bridge.pipeline import convert_newave_case

_EXAMPLE_CASE = Path("example/newave_rodada_2000_completo")
_GOLDEN_MANIFEST = (
    Path(__file__).parent / "golden" / "regression_2000_completo.sha256.json"
)

# Volatile provenance/log files whose bytes embed a git SHA, timestamp,
# version, or absolute path and therefore differ between the base baseline and
# any later conversion.  These are excluded from the hashed set.
_EXCLUDED_FILES: frozenset[str] = frozenset({"conversion_manifest.json"})


def _hash_tree(root: Path) -> dict[str, str]:
    """Return ``{relative_posix_path: sha256_hex}`` for every file under *root*.

    Files named in :data:`_EXCLUDED_FILES` are skipped.  The mapping is sorted
    by path so the comparison and any failure message are deterministic.
    """
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if rel in _EXCLUDED_FILES:
            continue
        out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(out.items()))


def _format_mismatch(expected: dict[str, str], actual: dict[str, str]) -> str:
    """Build a human-readable diff naming every file that differs."""
    missing = sorted(set(expected) - set(actual))
    added = sorted(set(actual) - set(expected))
    changed = sorted(
        rel for rel in expected.keys() & actual.keys() if expected[rel] != actual[rel]
    )
    lines: list[str] = ["2000_completo conversion is NOT byte-identical to baseline."]
    if changed:
        lines.append(f"  Changed bytes ({len(changed)}): {', '.join(changed)}")
    if missing:
        lines.append(f"  Missing files ({len(missing)}): {', '.join(missing)}")
    if added:
        lines.append(f"  Unexpected new files ({len(added)}): {', '.join(added)}")
    return "\n".join(lines)


def test_convert_2000_completo_byte_identical(tmp_path: Path) -> None:
    """The full ``2000_completo`` conversion must match the committed manifest.

    Proves epic-02's ``NE``-admission surfaces leave an ``EX``-only case
    byte-for-byte unperturbed.  Skips when the (gitignored) example case is not
    present in the checkout.
    """
    if not (_EXAMPLE_CASE / "caso.dat").exists():
        pytest.skip("example/newave_rodada_2000_completo not available")

    expected: dict[str, str] = json.loads(_GOLDEN_MANIFEST.read_text(encoding="utf-8"))

    dst = tmp_path / "out"
    convert_newave_case(_EXAMPLE_CASE, dst)
    actual = _hash_tree(dst)

    assert actual == expected, _format_mismatch(expected, actual)
