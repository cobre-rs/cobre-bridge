"""Standing audit against a guard on a retired or absent deck going silent.

A `skipif`-guarded test whose deck no longer exists anywhere runs **nowhere
and reports nothing** -- a dead skip that reads as a pass. This module scans
every sibling ``tests/test_*.py`` module's own source text for the deck
directory names its guards reference and checks them against two conditions:
no guard names a deck that was deliberately retired, and (dev-only) every
other guarded deck actually exists under ``example/``.

This module is itself tier 1 for its primary assertion: it imports no
``cobre``, and ``test_no_guard_references_a_retired_deck`` reads no file
under ``example/`` -- discovery is a plain ``pathlib`` + ``re`` scan over
test source text, never an import of the guarded modules.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_EXAMPLE_ROOT = Path("example")

# Decks retired because no surviving test guards on them; re-adding a guard
# on one of these must fail loudly instead of silently skipping forever.
_RETIRED_DECKS: frozenset[str] = frozenset(
    {
        "decomp-jul-26-rv3",
        "decomp-set-24-rv0",
        "decomp-set-24-rv0-small",
        "newave_rodada_2001_completo",
    }
)

# Captures the first path segment after `example/` in a `Path("example/...")`
# / `Path('example/...')` literal. `.` is excluded from the capture alongside
# the quote/slash/paren delimiters: without it, an illustrative literal
# inside a docstring or comment (e.g. `` `Path("example/...")` `` describing
# the convention itself) false-matches the ellipsis as a deck named "...",
# which then fails the tier-3 presence check below on a deck that never
# existed.
_DECK_PATTERN = re.compile(r'Path\(["\']example/([^"\'/).]+)')


def _discover_guarded_decks() -> dict[str, list[str]]:
    """Map each deck name referenced by a `Path("example/...")` literal in a
    sibling test module to the module filename(s) that reference it.

    Static text scan only -- never imports the guarded modules. Skips this
    module's own file so `_RETIRED_DECKS` and this docstring's example
    literal do not self-match.
    """
    self_name = Path(__file__).name
    discovered: dict[str, list[str]] = {}
    for module in sorted(Path(__file__).parent.glob("test_*.py")):
        if module.name == self_name:
            continue
        source = module.read_text(encoding="utf-8")
        for deck_name in _DECK_PATTERN.findall(source):
            discovered.setdefault(deck_name, []).append(module.name)
    return discovered


def test_no_guard_references_a_retired_deck() -> None:
    """Tier 1: no test module guards on a deck that was deliberately retired.

    Reads only test source text under `tests/` -- no file under `example/`.
    """
    discovered = _discover_guarded_decks()
    offending = set(discovered) & _RETIRED_DECKS
    assert not offending, "; ".join(
        f"{deck!r} referenced by {discovered[deck]}" for deck in sorted(offending)
    )


@pytest.mark.skipif(
    not _EXAMPLE_ROOT.exists(),
    reason="tier 3: requires the gitignored example/ deck root",
)
def test_every_non_retired_guarded_deck_is_present() -> None:
    """Tier 3: every surviving guarded deck actually exists under `example/`.

    Catches a guard on a deck that is neither retired nor actually present
    in a populated dev checkout.
    """
    discovered = _discover_guarded_decks()
    missing = [
        deck
        for deck in sorted(discovered)
        if deck not in _RETIRED_DECKS and not (_EXAMPLE_ROOT / deck).is_dir()
    ]
    assert not missing, f"guarded deck(s) not found under {_EXAMPLE_ROOT}: {missing}"
