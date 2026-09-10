"""Standing audit against a guard on a retired or absent deck going silent.

A `skipif`-guarded test whose deck no longer exists anywhere runs **nowhere
and reports nothing** -- a dead skip that reads as a pass. This module scans
every sibling ``tests/**/test_*.py`` module's own source text for the deck
directory names its guards reference and checks them against two conditions:
no guard names a deck that was deliberately retired, and (dev-only) every
other guarded deck actually exists under ``example/``. A third scan makes the
three-tier discipline (`.claude/rules/testing.md`) itself a tier-1 assertion:
every module referencing a ``Path("example/...")`` deck literal must carry a
tier-3 guard.

This module is itself tier 1 for its primary assertions: it imports no
``cobre``, and neither `test_no_guard_references_a_retired_deck` nor
`test_every_example_reference_carries_a_tier3_guard` reads any file under
``example/`` -- discovery is a plain ``pathlib`` + ``re`` scan over test
source text, never an import of the guarded modules.
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

# Either the decorator form (`@pytest.mark.skipif`) or the runtime-guard form
# (a bare `pytest.skip(...)` call, always reached through an `if not
# deck.exists():` check) counts as a tier-3 guard -- both are load-bearing
# conventions in this repo (the latter predates `skipif` adoption in
# `tests/newave/test_convert_network.py`'s `TestConvertLineBoundsRealDeckFidelity`
# and `tests/newave/test_rule43_regression.py`'s
# `TestNewaveRule43NoRaising`, which document deferring to it deliberately).
_TIER3_GUARD_PATTERN = re.compile(r"skipif|pytest\.skip\(")


def _iter_sibling_test_modules() -> list[Path]:
    """Every ``tests/**/test_*.py`` module except this file itself, sorted.

    Recurses the whole mirrored tree -- a plain ``.glob`` here would only see
    this file's own directory and silently scan nothing, the exact
    dead-guard failure mode this module exists to catch. Skips this module's
    own file so `_RETIRED_DECKS` and this docstring's example literal do not
    self-match.
    """
    self_name = Path(__file__).name
    return [
        module
        for module in sorted(Path(__file__).parent.rglob("test_*.py"))
        if module.name != self_name
    ]


def _discover_guarded_decks() -> dict[str, list[str]]:
    """Map each deck name referenced by a `Path("example/...")` literal in a
    sibling test module to the module filename(s) that reference it.

    Static text scan only -- never imports the guarded modules.
    """
    discovered: dict[str, list[str]] = {}
    for module in _iter_sibling_test_modules():
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


def test_every_example_reference_carries_a_tier3_guard() -> None:
    """Tier 1: the 3-tier discipline (`.claude/rules/testing.md`) as a test.

    Any module holding a `Path("example/...")` deck literal must also carry
    a tier-3 guard (`skipif`, or the equivalent runtime `pytest.skip(...)`)
    -- a real-deck read with no guard would fail every CI job outright, and
    a guard that silently vanished along with the guarding text would let
    the read through unguarded. Reads only test source text under `tests/`
    -- no file under `example/`, no `cobre` import.
    """
    offenders = [
        module.name
        for module in _iter_sibling_test_modules()
        if _DECK_PATTERN.search(source := module.read_text(encoding="utf-8"))
        and not _TIER3_GUARD_PATTERN.search(source)
    ]
    assert not offenders, (
        'module(s) reference a `Path("example/...")` deck with no tier-3 '
        f"guard (`skipif` / `pytest.skip(...)`): {offenders}"
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
