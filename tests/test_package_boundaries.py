"""Boundary guard: no `decomp/*` module imports an underscore-private name
from `converters/*` or `pipeline` (`.claude/rules/bridge.md` §1).

Tier-1: pure ``ast`` parsing of the source tree, no ``cobre``/``idecomp``
import, so this collects and runs even in a cobre-free environment.
"""

from __future__ import annotations

import ast
from pathlib import Path

_DECOMP_DIR = Path(__file__).resolve().parent.parent / "src" / "cobre_bridge" / "decomp"

# ticket-027 (physics/infra promotion) promoted every pending name above to a
# public home; no decomp/* module imports a converters/*/pipeline underscore
# name any more. Keep this empty — do not add to it (see .claude/rules/bridge.md
# §1: promote to a public home instead of reaching for a private one).
_PENDING_PROMOTION_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


def _is_guarded_module(module: str) -> bool:
    return (
        module.startswith("cobre_bridge.converters")
        or module == "cobre_bridge.pipeline"
    )


def _underscore_cross_imports(path: Path) -> list[tuple[str, str]]:
    """Return every ``(module, name)`` underscore import `path` takes from a
    guarded module (``converters/*`` or ``pipeline``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if not _is_guarded_module(node.module):
            continue
        for alias in node.names:
            if alias.name.startswith("_"):
                found.append((node.module, alias.name))
    return found


def _all_cross_imports() -> list[tuple[Path, str, str]]:
    return [
        (path, module, name)
        for path in sorted(_DECOMP_DIR.rglob("*.py"))
        for module, name in _underscore_cross_imports(path)
    ]


def test_no_unallowlisted_underscore_cross_imports() -> None:
    """Every `decomp/*` underscore import of a `converters/*`/`pipeline` name
    is either gone or on the ticket-027-pending allowlist above — this is the
    standing enforcement of never importing an underscore-private name across
    the `converters/` <-> `decomp/` boundary."""
    offenders = [
        (str(path), module, name)
        for path, module, name in _all_cross_imports()
        if (module, name) not in _PENDING_PROMOTION_ALLOWLIST
    ]
    assert offenders == []


def test_allowlist_has_no_stale_entries() -> None:
    """An allowlist entry with no matching import left is dead weight that
    would silently mask a real regression if ``ticket-027`` only partly lands;
    delete it from the allowlist in the same change that removes the import."""
    present = {(module, name) for _, module, name in _all_cross_imports()}
    stale = _PENDING_PROMOTION_ALLOWLIST - present
    assert stale == set()
