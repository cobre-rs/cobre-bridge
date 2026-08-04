"""Regression guard for the boundary-FCF suite's deck-independence convention.

Epic 02 of ``plans/decomp-fcf-ci-hardening`` (see ``CLAUDE.md``'s "boundary-FCF
test tiers" subsection) drove every ``tests/test_decomp_fcf_*.py`` module to
collect without the optional ``cobre-python`` wheel and without the local,
gitignored ``example/`` decks. Nothing in the test runner enforces that going
forward, so this module is a lightweight source-scan guard against
regression: it reads each FCF test module's own text (never the decks it may
reference) and asserts the two conditions the epic established.

This module is itself tier 1: a plain ``pathlib`` + ``re`` scan over test
source text, no ``import cobre`` anywhere and no ``example/`` path read, so
it collects and runs in every CI job.
"""

from __future__ import annotations

import re
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
# Matches any column-0 import of the `cobre` package itself — `import
# cobre`, `import cobre as ...`, `import cobre.submodule`, or `from cobre
# import ...` — including a trailing comment. The `\b` after `cobre`
# excludes an unrelated package sharing the prefix (e.g. this project's own
# `cobre_bridge`), and the `^` anchor (MULTILINE) excludes an indented
# call-site import.
_TOP_LEVEL_COBRE_IMPORT = re.compile(r"^(?:import cobre\b|from cobre\b)", re.MULTILINE)


def _fcf_test_modules() -> list[Path]:
    """Every ``tests/test_decomp_fcf_*.py`` module, sorted for a stable order."""
    return sorted(_TESTS_DIR.glob("test_decomp_fcf_*.py"))


def test_fcf_test_modules_have_no_top_level_cobre_import() -> None:
    """No FCF test module blocks cobre-free collection with a module-top import.

    A call-site import of ``cobre`` (inside a function/test body, always
    indented) is fine — every tier-2/3 test defers it there. Only a
    column-0 import of the ``cobre`` package itself — ``import cobre``,
    ``import cobre as ...``, ``import cobre.submodule``, or ``from cobre
    import ...`` — which pytest would execute at collection time regardless
    of markers, is disallowed.
    """
    modules = _fcf_test_modules()
    assert modules, f"no test_decomp_fcf_*.py modules found under {_TESTS_DIR}"

    offenders = [
        module.name
        for module in modules
        if _TOP_LEVEL_COBRE_IMPORT.search(module.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        "module-top `import cobre`/`from cobre import ...` blocks cobre-free "
        f"collection in: {offenders}; move the import into a call site or "
        "test body"
    )


def test_fcf_example_reads_are_skipif_guarded() -> None:
    """Every FCF test module reading ``example/`` also carries a skip guard.

    A coarse, module-level check — it does not verify the guard sits next to
    every individual ``Path("example/...")`` literal, only that a module
    referencing one also contains a ``skipif`` marker somewhere, so a bare,
    unconditional real-deck read cannot land uncaught.
    """
    modules = _fcf_test_modules()
    assert modules, f"no test_decomp_fcf_*.py modules found under {_TESTS_DIR}"

    offenders = [
        module.name
        for module in modules
        if (text := module.read_text(encoding="utf-8"))
        and 'Path("example/' in text
        and "skipif" not in text
    ]
    assert not offenders, (
        f"module(s) read a real `example/` deck with no `skipif` guard: {offenders}"
    )
