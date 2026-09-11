"""Regression guard: every boundary-FCF test module collects without cobre.

Every ``tests/decomp/test_fcf_*.py`` module must collect in a cobre-free
environment: a module-top ``import cobre`` breaks collection everywhere
regardless of skip markers. Nothing in the test runner enforces that, so this
module scans each FCF test module's own source text for one. It is itself a
plain ``pathlib`` + ``re`` scan with no ``cobre`` import.
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
    """Every ``tests/decomp/test_fcf_*.py`` module, sorted for a stable order."""
    return sorted(_TESTS_DIR.glob("test_fcf_*.py"))


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
    assert modules, f"no test_fcf_*.py modules found under {_TESTS_DIR}"

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
