#!/usr/bin/env python3
"""check_comment_refs.py — un-rottable-reference gate for shipped prose.

Enforces `.claude/rules/comments.md` N3: a comment or docstring in src/ must
reference only things that cannot rot for a reader without this checkout — a
symbol, a named test, or a stable external anchor (the cobre book, a schema
name, a published reference manual `§`). Everything below rots for a reader
who has only the pip-installed package.

HARD (exit 1):
  * source-file line references — this repo's ``file.py:NNN`` and cobre's
    ``file.rs:NNN`` / ``file.rs ~NN`` alike (drift on every edit above the
    line). A ``file.rs::symbol`` anchor is durable and deliberately allowed.
  * the literal token ``MEMORY.md`` (agent memory, unresolvable for readers),
  * ``.claude/`` paths outside ``.claude/rules/`` (private tooling; the
    ``.claude/rules/*`` contract-mirror class is allowed),
  * machine-local ``~/git/...`` paths and gitignored ``plans/...`` paths,
  * repo-relative paths into the cobre source tree (``crates/...``,
    ``cobre-io/src/...``): a pip reader has no cobre checkout — cite the cobre
    symbol or the cobre book instead,
  * the bridge's own design-doc section refs (``design §5``): those docs live
    in gitignored ``plans/`` — inline the durable content or name a shipped
    ``docs/`` path.

Exit codes: 0 = no violations; 1 = violations (details printed).
"""

from __future__ import annotations

import re
import sys

from _scan import SRC_ROOT, iter_prose, iter_py_files, rel

HARD_PATTERNS = [
    # Source-file line references drift on every edit above the line — this
    # repo's .py and cobre's .rs alike. A ``file.rs::symbol`` anchor carries no
    # digit after the colons, so it is deliberately not matched.
    ("source-line-ref", re.compile(r"\b[A-Za-z0-9_./-]+\.(?:py|rs)\s*[:~]\s*[0-9]+")),
    ("memory-md", re.compile(r"MEMORY\.md")),
    ("claude-path", re.compile(r"\.claude/(?!rules/)[A-Za-z0-9_./-]+")),
    ("machine-local", re.compile(r"~/git/[A-Za-z0-9_./-]+")),
    ("plans-path", re.compile(r"(?<![\w/-])plans/[A-Za-z0-9_./-]+")),
    # Repo-relative paths into the cobre source tree: unresolvable for a pip or
    # GitHub reader of this package.
    (
        "cross-repo-path",
        re.compile(
            r"\b(?:cobre/)?crates/[\w./-]+|\bcobre-(?:io|sddp|core|python)/src/[\w./-]+"
        ),
    ),
    # The bridge's own design docs live in gitignored plans/; a bare "design §N"
    # section pointer is dead for a reader without that checkout.
    ("internal-design-ref", re.compile(r"\bdesign[ \-]*§")),
]


def main() -> int:
    hard: list[str] = []
    for path in iter_py_files(SRC_ROOT):
        for lineno, text in iter_prose(path):
            for offset, line in enumerate(text.splitlines()):
                where = f"{rel(path)}:{lineno + offset}"
                for tag, pattern in HARD_PATTERNS:
                    for match in pattern.findall(line):
                        hard.append(f"{where}: [{tag}] {match}")

    if hard:
        print("FAIL: un-rottable-reference violations in src/ prose.")
        print()
        print("\n".join(hard))
        print()
        print(
            "Reference by symbol, named test, or stable external anchor; strip "
            ":NNN line numbers and cross-repo paths — .claude/rules/comments.md N3."
        )
        return 1

    print("OK: no hard rot-ref violations in src/ prose.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
