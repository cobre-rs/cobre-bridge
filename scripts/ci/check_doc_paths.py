#!/usr/bin/env python3
"""check_doc_paths.py — resolvable-path gate for shipped docs.

Every repo-relative path cited in README.md and docs/**/*.md must resolve
against the live tree (`.claude/rules/doc-integrity.md` §3.4). Citing a
gitignored dir (``plans/``) or a machine-local path (``~/git/...``) in a
shipped doc is a violation outright — no reader can resolve it.

CLAUDE.md is scanned in ADVISORY mode: it documents this checkout (so
``example/`` deck paths are legitimate there), and known drift is tracked in
the audit registry. Promote it to hard once its documented paths are clean.

Recognized citation shapes: backtick-quoted tokens that start with a known
top-level dir (``src/``, ``docs/``, ``scripts/``, ``tests/``, ``.github/``,
``.claude/``, ``plans/``) or name a known root file. Bare command examples and
external URLs are not paths and are ignored.

Exit codes: 0 = all hard-scope citations resolve; 1 = violations printed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _scan import REPO_ROOT, rel

TOKEN = re.compile(r"`([^`\s]+)`")
PATH_PREFIXES = ("src/", "docs/", "scripts/", "tests/", ".github/", ".claude/")
ROOT_FILES = {
    "README.md",
    "CHANGELOG.md",
    "CLAUDE.md",
    "pyproject.toml",
    "ruff.toml",
    "uv.lock",
}
# `cobre-bridge.toml` and `~/.config/...` are user-created locations the docs
# legitimately name (external contracts, not repo paths) — not checked.
# `plans/` is gitignored and `~/git/` is a developer machine — always dead
# for a reader.
ALWAYS_DEAD_PREFIXES = ("plans/", "~/git/")

HARD_FILES = ["README.md"]
HARD_GLOBS = ["docs/**/*.md"]
ADVISORY_FILES = ["CLAUDE.md"]


def looks_like_repo_path(token: str) -> bool:
    if token in ROOT_FILES:
        return True
    return token.startswith(PATH_PREFIXES) or token.startswith(ALWAYS_DEAD_PREFIXES)


def strip_anchor(token: str) -> str:
    return token.split("#", 1)[0].rstrip("/")


def scan(path: Path) -> list[str]:
    problems: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for token in TOKEN.findall(line):
            if not looks_like_repo_path(token):
                continue
            if token.startswith(ALWAYS_DEAD_PREFIXES):
                problems.append(
                    f"{rel(path)}:{lineno}: `{token}` — gitignored/machine-local; "
                    "no reader can resolve it"
                )
                continue
            cleaned = strip_anchor(token)
            if "*" in cleaned:
                if not any(REPO_ROOT.glob(cleaned)):
                    problems.append(
                        f"{rel(path)}:{lineno}: `{token}` glob matches nothing"
                    )
                continue
            if not (REPO_ROOT / cleaned).exists():
                problems.append(f"{rel(path)}:{lineno}: `{token}` does not resolve")
    return problems


def collect(names: list[str], globs: list[str] | None = None) -> list[Path]:
    files = [REPO_ROOT / name for name in names]
    for glob in globs or []:
        files.extend(sorted(REPO_ROOT.glob(glob)))
    return [f for f in files if f.is_file()]


def main() -> int:
    hard: list[str] = []
    for path in collect(HARD_FILES, HARD_GLOBS):
        hard.extend(scan(path))

    advisory: list[str] = []
    for path in collect(ADVISORY_FILES):
        advisory.extend(scan(path))

    if advisory:
        print(f"ADVISORY: {len(advisory)} unresolved citation(s) (never fails):")
        print("\n".join(f"  {hit}" for hit in advisory))

    if hard:
        print()
        print("FAIL: unresolvable path citations in shipped docs.")
        print()
        print("\n".join(hard))
        print()
        print(
            "Repoint the citation, or state the invariant instead of the path "
            "— .claude/rules/doc-integrity.md §3."
        )
        return 1

    print("OK: all shipped-doc path citations resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
