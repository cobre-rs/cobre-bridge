"""Content-presence checks for the generated CLI reference (``docs/cli.md``).

The committed reference is produced verbatim by ``scripts/gen-cli-docs.sh``.
These tests assert the documented surface is present without coupling to the
exact byte formatting of Typer/Click output, which drifts across the
3.12/3.13/3.14 CI matrix.
"""

from __future__ import annotations

from pathlib import Path

_CLI_DOCS = Path(__file__).resolve().parents[1] / "docs" / "cli.md"

_REQUIRED_TOKENS: tuple[str, ...] = (
    "convert newave",
    "convert decomp",
    "check newave",
    "check decomp",
    "compare newave",
    "compare decomp",
    "dashboard",
    "--json",
    "--dry-run",
    "--open",
    "--log-file",
    "--validate",
    "--tolerance",
    "--format",
    "--out-dir",
    "--install-completion",
    "--version",
)


def test_cli_docs_file_exists_and_is_non_empty() -> None:
    assert _CLI_DOCS.is_file(), (
        f"{_CLI_DOCS} missing; regenerate with scripts/gen-cli-docs.sh"
    )
    assert _CLI_DOCS.stat().st_size > 0


def test_cli_docs_cover_every_command_and_new_flag() -> None:
    text = _CLI_DOCS.read_text(encoding="utf-8")
    missing = [tok for tok in _REQUIRED_TOKENS if tok not in text]
    assert not missing, (
        f"docs/cli.md is stale; regenerate with scripts/gen-cli-docs.sh "
        f"(missing: {missing})"
    )
