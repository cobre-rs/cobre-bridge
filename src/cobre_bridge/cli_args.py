"""Typed, frozen replacement for the CLI's per-command ``SimpleNamespace``.

Each Typer command builds one of these dataclasses instead of a duck-typed
``SimpleNamespace`` and hands it to its ``_run_*`` handler. Frozen construction
forbids the post-hoc mutation the untyped namespace allowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cobre_bridge.ui.console import get_console

if TYPE_CHECKING:
    from rich.console import Console


@dataclass(frozen=True)
class CommonArgs:
    """The five flags every command accepts."""

    json_output: bool
    verbose: int
    log_file: Path | None
    no_color: bool
    quiet: bool

    def out_console(self) -> Console:
        """Build the stdout console for this invocation."""
        return get_console(no_color=self.no_color)

    def err_console(self) -> Console:
        """Build the stderr console for this invocation."""
        return get_console(stderr=True, no_color=self.no_color)


@dataclass(frozen=True)
class ConvertArgs(CommonArgs):
    """Options for ``convert newave`` / ``convert decomp``."""

    src: Path
    dst: Path
    validate: bool
    force: bool
    diagnostics_json: Path | None
    dry_run: bool
    no_fcf: bool = False


@dataclass(frozen=True)
class CompareArgs(CommonArgs):
    """Options for ``compare newave`` / ``compare decomp``.

    ``source_dir`` generalizes the per-command ``newave_dir``/``decomp_dir``
    argument to the name the comparison manifest already treats generically.
    """

    source_dir: Path
    cobre_output_dir: Path
    tolerance: float | None
    format: list[str] | None
    out_dir: Path | None


@dataclass(frozen=True)
class CheckArgs(CommonArgs):
    """Options for ``check newave``."""

    src: Path


@dataclass(frozen=True)
class DashboardArgs(CommonArgs):
    """Options for ``dashboard``."""

    case_dir: Path
    output: Path | None
    open_browser: bool
