"""Bootstrap the terminal cut-slot manifest from a 1-iteration cobre run.

The importer must author boundary cuts whose per-slot layout
(``entity_type``, ``entity_id``, ``subindex``) matches cobre's own terminal
pool exactly — cobre validates that identity on load. Rather than
re-implement cobre's slot-layout rules in Python (fragile, and it drifts
silently the moment cobre's own layout changes), this module runs cobre
itself for exactly one training iteration on the converted case, with
checkpointing on, and reads the emitted checkpoint's terminal
``entity_manifest`` + ``state_dimension`` back via
:func:`cobre.results.load_policy`. The mapper (epic 2's ticket-005) and the
writer (ticket-008) then author cuts against that manifest verbatim, so any
future cobre layout change breaks loudly — a manifest mismatch on load — not
silently.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

#: Number of trailing stderr lines kept in a subprocess failure message.
_STDERR_TAIL_LINES = 40


def ensure_writer_binding() -> None:
    """Raise if the installed ``cobre`` wheel lacks ``write_policy_checkpoint``.

    The importer's writer (ticket-008) needs
    ``cobre.write_policy_checkpoint`` to author a boundary-cut checkpoint;
    this is the epic's environment gate, callable standalone before any
    bootstrap/mapper/writer work runs.

    Raises
    ------
    RuntimeError
        Naming the missing binding and the remediation: reinstall the
        wheel from ``~/git/cobre`` (e.g. ``maturin develop --release -m
        ~/git/cobre/crates/cobre-python/Cargo.toml``). Raised both when
        ``cobre`` is not installed at all (caught as ``ModuleNotFoundError``)
        and when it is installed but predates ``write_policy_checkpoint``.
    """
    try:
        import cobre
    except ModuleNotFoundError:
        raise RuntimeError(
            "cobre is not installed; install/reinstall it from "
            "~/git/cobre, e.g. `VIRTUAL_ENV=<bridge>/.venv "
            "<bridge>/.venv/bin/maturin develop --release -m "
            "~/git/cobre/crates/cobre-python/Cargo.toml`"
        ) from None
    if not hasattr(cobre, "write_policy_checkpoint"):
        raise RuntimeError(
            "the installed cobre wheel does not expose "
            "write_policy_checkpoint; rebuild/reinstall it from "
            "~/git/cobre, e.g. `VIRTUAL_ENV=<bridge>/.venv "
            "<bridge>/.venv/bin/maturin develop --release -m "
            "~/git/cobre/crates/cobre-python/Cargo.toml`"
        )


@dataclass(frozen=True)
class TerminalManifest:
    """The terminal stage's per-slot entity identity and state dimension.

    ``entity_manifest`` is the tuple of per-slot dicts
    :func:`cobre.results.load_policy` returns verbatim (each carrying at
    least ``entity_type``, ``entity_id``, ``subindex``); ``state_dimension``
    is the terminal stage's own slot count, cross-checked against the
    checkpoint metadata's ``state_dimension`` in
    :func:`bootstrap_terminal_manifest`.
    """

    entity_manifest: tuple[dict[str, object], ...]
    state_dimension: int


def _set_iteration_limit_one(config_path: Path) -> None:
    """Overwrite ``config_path``'s stopping rule to a single iteration.

    Reads and rewrites the *copy* at ``config_path`` in place — the caller
    is responsible for ``config_path`` living under a scratch copy, never
    the original input case.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["stopping_rules"] = [{"type": "iteration_limit", "limit": 1}]
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _stderr_tail(stderr: str) -> str:
    """The trailing ``_STDERR_TAIL_LINES`` lines of ``stderr``, for error messages."""
    lines = stderr.splitlines()
    return "\n".join(lines[-_STDERR_TAIL_LINES:])


def bootstrap_terminal_manifest(
    case_dir: Path, cobre_bin: Path, *, work_dir: Path
) -> TerminalManifest:
    """Run cobre for 1 iteration on a copy of ``case_dir`` and read its terminal
    manifest.

    Copies ``case_dir`` into ``work_dir`` (never mutating the input case),
    overrides the copy's ``config.json`` stopping rule to a single training
    iteration, runs the local ``cobre_bin run`` binary with checkpointing on
    (the default), then reads the emitted checkpoint back via
    :func:`cobre.results.load_policy` and returns the terminal stage's (the
    entry whose ``stage_id`` is max) ``entity_manifest`` and
    ``state_dimension``.

    Depends only on ``cobre_bin run`` and ``cobre.results.load_policy`` —
    neither needs the ``write_policy_checkpoint`` binding, so this function
    does not call :func:`ensure_writer_binding`. The mapper/writer stages
    that consume the returned :class:`TerminalManifest` do need it; a caller
    composing the full bootstrap -> mapper -> writer pipeline calls
    :func:`ensure_writer_binding` once, up front.

    Raises
    ------
    RuntimeError
        If ``cobre_bin run`` exits nonzero (message includes the stderr
        tail), or if the loaded checkpoint has no stage cuts / an empty
        terminal ``entity_manifest`` / a terminal ``state_dimension`` that
        disagrees with the checkpoint metadata.
    """
    scratch_case = work_dir / "case"
    shutil.copytree(case_dir, scratch_case)
    _set_iteration_limit_one(scratch_case / "config.json")

    output_dir = work_dir / "output"
    completed = subprocess.run(
        [
            str(cobre_bin),
            "run",
            str(scratch_case),
            "--output",
            str(output_dir),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cobre run failed (exit {completed.returncode}) on {scratch_case}:\n"
            f"{_stderr_tail(completed.stderr)}"
        )

    import cobre

    policy = cobre.results.load_policy(output_dir, policy_subdir="policy")
    stage_cuts = policy["stage_cuts"]
    if not stage_cuts:
        raise RuntimeError(f"checkpoint at {output_dir} has no stage cuts")

    terminal = max(stage_cuts, key=lambda stage: stage["stage_id"])
    terminal_state_dimension = int(terminal["state_dimension"])
    metadata_state_dimension = int(policy["metadata"]["state_dimension"])
    if terminal_state_dimension != metadata_state_dimension:
        raise RuntimeError(
            f"terminal stage state_dimension {terminal_state_dimension} "
            "disagrees with checkpoint metadata state_dimension "
            f"{metadata_state_dimension} at {output_dir}"
        )

    entity_manifest = tuple(terminal["entity_manifest"])
    if not entity_manifest:
        raise RuntimeError(
            f"checkpoint at {output_dir} has an empty terminal entity_manifest"
        )

    return TerminalManifest(
        entity_manifest=entity_manifest,
        state_dimension=terminal_state_dimension,
    )
