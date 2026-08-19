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

The run is driven **in-process** through :func:`cobre.run.run` (the same
cobre-python wheel the writer and reader already require), not a subprocess
``cobre`` binary: one dependency, no binary/wheel version skew, and no
``--cobre-bin`` path to resolve. ``cobre.run.run``'s ``config_overrides``
deep-merges an override into ``config.json`` **in memory** (it never mutates
the case), so no scratch copy is needed, and its ``on_iteration`` callback
stops the run cooperatively after the first iteration boundary — the terminal
state-vector layout is structural, present from iteration one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def ensure_writer_binding() -> None:
    """Raise if the installed ``cobre`` wheel lacks ``write_policy_checkpoint``.

    The importer's writer (ticket-008) needs
    ``cobre.write_policy_checkpoint`` to author a boundary-cut checkpoint;
    this is the epic's environment gate, callable standalone before any
    bootstrap/mapper/writer work runs.

    Raises
    ------
    RuntimeError
        With a self-contained, end-user-facing remediation (reinstall/upgrade
        the required ``cobre-python`` dependency via ``pip``, or re-run with
        ``--no-fcf``). Raised both when ``cobre`` is not installed at all
        (caught as ``ModuleNotFoundError``) and when it is installed but
        predates ``write_policy_checkpoint``.
    """
    try:
        import cobre
    except ModuleNotFoundError:
        raise RuntimeError(
            "cobre is not installed. cobre-python is a required dependency of "
            "cobre-bridge — install it (for example: pip install cobre-python), "
            "or reinstall cobre-bridge. To convert without the boundary "
            "cost-to-go function, re-run with --no-fcf."
        ) from None
    if not hasattr(cobre, "write_policy_checkpoint"):
        raise RuntimeError(
            "The installed cobre package is too old for the boundary cost-to-go "
            "import (it does not provide the policy checkpoint writer). Upgrade "
            "it (for example: pip install --upgrade cobre-python), or reinstall "
            "cobre-bridge. To convert without the boundary cost-to-go function, "
            "re-run with --no-fcf."
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


def bootstrap_terminal_manifest(case_dir: Path, *, work_dir: Path) -> TerminalManifest:
    """Run cobre for 1 iteration on ``case_dir`` and read its terminal manifest.

    Trains ``case_dir`` in-process via :func:`cobre.run.run`, writing the
    checkpoint under ``work_dir`` (``case_dir`` is never mutated: the
    single-iteration cap is applied through ``config_overrides``'
    in-memory deep-merge, and the ``on_iteration`` callback requests a
    cooperative stop at the first iteration boundary). Simulation is skipped.

    The bootstrap reserves no inflow-lag depth explicitly: cobre sizes the
    ``HydroInflowLag`` state block from the case's own PAR(p) model order
    (``resolve_state_layout``), the same autoregressive structure the DECOMP
    boundary cuts' ``pi_qafl`` terms derive from, so the terminal manifest
    already carries the slots the mapper places those terms onto. (The former
    ``state_space.inflow_lag_depth`` config override was redundant with that
    PAR-order sizing and is rejected outright by cobre >= 0.14, which removed
    the ``state_space`` config field.)
    It then reads the emitted checkpoint back via
    :func:`cobre.results.load_policy` and returns the terminal stage's (the
    entry whose ``stage_id`` is max) ``entity_manifest`` and
    ``state_dimension``.

    Depends only on ``cobre.run.run`` and ``cobre.results.load_policy`` —
    neither needs the ``write_policy_checkpoint`` binding, so this function
    does not call :func:`ensure_writer_binding`. The mapper/writer stages
    that consume the returned :class:`TerminalManifest` do need it; a caller
    composing the full bootstrap -> mapper -> writer pipeline calls
    :func:`ensure_writer_binding` once, up front.

    Raises
    ------
    RuntimeError
        If the loaded checkpoint has no stage cuts or an empty terminal
        ``entity_manifest``. A failure inside the run itself propagates as
        ``cobre.run.run``'s own exception.
    """
    import cobre

    output_dir = work_dir / "output"
    config_overrides: dict[str, object] = {
        "training.stopping_rules": [{"type": "iteration_limit", "limit": 1}]
    }
    cobre.run.run(
        str(case_dir),
        output_dir=str(output_dir),
        skip_simulation=True,
        config_overrides=config_overrides,
        on_iteration=lambda _info: True,
    )

    policy = cobre.results.load_policy(output_dir, policy_subdir="policy")
    stage_cuts = policy["stage_cuts"]
    if not stage_cuts:
        raise RuntimeError(f"checkpoint at {output_dir} has no stage cuts")

    # cobre 0.14 carries state_dimension per pool (on each stage_cuts entry),
    # not on the checkpoint metadata, so the terminal pool's own value is the
    # authoritative state-vector width — there is no separate global metadata
    # value left to cross-check it against.
    terminal = max(stage_cuts, key=lambda stage: stage["stage_id"])
    terminal_state_dimension = int(terminal["state_dimension"])

    entity_manifest = tuple(terminal["entity_manifest"])
    if not entity_manifest:
        raise RuntimeError(
            f"checkpoint at {output_dir} has an empty terminal entity_manifest"
        )

    return TerminalManifest(
        entity_manifest=entity_manifest,
        state_dimension=terminal_state_dimension,
    )
