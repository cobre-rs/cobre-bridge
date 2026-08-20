"""Tests for the terminal-manifest bootstrap (``fcf/bootstrap.py``).

The bootstrap runs cobre in-process via ``cobre.run.run`` (no subprocess
binary), so the mocked tests below stub ``sys.modules['cobre']`` with a
``run.run`` no-op plus a ``results.load_policy`` returning a synthetic
checkpoint — exercising the read-back and error paths without live cobre I/O
or a ``--cobre-bin`` to resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from cobre_bridge.decomp.fcf.bootstrap import (
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)

_MINIMAL_CONFIG = (
    '{"training": {"stopping_rules": [{"type": "iteration_limit", "limit": 500}]}}'
)


def _stub_cobre(policy: dict, *, run: object | None = None) -> SimpleNamespace:
    """A ``cobre`` stand-in whose ``run.run`` is a no-op (or ``run``) and whose
    ``results.load_policy`` returns ``policy`` verbatim."""

    def _default_run(*_args: object, **_kwargs: object) -> None:
        return None

    return SimpleNamespace(
        run=SimpleNamespace(run=run if run is not None else _default_run),
        results=SimpleNamespace(load_policy=lambda *_a, **_k: policy),
    )


def test_ensure_writer_binding_raises_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cobre = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)
    with pytest.raises(RuntimeError, match="cobre") as exc_info:
        ensure_writer_binding()
    message = str(exc_info.value)
    # End-user-facing: actionable pip guidance + the --no-fcf escape hatch,
    # and no repo-internal leaks (worktree paths, maturin build commands).
    assert "pip install" in message
    assert "--no-fcf" in message
    assert "~/git" not in message
    assert "maturin" not in message


def test_ensure_writer_binding_passes_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cobre = SimpleNamespace(write_policy_checkpoint=lambda: None)
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)
    ensure_writer_binding()  # must not raise


def test_bootstrap_does_not_mutate_input_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-process run applies its 1-iteration cap through
    ``config_overrides`` (in memory) and writes only to ``work_dir``, so the
    input case's ``config.json`` is left byte-for-byte untouched — no scratch
    copy, no file edit."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")
    original_bytes = (case_dir / "config.json").read_bytes()

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "cobre", _stub_cobre(fake_policy))
    bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert (case_dir / "config.json").read_bytes() == original_bytes


def test_bootstrap_passes_single_iteration_run_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bootstrap drives cobre in-process for a single iteration: it calls
    ``cobre.run.run`` on ``case_dir`` with ``skip_simulation`` on, an
    ``iteration_limit`` 1 ``config_overrides``, and an ``on_iteration`` stop
    callback — never a subprocess binary."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    calls: list[dict[str, object]] = []

    def _record_run(*_args: object, **kwargs: object) -> None:
        calls.append(kwargs)

    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
            },
        ],
    }
    monkeypatch.setitem(sys.modules, "cobre", _stub_cobre(fake_policy, run=_record_run))
    bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["skip_simulation"] is True
    assert kwargs["config_overrides"]["training.stopping_rules"] == [
        {"type": "iteration_limit", "limit": 1}
    ]
    assert callable(kwargs["on_iteration"])
    assert kwargs["on_iteration"](object()) is True  # stops at the first boundary


def test_bootstrap_raises_on_cobre_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure inside the run propagates as ``cobre.run.run``'s exception."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom: bad case")

    monkeypatch.setitem(sys.modules, "cobre", _stub_cobre({}, run=_boom))

    with pytest.raises(RuntimeError, match="boom: bad case"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_empty_stage_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose ``stage_cuts`` list is empty is a load error, not a
    legitimate empty terminal pool (an empty *terminal* pool still carries an
    ``entity_manifest``; an empty ``stage_cuts`` list means no pool at all)."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    monkeypatch.setitem(sys.modules, "cobre", _stub_cobre({"stage_cuts": []}))

    with pytest.raises(RuntimeError, match="no stage cuts"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_empty_terminal_entity_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    fake_policy = {
        "stage_cuts": [
            {"stage_id": 0, "state_dimension": 5, "entity_manifest": []},
        ],
    }
    monkeypatch.setitem(sys.modules, "cobre", _stub_cobre(fake_policy))

    with pytest.raises(RuntimeError, match="empty terminal entity_manifest"):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")


def test_bootstrap_raises_when_cobre_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no ``cobre`` importable, the in-process bootstrap fails loudly at
    its ``import cobre`` rather than silently. Setting the ``sys.modules``
    entry to ``None`` (rather than deleting it) forces ``ModuleNotFoundError``
    even where cobre is genuinely installed, so this test behaves the same in
    the dev venv and in a cobre-free venv."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    monkeypatch.setitem(sys.modules, "cobre", None)

    with pytest.raises(ModuleNotFoundError):
        bootstrap_terminal_manifest(case_dir, work_dir=tmp_path / "work")
