"""Tests for the terminal-manifest bootstrap (``fcf/bootstrap.py``)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cobre_bridge.decomp.fcf.bootstrap import (
    TerminalManifest,
    bootstrap_terminal_manifest,
    ensure_writer_binding,
)

# Real, gitignored deck + local cobre build (see example/README.md and
# tests/test_decomp_contracts.py's identical constants) — CI has neither, so
# only the one test that genuinely needs a live ``cobre run`` is
# skipif-guarded on both.
_RV3_DECK = Path("example/decomp-jul-26-rv3")
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"

_MINIMAL_CONFIG = (
    '{"training": {"stopping_rules": [{"type": "iteration_limit", "limit": 500}]}}'
)


def _write_fake_binary(path: Path, *, exit_code: int, stderr: str = "") -> None:
    """Write an executable shell stand-in for ``cobre run`` at ``path``."""
    script = "#!/bin/sh\n"
    if stderr:
        script += f"echo '{stderr}' >&2\n"
    script += f"exit {exit_code}\n"
    path.write_text(script)
    path.chmod(0o755)


def test_ensure_writer_binding_raises_when_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cobre = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)
    with pytest.raises(RuntimeError, match="write_policy_checkpoint") as exc_info:
        ensure_writer_binding()
    assert "~/git/cobre" in str(exc_info.value)


def test_ensure_writer_binding_passes_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub_cobre = SimpleNamespace(write_policy_checkpoint=lambda: None)
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)
    ensure_writer_binding()  # must not raise


@pytest.mark.skipif(
    not _COBRE_BIN.exists(),
    reason="cobre binary not present (~/git/cobre/target/release/cobre required)",
)
@pytest.mark.skipif(not _RV3_DECK.exists(), reason="decomp-jul-26-rv3 deck not present")
def test_bootstrap_reads_terminal_manifest(tmp_path: Path) -> None:
    from cobre_bridge.decomp.pipeline import convert_decomp_case

    case_dir = tmp_path / "converted"
    convert_decomp_case(_RV3_DECK, case_dir, force=True)

    manifest = bootstrap_terminal_manifest(
        case_dir, _COBRE_BIN, work_dir=tmp_path / "work"
    )

    assert isinstance(manifest, TerminalManifest)
    assert isinstance(manifest.state_dimension, int)
    assert manifest.state_dimension > 0
    assert isinstance(manifest.entity_manifest, tuple)
    assert len(manifest.entity_manifest) > 0
    for slot in manifest.entity_manifest:
        assert isinstance(slot, dict)
        assert "entity_type" in slot
        assert "entity_id" in slot
        assert "subindex" in slot


def test_bootstrap_does_not_mutate_input_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 3 — needs neither the real binary nor a real deck: a fake ``cobre``
    stand-in that exits 0 plus a stubbed ``load_policy`` exercise the same
    copy -> edit-the-copy -> read-back path without live cobre I/O."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")
    original_bytes = (case_dir / "config.json").read_bytes()

    fake_bin = tmp_path / "fake_cobre.sh"
    _write_fake_binary(fake_bin, exit_code=0)

    fake_policy = {
        "metadata": {"state_dimension": 1},
        "stage_cuts": [
            {
                "stage_id": 0,
                "state_dimension": 1,
                "entity_manifest": [{"entity_type": 0, "entity_id": 0, "subindex": 0}],
            },
        ],
    }
    stub_cobre = SimpleNamespace(
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy)
    )
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)
    bootstrap_terminal_manifest(case_dir, fake_bin, work_dir=tmp_path / "work")

    assert (case_dir / "config.json").read_bytes() == original_bytes


def test_bootstrap_raises_on_cobre_failure(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    broken_bin = tmp_path / "broken_cobre.sh"
    _write_fake_binary(broken_bin, exit_code=1, stderr="boom: bad case")

    with pytest.raises(RuntimeError, match="boom: bad case"):
        bootstrap_terminal_manifest(case_dir, broken_bin, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_empty_stage_cuts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint whose ``stage_cuts`` list is empty is a load error, not a
    legitimate empty terminal pool (an empty *terminal* pool still carries an
    ``entity_manifest``; an empty ``stage_cuts`` list means no pool at all)."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    fake_bin = tmp_path / "fake_cobre.sh"
    _write_fake_binary(fake_bin, exit_code=0)

    stub_cobre = SimpleNamespace(
        results=SimpleNamespace(
            load_policy=lambda *args, **kwargs: {
                "metadata": {"state_dimension": 0},
                "stage_cuts": [],
            }
        )
    )
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)

    with pytest.raises(RuntimeError, match="no stage cuts"):
        bootstrap_terminal_manifest(case_dir, fake_bin, work_dir=tmp_path / "work")


def test_bootstrap_raises_on_empty_terminal_entity_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    fake_bin = tmp_path / "fake_cobre.sh"
    _write_fake_binary(fake_bin, exit_code=0)

    fake_policy = {
        "metadata": {"state_dimension": 5},
        "stage_cuts": [
            {"stage_id": 0, "state_dimension": 5, "entity_manifest": []},
        ],
    }
    stub_cobre = SimpleNamespace(
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy)
    )
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)

    with pytest.raises(RuntimeError, match="empty terminal entity_manifest"):
        bootstrap_terminal_manifest(case_dir, fake_bin, work_dir=tmp_path / "work")


def test_bootstrap_read_back_raises_when_cobre_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 4 — with no ``cobre`` importable and a fake binary exiting 0, the
    read-back ``import cobre`` fails loudly rather than silently. Setting the
    ``sys.modules`` entry to ``None`` (rather than deleting it) forces the
    import to raise ``ModuleNotFoundError`` even in an environment where
    cobre is genuinely installed on disk, so this test behaves the same in
    the dev venv and in a cobre-free venv."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")

    fake_bin = tmp_path / "fake_cobre.sh"
    _write_fake_binary(fake_bin, exit_code=0)

    monkeypatch.setitem(sys.modules, "cobre", None)

    with pytest.raises(ModuleNotFoundError):
        bootstrap_terminal_manifest(case_dir, fake_bin, work_dir=tmp_path / "work")


def test_bootstrap_never_calls_subprocess_with_cargo_bin(tmp_path: Path) -> None:
    """Guards the "never ``~/.cargo/bin/cobre``" pitfall: the binary path
    passed in is threaded verbatim into ``subprocess.run``'s argv[0]."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "config.json").write_text(_MINIMAL_CONFIG, encoding="utf-8")
    local_bin = tmp_path / "local_cobre"

    with patch(
        "cobre_bridge.decomp.fcf.bootstrap.subprocess.run",
        return_value=subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="stop here"
        ),
    ) as mock_run:
        with pytest.raises(RuntimeError, match="stop here"):
            bootstrap_terminal_manifest(case_dir, local_bin, work_dir=tmp_path / "work")

    called_argv = mock_run.call_args[0][0]
    assert called_argv[0] == str(local_bin)
    assert ".cargo" not in called_argv[0]
