"""Tests for the boundary-FCF capability probe (``fcf/capability.py``)."""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from cobre_bridge.decomp.fcf import capability
from cobre_bridge.decomp.fcf.capability import (
    REMEDIATION,
    ensure_boundary_fcf_capability,
)
from tests.conftest import requires_cobre_python, requires_writer_binding

#: Every fact the remediation message must name: the dependency at fault, the
#: install/upgrade fix, and the ``--no-fcf`` escape hatch. Shared by every
#: "raises" test below.
_REMEDIATION_MARKERS = (
    "cobre-python",
    "pip install",
    "--no-fcf",
)

#: Repo-internal references that must NEVER leak into an end-user-facing
#: message: a pip-installed user has no repo checkout, so doc paths, developer
#: worktrees, and internal spec/ticket codes are noise (and, as this message's
#: history proved, actively misleading). This is the regression guard.
_REPO_INTERNAL_LEAKS = (
    "docs/",
    "plans/",
    "~/git",
    "feat/",
    "ticket-",
    "epic-",
)


def test_remediation_names_install_fix_and_escape_hatch() -> None:
    for marker in _REMEDIATION_MARKERS:
        assert marker in REMEDIATION


def test_remediation_has_no_repo_internal_references() -> None:
    """The remediation reaches pip-installed end users, who have no repo — it
    must stay self-contained (no doc paths, worktrees, or ticket/spec codes)."""
    for leak in _REPO_INTERNAL_LEAKS:
        assert leak not in REMEDIATION, (
            f"remediation leaks repo-internal reference {leak!r}: {REMEDIATION!r}"
        )


def test_ensure_boundary_fcf_capability_raises_when_wheel_lacks_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 1 -- an installed-but-incapable wheel (no `write_policy_checkpoint`
    attribute, mirroring `test_decomp_fcf_bootstrap.py`'s own
    `ensure_writer_binding` test convention) raises with the remediation
    text, chained from the underlying `AttributeError`.
    """
    stub_cobre = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, AttributeError)


def test_ensure_boundary_fcf_capability_raises_when_cobre_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 1 (variant) -- cobre entirely absent raises the same remediation,
    chained from `ModuleNotFoundError`. Setting the `sys.modules` entry to
    `None` (rather than deleting it) forces the import to fail even in an
    environment where cobre is genuinely installed, the same trick
    `test_decomp_fcf_bootstrap.py` uses.
    """
    monkeypatch.setitem(sys.modules, "cobre", None)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_ensure_boundary_fcf_capability_raises_when_delivery_date_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 3 -- a wheel that writes and reloads fine but whose reloaded
    terminal manifest slot lacks `delivery_date` (a cobre that predates the
    CBVF format) still raises -- proving the probe checks the CBVF format, not
    merely the `write_policy_checkpoint` attribute.
    """
    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "cost_scale_factor": 1.0,
                "node_id": 0,
                "graph_stage_id": 0,
                "entity_manifest": [
                    {
                        "entity_type": 0,
                        "entity_id": 0,
                        "subindex": 0,
                        "was_active": True,
                    }
                ],
            }
        ]
    }
    stub_cobre = SimpleNamespace(
        write_policy_checkpoint=lambda *args, **kwargs: None,
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy),
    )
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "delivery_date" in str(exc_info.value.__cause__)


def test_ensure_boundary_fcf_capability_raises_when_cost_scale_factor_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 2 (failure variant) -- a wheel that writes and reloads fine, and
    whose reloaded terminal pool carries `delivery_date` on its manifest
    slot, but whose reloaded terminal pool itself omits `cost_scale_factor`
    (a wheel that drops the self-describing fields on reload) still raises --
    proving the probe checks the pool-level fields, not merely the slot-level
    `delivery_date`.
    """
    fake_policy = {
        "stage_cuts": [
            {
                "stage_id": 0,
                "node_id": 0,
                "graph_stage_id": 0,
                "entity_manifest": [
                    {
                        "entity_type": 0,
                        "entity_id": 0,
                        "subindex": 0,
                        "was_active": True,
                    }
                ],
            }
        ]
    }
    stub_cobre = SimpleNamespace(
        write_policy_checkpoint=lambda *args, **kwargs: None,
        results=SimpleNamespace(load_policy=lambda *args, **kwargs: fake_policy),
    )
    monkeypatch.setitem(sys.modules, "cobre", stub_cobre)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_boundary_fcf_capability()

    for marker in _REMEDIATION_MARKERS:
        assert marker in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert "cost_scale_factor" in str(exc_info.value.__cause__)


def test_capability_module_imports_with_cobre_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC 4 -- no module-scope `import cobre`: forcing `import cobre` to
    raise `ModuleNotFoundError` and reloading the already-imported module
    must not raise.
    """
    monkeypatch.setitem(sys.modules, "cobre", None)

    reloaded = importlib.reload(capability)

    assert hasattr(reloaded, "ensure_boundary_fcf_capability")


@requires_cobre_python
@requires_writer_binding
def test_ensure_boundary_fcf_capability_passes_against_installed_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC 2 -- against the installed (CBVF-capable) wheel, the guard passes
    silently and writes only under a temporary directory: forcing
    `tempfile.gettempdir()` to resolve under `tmp_path` and asserting
    nothing is left behind afterward (the probe's own `TemporaryDirectory`
    cleans itself up on exit). With `cobre-python` a core dependency this
    runs on every CI job, proving the pinned/released wheel supports the
    boundary-FCF checkpoint format end to end.
    """
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    ensure_boundary_fcf_capability()
    assert list(tmp_path.iterdir()) == []
