"""Packaging guards: the declared dependencies must let a plain install work.

`convert decomp` imports the deck's boundary FCF by default and needs
`import cobre` to succeed, so `cobre-python` must be a CORE runtime dependency —
not an optional extra. This module locks that down: a fresh
`pip install cobre-bridge` (no extras) must pull a checkpoint-capable cobre. It was the
absence of exactly this guard that let a release ship with `cobre-python` as an
extra, so a plain install failed `convert decomp` on any real deck. It also keeps
the ruff CI installs on the version `uv.lock` pins. Tier-1: reads `pyproject.toml`,
`uv.lock` and the CI workflow, never imports cobre.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _core_dependencies() -> list[str]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    return data["project"]["dependencies"]


def _cobre_python_pin(deps: list[str]) -> str | None:
    for dep in deps:
        if dep.replace(" ", "").startswith("cobre-python"):
            return dep
    return None


def test_cobre_python_is_a_core_runtime_dependency() -> None:
    """`cobre-python` must be in `[project].dependencies`, not an extra, so a
    plain `pip install cobre-bridge` gives a working `convert decomp`."""
    core = _core_dependencies()
    assert _cobre_python_pin(core) is not None, (
        "cobre-python must be a core runtime dependency, not an optional extra; "
        f"found core dependencies: {core}"
    )


def test_cobre_python_core_pin_is_exactly_min_cobre_version() -> None:
    """The core `cobre-python` pin must be exactly `MIN_COBRE_VERSION`: cobre
    loads a policy checkpoint only in the version that wrote it, so the
    boundary `convert decomp` writes must come from the paired release."""
    from cobre_bridge.cli import MIN_COBRE_VERSION

    pin = _cobre_python_pin(_core_dependencies())
    assert pin is not None
    assert pin.replace(" ", "") == f"cobre-python=={MIN_COBRE_VERSION}", (
        f"cobre-python core pin {pin!r} must be exactly "
        f"cobre-python=={MIN_COBRE_VERSION}"
    )


_UV_LOCK = Path(__file__).resolve().parent.parent / "uv.lock"


def _lock_cobre_python_requirement() -> dict[str, str] | None:
    """The `cobre-bridge` package's `cobre-python` requires-dist entry from
    uv.lock, or None if absent."""
    data = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))
    for package in data["package"]:
        if package.get("name") == "cobre-bridge":
            for req in package.get("metadata", {}).get("requires-dist", []):
                if req.get("name") == "cobre-python":
                    return req
    return None


def test_uv_lock_cobre_python_is_a_core_dependency() -> None:
    """uv.lock must record cobre-python as a core requirement — not gated
    behind an `extra` — so `uv sync` installs it by default, matching
    pyproject. Guards against the lock drifting back to a `validation` extra."""
    req = _lock_cobre_python_requirement()
    assert req is not None, "cobre-python missing from uv.lock requires-dist"
    marker = req.get("marker", "")
    assert "extra" not in marker, (
        "cobre-python must be a core dependency in uv.lock, not gated behind "
        f"an extra; found marker {marker!r}"
    )


def test_uv_lock_cobre_python_is_pinned_exactly_to_min_cobre_version() -> None:
    """The uv.lock cobre-python specifier must be exactly MIN_COBRE_VERSION, so
    a regenerated lock never resolves a release other than the paired one."""
    from cobre_bridge.cli import MIN_COBRE_VERSION

    req = _lock_cobre_python_requirement()
    assert req is not None
    specifier = req.get("specifier", "").replace(" ", "")
    assert specifier == f"=={MIN_COBRE_VERSION}", (
        f"uv.lock cobre-python specifier {specifier!r} must be exactly "
        f"=={MIN_COBRE_VERSION}"
    )


_CI_WORKFLOW = (
    Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
)


def _lock_package_version(name: str) -> str | None:
    data = tomllib.loads(_UV_LOCK.read_text(encoding="utf-8"))
    for package in data["package"]:
        if package.get("name") == name:
            return package.get("version")
    return None


def test_ci_installs_the_ruff_version_uv_lock_pins() -> None:
    """CI's lint job must install exactly the ruff uv.lock resolves, so a new
    ruff release cannot turn CI red, or green on code the pre-commit hook
    rejects, without a change in the repository."""
    locked = _lock_package_version("ruff")
    assert locked is not None, "ruff missing from uv.lock"
    installs = re.findall(
        r"pip install\s+[\"']?(ruff[^\s\"']*)",
        _CI_WORKFLOW.read_text(encoding="utf-8"),
    )
    assert installs == [f"ruff=={locked}"], (
        f"ci.yml must install ruff=={locked} (the uv.lock version) exactly once; "
        f"found {installs}"
    )
