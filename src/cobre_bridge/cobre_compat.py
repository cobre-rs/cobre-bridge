"""The cobre-compat policy: the minimum cobre-python version gate."""

from __future__ import annotations

#: Minimum cobre / cobre-python version that can load the bridge's converted
#: output. The manifest records it (single source of truth) and the
#: ``--validate`` gate uses it to decide whether the installed cobre-python is
#: new enough to validate the output. Keep the ``cobre-python`` pin in
#: ``pyproject.toml`` in lockstep with this constant on any future bump.
#:
#: The floor is 0.14.3 because the emitted terminal boundary policy depends on
#: it: 0.14.3's ``write_policy_checkpoint`` reserves the canonical inflow-lag
#: state slots. On an older cobre those slots are absent and a boundary carrying
#: inflow-lag gradient terms has its lag coupling silently dropped.
MIN_COBRE_VERSION = "0.14.3"


def _installed_cobre_python_version() -> str | None:
    """Return the installed ``cobre-python`` distribution version, or ``None``.

    The package imports as ``cobre`` but is distributed as ``cobre-python``;
    this reads the distribution metadata. Returns ``None`` when it is not
    installed, so the caller falls through to the generic "not installed" skip.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _dist_version

    try:
        return _dist_version("cobre-python")
    except PackageNotFoundError:
        return None


def _cobre_python_supports_output(installed: str) -> bool:
    """Whether an installed cobre-python *version* can load the bridge's output.

    ``True`` when *installed* is at least :data:`MIN_COBRE_VERSION` by a numeric
    release-segment comparison (so ``"0.10.0"`` and ``"0.11.2"`` qualify,
    ``"0.9.1"`` does not). A non-numeric pre-release suffix on a segment is
    ignored (``"0.10.0rc1"`` reads as ``0.10.0``); the gate only guards against an
    obviously-older install, so the leniency is deliberate.
    """

    def _release(value: str) -> tuple[int, ...]:
        parts: list[int] = []
        for segment in value.split("."):
            digits = ""
            for char in segment:
                if not char.isdigit():
                    break
                digits += char
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    return _release(installed) >= _release(MIN_COBRE_VERSION)
