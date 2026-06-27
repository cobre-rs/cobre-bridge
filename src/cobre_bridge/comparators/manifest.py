"""Provenance manifest for comparison artifacts.

Defines :class:`ComparisonManifest`, a provenance record emitted alongside
comparison artifacts so that a divergence-investigation agent can know exactly
which cases, tolerance, tool versions, and git state produced a given set of
artifacts, plus which artifacts were emitted and the headline divergences.

This module is standalone: it adds no callers in the comparison engines and
never runs the git subprocess at import time.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

import cobre_bridge
from cobre_bridge._git import git_sha


@dataclass
class ComparisonManifest:
    """Provenance record for a comparison run.

    Carries the command, case paths, tolerance, tool/git versions, a UTC
    timestamp, the emitted artifact filenames, and the headline divergences.
    """

    command: str
    newave_dir: str
    cobre_output_dir: str
    tolerance: float
    bridge_version: str
    git_sha: str | None
    timestamp: str
    cobre_version: str | None = None
    newave_version: str | None = None
    artifacts: list[str] = field(default_factory=list)
    top_divergences: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        command: str,
        newave_dir: Path,
        cobre_output_dir: Path,
        tolerance: float,
        *,
        cobre_version: str | None = None,
        newave_version: str | None = None,
    ) -> ComparisonManifest:
        """Build a manifest, capturing bridge version, git SHA, and UTC time.

        ``bridge_version`` is taken from :data:`cobre_bridge.__version__`,
        ``timestamp`` from :func:`datetime.now` in UTC (ISO 8601), and
        ``git_sha`` from :func:`cobre_bridge._git.git_sha`. ``cobre_version`` /
        ``newave_version`` remain caller-supplied (default ``None``); deriving
        them is out of scope for this module.
        """
        return cls(
            command=command,
            newave_dir=str(newave_dir),
            cobre_output_dir=str(cobre_output_dir),
            tolerance=tolerance,
            bridge_version=cobre_bridge.__version__,
            git_sha=git_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            cobre_version=cobre_version,
            newave_version=newave_version,
        )

    def to_json(self, path: Path) -> None:
        """Write the manifest to ``path`` as indented JSON.

        Parent directories are created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> ComparisonManifest:
        """Reconstruct a manifest from a JSON file written by :meth:`to_json`.

        Unknown keys (e.g. from a manifest written by a newer bridge) are
        dropped, and absent optional keys fall back to their dataclass defaults,
        so reading a manifest across bridge versions does not raise. Raises
        :class:`FileNotFoundError` (naming ``path``) when the file is absent.
        """
        if not path.exists():
            raise FileNotFoundError(f"Comparison manifest not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)
