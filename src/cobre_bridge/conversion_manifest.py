"""Provenance manifest for a conversion run.

Defines :class:`ConversionManifest`, a provenance record emitted alongside a
converted Cobre case directory so that a downstream agent can know exactly which
bridge version, git state, source-model case directory, and input files produced
a given conversion, plus the entity counts and the diagnostics raised during the
run. It mirrors :mod:`cobre_bridge.comparators.manifest`.

The shared :func:`cobre_bridge._git.git_sha` runs the git subprocess only
inside :meth:`ConversionManifest.create`, never at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import cobre_bridge
from cobre_bridge._git import git_sha
from cobre_bridge.newave_files import NewaveFiles

if TYPE_CHECKING:
    from cobre_bridge.diagnostics import Diagnostic

_HASH_CHUNK_BYTES = 8192


@dataclass
class ConversionManifest:
    """Provenance record for a conversion run.

    Carries the originating command, the source-model and output paths, the
    bridge version and git SHA, a UTC timestamp, the converted entity counts,
    the hashed input files, a per-severity diagnostic summary, and the full
    diagnostic list.
    """

    command: str
    source_dir: str
    output_dir: str
    bridge_version: str
    git_sha: str | None
    timestamp: str
    entity_counts: dict[str, int] = field(default_factory=dict)
    input_files: list[dict[str, object]] = field(default_factory=list)
    diagnostics_summary: dict[str, int] = field(default_factory=dict)
    diagnostics: list[dict[str, object]] = field(default_factory=list)
    # Minimum cobre version the converted output requires, or ``None`` when the
    # output is loadable by any released cobre-python. Set only for cases that
    # emit the unreleased dead-volume filling schema (``NE``-with-filling
    # plants); ``None`` for EX-only cases. Appended last so existing positional
    # and ``from_json`` round-tripping stay unchanged.
    min_cobre_version: str | None = None

    @classmethod
    def create(
        cls,
        command: str,
        source_dir: Path,
        output_dir: Path,
        *,
        entity_counts: dict[str, int],
        input_files: list[dict[str, object]],
        diagnostics_summary: dict[str, int],
        diagnostics: list[dict[str, object]],
        min_cobre_version: str | None = None,
    ) -> ConversionManifest:
        """Build a manifest, capturing bridge version, git SHA, and UTC time.

        ``bridge_version`` is taken from :data:`cobre_bridge.__version__`,
        ``timestamp`` from :func:`datetime.now` in UTC (ISO 8601) — the only
        non-deterministic field — and ``git_sha`` from
        :func:`cobre_bridge._git.git_sha`. The
        ``source_dir`` / ``output_dir`` paths are stringified via ``str(...)``.
        ``min_cobre_version`` records the minimum cobre version the output
        requires (``None`` for EX-only cases). The remaining data fields are
        caller-supplied.
        """
        return cls(
            command=command,
            source_dir=str(source_dir),
            output_dir=str(output_dir),
            bridge_version=cobre_bridge.__version__,
            git_sha=git_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            entity_counts=entity_counts,
            input_files=input_files,
            diagnostics_summary=diagnostics_summary,
            diagnostics=diagnostics,
            min_cobre_version=min_cobre_version,
        )

    def to_json(self, path: Path) -> None:
        """Write the manifest to ``path`` as indented JSON.

        Parent directories are created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> ConversionManifest:
        """Reconstruct a manifest from a JSON file written by :meth:`to_json`.

        Unknown keys (e.g. from a manifest written by a newer bridge) are
        dropped, and absent optional keys fall back to their dataclass defaults,
        so reading a manifest across bridge versions does not raise. Raises
        :class:`FileNotFoundError` (naming ``path``) when the file is absent.
        """
        if not path.exists():
            raise FileNotFoundError(f"Conversion manifest not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def hash_input_files(files: NewaveFiles) -> list[dict[str, object]]:
    """Hash every present input file of *files* into a sorted manifest list.

    Introspects :func:`dataclasses.fields` of :class:`NewaveFiles`, skipping the
    ``directory`` field and any optional field whose value is ``None``. For each
    remaining ``(name, path)`` it produces an entry
    ``{"field", "path", "sha256", "size_bytes"}`` where ``sha256`` is the
    SHA-256 hex digest of the file's raw bytes and ``size_bytes`` is the file
    size.

    A file that cannot be read (:class:`OSError`) is recorded with
    ``sha256=None`` and ``size_bytes=None`` rather than aborting the manifest.
    The returned list is sorted by ``"field"`` for deterministic ordering.
    """
    entries: list[dict[str, object]] = []
    for spec in fields(NewaveFiles):
        if spec.name == "directory":
            continue
        value = getattr(files, spec.name)
        if value is None:
            continue
        path: Path = value
        sha256, size_bytes = _hash_file(path)
        entries.append(
            {
                "field": spec.name,
                "path": str(path),
                "sha256": sha256,
                "size_bytes": size_bytes,
            }
        )
    entries.sort(key=lambda entry: entry["field"])
    return entries


def _hash_file(path: Path) -> tuple[str | None, int | None]:
    """Return the SHA-256 hex digest and byte size of *path*.

    Reads the file in ``_HASH_CHUNK_BYTES``-byte binary chunks. The size is read
    from the open descriptor (``os.fstat``) so it is consistent with the bytes
    just hashed even if the file is replaced afterwards. On :class:`OSError`
    (unreadable / missing file) returns ``(None, None)`` so the caller records
    the failure instead of raising.
    """
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
            size_bytes = os.fstat(handle.fileno()).st_size
    except OSError:
        return None, None
    return digest.hexdigest(), size_bytes


def summarize_diagnostics(diagnostics: list[Diagnostic]) -> dict[str, int]:
    """Count *diagnostics* by severity value, omitting absent severities.

    Returns a plain ``dict[str, int]`` keyed by ``severity.value`` (``"info"`` /
    ``"warning"`` / ``"error"``); a severity with no diagnostics is absent from
    the result rather than mapped to zero.
    """
    counts = Counter(diagnostic.severity.value for diagnostic in diagnostics)
    return dict(counts)
