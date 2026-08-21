"""Provenance manifest for a conversion run.

Defines :class:`ConversionManifest`, a provenance record emitted alongside a
converted Cobre case directory so that a downstream agent can know exactly which
bridge version, git state, source-model case directory, and input files produced
a given conversion, plus the entity counts and the diagnostics raised during the
run. It mirrors :mod:`cobre_bridge.comparators.manifest`; both subclass
:class:`cobre_bridge.provenance_manifest.ProvenanceManifest` for their shared
``to_json``/``from_json`` behaviour.

The shared :func:`cobre_bridge._git.git_sha` runs the git subprocess only
inside :meth:`ConversionManifest.create`, never at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar

import cobre_bridge
from cobre_bridge._git import git_sha
from cobre_bridge.provenance_manifest import ProvenanceManifest


@dataclass
class ConversionManifest(ProvenanceManifest):
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
    # Minimum cobre version the converted output requires. ``convert`` sets it on
    # every case (all system entities now carry ``operational_start_date``, added
    # in cobre 0.10.0). Defaults to ``None`` so an older manifest that predates the
    # field round-trips through ``from_json`` unchanged. Appended last so existing
    # positional construction stays unchanged.
    min_cobre_version: str | None = None

    _NOT_FOUND_LABEL: ClassVar[str] = "Conversion manifest"

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
        requires (``None`` only when omitted, e.g. by an older caller). The
        remaining data fields are caller-supplied.
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
