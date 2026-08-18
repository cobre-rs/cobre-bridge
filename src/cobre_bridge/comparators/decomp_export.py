"""Machine-readable artifact export for the ``compare decomp`` comparison.

This module writes a :class:`DecompComparison` (:mod:`decomp_results`) directly
to disk — the tidy ``comparison.parquet``/``.csv`` rows, the per-variable
``summary.parquet``/``.json``/``.csv`` roll-up, the two solvers' bound
trajectories in ``convergence.parquet``/``.json``/``.csv``, and a provenance
manifest at ``comparison.json`` (always written) — plus the
:class:`DecompComparisonManifest` record it writes.

``DecompComparison`` is a different shape from the canonical
:class:`~cobre_bridge.comparators.dataset.ComparisonDataset` (its summary
columns and tidy ``source``/``cobre`` layout do not satisfy that dataset's
schema), so this is a parallel, DECOMP-shaped writer rather than a reuse of
:func:`~cobre_bridge.comparators.export.write_artifacts`. It composes existing
primitives only (:func:`~cobre_bridge.comparators.export._read_cobre_version`,
:func:`~cobre_bridge._git.git_sha`, :data:`cobre_bridge.__version__`) and
imports neither ``render_decomp_comparison`` nor any HTML renderer, so the
console path is unaffected.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import cobre_bridge
from cobre_bridge._git import git_sha
from cobre_bridge.comparators.export import _read_cobre_version

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from cobre_bridge.comparators.decomp_results import DecompComparison

#: The artifact formats :func:`write_decomp_artifacts` knows how to emit.
_VALID_FORMATS: frozenset[str] = frozenset({"parquet", "json", "csv"})

#: Always-written manifest filename (basename); shared with `compare newave`
#: since the manifest's ``command`` field disambiguates the two.
_MANIFEST_FILE: str = "comparison.json"

#: Per-format artifact basenames.
_ROWS_PARQUET: str = "comparison.parquet"
_SUMMARY_PARQUET: str = "summary.parquet"
_CONVERGENCE_PARQUET: str = "convergence.parquet"
_SUMMARY_JSON: str = "summary.json"
_CONVERGENCE_JSON: str = "convergence.json"
_ROWS_CSV: str = "comparison.csv"
_SUMMARY_CSV: str = "summary.csv"
_CONVERGENCE_CSV: str = "convergence.csv"


@dataclass
class DecompComparisonManifest:
    """Provenance record for a ``compare decomp`` export run.

    Field names follow the generic ``source_dir``/``output_dir`` precedent set
    by :class:`~cobre_bridge.conversion_manifest.ConversionManifest`
    (``decomp_dir``/``cobre_output_dir``) rather than
    :class:`~cobre_bridge.comparators.manifest.ComparisonManifest`'s
    source-model-specific field names, and it carries an ``unmapped`` field
    that manifest has no home for.
    """

    command: str
    decomp_dir: str
    cobre_output_dir: str
    tolerance: float
    bridge_version: str
    git_sha: str | None
    timestamp: str
    cobre_version: str | None = None
    artifacts: list[str] = field(default_factory=list)
    unmapped: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        command: str,
        decomp_dir: Path,
        cobre_output_dir: Path,
        tolerance: float,
    ) -> DecompComparisonManifest:
        """Build a manifest, capturing bridge version, git SHA, and UTC time.

        ``bridge_version`` is taken from :data:`cobre_bridge.__version__`,
        ``timestamp`` from :func:`datetime.now` in UTC (ISO 8601), ``git_sha``
        from :func:`cobre_bridge._git.git_sha`, and ``cobre_version`` from
        :func:`~cobre_bridge.comparators.export._read_cobre_version`.
        ``artifacts`` and ``unmapped`` are left at their empty defaults; the
        caller fills them in once the export is complete.
        """
        return cls(
            command=command,
            decomp_dir=str(decomp_dir),
            cobre_output_dir=str(cobre_output_dir),
            tolerance=tolerance,
            bridge_version=cobre_bridge.__version__,
            git_sha=git_sha(),
            timestamp=datetime.now(tz=UTC).isoformat(),
            cobre_version=_read_cobre_version(cobre_output_dir),
        )

    def to_json(self, path: Path) -> None:
        """Write the manifest to ``path`` as indented JSON.

        Parent directories are created if missing.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def write_decomp_artifacts(
    comparison: DecompComparison,
    *,
    command: str,
    decomp_dir: Path,
    cobre_output_dir: Path,
    tolerance: float,
    out_dir: Path,
    formats: Sequence[str],
) -> DecompComparisonManifest:
    """Emit ``comparison``'s artifacts and return the written manifest.

    Validates ``formats`` against ``{"parquet", "json", "csv"}`` (fail fast),
    creates ``out_dir``, emits the requested per-format artifacts, and always
    writes the provenance manifest to ``out_dir / "comparison.json"``.

    Per-format outputs:

    - ``"parquet"`` → ``comparison.parquet`` (``comparison.rows``),
      ``summary.parquet`` (``comparison.summary``), and ``convergence.parquet``
      (``comparison.convergence``).
    - ``"json"`` → ``summary.json`` and ``convergence.json``, each the
      corresponding frame's ``to_dicts()`` as indented JSON.
    - ``"csv"`` → ``comparison.csv``, ``summary.csv``, ``convergence.csv``
      (each frame's ``write_csv``).

    An empty comparison (e.g. a ``no-comparable-rows`` run) still writes
    header-only, typed artifacts plus the manifest — the DECOMP frames are
    already typed when empty, so no special-casing is needed.

    The returned manifest's :attr:`~DecompComparisonManifest.artifacts` is the
    sorted list of every emitted basename, *including* ``comparison.json``
    itself. Its :attr:`~DecompComparisonManifest.unmapped` mirrors
    ``comparison.unmapped`` with every entity id coerced to ``int``.

    Args:
        comparison: The comparison to export.
        command: The originating command label (e.g. ``"compare decomp"``).
        decomp_dir: The source-deck directory (recorded in the manifest).
        cobre_output_dir: The Cobre output directory (recorded in the manifest
            and probed for the Cobre version).
        tolerance: The comparison tolerance (recorded in the manifest).
        out_dir: Destination directory; created with ``parents=True``.
        formats: The artifact formats to emit; each must be in
            ``{"parquet", "json", "csv"}``.

    Returns:
        The :class:`DecompComparisonManifest` written to
        ``out_dir / "comparison.json"``.

    Raises:
        ValueError: If ``formats`` contains a value outside the allowed set;
            the message names the offending value(s).
    """
    unknown = sorted(set(formats) - _VALID_FORMATS)
    if unknown:
        allowed = sorted(_VALID_FORMATS)
        msg = f"unknown export format(s) {unknown}; allowed formats are {allowed}"
        raise ValueError(msg)

    out_dir.mkdir(parents=True, exist_ok=True)
    requested = set(formats)
    written: list[str] = []

    if "parquet" in requested:
        comparison.rows.write_parquet(out_dir / _ROWS_PARQUET)
        written.append(_ROWS_PARQUET)
        comparison.summary.write_parquet(out_dir / _SUMMARY_PARQUET)
        written.append(_SUMMARY_PARQUET)
        comparison.convergence.write_parquet(out_dir / _CONVERGENCE_PARQUET)
        written.append(_CONVERGENCE_PARQUET)

    if "json" in requested:
        (out_dir / _SUMMARY_JSON).write_text(
            json.dumps(comparison.summary.to_dicts(), indent=2), encoding="utf-8"
        )
        written.append(_SUMMARY_JSON)
        (out_dir / _CONVERGENCE_JSON).write_text(
            json.dumps(comparison.convergence.to_dicts(), indent=2), encoding="utf-8"
        )
        written.append(_CONVERGENCE_JSON)

    if "csv" in requested:
        comparison.rows.write_csv(out_dir / _ROWS_CSV)
        written.append(_ROWS_CSV)
        comparison.summary.write_csv(out_dir / _SUMMARY_CSV)
        written.append(_SUMMARY_CSV)
        comparison.convergence.write_csv(out_dir / _CONVERGENCE_CSV)
        written.append(_CONVERGENCE_CSV)

    # The manifest always lists itself.
    written.append(_MANIFEST_FILE)

    manifest = DecompComparisonManifest.create(
        command, decomp_dir, cobre_output_dir, tolerance
    )
    manifest.artifacts = sorted(written)
    manifest.unmapped = {
        level: [int(code) for code in codes]
        for level, codes in comparison.unmapped.items()
    }
    manifest.to_json(out_dir / _MANIFEST_FILE)

    return manifest
