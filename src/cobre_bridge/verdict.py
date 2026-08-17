"""The unified ``--json`` verdict envelope shared by every command.

Each command's machine-readable ``--json`` output is one self-describing
document with EXACTLY five top-level keys, in this fixed insertion order::

    {
        "schema_version": <int>,   # always first; discriminator version
        "command": <str>,          # which command produced this document
        "status": <str>,           # command-specific outcome ("ok", "error", …)
        "summary": <object>,       # ALL command-specific payload lives here
        "diagnostics": [<object>]  # zero or more Diagnostic.to_dict() entries
    }

``schema_version`` is ALWAYS the first key so a consumer can read it before
deciding how to interpret the rest.

The nesting rule (the single design rule every command obeys): the ONLY
top-level keys that ever appear are the five envelope keys above. Every
command-specific payload — convert's counts, check's checklist, compare's
headline, dashboard's artifact — lives UNDER ``summary``. A new command adds
its payload by supplying a ``summary`` dict, never a new top-level key.

The envelope itself is deterministic: it carries no timestamp, no git SHA, no
absolute path. Determinism of the ``summary`` contents is the caller's
responsibility; provenance (if ever needed) lives in
:mod:`cobre_bridge.conversion_manifest`, never here.

Version bump policy for :data:`SCHEMA_VERSION`: increment it when a key is
renamed or removed, or when an existing key's meaning changes — a breaking
change for consumers. Adding a NEW optional key under ``summary`` is
backward-compatible and does NOT bump the version.

This is a pure leaf: it imports stdlib only (the cross-module ``Diagnostic`` /
``CompareVerdict`` types are referenced under ``TYPE_CHECKING``), reads its
inputs, allocates fresh dicts/lists, mutates nothing, and does no I/O. The
``cli.py`` layer serializes the returned dict to stdout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from cobre_bridge.comparators.dataset import ComparisonDataset
    from cobre_bridge.comparators.decomp_results import DecompComparison
    from cobre_bridge.comparators.verdict import CompareVerdict
    from cobre_bridge.diagnostics import Diagnostic

# Bump only on a breaking change: a key rename/removal or a meaning change.
# Adding a NEW optional key under ``summary`` is backward-compatible — do NOT bump.
SCHEMA_VERSION: int = 1


def build_verdict(
    command: str,
    status: str,
    summary: Mapping[str, object],
    diagnostics: Sequence[Diagnostic] = (),
) -> dict[str, object]:
    """Build the unified verdict envelope for one command invocation.

    Returns a JSON-serializable dict with the five fixed top-level keys in fixed
    insertion order, so ``json.dump(..., sort_keys=False)`` is byte-stable:
    ``schema_version`` (always first), ``command``, ``status``, ``summary``,
    ``diagnostics``.

    Args:
        command: The command discriminator (e.g. ``"convert newave"``). Passed
            through verbatim — NOT validated against an allow-list.
        status: The command-specific outcome (e.g. ``"ok"`` / ``"error"`` /
            ``"dry-run"``). Passed through verbatim — NOT validated.
        summary: The command-specific payload. Copied via ``dict(summary)`` so the
            caller's key insertion order is preserved and the caller's mapping is
            never aliased or mutated by the returned document.
        diagnostics: The findings to serialize, each via :meth:`Diagnostic.to_dict`.
            Defaults to the empty tuple (no mutable default); empty → ``[]``.

    Returns:
        A new dict with exactly ``{schema_version, command, status, summary,
        diagnostics}``. Adds NO timestamp, provenance, or absolute path.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "status": status,
        "summary": dict(summary),
        "diagnostics": [d.to_dict() for d in diagnostics],
    }


def convert_summary(
    hydros: int,
    thermals: int,
    buses: int,
    lines: int,
    stages: int,
) -> dict[str, object]:
    """The ``convert`` command's ``summary`` block — entity counts in fixed order.

    Returns ``{"hydros", "thermals", "buses", "lines", "stages"}`` in that
    insertion order. The ``--dry-run`` / ``validation`` sub-keys are appended
    inline by the convert retrofit; they are not constructed here.
    """
    return {
        "hydros": hydros,
        "thermals": thermals,
        "buses": buses,
        "lines": lines,
        "stages": stages,
    }


def check_summary(checks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The ``check`` command's ``summary`` block — the checklist under ``checks``.

    Each row is copied via ``dict(c)`` so the returned document never aliases the
    caller's check mappings. Returns ``{"checks": [...]}``.
    """
    return {"checks": [dict(c) for c in checks]}


def compare_summary(verdict: CompareVerdict) -> dict[str, object]:
    """The ``compare`` command's ``summary`` block — the headline verdict.

    Returns ``{"within_tol", "total", "worst_variable", "worst_smape",
    "all_within_tol"}`` in that order. ``worst_smape`` is the raw sMAPE ratio
    (NOT a percentage), matching :attr:`CompareVerdict.worst_smape`.

    Mirrors the console ``all_within_tol`` guard in
    :func:`cobre_bridge.ui.console.render_compare_verdict`: on a perfect match the
    "worst" clause is meaningless, so ``worst_variable`` is nulled to ``None`` and
    ``worst_smape`` to ``0.0``. Otherwise both pass through verbatim.
    """
    all_within_tol = verdict.all_within_tol
    return {
        "within_tol": verdict.within_tol,
        "total": verdict.total,
        "worst_variable": None if all_within_tol else verdict.worst_variable,
        "worst_smape": 0.0 if all_within_tol else verdict.worst_smape,
        "all_within_tol": all_within_tol,
    }


def decomp_compare_summary(
    comparison: DecompComparison, tolerance: float
) -> dict[str, object]:
    """The ``compare decomp`` command's ``summary`` block — per-variable rows.

    Returns ``{"stages", "variables", "unmapped", "within_tol", "total",
    "all_within_tol"}`` in that order. ``stages`` is
    :attr:`DecompComparison.stage_count`; ``variables`` is
    ``comparison.summary.to_dicts()`` verbatim — polars' ``to_dicts()`` already
    yields JSON-native ``int``/``float``/``str``/``None`` cells, so the rows are
    passed through unchanged; ``unmapped`` mirrors
    :attr:`DecompComparison.unmapped` with every entity id coerced to ``int``.

    The trailing three keys mirror :func:`compare_summary`'s headline fields so
    the two compare commands' summaries are symmetric. They thread NO new
    comparison science: a (level, variable) group is within tolerance iff
    *every* one of its rows in :attr:`DecompComparison.rows` already satisfies
    ``smape_pct / 100.0 <= tolerance`` — the per-row sMAPE percentage
    ``DecompComparison.rows`` already carries, folded with a plain Python loop
    (no polars import needed) rather than recomputed. ``within_tol`` counts the
    groups that pass; ``total`` is the group count; ``all_within_tol`` is
    ``True`` iff ``total > 0`` and every group passes. For an empty comparison
    this yields ``within_tol=0``, ``total=0``, ``all_within_tol=False``.
    """
    within_by_group: dict[tuple[str, str], bool] = {}
    for row in comparison.rows.iter_rows(named=True):
        key = (row["level"], row["variable"])
        passes = (row["smape_pct"] / 100.0) <= tolerance
        within_by_group[key] = within_by_group.get(key, True) and passes

    total = len(within_by_group)
    within_tol = sum(1 for passes in within_by_group.values() if passes)
    all_within_tol = total > 0 and within_tol == total

    return {
        "stages": int(comparison.stage_count),
        "variables": comparison.summary.to_dicts(),
        "unmapped": {
            level: [int(code) for code in codes]
            for level, codes in comparison.unmapped.items()
        },
        "within_tol": within_tol,
        "total": total,
        "all_within_tol": all_within_tol,
    }


def decomp_dataset_summary(
    dataset: ComparisonDataset, tolerance: float
) -> dict[str, object]:
    """The ``compare decomp`` command's dataset-sourced ``summary`` block.

    Supersedes :func:`decomp_compare_summary` at the ``compare decomp --json``
    call site only (D-STRANGLER: the old summary stays for now, still
    exercised by its own tests; E8 retires it). Returns the shared headline
    fields from :func:`compare_summary` (``within_tol``, ``total``,
    ``worst_variable``, ``worst_smape``, ``all_within_tol`` — sourced via
    :func:`~cobre_bridge.comparators.verdict.build_compare_verdict`, so the
    headline is computed once from ``dataset.summary`` and matches
    ``compare newave``'s) PLUS the three DECOMP-specific keys read straight
    from *dataset*, in this fixed order: ``stages`` (the count of distinct
    ``dataset.tidy["stage"]`` values), ``variables``
    (``dataset.summary.to_dicts()`` verbatim), and ``unmapped``
    (``dataset.metadata["unmapped"]``, with every entity id coerced to
    ``int``).

    *tolerance* is accepted only for call-site symmetry with
    :func:`decomp_compare_summary` — it is not consumed here because
    ``dataset.summary``'s ``within_tol_rate`` (which
    :func:`~cobre_bridge.comparators.verdict.build_compare_verdict` reads) was
    already computed against a tolerance when the caller built *dataset*
    (e.g. via ``build_decomp_dataset(..., tolerance=...)``).
    """
    from cobre_bridge.comparators.verdict import build_compare_verdict

    summary = dict(compare_summary(build_compare_verdict(dataset)))
    summary["stages"] = int(dataset.tidy["stage"].n_unique())
    summary["variables"] = dataset.summary.to_dicts()
    summary["unmapped"] = {
        level: [int(code) for code in codes]
        for level, codes in dataset.metadata["unmapped"].items()
    }
    return summary


def dashboard_summary(output: str, size_kb: float) -> dict[str, object]:
    """The ``dashboard`` command's ``summary`` block — the written artifact.

    ``output`` is the artifact path as a string (the caller passes
    ``str(path)``); ``size_kb`` is its size in kibibytes as a float. Returns
    ``{"output", "size_kb"}``.
    """
    return {"output": output, "size_kb": size_kb}
