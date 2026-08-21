"""``compare decomp`` renders the full shared per-variable table.

Point ``compare decomp`` at the same ``print_results_summary_from_dataset``
renderer ``compare newave`` uses (``reference_label="DECOMP"``), console-
threaded and ``--quiet``-gated. Tier 1 -- pure Python, no ``cobre`` import, no
``example/`` deck: the dataset build is stubbed via the same
``cobre_bridge.comparators.decomp_results.build_decomp_dataset`` monkeypatch
seam ``TestCompareDecompCommand`` (``tests/test_decomp_results_compare.py``)
and ``TestCompareDiagnosticsWiring`` (``tests/test_compare.py``) already use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from typer.testing import CliRunner

if TYPE_CHECKING:
    from cobre_bridge.comparators.dataset import ComparisonDataset


def _two_variable_dataset() -> ComparisonDataset:
    """A hermetic dataset with two variables, built through the shared
    ``analyze.build_results_dataset`` assembly kernel (as
    ``build_decomp_dataset`` does), so ``footer_counts`` metadata and the
    per-variable ``summary`` rows the renderer reads are both populated."""
    from cobre_bridge.comparators.analyze import build_results_dataset
    from cobre_bridge.comparators.results import PercentileData, ResultComparison

    results = [
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            cobre_id=0,
            stage=0,
            variable="generation_mw",
            newave_value=100.0,
            cobre_value=110.0,
            abs_diff=10.0,
            rel_diff=0.1,
        ),
        ResultComparison(
            entity_type="hydro",
            entity_name="ITAIPU",
            newave_code=10,
            cobre_id=0,
            stage=0,
            variable="turbined_m3s",
            newave_value=200.0,
            cobre_value=180.0,
            abs_diff=20.0,
            rel_diff=0.1,
        ),
    ]
    dataset = build_results_dataset(results, PercentileData(), 1e-2)
    dataset.metadata["unmapped"] = {"hydro": [], "thermal": [], "bus": []}
    return dataset


def _invoke(argv_tail: list[str], monkeypatch: pytest.MonkeyPatch, deck: Path) -> Any:
    """Invoke ``compare decomp`` through the real Typer app via ``CliRunner``,
    with the dataset build stubbed at its public entry point."""
    from cobre_bridge.cli import app
    from tests.conftest import make_decomp_case

    # ``DecompCase.from_directory`` is re-invoked (for manifest hashing) after
    # ``build_decomp_dataset`` is mocked away; give it a real ``DecompFiles``
    # dataclass instead of trying to discover a deck under the fake ``deck`` dir.
    monkeypatch.setattr(
        "cobre_bridge.decomp.case.DecompCase.from_directory",
        classmethod(lambda cls, _dir: make_decomp_case(Path("decomp"))),
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.build_decomp_dataset",
        lambda *_args, **_kwargs: _two_variable_dataset(),
    )
    argv = ["compare", "decomp", str(deck), str(deck), *argv_tail]
    return CliRunner().invoke(app, argv)


def test_compare_decomp_renders_the_full_shared_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1: stdout carries the ``DECOMP``-labelled header and a table row per
    compared variable -- not just the one-line headline."""
    result = _invoke([], monkeypatch, tmp_path)

    assert result.exit_code == 0
    assert "Cobre vs DECOMP Results Comparison" in result.stdout
    assert "generation_mw" in result.stdout
    assert "turbined_m3s" in result.stdout


def test_compare_decomp_quiet_suppresses_header_and_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2: ``--quiet`` suppresses both the header and the per-variable table."""
    result = _invoke(["--quiet"], monkeypatch, tmp_path)

    assert result.exit_code == 0
    assert "Cobre vs DECOMP Results Comparison" not in result.stdout
    assert "generation_mw" not in result.stdout
    assert "turbined_m3s" not in result.stdout


def test_compare_decomp_json_emits_one_verdict_and_suppresses_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC3: ``--json`` emits exactly one JSON verdict on stdout (the full
    table is suppressed) and the process exits 0."""
    result = _invoke(["--json"], monkeypatch, tmp_path)

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["command"] == "compare decomp"
    assert "Cobre vs DECOMP Results Comparison" not in result.stdout
