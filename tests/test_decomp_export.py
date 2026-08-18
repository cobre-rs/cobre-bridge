"""Unit tests for the ``compare decomp`` artifact export."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cobre_bridge.comparators.decomp_export import (
    DecompComparisonManifest,
    write_decomp_artifacts,
)
from cobre_bridge.comparators.decomp_results import DecompComparison, _summarize


def _fake_comparison() -> DecompComparison:
    """A hydro-generation row, a thermal-generation row, and two iterations."""
    rows = pl.DataFrame(
        {
            "level": ["hydro", "thermal"],
            "variable": ["generation", "generation"],
            "unit": ["MW", "MW"],
            "entity_id": [0, 0],
            "entity_name": ["A", "T"],
            "stage_id": [0, 0],
            "source": [100.0, 20.0],
            "cobre": [90.0, 20.0],
            "delta": [-10.0, 0.0],
            "delta_pct": [-10.0, 0.0],
            "smape_pct": [10.5, 0.0],
        }
    )
    return DecompComparison(
        rows=rows,
        summary=_summarize(rows),
        convergence=pl.DataFrame(
            {
                "iteration": [1, 2],
                "source_lower": [1.0, 2.0],
                "source_upper": [3.0, 3.0],
                "cobre_lower": [10.0, None],
                "cobre_upper": [30.0, None],
            }
        ),
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


def test_write_decomp_artifacts_emits_expected_files(tmp_path: Path) -> None:
    comparison = _fake_comparison()
    src = tmp_path / "decomp"
    cobre = tmp_path / "output"
    out = tmp_path / "artifacts"

    write_decomp_artifacts(
        comparison,
        command="compare decomp",
        decomp_dir=src,
        cobre_output_dir=cobre,
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json", "csv"],
    )

    assert (out / "comparison.parquet").exists()
    assert (out / "summary.parquet").exists()
    assert (out / "convergence.parquet").exists()
    assert (out / "summary.json").exists()
    assert (out / "convergence.json").exists()
    assert (out / "comparison.csv").exists()
    assert (out / "summary.csv").exists()
    assert (out / "convergence.csv").exists()
    assert (out / "comparison.json").exists()


def test_manifest_records_command_tolerance_unmapped_and_artifacts(
    tmp_path: Path,
) -> None:
    comparison = _fake_comparison()
    src = tmp_path / "decomp"
    cobre = tmp_path / "output"
    out = tmp_path / "artifacts"

    write_decomp_artifacts(
        comparison,
        command="compare decomp",
        decomp_dir=src,
        cobre_output_dir=cobre,
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json", "csv"],
    )

    payload = json.loads((out / "comparison.json").read_text(encoding="utf-8"))
    assert payload["command"] == "compare decomp"
    assert payload["tolerance"] == 1e-2
    assert payload["unmapped"] == {"hydro": [], "thermal": [86, 224], "bus": []}
    assert payload["artifacts"] == sorted(payload["artifacts"])
    assert "comparison.json" in payload["artifacts"]


def test_unknown_format_raises_valueerror(tmp_path: Path) -> None:
    comparison = _fake_comparison()

    with pytest.raises(ValueError, match="xml"):
        write_decomp_artifacts(
            comparison,
            command="compare decomp",
            decomp_dir=tmp_path / "decomp",
            cobre_output_dir=tmp_path / "output",
            tolerance=1e-2,
            out_dir=tmp_path / "artifacts",
            formats=["xml"],
        )


def test_empty_comparison_still_writes_typed_artifacts(tmp_path: Path) -> None:
    empty_rows = _fake_comparison().rows.clear()
    empty_comparison = DecompComparison(
        rows=empty_rows,
        summary=_summarize(empty_rows),
        convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
        unmapped={"hydro": [], "thermal": [], "bus": []},
    )
    out = tmp_path / "artifacts"

    manifest = write_decomp_artifacts(
        empty_comparison,
        command="compare decomp",
        decomp_dir=tmp_path / "decomp",
        cobre_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json", "csv"],
    )

    assert (out / "comparison.parquet").exists()
    assert (out / "summary.parquet").exists()
    assert (out / "convergence.parquet").exists()
    assert (out / "comparison.json").exists()
    assert manifest.unmapped == {"hydro": [], "thermal": [], "bus": []}


def test_write_decomp_artifacts_returns_manifest_instance(tmp_path: Path) -> None:
    comparison = _fake_comparison()
    out = tmp_path / "artifacts"

    manifest = write_decomp_artifacts(
        comparison,
        command="compare decomp",
        decomp_dir=tmp_path / "decomp",
        cobre_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["json"],
    )

    assert isinstance(manifest, DecompComparisonManifest)
    assert manifest.decomp_dir == str(tmp_path / "decomp")
    assert manifest.cobre_output_dir == str(tmp_path / "output")
    assert manifest.bridge_version
    assert manifest.timestamp
