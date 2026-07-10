"""Unit and integration tests for the ANALYZE-layer artifact export."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.export import _read_cobre_version, write_artifacts


def _make_dataset() -> ComparisonDataset:
    """Build a small validated dataset with two tidy rows and one summary row."""
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "hydro"],
            "entity_id": [0, 0],
            "entity_name": ["ITAIPU", "ITAIPU"],
            "bus": [-1, -1],
            "stage": [1, 1],
            "block": [-1, -1],
            "variable": ["storage", "storage"],
            "source": ["newave", "cobre"],
            "value": [100.0, 95.0],
        },
        schema=TIDY_SCHEMA,
    )
    summary = pl.DataFrame(
        {
            "variable": ["storage"],
            "count": [1],
            "mean_abs_diff": [5.0],
            "max_abs_diff": [5.0],
            "mean_smape": [0.05],
            "max_smape": [0.05],
            "within_tol_rate": [0.0],
            "correlation": [None],
        },
        schema=SUMMARY_SCHEMA,
    )
    metadata: dict[str, object] = {
        "top_divergences": [
            {
                "entity_type": "hydro",
                "entity_name": "ITAIPU",
                "cobre_id": 0,
                "stage": 1,
                "variable": "storage",
                "newave_value": 100.0,
                "cobre_value": 95.0,
                "abs_diff": 5.0,
                "rel_diff": 0.05,
            }
        ],
    }
    dataset = ComparisonDataset(tidy=tidy, summary=summary, metadata=metadata)
    dataset.validate()
    return dataset


def test_write_artifacts_emits_expected_files(tmp_path: Path) -> None:
    dataset = _make_dataset()
    nw = tmp_path / "newave"
    cb = tmp_path / "output"
    out = tmp_path / "artifacts"

    write_artifacts(
        dataset,
        command="compare newave",
        newave_dir=nw,
        cobre_output_dir=cb,
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json"],
    )

    assert (out / "comparison.parquet").exists()
    assert (out / "summary.parquet").exists()
    assert (out / "summary.json").exists()
    assert (out / "top_divergences.json").exists()
    assert (out / "comparison.json").exists()


def test_manifest_lists_emitted_artifacts(tmp_path: Path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "artifacts"

    manifest = write_artifacts(
        dataset,
        command="compare newave",
        newave_dir=tmp_path / "newave",
        cobre_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json"],
    )

    assert "comparison.json" in manifest.artifacts
    assert "comparison.parquet" in manifest.artifacts
    assert "summary.parquet" in manifest.artifacts
    assert "summary.json" in manifest.artifacts
    assert "top_divergences.json" in manifest.artifacts
    assert manifest.artifacts == sorted(manifest.artifacts)
    assert manifest.command == "compare newave"
    assert manifest.top_divergences == dataset.metadata["top_divergences"]


def test_unknown_format_raises_valueerror(tmp_path: Path) -> None:
    dataset = _make_dataset()

    with pytest.raises(ValueError, match="xml"):
        write_artifacts(
            dataset,
            command="compare newave",
            newave_dir=tmp_path / "newave",
            cobre_output_dir=tmp_path / "output",
            tolerance=1e-2,
            out_dir=tmp_path / "artifacts",
            formats=["xml"],
        )


def test_read_cobre_version_missing_returns_none(tmp_path: Path) -> None:
    cobre_output_dir = tmp_path / "output"
    cobre_output_dir.mkdir()

    assert _read_cobre_version(cobre_output_dir) is None


def test_read_cobre_version_reads_version(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    cobre_output_dir = case_dir / "output"
    training_dir = cobre_output_dir / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "metadata.json").write_text(
        json.dumps({"version": "0.7.0"}), encoding="utf-8"
    )

    assert _read_cobre_version(cobre_output_dir) == "0.7.0"


def test_read_cobre_version_malformed_json_returns_none(tmp_path: Path) -> None:
    cobre_output_dir = tmp_path / "output"
    training_dir = cobre_output_dir / "training"
    training_dir.mkdir(parents=True)
    (training_dir / "metadata.json").write_text("{ not json", encoding="utf-8")

    assert _read_cobre_version(cobre_output_dir) is None


def test_write_artifacts_csv_format(tmp_path: Path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "artifacts"

    manifest = write_artifacts(
        dataset,
        command="compare bounds",
        newave_dir=tmp_path / "newave",
        cobre_output_dir=tmp_path / "output",
        tolerance=1.0,
        out_dir=out,
        formats=["csv"],
    )

    assert (out / "comparison.csv").exists()
    assert (out / "summary.csv").exists()
    assert "comparison.csv" in manifest.artifacts
    assert "summary.csv" in manifest.artifacts
    assert "comparison.json" in manifest.artifacts


def test_write_artifacts_roundtrip_reload(tmp_path: Path) -> None:
    dataset = _make_dataset()
    out = tmp_path / "artifacts"

    write_artifacts(
        dataset,
        command="compare newave",
        newave_dir=tmp_path / "newave",
        cobre_output_dir=tmp_path / "output",
        tolerance=1e-2,
        out_dir=out,
        formats=["parquet", "json"],
    )

    reloaded = ComparisonDataset.from_dir(out)

    assert reloaded.tidy.equals(dataset.tidy)
    assert reloaded.summary.equals(dataset.summary)
