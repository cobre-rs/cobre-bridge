"""Tests for the fail-loud contract of the Cobre output readers.

ROBUST-01: a read/parse/aggregation failure on an **already-existing**
Cobre output file must raise :class:`CobreReadError` (so the comparison
engine never reports a false "no divergence" on unreadable data), while a
genuinely **absent** optional output must still yield an empty frame.  The
CLI maps :class:`CobreReadError` to exit code 2 (distinct from exit 1 =
"mismatches found").
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from cobre_bridge.comparators.cobre_readers import (
    CobreReadError,
    read_cobre_bus_means,
    read_cobre_hydro_means,
    read_cobre_hydro_per_stage_bounds,
    read_cobre_hydro_withdrawal,
    read_cobre_line_means,
    read_cobre_lp_max_generation,
    read_cobre_thermal_means,
    read_cobre_thermal_metadata,
)

# ---------------------------------------------------------------------------
# Fixtures: build corrupt vs. valid-but-empty Cobre output layouts on disk.
# ---------------------------------------------------------------------------

_GARBAGE = b"this is definitely not a valid parquet file -- no PAR1 footer"


def _write_corrupt_sim_parquet(output_dir: Path, entity: str) -> Path:
    """Create ``output/simulation/<entity>/scenario_id=0000/data.parquet``
    containing non-parquet garbage and return its path.
    """
    p = output_dir / "simulation" / entity / "scenario_id=0000" / "data.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_GARBAGE)
    return p


def _write_corrupt_parquet(path: Path) -> Path:
    """Create a corrupt parquet at *path* (parents made) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_GARBAGE)
    return path


# ---------------------------------------------------------------------------
# Corrupt *existing* file -> CobreReadError.
# ---------------------------------------------------------------------------


class TestCorruptExistingRaises:
    def test_hydro_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "hydros")
        with pytest.raises(CobreReadError, match="scan parquets"):
            read_cobre_hydro_means(out)

    def test_thermal_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "thermals")
        with pytest.raises(CobreReadError):
            read_cobre_thermal_means(out)

    def test_bus_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "buses")
        with pytest.raises(CobreReadError):
            read_cobre_bus_means(out)

    def test_line_means_corrupt_parquet_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_sim_parquet(out, "exchanges")
        with pytest.raises(CobreReadError):
            read_cobre_line_means(out)

    def test_lp_max_generation_corrupt_bounds_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(out / "training" / "dictionaries" / "bounds.parquet")
        with pytest.raises(CobreReadError, match="bounds.parquet"):
            read_cobre_lp_max_generation(out)

    def test_hydro_withdrawal_corrupt_constraints_raises(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(tmp_path / "constraints" / "hydro_bounds.parquet")
        with pytest.raises(CobreReadError, match="hydro_bounds.parquet"):
            read_cobre_hydro_withdrawal(out)

    def test_hydro_per_stage_bounds_corrupt_constraints_raises(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        _write_corrupt_parquet(tmp_path / "constraints" / "hydro_bounds.parquet")
        with pytest.raises(CobreReadError, match="hydro_bounds.parquet"):
            read_cobre_hydro_per_stage_bounds(out)


# ---------------------------------------------------------------------------
# Absent file -> empty DataFrame (no raise).  PRESERVED behavior.
# ---------------------------------------------------------------------------


class TestAbsentReturnsEmpty:
    def test_hydro_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_hydro_means(out)
        assert df.is_empty()

    def test_thermal_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_thermal_means(out)
        assert df.is_empty()

    def test_bus_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_bus_means(out)
        assert df.is_empty()

    def test_line_means_absent_dir_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_line_means(out)
        assert df.is_empty()

    def test_lp_max_generation_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_lp_max_generation(out)
        assert df.is_empty()

    def test_hydro_withdrawal_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_hydro_withdrawal(out)
        assert df.is_empty()

    def test_hydro_per_stage_bounds_absent_file_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "output"
        out.mkdir()
        df = read_cobre_hydro_per_stage_bounds(out)
        assert df.is_empty()


# ---------------------------------------------------------------------------
# Soft (metadata) reader stays soft: locks in the judgment call.
# ---------------------------------------------------------------------------


class TestSoftMetadataReadersStaySoft:
    def test_thermal_metadata_corrupt_json_returns_empty(self, tmp_path: Path) -> None:
        # thermals.json is metadata (entity names only), not correctness-
        # bearing comparison data — a malformed file degrades to {}.
        out = tmp_path / "output"
        sysdir = tmp_path / "system"
        sysdir.mkdir(parents=True)
        (sysdir / "thermals.json").write_text("{ this is not valid json ")
        out.mkdir()
        assert read_cobre_thermal_metadata(out) == {}


# ---------------------------------------------------------------------------
# CLI maps CobreReadError -> exit code 2 (distinct from 1 = mismatches).
# ---------------------------------------------------------------------------


class TestCliExitCodeTwoOnCobreReadError:
    def _bounds_args(self, tmp_path: Path) -> argparse.Namespace:
        out = tmp_path / "output"
        # _run_bounds_comparison requires bounds.parquet to *exist* before it
        # reaches the comparison call we are exercising.
        _write_corrupt_parquet(out / "training" / "dictionaries" / "bounds.parquet")
        return argparse.Namespace(
            newave_dir=tmp_path / "newave",
            cobre_output_dir=out,
            tolerance=1e-3,
            output=None,
            summary=True,
            variables=None,
            verbose=False,
        )

    def test_bounds_cli_exits_2_when_reader_raises(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge import cli

        args = self._bounds_args(tmp_path)

        def _raise(**_kwargs: object) -> object:
            raise CobreReadError("Failed to read bounds.parquet: /x/bounds.parquet")

        with (
            patch("cobre_bridge.cli._load_lines_json", return_value=[]),
            patch(
                "cobre_bridge.newave_files.NewaveFiles.from_directory",
                return_value=object(),
            ),
            patch("cobre_bridge.pipeline._build_id_map", return_value=object()),
            patch(
                "cobre_bridge.comparators.alignment.build_entity_alignment",
                return_value=object(),
            ),
            patch(
                "cobre_bridge.comparators.bounds.compare_bounds",
                side_effect=_raise,
            ),
        ):
            with pytest.raises(SystemExit) as excinfo:
                cli._run_bounds_comparison(args)

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "ERROR:" in err
        assert "bounds.parquet" in err

    def test_results_cli_exits_2_when_reader_raises(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from cobre_bridge import cli

        args = argparse.Namespace(
            newave_dir=tmp_path / "newave",
            cobre_output_dir=tmp_path / "output",
            tolerance=1e-2,
            output=None,
            verbose=False,
        )

        def _raise(**_kwargs: object) -> object:
            raise CobreReadError("Failed to aggregate hydro simulation data: /x/hydros")

        with (
            patch("cobre_bridge.cli._load_lines_json", return_value=[]),
            patch(
                "cobre_bridge.newave_files.NewaveFiles.from_directory",
                return_value=object(),
            ),
            patch("cobre_bridge.pipeline._build_id_map", return_value=object()),
            patch(
                "cobre_bridge.comparators.alignment.build_entity_alignment",
                return_value=object(),
            ),
            patch(
                "cobre_bridge.comparators.results.compare_results",
                side_effect=_raise,
            ),
        ):
            with pytest.raises(SystemExit) as excinfo:
                cli._run_results_comparison(args)

        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "ERROR:" in err
        assert "hydro simulation data" in err


# ---------------------------------------------------------------------------
# End-to-end-ish: a corrupt existing parquet drives the reader through the
# real (unpatched) code path to CobreReadError, proving the antipattern fix
# rather than only the CLI wiring.
# ---------------------------------------------------------------------------


class TestCorruptDrivesRealReaderPath:
    def test_real_reader_raises_on_corrupt_then_empty_on_absent(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "output"
        bounds = out / "training" / "dictionaries" / "bounds.parquet"

        # Corrupt-but-existing -> raises.
        _write_corrupt_parquet(bounds)
        with pytest.raises(CobreReadError):
            read_cobre_lp_max_generation(out)

        # Same reader, file removed -> empty, no raise.
        bounds.unlink()
        result = read_cobre_lp_max_generation(out)
        assert isinstance(result, pl.DataFrame)
        assert result.is_empty()
