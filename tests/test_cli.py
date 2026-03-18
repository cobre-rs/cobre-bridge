"""Tests for the cobre-bridge CLI and conversion pipeline.

Pipeline unit tests use ``unittest.mock.patch`` to replace the converter
functions with canned return values so no real NEWAVE files are needed.

CLI integration tests use two strategies:
- Error-path tests invoke ``cobre-bridge`` as a subprocess (no mocking needed
  because the process exits before any inewave I/O occurs).
- Success-path and --force tests call ``cli.main()`` in-process via
  ``monkeypatch`` so the pipeline can be patched without a subprocess boundary.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pyarrow as pa
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_cli_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    """Invoke the cobre-bridge entry point as a real subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "cobre_bridge.cli", *args],
        capture_output=True,
        text=True,
    )


def _make_fake_newave_dir(tmp_path: Path) -> Path:
    """Create a directory with all required NEWAVE stub files (empty content)."""
    from cobre_bridge.pipeline import REQUIRED_FILES

    newave_dir = tmp_path / "newave_case"
    newave_dir.mkdir()
    for filename in REQUIRED_FILES:
        (newave_dir / filename).write_text("stub")
    return newave_dir


# ---------------------------------------------------------------------------
# ConversionReport
# ---------------------------------------------------------------------------


class TestConversionReport:
    def test_str_format(self) -> None:
        from cobre_bridge.pipeline import ConversionReport

        report = ConversionReport(
            hydro_count=3,
            thermal_count=5,
            bus_count=4,
            line_count=2,
            stage_count=60,
        )
        s = str(report)
        assert "3 hydros" in s
        assert "5 thermals" in s
        assert "4 buses" in s
        assert "2 lines" in s
        assert "60 stages" in s

    def test_default_zeros(self) -> None:
        from cobre_bridge.pipeline import ConversionReport

        report = ConversionReport()
        assert report.hydro_count == 0
        assert report.thermal_count == 0
        assert report.bus_count == 0
        assert report.line_count == 0
        assert report.stage_count == 0
        assert report.warnings == []


# ---------------------------------------------------------------------------
# Pipeline unit tests (all converters mocked)
# ---------------------------------------------------------------------------

# Canned return values used across pipeline tests.
_FAKE_HYDROS = {"$schema": "http://example", "hydros": [{"id": 0}, {"id": 1}]}
_FAKE_THERMALS = {"$schema": "http://example", "thermals": [{"id": 0}]}
_FAKE_BUSES = {"$schema": "http://example", "buses": [{"id": 0}, {"id": 1}, {"id": 2}]}
_FAKE_LINES = {"$schema": "http://example", "lines": [{"id": 0}]}
_FAKE_PENALTIES = {"bus": {}, "hydro": {}, "line": {}, "non_controllable_source": {}}
_FAKE_STAGES = {
    "$schema": "http://example",
    "policy_graph": {"type": "finite_horizon"},
    "stages": [{"id": i} for i in range(12)],
}
_FAKE_CONFIG = {
    "$schema": "http://example",
    "training": {"forward_passes": 5, "stopping_rules": []},
    "simulation": {"enabled": True, "num_scenarios": 200},
}
_FAKE_IC = {"$schema": "http://example", "storage": [], "filling_storage": []}
_FAKE_INFLOW_TABLE = pa.table(
    {
        "hydro_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "mean_m3s": pa.array([100.0], type=pa.float64()),
        "std_m3s": pa.array([10.0], type=pa.float64()),
    }
)
_FAKE_LOAD_TABLE = pa.table(
    {
        "bus_id": pa.array([0], type=pa.int32()),
        "stage_id": pa.array([0], type=pa.int32()),
        "mean_mw": pa.array([500.0], type=pa.float64()),
        "std_mw": pa.array([0.0], type=pa.float64()),
    }
)


def _all_converter_patches(fake_id_map: MagicMock) -> list:  # type: ignore[type-arg]
    """Return patch context managers for all converter functions and _build_id_map."""
    return [
        patch("cobre_bridge.pipeline._build_id_map", return_value=fake_id_map),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_hydros",
            return_value=_FAKE_HYDROS,
        ),
        patch(
            "cobre_bridge.pipeline.thermal_conv.convert_thermals",
            return_value=_FAKE_THERMALS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_buses",
            return_value=_FAKE_BUSES,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_lines",
            return_value=_FAKE_LINES,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_penalties",
            return_value=_FAKE_PENALTIES,
        ),
        patch(
            "cobre_bridge.pipeline.temporal_conv.convert_stages",
            return_value=_FAKE_STAGES,
        ),
        patch(
            "cobre_bridge.pipeline.temporal_conv.convert_config",
            return_value=_FAKE_CONFIG,
        ),
        patch(
            "cobre_bridge.pipeline.ic_conv.convert_initial_conditions",
            return_value=_FAKE_IC,
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_inflow_stats",
            return_value=_FAKE_INFLOW_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_load_stats",
            return_value=_FAKE_LOAD_TABLE,
        ),
    ]


def _run_with_all_mocks(src: Path, dst: Path) -> object:
    """Run convert_newave_case with all converters replaced by canned fakes."""
    from cobre_bridge.pipeline import convert_newave_case

    fake_id_map = MagicMock()
    patches = _all_converter_patches(fake_id_map)
    for p in patches:
        p.__enter__()
    try:
        return convert_newave_case(src, dst)
    finally:
        for p in patches:
            p.__exit__(None, None, None)


class TestConvertNewaweCasePipeline:
    """Unit tests for pipeline.convert_newave_case with all converters mocked."""

    def test_all_output_files_written(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        expected = [
            dst / "config.json",
            dst / "stages.json",
            dst / "penalties.json",
            dst / "initial_conditions.json",
            dst / "system" / "hydros.json",
            dst / "system" / "thermals.json",
            dst / "system" / "buses.json",
            dst / "system" / "lines.json",
            dst / "scenarios" / "inflow_seasonal_stats.parquet",
            dst / "scenarios" / "load_seasonal_stats.parquet",
        ]
        for f in expected:
            assert f.exists(), f"Expected output file not found: {f}"

    def test_json_files_are_valid_json(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        for json_path in [
            dst / "config.json",
            dst / "stages.json",
            dst / "system" / "hydros.json",
            dst / "system" / "thermals.json",
            dst / "system" / "buses.json",
            dst / "system" / "lines.json",
        ]:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
            assert data is not None, f"Invalid JSON: {json_path}"

    def test_parquet_files_are_readable(self, tmp_path: Path) -> None:
        import pyarrow.parquet as pq

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        inflow = pq.read_table(dst / "scenarios" / "inflow_seasonal_stats.parquet")
        assert inflow.num_columns == 4
        load = pq.read_table(dst / "scenarios" / "load_seasonal_stats.parquet")
        assert load.num_columns == 4

    def test_report_counts_from_converter_output(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        report = _run_with_all_mocks(src, dst)

        assert report.hydro_count == 2  # type: ignore[union-attr]
        assert report.thermal_count == 1
        assert report.bus_count == 3
        assert report.line_count == 1
        assert report.stage_count == 12

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        from cobre_bridge.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "cobre_case"

        with pytest.raises(FileNotFoundError) as exc_info:
            convert_newave_case(src, dst)
        assert "hidr.dat" in str(exc_info.value)


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCliExitCodes:
    """Subprocess-based tests for error paths that don't require inewave I/O."""

    def test_exit_code_1_when_src_missing(self, tmp_path: Path) -> None:
        dst = tmp_path / "dst"
        result = _run_cli_subprocess(
            "convert",
            "newave",
            str(tmp_path / "nonexistent"),
            str(dst),
        )
        assert result.returncode == 1
        assert "does not exist" in result.stderr

    def test_exit_code_1_when_dst_nonempty_no_force(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))
        assert result.returncode == 1
        assert "not empty" in result.stderr

    def test_exit_code_1_when_required_file_missing(self, tmp_path: Path) -> None:
        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "dst"

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))
        assert result.returncode == 1
        assert "hidr.dat" in result.stderr


class TestCliInProcess:
    """In-process CLI tests that patch the pipeline to avoid inewave I/O."""

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
        """Run cli.main() in-process, capturing stdout/stderr and exit code."""
        import io

        from cobre_bridge import cli

        monkeypatch.setattr(sys, "argv", ["cobre-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    def test_exit_code_0_with_force_on_nonempty_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "existing.txt").write_text("hello")

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--force"],
                monkeypatch,
            )

        assert code == 0
        assert "1 hydros" in stdout

    def test_stdout_contains_converted_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _ = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "10 hydros" in stdout
        assert "5 thermals" in stdout
        assert "60 stages" in stdout
