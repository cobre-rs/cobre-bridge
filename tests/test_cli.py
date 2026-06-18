"""Tests for the cobre-bridge CLI and conversion pipeline.

Pipeline unit tests use ``unittest.mock.patch`` to replace the converter functions with
canned return values so no real the source model files are needed.

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

# Fake data for converter functions.
_FAKE_LOAD_FACTORS: dict = {"load_factors": []}
_FAKE_LINE_BOUNDS_TABLE = pa.table(
    {
        "line_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "direct_mw": pa.array([], type=pa.float64()),
        "reverse_mw": pa.array([], type=pa.float64()),
    }
)
_FAKE_NCS: dict = {"non_controllable_sources": []}
_FAKE_EXCHANGE_FACTORS: dict = {"exchange_factors": []}
_FAKE_NCS_FACTORS: dict = {"non_controllable_factors": []}
_FAKE_NCS_BOUNDS_TABLE = pa.table(
    {
        "ncs_id": pa.array([], type=pa.int32()),
        "stage_id": pa.array([], type=pa.int32()),
        "available_generation_mw": pa.array([], type=pa.float64()),
    }
)

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


_ARQUIVOS_DAT_CONTENT = """\
DADOS GERAIS                : dger.dat
DADOS DOS SUBSISTEMAS       : sistema.dat
CONFIGURACAO HIDRAULICA     : confhd.dat
ALTERACAO DADOS USINAS HIDRO: modif.dat
CONFIGURACAO TERMICA        : conft.dat
DADOS DAS USINAS TERMICAS   : term.dat
DADOS DAS CLASSES TERMICAS  : clast.dat
DADOS DE EXPANSAO HIDRAULICA: exph.dat
ARQUIVO DE EXPANSAO TERMICA : expt.dat
ARQUIVO DE PATAMARES MERCADO: patamar.dat
ARQUIVO DE CORTES DE BENDERS: cortes.dat
ARQUIVO DE CABECALHO CORTES : cortesh.dat
RELATORIO DE CONVERGENCIA   : pmo.dat
RELATORIO DE E. SINTETICAS  : parp.dat
RELATORIO DETALHADO FORWARD : forward.dat
ARQUIVO DE CABECALHO FORWARD: forwarh.dat
ARQUIVO DE S.HISTORICAS S.F.: shist.dat
ARQUIVO DE MANUT.PROG. UTE'S: manutt.dat
ARQUIVO P/DESPACHO HIDROTERM: newdesp.dat
ARQUIVO C/TEND. HIDROLOGICA : vazpast.dat
ARQUIVO C/DADOS DE ITAIPU   : itaipu.dat
ARQUIVO C/DEMAND S. BIDDING : bid.dat
ARQUIVO C/CARGAS ADICIONAIS : c_adic.dat
ARQUIVO C/FATORES DE PERDAS : loss.dat
ARQUIVO C/PATAMARES GTMIN   : gtminpat.dat
ARQUIVO ENSO 1              : elnino.dat
ARQUIVO ENSO 2              : ensoaux.dat
ARQUIVO DSVAGUA             : dsvagua.dat
ARQUIVO P/PENALID. POR DESV.: penalid.dat
ARQUIVO C.GUIA / PENAL.VMINT: curva.dat
ARQUIVO AGRUPAMENTO LIVRE   : agrint.dat
ARQUIVO DESP. ANTEC. GNL    : adterm.dat
ARQUIVO GER. HIDR. MIN      : ghmin.dat
ARQUIVO AVERSAO RISCO - SAR : sar.dat
ARQUIVO AVERSAO RISCO - CVAR: cvar.dat
DADOS DOS RESER.EQ.ENERGIA  : ree.dat
ARQUIVO RESTRICOES ELETRICAS: re.dat
ARQUIVO DE TECNOLOGIAS      : tecno.dat
DADOS DE ABERTURAS          : abertura.dat
ARQUIVO DE EMISSOES GEE     : gee.dat
ARQUIVO DE RESTRICAO DE GAS : clasgas.dat
ARQUIVO DE DADOS SIM. FINAL : simfinal.dat
ARQ. DE CORTES POS ESTUDO   : cortes-pos.dat
ARQ. DE CABECALHO CORTES POS: cortesh-pos.dat
ARQ. C/ VOLUME REF. SAZONAL : volref_saz.dat
"""

_REQUIRED_STUB_FILES = [
    "dger.dat",
    "confhd.dat",
    "conft.dat",
    "sistema.dat",
    "clast.dat",
    "term.dat",
    "ree.dat",
    "patamar.dat",
    "hidr.dat",
    "vazoes.dat",
]


def _make_fake_newave_dir(tmp_path: Path) -> Path:
    """Create a directory with caso.dat, arquivos.dat, and all required stub files."""
    newave_dir = tmp_path / "newave_case"
    newave_dir.mkdir()
    (newave_dir / "caso.dat").write_text("arquivos.dat\n")
    (newave_dir / "arquivos.dat").write_text(_ARQUIVOS_DAT_CONTENT)
    for filename in _REQUIRED_STUB_FILES:
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
_FAKE_HYDRO_ENERGY_PRODUCTIVITY_TABLE = pa.table(
    {
        "hydro_id": pa.array([0, 1], type=pa.int32()),
        "stage_id": pa.array([None, None], type=pa.int32()),
        "equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64()),
        "reference_outflow_m3s": pa.array([None, None], type=pa.float64()),
        "specific_productivity_mw_per_m3s_per_m": pa.array(
            [None, None], type=pa.float64()
        ),
    }
)


def _all_converter_patches(fake_id_map: MagicMock) -> list:  # type: ignore[type-arg]
    """Return patch context managers for all converter functions.

    The parsed case is mocked at ``NewaveCase.from_directory``; its ``id_map``
    is the supplied ``fake_id_map`` (the pipeline now reads ``case.id_map``).
    """
    fake_case = MagicMock()
    fake_case.id_map = fake_id_map
    return [
        patch(
            "cobre_bridge.pipeline.NewaveCase.from_directory",
            return_value=fake_case,
        ),
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
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_recent_inflow_lags",
            return_value=[],
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_inflow_history",
            return_value=_FAKE_INFLOW_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.read_cadastro",
            return_value=MagicMock(),
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.generate_hydro_geometry",
            return_value=_FAKE_INFLOW_TABLE,  # reuse any small pa.Table
        ),
        patch(
            "cobre_bridge.pipeline.constraints_conv.convert_vminop_constraints",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.constraints_conv.convert_electric_constraints",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.constraints_conv.convert_agrint_constraints",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.stochastic_conv.convert_load_factors",
            return_value=_FAKE_LOAD_FACTORS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_line_bounds",
            return_value=_FAKE_LINE_BOUNDS_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_non_controllable_sources",
            return_value=_FAKE_NCS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_exchange_factors",
            return_value=_FAKE_EXCHANGE_FACTORS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_ncs_factors",
            return_value=_FAKE_NCS_FACTORS,
        ),
        patch(
            "cobre_bridge.pipeline.network_conv.convert_ncs_stats",
            return_value=_FAKE_NCS_BOUNDS_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_production_models",
            return_value={"production_models": []},
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.compute_base_productivities",
            return_value={},
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_hydro_energy_productivity",
            return_value=_FAKE_HYDRO_ENERGY_PRODUCTIVITY_TABLE,
        ),
        patch(
            "cobre_bridge.pipeline.thermal_conv.convert_thermal_bounds",
            return_value=None,
        ),
        patch(
            "cobre_bridge.pipeline.hydro_conv.convert_storage_bounds",
            return_value=None,
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
            dst / "scenarios" / "inflow_history.parquet",
            dst / "system" / "hydro_geometry.parquet",
            dst / "scenarios" / "load_factors.json",
            dst / "constraints" / "line_bounds.parquet",
            dst / "system" / "non_controllable_sources.json",
            dst / "constraints" / "exchange_factors.json",
            dst / "scenarios" / "non_controllable_factors.json",
            dst / "scenarios" / "non_controllable_stats.parquet",
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

    def test_production_models_written_when_converter_returns_data(
        self, tmp_path: Path
    ) -> None:
        """When convert_production_models returns data, the file is written."""
        from cobre_bridge.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _FAKE_PROD_MODELS = {
            "production_models": [
                {
                    "hydro_id": 0,
                    "selection_mode": "stage_ranges",
                    "stage_ranges": [
                        {
                            "start_stage_id": 0,
                            "end_stage_id": None,
                            "model": "constant_productivity",
                            "productivity_mw_per_m3s": 1.23,
                        }
                    ],
                }
            ]
        }

        fake_id_map = MagicMock()
        # Build patches with production_models returning data.
        # Use ExitStack for correct LIFO teardown to avoid mock leakage.
        import contextlib

        patches = _all_converter_patches(fake_id_map)
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            # Override the production_models patch (entered last -> exits first).
            stack.enter_context(
                patch(
                    "cobre_bridge.pipeline.hydro_conv.convert_production_models",
                    return_value=_FAKE_PROD_MODELS,
                )
            )
            convert_newave_case(src, dst)

        pm_path = dst / "system" / "hydro_production_models.json"
        assert pm_path.exists(), "hydro_production_models.json not written"
        with pm_path.open(encoding="utf-8") as f:
            data = json.load(f)
        assert data["production_models"][0]["hydro_id"] == 0

    def test_production_models_always_written(self, tmp_path: Path) -> None:
        """Cobre HEAD requires hydro_production_models.json — pipeline always writes it.

        Productivity moved out of `hydros.json:generation`, so the production
        models file is now mandatory for the converted case to load in cobre.
        """
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        assert (dst / "system" / "hydro_production_models.json").exists()

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        from cobre_bridge.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        with patch(
            "cobre_bridge.pipeline.NewaveCase.from_directory",
            side_effect=FileNotFoundError(
                f"Required NEWAVE file not found in {src}: hidr.dat"
            ),
        ):
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

    def test_convert_missing_file_renders_error_diagnostic_subprocess(
        self, tmp_path: Path
    ) -> None:
        """A real discovery failure renders an ERROR diagnostic (✖) naming the file."""
        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "dst"

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst))
        assert result.returncode == 1
        assert "✖" in result.stderr
        assert "hidr.dat" in result.stderr

    def test_convert_json_missing_source_emits_error_json_subprocess(
        self, tmp_path: Path
    ) -> None:
        """``--json`` on a source missing a required file emits error JSON to stdout."""
        src = _make_fake_newave_dir(tmp_path)
        (src / "hidr.dat").unlink()
        dst = tmp_path / "dst"

        result = _run_cli_subprocess("convert", "newave", str(src), str(dst), "--json")

        assert result.returncode == 1
        doc = json.loads(result.stdout)
        assert doc["command"] == "convert newave"
        assert doc["status"] == "error"
        assert doc["summary"]["hydros"] == 0
        assert doc["diagnostics"]
        # The Rich diagnostic block must not also render on stderr.
        assert "✖" not in result.stderr

    def test_convert_without_source_exits_nonzero(self) -> None:
        """``convert`` with no SOURCE must error (exit 2), not silently succeed."""
        result = _run_cli_subprocess("convert")
        assert result.returncode == 2

    def test_compare_without_source_exits_nonzero(self) -> None:
        """``compare`` with no SOURCE must error (exit 2), not silently succeed."""
        result = _run_cli_subprocess("compare")
        assert result.returncode == 2


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

    def test_convert_failure_renders_error_diagnostic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pipeline exception renders an ERROR diagnostic on stderr; exit 1."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            side_effect=ValueError("boom"),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 1
        # The failure is an ERROR-styled diagnostic on stderr, not stdout.
        assert "✖" in stderr
        assert "Conversion failed" in stderr
        assert "boom" in stderr
        assert "boom" not in stdout

    def test_convert_json_document_shape(self) -> None:
        """``_convert_json_document`` builds the verdict schema with fixed order."""
        from cobre_bridge.cli import _convert_json_document
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.pipeline import ConversionReport

        report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
        )
        info = Diagnostic(
            code="some-info",
            severity=Severity.INFO,
            category="Conversion",
            title="An info",
            summary="just so",
        )

        doc = _convert_json_document(report, [info])

        assert list(doc.keys()) == ["command", "status", "summary", "diagnostics"]
        assert doc["command"] == "convert newave"
        # No ERROR diagnostic → status is "ok".
        assert doc["status"] == "ok"
        assert doc["summary"] == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
        }
        assert doc["diagnostics"] == [info.to_dict()]

    def test_convert_json_document_error_status_on_error_diagnostic(self) -> None:
        """Any ERROR-severity diagnostic flips ``status`` to ``"error"``."""
        from cobre_bridge.cli import _convert_json_document
        from cobre_bridge.diagnostics import Diagnostic, Severity

        error = Diagnostic(
            code="boom-code",
            severity=Severity.ERROR,
            category="Conversion failure",
            title="Conversion failed",
            summary="boom",
        )

        doc = _convert_json_document(None, [error])

        assert doc["status"] == "error"
        # Failure path → zero counts.
        assert doc["summary"] == {
            "hydros": 0,
            "thermals": 0,
            "buses": 0,
            "lines": 0,
            "stages": 0,
        }
        assert doc["diagnostics"] == [error.to_dict()]

    def test_convert_json_success_emits_stdout_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` on a successful conversion emits one JSON verdict to stdout."""
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
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        assert doc["command"] == "convert newave"
        assert doc["status"] == "ok"
        assert doc["summary"]["hydros"] == 10
        # No Rich human summary leaked onto stdout.
        assert "Converted" not in stdout
        # Nothing on stderr on the success/diagnostics path.
        assert stderr == ""

    def test_convert_json_failure_emits_error_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` on a pipeline failure emits ``status == "error"``; exit 1."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            side_effect=ValueError("boom"),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert doc["status"] == "error"
        assert doc["summary"]["hydros"] == 0
        assert len(doc["diagnostics"]) == 1
        assert doc["diagnostics"][0]["summary"] == "boom"
        # The Rich diagnostic block must not also render.
        assert "✖" not in stderr

    def test_convert_json_coexists_with_diagnostics_json_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` and ``--diagnostics-json PATH`` both produce output."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        sidecar = tmp_path / "out.json"

        fake_report = ConversionReport(
            hydro_count=2,
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
                [
                    "convert",
                    "newave",
                    str(src),
                    str(dst),
                    "--json",
                    "--diagnostics-json",
                    str(sidecar),
                ],
                monkeypatch,
            )

        assert code == 0
        # stdout is the JSON verdict (carries command/status).
        verdict = json.loads(stdout)
        assert verdict["command"] == "convert newave"
        assert verdict["status"] == "ok"
        # The sidecar was also written (its payload has no command/status keys).
        assert sidecar.exists()
        sidecar_doc = json.loads(sidecar.read_text(encoding="utf-8"))
        assert sidecar_doc["summary"]["hydros"] == 2
        assert "command" not in sidecar_doc

    def test_convert_without_json_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``--json`` the human ``✓ Converted ...`` summary still prints."""
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
        assert "Converted" in stdout
        # The default path emits no JSON document.
        with pytest.raises(json.JSONDecodeError):
            json.loads(stdout)


class TestCompareDatasetWiring:
    """ticket-008: compare handlers sourced from the canonical dataset.

    Patch the heavy readers (``NewaveCase``, alignment, ``compare_*``) so the real
    dataset build + ``write_artifacts`` + dataset-driven printers run without the source
    model/Cobre I/O.
    """

    def _invoke_main(
        self,
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> tuple[int, str, str]:
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

    @staticmethod
    def _results() -> object:
        from cobre_bridge.comparators.results import ResultComparison

        return [
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
        ]

    @staticmethod
    def _bounds(*, all_match: bool) -> object:
        from cobre_bridge.comparators.bounds import BoundComparison

        rows = [
            BoundComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="storage_max",
                newave_value=29000.0,
                cobre_value=29000.0,
                diff=0.0,
                match=True,
            ),
        ]
        if not all_match:
            rows.append(
                BoundComparison(
                    entity_type="thermal",
                    entity_name="ANGRA",
                    newave_code=30,
                    cobre_id=1,
                    stage=0,
                    variable="generation_max",
                    newave_value=1350.0,
                    cobre_value=1300.0,
                    diff=50.0,
                    match=False,
                )
            )
        return rows

    def _patch_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.results import PercentileData

        monkeypatch.setattr(
            "cobre_bridge.case.NewaveCase.from_directory",
            classmethod(lambda cls, _dir: MagicMock(id_map=MagicMock())),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.alignment.build_entity_alignment",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "cobre_bridge.cli._load_lines_json",
            lambda _dir: [],
        )
        # ``compare_results`` now returns the canonical ``ComparisonDataset``;
        # build it from the same fixture rows so the CLI path is exercised.
        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            lambda **k: build_results_dataset(self._results(), PercentileData(), 1e-2),
        )

    @staticmethod
    def _make_cobre_dir_with_bounds(tmp_path: Path, name: str) -> Path:
        """Create a Cobre output dir containing the required bounds.parquet stub.

        The bounds handler validates ``training/dictionaries/bounds.parquet``
        exists before running; the file content is unused here because
        ``compare_bounds`` is patched.
        """
        import pyarrow.parquet as pq

        cobre_dir = tmp_path / name
        bounds_path = cobre_dir / "training" / "dictionaries" / "bounds.parquet"
        bounds_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"x": pa.array([0], pa.int32())}), bounds_path)
        return cobre_dir

    def _patch_bounds(
        self, monkeypatch: pytest.MonkeyPatch, *, all_match: bool
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.case.NewaveCase.from_directory",
            classmethod(lambda cls, _dir: MagicMock(id_map=MagicMock())),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.alignment.build_entity_alignment",
            lambda *a, **k: MagicMock(),
        )
        monkeypatch.setattr(
            "cobre_bridge.cli._load_lines_json",
            lambda _dir: [],
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.bounds.compare_bounds",
            lambda **k: self._bounds(all_match=all_match),
        )

    def test_compare_results_emits_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "results", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        manifest_path = cobre_dir / "comparison_artifacts" / "comparison.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["command"] == "compare results"
        assert "Artifacts written to" in stdout

    def test_compare_results_artifacts_without_output_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`compare results` WITHOUT ``-o`` still writes artifacts and exits 0.

        Guards the curated-serialization contract: the bulky render-only
        metadata (``results`` and the drained ``PercentileData`` frames) IS
        stored in ``dataset.metadata`` but is listed in
        ``RENDER_ONLY_METADATA_KEYS``, so the SAME dataset's ``to_dir`` (invoked
        by ``write_artifacts``) skips it and does not choke on a non-JSON-native
        value.
        """
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "results", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = cobre_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        # to_dir round-trip artifacts prove metadata serialized cleanly.
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "metadata.json").exists()
        assert "Artifacts written to" in stdout

    def test_compare_bounds_emits_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_bounds(monkeypatch, all_match=True)
        cobre_dir = self._make_cobre_dir_with_bounds(tmp_path, "cobre")

        code, _, _ = self._invoke_main(
            ["compare", "bounds", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        manifest_path = cobre_dir / "comparison_artifacts" / "comparison.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["command"] == "compare bounds"

    def test_compare_bounds_exit_code_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Full match -> exit 0.
        self._patch_bounds(monkeypatch, all_match=True)
        cobre_dir = self._make_cobre_dir_with_bounds(tmp_path, "cobre_match")
        code_match, _, _ = self._invoke_main(
            ["compare", "bounds", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )
        assert code_match == 0

        # Any mismatch -> exit 1.
        self._patch_bounds(monkeypatch, all_match=False)
        cobre_dir2 = self._make_cobre_dir_with_bounds(tmp_path, "cobre_mismatch")
        code_mismatch, _, _ = self._invoke_main(
            ["compare", "bounds", str(tmp_path / "nw"), str(cobre_dir2)],
            monkeypatch,
        )
        assert code_mismatch == 1

    def test_load_compare_context_missing_source_model_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ticket-015: bounds path gains FileNotFoundError -> exit 1 hardening.

        A missing the source model case directory now exits 1 with a clean stderr
        message (via the shared ``_load_compare_context`` helper) instead of surfacing
        an uncaught traceback. Results already had this; bounds gains it in this
        refactor.
        """

        def _raise_missing(cls: object, _dir: Path) -> object:
            raise FileNotFoundError("caso.dat not found")

        monkeypatch.setattr(
            "cobre_bridge.case.NewaveCase.from_directory",
            classmethod(_raise_missing),
        )
        cobre_dir = self._make_cobre_dir_with_bounds(tmp_path, "cobre")

        code, _, stderr = self._invoke_main(
            ["compare", "bounds", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 1
        assert "Error: caso.dat not found" in stderr

    def test_compare_results_html_tabs_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.comparators.html_report import COMPARISON_TABS

        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            [
                "compare",
                "results",
                str(tmp_path / "nw"),
                str(cobre_dir),
                "--format",
                "html",
            ],
            monkeypatch,
        )

        assert code == 0
        report_path = cobre_dir / "comparison_artifacts" / "report.html"
        assert report_path.exists()
        html = report_path.read_text(encoding="utf-8")
        for tab_id, _label in COMPARISON_TABS:
            assert tab_id in html

    def test_compare_results_default_writes_parquet_no_html(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``--format``: default writes queryable artifacts, no HTML."""
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "results", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = cobre_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "summary.json").exists()
        assert not (artifacts_dir / "report.html").exists()

    def test_compare_results_format_console_only_no_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format console``: only the manifest is written (opt out of data)."""
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            [
                "compare",
                "results",
                str(tmp_path / "nw"),
                str(cobre_dir),
                "--format",
                "console",
            ],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = cobre_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        assert not (artifacts_dir / "comparison.parquet").exists()

    def test_compare_results_format_parquet_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format parquet,json``: queryable artifacts present, exit 0."""
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            [
                "compare",
                "results",
                str(tmp_path / "nw"),
                str(cobre_dir),
                "--format",
                "parquet,json",
            ],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = cobre_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "summary.json").exists()

    def test_compare_bounds_html_warns_and_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format html`` on bounds: stderr warns, no HTML, exit 0 on match."""
        self._patch_bounds(monkeypatch, all_match=True)
        cobre_dir = self._make_cobre_dir_with_bounds(tmp_path, "cobre")

        code, _, stderr = self._invoke_main(
            [
                "compare",
                "bounds",
                str(tmp_path / "nw"),
                str(cobre_dir),
                "--format",
                "html",
            ],
            monkeypatch,
        )

        assert code == 0
        assert "not supported for 'compare bounds'" in stderr
        assert not (cobre_dir / "comparison_artifacts" / "report.html").exists()

    def test_compare_unknown_format_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--format bogus``: stderr names the bad token, exit 2."""
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, stderr = self._invoke_main(
            [
                "compare",
                "results",
                str(tmp_path / "nw"),
                str(cobre_dir),
                "--format",
                "bogus",
            ],
            monkeypatch,
        )

        assert code == 2
        assert "bogus" in stderr

    def test_compare_results_help_omits_output_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``compare results --help`` lists --format/--out-dir, not --output/-o."""
        code, stdout, _ = self._invoke_main(
            ["compare", "results", "--help"],
            monkeypatch,
        )

        assert code == 0
        assert "--format" in stdout
        assert "--out-dir" in stdout
        assert "--output" not in stdout
        assert "-o " not in stdout

    def test_compare_results_artifact_oserror_does_not_change_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        def _boom(*_a: object, **_k: object) -> object:
            raise OSError("disk full")

        monkeypatch.setattr("cobre_bridge.comparators.export.write_artifacts", _boom)

        code, _, stderr = self._invoke_main(
            ["compare", "results", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        assert "failed to write artifacts" in stderr


class TestParseFormats:
    """ticket-016: ``_parse_formats`` token parsing and validation."""

    def test_parse_formats_default(self) -> None:
        from cobre_bridge.cli import _parse_formats

        assert _parse_formats(None) == {"console", "parquet", "json"}

    def test_parse_formats_comma_and_repeat(self) -> None:
        from cobre_bridge.cli import _parse_formats

        assert _parse_formats(["csv,json", "parquet"]) == {
            "csv",
            "json",
            "parquet",
        }

    def test_parse_formats_all_expands(self) -> None:
        from cobre_bridge.cli import _parse_formats

        assert _parse_formats(["all"]) == {
            "console",
            "html",
            "csv",
            "parquet",
            "json",
        }

    def test_parse_formats_unknown_raises(self) -> None:
        from cobre_bridge.cli import _parse_formats

        with pytest.raises(ValueError, match="bogus"):
            _parse_formats(["bogus"])


# ---------------------------------------------------------------------------
# Pipeline integration tests for inflow_history.parquet
# ---------------------------------------------------------------------------


class TestPipelineInflowHistory:
    """Tests verifying that convert_newave_case always writes inflow_history.parquet."""

    def test_inflow_history_always_written(self, tmp_path: Path) -> None:
        """inflow_history.parquet is always written (from vazoes.dat)."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        history_path = dst / "scenarios" / "inflow_history.parquet"
        assert history_path.exists(), "inflow_history.parquet was not written"


class TestConversionWarningCapture:
    """``convert_newave_case`` surfaces converter warnings via ConversionReport."""

    def test_captures_and_dedupes_package_warnings(self, tmp_path: Path) -> None:
        import logging

        from cobre_bridge import pipeline
        from cobre_bridge.pipeline import ConversionReport, convert_newave_case

        log = logging.getLogger("cobre_bridge.converters.fake")

        def fake_impl(
            src: Path, dst: Path, on_phase: object = None
        ) -> ConversionReport:
            log.warning("vazpast.dat unreadable; using empty tendency")
            log.warning("vazpast.dat unreadable; using empty tendency")  # duplicate
            log.info("informational, not a degradation")  # below WARNING → ignored
            log.warning("REE.DAT has no entries")
            return ConversionReport(hydro_count=3)

        with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
            report = convert_newave_case(tmp_path, tmp_path)

        assert report.hydro_count == 3
        assert report.warnings == [
            "vazpast.dat unreadable; using empty tendency",
            "REE.DAT has no entries",
        ]

    def test_no_warnings_when_clean(self, tmp_path: Path) -> None:
        from cobre_bridge import pipeline
        from cobre_bridge.pipeline import ConversionReport, convert_newave_case

        with patch.object(
            pipeline,
            "_convert_newave_case_impl",
            return_value=ConversionReport(hydro_count=1),
        ):
            report = convert_newave_case(tmp_path, tmp_path)

        assert report.warnings == []

    def test_collector_detached_even_on_exception(self, tmp_path: Path) -> None:
        import logging

        from cobre_bridge import pipeline
        from cobre_bridge.pipeline import convert_newave_case

        pkg_logger = logging.getLogger("cobre_bridge")
        handlers_before = list(pkg_logger.handlers)
        with patch.object(
            pipeline,
            "_convert_newave_case_impl",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                convert_newave_case(tmp_path, tmp_path)

        # The capture handler must be removed in the finally block, leaving the
        # package logger's handler list exactly as it was.
        assert pkg_logger.handlers == handlers_before

    def test_partial_outputs_cleared_on_failure(self, tmp_path: Path) -> None:
        """A failure partway through the write phase must not leave a partial,
        valid-looking case behind: the known pipeline outputs are removed so a
        plain (no --force) re-run is not refused as non-empty."""
        from cobre_bridge import pipeline
        from cobre_bridge.pipeline import convert_newave_case

        dst = tmp_path / "dst"

        def fake_impl(src: Path, d: Path, on_phase: object = None) -> object:
            # Simulate a write phase that got partway: a top-level JSON and a
            # system/ subdir were written before the failure.
            (d / "system").mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text("{}")
            (d / "system" / "hydros.json").write_text("{}")
            raise RuntimeError("disk full mid-write")

        with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
            with pytest.raises(RuntimeError, match="disk full"):
                convert_newave_case(tmp_path, dst)

        # No pipeline outputs survive — dst holds no half-written case.
        assert not (dst / "config.json").exists()
        assert not (dst / "system").exists()
        # dst itself may remain but must be empty, so a no-force re-run proceeds.
        assert not any(dst.iterdir())


def test_constraint_id_allocator_advances_contiguously() -> None:
    """The allocator hands out contiguous, non-overlapping ID ranges."""
    from cobre_bridge.pipeline import _ConstraintIdAllocator

    alloc = _ConstraintIdAllocator()
    assert alloc.next_id == 0
    alloc.advance(3)  # VminOP used IDs 0,1,2
    assert alloc.next_id == 3  # electric starts here
    alloc.advance(0)  # electric produced nothing
    assert alloc.next_id == 3  # AGRINT still starts at 3
    alloc.advance(2)
    assert alloc.next_id == 5


class TestConversionDiagnosticsRendering:
    """The convert subcommand renders structured diagnostics (names/stages/values)."""

    def _invoke_main(
        self, argv: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> tuple[int, str, str]:
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

    def _report_with_gtmin(self):
        from cobre_bridge.diagnostics import Diagnostic, DiagnosticTable, Severity
        from cobre_bridge.pipeline import ConversionReport

        diag = Diagnostic(
            code="thermal-gtmin-above-capacity",
            severity=Severity.WARNING,
            category="Thermal bounds",
            title="GTMIN exceeds available capacity (1 plant(s))",
            summary="one plant affected",
            table=DiagnosticTable(
                columns=["Plant", "Code", "Stages", "GTMIN MW", "Cap MW"],
                rows=[["ANGRA 2", 13, "2-3", 481.3, 423.4]],
            ),
            remediation="Check EXPT FCMAX/GTMIN and MANUTT for these plants.",
        )
        return ConversionReport(hydro_count=1, diagnostics=[diag])

    def test_structured_diagnostic_shows_name_stages_values(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=self._report_with_gtmin(),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)], monkeypatch
            )
        assert code == 0
        assert "1 hydros" in stdout  # summary on stdout
        # The named pain: plant name, the stages, and the values are all surfaced.
        assert "ANGRA 2" in stderr
        assert "2-3" in stderr
        assert "481.3" in stderr
        assert "423.4" in stderr

    def test_quiet_suppresses_summary_but_keeps_warnings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=self._report_with_gtmin(),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--quiet"], monkeypatch
            )
        assert code == 0
        assert "Converted" not in stdout  # summary suppressed
        assert "ANGRA 2" in stderr  # warnings still shown

    def test_diagnostics_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import json

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"
        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=self._report_with_gtmin(),
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "newave",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                ],
                monkeypatch,
            )
        assert code == 0
        assert json_path.exists()
        payload = json.loads(json_path.read_text())
        assert payload["summary"]["hydros"] == 1
        codes = [d["code"] for d in payload["diagnostics"]]
        assert "thermal-gtmin-above-capacity" in codes


class TestTyperApp:
    """Typer-app behaviours via the idiomatic CliRunner invocation path."""

    @staticmethod
    def _invoke(argv: list[str]):
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        return CliRunner().invoke(app, argv)

    def test_version_exit_zero(self) -> None:
        result = self._invoke(["--version"])
        assert result.exit_code == 0
        assert "cobre-bridge" in result.stdout

    def test_help_lists_subcommands(self) -> None:
        result = self._invoke(["--help"])
        assert result.exit_code == 0
        assert "convert" in result.stdout
        assert "compare" in result.stdout
        assert "dashboard" in result.stdout

    def test_help_exposes_shell_completion(self) -> None:
        result = self._invoke(["--help"])
        assert "install-completion" in result.stdout

    def test_convert_missing_subcommand_exits_two(self) -> None:
        assert self._invoke(["convert"]).exit_code == 2

    def test_compare_missing_subcommand_exits_two(self) -> None:
        assert self._invoke(["compare"]).exit_code == 2

    def test_convert_newave_happy_path(self, tmp_path: Path) -> None:
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "out"
        report = ConversionReport(hydro_count=7, stage_count=12)
        with patch("cobre_bridge.pipeline.convert_newave_case", return_value=report):
            result = self._invoke(["convert", "newave", str(src), str(dst)])
        assert result.exit_code == 0
        assert "7 hydros" in result.stdout


class TestCheckCommand:
    """ticket-007: the ``check newave`` preflight command (exit 0/1/2 + --json)."""

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

    @staticmethod
    def _result(verdict: object) -> object:
        """Build a small ``PreflightResult`` with the given verdict.

        Warnings carry one ``WARNING`` diagnostic so the JSON/headline paths see a
        realistic payload; the verdict itself is taken verbatim by the handler and
        renderer (never recomputed from the checks/diagnostics here).
        """
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.preflight import (
            CheckItem,
            PreflightResult,
            PreflightVerdict,
        )

        if verdict is PreflightVerdict.WILL_NOT_CONVERT:
            return PreflightResult(
                verdict=PreflightVerdict.WILL_NOT_CONVERT,
                diagnostics=[
                    Diagnostic(
                        code="source-file-error",
                        severity=Severity.ERROR,
                        category="Preflight",
                        title="Required input missing",
                        summary="caso.dat not found",
                    )
                ],
                checks=[
                    CheckItem(
                        label="File discovery (caso.dat → arquivos.dat)",
                        passed=False,
                        detail="caso.dat not found",
                    )
                ],
            )
        if verdict is PreflightVerdict.WARNINGS:
            return PreflightResult(
                verdict=PreflightVerdict.WARNINGS,
                diagnostics=[
                    Diagnostic(
                        code="optional-file-absent",
                        severity=Severity.WARNING,
                        category="Preflight",
                        title="Optional input absent",
                        summary="Optional input 'modif' was not found.",
                    )
                ],
                checks=[
                    CheckItem(label="Required files present", passed=True),
                    CheckItem(
                        label="Optional: modif",
                        passed=True,
                        detail="absent (will use defaults)",
                    ),
                ],
            )
        return PreflightResult(
            verdict=PreflightVerdict.OK,
            diagnostics=[],
            checks=[CheckItem(label="Required files present", passed=True)],
        )

    # -- Unit tests ---------------------------------------------------------

    def test_check_json_document_shape(self) -> None:
        """``_check_json_document`` builds the verdict schema with fixed order."""
        from cobre_bridge.cli import _check_json_document
        from cobre_bridge.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        doc = _check_json_document(result)  # type: ignore[arg-type]

        assert list(doc.keys()) == ["command", "status", "checks", "diagnostics"]
        assert doc["command"] == "check newave"
        assert doc["status"] == "will-not-convert"
        assert doc["checks"] == [
            {
                "label": "File discovery (caso.dat → arquivos.dat)",
                "passed": False,
                "detail": "caso.dat not found",
            }
        ]
        assert doc["diagnostics"][0]["severity"] == "error"  # type: ignore[index]

    def test_verdict_to_exit_code_mapping(self) -> None:
        """The 0/1/2 mapping is exactly OK/WARNINGS/WILL_NOT_CONVERT (2 = severe)."""
        from cobre_bridge.cli import _VERDICT_EXIT_CODE
        from cobre_bridge.preflight import PreflightVerdict

        assert _VERDICT_EXIT_CODE == {
            PreflightVerdict.OK: 0,
            PreflightVerdict.WARNINGS: 1,
            PreflightVerdict.WILL_NOT_CONVERT: 2,
        }

    # -- Integration tests (in-process, patched run_preflight) --------------

    def test_check_ok_exits_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.OK)
        with patch("cobre_bridge.preflight.run_preflight", return_value=result):
            code, stdout, _ = self._invoke_main(
                ["check", "newave", str(tmp_path / "case")],
                monkeypatch,
            )

        assert code == 0
        assert "✓ Ready to convert" in stdout

    def test_check_warnings_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WARNINGS)
        with patch("cobre_bridge.preflight.run_preflight", return_value=result):
            code, stdout, _ = self._invoke_main(
                ["check", "newave", str(tmp_path / "case")],
                monkeypatch,
            )

        assert code == 1
        assert "Ready with warnings" in stdout

    def test_check_will_not_convert_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from cobre_bridge.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        with patch("cobre_bridge.preflight.run_preflight", return_value=result):
            code, stdout, _ = self._invoke_main(
                ["check", "newave", str(tmp_path / "case")],
                monkeypatch,
            )

        assert code == 2
        assert "✖ Will not convert" in stdout

    def test_check_json_emits_stdout_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` on a WILL_NOT_CONVERT result emits JSON to stdout; exit 2."""
        from cobre_bridge.preflight import PreflightVerdict

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        with patch("cobre_bridge.preflight.run_preflight", return_value=result):
            code, stdout, stderr = self._invoke_main(
                ["check", "newave", str(tmp_path / "case"), "--json"],
                monkeypatch,
            )

        assert code == 2
        doc = json.loads(stdout)
        assert doc["command"] == "check newave"
        assert doc["status"] == "will-not-convert"
        assert doc["checks"][0]["passed"] is False
        # No Rich checklist leaked onto either stream.
        assert "✖ Will not convert" not in stdout
        assert stderr == ""

    def test_check_writes_no_files_under_src(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``check`` must not write anything under the source directory."""
        from cobre_bridge.preflight import PreflightVerdict

        src = _make_fake_newave_dir(tmp_path)
        before = sorted(p.name for p in src.iterdir())

        result = self._result(PreflightVerdict.OK)
        with patch("cobre_bridge.preflight.run_preflight", return_value=result):
            code, _, _ = self._invoke_main(
                ["check", "newave", str(src)],
                monkeypatch,
            )

        assert code == 0
        assert sorted(p.name for p in src.iterdir()) == before

    # -- E2E test (real discovery failure via subprocess) -------------------

    def test_check_missing_caso_subprocess_exits_2(self, tmp_path: Path) -> None:
        """A real discovery failure (no caso.dat) exits 2 with the ✖ headline."""
        result = _run_cli_subprocess("check", "newave", str(tmp_path / "nonexistent"))

        assert result.returncode == 2
        combined = result.stdout + result.stderr
        assert "✖ Will not convert" in combined


def test_convert_newave_case_threads_on_phase(tmp_path: Path) -> None:
    """``convert_newave_case`` forwards its ``on_phase`` callback to the impl."""
    from cobre_bridge import pipeline
    from cobre_bridge.pipeline import ConversionReport, convert_newave_case

    received: list[str] = []

    def fake_impl(src: Path, dst: Path, on_phase: object = None) -> ConversionReport:
        if on_phase is not None:
            on_phase("Discovering files")  # type: ignore[operator]
        return ConversionReport()

    with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
        convert_newave_case(tmp_path, tmp_path, on_phase=received.append)

    assert received == ["Discovering files"]
