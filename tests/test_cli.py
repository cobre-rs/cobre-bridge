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
import logging
import subprocess
import sys
import webbrowser
from pathlib import Path
from types import SimpleNamespace
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
        "block_id": pa.array([], type=pa.int32()),
    }
)
_FAKE_NCS: dict = {"non_controllable_sources": []}
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


def _make_fake_decomp_dir(tmp_path: Path) -> Path:
    """Create a minimal discoverable deck directory: ``caso.dat`` naming ``rv0``,
    plus ``dadger``/``vazoes``/``hidr`` stubs. ``discover_decomp_files`` only
    stats/globs these, never parses them."""
    decomp_dir = tmp_path / "decomp_case"
    decomp_dir.mkdir()
    (decomp_dir / "caso.dat").write_text("rv0\n")
    (decomp_dir / "dadger.rv0").write_text("stub")
    (decomp_dir / "vazoes.rv0").write_text("stub")
    (decomp_dir / "hidr.dat").write_text("stub")
    return decomp_dir


def _make_fake_decomp_dir_with_cuts(tmp_path: Path) -> Path:
    """Like :func:`_make_fake_decomp_dir`, plus stub ``cortesh``/``cortes``
    files so the real (unmocked) ``discover_decomp_files`` resolves both via
    its glob fallback — the boundary-FCF gating path only globs/stats these,
    never parses their contents."""
    decomp_dir = _make_fake_decomp_dir(tmp_path)
    (decomp_dir / "cortesh.rv0").write_text("stub")
    (decomp_dir / "cortes.rv0").write_text("stub")
    return decomp_dir


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
    "training": {
        "selection": {"method": "sampled", "forward_passes": 5},
        "stopping_rules": [],
    },
    "simulation": {
        "enabled": True,
        "selection": {"method": "sampled", "num_scenarios": 200},
    },
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
            "cobre_bridge.pipeline.inflow_windows.convert_recent_observation_windows",
            return_value=[],
        ),
        patch(
            "cobre_bridge.pipeline.inflow_windows.convert_inflow_history_windows",
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
    import contextlib

    from cobre_bridge.pipeline import convert_newave_case

    fake_id_map = MagicMock()
    with contextlib.ExitStack() as stack:
        for p in _all_converter_patches(fake_id_map):
            stack.enter_context(p)
        return convert_newave_case(src, dst)


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
            dst / "scenarios" / "non_controllable_factors.json",
            dst / "scenarios" / "non_controllable_stats.parquet",
        ]
        for f in expected:
            assert f.exists(), f"Expected output file not found: {f}"

    def test_exchange_factors_json_is_not_written(self, tmp_path: Path) -> None:
        """The per-block exchange factors are folded into line_bounds.parquet
        (cobre decision 10); the pipeline must not write the deleted file."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        assert not (dst / "constraints" / "exchange_factors.json").exists()

    def test_load_factors_and_ncs_factors_still_byte_identical(
        self, tmp_path: Path
    ) -> None:
        """``load_factors.json`` and ``non_controllable_factors.json`` are
        untouched by the exchange-factors migration (epic 02 draws the line at
        authored-vs-sampled data; deleting these by analogy would be wrong)."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        _run_with_all_mocks(src, dst)

        load_factors_path = dst / "scenarios" / "load_factors.json"
        with load_factors_path.open(encoding="utf-8") as f:
            assert json.load(f) == _FAKE_LOAD_FACTORS

        ncs_factors_path = dst / "scenarios" / "non_controllable_factors.json"
        with ncs_factors_path.open(encoding="utf-8") as f:
            assert json.load(f) == _FAKE_NCS_FACTORS

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

        with (
            patch(
                "cobre_bridge.pipeline.NewaveCase.from_directory",
                side_effect=FileNotFoundError(
                    f"Required NEWAVE file not found in {src}: hidr.dat"
                ),
            ),
            pytest.raises(FileNotFoundError) as exc_info,
        ):
            convert_newave_case(src, dst)
        assert "hidr.dat" in str(exc_info.value)

    def test_dry_run_does_not_call_write_table(self, tmp_path: Path) -> None:
        """``dry_run=True`` writes nothing yet records the would-write paths."""
        import contextlib

        from cobre_bridge.pipeline import ConversionReport, convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        fake_id_map = MagicMock()
        with (
            patch("cobre_bridge.pipeline.pq.write_table") as write_table,
            contextlib.ExitStack() as stack,
        ):
            for p in _all_converter_patches(fake_id_map):
                stack.enter_context(p)
            report = convert_newave_case(src, dst, dry_run=True)

        assert isinstance(report, ConversionReport)
        # No Parquet table is written and no destination directory is created.
        assert write_table.call_count == 0
        assert not dst.exists() or list(dst.iterdir()) == []
        # The would-write listing is still populated (covers JSON and Parquet).
        assert report.would_write_paths
        assert str(dst / "config.json") in report.would_write_paths
        assert str(dst / "system" / "hydros.json") in report.would_write_paths


class TestEmissionCheckWiring:
    """The post-emission self-checks (ticket-016, epic-04) run inside the real
    pipeline body, before the writes, and their findings flip the convert
    verdict through ``_convert_status`` — not merely by inspecting the
    diagnostic."""

    def test_no_hydro_bounds_reports_rule_43_not_applicable(
        self, tmp_path: Path
    ) -> None:
        """The fully-mocked fixture never builds a hydro_bounds table, so rule
        43 is explicitly "not applicable" (INFO), not silently absent, and the
        convert verdict stays clean."""
        from cobre_bridge.cli import _convert_status
        from cobre_bridge.diagnostics import Severity

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"
        report = _run_with_all_mocks(src, dst)

        not_applicable = [
            d
            for d in report.diagnostics  # type: ignore[union-attr]
            if d.code == "hydro-bounds-raising-not-applicable"
        ]
        assert len(not_applicable) == 1
        assert not_applicable[0].severity is Severity.INFO
        assert _convert_status(report.diagnostics, success="ok") == "ok"  # type: ignore[union-attr]

    def test_rule_43_violation_flips_convert_status_to_error(
        self, tmp_path: Path
    ) -> None:
        """A hydro_bounds row raising the plant's declared max_turbined_m3s is
        caught, and feeding the resulting diagnostics through
        ``_convert_status`` (the same function cli.py uses to derive the
        convert verdict) yields "error", never just an inspected diagnostic."""
        import contextlib

        from cobre_bridge.cli import _convert_status
        from cobre_bridge.pipeline import convert_newave_case

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "cobre_case"

        violating_hydros = {
            "$schema": "http://example",
            "hydros": [
                {
                    "id": 0,
                    "name": "Test Plant",
                    "generation": {
                        "max_turbined_m3s": 100.0,
                        "max_generation_mw": 50.0,
                    },
                    "unit_groups": [
                        {
                            "id": 0,
                            "max_turbined_m3s": 100.0,
                            "max_generation_mw": 50.0,
                        }
                    ],
                }
            ],
        }
        violating_storage_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([2], type=pa.int32()),
                "max_turbined_m3s": pa.array([150.0], type=pa.float64()),
            }
        )

        fake_id_map = MagicMock()
        with contextlib.ExitStack() as stack:
            for p in _all_converter_patches(fake_id_map):
                stack.enter_context(p)
            stack.enter_context(
                patch(
                    "cobre_bridge.pipeline.hydro_conv.convert_hydros",
                    return_value=violating_hydros,
                )
            )
            stack.enter_context(
                patch(
                    "cobre_bridge.pipeline.hydro_conv.convert_storage_bounds",
                    return_value=violating_storage_bounds,
                )
            )
            report = convert_newave_case(src, dst)

        errors = [
            d
            for d in report.diagnostics
            if d.code == "hydro-bounds-raises-declared-capacity"
        ]
        assert len(errors) == 1
        assert "Test Plant" not in errors[0].summary  # rule 43 names IDs, like cobre

        # Load-bearing: the verdict comes from _convert_status, not a bare
        # inspection of the diagnostic list.
        assert _convert_status(report.diagnostics, success="ok") == "error"
        assert _convert_status([], success="ok") == "ok"


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
        assert doc["schema_version"] == 1
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


class TestPartitionValidationWarnings:
    """``_partition_validation_warnings`` — the pure whitelist filter (ticket-007)."""

    def test_partition_whitelists_interop_warning(self) -> None:
        """The interop message is whitelisted; an unrelated one still renders."""
        from cobre_bridge.cli import _partition_validation_warnings

        interop = (
            "inflow lags are disabled on all study stages. This is a valid "
            "configuration for external-solver interoperability; otherwise "
            "it is likely a misconfiguration."
        )
        unrelated = "some unrelated warning"

        rendered, whitelisted = _partition_validation_warnings(
            [interop, unrelated], ("external-solver interoperability",)
        )

        assert rendered == [unrelated]
        assert whitelisted == [interop]

    def test_partition_empty_whitelist_is_identity(self) -> None:
        """An empty whitelist — what ``convert newave`` passes — changes nothing."""
        from cobre_bridge.cli import _partition_validation_warnings

        warnings: list[object] = ["w1", "w2", {"message": "w3"}]

        rendered, whitelisted = _partition_validation_warnings(warnings, ())

        assert rendered == warnings
        assert whitelisted == []


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

    def test_convert_verdict_shape(self) -> None:
        """The convert ``summary``+``status`` helpers feed the unified envelope."""
        from cobre_bridge.cli import _convert_status, _convert_verdict_summary
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.pipeline import ConversionReport
        from cobre_bridge.verdict import build_verdict

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

        summary = _convert_verdict_summary(report)
        status = _convert_status([info], success="ok")
        doc = build_verdict("convert newave", status, summary, [info])

        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
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

    def test_convert_verdict_error_status_on_error_diagnostic(self) -> None:
        """Any ERROR-severity diagnostic flips ``status`` to ``"error"``."""
        from cobre_bridge.cli import _convert_status, _convert_verdict_summary
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.verdict import build_verdict

        error = Diagnostic(
            code="boom-code",
            severity=Severity.ERROR,
            category="Conversion failure",
            title="Conversion failed",
            summary="boom",
        )

        summary = _convert_verdict_summary(None)
        status = _convert_status([error], success="ok")
        doc = build_verdict("convert newave", status, summary, [error])

        assert doc["schema_version"] == 1
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
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "convert newave"
        assert doc["status"] == "ok"
        assert doc["summary"]["hydros"] == 10
        # --validate not requested → no validation sub-object.
        assert "validation" not in doc["summary"]
        # No Rich human summary leaked onto stdout.
        assert "Converted" not in stdout
        # The only thing on stderr is the always-on conversion-manifest note,
        # which is intentionally routed to stderr to keep stdout byte-clean.
        assert "Conversion manifest written" in stderr
        assert "Converted" not in stderr

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
        assert doc["schema_version"] == 1
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["status"] == "error"
        assert doc["summary"]["hydros"] == 0
        # Failure path never requested validation.
        assert "validation" not in doc["summary"]
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
        # stdout is the JSON verdict (carries schema_version/command/status).
        verdict = json.loads(stdout)
        assert verdict["schema_version"] == 1
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

    def test_conversion_manifest_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful conversion leaves a valid provenance manifest in dst."""
        from cobre_bridge.conversion_manifest import ConversionManifest
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
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest_path = dst / "conversion_manifest.json"
        assert manifest_path.exists()
        manifest = ConversionManifest.from_json(manifest_path)
        assert manifest.entity_counts == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
        }
        assert manifest.command == "convert newave"
        # The fake source dir's stub files were discovered and hashed.
        assert manifest.input_files

    def test_conversion_manifest_records_min_cobre_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manifest's ``min_cobre_version`` tracks the CLI constant, pinned.

        ticket-005: a manifest written after the bump must record the real
        floor (``"0.14.2"``), not a stale value — the manifest is provenance,
        and a wrong floor there is false provenance. Pinning the literal (not
        just equality with the constant) catches an accidental revert of the
        constant itself.
        """
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.conversion_manifest import ConversionManifest
        from cobre_bridge.pipeline import ConversionReport

        assert MIN_COBRE_VERSION == "0.14.2"

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest = ConversionManifest.from_json(dst / "conversion_manifest.json")
        assert manifest.min_cobre_version == "0.14.2"
        assert manifest.min_cobre_version == MIN_COBRE_VERSION

    def test_manifest_not_in_json_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json`` stdout is the convert verdict; the manifest is a side file."""
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
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        # Only the deterministic envelope keys — no manifest timestamp leaks in.
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert "timestamp" not in doc
        # The manifest still exists separately on disk.
        assert (dst / "conversion_manifest.json").exists()

    def test_convert_json_validate_folds_into_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--json --validate`` folds the validation outcome under ``summary``.

        The machine outcome lands in ``summary["validation"]`` (status stays
        diagnostics-derived), stdout is pure JSON, and a failed validation still
        exits 2.
        """
        import types

        from cobre_bridge.cli import MIN_COBRE_VERSION
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

        # Fix the installed-version gate at exactly MIN_COBRE_VERSION so this
        # test exercises the injected fake ``cobre.io.validate`` below
        # regardless of whichever cobre-python happens to be installed in the
        # dev/CI venv (which may itself now be older than MIN_COBRE_VERSION).
        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version",
            lambda: MIN_COBRE_VERSION,
        )
        # Inject a fake ``cobre.io`` whose ``validate`` reports a failure with one
        # warning and two errors; the real ``cobre`` package is not installed.
        cobre_pkg = types.ModuleType("cobre")
        cobre_io = types.ModuleType("cobre.io")
        cobre_io.validate = lambda _dst: {  # type: ignore[attr-defined]
            "valid": False,
            "warnings": ["w"],
            "errors": ["e1", "e2"],
        }
        cobre_pkg.io = cobre_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cobre", cobre_pkg)
        monkeypatch.setitem(sys.modules, "cobre.io", cobre_io)

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        # Validation failure flips the exit code, never the verdict status.
        assert code == 2
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        # status is still diagnostics-derived ("ok"), NOT flipped by validation.
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": True,
            "valid": False,
            "warnings": 1,
            "errors": 2,
        }
        # No validation text leaked onto stdout; the human messages stay on stderr.
        assert "Validation" not in stdout
        assert "Validation failed." in stderr

    def test_convert_json_validate_raising_still_emits_one_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raising ``cobre.io.validate`` under ``--json`` still emits one verdict.

        The conversion succeeded, so stdout must still carry exactly one JSON
        object (the --json contract) even though validation crashed and the
        command exits 2; ``summary.validation`` records the failure.
        """
        import types

        from cobre_bridge.cli import MIN_COBRE_VERSION
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

        def _raise(_dst: str) -> dict[str, object]:
            raise RuntimeError("validator blew up")

        # Fix the installed-version gate at exactly MIN_COBRE_VERSION so this
        # test reaches (and exercises) the injected raising ``validate`` below
        # regardless of whichever cobre-python happens to be installed in the
        # dev/CI venv (which may itself now be older than MIN_COBRE_VERSION).
        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version",
            lambda: MIN_COBRE_VERSION,
        )
        cobre_pkg = types.ModuleType("cobre")
        cobre_io = types.ModuleType("cobre.io")
        cobre_io.validate = _raise  # type: ignore[attr-defined]
        cobre_pkg.io = cobre_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cobre", cobre_pkg)
        monkeypatch.setitem(sys.modules, "cobre.io", cobre_io)

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        assert code == 2
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": False,
            "valid": None,
            "warnings": 0,
            "errors": 1,
        }
        assert "validator blew up" in stderr
        assert "validator blew up" not in stdout

    def test_convert_json_without_validate_has_no_validation_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``--validate`` the convert ``summary`` carries no validation key."""
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
                ["convert", "newave", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        assert "validation" not in doc["summary"]

    def test_manifest_records_diagnostics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The manifest carries the conversion's diagnostics + their summary."""
        from cobre_bridge.conversion_manifest import ConversionManifest
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        warning = Diagnostic(
            code="some-warning",
            severity=Severity.WARNING,
            category="Conversion",
            title="A warning",
            summary="heads up",
        )
        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
            diagnostics=[warning],
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest = ConversionManifest.from_json(dst / "conversion_manifest.json")
        assert manifest.diagnostics_summary == {"warning": 1}
        assert len(manifest.diagnostics) == 1

    def test_clear_dst_removes_manifest(self, tmp_path: Path) -> None:
        """``_clear_dst_contents`` removes a stale top-level manifest on --force."""
        from cobre_bridge.pipeline import _clear_dst_contents

        dst = tmp_path / "dst"
        dst.mkdir()
        manifest_path = dst / "conversion_manifest.json"
        manifest_path.write_text("{}", encoding="utf-8")

        _clear_dst_contents(dst)

        assert not manifest_path.exists()

    def test_manifest_write_failure_does_not_fail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A manifest write OSError is warned-and-swallowed; exit stays 0."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
        )

        def _raise(self: object, path: Path) -> None:
            raise OSError("disk full")

        with (
            patch(
                "cobre_bridge.pipeline.convert_newave_case",
                return_value=fake_report,
            ),
            patch(
                "cobre_bridge.conversion_manifest.ConversionManifest.to_json",
                _raise,
            ),
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "failed to write conversion manifest" in stderr

    def test_dry_run_writes_nothing_to_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` into an empty dst writes nothing and exits 0."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        # No destination directory is created and nothing is written.
        assert not dst.exists() or list(dst.iterdir()) == []
        assert "Dry run — no files written" in stdout
        assert "config.json" in stdout

    def test_dry_run_converter_error_exits_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A converter error under ``--dry-run`` exits 1 and names the failure."""
        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            side_effect=ValueError("boom"),
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 1
        assert "Conversion failed" in stderr
        assert "boom" in stderr
        assert "boom" not in stdout

    def test_dry_run_json_document_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --json`` emits a sorted, dst-relative would-write document."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        # Deliberately unsorted absolute paths under dst.
        fake_report = ConversionReport(
            hydro_count=10,
            thermal_count=5,
            bus_count=4,
            line_count=3,
            stage_count=60,
            would_write_paths=[
                str(dst / "system" / "hydros.json"),
                str(dst / "config.json"),
            ],
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run", "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        # would_write moves UNDER summary; only the five envelope keys at the top.
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert "would_write" not in doc
        assert doc["schema_version"] == 1
        assert doc["command"] == "convert newave"
        assert doc["status"] == "dry-run"
        assert doc["summary"] == {
            "hydros": 10,
            "thermals": 5,
            "buses": 4,
            "lines": 3,
            "stages": 60,
            # dst-relative, forward-slash, sorted.
            "would_write": ["config.json", "system/hydros.json"],
        }
        assert "timestamp" not in doc

    def test_dry_run_nonempty_dst_without_force_exits_one_and_preserves_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-empty dst without ``--force`` is refused even under ``--dry-run``."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        dst.mkdir()
        existing = dst / "existing.txt"
        existing.write_text("keep me", encoding="utf-8")

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ) as convert:
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 1
        assert "Use --force" in stderr
        # The destination is untouched: the guard fired before the pipeline ran.
        assert convert.call_count == 0
        assert existing.read_text(encoding="utf-8") == "keep me"
        assert list(dst.iterdir()) == [existing]

    def test_dry_run_with_validate_emits_note_and_skips_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --validate`` skips validation and notes it on stderr."""
        import builtins

        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=12,
            would_write_paths=[str(dst / "config.json")],
        )

        real_import = builtins.__import__

        def _guard_import(name: str, *args: object, **kwargs: object) -> object:
            # No cobre validation import may be attempted under --dry-run.
            if name.startswith("cobre.io"):
                raise AssertionError("validation must not import cobre under --dry-run")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        with (
            patch(
                "cobre_bridge.pipeline.convert_newave_case",
                return_value=fake_report,
            ),
            patch.object(builtins, "__import__", _guard_import),
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--dry-run", "--validate"],
                monkeypatch,
            )

        assert code == 0
        assert "--validate is ignored under --dry-run" in stderr
        # No validation output of any kind reached stderr.
        assert "Validation" not in stderr

    @staticmethod
    def _inject_cobre_io(
        monkeypatch: pytest.MonkeyPatch, validate: MagicMock
    ) -> MagicMock:
        """Inject a fake ``cobre`` / ``cobre.io`` whose ``validate`` is *validate*.

        Mirrors the established pattern in the other ``--validate`` tests so the
        real (unreleased-schema) cobre package is never touched. Returns the
        ``validate`` mock so the test can assert on its call count.
        """
        import types

        cobre_pkg = types.ModuleType("cobre")
        cobre_io = types.ModuleType("cobre.io")
        cobre_io.validate = validate  # type: ignore[attr-defined]
        cobre_pkg.io = cobre_io  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "cobre", cobre_pkg)
        monkeypatch.setitem(sys.modules, "cobre.io", cobre_io)
        return validate

    def test_validate_skipped_when_cobre_python_old(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A too-old cobre-python skips ``validate`` and exits 0."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version", lambda: "0.9.0"
        )
        validate = self._inject_cobre_io(monkeypatch, MagicMock())

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()

    def test_validate_skip_json_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skip records the explicit reason under ``summary.validation``."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version", lambda: "0.9.0"
        )
        validate = self._inject_cobre_io(monkeypatch, MagicMock())

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()
        doc = json.loads(stdout)
        # status stays diagnostics-derived; the skip lands only under summary.
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": False,
            "valid": None,
            "warnings": 0,
            "errors": 0,
            "skipped_reason": "cobre-python-too-old",
        }
        assert doc["summary"]["validation"]["ran"] is False
        assert doc["summary"]["validation"]["skipped_reason"] == "cobre-python-too-old"

    def test_validate_skip_human_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The skip note lands on stderr with the version + skip phrasing."""
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version", lambda: "0.9.0"
        )
        validate = self._inject_cobre_io(monkeypatch, MagicMock())

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()
        assert "skipping cobre-python validation" in stderr
        assert MIN_COBRE_VERSION in stderr

    def test_validate_skipped_for_installed_0_12_below_new_min(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ticket-005: a 0.12 install is now too old and gets an honest skip.

        Before the ``MIN_COBRE_VERSION`` bump to ``"0.13.0"``, an installed
        ``0.12.0`` satisfied the gate and ``validate`` ran — against output
        that (post hydro unit-groups / windowed inflow_history) a 0.12 cobre
        cannot actually read, producing a false rejection. At the new floor
        the gate must skip instead: no ``validate`` call, exit 0, and a
        warning naming *both* the installed and required versions so the
        skip is diagnosable rather than a silent no-op.
        """
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.pipeline import ConversionReport

        assert MIN_COBRE_VERSION == "0.14.2"

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version", lambda: "0.12.0"
        )
        validate = self._inject_cobre_io(monkeypatch, MagicMock())

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--json", "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_not_called()
        # Does not fail, and does not silently pretend it validated: the
        # warning names both versions, and the JSON summary is explicit that
        # validation did not run (not that it ran and passed).
        assert "skipping cobre-python validation" in stderr
        assert "0.12.0" in stderr
        assert "0.14.2" in stderr
        assert MIN_COBRE_VERSION in stderr
        doc = json.loads(stdout)
        assert doc["status"] == "ok"
        assert doc["summary"]["validation"] == {
            "ran": False,
            "valid": None,
            "warnings": 0,
            "errors": 0,
            "skipped_reason": "cobre-python-too-old",
        }

    def test_validate_runs_when_cobre_python_supports_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The case validates when the installed cobre-python knows the schema."""
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version",
            lambda: MIN_COBRE_VERSION,
        )
        validate = self._inject_cobre_io(
            monkeypatch,
            MagicMock(return_value={"valid": True, "warnings": [], "errors": []}),
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()

    def test_validate_falls_through_when_cobre_python_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No installed cobre-python → the version gate defers to the generic skip."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version", lambda: None
        )
        # Force ``import cobre.io`` to raise regardless of whether the real
        # cobre-python is installed in this environment (a ``None`` entry in
        # sys.modules makes the import fail), so the generic "not installed"
        # branch runs rather than the version gate.
        monkeypatch.setitem(sys.modules, "cobre", None)
        monkeypatch.setitem(sys.modules, "cobre.io", None)

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        assert "cobre package not installed" in stderr

    def test_validate_runs_when_cobre_python_metadata_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No version metadata + an importable cobre.io → validation runs."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version", lambda: None
        )
        validate = self._inject_cobre_io(
            monkeypatch,
            MagicMock(return_value={"valid": True, "warnings": [], "errors": []}),
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()

    def test_convert_newave_validate_unchanged_by_helper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``convert newave --validate`` still renders the interop warning.

        ``convert newave`` passes an empty whitelist to the shared
        ``_run_cobre_validation`` helper, so a warning DECOMP whitelists (the
        cobre external-solver-interop note) must still render here — the
        byte-identical-behavior guarantee the ticket-007 helper extraction
        must not break. Contrast with
        ``test_convert_decomp_validate_whitelists_interop`` below.
        """
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version",
            lambda: MIN_COBRE_VERSION,
        )
        interop_warning = (
            "inflow lags are disabled on all study stages. This is a valid "
            "configuration for external-solver interoperability; otherwise "
            "it is likely a misconfiguration."
        )
        validate = self._inject_cobre_io(
            monkeypatch,
            MagicMock(
                return_value={
                    "valid": True,
                    "warnings": [interop_warning],
                    "errors": [],
                }
            ),
        )

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()
        assert "external-solver interoperability" in stderr
        assert "Validation warning:" in stderr

    def test_convert_decomp_success_shows_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful ``convert decomp`` prints the ``✓ Converted ...`` summary."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "✓ Converted 3 hydros" in stdout

    def test_convert_decomp_warning_diagnostic_renders_rollup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``WARNING`` diagnostic on the report renders the notes roll-up + title."""
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        warning = Diagnostic(
            code="decomp-some-warning",
            severity=Severity.WARNING,
            category="Conversion",
            title="A DECOMP warning",
            summary="heads up",
        )
        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[warning],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "Conversion notes:" in stderr
        assert "A DECOMP warning" in stderr

    def test_convert_decomp_json_success_emits_verdict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``convert decomp --json`` emits the unified verdict envelope."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["command"] == "convert decomp"
        assert doc["status"] == "ok"
        assert set(doc["summary"]) >= {"hydros", "thermals", "buses", "lines", "stages"}

    def test_convert_decomp_json_failure_emits_error_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conversion failure under ``--json`` emits ``status == "error"``; exit 1."""
        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=ValueError("bad"),
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert doc["status"] == "error"

    def test_convert_decomp_diagnostics_json_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--diagnostics-json`` writes the report-shaped sidecar (summary + findings)."""
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[
                Diagnostic(
                    code="decomp-some-warning",
                    severity=Severity.WARNING,
                    category="Conversion",
                    title="A DECOMP warning",
                    summary="heads up",
                )
            ],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
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
        assert set(payload) == {"summary", "diagnostics"}
        assert payload["summary"]["hydros"] == 3
        assert [d["code"] for d in payload["diagnostics"]] == ["decomp-some-warning"]

    def test_convert_decomp_diagnostics_json_coexists_with_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The sidecar and ``--json`` are independent: stdout verdict AND file written."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                    "--json",
                ],
                monkeypatch,
            )

        assert code == 0
        # stdout is exactly the unified verdict object …
        doc = json.loads(stdout)
        assert doc["command"] == "convert decomp"
        # … and the sidecar was written all the same.
        assert json_path.exists()
        assert set(json.loads(json_path.read_text())) == {"summary", "diagnostics"}

    def test_convert_decomp_no_diagnostics_json_writes_no_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent the flag, no sidecar file is created."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert not json_path.exists()

    def test_convert_decomp_validate_whitelists_interop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``convert decomp --validate`` whitelists the interop inflow-lags warning.

        The P3 lag-blind stage shape (``inflow_lags=false`` on every stage,
        locked in ``test_decomp_temporal.py``) trips cobre's non-fatal
        external-solver-interop warning on purpose; DECOMP's whitelist keeps
        it off stderr while newave's does not (see the sibling
        ``test_convert_newave_validate_unchanged_by_helper`` above).
        """
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version",
            lambda: MIN_COBRE_VERSION,
        )
        interop_warning = (
            "inflow lags are disabled on all study stages. This is a valid "
            "configuration for external-solver interoperability; otherwise "
            "it is likely a misconfiguration."
        )
        validate = self._inject_cobre_io(
            monkeypatch,
            MagicMock(
                return_value={
                    "valid": True,
                    "warnings": [interop_warning],
                    "errors": [],
                }
            ),
        )

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )
        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 0
        validate.assert_called_once()
        assert "external-solver interoperability" not in stderr
        assert "Validation warning:" not in stderr

    def test_convert_decomp_dry_run_writes_nothing_to_dst(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` into an empty dst writes nothing and exits 0."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        # No destination directory is created and nothing is written.
        assert not dst.exists() or list(dst.iterdir()) == []
        assert "Dry run — no files written" in stdout
        assert "config.json" in stdout

    def test_convert_decomp_dry_run_json_document_is_deterministic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --json`` emits a sorted, dst-relative would-write document."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        # Deliberately unsorted absolute paths under dst.
        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[
                str(dst / "system" / "hydros.json"),
                str(dst / "config.json"),
            ],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run", "--json"],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        # would_write moves UNDER summary; only the five envelope keys at the top.
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert "would_write" not in doc
        assert doc["command"] == "convert decomp"
        assert doc["status"] == "dry-run"
        assert doc["summary"] == {
            "hydros": 3,
            "thermals": 2,
            "buses": 1,
            "lines": 0,
            "stages": 4,
            # dst-relative, forward-slash, sorted.
            "would_write": ["config.json", "system/hydros.json"],
        }

    def test_convert_decomp_dry_run_json_failure_has_would_write_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conversion failure under ``--dry-run --json`` still carries the
        ``would_write`` key (parity with ``convert newave``): a consumer that
        reads ``summary["would_write"]`` on any dry-run verdict must not
        KeyError just because the conversion failed.
        """
        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=ValueError("bad"),
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run", "--json"],
                monkeypatch,
            )

        assert code == 1
        doc = json.loads(stdout)
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["command"] == "convert decomp"
        assert doc["summary"]["would_write"] == []

    def test_convert_decomp_dry_run_with_validate_emits_note_and_skips_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --validate`` skips validation and notes it on stderr."""
        import builtins

        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        real_import = builtins.__import__

        def _guard_import(name: str, *args: object, **kwargs: object) -> object:
            # No cobre validation import may be attempted under --dry-run.
            if name.startswith("cobre.io"):
                raise AssertionError("validation must not import cobre under --dry-run")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch.object(builtins, "__import__", _guard_import),
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run", "--validate"],
                monkeypatch,
            )

        assert code == 0
        assert "--validate is ignored under --dry-run" in stderr
        # No validation output of any kind reached stderr.
        assert "Validation" not in stderr

    def test_convert_decomp_dry_run_writes_no_diagnostics_json_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run --diagnostics-json`` writes no sidecar: the dry-run branch
        returns before the sidecar block runs."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--dry-run",
                    "--diagnostics-json",
                    str(json_path),
                ],
                monkeypatch,
            )

        assert code == 0
        assert not json_path.exists()

    def test_convert_decomp_manifest_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful ``convert decomp`` leaves a valid provenance manifest."""
        from cobre_bridge.conversion_manifest import ConversionManifest
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=3, thermal_count=2, bus_count=1, line_count=0, stage_count=4
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest_path = dst / "conversion_manifest.json"
        assert manifest_path.exists()
        manifest = ConversionManifest.from_json(manifest_path)
        assert manifest.command == "convert decomp"
        assert manifest.entity_counts == {
            "hydros": 3,
            "thermals": 2,
            "buses": 1,
            "lines": 0,
            "stages": 4,
        }
        # The stub deck's dadger/vazoes/hidr files were discovered and hashed.
        assert manifest.input_files

    def test_convert_decomp_manifest_records_min_cobre_version(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The DECOMP manifest's ``min_cobre_version`` tracks the CLI constant."""
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.conversion_manifest import ConversionManifest
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=12
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        manifest = ConversionManifest.from_json(dst / "conversion_manifest.json")
        assert manifest.min_cobre_version == MIN_COBRE_VERSION

    def test_convert_decomp_dry_run_writes_no_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` writes no provenance manifest: the dry-run branch
        returns before the manifest-write block runs."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        assert not (dst / "conversion_manifest.json").exists()

    # ------------------------------------------------------------------
    # Boundary FCF (default on; --no-fcf opts out; in-process, no --cobre-bin)
    # ------------------------------------------------------------------

    def test_convert_decomp_no_fcf_skips_import(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--no-fcf`` skips the importer even when the deck declares cut
        files: no ``boundary/`` directory, and the deck is never re-discovered
        for the FCF gate."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.import_boundary_fcf") as mock_import,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--no-fcf"],
                monkeypatch,
            )

        assert code == 0
        mock_import.assert_not_called()
        assert not (dst / "boundary").exists()

    def test_convert_decomp_boundary_fcf_happy_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With cut files present (the default), the importer runs with
        ``cost_scale_factor=1.0``, exits 0, surfaces the C8 run recipe on
        stderr, and the ``--json`` verdict carries ``summary["boundary_fcf"]``."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "cobre_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"
            ) as mock_capability,
            patch(
                "cobre_bridge.decomp.fcf.import_boundary_fcf",
                return_value=dst / "boundary",
            ) as mock_import,
        ):
            code, stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--json"],
                monkeypatch,
            )

        assert code == 0
        mock_capability.assert_called_once()
        mock_import.assert_called_once()
        assert mock_import.call_args.kwargs["cost_scale_factor"] == 1.0
        assert "cobre_bin" not in mock_import.call_args.kwargs
        assert mock_import.call_args.args[0] == dst
        # C8 recipe surfaced on stderr regardless of --json.
        assert f"cobre run {dst}" in stderr
        assert f"--output={dst}" in stderr
        doc = json.loads(stdout)
        assert doc["summary"]["boundary_fcf"] == {
            "imported": True,
            "path": "boundary",
            "run_constraint": f"--output={dst}",
        }

    def test_convert_decomp_missing_cortes_skips_fcf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deck that declares no cortes files converts with exit 0 and an
        INFO note (not an error): the importer never runs and no ``boundary/``
        directory is written."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)  # no cortesh/cortes files
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.import_boundary_fcf") as mock_import,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 0
        assert "no cortes/cortesh files" in stderr
        mock_import.assert_not_called()
        assert not (dst / "boundary").exists()

    def test_convert_decomp_boundary_fcf_capability_guard_failure_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing capability probe exits 1 with the install remediation."""
        from cobre_bridge.decomp.fcf.capability import REMEDIATION
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch(
                "cobre_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability",
                side_effect=RuntimeError(REMEDIATION),
            ),
            patch("cobre_bridge.decomp.fcf.import_boundary_fcf") as mock_import,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst)],
                monkeypatch,
            )

        assert code == 1
        assert "cobre-python" in stderr
        mock_import.assert_not_called()

    def test_convert_decomp_boundary_fcf_runs_before_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The boundary-FCF import runs BEFORE ``--validate``; a validation
        failure still exits 2 once the import already succeeded."""
        from cobre_bridge.cli import MIN_COBRE_VERSION
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        call_order: list[str] = []

        def _fake_import(*args: object, **kwargs: object) -> Path:
            call_order.append("import_boundary_fcf")
            return dst / "boundary"

        def _fake_validate(*args: object, **kwargs: object) -> dict[str, object]:
            call_order.append("validate")
            return {"valid": False, "warnings": [], "errors": ["boom"]}

        monkeypatch.setattr(
            "cobre_bridge.cli._installed_cobre_python_version",
            lambda: MIN_COBRE_VERSION,
        )
        self._inject_cobre_io(monkeypatch, MagicMock(side_effect=_fake_validate))

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"),
            patch(
                "cobre_bridge.decomp.fcf.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--validate"],
                monkeypatch,
            )

        assert code == 2
        assert call_order == ["import_boundary_fcf", "validate"]

    def test_convert_decomp_fcf_skipped_under_dry_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--dry-run`` skips the boundary FCF import and notes it on stderr,
        even when the deck declares cut files."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1,
            thermal_count=1,
            bus_count=1,
            line_count=0,
            stage_count=4,
            would_write_paths=[str(dst / "config.json")],
        )

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.import_boundary_fcf") as mock_import,
        ):
            code, _stdout, stderr = self._invoke_main(
                ["convert", "decomp", str(src), str(dst), "--dry-run"],
                monkeypatch,
            )

        assert code == 0
        assert "boundary FCF import is skipped under --dry-run" in stderr
        mock_import.assert_not_called()
        assert not dst.exists() or list(dst.iterdir()) == []

    def test_convert_decomp_boundary_fcf_importer_diagnostics_reach_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ticket-010 (D): the importer runs inside a ``dx.collect()`` sink,
        so a ``Diagnostic`` it emits — here the GNL anticipated-ring
        deviation — reaches the ``--json`` verdict's ``diagnostics`` array.
        This test fails against the pre-sink CLI (no ``dx.collect()``
        wrapping ``import_boundary_fcf``), proving the deferred Epic-03 gap
        is closed."""
        from cobre_bridge import diagnostics as dx
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        def _fake_import(*args: object, **kwargs: object) -> Path:
            dx.emit(
                dx.Diagnostic(
                    code="boundary-fcf-gnl-anticipated-deviation",
                    severity=dx.Severity.INFO,
                    category="Boundary FCF",
                    title="GNL anticipated ring carries a per-patamar sum",
                    summary="synthetic deviation diagnostic for the sink test",
                )
            )
            return dst / "boundary"

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"),
            patch(
                "cobre_bridge.decomp.fcf.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--json",
                ],
                monkeypatch,
            )

        assert code == 0
        doc = json.loads(stdout)
        codes = {d["code"] for d in doc["diagnostics"]}
        assert "boundary-fcf-gnl-anticipated-deviation" in codes
        # ``status`` stays "ok": the importer's diagnostic is INFO-severity,
        # so surfacing it never flips the verdict outcome.
        assert doc["status"] == "ok"

    def test_convert_decomp_boundary_fcf_importer_diagnostics_render_panel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ticket-010 (D): the same emitting mock, without ``--json`` — the
        diagnostic's title renders on stderr (the Rich panel), and the
        existing happy-path C8-recipe assertions still hold (no
        double-render, no exit-code change)."""
        from cobre_bridge import diagnostics as dx
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        def _fake_import(*args: object, **kwargs: object) -> Path:
            dx.emit(
                dx.Diagnostic(
                    code="boundary-fcf-gnl-anticipated-deviation",
                    severity=dx.Severity.INFO,
                    category="Boundary FCF",
                    title="GNL anticipated ring carries a per-patamar sum",
                    summary="synthetic deviation diagnostic for the sink test",
                )
            )
            return dst / "boundary"

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"),
            patch(
                "cobre_bridge.decomp.fcf.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, _stdout, stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                ],
                monkeypatch,
            )

        assert code == 0
        assert "GNL anticipated ring carries a per-patamar sum" in stderr
        # The C8 run-recipe note still surfaces (happy-path behaviour intact).
        assert f"cobre run {dst}" in stderr
        assert f"--output={dst}" in stderr

    def test_convert_decomp_boundary_fcf_importer_diagnostics_reach_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Epic-04 boundary-review finding #2: the same emitting mock as
        ``..._reach_json``, but without ``--json`` and with
        ``--diagnostics-json`` — the importer's ``Diagnostic`` (captured by
        the ``dx.collect()`` sink) must reach the sidecar file too, not just
        the ``--json`` stdout verdict. This test fails against the pre-fix
        CLI, where the sidecar is written BEFORE the boundary-FCF block runs
        and therefore only ever contains ``report.diagnostics``."""
        from cobre_bridge import diagnostics as dx
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir_with_cuts(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=1, thermal_count=1, bus_count=1, line_count=0, stage_count=4
        )

        def _fake_import(*args: object, **kwargs: object) -> Path:
            dx.emit(
                dx.Diagnostic(
                    code="boundary-fcf-gnl-anticipated-deviation",
                    severity=dx.Severity.INFO,
                    category="Boundary FCF",
                    title="GNL anticipated ring carries a per-patamar sum",
                    summary="synthetic deviation diagnostic for the sink test",
                )
            )
            return dst / "boundary"

        with (
            patch(
                "cobre_bridge.decomp.pipeline.convert_decomp_case",
                return_value=fake_report,
            ),
            patch("cobre_bridge.decomp.fcf.capability.ensure_boundary_fcf_capability"),
            patch(
                "cobre_bridge.decomp.fcf.import_boundary_fcf",
                side_effect=_fake_import,
            ),
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
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
        codes = {d["code"] for d in payload["diagnostics"]}
        assert "boundary-fcf-gnl-anticipated-deviation" in codes

    def test_convert_decomp_diagnostics_json_unchanged_without_fcf(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guard: a ``convert decomp --diagnostics-json --no-fcf`` sidecar is
        exactly ``report.diagnostics`` — deferring the sidecar write past the
        (here, skipped) boundary-FCF block must not regress the contract."""
        from cobre_bridge.diagnostics import Diagnostic, Severity
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_decomp_dir(tmp_path)
        dst = tmp_path / "dst"
        json_path = tmp_path / "diag.json"

        fake_report = ConversionReport(
            hydro_count=3,
            thermal_count=2,
            bus_count=1,
            line_count=0,
            stage_count=4,
            diagnostics=[
                Diagnostic(
                    code="decomp-some-warning",
                    severity=Severity.WARNING,
                    category="Conversion",
                    title="A DECOMP warning",
                    summary="heads up",
                )
            ],
        )

        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            return_value=fake_report,
        ):
            code, _stdout, _stderr = self._invoke_main(
                [
                    "convert",
                    "decomp",
                    str(src),
                    str(dst),
                    "--diagnostics-json",
                    str(json_path),
                    "--no-fcf",
                ],
                monkeypatch,
            )

        assert code == 0
        payload = json.loads(json_path.read_text())
        assert set(payload) == {"summary", "diagnostics"}
        assert [d["code"] for d in payload["diagnostics"]] == ["decomp-some-warning"]


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

    def test_compare_results_emits_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_results(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        manifest_path = cobre_dir / "comparison_artifacts" / "comparison.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["command"] == "compare newave"
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
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        artifacts_dir = cobre_dir / "comparison_artifacts"
        assert (artifacts_dir / "comparison.json").exists()
        # to_dir round-trip artifacts prove metadata serialized cleanly.
        assert (artifacts_dir / "comparison.parquet").exists()
        assert (artifacts_dir / "metadata.json").exists()
        assert "Artifacts written to" in stdout

    def test_load_compare_context_missing_source_model_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing source-model case directory exits 1 with a clean stderr message.

        The shared ``_load_compare_context`` helper turns a ``FileNotFoundError``
        from the case reader into a clean exit-1 message instead of surfacing an
        uncaught traceback.
        """

        def _raise_missing(cls: object, _dir: Path) -> object:
            raise FileNotFoundError("caso.dat not found")

        monkeypatch.setattr(
            "cobre_bridge.case.NewaveCase.from_directory",
            classmethod(_raise_missing),
        )
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
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
                "newave",
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
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
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
                "newave",
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
                "newave",
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
                "newave",
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
        self, monkeypatch: pytest.MonkeyPatch, dumb_terminal: None
    ) -> None:
        """``compare results --help`` lists --format/--out-dir, not --output/-o."""
        code, stdout, _ = self._invoke_main(
            ["compare", "newave", "--help"],
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
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        assert "failed to write artifacts" in stderr


class TestCompareJson:
    """ticket-020: ``compare bounds``/``compare results`` ``--json`` verdict.

    Patches the heavy readers (``NewaveCase``, alignment, ``compare_*``) so the
    real dataset build + verdict derivation run without source-model/Cobre I/O,
    then asserts the unified envelope on stdout, the exit-code contract, and the
    no-Rich-on-stdout property.
    """

    #: Rich glyphs / box-drawing that must NEVER appear on a ``--json`` stdout.
    _RICH_GLYPHS = ("✓", "⚠", "─", "━", "│", "┃")

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
    def _results(*, within_tol: bool) -> object:
        """Build result rows that are fully within tol (matched) or divergent.

        With ``within_tol`` the cobre value equals the newave value, so the
        derived ``within_tol_rate`` is ``1.0`` and ``all_within_tol`` is True.
        Otherwise the cobre value diverges past the default ``1e-2`` tolerance.
        """
        from cobre_bridge.comparators.results import ResultComparison

        cobre_value = 100.0 if within_tol else 110.0
        abs_diff = 0.0 if within_tol else 10.0
        rel_diff = 0.0 if within_tol else 0.1
        return [
            ResultComparison(
                entity_type="hydro",
                entity_name="ITAIPU",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=cobre_value,
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            ),
        ]

    def _patch_common(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Patch the shared context loaders (case, alignment, lines.json)."""
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

    def _patch_results(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        within_tol: bool,
    ) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.results import PercentileData

        self._patch_common(monkeypatch)
        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            lambda **k: build_results_dataset(
                self._results(within_tol=within_tol), PercentileData(), 1e-2
            ),
        )

    def _assert_no_rich_stdout(self, stdout: str) -> None:
        for glyph in self._RICH_GLYPHS:
            assert glyph not in stdout
        assert "Artifacts written to" not in stdout
        assert "HTML report written to" not in stdout

    def test_compare_results_json_divergent_status_mismatch_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC3: divergent results → ``status="mismatch"`` but always exit 0."""
        self._patch_results(monkeypatch, within_tol=False)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir), "--json"],
            monkeypatch,
        )

        assert code == 0
        doc = json.loads(stdout)
        assert doc["command"] == "compare newave"
        assert doc["status"] == "mismatch"
        assert doc["summary"]["all_within_tol"] is False
        assert doc["summary"]["within_tol"] < doc["summary"]["total"]
        self._assert_no_rich_stdout(stdout)

    def test_compare_results_json_within_tol_status_ok_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """AC4: fully-within-tol results → ``status="ok"`` and exit 0."""
        self._patch_results(monkeypatch, within_tol=True)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir), "--json"],
            monkeypatch,
        )

        assert code == 0
        doc = json.loads(stdout)
        assert doc["status"] == "ok"
        assert doc["summary"]["all_within_tol"] is True
        self._assert_no_rich_stdout(stdout)

    def test_compare_results_json_cobre_read_error_exit_2_no_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A CobreReadError exits 2 with stderr only — no stdout JSON."""
        from cobre_bridge.comparators.cobre_readers import CobreReadError

        self._patch_common(monkeypatch)

        def _raise(**_k: object) -> object:
            raise CobreReadError("bad parquet")

        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            _raise,
        )
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir), "--json"],
            monkeypatch,
        )

        assert code == 2
        assert stdout == ""
        assert "ERROR:" in stderr
        assert "bad parquet" in stderr

    def test_compare_results_partition_missing_exit_2_no_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FINDING-1 regression: CobrePartitionMissingError (raised by
        read_cobre_bus_aggregates against a pre-0.13 / 0.13-incomplete
        output dir lacking simulation/hydro_bus_generation/) extends
        BridgeError -- a hierarchy disjoint from CobreReadError
        (RuntimeError). The compare newave CLI handler must catch it too,
        rendering a clean ERROR line + exit 2, not an unhandled traceback.
        This drives the REAL reader against a genuinely 0.13-incomplete
        output dir (simulation/hydros/ present, simulation/
        hydro_bus_generation/ absent), so the exception message is
        production-generated, not hand-typed."""
        from cobre_bridge.comparators.cobre_readers import read_cobre_bus_aggregates

        self._patch_common(monkeypatch)

        cobre_dir = tmp_path / "cobre"
        (cobre_dir / "simulation" / "hydros").mkdir(parents=True)

        def _raise(**_k: object) -> object:
            return read_cobre_bus_aggregates(cobre_dir)

        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results",
            _raise,
        )

        code, stdout, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir), "--json"],
            monkeypatch,
        )

        assert code == 2
        assert stdout == ""
        assert "ERROR:" in stderr
        assert str(cobre_dir / "simulation" / "hydro_bus_generation") in stderr
        assert "0.13.0" in stderr


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
        from cobre_bridge import pipeline
        from cobre_bridge.pipeline import ConversionReport, convert_newave_case

        log = logging.getLogger("cobre_bridge.converters.fake")

        def fake_impl(
            src: Path,
            dst: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
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
        from cobre_bridge import pipeline
        from cobre_bridge.pipeline import convert_newave_case

        pkg_logger = logging.getLogger("cobre_bridge")
        handlers_before = list(pkg_logger.handlers)
        with (
            patch.object(
                pipeline,
                "_convert_newave_case_impl",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
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

        def fake_impl(
            src: Path,
            d: Path,
            on_phase: object = None,
            *,
            dry_run: bool = False,
        ) -> object:
            # Simulate a write phase that got partway: a top-level JSON and a
            # system/ subdir were written before the failure.
            (d / "system").mkdir(parents=True, exist_ok=True)
            (d / "config.json").write_text("{}")
            (d / "system" / "hydros.json").write_text("{}")
            raise RuntimeError("disk full mid-write")

        with (
            patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl),
            pytest.raises(RuntimeError, match="disk full"),
        ):
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

    def test_help_lists_subcommands(self, dumb_terminal: None) -> None:
        result = self._invoke(["--help"])
        assert result.exit_code == 0
        assert "convert" in result.stdout
        assert "compare" in result.stdout
        assert "dashboard" in result.stdout

    def test_help_exposes_shell_completion(self, dumb_terminal: None) -> None:
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

    def test_check_verdict_shape(self) -> None:
        """The check ``summary`` helper feeds the unified envelope (checks nested)."""
        from cobre_bridge.preflight import PreflightVerdict
        from cobre_bridge.verdict import build_verdict, check_summary

        result = self._result(PreflightVerdict.WILL_NOT_CONVERT)
        summary = check_summary(
            [
                {"label": c.label, "passed": c.passed, "detail": c.detail}
                for c in result.checks
            ]
        )
        doc = build_verdict(
            "check newave", result.verdict.value, summary, result.diagnostics
        )

        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "check newave"
        assert doc["status"] == "will-not-convert"
        # The checklist moves UNDER summary.
        assert "checks" not in doc
        assert doc["summary"]["checks"] == [  # type: ignore[index]
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
        assert list(doc.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert doc["schema_version"] == 1
        assert doc["command"] == "check newave"
        assert doc["status"] == "will-not-convert"
        # The checklist lives under summary now, not at the top level.
        assert "checks" not in doc
        assert doc["summary"]["checks"][0]["passed"] is False
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

    def fake_impl(
        src: Path,
        dst: Path,
        on_phase: object = None,
        *,
        dry_run: bool = False,
    ) -> ConversionReport:
        if on_phase is not None:
            on_phase("Discovering files")  # type: ignore[operator]
        return ConversionReport()

    with patch.object(pipeline, "_convert_newave_case_impl", side_effect=fake_impl):
        convert_newave_case(tmp_path, tmp_path, on_phase=received.append)

    assert received == ["Discovering files"]


class TestResolveCompareSettings:
    """ticket-014: unit tests for the ``_resolve_compare_settings`` helper.

    Drives the helper directly with a crafted ``SimpleNamespace`` and a patched
    ``load_config`` returning a hand-built ``BridgeConfig``, so the precedence
    logic (flag/env > config > built-in default) is exercised without any I/O.
    """

    @staticmethod
    def _resolve(
        config: object,
        *,
        tolerance: float | None,
        fmt: list[str] | None,
        out_dir: Path | None,
    ) -> SimpleNamespace:
        """Run ``_resolve_compare_settings`` with ``load_config`` patched."""
        import cobre_bridge.cli as cli

        args = SimpleNamespace(tolerance=tolerance, format=fmt, out_dir=out_dir)
        with patch.object(cli, "load_config", return_value=config):
            cli._resolve_compare_settings(args)
        return args

    def test_flag_or_env_value_wins_over_config(self) -> None:
        """A non-None ``args`` value (flag or env) is kept, ignoring config."""
        from cobre_bridge.config_resolution import BridgeConfig

        config = BridgeConfig(
            results_tolerance=5e-4,
            formats=("csv",),
            out_dir=Path("from_config"),
        )
        args = self._resolve(
            config,
            tolerance=9e-4,
            fmt=["json"],
            out_dir=Path("from_flag"),
        )

        assert args.tolerance == 9e-4
        assert args.format == ["json"]
        assert args.out_dir == Path("from_flag")

    def test_config_fills_when_flag_and_env_are_none(self) -> None:
        """With ``args`` all None, the config-file values fill every field."""
        from cobre_bridge.config_resolution import BridgeConfig

        config = BridgeConfig(
            results_tolerance=5e-4,
            formats=("json", "csv"),
            out_dir=Path("art"),
        )
        args = self._resolve(config, tolerance=None, fmt=None, out_dir=None)

        assert args.tolerance == 5e-4
        assert args.format == ["json", "csv"]
        assert args.out_dir == Path("art")

    def test_builtin_default_when_config_empty(self) -> None:
        """An empty config falls through to the built-in tolerance default."""
        from cobre_bridge.config_resolution import (
            RESULTS_TOLERANCE_DEFAULT,
            BridgeConfig,
        )

        args = self._resolve(BridgeConfig(), tolerance=None, fmt=None, out_dir=None)
        assert args.tolerance == RESULTS_TOLERANCE_DEFAULT
        # Format/out-dir stay None so the downstream defaults (``_parse_formats``
        # / derived out-dir) still apply.
        assert args.format is None
        assert args.out_dir is None

    def test_config_warning_emitted_to_stderr(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Each config-load warning is surfaced on stderr, never on stdout."""
        from cobre_bridge.config_resolution import BridgeConfig

        config = BridgeConfig(
            warnings=("Ignoring malformed config file cobre-bridge.toml: bad",),
        )
        self._resolve(config, tolerance=None, fmt=None, out_dir=None)

        captured = capsys.readouterr()
        assert "Ignoring malformed config file" in captured.err
        assert "Ignoring malformed config file" not in captured.out


class TestCompareConfigEnvPrecedence:
    """ticket-014: integration precedence tests for config/env wiring.

    Runs ``compare results`` in-process via ``cli.main``, with the heavy readers
    stubbed and ``compare_results`` / ``write_artifacts`` patched with recording
    wrappers so the resolved ``tolerance`` / ``out_dir`` / ``formats`` reaching
    them can be asserted. The cwd is an isolated tmp subdir and
    ``XDG_CONFIG_HOME`` / ``HOME`` point at empty tmp subdirs, so only the test's
    own ``cobre-bridge.toml`` (when written) is seen.
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
    def _isolate_config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Chdir to an empty workdir and point config discovery at empty dirs.

        Returns the workdir (where a ``cobre-bridge.toml`` may be written). The
        XDG/HOME fallbacks are redirected to empty tmp subdirs so no real user
        config leaks into the resolution.
        """
        workdir = tmp_path / "work"
        workdir.mkdir()
        xdg = tmp_path / "xdg"
        xdg.mkdir()
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.chdir(workdir)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
        monkeypatch.setenv("HOME", str(home))
        return workdir

    def _stub_readers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the heavy the-source-model / Cobre readers shared by both commands."""
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

    def _capture_results_tolerance(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> dict[str, object]:
        """Patch ``compare_results`` with a recorder; return the captured kwargs."""
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.results import PercentileData

        captured: dict[str, object] = {}

        def _recorder(**kwargs: object) -> object:
            captured.update(kwargs)
            return build_results_dataset([], PercentileData(), 1e-2)

        monkeypatch.setattr(
            "cobre_bridge.comparators.results.compare_results", _recorder
        )
        return captured

    def _capture_out_dir(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Patch ``write_artifacts`` with a recorder; return the captured kwargs."""
        captured: dict[str, object] = {}

        def _recorder(*_a: object, **kwargs: object) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(
            "cobre_bridge.comparators.export.write_artifacts", _recorder
        )
        return captured

    # -- results tolerance -------------------------------------------------

    def test_results_config_fills_tolerance(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``[compare.results]`` tolerance reaches compare_results."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "cobre-bridge.toml").write_text(
            "[compare.results]\ntolerance = 4e-2\n", encoding="utf-8"
        )
        self._stub_readers(monkeypatch)
        captured = self._capture_results_tolerance(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["tolerance"] == 4e-2

    # -- out-dir from config -----------------------------------------------

    def test_out_dir_from_config_reaches_write_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``[compare] out_dir`` flows to ``write_artifacts`` as a ``Path``."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "cobre-bridge.toml").write_text(
            '[compare]\nout_dir = "art"\n', encoding="utf-8"
        )
        self._stub_readers(monkeypatch)
        self._capture_results_tolerance(monkeypatch)
        captured = self._capture_out_dir(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["out_dir"] == Path("art")

    def test_out_dir_env_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``COBRE_BRIDGE_OUT_DIR`` overrides a ``[compare] out_dir`` config value."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "cobre-bridge.toml").write_text(
            '[compare]\nout_dir = "art_cfg"\n', encoding="utf-8"
        )
        monkeypatch.setenv("COBRE_BRIDGE_OUT_DIR", "art_env")
        self._stub_readers(monkeypatch)
        self._capture_results_tolerance(monkeypatch)
        captured = self._capture_out_dir(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["out_dir"] == Path("art_env")

    def test_format_env_beats_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``COBRE_BRIDGE_FORMAT`` overrides a ``[compare] format`` config value.

        The config asks for ``console`` only (which writes no file artifacts);
        the env asks for ``parquet``. If the env wins, ``write_artifacts`` is
        called with ``formats=["parquet"]``; if the config wrongly won, it would
        not be called at all.
        """
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "cobre-bridge.toml").write_text(
            '[compare]\nformat = ["console"]\n', encoding="utf-8"
        )
        monkeypatch.setenv("COBRE_BRIDGE_FORMAT", "parquet")
        self._stub_readers(monkeypatch)
        self._capture_results_tolerance(monkeypatch)
        captured = self._capture_out_dir(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, _, _ = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        assert code == 0
        assert captured["formats"] == ["parquet"]

    # -- malformed config: warn-but-run -------------------------------------

    def test_malformed_config_warns_on_stderr_and_still_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed config warns on stderr only and keeps the built-in default."""
        workdir = self._isolate_config_env(tmp_path, monkeypatch)
        (workdir / "cobre-bridge.toml").write_text(
            "this is = not valid = toml\n", encoding="utf-8"
        )
        self._stub_readers(monkeypatch)
        captured = self._capture_results_tolerance(monkeypatch)
        cobre_dir = tmp_path / "cobre"
        cobre_dir.mkdir()

        code, stdout, stderr = self._invoke_main(
            ["compare", "newave", str(tmp_path / "nw"), str(cobre_dir)],
            monkeypatch,
        )

        # Not the config-error exit; the comparison still runs at the default.
        assert code == 0
        assert captured["tolerance"] == 1e-2
        assert "cobre-bridge.toml" in stderr
        assert "cobre-bridge.toml" not in stdout


class TestDashboardOpen:
    """ticket-016: ``dashboard --open`` launches the written HTML in a browser.

    Runs ``dashboard`` in-process via ``cli.main`` with the real dashboard build
    stubbed (``cobre_bridge.dashboard.build_dashboard``) so it only writes a tiny
    file at the output path — enough for the ``output_path.stat()`` size line to
    succeed without building a real dashboard. ``webbrowser.open`` is patched at
    its ``cobre_bridge.cli`` import site so no actual browser is launched.
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
    def _make_case_dir(tmp_path: Path) -> Path:
        """Create a case dir whose ``output/simulation`` path exists."""
        case_dir = tmp_path / "case"
        (case_dir / "output" / "simulation").mkdir(parents=True)
        return case_dir

    @staticmethod
    def _stub_build_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``build_dashboard`` to write a tiny file at the output path.

        The real build is skipped; a 1-byte file keeps the ``stat()`` size line
        in ``_run_dashboard`` from raising.
        """

        def _fake_build(_case_dir: Path, output_path: Path) -> None:
            output_path.write_text("x", encoding="utf-8")

        monkeypatch.setattr("cobre_bridge.dashboard.build_dashboard", _fake_build)

    def test_dashboard_open_calls_webbrowser_with_file_uri(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)
        expected_uri = (case_dir / "dashboard.html").resolve().as_uri()

        with patch("cobre_bridge.cli.webbrowser.open", return_value=True) as mock_open:
            exit_code, _stdout, _stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--open"], monkeypatch
            )

        assert exit_code == 0
        mock_open.assert_called_once_with(expected_uri)

    def test_dashboard_without_open_does_not_call_webbrowser(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch("cobre_bridge.cli.webbrowser.open") as mock_open:
            exit_code, _stdout, _stderr = self._invoke_main(
                ["dashboard", str(case_dir)], monkeypatch
            )

        assert exit_code == 0
        mock_open.assert_not_called()

    def test_dashboard_open_swallows_webbrowser_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch(
            "cobre_bridge.cli.webbrowser.open",
            side_effect=webbrowser.Error("no browser"),
        ):
            exit_code, _stdout, stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--open"], monkeypatch
            )

        assert exit_code == 0
        assert "could not open a browser" in stderr

    def test_dashboard_open_swallows_false_return(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch("cobre_bridge.cli.webbrowser.open", return_value=False):
            exit_code, _stdout, stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--open"], monkeypatch
            )

        assert exit_code == 0
        assert "could not open a browser" in stderr


class TestDashboardJson:
    """ticket-021: ``dashboard --json`` emits the unified verdict envelope.

    Reuses ``TestDashboardOpen``'s in-process driver and build stub: the real
    dashboard build is replaced by a stub that writes a tiny file at the output
    path (so ``output_path.stat()`` succeeds), the CLI runs via ``cli.main`` with
    ``sys.stdout``/``sys.stderr`` captured, and the stdout is parsed as JSON. The
    file is STILL built under ``--json``; only the two Rich status lines are
    suppressed.
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
    def _make_case_dir(tmp_path: Path) -> Path:
        """Create a case dir whose ``output/simulation`` path exists."""
        case_dir = tmp_path / "case"
        (case_dir / "output" / "simulation").mkdir(parents=True)
        return case_dir

    @staticmethod
    def _stub_build_dashboard(monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``build_dashboard`` to write a tiny file at the output path."""

        def _fake_build(_case_dir: Path, output_path: Path) -> None:
            output_path.write_text("x", encoding="utf-8")

        monkeypatch.setattr("cobre_bridge.dashboard.build_dashboard", _fake_build)

    def test_dashboard_json_success_shape_and_exit_0(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "--json"], monkeypatch
        )

        assert exit_code == 0
        document = json.loads(stdout)
        assert list(document.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert document["command"] == "dashboard"
        assert document["status"] == "ok"
        assert document["summary"]["output"].endswith("dashboard.html")
        assert isinstance(document["summary"]["size_kb"], (int, float))
        assert document["diagnostics"] == []

    def test_dashboard_json_suppresses_status_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "--json"], monkeypatch
        )

        assert exit_code == 0
        assert "Building dashboard from" not in stdout
        assert "Dashboard written to" not in stdout
        # stdout parses as exactly one JSON object.
        assert json.loads(stdout)["command"] == "dashboard"

    def test_dashboard_json_no_simulation_exits_1_no_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A case dir WITHOUT output/simulation fires the exit-1 guard FIRST, so no
        # build and no verdict happen; stdout stays empty.
        case_dir = tmp_path / "case"
        case_dir.mkdir()
        self._stub_build_dashboard(monkeypatch)

        exit_code, stdout, stderr = self._invoke_main(
            ["dashboard", str(case_dir), "--json"], monkeypatch
        )

        assert exit_code == 1
        assert stdout == ""
        assert "no simulation output found" in stderr

    def test_dashboard_json_open_advisory_stays_on_stderr(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)

        with patch("cobre_bridge.cli.webbrowser.open", return_value=False):
            exit_code, stdout, stderr = self._invoke_main(
                ["dashboard", str(case_dir), "--json", "--open"], monkeypatch
            )

        assert exit_code == 0
        # stdout is exactly one verdict — no browser advisory leaked onto it.
        document = json.loads(stdout)
        assert document["command"] == "dashboard"
        assert "could not open a browser" not in stdout
        assert "could not open a browser" in stderr

    def test_dashboard_json_custom_output_path_in_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        case_dir = self._make_case_dir(tmp_path)
        self._stub_build_dashboard(monkeypatch)
        out_path = tmp_path / "OUT.html"

        exit_code, stdout, _stderr = self._invoke_main(
            ["dashboard", str(case_dir), "-o", str(out_path), "--json"], monkeypatch
        )

        assert exit_code == 0
        document = json.loads(stdout)
        assert document["summary"]["output"] == str(out_path)


class TestVerbosityAndLogFile:
    """ticket-017: graduated ``-v/-vv`` verbosity and the shared ``--log-file``.

    Drives ``cli.main`` in-process (NOT the Typer ``CliRunner``) so the ``main``
    ``finally`` teardown that removes the ``--log-file`` ``FileHandler`` actually
    runs, and so the progress gate sees a real (patchable) ``sys.stderr.isatty``.
    """

    @staticmethod
    def _invoke_main(
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        *,
        tty: bool = False,
    ) -> tuple[int, str, str]:
        """Run ``cli.main()`` in-process, capturing stdout/stderr and exit code.

        When *tty* is ``True`` the captured stderr buffer reports
        ``isatty() == True`` so the live-progress gate takes its interactive branch
        (otherwise a ``StringIO`` buffer is never a TTY and progress is always off).
        """
        import io

        from cobre_bridge import cli

        monkeypatch.setattr(sys, "argv", ["cobre-bridge", *argv])

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        if tty:
            stderr_buf.isatty = lambda: True  # type: ignore[method-assign]
        exit_code = 0

        with patch("sys.stdout", stdout_buf), patch("sys.stderr", stderr_buf):
            try:
                cli.main()
            except SystemExit as exc:
                exit_code = int(exc.code) if exc.code is not None else 0

        return exit_code, stdout_buf.getvalue(), stderr_buf.getvalue()

    @staticmethod
    def _file_handlers() -> list[logging.FileHandler]:
        """Return the ``FileHandler``s currently attached to the package logger."""
        pkg = logging.getLogger("cobre_bridge")
        return [h for h in pkg.handlers if isinstance(h, logging.FileHandler)]

    def test_configure_logging_levels(self) -> None:
        """The count maps 0 → warnings-only, 1 → INFO, 2 → DEBUG."""
        from cobre_bridge.cli import _NULL_HANDLER, _configure_logging

        pkg = logging.getLogger("cobre_bridge")

        _configure_logging(2, None)
        assert pkg.getEffectiveLevel() == logging.DEBUG

        _configure_logging(1, None)
        assert pkg.getEffectiveLevel() == logging.INFO

        _configure_logging(0, None)
        assert pkg.propagate is False
        assert _NULL_HANDLER in pkg.handlers

    def test_log_file_captures_debug_and_handler_removed_after_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--log-file`` writes the full DEBUG trace; the handler is gone after."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        log_path = tmp_path / "run.log"

        def _fake_convert(*_args: object, **_kwargs: object) -> ConversionReport:
            logging.getLogger("cobre_bridge.pipeline").debug("converting widgets")
            return ConversionReport(hydro_count=1, stage_count=12)

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            side_effect=_fake_convert,
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--log-file", str(log_path)],
                monkeypatch,
            )

        assert code == 0
        assert log_path.exists()
        contents = log_path.read_text(encoding="utf-8")
        assert "DEBUG cobre_bridge.pipeline: converting widgets" in contents
        # The FileHandler was removed + closed in main()'s finally.
        assert self._file_handlers() == []

    def test_consecutive_log_file_runs_leave_no_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two ``--log-file`` runs leave zero ``FileHandler``s (no leak)."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        report = ConversionReport(hydro_count=1, stage_count=12)

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=report,
        ):
            for run in ("a", "b"):
                code, _stdout, _stderr = self._invoke_main(
                    [
                        "convert",
                        "newave",
                        str(src),
                        str(tmp_path / f"dst_{run}"),
                        "--log-file",
                        str(tmp_path / f"run_{run}.log"),
                    ],
                    monkeypatch,
                )
                assert code == 0

        assert self._file_handlers() == []

    def test_log_file_without_verbose_keeps_progress_on_tty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a TTY, ``--log-file`` with no ``-v`` still shows live progress.

        Spies on ``ui.console._progress_enabled`` to assert it is consulted with
        ``verbose=False`` (progress NOT suppressed) and returns ``True``.
        """
        from cobre_bridge import ui
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        log_path = tmp_path / "run.log"
        report = ConversionReport(hydro_count=1, stage_count=12)

        real_progress_enabled = ui.console._progress_enabled
        # Record (verbose-flag, gate-result) pairs as they happen during the run,
        # so the TTY-dependent result is captured inside the patched context.
        calls: list[tuple[bool, bool]] = []

        def _spy(*, verbose: bool, quiet: bool) -> bool:
            result = real_progress_enabled(verbose=verbose, quiet=quiet)
            calls.append((verbose, result))
            return result

        with (
            patch("cobre_bridge.pipeline.convert_newave_case", return_value=report),
            patch.object(ui.console, "_progress_enabled", _spy),
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "--log-file", str(log_path)],
                monkeypatch,
                tty=True,
            )

        assert code == 0
        # conversion_progress consulted the gate with verbose=False (not suppressed)
        # and, on the simulated TTY, the gate returned True (progress shown).
        assert calls == [(False, True)]

    def test_vv_threads_int_and_suppresses_progress(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``dashboard -vv`` threads the int (2) and suppresses the spinner.

        Spies on ``_configure_logging`` (asserting it sees ``verbose=2``) and on
        ``ui.console._progress_enabled`` (asserting it is consulted with
        ``verbose=True`` — the ``> 0`` conversion of the count 2 — and returns
        ``False`` so the spinner is suppressed even on a TTY).
        """
        from cobre_bridge import cli, ui

        case_dir = tmp_path / "case"
        (case_dir / "output" / "simulation").mkdir(parents=True)

        def _fake_build(_case_dir: Path, output_path: Path) -> None:
            output_path.write_text("x", encoding="utf-8")

        monkeypatch.setattr("cobre_bridge.dashboard.build_dashboard", _fake_build)

        real_configure = cli._configure_logging
        configure_calls: list[int] = []

        def _configure_spy(verbose: int, log_file: Path | None) -> None:
            configure_calls.append(verbose)
            real_configure(verbose, log_file)

        progress_calls: list[bool] = []

        def _progress_spy(*, verbose: bool, quiet: bool) -> bool:
            progress_calls.append(verbose)
            return False

        with (
            patch.object(cli, "_configure_logging", _configure_spy),
            patch.object(ui.console, "_progress_enabled", _progress_spy),
        ):
            code, _stdout, _stderr = self._invoke_main(
                ["dashboard", str(case_dir), "-vv"],
                monkeypatch,
                tty=True,
            )

        assert code == 0
        # The int count threaded through unchanged.
        assert configure_calls == [2]
        # The spinner consulted the gate with verbose=True (count 2 → > 0) and the
        # gate (here forced False) suppressed it.
        assert progress_calls == [True]

    def test_vv_accepted_via_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Smoke: ``-vv`` is accepted on the real CLI and exits cleanly (code 0)."""
        from cobre_bridge.pipeline import ConversionReport

        src = _make_fake_newave_dir(tmp_path)
        dst = tmp_path / "dst"
        report = ConversionReport(hydro_count=1, stage_count=12)

        with patch(
            "cobre_bridge.pipeline.convert_newave_case",
            return_value=report,
        ):
            code, stdout, _stderr = self._invoke_main(
                ["convert", "newave", str(src), str(dst), "-vv"],
                monkeypatch,
            )

        assert code == 0
        assert "1 hydros" in stdout
