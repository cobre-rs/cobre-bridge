"""Tests for the DECOMP scenario emitters, pipeline, and CLI command."""

from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from cobre_bridge.cli import app
from cobre_bridge.decomp.bounds import _HYDRO_BOUNDS_SCHEMA
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.network import _LINE_BOUNDS_SCHEMA
from cobre_bridge.decomp.scenarios import (
    convert_external_inflows,
    convert_inflow_stats_identity,
    convert_scenario_probabilities,
    deterministic_external_scenarios,
    terminal_fan_probabilities,
)
from cobre_bridge.decomp.temporal import build_operative_calendar
from cobre_bridge.decomp.thermal import _THERMAL_BOUNDS_SCHEMA

_RV3_DECK = Path("example/decomp-jul-26-rv3")

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2),
    bus_names=("SE", "S"),
    hydro_codes=(1, 2),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            1: {"posto": 11, "codigo_usina_jusante": 2},
            2: {"posto": 12, "codigo_usina_jusante": 0},
        }
    ).T
    df.index.name = "codigo_usina"
    return df


class _StubVazoes:
    def __init__(self) -> None:
        self.previsoes = pd.DataFrame(
            [
                {"estagio": 1, "11": 100.0, "12": 150.0},
                {"estagio": 2, "11": 110.0, "12": 160.0},
            ]
        )
        self.cenarios_gerados = pd.DataFrame(
            [
                {"estagio": 3, "cenario": 1, "11": 90.0, "12": 140.0},
                {"estagio": 3, "cenario": 2, "11": 120.0, "12": 180.0},
            ]
        )
        self.probabilidades = pd.DataFrame(
            [
                {"estagio": 1, "no": 1, "cenario": 1, "probabilidade": 1.0},
                {"estagio": 2, "no": 2, "cenario": 1, "probabilidade": 1.0},
                {"estagio": 3, "no": 3, "cenario": 1, "probabilidade": 0.6},
                {"estagio": 3, "no": 4, "cenario": 2, "probabilidade": 0.4},
            ]
        )


class TestScenarioEmitters:
    def test_external_inflows_are_incremental(self) -> None:
        table = convert_external_inflows(
            _StubVazoes(), _hidr_frame(), _ID_MAP, _calendar()
        ).to_pandas()
        # 4 tree nodes × 2 hydros.
        assert len(table) == 8
        trunk = table[(table["stage_id"] == 0) & (table["hydro_id"] == 1)]
        # Plant 2's incremental subtracts its operated upstream (plant 1).
        assert trunk["value_m3s"].iloc[0] == pytest.approx(150.0 - 100.0)
        fan = table[(table["stage_id"] == 2) & (table["scenario_id"] == 1)]
        assert fan[fan["hydro_id"] == 0]["value_m3s"].iloc[0] == pytest.approx(120.0)
        assert fan[fan["hydro_id"] == 1]["value_m3s"].iloc[0] == pytest.approx(60.0)

    def test_identity_stats(self) -> None:
        stats = convert_inflow_stats_identity(_ID_MAP, _calendar()).to_pandas()
        assert len(stats) == 2 * 3
        assert set(stats["mean_m3s"]) == {0.0}
        assert set(stats["std_m3s"]) == {1.0}

    def test_deterministic_external_scenarios_replicate_across_columns(self) -> None:
        stats = pa.table(
            {
                "ncs_id": pa.array([0, 0, 1, 1], type=pa.int32()),
                "stage_id": pa.array([0, 1, 0, 1], type=pa.int32()),
                "mean": pa.array([0.5, 0.6, 0.7, 0.8], type=pa.float64()),
            }
        )
        # Trunk stage 0 -> 1 column; terminal stage 1 -> a 3-member fan.
        out = deterministic_external_scenarios(
            stats,
            entity_column="ncs_id",
            value_in="mean",
            value_out="value",
            scenario_counts=[1, 3],
        ).to_pandas()
        # stage 0: 2 entities × 1 col; stage 1: 2 entities × 3 cols.
        assert len(out) == 2 + 6
        fan = out[(out["stage_id"] == 1) & (out["ncs_id"] == 0)].sort_values(
            "scenario_id"
        )
        assert list(fan["scenario_id"]) == [0, 1, 2]
        assert list(fan["value"]) == [0.6, 0.6, 0.6]

    def test_probabilities_shape_and_sums(self) -> None:
        table = convert_scenario_probabilities(_StubVazoes(), _calendar()).to_pandas()
        sums = table.groupby("stage_id")["probability"].sum()
        assert sums.tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert table["scenario_id"].min() == 0

    def test_bad_probability_sum_raises(self) -> None:
        vazoes = _StubVazoes()
        vazoes.probabilidades.loc[3, "probabilidade"] = 0.3
        with pytest.raises(ValueError, match="sum"):
            convert_scenario_probabilities(vazoes, _calendar())

    def test_terminal_fan_probabilities_ordered(self) -> None:
        # Terminal (1-based stage 3) fan: cenario 1 -> 0.6, cenario 2 -> 0.4,
        # returned ordered by 0-based scenario id.
        probs = terminal_fan_probabilities(_StubVazoes(), _calendar())
        assert probs == pytest.approx([0.6, 0.4])

    def test_terminal_fan_probabilities_bad_sum_raises(self) -> None:
        vazoes = _StubVazoes()
        vazoes.probabilidades.loc[3, "probabilidade"] = 0.3  # terminal sums 0.9
        with pytest.raises(ValueError, match="sum"):
            terminal_fan_probabilities(vazoes, _calendar())

    def test_pre_terminal_fan_raises(self) -> None:
        vazoes = _StubVazoes()
        vazoes.cenarios_gerados.loc[0, "estagio"] = 2
        with pytest.raises(ValueError, match="node-graph"):
            convert_external_inflows(vazoes, _hidr_frame(), _ID_MAP, _calendar())


_EXPECTED_ARTIFACTS = [
    "config.json",
    "stages.json",
    "penalties.json",
    "initial_conditions.json",
    "system/buses.json",
    "system/hydros.json",
    "system/lines.json",
    "system/pumping_stations.json",
    "system/thermals.json",
    "system/hydro_production_models.json",
    "system/hydro_energy_productivity.parquet",
    "system/non_controllable_sources.json",
    "scenarios/inflow_seasonal_stats.parquet",
    "scenarios/external_inflow_scenarios.parquet",
    "scenarios/external_ncs_scenarios.parquet",
    "scenarios/external_load_scenarios.parquet",
    "scenarios/load_seasonal_stats.parquet",
    "scenarios/load_factors.json",
    "scenarios/non_controllable_stats.parquet",
    "scenarios/non_controllable_factors.json",
    "constraints/thermal_bounds.parquet",
    "constraints/line_bounds.parquet",
    "constraints/hydro_bounds.parquet",
]


class TestPipeline:
    @pytest.mark.skipif(
        not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
    )
    def test_rv3_full_conversion(self, tmp_path: Path) -> None:
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        dst = tmp_path / "case"
        convert_decomp_case(_RV3_DECK, dst)

        for artifact in _EXPECTED_ARTIFACTS:
            assert (dst / artifact).is_file(), artifact
        # The exchange-factors document is retired: line_bounds carries the
        # per-block absolute-MW rows directly, so nothing writes this file.
        assert not (dst / "constraints" / "exchange_factors.json").exists()

        stages = json.loads((dst / "stages.json").read_text())
        assert len(stages["stages"]) == 3
        # No per-stage num_openings on external-only DECOMP stages.
        assert all("num_openings" not in s for s in stages["stages"])
        graph = stages["policy_graph"]
        assert graph["annual_discount_rate"] == pytest.approx(0.12)
        # 3 stages: 2 trunk nodes (0,1) + a 353-node terminal fan (ids 2..354).
        assert len(graph["nodes"]) == 2 + 353
        assert sum(1 for n in graph["nodes"] if n["stage_id"] == 0) == 1
        fan_edges = [t for t in graph["transitions"] if t["source_id"] == 1]
        assert len(fan_edges) == 353
        assert sum(t["probability"] for t in fan_edges) == pytest.approx(1.0, abs=1e-4)

        config = json.loads((dst / "config.json").read_text())
        # Training enumerates the tree; simulation samples the fan (C10 gap).
        assert config["training"]["selection"] == {"method": "enumerated"}
        assert config["simulation"]["selection"] == {
            "method": "sampled",
            "num_scenarios": 353,
        }
        assert config["training"]["stopping_rules"] == [
            {"type": "gap", "relative_tolerance": 0.001},
            {"type": "iteration_limit", "limit": 500},
        ]
        source = config["training"]["scenario_source"]
        assert source["inflow"]["scheme"] == "external"
        # Every stochastic class is external: inflow (the tree), load, NCS.
        assert source["ncs"]["scheme"] == "external"
        assert source["load"]["scheme"] == "external"
        assert source["seed"] == 20260718

        buses = json.loads((dst / "system" / "buses.json").read_text())["buses"]
        assert len(buses) == 6

        external = pq.read_table(
            dst / "scenarios" / "external_inflow_scenarios.parquet"
        )
        hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
        assert external.num_rows == len(hydros) * (1 + 1 + 353)
        # External NCS library: 32 sources × (trunk col + trunk col + 353 fan).
        ext_ncs = pq.read_table(dst / "scenarios" / "external_ncs_scenarios.parquet")
        n_ncs = len(
            json.loads((dst / "system" / "non_controllable_sources.json").read_text())[
                "non_controllable_sources"
            ]
        )
        assert ext_ncs.num_rows == n_ncs * (1 + 1 + 353)
        # External load library: 6 buses × (trunk col + trunk col + 353 fan).
        ext_load = pq.read_table(dst / "scenarios" / "external_load_scenarios.parquet")
        assert set(ext_load.column_names) == {
            "stage_id",
            "scenario_id",
            "bus_id",
            "value_mw",
        }
        assert ext_load.num_rows == len(buses) * (1 + 1 + 353)

        with pytest.raises(FileExistsError, match="force"):
            convert_decomp_case(_RV3_DECK, dst)

    def test_missing_deck_raises(self, tmp_path: Path) -> None:
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        with pytest.raises(FileNotFoundError, match="caso.dat"):
            convert_decomp_case(tmp_path, tmp_path / "out")


class TestEmissionCheckWiring:
    """The post-emission self-checks (ticket-016, epic-04) run inside
    ``convert_decomp_case``, before the constraint writes."""

    @pytest.mark.skipif(
        not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
    )
    def test_rv3_conversion_reports_rule_43_not_applicable_and_no_errors(
        self, tmp_path: Path
    ) -> None:
        """DECOMP writes ``max_turbined_m3s``/``max_generation_mw`` only on the
        entity, never as per-stage ``hydro_bounds`` rows, so rule 43 is always
        "not applicable" here (AC #8) — and the production deck converts
        clean under the other three rules too (AC #6)."""
        from cobre_bridge import diagnostics as dx
        from cobre_bridge.cli import _convert_status
        from cobre_bridge.decomp.pipeline import convert_decomp_case
        from cobre_bridge.diagnostics import Severity

        dst = tmp_path / "case"
        with dx.collect() as collected:
            convert_decomp_case(_RV3_DECK, dst)

        not_applicable = [
            d for d in collected if d.code == "hydro-bounds-raising-not-applicable"
        ]
        assert len(not_applicable) == 1
        assert not_applicable[0].severity is Severity.INFO

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert errors == [], f"unexpected ERROR diagnostics: {errors}"
        assert _convert_status(collected, success="ok") == "ok"

    def test_decomp_shaped_violation_flips_convert_status_through_the_same_function(
        self,
    ) -> None:
        """A violation built from DECOMP-shaped artifacts still flips the
        verdict via ``cli._convert_status`` — the single function both
        pipelines' convert verdicts key off (AC #3), not a bare inspection of
        the diagnostic."""
        from cobre_bridge import diagnostics as dx
        from cobre_bridge.cli import _convert_status
        from cobre_bridge.emission_checks import check_hydro_bounds_no_raising

        hydros = {
            "hydros": [
                {
                    "id": 0,
                    "generation": {
                        "max_turbined_m3s": 100.0,
                        "max_generation_mw": 50.0,
                    },
                    "unit_groups": [
                        {"max_turbined_m3s": 100.0, "max_generation_mw": 50.0}
                    ],
                }
            ]
        }
        hydro_bounds = pa.table(
            {
                "hydro_id": pa.array([0], type=pa.int32()),
                "stage_id": pa.array([1], type=pa.int32()),
                "max_turbined_m3s": pa.array([999.0], type=pa.float64()),
            }
        )

        with dx.collect() as collected:
            check_hydro_bounds_no_raising(hydros, hydro_bounds)

        assert _convert_status(collected, success="ok") == "error"
        assert _convert_status([], success="ok") == "ok"

    @pytest.mark.skipif(
        not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
    )
    def test_a_real_emission_check_error_makes_the_real_conversion_fail(
        self, tmp_path: Path
    ) -> None:
        """A DECOMP deck carrying a synthetic emission-check violation makes
        the REAL conversion path fail: ``convert_decomp_case`` must raise, not
        merely compute an "error" status from hand-built diagnostics (Finding
        1, epic-04 boundary remediation). The violation is injected by
        patching ONE check function to unconditionally emit an ERROR
        diagnostic; file discovery, deck parsing, every converter, and the
        OTHER three checks all still run for real against the rv3 deck."""
        from cobre_bridge import diagnostics as dx
        from cobre_bridge import emission_checks
        from cobre_bridge.decomp.pipeline import convert_decomp_case
        from cobre_bridge.diagnostics import Diagnostic, Severity

        def _synthetic_violation(*args: object, **kwargs: object) -> None:
            dx.emit(
                Diagnostic(
                    code="synthetic-test-violation",
                    severity=Severity.ERROR,
                    category="Emission self-checks",
                    title="Synthetic violation injected by the test",
                    summary="synthetic ERROR diagnostic for Finding-1 coverage",
                )
            )

        dst = tmp_path / "case"
        with (
            patch.object(
                emission_checks,
                "check_unit_group_envelope",
                side_effect=_synthetic_violation,
            ),
            pytest.raises(ValueError, match="synthetic-test-violation"),
        ):
            convert_decomp_case(_RV3_DECK, dst)

        # The raise happens before the constraint tables are written, so no
        # half-valid case should look convertible on a bare retry.
        assert not (dst / "constraints" / "hydro_bounds.parquet").exists()

    @pytest.mark.skipif(
        not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
    )
    def test_normal_rv3_deck_still_converts_clean_via_the_real_pipeline(
        self, tmp_path: Path
    ) -> None:
        """A normal deck's rule-43 "not applicable" INFO finding must not
        trip the new raise — only ERROR severity does."""
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        dst = tmp_path / "case"
        convert_decomp_case(_RV3_DECK, dst)  # must not raise

        assert (dst / "constraints" / "hydro_bounds.parquet").is_file()


class TestContractWiring:
    """ticket-004 (epic-01): the CI/CE contract model wires into
    ``convert_decomp_case``, gated on non-empty like every other bound
    family it joins in ``bound_families``."""

    @pytest.mark.skipif(
        not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
    )
    def test_rv3_conversion_writes_no_contract_files(self, tmp_path: Path) -> None:
        """rv3 has no contracts (D6 skips its lone placeholder row), so
        neither gated write fires and the conversion still completes clean."""
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        dst = tmp_path / "case"
        convert_decomp_case(_RV3_DECK, dst)

        assert not (dst / "system" / "energy_contracts.json").exists()
        assert not (dst / "constraints" / "contract_bounds.parquet").exists()

    @pytest.mark.skipif(
        not (_RV3_DECK / "caso.dat").exists(), reason="rv3 deck not present"
    )
    def test_deck_with_contract_writes_both_files(self, tmp_path: Path) -> None:
        """A deck carrying one non-placeholder contract writes both gated
        files, non-empty — ``read_contracts`` is patched to yield it so the
        rest of the real rv3 conversion (dadger parsing, id_map, calendar,
        every other converter) still runs for real."""
        from collections.abc import Sequence

        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.contracts import Contract, ContractStage
        from cobre_bridge.decomp.pipeline import convert_decomp_case
        from cobre_bridge.decomp.temporal import OperativeStage

        def _fake_read_contracts(
            dadger: Dadger, calendar: Sequence[OperativeStage]
        ) -> list[Contract]:
            sb = dadger.sb(df=True)
            bus_code = int(sb["codigo_submercado"].iloc[0])
            stages = [
                ContractStage(
                    min_mw=[0.0] * len(stage.block_hours),
                    max_mw=[100.0] * len(stage.block_hours),
                    custo=[50.0] * len(stage.block_hours),
                    loss_factor=None,
                )
                for stage in calendar
            ]
            return [
                Contract(
                    id=0,
                    kind="import",
                    numero=1,
                    name="Test Contract",
                    bus_code=bus_code,
                    stages=stages,
                )
            ]

        dst = tmp_path / "case"
        with patch(
            "cobre_bridge.decomp.pipeline.contracts_conv.read_contracts",
            side_effect=_fake_read_contracts,
        ):
            convert_decomp_case(_RV3_DECK, dst)

        contracts_path = dst / "system" / "energy_contracts.json"
        bounds_path = dst / "constraints" / "contract_bounds.parquet"
        assert contracts_path.is_file()
        assert bounds_path.is_file()

        contracts_doc = json.loads(contracts_path.read_text())
        assert len(contracts_doc["contracts"]) == 1

        bounds_table = pq.read_table(bounds_path)
        assert bounds_table.num_rows > 0


class TestCli:
    def test_convert_decomp_invokes_pipeline(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch("cobre_bridge.decomp.pipeline.convert_decomp_case") as mock_convert:
            result = runner.invoke(
                app,
                ["convert", "decomp", str(tmp_path), str(tmp_path / "out"), "--force"],
            )
        assert result.exit_code == 0
        mock_convert.assert_called_once()
        assert mock_convert.call_args.kwargs["force"] is True

    def test_convert_decomp_failure_exits_one(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=FileNotFoundError("caso.dat not found"),
        ):
            result = runner.invoke(
                app, ["convert", "decomp", str(tmp_path), str(tmp_path / "out")]
            )
        assert result.exit_code == 1

    def test_convert_decomp_emission_check_error_exits_one(
        self, tmp_path: Path
    ) -> None:
        """The ``ValueError`` ``convert_decomp_case`` raises on an
        ERROR-severity post-emission self-check finding (Finding 1) is caught
        by the same except tuple as a missing/existing-destination failure —
        no dedicated CLI branch is needed."""
        runner = CliRunner()
        with patch(
            "cobre_bridge.decomp.pipeline.convert_decomp_case",
            side_effect=ValueError(
                "DECOMP conversion failed 1 post-emission self-check error(s)"
            ),
        ):
            result = runner.invoke(
                app, ["convert", "decomp", str(tmp_path), str(tmp_path / "out")]
            )
        assert result.exit_code == 1


# ticket-008 (epic-02, cadastro overrides): a fully synthetic Tier-1 harness
# for ``convert_decomp_case`` — no real deck under ``example/``. Every
# converter the pipeline calls other than the five this ticket wires
# (``build_effective_cadastro``, ``convert_initial_storage``,
# ``convert_hydros``, ``convert_hydro_bounds``, ``convert_storage_bounds``)
# is patched to a canned return value, so the test exercises only this
# ticket's own orchestration: threading the effective cadastro through,
# combining ``hydro_bounds`` with the storage-bounds overlay, and the
# resolution-report summary diagnostic.


def _cadastro_plant_row(
    name: str, sub: int, jusante: int, vmin: float, vmax: float
) -> dict:
    return {
        "nome_usina": name,
        "submercado": sub,
        "codigo_usina_jusante": jusante,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 100.0,
        "potencia_nominal_conjunto_1": 50.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": 100.0,
        "a1_volume_cota": 0.0,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        "canal_fuga_medio": 20.0,
        "produtibilidade_especifica": 0.009,
        "tipo_perda": 0,
        "perdas": 0.0,
    }


def _cadastro_hidr_frame() -> pd.DataFrame:
    """Two plants, codes 1 and 2, both on bus (submercado) 1, no cascade."""
    df = pd.DataFrame(
        {
            1: _cadastro_plant_row("PLANT_ONE", 1, 0, 20.0, 100.0),
            2: _cadastro_plant_row("PLANT_TWO", 1, 0, 10.0, 50.0),
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _cadastro_uh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            },
            {
                "codigo_usina": 2,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            },
        ]
    )


_CADASTRO_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))


class _CadastroDadger:
    """Mock ``Dadger`` covering only what the real consumers below need:
    ``.tx.taxa`` (the discount rate), ``.uh(df=True)`` (the operated
    registrations both ``convert_hydros``/``convert_initial_storage`` read),
    and ``.ac(...)`` — returning *ac_volmax_frame* for the ``ACVOLMAX``
    subtype (the real idecomp representation: string month abbreviation,
    float ``semana``/``ano``) and ``None`` (no override) for every other
    ``AC`` subtype, including ``ACVOLMIN`` and the four machine-configuration
    overrides ``convert_hydros`` also probes.
    """

    class _Tx:
        taxa = 12.0

    def __init__(self, ac_volmax_frame: pd.DataFrame | None) -> None:
        self.tx = self._Tx()
        self._uh = _cadastro_uh_frame()
        self._ac_volmax_frame = ac_volmax_frame

    def uh(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._uh

    def ac(
        self,
        codigo_usina: int | None = None,  # noqa: ARG002
        modificacao: type | None = None,
        df: bool = False,  # noqa: ARG002
    ) -> pd.DataFrame | None:
        from idecomp.decomp.modelos.dadger import ACVOLMAX

        if modificacao is ACVOLMAX:
            return self._ac_volmax_frame
        return None


def _run_cadastro_pipeline(
    tmp_path: Path, ac_volmax_frame: pd.DataFrame | None
) -> Path:
    """Run ``convert_decomp_case`` against the fully synthetic mock deck
    above, patching every converter this ticket does not wire to a canned
    return value. Returns the case directory.
    """
    from cobre_bridge.decomp.pipeline import DecompFiles, convert_decomp_case

    files = DecompFiles(
        revision="rv0",
        dadger=Path("unused/dadger.rv0"),
        vazoes=Path("unused/vazoes.rv0"),
        hidr=Path("unused/hidr.dat"),
        dadgnl=None,
        renovaveis=None,
    )
    dadger = _CadastroDadger(ac_volmax_frame)
    hidr = _cadastro_hidr_frame()
    calendar = _calendar()

    productivity_table = pa.table(
        {"equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64())}
    )
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean_mw": pa.array([10.0, 10.0, 10.0], type=pa.float64()),
        }
    )
    ncs_stats_table = pa.table(
        {
            "ncs_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean": pa.array([5.0, 5.0, 5.0], type=pa.float64()),
        }
    )
    external_inflow_table = pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "scenario_id": pa.array([], type=pa.int32()),
            "value_m3s": pa.array([], type=pa.float64()),
        }
    )
    # The pre-ticket-008 baseline hydro_bounds: one RQ/UH-derived min-outflow
    # row, no storage columns populated — this ticket's own combine logic
    # (concat + sort with the real convert_storage_bounds output) is what is
    # under test, not convert_hydro_bounds' own RQ/UH logic (out of scope).
    baseline_hydro_bounds = pa.table(
        {
            "hydro_id": pa.array([0], type=pa.int32()),
            "stage_id": pa.array([0], type=pa.int32()),
            "block_id": pa.array([None], type=pa.int32()),
            "min_outflow_m3s": pa.array([5.0], type=pa.float64()),
            "min_storage_hm3": pa.array([None], type=pa.float64()),
            "max_storage_hm3": pa.array([None], type=pa.float64()),
        },
        schema=_HYDRO_BOUNDS_SCHEMA,
    )

    patches: dict[str, object] = {
        "cobre_bridge.decomp.pipeline.discover_decomp_files": files,
        "cobre_bridge.decomp.pipeline.Dadger.read": dadger,
        "cobre_bridge.decomp.pipeline.Vazoes.read": object(),
        "cobre_bridge.decomp.pipeline.hydro_conv.read_hidr": hidr,
        "cobre_bridge.decomp.pipeline.DecompIdMap.from_dadger": _CADASTRO_ID_MAP,
        "cobre_bridge.decomp.pipeline"
        ".temporal_conv.operative_calendar_from_dadger": calendar,
        "cobre_bridge.decomp.pipeline.scenarios_conv.terminal_fan_probabilities": [1.0],
        "cobre_bridge.decomp.pipeline.config_conv.convert_config": {},
        "cobre_bridge.decomp.pipeline.network_conv._bus_deficit_costs": {},
        "cobre_bridge.decomp.pipeline"
        ".hydro_conv.convert_energy_productivity": productivity_table,
        "cobre_bridge.decomp.pipeline.network_conv.convert_buses": {"buses": []},
        "cobre_bridge.decomp.pipeline.network_conv.convert_lines": (
            {"lines": []},
            _LINE_BOUNDS_SCHEMA.empty_table(),
        ),
        "cobre_bridge.decomp.pipeline.network_conv.convert_pumping_stations": {
            "pumping_stations": []
        },
        "cobre_bridge.decomp.pipeline.thermal_conv.convert_thermals": {"thermals": []},
        "cobre_bridge.decomp.pipeline.ncs_conv.convert_non_controllable_sources": {
            "non_controllable_sources": []
        },
        "cobre_bridge.decomp.pipeline"
        ".scenarios_conv.convert_external_inflows": external_inflow_table,
        "cobre_bridge.decomp.pipeline.load_conv.convert_load_stats": load_stats_table,
        "cobre_bridge.decomp.pipeline.load_conv.convert_load_factors": {},
        "cobre_bridge.decomp.pipeline.ncs_conv.convert_ncs_stats": ncs_stats_table,
        "cobre_bridge.decomp.pipeline.ncs_conv.convert_ncs_factors": {},
        "cobre_bridge.decomp.pipeline"
        ".thermal_conv.convert_thermal_bounds": _THERMAL_BOUNDS_SCHEMA.empty_table(),
        "cobre_bridge.decomp.pipeline"
        ".bounds_conv.convert_hydro_bounds": baseline_hydro_bounds,
        "cobre_bridge.decomp.pipeline.hydro_conv.convert_hydro_group_availability": (
            {},
            [],
        ),
        "cobre_bridge.decomp.pipeline.contracts_conv.read_contracts": [],
    }
    with ExitStack() as stack:
        for target, value in patches.items():
            stack.enter_context(patch(target, return_value=value))
        dst = tmp_path / "case"
        convert_decomp_case(Path("unused-src"), dst)
    return dst


class TestCadastroPipelineWiring:
    """ticket-008: ``build_effective_cadastro`` threads into
    ``convert_decomp_case`` and the storage-bounds overlay folds into the
    same ``hydro_bounds.parquet`` the RQ/UH minimum-outflow rows populate."""

    def test_no_override_regresses_hydro_bounds_and_entity_output(
        self, tmp_path: Path
    ) -> None:
        """No ``AC`` volume record: the combined table equals the pre-ticket
        baseline (no storage rows), and ``initial_conditions.json`` /
        ``system/hydros.json`` reflect the base registry values unchanged."""
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=None)

        table = pq.read_table(dst / "constraints" / "hydro_bounds.parquet")
        assert table.num_rows == 1
        assert table["min_outflow_m3s"].to_pylist() == [5.0]
        assert table["min_storage_hm3"].to_pylist() == [None]
        assert table["max_storage_hm3"].to_pylist() == [None]

        hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
        reservoirs = {h["id"]: h["reservoir"] for h in hydros}
        assert reservoirs[0] == {"min_storage_hm3": 20.0, "max_storage_hm3": 100.0}
        assert reservoirs[1] == {"min_storage_hm3": 10.0, "max_storage_hm3": 50.0}

        storage = json.loads((dst / "initial_conditions.json").read_text())["storage"]
        values = {row["hydro_id"]: row["value_hm3"] for row in storage}
        # 50% of the base [20, 100] / [10, 50] ranges.
        assert values[0] == pytest.approx(60.0)
        assert values[1] == pytest.approx(30.0)

    def test_temporal_override_adds_storage_rows_and_raises_the_entity_envelope(
        self, tmp_path: Path
    ) -> None:
        """An ``ACVOLMAX`` row raising plant 1's ``volume_maximo`` to 250.0
        from the final stage forward widens its entity envelope to 250.0
        (``system/hydros.json``) and adds override rows to the combined
        ``hydro_bounds`` table. Per Rule A (ticket-006, ``storage_envelope``):
        the raised (final) stage itself now *equals* the widened envelope and
        needs no override row; the earlier stages, which still sit at the
        narrower pre-raise ceiling, are what differ from it and get the
        override rows — non-null ``max_storage_hm3``, null ``min_outflow_m3s``.
        """
        ac_volmax_frame = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "volume": 250.0,
                    "mes": "AGO",
                    "semana": 0.0,
                    "ano": 2026.0,
                }
            ]
        )
        dst = _run_cadastro_pipeline(tmp_path, ac_volmax_frame=ac_volmax_frame)

        table = pq.read_table(dst / "constraints" / "hydro_bounds.parquet")
        storage_rows = table.to_pylist()
        storage_only = [
            row
            for row in storage_rows
            if row["max_storage_hm3"] is not None and row["min_outflow_m3s"] is None
        ]
        assert len(storage_only) >= 1
        assert all(row["max_storage_hm3"] == 100.0 for row in storage_only)
        # The pre-ticket-008 baseline row (min_outflow, no storage columns)
        # is still present, untouched by the combine.
        outflow_only = [row for row in storage_rows if row["min_outflow_m3s"] == 5.0]
        assert len(outflow_only) == 1

        hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
        reservoirs = {h["id"]: h["reservoir"] for h in hydros}
        assert reservoirs[0]["max_storage_hm3"] == pytest.approx(250.0)

    def test_temporal_override_emits_exactly_one_summary_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """The resolution-report summary is a single INFO diagnostic naming
        ``volume_maximo`` among the applied overrides."""
        from cobre_bridge import diagnostics as dx

        ac_volmax_frame = pd.DataFrame(
            [
                {
                    "codigo_usina": 1,
                    "volume": 250.0,
                    "mes": "AGO",
                    "semana": 0.0,
                    "ano": 2026.0,
                }
            ]
        )
        with dx.collect() as collected:
            _run_cadastro_pipeline(tmp_path, ac_volmax_frame=ac_volmax_frame)

        cadastro_diagnostics = [
            d for d in collected if d.code == "cadastro-overrides-applied"
        ]
        assert len(cadastro_diagnostics) == 1
        assert cadastro_diagnostics[0].severity is dx.Severity.INFO
        assert "volume_maximo" in cadastro_diagnostics[0].summary
