"""Tests for the DECOMP scenario emitters, pipeline, and CLI command."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from cobre_bridge.cli import app
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.scenarios import (
    convert_external_inflows,
    convert_inflow_stats_identity,
    convert_noise_openings,
    convert_scenario_probabilities,
)
from cobre_bridge.decomp.temporal import build_operative_calendar

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

    def test_noise_openings_equal_external_values(self) -> None:
        vazoes, hidr, calendar = _StubVazoes(), _hidr_frame(), _calendar()
        external = convert_external_inflows(vazoes, hidr, _ID_MAP, calendar).to_pandas()
        noise = convert_noise_openings(vazoes, hidr, _ID_MAP, calendar, 0).to_pandas()
        assert list(noise["value"]) == list(external["value_m3s"])
        assert list(noise["opening_index"]) == list(external["scenario_id"])
        assert list(noise["entity_index"]) == list(external["hydro_id"])

    def test_noise_openings_span_the_whole_noise_vector(self) -> None:
        """Non-controllable sources hold their noise slots, at zero deviation.

        Their block follows the hydros in the solver's entity order, so
        omitting it would leave every consumer of the later slots reading
        past the end of each opening row.
        """
        vazoes, hidr, calendar = _StubVazoes(), _hidr_frame(), _calendar()
        n_hydros = len(_ID_MAP.hydro_codes)
        noise = convert_noise_openings(vazoes, hidr, _ID_MAP, calendar, 3).to_pandas()

        per_opening = noise.groupby(["stage_id", "opening_index"]).size()
        assert set(per_opening) == {n_hydros + 3}
        appended = noise[noise["entity_index"] >= n_hydros]
        assert sorted(set(appended["entity_index"])) == [n_hydros + i for i in range(3)]
        assert set(appended["value"]) == {0.0}

    def test_identity_stats(self) -> None:
        stats = convert_inflow_stats_identity(_ID_MAP, _calendar()).to_pandas()
        assert len(stats) == 2 * 3
        assert set(stats["mean_m3s"]) == {0.0}
        assert set(stats["std_m3s"]) == {1.0}

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
    "scenarios/noise_openings.parquet",
    "scenarios/scenario_probabilities.parquet",
    "scenarios/load_seasonal_stats.parquet",
    "scenarios/load_factors.json",
    "scenarios/non_controllable_stats.parquet",
    "scenarios/non_controllable_factors.json",
    "constraints/thermal_bounds.parquet",
    "constraints/line_bounds.parquet",
    "constraints/exchange_factors.json",
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

        stages = json.loads((dst / "stages.json").read_text())
        assert len(stages["stages"]) == 3
        assert [s["num_scenarios"] for s in stages["stages"]] == [1, 1, 353]
        assert stages["policy_graph"]["annual_discount_rate"] == pytest.approx(0.12)

        config = json.loads((dst / "config.json").read_text())
        assert config["training"]["stopping_rules"] == [
            {"type": "iteration_limit", "limit": 500}
        ]
        assert config["training"]["scenario_source"]["inflow"]["scheme"] == "external"
        assert config["training"]["scenario_source"]["seed"] == 20260718

        buses = json.loads((dst / "system" / "buses.json").read_text())["buses"]
        assert len(buses) == 6

        prob = pq.read_table(
            dst / "scenarios" / "scenario_probabilities.parquet"
        ).to_pandas()
        sums = prob.groupby("stage_id")["probability"].sum()
        assert sums.tolist() == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)

        external = pq.read_table(
            dst / "scenarios" / "external_inflow_scenarios.parquet"
        )
        hydros = json.loads((dst / "system" / "hydros.json").read_text())["hydros"]
        assert external.num_rows == len(hydros) * (1 + 1 + 353)

        with pytest.raises(FileExistsError, match="force"):
            convert_decomp_case(_RV3_DECK, dst)

    def test_missing_deck_raises(self, tmp_path: Path) -> None:
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        with pytest.raises(FileNotFoundError, match="caso.dat"):
            convert_decomp_case(tmp_path, tmp_path / "out")


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
