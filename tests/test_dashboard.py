"""Tests for the dashboard data layer and tab registry.

Covers:
- Data loader helpers in cobre_bridge.dashboard.data
- Tab registry (get_renderable_tabs, TAB_MODULES)
- can_render contracts for constraints and stochastic tabs
- TabModule protocol compliance for every registered module
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from cobre_bridge.dashboard.data import (
    entity_name,
    load_hydro_bus_map,
    load_hydro_metadata,
    load_names,
    load_ncs_bus_map,
    load_stage_labels,
    load_thermal_metadata,
    scan_entity,
)
from cobre_bridge.dashboard.tabs import TAB_MODULES, get_renderable_tabs
from cobre_bridge.dashboard.tabs import constraints as constraints_tab
from cobre_bridge.dashboard.tabs import stochastic as stochastic_tab

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# load_names
# ---------------------------------------------------------------------------


def test_load_names_returns_dict_with_hydros(tmp_path: Path) -> None:
    # Arrange
    hydros_json = {
        "hydros": [{"id": 0, "name": "Itaipu"}, {"id": 1, "name": "Tucurui"}]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    # Act
    result = load_names(tmp_path)

    # Assert
    assert result[("hydros", 0)] == "Itaipu"
    assert result[("hydros", 1)] == "Tucurui"


def test_load_names_missing_system_dir_returns_empty(tmp_path: Path) -> None:
    # No files written — system/ directory does not exist
    result = load_names(tmp_path)

    assert result == {}


def test_load_names_uses_id_as_fallback_when_name_absent(tmp_path: Path) -> None:
    # Arrange — item has no "name" key
    hydros_json = {"hydros": [{"id": 5}]}
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_names(tmp_path)

    assert result[("hydros", 5)] == "5"


def test_load_names_multiple_entity_types(tmp_path: Path) -> None:
    # Arrange
    _write_json(
        tmp_path / "system" / "hydros.json",
        {"hydros": [{"id": 0, "name": "H0"}]},
    )
    _write_json(
        tmp_path / "system" / "buses.json",
        {"buses": [{"id": 0, "name": "B0"}]},
    )

    result = load_names(tmp_path)

    assert result[("hydros", 0)] == "H0"
    assert result[("buses", 0)] == "B0"


# ---------------------------------------------------------------------------
# load_stage_labels
# ---------------------------------------------------------------------------


def test_load_stage_labels_returns_formatted_labels(tmp_path: Path) -> None:
    # Arrange
    stages_json = {
        "stages": [
            {"id": 0, "start_date": "2024-01-01"},
            {"id": 1, "start_date": "2024-02-01"},
        ]
    }
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    assert result[0] == "Jan 2024"
    assert result[1] == "Feb 2024"


def test_load_stage_labels_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_stage_labels(tmp_path)

    assert result == {}


def test_load_stage_labels_falls_back_to_stage_id_on_invalid_date(
    tmp_path: Path,
) -> None:
    stages_json = {
        "stages": [
            {"id": 3, "start_date": "not-a-date"},
        ]
    }
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    assert result[3] == "3"


def test_load_stage_labels_falls_back_to_stage_id_when_start_date_absent(
    tmp_path: Path,
) -> None:
    stages_json = {"stages": [{"id": 7}]}
    _write_json(tmp_path / "stages.json", stages_json)

    result = load_stage_labels(tmp_path)

    assert result[7] == "7"


# ---------------------------------------------------------------------------
# load_hydro_bus_map
# ---------------------------------------------------------------------------


def test_load_hydro_bus_map_returns_mapping(tmp_path: Path) -> None:
    hydros_json = {
        "hydros": [
            {"id": 0, "bus_id": 10},
            {"id": 1, "bus_id": 20},
        ]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_hydro_bus_map(tmp_path)

    assert result == {0: 10, 1: 20}


def test_load_hydro_bus_map_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_hydro_bus_map(tmp_path)

    assert result == {}


# ---------------------------------------------------------------------------
# load_thermal_metadata
# ---------------------------------------------------------------------------


def test_load_thermal_metadata_extracts_cost_segments(tmp_path: Path) -> None:
    thermals_json = {
        "thermals": [
            {
                "id": 0,
                "bus_id": 5,
                "name": "Gas Plant",
                "cost_segments": [
                    {"capacity_mw": 100.0, "cost_per_mwh": 150.0},
                    {"capacity_mw": 50.0, "cost_per_mwh": 200.0},
                ],
            }
        ]
    }
    _write_json(tmp_path / "system" / "thermals.json", thermals_json)

    result = load_thermal_metadata(tmp_path)

    assert result[0]["bus_id"] == 5
    assert result[0]["name"] == "Gas Plant"
    assert result[0]["max_mw"] == pytest.approx(150.0)
    assert result[0]["cost_per_mwh"] == pytest.approx(150.0)


def test_load_thermal_metadata_falls_back_to_generation_max_mw(tmp_path: Path) -> None:
    thermals_json = {
        "thermals": [
            {
                "id": 2,
                "bus_id": 3,
                "name": "Coal",
                "cost_segments": [],
                "generation": {"max_mw": 300.0},
            }
        ]
    }
    _write_json(tmp_path / "system" / "thermals.json", thermals_json)

    result = load_thermal_metadata(tmp_path)

    assert result[2]["max_mw"] == pytest.approx(300.0)
    assert result[2]["cost_per_mwh"] == pytest.approx(0.0)


def test_load_thermal_metadata_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_thermal_metadata(tmp_path)

    assert result == {}


# ---------------------------------------------------------------------------
# load_ncs_bus_map
# ---------------------------------------------------------------------------


def test_load_ncs_bus_map_returns_mapping(tmp_path: Path) -> None:
    ncs_json = {
        "non_controllable_sources": [
            {"id": 0, "bus_id": 7},
            {"id": 1, "bus_id": 8},
        ]
    }
    _write_json(tmp_path / "system" / "non_controllable_sources.json", ncs_json)

    result = load_ncs_bus_map(tmp_path)

    assert result == {0: 7, 1: 8}


def test_load_ncs_bus_map_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_ncs_bus_map(tmp_path)

    assert result == {}


# ---------------------------------------------------------------------------
# load_hydro_metadata
# ---------------------------------------------------------------------------


def test_load_hydro_metadata_extracts_fields(tmp_path: Path) -> None:
    hydros_json = {
        "hydros": [
            {
                "id": 0,
                "bus_id": 1,
                "name": "Belo Monte",
                "reservoir": {"max_storage_hm3": 5000.0, "min_storage_hm3": 100.0},
                "generation": {
                    "max_generation_mw": 11000.0,
                    "productivity_mw_per_m3s": 0.08,
                    "max_turbined_m3s": 150000.0,
                },
            }
        ]
    }
    _write_json(tmp_path / "system" / "hydros.json", hydros_json)

    result = load_hydro_metadata(tmp_path)

    assert result[0]["name"] == "Belo Monte"
    assert result[0]["bus_id"] == 1
    assert result[0]["vol_max"] == pytest.approx(5000.0)
    assert result[0]["vol_min"] == pytest.approx(100.0)
    assert result[0]["max_gen_mw"] == pytest.approx(11000.0)
    assert result[0]["max_turbined"] == pytest.approx(150000.0)
    assert result[0]["productivity"] == pytest.approx(0.08)


def test_load_hydro_metadata_missing_file_returns_empty(tmp_path: Path) -> None:
    result = load_hydro_metadata(tmp_path)

    assert result == {}


# ---------------------------------------------------------------------------
# entity_name
# ---------------------------------------------------------------------------


def test_entity_name_returns_name_when_key_present() -> None:
    names: dict[tuple[str, int], str] = {("hydros", 0): "Itaipu", ("buses", 5): "SE"}

    assert entity_name(names, "hydros", 0) == "Itaipu"
    assert entity_name(names, "buses", 5) == "SE"


def test_entity_name_returns_str_id_when_key_absent() -> None:
    names: dict[tuple[str, int], str] = {}

    assert entity_name(names, "hydros", 42) == "42"


# ---------------------------------------------------------------------------
# scan_entity
# ---------------------------------------------------------------------------


def test_scan_entity_calls_scan_parquet_with_correct_path(tmp_path: Path) -> None:
    expected_path = str(
        tmp_path / "output" / "simulation" / "hydros" / "**" / "*.parquet"
    )
    mock_lf = MagicMock(spec=pl.LazyFrame)

    with patch(
        "cobre_bridge.dashboard.data.pl.scan_parquet", return_value=mock_lf
    ) as mock_scan:
        result = scan_entity(tmp_path, "hydros")

    mock_scan.assert_called_once_with(expected_path, hive_partitioning=True)
    assert result is mock_lf


# ---------------------------------------------------------------------------
# get_renderable_tabs — ordering
# ---------------------------------------------------------------------------


def test_get_renderable_tabs_returns_tabs_sorted_by_tab_order() -> None:
    # Arrange: two mock modules with reversed ORDER values
    mock_high = MagicMock()
    mock_high.TAB_ID = "tab-high"
    mock_high.TAB_LABEL = "High"
    mock_high.TAB_ORDER = 200
    mock_high.can_render.return_value = True
    mock_high.render.return_value = "<div>high</div>"

    mock_low = MagicMock()
    mock_low.TAB_ID = "tab-low"
    mock_low.TAB_LABEL = "Low"
    mock_low.TAB_ORDER = 5
    mock_low.can_render.return_value = True
    mock_low.render.return_value = "<div>low</div>"

    fake_data = MagicMock()

    with patch("cobre_bridge.dashboard.tabs.TAB_MODULES", [mock_high, mock_low]):
        result = get_renderable_tabs(fake_data)

    ids = [tab_id for tab_id, _label, _html in result]
    assert ids == ["tab-low", "tab-high"]


# ---------------------------------------------------------------------------
# get_renderable_tabs — filtering
# ---------------------------------------------------------------------------


def test_get_renderable_tabs_excludes_modules_where_can_render_is_false() -> None:
    # Arrange: one renderable, one not
    mock_yes = MagicMock()
    mock_yes.TAB_ID = "tab-yes"
    mock_yes.TAB_LABEL = "Yes"
    mock_yes.TAB_ORDER = 10
    mock_yes.can_render.return_value = True
    mock_yes.render.return_value = "<div>yes</div>"

    mock_no = MagicMock()
    mock_no.TAB_ID = "tab-no"
    mock_no.TAB_LABEL = "No"
    mock_no.TAB_ORDER = 20
    mock_no.can_render.return_value = False

    fake_data = MagicMock()

    with patch("cobre_bridge.dashboard.tabs.TAB_MODULES", [mock_yes, mock_no]):
        result = get_renderable_tabs(fake_data)

    ids = [tab_id for tab_id, _label, _html in result]
    assert "tab-yes" in ids
    assert "tab-no" not in ids


# ---------------------------------------------------------------------------
# get_renderable_tabs — error handling
# ---------------------------------------------------------------------------


def test_get_renderable_tabs_skips_tab_when_render_raises() -> None:
    # Arrange: one tab raises, one succeeds
    mock_bad = MagicMock()
    mock_bad.TAB_ID = "tab-bad"
    mock_bad.TAB_LABEL = "Bad"
    mock_bad.TAB_ORDER = 10
    mock_bad.can_render.return_value = True
    mock_bad.render.side_effect = RuntimeError("rendering failed")

    mock_good = MagicMock()
    mock_good.TAB_ID = "tab-good"
    mock_good.TAB_LABEL = "Good"
    mock_good.TAB_ORDER = 20
    mock_good.can_render.return_value = True
    mock_good.render.return_value = "<div>good</div>"

    fake_data = MagicMock()

    with patch("cobre_bridge.dashboard.tabs.TAB_MODULES", [mock_bad, mock_good]):
        result = get_renderable_tabs(fake_data)

    ids = [tab_id for tab_id, _label, _html in result]
    assert "tab-bad" not in ids
    assert "tab-good" in ids


def test_get_renderable_tabs_returns_correct_tuple_structure() -> None:
    mock_mod = MagicMock()
    mock_mod.TAB_ID = "tab-x"
    mock_mod.TAB_LABEL = "X Tab"
    mock_mod.TAB_ORDER = 0
    mock_mod.can_render.return_value = True
    mock_mod.render.return_value = "<section>content</section>"

    fake_data = MagicMock()

    with patch("cobre_bridge.dashboard.tabs.TAB_MODULES", [mock_mod]):
        result = get_renderable_tabs(fake_data)

    assert len(result) == 1
    tab_id, label, html = result[0]
    assert tab_id == "tab-x"
    assert label == "X Tab"
    assert html == "<section>content</section>"


# ---------------------------------------------------------------------------
# constraints.can_render
# ---------------------------------------------------------------------------


def test_constraints_can_render_returns_false_when_gc_constraints_empty() -> None:
    data = MagicMock()
    data.gc_constraints = []

    assert constraints_tab.can_render(data) is False


def test_constraints_can_render_returns_true_when_gc_constraints_non_empty() -> None:
    data = MagicMock()
    data.gc_constraints = [
        {"id": 0, "name": "VminOP_test", "expression": "hydro_storage(0)"}
    ]

    assert constraints_tab.can_render(data) is True


# ---------------------------------------------------------------------------
# stochastic.can_render
# ---------------------------------------------------------------------------


def test_stochastic_can_render_returns_false_when_stochastic_unavailable() -> None:
    data = MagicMock()
    data.stochastic_available = False

    assert stochastic_tab.can_render(data) is False


def test_stochastic_can_render_returns_true_when_stochastic_available() -> None:
    data = MagicMock()
    data.stochastic_available = True

    assert stochastic_tab.can_render(data) is True


# ---------------------------------------------------------------------------
# TabModule protocol compliance — parametrized over all registered modules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_tab_id(module: Any) -> None:
    assert hasattr(module, "TAB_ID"), f"{module} missing TAB_ID"
    assert isinstance(module.TAB_ID, str), f"{module}.TAB_ID must be str"
    assert module.TAB_ID, f"{module}.TAB_ID must not be empty"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_tab_label(module: Any) -> None:
    assert hasattr(module, "TAB_LABEL"), f"{module} missing TAB_LABEL"
    assert isinstance(module.TAB_LABEL, str), f"{module}.TAB_LABEL must be str"
    assert module.TAB_LABEL, f"{module}.TAB_LABEL must not be empty"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_tab_order(module: Any) -> None:
    assert hasattr(module, "TAB_ORDER"), f"{module} missing TAB_ORDER"
    assert isinstance(module.TAB_ORDER, int), f"{module}.TAB_ORDER must be int"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_can_render_callable(module: Any) -> None:
    assert hasattr(module, "can_render"), f"{module} missing can_render"
    assert callable(module.can_render), f"{module}.can_render must be callable"


@pytest.mark.parametrize("module", TAB_MODULES)
def test_tab_module_has_render_callable(module: Any) -> None:
    assert hasattr(module, "render"), f"{module} missing render"
    assert callable(module.render), f"{module}.render must be callable"


# ---------------------------------------------------------------------------
# Integration test — full build_dashboard() pipeline
# ---------------------------------------------------------------------------


class TestDashboardIntegration:
    """Integration tests for build_dashboard() with a minimal mock case directory.

    Exercises the full pipeline: DashboardData.load(), get_renderable_tabs(),
    and build_html() assembling a complete self-contained HTML file.
    """

    @pytest.fixture()
    def case_dir(self, tmp_path: Path) -> Path:
        """Build a minimal case directory for DashboardData.load()."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        case = tmp_path / "test_case"
        case.mkdir()

        # ---- stages.json ----
        stages_data = {
            "stages": [
                {
                    "id": 0,
                    "start_date": "2026-01-01",
                    "blocks": [
                        {"id": 0, "hours": 120.0},
                        {"id": 1, "hours": 300.0},
                    ],
                },
                {
                    "id": 1,
                    "start_date": "2026-02-01",
                    "blocks": [
                        {"id": 0, "hours": 112.0},
                        {"id": 1, "hours": 280.0},
                    ],
                },
            ]
        }
        _write_json(case / "stages.json", stages_data)

        # ---- config.json ----
        _write_json(case / "config.json", {"num_scenarios": 2, "num_stages": 2})

        # ---- system/ JSON files ----
        _write_json(
            case / "system" / "hydros.json",
            {
                "hydros": [
                    {
                        "id": 0,
                        "name": "HYDRO_A",
                        "bus_id": 0,
                        "reservoir": {
                            "max_storage_hm3": 5000.0,
                            "min_storage_hm3": 100.0,
                        },
                        "generation": {
                            "max_generation_mw": 1000.0,
                            "productivity_mw_per_m3s": 0.08,
                            "max_turbined_m3s": 12000.0,
                        },
                    }
                ]
            },
        )
        _write_json(
            case / "system" / "buses.json",
            {"buses": [{"id": 0, "name": "SE"}]},
        )
        _write_json(
            case / "system" / "thermals.json",
            {
                "thermals": [
                    {
                        "id": 0,
                        "name": "GAS_A",
                        "bus_id": 0,
                        "cost_segments": [
                            {"capacity_mw": 300.0, "cost_per_mwh": 200.0}
                        ],
                    }
                ]
            },
        )
        _write_json(
            case / "system" / "lines.json",
            {"lines": []},
        )
        _write_json(
            case / "system" / "non_controllable_sources.json",
            {"non_controllable_sources": []},
        )

        # ---- scenarios/ ----
        (case / "scenarios").mkdir(parents=True, exist_ok=True)
        load_stats_table = pa.table(
            {
                "bus_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "mean_mw": pa.array([50000.0, 48000.0], type=pa.float64()),
                "std_mw": pa.array([0.0, 0.0], type=pa.float64()),
            }
        )
        pq.write_table(
            load_stats_table, case / "scenarios" / "load_seasonal_stats.parquet"
        )
        _write_json(
            case / "scenarios" / "load_factors.json",
            {
                "load_factors": [
                    {
                        "bus_id": 0,
                        "stage_id": 0,
                        "block_factors": [
                            {"block_id": 0, "factor": 1.05},
                            {"block_id": 1, "factor": 0.95},
                        ],
                    },
                    {
                        "bus_id": 0,
                        "stage_id": 1,
                        "block_factors": [
                            {"block_id": 0, "factor": 1.03},
                            {"block_id": 1, "factor": 0.97},
                        ],
                    },
                ]
            },
        )

        # ---- output/training/convergence.parquet ----
        conv_dir = case / "output" / "training"
        conv_dir.mkdir(parents=True)
        conv_table = pa.table(
            {
                "iteration": pa.array([1, 2], type=pa.int32()),
                "lower_bound": pa.array([1.0e9, 1.1e9], type=pa.float64()),
                "upper_bound_mean": pa.array([1.5e9, 1.4e9], type=pa.float64()),
                "upper_bound_std": pa.array([1.0e7, 9.0e6], type=pa.float64()),
                "gap_percent": pa.array([33.3, 21.4], type=pa.float64()),
                "cuts_added": pa.array([10, 8], type=pa.int32()),
                "cuts_removed": pa.array([0, 0], type=pa.int32()),
                "cuts_active": pa.array([10, 18], type=pa.int64()),
                "time_forward_ms": pa.array([100, 90], type=pa.int64()),
                "time_backward_ms": pa.array([200, 180], type=pa.int64()),
                "time_total_ms": pa.array([300, 270], type=pa.int64()),
                "forward_passes": pa.array([5, 5], type=pa.int32()),
                "lp_solves": pa.array([100, 90], type=pa.int64()),
            }
        )
        pq.write_table(conv_table, conv_dir / "convergence.parquet")

        # ---- Simulation entity directories (hive-partitioned) ----
        # Columns for each entity (scenario_id is inferred from directory name
        # by polars hive_partitioning=True; it must NOT be in the file columns).

        sim_base = case / "output" / "simulation"

        def _write_sim_parquet(entity: str, scenario_id: int, table: pa.Table) -> None:
            d = sim_base / entity / f"scenario_id={scenario_id:04d}"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, d / "data.parquet")

        # hydros
        hydro_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "generation_mw": pa.array([800.0, 750.0], type=pa.float64()),
                "generation_mwh": pa.array([96000.0, 84000.0], type=pa.float64()),
                "spillage_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "turbined_m3s": pa.array([10000.0, 9500.0], type=pa.float64()),
                "storage_final_hm3": pa.array([4500.0, 4600.0], type=pa.float64()),
                "storage_initial_hm3": pa.array([4400.0, 4500.0], type=pa.float64()),
                "inflow_m3s": pa.array([500.0, 480.0], type=pa.float64()),
                "outflow_m3s": pa.array([10000.0, 9500.0], type=pa.float64()),
                "incremental_inflow_m3s": pa.array([500.0, 480.0], type=pa.float64()),
                "water_value_per_hm3": pa.array([1.5, 1.4], type=pa.float64()),
                "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "evaporation_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "productivity_mw_per_m3s": pa.array([0.08, 0.08], type=pa.float64()),
                "storage_binding_code": pa.array([0, 0], type=pa.int8()),
                "operative_state_code": pa.array([1, 1], type=pa.int8()),
                "turbined_slack_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_slack_below_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_slack_above_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "generation_slack_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "storage_violation_below_hm3": pa.array([0.0, 0.0], type=pa.float64()),
                "filling_target_violation_hm3": pa.array([0.0, 0.0], type=pa.float64()),
                "diverted_inflow_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "diverted_outflow_m3s": pa.array([0.0, 0.0], type=pa.float64()),
                "evaporation_violation_pos_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "evaporation_violation_neg_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "inflow_nonnegativity_slack_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "water_withdrawal_violation_pos_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
                "water_withdrawal_violation_neg_m3s": pa.array(
                    [0.0, 0.0], type=pa.float64()
                ),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("hydros", sid, hydro_table)

        # thermals
        thermal_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "thermal_id": pa.array([0, 0], type=pa.int32()),
                "generation_mw": pa.array([200.0, 210.0], type=pa.float64()),
                "generation_mwh": pa.array([24000.0, 23520.0], type=pa.float64()),
                "generation_cost": pa.array([4.8e6, 4.7e6], type=pa.float64()),
                "is_gnl": pa.array([False, False]),
                "gnl_committed_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "gnl_decision_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "operative_state_code": pa.array([1, 1], type=pa.int8()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("thermals", sid, thermal_table)

        # non_controllables
        ncs_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "non_controllable_id": pa.array([0, 0], type=pa.int32()),
                "generation_mw": pa.array([100.0, 90.0], type=pa.float64()),
                "generation_mwh": pa.array([12000.0, 10080.0], type=pa.float64()),
                "available_mw": pa.array([110.0, 100.0], type=pa.float64()),
                "curtailment_mw": pa.array([10.0, 10.0], type=pa.float64()),
                "curtailment_mwh": pa.array([1200.0, 1120.0], type=pa.float64()),
                "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "operative_state_code": pa.array([1, 1], type=pa.int8()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("non_controllables", sid, ncs_table)

        # buses
        bus_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([0, 0], type=pa.int32()),
                "bus_id": pa.array([0, 0], type=pa.int32()),
                "load_mw": pa.array([1000.0, 980.0], type=pa.float64()),
                "load_mwh": pa.array([120000.0, 109760.0], type=pa.float64()),
                "deficit_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "deficit_mwh": pa.array([0.0, 0.0], type=pa.float64()),
                "excess_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "excess_mwh": pa.array([0.0, 0.0], type=pa.float64()),
                "spot_price": pa.array([150.0, 145.0], type=pa.float64()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("buses", sid, bus_table)

        # exchanges (empty — no lines defined)
        exchange_table = pa.table(
            {
                "stage_id": pa.array([], type=pa.int32()),
                "block_id": pa.array([], type=pa.int32()),
                "line_id": pa.array([], type=pa.int32()),
                "direct_flow_mw": pa.array([], type=pa.float64()),
                "reverse_flow_mw": pa.array([], type=pa.float64()),
                "net_flow_mw": pa.array([], type=pa.float64()),
                "net_flow_mwh": pa.array([], type=pa.float64()),
                "losses_mw": pa.array([], type=pa.float64()),
                "losses_mwh": pa.array([], type=pa.float64()),
                "exchange_cost": pa.array([], type=pa.float64()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("exchanges", sid, exchange_table)

        # costs
        cost_table = pa.table(
            {
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "block_id": pa.array([None, None], type=pa.int32()),
                "total_cost": pa.array([5.0e9, 4.8e9], type=pa.float64()),
                "immediate_cost": pa.array([5.0e8, 4.8e8], type=pa.float64()),
                "future_cost": pa.array([4.5e9, 4.32e9], type=pa.float64()),
                "discount_factor": pa.array([1.0, 0.99], type=pa.float64()),
                "thermal_cost": pa.array([4.8e6, 4.7e6], type=pa.float64()),
                "contract_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "deficit_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "excess_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "storage_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "filling_target_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "hydro_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_violation_below_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "outflow_violation_above_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "turbined_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "generation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "evaporation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "withdrawal_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "inflow_penalty_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "generic_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "fpha_turbined_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "exchange_cost": pa.array([0.0, 0.0], type=pa.float64()),
                "pumping_cost": pa.array([0.0, 0.0], type=pa.float64()),
            }
        )
        for sid in (0, 1):
            _write_sim_parquet("costs", sid, cost_table)

        return case

    def test_build_dashboard_integration(self, case_dir: Path, tmp_path: Path) -> None:
        """build_dashboard() writes a valid HTML file with at least 3 tab sections."""
        from cobre_bridge.dashboard import build_dashboard

        output_path = tmp_path / "dashboard.html"

        build_dashboard(case_dir, output_path)

        assert output_path.exists(), "Dashboard HTML file was not written"
        html = output_path.read_text(encoding="utf-8")

        assert "<!DOCTYPE html>" in html

        section_count = html.count('<section id="tab-')
        assert section_count >= 3, (
            f"Expected at least 3 tab sections, found {section_count}"
        )

        assert case_dir.resolve().name in html


# ---------------------------------------------------------------------------
# DashboardData.load() — new v2 fields (ticket-001)
#
# These tests reuse the full minimal case directory from
# TestDashboardIntegration via the module-level ``_v2_case`` fixture.
# Each test writes one optional file, calls DashboardData.load(), and
# asserts the corresponding new field.
# ---------------------------------------------------------------------------


@pytest.fixture()
def _v2_case(tmp_path: Path) -> Path:
    """Build the same minimal Cobre case used by TestDashboardIntegration.

    Returns the ``case`` Path so individual tests can write optional files
    before calling ``DashboardData.load()``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    case = tmp_path / "v2_case"
    case.mkdir()

    _write_json(
        case / "stages.json",
        {
            "stages": [
                {
                    "id": 0,
                    "start_date": "2026-01-01",
                    "blocks": [
                        {"id": 0, "hours": 120.0},
                        {"id": 1, "hours": 300.0},
                    ],
                },
                {
                    "id": 1,
                    "start_date": "2026-02-01",
                    "blocks": [
                        {"id": 0, "hours": 112.0},
                        {"id": 1, "hours": 280.0},
                    ],
                },
            ]
        },
    )
    _write_json(case / "system" / "hydros.json", {"hydros": []})
    _write_json(case / "system" / "buses.json", {"buses": []})
    _write_json(case / "system" / "thermals.json", {"thermals": []})
    _write_json(case / "system" / "lines.json", {"lines": []})
    _write_json(
        case / "system" / "non_controllable_sources.json",
        {"non_controllable_sources": []},
    )

    (case / "scenarios").mkdir(parents=True, exist_ok=True)
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "mean_mw": pa.array([], type=pa.float64()),
            "std_mw": pa.array([], type=pa.float64()),
        }
    )
    pq.write_table(load_stats_table, case / "scenarios" / "load_seasonal_stats.parquet")
    _write_json(case / "scenarios" / "load_factors.json", {"load_factors": []})

    conv_dir = case / "output" / "training"
    conv_dir.mkdir(parents=True)
    conv_table = pa.table(
        {
            "iteration": pa.array([1, 2], type=pa.int32()),
            "lower_bound": pa.array([1.0e9, 1.1e9], type=pa.float64()),
            "upper_bound_mean": pa.array([1.5e9, 1.4e9], type=pa.float64()),
            "upper_bound_std": pa.array([1.0e7, 9.0e6], type=pa.float64()),
            "gap_percent": pa.array([33.3, 21.4], type=pa.float64()),
            "cuts_added": pa.array([10, 8], type=pa.int32()),
            "cuts_removed": pa.array([0, 0], type=pa.int32()),
            "cuts_active": pa.array([10, 18], type=pa.int64()),
            "time_forward_ms": pa.array([100, 90], type=pa.int64()),
            "time_backward_ms": pa.array([200, 180], type=pa.int64()),
            "time_total_ms": pa.array([300, 270], type=pa.int64()),
            "forward_passes": pa.array([5, 5], type=pa.int32()),
            "lp_solves": pa.array([100, 90], type=pa.int64()),
        }
    )
    pq.write_table(conv_table, conv_dir / "convergence.parquet")

    sim_base = case / "output" / "simulation"

    def _write_sim_parquet(entity: str, scenario_id: int, table: pa.Table) -> None:
        d = sim_base / entity / f"scenario_id={scenario_id:04d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, d / "data.parquet")

    empty_entity = pa.table(
        {
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
        }
    )
    for entity in ("hydros", "thermals", "non_controllables", "buses", "exchanges"):
        for sid in (0, 1):
            _write_sim_parquet(entity, sid, empty_entity)

    costs_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([None, None], type=pa.int32()),
            "total_cost": pa.array([5.0e9, 4.8e9], type=pa.float64()),
            "immediate_cost": pa.array([5.0e8, 4.8e8], type=pa.float64()),
            "future_cost": pa.array([4.5e9, 4.32e9], type=pa.float64()),
            "discount_factor": pa.array([1.0, 0.99], type=pa.float64()),
            "thermal_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "contract_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "deficit_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "excess_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "storage_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "filling_target_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "hydro_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_violation_below_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_violation_above_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "turbined_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "generation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "evaporation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "withdrawal_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "inflow_penalty_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "generic_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "fpha_turbined_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "exchange_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "pumping_cost": pa.array([0.0, 0.0], type=pa.float64()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("costs", sid, costs_table)

    return case


def test_load_config_present(_v2_case: Path) -> None:
    """config and discount_rate are populated from config.json when it exists."""
    from cobre_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "config.json",
        {"discount_rate": 0.12, "iterations": 500},
    )

    data = DashboardData.load(_v2_case)

    assert data.config["iterations"] == 500
    assert data.discount_rate == pytest.approx(0.12)


def test_load_config_absent(_v2_case: Path) -> None:
    """config defaults to {} and discount_rate to 0.0 when config.json is absent."""
    from cobre_bridge.dashboard.data import DashboardData

    # Ensure no config.json is present
    config_path = _v2_case / "config.json"
    config_path.unlink(missing_ok=True)

    data = DashboardData.load(_v2_case)

    assert data.config == {}
    assert data.discount_rate == pytest.approx(0.0)


def test_load_training_manifest_present(_v2_case: Path) -> None:
    """training_manifest is populated from output/training/_manifest.json."""
    from cobre_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "output" / "training" / "_manifest.json",
        {"status": "converged", "total_cuts": 936},
    )

    data = DashboardData.load(_v2_case)

    assert data.training_manifest["status"] == "converged"
    assert data.training_manifest["total_cuts"] == 936


def test_load_policy_metadata_present(_v2_case: Path) -> None:
    """policy_metadata is populated from output/policy/metadata.json."""
    from cobre_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "output" / "policy" / "metadata.json",
        {"state_dimension": 1106},
    )

    data = DashboardData.load(_v2_case)

    assert data.policy_metadata["state_dimension"] == 1106


def test_load_stages_data_preserved(_v2_case: Path) -> None:
    """stages_data contains the raw stages list from stages.json."""
    from cobre_bridge.dashboard.data import DashboardData

    data = DashboardData.load(_v2_case)

    assert "stages" in data.stages_data
    assert isinstance(data.stages_data["stages"], list)
    assert len(data.stages_data["stages"]) > 0


def test_simulation_manifest_field(_v2_case: Path) -> None:
    """simulation_manifest is populated from the manifest file and metadata
    no longer carries a _sim_manifest side-effect key."""
    from cobre_bridge.dashboard.data import DashboardData

    _write_json(
        _v2_case / "output" / "simulation" / "_manifest.json",
        {"scenarios": {"completed": 2, "total": 2}, "status": "done"},
    )

    data = DashboardData.load(_v2_case)

    assert data.simulation_manifest["status"] == "done"
    assert data.simulation_manifest["scenarios"]["completed"] == 2
    assert "_sim_manifest" not in data.metadata


# ---------------------------------------------------------------------------
# DashboardData.load() — ticket-002: constraint bounds and scenario stats
# ---------------------------------------------------------------------------


def test_load_hydro_bounds_present(_v2_case: Path) -> None:
    """hydro_bounds is a non-empty DataFrame with expected columns when file exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    bounds_dir = _v2_case / "constraints"
    bounds_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "min_storage_hm3": pa.array([100.0, 100.0], type=pa.float64()),
                "max_storage_hm3": pa.array([5000.0, 5000.0], type=pa.float64()),
            }
        ),
        bounds_dir / "hydro_bounds.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.hydro_bounds.empty
    assert list(data.hydro_bounds.columns) == [
        "hydro_id",
        "stage_id",
        "min_storage_hm3",
        "max_storage_hm3",
    ]


def test_load_hydro_bounds_absent(_v2_case: Path) -> None:
    """hydro_bounds is an empty DataFrame when hydro_bounds.parquet is absent."""
    from cobre_bridge.dashboard.data import DashboardData

    # Ensure the file is not present
    hb_path = _v2_case / "constraints" / "hydro_bounds.parquet"
    hb_path.unlink(missing_ok=True)

    data = DashboardData.load(_v2_case)

    assert data.hydro_bounds.empty


def test_load_thermal_bounds_present(_v2_case: Path) -> None:
    """thermal_bounds is a non-empty DataFrame with expected columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    bounds_dir = _v2_case / "constraints"
    bounds_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "thermal_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "min_generation_mw": pa.array([0.0, 0.0], type=pa.float64()),
                "max_generation_mw": pa.array([300.0, 300.0], type=pa.float64()),
            }
        ),
        bounds_dir / "thermal_bounds.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.thermal_bounds.empty
    assert list(data.thermal_bounds.columns) == [
        "thermal_id",
        "stage_id",
        "min_generation_mw",
        "max_generation_mw",
    ]


def test_load_ncs_stats_present(_v2_case: Path) -> None:
    """ncs_stats is a non-empty DataFrame when non_controllable_stats.parquet exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    scenarios_dir = _v2_case / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "non_controllable_id": pa.array([0, 0], type=pa.int32()),
                "stage_id": pa.array([0, 1], type=pa.int32()),
                "mean_mw": pa.array([80.0, 75.0], type=pa.float64()),
                "std_mw": pa.array([5.0, 4.0], type=pa.float64()),
            }
        ),
        scenarios_dir / "non_controllable_stats.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.ncs_stats.empty
    assert "non_controllable_id" in data.ncs_stats.columns
    assert "mean_mw" in data.ncs_stats.columns


def test_load_exchange_factors_present(_v2_case: Path) -> None:
    """exchange_factors is a list with the correct element when file exists."""
    from cobre_bridge.dashboard.data import DashboardData

    constraints_dir = _v2_case / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        constraints_dir / "exchange_factors.json",
        {"exchange_factors": [{"line_id": 0, "stage_id": 0, "factor": 1.05}]},
    )

    data = DashboardData.load(_v2_case)

    assert len(data.exchange_factors) == 1
    assert data.exchange_factors[0]["line_id"] == 0
    assert data.exchange_factors[0]["factor"] == pytest.approx(1.05)


# ---------------------------------------------------------------------------
# ticket-003: stochastic data fields
# ---------------------------------------------------------------------------


def test_load_inflow_history_present(_v2_case: Path) -> None:
    """inflow_history is a non-empty DataFrame when inflow_history.parquet exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    scenarios_dir = _v2_case / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array([0, 0, 1, 1], type=pa.int32()),
                "year": pa.array([2000, 2001, 2000, 2001], type=pa.int32()),
                "month": pa.array([1, 1, 1, 1], type=pa.int32()),
                "inflow_m3s": pa.array([100.0, 110.0, 50.0, 55.0], type=pa.float64()),
            }
        ),
        scenarios_dir / "inflow_history.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert not data.inflow_history.empty
    assert "hydro_id" in data.inflow_history.columns


def test_load_inflow_history_absent(_v2_case: Path) -> None:
    """inflow_history is an empty DataFrame when the file is missing."""
    from cobre_bridge.dashboard.data import DashboardData

    ih_path = _v2_case / "scenarios" / "inflow_history.parquet"
    assert not ih_path.exists()

    data = DashboardData.load(_v2_case)

    assert data.inflow_history.empty


def test_load_correlation_present(_v2_case: Path) -> None:
    """correlation contains the expected key when correlation.json exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    stochastic_dir = _v2_case / "output" / "stochastic"
    stochastic_dir.mkdir(parents=True, exist_ok=True)

    # Write the required stochastic parquet files so stochastic_available is True
    for filename in (
        "inflow_seasonal_stats.parquet",
        "inflow_ar_coefficients.parquet",
        "noise_openings.parquet",
    ):
        pq.write_table(
            pa.table({"dummy": pa.array([], type=pa.int32())}),
            stochastic_dir / filename,
        )
    _write_json(stochastic_dir / "fitting_report.json", {})
    _write_json(
        stochastic_dir / "correlation.json",
        {"correlations": [[1.0, 0.3], [0.3, 1.0]]},
    )

    data = DashboardData.load(_v2_case)

    assert "correlations" in data.correlation
    assert data.correlation["correlations"] == [[1.0, 0.3], [0.3, 1.0]]


def test_load_correlation_absent_no_stochastic(_v2_case: Path) -> None:
    """correlation is an empty dict when the stochastic output directory is missing."""
    from cobre_bridge.dashboard.data import DashboardData

    stochastic_dir = _v2_case / "output" / "stochastic"
    assert not stochastic_dir.exists()

    data = DashboardData.load(_v2_case)

    assert data.correlation == {}


def test_load_inflow_lags_lf_present(_v2_case: Path) -> None:
    """inflow_lags_lf is a LazyFrame that collects when the directory exists."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    lags_dir = _v2_case / "output" / "simulation" / "inflow_lags" / "scenario_id=0"
    lags_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "hydro_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "lag_value": pa.array([1.5, 2.3], type=pa.float64()),
            }
        ),
        lags_dir / "data.parquet",
    )

    data = DashboardData.load(_v2_case)

    collected = data.inflow_lags_lf.collect()
    assert len(collected) > 0


# ---------------------------------------------------------------------------
# compute_non_fictitious_bus_ids — ticket-004
# ---------------------------------------------------------------------------


def test_compute_non_fictitious_bus_ids_filters_zero_load() -> None:
    """Bus with zero mean_mw in all stages is excluded; nonzero bus is included."""
    import pandas as pd

    from cobre_bridge.dashboard.data import compute_non_fictitious_bus_ids

    load_stats = pd.DataFrame(
        {
            "bus_id": [0, 0, 1, 1],
            "stage_id": [0, 1, 0, 1],
            "mean_mw": [100.0, 80.0, 0.0, 0.0],
        }
    )

    result = compute_non_fictitious_bus_ids(load_stats)

    assert result == [0]


def test_compute_non_fictitious_bus_ids_all_nonzero() -> None:
    """All buses with nonzero load in at least one stage are returned sorted."""
    import pandas as pd

    from cobre_bridge.dashboard.data import compute_non_fictitious_bus_ids

    load_stats = pd.DataFrame(
        {
            "bus_id": [0, 1, 0, 1],
            "stage_id": [0, 0, 1, 1],
            "mean_mw": [50.0, 10.0, 48.0, 9.0],
        }
    )

    result = compute_non_fictitious_bus_ids(load_stats)

    assert result == [0, 1]


def test_compute_non_fictitious_bus_ids_empty_df() -> None:
    """Empty DataFrame returns an empty list."""
    import pandas as pd

    from cobre_bridge.dashboard.data import compute_non_fictitious_bus_ids

    result = compute_non_fictitious_bus_ids(pd.DataFrame())

    assert result == []


def test_compute_non_fictitious_bus_ids_missing_column() -> None:
    """DataFrame missing mean_mw column returns an empty list (defensive)."""
    import pandas as pd

    from cobre_bridge.dashboard.data import compute_non_fictitious_bus_ids

    load_stats = pd.DataFrame({"bus_id": [0, 1], "stage_id": [0, 0]})

    result = compute_non_fictitious_bus_ids(load_stats)

    assert result == []


def test_non_fictitious_bus_ids_field_on_data(_v2_case: Path) -> None:
    """non_fictitious_bus_ids is populated as a sorted list of int on DashboardData."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from cobre_bridge.dashboard.data import DashboardData

    # Overwrite the load_stats with two buses: 0 has load, 1 is fictitious
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0, 1, 1], type=pa.int32()),
            "stage_id": pa.array([0, 1, 0, 1], type=pa.int32()),
            "mean_mw": pa.array([50000.0, 48000.0, 0.0, 0.0], type=pa.float64()),
            "std_mw": pa.array([0.0, 0.0, 0.0, 0.0], type=pa.float64()),
        }
    )
    pq.write_table(
        load_stats_table,
        _v2_case / "scenarios" / "load_seasonal_stats.parquet",
    )

    data = DashboardData.load(_v2_case)

    assert isinstance(data.non_fictitious_bus_ids, list)
    assert data.non_fictitious_bus_ids == [0]
    assert all(isinstance(bid, int) for bid in data.non_fictitious_bus_ids)
