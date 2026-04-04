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
