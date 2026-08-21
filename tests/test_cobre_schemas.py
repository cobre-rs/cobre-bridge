"""Drift guard: `cobre_schemas.SCHEMA_URLS` pinned against every live constant."""

from __future__ import annotations

import pytest

from cobre_bridge import cobre_schemas
from cobre_bridge.converters import constraints as converters_constraints
from cobre_bridge.converters import hydro as converters_hydro
from cobre_bridge.converters import initial_conditions as converters_initial_conditions
from cobre_bridge.converters import network as converters_network
from cobre_bridge.converters import scalar_parameters as converters_scalar_parameters
from cobre_bridge.converters import stochastic as converters_stochastic
from cobre_bridge.converters import thermal as converters_thermal
from cobre_bridge.decomp import config as decomp_config
from cobre_bridge.decomp import contracts as decomp_contracts
from cobre_bridge.decomp import network as decomp_network
from cobre_bridge.decomp import temporal as decomp_temporal

_NEWAVE_STAGES_INLINE_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/stages.schema.json"
)
_NEWAVE_CONFIG_INLINE_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/config.schema.json"
)
_NEWAVE_GENERIC_CONSTRAINTS_INLINE_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/generic_constraints.schema.json"
)


def _expected_schema_urls() -> dict[str, str]:
    return {
        "config.json": decomp_config._CONFIG_SCHEMA_URL,
        "stages.json": decomp_temporal._STAGES_SCHEMA_URL,
        "penalties.json": converters_network._PENALTIES_SCHEMA_URL,
        "initial_conditions.json": converters_initial_conditions._SCHEMA_URL,
        "system/buses.json": converters_network._BUSES_SCHEMA_URL,
        "system/lines.json": converters_network._LINES_SCHEMA_URL,
        "system/thermals.json": converters_thermal._SCHEMA_URL,
        "system/hydros.json": converters_hydro._SCHEMA_URL,
        "system/hydro_production_models.json": (
            converters_hydro._PRODUCTION_MODELS_SCHEMA_URL
        ),
        "system/non_controllable_sources.json": converters_network._NCS_SCHEMA_URL,
        "system/pumping_stations.json": decomp_network._PUMPING_SCHEMA_URL,
        "system/energy_contracts.json": decomp_contracts._SCHEMA_URL,
        "scenarios/load_factors.json": converters_stochastic._LOAD_FACTORS_SCHEMA_URL,
        "scenarios/non_controllable_factors.json": (
            converters_network._NCS_FACTORS_SCHEMA_URL
        ),
        "constraints/generic_constraints.json": converters_constraints._SCHEMA_URL,
        "constraints/generic_parameters.json": (
            converters_scalar_parameters._SCHEMA_URL
        ),
    }


def test_schema_urls_matches_live_constants() -> None:
    assert cobre_schemas.SCHEMA_URLS == _expected_schema_urls()


def test_schema_urls_has_sixteen_entries() -> None:
    assert len(cobre_schemas.SCHEMA_URLS) == 16


def test_schema_url_for_hit() -> None:
    assert (
        cobre_schemas.schema_url_for("system/buses.json")
        == converters_network._BUSES_SCHEMA_URL
    )


def test_schema_url_for_unregistered_raises_key_error() -> None:
    with pytest.raises(KeyError):
        cobre_schemas.schema_url_for("does/not/exist.json")


def test_config_json_pins_decomp_constant_and_newave_inline_value() -> None:
    assert cobre_schemas.SCHEMA_URLS["config.json"] == decomp_config._CONFIG_SCHEMA_URL
    assert cobre_schemas.SCHEMA_URLS["config.json"] == _NEWAVE_CONFIG_INLINE_URL


def test_stages_json_pins_decomp_constant_and_newave_inline_value() -> None:
    assert (
        cobre_schemas.SCHEMA_URLS["stages.json"] == decomp_temporal._STAGES_SCHEMA_URL
    )
    assert cobre_schemas.SCHEMA_URLS["stages.json"] == _NEWAVE_STAGES_INLINE_URL


def test_generic_constraints_json_pins_named_constant_and_newave_inline_value() -> None:
    relpath = "constraints/generic_constraints.json"
    assert cobre_schemas.SCHEMA_URLS[relpath] == converters_constraints._SCHEMA_URL
    assert cobre_schemas.SCHEMA_URLS[relpath] == _NEWAVE_GENERIC_CONSTRAINTS_INLINE_URL


def test_all_urls_have_canonical_shape() -> None:
    prefix = "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main/schemas/"
    for relpath, url in cobre_schemas.SCHEMA_URLS.items():
        assert url.startswith(prefix), relpath
        assert url.endswith(".schema.json"), relpath
