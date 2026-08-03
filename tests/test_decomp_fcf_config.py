"""Tests for the DECOMP config emitter's ``state_space`` section."""

from __future__ import annotations

from cobre_bridge.decomp.config import (
    _INFLOW_LAG_DEPTH,
    _SMOKE_FORWARD_PASSES,
    convert_config,
)


class _Ni:
    iteracoes = 250


class _Gp:
    data = [0.5]


class _Dt:
    dia, mes, ano = 18, 7, 2026


class _Dadger:
    ni = _Ni()
    gp = _Gp()
    dt = _Dt()


class TestConvertConfigStateSpace:
    def test_convert_config_emits_inflow_lag_depth_12(self) -> None:
        result = convert_config(_Dadger(), n_terminal_scenarios=259)  # type: ignore[arg-type]

        assert result["state_space"] == {"inflow_lag_depth": 12}
        assert _INFLOW_LAG_DEPTH == 12

    def test_convert_config_leaves_training_simulation_unchanged(self) -> None:
        result = convert_config(_Dadger(), n_terminal_scenarios=259)  # type: ignore[arg-type]

        expected_training = {
            "forward_passes": _SMOKE_FORWARD_PASSES,
            "stopping_rules": [{"type": "iteration_limit", "limit": 250}],
            "scenario_source": {
                "seed": 20260718,
                "inflow": {"scheme": "external"},
            },
        }
        expected_simulation = {
            "enabled": True,
            "num_scenarios": 259,
        }

        assert result["training"] == expected_training
        assert result["simulation"] == expected_simulation
