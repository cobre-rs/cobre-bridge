"""Tests for the DECOMP config emitter's ``state_space`` section."""

from __future__ import annotations

from cobre_bridge.decomp.config import (
    _INFLOW_LAG_DEPTH,
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

    def test_convert_config_training_enumerated_simulation_sampled(self) -> None:
        result = convert_config(_Dadger(), n_terminal_scenarios=259)  # type: ignore[arg-type]

        # Training enumerates the explicit trunk-plus-fan node graph; NCS is
        # external (32 canonical entities), inflow external, load stays
        # in-sample (deterministic, zero-entity, exempt).
        expected_training = {
            "selection": {"method": "enumerated"},
            "stopping_rules": [{"type": "iteration_limit", "limit": 250}],
            "scenario_source": {
                "seed": 20260718,
                "inflow": {"scheme": "external"},
                "ncs": {"scheme": "external"},
            },
        }
        # Simulation is sampled (C10 workaround) until cobre wires
        # branching-census simulation.
        expected_simulation = {
            "enabled": True,
            "selection": {"method": "sampled", "num_scenarios": 259},
        }

        assert result["training"] == expected_training
        assert result["simulation"] == expected_simulation
