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
        result = convert_config(_Dadger())  # type: ignore[arg-type]

        assert result["state_space"] == {"inflow_lag_depth": 12}
        assert _INFLOW_LAG_DEPTH == 12

    def test_convert_config_training_and_simulation_enumerated(self) -> None:
        result = convert_config(_Dadger())  # type: ignore[arg-type]

        # Training enumerates the explicit trunk-plus-fan node graph; every
        # stochastic class is external — inflow (the tree), load, and NCS.
        # cobre's scheme-aware load membership admits a deterministic (std = 0)
        # external load class (it standardizes to eta = 0). The seed is
        # schema-required for external schemes (inert at run time here).
        expected_training = {
            "selection": {"method": "enumerated"},
            "stopping_rules": [
                {"type": "gap", "relative_tolerance": 0.5},
                {"type": "iteration_limit", "limit": 250},
            ],
            "scenario_source": {
                "seed": 20260718,
                "inflow": {"scheme": "external"},
                "load": {"scheme": "external"},
                "ncs": {"scheme": "external"},
            },
        }
        # Simulation is the exact weighted census over the branching graph
        # (cobre 0.14+ wires it; the old C9 sampled fallback is retired). With
        # no simulation.scenario_source, cobre inherits training's external one.
        expected_simulation = {
            "enabled": True,
            "selection": {"method": "enumerated"},
        }

        assert result["training"] == expected_training
        assert result["simulation"] == expected_simulation
