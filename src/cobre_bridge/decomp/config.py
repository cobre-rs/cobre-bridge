"""Run configuration and penalties for DECOMP-like decks.

``config.json`` carries the fixed ``state_space.inflow_lag_depth`` (P3/D8),
the deck-faithful iteration backstop (``NI``), and the external inflow
scheme. The deck's ``GP`` gap criterion has no solver counterpart until the
enumeration work lands its absolute-gap rule; it is logged at conversion
(canonical value = GP × 1000 R$) so the omission is visible, never silent.

``penalties.json`` reuses the shared ρ-scaled hydro penalty construction
with the deck's deficit cost and the converted productivities — the same
formulas the other converter family applies, so the two case families
share one penalty convention.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cobre_bridge.converters.network import (
    _PCORTEOL,
    _PENALTIES_SCHEMA_URL,
    _PEXC,
    _PINT,
    _hydro_penalty_costs,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from idecomp.decomp import Dadger

_LOG = logging.getLogger(__name__)

_CONFIG_SCHEMA_URL = (
    "https://raw.githubusercontent.com/cobre-rs/cobre/refs/heads/main"
    "/schemas/config.schema.json"
)

# Trajectories per iteration for the smoke configuration: the trunk is
# deterministic, so forwards only choose terminal-fan members; full
# enumeration replaces this when the selection mode exists.
_SMOKE_FORWARD_PASSES = 5

# D8: the boundary cut's deepest lag term sets the state-space depth cobre
# must reserve, so the emitted config declares it explicitly rather than
# leaving cobre to infer one. The source model's boundary cuts carry
# ``pi_qafl`` lag coefficients out to depth 12, so a smaller value would
# drop lag coefficients cobre would then reject at load.
_INFLOW_LAG_DEPTH = 12


def convert_config(dadger: Dadger, n_terminal_scenarios: int) -> dict:
    """Build ``config.json``: state space depth, NI backstop, external
    inflows, simulation on.

    ``state_space.inflow_lag_depth`` is fixed at 12 (P3/D8): under
    no-folding, the source model's boundary cuts carry lag coefficients out
    to depth 12, so cobre's bookkeeping must reserve that many lag slots for
    the terminal boundary cut to price.
    """
    ni = int(dadger.ni.iteracoes or 500)
    gp = float(dadger.gp.data[0])
    _LOG.warning(
        "the deck's GP gap criterion (%g, canonical %g R$) has no solver "
        "counterpart yet; emitting the NI iteration backstop (%d) only — "
        "the absolute-gap rule lands with the enumeration work",
        gp,
        gp * 1000.0,
        ni,
    )

    dt = dadger.dt
    seed = int(dt.ano) * 10000 + int(dt.mes) * 100 + int(dt.dia)

    return {
        "$schema": _CONFIG_SCHEMA_URL,
        "state_space": {"inflow_lag_depth": _INFLOW_LAG_DEPTH},
        "training": {
            "forward_passes": _SMOKE_FORWARD_PASSES,
            "stopping_rules": [{"type": "iteration_limit", "limit": ni}],
            "scenario_source": {
                "seed": seed,
                "inflow": {"scheme": "external"},
            },
        },
        "simulation": {
            "enabled": True,
            "num_scenarios": n_terminal_scenarios,
        },
    }


def convert_penalties(
    deficit_cost: float,
    productivities: Sequence[float],
) -> dict:
    """Build ``penalties.json`` from the deck's deficit cost.

    ``ρ_avg`` is the mean converted productivity over every operated plant
    (zeros included) and ``ρ_max`` the maximum — the same convention the
    shared hydro-penalty construction expects; there is no per-deck
    penalty file, so every hydro slot takes its deficit-derived default.
    """
    values = list(productivities)
    rho_avg = sum(values) / len(values) if values else 1.0
    rho_max = max(values) if values else rho_avg
    hydro_costs = _hydro_penalty_costs(
        rho_avg=rho_avg,
        rho_max_acum=rho_max,
        penalid_costs={},
        max_deficit_cost=deficit_cost,
    )
    return {
        "$schema": _PENALTIES_SCHEMA_URL,
        "bus": {
            "deficit_segments": [{"depth_mw": None, "cost": deficit_cost}],
            "excess_cost": _PEXC,
        },
        "hydro": hydro_costs,
        "line": {"exchange_cost": _PINT},
        "non_controllable_source": {"curtailment_cost": _PCORTEOL},
    }
