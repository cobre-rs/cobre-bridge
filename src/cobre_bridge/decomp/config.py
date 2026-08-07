"""Run configuration and penalties for DECOMP-like decks.

``config.json`` carries the fixed ``state_space.inflow_lag_depth`` (P3/D8),
the deck's ``GP`` convergence criterion as a **relative** ``Gap`` stopping
rule (``relative_tolerance``, matching DECOMP's ``Zsup/Zinf - 1 <= GP``
convergence; cobre auto-injects a ``BoundStalling`` companion so an
unattainable tolerance degrades to a diagnosed stall), the deck-faithful
``NI`` iteration backstop, and the external scenario schemes.

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

# D8: the boundary cut's deepest lag term sets the state-space depth cobre
# must reserve, so the emitted config declares it explicitly rather than
# leaving cobre to infer one. The source model's boundary cuts carry
# ``pi_qafl`` lag coefficients out to depth 12, so a smaller value would
# drop lag coefficients cobre would then reject at load.
_INFLOW_LAG_DEPTH = 12


def convert_config(dadger: Dadger, n_terminal_scenarios: int) -> dict:
    """Build ``config.json``: state space depth, Gap + NI stopping rules,
    external scenario sources, simulation on.

    Training uses ``selection = {"method": "enumerated"}``: the explicit
    trunk-plus-fan node graph enumerates every root-to-leaf path, so the
    forward-pass count is derived from the graph. Simulation uses ``sampled``
    with ``n_terminal_scenarios`` as a TRACKED COBRE-GAP WORKAROUND (C9):
    cobre 0.14 cannot yet execute enumerated (weighted-census) simulation over
    a *branching* graph — only a single-realization tree — so an enumerated
    simulation aborts. Sampling the fan width approximates the census until
    cobre wires branching-census simulation (epic-06/epic-14).

    ``state_space.inflow_lag_depth`` is fixed at 12 (P3/D8): under
    no-folding, the source model's boundary cuts carry lag coefficients out
    to depth 12, so cobre's bookkeeping must reserve that many lag slots for
    the terminal boundary cut to price.
    """
    ni = int(dadger.ni.iteracoes or 500)
    gp = float(dadger.gp.data[0])
    _LOG.info(
        "emitting the deck's GP=%g as a relative Gap stopping rule "
        "(relative_tolerance, DECOMP's Zsup/Zinf-1 <= GP convergence); cobre "
        "auto-injects a BoundStalling companion, with the NI=%d iteration backstop",
        gp,
        ni,
    )

    dt = dadger.dt
    seed = int(dt.ano) * 10000 + int(dt.mes) * 100 + int(dt.dia)

    _LOG.warning(
        "TRACKED COBRE-GAP WORKAROUND (C9): cobre 0.14 executes enumerated "
        "selection only on a single-realization tree; a branching graph's "
        "weighted-census SIMULATION is not wired, so simulation falls back to "
        "sampled selection over the %d terminal-fan scenarios "
        "(~/git/cobre/plans/conversion-found-improvements.md). Training stays "
        "enumerated (it runs the full census). Remove when cobre wires "
        "branching-census simulation (epic-06/epic-14).",
        n_terminal_scenarios,
    )

    return {
        "$schema": _CONFIG_SCHEMA_URL,
        "state_space": {"inflow_lag_depth": _INFLOW_LAG_DEPTH},
        "training": {
            "selection": {"method": "enumerated"},
            "stopping_rules": [
                {"type": "gap", "relative_tolerance": gp},
                {"type": "iteration_limit", "limit": ni},
            ],
            # Under the node-native explicit tree every stochastic class is
            # external: inflow (the tree), NCS (renewables), and load. cobre's
            # scheme-aware load membership admits an external load class
            # regardless of σ (a deterministic std = 0 load standardizes to
            # eta = 0), so load is external here rather than the former
            # in-sample-with-null-std workaround.
            "scenario_source": {
                "seed": seed,
                "inflow": {"scheme": "external"},
                "load": {"scheme": "external"},
                "ncs": {"scheme": "external"},
            },
        },
        "simulation": {
            "enabled": True,
            # C9 workaround: sampled (not enumerated) until cobre wires
            # branching-census simulation — see the module note above.
            "selection": {
                "method": "sampled",
                "num_scenarios": n_terminal_scenarios,
            },
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
