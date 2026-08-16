"""Run configuration and penalties for DECOMP-like decks.

``config.json`` carries the deck's ``GP`` convergence criterion as a
**relative** ``Gap`` stopping rule (``relative_tolerance``, matching DECOMP's
``Zsup/Zinf - 1 <= GP`` convergence; cobre auto-injects a ``BoundStalling``
companion so an unattainable tolerance degrades to a diagnosed stall), the
deck-faithful ``NI`` iteration backstop, and the external scenario schemes.

No ``state_space.inflow_lag_depth`` is emitted. Under a deferred boundary FCF
the external white-noise inflow model contributes no inflow-lag state, so cobre
resolves a zero depth and reserving lag slots would be dead state (and would
raise cobre's lag-blind-stage advisory for nothing). The inflow-lag depth is a
property of the *boundary policy*: the boundary-FCF importer
(``fcf/__init__.py``) reserves exactly the depth the loaded cuts reference — and
only when a boundary policy is actually imported. cobre is slated to infer that
depth from the checkpoint itself, retiring even the importer's patch (see
``~/git/cobre/plans/state-space-inflow-lag-depth-inference-spec.md``).

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


def convert_config(dadger: Dadger, *, cvar_active: bool = False) -> dict:
    """Build ``config.json``: Gap + NI stopping rules, external scenario
    sources, simulation on.

    When ``cvar_active`` (the deck runs a CVaR risk measure, see
    :func:`cobre_bridge.decomp.temporal.resolve_cvar`), the ``gap`` stopping
    rule is dropped in favour of the ``NI`` iteration limit alone: cobre
    rejects a gap rule under any non-expectation risk measure (the risk-adjusted
    lower bound and the risk-neutral simulation upper bound do not bracket, so a
    persistent negative "gap" is expected, not convergence).

    Both training and simulation use ``selection = {"method": "enumerated"}``:
    the explicit trunk-plus-fan node graph enumerates every root-to-leaf path,
    so training runs the full forward/backward census and simulation runs the
    exact per-node-probability weighted census (cobre 0.14+ wires the
    branching-graph census simulation, retiring the earlier ``sampled`` fallback
    tracked as C9). The simulation omits its own ``scenario_source`` and inherits
    training's external one.

    ``scenario_source.seed`` is a fixed ``0``. A seed controls random sampling;
    with every class ``external`` (a deterministic replay of the explicit tree)
    and ``enumerated`` selection nothing samples, so the value is inert — but
    cobre's schema requires the field whenever any class is external, so a
    constant placeholder is emitted rather than a study-varying value that would
    misleadingly imply a meaningful random draw.

    No ``state_space`` block is emitted: the inflow-lag depth is a property of
    the boundary policy, not the case inputs. With the boundary FCF deferred the
    external inflow model needs no lag state, so cobre resolves a zero depth; the
    boundary-FCF importer reserves the cut-derived depth when a boundary is
    actually imported (see the module docstring).
    """
    ni = int(dadger.ni.iteracoes or 500)
    gp = float(dadger.gp.data[0])
    if cvar_active:
        _LOG.info(
            "CVaR risk measure active; the Gap stopping rule is inadmissible "
            "(cobre rejects it under a non-expectation risk measure), so "
            "converging on a lower-bound stall (tolerance=GP=%g) with the deck's "
            "NI=%d iteration backstop",
            gp,
            ni,
        )
        # A gap rule is inadmissible under CVaR (the risk-adjusted lower bound
        # and the risk-neutral upper bound do not bracket), but a bound-stalling
        # rule is: it watches the lower bound, which does converge under CVaR.
        # Reuse the deck's GP as the relative stall tolerance so a CVaR case
        # still stops on convergence rather than always grinding to NI.
        stopping_rules = [
            {"type": "bound_stalling", "iterations": 10, "tolerance": gp},
            {"type": "iteration_limit", "limit": ni},
        ]
    else:
        _LOG.info(
            "emitting the deck's GP=%g as a relative Gap stopping rule "
            "(relative_tolerance, DECOMP's Zsup/Zinf-1 <= GP convergence); cobre "
            "auto-injects a BoundStalling companion, with the NI=%d iteration backstop",
            gp,
            ni,
        )
        stopping_rules = [
            {"type": "gap", "relative_tolerance": gp},
            {"type": "iteration_limit", "limit": ni},
        ]

    return {
        "$schema": _CONFIG_SCHEMA_URL,
        "training": {
            "selection": {"method": "enumerated"},
            "stopping_rules": stopping_rules,
            # Under the node-native explicit tree every stochastic class is
            # external: inflow (the tree), NCS (renewables), and load. cobre's
            # scheme-aware load membership admits an external load class
            # regardless of σ (a deterministic std = 0 load standardizes to
            # eta = 0), so load is external here rather than the former
            # in-sample-with-null-std workaround. seed is a schema-required
            # inert placeholder (0) — external + enumerated never samples.
            "scenario_source": {
                "seed": 0,
                "inflow": {"scheme": "external"},
                "load": {"scheme": "external"},
                "ncs": {"scheme": "external"},
            },
        },
        "simulation": {
            "enabled": True,
            "selection": {"method": "enumerated"},
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
