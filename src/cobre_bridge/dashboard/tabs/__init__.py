"""Tab registry for the dashboard.

Defines the ``TabModule`` Protocol, the ``TAB_MODULES`` registry, and the
``get_renderable_tabs`` orchestration function.  The registry exclusively
uses v2 lifecycle-driven tab modules (epic-03 through epic-07).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from cobre_bridge.dashboard.data import DashboardData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class TabModule(Protocol):
    """Structural interface that every dashboard tab module must satisfy."""

    TAB_ID: str
    TAB_LABEL: str
    TAB_ORDER: int

    def can_render(self, data: DashboardData) -> bool:
        """Return True when this tab has sufficient data to be shown."""
        ...

    def render(self, data: DashboardData) -> str:
        """Return the full HTML string for the tab content area."""
        ...


# ---------------------------------------------------------------------------
# Ordering reference
# ---------------------------------------------------------------------------

DEFAULT_TAB_ORDER: dict[str, int] = {
    "tab-v2-overview": 0,
    "tab-v2-stochastic": 10,
    "tab-v2-training": 20,
    "tab-v2-energy-balance": 30,
    "tab-v2-costs": 40,
    "tab-v2-plants": 50,
    "tab-v2-network": 60,
    "tab-v2-constraints": 80,
    "tab-v2-performance": 90,
}

# ---------------------------------------------------------------------------
# Registry — v2 lifecycle-driven tab modules (epic-03 through epic-07)
# ---------------------------------------------------------------------------

from cobre_bridge.dashboard.tabs import (  # noqa: E402
    v2_constraints,
    v2_costs,
    v2_energy_balance,
    v2_network,
    v2_overview,
    v2_performance,
    v2_plants,
    v2_stochastic,
    v2_training,
)

TAB_MODULES: list[TabModule] = [
    v2_overview,
    v2_training,
    v2_stochastic,
    v2_energy_balance,
    v2_costs,
    v2_plants,
    v2_network,
    v2_constraints,
    v2_performance,
]

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def get_renderable_tabs(data: DashboardData) -> list[tuple[str, str, str]]:
    """Return ordered, renderable tabs for *data*.

    Steps:
    1. Sort ``TAB_MODULES`` by ``TAB_ORDER`` (ascending).
    2. Filter to modules where ``can_render(data)`` is True.
    3. Call ``render(data)`` on each; skip and log on any exception.

    Returns:
        List of ``(tab_id, tab_label, rendered_html)`` tuples.
    """
    result: list[tuple[str, str, str]] = []
    ordered = sorted(TAB_MODULES, key=lambda m: m.TAB_ORDER)
    for module in ordered:
        if not module.can_render(data):
            continue
        try:
            html = module.render(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Tab %r (%s) raised an exception during render and will be skipped: %s",
                module.TAB_ID,
                type(module).__name__,
                exc,
            )
            continue
        result.append((module.TAB_ID, module.TAB_LABEL, html))
    return result
