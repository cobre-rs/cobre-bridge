"""Tab registry for the dashboard.

Defines the ``TabModule`` Protocol, the ``TAB_MODULES`` registry, and the
``get_renderable_tabs`` orchestration function.  Tab modules added in tickets
010-014 append themselves to ``TAB_MODULES`` at import time.
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
    "tab-overview": 0,
    "tab-training": 10,
    "tab-energy": 20,
    "tab-hydro": 30,
    "tab-plants": 40,
    "tab-thermal": 50,
    "tab-thermal-plants": 60,
    "tab-exchanges": 70,
    "tab-costs": 80,
    "tab-ncs-thermal": 90,
    "tab-constraints": 100,
    "tab-stochastic": 110,
    "tab-perf": 120,
}

# ---------------------------------------------------------------------------
# Registry — populated by tickets 010-014
# ---------------------------------------------------------------------------

from cobre_bridge.dashboard.tabs import (  # noqa: E402
    constraints,
    costs,
    energy,
    exchanges,
    hydro,
    ncs,
    overview,
    performance,
    plants,
    stochastic,
    thermal,
    thermal_plants,
    training,
)

TAB_MODULES: list[TabModule] = [
    overview,
    training,
    energy,
    hydro,
    plants,
    thermal,
    thermal_plants,
    exchanges,
    costs,
    ncs,
    constraints,
    stochastic,
    performance,
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
