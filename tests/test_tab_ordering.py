"""Tests for tab ordering matching wireframe specification."""

from __future__ import annotations

from cobre_bridge.dashboard.tabs import DEFAULT_TAB_ORDER
from cobre_bridge.dashboard.tabs import v2_stochastic, v2_training


def test_stochastic_before_training() -> None:
    assert v2_stochastic.TAB_ORDER < v2_training.TAB_ORDER


def test_default_tab_order_stochastic() -> None:
    assert DEFAULT_TAB_ORDER["tab-v2-stochastic"] == 10


def test_default_tab_order_training() -> None:
    assert DEFAULT_TAB_ORDER["tab-v2-training"] == 20
