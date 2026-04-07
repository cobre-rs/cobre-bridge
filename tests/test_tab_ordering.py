"""Tests for tab ordering matching wireframe specification."""

from __future__ import annotations

from cobre_bridge.dashboard.tabs import DEFAULT_TAB_ORDER, stochastic, training


def test_stochastic_before_training() -> None:
    assert stochastic.TAB_ORDER < training.TAB_ORDER


def test_default_tab_order_stochastic() -> None:
    assert DEFAULT_TAB_ORDER["tab-stochastic"] == 10


def test_default_tab_order_training() -> None:
    assert DEFAULT_TAB_ORDER["tab-training"] == 20
