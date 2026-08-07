"""Tests for the cadastro-override calendar resolver.

Pins the ``(mes, semana, ano)`` -> stage-index resolution rules of
``cobre_bridge.decomp.cadastro.resolve_effective_stage`` against a synthetic
two-month operative calendar (two July weekly stages + one August monthly
stage), mirroring the calendar shape the source model's operative weeks and
months produce.
"""

from __future__ import annotations

from datetime import date

import pytest

from cobre_bridge.decomp.cadastro import resolve_effective_stage
from cobre_bridge.decomp.temporal import OperativeStage, build_operative_calendar


@pytest.fixture
def calendar() -> list[OperativeStage]:
    """Stages 0, 1 = July weekly stages; stage 2 = the August monthly stage."""
    stage_block_hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), stage_block_hours)


def test_blank_month_is_stage_zero(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(None, None, None, calendar) == 0


def test_semana_one_is_first_july_stage(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(7, 1, 2026, calendar) == 0


def test_semana_two_is_second_july_stage(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(7, 2, 2026, calendar) == 1


def test_blank_semana_is_monthly_stage(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(8, None, 2026, calendar) == 2


def test_semana_clamps_to_monthly_stage(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(8, 3, 2026, calendar) == 2


def test_month_before_horizon_is_stage_zero(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(6, None, 2026, calendar) == 0


def test_month_after_horizon_is_none(calendar: list[OperativeStage]) -> None:
    assert resolve_effective_stage(10, None, 2026, calendar) is None
