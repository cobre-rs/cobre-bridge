"""Tests for the canonical study-horizon module."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from cobre_bridge.horizon import (
    BIG_M,
    POST_STUDY_YEAR,
    build_stage_dates,
    stage_dates_for,
    study_horizon,
)


def _dger(start_year=2024, start_month=9, num_anos=3, num_anos_pos=3):
    return SimpleNamespace(
        ano_inicio_estudo=start_year,
        mes_inicio_estudo=start_month,
        num_anos_estudo=num_anos,
        num_anos_pos_estudo=num_anos_pos,
    )


def test_sentinels():
    assert POST_STUDY_YEAR == 9999
    assert BIG_M == 99990.0


def test_study_horizon_example_case():
    # Sept-start, 3 study years, 3 post-study years (the example case shape).
    h = study_horizon(_dger())
    assert h.start_year == 2024
    assert h.start_month == 9
    assert h.study_months == (13 - 9) + (3 - 1) * 12  # 28
    assert h.total_stages == 28 + 3 * 12  # 64
    assert h.last_study_stage == 27
    assert h.first_year_stages == 4
    assert h.pos_months == 36


def test_january_start_has_full_first_year():
    h = study_horizon(_dger(start_month=1, num_anos=5, num_anos_pos=0))
    assert h.study_months == 12 + (5 - 1) * 12  # 60
    assert h.total_stages == 60
    assert h.first_year_stages == 12


@pytest.mark.parametrize(
    "num_anos,num_anos_pos",
    [(None, None), (0, 0)],
)
def test_falsy_year_counts_default(num_anos, num_anos_pos):
    h = study_horizon(_dger(num_anos=num_anos, num_anos_pos=num_anos_pos))
    assert h.num_anos == 1  # falls back to one study year
    assert h.num_anos_pos == 0
    assert h.study_months == 13 - 9
    assert h.total_stages == h.study_months


def test_is_post_study_boundary():
    h = study_horizon(_dger())  # study_months == 28
    assert not h.is_post_study(27)  # last study stage
    assert h.is_post_study(28)  # first post-study stage
    assert h.is_post_study(63)


def test_build_stage_dates_wraps_year():
    dates = build_stage_dates(2024, 11, 4)
    assert dates == [
        date(2024, 11, 1),
        date(2024, 12, 1),
        date(2025, 1, 1),
        date(2025, 2, 1),
    ]


def test_stage_dates_for_matches_horizon_length():
    h = study_horizon(_dger())
    dates = stage_dates_for(h)
    assert len(dates) == h.total_stages
    assert dates[0] == date(2024, 9, 1)
