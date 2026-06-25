"""Unit tests for the stage-math and ζ pure helpers in ``cobre_bridge.filling``."""

from __future__ import annotations

from cobre_bridge.filling import (
    filling_min_rate_m3s,
    online_machines,
    stage_id,
    zeta,
)


def test_stage_id_study_start_is_zero() -> None:
    assert stage_id(2024, 9, 2024, 9) == 0


def test_stage_id_jan_2025_is_four() -> None:
    assert stage_id(2025, 1, 2024, 9) == 4


def test_stage_id_pre_start_is_negative() -> None:
    # No clamping in this helper: dates before the study start go negative.
    assert stage_id(2024, 8, 2024, 9) == -1


def test_zeta_october_31_days() -> None:
    # October has 31 days -> 744 h -> 744 * 3600 / 1e6 == 2.6784 (design §5).
    assert abs(zeta(2024, 10) - 2.6784) < 1e-9


def test_zeta_leap_february() -> None:
    # 2024 is a leap year: February has 29 days -> 696 h -> 2.5056.
    assert abs(zeta(2024, 2) - 2.5056) < 1e-9


def test_zeta_non_leap_february() -> None:
    # 2023 is not a leap year: February has 28 days -> 672 h -> 2.4192.
    assert abs(zeta(2023, 2) - 2.4192) < 1e-9


def test_fill_rate_juruena_single_stage() -> None:
    # JURUENA single-stage window [1, 2) = Oct 2024: 2.93 / 2.6784 ≈ 1.09 (§5).
    result = filling_min_rate_m3s(2.93, 0.0, 1, 2, lambda t: (2024, 10))
    assert abs(result - 2.93 / 2.6784) < 1e-6


def test_fill_rate_empty_window_is_zero() -> None:
    # Empty window (start >= entry): "no Filling phase" signal, not ZeroDivision.
    assert filling_min_rate_m3s(2.93, 0.0, 2, 2, lambda t: (2024, 10)) == 0.0


def test_fill_rate_nonzero_volume_morto() -> None:
    # 50% dead volume halves remaining: (2.93 * 0.5) / 2.6784.
    result = filling_min_rate_m3s(2.93, 50.0, 1, 2, lambda t: (2024, 10))
    assert abs(result - (2.93 * 0.5) / 2.6784) < 1e-6


def test_online_machines_none_before_unit_entry() -> None:
    # Both units enter stage 4; querying stage 3 -> no group online.
    assert online_machines([(1, 4), (1, 4)], 2, 3) == {}


def test_online_machines_full_count_at_entry() -> None:
    # At stage 4 both conjunto-1 units are online.
    assert online_machines([(1, 4), (1, 4)], 2, 4) == {1: 2}


def test_online_machines_clamps_to_entry_stage() -> None:
    # Unit nominally enters stage 1 but is clamped up to entry_stage_id=2,
    # so it is not online when querying stage 1.
    assert online_machines([(1, 1)], 2, 1) == {}


def test_online_machines_partial_groups() -> None:
    # Two conjuntos entering different stages; query stage 5 -> only group 1.
    assert online_machines([(1, 5), (2, 7)], 2, 5) == {1: 1}
