"""Canonical NEWAVE study-horizon arithmetic — the single source of truth.

Every per-stage table the pipeline emits (inflows, loads, bounds, penalties,
constraints) is sized against the study horizon, and the study/post-study
boundary drives post-study extrapolation. That arithmetic used to be hand-copied
at ~25 sites across the converters and comparators; this module owns it so all of
them agree by construction.

NEWAVE conventions encoded here:

- The study runs from ``mes_inicio_estudo`` of ``ano_inicio_estudo`` through
  December of ``ano_inicio_estudo + num_anos_estudo - 1`` — i.e.
  ``study_months = (13 - start_month) + (num_anos - 1) * 12`` stages.
- ``num_anos_pos_estudo`` full calendar years follow, adding ``num_anos_pos * 12``
  post-study stages, for ``total_stages`` in all.
- Seasonal records for the post-study (static final) period are tagged with the
  sentinel year ``9999`` (:data:`POST_STUDY_YEAR`).
- ``99990`` and above is NEWAVE's "big-M" sentinel meaning "no limit / restore
  default" in bound records (:data:`BIG_M`).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from inewave.newave import Dger

# NEWAVE tags post-study (static final period) seasonal data with year 9999.
POST_STUDY_YEAR = 9999

# NEWAVE bound records use 99999 as a "big-M" sentinel meaning "no limit".
# Compare with >= this threshold to catch the family of 9999x sentinels.
BIG_M = 99990.0


@dataclass(frozen=True)
class StudyHorizon:
    """Resolved study-horizon dimensions for a NEWAVE case.

    Build with :func:`study_horizon`; every field is a plain count/index so the
    object is cheap to pass around and compare.
    """

    start_year: int
    start_month: int
    num_anos: int
    """Number of study years (``num_anos_estudo``)."""
    num_anos_pos: int
    """Number of post-study years (``num_anos_pos_estudo``)."""
    study_months: int
    """Number of study stages."""
    total_stages: int
    """Number of stages overall (study + post-study)."""

    @property
    def last_study_stage(self) -> int:
        """0-based index of the final study stage (the freeze point)."""
        return self.study_months - 1

    @property
    def first_year_stages(self) -> int:
        """Number of stages in the (partial) first calendar year."""
        return 13 - self.start_month

    @property
    def pos_months(self) -> int:
        """Number of post-study stages."""
        return self.num_anos_pos * 12

    def is_post_study(self, stage_id: int) -> bool:
        """True if ``stage_id`` (0-based) falls in the post-study tail."""
        return stage_id >= self.study_months


def study_horizon(dger: Dger) -> StudyHorizon:
    """Resolve the :class:`StudyHorizon` from a parsed ``dger.dat``.

    ``num_anos_estudo`` falls back to 1 and ``num_anos_pos_estudo`` to 0 when
    absent/zero (the pipeline rejects a genuinely empty study earlier).
    """
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)
    num_anos = int(dger.num_anos_estudo or 1)
    num_anos_pos = int(dger.num_anos_pos_estudo or 0)
    study_months = (13 - start_month) + (num_anos - 1) * 12
    total_stages = study_months + num_anos_pos * 12
    return StudyHorizon(
        start_year=start_year,
        start_month=start_month,
        num_anos=num_anos,
        num_anos_pos=num_anos_pos,
        study_months=study_months,
        total_stages=total_stages,
    )


def build_stage_dates(
    start_year: int, start_month: int, total_stages: int
) -> list[date]:
    """Return the first-of-month date of each stage, in order."""
    stages: list[date] = []
    year, month = start_year, start_month
    for _ in range(total_stages):
        stages.append(date(year, month, 1))
        month += 1
        if month > 12:
            month = 1
            year += 1
    return stages


def stage_dates_for(horizon: StudyHorizon) -> list[date]:
    """Convenience: :func:`build_stage_dates` for a resolved horizon."""
    return build_stage_dates(
        horizon.start_year, horizon.start_month, horizon.total_stages
    )
