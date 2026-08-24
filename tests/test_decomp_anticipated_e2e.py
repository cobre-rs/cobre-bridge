"""Tier-3 harness: train + simulate the reduced deck on the installed cobre wheel.

Needs the ``cobre-python`` wheel's ``write_policy_checkpoint`` writer binding
(a version floor for the anticipated post-horizon emission this harness
exercises) and the real, gitignored reduced deck under ``example/``; every
test SKIPs cleanly (never fails, never runs in CI) when either is absent.
``_convert_and_run`` authors a boundary-enabled case and drives a bounded
in-process train+simulate; a later faithfulness gate imports and calls it
directly rather than duplicating the harness.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from tests.conftest import requires_cobre_python, requires_writer_binding

_LOG = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DECK = _REPO_ROOT / "example" / "decomp-mar-26-rv2-reduced"

# The reduced deck's own two-scenario fan converges within a handful of SDDP
# iterations, so this bound keeps the harness fast while still exercising a
# real train pass rather than a degenerate single-iteration stub.
_STOPPING_ITERATION_LIMIT = 5

_SANTA_CRUZ_THERMAL_ID = 94
_PSERGIPE_I_THERMAL_ID = 95
_GNL_THERMAL_IDS = (_SANTA_CRUZ_THERMAL_ID, _PSERGIPE_I_THERMAL_ID)

# The reduced deck's study horizon ends 2026-05-01 (`stages.json`'s last
# stage), cumulative 1152 study hours; the emitted class-4 (já-comandada,
# fixed)/class-3 (signaled, re-derived) boundary is 2026-05-16.
_HORIZON_END = date(2026, 5, 1)
_CLASS4_CLASS3_BOUNDARY = date(2026, 5, 16)
_STUDY_CUMULATIVE_HOURS = 1152
_POST_STUDY_MONTH_ANCHOR = 20260501


def _convert_and_run(tmp_path: Path) -> tuple[Path, Path]:
    """Author a boundary-enabled case from ``_DECK`` and train+simulate it.

    A bare ``convert_decomp_case`` call authors no boundary checkpoint — the
    boundary-FCF import is a separate step the CLI wires in after the
    pipeline, so this drives the real ``convert decomp`` command (as a
    subprocess) rather than the library function directly. The run below
    then points ``output_dir`` at the same *case_dir* (the recipe cobre
    resolves ``policy.boundary.path`` against), so the trained policy
    actually loads the boundary rather than starting cold.

    Returns ``(case_dir, case_dir)`` — the training/simulation partitions
    this run writes land under the same directory the case was converted
    into, so a later faithfulness gate reuses this exact harness instead of
    duplicating the convert+run pair. A run-side reject (boundary rejected,
    straddle, missing class-3 carrier) propagates as ``cobre.run.run``'s own
    exception; this helper never swallows it.
    """
    case_dir = tmp_path / "converted"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cobre_bridge.cli",
            "convert",
            "decomp",
            str(_DECK),
            str(case_dir),
            "--force",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"convert decomp exited {completed.returncode}:\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert (case_dir / "boundary" / "manifest.bin").is_file()

    import cobre

    cobre.run.run(
        str(case_dir),
        output_dir=str(case_dir),
        config_overrides={
            "training.stopping_rules": [
                {"type": "iteration_limit", "limit": _STOPPING_ITERATION_LIMIT}
            ]
        },
        on_iteration=lambda _info: True,
    )
    return case_dir, case_dir


@requires_cobre_python
@requires_writer_binding
@pytest.mark.skipif(not _DECK.exists(), reason=f"reduced deck ({_DECK}) not present")
def test_convert_and_run_trains_simulates_and_materialises_anticipated_lanes(
    tmp_path: Path,
) -> None:
    """The harness authors a boundary-enabled case, trains, and simulates it
    without raising (no boundary reject), and the anticipated post-horizon
    lanes partition materialises with real data.

    Covers the converted case's own boundary checkpoint (``boundary/
    manifest.bin``, asserted inside ``_convert_and_run``) and anticipated
    post-horizon emission (``post_study_stages.json`` at the case root, a
    non-empty ``past_anticipated_commitments`` in ``initial_conditions.json``)
    plus the run output's ``simulation/anticipated_lanes/`` partition; the
    exact faithfulness values are asserted elsewhere, against this same
    harness.
    """
    case_dir, output_dir = _convert_and_run(tmp_path)

    assert (case_dir / "post_study_stages.json").is_file()
    initial_conditions = json.loads(
        (case_dir / "initial_conditions.json").read_text(encoding="utf-8")
    )
    assert initial_conditions["past_anticipated_commitments"]

    lanes_dir = output_dir / "simulation" / "anticipated_lanes"
    scenario_partitions = sorted(lanes_dir.glob("scenario_id=*"))
    assert scenario_partitions, f"no scenario_id=*/ partition under {lanes_dir}"
    data_files = [partition / "data.parquet" for partition in scenario_partitions]
    assert any(f.is_file() and f.stat().st_size > 0 for f in data_files), (
        f"no populated data.parquet under {lanes_dir}"
    )


@requires_cobre_python
@requires_writer_binding
@pytest.mark.skipif(not _DECK.exists(), reason=f"reduced deck ({_DECK}) not present")
def test_post_horizon_split_is_faithful_and_run_reaches_every_post_study_delivery(
    tmp_path: Path,
) -> None:
    """The emitted class-4/class-3 split matches the source model's own
    anticipated-dispatch declaration, and the trained run's anticipated
    ring resolves a genuine decision for every post-study delivery on both
    GNL plants.

    Ground truth for the emitted split is the source model's `relgnl.rv2`
    report ("Sinalização de Despacho antecipado"): SANTA CRUZ's Sem 9*
    já-comandada (500 MW, delivered 09/05) is the only fixed post-horizon
    commitment either GNL plant declares, and Sem 10 (16/05) is the source
    model's first signaled (class-3) week -- matching the emitted class-4/
    class-3 boundary asserted below. The run-side check stops at reaching
    every post-study delivery target (the ring is wired end-to-end): the
    source model's own solved horizon ends at `_HORIZON_END`, so it has no
    result for any post-study delivery to compare cobre's re-derived
    `deposited_decision_mw` against on this reduced, cheap-hydro-dominated
    case -- that numeric comparison needs a full, unreduced deck.
    """
    case_dir, run_output_dir = _convert_and_run(tmp_path)

    initial_conditions = json.loads(
        (case_dir / "initial_conditions.json").read_text(encoding="utf-8")
    )
    assert "future_anticipated_deliveries" not in initial_conditions

    windows_by_thermal: dict[int, list[dict[str, object]]] = {}
    for window in initial_conditions["past_anticipated_commitments"]:
        windows_by_thermal.setdefault(window["thermal_id"], []).append(window)
        start = date.fromisoformat(window["start_date"])
        end = date.fromisoformat(window["end_date"])
        assert not (start < _HORIZON_END < end), (
            f"window {window} straddles horizon_end {_HORIZON_END}"
        )

    santa_cruz_post_horizon = {
        (w["start_date"], w["end_date"]): w["value_mw"]
        for w in windows_by_thermal[_SANTA_CRUZ_THERMAL_ID]
        if date.fromisoformat(w["start_date"]) >= _HORIZON_END
    }
    assert santa_cruz_post_horizon == {
        ("2026-05-01", "2026-05-02"): 0.0,
        ("2026-05-02", "2026-05-09"): 0.0,
        ("2026-05-09", "2026-05-16"): 500.0,
    }

    psergipe_post_horizon = [
        w
        for w in windows_by_thermal[_PSERGIPE_I_THERMAL_ID]
        if date.fromisoformat(w["start_date"]) >= _HORIZON_END
    ]
    assert psergipe_post_horizon
    assert all(w["value_mw"] == 0.0 for w in psergipe_post_horizon)

    post_study_stages = json.loads(
        (case_dir / "post_study_stages.json").read_text(encoding="utf-8")
    )
    stage_dates = [
        date.fromisoformat(s["start_date"]) for s in post_study_stages["stages"]
    ]
    thermal_bounds = post_study_stages["thermal_bounds"]
    assert thermal_bounds, (
        f"no thermal_bounds rows in {case_dir}/post_study_stages.json"
    )
    for bound in thermal_bounds:
        stage_date = stage_dates[bound["post_study_stage_index"]]
        assert stage_date >= _CLASS4_CLASS3_BOUNDARY, (
            f"thermal_bounds row {bound} is before the class-4/class-3 "
            f"boundary ({_CLASS4_CLASS3_BOUNDARY})"
        )

    thermals = json.loads(
        (case_dir / "system" / "thermals.json").read_text(encoding="utf-8")
    )["thermals"]
    thermals_by_id = {t["id"]: t for t in thermals}
    for thermal_id in _GNL_THERMAL_IDS:
        lead_hours = thermals_by_id[thermal_id]["anticipated_config"]["lead_time_hours"]
        assert math.isfinite(lead_hours)
        assert lead_hours > _STUDY_CUMULATIVE_HOURS, (
            f"thermal_id={thermal_id} lead_time_hours={lead_hours} does not "
            f"exceed the study's {_STUDY_CUMULATIVE_HOURS} cumulative hours "
            "-- the C13 cap may not be lifted"
        )

    lanes_dir = run_output_dir / "simulation" / "anticipated_lanes"
    partitions = sorted(lanes_dir.glob("scenario_id=*/data.parquet"))
    assert partitions, f"no anticipated_lanes partition under {lanes_dir}"
    lanes = pl.concat([pl.read_parquet(partition) for partition in partitions])
    assert lanes.height > 0, f"anticipated_lanes partition is empty under {lanes_dir}"

    for thermal_id in _GNL_THERMAL_IDS:
        rows = lanes.filter(pl.col("thermal_id") == thermal_id)
        assert rows.height > 0, (
            f"no anticipated_lanes row for thermal_id={thermal_id} under {lanes_dir}"
        )
        min_anchor = rows["delivery_date"].min()
        assert min_anchor >= _POST_STUDY_MONTH_ANCHOR, (
            f"thermal_id={thermal_id} anticipated_lanes delivery_date anchor "
            f"{min_anchor} is before the post-study window "
            f"{_POST_STUDY_MONTH_ANCHOR} -- the uncapped lead may have "
            "resolved to a different axis than the class-4/class-3 split"
        )
        _LOG.info(
            "thermal_id=%s deposited_decision_mw=%s",
            thermal_id,
            rows["deposited_decision_mw"].to_list(),
        )
