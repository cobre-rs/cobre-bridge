"""Tests for the scenario-tree unweighted-averages diagnostic (DASH-03).

Covers ``load_temporal_context``'s ``dashboard-unweighted-tree-averages``
emission/silence (tier-1, no ``cobre`` import, no ``example/`` deck) and its
surfacing through the ``dashboard`` CLI command's diagnostics sink.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from cobre_bridge import diagnostics as dx
from cobre_bridge.dashboard.data import load_temporal_context
from tests.conftest import hydro_with_group

_DIAG_CODE = "dashboard-unweighted-tree-averages"

#: Repo-internal references that must NEVER leak into this diagnostic's
#: summary/remediation: a pip-installed user has no repo checkout, no
#: converter vocabulary, and no file layout to resolve any of these against.
#: Mirrors ``test_remediation_has_no_repo_internal_references``
#: (tests/test_decomp_fcf_capability.py).
_REPO_INTERNAL_LEAKS = (
    ".py",
    ".json",
    "docs/",
    "plans/",
    "~/git",
    "feat/",
    "ticket-",
    "epic-",
    "policy_graph",
    "stages.json",
    "cobre_bridge",
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# ---------------------------------------------------------------------------
# Minimal fixture for load_temporal_context (tier-1 emission/silence tests)
# ---------------------------------------------------------------------------

#: A DECOMP-shaped node list: a trunk node feeding a two-way terminal fan
#: (mirrors ``decomp.temporal.build_node_graph``'s node shape).
_TREE_NODES: list[dict[str, Any]] = [
    {"id": 0, "stage_id": 0, "scenario_id": 0, "label": "trunk-0"},
    {"id": 1, "stage_id": 1, "scenario_id": 0, "label": "fan-0"},
    {"id": 2, "stage_id": 1, "scenario_id": 1, "label": "fan-1"},
]


def _minimal_case(tmp_path: Path, *, policy_graph: dict[str, Any] | None) -> Path:
    """The 3 files ``load_temporal_context`` reads, nothing else.

    *policy_graph*, when given, is written verbatim under ``stages.json``'s
    ``policy_graph`` key; omitted (``None``) reproduces the source-model
    track's shape, which carries no ``policy_graph`` key at all.
    """
    case_dir = tmp_path / "case"
    _write_json(case_dir / "config.json", {"discount_rate": 0.1})
    stages: dict[str, Any] = {
        "stages": [
            {"id": 0, "blocks": [{"id": 0, "hours": 730.0}]},
            {"id": 1, "blocks": [{"id": 0, "hours": 730.0}]},
        ]
    }
    if policy_graph is not None:
        stages["policy_graph"] = policy_graph
    _write_json(case_dir / "stages.json", stages)
    _write_json(case_dir / "system" / "lines.json", {"lines": []})
    return case_dir


def _tree_policy_graph() -> dict[str, Any]:
    return {
        "type": "finite_horizon",
        "annual_discount_rate": 0.1,
        "nodes": _TREE_NODES,
        "transitions": [
            {"source_id": 0, "target_id": 1, "probability": 0.5},
            {"source_id": 0, "target_id": 2, "probability": 0.5},
        ],
    }


class TestLoadTemporalContextAbsentLinesJson:
    """ticket-029: ``lines.json`` degrades to an empty line set, not a crash.

    Before ticket-029 routed this read through
    ``cobre_readers.read_cobre_lines``, an absent ``system/lines.json``
    crashed ``load_temporal_context`` with an unguarded ``json.load`` on a
    missing path -- the one sanctioned behaviour change the reader-failure
    contract makes (an input that previously had no valid output)."""

    def test_absent_lines_json_yields_empty_line_meta(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        _write_json(case_dir / "config.json", {"discount_rate": 0.1})
        _write_json(
            case_dir / "stages.json",
            {"stages": [{"id": 0, "blocks": [{"id": 0, "hours": 730.0}]}]},
        )

        context = load_temporal_context(case_dir)

        assert context.line_meta == []


class TestTreeAveragesDiagnosticEmission:
    """AC #1/#2: emit on a non-empty ``nodes`` list; silent otherwise."""

    def test_emits_on_nonempty_nodes(self, tmp_path: Path) -> None:
        case_dir = _minimal_case(tmp_path, policy_graph=_tree_policy_graph())

        with dx.collect() as collected:
            load_temporal_context(case_dir)

        matches = [d for d in collected if d.code == _DIAG_CODE]
        assert len(matches) == 1
        assert matches[0].severity is dx.Severity.WARNING
        assert matches[0].category == "Dashboard data"

    def test_silent_when_no_policy_graph_key(self, tmp_path: Path) -> None:
        case_dir = _minimal_case(tmp_path, policy_graph=None)

        with dx.collect() as collected:
            load_temporal_context(case_dir)

        assert not [d for d in collected if d.code == _DIAG_CODE]

    def test_silent_when_policy_graph_has_no_nodes_key(self, tmp_path: Path) -> None:
        """The source-model track's ``transitions``-only shape must not
        false-positive on a ``transitions``-based check (it has none — this
        pins the marker to ``nodes`` presence, not ``transitions``)."""
        policy_graph = {
            "type": "linear",
            "annual_discount_rate": 0.1,
            "transitions": [{"source_id": 0, "target_id": 1, "probability": 1.0}],
        }
        case_dir = _minimal_case(tmp_path, policy_graph=policy_graph)

        with dx.collect() as collected:
            load_temporal_context(case_dir)

        assert not [d for d in collected if d.code == _DIAG_CODE]

    def test_silent_when_nodes_is_empty_list(self, tmp_path: Path) -> None:
        policy_graph = {
            "type": "finite_horizon",
            "annual_discount_rate": 0.1,
            "nodes": [],
            "transitions": [],
        }
        case_dir = _minimal_case(tmp_path, policy_graph=policy_graph)

        with dx.collect() as collected:
            load_temporal_context(case_dir)

        assert not [d for d in collected if d.code == _DIAG_CODE]


class TestTreeAveragesDiagnosticMessageHygiene:
    """AC #3: the message reaches pip-installed end users with no repo checkout."""

    def test_no_repo_internal_references(self, tmp_path: Path) -> None:
        case_dir = _minimal_case(tmp_path, policy_graph=_tree_policy_graph())

        with dx.collect() as collected:
            load_temporal_context(case_dir)

        diag = next(d for d in collected if d.code == _DIAG_CODE)
        assert diag.remediation

        for text in (diag.summary, diag.remediation):
            for leak in _REPO_INTERNAL_LEAKS:
                assert leak not in text, f"leaks {leak!r}: {text!r}"
            # No "/"-separated path fragment of any kind.
            assert "/" not in text
            # No private (_-prefixed) symbol.
            for token in text.replace(",", " ").replace(".", " ").split():
                assert not token.startswith("_"), (
                    f"leaks a private symbol {token!r}: {text!r}"
                )


# ---------------------------------------------------------------------------
# Full synthetic case for the CliRunner surfacing test (AC #4)
# ---------------------------------------------------------------------------


def _build_full_case(tmp_path: Path, *, tree: bool) -> Path:
    """A complete, self-contained Cobre case directory ``build_dashboard()`` can
    render end-to-end (mirrors ``TestDashboardIntegration.case_dir`` in
    ``tests/test_dashboard.py``, proven against the real ``build_dashboard()``
    pipeline). *tree* adds a non-empty ``policy_graph.nodes`` list.
    """
    case = tmp_path / ("tree_case" if tree else "linear_case")
    case.mkdir()

    stages_data: dict[str, Any] = {
        "stages": [
            {
                "id": 0,
                "start_date": "2026-01-01",
                "blocks": [
                    {"id": 0, "hours": 120.0},
                    {"id": 1, "hours": 300.0},
                ],
            },
            {
                "id": 1,
                "start_date": "2026-02-01",
                "blocks": [
                    {"id": 0, "hours": 112.0},
                    {"id": 1, "hours": 280.0},
                ],
            },
        ]
    }
    if tree:
        stages_data["policy_graph"] = _tree_policy_graph()
    _write_json(case / "stages.json", stages_data)

    _write_json(case / "config.json", {"num_scenarios": 2, "num_stages": 2})

    _write_json(
        case / "system" / "hydros.json",
        {
            "hydros": [
                hydro_with_group(
                    0,
                    0,
                    name="HYDRO_A",
                    reservoir={"max_storage_hm3": 5000.0, "min_storage_hm3": 100.0},
                    generation={
                        "max_generation_mw": 1000.0,
                        "productivity_mw_per_m3s": 0.08,
                        "max_turbined_m3s": 12000.0,
                    },
                )
            ]
        },
    )
    _write_json(case / "system" / "buses.json", {"buses": [{"id": 0, "name": "SE"}]})
    _write_json(
        case / "system" / "thermals.json",
        {
            "thermals": [
                {
                    "id": 0,
                    "name": "GAS_A",
                    "bus_id": 0,
                    "cost_per_mwh": 200.0,
                    "generation": {"min_mw": 0.0, "max_mw": 300.0},
                }
            ]
        },
    )
    _write_json(case / "system" / "lines.json", {"lines": []})
    _write_json(
        case / "system" / "non_controllable_sources.json",
        {"non_controllable_sources": []},
    )

    (case / "scenarios").mkdir(parents=True, exist_ok=True)
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "mean_mw": pa.array([50000.0, 48000.0], type=pa.float64()),
            "std_mw": pa.array([0.0, 0.0], type=pa.float64()),
        }
    )
    pq.write_table(load_stats_table, case / "scenarios" / "load_seasonal_stats.parquet")
    _write_json(
        case / "scenarios" / "load_factors.json",
        {
            "load_factors": [
                {
                    "bus_id": 0,
                    "stage_id": 0,
                    "block_factors": [
                        {"block_id": 0, "factor": 1.05},
                        {"block_id": 1, "factor": 0.95},
                    ],
                },
                {
                    "bus_id": 0,
                    "stage_id": 1,
                    "block_factors": [
                        {"block_id": 0, "factor": 1.03},
                        {"block_id": 1, "factor": 0.97},
                    ],
                },
            ]
        },
    )

    conv_dir = case / "output" / "training"
    conv_dir.mkdir(parents=True)
    conv_table = pa.table(
        {
            "iteration": pa.array([1, 2], type=pa.int32()),
            "lower_bound": pa.array([1.0e9, 1.1e9], type=pa.float64()),
            "upper_bound_mean": pa.array([1.5e9, 1.4e9], type=pa.float64()),
            "upper_bound_std": pa.array([1.0e7, 9.0e6], type=pa.float64()),
            "gap_percent": pa.array([33.3, 21.4], type=pa.float64()),
            "cuts_added": pa.array([10, 8], type=pa.int32()),
            "cuts_removed": pa.array([0, 0], type=pa.int32()),
            "cuts_active": pa.array([10, 18], type=pa.int64()),
            "time_forward_ms": pa.array([100, 90], type=pa.int64()),
            "time_backward_ms": pa.array([200, 180], type=pa.int64()),
            "time_total_ms": pa.array([300, 270], type=pa.int64()),
            "forward_passes": pa.array([5, 5], type=pa.int32()),
            "lp_solves": pa.array([100, 90], type=pa.int64()),
        }
    )
    pq.write_table(conv_table, conv_dir / "convergence.parquet")

    sim_base = case / "output" / "simulation"

    def _write_sim_parquet(entity: str, scenario_id: int, table: pa.Table) -> None:
        d = sim_base / entity / f"scenario_id={scenario_id:04d}"
        d.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, d / "data.parquet")

    hydro_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([0, 0], type=pa.int32()),
            "hydro_id": pa.array([0, 0], type=pa.int32()),
            "generation_mw": pa.array([800.0, 750.0], type=pa.float64()),
            "generation_mwh": pa.array([96000.0, 84000.0], type=pa.float64()),
            "spillage_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "turbined_m3s": pa.array([10000.0, 9500.0], type=pa.float64()),
            "storage_final_hm3": pa.array([4500.0, 4600.0], type=pa.float64()),
            "storage_initial_hm3": pa.array([4400.0, 4500.0], type=pa.float64()),
            "inflow_m3s": pa.array([500.0, 480.0], type=pa.float64()),
            "outflow_m3s": pa.array([10000.0, 9500.0], type=pa.float64()),
            "incremental_inflow_m3s": pa.array([500.0, 480.0], type=pa.float64()),
            "water_value_per_hm3": pa.array([1.5, 1.4], type=pa.float64()),
            "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "evaporation_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "productivity_mw_per_m3s": pa.array([0.08, 0.08], type=pa.float64()),
            "storage_binding_code": pa.array([0, 0], type=pa.int8()),
            "operative_state_code": pa.array([1, 1], type=pa.int8()),
            "turbined_slack_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_slack_below_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_slack_above_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "generation_slack_mw": pa.array([0.0, 0.0], type=pa.float64()),
            "storage_violation_below_hm3": pa.array([0.0, 0.0], type=pa.float64()),
            "filling_target_violation_hm3": pa.array([0.0, 0.0], type=pa.float64()),
            "diverted_inflow_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "diverted_outflow_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "evaporation_violation_pos_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "evaporation_violation_neg_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "inflow_nonnegativity_slack_m3s": pa.array([0.0, 0.0], type=pa.float64()),
            "water_withdrawal_violation_pos_m3s": pa.array(
                [0.0, 0.0], type=pa.float64()
            ),
            "water_withdrawal_violation_neg_m3s": pa.array(
                [0.0, 0.0], type=pa.float64()
            ),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("hydros", sid, hydro_table)

    thermal_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([0, 0], type=pa.int32()),
            "thermal_id": pa.array([0, 0], type=pa.int32()),
            "generation_mw": pa.array([200.0, 210.0], type=pa.float64()),
            "generation_mwh": pa.array([24000.0, 23520.0], type=pa.float64()),
            "generation_cost": pa.array([4.8e6, 4.7e6], type=pa.float64()),
            "is_gnl": pa.array([False, False]),
            "gnl_committed_mw": pa.array([0.0, 0.0], type=pa.float64()),
            "gnl_decision_mw": pa.array([0.0, 0.0], type=pa.float64()),
            "operative_state_code": pa.array([1, 1], type=pa.int8()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("thermals", sid, thermal_table)

    ncs_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([0, 0], type=pa.int32()),
            "non_controllable_id": pa.array([0, 0], type=pa.int32()),
            "generation_mw": pa.array([100.0, 90.0], type=pa.float64()),
            "generation_mwh": pa.array([12000.0, 10080.0], type=pa.float64()),
            "available_mw": pa.array([110.0, 100.0], type=pa.float64()),
            "curtailment_mw": pa.array([10.0, 10.0], type=pa.float64()),
            "curtailment_mwh": pa.array([1200.0, 1120.0], type=pa.float64()),
            "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "operative_state_code": pa.array([1, 1], type=pa.int8()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("non_controllables", sid, ncs_table)

    bus_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([0, 0], type=pa.int32()),
            "bus_id": pa.array([0, 0], type=pa.int32()),
            "load_mw": pa.array([1000.0, 980.0], type=pa.float64()),
            "load_mwh": pa.array([120000.0, 109760.0], type=pa.float64()),
            "deficit_mw": pa.array([0.0, 0.0], type=pa.float64()),
            "deficit_mwh": pa.array([0.0, 0.0], type=pa.float64()),
            "excess_mw": pa.array([0.0, 0.0], type=pa.float64()),
            "excess_mwh": pa.array([0.0, 0.0], type=pa.float64()),
            "spot_price": pa.array([150.0, 145.0], type=pa.float64()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("buses", sid, bus_table)

    exchange_table = pa.table(
        {
            "stage_id": pa.array([], type=pa.int32()),
            "block_id": pa.array([], type=pa.int32()),
            "line_id": pa.array([], type=pa.int32()),
            "direct_flow_mw": pa.array([], type=pa.float64()),
            "reverse_flow_mw": pa.array([], type=pa.float64()),
            "net_flow_mw": pa.array([], type=pa.float64()),
            "net_flow_mwh": pa.array([], type=pa.float64()),
            "losses_mw": pa.array([], type=pa.float64()),
            "losses_mwh": pa.array([], type=pa.float64()),
            "exchange_cost": pa.array([], type=pa.float64()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("exchanges", sid, exchange_table)

    cost_table = pa.table(
        {
            "stage_id": pa.array([0, 1], type=pa.int32()),
            "block_id": pa.array([None, None], type=pa.int32()),
            "total_cost": pa.array([5.0e9, 4.8e9], type=pa.float64()),
            "immediate_cost": pa.array([5.0e8, 4.8e8], type=pa.float64()),
            "future_cost": pa.array([4.5e9, 4.32e9], type=pa.float64()),
            "discount_factor": pa.array([1.0, 0.99], type=pa.float64()),
            "thermal_cost": pa.array([4.8e6, 4.7e6], type=pa.float64()),
            "contract_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "deficit_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "excess_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "storage_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "filling_target_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "hydro_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_violation_below_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "outflow_violation_above_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "turbined_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "generation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "evaporation_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "withdrawal_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "inflow_penalty_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "generic_violation_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "spillage_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "turbined_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "curtailment_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "exchange_cost": pa.array([0.0, 0.0], type=pa.float64()),
            "pumping_cost": pa.array([0.0, 0.0], type=pa.float64()),
        }
    )
    for sid in (0, 1):
        _write_sim_parquet("costs", sid, cost_table)

    return case


class TestDashboardCliSurfacesTreeAveragesDiagnostic:
    """AC #4: the dashboard command surfaces the diagnostic via CliRunner."""

    @staticmethod
    def _invoke(argv: list[str]) -> Any:
        from cobre_bridge.cli import app

        return CliRunner().invoke(app, argv)

    def test_non_json_prints_title_on_stderr(self, tmp_path: Path) -> None:
        case_dir = _build_full_case(tmp_path, tree=True)

        result = self._invoke(["dashboard", str(case_dir)])

        assert result.exit_code == 0, result.output
        assert "Scenario tree averages are not probability-weighted" in result.stderr

    def test_json_verdict_includes_code_and_exits_zero(self, tmp_path: Path) -> None:
        case_dir = _build_full_case(tmp_path, tree=True)

        result = self._invoke(["dashboard", str(case_dir), "--json"])

        assert result.exit_code == 0, result.output
        document = json.loads(result.stdout)
        assert document["command"] == "dashboard"
        codes = [d["code"] for d in document["diagnostics"]]
        assert _DIAG_CODE in codes

    def test_linear_case_has_no_tree_diagnostic(self, tmp_path: Path) -> None:
        case_dir = _build_full_case(tmp_path, tree=False)

        result = self._invoke(["dashboard", str(case_dir), "--json"])

        assert result.exit_code == 0, result.output
        document = json.loads(result.stdout)
        codes = [d["code"] for d in document["diagnostics"]]
        assert _DIAG_CODE not in codes


@pytest.mark.parametrize("tree", [True, False])
def test_build_full_case_fixture_is_valid(tmp_path: Path, tree: bool) -> None:
    """Guards the fixture itself: ``build_dashboard()`` must complete without
    raising regardless of the ``tree`` flag, independent of the CLI layer."""
    from cobre_bridge.dashboard import build_dashboard

    case_dir = _build_full_case(tmp_path, tree=tree)
    output_path = tmp_path / "dashboard.html"

    build_dashboard(case_dir, output_path)

    assert output_path.exists()
