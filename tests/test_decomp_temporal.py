"""Tests for the DECOMP operative-calendar temporal conversion."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from cobre_bridge.decomp.temporal import (
    OperativeStage,
    build_node_graph,
    build_operative_calendar,
    convert_stages,
    operative_calendar_from_dadger,
    stage_records,
)

# The two committed production decks (integration tests skip when absent).
_RV0_DECK = Path("example/decomp-set-24-rv0/dadger.rv0")
_RV3_DECK = Path("example/decomp-jul-26-rv3/dadger.rv3")

# rv0-shaped block hours: five operative weeks then the aggregated month.
_RV0_WEEK = [40.0, 48.0, 80.0]
_RV0_MONTH = [152.0, 184.0, 312.0]
_RV0_HOURS = [_RV0_WEEK] * 5 + [_RV0_MONTH]


class TestBuildOperativeCalendar:
    def test_rv0_shape_dates_and_seasons(self) -> None:
        calendar = build_operative_calendar(date(2024, 8, 31), _RV0_HOURS)
        assert [s.start_date for s in calendar] == [
            date(2024, 8, 31),
            date(2024, 9, 7),
            date(2024, 9, 14),
            date(2024, 9, 21),
            date(2024, 9, 28),
            date(2024, 10, 5),
        ]
        assert calendar[-1].end_date == date(2024, 11, 1)
        # Weekly stages carry September (8); the monthly stage October (9) —
        # including the straddling head week (31/08) and tail week (28/09).
        assert [s.season_id for s in calendar] == [8, 8, 8, 8, 8, 9]
        assert calendar[0].total_hours == 168.0
        assert calendar[-1].total_hours == 648.0

    def test_rv3_shape(self) -> None:
        hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
        calendar = build_operative_calendar(date(2026, 7, 18), hours)
        assert [s.start_date for s in calendar] == [
            date(2026, 7, 18),
            date(2026, 7, 25),
            date(2026, 8, 1),
        ]
        assert calendar[-1].end_date == date(2026, 9, 1)
        assert [s.season_id for s in calendar] == [6, 6, 7]

    def test_year_wrap_december_deck(self) -> None:
        # A December rv0: first week 28/11→05/12 (Friday 04/12 ⇒ December),
        # five weekly stages then January aggregated.
        # 02/01→01/02 is 30 days = 720 h.
        hours = [[40.0, 48.0, 80.0]] * 5 + [[168.0, 200.0, 352.0]]
        calendar = build_operative_calendar(date(2026, 11, 28), hours)
        assert calendar[-1].start_date == date(2027, 1, 2)
        assert calendar[-1].end_date == date(2027, 2, 1)
        assert [s.season_id for s in calendar] == [11, 11, 11, 11, 11, 0]

    def test_rejects_non_saturday_start(self) -> None:
        with pytest.raises(ValueError, match="Saturday"):
            build_operative_calendar(date(2024, 9, 1), _RV0_HOURS)

    def test_rejects_non_weekly_stage(self) -> None:
        hours = [[40.0, 48.0, 56.0]] + [_RV0_WEEK] * 4 + [_RV0_MONTH]
        with pytest.raises(ValueError, match="168"):
            build_operative_calendar(date(2024, 8, 31), hours)

    def test_rejects_fractional_days(self) -> None:
        hours = [_RV0_WEEK] * 5 + [[150.0, 184.0, 312.0]]
        with pytest.raises(ValueError, match="whole"):
            build_operative_calendar(date(2024, 8, 31), hours)

    def test_rejects_final_stage_not_month_end(self) -> None:
        hours = [_RV0_WEEK] * 5 + [[152.0, 184.0, 288.0]]  # 624 h → 31/10
        with pytest.raises(ValueError, match="month boundary"):
            build_operative_calendar(date(2024, 8, 31), hours)

    def test_rejects_single_stage(self) -> None:
        with pytest.raises(ValueError, match="two stages"):
            build_operative_calendar(date(2024, 8, 31), [_RV0_WEEK])


class TestStageRecords:
    def _calendar(self) -> list[OperativeStage]:
        return build_operative_calendar(date(2024, 8, 31), _RV0_HOURS)

    def test_record_shape(self) -> None:
        records = stage_records(self._calendar())
        assert [r["id"] for r in records] == list(range(6))
        first = records[0]
        assert first["start_date"] == "2024-08-31"
        assert first["season_id"] == 8
        assert first["blocks"] == [
            {"id": 0, "name": "HEAVY", "hours": 40.0},
            {"id": 1, "name": "MEDIUM", "hours": 48.0},
            {"id": 2, "name": "LIGHT", "hours": 80.0},
        ]
        assert first["risk_measure"] == "expectation"
        assert first["state_variables"] == {"storage": True, "inflow_lags": False}

    def test_no_num_openings_on_external_stages(self) -> None:
        """Every DECOMP stage is external-only, so cobre rejects a per-stage
        ``num_openings`` (and the retired ``num_scenarios``); the node graph's
        per-node ``scenario_id`` binds the openings instead."""
        for record in stage_records(self._calendar()):
            assert "num_openings" not in record
            assert "num_scenarios" not in record

    def test_stage_records_inflow_lags_false(self) -> None:
        """Locks the P3 lag-blind convention (ticket-007) on EVERY stage, not
        just the first: ``inflow_lags`` disabled, ``storage`` enabled. This is
        the exact shape that trips cobre's non-fatal external-solver-interop
        validation warning on purpose — ``convert decomp --validate``
        whitelists that warning by relying on this convention holding.
        """
        records = stage_records(self._calendar())
        assert len(records) == 6
        for record in records:
            assert record["state_variables"]["inflow_lags"] is False
            assert record["state_variables"]["storage"] is True

    def test_convert_stages_document(self) -> None:
        doc = convert_stages(
            self._calendar(),
            annual_discount_rate=0.12,
            fan_probabilities=[0.5, 0.5],
        )
        assert doc["season_definitions"]["cycle_type"] == "monthly"
        assert len(doc["season_definitions"]["seasons"]) == 12
        graph = doc["policy_graph"]
        assert graph["type"] == "finite_horizon"
        assert graph["annual_discount_rate"] == 0.12
        # 6 stages: 5 trunk nodes (0..4) + a 2-node terminal fan (ids 5, 6).
        assert [n["id"] for n in graph["nodes"]] == [0, 1, 2, 3, 4, 5, 6]
        assert graph["nodes"][0] == {
            "id": 0,
            "stage_id": 0,
            "scenario_id": 0,
            "label": "trunk-0",
        }
        assert graph["nodes"][-1] == {
            "id": 6,
            "stage_id": 5,
            "scenario_id": 1,
            "label": "fan-1",
        }
        # 4 trunk edges (0->1..3->4) + 2 terminal branch edges (4->5, 4->6).
        assert len(graph["transitions"]) == 6
        assert graph["transitions"][0] == {
            "source_id": 0,
            "target_id": 1,
            "probability": 1.0,
        }
        assert graph["transitions"][-2:] == [
            {"source_id": 4, "target_id": 5, "probability": 0.5},
            {"source_id": 4, "target_id": 6, "probability": 0.5},
        ]
        assert "pre_study_stages" not in doc


class TestBuildNodeGraph:
    def test_trunk_plus_fan_ids_and_stages(self) -> None:
        # 4 stages: trunk stages 0,1,2 then a 3-opening terminal fan at stage 3.
        nodes, transitions = build_node_graph(4, [0.2, 0.3, 0.5])
        assert nodes == [
            {"id": 0, "stage_id": 0, "scenario_id": 0, "label": "trunk-0"},
            {"id": 1, "stage_id": 1, "scenario_id": 0, "label": "trunk-1"},
            {"id": 2, "stage_id": 2, "scenario_id": 0, "label": "trunk-2"},
            {"id": 3, "stage_id": 3, "scenario_id": 0, "label": "fan-0"},
            {"id": 4, "stage_id": 3, "scenario_id": 1, "label": "fan-1"},
            {"id": 5, "stage_id": 3, "scenario_id": 2, "label": "fan-2"},
        ]
        assert transitions == [
            {"source_id": 0, "target_id": 1, "probability": 1.0},
            {"source_id": 1, "target_id": 2, "probability": 1.0},
            {"source_id": 2, "target_id": 3, "probability": 0.2},
            {"source_id": 2, "target_id": 4, "probability": 0.3},
            {"source_id": 2, "target_id": 5, "probability": 0.5},
        ]

    def test_single_root_at_stage_zero(self) -> None:
        """Exactly one node sits at stage 0 (the trunk head / graph root)."""
        nodes, _ = build_node_graph(4, [0.5, 0.5])
        assert sum(1 for n in nodes if n["stage_id"] == 0) == 1

    def test_leaves_only_at_terminal_stage(self) -> None:
        """Every node with no out-edge sits at the final stage (no mid-horizon
        leaf), and every edge advances exactly one stage."""
        nodes, transitions = build_node_graph(4, [0.5, 0.5])
        terminal_stage = max(n["stage_id"] for n in nodes)
        stage_of = {n["id"]: n["stage_id"] for n in nodes}
        sources = {t["source_id"] for t in transitions}
        for node in nodes:
            if node["id"] not in sources:
                assert node["stage_id"] == terminal_stage
        for t in transitions:
            assert stage_of[t["target_id"]] == stage_of[t["source_id"]] + 1

    def test_degenerate_single_scenario_fan(self) -> None:
        """A one-member fan degenerates to a pure chain into one terminal leaf."""
        nodes, transitions = build_node_graph(3, [1.0])
        assert [n["id"] for n in nodes] == [0, 1, 2]
        assert transitions[-1] == {"source_id": 1, "target_id": 2, "probability": 1.0}

    def test_rejects_single_stage(self) -> None:
        with pytest.raises(ValueError, match="trunk stage and a terminal"):
            build_node_graph(1, [0.5, 0.5])


class TestFromDadger:
    @pytest.mark.skipif(not _RV0_DECK.exists(), reason="rv0 deck not present")
    def test_rv0_deck(self) -> None:
        from idecomp.decomp import Dadger

        calendar = operative_calendar_from_dadger(Dadger.read(str(_RV0_DECK)))
        assert len(calendar) == 6
        assert calendar[0].start_date == date(2024, 8, 31)
        assert calendar[-1].end_date == date(2024, 11, 1)
        assert [s.season_id for s in calendar] == [8, 8, 8, 8, 8, 9]
        assert calendar[0].block_hours == (40.0, 48.0, 80.0)
        assert calendar[-1].block_hours == (152.0, 184.0, 312.0)

    @pytest.mark.skipif(not _RV3_DECK.exists(), reason="rv3 deck not present")
    def test_rv3_deck(self) -> None:
        from idecomp.decomp import Dadger

        calendar = operative_calendar_from_dadger(Dadger.read(str(_RV3_DECK)))
        assert len(calendar) == 3
        assert calendar[0].start_date == date(2026, 7, 18)
        assert calendar[-1].start_date == date(2026, 8, 1)
        assert calendar[-1].end_date == date(2026, 9, 1)
        assert [s.season_id for s in calendar] == [6, 6, 7]
        assert calendar[0].block_hours == (15.0, 64.0, 89.0)
        assert calendar[-1].block_hours == (63.0, 280.0, 401.0)

    def test_rejects_cross_subsystem_mismatch(self) -> None:
        import pandas as pd

        class _Dt:
            dia, mes, ano = 31, 8, 2024

        class _Dadger:
            dt = _Dt()

            @staticmethod
            def dp(df: bool = False) -> pd.DataFrame:
                rows = []
                for estagio in (1, 2):
                    for sbm, d1 in ((1, 40.0), (2, 40.0 if estagio == 1 else 48.0)):
                        rows.append(
                            {
                                "codigo_submercado": sbm,
                                "estagio": estagio,
                                "numero_patamares": 3,
                                "duracao_1": d1,
                                "duracao_2": 48.0,
                                "duracao_3": 80.0,
                            }
                        )
                return pd.DataFrame(rows)

        with pytest.raises(ValueError, match="differ across"):
            operative_calendar_from_dadger(_Dadger())  # type: ignore[arg-type]
