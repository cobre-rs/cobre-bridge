"""Tests for the DECOMP-vs-Cobre results comparison slice."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.comparators.charts import (
    _BALANCE_VARS,
    _COST_MAP,
    hydro_slack_aggregate_chart,
    performance_fwd_bwd_split_chart,
    ree_energy_chart,
)
from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.decomp_results import (
    _BUS_VARIABLES,
    _CANONICAL_VARIABLE,
    _CONSTRAINT_NAME_RE,
    _DEVIATION_VIOLATION_LABEL,
    _EARM_MWH_TO_MWMES,
    _GC_LHS_SCHEMA,
    _HM3_TO_M3S_HOUR_FACTOR,
    _HYDRO_VARIABLES,
    _NW_COST_LABELS,
    _PRODUCTIVITY_TURBINED_EPS,
    _THERMAL_VARIABLES,
    _UNSUPPORTED_TERM_VARIABLES,
    DecompComparison,
    _AlignedDecompFrames,
    _build_line_id_map,
    _bus_side,
    _cobre_ree_sums,
    _cobre_stage_hours,
    _corridor_line_alignment,
    _cost_frames,
    _decomp_constraint_context,
    _decomp_convergence_frame,
    _decomp_max_stage,
    _decomp_ree_frame,
    _decomp_tim_iterations,
    _decomp_tim_stages,
    _DecompConstraintContext,
    _DecompConstraintLookups,
    _energy_balance_frames,
    _evap_side,
    _evaporation_result_comparisons,
    _fpha_metrics,
    _generic_constraint_lhs_decomp,
    _hm3_to_m3s,
    _hydro_productivity_results,
    _interc_side,
    _line_bounds_and_meta,
    _line_entity_names,
    _line_result_comparisons,
    _map_entities,
    _merge_hydro_bus_ids,
    _read_cobre_lines_index,
    _ree_membership_map,
    _ree_result_comparisons,
    _result_comparisons,
    _rhe_lhs_lookup,
    _scenario_mean,
    _stage_frame_to_lookup,
    _stage_rows,
    _storage_lookup,
    _summarize,
    _term_lookup_value,
    _tidy,
    _union_cost_rows,
    build_decomp_dataset,
    compare_decomp_results,
)
from cobre_bridge.comparators.report_builder import build_comparison_report
from cobre_bridge.comparators.results import ResultComparison
from cobre_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.verdict import decomp_compare_summary, decomp_dataset_summary


def _source_frame() -> pl.DataFrame:
    """Two stages of one plant: per-block rows plus the stage-aggregate row."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 2, 2, 2],
            "no": [1, 1, 1, 2, 2, 2],
            "patamar": [1.0, 2.0, None, 1.0, 2.0, None],
            "duracao": [24.0, 144.0, None, 24.0, 144.0, None],
            "codigo_usina": [10, 10, 10, 10, 10, 10],
            "geracao_MW": [120.0, 60.0, 68.57, 100.0, 50.0, 57.14],
        }
    )


class TestStageRows:
    def test_prefers_the_aggregate_row(self) -> None:
        rows = _stage_rows(_source_frame())
        assert len(rows) == 2
        assert "patamar" not in rows.columns
        assert rows["geracao_MW"].to_list() == [68.57, 57.14]

    def test_falls_back_to_block_rows_when_absent(self) -> None:
        frame = _source_frame().filter(pl.col("patamar").is_not_null())
        rows = _stage_rows(frame)
        assert len(rows) == 4


class TestScenarioMean:
    def test_averages_over_nodes(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [3, 3, 3],
                "codigo_usina": [10, 10, 10],
                "geracao_MW": [10.0, 20.0, 60.0],
            }
        )
        out = _scenario_mean(
            frame, "estagio", ["geracao_MW"], entity_column="codigo_usina"
        )
        assert out["geracao_MW"].to_list() == [30.0]


class TestScenarioMeanCompositeKey:
    """ticket-007: grouping by a composite entity key -- an interchange
    corridor's ``(de, para)`` code pair, not a single entity code."""

    def test_averages_over_nodes_per_corridor(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [1, 1, 1, 1],
                "codigo_submercado_de": [1, 1, 2, 2],
                "codigo_submercado_para": [2, 2, 3, 3],
                "intercambio_origem_MW": [100.0, 120.0, 10.0, 30.0],
            }
        )
        out = _scenario_mean(
            frame,
            "estagio",
            ["intercambio_origem_MW"],
            entity_column=["codigo_submercado_de", "codigo_submercado_para"],
        )
        by_pair = {
            (row["codigo_submercado_de"], row["codigo_submercado_para"]): row[
                "intercambio_origem_MW"
            ]
            for row in out.iter_rows(named=True)
        }
        assert by_pair == {(1, 2): 110.0, (2, 3): 20.0}


class TestMapEntities:
    def test_maps_codes_and_rebases_stages(self) -> None:
        frame = pl.DataFrame({"estagio": [1, 2], "codigo_usina": [10, 10]})
        mapped, unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["entity_id"].to_list() == [4, 4]
        assert mapped["stage_id"].to_list() == [0, 1]
        assert unmapped == []

    def test_reports_unmapped_codes_instead_of_dropping_silently(self) -> None:
        frame = pl.DataFrame({"estagio": [1, 1], "codigo_usina": [10, 99]})
        mapped, unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["entity_id"].to_list() == [4]
        assert unmapped == [99]

    def test_keeps_the_original_code_as_newave_code(self) -> None:
        """``build_decomp_dataset`` needs the reference code back to fill
        ``ResultComparison.newave_code`` -- ``_map_entities`` must not drop it
        once it has been used to derive ``entity_id``."""
        frame = pl.DataFrame({"estagio": [1, 2], "codigo_usina": [10, 10]})
        mapped, _unmapped = _map_entities(frame, "codigo_usina", {10: 4})
        assert mapped["newave_code"].to_list() == [10, 10]


class TestTidyAndSummary:
    def _pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        source = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "geracao_MW": [100.0, 50.0],
            }
        )
        cobre = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw": [90.0, 50.0],
            }
        )
        return source, cobre

    def test_tidy_rows_carry_both_sides_and_the_difference(self) -> None:
        source, cobre = self._pair()
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={0: "A", 1: "B"})
        generation = rows.filter(pl.col("variable") == "generation").sort("entity_id")
        assert generation["source"].to_list() == [100.0, 50.0]
        assert generation["cobre"].to_list() == [90.0, 50.0]
        assert generation["delta"].to_list() == [-10.0, 0.0]
        assert generation["delta_pct"].to_list() == [pytest.approx(-10.0), 0.0]
        assert generation["entity_name"].to_list() == ["A", "B"]

    def test_variables_missing_on_either_side_are_skipped(self) -> None:
        source, cobre = self._pair()
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={})
        # Only generation is present in both frames.
        assert set(rows["variable"]) == {"generation"}

    def test_summary_totals_and_worst_entity(self) -> None:
        source, cobre = self._pair()
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={0: "A", 1: "B"})
        summary = _summarize(rows)
        assert len(summary) == 1
        row = summary.to_dicts()[0]
        assert row["n"] == 2
        assert row["source_total"] == 150.0
        assert row["cobre_total"] == 140.0
        assert row["delta_total"] == -10.0
        assert row["delta_total_pct"] == pytest.approx(-100.0 / 15.0)
        assert row["worst_entity"] == "A"

    def test_zero_versus_zero_reads_as_agreement(self) -> None:
        source = pl.DataFrame({"entity_id": [0], "stage_id": [0], "geracao_MW": [0.0]})
        cobre = pl.DataFrame(
            {"entity_id": [0], "stage_id": [0], "generation_mw": [0.0]}
        )
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={})
        assert rows["smape_pct"].to_list() == [0.0]
        assert rows["delta_pct"].to_list() == [None]

    def test_empty_join_yields_an_empty_but_typed_frame(self) -> None:
        source = pl.DataFrame({"entity_id": [0], "stage_id": [0], "geracao_MW": [1.0]})
        cobre = pl.DataFrame(
            {"entity_id": [9], "stage_id": [9], "generation_mw": [1.0]}
        )
        rows = _tidy(source, cobre, _HYDRO_VARIABLES, names={})
        assert rows.is_empty()
        assert "smape_pct" in rows.columns
        assert _summarize(rows).is_empty()


class TestResultComparisons:
    """``_result_comparisons`` is the ``ResultComparison`` counterpart of
    ``_tidy``, feeding ``build_decomp_dataset`` instead of ``DecompComparison``."""

    def _pair(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        source = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "newave_code": [10, 11],
                "stage_id": [0, 0],
                "geracao_MW": [100.0, 50.0],
            }
        )
        cobre = pl.DataFrame(
            {
                "entity_id": [0, 1],
                "stage_id": [0, 0],
                "generation_mw": [90.0, 50.0],
            }
        )
        return source, cobre

    def test_emits_one_result_comparison_per_row_with_the_canonical_variable(
        self,
    ) -> None:
        source, cobre = self._pair()
        results = _result_comparisons(
            source, cobre, _HYDRO_VARIABLES, names={0: "A", 1: "B"}
        )
        assert {r.variable for r in results} == {"generation_mw"}
        by_id = {r.cobre_id: r for r in results}
        assert by_id[0].newave_code == 10
        assert by_id[0].entity_name == "A"
        assert by_id[0].entity_type == "hydro"
        assert by_id[0].stage == 0
        assert by_id[0].newave_value == 100.0
        assert by_id[0].cobre_value == 90.0
        assert by_id[0].abs_diff == pytest.approx(10.0)
        assert by_id[0].rel_diff == pytest.approx(0.1)
        assert by_id[1].newave_code == 11
        assert by_id[1].abs_diff == pytest.approx(0.0)

    def test_variables_missing_on_either_side_are_skipped(self) -> None:
        source, cobre = self._pair()
        results = _result_comparisons(source, cobre, _HYDRO_VARIABLES, names={})
        # Only generation's columns are present in both frames.
        assert {r.variable for r in results} == {"generation_mw"}

    def test_empty_join_yields_no_results(self) -> None:
        source = pl.DataFrame(
            {
                "entity_id": [0],
                "newave_code": [10],
                "stage_id": [0],
                "geracao_MW": [1.0],
            }
        )
        cobre = pl.DataFrame(
            {"entity_id": [9], "stage_id": [9], "generation_mw": [1.0]}
        )
        assert _result_comparisons(source, cobre, _HYDRO_VARIABLES, names={}) == []

    def test_canonical_variable_covers_all_eight_today_variables(self) -> None:
        """D-SOURCE-TOKEN-adjacent guard: every ``_Variable`` spec this module
        ships must resolve to a canonical chart name -- a spec with no entry
        would raise a ``KeyError`` deep inside ``_result_comparisons``."""
        all_vars = _HYDRO_VARIABLES + _THERMAL_VARIABLES + _BUS_VARIABLES
        assert len(all_vars) == 8
        canonical_names = {_CANONICAL_VARIABLE[(v.level, v.name)] for v in all_vars}
        assert canonical_names == {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
        }


def _aligned_fixture() -> _AlignedDecompFrames:
    """One hydro plant, one thermal plant, one bus -- already aligned to Cobre
    ids/stages, matching the shape :func:`_read_aligned_frames` returns."""
    source_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "newave_code": [10, 20],
            "stage_id": [0, 0],
            "geracao_MW": [120.0, 60.0],
            "vazao_turbinada_m3s": [80.0, 40.0],
            "vazao_vertida_m3s": [0.0, 0.0],
            "vazao_defluente_m3s": [80.0, 40.0],
            "volume_util_final_hm3": [500.0, 300.0],
        }
    )
    cobre_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw": [110.0, 60.0],
            "turbined_m3s": [78.0, 40.0],
            "spillage_m3s": [0.0, 0.0],
            "outflow_m3s": [78.0, 40.0],
            "useful_storage_hm3": [480.0, 300.0],
        }
    )
    source_thermal = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [5],
            "stage_id": [0],
            "geracao_MW": [30.0],
        }
    )
    cobre_thermal = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "generation_mw": [28.0]}
    )
    source_bus = pl.DataFrame(
        {
            "entity_id": [0],
            "newave_code": [1],
            "stage_id": [0],
            "deficit_MW": [0.0],
            "cmo": [45.0],
        }
    )
    cobre_bus = pl.DataFrame(
        {"entity_id": [0], "stage_id": [0], "deficit_mw": [0.0], "spot_price": [44.0]}
    )
    return _AlignedDecompFrames(
        source_hydro=source_hydro,
        source_thermal=source_thermal,
        source_bus=source_bus,
        cobre_hydro=cobre_hydro,
        cobre_thermal=cobre_thermal,
        cobre_bus=cobre_bus,
        hydro_names={0: "A", 1: "B"},
        thermal_names={0: "T"},
        bus_names={0: "SE"},
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


def _patch_aligned_frames(
    monkeypatch: pytest.MonkeyPatch, aligned: _AlignedDecompFrames
) -> None:
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._read_aligned_frames",
        lambda *_args, **_kwargs: aligned,
    )
    # ``compare_decomp_results`` also reads the convergence report directly
    # (outside ``_read_aligned_frames``); stub it so the parity fixture never
    # has to touch a real deck/output directory.
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._convergence",
        lambda *_args, **_kwargs: pl.DataFrame(schema={"iteration": pl.Int64}),
    )
    # ticket-006: ``build_decomp_dataset`` also calls
    # ``read_cobre_bus_aggregates`` directly (outside ``_read_aligned_frames``).
    # Unlike the other cobre readers it does NOT degrade to empty on a missing
    # case -- it raises ``CobrePartitionMissingError`` for the pre-0.13
    # ``hydro_bus_generation`` partition, which a bare ``tmp_path`` always
    # trips. Stub it here too, so every fixture that does not care about
    # ticket-006's Energy Balance metadata (the vast majority) keeps working
    # against a bare ``tmp_path``; tests that DO care override this again
    # afterwards (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_bus_aggregates",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    # ticket-010: ``build_decomp_dataset`` also calls ``_cost_frames`` directly
    # (outside ``_read_aligned_frames``), which reads ``read_relato_costs`` --
    # unlike every other reader here, it RAISES on a missing/empty parse
    # (ticket-009's "no silent-empty" reader contract), which a bare
    # ``tmp_path`` always trips. Stub it here too, so every fixture that does
    # not care about ticket-010's cost metadata keeps working against a bare
    # ``tmp_path``; tests that DO care override this again afterwards
    # (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._cost_frames",
        lambda *_args, **_kwargs: ({}, pl.DataFrame()),
    )
    # ticket-014: ``build_decomp_dataset`` also calls
    # ``read_cobre_hydro_bus_labels`` directly (outside ``_read_aligned_frames``).
    # Like ``read_cobre_bus_aggregates`` above (ticket-006), it reads the
    # ``simulation/hydro_bus_generation/`` partition and RAISES
    # ``CobrePartitionMissingError`` on a bare ``tmp_path`` instead of
    # degrading to empty. Stub it here too, so every fixture that does not
    # care about ticket-014's hydro metadata keeps working against a bare
    # ``tmp_path``; tests that DO care override this again afterwards
    # (monkeypatch's last ``setattr`` wins).
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_hydro_bus_labels",
        lambda *_args, **_kwargs: {},
    )


class TestBuildDecompDataset:
    """``build_decomp_dataset`` assembles the canonical dataset for the
    current 8 DECOMP variables via the shared ``_read_aligned_frames`` seam."""

    def test_dataset_validates_with_the_canonical_variables(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset.validate()
        assert set(dataset.summary["variable"].to_list()) == {
            "generation_mw",
            "turbined_m3s",
            "spillage_m3s",
            "outflow_m3s",
            "storage_final_hm3",
            "deficit_mw",
            "spot_price",
            # ticket-016: derived realized hydro productivity (both fixture
            # plants turbine well above the zero-guard on both sides).
            "productivity_mw_per_m3s",
        }

    def test_tidy_sources_are_newave_and_cobre_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert set(dataset.tidy["source"].unique().to_list()) == {"newave", "cobre"}

    def test_hydro_storage_rows_compare_useful_volume(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The cobre-side value must be ``useful_storage_hm3`` (already
        ``storage_final_hm3 - min_storage_hm3`` upstream in ``_cobre_hydro``),
        not the raw absolute ``storage_final_hm3``."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        cobre_storage = (
            dataset.tidy.filter(
                (pl.col("variable") == "storage_final_hm3")
                & (pl.col("source") == "cobre")
            )
            .sort("entity_id")["value"]
            .to_list()
        )
        assert cobre_storage == [480.0, 300.0]

    def test_unmapped_codes_surface_in_metadata_and_are_excluded_from_tidy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [86, 224],
            "bus": [],
            "line": [],
            "ree": [],
            "evaporation": [],
        }
        thermal_codes = {
            r.newave_code
            for r in dataset.metadata["results"]
            if r.entity_type == "thermal"
        }
        assert 86 not in thermal_codes
        assert 224 not in thermal_codes

    def test_calls_the_shared_stat_kernel_not_a_local_reimplementation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``top_divergences``/``footer_counts`` are only populated by
        ``analyze.build_results_dataset``, so their presence is proof that
        function ran instead of a locally re-derived summary."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert "top_divergences" in dataset.metadata
        assert "footer_counts" in dataset.metadata

    def test_empty_comparison_returns_a_schema_valid_empty_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fixture = _aligned_fixture()
        empty = _AlignedDecompFrames(
            source_hydro=fixture.source_hydro.clear(),
            source_thermal=fixture.source_thermal.clear(),
            source_bus=fixture.source_bus.clear(),
            cobre_hydro=fixture.cobre_hydro.clear(),
            cobre_thermal=fixture.cobre_thermal.clear(),
            cobre_bus=fixture.cobre_bus.clear(),
            hydro_names={},
            thermal_names={},
            bus_names={},
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        _patch_aligned_frames(monkeypatch, empty)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset.validate()
        assert dataset.tidy.is_empty()
        assert dataset.summary.is_empty()
        assert dataset.metadata["unmapped"] == {
            "hydro": [],
            "thermal": [],
            "bus": [],
            "line": [],
            "ree": [],
            "evaporation": [],
        }


class TestBusSideExcludesTranshipment:
    """The converter-created transhipment bus (``DecompIdMap.transhipment_bus_id``)
    has no source-model subsystem code -- it is referenced only by name in
    ``IA`` records, never emitted as an ``SB`` row -- so it can never appear
    among the ``codigo_submercado`` values ``_bus_side`` reads from the
    source model's system results table. The code -> id mapping ``_bus_side``
    builds its rows from (``{code: id_map.bus_id(code) for code in
    id_map.bus_codes}``) only ever holds values in ``range(len(bus_codes))``,
    one short of ``transhipment_bus_id`` -- so this is a pre-existing
    structural guarantee, not a new filter. This test pins that guarantee
    down as a regression guard rather than changing behaviour."""

    def test_bus_side_never_emits_the_transhipment_bus_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SUDESTE", "SUL"))
        source_frame = pl.DataFrame(
            {
                "estagio": [1, 1],
                "no": [1, 1],
                "patamar": [None, None],
                "codigo_submercado": [1, 2],
                "deficit_MW": [0.0, 0.0],
                "cmo": [40.0, 45.0],
            }
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_sist",
            lambda *_args, **_kwargs: source_frame,
        )
        bus_codes = {code: id_map.bus_id(code) for code in id_map.bus_codes}

        mapped, unmapped = _bus_side(tmp_path, bus_codes)

        assert unmapped == []
        assert id_map.transhipment_bus_id not in mapped["entity_id"].to_list()


def _decomp_id_map_three_subsystems() -> DecompIdMap:
    """SE (code 1) -> bus 0, S (code 2) -> bus 1, NE (code 3) -> bus 2, plus
    the converter-created transhipment bus at id 3."""
    return DecompIdMap(bus_codes=(1, 2, 3), bus_names=("SE", "S", "NE"))


def _line_entry(
    line_id: int,
    source_bus_id: int,
    target_bus_id: int,
    *,
    name: str | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": line_id,
        "source_bus_id": source_bus_id,
        "target_bus_id": target_bus_id,
        "capacity": {"direct_mw": 1000.0, "reverse_mw": 1000.0},
    }
    if name is not None:
        entry["name"] = name
    return entry


def _write_lines_json(case_dir: Path, lines: list[dict[str, Any]]) -> Path:
    """Write ``system/lines.json`` under *case_dir* and return the Cobre
    output dir (``case_dir/output``) that ``case_dir_for`` resolves back to
    *case_dir* from."""
    system_dir = case_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "lines.json").write_text(json.dumps({"lines": lines}))
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def _write_line_bounds_parquet(case_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``constraints/line_bounds.parquet`` under *case_dir* -- the
    ticket-008 Network tab's per-stage capacity source (see
    ``decomp/network.py::convert_lines``'s own ``_LINE_BOUNDS_SCHEMA``)."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(constraints_dir / "line_bounds.parquet")


class TestCorridorLineAlignment:
    """ticket-007: ``_corridor_line_alignment`` -- corridor (bus_de, bus_para)
    -> ordered cobre (line_id, sign) legs."""

    def test_direct_line_corridor_maps_with_positive_sign(self, tmp_path: Path) -> None:
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert alignment[(0, 1)] == [(0, 1)]

    def test_reverse_declared_line_gets_negative_sign(self, tmp_path: Path) -> None:
        """A cobre line declared S -> SE realizes the SE -> S corridor with
        sign -1, orienting the line's own net_flow_mw onto the requested
        direction."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 1, 0)])

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert alignment[(0, 1)] == [(0, -1)]

    def test_star_topology_two_leg_corridor(self, tmp_path: Path) -> None:
        """No direct SE-S line; SE<->IV + IV<->S realizes the SE -> S
        corridor as two consistently-oriented legs."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [
                _line_entry(0, 0, transhipment),
                _line_entry(1, transhipment, 1),
            ],
        )

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert alignment[(0, 1)] == [(0, 1), (1, 1)]

    def test_corridor_with_no_realizing_path_is_absent(self, tmp_path: Path) -> None:
        id_map = _decomp_id_map_three_subsystems()
        # Only SE-S is wired; NE (bus id 2) has no line at all, direct or
        # via the transhipment bus.
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])

        alignment = _corridor_line_alignment(output_dir, id_map)

        assert (0, 2) not in alignment
        assert (2, 0) not in alignment

    def test_missing_lines_json_yields_an_empty_alignment(self, tmp_path: Path) -> None:
        id_map = _decomp_id_map_three_subsystems()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        assert _corridor_line_alignment(output_dir, id_map) == {}

    def test_ambiguous_duplicate_line_pair_is_excluded_not_guessed(
        self, tmp_path: Path
    ) -> None:
        """Two lines declared for the same ordered bus pair make that leg
        ambiguous under the star-topology assumption -- both are dropped
        rather than one picked arbitrarily."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(
            tmp_path, [_line_entry(0, 0, 1), _line_entry(1, 0, 1)]
        )

        assert (0, 1) not in _read_cobre_lines_index(output_dir)
        assert (0, 1) not in _corridor_line_alignment(output_dir, id_map)


def _dec_oper_interc_frame(
    *, de: int, para: int, origem_mw: float, perdas_mw: float = 5.0
) -> pl.DataFrame:
    """One corridor, one stage, one node -- the patamar-null aggregate row
    ``_interc_side`` reads via ``_stage_rows``."""
    return pl.DataFrame(
        {
            "estagio": [1],
            "no": [1],
            "cenario": [1],
            "patamar": [None],
            "codigo_submercado_de": [de],
            "codigo_submercado_para": [para],
            "intercambio_origem_MW": [origem_mw],
            "intercambio_destino_MW": [origem_mw - perdas_mw],
            "perdas_MW": [perdas_mw],
        }
    )


class TestIntercSide:
    """ticket-007: ``_interc_side`` -- the aligned per-(cobre line_id, stage)
    DECOMP net-flow frame, plus the unresolved-corridor report."""

    def _patch_interc(
        self, monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_interc",
            lambda *_args, **_kwargs: frame,
        )

    def test_direct_line_corridor_reproduces_a_positive_flow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Sign orientation: a known de -> para positive flow, on a cobre
        line declared in that same de -> para direction, yields a positive
        aligned cobre net-flow."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=250.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": 250.0}
        ]

    def test_star_topology_corridor_reproduces_the_flow_on_both_legs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [_line_entry(0, 0, transhipment), _line_entry(1, transhipment, 1)],
        )
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=180.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        by_line = {
            row["entity_id"]: row["net_flow_mw"] for row in result.iter_rows(named=True)
        }
        assert by_line == {0: 180.0, 1: 180.0}

    def test_unresolved_corridor_is_reported_and_excluded_from_the_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        # NE (code 3, bus id 2) has no realizing line at all.
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=3, origem_mw=90.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result.is_empty()
        assert unresolved == [(1, 3)]

    def test_missing_lines_json_leaves_every_corridor_unresolved(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=100.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result.is_empty()
        assert unresolved == [(1, 2)]

    def test_reverse_declared_line_flips_the_aligned_sign(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The cobre line runs S -> SE; the DECOMP corridor is declared
        SE -> S with a positive flow, so the aligned cobre net-flow must be
        negative to stay oriented to the line's own
        source_bus_id -> target_bus_id convention."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 1, 0)])
        self._patch_interc(
            monkeypatch, _dec_oper_interc_frame(de=1, para=2, origem_mw=100.0)
        )

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": -100.0}
        ]

    def test_losses_are_read_but_not_subtracted_from_the_net_flow(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The aligned net-flow is the origin-side reading, matching a
        lossless cobre line -- not ``intercambio_origem_MW - perdas_MW``."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        self._patch_interc(
            monkeypatch,
            _dec_oper_interc_frame(de=1, para=2, origem_mw=200.0, perdas_mw=12.0),
        )

        result, _unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result["net_flow_mw"].to_list() == [200.0]

    def test_a_corridor_reported_by_name_only_is_unresolved_not_a_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression guard: the source model reports some corridors (the
        Itaipu 50 Hz ``IV`` node, in particular) with a null
        ``codigo_submercado_de``/``_para`` -- a name-only endpoint outside
        the ``SB`` register. This must surface as unresolved (``-1``
        sentinel), never raise, since it can never resolve through
        ``_corridor_line_alignment`` regardless (the transhipment bus is
        never an outer-loop endpoint there)."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        frame = pl.DataFrame(
            {
                "estagio": [1],
                "no": [1],
                "cenario": [1],
                "patamar": [None],
                "codigo_submercado_de": [None],
                "codigo_submercado_para": [2],
                "intercambio_origem_MW": [50.0],
                "intercambio_destino_MW": [50.0],
                "perdas_MW": [0.0],
            },
            schema_overrides={"codigo_submercado_de": pl.Float64},
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert result.is_empty()
        assert unresolved == [(-1, 2)]

    def test_both_directions_of_one_interface_net_to_a_single_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Regression: the source model reports every physical interface as
        two corridor rows, one per direction (both ``SE -> S`` and
        ``S -> SE``). Both align onto the same direct cobre line leg with
        opposite sign -- the returned frame must collapse them to a single
        ``(entity_id, stage_id)`` row (not the duplicate keys that used to
        fan out ``_line_result_comparisons``'s join into a spurious extra
        row), and that row's net flow must be the signed sum of the two
        directions."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        frame = pl.concat(
            [
                _dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
                _dec_oper_interc_frame(de=2, para=1, origem_mw=90.0),
            ]
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        assert result.to_dicts() == [
            {"entity_id": 0, "stage_id": 0, "net_flow_mw": 160.0}
        ]

    def test_both_directions_of_one_interface_net_correctly_on_a_shared_star_leg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The same duplicate-direction netting must hold on each leg of a
        two-leg star corridor: ``SE -> S`` and ``S -> SE`` both realize
        through ``SE <-> transhipment <-> S``, so each leg must collapse to
        one row equal to ``flow(SE -> S) - flow(S -> SE)``, not two rows
        colliding on the same ``(entity_id, stage_id)`` key."""
        id_map = _decomp_id_map_three_subsystems()
        transhipment = id_map.transhipment_bus_id
        output_dir = _write_lines_json(
            tmp_path,
            [_line_entry(0, 0, transhipment), _line_entry(1, transhipment, 1)],
        )
        frame = pl.concat(
            [
                _dec_oper_interc_frame(de=1, para=2, origem_mw=180.0),
                _dec_oper_interc_frame(de=2, para=1, origem_mw=70.0),
            ]
        )
        self._patch_interc(monkeypatch, frame)

        result, unresolved = _interc_side(tmp_path, output_dir, id_map)

        assert unresolved == []
        by_line = {
            row["entity_id"]: row["net_flow_mw"] for row in result.iter_rows(named=True)
        }
        assert by_line == {0: 110.0, 1: 110.0}


class TestLineEntityNames:
    """ticket-008: display name per cobre line id for the Network tab."""

    def test_uses_the_name_from_line_meta_when_present(self) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 1, name="SE-S")]

        names = _line_entity_names(line_meta, id_map)

        assert names == {0: "SE-S"}

    def test_falls_back_to_a_bus_name_pair_label_when_name_is_absent(self) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 1)]  # no "name" key

        names = _line_entity_names(line_meta, id_map)

        assert names == {0: "SE<->S"}

    def test_falls_back_to_a_line_id_label_when_the_bus_ids_are_unresolvable(
        self,
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 99)]  # bus id 99 is not declared

        names = _line_entity_names(line_meta, id_map)

        assert names == {0: "line_0"}


class TestBuildLineIdMap:
    """ticket-008: best-effort id map for the corridor -> line alignment."""

    def test_returns_none_when_the_directory_has_no_deck(self, tmp_path: Path) -> None:
        """A bare directory (no ``caso.dat``) must degrade to ``None``, not
        raise -- every other level's own ``build_decomp_dataset`` fixture
        exercises exactly this directory shape."""
        assert _build_line_id_map(tmp_path) is None


class TestLineResultComparisons:
    """ticket-008: corridor-aligned line ``ResultComparison`` rows, joining
    ticket-007's DECOMP net-flow onto Cobre's per-line simulation means."""

    def test_id_map_none_returns_no_rows_and_no_unresolved(
        self, tmp_path: Path
    ) -> None:
        results, unresolved = _line_result_comparisons(tmp_path, tmp_path, None, [])

        assert results == []
        assert unresolved == []

    def test_missing_source_interchange_table_degrades_to_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("dec_oper_interc.csv not found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_interc", _boom
        )

        results, unresolved = _line_result_comparisons(tmp_path, tmp_path, id_map, [])

        assert results == []
        assert unresolved == []

    def test_joins_source_and_cobre_into_result_comparison_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        id_map = _decomp_id_map_three_subsystems()
        line_meta = [_line_entry(0, 0, 1, name="SE-S")]
        output_dir = _write_lines_json(tmp_path, line_meta)
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_interc",
            lambda *_args, **_kwargs: _dec_oper_interc_frame(
                de=1, para=2, origem_mw=250.0
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_line_means",
            lambda *_args, **_kwargs: pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        results, unresolved = _line_result_comparisons(
            tmp_path, output_dir, id_map, line_meta
        )

        assert unresolved == []
        assert len(results) == 1
        row = results[0]
        assert row.entity_type == "line"
        assert row.entity_name == "SE-S"
        assert row.cobre_id == 0
        assert row.stage == 0
        assert row.variable == "net_flow_mw"
        assert row.newave_value == 250.0
        assert row.cobre_value == 240.0
        # D-SOURCE-TOKEN-adjacent: no single source code covers a line that
        # may be a shared leg of more than one corridor -- see the module
        # docstring.
        assert row.newave_code == 0

    def test_unresolved_corridors_surface_even_with_no_cobre_line_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """NE (code 3) has no realizing line -- reported unresolved even
        though there is nothing to join against on the Cobre side."""
        id_map = _decomp_id_map_three_subsystems()
        output_dir = _write_lines_json(tmp_path, [_line_entry(0, 0, 1)])
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_interc",
            lambda *_args, **_kwargs: _dec_oper_interc_frame(
                de=1, para=3, origem_mw=90.0
            ),
        )

        results, unresolved = _line_result_comparisons(tmp_path, output_dir, id_map, [])

        assert results == []
        assert unresolved == [[1, 3]]


class TestLineBoundsAndMeta:
    """ticket-008: cobre-side line capacity bounds + metadata, read straight
    from the converted case (mirrors ``results.compare_results``)."""

    def test_reads_line_bounds_parquet_and_lines_json(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case"
        line_meta_in = [_line_entry(0, 0, 1, name="SE-S")]
        output_dir = _write_lines_json(case_dir, line_meta_in)
        _write_line_bounds_parquet(
            case_dir,
            [
                {
                    "line_id": 0,
                    "stage_id": 0,
                    "block_id": None,
                    "direct_mw": 1200.0,
                    "reverse_mw": 800.0,
                }
            ],
        )

        line_bounds, line_meta = _line_bounds_and_meta(output_dir)

        assert isinstance(line_bounds, pd.DataFrame)
        assert not isinstance(line_bounds, pl.DataFrame)
        assert {"line_id", "stage_id", "direct_mw", "reverse_mw"}.issubset(
            set(line_bounds.columns)
        )
        assert line_bounds.iloc[0]["direct_mw"] == 1200.0
        assert line_bounds.iloc[0]["reverse_mw"] == 800.0
        assert line_meta == line_meta_in
        assert line_meta[0]["capacity"]["direct_mw"] == 1000.0

    def test_missing_files_degrade_to_empty(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        line_bounds, line_meta = _line_bounds_and_meta(output_dir)

        assert isinstance(line_bounds, pd.DataFrame)
        assert line_bounds.empty
        assert line_meta == []


def _patch_network(
    monkeypatch: pytest.MonkeyPatch,
    *,
    id_map: DecompIdMap,
    interc_frame: pl.DataFrame,
    cobre_line_means: pl.DataFrame,
    cobre_line_pct: pl.DataFrame | None = None,
) -> None:
    """Stub the ticket-008 line seam: the deck's id map, its interchange
    table, and Cobre's own per-line simulation output -- mirroring
    ``_patch_aligned_frames``'s "patch at the seam" convention."""
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._build_line_id_map",
        lambda *_args, **_kwargs: id_map,
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_dec_oper_interc",
        lambda *_args, **_kwargs: interc_frame,
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers.read_cobre_line_means",
        lambda *_args, **_kwargs: cobre_line_means,
    )
    if cobre_line_pct is not None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_line_percentiles",
            lambda *_args, **_kwargs: cobre_line_pct,
        )


class TestBuildDecompDatasetNetwork:
    """ticket-008: line rows + Network tab metadata in
    ``build_decomp_dataset``, on top of ticket-002's ``_aligned_fixture``."""

    def _case_dirs(self, tmp_path: Path) -> tuple[Path, Path]:
        """A deck dir and a converted-case ``output/`` dir, isolated under
        *tmp_path* -- unlike ``build_decomp_dataset(tmp_path, tmp_path)``,
        ``case_dir_for(cobre_output_dir)`` must resolve to a real directory
        this test controls, since ``_line_bounds_and_meta`` reads
        ``system/``/``constraints/`` from it."""
        case_dir = tmp_path / "case"
        output_dir = _write_lines_json(case_dir, [_line_entry(0, 0, 1, name="SE-S")])
        _write_line_bounds_parquet(
            case_dir,
            [
                {
                    "line_id": 0,
                    "stage_id": 0,
                    "block_id": None,
                    "direct_mw": 1200.0,
                    "reverse_mw": 800.0,
                }
            ],
        )
        return tmp_path / "deck", output_dir

    def test_line_rows_and_summary_present_with_expected_sources(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)

        line_rows = dataset.tidy.filter(pl.col("entity_type") == "line")
        assert not line_rows.is_empty()
        assert set(line_rows["variable"].unique().to_list()) == {"net_flow_mw"}
        assert set(line_rows["source"].unique().to_list()) <= {"newave", "cobre"}
        assert "net_flow_mw" in dataset.summary["variable"].to_list()

    def test_line_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
            cobre_line_pct=pl.DataFrame(
                {
                    "entity_id": [0],
                    "stage_id": [0],
                    "net_flow_mw_p10": [200.0],
                    "net_flow_mw_p50": [240.0],
                    "net_flow_mw_p90": [280.0],
                }
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)

        line_pct = dataset.metadata["line"]
        assert isinstance(line_pct, pl.DataFrame)
        assert not line_pct.is_empty()
        assert {"net_flow_mw_p10", "net_flow_mw_p50", "net_flow_mw_p90"}.issubset(
            set(line_pct.columns)
        )

    def test_line_percentiles_stay_empty_when_cobre_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock: no ``simulation/exchanges`` partition under
        the case's output dir, so the reader degrades to its own empty
        default and no band is fabricated (deterministic-tree caveat)."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)

        line_pct = dataset.metadata["line"]
        assert isinstance(line_pct, pl.DataFrame)
        assert line_pct.is_empty()

    def test_line_bounds_is_pandas_with_the_expected_columns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)

        line_bounds = dataset.metadata["line_bounds"]
        assert isinstance(line_bounds, pd.DataFrame)
        assert not isinstance(line_bounds, pl.DataFrame)
        assert {"line_id", "stage_id", "direct_mw", "reverse_mw"}.issubset(
            set(line_bounds.columns)
        )
        assert line_bounds.iloc[0]["direct_mw"] == 1200.0

    def test_line_meta_has_the_nested_capacity_shape(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)

        line_meta = dataset.metadata["line_meta"]
        assert isinstance(line_meta, list)
        assert line_meta[0]["id"] == 0
        assert line_meta[0]["capacity"]["direct_mw"] == 1000.0
        assert line_meta[0]["capacity"]["reverse_mw"] == 1000.0
        assert line_meta[0]["name"] == "SE-S"

    def test_network_tab_renders_in_the_html_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=2, origem_mw=250.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)
        html = build_comparison_report(dataset)

        assert "Line Net Flow" in html
        assert "No line interchange data available." not in html
        assert "Plotly.newPlot" in html

    def test_unresolved_corridors_surface_in_unmapped_line(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        decomp_dir, cobre_output_dir = self._case_dirs(tmp_path)
        # NE (code 3) has no realizing line in this case's lines.json.
        _patch_network(
            monkeypatch,
            id_map=_decomp_id_map_three_subsystems(),
            interc_frame=_dec_oper_interc_frame(de=1, para=3, origem_mw=90.0),
            cobre_line_means=pl.DataFrame(
                {"entity_id": [0], "stage_id": [0], "net_flow_mw": [240.0]}
            ),
        )

        dataset = build_decomp_dataset(decomp_dir, cobre_output_dir)

        assert dataset.metadata["unmapped"]["line"] == [[1, 3]]
        line_rows = dataset.tidy.filter(pl.col("entity_type") == "line")
        assert line_rows.is_empty()


class TestSystemTabMetadata:
    """ticket-005: the System tab's cobre bus percentile band + the
    exclusion of the transhipment bus from ``results`` bus rows."""

    def _bus_percentiles(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "spot_price_p10": [40.0],
                "spot_price_p50": [44.0],
                "spot_price_p90": [48.0],
                "deficit_mw_p10": [0.0],
                "deficit_mw_p50": [0.0],
                "deficit_mw_p90": [0.0],
            }
        )

    def test_bus_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_bus_percentiles",
            lambda *_args, **_kwargs: self._bus_percentiles(),
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        bus_pct = dataset.metadata["bus"]
        assert isinstance(bus_pct, pl.DataFrame)
        assert not bus_pct.is_empty()
        assert {
            "spot_price_p10",
            "spot_price_p50",
            "spot_price_p90",
            "deficit_mw_p10",
            "deficit_mw_p50",
            "deficit_mw_p90",
        }.issubset(set(bus_pct.columns))

    def test_bus_percentiles_stay_empty_when_cobre_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock: ``tmp_path`` has no ``simulation/buses``
        partition, so ``read_cobre_bus_percentiles`` degrades to its own
        empty-frame default and the dataset must not fabricate a band."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        bus_pct = dataset.metadata["bus"]
        assert isinstance(bus_pct, pl.DataFrame)
        assert bus_pct.is_empty()

    def test_system_tab_renders_with_the_percentile_band(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_bus_percentiles",
            lambda *_args, **_kwargs: self._bus_percentiles(),
        )
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        html = build_comparison_report(dataset)

        assert "Spot Price by Bus" in html
        assert "Deficit" in html
        assert "No spot_price data available." not in html
        assert "No deficit_mw data available." not in html
        assert "Plotly.newPlot" in html

    def test_system_tab_renders_without_a_band_when_percentiles_are_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No band, no error: the System tab still renders both sections
        from the ``results`` bus rows alone."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        html = build_comparison_report(dataset)

        assert "Spot Price by Bus" in html
        assert "Deficit" in html
        assert "No spot_price data available." not in html
        assert "No deficit_mw data available." not in html
        assert "Plotly.newPlot" in html


def _dec_oper_sist_frame() -> pl.DataFrame:
    """Two submarkets, one stage, two nodes (exercises scenario averaging),
    carrying the raw ``dec_oper_sist`` columns ``_energy_balance_frames``
    reads."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 1],
            "no": [1, 2, 1, 2],
            "patamar": [None, None, None, None],
            "codigo_submercado": [1, 1, 2, 2],
            "demanda_MW": [1000.0, 1000.0, 500.0, 500.0],
            "geracao_hidroeletrica_MW": [600.0, 620.0, 300.0, 300.0],
            "geracao_termica_MW": [200.0, 200.0, 100.0, 100.0],
            "geracao_termica_antecipada_MW": [50.0, 50.0, 0.0, 0.0],
            "geracao_eolica_MW": [30.0, 30.0, 10.0, 10.0],
            "geracao_pequenas_usinas_MW": [20.0, 20.0, 5.0, 5.0],
            "deficit_MW": [0.0, 0.0, 0.0, 0.0],
            "ena_MWmes": [1200.0, 1200.0, 400.0, 400.0],
            "earm_final_MWmes": [5000.0, 5000.0, 2000.0, 2000.0],
        }
    )


class TestEnergyBalanceFrames:
    """``_energy_balance_frames`` -- ticket-006's Energy Balance tab
    reference frames, built from ``dec_oper_sist``'s stage-aggregate rows."""

    def _bus_codes(self) -> dict[int, int]:
        return {1: 0, 2: 1}

    def _patch_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_sist",
            lambda *_args, **_kwargs: _dec_oper_sist_frame(),
        )

    def test_nw_market_carries_only_the_tokens_the_tab_consumes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        nw_market, _nw_net_load, _nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        tab_tokens = {nw_var for _, nw_var, _, _ in _BALANCE_VARS if nw_var}
        emitted = set(nw_market["variable"].unique().to_list())
        assert emitted, "fixture must exercise real GHTOT/GTERM/DEFT rows"
        assert emitted <= tab_tokens
        assert emitted == {"GHTOT", "GTERM", "DEFT"}
        assert "EXCESSO" not in emitted

    def test_ghtot_gterm_deft_values_use_the_mapped_cobre_bus_id(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        nw_market, _nw_net_load, _nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        by_bus = {
            (row["newave_code"], row["variable"]): row["value"]
            for row in nw_market.iter_rows(named=True)
        }
        # Submarket 1 -> cobre bus 0: hydro gen averaged over the two nodes
        # ((600+620)/2), GTERM = live (200) + anticipated (50).
        assert by_bus[(0, "GHTOT")] == pytest.approx(610.0)
        assert by_bus[(0, "GTERM")] == pytest.approx(250.0)
        assert by_bus[(0, "DEFT")] == pytest.approx(0.0)
        # Submarket 2 -> cobre bus 1.
        assert by_bus[(1, "GHTOT")] == pytest.approx(300.0)
        assert by_bus[(1, "GTERM")] == pytest.approx(100.0)
        assert by_bus[(1, "DEFT")] == pytest.approx(0.0)

    def test_stage_stays_one_based(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        nw_market, nw_net_load, nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        assert set(nw_market["stage"].unique().to_list()) == {1}
        assert set(nw_net_load["stage"].unique().to_list()) == {1}
        assert set(nw_sin["stage"].unique().to_list()) == {1}

    def test_net_load_subtracts_wind_and_small_plants_from_demand(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        _nw_market, nw_net_load, _nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        assert set(nw_net_load["variable"].unique().to_list()) == {"NET_LOAD"}
        by_bus = {
            row["newave_code"]: row["value"]
            for row in nw_net_load.iter_rows(named=True)
        }
        assert by_bus[0] == pytest.approx(1000.0 - 30.0 - 20.0)
        assert by_bus[1] == pytest.approx(500.0 - 10.0 - 5.0)

    def test_nw_sin_sums_earmf_and_ena_across_every_submarket(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_source(monkeypatch)

        _nw_market, _nw_net_load, nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        by_var = {row["variable"]: row["value"] for row in nw_sin.iter_rows(named=True)}
        assert by_var["EARMF"] == pytest.approx(5000.0 + 2000.0)
        assert by_var["ENA"] == pytest.approx(1200.0 + 400.0)
        # The constant SIN placeholder, matching read_medias_sin's convention.
        assert set(nw_sin["newave_code"].unique().to_list()) == {0}

    def test_transhipment_bus_never_appears(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The converter-created transhipment bus has no ``codigo_submercado``
        of its own -- ``bus_codes`` (``{code: id_map.bus_id(code) for code in
        id_map.bus_codes}``) only ever holds Cobre ids in
        ``range(len(bus_codes))``, one short of ``transhipment_bus_id`` -- so
        it structurally cannot appear in ``newave_code``. Regression guard,
        mirroring ``TestBusSideExcludesTranshipment``."""
        id_map = DecompIdMap(bus_codes=(1, 2), bus_names=("SUDESTE", "SUL"))
        bus_codes = {code: id_map.bus_id(code) for code in id_map.bus_codes}
        self._patch_source(monkeypatch)

        nw_market, nw_net_load, _nw_sin = _energy_balance_frames(tmp_path, bus_codes)

        assert id_map.transhipment_bus_id not in nw_market["newave_code"].to_list()
        assert id_map.transhipment_bus_id not in nw_net_load["newave_code"].to_list()

    def test_empty_source_returns_empty_typed_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_sist",
            lambda *_args, **_kwargs: pl.DataFrame(),
        )

        nw_market, nw_net_load, nw_sin = _energy_balance_frames(
            tmp_path, self._bus_codes()
        )

        for frame in (nw_market, nw_net_load, nw_sin):
            assert frame.is_empty()
            assert frame.columns == ["newave_code", "stage", "variable", "value"]


def _balance_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture`` extended with ticket-006's Energy Balance
    reference frames, keyed to the same bus (cobre id 0, name "SE")."""
    nw_market = pl.DataFrame(
        {
            "newave_code": [0, 0, 0],
            "stage": [1, 1, 1],
            "variable": ["GHTOT", "GTERM", "DEFT"],
            "value": [600.0, 250.0, 0.0],
        }
    )
    nw_net_load = pl.DataFrame(
        {
            "newave_code": [0],
            "stage": [1],
            "variable": ["NET_LOAD"],
            "value": [950.0],
        }
    )
    nw_sin = pl.DataFrame(
        {
            "newave_code": [0, 0],
            "stage": [1, 1],
            "variable": ["EARMF", "ENA"],
            "value": [7000.0, 1600.0],
        }
    )
    return dataclasses.replace(
        _aligned_fixture(), nw_market=nw_market, nw_net_load=nw_net_load, nw_sin=nw_sin
    )


def _bus_aggregates_fixture() -> pl.DataFrame:
    """Per-bus Cobre percentile aggregates for all five ``_BALANCE_VARS``
    quantities -- the shape :func:`cobre_readers.read_cobre_bus_aggregates`
    returns."""
    return pl.DataFrame(
        {
            "bus_id": [0],
            "stage_id": [0],
            "hydro_gen_mw_p10": [580.0],
            "hydro_gen_mw_p50": [600.0],
            "hydro_gen_mw_p90": [620.0],
            "thermal_gen_mw_p10": [240.0],
            "thermal_gen_mw_p50": [250.0],
            "thermal_gen_mw_p90": [260.0],
            "net_load_mw_p10": [900.0],
            "net_load_mw_p50": [950.0],
            "net_load_mw_p90": [1000.0],
            "deficit_mw_p10": [0.0],
            "deficit_mw_p50": [0.0],
            "deficit_mw_p90": [0.0],
            "excess_mw_p10": [0.0],
            "excess_mw_p50": [0.0],
            "excess_mw_p90": [0.0],
        }
    )


def _cobre_hydro_means_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [0],
            "stored_energy_final_mwh": [5100000.0],
            "incremental_inflow_energy_mw": [1550.0],
        }
    )


class TestBuildDecompDatasetEnergyBalance:
    """ticket-006: Energy Balance tab metadata (demand / gen-by-source /
    EARM / ENA) filled by ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _balance_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_bus_aggregates",
            lambda *_args, **_kwargs: _bus_aggregates_fixture(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_bus_metadata",
            lambda *_args, **_kwargs: {0: {"name": "SE"}},
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_hydro_means",
            lambda *_args, **_kwargs: _cobre_hydro_means_fixture(),
        )

    def test_metadata_keys_are_present_and_typed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        for key in (
            "nw_market",
            "nw_net_load",
            "bus_aggregates",
            "cobre_hydro_means",
            "nw_sin",
        ):
            value = dataset.metadata[key]
            assert isinstance(value, pl.DataFrame)
            assert not value.is_empty()
        assert isinstance(dataset.metadata["cobre_bus_meta"], dict)
        assert dataset.metadata["cobre_bus_meta"]
        # D-STAGE-OFFSET: fixed at 1 for DECOMP's 1-based estagio.
        assert dataset.metadata["nw_offset"] == 1

    def test_nw_market_tokens_are_all_consumed_by_the_tab(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        tab_tokens = {nw_var for _, nw_var, _, _ in _BALANCE_VARS if nw_var}
        emitted = set(dataset.metadata["nw_market"]["variable"].unique().to_list())
        assert emitted
        assert emitted <= tab_tokens

    def test_nw_sin_earm_ena_sums_match_the_fixture(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_sin = dataset.metadata["nw_sin"]
        earmf = nw_sin.filter(pl.col("variable") == "EARMF")["value"].to_list()
        ena = nw_sin.filter(pl.col("variable") == "ENA")["value"].to_list()
        assert earmf == pytest.approx([7000.0])
        assert ena == pytest.approx([1600.0])

    def test_excess_panel_renders_cobre_only_with_no_fabricated_newave_row(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """DECOMP has no energy-excess quantity (see
        ``_energy_balance_frames``'s docstring): EXCESSO must never appear in
        ``nw_market`` (no dead row), while the tab's Excess panel still
        renders using Cobre data alone."""
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert "EXCESSO" not in dataset.metadata["nw_market"]["variable"].to_list()

        html = build_comparison_report(dataset)

        assert "Excess" in html

    def test_report_renders_energy_balance_tab_and_system_energy_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        html = build_comparison_report(dataset)

        assert "Hydro Generation" in html
        assert "Thermal Generation" in html
        assert "Net Load" in html
        assert "Deficit" in html
        assert "System Energy (EARM / ENA)" in html
        # The DECOMP overlay line on the SIN EARM/ENA charts.
        assert "NEWAVE SIN" in html
        assert "Plotly.newPlot" in html


_REDUCED_DECOMP_DECK = Path("example/decomp-mar-26-rv2-reduced")
_REDUCED_COBRE_OUTPUT = Path("example/cobre-mar-26-rv2-reduced/output")


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetEnergyBalanceE2E:
    """Tier 3 (dev-only smoke): the reduced deck's real Energy Balance tab
    renders end to end. Both directories are gitignored, so this never runs
    in CI."""

    def test_energy_balance_tab_is_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert "No energy balance data available." not in html
        assert "Plotly.newPlot" in html


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetNetworkE2E:
    """Tier 3 (dev-only smoke, ticket-008): the reduced deck's real Network
    tab renders end to end. Both directories are gitignored, so this never
    runs in CI."""

    def test_network_tab_is_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert "Line Net Flow" in html
        assert "Plotly.newPlot" in html


def _relato_costs_frame() -> pl.DataFrame:
    """Two stages, two scenarios each -- known k$ values that pin the exact
    NPV dict and per-stage ``nw_sin`` magnitudes ticket-010 must produce.

    Per-stage means (across ``cenario``): stage 1 -> geracao_termica=120,
    custo_presente=600, custo_futuro=2200; stage 2 -> geracao_termica=100,
    custo_presente=400, custo_futuro=1200. NPV (summed across stage, x1e3):
    GERACAO TERMICA=220_000, INTERCAMBIO=8_000, VERTIMENTO=12_000,
    VIOL. TURB. MINIMO=2_000, VIOL. TURB. MAXIMO=4_000,
    VIOLACAO DESVIO=16_000.
    """
    return pl.DataFrame(
        {
            "estagio": [1, 1, 2, 2],
            "cenario": [1, 2, 1, 2],
            "probabilidade": [0.5, 0.5, 0.5, 0.5],
            "custo_presente": [500.0, 700.0, 300.0, 500.0],
            "custo_futuro": [2000.0, 2400.0, 1000.0, 1400.0],
            "geracao_termica": [100.0, 140.0, 80.0, 120.0],
            "violacao_desvio": [10.0, 10.0, 6.0, 6.0],
            "penalidade_vertimento_reservatorio": [5.0, 5.0, 3.0, 3.0],
            "penalidade_vertimento_fio": [3.0, 3.0, 1.0, 1.0],
            "violacao_turbinamento_reservatorio": [1.0, 1.0, 1.0, 1.0],
            "violacao_turbinamento_fio": [2.0, 2.0, 2.0, 2.0],
            "penalidade_intercambio": [4.0, 4.0, 4.0, 4.0],
        }
    )


class TestCostFrames:
    """ticket-010: ``_cost_frames`` -- the DECOMP-side NPV dict (R$) + the
    per-stage ``nw_sin`` cost rows (10^6 R$), reconciled from native k$."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_costs",
            lambda *_args, **_kwargs: _relato_costs_frame(),
        )

    def test_kdollars_to_reais_reconciliation_on_both_unit_paths(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The epic's TOP RISK guard, pinned in one place: a known k$ input
        must land at the correct magnitude on BOTH the ``nw_costs`` dict (R$,
        x1e3) and the ``nw_sin`` CTERM per-stage rows (10^6 R$, /1e3).
        ``geracao_termica`` aggregates to 220.0 k$ NPV (120 + 100 stage
        means), so the dict must read 220_000.0 R$; each stage's CTERM row
        must read that stage's mean k$ /1e3."""
        self._patch(monkeypatch)

        nw_costs, nw_sin = _cost_frames(tmp_path)

        assert nw_costs["GERACAO TERMICA"] == pytest.approx(220_000.0)
        cterm = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "CTERM").iter_rows(
                named=True
            )
        }
        assert cterm == pytest.approx({1: 0.12, 2: 0.1})

    def test_all_cost_map_categories_are_populated_with_known_magnitudes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        nw_costs, _nw_sin = _cost_frames(tmp_path)

        assert nw_costs["INTERCAMBIO"] == pytest.approx(8_000.0)
        assert nw_costs["VERTIMENTO"] == pytest.approx(12_000.0)
        assert nw_costs["VIOL. TURB. MINIMO"] == pytest.approx(2_000.0)
        assert nw_costs["VIOL. TURB. MAXIMO"] == pytest.approx(4_000.0)

    def test_violacao_desvio_surfaces_as_a_descriptive_residual_key(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``violacao_desvio`` has no ``charts._COST_MAP`` slot -- it must
        still surface (not be silently dropped) under its own key."""
        self._patch(monkeypatch)

        nw_costs, _nw_sin = _cost_frames(tmp_path)

        assert nw_costs[_DEVIATION_VIOLATION_LABEL] == pytest.approx(16_000.0)

    def test_coper_and_custo_futuro_rows_match_the_stage_means(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        coper = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "COPER").iter_rows(
                named=True
            )
        }
        custo_futuro = {
            row["stage"]: row["value"]
            for row in nw_sin.filter(pl.col("variable") == "CUSTO_FUTURO").iter_rows(
                named=True
            )
        }
        assert coper == pytest.approx({1: 0.6, 2: 0.4})
        assert custo_futuro == pytest.approx({1: 2.2, 2: 1.2})

    def test_every_nw_costs_key_maps_to_a_known_cost_map_category(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No dead rows: every emitted key is either a ``charts._COST_MAP``
        the source model label or the one intentional residual
        (``violacao_desvio``)."""
        self._patch(monkeypatch)

        nw_costs, _nw_sin = _cost_frames(tmp_path)

        known_labels = {label for label, *_ in _NW_COST_LABELS}
        mapped_cost_map_keys = {k for _, nw_keys, _, _ in _COST_MAP for k in nw_keys}
        assert nw_costs  # the fixture must exercise real categories
        for key in nw_costs:
            assert key in known_labels
            assert key == _DEVIATION_VIOLATION_LABEL or key in mapped_cost_map_keys

    def test_every_nw_sin_variable_is_a_chart_consumed_token(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        assert set(nw_sin["variable"].unique().to_list()) <= {
            "COPER",
            "CUSTO_FUTURO",
            "CTERM",
        }

    def test_stage_column_stays_one_based(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        _nw_costs, nw_sin = _cost_frames(tmp_path)

        assert set(nw_sin["stage"].unique().to_list()) == {1, 2}

    def test_propagates_the_readers_raise_on_missing_relato(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No new swallowing: a missing/empty relato surfaces exactly the
        error ``read_relato_costs`` raises."""

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_costs", _boom
        )

        with pytest.raises(FileNotFoundError):
            _cost_frames(tmp_path)


class TestUnionCostRows:
    """ticket-010: ``_union_cost_rows`` -- additive union onto ``nw_sin``,
    defensive against the dataclass-default columnless ``pl.DataFrame()``."""

    def _typed(self, variable: str, value: float) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "newave_code": [0],
                "stage": [1],
                "variable": [variable],
                "value": [value],
            }
        )

    def test_unions_cost_rows_onto_existing_rows(self) -> None:
        earm = self._typed("EARMF", 7000.0)
        cost = self._typed("CTERM", 0.12)

        combined = _union_cost_rows(earm, cost)

        assert set(combined["variable"].to_list()) == {"EARMF", "CTERM"}

    def test_columnless_nw_sin_default_returns_cost_rows_unchanged(self) -> None:
        cost = self._typed("CTERM", 0.12)

        combined = _union_cost_rows(pl.DataFrame(), cost)

        assert combined is cost

    def test_columnless_cost_rows_returns_nw_sin_unchanged(self) -> None:
        earm = self._typed("EARMF", 7000.0)

        combined = _union_cost_rows(earm, pl.DataFrame())

        assert combined is earm


def _cobre_cost_breakdown_fixture() -> dict[str, float]:
    return {"thermal_cost": 200_000.0, "deficit_cost": 1_000.0}


def _cobre_stage_costs_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stage_id": [0, 1],
            "immediate_cost": [600_000.0, 400_000.0],
            "future_cost": [2_200_000.0, 1_200_000.0],
            "thermal_cost": [110_000.0, 95_000.0],
            "anticipated_thermal_cost": [10_000.0, 5_000.0],
            "thermal_cost_total": [120_000.0, 100_000.0],
        }
    )


class TestBuildDecompDatasetCosts:
    """ticket-010: Overview cost metadata (nw_costs/cobre_costs/nw_sin cost
    rows/cobre_stage_costs) filled by ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _balance_fixture())
        # ``_patch_aligned_frames`` stubs ``_cost_frames`` itself to an
        # empty default (see its own docstring) -- re-point it back to the
        # real function (the module-level name this test file imported,
        # unaffected by that stub) so patching ``read_relato_costs`` below
        # actually takes effect through it.
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._cost_frames", _cost_frames
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_costs",
            lambda *_args, **_kwargs: _relato_costs_frame(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_cost_breakdown",
            lambda *_args, **_kwargs: _cobre_cost_breakdown_fixture(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_stage_costs",
            lambda *_args, **_kwargs: _cobre_stage_costs_fixture(),
        )

    def test_nw_sin_retains_earm_ena_rows_alongside_the_new_cost_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Additive union, not overwrite: the ticket-006 EARM/ENA rows must
        survive the ticket-010 cost-row union."""
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        variables = set(dataset.metadata["nw_sin"]["variable"].unique().to_list())
        assert {"EARMF", "ENA"} <= variables
        assert {"COPER", "CUSTO_FUTURO", "CTERM"} <= variables

    def test_nw_costs_cobre_costs_stage_costs_and_offset_are_populated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["nw_costs"]["GERACAO TERMICA"] == pytest.approx(
            220_000.0
        )
        assert dataset.metadata["cobre_costs"]["thermal_cost"] == pytest.approx(
            200_000.0
        )
        stage_costs = dataset.metadata["cobre_stage_costs"]
        assert isinstance(stage_costs, pl.DataFrame)
        assert not stage_costs.is_empty()
        assert dataset.metadata["nw_offset"] == 1

    def test_overview_cost_sections_render_non_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Cost Breakdown" in html
        assert "Per-Stage Cost" in html
        assert "No cost data available." not in html
        assert "NEWAVE Thermal Cost" in html
        assert "Cobre Thermal Cost" in html
        assert "Plotly.newPlot" in html


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetCostsE2E:
    """Tier 3 (dev-only smoke, ticket-010): the reduced deck's real Overview
    cost sections render end to end. Both directories are gitignored, so
    this never runs in CI."""

    def test_overview_cost_sections_are_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert "No cost data available." not in html
        assert "Plotly.newPlot" in html


def _relato_convergence_frame() -> pl.DataFrame:
    """Three iterations of the source model's own convergence table.

    Carries ``gap_percentual``/``tempo`` alongside ``iteracao``/``zinf``/
    ``zsup`` so a test can pin that ticket-012 selects only the three chart
    columns and drops the rest (out of scope per the ticket)."""
    return pl.DataFrame(
        {
            "iteracao": [1, 2, 3],
            "zinf": [100.0, 150.0, 180.0],
            "zsup": [500.0, 300.0, 190.0],
            "gap_percentual": [400.0, 100.0, 5.0],
            "tempo": [12.0, 15.0, 9.0],
        }
    )


def _cobre_convergence_fixture() -> pl.DataFrame:
    """Cobre's own canonical convergence schema, straight from
    ``read_cobre_convergence``'s contract."""
    return pl.DataFrame(
        {
            "iteration": [1, 2, 3],
            "lower_bound": [95.0, 145.0, 178.0],
            "upper_bound_mean": [520.0, 310.0, 192.0],
        },
        schema={
            "iteration": pl.Int64,
            "lower_bound": pl.Float64,
            "upper_bound_mean": pl.Float64,
        },
    )


class TestDecompConvergenceFrame:
    """ticket-012: ``_decomp_convergence_frame`` -- renames the source
    model's ``relato.convergencia`` onto the canonical ``iteration``/
    ``lower_bound``/``upper_bound_mean`` schema ``read_cobre_convergence``
    emits, so ``convergence_chart`` can read both sides without a
    source-specific branch."""

    _CANONICAL_SCHEMA = {
        "iteration": pl.Int64,
        "lower_bound": pl.Float64,
        "upper_bound_mean": pl.Float64,
    }

    def _patch(self, monkeypatch: pytest.MonkeyPatch, frame: pl.DataFrame) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence",
            lambda *_args, **_kwargs: frame,
        )

    def test_renames_and_casts_onto_the_canonical_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch, _relato_convergence_frame())

        frame = _decomp_convergence_frame(tmp_path)

        assert frame.columns == ["iteration", "lower_bound", "upper_bound_mean"]
        assert frame.schema == self._CANONICAL_SCHEMA
        assert frame["iteration"].to_list() == [1, 2, 3]
        assert frame["lower_bound"].to_list() == pytest.approx([100.0, 150.0, 180.0])
        assert frame["upper_bound_mean"].to_list() == pytest.approx(
            [500.0, 300.0, 190.0]
        )

    def test_gap_and_timing_columns_are_not_carried_over(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Out of scope per the ticket: ``gap_percentual``/``tempo`` are not
        consumed by ``convergence_chart``, so they must not leak through."""
        self._patch(monkeypatch, _relato_convergence_frame())

        frame = _decomp_convergence_frame(tmp_path)

        assert "gap_percentual" not in frame.columns
        assert "tempo" not in frame.columns

    def test_missing_relato_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence", _boom
        )

        frame = _decomp_convergence_frame(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA

    def test_empty_relato_table_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("relato.rv0 has no convergencia table")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence", _boom
        )

        frame = _decomp_convergence_frame(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA


class TestBuildDecompDatasetConvergence:
    """ticket-012: the Overview tab's Convergence overlay
    (``nw_convergence``/``cobre_convergence``) filled by
    ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence",
            lambda *_args, **_kwargs: _relato_convergence_frame(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_convergence",
            lambda *_args, **_kwargs: _cobre_convergence_fixture(),
        )

    def test_nw_convergence_matches_the_source_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_conv = dataset.metadata["nw_convergence"]
        assert nw_conv.columns == ["iteration", "lower_bound", "upper_bound_mean"]
        assert nw_conv["iteration"].to_list() == [1, 2, 3]
        assert nw_conv["lower_bound"].to_list() == pytest.approx([100.0, 150.0, 180.0])
        assert nw_conv["upper_bound_mean"].to_list() == pytest.approx(
            [500.0, 300.0, 190.0]
        )

    def test_cobre_convergence_matches_the_reader_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        cobre_conv = dataset.metadata["cobre_convergence"]
        assert cobre_conv.columns == ["iteration", "lower_bound", "upper_bound_mean"]
        assert cobre_conv["lower_bound"].to_list() == pytest.approx(
            [95.0, 145.0, 178.0]
        )
        assert cobre_conv["upper_bound_mean"].to_list() == pytest.approx(
            [520.0, 310.0, 192.0]
        )

    def test_report_renders_the_convergence_section_and_both_traces(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Convergence" in html
        assert "NEWAVE ZINF" in html
        assert "Cobre Lower" in html
        assert "Plotly.newPlot" in html

    def test_absent_decomp_convergence_yields_an_empty_canonical_frame_no_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A missing relato must not abort the dataset build; the chart
        degrades to a Cobre-only overlay."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence", _boom
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_convergence",
            lambda *_args, **_kwargs: _cobre_convergence_fixture(),
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_conv = dataset.metadata["nw_convergence"]
        assert nw_conv.is_empty()
        assert nw_conv.columns == ["iteration", "lower_bound", "upper_bound_mean"]

        html = build_comparison_report(dataset)
        assert "Convergence" in html
        assert "Cobre Lower" in html


class TestBuildDecompDatasetParityWithLegacyComparison:
    """No drift from the migration: on the same aligned fixture,
    ``build_decomp_dataset`` and the legacy ``compare_decomp_results`` must
    report the same per-(entity, stage) newave/cobre values."""

    def test_every_legacy_row_has_a_matching_dataset_pair(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        legacy = compare_decomp_results(tmp_path, tmp_path)
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert legacy.rows.height > 0  # the fixture must exercise real rows
        for row in legacy.rows.iter_rows(named=True):
            canonical = _CANONICAL_VARIABLE[(row["level"], row["variable"])]
            match = dataset.tidy.filter(
                (pl.col("entity_type") == row["level"])
                & (pl.col("variable") == canonical)
                & (pl.col("entity_id") == row["entity_id"])
                & (pl.col("stage") == row["stage_id"])
            )
            newave_value = match.filter(pl.col("source") == "newave")["value"].to_list()
            cobre_value = match.filter(pl.col("source") == "cobre")["value"].to_list()
            assert newave_value == pytest.approx([row["source"]])
            assert cobre_value == pytest.approx([row["cobre"]])

    def test_unmapped_dict_is_identical_between_entry_points(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``build_decomp_dataset`` and the legacy comparison must agree on
        every level ``_read_aligned_frames`` reports. ``"line"``/``"ree"``/
        ``"evaporation"`` are ticket-008/ticket-018/ticket-020 additions
        sourced outside ``_read_aligned_frames`` (the legacy comparison never
        gained line, REE, or evaporation rows), so they are compared
        separately rather than folded into the shared-levels equality."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        legacy = compare_decomp_results(tmp_path, tmp_path)
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset_unmapped = dict(dataset.metadata["unmapped"])
        assert dataset_unmapped.pop("line") == []
        assert dataset_unmapped.pop("ree") == []
        assert dataset_unmapped.pop("evaporation") == []
        assert dataset_unmapped == legacy.unmapped


# ---------------------------------------------------------------------------
# ticket-013: Performance tab timing metadata (Caveat #2 -- no fabricated
# DECOMP forward/backward split).
# ---------------------------------------------------------------------------


def _decomp_tim_frame() -> pl.DataFrame:
    """``read_decomp_tim``'s shape: ``Etapa``/``Tempo`` (timedelta), matching
    the real reduced deck's ``decomp.tim`` (``Leitura de Dados`` 8s,
    ``Convergencia`` 2m10s, ``Impressao`` 3s, ``Tempo Total`` 2m21s)."""
    return pl.DataFrame(
        {
            "Etapa": ["Leitura de Dados", "Convergencia", "Impressao", "Tempo Total"],
            "Tempo": [
                timedelta(seconds=8),
                timedelta(minutes=2, seconds=10),
                timedelta(seconds=3),
                timedelta(minutes=2, seconds=21),
            ],
        }
    )


class TestDecompTimStages:
    """``_decomp_tim_stages``: ``decomp.tim`` ``Etapa`` phase totals mapped
    onto the two keys ``performance_metric_cards`` reads."""

    def test_maps_the_two_etapa_rows_onto_the_shared_card_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_decomp_tim",
            lambda *_args, **_kwargs: _decomp_tim_frame(),
        )

        stages = _decomp_tim_stages(tmp_path)

        assert stages == {"Tempo Total": 141.0, "Calculo da Politica": 130.0}

    def test_missing_convergencia_phase_falls_back_to_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A truncated ``decomp.tim`` missing the ``Convergencia`` row still
        yields both keys -- ``Calculo da Politica`` defaults to ``0.0``
        rather than being omitted (never KeyError downstream)."""
        table = pl.DataFrame(
            {
                "Etapa": ["Leitura de Dados", "Tempo Total"],
                "Tempo": [timedelta(seconds=8), timedelta(minutes=2, seconds=21)],
            }
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_decomp_tim",
            lambda *_args, **_kwargs: table,
        )

        stages = _decomp_tim_stages(tmp_path)

        assert stages == {"Tempo Total": 141.0, "Calculo da Politica": 0.0}

    def test_missing_decomp_tim_degrades_to_an_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("decomp.tim not found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_decomp_tim", _boom
        )

        assert _decomp_tim_stages(tmp_path) == {}

    def test_empty_decomp_tim_value_error_degrades_to_an_empty_dict(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("decomp.tim parsed empty")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_decomp_tim", _boom
        )

        assert _decomp_tim_stages(tmp_path) == {}


class TestDecompTimIterations:
    """``_decomp_tim_iterations``: ``relato.convergencia``'s per-iteration
    ``tempo`` renamed onto the canonical ``iteration``/``total_seconds``
    schema -- Caveat #2: never a forward/backward split."""

    _CANONICAL_SCHEMA = {"iteration": pl.Int64, "total_seconds": pl.Float64}

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence",
            lambda *_args, **_kwargs: _relato_convergence_frame(),
        )

    def test_renames_iteracao_and_tempo_onto_the_canonical_schema(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        frame = _decomp_tim_iterations(tmp_path)

        assert frame.columns == ["iteration", "total_seconds"]
        assert frame.schema == self._CANONICAL_SCHEMA
        assert frame["iteration"].to_list() == [1, 2, 3]
        assert frame["total_seconds"].to_list() == pytest.approx([12.0, 15.0, 9.0])

    def test_no_forward_or_backward_columns_are_ever_added(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Caveat #2: DECOMP has no forward/backward pass structure to
        split -- the frame must carry EXACTLY the two canonical columns."""
        self._patch(monkeypatch)

        frame = _decomp_tim_iterations(tmp_path)

        assert "forward_seconds" not in frame.columns
        assert "backward_seconds" not in frame.columns
        assert set(frame.columns) == {"iteration", "total_seconds"}

    def test_missing_relato_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence", _boom
        )

        frame = _decomp_tim_iterations(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA

    def test_empty_relato_value_error_degrades_to_an_empty_canonical_frame(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise ValueError("relato.rv0 has no convergencia table")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence", _boom
        )

        frame = _decomp_tim_iterations(tmp_path)

        assert frame.is_empty()
        assert frame.schema == self._CANONICAL_SCHEMA


class TestDecompMaxStage:
    """``_decomp_max_stage``: the max already-aligned (0-based) ``stage_id``
    across every level, derived without an extra read."""

    def test_uses_the_max_stage_id_across_every_level(self) -> None:
        aligned = dataclasses.replace(
            _aligned_fixture(),
            source_hydro=pl.DataFrame(
                {"entity_id": [0, 0], "stage_id": [0, 2], "geracao_MW": [1.0, 1.0]}
            ),
            source_thermal=pl.DataFrame(
                {"entity_id": [0, 0], "stage_id": [3, 4], "geracao_MW": [1.0, 1.0]}
            ),
            source_bus=pl.DataFrame({"entity_id": [0], "stage_id": [1], "cmo": [1.0]}),
        )

        assert _decomp_max_stage(aligned) == 4

    def test_returns_none_when_every_level_is_empty(self) -> None:
        aligned = dataclasses.replace(
            _aligned_fixture(),
            source_hydro=pl.DataFrame(),
            source_thermal=pl.DataFrame(),
            source_bus=pl.DataFrame(),
        )

        assert _decomp_max_stage(aligned) is None


class TestPerformanceFwdBwdSplitChartColumnGuard:
    """charts.py fix (approved scope expansion): ``has_nw`` must require the
    ``forward_seconds``/``backward_seconds`` columns, not just a non-empty
    frame -- a DECOMP-shaped ``iteration``/``total_seconds``-only frame must
    render Cobre-only instead of raising ``ColumnNotFoundError``."""

    _COBRE_TIMING = pl.DataFrame(
        {
            "iteration": [1, 2],
            "time_forward_ms": [1000.0, 900.0],
            "time_backward_ms": [2000.0, 1800.0],
            "time_total_ms": [3000.0, 2700.0],
        }
    )

    def test_decomp_shaped_frame_renders_cobre_only_without_raising(self) -> None:
        decomp_shaped = pl.DataFrame(
            {"iteration": [1, 2], "total_seconds": [12.0, 15.0]}
        )

        html = performance_fwd_bwd_split_chart(decomp_shaped, self._COBRE_TIMING)

        assert "NEWAVE (s)" not in html
        assert "Cobre (s)" in html

    def test_newave_shaped_frame_still_renders_its_own_panel(self) -> None:
        """Parity guard: the source model's own ``newave.tim`` frame (all
        four columns) is unaffected by the DECOMP-motivated guard change."""
        newave_shaped = pl.DataFrame(
            {
                "iteration": [1, 2],
                "backward_seconds": [20.0, 18.0],
                "forward_seconds": [10.0, 9.0],
                "total_seconds": [30.0, 27.0],
            }
        )

        html = performance_fwd_bwd_split_chart(newave_shaped, self._COBRE_TIMING)

        assert "NEWAVE (s)" in html
        assert "Cobre (s)" in html

    def test_columnless_empty_frame_still_renders_cobre_only(self) -> None:
        """The pre-ticket-013 default (an entirely empty frame) must keep
        working exactly as before -- the guard's ``is_empty()`` half."""
        html = performance_fwd_bwd_split_chart(pl.DataFrame(), self._COBRE_TIMING)

        assert "NEWAVE (s)" not in html
        assert "Cobre (s)" in html


class TestBuildDecompDatasetPerformance:
    """ticket-013: the Performance tab's timing metadata
    (``nw_tim_stages``/``nw_tim_iterations``/``nw_max_stage``/
    ``cobre_training_seconds``/``cobre_iteration_timing``) filled by
    ``build_decomp_dataset``."""

    def _patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_decomp_tim",
            lambda *_args, **_kwargs: _decomp_tim_frame(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence",
            lambda *_args, **_kwargs: _relato_convergence_frame(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_training_duration",
            lambda *_args, **_kwargs: 26.0,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_iteration_timing",
            lambda *_args, **_kwargs: pl.DataFrame(
                {
                    "iteration": [1, 2, 3],
                    "time_forward_ms": [4000.0, 3500.0, 3000.0],
                    "time_backward_ms": [6000.0, 5500.0, 5000.0],
                    "time_total_ms": [10000.0, 9000.0, 8000.0],
                }
            ),
        )

    def test_nw_tim_stages_carries_the_two_mapped_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["nw_tim_stages"] == {
            "Tempo Total": 141.0,
            "Calculo da Politica": 130.0,
        }

    def test_nw_tim_iterations_has_exactly_iteration_and_total_seconds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_tim_iterations = dataset.metadata["nw_tim_iterations"]
        assert nw_tim_iterations.columns == ["iteration", "total_seconds"]
        assert nw_tim_iterations["total_seconds"].to_list() == pytest.approx(
            [12.0, 15.0, 9.0]
        )

    def test_cobre_training_seconds_and_iteration_timing_are_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["cobre_training_seconds"] == 26.0
        cobre_iter_timing = dataset.metadata["cobre_iteration_timing"]
        assert cobre_iter_timing["time_total_ms"].to_list() == pytest.approx(
            [10000.0, 9000.0, 8000.0]
        )

    def test_nw_max_stage_is_derived_from_the_aligned_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        # ``_aligned_fixture``'s every level is single-stage, stage_id 0.
        assert dataset.metadata["nw_max_stage"] == 0

    def test_report_renders_metric_cards_and_time_per_iteration_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Time per Iteration" in html
        assert "NEWAVE Total Wall-Clock" in html
        assert "Plotly.newPlot" in html

    def test_forward_backward_split_is_cobre_only_no_decomp_trace_no_crash(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The crash-guard proof: a NON-EMPTY DECOMP ``nw_tim_iterations``
        (``iteration``/``total_seconds`` only, per Caveat #2) must not raise
        ``ColumnNotFoundError`` inside ``build_comparison_report`` -- the
        "Forward / Backward Split" section renders with a Cobre trace and no
        DECOMP forward/backward trace."""
        self._patch(monkeypatch)
        dataset = build_decomp_dataset(tmp_path, tmp_path)
        assert not dataset.metadata["nw_tim_iterations"].is_empty()

        html = build_comparison_report(dataset)  # must not raise

        content = _extract_tab_content(html, "tab-performance")
        assert "Forward / Backward Split" in content
        assert "Cobre (s)" in content
        assert "NEWAVE (s)" not in content

    def test_missing_decomp_tim_and_relato_degrades_to_empty_zero_no_raise(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        def _boom_tim(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("decomp.tim not found")

        def _boom_conv(*_args: object, **_kwargs: object) -> pl.DataFrame:
            raise FileNotFoundError("no relato.rvN found")

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_decomp_tim", _boom_tim
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_convergence",
            _boom_conv,
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)  # must not raise

        assert dataset.metadata["nw_tim_stages"] == {}
        assert dataset.metadata["nw_tim_iterations"].is_empty()

        html = build_comparison_report(dataset)  # must not raise either
        assert "Plotly.newPlot" in html


def _extract_tab_content(html: str, tab_id: str) -> str:
    """Slice out one ``<div id="tab-...">...</div>`` tab's own content.

    Mirrors ``test_chart_helpers._extract_tab_content``'s contract (locate
    the tab by its id, stop at the next ``id="tab-`` marker) -- duplicated
    locally rather than imported so this file has no cross-test-module
    dependency.
    """
    import re

    match = re.search(rf'id="{tab_id}".*?(?=id="tab-|\Z)', html, re.S)
    return match.group(0) if match else ""


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetPerformanceE2E:
    """Tier 3 (dev-only smoke, ticket-013): the reduced deck's real
    Performance tab renders end to end. Both directories are gitignored, so
    this never runs in CI."""

    def test_performance_tab_is_non_empty_on_the_reduced_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)

        assert dataset.metadata["nw_tim_stages"]
        assert not dataset.metadata["nw_tim_iterations"].is_empty()
        assert "Time per Iteration" in html
        assert "Forward / Backward Split" in html
        assert "Plotly.newPlot" in html


def _hydro_percentiles_fixture() -> pl.DataFrame:
    """Cobre p10/p50/p90 for the two ``_aligned_fixture`` hydro entities."""
    return pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw_p10": [100.0, 55.0],
            "generation_mw_p50": [110.0, 60.0],
            "generation_mw_p90": [120.0, 65.0],
            "storage_final_hm3_p10": [470.0, 290.0],
            "storage_final_hm3_p50": [480.0, 300.0],
            "storage_final_hm3_p90": [490.0, 310.0],
        }
    )


def _hydro_metadata_fixture() -> dict[int, dict]:
    return {
        0: {"name": "A", "min_storage_hm3": 20.0},
        1: {"name": "B", "min_storage_hm3": 10.0},
    }


def _hydro_bus_labels_fixture() -> dict[int, frozenset[int]]:
    return {0: frozenset({0}), 1: frozenset({0})}


def _hydro_per_stage_bounds_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "min_storage_hm3": [20.0, 10.0],
            "max_storage_hm3": [1000.0, 600.0],
        }
    )


def _patch_hydro_detail_readers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    percentiles: pl.DataFrame | None = None,
    metadata: dict[int, dict] | None = None,
    bus_labels: dict[int, frozenset[int]] | None = None,
    per_stage_bounds: pl.DataFrame | None = None,
) -> None:
    """Stub ticket-014's four cobre readers, each defaulting to empty --
    matching how a Cobre run with no hydro percentile/metadata output
    (e.g. the deterministic 2-node tree) degrades in production."""
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_hydro_percentiles",
        lambda *_a, **_k: pl.DataFrame() if percentiles is None else percentiles,
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_hydro_metadata",
        lambda *_a, **_k: {} if metadata is None else metadata,
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_hydro_bus_labels",
        lambda *_a, **_k: {} if bus_labels is None else bus_labels,
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_hydro_per_stage_bounds",
        lambda *_a, **_k: (
            pl.DataFrame() if per_stage_bounds is None else per_stage_bounds
        ),
    )


class TestMergeHydroBusIds:
    """ticket-014's ``bus_ids`` merge helper -- the per-bus hydro charts
    KeyError without it (see ``analyze._bus_name_lookups``)."""

    def test_injects_bus_ids_from_labels(self) -> None:
        meta = {0: {"name": "A"}, 1: {"name": "B"}}
        labels = {0: frozenset({7})}

        merged = _merge_hydro_bus_ids(meta, labels)

        assert merged[0]["bus_ids"] == frozenset({7})
        assert merged[1]["bus_ids"] == []

    def test_does_not_mutate_the_inputs(self) -> None:
        meta = {0: {"name": "A"}}
        labels = {0: frozenset({7})}

        merged = _merge_hydro_bus_ids(meta, labels)
        merged[0]["name"] = "changed"
        merged[0]["bus_ids"] = frozenset({99})

        assert meta[0] == {"name": "A"}
        assert labels[0] == frozenset({7})

    def test_every_plant_gets_a_bus_ids_key_even_with_no_labels(self) -> None:
        meta = {5: {"name": "C"}}

        merged = _merge_hydro_bus_ids(meta, {})

        assert merged[5]["bus_ids"] == []


class TestBuildDecompDatasetHydroDetail:
    """ticket-014: the Hydro Operation + Hydro Plant Details tabs' four
    remaining ``PercentileData`` fields (``hydro``, ``cobre_hydro_meta``,
    ``cobre_hydro_per_stage_bounds``, ``nw_hydro_slacks``)."""

    def test_hydro_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch, percentiles=_hydro_percentiles_fixture()
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        hydro_pct = dataset.metadata["hydro"]
        assert isinstance(hydro_pct, pl.DataFrame)
        assert not hydro_pct.is_empty()
        assert "generation_mw_p50" in hydro_pct.columns
        assert "storage_final_hm3_p50" in hydro_pct.columns

    def test_hydro_percentiles_stay_empty_when_cobre_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock (deterministic-tree low-N, master-plan caveat
        1): ``read_cobre_hydro_percentiles`` degrades to its own empty-frame
        default and the dataset must not fabricate a spread."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        hydro_pct = dataset.metadata["hydro"]
        assert isinstance(hydro_pct, pl.DataFrame)
        assert hydro_pct.is_empty()

    def test_cobre_hydro_meta_entries_carry_bus_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch,
            metadata=_hydro_metadata_fixture(),
            bus_labels=_hydro_bus_labels_fixture(),
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        cobre_hydro_meta = dataset.metadata["cobre_hydro_meta"]
        assert set(cobre_hydro_meta) == {0, 1}
        for entry in cobre_hydro_meta.values():
            assert "bus_ids" in entry
        assert cobre_hydro_meta[0]["bus_ids"] == frozenset({0})
        assert cobre_hydro_meta[1]["bus_ids"] == frozenset({0})
        # Plant physics from ``read_cobre_hydro_metadata`` survive the merge.
        assert cobre_hydro_meta[0]["name"] == "A"

    def test_cobre_hydro_meta_bus_ids_empty_when_plant_has_no_label(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch, metadata=_hydro_metadata_fixture(), bus_labels={}
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        cobre_hydro_meta = dataset.metadata["cobre_hydro_meta"]
        assert cobre_hydro_meta[0]["bus_ids"] == []
        assert cobre_hydro_meta[1]["bus_ids"] == []

    def test_hydro_per_stage_bounds_populate_metadata_verbatim(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch, per_stage_bounds=_hydro_per_stage_bounds_fixture()
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        bounds = dataset.metadata["cobre_hydro_per_stage_bounds"]
        assert bounds["max_storage_hm3"].to_list() == [1000.0, 600.0]

    def test_nw_hydro_slacks_is_empty_decomp_has_no_slack_table(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        nw_hydro_slacks = dataset.metadata["nw_hydro_slacks"]
        assert nw_hydro_slacks is None or nw_hydro_slacks.is_empty()

    def test_hydro_operation_and_detail_tabs_render_with_metadata_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(
            monkeypatch,
            percentiles=_hydro_percentiles_fixture(),
            metadata=_hydro_metadata_fixture(),
            bus_labels=_hydro_bus_labels_fixture(),
            per_stage_bounds=_hydro_per_stage_bounds_fixture(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_bus_metadata",
            lambda *_a, **_k: {0: {"name": "SUDESTE"}},
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Storage by Bus (hm³)" in html
        assert 'id="tab-hydro-detail"' in html
        hydro_tab = _extract_tab_content(html, "tab-hydro")
        assert "No hydro storage_final_hm3 data mapped to buses." not in hydro_tab
        assert "Plotly.newPlot" in html

    def test_both_hydro_tabs_render_without_exception_when_percentiles_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No cobre hydro percentiles/metadata/bus-labels at all -- both tabs
        must degrade gracefully (fallback panels), never raise."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_hydro_detail_readers(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)  # must not raise

        assert "Storage by Bus (hm³)" in html
        assert 'id="tab-hydro-detail"' in html

    def test_sin_slack_charts_render_cobre_only_no_newave_trace(self) -> None:
        """``nw_hydro_slacks`` empty (or ``None``) -> the SIN-total slack
        chart renders the Cobre Mean trace with no ``NEWAVE`` trace -- the
        documented ``has_newave=False`` degrade path -- exercised directly
        on the shared, unmodified chart function."""
        cobre_hydro = pl.DataFrame(
            {
                "entity_id": [0],
                "stage_id": [0],
                "water_withdrawal_violation_neg_m3s": [1.5],
            }
        )

        html = hydro_slack_aggregate_chart(
            cobre_hydro,
            None,
            "water_withdrawal_violation_neg_m3s",
            "Withdrawal Slack Pos (m³/s)",
        )

        assert '"name":"Cobre Mean"' in html
        assert '"name":"NEWAVE"' not in html


def _thermal_percentiles_fixture() -> pl.DataFrame:
    """Cobre p10/p50/p90 for the one ``_aligned_fixture`` thermal entity."""
    return pl.DataFrame(
        {
            "entity_id": [0],
            "stage_id": [0],
            "generation_mw_p10": [24.0],
            "generation_mw_p50": [28.0],
            "generation_mw_p90": [32.0],
        }
    )


def _patch_thermal_percentiles(
    monkeypatch: pytest.MonkeyPatch, percentiles: pl.DataFrame | None = None
) -> None:
    """Stub ticket-015's cobre thermal-percentile reader -- defaults to
    empty, matching how a Cobre run with no thermal percentile output
    (e.g. the deterministic 2-node tree) degrades in production."""
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers."
        "read_cobre_thermal_percentiles",
        lambda *_a, **_k: pl.DataFrame() if percentiles is None else percentiles,
    )


class TestBuildDecompDatasetThermalDetail:
    """ticket-015: fills ``PercentileData.thermal`` -- the disjoint thermal
    counterpart to ticket-014's hydro percentile band -- for the shared
    Thermal Operation and Thermal Plant Details tabs."""

    def test_thermal_percentiles_populate_metadata_when_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch, _thermal_percentiles_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        thermal_pct = dataset.metadata["thermal"]
        assert isinstance(thermal_pct, pl.DataFrame)
        assert not thermal_pct.is_empty()
        assert "generation_mw_p10" in thermal_pct.columns
        assert "generation_mw_p50" in thermal_pct.columns
        assert "generation_mw_p90" in thermal_pct.columns

    def test_thermal_percentiles_stay_empty_when_cobre_output_lacks_them(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No percentile mock (deterministic-tree low-N, master-plan caveat
        1): ``read_cobre_thermal_percentiles`` degrades to its own
        empty-frame default and the dataset must not fabricate a spread."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        thermal_pct = dataset.metadata["thermal"]
        assert isinstance(thermal_pct, pl.DataFrame)
        assert thermal_pct.is_empty()

    def test_thermal_tidy_rows_carry_generation_mw_and_known_sources(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """E1 already emits the thermal ``ResultComparison`` rows -- this
        ticket must not re-emit or re-map them (Pitfalls to Avoid)."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        thermal_rows = dataset.tidy.filter(pl.col("entity_type") == "thermal")
        assert not thermal_rows.is_empty()
        assert set(thermal_rows["variable"].to_list()) == {"generation_mw"}
        assert set(thermal_rows["source"].to_list()) == {"newave", "cobre"}

    def test_thermal_operation_and_detail_tabs_render_with_metadata_present(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch, _thermal_percentiles_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "Thermal Generation Comparison" in html
        assert 'id="tab-thermal-detail"' in html
        assert 'id="thermal-select"' in html
        assert "Plotly.newPlot" in html

    def test_both_thermal_tabs_render_without_exception_when_percentiles_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No cobre thermal percentiles at all -- both tabs must degrade
        gracefully (Cobre-only band suppressed), never raise."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_thermal_percentiles(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)  # must not raise

        assert "Thermal Generation Comparison" in html
        assert 'id="tab-thermal-detail"' in html


# ---------------------------------------------------------------------------
# ticket-016: Productivity tab realized per-stage metadata.
# ---------------------------------------------------------------------------


class TestHydroProductivityResults:
    """``_hydro_productivity_results`` derives per-(plant, stage) realized
    productivity = generation / turbined from the E1 hydro
    ``ResultComparison`` rows, mirroring
    ``cobre_bridge.comparators.results``'s own ``_compare_hydros``
    productivity derivation (same ratio, same zero-guard)."""

    @staticmethod
    def _hydro_pair(
        *,
        nw_gen: float,
        nw_turb: float,
        cb_gen: float,
        cb_turb: float,
        code: int = 10,
        cobre_id: int = 0,
        stage: int = 0,
        name: str = "ALPHA",
    ) -> list[ResultComparison]:
        """One plant/stage's generation_mw + turbined_m3s rows -- the shape
        ``_result_comparisons`` already emits for hydro entities."""
        return [
            ResultComparison(
                entity_type="hydro",
                entity_name=name,
                newave_code=code,
                cobre_id=cobre_id,
                stage=stage,
                variable="generation_mw",
                newave_value=nw_gen,
                cobre_value=cb_gen,
                abs_diff=abs(nw_gen - cb_gen),
                rel_diff=None,
            ),
            ResultComparison(
                entity_type="hydro",
                entity_name=name,
                newave_code=code,
                cobre_id=cobre_id,
                stage=stage,
                variable="turbined_m3s",
                newave_value=nw_turb,
                cobre_value=cb_turb,
                abs_diff=abs(nw_turb - cb_turb),
                rel_diff=None,
            ),
        ]

    def test_ratio_math(self) -> None:
        """AC1: geracao_MW=100, vazao_turbinada_m3s=50 -> newave_value 2.0."""
        rows = self._hydro_pair(nw_gen=100.0, nw_turb=50.0, cb_gen=90.0, cb_turb=45.0)

        productivity = _hydro_productivity_results(rows)

        assert len(productivity) == 1
        row = productivity[0]
        assert row.entity_type == "hydro"
        assert row.variable == "productivity_mw_per_m3s"
        assert row.newave_value == pytest.approx(2.0)
        assert row.cobre_value == pytest.approx(2.0)

    def test_zero_guard_drops_the_row_when_source_turbined_is_zero(self) -> None:
        """AC2: vazao_turbinada_m3s=0 -> no non-null newave_value -- the row
        is DROPPED entirely (never null-kept)."""
        rows = self._hydro_pair(nw_gen=100.0, nw_turb=0.0, cb_gen=90.0, cb_turb=45.0)

        assert _hydro_productivity_results(rows) == []

    def test_zero_guard_drops_the_row_when_cobre_turbined_is_zero(self) -> None:
        """Either side's turbined flow at zero drops the row -- not just the
        source model's."""
        rows = self._hydro_pair(nw_gen=100.0, nw_turb=50.0, cb_gen=90.0, cb_turb=0.0)

        assert _hydro_productivity_results(rows) == []

    def test_zero_guard_uses_the_eps_floor_not_only_exact_zero(self) -> None:
        """Below the eps floor (not just exactly zero) is also dropped -- an
        undefined 0/0, not a genuinely tiny but real ratio."""
        rows = self._hydro_pair(
            nw_gen=100.0,
            nw_turb=_PRODUCTIVITY_TURBINED_EPS / 2,
            cb_gen=90.0,
            cb_turb=45.0,
        )

        assert _hydro_productivity_results(rows) == []

    def test_row_dropped_when_the_matching_variable_is_absent(self) -> None:
        """Only a generation_mw row, no turbined_m3s row for that key --
        never an exception, just no productivity row (missing turbined flow
        per the ticket's Error Handling)."""
        rows = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ALPHA",
                newave_code=10,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=90.0,
                abs_diff=10.0,
                rel_diff=0.1,
            )
        ]

        assert _hydro_productivity_results(rows) == []

    def test_non_hydro_rows_are_ignored(self) -> None:
        rows = [
            ResultComparison(
                entity_type="thermal",
                entity_name="GAS_A",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=90.0,
                abs_diff=10.0,
                rel_diff=0.1,
            )
        ]

        assert _hydro_productivity_results(rows) == []

    def test_two_plants_sorted_by_code_then_stage(self) -> None:
        rows = self._hydro_pair(
            nw_gen=100.0,
            nw_turb=50.0,
            cb_gen=90.0,
            cb_turb=45.0,
            code=20,
            cobre_id=1,
            stage=1,
            name="BETA",
        ) + self._hydro_pair(
            nw_gen=60.0,
            nw_turb=30.0,
            cb_gen=55.0,
            cb_turb=25.0,
            code=10,
            cobre_id=0,
            stage=0,
            name="ALPHA",
        )

        productivity = _hydro_productivity_results(rows)

        assert [(r.newave_code, r.stage) for r in productivity] == [(10, 0), (20, 1)]


def _productivity_aligned_fixture() -> _AlignedDecompFrames:
    """Two hydro plants, one stage: plant A (code 10) exercises the ratio
    math (AC1: 100/50 -> 2.0 on the source side); plant B (code 20)'s source
    turbined is 0 (AC2: zero-guard drops it, never null-keeps it)."""
    source_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "newave_code": [10, 20],
            "stage_id": [0, 0],
            "geracao_MW": [100.0, 60.0],
            "vazao_turbinada_m3s": [50.0, 0.0],
            "vazao_vertida_m3s": [0.0, 0.0],
            "vazao_defluente_m3s": [50.0, 0.0],
            "volume_util_final_hm3": [500.0, 300.0],
        }
    )
    cobre_hydro = pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "generation_mw": [105.0, 60.0],
            "turbined_m3s": [50.0, 40.0],
            "spillage_m3s": [0.0, 0.0],
            "outflow_m3s": [50.0, 40.0],
            "useful_storage_hm3": [480.0, 300.0],
        }
    )
    return dataclasses.replace(
        _aligned_fixture(),
        source_hydro=source_hydro,
        cobre_hydro=cobre_hydro,
        hydro_names={0: "ALPHA", 1: "BETA"},
    )


class TestBuildDecompDatasetProductivity:
    """ticket-016: fills the Productivity tab's realized per-stage half
    (``dataset.metadata["productivity_per_stage"]``) and leaves the static
    pmo-derived half (``productivity_detail``) empty -- DECOMP ships no
    pmo.dat ([ASSUMPTION] option a, no fabricated static comparison)."""

    def test_productivity_per_stage_has_the_six_columns(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        per_stage = dataset.metadata["productivity_per_stage"]
        assert isinstance(per_stage, pl.DataFrame)
        assert set(per_stage.columns) == {
            "plant_name",
            "newave_code",
            "cobre_id",
            "stage",
            "newave_value",
            "cobre_value",
        }

    def test_ratio_math_matches_dec_oper_usih_generation_over_turbined(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC1: geracao_MW=100, vazao_turbinada_m3s=50 -> newave_value 2.0,
        end to end from ``build_decomp_dataset``."""
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        per_stage = dataset.metadata["productivity_per_stage"]
        alpha = per_stage.filter(pl.col("newave_code") == 10)
        assert alpha["newave_value"].to_list() == [pytest.approx(2.0)]
        assert alpha["cobre_value"].to_list() == [pytest.approx(2.1)]

    def test_zero_turbined_plant_emits_no_non_null_newave_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC2: plant B's source turbined is 0 -> the zero-guard holds (the
        row is DROPPED entirely, so no non-null -- or null -- value for it
        survives into the frame)."""
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        per_stage = dataset.metadata["productivity_per_stage"]
        assert per_stage.filter(pl.col("newave_code") == 20).is_empty()
        assert per_stage["newave_value"].null_count() == 0

    def test_productivity_detail_stays_empty_no_pmo_fabrication(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["productivity_detail"].is_empty()

    def test_report_renders_both_the_realized_title_and_the_static_no_data_note(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC4 + AC5 together, on the SAME DECOMP dataset: with
        ``productivity_detail`` empty AND ``productivity_per_stage``
        non-empty, the decoupled report_builder gate (ticket-016) renders
        BOTH the realized-productivity section and the static section's "No
        productivity data available" note."""
        _patch_aligned_frames(monkeypatch, _productivity_aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        productivity_tab = _extract_tab_content(html, "tab-productivity")
        assert "Realized productivity across stages" in productivity_tab
        assert "No productivity data available." in productivity_tab
        # The static-only Building Blocks table has no DECOMP data to render.
        assert "Productivity Building Blocks" not in productivity_tab


class TestReportBuilderProductivityGateDecoupling:
    """ticket-016 decoupled report_builder's single ``if prod_df.is_empty():
    ... else: ...`` Productivity-tab gate into two independent gates (one per
    frame) so DECOMP's realized-only shape can render. This guards the
    NEWAVE-shaped case -- both frames populated, which is what NEWAVE always
    has -- still renders every section in the exact original order/content,
    i.e. ``compare newave`` is unaffected by the decoupling."""

    @staticmethod
    def _both_frames_dataset() -> ComparisonDataset:
        from cobre_bridge.comparators.analyze import (
            _PRODUCTIVITY_DETAIL_SCHEMA,
            build_results_dataset,
        )
        from cobre_bridge.comparators.results import PercentileData

        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="ALPHA",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="productivity_mw_per_m3s",
                newave_value=0.78,
                cobre_value=0.80,
                abs_diff=0.02,
                rel_diff=0.026,
            )
        ]
        prod_detail = pl.DataFrame(
            {
                "plant_name": ["ALPHA"],
                "newave_code": [1],
                "cobre_id": [0],
                "nw_altura_min": [0.69],
                "nw_altura_65": [0.81],
                "nw_altura_max": [0.85],
                "nw_equivalent": [0.7865],
                "nw_accumulated_earm": [5.35],
                "nw_specific_productivity": [0.009],
                "nw_tailwater_m": [672.0],
                "nw_losses_m": [0.8],
                "nw_vmin_hm3": [100.0],
                "nw_vmax_hm3": [500.0],
                "cb_point": [0.811],
                "cb_equivalent": [0.7860],
                "cb_accumulated": [5.349],
                "cb_specific_productivity": [0.009],
                "cb_tailwater_m": [672.0],
                "cb_losses_m": [0.8],
                "cb_vmin_hm3": [100.0],
                "cb_vmax_hm3": [500.0],
            },
            schema=_PRODUCTIVITY_DETAIL_SCHEMA,
        )
        pct = PercentileData(productivity_detail=prod_detail)
        return build_results_dataset(results, pct, 0.05)

    def test_both_populated_renders_all_three_sections_in_the_original_order(
        self,
    ) -> None:
        dataset = self._both_frames_dataset()

        html = build_comparison_report(dataset)
        productivity_tab = _extract_tab_content(html, "tab-productivity")

        static_idx = productivity_tab.index(
            "Static productivity — pmo vs cobre-bridge conversion"
        )
        realized_idx = productivity_tab.index("Realized productivity across stages")
        blocks_idx = productivity_tab.index("Productivity Building Blocks")
        assert static_idx < realized_idx < blocks_idx
        assert "No productivity data available." not in productivity_tab

    def test_only_detail_populated_renders_static_only_no_realized_section(
        self,
    ) -> None:
        """The mirror case: ``prod_df`` non-empty, ``per_stage_df`` empty --
        the realized section must NOT render. Proves the two gates are
        independent, not still coupled to one another."""
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.results import PercentileData

        detail_only = self._both_frames_dataset()
        pct = PercentileData(
            productivity_detail=detail_only.metadata["productivity_detail"]
        )
        dataset = build_results_dataset([], pct, 0.05)

        html = build_comparison_report(dataset)
        productivity_tab = _extract_tab_content(html, "tab-productivity")

        assert "Productivity Building Blocks" in productivity_tab
        assert "Realized productivity across stages" not in productivity_tab


# ---------------------------------------------------------------------------
# ticket-017: Productivity tab's "Fitted production functions (FPHA)"
# section. See `_fpha_metrics`'s [ASSUMPTION] docstring in decomp_results.py:
# fallback (b) (fit-fidelity statistics only, no dense surface) is the
# documented degradation from the epic's chosen-target full-surface
# reconstruction (a), which this ticket's three declared readers cannot
# support (no fitted plane coefficients on the DECOMP side).
# ---------------------------------------------------------------------------


def _fpha_id_map() -> DecompIdMap:
    """One hydro plant (code 10 -> cobre id 0), matching `_aligned_fixture`'s
    own hydro code/id/name so the SAME `_patch_aligned_frames` fixture can
    back both the E1 result rows and the ticket-017 FPHA metrics in the same
    ``build_decomp_dataset`` test."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20))


def _fpha_cobre_planes_fixture() -> pl.DataFrame:
    """One fitted Cobre plane for (hydro_id=0, stage_id=0): ``GH = q``
    (``kappa=1``, every other coefficient zero except ``gamma_q``) --
    deliberately trivial so the envelope's value at any point is just its
    own ``q_m3s``, matching `cobre_readers.read_cobre_fpha_planes`'s own
    ``hydro_id``/``stage_id``/``gamma_0``/``gamma_v``/``gamma_q``/
    ``gamma_s``/``kappa`` schema."""
    return pl.DataFrame(
        {
            "hydro_id": [0],
            "stage_id": [0],
            "gamma_0": [0.0],
            "gamma_v": [0.0],
            "gamma_q": [1.0],
            "gamma_s": [0.0],
            "kappa": [1.0],
        }
    )


def _fpha_deviations_fixture() -> pl.DataFrame:
    """One realized source-model operating point for hydro code 10, stage 1
    (``estagio``, 1-based): ``vazao_turbinada_m3s=80`` (so Cobre's envelope
    evaluates to 80 MW under the trivial plane above) against the source
    model's own LP-consumed ``geracao_hidraulica_fpha=76`` -- a deliberate,
    exact 4 MW gap so nmae/bias/max_abs_dev/gh_max_ratio are hand-checkable.
    Matches `read_dec_desvfpha`'s real column set (a subset sufficient for
    `_fpha_metrics`)."""
    return pl.DataFrame(
        {
            "codigo_usina": [10],
            "estagio": [1],
            "volume_total_hm3": [500.0],
            "vazao_turbinada_m3s": [80.0],
            "vazao_vertida_m3s": [0.0],
            "geracao_hidraulica_fpha": [76.0],
        }
    )


def _patch_fpha_planes_and_deviations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire `build_decomp_dataset`'s three ticket-017 sources -- outside
    `_read_aligned_frames` -- to the fixtures above: Cobre's planes reader,
    the deck id map (`_build_line_id_map`, reused verbatim by
    `_fpha_metrics` rather than rebuilt), and the source model's own
    deviation table. ``read_eco_fpha``/``read_dec_estatfpha`` are left
    unmocked -- they raise `FileNotFoundError` against a bare `tmp_path`,
    exercising `_fpha_metrics`'s own graceful degrade for those two optional
    sources (``n_v`` stays null; the deck-wide summary is only logged).
    """
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.cobre_readers.read_cobre_fpha_planes",
        lambda *_a, **_k: _fpha_cobre_planes_fixture(),
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._build_line_id_map",
        lambda *_a, **_k: _fpha_id_map(),
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_dec_desvfpha",
        lambda *_a, **_k: _fpha_deviations_fixture(),
    )


class TestFphaMetrics:
    """`_fpha_metrics`: fallback (b)'s per-(hydro, stage) fit-fidelity table
    -- Cobre's own fitted envelope evaluated at the source model's realized
    operating points, compared to the source model's own realized
    ``geracao_hidraulica_fpha``."""

    def test_both_sides_present_computes_real_cross_solver_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_fpha_planes",
            lambda *_a, **_k: _fpha_cobre_planes_fixture(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_desvfpha",
            lambda *_a, **_k: _fpha_deviations_fixture(),
        )

        metrics = _fpha_metrics(tmp_path, tmp_path, _fpha_id_map(), {0: "A", 1: "B"})

        assert metrics is not None
        assert metrics.height == 1
        row = metrics.row(0, named=True)
        assert row["cobre_id"] == 0
        assert row["plant_name"] == "A"
        assert row["stage"] == 0
        # No declared FPHA reader counts the source model's own planes.
        assert row["n_planes_newave"] is None
        assert row["n_planes_cobre"] == 1
        # `read_eco_fpha` is unmocked -> raises against a bare tmp_path ->
        # degrades to null, never fabricated.
        assert row["n_v"] is None
        assert row["nmae"] == pytest.approx(4.0 / 76.0)
        assert row["bias"] == pytest.approx(4.0 / 76.0)
        assert row["max_abs_dev"] == pytest.approx(4.0)
        assert row["gh_max_ratio"] == pytest.approx(80.0 / 76.0)

    def test_cobre_has_no_planes_returns_none(self, tmp_path: Path) -> None:
        """`read_cobre_fpha_planes` naturally returns `None` against a bare
        Cobre output dir (no ``hydro_models/fpha_hyperplanes.parquet``)."""
        assert _fpha_metrics(tmp_path, tmp_path, _fpha_id_map(), {}) is None

    def test_no_id_map_returns_none_even_with_cobre_planes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_fpha_planes",
            lambda *_a, **_k: _fpha_cobre_planes_fixture(),
        )
        assert _fpha_metrics(tmp_path, tmp_path, None, {}) is None

    def test_missing_source_deviation_table_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_fpha_planes",
            lambda *_a, **_k: _fpha_cobre_planes_fixture(),
        )
        # `read_dec_desvfpha` left unmocked -> raises FileNotFoundError.
        assert _fpha_metrics(tmp_path, tmp_path, _fpha_id_map(), {}) is None

    def test_unresolvable_hydro_code_returns_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A deviation table whose plant code the id map cannot resolve at
        all (no hydro codes declared) never reaches the Cobre planes it
        would otherwise match."""
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_fpha_planes",
            lambda *_a, **_k: _fpha_cobre_planes_fixture(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_desvfpha",
            lambda *_a, **_k: _fpha_deviations_fixture(),
        )
        no_hydros = DecompIdMap(bus_codes=(1,), bus_names=("SE",))
        assert _fpha_metrics(tmp_path, tmp_path, no_hydros, {}) is None


class TestBuildDecompDatasetFpha:
    """ticket-017: fills ``dataset.metadata["fpha_metrics"]``;
    ``fpha_surface``/``fpha_spill`` always stay `None` (fallback (b))."""

    def test_cobre_has_no_planes_fpha_metrics_absent_no_section(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: ``read_cobre_fpha_planes`` returns ``None`` (the default on a
        bare Cobre output dir) -> no FPHA metadata, the report omits the
        section entirely, no exception."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata.get("fpha_metrics") is None
        html = build_comparison_report(dataset)  # must not raise
        assert "Fitted production functions (FPHA)" not in html

    def test_both_sides_fitted_planes_fpha_metrics_populated_section_renders(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: mocked frames on both sides -> a non-empty ``fpha_metrics``
        with the required columns, and the report renders the FPHA section
        title."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_fpha_planes_and_deviations(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        fpha_metrics = dataset.metadata["fpha_metrics"]
        assert isinstance(fpha_metrics, pl.DataFrame)
        assert not fpha_metrics.is_empty()
        for column in ("cobre_id", "plant_name", "nmae", "bias"):
            assert column in fpha_metrics.columns

        html = build_comparison_report(dataset)  # must not raise
        assert "Fitted production functions (FPHA)" in html

    def test_fpha_surface_and_spill_always_stay_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Fallback (b): never a dense surface, even once ``fpha_metrics``
        is populated -- the section renders the metrics table but the
        surface/spill widget falls back to its own "No production-function
        (FPHA) data available" placeholder."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        _patch_fpha_planes_and_deviations(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata.get("fpha_surface") is None
        assert dataset.metadata.get("fpha_spill") is None
        html = build_comparison_report(dataset)  # must not raise
        assert "No production-function (FPHA) data available." in html

    def test_report_renders_without_exception_present_and_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: ``build_comparison_report`` raises no exception whether the
        FPHA section is present or omitted."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        dataset_absent = build_decomp_dataset(tmp_path, tmp_path)
        build_comparison_report(dataset_absent)  # must not raise

        _patch_fpha_planes_and_deviations(monkeypatch)
        dataset_present = build_decomp_dataset(tmp_path, tmp_path)
        build_comparison_report(dataset_present)  # must not raise


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetFphaE2E:
    """Tier 3 (dev-only smoke, ticket-017): the reduced deck's real FPHA
    section renders end to end, and the full report renders without
    exception across every E1-E6 tab. Both directories are gitignored, so
    this never runs in CI."""

    def test_report_renders_every_tab_without_exception(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        html = build_comparison_report(dataset)  # must not raise

        for tab_id in (
            "tab-overview",
            "tab-system",
            "tab-balance",
            "tab-network",
            "tab-constraints",
            "tab-hydro",
            "tab-hydro-detail",
            "tab-thermal",
            "tab-thermal-detail",
            "tab-productivity",
            "tab-performance",
        ):
            assert f'id="{tab_id}"' in html

    def test_fpha_section_renders_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        fpha_metrics = dataset.metadata.get("fpha_metrics")
        assert fpha_metrics is not None
        assert not fpha_metrics.is_empty()
        for column in ("cobre_id", "plant_name", "nmae", "bias"):
            assert column in fpha_metrics.columns

        html = build_comparison_report(dataset)
        assert "Fitted production functions (FPHA)" in html


# ---------------------------------------------------------------------------
# ticket-018: REE energy rollup via membership + additive REE section.
# ---------------------------------------------------------------------------


def _ree_id_map() -> DecompIdMap:
    """Two hydro plants (codes 10, 20 -> cobre ids 0, 1) -- matches
    ``_aligned_fixture``'s own hydro codes/ids so the same
    ``_patch_aligned_frames`` fixture can back both the E1 result rows and
    the REE rollup in the same ``build_decomp_dataset`` test."""
    return DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20))


def _ree_membership_fixture() -> pl.DataFrame:
    """Both fixture plants (codes 10, 20) belong to REE 100 ('SUDESTE')."""
    return pl.DataFrame(
        {
            "codigo_usina": [10, 20],
            "nome_usina": ["A", "B"],
            "codigo_ree": [100, 100],
            "nome_ree": ["SUDESTE", "SUDESTE"],
            "codigo_submercado": [1, 1],
            "nome_submercado": ["SE", "SE"],
            "nome_submercado_newave": ["SUDESTE", "SUDESTE"],
        }
    )


def _ree_dec_oper_ree_fixture() -> pl.DataFrame:
    """One REE (100), stage 1 (1-based), two nodes -- scenario-mean
    ``ena_MWmes=145.0``, ``earm_final_MWmes=1010.0``, deliberately offset from
    the Cobre-side fixture's ``150.0`` / ``1000.0`` (see
    :func:`_ree_cobre_hydro_fixture`) so the per-variable diff is
    hand-checkable rather than trivially zero."""
    return pl.DataFrame(
        {
            "estagio": [1, 1],
            "no": [1, 2],
            "cenario": [1, 1],
            "codigo_ree": [100, 100],
            "nome_ree": ["SUDESTE", "SUDESTE"],
            "codigo_submercado": [1, 1],
            "nome_submercado": ["SE", "SE"],
            "ena_MWmes": [140.0, 150.0],
            "earm_inicial_MWmes": [900.0, 900.0],
            "earm_inicial_percentual": [70.0, 70.0],
            "earm_final_MWmes": [1000.0, 1020.0],
            "earm_final_percentual": [72.0, 74.0],
            "earm_maximo_MWmes": [2000.0, 2000.0],
        }
    )


def _ree_cobre_hydro_fixture() -> pl.DataFrame:
    """Two Cobre hydro plants (ids 0, 1), one stage: ENA sums to 150.0 MW,
    EARM sums to 730000.0 MWh -- exactly ``1000.0 * _EARM_MWH_TO_MWMES``, so
    the MWh -> MWmes reconciliation lands on a round number."""
    return pl.DataFrame(
        {
            "entity_id": [0, 1],
            "stage_id": [0, 0],
            "incremental_inflow_energy_mw": [90.0, 60.0],
            "stored_energy_final_mwh": [400000.0, 330000.0],
        }
    )


def _ree_aligned_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture()`` with its ``cobre_hydro`` extended to carry the
    ENA/EARM columns :func:`_cobre_ree_sums` reads -- the base fixture is
    trimmed to only the columns E1's ``_HYDRO_VARIABLES`` needs."""
    base = _aligned_fixture()
    return dataclasses.replace(
        base,
        cobre_hydro=base.cobre_hydro.with_columns(
            pl.Series("incremental_inflow_energy_mw", [90.0, 60.0]),
            pl.Series("stored_energy_final_mwh", [400000.0, 330000.0]),
        ),
    )


def _patch_ree_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wire ``read_relato_membership``/``read_dec_oper_ree`` -- outside
    ``_read_aligned_frames`` -- to the fixtures above."""
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_relato_membership",
        lambda *_a, **_k: _ree_membership_fixture(),
    )
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_dec_oper_ree",
        lambda *_a, **_k: _ree_dec_oper_ree_fixture(),
    )


class TestReeMembershipMap:
    """``_ree_membership_map``: ``{cobre_hydro_id: codigo_ree}`` via
    membership, restricted to the operated hydro codes."""

    def test_maps_cobre_ids_to_codigo_ree(self) -> None:
        ree_by_cobre_id, unmapped = _ree_membership_map(
            _ree_membership_fixture(), {10: 0, 20: 1}
        )
        assert ree_by_cobre_id == {0: 100, 1: 100}
        assert unmapped == []

    def test_reports_unmapped_hydro_codes_instead_of_dropping_silently(self) -> None:
        membership = pl.DataFrame({"codigo_usina": [10], "codigo_ree": [100]})

        ree_by_cobre_id, unmapped = _ree_membership_map(membership, {10: 0, 99: 5})

        assert ree_by_cobre_id == {0: 100}
        assert unmapped == [99]

    def test_empty_membership_excludes_every_hydro_code(self) -> None:
        ree_by_cobre_id, unmapped = _ree_membership_map(pl.DataFrame(), {10: 0, 20: 1})

        assert ree_by_cobre_id == {}
        assert unmapped == [10, 20]


class TestCobreReeSums:
    """``_cobre_ree_sums``: membership-weighted per-(codigo_ree, stage) sum
    of Cobre hydro ENA/EARM."""

    def test_sums_ena_and_earm_across_member_plants(self) -> None:
        out = _cobre_ree_sums(_ree_cobre_hydro_fixture(), {0: 100, 1: 100})

        assert out.height == 1
        row = out.row(0, named=True)
        assert row["entity_id"] == 100
        assert row["stage_id"] == 0
        assert row["ena_mw"] == pytest.approx(150.0)
        assert row["earm_mwh"] == pytest.approx(730000.0)

    def test_cobre_id_absent_from_membership_excluded_from_sum(self) -> None:
        cobre_hydro = pl.DataFrame(
            {
                "entity_id": [0, 9],
                "stage_id": [0, 0],
                "incremental_inflow_energy_mw": [90.0, 999.0],
                "stored_energy_final_mwh": [400000.0, 999.0],
            }
        )

        out = _cobre_ree_sums(cobre_hydro, {0: 100})

        assert out.height == 1
        row = out.row(0, named=True)
        assert row["ena_mw"] == pytest.approx(90.0)
        assert row["earm_mwh"] == pytest.approx(400000.0)

    def test_empty_cobre_hydro_yields_empty_frame(self) -> None:
        assert _cobre_ree_sums(pl.DataFrame(), {0: 100}).is_empty()

    def test_empty_membership_map_yields_empty_frame(self) -> None:
        assert _cobre_ree_sums(_ree_cobre_hydro_fixture(), {}).is_empty()

    def test_missing_energy_columns_degrades_to_empty_instead_of_raising(self) -> None:
        """A ``cobre_hydro`` frame that carries no ENA/EARM columns at all --
        e.g. the trimmed ``_aligned_fixture()`` shape other tickets' fixtures
        use -- must degrade gracefully rather than raising a Polars
        ``ColumnNotFoundError``."""
        cobre_hydro = pl.DataFrame({"entity_id": [0], "stage_id": [0]})

        assert _cobre_ree_sums(cobre_hydro, {0: 100}).is_empty()


class TestDecompReeFrame:
    """``_decomp_ree_frame``: DECOMP-side per-(codigo_ree, stage) ENA/EARM,
    scenario-averaged, plus the REE display-name lookup."""

    def test_scenario_means_and_rebases_stage(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_ree",
            lambda *_a, **_k: _ree_dec_oper_ree_fixture(),
        )

        frame, names = _decomp_ree_frame(tmp_path)

        assert frame.height == 1
        row = frame.row(0, named=True)
        assert row["entity_id"] == 100
        assert row["stage_id"] == 0
        assert row["ena_MWmes"] == pytest.approx(145.0)
        assert row["earm_final_MWmes"] == pytest.approx(1010.0)
        assert names == {100: "SUDESTE"}

    def test_missing_table_raises(self, tmp_path: Path) -> None:
        """``read_dec_oper_ree`` unmocked against a bare ``tmp_path`` raises
        -- the caller (``_ree_result_comparisons``) is what degrades this to
        an absent REE section, not this helper."""
        with pytest.raises(FileNotFoundError):
            _decomp_ree_frame(tmp_path)


class TestReeResultComparisons:
    """``_ree_result_comparisons``: the full REE rollup -- membership map,
    scenario-averaged DECOMP side, membership-weighted Cobre side, the EARM
    MWh -> MWmes reconciliation, and the never-silently-dropped
    unmapped-plant diagnostic (ticket-018 requirement 4)."""

    def test_ena_is_a_plain_sum_earm_is_divided_by_730(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_ree_sources(monkeypatch)

        results, unmapped = _ree_result_comparisons(
            tmp_path, _ree_cobre_hydro_fixture(), _ree_id_map()
        )

        assert unmapped == []
        by_variable = {r.variable: r for r in results}
        assert set(by_variable) == {"ena_mwmes", "earm_final_mwmes"}

        ena = by_variable["ena_mwmes"]
        assert ena.entity_type == "ree"
        assert ena.entity_name == "SUDESTE"
        assert ena.newave_code == 100
        assert ena.cobre_id == 100
        assert ena.stage == 0
        assert ena.newave_value == pytest.approx(145.0)
        assert ena.cobre_value == pytest.approx(150.0)  # plain sum, no scaling
        assert ena.abs_diff == pytest.approx(5.0)

        earm = by_variable["earm_final_mwmes"]
        assert earm.newave_value == pytest.approx(1010.0)
        # 730000.0 MWh / _EARM_MWH_TO_MWMES(730) == 1000.0 MWmes.
        assert earm.cobre_value == pytest.approx(730000.0 / _EARM_MWH_TO_MWMES)
        assert earm.cobre_value == pytest.approx(1000.0)
        assert earm.abs_diff == pytest.approx(10.0)

    def test_none_id_map_returns_no_rows_and_no_unmapped(self, tmp_path: Path) -> None:
        results, unmapped = _ree_result_comparisons(
            tmp_path, _ree_cobre_hydro_fixture(), None
        )

        assert results == []
        assert unmapped == []

    def test_plant_absent_from_membership_excluded_and_recorded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: a hydro code absent from the membership table is excluded
        from every REE sum and its code is recorded via the diagnostics
        path, not silently dropped."""
        # The id map declares a THIRD hydro code (30) the membership fixture
        # never lists.
        id_map = DecompIdMap(
            bus_codes=(1,), bus_names=("SE",), hydro_codes=(10, 20, 30)
        )
        _patch_ree_sources(monkeypatch)

        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                tmp_path, _ree_cobre_hydro_fixture(), id_map
            )

        assert unmapped == [30]
        assert len(collected) == 1
        assert collected[0].code == "ree-membership-plant-unmapped"
        assert "30" in " ".join(str(n) for n in collected[0].notes)
        # The unmapped plant's cobre id (2) was never a REE member -- the
        # sums are unaffected (still exactly the two-plant fixture's totals).
        by_variable = {r.variable: r for r in results}
        assert by_variable["ena_mwmes"].cobre_value == pytest.approx(150.0)

    def test_no_membership_table_degrades_to_no_rows_no_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """A missing relato (``read_relato_membership`` raising against a
        bare ``tmp_path``) is a genuinely unavailable REE section, not a
        per-plant gap -- no diagnostic, just an empty result."""
        with dx.collect() as collected:
            results, unmapped = _ree_result_comparisons(
                tmp_path, _ree_cobre_hydro_fixture(), _ree_id_map()
            )

        assert results == []
        assert unmapped == []
        assert collected == []

    def test_no_dec_oper_ree_table_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_relato_membership",
            lambda *_a, **_k: _ree_membership_fixture(),
        )
        # ``read_dec_oper_ree`` left unmocked -> raises FileNotFoundError.

        results, unmapped = _ree_result_comparisons(
            tmp_path, _ree_cobre_hydro_fixture(), _ree_id_map()
        )

        assert results == []
        assert unmapped == []


class TestBuildDecompDatasetRee:
    """ticket-018: fills ``results`` with ``entity_type="ree"`` rows and
    ``dataset.metadata["unmapped"]["ree"]``."""

    def test_no_deck_no_ree_rows_and_empty_unmapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_build_line_id_map`` returns ``None`` against a bare
        ``tmp_path`` (no deck to read) -> no REE rollup, empty
        ``unmapped["ree"]``, no exception, and no REE section in the report."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert ree_rows.is_empty()
        assert dataset.metadata["unmapped"]["ree"] == []
        html = build_comparison_report(dataset)  # must not raise
        assert "REE Energy" not in html

    def test_both_sides_present_tidy_carries_ree_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: ``tidy`` has ``entity_type=="ree"`` rows for ``ena_mwmes`` and
        ``earm_final_mwmes``, with ``source`` in {"newave", "cobre"}."""
        _patch_aligned_frames(monkeypatch, _ree_aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._build_line_id_map",
            lambda *_a, **_k: _ree_id_map(),
        )
        _patch_ree_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert set(ree_rows["variable"].unique().to_list()) == {
            "ena_mwmes",
            "earm_final_mwmes",
        }
        assert set(ree_rows["source"].unique().to_list()) == {"newave", "cobre"}
        assert dataset.metadata["unmapped"]["ree"] == []

    def test_report_ree_section_present_for_decomp_dataset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: ``build_comparison_report(dataset)`` renders a non-empty REE
        energy section for a DECOMP dataset."""
        _patch_aligned_frames(monkeypatch, _ree_aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._build_line_id_map",
            lambda *_a, **_k: _ree_id_map(),
        )
        _patch_ree_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert "REE Energy" in html
        balance_tab = _extract_tab_content(html, "tab-balance")
        assert "No ena_mwmes data available." not in balance_tab
        assert "No earm_final_mwmes data available." not in balance_tab
        assert "Plotly.newPlot" in balance_tab


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetReeE2E:
    """Tier 3 (dev-only smoke, ticket-018): the reduced deck's real REE
    section renders end to end. Both directories are gitignored, so this
    never runs in CI."""

    def test_ree_section_renders_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        ree_rows = dataset.tidy.filter(pl.col("entity_type") == "ree")
        assert not ree_rows.is_empty()
        assert set(ree_rows["variable"].unique().to_list()) == {
            "ena_mwmes",
            "earm_final_mwmes",
        }

        html = build_comparison_report(dataset)
        assert "REE Energy" in html


class TestReeEnergyChart:
    """``charts.ree_energy_chart``: mirrors ``system_comparison_chart``'s
    aggregate-line shape, keyed on ``entity_type == "ree"``."""

    def _results(self) -> list[ResultComparison]:
        return [
            ResultComparison(
                entity_type="ree",
                entity_name="SUDESTE",
                newave_code=100,
                cobre_id=100,
                stage=0,
                variable="ena_mwmes",
                newave_value=145.0,
                cobre_value=150.0,
                abs_diff=5.0,
                rel_diff=5.0 / 145.0,
            ),
            ResultComparison(
                entity_type="ree",
                entity_name="SUL",
                newave_code=200,
                cobre_id=200,
                stage=0,
                variable="ena_mwmes",
                newave_value=50.0,
                cobre_value=48.0,
                abs_diff=2.0,
                rel_diff=2.0 / 50.0,
            ),
        ]

    def test_no_matching_rows_renders_placeholder(self) -> None:
        html = ree_energy_chart([], "ena_mwmes", "REE ENA")
        assert "No ena_mwmes data available." in html

    def test_sums_across_matched_rees_per_stage(self) -> None:
        html = ree_energy_chart(self._results(), "ena_mwmes", "REE ENA")

        assert "Plotly.newPlot" in html
        assert "195" in html  # 145 + 50 == 195 (newave aggregate)
        assert "198" in html  # 150 + 48 == 198 (cobre aggregate)

    def test_ignores_rows_of_a_different_variable(self) -> None:
        html = ree_energy_chart(self._results(), "earm_final_mwmes", "REE EARM")
        assert "No earm_final_mwmes data available." in html


class TestReportBuilderReeSectionByteIdentityGuard:
    """ticket-018 requirement 5: the REE section is additive and must leave
    ``compare newave`` untouched -- it renders only when ``entity_type ==
    "ree"`` rows exist, which a NEWAVE-shaped dataset never carries."""

    def test_newave_shaped_dataset_has_no_ree_section(self) -> None:
        from cobre_bridge.comparators.analyze import build_results_dataset
        from cobre_bridge.comparators.results import PercentileData

        results = [
            ResultComparison(
                entity_type="hydro",
                entity_name="CAMARGOS",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="generation_mw",
                newave_value=100.0,
                cobre_value=98.0,
                abs_diff=2.0,
                rel_diff=0.02,
            ),
            ResultComparison(
                entity_type="bus",
                entity_name="SE",
                newave_code=1,
                cobre_id=0,
                stage=0,
                variable="deficit_mw",
                newave_value=0.0,
                cobre_value=0.0,
                abs_diff=0.0,
                rel_diff=None,
            ),
        ]
        dataset = build_results_dataset(results, PercentileData(), 0.05)
        assert dataset.tidy.filter(pl.col("entity_type") == "ree").is_empty()

        html = build_comparison_report(dataset)

        assert "REE Energy" not in html


# ---------------------------------------------------------------------------
# ticket-020: evaporation comparison (hydro, "evaporation_m3s") + C11 surface.
# ---------------------------------------------------------------------------


def _write_stages_json(case_dir: Path, stage_hours: dict[int, list[float]]) -> Path:
    """Write a minimal ``stages.json`` -- *stage_hours* maps
    ``stage_id -> [block_hours, ...]``. Mirrors ``test_cobre_readers.py``'s
    own ``_write_stages_json`` helper shape, duplicated locally rather than
    imported so this file keeps no cross-test-module dependency."""
    data = {
        "stages": [
            {
                "id": stage_id,
                "blocks": [
                    {"id": bid, "hours": hours} for bid, hours in enumerate(block_hours)
                ],
            }
            for stage_id, block_hours in stage_hours.items()
        ]
    }
    path = case_dir / "stages.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestHm3ToM3s:
    """`_hm3_to_m3s`: hm³ (stage volume) -> m³/s (mean flow), driven by the
    stage's own hours -- ticket-020 requirement 2."""

    def test_matches_the_fixed_monthly_factor_at_730_hours(self) -> None:
        """1 m³/s sustained over a 730h month deposits exactly
        ``converters.network.C_M3S2HM3`` (2.628 hm³); converting that volume
        back at 730h must recover 1.0 m³/s exactly."""
        assert _hm3_to_m3s(2.628, 730.0) == pytest.approx(1.0)

    def test_sub_monthly_stage_uses_its_own_hours_not_730(self) -> None:
        """A 168h (weekly) DECOMP stage: 1 m³/s deposits
        168 * 3600 / 1e6 = 0.6048 hm³ over that stage -- converting back at
        the stage's own 168h must recover 1.0 m³/s, not the ~0.23 m³/s a
        (wrong) fixed 730h monthly divisor would yield."""
        assert _hm3_to_m3s(0.6048, 168.0) == pytest.approx(1.0)
        assert _hm3_to_m3s(0.6048, 730.0) == pytest.approx(168.0 / 730.0)

    def test_hour_factor_matches_the_converter_side_monthly_constant(self) -> None:
        """`_HM3_TO_M3S_HOUR_FACTOR * 730` reproduces
        ``converters.network.C_M3S2HM3`` (2.628) -- the same physical
        relationship, generalized from the fixed monthly constant to any
        stage's own hours."""
        assert _HM3_TO_M3S_HOUR_FACTOR * 730.0 == pytest.approx(2.628)


class TestCobreStageHours:
    """`_cobre_stage_hours`: per-stage total hours from the Cobre case's own
    ``stages.json``, via `cobre_readers._load_block_hours` -- ticket-020."""

    def test_sums_block_hours_per_stage(self, tmp_path: Path) -> None:
        _write_stages_json(tmp_path, {0: [24.0, 144.0], 1: [168.0]})

        hours = _cobre_stage_hours(tmp_path)

        assert hours == {0: pytest.approx(168.0), 1: pytest.approx(168.0)}

    def test_no_stages_json_returns_empty_dict(self, tmp_path: Path) -> None:
        assert _cobre_stage_hours(tmp_path) == {}


def _evap_dec_oper_evap_fixture() -> pl.DataFrame:
    """One stage (``estagio=1``), two nodes, two plants (codes 10, 20 --
    matching ``_ree_id_map()``'s hydro codes): the scenario mean is
    ``1.2 hm³`` for plant 10 and ``2.2 hm³`` for plant 20, deliberately not
    round numbers so the hm³ -> m³/s conversion is hand-checkable rather than
    trivially exact."""
    return pl.DataFrame(
        {
            "estagio": [1, 1, 1, 1],
            "no": [1, 2, 1, 2],
            "cenario": [1, 1, 1, 1],
            "codigo_usina": [10, 10, 20, 20],
            "nome_usina": ["A", "A", "B", "B"],
            "evaporacao_calculada_hm3": [1.0, 1.4, 2.0, 2.4],
        }
    )


def _evap_aligned_fixture() -> _AlignedDecompFrames:
    """``_aligned_fixture()`` with its ``cobre_hydro`` extended to carry
    ``evaporation_m3s`` -- the base fixture only carries E1's
    ``_HYDRO_VARIABLES`` columns. Plant 0 (code 10) = 2.5 m³/s, plant 1
    (code 20) = 3.0 m³/s."""
    base = _aligned_fixture()
    return dataclasses.replace(
        base,
        cobre_hydro=base.cobre_hydro.with_columns(
            pl.Series("evaporation_m3s", [2.5, 3.0])
        ),
    )


def _patch_evap_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stage_hours: dict[int, list[float]] | None = None,
) -> None:
    """Wire ``read_dec_oper_evap`` (outside ``_read_aligned_frames``) to
    :func:`_evap_dec_oper_evap_fixture`, and ``_cobre_stage_hours`` to a
    fixed one-stage 168h lookup unless *stage_hours* overrides it."""
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results.read_dec_oper_evap",
        lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
    )
    hours = {0: 168.0} if stage_hours is None else stage_hours
    monkeypatch.setattr(
        "cobre_bridge.comparators.decomp_results._cobre_stage_hours",
        lambda *_a, **_k: hours,
    )


class TestEvapSide:
    """`_evap_side`: the source model's per-(hydro, stage) evaporated volume,
    scenario-averaged and mapped onto Cobre ids -- mirrors `_hydro_side`'s
    own fold exactly."""

    def test_scenario_means_and_maps_to_cobre_ids(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
        )

        frame, unmapped = _evap_side(tmp_path, {10: 0, 20: 1})

        assert unmapped == []
        by_id = {
            int(row["entity_id"]): row["evaporacao_calculada_hm3"]
            for row in frame.iter_rows(named=True)
        }
        assert by_id == {0: pytest.approx(1.2), 1: pytest.approx(2.2)}
        assert frame["stage_id"].unique().to_list() == [0]

    def test_unmapped_code_is_reported_not_dropped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
        )

        frame, unmapped = _evap_side(tmp_path, {10: 0})  # code 20 unmapped

        assert unmapped == [20]
        assert frame["entity_id"].to_list() == [0]

    def test_missing_table_raises(self, tmp_path: Path) -> None:
        """``read_dec_oper_evap`` unmocked against a bare ``tmp_path`` raises
        -- the caller (``_evaporation_result_comparisons``) is what degrades
        this to an absent evaporation section, not this helper."""
        with pytest.raises(FileNotFoundError):
            _evap_side(tmp_path, {10: 0})


class TestEvaporationResultComparisons:
    """`_evaporation_result_comparisons`: the full evaporation reconciliation
    -- scenario-averaged source-model volume converted to m³/s via the
    stage's own hours, joined against Cobre's ``evaporation_m3s``, with a
    one-sided plant excluded from the pairing and counted rather than
    silently dropped (ticket-020 requirement 5)."""

    def test_paired_rows_use_the_stage_hours_conversion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_evap_sources(monkeypatch)

        results, one_sided = _evaporation_result_comparisons(
            tmp_path,
            tmp_path,
            _evap_aligned_fixture().cobre_hydro,
            _ree_id_map(),
            {0: "A", 1: "B"},
        )

        assert one_sided == []
        by_id = {r.cobre_id: r for r in results}
        assert set(by_id) == {0, 1}

        for r in results:
            assert r.entity_type == "hydro"
            assert r.variable == "evaporation_m3s"

        # Plant 0 (code 10): scenario-mean 1.2 hm³ over a 168h stage.
        assert by_id[0].newave_code == 10
        assert by_id[0].newave_value == pytest.approx(_hm3_to_m3s(1.2, 168.0))
        assert by_id[0].cobre_value == pytest.approx(2.5)
        # Plant 1 (code 20): scenario-mean 2.2 hm³ over the same stage.
        assert by_id[1].newave_code == 20
        assert by_id[1].newave_value == pytest.approx(_hm3_to_m3s(2.2, 168.0))
        assert by_id[1].cobre_value == pytest.approx(3.0)

    def test_plant_present_only_on_cobre_side_excluded_and_counted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: a hydro present in Cobre's ``evaporation_m3s`` but absent from
        the source model's own evaporation table is excluded from the paired
        comparison and counted."""
        # Source model reports evaporation for plant 10 (cobre id 0) only.
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture().filter(
                pl.col("codigo_usina") == 10
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._cobre_stage_hours",
            lambda *_a, **_k: {0: 168.0},
        )

        with dx.collect() as collected:
            results, one_sided = _evaporation_result_comparisons(
                tmp_path,
                tmp_path,
                _evap_aligned_fixture().cobre_hydro,  # carries ids 0 AND 1
                _ree_id_map(),
                {0: "A", 1: "B"},
            )

        assert [r.cobre_id for r in results] == [0]
        assert one_sided == [1]
        assert len(collected) == 1
        assert collected[0].code == "evaporation-plant-one-sided"
        assert "1" in " ".join(str(n) for n in collected[0].notes)

    def test_plant_present_only_on_source_side_excluded_and_counted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC (reverse direction): a hydro present in the source model's own
        evaporation table but absent from Cobre's ``evaporation_m3s`` is
        excluded from the paired comparison and counted."""
        _patch_evap_sources(monkeypatch)
        # Cobre reports evaporation_m3s for plant 0 (code 10) only.
        cobre_hydro = _evap_aligned_fixture().cobre_hydro.filter(
            pl.col("entity_id") == 0
        )

        with dx.collect() as collected:
            results, one_sided = _evaporation_result_comparisons(
                tmp_path, tmp_path, cobre_hydro, _ree_id_map(), {0: "A", 1: "B"}
            )

        assert [r.cobre_id for r in results] == [0]
        assert one_sided == [1]
        assert len(collected) == 1
        assert collected[0].code == "evaporation-plant-one-sided"

    def test_none_id_map_returns_no_rows_and_no_unmapped(self, tmp_path: Path) -> None:
        results, one_sided = _evaporation_result_comparisons(
            tmp_path, tmp_path, _evap_aligned_fixture().cobre_hydro, None, {}
        )

        assert results == []
        assert one_sided == []

    def test_no_source_evaporation_table_degrades_to_no_rows(
        self, tmp_path: Path
    ) -> None:
        """``read_dec_oper_evap`` left unmocked -> raises ``FileNotFoundError``
        -- a genuinely unavailable section, not a per-plant gap."""
        with dx.collect() as collected:
            results, one_sided = _evaporation_result_comparisons(
                tmp_path,
                tmp_path,
                _evap_aligned_fixture().cobre_hydro,
                _ree_id_map(),
                {0: "A", 1: "B"},
            )

        assert results == []
        assert one_sided == []
        assert collected == []

    def test_no_evaporation_m3s_column_on_cobre_side_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_evap_sources(monkeypatch)

        results, one_sided = _evaporation_result_comparisons(
            tmp_path,
            tmp_path,
            _aligned_fixture().cobre_hydro,  # no evaporation_m3s column
            _ree_id_map(),
            {0: "A", 1: "B"},
        )

        assert results == []
        assert one_sided == []

    def test_no_stage_hours_available_degrades_to_no_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """No ``stages.json`` reconciliation denominator -- degrades
        gracefully rather than raising or fabricating a divisor."""
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._cobre_stage_hours",
            lambda *_a, **_k: {},
        )

        results, one_sided = _evaporation_result_comparisons(
            tmp_path,
            tmp_path,
            _evap_aligned_fixture().cobre_hydro,
            _ree_id_map(),
            {0: "A", 1: "B"},
        )

        assert results == []
        assert one_sided == []


class TestBuildDecompDatasetEvaporation:
    """ticket-020: fills ``results`` with per-(hydro, stage)
    ``evaporation_m3s`` rows and ``dataset.metadata["unmapped"]["evaporation"]``."""

    def test_no_deck_no_evaporation_rows_and_empty_unmapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_build_line_id_map`` returns ``None`` against a bare
        ``tmp_path`` (no deck to read) -> no evaporation rollup, empty
        ``unmapped["evaporation"]``, no exception."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        evap_rows = dataset.tidy.filter(
            (pl.col("entity_type") == "hydro")
            & (pl.col("variable") == "evaporation_m3s")
        )
        assert evap_rows.is_empty()
        assert dataset.metadata["unmapped"]["evaporation"] == []
        build_comparison_report(dataset)  # must not raise

    def test_both_sides_present_tidy_carries_evaporation_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: ``tidy`` has ``entity_type=="hydro"``/
        ``variable=="evaporation_m3s"`` rows with ``source`` in
        {"newave", "cobre"}, and ``dataset.summary`` includes the
        ``evaporation_m3s`` variable."""
        _patch_aligned_frames(monkeypatch, _evap_aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._build_line_id_map",
            lambda *_a, **_k: _ree_id_map(),
        )
        _patch_evap_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        evap_rows = dataset.tidy.filter(
            (pl.col("entity_type") == "hydro")
            & (pl.col("variable") == "evaporation_m3s")
        )
        assert set(evap_rows["source"].unique().to_list()) == {"newave", "cobre"}
        assert evap_rows.height == 4  # 2 plants * 2 sources
        assert dataset.metadata["unmapped"]["evaporation"] == []

        summary_vars = set(dataset.summary["variable"].to_list())
        assert "evaporation_m3s" in summary_vars

    def test_one_sided_plant_counted_in_dataset_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _evap_aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._build_line_id_map",
            lambda *_a, **_k: _ree_id_map(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_evap",
            lambda *_a, **_k: _evap_dec_oper_evap_fixture().filter(
                pl.col("codigo_usina") == 10
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._cobre_stage_hours",
            lambda *_a, **_k: {0: 168.0},
        )

        dataset = build_decomp_dataset(tmp_path, tmp_path)

        assert dataset.metadata["unmapped"]["evaporation"] == [1]

    def test_hydro_plant_detail_tab_carries_evaporation_content(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: ``build_comparison_report(dataset)`` renders the Hydro Plant
        Details tab's ``evaporation_m3s`` panel content -- the exact token
        ``charts._HYDRO_VARIABLES`` already wires into that tab, so no new
        chart is required (ticket-020 requirement 4)."""
        _patch_aligned_frames(monkeypatch, _evap_aligned_fixture())
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._build_line_id_map",
            lambda *_a, **_k: _ree_id_map(),
        )
        _patch_evap_sources(monkeypatch)

        dataset = build_decomp_dataset(tmp_path, tmp_path)
        html = build_comparison_report(dataset)

        assert 'id="tab-hydro-detail"' in html
        detail_tab = _extract_tab_content(html, "tab-hydro-detail")
        # The per-plant JSON payload's evaporation_m3s series is non-empty
        # for the one stage both fixture plants carry it at (compact JSON,
        # no spaces -- see ``ui.html.json_for_script``).
        assert '"evaporation_m3s_stages":[0]' in detail_tab


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetEvaporationE2E:
    """Tier 3 (dev-only smoke, ticket-020): the reduced deck's real
    evaporation comparison renders end to end. Both directories are
    gitignored, so this never runs in CI."""

    def test_evaporation_rows_render_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        evap_rows = dataset.tidy.filter(
            (pl.col("entity_type") == "hydro")
            & (pl.col("variable") == "evaporation_m3s")
        )
        assert not evap_rows.is_empty()
        # The real deck's Cobre run also carries a p10/p50/p90 percentile
        # band for evaporation_m3s (generic per-variable percentile
        # unpivoting, not ticket-020-specific) -- assert the two E1-shaped
        # sources are present rather than an exact source set.
        assert {"newave", "cobre"} <= set(evap_rows["source"].unique().to_list())

        html = build_comparison_report(dataset)
        assert 'id="tab-hydro-detail"' in html


# --- ticket-019: Constraints tab (gc_* metadata, DECOMP-side LHS) ---


def _re_record(
    constraint_id: int,
    terms: tuple[ConstraintTerm, ...],
    *,
    stage_start: int = 0,
    stage_end: int = 1,
    family: str = "RE",
) -> ConstraintRecord:
    """A minimal `ConstraintRecord` fixture carrying only the fields
    `_generic_constraint_lhs_decomp` reads (`family`, `constraint_id`,
    `stage_start`/`stage_end`, `terms`); `bounds` is never read by this
    ticket's LHS derivation."""
    return ConstraintRecord(
        family=family,
        constraint_id=constraint_id,
        stage_start=stage_start,
        stage_end=stage_end,
        terms=terms,
        bounds={},
        per_block=family in ("RE", "HQ"),
    )


def _usih_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    """A ``dec_oper_usih``-shaped frame: one stage-aggregate
    (``patamar=None``) row per (code, stage) -- the shape `_stage_rows`
    keeps."""
    base = {
        "no": 1,
        "cenario": 1,
        "patamar": None,
        "duracao": None,
        "vazao_defluente_m3s": 0.0,
        "vazao_turbinada_m3s": 0.0,
        "vazao_desviada_m3s": 0.0,
        "vazao_vertida_m3s": 0.0,
        "volume_util_final_hm3": 0.0,
        "geracao_MW": 0.0,
    }
    return pl.DataFrame([{**base, **row} for row in rows])


def _no_dec_oper(*_args: object, **_kwargs: object) -> pl.DataFrame:
    """A ``read_dec_oper_usih``/``read_dec_oper_usit`` stub for "this deck has
    no such table": raises ``FileNotFoundError`` like the real reader would,
    so `_dec_oper_hydro_stage_frame`/`_dec_oper_thermal_stage_frame`'s own
    degrade-to-empty ``except`` path is exercised -- a bare ``pl.DataFrame()``
    (no columns at all) is not a shape the real reader ever returns (it
    raises on an empty parse) and trips `_scenario_mean`'s ``group_by``."""
    raise FileNotFoundError("dec_oper_*.csv not found")


def _gc(cid: int, name: str, expression: str = "") -> dict[str, object]:
    """A minimal ``generic_constraints.json`` entry: only ``id``/``name``
    are read by `_generic_constraint_lhs_decomp` itself."""
    return {"id": cid, "name": name, "expression": expression}


def _write_generic_constraints_case(
    case_dir: Path,
    constraints: list[dict[str, Any]],
    bound_rows: list[dict[str, Any]],
) -> Path:
    """Write ``constraints/generic_constraints.json`` +
    ``constraints/generic_constraint_bounds.parquet`` under *case_dir* and
    return the Cobre output dir (``case_dir/output``) `case_dir_for`
    resolves back to *case_dir* from -- mirrors `_write_lines_json`."""
    constraints_dir = case_dir / "constraints"
    constraints_dir.mkdir(parents=True, exist_ok=True)
    (constraints_dir / "generic_constraints.json").write_text(
        json.dumps({"constraints": constraints})
    )
    pd.DataFrame(
        bound_rows,
        columns=[
            "constraint_id",
            "stage_id",
            "block_id",
            "bound_lower",
            "bound_upper",
        ],
    ).to_parquet(constraints_dir / "generic_constraint_bounds.parquet")
    output_dir = case_dir / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


class TestDecompConstraintContext:
    """`_decomp_constraint_context`: best-effort census + id map, mirroring
    `_build_line_id_map`'s degrade-gracefully pattern."""

    def test_returns_none_when_the_directory_has_no_deck(self, tmp_path: Path) -> None:
        assert _decomp_constraint_context(tmp_path) is None


class TestStageFrameToLookup:
    """`_stage_frame_to_lookup`: one `dec_oper_*` column -> {(code, stage):
    value}, 1-based `estagio` converted to 0-based `stage_id`."""

    def test_builds_zero_based_stage_lookup(self) -> None:
        frame = pl.DataFrame(
            {"codigo_usina": [10, 10], "estagio": [1, 2], "geracao_MW": [100.0, 90.0]}
        )
        assert _stage_frame_to_lookup(frame, "geracao_MW") == {
            (10, 0): 100.0,
            (10, 1): 90.0,
        }

    def test_empty_frame_returns_empty_dict(self) -> None:
        assert _stage_frame_to_lookup(pl.DataFrame(), "geracao_MW") == {}

    def test_missing_column_returns_empty_dict(self) -> None:
        frame = pl.DataFrame({"codigo_usina": [10], "estagio": [1]})
        assert _stage_frame_to_lookup(frame, "geracao_MW") == {}

    def test_null_values_are_skipped(self) -> None:
        frame = pl.DataFrame(
            {"codigo_usina": [10, 11], "estagio": [1, 1], "geracao_MW": [100.0, None]}
        )
        assert _stage_frame_to_lookup(frame, "geracao_MW") == {(10, 0): 100.0}


class TestStorageLookup:
    """`_storage_lookup`: absolute storage = useful volume + Vmin, Vmin
    resolved via the id map onto the cobre-side `min_storage_hm3` registry."""

    def test_adds_min_storage_floor_via_id_map(self) -> None:
        hydro_frame = pl.DataFrame(
            {"codigo_usina": [10], "estagio": [1], "volume_util_final_hm3": [120.0]}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        assert _storage_lookup(hydro_frame, id_map, {0: 30.0}) == {(10, 0): 150.0}

    def test_unmapped_code_is_excluded(self) -> None:
        hydro_frame = pl.DataFrame(
            {"codigo_usina": [99], "estagio": [1], "volume_util_final_hm3": [120.0]}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        assert _storage_lookup(hydro_frame, id_map, {0: 30.0}) == {}

    def test_missing_min_storage_entry_is_excluded(self) -> None:
        hydro_frame = pl.DataFrame(
            {"codigo_usina": [10], "estagio": [1], "volume_util_final_hm3": [120.0]}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        assert _storage_lookup(hydro_frame, id_map, {}) == {}


class TestTermLookupValue:
    """`_term_lookup_value`: dispatches one register term to its lookup."""

    def _lookups(self) -> _DecompConstraintLookups:
        return _DecompConstraintLookups(
            hydro_generation={(10, 0): 100.0},
            thermal_generation={(5, 0): 30.0},
            flow={"QDEF": {(10, 0): 80.0}},
            storage={(10, 0): 500.0},
        )

    def test_generation_term(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="generation")
        assert _term_lookup_value(term, self._lookups(), 0) == 100.0

    def test_thermal_generation_term(self) -> None:
        term = ConstraintTerm(code=5, coefficient=1.0, variable="thermal_generation")
        assert _term_lookup_value(term, self._lookups(), 0) == 30.0

    def test_varm_term(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="VARM")
        assert _term_lookup_value(term, self._lookups(), 0) == 500.0

    def test_flow_term(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="QDEF")
        assert _term_lookup_value(term, self._lookups(), 0) == 80.0

    def test_unresolvable_stage_returns_none(self) -> None:
        term = ConstraintTerm(code=10, coefficient=1.0, variable="generation")
        assert _term_lookup_value(term, self._lookups(), 5) is None

    def test_unsupported_variable_returns_none(self) -> None:
        term = ConstraintTerm(code=0, coefficient=1.0, variable="interchange")
        assert _term_lookup_value(term, self._lookups(), 0) is None


class TestUnsupportedTermVariables:
    def test_contains_interchange_and_qbom(self) -> None:
        assert _UNSUPPORTED_TERM_VARIABLES == frozenset({"interchange", "QBOM"})


class TestConstraintNameRe:
    """`_CONSTRAINT_NAME_RE`: recovers a cobre generic constraint's source
    family + register id from the emitter-authored ``name`` field."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("RE_401", ("RE", "401")),
            ("HQ_12", ("HQ", "12")),
            ("HV_5", ("HV", "5")),
            ("RHE_101", ("RHE", "101")),
        ],
    )
    def test_matches_known_prefixes(self, name: str, expected: tuple[str, str]) -> None:
        match = _CONSTRAINT_NAME_RE.match(name)
        assert match is not None
        assert match.groups() == expected

    @pytest.mark.parametrize("name", ["VminOP_1", "AGRINT_3", "RE401", ""])
    def test_rejects_unknown_names(self, name: str) -> None:
        assert _CONSTRAINT_NAME_RE.match(name) is None


class TestRheLhsLookup:
    """`_rhe_lhs_lookup`: RHE achieved LHS = valor_MW (the achieved value)."""

    def test_uses_valor_as_the_achieved_lhs(self) -> None:
        frame = pl.DataFrame(
            {
                "estagio": [4],
                "no": [4],
                "cenario": [1],
                "codigo_restricao": [101],
                "valor_MW": [34550.64],
                "violacao_absoluta_MW": [0.0],
            }
        )
        assert _rhe_lhs_lookup(frame) == {101: {3: 34550.64}}

    def test_uses_valor_not_the_bound_during_violation(self) -> None:
        # valor_MW is the achieved LHS; valor_MW + violacao_absoluta_MW would be
        # the target/bound (Meta), which must NOT be used as the LHS.
        frame = pl.DataFrame(
            {
                "estagio": [4],
                "no": [4],
                "cenario": [1],
                "codigo_restricao": [115],
                "valor_MW": [2951.58],
                "violacao_absoluta_MW": [145.83],
            }
        )
        lookup = _rhe_lhs_lookup(frame)
        assert lookup[115][3] == pytest.approx(2951.58)

    def test_empty_frame_returns_empty_dict(self) -> None:
        assert _rhe_lhs_lookup(pl.DataFrame()) == {}


class TestGenericConstraintLhsDecomp:
    """`_generic_constraint_lhs_decomp`: the DECOMP-side LHS derivation --
    this plan's least-certain crux (ticket-019)."""

    def test_no_constraints_returns_empty_schema(self, tmp_path: Path) -> None:
        result = _generic_constraint_lhs_decomp(tmp_path, tmp_path, [])
        assert result.schema == _GC_LHS_SCHEMA
        assert result.is_empty()

    def test_re_hydro_generation_terms_sum_by_coefficient(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        record = _re_record(
            401,
            (
                ConstraintTerm(code=155, coefficient=1.0, variable="generation"),
                ConstraintTerm(code=157, coefficient=1.0, variable="generation"),
            ),
        )
        census = ConstraintCensus(
            by_family={"RE": (record,), "HQ": (), "HV": (), "HE": ()}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(155, 157))
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._decomp_constraint_context",
            lambda *_a, **_k: _DecompConstraintContext(census=census, id_map=id_map),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [
                    {"codigo_usina": 155, "estagio": 1, "geracao_MW": 3000.0},
                    {"codigo_usina": 157, "estagio": 1, "geracao_MW": 3077.78},
                ]
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(tmp_path, tmp_path, [_gc(0, "RE_401")])

        row = result.row(0, named=True)
        assert row["constraint_id"] == 0
        assert row["stage_id"] == 0
        assert row["lhs_value"] == pytest.approx(6077.78)

    def test_hv_varm_term_uses_absolute_storage_floor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        record = _re_record(
            9001,
            (ConstraintTerm(code=10, coefficient=1.0, variable="VARM"),),
            family="HV",
            stage_start=0,
            stage_end=0,
        )
        census = ConstraintCensus(
            by_family={"RE": (), "HQ": (), "HV": (record,), "HE": ()}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(10,))
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._decomp_constraint_context",
            lambda *_a, **_k: _DecompConstraintContext(census=census, id_map=id_map),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [{"codigo_usina": 10, "estagio": 1, "volume_util_final_hm3": 120.0}]
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.cobre_readers."
            "read_cobre_hydro_metadata",
            lambda *_a, **_k: {0: {"min_storage_hm3": 30.0}},
        )

        result = _generic_constraint_lhs_decomp(tmp_path, tmp_path, [_gc(3, "HV_9001")])

        assert result.to_dicts() == [
            {"constraint_id": 3, "stage_id": 0, "lhs_value": 150.0}
        ]

    def test_rhe_soft_constraint_lhs_is_the_achieved_valor(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """AC: a soft (RHE) constraint's per-stage ``lhs_value`` is the
        operation's achieved value (``valor_MW``) -- NOT valor + the shortfall
        (which would be the target/bound ``Meta``). Verified on a constructed
        fixture, independent of any deck/census (RHE reads `DecOperRheSoft`)."""
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._decomp_constraint_context",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(
                {
                    "estagio": [4],
                    "no": [4],
                    "cenario": [1],
                    "codigo_restricao": [115],
                    "limite_MW": [3097.40],
                    "valor_MW": [2951.58],
                    "violacao_absoluta_MW": [145.83],
                    "violacao_percentual": [1.412423],
                }
            ),
        )

        result = _generic_constraint_lhs_decomp(tmp_path, tmp_path, [_gc(7, "RHE_115")])

        rows = result.to_dicts()
        assert len(rows) == 1
        assert rows[0]["constraint_id"] == 7
        assert rows[0]["stage_id"] == 3
        assert rows[0]["lhs_value"] == pytest.approx(2951.58)

    def test_interchange_term_skips_whole_constraint_cobre_only(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """An RE record mixing a resolvable ``generation`` term with an
        unresolvable ``interchange`` term must not fabricate a partial LHS
        from the resolvable term alone -- the whole constraint renders
        cobre-only."""
        record = _re_record(
            405,
            (
                ConstraintTerm(code=141, coefficient=1.0, variable="generation"),
                ConstraintTerm(
                    code=0,
                    coefficient=1.0,
                    variable="interchange",
                    submarket_de="SE",
                    submarket_para="S",
                ),
            ),
        )
        census = ConstraintCensus(
            by_family={"RE": (record,), "HQ": (), "HV": (), "HE": ()}
        )
        id_map = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(141,))
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._decomp_constraint_context",
            lambda *_a, **_k: _DecompConstraintContext(census=census, id_map=id_map),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [{"codigo_usina": 141, "estagio": 1, "geracao_MW": 1000.0}]
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(tmp_path, tmp_path, [_gc(2, "RE_405")])

        assert result.is_empty()

    def test_unrecognized_name_skips_constraint(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._decomp_constraint_context",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(
            tmp_path, tmp_path, [_gc(9, "VminOP_1")]
        )

        assert result.is_empty()

    def test_no_deck_context_yields_no_re_hq_hv_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_decomp_constraint_context`` returning ``None`` (no readable
        deck) must degrade RE/HQ/HV to cobre-only, never raise."""
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results._decomp_constraint_context",
            lambda *_a, **_k: None,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            lambda *_a, **_k: _usih_frame(
                [{"codigo_usina": 155, "estagio": 1, "geracao_MW": 3000.0}]
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(),
        )

        result = _generic_constraint_lhs_decomp(tmp_path, tmp_path, [_gc(0, "RE_401")])

        assert result.is_empty()


class TestBuildDecompDatasetConstraints:
    """ticket-019: fills ``gc_constraints``/``gc_bounds``/``gc_lhs_newave``/
    ``gc_lhs_cobre`` -- the cobre-side pieces reused verbatim from
    `constraints_compare`, the DECOMP-side LHS newly derived."""

    def test_no_generic_constraints_case_renders_empty_no_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        case_dir = tmp_path / "case"
        output_dir = _write_generic_constraints_case(case_dir, [], [])

        dataset = build_decomp_dataset(case_dir, output_dir)

        assert dataset.metadata["gc_constraints"] == []
        assert dataset.metadata["gc_lhs_newave"].is_empty()
        assert dataset.metadata["gc_lhs_cobre"].is_empty()
        html = build_comparison_report(dataset)  # must not raise
        constraints_tab = _extract_tab_content(html, "tab-constraints")
        assert "Generic Constraints — LHS vs Bound" in constraints_tab

    def test_populated_case_wires_gc_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        _patch_aligned_frames(monkeypatch, _aligned_fixture())
        case_dir = tmp_path / "case"
        constraints = [
            {
                "id": 0,
                "name": "RHE_115",
                "description": "RHE stored-energy constraint 115",
                "expression": "@rho_acum_h0 * hydro_storage(0)",
                "slack": {"enabled": True, "penalty": 1000.0},
            }
        ]
        bound_rows = [
            {
                "constraint_id": 0,
                "stage_id": 0,
                "block_id": None,
                "bound_lower": 3097.40,
                "bound_upper": None,
            }
        ]
        output_dir = _write_generic_constraints_case(case_dir, constraints, bound_rows)
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usih",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_usit",
            _no_dec_oper,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.read_dec_oper_rhesoft",
            lambda *_a, **_k: pl.DataFrame(
                {
                    "estagio": [1],
                    "no": [1],
                    "cenario": [1],
                    "codigo_restricao": [115],
                    "valor_MW": [2951.58],
                    "violacao_absoluta_MW": [145.83],
                }
            ),
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.constraints_compare.evaluate_lhs_cobre",
            lambda *_a, **_k: pl.DataFrame(
                {"constraint_id": [0], "stage_id": [0], "lhs_value": [3000.0]}
            ),
        )

        dataset = build_decomp_dataset(case_dir, output_dir)

        assert dataset.metadata["gc_constraints"] == constraints
        assert dataset.metadata["gc_bounds"].height == 1
        nw_row = dataset.metadata["gc_lhs_newave"].row(0, named=True)
        assert nw_row["constraint_id"] == 0
        assert nw_row["stage_id"] == 0
        assert nw_row["lhs_value"] == pytest.approx(2951.58)
        cb_row = dataset.metadata["gc_lhs_cobre"].row(0, named=True)
        assert cb_row == {"constraint_id": 0, "stage_id": 0, "lhs_value": 3000.0}

        html = build_comparison_report(dataset)
        constraints_tab = _extract_tab_content(html, "tab-constraints")
        assert "Generic Constraints — LHS vs Bound" in constraints_tab
        assert "Plotly.newPlot" in constraints_tab


@pytest.mark.skipif(
    not _REDUCED_DECOMP_DECK.is_dir() or not _REDUCED_COBRE_OUTPUT.is_dir(),
    reason="reduced deck + converted cobre output not present",
)
class TestBuildDecompDatasetConstraintsE2E:
    """Tier 3 (dev-only smoke, ticket-019): the reduced deck's real
    Constraints tab renders end to end. Both directories are gitignored, so
    this never runs in CI."""

    def test_constraints_tab_renders_on_the_real_deck(self) -> None:
        dataset = build_decomp_dataset(_REDUCED_DECOMP_DECK, _REDUCED_COBRE_OUTPUT)

        assert dataset.metadata["gc_constraints"]
        assert not dataset.metadata["gc_lhs_newave"].is_empty()

        html = build_comparison_report(dataset)
        constraints_tab = _extract_tab_content(html, "tab-constraints")
        assert "Generic Constraints — LHS vs Bound" in constraints_tab
        assert "Plotly.newPlot" in constraints_tab


class TestComparisonRecord:
    def test_stage_count_counts_distinct_stages(self) -> None:
        rows = pl.DataFrame(
            {
                "level": ["hydro"] * 3,
                "variable": ["generation"] * 3,
                "unit": ["MW"] * 3,
                "entity_id": [0, 0, 1],
                "entity_name": ["A", "A", "B"],
                "stage_id": [0, 1, 1],
                "source": [1.0, 2.0, 3.0],
                "cobre": [1.0, 2.0, 3.0],
                "delta": [0.0, 0.0, 0.0],
                "delta_pct": [0.0, 0.0, 0.0],
                "smape_pct": [0.0, 0.0, 0.0],
            }
        )
        comparison = DecompComparison(
            rows=rows,
            summary=_summarize(rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        assert comparison.stage_count == 2


def _fake_comparison() -> DecompComparison:
    rows = pl.DataFrame(
        {
            "level": ["hydro", "thermal"],
            "variable": ["generation", "generation"],
            "unit": ["MW", "MW"],
            "entity_id": [0, 0],
            "entity_name": ["A", "T"],
            "stage_id": [0, 0],
            "source": [100.0, 20.0],
            "cobre": [90.0, 20.0],
            "delta": [-10.0, 0.0],
            "delta_pct": [-10.0, 0.0],
            "smape_pct": [10.5, 0.0],
        }
    )
    return DecompComparison(
        rows=rows,
        summary=_summarize(rows),
        convergence=pl.DataFrame(
            {
                "iteration": [1, 2],
                "source_lower": [1.0, 2.0],
                "source_upper": [3.0, 3.0],
                "cobre_lower": [10.0, None],
                "cobre_upper": [30.0, None],
            }
        ),
        unmapped={"hydro": [], "thermal": [86, 224], "bus": []},
    )


def _fake_dataset(*, all_within_tol: bool = False) -> ComparisonDataset:
    """The canonical-dataset counterpart of ``_fake_comparison``: two stages,
    two variables. ``generation_mw`` always matches; ``turbined_m3s`` diverges
    unless *all_within_tol* asks for a fully-passing dataset instead."""
    tidy = pl.DataFrame(
        {
            "entity_type": ["hydro", "hydro", "hydro", "hydro"],
            "entity_id": [0, 0, 0, 0],
            "entity_name": ["A", "A", "A", "A"],
            "bus": [-1, -1, -1, -1],
            "stage": [0, 0, 1, 1],
            "block": [-1, -1, -1, -1],
            "variable": [
                "generation_mw",
                "generation_mw",
                "turbined_m3s",
                "turbined_m3s",
            ],
            "source": ["newave", "cobre", "newave", "cobre"],
            "value": [100.0, 100.0, 100.0, 100.0 if all_within_tol else 90.0],
        },
        schema=TIDY_SCHEMA,
    )
    turbined_within_tol_rate = 1.0 if all_within_tol else 0.0
    turbined_smape = 0.0 if all_within_tol else 0.12
    summary = pl.DataFrame(
        {
            "variable": ["generation_mw", "turbined_m3s"],
            "count": [1, 1],
            "mean_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "max_abs_diff": [0.0, 0.0 if all_within_tol else 10.0],
            "mean_smape": [0.0, turbined_smape],
            "max_smape": [0.0, turbined_smape],
            "within_tol_rate": [1.0, turbined_within_tol_rate],
            "correlation": [1.0, 0.9],
        },
        schema=SUMMARY_SCHEMA,
    )
    return ComparisonDataset(
        tidy=tidy,
        summary=summary,
        metadata={"unmapped": {"hydro": [], "thermal": [86, 224], "bus": []}},
    )


def _empty_fake_dataset() -> ComparisonDataset:
    return ComparisonDataset(
        tidy=pl.DataFrame(schema=TIDY_SCHEMA),
        summary=pl.DataFrame(schema=SUMMARY_SCHEMA),
        metadata={"unmapped": {"hydro": [], "thermal": [], "bus": []}},
    )


class TestDecompDatasetSummary:
    """``decomp_dataset_summary`` -- the superset ``--json`` payload builder
    that supersedes ``decomp_compare_summary`` at the ``compare decomp``
    call site (the legacy summary itself is untouched, per D-STRANGLER)."""

    def test_returns_the_superset_shape_in_the_documented_key_order(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert list(summary.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
            "stages",
            "variables",
            "unmapped",
        ]

    def test_headline_fields_match_build_compare_verdict(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["worst_variable"] == "turbined_m3s"
        assert summary["worst_smape"] == pytest.approx(0.12)
        assert summary["all_within_tol"] is False

    def test_stages_counts_distinct_tidy_stage_values(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["stages"] == 2

    def test_variables_is_the_dataset_summary_verbatim(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["variables"] == dataset.summary.to_dicts()

    def test_unmapped_is_sourced_from_dataset_metadata_with_int_codes(self) -> None:
        dataset = _fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["unmapped"] == {"hydro": [], "thermal": [86, 224], "bus": []}
        assert all(
            isinstance(code, int)
            for codes in summary["unmapped"].values()
            for code in codes
        )

    def test_all_within_tol_dataset_reports_no_worst_clause(self) -> None:
        dataset = _fake_dataset(all_within_tol=True)

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 2
        assert summary["total"] == 2
        assert summary["all_within_tol"] is True
        assert summary["worst_variable"] is None
        assert summary["worst_smape"] == 0.0

    def test_empty_dataset_reports_zero_totals_and_no_stages(self) -> None:
        dataset = _empty_fake_dataset()

        summary = decomp_dataset_summary(dataset, tolerance=1e-2)

        assert summary["within_tol"] == 0
        assert summary["total"] == 0
        assert summary["all_within_tol"] is False
        assert summary["stages"] == 0
        assert summary["variables"] == []
        assert summary["unmapped"] == {"hydro": [], "thermal": [], "bus": []}


class TestDecompCompareSummaryTolerance:
    """The within-tolerance verdict keys ``decomp_compare_summary`` appends."""

    def test_mixed_fixture_has_one_variable_exceeding_tolerance(self) -> None:
        """Hydro's ``smape_pct == 10.5`` exceeds 1e-2; thermal's ``0.0`` is within."""
        comparison = _fake_comparison()

        summary = decomp_compare_summary(comparison, tolerance=1e-2)

        assert list(summary.keys()) == [
            "stages",
            "variables",
            "unmapped",
            "within_tol",
            "total",
            "all_within_tol",
        ]
        assert summary["total"] == 2
        assert summary["within_tol"] == 1
        assert summary["all_within_tol"] is False

    def test_all_rows_within_tolerance_report_all_within_tol_true(self) -> None:
        rows = pl.DataFrame(
            {
                "level": ["hydro", "thermal"],
                "variable": ["generation", "generation"],
                "unit": ["MW", "MW"],
                "entity_id": [0, 0],
                "entity_name": ["A", "T"],
                "stage_id": [0, 0],
                "source": [100.0, 20.0],
                "cobre": [100.0, 20.0],
                "delta": [0.0, 0.0],
                "delta_pct": [0.0, 0.0],
                "smape_pct": [0.0, 0.0],
            }
        )
        comparison = DecompComparison(
            rows=rows,
            summary=_summarize(rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )

        summary = decomp_compare_summary(comparison, tolerance=1e-2)

        assert summary["total"] == 2
        assert summary["within_tol"] == 2
        assert summary["all_within_tol"] is True

    def test_empty_comparison_reports_zero_totals(self) -> None:
        empty_rows = _fake_comparison().rows.clear()
        comparison = DecompComparison(
            rows=empty_rows,
            summary=_summarize(empty_rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )

        summary = decomp_compare_summary(comparison, tolerance=1e-2)

        assert summary["within_tol"] == 0
        assert summary["total"] == 0
        assert summary["all_within_tol"] is False


class TestCompareDecompCommand:
    """The ``compare decomp`` subcommand, with the comparison itself stubbed."""

    @staticmethod
    def _invoke(
        argv: list[str],
        monkeypatch: pytest.MonkeyPatch,
        comparison: DecompComparison | None = None,
        dataset: ComparisonDataset | None = None,
    ) -> Any:
        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        resolved_comparison = (
            comparison if comparison is not None else _fake_comparison()
        )
        resolved_dataset = dataset if dataset is not None else _fake_dataset()
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results",
            lambda *_args, **_kwargs: resolved_comparison,
        )
        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.build_decomp_dataset",
            lambda *_args, **_kwargs: resolved_dataset,
        )
        return CliRunner().invoke(app, argv)

    def test_renders_tables_and_exits_zero(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)], monkeypatch
        )
        assert result.exit_code == 0
        assert "Operation comparison" in result.stdout
        assert "Final bounds" in result.stdout

    def test_headline_leads_stdout_on_a_diverging_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The shared ``build_compare_verdict`` headline, from ``_fake_dataset``'s
        mismatch (1/2 within tol, worst ``turbined_m3s`` at 12% sMAPE), leads
        stdout ahead of the legacy per-variable table."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)], monkeypatch
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "⚠ 1/2 variables within tol — worst: turbined_m3s sMAPE 12%"
        assert "Operation comparison" in result.stdout
        assert result.stdout.index("Operation comparison") > result.stdout.index(
            lines[0]
        )

    def test_headline_leads_stdout_when_all_within_tol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path)],
            monkeypatch,
            dataset=_fake_dataset(all_within_tol=True),
        )
        assert result.exit_code == 0
        lines = result.stdout.splitlines()
        assert lines[0] == "✓ 2/2 variables within tol"
        assert "Operation comparison" in result.stdout

    def test_json_carries_the_summary_and_unmapped_codes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``_fake_dataset``'s ``turbined_m3s`` row (``within_tol_rate=0.0``)
        diverges while ``generation_mw`` (``within_tol_rate=1.0``) does not, so
        the default fixture reports a mismatch."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"], monkeypatch
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert list(payload.keys()) == [
            "schema_version",
            "command",
            "status",
            "summary",
            "diagnostics",
        ]
        assert payload["command"] == "compare decomp"
        assert payload["status"] == "mismatch"
        summary = payload["summary"]
        assert list(summary.keys()) == [
            "within_tol",
            "total",
            "worst_variable",
            "worst_smape",
            "all_within_tol",
            "stages",
            "variables",
            "unmapped",
        ]
        assert summary["within_tol"] == 1
        assert summary["total"] == 2
        assert summary["worst_variable"] == "turbined_m3s"
        assert summary["worst_smape"] == pytest.approx(0.12)
        assert summary["all_within_tol"] is False
        assert summary["stages"] == 2
        assert {row["variable"] for row in summary["variables"]} == {
            "generation_mw",
            "turbined_m3s",
        }
        assert summary["unmapped"]["thermal"] == [86, 224]

    def test_json_status_is_ok_when_dataset_reports_all_within_tol(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the dataset's ``within_tol_rate`` is 1.0 for every variable
        (the tolerance was already applied upstream when the caller built the
        dataset), the CLI reports ``ok``."""
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
            dataset=_fake_dataset(all_within_tol=True),
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"
        assert payload["summary"]["all_within_tol"] is True

    def test_json_reports_no_comparable_rows_when_comparison_is_empty(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        empty_rows = _fake_comparison().rows.clear()
        empty_comparison = DecompComparison(
            rows=empty_rows,
            summary=_summarize(empty_rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(tmp_path), "--json"],
            monkeypatch,
            comparison=empty_comparison,
            dataset=_empty_fake_dataset(),
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["status"] == "no-comparable-rows"
        assert payload["summary"]["stages"] == 0
        assert payload["summary"]["variables"] == []
        assert payload["summary"]["within_tol"] == 0
        assert payload["summary"]["total"] == 0
        assert payload["summary"]["all_within_tol"] is False

    def test_unreadable_output_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _boom(*_args: object, **_kwargs: object) -> DecompComparison:
            raise FileNotFoundError("dec_oper_sist.csv not found")

        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        assert result.exit_code == 2

    def test_writes_artifacts_to_the_default_out_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cobre_output_dir = tmp_path / "cobre"
        result = self._invoke(
            ["compare", "decomp", str(tmp_path), str(cobre_output_dir)], monkeypatch
        )
        assert result.exit_code == 0
        artifacts = cobre_output_dir / "comparison_artifacts"
        assert (artifacts / "comparison.parquet").exists()
        assert (artifacts / "comparison.json").exists()

    def test_format_and_out_dir_flags_with_json_keep_stdout_pure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        cobre_output_dir = tmp_path / "cobre"
        other = tmp_path / "other"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(cobre_output_dir),
                "--format",
                "json",
                "--out-dir",
                str(other),
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["command"] == "compare decomp"
        assert (other / "summary.json").exists()
        assert "Artifacts written to" not in result.stdout

    def test_format_html_writes_the_shared_multi_tab_report_labelled_decomp(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """ticket-022: the HTML block now renders the SAME shared multi-tab
        report as ``compare newave`` (``build_comparison_report``), not the
        legacy single-page renderer, with the reference series labelled
        "DECOMP" (never "NEWAVE")."""
        out_dir = tmp_path / "artifacts"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--format",
                "html",
                "--out-dir",
                str(out_dir),
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        report_path = out_dir / "report.html"
        assert report_path.exists()
        report = report_path.read_text(encoding="utf-8")
        assert "tab-system" in report
        assert "tab-overview" in report
        assert "Storage by Bus" in report
        assert "DECOMP" in report
        assert "NEWAVE" not in report

    def test_format_html_advisory_routes_to_stderr_under_json(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "artifacts"
        result = self._invoke(
            [
                "compare",
                "decomp",
                str(tmp_path),
                str(tmp_path),
                "--format",
                "html",
                "--out-dir",
                str(out_dir),
                "--json",
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        assert (out_dir / "report.html").exists()
        json.loads(result.stdout)  # stdout carries only the JSON verdict
        assert "HTML report written to" not in result.stdout

    def test_partition_missing_output_exits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """FINDING-1 regression: CobrePartitionMissingError extends
        BridgeError, a hierarchy disjoint from CobreReadError (RuntimeError)
        and FileNotFoundError/ValueError. The compare decomp CLI handler
        must catch it too -- a clean ERROR line + exit 2, not an unhandled
        traceback -- mirroring the compare newave fix and this class's own
        CobreReadError-analogue test above."""
        from cobre_bridge.errors import CobrePartitionMissingError

        sim_dir = tmp_path / "cobre" / "simulation" / "hydro_bus_generation"

        def _boom(*_args: object, **_kwargs: object) -> DecompComparison:
            raise CobrePartitionMissingError(
                f"Cobre output partition not found: {sim_dir}. The "
                "hydro_bus_generation partition is produced by cobre "
                ">= 0.13.0; this output directory may predate that cobre "
                "version.",
                path=str(sim_dir),
            )

        from typer.testing import CliRunner

        from cobre_bridge.cli import app

        monkeypatch.setattr(
            "cobre_bridge.comparators.decomp_results.compare_decomp_results", _boom
        )
        result = CliRunner().invoke(
            app, ["compare", "decomp", str(tmp_path), str(tmp_path)]
        )
        # exit_code == 2 (not 1) proves this is the clean typer.Exit(code=2)
        # path, not an unhandled exception caught by CliRunner's default
        # catch_exceptions=True (which would report exit_code == 1).
        assert result.exit_code == 2
        assert "ERROR:" in result.stderr
        assert str(sim_dir) in result.stderr
        assert "0.13.0" in result.stderr
