"""Tests for the DECOMP-vs-Cobre results comparison slice."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import pytest

from cobre_bridge.comparators.charts import _BALANCE_VARS, _COST_MAP
from cobre_bridge.comparators.dataset import (
    SUMMARY_SCHEMA,
    TIDY_SCHEMA,
    ComparisonDataset,
)
from cobre_bridge.comparators.decomp_html_report import build_decomp_comparison_report
from cobre_bridge.comparators.decomp_results import (
    _BUS_VARIABLES,
    _CANONICAL_VARIABLE,
    _DEVIATION_VIOLATION_LABEL,
    _HYDRO_VARIABLES,
    _NW_COST_LABELS,
    _THERMAL_VARIABLES,
    DecompComparison,
    _AlignedDecompFrames,
    _build_line_id_map,
    _bus_side,
    _corridor_line_alignment,
    _cost_frames,
    _energy_balance_frames,
    _interc_side,
    _line_bounds_and_meta,
    _line_entity_names,
    _line_result_comparisons,
    _map_entities,
    _read_cobre_lines_index,
    _result_comparisons,
    _scenario_mean,
    _stage_rows,
    _summarize,
    _tidy,
    _union_cost_rows,
    build_decomp_dataset,
    compare_decomp_results,
)
from cobre_bridge.comparators.report_builder import build_comparison_report
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


class TestBuildDecompDataset:
    """``build_decomp_dataset`` assembles the canonical dataset for the
    current 8 DECOMP variables via the shared ``_read_aligned_frames`` seam."""

    def test_dataset_validates_with_the_eight_canonical_variables(
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
        every level ``_read_aligned_frames`` reports. ``"line"`` is a
        ticket-008 addition sourced outside ``_read_aligned_frames`` (the
        legacy comparison never gained line rows), so it is compared
        separately rather than folded into the shared-levels equality."""
        _patch_aligned_frames(monkeypatch, _aligned_fixture())

        legacy = compare_decomp_results(tmp_path, tmp_path)
        dataset = build_decomp_dataset(tmp_path, tmp_path)

        dataset_unmapped = dict(dataset.metadata["unmapped"])
        assert dataset_unmapped.pop("line") == []
        assert dataset_unmapped == legacy.unmapped


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


class TestBuildDecompComparisonReport:
    """``build_decomp_comparison_report`` renders the same frames as the console."""

    def test_populated_comparison_renders_all_three_frames(self) -> None:
        comparison = _fake_comparison()

        report = build_decomp_comparison_report(comparison)

        assert "<!DOCTYPE html>" in report
        assert "Operation comparison" in report
        assert "generation" in report
        assert "Final bounds" in report

    def test_unmapped_codes_appear_in_the_report(self) -> None:
        """``unmapped={"thermal": [86, 224]}`` in the fixture must surface."""
        report = build_decomp_comparison_report(_fake_comparison())

        assert "86" in report
        assert "224" in report

    def test_empty_comparison_short_circuits_without_raising(self) -> None:
        empty_rows = _fake_comparison().rows.clear()
        comparison = DecompComparison(
            rows=empty_rows,
            summary=_summarize(empty_rows),
            convergence=pl.DataFrame(schema={"iteration": pl.Int64}),
            unmapped={"hydro": [], "thermal": [], "bus": []},
        )

        report = build_decomp_comparison_report(comparison)

        assert "no comparable rows" in report
        assert "<!DOCTYPE html>" in report


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

    def test_format_html_writes_a_self_contained_report(
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
            ],
            monkeypatch,
        )
        assert result.exit_code == 0
        report_path = out_dir / "report.html"
        assert report_path.exists()
        assert "Operation comparison" in report_path.read_text(encoding="utf-8")

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
