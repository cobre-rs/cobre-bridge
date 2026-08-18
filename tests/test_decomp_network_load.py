"""Tests for the DECOMP bus and load converters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.load import convert_load_factors, convert_load_stats
from cobre_bridge.decomp.network import convert_buses, convert_lines
from cobre_bridge.decomp.temporal import OperativeStage, build_operative_calendar

_RV0_DECK = Path("example/decomp-set-24-rv0/dadger.rv0")
_RV3_DECK = Path("example/decomp-jul-26-rv3/dadger.rv3")

_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
)


def _calendar_rv3():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


class _StubDadger:
    """Minimal Dadger stand-in carrying sb/cd/dp/ia DataFrames."""

    def __init__(
        self,
        sb: pd.DataFrame | None = None,
        cd: pd.DataFrame | None = None,
        dp: pd.DataFrame | None = None,
        ia: pd.DataFrame | None = None,
    ) -> None:
        self._sb, self._cd, self._dp, self._ia = sb, cd, dp, ia

    def sb(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._sb

    def cd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._cd

    def dp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._dp

    def ia(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ia


def _cd_frame(
    cost: float = 7810.62,
    limit: float = 100.0,
    codes: tuple[int, ...] = (1, 2, 3, 4),
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_curva": 1,
                "codigo_submercado": code,
                "estagio": 1,
                "nome_curva": "1PDEF",
                "custo_1": cost,
                "custo_2": cost,
                "custo_3": cost,
                "limite_superior_1": limit,
                "limite_superior_2": limit,
                "limite_superior_3": limit,
            }
            for code in codes
        ]
    )


def _dp_frame(calendar_stages: int = 3) -> pd.DataFrame:
    rows = []
    weekly = {"duracao_1": 15.0, "duracao_2": 64.0, "duracao_3": 89.0}
    monthly = {"duracao_1": 63.0, "duracao_2": 280.0, "duracao_3": 401.0}
    for estagio in range(1, calendar_stages + 1):
        durations = monthly if estagio == calendar_stages else weekly
        for sbm, base in ((1, 40000.0), (2, 15000.0), (11, None)):
            rows.append(
                {
                    "codigo_submercado": sbm,
                    "estagio": estagio,
                    "numero_patamares": 3,
                    "carga_1": None if base is None else base * 1.2,
                    "carga_2": None if base is None else base,
                    "carga_3": None if base is None else base * 0.8,
                    **durations,
                }
            )
    return pd.DataFrame(rows)


class TestDecompIdMap:
    def test_deterministic_ids_and_transhipment(self) -> None:
        assert _ID_MAP.bus_id(1) == 0
        assert _ID_MAP.bus_id(11) == 4
        assert _ID_MAP.transhipment_bus_id == 5
        assert _ID_MAP.n_buses == 6
        assert _ID_MAP.bus_id_by_name("SE") == 0
        assert _ID_MAP.bus_id_by_name("IV") == 5
        assert _ID_MAP.bus_name(5) == "IV"

    def test_unknown_code_raises(self) -> None:
        with pytest.raises(KeyError, match="99"):
            _ID_MAP.bus_id(99)

    def test_reserved_name_rejected(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "IV"))

    @pytest.mark.skipif(not _RV0_DECK.exists(), reason="rv0 deck not present")
    def test_from_dadger_rv0(self) -> None:
        from idecomp.decomp import Dadger

        id_map = DecompIdMap.from_dadger(Dadger.read(str(_RV0_DECK)))
        assert id_map.bus_codes == (1, 2, 3, 4, 11)
        assert id_map.bus_names == ("SE", "S", "NE", "N", "FC")


class TestConvertBuses:
    def test_buses_with_and_without_deficit(self) -> None:
        dadger = _StubDadger(cd=_cd_frame())
        doc = convert_buses(dadger, _ID_MAP, date(2024, 8, 31))
        buses = doc["buses"]
        assert [b["id"] for b in buses] == [0, 1, 2, 3, 4, 5]
        assert buses[0]["deficit_segments"] == [{"depth_mw": None, "cost": 7810.62}]
        assert "deficit_segments" not in buses[4]  # FC
        assert buses[5]["name"] == "IV"
        assert "deficit_segments" not in buses[5]
        assert buses[0]["operational_start_date"] == "2024-08-31"

    def test_rejects_non_full_depth(self) -> None:
        dadger = _StubDadger(cd=_cd_frame(limit=95.0))
        with pytest.raises(ValueError, match="depth"):
            convert_buses(dadger, _ID_MAP, date(2024, 8, 31))

    def test_rejects_block_varying_cost(self) -> None:
        cd = _cd_frame()
        cd.loc[0, "custo_2"] = 9000.0
        with pytest.raises(ValueError, match="uniform"):
            convert_buses(_StubDadger(cd=cd), _ID_MAP, date(2024, 8, 31))

    def test_rejects_multi_segment_curves(self) -> None:
        cd = pd.concat([_cd_frame(), _cd_frame(cost=9000.0)], ignore_index=True)
        cd.loc[cd.index[-4:], "codigo_curva"] = 2
        with pytest.raises(ValueError, match="multi-segment"):
            convert_buses(_StubDadger(cd=cd), _ID_MAP, date(2024, 8, 31))


class TestConvertLoad:
    def test_stats_cover_every_bus_and_stage(self) -> None:
        table = convert_load_stats(
            _StubDadger(dp=_dp_frame()), _ID_MAP, _calendar_rv3()
        )
        df = table.to_pandas()
        assert len(df) == 6 * 3
        assert set(df["std_mw"]) == {0.0}
        # SE stage 0: (48000·15 + 40000·64 + 32000·89)/168
        se = df[(df["bus_id"] == 0) & (df["stage_id"] == 0)]["mean_mw"].iloc[0]
        assert se == pytest.approx((48000 * 15 + 40000 * 64 + 32000 * 89) / 168)
        # FC (NaN cargas) and IV (absent from DP) carry zero load.
        assert set(df[df["bus_id"].isin([4, 5])]["mean_mw"]) == {0.0}

    def test_factors_invariant_and_zero_mean_omission(self) -> None:
        calendar = _calendar_rv3()
        doc = convert_load_factors(_StubDadger(dp=_dp_frame()), _ID_MAP, calendar)
        entries = doc["load_factors"]
        assert {e["bus_id"] for e in entries} == {0, 1}
        for entry in entries:
            stage = calendar[entry["stage_id"]]
            weighted = sum(
                bf["factor"] * stage.block_hours[bf["block_id"]]
                for bf in entry["block_factors"]
            )
            assert weighted == pytest.approx(stage.total_hours, rel=1e-12)

    def test_rejects_block_count_mismatch(self) -> None:
        dp = _dp_frame()
        dp.loc[0, "numero_patamares"] = 2
        with pytest.raises(ValueError, match="blocks"):
            convert_load_stats(_StubDadger(dp=dp), _ID_MAP, _calendar_rv3())

    def test_rejects_stage_outside_calendar(self) -> None:
        dp = _dp_frame(calendar_stages=4)
        with pytest.raises(ValueError, match="outside the calendar"):
            convert_load_stats(_StubDadger(dp=dp), _ID_MAP, _calendar_rv3())


class TestRealDecks:
    @pytest.mark.skipif(not _RV0_DECK.exists(), reason="rv0 deck not present")
    def test_rv0_end_to_end(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV0_DECK))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        buses = convert_buses(dadger, id_map, calendar[0].start_date)["buses"]
        assert buses[0]["deficit_segments"][0]["cost"] == 7810.62

        stats = convert_load_stats(dadger, id_map, calendar).to_pandas()
        assert len(stats) == id_map.n_buses * len(calendar)
        assert (stats[stats["bus_id"] == 0]["mean_mw"] > 0).all()

        factors = convert_load_factors(dadger, id_map, calendar)["load_factors"]
        assert {e["bus_id"] for e in factors} == {0, 1, 2, 3}

    @pytest.mark.skipif(not _RV3_DECK.exists(), reason="rv3 deck not present")
    def test_rv3_end_to_end(self) -> None:
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(_RV3_DECK))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)

        buses = convert_buses(dadger, id_map, calendar[0].start_date)["buses"]
        assert buses[0]["deficit_segments"][0]["cost"] == 8291.25

        stats = convert_load_stats(dadger, id_map, calendar).to_pandas()
        se_stage0 = stats[(stats["bus_id"] == 0) & (stats["stage_id"] == 0)]
        expected = (47751.0 * 15 + 45562.0 * 64 + 38982.0 * 89) / 168.0
        assert se_stage0["mean_mw"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Ticket 008: convert_lines migration fidelity + zero-capability.
#
# Ticket 007's `test_block_bounds_are_absolute_mw_no_factor`
# (test_decomp_idecomp112.py) already pins the pass-through identity on a
# synthetic fixture. These tests pin the same identity plus a row-count
# guard on BOTH production decks -- independently re-reading the raw ``IA``
# dataframe rather than convert_lines'/``_ia_dense``'s own dense dict -- and
# the synthetic zero-capability case no real deck exercises (measured: zero
# zeros in either production deck's ``IA`` records).
# ---------------------------------------------------------------------------


def _declared_ia_rows(ia: pd.DataFrame) -> dict[tuple[str, str], dict[int, pd.Series]]:
    """Independent per-pair, per-declared-stage IA rows (0-based stage index).

    Mirrors the sparse-stage-inheritance contract documented on
    ``decomp.network._ia_dense`` ("declared stages forward-filled") --
    built fresh from the raw IA dataframe, never touching that function's
    own dense dict.
    """
    declared: dict[tuple[str, str], dict[int, pd.Series]] = {}
    for _, row in ia.iterrows():
        pair = (
            str(row["nome_submercado_de"]).strip(),
            str(row["nome_submercado_para"]).strip(),
        )
        declared.setdefault(pair, {})[int(row["estagio"]) - 1] = row
    return declared


def _forward_filled_row(declared: dict[int, pd.Series], stage_index: int) -> pd.Series:
    """The declared IA row governing *stage_index* under forward-fill
    inheritance: the latest declared stage at or before it."""
    candidates = [s for s in declared if s <= stage_index]
    return declared[max(candidates)]


class TestConvertLinesRealDeckFidelity:
    """Ticket 008 acceptance criteria 2 + 4: block rows equal the ``IA``
    record's own per-block limit columns exactly on both production decks,
    and the per-block row count is asserted against the number of (line,
    stage) pairs whose blocks genuinely differ from the base -- this
    pipeline emits block rows per (line, stage), not per block, so a
    non-uniform stage's *every* block gets a row, including any block that
    individually equals the base (see ``convert_lines``'s ``uniform`` check).
    """

    def _load(
        self, deck_path: Path
    ) -> tuple[
        list[dict],
        pd.DataFrame,
        Sequence[OperativeStage],
        DecompIdMap,
        dict[tuple[str, str], dict[int, pd.Series]],
    ]:
        if not deck_path.exists():
            pytest.skip("real deck not present")
        from idecomp.decomp import Dadger

        from cobre_bridge.decomp.temporal import operative_calendar_from_dadger

        dadger = Dadger.read(str(deck_path))
        id_map = DecompIdMap.from_dadger(dadger)
        calendar = operative_calendar_from_dadger(dadger)
        lines_doc, bounds = convert_lines(
            dadger, id_map, calendar, calendar[0].start_date
        )
        declared = _declared_ia_rows(dadger.ia(df=True))
        return lines_doc["lines"], bounds.to_pandas(), calendar, id_map, declared

    def _expected_blocks(
        self,
        lines: list[dict],
        calendar: Sequence[OperativeStage],
        id_map: DecompIdMap,
        declared: dict[tuple[str, str], dict[int, pd.Series]],
    ) -> dict[tuple[int, int], tuple[list[float], list[float]] | None]:
        """``{(line_id, stage_index): (direct_vals, reverse_vals) | None}``.

        ``None`` marks a (line, stage) whose blocks are uniform with the
        base -- no block rows expected. Otherwise the full per-block
        direct/reverse value lists, read straight off the forward-filled
        IA row.
        """
        expected: dict[tuple[int, int], tuple[list[float], list[float]] | None] = {}
        for line in lines:
            pair = (
                id_map.bus_name(line["source_bus_id"]),
                id_map.bus_name(line["target_bus_id"]),
            )
            for stage in calendar:
                row = _forward_filled_row(declared[pair], stage.index)
                n_blocks = len(stage.block_hours)
                direct_vals = [
                    float(row[f"limite_de_para_{k}"]) for k in range(1, n_blocks + 1)
                ]
                reverse_vals = [
                    float(row[f"limite_para_de_{k}"]) for k in range(1, n_blocks + 1)
                ]
                base_direct, base_reverse = max(direct_vals), max(reverse_vals)
                uniform = all(
                    d == base_direct and r == base_reverse
                    for d, r in zip(direct_vals, reverse_vals, strict=True)
                )
                key = (line["id"], stage.index)
                expected[key] = None if uniform else (direct_vals, reverse_vals)
        return expected

    @pytest.mark.parametrize("deck_path", [_RV0_DECK, _RV3_DECK], ids=["rv0", "rv3"])
    def test_block_rows_are_the_ia_columns_verbatim(self, deck_path: Path) -> None:
        lines, df, calendar, id_map, declared = self._load(deck_path)
        expected = self._expected_blocks(lines, calendar, id_map, declared)

        checked = 0
        for (line_id, stage_index), values in expected.items():
            if values is None:
                continue
            direct_vals, reverse_vals = values
            block_rows = df[
                (df["line_id"] == line_id)
                & (df["stage_id"] == stage_index)
                & df["block_id"].notna()
            ]
            for b in range(len(direct_vals)):
                emitted = block_rows[block_rows["block_id"] == b]
                assert len(emitted) == 1
                assert emitted.iloc[0]["direct_mw"] == direct_vals[b]
                assert emitted.iloc[0]["reverse_mw"] == reverse_vals[b]
                checked += 1
        assert checked > 0, (
            "the real deck must exercise at least one differing (line, "
            "stage) pair for this fidelity check to be meaningful"
        )

    @pytest.mark.parametrize("deck_path", [_RV0_DECK, _RV3_DECK], ids=["rv0", "rv3"])
    def test_block_row_count_matches_differing_stage_pairs(
        self, deck_path: Path
    ) -> None:
        lines, df, calendar, id_map, declared = self._load(deck_path)
        expected = self._expected_blocks(lines, calendar, id_map, declared)

        expected_rows = sum(
            len(values[0]) for values in expected.values() if values is not None
        )
        actual_rows = int(df["block_id"].notna().sum())
        assert expected_rows > 0, (
            "the real deck must exercise at least one differing (line, "
            "stage) pair for this row-count guard to be meaningful"
        )
        assert actual_rows == expected_rows


def _ia_zero_block_frame() -> pd.DataFrame:
    """One SE-IV exchange row whose block-0 ``de_para`` limit sits at
    exactly zero and whose other blocks are non-zero and distinct from the
    base -- the shape the deleted multiplicative-factor encoding could
    never represent (factors are strictly positive)."""
    return pd.DataFrame(
        [
            {
                "estagio": 1,
                "nome_submercado_de": "SE",
                "nome_submercado_para": "IV",
                "limite_de_para_1": 0.0,
                "limite_de_para_2": 5000.0,
                "limite_de_para_3": 6000.0,
                "limite_para_de_1": 9000.0,
                "limite_para_de_2": 9000.0,
                "limite_para_de_3": 9000.0,
            }
        ]
    )


class TestConvertLinesZeroCapability:
    """cobre decision 10 makes ``direct_mw = 0.0`` an ordinary bound; the
    ``raise`` this replaced only ever fired on a zero-limit round-trip that
    no longer exists (see the capability-gain comment at ``convert_lines``,
    ``decomp/network.py``). No current deck exercises a zero IA limit
    (measured: zero zeros in both production decks' ``IA`` records), so
    this synthetic fixture pins the new capability (ticket 008 acceptance
    criterion 3)."""

    def test_zero_block_limit_converts_without_raising(self) -> None:
        calendar = _calendar_rv3()
        lines_doc, bounds = convert_lines(
            _StubDadger(ia=_ia_zero_block_frame()),
            _ID_MAP,
            calendar,
            date(2026, 7, 18),
        )  # must not raise
        line_id = lines_doc["lines"][0]["id"]
        df = bounds.to_pandas()

        zero_row = df[
            (df["line_id"] == line_id) & (df["stage_id"] == 0) & (df["block_id"] == 0)
        ]
        assert len(zero_row) == 1
        assert zero_row.iloc[0]["direct_mw"] == 0.0
        assert zero_row.iloc[0]["reverse_mw"] == 9000.0

        # Block 1 is non-zero and distinct from the base (6000.0), proving
        # it is genuinely unaffected rather than coincidentally matching it.
        other_block = df[
            (df["line_id"] == line_id) & (df["stage_id"] == 0) & (df["block_id"] == 1)
        ]
        assert len(other_block) == 1
        assert other_block.iloc[0]["direct_mw"] == 5000.0
        assert other_block.iloc[0]["reverse_mw"] == 9000.0
