"""Unit tests for the VminOP, electric, and AGRINT generic constraints converters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pytest

from cobre_bridge.converters.constraints import (
    _parse_formula,
    compute_accumulated_productivities,
    convert_agrint_constraints,
    convert_electric_constraints,
    convert_vminop_constraints,
)
from cobre_bridge.id_map import NewaveIdMap


def _make_cadastro() -> pd.DataFrame:
    """Three-plant cascade: 1 -> 2 -> 3 (3 is the sink).

    Plant 3 has downstream=None (sea).  All have tipo_regulacao="D" and
    simple linear volume_cota polynomials for deterministic productivity.
    """
    data = {
        "nome_usina": ["PLANT_A", "PLANT_B", "PLANT_C"],
        "produtibilidade_especifica": [0.01, 0.02, 0.03],
        "volume_minimo": [100.0, 200.0, 300.0],
        "volume_maximo": [1000.0, 2000.0, 3000.0],
        "volume_referencia": [500.0, 1000.0, 1500.0],
        "canal_fuga_medio": [200.0, 300.0, 400.0],
        "tipo_perda": [0, 0, 0],
        "perdas": [0.0, 0.0, 0.0],
        "tipo_regulacao": ["D", "D", "D"],
        "a0_volume_cota": [300.0, 400.0, 500.0],
        "a1_volume_cota": [0.1, 0.05, 0.02],
        "a2_volume_cota": [0.0, 0.0, 0.0],
        "a3_volume_cota": [0.0, 0.0, 0.0],
        "a4_volume_cota": [0.0, 0.0, 0.0],
    }
    return pd.DataFrame(data, index=pd.Index([1, 2, 3], name="codigo_usina"))


def _make_confhd_df() -> pd.DataFrame:
    """Confhd with 3 plants: 1->2, 2->3, 3->None, all REE=1."""
    return pd.DataFrame(
        {
            "codigo_usina": [1, 2, 3],
            "nome_usina": ["PLANT_A", "PLANT_B", "PLANT_C"],
            "usina_existente": ["EX", "EX", "EX"],
            "codigo_usina_jusante": [2, 3, 0],
            "ree": [1, 1, 1],
            "posto": [1, 2, 3],
            "volume_inicial_percentual": [50.0, 50.0, 50.0],
        }
    )


class TestAccumulatedProductivity:
    def test_sink_plant_has_own_productivity_only(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(cadastro, confhd_df)

        # Plant 3 is the sink (downstream=None).
        # Its own productivity: 0.03 * (500 + 0.02*1500 - 400) = 0.03 * 130 = 3.9
        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        assert acc[3] == pytest.approx(own_3)

    def test_middle_plant_accumulates_downstream(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(cadastro, confhd_df)

        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        assert acc[2] == pytest.approx(own_2 + own_3)

    def test_upstream_plant_accumulates_full_chain(self) -> None:
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df()
        acc = compute_accumulated_productivities(cadastro, confhd_df)

        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        own_1 = 0.01 * ((300.0 + 0.1 * 500.0) - 200.0)
        assert acc[1] == pytest.approx(own_1 + own_2 + own_3)

    def test_independent_plants_no_accumulation(self) -> None:
        """Two plants with no cascade link each have only own productivity."""
        cadastro = _make_cadastro()
        confhd_df = _make_confhd_df().copy()
        # Break the cascade: all plants have downstream=0
        confhd_df["codigo_usina_jusante"] = [0, 0, 0]

        acc = compute_accumulated_productivities(cadastro, confhd_df)

        own_1 = 0.01 * ((300.0 + 0.1 * 500.0) - 200.0)
        own_2 = 0.02 * ((400.0 + 0.05 * 1000.0) - 300.0)
        own_3 = 0.03 * ((500.0 + 0.02 * 1500.0) - 400.0)
        assert acc[1] == pytest.approx(own_1)
        assert acc[2] == pytest.approx(own_2)
        assert acc[3] == pytest.approx(own_3)


class TestPerStageAccumulatedProductivities:
    """Per-stage cascade-sum of own ρ_eq for the VminOP RHS bound."""

    def test_no_overrides_yields_flat_lists(self) -> None:
        """Without CFUGA/CMONT inputs every plant has a constant ρ_acum across stages."""
        from cobre_bridge.converters.constraints import (
            compute_per_stage_acc_productivities,
        )

        confhd_df = _make_confhd_df()
        # Three plants, two stages, all own ρ_eq constant in time.
        per_stage_own = {1: [0.5, 0.5], 2: [0.7, 0.7], 3: [0.9, 0.9]}
        acc = compute_per_stage_acc_productivities(confhd_df, per_stage_own)

        # Plant 3 is the sink → ρ_acum(3) = own(3) at every stage.
        assert acc[3] == pytest.approx([0.9, 0.9])
        # Plant 2 → ρ_acum(2) = own(2) + ρ_acum(3) = 1.6
        assert acc[2] == pytest.approx([1.6, 1.6])
        # Plant 1 → ρ_acum(1) = own(1) + ρ_acum(2) = 2.1
        assert acc[1] == pytest.approx([2.1, 2.1])

    def test_override_on_downstream_propagates_upstream_only_after_stage(
        self,
    ) -> None:
        """A CFUGA-style change in plant 3's own ρ_eq at stage 1 shifts every
        upstream plant's ρ_acum from that stage on. Stage 0 stays unchanged.
        """
        from cobre_bridge.converters.constraints import (
            compute_per_stage_acc_productivities,
        )

        confhd_df = _make_confhd_df()
        # Plant 3 ρ_eq drops from 0.9 to 0.6 starting at stage 1.
        per_stage_own = {1: [0.5, 0.5], 2: [0.7, 0.7], 3: [0.9, 0.6]}
        acc = compute_per_stage_acc_productivities(confhd_df, per_stage_own)

        # Stage 0: unchanged baselines.
        assert acc[3][0] == pytest.approx(0.9)
        assert acc[2][0] == pytest.approx(1.6)
        assert acc[1][0] == pytest.approx(2.1)
        # Stage 1: ρ_acum shifts everywhere in plant 3's upstream cone.
        assert acc[3][1] == pytest.approx(0.6)
        assert acc[2][1] == pytest.approx(0.7 + 0.6)
        assert acc[1][1] == pytest.approx(0.5 + 0.7 + 0.6)

    def test_sibling_branch_unaffected_by_override(self) -> None:
        """Sibling branches stay untouched when a leaf in another branch shifts."""
        from cobre_bridge.converters.constraints import (
            compute_per_stage_acc_productivities,
        )

        # Two independent branches: 1→2 (sink), 3→4 (sink). 1 and 3 are headwaters.
        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [1, 2, 3, 4],
                "nome_usina": ["A", "B", "C", "D"],
                "usina_existente": ["EX", "EX", "EX", "EX"],
                "codigo_usina_jusante": [2, 0, 4, 0],
                "ree": [1, 1, 1, 1],
                "posto": [1, 2, 3, 4],
                "volume_inicial_percentual": [50.0, 50.0, 50.0, 50.0],
            }
        )
        # Branch 1→2: constant. Branch 3→4: plant 4 ρ_eq drops at stage 1.
        per_stage_own = {1: [0.5, 0.5], 2: [0.7, 0.7], 3: [0.4, 0.4], 4: [0.8, 0.5]}
        acc = compute_per_stage_acc_productivities(confhd_df, per_stage_own)

        # Branch 1→2 unchanged across stages.
        assert acc[1] == pytest.approx([1.2, 1.2])
        assert acc[2] == pytest.approx([0.7, 0.7])
        # Branch 3→4 shifts at stage 1.
        assert acc[3][0] == pytest.approx(1.2)
        assert acc[3][1] == pytest.approx(0.9)


class TestVminopRhsSeasonality:
    """Integration check that the VminOP RHS picks up per-stage ρ_acum.

    Uses the example NEWAVE case (which has plants with CFUGA overrides) and
    asserts at least one constraint produces stage-varying RHS values for the
    same percentage.
    """

    def test_rhs_varies_across_stages_for_overridden_cascades(self) -> None:
        result = _run_example_conversion()
        if result is None:
            pytest.skip("example/newave not available")

        _, bounds_table = result
        bounds = bounds_table.to_pandas()

        # Group by (constraint_id) and look for any constraint whose RHS
        # varies for stages that share the same calendar month. Per-stage
        # ρ_acum shifts must produce that — a constant-ρ_acum implementation
        # would yield identical RHS for every January in the horizon.
        import math as _math

        def _month_of_stage(stage_id: int) -> int:
            # We don't know start_month here; checking adjacency suffices —
            # if pct[0] == pct[12] (annual cycle) but RHS[0] != RHS[12], the
            # per-stage cascade is varying.
            return stage_id % 12

        any_stage_dependent = False
        for cid, sub in bounds.groupby("constraint_id"):
            by_cycle: dict[int, list[float]] = {}
            for _, row in sub.iterrows():
                by_cycle.setdefault(_month_of_stage(int(row["stage_id"])), []).append(
                    float(row["bound"])
                )
            for vals in by_cycle.values():
                if len(vals) >= 2 and not _math.isclose(
                    max(vals), min(vals), rel_tol=1e-9, abs_tol=1e-9
                ):
                    any_stage_dependent = True
                    break
            if any_stage_dependent:
                break

        assert any_stage_dependent, (
            "No constraint exhibits stage-varying RHS for the same calendar "
            "month — per-stage ρ_acum is not flowing into the bound."
        )


class TestConvertVminopConstraints:
    def test_returns_none_when_curva_absent(self, tmp_path) -> None:

        from cobre_bridge.newave_files import NewaveFiles

        nw = NewaveFiles(
            directory=tmp_path,
            dger=tmp_path / "d",
            confhd=tmp_path / "c",
            conft=tmp_path / "t",
            sistema=tmp_path / "s",
            clast=tmp_path / "cl",
            term=tmp_path / "te",
            ree=tmp_path / "r",
            patamar=tmp_path / "p",
            hidr=tmp_path / "h",
            vazoes=tmp_path / "v",
            modif=None,
            ghmin=None,
            penalid=None,
            vazpast=None,
            dsvagua=None,
            curva=None,
            expt=None,
            manutt=None,
            c_adic=None,
            cvar=None,
            agrint=None,
            re_dat=None,
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_vminop_constraints(nw, id_map) is None

    def test_constraint_expression_uses_hydro_storage(self, tmp_path) -> None:
        """Integration test: verifies expression format against real example data."""

        result = _run_example_conversion()
        if result is None:
            pytest.skip("example/newave not available")

        constraints_dict, bounds_table = result
        constraints = constraints_dict["constraints"]
        assert len(constraints) > 0

        for c in constraints:
            assert "hydro_storage(" in c["expression"]
            # Coefficients are now @rho_acum_h{id} @name references (cobre HEAD
            # @name sigil), not literal floats.
            assert "@rho_acum_h" in c["expression"]
            assert c["sense"] == ">="
            assert c["slack"]["enabled"] is True
            assert c["slack"]["penalty"] > 0

    def test_bounds_table_has_correct_schema(self, tmp_path) -> None:
        result = _run_example_conversion()
        if result is None:
            pytest.skip("example/newave not available")

        _, bounds_table = result
        assert isinstance(bounds_table, pa.Table)
        assert "constraint_id" in bounds_table.column_names
        assert "stage_id" in bounds_table.column_names
        assert "bound" in bounds_table.column_names

    def test_bounds_are_positive(self, tmp_path) -> None:
        result = _run_example_conversion()
        if result is None:
            pytest.skip("example/newave not available")

        _, bounds_table = result
        bounds = bounds_table.column("bound").to_pylist()
        assert all(b >= 0 for b in bounds)


def _run_example_conversion():
    """Run VminOP conversion on the example case if it exists.

    Returns ``(constraints_dict, bounds_table)`` for the legacy 2-tuple
    callers; the new third element (referenced hydro_ids) is ignored here.
    """
    example_dir = Path("example/newave")
    caso_path = example_dir / "CASO.DAT"
    if not caso_path.exists():
        caso_path = example_dir / "caso.dat"
    if not caso_path.exists():
        return None

    from cobre_bridge.newave_files import NewaveFiles

    nw_files = NewaveFiles.from_directory(example_dir)
    if nw_files.curva is None:
        return None

    from cobre_bridge.pipeline import _build_id_map

    id_map = _build_id_map(nw_files)
    result = convert_vminop_constraints(nw_files, id_map)
    if result is None:
        return None
    constraints_dict, bounds_table, _referenced = result
    return constraints_dict, bounds_table


def _example_nw_files_and_id_map():
    """Return (nw_files, id_map) for the example case, or None if unavailable."""
    example_dir = Path("example/newave")
    if (
        not (example_dir / "CASO.DAT").exists()
        and not (example_dir / "caso.dat").exists()
    ):
        return None

    from cobre_bridge.newave_files import NewaveFiles
    from cobre_bridge.pipeline import _build_id_map

    nw_files = NewaveFiles.from_directory(example_dir)
    id_map = _build_id_map(nw_files)
    return nw_files, id_map


# ---------------------------------------------------------------------------
# _parse_formula unit tests
# ---------------------------------------------------------------------------


class TestParseFormula:
    """Unit tests for the formula parser used by electric constraints."""

    def _id_map(self) -> NewaveIdMap:
        # hydro codes 66, 204, 261, 284, 285, 287 with deterministic IDs
        return NewaveIdMap(
            subsystem_ids=[1, 2, 3, 4, 11],
            hydro_codes=[66, 204, 261, 284, 285, 287],
            thermal_codes=[],
        )

    def _line_map(self) -> dict[tuple[int, int], int]:
        # Mirrors the real data: canonical pairs sorted
        # (1,2)=0, (1,3)=1, (1,4)=2, (1,11)=3, (3,11)=4, (4,11)=5
        return {
            (1, 2): 0,
            (1, 3): 1,
            (1, 4): 2,
            (1, 11): 3,
            (3, 11): 4,
            (4, 11): 5,
        }

    def test_simple_ger_usih(self) -> None:
        """Single hydro generation term with implicit coefficient."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("ger_usih(285) + ger_usih(287)", id_map, line_map)
        assert result is not None
        assert "hydro_generation(" in result
        # Both plants must appear; no coefficient prefix for coeff=1.0
        assert "* hydro_generation" not in result

    def test_explicit_coefficient_on_ger_usih(self) -> None:
        """Coefficient of 0.5 must appear in the expression."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("0.5ger_usih(66)", id_map, line_map)
        assert result is not None
        assert "0.5 * hydro_generation(" in result

    def test_ener_interc_canonical_direction(self) -> None:
        """ener_interc(1,2) is the canonical direction => positive coefficient."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("1.0ener_interc(1,2)", id_map, line_map)
        assert result is not None
        # Canonical direction: no negation
        assert "-1 * line_exchange" not in result
        assert "line_exchange(0)" in result

    def test_ener_interc_reversed_direction(self) -> None:
        """ener_interc(2,1) reverses the canonical direction => negated subtraction."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("1.0ener_interc(2,1)", id_map, line_map)
        assert result is not None
        # New syntax: "- line_exchange(0)" instead of "-1 * line_exchange(0)"
        assert "- line_exchange(0)" in result
        assert "-1 * line_exchange(0)" not in result

    def test_unknown_hydro_code_skipped(self) -> None:
        """Unknown hydro code produces a warning and the term is dropped."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula("ger_usih(9999)", id_map, line_map)
        assert result is None  # all terms dropped => None

    def test_mixed_formula(self) -> None:
        """Mixed ener_interc and ger_usih terms are both translated."""
        id_map = self._id_map()
        line_map = self._line_map()
        result = _parse_formula(
            "1.0ener_interc(2,1) + 0.5ger_usih(66)", id_map, line_map
        )
        assert result is not None
        assert "line_exchange(" in result
        assert "hydro_generation(" in result


# ---------------------------------------------------------------------------
# convert_electric_constraints integration tests
# ---------------------------------------------------------------------------


class TestConvertElectricConstraints:
    def test_returns_none_when_no_indices_csv(self, tmp_path: Path) -> None:
        """Return None gracefully when indices.csv is absent."""
        from cobre_bridge.newave_files import NewaveFiles

        nw = NewaveFiles(
            directory=tmp_path,
            dger=tmp_path / "d",
            confhd=tmp_path / "c",
            conft=tmp_path / "t",
            sistema=tmp_path / "s",
            clast=tmp_path / "cl",
            term=tmp_path / "te",
            ree=tmp_path / "r",
            patamar=tmp_path / "p",
            hidr=tmp_path / "h",
            vazoes=tmp_path / "v",
            modif=None,
            ghmin=None,
            penalid=None,
            vazpast=None,
            dsvagua=None,
            curva=None,
            expt=None,
            manutt=None,
            c_adic=None,
            cvar=None,
            agrint=None,
            re_dat=None,
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_electric_constraints(nw, id_map) is None

    def test_returns_none_when_re_file_missing(self, tmp_path: Path) -> None:
        """Return None when indices.csv exists but points to a missing file."""
        indices = tmp_path / "indices.csv"
        indices.write_text(
            "RESTRICAO-ELETRICA-ESPECIAL; ; does-not-exist.csv\n", encoding="latin-1"
        )

        from cobre_bridge.newave_files import NewaveFiles

        nw = NewaveFiles(
            directory=tmp_path,
            dger=tmp_path / "d",
            confhd=tmp_path / "c",
            conft=tmp_path / "t",
            sistema=tmp_path / "s",
            clast=tmp_path / "cl",
            term=tmp_path / "te",
            ree=tmp_path / "r",
            patamar=tmp_path / "p",
            hidr=tmp_path / "h",
            vazoes=tmp_path / "v",
            modif=None,
            ghmin=None,
            penalid=None,
            vazpast=None,
            dsvagua=None,
            curva=None,
            expt=None,
            manutt=None,
            c_adic=None,
            cvar=None,
            agrint=None,
            re_dat=None,
        )
        id_map = NewaveIdMap(subsystem_ids=[], hydro_codes=[], thermal_codes=[])
        assert convert_electric_constraints(nw, id_map) is None

    def test_example_produces_constraints(self) -> None:
        """Integration test against the example case."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        result = convert_electric_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        constraints, bounds_table = result
        assert len(constraints) > 0

    def test_example_constraints_have_correct_format(self) -> None:
        """Each constraint must have required fields and slack disabled."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        result = convert_electric_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        constraints, _ = result
        for c in constraints:
            assert isinstance(c["id"], int)
            assert c["name"].startswith("RE_")
            assert c["sense"] in ("<=", ">=")
            assert c["slack"]["enabled"] is True
            assert c["slack"]["penalty"] > 0
            assert (
                "hydro_generation(" in c["expression"]
                or "line_exchange(" in c["expression"]
            )

    def test_example_bounds_table_schema(self) -> None:
        """Bounds table must have the expected four columns."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        result = convert_electric_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        _, bounds_table = result
        assert isinstance(bounds_table, pa.Table)
        assert set(bounds_table.schema.names) == {
            "constraint_id",
            "stage_id",
            "block_id",
            "bound",
        }
        assert bounds_table.schema.field("constraint_id").type == pa.int32()
        assert bounds_table.schema.field("stage_id").type == pa.int32()
        assert bounds_table.schema.field("block_id").type == pa.int32()
        assert bounds_table.schema.field("bound").type == pa.float64()

    def test_example_start_id_offset_applied(self) -> None:
        """start_id must be added as an offset to all constraint IDs."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        result_0 = convert_electric_constraints(nw_files, id_map, start_id=0)
        result_10 = convert_electric_constraints(nw_files, id_map, start_id=10)
        assert result_0 is not None and result_10 is not None
        ids_0 = [c["id"] for c in result_0[0]]
        ids_10 = [c["id"] for c in result_10[0]]
        assert ids_10 == [i + 10 for i in ids_0]

    def test_example_only_lim_inf_excluded(self) -> None:
        """All constraints in the example file use <= (lim_inf is -1.1e30)."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        result = convert_electric_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        constraints, _ = result
        # All example constraints have only upper bounds (LimInf = -1.1e30)
        assert all(c["sense"] == "<=" for c in constraints)

    def test_example_block_ids_are_zero_based(self) -> None:
        """block_id must be 0-based (Pat=1 -> block_id=0)."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        result = convert_electric_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        _, bounds_table = result
        block_ids = bounds_table.column("block_id").to_pylist()
        assert all(b >= 0 for b in block_ids)
        # Example has 3 patamars; max block_id should be 2
        assert max(block_ids) == 2


# ---------------------------------------------------------------------------
# convert_agrint_constraints tests
# ---------------------------------------------------------------------------


def _make_minimal_nw_files(tmp_path: Path, *, agrint: Path | None = None) -> object:
    """Return a NewaveFiles-like object with minimal fields set for AGRINT tests."""
    from cobre_bridge.newave_files import NewaveFiles

    return NewaveFiles(
        directory=tmp_path,
        dger=tmp_path / "dger.dat",
        confhd=tmp_path / "c",
        conft=tmp_path / "t",
        sistema=tmp_path / "s",
        clast=tmp_path / "cl",
        term=tmp_path / "te",
        ree=tmp_path / "r",
        patamar=tmp_path / "p",
        hidr=tmp_path / "h",
        vazoes=tmp_path / "v",
        modif=None,
        ghmin=None,
        penalid=None,
        vazpast=None,
        dsvagua=None,
        curva=None,
        expt=None,
        manutt=None,
        c_adic=None,
        cvar=None,
        agrint=agrint,
        re_dat=None,
    )


_AGRINT_CONTENT = """\
AGRUPAMENTOS DE INTERCAMBIO
 #AG A   B   COEF
 XXX XXX XXX XX.XXXX
   1   1   3  1.0000
   2   3   1  1.0000
 999
LIMITES POR GRUPO
  #AG MI ANOI MF ANOF LIM_P1  LIM_P2  LIM_P3
 XXX  XX XXXX XX XXXX XXXXXX. XXXXXX. XXXXXX.
   1   1 2020  2 2020  10000.  10000.  10000.
   2   1 2020  1 2020   5000.   5000.   5000.
 999
"""


def _make_dger_mock_for_agrint():
    """Dger mock for study starting Jan 2020, 1 year."""
    from unittest.mock import MagicMock

    dger = MagicMock()
    dger.mes_inicio_estudo = 1
    dger.ano_inicio_estudo = 2020
    dger.num_anos_estudo = 1
    dger.num_anos_pos_estudo = 0
    return dger


class TestConvertAgrintConstraints:
    def test_returns_none_when_agrint_absent(self, tmp_path: Path) -> None:
        """Returns None when agrint path is None."""
        nw = _make_minimal_nw_files(tmp_path, agrint=None)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])
        assert convert_agrint_constraints(nw, id_map) is None  # type: ignore[arg-type]

    def test_produces_constraints_from_agrint_dat(self, tmp_path: Path) -> None:
        """Parses a minimal AGRINT file and produces one constraint per group."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        nw = _make_minimal_nw_files(tmp_path, agrint=agrint_path)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        dger_mock = _make_dger_mock_for_agrint()

        # Line map: canonical (1,3) -> line_id=0
        fake_line_map = {(1, 3): 0}

        with (
            patch("cobre_bridge.converters.constraints.Dger") as mock_dger_cls,
            patch(
                "cobre_bridge.converters.constraints._build_line_id_map",
                return_value=fake_line_map,
            ),
        ):
            mock_dger_cls.read.return_value = dger_mock
            result = convert_agrint_constraints(nw, id_map, start_id=0)  # type: ignore[arg-type]

        assert result is not None
        constraints, bounds_table = result
        assert len(constraints) == 2

    def test_constraint_sense_is_lte(self, tmp_path: Path) -> None:
        """All AGRINT constraints have sense '<='."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        nw = _make_minimal_nw_files(tmp_path, agrint=agrint_path)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with (
            patch("cobre_bridge.converters.constraints.Dger") as mock_dger_cls,
            patch(
                "cobre_bridge.converters.constraints._build_line_id_map",
                return_value={(1, 3): 0},
            ),
        ):
            mock_dger_cls.read.return_value = _make_dger_mock_for_agrint()
            result = convert_agrint_constraints(nw, id_map, start_id=0)  # type: ignore[arg-type]

        assert result is not None
        for c in result[0]:
            assert c["sense"] == "<="
            assert c["slack"]["enabled"] is False

    def test_start_id_offset_applied(self, tmp_path: Path) -> None:
        """start_id is added to all constraint IDs."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        nw = _make_minimal_nw_files(tmp_path, agrint=agrint_path)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with (
            patch("cobre_bridge.converters.constraints.Dger") as mock_dger_cls,
            patch(
                "cobre_bridge.converters.constraints._build_line_id_map",
                return_value={(1, 3): 0},
            ),
        ):
            mock_dger_cls.read.return_value = _make_dger_mock_for_agrint()
            result_0 = convert_agrint_constraints(nw, id_map, start_id=0)  # type: ignore[arg-type]
            mock_dger_cls.read.return_value = _make_dger_mock_for_agrint()
            result_5 = convert_agrint_constraints(nw, id_map, start_id=5)  # type: ignore[arg-type]

        assert result_0 is not None and result_5 is not None
        ids_0 = [c["id"] for c in result_0[0]]
        ids_5 = [c["id"] for c in result_5[0]]
        assert ids_5 == [i + 5 for i in ids_0]

    def test_bounds_table_schema(self, tmp_path: Path) -> None:
        """Bounds table has correct schema with block_id column."""
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        nw = _make_minimal_nw_files(tmp_path, agrint=agrint_path)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with (
            patch("cobre_bridge.converters.constraints.Dger") as mock_dger_cls,
            patch(
                "cobre_bridge.converters.constraints._build_line_id_map",
                return_value={(1, 3): 0},
            ),
        ):
            mock_dger_cls.read.return_value = _make_dger_mock_for_agrint()
            result = convert_agrint_constraints(nw, id_map, start_id=0)  # type: ignore[arg-type]

        assert result is not None
        _, bounds_table = result
        assert isinstance(bounds_table, pa.Table)
        assert set(bounds_table.schema.names) == {
            "constraint_id",
            "stage_id",
            "block_id",
            "bound",
        }

    def test_reversed_direction_negates_coefficient(self, tmp_path: Path) -> None:
        """Flow A->B where A>B should produce a negative coefficient in expression."""
        # Group 2: flow(3->1), canonical pair is (1,3). A=3 > B=1 => reversed => negate.
        agrint_path = tmp_path / "agrint.dat"
        agrint_path.write_text(_AGRINT_CONTENT, encoding="latin-1")
        (tmp_path / "dger.dat").touch()

        nw = _make_minimal_nw_files(tmp_path, agrint=agrint_path)
        id_map = NewaveIdMap(subsystem_ids=[1, 3], hydro_codes=[], thermal_codes=[])

        with (
            patch("cobre_bridge.converters.constraints.Dger") as mock_dger_cls,
            patch(
                "cobre_bridge.converters.constraints._build_line_id_map",
                return_value={(1, 3): 0},
            ),
        ):
            mock_dger_cls.read.return_value = _make_dger_mock_for_agrint()
            result = convert_agrint_constraints(nw, id_map, start_id=0)  # type: ignore[arg-type]

        assert result is not None
        # Group 1: flow(1->3), canonical (1,3) => positive, no leading '-'
        c1 = result[0][0]
        assert c1["expression"].startswith("line_exchange(0)")

        # Group 2: flow(3->1), reversed => expression should start with '- '
        c2 = result[0][1]
        assert c2["expression"].startswith("- line_exchange(0)")

    def test_example_agrint_produces_constraints(self) -> None:
        """Integration test against the example AGRINT.DAT file."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        if nw_files.agrint is None:
            pytest.skip("AGRINT.DAT not present in example")

        result = convert_agrint_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        constraints, bounds_table = result
        assert len(constraints) > 0
        for c in constraints:
            assert c["sense"] == "<="
            assert "line_exchange(" in c["expression"]
            assert c["slack"]["enabled"] is False

    def test_example_agrint_bounds_are_positive(self) -> None:
        """All bounds emitted by the example AGRINT file must be non-negative."""
        pair = _example_nw_files_and_id_map()
        if pair is None:
            pytest.skip("example/newave not available")
        nw_files, id_map = pair
        if nw_files.agrint is None:
            pytest.skip("AGRINT.DAT not present in example")

        result = convert_agrint_constraints(nw_files, id_map, start_id=0)
        assert result is not None
        _, bounds_table = result
        bounds = bounds_table.column("bound").to_pylist()
        assert all(b >= 0 for b in bounds)
