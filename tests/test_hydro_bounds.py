"""Unit tests for the source model hydro bounds converter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pytest

from cobre_bridge.id_map import NewaveIdMap
from tests.conftest import (
    _hydro_case,
    _make_cfuga_rec,
    _make_confhd_df,
    _make_dger_mock,
    _make_hydro_dger_mock,
    _make_ne_cadastro,
    _make_ne_confhd_df,
    _make_ne_exph_mock,
    _make_ree_df,
    _ne_filling_case,
    _ne_filling_id_map,
    make_case,
    make_nw_files,
)

# ---------------------------------------------------------------------------
# max_turbined_m3s envelope declaration  (ticket-015b)
# ---------------------------------------------------------------------------


def _head_corrected_two_plant_cadastro() -> pd.DataFrame:
    """Two-plant cadastro with the head-correction inputs the shared "simple"
    fixture (``_make_hidr_cadastro``) lacks — ``queda_nominal_conjunto_*`` —
    so ``_compute_max_turbined_head_corrected`` runs its head-corrected branch
    instead of falling back to the plain ``Σ n·q_nom`` sum.

    Both plants (USINA_A code 1, USINA_B code 2) share the same polynomial /
    machine-set / productivity numbers on a realistic ρ_esp scale (~0.009
    MW/(m³/s·m), not the toy 0.9 the shared fixture uses) so the installed-
    power cap (``cap_pinst``) stays a loose ceiling and the affinity-corrected
    flow term is what actually binds — matching the real deck behaviour the
    ticket-015 spike found. The only thing that varies between the two plants
    across this module's tests is whether a MODIF.DAT CFUGA record targets the
    plant, which is exactly the "with per-stage head variation" / "without"
    pairing acceptance criteria 5 and 6 need.
    """
    months = [
        "JAN",
        "FEV",
        "MAR",
        "ABR",
        "MAI",
        "JUN",
        "JUL",
        "AGO",
        "SET",
        "OUT",
        "NOV",
        "DEZ",
    ]
    base: dict[str, list] = {
        "nome_usina": ["USINA_A", "USINA_B"],
        "posto": [1, 2],
        "submercado": [1, 1],
        "empresa": [1, 1],
        "codigo_usina_jusante": [pd.NA, pd.NA],
        "desvio": [pd.NA, pd.NA],
        "volume_minimo": [100.0, 100.0],
        "volume_maximo": [1000.0, 1000.0],
        "volume_referencia": [550.0, 550.0],
        "canal_fuga_medio": [50.0, 50.0],
        "tipo_regulacao": ["M", "M"],
        "tipo_perda": [1, 1],
        "perdas": [0.0, 0.0],
        "a0_volume_cota": [300.0, 300.0],
        "a1_volume_cota": [0.1, 0.1],
        "a2_volume_cota": [0.0, 0.0],
        "a3_volume_cota": [0.0, 0.0],
        "a4_volume_cota": [0.0, 0.0],
        "produtibilidade_especifica": [0.009, 0.009],
        "numero_conjuntos_maquinas": [1, 1],
        "maquinas_conjunto_1": [1, 1],
        "maquinas_conjunto_2": [0, 0],
        "maquinas_conjunto_3": [0, 0],
        "maquinas_conjunto_4": [0, 0],
        "maquinas_conjunto_5": [0, 0],
        "potencia_nominal_conjunto_1": [1000.0, 1000.0],
        "potencia_nominal_conjunto_2": [0.0, 0.0],
        "potencia_nominal_conjunto_3": [0.0, 0.0],
        "potencia_nominal_conjunto_4": [0.0, 0.0],
        "potencia_nominal_conjunto_5": [0.0, 0.0],
        "vazao_nominal_conjunto_1": [100.0, 100.0],
        "vazao_nominal_conjunto_2": [0.0, 0.0],
        "vazao_nominal_conjunto_3": [0.0, 0.0],
        "vazao_nominal_conjunto_4": [0.0, 0.0],
        "vazao_nominal_conjunto_5": [0.0, 0.0],
        "queda_nominal_conjunto_1": [200.0, 200.0],
        "queda_nominal_conjunto_2": [0.0, 0.0],
        "queda_nominal_conjunto_3": [0.0, 0.0],
        "queda_nominal_conjunto_4": [0.0, 0.0],
        "queda_nominal_conjunto_5": [0.0, 0.0],
        "vazao_minima_historica": [0.0, 0.0],
        "teif": [0.0, 0.0],
        "ip": [0.0, 0.0],
        "fator_carga_maximo": [1.0, 1.0],
        "fator_carga_minimo": [0.0, 0.0],
    }
    for m in months:
        base[f"evaporacao_{m}"] = [0.0, 0.0]
    return pd.DataFrame(base, index=pd.Index([1, 2], name="codigo_usina"))


def _cfuga_modif_mock(code: int, *, month: int, year: int, nivel: float) -> MagicMock:
    """A MODIF.DAT mock carrying a single CFUGA record for *code* only."""
    usina_rec = MagicMock()
    usina_rec.codigo = code
    mock_modif = MagicMock()
    mock_modif.usina.return_value = [usina_rec]
    mock_modif.modificacoes_usina.return_value = [
        _make_cfuga_rec(month=month, year=year, nivel=nivel)
    ]
    return mock_modif


class TestConvertHydrosMaxTurbinedEnvelope:
    """ticket-015b: declared ``max_turbined_m3s`` must rise to cover every
    per-stage head-corrected cap ``convert_turbined_bounds_head_corrected``
    emits for that hydro (cobre rule 43), instead of staying pinned at the
    single reference-head value it used to declare unconditionally.

    ``130.0961183125769`` (offset from the reference ``120.26013470805694``)
    is the exact value ``_compute_max_turbined_head_corrected`` returns for
    USINA_A's fixture with a CFUGA-override head of 338.5 m — cross-checked
    directly against that function in
    ``test_per_stage_envelope_exceeds_reference_head_value`` rather than
    hand-derived, so a fixture-numbers change cannot silently desync the
    hard-coded expectations below it.
    """

    def _make_id_map(self) -> NewaveIdMap:
        return NewaveIdMap(subsystem_ids=[1], hydro_codes=[1, 2], thermal_codes=[])

    def _case_with_cfuga_on_usina_a(self, tmp_path: Path, *, nivel: float = 30.0):
        """USINA_A (code 1) carries a CFUGA override effective from stage 0
        (month=1 matches the horizon's start month/year), so every stage of
        the 12-stage horizon picks up the override uniformly. USINA_B
        (code 2) carries none.
        """
        modif = _cfuga_modif_mock(1, month=1, year=2025, nivel=nivel)
        return _hydro_case(
            tmp_path,
            cadastro=_head_corrected_two_plant_cadastro(),
            confhd=_make_confhd_df(),
            rees=_make_ree_df(),
            modif=modif,
            dger=_make_hydro_dger_mock(
                start_year=2025, start_month=1, num_anos=1, num_anos_pos=0
            ),
        )

    def test_per_stage_envelope_exceeds_reference_head_value(
        self, tmp_path: Path
    ) -> None:
        """Sanity check on the fixture itself: the per-stage head-corrected cap
        this CFUGA override produces genuinely exceeds the reference-head
        value — otherwise the rest of this class would be exercising a no-op."""
        from cobre_bridge.converters.hydro import (
            _compute_max_turbined_head_corrected,
            convert_turbined_bounds_head_corrected,
        )

        case = self._case_with_cfuga_on_usina_a(tmp_path)
        hreg_a = _head_corrected_two_plant_cadastro().loc[1]

        reference = _compute_max_turbined_head_corrected(hreg_a, "USINA_A")[0]
        assert reference == pytest.approx(120.26013470805694)

        table = convert_turbined_bounds_head_corrected(case, self._make_id_map())
        assert table is not None
        rows = [r for r in table.to_pylist() if r["hydro_id"] == 0]
        assert len(rows) == 12  # every stage of the 1-year horizon
        for row in rows:
            assert row["max_turbined_m3s"] == pytest.approx(130.0961183125769)
        assert rows[0]["max_turbined_m3s"] > reference

    def test_declared_value_raised_to_per_stage_envelope(self, tmp_path: Path) -> None:
        """AC #6: the declared value equals ``max(reference, per-stage max)``."""
        from cobre_bridge.converters.hydro import convert_hydros

        case = self._case_with_cfuga_on_usina_a(tmp_path)
        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")

        assert hydro_a["generation"]["max_turbined_m3s"] == pytest.approx(
            130.0961183125769
        )

    def test_mirror_group_matches_raised_declared_value(self, tmp_path: Path) -> None:
        """AC #3: the mirror unit group still carries the SAME (now-raised)
        value as ``generation.max_turbined_m3s`` — rule 41's mirror invariant
        holds at the envelope value, not just at the un-raised reference."""
        from cobre_bridge.converters.hydro import convert_hydros

        case = self._case_with_cfuga_on_usina_a(tmp_path)
        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")

        assert hydro_a["unit_groups"][0]["max_turbined_m3s"] == pytest.approx(
            hydro_a["generation"]["max_turbined_m3s"]
        )

    def test_plant_without_per_stage_variation_keeps_reference_value(
        self, tmp_path: Path
    ) -> None:
        """AC #5: USINA_B carries no CFUGA/CMONT/VOLREF_SAZ override, so it is
        absent from the envelope and keeps its un-raised reference-head value
        — the CFUGA override on its sibling USINA_A must not leak into it."""
        from cobre_bridge.converters.hydro import (
            _compute_max_turbined_head_corrected,
            convert_hydros,
        )

        case = self._case_with_cfuga_on_usina_a(tmp_path)
        hreg_b = _head_corrected_two_plant_cadastro().loc[2]
        expected_b = _compute_max_turbined_head_corrected(hreg_b, "USINA_B")[0]

        result = convert_hydros(case, self._make_id_map())
        hydro_b = next(h for h in result["hydros"] if h["name"] == "USINA_B")

        assert hydro_b["generation"]["max_turbined_m3s"] == pytest.approx(expected_b)
        assert hydro_b["unit_groups"][0]["max_turbined_m3s"] == pytest.approx(
            expected_b
        )

    def test_no_overrides_anywhere_keeps_declared_value_unchanged(
        self, tmp_path: Path
    ) -> None:
        """A hydro case with no MODIF/VOLREF_SAZ at all (the common case) must
        declare exactly the reference-head value — the envelope lookup is a
        pure no-op when ``convert_turbined_bounds_head_corrected`` returns
        ``None``."""
        from cobre_bridge.converters.hydro import (
            _compute_max_turbined_head_corrected,
            convert_hydros,
        )

        case = _hydro_case(
            tmp_path,
            cadastro=_head_corrected_two_plant_cadastro(),
            confhd=_make_confhd_df(),
            rees=_make_ree_df(),
            dger=_make_hydro_dger_mock(
                start_year=2025, start_month=1, num_anos=1, num_anos_pos=0
            ),
        )
        hreg_a = _head_corrected_two_plant_cadastro().loc[1]
        expected_a = _compute_max_turbined_head_corrected(hreg_a, "USINA_A")[0]

        result = convert_hydros(case, self._make_id_map())
        hydro_a = next(h for h in result["hydros"] if h["name"] == "USINA_A")
        assert hydro_a["generation"]["max_turbined_m3s"] == pytest.approx(expected_a)

    def test_per_stage_hydro_bounds_rows_unaffected_by_the_declaration_fix(
        self, tmp_path: Path
    ) -> None:
        """AC #4 (LP-neutrality): the per-stage rows
        ``convert_turbined_bounds_head_corrected`` emits for ``hydro_bounds.parquet``
        are IDENTICAL whether or not ``convert_hydros``'s envelope-raising
        declaration fix (ticket-015b) already ran against an equivalent case —
        the fix only ever raises the entity-level declaration, never the
        per-stage table. Two independently-built (but data-identical) cases
        isolate this: one has ``convert_hydros`` run against it FIRST, the
        other never sees ``convert_hydros`` at all, and the per-stage tables
        must still match row for row."""
        from cobre_bridge.converters.hydro import (
            convert_hydros,
            convert_turbined_bounds_head_corrected,
        )

        id_map = self._make_id_map()

        case_without_declaration_fix = self._case_with_cfuga_on_usina_a(tmp_path)
        rows_without_declaration_fix = [
            r
            for r in convert_turbined_bounds_head_corrected(
                case_without_declaration_fix, id_map
            ).to_pylist()
            if r["hydro_id"] == 0
        ]

        case_with_declaration_fix = self._case_with_cfuga_on_usina_a(tmp_path)
        convert_hydros(case_with_declaration_fix, id_map)  # exercises the raise
        rows_with_declaration_fix = [
            r
            for r in convert_turbined_bounds_head_corrected(
                case_with_declaration_fix, id_map
            ).to_pylist()
            if r["hydro_id"] == 0
        ]

        assert len(rows_without_declaration_fix) == len(rows_with_declaration_fix) == 12
        assert rows_without_declaration_fix == rows_with_declaration_fix
        assert all(
            r["max_turbined_m3s"] == pytest.approx(130.0961183125769)
            for r in rows_with_declaration_fix
        )


class TestPerStageTurbinedEnvelopeHelper:
    """Unit tests for ``_per_stage_turbined_envelope`` in isolation, via a
    mocked ``convert_turbined_bounds_head_corrected`` — the real head physics
    are covered end-to-end by ``TestConvertHydrosMaxTurbinedEnvelope``; this
    class only checks the max-per-hydro-id aggregation contract."""

    def test_empty_dict_when_table_is_none(self) -> None:
        from cobre_bridge.converters.hydro import _per_stage_turbined_envelope

        with patch(
            "cobre_bridge.converters.hydro.bounds.convert_turbined_bounds_head_corrected",
            return_value=None,
        ):
            envelope = _per_stage_turbined_envelope(MagicMock(), MagicMock())

        assert envelope == {}

    def test_max_per_hydro_id_across_stages(self) -> None:
        from cobre_bridge.converters.hydro import _per_stage_turbined_envelope

        table = pa.table(
            {
                "hydro_id": pa.array([0, 0, 0, 1, 1], type=pa.int32()),
                "stage_id": pa.array([0, 1, 2, 0, 1], type=pa.int32()),
                "max_turbined_m3s": pa.array(
                    [100.0, 150.0, 120.0, 30.0, 45.0], type=pa.float64()
                ),
            }
        )
        with patch(
            "cobre_bridge.converters.hydro.bounds.convert_turbined_bounds_head_corrected",
            return_value=table,
        ):
            envelope = _per_stage_turbined_envelope(MagicMock(), MagicMock())

        assert envelope == {0: 150.0, 1: 45.0}


class TestConvertStorageBoundsPostStudy:
    """Per-quantity post-study extrapolation in convert_storage_bounds.

    VMINT/VMAXT repeat the last study year's seasonal pattern only when their
    dger ``sazonaliza_*`` flag is set; outflow (VAZMINT) and turbined
    (TURBMINT/TURBMAXT) have no flag and freeze the last study stage value.
    """

    def _run(self, tmp_path, overrides, *, vmaxt_flag=1, vmint_flag=1):
        from cobre_bridge.converters.hydro import convert_storage_bounds

        # start_month=1, 1 study year → study_months=12 (Jan–Dec); 1 post-study
        # year → stages 12–23 (Jan–Dec again).
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2024
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 1
        mock_dger.num_anos_pos_estudo = 1
        mock_dger.sazonaliza_vmaxt = vmaxt_flag
        mock_dger.sazonaliza_vmint = vmint_flag

        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [10],
                "usina_existente": ["EX"],
                "nome_usina": ["TEST"],
            }
        )
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        cadastro = pd.DataFrame(
            {"volume_minimo": [0.0], "volume_maximo": [100.0]}, index=[10]
        )
        id_map = MagicMock()
        id_map.hydro_id = lambda c: 0

        case = make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            dger=mock_dger,
            confhd=mock_confhd,
        )
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=cadastro,
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._extract_temporal_overrides",
                return_value={10: overrides},
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={},
            ),
        ):
            tbl = convert_storage_bounds(case, id_map)
        assert tbl is not None
        return tbl.to_pandas().set_index("stage_id")

    def test_outflow_freezes_post_study(self, tmp_path) -> None:
        """VAZMINT (no flag) freezes the post-study tail at last study Dec."""
        overrides = [
            {"type": "VAZMINT", "year": 2024, "month": 1, "value": 10.0},
            {"type": "VAZMINT", "year": 2024, "month": 12, "value": 120.0},
        ]
        df = self._run(tmp_path, overrides)
        # Study: Jan–Nov step-carry 10, Dec=120.
        assert df.loc[0, "min_outflow_m3s"] == pytest.approx(10.0)
        assert df.loc[11, "min_outflow_m3s"] == pytest.approx(120.0)
        # Post-study (12–23): all frozen at Dec=120, NOT the seasonal Jan=10.
        for s in range(12, 24):
            assert df.loc[s, "min_outflow_m3s"] == pytest.approx(120.0)

    def test_turbined_min_freezes_post_study(self, tmp_path) -> None:
        """TURBMINT (no flag) freezes the post-study tail."""
        overrides = [
            {"type": "TURBMINT", "year": 2024, "month": 1, "value": 5.0},
            {"type": "TURBMINT", "year": 2024, "month": 12, "value": 50.0},
        ]
        df = self._run(tmp_path, overrides)
        assert df.loc[11, "min_turbined_m3s"] == pytest.approx(50.0)
        for s in range(12, 24):
            assert df.loc[s, "min_turbined_m3s"] == pytest.approx(50.0)

    def test_vmaxt_seasonalizes_when_flag_set(self, tmp_path) -> None:
        """VMAXT with sazonaliza_vmaxt=1 repeats the seasonal pattern."""
        overrides = [
            {"type": "VMAXT", "year": 2024, "month": 1, "value": 50.0},
            {"type": "VMAXT", "year": 2024, "month": 12, "value": 80.0},
        ]
        df = self._run(tmp_path, overrides, vmaxt_flag=1)
        # vol_min=0, useful=100 → pct == hm3. Study Jan=50, Dec=80.
        assert df.loc[0, "max_storage_hm3"] == pytest.approx(50.0)
        assert df.loc[11, "max_storage_hm3"] == pytest.approx(80.0)
        # Post-study seasonal: stage 12 (Jan) keeps 50, stage 23 (Dec) keeps 80.
        assert df.loc[12, "max_storage_hm3"] == pytest.approx(50.0)
        assert df.loc[23, "max_storage_hm3"] == pytest.approx(80.0)

    def test_vmaxt_freezes_when_flag_clear(self, tmp_path) -> None:
        """VMAXT with sazonaliza_vmaxt=0 freezes the post-study tail."""
        overrides = [
            {"type": "VMAXT", "year": 2024, "month": 1, "value": 50.0},
            {"type": "VMAXT", "year": 2024, "month": 12, "value": 80.0},
        ]
        df = self._run(tmp_path, overrides, vmaxt_flag=0)
        # Post-study frozen at Dec=80, NOT seasonal Jan=50.
        assert df.loc[12, "max_storage_hm3"] == pytest.approx(80.0)
        assert df.loc[23, "max_storage_hm3"] == pytest.approx(80.0)


class TestConvertStorageBoundsMaxGenColumn:
    """ticket-010: the ``max_generation_mw`` column is gated on filling plants.

    EX-only cases (no ``NE``-with-filling plant) keep the existing 8-column
    schema byte-identical; a case with a filling plant gains a 9th
    ``max_generation_mw`` float64 column that is all-null until ticket-011
    populates the per-stage ramp caps.
    """

    _EXPECTED_8_COLUMNS = [
        "hydro_id",
        "stage_id",
        "min_storage_hm3",
        "max_storage_hm3",
        "min_turbined_m3s",
        "max_turbined_m3s",
        "min_outflow_m3s",
        "min_generation_mw",
    ]

    def _run_ex_only(self, tmp_path):
        """An EX-only case (no exph) with one GHMIN row, mirroring ``_run``."""
        from cobre_bridge.converters.hydro import convert_storage_bounds

        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2024
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 1
        mock_dger.num_anos_pos_estudo = 0
        mock_dger.sazonaliza_vmaxt = 1
        mock_dger.sazonaliza_vmint = 1

        confhd_df = pd.DataFrame(
            {
                "codigo_usina": [10],
                "usina_existente": ["EX"],
                "nome_usina": ["TEST"],
            }
        )
        mock_confhd = MagicMock()
        mock_confhd.usinas = confhd_df
        cadastro = pd.DataFrame(
            {"volume_minimo": [0.0], "volume_maximo": [100.0]}, index=[10]
        )
        id_map = MagicMock()
        id_map.hydro_id = lambda c: 0

        case = make_case(
            make_nw_files(tmp_path),
            dger=mock_dger,
            confhd=mock_confhd,
            exph=None,
        )
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=cadastro,
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={10: {0: 100.0}},
            ),
        ):
            return convert_storage_bounds(case, id_map)

    def _run_filling(self, tmp_path):
        """A filling case (JURUENA via exph) with one GHMIN row on an EX plant.

        ``convert_storage_bounds`` needs at least one per-stage override/GHMIN
        row to emit any table, so a synthetic GHMIN entry for EX plant 1 is
        supplied. The non-empty ``filling_codes`` (309) is what gates the
        column on.
        """
        from cobre_bridge.converters.hydro import convert_storage_bounds

        case = _ne_filling_case(tmp_path)
        id_map = _ne_filling_id_map()
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=_make_ne_cadastro(),
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={1: {0: 100.0}},
            ),
        ):
            return convert_storage_bounds(case, id_map)

    def test_ex_only_case_omits_max_generation_column(self, tmp_path) -> None:
        """EX-only case keeps exactly the 8 existing columns, no max_generation."""
        tbl = self._run_ex_only(tmp_path)
        assert tbl is not None
        assert tbl.column_names == self._EXPECTED_8_COLUMNS
        assert "max_generation_mw" not in tbl.column_names

    def test_filling_case_includes_max_generation_column(self, tmp_path) -> None:
        """A filling plant adds a float64 max_generation_mw column after min_gen."""
        tbl = self._run_filling(tmp_path)
        assert tbl is not None
        assert "max_generation_mw" in tbl.column_names
        assert tbl.schema.field("max_generation_mw").type == pa.float64()
        # Positioned immediately after min_generation_mw.
        names = tbl.column_names
        assert names[names.index("min_generation_mw") + 1] == "max_generation_mw"

    def test_max_generation_populated_for_filling_ramp_rows(self, tmp_path) -> None:
        """ticket-011 populates the column for JURUENA's ramp rows.

        ticket-010 introduced the column all-null; ticket-011's ramp branch now
        emits explicit ``0.0`` caps for JURUENA's pre-operating stages (0–3) while
        the non-ramp GHMIN row (EX plant 1) keeps ``max_generation_mw`` null. So
        the column is no longer all-null: exactly the GHMIN row is null and the
        JURUENA ramp rows carry ``0.0``.
        """
        tbl = self._run_filling(tmp_path)
        assert tbl is not None
        assert tbl.num_rows > 0
        col = tbl.column("max_generation_mw")
        # Only the single non-ramp GHMIN row is null; the rest are populated.
        assert col.null_count == 1
        assert col.null_count < tbl.num_rows


class TestConvertStorageBoundsRamp:
    """ticket-011: the filling-plant unit-ramp branch in ``convert_storage_bounds``.

    A ``NE``-with-filling plant (JURUENA, code 309) operates from its
    ``entry_stage_id`` but its turbine / generation capacity is whatever
    generating units are online. The branch exports EXPLICIT 0/reduced caps over
    the full pre-operating window ``[0, full_online_sid)`` (clamped to the
    horizon) so the parquet carries the true 0→full profile for plottability;
    this matches cobre's internal PreFilling/Filling forcing and leaves the
    simulation unchanged. Under the Sep-2024 3-year horizon JURUENA's filling
    completes at ``entry_sid == 2`` and both units enter Jan 2025 (stage 4), so
    stages 0–3 have zero online machines (explicit ``0.0`` caps) and there is no
    override row at stage 4+ (the base ``hydros.json`` caps apply).
    """

    def _run_juruena(self, tmp_path, *, duracao: int = 1):
        """Run ``convert_storage_bounds`` for the JURUENA filling case.

        ``_read_ghmin_per_stage`` is patched to ``{}`` so the table contains only
        the ramp rows (no GHMIN/MODIF rows to filter past).
        """
        from cobre_bridge.converters.hydro import convert_storage_bounds

        case = _ne_filling_case(tmp_path, duracao=duracao)
        id_map = _ne_filling_id_map()
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=_make_ne_cadastro(),
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={},
            ),
        ):
            tbl = convert_storage_bounds(case, id_map)
        return tbl, id_map.hydro_id(309)

    def test_juruena_emits_zero_cap_ramp_rows_through_filling_and_idle(
        self, tmp_path
    ) -> None:
        """JURUENA emits 0-cap rows for the whole pre-operating window (stages 0–3).

        The full pre-operating window ``[0, full_online_sid)`` covers PreFilling/
        Filling (stages 0, 1) and the operating-but-idle ramp (stages 2, 3); every
        row has all caps ``0.0`` (no units online before stage 4).
        """
        tbl, juruena_id = self._run_juruena(tmp_path)
        assert tbl is not None
        df = tbl.to_pandas()
        jur = df[df["hydro_id"] == juruena_id].sort_values("stage_id")
        assert list(jur["stage_id"]) == [0, 1, 2, 3]
        for col in ("max_turbined_m3s", "max_generation_mw", "min_generation_mw"):
            assert list(jur[col]) == [0.0, 0.0, 0.0, 0.0], col

    def test_juruena_no_ramp_row_at_full_online_stage(self, tmp_path) -> None:
        """No JURUENA ramp row at stage 4+ (both units online ⇒ base caps apply)."""
        tbl, juruena_id = self._run_juruena(tmp_path)
        assert tbl is not None
        df = tbl.to_pandas()
        jur = df[df["hydro_id"] == juruena_id]
        assert not (jur["stage_id"] >= 4).any()

    def test_juruena_ramp_rows_have_null_storage_and_outflow(self, tmp_path) -> None:
        """JURUENA ramp rows (stages 0–3) leave storage/turbined/outflow null."""
        tbl, juruena_id = self._run_juruena(tmp_path)
        assert tbl is not None
        df = tbl.to_pandas()
        jur = df[df["hydro_id"] == juruena_id]
        for col in (
            "min_storage_hm3",
            "max_storage_hm3",
            "min_turbined_m3s",
            "min_outflow_m3s",
        ):
            assert jur[col].isna().all(), col

    def test_filling_past_horizon_emits_zero_caps_all_in_study_stages(
        self, tmp_path
    ) -> None:
        """A plant entering at/after the horizon emits 0 caps for ALL in-study stages.

        Mirrors ``test_ne_plant_filling_past_horizon_no_crash``: a 3-stage horizon
        (Oct 2024 start) with ``duracao=6`` pushes ``entry_sid == 6 > total_stages
        (3)`` — the plant never operates in-study, so the full pre-operating window
        ``[0, full_online_sid)`` clamps to ``[0, total_stages)`` and every in-study
        stage (0, 1, 2) gets an explicit ``0.0``-cap row. No row is emitted at/past
        the horizon (no stage_id ≥ 3) and the clamp prevents the epic-03
        IndexError.
        """
        from cobre_bridge.converters.hydro import convert_storage_bounds

        case = _hydro_case(
            tmp_path,
            cadastro=_make_ne_cadastro(),
            confhd=_make_ne_confhd_df(),
            dger=_make_hydro_dger_mock(
                start_year=2024, start_month=10, num_anos=1, num_anos_pos=0
            ),
            exph=_make_ne_exph_mock(duracao=6, volume_morto=0.0),
        )
        id_map = _ne_filling_id_map()
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=_make_ne_cadastro(),
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={},
            ),
        ):
            # Must not raise the epic-03 IndexError; the clamp keeps every emitted
            # stage in-study.
            tbl = convert_storage_bounds(case, id_map)
        assert tbl is not None
        df = tbl.to_pandas()
        jur = df[df["hydro_id"] == id_map.hydro_id(309)].sort_values("stage_id")
        # total_stages == 3 ⇒ explicit 0-cap rows for stages 0, 1, 2; none ≥ 3.
        assert list(jur["stage_id"]) == [0, 1, 2]
        for col in ("max_turbined_m3s", "max_generation_mw", "min_generation_mw"):
            assert list(jur[col]) == [0.0, 0.0, 0.0], col
        assert not (df["stage_id"] >= 3).any()

    def test_reduced_caps_partial_online(self) -> None:
        """``_reduced_caps`` rises monotonically as machine groups come online.

        A 2-conjunto cadastro with the head columns absent forces the simple
        ``Σ n·p_nom`` / ``Σ n·q_nom`` fallback (no head correction, no
        availability derating with ``teif == ip == 0``), giving hand-computable
        caps: conjunto 1 = 2 units × (50 m³/s, 100 MW); conjunto 2 = 3 units ×
        (80 m³/s, 200 MW).
        """
        from cobre_bridge.converters.hydro import _reduced_caps

        hreg = pd.Series(
            {
                "numero_conjuntos_maquinas": 2,
                "maquinas_conjunto_1": 2,
                "maquinas_conjunto_2": 3,
                "potencia_nominal_conjunto_1": 100.0,
                "potencia_nominal_conjunto_2": 200.0,
                "vazao_nominal_conjunto_1": 50.0,
                "vazao_nominal_conjunto_2": 80.0,
                "teif": 0.0,
                "ip": 0.0,
            }
        )
        # No machines online ⇒ both caps are an explicit zero.
        assert _reduced_caps(hreg, {}, "X") == (0.0, 0.0)
        # Only conjunto 1 online: 2×50 = 100 m³/s, 2×100 = 200 MW.
        assert _reduced_caps(hreg, {1: 2}, "X") == (100.0, 200.0)
        # Both groups online: 100 + 3×80 = 340 m³/s, 200 + 3×200 = 800 MW.
        assert _reduced_caps(hreg, {1: 2, 2: 3}, "X") == (340.0, 800.0)
        # Generation cap rises monotonically as units come online.
        mgs = [
            _reduced_caps(hreg, online, "X")[1] for online in ({}, {1: 2}, {1: 2, 2: 3})
        ]
        assert mgs == sorted(mgs)
        assert mgs[0] < mgs[1] < mgs[2]

    def test_ramp_branch_skips_unit_row_with_missing_date(self, tmp_path) -> None:
        """A unit row with a conjunto but a BLANK online date is skipped, not crashed.

        inewave parses ``conjunto_maquina_entrada`` and ``data_entrada_operacao``
        independently, so a malformed exph can carry a unit row that has a machine
        group set but a ``NaT`` online date. Filtering on the date (the defining
        field of a unit row) and skipping a ``NaN`` conjunto means that row never
        reaches ``filling_stage_id`` with a ``NaT`` (which would yield ``nan`` and
        raise ``TypeError`` in ``range(...)``). The remaining valid unit row still
        drives ``full_online_sid``, so JURUENA still emits its ramp rows.
        """
        from cobre_bridge.converters.hydro import convert_storage_bounds

        # JURUENA exph: schedule row + ONE malformed unit row (conjunto 1 set,
        # date NaT) ORDERED BEFORE the valid unit row (conjunto 1, Jan-2025 →
        # stage 4). The malformed row is placed first deliberately: the old
        # conjunto-based filter kept it, and its ``NaT`` → ``nan`` stage made the
        # running ``max(...)`` for ``full_online_sid`` collapse to ``nan`` (nan
        # comparisons are always False, so a leading nan is never displaced),
        # crashing ``range(...)`` with a TypeError. The date filter + NaN-conjunto
        # guard drop it so only the valid stage-4 row survives.
        expansoes = pd.DataFrame(
            {
                "codigo_usina": [309, 309, 309],
                "nome_usina": ["JURUENA", "JURUENA", "JURUENA"],
                "data_inicio_enchimento": [
                    pd.Timestamp("2024-10-01"),
                    pd.NaT,
                    pd.NaT,
                ],
                "duracao_enchimento": [1, 0, 0],
                "volume_morto": [0.0, 0.0, 0.0],
                "data_entrada_operacao": [
                    pd.Timestamp("2024-11-01"),
                    pd.NaT,  # malformed: conjunto set but no online date
                    pd.Timestamp("2025-01-01"),
                ],
                "conjunto_maquina_entrada": [pd.NA, 1, 1],
                "maquina_entrada": [pd.NA, 2, 1],
            }
        )
        exph = MagicMock()
        exph.expansoes = expansoes
        case = _hydro_case(
            tmp_path,
            cadastro=_make_ne_cadastro(),
            confhd=_make_ne_confhd_df(),
            dger=_make_hydro_dger_mock(
                start_year=2024, start_month=9, num_anos=3, num_anos_pos=0
            ),
            exph=exph,
        )
        id_map = _ne_filling_id_map()
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=_make_ne_cadastro(),
            ),
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={},
            ),
        ):
            # Must not raise TypeError from a NaT-derived nan range bound.
            tbl = convert_storage_bounds(case, id_map)
        assert tbl is not None
        df = tbl.to_pandas()
        jur = df[df["hydro_id"] == id_map.hydro_id(309)].sort_values("stage_id")
        # The single valid unit row (stage 4) still drives full_online_sid, so the
        # pre-operating window is [0, 4) → stages 0, 1, 2, 3, exactly as the
        # all-valid mock.
        assert list(jur["stage_id"]) == [0, 1, 2, 3]

    def test_ramp_row_wins_over_modif_at_same_stage(self, tmp_path) -> None:
        """A ramp row wins over a colliding GHMIN row at the same (hydro, stage).

        JURUENA's ramp window is stages 2 and 3. Injecting a GHMIN override for
        JURUENA (309) at stage 2 creates a duplicate ``(hydro_id, stage_id)``
        pair. Per the ramp-wins rule the ramp row is kept and the GHMIN row is
        dropped: exactly one row remains at that key, with all caps ``0.0`` and a
        null ``min_generation`` ramp signature — the GHMIN minimum is gone.
        """
        from cobre_bridge.converters.hydro import convert_storage_bounds

        case = _ne_filling_case(tmp_path)
        id_map = _ne_filling_id_map()
        juruena_id = id_map.hydro_id(309)
        with (
            patch(
                "cobre_bridge.converters.hydro.bounds.read_cadastro",
                return_value=_make_ne_cadastro(),
            ),
            # GHMIN for JURUENA (309) at stage 2 — inside the ramp window [2, 4).
            patch(
                "cobre_bridge.converters.hydro.bounds._read_ghmin_per_stage",
                return_value={309: {2: 123.0}},
            ),
        ):
            tbl = convert_storage_bounds(case, id_map)
        assert tbl is not None
        df = tbl.to_pandas()
        at_stage_2 = df[(df["hydro_id"] == juruena_id) & (df["stage_id"] == 2)]
        # Exactly one row survives at the colliding key.
        assert len(at_stage_2) == 1
        row = at_stage_2.iloc[0]
        # It is the ramp row: zero caps, not the GHMIN minimum of 123.0.
        assert row["max_turbined_m3s"] == 0.0
        assert row["max_generation_mw"] == 0.0
        assert row["min_generation_mw"] == 0.0
        # Stage 3 (no GHMIN collision) keeps its ramp row too.
        at_stage_3 = df[(df["hydro_id"] == juruena_id) & (df["stage_id"] == 3)]
        assert len(at_stage_3) == 1
        assert at_stage_3.iloc[0]["max_generation_mw"] == 0.0


class TestMergeHydroBoundsMaxGenColumn:
    """ticket-010 C4: ``_merge_hydro_bounds`` carries ``max_generation_mw``.

    The storage-side bounds table may include a ``max_generation_mw`` column
    (added for filling plants). The polars ``how="full"`` join in
    ``_merge_hydro_bounds`` must preserve that storage-only column in the
    merged result, since it has no counterpart on the withdrawal side.
    """

    def test_storage_side_max_generation_column_survives_full_join(self) -> None:
        """A storage-only max_generation_mw column appears in the merged table."""
        from cobre_bridge.pipeline import _merge_hydro_bounds

        withdrawal = pa.table(
            {
                "hydro_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "water_withdrawal_m3s": pa.array([1.5, 2.5], type=pa.float64()),
            }
        )
        storage = pa.table(
            {
                "hydro_id": pa.array([0, 1], type=pa.int32()),
                "stage_id": pa.array([0, 0], type=pa.int32()),
                "min_generation_mw": pa.array([10.0, 20.0], type=pa.float64()),
                "max_generation_mw": pa.array([100.0, None], type=pa.float64()),
            }
        )

        result = _merge_hydro_bounds(withdrawal, storage)

        assert result is not None
        assert "max_generation_mw" in result.column_names


# ---------------------------------------------------------------------------
# Helper builders for water-withdrawal tests.
# ---------------------------------------------------------------------------


def _make_dsvagua_df(rows: list[dict]) -> pd.DataFrame:
    """Build a synthetic dsvagua desvios DataFrame from explicit rows."""
    return pd.DataFrame(rows)


class TestWaterWithdrawalConversion:
    """Unit tests for ``convert_water_withdrawal`` in ``hydro.py``."""

    def _make_id_map(self) -> NewaveIdMap:
        """Two hydros: The source model codes 10 and 20 -> Cobre IDs 0 and 1."""
        return NewaveIdMap(
            subsystem_ids=[1],
            hydro_codes=[10, 20],
            thermal_codes=[],
        )

    def _withdrawal_case(self, tmp_path, *, dsvagua, dger, confhd=None):
        """Build a case for ``convert_water_withdrawal`` with dsvagua/dger/confhd
        mocks pre-cached (dsvagua path set so the present-file guard passes)."""
        parsed: dict = {"dger": dger, "dsvagua": dsvagua}
        if confhd is not None:
            parsed["confhd"] = confhd
        return make_case(
            make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"), **parsed
        )

    def test_basic_returns_correct_schema(self, tmp_path: Path) -> None:
        """Two plants, three dates each: table has the three expected columns."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        rows = [
            {
                "codigo_usina": 10,
                "data": datetime.datetime(2020, 1, 1),
                "valor": -2.0,
            },
            {
                "codigo_usina": 10,
                "data": datetime.datetime(2020, 2, 1),
                "valor": -3.0,
            },
            {
                "codigo_usina": 20,
                "data": datetime.datetime(2020, 1, 1),
                "valor": -1.0,
            },
        ]
        dger_mock = _make_dger_mock(2020, 1, 5)

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = dger_mock.ano_inicio_estudo
        mock_dger.mes_inicio_estudo = dger_mock.mes_inicio_estudo
        mock_dger.num_anos_estudo = dger_mock.num_anos_estudo

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, self._make_id_map())

        assert result is not None
        assert result.schema.names == ["hydro_id", "stage_id", "water_withdrawal_m3s"]
        assert result.schema.field("hydro_id").type == pa.int32()
        assert result.schema.field("stage_id").type == pa.int32()
        assert result.schema.field("water_withdrawal_m3s").type == pa.float64()

    def test_sign_negation_and_stage_mapping(self, tmp_path: Path) -> None:
        """valor=-5.0 at 2020-02 -> water_withdrawal_m3s=5.0, stage_id=1."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 10, "data": datetime.datetime(2020, 2, 1), "valor": -5.0}
        ]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5
        mock_dger.num_anos_pos_estudo = 0

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, id_map)

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        assert row["hydro_id"][0] == id_map.hydro_id(10)
        assert row["stage_id"][0] == 1
        assert row["water_withdrawal_m3s"][0] == pytest.approx(5.0)

    def test_groupby_sum_same_plant_same_date(self, tmp_path: Path) -> None:
        """Two rows with the same plant/date are summed then negated."""
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 10, "data": datetime.datetime(2020, 1, 1), "valor": -3.0},
            {"codigo_usina": 10, "data": datetime.datetime(2020, 1, 1), "valor": -7.0},
        ]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5
        mock_dger.num_anos_pos_estudo = 0

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, id_map)

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        # -3.0 + -7.0 = -10.0; negated -> 10.0
        assert row["water_withdrawal_m3s"][0] == pytest.approx(10.0)
        assert row["stage_id"][0] == 0

    def test_missing_dsvagua_file_returns_none(self, tmp_path: Path) -> None:
        """When dsvagua.dat is absent the converter returns None without error."""
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        # dsvagua absent (path None) — the converter must return None before
        # touching dger.
        case = make_case(make_nw_files(tmp_path, dsvagua=None))
        result = convert_water_withdrawal(case, self._make_id_map())
        assert result is None

    def test_empty_desvios_returns_none(self, tmp_path: Path) -> None:
        """When desvios is None the converter returns None."""
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = None
        mock_dger = MagicMock()
        mock_dger.outros_usos_da_agua = 1
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        case = self._withdrawal_case(tmp_path, dsvagua=mock_dsvagua, dger=mock_dger)
        result = convert_water_withdrawal(case, self._make_id_map())

        assert result is None

    def test_dger_outros_usos_da_agua_zero_skips_dsvagua(self, tmp_path: Path) -> None:
        """``dger.outros_usos_da_agua == 0`` short-circuits the conversion.

        Mirrors the source model's own behaviour — when the dger switch is 0 the solver
        ignores ``dsvagua.dat`` regardless of its content, so the converter must not
        emit any water-withdrawal rows.
        """
        from cobre_bridge.converters.hydro import convert_water_withdrawal

        mock_dger = MagicMock()
        mock_dger.outros_usos_da_agua = 0
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5

        # Do NOT pre-cache the dsvagua slot: the dger switch must short-circuit
        # before ``case.dsvagua`` is ever accessed. If the short-circuit
        # regressed, the cached_property would attempt to parse the (absent)
        # dsvagua.dat path and raise, failing this test loudly.
        case = make_case(
            make_nw_files(tmp_path, dsvagua=tmp_path / "dsvagua.dat"),
            dger=mock_dger,
        )
        result = convert_water_withdrawal(case, self._make_id_map())

        assert result is None
        # The short-circuit must happen before any dsvagua access.
        assert "dsvagua" not in case.__dict__

    def test_codes_outside_id_map_are_dropped(self, tmp_path: Path) -> None:
        """``codigo_usina`` codes the id_map doesn't know are silently dropped.

        ``dsvagua.dat`` frequently carries codes for non-dispatchable
        plants (fictitious nodes, RHEs, etc.) that are filtered out of
        the id_map; logging a warning for each would be noisy.
        """
        import datetime

        from cobre_bridge.converters.hydro import convert_water_withdrawal

        (tmp_path / "dsvagua.dat").touch()
        (tmp_path / "dger.dat").touch()

        rows = [
            {"codigo_usina": 10, "data": datetime.datetime(2020, 1, 1), "valor": -4.0},
            {"codigo_usina": 99, "data": datetime.datetime(2020, 1, 1), "valor": -2.0},
        ]
        id_map = NewaveIdMap(subsystem_ids=[1], hydro_codes=[10], thermal_codes=[])

        mock_dsvagua = MagicMock()
        mock_dsvagua.desvios = _make_dsvagua_df(rows)
        mock_dger = MagicMock()
        mock_dger.ano_inicio_estudo = 2020
        mock_dger.mes_inicio_estudo = 1
        mock_dger.num_anos_estudo = 5
        mock_dger.num_anos_pos_estudo = 0

        mock_confhd = MagicMock()
        mock_confhd.usinas = pd.DataFrame(
            columns=["codigo_usina", "codigo_usina_jusante", "nome_usina"]
        )
        case = self._withdrawal_case(
            tmp_path, dsvagua=mock_dsvagua, dger=mock_dger, confhd=mock_confhd
        )
        result = convert_water_withdrawal(case, id_map)

        assert result is not None
        assert result.num_rows == 1
        row = result.to_pydict()
        assert row["water_withdrawal_m3s"][0] == pytest.approx(4.0)
