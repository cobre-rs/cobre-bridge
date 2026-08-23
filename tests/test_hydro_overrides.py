"""Unit tests for the source model hydro overrides converter."""

from __future__ import annotations

import logging
import math
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tests.conftest import (
    _make_hidr_cadastro,
    _make_penalid_df,
    make_case,
    make_nw_files,
)

# ---------------------------------------------------------------------------
# _apply_permanent_overrides unit tests  (ticket-004)
# ---------------------------------------------------------------------------


class TestApplyPermanentOverrides:
    """Unit tests for ``_apply_permanent_overrides``."""

    def _base_cadastro(self) -> pd.DataFrame:
        return _make_hidr_cadastro()

    def _modif_case(self, tmp_path, mock_modif):
        """Build a case whose MODIF reader is *mock_modif* (path set)."""
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_missing_modif_returns_unchanged(self, tmp_path) -> None:
        """No MODIF.DAT -> cadastro returned unchanged."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        cadastro = self._base_cadastro()
        result = _apply_permanent_overrides(cadastro, make_case(tmp_path, modif=None))
        pd.testing.assert_frame_equal(result, cadastro)

    def test_volmax_override(self, tmp_path) -> None:
        """VOLMAX record updates volume_maximo for the target plant."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        # Build MODIF mock: plant 1 gets VOLMAX=2000.
        volmax_rec = MagicMock()
        volmax_rec.__class__.__name__ = "VOLMAX"
        type(volmax_rec).__name__ = "VOLMAX"
        volmax_rec.volume = 2000.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volmax_rec]

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert float(result.loc[1, "volume_maximo"]) == pytest.approx(2000.0)
        # Plant 2 must be unchanged.
        assert float(result.loc[2, "volume_maximo"]) == pytest.approx(500.0)

    def test_vazmin_override(self, tmp_path) -> None:
        """VAZMIN record updates vazao_minima_historica for the target plant."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        vazmin_rec = MagicMock()
        type(vazmin_rec).__name__ = "VAZMIN"
        vazmin_rec.vazao = 75.5

        usina_rec = MagicMock()
        usina_rec.codigo = 2

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmin_rec]

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert float(result.loc[2, "vazao_minima_historica"]) == pytest.approx(75.5)
        # Plant 1 must be unchanged (was 0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)

    def test_numcnj_nummaq_override(self, tmp_path) -> None:
        """NUMCNJ + NUMMAQ records update machine set counts."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        numcnj_rec = MagicMock()
        type(numcnj_rec).__name__ = "NUMCNJ"
        numcnj_rec.numero = 2

        nummaq_rec = MagicMock()
        type(nummaq_rec).__name__ = "NUMMAQ"
        nummaq_rec.conjunto = 2
        nummaq_rec.numero_maquinas = 3

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [numcnj_rec, nummaq_rec]

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        assert int(result.loc[1, "numero_conjuntos_maquinas"]) == 2
        assert int(result.loc[1, "maquinas_conjunto_2"]) == 3

    def test_volcota_override_warns_and_skips(self, tmp_path, caplog) -> None:
        """VOLCOTA records produce a warning and are skipped gracefully."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        volcota_rec = MagicMock()
        type(volcota_rec).__name__ = "VOLCOTA"

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [volcota_rec]

        with caplog.at_level(logging.WARNING, logger="cobre_bridge.converters.hydro"):
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        # Values must be unchanged (dtype may differ due to float cast for safety).
        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert any("VOLCOTA" in msg for msg in caplog.messages)

    def test_unknown_plant_code_skipped(self, tmp_path, caplog) -> None:
        """Plant code not in cadastro: warning logged, no crash."""
        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        usina_rec = MagicMock()
        usina_rec.codigo = 999  # not in cadastro

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = []

        with caplog.at_level(logging.WARNING, logger="cobre_bridge.converters.hydro"):
            result = _apply_permanent_overrides(
                self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
            )

        pd.testing.assert_frame_equal(result, self._base_cadastro(), check_dtype=False)
        assert any("999" in msg for msg in caplog.messages)

    def test_temporal_records_skipped_in_permanent_pass(self, tmp_path) -> None:
        """Temporal override types are ignored in _apply_permanent_overrides."""
        import datetime

        from cobre_bridge.converters.hydro import _apply_permanent_overrides

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 999.0  # large value that should NOT be applied

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        result = _apply_permanent_overrides(
            self._base_cadastro(), self._modif_case(tmp_path, mock_modif)
        )

        # vazao_minima_historica must stay at the base value (0).
        assert float(result.loc[1, "vazao_minima_historica"]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _extract_temporal_overrides unit tests  (ticket-005)
# ---------------------------------------------------------------------------


class TestExtractTemporalOverrides:
    """Unit tests for ``_extract_temporal_overrides``."""

    def _modif_case(self, tmp_path, mock_modif):
        """Build a case whose MODIF reader is *mock_modif* (path set)."""
        return make_case(
            make_nw_files(tmp_path, modif=tmp_path / "modif.dat"),
            modif=mock_modif,
        )

    def test_missing_modif_returns_empty(self, tmp_path) -> None:
        """No MODIF.DAT -> empty dict returned, no error."""
        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        result = _extract_temporal_overrides(make_case(tmp_path, modif=None), [1, 2])
        assert result == {}

    def test_extracts_vazmint_records(self, tmp_path) -> None:
        """VAZMINT record is extracted with correct month, year, value."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 1, 1)
        vazmint_rec.vazao = 50.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1, 2]
        )

        assert 1 in result
        assert result[1] == [
            {"type": "VAZMINT", "month": 1, "year": 2025, "value": 50.0}
        ]

    def test_filters_by_confhd_codes(self, tmp_path) -> None:
        """Plants not in confhd_codes are excluded from the result."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        vazmint_rec = MagicMock()
        type(vazmint_rec).__name__ = "VAZMINT"
        vazmint_rec.data_inicio = datetime.datetime(2025, 3, 1)
        vazmint_rec.vazao = 40.0

        # Plant 99 is NOT in confhd_codes [1, 2].
        usina_rec = MagicMock()
        usina_rec.codigo = 99

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [vazmint_rec]

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1, 2]
        )

        assert result == {}

    def test_preserves_file_order(self, tmp_path) -> None:
        """Multiple records for the same plant are returned in file order."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        def _vazmint(month: int, vazao: float) -> MagicMock:
            r = MagicMock()
            type(r).__name__ = "VAZMINT"
            r.data_inicio = datetime.datetime(2025, month, 1)
            r.vazao = vazao
            return r

        recs = [_vazmint(1, 50.0), _vazmint(6, 60.0), _vazmint(3, 55.0)]

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = recs

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
        )

        assert len(result[1]) == 3
        assert result[1][0]["value"] == pytest.approx(50.0)
        assert result[1][1]["value"] == pytest.approx(60.0)
        assert result[1][2]["value"] == pytest.approx(55.0)

    def test_extracts_cfuga_records(self, tmp_path) -> None:
        """CFUGA record extracted with correct level value."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        cfuga_rec = MagicMock()
        type(cfuga_rec).__name__ = "CFUGA"
        cfuga_rec.data_inicio = datetime.datetime(2025, 6, 1)
        cfuga_rec.nivel = 75.4

        usina_rec = MagicMock()
        usina_rec.codigo = 2

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [cfuga_rec]

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [2]
        )

        assert result[2] == [
            {"type": "CFUGA", "month": 6, "year": 2025, "value": pytest.approx(75.4)}
        ]

    def test_extracts_turbmint_turbmaxt_records(self, tmp_path) -> None:
        """TURBMINT and TURBMAXT records use turbinamento field."""
        import datetime

        from cobre_bridge.converters.hydro import _extract_temporal_overrides

        turbmint_rec = MagicMock()
        type(turbmint_rec).__name__ = "TURBMINT"
        turbmint_rec.data_inicio = datetime.datetime(2025, 11, 1)
        turbmint_rec.turbinamento = 330.0

        turbmaxt_rec = MagicMock()
        type(turbmaxt_rec).__name__ = "TURBMAXT"
        turbmaxt_rec.data_inicio = datetime.datetime(2025, 3, 1)
        turbmaxt_rec.turbinamento = 322.0

        usina_rec = MagicMock()
        usina_rec.codigo = 1

        mock_modif = MagicMock()
        mock_modif.usina.return_value = [usina_rec]
        mock_modif.modificacoes_usina.return_value = [turbmint_rec, turbmaxt_rec]

        result = _extract_temporal_overrides(
            self._modif_case(tmp_path, mock_modif), [1]
        )

        assert result[1][0] == {
            "type": "TURBMINT",
            "month": 11,
            "year": 2025,
            "value": pytest.approx(330.0),
        }
        assert result[1][1] == {
            "type": "TURBMAXT",
            "month": 3,
            "year": 2025,
            "value": pytest.approx(322.0),
        }


# ---------------------------------------------------------------------------
# _read_ghmin_per_stage unit tests
# ---------------------------------------------------------------------------


class TestReadGhminPerStage:
    """Unit tests for ``_read_ghmin_per_stage``.

    GHMIN values are time-varying and now live in
    ``hydro_bounds.parquet:min_generation_mw`` rather than the static
    ``hydros.json:generation.min_generation_mw``.  This helper expands
    each (plant, month, year) record into a per-(plant, stage_0based)
    mapping with step-function semantics and seasonal post-study
    repetition.
    """

    def _ghmin_case(self, tmp_path, mock_ghmin):
        """Build a case whose GHMIN reader is *mock_ghmin* (path set)."""
        return make_case(
            make_nw_files(tmp_path, ghmin=tmp_path / "ghmin.dat"),
            ghmin=mock_ghmin,
        )

    def test_missing_ghmin_returns_empty(self, tmp_path) -> None:
        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        result = _read_ghmin_per_stage(
            make_case(tmp_path, ghmin=None),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=24,
        )
        assert result == {}

    def test_step_function_persists_until_next_entry(self, tmp_path) -> None:
        """Sparse entries persist the last applied value forward."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        # Plant 1 at Sep 2024 = 100 MW, Dec 2024 = 80 MW.
        # Stages 0 (Sep) and 1 (Oct) and 2 (Nov) should all be 100.
        # Stage 3 (Dec) and onwards should be 80 within the study.
        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),
                    datetime.datetime(2024, 12, 1),
                ],
                "patamar": [0, 0],
                "geracao": [100.0, 80.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=12,
        )

        per_stage = result[1]
        assert per_stage[0] == pytest.approx(100.0)
        assert per_stage[1] == pytest.approx(100.0)
        assert per_stage[2] == pytest.approx(100.0)
        assert per_stage[3] == pytest.approx(80.0)
        assert per_stage[4] == pytest.approx(80.0)

    def test_post_study_uses_pos_seasonal_pattern(self, tmp_path) -> None:
        """POS year=9999 entries supply per-calendar-month values."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),  # study Sep 2024
                    datetime.datetime(9999, 9, 1),  # POS Sep
                    datetime.datetime(9999, 12, 1),  # POS Dec
                ],
                "patamar": [0, 0, 0],
                "geracao": [100.0, 150.0, 200.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,  # study ends Aug 2025
            total_stages=24,  # post-study: Sep 2025 – Aug 2026
        )

        per_stage = result[1]
        # Stage 12 = Sep 2025 → POS Sep = 150.
        assert per_stage[12] == pytest.approx(150.0)
        # Stage 15 = Dec 2025 → POS Dec = 200.
        assert per_stage[15] == pytest.approx(200.0)

    def test_patamar_nonzero_excluded(self, tmp_path) -> None:
        """Rows with patamar != 0 are excluded — only the all-blocks mean
        is meaningful at hydro_bounds' stage granularity."""
        import datetime

        from cobre_bridge.converters.hydro import _read_ghmin_per_stage

        ghmin_df = pd.DataFrame(
            {
                "codigo_usina": [1, 1],
                "data": [
                    datetime.datetime(2024, 9, 1),
                    datetime.datetime(2024, 9, 1),
                ],
                "patamar": [1, 2],
                "geracao": [50.0, 60.0],
            }
        )
        mock_ghmin = MagicMock()
        mock_ghmin.geracoes = ghmin_df

        result = _read_ghmin_per_stage(
            self._ghmin_case(tmp_path, mock_ghmin),
            start_year=2024,
            start_month=9,
            study_months=12,
            total_stages=12,
        )

        assert result == {}


# ---------------------------------------------------------------------------
# _read_penalid unit tests  (ticket-007)
# ---------------------------------------------------------------------------


class TestReadPenalid:
    """Unit tests for ``_read_penalid``."""

    def _penalid_case(self, tmp_path, mock_penalid):
        """Build a case whose PENALID reader is *mock_penalid* (path set)."""
        return make_case(
            make_nw_files(tmp_path, penalid=tmp_path / "penalid.dat"),
            penalid=mock_penalid,
        )

    def test_reads_penalties_by_ree(self, tmp_path) -> None:
        """Correct Cobre field names and values are returned per REE."""
        from cobre_bridge.converters.hydro import _read_penalid

        mock_penalid = MagicMock()
        mock_penalid.penalidades = _make_penalid_df()

        result = _read_penalid(self._penalid_case(tmp_path, mock_penalid))

        # REE 1 checks.
        assert 1 in result
        assert result[1]["water_withdrawal_violation_cost"] == pytest.approx(8300.0)
        assert result[1]["outflow_violation_below_cost"] == pytest.approx(3179.35)
        assert result[1]["generation_violation_below_cost"] == pytest.approx(4500.0)
        # TURBMX must not appear (no Cobre mapping).
        assert "turbined_violation_below_cost" not in result[1]

        # REE 2 checks.
        assert 2 in result
        assert result[2]["water_withdrawal_violation_cost"] == pytest.approx(9100.0)
        assert result[2]["outflow_violation_below_cost"] == pytest.approx(2800.0)

    def test_missing_file_returns_empty(self, tmp_path) -> None:
        """Absent PENALID.DAT returns an empty dict without raising."""
        from cobre_bridge.converters.hydro import _read_penalid

        # No penalid.dat — pass penalid=None.
        result = _read_penalid(make_case(tmp_path, penalid=None))

        assert result == {}

    def test_nan_values_are_skipped(self, tmp_path) -> None:
        """NaN cost values at patamar 1 do not appear in the output dict."""
        from cobre_bridge.converters.hydro import _read_penalid

        df = pd.DataFrame(
            {
                "variavel": ["DESVIO", "VAZMIN"],
                "codigo_ree_submercado": [1, 1],
                "patamar_penalidade": [1, 1],
                "patamar_carga": [1, 1],
                "valor_R$_MWh": [math.nan, 5000.0],
                "valor_R$_hm3": [0.0, 0.0],
            }
        )

        mock_penalid = MagicMock()
        mock_penalid.penalidades = df

        result = _read_penalid(self._penalid_case(tmp_path, mock_penalid))

        assert 1 in result
        # DESVIO had NaN — must be absent.
        assert "water_withdrawal_violation_cost" not in result[1]
        # VAZMIN had 5000.0 — must be present.
        assert result[1]["outflow_violation_below_cost"] == pytest.approx(5000.0)

    def test_patamar2_rows_ignored(self, tmp_path) -> None:
        """Tier-2 patamar rows are excluded even when they have numeric values."""
        from cobre_bridge.converters.hydro import _read_penalid

        df = pd.DataFrame(
            {
                "variavel": ["DESVIO", "DESVIO"],
                "codigo_ree_submercado": [1, 1],
                "patamar_penalidade": [2, 2],  # only tier-2 rows — should be skipped
                "patamar_carga": [1, 1],
                "valor_R$_MWh": [8300.0, 8300.0],
                "valor_R$_hm3": [0.0, 0.0],
            }
        )

        mock_penalid = MagicMock()
        mock_penalid.penalidades = df

        result = _read_penalid(self._penalid_case(tmp_path, mock_penalid))

        assert result == {}
