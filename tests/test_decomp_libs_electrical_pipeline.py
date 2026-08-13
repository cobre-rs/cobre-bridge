"""TICKET-013: the LIBs-era electrical-constraint emitter spliced into
``convert_decomp_case`` -- discovery of the deck's ``lib_restricao-eletrica-
especial.csv`` file, the narrowed ``detect_libs_electrical`` warn, the
census INFO diagnostic, and the end-to-end ``cobre validate`` smoke.

Tier 1 (the bulk of this module): synthetic stub decks only, mirroring
``tests/test_decomp_pipeline.py``'s own ``_run_cadastro_pipeline`` convention
-- no ``example/`` read, no ``import cobre`` at module scope.

Tier 3 (the tail of this module, ``TestRealDeckValidation``): the real
``example/decomp-abr-26-lpp`` deck converted and validated against the local
``cobre`` binary, guarded exactly like ``tests/test_decomp_fcf_roundtrip.py``.
"""

from __future__ import annotations

import json
import subprocess
from contextlib import ExitStack
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.decomp.bounds_accumulator import BoundContribution
from cobre_bridge.decomp.constraint_registers import (
    ConstraintCensus,
    ConstraintRecord,
    ConstraintTerm,
    StageBounds,
    resolve_libs_electrical_path,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.libs_electrical import (
    ElectricalRestriction,
    LibsElectricalModel,
)
from cobre_bridge.decomp.network import _LINE_BOUNDS_SCHEMA
from cobre_bridge.decomp.pipeline import ConversionReport, DecompFiles
from cobre_bridge.decomp.temporal import build_operative_calendar
from cobre_bridge.decomp.thermal import _THERMAL_COST_SCHEMA, ThermalBounds

# ---------------------------------------------------------------------------
# resolve_libs_electrical_path -- pure path resolution, no pipeline involved
# ---------------------------------------------------------------------------


def _write_indices_csv(deck_dir: Path, entries: list[tuple[str, str]]) -> None:
    lines = [
        "&**** ID ; Descricao ; Caminho",
        *(
            f" {entry_id} ;description here;{filename}"
            for entry_id, filename in entries
        ),
    ]
    (deck_dir / "indices.csv").write_text("\n".join(lines) + "\n", encoding="latin-1")


class TestResolveLibsElectricalPath:
    def test_indices_entry_resolves_to_its_own_filename(self, tmp_path: Path) -> None:
        _write_indices_csv(
            tmp_path,
            [("RESTRICAO-ELETRICA-ESPECIAL", "lib_restricao-eletrica-especial.csv")],
        )
        target = tmp_path / "lib_restricao-eletrica-especial.csv"
        target.write_text("", encoding="latin-1")

        assert resolve_libs_electrical_path(tmp_path) == target

    def test_exact_entry_match_not_the_ativacao_sibling(self, tmp_path: Path) -> None:
        """The ``-ATIVACAO``/``-VIOLACAO`` sibling entries must never be
        mistaken for the base entry -- only an exact ``ID`` match resolves."""
        _write_indices_csv(
            tmp_path,
            [
                ("RESTRICAO-ELETRICA-ESPECIAL-ATIVACAO", "other-file.csv"),
                ("RESTRICAO-ELETRICA-ESPECIAL", "lib_restricao-eletrica-especial.csv"),
            ],
        )
        target = tmp_path / "lib_restricao-eletrica-especial.csv"
        target.write_text("", encoding="latin-1")
        (tmp_path / "other-file.csv").write_text("", encoding="latin-1")

        assert resolve_libs_electrical_path(tmp_path) == target

    def test_no_indices_csv_falls_back_to_glob(self, tmp_path: Path) -> None:
        target = tmp_path / "lib_restricao-eletrica-especial.csv"
        target.write_text("", encoding="latin-1")

        assert resolve_libs_electrical_path(tmp_path) == target

    def test_indices_entry_pointing_at_missing_file_falls_back_to_glob(
        self, tmp_path: Path
    ) -> None:
        _write_indices_csv(
            tmp_path,
            [("RESTRICAO-ELETRICA-ESPECIAL", "does-not-exist.csv")],
        )
        target = tmp_path / "lib_restricao-eletrica-especial.csv"
        target.write_text("", encoding="latin-1")

        assert resolve_libs_electrical_path(tmp_path) == target

    def test_nothing_present_returns_none(self, tmp_path: Path) -> None:
        assert resolve_libs_electrical_path(tmp_path) is None

    def test_indices_csv_present_but_no_matching_entry_and_no_glob_returns_none(
        self, tmp_path: Path
    ) -> None:
        _write_indices_csv(tmp_path, [("SOME-OTHER-CARD", "other.csv")])
        (tmp_path / "other.csv").write_text("", encoding="latin-1")

        assert resolve_libs_electrical_path(tmp_path) is None


class TestDiscoverDecompFilesLibsElectrical:
    """``discover_decomp_files`` threads :func:`resolve_libs_electrical_path`
    into ``DecompFiles.libs_restricao_eletrica`` (Requirement 1)."""

    @staticmethod
    def _minimal_deck(deck_dir: Path) -> None:
        (deck_dir / "caso.dat").write_text("rv0", encoding="latin-1")
        (deck_dir / "dadger.rv0").write_text("", encoding="latin-1")
        (deck_dir / "vazoes.rv0").write_text("", encoding="latin-1")
        (deck_dir / "hidr.dat").write_text("", encoding="latin-1")

    def test_libs_restricao_eletrica_resolved_via_indices_csv(
        self, tmp_path: Path
    ) -> None:
        from cobre_bridge.decomp.pipeline import discover_decomp_files

        self._minimal_deck(tmp_path)
        _write_indices_csv(
            tmp_path,
            [("RESTRICAO-ELETRICA-ESPECIAL", "lib_restricao-eletrica-especial.csv")],
        )
        (tmp_path / "lib_restricao-eletrica-especial.csv").write_text(
            "", encoding="latin-1"
        )

        files = discover_decomp_files(tmp_path)
        assert files.libs_restricao_eletrica == (
            tmp_path / "lib_restricao-eletrica-especial.csv"
        )

    def test_libs_restricao_eletrica_none_when_absent(self, tmp_path: Path) -> None:
        from cobre_bridge.decomp.pipeline import discover_decomp_files

        self._minimal_deck(tmp_path)

        files = discover_decomp_files(tmp_path)
        assert files.libs_restricao_eletrica is None

    def test_decomp_files_still_constructs_without_the_new_field(self) -> None:
        """Every pre-existing ``DecompFiles(...)`` call site (other test
        modules this ticket must not touch) keeps constructing unchanged --
        the new field defaults to ``None``."""
        files = DecompFiles(
            revision="rv0",
            dadger=Path("dadger.rv0"),
            vazoes=Path("vazoes.rv0"),
            hidr=Path("hidr.dat"),
            dadgnl=None,
            renovaveis=None,
            polinjus=None,
        )
        assert files.libs_restricao_eletrica is None


# ---------------------------------------------------------------------------
# Tier-1 pipeline wiring -- a fully synthetic mock deck, mirroring
# tests/test_decomp_pipeline.py::_run_cadastro_pipeline's own convention.
# ---------------------------------------------------------------------------


def _plant_row(
    name: str, sub: int, jusante: int, vmin: float, vmax: float, posto: int
) -> dict:
    return {
        "nome_usina": name,
        "submercado": sub,
        "codigo_usina_jusante": jusante,
        "posto": posto,
        "desvio": 0,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "tipo_regulacao": "M",
        "volume_referencia": vmin + 0.5 * (vmax - vmin),
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 100.0,
        "potencia_nominal_conjunto_1": 50.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": 100.0,
        "a1_volume_cota": 0.0,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        "a0_cota_area": 1.0,
        "a1_cota_area": 0.0,
        "a2_cota_area": 0.0,
        "a3_cota_area": 0.0,
        "a4_cota_area": 0.0,
        "canal_fuga_medio": 20.0,
        "produtibilidade_especifica": 0.009,
        "tipo_perda": 0,
        "perdas": 0.0,
    }


def _hidr_frame() -> pd.DataFrame:
    """Two plants, codes 1 and 2, both on bus (submercado) 1, no cascade."""
    df = pd.DataFrame(
        {
            1: _plant_row("PLANT_ONE", 1, 0, 20.0, 100.0, posto=11),
            2: _plant_row("PLANT_TWO", 1, 0, 10.0, 50.0, posto=12),
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _uh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
                "codigo_ree": 1,
            },
            {
                "codigo_usina": 2,
                "volume_inicial": 50.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
                "codigo_ree": 1,
            },
        ]
    )


_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1, 2))
_N_BLOCKS = 3


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _dp_frame() -> pd.DataFrame:
    """One ``DP`` row per calendar stage, all zero load -- ``build_data_context``
    (Requirement 2) eagerly reads ``DP`` via ``_per_stage_block_loads``
    regardless of whether a restriction's bucket-B side ever references
    ``demanda``/``demanda_sin``, so a valid (if trivial) frame is required
    even for the pure bucket-A restrictions this module's fixtures use."""
    return pd.DataFrame(
        [
            {
                "estagio": stage,
                "codigo_submercado": 1,
                "numero_patamares": _N_BLOCKS,
                "carga_1": 0.0,
                "carga_2": 0.0,
                "carga_3": 0.0,
            }
            for stage in range(1, 4)
        ]
    )


class _MockDadger:
    """Covers only what the real (unmocked) consumers this fixture exercises
    need: ``.tx.taxa``, ``.uh(df=True)``, ``.vi``, ``.ac(...)``, ``.dp(...)``,
    ``.ri(...)`` -- mirrors ``tests/test_decomp_pipeline.py``'s own
    ``_CadastroDadger``, extended for ``build_data_context``'s own reads."""

    class _Tx:
        taxa = 12.0

    def __init__(self) -> None:
        self.tx = self._Tx()
        self._uh = _uh_frame()
        self._dp = _dp_frame()

    def uh(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._uh

    def vi(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def dp(self, df: bool = False) -> pd.DataFrame:  # noqa: ARG002
        return self._dp

    def ri(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return None

    def ac(
        self,
        codigo_usina: int | None = None,  # noqa: ARG002
        modificacao: type | None = None,  # noqa: ARG002
        df: bool = False,  # noqa: ARG002
    ) -> pd.DataFrame | None:
        return None


def _cap_restriction(code: int, plant_code: int, cap: float) -> ElectricalRestriction:
    """A pure bucket-A/constant INEQUACAO restriction (``ger_usih(plant_code)
    <= cap``), active across the whole calendar (no ``habilita``, no
    horizon) -- needs neither ``a_h``/``disp_usih`` resolution nor any
    per-(stage,block) data-context lookup, so it isolates the pipeline's own
    wiring (threading ``id_map``/``context_factory``/``big_m``/
    ``next_generic_id`` correctly) from the emitter's own correctness
    (covered by ``tests/test_decomp_libs_electrical_emit.py``)."""
    return ElectricalRestriction(
        code=code,
        lhs=f"ger_usih({plant_code})",
        operator="<=",
        rhs=str(cap),
        is_formula=False,
        limits={},
        overrides=(),
    )


def _libs_model() -> LibsElectricalModel:
    return LibsElectricalModel(
        expressions={},
        aliases={},
        restrictions={
            601: _cap_restriction(601, plant_code=1, cap=500.0),
            602: _cap_restriction(602, plant_code=2, cap=300.0),
        },
        rules={},
    )


def _re_record() -> ConstraintRecord:
    """A multi-term RE record (never lowers to a bound) so
    ``emit_re_generics`` emits exactly one id ahead of the LIBs-electrical
    ones -- proves the shared allocator, not just the LIBs link in
    isolation."""
    return ConstraintRecord(
        family="RE",
        constraint_id=901,
        stage_start=0,
        stage_end=2,
        terms=(
            ConstraintTerm(code=1, coefficient=1.0, variable="generation"),
            ConstraintTerm(code=2, coefficient=1.0, variable="generation"),
        ),
        bounds={
            stage: StageBounds(lower=(None,) * _N_BLOCKS, upper=(100.0,) * _N_BLOCKS)
            for stage in range(3)
        },
        per_block=True,
    )


def _run_libs_pipeline(
    tmp_path: Path,
    *,
    libs_restricao_eletrica: Path | None = None,
    libs_electrical_model: object | None = None,
    detect_libs_electrical_return: dx.Diagnostic | None = None,
    to_generic: tuple[ConstraintRecord, ...] = (),
    diagnostics_out: list[dx.Diagnostic] | None = None,
    report_out: list[ConversionReport] | None = None,
) -> Path:
    """Run ``convert_decomp_case`` against a fully synthetic mock deck,
    patching every converter this ticket does not wire to a canned return
    value -- mirrors ``tests/test_decomp_pipeline.py``'s
    ``_run_cadastro_pipeline`` (duplicated rather than imported, per this
    codebase's own convention of copying a tier-3/tier-1 fixture verbatim
    across test modules, e.g. ``test_decomp_fcf_roundtrip.py``'s docstring).

    *libs_restricao_eletrica*/*libs_electrical_model* control
    ``files.libs_restricao_eletrica`` and the (patched)
    ``read_libs_electrical`` return value; *detect_libs_electrical_return*
    controls the (patched) flat presence-detection diagnostic, independent
    of whether the model itself converted -- so a test can prove the
    pipeline suppresses it even when it WOULD have fired.
    """
    from cobre_bridge.decomp import pipeline as decomp_pipeline

    files = DecompFiles(
        revision="rv0",
        dadger=Path("unused/dadger.rv0"),
        vazoes=Path("unused/vazoes.rv0"),
        hidr=Path("unused/hidr.dat"),
        dadgnl=None,
        renovaveis=None,
        polinjus=None,
        libs_restricao_eletrica=libs_restricao_eletrica,
    )
    dadger = _MockDadger()
    hidr = _hidr_frame()
    calendar = _calendar()

    productivity_table = pa.table(
        {"equivalent_productivity_mw_per_m3s": pa.array([0.5, 0.6], type=pa.float64())}
    )
    load_stats_table = pa.table(
        {
            "bus_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean_mw": pa.array([10.0, 10.0, 10.0], type=pa.float64()),
        }
    )
    ncs_stats_table = pa.table(
        {
            "ncs_id": pa.array([0, 0, 0], type=pa.int32()),
            "stage_id": pa.array([0, 1, 2], type=pa.int32()),
            "mean": pa.array([5.0, 5.0, 5.0], type=pa.float64()),
        }
    )
    external_inflow_table = pa.table(
        {
            "hydro_id": pa.array([], type=pa.int32()),
            "stage_id": pa.array([], type=pa.int32()),
            "scenario_id": pa.array([], type=pa.int32()),
            "value_m3s": pa.array([], type=pa.float64()),
        }
    )
    baseline_hydro_bounds = [
        BoundContribution(
            family="hydro",
            entity_id=0,
            stage_id=0,
            block_id=None,
            axis="outflow",
            lower=5.0,
            upper=None,
            contributor="RQ",
        )
    ]

    patches: dict[str, object] = {
        "cobre_bridge.decomp.pipeline.discover_decomp_files": files,
        "cobre_bridge.decomp.pipeline.Dadger.read": dadger,
        "cobre_bridge.decomp.pipeline.Vazoes.read": object(),
        "cobre_bridge.decomp.pipeline.hydro_conv.read_hidr": hidr,
        "cobre_bridge.decomp.pipeline.DecompIdMap.from_dadger": _ID_MAP,
        "cobre_bridge.decomp.pipeline"
        ".temporal_conv.operative_calendar_from_dadger": calendar,
        "cobre_bridge.decomp.pipeline.scenarios_conv.terminal_fan_probabilities": [1.0],
        "cobre_bridge.decomp.pipeline.config_conv.convert_config": {},
        "cobre_bridge.decomp.pipeline.network_conv._bus_deficit_costs": {},
        "cobre_bridge.decomp.pipeline"
        ".hydro_conv.convert_energy_productivity": productivity_table,
        "cobre_bridge.decomp.pipeline.network_conv.convert_buses": {"buses": []},
        "cobre_bridge.decomp.pipeline.network_conv.convert_lines": (
            {"lines": []},
            _LINE_BOUNDS_SCHEMA.empty_table(),
        ),
        "cobre_bridge.decomp.pipeline.network_conv.convert_pumping_stations": {
            "pumping_stations": []
        },
        "cobre_bridge.decomp.pipeline.thermal_conv.convert_thermals": {"thermals": []},
        "cobre_bridge.decomp.pipeline.ncs_conv.convert_non_controllable_sources": {
            "non_controllable_sources": []
        },
        "cobre_bridge.decomp.pipeline"
        ".scenarios_conv.convert_external_inflows": external_inflow_table,
        "cobre_bridge.decomp.pipeline.load_conv.convert_load_stats": load_stats_table,
        "cobre_bridge.decomp.pipeline.load_conv.convert_load_factors": {},
        "cobre_bridge.decomp.pipeline.ncs_conv.convert_ncs_stats": ncs_stats_table,
        "cobre_bridge.decomp.pipeline.ncs_conv.convert_ncs_factors": {},
        "cobre_bridge.decomp.pipeline.thermal_conv.convert_thermal_bounds": (
            ThermalBounds(generation=[], cost=_THERMAL_COST_SCHEMA.empty_table())
        ),
        "cobre_bridge.decomp.pipeline"
        ".bounds_conv.convert_hydro_bounds": baseline_hydro_bounds,
        "cobre_bridge.decomp.pipeline.hydro_conv.convert_hydro_group_availability": {},
        "cobre_bridge.decomp.pipeline.contracts_conv.read_contracts": [],
        "cobre_bridge.decomp.pipeline.constraint_registers.read_constraints": (
            ConstraintCensus(by_family={}, to_bounds=(), to_generic=to_generic)
        ),
        "cobre_bridge.decomp.pipeline.network_conv.pumping_station_id_map": {},
        "cobre_bridge.decomp.pipeline"
        ".constraint_registers.detect_unreadable_electrical": [],
        "cobre_bridge.decomp.pipeline.constraint_registers.detect_libs_electrical": (
            detect_libs_electrical_return
        ),
        "cobre_bridge.decomp.pipeline.libs_electrical_conv.read_libs_electrical": (
            libs_electrical_model
        ),
    }
    with ExitStack() as stack:
        for target, value in patches.items():
            stack.enter_context(patch(target, return_value=value))
        # `Restricoes.read` is a local (deferred) import inside
        # `_convert_decomp_case_impl`, so it must be patched at its origin,
        # not via a `cobre_bridge.decomp.pipeline` attribute path -- it is
        # only ever reached when `files.libs_restricao_eletrica` is set, and
        # `read_libs_electrical` above is independently patched to ignore
        # its `restricoes` argument, so the sentinel return value here is
        # never inspected.
        stack.enter_context(
            patch("idecomp.libs.restricoes.Restricoes.read", return_value=object())
        )
        dst = tmp_path / "case"
        report = decomp_pipeline.convert_decomp_case(Path("unused-src"), dst)
    if diagnostics_out is not None:
        diagnostics_out.extend(report.diagnostics)
    if report_out is not None:
        report_out.append(report)
    return dst


class TestLibsElectricalPipelineWiring:
    def test_ac1_converted_restrictions_appended_after_rhe_with_dense_ids(
        self, tmp_path: Path
    ) -> None:
        """AC1: a stub deck whose ``read_libs_electrical`` yields a model
        with convertible restrictions gets ``LIBS_ELEC_<code>`` constraints
        appended AFTER the (empty here) RHQ/RHV/RHE links, sharing the
        allocator with a synthetic RE record so the id space never
        collides."""
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=Path("unused/lib_restricao-eletrica-especial.csv"),
            libs_electrical_model=_libs_model(),
            to_generic=(_re_record(),),
        )

        doc = json.loads((dst / "constraints" / "generic_constraints.json").read_text())
        names = [c["name"] for c in doc["constraints"]]
        assert names == ["RE_901", "LIBS_ELEC_601", "LIBS_ELEC_602"]
        ids = [c["id"] for c in doc["constraints"]]
        assert sorted(ids) == list(range(3))
        assert len(set(ids)) == 3

        bounds_table = pq.read_table(
            dst / "constraints" / "generic_constraint_bounds.parquet"
        )
        libs_ids = {ids[1], ids[2]}
        libs_rows = [
            row for row in bounds_table.to_pylist() if row["constraint_id"] in libs_ids
        ]
        assert libs_rows

    def test_ac2_no_entry_is_a_no_op(self, tmp_path: Path) -> None:
        """AC2: no ``RESTRICAO-ELETRICA-ESPECIAL`` entry -- no electrical
        constraint emitted, no libs diagnostic produced, byte-identical to
        the pre-ticket pipeline for that path."""
        diagnostics_out: list[dx.Diagnostic] = []
        dst = _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=None,
            libs_electrical_model=None,
            detect_libs_electrical_return=None,
            diagnostics_out=diagnostics_out,
        )

        assert not (dst / "constraints" / "generic_constraints.json").exists()
        assert not (dst / "constraints" / "generic_constraint_bounds.parquet").exists()
        codes = {d.code for d in diagnostics_out}
        assert "decomp-libs-electrical-present" not in codes
        assert "decomp-libs-electrical-converted" not in codes

    def test_ac3_short_form_only_still_warns(self, tmp_path: Path) -> None:
        """AC3: the entry is present but ``read_libs_electrical`` returns
        ``None`` (short-form-only) -- the narrowed warn IS still emitted
        (OQ-4 fallback preserved)."""
        canned_warning = dx.Diagnostic(
            code="decomp-libs-electrical-present",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="LIBs electrical constraints present",
            summary="synthetic short-form-only finding",
        )
        diagnostics_out: list[dx.Diagnostic] = []
        _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=Path("unused/lib_restricao-eletrica-especial.csv"),
            libs_electrical_model=None,
            detect_libs_electrical_return=canned_warning,
            diagnostics_out=diagnostics_out,
        )

        codes = {d.code for d in diagnostics_out}
        assert "decomp-libs-electrical-present" in codes
        assert "decomp-libs-electrical-converted" not in codes

    def test_ac4_census_info_and_suppressed_warning_for_converted_subset(
        self, tmp_path: Path
    ) -> None:
        """AC4: exactly one INFO census diagnostic reporting the converted
        count and per-reason deferred counts, and NO
        ``decomp-libs-electrical-present`` WARNING for the converted subset
        -- even though ``detect_libs_electrical`` is stubbed to return one,
        proving the pipeline itself suppresses it rather than it merely
        being absent by coincidence."""
        would_have_warned = dx.Diagnostic(
            code="decomp-libs-electrical-present",
            severity=dx.Severity.WARNING,
            category="Special constraints",
            title="LIBs electrical constraints present",
            summary="would fire if not suppressed",
        )
        diagnostics_out: list[dx.Diagnostic] = []
        _run_libs_pipeline(
            tmp_path,
            libs_restricao_eletrica=Path("unused/lib_restricao-eletrica-especial.csv"),
            libs_electrical_model=_libs_model(),
            detect_libs_electrical_return=would_have_warned,
            diagnostics_out=diagnostics_out,
        )

        census = [
            d for d in diagnostics_out if d.code == "decomp-libs-electrical-converted"
        ]
        assert len(census) == 1
        [diagnostic] = census
        assert diagnostic.severity is dx.Severity.INFO
        assert diagnostic.table is not None
        counts = dict(diagnostic.table.rows)
        assert counts["Converted"] == 2
        assert counts["inactive"] == 0
        assert counts["unresolved-bucket-a"] == 0
        assert counts["unresolved-bucket-bc"] == 0

        assert "decomp-libs-electrical-present" not in {d.code for d in diagnostics_out}


# ---------------------------------------------------------------------------
# Tier 3: the real deck + the local cobre binary, guarded exactly like
# tests/test_decomp_fcf_roundtrip.py.
# ---------------------------------------------------------------------------

_DECK = Path("example/decomp-abr-26-lpp")
_COBRE_BIN = Path.home() / "git" / "cobre" / "target" / "release" / "cobre"
_HAS_E2E_DEPS = _COBRE_BIN.exists() and (_DECK / "caso.dat").exists()
_skip_e2e = pytest.mark.skipif(
    not _HAS_E2E_DEPS,
    reason=f"requires the local cobre binary ({_COBRE_BIN}) and the {_DECK} deck",
)


class TestRealDeckValidation:
    @_skip_e2e
    def test_abr_26_lpp_converts_and_validates_with_libs_electrical_generics(
        self, tmp_path: Path
    ) -> None:
        """AC5: converting the real deck emits >= 1 ``LIBS_ELEC_*`` generic
        constraint with matching rows in
        ``generic_constraint_bounds.parquet``, and ``cobre validate`` on the
        converted case exits 0.

        ``cobre validate`` is a load/schema gate, not a solve -- it does NOT
        prove LP feasibility (see ticket-013's documented B1 residual: the
        ``IV`` transshipment bus carries ``carga_ande`` with no deficit
        curve, so an ANDE shortfall would be an LP infeasibility rather than
        a priced deficit; benign on this deck per epic-02's own findings,
        but not exercised by a bare ``validate``)."""
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        dst = tmp_path / "decomp-abr-26-lpp-converted"
        convert_decomp_case(_DECK, dst, force=True)

        generics_path = dst / "constraints" / "generic_constraints.json"
        assert generics_path.is_file()
        doc = json.loads(generics_path.read_text())
        libs_constraints = [
            c for c in doc["constraints"] if c["name"].startswith("LIBS_ELEC_")
        ]
        assert len(libs_constraints) >= 1, "expected >= 1 LIBS_ELEC_* constraint"

        bounds_table = pq.read_table(
            dst / "constraints" / "generic_constraint_bounds.parquet"
        )
        libs_ids = {c["id"] for c in libs_constraints}
        libs_rows = [
            row for row in bounds_table.to_pylist() if row["constraint_id"] in libs_ids
        ]
        assert libs_rows, "expected matching generic_constraint_bounds.parquet rows"

        result = subprocess.run(
            [str(_COBRE_BIN), "validate", str(dst)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"cobre validate failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
