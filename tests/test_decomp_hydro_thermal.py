"""Tests for the DECOMP hydro and thermal converters."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import pytest

from cobre_bridge.decomp.cadastro import EffectiveCadastro
from cobre_bridge.decomp.hydro import (
    _build_split_unit_groups,
    _evaporation_coefficients_mm,
    _evaporation_flag_codes,
    convert_energy_productivity,
    convert_hydros,
    convert_initial_storage,
    convert_production_models,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import build_operative_calendar
from cobre_bridge.decomp.thermal import convert_thermal_bounds, convert_thermals

_EVAPORATION_COLUMNS = (
    "evaporacao_JAN",
    "evaporacao_FEV",
    "evaporacao_MAR",
    "evaporacao_ABR",
    "evaporacao_MAI",
    "evaporacao_JUN",
    "evaporacao_JUL",
    "evaporacao_AGO",
    "evaporacao_SET",
    "evaporacao_OUT",
    "evaporacao_NOV",
    "evaporacao_DEZ",
)


def _evaporation_hidr(rows: dict[int, list[float]]) -> pd.DataFrame:
    """A ``hidr``-shaped frame carrying only the 12 monthly evaporation columns."""
    df = pd.DataFrame(
        {
            code: dict(zip(_EVAPORATION_COLUMNS, vals, strict=True))
            for code, vals in rows.items()
        }
    ).T
    df.index.name = "codigo_usina"
    return df


class TestEvaporationEmission:
    """Reservoir-evaporation conversion (C11 fixed in cobre 0.14): the per-plant
    UH ``evaporacao`` flag switches it on; the 12 monthly mm rates come from
    ``hidr.dat`` in calendar order (Jan..Dec)."""

    def test_coefficients_are_hidr_months_jan_to_dec(self) -> None:
        hidr = _evaporation_hidr({7: [0, 2, 29, 40, 51, 46, 32, 23, 24, 15, 4, 7]})
        assert _evaporation_coefficients_mm(hidr, 7) == [
            0.0,
            2.0,
            29.0,
            40.0,
            51.0,
            46.0,
            32.0,
            23.0,
            24.0,
            15.0,
            4.0,
            7.0,
        ]

    def test_all_zero_months_is_none(self) -> None:
        hidr = _evaporation_hidr({7: [0.0] * 12})
        assert _evaporation_coefficients_mm(hidr, 7) is None

    def test_absent_plant_is_none(self) -> None:
        hidr = _evaporation_hidr({7: [1.0] * 12})
        assert _evaporation_coefficients_mm(hidr, 99) is None

    def test_missing_columns_is_none_not_keyerror(self) -> None:
        # A hidr frame without the evaporation columns (e.g. a partial fixture)
        # yields None rather than raising, so a stray UH flag never crashes.
        hidr = pd.DataFrame({7: {"volume_maximo": 100.0}}).T
        hidr.index.name = "codigo_usina"
        assert _evaporation_coefficients_mm(hidr, 7) is None

    def test_flag_codes_reads_uh_evaporacao(self) -> None:
        uh = pd.DataFrame(
            [
                {"codigo_usina": 1, "evaporacao": 1},
                {"codigo_usina": 2, "evaporacao": 0},
                {"codigo_usina": 3, "evaporacao": 1},
            ]
        )
        assert _evaporation_flag_codes(_StubDadger(uh=uh)) == {1, 3}

    def test_flag_codes_absent_column_is_empty(self) -> None:
        assert _evaporation_flag_codes(_StubDadger(uh=pd.DataFrame())) == set()


_ID_MAP = DecompIdMap(
    bus_codes=(1, 2, 3, 4, 11),
    bus_names=("SE", "S", "NE", "N", "FC"),
    hydro_codes=(1, 2, 5),
    thermal_codes=(10, 20),
)


def _calendar():
    hours = [[15.0, 64.0, 89.0]] * 2 + [[63.0, 280.0, 401.0]]
    return build_operative_calendar(date(2026, 7, 18), hours)


def _plant_row(
    name: str,
    sub: int,
    jusante: int,
    vmin: float,
    vmax: float,
    cota: float = 100.0,
    cf: float = 20.0,
) -> dict:
    return {
        "nome_usina": name,
        "submercado": sub,
        "codigo_usina_jusante": jusante,
        "volume_minimo": vmin,
        "volume_maximo": vmax,
        "numero_conjuntos_maquinas": 1,
        "maquinas_conjunto_1": 2,
        "vazao_nominal_conjunto_1": 100.0,
        "potencia_nominal_conjunto_1": 50.0,
        "teif": 0.0,
        "ip": 0.0,
        "a0_volume_cota": cota,
        "a1_volume_cota": 0.0,
        "a2_volume_cota": 0.0,
        "a3_volume_cota": 0.0,
        "a4_volume_cota": 0.0,
        "canal_fuga_medio": cf,
        "produtibilidade_especifica": 0.009,
        "tipo_perda": 0,
        "perdas": 0.0,
    }


def _hidr_frame() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            1: _plant_row("UP_RES", 1, 3, 100.0, 500.0),
            # Plant 3 exists in the registry but is not operated (skipped through).
            3: _plant_row("SKIPPED", 1, 2, 0.0, 0.0),
            2: _plant_row("MID_RES", 2, 0, 50.0, 250.0),
            5: _plant_row("LEAF", 4, 0, 10.0, 10.0),
        }
    ).T
    df.index.name = "codigo_usina"
    return df


def _no_override_effective(hidr: pd.DataFrame, n_stages: int = 1) -> EffectiveCadastro:
    """Empty-override view of *hidr*: falls through to the base scalar at
    every stage (outer envelope == base, stage-0 == base) — the ticket-007
    regression fixture: reservoir/initial-storage output must be
    byte-identical to the pre-layer base-registry reads.
    """
    return EffectiveCadastro(base=hidr, n_stages=n_stages, stage_varying={})


def _temporal_hidr_frame() -> pd.DataFrame:
    """Single plant, code 1: base ``volume_minimo``/``volume_maximo`` 20.0/100.0."""
    df = pd.DataFrame({1: _plant_row("TEMPORAL", 1, 0, 20.0, 100.0)}).T
    df.index.name = "codigo_usina"
    return df


def _temporal_uh_frame(volume_inicial: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": 1,
                "volume_inicial": volume_inicial,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
        ]
    )


_TEMPORAL_ID_MAP = DecompIdMap(bus_codes=(1,), bus_names=("SE",), hydro_codes=(1,))


def _raised_envelope_effective() -> EffectiveCadastro:
    """Plant 1: flat ``volume_minimo`` 20.0; ``volume_maximo`` raised to
    250.0 at stage 2 (mirrors the storage-bounds emitter's own fixture,
    ticket-006) — the outer envelope is ``(20.0, 250.0)``, wider than the
    stage-0 ``(20.0, 100.0)`` used for the initial-storage % clamp.
    """
    return EffectiveCadastro(
        base=_temporal_hidr_frame(),
        n_stages=3,
        stage_varying={(1, "volume_maximo"): (100.0, 100.0, 250.0)},
    )


class _StubDadger:
    def __init__(
        self,
        uh: pd.DataFrame | None = None,
        ct: pd.DataFrame | None = None,
        mp: pd.DataFrame | None = None,
        fd: pd.DataFrame | None = None,
    ) -> None:
        self._uh, self._ct, self._mp, self._fd = uh, ct, mp, fd

    def uh(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._uh

    def ct(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._ct

    def mp(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._mp

    def fd(self, df: bool = False) -> pd.DataFrame | None:  # noqa: ARG002
        return self._fd

    def ac(  # noqa: ARG002
        self,
        codigo_usina: int | None = None,
        modificacao: type | None = None,
        df: bool = False,
    ) -> pd.DataFrame:
        # No AC machine-configuration overrides on this synthetic stub — every
        # plant here has teif=ip=0.0 anyway, so the AC-adjusted rated capacity
        # (decomp/hydro.py::_compute_max_turbined_rated_ac_adjusted) reduces
        # to the plain registry rated sum regardless.
        return pd.DataFrame()


def _uh_frame() -> pd.DataFrame:
    rows = [
        {
            "codigo_usina": 1,
            "volume_inicial": 50.0,
            "vazao_defluente_minima": 30.0,
            "volume_morto_inicial": None,
        },
        {
            "codigo_usina": 2,
            "volume_inicial": 100.0,
            "vazao_defluente_minima": None,
            "volume_morto_inicial": None,
        },
        {
            "codigo_usina": 5,
            "volume_inicial": 0.0,
            "vazao_defluente_minima": None,
            "volume_morto_inicial": None,
        },
        # Coupling-only registration: no initial volume.
        {
            "codigo_usina": 99,
            "volume_inicial": None,
            "vazao_defluente_minima": None,
            "volume_morto_inicial": None,
        },
    ]
    return pd.DataFrame(rows)


#: ticket-006: a minimal per-frequency split-plant (Itaipu-shaped) fixture —
#: two identical conjuntos (2 machines x 100 m3/s x 50 MW each), submercado
#: 1 (SE) — so the bus-relocation math reuses the same 100.0 / 0.72 head-free
#: engolimento ratio already pinned for ``UP_RES`` above, just per-conjunto.
_ITAIPU_ID_MAP = DecompIdMap(bus_codes=(1, 2), bus_names=("SE", "S"), hydro_codes=(66,))


def _itaipu_hidr_frame() -> pd.DataFrame:
    row = _plant_row("ITAIPU", 1, 0, 100.0, 100.0)
    row["numero_conjuntos_maquinas"] = 2
    row["maquinas_conjunto_2"] = 2
    row["vazao_nominal_conjunto_2"] = 100.0
    row["potencia_nominal_conjunto_2"] = 50.0
    df = pd.DataFrame({66: row}).T
    df.index.name = "codigo_usina"
    return df


def _itaipu_uh_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "codigo_usina": 66,
                "volume_inicial": 100.0,
                "vazao_defluente_minima": None,
                "volume_morto_inicial": None,
            }
        ]
    )


def _itaipu_frequency_frame() -> pd.DataFrame:
    """Shared shape for the ``MP``/``FD`` stub tables: one row per
    ``frequencia`` (50, 60), matching the real registers'
    ``codigo_usina``/``frequencia`` key that :func:`_split_plant_frequencies`
    reads."""
    return pd.DataFrame(
        [
            {"codigo_usina": 66, "frequencia": 50.0},
            {"codigo_usina": 66, "frequencia": 60.0},
        ]
    )


class TestConvertHydros:
    def test_registry_entries_and_cascade_skip(self, caplog) -> None:
        hidr = _hidr_frame()
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.hydro"):
            doc = convert_hydros(
                _StubDadger(uh=_uh_frame()),
                hidr,
                _ID_MAP,
                date(2026, 7, 18),
                _no_override_effective(hidr),
            )
        hydros = doc["hydros"]
        assert [h["id"] for h in hydros] == [0, 1, 2]

        up = hydros[0]
        assert up["name"] == "UP_RES"
        assert "bus_id" not in up
        assert up["unit_groups"][0]["bus_id"] == 0
        # Plant 1 → 3 (not operated, skipped) → 2 (operated).
        assert up["downstream_id"] == 1
        assert up["reservoir"] == {
            "min_storage_hm3": 100.0,
            "max_storage_hm3": 500.0,
        }
        assert up["outflow"]["min_outflow_m3s"] == 30.0
        # 2 machines × 50 MW, no derating: max_generation_mw stays rated.
        # max_turbined_m3s is head-corrected (ticket-017): this fixture
        # carries no queda_nominal_conjunto_1, so the affinity ratio is a
        # no-op (falls back to the rated 200 m³/s), but the installed-power
        # cap Σ n·p_nom / ρ_eq (= 100 / 0.72) still binds below it.
        assert up["generation"]["max_turbined_m3s"] == pytest.approx(100.0 / 0.72)
        assert up["generation"]["max_generation_mw"] == 100.0

        assert hydros[1]["downstream_id"] is None
        assert "coupling-only" in caplog.text
        assert "99" in caplog.text

    def test_unit_groups_present_and_mirror_generation(self, caplog) -> None:
        """Every hydro carries exactly one mirror unit group (cobre rule 41)
        and no top-level ``bus_id`` (removed field)."""
        hidr = _hidr_frame()
        with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.hydro"):
            doc = convert_hydros(
                _StubDadger(uh=_uh_frame()),
                hidr,
                _ID_MAP,
                date(2026, 7, 18),
                _no_override_effective(hidr),
            )
        for h in doc["hydros"]:
            assert "bus_id" not in h
            assert len(h["unit_groups"]) == 1
            group = h["unit_groups"][0]
            assert group.keys() == {
                "id",
                "name",
                "bus_id",
                "min_generation_mw",
                "max_generation_mw",
                "min_turbined_m3s",
                "max_turbined_m3s",
            }
            gen = h["generation"]
            assert group["min_generation_mw"] == pytest.approx(gen["min_generation_mw"])
            assert group["max_generation_mw"] == pytest.approx(gen["max_generation_mw"])
            assert group["min_turbined_m3s"] == pytest.approx(gen["min_turbined_m3s"])
            assert group["max_turbined_m3s"] == pytest.approx(gen["max_turbined_m3s"])

    def test_initial_storage_formula(self) -> None:
        hidr = _hidr_frame()
        storage = convert_initial_storage(
            _StubDadger(uh=_uh_frame()), hidr, _ID_MAP, _no_override_effective(hidr)
        )
        by_id = {e["hydro_id"]: e["value_hm3"] for e in storage}
        assert by_id[0] == pytest.approx(100.0 + 0.5 * 400.0)
        assert by_id[1] == pytest.approx(250.0)  # 100 %
        assert by_id[2] == pytest.approx(10.0)  # run-of-river, vmin == vmax

    def test_energy_productivity_head(self) -> None:
        hidr = _hidr_frame()
        table = convert_energy_productivity(
            _no_override_effective(hidr), _ID_MAP
        ).to_pandas()
        # Flat cota 100, tailrace 20 → head 80; ρ = 0.009 × 80.
        assert table["equivalent_productivity_mw_per_m3s"].iloc[0] == pytest.approx(
            0.009 * 80.0
        )
        assert table["stage_id"].isna().all()

    def test_production_models_constant(self) -> None:
        doc = convert_production_models(_ID_MAP)
        models = doc["production_models"]
        assert [m["hydro_id"] for m in models] == [0, 1, 2]
        assert all(
            m["stage_ranges"][0]["model"] == "constant_productivity" for m in models
        )


class TestItaipuBusRelabel:
    """ticket-006: the per-frequency split plant's 50 Hz unit group (group
    id 0, the ascending-frequency convention pinned by
    :func:`_split_plant_frequencies`) is unconditionally relocated to the
    ``IV`` transshipment bus; the 60 Hz group (group id 1) stays on the
    plant's own submercado bus. Pure/synthetic, deck-independent — no
    ``example/`` read.
    """

    def test_50hz_group_on_iv_60hz_group_on_se(self) -> None:
        hidr = _itaipu_hidr_frame()
        dadger = _StubDadger(
            uh=_itaipu_uh_frame(),
            mp=_itaipu_frequency_frame(),
            fd=_itaipu_frequency_frame(),
        )
        doc = convert_hydros(
            dadger,
            hidr,
            _ITAIPU_ID_MAP,
            date(2026, 7, 18),
            _no_override_effective(hidr),
        )
        itaipu = doc["hydros"][0]
        groups = {g["id"]: g for g in itaipu["unit_groups"]}
        assert groups.keys() == {0, 1}
        assert groups[0]["bus_id"] == _ITAIPU_ID_MAP.transhipment_bus_id  # 50 Hz -> IV
        assert groups[1]["bus_id"] == _ITAIPU_ID_MAP.bus_id(1)  # 60 Hz -> SE
        assert groups[0]["bus_id"] != groups[1]["bus_id"]

    def test_envelope_unchanged_by_the_relabel(self) -> None:
        """The relabel moves only the 50 Hz group's ``bus_id`` — the summed
        plant envelope is exactly what it was before the split existed: two
        identical conjuntos, each capped by its own installed power at
        ``100.0 / 0.72`` m3/s (no head data in this fixture, same head-free
        ratio ``UP_RES`` pins above), never the plant-wide power cap.
        """
        hidr = _itaipu_hidr_frame()
        dadger = _StubDadger(
            uh=_itaipu_uh_frame(),
            mp=_itaipu_frequency_frame(),
            fd=_itaipu_frequency_frame(),
        )
        doc = convert_hydros(
            dadger,
            hidr,
            _ITAIPU_ID_MAP,
            date(2026, 7, 18),
            _no_override_effective(hidr),
        )
        itaipu = doc["hydros"][0]
        gen = itaipu["generation"]
        groups = itaipu["unit_groups"]

        expected_generation = 2 * 100.0
        expected_turbined = 2 * (100.0 / 0.72)
        assert gen["max_generation_mw"] == pytest.approx(expected_generation)
        assert gen["max_turbined_m3s"] == pytest.approx(expected_turbined)
        assert sum(g["max_generation_mw"] for g in groups) == pytest.approx(
            gen["max_generation_mw"]
        )
        assert sum(g["max_turbined_m3s"] for g in groups) == pytest.approx(
            gen["max_turbined_m3s"]
        )

    def test_bus_count_mismatch_raises(self) -> None:
        """A mis-wired split (fewer/more per-group buses than frequencies)
        must fail loud, not silently colocate the groups."""
        hidr = _itaipu_hidr_frame()
        hreg = hidr.loc[66]
        with pytest.raises(
            ValueError,
            match="plant 66: 2 split frequencies but 1 per-group buses were supplied",
        ):
            _build_split_unit_groups(
                hreg, 66, "ITAIPU", [0], [50.0, 60.0], _no_override_effective(hidr)
            )


def test_deferred_note_excludes_head_productivity(caplog) -> None:
    """ticket-013 (AC6) + ticket-015 (E6): the head/productivity ``AC``
    family (``PROESP``/``PERHID``/``JUSMED``/``COTVOL``) has a live consumer
    (``_equivalent_productivity_mw_per_m3s``), so it was never listed as
    deferred. ticket-015 retires the blanket, hand-maintained
    "deferred hydro fidelity" warning entirely — deck-aware, per-family
    ``AC`` coverage now lives in ``check decomp`` — while the module
    docstring still documents the genuinely-deferred families (``VI``
    travel time, ``COTVAZ``/``COTARE``/``COFEVA``) and now points at
    ``check decomp`` for their per-deck coverage.
    """
    import cobre_bridge.decomp.hydro as hydro_module

    hidr = _hidr_frame()
    with caplog.at_level(logging.WARNING, logger="cobre_bridge.decomp.hydro"):
        convert_hydros(
            _StubDadger(uh=_uh_frame()),
            hidr,
            _ID_MAP,
            date(2026, 7, 18),
            _no_override_effective(hidr),
        )
    assert not any("deferred hydro fidelity" in r.message for r in caplog.records)

    docstring = hydro_module.__doc__
    assert docstring is not None
    for mnemonic in ("PROESP", "PERHID", "JUSMED", "COTVOL"):
        assert mnemonic not in docstring
    for marker in ("``VI``", "COTVAZ", "COTARE", "COFEVA"):
        assert marker in docstring
    assert "check decomp" in docstring


class TestEffectiveCadastroSourcing:
    """ticket-007: the entity ``reservoir`` block and the initial-storage
    start volume are re-sourced off the per-stage-effective cadastro layer.
    """

    def test_no_override_matches_the_pre_layer_base_registry_reads(self) -> None:
        """An ``EffectiveCadastro`` with no stage-varying volumes reduces to
        the base registry scalars everywhere, so the reservoir block and the
        initial storage volume equal the pre-ticket base-registry reads."""
        hidr = _temporal_hidr_frame()
        effective = _no_override_effective(hidr)
        uh = _temporal_uh_frame(volume_inicial=50.0)

        doc = convert_hydros(
            _StubDadger(uh=uh), hidr, _TEMPORAL_ID_MAP, date(2026, 7, 18), effective
        )
        assert doc["hydros"][0]["reservoir"] == {
            "min_storage_hm3": 20.0,
            "max_storage_hm3": 100.0,
        }

        storage = convert_initial_storage(
            _StubDadger(uh=uh), hidr, _TEMPORAL_ID_MAP, effective
        )
        assert storage[0]["value_hm3"] == pytest.approx(20.0 + 0.5 * (100.0 - 20.0))

    def test_reservoir_block_is_the_outer_envelope_not_the_stage_zero_value(
        self,
    ) -> None:
        """A temporal ``VOLMAX`` raise widens the entity ``reservoir`` block
        to the outer envelope, so per-stage bound rows (ticket-006) always
        sit inside it."""
        effective = _raised_envelope_effective()
        hidr = _temporal_hidr_frame()
        uh = _temporal_uh_frame(volume_inicial=50.0)

        doc = convert_hydros(
            _StubDadger(uh=uh), hidr, _TEMPORAL_ID_MAP, date(2026, 7, 18), effective
        )
        reservoir = doc["hydros"][0]["reservoir"]
        assert reservoir["max_storage_hm3"] == 250.0
        assert reservoir["min_storage_hm3"] == 20.0

    def test_initial_storage_uses_stage_zero_not_the_raised_envelope(self) -> None:
        """``volume_inicial`` is a percentage of the *initial stage's*
        useful volume — a later-stage ``VOLMAX`` raise must not leak into the
        start volume."""
        effective = _raised_envelope_effective()
        hidr = _temporal_hidr_frame()
        uh = _temporal_uh_frame(volume_inicial=50.0)

        storage = convert_initial_storage(
            _StubDadger(uh=uh), hidr, _TEMPORAL_ID_MAP, effective
        )
        assert storage[0]["value_hm3"] == pytest.approx(60.0)


def _ct_frame() -> pd.DataFrame:
    def row(code, name, estagio, cvu, disp, inflex):
        return {
            "codigo_submercado": 1,
            "codigo_usina": code,
            "estagio": estagio,
            "nome_usina": name,
            "cvu_1": cvu,
            "cvu_2": cvu,
            "cvu_3": cvu,
            "disponibilidade_1": disp[0],
            "disponibilidade_2": disp[1],
            "disponibilidade_3": disp[2],
            "inflexibilidade_1": inflex,
            "inflexibilidade_2": inflex,
            "inflexibilidade_3": inflex,
        }

    return pd.DataFrame(
        [
            # Plant 10: per-block spread on availability; declares stages 1 and 3.
            row(10, "SPREAD", 1, 100.0, (400.0, 300.0, 200.0), 0.0),
            row(10, "SPREAD", 3, 120.0, (400.0, 300.0, 200.0), 0.0),
            # Plant 20: flat, stage 1 only (stages 2-3 inherit).
            row(20, "FLAT", 1, 50.0, (640.0, 640.0, 640.0), 640.0),
        ]
    )


class TestConvertThermals:
    def test_registry_and_bounds(self) -> None:
        calendar = _calendar()
        doc = convert_thermals(
            _StubDadger(ct=_ct_frame()), _ID_MAP, calendar, date(2026, 7, 18)
        )
        thermals = doc["thermals"]
        assert [t["id"] for t in thermals] == [0, 1]
        spread_expected = (400.0 * 15 + 300.0 * 64 + 200.0 * 89) / 168.0
        assert thermals[0]["generation"]["max_mw"] == pytest.approx(spread_expected)
        assert thermals[1]["cost_per_mwh"] == pytest.approx(50.0)
        assert thermals[1]["generation"]["min_mw"] == pytest.approx(640.0)

        bounds = convert_thermal_bounds(_StubDadger(ct=_ct_frame()), _ID_MAP, calendar)
        generation = bounds.generation
        cost = bounds.cost.to_pandas()

        base = [c for c in generation if c.block_id is None]
        overrides = [c for c in generation if c.block_id is not None]
        # Plant 20 (FLAT, thermal_id=1) is block-uniform on every stage: one
        # base contribution per stage. Plant 10 (SPREAD, thermal_id=0) is
        # non-uniform on every stage: no base contribution at all, per-block
        # only — never both (the replace-vs-intersect discipline).
        assert len(base) == 1 * 3  # one base contribution per (FLAT, stage)
        assert {c.entity_id for c in base} == {1}

        flat = [c for c in base if c.entity_id == 1]
        assert {c.upper for c in flat} == {640.0}

        cost_by_cell = {
            (int(row.thermal_id), int(row.stage_id)): row.cost_per_mwh
            for row in cost.itertuples()
        }
        assert cost_by_cell[(0, 1)] == pytest.approx(100.0)  # inherited
        assert cost_by_cell[(0, 2)] == pytest.approx(120.0)

        # Plant 10's availability varies by block on every stage (its own
        # declarations at stage 1 and 3, plus stage 2 inheriting stage 1) —
        # one per-block contribution per (stage, block), inflexibilidade
        # flat at 0.0.
        spread_overrides = [c for c in overrides if c.entity_id == 0]
        assert len(spread_overrides) == 3 * 3
        assert {c.upper for c in spread_overrides} == {400.0, 300.0, 200.0}
        assert {c.lower for c in spread_overrides} == {0.0}

    def test_missing_stage_one_raises(self) -> None:
        ct = _ct_frame()
        ct.loc[ct["codigo_usina"] == 20, "estagio"] = 2
        with pytest.raises(ValueError, match="stage 1"):
            convert_thermals(
                _StubDadger(ct=ct), _ID_MAP, _calendar(), date(2026, 7, 18)
            )
