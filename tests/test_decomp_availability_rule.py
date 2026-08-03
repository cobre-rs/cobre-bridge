"""Acceptance test pinning the availability rule against the reference run.

The source model's per-stage available capacity is a product of two
registers, and the discovery report flagged the combination as the highest
silent-wrong-answer risk in the conversion: get it wrong and every plant
carries a plausible but false cap. The rule is pinned here against the
reference run's own reported availability rather than against prose.

Verdict (deck ``decomp-jul-26-rv3``, 2026-07-25)::

    available_MW(plant, stage) = Σ_g installed_g × MP(plant, g, stage)
                                              × FD(plant, g, stage)

with one implicit group for an ordinary plant. The generating-unit group
axis is not decoration: the only plant carrying per-group register rows
(keyed by frequency) is the one whose two halves are maintained
independently, and the plant-level product misses it by up to 672 MW while
the group-wise sum lands on the reported value exactly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.comparators.decomp_readers import read_dec_oper_usih
from cobre_bridge.decomp.group_bounds import convert_hydro_unit_group_bounds
from cobre_bridge.decomp.hydro import (
    convert_energy_productivity,
    convert_hydro_group_availability,
    convert_hydros,
    read_hidr,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import operative_calendar_from_dadger
from cobre_bridge.diagnostics import Severity
from cobre_bridge.emission_checks import check_group_bound_envelope

_DECK = Path("example/decomp-jul-26-rv3")
_needs_deck = pytest.mark.skipif(
    not (_DECK / "saidas" / "dec_oper_usih.csv").exists(),
    reason="reference deck outputs not present",
)

#: Tolerance in MW: the reference reports availability to the cent.
_TOL = 0.01

#: The plant whose maintenance and availability registers are declared per
#: generating-unit group, and the installed capacity of each group.
_SPLIT_PLANT = 66
_SPLIT_GROUPS_MW = (7000.0, 7000.0)


def _registers() -> tuple[dict, dict]:
    from idecomp.decomp import Dadger

    dadger = Dadger.read(str(_DECK / "dadger.rv3"))
    maintenance = dadger.mp(df=True)
    availability = dadger.fd(df=True)
    return maintenance, availability


def _reported() -> pl.DataFrame:
    stage_rows = read_dec_oper_usih(_DECK / "saidas").filter(
        pl.col("patamar").is_null()
    )
    return (
        stage_rows.group_by(["codigo_usina", "estagio"])
        .agg(
            pl.col("potencia_instalada_MW").mean().alias("installed"),
            pl.col("potencia_disponivel_MW").mean().alias("available"),
        )
        .sort(["codigo_usina", "estagio"])
    )


@_needs_deck
class TestAvailabilityRule:
    def test_product_of_both_registers_reproduces_single_group_plants(self) -> None:
        maintenance, availability = _registers()
        mp_by_plant = {
            int(r["codigo_usina"]): r
            for _, r in maintenance.drop_duplicates(
                "codigo_usina", keep=False
            ).iterrows()
        }
        fd_by_plant = {
            int(r["codigo_usina"]): r
            for _, r in availability.drop_duplicates(
                "codigo_usina", keep=False
            ).iterrows()
        }

        checked = 0
        for row in _reported().iter_rows(named=True):
            code, stage = int(row["codigo_usina"]), int(row["estagio"])
            if code == _SPLIT_PLANT:
                continue
            factor_m = (
                float(mp_by_plant[code][f"manutencao_{stage}"])
                if code in mp_by_plant
                else 1.0
            )
            factor_f = (
                float(fd_by_plant[code][f"fator_{stage}"])
                if code in fd_by_plant
                else 1.0
            )
            predicted = row["installed"] * factor_m * factor_f
            assert predicted == pytest.approx(row["available"], abs=_TOL), (
                f"plant {code} stage {stage}: predicted {predicted}, "
                f"reported {row['available']}"
            )
            checked += 1

        assert checked > 400, f"only {checked} (plant, stage) rows checked"

    def test_group_split_plant_needs_the_group_axis(self) -> None:
        """The split plant's availability is a per-group sum, not a plant product.

        Both halves carry their own maintenance and availability factors, so
        the plant-level product is wrong whenever the two differ — which is
        exactly what the generating-unit group representation exists for.
        """
        maintenance, availability = _registers()
        mp_rows = maintenance[maintenance["codigo_usina"] == _SPLIT_PLANT]
        fd_rows = availability[availability["codigo_usina"] == _SPLIT_PLANT]
        assert len(mp_rows) == len(_SPLIT_GROUPS_MW)
        assert set(mp_rows["frequencia"]) == {50.0, 60.0}

        reported = _reported().filter(pl.col("codigo_usina") == _SPLIT_PLANT)
        assert len(reported) > 0

        for row in reported.iter_rows(named=True):
            stage = int(row["estagio"])
            group_wise = 0.0
            for installed, (_, mp_row) in zip(
                _SPLIT_GROUPS_MW, mp_rows.iterrows(), strict=True
            ):
                frequency = mp_row["frequencia"]
                fd_row = fd_rows[fd_rows["frequencia"] == frequency].iloc[0]
                group_wise += (
                    installed
                    * float(mp_row[f"manutencao_{stage}"])
                    * float(fd_row[f"fator_{stage}"])
                )
            assert group_wise == pytest.approx(row["available"], abs=_TOL)

            plant_wise = (
                row["installed"]
                * float(mp_rows.iloc[0][f"manutencao_{stage}"])
                * float(fd_rows.iloc[0][f"fator_{stage}"])
            )
            if stage == 1:
                # Stage 1 is where the two halves differ most; the plant-level
                # product understates the fleet by the whole 60 Hz allowance.
                assert abs(plant_wise - row["available"]) > 100.0


# ---------------------------------------------------------------------------
# Converter-emission criteria (ticket-026, criteria 1-6)
#
# The classes above pin the *rule* against the reference registers without
# running any converter. These extend that to "we emit the rule": every
# assertion below reconstructs its expected value independently from the
# registers (never from decomp/hydro.py's own AC-adjustment or availability
# helpers) and grades the converter's actual output against it — including
# the AC NUMCON/NUMMAQ/POTEFE/VAZEFE machine-configuration overrides D-1's
# scope amendment (Option B, 2026-08-03) folded into the declared envelope,
# so all 504 single-group rows (not just the ~480 unaffected by an AC
# override) reproduce the oracle exactly off the hydraulic-binding rows.
# ---------------------------------------------------------------------------


def _load_deck():
    """Parse the deck once: dadger, registry, id map, and calendar."""
    from idecomp.decomp import Dadger

    dadger = Dadger.read(str(_DECK / "dadger.rv3"))
    hidr = read_hidr(_DECK / "hidr.dat")
    id_map = DecompIdMap.from_dadger(dadger)
    calendar = operative_calendar_from_dadger(dadger)
    return dadger, hidr, id_map, calendar


def _independent_ac_overrides(
    dadger,
) -> tuple[
    dict[int, int],
    dict[tuple[int, int], int],
    dict[tuple[int, int], float],
    dict[tuple[int, int], float],
]:
    """``(numero_conjuntos, numero_maquinas, potencia, vazao)`` parsed
    straight from the ``AC NUMCON``/``AC NUMMAQ``/``AC POTEFE``/``AC VAZEFE``
    registers — reimplemented here rather than imported from
    ``decomp/hydro.py``, matching the ticket's "reconstruct independently
    from the registers" discipline (as ``test_decomp_rq_bounds.py`` does for
    ``RQ``/``AC VAZMIN``).
    """
    from idecomp.decomp.modelos.dadger import ACNUMCON, ACNUMMAQ, ACPOTEFE, ACVAZEFE

    def _table(modificacao: type) -> pd.DataFrame | None:
        table = dadger.ac(codigo_usina=None, modificacao=modificacao, df=True)
        return table if isinstance(table, pd.DataFrame) and not table.empty else None

    numero_conjuntos: dict[int, int] = {}
    table = _table(ACNUMCON)
    if table is not None:
        for _, row in table.iterrows():
            numero_conjuntos[int(row["codigo_usina"])] = int(row["numero_conjuntos"])

    numero_maquinas: dict[tuple[int, int], int] = {}
    table = _table(ACNUMMAQ)
    if table is not None:
        for _, row in table.iterrows():
            key = (int(row["codigo_usina"]), int(row["indice_conjunto"]))
            numero_maquinas[key] = int(row["numero_maquinas"])

    potencia: dict[tuple[int, int], float] = {}
    table = _table(ACPOTEFE)
    if table is not None:
        for _, row in table.iterrows():
            key = (int(row["codigo_usina"]), int(row["indice_conjunto"]))
            potencia[key] = float(row["potencia"])

    vazao: dict[tuple[int, int], float] = {}
    table = _table(ACVAZEFE)
    if table is not None:
        for _, row in table.iterrows():
            key = (int(row["codigo_usina"]), int(row["indice_conjunto"]))
            vazao[key] = float(row["vazao"])

    return numero_conjuntos, numero_maquinas, potencia, vazao


def _independent_ac_adjusted_rated(
    code: int,
    hreg: pd.Series,
    numero_conjuntos: dict[int, int],
    numero_maquinas: dict[tuple[int, int], int],
    potencia: dict[tuple[int, int], float],
    vazao: dict[tuple[int, int], float],
) -> tuple[float, float]:
    """``(q_max, p_max)`` re-derived straight from the registry plus the AC
    overrides — independent of ``decomp/hydro.py::_compute_max_turbined_rated_ac_adjusted``.
    """
    n_sets = numero_conjuntos.get(code, int(hreg["numero_conjuntos_maquinas"]))
    q_max = 0.0
    p_max = 0.0
    for i in range(1, n_sets + 1):
        n_machines = numero_maquinas.get((code, i), int(hreg[f"maquinas_conjunto_{i}"]))
        q_nom = vazao.get((code, i), float(hreg[f"vazao_nominal_conjunto_{i}"]))
        p_nom = potencia.get((code, i), float(hreg[f"potencia_nominal_conjunto_{i}"]))
        q_max += n_machines * q_nom
        p_max += n_machines * p_nom
    return q_max, p_max


def _independent_single_group_rows(
    records: pd.DataFrame | None,
) -> dict[int, pd.Series]:
    """``{code: row}`` for an ``MP``/``FD`` register, single-group plants
    only — the same ``drop_duplicates(keep=False)`` idiom
    :class:`TestAvailabilityRule` already uses above.
    """
    if records is None or records.empty:
        return {}
    single = records.drop_duplicates("codigo_usina", keep=False)
    return {int(row["codigo_usina"]): row for _, row in single.iterrows()}


def _rho_by_hydro_id(hidr: pd.DataFrame, id_map: DecompIdMap) -> dict[int, float]:
    """ρ_eq per hydro id via :func:`convert_energy_productivity` — Requirement
    3 names this function's formula explicitly as the hydraulic cap's ρ_eq
    source, and it is independently pinned by
    ``test_decomp_hydro_thermal.py::test_energy_productivity_head`` already,
    so reusing it here (rather than re-deriving the cota-polynomial/
    hydraulic-loss integral a third time) still keeps the AC/availability
    logic under test independent.
    """
    table = convert_energy_productivity(hidr, id_map)
    return dict(
        zip(
            table["hydro_id"].to_pylist(),
            table["equivalent_productivity_mw_per_m3s"].to_pylist(),
            strict=True,
        )
    )


@_needs_deck
class TestConverterEmitsAvailability:
    """Criteria 1-5: grade the converter's actual emission."""

    def test_declared_envelope_is_ac_adjusted_installed(self) -> None:
        """Criterion 1 (+ D-1's scope amendment): every plant's declared
        entity envelope is the AC-adjusted rated capacity, not the old
        TEIF/IP-derated value. Covers all 169 plants, Itaipu included (D-1
        applies to the declared envelope uniformly). The single-group
        "mirror group matches the entity" check (rule 41 by construction)
        applies to every plant *except* Itaipu, whose two per-frequency
        groups instead sum to the entity (ticket-027,
        ``TestItaipuSplitGroups`` below covers that shape directly).
        """
        dadger, hidr, id_map, calendar = _load_deck()
        numero_conjuntos, numero_maquinas, potencia, vazao = _independent_ac_overrides(
            dadger
        )

        doc = convert_hydros(dadger, hidr, id_map, calendar[0].start_date)
        hydros_by_id = {h["id"]: h for h in doc["hydros"]}
        assert len(hydros_by_id) == len(id_map.hydro_codes)

        for code in id_map.hydro_codes:
            hreg = hidr.loc[code]
            q_expected, p_expected = _independent_ac_adjusted_rated(
                code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
            )
            hydro = hydros_by_id[id_map.hydro_id(code)]
            gen = hydro["generation"]
            assert gen["max_turbined_m3s"] == pytest.approx(q_expected, abs=1e-6)
            assert gen["max_generation_mw"] == pytest.approx(p_expected, abs=1e-6)

            if code == _SPLIT_PLANT:
                # Itaipu (ticket-027): two per-frequency groups, not a single
                # mirror — the group-wise *sum* (not the first group alone)
                # equals the entity; covered by TestItaipuSplitGroups.
                continue

            # Rule 41 holds by construction: the mirror group matches the
            # entity exactly.
            group = hydro["unit_groups"][0]
            assert group["max_turbined_m3s"] == pytest.approx(gen["max_turbined_m3s"])
            assert group["max_generation_mw"] == pytest.approx(gen["max_generation_mw"])

        # Itaipu specifically (Requirement 1's own example): 14000 MW, not
        # the pre-D-1 TEIF/IP-derated ~12236.94 MW. Itaipu has no AC override
        # on this deck, so its declared envelope is the plain registry rated
        # sum once TEIF/IP stops derating it.
        itaipu = hydros_by_id[id_map.hydro_id(_SPLIT_PLANT)]
        assert itaipu["generation"]["max_generation_mw"] == pytest.approx(
            14000.0, abs=1e-6
        )
        assert itaipu["generation"]["max_turbined_m3s"] == pytest.approx(
            13240.0, abs=1e-6
        )

    def test_overlay_reconstruction_matches_independent_registers(self) -> None:
        """Criterion 2: the effective per-stage ``max_generation_mw`` (the
        overlay row if present, else the declared envelope) equals
        ``min(hydraulic, installed×MP×FD)`` recomputed independently from
        the registers — for every one of the 504 single-group (code != 66)
        rows, not a sample. Also pins the binding-row set (criterion 4's own
        obligation) by cross-checking it against the converter's own delta
        list.
        """
        dadger, hidr, id_map, calendar = _load_deck()
        numero_conjuntos, numero_maquinas, potencia, vazao = _independent_ac_overrides(
            dadger
        )
        mp_by_code = _independent_single_group_rows(dadger.mp(df=True))
        fd_by_code = _independent_single_group_rows(dadger.fd(df=True))
        rho_by_hydro_id = _rho_by_hydro_id(hidr, id_map)

        values, deltas = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar
        )
        actual_binding = {(d.code, d.stage_id) for d in deltas}

        expected_binding: set[tuple[int, int]] = set()
        checked = 0
        for code in id_map.hydro_codes:
            if code == _SPLIT_PLANT:
                continue
            hreg = hidr.loc[code]
            hydro_id = id_map.hydro_id(code)
            q_max, p_max = _independent_ac_adjusted_rated(
                code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
            )
            hydraulic_mw = rho_by_hydro_id[hydro_id] * q_max
            mp_row = mp_by_code.get(code)
            fd_row = fd_by_code.get(code)

            for stage in calendar:
                mp_factor = (
                    1.0
                    if mp_row is None
                    else float(mp_row[f"manutencao_{stage.index + 1}"])
                )
                fd_factor = (
                    1.0 if fd_row is None else float(fd_row[f"fator_{stage.index + 1}"])
                )
                availability_mw = p_max * mp_factor * fd_factor
                expected_emitted = min(hydraulic_mw, availability_mw)

                key = (hydro_id, 0, stage.index)
                effective = values[key].max_generation_mw if key in values else p_max
                assert effective == pytest.approx(expected_emitted, abs=1e-6), (
                    f"plant {code} stage {stage.index}: converter emitted "
                    f"{effective}, independent reconstruction {expected_emitted}"
                )

                if hydraulic_mw < availability_mw:
                    expected_binding.add((code, stage.index))
                checked += 1

        assert checked == 504, f"expected 504 single-group rows, got {checked}"
        # Criterion 4: pinned — measured here (not assumed), 2026-08-03.
        assert len(expected_binding) == 110
        assert actual_binding == expected_binding

    def test_non_binding_rows_match_oracle_binding_rows_stay_below(self) -> None:
        """Criteria 3 and 4 against the ``dec_oper_usih`` oracle directly (as
        opposed to the previous test's purely internal reconstruction): every
        single-group row where the hydraulic ceiling does *not* bind
        reproduces ``potencia_disponivel_MW`` to < 0.01 MW — now on all 504
        rows, the AC join having closed the 24-row gap a bare
        ``_compute_max_turbined_rated`` would have left. Every binding row
        stays at or below it (the accepted, B8-sanctioned cost).
        """
        dadger, hidr, id_map, calendar = _load_deck()
        numero_conjuntos, numero_maquinas, potencia, vazao = _independent_ac_overrides(
            dadger
        )
        mp_by_code = _independent_single_group_rows(dadger.mp(df=True))
        fd_by_code = _independent_single_group_rows(dadger.fd(df=True))
        rho_by_hydro_id = _rho_by_hydro_id(hidr, id_map)

        checked_nonbinding = 0
        checked_binding = 0
        for row in _reported().iter_rows(named=True):
            code = int(row["codigo_usina"])
            stage_number = int(row["estagio"])
            if code == _SPLIT_PLANT:
                continue
            hreg = hidr.loc[code]
            hydro_id = id_map.hydro_id(code)
            q_max, p_max = _independent_ac_adjusted_rated(
                code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
            )
            hydraulic_mw = rho_by_hydro_id[hydro_id] * q_max
            mp_row = mp_by_code.get(code)
            fd_row = fd_by_code.get(code)
            mp_factor = (
                1.0 if mp_row is None else float(mp_row[f"manutencao_{stage_number}"])
            )
            fd_factor = (
                1.0 if fd_row is None else float(fd_row[f"fator_{stage_number}"])
            )
            availability_mw = p_max * mp_factor * fd_factor
            emitted = min(hydraulic_mw, availability_mw)

            if hydraulic_mw >= availability_mw:
                assert emitted == pytest.approx(row["available"], abs=_TOL), (
                    f"plant {code} stage {stage_number}: emitted {emitted}, "
                    f"oracle {row['available']}"
                )
                checked_nonbinding += 1
            else:
                assert emitted <= row["available"] + _TOL
                checked_binding += 1

        assert checked_nonbinding == 504 - 110
        assert checked_binding == 110

    def test_pipeline_emits_the_availability_delta_diagnostic(
        self, tmp_path: Path
    ) -> None:
        """Criterion 4's Diagnostic half: the pipeline reports the same
        binding-row count via one INFO Diagnostic, ten worst offenders."""
        from cobre_bridge.decomp.pipeline import convert_decomp_case

        with dx.collect() as diagnostics:
            convert_decomp_case(_DECK, tmp_path / "out", force=True)

        found = [
            d for d in diagnostics if d.code == "hydro-availability-hydraulic-cap-binds"
        ]
        assert len(found) == 1
        diagnostic = found[0]
        assert diagnostic.severity is Severity.INFO
        assert "110" in diagnostic.summary
        assert diagnostic.table is not None
        assert len(diagnostic.table.rows) == 10

    def test_static_over_allow_defect_is_closed_for_code_42(self) -> None:
        """Criterion 5: code 42 (``MP = 0.333`` at stages 1-2, 0.109 for code
        40 stage 2 is the more extreme sibling example) used to declare a
        stage-invariant static cap — the pre-ticket ``_compute_max_turbined_simple``
        value, ≈ 332.9 MW. The overlay now reduces stages 1-2's
        ``max_generation_mw`` to the maintenance-derated value ≈ 115.7 MW.
        """
        dadger, hidr, id_map, calendar = _load_deck()
        values, _ = convert_hydro_group_availability(dadger, hidr, id_map, calendar)
        hydro_id = id_map.hydro_id(42)

        for stage_index in (0, 1):
            key = (hydro_id, 0, stage_index)
            assert key in values, f"expected an overlay row at stage {stage_index}"
            assert values[key].max_generation_mw == pytest.approx(115.68, abs=0.01)

    def test_rule_45_mirror_raises_nothing_on_the_real_emission(self) -> None:
        """Criterion 6's rule-45 half, run directly against the real
        deck's emitted ``hydros.json`` + overlay (the end-to-end ``cobre
        validate`` pass, recorded in ``epic-08/learnings.md``, covers the
        rest)."""
        dadger, hidr, id_map, calendar = _load_deck()
        doc = convert_hydros(dadger, hidr, id_map, calendar[0].start_date)
        values, _ = convert_hydro_group_availability(dadger, hidr, id_map, calendar)
        group_bounds = convert_hydro_unit_group_bounds(values, calendar)

        with dx.collect() as collected:
            check_group_bound_envelope(doc, group_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert not errors, [d.summary for d in errors]


# ---------------------------------------------------------------------------
# Itaipu's own two-group emission (ticket-027, criteria 1-4)
#
# Itaipu (code 66) is the one plant excluded from TestConverterEmitsAvailability
# above (its MP/FD registers carry two per-frequency rows, dropped by that
# class's own single-group selection). This class grades the converter's
# *split*-plant emission the same way: every expected value is reconstructed
# independently from the registers, never read back from the converter's own
# intermediate — mirroring test_group_split_plant_needs_the_group_axis.
# ---------------------------------------------------------------------------


def _itaipu_conjunto_rated(hidr: pd.DataFrame) -> list[tuple[float, float]]:
    """``[(q, p), ...]`` for Itaipu's two hidr conjuntos, in registry
    (1-based) order — independent of ``decomp/hydro.py::_conjunto_rated_ac_adjusted``
    (Itaipu carries no AC override on either reference deck, so a bare
    registry read is the correct independent reconstruction here).
    """
    hreg = hidr.loc[_SPLIT_PLANT]
    return [
        (
            float(hreg[f"vazao_nominal_conjunto_{i}"])
            * float(hreg[f"maquinas_conjunto_{i}"]),
            float(hreg[f"potencia_nominal_conjunto_{i}"])
            * float(hreg[f"maquinas_conjunto_{i}"]),
        )
        for i in (1, 2)
    ]


@_needs_deck
class TestItaipuSplitGroups:
    """Criteria 1-4: Itaipu's own two conjunto-backed groups + their overlay."""

    def test_registry_shape_matches_the_ticket(self) -> None:
        """Precondition pinned loudly: 2 identical conjuntos of 10 machines
        x 662 m3/s x 700 MW, submercado SE, and both MP/FD carry exactly the
        2 frequencia rows (50, 60) this whole ticket is built on. A registry
        drift here is a STOP-and-escalate condition, not silently absorbed.
        """
        _dadger, hidr, id_map, _calendar = _load_deck()
        hreg = hidr.loc[_SPLIT_PLANT]
        assert int(hreg["numero_conjuntos_maquinas"]) == 2
        assert int(hreg["submercado"]) == 1
        for i in (1, 2):
            assert int(hreg[f"maquinas_conjunto_{i}"]) == 10
            assert float(hreg[f"vazao_nominal_conjunto_{i}"]) == pytest.approx(662.0)
            assert float(hreg[f"potencia_nominal_conjunto_{i}"]) == pytest.approx(700.0)
        assert id_map.bus_id(int(hreg["submercado"])) == id_map.bus_id(1)

        maintenance, availability = _registers()
        mp_rows = maintenance[maintenance["codigo_usina"] == _SPLIT_PLANT]
        fd_rows = availability[availability["codigo_usina"] == _SPLIT_PLANT]
        assert len(mp_rows) == 2
        assert len(fd_rows) == 2
        assert set(mp_rows["frequencia"]) == {50.0, 60.0}
        assert set(fd_rows["frequencia"]) == {50.0, 60.0}

    def test_two_declared_groups_sum_to_the_installed_plant_envelope(self) -> None:
        """Criterion 1: two unique-id groups (id 0 = 50 Hz, id 1 = 60 Hz),
        each 7000 MW / 6620 m3/s, both on bus SE; the plant's own envelope
        stays installed (14000/13240), so rule 41 holds exactly by
        construction (7000+7000 == 14000, 6620+6620 == 13240).
        """
        dadger, hidr, id_map, calendar = _load_deck()
        doc = convert_hydros(dadger, hidr, id_map, calendar[0].start_date)
        hydro_id = id_map.hydro_id(_SPLIT_PLANT)
        itaipu = next(h for h in doc["hydros"] if h["id"] == hydro_id)
        assert itaipu["name"] == "ITAIPU"

        groups = itaipu["unit_groups"]
        assert len(groups) == 2
        assert {g["id"] for g in groups} == {0, 1}

        conjuntos = _itaipu_conjunto_rated(hidr)
        expected_bus = id_map.bus_id(1)
        for group in sorted(groups, key=lambda g: g["id"]):
            q_expected, p_expected = conjuntos[group["id"]]
            assert group["max_generation_mw"] == pytest.approx(p_expected, abs=1e-6)
            assert group["max_generation_mw"] == pytest.approx(7000.0, abs=1e-6)
            assert group["max_turbined_m3s"] == pytest.approx(q_expected, abs=1e-6)
            assert group["max_turbined_m3s"] == pytest.approx(6620.0, abs=1e-6)
            assert group["min_generation_mw"] == 0.0
            assert group["min_turbined_m3s"] == 0.0
            assert group["bus_id"] == expected_bus

        gen = itaipu["generation"]
        assert gen["max_generation_mw"] == pytest.approx(14000.0, abs=1e-6)
        assert gen["max_turbined_m3s"] == pytest.approx(13240.0, abs=1e-6)
        assert sum(g["max_generation_mw"] for g in groups) == pytest.approx(
            gen["max_generation_mw"]
        )
        assert sum(g["max_turbined_m3s"] for g in groups) == pytest.approx(
            gen["max_turbined_m3s"]
        )

    def test_availability_rule_is_exact_vs_oracle_group_wise(self) -> None:
        """Criterion 2: Σ_g 7000 x MP_g(stage) x FD_g(stage) — id 0 =
        frequencia 50, id 1 = frequencia 60, sorted ascending — equals the
        dec_oper_usih oracle to < 0.01 MW at every stage. Pure availability
        rule, no hydraulic cap: this is the group-axis proof, independent of
        the converter's own emission (graded separately below)."""
        maintenance, availability = _registers()
        mp_rows = maintenance[maintenance["codigo_usina"] == _SPLIT_PLANT].sort_values(
            "frequencia"
        )
        fd_rows = availability[
            availability["codigo_usina"] == _SPLIT_PLANT
        ].sort_values("frequencia")
        assert list(mp_rows["frequencia"]) == [50.0, 60.0]
        assert list(fd_rows["frequencia"]) == [50.0, 60.0]

        reported = (
            _reported().filter(pl.col("codigo_usina") == _SPLIT_PLANT).sort("estagio")
        )
        assert len(reported) == 3
        expected_oracle = {1: 13328.0, 2: 13937.0, 3: 12236.0}

        for row in reported.iter_rows(named=True):
            stage = int(row["estagio"])
            group_wise = sum(
                7000.0
                * float(mp_row[f"manutencao_{stage}"])
                * float(fd_row[f"fator_{stage}"])
                for (_, mp_row), (_, fd_row) in zip(
                    mp_rows.iterrows(), fd_rows.iterrows(), strict=True
                )
            )
            assert group_wise == pytest.approx(row["available"], abs=_TOL)
            assert group_wise == pytest.approx(expected_oracle[stage], abs=_TOL)

    def test_overlay_reconstruction_and_binding_delta(self) -> None:
        """Criteria 3 and 4 together (both reconstructed independently from
        the registers, ρ_eq re-derived the same way
        ``convert_energy_productivity`` does — never hardcoded):

        - Criterion 3: each emitted ``(group, stage)`` ``max_generation_mw``
          equals ``min(ρ_eq x 6620, 7000 x MP_g x FD_g)`` to < 0.01 MW, with
          ``hydro_unit_group_id`` in {0, 1}, ``block_id`` null, every value
          <= 7000 (rule 45 holds per group).
        - Criterion 4: the hydraulic-binding set is exactly
          {(group1, stage1), (group0, stage2), (group1, stage2)} (0-based:
          {(1, 0), (0, 1), (1, 1)}); the emitted group sum is below the
          oracle at stages 1-2 (delta > 0, ~= 189 / 315 MW) and exact at
          stage 3 (< 0.01 MW) — the measured B8 accepted cost, pinned here,
          not asserted equal to the oracle.
        """
        dadger, hidr, id_map, calendar = _load_deck()
        hydro_id = id_map.hydro_id(_SPLIT_PLANT)
        rho_by_hydro_id = _rho_by_hydro_id(hidr, id_map)
        rho_eq = rho_by_hydro_id[hydro_id]

        conjuntos = _itaipu_conjunto_rated(hidr)
        maintenance, availability = _registers()
        mp_rows = list(
            maintenance[maintenance["codigo_usina"] == _SPLIT_PLANT]
            .sort_values("frequencia")
            .iterrows()
        )
        fd_rows = list(
            availability[availability["codigo_usina"] == _SPLIT_PLANT]
            .sort_values("frequencia")
            .iterrows()
        )

        values, _ = convert_hydro_group_availability(dadger, hidr, id_map, calendar)
        group_bounds = convert_hydro_unit_group_bounds(values, calendar)
        bounds_pd = group_bounds.to_pandas()
        itaipu_bounds = bounds_pd[bounds_pd["hydro_id"] == hydro_id]
        assert set(itaipu_bounds["hydro_unit_group_id"]) == {0, 1}
        assert itaipu_bounds["block_id"].isna().all()

        binding: set[tuple[int, int]] = set()
        emitted_sum = {0: 0.0, 1: 0.0, 2: 0.0}
        for group_id in (0, 1):
            q_g, p_g = conjuntos[group_id]
            assert p_g == pytest.approx(7000.0)
            hydraulic_mw = rho_eq * q_g
            _, mp_row = mp_rows[group_id]
            _, fd_row = fd_rows[group_id]

            for stage_index in range(3):
                stage_number = stage_index + 1
                availability_mw = (
                    p_g
                    * float(mp_row[f"manutencao_{stage_number}"])
                    * float(fd_row[f"fator_{stage_number}"])
                )
                expected_emitted = min(hydraulic_mw, availability_mw)

                key = (hydro_id, group_id, stage_index)
                assert key in values, (
                    f"expected an overlay row at group {group_id} stage {stage_index}"
                )
                emitted = values[key].max_generation_mw
                assert emitted == pytest.approx(expected_emitted, abs=1e-6)
                assert emitted <= 7000.0 + 1e-6

                emitted_sum[stage_index] += emitted
                if hydraulic_mw < availability_mw:
                    binding.add((group_id, stage_index))

        # Criterion 4: pinned — measured, not assumed.
        assert binding == {(1, 0), (0, 1), (1, 1)}

        oracle = {0: 13328.0, 1: 13937.0, 2: 12236.0}
        delta = {s: oracle[s] - emitted_sum[s] for s in oracle}
        assert delta[0] == pytest.approx(189.20, abs=0.01)
        assert delta[1] == pytest.approx(315.40, abs=0.01)
        assert delta[2] == pytest.approx(0.0, abs=0.01)
        assert delta[0] > 0.0
        assert delta[1] > 0.0
        assert emitted_sum[0] == pytest.approx(13138.80, abs=0.01)
        assert emitted_sum[1] == pytest.approx(13621.60, abs=0.01)
        assert emitted_sum[2] == pytest.approx(12236.0, abs=0.01)

    def test_rule_45_mirror_raises_nothing_on_itaipus_own_emission(self) -> None:
        """The rule-45 mirror, run directly against Itaipu's own declared
        groups and overlay — every emitted value sits at or below
        ρ_eq x 6620 (~6810.80), itself below each group's declared 7000."""
        dadger, hidr, id_map, calendar = _load_deck()
        doc = convert_hydros(dadger, hidr, id_map, calendar[0].start_date)
        values, _ = convert_hydro_group_availability(dadger, hidr, id_map, calendar)
        group_bounds = convert_hydro_unit_group_bounds(values, calendar)

        with dx.collect() as collected:
            check_group_bound_envelope(doc, group_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert not errors, [d.summary for d in errors]
