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
from typing import TYPE_CHECKING

import pandas as pd
import polars as pl
import pytest

from cobre_bridge import diagnostics as dx
from cobre_bridge.comparators.decomp_readers import read_dec_oper_usih
from cobre_bridge.converters.hydro import _KTURB_BY_TIPO_TURBINA
from cobre_bridge.decomp.cadastro import build_effective_cadastro
from cobre_bridge.decomp.group_bounds import convert_hydro_unit_group_bounds
from cobre_bridge.decomp.hydro import (
    _equivalent_productivity_mw_per_m3s,
    convert_hydro_group_availability,
    convert_hydros,
    read_hidr,
)
from cobre_bridge.decomp.id_map import DecompIdMap
from cobre_bridge.decomp.temporal import operative_calendar_from_dadger
from cobre_bridge.diagnostics import Severity
from cobre_bridge.emission_checks import check_group_bound_envelope

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cobre_bridge.decomp.cadastro import EffectiveCadastro
    from cobre_bridge.decomp.temporal import OperativeStage

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
    """Parse the deck once: dadger, registry, id map, calendar, and the
    machine-set-populated effective cadastro (ticket-012: the sole machine-
    set source for both ``convert_hydros`` and
    ``convert_hydro_group_availability`` now)."""
    from idecomp.decomp import Dadger

    dadger = Dadger.read(str(_DECK / "dadger.rv3"))
    hidr = read_hidr(_DECK / "hidr.dat")
    id_map = DecompIdMap.from_dadger(dadger)
    calendar = operative_calendar_from_dadger(dadger)
    effective, _ = build_effective_cadastro(dadger, hidr, calendar)
    return dadger, hidr, id_map, calendar, effective


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


def _independent_head_corrected_max_turbined(
    code: int,
    hidr: pd.DataFrame,
    effective: EffectiveCadastro,
    calendar: Sequence[OperativeStage],
    rho_by_hydro_id: dict[tuple[int, int], float],
    hydro_id: int,
    numero_conjuntos: dict[int, int],
    numero_maquinas: dict[tuple[int, int], int],
    potencia: dict[tuple[int, int], float],
    vazao: dict[tuple[int, int], float],
) -> float:
    """Max-over-stages head-corrected ``max_turbined_m3s`` for *code*,
    reconstructed independently of ``decomp/hydro.py``'s own head-corrected
    helpers (ticket-017): reuses only ``rho_by_hydro_id`` (itself the shared
    ``_equivalent_productivity_mw_per_m3s`` formula, already treated as an
    independent building block above) and the raw AC NUMCON/NUMMAQ/POTEFE/
    VAZEFE overrides already parsed into *numero_conjuntos*/*numero_maquinas*/
    *potencia*/*vazao* — the affinity-ratio/power-cap arithmetic itself is
    redone here, not called from ``decomp/hydro.py``. ``AC ALTEFE`` is not
    consumed by either side (ticket-017's tracked gap), so ``h_nom``/
    ``tipo_turbina`` are always read straight off the base ``hidr`` row.
    """
    hreg = hidr.loc[code]
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    kturb = _KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    n_sets = numero_conjuntos.get(code, int(hreg["numero_conjuntos_maquinas"]))

    per_stage: list[float] = []
    for stage in calendar:
        rho_eq = rho_by_hydro_id[(hydro_id, stage.index)]
        rho_esp = effective.value(code, "produtibilidade_especifica", stage.index)
        h_op = rho_eq / rho_esp if rho_eq > 0.0 and rho_esp > 0.0 else 0.0

        if h_op <= 0.0:
            per_stage.append(
                _independent_ac_adjusted_rated(
                    code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
                )[0]
            )
            continue

        sum_affinity = 0.0
        sum_p = 0.0
        for i in range(1, n_sets + 1):
            n_machines = numero_maquinas.get(
                (code, i), int(hreg[f"maquinas_conjunto_{i}"])
            )
            q_nom = vazao.get((code, i), float(hreg[f"vazao_nominal_conjunto_{i}"]))
            p_nom = potencia.get(
                (code, i), float(hreg[f"potencia_nominal_conjunto_{i}"])
            )
            n_q = n_machines * q_nom
            n_p = n_machines * p_nom
            h_nom = float(hreg[f"queda_nominal_conjunto_{i}"])
            sum_affinity += n_q if h_nom <= 0.0 else n_q * (h_op / h_nom) ** kturb
            sum_p += n_p
        per_stage.append(min(sum_affinity, sum_p / rho_eq))

    return max(per_stage)


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


def _rho_by_hydro_id(
    effective: EffectiveCadastro,
    id_map: DecompIdMap,
    calendar: Sequence[OperativeStage],
) -> dict[tuple[int, int], float]:
    """ρ_eq per ``(hydro_id, stage_index)`` via
    ``decomp.hydro._equivalent_productivity_mw_per_m3s`` — ticket-013's
    Requirement 3 names this function's formula explicitly as the hydraulic
    cap's ρ_eq source, and its stage-0 value is independently pinned by
    ``test_decomp_hydro_thermal.py::test_energy_productivity_head`` already.
    ``convert_energy_productivity`` itself only exposes that stage-0 value
    (one ρ_eq per plant, penalties' own input); the availability overlay's
    hydraulic ceiling instead needs ρ_eq at every stage, so this calls the
    same per-stage production formula directly (rather than re-deriving the
    cota-polynomial/hydraulic-loss integral a third time), which still keeps
    the AC/availability logic under test independent.
    """
    return {
        (id_map.hydro_id(code), stage.index): _equivalent_productivity_mw_per_m3s(
            effective, code, stage.index
        )
        for code in id_map.hydro_codes
        for stage in calendar
    }


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
        dadger, hidr, id_map, calendar, effective = _load_deck()
        numero_conjuntos, numero_maquinas, potencia, vazao = _independent_ac_overrides(
            dadger
        )
        rho_by_hydro_id = _rho_by_hydro_id(effective, id_map, calendar)

        doc = convert_hydros(
            dadger,
            hidr,
            id_map,
            calendar[0].start_date,
            effective,
        )
        hydros_by_id = {h["id"]: h for h in doc["hydros"]}
        assert len(hydros_by_id) == len(id_map.hydro_codes)

        for code in id_map.hydro_codes:
            hreg = hidr.loc[code]
            hydro_id = id_map.hydro_id(code)
            _, p_expected = _independent_ac_adjusted_rated(
                code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
            )
            # max_turbined_m3s is head-corrected (ticket-017); Itaipu's own
            # per-conjunto-capped entity sum happens to equal this
            # plant-wide-cap formula here because its two conjuntos are
            # identical (min is homogeneous: sum of two equal per-conjunto
            # mins == min of the doubled sums) — verified below regardless.
            q_expected = _independent_head_corrected_max_turbined(
                code,
                hidr,
                effective,
                calendar,
                rho_by_hydro_id,
                hydro_id,
                numero_conjuntos,
                numero_maquinas,
                potencia,
                vazao,
            )
            hydro = hydros_by_id[hydro_id]
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
        # sum once TEIF/IP stops derating it. max_turbined_m3s is
        # head-corrected (ticket-017): 13240 rated derates to ~13060.65 at
        # Itaipu's own nominal (117 m) vs operating head (both conjuntos
        # symmetric, TestItaipuSplitGroups below covers the per-group shape).
        itaipu = hydros_by_id[id_map.hydro_id(_SPLIT_PLANT)]
        assert itaipu["generation"]["max_generation_mw"] == pytest.approx(
            14000.0, abs=1e-6
        )
        assert itaipu["generation"]["max_turbined_m3s"] == pytest.approx(
            13060.65, abs=0.01
        )

    def test_overlay_reconstruction_matches_independent_registers(self) -> None:
        """Criterion 2: the effective per-stage ``max_generation_mw`` (the
        overlay row if present, else the declared envelope) equals the
        ``installed×MP×FD`` availability recomputed independently from the
        registers — for every one of the 504 single-group (code != 66) rows,
        not a sample. The head-derived hydraulic ceiling no longer caps
        generation (the head-corrected turbined-flow cap owns the physical
        limit), so the emitted generation cap is the availability alone.
        """
        dadger, hidr, id_map, calendar, effective = _load_deck()
        numero_conjuntos, numero_maquinas, potencia, vazao = _independent_ac_overrides(
            dadger
        )
        mp_by_code = _independent_single_group_rows(dadger.mp(df=True))
        fd_by_code = _independent_single_group_rows(dadger.fd(df=True))

        values = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar, effective
        )

        checked = 0
        for code in id_map.hydro_codes:
            if code == _SPLIT_PLANT:
                continue
            hreg = hidr.loc[code]
            hydro_id = id_map.hydro_id(code)
            _, p_max = _independent_ac_adjusted_rated(
                code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
            )
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

                key = (hydro_id, 0, stage.index)
                emitted = values[key].max_generation_mw if key in values else None
                effective = p_max if emitted is None else emitted
                assert effective == pytest.approx(availability_mw, abs=1e-6), (
                    f"plant {code} stage {stage.index}: converter emitted "
                    f"{effective}, independent reconstruction {availability_mw}"
                )
                checked += 1

        assert checked == 504, f"expected 504 single-group rows, got {checked}"

    def test_every_row_reproduces_the_oracle(self) -> None:
        """Criterion 3 against the ``dec_oper_usih`` oracle directly (as
        opposed to the previous test's purely internal reconstruction): every
        single-group row's effective ``max_generation_mw`` reproduces the
        source model's own reported ``potencia_disponivel_MW`` to < 0.01 MW —
        on all 504 rows. The generation cap is now exactly the availability
        (installed×MP×FD), which is what the oracle reports, so there is no
        longer a hydraulic-binding subset sitting below it.
        """
        dadger, hidr, id_map, calendar, effective = _load_deck()
        numero_conjuntos, numero_maquinas, potencia, vazao = _independent_ac_overrides(
            dadger
        )

        values = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar, effective
        )

        checked = 0
        for row in _reported().iter_rows(named=True):
            code = int(row["codigo_usina"])
            stage_number = int(row["estagio"])
            if code == _SPLIT_PLANT:
                continue
            hreg = hidr.loc[code]
            hydro_id = id_map.hydro_id(code)
            _, p_max = _independent_ac_adjusted_rated(
                code, hreg, numero_conjuntos, numero_maquinas, potencia, vazao
            )
            key = (hydro_id, 0, stage_number - 1)
            emitted = values[key].max_generation_mw if key in values else None
            effective = p_max if emitted is None else emitted
            assert effective == pytest.approx(row["available"], abs=_TOL), (
                f"plant {code} stage {stage_number}: emitted {effective}, "
                f"oracle {row['available']}"
            )
            checked += 1

        assert checked == 504

    def test_static_over_allow_defect_is_closed_for_code_42(self) -> None:
        """Criterion 5: code 42 (``MP = 0.333`` at stages 1-2, 0.109 for code
        40 stage 2 is the more extreme sibling example) used to declare a
        stage-invariant static cap — the pre-ticket ``_compute_max_turbined_simple``
        value, ≈ 332.9 MW. The overlay now reduces stages 1-2's
        ``max_generation_mw`` to the maintenance-derated value ≈ 115.7 MW.
        """
        dadger, hidr, id_map, calendar, effective = _load_deck()
        values = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar, effective
        )
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
        dadger, hidr, id_map, calendar, effective = _load_deck()
        doc = convert_hydros(
            dadger,
            hidr,
            id_map,
            calendar[0].start_date,
            effective,
        )
        values = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar, effective
        )
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


def _itaipu_conjunto_head_corrected(
    hidr: pd.DataFrame,
    effective: EffectiveCadastro,
    calendar: Sequence[OperativeStage],
    rho_by_hydro_id: dict[tuple[int, int], float],
    hydro_id: int,
) -> list[float]:
    """Max-over-stages head-corrected flow for each of Itaipu's two hidr
    conjuntos (ticket-017), independent of ``decomp/hydro.py::
    _conjunto_head_corrected_ac_adjusted`` — Itaipu carries no AC override
    on either reference deck, so :func:`_itaipu_conjunto_rated`'s bare
    registry pair plus a bare ``queda_nominal_conjunto_i``/``tipo_turbina``
    read is the correct independent reconstruction here, with the
    per-conjunto affinity/power-cap arithmetic redone rather than called.
    """
    hreg = hidr.loc[_SPLIT_PLANT]
    tipo_turbina = int(hreg.get("tipo_turbina", 0) or 0)
    kturb = _KTURB_BY_TIPO_TURBINA.get(tipo_turbina, 0.5)
    rated = _itaipu_conjunto_rated(hidr)

    result: list[float] = []
    for i, (n_q, n_p) in zip((1, 2), rated, strict=True):
        h_nom = float(hreg[f"queda_nominal_conjunto_{i}"])
        per_stage: list[float] = []
        for stage in calendar:
            rho_eq = rho_by_hydro_id[(hydro_id, stage.index)]
            rho_esp = effective.value(
                _SPLIT_PLANT, "produtibilidade_especifica", stage.index
            )
            h_op = rho_eq / rho_esp if rho_eq > 0.0 and rho_esp > 0.0 else 0.0
            if h_op <= 0.0:
                per_stage.append(n_q)
                continue
            affinity = n_q if h_nom <= 0.0 else n_q * (h_op / h_nom) ** kturb
            per_stage.append(min(affinity, n_p / rho_eq))
        result.append(max(per_stage))
    return result


@_needs_deck
class TestItaipuSplitGroups:
    """Criteria 1-4: Itaipu's own two conjunto-backed groups + their overlay."""

    def test_registry_shape_matches_the_ticket(self) -> None:
        """Precondition pinned loudly: 2 identical conjuntos of 10 machines
        x 662 m3/s x 700 MW, submercado SE, and both MP/FD carry exactly the
        2 frequencia rows (50, 60) this whole ticket is built on. A registry
        drift here is a STOP-and-escalate condition, not silently absorbed.
        """
        _dadger, hidr, id_map, _calendar, _effective = _load_deck()
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
        each 7000 MW installed (rated, unchanged); the plant's own
        ``max_generation_mw`` envelope stays installed (14000), so rule 41
        holds exactly by construction (7000+7000 == 14000). Each group's
        ``max_turbined_m3s`` is head-corrected (ticket-017, per-conjunto
        cap: :func:`_itaipu_conjunto_head_corrected`), and the entity's
        ``max_turbined_m3s`` is their sum, per :func:`_build_split_unit_groups`.
        Group id 0 (50 Hz) is unconditionally relocated to the ``IV``
        transshipment bus; group id 1 (60 Hz) stays on the plant's own SE
        bus (ticket-006) — the relocation moves no envelope quantity, only
        the 50 Hz cell's ``bus_id``.
        """
        dadger, hidr, id_map, calendar, effective = _load_deck()
        doc = convert_hydros(
            dadger,
            hidr,
            id_map,
            calendar[0].start_date,
            effective,
        )
        hydro_id = id_map.hydro_id(_SPLIT_PLANT)
        itaipu = next(h for h in doc["hydros"] if h["id"] == hydro_id)
        assert itaipu["name"] == "ITAIPU"

        groups = itaipu["unit_groups"]
        assert len(groups) == 2
        assert {g["id"] for g in groups} == {0, 1}

        conjuntos = _itaipu_conjunto_rated(hidr)
        rho_by_hydro_id = _rho_by_hydro_id(effective, id_map, calendar)
        q_head_expected = _itaipu_conjunto_head_corrected(
            hidr, effective, calendar, rho_by_hydro_id, hydro_id
        )
        # id 0 (50 Hz) -> IV transshipment bus; id 1 (60 Hz) -> the plant's
        # own SE bus (ticket-006, unconditional whenever Itaipu is operated).
        expected_bus_by_group = {0: id_map.transhipment_bus_id, 1: id_map.bus_id(1)}
        for group in sorted(groups, key=lambda g: g["id"]):
            _, p_expected = conjuntos[group["id"]]
            assert group["max_generation_mw"] == pytest.approx(p_expected, abs=1e-6)
            assert group["max_generation_mw"] == pytest.approx(7000.0, abs=1e-6)
            assert group["max_turbined_m3s"] == pytest.approx(
                q_head_expected[group["id"]], abs=1e-6
            )
            assert group["min_generation_mw"] == 0.0
            assert group["min_turbined_m3s"] == 0.0
            assert group["bus_id"] == expected_bus_by_group[group["id"]]

        # Envelope unchanged by the relabel: the summed plant envelope is
        # exactly what it was before the bus relocation (only the 50 Hz
        # group's own bus_id moved).
        gen = itaipu["generation"]
        assert gen["max_generation_mw"] == pytest.approx(14000.0, abs=1e-6)
        assert gen["max_turbined_m3s"] == pytest.approx(sum(q_head_expected), abs=1e-6)
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

    def test_overlay_reconstruction_and_group_sum_matches_oracle(self) -> None:
        """Criterion 3, reconstructed independently from the registers: each
        emitted ``(group, stage)`` ``max_generation_mw`` equals the group's
        ``7000 x MP_g x FD_g`` availability to < 0.01 MW, with
        ``hydro_unit_group_id`` in {0, 1}, ``block_id`` null, every value
        <= 7000 (rule 45 holds per group).

        The head-derived hydraulic ceiling no longer caps generation (the
        head-corrected turbined-flow cap owns the physical limit), so the
        emitted per-group sum now equals the oracle's ``potencia_disponivel_MW``
        at every stage — no B8 accepted-cost delta — matching the register-side
        group-wise sum pinned by
        :meth:`test_availability_rule_is_exact_vs_oracle_group_wise`.
        """
        dadger, hidr, id_map, calendar, effective = _load_deck()
        hydro_id = id_map.hydro_id(_SPLIT_PLANT)

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

        values = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar, effective
        )
        group_bounds = convert_hydro_unit_group_bounds(values, calendar)
        bounds_pd = group_bounds.to_pandas()
        itaipu_bounds = bounds_pd[bounds_pd["hydro_id"] == hydro_id]
        assert set(itaipu_bounds["hydro_unit_group_id"]) == {0, 1}
        assert itaipu_bounds["block_id"].isna().all()

        emitted_sum = {0: 0.0, 1: 0.0, 2: 0.0}
        for group_id in (0, 1):
            _, p_g = conjuntos[group_id]
            assert p_g == pytest.approx(7000.0)
            _, mp_row = mp_rows[group_id]
            _, fd_row = fd_rows[group_id]

            for stage_index in range(3):
                stage_number = stage_index + 1
                availability_mw = (
                    p_g
                    * float(mp_row[f"manutencao_{stage_number}"])
                    * float(fd_row[f"fator_{stage_number}"])
                )
                key = (hydro_id, group_id, stage_index)
                emitted = values[key].max_generation_mw if key in values else None
                # No overlay row means availability sits at the group's rated
                # envelope (p_g) — the effective cap is then that envelope.
                effective = p_g if emitted is None else emitted
                assert effective == pytest.approx(availability_mw, abs=_TOL)
                assert effective <= 7000.0 + 1e-6
                emitted_sum[stage_index] += effective

        oracle = {0: 13328.0, 1: 13937.0, 2: 12236.0}
        for stage_index, expected in oracle.items():
            assert emitted_sum[stage_index] == pytest.approx(expected, abs=_TOL)

    def test_rule_45_mirror_raises_nothing_on_itaipus_own_emission(self) -> None:
        """The rule-45 mirror, run directly against Itaipu's own declared
        groups and overlay — every emitted value sits at or below
        ρ_eq x 6620 (~6810.80), itself below each group's declared 7000."""
        dadger, hidr, id_map, calendar, effective = _load_deck()
        doc = convert_hydros(
            dadger,
            hidr,
            id_map,
            calendar[0].start_date,
            effective,
        )
        values = convert_hydro_group_availability(
            dadger, hidr, id_map, calendar, effective
        )
        group_bounds = convert_hydro_unit_group_bounds(values, calendar)

        with dx.collect() as collected:
            check_group_bound_envelope(doc, group_bounds)

        errors = [d for d in collected if d.severity is Severity.ERROR]
        assert not errors, [d.summary for d in errors]
