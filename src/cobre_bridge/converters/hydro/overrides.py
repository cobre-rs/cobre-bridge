"""Source-file override readers for hydro entity conversion.

Reads MODIF.DAT (permanent + temporal overrides), GHMIN.DAT, and
PENALID.DAT. The package's lowest layer: imports nothing from a sibling
submodule.
"""

from __future__ import annotations

import logging

import pandas as pd

from cobre_bridge.case import NewaveCase
from cobre_bridge.horizon import POST_STUDY_YEAR

_LOG = logging.getLogger(__name__)


# Temporal override types extracted from MODIF.DAT.
_TEMPORAL_OVERRIDE_TYPES = frozenset(
    {"VAZMINT", "VMAXT", "VMINT", "CFUGA", "CMONT", "TURBMINT", "TURBMAXT"}
)


def _apply_permanent_overrides(
    cadastro: pd.DataFrame, case: NewaveCase
) -> pd.DataFrame:
    """Apply MODIF.DAT permanent overrides to the hidr.dat cadastro.

    Reads ``MODIF.DAT`` from *case* and
    applies permanent override records — VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
    NUMMAQ — to a *copy* of *cadastro*.  The original DataFrame is not
    mutated.

    Parameters
    ----------
    cadastro:
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina``.
    case:
        Parsed the source model case.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with permanent overrides applied.
    """
    modif = case.modif
    if modif is None:
        _LOG.debug("MODIF.DAT not found; skipping permanent overrides.")
        return cadastro

    result = cadastro.copy()

    # Ensure float dtype for columns that permanent overrides may assign floats
    # into.  Without this, pandas 2.x raises TypeError when the column was
    # inferred as int64 (e.g. vazao_minima_historica=[0, 0]).
    _float_override_cols = (
        "vazao_minima_historica",
        "volume_maximo",
        "volume_minimo",
    )
    for _col in _float_override_cols:
        if _col in result.columns and result[_col].dtype.kind == "i":
            result[_col] = result[_col].astype(float)

    usina_records = modif.usina()
    if not usina_records:
        return result

    for usina_rec in usina_records:
        code = int(usina_rec.codigo)
        if code not in result.index:
            _LOG.warning(
                "MODIF.DAT references plant code %d which is not in hidr.dat;"
                " skipping.",
                code,
            )
            continue

        for rec in modif.modificacoes_usina(code):
            type_name = type(rec).__name__

            # Skip temporal override types — handled separately.
            if type_name in _TEMPORAL_OVERRIDE_TYPES:
                continue

            if type_name == "VAZMIN":
                result.loc[code, "vazao_minima_historica"] = float(rec.vazao)

            elif type_name == "VOLMAX":
                result.loc[code, "volume_maximo"] = float(rec.volume)

            elif type_name == "VOLMIN":
                result.loc[code, "volume_minimo"] = float(rec.volume)

            elif type_name == "NUMCNJ":
                result.loc[code, "numero_conjuntos_maquinas"] = int(rec.numero)

            elif type_name == "NUMMAQ":
                set_num = int(rec.conjunto)
                n_maq = int(rec.numero_maquinas)
                result.loc[code, f"maquinas_conjunto_{set_num}"] = n_maq

            elif type_name in ("VOLCOTA", "COTARE"):
                # VOLCOTA/COTARE are not present in the example case.
                # The spec mentions them but the inewave API does not expose
                # them as separate methods in the tested version.  Log a
                # warning if they appear so the operator knows to investigate.
                _LOG.warning(
                    "MODIF.DAT contains unsupported permanent override type"
                    " '%s' for plant %d; skipping.",
                    type_name,
                    code,
                )

            elif type_name == "DefaultRegister":
                # inewave emits DefaultRegister for records it does not model
                # (e.g. COTAREA). These are benign for the conversion, so log at
                # debug level only — no user-facing warning.
                _LOG.debug(
                    "MODIF.DAT contains an unmodeled record (DefaultRegister)"
                    " for plant %d; skipping.",
                    code,
                )

            else:
                _LOG.warning(
                    "MODIF.DAT contains unknown permanent override type '%s'"
                    " for plant %d; skipping.",
                    type_name,
                    code,
                )

    return result


def read_cadastro(case: NewaveCase) -> pd.DataFrame:
    """Read ``hidr.dat`` and apply permanent MODIF.DAT overrides.

    Parameters
    ----------
    case:
        Parsed the source model case.

    Returns
    -------
    pd.DataFrame
        The ``Hidr.cadastro`` DataFrame indexed by ``codigo_usina`` with all
        permanent MODIF.DAT overrides (VAZMIN, VOLMAX, VOLMIN, NUMCNJ,
        NUMMAQ) already applied.
    """
    cadastro = case.hidr.cadastro
    return _apply_permanent_overrides(cadastro, case)


def _extract_temporal_overrides(
    case: NewaveCase, confhd_codes: list[int]
) -> dict[int, list[dict]]:
    """Extract MODIF.DAT temporal overrides for plants in *confhd_codes*.

    Reads ``MODIF.DAT`` and returns a dict keyed by plant code.  Each value
    is a list of override dicts in file order::

        {"type": str, "month": int, "year": int, "value": float}

    For CFUGA/CMONT the ``"value"`` field is the level in metres.  For
    TURBMINT/TURBMAXT it is the turbined flow in m³/s.  For VAZMINT/VMAXT/
    VMINT it is the volume or flow as stored in the record.

    Parameters
    ----------
    case:
        Parsed the source model case.
    confhd_codes:
        List of plant codes present in the study (from confhd.dat).  Records
        for plants not in this list are excluded.

    Returns
    -------
    dict[int, list[dict]]
        Temporal override records per plant code.  Empty dict if MODIF.DAT is
        absent.
    """
    modif = case.modif
    if modif is None:
        _LOG.debug("MODIF.DAT not found; no temporal overrides extracted.")
        return {}

    confhd_set = set(confhd_codes)
    result: dict[int, list[dict]] = {}

    usina_records = modif.usina()
    if not usina_records:
        return result

    for usina_rec in usina_records:
        code = int(usina_rec.codigo)
        if code not in confhd_set:
            continue

        plant_overrides: list[dict] = []
        for rec in modif.modificacoes_usina(code):
            type_name = type(rec).__name__
            if type_name not in _TEMPORAL_OVERRIDE_TYPES:
                continue

            data = rec.data_inicio
            month = int(data.month)
            year = int(data.year)

            if type_name in ("VAZMINT",):
                value = float(rec.vazao)
            elif type_name in ("VMAXT", "VMINT"):
                value = float(rec.volume)
            elif type_name in ("CFUGA", "CMONT"):
                value = float(rec.nivel)
            elif type_name in ("TURBMINT", "TURBMAXT"):
                value = float(rec.turbinamento)
            else:
                _LOG.warning(
                    "Unknown temporal override type '%s' for plant %d; skipping.",
                    type_name,
                    code,
                )
                continue

            plant_overrides.append(
                {"type": type_name, "month": month, "year": year, "value": value}
            )

        if plant_overrides:
            result[code] = plant_overrides

    return result


def _read_ghmin_per_stage(
    case: NewaveCase,
    start_year: int,
    start_month: int,
    study_months: int,
    total_stages: int,
) -> dict[int, dict[int, float]]:
    """Read GHMIN.DAT and expand into ``{plant_code: {stage_0based: min_gen_mw}}``.

    GHMIN values are time-varying minimum-generation requirements in MWmes that
    source-model enforces per plant per stage.  Each (plant, month, year) record sets
    the value from that stage forward until the next record overrides it (step
    function).  Records with ``year == 9999`` are post-study seasonal entries: each
    calendar month they appear for becomes the value used in every post-study stage with
    that calendar month, falling back to a seasonal repeat of the last study year for
    unspecified months.

    Only ``patamar == 0`` rows are used — they represent the all-blocks
    mean, which matches the per-stage granularity of
    ``hydro_bounds.parquet``.

    Returns an empty mapping when ``GHMIN.DAT`` is absent.

    Parameters
    ----------
    case:
        Parsed the source model case.
    start_year, start_month:
        Study start (Cobre stage 0 corresponds to this calendar month).
    study_months:
        Number of in-study stages.
    total_stages:
        Total number of stages (study + post-study).
    """
    ghmin = case.ghmin
    if ghmin is None:
        _LOG.debug("GHMIN.DAT not found; emitting no per-stage min_generation.")
        return {}

    df = ghmin.geracoes
    if df is None or df.empty:
        return {}

    patamar0 = df[df["patamar"] == 0]
    if patamar0.empty:
        return {}

    result: dict[int, dict[int, float]] = {}
    for code, group in patamar0.groupby("codigo_usina"):
        code_int = int(code)
        study_changes: list[tuple[int, float]] = []
        pos_by_month: dict[int, float] = {}
        for _, row in group.iterrows():
            dt = row["data"]
            yr = int(dt.year)
            mo = int(dt.month)
            value = float(row["geracao"])
            if yr == POST_STUDY_YEAR:
                pos_by_month[mo] = value
                continue
            sid = (yr - start_year) * 12 + (mo - start_month)
            if sid < 0:
                sid = 0
            study_changes.append((sid, value))
        study_changes.sort()

        per_stage: dict[int, float] = {}
        # Step function across the study period.
        if study_changes:
            first_stage = study_changes[0][0]
            cp_idx = 0
            current: float | None = None
            for stage_id in range(first_stage, study_months):
                while (
                    cp_idx < len(study_changes) and study_changes[cp_idx][0] <= stage_id
                ):
                    current = study_changes[cp_idx][1]
                    cp_idx += 1
                if current is not None:
                    per_stage[stage_id] = current

        # Seasonal pattern for post-study: prefer explicit POS entries
        # for each calendar month; fall back to the last study year's
        # value for that calendar month.
        if total_stages > study_months:
            last_year_seasonal: dict[int, float] = {}
            for stage_id in range(max(0, study_months - 12), study_months):
                if stage_id in per_stage:
                    cal = ((start_month - 1 + stage_id) % 12) + 1
                    last_year_seasonal[cal] = per_stage[stage_id]

            for stage_id in range(study_months, total_stages):
                cal = ((start_month - 1 + stage_id) % 12) + 1
                if cal in pos_by_month:
                    per_stage[stage_id] = pos_by_month[cal]
                elif cal in last_year_seasonal:
                    per_stage[stage_id] = last_year_seasonal[cal]

        if per_stage:
            result[code_int] = per_stage

    return result


# Mapping from PENALID.DAT variable names to Cobre penalty field names.
_PENALID_VAR_MAP: dict[str, str] = {
    "DESVIO": "water_withdrawal_violation_cost",
    "VAZMIN": "outflow_violation_below_cost",
    "VAZMAX": "outflow_violation_above_cost",
    "GHMIN": "generation_violation_below_cost",
    "TURBMN": "turbined_violation_below_cost",
    "TURBMX": "outflow_violation_above_cost",
}


def _read_penalid(case: NewaveCase) -> dict[int, dict[str, float]]:
    """Read PENALID.DAT and return per-REE penalty override mappings.

    If ``PENALID.DAT`` is absent (``case.penalid is None``), returns an
    empty dict.  Only the first patamar tier (``patamar_penalidade == 1``)
    is used — tier 2 has NaN costs (unbounded) and is skipped.  NaN values
    within tier 1 are also skipped.

    Parameters
    ----------
    case:
        Parsed the source model case.

    Returns
    -------
    dict[int, dict[str, float]]
        Mapping from REE/subsystem code to a dict of Cobre penalty field
        names -> cost in R$/MWh.  Only fields with valid (non-NaN) values
        are included.  Returns an empty dict if the file is absent or
        contains no usable rows.
    """
    penalid = case.penalid
    if penalid is None:
        _LOG.debug("PENALID.DAT not found; leaving all plant penalties as None.")
        return {}

    df: pd.DataFrame | None = penalid.penalidades
    if df is None or df.empty:
        return {}

    # Keep only first-tier rows (patamar_penalidade == 1).
    tier1 = df[df["patamar_penalidade"] == 1]
    if tier1.empty:
        return {}

    result: dict[int, dict[str, float]] = {}
    for _, row in tier1.iterrows():
        variavel = str(row["variavel"]).strip()
        cobre_field = _PENALID_VAR_MAP.get(variavel)
        if cobre_field is None:
            # Variable not mapped (e.g. TURBMX, ELETRI) — skip silently.
            continue

        ree_code = int(row["codigo_ree_submercado"])
        valor = row["valor_R$_MWh"]

        if pd.isna(valor):
            continue

        cost = float(valor)
        result.setdefault(ree_code, {})[cobre_field] = cost

    return result


def _per_stage_drop_overrides(
    drop_overrides: list[dict],
    case: NewaveCase,
    total_stages: int,
) -> list[tuple[float | None, float | None]]:
    """Per-stage effective ``(CFUGA, CMONT)`` from MODIF.DAT step-functions.

    Evolves the CFUGA / CMONT temporal overrides into a per-stage
    ``(canal_fuga, cmont)`` state: each event applies from its stage of effect
    forward; events before the horizon fold in at stage 0; and when
    ``sazonaliza_cfuga_cmont == 1`` the calendar month's latest-year value
    repeats after the last explicit event. Shared by both
    :func:`_per_stage_productivities` (point ρ, combined with VOLREF_SAZ) and
    :func:`_per_stage_equivalent_productivities` (PRODT, which ignores VOLREF_SAZ).
    """
    if not drop_overrides:
        return [(None, None)] * total_stages

    dger = case.dger
    start_year = int(dger.ano_inicio_estudo)
    start_month = int(dger.mes_inicio_estudo)
    seasonalize = int(getattr(dger, "sazonaliza_cfuga_cmont", 0) or 0) == 1

    events_by_stage: dict[int, list[tuple[float | None, float | None]]] = {}
    last_event_stage = -1
    for override in drop_overrides:
        stage_id = (override["year"] - start_year) * 12 + (
            override["month"] - start_month
        )
        last_event_stage = max(last_event_stage, stage_id)
        if override["type"] == "CFUGA":
            events_by_stage.setdefault(stage_id, []).append(
                (float(override["value"]), None)
            )
        else:  # CMONT
            events_by_stage.setdefault(stage_id, []).append(
                (None, float(override["value"]))
            )

    # sazonaliza_cfuga_cmont == 1: after the last explicit entry the source model
    # repeats the seasonal pattern from the latest year defining each calendar month.
    seasonal_cfuga: dict[int, float] = {}
    seasonal_cmont: dict[int, float] = {}
    if seasonalize:
        latest_cfuga: dict[int, int] = {}
        latest_cmont: dict[int, int] = {}
        for override in drop_overrides:
            year = int(override["year"])
            month = int(override["month"])
            value = float(override["value"])
            if override["type"] == "CFUGA":
                if month not in latest_cfuga or year > latest_cfuga[month]:
                    latest_cfuga[month] = year
                    seasonal_cfuga[month] = value
            elif month not in latest_cmont or year > latest_cmont[month]:
                latest_cmont[month] = year
                seasonal_cmont[month] = value

    drops: list[tuple[float | None, float | None]] = []
    active_cfuga: float | None = None
    active_cmont: float | None = None
    for stage_id in range(total_stages):
        if stage_id == 0:
            applicable_stages = sorted(s for s in events_by_stage if s <= 0)
        elif stage_id in events_by_stage:
            applicable_stages = [stage_id]
        else:
            applicable_stages = []
        for past_stage in applicable_stages:
            for cfuga_val, cmont_val in events_by_stage[past_stage]:
                if cfuga_val is not None:
                    active_cfuga = cfuga_val
                if cmont_val is not None:
                    active_cmont = cmont_val

        calendar_month = ((start_month - 1 + stage_id) % 12) + 1
        if seasonalize and stage_id > last_event_stage:
            if calendar_month in seasonal_cfuga:
                active_cfuga = seasonal_cfuga[calendar_month]
            if calendar_month in seasonal_cmont:
                active_cmont = seasonal_cmont[calendar_month]

        drops.append((active_cfuga, active_cmont))
    return drops
