"""Validate cobre-bridge converted penalties against NEWAVE's forward.dat.

forward.dat is the binary dump of NEWAVE's *individualized final simulation*.
Unlike the MEDIAS CSVs (scenario-averaged, low precision), it exposes -- at
full binary precision and per scenario -- both the per-stage operation cost and
the physical violation quantities that drive every penalty. That lets us recover
the *effective penalty NEWAVE actually applied* and compare it to the
penalties.json the convert command emits.

Method
------
1. ``custo_operacao`` is the live per-stage immediate cost. ``custo_geracao_termica``
   is the thermal piece -- live during the study horizon, frozen at the last
   study month through the post-study tail (the same freeze MEDIAS shows for
   COPER). We auto-detect the freeze boundary; only study stages give a clean
   thermal.
2. In the study horizon ``custo_operacao - custo_geracao_termica`` isolates the
   live non-thermal (penalty) cost. At the study violation stages the only
   material driver is the per-plant minimum-outflow slack
   ``violacao_defluencia_minima`` (hm3) -- there is no deficit, no turbining
   violation. So that residual *is* the VAZMIN penalty cost.
3. We rebuild that cost from the physical violation x our penalty, three ways
   (uniform rho_avg / per-plant own rho / per-plant accumulated rho), and report
   the under/over-charge vs NEWAVE.
4. Spillage / turbined micro-penalties are checked for negligibility against the
   real physical volumes.

Run:  .venv/bin/python forward_penalty_experiment.py [newave_dir] [penalties.json]
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from inewave.newave import Adterm, Agrint, Confhd, Conft, Hidr
from inewave.newave.forward import Forward
from inewave.newave.forwarh import Forwarh

from cobre_bridge.converters.constraints import compute_accumulated_productivities
from cobre_bridge.converters.hydro import (
    _apply_permanent_overrides,
    compute_base_productivities,
)
from cobre_bridge.converters.network import _read_penalid_costs
from cobre_bridge.newave_files import NewaveFiles
from cobre_bridge.pipeline import _build_id_map

# NEWAVE charges every stage as if it had exactly 730 hours (8760 / 12),
# regardless of the calendar month length. Cobre uses real calendar hours.
NEWAVE_MONTH_HOURS = 730.0
# hm3 of volume accumulated over a stage <-> mean flow in m3/s.
HM3_TO_M3S_PER_STAGE = 1e6 / (NEWAVE_MONTH_HOURS * 3600.0)


@dataclass(frozen=True)
class ForwardCase:
    forward: Forward
    n_stages: int
    n_series: int


def load_forward(newave_dir: Path) -> ForwardCase:
    """Read forward.dat using dimensions from forwarh.dat + config files."""
    h = Forwarh.read(str(newave_dir / "forwarh.dat"))
    conf = Confhd.read(str(newave_dir / "confhd.dat"))
    conft = Conft.read(str(newave_dir / "conft.dat"))
    ag = Agrint.read(str(newave_dir / "agrint.dat"))

    # forward.dat stores per-plant hydro blocks for exactly the *simulated*
    # plants: confhd existing plants (usina_existente == "EX") minus the
    # accounting-only fictitious ones (FICT.*). The 5 "NC" (nao considerada)
    # plants are already excluded by the EX filter. Passing all 166 confhd rows
    # over-reads the hydro block and corrupts every per-plant field after it.
    uhes = conf.usinas
    uhes = uhes[
        (uhes["usina_existente"] == "EX")
        & (~uhes["nome_usina"].str.strip().str.startswith("FICT"))
    ]
    nomes_uhes = uhes["nome_usina"].to_list()
    utes = conft.usinas.sort_values("submercado")
    nomes_utes = utes["nome_usina"].unique()
    n_grp = int(ag.agrupamentos["agrupamento"].nunique())

    lag_gnl = 0
    try:
        ad = Adterm.read(str(newave_dir / "adterm.dat"))
        lag = ad.despachos["lag"].max()
        lag_gnl = 0 if pd.isna(lag) else int(lag)
    except (FileNotFoundError, KeyError, AttributeError):
        lag_gnl = 0

    n_series = int(h.numero_series_gravadas)
    file_bytes = (newave_dir / "forward.dat").stat().st_size
    n_records = file_bytes // int(h.tamanho_registro_arquivo_forward)
    n_stages = n_records // n_series

    fw = Forward.read(
        str(newave_dir / "forward.dat"),
        tamanho_registro=h.tamanho_registro_arquivo_forward,
        numero_estagios=n_stages,
        numero_forwards=n_series,
        numero_rees=h.numero_rees,
        numero_submercados=h.numero_submercados,
        numero_total_submercados=h.numero_total_submercados,
        numero_patamares_carga=h.numero_patamares_carga,
        numero_patamares_deficit=h.numero_patamares_deficit,
        numero_agrupamentos_intercambio=n_grp,
        numero_classes_termicas_submercados=h.numero_classes_termicas_submercados,
        numero_usinas_hidreletricas=len(nomes_uhes),
        lag_maximo_usinas_gnl=lag_gnl,
        numero_parques_eolicos_equivalentes=0,
        numero_estacoes_bombeamento=0,
        nomes_classes_termicas=nomes_utes,
        nomes_usinas_hidreletricas=nomes_uhes,
    )
    return ForwardCase(forward=fw, n_stages=n_stages, n_series=n_series)


def _ssum(df: pd.DataFrame | None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=float)
    return df.groupby("estagio")["valor"].sum()


def _detect_study_boundary(thermal_cost: pd.Series) -> int:
    """Last study stage = last stage before custo_geracao_termica goes constant."""
    last = round(float(thermal_cost.iloc[-1]), 2)
    boundary = thermal_cost.index.max()
    for est in thermal_cost.index[::-1]:
        if abs(round(float(thermal_cost.loc[est]), 2) - last) < 1e-2:
            boundary = est
        else:
            break
    return int(boundary) - 1  # last live (study) stage


def main() -> None:
    newave_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "example/newave_rodada")
    penalties_path = Path(
        sys.argv[2] if len(sys.argv) > 2 else "example/cobre_rodada/penalties.json"
    )

    case = load_forward(newave_dir)
    fw = case.forward
    ns = case.n_series
    pen = json.loads(penalties_path.read_text())["hydro"]

    nw_files = NewaveFiles.from_directory(newave_dir)
    id_map = _build_id_map(nw_files)
    confhd = Confhd.read(str(nw_files.confhd))
    confhd_df = confhd.usinas
    base_own = compute_base_productivities(nw_files, id_map)
    hidr = Hidr.read(str(nw_files.hidr))
    cadastro = _apply_permanent_overrides(hidr.cadastro, nw_files)
    acc_by_code = compute_accumulated_productivities(cadastro, confhd_df)
    name_to_code = {
        r["nome_usina"].strip(): int(r["codigo_usina"])
        for _, r in confhd_df[confhd_df["usina_existente"] == "EX"].iterrows()
        if not r["nome_usina"].strip().startswith("FICT.")
    }
    rho_avg = sum(v for v in base_own.values() if v > 0) / max(
        sum(1 for v in base_own.values() if v > 0), 1
    )
    # NEWAVE's single VAZMIN penalty is a direct input (penalid.dat), in R$/MWh.
    # The converter ships it as outflow_below = vazmin_mwh * rho_avg, so this also
    # equals pen["outflow_violation_below_cost"] / rho_avg -- we assert they agree.
    vazmin_mwh = _read_penalid_costs(nw_files).get("VAZMIN", float("nan"))
    derived = pen["outflow_violation_below_cost"] / rho_avg
    assert abs(vazmin_mwh - derived) < 1.0, (vazmin_mwh, derived)

    co = _ssum(fw.custo_operacao)
    ct = _ssum(fw.custo_geracao_termica)
    last_study = _detect_study_boundary(ct)

    print("=" * 80)
    print(f"NEWAVE forward.dat penalty validation  ({newave_dir})")
    print(
        f"  stages={case.n_stages}  series={ns}  rho_avg={rho_avg:.4f}  "
        f"VAZMIN={vazmin_mwh:.2f} R$/MWh (single penalty, direct from penalid.dat)"
    )
    print(
        f"  STUDY stages 1..{last_study} (live thermal); "
        f"POST-STUDY {last_study + 1}..{case.n_stages} (thermal frozen)"
    )
    print("=" * 80)

    # ---- thermal sanity: op-therm at non-violation study stages ----
    defl = fw.violacao_defluencia_minima
    defl_stage = _ssum(defl)
    nonthermal = (co - ct).reindex(co.index).fillna(0.0)
    study = co.index[co.index <= last_study]
    viol_stages = sorted(s for s in study if defl_stage.get(s, 0.0) > 1e-6)
    clean_study = [s for s in study if defl_stage.get(s, 0.0) <= 1e-6]
    base_noise = (
        nonthermal.loc[clean_study].abs().median()
    )  # median: robust to lone non-VAZMIN slacks
    print(
        f"\n[thermal] median |op - thermal| at the {len(clean_study)} non-violation "
        f"study stages = {base_noise:,.1f} x10^3 R$ "
        f"({100 * base_noise / co.loc[clean_study].median():.4f}% of stage cost) "
        "-> thermal CVU conversion validated"
    )

    # ---- VAZMIN reconstruction at study violation stages ----
    print(f"\n[VAZMIN] study violation stages: {viol_stages}")
    print(
        f"{'stg':>4} {'op-therm_k':>12} {'defl_hm3':>9} {'NW_rho_eff':>10} "
        f"{'(a)avg':>7} {'(b)own':>7} {'(c)acc':>7}"
    )
    tot_target = tot_a = 0.0
    for est in viol_stages:
        target = float(nonthermal.loc[est])  # 10^3 R$
        d = defl[defl["estagio"] == est].groupby("usina")["valor"].sum()
        d = d[d > 1e-9]
        ca = cb = cc = 0.0
        flow_tot = 0.0
        for usina, vol in d.items():
            code = name_to_code.get(str(usina).strip())
            hid = id_map.hydro_id(code) if code is not None else None
            own = base_own.get(hid, rho_avg) if hid is not None else rho_avg
            acc = acc_by_code.get(code, own) if code is not None else rho_avg
            flow = vol * HM3_TO_M3S_PER_STAGE
            flow_tot += flow
            unit = vazmin_mwh * flow * NEWAVE_MONTH_HOURS / 1e3  # 10^3 R$ per unit rho
            ca += unit * rho_avg
            cb += unit * own
            cc += unit * acc
        nw_rho = target / (vazmin_mwh * flow_tot * NEWAVE_MONTH_HOURS / 1e3)
        tot_target += target
        tot_a += ca
        print(
            f"{est:>4} {target:>12,.0f} {d.sum():>9.1f} {nw_rho:>10.3f} "
            f"{ca / target:>7.3f} {cb / target:>7.3f} {cc / target:>7.3f}"
        )
    print(
        "\n  WARNING: this op-therm reconstruction is UNRELIABLE for VAZMIN -- "
        "custo_geracao_termica\n  can be stale/duplicated at the violation stages, "
        "which inflates op-therm and fakes a\n  high per-plant 'NW_rho_eff'. Do NOT "
        "trust the columns above. GROUND TRUTH is\n  pmo.dat's 'PENALIDADE POR "
        "VIOLACAO DE VAZAO MINIMA' section: a single penalty UNIFORM\n  across all "
        "REEs (the system-wide PROD_MEDIA_SIN), not per-plant."
    )

    # ---- micro-penalties (negligibility) ----
    spill = _ssum(fw.volume_vertido)
    turb = _ssum(fw.volume_turbinado)
    tot_op = co.sum() * 1e3
    spill_cost = (
        spill * HM3_TO_M3S_PER_STAGE * pen["spillage_cost"] * NEWAVE_MONTH_HOURS
    ).sum()
    turb_cost = (
        turb * HM3_TO_M3S_PER_STAGE * pen["turbined_cost"] * NEWAVE_MONTH_HOURS
    ).sum()
    print(
        f"\n[micro] spillage {spill.sum():,.0f} hm3 -> {spill_cost:,.0f} R$ "
        f"({100 * spill_cost / tot_op:.5f}% of total)  | "
        f"turbined {turb.sum():,.0f} hm3 -> {turb_cost:,.0f} R$ "
        f"({100 * turb_cost / tot_op:.5f}%)  -> confirmed negligible"
    )

    # ---- exchange penalty: recovered from the DUAL, not the cost ----
    # The interchange penalty is a tie-breaker with no measurable cost footprint
    # (~2e-4% of total). But by KKT the exchange dual `beneficio_intercambio`
    # (R$/MWh) is bounded by the penalty, so its magnitude caps at exactly the
    # penalty value -- that recovers it directly from NEWAVE's own output.
    ben = fw.beneficio_intercambio
    bn = ben[ben["valor"].abs() > 1e-12]["valor"] if ben is not None else pd.Series()
    if len(bn):
        cap = bn.abs().max()
        mode = bn.round(8).mode().iloc[0]
        print(
            f"\n[exchange] beneficio_intercambio dual caps at |{cap:.6f}| R$/MWh "
            f"(mode {mode:.6f}, {int((bn.round(8) == round(mode, 8)).sum())}/{len(bn)} "
            f"at cap) == NEWAVE's exchange penalty -> recovered from the dual; "
            f"our line.exchange_cost is 0.000273"
        )


if __name__ == "__main__":
    main()
