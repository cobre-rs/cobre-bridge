"""Compare NEWAVE visited storage states (``cortese.dat``) against Cobre's.

NEWAVE writes ``VARM`` (useful volume = absolute − ``vmin``) per UHE at each
cut-generation trial point; Cobre exports ``StageStates`` (absolute hm³). We align
NEWAVE estudo stage ``s`` to Cobre stage ``s − NW_OFFSET`` (same convention as
``compare_cuts.py``: Cobre stage ``k`` ↔ NEWAVE estudo ``k + 10``), convert Cobre to
useful (``absolute − vmin`` with permanent VOLMIN overrides), and compare per
reservoir by name.

Both models visit *different* trial points (the forward policies differ — that's the
investigation), so divergence is expected; the value is *where* and *how much* the
visited storages differ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from inewave.newave.confhd import Confhd
from inewave.newave.hidr import Hidr

from cobre_bridge.converters.hydro import _apply_permanent_overrides
from cobre_bridge.newave_files import NewaveFiles
from cobre_cut_investigation import cobre_states
from cortese_investigation import read_cortese

NEWAVE_DIR = Path("./example/newave_rodada")
COBRE_DIR = Path("./example/cobre_rodada")
NW_OFFSET = 10  # Cobre stage k <-> NEWAVE estudo (k + NW_OFFSET)


def _key(s: str) -> str:
    return str(s).strip().upper()


def newave_useful_by_name() -> tuple[dict[int, pd.Series], dict[int, str], pd.Series]:
    """Return (estudo_stage -> VARM useful by UHE code), code->name, vmin by code."""
    nw = NewaveFiles.from_directory(NEWAVE_DIR)
    hidr = _apply_permanent_overrides(Hidr.read(str(nw.hidr)).cadastro, nw)
    confhd = Confhd.read(str(nw.confhd)).usinas
    code_name = {
        int(r["codigo_usina"]): _key(r["nome_usina"]) for _, r in confhd.iterrows()
    }
    # Per stage, keep the converged trial point: highest construction iteration
    # with real (non-placeholder) content.
    best: dict[int, tuple[int, pd.Series]] = {}
    for s in read_cortese(NEWAVE_DIR):
        if s.tipo_estagio != "estudo" or not bool((s.varm.abs() > 1e-6).any()):
            continue
        cur = best.get(s.estagio)
        if cur is None or s.iteracao > cur[0]:
            best[s.estagio] = (s.iteracao, s.varm)
    by_stage = {k: v[1] for k, v in best.items()}
    vmin = hidr["volume_minimo"]
    return by_stage, code_name, vmin


def compare_stage(cobre_stage: int) -> pd.DataFrame:
    """Per-reservoir NEWAVE-useful vs Cobre-useful storage at one Cobre stage."""
    by_stage, code_name, vmin = newave_useful_by_name()
    nw_stage = cobre_stage + NW_OFFSET
    if nw_stage not in by_stage:
        raise ValueError(f"no NEWAVE cortese state for estudo stage {nw_stage}")
    varm = by_stage[nw_stage]  # useful volume by UHE code
    nw = pd.DataFrame(
        {
            "nome": [code_name.get(c, "") for c in varm.index],
            "nw_useful": varm.to_numpy(),
            "vmin": vmin.reindex(varm.index).to_numpy(),
        }
    )
    nw["key"] = nw["nome"].map(_key)

    cob = cobre_states(COBRE_DIR, cobre_stage)
    cob = cob[cob["trial_index"] == 0].copy()  # first visited trial point
    cob["key"] = cob["nome"].map(_key)
    # vmin by name for the Cobre side (same physical reference as NEWAVE)
    vmin_by_key = dict(zip(nw["key"], nw["vmin"]))
    cob["cobre_useful"] = cob["storage_hm3"] - cob["key"].map(
        lambda k: vmin_by_key.get(k, 0.0)
    )

    m = nw.merge(cob[["key", "cobre_useful"]], on="key", how="inner")
    m["delta"] = m["nw_useful"] - m["cobre_useful"]
    return m.sort_values("delta", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    cobre_stage = 0
    m = compare_stage(cobre_stage)
    nw_stage = cobre_stage + NW_OFFSET
    tol = 1.0
    matched = int(
        (m["delta"].abs() <= np.maximum(tol, 0.02 * m["nw_useful"].abs())).sum()
    )
    print(
        f"Visited-state storage: Cobre stage {cobre_stage} <-> NEWAVE estudo "
        f"{nw_stage}  ({len(m)} reservoirs matched by name)"
    )
    print(f"  within ~2%: {matched}/{len(m)}")
    print(
        f"  mean |Δuseful| = {m['delta'].abs().mean():.1f} hm³   "
        f"max = {m['delta'].abs().max():.1f} hm³"
    )
    hdr = f"{'reservoir':<16}{'NEWAVE useful':>15}{'Cobre useful':>15}{'Δ hm³':>14}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for _, r in m.head(20).iterrows():
        print(
            f"{str(r['nome']):<16}{r['nw_useful']:>15.1f}"
            f"{r['cobre_useful']:>15.1f}{r['delta']:>14.1f}"
        )


if __name__ == "__main__":
    main()
