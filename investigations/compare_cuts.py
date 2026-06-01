"""Side-by-side NEWAVE vs Cobre FCF cut comparison (per-reservoir water values).

Stage alignment (study period). NEWAVE indexes ``estudo`` stage 1 = January of
its first study year; Cobre is 0-based from the case's first month, and
``stage_s.bin`` holds the cost-to-go that stage ``s`` accesses. So the mapping is
a fixed offset that depends on the case's start month:

    Cobre stage s  <->  NEWAVE ('estudo', s + NEWAVE_ESTUDO_OFFSET)

Set ``NEWAVE_ESTUDO_OFFSET`` / ``MAX_ALIGNED_COBRE_STAGE`` below per case.
Post-study ('pos') stages need separate handling (NEWAVE may REE-aggregate them).

Units are pinned: the readers normalize NEWAVE (10³ R$) and Cobre (10⁶ R$) to R$,
so the water values are directly comparable in R$/hm³.

Filtering is explicit: we select a single cut per side via ``iteration`` (and
``forward_pass``) — no implicit averaging. First investigation: ``iteration=1``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from cobre_cut_investigation import cobre_water_values
from newave_cut_investigation import newave_water_values

NEWAVE_DIR = Path("./example/newave_rodada")
COBRE_DIR = Path("./example/cobre_rodada")
COBRE_STAGE = 0  # which Cobre study stage's cost-to-go to compare
# Case-dependent stage alignment (defaults are for the repo's example case):
NEWAVE_ESTUDO_OFFSET = 10  # Cobre stage s <-> NEWAVE estudo (s + offset)
MAX_ALIGNED_COBRE_STAGE = 26  # last study stage that maps to an estudo cut
# Which cut to compare. "first_real" = each solver's first real cut (skips
# NEWAVE's iteration-0 zero placeholder); "newest" = the converged cut.
SELECT = "first_real"


def newave_stage_for(cobre_stage: int) -> tuple[str, int]:
    """Map a Cobre study stage to its NEWAVE (tipo_estagio, estagio)."""
    if 0 <= cobre_stage <= MAX_ALIGNED_COBRE_STAGE:
        return ("estudo", cobre_stage + NEWAVE_ESTUDO_OFFSET)
    raise ValueError(
        f"Cobre stage {cobre_stage} is post-study; only study stages "
        f"0..{MAX_ALIGNED_COBRE_STAGE} are aligned for now (pos stages need the "
        "aggregated layout)."
    )


def _key(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper()


def main() -> None:
    tipo, est = newave_stage_for(COBRE_STAGE)
    nw = newave_water_values(NEWAVE_DIR, tipo, est, select=SELECT)
    cb = cobre_water_values(COBRE_DIR, COBRE_STAGE, select=SELECT)

    print(
        f"Alinhamento: Cobre stage {COBRE_STAGE}  <->  NEWAVE {tipo} stage {est} "
        f"(corte: {SELECT})"
    )
    print(f"  NEWAVE: {nw.attrs['n_cuts']} corte, rhs={nw.attrs['rhs']:.4e} R$")
    print(
        f"  Cobre : {cb.attrs['n_cuts']} corte, "
        f"intercept={cb.attrs['intercept']:.4e} R$"
    )

    nw = nw.assign(key=_key(nw["nome"]))
    cb = cb.assign(key=_key(cb["nome"]))
    merged = nw.merge(cb, on="key", how="outer", suffixes=("_nw", "_cb"))
    merged["nome"] = merged["nome_nw"].fillna(merged["nome_cb"])
    merged["delta"] = merged["water_value_newave"] - merged["water_value_cobre"]
    merged["abs_max"] = (
        merged[["water_value_newave", "water_value_cobre"]].abs().max(axis=1)
    )
    merged = merged.sort_values("abs_max", ascending=False)

    matched = (
        merged[["water_value_newave", "water_value_cobre"]].notna().all(axis=1).sum()
    )
    print(
        f"\nreservatórios: NEWAVE={len(nw)} Cobre={len(cb)} casados_por_nome={matched}"
    )
    cols_hdr = (
        f"{'reservatório':<16}{'NEWAVE R$/hm³':>18}"
        f"{'Cobre R$/hm³':>18}{'Δ (R$/hm³)':>18}"
    )
    print(f"\n{cols_hdr}")
    print("-" * 70)
    for _, r in merged.head(20).iterrows():
        nwv, cbv, dv = r["water_value_newave"], r["water_value_cobre"], r["delta"]
        nws = f"{nwv:>18.2f}" if pd.notna(nwv) else f"{'—':>18}"
        cbs = f"{cbv:>18.2f}" if pd.notna(cbv) else f"{'—':>18}"
        dvs = f"{dv:>18.2f}" if pd.notna(dv) else f"{'—':>18}"
        print(f"{str(r['nome']):<16}{nws}{cbs}{dvs}")


if __name__ == "__main__":
    main()
