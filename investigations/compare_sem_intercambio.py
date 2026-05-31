"""Per-bus energy-balance comparison NEWAVE vs Cobre on the no-exchange case.

With every exchange removed, each submarket/bus is a closed system: net load
(NCS treated as non-curtailable) must be met by *local* thermal + hydro, with
excess/deficit absorbing any imbalance. That isolates each bus, so a per-bus
balance directly shows *which* bus and *which* component (thermal vs hydro)
drives Cobre's much larger excess.

All quantities are reported as stage-average MW (MWmed), block-hour weighted, so
the 730h-vs-real-calendar-hours difference between the two models is neutralised.
Summing the per-stage MWmed over the 64 stages gives the "MWmes" totals.

Run:  .venv/bin/python compare_sem_intercambio.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from inewave.newave import Confhd, Conft, Ree

import forward_penalty_experiment as fpe

COBRE = Path("example/cobre_rodada_sem_intercambio")
NEWAVE = Path("example/newave_rodada_sem_intercambio")

SUB_NAME = {1: "SUDESTE", 2: "SUL", 3: "NORDESTE", 4: "NORTE"}


def _load(path: str) -> dict:
    return json.loads((COBRE / path).read_text())


def cobre_per_bus() -> pl.DataFrame:
    """Stage-average MW per (stage_id, bus_name) for every balance component."""
    buses = {b["id"]: b["name"] for b in _load("system/buses.json")["buses"]}
    hydro_bus = {h["id"]: h["bus_id"] for h in _load("system/hydros.json")["hydros"]}
    therm_bus = {
        t["id"]: t["bus_id"] for t in _load("system/thermals.json")["thermals"]
    }
    ncs_bus = {
        n["id"]: n["bus_id"]
        for n in _load("system/non_controllable_sources.json")[
            "non_controllable_sources"
        ]
    }

    stages = json.loads((COBRE / "stages.json").read_text())["stages"]
    stage_hours = {s["id"]: sum(b["hours"] for b in s["blocks"]) for s in stages}

    sim = COBRE / "output/simulation"

    def avg_mw(parquet: Path, val: str, id_col: str, id2bus: dict) -> pl.DataFrame:
        df = pl.read_parquet(parquet)
        return (
            df.with_columns(
                pl.col(id_col).replace_strict(id2bus, default=-1).alias("bus_id")
            )
            .group_by(["stage_id", "bus_id"])
            .agg(pl.col(val).sum().alias("mwh"))
            .with_columns(
                pl.col("stage_id").replace_strict(stage_hours, default=730.0).alias("h")
            )
            .with_columns((pl.col("mwh") / pl.col("h")).alias("mw"))
            .select("stage_id", "bus_id", "mw")
        )

    # bus-level (load/excess/deficit) -- bus_id already present
    bdf = pl.read_parquet(sim / "buses/scenario_id=0000/data.parquet")
    bh = bdf.group_by(["stage_id", "bus_id"]).agg(
        pl.col("load_mwh").sum(),
        pl.col("excess_mwh").sum(),
        pl.col("deficit_mwh").sum(),
    )
    bh = (
        bh.with_columns(
            pl.col("stage_id").replace_strict(stage_hours, default=730.0).alias("h")
        )
        .with_columns(
            (pl.col("load_mwh") / pl.col("h")).alias("load"),
            (pl.col("excess_mwh") / pl.col("h")).alias("excess"),
            (pl.col("deficit_mwh") / pl.col("h")).alias("deficit"),
        )
        .select("stage_id", "bus_id", "load", "excess", "deficit")
    )

    th = avg_mw(
        sim / "thermals/scenario_id=0000/data.parquet",
        "generation_mwh",
        "thermal_id",
        therm_bus,
    ).rename({"mw": "thermal"})
    hy = avg_mw(
        sim / "hydros/scenario_id=0000/data.parquet",
        "generation_mwh",
        "hydro_id",
        hydro_bus,
    ).rename({"mw": "hydro"})
    nc = avg_mw(
        sim / "non_controllables/scenario_id=0000/data.parquet",
        "generation_mwh",
        "non_controllable_id",
        ncs_bus,
    ).rename({"mw": "ncs"})

    out = (
        bh.join(th, on=["stage_id", "bus_id"], how="left")
        .join(hy, on=["stage_id", "bus_id"], how="left")
        .join(nc, on=["stage_id", "bus_id"], how="left")
        .fill_null(0.0)
        .with_columns(pl.col("bus_id").replace_strict(buses, default="?").alias("bus"))
        .with_columns((pl.col("load") - pl.col("ncs")).alias("net_load"))
    )
    return out


def newave_per_bus() -> pl.DataFrame:
    """Stage-average MW per (stage_id, submarket) from forward.dat."""
    case = fpe.load_forward(NEWAVE)
    fw = case.forward

    ree = Ree.read(str(NEWAVE / "ree.dat")).rees
    ree_code2sub = {
        int(r["codigo"]): SUB_NAME[int(r["submercado"])] for _, r in ree.iterrows()
    }
    conft = Conft.read(str(NEWAVE / "conft.dat")).usinas
    ute2sub = {
        str(r["nome_usina"]).strip(): SUB_NAME[int(r["submercado"])]
        for _, r in conft.iterrows()
    }
    confhd = Confhd.read(str(NEWAVE / "confhd.dat")).usinas
    uhe2sub = {
        str(r["nome_usina"]).strip(): ree_code2sub.get(int(r["ree"]))
        for _, r in confhd.iterrows()
    }

    rows: dict[tuple[int, str], dict[str, float]] = {}

    def add(stage_id: int, sub: str, field: str, val: float) -> None:
        rows.setdefault((stage_id, sub), {}).setdefault(field, 0.0)
        rows[(stage_id, sub)][field] += val

    # forward.dat per-patamar values are MWmed *contributions* (instantaneous MW
    # already weighted by patamar duration), so the stage MWmed is their plain
    # SUM over patamares -- matching mercado_liquido, which is stage-level MWmed.
    # mercado_liquido: stage-level MWmed per submarket
    for _, r in fw.mercado_liquido.iterrows():
        add(int(r["estagio"]) - 1, str(r["submercado"]).strip(), "net_load", r["valor"])
    # thermal: per usina, patamar
    for _, r in fw.geracao_termica.iterrows():
        sub = ute2sub.get(str(r["usina"]).strip())
        if sub is None:
            continue
        add(int(r["estagio"]) - 1, sub, "thermal", r["valor"])
    # hydro generation: per usina, patamar (individualized dump -> plant-level;
    # the REE-aggregated controllable/fio fields are empty here)
    for _, r in fw.geracao_hidraulica_usina.iterrows():
        sub = uhe2sub.get(str(r["usina"]).strip())
        if sub is None:
            continue
        add(int(r["estagio"]) - 1, sub, "hydro", r["valor"])
    # excess: per submarket, patamar
    for _, r in fw.excesso.iterrows():
        add(int(r["estagio"]) - 1, str(r["submercado"]).strip(), "excess", r["valor"])
    # deficit: per submarket, patamarDeficit, patamar
    for _, r in fw.deficit.iterrows():
        add(int(r["estagio"]) - 1, str(r["submercado"]).strip(), "deficit", r["valor"])

    recs = []
    for (stage_id, sub), d in rows.items():
        recs.append(
            {
                "stage_id": stage_id,
                "bus": sub,
                "net_load": d.get("net_load", 0.0),
                "thermal": d.get("thermal", 0.0),
                "hydro": d.get("hydro", 0.0),
                "excess": d.get("excess", 0.0),
                "deficit": d.get("deficit", 0.0),
            }
        )
    return pl.DataFrame(recs)


def main() -> None:
    cob = cobre_per_bus()
    nw = newave_per_bus()

    # --- balance closure check (per side) ---
    cob_chk = cob.with_columns(
        (
            pl.col("thermal")
            + pl.col("hydro")
            + pl.col("deficit")
            - pl.col("excess")
            - pl.col("net_load")
        ).alias("resid")
    )
    nw_chk = nw.with_columns(
        (
            pl.col("thermal")
            + pl.col("hydro")
            + pl.col("deficit")
            - pl.col("excess")
            - pl.col("net_load")
        ).alias("resid")
    )
    print("=" * 78)
    print("BALANCE CLOSURE (max |thermal+hydro+deficit-excess-net_load|, MWmed)")
    print(f"  Cobre : {cob_chk['resid'].abs().max():.3f}")
    print(f"  NEWAVE: {nw_chk['resid'].abs().max():.3f}")
    print("=" * 78)

    # --- macro: sum over stages, per bus ---
    def agg(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("bus")
            .agg(
                pl.col("net_load").sum(),
                pl.col("thermal").sum(),
                pl.col("hydro").sum(),
                pl.col("excess").sum(),
                pl.col("deficit").sum(),
            )
            .sort("bus")
        )

    ca, na = agg(cob), agg(nw)
    print("\nMACRO per bus -- Σ stage-avg MW over 64 stages (MWmes)\n")
    hdr = f"{'bus':<10}{'src':<7}{'net_load':>11}{'thermal':>11}{'hydro':>11}{'excess':>11}{'deficit':>11}"
    print(hdr)
    print("-" * len(hdr))
    for bus in sorted(set(ca["bus"]) | set(na["bus"])):
        for src, a in (("NEWAVE", na), ("Cobre", ca)):
            r = a.filter(pl.col("bus") == bus)
            if r.height == 0:
                continue
            r = r.row(0, named=True)
            print(
                f"{bus:<10}{src:<7}{r['net_load']:>11,.0f}{r['thermal']:>11,.0f}"
                f"{r['hydro']:>11,.0f}{r['excess']:>11,.0f}{r['deficit']:>11,.0f}"
            )
        # delta row
        rc = ca.filter(pl.col("bus") == bus)
        rn = na.filter(pl.col("bus") == bus)
        if rc.height and rn.height:
            rc, rn = rc.row(0, named=True), rn.row(0, named=True)
            print(
                f"{'':<10}{'Δ C-N':<7}{rc['net_load'] - rn['net_load']:>11,.0f}"
                f"{rc['thermal'] - rn['thermal']:>11,.0f}{rc['hydro'] - rn['hydro']:>11,.0f}"
                f"{rc['excess'] - rn['excess']:>11,.0f}{rc['deficit'] - rn['deficit']:>11,.0f}"
            )
        print()


if __name__ == "__main__":
    main()
