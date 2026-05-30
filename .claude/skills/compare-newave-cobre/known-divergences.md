# Known Divergences: NEWAVE vs Cobre

Accumulated catalogue of differences seen when comparing a NEWAVE run against its
cobre-bridge conversion. **Check here first** during triage (Step 3 of the skill).

For each entry, record the *symptom* (what you observe in the diff/report), where
it appears, the *root cause*, and the *verdict*:

- **expected** — a consequence of conversion or modelling semantics; safe to accept.
- **concerning** — investigate as a likely bug.

> How to use: when you confirm a new cause, add an entry under **Confirmed
> divergences**. When a verdict changes (e.g. a bug gets fixed), update the entry
> and note the commit/PR.

---

## Entry template

```
### <short symptom name>
- **Symptom:** what you see in the comparison output
- **Where:** bounds / results; which variable, entity, stage range
- **Root cause:** the conversion or modelling reason
- **Verdict:** expected | concerning
- **Evidence/ref:** commit, PR, file path, or [[memory-name]] link
```

---

## Seed topics (from recent git history — DETAILS/VERDICTS TODO)

These three topics surfaced in recent commits and are likely recurring discussion
points. They are starting points only — **the interpretation is not yet filled
in. Confirm and complete each before relying on it.**

### Withdrawal slack pos/neg labels

- **Symptom:** _TODO — what does the diff/report show?_
- **Where:** results (slack variables); _TODO entity/stage_
- **Root cause:** NEWAVE and Cobre label the withdrawal slack with opposite
  pos/neg sign; the comparator was realigned to the NEWAVE convention (per the
  commit below). _TODO: confirm exact semantics._
- **Verdict:** _TODO (expected once labels aligned?)_
- **Evidence/ref:** commit `e6e6601` — "fix(compare): align withdrawal slack
  pos/neg labels with NEWAVE convention"

### Inflow non-negativity handling

- **Symptom:** _TODO_
- **Where:** _TODO (inflow scenarios / results)_
- **Root cause:** converter default for inflow non-negativity is
  `truncation_with_penalty`. _TODO: how/whether this makes Cobre diverge from
  NEWAVE's treatment._
- **Verdict:** _TODO_
- **Evidence/ref:** commit `3a70e5e` — "chore(converter): default inflow
  non-negativity to truncation_with_penalty"

### Cut selection (lml1, memory_window=0)

- **Symptom:** _TODO_
- **Where:** policy / convergence / downstream results
- **Root cause:** converter default cut selection is `lml1` with
  `memory_window=0`. _TODO: relationship to NEWAVE's cut handling._
- **Verdict:** _TODO_
- **Evidence/ref:** commit `c047060` — "chore(converter): default cut selection to
  lml1 with memory_window=0"

---

## Observed — under investigation

### Energy excess in Cobre (zero-priced) — invisible to cost

- **Symptom:** Cobre shows **9844 MW-stage of energy excess** (up to 2736 MW at a
  single bus-stage, 14 stages, all 5 buses) while NEWAVE `EXCESSO` is **0**. It is
  **cost-invisible**: Cobre's `excess_cost` ≈ 2240 R$ total NPV (≈0 in 10⁹ R$), so
  a cost-first pass misses it entirely.
- **Where:** results / Energy Balance tab; Cobre `excess_mw` (bus_aggregates).
  Largest source is **bus 4** (the FC / 5th submarket — 0 load, no generation: 2736
  @ s19, 1014 @ s20, an exchange/fictitious-node artifact); plus **NCS-driven**
  buses where non-controllable generation exceeds local load (bus 2/NE: NCS ~20000
  MW, net_load negative → un-exportable surplus dumped as excess).
- **Root cause:** _under investigation._ Cobre dumps surplus energy as ~free excess
  where NEWAVE has none — suspect (a) FC/fictitious-submarket exchange routing and
  (b) the excess penalty price (`excess_cost`) being ~0. Sits at the network /
  exchange + excess-pricing layer (not the trusted NCS/load conversion). May share
  a cause with the turbine/spill reallocation (both = "Cobre has surplus energy it
  doesn't use").
- **Verdict:** concerning — investigate.
- **Evidence/ref:** example case, 2026-05-30. (Corrects an earlier mis-read that
  excess was a non-issue — it is zero in _cost_, not in _energy_.)

### Turbine↔spill reallocation at Madeira run-of-river (Jirau, Sto Antônio)

- **Symptom:** `turbined_m3s` and `spillage_m3s` diverge strongly (WithinTol 21% /
  41%, sMAPE 67% / 97%) but are mirror images: Σ|Δturb|=1.74M, Σ|Δspill|=1.43M, yet
  Σ|Δ(turb+spill)| = Σ|Δoutflow| = 812k (offset ratio 0.26) — water conserved.
  Concentrated at JIRAU (turbined 13879→4702, spillage 1732→10909, NW→CB) and STO
  ANTONIO: **NEWAVE turbines the Madeira flow, Cobre spills it.**
- **Where:** results; `turbined_m3s` / `spillage_m3s`, Madeira run-of-river plants.
- **Root cause:** _likely alternative optima._ Total cost matches to 0.1%, so this
  is a cost-neutral spatial turbine/spill reallocation under hydro surplus (turbine
  here vs spill-here-and-turbine-elsewhere are tied). Check whether a turbine-
  preference tie-break (spill only once turbines are full) aligns Cobre to NEWAVE.
- **Verdict:** likely expected (degenerate) — confirm via a tie-break test.
- **Evidence/ref:** example case, 2026-05-30 §3b operation analysis.

### Water value / FCF gap — cut coefficients, seeded at stage 0

- **Symptom:** `water_value_per_hm3` barely matches (WithinTol 0.9%, r 0.76); some
  reservoirs large-negative in NEWAVE but ~0 in Cobre (Sao Roque, Garibaldi,
  Manso, Batalha), Caconde reversed, 41 Cobre zeros. At **stage 0** — identical
  initial state + inflow, matching immediate cost (235 vs 233) — Cobre's **future
  cost is ~0.44 bi R$ below NEWAVE's** (23.38 vs 23.82), also visible as the
  `lower_bound` divergence. Storage redistributes from t0; largest single t0
  dispatch gap is ITAIPU (−2054 MW).
- **Where:** results; `water_value_per_hm3`, `lower_bound`, and storage/generation
  from stage 0 onward.
- **Root cause:** _under investigation._ **Cut selection is disabled in both
  models for this run**, so this is NOT a selection effect (corrects an earlier
  note) — the gap is in the cut **coefficients/representation** (the FCF Cobre
  builds vs NEWAVE), seeded at stage 0. Leading suspects: the aggregated (NEWAVE
  per-REE energy) vs individual (Cobre per-reservoir storage) state-space mapping,
  the backward-pass scenario setup, or cut scaling. To be probed with **1-iteration**
  runs of both models + direct cut-coefficient comparison.
- **Verdict:** concerning — investigate (primary systematic driver of the hydro
  decisions, above the converter tie-breaks).
- **Evidence/ref:** example case, 2026-05-30 stage-0 analysis.

## Resolved

### Hardcoded Belo Monte diversion addition → VZMIN penalty + Pimental/Belo Monte routing

Two linked symptoms, one root cause.

- **Symptom:** (1) total operation cost (NPV) was 10.58 bi R$ (NEWAVE) vs 9.60 bi
  (Cobre), a −9% gap, ~99% of which was the minimum-outflow violation penalty
  (Cobre `outflow_violation_below` 0.014 bi vs NEWAVE `VIOLACAO VZMIN` 0.969 bi);
  (2) at the Belo Monte complex Cobre kept only ~281 m³/s in the PIMENTAL riverbed
  (vs NEWAVE 1655) and routed the rest through BELO MONTE (~4825 vs 3451), totals
  matching (~5106).
- **Where:** results; cost breakdown, per-stage immediate cost (stages 11–16),
  `outflow_m3s` at the Belo Monte complex.
- **Root cause:** a **hardcoded diversion addition** for the Belo Monte complex in
  `converters/hydro.py` mis-routed water — keeping Pimental's riverbed artificially
  low so Cobre dodged the Volta Grande minimum-outflow violations NEWAVE incurs.
- **Resolution:** removed the hardcoded diversion addition and re-ran. Cobre now
  keeps 1655 m³/s in Pimental (exact match) and incurs the VZMIN penalty like
  NEWAVE (1.002 vs 0.969 bi); **total cost agrees to +0.1%**. `compare bounds`
  stayed 100% clean throughout.
- **Verdict:** **resolved** (was concerning).
- **Evidence/ref:** example case, 2026-05-30 before/after cost-first analysis; fix
  in `converters/hydro.py`.

## Confirmed divergences

_(none yet — add entries here, using the template above, as you confirm them)_
