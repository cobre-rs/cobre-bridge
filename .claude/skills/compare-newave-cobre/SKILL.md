---
name: compare-newave-cobre
description: Compare a NEWAVE run against its cobre-bridge-converted Cobre case and explain the divergences. Use when investigating why NEWAVE and Cobre results differ — LP bounds, simulation results, operating costs, generation, storage, exchanges, slacks — or when reviewing a comparacao.html report. Trigger phrases: compare results, NEWAVE vs Cobre, why do results differ, bounds mismatch, divergence, comparacao.
effort: medium
argument-hint: "<newave_dir> <cobre_output_dir> [--what=bounds|results|both] [--focus=<variable>]"
disable-model-invocation: false
allowed-tools: [Bash, Read, Write, Grep, Glob]
---

# Compare NEWAVE vs Cobre

Investigate and explain differences between a NEWAVE run and the Cobre case
produced from it by cobre-bridge. The mechanical comparison is done by the
`cobre-bridge compare` CLI; **this skill is the interpretation layer** — how to
run it, what to look at first, which divergences are expected vs concerning, and
how to report findings.

## When to use

- "Why do the results differ between NEWAVE and the converted case?"
- A `compare bounds` or `compare results` run shows mismatches and you need to
  decide whether they are expected (conversion semantics) or a real bug.
- Reviewing a `comparacao.html` report.

## Inputs

Parse from the request; ask only if a path is genuinely unknown.

- `newave_dir` — NEWAVE case dir (must contain `saidas/` for results compare).
  Example: `example/newave_rodada`.
- `cobre_output_dir` — Cobre **output** dir containing
  `training/dictionaries/bounds.parquet`. Example: `example/cobre_rodada/output`.
- `--what` — `bounds`, `results`, or `both` (default: both).
- `--focus` — narrow to a variable/entity (e.g. `storage_min`, a specific REE/hydro).

## Step 1 — Decide what to compare

Two independent comparisons, with **different tolerance semantics**:

| Comparison        | Checks                                              | Tolerance                | Prerequisite                                              |
| ----------------- | --------------------------------------------------- | ------------------------ | --------------------------------------------------------- |
| `compare bounds`  | LP variable bounds (cobre-bridge-computed vs Cobre) | **absolute**, def `1e-3` | `<cobre_output_dir>/training/dictionaries/bounds.parquet` |
| `compare results` | Published NEWAVE results vs Cobre simulation output | **relative**, def `1e-2` | NEWAVE `saidas/` + Cobre `simulation/`                    |

> ⚠️ Bounds tolerance is **absolute**, results tolerance is **relative** — do not
> conflate them when judging magnitude.

Start with **bounds** when a result divergence is unexplained: if the LP feasible
region already differs, downstream result differences follow. Bounds are a
cheaper, more localized signal than full simulation results.

## Step 2 — Run the comparison

See `commands.md` for copy-paste commands against the example case. General form:

```bash
# Bounds: exit 0 = no mismatch, exit 1 = mismatches. Add --output for a Parquet diff.
cobre-bridge compare bounds <newave_dir> <cobre_output_dir> [--variables a,b] [--summary]

# Results: writes HTML with --output/-o. Always exits 0.
cobre-bridge compare results <newave_dir> <cobre_output_dir> -o report.html
```

Set `--tolerance` deliberately and **state which value you used** in the report —
a "match" at `1e-1` is not the same claim as a match at `1e-3`.

## Step 3 — Triage divergences

For a **results** comparison, always begin at the cost and work **macro → micro**
(§3a) — the cost tells you _what_ diverged and _where_ to drill in before you open
any per-entity output. For a **bounds** comparison, skip to the general principles
(§3c).

### 3a. Cost-first — the results entry point

1. **What diverged — Overview tab → Cost Breakdown (NPV).**
   `cost_breakdown_chart` / `cost_breakdown_table` (charts.py) show each cost
   category (thermal, deficit, penalties, …) for NEWAVE vs Cobre in 10⁹ R$, with a
   Δ / Δ% table **sorted by |Δ|**. Read which components differ — that is **what
   to look for** downstream.
2. **Where + composition — Per-Stage Cost → Immediate.**
   `immediate_cost_chart` (NEWAVE `COPER` vs Cobre `immediate_cost`, per 0-based
   stage, 10⁶ R$). Take the **stages with the largest cost differences**, and at
   those stages work out the **cost composition** — which component (thermal,
   deficit, penalties) drives the gap — by cross-referencing the NPV categories
   (§3a.1) with the operation variables at those stages (§3b). Then **filter the
   simulation outputs to those stages** for deeper investigation. (There is no
   single per-stage cost-by-category chart; this is a cross-reference.)
3. **If the stage-level cause isn't clear — Per-Stage Cost → Future.**
   `future_cost_chart` (NEWAVE `CUSTO_FUTURO` vs Cobre `future_cost`). Use it to
   compare the two models' **behavior / shape**, not to pinpoint stages: is the
   future-cost function broadly consistent, or off by **orders of magnitude** (a
   red flag for a structural problem)?

> Cost breakdown is discounted NPV; per-stage costs are nominal per stage. NEWAVE
> stages are offset to Cobre's 0-based `stage_id` (`nw_offset` = min stage in
> MEDIAS-SIN).

> ⚠️ **Cost agreement ≠ operation agreement.** Near-zero-priced quantities are
> invisible to the cost breakdown — notably **energy excess** (Cobre can dump
> surplus energy at ~zero `excess_cost` while NEWAVE has none). Always run §3b and
> check `excess_mw` on the Energy Balance tab even when total cost matches.

### 3b. Operation results — system → bus

After the cost comparison is exhausted, move to operation results, still
**macro → micro**:

1. **System-level (aggregate / SIN totals) first.** Compare NEWAVE vs Cobre on the
   dispatchable quantities to see where they diverge:
   - **Deficit** → System tab (`system_comparison_chart`, `deficit_mw`).
   - **Thermal generation** → Thermal Operation tab (`thermal_generation_chart`).
   - **Hydro generation** (plus storage / spillage / turbined / inflow / water
     value) → Hydro Operation tab → **System Totals (SIN)**
     (`hydro_aggregate_chart`).
   - **Penalty / violation slacks** → Hydro Operation tab → **Hydro Slacks (SIN)**
     (`hydro_slack_aggregate_chart`).
   - **Energy excess / load balance** → Energy Balance tab.
2. **Then localize at bus level.** For whichever variable diverges (e.g. hydro
   generation), drop to the per-bus view to find **which bus(es)** diverge — the
   pivotal macro → micro step:
   - Hydro by bus → Hydro Operation tab "… by Bus" (`hydro_per_bus_chart`); slacks
     by bus (`hydro_slack_per_bus_chart`).
   - Deficit / spot price by bus → System tab (`system_per_bus_chart`).

   Don't spend this effort on **load** or **non-controllable generation** — those
   are trusted (see Conventions & gotchas).

### 3c. General principles (both comparisons)

- **Check known divergences first** (`known-divergences.md`) — if the symptom
  matches an entry, classify it and move on; don't re-derive it.
- **Sort by magnitude, not count.** One large structural divergence outweighs many
  near-tolerance ones; for results, large relative error on a tiny quantity may be
  noise — check absolute scale and `WithinTol` / `sMAPE`.
- **Localize.** One REE / hydro? First stage only? Post-study stages? Localization
  is the strongest root-cause clue (e.g. post-study extrapolation, a mis-mapped ID).
- **Trace to source.** Map the symptom to a converter (`src/cobre_bridge/converters/`)
  or a comparator function (`comparators/results.py`, `bounds_from_inputs.py`,
  `{newave,cobre}_readers.py`). Comparator reimplementations have caused false
  positives before — see `known-divergences.md`.
- **For a water-value / future-cost (FCF) gap, drop below the simulated results to
  the cuts themselves.** The simulated `water_value_per_hm3` is a dual; the FCF Cobre
  _builds_ lives in `output/policy/cuts/stage_NNN.bin`. The repo-root WIP scripts
  `compare_cuts.py` / `cobre_cut_investigation.py` / `newave_cut_investigation.py`
  decode and align a single cut per side (by reservoir name, units pinned to R$) so
  you can compare coefficients and intercepts directly. **Beware the state space:**
  NEWAVE's cut gradient is w.r.t. REE-aggregated energy, Cobre's w.r.t. individual
  storage — confirm they're apples-to-apples before reading a per-reservoir mismatch
  as a bug. See the "Water value / FCF gap" entry in `known-divergences.md`.

## Step 4 — Report

- State **what** diverged, **by how much** (with the tolerance/units used), and
  **where** (entities/stages).
- Classify each finding: **expected** (conversion semantics — cite the entry in
  `known-divergences.md`) vs **concerning** (likely bug — name the suspect
  converter/reader).
- When a new root cause is confirmed, add it to `known-divergences.md`; if it is a
  durable fact, also save a memory and link it.

## Reference files

- `commands.md` — verified copy-paste commands for the example and production cases.
- `known-divergences.md` — accumulating symptom → cause → verdict catalogue. This
  is the heart of the skill; **check it first and keep it current.**

## Conventions & gotchas

Recurring rules for how we run these comparisons (add as they come up):

- **Trust load and non-controllable generation.** The conversion of demand/load
  and non-controllable (non-dispatchable) generation is considered correct — do
  **not** spend triage effort distrusting them. Focus on the dispatchable side:
  thermal and hydro dispatch, deficit, penalties/slacks, exchanges. (Load /
  net-load lives on the Energy Balance tab.)

<!-- More to come — e.g. variables to ignore, trusted tolerances, sign/label
conventions, quantities expected to differ by construction, aggregation levels
(SIN / REE / submarket / plant), example-vs-production differences. -->
