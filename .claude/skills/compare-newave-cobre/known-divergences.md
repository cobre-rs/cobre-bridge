# NEWAVE vs Cobre — what to compare, what should match, what always differs

A generic reference for judging a NEWAVE↔Cobre comparison. It answers three questions:

- **What to compare** and in what order (the procedure lives in `SKILL.md`; this is the
  expectation layer).
- **What should be equal** — a divergence here is a likely **bug**, worth tracing to a
  converter/reader.
- **What always differs by construction** — divergences here are **expected**; explain and
  move on, do not chase them.

This is a catalogue of _classes_ of difference and their mechanisms, not a log of any
particular run. Concrete magnitudes always depend on the case — re-derive them with
`cobre-bridge compare`.

---

## Prerequisites — confirm before reading any divergence

A faithful conversion will still diverge if the NEWAVE deck is not set up for an
apples-to-apples final simulation. Check these first; if one is off, the divergence is a
**deck-config artifact, not a converter problem**.

1. **Security curve uses FIXA** (`curva.dat` `TIPO DE PENALIZACAO = 0`). Cobre-bridge models
   the VminOP curve as a per-stage slack penalized at the fixed curve cost, which **is** the
   equivalent of NEWAVE's FIXA. It does **not** reproduce NEWAVE's iterative/variable
   (`ETAPA-2`) penalization — the converter logs a warning when the deck selects a non-FIXA
   mode (`_warn_if_non_fixa_penalization`).
2. **Preventive rationing is ON in the final simulation** (`dger.dat`
   `RACIONAMENTO PREVENT. = 1`, i.e. `CONSIDERA`; NEWAVE's default is `0`). This is a
   **NEWAVE final-simulation setting, not something the bridge can change.** With it off,
   NEWAVE's final simulation ignores the preventive-rationing signal embedded in its own
   policy and **drains reservoirs to the floor to avoid taking deficit** — Cobre instead
   follows the converted policy (hold storage at the security level, take deficit/rationing
   when scarce), so the two diverge sharply even though the policy is faithfully converted.
   With it on, NEWAVE's final simulation holds and deficits like Cobre and the two agree to
   within the structural differences below.
3. **Deck drift** — run `stat -c '%y  %n' <case>/*.dat <case>/saidas/pmo.dat`. If any input
   `.dat` is **newer than `saidas/pmo.dat`**, the deck was edited after the run that produced
   the outputs, so the converted Cobre case and the NEWAVE outputs may come from different
   decks. Cross-check `pmo.dat`'s config echo (anticipation flag, plant counts, sim type,
   FPHA, POS) against the current `.dat` files before trusting anything.

> **Common misdiagnosis.** The "NEWAVE drains while Cobre holds and deficits" symptom is the
> `RACIONAMENTO PREVENT. = 0` artifact above — **not** a VminOP-penalty formulation problem.
> The bridge's per-stage curve slack is correct (= FIXA); there is no per-stage-vs-once
> penalty bug to fix.

---

## What should be equal (a divergence here is a likely bug)

Trace any real mismatch in these to a converter (`src/cobre_bridge/converters/`) or comparator
(`comparators/`) — comparator reimplementations have produced false positives before (see
_Comparator false positives_).

- **LP variable bounds** — the feasible region. Compared by `compare bounds` (absolute
  tolerance). If the bounds already differ, downstream result differences follow; check
  bounds first when a result divergence is unexplained.
- **Load / net load / non-controllable generation** — trusted by construction. Do **not**
  spend triage effort distrusting them; focus on the dispatchable side.
- **Security-curve levels** (% per REE) and the **curve penalty cost** (the curva CUSTO in
  R$/MWh). Both sides should hold the **linear** stored energy (`Σ ρ_acum · storage`, base
  `vmin`) at the curve.
- **Penalty-conversion productivities**, matching `pmo.dat`'s applied values: flow/storage
  micro-penalties (VAZMIN / TURBMN / TURBMX / spillage / turbined) use the **mean equivalent
  productivity** (`PROD_MEDIA_SIN`); withdrawal (DESVIO) and evaporation use the **max
  accumulated productivity at maximum head** (`MAX_PRODTACUM_SIN`).
- **Per-plant hydro generation and productivity** — validate via the **operational**
  `GHIDUH / QTURUH` (NEWAVE) vs Cobre `equivalent_productivity_mw_per_m3s`, **not** the
  `pmo.produtibilidades_equivalentes` diagnostic table (which uses a different tailrace
  convention and will read a few percent off).
- **Per-plant thermal generation** — via **`GTERMTOT` (= `GTERM` + `GTMIN`)**, the SIN total;
  `GTERM` alone is generation _above_ must-run minimum and under-reports each plant.
- **Per-stage min-outflow / min-generation floors** — converted identically; plants that
  violate on one side violate on the other.
- **Inflows and water conservation** (turbined + spilled). Inflows match; a turbine↔spill
  swap that conserves water is a reallocation, not a missing/extra quantity (see below).
- **Total operation cost** — within the structural band of the next section. Compare via the
  `pmo.dat` NPV breakdown and per-stage `COPER` vs Cobre `immediate_cost`; do **not** read
  `MEDIAS-SIN` `DEFT`/`CDEF` (see gotchas).

---

## What always differs by construction (expected — explain, don't chase)

Each item: the mechanism, why it is irreducible, and how to confirm the benign cause.

### Stage length and discounting

- **Mechanism.** Cobre uses each stage's **actual month length** (28–31 days; ~672–744 h) for
  block hours and for discounting by elapsed days. NEWAVE uses a **fixed 730 h** per stage and
  a fixed m³/s→hm³ factor (≈ 2.63) every month.
- **Effect.** Per-stage nominal costs differ slightly (roughly ±a couple of percent on a
  stage), because the same power over a different number of hours is a different energy/cost.
  The m³/s↔hm³ factor difference is in **both** the VminOP LHS and RHS (and in generation vs
  turbined flow), so it **cancels in the water value** — it shifts per-stage magnitudes, not
  the dispatch.
- **Confirm benign:** the discrepancy tracks the month-length ratio and disappears in
  discounted/normalized terms; it does not localize to a particular entity.

### GNL (anticipated thermal) — granularity and cost accounting

- **Granularity.** The anticipated GNL thermal dispatch is represented at different
  resolutions — cobre-bridge/Cobre carry it **per plant and per stage**, NEWAVE resolves it
  **per subsystem and per load block (patamar)**. The two therefore cannot produce a
  bit-identical anticipated GNL dispatch; expect a small thermal-timing difference on GNL
  plants in their lead stages.
- **Accounting (where the fuel is booked).** Cobre charges anticipated GNL fuel on the
  **decision column at the decision stage, discounted to delivery**
  (`fill_anticipated_decision_objective`) and **zeroes the delivery-stage generation cost**
  (`zero_anticipated_delivery_thermal_cost`). NEWAVE books it in **`CTERM` at delivery**,
  nominal, in the plant's submarket. So NEWAVE `CTERM` runs **above** Cobre `thermal_cost` by
  exactly the GNL fuel, even when thermal _generation_ is identical.
- **Confirm benign / how to compare:** total `immediate_cost` includes the GNL fuel (it lands
  in the unattributed remainder, not `contract_cost`); compare **`immediate_cost`**, not the
  thermal category. The gap localizes to exactly the GNL submarkets/plants. Per-stage it is
  shifted by `lead_stages` (Cobre at commitment, NEWAVE at delivery) and discounted.

### Energy excess and turbine↔spill reallocation

- **Mechanism.** The micro-penalty merit order that should make spilling cheaper than
  turbining-into-excess sits several orders of magnitude below the dominant LP costs
  (thermal/deficit and the FCF cut gradients), **below the solver's effective optimality
  tolerance.** The solver is then numerically indifferent between spill and
  turbine-into-excess and picks arbitrarily — Cobre may turbine a run-of-river surplus that
  NEWAVE spills and dump the result as zero-priced `excess_mw`.
- **Where it shows.** Concentrated where non-controllable generation exceeds local load
  (negative net load) and at fictitious nodes; the per-plant signature is mirror-image
  `turbined_m3s`/`spillage_m3s` with water conserved.
- **Confirm benign:** it is **cost-invisible** (excess priced at the micro-penalty).
  Tightening the Cobre solver feasibility tolerance (or scaling the spill/turbine/excess
  micro-penalties up together) removes it. Do **not** mask it by inflating `excess_cost`.
- **Caveat:** because it is near-zero-priced it is **invisible to the cost breakdown** —
  always check `excess_mw` on the Energy Balance tab even when total cost matches.

### Per-reservoir / per-bus storage allocation (same-cost substitution)

- **Mechanism.** When water has no marginal value (surplus / wet year, CMO ≈ 0) or when many
  reservoirs in a cascade/REE share the **same** marginal value under a single aggregate-REE
  security curve, **which reservoir holds the energy is a free direction** of the LP. NEWAVE
  and Cobre place the same total storage in different reservoirs/buses.
- **Confirm benign:** judge fidelity on **cost** (immediate + future) and on **constrained**
  quantities, not on where the water sits. Storage scatter is only concerning if it moves
  **total cost** or produces a **curve/VminOP violation on one side only**. Rule: do not read
  per-reservoir storage scatter as a conversion error while **CMO ≈ 0 and curve violation = 0
  on both sides**. (Breaking the degeneracy requires genuine scarcity — nonzero CMO/deficit.)

### Withdrawal-violation placement (conserved cascade-wide)

- **Mechanism.** Along a water-withdrawal (diversion) cascade the **total** under-delivery is
  conserved — it matches NEWAVE to the digit and the penalty is the same — but the two place
  the conserved shortfall differently. Cobre's under-withdrawal slack is unbounded per plant
  (`fill_withdrawal_slack_columns`), so the LP can dump the cascade shortfall as a slack (a
  water source) at **upstream run-of-river** plants and turbine that relief water; NEWAVE's
  shortfall is bounded per plant and lands at the **downstream** withdrawal plant.
- **Confirm benign:** inflows and spillage match exactly (not a turbine↔spill issue); the
  per-stage cascade total and its penalty cost match. It is cost-neutral redistribution.

### Water value / FCF cut gradients (trial-point degeneracy)

- **Mechanism.** The two policies' cuts are **tangent at different trial points** — the
  forward passes visit different storages, so different (usually small) reservoirs bind. A
  per-reservoir cut-gradient mismatch is a _consequence_ of any dispatch difference, not an
  independent cut-construction bug. In surplus the per-reservoir water value is a flat
  direction, so gradients differ harmlessly.
- **Confirm benign:** compare the FCF **as a function** — evaluate each model's full cut set
  `FCF(x) = maxₖ(intercept_k + Σ coef_k·x)` at a shared storage point and compare the
  resulting future-cost _values_. Confirm state-space alignment first (storage gradients are
  apples-to-apples only when both models are individualized; the cut/FCF stage offset differs
  from the results offset by one).

### Converter configuration defaults

These are deliberate converter choices that shape the comparison, not bugs: inflow
non-negativity defaults to `truncation_with_penalty`; cut selection defaults to `lml1` with
`memory_window = 0`. If a divergence tracks one of these, it is a configuration difference.

---

## What signals a real bug (concerning patterns)

- **Total cost diverges beyond the structural band _and_ localizes** to one entity or one
  class of stage — the localization is the strongest root-cause clue.
- **A curve/VminOP violation on one side only**, with the other sitting on the curve.
- **A whole entity shifted** (ID-mapping / off-by-one): NEWAVE's 1-based codes vs Cobre's
  dense 0-based indices are remapped by `NewaveIdMap`; a systematic per-entity offset points
  here.
- **Fictitious-plant leakage** — accounting-only entities that should be filtered out
  appearing as real dispatch.
- **Post-study (pós-estudo) artifacts** — converters extend the study horizon by last-year
  seasonal repetition; this construction has carried real bugs (e.g. water-withdrawal
  post-study). A divergence that appears only in post-study stages and tracks the
  extrapolation is suspect.

### Comparator false positives — check the comparator before the converter

When a divergence looks structural, suspect the comparison code before the converter. Past
false positives:

- A **reader defaulting a layout parameter** that was legitimately zero (e.g. a GNL lag
  count), misaligning every derived coefficient. Rule: never default a legitimate `0` to a
  nonzero — validate against the file.
- A comparator **plotting a different quantity on each side** — e.g. NEWAVE's reported
  _nonlinear_ stored energy (`EARMF`) against Cobre's _linear_ curve energy, producing a
  spurious "NEWAVE below the bound without penalty." The curve constraint is linear on both
  sides; the converter was correct.
- A docstring claiming an **apples-to-apples pairing that isn't** (e.g. `CTERM` ↔
  `thermal_cost`, which differ by the GNL fuel).

---

## Magnitude and reading gotchas

- **Tolerances differ by comparison.** Bounds tolerance is **absolute**; results tolerance is
  **relative**. State which value you used — a "match" at `1e-1` is not a match at `1e-3`.
- **Sort by |Δ|, not count.** One large structural divergence outweighs many near-tolerance
  ones; a large _relative_ error on a tiny quantity is usually noise — check absolute scale
  and `WithinTol`/`sMAPE`.
- **Cost agreement ≠ operation agreement.** Near-zero-priced quantities (excess, spill/turbine
  split) are invisible to the cost breakdown — check operation results too.
- **Deficit:** do **not** use `MEDIAS-SIN` `DEFT`/`CDEF` — they are mis-scaled, much smaller
  than the true deficit (a NEWAVE reporting-unit artifact). Use the **pmo.dat NPV breakdown**
  plus per-stage **`COPER`** vs Cobre **`immediate_cost`** (note `COPER` is in 10⁶ R$,
  `immediate_cost` in R$ — a match shows as a ratio ≈ 1e6).
- **Per-plant thermal:** `GTERMTOT`, not `GTERM`. **Thermal cost:** `CTERM` ≠ `thermal_cost`
  (GNL) — use `immediate_cost`.
- **Productivity:** validate via operational `GHIDUH/QTURUH`, not the pmo
  `produtibilidades_equivalentes` reference table.
- **Cobre per-block parquets** (hydro/thermal sim) are **per load block** — always weight by
  `stages.json` `blocks[].hours` (or use `generation_mwh / Σ hours`). Unweighted block means
  fabricate false "Cobre excess / phantom water" findings.

---

## Reference — decode recipes (generic tooling)

- **`pmo.dat` is the penalty/cost ground truth** (the _applied_ penalties and the NPV cost
  breakdown). Prefer it over reconstructions.
- **`forward.dat`** — NEWAVE's binary forward dump, full precision per scenario
  (`custo_operacao`, `custo_geracao_termica`, every physical violation quantity). Decode dims
  from `forwarh.dat` (`tamanho_registro`, series count, REE/submarket/patamar counts);
  `n_stages = filesize / record_size / n_series`. **Per-plant ordering is critical:** pass
  exactly the simulated hydro plants (`confhd` existing `EX`, minus `FICT.*`) — extra rows
  corrupt every per-plant field after the hydro block. Sum per-patamar contributions for the
  stage value. `custo_geracao_termica` freezes at the last study month through the post-study
  tail, so `op − thermal` only isolates non-thermal cost _within_ the study horizon.
- **FCF cuts** — NEWAVE `cortes.dat`/`cortesh.dat` (via `inewave`) vs Cobre
  `output/policy/cuts/stage_*.bin` (flatbuffer). Pin units to R$ (NEWAVE and Cobre use
  different monetary scales). Confirm state-space alignment before reading a per-reservoir
  mismatch (storage gradients are R$/hm³ on both sides only when both are individualized;
  NEWAVE `pi_varm` is w.r.t. _useful_ volume).
- **Per-plant NEWAVE generation:** thermal from `MEDIAS-USIT` (`GTMIN`/`GTERM`/`GTERMTOT`),
  hydro from `MEDIAS-USIH`. With a single simulation series, MEDIAS equals the exact scenario.
  Stage map: MEDIAS stage = Cobre stage + offset (`nw_offset` = min stage in `MEDIAS-SIN`); the
  cut/FCF offset differs from the results offset by one.

---

## Maintaining this file

When you confirm a new **class** of difference, add it to the right bucket — _should match_,
_always differs_, or _bug signal_ — with its **mechanism** and how to tell the benign case
from the bug. Keep it general (a pattern other cases will hit), not a report of one run.
