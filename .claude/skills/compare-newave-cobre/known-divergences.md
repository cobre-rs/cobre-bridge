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
3. **Production function is LINEAR** (constant productivity). The bridge models hydro
   generation as `GH = ρ · Q` with constant productivity. Confirm the deck uses the linear
   model — dger's hydro production-function flag and `pmo.dat`'s `TIPO DE FUNCAO DE PRODUCAO`
   should read **LINEAR** (energies then use the static `canal de fuga médio` tailrace). If
   the deck uses **FPHA** (the head-dependent piecewise model with a tailrace polynomial),
   that is a different production model and head/tailrace differences are expected by
   construction.
4. **Simulation mode** — know whether it is deterministic or stochastic. In a **deterministic
   final simulation** (a single historical series; `num_forwards = num_aberturas = 1`,
   `tipo_simulacao_final` selecting one historical year) NEWAVE **ignores the PAR(p) inflow
   model** and both programs replay only that series — so the inflow model is **not** a
   divergence source, and the converter's forcing of inflow `max_order → 0` is a correct
   no-op. Treat a single-scenario run's results as exact realizations, not expectations.
5. **Deck drift** — run `stat -c '%y  %n' <case>/*.dat <case>/saidas/pmo.dat`. If any input
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
  R$/MWh). The curve constraint is on the **linear** stored energy
  `Σ ρ_acum · (storage − vmin)`, with `ρ_acum` converted to **MWmês/hm³** (the
  hm³↔(m³/s)·month factor ≈ 2.63 = 730·3600/1e6); both sides hold that linear energy at the
  curve. **Validate `VIOL_CAR` against this linear energy, not NEWAVE's reported nonlinear
  `EARMF`** (the head-dependent physical energy, a few percent lower mid-storage) — see
  _Comparator false positives_.
- **Penalty-conversion productivities**, matching `pmo.dat`'s applied values: flow/storage
  micro-penalties (VAZMIN / TURBMN / TURBMX / spillage / turbined) use the **mean equivalent
  productivity** (`PROD_MEDIA_SIN`); withdrawal (DESVIO) and evaporation use the **max
  accumulated productivity at maximum head** (`MAX_PRODTACUM_SIN`). These are **uniform
  system-wide**, not per-plant.
- **Per-plant hydro generation and productivity** — validate via the **operational**
  `GHIDUH / QTURUH` (NEWAVE) vs Cobre `equivalent_productivity_mw_per_m3s`, **not** the
  `pmo.produtibilidades_equivalentes` diagnostic table (which uses a different tailrace
  convention and will read a few percent off).
- **Per-plant thermal generation** — via **`GTERMTOT` (= `GTERM` + `GTMIN`)**, the SIN total;
  `GTERM` alone is generation _above_ must-run minimum and under-reports each plant.
- **Per-stage min-outflow / min-generation floors** — converted identically (Cobre min-outflow
  = `modif.dat` VAZMIN where present, else `hidr.dat` `vazao_minima_historica`, which NEWAVE
  also enforces); plants that violate on one side violate on the other.
- **Evaporation** — volume-coupled on both sides (mm coefficient × area(storage) via the
  cota-area-volume curve); the coefficients come from `hidr` `evaporacao_{MES}`. Evaporated
  _volume_ differs only when storage differs (a symptom, not a cause).
- **Exchange limits** (per-stage, from `sistema.dat`) match; the exchange penalty is a
  tie-breaker micro-penalty with no measurable cost footprint (recover it from the dual, not
  the cost — see gotchas).
- **Inflows and water conservation** (turbined + spilled). Inflows match; a turbine↔spill
  swap that conserves water is a reallocation, not a missing/extra quantity (see below).
- **Discounting** matches within rounding — Cobre discounts by **actual elapsed days**
  (`1/(1+r)^(Δdays/365.25)`), NEWAVE by nominal month/12; the per-stage factor difference is
  negligible (well under 0.1 %), not a divergence source.
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
- **Seeding.** The converter seeds the **real block-weighted committed MW** for the
  pre-horizon anticipated dispatch (it no longer writes zeros); honouring a non-zero seed
  requires `cobre-python ≥ 0.7.0`. So a first-stage GNL dispatch gap should be investigated as
  a **real** difference, not dismissed as the old zeroing artifact (the giveaway of a
  stale conversion is a converter warning about "writing zeros").

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

### Per-reservoir / per-bus storage allocation & per-plant water value (same-cost substitution)

- **Mechanism.** When water has no marginal value (surplus / wet year, CMO ≈ 0) or when many
  reservoirs in a cascade/REE share the **same** marginal value under a single aggregate-REE
  security curve, **which reservoir holds the energy is a free direction** of the LP. NEWAVE
  and Cobre place the same total storage — and the same total water value — in different
  reservoirs/buses. With _N_ reservoirs serving a common load the **system** marginal water
  value is determined but its **attribution** to the _N_ reservoirs has ~_N−1_ degenerate
  degrees of freedom the two solvers resolve differently.
- **It compounds with reservoir count.** A single self-contained cascade reproduces dispatch
  and per-plant water values almost exactly; as reservoirs are added the ill-determined
  per-plant duals accumulate into broad water-value scatter that nonetheless **cancels at the
  system level**. So per-plant water-value scatter in a full case is expected, not a bug.
- **Hotspots** are substitutable parallel tributaries / reservoirs shedding gen-capped surplus
  near a tight seasonal floor (`VMINT`), and reservoirs a water diversion (`dsvagua.dat`) parks
  barely above a floor — there the storage-balance dual is non-unique. Cobre breaks the tie by
  pivot order; per-plant `spillage_cost` (normally a single global value) can force-align it to
  NEWAVE's split but that is cosmetic dual-alignment, cost-neutral, not a fix.
- **Run-of-river plants** (`'D'`/`'S'`, useful volume ≈ 0, storage collapsed to a fixed level)
  have a **degenerate `water_value_per_hm3`** — there is no real storage to value. Do not read
  their per-hm³ water value as a divergence.
- **Confirm benign:** judge fidelity on **cost** (immediate + future) and on **constrained**
  quantities, not on where the water sits. Storage/water-value scatter is only concerning if it
  moves **total cost** or produces a **curve/VminOP violation on one side only**. Rule: do not
  read per-reservoir scatter as a conversion error while **CMO ≈ 0 and curve violation = 0 on
  both sides**. (Breaking the degeneracy requires genuine scarcity — nonzero CMO/deficit.)

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

## Localizing a divergence by case reduction

The strongest way to separate a **real converter difference** from **emergent dual
degeneracy** is to shrink the case until the divergence either vanishes or isolates.

- **The spot-price lens.** Spot price (CMO) exact ⟺ the **dispatch** matches (a real,
  cost-bearing agreement). Spot diverging ⟺ a genuine operational difference. Per-plant
  **water-value** scatter alone, with spot/generation exact, is almost always cost-neutral
  dual degeneracy (see the storage-allocation entry) — not a converter bug.
- **Reduce to a self-contained cascade or single REE** (one that has no cross-REE flows), with
  a hand-scaled load that keeps a meaningful scarcity regime (CMO > 0, not pinned). A faithful
  conversion makes a small case **exact** (spot 0.0 %, per-plant water values ratio ≈ 1.0); the
  full-case scatter then reveals itself as degeneracy that **compounds with reservoir count**,
  not a converter error. Remove one suspected degenerate plant (set it `NC`) and re-compare — if
  the rest snap to tolerance, that plant was a degenerate-dual hotspot.
- **Build gotchas when reducing a NEWAVE deck:**
  - The `MERCADO` load field in `sistema.dat` is **fixed-width, right-aligned** — editing it by
    substitution can silently truncate (e.g. `22000` → `2200`); splice into the exact columns.
  - A minimal deck can make NEWAVE **SIGABRT in its post-run teardown** ("double free") after it
    has already converged and written valid `pmo.dat`/`forward.dat`/`cortes.dat`. Common trigger:
    an **electric restriction (`restricao-eletrica.csv`) referencing plants absent** from the
    reduced case — replace it with an inert valid restriction (a present plant, ±1e30 limits;
    LIBS rejects a fully-empty file). Make `roda.sh` tolerate the crash (don't `set -e` through
    it) and run `nwlistop` manually to produce the `MEDIAS` files for `compare`.

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
- **Tie-breaker micro-penalties** (exchange, spillage, turbined) are buried in noise in the
  **cost** — recover and validate them from the **dual / marginal** instead (by KKT the
  relevant dual is bounded by the penalty, so its magnitude caps at the penalty value).
- **Energy excess reads zero?** Confirm `penalties.json` `excess_cost` wasn't bumped above its
  stock micro-penalty as a diagnostic — an inflated value **masks** real excess rather than
  fixing it (the excess is structural; see the Energy-excess entry).

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
- **FCF cuts** — the cost-to-go the policy _builds_; drop to these for a water-value / future-cost
  gap. Decoders and aligners live in `investigations/` (`newave_cut_investigation.py`,
  `cobre_cut_investigation.py`, `compare_cuts.py`, `compare_states.py`).
  - **NEWAVE:** `inewave.newave.Cortesh.read(cortesh.dat)` gives the header (`tamanho_corte`,
    last-cut index per stage). Then `Cortes.read(cortes.dat, tamanho_registro=<tamanho_corte>,
indice_ultimo_corte=…, codigos_rees=[]  (EMPTY ⇒ the individualized per-plant branch),
codigos_uhes=<hydro codes>, codigos_submercados=[…], ordem_maxima_parp=…,
numero_patamares_carga=…, lag_maximo_gnl=<from adterm, 0 if anticipation off>)` → a frame of
    `rhs` + `pi_varm_uhe{code}` (∂FCF/∂storage, per **useful** volume) + `pi_qafl…` (inflow-lag
    duals; **0** in deterministic mode). Record = 4 int32 + N float64, `N = n_uhes·(parp_order+1)+2`.
    Filter `iteracao_construcao > 0` for real cuts.
  - **Cobre:** `flatc -t ~/git/cobre/crates/cobre-io/schemas/policy.fbs --root-type StageCuts
--raw-binary -- output/policy/cuts/stage_NNN.bin` → JSON `cuts[]{intercept, coefficients[],
iteration, is_active}`; coefficient order follows `state_dictionary.json` (state index →
    hydro entity_id, storage in hm³).
  - **Units:** Cobre cut FCF/coef ×1e6 → R$. NEWAVE cut FCF ×(stage hours ≈ 730) → R$; its
    `pi_varm` is per useful-volume. **Compare the FCF as a function:** evaluate each model's full
    cut set at a **shared** storage state and compare the future-cost _values_ — a per-plant
    gradient compare is noisy because the binding cut sits at a different iteration / trial point
    on each side. **Stage map:** the cut/FCF offset differs from the results offset by one.
- **Per-plant NEWAVE generation:** thermal from `MEDIAS-USIT` (`GTMIN`/`GTERM`/`GTERMTOT`),
  hydro from `MEDIAS-USIH` (`VARMUH` is storage **relative to the operative minimum**; `VEVAPUH`
  /`VRETIRUH` are monthly **volumes in hm³** — ÷ the ≈2.63 factor to compare against Cobre m³/s).
  With a single simulation series, MEDIAS equals the exact scenario. Stage map: MEDIAS stage =
  Cobre stage + offset (`nw_offset` = min stage in `MEDIAS-SIN`).

---

## Maintaining this file

When you confirm a new **class** of difference, add it to the right bucket — _should match_,
_always differs_, or _bug signal_ — with its **mechanism** and how to tell the benign case
from the bug. Keep it general (a pattern other cases will hit), not a report of one run.
