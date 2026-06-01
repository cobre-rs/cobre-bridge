# Known Divergences: NEWAVE vs Cobre

Accumulated catalogue of differences seen when comparing a NEWAVE run against its
cobre-bridge conversion. **Check here first** during triage (Step 3 of the skill).

Each entry records the _symptom_ (what you observe in the diff/report), _where_ it
appears, the _root cause_, and the _verdict_:

- **expected** — a consequence of conversion or modelling semantics; safe to accept.
- **concerning** — investigate as a likely bug.

> How to use: when you confirm a new cause, add an entry under the appropriate
> section. When a verdict changes (e.g. a bug gets fixed), update the entry and note
> the commit/PR.

> **Numbers are illustrative.** Entries describe patterns and mechanisms, not a
> specific run. Concrete magnitudes, plant names, and stage indices depend on the
> case and config — **re-derive them** (`cobre-bridge compare` + the relevant
> investigation script) rather than trusting any figure verbatim. In particular,
> before reading any excess number, check whether `penalties.json` carries a
> **diagnostic `excess_cost`** (it has historically been inflated to mask excess —
> see the Energy-excess entry).

---

## Entry template

```
### <short symptom name>
- **Symptom:** what you see in the comparison output
- **Where:** bounds / results; which variable, entity, stage range
- **Root cause:** the conversion or modelling reason
- **Verdict:** expected | concerning
- **Evidence/ref:** commit, file path, or [[memory-name]] link
```

---

## Seed topics (starting points — confirm before relying on)

These surfaced in git history and are likely recurring; the interpretation is not
yet filled in.

### Withdrawal slack pos/neg labels

- **Root cause:** NEWAVE and Cobre label the withdrawal slack with opposite pos/neg
  sign; the comparator was realigned to the NEWAVE convention. _Confirm exact semantics._
- **Evidence/ref:** commit "fix(compare): align withdrawal slack pos/neg labels".

### Inflow non-negativity handling

- **Root cause:** converter default is `truncation_with_penalty`. _Confirm how/whether
  this makes Cobre diverge from NEWAVE's treatment._
- **Evidence/ref:** commit "chore(converter): default inflow non-negativity to
  truncation_with_penalty".

### Cut selection (lml1, memory_window=0)

- **Root cause:** converter default cut selection is `lml1` / `memory_window=0`.
  _Confirm relationship to NEWAVE's cut handling._
- **Evidence/ref:** commit "chore(converter): default cut selection to lml1".

---

## Confirmed / resolved

### Energy excess in Cobre, and turbine↔spill reallocation — solver-tolerance artifact

- **Symptom:** Cobre dumps zero-priced energy `excess_mw` where NEWAVE has none, and
  it is cost-invisible (excess priced at a tiny micro-penalty). At the plant level the
  same effect shows as a turbine↔spill mismatch: `turbined_m3s` / `spillage_m3s`
  diverge as mirror images (water conserved) — **NEWAVE spills run-of-river surplus,
  Cobre turbines it** and dumps the resulting energy as excess. Concentrated where
  non-controllable generation exceeds local load (negative net load) and at fictitious
  nodes.
- **Where:** results; Energy Balance (`excess_mw`) and Hydro Operation
  (`turbined_m3s`/`spillage_m3s`), run-of-river plants.
- **Root cause:** the micro-penalty merit order that should make spilling cheaper than
  turbining-into-excess sits ~4–5 orders of magnitude below the dominant LP costs
  (thermal/deficit and especially the FCF cut gradients). At the default LP solver
  feasibility tolerance it falls **below the effective optimality tolerance**, so the
  solver is numerically indifferent between spill and turbine-into-excess and picks
  arbitrarily. Confirmed: **tightening the Cobre solver feasibility tolerance removes
  the excess and aligns the turbine/spill split with NEWAVE.**
- **Verdict:** **resolved (numerical).** The fix is the solver tolerance (or scaling
  the spill/turbine/excess micro-penalties up together so the ordering survives
  scaling) — **not** inflating `excess_cost`, which only masks it.
- **Evidence/ref:** [[project_cobre_excess_root_cause]],
  [[project_excess_penalty_diagnostic]]. Per-bus method:
  `investigations/compare_sem_intercambio.py`.

### Hardcoded hydro-complex diversion → spurious min-outflow advantage (resolved)

- **Symptom:** large operation-cost gap dominated by the minimum-outflow violation
  penalty; at one hydro complex Cobre kept too little water in a riverbed reach and
  routed the rest through the powerhouse, dodging a min-outflow violation that NEWAVE
  incurs.
- **Root cause:** a **hardcoded diversion addition** for a specific complex in
  `converters/hydro.py` mis-routed water.
- **Resolution:** removed the hardcoded addition; riverbed outflow now matches NEWAVE
  and Cobre incurs the same penalty. Total cost agrees closely.
- **Verdict:** **resolved** (was concerning).
- **Evidence/ref:** fix in `converters/hydro.py`.

### Tooling false-positive: NEWAVE cut parser defaulted the GNL lag count (fixed)

- **Symptom:** the NEWAVE↔Cobre cut comparison looked structurally disjoint
  (near-zero correlation, storage value attributed to the wrong reservoirs, a large
  phantom `pi_gnl` "thermal-anticipation" dual). Drove a long-lived "the cut
  construction is fundamentally different" hypothesis.
- **Where:** **the investigation tooling, not the models** —
  `investigations/newave_cut_investigation.py`.
- **Root cause:** the reader forced the GNL lag count to a nonzero default when the
  header legitimately reported **0** (GNL anticipation off). The extra phantom `pi_gnl`
  columns misaligned the whole storage-gradient (`pi_varm`) block, re-attributing
  water values to the wrong reservoir codes.
- **Fix:** trust the header (`lag_maximo_gnl = int(header or 0)`). After the fix the
  phantom columns vanish, the dominant reservoirs agree on both sides, and the cuts are
  positively correlated.
- **Verdict:** **confirmed tooling bug, fixed.** Reclassifies the Water-value/FCF entry
  from "structural cut difference" to "cuts largely agree; residual is trial-point
  degeneracy".
- **Lesson:** when a binary reader takes a layout parameter, **never default a
  legitimate 0 to a nonzero** — validate it against the file before trusting derived
  coefficients.
- **Evidence/ref:** fix in `investigations/newave_cut_investigation.py`.
  [[project_cut_investigation_state]].

### GNL anticipated dispatch (adterm.dat) is now seeded, not zeroed (fixed)

- **Symptom:** in a **GNL case** (`dger.despacho_antecipado_gnl` on, `adterm.dat`
  present), the first 1–2 stages of a pre-committed thermal diverge — NEWAVE forces
  the anticipated dispatch (e.g. ~204.6 MW at stage 0) but Cobre let the LP choose
  freely. Historically accompanied by a converter WARNING _"…cobre's current LP
  cannot honour; writing zeros…"_.
- **Where:** results; thermal generation / immediate cost, first `lead_stages`
  stages of the GNL plant only.
- **Root cause (historical):** the converter wrote
  `past_anticipated_commitments.values_mw = [0.0]*lead_stages` because **an older
  Cobre** couldn't honour a non-zero pre-horizon seed. That limitation is **gone in
  Cobre ≥ 0.7.0**: the always-active anticipated "fishing" equality pins generation
  to the committed MW (`setup/mod.rs` "Pre-horizon seeding is enabled"; validator
  only rejects values outside `[min_generation_mw, max_generation_mw]`).
- **Fix:** the converter now passes the **real block-weighted committed MW** through
  (clamping into the plant's static generation bounds, with a warning, only on the
  rare out-of-range value). Requires the consuming Cobre to be **≥ 0.7.0**
  (`cobre-python>=0.7.0`); an older Cobre will reject the now-non-zero seed.
- **Verdict:** **expected divergence eliminated.** A residual first-stage gap on a
  GNL plant should now be investigated as a real difference, _not_ dismissed as the
  zeroing artifact. If you still see the old "writing zeros" warning, the conversion
  predates this fix.
- **Evidence/ref:** `converters/initial_conditions.py` (seeding loop),
  `converters/anticipated.py` (block-weighted MW), `converters/thermal.py`
  (`thermal_generation_bounds`). Verified against Cobre v0.8.0
  `crates/cobre-io/src/validation/semantic/thermal.rs`,
  `crates/cobre-sddp/src/setup/mod.rs`.

### End-of-study immediate-cost gap = min-outflow / min-gen violations (dispatch, not bug)

- **Symptom:** per-stage immediate cost (NEWAVE `COPER` vs Cobre `immediate_cost`)
  diverges most at the **last study stages**; the gap is the part of `COPER` above
  thermal, which `pmo.dat` attributes to **min-outflow (`VIOLACAO VZMIN`)** and
  **min-generation (`VIOLACAO GHMINU`)** violation penalties.
- **Root cause — dispatch, not conversion.** The per-stage floors are converted
  **identically** (NEWAVE's violations match the converted floors to the digit; plants
  that violate on the NEWAVE side also violate on the Cobre side). The difference is
  **end-of-study water management** — the two policies manage stored water differently
  near the horizon — a consequence of the water-value/FCF divergence, not a converter
  bug.
- **Observability caveat:** Cobre's own large violations are typically **post-study**,
  where NEWAVE's MEDIAS stops, so they are **invisible to `compare results`**. To
  compare that region, configure NEWAVE to simulate the post-study years.
- **Verdict:** **dispatch/policy difference (expected), not a converter bug.**
- **Evidence/ref:** per-plant `MEDIAS-USIH` violation columns vs Cobre `hydros`
  slacks; floors from the converted per-stage hydro bounds.

### Penalty-conversion productivities match NEWAVE (PRODT / altura-máxima)

- **Status: implemented.** The productivities that convert NEWAVE's energy-domain
  penalties to Cobre's flow/storage-domain costs were re-derived to match pmo.dat's
  applied values:
  - **PROD_MEDIA_SIN** (VAZMIN / TURBMN / TURBMX / spillage / turbined micro) = mean
    **equivalent PRODT** (`produtibilidade_equivalente_volmin_volmax`, the
    vol_min→vol_max equivalent via analytic head-integral,
    `hydro._equivalent_productivity`) over existing plants incl. zeros, no FICT fold.
    Drives `outflow/turbined_violation_below_cost`. (Earlier used a 65%-reference point
    value, ρ>0-only, FICT-folded → a small uniform overcharge.)
  - **Per-stage drift:** PROD*MEDIA_SIN varies slightly per config via the CFUGA/CMONT
    movers in `modif.dat` (VOLREF_SAZ is \_not* applied — PRODT is volume-independent),
    so the VAZMIN/turbined override columns vary in the decimals while
    water_withdrawal/evaporation stay fixed — matching pmo's behaviour.
    (`hydro.compute_per_stage_prodt_sin_mean`, `_per_stage_equivalent_productivities`,
    shared `_per_stage_drop_overrides`.)
  - **MAX_PRODTACUM_SIN** (DESVIO "outros usos" / evaporation) = max accumulated ρ at
    **altura máxima** (`produtibilidade_acumulada_calculo_altura_maxima`), constant
    (`constraints.compute_max_prodtacum_sin`), putting `water_withdrawal_violation_cost`
    within a fraction of a percent of pmo's OUTROS USOS.
- **Verdict:** **resolved.** All reconcile with pmo.dat to within rounding.
- **Evidence/ref:** pmo.dat `PENALIDADE POR VIOLACAO ...` sections;
  `Pmo.produtibilidades_equivalentes`; `penalid.dat`.
  [[project_forward_penalty_validation]].

---

## Under investigation

### Water value / FCF gap — cut coefficients

- **Symptom:** simulated `water_value_per_hm3` only partly matches; some reservoirs are
  large-negative in NEWAVE but ~0 in Cobre (and vice-versa).
- **State-space note:** when NEWAVE is fully individualized (no REE), its cut carries
  per-reservoir storage gradients (`pi_varm_uhe`) and inflow-lag duals (`pi_qafl`), no
  `pi_earm_ree`; Cobre's state is storage-only (`max_order = 0`). So the storage
  gradients are the **same units/space** (R$/hm³) — the per-reservoir comparison is
  apples-to-apples; the gap is **not** REE aggregation.
- **Root cause (partial):** single-cut gradient differences are dominated by the two
  policies being **tangent at different trial points** — the forward policies visit
  different storages, so different (usually small) reservoirs bind — which is a
  _consequence_ of the dispatch divergence, not an independent cut-construction bug.
  The remaining open question is a **scalar FCF-value gap** (the first-stage future cost
  differs).
- **Verdict:** likely-expected (degenerate / trial-point), pending a common-point
  FCF-value check.
- **Next step:** build a **common-point FCF evaluator** — for each model, take all of a
  stage's cuts and evaluate `FCF(x) = max_k (intercept_k + Σ coef_k · x)` at a shared
  storage point, and compare the resulting future-cost **values** (tests whether the
  two FCFs agree _as functions_).
- **Tooling:** `investigations/compare_cuts.py` (aligns one cut per side, merges by
  reservoir name), `cobre_cut_investigation.py` / `newave_cut_investigation.py`
  (decoders), `compare_states.py` (visited-state overlap). Stage map for the
  cost-to-go: `Cobre s ↔ NEWAVE estudo (s + offset)`; the cut/FCF offset differs from
  the results immediate-cost offset by one. [[project_cut_investigation_state]].

---

## Reference

### `forward.dat` — high-precision penalty/cost oracle (decode recipe)

- **What:** NEWAVE's binary forward-simulation dump exposes, at full precision per
  scenario, `custo_operacao` (live per-stage immediate cost), `custo_geracao_termica`,
  and every physical violation quantity (`violacao_defluencia_minima`,
  `volume_vertido`/`volume_turbinado`, `deficit`, `geracao_termica`, `intercambio`, …).
  Higher precision than MEDIAS. Decode with
  `investigations/forward_penalty_experiment.py`.
- **Decode dims:** read `forwarh.dat` (`Forwarh`) for `tamanho_registro`,
  `numero_series_gravadas`, REE/submarket/patamar counts; `n_stages =
filesize / record_size / n_series`. Plant counts from `confhd`/`conft`, agrint groups
  from `agrint`, GNL lag from `adterm` (NaN→0). Stage map: forward `estagio` k ↔ Cobre
  `stage_id` k−1.
- **Per-plant ordering (critical):** pass exactly the **simulated** hydro plants =
  `confhd` existing (`usina_existente == "EX"`) minus fictitious (`FICT.*`); the `NC`
  plants are already excluded. Passing all `confhd` rows over-reads the per-plant hydro
  block and corrupts every per-plant field after it. Per-patamar values are MWmed
  **contributions** → **sum** over patamares for the stage value. (The per-plant
  _thermal_ decode ordering is unresolved — prefer MEDIAS-USIT for per-plant thermal.)
- **Freeze structure (≠ MEDIAS):** `custo_operacao` is **live** (has the violation
  spikes) while `custo_geracao_termica` **freezes** at the last study month through the
  post-study tail. So `custo_operacao − custo_geracao_termica` isolates the live
  non-thermal (penalty) cost — but only within the study horizon where thermal is live.
  The `custo_violacao_*` breakout props may be dumped as zero — reconstruct from
  `op − thermal` instead.
- **Penalty ground truth = pmo.dat**, not `forward.dat` `op − thermal` (the thermal
  field can be stale at violation stages). pmo.dat reports the _applied_ penalties in
  `(R$/hm³)·(mês/h)` and the VAZMIN/TURBMIN/TURBMAX penalty is **uniform across REEs**
  (a single system-wide PROD_MEDIA_SIN), not per-plant.
  [[project_forward_penalty_validation]].

### Per-plant NEWAVE generation sources

- **Thermal:** MEDIAS-USIT.CSV via `comparators.newave_readers.read_medias_thermal`
  (`newave_code`, variables `GTMIN`/`GTERM`/`GTERMTOT`). With a single simulation
  series, MEDIAS equals the exact scenario. Stage map: MEDIAS stage = Cobre stage +
  offset.
- **Cobre stage-varying thermal bounds** live in `constraints/thermal_bounds.parquet`
  (per `thermal_id`/`stage_id`: min/max generation + cost), overriding the scalar in
  `system/thermals.json`. Capacity = POTEF × FCMAX × (1−TEIF) × (1−IP) with `expt.dat`
  period overrides; cost overrides come from the `clast.dat` modificacoes block.
