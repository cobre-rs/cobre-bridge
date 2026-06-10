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

### Deck drift — converted deck edited after the NEWAVE run

- **Triage rule (pre-flight, any comparison):** before trusting a divergence, run
  `stat -c '%y  %n' <case>/*.dat <case>/saidas/pmo.dat`. If any input `.dat` is **newer
  than `saidas/pmo.dat`**, the deck was edited after the run that produced `saidas/`, so
  the converted Cobre case and the NEWAVE outputs may be from different decks. Cross-check
  `pmo.dat`'s config echo (anticipation flag, plant counts, sim type, FPHA, POS) against
  the current `.dat` files before reading any result.
- **Verdict:** input mismatch — **fix the deck, not the code** (the converter faithfully
  mirrors whatever deck it is given). _The rodada_2000 cases pass this check (inputs
  ~10:59, `pmo.dat` 11:05)._

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

### Water-withdrawal violation — conserved cascade-wide, degenerate per-plant redistribution

- **Symptom:** Cobre shows large per-plant `water_withdrawal_violation_neg_m3s` at
  **upstream run-of-river** plants of a withdrawal cascade (e.g. PICADA ~25–30 m³/s,
  SOBRAGI; CASTRO ALVES only in post-study) — **far exceeding those plants' own
  withdrawal targets** (PICADA target ≈ 1.08 m³/s, slack ≈ 24.7). NEWAVE instead
  reports the shortfall (`VIOL_POS_VRETIRUH`) concentrated at the **downstream**
  withdrawal plant (SIMPLICIO). Reads as "Cobre violates withdrawal a lot more."
- **Where:** results; Hydro Operation withdrawal-slack panels. The Paraíba do Sul /
  Santa Cecília withdrawal cascade `PICADA(126) → SOBRAGI(127) → SIMPLICIO(129) → …`
  (the urban-supply diversion to the Guandu).
- **Root cause — conserved degeneracy, not a bug.** Verified on
  `example/{newave,cobre}_rodada_2000_sem_pos` and `…_completo`:
  - **Bounds 100 % match**; the LP feasible region is identical.
  - The **per-stage TOTAL** under-delivery across the cascade is **identical** to the
    digit (sem_pos Σ = 503.2 m³/s-stages NEWAVE vs 503.2 Cobre; completo 632.9 vs
    632.1), so the penalty is the same: Cobre `withdrawal_violation_cost` ≈ NEWAVE's
    **unattributed COPER residual** (`COPER − CTERM − CDEF`; NEWAVE leaves its named
    `CDSVC/CDSVF/VIOL_DSV` diversion-cost columns at 0). **Inflows and spillage match
    exactly** — it is **not** turbine↔spill.
  - The divergence is purely **where the conserved slack is placed**. Cobre's
    under-withdrawal (`neg`) slack has upper bound **`[0, +∞)`** per plant whenever
    `water_withdrawal_m3s > 0` (cobre `crates/cobre-sddp/src/lp_builder/matrix.rs`
    `fill_withdrawal_slack_columns`, ~line 567) — **not** bounded by the plant's own
    target. So the LP can dump the whole cascade shortfall as an unbounded `neg` slack
    (a water source) at the upstream run-of-river plants and **turbine that relief
    water** (PICADA turbined = 41.6 m³/s cap vs inflow 18). NEWAVE's shortfall is
    bounded per plant, so it lands at the big downstream withdrawal (SIMPLICIO).
  - **Converter is correct:** the per-plant target it writes
    (`hydro_bounds.water_withdrawal_m3s`) equals NEWAVE's required withdrawal
    (`VRETIRUH` realized + `VIOL_POS_VRETIRUH`, ÷2.63) to the digit.
- **Verdict:** **expected** — cost-neutral degenerate redistribution (total operation
  cost COPER vs immediate matches stage-by-stage on sem_pos, within ~2 %).
  **Independent of post-study** (sem_pos and completo behave identically).
- **Latent Cobre-side improvement (optional, not a bridge fix):** bound the `neg`
  under-withdrawal slack at `[0, water_withdrawal_m3s]` (you cannot under-withdraw more
  than the target). That would force the shortfall onto the downstream withdrawal plant
  and make the per-plant attribution match NEWAVE, removing the non-physical upstream
  "injection."
- **Evidence/ref:** cobre `matrix.rs:567`; MEDIAS-USIH `VRETIRUH`/`VIOL_POS_VRETIRUH`
  (hm³/month, ÷2.63 → m³/s); converter `hydro_bounds.water_withdrawal_m3s`.
  [[project_withdrawal_violation_degeneracy]].

### Per-bus / per-reservoir storage allocation differs — same-cost degeneracy (two mechanisms)

- **Symptom:** aggregating reservoir storage by submarket/bus, NEWAVE and Cobre place
  the stored water differently — NEWAVE holds **more in NORTE**, Cobre more in
  **SUDESTE/SUL** (roughly, not the same hm³/stages; per-bus and SIN totals wander
  ±~4000 hm³). At the reservoir level individual reservoirs diverge by **40–100 % of
  usable capacity** (EMBORCAÇÃO, ITUMBIARA, NOVA PONTE, CORUMBÁ, SANTA BRANCA), some
  even anti-correlated (BATALHA). Feels wrong for a deterministic single-scenario run.
- **Where:** results; Hydro Operation storage (`storage_final_hm3` vs NEWAVE
  `VARMUH`+vol_min). Verified on `…_2000_sem_pos` and `…_2000_sem_pos_carga`.
- **Root cause — same-cost degeneracy, two coexisting mechanisms.** Year 2000 is wet, so
  **CMO/spot ≈ 0 in every submarket, every stage** (Cobre `spot_price` ~1e-4, NEWAVE
  `CMO` = 0.00): an extra MWh is free, must-run thermal aside.
  1. **Zero water value** in SUL / NORDESTE / NORTE (`PIVARM` and `water_value_per_hm3`
     ≈ 0–0.1 R$/hm³) — storage there is a flat direction; chiefly **BALBINA** (isolated
     Amazon) and **TUCURUÍ** in transitions.
  2. **Reservoir substitution** in SUDESTE, where water **is** valued (~21k–600k R$/hm³)
     but many reservoirs in a cascade/REE share the **same marginal value** (e.g.
     EMBORCAÇÃO/ITUMBIARA/NOVA PONTE/SERRA FAÇÃO/MIRANDA all 21,423) and the security
     curve (curva.dat) is a **single aggregate-REE VminOP** soft constraint, so which
     reservoir holds the energy is free as long as the REE total stays above the curve.
     (SANTA BRANCA pinned at its 10 % floor in Cobre while NEWAVE cycles it 10 %↔100 % is
     this substitution — both keep the SUDESTE aggregate above the curve, **VminOP
     violation = 0 on both sides**.)
     Total operation cost matches to **0.65 %** (`COPER` vs `immediate`, 1.0065) and the
     **future cost matches to ~1 %** (NEWAVE `CUSTO_FUTURO` vs Cobre `future_cost`),
     confirming cost-equivalence. _(Correction to an earlier read: water is **not** ≈0
     everywhere — that was true only for the NORTE reservoirs first inspected. SUDESTE water
     is valued; its storage scatter is substitution-degenerate, not zero-value.)_
- **The +10 GW SUDESTE load experiment (`…_carga`) did NOT break this.** It was absorbed
  entirely by **free hydro drawdown** (GHTOT +10 000 MWmed, thermal unchanged, deficit 0,
  EARMF −54 000 MWmês, COPER **identical**, CMO still 0). The storage divergences are
  **unchanged** from the base case (not load-amplified) — pre-existing policy/FCF
  differences, still cost-equivalent. A residual systematic tilt remains (Cobre stores
  ~1–2 % less value, FCF ~0.8 % lower) — the _Water value / FCF gap_.
- **Verdict:** **expected** — same-cost degenerate allocation; the forward-pass signature
  of the _Water value / FCF gap_. **Not a converter bug.**
- **Validation rule:** do **not** read per-bus / per-reservoir storage scatter as a
  conversion error while **CMO ≈ 0 and VminOP/curva violation = 0 on both sides**. Judge
  fidelity on **cost** (immediate + future) and on **constrained** quantities. A storage
  divergence is only concerning if it moves **total cost** or causes a **VminOP/curva
  violation on one side only**. **To actually break the degeneracy you need true scarcity
  — nonzero CMO/deficit; +10 GW was not enough, try a much larger load or a dry year.**
- **Evidence/ref:** NEWAVE `MEDIAS-MERC` `CMO`, `MEDIAS-SIN` `CUSTO_FUTURO`/`VIOL_CAR`,
  `MEDIAS-USIH` `PIVARM`/`VARMUH`; Cobre `buses.spot_price`,
  `costs.future_cost`/`generic_violation_cost`, `hydros.water_value_per_hm3`; curva.dat →
  `converters/constraints.py:convert_vminop_constraints`.
  [[project_storage_allocation_degeneracy]]; relates to _Water value / FCF gap_.

### Security-curve (VminOP) slack penalty ~2.6× too high — Cobre deficits instead of violating the curve (CONFIRMED unit bug)

- **Symptom:** under **scarcity only** (load raised until CMO>0 / deficit appears), NEWAVE
  and Cobre diverge in operating strategy and **total operation cost by ~2×**
  (`…_2000_sem_pos_carga` with +10 GW in **each** of SUDESTE/SUL/NORDESTE: NEWAVE `COPER`
  2.35e12 vs Cobre `immediate` 1.12e12, ratio 0.48). NEWAVE **drains reservoirs to ~3–7 %**,
  violating the security curve massively (`VIOL_CAR` huge) and taking terminal deficits;
  **Cobre holds storage at the curve (~25 %+)**, `generic_violation_cost ≈ 0`, and **deficits
  ~40 % more** (9.66e11 vs 6.96e11). Even at the **terminal stage** (sem_pos ⇒ future value
  = 0) Cobre holds ~58 000 MWmês while deficiting — clearly suboptimal.
- **Where:** results under scarcity; per-stage stored energy (`EARMF` vs Cobre
  `stored_energy_final_mwh`), deficit, `MEDIAS-SIN VIOL_CAR` vs Cobre
  `generic_violation_cost`. **Invisible in surplus** (CMO=0, the curve is never the binding
  trade-off — see the storage-degeneracy entry; this is why +10 GW in one bus showed nothing).
- **Root cause — units mismatch in the VminOP slack penalty (converter).** The VminOP LHS is
  `Σ ρ_acum[MW/(m³/s)] · hydro_storage[hm³]`, which is **not** energy — it is the true stored
  energy (MWmês) **times the hm³↔(m³/s)·month factor ≈ 2.628** (= 730 h·3600 s / 1e6).
  Verified: `Σ ρ_acum·(V−vmin)` for REE 1 = 108 526 vs NEWAVE `EARMF` = 39 555 MWmês →
  **ratio 2.74 ≈ 2.628**. The slack penalty is the penalid.dat curva CUSTO **3431.22 R$/MWh**,
  and Cobre multiplies it by `block_hours` (`matrix.rs:~1575`). Net effective penalty on a
  unit of _real_ energy shortfall = **3431.22 × 2.628 ≈ 9017 R$/MWh**, which **exceeds the
  deficit cost 7810.62 R$/MWh** → the merit order is **flipped**: Cobre deficits rather than
  dip below the curve. NEWAVE has the correct ordering (3431 < 7810) so it drains and violates
  the CAR before deficiting. (Draining an upstream reservoir generates the **accumulated**
  ρ_acum energy as the water passes the whole cascade — same ρ_acum as the EARM drop — so once
  the units are right the per-MWh comparison is simply 3431 vs 7810; the FCF/cuts agree to
  ~5 %, so this is **not** a policy/cut gap and **not** a structural CAR-usage difference.)
- **The vmin term is already handled** (not a bug): the LHS uses **absolute** storage but
  `_rhs_at` adds the dead-volume energy back (`dead_s = Σ ρ·vmin`), so the binding condition is
  `Σ ρ·(V−vmin) ≥ pct·E_useful` and the slack measures the **useful**-energy shortfall.
- **Verdict:** **FIXED (converter).** First case where NEWAVE and Cobre genuinely disagreed on
  dispatch (not same-cost).
- **Fix — IMPLEMENTED (Option A, per-stage).** `converters/constraints.py:convert_vminop_constraints`
  now converts `per_stage_acc` (ρ_acum) from MW/(m³/s) to MWmês/hm³ before it feeds both the LHS
  coefficients and the RHS, so the VminOP is in true stored energy and `penalty × block_hours`
  resolves to the penalid.dat R$/MWh. The factor cancels between LHS and RHS, so the binding
  storage level is unchanged; `@rho_acum_h{id}` was verified to be consumed **only** by the VminOP
  expression (Cobre computes `stored_energy_final_mwh` independently and correctly). The factor is
  applied **per stage using each stage's real month length** (`_month_hours`; cobre stage
  durations are actual month hours — 672 h Feb … 744 h Dec — not NEWAVE's fixed 730 h), so the
  effective penalty is exactly 3431.22 R$/MWh on every stage (the fixed 730 h left a ±~9 % error,
  worst in February). Restores `CAR (3431) < deficit (7810)` so Cobre violates the curve before
  deficiting. Verified at conversion level (RHS scales by the per-stage factor). **Still TODO:
  re-run Cobre on carga3** and confirm (1) `generic_violation_cost` > 0, (2) drawdown matches
  NEWAVE, (3) cost ratio → ~1.0, (4) surplus cases (sem_pos/completo) unregressed.
- **Evidence/ref:** NEWAVE `MEDIAS-SIN` `COPER`/`CDEF`/`VIOL_CAR`/`EARMF`/`CUSTO_FUTURO`;
  Cobre `costs.{immediate,deficit,future}_cost`/`generic_violation_cost`,
  `hydros.stored_energy_final_mwh`; `curva.dat`; `converters/constraints.py`;
  cobre `crates/cobre-sddp/src/lp_builder/matrix.rs:~1575`.
  [[project_storage_allocation_degeneracy]].

### Thermal cost (CTERM) > Cobre `thermal_cost` — GNL anticipated-commitment cost booked outside the thermal category (expected)

- **Symptom:** per-stage and total NEWAVE `CTERM` runs **systematically above** Cobre
  `thermal_cost` (~6 % total on `…_2000_sem_pos_carga`: 1.354e11 vs 1.279e11) **even
  though thermal generation is identical** — SIN `GTERM` matches ratio 1.000 every
  stage, and per-plant to <0.2 %. It localizes per **submarket** to exactly the buses
  holding **GNL anticipated plants**: NORDESTE (P. SERGIPE I, ~5.6e9) and SUDESTE
  (ST.CRUZ NOVA + LINHARES, ~2e9) — together the whole gap (~7.6e9). SUL and NORTE match
  to the cent. A small extra wobble at the **first 2 / last 2 stages** (±8–17 %) is the
  lead-time boundary plus a clast `modificacoes` date-edge.
- **Where:** results; Per-Stage Cost → Thermal (`CTERM` vs `thermal_cost`). Per-bus via
  `MEDIAS-MERC` (`GTERM`/`CTERM` per submarket code 1=SE,2=S,3=NE,4=N) vs Cobre
  `thermals` aggregated by `bus_id`.
- **Root cause — accounting-basis difference (_where_ GNL fuel is booked), not a missing
  cost.** Cobre models GNL plants as **anticipated dispatch**
  (`anticipated_config.lead_stages`). It charges their fuel on the **decision column at
  the decision stage, discounted to delivery** —
  `objective = cost_per_mwh × delivery_hours × discount_factor[delivery]`
  (`crates/cobre-sddp/src/lp_builder/matrix.rs::fill_anticipated_decision_objective`) —
  and **zeroes the delivery-stage generation cost**
  (`zero_anticipated_delivery_thermal_cost`, so `generation_cost = 0` for these plants
  at _every_ stage). In `simulation/extraction.rs`, `immediate_cost` (L1211, whole
  objective − θ) **includes** this commitment cost but `thermal_cost` (L1213,
  `range_sum(indexer.thermal)` = generation columns only) **excludes** it; it lands in
  the **unattributed remainder of `immediate_cost`** (not `contract_cost`, which is 0).
  NEWAVE instead books GNL fuel in `CTERM` **at delivery**, nominal, in the plant's
  submarket.
- **Categorization is Cobre-side; the mismatched pairing is bridge-side.** The exclusion
  is decided in Cobre's `extraction.rs`; cobre-bridge reads the `thermal_cost` column
  **verbatim** (`comparators/cobre_readers.py::read_cobre_stage_costs`) and pairs it with
  `CTERM` (`comparators/charts.py::thermal_cost_chart`). That chart's docstring claim of
  _"apples-to-apples … live thermal generation cost on both sides"_ is **wrong** — the
  gap is exactly the GNL anticipated fuel.
- **Verdict:** **expected** — basis difference, **not** a conversion bug and **not** a
  missing cost (the total `immediate_cost` includes it). The converter correctly emits
  GNL plants with `cost_per_mwh` and `anticipated_config`.
- **Rules:**
  - **Per-plant thermal — use `GTERMTOT`, not `GTERM`.** NEWAVE reports `GTERM` =
    generation **above** the must-run minimum, `GTMIN` = the minimum, and
    `GTERMTOT = GTERM + GTMIN = SIN total` (verified ratio 1.000 every stage). Using
    `GTERM` under-reports each plant by ~21 % and fabricates phantom dispatch-mix
    differences (it caused a false "Cobre runs ANGRA full / +35 % thermal energy" read).
  - **Thermal cost — `CTERM` ↔ `thermal_cost` is NOT apples-to-apples.** Expect a gap =
    the GNL anticipated fuel, localized to the GNL submarkets. Compare GNL-inclusive
    **total `immediate_cost`** instead; per-stage it is shifted by `lead_stages` (Cobre
    at commitment, NEWAVE at delivery) and discounted.
- **Cobre-side improvement (reporting only):** surface the anticipated commitment cost as
  its own cost category (or attribute it to `thermal_cost` at delivery) so the breakdown
  reconciles — `investigations/cobre_side_findings.md` Finding 4.
- **Evidence/ref:** cobre `matrix.rs::fill_anticipated_decision_objective` /
  `zero_anticipated_delivery_thermal_cost`; `simulation/extraction.rs:1211,1213`; NEWAVE
  `MEDIAS-MERC`/`MEDIAS-USIT` `GTERM`/`GTMIN`/`GTERMTOT`/`CTERM`; bridge
  `comparators/cobre_readers.py`, `comparators/charts.py`.
  [[project_thermal_cost_gnl_accounting]].

---

## Under investigation

### Post-study (pós-estudo) cost divergence — completo only, not from withdrawal

- **Symptom:** total operation cost (NEWAVE `COPER` vs Cobre `immediate_cost`) matches
  within ~2 % on the **no-post-study** case (`…_2000_sem_pos`, 28 stages) but diverges
  **+4 % overall (up to +16 % per stage)** on the **post-study** case
  (`…_2000_completo`, 64 stages = study + 3 + 3 post). The withdrawal violation is
  **conserved in both** (see Confirmed entry), so this residual is **not** the
  withdrawal — it sits in the post-study tail.
- **Where:** results; per-stage immediate cost in the post-study stages.
- **Root cause:** not yet isolated. Candidate: end-of-horizon water management / FCF and
  the converter's post-study extrapolation (last-year seasonal repetition; cf. the known
  water-withdrawal post-study converter bug at `hydro.py:1083`). Running `sem_pos` was the
  right move — it confirms the **core conversion is sound** and localizes the residual to
  the post-study construction the user already suspected.
- **Verdict:** open — investigate next; distinct from the (resolved) withdrawal degeneracy.
- **Next step:** drill `compare results` on completo stages 28–63 (cost-first §3a): which
  cost component drives the +16 % spikes, and whether it tracks post-study inflow/FCF
  extrapolation.

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
