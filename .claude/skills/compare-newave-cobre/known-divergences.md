# Known Divergences: NEWAVE vs Cobre

Accumulated catalogue of differences seen when comparing a NEWAVE run against its
cobre-bridge conversion. **Check here first** during triage (Step 3 of the skill).

For each entry, record the _symptom_ (what you observe in the diff/report), where
it appears, the _root cause_, and the _verdict_:

- **expected** — a consequence of conversion or modelling semantics; safe to accept.
- **concerning** — investigate as a likely bug.

> How to use: when you confirm a new cause, add an entry under **Confirmed
> divergences**. When a verdict changes (e.g. a bug gets fixed), update the entry
> and note the commit/PR.

---

## Current run context (numbers are run-specific — read this first)

All numbers come from the **example case** (`example/newave_rodada` vs
`example/cobre_rodada/output`), re-executed **2026-05-30**. The config is being
**varied deliberately** during this investigation, so each entry says which run it
came from. Two run modes are in play:

- **(A) Comparison run** — `iteration_limit ≈ 100`, single forward/sim scenario
  (`forward_passes = 1`, `simulation.num_scenarios = 1`, `inflow.scheme =
"historical"`, `historical_years = [2000]`, `seed = 42`) → p10=p50=p90.
  Cut selection **disabled** (`cut_selection.enabled = false`). Source for the
  results/bounds/cost/turbine-spill/water-value-sim numbers. Bounds **100% clean**
  (19,850/19,850).
- **(B) Cut-study run** — same but `iteration_limit = 1` (`policy_mode: fresh`,
  one cut per stage, gap ~99.9%). Source for the **first-cut** comparison in the
  Water-value/FCF entry. Results are unconverged in (B) — don't read its
  simulation numbers as a comparison.

Cross-cutting config facts (both modes): **NEWAVE is fully individualized — no REE
concept** (cuts carry `pi_varm_uhe` storage gradients + `pi_qafl_uhe*_lag` inflow
lags, no `pi_earm_ree`); Cobre `estimation.max_order = 0` (storage-only state, **no
inflow-lag state**); `inflow_non_negativity = truncation_with_penalty`.

⚠️ **`penalties.json` `bus.excess_cost` is toggled**: stock base ≈ 3.55e-4; it was
**inflated ×1e3 → 0.355** as a diagnostic (drives excess to 0 — see the Energy
Excess entry; [[project_excess_penalty_diagnostic]]). Check the file before trusting
any excess number.

> When you re-run with a different config, **re-derive** (the `cobre-bridge compare`
> summary + the cost/excess/stage-0 digest + `compare_cuts.py`) rather than trusting
> the figures below verbatim.

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

### Energy excess in Cobre (zero-priced) — currently masked by a ×1e3 penalty bump

- **Symptom (base penalty):** with `excess_cost` at its ~3.55e-4 base, Cobre dumped
  **9844 MW-stage of energy excess** (up to 2736 MW at a single bus-stage, 14
  stages, all 5 buses) while NEWAVE `EXCESSO` is **0**, and it was
  **cost-invisible** (Cobre `excess_cost` ≈ 2240 R$ NPV ≈ 0 in 10⁹ R$).
- **Symptom (this run):** `excess_cost` was raised **×1e3 → 0.355** as a diagnostic,
  and Cobre excess collapses to **exactly 0** — `excess_mw` p10/p50/p90 all 0, **0
  bus-stages > 1 MW** — while total cost still matches NEWAVE to **−0.07%**
  (10.146 vs 10.154 bi). So the dumping was a **pure pricing artifact**: a tiny
  per-MW excess charge removes it at negligible cost.
- **Where:** results / Energy Balance tab; Cobre `excess_mw` (bus_aggregates).
  Under the base penalty the largest source was **bus 4** (the FC / 5th submarket —
  0 load, no generation: 2736 @ s19, 1014 @ s20, an exchange/fictitious-node
  artifact); plus **NCS-driven** buses where non-controllable generation exceeds
  local load (bus 2/NE: NCS ~20000 MW, net_load negative → un-exportable surplus).
- **Root cause:** _still open._ The penalty bump **masks** the symptom but does not
  explain **why Cobre has surplus energy to dump at all** — that question remains and
  likely **shares a cause with the turbine→spill reallocation** (both = "Cobre has
  surplus energy it doesn't use"). Suspect the FC/fictitious-submarket exchange
  routing and the un-exportable NCS surplus, at the network/exchange layer (not the
  trusted NCS/load conversion). The ×1e3 bump is a **diagnostic, not a fix.**
- **Verdict:** concerning — **investigate later** (flagged by the user). Treat the
  current zero-excess as masked, not resolved.
- **Evidence/ref:** example case, 2026-05-30 re-run; `penalties.json`
  `excess_cost = 0.355`. [[project_excess_penalty_diagnostic]]. (Corrects an earlier
  mis-read that excess was a non-issue — it is/was zero in _cost_, not in _energy_.)

### Turbine↔spill reallocation at Madeira run-of-river (Jirau, Sto Antônio)

- **Symptom:** `turbined_m3s` and `spillage_m3s` diverge strongly (WithinTol 17% /
  35%, sMAPE 72% / 106%) but are mirror images: Σ|Δturb|=1.93M, Σ|Δspill|=1.63M, yet
  Σ|Δoutflow| = 810k (offset ratio 0.23) — water conserved. Concentrated at JIRAU
  (mean m³/s turbined 15217→5228, spillage 1511→11500, NW→CB) and STO ANTONIO
  (turbined 15660→4072, spillage 1327→12916): **NEWAVE turbines the Madeira flow,
  Cobre spills it.** (Reverse-signed at a few plants — TUCURUI, XINGO — where Cobre
  turbines more, consistent with a system-wide reshuffle.)
- **Where:** results; `turbined_m3s` / `spillage_m3s`, Madeira run-of-river plants.
- **Root cause:** _likely alternative optima._ Total cost matches to **0.07%**, so
  this is a cost-neutral spatial turbine/spill reallocation under hydro surplus
  (turbine here vs spill-here-and-turbine-elsewhere are tied). Likely **shares a
  cause with the energy-excess entry** (both = "Cobre has surplus energy it doesn't
  use"). Check whether a turbine-preference tie-break (spill only once turbines are
  full) aligns Cobre to NEWAVE.
- **Verdict:** likely expected (degenerate) — confirm via a tie-break test.
- **Evidence/ref:** example case, 2026-05-30 re-run §3b operation analysis.

### Water value / FCF gap — cut coefficients, seeded at stage 0

**This is the active investigation thread.** Tooling and next steps at the bottom.

- **Symptom (simulated duals):** `water_value_per_hm3` barely matches (WithinTol
  0.5%, r 0.70); some reservoirs large-negative in NEWAVE but ~0 in Cobre (MANSO,
  QUEIMADO, CACONDE, IRAPE), 19 Cobre zeros where NEWAVE ≠ 0.
- **Symptom (stage 0 — same initial state, same inflow):** immediate cost matches
  (NEWAVE `COPER` 212 vs Cobre 209, 10⁶ R$) but Cobre's **future cost is ~0.46 bi R$
  below NEWAVE's\*\* (Cobre `future_cost` 23.174 vs NEWAVE `CUSTO_FUTURO` 23.630 bi),
  also visible in `lower_bound`. Storage/dispatch redistribute from t0: largest t0
  generation gaps ITAIPU −2708 MW, TUCURUI −2212, STO ANTONIO −1190 (Cobre <
  NEWAVE), XINGO/FOZ CHAPECO +1078/+838 (Cobre > NEWAVE); storage TUCURUI +12674 hm³.
- **Symptom (iteration-2 cut, `compare_cuts.py`) — CORRECTED 2026-05-30:** comparing
  each model's first real cut (Cobre stage 0 ↔ NEWAVE estudo stage 10), storage
  gradient by reservoir name, the cuts are **positively correlated: Pearson r ≈
  +0.55–0.65** across stages 0–7 (not "uncorrelated"). The **two dominant reservoirs
  agree on both sides**: IRAPE (NEWAVE −752k / Cobre −570k R$/hm³) and QUEIMADO
  (−729k / −553k). Residual per-reservoir differences are concentrated in **small
  reservoirs that bind at one model's trial point but not the other's** (Cobre loads
  LAJES/CACONDE/MANSO ≈ −560…−625k where NEWAVE ≈ 0; NEWAVE loads GUARAPIRANGA −393k
  where Cobre ≈ 0) — and the two cuts are tangent at **different trial points** (only
  ~52/151 visited storages coincide, `compare_states.py`), so those single-cut
  gradients are **not directly comparable**. Σ|wv|: NEWAVE 1.88M vs Cobre 2.93M;
  intercepts NEWAVE rhs 1.74e10 vs Cobre 6.15e9 R$ (also not comparable across
  different tangent points). ⚠️ **The earlier "uncorrelated r = −0.02, disjoint
  support, FOZ R. CLARO/CORUMBA III dominate, 39% GNL" reading was a parser bug** —
  see "NEWAVE cut parser `lag_maximo_gnl or 2`" below.
- **State-space check (REE hypothesis overturned; GNL claim RETRACTED):** NEWAVE is
  **fully individualized, no REE** (the cut carries 151 `pi_varm_uhe` storage
  gradients + 1812 `pi_qafl_uhe*_lag` inflow lags [151 reservoirs × 12 PAR(p) lags];
  **zero** `pi_earm_ree`). Cobre's state is **storage only** (151 vars,
  `max_order = 0`). So both storage gradients are the same units/space (R$/hm³) — the
  per-reservoir comparison **is** apples-to-apples; the gap is **not** REE
  aggregation. NEWAVE's inflow-lag duals (`pi_qafl`) are **0** at these cuts. **GNL
  anticipation is OFF in this case** (`despacho_antecipado_gnl = 0`,
  `cortesh.lag_maximo_gnl = 0`, **no `pi_gnl` columns**); the previously-reported
  "39% `pi_gnl` thermal state" was a misparse — that −729k value is really QUEIMADO's
  storage water value. There is **no thermal-anticipation state** mismatch here.
- **Where:** the raw policy cuts (`output/policy/cuts/stage_NNN.bin`,
  `cortes.dat`); downstream `water_value_per_hm3`, `lower_bound`, stage-0 storage/gen.
- **Root cause:** _substantially de-escalated._ Not selection (disabled), not REE,
  not inflows (identical, user-confirmed), and **not a fundamentally different cut
  construction** — once the parser bug is fixed the cuts are positively correlated
  and agree on the dominant reservoirs. The residual per-reservoir gradient
  differences are dominated by **different trial points** (the forward policies visit
  different storages, so different small reservoirs bind), which is a _consequence_ of
  the dispatch divergence, not an independent cut-construction bug. The remaining open
  question is the **scalar FCF-value gap** (stage-0 future cost 23.17 vs 23.63 bi) —
  to test it cleanly, evaluate each model's converged stage-0 FCF (max over its cuts)
  at a **common** storage point rather than comparing single-cut gradients.
- **Verdict:** **downgraded to likely-expected (degenerate / trial-point), pending the
  common-point FCF-value check.** The earlier "concerning, primary systematic driver"
  verdict rested on the now-retracted parser-bug numbers.
- **Evidence/ref:** example case, 2026-05-30 — 2-iteration run, `compare_cuts.py` +
  `compare_states.py` with the **fixed** `newave_cut_investigation.py` reader; inflows
  confirmed identical by the user. [[project_cut_investigation_state]].
- **Tooling (repo-root scripts, committed `36757458`):** `compare_cuts.py` (driver:
  aligns one cut per side, merges by reservoir name, prints Δ R$/hm³),
  `cobre_cut_investigation.py` (`cobre_water_values` — decodes
  `policy/cuts/stage_NNN.bin` via `flatc` + `policy.fbs`; `SCHEMA_PATH` resolves via
  `COBRE_SCHEMA_PATH` / `COBRE_REPO` / `~/git/cobre`), `newave_cut_investigation.py`
  (`newave_water_values`; `AGREGADO_EM_REE = False` for this individualized case).
  Units pinned to R$ (NEWAVE 10³, Cobre 10⁶). Stage map: `Cobre s ↔ NEWAVE estudo
(s+10)` for the **cost-to-go** (the _results_ immediate-cost offset is +9; the
  cut/FCF offset is +10 because `stage_s.bin` holds the cost-to-go stage `s`
  accesses).
- **Next steps:** the single-cut gradient comparison has reached its useful limit
  (cuts tangent at different trial points). Build a **common-point FCF evaluator**:
  take all of each model's stage-0 cuts, evaluate `FCF(x) = max_k (intercept_k +
Σ coef_k · x)` at a shared storage point `x` (initial condition; NEWAVE's trial
  point; Cobre's trial point), and compare the resulting future-cost **values**. That
  tests whether the two FCFs agree **as functions**, which the scalar 23.17-vs-23.63
  gap says they don't quite. (`pi_qafl = 0`, so the inflow-lag state can be ignored in
  the evaluation; no `pi_gnl` with GNL off.)

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
  keeps 1655 m³/s in Pimental (exact match) and incurs the VZMIN penalty like NEWAVE.
  **Reconfirmed on the 2026-05-30 re-run:** PIMENTAL outflow 1655.4 = 1655.4 (exact),
  BELO MONTE 4440.9 = 4440.9; `Outflow Min Viol.` (VZMIN) NEWAVE 0.542 vs Cobre 0.559
  bi; **total cost agrees to −0.07%** (10.154 vs 10.146 bi). `compare bounds` stayed
  100% clean throughout. (Absolute VZMIN/total differ from the earlier post-fix run —
  10.58/9.60 → ~10.15 bi — because the case configuration changed; the _resolution_
  holds.)
- **Verdict:** **resolved** (was concerning).
- **Evidence/ref:** example case, 2026-05-30 before/after cost-first analysis + re-run
  reconfirmation; fix in `converters/hydro.py`.

## Confirmed divergences

### Tooling false-positive: NEWAVE cut parser `lag_maximo_gnl or 2` (FIXED)

- **Symptom:** the NEWAVE↔Cobre cut comparison looked structurally disjoint — Pearson
  r ≈ −0.02, NEWAVE storage value concentrated on FOZ R. CLARO/CORUMBA III, a large
  `pi_gnl_sbm3_pat2_lag1 ≈ −729k` "GNL thermal-anticipation dual" (~39% of L1), and a
  disjoint top reservoir set vs Cobre. This drove a months-long "the cut construction
  is fundamentally different / GNL state Cobre lacks" hypothesis.
- **Where:** **the investigation tooling, not the models.**
  `newave_cut_investigation.py::read_newave_cut_coefficients`.
- **Root cause:** `lag_maximo_gnl = int(cortesh.lag_maximo_gnl) or 2` forced the GNL
  lag count to **2** whenever the header legitimately reported **0** (GNL anticipation
  off, `despacho_antecipado_gnl = 0`). Passing 2 to `inewave`'s `Cortes.read` reserved
  2 phantom `pi_gnl` lags per submarket × patamar (24 columns), **misaligning the
  whole `pi_varm` block**: storage water values were re-attributed to the wrong UHE
  codes (IRAPE→"FOZ R. CLARO", GUARAPIRANGA→"CORUMBA III") and QUEIMADO's −729k
  storage dual was misread as a `pi_gnl` term.
- **Fix:** trust the header — `lag_maximo_gnl = int(cortesh.lag_maximo_gnl or 0)`.
  After the fix: **0 `pi_gnl` columns**, dominant reservoirs **IRAPE & QUEIMADO agree
  on both sides**, and **Pearson r ≈ +0.55–0.65** across stages 0–7.
- **Verdict:** **confirmed tooling bug, fixed.** Reclassifies the "Water value / FCF
  gap" entry from "concerning structural cut-construction difference" to "cuts largely
  agree; residual is trial-point degeneracy + an open scalar FCF-value gap."
- **Lesson:** when a binary reader takes a layout parameter, **never default a
  legitimate 0 to a nonzero** — validate the parameter against the file (here:
  `tamanho_corte` and the post-parse column set) before trusting derived coefficients.
- **Evidence/ref:** example case, 2026-05-30; fix in `newave_cut_investigation.py:62`.
  [[project_cut_investigation_state]].

### End-of-study immediate-cost gap = min-outflow/min-gen violations (dispatch, not bug)

- **Symptom:** per-stage immediate cost (NEWAVE `COPER` vs Cobre `immediate_cost`)
  diverges most at the **last study stages**: Cobre **23/24/25 ↔ NEWAVE 32/33/34**,
  Cobre **−26% to −47%** below NEWAVE (Δ −267/−405/−868 ×10⁶R$). Over the 28 study
  stages Cobre's Σ immediate is ~1771 ×10⁶R$ below NEWAVE.
- **Decomposition:** thermal (`CTERM`) is ~equal at these stages (~704–715); the gap
  is **`COPER − CTERM`** (342/783/1130), which MEDIAS-SIN folds into COPER without
  itemizing (flagged by `COPER_violcurva == COPER`; `VIOL_CAR = 0`, so **not** the
  curva de aversão). `pmo.dat` names it: **`VIOLACAO VZMIN` 1763** (outflow-min) +
  **`VIOLACAO GHMINU` 316** (gen-min), ×10⁶R$ NPV.
- **Plants (per-plant `VIOL_*UH`):** gen-min is a **single plant, TUCURUI** (s25: gen
  1141.7 < floor 1300 → viol 158.3); outflow-min led by **SERRA MESA** (s32: out 147.2
  < 300 → 152.8), **PEIXE ANGIC** (s34: 295.3 < 360 → 64.7), **BELO MONTE** (Volta
  Grande 300), MANSO, A.A. LAYDNER, ESTREITO TOC, CACONDE…
- **Root cause — dispatch, not conversion.** The per-stage floors are converted
  **identically** (Cobre `hydro_per_stage_bounds`: TUCURUI `min_generation_mw` =
  1500/1400/1300 at s23/24/25; SERRA MESA/PEIXE `min_outflow` = 300/360) — NEWAVE's
  violations match those floors **to the digit**. The difference is **end-of-study
  water management**: NEWAVE depletes TUCURUI (gen-min viol) and hoards SERRA
  MESA/PEIXE (releases the minimum, pays the outflow penalty rather than spend stored
  water), while Cobre keeps them above the floor. **BELO MONTE violates on both**
  (NEWAVE 239 vs Cobre 214 at s24) — proof the constraint is modelled on both sides.
  This is a consequence of the water-value/FCF differences + end-of-horizon, tied to
  the cut/FCF entry above.
- **Cobre's own violations are POST-STUDY (uncompared).** In-study Cobre
  `generation_violation_cost` = **0.0**; `outflow_violation_below` = 746.8 (< NEWAVE).
  But post-study (stages 28–63) Cobre's 2-iteration policy **collapses**: gen-viol
  1891, out-viol 9760 ×10⁶R$, with stage 37 alone ≈ **11.4 bi** (deficit 1882 +
  out-viol 3469 + gen-viol 1015). NEWAVE's MEDIAS stops at the study horizon (stage
  36 ↔ Cobre 27), so this collapse is **invisible to `compare results`** today.
- **Verdict:** **dispatch/policy difference (expected given the diverging policies),
  NOT a converter bug.** The actionable gap is observability: to compare the
  post-study region, NEWAVE must be set to **simulate the post-study years** so its
  MEDIAS extend past stage 36.
- **Evidence/ref:** example case 2-iteration run, 2026-05-30; per-plant `MEDIAS-USIH`
  `VIOL_GHMINUH`/`VIOL_VAZMINUH` vs Cobre `hydros` `generation_slack_mw`/
  `outflow_slack_below_m3s`; floors from `hydro_per_stage_bounds`.

### `forward.dat` is the high-precision penalty/cost oracle (decode recipe)

- **What:** NEWAVE's binary forward-simulation dump exposes — at full precision,
  per scenario — `custo_operacao` (live per-stage immediate cost, 10³R$),
  `custo_geracao_termica` (10³R$), and every physical violation quantity
  (`violacao_defluencia_minima` hm³, `volume_vertido`/`volume_turbinado` hm³,
  `deficit`, `geracao_termica`, `intercambio`, …). Far better than MEDIAS
  (scenario-averaged, frozen COPER). Decode it with `forward_penalty_experiment.py`
  (repo root).
- **Decode dims:** read `forwarh.dat` (`Forwarh`) for `tamanho_registro`,
  `numero_series_gravadas`, REE/submarket/patamar counts; n_stages =
  filesize / record_size / n_series. Plant counts from `confhd`/`conft`, agrint
  groups from `agrint`, GNL lag from `adterm` (NaN→0). Example: **64 stages, 1
  series**. Stage map: **forward `estagio` k ↔ Cobre `stage_id` k−1 ↔ MEDIAS stage
  k+8**.
- **Freeze structure (≠ MEDIAS):** in forward.dat `custo_operacao` is **LIVE**
  (varies per stage, has the violation spikes), while `custo_geracao_termica`
  **freezes** at the last study month through the post-study tail. (Example:
  study = stages 1–28, last study Dec-2026; thermal frozen 813,618 ×10³R$ /
  5,739.6 MWmes for stages 29–64.) So `custo_operacao − custo_geracao_termica`
  isolates the live **non-thermal (penalty) cost** — but only in the **study
  horizon** where thermal is live.
- **Violation-cost breakout props are 0** in this run (`custo_violacao_*`,
  `penalidade_curva_aversao` all dumped as zero) — don't pair them; reconstruct
  from `op − thermal` instead.

### Penalty productivities now match NEWAVE (PRODT / altura-máxima) — FIXED

- **Status: implemented (2026-05-31).** The penalty-conversion productivities were
  re-derived from inputs to match NEWAVE's pmo.dat-applied values exactly:
  - **PROD_MEDIA_SIN** (VAZMIN / TURBMN / TURBMX / spillage / turbined micro) = mean
    **PRODT** (`produtibilidade_equivalente_volmin_volmax`, the vol_min→vol_max
    _equivalent_ via analytic head-integral, `hydro._equivalent_productivity`), over
    **all** existing plants incl. zeros, **no FICT fold**. = **0.62916** (pmo VAZMIN
    821.78 ⟹ 0.6294). Was 0.656 (65%-reference point value, ρ>0-only, FICT-folded).
    → `outflow_violation_below_cost` 2250.55 → **2158.78**.
  - **Per-stage:** PROD_MEDIA_SIN drifts ~0.15% per config via the 5 CFUGA/CMONT
    movers (STO ANTONIO, TUCURUI, JIRAU, BELO MONTE, PIMENTAL) — VOLREF_SAZ is _not_
    applied (PRODT is volume-independent). Reproduces pmo's 820.53→821.78 wiggle. So
    the `penalty_overrides_hydro.parquet` VAZMIN/turbined columns now **vary in the
    decimals** while water_withdrawal/evaporation are **fixed** (absent from the
    override) — matching pmo exactly. (`hydro.compute_per_stage_prodt_sin_mean`,
    `_per_stage_equivalent_productivities`, shared `_per_stage_drop_overrides`.)
  - **MAX_PRODTACUM_SIN** (DESVIO "outros usos" / evaporation) = max accumulated ρ at
    **altura máxima** (`produtibilidade_acumulada_calculo_altura_maxima`), **constant**
    = **6.4458** (pmo OUTROS USOS 19,149.55 ⟹ 6.4371; Pmo 6.4420). Was 6.3542
    (65%-ref). → `water_withdrawal_violation_cost` 49,676.85 → **50,392.90** (now within
    0.13% of pmo). `constraints.compute_max_prodtacum_sin`.
- **Validated end-to-end:** all reconcile with pmo.dat to ≤0.15%. 919 tests pass (+9
  new in `TestEquivalentProductivity` / `TestProductivitySinMeansExample`).
- **Evidence/ref:** `pmo.dat:28371` (VAZMIN) `:28271` (OUTROS USOS); `Pmo.
produtibilidades_equivalentes` (`volmin_volmax`, `acumulada_calculo_altura_maxima`);
  `penalid.dat` VAZMIN 3431.22 / DESVIO 7818. [[project_forward_penalty_validation]]

### Outflow (VAZMIN) penalty overcharged ~4% — `rho_avg` vs PROD_MEDIA_SIN (uniform) [superseded by FIX above]

- **GROUND TRUTH = pmo.dat, not forward.dat.** `pmo.dat` reports the _final_ penalty
  values NEWAVE used, in `(R$/hm³)·(mês/h)`, in the section
  `PENALIDADE POR VIOLACAO DE VAZAO MINIMA`. The value is **uniform across every REE**
  (821.78 at Sep/stage 0, seasonally ~820.5–821.8) — a **single system-wide penalty
  applying PROD_MEDIA_SIN**, exactly as the manual says. TURBMIN/TURBMAX share the same
  821.78; OUTROS USOS (DESVIO) = 19,149.55; EVAPORAÇÃO = 191,314.77; FPHA = 78,106.20
  R$/MWh (= 10 × max_deficit_cost).
- **The penalty is NOT per-plant.** (Earlier revisions of this entry claimed a per-plant
  head-dependent ρ from a `forward.dat` `op−therm` reconstruction — that was WRONG, an
  artifact of a corrupted `custo_geracao_termica` field, byte-identical at stages 24/25.
  Retracted.)
- **Reconciliation:** pmo VAZMIN 821.78 × 730 h = 599,900 R$/hm³; energy identity
  `599,900 = 3431.22 R$/MWh (penalid) × 277.78 MWh/(hm³·ρ) × ρ` ⟹ NEWAVE
**ρ_SIN = 0.629**. Our converter uses **rho_avg = 0.656**
(`network.py:\_hydro_penalty_costs`, `outflow_violation_below_cost = 3431.22 × 0.656 =
  2250.55` R$/(m³/s)/h ≡ 856 in pmo units). **856 / 821.78 = 1.042 → we OVERCHARGE
  VAZMIN by ~4.2%, uniformly.** Matches the original pre-forward.dat figure
  (NEWAVE ≈ 1.577e6 vs ours 1.643e6 R$/(m³/s)·stage).
- **Verdict:** **minor, expected, low-priority.** The ~4.2% is purely that our
  `rho_avg` (0.656) is ~4% above NEWAVE's PROD_MEDIA_SIN (0.629) — a productivity-mean
  definition difference (plant set / weighting / reference volume), not a structural
  bug. Same 4% rides `turbined_violation_below` (pmo 821.78 too). To close it, match
  NEWAVE's PROD_MEDIA_SIN definition. Cost impact negligible; merit order unaffected.
- **Method note:** `forward.dat` `op−therm` is **unreliable for VAZMIN** in this run —
  `custo_geracao_termica` is duplicated/stale at the violation stages. Use **pmo.dat**
  penalty sections as the authoritative source for the _applied_ penalty values.
- **Evidence/ref:** `pmo.dat:28371` (VAZMIN), `:28271/28595/28695/28705` (other
  penalties); penalid VAZMIN=3431.22 (`penalid.dat`); micro-penalties confirmed
  negligible. [[project_forward_penalty_validation]]
