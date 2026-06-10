# Findings requiring Cobre-side intervention

Issues in **Cobre** (the Rust SDDP solver, `cobre-rs/cobre`) surfaced while
validating **cobre-bridge**'s NEWAVE → Cobre conversion against NEWAVE reference
runs. Each item below is a **Cobre-side** change — the bridge already does its part
correctly (or has been fixed). Line references are against the `cobre` repo as of
this writing; re-locate by symbol name if they drift.

Audience: Cobre maintainers / agents. Ordered by priority.

| #   | Finding                                                                   | Type                   | Severity |
| --- | ------------------------------------------------------------------------- | ---------------------- | -------- |
| 1   | Under-withdrawal slack is unbounded                                       | correctness (physical) | medium   |
| 2   | Per-stage generic constraints replicated per block                        | cleanliness / LP size  | low      |
| 3   | Micro-penalty merit order below solver tolerance → spurious energy excess | numerical              | medium   |
| 4   | Anticipated (GNL) commitment cost unattributed in the cost breakdown      | reporting / observ.    | low      |

---

## Finding 1 — Under-withdrawal (`water_withdrawal_violation_neg`) slack is unbounded

**Where:** `crates/cobre-sddp/src/lp_builder/matrix.rs`, `fill_withdrawal_slack_columns`
(the `neg` slack `col_upper` is set to `f64::INFINITY` whenever
`water_withdrawal_m3s > 0`).

**Symptom:** at run-of-river plants of a withdrawal cascade (Paraíba do Sul / Santa
Cecília: PICADA, SOBRAGI, …), Cobre reports a huge `water_withdrawal_violation_neg_m3s`
(e.g. ~25 m³/s) that **exceeds the plant's own withdrawal target** (~1 m³/s). NEWAVE
instead reports the shortfall (`VIOL_POS_VRETIRUH`) at the downstream withdrawal point.
The aggregate cascade violation and total cost are conserved, but the per-plant
attribution is non-physical (a plant "un-withdraws" far more than it was ever asked to
withdraw, effectively injecting water).

**Root cause:** the under-withdrawal slack lets the _realized_ withdrawal go **negative**
(turning a scheduled removal into an injection). NEWAVE bounds the realized withdrawal to
`[0, target]` — you can under-deliver at most the whole target, never more — so its
shortfall is capped at `|target|`.

**Recommended change:** cap the under-application slack at the target magnitude rather
than `+∞`:

```rust
// realized withdrawal must stay on the [0, T] segment (cannot flip sign)
let cap = hb.water_withdrawal_m3s.abs();
bufs.col_upper[col] = cap;   // was: if water_withdrawal_m3s > 0.0 { INFINITY } else { 0.0 }
```

Make it **magnitude-based** (not gated on `> 0`): the converter writes **signed** targets
(negative = inter-basin return/addition points), and at least one plant has a negative
target _and_ violates.

**Verification:** per-plant `water_withdrawal_violation_neg_m3s` should drop to ≤ the
plant's own `water_withdrawal_m3s`, the cascade shortfall should land on the downstream
withdrawal plant (matching NEWAVE plant-by-plant, not just in aggregate), and total cost
should be unchanged (it was already conserved).

**Full derivation & evidence:** `investigations/withdrawal_slack_bound_reference.md`.

---

## Finding 2 — Per-stage generic constraints (`block_id = None`) are replicated per block

**Where:** `crates/cobre-sddp/src/lp_builder/layout.rs`, `enumerate_generic_constraint_rows`
(~L588–619); slack pricing in `crates/cobre-sddp/src/lp_builder/matrix.rs` (~L1532, L1575).

**Symptom:** a generic constraint whose bound is declared for the **whole stage**
(`block_id = None`) — e.g. the VminOP / security-curve constraint, which sums
`hydro_storage` (an end-of-stage quantity) — is materialised as **one identical row per
block** (`for block_idx in 0..n_blks`), each with its own slack column priced by **that
block's** hours (`penalty * stage.blocks[entry.block_idx].duration_hours`).

**Is it wrong?** **No — not a correctness bug, and not triple-counting.** The N rows are
identical (same storage column, same RHS), so the solver puts the same shortfall `D` in
each, and each is charged only its own block's hours. Since `Σ block_hours = stage hours`,
the total penalty is `penalty × stage_hours × D` — i.e. **one month**, not N. (Verified on
a real run: a 3-block, 720 h stage with shortfall 920.99 paid
`3431.22 × (152+207+361) × 920.99 = 2.275e9`, exactly one-month pricing, not 3×.)

**Why change it anyway:** it is **redundant and fragile**:

- N identical rows + N slack columns where **1 would do** — LP bloat on every per-stage
  generic constraint (VminOP, per-stage RE, …).
- It is correct _only because_ the rows happen to be identical and `Σ block_hours` equals
  the stage hours. A future per-stage constraint that mixes a stock term with a per-block
  term would make the rows non-identical and the implicit "sums to one month" reasoning
  would break silently.
- Cobre **already has the clean pattern**: the inflow / evaporation / withdrawal slacks
  use a **single** per-stage column priced by `total_stage_hours`
  (`matrix.rs` ~L470, L544, L573 — `fill_*` called with `total_stage_hours`). A
  `block_id = None` generic constraint should do the same.

**Recommended change:** for `block_id = None` generic-constraint bounds, emit **one** row
priced by `total_stage_hours` instead of one row per block. Cost is unchanged; the LP gets
smaller and the per-stage-vs-per-block intent becomes explicit. (Keep the per-block path
for bounds that _do_ specify a `block_id`.)

**Bridge note:** cobre-bridge currently compensates for the per-block-replicated behavior
by converting the VminOP coefficients to MWmonth using the **stage-total** month hours
(`converters/constraints.py:_vminop_energy_factor`). If Cobre moves to the single-row
`total_stage_hours` form, that bridge conversion stays correct (it already targets the
stage-total hours), so the two changes are compatible and independent.

---

## Finding 3 — Micro-penalty merit order sits below the solver tolerance → spurious energy excess

**Where:** LP solver feasibility/optimality tolerance + the hydro micro-penalty values
(`spillage_cost`, `turbined_cost`, `diversion_cost`, bus `excess_cost`).

**Symptom:** Cobre dumps zero-priced `excess_mw` where NEWAVE has none; at the plant level
the same effect shows as a **turbine↔spill** mismatch (water conserved): NEWAVE spills a
run-of-river surplus, Cobre turbines it and dumps the resulting energy as excess.
Concentrated where non-controllable generation exceeds local load (negative net load) and
at fictitious nodes.

**Root cause:** the micro-penalty ordering that should make spilling cheaper than
turbining-into-excess sits ~4–5 orders of magnitude below the dominant LP costs
(thermal / deficit and the FCF cut gradients). At the default LP solver feasibility
tolerance it falls **below the effective optimality tolerance**, so the solver is
numerically indifferent between spill and turbine-into-excess and picks arbitrarily.

**Recommended change (one of):**

- tighten the LP solver feasibility/optimality tolerance so the micro-penalty merit order
  is resolved; **or**
- scale the spill / turbine / excess micro-penalties up _together_ (preserving their
  ordering) so it survives the tolerance.

**Verification (confirmed):** tightening the solver feasibility tolerance **removes the
excess** and aligns the turbine/spill split with NEWAVE. (Do **not** "fix" this by
inflating `excess_cost`, which only masks it.)

**Evidence:** cobre-bridge memories `project_cobre_excess_root_cause`,
`project_excess_penalty_diagnostic`; per-bus method
`investigations/compare_sem_intercambio.py`.

---

## Finding 4 — Anticipated-dispatch (GNL) commitment cost is unattributed in the simulation cost report

**Where:** `crates/cobre-sddp/src/simulation/extraction.rs` (cost extraction, ~L1211–L1213)
and the cost-record schema in `crates/cobre-io/src/output/simulation_writer.rs`.

**Symptom:** for a GNL plant (`anticipated_config.lead_stages ≥ 1`), the `thermals`
output reports `generation_cost = 0` at every delivery stage, and the per-stage `costs`
output books the plant's fuel under **no** category — `thermal_cost` excludes it and
`contract_cost` is 0. The fuel cost is real and **is** in `immediate_cost`, but only as
an **unattributed remainder**, so (a) the per-category breakdown does not reconcile
(Σ named categories ≠ `immediate_cost`), and (b) a thermal-cost comparison against a
reference model that books GNL fuel in its thermal line (NEWAVE `CTERM`) shows a spurious
gap equal to the GNL fuel (~7.6e9 on the validation case, localized to the GNL
submarkets).

**Why it happens (correct optimization, incomplete reporting):** the commitment is charged
on the **decision column** at the decision stage —
`objective = cost_per_mwh × delivery_hours × discount_factor[delivery]`
(`lp_builder/matrix.rs::fill_anticipated_decision_objective`) — and the delivery-stage
generation cost is zeroed (`zero_anticipated_delivery_thermal_cost`) to avoid
double-counting. `immediate_cost` (`objective − θ`) therefore includes it, but the
category extraction sums only `indexer.thermal` (generation columns) for `thermal_cost`
and has **no** range covering the anticipated-decision columns.

**Recommended change:** extract the anticipated-decision column range into its own
reported field (e.g. `anticipated_thermal_cost`), **or** attribute it to `thermal_cost`
at the delivery stage. Either makes the breakdown reconcile to `immediate_cost` and the
thermal-cost line comparable to a delivery-basis reference.

**Verification:** after the change, Σ(named cost categories) = `immediate_cost` per stage
(within the existing micro-penalty columns), and for a deterministic GNL case the total
thermal+anticipated cost matches the reference model's thermal cost (basis/discount/timing
aside).

**Bridge note:** cobre-bridge reads `thermal_cost` verbatim and pairs it with NEWAVE
`CTERM` (`comparators/charts.py::thermal_cost_chart`) under a now-known-wrong
"apples-to-apples" assumption; the bridge documents this as an **expected** divergence
(`known-divergences.md`) and compares GNL-inclusive totals via `immediate_cost` until this
lands.

---

## Not in this list (handled bridge-side / under investigation)

- **VminOP slack penalty was ~2.628× too high** (security curve never violated under
  scarcity) — **fixed in the bridge** (`converters/constraints.py:_vminop_energy_factor`):
  ρ_acum·storage is now converted to true MWmonth per stage. No Cobre change required,
  though Finding 2 above is the related cleanliness item.
- **Residual water-value / FCF gap under scarcity** (Cobre's trained policy values water
  more steeply than NEWAVE's → holds water and deficits where NEWAVE drains) — **under
  investigation**; not yet a confirmed Cobre intervention (may be cut-construction,
  convergence, or risk-measure/CVaR setup). Tracked separately.
