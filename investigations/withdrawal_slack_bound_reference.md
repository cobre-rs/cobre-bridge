# Reference: bounding the water-withdrawal violation slack (NEWAVE → Cobre)

**Purpose.** Document the bound NEWAVE implicitly applies to the water-withdrawal
("retirada para outros usos" / `VRETIRUH`) violation, so it can be reproduced in
Cobre later. Cobre currently leaves the under-withdrawal slack **unbounded**, which
produces a non-physical "phantom water injection" at upstream run-of-river plants.
The total cost and total cascade violation are unaffected (it is a cost-neutral
degeneracy), but the per-plant attribution diverges sharply from NEWAVE.

Derived from `example/{newave,cobre}_rodada_2000_sem_pos` (28 stages, no post-study)
and cross-checked on `…_2000_completo` (64 stages). Bounds compare = 100 % match on
both; this is purely a dispatch/LP-feasibility difference, not a converter bug.

---

## 1. TL;DR — the bound

NEWAVE enforces that the **realized** withdrawal `R` stays between `0` and the
target `T` (same sign as `T`):

```
T > 0 (removal):   R ∈ [0, T]        under-withdrawal slack  s_under = T − R ∈ [0, T]
T < 0 (addition):  R ∈ [T, 0]        under-application slack  |s_under| ≤ |T|
```

i.e. **the realized withdrawal can never cross zero** — you cannot turn a scheduled
_removal_ into an _injection_ (or vice-versa) via the violation slack. The maximum
under-withdrawal is therefore `|T|` (deliver none of it), never more.

Cobre today: the under-withdrawal (`neg`) slack has upper bound **`+∞`**, so the LP
can drive `R` arbitrarily negative — injecting water the plant never scheduled to
remove.

**Proposed change (Cobre):** cap the under-withdrawal slack upper bound at
`|water_withdrawal_m3s|` instead of `+∞`.

---

## 2. Evidence — NEWAVE floors realized withdrawal at 0

PICADA (NEWAVE code 126), head of the Paraíba do Sul / Santa Cecília diversion
cascade `PICADA(126) → SOBRAGI(127) → SIMPLICIO(129) → ILHA POMBOS(130)`, sem_pos
case. `target = (VRETIRUH + VIOL_POS_VRETIRUH)/2.63`; `nw_realized = VRETIRUH/2.63`;
`cb_realized = cobre water_withdrawal_m3s − water_withdrawal_violation_neg_m3s`:

| stage | target m³/s | NEWAVE realized | Cobre realized | Cobre neg slack |
| ----: | ----------: | --------------: | -------------: | --------------: |
|     1 |        1.07 |        **0.00** |     **−23.65** |           24.72 |
|     9 |        1.09 |        **0.00** |     **−25.67** |           26.76 |
|    11 |        1.11 |        **0.00** |     **−28.66** |           29.77 |
|    23 |        1.11 |        **0.00** |     **−26.66** |           27.77 |
|    25 |        1.07 |        **0.00** |     **−28.65** |           29.72 |

- **NEWAVE**: realized = 0.00 at every violating stage — it under-delivers PICADA's
  entire ~1.08 m³/s target and stops there. The shortfall is pinned at the upper
  bound `T`; it cannot go further. (Confirmed system-wide: across all plant-stages
  `VRETIRUH ≥ 0` for positive-target plants, and `VIOL_NEG_VRETIRUH ≡ 0` — NEWAVE
  never over-withdraws.)
- **Cobre**: realized = `T − neg` goes to ≈ −24…−29 m³/s. The unbounded `neg` slack
  acts as a free water source (priced at the withdrawal penalty), which PICADA then
  turbines (turbined = 41.6 m³/s cap vs inflow 18; spillage matches NEWAVE exactly).

### Why it is cost-neutral (so it survived unnoticed)

The **per-stage total** under-delivery across the whole cascade is identical to the
digit (sem_pos Σ = 503.2 m³/s-stages on both sides; completo 632.9 vs 632.1), so the
penalty cost matches: Cobre `withdrawal_violation_cost` ≈ NEWAVE's unattributed
`COPER − CTERM − CDEF` residual, and total operation cost (`COPER` vs
`immediate_cost`) agrees within ~2 % on sem_pos. The LP just has many cost-equal
optima for **where** to place the conserved shortfall; the unbounded slack lets it
concentrate it upstream (and turbine the relief water) instead of leaving it at the
downstream withdrawal point (SIMPLICIO), which is NEWAVE's choice.

---

## 3. Current Cobre implementation

`crates/cobre-sddp/src/lp_builder/matrix.rs`, `fill_withdrawal_slack_columns`
(~L549–587):

```rust
// Neg slacks (under-withdrawal).
for h_idx in 0..layout.n_h {
    let col = layout.col_withdrawal_neg_start + h_idx;
    let hb = ctx.bounds.hydro_bounds(h_idx, stage_idx);
    bufs.col_upper[col] = if hb.water_withdrawal_m3s > 0.0 {
        f64::INFINITY          // <-- unbounded: lets realized withdrawal go negative
    } else {
        0.0
    };
    let hp = ctx.penalties.hydro_penalties(h_idx, stage_idx);
    bufs.objective[col] = hp.water_withdrawal_violation_neg_cost * total_stage_hours;
}
// Pos slacks (over-withdrawal): same [0, +∞) / 0 structure.
```

The water-balance RHS subtracts the target (`…matrix.rs:774`, "static RHS =
ζ \* (deterministic_base_h − water_withdrawal_m3s_h)"), and the `neg` column adds
water back into the balance — so `neg ≤ T` ⇔ realized removal `R = T − neg ≥ 0`.

### Two correctness notes for the implementer

1. **Signed targets are real.** The converter writes
   `water_withdrawal_m3s ∈ [−89.9, +115.2]` for this case; negative targets are the
   inter-basin _return/addition_ points (e.g. ILHA POMBOS ≈ −235 m³/s). The current
   `> 0.0` guard pins **both** slacks to 0 for `T ≤ 0`, so negative-target plants get
   _no_ relief at all (rigidly enforced). NEWAVE confirms these return plants never
   violate here — but at least one plant (code 132) has a **negative target and
   violates**, so the bound must be **magnitude-based** (`|T|`), not gated on `T>0`.
2. **NEWAVE never over-withdraws** (`VIOL_NEG_VRETIRUH ≡ 0`). The `pos`
   (over-withdrawal) slack is unused in practice; the fix below targets the `neg`
   direction, which is where the divergence lives. Bounding `pos` symmetrically (or
   to 0) is optional and lower priority.

---

## 4. Proposed change

Replace the `+∞` upper bound with the target magnitude so the realized withdrawal
cannot cross zero, and make it sign-aware so it also covers negative (return)
targets:

```rust
// Under-application slack: realized withdrawal must stay on the [0, T] segment,
// i.e. it cannot flip sign. Cap the slack at |T| (NEWAVE floors realized at 0).
let cap = hb.water_withdrawal_m3s.abs();
bufs.col_upper[col] = cap;   // was: if hb.water_withdrawal_m3s > 0.0 { INFINITY } else { 0.0 }
```

(When `T = 0` this yields `0.0`, preserving today's presolve-elimination for plants
with no scheduled withdrawal.)

This makes the under-withdrawal slack `∈ [0, |T|]`, matching NEWAVE's implicit
`R ∈ [0, T]`. PICADA's neg slack would then be capped at ≈ 1.08 m³/s (its own
target), forcing the remaining conserved cascade shortfall onto the large downstream
withdrawal (SIMPLICIO), exactly as NEWAVE distributes it.

---

## 5. Expected effect & validation

- **Per-plant attribution converges to NEWAVE**: PICADA/SOBRAGI under-withdrawal
  drops from ~25–30 m³/s to ≤ their own ~1–2 m³/s targets; the balance lands at
  SIMPLICIO. The non-physical PICADA over-turbining (41.6 vs inflow 18) disappears.
- **Total cost / total cascade violation should be ~unchanged** (it was already
  conserved): a good regression guard is "Σ withdrawal violation per stage and total
  `immediate_cost` move < 0.5 %".
- **Validate** by re-running `cobre-bridge compare results` on
  `…_2000_sem_pos` and re-checking the per-stage cascade conservation
  (NEWAVE `VIOL_POS_VRETIRUH/2.63` total vs Cobre neg total — currently both 503.2)
  _and_ the per-plant `water_withdrawal_violation_neg_m3s` vs NEWAVE
  `VIOL_POS_VRETIRUH/2.63` now matching plant-by-plant, not just in aggregate.

---

## 6. Scope

- **Cobre-side change only.** The cobre-bridge converter is correct — its
  `hydro_bounds.water_withdrawal_m3s` equals NEWAVE's required withdrawal
  (`VRETIRUH` + `VIOL_POS_VRETIRUH`, ÷2.63) to the digit.
- Independent of the post-study construction (sem_pos and completo behave
  identically on withdrawal). The separate **post-study cost divergence**
  (completo total cost +4 %, up to +16 %/stage; sem_pos within ~2 %) is unrelated to
  this slack and tracked separately in the skill's `known-divergences.md`.

See `compare-newave-cobre/known-divergences.md` →
"Water-withdrawal violation — conserved cascade-wide, degenerate per-plant
redistribution" and memory `project_withdrawal_violation_degeneracy`.
