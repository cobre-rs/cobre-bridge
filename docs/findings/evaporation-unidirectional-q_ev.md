# Cobre evaporation slack: `Q_ev` is unidirectional and forces phantom over-evaporation

**Audience:** cobre development agent (handoff from cobre-bridge investigation)
**Status:** Open — explained gap between NEWAVE and cobre cost composition.
**Severity:** Modeling completeness. The dispatch policy that cobre produces is
plausible, but ~30% of the reported expected operating cost in NEWAVE-derived
cases is phantom evaporation-slack cost rather than real operation. Convergence
bounds, cost breakdowns, and the SDDP gap are all distorted by it.
**Date:** 2026-05-18 — observed while comparing NEWAVE pmo.dat and cobre training
output for a fully-individualized 155-plant NEWAVE case.

---

## TL;DR

Cobre's evaporation LP variable `Q_ev` is hard-bounded to `[0, max(0, k_evap0 + k_evap_v · v_max) · margin]`
(`crates/cobre-sddp/src/lp_builder/matrix.rs:323-336`). Whenever the linearized
evaporation predicts a _negative_ value at max storage — i.e. net rainfall over the
reservoir surface exceeds evaporation — the upper bound clamps to zero, `Q_ev` is
pinned at 0, and the equality row forces the `f_evap_minus` ("over-evaporation")
slack to absorb the entire `|Q̂_ev|` magnitude in `hm³`. The plant did not
actually over-evaporate; the slack is paying the toll for a water input cobre's
balance can't express.

For a NEWAVE-derived case where 80 of 155 plants have at least one month with
`c_ev < 0` (net rainfall gain — common in Brazilian basins), this fires roughly
9,500 m³/s of `evaporation_violation_pos` summed system-wide per stage and adds
roughly 30% to the cobre-reported expected total cost compared to what NEWAVE
reports on the same case.

The fix is architectural: either allow `Q_ev` to be free-signed, or split the
evap term into a sign-paired (`Q_ev`, `Q_rain`) variable pair, or perform a
pre-conversion split on the cobre-bridge side. Each option has trade-offs; this
document explains the mechanism and quantifies the impact so cobre can pick.

---

## Background

In Brazilian hydrothermal dispatch, NEWAVE's `hidr.dat` stores per-plant per-month
"evaporação efetiva" (effective evaporation) coefficients in mm/month. These are
**signed**: positive means net evaporation loss over the reservoir surface,
negative means net rainfall input over the surface exceeds evaporation losses.
This is a standard hydrological convention — many reservoirs in southern and
northern Brazil have several wet months where rainfall on the lake surface is
larger than open-water evaporation.

NEWAVE consumes these signed values directly in its reservoir water balance and
its `VIOL. EVAP. UHE` slack stays essentially at zero across the simulation.
Cobre's modeling choice — treating `Q_ev` as a non-negative outflow only —
means cobre cannot represent the rainfall side and must absorb it via the
violation slack.

---

## Cobre's current evap LP construction

References below are at `cobre` HEAD as of 2026-05-18.

### Coefficients (per (hydro, stage))

From `crates/cobre-sddp/src/hydro_models.rs:1615-1626`:

```text
k_evap_v = zeta_evap · c_ev[month] · dA/dV|_ref
k_evap0  = zeta_evap · c_ev[month] · A(V_ref) − k_evap_v · V_ref
```

`zeta_evap > 0` and `dA/dV > 0` for any sensible reservoir; `A(V_ref) > 0`.
The **sign of both `k_evap0` and `k_evap_v` follows the sign of `c_ev[month]`**.

### Columns (per evaporation hydro per stage)

From `crates/cobre-sddp/src/lp_builder/matrix.rs:306-348` (`fill_evaporation_columns`):

| Column         | Lower | Upper                                                     | Objective coefficient                                |
| -------------- | ----- | --------------------------------------------------------- | ---------------------------------------------------- |
| `Q_ev`         | 0     | `max(0, k_evap0 + k_evap_v · v_max) · Q_EV_SAFETY_MARGIN` | 0                                                    |
| `f_evap_plus`  | 0     | +∞                                                        | `evaporation_violation_neg_cost · total_stage_hours` |
| `f_evap_minus` | 0     | +∞                                                        | `evaporation_violation_pos_cost · total_stage_hours` |

### Row (equality, per evaporation hydro per stage)

From `crates/cobre-sddp/src/lp_builder/matrix.rs:629-647` (RHS) and
`crates/cobre-sddp/src/lp_builder/matrix.rs:1062-1113` (CSC entries):

```text
Q_ev − (k_evap_v / 2)·v − (k_evap_v / 2)·v_in + f_plus − f_minus = k_evap0
```

Where `v` is end-of-stage storage and `v_in` is the start-of-stage storage
(fixed by the storage-fixing row). Operating at the midpoint volume
`V_op = (v + v_in) / 2`, the constraint reads

```text
Q_ev + f_plus − f_minus = k_evap0 + k_evap_v · V_op  =  Q̂_ev(V_op)
```

where `Q̂_ev(V_op)` is the linearized evaporation at the operating volume.

---

## The failure mode

If `c_ev[month] < 0`:

1. `k_evap0 < 0` and `k_evap_v < 0`.
2. `k_evap0 + k_evap_v · v_max < 0` (linearized evap at max storage is negative).
3. `Q_ev` upper bound clamps to **0** (line 332: `(…).max(0.0)`).
4. The constraint becomes `f_plus − f_minus = Q̂_ev(V_op)`, with `Q̂_ev(V_op) < 0`.
5. The cheapest LP feasibility move is to set `f_evap_minus = −Q̂_ev(V_op) > 0`
   and `f_evap_plus = 0`.
6. The objective grows by
   `−Q̂_ev(V_op) · evaporation_violation_pos_cost · total_stage_hours`
   for **every month in which `c_ev` is negative on that plant**, every stage
   that maps to such a month, every scenario.

The slack fires not because the LP chooses to violate evaporation in some
edge case but because the constraint is **structurally infeasible** otherwise.
`Q_ev = 0` is the only value the variable can take, and the equality demands
the slack absorbs the imbalance.

`f_evap_plus` and `f_evap_minus` are otherwise correctly directional:
`f_evap_plus` is genuine under-evaporation slack (target higher than realized)
and `f_evap_minus` is genuine over-evaporation slack — directionality of the
slacks is not the issue. The issue is the unidirectional bound on `Q_ev`
itself.

---

## Quantification on a real NEWAVE case

Case: 155 individualized hydro plants, 3 study years (Sep 2024 – Aug 2027) plus 3
post-study years, 12% annual discount, CVaR `α=0.15 λ=0.4`. The cobre-bridge
conversion emits `evaporation.coefficients_mm` verbatim from `hidr.dat`.

### Sign distribution of `c_ev` across plants

| Group           | Plants |
| --------------- | -----: |
| no evap section |      4 |
| all months ≥ 0  |     71 |
| all months ≤ 0  |      0 |
| **mixed sign**  | **80** |

Examples of mixed-sign coefficients (`mm/month`, Jan…Dec):

- ITAIPU: `[14, 32, 51, 57, 60, 49, 30, 26, 13, −20, −33, −20]`
- JIRAU / STO ANTONIO: `[−80, −67, −93, −81, −74, −29, 11, 41, −25, −82, −66, −81]`
- PASSO REAL: range `[−92, +121]` (7 positive, 5 negative months)
- ITA: range `[−75, +111]` (5 negative months)

### Cost composition over the matched NEWAVE horizon (28 stages)

Mean across scenarios, present value, summed across stages 0..27:

| Cost item                          |           Cobre |          NEWAVE |    Ratio |
| ---------------------------------- | --------------: | --------------: | -------: |
| `thermal_cost`                     |     1.31 × 10¹⁰ |     1.14 × 10¹⁰ |    1.15× |
| `outflow_violation_below`          |      3.25 × 10⁹ |      7.90 × 10⁹ |    0.41× |
| **`evaporation_violation_cost`**   | **3.16 × 10¹⁰** |  **3.48 × 10⁷** | **905×** |
| `hydro_violation_cost` (gen below) |     3.51 × 10¹⁰ |      3.90 × 10⁷ |     900× |
| `inflow_penalty_cost`              |     2.01 × 10¹⁰ |               0 |        — |
| **Total**                          | **1.04 × 10¹¹** | **1.94 × 10¹⁰** | **5.3×** |

`thermal_cost` aligns to 1.15× — the dispatch policy is comparable. The 5.3×
gap is dominated by penalty cost. Of that gap, the evaporation slack alone
accounts for ~3.1 × 10¹⁰ R$, roughly **35% of the cobre-reported expected cost
over the matched horizon**.

### Worked example — ITAIPU November

```text
c_ev[Nov] = −33 mm/month
A(V_ref)  ≈ 1.35 × 10⁹ m² (≈ 1350 km², ITAIPU reservoir at operating volume)
k_evap0   ≈ ζ · (−33 × 10⁻³ m) · 1.35 × 10⁹ m² ≈ −4.45 × 10⁷ m³/month
          = −44.5 hm³/month

Q_ev      = 0 (clamped, since k_evap0 + k_evap_v · v_max < 0)
f_minus   ≈ |k_evap0|  ≈ 44.5 hm³/stage

In flow units: 44.5 hm³ / 730 h / 3.6 s/h × 10⁶ ≈ 17 m³/s of phantom over-evap
in every November stage of every scenario for ITAIPU alone.
```

System-wide we observe ~9,500 m³/s of `evaporation_violation_pos_m3s` summed
across plants per stage — averaging ~63 m³/s per plant if spread evenly,
consistent with this mechanism firing on the negative months of the 80
mixed-sign plants.

---

## Why the slack is so expensive

The `evaporation_violation_pos_cost` is set by cobre-bridge to
`1.1 · MAX_DEFICIT · ρ_max_acum` (~5.7 × 10⁴ R$/(m³/s·h), see
`src/cobre_bridge/converters/network.py`). The 1.1× over `MAX_DEFICIT` was
chosen deliberately to keep evaporation at the top of the merit order
(physical-cycle constraint, should be violated only as a last resort) while
not blowing up the LP coefficient range. This is the right merit order
_assuming the slack only fires when truly infeasible_. With the structural
mechanism described above, "truly infeasible" includes every negative-`c_ev`
month on every mixed-sign plant — a regime the merit-order argument was not
designed for.

Lowering the slack cost in cobre-bridge would reduce the cost report's
distortion but is _not_ a fix: the policy cobre learns would still treat the
"violation" as a real degree of freedom, allowing it to be substituted for
genuine constraint violations and corrupting the dispatch.

---

## How NEWAVE handles the same data

NEWAVE's reservoir continuity (per its manual) accepts the signed
`evaporacao_*` value directly in the storage balance:

```text
V_{t+1} = V_t + INFLOW_t − TURB_t − SPILL_t − EVAP(V_t, c_ev[month])
```

with `EVAP` taking the sign of `c_ev`. A negative `c_ev[month]` simply adds
water to the reservoir continuity in that month. NEWAVE's `VIOL. EVAP. UHE`
slack is reserved for actual physical infeasibility (rare); for our case
it totals 3.48 × 10⁷ R$ over 28 stages, four orders of magnitude below
cobre's slack on the same data.

---

## Options for cobre

In rough order of invasiveness:

### Option A — Make `Q_ev` free-signed

```rust
// matrix.rs:323
bufs.col_lower[col_q_ev] = f64::NEG_INFINITY;  // was 0.0
let q_ev_max = (coeff.k_evap0 + coeff.k_evap_v * hb.max_storage_hm3).abs();
bufs.col_upper[col_q_ev] = q_ev_max * Q_EV_SAFETY_MARGIN;
```

`Q_ev` becomes a signed evaporation flux. Negative values represent net
rainfall input absorbed by the reservoir. The storage continuity and energy
output equations consume `Q_ev` as an outflow with its signed sign, so
negative `Q_ev` automatically becomes inflow with no other code changes.

Risks to audit:

- Anywhere else in the LP that assumes `Q_ev ≥ 0` (e.g., storage upper bound
  feasibility analysis, cut generation, simulation extraction).
- Whether the bidirectional slacks remain meaningful (they should: with
  `Q_ev` free, the equality always solves with both slacks at 0 and the
  slacks only fire on genuine modeling infeasibility).

Estimated work: one-day change plus a careful audit and regression tests.

### Option B — Add a `Q_rain` sibling

Introduce a non-negative `Q_rain` and rewrite the row as

```text
Q_ev − Q_rain − (k_evap_v/2)·v − (k_evap_v/2)·v_in + f_plus − f_minus = k_evap0
```

with bounds

```text
Q_ev  ∈ [0, max(0,  k_evap0 + k_evap_v · v_max) · margin]
Q_rain ∈ [0, max(0, −(k_evap0 + k_evap_v · v_max)) · margin]
```

Cleaner physically, more invasive in the schema and the LP builder; also
affects simulation output schema (a new `q_rain_hm3` column).

### Option C — Pre-split coefficients in cobre-bridge

Split `c_ev[month]` into

```text
c_evap_only[month] = max(c_ev[month], 0)
c_rain_only[month] = max(−c_ev[month], 0)
```

emit `c_evap_only` as the evaporation series and inject `c_rain_only · A(V)`
as a non-controllable source on the bus or as a deterministic addition to the
plant's inflow. Doable entirely on the cobre-bridge side without any cobre
change, but the result is approximate (the rain term is not coupled to the
reservoir's surface-area-versus-volume curve at solve time — it's evaluated
once at conversion time at a reference volume).

---

## Recommendation

Option A is the cleanest cobre-side change and has the smallest scope.
Option B is the most physically correct long-term model. Option C is the
fastest unblock for NEWAVE-derived cases but is a workaround, not a fix.

This finding is logged here because (a) the slack-firing is a deliberate
LP-formulation choice in cobre, not a cobre-bridge bug; (b) any of the three
options has trade-offs that should be a cobre design decision; and (c) the
gap is large enough (~30% of reported expected cost on Brazilian individualized
cases) that downstream comparison reports and bound diagnostics will continue
to be misleading until it is closed.

---

## Units verification (and a stray docstring bug)

We traced the units end-to-end on the live case to confirm that the
penalty conversion on the cobre-bridge side is dimensionally correct,
and to nail down what `Q_ev`/`f_evap_*` actually represent. There is one
mismatch between docstrings and reality on cobre's side, surfaced here.

### What the docstring says

`crates/cobre-sddp/src/hydro_models.rs:170-187`:

```text
/// The evaporation volume (hm³) is approximated as:
///     evap = k_evap0 + k_evap_v * (V - V_ref)
/// ...
pub k_evap0: f64,        // Constant term of the linearized evaporation (hm³).
pub k_evap_v: f64,       // Volume-dependent slope (hm³/hm³ = dimensionless).
```

Reading this in isolation it looks like `Q_ev` is a per-stage volume in
hm³, which would imply the slack variables are in hm³ too, and our
penalty would need units of `R$/hm³` (not `R$/(m³/s·h)`).

### What the math actually does

`crates/cobre-sddp/src/hydro_models.rs:1611-1616`:

```rust
let stage_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
let zeta_evap = 1.0 / (3.6 * stage_hours);
let k_evap_v = zeta_evap * c_ev * da_dv;
let k_evap0  = zeta_evap * c_ev * a_ref - k_evap_v * v_ref;
```

with `c_ev` in mm/month (from `hidr.dat`), `a_ref` in km² (from
`interpolate_area`), and `da_dv` in km²/hm³.

Dimensional analysis of `zeta_evap · c_ev · a_ref`:

```text
[1 / (3.6 · h)] · [mm/month] · [km²]
= [1 / (3.6 · h)] · [10⁻³ m / month] · [10⁶ m²]
= [10³ m³ / (3.6 · h · month)]
```

`3.6 = 3600 s/h / 1000` is the standard `(m³/s) → (m³/(0.001·s·h))`
conversion factor cobre uses everywhere (cf. `M3S_TO_HM3 = 3600/1e6`
which is the same constant flipped). Numerically `3.6 · stage_hours`
for a 730 h stage is 2628, equal to `2.628 × 10⁶ s / 1000 m³` —
exactly the divisor that turns a hm³/month volume into an m³/s average
flow. So:

```text
k_evap0 = zeta_evap · c_ev · a_ref  →  m³/s   (averaged over the stage)
```

NOT hm³. The docstring is wrong.

### How that unit gets propagated

`crates/cobre-sddp/src/lp_builder/matrix.rs:834-838` (water balance row):

```rust
// Evaporation: Q_ev_h enters water balance with +ζ.
let col_q_ev = layout.col_evap_start + local_idx * 3;
col_entries[col_q_ev].push((row, zeta));
```

where the water-balance `zeta = Σ block.duration_hours · M3S_TO_HM3`
(from `layout.rs:587`). For a flow in m³/s to contribute a volume in hm³
to the storage continuity row, it must be multiplied by `total_stage_hours · 3600/1e6`,
which is exactly this `zeta`. Therefore `Q_ev` is **m³/s** and the
water balance is dimensionally consistent.

The slack columns `f_evap_plus` and `f_evap_minus` enter the equality
row with coefficients ±1 alongside `Q_ev`, so they share `Q_ev`'s unit
(m³/s).

`crates/cobre-sddp/src/lp_builder/matrix.rs:346-347` (objective):

```rust
bufs.objective[col_f_plus]  = hp.evaporation_violation_neg_cost * total_stage_hours;
bufs.objective[col_f_minus] = hp.evaporation_violation_pos_cost * total_stage_hours;
```

For the objective term `coef · f_evap_minus` to be in R$ with
`f_evap_minus` in m³/s, the coefficient must be in `R$/(m³/s)`.
Multiplying through by `total_stage_hours` means the **input
`evaporation_violation_pos_cost` must be in R$/(m³/s · h)**. That's
what cobre-bridge produces.

### End-to-end identity check on a real stage

From `example/convertido/output` (live run, mid-May 2026 conversion):

```text
penalties.json::hydro.evaporation_violation_cost = 54,306.41   R$/(m³/s · h)
stage 0 total hours                              = 720.0       h
stage 0, scenario 0 evap slack (sum over plants) = 15.95469    m³/s
                                                              (same in all 3 blocks,
                                                               consistent with stage-level)

predicted cost = 15.95469 × 54,306.41 × 720.0   ≈ 6.236 × 10⁸ R$
cobre-reported costs/<scen>/data.parquet
   evaporation_violation_cost at stage 0       =  6.2384 × 10⁸ R$  ✓
```

Match within rounding. The slack is in m³/s, the penalty is in
R$/(m³/s·h), and `total_stage_hours` is the multiplier that joins them.

### Penalty conversion on the cobre-bridge side

`src/cobre_bridge/converters/network.py` (Family C):

```python
evaporation_cost = _EVAPORATION_MULT * max_deficit_cost * rho_max_acum
#  [R$/(m³/s·h)] = 1.1 ·  [R$/MWh]      ·  [MW/(m³/s)]
```

`MAX_DEFICIT` is the deficit cost in R$/MWh and `ρ_max_acum` is in
MW/(m³/s). The product is in R$/(m³/s·h) — exactly the unit cobre
expects. **No change needed in the converter.**

### Suggested cobre-side cleanup (independent of the main fix)

Whichever of options A/B/C above is chosen, the
`LinearizedEvaporation` docstring should be corrected from

```text
/// The evaporation volume (hm³) is approximated as:
/// pub k_evap0: f64,   // Constant term of the linearized evaporation (hm³).
/// pub k_evap_v: f64,  // Volume-dependent slope (hm³/hm³ = dimensionless).
```

to:

```text
/// The evaporation flow (m³/s, stage-averaged) is approximated as:
///     Q_ev = k_evap0 + k_evap_v * (V - V_ref)
/// pub k_evap0: f64,   // Constant term of the linearized evap (m³/s).
/// pub k_evap_v: f64,  // Volume-dependent slope ((m³/s) / hm³).
```

`simulation/types.rs:165` field name `evaporation_violation_pos_m3s`
already correctly advertises m³/s — only the linearized-coefficient
docstring is misleading.

---

## Reproduction

From cobre-bridge `develop` branch:

```bash
# Generate the cobre case from the bundled NEWAVE example
cobre-bridge convert newave example/newave example/convertido --force

# Run cobre against the case (50-iter, 200-fwd config recommended)
# Then compare:
cobre-bridge compare results example/newave example/convertido/output -o cmp.html
```

Cost composition is reproducible from
`example/convertido/output/simulation/costs/` (Polars `scan_parquet`,
discount with `discount_factor`, sum per scenario, mean across scenarios).
Per-plant slack firing patterns are in `simulation/hydros/` —
filter `evaporation_violation_pos_m3s > 0` and group by `hydro_id`.

Relevant cobre source (HEAD, 2026-05-18):

- `crates/cobre-sddp/src/lp_builder/matrix.rs:306-348` — column bounds and objective
- `crates/cobre-sddp/src/lp_builder/matrix.rs:629-647` — row RHS
- `crates/cobre-sddp/src/lp_builder/matrix.rs:1062-1113` — row CSC entries
- `crates/cobre-sddp/src/hydro_models.rs:170-214` — `EvaporationModel` definition
- `crates/cobre-sddp/src/hydro_models.rs:1615-1626` — `k_evap0`/`k_evap_v` derivation
- `crates/cobre-sddp/src/simulation/types.rs:165-167` — `evaporation_violation_pos_m3s` / `_neg_m3s`

Relevant cobre-bridge source:

- `src/cobre_bridge/converters/hydro.py:528` — `evap_coeffs` read verbatim from `hidr.dat`
- `src/cobre_bridge/converters/network.py:166` — `_EVAPORATION_MULT = 1.1` rationale
