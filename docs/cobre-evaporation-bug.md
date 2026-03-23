# Bug Report: Evaporation Linearization Produces Values 1000-10000x Too Large

## Summary

The LP evaporation values (`evaporation_m3s` in simulation output) are orders of magnitude larger than expected for many hydro plants. The water balance closes perfectly using the reported value, confirming it IS the value used in the LP. The `evaporation_violation_m3s` is a separate diagnostic — it does NOT reduce the applied evaporation.

## Semantics

The LP water balance constraint is:

```
v_out = v_in + zeta * (inflow - evaporation_m3s - water_withdrawal) - sum_blocks(tau_b * (turb_b + spill_b))
```

`evaporation_m3s` is the value applied inside this constraint. For TUCURUI (stage 0, scenario 0):

| Term             | Value          |
| ---------------- | -------------- |
| v_in             | 30,822.98 hm3  |
| v_out            | 50,275.00 hm3  |
| inflow           | 20,769.02 m3/s |
| evaporation_m3s  | 5,144.54 m3/s  |
| water_withdrawal | 17.93 m3/s     |
| outflow_vol      | 22,348.57 hm3  |
| **balance diff** | **0.0000 hm3** |

The balance closes to machine precision.

`evaporation_violation_m3s` does NOT subtract from `evaporation_m3s`. For TUCURUI: evap=5144.54, violation=5140.68. If the violation were a slack reducing evaporation, the net would be 3.85 m3/s and the balance would be off by 13,769 hm3.

## Magnitude Error

Expected evaporation: `area_km2 * coefficient_mm / 1000` gives hm3, converted to m3/s via `hm3 * 1e6 / (stage_hours * 3600)`.

| Plant        | Geo Pts | Area (km2) | Coeff (mm) | Expected (m3/s) | LP Value (m3/s) | Violation (m3/s) | Ratio        |
| ------------ | ------- | ---------- | ---------- | --------------- | --------------- | ---------------- | ------------ |
| SOBRADINHO   | 100     | 2,771      | 61         | 63.12           | 82.7            | 0.0              | **1x**       |
| STO ANT JARI | 100     | 29         | 38         | 0.42            | 0.4             | 0.0              | **1x**       |
| PEIXE ANGIC  | 100     | 264        | 11         | 1.08            | 1.1             | 0.0              | **1x**       |
| TUCURUI      | 100     | 2,080      | 4          | 3.11            | 5,144.5         | 5,140.7          | **1,656x**   |
| BELO MONTE   | 100     | 342        | 8          | 1.02            | 12,904.5        | 12,903.5         | **12,631x**  |
| JIRAU        | 100     | 205        | -93        | -7.13           | 4,476.6         | 4,483.8          | **-628x**    |
| ESTREITO TOC | 1 (RoR) | 590        | 4          | 0.88            | 6,348.6         | 6,347.7          | **7,205x**   |
| DARDANELOS   | 1 (RoR) | 0.2        | 18         | 0.00            | 570.6           | 570.6            | **353,759x** |

## Root Cause (Confirmed)

The evaporation violation slack has the same cost as spillage (both 0.1 R$/m3s). The LP uses evaporation violation as a free spillage pathway — it's cheaper (or equal) to "violate" the evaporation constraint than to spill water.

For TUCURUI at stage 0:

- Inflow volume: 55,628 hm3
- Reservoir room: 50,275 - 30,823 = 19,452 hm3
- Turbine outflow: 22,349 hm3
- Excess to dump: 55,628 - 19,452 - 22,349 = 13,827 hm3
- LP dumps via evaporation violation (5,141 m3/s) instead of spillage (0 m3/s)

## Deep Investigation: Why the LP Inflates Evaporation

The evaporation linearization code in cobre is **mathematically correct**. The coefficients `k_evap0` and `k_evap_v` are computed correctly from the geometry and monthly coefficients. A manual Taylor expansion at the midpoint volume matches the expected evaporation values (~4 m3/s for TUCURUI).

The problem is that the LP **deliberately inflates** Q_ev above the linearized value to dump excess water. The evaporation violation slack absorbs the excess:

For TUCURUI at stage 0 (scenario 0):

- Inflow volume: 55,628 hm3
- Turbine outflow: 22,296 hm3
- With correct evaporation (~4 m3/s): v_out would be **64,096 hm3** — exceeding vol_max (50,275) by 13,821 hm3
- The LP sets Q_ev = 5,164 m3/s (instead of 4) to dump exactly 13,821 hm3
- The evaporation violation slack (f_minus = 5,160 m3/s) absorbs the excess

The mechanism: Q_ev appears in the water balance as `- zeta * Q_ev`, so inflating Q_ev removes water at the **stage level** without using the per-block spillage variables.

### Why Not Spillage?

Spillage should be preferred (0.1 R$/m3s × block_hours ≈ 12-32 R$ per m3/s per block). But the LP uses evaporation violation (100 R$/m3s) instead. This suggests either:

1. The spillage penalty is applied differently than expected (per-block vs per-stage accounting)
2. There are block-level constraints that make spillage infeasible in some blocks
3. The evaporation violation penalty is NOT being applied correctly (cost might be divided by zeta or stage_hours somewhere)

### Recommended Fix (in cobre)

The evaporation violation should be **asymmetric**:

- `f_minus` (Q_ev > linearized): should have VERY high cost (prohibit over-evaporation), or the Q_ev variable should have an upper bound
- `f_plus` (Q_ev < linearized): can keep moderate cost (linearization approximation error)

**Update: Root cause is LP prescaling, not cost values.**

The prescaler inverts the cost ordering for large reservoirs:

1. Water balance RHS for TUCURUI is ~55,000 hm3 (zeta × inflow)
2. Row prescaler divides the row by ~55,000 to normalize
3. Spillage column coefficients (tau_b ≈ 0.43) become ~7.7e-6 after row scaling
4. Column prescaler compensates by scaling UP spillage columns by ~130,000x
5. Effective spillage objective: `0.012 × 130,000 = 1,528`
6. Evaporation violation (f_minus) lives in the evaporation constraint (coefficients ≈ 1.0), scale stays ~1.0
7. Effective f_minus objective: `74.4 × 1.0 = 74.4`
8. **After prescaling: spillage (1,528) appears 20x MORE expensive than evap violation (74.4)**

The solver correctly picks the "cheaper" option (evaporation violation) in the scaled space, but this is the WRONG answer in the original space (spillage should be 6,200x cheaper).

### Fix should target cobre's prescaler

The prescaler must preserve cost ordering between variables that share a constraint. Options:
- Pin spillage column scale to match evaporation column scale
- Use a different scaling strategy for water balance rows (e.g., geometric mean scaling)
- Add Q_ev upper bounds (as below)
- Don't column-scale variables that participate in both large-RHS and small-RHS rows

A simpler fix: add a column upper bound on Q_ev equal to `max_area_km2 * max_coefficient_mm / (3.6 * stage_hours)`, preventing the LP from using Q_ev as a spillage pathway.

## Partial Fix Applied (in cobre-bridge)

Changed `_DEFAULT_EVAPORATION_VIOLATION_COST` from `0.1` to `100.0` in `src/cobre_bridge/converters/network.py`. This reduced the issue but did NOT eliminate it — the LP still inflates evaporation when it needs to dump water and evaporation violation is the cheapest stage-level mechanism.

## Environment

- cobre v0.1.9+
- Case: Brazilian NEWAVE PMO, 158 hydros, 118 stages
- Evaporation coefficients from NEWAVE hidr.dat (net evaporation in mm/month)
- Geometry from hidr.dat volume-area polynomials (100-point VHA curves)
- Evaporation linearization code verified correct (hydro_models.rs lines 1419-1585)
- Q_ev output extraction verified: reads directly from LP primal solution (extraction.rs)
- 21 of 150 plants with evaporation show inflated values (ratio > 10x); all occur during wet-season stages when reservoirs are filling
