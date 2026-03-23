# Bug Report: Spillage Cost Extraction Inflated by Column Prescale Factor

## Summary

The `spillage_cost` column in simulation hydro output and the aggregated `spillage_cost` in the costs output are inflated by a factor of exactly **83,000x** (83 × cost_scale_factor). This is specific to spillage — thermal, exchange, and excess costs are correct. The likely cause is that the spillage cost extraction reads the LP objective contribution without undoing the column prescale factor.

## Reproduction

Case: `example/convertido` (158 hydros, 118 stages, 50 training iterations, 100 simulation scenarios)

## Verification Method

Manual computation of spillage cost using the penalty formula:

```
spillage_cost = spillage_penalty × block_hours × spillage_m3s
             = 0.1 × hours × m3s
```

## Per-Component Verification (Stage 0, Scenario 0)

| Component         | Output Value        | Manual Value  | Ratio       | Status     |
| ----------------- | ------------------- | ------------- | ----------- | ---------- |
| thermal_cost      | 357,599,829         | 357,599,829   | **1.0x**    | Correct    |
| exchange_cost     | 128,643             | 128,643       | **1.0x**    | Correct    |
| excess_cost       | 1,745,853           | 1,745,853     | **1.0x**    | Correct    |
| curtailment_cost  | 1,130,150           | 2,365,133     | **0.5x**    | Off by 2x  |
| **spillage_cost** | **361,238,353,966** | **4,352,269** | **83,000x** | **BROKEN** |

## The 83,000x Factor

The ratio is consistent across ALL hydro plants (verified for top 10 spillers):

```
JIRAU:         spillage_cost = 32,166,141,933 / manual 387,521 = 83,005x
STO ANTONIO:   spillage_cost = 24,647,933,762 / manual 296,945 = 83,005x
TUCURUI:       spillage_cost = 19,378,703,281 / manual 233,464 = 83,005x
...all consistent at 83,000-83,005x
```

The factor decomposes as: **83,000 = 83 × cost_scale_factor (1000)**

The 83 factor is NOT explained by:

- `col_scale_max` (592.18) — doesn't match
- `1/row_scale_min` (592) — doesn't match
- `zeta` (2.678) — doesn't match
- Stage count or block count — doesn't match

It may be a product of the column prescale factor for the specific spillage column of the hydro being extracted, which varies per hydro and stage.

## Per-Block Verification (JIRAU, Stage 0)

```
Block 0: spill=32,293 m3/s, hours=120
  Output spillage_cost:  32,166,141,933
  Manual: 0.1 × 120 × 32,293 = 387,516
  Effective penalty: 32,166,141,933 / 32,293 / 120 = 8,300 R$/m3s (vs expected 0.1)
  Factor: 8,300 / 0.1 = 83,000
```

Block 1 and 2 have zero spillage, so their contribution is zero (consistent).

## Hypothesis: Column Prescale Not Undone

In the prescaled LP:

1. Each column is scaled by a `col_scale` factor to normalize matrix coefficients
2. The LP objective coefficient becomes: `obj_scaled = obj_original × col_scale`
3. The LP primal solution is: `x_scaled = x_original / col_scale`
4. The cost contribution: `obj_scaled × x_scaled = obj_original × x_original` (cancels)
5. But if the extraction computes: `obj_original × x_original × col_scale` (not cancelling), the result is inflated by `col_scale`

If `col_scale ≈ 83` for the spillage column at stage 0, this would produce the observed 83x factor. Multiplied by `cost_scale_factor = 1000` gives 83,000x.

Alternatively, the extraction may read the prescaled objective coefficient (`obj_scaled`) and multiply by the unscaled primal (`x_original`), giving `obj_original × col_scale × x_original` — inflated by `col_scale`.

## Impact on Dashboard

1. **Cost Breakdown chart**: Spillage dominates (99%+ of immediate costs), hiding thermal and other real costs
2. **Cost Composition by Stage**: Spillage artifact overwhelms the stacked area chart
3. **Total cost metrics**: Trillions instead of millions per stage
4. **Future cost (alpha)**: May also be affected if cuts store prescaled coefficients, making the total_cost nonsensical

## What Is NOT Affected

- `spillage_m3s` in hydro output — correct (read from LP primal, properly unscaled)
- `thermal_cost` — correct (1.0x ratio)
- `exchange_cost` — correct (1.0x ratio)
- `excess_cost` — correct (1.0x ratio)
- `spot_price` — correct (reasonable values -0.18 to 176 R$/MWh)
- `generation_mw` — correct (matches productivity × turbined exactly)

## Additional Finding: curtailment_cost Off by 2x

The `curtailment_cost` output is 0.5x the expected value. This may be a separate issue (e.g., only counting one of the two curtailment components, or averaging instead of summing across blocks).

## Additional Finding: Water Values

TUCURUI `water_value_per_hm3` = 2,305,556 R$/hm3. Dividing by `cost_scale_factor` (1000) gives 2,306 R$/hm3, which is plausible for a poorly converged policy (5 iterations). NEWAVE reports -13,885 R$/hm3 for the same plant (well-converged with 10,000 cuts). The sign difference is expected (different policy quality), but the magnitude warrants verification that `water_value_per_hm3` is properly unscaled.

## Environment

- cobre v0.2.0-dev (post scaling implementation, post S1 model persistence)
- cost_scale_factor: 1000
- Column prescale range: [0.372, 592.18]
- Row prescale range: [0.00169, 34.01]
