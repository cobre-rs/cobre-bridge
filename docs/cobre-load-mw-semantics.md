# Investigation: load_mw Output Semantics

## Summary

The `load_mw` column in the simulation bus output does NOT equal the sum of generation supply. The per-block energy balance `hydro + thermal + ncs + import + deficit - excess = load_mw` does NOT close when using the output `load_mw`.

## Observations

At stage 0, block 0, scenario 0:

| Bus | Hydro  | Thermal | NCS   | Import | Deficit | Excess | Supply | load_mw | Gap     |
| --- | ------ | ------- | ----- | ------ | ------- | ------ | ------ | ------- | ------- |
| 0   | 54,430 | 2,381   | 6,279 | -8,644 | 0       | 0      | 54,446 | 43,478  | +10,968 |
| 1   | 5,229  | 1,137   | 2,460 | 8,644  | 0       | 0      | 17,471 | 11,720  | +5,751  |
| 2   | 7,297  | 28      | 6,698 | 0      | 0       | 0      | 14,023 | 13,095  | +928    |
| 3   | 16,149 | 675     | 0     | 0      | 0       | 8,527  | 8,297  | 9,896   | -1,599  |

System total gap: +16,048 MW.

## Comparisons

- Deterministic LP load (from input `load_seasonal_stats * load_factor`): 96,682 MW
- Output `load_mw` sum: 78,188 MW
- Difference: 18,494 MW (close to total NCS available: 18,827 MW)

This suggests `load_mw` in the output might be: `LP_RHS_perturbed - NCS_available` or a similar net quantity, but the exact formula needs verification from cobre source code.

## Impact on Dashboard

The Investigation tab's "Per-Block Energy Balance" chart compares `hydro + thermal + NCS` (stacked) against `load_mw` (line). Since `load_mw` appears to already net out NCS, the chart shows a systematic gap between supply and demand that doesn't represent a real imbalance.

## Action Required

Clarify the exact semantics of `load_mw` in cobre's simulation output and update the dashboard chart accordingly.
