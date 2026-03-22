# Bug Report: Water Balance Inconsistency in Simulation Output

## Summary

The simulation output for certain hydro plants shows a water balance that does not close — the total outflow (turbined + spillage) across all blocks significantly exceeds the total inflow, yet storage remains unchanged. This leads to physically impossible spillage values and distorted water values (shadow prices).

## Reproduction

Case: `example/convertido` (converted from NEWAVE using cobre-bridge)

Plant: **PIMENTAL** (cobre hydro_id=156, NEWAVE code 314)

- Headwater plant (no upstream cascade)
- vol_min = 1770.9 hm3, vol_max = 2276.3 hm3
- max_turbined = 2436 m3/s
- All input data verified correct against NEWAVE source

## Observed Behavior

**Scenario 1, Stage 0 (March 2026), all blocks:**

| Block      | Hours | Inflow (m3/s) | Turbined (m3/s) | Spillage (m3/s) | Storage Init/Final (hm3) |
| ---------- | ----- | ------------- | --------------- | --------------- | ------------------------ |
| 0 (HEAVY)  | 120   | 1538          | 2436            | 1864            | 1771 / 1771              |
| 1 (MEDIUM) | 317   | 1538          | 2436            | 0               | 1771 / 1771              |
| 2 (LIGHT)  | 307   | 1538          | 2436            | 0               | 1771 / 1771              |

No evaporation, no diversions, no upstream cascade, no nonnegativity slack.

## Water Balance Check

Using the formula from `lp_builder.rs`:

```
v_out = v_in + zeta * inflow - sum_blocks[tau_b * (turb_b + spill_b)]
```

Where:

- `zeta = 744 * 3600 / 1e6 = 2.6784` (total month hours × M3S_TO_HM3)
- `tau_0 = 120 * 3600 / 1e6 = 0.432`
- `tau_1 = 317 * 3600 / 1e6 = 1.1412`
- `tau_2 = 307 * 3600 / 1e6 = 1.1052`

```
inflow volume     = 2.6784 × 1538 = 4119.4 hm3
block 0 outflow   = 0.432 × (2436 + 1864) = 1857.6 hm3
block 1 outflow   = 1.1412 × 2436 = 2780.0 hm3
block 2 outflow   = 1.1052 × 2436 = 2692.3 hm3
total outflow     = 7329.8 hm3

expected v_out    = 1771 + 4119.4 - 7329.8 = -1439.5 hm3
actual v_out      = 1771.0 hm3
discrepancy       = 3210.5 hm3
```

The reservoir would need to go to **-1440 hm3** (far below the physical minimum of 1771), yet storage stays at minimum. The LP reports zero slack and zero violations.

## Impact on Results

1. **Physically impossible spillage**: Block 0 shows 1864 m3/s spillage when inflow is only 1538 m3/s and the plant is already turbining at maximum. The total outflow (4300 m3/s in block 0) vastly exceeds inflow.

2. **77 out of 100 scenarios show spillage** at PIMENTAL stage 0 (avg 26,309 m3/s in block 0 across all scenarios), despite the plant being a headwater with manageable inflow.

3. **Distorted water values**: PIMENTAL shows water values of +1,753,056 R$/hm3 (stage 0) and -11,468 R$/hm3 (stages 12+). The extreme positive value in wet months suggests the optimizer assigns enormous value to water that it's simultaneously spilling.

4. **Only 2 plants spill** (PIMENTAL: 61,384 m3/s avg, FONTES: 413 m3/s avg) — all spillage is concentrated rather than distributed realistically.

## Verified: Input Data is Correct

All PIMENTAL input data was verified against the NEWAVE source:

| Parameter         | Cobre Value                   | NEWAVE Value              | Match |
| ----------------- | ----------------------------- | ------------------------- | ----- |
| vol_min           | 1770.9 hm3                    | 1770.9 hm3                | Yes   |
| vol_max           | 2276.3 hm3                    | 2276.3 hm3                | Yes   |
| max_turbined      | 2436.0 m3/s                   | 2436.0 m3/s               | Yes   |
| max_gen (derated) | 223.4 MW                      | 223.4 MW                  | Yes   |
| Initial storage   | 1770.9 hm3                    | 1770.9 hm3 (0% useful)    | Yes   |
| Downstream        | None (sea)                    | 0 (sea)                   | Yes   |
| Upstream plants   | 0                             | 0                         | Yes   |
| Inflow history    | mean=2424, min=0, 1152 months | Same                      | Yes   |
| Posto             | 302                           | 302                       | Yes   |
| Geometry          | 100 rows, valid VHA           | From hidr.dat polynomials | Yes   |
| Water withdrawal  | 2-3 m3/s (from dsvagua)       | Same                      | Yes   |

The bug is in cobre's LP formulation or solve-time patching, not in the input data.

## Possible Root Causes

1. **Inflow RHS computation**: The `deterministic_base` used in the water balance RHS may not match the `inflow_m3s` reported in the output. If the RHS includes a much larger base value (e.g., the seasonal mean of 5722 m3/s instead of the realized 1538 m3/s), the LP would have more water available than reported.

2. **Noise perturbation sign/magnitude**: The PAR noise innovation added at solve time may have the wrong sign or scaling, effectively injecting water into the balance that doesn't appear in the output's `inflow_m3s`.

3. **Zeta/tau mismatch**: If the block-level `tau` and stage-level `zeta` factors are inconsistent (e.g., `zeta` used for outflows instead of `tau`), the volume accounting would be wrong.

4. **Output reporting error**: The `inflow_m3s` in the simulation output may not correctly reflect the actual RHS value used in the LP solve for that scenario.

## Environment

- cobre v0.1.9
- cobre-bridge: latest (with C_ADIC, EXPT, MANUTT, CVaR, AGRINT conversions)
- Case: Brazilian NEWAVE PMO example, 158 hydros, 104 thermals, 5 buses, 118 stages
