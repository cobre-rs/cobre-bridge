# ticket-001: Implement Average Productivity Calculation

## Context

### Background

The current hydro converter at `src/cobre_bridge/converters/hydro.py` line 102 directly uses the `produtibilidade_especifica` column from hidr.dat as the productivity value. This is incorrect -- `produtibilidade_especifica` is measured in MW/((m^3/s)/m) and is a coefficient that must be multiplied by the net drop (adjusted head) to produce the actual productivity in MW/(m^3/s).

The correct calculation requires:

1. Evaluating the `volume_cota` polynomial to map volume to upstream height
2. Subtracting `canal_fuga_medio` (downstream level) to get gross drop
3. Applying the loss model based on `tipo_perda` (1=multiplicative factor, 2=additive meters)
4. For monthly-regulated plants (`tipo_regulacao == "M"`): computing the integral average of the production function over [volume_minimo, volume_maximo] using the mean value theorem
5. For other plants: evaluating at `volume_referencia`
6. Multiplying `produtibilidade_especifica * adjusted_net_drop` to get final productivity

### Relation to Epic

This is the core correctness fix for Epic 01. Tickets 002 and 003 build on the corrected productivity values.

### Current State

File `src/cobre_bridge/converters/hydro.py`, line 102:

```python
productivity = float(hreg["produtibilidade_especifica"])
```

The hidr.dat cadastro DataFrame contains these relevant columns per plant:

- `produtibilidade_especifica` -- coefficient in MW/((m^3/s)/m)
- `volume_minimo`, `volume_maximo` -- storage bounds in hm3
- `volume_referencia` -- reference volume for non-monthly-regulated plants
- `canal_fuga_medio` -- downstream water level in meters
- `tipo_perda` -- loss model type (1=multiplicative, 2=additive)
- `perdas` -- loss value (fraction if tipo_perda=1, meters if tipo_perda=2)
- `tipo_regulacao` -- "M" for monthly regulation, others for run-of-river etc.
- `volume_cota_0` through `volume_cota_4` -- 5th-degree polynomial coefficients mapping volume (hm3) to upstream height (m)

## Specification

### Requirements

1. Add a helper function `_compute_productivity(hreg: pd.Series) -> float` in `hydro.py` that:
   - Reads polynomial coefficients `volume_cota_0..4` from the plant's cadastro row
   - For `tipo_regulacao == "M"` plants: computes the integral average of `(poly(v) - canal_fuga_medio)` over `[volume_minimo, volume_maximo]` using the antiderivative of the polynomial, then applies the loss model, then multiplies by `produtibilidade_especifica`
   - For other plants: evaluates `poly(volume_referencia) - canal_fuga_medio`, applies loss model, multiplies by `produtibilidade_especifica`
2. Replace line 102 with a call to this helper
3. Update `min_generation` calculation at line 121 to use the new productivity value

### Inputs/Props

The function receives a pandas Series `hreg` (one row of hidr.cadastro indexed by column name).

### Outputs/Behavior

Returns a `float` representing the plant's average productivity in MW/(m^3/s).

### Error Handling

- If `volume_minimo == volume_maximo` for a tipo_regulacao="M" plant, fall back to point evaluation at volume_minimo (avoid division by zero in integral average)
- If polynomial coefficients are all zero, log a warning and return `produtibilidade_especifica * 0.0` (zero productivity -- the plant has no height)

## Acceptance Criteria

- [ ] Given a plant with `tipo_regulacao="M"`, `volume_cota_0=300, volume_cota_1=0.1, volume_cota_2=0, volume_cota_3=0, volume_cota_4=0`, `volume_minimo=100`, `volume_maximo=1000`, `canal_fuga_medio=250`, `tipo_perda=1`, `perdas=0.05`, `produtibilidade_especifica=0.009`, when `_compute_productivity` is called, then the returned value equals `produtibilidade_especifica * (1 - 0.05) * (avg_height - 250)` where `avg_height` is the integral average of the linear polynomial `300 + 0.1*v` over `[100, 1000]`
- [ ] Given a plant with `tipo_regulacao="D"` (run-of-river) and `volume_referencia=500`, when `_compute_productivity` is called, then it evaluates `poly(500) - canal_fuga_medio`, applies loss, and multiplies by `produtibilidade_especifica`
- [ ] Given a plant with `tipo_perda=2` and `perdas=3.5`, when computing adjusted drop, then `adjusted_drop = net_drop - 3.5` (additive loss)
- [ ] Given the existing test fixtures in `tests/test_entity_conversion.py`, when the tests are updated to include volume_cota columns and loss parameters, then all hydro tests pass with the new productivity formula
- [ ] Given the `convert_hydros` function output, when examining `generation.productivity_mw_per_m3s`, then the value differs from raw `produtibilidade_especifica` for any plant with nonzero `canal_fuga_medio`

## Implementation Guide

### Suggested Approach

1. Add numpy as a dependency if not already present (check pyproject.toml -- it is not listed, but numpy comes transitively via pandas; use `numpy.polynomial.polynomial` or manual integration)
2. Extract polynomial coefficients from the cadastro row. The hidr.dat columns are named `volume_cota_0` through `volume_cota_4` (5 coefficients for a 4th-degree polynomial in standard form: `h(v) = c0 + c1*v + c2*v^2 + c3*v^3 + c4*v^4`)
3. For tipo_regulacao="M": compute the antiderivative analytically (`C0*v + C1*v^2/2 + ...`), evaluate at vol_max and vol_min, divide by `(vol_max - vol_min)` to get average height. Subtract `canal_fuga_medio` for average net drop.
4. For other plants: evaluate polynomial at `volume_referencia`, subtract `canal_fuga_medio`.
5. Apply loss model to get adjusted drop.
6. Return `produtibilidade_especifica * adjusted_drop`.
7. Update the test fixtures in `_make_hidr_cadastro()` to include all new columns.

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` -- add `_compute_productivity`, replace line 102
- `tests/test_entity_conversion.py` -- update `_make_hidr_cadastro()` fixtures, add `TestComputeProductivity` class

### Patterns to Follow

- Follow the existing pattern of reading columns from `hreg` Series by name
- Keep the helper private (`_compute_productivity`) since it is an implementation detail
- Use standard Python math for polynomial evaluation (no need for numpy -- it's a 4th-degree polynomial)

### Pitfalls to Avoid

- Do NOT use `numpy.polyval` which expects highest-degree-first coefficient order. The hidr.dat coefficients are lowest-degree-first (`volume_cota_0` is the constant term).
- Do NOT forget to handle the `volume_minimo == volume_maximo` edge case for tipo_regulacao="M" plants
- The existing test fixture at `_make_hidr_cadastro()` does not include volume_cota, tipo_regulacao, tipo_perda, perdas, canal_fuga_medio, or volume_referencia columns. These must all be added or the tests will break.

## Testing Requirements

### Unit Tests

- `TestComputeProductivity.test_monthly_regulated_linear_polynomial` -- tipo_regulacao="M" with a simple linear polynomial, verify integral average calculation
- `TestComputeProductivity.test_run_of_river_point_evaluation` -- tipo_regulacao="D", verify point evaluation at volume_referencia
- `TestComputeProductivity.test_multiplicative_loss` -- tipo_perda=1
- `TestComputeProductivity.test_additive_loss` -- tipo_perda=2
- `TestComputeProductivity.test_no_loss` -- tipo_perda=0 or unknown
- `TestComputeProductivity.test_equal_volumes_fallback` -- volume_minimo == volume_maximo, tipo_regulacao="M"

### Integration Tests

- Update `TestConvertHydros.test_generation_values_match_machine_sets` to use new productivity

### E2E Tests (if applicable)

N/A

## Dependencies

- **Blocked By**: None
- **Blocks**: ticket-002-apply-teif-ip-derating.md, ticket-003-add-hydraulic-losses-field.md

## Effort Estimate

**Points**: 3
**Confidence**: High
