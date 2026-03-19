# ticket-002: Apply TEIF/IP Availability Derating to Generation Bounds

## Context

### Background

NEWAVE hydro plants have two availability rates: TEIF (forced unavailability rate, %) and IP (scheduled unavailability rate, %). These reduce the effective maximum generation capacity. The current converter ignores these values entirely, producing overly optimistic max_generation_mw bounds.

### Relation to Epic

Second ticket in Epic 01. Depends on ticket-001 only in that both modify the same conversion loop, but the derating logic is independent of the productivity calculation. The derating applies to `max_generation_mw` which is computed from nominal powers (not from productivity).

### Current State

File `src/cobre_bridge/converters/hydro.py`, lines 106-112: max_generation is computed as the sum of `potencia_nominal_conjunto_i * maquinas_conjunto_i` with no derating applied. The hidr.dat cadastro has `teif` and `ip` columns (percentage values, e.g., 5.0 means 5%).

## Specification

### Requirements

1. After computing `max_generation` from machine sets (line 112), apply the derating formula: `max_generation *= ((100 - teif) / 100) * ((100 - ip) / 100)`
2. Read `teif` and `ip` from `hreg` (the cadastro row)
3. Handle missing/zero values gracefully (if teif or ip is 0 or NaN, treat as 0 -- no derating for that component)

### Inputs/Props

`teif` and `ip` are float columns in the hidr.dat cadastro, expressed as percentages (0-100).

### Outputs/Behavior

`max_generation_mw` in the output JSON is reduced by the derating factor. If both TEIF=5 and IP=3, the factor is `0.95 * 0.97 = 0.9215`.

### Error Handling

- If `teif` or `ip` is NaN or missing, treat as 0.0 (no derating)
- If `teif` or `ip` exceeds 100, clamp to 100 (would zero out generation -- log a warning)

## Acceptance Criteria

- [ ] Given a plant with `teif=5.0`, `ip=3.0`, and nominal max_generation=800 MW, when `convert_hydros` is called, then `generation.max_generation_mw` equals `800 * 0.95 * 0.97 = 737.2`
- [ ] Given a plant with `teif=0.0` and `ip=0.0`, when `convert_hydros` is called, then `generation.max_generation_mw` equals the unmodified sum of nominal powers
- [ ] Given a plant with `teif=NaN` (missing), when `convert_hydros` is called, then the derating treats teif as 0.0 without raising an error
- [ ] Given the updated `_make_hidr_cadastro()` test fixture includes `teif` and `ip` columns, when `TestConvertHydros` tests run, then all tests pass

## Implementation Guide

### Suggested Approach

1. In `convert_hydros`, after the machine-set loop (after line 112), read teif and ip:
   ```python
   teif = float(hreg.get("teif", 0.0) or 0.0)
   ip = float(hreg.get("ip", 0.0) or 0.0)
   ```
2. Handle NaN: `if math.isnan(teif): teif = 0.0` (same for ip)
3. Apply: `max_generation *= ((100 - teif) / 100) * ((100 - ip) / 100)`
4. Update `_make_hidr_cadastro()` in tests to add `teif` and `ip` columns
5. Add a dedicated test for the derating calculation

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` -- add 3-4 lines after line 112
- `tests/test_entity_conversion.py` -- update fixture, add test method

### Patterns to Follow

- Same NaN-safe reading pattern used for `vazao_minima_historica` at lines 115-120
- Use `math.isnan` guard (already imported in `_is_na` helper)

### Pitfalls to Avoid

- Do NOT apply derating to `max_turbined_m3s` -- only `max_generation_mw` is derated
- Do NOT apply derating to `min_generation_mw` -- minimum generation is a physical constraint, not affected by availability
- The `teif` column name in hidr.dat uses lowercase -- verify against the actual inewave cadastro column names

## Testing Requirements

### Unit Tests

- `TestConvertHydros.test_teif_ip_derating_reduces_max_generation` -- verify formula
- `TestConvertHydros.test_zero_teif_ip_no_derating` -- verify no change when both are zero

### Integration Tests

N/A

### E2E Tests (if applicable)

N/A

## Dependencies

- **Blocked By**: ticket-001-implement-average-productivity.md (shares test fixture changes)
- **Blocks**: None

## Effort Estimate

**Points**: 1
**Confidence**: High
