# ticket-006: Parse GHMIN.DAT for Minimum Generation Constraints

## Context

### Background

The current hydro converter computes `min_generation` as `min_outflow * productivity` (line 121 of hydro.py), which is a rough approximation. NEWAVE provides explicit per-plant minimum generation constraints in GHMIN.DAT. When a plant has an entry in GHMIN.DAT, that value should be used instead of the approximation.

### Relation to Epic

Third ticket in Epic 02. Independent of ticket-004/005 (MODIF.DAT), but logically grouped with them as "additional NEWAVE file parsing". Can be implemented in parallel with ticket-005.

### Current State

File `src/cobre_bridge/converters/hydro.py`, line 121:

```python
min_generation = min_outflow * productivity
```

GHMIN.DAT exists in the example case at `/home/rogerio/git/cobre-bridge/example/newave/GHMIN.DAT`. The inewave library likely provides a `Ghmin` reader.

## Specification

### Requirements

1. Add a helper function `_read_ghmin(newave_dir: Path) -> dict[int, float]` that:
   - Reads GHMIN.DAT using inewave
   - Returns a dict mapping plant code to minimum generation in MW
   - Returns empty dict if GHMIN.DAT does not exist
2. In the conversion loop, after computing the default min_generation, check if the plant has a GHMIN entry:
   ```python
   ghmin_value = ghmin_map.get(newave_code)
   if ghmin_value is not None:
       min_generation = ghmin_value
   ```
3. Add `ghmin.dat` to the pipeline's optional file awareness (no hard requirement -- graceful fallback)

### Inputs/Props

- `newave_dir`: Path to the NEWAVE case directory

### Outputs/Behavior

Returns a `dict[int, float]` mapping NEWAVE plant codes to minimum generation values in MW.

### Error Handling

- Missing GHMIN.DAT: return empty dict, log debug message
- Plant codes in GHMIN.DAT not found in confhd: ignored (they are for other plants not in study)
- Zero or negative values: use as-is (0.0 means no minimum generation constraint)

## Acceptance Criteria

- [ ] Given GHMIN.DAT contains an entry `{code: 1, min_generation: 50.0}`, when `_read_ghmin` is called, then it returns `{1: 50.0}`
- [ ] Given GHMIN.DAT contains an entry for plant 1 with min_generation=50.0, when `convert_hydros` processes plant 1, then `generation.min_generation_mw` is 50.0 (not the approximation)
- [ ] Given GHMIN.DAT does not exist, when `_read_ghmin` is called, then it returns `{}` without error
- [ ] Given a plant has no entry in GHMIN.DAT, when `convert_hydros` processes it, then `min_generation_mw` uses the existing approximation `min_outflow * productivity`

## Implementation Guide

### Suggested Approach

1. Investigate the inewave Ghmin API:
   ```python
   from inewave.newave import Ghmin
   ghmin = Ghmin.read(str(newave_dir / "ghmin.dat"))
   # Inspect ghmin properties -- likely a DataFrame with codigo_usina and valor columns
   ```
2. Build the plant-code-to-value mapping from the DataFrame
3. Call `_read_ghmin` once at the top of `convert_hydros`, pass the result dict into the loop
4. Replace the min_generation assignment with a conditional

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` -- add `_read_ghmin`, modify lines 114-121
- `tests/test_entity_conversion.py` -- add tests with mocked Ghmin data

### Patterns to Follow

- Same file-existence check and inewave reader pattern as existing code
- Same mocking pattern for tests

### Pitfalls to Avoid

- GHMIN.DAT may have multiple entries per plant (different periods/stages). Check if the inewave parser returns a single value or time-series. If time-series, take the value for the first study period.
- The inewave column name for the generation value may be `"valor"`, `"geracao_minima"`, or similar -- inspect the actual DataFrame
- Do NOT remove the fallback approximation -- it is still needed for plants without GHMIN entries

## Testing Requirements

### Unit Tests

- `TestReadGhmin.test_reads_plant_min_generation` -- verify correct parsing
- `TestReadGhmin.test_missing_ghmin_returns_empty` -- graceful handling
- `TestConvertHydros.test_ghmin_overrides_approximation` -- verify min_generation_mw uses GHMIN value
- `TestConvertHydros.test_no_ghmin_uses_fallback` -- verify fallback still works

### Integration Tests

N/A

### E2E Tests (if applicable)

N/A

## Dependencies

- **Blocked By**: ticket-001-implement-average-productivity.md (uses the corrected productivity for the fallback calculation)
- **Blocks**: None

## Effort Estimate

**Points**: 2
**Confidence**: Medium (inewave Ghmin API needs verification)
