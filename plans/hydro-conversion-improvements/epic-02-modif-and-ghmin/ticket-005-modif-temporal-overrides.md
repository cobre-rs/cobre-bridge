# ticket-005: Parse MODIF.DAT Temporal Overrides

## Context

### Background

Beyond permanent overrides, MODIF.DAT also contains temporal overrides that change hydro plant parameters at specific month/year dates. These hold from that date onwards until another override for the same plant and parameter is declared. Temporal override types: VAZMINT (minimum outflow), VMAXT (maximum storage), CFUGA (downstream level), CMONT (upstream level), TURBMINT (minimum turbined flow), TURBMAXT (maximum turbined flow).

For the cobre-bridge converter, temporal overrides that fall within the study horizon need to be mapped to stage-indexed bounds. CFUGA and CMONT for run-of-river plants (tipo_regulacao != "M") directly set the drop for productivity evaluation at those stages.

### Relation to Epic

Second ticket in Epic 02. Depends on ticket-004 which establishes the MODIF.DAT parsing infrastructure. This ticket adds temporal override extraction.

### Current State

MODIF.DAT temporal overrides are completely ignored. The converter uses static bounds from hidr.dat for the entire study horizon. Temporal bounds variations are not represented in the output.

## Specification

### Requirements

1. Add a function `_extract_temporal_overrides(newave_dir: Path, confhd_codes: list[int]) -> dict[int, list[dict]]` that:
   - Reads MODIF.DAT (reusing the parsing from ticket-004)
   - Extracts temporal override records (VAZMINT, VMAXT, CFUGA, CMONT, TURBMINT, TURBMAXT)
   - Returns a dict keyed by plant code, each value being a list of `{"type": str, "month": int, "year": int, "value": float}` records in file order
2. Integrate into `convert_hydros` so that temporal overrides for VAZMINT and TURBMAXT/TURBMINT are used to tighten/relax bounds on the hydro entry
3. For CFUGA/CMONT temporal overrides on run-of-river plants: these affect productivity at specific stages. For now, store them in the return structure but do NOT attempt per-stage productivity -- just log a warning that temporal drop overrides are present but not yet mapped to stages.

### Inputs/Props

- `newave_dir`: Path to the NEWAVE case directory
- `confhd_codes`: list of plant codes in the study (to filter relevant records)

### Outputs/Behavior

Returns a dict of temporal override records per plant. The calling code can use these to:

- Set `outflow.min_outflow_m3s` from the first VAZMINT override (if it precedes or equals study start)
- Set `generation.min_turbined_m3s` / `generation.max_turbined_m3s` similarly

### Error Handling

- Missing MODIF.DAT: return empty dict
- Unknown temporal override types: log warning, skip

## Acceptance Criteria

- [ ] Given a MODIF.DAT with VAZMINT=50.0 for plant 1 at month 1/year 2025, when `_extract_temporal_overrides` is called, then the returned dict contains `{1: [{"type": "VAZMINT", "month": 1, "year": 2025, "value": 50.0}]}`
- [ ] Given temporal overrides for a plant code not in confhd_codes, when extraction runs, then those records are excluded from the result
- [ ] Given CFUGA temporal overrides exist for a run-of-river plant, when `convert_hydros` processes it, then a warning is logged indicating temporal drop overrides are not yet mapped to per-stage productivity
- [ ] Given MODIF.DAT is missing, when `_extract_temporal_overrides` is called, then an empty dict is returned without error

## Implementation Guide

### Suggested Approach

1. Reuse the inewave Modif reader established in ticket-004
2. Filter records by type -- temporal types are VAZMINT, VMAXT, CFUGA, CMONT, TURBMINT, TURBMAXT (they have month/year fields)
3. Build the per-plant override list, preserving file order (important for override semantics)
4. In `convert_hydros`, for each plant, check if there are temporal overrides that affect initial bounds:
   - Find the earliest VAZMINT override at or before study start -> set as `min_outflow_m3s`
   - Find the earliest TURBMINT/TURBMAXT at or before study start -> set as `min_turbined_m3s`/`max_turbined_m3s`
5. Store the full temporal override structure in a module-level or return dict for future use by stage-level conversion

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` -- add `_extract_temporal_overrides`, integrate into `convert_hydros`
- `tests/test_entity_conversion.py` -- add tests with mocked temporal override data

### Patterns to Follow

- Same mocking pattern as ticket-004 tests
- Return structured dicts, not raw DataFrames, for clarity

### Pitfalls to Avoid

- Temporal overrides "hold" from their date onwards -- the latest override before study start is the effective one, not the first
- Do NOT attempt to modify per-stage JSON output in this ticket -- that is a separate future concern
- CFUGA and CMONT are meter values, not flow values -- do not confuse units

## Testing Requirements

### Unit Tests

- `TestExtractTemporalOverrides.test_extracts_vazmint_records` -- verify correct extraction
- `TestExtractTemporalOverrides.test_filters_by_confhd_codes` -- only study plants included
- `TestExtractTemporalOverrides.test_missing_modif_returns_empty` -- graceful handling
- `TestExtractTemporalOverrides.test_preserves_file_order` -- order matters

### Integration Tests

N/A

### E2E Tests (if applicable)

N/A

## Dependencies

- **Blocked By**: ticket-004-modif-permanent-overrides.md
- **Blocks**: None

## Effort Estimate

**Points**: 3
**Confidence**: Medium (temporal override semantics need careful handling)
