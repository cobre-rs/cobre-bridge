# ticket-003: Add hydraulic_losses Field to Hydro Entries

## Context

### Background

The Cobre hydro schema supports a `hydraulic_losses` field with two variants: `{"type": "factor", "value": 0.05}` for multiplicative losses and `{"type": "constant", "value_m": 3.5}` for constant head losses. The current converter always sets `"hydraulic_losses": None`. Since ticket-001 reads `tipo_perda` and `perdas` to compute productivity, we should also populate this field for downstream consumers.

### Relation to Epic

Last ticket in Epic 01. Builds on the same hidr.dat columns used by ticket-001 (tipo_perda, perdas) but writes to a different output field.

### Current State

File `src/cobre_bridge/converters/hydro.py`, line 179: `"hydraulic_losses": None`. The Cobre schema at `/home/rogerio/git/cobre/book/src/schemas/hydros.schema.json` defines `RawHydraulicLosses` as a tagged union with variants `factor` (value: float) and `constant` (value_m: float).

## Specification

### Requirements

1. After reading `tipo_perda` and `perdas` from `hreg` (already done in ticket-001 for productivity), populate the `hydraulic_losses` field:
   - `tipo_perda == 1` and `perdas > 0`: `{"type": "factor", "value": perdas}`
   - `tipo_perda == 2` and `perdas > 0`: `{"type": "constant", "value_m": perdas}`
   - Otherwise: `None`

### Inputs/Props

`tipo_perda` (int) and `perdas` (float) from hidr.dat cadastro.

### Outputs/Behavior

The `hydraulic_losses` key in each hydro entry dict is set to the appropriate dict or None.

### Error Handling

- If `perdas` is 0.0 or NaN, set `hydraulic_losses` to None regardless of tipo_perda

## Acceptance Criteria

- [ ] Given a plant with `tipo_perda=1` and `perdas=0.05`, when `convert_hydros` is called, then the hydro entry has `"hydraulic_losses": {"type": "factor", "value": 0.05}`
- [ ] Given a plant with `tipo_perda=2` and `perdas=3.5`, when `convert_hydros` is called, then the hydro entry has `"hydraulic_losses": {"type": "constant", "value_m": 3.5}`
- [ ] Given a plant with `tipo_perda=0` or `perdas=0.0`, when `convert_hydros` is called, then `"hydraulic_losses"` is `None`
- [ ] Given the output JSON, when validated against `/home/rogerio/git/cobre/book/src/schemas/hydros.schema.json`, then the `hydraulic_losses` field matches the `RawHydraulicLosses` schema

## Implementation Guide

### Suggested Approach

1. In the conversion loop, after computing productivity (which already reads tipo_perda and perdas), build the hydraulic_losses dict:
   ```python
   tipo_perda = int(hreg.get("tipo_perda", 0) or 0)
   perdas_val = float(hreg.get("perdas", 0.0) or 0.0)
   if tipo_perda == 1 and perdas_val > 0:
       hydraulic_losses = {"type": "factor", "value": perdas_val}
   elif tipo_perda == 2 and perdas_val > 0:
       hydraulic_losses = {"type": "constant", "value_m": perdas_val}
   else:
       hydraulic_losses = None
   ```
2. Replace `"hydraulic_losses": None` at line 179 with `"hydraulic_losses": hydraulic_losses`

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` -- add ~8 lines, update one line
- `tests/test_entity_conversion.py` -- add test for hydraulic_losses field

### Patterns to Follow

- Same dict construction pattern used for `evaporation` at lines 172-173

### Pitfalls to Avoid

- The schema field names are `"type"` and `"value"` / `"value_m"` -- do not use `"kind"` or `"loss_value"`
- Do NOT share the `perdas` variable name with the productivity calculation to avoid confusion -- use a distinct local variable name like `perdas_val`

## Testing Requirements

### Unit Tests

- `TestConvertHydros.test_hydraulic_losses_factor` -- tipo_perda=1
- `TestConvertHydros.test_hydraulic_losses_constant` -- tipo_perda=2
- `TestConvertHydros.test_hydraulic_losses_none_when_zero` -- perdas=0

### Integration Tests

N/A

### E2E Tests (if applicable)

N/A

## Dependencies

- **Blocked By**: ticket-001-implement-average-productivity.md (test fixture must include tipo_perda/perdas columns)
- **Blocks**: None

## Effort Estimate

**Points**: 1
**Confidence**: High
