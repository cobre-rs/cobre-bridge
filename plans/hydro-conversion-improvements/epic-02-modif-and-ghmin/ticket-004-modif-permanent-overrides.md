# ticket-004: Apply MODIF.DAT Permanent Overrides to Hydro Cadastro

## Context

### Background

NEWAVE's MODIF.DAT file contains per-plant overrides to the hidr.dat base data. Some overrides are "permanent" -- they apply once and change the effective plant parameters for the entire study. The current converter ignores MODIF.DAT entirely, meaning any case that relies on MODIF.DAT for corrected bounds, machine counts, or polynomial coefficients produces incorrect output.

Permanent override types: VAZMIN (minimum outflow), VOLMAX (maximum storage volume), NUMCNJ (number of machine sets), NUMMAQ (number of machines per set), VOLCOTA (volume-to-height polynomial coefficients), COTARE (height-to-area polynomial coefficients).

### Relation to Epic

First ticket in Epic 02. Must be implemented before temporal overrides (ticket-005) since they share the same inewave parsing infrastructure. Permanent overrides must be applied before the main conversion loop so that ticket-001's productivity calculation uses corrected polynomial coefficients.

### Current State

`convert_hydros` in `src/cobre_bridge/converters/hydro.py` reads hidr.dat and confhd.dat only. MODIF.DAT is not read anywhere. The inewave library provides a way to read MODIF.DAT -- the exact API needs to be confirmed (likely `Modif.read()` or via `Arquivos` indirection, with a `modificacoes` or similar property returning a DataFrame).

## Specification

### Requirements

1. Add a helper function `_apply_permanent_overrides(cadastro: pd.DataFrame, newave_dir: Path) -> pd.DataFrame` that:
   - Reads MODIF.DAT using inewave (determine the correct API by inspecting `from inewave.newave import Modif` or similar)
   - Iterates over modification records for each plant
   - For permanent override types (VAZMIN, VOLMAX, NUMCNJ, NUMMAQ, VOLCOTA, COTARE), updates the corresponding columns in the cadastro DataFrame
   - Returns the modified cadastro
2. Call this function in `convert_hydros` after reading hidr.dat and before the main conversion loop
3. If MODIF.DAT does not exist, skip (no overrides to apply) -- log a debug message
4. MODIF.DAT should be read from the path specified in ARQUIVOS.DAT or directly from `newave_dir / "modif.dat"` (case-insensitive file lookup)

### Inputs/Props

- `cadastro`: the hidr.dat cadastro DataFrame (indexed by codigo_usina)
- `newave_dir`: Path to the NEWAVE case directory

### Outputs/Behavior

Returns a new DataFrame with permanent overrides applied. The original cadastro is not mutated.

### Error Handling

- If MODIF.DAT is missing, return the cadastro unchanged (with a debug log)
- If a plant code in MODIF.DAT is not in the cadastro, log a warning and skip that record
- If an unknown override type is encountered, log a warning and skip

## Acceptance Criteria

- [ ] Given a MODIF.DAT that sets VOLMAX=2000 for plant code 1, when `_apply_permanent_overrides` is called with a cadastro where plant 1 has `volume_maximo=1000`, then the returned DataFrame has `volume_maximo=2000` for plant 1
- [ ] Given a MODIF.DAT that sets NUMCNJ=2 and NUMMAQ for set 2 to 3 for plant code 1, when the override is applied, then the cadastro reflects `numero_conjuntos_maquinas=2` and `maquinas_conjunto_2=3`
- [ ] Given a MODIF.DAT that sets VOLCOTA coefficients for plant code 1, when the override is applied, then the cadastro columns `volume_cota_0` through `volume_cota_4` are updated for that plant
- [ ] Given MODIF.DAT does not exist in the newave directory, when `_apply_permanent_overrides` is called, then the cadastro is returned unchanged and no error is raised
- [ ] Given a MODIF.DAT record references plant code 999 not in cadastro, when the override is applied, then a warning is logged and the record is skipped

## Implementation Guide

### Suggested Approach

1. Investigate the inewave Modif API. Try:
   ```python
   from inewave.newave import Modif
   modif = Modif.read(str(newave_dir / "modif.dat"))
   # Inspect modif properties -- likely modif.modificacoes or similar
   ```
   The `modificacoes_usina()` function mentioned in the spec may be a method that filters by plant code.
2. Create a mapping from MODIF type names to cadastro column names:
   ```python
   _PERMANENT_OVERRIDES = {
       "VAZMIN": "vazao_minima_historica",
       "VOLMAX": "volume_maximo",
       "NUMCNJ": "numero_conjuntos_maquinas",
       # NUMMAQ and VOLCOTA/COTARE need special handling (indexed fields)
   }
   ```
3. For NUMMAQ: the override specifies which machine set and how many machines. Map to `maquinas_conjunto_{set_num}`.
4. For VOLCOTA: the override provides 5 coefficients. Map to `volume_cota_0` through `volume_cota_4`.
5. For COTARE: the override provides 5 coefficients. Map to `cota_area_0` through `cota_area_4`.
6. Apply overrides to a copy of the cadastro DataFrame.
7. Wire the call in `convert_hydros` between `cadastro = hidr.cadastro` (line 71) and the main loop (line 85).

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` -- add `_apply_permanent_overrides`, call it after line 71
- `tests/test_entity_conversion.py` -- add tests with mocked Modif data

### Patterns to Follow

- Follow the existing file-existence check pattern (lines 63-65 in hydro.py)
- Use `cadastro.copy()` to avoid mutating the original
- Mock the inewave Modif class in tests, same pattern as Hidr/Confhd mocking

### Pitfalls to Avoid

- MODIF.DAT records are ordered and must be processed sequentially -- later records override earlier ones for the same plant and field
- NUMMAQ overrides have a set index parameter -- do not confuse with NUMCNJ
- The inewave API may return records as a list of objects or a DataFrame -- inspect the actual return type before coding
- Do NOT apply temporal overrides (VAZMINT, VMAXT, etc.) in this ticket -- those are handled in ticket-005

## Testing Requirements

### Unit Tests

- `TestApplyPermanentOverrides.test_volmax_override` -- verify volume_maximo is updated
- `TestApplyPermanentOverrides.test_vazmin_override` -- verify vazao_minima_historica is updated
- `TestApplyPermanentOverrides.test_numcnj_nummaq_override` -- verify machine count changes
- `TestApplyPermanentOverrides.test_volcota_override` -- verify polynomial coefficient update
- `TestApplyPermanentOverrides.test_missing_modif_returns_unchanged` -- no MODIF.DAT
- `TestApplyPermanentOverrides.test_unknown_plant_code_skipped` -- plant not in cadastro

### Integration Tests

N/A

### E2E Tests (if applicable)

N/A

## Dependencies

- **Blocked By**: ticket-001-implement-average-productivity.md (productivity calc must use volume_cota columns that this ticket overrides)
- **Blocks**: ticket-005-modif-temporal-overrides.md

## Effort Estimate

**Points**: 3
**Confidence**: Medium (inewave Modif API needs verification)
