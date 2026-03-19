# ticket-009: Filter Fictitious Plants from Conversion

## Context

### Background

NEWAVE has ~15 "fictitious" plants (names starting with "FICT.") that are modeling artifacts. These must be excluded from the Cobre output. The to-do.md states energy redistribution is NOT handled — this ticket only removes the plants.

### Current State

All plants with `usina_existente == "EX"` in confhd.dat are included, regardless of name. Fictitious plants pass through to hydros.json, initial_conditions.json, and inflow stats.

## Specification

### Requirements

1. In `pipeline.py::_build_id_map`, filter confhd_df to exclude plants whose `nome_usina` starts with "FICT." BEFORE building hydro_codes for NewaveIdMap
2. This ensures fictitious plants are excluded from `NewaveIdMap` and therefore from ALL downstream outputs
3. Handle downstream_id references: if a non-fictitious plant has a fictitious plant as its downstream, set downstream_id to None (the cascade link is broken)
4. Log a warning listing which fictitious plants were excluded

### Error Handling

- No fictitious plants found: proceed normally
- Downstream reference to fictitious plant: set to None, log warning

## Acceptance Criteria

- [ ] Given confhd.dat contains a plant named "FICT.SERRA M", when `_build_id_map` runs, then the plant's code is NOT in `id_map.all_hydro_codes`
- [ ] Given a non-fictitious plant has `codigo_usina_jusante` pointing to a fictitious plant, when `convert_hydros` runs, then `downstream_id` is None for that plant
- [ ] Given confhd.dat has 15 fictitious plants among 160 existing plants, when conversion runs, then `hydros.json` contains exactly 145 entries
- [ ] Given fictitious plants are excluded from id_map, then they are also excluded from initial_conditions.json and inflow_seasonal_stats.parquet

## Implementation Guide

### Key Files to Modify

- `src/cobre_bridge/pipeline.py` — filter confhd in `_build_id_map`
- `src/cobre_bridge/converters/hydro.py` — handle broken downstream refs gracefully (already does via KeyError catch)
- `tests/test_entity_conversion.py` — add test with FICT plant in fixture

### Suggested Approach

1. In `_build_id_map`, after reading confhd, filter:
   ```python
   existing = confhd_df[confhd_df["usina_existente"] == "EX"]
   non_fict = existing[~existing["nome_usina"].str.strip().str.startswith("FICT.")]
   hydro_codes = [int(r["codigo_usina"]) for _, r in non_fict.iterrows()]
   ```
2. Log excluded plant names
3. The existing KeyError handling in `hydro.py` line 124 already handles missing downstream IDs gracefully

## Testing Requirements

- `TestBuildIdMap.test_excludes_fictitious_plants` — verify FICT plants absent from id_map
- `TestConvertHydros.test_downstream_to_fict_is_none` — broken cascade link

## Dependencies

- **Blocked By**: ticket-001 (base converter)
- **Blocks**: None

## Effort Estimate

**Points**: 1
**Confidence**: High
