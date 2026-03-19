# ticket-008: Generate hydro_geometry.parquet from Polynomials

## Context

### Background

Cobre requires `system/hydro_geometry.parquet` for FPHA and evaporation calculations. The schema is `(hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE, area_km2: DOUBLE)` — a VHA curve per plant.

The polynomials come from hidr.dat (after MODIF.DAT permanent overrides are applied):

- `a0_volume_cota` through `a4_volume_cota` — maps volume (hm3) to height (m)
- `a0_cota_area` through `a4_cota_area` — maps height (m) to area (km2)

NOTE: The current `_compute_productivity` function uses column names `volume_cota_0` through `volume_cota_4` in the test fixtures. The REAL inewave column names are `a0_volume_cota` etc. For this ticket, use the REAL inewave column names since this function will be used with real data. The test fixtures for `_compute_productivity` will continue using their mock column names independently.

### Current State

No hydro_geometry.parquet is generated. The pipeline writes to `system/` and `scenarios/` directories but has no geometry output.

## Specification

### Requirements

1. Add `generate_hydro_geometry(cadastro: pd.DataFrame, id_map: NewaveIdMap) -> pa.Table` in `hydro.py` that:
   - For each hydro plant in id_map, creates 100 uniformly spaced volume points in [volume_minimo, volume_maximo]
   - Evaluates the volume_cota polynomial at each point to get height
   - Evaluates the cota_area polynomial at each height to get area
   - Returns a PyArrow Table with schema (hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE, area_km2: DOUBLE)
2. Skip plants where volume_minimo == volume_maximo (run-of-river with no reservoir)
3. Wire into `pipeline.py` — call after `convert_hydros`, write to `system/hydro_geometry.parquet`

### Error Handling

- All-zero polynomial: log warning, skip plant
- Negative height/area from polynomial evaluation: clamp to 0.0

## Acceptance Criteria

- [ ] Given a plant with volume_minimo=100, volume_maximo=1000, and a linear volume_cota polynomial, when `generate_hydro_geometry` is called, then the output table has 100 rows for that plant with uniformly spaced volumes
- [ ] Given a plant with volume_minimo == volume_maximo, when `generate_hydro_geometry` is called, then no rows are produced for that plant
- [ ] Given the output table, when written as Parquet and read back, then columns match the expected schema (hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE, area_km2: DOUBLE)
- [ ] Given `pipeline.py` runs, then `system/hydro_geometry.parquet` is created in the output directory

## Implementation Guide

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` — add `generate_hydro_geometry`
- `src/cobre_bridge/pipeline.py` — wire geometry output
- `tests/test_entity_conversion.py` — add tests

### Suggested Approach

1. Reuse the `_poly` helper from `_compute_productivity` for polynomial evaluation (or create a shared one)
2. Use `numpy.linspace(vol_min, vol_max, 100)` for uniform grid
3. Evaluate volume_cota polynomial for height, then cota_area polynomial for area
4. Build PyArrow arrays and construct the table
5. In pipeline.py, call `hydro_conv.generate_hydro_geometry(cadastro, id_map)` and write with `pq.write_table`

### Column Names

Real inewave hidr.dat columns:

- `a0_volume_cota`, `a1_volume_cota`, `a2_volume_cota`, `a3_volume_cota`, `a4_volume_cota`
- `a0_cota_area`, `a1_cota_area`, `a2_cota_area`, `a3_cota_area`, `a4_cota_area`

## Testing Requirements

- `TestGenerateHydroGeometry.test_produces_100_rows_per_plant` — basic output shape
- `TestGenerateHydroGeometry.test_skips_run_of_river` — vol_min == vol_max
- `TestGenerateHydroGeometry.test_correct_schema` — verify Parquet column types
- Integration: verify pipeline writes the file

## Dependencies

- **Blocked By**: ticket-004 (MODIF permanent overrides affect polynomial coefficients)
- **Blocks**: None

## Effort Estimate

**Points**: 2
**Confidence**: High
