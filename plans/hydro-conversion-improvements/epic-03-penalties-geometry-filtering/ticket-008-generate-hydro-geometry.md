# ticket-008: Generate hydro_geometry.parquet from Polynomials

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Generate a `hydro_geometry.parquet` file by sampling the volume_cota and cota_area polynomials from hidr.dat (after MODIF.DAT VOLCOTA/COTARE overrides) over a 100-point uniform grid [volume_minimo, volume_maximo] for each hydro plant. The output schema is `(hydro_id: INT32, volume_hm3: DOUBLE, height_m: DOUBLE, area_km2: DOUBLE)` with multiple rows per hydro forming VHA (Volume-Height-Area) curves.

## Anticipated Scope

- **Files likely to be modified**: `src/cobre_bridge/converters/hydro.py` (new `generate_hydro_geometry` function), `src/cobre_bridge/pipeline.py` (wire output), `tests/test_entity_conversion.py`
- **Key decisions needed**: Whether to use the post-MODIF.DAT cadastro (requiring this ticket to depend on ticket-004) or the raw cadastro; exact polynomial coefficient column names for cota_area
- **Open questions**: Are the cota_area polynomial columns named `cota_area_0` through `cota_area_4` in inewave? Should the geometry be generated only for plants with non-trivial reservoirs (volume_min != volume_max)?

## Dependencies

- **Blocked By**: ticket-004-modif-permanent-overrides.md (VOLCOTA/COTARE overrides affect polynomial coefficients)
- **Blocks**: None

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
