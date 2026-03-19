# Epic 03: Penalties, Geometry, and Fictitious Plant Filtering

## Goal

Parse PENALID.DAT for per-plant penalty overrides, generate hydro_geometry.parquet from volume_cota/cota_area polynomials, and filter fictitious (FICT.\*) plants from the conversion output.

## Scope

- Parse PENALID.DAT and populate the `penalties` field on hydro entries
- Generate hydro_geometry.parquet with 100-point VHA curves per plant
- Filter plants whose name starts with "FICT." from the conversion pipeline
- Wire hydro_geometry.parquet output into pipeline.py

## Tickets

| ID         | Title                                             | Points |
| ---------- | ------------------------------------------------- | ------ |
| ticket-007 | Parse PENALID.DAT for per-plant penalty overrides | 3      |
| ticket-008 | Generate hydro_geometry.parquet from polynomials  | 2      |
| ticket-009 | Filter fictitious plants from conversion          | 2      |

## Dependencies

- Epic 01 (productivity fix must be in place before geometry generation uses the same polynomials)
- Epic 02 (MODIF.DAT VOLCOTA/COTARE overrides affect geometry polynomials)

## Acceptance Criteria

- Per-plant penalty overrides appear in hydro entries for plants listed in PENALID.DAT
- hydro_geometry.parquet is generated with correct schema and 100 rows per plant
- Fictitious plants are excluded from all output files
