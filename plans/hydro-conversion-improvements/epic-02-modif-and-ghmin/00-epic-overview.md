# Epic 02: MODIF.DAT Overrides & GHMIN.DAT Minimum Generation

## Goal

Parse MODIF.DAT for permanent and temporal overrides to hydro plant data, and parse GHMIN.DAT for explicit per-plant minimum generation constraints. These replace rough approximations with actual NEWAVE input data.

## Scope

- Parse MODIF.DAT using inewave's `modificacoes_usina()` for permanent overrides (VAZMIN, VOLMAX, NUMCNJ, NUMMAQ, VOLCOTA, COTARE)
- Apply permanent overrides to the cadastro before the main conversion loop
- Parse temporal overrides (VAZMINT, VMAXT, CFUGA, CMONT, TURBMINT, TURBMAXT) and store them for later use
- Parse GHMIN.DAT for per-plant minimum generation values
- Replace the current min_generation approximation with explicit GHMIN values where available

## Tickets

| ID         | Title                                                 | Points |
| ---------- | ----------------------------------------------------- | ------ |
| ticket-004 | Apply MODIF.DAT permanent overrides to hydro cadastro | 3      |
| ticket-005 | Parse MODIF.DAT temporal overrides                    | 3      |
| ticket-006 | Parse GHMIN.DAT for minimum generation constraints    | 2      |

## Dependencies

- Epic 01 (ticket-001 changes the productivity calculation which is affected by VOLCOTA/COTARE overrides from MODIF.DAT)

## Acceptance Criteria

- MODIF.DAT permanent overrides change the effective cadastro values before productivity is calculated
- Temporal overrides are stored in a structured format for downstream consumption
- GHMIN.DAT values replace the rough min_generation approximation
