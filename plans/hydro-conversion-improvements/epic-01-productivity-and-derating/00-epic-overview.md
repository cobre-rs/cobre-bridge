# Epic 01: Productivity Fix & TEIF/IP Derating

## Goal

Replace the incorrect direct use of `produtibilidade_especifica` with proper average productivity calculation, and apply TEIF/IP availability derating to generation bounds.

## Scope

- Implement `_compute_productivity()` helper that evaluates volume_cota polynomial, applies canal_fuga_medio, tipo_perda loss model, and integral mean value theorem
- Replace line 102 in `hydro.py` with the new calculation
- Apply TEIF/IP derating factor to `max_generation_mw`
- Also populate the `hydraulic_losses` field in the hydro entry
- Update existing tests to match new productivity formula
- Add new unit tests for the productivity helper

## Tickets

| ID         | Title                                                    | Points |
| ---------- | -------------------------------------------------------- | ------ |
| ticket-001 | Implement average productivity calculation               | 3      |
| ticket-002 | Apply TEIF/IP availability derating to generation bounds | 2      |
| ticket-003 | Add hydraulic_losses field to hydro entries              | 1      |

## Dependencies

- None (this is the first epic)

## Acceptance Criteria

- Productivity values differ from raw `produtibilidade_especifica` for plants with non-trivial volume_cota polynomials
- TEIF/IP derating reduces max_generation_mw below the sum of nominal powers
- All existing tests updated and passing
