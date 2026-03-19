# Master Plan: NEWAVE Hydro Conversion Improvements

## Executive Summary

Fix and extend the hydro converter in cobre-bridge to produce correct productivity values, apply availability derating, parse additional NEWAVE input files (MODIF.DAT, GHMIN.DAT, PENALID.DAT), generate hydro_geometry.parquet, and filter fictitious plants. Seven work items organized into three epics by dependency and risk.

## Goals & Non-Goals

**Goals:**

- Correct productivity calculation using volume_cota polynomials, canal_fuga_medio, tipo_perda loss model, and integral mean value theorem
- Apply TEIF/IP derating to max_generation_mw
- Parse MODIF.DAT for permanent and temporal hydro data overrides
- Parse GHMIN.DAT for explicit minimum generation constraints
- Parse PENALID.DAT for per-plant penalty overrides
- Generate hydro_geometry.parquet from volume_cota and cota_area polynomials
- Filter fictitious plants (FICT.\* names) from output

**Non-Goals:**

- Tailrace model (deferred to FPHA stage)
- FPHA generation model (separate future work)
- Temporal overrides in stages.json (temporal bounds go into hydro conversion only for now)

## Architecture Overview

### Current State

`hydro.py` reads `hidr.dat`, `confhd.dat`, `ree.dat`. It directly uses `produtibilidade_especifica` as productivity (wrong). No parsing of MODIF.DAT, GHMIN.DAT, or PENALID.DAT. No hydro_geometry.parquet generation. No TEIF/IP derating. No fictitious plant filtering.

### Target State

`hydro.py` computes correct average productivity per plant. Reads MODIF.DAT for permanent overrides before computing bounds, reads GHMIN.DAT for min generation, reads PENALID.DAT for per-plant penalties. Generates hydro_geometry.parquet. Applies TEIF/IP derating. Filters FICT.\* plants. `pipeline.py` orchestrates the new outputs.

### Key Design Decisions

1. **Productivity calculation lives in a helper function** inside `hydro.py` -- no separate module needed for this.
2. **MODIF.DAT permanent overrides** are applied to the hidr cadastro DataFrame before the main conversion loop. Temporal overrides produce a separate structure consumed downstream.
3. **hydro_geometry.parquet** is generated as a separate function in `hydro.py` called from `pipeline.py`.
4. **Fictitious plant filtering** happens at the confhd level before ID mapping, which means fictitious plants are excluded from `NewaveIdMap` entirely.

## Technical Approach

### Tech Stack

- Python 3.12+, inewave, pyarrow, pandas, numpy (for polynomial evaluation)

### Component/Module Breakdown

- `src/cobre_bridge/converters/hydro.py` -- main changes (productivity, derating, geometry, filtering, penalties, MODIF permanent overrides)
- `src/cobre_bridge/pipeline.py` -- wire new outputs (hydro_geometry.parquet, MODIF/GHMIN/PENALID file requirements)
- `src/cobre_bridge/id_map.py` -- may need update if FICT filtering changes hydro_codes list
- `tests/test_entity_conversion.py` -- extend with new test cases

### Testing Strategy

- Unit tests with mocked inewave I/O (existing pattern)
- Integration test against example/newave/ case for smoke testing

## Phases & Milestones

| Epic | Name                           | Scope                                                         | Tickets |
| ---- | ------------------------------ | ------------------------------------------------------------- | ------- |
| 1    | Productivity & Derating        | Fix productivity calculation + TEIF/IP derating               | 3       |
| 2    | MODIF.DAT & GHMIN.DAT          | Parse and apply permanent/temporal overrides + min generation | 3       |
| 3    | Penalties, Geometry, Filtering | PENALID.DAT, hydro_geometry.parquet, FICT filtering           | 3       |

## Risk Analysis

- **inewave API uncertainty**: The `modificacoes_usina()` function behavior needs verification against the example case. Mitigated by reading inewave source or testing interactively.
- **Polynomial coefficient ordering**: volume_cota and cota_area polynomial coefficient order in hidr.dat must match numpy's convention (or be reversed). Verify against example data.
- **Fictitious plant energy redistribution**: Open design question -- filtering FICT plants may lose energy. Deferred to user decision.

## Success Metrics

- All existing tests pass unchanged (or updated to match new productivity formula)
- Converted example case produces valid Cobre input (validated by cobre-python)
- Productivity values match manual calculations for at least 2 sample plants
