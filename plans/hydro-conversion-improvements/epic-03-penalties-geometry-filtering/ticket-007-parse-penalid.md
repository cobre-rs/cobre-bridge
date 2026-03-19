# ticket-007: Parse PENALID.DAT for Per-Plant Penalty Overrides

## Context

### Background

PENALID.DAT contains penalty cost overrides per subsystem/REE for several violation types. The Cobre hydro schema supports per-plant `penalties` overrides via `RawHydroPenaltyOverrides`. Currently all hydro entries have `"penalties": None`, meaning only global defaults apply.

### Relation to Epic

First ticket in Epic 03. Independent of tickets 008 and 009.

### Current State

`hydro.py` sets `"penalties": None` for every hydro entry. PENALID.DAT is not read anywhere.

### inewave API (verified)

```python
from inewave.newave import Penalid
p = Penalid.read(str(path))
df = p.penalidades
# Columns: variavel, codigo_ree_submercado, patamar_penalidade, patamar_carga, valor_R$_MWh, valor_R$_hm3
# Variables: DESVIO, VAZMIN, VAZMAX (if present), GHMIN, TURBMN, TURBMX, ELETRI
# Multiple rows per variable (one per REE/subsystem, multiple patamar levels)
# patamar_penalidade=1 is the first tier cost; patamar_penalidade=2 has NaN cost (unbounded)
```

The PENALID.DAT penalties are **per REE/subsystem**, not per plant. To map them to per-plant penalties, we use the plant's REE assignment from confhd.dat -> ree.dat.

### Cobre Schema Mapping

| PENALID variable | Cobre penalty field             | Unit   |
| ---------------- | ------------------------------- | ------ |
| DESVIO           | spillage_cost                   | R$/MWh |
| VAZMIN           | outflow_violation_below_cost    | R$/MWh |
| VAZMAX           | outflow_violation_above_cost    | R$/MWh |
| GHMIN            | generation_violation_below_cost | R$/MWh |
| TURBMN           | turbined_violation_below_cost   | R$/MWh |
| TURBMX           | (no direct mapping — skip)      | R$/MWh |

## Specification

### Requirements

1. Add `_read_penalid(newave_dir: Path) -> dict[int, dict[str, float]]` that returns a mapping from REE/subsystem code to penalty overrides dict
2. In the conversion loop, look up the plant's REE code, find its penalty overrides, and populate the `penalties` field
3. Only set fields that have actual non-NaN values in PENALID.DAT (use first patamar tier)
4. If PENALID.DAT is missing, leave all penalties as None

### Error Handling

- Missing PENALID.DAT: return empty dict, log debug
- NaN values: skip that penalty field (leave as None in the override dict)

## Acceptance Criteria

- [ ] Given PENALID.DAT has DESVIO=8300.0 for REE 1 at patamar 1, when a plant in REE 1 is converted, then `penalties.spillage_cost` is 8300.0
- [ ] Given PENALID.DAT has VAZMIN=3179.35 for REE 1, when a plant in REE 1 is converted, then `penalties.outflow_violation_below_cost` is 3179.35
- [ ] Given PENALID.DAT is missing, when convert_hydros runs, then all plants have `penalties: None`
- [ ] Given a plant in REE 2 with different penalty values than REE 1, when converted, then the penalties reflect REE 2's values

## Implementation Guide

### Key Files to Modify

- `src/cobre_bridge/converters/hydro.py` — add `_read_penalid`, update conversion loop
- `tests/test_entity_conversion.py` — add tests

### Suggested Approach

1. Read PENALID.DAT using `Penalid.read()`, get `penalidades` DataFrame
2. Filter to `patamar_penalidade == 1` (first tier only — the base cost)
3. Pivot by `(codigo_ree_submercado, variavel)` to get per-REE penalties
4. Map variable names to Cobre field names
5. In the conversion loop, look up penalties by the plant's REE code
6. Build `{"spillage_cost": ..., "outflow_violation_below_cost": ...}` dict, omitting None fields
7. If the dict is empty (no overrides for that REE), set `penalties` to None

## Testing Requirements

- `TestReadPenalid.test_reads_penalties_by_ree` — verify correct mapping
- `TestReadPenalid.test_missing_file_returns_empty` — graceful handling
- `TestConvertHydros.test_penalties_from_penalid` — integration test

## Dependencies

- **Blocked By**: ticket-001 (base converter)
- **Blocks**: None

## Effort Estimate

**Points**: 2
**Confidence**: High
