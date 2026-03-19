# ticket-007: Parse PENALID.DAT for Per-Plant Penalty Overrides

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Parse PENALID.DAT using inewave and populate the `penalties` field on hydro entries in hydros.json. The PENALID.DAT file contains per-plant penalty overrides for DESVIO (spillage), VAZMIN (minimum outflow violation), VAZMAX (maximum outflow violation), GHMIN (minimum generation violation), TURBMN (minimum turbined violation), and TURBMX (maximum turbined violation). These map to the `RawHydroPenaltyOverrides` schema fields: `spillage_cost`, `outflow_violation_below_cost`, `outflow_violation_above_cost`, `generation_violation_below_cost`, `turbined_violation_below_cost`.

## Anticipated Scope

- **Files likely to be modified**: `src/cobre_bridge/converters/hydro.py`, `tests/test_entity_conversion.py`
- **Key decisions needed**: Mapping from PENALID.DAT field names (DESVIO, VAZMIN, etc.) to Cobre penalty field names; how to handle plants with partial penalty entries (only some fields overridden)
- **Open questions**: What is the exact inewave API for reading PENALID.DAT? Does it return a DataFrame or structured objects? What are the PENALID.DAT column names in inewave?

## Dependencies

- **Blocked By**: ticket-001-implement-average-productivity.md (base hydro converter changes)
- **Blocks**: None

## Effort Estimate

**Points**: 3
**Confidence**: Low (will be re-estimated during refinement)
