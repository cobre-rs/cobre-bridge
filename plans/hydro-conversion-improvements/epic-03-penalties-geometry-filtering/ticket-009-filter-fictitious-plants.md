# ticket-009: Filter Fictitious Plants from Conversion

> **[OUTLINE]** This ticket requires refinement before execution.
> It will be refined with learnings from earlier epics.

## Objective

Filter out hydro plants whose name starts with "FICT." from the conversion pipeline. These are modeling artifacts from early NEWAVE limitations and are not actual hydro plants. The filtering must happen at the confhd level before ID mapping, so that fictitious plants are excluded from `NewaveIdMap` and all downstream outputs (hydros.json, initial_conditions.json, inflow stats, past inflows).

## Anticipated Scope

- **Files likely to be modified**: `src/cobre_bridge/pipeline.py` (\_build_id_map), `src/cobre_bridge/converters/hydro.py`, `src/cobre_bridge/converters/initial_conditions.py`, `src/cobre_bridge/converters/stochastic.py`, `tests/test_entity_conversion.py`
- **Key decisions needed**: Whether filtering happens in \_build_id_map (pipeline.py) or in each converter; how to handle downstream_id references to fictitious plants (set to None?)
- **Open questions**: [ASSUMPTION] Energy redistribution from fictitious plants is NOT handled in this ticket. The user's to-do.md notes "I have to check on how we are going to do with their hydro resources." This ticket only removes the plants. Energy redistribution (if needed) is a separate future decision.

## Dependencies

- **Blocked By**: ticket-001-implement-average-productivity.md (base hydro converter changes)
- **Blocks**: None

## Effort Estimate

**Points**: 2
**Confidence**: Low (will be re-estimated during refinement)
