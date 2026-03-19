# NEWAVE Hydro Conversion Improvements

## Overview

Seven improvements to the cobre-bridge NEWAVE hydro converter covering correctness fixes, additional file parsing, geometry generation, and plant filtering.

## Tech Stack

Python 3.12+, inewave, pyarrow, pandas

## Epics

| Epic    | Name                           | Tickets | Detail Level |
| ------- | ------------------------------ | ------- | ------------ |
| epic-01 | Productivity & Derating        | 3       | Detailed     |
| epic-02 | MODIF.DAT & GHMIN.DAT          | 3       | Detailed     |
| epic-03 | Penalties, Geometry, Filtering | 3       | Outline      |

## Progress

| Ticket     | Title                                       | Epic    | Status    | Detail Level | Readiness | Quality | Badge     |
| ---------- | ------------------------------------------- | ------- | --------- | ------------ | --------- | ------- | --------- |
| ticket-001 | Implement average productivity calculation  | epic-01 | completed | Detailed     | 0.94      | 1.00    | EXCELLENT |
| ticket-002 | Apply TEIF/IP availability derating         | epic-01 | completed | Detailed     | 0.97      | 1.00    | EXCELLENT |
| ticket-003 | Add hydraulic_losses field to hydro entries | epic-01 | completed | Detailed     | 0.97      | 1.00    | EXCELLENT |
| ticket-004 | Apply MODIF.DAT permanent overrides         | epic-02 | completed | Detailed     | 0.97      | 0.97    | EXCELLENT |
| ticket-005 | Parse MODIF.DAT temporal overrides          | epic-02 | completed | Detailed     | 0.97      | 1.00    | EXCELLENT |
| ticket-006 | Parse GHMIN.DAT for minimum generation      | epic-02 | completed | Detailed     | 0.97      | 0.93    | EXCELLENT |
| ticket-007 | Parse PENALID.DAT for penalties             | epic-03 | pending   | Outline      | --        | --      | --        |
| ticket-008 | Generate hydro_geometry.parquet             | epic-03 | pending   | Outline      | --        | --      | --        |
| ticket-009 | Filter fictitious plants                    | epic-03 | pending   | Outline      | --        | --      | --        |

## Dependency Graph

```
ticket-001 (productivity)
  ├── ticket-002 (TEIF/IP derating)
  ├── ticket-003 (hydraulic_losses)
  ├── ticket-004 (MODIF permanent) ──> ticket-005 (MODIF temporal)
  ├── ticket-006 (GHMIN)
  ├── ticket-007 (PENALID) [outline]
  ├── ticket-008 (geometry) [outline, also depends on ticket-004]
  └── ticket-009 (FICT filtering) [outline]
```

## Pending Decisions

**ticket-009**: Energy redistribution from fictitious plants is NOT handled. The user's to-do.md notes uncertainty about how to handle their hydro resources. This ticket only removes the plants; redistribution is a separate future decision.
