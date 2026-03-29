# Plan: Refactor `compare` command — bounds from inputs + results comparison

## Overview

Refactor the `cobre-bridge compare` CLI to compute bounds from NEWAVE input files (no sintetizador) and add results comparison against Cobre simulation output with HTML reports.

## Progress

| Ticket | Title                         | Epic   | Status    | Readiness | Effort |
| ------ | ----------------------------- | ------ | --------- | --------- | ------ |
| T-001  | Refactor entity alignment     | Epic 1 | completed | 0.92      | 20min  |
| T-002  | Extract bounds computation    | Epic 1 | completed | 0.88      | 30min  |
| T-003  | Rewrite bounds comparison     | Epic 1 | completed | 0.92      | 20min  |
| T-004  | Update CLI + tests            | Epic 1 | completed | 0.94      | 15min  |
| T-005  | NEWAVE output readers         | Epic 2 | completed | 0.90      | 25min  |
| T-006  | Cobre simulation readers      | Epic 2 | completed | 0.90      | 20min  |
| T-007  | Results comparison engine     | Epic 2 | completed | 0.88      | 25min  |
| T-008  | CLI integration + text report | Epic 2 | completed | 0.90      | 20min  |
| T-009  | HTML template + CSS           | Epic 3 | completed | 0.88      | 20min  |
| T-010  | Chart implementations         | Epic 3 | completed | 0.85      | 30min  |
| T-011  | Report assembly               | Epic 3 | completed | 0.85      | 15min  |
| T-012  | Integration tests             | Epic 4 | completed | 0.85      | 25min  |
| T-013  | Edge cases                    | Epic 4 | completed | 0.85      | 20min  |

**Total estimated effort**: ~285min (~4.75 hours)

## Dependency Graph

```
T-001 (alignment) ─┬── T-002 (bounds extraction) ── T-003 (comparison) ── T-004 (CLI)
                    ├── T-005 (NEWAVE readers) ─┬── T-007 (results engine) ─┬── T-008 (results CLI)
                    └── T-006 (Cobre readers) ──┘                           │
                                                                            ├── T-009 (HTML template)
                                                                            │     └── T-010 (charts)
                                                                            │           └── T-011 (assembly)
                                                                            ├── T-012 (integration tests)
                                                                            └── T-013 (edge cases)
```

## Execution Order

1. **T-001** (no dependencies)
2. **T-002** + **T-005** + **T-006** (parallel, all depend only on T-001)
3. **T-003** (depends on T-002)
4. **T-004** + **T-007** (parallel: T-004 depends on T-003, T-007 depends on T-005+T-006)
5. **T-008** (depends on T-004 + T-007)
6. **T-009** (depends on T-008)
7. **T-010** (depends on T-009)
8. **T-011** (depends on T-010)
9. **T-012** + **T-013** (parallel, depend on T-008+T-011)

## Plan Structure

```
plans/compare-refactor/
  00-master-plan.md
  README.md
  .implementation-state.json
  epic-01-bounds-refactor/
    00-epic-overview.md
    ticket-001-refactor-entity-alignment.md
    ticket-002-extract-bounds-computation.md
    ticket-003-rewrite-bounds-comparison.md
    ticket-004-update-cli-and-tests.md
  epic-02-results-comparison/
    00-epic-overview.md
    ticket-005-newave-output-readers.md
    ticket-006-cobre-simulation-readers.md
    ticket-007-results-comparison-engine.md
    ticket-008-cli-integration-text-report.md
  epic-03-html-report/
    00-epic-overview.md
    ticket-009-html-template-css.md
    ticket-010-chart-implementations.md
    ticket-011-report-assembly.md
  epic-04-testing-polish/
    00-epic-overview.md
    ticket-012-integration-tests.md
    ticket-013-edge-cases.md
```
