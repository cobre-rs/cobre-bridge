---
paths:
  - "tests/**/*.py"
---

# Cobre-Bridge Testing Rules

The standing contract for how this repo is tested. Generic Python testing
style lives in the global Python rules; these are the bridge-specific rules.

## The 3-tier convention (load-bearing — CI depends on it)

- **Tier 1 — pure Python.** Imports no `cobre`. Runs on every CI job
  (3.12/3.13/3.14). No module at any tier may `import cobre` at module scope —
  the suite must *collect* cleanly in a cobre-free environment.
- **Tier 2 — needs `cobre-python`** (a required runtime dep) but no solver
  binary and no deck. Guarded by `tests.conftest.requires_cobre_python`
  (`find_spec`-based) and, where relevant, `requires_writer_binding`; any
  `import cobre` lives inside the guarded test/helper body.
- **Tier 3 — needs the real solver binary and/or a real gitignored deck under
  `example/`.** Always `skipif`-guarded on the path's existence. Dev-only
  smoke; never runs in CI.

**No tier-1 or tier-2 test reads `example/`.** Any test that does is tier 3
and carries the path guard.

### Deck-guard discipline (the 69-dead-tests lesson)

A `skipif`-guarded test whose deck no longer exists anywhere runs **nowhere
and reports nothing**. Rules:

- Every deck path a test guards on is declared as a module-level `_DECK`-style
  constant, never inline in the decorator.
- When retiring or renaming a deck under `example/`, retarget or delete every
  test that guards on it **in the same change** — a skip is not a pass.
- Prefer guarding on the *surviving* fast fixture
  (`example/decomp-mar-26-rv2-reduced`, the 2-fan case) over per-revision decks.

## Test architecture

- **Mock discipline: patch public entry points only.** Patching an
  underscore-private symbol couples the test to an implementation detail;
  the repo's baseline is ~5 private patches out of ~300 — keep it there.
  Pipeline tests mock converter entry points; CLI error-path tests run as
  subprocess to verify real exit codes.
- **One home per source module.** New converter/comparator tests go in a
  `test_<module>.py` named for the source module — do not grow the mega-files
  (`test_entity_conversion.py`, `test_decomp_results_compare.py`, `test_cli.py`
  are legacy exceptions scheduled for splitting, not a pattern to extend).
- **Shared case builders live in `tests/conftest.py`** (or a shared fixtures
  module) — never re-defined per file.
- **Test output goes to `tmp_path`.** No test writes into `example/*/` or the
  repo tree; generated artifacts self-delete.
- **Golden files** (`tests/golden/`) pin rendered HTML/dataset shapes. A
  deliberate rendering change regenerates them via the documented regeneration
  path — never hand-edit a golden file, and never loosen an assertion to avoid
  regenerating.
- **CliRunner + Rich:** Typer help/rendering tests run under a dumb terminal
  (`TERM=dumb` fixture) — Rich box-drawing output differs across CI terminals
  and versions.
- **Logger state:** an autouse fixture snapshots/restores the `cobre_bridge`
  logger so a CliRunner invocation cannot leak logging state into a later
  `caplog` test — new tests that reconfigure logging must go through it.

## What a new feature must ship with

- A tier-1 test for the pure conversion/analysis logic (synthetic inputs).
- For a new CLI flag or exit-code path: a CliRunner or subprocess test
  asserting the exit code and, when `--json` is involved, the verdict envelope.
- For a new output artifact: an emission-shape test (schema/columns), mirrored
  on **both** tracks when the artifact exists on both (see
  `.claude/rules/bridge.md` — twin-track symmetry).
