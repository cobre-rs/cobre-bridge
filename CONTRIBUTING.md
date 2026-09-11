# Contributing to cobre-bridge

Thanks for helping. This page gets a checkout running, explains how the tests
and quality gates are organised, and lists the conventions CI enforces. Read
`docs/architecture.md` first if you are about to touch the code.

## Development setup

```bash
git clone https://github.com/cobre-rs/cobre-bridge.git
cd cobre-bridge
uv sync --extra dev             # or: python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
uv run cobre-bridge --version
```

`uv.lock` is committed, so `uv sync` reproduces the CI environment.
`cobre-python` is a core dependency with prebuilt wheels for a fixed set of
platforms; on a platform without one the sync stops at that package. You can
still work on the pure-Python parts: install the remaining dependencies by
hand and run the tier-1 tests described below, which never import cobre.

Install the pre-commit hook to run the same lint and gate steps as CI before
each commit:

```bash
ln -sf ../../scripts/pre-commit .git/hooks/pre-commit
```

## Running the checks

```bash
pytest                                        # the test suite
ruff check . && ruff format --check .         # lint and formatting, whole tree
python3 scripts/ci/check_no_plan_leaks.py     # no planning vocabulary in shipped artifacts
python3 scripts/ci/check_comment_refs.py      # no rot-prone references in source prose
python3 scripts/ci/check_doc_paths.py         # every path cited in the docs resolves
```

CI (`.github/workflows/ci.yml`) runs ruff and the three gates on one Python,
the test suite on every Python version the project supports, and a coverage
floor on the comparator reader layer (`[tool.coverage]` in `pyproject.toml`).
`scripts/README.md` describes each gate and the advisory reports next to them.

### Test tiers

The suite must collect and run in three environments, so every test sits in
one tier:

- **Tier 1**: pure Python, imports no `cobre`. Runs everywhere. No test module
  at any tier may `import cobre` at module scope.
- **Tier 2**: needs `cobre-python` but no solver binary and no real deck.
  Marked with `requires_cobre_python` from `tests/conftest.py`; the `import
  cobre` lives inside the guarded test body.
- **Tier 3**: needs the real solver binary or a real deck under the gitignored
  `example/` directory. Guarded with `skipif` on the path's existence. Never
  runs in CI.

`tests/decks/` holds two committed synthetic mini-decks (one per track) for
CI-tier end-to-end tests; no tier-1 or tier-2 test reads `example/`. On a
checkout without the local decks, one test in `tests/test_deck_inventory.py`
reports the guarded decks it cannot find; that check is dev-only and expected
to fail there.

The `tests/` tree mirrors `src/cobre_bridge/`: one `test_<module>.py` per
source module in the matching directory. Shared case builders (`make_case`,
`make_decomp_case`) live in `tests/conftest.py`. Golden files under
`tests/golden/` are regenerated with `scripts/regen-goldens.sh`, never edited
by hand. The full testing rules are in `.claude/rules/testing.md`.

## Conventions CI enforces

The rule files under `.claude/rules/` are written so a coding agent can act
on them, and they are the canonical statements for humans too. In short:

- **Package boundaries.** Imports flow downward only, nothing imports `cli`,
  and no underscore-private name crosses a package boundary
  (`tests/test_package_boundaries.py`; the map is in `docs/architecture.md`).
- **Twin-track symmetry.** A behaviour added to the NEWAVE track lands on the
  DECOMP track in the same change, and vice versa. Shared physics and
  calendar math live once, in `core/`.
- **Diagnostics, not strings.** A converter reports a degraded input as a
  `Diagnostic` through the `collect()` sink. Rich is imported only in `ui/`.
  CLI messages are self-contained: no repository paths or internal symbols,
  because most users have no checkout.
- **Cobre gaps are tracked.** A workaround for a cobre limitation carries a
  `TRACKED COBRE-GAP` comment at the site, a log line or diagnostic, and an
  entry with its removal condition in the cobre repository. Those comments
  are never removed in a cleanup pass.
- **Exit codes and `--json`.** An error status and a non-zero exit code travel
  together, and every failure path under `--json` emits exactly one verdict
  envelope on stdout.
- **Comments are default-off.** A comment ships only if deleting it would let
  a competent reader introduce a bug or lose a fact that lives outside the
  file (`.claude/rules/comments.md`).
- **Docs never freeze a count or version by hand.** State the rule, or pin the
  literal with a test. No planning vocabulary in shipped artifacts
  (`.claude/rules/doc-integrity.md`).

## Documentation

- `docs/cli.md` is generated from the Typer app. After changing any command
  or option help text, run `scripts/gen-cli-docs.sh` and commit the result;
  `tests/test_docs.py` fails when a command or flag is missing from it.
- `README.md` is also the PyPI description, so it links to `docs/` with
  absolute URLs and cites no repository-only paths.
- Add a line under `Unreleased` in `CHANGELOG.md` for every user-visible
  change.

## Adding a converter

1. Write the converter as a module under the track's `converters/` package: a
   function from the parsed case (and the id map) to tables or dicts, with no
   file I/O and no printing.
2. Emit a `Diagnostic` for every input the converter cannot carry over
   faithfully, with the affected entities and stages in its detail table.
3. Wire it into the track's `pipeline.py`. Write output only through the
   `CaseWriter`; register a new file's `$schema` URL in `cobre/schemas.py`.
4. Mirror it on the other track, or record the asymmetry where the team tracks
   architecture debt.
5. Add a tier-1 test with synthetic input, an emission-shape test for a new
   output file, and a CliRunner or subprocess test for a new flag or exit
   path.
6. Regenerate `docs/cli.md` if help text changed, and add a `CHANGELOG.md`
   entry.

## Releasing

A bridge release `X.Y.Z` pairs with cobre `X.Y.Z`.

1. Bump `version` in `pyproject.toml`. When the cobre pairing moves, bump the
   `cobre-python` floor and `MIN_COBRE_VERSION` (`src/cobre_bridge/cobre/compat.py`)
   together and refresh `uv.lock`; `tests/test_packaging.py` fails if they
   drift.
2. Move the `Unreleased` entries in `CHANGELOG.md` under the new version.
3. Tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` builds,
   runs the suite, and publishes to PyPI through trusted publishing.
