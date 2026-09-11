# cobre-bridge

`cobre-bridge` converts hydrothermal dispatch cases written for the Brazilian
planning models **NEWAVE** (long term) and **DECOMP** (short term) into the
input format of [Cobre](https://github.com/cobre-rs/cobre), an open-source
SDDP solver, and compares the two models' results once both have been run.

It is a command-line tool. A session typically goes:

1. `check` that a source case is complete enough to convert;
2. `convert` it into a Cobre case directory;
3. solve that directory with the `cobre` solver;
4. `compare` the source model's published results against Cobre's simulation;
5. open an interactive `dashboard` of the Cobre run.

## Installation

```bash
uv tool install cobre-bridge    # isolated, on-PATH CLI (recommended)
pipx install cobre-bridge       # alternative
pip install cobre-bridge        # into the current environment
```

Requires Python 3.12 or newer. The install pulls in `cobre-python`, Cobre's
Python bindings, so `convert --validate`, `compare`, and the DECOMP boundary
cost-to-go import work without further setup. `cobre-python` ships prebuilt
wheels for common platforms; if pip reports that none matches yours, see the
[cobre repository](https://github.com/cobre-rs/cobre) for build options.

The `cobre` solver itself is a separate install (`cargo install cobre-cli`;
see the cobre README). cobre-bridge does not need it to convert or compare.
You need it to solve the converted case between those two steps.

### Versions

A cobre-bridge release `X.Y.Z` targets cobre `X.Y.Z`: the converted case
follows that cobre release's input contract, and the `cobre-python`
dependency floor is the oldest cobre the output loads on. A packaging test
keeps the two in lockstep. `convert --validate` skips its validation step,
with a note, when the installed `cobre-python` is older than that floor.

## Quick start

```bash
# 1. Preflight the source case. Exit 0 = ready, 1 = ready with warnings, 2 = will not convert.
cobre-bridge check newave /path/to/newave_case

# 2. Convert into a new Cobre case directory (--force overwrites a non-empty one).
cobre-bridge convert newave /path/to/newave_case ./my_case

# 3. Solve with cobre. By default the solver writes its results to ./my_case/output/.
cobre run ./my_case

# 4. Compare NEWAVE's published results with Cobre's simulation. Always exits 0.
cobre-bridge compare newave /path/to/newave_case ./my_case/output --format html

# 5. Browse the Cobre run.
cobre-bridge dashboard ./my_case --open
```

Replace `newave` with `decomp` for a DECOMP deck; the flow is the same. The
DECOMP track imports the deck's boundary cost-to-go function by default, which
adds one rule for running the case; see the
[DECOMP track page](https://github.com/cobre-rs/cobre-bridge/blob/main/docs/decomp.md).

## Commands

| Command                                        | What it does                                                                                                        |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `convert newave SRC DST`                       | Convert a NEWAVE case directory into a Cobre case directory.                                                        |
| `convert decomp SRC DST`                       | Convert a DECOMP deck revision into a Cobre case directory.                                                         |
| `check newave SRC` / `check decomp SRC`        | Validate the source inputs without writing anything; report missing inputs and what the conversion would leave out. |
| `compare newave NEWAVE_DIR COBRE_OUTPUT_DIR`   | Compare NEWAVE's published results (`MEDIAS-*.CSV`, `pmo.dat`) against a Cobre simulation output directory.         |
| `compare decomp DECOMP_DIR COBRE_OUTPUT_DIR`   | Compare a DECOMP run's operation tables (`dec_oper_*.csv`) against a Cobre simulation output directory.             |
| `dashboard CASE_DIR`                           | Build an interactive HTML dashboard from a solved Cobre case (reads `CASE_DIR/output/`).                            |

Every command accepts `--json`, `-v`/`-vv`, `--log-file PATH`, `--no-color`,
and `--quiet`. `convert` adds `--validate`, `--force`, `--dry-run`, and
`--diagnostics-json PATH`; `convert decomp` also has `--no-fcf`. `compare`
adds `--tolerance`, `--format`, and `--out-dir`. `dashboard` adds `--output`
and `--open`. The generated reference with every option's help text is
[docs/cli.md](https://github.com/cobre-rs/cobre-bridge/blob/main/docs/cli.md).
Shell completion: `cobre-bridge --install-completion`.

### What `convert` writes

`DST` becomes a Cobre case directory: `config.json`, `stages.json`,
`penalties.json`, and `initial_conditions.json` at the top, entity registries
under `system/`, inflow and load data under `scenarios/`, and bounds and
generic constraints under `constraints/`. Stage-varying tables are Parquet;
everything else is JSON. The layout is Cobre's documented input format (see
the [cobre documentation](https://docs.cobre-rs.dev/)).

Next to the case, `conversion_manifest.json` records provenance: the bridge
version and git commit, the source directory, a hash of every input file read,
the entity counts, and every diagnostic raised during the run.

Diagnostics (a missing optional file, a bound clamped to keep the LP feasible,
a feature the converter leaves out) are rendered as grouped panels on stderr
with the affected plants and stages. `--diagnostics-json` saves them to a
file; `--dry-run` runs the whole conversion in memory and lists what would be
written.

### What `compare` writes

`compare` prints a per-variable summary table and writes artifacts to
`COBRE_OUTPUT_DIR/comparison_artifacts/` (or `--out-dir`). `--format` chooses
the set: `parquet` and `json` (the tidy comparison dataset) by default, `csv`,
`html` for a multi-tab `report.html` covering costs, system and energy
balance, network, convergence, performance, and per-plant detail, or `all`.
`compare` is informational and always exits 0; the report is where you judge
the differences.

### Machine-readable output

With `--json`, a command prints exactly one JSON document to stdout,
`{schema_version, command, status, summary, diagnostics}`, and nothing else on
stdout, including when it fails. Human-readable rendering is suppressed.

| Command     | Exit codes                                                                |
| ----------- | ------------------------------------------------------------------------- |
| `check`     | 0 ready, 1 ready with warnings, 2 will not convert                        |
| `convert`   | 0 converted, 1 conversion error, 2 `--validate` reported a load failure   |
| `compare`   | always 0                                                                  |
| `dashboard` | 0 written, 1 error                                                        |

An error status and a non-zero exit code always travel together.

## Configuration

Defaults for the `compare` options can live in a `cobre-bridge.toml`:

```toml
[compare]
format = ["console", "html"]   # or one string; same tokens as --format
out_dir = "comparisons"

[compare.results]
tolerance = 0.01               # relative
```

The file is looked up in the working directory and its parents, then
`$XDG_CONFIG_HOME/cobre-bridge/config.toml`, then
`~/.config/cobre-bridge/config.toml`; the first one found is used. The same
settings exist as environment variables (`COBRE_BRIDGE_RESULTS_TOLERANCE`,
`COBRE_BRIDGE_FORMAT`, `COBRE_BRIDGE_OUT_DIR`). Precedence is flag >
environment > file > built-in default.

## Documentation

- [CLI reference](https://github.com/cobre-rs/cobre-bridge/blob/main/docs/cli.md), generated from the command help.
- [NEWAVE track](https://github.com/cobre-rs/cobre-bridge/blob/main/docs/newave.md): inputs, what the converter changes, comparing results.
- [DECOMP track](https://github.com/cobre-rs/cobre-bridge/blob/main/docs/decomp.md): inputs, deferred features, the boundary cost-to-go import, running the case.
- [Architecture](https://github.com/cobre-rs/cobre-bridge/blob/main/docs/architecture.md): how the code is organised, for contributors.
- [Contributing](https://github.com/cobre-rs/cobre-bridge/blob/main/CONTRIBUTING.md): development setup, tests, quality gates, releasing.
- [Changelog](https://github.com/cobre-rs/cobre-bridge/blob/main/CHANGELOG.md).

## License

Apache-2.0
