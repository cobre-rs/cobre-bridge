# cobre-bridge

Convert power system data formats to [Cobre](https://github.com/cobre-rs/cobre) input format.

## Installation

```bash
# Recommended: isolated, on-PATH CLI
uv tool install cobre-bridge

# Alternative
pipx install cobre-bridge

# With cobre-python, so `convert --validate` can validate the output
uv tool install "cobre-bridge[validation]"
```

Requires Python >= 3.12.

## Usage

| Command                                                 | What it does                                            |
| ------------------------------------------------------- | ------------------------------------------------------- |
| `cobre-bridge convert newave <SRC> <DST>`               | Convert a NEWAVE case to Cobre input format.            |
| `cobre-bridge check newave <SRC>`                       | Preflight-validate NEWAVE inputs (no conversion).       |
| `cobre-bridge compare bounds <NEWAVE_DIR> <COBRE_DIR>`  | Compare LP bounds between the two cases.                |
| `cobre-bridge compare results <NEWAVE_DIR> <COBRE_DIR>` | Compare simulation results against Cobre output.        |
| `cobre-bridge dashboard <CASE_DIR>`                     | Build an interactive HTML dashboard from Cobre results. |

Every command accepts `--json` (a single machine-readable verdict
`{schema_version, command, status, summary, diagnostics}` on stdout) and
`-v/-vv`, `--log-file PATH`, `--no-color`, `--quiet`. `convert` adds
`--validate`, `--force`, `--dry-run`, `--diagnostics-json PATH`. `dashboard`
adds `--open`. `compare` adds `--tolerance`, `--format`, `--out-dir` (and
`compare bounds` also `--summary`, `--variables`).

See [docs/cli.md](docs/cli.md) for the full per-command reference.

> **Future (`NE`) hydro plants.** `convert newave` now converts NEWAVE `NE`
> (future, will-be-built) plants using cobre's dead-volume **filling** schema.
> A converted case containing `NE` plants requires **cobre >= 0.9.0**; EX-only
> cases still load on cobre >= 0.8.2. `convert --validate` skips validation for
> `NE`-with-filling cases until a `cobre-python` 0.9.0 release ships the schema.

## Configuration

Defaults for compare `tolerance`, `format`, and `out_dir` can be set in a
`cobre-bridge.toml` file (discovered from the working directory upward, then
`$XDG_CONFIG_HOME/cobre-bridge/config.toml`, then
`~/.config/cobre-bridge/config.toml`) or via `COBRE_BRIDGE_*` environment
variables (e.g. `COBRE_BRIDGE_BOUNDS_TOLERANCE`, `COBRE_BRIDGE_FORMAT`,
`COBRE_BRIDGE_OUT_DIR`). Precedence is flag > env > config file > default.

## Shell completion

```bash
cobre-bridge --install-completion   # install for your shell
cobre-bridge --show-completion      # print the script
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## License

Apache-2.0
