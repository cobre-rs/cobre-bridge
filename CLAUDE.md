# Cobre-Bridge — Development Guidelines

## Project Overview

Cobre-bridge is a Python CLI that converts power system case data
(currently NEWAVE) into [Cobre](https://github.com/cobre-rs/cobre) input
format, and provides tools to compare solver outputs between the original
and converted cases.

- **Language**: Python >=3.12
- **License**: Apache-2.0
- **Build system**: hatchling
- **Entry point**: `cobre-bridge` CLI (`cobre_bridge.cli:main`)

## Quick Reference

```bash
# Install (development)
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Format
ruff format

# Run CLI
cobre-bridge convert newave <SRC> <DST>
cobre-bridge compare bounds <NEWAVE_DIR> <COBRE_OUTPUT_DIR>
cobre-bridge compare results <NEWAVE_DIR> <COBRE_OUTPUT_DIR>
cobre-bridge dashboard <CASE_DIR>
```

## CI

- **Lint**: `ruff check src/` on Python 3.13
- **Test**: `pytest --tb=short` on Python 3.12, 3.13, 3.14
- **Release**: tag `v*` triggers PyPI publish via trusted publishing

---

## Architecture

### CLI Commands

| Command           | Module                   | Purpose                                     |
| ----------------- | ------------------------ | ------------------------------------------- |
| `convert newave`  | `pipeline.py`            | Convert NEWAVE case to Cobre input format   |
| `compare bounds`  | `comparators/bounds.py`  | Compare LP bounds between NEWAVE and Cobre  |
| `compare results` | `comparators/results.py` | Compare simulation results with HTML report |
| `dashboard`       | `dashboard.py`           | Generate interactive Plotly dashboard       |

### Core Modules

```
src/cobre_bridge/
  cli.py                    # Argparse CLI routing
  pipeline.py               # Conversion orchestrator (calls all converters)
  newave_files.py           # NEWAVE file discovery via caso.dat -> arquivos.dat
  id_map.py                 # NEWAVE 1-based -> Cobre 0-based ID mapping
  dashboard.py              # Interactive HTML dashboard (large, line-length exempt)
  converters/               # One module per entity/data category
    hydro.py                #   Hydro plants, geometry, production models
    thermal.py              #   Thermal plants, maintenance bounds
    network.py              #   Buses, lines, NCS, exchange factors
    temporal.py             #   Stages, config, CVaR from dger.dat/patamar.dat
    stochastic.py           #   Inflow/load stats and history
    initial_conditions.py   #   Reservoir storage, filling targets
    constraints.py          #   VminOP, electric, AGRINT constraints
  comparators/              # Validation against reference model outputs
    alignment.py            #   Entity mapping between NEWAVE and Cobre
    bounds.py               #   LP bounds comparison engine
    bounds_from_inputs.py   #   Compute expected bounds from NEWAVE inputs
    results.py              #   Results comparison engine
    charts.py               #   Plotly chart generation
    html_report.py          #   HTML report templates
    report_builder.py       #   Report assembly
    report.py               #   Summary reporting
    cobre_readers.py        #   Read Cobre output Parquet files
    newave_readers.py       #   Read NEWAVE result files
```

### Conversion Pipeline

`pipeline.convert_newave_case(src, dst)` orchestrates 25+ converter functions:

1. Discover NEWAVE files via `caso.dat` -> `arquivos.dat` (case-insensitive)
2. Build `NewaveIdMap` (deterministic 1-based -> 0-based remapping)
3. Run entity converters (hydro, thermal, network) -> JSON
4. Run temporal/stochastic converters -> JSON + Parquet
5. Run constraint converters -> JSON + Parquet bounds
6. Optionally validate output with `cobre-python` (`--validate` flag)

### Output Structure

Conversion produces a Cobre case directory:

```
output/
  config.json, stages.json, penalties.json, initial_conditions.json
  system/       hydros.json, thermals.json, buses.json, lines.json, ...
  scenarios/    inflow_*.parquet, load_*.parquet, load_factors.json, ...
  constraints/  hydro_bounds.parquet, thermal_bounds.parquet, line_bounds.parquet, ...
```

This matches the Cobre input format documented in the
[Cobre book](https://github.com/cobre-rs/cobre/tree/main/book).

---

## Key Dependencies

| Package        | Role                                                     |
| -------------- | -------------------------------------------------------- |
| `inewave`      | Read NEWAVE binary/text files (caso.dat, hidr.dat, etc.) |
| `pandas`       | Data manipulation in converters                          |
| `pyarrow`      | Parquet I/O for stage-varying data                       |
| `polars`       | DataFrames in comparators                                |
| `plotly`       | Charts and interactive dashboards                        |
| `cobre-python` | Optional validation of converted output                  |

---

## Testing

- Framework: pytest with `--tb=short`
- Test directory: `tests/`
- Pipeline tests use `unittest.mock.patch` to avoid requiring real NEWAVE files
- CLI error-path tests run as subprocess to verify exit codes
- Example data in `example/newave/` and `example/convertido/`

---

## Coding Conventions

- **Linter/formatter**: ruff (target Python 3.12, line-length 88)
- **Lint rules**: E (errors), F (pyflakes), I (isort), UP (pyupgrade)
- `dashboard.py` is exempt from E501 (line length) due to embedded HTML/JS
- Follow the Python rules in `~/.claude/rules/python.md` (type hints, pathlib,
  no bare except, no mutable defaults, etc.)

---

## Domain Context

### What is NEWAVE?

NEWAVE is a Brazilian hydrothermal dispatch tool. It uses proprietary
binary/text file formats with 1-based entity IDs, fixed-width fields, and
indirect file references (`caso.dat` -> `arquivos.dat` -> actual data files).

### What is Cobre?

Cobre is a modern Rust-based SDDP solver for hydrothermal dispatch. It uses
JSON for structured data and Parquet for time-series/stage-varying data, with
0-based entity IDs. See `/home/rogerio/git/cobre` for the main repository.

### Key Conversion Challenges

- **ID remapping**: NEWAVE uses arbitrary 1-based codes; Cobre uses dense 0-based indices.
  `NewaveIdMap` handles this deterministically.
- **Fictitious plants**: NEWAVE includes accounting-only entities that must be filtered out.
- **Post-study extrapolation**: Some NEWAVE data covers only the study period; converters
  must extend via last-year seasonal repetition for post-study stages.
- **Constraint merging**: VminOP, Electric, and AGRINT constraints map to Cobre's unified
  generic constraint format.
- **Risk parameters**: CVaR alpha/lambda from `cvar.dat` map to per-stage risk config.

---

## Working with the Example Case

```bash
# Convert the example NEWAVE case
cobre-bridge convert newave example/newave example/convertido --force

# Compare bounds (requires both NEWAVE and Cobre output)
cobre-bridge compare bounds example/newave <cobre_output_dir>

# Generate dashboard from Cobre results
cobre-bridge dashboard <cobre_output_dir>
```
