# `cobre-bridge`

Convert power system data to Cobre input format.

**Usage**:

```console
$ cobre-bridge [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--version`: Show the version and exit.
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `dashboard`: Generate an interactive HTML dashboard...
* `convert`: Convert data from a source format to Cobre...
* `compare`: Compare source model inputs/results...
* `check`: Validate source-model inputs without...

## `cobre-bridge dashboard`

Generate an interactive HTML dashboard from Cobre simulation results.

**Usage**:

```console
$ cobre-bridge dashboard [OPTIONS] CASE_DIR
```

**Arguments**:

* `CASE_DIR`: Path to the Cobre case directory.  [required]

**Options**:

* `-o, --output PATH`: Output HTML file path (default: &lt;case_dir&gt;/dashboard.html).
* `--open`: Open the generated dashboard in the default web browser after writing it.
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) status lines.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

## `cobre-bridge convert`

Convert data from a source format to Cobre JSON.

**Usage**:

```console
$ cobre-bridge convert [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `newave`: Convert a NEWAVE case directory to a Cobre...

### `cobre-bridge convert newave`

Convert a NEWAVE case directory to a Cobre case directory.

**Usage**:

```console
$ cobre-bridge convert newave [OPTIONS] SRC DST
```

**Arguments**:

* `SRC`: Path to the NEWAVE case directory.  [required]
* `DST`: Path to the output Cobre case directory.  [required]

**Options**:

* `--validate`: After conversion, validate the output with the cobre package.
* `--force`: Overwrite destination directory if it already contains files.
* `--diagnostics-json PATH`: Also write the conversion diagnostics (counts + findings) as JSON.
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) rendering.
* `--dry-run`: Run the full conversion in memory and report what would be written, without creating or modifying the destination directory.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

## `cobre-bridge compare`

Compare source model inputs/results against Cobre.

**Usage**:

```console
$ cobre-bridge compare [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `bounds`: Compare LP bounds computed from NEWAVE...
* `results`: Compare NEWAVE published results against...

### `cobre-bridge compare bounds`

Compare LP bounds computed from NEWAVE inputs against Cobre bounds.

**Usage**:

```console
$ cobre-bridge compare bounds [OPTIONS] NEWAVE_DIR COBRE_OUTPUT_DIR
```

**Arguments**:

* `NEWAVE_DIR`: Path to the NEWAVE case directory.  [required]
* `COBRE_OUTPUT_DIR`: Path to the Cobre output directory (has bounds.parquet).  [required]

**Options**:

* `--tolerance FLOAT`: Absolute tolerance for bound comparison (default 1e-3; overridable via COBRE_BRIDGE_BOUNDS_TOLERANCE or cobre-bridge.toml).  [env var: COBRE_BRIDGE_BOUNDS_TOLERANCE]
* `--format FORMAT`: Output format(s): console,html,csv,parquet,json,all. Comma-separated and/or repeatable. Overridable via COBRE_BRIDGE_FORMAT or cobre-bridge.toml. (default: console,parquet,json)  [env var: COBRE_BRIDGE_FORMAT]
* `--out-dir PATH`: Directory for file artifacts. Overridable via COBRE_BRIDGE_OUT_DIR or cobre-bridge.toml. (default: &lt;cobre_output_dir&gt;/comparison_artifacts).  [env var: COBRE_BRIDGE_OUT_DIR]
* `--summary`: Print only summary counts, not individual mismatches.
* `--variables TEXT`: Comma-separated variables to compare (e.g. storage_min,turbined).
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) tables.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

### `cobre-bridge compare results`

Compare NEWAVE published results against Cobre simulation output.

Informational: always exits 0, whereas &#x27;compare bounds&#x27; exits 1 on any mismatch.

**Usage**:

```console
$ cobre-bridge compare results [OPTIONS] NEWAVE_DIR COBRE_OUTPUT_DIR
```

**Arguments**:

* `NEWAVE_DIR`: Path to the NEWAVE case directory (has saidas/).  [required]
* `COBRE_OUTPUT_DIR`: Path to the Cobre output directory.  [required]

**Options**:

* `--tolerance FLOAT`: Relative tolerance for results comparison (default 1e-2; overridable via COBRE_BRIDGE_RESULTS_TOLERANCE or cobre-bridge.toml).  [env var: COBRE_BRIDGE_RESULTS_TOLERANCE]
* `--format FORMAT`: Output format(s): console,html,csv,parquet,json,all. Comma-separated and/or repeatable. Overridable via COBRE_BRIDGE_FORMAT or cobre-bridge.toml. (default: console,parquet,json)  [env var: COBRE_BRIDGE_FORMAT]
* `--out-dir PATH`: Directory for file artifacts. Overridable via COBRE_BRIDGE_OUT_DIR or cobre-bridge.toml. (default: &lt;cobre_output_dir&gt;/comparison_artifacts).  [env var: COBRE_BRIDGE_OUT_DIR]
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) tables.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

## `cobre-bridge check`

Validate source-model inputs without converting.

**Usage**:

```console
$ cobre-bridge check [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `newave`: Validate a NEWAVE case directory without...

### `cobre-bridge check newave`

Validate a NEWAVE case directory without converting or writing any files.

**Usage**:

```console
$ cobre-bridge check newave [OPTIONS] SRC
```

**Arguments**:

* `SRC`: Path to the NEWAVE case directory.  [required]

**Options**:

* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) checklist.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.
