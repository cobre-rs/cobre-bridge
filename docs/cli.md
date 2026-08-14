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
$ cobre-bridge dashboard [OPTIONS] {case_dir}
```

**Arguments**:

* `case_dir`: Path to the Cobre case directory.  [required]

**Options**:

* `-o, --output <path>`: Output HTML file path (default: &lt;case_dir&gt;/dashboard.html).
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
* `decomp`: Convert a DECOMP deck revision to a Cobre...

### `cobre-bridge convert newave`

Convert a NEWAVE case directory to a Cobre case directory.

**Usage**:

```console
$ cobre-bridge convert newave [OPTIONS] {src} {dst}
```

**Arguments**:

* `src`: Path to the NEWAVE case directory.  [required]
* `dst`: Path to the output Cobre case directory.  [required]

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

### `cobre-bridge convert decomp`

Convert a DECOMP deck revision to a Cobre case directory.

Loop-closing subset: the exchange network, renewables card file, and GNL
anticipation are deferred and reported as warnings. The boundary FCF is
imported by default whenever the deck declares its cut files; ``--no-fcf``
skips it.

**Usage**:

```console
$ cobre-bridge convert decomp [OPTIONS] {src} {dst}
```

**Arguments**:

* `src`: Path to the DECOMP deck directory.  [required]
* `dst`: Path to the output Cobre case directory.  [required]

**Options**:

* `--force`: Overwrite destination directory if it already contains files.
* `--validate`: After conversion, validate the output with the cobre package.
* `--diagnostics-json PATH`: Also write the conversion diagnostics (counts + findings) as JSON.
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) rendering.
* `--dry-run`: Run the full conversion in memory and report what would be written, without creating or modifying the destination directory.
* `--no-fcf`: Skip importing the deck&#x27;s boundary FCF. By default, when the deck declares cortes/cortesh files (its FC records), they are imported as a terminal-stage cobre policy checkpoint via an in-process 1-iteration cobre pass (slow; requires cobre-python). Pass this for a quick conversion without the terminal FCF. The FCF is always skipped under --dry-run.
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

* `decomp`: Compare a DECOMP run&#x27;s published operation...
* `newave`: Compare NEWAVE published results against...

### `cobre-bridge compare decomp`

Compare a DECOMP run&#x27;s published operation against Cobre&#x27;s simulation.

Informational: always exits 0, reporting divergences without failing.

**Usage**:

```console
$ cobre-bridge compare decomp [OPTIONS] {decomp_dir} {cobre_output_dir}
```

**Arguments**:

* `decomp_dir`: Path to the DECOMP deck directory (deck + dec_oper_*.csv result files, all directly in it).  [required]
* `cobre_output_dir`: Path to the Cobre output directory.  [required]

**Options**:

* `--tolerance <float>`: Relative tolerance for the within-tolerance verdict (default 1e-2; overridable via COBRE_BRIDGE_RESULTS_TOLERANCE or cobre-bridge.toml).  [env var: COBRE_BRIDGE_RESULTS_TOLERANCE]
* `--format FORMAT`: Output format(s): console,html,csv,parquet,json,all. Comma-separated and/or repeatable. Overridable via COBRE_BRIDGE_FORMAT or cobre-bridge.toml. (default: console,parquet,json)  [env var: COBRE_BRIDGE_FORMAT]
* `--out-dir <path>`: Directory for file artifacts. Overridable via COBRE_BRIDGE_OUT_DIR or cobre-bridge.toml. (default: &lt;cobre_output_dir&gt;/comparison_artifacts).  [env var: COBRE_BRIDGE_OUT_DIR]
* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) tables.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

### `cobre-bridge compare newave`

Compare NEWAVE published results against Cobre simulation output.

Informational: always exits 0, reporting divergences without failing.

**Usage**:

```console
$ cobre-bridge compare newave [OPTIONS] {newave_dir} {cobre_output_dir}
```

**Arguments**:

* `newave_dir`: Path to the NEWAVE case directory (case + MEDIAS-*.CSV result files, all directly in it).  [required]
* `cobre_output_dir`: Path to the Cobre output directory.  [required]

**Options**:

* `--tolerance <float>`: Relative tolerance for results comparison (default 1e-2; overridable via COBRE_BRIDGE_RESULTS_TOLERANCE or cobre-bridge.toml).  [env var: COBRE_BRIDGE_RESULTS_TOLERANCE]
* `--format FORMAT`: Output format(s): console,html,csv,parquet,json,all. Comma-separated and/or repeatable. Overridable via COBRE_BRIDGE_FORMAT or cobre-bridge.toml. (default: console,parquet,json)  [env var: COBRE_BRIDGE_FORMAT]
* `--out-dir <path>`: Directory for file artifacts. Overridable via COBRE_BRIDGE_OUT_DIR or cobre-bridge.toml. (default: &lt;cobre_output_dir&gt;/comparison_artifacts).  [env var: COBRE_BRIDGE_OUT_DIR]
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
* `decomp`: Validate a DECOMP deck revision without...

### `cobre-bridge check newave`

Validate a NEWAVE case directory without converting or writing any files.

**Usage**:

```console
$ cobre-bridge check newave [OPTIONS] {src}
```

**Arguments**:

* `src`: Path to the NEWAVE case directory.  [required]

**Options**:

* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) checklist.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.

### `cobre-bridge check decomp`

Validate a DECOMP deck revision without converting or writing any files.

Also reports what the conversion will leave behind, so a deferred feature
is never a silent omission.

**Usage**:

```console
$ cobre-bridge check decomp [OPTIONS] {src}
```

**Arguments**:

* `src`: Path to the DECOMP deck directory.  [required]

**Options**:

* `--json`: Emit a single machine-readable JSON verdict to stdout and suppress the human (Rich) checklist.
* `-v, --verbose`: Increase console log verbosity (-v INFO, -vv DEBUG).  [default: 0]
* `--log-file PATH`: Write the full DEBUG log to PATH (the console verbosity is unaffected).
* `--no-color`: Disable coloured output (also honoured via the NO_COLOR env var).
* `--quiet`: Suppress the summary and info notes; warnings/errors still show.
* `--help`: Show this message and exit.
