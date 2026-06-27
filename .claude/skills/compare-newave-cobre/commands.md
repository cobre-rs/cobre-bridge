# Comparison Commands

Copy-paste commands for comparing cases. Paths below are **verified against this
repo**: NEWAVE results live in `example/newave/` (has `saidas/`); the
converted-and-run Cobre case is `example/cobre/`, whose bounds dictionary
is at `output/training/dictionaries/bounds.parquet`.

> Prerequisite: the Cobre case must already be **converted and run** so that
> `output/` exists (`training/`, `simulation/`, …). `example/cobre` already
> satisfies this. To produce a fresh case, see `convert newave` in the project
> `CLAUDE.md` — convert into a *new* directory and run Cobre before comparing, so
> you don't clobber an existing run's `output/`.

## Example case (canonical)

```bash
# Compare LP bounds — absolute tolerance, default 1e-3.
# exit 0 = no mismatch, exit 1 = mismatches.
cobre-bridge compare bounds example/newave example/cobre/output

# ...summary counts only:
cobre-bridge compare bounds example/newave example/cobre/output --summary

# ...focus on specific variables and dump a Parquet diff:
cobre-bridge compare bounds example/newave example/cobre/output \
    --variables storage_min,turbined_max --output bounds_diff.parquet

# Compare results — relative tolerance, default 1e-2; writes the HTML report.
cobre-bridge compare results example/newave example/cobre/output \
    -o example/comparacao.html

# Interactive dashboard from Cobre simulation output.
cobre-bridge dashboard example/cobre
```

## Production case (template)

```bash
NW=/path/to/newave_case          # must contain saidas/
CB=/path/to/cobre_case/output    # must contain training/dictionaries/bounds.parquet

cobre-bridge compare bounds  "$NW" "$CB" --output bounds_diff.parquet
cobre-bridge compare results "$NW" "$CB" -o comparacao.html --tolerance 1e-2
```

## Gotchas

- `compare bounds` reads `<cobre_output_dir>/training/dictionaries/bounds.parquet`
  — point `cobre_output_dir` at the `output/` dir, **not** the case root.
- `compare results` needs both the NEWAVE `saidas/` and the Cobre `simulation/`
  output; it also reads `system/lines.json` from the case root.
- Bounds tolerance is **absolute**; results tolerance is **relative**. Always
  report which value you used.
- `compare bounds` exits non-zero on mismatches (useful in CI); `compare results`
  always exits 0.
