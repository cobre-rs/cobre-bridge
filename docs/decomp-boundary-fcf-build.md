# The boundary-FCF path: requirements and running

`convert decomp` imports the deck's boundary FCF by default. The importer
(`decomp/fcf/`) reads terminal cost-to-go cuts from a source-model deck,
bootstraps the terminal manifest with a real 1-iteration **in-process** cobre
pass (`cobre.run.run`), and writes the cuts into a Cobre policy checkpoint via
`cobre.write_policy_checkpoint`, using the **CBVF** checkpoint format (keyed on a
`delivery_date` entity-manifest field). Every cobre call is in-process against
the `cobre-python` wheel — the import needs no external `cobre` binary.

## Requirements

- **`cobre-python`** — and nothing else. The CBVF `delivery_date` checkpoint
  format ships in the released cobre wheels on PyPI, and `cobre-python` is a
  **required runtime dependency** of `cobre-bridge` (in `pyproject.toml`'s
  core `dependencies`). A normal `pip install cobre-bridge` therefore installs
  everything the import needs — no wheel to build, no binary to supply. The
  version floor is stated once, in `pyproject.toml`, in lockstep with
  `MIN_COBRE_VERSION` in `src/cobre_bridge/cli.py`; `tests/test_packaging.py`
  guards that lockstep, so this doc deliberately repeats no number.

An external `cobre` binary is needed only later, to _solve_ the converted case
(`cobre run`), which is a separate step downstream of the bridge — see
"Running an imported case" below.

If the FCF is not wanted for a given conversion, pass `--no-fcf` to skip the
whole step (it is also always skipped under `--dry-run`).

## Confirming capability

The bridge does not trust the wheel's self-reported version. The authoritative
check is the runtime **capability probe** in
`src/cobre_bridge/decomp/fcf/capability.py` (`ensure_boundary_fcf_capability()`):
it performs a real CBVF write→load round trip against the installed wheel and
raises a `RuntimeError` with a self-contained remediation message if the wheel
cannot write or reload the `delivery_date` checkpoint format. A round trip (rather than a version string)
also catches a broken, partial, or ABI-mismatched wheel. Any bridge command that
gates on it — e.g. `convert decomp` on a deck that declares cut files — exercises
the probe. If it fails, reinstall or upgrade `cobre-python` (`pip install
--upgrade cobre-python`, or reinstall `cobre-bridge`).

## Running an imported case

Once a case has been converted with the boundary FCF imported, `cobre run` must
be invoked with an explicit `--output` pointed at the case directory itself, not
a separate output directory:

```bash
cobre run <case_dir> --output <case_dir>
```

This is a tracked cobre-gap (C8, D7): `policy.boundary.path` in the emitted
checkpoint metadata resolves relative to the run's `--output` directory, not the
case directory it was authored into, so a plain `cobre run <case_dir>` (default
output elsewhere) fails to locate the boundary checkpoint. The bridge's
`--boundary-fcf` wiring surfaces this constraint in its own output after a
successful import; it is worked around at the call site, not solved by the
bridge. Its removal condition is tracked in the cobre repository's
conversion-found-improvements registry.
