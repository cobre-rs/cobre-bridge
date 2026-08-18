# Building cobre for the boundary-FCF path

The boundary-FCF importer (`decomp/fcf/`) reads terminal cost-to-go cuts from a
source-model deck and writes them into a Cobre policy checkpoint via `cobre.write_policy_checkpoint`.
That checkpoint format — CBVF, with a `delivery_date` entity-manifest key — only exists on
the cobre `feat/cobre-gnl-boundary-pricing` branch. It is **not a released wheel**: there is
no tagged version to `pip install`, so the branch must be built locally into the bridge
`.venv` from a worktree. This doc is the exact, verified recipe.

See `plans/decomp-boundary-fcf-feature.md` §2/§5 (D1) for the decision this recipe
implements, and `plans/decomp-boundary-fcf/epic-01-adoption-environment/` for the tickets
that consume it.

## Prerequisites

- A git worktree of `~/git/cobre` checked out at `feat/cobre-gnl-boundary-pricing`, at
  `~/git/cobre-gnlbp`. If it does not exist yet, create it from the main cobre checkout:

  ```bash
  git -C ~/git/cobre worktree add ~/git/cobre-gnlbp feat/cobre-gnl-boundary-pricing
  ```

- Submodules initialized in that worktree:

  ```bash
  git -C ~/git/cobre-gnlbp submodule update --init --recursive
  ```

- Rust toolchain **1.94.1** (pinned by `~/git/cobre-gnlbp/rust-toolchain.toml`; `rustup`
  installs it automatically on first build inside the worktree).
- `maturin` and `patchelf` installed into the bridge `.venv`:

  ```bash
  <bridge>/.venv/bin/pip install maturin
  <bridge>/.venv/bin/pip install patchelf
  ```

  `patchelf` silences maturin's rpath warning when it packages the extension module; the
  build still succeeds without it, but the warning is noise on every rebuild.

## Build the wheel

Build the Python extension module directly into the bridge `.venv` with
`maturin develop --release`, pointed at the worktree's `cobre-python` crate:

```bash
VIRTUAL_ENV=<bridge>/.venv <bridge>/.venv/bin/maturin develop --release \
    -m ~/git/cobre-gnlbp/crates/cobre-python/Cargo.toml
```

This installs a `cobre` package into the bridge `.venv` that exposes
`cobre.write_policy_checkpoint` and `cobre.results.load_policy`, both required by
`decomp/fcf/bootstrap.py` and `decomp/fcf/writer.py`.

## Build the binary

The importer's bootstrap step (`decomp/fcf/bootstrap.py::bootstrap_terminal_manifest`) also
needs a `cobre` binary to run a real 1-iteration training pass. Build it from the same
worktree:

```bash
cargo build --release --bin cobre --manifest-path ~/git/cobre-gnlbp/Cargo.toml
```

The resulting binary lives at `~/git/cobre-gnlbp/target/release/cobre`. Pass it explicitly
wherever the bridge accepts a `--cobre-bin` option (see `--boundary-fcf` in
`convert decomp`'s CLI reference); the bridge does not assume it is on `PATH`.

## Confirm the build

Do not trust the wheel's self-reported version to confirm the build succeeded — see
"Why not a version bump" below. The authoritative confirmation is the runtime **capability
probe** in `src/cobre_bridge/decomp/fcf/capability.py`
(`ensure_boundary_fcf_capability()`, epic-01 ticket-002): it performs a real CBVF
write→load round-trip against the installed wheel and raises a `RuntimeError` naming this
doc if the wheel cannot write or reload the `delivery_date` checkpoint format. Run it (or
any bridge command that gates on it, e.g. `convert decomp --boundary-fcf`) after a build to
confirm the wheel and binary are the branch build, not a stale or released install.

## Run an imported case

Once a case has been converted with the boundary FCF imported, `cobre run` must be invoked
with an explicit `--output` pointed at the case directory itself, not a separate output
directory:

```bash
~/git/cobre-gnlbp/target/release/cobre run <case_dir> --output <case_dir>
```

This is a tracked cobre-gap on the branch (C8, D7): `policy.boundary.path` in the emitted
checkpoint metadata resolves relative to the run's `--output` directory, not the case
directory it was authored into, so a plain `cobre run <case_dir>` (default output elsewhere)
fails to locate the boundary checkpoint. The bridge's `--boundary-fcf` wiring surfaces this
constraint in its own output; it is not solved by this recipe or by the bridge, only worked
around at the call site.

## Why not a version bump

The branch wheel **self-reports version `0.13.0`** (its `Cargo.toml` version was never
bumped off the branch point), yet it writes the CBVF checkpoint format keyed on the
`delivery_date` entity-manifest field — a format the released `0.13.0` wheel on PyPI/the
`cobre-python` extra does **not** understand and rejects on load. A version-string
comparison therefore cannot distinguish "the branch build is installed" from "the released
0.13.0 is installed": both report the identical version string, but only one can read/write
the boundary-FCF checkpoint.

For this reason, adoption of the branch is a **documented local build** (this doc) plus a
**runtime capability probe** (`ensure_boundary_fcf_capability()`), never a dependency
version bump:

- `cobre-python>=0.13,<0.14` (the `validation` / `test-roundtrip` extras in
  `pyproject.toml`) stays as-is.
- `MIN_COBRE_VERSION` in `src/cobre_bridge/cli.py` stays `"0.13.0"`.

Bumping either would be misleading (there is no new tagged release to bump to) and would
not actually gate the boundary-FCF path on the capability that matters. Only the
`--boundary-fcf` path is gated, via the capability probe; a plain `convert decomp` already
validates 0 errors against both the branch build and a released 0.13.0 wheel and does not
need this recipe at all.
