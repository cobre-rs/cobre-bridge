#!/usr/bin/env bash
# Build the cobre boundary-FCF branch wheel + binary into the bridge .venv.
# Mirrors docs/decomp-boundary-fcf-build.md; usable from the repo root. Dev/build-time only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
VENV="${VENV:-$REPO_ROOT/.venv}"
WORKTREE="${1:-$HOME/git/cobre-gnlbp}"

STEP="locate worktree"
trap 'echo "build-cobre-boundary-fcf: failed during: $STEP" >&2' ERR

if [[ ! -d "$WORKTREE" ]]; then
    echo "build-cobre-boundary-fcf: worktree not found at $WORKTREE" >&2
    echo "create it with: git -C ~/git/cobre worktree add $WORKTREE feat/cobre-gnl-boundary-pricing" >&2
    exit 1
fi

for cmd in git cargo; do
    STEP="check for $cmd"
    command -v "$cmd" >/dev/null 2>&1 || {
        echo "build-cobre-boundary-fcf: required command not found: $cmd" >&2
        exit 1
    }
done

if [[ ! -x "$VENV/bin/pip" ]]; then
    echo "build-cobre-boundary-fcf: no venv pip at $VENV/bin/pip (set VENV=<path> to override)" >&2
    exit 1
fi

STEP="git submodule update --init --recursive"
git -C "$WORKTREE" submodule update --init --recursive

STEP="pip install maturin patchelf"
"$VENV/bin/pip" install maturin patchelf

STEP="maturin develop --release (wheel)"
VIRTUAL_ENV="$VENV" "$VENV/bin/maturin" develop --release \
    -m "$WORKTREE/crates/cobre-python/Cargo.toml"

STEP="cargo build --release --bin cobre (binary)"
cargo build --release --bin cobre --manifest-path "$WORKTREE/Cargo.toml"

echo "build-cobre-boundary-fcf: done." >&2
echo "  wheel installed into: $VENV" >&2
echo "  binary at:            $WORKTREE/target/release/cobre" >&2
