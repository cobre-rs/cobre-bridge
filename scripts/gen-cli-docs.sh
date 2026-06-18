#!/usr/bin/env bash
# Regenerate docs/cli.md from the Typer app. Dev/build-time only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${PYTHON:-python}"

"$PYTHON" -m typer cobre_bridge.cli utils docs \
    --name cobre-bridge \
    --output "$REPO_ROOT/docs/cli.md"

echo "Regenerated $REPO_ROOT/docs/cli.md" >&2
