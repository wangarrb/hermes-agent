#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$ROOT/local/bin/start-kanban.sh"

shared_dirs_line="$(grep -E '^[[:space:]]*for d in claude-skills ' "$SCRIPT")"
shared_dirs="${shared_dirs_line#* in }"
shared_dirs="${shared_dirs%; do}"
if [[ " $shared_dirs " != *" agents "* ]]; then
    echo "start-kanban.sh does not expose ~/.codex/agents to per-role CODEX_HOME" >&2
    exit 1
fi

echo "per-role Codex custom agents: PASS"
