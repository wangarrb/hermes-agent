#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$ROOT/local/lib/codex_role_home.sh"

# shellcheck source=../local/lib/codex_role_home.sh
source "$LIB"

# ── Test: per-role home shares agents and config from real home ──

REAL_HOME="$(mktemp -d)"
ROLE_HOME="$(mktemp -d)"
trap 'rm -rf "$REAL_HOME" "$ROLE_HOME"' EXIT

# Simulate a real ~/.codex with config and agents
mkdir -p "$REAL_HOME/.codex/agents"
echo '[model]' > "$REAL_HOME/.codex/config.toml"
echo 'name = "luna"' > "$REAL_HOME/.codex/agents/luna.toml"

ensure_codex_role_home "$REAL_HOME" "$ROLE_HOME"

# sessions directory must exist
if [ ! -d "$ROLE_HOME/sessions" ]; then
    echo "FAIL: sessions directory not created" >&2
    exit 1
fi

# config.toml must be a symlink resolving to the real file
if [ ! -L "$ROLE_HOME/config.toml" ]; then
    echo "FAIL: config.toml not symlinked" >&2
    exit 1
fi

# agents/luna.toml must resolve to the same file as the source
if [ ! -e "$ROLE_HOME/agents/luna.toml" ]; then
    echo "FAIL: agents/luna.toml not accessible" >&2
    exit 1
fi

real_file="$REAL_HOME/.codex/agents/luna.toml"
role_file="$ROLE_HOME/agents/luna.toml"
if ! diff -q "$real_file" "$role_file" >/dev/null 2>&1; then
    echo "FAIL: agents/luna.toml content mismatch" >&2
    exit 1
fi

# Existing entries must not be overwritten
echo 'custom' > "$ROLE_HOME/config.toml"
ensure_codex_role_home "$REAL_HOME" "$ROLE_HOME"
if [ "$(cat "$ROLE_HOME/config.toml")" != "custom" ]; then
    echo "FAIL: existing config.toml was overwritten" >&2
    exit 1
fi

echo "per-role Codex custom agents: PASS"
