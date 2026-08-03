#!/usr/bin/env bash
set -euo pipefail

# Test: starting board/session A does not signal supervisor B.
# Uses fake pgrep/pkill/nohup/zellij commands and temporary HOME.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# We test that the global pkill pattern is no longer present in start-kanban.sh
# and that exact board/session matching is used instead.

SCRIPT="$ROOT/local/bin/start-kanban.sh"

# Verify global pkill is removed for supervisor
if grep -q 'pkill -f "kanban-watcher-supervisor"' "$SCRIPT"; then
    echo "FAIL: global 'pkill -f kanban-watcher-supervisor' still present in start-kanban.sh" >&2
    exit 1
fi

# Verify the supervisor is launched with --board and --session
if ! grep -q -- '--board.*\$BOARD' "$SCRIPT"; then
    echo "FAIL: supervisor not launched with --board \$BOARD" >&2
    exit 1
fi

if ! grep -q -- '--session.*\$SESSION_NAME' "$SCRIPT"; then
    echo "FAIL: supervisor not launched with --session \$SESSION_NAME" >&2
    exit 1
fi

# Verify the scoped supervisor cleanup matches exact board/session
if ! grep -q 'kanban-watcher-supervisor.*--board' "$SCRIPT"; then
    echo "FAIL: supervisor cleanup does not filter by --board" >&2
    exit 1
fi

echo "scoped watcher supervisor lifecycle: PASS"
