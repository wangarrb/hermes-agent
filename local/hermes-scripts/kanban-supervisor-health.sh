#!/bin/bash
# Kanban watcher supervisor health check & auto-restart
# Called by cron every 5 minutes — silent when healthy

set -euo pipefail

SOURCE_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "$SOURCE_PATH")" && pwd)"
SUPERVISOR_SCRIPT="$SCRIPT_DIR/../bin/kanban-watcher-supervisor.py"
LOG_DIR="$HOME/.hermes/hermes-agent/kanban_logs/egomotion4d"
STDERR_LOG="$LOG_DIR/watcher-supervisor-stderr.log"
SUPERVISOR_LOG="$LOG_DIR/watcher-supervisor.log"

# Check if supervisor is alive
if pgrep -f "$SUPERVISOR_SCRIPT" >/dev/null 2>&1; then
    # Healthy — silent exit
    exit 0
fi

# Dead — restart
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
echo "[$TIMESTAMP] supervisor not found, restarting..." >> "$SUPERVISOR_LOG"
nohup python3 "$SUPERVISOR_SCRIPT" --session kanban-egomotion4d --poll-s 30 \
    > "$STDERR_LOG" 2>&1 &
echo "[$TIMESTAMP] restarted: pid=$!" >> "$SUPERVISOR_LOG"
