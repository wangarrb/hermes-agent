#!/bin/bash
# Kanban worker auto-restart wrapper for zellij panes.
# Usage in zellij pane:
#   hermes-kanban-worker implementer
#   hermes-kanban-worker critic
#   hermes-kanban-worker planner
#
# On Ctrl+C (exit code 130/143), auto-restarts hermes + /listen-kanban.
# On explicit "quit" or "exit" (exit code 0), stops.
# On crash/error, restart with delay.

PROFILE="${1:-implementer}"
BOARD="${2:-egomotion4d}"

echo "[kanban-worker] Starting profile=$PROFILE board=$BOARD"
echo "[kanban-worker] Ctrl+C to stop; auto-restarts on crash"

while true; do
    echo "[kanban-worker] $(date '+%H:%M:%S') launching hermes -p $PROFILE chat --listen-kanban ..."

    # Run hermes with listen-kanban auto-command
    # --listen-kanban flag doesn't exist in hermes CLI, so we use a subshell approach:
    # We pipe the '/listen-kanban' command into hermes after a short delay
    (
        sleep 2
        echo "/listen-kanban --board $BOARD"
    ) | hermes -p "$PROFILE" chat &

    HERMES_PID=$!
    wait $HERMES_PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ] || [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 143 ]; then
        echo "[kanban-worker] $(date '+%H:%M:%S') Stopped by user (exit=$EXIT_CODE)"
        echo "[kanban-worker] Press Enter to restart, or Ctrl+C again to exit"
        read -t 1 -r 2>/dev/null || true
        sleep 0.5
        continue
    else
        echo "[kanban-worker] $(date '+%H:%M:%S') Crashed (exit=$EXIT_CODE), restarting in 3s..."
        sleep 3
        continue
    fi
done