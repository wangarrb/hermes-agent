#!/bin/bash
# Kanban worker auto-restart wrapper for zellij panes.
# Usage in zellij pane:  hermes-kanban-worker implementer
# Ctrl+C exits hermes, then auto-restarts with /listen-kanban

PROFILE="${1:-implementer}"
BOARD="${2:-egomotion4d}"

echo "=== Kanban Worker: $PROFILE ==="
echo "Ctrl+C to stop hermes (auto-restarts with /listen-kanban)"
echo "Ctrl+C twice rapidly within 2s to fully exit"
echo ""

while true; do
    # Run hermes chat. On Ctrl+C (exit 130), the while loop continues.
    # Run with expect-like auto-send of /listen-kanban after startup
    (
        # Wait for hermes to start, then inject /listen-kanban
        sleep 3
        echo "/listen-kanban --board $BOARD"
    ) | hermes -p "$PROFILE" chat 2>/dev/null

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 143 ]; then
        echo ""
        echo "=== Restarting $PROFILE (Ctrl+C again within 2s to stop) ==="
        # Quick double-Ctrl+C detection
        read -t 2 -r dummy 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "=== Stopped ==="
            exit 0
        fi
    elif [ $EXIT_CODE -eq 0 ]; then
        echo "=== Hermes exited cleanly, restarting ==="
        sleep 1
    else
        echo "=== Hermes crashed (exit=$EXIT_CODE), restarting in 3s ==="
        sleep 3
    fi
done