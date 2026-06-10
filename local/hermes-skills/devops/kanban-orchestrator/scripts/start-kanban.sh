#!/usr/bin/env bash
# Canonical launcher wrapper.
# The maintained script lives at /home/wyr/bin/start-kanban.sh and supports:
#   start-kanban.sh -b <board> [-n]
# Only inject mode is supported. Self-poll and worker modes removed 2026-06-06.
# Do NOT pass --task-delivery to Python interactive scripts — they no longer accept it.
set -euo pipefail
exec /home/wyr/bin/start-kanban.sh "$@"
