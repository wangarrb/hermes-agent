#!/usr/bin/env bash
# Canonical stop wrapper.
# The maintained script lives at /home/wyr/bin/stop-kanban.sh and supports:
#   stop-kanban.sh [-f] [-n] [-s <session>]
# It handles Hermes profile panes plus Codex planner listener/children.
set -euo pipefail
exec /home/wyr/bin/stop-kanban.sh "$@"
