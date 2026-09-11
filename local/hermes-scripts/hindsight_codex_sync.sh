#!/usr/bin/env bash
# Nightly codex rollout -> hermes session snapshot sync feed for the offline
# Hindsight pipeline (cron job: hindsight-codex-sync, 23:00).
#
# Converts new/grown codex CLI rollouts (direct ~/.codex/sessions + kanban
# reviewer ~/.codex-kanban/reviewer/sessions) into session_*.json snapshots so
# the 00:01 offline pipeline picks them up the same night (snapshots must be
# >=15 min old before the manifest scan). Idempotent: up-to-date snapshots are
# skipped; growing/live sessions refresh their snapshot on later runs.
# implementer home is intentionally excluded (user decision).
# See hermes-daily-report skill 16.3.
set -euo pipefail
exec /home/wyr/.hermes/hermes-agent/venv/bin/python \
  /home/wyr/.hermes/scripts/hindsight_backfill_codex_rollouts.py --write
