#!/bin/bash
# Weekly Hindsight wiki maintenance runner
set -euo pipefail

LOG_DIR="/home/wyr/.hermes/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/hindsight-wiki-${TIMESTAMP}.log"

cd /home/wyr/.hermes

/home/wyr/.hermes/hermes-agent/venv/bin/python \
  /home/wyr/.hermes/scripts/wiki_auto_maintenance_cron_runner.py \
  >> "$LOG_FILE" 2>&1

echo "Wiki maintenance exit code: $?" >> "$LOG_FILE"
