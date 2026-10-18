#!/bin/bash
# Daily Hindsight pipeline runner (detached from crontab)
set -euo pipefail

LOG_DIR="/home/wyr/.hermes/logs/hindsight-offline-pipeline"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="${LOG_DIR}/${TIMESTAMP}-daily-noagent.log"

cd /home/wyr/.hermes

/home/wyr/.hermes/hermes-agent/venv/bin/python \
  /home/wyr/.hermes/scripts/hindsight_memory_pipeline.py \
  daily \
  --config /home/wyr/.hermes/hindsight/pipeline_config.json \
  --execute \
  --confirm run-hindsight-pipeline \
  >> "$LOG_FILE" 2>&1

echo "Pipeline exit code: $?" >> "$LOG_FILE"
