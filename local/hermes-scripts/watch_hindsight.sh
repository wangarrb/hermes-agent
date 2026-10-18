#!/bin/bash
# Hindsight pipeline monitor — concise, runs inline

while true; do
  NOW=$(date +%H:%M:%S)
  LOG=$(ls -t /home/wyr/.hermes/logs/hindsight-offline-pipeline/*daily-noagent* 2>/dev/null | head -1)
  LOGSIZE=$(stat --format=%s "$LOG" 2>/dev/null || echo 0)

  # Last step from log
  STEP=$(tail -5 "$LOG" 2>/dev/null | grep -oP '(RUN|queue drained|health OK|completed|Exit code:)' | tail -1)

  # Hindsight stats
  STATS=$(python3 -c "
import json, requests as r
try:
    d = r.get('http://127.0.0.1:8888/v1/default/banks/hermes/stats', timeout=5).json()
    f=d.get('operations_by_status',{})
    print(f'docs={d[\"total_documents\"]} nodes={d[\"total_nodes\"]} obs={d[\"total_observations\"]} pend={d.get(\"pending_operations\",0)} comp={f.get(\"completed\",0)} fail={f.get(\"failed\",0)}')
except Exception as e:
    print(f'err={e}')
" 2>/dev/null)

  echo "[$NOW] ${STEP:-waiting} log=${LOGSIZE}B | $STATS"
  sleep 30
done
