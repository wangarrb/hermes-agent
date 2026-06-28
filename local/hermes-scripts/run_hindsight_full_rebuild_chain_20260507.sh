#!/usr/bin/env bash
set -euo pipefail

RETAIN_PID="${1:?retain pid required}"
RETAIN_LOG="${2:?retain log required}"
LOG_DIR="/home/wyr/.hermes/logs/hindsight-full-rebuild"
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d-%H%M%S)
CHAIN_LOG="$LOG_DIR/${TS}-full-chain.log"
export HINDSIGHT_OFFLINE_LLM_CONCURRENCY=4

log() {
  echo "[$(date -Is)] $*" | tee -a "$CHAIN_LOG"
}

status_json() {
  python3 - <<'PY'
import json, requests
api='http://127.0.0.1:8888'
health=requests.get(api+'/health', timeout=10).json()
stats=requests.get(api+'/v1/default/banks/hermes/stats', timeout=30).json()
print(json.dumps({
  'health': health,
  'pending_operations': stats.get('pending_operations'),
  'failed_operations': stats.get('failed_operations'),
  'total_documents': stats.get('total_documents'),
  'total_nodes': stats.get('total_nodes'),
  'total_observations': stats.get('total_observations'),
  'operations_by_status': stats.get('operations_by_status'),
}, ensure_ascii=False))
PY
}

log "chain_start retain_pid=$RETAIN_PID retain_log=$RETAIN_LOG chain_log=$CHAIN_LOG concurrency=$HINDSIGHT_OFFLINE_LLM_CONCURRENCY"
log "waiting_for_full_retain_process"
while kill -0 "$RETAIN_PID" 2>/dev/null; do
  if (( $(date +%s) % 300 < 60 )); then
    status_json 2>/dev/null | tee -a "$CHAIN_LOG" || true
  fi
  sleep 60
done
log "full_retain_process_exited"

if ! grep -q "queue drained" "$RETAIN_LOG"; then
  log "ERROR retain log missing 'queue drained'; refusing to continue"
  exit 2
fi
if ! grep -q "mode=normal-local" "$RETAIN_LOG"; then
  log "ERROR retain log missing normal-local restore; refusing to continue"
  exit 2
fi

STATUS=$(status_json)
log "post_retain_status=$STATUS"
python3 - <<'PY' <<<"$STATUS"
import json, sys
s=json.load(sys.stdin)
if int(s.get('failed_operations') or 0) != 0:
    raise SystemExit('failed_operations nonzero after retain')
if int(s.get('pending_operations') or 0) != 0:
    raise SystemExit('pending_operations nonzero after retain')
PY

log "starting_offline_reflect_consolidation_and_v2_publish"
python3 /home/wyr/.hermes/scripts/hindsight_offline_cron_runner.py both \
  --llm-profile minimax \
  --date-mode auto \
  --week-mode current \
  --prefilter safe \
  --poll 60 \
  --timeout 0 \
  --lock-timeout 21600 \
  --weekly-budget-max-pending-units 9999 \
  --weekly-budget-max-pending-chars 999999999 \
  2>&1 | tee -a "$CHAIN_LOG"

STATUS=$(status_json)
log "post_reflect_status=$STATUS"
python3 - <<'PY' <<<"$STATUS"
import json, sys
s=json.load(sys.stdin)
if int(s.get('failed_operations') or 0) != 0:
    raise SystemExit('failed_operations nonzero after reflect/v2')
if int(s.get('pending_operations') or 0) != 0:
    raise SystemExit('pending_operations nonzero after reflect/v2')
PY

log "starting_wiki_auto_maintenance"
python3 /home/wyr/.hermes/scripts/wiki_auto_maintenance.py --days 30 2>&1 | tee -a "$CHAIN_LOG"

STATUS=$(status_json)
log "final_status=$STATUS"
log "chain_done"
