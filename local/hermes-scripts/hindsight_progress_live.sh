#!/usr/bin/env bash
set -u
INTERVAL="${1:-15}"
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
SCRIPT="${HINDSIGHT_PROGRESS_BAR:-$HERMES_HOME/scripts/hindsight_progress_bar.py}"

echo "Hindsight offline progress monitor"
echo "interval=${INTERVAL}s"
echo "script=${SCRIPT}"
sleep 1

while true; do
  clear 2>/dev/null || printf '\033c'
  echo "Hindsight offline progress monitor"
  echo "刷新时间: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "----------------------------------------"
  python3 "$SCRIPT"
  echo "----------------------------------------"
  if ! pgrep -af 'hindsight_minimax_import.py offline-reflect-llm|offline_hindsight_reflect_consolidate.py' | grep -vE 'pgrep|hindsight_progress' >/dev/null; then
    echo "后台离线处理进程已结束或未找到。"
    echo "可以让我做最终核验。"
    break
  fi
  echo "每 ${INTERVAL}s 自动刷新；Ctrl-C 退出监控，不影响 Hindsight 处理。"
  sleep "$INTERVAL"
done

echo
read -r -p "按 Enter 保留/退出这个 shell... " _ || true
