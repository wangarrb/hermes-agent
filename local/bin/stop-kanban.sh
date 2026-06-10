#!/bin/bash
# 安全停止 kanban zellij session：停止 Hermes/Codex kanban worker，再 kill session
# 用法: stop-kanban.sh [-f] [-n] [-s <session>]
#   -f  强制：直接 SIGKILL kanban worker，再 kill session
#   -n  dry-run：只显示将要停止的 session / 进程，不执行 kill
#   -s  指定 session 名（默认自动检测当前 kanban session）
# 示例:
#   stop-kanban.sh              # 优雅停止当前 kanban session
#   stop-kanban.sh -f           # 强制 kill
#   stop-kanban.sh -n           # 只检查不停止
#   stop-kanban.sh -s old-sess  # 停止指定 session

set -euo pipefail

FORCE=0
DRY_RUN=0
SESSION=""

usage() {
    echo "用法: $0 [-f] [-n] [-s <session>]"
    echo "  -f  强制 kill，不等待 SIGTERM"
    echo "  -n  dry-run，只显示将处理的 session / 进程"
    echo "  -s  指定 zellij session 名"
    echo ""
    echo "自动检测包含 Hermes coordinator/implementer/critic 或 Codex planner kanban listener 的 session/进程。"
}

while getopts "fns:h" opt; do
    case $opt in
        f) FORCE=1 ;;
        n) DRY_RUN=1 ;;
        s) SESSION="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) echo "未知选项"; usage; exit 1 ;;
    esac
done

list_sessions_raw() {
    zellij list-sessions --no-formatting 2>/dev/null || true
}

list_session_names() {
    zellij list-sessions --no-formatting --short 2>/dev/null || true
}

list_sessions() {
    list_session_names
}

clean_session_name() {
    # Strip zellij decorations like " (current)" and " (EXITED - attach to resurrect)".
    sed 's/ (current)//; s/ \[Created.*//; s/ (EXITED.*//'
}

kanban_pid_lines() {
    # Print: PID<TAB>reason<TAB>cmd
    # Includes:
    # - Hermes visible panes: hermes -p coordinator/planner/implementer/critic
    # - Codex Kanban listener wrapper/python, including interactive bridge
    # - Codex child processes spawned by the listener, detected via HERMES_KANBAN_TASK/PROFILE env
    python3 - <<'PY'
from __future__ import annotations
import os
import re

hermes_re = re.compile(r"\bhermes\b.*\s-p\s+(coordinator|planner|implementer|critic)\b")
codex_listener_re = re.compile(r"(codex-kanban-listen|codex_kanban_listener\.py|codex-kanban-interactive|codex_kanban_interactive\.py)")
deepseek_listener_re = re.compile(r"(codewhale-kanban-interactive|codewhale_kanban_interactive\.py|deepseek-kanban-interactive|deepseek_kanban_interactive\.py|deepseek-kanban-listen|deepseek_kanban_listener\.py)")
self_pid = os.getpid()
out = []
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == self_pid:
        continue
    proc = f'/proc/{name}'
    try:
        raw_cmd = open(f'{proc}/cmdline', 'rb').read()
    except OSError:
        continue
    cmd = ' '.join(part.decode('utf-8', 'replace') for part in raw_cmd.split(b'\0') if part)
    if not cmd:
        continue
    reason = ''
    if hermes_re.search(cmd):
        reason = 'hermes-kanban-profile'
    elif codex_listener_re.search(cmd):
        reason = 'codex-kanban-listener'
    elif deepseek_listener_re.search(cmd):
        reason = 'deepseek-kanban-listener'
    else:
        try:
            env = open(f'{proc}/environ', 'rb').read()
        except OSError:
            env = b''
        if b'HERMES_KANBAN_TASK=' in env or b'HERMES_KANBAN_PROFILE=' in env:
            reason = 'kanban-env-child'
    if reason:
        out.append((pid, reason, cmd[:260]))
for pid, reason, cmd in sorted(out):
    print(f'{pid}\t{reason}\t{cmd}')
PY
}

kanban_pid_count() {
    kanban_pid_lines | awk 'NF {n++} END {print n+0}'
}

detect_session() {
    local sessions_raw current live_count live_name total_count only_name
    sessions_raw=$(list_sessions_raw)

    # 1. If running inside a zellij session, prefer the current session.
    current=$(echo "$sessions_raw" | grep '(current)' | head -n 1 | clean_session_name) || true
    if [ -n "$current" ]; then
        echo "$current"
        return
    fi

    # 2. If ZELLIJ_SESSION_NAME is available, use it.
    if [ -n "${ZELLIJ_SESSION_NAME:-}" ]; then
        echo "$ZELLIJ_SESSION_NAME"
        return
    fi

    # 3. Prefer a uniquely named kanban session created by start-kanban.sh.
    live_kanban=$(echo "$sessions_raw" | grep -v 'EXITED' | clean_session_name | grep '^kanban-' || true)
    live_count=$(echo "$live_kanban" | sed '/^$/d' | wc -l)
    if [ "$live_count" -eq 1 ]; then
        echo "$live_kanban" | head -n 1
        return
    fi

    # 4. If exactly one live (non-EXITED) session exists, use it.
    live_count=$(echo "$sessions_raw" | grep -v 'EXITED' | sed '/^$/d' | wc -l)
    if [ "$live_count" -eq 1 ]; then
        live_name=$(echo "$sessions_raw" | grep -v 'EXITED' | head -n 1 | clean_session_name)
        echo "$live_name"
        return
    fi

    # 5. Fallback: if exactly one session total exists, use it.
    total_count=$(echo "$sessions_raw" | sed '/^$/d' | wc -l)
    if [ "$total_count" -eq 1 ]; then
        only_name=$(echo "$sessions_raw" | head -n 1 | clean_session_name)
        echo "$only_name"
        return
    fi

    echo ""
}

if [ -z "$SESSION" ]; then
    SESSION=$(detect_session)
    if [ -z "$SESSION" ]; then
        echo "未检测到唯一 kanban zellij session"
        echo "当前 sessions:"
        list_sessions | sed 's/^/  /' || echo "  (无)"
        echo ""
        echo "当前 kanban worker 进程:"
        kanban_pid_lines | sed 's/^/  /' || true
        echo ""
        echo "请用 -s <session> 指定，或先确认 zellij 正在运行。"
        exit 1
    fi
fi

echo "目标 session: $SESSION"

echo ""
echo "=== 当前 kanban worker 进程 ==="
pids_snapshot=$(kanban_pid_lines || true)
if [ -n "$pids_snapshot" ]; then
    echo "$pids_snapshot" | sed 's/^/  /'
else
    echo "  无"
fi

all_sessions=$(list_sessions | clean_session_name | sed '/^$/d')

echo ""
echo "=== 当前 zellij sessions ==="
if [ -n "$all_sessions" ]; then
    echo "$all_sessions" | sed 's/^/  /'
else
    echo "  无"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "dry-run: 不执行 kill。"
    exit 0
fi

# --- 不再清理非目标 session ---
echo ""
echo "=== 保留其他 zellij sessions ==="
echo "  stop-kanban.sh 现在只停止目标 session: $SESSION"
echo "  其他 zellij session 不会被 kill。"

# --- 停止 worker 进程 ---
echo ""
if [ "$FORCE" -eq 1 ]; then
    echo "强制模式：SIGKILL 所有 kanban worker 进程"
    sig="KILL"
else
    echo "优雅停止：SIGTERM 所有 kanban worker 进程"
    sig="TERM"
fi

killed_pids=0
while IFS=$'\t' read -r pid reason cmd; do
    [ -n "${pid:-}" ] || continue
    echo "  kill -$sig -> PID $pid [$reason] $cmd"
    kill "-$sig" "$pid" 2>/dev/null || true
    killed_pids=$((killed_pids + 1))
done <<< "$(kanban_pid_lines || true)"

if [ "$killed_pids" -eq 0 ]; then
    echo "  没有找到运行中的 kanban worker 进程"
elif [ "$FORCE" -eq 0 ]; then
    echo "  等待进程退出 (最多 10s)..."
    waited=0
    while [ "$waited" -lt 10 ]; do
        remaining=$(kanban_pid_count)
        if [ "$remaining" -le 0 ]; then
            echo "  所有 kanban worker 进程已退出"
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    if [ "$waited" -ge 10 ]; then
        echo "  超时，SIGKILL 剩余 kanban worker 进程..."
        while IFS=$'\t' read -r pid reason cmd; do
            [ -n "${pid:-}" ] || continue
            echo "    kill -KILL -> PID $pid [$reason] $cmd"
            kill -KILL "$pid" 2>/dev/null || true
        done <<< "$(kanban_pid_lines || true)"
    fi
fi

# --- Kill 目标 session ---
echo ""
echo "Kill zellij session $SESSION..."
zellij kill-session "$SESSION" 2>/dev/null || true
# kill-session 只杀进程，session 变 EXITED 但仍占名；delete-session 彻底删除
zellij delete-session "$SESSION" 2>/dev/null || true
echo "Session $SESSION 已停止并删除"

# --- 最终确认 ---
echo ""
echo "=== 清理确认 ==="
remaining=$(kanban_pid_count)
if [ "$remaining" -le 0 ]; then
    echo "OK: 无残留 kanban worker 进程"
else
    echo "WARNING: 仍有 $remaining 个 kanban worker 进程残留"
    kanban_pid_lines | sed 's/^/  /'
fi

sessions=$(list_sessions)
if [ -z "$sessions" ]; then
    echo "OK: 无残留 zellij session"
else
    echo "剩余 zellij sessions:"
    echo "$sessions" | sed 's/^/  /'
fi

echo ""
echo "完成。下次启动: start-kanban.sh -b <board>"
