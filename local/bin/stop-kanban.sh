#!/bin/bash
# 安全停止 kanban zellij session：停止所有 kanban worker 进程，再 kill session
# 用法: stop-kanban.sh [-f] [-n] [-s <session>] [-b <board>]
#   -f  强制：直接 SIGKILL kanban worker，再 kill session
#   -n  dry-run：只显示将要停止的 session / 进程，不执行 kill
#   -s  指定 session 名（默认自动检测当前 kanban session）
#   -b  指定 board 名（用于精确匹配 worker 进程）

set -euo pipefail

FORCE=0
DRY_RUN=0
SESSION=""
BOARD=""

usage() {
    echo "用法: $0 [-f] [-n] [-s <session>] [-b <board>]"
    echo "  -f  强制 kill，不等待 SIGTERM"
    echo "  -n  dry-run，只显示将处理的 session / 进程"
    echo "  -s  指定 zellij session 名"
    echo "  -b  指定 kanban board 名（用于精确匹配 worker 进程）"
    echo ""
    echo "自动检测包含 Hermes/Codex/CodeWhale/Claude/Reasonix kanban listener 的 session/进程。"
}

while getopts "fns:b:h" opt; do
    case $opt in
        f) FORCE=1 ;;
        n) DRY_RUN=1 ;;
        s) SESSION="$OPTARG" ;;
        b) BOARD="$OPTARG" ;;
        h) usage; exit 0 ;;
        *) echo "未知选项"; usage; exit 1 ;;
    esac
done

clean_session_name() {
    sed 's/ (current)//; s/ \[Created.*//; s/ (EXITED.*//'
}

list_sessions() {
    zellij list-sessions --no-formatting --short 2>/dev/null || true
}

# Use the same kanban_pid_lines as start-kanban.sh — matches all 5 agent types
kanban_pid_lines() {
    BOARD_FILTER="$BOARD" python3 - <<'PY'
from __future__ import annotations
import os
import re

board = os.environ.get("BOARD_FILTER", "")
self_pid = os.getpid()

roles = {"coordinator", "planner", "implementer", "critic", "reviewer"}
hermes_re = re.compile(r"\bhermes\b.*\s-p\s+(coordinator|planner|implementer|critic|reviewer)\b")
listener_re = re.compile(r"(codex-kanban-interactive|codex_kanban_interactive\.py|codex-kanban-listen|codex_kanban_listener\.py|codewhale-kanban-interactive|codewhale_kanban_interactive\.py|deepseek-kanban-interactive|deepseek_kanban_interactive\.py|deepseek-kanban-listen|deepseek_kanban_listener\.py|reasonix-kanban-interactive|reasonix_kanban_interactive\.py|claude-kanban-interactive|claude_kanban_interactive\.py)")
board_arg_re = re.compile(r"(?:--board(?:=|\s+)|HERMES_KANBAN_BOARD=)" + re.escape(board) + r"(?=\s|$)") if board else None

def read_cmd(pid: int) -> str:
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
    except OSError:
        return ""
    return " ".join(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)

def read_env(pid: int) -> dict[str, str]:
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read()
    except OSError:
        return {}
    env: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if not item or b"=" not in item:
            continue
        k, v = item.split(b"=", 1)
        try:
            env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
        except Exception:
            pass
    return env

def board_matches(cmd: str, env: dict[str, str]) -> bool:
    if env.get("HERMES_KANBAN_BOARD") == board:
        return True
    return bool(board_arg_re and board_arg_re.search(cmd))

rows: list[tuple[int, str, str]] = []
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    pid = int(name)
    if pid == self_pid:
        continue
    cmd = read_cmd(pid)
    if not cmd:
        continue
    if "start-kanban.sh" in cmd or "stop-kanban.sh" in cmd:
        continue
    env: dict[str, str] | None = None
    reason = ""
    if hermes_re.search(cmd):
        env = read_env(pid)
        if board_matches(cmd, env):
            reason = "hermes-kanban-profile"
    elif listener_re.search(cmd):
        env = read_env(pid)
        if board_matches(cmd, env):
            reason = "kanban-listener"
    else:
        env = read_env(pid)
        profile = env.get("HERMES_KANBAN_PROFILE", "")
        if profile in roles and board_matches(cmd, env):
            reason = "kanban-env-child"
    if reason:
        rows.append((pid, reason, cmd[:260]))

for pid, reason, cmd in sorted(rows):
    print(f"{pid}\t{reason}\t{cmd}")
PY
}

kanban_pid_count() {
    kanban_pid_lines | awk 'NF {n++} END {print n+0}'
}

detect_session() {
    local sessions_raw current live_count live_name total_count only_name
    sessions_raw=$(zellij list-sessions --no-formatting 2>/dev/null || true)

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

    # 3. If exactly one live (non-EXITED) session exists, use it.
    live_count=$(echo "$sessions_raw" | grep -v 'EXITED' | sed '/^$/d' | wc -l)
    if [ "$live_count" -eq 1 ]; then
        live_name=$(echo "$sessions_raw" | grep -v 'EXITED' | head -n 1 | clean_session_name)
        echo "$live_name"
        return
    fi

    # 4. Fallback: if exactly one session total exists, use it.
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
        echo "请用 -s <session> 或 -b <board> 指定。"
        exit 1
    fi
fi

echo "目标 session: $SESSION"
[ -n "$BOARD" ] && echo "目标 board:   $BOARD"

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

# --- Kill worker processes ---
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

# --- Kill target session ---
echo ""
echo "Kill zellij session $SESSION..."
zellij kill-session "$SESSION" 2>/dev/null || true
zellij delete-session "$SESSION" 2>/dev/null || true
echo "Session $SESSION 已停止"

# --- Final confirmation ---
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
