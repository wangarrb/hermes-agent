#!/bin/bash
# 启动 kanban 4分窗口。角色固定，底层 agent 可切换。
# 默认布局：coordinator=hermes, planner=codex, implementer=codewhale, critic=codewhale
# 默认任务交付：inject（通过 Zellij 注入）；self-poll 已移除
# 双角色用法：--assist-role coordinator:implementer --assist-role critic:implementer

set -euo pipefail

# 坑63: Hermes profile 下 HOME 指向虚拟目录
export REAL_HOME="/home/wyr"
# 不要 export HOME=/home/wyr — zellij pane 启动的 hermes -p 需要 profile 的虚拟 HOME

usage() {
    local status="${1:-1}"
    cat <<'EOF'
用法:
  start-kanban.sh -b <board> [options]

核心参数:
  -b, --board <board>              Kanban board 名称，例如 egomotion4d
  -w, --workspace <path>           工作目录，默认 $HOME/code/Egomotion4D
  -n, --dry-run                    只生成并打印 zellij layout，不启动/不清理

角色 -> agent 映射（默认就是当前常用配置）:
  -o, --coordinator-agent <agent>  coordinator 使用的 agent，默认 hermes
  -p, --planner-agent <agent>      planner 使用的 agent，默认 codex
  -i, --implementer-agent <agent>  implementer 使用的 agent，默认 codewhale
  -c, --critic-agent <agent>       critic 使用的 agent，默认 codewhale

支持的 agent:
  hermes
  codex          （interactive Codex + Kanban watcher）
  codewhale      （interactive CodeWhale + Kanban watcher，原 deepseek-tui）
  deepseek-reasonix （interactive Reasonix + Kanban watcher）

Agent 参数:
  --codex-model <model>            可选 Codex model override
  --codex-sandbox <mode>           Codex sandbox，默认 danger-full-access
  --deepseek-provider <provider>   CodeWhale/DeepSeek provider，默认 openrouter；可用 opencode-go
  --deepseek-model <model>         可选 CodeWhale/DeepSeek model override；不填时由 bridge 按 provider 选择
  --deepseek-continue-policy <p>   CodeWhale 会话续接策略：auto/primary-only/all/none，默认 auto。
                                   auto/primary-only: 只有一个 codewhale pane 时继续旧会话；
                                   多个 codewhale pane 时仅 primary 继续，其他 fresh。
  --deepseek-continue-primary <r>  覆盖 continue primary 角色（默认自动选 implementer > planner > coordinator > critic）。
                                   设为 critic 可让 critic pane 继续旧会话而非 implementer。
  --idle-pane-reclaim-s <sec>      CodeWhale pane 连续空闲多久后回收 running 任务；默认 bridge=600
  --task-delivery <mode>           （已废弃）仅保留 inject 模式；self-poll/worker 已移除。
  --assist-claim-delay-s <sec>     辅助 assignee 的 ready 任务等待多久后才允许被本 pane claim
  --assist-role-delay <spec>       控制某个 profile 辅助 claim 的延时；支持 profile:assignee:sec、
                                   profile:sec、:sec、sec。省略时默认 profile/assignee=implementer
  --assist-profile-delay <spec>    同上，但总是作为全局 profile 规则下发，适合 backup_immplementer
  --assist-role <role:assignee>    让某个 pane 空闲时辅助 claim 指定 assignee 的 ready 任务；
                                   例如 planner:implementer。可重复。主角色优先。

Zellij:
  --session-name <name>            新 session 名，默认 kanban-<board>
  --no-clean                       启动前不清理同名 session/同 board worker（可能造成抢任务，仅调试用）

示例:
  start-kanban.sh -b egomotion4d
  start-kanban.sh -b egomotion4d -i codewhale -p codex -c hermes
  start-kanban.sh -b egomotion4d -p hermes -i codex -c codewhale -n
  start-kanban.sh -b egomotion4d --deepseek-provider opencode-go --deepseek-model deepseek-v4-pro
EOF
    exit "$status"
}

need_value() {
    local opt="$1"
    local val="${2:-}"
    if [ -z "$val" ]; then
        echo "错误: $opt 需要一个参数" >&2
        exit 1
    fi
}

normalize_agent() {
    local raw="${1:-}"
    raw="${raw,,}"
    raw="${raw//_/-}"
    case "$raw" in
        hermes|h)
            echo "hermes" ;;
        codex|codex-tui|codex-interactive)
            echo "codex" ;;
        codewhale)
            echo "codewhale" ;;
        reasonix|deepseek-reasonix)
            echo "deepseek-reasonix" ;;
        *)
            echo "错误: 不支持的 agent: ${1:-<empty>}；支持 hermes/codex/codewhale/deepseek-reasonix" >&2
            return 1 ;;
    esac
}

agent_is_used() {
    local target="$1"
    [ "$COORDINATOR_AGENT" = "$target" ] || \
    [ "$PLANNER_AGENT" = "$target" ] || \
    [ "$IMPLEMENTER_AGENT" = "$target" ] || \
    [ "$CRITIC_AGENT" = "$target" ]
}

shell_quote() {
    printf '%q' "$1"
}

kdl_string() {
    python3 - "$1" <<'PY'
import json
import sys
print(json.dumps(sys.argv[1], ensure_ascii=False))
PY
}

clean_session_name() {
    sed 's/ (current)//; s/ \[Created.*//; s/ (EXITED.*//'
}

BOARD=""
WORKSPACE="${KANBAN_WORKSPACE:-${CODEWHALE_KANBAN_WORKSPACE:-${CODEX_KANBAN_WORKSPACE:-${DEEPSEEK_KANBAN_WORKSPACE:-${REAL_HOME}/code/Egomotion4D}}}}"
COORDINATOR_AGENT="${KANBAN_COORDINATOR_AGENT:-hermes}"
PLANNER_AGENT="${KANBAN_PLANNER_AGENT:-codex}"
IMPLEMENTER_AGENT="${KANBAN_IMPLEMENTER_AGENT:-codewhale}"
CRITIC_AGENT="${KANBAN_CRITIC_AGENT:-codewhale}"
DEEPSEEK_PROVIDER="${DEEPSEEK_KANBAN_PROVIDER:-openrouter}"
DEEPSEEK_MODEL="${DEEPSEEK_KANBAN_MODEL:-}"
DEEPSEEK_CONTINUE_POLICY="${DEEPSEEK_KANBAN_CONTINUE_POLICY:-auto}"
DEEPSEEK_CONTINUE_PRIMARY="${DEEPSEEK_KANBAN_CONTINUE_PRIMARY:-}"
DEEPSEEK_TASK_TIMEOUT="${DEEPSEEK_KANBAN_TASK_TIMEOUT:-}"
DEEPSEEK_IDLE_PANE_RECLAIM="${DEEPSEEK_KANBAN_IDLE_PANE_RECLAIM:-}"
TASK_DELIVERY="${KANBAN_TASK_DELIVERY:-inject}"
ASSIST_CLAIM_DELAY="${KANBAN_ASSIST_CLAIM_DELAY_S:-${HERMES_KANBAN_ASSIST_CLAIM_DELAY_S:-}}"
CODEX_MODEL="${CODEX_KANBAN_MODEL:-}"
CODEX_SANDBOX="${CODEX_KANBAN_SANDBOX:-danger-full-access}"
SESSION_NAME=""
DRY_RUN=0
CLEAN=1
COORDINATOR_ASSISTS=""
PLANNER_ASSISTS=""
IMPLEMENTER_ASSISTS=""
CRITIC_ASSISTS=""
COORDINATOR_ASSIST_DELAYS=""
PLANNER_ASSIST_DELAYS=""
IMPLEMENTER_ASSIST_DELAYS=""
CRITIC_ASSIST_DELAYS=""
GLOBAL_ASSIST_PROFILE_DELAYS=""

validate_role_name() {
    local kind="$1"
    local role="$2"
    case "$role" in
        coordinator|planner|implementer|critic)
            return 0 ;;
        *)
            echo "错误: $kind 只能是 coordinator/planner/implementer/critic，不能是: ${role:-<empty>}" >&2
            exit 1 ;;
    esac
}

append_csv_unique() {
    local current="$1"
    local item="$2"
    local part
    IFS=',' read -ra parts <<< "$current"
    for part in "${parts[@]}"; do
        if [ "$part" = "$item" ]; then
            printf '%s' "$current"
            return
        fi
    done
    if [ -z "$current" ]; then
        printf '%s' "$item"
    else
        printf '%s,%s' "$current" "$item"
    fi
}

append_assist_delay() {
    local current="$1"
    local spec="$2"
    local key="${spec%%=*}"
    local item
    local kept=""
    IFS=',' read -ra parts <<< "$current"
    for item in "${parts[@]}"; do
        [ -n "$item" ] || continue
        if [ "${item%%=*}" = "$key" ]; then
            continue
        fi
        kept="$(append_csv_unique "$kept" "$item")"
    done
    append_csv_unique "$kept" "$spec"
}

is_number() {
    [[ "${1:-}" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

normalize_assist_profile_delay_spec() {
    local spec="$1"
    local key value profile assignee extra
    local fields
    if [[ "$spec" == *"="* ]]; then
        key="${spec%=*}"
        value="${spec##*=}"
        IFS=':' read -r profile assignee extra <<< "$key"
    else
        IFS=':' read -ra fields <<< "$spec"
        case "${#fields[@]}" in
            1)
                profile="implementer"
                assignee="implementer"
                value="${fields[0]}"
                ;;
            2)
                profile="${fields[0]:-implementer}"
                assignee="implementer"
                value="${fields[1]}"
                ;;
            *)
                profile="${fields[0]:-implementer}"
                assignee="${fields[1]:-implementer}"
                value="${fields[2]}"
                ;;
        esac
    fi
    profile="${profile:-implementer}"
    assignee="${assignee:-implementer}"
    if ! is_number "$value"; then
        echo "错误: assist delay 秒数必须是数字: ${spec:-<empty>}" >&2
        exit 1
    fi
    if [[ ! "$profile" =~ ^[A-Za-z0-9._-]+$ ]] || [[ ! "$assignee" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "错误: assist delay profile/assignee 只能包含字母、数字、点、下划线和短横线: $spec" >&2
        exit 1
    fi
    printf '%s:%s=%s' "$profile" "$assignee" "$value"
}

claim_assignees_for_role() {
    local role="$1"
    local assists="$2"
    local out="$role"
    local item
    if [ -n "$assists" ]; then
        IFS=',' read -ra assist_parts <<< "$assists"
        for item in "${assist_parts[@]}"; do
            [ -n "$item" ] || continue
            out="$(append_csv_unique "$out" "$item")"
        done
    fi
    printf '%s' "$out"
}

add_assist_profile_delay() {
    local normalized="$1"
    GLOBAL_ASSIST_PROFILE_DELAYS="$(append_assist_delay "$GLOBAL_ASSIST_PROFILE_DELAYS" "$normalized")"
}

add_assist_role_delay() {
    local normalized profile assignee delay target_spec
    normalized="$(normalize_assist_profile_delay_spec "$1")"
    profile="${normalized%%:*}"
    assignee="${normalized#*:}"
    assignee="${assignee%%=*}"
    delay="${normalized##*=}"
    target_spec="${assignee}=${delay}"
    case "$profile" in
        coordinator)
            COORDINATOR_ASSIST_DELAYS="$(append_assist_delay "$COORDINATOR_ASSIST_DELAYS" "$target_spec")" ;;
        planner)
            PLANNER_ASSIST_DELAYS="$(append_assist_delay "$PLANNER_ASSIST_DELAYS" "$target_spec")" ;;
        implementer)
            IMPLEMENTER_ASSIST_DELAYS="$(append_assist_delay "$IMPLEMENTER_ASSIST_DELAYS" "$target_spec")" ;;
        critic)
            CRITIC_ASSIST_DELAYS="$(append_assist_delay "$CRITIC_ASSIST_DELAYS" "$target_spec")" ;;
        *)
            add_assist_profile_delay "$normalized" ;;
    esac
}

add_assist_role() {
    local spec="$1"
    local base="${spec%%:*}"
    local assist="${spec#*:}"
    if [ "$base" = "$spec" ] || [ -z "$base" ] || [ -z "$assist" ]; then
        echo "错误: --assist-role 需要 role:assignee，例如 planner:implementer" >&2
        exit 1
    fi
    validate_role_name "assist role 左侧 role" "$base"
    validate_role_name "assist role 右侧 assignee" "$assist"
    case "$base" in
        coordinator)
            COORDINATOR_ASSISTS="$(append_csv_unique "$COORDINATOR_ASSISTS" "$assist")" ;;
        planner)
            PLANNER_ASSISTS="$(append_csv_unique "$PLANNER_ASSISTS" "$assist")" ;;
        implementer)
            IMPLEMENTER_ASSISTS="$(append_csv_unique "$IMPLEMENTER_ASSISTS" "$assist")" ;;
        critic)
            CRITIC_ASSISTS="$(append_csv_unique "$CRITIC_ASSISTS" "$assist")" ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--board)
            need_value "$1" "${2:-}"; BOARD="$2"; shift 2 ;;
        -w|--workspace)
            need_value "$1" "${2:-}"; WORKSPACE="$2"; shift 2 ;;
        -o|--coordinator-agent)
            need_value "$1" "${2:-}"; COORDINATOR_AGENT="$2"; shift 2 ;;
        -p|--planner-agent)
            need_value "$1" "${2:-}"; PLANNER_AGENT="$2"; shift 2 ;;
        -i|--implementer-agent)
            need_value "$1" "${2:-}"; IMPLEMENTER_AGENT="$2"; shift 2 ;;
        -c|--critic-agent)
            need_value "$1" "${2:-}"; CRITIC_AGENT="$2"; shift 2 ;;
        --deepseek-provider)
            need_value "$1" "${2:-}"; DEEPSEEK_PROVIDER="$2"; shift 2 ;;
        --deepseek-model)
            need_value "$1" "${2:-}"; DEEPSEEK_MODEL="$2"; shift 2 ;;
        --deepseek-continue-policy)
            need_value "$1" "${2:-}"; DEEPSEEK_CONTINUE_POLICY="$2"; shift 2 ;;
        --deepseek-continue-primary)
            need_value "$1" "${2:-}"; DEEPSEEK_CONTINUE_PRIMARY="$2"; shift 2 ;;
        --task-timeout-s)
            need_value "$1" "${2:-}"; DEEPSEEK_TASK_TIMEOUT="$2"; shift 2 ;;
        --idle-pane-reclaim-s)
            need_value "$1" "${2:-}"; DEEPSEEK_IDLE_PANE_RECLAIM="$2"; shift 2 ;;
        --task-delivery)
            need_value "$1" "${2:-}"; TASK_DELIVERY="$2"; shift 2 ;;
        --assist-claim-delay-s)
            need_value "$1" "${2:-}"; ASSIST_CLAIM_DELAY="$2"; shift 2 ;;
        --assist-role-delay)
            need_value "$1" "${2:-}"; add_assist_role_delay "$2"; shift 2 ;;
        --assist-profile-delay)
            need_value "$1" "${2:-}"; add_assist_profile_delay "$(normalize_assist_profile_delay_spec "$2")"; shift 2 ;;
        --codex-model)
            need_value "$1" "${2:-}"; CODEX_MODEL="$2"; shift 2 ;;
        --codex-sandbox)
            need_value "$1" "${2:-}"; CODEX_SANDBOX="$2"; shift 2 ;;
        --assist-role|--extra-role)
            need_value "$1" "${2:-}"; add_assist_role "$2"; shift 2 ;;
        --planner-assist)
            need_value "$1" "${2:-}"; add_assist_role "planner:$2"; shift 2 ;;
        --critic-assist)
            need_value "$1" "${2:-}"; add_assist_role "critic:$2"; shift 2 ;;
        --session-name)
            need_value "$1" "${2:-}"; SESSION_NAME="$2"; shift 2 ;;
        --no-clean)
            CLEAN=0; shift ;;
        -n|--dry-run)
            DRY_RUN=1; shift ;;
        -h|--help)
            usage 0 ;;
        *)
            echo "错误: 未知参数 $1" >&2
            usage ;;
    esac
done

if [ -z "$BOARD" ]; then
    echo "错误: 必须指定 board 名称 (-b)" >&2
    usage
fi

if [[ ! "$BOARD" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "错误: board 名称只能包含字母、数字、点、下划线和短横线: $BOARD" >&2
    exit 1
fi

if [ -z "$SESSION_NAME" ]; then
    SESSION_NAME="kanban-${BOARD}"
fi
if [[ ! "$SESSION_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "错误: session 名称只能包含字母、数字、点、下划线和短横线: $SESSION_NAME" >&2
    exit 1
fi

COORDINATOR_AGENT="$(normalize_agent "$COORDINATOR_AGENT")"
PLANNER_AGENT="$(normalize_agent "$PLANNER_AGENT")"
IMPLEMENTER_AGENT="$(normalize_agent "$IMPLEMENTER_AGENT")"
CRITIC_AGENT="$(normalize_agent "$CRITIC_AGENT")"

DEEPSEEK_CONTINUE_POLICY="${DEEPSEEK_CONTINUE_POLICY,,}"
case "$DEEPSEEK_CONTINUE_POLICY" in
    auto|primary-only|all|none)
        ;;
    *)
        echo "错误: --deepseek-continue-policy 只能是 auto/primary-only/all/none，不能是: $DEEPSEEK_CONTINUE_POLICY" >&2
        exit 1
        ;;
esac

TASK_DELIVERY="${TASK_DELIVERY,,}"
case "$TASK_DELIVERY" in
    inject)
        ;;
    worker|self-poll)
        echo "注意: --task-delivery $TASK_DELIVERY 已废弃，强制使用 inject" >&2
        TASK_DELIVERY="inject"
        ;;
    *)
        echo "错误: --task-delivery 只能是 inject，不能是: $TASK_DELIVERY" >&2
        exit 1
        ;;
esac

if [ ! -d "$WORKSPACE" ]; then
    echo "错误: workspace 不存在: $WORKSPACE" >&2
    exit 1
fi
WORKSPACE="$(readlink -f "$WORKSPACE")"

if ! command -v zellij >/dev/null 2>&1; then
    echo "错误: 找不到 zellij" >&2
    exit 1
fi
if agent_is_used hermes && ! command -v hermes >/dev/null 2>&1; then
    echo "错误: 找不到 hermes 命令" >&2
    exit 1
fi

CODEX_INTERACTIVE="${REAL_HOME}/.local/bin/codex-kanban-interactive"
CODEWHALE_INTERACTIVE="${REAL_HOME}/.local/bin/codewhale-kanban-interactive"
REASONIX_INTERACTIVE="${REAL_HOME}/.local/bin/reasonix-kanban-interactive"
if agent_is_used codex && [ ! -x "$CODEX_INTERACTIVE" ]; then
    echo "错误: 找不到可执行 Codex kanban interactive: $CODEX_INTERACTIVE" >&2
    exit 1
fi
if agent_is_used codewhale && [ ! -x "$CODEWHALE_INTERACTIVE" ]; then
    echo "错误: 找不到可执行 CodeWhale kanban interactive: $CODEWHALE_INTERACTIVE" >&2
    exit 1
fi
if agent_is_used deepseek-reasonix && [ ! -x "$REASONIX_INTERACTIVE" ]; then
    echo "错误: 找不到可执行 Reasonix kanban interactive: $REASONIX_INTERACTIVE" >&2
    exit 1
fi

deepseek_pane_count() {
    local n=0
    [ "$COORDINATOR_AGENT" = "codewhale" ] && n=$((n + 1))
    [ "$PLANNER_AGENT" = "codewhale" ] && n=$((n + 1))
    [ "$IMPLEMENTER_AGENT" = "codewhale" ] && n=$((n + 1))
    [ "$CRITIC_AGENT" = "codewhale" ] && n=$((n + 1))
    [ "$COORDINATOR_AGENT" = "deepseek-reasonix" ] && n=$((n + 1))
    [ "$PLANNER_AGENT" = "deepseek-reasonix" ] && n=$((n + 1))
    [ "$IMPLEMENTER_AGENT" = "deepseek-reasonix" ] && n=$((n + 1))
    [ "$CRITIC_AGENT" = "deepseek-reasonix" ] && n=$((n + 1))
    printf '%s' "$n"
}

deepseek_primary_role() {
    # critic 优先续接旧会话：critic 需要完整的上下文来做审查判断
    if [ "$CRITIC_AGENT" = "codewhale" ] || [ "$CRITIC_AGENT" = "deepseek-reasonix" ]; then
        printf '%s' "critic"
    elif [ "$IMPLEMENTER_AGENT" = "codewhale" ] || [ "$IMPLEMENTER_AGENT" = "deepseek-reasonix" ]; then
        printf '%s' "implementer"
    elif [ "$PLANNER_AGENT" = "codewhale" ] || [ "$PLANNER_AGENT" = "deepseek-reasonix" ]; then
        printf '%s' "planner"
    elif [ "$COORDINATOR_AGENT" = "codewhale" ] || [ "$COORDINATOR_AGENT" = "deepseek-reasonix" ]; then
        printf '%s' "coordinator"
    else
        printf '%s' "critic"
    fi
}

deepseek_continue_flag_for_role() {
    local role="$1"
    local count primary policy
    count="$(deepseek_pane_count)"
    policy="$DEEPSEEK_CONTINUE_POLICY"
    [ "$policy" = "auto" ] && policy="primary-only"
    case "$policy" in
        all)
            printf '%s' "--continue"
            ;;
        none)
            printf '%s' "--no-continue"
            ;;
        primary-only)
            if [ "$count" -le 1 ]; then
                printf '%s' "--continue"
                return
            fi
            primary="$(deepseek_primary_role)"
            if [ "$role" = "$primary" ]; then
                printf '%s' "--continue"
            else
                printf '%s' "--no-continue"
            fi
            ;;
    esac
}

kanban_pid_lines() {
    BOARD_FILTER="$BOARD" python3 - <<'PY'
from __future__ import annotations
import os
import re

board = os.environ.get("BOARD_FILTER", "")
board_b = board.encode()
self_pid = os.getpid()

roles = {"coordinator", "planner", "implementer", "critic"}
hermes_re = re.compile(r"\bhermes\b.*\s-p\s+(coordinator|planner|implementer|critic)\b")
listener_re = re.compile(r"(codex-kanban-interactive|codex_kanban_interactive\.py|codex-kanban-listen|codex_kanban_listener\.py|codewhale-kanban-interactive|codewhale_kanban_interactive\.py|deepseek-kanban-interactive|deepseek_kanban_interactive\.py|deepseek-kanban-listen|deepseek_kanban_listener\.py|reasonix-kanban-interactive|reasonix_kanban_interactive\.py)")
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
    if "start-kanban.sh" in cmd:
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

matching_sessions() {
    zellij list-sessions --no-formatting --short 2>/dev/null | clean_session_name | grep -Fx "$SESSION_NAME" || true
}

process_children() {
    local pid="$1"
    pgrep -P "$pid" 2>/dev/null || true
}

process_tree_postorder() {
    local pid="$1"
    local child
    for child in $(process_children "$pid"); do
        process_tree_postorder "$child"
    done
    printf '%s\n' "$pid"
}

terminate_process_tree() {
    local root_pid="$1"
    local reason="$2"
    local pid
    local pids=()
    mapfile -t pids < <(process_tree_postorder "$root_pid")
    if [ "${#pids[@]}" -eq 0 ]; then
        return
    fi
    echo "  kill SIGTERM tree -> root PID $root_pid ($reason): ${pids[*]}"
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    for pid in "${pids[@]}"; do
        if ps -p "$pid" >/dev/null 2>&1; then
            echo "  kill SIGKILL residual -> PID $pid"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

if [ "$DRY_RUN" != "1" ] && [ "$CLEAN" = "1" ]; then
    echo "=== 检查旧 kanban session/process ==="
    echo "  target session: $SESSION_NAME"
    echo "  target board:   $BOARD"
    target_sessions="$(matching_sessions)"
    stale_workers="$(kanban_pid_lines || true)"
    if [ -z "$target_sessions" ] && [ -z "$stale_workers" ]; then
        echo "未发现同名 session 或同 board kanban worker；不会触碰其他 zellij session。"
    fi
    if [ -n "$stale_workers" ]; then
        echo "发现同 board kanban worker，将只清理这些进程:"
        echo "$stale_workers"
        while IFS=$'\t' read -r pid reason cmd; do
            [ -n "${pid:-}" ] || continue
            terminate_process_tree "$pid" "$reason"
        done <<< "$stale_workers"
        # Wait for flock files to be released after process death
        sleep 1
    fi
    if [ -n "$target_sessions" ]; then
        echo "发现同名 zellij session，将只清理这些 session:"
        echo "$target_sessions"
        for s in $target_sessions; do
            echo "  kill session: $s"
            zellij kill-session "$s" 2>/dev/null || true
            # kill-session 只杀进程，session 变 EXITED 但仍占名；delete-session 彻底删除
            zellij delete-session "$s" 2>/dev/null || true
        done
    fi
    echo ""
elif [ "$DRY_RUN" != "1" ] && [ "$CLEAN" = "0" ]; then
    echo "跳过清理 (--no-clean)。注意：如果已有同 board listener，可能会抢同一批任务。"
    echo ""
fi

append_assist_delay_args() {
    local cmd="$1"
    local assist_delays="$2"
    local item
    if [ -n "$ASSIST_CLAIM_DELAY" ]; then
        cmd+=" --assist-claim-delay-s $(shell_quote "$ASSIST_CLAIM_DELAY")"
    fi
    if [ -n "$assist_delays" ]; then
        IFS=',' read -ra delay_parts <<< "$assist_delays"
        for item in "${delay_parts[@]}"; do
            [ -n "$item" ] || continue
            cmd+=" --assist-claim-delay-for $(shell_quote "$item")"
        done
    fi
    if [ -n "$GLOBAL_ASSIST_PROFILE_DELAYS" ]; then
        IFS=',' read -ra profile_delay_parts <<< "$GLOBAL_ASSIST_PROFILE_DELAYS"
        for item in "${profile_delay_parts[@]}"; do
            [ -n "$item" ] || continue
            cmd+=" --assist-claim-profile-delay $(shell_quote "$item")"
        done
    fi
    printf '%s' "$cmd"
}

build_role_command() {
    local role="$1"
    local agent="$2"
    local board_q role_q workspace_q codex_q provider_q model_q sandbox_q cmd claim_assignees claim_q assist_delay_q assist_delay_env assist_delays profile_delays_q item
    board_q="$(shell_quote "$BOARD")"
    role_q="$(shell_quote "$role")"
    workspace_q="$(shell_quote "$WORKSPACE")"
    codex_q="$(shell_quote "$CODEX_INTERACTIVE")"
    provider_q="$(shell_quote "$DEEPSEEK_PROVIDER")"

    # Stagger delay: 各 pane 启动间隔 0.5s，避免同时抢 DB 导致 race
    local stagger_s=0
    case "$role" in
        coordinator) stagger_s=0 ;;
        planner)    stagger_s=0.5 ;;
        implementer) stagger_s=1.0 ;;
        critic)     stagger_s=1.5 ;;
    esac

    case "$role" in
        coordinator)
            claim_assignees="$(claim_assignees_for_role "$role" "$COORDINATOR_ASSISTS")"
            assist_delays="$COORDINATOR_ASSIST_DELAYS" ;;
        planner)
            claim_assignees="$(claim_assignees_for_role "$role" "$PLANNER_ASSISTS")"
            assist_delays="$PLANNER_ASSIST_DELAYS" ;;
        implementer)
            claim_assignees="$(claim_assignees_for_role "$role" "$IMPLEMENTER_ASSISTS")"
            assist_delays="$IMPLEMENTER_ASSIST_DELAYS" ;;
        critic)
            claim_assignees="$(claim_assignees_for_role "$role" "$CRITIC_ASSISTS")"
            assist_delays="$CRITIC_ASSIST_DELAYS" ;;
    esac
    claim_q="$claim_assignees"
    assist_delay_env=""
    if [ -n "$ASSIST_CLAIM_DELAY" ]; then
        assist_delay_q="$(shell_quote "$ASSIST_CLAIM_DELAY")"
        assist_delay_env=" HERMES_KANBAN_ASSIST_CLAIM_DELAY_S=${assist_delay_q}"
    fi
    if [ -n "$assist_delays" ]; then
        assist_delay_q="$(shell_quote "$assist_delays")"
        assist_delay_env+=" HERMES_KANBAN_ASSIST_CLAIM_DELAYS=${assist_delay_q}"
    fi
    if [ -n "$GLOBAL_ASSIST_PROFILE_DELAYS" ]; then
        profile_delays_q="$(shell_quote "$GLOBAL_ASSIST_PROFILE_DELAYS")"
        assist_delay_env+=" HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS=${profile_delays_q}"
    fi

    case "$agent" in
        hermes)
            printf 'sleep %s && cd %s && HERMES_KANBAN_BOARD=%s HERMES_KANBAN_CLAIM_ASSIGNEES=%s%s hermes -p %s --continue' "$stagger_s" "$workspace_q" "$board_q" "$claim_q" "$assist_delay_env" "$role_q"
            ;;
        codex)
            cmd="cd ${workspace_q} && HERMES_KANBAN_BOARD=${board_q} CODEX_KANBAN_WORKSPACE=${workspace_q} ${codex_q} --profile ${role_q} --claim-assignees ${claim_q} --board ${board_q} --workspace ${workspace_q}"
            cmd="$(append_assist_delay_args "$cmd" "$assist_delays")"
            if [ -n "$CODEX_MODEL" ]; then
                model_q="$(shell_quote "$CODEX_MODEL")"
                cmd+=" --model ${model_q}"
            fi
            if [ -n "$CODEX_SANDBOX" ]; then
                sandbox_q="$(shell_quote "$CODEX_SANDBOX")"
                cmd+=" --sandbox ${sandbox_q}"
            fi
            cmd+=" --auto-start"
            printf 'sleep %s && %s' "$stagger_s" "$cmd"
            ;;
        codewhale)
            # 用 codewhale-kanban-interactive
            local cw_q
            cw_q="$(shell_quote "$CODEWHALE_INTERACTIVE")"
            cmd="cd ${workspace_q} && HERMES_KANBAN_BOARD=${board_q} CODEWHALE_KANBAN_WORKSPACE=${workspace_q} DEEPSEEK_KANBAN_WORKSPACE=${workspace_q} ${cw_q} --profile ${role_q} --claim-assignees ${claim_q} --board ${board_q} --workspace ${workspace_q}"
            cmd="$(append_assist_delay_args "$cmd" "$assist_delays")"
            if [ -n "$DEEPSEEK_MODEL" ]; then
                model_q="$(shell_quote "$DEEPSEEK_MODEL")"
                cmd+=" --model ${model_q}"
            fi
            if [ -n "$DEEPSEEK_TASK_TIMEOUT" ]; then
                timeout_q="$(shell_quote "$DEEPSEEK_TASK_TIMEOUT")"
                cmd+=" --task-timeout-s ${timeout_q}"
            fi
            if [ -n "$DEEPSEEK_IDLE_PANE_RECLAIM" ]; then
                timeout_q="$(shell_quote "$DEEPSEEK_IDLE_PANE_RECLAIM")"
                cmd+=" --idle-pane-reclaim-s ${timeout_q}"
            fi
            cmd+=" $(deepseek_continue_flag_for_role "$role")"
            cmd+=" --auto-start"
            printf 'sleep %s && %s' "$stagger_s" "$cmd"
            ;;
        deepseek-reasonix)
            reasonix_q="$(shell_quote "$REASONIX_INTERACTIVE")"
            cmd="cd ${workspace_q} && HERMES_KANBAN_BOARD=${board_q} ${reasonix_q} --profile ${role_q} --claim-assignees ${claim_q} --board ${board_q} --workspace ${workspace_q}"
            cmd="$(append_assist_delay_args "$cmd" "$assist_delays")"
            if [ -n "$DEEPSEEK_MODEL" ]; then
                model_q="$(shell_quote "$DEEPSEEK_MODEL")"
                cmd+=" --model ${model_q}"
            fi
            cmd+=" $(deepseek_continue_flag_for_role "$role")"
            cmd+=" --auto-start"
            printf 'sleep %s && %s' "$stagger_s" "$cmd"
            ;;
        *)
            echo "内部错误: unknown agent $agent" >&2
            return 1
            ;;
    esac
}

write_pane() {
    local indent="$1"
    local role="$2"
    local agent="$3"
    local name cmd
    name="${role}-${agent}"
    cmd="$(build_role_command "$role" "$agent")"
    printf '%spane name=%s command="bash" cwd=%s {\n' "$indent" "$(kdl_string "$name")" "$(kdl_string "$WORKSPACE")"
    printf '%s    args "-lc" %s\n' "$indent" "$(kdl_string "$cmd")"
    printf '%s}\n' "$indent"
}

LAYOUT_DIR="${REAL_HOME}/.config/zellij/layouts"
LAYOUT_FILE="${LAYOUT_DIR}/kanban-launcher.kdl"
mkdir -p "$LAYOUT_DIR"

{
    printf 'layout {\n'
    printf '    pane split_direction="horizontal" {\n'
    printf '        pane split_direction="vertical" {\n'
    write_pane "            " "coordinator" "$COORDINATOR_AGENT"
    write_pane "            " "planner" "$PLANNER_AGENT"
    printf '        }\n'
    printf '        pane split_direction="vertical" {\n'
    write_pane "            " "implementer" "$IMPLEMENTER_AGENT"
    write_pane "            " "critic" "$CRITIC_AGENT"
    printf '        }\n'
    printf '    }\n'
    printf '}\n'
} > "$LAYOUT_FILE"

echo "启动 kanban 4 分窗口 (board=${BOARD}, session=${SESSION_NAME})..."
echo "  coordinator-${COORDINATOR_AGENT} | planner-${PLANNER_AGENT}"
echo "  implementer-${IMPLEMENTER_AGENT} | critic-${CRITIC_AGENT}"
echo "  workspace: ${WORKSPACE}"
echo "  Task delivery: ${TASK_DELIVERY}"
if agent_is_used codex; then
    echo "  Codex model/sandbox: ${CODEX_MODEL:-auto}/${CODEX_SANDBOX:-auto}"
fi
if agent_is_used codewhale; then
    echo "  CodeWhale provider/model: ${DEEPSEEK_PROVIDER}/${DEEPSEEK_MODEL:-auto}"
    echo "  CodeWhale continue policy: ${DEEPSEEK_CONTINUE_POLICY} (primary=$(deepseek_primary_role), panes=$(deepseek_pane_count))"
fi
if agent_is_used deepseek-reasonix; then
    echo "  Reasonix model: ${DEEPSEEK_MODEL:-auto}"
    echo "  DeepSeek continue policy: ${DEEPSEEK_CONTINUE_POLICY} (primary=$(deepseek_primary_role), panes=$(deepseek_pane_count))"
fi
echo ""

if [ "$DRY_RUN" = "1" ]; then
    echo "dry-run: 已生成 layout: $LAYOUT_FILE"
    sed -n '1,180p' "$LAYOUT_FILE"
    exit 0
fi

# ── Kill any stray dispatch/daemon processes ──────────────────────
# dispatch 和 daemon 会启动 headless hermes worker，与 interactive listener 竞争
# claim 导致 DB 锁冲突和索引损坏。启动 kanban 前必须清理。
for proc_pattern in "hermes.*kanban.*dispatch" "hermes.*kanban.*daemon"; do
    if pgrep -f "$proc_pattern" >/dev/null 2>&1; then
        echo "⚠️  发现残留的 kanban dispatch/daemon 进程，正在杀掉："
        pgrep -af "$proc_pattern" 2>/dev/null | head -5
        pkill -f "$proc_pattern" 2>/dev/null || true
        sleep 1
        # 如果还没死就 SIGKILL
        if pgrep -f "$proc_pattern" >/dev/null 2>&1; then
            pkill -9 -f "$proc_pattern" 2>/dev/null || true
        fi
    fi
done

exec zellij --session "$SESSION_NAME" --new-session-with-layout kanban-launcher
