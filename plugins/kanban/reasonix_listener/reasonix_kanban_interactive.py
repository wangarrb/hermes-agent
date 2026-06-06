#!/usr/bin/env python3
"""Interactive Reasonix + Hermes Kanban listener bridge.

This mode keeps Reasonix as a visible interactive TUI while a small
background watcher claims Hermes Kanban tasks and injects a short instruction
into the same Zellij pane.  Reasonix itself is responsible for completing/
blocking the task via `hermes kanban`.

Architecture mirrors codex_kanban_interactive.py but targets Reasonix CLI
instead of Codex CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Source layout: <repo>/plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py
HERMES_REPO = Path(__file__).resolve().parents[3]
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402

_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


def claim_lock() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:reasonix-interactive"


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_label()}] {msg}\n")


def prompt_dir(workspace: Path, board: str, pane_profile: str) -> Path:
    safe_board = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in board)
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pane_profile)
    return workspace / ".reasonix-kanban" / safe_board / safe_profile


def self_poll_prompt_dir(workspace: Path, board: str, pane_profile: str) -> Path:
    safe_board = worker_runtime.safe_path_component(board)
    safe_profile = worker_runtime.safe_path_component(pane_profile)
    return workspace / ".hermes-kanban" / safe_board / safe_profile


def write_self_poll_startup_prompt(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    ttl: int,
    listener_kind: str,
    pane_id: str | None = None,
) -> tuple[Path, str]:
    owner = worker_runtime.default_self_poll_owner(
        profile=profile,
        listener_kind=listener_kind,
        pane_id=pane_id,
    )
    prompt = worker_runtime.build_self_poll_startup_prompt(
        agent_label="interactive Reasonix",
        board=board,
        profile=profile,
        claim_assignees=claim_assignees,
        workspace=workspace,
        ttl=ttl,
        listener_kind=listener_kind,
        owner=owner,
        role_guidance_text=role_guidance(profile),
    )
    path = self_poll_prompt_dir(workspace, board, profile) / "self-poll-startup.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    return path, prompt


def claim_assignees(args: argparse.Namespace) -> list[str]:
    """Return assignees this single pane may claim, in priority order."""
    return worker_runtime.claim_assignees_from_args(args, default_profile="implementer")


def reset_kanban_claims(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    reason: str = "operator reset-kanban",
) -> list[str]:
    reset_ids: list[str] = []
    reset_ids.extend(
        worker_runtime.reset_interactive_claims(
            board=board,
            profile=profile,
            claim_assignees=claim_assignees,
            workspace=workspace,
            listener_kind="reasonix-interactive",
            reason=reason,
        )
    )
    owner = worker_runtime.default_self_poll_owner(
        profile=profile,
        listener_kind="reasonix-self-poll",
    )
    reset_ids.extend(
        worker_runtime.reset_self_poll_claims(
            board=board,
            profile=profile,
            claim_assignees=claim_assignees,
            workspace=workspace,
            listener_kind="reasonix-self-poll",
            owner=owner,
            reason=reason,
        )
    )
    return list(dict.fromkeys(reset_ids))


def assist_claim_delay_s(args: argparse.Namespace) -> float:
    return worker_runtime.claim_policy_from_args(args, default_profile="implementer").assist_claim_delay_s


def _delay_specs(raw: Any) -> list[str]:
    return worker_runtime.split_csv_values(raw)


def _split_delay_spec(spec: str) -> tuple[str, float] | None:
    if "=" in spec:
        key, value = spec.rsplit("=", 1)
    elif ":" in spec:
        key, value = spec.rsplit(":", 1)
    else:
        key, value = "implementer", spec
    key = key.strip()
    if not key:
        key = "implementer"
    try:
        delay_s = max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None
    return key, delay_s


def _split_profile_delay_spec(spec: str) -> tuple[str, str, float] | None:
    if "=" in spec:
        key, value = spec.rsplit("=", 1)
        parts = [part.strip() for part in key.split(":")]
    else:
        parts = [part.strip() for part in spec.split(":")]
        value = parts.pop() if parts else ""
    if not parts:
        profile = "implementer"
        assignee = "implementer"
    elif len(parts) == 1:
        profile = parts[0] or "implementer"
        assignee = "implementer"
    else:
        profile = parts[0] or "implementer"
        assignee = parts[1] or "implementer"
    try:
        delay_s = max(0.0, float(value.strip()))
    except (TypeError, ValueError):
        return None
    return profile, assignee, delay_s


def assist_claim_delays(args: argparse.Namespace) -> dict[str, float]:
    return worker_runtime.assist_claim_delays_from_args(args)


def assist_claim_profile_delays(args: argparse.Namespace) -> dict[tuple[str, str], float]:
    return worker_runtime.assist_claim_profile_delays_from_args(args)


def assist_claim_delay_for(args: argparse.Namespace, assignee: str) -> float:
    policy = worker_runtime.claim_policy_from_args(args, default_profile="implementer")
    return worker_runtime.assist_claim_delay_for(policy, assignee)


def _ready_since(conn, task_id: str, fallback_created_at: int) -> int:
    return worker_runtime.ready_since(conn, task_id, fallback_created_at)


def _assist_candidate_ready(conn, args: argparse.Namespace, task: kb.Task, assignee: str) -> bool:
    policy = worker_runtime.claim_policy_from_args(args, default_profile="implementer")
    return worker_runtime.assist_candidate_ready(conn, policy=policy, task=task, assignee=assignee)


def _select_ready_candidate(conn, args: argparse.Namespace) -> kb.Task | None:
    policy = worker_runtime.claim_policy_from_args(args, default_profile="implementer")
    return worker_runtime.select_ready_candidate(conn, policy=policy)


def role_guidance(profile: str) -> str:
    """Role-bound guidance shared by all agent backends.

    The visible pane can be Codex, Reasonix, or Hermes; the job semantics
    must still come from the Kanban role/profile.
    """
    p = (profile or "").strip().lower()
    common = "职责由 Kanban profile/assignee 决定，而不是由底层 agent 类型决定；即使用 Reasonix 运行，也要按当前角色工作。"
    per_role = {
        "coordinator": "你是 coordinator：和用户对齐目标，拆分任务，维护 Kanban 流转；除非任务明确很小，否则不要替 planner/implementer/critic 做大段执行。",
        "planner": "你是 planner：负责方案设计、实验计划和任务拆分。输出必须具体到文件路径、函数/类名、命令、预期结果和验收标准。",
        "implementer": "你是 implementer：负责落地执行。先读上下文和相关代码，再小步修改；改完运行最小可行验证，并在结果里说明改了什么、如何验证。",
        "critic": "你是 critic：负责审查、找漏洞和独立验证。不要默认相信 planner/implementer 结论；重点检查证据链、遗漏风险、指标口径和可复现性。最终放行只有 PASS/FAIL，没有带病通过；所有任务范围内能解决的问题都解决后才能 PASS，否则必须 FAIL 并列出具体修复/复审要求。NO_CLAIM 只表示指标不支持 claim，不是缺陷豁免。",
    }
    return common + "\n" + per_role.get(p, f"你当前角色是 {profile}：按该 assignee 的职责完成任务，不要因为运行在 Reasonix 中而改变角色边界。")


def build_interactive_prompt(
    *,
    board: str,
    pane_profile: str,
    task_assignee: str,
    task: kb.Task,
    context: str,
    workspace: Path,
) -> str:
    metadata = {
        "executor": "reasonix-interactive",
        "listener": "reasonix_kanban_interactive",
        "board": board,
        "pane_profile": pane_profile,
        "task_assignee": task_assignee,
        "workspace": str(workspace),
    }
    metadata_json = json.dumps(metadata, ensure_ascii=False)
    complete_template = (
        f"hermes kanban --board {board} complete {task.id} "
        "--summary '<一句话摘要>' "
        "--result '<交付细节/关键结论>' "
        f"--metadata '{metadata_json}'"
    )
    block_template = f"hermes kanban --board {board} block {task.id} '<阻塞原因>'"
    role_prompt = textwrap.indent(role_guidance(task_assignee), "        ")
    assist_note = ""
    if pane_profile != task_assignee:
        assist_note = (
            f"虽然当前 TUI pane/profile 是 {pane_profile}，但这个 Kanban 任务的 assignee 是 "
            f"{task_assignee}；本次必须按 {task_assignee} 职责工作。"
        )
    return textwrap.dedent(
        f"""
        # Hermes Kanban task for role: {task_assignee}

        你现在承担 Hermes Kanban 角色：{task_assignee}。
        底层执行器：interactive Reasonix；职责/提示词按角色绑定，不按 agent 类型绑定。
        Board: {board}
        Pane/profile: {pane_profile}
        Task assignee/role: {task_assignee}
        Task: {task.id} — {task.title}
        Workspace: {workspace}
        {assist_note}

        角色提示：
{role_prompt}

        这个任务已经被后台 listener 原子 claim，其他 worker 不会抢同一个任务。
        交互式模式下，你需要自己在完成后流转 Kanban 状态；listener 只负责 claim、注入、heartbeat。

        必须遵守：
        1. 默认中文输出；路径、命令、技术名词保留英文。
        2. 先读任务上下文，再行动。需要查父任务/评论时可运行：
           hermes kanban --board {board} context {task.id}
           hermes kanban --board {board} show {task.id}
        3. 如果要创建后继任务，必须用真实返回 ID，不要编造：
           hermes kanban --board {board} create ... --json
        4. 完成后必须运行 complete 或 block，不能只在对话里说完成。
        5. 如果任务涉及 Egomotion4D 仓库命令，遵守仓库 AGENTS.md：不要裸跑 python/pytest/pip；本地用 conda run -n egomotion4d env PYTHONPATH=src ...；正式 GPU/benchmark 默认去 gpuserver。

        6. 计划质量要求（当你的角色是 planner 或任务涉及制定计划/拆分子任务时）：
           - 计划必须让低能力 coding agent（如 Reasonix implementer）也能无歧义理解执行。每个步骤要给出：具体文件路径、具体函数/类名、具体命令、预期输出/行为。不要写"修改相关文件"这种模糊描述。
           - 预估每个步骤可能踩的坑，明确写出来让执行者避坑。例如：某函数参数顺序容易搞反、某配置项必须用特定值否则会静默失败、某步骤依赖上一步的特定输出格式。
           - 计划必须可验证：每个步骤完成后，给出验证命令或检查点（如：运行某测试、检查某文件存在且包含某内容、对比某指标值)。执行者做完一步就能自己确认对不对，而不是做完一整串才发现方向错了。

        长期记忆：
        你可以使用 hindsight 作为长期记忆工具。遇到需要回忆之前讨论、项目历史、技术决策时，用 shell 执行：
          hindsight recall "<查询内容>"
          hindsight reflect "<需要综合推理的问题>"
        存储新知识时：
          hindsight retain "<要记住的事实>" --context "<上下文标签>"
        这比靠对话上下文记忆更可靠，跨 session 持久化。

        完成命令模板：
        {complete_template}

        阻塞命令模板：
        {block_template}

        下面是 Hermes Kanban worker context：

        {context}
        """
    ).strip() + "\n"


def write_task_prompt(
    *,
    workspace: Path,
    board: str,
    pane_profile: str,
    task: kb.Task,
    context: str,
) -> Path:
    task_assignee = getattr(task, "assignee", None) or pane_profile
    d = prompt_dir(workspace, board, pane_profile)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{task.id}.md"
    p.write_text(
        build_interactive_prompt(
            board=board,
            pane_profile=pane_profile,
            task_assignee=task_assignee,
            task=task,
            context=context,
            workspace=workspace,
        ),
        encoding="utf-8",
    )
    return p


def write_workspace_reasonix_toml(
    *,
    workspace: Path,
    role_instructions_path: Path | None,
) -> Path:
    """Write a per-workspace reasonix.toml with system_prompt_file.

    Reasonix v1.1.0+ (Go rewrite) uses reasonix.toml instead of CLI flags:
    - system_prompt_file replaces --system-append-file
    - 'reasonix chat --dir <workspace>' replaces 'reasonix code <workspace>'
    - --no-mouse / --no-dashboard / --new are removed
    """
    toml_path = workspace / "reasonix.toml"

    # Build TOML content
    lines = [
        "# Auto-generated by Hermes kanban listener — do not edit manually",
        "# Reasonix v1.1.0+ workspace config for kanban mode",
    ]

    if role_instructions_path and role_instructions_path.exists():
        # Use relative path if possible, absolute as fallback
        try:
            rel = role_instructions_path.relative_to(workspace)
            prompt_path = str(rel)
        except ValueError:
            prompt_path = str(role_instructions_path)
        lines.append(f'system_prompt_file = "{prompt_path}"')

    # If a user-managed reasonix.toml already exists, back it up
    if toml_path.exists():
        backup = toml_path.with_suffix(".toml.user-bak")
        if not backup.exists():
            import shutil
            shutil.copy2(toml_path, backup)

    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return toml_path


def write_role_instructions_file(
    *,
    workspace: Path,
    board: str,
    pane_profile: str,
) -> Path:
    """Write role instructions file for system_prompt_file (reasonix.toml).

    Reasonix v1.1.0+ uses system_prompt_file in reasonix.toml to inject
    role definitions, replacing the old --system-append-file CLI flag.
    The file content format is unchanged.
    """
    d = prompt_dir(workspace, board, pane_profile)
    d.mkdir(parents=True, exist_ok=True)
    p = d / "role-instructions.md"
    p.write_text(role_guidance(pane_profile), encoding="utf-8")
    return p


def zellij_inject(*, session: str, pane_id: str, text: str, log_path: Path) -> bool:
    """Type text into a target Zellij pane and press Enter."""
    try:
        subprocess.run(
            ["zellij", "--session", session, "action", "write-chars", "-p", str(pane_id), text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.3)
        subprocess.run(
            ["zellij", "--session", session, "action", "send-keys", "-p", str(pane_id), "Enter"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log_line(log_path, f"zellij injection failed rc={exc.returncode}: {exc.stderr.strip()}")
        return False


def zellij_rename_pane(*, session: str, pane_id: str, name: str, log_path: Path) -> bool:
    """Set a persistent Zellij pane title so listener state is visible."""
    try:
        subprocess.run(
            ["zellij", "--session", session, "action", "rename-pane", "-p", str(pane_id), name],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log_line(log_path, f"zellij rename-pane failed rc={exc.returncode}: {exc.stderr.strip()}")
        return False


_REASONIX_IDLE_MARKERS = (
    # v0.x idle prompt (English)
    "ask anything · slash for commands · at-sign for files",
    # v0.x idle prompt (Chinese)
    "输入任何内容 · / 使用命令 · @ 引用文件",
    "type to steer the current task",
    # v1.x idle prompts (Go rewrite TUI)
    "输入 'exit' 或按 ctrl-d 退出",
    "enter 'exit' or press ctrl-d to exit",
    "对话上下文将跨轮保留",
    "conversation context will be preserved",
    # v1.x status bar idle indicators
    "yolo  · 已跳过批准",
    "yolo · auto-approved",
    "已跳过批准",
)

_REASONIX_BUSY_MARKERS = (
    "kanban_task_boundary",
    "hermes kanban 已领取任务",
    "完成后必须运行",
    "processing",
    "run running",
    "read running",
    "write running",
    "edit running",
)

_REASONIX_QUEUED_INPUT_MARKERS = (
    "pending inputs",
    "edit last queued message",
)


def zellij_dump_screen(*, session: str, pane_id: str, log_path: Path) -> str | None:
    """Return visible Zellij pane text, or None if the pane cannot be inspected."""
    try:
        result = subprocess.run(
            ["zellij", "--session", session, "action", "dump-screen", "--pane-id", str(pane_id), "--full"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        return result.stdout or ""
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        log_line(log_path, f"zellij dump-screen failed: {detail.strip()}")
        return None


def _tail_nonempty_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-limit:]


def _normalize_ws(s: str) -> str:
    """Collapse consecutive whitespace into single spaces for robust matching.

    Reasonix TUI renders idle markers with variable spacing (e.g. double
    spaces around ·) depending on font/terminal width; normalizing lets
    our markers match regardless.
    """
    return " ".join(s.split())


def _looks_like_idle_reasonix_pane(text: str) -> bool:
    """Detect a Reasonix pane that is waiting for user input.

    We only inspect the visible tail so old scrollback containing an idle prompt
    does not override current task output.  Whitespace is normalized before
    matching because Reasonix TUI uses variable spacing around · separators.
    """
    tail = "\n".join(_tail_nonempty_lines(text)).lower()
    if not tail:
        return False
    tail_norm = _normalize_ws(tail)
    if any(marker in tail_norm for marker in (*_REASONIX_BUSY_MARKERS, *_REASONIX_QUEUED_INPUT_MARKERS)):
        return False
    return any(marker in tail_norm for marker in _REASONIX_IDLE_MARKERS)


def _looks_like_busy_reasonix_pane(text: str) -> bool:
    """Return True when the visible pane claims work is in progress."""
    tail = "\n".join(_tail_nonempty_lines(text)).lower()
    if not tail:
        return False
    tail_norm = _normalize_ws(tail)
    return any(marker in tail_norm for marker in (*_REASONIX_BUSY_MARKERS, *_REASONIX_QUEUED_INPUT_MARKERS))


def _pane_can_accept_new_kanban_task(text: str) -> bool:
    """Return True when it is safe to inject a new Kanban prompt.

    Reasonix can buffer text typed into the pane while a previous prompt is
    still visible or being processed.  Injecting another KANBAN_TASK_BOUNDARY at
    that point stacks pending user input and can make later commands impossible
    to enter.  Only an explicit idle prompt with no queued-input or active-tool
    marker is safe enough for an automated injection.
    """
    return _looks_like_idle_reasonix_pane(text)


def _should_restart_watcher(returncode: int | None) -> bool:
    """Restart only crashed watchers; rc=0 means an intentional clean exit."""
    return returncode is not None and int(returncode) != 0


def _pane_screen(args: argparse.Namespace, *, log_path: Path) -> str | None:
    session = getattr(args, "zellij_session", None)
    pane_id = getattr(args, "zellij_pane_id", None)
    if not session or not pane_id:
        return None
    return zellij_dump_screen(session=str(session), pane_id=str(pane_id), log_path=log_path)


def _pane_looks_idle(args: argparse.Namespace, *, log_path: Path) -> bool:
    screen = _pane_screen(args, log_path=log_path)
    if screen is None:
        return False
    return _looks_like_idle_reasonix_pane(screen)


def _screen_fingerprint(screen: str) -> str:
    return "\n".join(_tail_nonempty_lines(screen, limit=40))


class _PaneProgressWatch:
    def __init__(self) -> None:
        self.fingerprint: str | None = None
        self.latest_session_mtime = 0.0
        self.stalled_busy_seen_at: float | None = None


def _observe_pane_progress(
    watch: _PaneProgressWatch,
    *,
    screen: str,
    now: float,
    latest_session_mtime: float,
    has_external_child: bool,
    reclaim_s: float,
) -> str | None:
    """Detect a fake-busy Reasonix pane without penalizing real long commands.

    A task is reclaimable only when the pane continues to look busy while all
    independent progress signals stay still: no screen change, no Reasonix
    session write, and no external subprocess under the TUI.
    """
    if reclaim_s <= 0:
        watch.stalled_busy_seen_at = None
        return None
    if has_external_child:
        watch.stalled_busy_seen_at = None
        return None

    fingerprint = _screen_fingerprint(screen)
    screen_changed = watch.fingerprint != fingerprint
    session_changed = (
        latest_session_mtime > 0
        and watch.latest_session_mtime > 0
        and latest_session_mtime > watch.latest_session_mtime
    )
    if latest_session_mtime > watch.latest_session_mtime:
        watch.latest_session_mtime = latest_session_mtime
    watch.fingerprint = fingerprint

    if not _looks_like_busy_reasonix_pane(screen):
        watch.stalled_busy_seen_at = None
        return None
    if screen_changed:
        watch.stalled_busy_seen_at = now
        return None
    if session_changed:
        watch.stalled_busy_seen_at = None
        return None
    if watch.stalled_busy_seen_at is None:
        watch.stalled_busy_seen_at = now
        return None

    stalled_s = now - watch.stalled_busy_seen_at
    if stalled_s >= reclaim_s:
        return (
            "reasonix-interactive pane looked busy but made no progress "
            f"for {stalled_s:.0f}s (limit {reclaim_s:.0f}s)"
        )
    return None


def _skip_reclaim_signal(pid: int, signum: int) -> None:  # noqa: ARG001
    """Tell kanban_db reclaim logic not to signal the interactive watcher.

    For interactive panes, ``worker_pid`` is the small watcher process, not the
    visible TUI doing the work.  When the watcher itself decides to requeue a
    stale/idle task, killing ``worker_pid`` would kill the recovery process.
    """
    raise ProcessLookupError(pid)


def _reclaim_task_without_signaling_worker(conn, task_id: str, *, reason: str) -> bool:
    return kb.reclaim_task(conn, task_id, reason=reason, signal_fn=_skip_reclaim_signal)


def _task_status(conn, task_id: str) -> tuple[str | None, int | None]:
    row = conn.execute("SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)).fetchone()
    if not row:
        return None, None
    return row["status"], row["current_run_id"]


# ── dynamic timeout: liveness via session file updates ──────────────────────

_REASONIX_SESSIONS_DIR = Path.home() / ".reasonix" / "sessions"


def _sessions_latest_mtime() -> float:
    """Return the most recent mtime among all session JSON files.

    Returns 0.0 if no session files exist yet (fresh install / --new start).
    """
    latest = 0.0
    try:
        for entry in _REASONIX_SESSIONS_DIR.iterdir():
            if not entry.is_file() or not entry.suffix == ".json":
                continue
            try:
                mtime = entry.stat().st_mtime
                if mtime > latest:
                    latest = mtime
            except OSError:
                continue
    except FileNotFoundError:
        pass
    return latest


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _proc_children(pid: int) -> list[int]:
    try:
        text = Path(f"/proc/{int(pid)}/task/{int(pid)}/children").read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return []
    out: list[int] = []
    for item in text.split():
        try:
            out.append(int(item))
        except ValueError:
            continue
    return out


def _proc_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return []
    return [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]


def _cmdline_arg(args: list[str], name: str) -> str:
    for i, arg in enumerate(args):
        if arg == name and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{name}="):
            return arg.split("=", 1)[1]
    return ""


def _cmdline_session_matches(args: list[str], session_name: str) -> bool:
    """Unused — reasonix has no --session param. Kept as no-op for compat."""
    return False


def _reasonix_pids_for_session(session_name: str) -> list[int]:
    """Find Reasonix siblings launched by this interactive bridge."""
    parent_pid = os.getppid()
    out: list[int] = []
    for pid in _proc_children(parent_pid):
        if pid == os.getpid():
            continue
        args = _proc_cmdline(pid)
        if not args:
            continue
        exe = Path(args[0]).name
        if exe != "reasonix" and not any("reasonix" in part for part in args[:2]):
            continue
        if _cmdline_session_matches(args, session_name):
            out.append(pid)
    return out


def _reasonix_has_external_child(session_name: str) -> bool:
    for pid in _reasonix_pids_for_session(session_name):
        if _proc_children(pid):
            return True
    return False


def _claim_delay_remaining(*, last_transition_at: float, now: float, delay_s: float) -> float:
    """Return remaining delay before claiming another task after a task boundary."""
    if last_transition_at <= 0.0 or delay_s <= 0.0:
        return 0.0
    return max(0.0, float(last_transition_at) + float(delay_s) - float(now))


def _workspace_matches(row_workspace: str | None, workspace: Path) -> bool:
    if not row_workspace:
        return True
    try:
        return Path(row_workspace).expanduser().resolve() == workspace
    except Exception:
        return False


def adopt_orphaned_running_task(args: argparse.Namespace, *, log_path: Path) -> tuple[str | None, int | None]:
    """Adopt a running task whose previous Reasonix bridge died.

    This is intentionally conservative.  It only adopts tasks already claimed
    by the Reasonix interactive bridge on this host whose recorded worker PID is
    no longer alive.  That covers the common recovery case where the visible
    Reasonix is still open but the small Kanban watcher disappeared.
    """
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    host = socket.gethostname()
    now = int(time.time())
    new_lock = claim_lock()
    expires = now + int(args.ttl)
    # Adoption is only safe for this pane's primary profile: it assumes the
    # visible TUI already received the task prompt before the watcher restarted.
    # Assisted assignees must be reclaimed to ready first, then claimed through
    # claim_and_inject_one so the assist pane gets a fresh prompt.
    assignees = [args.profile]
    with kb.connect(board=board) as conn:
        rows = conn.execute(
            f"""
            SELECT id, current_run_id, claim_lock, claim_expires, worker_pid, workspace_path
              FROM tasks
             WHERE assignee IN ({",".join("?" for _ in assignees)})
               AND status = 'running'
               AND claim_lock IS NOT NULL
             ORDER BY started_at ASC, id ASC
            """,
            tuple(assignees),
        ).fetchall()
        for row in rows:
            old_lock = row["claim_lock"] or ""
            if not old_lock.endswith(":reasonix-interactive"):
                continue
            lock_host = old_lock.split(":", 1)[0] if ":" in old_lock else ""
            if lock_host != host:
                continue
            if not _workspace_matches(row["workspace_path"], workspace):
                continue
            old_pid = int(row["worker_pid"]) if row["worker_pid"] else None
            if _pid_alive(old_pid):
                continue
            task_id = row["id"]
            run_id = int(row["current_run_id"]) if row["current_run_id"] else None
            if _pane_looks_idle(args, log_path=log_path):
                if _reclaim_task_without_signaling_worker(
                    conn,
                    task_id,
                    reason=(
                        "reasonix-interactive startup found orphaned running task "
                        "but target pane is idle; prompt was not active after restart"
                    ),
                ):
                    log_line(log_path, f"reclaimed orphaned idle-pane task {task_id} old_pid={old_pid} run={run_id}")
                return None, None
            with kb.write_txn(conn):  # type: ignore[attr-defined]
                cur = conn.execute(
                    """
                    UPDATE tasks
                       SET claim_lock = ?,
                           claim_expires = ?,
                           worker_pid = ?
                     WHERE id = ?
                       AND status = 'running'
                       AND claim_lock = ?
                    """,
                    (new_lock, expires, os.getpid(), task_id, old_lock),
                )
                if cur.rowcount != 1:
                    continue
                if run_id is not None:
                    conn.execute(
                        """
                        UPDATE task_runs
                           SET claim_lock = ?,
                               claim_expires = ?,
                               worker_pid = ?
                         WHERE id = ?
                        """,
                        (new_lock, expires, os.getpid(), run_id),
                    )
                kb._append_event(  # type: ignore[attr-defined]
                    conn,
                    task_id,
                    "adopted",
                    {
                        "old_lock": old_lock,
                        "old_pid": old_pid,
                        "new_lock": new_lock,
                        "new_pid": os.getpid(),
                        "reason": "reasonix-interactive watcher restarted",
                    },
                    run_id=run_id,
                )
            kb.heartbeat_worker(
                conn,
                task_id,
                note="reasonix-interactive adopted existing running task",
                expected_run_id=run_id,
            )
            log_line(log_path, f"adopted running task {task_id} old_pid={old_pid} run={run_id}")
            return task_id, run_id
    return None, None


def _cleanup_active_claim(*, board: str, task_id: str | None, run_id: int | None, log_path: Path) -> None:
    if not task_id:
        return
    try:
        with kb.connect(board=board) as conn:
            status, current_run_id = _task_status(conn, task_id)
            if status == "running" and (run_id is None or current_run_id == run_id):
                if _reclaim_task_without_signaling_worker(
                    conn,
                    task_id,
                    reason="reasonix-interactive listener stopped before completion",
                ):
                    log_line(log_path, f"reclaimed active task on stop: {task_id}")
    except Exception as exc:
        log_line(log_path, f"cleanup active claim failed for {task_id}: {type(exc).__name__}: {exc}")


def claim_and_inject_one(args: argparse.Namespace, *, log_path: Path) -> tuple[str | None, int | None]:
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    pane_profile = args.profile
    screen = _pane_screen(args, log_path=log_path)
    if screen is None or not _pane_can_accept_new_kanban_task(screen):
        log_line(log_path, "skip claim: Reasonix pane is not explicitly idle/safe for Kanban injection")
        return None, None
    with kb.connect(board=board) as conn:
        kb.release_stale_claims(conn)
        kb.recompute_ready(conn)
        candidate = _select_ready_candidate(conn, args)
        if candidate is None:
            return None, None
        claimed = kb.claim_task(conn, candidate.id, ttl_seconds=args.ttl, claimer=claim_lock())
        if claimed is None:
            return None, None
        kb.set_workspace_path(conn, claimed.id, workspace)
        claimed = kb.get_task(conn, claimed.id) or claimed
        try:
            kb._set_worker_pid(conn, claimed.id, os.getpid())  # type: ignore[attr-defined]
        except Exception:
            pass
        context = kb.build_worker_context(conn, claimed.id)

    prompt_path = write_task_prompt(
        workspace=workspace,
        board=board,
        pane_profile=pane_profile,
        task=claimed,
        context=context,
    )
    screen = _pane_screen(args, log_path=log_path)
    if screen is None or not _pane_can_accept_new_kanban_task(screen):
        with kb.connect(board=board) as conn:
            _reclaim_task_without_signaling_worker(
                conn,
                claimed.id,
                reason="reasonix-interactive pane became unsafe before prompt injection",
            )
        log_line(log_path, f"reclaimed {claimed.id}: pane unsafe before injection")
        return None, None
    task_assignee = getattr(claimed, "assignee", None) or pane_profile
    inject_text = (
        "KANBAN_TASK_BOUNDARY\n"
        "读取并执行当前 Hermes Kanban 任务文件；不要延续上一轮输出。\n"
        f"Task: {claimed.id} — {claimed.title}\n"
        f"Pane/profile: {pane_profile}; task assignee/role: {task_assignee}\n"
        f"File: {prompt_path}\n"
        f"Finish with: hermes kanban --board {board} complete {claimed.id} ... OR block {claimed.id} ..."
    )
    ok = zellij_inject(
        session=args.zellij_session,
        pane_id=args.zellij_pane_id,
        text=inject_text,
        log_path=log_path,
    )
    if not ok:
        with kb.connect(board=board) as conn:
            _reclaim_task_without_signaling_worker(
                conn,
                claimed.id,
                reason="reasonix-interactive zellij injection failed",
            )
        return None, None

    with kb.connect(board=board) as conn:
        kb.add_comment(
            conn,
            claimed.id,
            "reasonix-interactive-listener",
            f"Injected into Zellij pane {args.zellij_pane_id}; prompt file: {prompt_path}",
        )
        kb.heartbeat_worker(
            conn,
            claimed.id,
            note=f"reasonix-interactive injected prompt: {prompt_path}",
            expected_run_id=claimed.current_run_id,
        )
    zellij_rename_pane(
        session=args.zellij_session,
        pane_id=args.zellij_pane_id,
        name=f"{pane_profile}-reasonix running {claimed.id}",
        log_path=log_path,
    )
    log_line(log_path, f"claimed+injected {claimed.id}: {claimed.title} prompt={prompt_path}")
    return claimed.id, claimed.current_run_id


def watcher_main(args: argparse.Namespace) -> int:
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    board = args.board or kb.get_current_board() or "default"
    log_path = kb.worker_logs_dir(board=board) / f"reasonix-interactive-{args.profile}.log"
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        log_line(log_path, f"workspace does not exist: {workspace}")
        return 2
    if not args.zellij_session or not args.zellij_pane_id:
        log_line(log_path, "missing zellij session/pane id; cannot inject into Reasonix TUI")
        return 2

    poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
    log_line(
        log_path,
        f"interactive watcher started profile={args.profile} claim_assignees={','.join(claim_assignees(args))} board={board} workspace={workspace} "
        f"pane={args.zellij_session}:{args.zellij_pane_id} poll={poll_s:g}s",
    )
    zellij_rename_pane(
        session=args.zellij_session,
        pane_id=args.zellij_pane_id,
        name=f"{args.profile}-reasonix listening",
        log_path=log_path,
    )
    if args.startup_delay_s > 0:
        time.sleep(args.startup_delay_s)

    task_timeout_s = float(args.task_timeout_s)
    idle_pane_reclaim_s = float(
        getattr(args, "idle_pane_reclaim_s", listener_policy.INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS)
    )
    last_activity_at = time.time()
    last_liveness_check = 0.0
    idle_pane_seen_at: float | None = None
    pane_progress_watch = _PaneProgressWatch()

    active_task: str | None = None
    active_run_id: int | None = None
    last_task_transition_at = 0.0
    last_hb = 0.0
    session_name = getattr(args, "reasonix_session", None) or args.profile
    try:
        while not _STOP:
            now = time.time()
            if active_task:
                with kb.connect(board=board) as conn:
                    status, current_run_id = _task_status(conn, active_task)
                    if status == "running" and (active_run_id is None or current_run_id == active_run_id):
                        # ── liveness check: if reasonix is producing output,
                        #     reset the inactivity timer ──
                        if now - last_liveness_check >= 30.0:
                            latest_mtime = _sessions_latest_mtime()
                            if latest_mtime > 0 and latest_mtime > last_activity_at:
                                log_line(
                                    log_path,
                                    f"liveness: session updated {now - last_activity_at:.0f}s ago → resetting timeout",
                                )
                                last_activity_at = now
                            if idle_pane_reclaim_s > 0:
                                screen = _pane_screen(args, log_path=log_path)
                                if screen is not None and _looks_like_idle_reasonix_pane(screen):
                                    if idle_pane_seen_at is None:
                                        idle_pane_seen_at = now
                                        log_line(log_path, f"idle pane observed for active task {active_task}")
                                    idle_s = now - idle_pane_seen_at
                                    if idle_s >= idle_pane_reclaim_s:
                                        log_line(
                                            log_path,
                                            f"idle pane reclaim {active_task}: idle {idle_s:.0f}s >= {idle_pane_reclaim_s:.0f}s",
                                        )
                                        _reclaim_task_without_signaling_worker(
                                            conn,
                                            active_task,
                                            reason=(
                                                "reasonix-interactive pane stayed idle while task remained running "
                                                f"for {idle_s:.0f}s (limit {idle_pane_reclaim_s:.0f}s)"
                                            ),
                                        )
                                        active_task = None
                                        active_run_id = None
                                        idle_pane_seen_at = None
                                        pane_progress_watch = _PaneProgressWatch()
                                        last_task_transition_at = time.time()
                                        last_hb = 0.0
                                        continue
                                else:
                                    idle_pane_seen_at = None
                                if screen is not None:
                                    stuck_reason = _observe_pane_progress(
                                        pane_progress_watch,
                                        screen=screen,
                                        now=now,
                                        latest_session_mtime=latest_mtime,
                                        has_external_child=_reasonix_has_external_child(session_name),
                                        reclaim_s=idle_pane_reclaim_s,
                                    )
                                    if stuck_reason:
                                        log_line(log_path, f"stalled busy pane reclaim {active_task}: {stuck_reason}")
                                        _reclaim_task_without_signaling_worker(
                                            conn,
                                            active_task,
                                            reason=stuck_reason,
                                        )
                                        active_task = None
                                        active_run_id = None
                                        idle_pane_seen_at = None
                                        pane_progress_watch = _PaneProgressWatch()
                                        last_task_transition_at = time.time()
                                        last_hb = 0.0
                                        continue
                            last_liveness_check = now
                        # ── timeout: no activity for too long → reclaim ──
                        if task_timeout_s > 0 and now - last_activity_at > task_timeout_s:
                            idle_s = now - last_activity_at
                            log_line(
                                log_path,
                                f"task timeout {active_task}: idle {idle_s:.0f}s > {task_timeout_s:.0f}s; reclaiming",
                            )
                            _reclaim_task_without_signaling_worker(
                                conn,
                                active_task,
                                reason=f"reasonix-interactive idle timeout after {idle_s:.0f}s (limit {task_timeout_s:.0f}s)",
                            )
                            active_task = None
                            active_run_id = None
                            idle_pane_seen_at = None
                            pane_progress_watch = _PaneProgressWatch()
                            last_task_transition_at = time.time()
                            last_hb = 0.0
                            continue
                        if now - last_hb >= max(15.0, min(float(args.ttl) / 3.0, 120.0)):
                            kb.heartbeat_claim(conn, active_task, ttl_seconds=args.ttl, claimer=claim_lock())
                            kb.heartbeat_worker(
                                conn,
                                active_task,
                                note="reasonix-interactive waiting for complete/block from Reasonix TUI",
                                expected_run_id=active_run_id,
                            )
                            last_hb = now
                        time.sleep(min(poll_s, 5.0))
                        continue
                    log_line(log_path, f"active task left running state: {active_task} status={status} run={current_run_id}")
                    active_task = None
                    active_run_id = None
                    idle_pane_seen_at = None
                    pane_progress_watch = _PaneProgressWatch()
                    last_task_transition_at = time.time()
                    last_hb = 0.0
                    zellij_rename_pane(
                        session=args.zellij_session,
                        pane_id=args.zellij_pane_id,
                        name=f"{args.profile}-reasonix listening",
                        log_path=log_path,
                    )

            remaining_delay = _claim_delay_remaining(
                last_transition_at=last_task_transition_at,
                now=time.time(),
                delay_s=float(args.task_boundary_delay_s),
            )
            if remaining_delay > 0.0:
                time.sleep(min(remaining_delay, poll_s, 5.0))
                continue

            screen = _pane_screen(args, log_path=log_path)
            if screen is not None and not _pane_can_accept_new_kanban_task(screen):
                log_line(
                    log_path,
                    "skip claim: Reasonix pane still shows an active/pending Kanban prompt",
                )
                time.sleep(min(poll_s, 5.0))
                continue

            active_task, active_run_id = adopt_orphaned_running_task(args, log_path=log_path)
            if active_task:
                last_activity_at = time.time()
                last_liveness_check = time.time()
                idle_pane_seen_at = None
                pane_progress_watch = _PaneProgressWatch()
                last_hb = 0.0
                continue

            active_task, active_run_id = claim_and_inject_one(args, log_path=log_path)
            if active_task:
                last_activity_at = time.time()
                last_liveness_check = time.time()
                idle_pane_seen_at = None
                pane_progress_watch = _PaneProgressWatch()
                last_hb = 0.0
                if args.once:
                    continue
            else:
                if args.once:
                    log_line(log_path, "no ready task; exiting --once")
                    return 0
                time.sleep(poll_s)
    finally:
        _cleanup_active_claim(board=board, task_id=active_task, run_id=active_run_id, log_path=log_path)
    log_line(log_path, "interactive watcher stopped")
    return 0


def has_saved_sessions(*, workspace: str | None = None) -> bool:
    """Check if reasonix has any saved sessions for the workspace.

    v1.1.0+: no 'sessions' subcommand; check session files directly.
    Sessions are stored in ~/.config/reasonix/sessions/ as JSONL files.
    Session files are named: chat-<dirbasename>-<timestamp>.jsonl
    """
    sessions_dir = Path.home() / ".config" / "reasonix" / "sessions"
    if not sessions_dir.exists():
        # Fallback: check old v0.x location
        sessions_dir = Path.home() / ".reasonix" / "sessions"
    if not sessions_dir.exists():
        return False

    try:
        files = list(sessions_dir.iterdir())
    except OSError:
        return False

    # Filter for non-archived, non-event session files
    # v1.1.0 migrated sessions have 'chat-' prefix; old ones have 'code-' prefix
    for f in files:
        name = f.name
        if not name.endswith(".jsonl"):
            continue
        # Skip archived and event files
        if "__archive_" in name or ".events." in name or name.endswith(".bak"):
            continue
        # If workspace specified, check session matches workspace dir name
        if workspace:
            dir_name = Path(workspace).name
            if f"chat-{dir_name}" in name or f"code-{dir_name}" in name:
                return True
        else:
            # Any non-archived session file counts
            if name.startswith("chat-") or name.startswith("code-"):
                return True
    return False


def _cmd_arg_value(parts: list[str], *names: str) -> str | None:
    for i, part in enumerate(parts):
        for name in names:
            if part == name and i + 1 < len(parts):
                return parts[i + 1]
            if part.startswith(f"{name}="):
                return part.split("=", 1)[1]
    return None


def _same_session_arg(parts: list[str], session_name: str) -> bool:
    """Unused — reasonix has no --session param. Kept as no-op for compat."""
    return False


def other_continue_reasonix_active(session_name: str) -> bool:
    """Return True when another TUI already owns --continue for same workspace.

    Since reasonix has no --session param, sessions are keyed by workspace.
    We check if another reasonix process with --continue is running in the
    same workspace directory.
    """
    current_pid = os.getpid()
    try:
        proc_entries = list(Path("/proc").iterdir())
    except OSError:
        return False
    for entry in proc_entries:
        if not entry.name.isdigit() or int(entry.name) == current_pid:
            continue
        parts = _proc_cmdline(entry.name)
        if not parts:
            continue
        exe_names = {Path(part).name for part in parts[:2]}
        if "reasonix" not in exe_names:
            continue
        if "--continue" not in parts and "-c" not in parts:
            continue
        # v1.1.0+: reasonix chat --dir <workspace> --continue
        # Sessions are keyed by workspace directory.
        # Just check for any other reasonix --continue process.
        return True
    return False


def _read_hermes_dotenv_key(name: str) -> str | None:
    env_path = Path.home() / ".hermes" / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == name:
                return v.strip().strip('"').strip("'")
    except Exception:
        return None
    return None


def _read_reasonix_config_api_key() -> str | None:
    """Read API key from reasonix config.

    v1.1.0+: reads from ~/.env (DEEPSEEK_API_KEY).
    v0.x fallback: reads from ~/.reasonix/config.json apiKey field.
    """
    # v1.1.0+: ~/.env
    env_path = Path.home() / ".env"
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    # v0.x fallback: ~/.reasonix/config.json
    config_path = Path.home() / ".reasonix" / "config.json"
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        return data.get("apiKey") or data.get("api_key")
    except Exception:
        return None


def default_model_for_reasonix() -> str:
    """Return default model for Reasonix."""
    return "deepseek-v4-flash"


def build_reasonix_cmd(args: argparse.Namespace, *, role_instructions_path: Path | None = None) -> list[str]:
    """Build Reasonix command line.

    Reasonix v1.1.0+ (Go rewrite) uses:
    - reasonix chat --dir <workspace>  (replaces 'reasonix code <workspace>')
    - system_prompt_file in reasonix.toml  (replaces --system-append-file)
    - --model <id> for model selection
    - --continue to resume most recent session
    - --yolo for auto-approve (kanban mode, replaces --no-mouse/--no-dashboard)

    Removed in v1.1.0:
    - --system-append-file → use system_prompt_file in reasonix.toml
    - --no-mouse / --no-dashboard → no longer needed
    - --new → just omit --continue (new session is default)
    - 'code' subcommand → use 'chat'
    """
    workspace = str(Path(args.workspace).expanduser().resolve())
    profile = args.profile or "implementer"

    # v1.1.0+: 'chat' subcommand with --dir flag
    cmd = [args.reasonix_bin, "chat", "--dir", workspace]

    # Role instructions are now injected via reasonix.toml's system_prompt_file,
    # written by write_workspace_reasonix_toml() before this function is called.
    # No --system-append-file flag needed.

    # Add model if specified
    if args.model:
        cmd.extend(["--model", args.model])

    # Kanban mode: use --yolo to auto-approve tool calls
    # (replaces --no-mouse + --no-dashboard from v0.x)
    cmd.append("--yolo")

    # Session continuation
    # v1.1.0+: --continue exits rc=1 if no compatible sessions exist.
    # Migrated v0.x session files are NOT recognized by v1.1.0's --continue.
    # For kanban mode, always start a fresh session (omitting --continue):
    # - Kanban sessions are ephemeral; the watcher injects tasks each time.
    # - --continue with incompatible sessions causes rc=1, crashing the listener.
    # - Once v1.1.0-native sessions accumulate, --continue will work,
    #   but kanban workers don't need session persistence anyway.
    # (args.continue_session is intentionally ignored for kanban mode)

    for extra in args.reasonix_arg or []:
        cmd.append(extra)
    return cmd


def launcher_main(args: argparse.Namespace) -> int:
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    if not workspace.exists():
        print(f"错误: workspace 不存在: {workspace}", file=sys.stderr)
        return 2

    zellij_session = args.zellij_session or os.environ.get("ZELLIJ_SESSION_NAME")
    zellij_pane_id = args.zellij_pane_id or os.environ.get("ZELLIJ_PANE_ID")
    if not zellij_session or not zellij_pane_id:
        print("错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 interactive Reasonix。", file=sys.stderr)
        print("请在 zellij pane 内运行，或显式传 --zellij-session / --zellij-pane-id。", file=sys.stderr)
        return 2

    log_path = kb.worker_logs_dir(board=board) / f"reasonix-interactive-{args.profile}.log"
    reasonix_model = args.model or default_model_for_reasonix()

    # Write role instructions file for system_prompt_file (reasonix.toml)
    role_instructions_path = write_role_instructions_file(
        workspace=workspace,
        board=board,
        pane_profile=args.profile,
    )
    # Write workspace reasonix.toml with system_prompt_file pointing to role instructions
    write_workspace_reasonix_toml(
        workspace=workspace,
        role_instructions_path=role_instructions_path,
    )

    def build_env(*, task_delivery: str) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "HERMES_KANBAN_BOARD": board,
                "HERMES_KANBAN_PROFILE": args.profile,
                "HERMES_KANBAN_CLAIM_ASSIGNEES": ",".join(claim_assignees(args)),
                "HERMES_KANBAN_WORKSPACE": str(workspace),
                "HERMES_KANBAN_TASK_DELIVERY": task_delivery,
            }
        )
        # Reasonix uses REASONIX_API_KEY or ~/.reasonix/config.json
        reasonix_key = env.get("REASONIX_API_KEY") or _read_hermes_dotenv_key("REASONIX_API_KEY")
        if not reasonix_key:
            reasonix_key = _read_reasonix_config_api_key()
        if reasonix_key:
            env["REASONIX_API_KEY"] = reasonix_key
        return env

    delivery = getattr(args, "task_delivery", "inject")
    if delivery == "self-poll":
        prompt_path, startup_prompt = write_self_poll_startup_prompt(
            board=board,
            profile=args.profile,
            claim_assignees=claim_assignees(args),
            workspace=workspace,
            ttl=args.ttl,
            listener_kind="reasonix-self-poll",
            pane_id=zellij_pane_id,
        )
        print("Reasonix self-poll kanban mode")
        print(f"  board:          {board}")
        print(f"  profile:        {args.profile}")
        print(f"  claims:         {', '.join(claim_assignees(args))}")
        print(f"  workspace:      {workspace}")
        print(f"  pane:           {zellij_session}:{zellij_pane_id}")
        print(f"  model:          {reasonix_model}")
        print(f"  startup prompt: {prompt_path}")
        env = build_env(task_delivery="self-poll")
        env["HERMES_KANBAN_SELF_POLL_PROMPT"] = str(prompt_path)
        env["HERMES_KANBAN_SELF_POLL_OWNER"] = worker_runtime.default_self_poll_owner(
            profile=args.profile,
            listener_kind="reasonix-self-poll",
            pane_id=zellij_pane_id,
        )
        if args.watch_only:
            ok = zellij_inject(
                session=zellij_session,
                pane_id=zellij_pane_id,
                text=startup_prompt,
                log_path=log_path,
            )
            return 0 if ok else 1
        reasonix_cmd = build_reasonix_cmd(args, role_instructions_path=role_instructions_path)
        log_line(log_path, f"launcher starting reasonix self-poll: {' '.join(reasonix_cmd)}")
        proc = subprocess.Popen(reasonix_cmd, cwd=str(workspace), env=env)
        time.sleep(max(0.0, float(args.startup_delay_s or 0.0)))
        zellij_inject(
            session=zellij_session,
            pane_id=zellij_pane_id,
            text=startup_prompt,
            log_path=log_path,
        )
        return int(proc.wait())

    watcher_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--watch-child",
        "--profile",
        args.profile,
        "--claim-assignees",
        ",".join(claim_assignees(args)),
        "--board",
        board,
        "--workspace",
        str(workspace),
        "--ttl",
        str(args.ttl),
        "--zellij-session",
        zellij_session,
        "--zellij-pane-id",
        zellij_pane_id,
        "--startup-delay-s",
        str(args.startup_delay_s),
        "--task-boundary-delay-s",
        str(args.task_boundary_delay_s),
        "--task-timeout-s",
        str(args.task_timeout_s),
        "--idle-pane-reclaim-s",
        str(args.idle_pane_reclaim_s),
        "--assist-claim-delay-s",
        str(args.assist_claim_delay_s),
    ]
    for spec in _delay_specs(getattr(args, "assist_claim_delay_for", None)):
        watcher_cmd.extend(["--assist-claim-delay-for", spec])
    for spec in _delay_specs(getattr(args, "assist_claim_profile_delay", None)):
        watcher_cmd.extend(["--assist-claim-profile-delay", spec])
    if args.poll is not None:
        watcher_cmd.extend(["--poll", str(args.poll)])

    poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
    poll_label = f"{poll_s:g}s" + (" override" if args.poll is not None else " shared-policy")

    print("Reasonix interactive kanban mode")
    print(f"  board:     {board}")
    print(f"  profile:   {args.profile}")
    print(f"  claims:    {', '.join(claim_assignees(args))}")
    print(f"  workspace: {workspace}")
    print(f"  pane:      {zellij_session}:{zellij_pane_id}")
    print(f"  log:       {log_path}")
    print(f"  model:     {reasonix_model}")
    print("")
    print(f"按 Enter 进入 interactive Reasonix；后台 listener 会按优先级 claim {', '.join(claim_assignees(args))} ready 任务并注入到当前 Reasonix。")
    print("退出 Reasonix 后 listener 会一起停止；如果任务未 complete/block，会自动 reclaim。")
    print(f"reasonix-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace} model={reasonix_model}")

    if args.watch_only:
        print("listener-only 模式：不会启动 Reasonix，只运行后台 listener 并向指定 Zellij pane 注入任务。")
        return watcher_main(args)

    try:
        input()
    except EOFError:
        pass

    env = build_env(task_delivery="inject")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a", encoding="utf-8")

    def start_watcher(*, restart_reason: str | None = None) -> subprocess.Popen[str]:
        if restart_reason:
            log_line(log_path, f"launcher restarting watcher: {restart_reason}")
        log_line(log_path, f"launcher starting watcher: {' '.join(watcher_cmd)}")
        proc = subprocess.Popen(
            watcher_cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        print(
            f"reasonix-kanban listener started: pid={proc.pid} profile={args.profile} "
            f"board={board} poll={poll_label} pane={zellij_session}:{zellij_pane_id} log={log_path}",
            flush=True,
        )
        return proc

    watcher = start_watcher()
    reasonix_cmd = build_reasonix_cmd(args, role_instructions_path=role_instructions_path)
    log_line(log_path, f"launcher starting reasonix: {' '.join(reasonix_cmd)}")
    rc = 0
    try:
        reasonix_proc = subprocess.Popen(reasonix_cmd, cwd=str(workspace), env=env)
        watcher_restart_count = 0
        while True:
            reasonix_rc = reasonix_proc.poll()
            if reasonix_rc is not None:
                rc = int(reasonix_rc)
                break

            if watcher is not None:
                watcher_rc = watcher.poll()
                if watcher_rc is not None:
                    if not _should_restart_watcher(watcher_rc):
                        log_line(
                            log_path,
                            f"watcher exited cleanly rc={watcher_rc} while Reasonix is still running; not restarting",
                        )
                        watcher = None
                        continue
                    watcher_restart_count += 1
                    log_line(
                        log_path,
                        f"watcher exited unexpectedly rc={watcher_rc} while Reasonix is still running; "
                        f"restart_count={watcher_restart_count}",
                    )
                    time.sleep(min(30.0, 2.0 * watcher_restart_count))
                    watcher = start_watcher(
                        restart_reason=f"previous watcher exited rc={watcher_rc}"
                    )

            time.sleep(2.0)
    finally:
        watcher_pid = watcher.pid if watcher is not None else None
        log_line(log_path, f"reasonix exited rc={rc}; stopping watcher pid={watcher_pid}")
        try:
            if watcher is not None and watcher.poll() is None:
                watcher.terminate()
                watcher.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if watcher is not None:
                watcher.kill()
                watcher.wait(timeout=5)
        log_f.close()
    return int(rc)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive Reasonix while listening to Hermes Kanban")
    parser.add_argument("--profile", default=os.environ.get("HERMES_PROFILE") or "implementer", help="Kanban assignee/profile to claim (default: implementer)")
    parser.add_argument("--claim-assignees", default=os.environ.get("HERMES_KANBAN_CLAIM_ASSIGNEES"), help="Comma-separated Kanban assignees this pane may claim, in priority order; defaults to --profile")
    parser.add_argument("--board", default=os.environ.get("HERMES_KANBAN_BOARD"), help="Board slug; defaults to Hermes current board")
    parser.add_argument("--workspace", default=os.environ.get("REASONIX_KANBAN_WORKSPACE") or os.getcwd(), help="Reasonix workspace/git repo root")
    parser.add_argument("--poll", type=float, default=None, help="Ready-task poll interval override")
    parser.add_argument("--ttl", type=int, default=listener_policy.LISTENER_HEALTH_CLAIM_TTL_SECONDS, help="Claim TTL seconds")
    parser.add_argument("--startup-delay-s", type=float, default=8.0, help="Delay after Reasonix launch before injecting first task")
    parser.add_argument(
        "--task-delivery",
        choices=("inject", "self-poll"),
        default=os.environ.get("HERMES_KANBAN_TASK_DELIVERY") or "self-poll",
        help="Task delivery mode: one-time self-poll startup prompt, or legacy per-task zellij injection.",
    )
    parser.add_argument("--task-boundary-delay-s", type=float, default=8.0, help="Delay after a task leaves running state before claiming/injecting the next task")
    parser.add_argument("--task-timeout-s", type=float, default=listener_policy.INTERACTIVE_TASK_TIMEOUT_SECONDS, help="Dynamic idle timeout: reclaim task after this many seconds of continuous TUI inactivity (session files not updating). Reset on each TUI output. 0 to disable.")
    parser.add_argument("--idle-pane-reclaim-s", type=float, default=listener_policy.INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS, help="Reclaim a running task only after the target Reasonix pane visibly stays idle for this many seconds. 0 to disable.")
    parser.add_argument("--assist-claim-delay-s", type=float, default=float(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_DELAY_S") or 0.0), help="Only claim non-primary assist assignees after their ready task has waited this many seconds. 0 disables the delay.")
    parser.add_argument(
        "--assist-claim-delay-for",
        action="append",
        default=_delay_specs(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_DELAYS")),
        metavar="ASSIGNEE=SECONDS",
        help="Override assist claim delay for a target assignee in this listener, e.g. implementer=180. Repeatable.",
    )
    parser.add_argument(
        "--assist-claim-profile-delay",
        action="append",
        default=_delay_specs(os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_PROFILE_DELAYS")),
        metavar="PROFILE:ASSIGNEE=SECONDS",
        help="Profile-qualified assist delay rule, e.g. backup_implementer:implementer=300. Repeatable.",
    )
    parser.add_argument("--model", default=None, help="Optional Reasonix model override (e.g. deepseek-v4-pro, deepseek-v4-flash)")
    parser.add_argument("--reasonix-bin", default="reasonix", help="reasonix binary path/name (default: reasonix)")
    parser.add_argument("--reasonix-session", default=None, help="Reasonix session name (default: same as --profile)")
    parser.add_argument("--reasonix-arg", action="append", default=[], help="Extra raw arg passed to reasonix; repeatable")
    parser.add_argument("--continue", dest="continue_session", action="store_true", default=True, help="Resume most recent session for workspace (default: True)")
    parser.add_argument("--no-continue", dest="continue_session", action="store_false", help="Start a new session instead of resuming")
    parser.add_argument("--zellij-session", default=os.environ.get("ZELLIJ_SESSION_NAME"), help="Target Zellij session for task injection")
    parser.add_argument("--zellij-pane-id", default=os.environ.get("ZELLIJ_PANE_ID"), help="Target Zellij pane id for task injection")
    parser.add_argument("--watch-only", action="store_true", help="Only run the background listener; do not launch interactive Reasonix")
    parser.add_argument("--watch-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--reset-kanban", action="store_true", help="Reclaim this profile/pane's running interactive Kanban task(s) and exit")
    parser.add_argument("--once", action="store_true", help="Watcher child: process at most one task")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset_kanban:
        board = args.board or kb.get_current_board() or "default"
        workspace = Path(args.workspace).expanduser().resolve()
        reset_ids = reset_kanban_claims(
            board=board,
            profile=args.profile,
            claim_assignees=claim_assignees(args),
            workspace=workspace,
        )
        if reset_ids:
            print(f"reset-kanban reclaimed: {', '.join(reset_ids)}")
        else:
            print("reset-kanban: no matching running claim")
        return 0
    if args.watch_child:
        return watcher_main(args)
    return launcher_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
