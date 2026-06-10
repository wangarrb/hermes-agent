#!/usr/bin/env python3
"""Base classes and shared utilities for Hermes Kanban listeners.

All agent-backed listeners (CodeWhale, Codex, Reasonix) share the same
poll/claim/inject lifecycle.  This module extracts the common base so
that each listener only provides its own:

  - binary name / command construction
  - environment setup
  - result parsing
  - process detection (for interactive mode)

Two base classes are provided:

  ``BaseKanbanListener``   — non-interactive (headless exec) listener
  ``BaseInteractiveListener`` — interactive (visible TUI) listener
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

# Source layout: <repo>/plugins/kanban/base_listener.py
HERMES_REPO = Path(__file__).resolve().parents[1]
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402


# ──────────────────────────────────────────────
# Shared exceptions / globals
# ──────────────────────────────────────────────

class ListenerStopped(Exception):
    """Raised when a SIGINT/SIGTERM requests a graceful stop."""
    pass


_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


# ──────────────────────────────────────────────
# Shared utility functions
# ──────────────────────────────────────────────

def now_s() -> int:
    return int(time.time())


def now_label() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def log_line(log_path: Path, msg: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{now_label()}] {msg}\n")


def task_log_path(task_id: str, board: str | None) -> Path:
    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{task_id}.log"


def _noop_signal(_pid: int, _sig: int) -> None:
    pass


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _wait_agent_idle(tui_pid: int, *, task_id: str = "", max_wait: float = 10.0, board: str = "") -> bool:
    """Block until the TUI agent at *tui_pid* appears idle."""
    deadline = time.time() + max_wait
    waited = 0.0
    sample = listener_policy.AGENT_IDLE_SAMPLE_SECONDS
    while time.time() < deadline:
        if not listener_policy.agent_pid_is_busy(tui_pid, sample_s=sample):
            return True
        waited += sample
        if waited >= 3.0:
            tag = f" [{task_id}]" if task_id else ""
            log(f"waiting for agent to become idle{tag} ({waited:.0f}s elapsed)")
    _reclaim_and_release(task_id, board=board, reason=f"agent busy for {max_wait:.0f}s, releasing for other listeners")
    return False


def _reclaim_and_release(task_id: str, *, board: str, reason: str) -> None:
    try:
        with kb.connect(board=board) as conn:
            kb.reclaim_task(conn, task_id, reason=reason)
        log(f"released {task_id}: {reason}")
    except Exception as exc:
        log(f"failed to release {task_id}: {exc}")


def _pid_alive(pid: int | None) -> bool:
    """Check whether a PID is still alive (via /proc)."""
    if not pid:
        return False
    try:
        return Path(f"/proc/{pid}").exists()
    except OSError:
        return False


def _cooldown_heartbeat_interval(ttl_s: int) -> float:
    return max(5.0, ttl_s * 0.4)


# ──────────────────────────────────────────────
# Shared role guidance (used by all listeners)
# ──────────────────────────────────────────────

def role_guidance(profile: str) -> str:
    """Role-bound guidance shared by all agent backends."""
    p = (profile or "").strip().lower()
    common = "职责由 Kanban profile/assignee 决定，而不是由底层 agent 类型决定；即使用不同 agent 运行，也要按当前角色工作。"
    per_role = {
        "coordinator": "你是 coordinator：和用户对齐目标，拆分任务，维护 Kanban 流转；除非任务明确很小，否则不要替 planner/implementer/critic 做大段执行。",
        "planner": "你是 planner：负责方案设计、实验计划和任务拆分。输出必须具体到文件路径、函数/类名、命令、预期结果和验收标准。",
        "implementer": "你是 implementer：负责落地执行。先读上下文和相关代码，再小步修改；改完运行最小可行验证，并在结果里说明改了什么、如何验证。",
        "critic": "你是 critic：负责审查、找漏洞和独立验证。不要默认相信 planner/implementer 结论；重点检查证据链、遗漏风险、指标口径和可复现性。最终放行只有 PASS/FAIL，没有带病通过；所有任务范围内能解决的问题都解决后才能 PASS，否则必须 FAIL 并列出具体修复/复审要求。NO_CLAIM 只表示指标不支持 claim，不是缺陷豁免。",
    }
    return common + "\n" + per_role.get(p, f"你当前角色是 {profile}：按该 assignee 的职责完成任务，不要因为运行在不同 agent 中而改变角色边界。")


# ──────────────────────────────────────────────
# Shared build_prompt (used by non-interactive listeners)
# ──────────────────────────────────────────────

def build_prompt(*, agent_name: str, board: str, profile: str, task_id: str,
                 task_assignee: str, context: str, workspace: Path) -> str:
    """Build the Kanban task prompt injected into the agent."""
    assist_note = ""
    if task_assignee != profile:
        assist_note = (
            f"\n        当前 listener profile 是 {profile}，但本任务 assignee/role 是 {task_assignee}。"
            f"本次必须按 {task_assignee} 职责完成，不要按 {profile} 职责改写目标。\n"
        )
    return textwrap.dedent(
        f"""
        你现在是 Hermes Kanban 中的 {agent_name} profile：{profile}。
        你正在执行 board={board} 上的任务 {task_id}。
        Task assignee/role: {task_assignee}。
        {assist_note}

        关键规则：
        1. 完成任务后**必须**调用 `hermes kanban --board {board} complete {task_id} --summary "..."` 完成，或遇阻塞时调用 `hermes kanban --board {board} block {task_id} --reason "..."` 然后 `/exit` 退出。
        2. 如果需要查看任务/父任务/评论，可用：`hermes kanban --board {board} show <task_id>` 或 `hermes kanban --board {board} context <task_id>`。
        3. 如果确实需要创建后继任务，用：`hermes kanban --board {board} create ... --json`，并在退出前告知 listener（通过 comment 或 result summary）。
        4. 默认中文输出；技术名词/路径/命令保留英文。
        5. 务实、简洁、可验证。planner 任务重点给方案、步骤、风险、验收标准；不要做不必要的代码改动。
        6. 当前工作目录是：{workspace}

        **完成后务必调用 Kanban 工具完成任务：**
        - 成功: `hermes kanban --board {board} complete {task_id} --summary "一句话总结"`
        - 阻塞: `hermes kanban --board {board} block {task_id} --reason "阻塞原因"`
        - 然后输入 `/exit` 或 Ctrl+D 退出，listener 会自动回收。

        下面是 Hermes Kanban worker context：

        {context}
        """
    ).strip()


# ──────────────────────────────────────────────
# Shared interactive prompt (used by interactive listeners)
# ──────────────────────────────────────────────

def build_interactive_prompt(
    *,
    agent_name: str,
    board: str,
    profile: str,
    task_id: str,
    task_assignee: str,
    task_title: str,
    context: str,
    workspace: Path,
) -> str:
    """Build the short instruction injected into a running agent pane."""
    assist_note = ""
    if task_assignee != profile:
        assist_note = f"（注意：当前 profile={profile}，但本任务 assignee={task_assignee}，按 {task_assignee} 职责执行）"

    return textwrap.dedent(
        f"""
        ─── Kanban 任务 {task_id} ───
        标题：{task_title}
        角色：{task_assignee} {assist_note}

        关键规则：
        1. 完成后调用 `hermes kanban --board {board} complete {task_id} --summary "..."`
        2. 阻塞时调用 `hermes kanban --board {board} block {task_id} --reason "..."`
        3. 默认中文；路径/命令保留英文

        上下文：
        {context}

        ─── 开始执行任务 {task_id} ───
        """
    ).strip()


# ──────────────────────────────────────────────
# Shared prompt file writers
# ──────────────────────────────────────────────

def prompt_dir(workspace: Path, board: str, pane_profile: str, *, agent_slug: str) -> Path:
    safe_board = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in board)
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pane_profile)
    return workspace / f".{agent_slug}-kanban" / safe_board / safe_profile


def write_task_prompt(
    *,
    agent_name: str,
    agent_slug: str,
    board: str,
    profile: str,
    task_id: str,
    task_assignee: str,
    task_title: str,
    context: str,
    workspace: Path,
) -> Path:
    d = prompt_dir(workspace, board, profile, agent_slug=agent_slug)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"task-{task_id}.md"
    p.write_text(
        build_interactive_prompt(
            agent_name=agent_name,
            board=board,
            profile=profile,
            task_id=task_id,
            task_assignee=task_assignee,
            task_title=task_title,
            context=context,
            workspace=workspace,
        ),
        encoding="utf-8",
    )
    return p


# ──────────────────────────────────────────────
# Shared zellij helpers
# ──────────────────────────────────────────────

def zellij_inject(*, session: str, pane_id: str, text: str, log_path: Path) -> None:
    """Inject text into a Zellij pane."""
    try:
        subprocess.run(
            ["zellij", "action", "write-chars", "-p", pane_id, text],
            timeout=5, check=False,
        )
    except Exception as exc:
        log_line(log_path, f"zellij inject failed: {exc}")


def zellij_rename_pane(*, session: str, pane_id: str, name: str, log_path: Path) -> None:
    """Rename a Zellij pane."""
    try:
        subprocess.run(
            ["zellij", "action", "rename-pane", "-p", pane_id, name],
            timeout=5, check=False,
        )
    except Exception as exc:
        log_line(log_path, f"zellij rename-pane failed: {exc}")


def zellij_dump_screen(*, session: str, pane_id: str, log_path: Path) -> str:
    """Dump the current screen content of a Zellij pane."""
    try:
        result = subprocess.run(
            ["zellij", "action", "dump-screen", "-p", pane_id],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout or ""
    except Exception as exc:
        log_line(log_path, f"zellij dump-screen failed: {exc}")
        return ""


# ──────────────────────────────────────────────
# Shared claim_assignees / reset / delay helpers
# ──────────────────────────────────────────────

def claim_assignees(args: argparse.Namespace, *, default_profile: str = "implementer") -> list[str]:
    return worker_runtime.claim_assignees_from_args(args, default_profile=default_profile)


def reset_kanban_claims(
    *,
    board: str,
    profile: str,
    claim_assignees: list[str],
    workspace: Path,
    listener_kind: str,
    reason: str = "operator reset-kanban",
) -> list[str]:
    reset_ids = worker_runtime.reset_interactive_claims(
        board=board,
        profile=profile,
        claim_assignees=claim_assignees,
        workspace=workspace,
        listener_kind=listener_kind,
        reason=reason,
    )
    return list(dict.fromkeys(reset_ids))


def assist_claim_delay_s(args: argparse.Namespace, *, default_profile: str = "implementer") -> float:
    return worker_runtime.claim_policy_from_args(args, default_profile=default_profile).assist_claim_delay_s


def assist_claim_delays(args: argparse.Namespace) -> dict[str, float]:
    return worker_runtime.assist_claim_delays_from_args(args)


def assist_claim_profile_delays(args: argparse.Namespace) -> dict[tuple[str, str], float]:
    return worker_runtime.assist_claim_profile_delays_from_args(args)


def assist_claim_delay_for(args: argparse.Namespace, assignee: str, *, default_profile: str = "implementer") -> float:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    return worker_runtime.assist_claim_delay_for(policy, assignee)


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


def _ready_since(conn, task_id: str, fallback_created_at: int) -> int:
    return worker_runtime.ready_since(conn, task_id, fallback_created_at)


def _assist_candidate_ready(conn, args: argparse.Namespace, task: kb.Task,
                             assignee: str, *, default_profile: str = "implementer") -> bool:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    return worker_runtime.assist_candidate_ready(conn, policy=policy, task=task, assignee=assignee)


def _select_ready_candidate(conn, args: argparse.Namespace, *, default_profile: str = "implementer") -> kb.Task | None:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    return worker_runtime.select_ready_candidate(conn, policy=policy)


# ──────────────────────────────────────────────
# Shared result sanitization
# ──────────────────────────────────────────────

def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a parsed agent result dict into a canonical form."""
    status = str(result.get("status") or "done").strip().lower()
    if status not in {"done", "blocked"}:
        status = "done"
    summary = str(result.get("summary") or "").strip()
    if not summary:
        if status == "blocked":
            summary = "Agent blocked without summary"
        else:
            summary = "Agent completed without summary"
    details = str(result.get("details") or "").strip()
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    block_reason = str(result.get("block_reason") or "").strip() if status == "blocked" else ""
    return {
        "status": status,
        "summary": summary,
        "details": details,
        "metadata": metadata,
        **({"block_reason": block_reason} if block_reason else {}),
    }


def _result_failure_text(result: dict[str, Any]) -> str:
    return str(result.get("block_reason") or result.get("summary") or "unknown failure")


def _is_provider_failure_result(rc: int, result: dict[str, Any]) -> bool:
    """Heuristic: did the agent fail because of a provider/API error?"""
    if rc == 0:
        return False
    text = _result_failure_text(result).lower()
    provider_keywords = ["rate limit", "api key", "429", "503", "quota", "provider", "authentication", "unauthorized"]
    return any(kw in text for kw in provider_keywords)


def _task_status(conn, task_id: str) -> tuple[str | None, int | None]:
    row = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if not row:
        return None, None
    return row["status"], row["current_run_id"]


# ──────────────────────────────────────────────
# Shared tail / idle detection helpers
# ──────────────────────────────────────────────

def _tail_nonempty_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-limit:] if len(lines) > limit else lines
