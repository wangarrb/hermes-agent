#!/usr/bin/env python3
"""Base class for all Hermes Kanban interactive listeners.

All agent-backed listeners (Codex, CodeWhale, Claude, Reasonix) share the
same poll/claim/inject lifecycle.  This module provides BaseInteractiveListener
which implements the complete watcher+launcher architecture; each subclass
only provides agent-specific methods:

  - agent_name / agent_slug (identity)
  - build_tui_cmd (how to launch the TUI)
  - has_saved_sessions (session detection)
  - inject_text (what text to inject into the pane)
  - pane_label (pane title formatting)

Optional hooks for further customization:

  - on_claim_pre_check / on_claim_post_confirm
  - on_task_running_monitor / on_watcher_loop_idle
  - build_launch_env / build_watcher_extra_args
  - idle_markers / busy_markers / queued_input_markers

IMPORTANT: inject_text must NEVER contain ``\\n`` (LF, 0x0A).
In PTY raw mode, LF is NOT the same as Enter (CR, 0x0D).
LF inserts a newline in the input buffer but does NOT submit.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sqlite3
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
# Exceptions / globals
# ──────────────────────────────────────────────

_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


# ──────────────────────────────────────────────
# Shared utility functions (used by all listeners)
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


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        return Path(f"/proc/{pid}").exists()
    except OSError:
        return False


def _task_status(conn: Any, task_id: str) -> tuple[str | None, int | None]:
    row = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id=?", (task_id,)
    ).fetchone()
    if not row:
        return None, None
    return row["status"], row["current_run_id"]


def _workspace_matches(row_workspace: str | None, workspace: Path) -> bool:
    if not row_workspace:
        return False
    return str(workspace) == row_workspace


def _skip_reclaim_signal(pid: int, signum: int) -> None:  # noqa: ARG001
    pass


def _reclaim_task_without_signaling_worker(
    conn: Any, task_id: str, *, reason: str
) -> bool:
    """Reclaim a task without sending SIGTERM to its worker process."""
    try:
        return kb.reclaim_task(conn, task_id, reason=reason, signal_fn=_skip_reclaim_signal)
    except Exception:
        return False


def _cleanup_active_claim(
    *, board: str, task_id: str | None, run_id: int | None, log_path: Path
) -> None:
    """On watcher exit, if a task is still running, reclaim it."""
    if not task_id:
        return
    try:
        with kb.connect(board=board) as conn:
            status, _ = _task_status(conn, task_id)
            if status == "running":
                kb.reclaim_task(
                    conn, task_id,
                    reason="watcher exited while task still running",
                    signal_fn=_skip_reclaim_signal,
                )
                log_line(log_path, f"reclaimed {task_id} on watcher exit")
    except Exception as exc:
        log_line(log_path, f"cleanup reclaim failed: {exc}")


# ──────────────────────────────────────────────
# Shared role guidance
# ──────────────────────────────────────────────

def role_guidance(profile: str) -> str:
    """Role-bound guidance shared by all agent backends."""
    p = (profile or "").strip().lower()
    common = "职责由 Kanban profile/assignee 决定，而不是由底层 agent 类型决定；即使用不同 agent 运行，也要按当前角色工作。"
    per_role = {
        "coordinator": "你是 coordinator：和用户对齐目标，拆分任务，维护 Kanban 流转；除非任务明确很小，否则不要替 planner/implementer/critic 做大段执行。",
        "planner": "你是 planner：负责方案设计、实验计划和任务拆分。输出必须具体到文件路径、函数/类名、命令、预期结果和验收标准。注意 reviewer 会独立制定计划并审核你的方案，你们需要多轮协商才能敲定最终计划——你应当在方案中充分说明假设和取舍理由，方便 reviewer 对比和补充。reviewer 反馈后，你负责修改计划并再次提交审核，直到双方达成一致。",
        "reviewer": "你是 reviewer：既能独立制定计划，也能审核 planner 的计划，与 planner 多轮协商直到敲定最终计划。你的职责不是找 planner 的纰漏，而是从全局角度把控计划的方向、范围和内容是否合理、完整、有效。具体来说：(1) 方向——计划是否在解决正确的问题？是否与项目目标对齐？有没有偏离核心目标做无关优化？(2) 范围——计划的边界是否清晰？哪些该做哪些不该做？有没有遗漏的关键路径或不需要的过度设计？(3) 内容——方案是否完整覆盖目标？假设是否成立？验收标准是否可测试无歧义？依赖和风险是否充分识别？有无更简单可靠的替代方案？工作流程：(a) 收到 planner 的计划后，先独立思考同一目标你会怎么做——形成自己的计划草案；(b) 从全局视角对比两份计划，找出方向偏差、范围遗漏、内容缺陷；(c) 通过 kanban comment 反馈你的审核意见、独立方案和修改建议；(d) planner 根据你的反馈修改计划后，再次审核——可能需要多轮协商才能达成一致；(e) 双方认可后，最终计划交给 implementer 执行。不要为了结束协商而妥协——真正有分歧的点必须充分讨论清楚。",
        "implementer": "你是 implementer：负责落地执行。先读上下文和相关代码，再小步修改；改完运行最小可行验证，并在结果里说明改了什么、如何验证。",
        "critic": "你是 critic：负责审查、找漏洞和独立验证。不要默认相信 planner/implementer 结论；重点检查证据链、遗漏风险、指标口径和可复现性。",
    }
    return common + "\n" + per_role.get(p, f"你当前角色是 {profile}：按该 assignee 的职责完成任务。")


# ──────────────────────────────────────────────
# Shared prompt builders
# ──────────────────────────────────────────────

def build_interactive_prompt(
    *, agent_name: str, board: str, profile: str, task_id: str,
    task_assignee: str, task_title: str, context: str, workspace: Path,
) -> str:
    """Build the full task prompt (written to a file, not injected directly)."""
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


def prompt_dir(workspace: Path, board: str, pane_profile: str, *, agent_slug: str) -> Path:
    safe_board = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in board)
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in pane_profile)
    return workspace / f".{agent_slug}-kanban" / safe_board / safe_profile


def write_task_prompt(
    *, agent_name: str, agent_slug: str, board: str, profile: str,
    task_id: str, task_assignee: str, task_title: str, context: str,
    workspace: Path,
) -> Path:
    d = prompt_dir(workspace, board, profile, agent_slug=agent_slug)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"task-{task_id}.md"
    p.write_text(
        build_interactive_prompt(
            agent_name=agent_name, board=board, profile=profile,
            task_id=task_id, task_assignee=task_assignee,
            task_title=task_title, context=context, workspace=workspace,
        ),
        encoding="utf-8",
    )
    return p


# ──────────────────────────────────────────────
# Shared zellij helpers
# ──────────────────────────────────────────────

def zellij_inject(*, session: str, pane_id: str, text: str, log_path: Path) -> bool:
    """Inject text into a Zellij pane and press Enter.

    IMPORTANT: *text* must NOT contain ``\\n`` (LF) characters.
    In PTY raw mode, LF (0x0A) is NOT the same as Enter (CR, 0x0D).
    LF inserts a newline in the input buffer but does NOT submit the
    prompt, causing the "typed but not sent" bug.
    Use single-line text only; the TUI will word-wrap.
    """
    try:
        cmd_base = ["zellij", "--session", session, "action"] if session else ["zellij", "action"]
        subprocess.run(
            cmd_base + ["write-chars", "-p", str(pane_id), text],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.8)
        subprocess.run(
            cmd_base + ["send-keys", "-p", str(pane_id), "Enter"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log_line(log_path, f"zellij injection failed rc={exc.returncode}: {exc.stderr.strip()}")
        return False


def zellij_rename_pane(*, session: str, pane_id: str, name: str, log_path: Path) -> bool:
    try:
        cmd_base = ["zellij", "--session", session, "action"] if session else ["zellij", "action"]
        subprocess.run(
            cmd_base + ["rename-pane", "-p", str(pane_id), name],
            check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        return True
    except subprocess.CalledProcessError as exc:
        log_line(log_path, f"zellij rename-pane failed rc={exc.returncode}: {exc.stderr.strip()}")
        return False


def zellij_dump_screen(*, session: str, pane_id: str, log_path: Path) -> str | None:
    """Dump the current screen content of a Zellij pane."""
    try:
        cmd_base = ["zellij", "--session", session, "action"] if session else ["zellij", "action"]
        result = subprocess.run(
            cmd_base + ["dump-screen", "-p", str(pane_id), "--full"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout or ""
    except Exception as exc:
        log_line(log_path, f"zellij dump-screen failed: {exc}")
        return None


# ──────────────────────────────────────────────
# Shared claim / delay / reset helpers
# ──────────────────────────────────────────────

def claim_assignees(args: argparse.Namespace, *, default_profile: str = "implementer") -> list[str]:
    return worker_runtime.claim_assignees_from_args(args, default_profile=default_profile)


def reset_kanban_claims(
    *, board: str, profile: str, claim_assignees_list: list[str],
    workspace: Path, listener_kind: str, reason: str = "operator reset-kanban",
) -> list[str]:
    reset_ids = worker_runtime.reset_interactive_claims(
        board=board, profile=profile,
        claim_assignees=claim_assignees_list,
        workspace=workspace, listener_kind=listener_kind, reason=reason,
    )
    return list(dict.fromkeys(reset_ids))


def assist_claim_delay_s(args: argparse.Namespace, *, default_profile: str = "implementer") -> float:
    return worker_runtime.claim_policy_from_args(args, default_profile=default_profile).assist_claim_delay_s


def _delay_specs(raw: Any) -> list[str]:
    return worker_runtime.split_csv_values(raw)


def assist_claim_delays(args: argparse.Namespace) -> dict[str, float]:
    return worker_runtime.assist_claim_delays_from_args(args)


def assist_claim_delay_for(args: argparse.Namespace, assignee: str, *, default_profile: str = "implementer") -> float:
    policy = worker_runtime.claim_policy_from_args(args, default_profile=default_profile)
    return worker_runtime.assist_claim_delay_for(policy, assignee)


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
    status = str(result.get("status") or "done").strip().lower()
    if status not in {"done", "blocked"}:
        status = "done"
    summary = str(result.get("summary") or "").strip()
    if not summary:
        summary = "Agent completed without summary" if status != "blocked" else "Agent blocked without summary"
    details = str(result.get("details") or "").strip()
    metadata = result.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": str(metadata)}
    block_reason = str(result.get("block_reason") or "").strip() if status == "blocked" else ""
    return {
        "status": status, "summary": summary, "details": details,
        "metadata": metadata,
        **({"block_reason": block_reason} if block_reason else {}),
    }


def _result_failure_text(result: dict[str, Any]) -> str:
    return str(result.get("block_reason") or result.get("summary") or "unknown failure")


def _is_provider_failure_result(rc: int, result: dict[str, Any]) -> bool:
    if rc == 0:
        return False
    text = _result_failure_text(result).lower()
    keywords = ["rate limit", "api key", "429", "503", "quota", "provider",
                "authentication", "unauthorized", "overloaded"]
    return any(kw in text for kw in keywords)


# ──────────────────────────────────────────────
# Shared tail / idle detection helpers
# ──────────────────────────────────────────────

def _tail_nonempty_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [l for l in text.splitlines() if l.strip()]
    return lines[-limit:] if len(lines) > limit else lines


def _looks_like_idle_pane(
    text: str,
    idle_markers: tuple[str, ...],
    busy_markers: tuple[str, ...],
) -> bool:
    """Check if a pane screen looks idle, using configurable markers."""
    if not idle_markers:
        return True  # no markers configured → assume idle (trust heartbeat)
    tail = "\n".join(_tail_nonempty_lines(text, limit=40)).lower()
    has_idle = any(marker in tail for marker in idle_markers)
    has_busy = any(marker in tail for marker in busy_markers)
    if has_busy and not has_idle:
        return False
    if has_idle and not has_busy:
        return True
    # Ghost state: both idle and busy markers visible
    return has_idle


def _pane_can_accept_new_kanban_task(
    text: str,
    idle_markers: tuple[str, ...],
    busy_markers: tuple[str, ...],
    queued_input_markers: tuple[str, ...],
) -> bool:
    """Return True when it is safe to inject a new Kanban prompt."""
    if not idle_markers:
        return True  # no screen detection → always accept
    tail = "\n".join(_tail_nonempty_lines(text, limit=40)).lower()
    has_idle = any(marker in tail for marker in idle_markers)
    has_busy = any(marker in tail for marker in busy_markers)
    has_queued = any(marker in tail for marker in queued_input_markers)
    if has_queued:
        return False
    if has_busy and not has_idle:
        return False
    return has_idle


# ──────────────────────────────────────────────
# Shared reclaim-orphan helper
# ──────────────────────────────────────────────

def reclaim_orphaned_running_task(
    args: argparse.Namespace, *, log_path: Path, conn: Any,
    listener_kind: str = "interactive",
) -> bool:
    """Reclaim tasks whose worker_pid is no longer alive."""
    board = args.board or kb.get_current_board() or "default"
    workspace = Path(args.workspace).expanduser().resolve()
    reclaimed = False
    try:
        rows = conn.execute(
            "SELECT id, worker_pid, workspace_path FROM tasks WHERE status='running'"
        ).fetchall()
        for row in rows:
            pid = row["worker_pid"]
            if pid and not _pid_alive(pid):
                ws = row.get("workspace_path")
                if ws and not _workspace_matches(ws, workspace):
                    continue
                reason = f"orphaned running task {row['id']} old_pid={pid}"
                ok = _reclaim_task_without_signaling_worker(conn, row["id"], reason=reason)
                if ok:
                    reclaimed = True
                    log_line(log_path, reason)
    except Exception as exc:
        log_line(log_path, f"reclaim_orphaned error: {exc}")
    return reclaimed


# ──────────────────────────────────────────────
# BASE CLASS
# ──────────────────────────────────────────────

class BaseInteractiveListener:
    """Base class for all interactive Kanban listeners.

    Subclasses must define:
      - agent_name, agent_slug (str properties)
      - build_tui_cmd(workspace, continue_session, model, sandbox, extra_args) -> list[str]
      - has_saved_sessions(workspace) -> bool
      - inject_text(task_id, title, assignee, profile, prompt_path, board) -> str
      - pane_label(task_id=None) -> str

    Subclasses may override hooks:
      - idle_markers, busy_markers, queued_input_markers
      - on_claim_pre_check, on_claim_post_confirm
      - on_task_running_monitor, on_watcher_loop_idle
      - build_launch_env, build_watcher_extra_args
    """

    # ── Identity (subclass must override) ──
    agent_name: str = ""
    agent_slug: str = ""

    # ── Idle/busy markers for screen-based detection ──
    # Empty tuple = no screen-based idle detection (always accept).
    idle_markers: tuple[str, ...] = ()
    busy_markers: tuple[str, ...] = ()
    queued_input_markers: tuple[str, ...] = ()

    # ── Abstract methods (subclass MUST implement) ──

    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build the command list to launch the TUI process."""
        raise NotImplementedError

    def has_saved_sessions(self, workspace: Path) -> bool:
        """Check whether the agent has saved sessions for this workspace."""
        raise NotImplementedError

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        """Build the single-line text to inject into the pane.

        MUST NOT contain \\n (LF). Use spaces; TUI word-wraps."""
        raise NotImplementedError

    def pane_label(self, task_id: str | None = None) -> str:
        """Return the zellij pane title."""
        raise NotImplementedError

    # ── Optional hooks (default: no-op / basic) ──

    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        """Return True if the pane is ready to accept a new task."""
        if not self.idle_markers:
            return True
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return True
        screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen is None:
            return True
        return _pane_can_accept_new_kanban_task(
            screen, self.idle_markers, self.busy_markers, self.queued_input_markers,
        )

    def on_claim_post_confirm(self, args: argparse.Namespace, log_path: Path) -> bool:
        """After claim, confirm the pane is still idle before injecting.

        Default: no extra confirmation (return True immediately).
        DeepSeek overrides this with a 2-round idle check.
        """
        return True

    def on_post_inject(
        self, args: argparse.Namespace, *,
        zellij_session: str, zellij_pane_id: str, log_path: Path,
    ) -> None:
        """Hook called after zellij_inject succeeds.

        Subclasses can override to perform extra actions after injection,
        e.g. sending an additional Enter for TUIs that queue input.
        Default: no-op.
        """
        pass

    def on_task_running_monitor(
        self, args: argparse.Namespace, conn: Any,
        task_id: str, log_path: Path,
    ) -> None:
        """Extra monitoring while a task is running (progress watch, etc).

        Default: check for API failure on idle pane and inject '继续' to retry.
        """
        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = str(getattr(args, "zellij_pane_id", ""))
        if not zellij_session or not zellij_pane_id:
            return

        screen = zellij_dump_screen(session=zellij_session, pane_id=zellij_pane_id, log_path=log_path)
        if not screen:
            return

        # Check if pane is idle (not busy)
        if not _looks_like_idle_pane(screen, idle_markers=self.idle_markers, busy_markers=self.busy_markers):
            return

        # Pane is idle while task is running — check for API error
        self.check_api_failure_retry(
            session=zellij_session, pane_id=zellij_pane_id, screen=screen,
            task_id=task_id, log_path=log_path,
        )

    def on_watcher_loop_idle(
        self, args: argparse.Namespace, conn: Any, log_path: Path,
    ) -> None:
        """Called each watcher loop tick when no task is active.

        Default: no action. DeepSeek uses this for auto-dismiss steering.
        """
        pass

    def build_launch_env(self, args: argparse.Namespace) -> dict[str, str]:
        """Build environment variables for the TUI process."""
        env = os.environ.copy()
        board = args.board or kb.get_current_board() or "default"
        workspace = Path(args.workspace).expanduser().resolve()
        env.update({
            "HERMES_KANBAN_BOARD": board,
            "HERMES_KANBAN_PROFILE": args.profile,
            "HERMES_KANBAN_CLAIM_ASSIGNEES": ",".join(claim_assignees(args)),
            "HERMES_KANBAN_WORKSPACE": str(workspace),
        })
        return env

    def build_watcher_extra_args(self, args: argparse.Namespace) -> list[str]:
        """Extra args to append to the watcher --watch-child command."""
        return []

    # ── Instance state ──
    def __init__(self):
        self._profile: str = ""
        self._board: str = ""
        self._workspace: Path = Path(".")
        self._log_path: Path = Path("/tmp/kanban.log")
        self._api_retry_count: int = 0       # per-task API failure retry counter
        self._api_retry_first_at: float | None = None  # when first API-idle was seen

    # ── API failure retry on idle ──
    # When agent goes idle mid-task due to API error, inject "继续"
    # up to API_RETRY_MAX times with backoff. After max retries,
    # fall through to existing idle-pane-reclaim logic.
    API_RETRY_MAX: int = 2
    API_RETRY_BACKOFF: list[float] = [30.0, 60.0]  # seconds before each retry
    API_ERROR_MARKERS: tuple[str, ...] = (
        "⚠", "API call failed", "APIError", "request failed",
        "API request failed", "xunfei request failed",
        "Invalid Params", "AppIdNoAuth", "rate limit",
        "503", "502", "429", "timeout",
    )

    def check_api_failure_retry(
        self, *, session: str, pane_id: str, screen: str,
        task_id: str, log_path: Path,
    ) -> bool:
        """Check if pane is idle with an API error and inject '继续' to retry.

        Returns True if a retry was injected this tick (caller should skip reclaim).
        Returns False if no retry was needed or max retries exhausted.
        """
        if self._api_retry_count >= self.API_RETRY_MAX:
            return False

        tail = "\n".join(_tail_nonempty_lines(screen)).lower()
        has_error = any(m.lower() in tail for m in self.API_ERROR_MARKERS)

        if not has_error:
            # No API error visible — reset retry state
            self._api_retry_first_at = None
            return False

        # API error detected on pane — check if we should retry now
        now = time.time()
        if self._api_retry_first_at is None:
            self._api_retry_first_at = now
            log_line(log_path, f"api-error-idle observed for task {task_id} (retry {self._api_retry_count}/{self.API_RETRY_MAX})")

        elapsed = now - self._api_retry_first_at
        backoff = self.API_RETRY_BACKOFF[self._api_retry_count] if self._api_retry_count < len(self.API_RETRY_BACKOFF) else 60.0

        if elapsed < backoff:
            return True  # still waiting for backoff; skip reclaim this tick

        # Backoff elapsed — inject "继续"
        self._api_retry_count += 1
        self._api_retry_first_at = None  # reset timer; next error detection starts fresh
        log_line(log_path, f"api-error-retry {self._api_retry_count}/{self.API_RETRY_MAX} for task {task_id}: injecting 继续 after {elapsed:.0f}s")

        zellij_inject(session=session, pane_id=pane_id, text="继续", log_path=log_path)
        time.sleep(0.5)
        zellij_inject(session=session, pane_id=pane_id, text="\r", log_path=log_path)

        return True

    def _init_from_args(self, args: argparse.Namespace) -> None:
        self._profile = args.profile
        self._board = args.board or kb.get_current_board() or "default"
        self._workspace = Path(args.workspace).expanduser().resolve()
        self._log_path = kb.worker_logs_dir(board=self._board) / f"{self.agent_slug}-interactive-{self._profile}.log"

    # ── Claim lock ──
    def _claim_lock(self) -> str:
        return f"{socket.gethostname()}:{os.getpid()}:{self.agent_slug}-interactive"

    # ── claim_and_inject_one ──
    def claim_and_inject_one(
        self, args: argparse.Namespace, *, log_path: Path, conn: Any | None = None,
    ) -> tuple[str | None, int | None]:
        board = self._board
        workspace = self._workspace
        pane_profile = self._profile

        # Pre-check: is pane ready?
        if not self.on_claim_pre_check(args, log_path):
            log_line(log_path, f"skip claim: {self.agent_name} pane not ready for injection")
            return None, None

        _owns_conn = conn is None
        if _owns_conn:
            try:
                conn = kb.connect(board=board)
            except Exception as exc:
                log_line(log_path, f"claim DB error (non-fatal): {type(exc).__name__}: {exc}")
                return None, None

        try:
            kb.release_stale_claims(conn)
            kb.recompute_ready(conn)
            candidate = _select_ready_candidate(conn, args)
            if candidate is None:
                return None, None
            claimed = kb.claim_task(conn, candidate.id, ttl_seconds=args.ttl, claimer=self._claim_lock())
            if claimed is None:
                return None, None
            kb.set_workspace_path(conn, claimed.id, workspace)
            claimed = kb.get_task(conn, claimed.id) or claimed
            try:
                kb._set_worker_pid(conn, claimed.id, os.getpid())  # type: ignore[attr-defined]
            except Exception:
                pass
            context = kb.build_worker_context(conn, claimed.id)
        except Exception as exc:
            log_line(log_path, f"claim DB error (non-fatal): {type(exc).__name__}: {exc}")
            return None, None

        prompt_path = write_task_prompt(
            agent_name=self.agent_name, agent_slug=self.agent_slug,
            board=board, profile=pane_profile,
            task_id=claimed.id,
            task_assignee=getattr(claimed, "assignee", None) or pane_profile,
            task_title=claimed.title,
            context=context, workspace=workspace,
        )

        # Post-claim idle confirmation
        if not self.on_claim_post_confirm(args, log_path):
            log_line(log_path, f"idle confirmation failed; aborting injection for {claimed.id}")
            try:
                _reclaim_task_without_signaling_worker(
                    conn, claimed.id,
                    reason=f"{self.agent_slug}-interactive pane not stably idle before injection",
                )
            except Exception:
                pass
            return None, None

        task_assignee = getattr(claimed, "assignee", None) or pane_profile
        inject_str = self.inject_text(
            task_id=claimed.id, title=claimed.title,
            assignee=task_assignee, profile=pane_profile,
            prompt_path=prompt_path, board=board,
        )

        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = getattr(args, "zellij_pane_id", "")

        ok = zellij_inject(
            session=zellij_session, pane_id=str(zellij_pane_id),
            text=inject_str, log_path=log_path,
        )
        if not ok:
            try:
                _reclaim_task_without_signaling_worker(
                    conn, claimed.id,
                    reason=f"{self.agent_slug}-interactive zellij injection failed",
                )
            except Exception:
                pass
            return None, None

        # Hook: subclass post-inject actions (e.g. extra Enter for queued-input TUIs)
        self.on_post_inject(args, zellij_session=zellij_session, zellij_pane_id=str(zellij_pane_id), log_path=log_path)

        # Post-inject DB ops
        try:
            kb.add_comment(
                conn, claimed.id,
                f"{self.agent_slug}-interactive-listener",
                f"Injected into Zellij pane {zellij_pane_id}; prompt file: {prompt_path}",
            )
            kb.heartbeat_worker(
                conn, claimed.id,
                note=f"{self.agent_slug}-interactive injected prompt: {prompt_path}",
                expected_run_id=claimed.current_run_id,
            )
        except Exception as exc:
            log_line(log_path, f"post-inject DB op failed (non-fatal): {exc}")

        zellij_rename_pane(
            session=zellij_session, pane_id=str(zellij_pane_id),
            name=self.pane_label(task_id=claimed.id),
            log_path=log_path,
        )
        log_line(log_path, f"claimed+injected {claimed.id}: {claimed.title} prompt={prompt_path}")
        return claimed.id, claimed.current_run_id

    # ── watcher_main ──
    def watcher_main(self, args: argparse.Namespace) -> int:
        self._init_from_args(args)
        board = self._board
        log_path = self._log_path
        workspace = self._workspace

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        if not workspace.exists():
            log_line(log_path, f"workspace does not exist: {workspace}")
            return 2

        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = getattr(args, "zellij_pane_id", "")
        if not zellij_session or not zellij_pane_id:
            log_line(log_path, f"missing zellij session/pane id; cannot inject into {self.agent_name} TUI")
            return 2

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        log_line(
            log_path,
            f"interactive watcher started profile={args.profile} "
            f"claim_assignees={','.join(claim_assignees(args))} board={board} "
            f"workspace={workspace} pane={zellij_session}:{zellij_pane_id} "
            f"poll={poll_s:g}s agent={self.agent_name}",
        )
        zellij_rename_pane(
            session=zellij_session, pane_id=str(zellij_pane_id),
            name=self.pane_label(), log_path=log_path,
        )

        startup_delay = getattr(args, "startup_delay_s", 0) or 0
        if startup_delay > 0:
            time.sleep(startup_delay)

        # ── Persistent DB connection ──
        MAX_CONSECUTIVE_DB_ERRORS = 5
        consecutive_db_errors = 0
        _CONN_RECYCLE_S = 60.0
        _conn: Any = None
        _conn_created_at: float = 0.0

        def _ensure_conn() -> Any:
            nonlocal _conn, _conn_created_at, consecutive_db_errors

            if _conn is not None and (time.time() - _conn_created_at) >= _CONN_RECYCLE_S:
                try:
                    _conn.close()
                except Exception:
                    pass
                _conn = None

            if _conn is not None:
                try:
                    _conn.execute("SELECT 1")
                    return _conn
                except sqlite3.OperationalError:
                    try:
                        _conn.close()
                    except Exception:
                        pass
                    _conn = None

            for attempt in range(3):
                try:
                    _conn = kb.connect(board=board)
                    _conn_created_at = time.time()
                    consecutive_db_errors = 0
                    return _conn
                except sqlite3.OperationalError as exc:
                    consecutive_db_errors += 1
                    delay = 2.0 * (2 ** attempt)
                    log_line(log_path, f"DB OperationalError (attempt {attempt+1}/3, consecutive={consecutive_db_errors}): {exc}")
                    time.sleep(delay)
                except sqlite3.DatabaseError as exc:
                    msg = str(exc).lower()
                    if "malformed" in msg or "corrupt" in msg:
                        log_line(log_path, f"DB corruption detected: {exc}; attempting REINDEX repair")
                        try:
                            repair_conn = sqlite3.connect(str(kb.kanban_db_path(board=board)), timeout=120.0)
                            repair_conn.execute("PRAGMA integrity_check")
                            repair_conn.execute("REINDEX")
                            repair_conn.close()
                            log_line(log_path, "REINDEX repair succeeded; retrying connect")
                        except Exception as repair_exc:
                            log_line(log_path, f"REINDEX repair failed: {repair_exc}")
                    consecutive_db_errors += 1
                    time.sleep(4.0)
                except Exception as exc:
                    consecutive_db_errors += 1
                    log_line(log_path, f"DB connect error: {type(exc).__name__}: {exc}")
                    time.sleep(4.0)
            return None

        active_task: str | None = None
        active_run_id: int | None = None
        last_hb = 0.0

        try:
            while not _STOP:
                now = time.time()
                conn = _ensure_conn()
                if conn is None:
                    if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                        log_line(log_path, f"too many consecutive DB errors ({consecutive_db_errors}); stopping watcher")
                        break
                    time.sleep(poll_s)
                    continue

                if active_task:
                    try:
                        status, current_run_id = _task_status(conn, active_task)
                    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                        consecutive_db_errors += 1
                        log_line(log_path, f"DB error checking task status: {exc}; will retry")
                        time.sleep(min(poll_s, 5.0))
                        continue
                    consecutive_db_errors = 0

                    if status == "running" and (active_run_id is None or current_run_id == active_run_id):
                        # Hook: subclass may do progress watch / idle reclaim / etc
                        self.on_task_running_monitor(args, conn, active_task, log_path)

                        if now - last_hb >= max(15.0, min(float(args.ttl) / 3.0, 120.0)):
                            try:
                                kb.heartbeat_claim(conn, active_task, ttl_seconds=args.ttl, claimer=self._claim_lock())
                                kb.heartbeat_worker(
                                    conn, active_task,
                                    note=f"{self.agent_slug}-interactive waiting for complete/block from {self.agent_name} TUI",
                                    expected_run_id=active_run_id,
                                )
                            except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                                consecutive_db_errors += 1
                                log_line(log_path, f"DB error on heartbeat: {exc}")
                            else:
                                last_hb = now
                        time.sleep(min(poll_s, 5.0))
                        continue

                    log_line(log_path, f"active task left running state: {active_task} status={status} run={current_run_id}")
                    active_task = None
                    active_run_id = None
                    last_hb = 0.0
                    zellij_rename_pane(
                        session=zellij_session, pane_id=str(zellij_pane_id),
                        name=self.pane_label(), log_path=log_path,
                    )

                # Hook: subclass idle-loop actions (e.g. auto-dismiss steering)
                self.on_watcher_loop_idle(args, conn, log_path)

                reclaim_orphaned_running_task(args, log_path=log_path, conn=conn)
                active_task, active_run_id = self.claim_and_inject_one(args, log_path=log_path, conn=conn)
                if active_task:
                    consecutive_db_errors = 0
                    last_hb = 0.0
                    if args.once:
                        continue
                else:
                    if args.once:
                        log_line(log_path, "no ready task; exiting --once")
                        return 0
                    time.sleep(poll_s)
        finally:
            if _conn is not None:
                try:
                    _conn.close()
                except Exception:
                    pass
            _cleanup_active_claim(board=board, task_id=active_task, run_id=active_run_id, log_path=log_path)
        log_line(log_path, "interactive watcher stopped")
        return 0

    # ── launcher_main ──
    def launcher_main(self, args: argparse.Namespace) -> int:
        self._init_from_args(args)
        board = self._board
        workspace = self._workspace
        log_path = self._log_path

        if not workspace.exists():
            print(f"错误: workspace 不存在: {workspace}", file=sys.stderr)
            return 2

        zellij_session = getattr(args, "zellij_session", "") or os.environ.get("ZELLIJ_SESSION_NAME")
        zellij_pane_id = getattr(args, "zellij_pane_id", "") or os.environ.get("ZELLIJ_PANE_ID")
        if not zellij_session or not zellij_pane_id:
            print(f"错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 {self.agent_name} TUI。", file=sys.stderr)
            return 2

        # Store zellij info in args for watcher
        args.zellij_session = zellij_session
        args.zellij_pane_id = zellij_pane_id

        # Build watcher command — use the subclass file, not base_listener.py
        subclass_file = Path(sys.modules[type(self).__module__].__file__ or __file__).resolve()
        watcher_cmd = [
            sys.executable, str(subclass_file),
            "--watch-child",
            "--profile", args.profile,
            "--claim-assignees", ",".join(claim_assignees(args)),
            "--board", board,
            "--workspace", str(workspace),
            "--ttl", str(args.ttl),
            "--zellij-session", zellij_session,
            "--zellij-pane-id", zellij_pane_id,
            "--startup-delay-s", str(getattr(args, "startup_delay_s", 0) or 0),
            "--assist-claim-delay-s", str(assist_claim_delay_s(args)),
        ]
        watcher_cmd.extend(self.build_watcher_extra_args(args))
        for spec in _delay_specs(getattr(args, "assist_claim_delay_for", None)):
            watcher_cmd.extend(["--assist-claim-delay-for", spec])
        if args.poll is not None:
            watcher_cmd.extend(["--poll", str(args.poll)])

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        poll_label = f"{poll_s:g}s" + (" override" if args.poll is not None else " shared-policy")

        print(f"{self.agent_name} interactive kanban mode")
        print(f"  board:     {board}")
        print(f"  profile:   {args.profile}")
        print(f"  claims:    {', '.join(claim_assignees(args))}")
        print(f"  workspace: {workspace}")
        print(f"  pane:      {zellij_session}:{zellij_pane_id}")
        print(f"  log:       {log_path}")
        print("")
        print(f"按 Enter 进入 interactive {self.agent_name}；后台 listener 会按优先级 claim ready 任务并注入到当前 TUI。")
        print(f"{self.agent_slug}-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace}")

        watch_only = getattr(args, "watch_only", False)
        if watch_only:
            print("listener-only 模式：不会启动 TUI，只运行后台 listener 并向指定 Zellij pane 注入任务。")
            return self.watcher_main(args)

        auto_start = getattr(args, "auto_start", False)
        if not auto_start:
            try:
                input()
            except EOFError:
                pass

        env = self.build_launch_env(args)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = log_path.open("a", encoding="utf-8")
        log_line(log_path, f"launcher starting watcher: {' '.join(watcher_cmd)}")

        watcher = subprocess.Popen(
            watcher_cmd,
            stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
            text=True, env=env, start_new_session=True,
        )
        print(f"{self.agent_slug}-kanban listener started: pid={watcher.pid} profile={args.profile} board={board} poll={poll_label}", flush=True)

        continue_session = self.has_saved_sessions(workspace)
        tui_cmd = self.build_tui_cmd(
            workspace, continue_session=continue_session,
            model=args.model if hasattr(args, "model") else None,
            sandbox=args.sandbox if hasattr(args, "sandbox") else None,
            extra_args=getattr(args, f"{self.agent_slug}_arg", None) or [],
        )

        log_line(log_path, f"launcher starting {self.agent_name}: {' '.join(tui_cmd)}")
        rc = 0
        try:
            if hasattr(sys.stdin, "isatty") and sys.stdin.isatty():
                rc = subprocess.call(tui_cmd, cwd=str(workspace), env=env)
            else:
                try:
                    with open("/dev/tty", "rb") as tty_stdin:
                        rc = subprocess.call(tui_cmd, cwd=str(workspace), env=env, stdin=tty_stdin)
                except OSError:
                    rc = subprocess.call(tui_cmd, cwd=str(workspace), env=env, stdin=sys.stdin)
        finally:
            log_line(log_path, f"{self.agent_name} exited rc={rc}; stopping watcher pid={watcher.pid}")
            try:
                watcher.terminate()
                watcher.wait(timeout=10)
            except subprocess.TimeoutExpired:
                watcher.kill()
        return rc

    # ── main ──
    def main(self, argv: list[str] | None = None) -> int:
        parser = self._build_parser()
        args = parser.parse_args(argv)
        self._init_from_args(args)

        if getattr(args, "reset_kanban", False):
            board = self._board
            workspace = self._workspace
            reset_ids = reset_kanban_claims(
                board=board, profile=args.profile,
                claim_assignees_list=claim_assignees(args),
                workspace=workspace,
                listener_kind=f"{self.agent_slug}-interactive",
            )
            if reset_ids:
                print(f"reset-kanban reclaimed: {', '.join(reset_ids)}")
            else:
                print("reset-kanban: no matching running claim")
            return 0

        if getattr(args, "watch_child", False):
            return self.watcher_main(args)
        return self.launcher_main(args)

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=f"Run {self.agent_name} as a Hermes Kanban listener/profile"
        )
        parser.add_argument("--profile", default=os.environ.get("HERMES_PROFILE") or "", help="Kanban assignee/profile")
        parser.add_argument("--claim-assignees", default=os.environ.get("HERMES_KANBAN_CLAIM_ASSIGNEES") or "", help="Comma-separated assignees this worker may claim")
        parser.add_argument("--board", default=os.environ.get("HERMES_KANBAN_BOARD"), help="Board slug")
        parser.add_argument("--workspace", default=os.environ.get(f"{self.agent_slug.upper()}_KANBAN_WORKSPACE") or os.environ.get("HERMES_KANBAN_WORKSPACE") or ".", help="Workspace directory")
        parser.add_argument("--poll", type=float, default=None, help="Poll interval override (seconds)")
        parser.add_argument("--ttl", type=int, default=listener_policy.LISTENER_HEALTH_CLAIM_TTL_SECONDS, help="Claim TTL (seconds)")
        parser.add_argument("--model", default=None, help="Optional model override")
        parser.add_argument("--sandbox", default=None, help="Optional sandbox override")
        parser.add_argument("--assist-claim-delay-s", type=float, default=0.0, help="Delay before claiming secondary assignees")
        parser.add_argument("--assist-claim-delay-for", action="append", default=[], help="Per-assignee assist delay")
        parser.add_argument("--startup-delay-s", type=float, default=8.0, help="Delay before first claim (seconds)")
        parser.add_argument("--once", action="store_true", help="Process at most one task then exit")
        parser.add_argument("--watch-child", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--reset-kanban", action="store_true", help="Reclaim running tasks and exit")
        parser.add_argument("--auto-start", action="store_true", help="Skip the Enter prompt")
        parser.add_argument("--watch-only", action="store_true", help="Only run watcher, don't launch TUI")
        parser.add_argument("--zellij-session", default=os.environ.get("ZELLIJ_SESSION_NAME"), help="Target Zellij session for task injection")
        parser.add_argument("--zellij-pane-id", default=os.environ.get("ZELLIJ_PANE_ID"), help="Target Zellij pane id for task injection")
        # Agent-specific args placeholder
        parser.add_argument(f"--{self.agent_slug}-arg", action="append", default=[], help=f"Extra args for {self.agent_name}")
        return parser
