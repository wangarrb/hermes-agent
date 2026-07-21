#!/usr/bin/env python3
"""ReasonixInteractiveListener — Hermes Kanban listener for Reasonix CLI.

Reasonix TUI idle markers (v0.x + v1.x Go rewrite):
  - Idle: "ask anything · slash for commands · at-sign for files",
          "输入任何内容 · / 使用命令 · @ 引用文件",
          "输入 'exit' 或按 ctrl-d 退出", "yolo · auto-approved", etc.
  - Busy: "kanban_task_boundary", "processing", "run running", etc.
  - Queued: "pending inputs", "edit last queued message"

Injection strategy: write prompt to .md file, inject multi-line text
with KANBAN_TASK_BOUNDARY marker.  Reasonix handles multi-line input
natively (no LF/CR issue — Reasonix uses prompt_toolkit which
correctly handles LF in the input buffer).

NOTE: Reasonix is the ONLY agent where inject_text contains \\n.
This is safe because Reasonix's prompt_toolkit-based TUI correctly
handles LF in the input buffer (inserts newline, not submit).
For Codex/CodeWhale, LF would NOT submit and causes the
"typed but not sent" bug.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── Base class import ──
HERMES_AGENT_ROOT = Path(__file__).resolve().parents[3]
if str(HERMES_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(HERMES_AGENT_ROOT))
PLUGIN_KANBAN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_KANBAN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_KANBAN_DIR))

from base_listener import (  # noqa: E402
    BaseInteractiveListener,
    claim_assignees,
    log,
    log_line,
    now_s,
    prompt_dir,
    role_guidance,
    zellij_dump_screen,
    zellij_inject,
    zellij_rename_pane,
    _noop_signal,
    _pid_alive,
    _pane_can_accept_new_kanban_task,
    _reclaim_task_without_signaling_worker,
    _tail_nonempty_lines,
    _task_status,
    reclaim_orphaned_running_task,
)

# ── Reasonix-specific imports ──
HERMES_REPO = HERMES_AGENT_ROOT
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402


# ──────────────────────────────────────────────
# Reasonix-specific constants
# ──────────────────────────────────────────────

_REASONIX_IDLE_MARKERS = (
    "ask anything · slash for commands · at-sign for files",
    "输入任何内容 · / 使用命令 · @ 引用文件",
    "type to steer the current task",
    "输入 'exit' 或按 ctrl-d 退出",
    "enter 'exit' or press ctrl-d to exit",
    "对话上下文将跨轮保留",
    "conversation context will be preserved",
    "yolo  · 已跳过批准",
    "yolo · auto-approved",
    "已跳过批准",
)

_REASONIX_BUSY_MARKERS = (
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

_STALE_KANBAN_INJECTION_MARKERS = (
    "kanban_task_boundary",
    "hermes kanban 已领取任务",
    "完成后必须运行",
)


# ──────────────────────────────────────────────
# Reasonix-specific helpers
# ──────────────────────────────────────────────

def _normalize_ws(s: str) -> str:
    """Collapse consecutive whitespace into single spaces for robust matching."""
    return " ".join(s.split())


def _looks_like_idle_reasonix_pane(text: str) -> bool:
    tail = "\n".join(_tail_nonempty_lines(text)).lower()
    if not tail:
        return False
    tail_norm = _normalize_ws(tail)
    if any(marker in tail_norm for marker in (*_REASONIX_BUSY_MARKERS, *_REASONIX_QUEUED_INPUT_MARKERS)):
        return False
    return any(marker in tail_norm for marker in _REASONIX_IDLE_MARKERS)


def _reasonix_pane_can_accept_new_kanban_task(text: str) -> bool:
    """Return True when it is safe to inject a new Kanban prompt.

    Reasonix can buffer text while a previous prompt is still visible.
    Only inject when the pane is idle and not queued.
    Stale Kanban injection markers from a completed task should NOT
    block a new injection when the pane is otherwise idle.
    """
    tail = "\n".join(_tail_nonempty_lines(text)).lower()
    if not tail:
        return False
    tail_norm = _normalize_ws(tail)
    if any(marker in tail_norm for marker in _REASONIX_QUEUED_INPUT_MARKERS):
        return False
    # Check for truly busy markers (not stale Kanban markers)
    truly_busy = tuple(
        m for m in _REASONIX_BUSY_MARKERS
        if m not in _STALE_KANBAN_INJECTION_MARKERS
    )
    if any(marker in tail_norm for marker in truly_busy):
        return False
    has_idle = any(marker in tail_norm for marker in _REASONIX_IDLE_MARKERS)
    return has_idle


def _write_role_instructions(*, workspace: Path, board: str, pane_profile: str) -> Path:
    """Write role instructions file and reasonix.toml for system_prompt_file injection."""
    d = prompt_dir(workspace, board, pane_profile, agent_slug="reasonix")
    d.mkdir(parents=True, exist_ok=True)
    p = d / "role-instructions.md"
    p.write_text(role_guidance(pane_profile), encoding="utf-8")

    toml_path = workspace / "reasonix.toml"
    lines = [
        "# Auto-generated by Hermes kanban listener — do not edit manually",
        "# Reasonix v1.1.0+ workspace config for kanban mode",
    ]
    try:
        rel = p.relative_to(workspace)
        prompt_path = str(rel)
    except ValueError:
        prompt_path = str(p)
    lines.append(f'system_prompt_file = "{prompt_path}"')
    lines.append("")
    lines.append("[agent]")
    lines.append("max_steps = 0")
    lines.append('auto_plan = "off"')

    if toml_path.exists():
        backup = toml_path.with_suffix(".toml.user-bak")
        if not backup.exists():
            shutil.copy2(toml_path, backup)

    toml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _sessions_latest_mtime() -> float:
    try:
        session_dir = Path.home() / ".reasonix" / "sessions"
        if not session_dir.is_dir():
            return 0.0
        return max(
            (f.stat().st_mtime for f in session_dir.rglob("*") if f.is_file()),
            default=0.0,
        )
    except Exception:
        return 0.0


def _screen_fingerprint(screen: str) -> str:
    lines = _tail_nonempty_lines(screen, limit=20)
    return "|".join(lines[-5:])


class _PaneProgressWatch:
    def __init__(self) -> None:
        self.fingerprint: str = ""
        self.stalled_busy_seen_at: float | None = None
        self.latest_session_mtime: float = 0.0


def _should_restart_watcher(returncode: int | None, *, reasonix_alive: bool = True) -> bool:
    # A clean watcher exit is still a lost listener while the TUI is alive.
    return returncode is not None and reasonix_alive


# ──────────────────────────────────────────────
# Reasonix subclass
# ──────────────────────────────────────────────

class ReasonixInteractiveListener(BaseInteractiveListener):
    agent_name = "Reasonix"
    agent_slug = "reasonix"

    idle_markers = _REASONIX_IDLE_MARKERS
    busy_markers = _REASONIX_BUSY_MARKERS
    queued_input_markers = _REASONIX_QUEUED_INPUT_MARKERS

    # ── Build TUI command ──
    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build Reasonix v1.1.0+ command line.

        Uses 'reasonix chat --dir <workspace>' with system_prompt_file
        in reasonix.toml (no --system-append-file flag needed).
        """
        cmd = [getattr(self, "_reasonix_bin", "reasonix"), "chat", "--dir", str(workspace)]
        if model:
            cmd.extend(["--model", model])
        cmd.append("--yolo")
        if continue_session:
            cmd.append("--continue")
        cmd.extend(extra_args or [])
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        session_dir = Path.home() / ".reasonix" / "sessions"
        if not session_dir.is_dir():
            return False
        try:
            return any(session_dir.iterdir())
        except Exception:
            return False

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        """Build multi-line injection for Reasonix.

        NOTE: Reasonix is the ONLY agent where inject_text contains \\n.
        This is safe because Reasonix's prompt_toolkit-based TUI correctly
        handles LF in the input buffer (inserts newline, not submit).
        """
        return (
            f"KANBAN_TASK_BOUNDARY\n"
            f"Task: {task_id} — {title}\n"
            f"Role: {assignee} | File: {prompt_path}\n"
            f"Finish: hermes kanban --board {board} complete {task_id} ... OR block {task_id} ..."
        )

    def pane_label(self, task_id: str | None = None) -> str:
        profile = self._profile or "implementer"
        if task_id:
            return f"{profile}-reasonix [{task_id}]"
        return f"{profile}-reasonix listening"

    # ── Override: on_claim_pre_check with Reasonix-specific idle detection ──
    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return False
        screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if not screen or not screen.strip():
            return False
        return _reasonix_pane_can_accept_new_kanban_task(screen)

    # ── Override: on_claim_post_confirm (2-round idle check) ──
    def on_claim_post_confirm(self, args: argparse.Namespace, log_path: Path) -> bool:
        """Reasonix-specific: confirm pane is stably idle before injection."""
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return True
        screen1 = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen1 is None:
            return True
        if not _reasonix_pane_can_accept_new_kanban_task(screen1):
            return False
        time.sleep(1.0)
        screen2 = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen2 is None:
            return True
        return _reasonix_pane_can_accept_new_kanban_task(screen2)

    # ── Override: build_launch_env ──
    def build_launch_env(self, args: argparse.Namespace) -> dict[str, str]:
        env = super().build_launch_env(args)
        # Reasonix-specific env vars
        if getattr(args, "yolo", True):
            env["REASONIX_YOLO"] = "true"
        return env

    # ── Override: build_watcher_extra_args ──
    def build_watcher_extra_args(self, args: argparse.Namespace) -> list[str]:
        extra = []
        for attr in ("task_boundary_delay_s", "task_timeout_s", "idle_pane_reclaim_s"):
            val = getattr(args, attr, None)
            if val is not None:
                extra.extend([f"--{attr.replace('_', '-')}", str(val)])
        return extra

    # ── Instance state ──
    def __init__(self):
        super().__init__()
        self._progress_watch = _PaneProgressWatch()
        self._idle_pane_seen_at: float | None = None
        self._last_activity_at: float = time.time()
        self._last_liveness_check: float = 0.0
        self._last_task_transition_at: float = 0.0
        self._reasonix_bin: str = "reasonix"

    def _reset_active_task(self) -> None:
        self._idle_pane_seen_at = None
        self._progress_watch = _PaneProgressWatch()
        self._last_task_transition_at = time.time()

    # ── Override: watcher_main with Reasonix-specific monitoring ──
    def watcher_main(self, args: argparse.Namespace) -> int:
        """Reasonix watcher with idle-pane reclaim and progress watch."""
        self._init_from_args(args)
        board = self._board
        log_path = self._log_path
        workspace = self._workspace

        # Raise FD limit
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < 4096:
                resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
        except Exception:
            pass

        import signal as sig
        from base_listener import _handle_stop, stop_requested
        sig.signal(sig.SIGINT, _handle_stop)
        sig.signal(sig.SIGTERM, _handle_stop)

        if not workspace.exists():
            log_line(log_path, f"workspace does not exist: {workspace}")
            return 2

        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = getattr(args, "zellij_pane_id", "")
        if not zellij_session or not zellij_pane_id:
            log_line(log_path, f"missing zellij session/pane id; cannot inject into Reasonix TUI")
            return 2

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        task_timeout_s = float(getattr(args, "task_timeout_s", 0))
        idle_pane_reclaim_s = float(
            getattr(args, "idle_pane_reclaim_s", listener_policy.INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS)
        )
        task_boundary_delay_s = float(getattr(args, "task_boundary_delay_s", 8.0))

        log_line(
            log_path,
            f"interactive watcher started profile={args.profile} "
            f"claim_assignees={','.join(claim_assignees(args))} board={board} "
            f"workspace={workspace} pane={zellij_session}:{zellij_pane_id} "
            f"poll={poll_s:g}s agent=Reasonix",
        )
        zellij_rename_pane(
            session=zellij_session, pane_id=str(zellij_pane_id),
            name=self.pane_label(), log_path=log_path,
        )

        startup_delay = getattr(args, "startup_delay_s", 0) or 0
        if startup_delay > 0:
            time.sleep(startup_delay)

        # ── Persistent DB connection ──
        import sqlite3
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
                except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
                    consecutive_db_errors += 1
                    delay = 2.0 * (2 ** attempt)
                    log_line(log_path, f"DB error (attempt {attempt+1}/3, consecutive={consecutive_db_errors}): {exc}")
                    time.sleep(delay)
                except Exception as exc:
                    consecutive_db_errors += 1
                    log_line(log_path, f"DB connect error: {type(exc).__name__}: {exc}")
                    time.sleep(4.0)
            return None

        active_task: str | None = None
        active_run_id: int | None = None
        last_hb = 0.0
        self._last_activity_at = time.time()
        self._last_liveness_check = 0.0
        self._idle_pane_seen_at = None
        self._progress_watch = _PaneProgressWatch()
        self._last_task_transition_at = 0.0

        try:
            while not stop_requested():
                now = time.time()
                conn = _ensure_conn()
                if conn is None:
                    if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
                        log_line(log_path, f"too many consecutive DB errors ({consecutive_db_errors}); stopping")
                        break
                    time.sleep(poll_s)
                    continue

                if active_task:
                    try:
                        status, current_run_id = _task_status(conn, active_task)
                    except Exception as exc:
                        consecutive_db_errors += 1
                        log_line(log_path, f"DB error checking task status: {exc}")
                        time.sleep(min(poll_s, 5.0))
                        continue
                    consecutive_db_errors = 0

                    if status == "running" and (active_run_id is None or current_run_id == active_run_id):
                        # Liveness check
                        if now - self._last_liveness_check >= 30.0:
                            latest_mtime = _sessions_latest_mtime()
                            if latest_mtime > 0 and latest_mtime > self._last_activity_at:
                                self._last_activity_at = now
                            self._last_liveness_check = now

                        # Task timeout
                        if task_timeout_s > 0 and now - self._last_activity_at > task_timeout_s:
                            idle_s = now - self._last_activity_at
                            log_line(log_path, f"task timeout {active_task}: idle {idle_s:.0f}s > {task_timeout_s:.0f}s; reclaiming")
                            _reclaim_task_without_signaling_worker(
                                conn, active_task,
                                reason=f"reasonix-interactive idle timeout after {idle_s:.0f}s (limit {task_timeout_s:.0f}s)",
                            )
                            active_task = None
                            active_run_id = None
                            self._reset_active_task()
                            last_hb = 0.0
                            continue

                        # Heartbeat
                        if now - last_hb >= max(15.0, min(float(args.ttl) / 3.0, 120.0)):
                            try:
                                kb.heartbeat_claim(conn, active_task, ttl_seconds=args.ttl, claimer=self._claim_lock())
                                kb.heartbeat_worker(
                                    conn, active_task,
                                    note="reasonix-interactive waiting for complete/block from Reasonix TUI",
                                    expected_run_id=active_run_id,
                                )
                            except Exception as exc:
                                consecutive_db_errors += 1
                                log_line(log_path, f"DB error on heartbeat: {exc}")
                            else:
                                last_hb = now
                        time.sleep(min(poll_s, 5.0))
                        continue

                    # Task left running state
                    log_line(log_path, f"active task left running state: {active_task} status={status} run={current_run_id}")
                    active_task = None
                    active_run_id = None
                    self._reset_active_task()
                    last_hb = 0.0
                    zellij_rename_pane(
                        session=zellij_session, pane_id=str(zellij_pane_id),
                        name=self.pane_label(), log_path=log_path,
                    )

                # Task boundary delay
                remaining_delay = task_boundary_delay_s - (time.time() - self._last_task_transition_at)
                if remaining_delay > 0.0:
                    time.sleep(min(remaining_delay, poll_s, 5.0))
                    continue

                # Orphan reclaim
                reclaim_orphaned_running_task(args, log_path=log_path, conn=conn)

                # Claim and inject
                active_task, active_run_id = self.claim_and_inject_one(args, log_path=log_path, conn=conn)
                if active_task:
                    consecutive_db_errors = 0
                    self._last_activity_at = time.time()
                    self._last_liveness_check = time.time()
                    self._idle_pane_seen_at = None
                    self._progress_watch = _PaneProgressWatch()
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
            from base_listener import _cleanup_active_claim
            _cleanup_active_claim(board=board, task_id=active_task, run_id=active_run_id, log_path=log_path)
        log_line(log_path, "interactive watcher stopped")
        return 0

    # ── Override: launcher_main with watcher auto-restart ──
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
            print(f"错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 Reasonix TUI。", file=sys.stderr)
            return 2

        args.zellij_session = zellij_session
        args.zellij_pane_id = zellij_pane_id

        # Write role instructions + reasonix.toml
        _write_role_instructions(workspace=workspace, board=board, pane_profile=args.profile)

        # Build watcher command
        watcher_cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--watch-child",
            "--profile", args.profile,
            "--claim-assignees", ",".join(claim_assignees(args)),
            "--board", board,
            "--workspace", str(workspace),
            "--ttl", str(args.ttl),
            "--zellij-session", zellij_session,
            "--zellij-pane-id", zellij_pane_id,
            "--startup-delay-s", str(getattr(args, "startup_delay_s", 0) or 0),
            "--assist-claim-delay-s", str(getattr(args, "assist_claim_delay_s", 0.0)),
        ]
        watcher_cmd.extend(self.build_watcher_extra_args(args))
        if args.poll is not None:
            watcher_cmd.extend(["--poll", str(args.poll)])

        poll_s = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
        poll_label = f"{poll_s:g}s" + (" override" if args.poll is not None else " shared-policy")

        print(f"Reasonix interactive kanban mode")
        print(f"  board:     {board}")
        print(f"  profile:   {args.profile}")
        print(f"  claims:    {', '.join(claim_assignees(args))}")
        print(f"  workspace: {workspace}")
        print(f"  pane:      {zellij_session}:{zellij_pane_id}")
        print(f"  log:       {log_path}")
        print("")
        print(f"按 Enter 进入 interactive Reasonix；后台 listener 会按优先级 claim ready 任务并注入到当前 TUI。")
        print(f"reasonix-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace}")

        watch_only = getattr(args, "watch_only", False)
        if watch_only:
            print("listener-only 模式：不会启动 Reasonix，只运行后台 listener 并向指定 Zellij pane 注入任务。")
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

        # Build Reasonix command
        self._reasonix_bin = getattr(args, "reasonix_bin", "reasonix")
        continue_session = getattr(args, "continue_session", True) and self.has_saved_sessions(workspace)
        reasonix_cmd = self.build_tui_cmd(
            workspace, continue_session=continue_session,
            model=getattr(args, "model", None),
            extra_args=getattr(args, "reasonix_arg", None) or [],
        )

        log_line(log_path, f"launcher starting watcher: {' '.join(watcher_cmd)}")
        log_line(log_path, f"launcher starting reasonix: {' '.join(reasonix_cmd)}")

        # Start watcher with auto-restart
        MAX_WATCHER_RESTARTS = 10

        def start_watcher(*, restart_reason: str | None = None) -> subprocess.Popen:
            if restart_reason:
                log_line(log_path, f"launcher restarting watcher: {restart_reason}")
            proc = subprocess.Popen(
                watcher_cmd, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
                text=True, env=env, start_new_session=True,
            )
            log_line(log_path, f"reasonix-kanban listener started: pid={proc.pid}")
            return proc

        watcher = start_watcher()
        rc = 0
        try:
            reasonix_proc = subprocess.Popen(reasonix_cmd, cwd=str(workspace), env=env)
            watcher_restart_count = 0
            while True:
                reasonix_rc = reasonix_proc.poll()
                if reasonix_rc is not None:
                    rc = int(reasonix_rc)
                    break
                if not _pid_alive(reasonix_proc.pid):
                    for _ in range(3):
                        time.sleep(1.0)
                        p = reasonix_proc.poll()
                        if p is not None:
                            rc = int(p)
                            break
                    else:
                        log_line(log_path, f"reasonix pid {reasonix_proc.pid} dead but poll() not returned; forcing break")
                        rc = -1
                    break
                if watcher is not None:
                    watcher_rc = watcher.poll()
                    if watcher_rc is not None:
                        if not _should_restart_watcher(watcher_rc, reasonix_alive=True):
                            log_line(log_path, f"watcher exited cleanly rc={watcher_rc}; not restarting")
                            watcher = None
                            continue
                        watcher_restart_count += 1
                        if watcher_restart_count > MAX_WATCHER_RESTARTS:
                            log_line(log_path, f"watcher restart limit ({MAX_WATCHER_RESTARTS}) exceeded; stopping restart")
                            watcher = None
                            continue
                        log_line(log_path, f"watcher exited rc={watcher_rc}; restarting ({watcher_restart_count})")
                        time.sleep(min(30.0, 2.0 * watcher_restart_count))
                        watcher = start_watcher(restart_reason=f"previous watcher exited rc={watcher_rc}")
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
        return rc

    # ── Override: _build_parser with Reasonix-specific args ──
    def _build_parser(self) -> argparse.ArgumentParser:
        parser = super()._build_parser()
        parser.add_argument("--task-boundary-delay-s", type=float, default=8.0, help="Delay after task leaves running before claiming next")
        parser.add_argument("--task-timeout-s", type=float, default=listener_policy.INTERACTIVE_TASK_TIMEOUT_SECONDS, help="Reclaim task after this many seconds of TUI inactivity")
        parser.add_argument("--idle-pane-reclaim-s", type=float, default=listener_policy.INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS, help="Reclaim running task after pane stays idle this many seconds")
        parser.add_argument("--reasonix-bin", default="reasonix", help="Reasonix binary path/name (default: reasonix)")
        parser.add_argument("--continue", dest="continue_session", action="store_true", default=True, help="Resume most recent session (default: True)")
        parser.add_argument("--no-continue", dest="continue_session", action="store_false", help="Start a new session")
        return parser


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = ReasonixInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
