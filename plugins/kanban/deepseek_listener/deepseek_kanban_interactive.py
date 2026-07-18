#!/usr/bin/env python3
"""CodeWhaleInteractiveListener — Hermes Kanban listener for CodeWhale/DeepSeek-TUI.

CodeWhale (codewhale / codewhale-tui / deepseek-tui) has the most complex
interactive listener because it needs:

  1. Steering menu auto-dismiss (Tab+Enter when "Steering" popup appears)
  2. Progress watch / stalled-busy pane reclaim (screen fingerprint + session mtime)
  3. Orphaned running task adoption (when watcher restarts but TUI still has the prompt)
  4. Idle pane reclaim (task running but pane looks idle for too long)
  5. Task timeout (no session activity for too long)
  6. Watcher auto-restart in launcher (if watcher crashes while TUI still running)
  7. Provider/model/env customization (openrouter, opencode-go, etc.)

Idle markers:
  - "编写任务或使用 /" (composer prompt)
  - "输入消息" (alternative composer prompt)
  - "❯ " (suggested composer prompt)
  - "KANBAN_TASK_BOUNDARY" (our injection marker — pane is idle after previous task)

Busy markers:
  - "activity: thinking" (ghost: stale after response completes)
  - "工作中" (localized active response indicator)
  - "running" / "executing" (tool execution)

Injection strategy: write prompt to .md file, inject single-line
command referencing the file.  No \\n in injected text.
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
    _workspace_matches,
    reclaim_orphaned_running_task,
    sanitize_result,
)

# ── CodeWhale-specific imports ──
HERMES_REPO = HERMES_AGENT_ROOT
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402


# ──────────────────────────────────────────────
# CodeWhale-specific constants
# ──────────────────────────────────────────────

_DEEPSEEK_IDLE_MARKERS = (
    "编写任务或使用 /",
    "输入消息",
    "❯ ",
    "kanban_task_boundary",
)
_DEEPSEEK_BUSY_MARKERS = (
    "activity: thinking",
    "工作中",
    "running",
    "executing",
)
_DEEPSEEK_QUEUED_INPUT_MARKERS = ()

_STEERING_MARKER = "steering"
_STEERING_COOLDOWN_S = 30.0
_last_steering_dismiss_at: dict[str, float] = {}


# ──────────────────────────────────────────────
# CodeWhale-specific helpers
# ──────────────────────────────────────────────

def _proc_children(pid: int) -> list[int]:
    try:
        out = subprocess.run(
            ["ps", "--ppid", str(pid), "-o", "pid=", "--no-headers"],
            capture_output=True, text=True, timeout=3,
        )
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


def _proc_cmdline(pid: int) -> list[str]:
    try:
        return (Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode().split()
                if Path(f"/proc/{pid}/cmdline").exists() else [])
    except Exception:
        return []


def _cmdline_workspace_matches(args: list[str], workspace: Path) -> bool:
    for i, a in enumerate(args):
        if a in ("--workspace", "-w") and i + 1 < len(args):
            try:
                return Path(args[i + 1]).resolve() == workspace
            except Exception:
                pass
    return False


def _deepseek_tui_pids_for_workspace(workspace: Path) -> list[int]:
    parent_pid = os.getppid()
    out: list[int] = []
    for pid in _proc_children(parent_pid):
        if pid == os.getpid():
            continue
        a = _proc_cmdline(pid)
        if not a:
            continue
        exe = Path(a[0]).name
        if exe not in ("codewhale", "codewhale-tui", "deepseek-tui") and not any(
            p in ("codewhale", "codewhale-tui", "deepseek-tui") for p in a[:2]
        ):
            continue
        if _cmdline_workspace_matches(a, workspace):
            out.append(pid)
    return out


def _deepseek_tui_has_external_child(workspace: Path) -> bool:
    for pid in _deepseek_tui_pids_for_workspace(workspace):
        children = _proc_children(pid)
        if children:
            return True
    return False


def _sessions_latest_mtime() -> float:
    try:
        session_dir = Path.home() / ".codewhale" / "sessions"
        if not session_dir.is_dir():
            session_dir = Path.home() / ".deepseek" / "sessions"
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


def _auto_dismiss_steering(
    *, session: str, pane_id: str, screen: str,
    profile: str, log_path: Path,
) -> bool:
    global _last_steering_dismiss_at
    screen_lower = screen.lower()
    if _STEERING_MARKER not in screen_lower:
        return False
    now = time.time()
    last = _last_steering_dismiss_at.get(profile, 0.0)
    if now - last < _STEERING_COOLDOWN_S:
        return False
    log_line(log_path, "auto-dismissing codewhale Steering menu (Tab+Enter)")
    try:
        subprocess.run(
            ["zellij", "--session", session, "action", "write", "-p", str(pane_id), "9", "13"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=5,
        )
        _last_steering_dismiss_at[profile] = now
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        err = getattr(exc, "stderr", "") or str(exc)
        log_line(log_path, f"auto-dismiss Steering failed: {err}")
        return False


class _PaneProgressWatch:
    """Track pane screen changes to detect stalled-busy state."""
    def __init__(self) -> None:
        self.fingerprint: str = ""
        self.stalled_busy_seen_at: float | None = None
        self.latest_session_mtime: float = 0.0


def _observe_pane_progress(
    watch: _PaneProgressWatch, *, screen: str, now: float,
    latest_session_mtime: float, has_external_child: bool,
    reclaim_s: float,
) -> str | None:
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
    tail = "\n".join(_tail_nonempty_lines(screen)).lower()
    is_busy = any(marker in tail for marker in _DEEPSEEK_BUSY_MARKERS)
    is_queued = any(marker in tail for marker in _DEEPSEEK_QUEUED_INPUT_MARKERS)
    if not (is_busy or is_queued):
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
            "codewhale-interactive pane looked busy but made no progress "
            f"for {stalled_s:.0f}s (limit {reclaim_s:.0f}s)"
        )
    return None


def _cmd_arg_value(parts: list[str], *names: str) -> str | None:
    for i, p in enumerate(parts):
        if p in names and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _same_workspace_arg(parts: list[str], workspace: Path) -> bool:
    ws = _cmd_arg_value(parts, "--workspace", "-w")
    if ws:
        try:
            return Path(ws).resolve() == workspace
        except Exception:
            pass
    return False


def other_continue_deepseek_active(workspace: Path) -> bool:
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
        if not exe_names & {"codewhale", "codewhale-tui", "deepseek-tui"}:
            continue
        if "--continue" not in parts and "-c" not in parts:
            continue
        if _same_workspace_arg(parts, workspace):
            return True
    return False


def _read_hermes_dotenv_key(name: str) -> str | None:
    for candidate in (Path.home() / ".hermes" / ".env", Path.home() / ".env"):
        if candidate.is_file():
            try:
                for line in candidate.read_text().splitlines():
                    if line.strip().startswith(f"{name}="):
                        return line.split("=", 1)[1].strip().strip("\"'")
            except Exception:
                pass
    return None


def normalize_provider(provider: str | None) -> tuple[str, str]:
    if not provider:
        return "", ""
    p = provider.strip().lower()
    if p in ("openrouter", "topenrouter"):
        return "openrouter", "openrouter"
    if p in ("opencode-go", "opencode_go", "opencodego"):
        return "openai", "opencode-go"
    return provider, p


def default_model_for_provider(friendly_provider: str) -> str:
    if friendly_provider == "openrouter":
        return "deepseek/deepseek-r1-0528:free"
    if friendly_provider == "opencode-go":
        return "deepseek-v4-pro"
    return "deepseek-v4-pro"


def _should_restart_watcher(returncode: int | None) -> bool:
    if returncode is None:
        return True
    if returncode == 0:
        return False
    return True


def _should_continue_session(
    *,
    continue_requested: bool,
    full_access_requested: bool,
    other_active: bool,
    has_sessions: bool,
) -> bool:
    """Resume only when stale session permissions cannot override the launch mode."""
    return (
        continue_requested
        and not full_access_requested
        and not other_active
        and has_sessions
    )


# ──────────────────────────────────────────────
# CodeWhale subclass
# ──────────────────────────────────────────────

class CodeWhaleInteractiveListener(BaseInteractiveListener):
    agent_name = "CodeWhale"
    agent_slug = "codewhale"

    idle_markers = _DEEPSEEK_IDLE_MARKERS
    busy_markers = _DEEPSEEK_BUSY_MARKERS
    queued_input_markers = _DEEPSEEK_QUEUED_INPUT_MARKERS

    # ── Build TUI command ──
    def build_tui_cmd(
        self, workspace: Path, *,
        continue_session: bool = False,
        model: str | None = None,
        sandbox: str | None = None,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """Build the codewhale/deepseek-tui launch command.

        This is a simplified version; the full build_deepseek_cmd is used
        in the overridden launcher_main which handles provider/model/session logic.
        """
        cmd = ["codewhale", "run", "--workspace", str(workspace)]
        if continue_session:
            cmd.append("--continue")
        else:
            cmd.append("--fresh")
        cmd.extend(extra_args or [])
        return cmd

    def has_saved_sessions(self, workspace: Path) -> bool:
        tui_bin = shutil.which("codewhale") or shutil.which("codewhale-tui") or "deepseek-tui"
        try:
            result = subprocess.run(
                [tui_bin, "sessions"],
                capture_output=True, text=True, cwd=str(workspace), timeout=10,
            )
            out = result.stdout.strip()
            return bool(out) and "no sessions found" not in out.lower()
        except Exception:
            return False

    def inject_text(
        self, task_id: str, title: str, assignee: str,
        profile: str, prompt_path: Path, board: str,
    ) -> str:
        return (
            f"kanban_task_boundary 请读取 {prompt_path} 中的 Kanban 任务并执行。"
            f" [任务 {task_id}: {title}]"
        )

    def pane_label(self, task_id: str | None = None) -> str:
        profile = self._profile or "implementer"
        if task_id:
            return f"{profile}-deepseek [{task_id}]"
        return f"{profile}-deepseek listening"

    # ── Override: on_claim_pre_check with steering dismiss ──
    def on_claim_pre_check(self, args: argparse.Namespace, log_path: Path) -> bool:
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return False
        screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if not screen or not screen.strip():
            return False
        # Auto-dismiss steering before checking idle
        _auto_dismiss_steering(
            session=session, pane_id=str(pane_id), screen=screen,
            profile=args.profile, log_path=log_path,
        )
        return _pane_can_accept_new_kanban_task(
            screen, self.idle_markers, self.busy_markers, self.queued_input_markers,
        )

    # ── Override: on_claim_post_confirm (2-round idle check) ──
    def on_claim_post_confirm(self, args: argparse.Namespace, log_path: Path) -> bool:
        """DeepSeek-specific: confirm pane is stably idle before injection.

        Two-round check: verify idle, wait 1s, verify idle again.
        This avoids injecting during a brief idle flash between
        tool calls.
        """
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return True
        screen1 = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen1 is None:
            return True
        if not _pane_can_accept_new_kanban_task(
            screen1, self.idle_markers, self.busy_markers, self.queued_input_markers,
        ):
            return False
        time.sleep(1.0)
        screen2 = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen2 is None:
            return True
        return _pane_can_accept_new_kanban_task(
            screen2, self.idle_markers, self.busy_markers, self.queued_input_markers,
        )

    # ── Override: on_task_running_monitor (progress watch + idle reclaim) ──
    def on_task_running_monitor(
        self, args: argparse.Namespace, conn: Any,
        task_id: str, log_path: Path,
    ) -> None:
        """CodeWhale-specific: steering dismiss, idle pane reclaim, progress watch."""
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return

        idle_pane_reclaim_s = float(
            getattr(args, "idle_pane_reclaim_s", listener_policy.INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS)
        )
        task_timeout_s = float(getattr(args, "task_timeout_s", 0))

        # Steering dismiss
        screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen is not None:
            dismissed = _auto_dismiss_steering(
                session=session, pane_id=str(pane_id), screen=screen,
                profile=args.profile, log_path=log_path,
            )
            if dismissed:
                time.sleep(2.0)
                screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)

        # Idle pane reclaim
        if screen is not None and idle_pane_reclaim_s > 0:
            tail = "\n".join(_tail_nonempty_lines(screen)).lower()
            is_idle = any(marker in tail for marker in _DEEPSEEK_IDLE_MARKERS)
            is_busy = any(marker in tail for marker in _DEEPSEEK_BUSY_MARKERS)
            has_thinking = "activity: thinking" in tail
            other_busy = any(
                marker in tail for marker in _DEEPSEEK_BUSY_MARKERS if marker != "activity: thinking"
            )
            is_queued = any(marker in tail for marker in _DEEPSEEK_QUEUED_INPUT_MARKERS)
            truly_idle = (is_idle and not other_busy and not is_queued) or (is_idle and has_thinking and not other_busy)

            # Progress watch (stalled-busy detection)
            if not truly_idle:
                workspace = self._workspace
                stuck_reason = _observe_pane_progress(
                    self._progress_watch, screen=screen, now=time.time(),
                    latest_session_mtime=_sessions_latest_mtime(),
                    has_external_child=_deepseek_tui_has_external_child(workspace),
                    reclaim_s=idle_pane_reclaim_s,
                )
                if stuck_reason:
                    log_line(log_path, f"stalled busy pane reclaim {task_id}: {stuck_reason}")
                    _reclaim_task_without_signaling_worker(conn, task_id, reason=stuck_reason)
                    self._reset_active_task()

    # ── Override: on_watcher_loop_idle (steering dismiss while idle) ──
    def on_watcher_loop_idle(self, args: argparse.Namespace, conn: Any, log_path: Path) -> None:
        session = getattr(args, "zellij_session", "")
        pane_id = getattr(args, "zellij_pane_id", "")
        if not session or not pane_id:
            return
        screen = zellij_dump_screen(session=session, pane_id=str(pane_id), log_path=log_path)
        if screen is not None:
            _auto_dismiss_steering(
                session=session, pane_id=str(pane_id), screen=screen,
                profile=args.profile, log_path=log_path,
            )

    # ── Override: build_launch_env (provider/model customization) ──
    def build_launch_env(self, args: argparse.Namespace) -> dict[str, str]:
        env = super().build_launch_env(args)
        deepseek_provider = getattr(args, "provider", None)
        provider_str, provider_label = normalize_provider(deepseek_provider)
        deepseek_model = getattr(args, "model", None) or default_model_for_provider(provider_label)
        if provider_str:
            env["DEEPSEEK_PROVIDER"] = provider_str
            if provider_label == "openrouter":
                env["OPENROUTER_BASE_URL"] = "https://tp-api.chinadatapay.com:8000/v1"
            elif provider_label == "opencode-go":
                env["OPENAI_BASE_URL"] = "https://opencode.ai/zen/go/v1"
        if getattr(args, "model", None):
            env["DEEPSEEK_MODEL"] = deepseek_model
        topenrouter_key = env.get("TOPENROUTER_API_KEY") or _read_hermes_dotenv_key("TOPENROUTER_API_KEY")
        opencode_key = env.get("OPENCODE_GO_API_KEY") or _read_hermes_dotenv_key("OPENCODE_GO_API_KEY")
        if topenrouter_key:
            env["OPENROUTER_API_KEY"] = topenrouter_key
        if opencode_key:
            env["OPENAI_API_KEY"] = opencode_key
        if getattr(args, "yolo", True):
            env["DEEPSEEK_YOLO"] = "true"
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

    def _reset_active_task(self) -> None:
        """Reset all active-task tracking state."""
        self._idle_pane_seen_at = None
        self._progress_watch = _PaneProgressWatch()
        self._last_task_transition_at = time.time()
        self._api_retry_count = 0
        self._api_retry_first_at = None

    # ── Override: watcher_main with CodeWhale-specific monitoring ──
    def watcher_main(self, args: argparse.Namespace) -> int:
        """CodeWhale watcher with idle-pane reclaim, progress watch, and task timeout."""
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
        from base_listener import _handle_stop
        sig.signal(sig.SIGINT, _handle_stop)
        sig.signal(sig.SIGTERM, _handle_stop)

        if not workspace.exists():
            log_line(log_path, f"workspace does not exist: {workspace}")
            return 2

        zellij_session = getattr(args, "zellij_session", "")
        zellij_pane_id = getattr(args, "zellij_pane_id", "")
        if not zellij_session or not zellij_pane_id:
            log_line(log_path, f"missing zellij session/pane id; cannot inject into CodeWhale TUI")
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
            f"poll={poll_s:g}s agent=CodeWhale",
        )
        zellij_rename_pane(
            session=zellij_session, pane_id=str(zellij_pane_id),
            name=self.pane_label(), log_path=log_path,
        )

        startup_delay = getattr(args, "startup_delay_s", 0) or 0
        if startup_delay > 0:
            time.sleep(startup_delay)

        # ── Persistent DB connection (same as base) ──
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
            while True:
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
                        # ── Liveness check ──
                        if now - self._last_liveness_check >= 30.0:
                            latest_mtime = _sessions_latest_mtime()
                            if latest_mtime > 0 and latest_mtime > self._last_activity_at:
                                log_line(log_path, f"liveness: session updated → resetting timeout")
                                self._last_activity_at = now

                            # Steering dismiss + idle/progress monitoring
                            screen = zellij_dump_screen(
                                session=zellij_session, pane_id=str(zellij_pane_id), log_path=log_path,
                            )
                            if screen is not None:
                                dismissed = _auto_dismiss_steering(
                                    session=zellij_session, pane_id=str(zellij_pane_id),
                                    screen=screen, profile=args.profile, log_path=log_path,
                                )
                                if dismissed:
                                    self._idle_pane_seen_at = None
                                    time.sleep(2.0)
                                    screen = zellij_dump_screen(
                                        session=zellij_session, pane_id=str(zellij_pane_id), log_path=log_path,
                                    )

                            # Idle pane reclaim
                            if screen is not None and idle_pane_reclaim_s > 0:
                                tail = "\n".join(_tail_nonempty_lines(screen)).lower()
                                is_idle = any(m in tail for m in _DEEPSEEK_IDLE_MARKERS)
                                has_thinking = "activity: thinking" in tail
                                other_busy = any(
                                    m in tail for m in _DEEPSEEK_BUSY_MARKERS if m != "activity: thinking"
                                )
                                is_queued = any(m in tail for m in _DEEPSEEK_QUEUED_INPUT_MARKERS)
                                truly_idle = (is_idle and not other_busy and not is_queued) or (is_idle and has_thinking and not other_busy)

                                if truly_idle:
                                    if self._idle_pane_seen_at is None:
                                        self._idle_pane_seen_at = now
                                        log_line(log_path, f"idle pane observed for active task {active_task}")
                                    # Before reclaiming: check if this is an API failure idle
                                    # and try injecting "继续" to retry (up to API_RETRY_MAX)
                                    if screen is not None:
                                        retried = self.check_api_failure_retry(
                                            session=zellij_session, pane_id=str(zellij_pane_id),
                                            screen=screen, task_id=active_task,
                                            log_path=log_path,
                                        )
                                        if retried:
                                            # Retry injected; reset idle timer so reclaim doesn't fire prematurely
                                            self._idle_pane_seen_at = now
                                            continue
                                    idle_s = now - self._idle_pane_seen_at
                                    if idle_s >= idle_pane_reclaim_s:
                                        log_line(log_path, f"idle pane reclaim {active_task}: idle {idle_s:.0f}s >= {idle_pane_reclaim_s:.0f}s")
                                        _reclaim_task_without_signaling_worker(
                                            conn, active_task,
                                            reason=f"codewhale-interactive pane stayed idle while task remained running for {idle_s:.0f}s (limit {idle_pane_reclaim_s:.0f}s)",
                                        )
                                        active_task = None
                                        active_run_id = None
                                        self._reset_active_task()
                                        last_hb = 0.0
                                        continue
                                else:
                                    self._idle_pane_seen_at = None

                                # Progress watch
                                stuck_reason = _observe_pane_progress(
                                    self._progress_watch, screen=screen, now=now,
                                    latest_session_mtime=latest_mtime,
                                    has_external_child=_deepseek_tui_has_external_child(workspace),
                                    reclaim_s=idle_pane_reclaim_s,
                                )
                                if stuck_reason:
                                    log_line(log_path, f"stalled busy pane reclaim {active_task}: {stuck_reason}")
                                    _reclaim_task_without_signaling_worker(conn, active_task, reason=stuck_reason)
                                    active_task = None
                                    active_run_id = None
                                    self._reset_active_task()
                                    last_hb = 0.0
                                    continue
                            self._last_liveness_check = now

                        # ── Task timeout ──
                        if task_timeout_s > 0 and now - self._last_activity_at > task_timeout_s:
                            idle_s = now - self._last_activity_at
                            log_line(log_path, f"task timeout {active_task}: idle {idle_s:.0f}s > {task_timeout_s:.0f}s; reclaiming")
                            _reclaim_task_without_signaling_worker(
                                conn, active_task,
                                reason=f"codewhale-interactive idle timeout after {idle_s:.0f}s (limit {task_timeout_s:.0f}s)",
                            )
                            active_task = None
                            active_run_id = None
                            self._reset_active_task()
                            last_hb = 0.0
                            continue

                        # ── Heartbeat ──
                        if now - last_hb >= max(15.0, min(float(args.ttl) / 3.0, 120.0)):
                            try:
                                kb.heartbeat_claim(conn, active_task, ttl_seconds=args.ttl, claimer=self._claim_lock())
                                kb.heartbeat_worker(
                                    conn, active_task,
                                    note="codewhale-interactive waiting for complete/block from CodeWhale TUI",
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

                # ── Task boundary delay ──
                remaining_delay = task_boundary_delay_s - (time.time() - self._last_task_transition_at)
                if remaining_delay > 0.0:
                    time.sleep(min(remaining_delay, poll_s, 5.0))
                    continue

                # ── Idle loop: steering dismiss ──
                self.on_watcher_loop_idle(args, conn, log_path)

                # ── Orphan adoption ──
                reclaim_orphaned_running_task(args, log_path=log_path, conn=conn)

                # ── Claim and inject ──
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
            print(f"错误: 没检测到 Zellij session/pane，无法把 Kanban 任务注入 CodeWhale TUI。", file=sys.stderr)
            return 2

        args.zellij_session = zellij_session
        args.zellij_pane_id = zellij_pane_id

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

        deepseek_provider, provider_label = normalize_provider(getattr(args, "provider", None))
        deepseek_model = getattr(args, "model", None) or default_model_for_provider(provider_label)

        print(f"CodeWhale interactive kanban mode")
        print(f"  board:     {board}")
        print(f"  profile:   {args.profile}")
        print(f"  claims:    {', '.join(claim_assignees(args))}")
        print(f"  workspace: {workspace}")
        print(f"  pane:      {zellij_session}:{zellij_pane_id}")
        print(f"  provider:  {provider_label} ({deepseek_provider})")
        print(f"  model:     {deepseek_model}")
        print(f"  log:       {log_path}")
        print("")
        print(f"按 Enter 进入 interactive CodeWhale；后台 listener 会按优先级 claim ready 任务并注入到当前 TUI。")
        print(f"codewhale-kanban listener armed: profile={args.profile} board={board} poll={poll_label} workspace={workspace}")

        watch_only = getattr(args, "watch_only", False)
        if watch_only:
            print("listener-only 模式：不会启动 CodeWhale，只运行后台 listener 并向指定 Zellij pane 注入任务。")
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

        # ── Build deepseek command with session/provider logic ──
        base_workspace = str(workspace)
        profile = args.profile or "implementer"
        session_workspace = str(Path(base_workspace) / ".ds-sessions" / profile)
        Path(session_workspace).mkdir(parents=True, exist_ok=True)
        ws_path = Path(session_workspace)
        other_active = other_continue_deepseek_active(ws_path)
        has_sessions = self.has_saved_sessions(ws_path)
        full_access_requested = getattr(args, "yolo", True)
        want_continue = _should_continue_session(
            continue_requested=getattr(args, "continue_session", True),
            full_access_requested=full_access_requested,
            other_active=other_active,
            has_sessions=has_sessions,
        )

        if want_continue:
            runtime_bin = shutil.which("codewhale-tui") or getattr(args, "deepseek_tui_bin", "codewhale")
            deepseek_cmd = [runtime_bin, "--workspace", session_workspace, "--continue"]
        else:
            deepseek_cmd = [getattr(args, "deepseek_tui_bin", "codewhale"), "run", "--workspace", session_workspace, "--fresh"]

        if full_access_requested:
            deepseek_cmd.append("--yolo")
        for extra in getattr(args, "deepseek_arg", None) or []:
            deepseek_cmd.append(extra)

        log_line(log_path, f"launcher starting watcher: {' '.join(watcher_cmd)}")
        log_line(log_path, f"launcher starting codewhale: {' '.join(deepseek_cmd)}")

        # ── Start watcher with auto-restart ──
        MAX_WATCHER_RESTARTS = 10

        def start_watcher(*, restart_reason: str | None = None) -> subprocess.Popen:
            if restart_reason:
                log_line(log_path, f"launcher restarting watcher: {restart_reason}")
            proc = subprocess.Popen(
                watcher_cmd, stdin=subprocess.DEVNULL, stdout=log_f, stderr=subprocess.STDOUT,
                text=True, env=env, start_new_session=True,
            )
            log_line(log_path, f"codewhale-kanban listener started: pid={proc.pid}")
            return proc

        watcher = start_watcher()
        rc = 0
        try:
            deepseek_proc = subprocess.Popen(deepseek_cmd, cwd=str(workspace), env=env)
            watcher_restart_count = 0
            while True:
                deepseek_rc = deepseek_proc.poll()
                if deepseek_rc is not None:
                    rc = int(deepseek_rc)
                    break
                if not _pid_alive(deepseek_proc.pid):
                    for _ in range(3):
                        time.sleep(1.0)
                        p = deepseek_proc.poll()
                        if p is not None:
                            rc = int(p)
                            break
                    else:
                        log_line(log_path, f"deepseek pid {deepseek_proc.pid} dead but poll() not returned; forcing break")
                        rc = -1
                    break
                if watcher is not None:
                    watcher_rc = watcher.poll()
                    if watcher_rc is not None:
                        if not _should_restart_watcher(watcher_rc):
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
            log_line(log_path, f"codewhale exited rc={rc}; stopping watcher pid={watcher_pid}")
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

    # ── Override: _build_parser with CodeWhale-specific args ──
    def _build_parser(self) -> argparse.ArgumentParser:
        parser = super()._build_parser()
        parser.add_argument("--task-boundary-delay-s", type=float, default=8.0, help="Delay after task leaves running before claiming next")
        parser.add_argument("--task-timeout-s", type=float, default=listener_policy.INTERACTIVE_TASK_TIMEOUT_SECONDS, help="Reclaim task after this many seconds of TUI inactivity")
        parser.add_argument("--idle-pane-reclaim-s", type=float, default=listener_policy.INTERACTIVE_IDLE_PANE_RECLAIM_SECONDS, help="Reclaim running task after pane stays idle this many seconds")
        parser.add_argument("--provider", default=None, help="Provider override: openrouter/topenrouter or opencode-go")
        parser.add_argument("--yolo", action="store_true", default=True, help="Auto-approve all tools (default: True)")
        parser.add_argument("--no-yolo", dest="yolo", action="store_false", help="Disable YOLO mode")
        parser.add_argument("--deepseek-tui-bin", default="codewhale", help="CodeWhale dispatcher binary (default: codewhale)")
        parser.add_argument("--continue", dest="continue_session", action="store_true", default=True, help="Resume most recent session (default: True)")
        parser.add_argument("--no-continue", dest="continue_session", action="store_false", help="Start a new session")
        return parser


# ── Entry point ──
def main(argv: list[str] | None = None) -> int:
    listener = CodeWhaleInteractiveListener()
    return listener.main(argv)


if __name__ == "__main__":
    sys.exit(main())
