#!/usr/bin/env python3
"""Codex-backed Hermes Kanban listener.

This is the canonical source for exposing Hermes kanban tasks to Codex CLI.
It intentionally reuses Hermes' kanban_db module instead of reimplementing the
SQLite schema.  The Codex plugin installed under ~/.agents / ~/.codex should
only symlink or thin-wrap this file.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

# Source layout: <repo>/plugins/kanban/codex_listener/codex_kanban_listener.py
HERMES_REPO = Path(__file__).resolve().parents[3]
if str(HERMES_REPO) not in sys.path:
    sys.path.insert(0, str(HERMES_REPO))

from hermes_cli import kanban_db as kb  # noqa: E402
from hermes_cli import kanban_listener_policy as listener_policy  # noqa: E402
from hermes_cli import kanban_worker_runtime as worker_runtime  # noqa: E402

RESULT_SCHEMA = Path(__file__).resolve().parent / "assets" / "kanban_result_schema.json"


class ListenerStopped(Exception):
    pass


_STOP = False


def _handle_stop(signum: int, frame: Any) -> None:  # noqa: ARG001
    global _STOP
    _STOP = True


def _wait_agent_idle(tui_pid: int, *, task_id: str = "", max_wait: float = 10.0, board: str = "") -> bool:
    """Block until the TUI agent at *tui_pid* appears idle (not streaming output).

    Samples CPU jiffies in ~2 s windows.  Logs each busy cycle.
    Returns True if agent became idle — safe to inject.
    Returns False if *max_wait* elapsed — task has been reclaimed so
    another listener can pick it up.
    """
    deadline = time.time() + max_wait
    waited = 0.0
    sample = listener_policy.AGENT_IDLE_SAMPLE_SECONDS
    while time.time() < deadline:
        if not listener_policy.agent_pid_is_busy(tui_pid, sample_s=sample):
            return True  # idle → go
        waited += sample
        if waited >= 3.0:
            tag = f" [{task_id}]" if task_id else ""
            log(f"waiting for agent to become idle{tag} ({waited:.0f}s elapsed)")
    # Timed out — reclaim so another listener can take it
    _reclaim_and_release(task_id, board=board, reason=f"agent busy for {max_wait:.0f}s, releasing for other listeners")
    return False


def _reclaim_and_release(task_id: str, *, board: str, reason: str) -> None:
    """Reclaim *task_id* back to ready so other listeners can claim it."""
    try:
        with kb.connect(board=board) as conn:
            kb.reclaim_task(conn, task_id, reason=reason)
        log(f"released {task_id}: {reason}")
    except Exception as exc:
        log(f"failed to release {task_id}: {exc}")


def now_s() -> int:
    return int(time.time())


def claim_lock() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def task_log_path(task_id: str, board: str | None) -> Path:
    log_dir = kb.worker_logs_dir(board=board)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{task_id}.log"


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def claim_assignees(args: argparse.Namespace) -> list[str]:
    """Return assignees this listener may claim, in priority order."""
    return worker_runtime.claim_assignees_from_args(args, default_profile="planner")


def build_prompt(*, board: str, profile: str, task_id: str, task_assignee: str, context: str, workspace: Path) -> str:
    assist_note = ""
    if task_assignee != profile:
        assist_note = (
            f"\n        当前 listener profile 是 {profile}，但本任务 assignee/role 是 {task_assignee}。"
            f"本次必须按 {task_assignee} 职责完成，不要按 {profile} 职责改写目标。\n"
        )
    return textwrap.dedent(
        f"""
        你现在是 Hermes Kanban 中的 Codex profile：{profile}。
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
        - 然后输入 `/exit` 或 Ctrl+D 退出 Codex，listener 会自动回收。

        下面是 Hermes Kanban worker context：

        {context}
        """
    ).strip()


def parse_codex_result(output_file: Path, fallback_text: str) -> dict[str, Any]:
    raw = ""
    if output_file.exists():
        raw = output_file.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        raw = fallback_text.strip()
    if not raw:
        return {
            "status": "blocked",
            "summary": "Codex returned no final output",
            "details": "",
            "metadata": {},
            "block_reason": "Codex returned no final output",
        }
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Some model/provider paths may wrap JSON in prose.  Try extracting the
    # outermost object before falling back to a plain completion.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {
        "status": "done",
        "summary": raw.splitlines()[0][:300],
        "details": raw,
        "metadata": {"codex_result_parse": "plain_text_fallback"},
    }


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    status = str(result.get("status") or "done").strip().lower()
    if status not in {"done", "blocked"}:
        status = "done"
    summary = str(result.get("summary") or "").strip()
    details = str(result.get("details") or "").strip()
    metadata = result.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    created = result.get("created_task_ids") or metadata.get("created_task_ids")
    if isinstance(created, list):
        metadata["created_task_ids"] = [str(x) for x in created if x]
    block_reason = str(result.get("block_reason") or "").strip()
    if not summary:
        summary = (details.splitlines()[0] if details else "Codex task completed")[:300]
    return {
        "status": status,
        "summary": summary,
        "details": details or summary,
        "metadata": metadata,
        "block_reason": block_reason,
    }


def _result_failure_text(result: dict[str, Any]) -> str:
    return "\n".join(
        str(result.get(key) or "")
        for key in ("summary", "details", "block_reason")
        if result.get(key)
    )


def _is_provider_failure_result(rc: int, result: dict[str, Any]) -> bool:
    """True for transient provider/API failures that should cooldown+retry.

    Codex exits non-zero both for model-provider failures and for real task
    errors.  Only the former should stay claimed quietly; user/task failures
    still block immediately so they surface for human intervention.
    """
    if rc == 0:
        return False
    return listener_policy.provider_failure_text(_result_failure_text(result))


def _noop_signal(_pid: int, _sig: int) -> None:
    """Never signal an already-finished Codex child during retry requeue.

    The worker_pid in the task row points at the Codex child, not the listener.
    After a 10-minute cooldown that PID may be gone or even reused, so a retry
    requeue must clear DB ownership without sending SIGTERM/SIGKILL.
    """


def _cooldown_heartbeat_interval(ttl_s: int) -> float:
    return max(5.0, min(float(ttl_s) / 3.0, 120.0))


def _wait_provider_retry_cooldown(
    *,
    board: str,
    task_id: str,
    expected_run_id: int | None,
    ttl_s: int,
    reason: str,
    now_fn=time.time,
    sleep_fn=time.sleep,
) -> str:
    """Hold the claim and heartbeat during shared provider retry cooldown.

    Returns 'recovered' if the task completed during cooldown (server came back),
    None if the full cooldown elapsed.
    """
    cooldown = float(listener_policy.RETRY_COOLDOWN_SECONDS)
    if cooldown <= 0:
        return "timeout"
    deadline = float(now_fn()) + cooldown
    next_hb = 0.0
    interval = _cooldown_heartbeat_interval(ttl_s)
    while not _STOP:
        now = float(now_fn())
        remaining = deadline - now
        if remaining <= 0:
            return "timeout"
        if now >= next_hb:
            with kb.connect(board=board) as conn:
                # Check if task completed (server recovered)
                row = conn.execute(
                    "SELECT status FROM tasks WHERE id=?", (task_id,)
                ).fetchone()
                if row and row["status"] in ("done", "blocked"):
                    log(f"cooldown aborted: task {task_id} → {row['status']} (server recovered)")
                    return "recovered"
                kb.heartbeat_claim(conn, task_id, ttl_seconds=ttl_s, claimer=claim_lock())
                kb.heartbeat_worker(
                    conn,
                    task_id,
                    note=f"codex provider failure cooldown: {reason[:200]}",
                    expected_run_id=expected_run_id,
                )
            next_hb = now + interval
        sleep_fn(min(remaining, max(1.0, min(interval, 30.0))))
    raise ListenerStopped()


def _codex_provider_failure_retry(
    *,
    board: str,
    task_id: str,
    expected_run_id: int | None,
    reason: str,
    ttl_s: int,
    now_fn=time.time,
    sleep_fn=time.sleep,
) -> str:
    """Cooldown after provider failure, then requeue or block if exhausted.

    Returns: "ready", "blocked", "recovered", or "changed".
    """
    result = _wait_provider_retry_cooldown(
        board=board,
        task_id=task_id,
        expected_run_id=expected_run_id,
        ttl_s=ttl_s,
        reason=reason,
        now_fn=now_fn,
        sleep_fn=sleep_fn,
    )
    if result == "recovered":
        return "recovered"  # task completed during cooldown, server came back

    with kb.connect(board=board) as conn:
        row = conn.execute(
            "SELECT status, current_run_id, max_retries, consecutive_failures "
            "FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
        if not row:
            return "changed"
        if row["status"] != "running":
            return "changed"
        if expected_run_id is not None and row["current_run_id"] != expected_run_id:
            return "changed"

        configured_max = row["max_retries"] if row["max_retries"] is not None else 10
        max_retries = max(int(configured_max), listener_policy.MIN_PROVIDER_FAILURE_SILENT_RETRIES)
        consecutive = int(row["consecutive_failures"] or 0) + 1
        short_reason = reason[:500]

        if consecutive > max_retries + 1:
            ok = kb.block_task(
                conn,
                task_id,
                reason=f"codex provider failure retries exhausted ({consecutive}/{max_retries}): {short_reason}",
                expected_run_id=expected_run_id,
            )
            if ok:
                conn.execute(
                    "UPDATE tasks SET consecutive_failures=?, last_failure_error=? WHERE id=?",
                    (consecutive, short_reason, task_id),
                )
                kb._append_event(conn, task_id, "codex_provider_failure_gave_up", {
                    "reason": short_reason,
                    "failures": consecutive,
                    "max_retries": max_retries,
                    "cooldown_seconds": listener_policy.RETRY_COOLDOWN_SECONDS,
                })
                log(f"blocked {task_id}: codex provider failure exhausted {consecutive}/{max_retries}")
                return "blocked"
            return "changed"

        ok = kb.reclaim_task(
            conn,
            task_id,
            reason=f"codex provider failure retry after cooldown: {short_reason}",
            signal_fn=_noop_signal,
        )
        if ok:
            # reclaim_task intentionally clears operator-stale failures.  This
            # retry path is automatic, so restore/increment the failure budget.
            conn.execute(
                "UPDATE tasks SET consecutive_failures=?, last_failure_error=? WHERE id=?",
                (consecutive, short_reason, task_id),
            )
            kb._append_event(conn, task_id, "codex_provider_failure_retry", {
                "reason": short_reason,
                "failures": consecutive,
                "max_retries": max_retries,
                "cooldown_seconds": listener_policy.RETRY_COOLDOWN_SECONDS,
            })
            log(f"requeued {task_id}: codex provider failure retry {consecutive}/{max_retries}")
            return "ready"
        return "changed"


def has_saved_codex_sessions(workspace: Path) -> bool:
    """Check whether Codex has saved interactive sessions for this workspace."""
    sessions_root = Path.home() / ".codex" / "sessions"
    if not sessions_root.exists():
        return False
    ws = str(workspace)
    try:
        for path in sorted(
            sessions_root.glob("**/*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            if not path.is_file():
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    first_line = f.readline()
                if ws in first_line:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False

def run_codex_for_task(
    *,
    task: kb.Task,
    board: str,
    profile: str,
    workspace: Path,
    context: str,
    sandbox: str | None,
    model: str | None,
    codex_bin: str,
    extra_args: list[str],
    poll_s: float,
    ttl_s: int,
    task_delivery: str,  # "inject" only
) -> tuple[int, dict[str, Any], Path]:
    prompt = build_prompt(
        board=board,
        profile=profile,
        task_id=task.id,
        task_assignee=getattr(task, "assignee", None) or profile,
        context=context,
        workspace=workspace,
    )
    log_path = task_log_path(task.id, board)
    startup_file = workspace / ".codex-kanban-startup.md"
    startup_file.write_text(prompt, encoding="utf-8")

    session_workspace = Path(os.environ.get("CODEX_KANBAN_WORKSPACE", str(workspace)))
    has_session = has_saved_codex_sessions(session_workspace)

    if has_session:
        cmd = [codex_bin, "resume", "--last", "--dangerously-bypass-approvals-and-sandbox"]
        mode_label = "resume"
    else:
        cmd = [codex_bin, "--dangerously-bypass-approvals-and-sandbox", "--cd", str(session_workspace), str(startup_file)]
        mode_label = "fresh"

    if sandbox and not has_session:
        cmd.insert(1, "--sandbox")
        cmd.insert(2, sandbox)
    if model:
        cmd.extend(["--model", model])
    cmd.extend(extra_args)

    env = os.environ.copy()
    env.update({
        "HERMES_KANBAN_TASK": task.id,
        "HERMES_KANBAN_BOARD": board,
        "HERMES_KANBAN_PROFILE": profile,
        "HERMES_KANBAN_TASK_ASSIGNEE": getattr(task, "assignee", None) or profile,
        "HERMES_KANBAN_WORKSPACE": str(session_workspace),
    })

    log(f"launch Codex ({mode_label}/{task_delivery}) for {task.id}: {' '.join(cmd[:4])} ...")
    with open(log_path, "ab") as log_f:
        log_f.write(f"\n=== codex-kanban {mode_label}/{task_delivery} start {time.strftime('%Y-%m-%d %H:%M:%S')} task={task.id} ===\n".encode())
        log_f.flush()

    proc = subprocess.Popen(cmd, cwd=str(session_workspace), env=env)
    with kb.connect(board=board) as conn:
        try:
            kb._set_worker_pid(conn, task.id, proc.pid)
        except Exception:
            pass

    # --- Inject mode: send task_id to TUI ---
    # IMPORTANT: inject text must NOT contain \n (LF).
    # In PTY raw mode, LF (0x0A) != Enter (CR, 0x0D).
    # LF only inserts a newline in the input buffer without submitting.
    if task_delivery == "inject":
        raw_pane = os.environ.get("ZELLIJ_PANE_ID", "")
        if raw_pane:
            pane_id = f"terminal_{raw_pane}" if raw_pane.isdigit() else raw_pane
            time.sleep(2.0)
            inject = f"work on kanban task {task.id}"
            subprocess.run(["zellij", "action", "write-chars", "-p", pane_id, inject], timeout=5, check=False)
            time.sleep(0.8)
            subprocess.run(["zellij", "action", "send-keys", "-p", pane_id, "Enter"], timeout=5, check=False)
            log(f"injected to {pane_id}: {inject.strip()}")

    last_hb = 0.0
    last_status_check = 0.0
    while True:
        if _STOP:
            try:
                proc.terminate()
            except Exception:
                pass
            raise ListenerStopped()

        rc = proc.poll()
        if rc is not None:
            break

        time.sleep(max(1.0, min(poll_s, 5.0)))
        now = time.time()
        if now - last_hb >= max(30.0, min(float(ttl_s) / 3.0, 120.0)):
            with kb.connect(board=board) as conn:
                kb.heartbeat_claim(conn, task.id, ttl_seconds=ttl_s, claimer=claim_lock())
                kb.heartbeat_worker(conn, task.id, note="codex interactive running", expected_run_id=task.current_run_id)
            last_hb = now
        # Check if agent already completed/blocked via kanban CLI while codex still runs.
        if now - last_status_check >= 10.0:
            with kb.connect(board=board) as conn:
                row = conn.execute(
                    "SELECT status FROM tasks WHERE id=?", (task.id,)
                ).fetchone()
                if row and row["status"] in ("done", "blocked"):
                    log(f"task {task.id} transitioned to {row['status']} while codex still running; waiting for codex to exit")
            last_status_check = now

    # Codex exited — check if the agent already completed/blocked via kanban CLI.
    with kb.connect(board=board) as conn:
        row = conn.execute(
            "SELECT status, result FROM tasks WHERE id=?",
            (task.id,),
        ).fetchone()
        if row and row["status"] in ("done", "blocked"):
            result = {
                "status": row["status"],
                "summary": (row["result"] or "")[:300],
                "details": row["result"] or "",
                "metadata": {"codex_exit_rc": int(rc)},
            }
            if row["status"] == "blocked":
                result["block_reason"] = row["result"] or "agent blocked via kanban CLI"
            log(f"codex exited rc={rc}; task already {row['status']} in DB")
            with open(log_path, "ab") as log_f:
                log_f.write(f"\n=== codex-kanban interactive exit rc={rc} task_status={row['status']} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
            return int(rc), sanitize_result(result), log_path

        # Agent didn't call kanban complete/block — treat as failure.
        with open(log_path, "ab") as log_f:
            log_f.write(f"\n=== codex-kanban interactive exit rc={rc} task_still_running at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        log(f"codex exited rc={rc} but task {task.id} still running (agent may have forgotten kanban complete)")
        result = {
            "status": "done" if rc == 0 else "blocked",
            "summary": f"Codex interactive exited with rc={rc}" if rc != 0 else "Codex completed without calling kanban complete",
            "details": f"Codex process exited rc={rc}. Agent may have forgotten to call `hermes kanban complete`.",
            "metadata": {"codex_exit_rc": int(rc), "interactive_mode": True},
            "block_reason": f"Codex exited rc={rc} without completing kanban task" if rc != 0 else "",
        }
        return int(rc), sanitize_result(result), log_path


def handle_one_task(args: argparse.Namespace) -> bool:
    board = args.board or kb.get_current_board()
    profile = args.profile

    # --- Inject mode: keep TUI always visible ---
    # Step 1: TUI always kept alive (restart immediately if dead)
    proc = getattr(args, "_tui_proc", None)
    if proc is None or proc.poll() is not None:
        # TUI exited — if task was active, enter cooldown (non-blocking)
        active_task_id = getattr(args, "_active_task_id", None)
        if active_task_id is not None and proc is not None:
            with kb.connect(board=board) as conn:
                row = conn.execute(
                    "SELECT status, current_run_id FROM tasks WHERE id=?",
                    (active_task_id,),
                ).fetchone()
                if row and row["status"] == "running":
                    if not getattr(args, "_cooldown_until", 0):
                        log(f"TUI exited with active task {active_task_id} — start 10min cooldown")
                        args._cooldown_until = time.time() + 600
                else:
                    log(f"task {active_task_id} already {row['status'] if row else 'gone'}, clearing")
                    args._active_task_id = None
                    args._cooldown_until = 0

        # ALWAYS restart TUI immediately
        session_workspace = Path(os.environ.get("CODEX_KANBAN_WORKSPACE", "/home/wyr/code/Egomotion4D"))
        has_session = has_saved_codex_sessions(session_workspace)
        if has_session:
            cmd = [args.codex_bin, "resume", "--last", "--dangerously-bypass-approvals-and-sandbox"]
        else:
            cmd = [args.codex_bin, "--dangerously-bypass-approvals-and-sandbox", "--cd", str(session_workspace)]
            if args.sandbox:
                cmd.insert(1, "--sandbox")
                cmd.insert(2, args.sandbox)
        if args.model:
            cmd.extend(["--model", args.model])
        cmd.extend(args.codex_arg or [])
        env = os.environ.copy()
        env.update({"HERMES_KANBAN_BOARD": board, "HERMES_KANBAN_PROFILE": profile})
        args._tui_proc = subprocess.Popen(cmd, cwd=str(session_workspace), env=env)
        log(f"inject TUI {'restarted' if proc else 'launched'}: {'resume' if has_session else 'fresh'}")
        time.sleep(2.0)

    # Step 2: handle cooldown state
    cooldown_until = getattr(args, "_cooldown_until", 0)
    if cooldown_until:
        active_id = args._active_task_id
        if not active_id:
            args._cooldown_until = 0  # shouldn't happen, but clean up
        elif time.time() >= cooldown_until:
            # Cooldown expired — requeue task for retry
            with kb.connect(board=board) as conn:
                row = conn.execute(
                    "SELECT status, current_run_id, max_retries, consecutive_failures FROM tasks WHERE id=?",
                    (active_id,),
                ).fetchone()
                if row and row["status"] == "running":
                    configured_max = row["max_retries"] if row["max_retries"] is not None else 10
                    max_retries = max(int(configured_max), listener_policy.MIN_PROVIDER_FAILURE_SILENT_RETRIES)
                    consecutive = int(row["consecutive_failures"] or 0) + 1
                    if consecutive > max_retries + 1:
                        kb.block_task(conn, active_id,
                                      reason=f"provider failure retries exhausted ({consecutive}/{max_retries})",
                                      expected_run_id=row["current_run_id"])
                        conn.execute("UPDATE tasks SET consecutive_failures=?, last_failure_error=? WHERE id=?",
                                     (consecutive, "cooldown retries exhausted", active_id))
                        log(f"blocked {active_id}: retries exhausted")
                    else:
                        kb.reclaim_task(conn, active_id,
                                        reason=f"cooldown expired, requeue for retry {consecutive}/{max_retries}",
                                        signal_fn=_noop_signal)
                        conn.execute("UPDATE tasks SET consecutive_failures=?, last_failure_error=? WHERE id=?",
                                     (consecutive, "cooldown retry", active_id))
                        log(f"requeued {active_id}: cooldown retry {consecutive}/{max_retries}")
                args._active_task_id = None
                args._cooldown_until = 0
        else:
            # Still in cooldown — check if task completed via TUI
            with kb.connect(board=board) as conn:
                row = conn.execute("SELECT status FROM tasks WHERE id=?", (active_id,)).fetchone()
                if row and row["status"] in ("done", "blocked"):
                    log(f"task {active_id} completed during cooldown — clearing")
                    args._active_task_id = None
                    args._cooldown_until = 0
        return False  # Don't claim during cooldown

    # Step 3: try to claim a task and inject
    policy = worker_runtime.claim_policy_from_args(args, default_profile="planner")
    with kb.connect(board=board) as conn:
        kb.release_stale_claims(conn)
        kb.recompute_ready(conn)
        claimed = worker_runtime.claim_ready_candidate(conn, policy=policy, ttl_seconds=args.ttl, claimer=claim_lock())
    if claimed is not None:
        args._active_task_id = claimed.id
        log(f"inject claimed {claimed.id}: {claimed.title}")
        raw_pane = os.environ.get("ZELLIJ_PANE_ID", "")
        if raw_pane:
            pane_id = f"terminal_{raw_pane}" if raw_pane.isdigit() else raw_pane
            # Wait for agent to be idle before injecting
            tui_proc = getattr(args, "_tui_proc", None)
            if tui_proc is not None and tui_proc.pid is not None:
                if not _wait_agent_idle(tui_proc.pid, task_id=claimed.id, board=board):
                    # Agent was busy too long — task already reclaimed
                    args._active_task_id = None
                    return True  # did work (released), don't count as idle
            # Clear input line: "Ctrl u" must be a single argument for zellij send-keys
            subprocess.run(["zellij", "action", "send-keys", "-p", pane_id, "Ctrl u"], timeout=5, check=False)
            time.sleep(0.3)
            # IMPORTANT: inject text must NOT contain \n (LF).
            # In PTY raw mode, LF (0x0A) != Enter (CR, 0x0D).
            # LF only inserts a newline without submitting.
            inject = (
                f"Please work on kanban task {claimed.id} on board {board}. "
                f"Use `hermes kanban --board {board} show {claimed.id}` to read it. "
                f"When done: `hermes kanban --board {board} complete {claimed.id} --summary '...'`"
            )
            subprocess.run(["zellij", "action", "write-chars", "-p", pane_id, inject], timeout=5, check=False)
            time.sleep(0.8)
            subprocess.run(["zellij", "action", "send-keys", "-p", pane_id, "Enter"], timeout=5, check=False)
        return True

    # No task claimed — check if active task completed via DB, TUI stays alive
    active_id = getattr(args, "_active_task_id", None)
    if active_id:
        with kb.connect(board=board) as conn:
            row = conn.execute("SELECT status FROM tasks WHERE id=?", (active_id,)).fetchone()
            if row and row["status"] in ("done", "blocked"):
                log(f"active task {active_id} completed: {row['status']}")
                args._active_task_id = None
    return False



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Codex as a Hermes Kanban listener/profile")
    parser.add_argument("--profile", default=os.environ.get("HERMES_PROFILE") or "planner", help="Kanban assignee/profile to claim (default: planner)")
    parser.add_argument("--claim-assignees", default=os.environ.get("HERMES_KANBAN_CLAIM_ASSIGNEES") or "", help="Comma-separated assignees this worker may claim, primary profile first")
    parser.add_argument("--board", default=os.environ.get("HERMES_KANBAN_BOARD"), help="Board slug; defaults to Hermes current board")
    parser.add_argument("--poll", type=float, default=None, help="Ready-task poll interval override in seconds; default uses shared Hermes listener policy")
    parser.add_argument("--ttl", type=int, default=listener_policy.LISTENER_HEALTH_CLAIM_TTL_SECONDS, help="Claim TTL in seconds; heartbeat extends it")
    parser.add_argument("--assist-claim-delay-s", type=float, default=0.0, help="Delay before this profile may claim secondary assignees")
    parser.add_argument("--assist-claim-delay-for", action="append", default=[], help="Per-assignee assist delay, e.g. implementer=60; repeatable")
    parser.add_argument("--assist-claim-profile-delay", action="append", default=[], help="Profile-qualified assist delay, e.g. critic:implementer=10; repeatable")
    parser.add_argument("--once", action="store_true", help="Process at most one ready task then exit")
    parser.add_argument("--idle-exit-s", type=int, default=0, help="Exit after this many idle seconds (0 = never)")
    parser.add_argument("--sandbox", default=None, help="Optional codex exec sandbox override: read-only/workspace-write/danger-full-access")
    parser.add_argument("--model", default=None, help="Optional codex model override")
    parser.add_argument("--codex-bin", default="codex", help="Codex executable path/name (default: codex)")
    parser.add_argument("--codex-arg", action="append", default=[], help="Extra raw arg passed to codex exec; repeatable")
    parser.add_argument("--task-delivery", default="inject",
                        choices=["inject"],
                        help="inject=keep TUI visible + inject tasks")
    args = parser.parse_args(argv)


    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    board = args.board or kb.get_current_board()
    initial_poll = float(args.poll if args.poll is not None else listener_policy.poll_seconds())
    poll_label = f"{initial_poll:g}s" + (" override" if args.poll is not None else " shared-policy")
    log(f"codex-kanban listener started: profile={args.profile} claims={','.join(claim_assignees(args))} board={board} delivery={args.task_delivery} poll={poll_label}")
    idle_since = time.time()
    try:
        while not _STOP:
            did_work = handle_one_task(args)
            if did_work:
                idle_since = time.time()
                if args.once:
                    return 0
            else:
                if args.once:
                    log("no ready task; exiting --once")
                    return 0
                if args.idle_exit_s and time.time() - idle_since >= args.idle_exit_s:
                    log(f"idle for {args.idle_exit_s}s; exiting")
                    return 0
                time.sleep(float(args.poll if args.poll is not None else listener_policy.poll_seconds()))
    except ListenerStopped:
        log("listener stopped")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
