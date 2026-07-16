"""
``/listen-kanban`` — worker-loop that runs inside an interactive Hermes CLI pane.

Usage::

    /listen-kanban                    # profile = current profile, board = current board
    /listen-kanban planner            # profile = planner, board derived from profile config
    /listen-kanban --board egomotion4d --assignee planner

Once started, the pane continuously finds ``ready`` tasks for the assignee,
claims them atomically, and foreground-executes them via natural language:

    1. ``kanban_show()`` → inspect the task body + comments + handoffs
    2. LLM works the task with full tool access (terminal, file, web, …)
    3. ``kanban_complete()`` or ``kanban_block()``

Ctrl+C behaviour::

    first Ctrl+C   to interrupt current agent run + reclaim the task to return to idle listener
    second Ctrl+C  to exit listener, return to regular chat prompt

Subcommands::

    /listen-kanban pause      pause polling (current task finishes if running)
    /listen-kanban resume     resume polling
    /listen-kanban stop       stop the listener, go back to normal chat
    /listen-kanban status     show listener state
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Listener state — held on the HermesCLI instance
# ---------------------------------------------------------------------------

@dataclass
class ListenerState:
    """Mutable state for a running kanban listener loop."""

    # Config (set once at /listen-kanban)
    board: str = ""
    assignee: str = ""
    claim_assignees: list[str] = field(default_factory=list)  # all assignees in priority order
    assist_claim_delays: dict[str, float] = field(default_factory=dict)  # per-assignee assist delay
    poll_seconds: float = 15.0

    # Run state
    running: bool = True        # False = exit the listener loop
    paused: bool = False       # True = skip polling until resumed
    current_task_id: str = ""  # non-empty when a task is claimed
    current_run_id: Optional[int] = None
    current_generation: Optional[int] = None

    # Coordination
    interrupt_requested: bool = False   # set on Ctrl+C while agent is busy
    listener_thread: Optional[threading.Thread] = None
    listener_stop: threading.Event = field(default_factory=threading.Event)
    _dispatch_time: float = 0.0         # monotonic time when task was dispatched
    _last_watcher_check: float = 0.0    # monotonic time of last watcher health check

    def cleanup(self) -> None:
        self.running = False
        self.current_task_id = ""
        self.current_run_id = None
        self.current_generation = None
        self.interrupt_requested = False


# ---------------------------------------------------------------------------
# Core listener logic
# ---------------------------------------------------------------------------

def _resolve_board(cli_ref: Any, board_arg: str) -> str:
    """Resolve board slug from argument, profile config, or session default."""
    if board_arg:
        return board_arg
    # Check profile config for kanban.current_board
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        kc = cfg.get("kanban", {}) or {}
        board = kc.get("current_board", "")
        if board:
            return board
    except Exception:
        pass
    # Fall back to env var
    env_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    if env_board:
        return env_board
    # Fall back to profile name
    return board_arg or os.environ.get("HERMES_PROFILE", "default")


def _resolve_assignee(cli_ref: Any, assignee_arg: str) -> str:
    """Resolve assignee from argument or current profile."""
    if assignee_arg:
        return assignee_arg
    profile = os.environ.get("HERMES_PROFILE", "")
    if profile:
        return profile
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


def _resolve_claim_assignees(primary_assignee: str) -> list[str]:
    """Resolve claim_assignees from HERMES_KANBAN_CLAIM_ASSIGNEES env var.

    Returns [primary] if env is unset. Otherwise comma-separated list
    with primary_assignee always first.
    """
    raw = os.environ.get("HERMES_KANBAN_CLAIM_ASSIGNEES", "")
    if not raw:
        return [primary_assignee]
    seen: set[str] = {primary_assignee}
    out = [primary_assignee]
    for item in raw.split(","):
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _resolve_assist_claim_delays() -> dict[str, float]:
    """Parse HERMES_KANBAN_ASSIST_CLAIM_DELAYS env var into per-assignee delays.

    Format: ``assignee1=delay1,assignee2=delay2``  (e.g. ``implementer=60``).
    """
    out: dict[str, float] = {}
    raw = os.environ.get("HERMES_KANBAN_ASSIST_CLAIM_DELAYS", "")
    for spec in raw.split(","):
        spec = spec.strip()
        if "=" not in spec:
            continue
        k, v = spec.rsplit("=", 1)
        k = k.strip()
        try:
            delay = float(v.strip())
        except ValueError:
            continue
        if k and delay >= 0:
            out[k] = delay
    return out


def _ready_since(conn, task_id: str, fallback_created_at: int) -> int:
    """Return the most recent timestamp (epoch) when *task_id* became claimable."""
    row = conn.execute(
        "SELECT MAX(created_at) AS ts FROM task_events "
        "WHERE task_id=? AND kind IN ('created','promoted','assigned','reclaimed','unblocked')",
        (task_id,),
    ).fetchone()
    try:
        return int(row["ts"] or fallback_created_at or 0)
    except Exception:
        return int(fallback_created_at or 0)


def _print(msg: str) -> None:
    """Print a listener-status line with a clear prefix."""
    from hermes_cli.colors import Colors as C
    print(f"{C.DIM}[kanban-listener]{C.RESET} {msg}")


def _workspace_path_matches(row_workspace: str | None, workspace: Path | str | None) -> bool:
    """Return True when a task row belongs to the requested workspace."""
    if workspace is None:
        return True
    if not row_workspace:
        return False
    try:
        return (
            Path(row_workspace).expanduser().resolve()
            == Path(workspace).expanduser().resolve()
        )
    except Exception:
        return False


def reset_kanban_running_claims(
    *,
    board: str,
    assignees: list[str],
    workspace: Path,
    claim_lock_suffix: str,
    reason: str,
) -> list[str]:
    """Reclaim running Kanban tasks owned by one visible listener pane.

    This is deliberately narrower than the generic operator reclaim path:
    callers must provide a non-empty claim-lock suffix and a workspace so a
    reset from one pane cannot release another pane's active task.
    """
    from hermes_cli import kanban_db as kb

    suffix = (claim_lock_suffix or "").strip()
    targets = [a for a in dict.fromkeys(assignees or []) if a]
    if not suffix or not targets:
        return []

    kb.init_db(board=board)
    reclaimed: list[str] = []
    placeholders = ",".join("?" for _ in targets)
    with kb.connect(board=board) as conn:
        rows = conn.execute(
            "SELECT id, assignee, workspace_path, claim_lock "
            "FROM tasks "
            "WHERE status = 'running' "
            "AND claim_lock IS NOT NULL "
            f"AND assignee IN ({placeholders}) "
            "ORDER BY started_at ASC, id ASC",
            tuple(targets),
        ).fetchall()
        for row in rows:
            lock = (row["claim_lock"] or "").strip()
            if not lock.endswith(suffix):
                continue
            if not _workspace_path_matches(row["workspace_path"], workspace):
                continue
            task_id = row["id"]
            if kb.reclaim_task(conn, task_id, reason=reason):
                reclaimed.append(task_id)
    return reclaimed


def _find_ready_task(
    board: str,
    claim_assignees: list[str],
    assist_claim_delays: Optional[dict[str, float]] = None,
) -> Optional[dict]:
    """Find the next ready task for any of *claim_assignees* on *board*.

    Respects per-assignee assist delays: an assignee is only considered
    if its ready tasks have been waiting longer than the configured delay.

    Returns a task dict with ``assignee`` included, or None.
    """
    import json as _json
    from hermes_cli import kanban_db as kb

    delays = assist_claim_delays or {}
    old_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            os.environ["HERMES_KANBAN_BOARD"] = board
        kb.init_db()
        with kb.connect() as conn:
            kb.recompute_ready(conn)
            now = time.time()
            for assignee in claim_assignees:
                ready = kb.list_tasks(
                    conn,
                    assignee=assignee,
                    status="ready",
                    limit=1,
                )
                if not ready:
                    continue
                t = ready[0]
                # Check assist delay
                delay = delays.get(assignee, 0.0)
                if delay > 0:
                    created = int(getattr(t, "created_at", 0) or 0)
                    ready_age = now - _ready_since(conn, t.id, created)
                    if ready_age < delay:
                        continue  # not eligible yet
                return {
                    "id": t.id,
                    "title": t.title or "",
                    "body": t.body or "",
                    "assignee": assignee,
                }
            return None
    finally:
        if old_board:
            os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in os.environ:
            del os.environ["HERMES_KANBAN_BOARD"]
def _heal_blocked_tasks(board: str) -> None:
    """Scan for blocked tasks that can be auto-healed and fix them.

    Called by coordinator/planner listener when no ready tasks are found.
    Only heals tasks that were blocked by technical failures (503/crash/timeout).
    Tasks blocked by critic review (code issues) are skipped.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli.colors import Colors as C

    old_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            os.environ["HERMES_KANBAN_BOARD"] = board
        kb.init_db()
        with kb.connect() as conn:
            # Find blocked tasks that were auto-blocked (crash/503/timeout)
            blocked = conn.execute(
                """
                SELECT t.id, t.title, t.assignee, t.max_retries, t.consecutive_failures,
                       r.outcome AS last_outcome, r.error AS last_error, r.ended_at
                FROM tasks t
                LEFT JOIN task_runs r ON r.task_id = t.id AND r.id = (
                    SELECT MAX(id) FROM task_runs WHERE task_id = t.id
                )
                WHERE t.status = 'blocked'
                  AND t.title NOT LIKE '[Review]%'
                  AND t.assignee != 'critic'
                ORDER BY t.created_at ASC
                LIMIT 3
                """,
            ).fetchall()

            if not blocked:
                # Also check for [Review] tasks that were auto-completed without review
                fake_reviews = conn.execute(
                    """
                    SELECT t.id, t.title, t.assignee
                    FROM tasks t
                    WHERE t.status = 'done'
                      AND (t.title LIKE '[Review]%' OR t.assignee = 'critic')
                      AND (t.result LIKE '%auto-completed%' OR t.result IS NULL)
                      AND t.completed_at > ?
                    ORDER BY t.completed_at DESC
                    LIMIT 5
                    """,
                    (int(_time.time()) - 3600,),  # last hour
                ).fetchall()

                if fake_reviews:
                    import time as _time
                    for rt in fake_reviews:
                        rid = rt["id"]
                        conn.execute(
                            "UPDATE tasks SET status='ready', claim_lock=NULL, "
                            "claim_expires=NULL, worker_pid=NULL, result='Reopened: was auto-completed without actual review' "
                            "WHERE id=? AND status='done'",
                            (rid,),
                        )
                        kb._append_event(conn, rid, "unblocked", {
                            "reason": "auto-healed by coordinator: review task was auto-completed without actual review",
                            "healer": "coordinator",
                        })
                        print(f"{C.GREEN}[kanban-healer] {rid}: reopened auto-completed review → ready{C.RESET}")

                return

            import time as _time
            now = _time.time()

            for task in blocked:
                tid = task["id"]
                last_outcome = task["last_outcome"]
                last_error = (task["last_error"] or "").lower()
                last_ended = task["ended_at"]
                consecutive = task["consecutive_failures"] or 0
                max_retries = task["max_retries"] if task["max_retries"] is not None else 10

                # Determine block reason
                is_auto = "auto-blocked" in last_error or "auto-retry" in last_error
                is_retry_exhausted = "retry exhausted" in last_error
                is_crash = last_outcome in ("crashed", "timed_out", "failed")

                if is_auto or is_crash:
                    # Technical failure (503/timeout/crash) — reopen for retry
                    if is_retry_exhausted:
                        # Retries exhausted — no point retrying, but mark for user attention
                        print(f"{C.YELLOW}[kanban-healer] {tid}: retry exhausted ({consecutive}/{max_retries+1}), marking as needs-human{C.RESET}")
                        continue

                    # Check cooldown: only heal if 10+ minutes since last attempt
                    if last_ended and (now - last_ended) < 600:
                        continue

                    conn.execute(
                        "UPDATE tasks SET status='ready', claim_lock=NULL, "
                        "claim_expires=NULL, worker_pid=NULL WHERE id=? AND status='blocked'",
                        (tid,),
                    )
                    kb._append_event(conn, tid, "unblocked", {
                        "reason": f"auto-healed by coordinator: previous block was {last_outcome} ({last_error[:80]})",
                        "healer": "coordinator",
                    })
                    print(f"{C.GREEN}[kanban-healer] {tid}: auto-healed ({last_outcome}) → ready{C.RESET}")

                else:
                    # Not auto-blocked — human-caused or critic block, skip
                    print(f"{C.DIM}[kanban-healer] {tid}: skipped (not auto-blocked: {last_error[:60]}){C.RESET}")

    except Exception as exc:
        logger.warning("heal_blocked_tasks failed: %s", exc)
    finally:
        if old_board:
            os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in os.environ:
            del os.environ["HERMES_KANBAN_BOARD"]


def _claim_and_run(
    cli_ref: Any,
    state: ListenerState,
    task_info: dict,
) -> bool:
    """Claim *task_info* and foreground-execute it in this CLI session.

    Returns True if the task was completed or blocked cleanly.
    Returns False if reclaim/interrupt happened mid-run.
    """
    task_id = task_info["id"]
    board = state.board
    primary_assignee = state.assignee
    task_assignee = task_info.get("assignee", primary_assignee)

    from hermes_cli import kanban_db as kb

    # Set board env for DB access
    old_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            os.environ["HERMES_KANBAN_BOARD"] = board

        with kb.connect() as conn:
            claimed = kb.claim_task(conn, task_id, ttl_seconds=3600)
            if claimed is None:
                _print(f"task {task_id} already claimed by another worker")
                return False

        state.current_task_id = task_id
        state.current_run_id = claimed.current_run_id
        state.current_generation = claimed.generation
        state.interrupt_requested = False

        _print(f"claimed task {task_id}: {task_info.get('title', '')}" +
               (f"  [assist: {task_assignee}]" if task_assignee != primary_assignee else ""))
        _print(f"running in this pane — Ctrl+C to interrupt")

        # Activate the module-level flag so kanban tools appear in agent schema.
        # This is a process-wide flag; we set it before dispatch and clear it
        # when the task completes.
        try:
            from tools.kanban_tools import set_listener_active
            set_listener_active(True)
        except Exception:
            pass

        # Force agent rebuild so kanban tools (gated by _check_kanban_mode)
        # appear in this session's tool schema.  The agent is re-created
        # lazily when set to None.
        cli_ref.agent = None

        # Ensure the kanban-worker skill is loaded into the new agent's
        # system prompt.  The skill provides KANBAN_GUIDANCE lifecycle
        # instructions (claim -> work -> complete/block).  Without this,
        # the rebuilt agent won't know to call kanban_complete().
        try:
            from agent.skill_commands import build_preloaded_skills_prompt
            _prompt, _loaded, _missing = build_preloaded_skills_prompt(
                ["kanban-worker"],
                task_id=getattr(cli_ref, "session_id", ""),
            )
            if _prompt:
                cli_ref.system_prompt = "\n\n".join(
                    p for p in (cli_ref.system_prompt or "", _prompt) if p
                ).strip()
                # Merge with existing preloaded_skills, avoiding duplicates
                existing = set(getattr(cli_ref, "preloaded_skills", []) or [])
                for sk in _loaded:
                    existing.add(sk)
                cli_ref.preloaded_skills = list(existing)
        except Exception as exc:
            logger.warning("Failed to preload kanban-worker skill: %s", exc)

        # Set HERMES_KANBAN_TASK so kanban tool handlers have a default task_id
        os.environ["HERMES_KANBAN_TASK"] = task_id
        os.environ["HERMES_KANBAN_CLAIM_LOCK"] = claimed.claim_lock or ""
        if claimed.current_run_id is not None:
            os.environ["HERMES_KANBAN_RUN_ID"] = str(claimed.current_run_id)
        os.environ["HERMES_KANBAN_GENERATION"] = str(claimed.generation)

        # Keep HERMES_KANBAN_TASK set for the duration of this task.
        # We don't restore it here because process_loop picks up the env
        # when the agent processes the _pending_input. Instead, clear it
        # in the listener loop when the task completes or is interrupted.
        prompt = f"work kanban task {task_id}"
        state._dispatch_time = time.monotonic()
        cli_ref._pending_input.put(prompt)
        _print(f"dispatched kanban task {task_id} to agent")
        return True

    finally:
        if old_board:
            os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in os.environ:
            del os.environ["HERMES_KANBAN_BOARD"]


def _cleanup_after_task(state: ListenerState) -> None:
    """Clean up env vars after a task completes or is interrupted."""
    state.current_task_id = ""
    state.current_run_id = None
    state.current_generation = None

    # Clear the listener flag so kanban tools go away
    try:
        from tools.kanban_tools import set_listener_active
        set_listener_active(False)
    except Exception:
        pass

    # Clear env vars
    for key in (
        "HERMES_KANBAN_TASK",
        "HERMES_KANBAN_RUN_ID",
        "HERMES_KANBAN_GENERATION",
        "HERMES_KANBAN_CLAIM_LOCK",
    ):
        if key in os.environ:
            del os.environ[key]


def _retry_exhausted_block(conn, tid, consecutive, max_retries, last_run, C):
    """Block task permanently after exhausting retries."""
    conn.execute(
        "UPDATE tasks SET status='blocked', claim_lock=NULL, "
        "claim_expires=NULL, worker_pid=NULL "
        "WHERE id=? AND status='running'",
        (tid,),
    )
    from hermes_cli import kanban_db as kb
    kb._append_event(conn, tid, "blocked", {
        "reason": f"retry exhausted ({consecutive}/{max_retries+1}): last run {last_run['outcome']} ({last_run['error'] or 'unknown'})",
        "auto": True,
    })
    print(f"{C.DIM}[kanban-listener] retry exhausted for {tid} ({consecutive}/{max_retries+1}) — blocked{C.RESET}")


def _auto_complete_if_still_running(state: ListenerState) -> None:
    """If the task is still running after agent goes idle, handle appropriately.

    Decision logic (in order):

    1. **Crash/timeout/failure** → reclaim + auto-retry (up to max_retries).
    2. **Run too short (<60s)** → agent barely started before 503/error killed it.
       Reclaim + auto-retry instead of auto-completing with empty result.
    3. **Run has real summary** → agent genuinely forgot ``kanban_complete()``.
       Auto-complete as safety net, preserving the agent's summary.
    4. **Run has no/empty summary** → agent exited without producing output.
       Reclaim + auto-retry instead of auto-completing with placeholder text.

    Respects retry_cooldown: if less than 10 minutes since last run, skip retry.
    After exhausting max_retries, block permanently.
    """
    if not state.current_task_id:
        return
    tid = state.current_task_id
    board = state.board

    from hermes_cli import kanban_db as kb
    from hermes_cli.colors import Colors as C
    import time as _time

    # Threshold: runs shorter than this are considered "agent didn't really work".
    MIN_PRODUCTIVE_SECONDS = 60

    old_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            os.environ["HERMES_KANBAN_BOARD"] = board
        with kb.connect() as conn:
            task = kb.get_task(conn, tid)
            if task and task.status == "running":
                if (
                    state.current_run_id is None
                    or state.current_generation is None
                    or task.current_run_id != state.current_run_id
                    or task.generation != state.current_generation
                ):
                    print(
                        f"{C.DIM}[kanban-listener] ignored stale callback for "
                        f"{tid}: expected run={state.current_run_id} "
                        f"generation={state.current_generation}, current "
                        f"run={task.current_run_id} generation={task.generation}"
                        f"{C.RESET}"
                    )
                    return
                # Get last run with full detail
                last_run = conn.execute(
                    "SELECT id, generation, outcome, summary, error, "
                    "started_at, ended_at FROM task_runs "
                    "WHERE task_id = ? AND id = ?",
                    (tid, state.current_run_id),
                ).fetchone()

                # --- Determine if the run was genuinely productive ---
                is_crash = last_run and last_run["outcome"] in ("crashed", "timed_out", "failed")
                is_too_short = False
                has_real_summary = False

                if last_run:
                    elapsed = (last_run["ended_at"] or 0) - (last_run["started_at"] or 0)
                    is_too_short = elapsed < MIN_PRODUCTIVE_SECONDS and elapsed >= 0

                    summary_text = (last_run["summary"] or "").strip()
                    # Template auto-complete messages are NOT real summaries
                    is_template = (
                        "auto-completed" in summary_text.lower()
                        or "did not call kanban_complete" in summary_text.lower()
                        or "watchdog" in summary_text.lower()
                    )
                    has_real_summary = bool(summary_text) and not is_template

                # --- Decision tree ---
                should_reclaim = False
                reclaim_reason = ""

                if is_crash:
                    # Case 1: explicit crash/timeout/failure
                    should_reclaim = True
                    reclaim_reason = f"auto-retry: last run {last_run['outcome']}"
                elif is_too_short:
                    # Case 2: run was too short — agent barely started (likely 503)
                    should_reclaim = True
                    elapsed_val = (last_run["ended_at"] or 0) - (last_run["started_at"] or 0) if last_run else 0
                    reclaim_reason = f"auto-retry: run too short ({elapsed_val}s < {MIN_PRODUCTIVE_SECONDS}s), likely API error"
                elif has_real_summary:
                    # Case 3: agent produced real output but forgot kanban_complete()
                    # → safe to auto-complete, preserve the agent's summary
                    completed = kb.complete_task(
                        conn, tid,
                        result="auto-completed by listener safety net",
                        summary=last_run["summary"],
                        expected_run_id=state.current_run_id,
                        expected_generation=state.current_generation,
                    )
                    if completed:
                        print(f"{C.DIM}[kanban-listener] auto-completed task {tid} (agent had real output){C.RESET}")
                    else:
                        print(
                            f"{C.DIM}[kanban-listener] completion fence "
                            f"rejected stale callback for {tid}{C.RESET}"
                        )
                    return
                else:
                    # Case 4: no crash, not too short, but no real summary
                    # → agent exited without producing meaningful output (503 mid-run, etc.)
                    should_reclaim = True
                    reclaim_reason = "auto-retry: agent exited without producing output (no real summary)"

                if not should_reclaim:
                    # Fallback: shouldn't reach here, but if we do, don't auto-complete
                    # with placeholder text — reclaim instead.
                    should_reclaim = True
                    reclaim_reason = "auto-retry: no productive output detected"

                # --- Reclaim + auto-retry ---
                task_row = conn.execute(
                    "SELECT max_retries, consecutive_failures FROM tasks WHERE id=?",
                    (tid,),
                ).fetchone()
                max_retries = task_row["max_retries"] if task_row and task_row["max_retries"] is not None else 10
                consecutive_failures = (task_row["consecutive_failures"] or 0) + 1 if task_row else 1

                # Check retry cooldown (10 minutes)
                now = _time.time()
                last_ended = last_run.get("ended_at") if last_run else None
                if last_ended and (now - last_ended) < 600:
                    print(f"{C.DIM}[kanban-listener] cooldown for {tid} ({consecutive_failures}/{max_retries}, {(now - last_ended)/60:.0f}min since last retry){C.RESET}")
                    return

                if consecutive_failures > max_retries + 1:
                    # Exhausted - block
                    _retry_exhausted_block(conn, tid, consecutive_failures, max_retries, last_run, C)
                    return

                # Reclaim -> ready for retry
                kb.reclaim_task(conn, tid, reason=reclaim_reason)
                # Increment consecutive_failures after reclaim_task cleared it
                conn.execute(
                    "UPDATE tasks SET consecutive_failures=? WHERE id=?",
                    (consecutive_failures, tid),
                )
                kb._append_event(conn, tid, "reclaimed", {
                    "reason": f"auto-retry #{consecutive_failures}/{max_retries}: {reclaim_reason}",
                    "auto": True,
                })
                print(f"{C.DIM}[kanban-listener] auto-retry #{consecutive_failures}/{max_retries} for {tid}: {reclaim_reason}{C.RESET}")
    except Exception as exc:
        logger.warning("auto-retry handler failed for %s: %s", tid, exc)
    finally:
        if old_board:
            os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in os.environ:
            del os.environ["HERMES_KANBAN_BOARD"]


def _reclaim_current_immediate(cli_ref: Any, listener: ListenerState) -> None:
    """Reclaim immediately from the Ctrl+C handler thread (synchronous)."""
    if not listener.current_task_id:
        return
    tid = listener.current_task_id
    listener.current_task_id = ""
    listener.interrupt_requested = True
    board = listener.board

    from hermes_cli import kanban_db as kb
    from hermes_cli.colors import Colors as C

    old_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            os.environ["HERMES_KANBAN_BOARD"] = board
        with kb.connect() as conn:
            kb.reclaim_task(conn, tid, reason="user interrupted via Ctrl+C in listen-kanban")
        # Use raw print so user sees the reclaim happened
        print(f"\n{C.DIM}[kanban-listener] reclaimed task {tid}{C.RESET}")
    except Exception as exc:
        print(f"\n{C.DIM}[kanban-listener] reclaim failed for {tid}: {exc}{C.RESET}")
    finally:
        if old_board:
            os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in os.environ:
            del os.environ["HERMES_KANBAN_BOARD"]
    _cleanup_after_task(listener)


def _reclaim_current(state: ListenerState) -> None:
    """Reclaim the currently claimed task, if any."""
    if not state.current_task_id:
        return
    tid = state.current_task_id
    state.current_task_id = ""
    board = state.board

    from hermes_cli import kanban_db as kb
    old_board = os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            os.environ["HERMES_KANBAN_BOARD"] = board
        with kb.connect() as conn:
            kb.reclaim_task(conn, tid, reason="user interrupted via Ctrl+C in listen-kanban")
        _print(f"reclaimed task {tid}")
    except Exception as exc:
        _print(f"reclaim failed for {tid}: {exc}")
    finally:
        if old_board:
            os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in os.environ:
            del os.environ["HERMES_KANBAN_BOARD"]
    _cleanup_after_task(state)


def _check_watcher_health(board: str, interval_seconds: float = 180.0) -> None:
    """Check health of kanban watcher subprocesses for all panes.
    
    This is called periodically from the coordinator's /listen-kanban loop.
    It detects when a pane's watcher subprocess has died but the launcher
    is still alive (stale state). Prints a warning so the operator can
    restart the watcher manually.
    
    Args:
        board: The kanban board slug (e.g., "egomotion4d")
        interval_seconds: How often to run the check (default 180s)
    """
    import os as _os
    import re as _re
    import time as _time
    from hermes_cli.colors import Colors as C
    
    # Find all launcher and watcher processes for this board
    roles = {"coordinator", "planner", "implementer", "critic"}
    pane_info = {}
    
    def _read_cmd(pid):
        try:
            raw = open(f"/proc/{pid}/cmdline", "rb").read()
            return " ".join(part.decode("utf-8", "replace") for part in raw.split(b"\0") if part)
        except OSError:
            return ""
    
    def _read_env(pid):
        try:
            raw = open(f"/proc/{pid}/environ", "rb").read()
            env = {}
            for item in raw.split(b"\0"):
                if not item or b"=" not in item:
                    continue
                k, v = item.split(b"=", 1)
                env[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
            return env
        except OSError:
            return {}
    
    try:
        for name in _os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            cmd = _read_cmd(pid)
            if not cmd:
                continue
            
            # Launcher processes: --auto-start without --watch-child
            if "kanban_interactive" in cmd and "--auto-start" in cmd and "--watch-child" not in cmd:
                env = _read_env(pid)
                if env.get("HERMES_KANBAN_BOARD") != board and f"--board {board}" not in cmd:
                    continue
                m = _re.search(r"--profile\s+(\w+)", cmd)
                profile = m.group(1) if m else None
                if not profile or profile not in roles:
                    continue
                if profile not in pane_info:
                    pane_info[profile] = {"launcher_pid": None, "watcher_pid": None, "agent": ""}
                pane_info[profile]["launcher_pid"] = pid
                if "reasonix_kanban_interactive" in cmd:
                    pane_info[profile]["agent"] = "reasonix"
                elif "codex_kanban_interactive" in cmd:
                    pane_info[profile]["agent"] = "codex"
                elif "deepseek_kanban_interactive" in cmd:
                    pane_info[profile]["agent"] = "deepseek"
            
            # Watcher processes: --watch-child
            if "kanban_interactive" in cmd and "--watch-child" in cmd:
                env = _read_env(pid)
                if env.get("HERMES_KANBAN_BOARD") != board and f"--board {board}" not in cmd:
                    continue
                m = _re.search(r"--profile\s+(\w+)", cmd)
                profile = m.group(1) if m else None
                if not profile or profile not in roles:
                    continue
                if profile not in pane_info:
                    pane_info[profile] = {"launcher_pid": None, "watcher_pid": None, "agent": ""}
                pane_info[profile]["watcher_pid"] = pid
        
        # coordinator (hermes) doesn't have launcher/watcher subprocess
        for name in _os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            cmd = _read_cmd(pid)
            if not cmd:
                continue
            if "hermes" in cmd and "-p" in cmd and "coordinator" in cmd and "kanban_interactive" not in cmd:
                env = _read_env(pid)
                if env.get("HERMES_KANBAN_BOARD") == board:
                    if "coordinator" not in pane_info:
                        pane_info["coordinator"] = {"launcher_pid": pid, "watcher_pid": pid, "agent": "hermes"}
                    break
        
        # Find anomalies: launcher alive but watcher dead
        anomalies = {p: i for p, i in pane_info.items() if i["launcher_pid"] and not i["watcher_pid"]}
        
        if anomalies:
            _print(f"{C.RED}WATCHER HEALTH CHECK: {len(anomalies)} pane(s) missing watcher subprocess!{C.RESET}")
            for profile, info in sorted(anomalies.items()):
                _print(f"  {C.YELLOW}{profile}{C.RESET}: launcher PID {info['launcher_pid']} alive but watcher is DEAD")
            _print(f"{C.DIM}To restart: zellij kill-session --force kanban-{board} && bash scripts/start-kanban.sh{C.RESET}")
        else:
            # Only log on success if there are panes to check
            if pane_info:
                _print(f"watcher health check OK ({len(pane_info)} panes)")
    
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("watcher health check failed: %s", exc)


def _reclaim_orphan_running_tasks(board: str, assignee: str) -> None:
    """Reclaim running tasks whose worker process is no longer alive.

    Only handles tasks assigned to this listener's profile. Checks the
    task's ``claim_lock`` for a host-local PID and reclaims if the PID
    is dead. This catches workers spawned by dispatch that died silently
    (503/crash/oom) without updating task status.
    """
    import os as _os
    from hermes_cli import kanban_db as kb
    from hermes_cli.colors import Colors as C
    try:
        import psutil
    except ImportError:
        psutil = None  # fallback to /proc check

    old_board = _os.environ.get("HERMES_KANBAN_BOARD", "")
    try:
        if board:
            _os.environ["HERMES_KANBAN_BOARD"] = board
        kb.init_db()
        with kb.connect() as conn:
            running = conn.execute(
                "SELECT id, claim_lock, consecutive_failures, max_retries "
                "FROM tasks WHERE status='running' AND assignee=?",
                (assignee,),
            ).fetchall()
            for task in running:
                tid = task["id"]
                lock = (task["claim_lock"] or "").strip()
                if not lock or ":" not in lock:
                    continue
                # Extract PID from lock string: "host:pid"
                try:
                    pid_str = lock.rsplit(":", 1)[-1]
                    pid = int(pid_str)
                except (ValueError, IndexError):
                    continue

                # Check if PID is alive
                pid_alive = False
                if psutil is not None:
                    try:
                        pid_alive = psutil.pid_exists(pid) and psutil.Process(pid).is_running()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pid_alive = False
                else:
                    # Fallback: /proc exists on Linux
                    try:
                        pid_alive = _os.path.isdir(f"/proc/{pid}")
                    except Exception:
                        pid_alive = False

                if not pid_alive:
                    # Worker is dead — reclaim and log
                    kb.reclaim_task(conn, tid, reason=f"orphan recovery: worker PID {pid} is dead")
                    # Increment consecutive_failures so retry tracking works
                    max_retries = task["max_retries"] if task["max_retries"] is not None else 10
                    consecutive = (task["consecutive_failures"] or 0) + 1
                    conn.execute(
                        "UPDATE tasks SET consecutive_failures=? WHERE id=?",
                        (consecutive, tid),
                    )
                    kb._append_event(conn, tid, "reclaimed", {
                        "reason": f"orphan worker PID {pid} dead; reclaimed by {assignee} listener",
                        "auto": True,
                    })
                    print(f"{C.YELLOW}[kanban-listener] orphan recovery reclaimed {tid} (dead PID {pid}){C.RESET}")
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("orphan recovery failed: %s", exc)
    finally:
        if old_board:
            _os.environ["HERMES_KANBAN_BOARD"] = old_board
        elif not old_board and "HERMES_KANBAN_BOARD" in _os.environ:
            del _os.environ["HERMES_KANBAN_BOARD"]


def _listener_loop(cli_ref: Any, state: ListenerState) -> None:
    """Background loop: poll for ready tasks and dispatch them.

    Runs in a daemon thread. It only polls when the agent is idle
    (not running a task). The actual task execution happens in the
    foreground via _pending_input to process_loop.
    """
    roles = ",".join(state.claim_assignees) if state.claim_assignees else state.assignee
    _print(f"listener started — board={state.board}, claims={roles}")
    _print("watching for ready tasks (poll every {:.0f}s)".format(state.poll_seconds))
    _print("Ctrl+C to interrupt current task; Ctrl+C again to stop listener")

    while state.running and not state.listener_stop.is_set():
        try:
            if state.paused:
                time.sleep(1.0)
                continue

            if state.current_task_id:
                # A task is active. Check if the agent has finished.
                agent_busy = getattr(cli_ref, "_agent_running", False)
                if not agent_busy and not state.interrupt_requested:
                    # Agent is idle. Check how long ago we dispatched -
                    # if it's very recent (<5s), the process_loop hasn't
                    # started yet. Wait longer.
                    if state._dispatch_time and (time.monotonic() - state._dispatch_time) < 5.0:
                        time.sleep(1.0)
                        continue
                    # Agent is idle and task was active — it probably completed.
                    _print(f"task {state.current_task_id} finished (agent idle)")
                    # Safety net: if the agent didn't call kanban_complete(),
                    # auto-complete with a summary so the task doesn't get stuck.
                    _auto_complete_if_still_running(state)
                    _cleanup_after_task(state)
                    state._dispatch_time = 0.0
                    time.sleep(2.0)
                elif state.interrupt_requested:
                    # Ctrl+C was pressed while agent was running.
                    # Reclaim was handled by the interrupt handler.
                    _cleanup_after_task(state)
                    state.interrupt_requested = False
                    _print("ready for next task")
                    time.sleep(2.0)
                else:
                    # Still running
                    time.sleep(1.0)
                continue

            # Check if agent is busy (shouldn't happen if no current_task,
            # but guard against races)
            if getattr(cli_ref, "_agent_running", False):
                time.sleep(1.0)
                continue

            # Orphan task recovery: reclaim running tasks whose worker
            # processes are dead (backed by this listener's profile).
            # This catches workers spawned by dispatch that died without
            # updating task status (503/crash/oom).
            for a in state.claim_assignees:
                _reclaim_orphan_running_tasks(state.board, a)

            # Poll for ready tasks
            task_info = _find_ready_task(state.board, state.claim_assignees, state.assist_claim_delays)
            if task_info is None:
                # No ready tasks — try to heal blocked tasks we can fix
                if state.assignee in ("coordinator", "planner"):
                    _heal_blocked_tasks(state.board)
                # Periodic watcher health check (every 180s for coordinator)
                if state.assignee == "coordinator":
                    now = time.monotonic()
                    if now - state._last_watcher_check >= 180.0:
                        _check_watcher_health(state.board)
                        state._last_watcher_check = now
                time.sleep(state.poll_seconds)
                continue

            # Found a task — claim and run it
            _claim_and_run(cli_ref, state, task_info)

            # Give the agent time to start processing before we poll again
            time.sleep(2.0)

        except Exception as exc:
            logger.exception("kanban listener error: %s", exc)
            time.sleep(state.poll_seconds)

    _print("listener stopped")


# ---------------------------------------------------------------------------
# Public API — called from cli.py process_command()
# ---------------------------------------------------------------------------

def start_listener(
    cli_ref: Any,
    *,
    board_arg: str = "",
    assignee_arg: str = "",
    poll_seconds: float = 15.0,
) -> None:
    """Start the kanban listener loop on a background thread.

    Stores the state on ``cli_ref._kanban_listener``.
    """
    # Stops any existing listener
    stop_listener(cli_ref)

    board = _resolve_board(cli_ref, board_arg)
    assignee = _resolve_assignee(cli_ref, assignee_arg)

    state = ListenerState(
        board=board,
        assignee=assignee,
        claim_assignees=_resolve_claim_assignees(assignee),
        assist_claim_delays=_resolve_assist_claim_delays(),
        poll_seconds=poll_seconds,
        listener_stop=threading.Event(),
    )
    cli_ref._kanban_listener = state

    t = threading.Thread(
        target=_listener_loop,
        args=(cli_ref, state),
        daemon=True,
    )
    state.listener_thread = t
    t.start()


def stop_listener(cli_ref: Any) -> None:
    """Stop the kanban listener if running."""
    state = getattr(cli_ref, "_kanban_listener", None)
    if state is None:
        return

    # Reclaim any active task
    if state.current_task_id:
        _reclaim_current(state)

    state.listener_stop.set()
    state.running = False

    # Wait briefly for thread to exit
    if state.listener_thread and state.listener_thread.is_alive():
        state.listener_thread.join(timeout=5.0)

    cli_ref._kanban_listener = None
    _print("listener stopped")


def pause_listener(cli_ref: Any) -> bool:
    """Pause polling. If a task is active, it finishes first."""
    state = getattr(cli_ref, "_kanban_listener", None)
    if state is None:
        _print("no active listener")
        return False
    if state.paused:
        _print("already paused")
        return False
    state.paused = True
    _print("listener paused (current task will finish if running)")
    return True


def resume_listener(cli_ref: Any) -> bool:
    state = getattr(cli_ref, "_kanban_listener", None)
    if state is None:
        _print("no active listener")
        return False
    if not state.paused:
        _print("already running")
        return False
    state.paused = False
    _print("listener resumed")
    return True


def listener_status(cli_ref: Any) -> dict:
    """Return current listener state as a dict."""
    state = getattr(cli_ref, "_kanban_listener", None)
    if state is None:
        return {"active": False}
    return {
        "active": True,
        "board": state.board,
        "assignee": state.assignee,
        "claim_assignees": state.claim_assignees,
        "assist_claim_delays": state.assist_claim_delays,
        "paused": state.paused,
        "current_task_id": state.current_task_id,
    }
