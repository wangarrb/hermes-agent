# Idle-Pane Reclaim Loop Fix (2026-06-06)

## Incident

Implementer reasonix pane's watcher entered a claim-reclaim loop, reclaiming the same task every ~8 seconds across dozens of watcher restarts.

## Timeline (from watcher log)

```
20:19:04  watcher started
20:19:14  reclaimed orphaned idle-pane task t_b8e4411f old_pid=None run=338
          → watcher crash (OperationalError: disk I/O error in add_comment)
20:19:16  launcher restarted watcher (restart_count=1)
20:19:26  reclaimed orphaned idle-pane task t_b8e4411f old_pid=None run=338
          → watcher crash again
20:19:28  restart_count=2
20:19:40  reclaimed orphaned idle-pane task t_b8e4411f old_pid=1163332 run=339
          → crash
... (continues for 10+ iterations)
22:50:13  reclaimed orphaned idle-pane task t_e573a455 old_pid=1303757 run=349
22:52:20  reclaimed orphaned idle-pane task t_e573a455 old_pid=1304158 run=350
22:52:37  reclaimed orphaned idle-pane task t_e573a455 old_pid=1304158 run=350
... (still continuing)
```

Each iteration: watcher starts → waits 8s (startup_delay) → adopt_orphaned → pane idle → reclaim → claim+inject → DB error → crash → restart.

## Root Cause Analysis

### Layer 1: Idle detection too eager

`adopt_orphaned_running_task()` at line 748:
```python
if _pane_looks_idle(args, log_path=log_path):
    # Immediately reclaim — no grace period
    _reclaim_task_without_signaling_worker(conn, task_id, reason="...idle; prompt was not active after restart")
    return None, None
```

After watcher restarts, the pane may show idle because:
- Prompt was just injected but TUI hasn't rendered the busy state yet
- Reasonix is still initializing / loading the prompt
- The AI hasn't started processing (thinking delay)

The watcher trusts the idle detection immediately with zero grace period.

### Layer 2: DB I/O error crashes watcher after successful injection

After reclaim → claim+inject succeeds (prompt is in the pane), the watcher tries:
```python
with kb.connect(board=board) as conn:
    kb.add_comment(conn, claimed.id, ...)    # ← OperationalError: disk I/O error
    kb.heartbeat_worker(conn, claimed.id, ...)
```

This crashes the watcher. The launcher restarts it. It goes back to `adopt_orphaned_running_task` → pane still idle (Reasonix hasn't started yet) → reclaim again.

### Layer 3: Feedback loop

The two layers create a self-reinforcing loop:
- Idle detection triggers reclaim → prompt re-injected → DB error crashes watcher → watcher restarts → idle detection triggers reclaim again

## Fixes Applied

### Fix 1: Grace period in adopt_orphaned_running_task

**Files**: `reasonix_kanban_interactive.py` (line ~748), `deepseek_kanban_interactive.py` (line ~700)

When pane is idle, check `task_runs.started_at` before reclaiming:
- If run started < 30 seconds ago → **adopt** instead of reclaim (fall through to adopt path)
- If run started ≥ 30 seconds ago → reclaim as before (genuinely idle)

```python
run_started_at = None
if run_id is not None:
    run_row = conn.execute(
        "SELECT started_at FROM task_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run_row:
        run_started_at = run_row["started_at"]
grace_period_s = 30
if run_started_at and (now - int(run_started_at)) < grace_period_s:
    log_line(log_path, f"orphaned task {task_id} run={run_id} pane idle but run started "
             f"{now - int(run_started_at)}s ago (< {grace_period_s}s); adopting instead of reclaiming")
    # Fall through to adopt path
else:
    # Original reclaim logic
```

### Fix 2: DB error protection in claim_and_inject_one

**Files**: `reasonix_kanban_interactive.py`, `deepseek_kanban_interactive.py`

Four `kb.connect()` call sites in `claim_and_inject_one()` wrapped with try/except:

1. **Claim-phase** (release_stale_claims → claim_task → build_worker_context):
   ```python
   try:
       with kb.connect(board=board) as conn:
           # ... claim logic ...
   except Exception as exc:
       log_line(log_path, f"claim DB error (non-fatal): {type(exc).__name__}: {exc}")
       return None, None
   ```

2. **Pre-inject reclaim** (pane became unsafe before injection):
   ```python
   try:
       with kb.connect(board=board) as conn:
           _reclaim_task_without_signaling_worker(conn, claimed.id, ...)
   except Exception as exc:
       log_line(log_path, f"pre-inject reclaim DB error (non-fatal): ...")
   ```

3. **Inject-fail reclaim** (zellij injection failed):
   ```python
   try:
       with kb.connect(board=board) as conn:
           _reclaim_task_without_signaling_worker(conn, claimed.id, ...)
   except Exception as exc:
       log_line(log_path, f"inject-fail reclaim DB error (non-fatal): ...")
   ```

4. **Post-inject** (add_comment + heartbeat_worker — **most critical**):
   ```python
   try:
       with kb.connect(board=board) as conn:
           kb.add_comment(conn, claimed.id, ...)
           kb.heartbeat_worker(conn, claimed.id, ...)
   except Exception as exc:
       log_line(log_path, f"post-inject DB op failed (non-fatal): {type(exc).__name__}: {exc}")
   ```

   This was the crash point. The prompt was already successfully injected into the pane — `add_comment` and `heartbeat_worker` are record-keeping. Failing them should not crash the watcher.

## Verification

Both files compile cleanly:
```
plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py: OK
plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py: OK
```

No kanban processes were running at the time of the fix, so the new code will take effect on next `start-kanban.sh` launch.

## Design Rationale

**Why 30-second grace period?**
- Reasonix TUI typically takes 2-5 seconds to start rendering after a prompt injection
- AI thinking time before showing "processing" can be 5-15 seconds
- 30 seconds gives comfortable margin without being so long that genuinely-idle tasks stick around
- The main-loop idle_pane_reclaim_s is already 600s, so the grace period only affects the startup-orphan case

**Why adopt instead of just waiting?**
- If we just skip the reclaim and return None, the watcher will try `claim_and_inject_one` next, which would create a SECOND claim on an already-running task
- Adopting is correct: the task IS running in the pane, the watcher just needs to resume monitoring it
- The adopt path updates claim_lock/worker_pid to the new watcher process

**Why not apply to codex listener?**
- Codex listener doesn't use the same `_pane_looks_idle` + `adopt_orphaned_running_task` pattern
- It uses a different architecture (codex_kanban_listener.py with subprocess management)
