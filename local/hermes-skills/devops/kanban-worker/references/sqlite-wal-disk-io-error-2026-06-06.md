# SQLite WAL disk I/O Error — Full Diagnostic Trace (2026-06-06)

## Context

Hermes Kanban reasonix-interactive-implementer listener showed 72 `sqlite3.OperationalError: disk I/O error` errors and 6 `.corrupt.*.bak` files were generated between 16:13-20:20 on June 6, 2026.

## Observations

1. **Error always on cleanup active claim**: `cleanup active claim failed for t_a3dad649: OperationalError: disk I/O error` — every error appeared in `_cleanup_active_claim()` or `_task_status()` (SELECT query, not just writes).

2. **Only reasonix listener affected**: deepseek/codex listeners had zero disk I/O errors. The reasonix watcher had the most aggressive crash-loop (idle-pane reclaim every 8 seconds).

3. **6 corrupt backups**: `kanban.db.corrupt.{hash}.bak` files at:
   - 16:13-16:17 (4 files, created during gateway restarts detecting WAL inconsistency)
   - 20:20 (2 files)

4. **DB integrity OK**: `PRAGMA integrity_check` returned `ok`. Filesystem is local ext4 (`/dev/sda1 on /home type ext4`).

5. **WAL checkpoint clean**: `PRAGMA wal_checkpoint(PASSIVE)` returned `(0, 0, 0)` — no pending WAL pages.

6. **busy_timeout mismatch**: Raw `sqlite3.connect(db)` gives 5000ms (Python default); `kanban_db.connect(board=...)` gives 120000ms (Hermes default with `DEFAULT_BUSY_TIMEOUT_MS = 120_000`). Test with the right path.

7. **Crash-loop pattern**: Every 8 seconds (matching `--startup-delay-s 8.0`):
   - Watcher starts → adopt_orphaned_running_task → pane idle → reclaim
   - claim_and_inject_one → prompt injected → add_comment/heartbeat hits DB error → watcher crashes
   - Launcher restarts watcher → cycle repeats
   - Observed: 14 cycles in 43 seconds (run 338-351)

8. **kb.connect() context manager does NOT close connections**: `with kb.connect() as conn:` only calls `__enter__` (which does PRAGMA setup) and `__exit__` (which commits or rolls back). It NEVER calls `conn.close()`. This means every `with kb.connect()` call leaks a file descriptor and holds a WAL reader lock until the Python GC eventually collects the connection object. With 8 `with kb.connect()` call sites in reasonix_kanban_interactive.py alone, a single claim-and-inject cycle leaks 4+ FDs.

## Root Cause Analysis

**Primary cause: SQLite WAL-mode multi-process concurrent write contention.**

NOT disk hardware failure. Evidence:
- integrity_check = ok
- ext4 local filesystem (not NFS)
- 30% disk space available
- only one listener shows errors (the one with crash-loop)

**Mechanism**:
1. 5+ processes simultaneously access kanban.db (gateway dispatch + 3-5 listeners)
2. Each `claim_and_inject_one` call opens 4 separate `kb.connect()` connections (claim, pre-inject-reclaim, inject-fail-reclaim, add_comment/heartbeat)
3. `kb.connect()` context manager never closes connections → FD leak + WAL reader lock accumulation
4. WAL autocheckpoint=100 means frequent checkpoint operations requiring exclusive locks
5. Crash-loop creates resonance: 8s watcher cycle vs 5s gateway cycle = high collision probability
6. When watcher crashes mid-operation, WAL state may be inconsistent → corrupt backups generated on next gateway startup

**Why only reasonix**: Most aggressive crash-loop parameters + idle-pane reclaim amplifying DB contention.

## Diagnostic Commands Used

```bash
# Check corrupt backups
ls -lt ~/.hermes/kanban/boards/<board>/kanban.db.corrupt.*.bak

# Integrity check
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db "PRAGMA integrity_check;"

# WAL checkpoint status
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db "PRAGMA wal_checkpoint(PASSIVE);"

# busy_timeout (must use kanban_db.connect, not raw sqlite3.connect!)
/home/wyr/.hermes/hermes-agent/venv/bin/python3 -c "
import sys; sys.path.insert(0, '/home/wyr/.hermes/hermes-agent')
from hermes_cli.kanban_db import connect
conn = connect(board='egomotion4d')
print('busy_timeout:', conn.execute('PRAGMA busy_timeout').fetchone()[0], 'ms')
print('journal_mode:', conn.execute('PRAGMA journal_mode').fetchone()[0])
print('synchronous:', conn.execute('PRAGMA synchronous').fetchone()[0])
conn.close()
"

# Filesystem type
df -T ~/.hermes/kanban/boards/<board>/

# Error frequency
grep -c "disk I/O error" ~/.hermes/kanban/boards/<board>/logs/reasonix-interactive-implementer.log

# Error context
grep -B1 "disk I/O error" ~/.hermes/kanban/boards/<board>/logs/reasonix-interactive-implementer.log | head -30

# Concurrent processes
fuser ~/.hermes/kanban/boards/<board>/kanban.db

# WAL/SHM sidecar files
ls -la ~/.hermes/kanban/boards/<board>/kanban.db-wal ~/.hermes/kanban/boards/<board>/kanban.db-shm
```

## Fixes Applied

### Session 1 (2026-06-06): Crash-loop prevention

1. **Idle grace period** (see Failure 6 in SKILL.md): 30-second grace in `adopt_orphaned_running_task` before reclaiming idle-pane tasks. Prevents crash-loop trigger.

2. **DB error non-fatal handling**: try/except around all `kb.connect()` calls in `claim_and_inject_one`. Post-inject operations (add_comment, heartbeat) don't crash the watcher if they fail.

### Session 2 (2026-06-07): Connection reduction and reuse (Plan 1 + Plan 2)

**Plan 1 — Merge connections in claim_and_inject_one** (reasonix + deepseek):

- Added optional `conn` parameter to `claim_and_inject_one()`. When caller passes a connection, it's reused; when omitted, a new connection is created and closed via `_owns_conn` + `finally`.
- Post-inject `add_comment` + `heartbeat_worker` merged into the same connection as the claim phase (no separate `with kb.connect()` needed).
- Pre-inject reclaim and inject-fail reclaim both reuse the same `conn` parameter (no new connections opened).
- Connection count reduced from 4 per call → 1-2 (1 when caller passes persistent conn, 2 when self-managed: claim+post-inject combined + error-path fallback).
- `_owns_conn` flag ensures the connection is only closed when the function created it, not when it was passed from the caller.

**Plan 2 — Watcher connection reuse** (reasonix + deepseek):

- Removed `with kb.connect(board=board) as conn:` from every loop iteration in `watcher_main()`.
- Added `_ensure_conn()` function that returns a persistent connection, with:
  - **Liveness probe**: `SELECT 1` before each use; if it fails, reconnect.
  - **Connection recycling**: Every 60 seconds (`_CONN_RECYCLE_S = 60.0`), close and reconnect to release accumulated WAL reader locks and FDs.
  - **Retry on OperationalError**: Up to 3 retries with exponential backoff (2/4/8s) when opening a new connection.
  - **Consecutive error tracking**: After 5 consecutive DB errors, exit the watcher cleanly (prevents infinite crash-loop).
  - **Returns None on exhausted retries**: Caller skips this cycle and sleeps, avoiding crash.
- All DB operations in the main loop use `conn = _ensure_conn()` instead of `with kb.connect()`.
- `try/except sqlite3.OperationalError` wraps the entire active-task monitoring block: on DB error, force reconnect (`_conn = None`) and `continue` (skip this cycle, not crash).
- `adopt_orphaned_running_task()` and `claim_and_inject_one()` both accept `conn` parameter, called with `_conn` from watcher_main.
- `finally` block in watcher_main closes `_conn` explicitly (no FD leak on watcher exit).
- Also updated `adopt_orphaned_running_task()` with the same `_owns_conn` pattern.

**Files modified**:
- `plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py`
- `plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py`

**Codex listener**: Not yet patched. Apply same Plan 1 + Plan 2 pattern when needed.

## Key Implementation Pattern — Persistent DB Connection in Long-Running Watcher

```python
# In watcher_main, before the main loop:
_conn: Any = None
_conn_created_at: float = 0.0
_CONN_RECYCLE_S = 60.0  # reconnect every 60s to release WAL locks
consecutive_db_errors = 0

def _ensure_conn() -> Any:
    """Return a live DB connection, reconnecting if necessary."""
    nonlocal _conn, _conn_created_at, consecutive_db_errors
    import sqlite3

    # Recycle stale connection (release WAL reader lock)
    if _conn is not None and (time.time() - _conn_created_at) >= _CONN_RECYCLE_S:
        try:
            _conn.close()
        except Exception:
            pass
        _conn = None

    # Liveness probe — detect dead connections before using them
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

    # Open new connection with retry
    for attempt in range(3):
        try:
            _conn = kb.connect(board=board)
            _conn_created_at = time.time()
            consecutive_db_errors = 0
            return _conn
        except sqlite3.OperationalError as exc:
            consecutive_db_errors += 1
            delay = 2.0 * (2 ** attempt)
            log_line(log_path, f"DB OperationalError (attempt {attempt+1}/3, consecutive={consecutive_db_errors}): {exc}; retrying in {delay:.0f}s")
            time.sleep(delay)

    if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
        raise SystemExit(2)
    return None  # caller should skip this cycle

# In main loop:
conn = _ensure_conn()
if conn is None:
    time.sleep(min(poll_s, 5.0))
    continue
try:
    # ... all DB operations using conn ...
except Exception as exc:
    import sqlite3
    if isinstance(exc, sqlite3.OperationalError):
        consecutive_db_errors += 1
        # Force reconnect on next cycle
        try: _conn.close()
        except Exception: pass
        _conn = None
        time.sleep(min(poll_s, 5.0))
        continue
    raise

# In finally:
if _conn is not None:
    try: _conn.close()
    except Exception: pass
```

## Remaining Proposed Fixes (Not Yet Implemented)

1. **Increase wal_autocheckpoint**: From 100 to 1000. Reduces checkpoint contention frequency.

2. **Critic assist-claim delay**: Increase `--assist-claim-delay-for implementer` from 10s to 300s. Prevents critic from grabbing reclaiming implementer tasks too quickly.