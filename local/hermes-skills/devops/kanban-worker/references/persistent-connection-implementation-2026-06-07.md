# Persistent Connection Implementation (Plan 1 + Plan 2)

Date: 2026-06-07

## Plan 1: claim_and_inject_one connection merge

**Before**: `claim_and_inject_one` opened 4 separate `kb.connect()` calls:
1. Claim + prep (release_stale_claims, recompute_ready, claim_task, set_workspace_path, build_worker_context)
2. Pre-inject reclaim (pane became unsafe before injection)
3. Inject-fail reclaim (zellij injection failed)
4. Post-inject (add_comment + heartbeat_worker)

**After**: 1 connection per call — all operations reuse the same connection:
- Claim phase uses a single connection (either caller-provided or self-created)
- Pre-inject reclaim reuses the same connection
- Inject-fail reclaim reuses the same connection
- Post-inject add_comment + heartbeat_worker reuses the same connection

**Key pattern**: `conn` parameter (not `_owns_conn` for codex)
```python
def claim_and_inject_one(args, *, log_path, conn=None):
    if conn is None:
        conn = kb.connect(board=board)
    # ... all DB operations using conn ...
    # No explicit close — caller (watcher_main) manages the connection lifecycle
```

When the caller (watcher_main via `_ensure_conn()`) passes a persistent connection, the function reuses it. When called standalone (conn=None), the function creates its own connection.

**Applied to**: reasonix_kanban_interactive.py, deepseek_kanban_interactive.py, codex_kanban_interactive.py

## Plan 2: watcher_main persistent connection

**Before**: `watcher_main` opened a new `kb.connect()` on every loop iteration (every 6 seconds). This caused:
- PRAGMA journal_mode=WAL on every connection (unnecessary)
- WAL reader lock accumulated (never released until GC)
- FD leak (kb.connect() context manager never closes)
- Connection setup overhead on every poll cycle

**After**: `_ensure_conn()` returns a persistent connection:
- 60-second recycling (`_CONN_RECYCLE_S = 60.0`) — periodically closes and reconnects to release WAL locks
- Liveness probe (`conn.execute("SELECT 1")`) — detects dead connections before use
- Retry on OperationalError — 3 attempts with exponential backoff (2/4/8s)
- REINDEX auto-repair on DatabaseError with "malformed"/"corrupt"
- Consecutive error tracking — after 5 consecutive DB errors, exits watcher
- `finally` block in watcher_main closes the persistent connection on exit

```python
_conn: Any = None
_conn_created_at: float = 0.0

def _ensure_conn() -> Any:
    nonlocal _conn, _conn_created_at, consecutive_db_errors
    # 1. Recycle stale connection (>=60s old)
    # 2. Liveness probe (SELECT 1)
    # 3. Open new connection with retry on OperationalError
    # 4. REINDEX on DatabaseError("malformed"/"corrupt")
    # 5. Return None on failure (caller skips this cycle)
```

**Error handling in main loop**:
```python
conn = _ensure_conn()
if conn is None:
    if consecutive_db_errors >= MAX_CONSECUTIVE_DB_ERRORS:
        break  # exit watcher
    time.sleep(poll_s)
    continue
if active_task:
    try:
        status, current_run_id = _task_status(conn, active_task)
    except (sqlite3.OperationalError, sqlite3.DatabaseError) as exc:
        consecutive_db_errors += 1
        time.sleep(min(poll_s, 5.0))
        continue
    consecutive_db_errors = 0
    # ... heartbeat with try/except ...
```

**Also applied to**: `reclaim_orphaned_running_task()` (codex) / `adopt_orphaned_running_task()` (reasonix/deepseek) — accepts optional `conn` parameter.

**Applied to**: reasonix_kanban_interactive.py, deepseek_kanban_interactive.py, codex_kanban_interactive.py

## Codex watcher Plan 2 implementation notes (2026-06-07)

The codex watcher was the last to be patched. Key differences from reasonix/deepseek:

1. **No `_owns_conn` pattern** — codex `claim_and_inject_one` simply accepts `conn=None` and creates a new connection if None. The caller (watcher_main) manages the connection lifecycle via `_ensure_conn()` + `finally: _conn.close()`.

2. **`reclaim_orphaned_running_task`** also accepts `conn=None` — same pattern as `claim_and_inject_one`.

3. **`sqlite3` imported at module top** — added `import sqlite3` to the file's top-level imports since the DB error handlers need it.

4. **`kb.kanban_db_path()`** (not `board_db_path`) — used in the REINDEX repair code to locate the DB file.

5. **Indentation fix** — removing the `with kb.connect(board=board) as conn:` wrapper in `reclaim_orphaned_running_task` required dedenting the entire function body by one level. Missing this causes `IndentationError`.

## kb.connect() context manager FD leak — the key discovery

`with kb.connect() as conn:` is equivalent to:
```python
conn = kb.connect(board=board)  # Opens sqlite3.Connection
try:
    yield conn
except Exception:
    conn.__exit__(None, None, None)  # rollback
    raise
else:
    conn.__exit__(None, None, None)  # commit
```

It **NEVER** calls `conn.close()`. The connection stays open until Python's garbage collector collects it. With 8 `with kb.connect()` call sites per listener file, a single claim-and-inject cycle leaks 4+ file descriptors and WAL reader locks.

`kb.connect_closing()` is the version that actually closes connections — but it's only used in CLI commands (one-shot operations), not in the long-running watcher loops.

## Compilation verification

All three files pass `python3 -c "import ast; ast.parse(open(f).read())"` after changes.
