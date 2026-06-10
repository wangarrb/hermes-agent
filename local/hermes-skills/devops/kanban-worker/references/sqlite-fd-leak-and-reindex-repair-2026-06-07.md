# SQLite FD Leak Root Cause + REINDEX Auto-Repair + Dispatch Prevention

Date: 2026-06-07

## Root cause: FD exhaustion, not WAL contention

Previous analysis (Failure 7 in SKILL.md) attributed `disk I/O error` to WAL checkpoint
competition between concurrent writers. Deeper investigation revealed the primary root
cause is **FD (file descriptor) exhaustion**:

### Evidence chain

1. **Corrupt backup pattern**: 9 `.corrupt.*.bak` files across 2 days (6/6 16:13-16:16,
   6/6 20:20, 6/7 01:03, 6/7 03:59). All showed index corruption
   (`idx_events_task` or `idx_runs_status`).

2. **Error timing**: Every `disk I/O error` occurred immediately after a successful
   `claimed+injected` operation. The claim wrote to DB successfully, but the NEXT
   DB operation (heartbeat, status check) failed with `disk I/O error`.

3. **Concurrent crashes**: Both implementer and critic watchers crashed at the same
   time (00:53:16 and 00:53:23), indicating a global DB unavailability, not a
   per-watcher issue.

4. **ulimit -n = 1024**: The default soft FD limit. With `with kb.connect()` leaking
   FDs on every poll cycle, and each watcher polling every 6 seconds, FDs accumulate
   toward 1024 in hours.

5. **SQLite error mapping**: When `open()` fails due to FD exhaustion, SQLite returns
   `SQLITE_IOERR` which Python surfaces as `OperationalError: disk I/O error`. This
   is misleading — it's NOT a disk problem, it's a process resource limit.

### Why FD leak happens

`with kb.connect() as conn:` only calls `__enter__` (PRAGMA setup) and `__exit__`
(commit/rollback). It NEVER calls `conn.close()`. The connection stays open until
Python GC collects it. With 8 `with kb.connect()` call sites per listener file, a
single claim-and-inject cycle leaks 4+ FDs.

### Index corruption mechanism

When FDs are exhausted and SQLite cannot complete a write transaction (cannot open
WAL/SHM file for checkpoint), the write may be partially committed. This leaves
indexes in an inconsistent state — specifically `idx_events_task` and
`idx_runs_status` which are the most frequently written indexes.

## Three-layer fix

### Layer 1: Eliminate FD leak (Plan 1 + Plan 2)

Already documented in `references/persistent-connection-implementation-2026-06-07.md`.

### Layer 2: Raise ulimit

```python
# In watcher_main(), before any DB operations:
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 4096:
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
except Exception:
    pass
```

Default soft limit is 1024. Raising to 4096 provides 4x safety margin. The hard
limit on this system is 1048576, so 4096 is well within bounds.

### Layer 3: REINDEX auto-repair

When `DatabaseError("malformed")` or `DatabaseError("corrupt")` is detected, the
watcher now attempts `REINDEX` before crashing:

1. Open a raw sqlite3 connection (bypassing kb.connect which would also fail)
2. Execute `PRAGMA integrity_check` to confirm corruption
3. Execute `REINDEX` to rebuild all indexes
4. Execute `PRAGMA integrity_check` again to verify repair
5. Close repair connection and retry `kb.connect()`

This is applied in two places:
- `_ensure_conn()` — during connection establishment
- Active-task loop exception handler — during normal operations

**Important**: `OperationalError` must be caught BEFORE `DatabaseError` since
`OperationalError` is a subclass of `DatabaseError`. If the order is reversed,
all `OperationalError` instances are caught by the `DatabaseError` handler and
the REINDEX logic runs unnecessarily.

## Dispatch prevention

### AGENTS.md §8.0

Added hard rule prohibiting `hermes kanban dispatch` and `hermes kanban daemon`.
Corrected the erroneous description that "dispatch daemon automatically dispatches
tasks" — it does NOT auto-start. It only runs when explicitly invoked.

### start-kanban.sh auto-kill

Before launching the zellij session, the script now detects and kills any stray
dispatch/daemon processes. This prevents headless workers from competing with
interactive listeners for task claims.

### Why dispatch is harmful with interactive listeners

| Mechanism | Claim path | Task visibility | Claim lock format |
|-----------|-----------|----------------|-------------------|
| Interactive listener | Watcher → zellij inject | Visible in TUI | `hostname:pid:reasonix-interactive` |
| Dispatch daemon | `_default_spawn` → headless `hermes chat` | Background only | `hostname:pid` (no suffix) |

When dispatch claims a task, the interactive listener sees it as already `running`
and skips it. The user sees nothing in their visible panes.

## Files modified

| File | Changes |
|------|---------|
| `reasonix_kanban_interactive.py` | REINDEX auto-repair + ulimit + DatabaseError handling |
| `deepseek_kanban_interactive.py` | Same |
| `AGENTS.md` (Egomotion4D) | §8.0 dispatch prohibition rule |
| `start-kanban.sh` | Auto-kill stray dispatch processes |

## Verification

- `python3 -m py_compile` passes for both listener files
- `resource.setrlimit` verified: soft goes from 1024 to 4096
- `kb.connect(board='egomotion4d')` verified: busy_timeout=120000, integrity=ok
- DB integrity restored: `PRAGMA integrity_check` returns `ok` after manual REINDEX
- 9 corrupt backup files moved to trash
