# Kanban DB WAL→DELETE Journal Mode Migration (2026-06-08)

## Why

WAL mode under multi-process concurrent writes (3+ watchers + gateway) caused repeated "database disk image is malformed" corruption. DELETE mode serializes writes via rollback journal — slower but far more robust for kanban's lightweight write workload.

## Implementation

### Code change in `kanban_db.py:connect()`

Location: `~/.hermes/hermes-agent/hermes_cli/kanban_db.py`, inside `_INIT_LOCK` block.

```python
_force_delete = os.environ.get("HERMES_KANBAN_FORCE_DELETE_JOURNAL", "1").strip()
if _force_delete not in ("0", "false"):
    # If on-disk is already WAL, switch to DELETE + checkpoint.
    mode_row = conn.execute("PRAGMA journal_mode").fetchone()
    if mode_row and mode_row[0] == "wal":
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
else:
    from hermes_state import apply_wal_with_fallback
    apply_wal_with_fallback(conn, db_label=f"kanban.db ({path.name})")
```

Key design decisions:
- **Default "1"** instead of `os.environ.setdefault`: systemd-launched gateway doesn't inherit shell env vars reliably. Using `.get("KEY", "1")` ensures the default is always "force DELETE" regardless of environment.
- **`from hermes_state import apply_wal_with_fallback` moved to else branch**: The original top-level import was removed. If DELETE mode is active (the default), `apply_wal_with_fallback` is never imported or called.
- **Removed `wal_autocheckpoint=100` PRAGMA**: This PRAGMA only applies in WAL mode. Since DELETE is now the default, running it is harmless but confusing. Left in place for now (SQLite ignores it in DELETE mode).

### How to verify

```bash
# Check journal mode (must NOT be "wal")
python3 -c "
import sqlite3
conn = sqlite3.connect('$HOME/.hermes/kanban/boards/<board>/kanban.db')
print('journal_mode:', conn.execute('PRAGMA journal_mode').fetchone()[0])
conn.close()
"
```

## Critical pitfalls discovered during implementation

### 1. Residual processes block journal_mode switching

**Problem**: `PRAGMA journal_mode=DELETE` silently fails when other processes have the DB open in WAL mode. The PRAGMA returns "wal" instead of "delete", but the code may not check the return value.

**Scenario**:
1. `hermes gateway stop` — gateway process exits
2. But watcher subprocesses (reasonix_kanban_interactive.py --watch-child) are NOT killed by gateway stop
3. These watchers hold WAL connections to kanban.db
4. `PRAGMA journal_mode=DELETE` fails silently because SQLite can't change journal mode while other connections exist
5. Next `hermes gateway start` — gateway opens the DB, reads WAL from the header, and operates in WAL mode
6. Result: journal_mode appears to "revert" to WAL after every restart

**Fix**: Kill ALL processes holding the DB before switching:

```bash
hermes gateway stop
fuser <db_path> | while read pid; do kill -9 $pid 2>/dev/null; done
sleep 1
# Verify no holders remain
fuser <db_path>  # should return nothing (exit code 1)
# Now switch
python3 -c "
import sqlite3
conn = sqlite3.connect('<db_path>')
conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
conn.execute('PRAGMA journal_mode=DELETE')
print(conn.execute('PRAGMA journal_mode').fetchone()[0])  # should say 'delete'
conn.close()
"
hermes gateway start
```

### 2. Gateway stderr is a socket

**Problem**: Debug `print(..., file=sys.stderr)` from `kanban_db.py:connect()` does NOT appear in:
- `journalctl --user -u hermes-gateway`
- `~/.hermes/logs/gateway.log`
- `~/.hermes/logs/errors.log`

The gateway process's fd/2 points to a socket (`socket:[...]`), not a pipe or file. Stderr output is swallowed.

**Fix**: Use Python logging instead:
```python
import logging
logging.getLogger("kanban_db").warning("...")
```

This appears in gateway.log and errors.log.

### 3. `os.environ.setdefault` timing issue

**Problem**: `os.environ.setdefault("HERMES_KANBAN_FORCE_DELETE_JOURNAL", "1")` sets the var in the current process's environment, but it may not be seen by code that already imported and cached the value before `setdefault` ran. More importantly, systemd-launched gateway processes may have the var already set to something else (or not set at all).

**Fix**: Use `os.environ.get("HERMES_KANBAN_FORCE_DELETE_JOURNAL", "1")` with default "1" inline. This reads the env var fresh on every `connect()` call and doesn't depend on `setdefault` having been called earlier.

### 4. `_guard_existing_db_is_healthy()` probe connection

The `_guard_existing_db_is_healthy()` function (called before `_INIT_LOCK`) opens a short-lived probe connection to check `PRAGMA integrity_check`. This probe:
- Inherits the on-disk journal_mode (WAL if that's what the header says)
- Is opened and closed before the main `connect()` logic runs
- Temporarily holds a WAL reader slot
- Is harmless in practice, but means the first `connect()` is not the only DB access during initialization

## Post-migration verification checklist

After switching to DELETE mode:

1. **No WAL/SHM sidecar files**: `ls <db_path>-wal <db_path>-shm` should return "not found"
2. **Journal mode persists across gateway restart**: Stop and start gateway, then re-check `PRAGMA journal_mode`
3. **Integrity OK**: `PRAGMA integrity_check` returns "ok"
4. **No corrupt backups appearing**: Monitor `ls <db_path>.corrupt.*` over 24-48 hours
5. **Kanban operations work**: `hermes kanban --board <slug> list --status ready` succeeds without "database is locked"

## Debugging trace: WAL reverting after restart (2026-06-08 session)

**Observed**: After manually setting `PRAGMA journal_mode=DELETE` and starting gateway, DB reverted to WAL within seconds. This happened 5+ times consecutively.

**Root cause**: Residual watcher subprocess PIDs (2439479, 2441424, 2447357) from a previous zellij kanban session were still holding WAL connections. These processes survived `hermes gateway stop` because they are independent launcher processes, not gateway children.

**Key diagnostic steps**:
1. `fuser <db_path>` revealed PIDs that `hermes gateway stop` didn't kill
2. `ps -p <pid> -o pid,comm,args` confirmed they were watcher `--watch-child` processes
3. `kill -9 <pid>` was required (SIGTERM wasn't sufficient — process was stuck in SQLite I/O wait)
4. After killing all residual holders, `PRAGMA journal_mode=DELETE` succeeded and persisted

**Also discovered**: `.pyc` cache was initially suspected but NOT the cause. The editable install (`__editable___hermes_agent_0_8_0_finder.py`) correctly maps `hermes_cli` to the source tree, so code changes take effect immediately. The real issue was always the residual processes.

**Also discovered**: `hermes kanban list` from CLI successfully executed the DELETE-switch code (confirmed via debug logging), but gateway's internal `connect()` calls were not visible because gateway stderr is a socket. This led to a false conclusion that the patch wasn't loaded — it was loaded, but residual processes prevented it from taking effect.

**Correct fix sequence** (validated):
1. `hermes gateway stop`
2. `fuser <db_path>` → identify all holders → `kill -9` each
3. Verify `fuser <db_path>` returns nothing
4. Manually switch: `PRAGMA wal_checkpoint(TRUNCATE)` then `PRAGMA journal_mode=DELETE`
5. Verify: `PRAGMA journal_mode` returns "delete"
6. `hermes gateway start`
7. Wait 5s, re-verify `PRAGMA journal_mode` still returns "delete"