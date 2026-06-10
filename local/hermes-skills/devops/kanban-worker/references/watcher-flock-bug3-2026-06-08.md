# Watcher flock conflict — Bug3 (start_new_session breaks fd inheritance)

**Date**: 2026-06-08
**Affected**: reasonix_kanban_interactive.py, all panes using Reasonix

## Symptom

The watcher subprocess (started with `--watch-child`) immediately exits with rc=2, logging `"refusing to start: <profile> watcher already running (PID <launcher_pid>)"`. The launcher retries the watcher up to `MAX_WATCHER_RESTARTS` (10) times, then gives up. No tasks are ever claimed or injected.

## Root cause

`launcher_main()` calls `_acquire_watcher_lock()` and writes its own PID to the lock file. The watcher subprocess is spawned with `start_new_session=True` in `subprocess.Popen()`, which creates a new process session. This means the flock fd is NOT inherited — the watcher child cannot use the parent's lock. When `watcher_main()` calls `_acquire_watcher_lock()`, it sees the lock file contains the launcher's PID (which is still alive), and `fcntl.flock(LOCK_EX|LOCK_NB)` fails because the launcher still holds the flock. The watcher refuses to start.

## Fix applied

In `watcher_main()`, after `_acquire_watcher_lock()` returns None:
1. Read the lock file PID (`_read_lock_pid()`)
2. Check if it equals `os.getppid()` (the launcher) AND the parent is alive (`_pid_alive()`)
3. If yes, call `_acquire_watcher_lock_after_parent()` to steal the lock
   - Open the lock file
   - `fcntl.flock(fd, LOCK_EX|LOCK_NB)` — will succeed since launcher is alive but flock was not inherited
   - Write our own PID (`os.getpid()`) into the lock file
   - Return fd

New function `_acquire_watcher_lock_after_parent(board, profile, parent_pid)` added at line ~706 in `reasonix_kanban_interactive.py`.

## Diagnostic

If watcher logs show repeated `"refusing to start"` messages followed by `"watcher restart limit exceeded"`:
1. Check lock file: `cat ~/.hermes/kanban/boards/<board>/logs/watcher-<profile>.lock`
2. If the PID matches the launcher process, Bug3 is the root cause
3. Kill the launcher, clear the lock file, restart the pane

## Stale lock cleanup

If the launcher crashes, the flock is auto-released by the kernel. The lock file still contains the dead PID, but `_acquire_watcher_lock()` will succeed on next start. No manual cleanup needed. However, if you delete the `.lock` file while the launcher is running and it has exceeded `MAX_WATCHER_RESTARTS`, you must also restart the launcher — it won't retry after hitting the limit.