# Slow-API Idle-Reclaim Loop (2026-06-08) — RESOLVED

## Incident

Implementer reasonix pane's watcher entered a 600s-interval idle-reclaim loop, re-injecting the same task every ~10 minutes. Unlike the 8-second crash-loop (Failure 6), this one happened at the configured `idle_pane_reclaim_s` interval because the agent was waiting for an API response and the screen didn't update.

## Timeline (from watcher log)

```
10:00:12  claimed+injected t_037b1e0c
10:01:07  idle pane observed for active task t_037b1e0c
10:11:09  idle pane reclaim t_037b1e0c: idle 602s >= 600s
10:11:19  claimed+injected t_037b1e0c (re-injected same task)
10:11:49  idle pane observed
10:21:51  idle pane reclaim (again)
10:22:01  claimed+injected t_037b1e0c
10:22:31  idle pane observed
... (16 total reclaim cycles over 1h23m)
11:23:38  reasonix exited rc=-15 (external SIGTERM killed all panes simultaneously)
```

Total spend: ¥7.60-8.24 across 16 cycles, but the task was never completed.

## Root Cause — Two Bugs

### Bug 1: `_sessions_latest_mtime()` scanned wrong directory and suffix

The liveness-check function `_sessions_latest_mtime()` only scanned `~/.reasonix/sessions/*.json` (Codex legacy layout, mtime stuck at June 4). Active Reasonix sessions are in `~/.config/reasonix/sessions/*.jsonl` (last updated June 8 11:13). The liveness check was effectively a no-op — it never detected that reasonix was working.

**Fix**: Changed `_REASONIX_SESSIONS_DIR` to `_REASONIX_SESSIONS_DIRS` (a list of two directories), and `_sessions_latest_mtime()` now scans both `~/.config/reasonix/sessions/` and `~/.reasonix/sessions/`, matching both `.jsonl` and `.json` suffixes.

### Bug 2: Idle detection ignored liveness data

Even when liveness found session file updates (proving reasonix was working), the idle-pane detection path only checked screen content. If the screen didn't change for 600s while reasonix was waiting for an API response → reclaim → interrupt work → restart → repeat.

**Fix**: Added session-liveness gate inside the idle-pane detection block. Before declaring a pane idle and starting the idle timer, check if `latest_mtime` (now correctly populated) indicates a session file was modified within the last `idle_pane_reclaim_s`. If so, reset the idle timer — the agent is working even though the screen hasn't refreshed.

Logic flow after fix:
```
Screen looks idle?
  ├─ Yes → Session file updated recently (< idle_pane_reclaim_s)?
  │       ├─ Yes → Agent is working, reset idle timer, no reclaim
  │       └─ No  → Genuinely idle, start/continue idle timer
  └─ No  → Not idle, reset idle timer
```

## Distinguishing From Failure 6 (Crash-Loop Reclaim)

| Aspect | Failure 6 (Crash-Loop) | Failure 13 (Slow-API) |
|--------|----------------------|----------------------|
| Interval | ~8 seconds (startup_delay_s) | ~600 seconds (idle_pane_reclaim_s) |
| Cause | DB crash → watcher restart → immediate re-reclaim | Screen idle during API wait → reclaim |
| Log pattern | `reclaimed orphaned idle-pane` every 8s + DB errors | `idle pane reclaim: idle 602s >= 600s` every 10min, no errors |
| Fix | Grace period in adopt_orphaned_running_task + DB error protection | Session-liveness gate in idle detection + fix session dir/suffix |

## Additional Finding: Double Launcher

At 10:00:12 AND 10:00:27, two launchers started for the same implementer pane. This happened because manual restart (during upgrade) and start-kanban.sh restart overlapped. The second watcher detected an orphaned task and adopted it (benign but confusing). Two watchers polling the same pane can cause race conditions.

## Files Changed

- `plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py`:
  - Lines ~619-653: `_REASONIX_SESSIONS_DIRS` (list of 2 dirs), `_REASONIX_SESSION_SUFFIXES`, `_sessions_latest_mtime()` now scans both dirs and both suffixes
  - Lines ~1196-1237: idle-pane detection now checks `latest_mtime` before starting idle timer — session-liveness gate

## Diagnostic Commands

```bash
# Check watcher reclaim frequency
grep "idle pane reclaim" ~/.hermes/kanban/boards/<board>/logs/reasonix-interactive-<profile>.log | tail -20

# Check session-liveness gate in action
grep "session-liveness" ~/.hermes/kanban/boards/<board>/logs/reasonix-interactive-<profile>.log | tail -10

# Check reasonix session activity (is the agent actually working?)
ls -lt ~/.config/reasonix/sessions/ | head -5

# Check pane content
zellij action dump-screen --pane-id <id> | head -50

# Check double launchers
grep "launcher starting watcher" ~/.hermes/kanban/boards/<board>/logs/reasonix-interactive-<profile>.log | tail -5
```

## Pitfall for Future Debugging

When adding liveness checks that depend on file mtime, always verify the actual file paths and suffixes match what the running tool produces. Reasonix v1.x uses `~/.config/reasonix/sessions/*.jsonl`; Codex uses `~/.reasonix/sessions/*.json`. A liveness check that scans the wrong directory is worse than no liveness check — it creates false confidence that the system is working when it's actually blind.
