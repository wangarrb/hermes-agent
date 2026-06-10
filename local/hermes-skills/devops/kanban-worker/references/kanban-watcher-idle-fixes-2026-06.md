# Kanban Watcher: Idle-Reclaim and Double-Launcher Fixes (2026-06-08)

> Addendum to Failure 13 in kanban-worker SKILL.md. The mitigation options have been extended.

## New mitigation: Session-liveness gate (option 2, now implemented)

The watcher now checks Reasonix session file mtime before reclaiming. If `~/.config/reasonix/sessions/*.jsonl` was updated within `idle_pane_reclaim_s`, the agent is considered "working" (e.g., waiting for API response during planner mode) and the idle timer resets.

**Why this is better than just increasing the threshold:**
- Distinguishes "waiting for API" from "truly stuck" without increasing reclaim latency for genuinely stuck agents
- No configuration change needed — uses the existing `idle_pane_reclaim_s` threshold as the session-liveness window
- Works for any Reasonix agent state where the screen is static but the session file is being updated

**Root cause of the original bug**: `_sessions_latest_mtime()` was scanning `~/.reasonix/sessions/*.json` (Codex/legacy layout). Active Reasonix v1.x sessions are at `~/.config/reasonix/sessions/*.jsonl`. The liveness check was completely ineffective.

## New mitigation: Double launcher prevention (flock)

`fcntl.flock()` on `~/.hermes/kanban/boards/<board>/logs/watcher-<profile>.lock` prevents two watcher instances from competing for the same pane.

- Both `launcher_main()` and `watcher_main()` acquire the lock at startup
- If another instance holds it, the new one prints the old PID and exits
- Lock is auto-released on process death and inherited by the watcher subprocess
- Each (board, profile) pair gets its own lock file

If you see "已有 X watcher 在运行" at startup, the previous instance is still alive. Kill it with the suggested command or wait for it to exit.

## Reference

Full diagnosis and design rationale: `kanban-orchestrator/references/kanban-watcher-idle-and-lock.md`
