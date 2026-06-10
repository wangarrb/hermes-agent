# Kanban Watcher: Idle Detection & Double-Launcher Prevention

> Key bugs fixed 2026-06-08/09 in watcher scripts and design decisions for the flock-based single-instance guard.

## Bug 1: `_sessions_latest_mtime()` scanned wrong directory and suffix

**Symptom**: Watcher's liveness check (`_sessions_latest_mtime()`) always returned stale or 0.0 mtime, making it useless for detecting that Reasonix was actively working.

**Root cause**: The function scanned `~/.reasonix/sessions/*.json` (Codex/legacy layout). Active Reasonix v1.x sessions are at `~/.config/reasonix/sessions/*.jsonl`.

| Directory | Suffix | Content | Latest mtime (as of fix) |
|---|---|---|---|
| `~/.reasonix/sessions/` | `.json` | Codex meta.json files | 2026-06-04 |
| `~/.config/reasonix/sessions/` | `.jsonl` | Reasonix v1.x active sessions | 2026-06-08 (live) |

**Fix**: `_sessions_latest_mtime()` now scans both directories and matches both `.jsonl` and `.json` suffixes:

```python
_REASONIX_SESSIONS_DIRS = [
    Path.home() / ".config" / "reasonix" / "sessions",  # Reasonix v1.x default
    Path.home() / ".reasonix" / "sessions",              # Codex / older layout
]
_REASONIX_SESSION_SUFFIXES = (".jsonl", ".json")
```

## Bug 2: Idle detection ignored liveness data (idle-reclaim loop)

**Symptom**: Watcher repeatedly reclaimed running tasks every 600s while Reasonix was actively working (e.g., waiting for API responses). Screen appeared static but Reasonix was still processing.

**Root cause**: The liveness check (line ~1178) and the idle detection (line ~1186) were independent paths:
- Liveness found session files updating → reset `last_activity_at` timeout
- Idle detection only checked screen content (`_looks_like_idle_reasonix_pane()`) → started idle timer
- Even if liveness proved Reasonix was working, idle detection still triggered reclaim after `idle_pane_reclaim_s` (600s)

**Fix**: Added session-liveness gate inside the idle detection block. When the screen looks idle but the session file was updated within `idle_pane_reclaim_s`, the idle timer is reset:

```
Screen looks idle?
  ├─ Yes → Session file updated recently (< idle_pane_reclaim_s)?
  │       ├─ Yes → Agent is working (e.g., waiting for API), reset idle timer
  │       └─ No  → Truly idle, start/continue idle timer → reclaim after threshold
  └─ No  → Not idle, reset idle timer
```

**Why this matters**: Reasonix with `auto_plan = "ask"` can spend minutes in planner mode (deepseek-v4-pro), during which the screen doesn't refresh. The old code would reclaim every 600s, interrupting the API wait → Reasonix restarts → waits again → reclaimed again → infinite loop.

## Bug 3: CodeWhale v0.8.53 "active ctx" false busy marker

**Symptom**: Critic watcher continuously prints `skip claim: DeepSeek pane still shows an active/pending Kanban prompt` every 5s, never claiming ready tasks. The pane is visibly idle (showing idle prompt like `编写任务或使用 /。`).

**Root cause**: CodeWhale v0.8.53 changed its status bar format. At idle, the bottom of the TUI shows:
```
编写任务或使用 /。
Ctrl+O Activity: thinking               active ctx 5%  ggpt-dlt-verified-2026-05-30  MCP 2
```
The `_DEEPSEEK_BUSY_MARKERS` list contained `" active ctx"`. The idle-detection function `_looks_like_idle_deepseek_pane()` first checks BUSY markers — if any BUSY marker appears in the tail, it returns False immediately, skipping the IDLE marker check. So `active ctx` (a context-usage percentage indicator) permanently blocked task claiming.

**Diagnostic procedure**: When a watcher permanently skips claims:
1. Check watcher log for repeated `skip claim` lines: `tail -20 ~/.hermes/kanban/boards/<board>/logs/deepseek-interactive-<profile>.log`
2. Dump the pane screen: `zellij --session <s> action dump-screen --pane-id <id> --full | tail -5`
3. Run marker matching against the tail: pipe through Python to check which BUSY/IDLE markers match
4. If a BUSY marker hits on a genuinely idle pane, remove it from the marker list

**Fix** (2026-06-09): Removed `" active ctx"` from `_DEEPSEEK_BUSY_MARKERS` in `deepseek_kanban_interactive.py`. The `" active ·"` marker remains and correctly matches genuine busy state (e.g. when CodeWhale is actively processing a tool call). `active ctx` is only a context-usage indicator ("5% of context window used"), not an activity indicator.

**After editing markers, restart the watcher process** — kill the launcher (parent) and watcher (child) PIDs, then relaunch. The Python module change only takes effect after process restart.

**General lesson**: When a TUI tool upgrades and changes its status bar format, re-verify that all idle/busy markers still correctly classify idle vs busy states. The marker lists are hardcoded string matches — they can't adapt to format changes automatically.

## Feature: Double-launcher prevention via flock

**Problem**: Running two launcher+watcher instances for the same (board, profile) pair causes:
- Two watchers competing to claim tasks for the same pane
- Same Reasonix receiving two task injections simultaneously → input buffer chaos
- Log file corruption from concurrent writes

**Solution**: `fcntl.flock()` on a per-(board, profile) lock file.

| Design choice | Rationale |
|---|---|
| flock vs PID file | Atomic (no check-then-lock race), auto-released on process death |
| flock vs DB registration | Simpler, no DB dependency, works even if SQLite is locked/corrupt |
| Lock file path | `~/.hermes/kanban/boards/<board>/watcher-<profile>.lock` |
| Inheritance | Watcher subprocess inherits launcher's flock fd → lock persists even if launcher crashes |

**Implementation**:
- `_acquire_watcher_lock(board, profile)`: tries `LOCK_EX|LOCK_NB`, returns fd on success, None on failure
- `_read_lock_pid(board, profile)`: reads PID from lock file for error messages
- Both `launcher_main()` and `watcher_main()` call `_acquire_watcher_lock()` at startup
- When launched by launcher_main, the watcher's second acquire succeeds harmlessly (fd already inherited)
- Error message includes old PID and `kill`/`pkill` command for user convenience

**Edge case — start-kanban.sh cleanup**: The script already kills stale workers before starting new ones. After `terminate_process_tree()`, added `sleep 1` to wait for kernel to release the flock before new launchers try to acquire it.

## Bug 4: Watcher silent-fail loop when no ready tasks exist

**Symptom**: A watcher (e.g., critic) appears completely dead — no log entries for 20+ minutes — but the process is alive (sleeping in `hrtimer_nanosleep`). The pane shows idle markers (e.g., "编写任务或使用 /。"). No tasks are being claimed.

**Root cause**: `claim_and_inject_one()` (lines 885-914 in `deepseek_kanban_interactive.py`) has multiple **silent return paths** with no logging:

```python
# Line 906-907: _select_ready_candidate() returns None → no log
candidate = _select_ready_candidate(conn, args)
if candidate is None:
    return None, None

# Line 909-910: claim_task() fails → no log
claimed = kb.claim_task(conn, candidate.id, ...)
if claimed is None:
    return None, None
```

When `_select_ready_candidate()` returns None (no ready tasks), the watcher silently falls through to `time.sleep(poll_s=60)` and loops. No log entry is written. The watcher is alive and functional, but appears dead because there's nothing to claim.

**Most common trigger**: Dependency-gated tasks. A task with `status=todo` stays todo until ALL its parent tasks reach `status=done`. The watcher calls `kb.recompute_ready(conn)` before selecting, but if the parent is still `running`, the child stays `todo` and never becomes `ready`.

Example from Egomotion4D:
```
t_344fd979 (implementer)  status=running  ← still executing
  ↓ parent
t_3e74b985 (critic)       status=todo     ← parent not done, cannot become ready
  ↓ parent
t_e70962b1 (planner)      status=todo     ← parent not done, cannot become ready
```

The critic watcher sees: idle pane → `recompute_ready()` → no tasks become ready → `_select_ready_candidate()` returns None → silent return → sleep 60s → repeat forever.

**Diagnostic procedure** — when a watcher appears dead but the pane is idle:
1. Check watcher process is alive: `ps aux | grep 'deepseek_kanban_interactive.*<profile>'`
2. Check watcher log for gaps: `tail -20 ~/.hermes/kanban/boards/<board>/logs/deepseek-interactive-<profile>.log`
3. Check task dependency status: `hermes kanban --board <board> show <task_id>` — look for `status: todo` with `parents:` pointing to a `running` task
4. If parent is still running → watcher is correctly waiting. No bug.
5. If parent is done but child is still `todo` → `recompute_ready()` may have a bug; manually unblock: `hermes kanban --board <board> unblock <child_task_id>`

**Fix recommendation** (code-level): Add logging to the silent return paths in `claim_and_inject_one()`:

```python
candidate = _select_ready_candidate(conn, args)
if candidate is None:
    log_line(log_path, "no ready task available for claim")
    return None, None

claimed = kb.claim_task(conn, candidate.id, ttl_seconds=args.ttl, claimer=claim_lock())
if claimed is None:
    log_line(log_path, f"claim_task lost race for {candidate.id}")
    return None, None
```

This makes the watcher's behavior visible in logs without changing any functional logic. The log entry rate is bounded by poll interval (60s daytime), so it won't spam.

**Distinguishing from Bug 3 (false busy marker)**: Bug 3 causes `skip claim: DeepSeek pane still shows an active/pending Kanban prompt` entries every 30s. Bug 4 causes NO log entries at all. If the log has recent `skip claim` lines → Bug 3. If the log has a gap of 10+ minutes with no entries → Bug 4 (or genuinely idle board).

## Bug 5: Launcher print() overwrites codewhale idle marker (PTY contamination)

**Symptom**: The codewhale/DeepSeek-TUI pane shows garbled content on its first line — the idle marker ("编写任务或使用 /。") is partially or fully overwritten by launcher status messages like "codewhale-kanban listener armed: profile=critic...". The watcher then can't detect idle state → permanent `skip claim` loop.

**Root cause**: The launcher process writes to the PTY master side using `print()`. The codewhale-tui process reads from the PTY slave side for its display. When the launcher prints text, it goes directly into codewhale's display buffer, overwriting whatever was on screen — typically the idle marker on line 1. The watcher reads the pane screen via `zellij dump-screen`, sees the launcher text instead of the idle marker, and classifies the pane as busy.

**Why this only affects interactive mode**: In non-interactive (worker) mode, the launcher doesn't share a PTY with codewhale. In interactive mode, the launcher IS the parent process of the codewhale PTY, so its stdout goes to the same terminal.

**Fix** (2026-06-09/10): Replaced all `print()` calls in the launcher path with `log_line()` which writes to the watcher log file instead of stdout. This includes:
- Startup banner lines ("codewhale-kanban listener armed:...")
- `build_deepseek_cmd()` stderr output
- Any other diagnostic prints in the launcher_main() path

**After fixing print() → log_line(), restart the watcher**: kill launcher + watcher PIDs, then relaunch via `start-kanban.sh` or manual invocation. Existing running processes use the old code.

## Bug 6: "turn stalled" state not recognized as busy

**Symptom**: When codewhale enters a "Turn stalled" state (LLM response timeout or error), the watcher treats the pane as idle and injects kanban task text into codewhale's stdin. The text accumulates in the input buffer, making subsequent keyboard input extremely difficult (80%+ character loss).

**Root cause**: The `_DEEPSEEK_BUSY_MARKERS` list did not include "turn stalled", so the idle detection function `_looks_like_idle_deepseek_pane()` returned True when "Turn stalled" was on screen. The watcher then proceeded to inject task text into the pane's stdin.

**Why this is destructive**: During "Turn stalled", codewhale's input box is still accepting characters (unlike during active tool calls where stdin is discarded). The injected kanban task text fills the input buffer. When the user tries to type, their input mixes with the buffered task text, causing severe input corruption.

**Fix** (2026-06-09): Added `"turn stalled"` to `_DEEPSEEK_BUSY_MARKERS` in `deepseek_kanban_interactive.py`. When "Turn stalled" appears on screen, the watcher skips claiming and waits for the state to resolve.

**Design decision — no auto-restart on Turn stalled**: Restarting codewhale on Turn stalled would lose context, kill subprocesses, and potentially corrupt task state. The watcher simply waits. The user can manually intervene (Escape → retry) if they want to unstick the pane.

## Key file locations

- Reasonix watcher: `~/.hermes/hermes-agent/plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py`
- DeepSeek/CodeWhale watcher: `~/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py`
- Codex watcher: `~/.hermes/hermes-agent/plugins/kanban/codex_listener/codex_kanban_interactive.py`
- Worker runtime (claim policy, candidate selection): `~/.hermes/hermes-agent/hermes_cli/kanban_worker_runtime.py`
- Reasonix sessions (v1.x): `~/.config/reasonix/sessions/*.jsonl`
- Codex sessions (legacy): `~/.reasonix/sessions/*.json`
- Watcher logs: `~/.hermes/kanban/boards/<board>/logs/<agent>-interactive-<profile>.log`
- Lock files: `~/.hermes/kanban/boards/<board>/logs/watcher-<profile>.lock`
- Launch script: `~/bin/start-kanban.sh`