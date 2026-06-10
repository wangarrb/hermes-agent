# Interactive Launcher Crash-Loop Debugging (2026-06-06)

## Incident

Implementer reasonix pane in `kanban-egomotion4d` session silently died. No reasonix or watcher process running. Zellij pane showed stale screen content.

## Diagnosis Steps

1. `ps aux | grep reasonix` — no processes found
2. `zellij -s kanban-egomotion4d action list-panes` — pane still exists as `implementer-reasonix listening`
3. `zellij -s kanban-egomotion4d action dump-screen --pane-id terminal_1` — showed reasonix was "thinking" for 1150+ seconds with a bash timeout error
4. Checked watcher log: `tail -50 ~/.hermes/kanban/boards/egomotion4d/logs/reasonix-interactive-implementer.log`
   - Found 37996 lines of `skip claim: Reasonix pane still shows an active/pending Kanban prompt`
   - Found `OperationalError: disk I/O error` on claim attempts
   - Found watcher crash-loop: claim → I/O error → crash → restart → repeat, 319 times
5. Checked board DB directory: found 4 `.corrupt.*.bak` files indicating recurring DB corruption
6. `PRAGMA integrity_check` — currently OK (auto-recovered from corruption)
7. Task `t_e573a455` stuck in `running` with dead `worker_pid=1118486`

## Root Causes (3 layers)

### Layer 1: Reasonix TUI hung
- Bash command timed out (>2min) while running ssh to gpuserver
- Reasonix entered "thinking" state for 1150+ seconds
- Eventually exited

### Layer 2: Watcher detected stale pane but couldn't break out
- After reasonix exited, watcher saw old screen content → "skip claim" forever
- No PID liveness check in launcher's main loop
- `_pid_alive()` function existed but wasn't used in the launcher loop

### Layer 3: DB I/O error → watcher crash-loop
- `OperationalError: disk I/O error` on `kb.connect()` during claim
- Watcher crashed (rc=1), launcher restarted it immediately
- No restart limit, no DB retry, no backoff
- 319 iterations of: claim → I/O error → crash → restart → claim again

## Code Changes Applied

### File: `plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py`

1. **Launcher PID liveness check** (Protection 1):
   ```python
   if not _pid_alive(reasonix_proc.pid):
       second_poll = reasonix_proc.poll()
       if second_poll is not None:
           rc = int(second_poll)
           break
       time.sleep(1.0)
       third_poll = reasonix_proc.poll()
       if third_poll is not None:
           rc = int(third_poll)
           break
       log_line(log_path, f"reasonix pid {reasonix_proc.pid} is dead but poll() hasn't returned; forcing break")
       rc = -1
       break
   ```

2. **Watcher restart limit** (Protection 2):
   ```python
   MAX_WATCHER_RESTARTS = 10
   # ... in the restart branch:
   if watcher_restart_count > MAX_WATCHER_RESTARTS:
       log_line(log_path, f"watcher restart limit ({MAX_WATCHER_RESTARTS}) exceeded; ...")
       watcher = None
       continue
   ```

3. **DB connect retry wrapper** (Protection 3):
   ```python
   def _db_connect_with_retry(*, max_retries=3, base_delay=2.0):
       # Context manager wrapping kb.connect()
       # Retries on sqlite3.OperationalError with exponential backoff
       # Tracks consecutive_db_errors; exits watcher after MAX_CONSECUTIVE_DB_ERRORS=5
   ```
   - Replaced `kb.connect(board=board)` in watcher loop with `_db_connect_with_retry()`

### File: `plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py`

- Protections 1 and 2 applied (PID liveness + restart limit)
- Protection 3 (DB retry) NOT yet applied — add if DB errors appear

## Task Recovery

```bash
# Reclaim stuck task (but reclaim set it to running with another dead pid)
hermes kanban --board egomotion4d reclaim t_e573a455 --reason "worker dead"

# Force reset via SQL when reclaim doesn't work
sqlite3 ~/.hermes/kanban/boards/egomotion4d/kanban.db \
  "UPDATE tasks SET status='todo', worker_pid=NULL, claim_lock=NULL, claim_expires=NULL, started_at=NULL WHERE id='t_e573a455';"
```

## Follow-up: Assist-Role Double-Claim (same session)

After the crash-loop was fixed, the user reported the same task (`t_e573a455`) was claimed by two "implementers". Investigation showed:

- **Implementer pane**: `--claim-assignees implementer`
- **Critic pane**: `--claim-assignees critic,implementer` (from `--assist-role critic:implementer`)

Both panes can claim `implementer` tasks. When the implementer's watcher crash-looped, its claim expired and the critic's watcher claimed the same task. The implementer watcher restarted and tried to adopt it back. This created an alternating claim/reclaim cycle visible in `task_events`:

- Run 324 (19:57:29): implementer claimed
- Runs 342-346 (20:21-20:34): critic claimed after implementer's claims expired
- Event 1848: `reasonix-interactive injected prompt: /home/wyr/code/Egomotion4D/.reasonix-kanban/egomotion4d/critic/t_e573a455.md`

**Root cause**: `--assist-claim-delay-for implementer=10` only delays the initial claim when a task first becomes ready. After reclaim (claim expired/crashed), the task goes back to `ready` and the delay threshold may not reset. The assist pane grabs it immediately.

**Mitigation**: Increase assist-claim-delay significantly (e.g. 300s) when the primary pane is unstable, or remove the assist-role entirely until the primary pane stabilizes.

## Follow-up: Self-Poll Mode Removal (2026-06-06, same session)

Self-poll task delivery mode was removed from all five listener files:

- `plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py`
- `plugins/kanban/codex_listener/codex_kanban_interactive.py`
- `plugins/kanban/codex_listener/codex_kanban_listener.py`
- `plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py`
- `plugins/kanban/deepseek_listener/deepseek_kanban_listener.py`

### Pitfall 1: Shared functions deleted with self-poll code

`write_role_instructions_file` and `write_workspace_reasonix_toml` in reasonix_kanban_interactive.py were only called from the self-poll branch, but inject mode also needed the `role_instructions_path` they produced. Deleting them caused `NameError: name 'role_instructions_path' is not defined`.

**Fix**: Restored a combined `_write_role_instructions()` function called from `launcher_main` regardless of delivery mode.

**Lesson**: When removing a code path, trace ALL consumers of each function's return values — not just the call graph from the removed branch, but also downstream references in retained branches.

### Pitfall 2: Shell launch script out of sync with Python scripts

After removing `--task-delivery` from the Python interactive scripts' argparse, `start-kanban.sh` still passed `--task-delivery inject` and set `HERMES_KANBAN_TASK_DELIVERY` / `HERMES_KANBAN_SELF_POLL_OWNER` env vars. This caused `error: unrecognized arguments: --task-delivery inject` on launch.

**Fix**: In `start-kanban.sh`:
- Removed `--task-delivery` from all `build_role_command` branches (codex/deepseek-tui/deepseek-reasonix)
- Removed `HERMES_KANBAN_TASK_DELIVERY` and `HERMES_KANBAN_SELF_POLL_OWNER` env vars
- Removed `CODEX_LISTEN` and `DEEPSEEK_LISTEN` variables and worker-mode branches
- Kept `--task-delivery` in the shell arg parser as a backward-compatible no-op (warns and forces `inject`)

**Lesson**: When removing a CLI argument from Python scripts, always check if shell wrappers / launch scripts also pass that argument. Grep for the flag name across all wrapper scripts, systemd units, and cron jobs.

### Pitfall 3: KANBAN_TASK_BOUNDARY causing SSH kill-0 infinite loops

The `KANBAN_TASK_BOUNDARY` injection text said "不要延续上一轮输出" — this caused the AI to forget it had already tried monitoring a remote process, and re-derive the same `ssh + kill -0` strategy from conversation traces.

**Fix**: Changed the injection text to explicit context-continuation rules (don't repeat previous actions, don't poll dead PIDs, check logs instead of kill-0 loops). Added "远程进程管理" rule to `build_interactive_prompt()`.

## Lessons

- Always check watcher logs first when an interactive pane dies silently
- Multiple `.corrupt.*.bak` files in the board DB directory = recurring DB corruption
- `hermes kanban reclaim` may not properly reset a task if the reclaim itself hits DB issues — SQL fallback may be needed
- The `_pid_alive()` helper existed but wasn't used in the launcher loop — always check existing utilities before adding new ones
- Assist-role panes can create double-claim cycles when the primary pane is crash-looping; check `task_events` for alternating claims from different PIDs/profiles
- When removing a code path, check that functions' return values aren't used by retained paths
- When removing CLI arguments from Python, grep shell wrappers and launch scripts for the same flag
- Task boundary injection text must NOT instruct the AI to discard context — instead provide explicit rules about what NOT to repeat
