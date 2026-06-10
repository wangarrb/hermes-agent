# Reasonix Session Continuation for Kanban Workers

## Background

Reasonix kanban workers used to start a fresh session every time (`--new` / no `--continue`). Each restart lost all conversation context — critic had to re-learn the project state from scratch. User (老王) requested that at least critic's Reasonix should be able to continue previous conversations.

## Current Behavior (2026-06-08, v1.3.0+)

`build_reasonix_cmd()` in `reasonix_kanban_interactive.py` now passes `--continue` by default (`args.continue_session` defaults to True). This lets Reasonix resume the most recent session, preserving conversation history across restarts.

### CLI Args

| Arg | Default | Effect |
|-----|---------|--------|
| `--continue` | True (default) | Resume most recent session |
| `--no-continue` | N/A | Force fresh session |

### Fallback Mechanism

If `--continue` fails (rc=1 — no compatible session exists, e.g. first launch), the launcher detects `rc=1 + "--continue" in cmd` and retries without `--continue`. This is a one-shot fallback (`_continue_fallback_used` flag prevents loops).

```python
# In launcher_main():
if rc == 1 and "--continue" in reasonix_cmd and not _continue_fallback_used:
    _continue_fallback_used = True
    reasonix_cmd_fallback = [a for a in reasonix_cmd if a != "--continue"]
    # Stop old watcher, restart with fallback cmd
    watcher = start_watcher(restart_reason="--continue fallback: watcher restart")
    reasonix_proc = subprocess.Popen(reasonix_cmd_fallback, cwd=str(workspace), env=env)
    continue
```

### Code Location

- `build_reasonix_cmd()`: line ~1708 in `reasonix_kanban_interactive.py`
- Fallback logic: line ~1880 in `reasonix_kanban_interactive.py`
- `--continue` / `--no-continue` argparse: line ~2003

## Implications for Workers

- Workers now have context from previous sessions — they remember what they've already tried, what the planner said, etc.
- This is especially useful for **critic**, which needs accumulated understanding of the project to review effectively.
- If a worker's session gets corrupted or stuck, restart with `--no-continue` to get a clean slate.
- The fallback mechanism means `--continue` never blocks startup — if it fails, the worker still starts with a fresh session.

## Known Issues

### Watcher lock race during fallback

The fallback restart may cause a brief watcher lock race: old watcher still holds `fcntl.flock()`, new watcher gets `rc=2` and refuses to start. This usually resolves in 2-3 seconds when the old watcher terminates. The fallback code does `watcher.terminate() + wait(timeout=5)` before restarting.

### `_process_is_launcher` NameError (stale process)

If the launcher process was started before a code update that added `_process_is_launcher()`, the watcher subprocess may crash with `NameError: name '_process_is_launcher' is not defined`. Fix: kill all kanban processes and restart (`pkill -f reasonix_kanban_interactive`).

See `upgrading-reasonix` skill 陷阱 18 for full diagnosis.

## Session File Format

| Version | Directory | Filename Pattern | `--continue` Compatible |
|---------|-----------|-----------------|------------------------|
| v0.x | `~/.reasonix/sessions/` | `code-<slug>-<ts>.json` | ❌ Format incompatible |
| v1.1.0+ | `~/.reasonix/sessions/` | `code-<slug>-<ts>.jsonl` | ✅ Native format |
| v1.1.0+ | `~/.reasonix/sessions/` | `code-<slug>-<ts>.events.jsonl` | ❌ Events log only |

Exclude `__archive_` and `.events.` files — they are not resumable main sessions.
