# Interactive Listener — codewhale TUI Debugging Reference

> Session-specific detail from 2026-06-10 debugging of implementer/critic pane input difficulty.

## "Turn stalled" stdin buffer pollution

### Symptoms
- User clicks pane, types, but characters don't appear in codewhale input box
- codewhale shows "Turn stalled — no completion signal received. Please try again."
- watcher log shows repeated `claimed+injected` entries every 600s for the same task

### Root Cause
When codewhale TUI enters "Turn stalled" (API turn timeout), the interactive watcher
continues to inject kanban prompts via `zellij write-chars`. These writes go directly
into codewhale's stdin PTY buffer. Codewhale can't process them (stalled), so text piles
up. When user later types, their keyboard input is queued behind stale watcher injections.

**Critical distinction**: This is NOT launcher `print()` pollution. PTY slave writes
(launcher stdout → /dev/pts/N) only affect screen display. They do NOT echo back to
the read side (codewhale stdin). The stdin pollution comes from `zellij write-chars`
which writes through the PTY master → slave → codewhale stdin path.

### Fix
`"turn stalled"` added to `_DEEPSEEK_BUSY_MARKERS` in `deepseek_kanban_interactive.py`:

```python
_DEEPSEEK_BUSY_MARKERS = (
    ...
    "steering current turn",  # codewhale v0.8+ interactive confirmation menu
    "turn stalled",           # codewhale v0.8+ API turn timeout — pane cannot accept input
)
```

Effect: `_looks_like_idle_deepseek_pane()` returns False when stalled → watcher won't
inject or idle-reclaim. `_looks_like_busy_deepseek_pane()` returns True → enters
stalled_busy timer. After 600s, stalled_busy reclaim releases the task claim but
does NOT inject new prompts. Codewhale sits quietly waiting for manual intervention.

### Manual recovery
1. Click the pane
2. Press Tab to switch focus to the input box
3. Type a recovery command or just press Enter to dismiss stalled state
4. If completely stuck: `kill <codewhale-tui PID>` — launcher auto-restarts with fresh session

## launcher print() → PTY visual pollution

### Root Cause
Launcher process's stdout/stderr = `/dev/pts/N` (same as codewhale, sharing zellij pane PTY).
`print()` in launcher code writes to the pane display. This doesn't affect keyboard input,
but corrupts visual output — pushes idle marker off-screen, breaks watcher's idle detection
via `zellij action dump-screen`.

### Fix
All `print()` in `launcher_main()` and `start_watcher()` replaced with `log_line(log_path, ...)`.
`build_deepseek_cmd()` debug `print(stderr)` converted to `log_line()` with new `log_path` kwarg.

### Rule
Never use `print()` / `sys.stderr.write()` in launcher code running inside a zellij pane.
Always use `log_line()`. Only acceptable `print()` calls:
- Early-exit error paths (before codewhale starts, user needs to see the error)
- Standalone subcommands (`reset-kanban`) that don't share PTY with codewhale

## PTY Architecture (for future debugging)

```
zellij PTY master (kernel)
    ↕
/dev/pts/N (PTY slave)
    ├── codewhale stdin  ← reads keyboard input (from PTY master)
    ├── codewhale stdout → writes TUI display (to screen via PTY master)
    ├── launcher stdout  → writes to screen (visual only, NOT to codewhale stdin)
    └── launcher stderr  → writes to screen (visual only, NOT to codewhale stdin)

zellij write-chars (external action)
    → writes through PTY master → appears on /dev/pts/N slave read side
    → enters codewhale stdin ← THIS IS THE POLLUTION PATH
```

Key: PTY slave writes (launcher print/codewhale output) → screen only.
PTY master writes (zellij inject / keyboard) → enters stdin buffer.

## Files modified (2026-06-10)

- `/home/wyr/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py`
  - Line 363: `"turn stalled"` added to `_DEEPSEEK_BUSY_MARKERS`
  - Line 1522: `build_deepseek_cmd()` gained `log_path` kwarg
  - Lines 1535-1540: `print(stderr)` → `log_line(log_path, ...)`
  - Line 1708: `build_deepseek_cmd(args)` → `build_deepseek_cmd(args, log_path=log_path)`
  - Lines 1659-1704: all `print()` in launcher_main/start_watcher → `log_line()`