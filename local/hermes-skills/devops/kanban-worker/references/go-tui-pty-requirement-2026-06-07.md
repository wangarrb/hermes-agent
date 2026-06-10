# Go TUI PTY Requirement — Diagnostic Session 2026-06-07

## Context

All kanban panes (reasonix implementer, reasonix critic, codex planner) exited
simultaneously around 13:26-13:35. The zellij session `kanban-egomotion4d` was
entirely gone — no panes, no session.

## Diagnostic timeline

1. **zellij session exists but panes are empty** — `zellij list-sessions` showed
   `kanban-egomotion4d` but `dump-screen` returned empty content for all panes.

2. **No reasonix or codex processes running** — `ps aux | grep -E 'reasonix|codex'`
   returned nothing.

3. **Watcher logs tell the story**:
   - `reasonix-interactive-implementer.log`: 13:26:15 `reasonix exited rc=0` →
     watcher stopped
   - `reasonix-interactive-critic.log`: 13:35:29 watcher stopped (no explicit
     reasonix exit log)
   - `codex-interactive-planner.log`: last entry was `sqlite3.OperationalError:
     disk I/O error` at `add_comment`

4. **Testing from Hermes agent terminal** — `bash -lc 'reasonix chat --dir /tmp'`
   failed with `bubbletea: error opening TTY`. `bash -lc 'deepseek-tui'` failed
   with `Failed to enable raw mode: No such device or address`.

5. **Python TTY check**:
   ```python
   import sys
   print(sys.stdin.isatty())  # False in Hermes agent terminal
   open("/dev/tty")           # OSError: No such device or address
   ```

6. **DB state**: `PRAGMA integrity_check` = `ok`, `journal_mode` = `wal`,
   114 tasks. No WAL or SHM sidecar files.

## Root cause

Two independent failures caused a cascade:

1. **reasonix exited rc=0** — likely due to idle timeout or SIGHUP. From the
   session log, the last activity was at 13:31:33. The process then exited
   normally (rc=0).

2. **codex watcher DB I/O error** — `add_comment` threw
   `sqlite3.OperationalError: disk I/O error`, crashing the codex watcher.

3. **Cascade** — each pane exit caused its `bash -lc` wrapper to exit, which
   closed the zellij pane. When all panes closed, zellij's `auto_close` deleted
   the session.

## Key finding: Go TUI tools need PTY

`reasonix` (Go/bubbletea) and `deepseek-tui` (Go/bubbletea) both require:
- `open("/dev/tty")` to succeed
- Terminal raw mode to be enabled

These fail immediately when:
- stdin is a pipe (Hermes agent terminal, cron, SSH without `-t`)
- No PTY is available in the process tree

**Implication**: You cannot launch or test Go TUI tools from Hermes agent's
`terminal()` tool. You must use a real terminal or a PTY wrapper like
`script -qc 'command' /dev/null`.

## Zellij auto_close behavior

When all panes exit (regardless of exit code), zellij with default settings
removes the session entirely. This makes a cascade look like "everything
crashed at once" when the reality is sequential pane exits.

The `auto_close` setting is in zellij config. It's useful for cleanup but
can mask the root cause of multi-pane failures.

## Prevention

1. The zellij layout (`kanban-launcher.kdl`) uses `bash -lc` which inherits
   the PTY from zellij — this is the correct approach for Go TUI tools.
2. Never test Go TUI tools from environments without a PTY.
3. When diagnosing "all panes died," check watcher logs for the FIRST pane
   to exit — the others may be cascade victims.
