# PTY Input Contamination (codewhale-tui / deepseek-tui)

> Added 2026-06-10 after two rounds of debugging "input difficulty" in zellij kanban panes.

The kanban watcher communicates with codewhale via zellij PTY. Two independent contamination pathways can cause **"input difficulty"** — the user types but characters are partially or fully lost.

## Pathway 1: Screen contamination (visual — idle marker pushed off-screen)

- **Cause**: `launcher_main()` and `build_deepseek_cmd()` used `print()` to stderr, which writes to the PTY slave (/dev/pts/N). This pushes codewhale's idle marker ("编写任务或使用 /。") off the visible screen tail.
- **Effect**: Watcher's `_looks_like_idle_deepseek_pane()` can't find idle marker → thinks pane is busy → never claims new tasks → permanent skip-claim loop.
- **Fix**: All `print()` in launcher replaced with `log_line()` writing to log file instead of PTY.
- **Key insight**: PTY slave write does NOT loop back to PTY slave read side — it only contaminates screen display, not keyboard input.

## Pathway 2: Stdin buffer accumulation (functional — user input blocked)

- **Cause**: Watcher calls `zellij_inject()` → `zellij action write-chars -p <pane> <text>` → writes to PTY master → enters codewhale stdin. When codewhale is "Turn stalled" (API turn timeout), it doesn't process stdin → inject text accumulates in buffer → user's keyboard input queues behind inject text → appears swallowed/delayed/lost.
- **Effect**: User selects pane with mouse, types, but characters don't appear or are severely delayed.
- **Fix**: Added `"turn stalled"` to `_DEEPSEEK_BUSY_MARKERS`. When detected, watcher treats pane as busy → won't inject → no stdin accumulation.
- **Key insight**: This is the **real** cause of "input difficulty", not the screen contamination from Pathway 1.

## Remaining known issue: fcitx5 + zellij IME compatibility

- User uses fcitx5 input method. In some cases, partial character loss occurs even when codewhale is idle and no watcher injection happens.
- Likely cause: fcitx5 preedit protocol not fully supported by zellij PTY multiplexer.
- Status: Unresolved. Workaround: type slowly or use English input in codewhale panes.

## Debugging checklist for "codewhale can't accept input / won't claim tasks"

1. Check watcher log for repeated inject attempts: `tail -50 <watcher-log>`
2. Check screen state: `zellij action dump-screen -p <pane> --full` — look for "Turn stalled" or missing idle marker
3. Check if stdin buffer is polluted: test with `zellij action write-chars -p <pane> 'test'` — if it doesn't appear, stdin is clogged
4. Kill codewhale process → launcher auto-restarts with clean stdin → verify recovery

## Code locations (deepseek_kanban_interactive.py)

| Item | Lines | Description |
|------|-------|-------------|
| `_DEEPSEEK_IDLE_MARKERS` | 337-350 | Idle detection strings |
| `_DEEPSEEK_BUSY_MARKERS` | 352-363 | Busy detection strings (includes "turn stalled") |
| `_DEEPSEEK_QUEUED_INPUT_MARKERS` | 366-370 | Queued input detection |
| `_auto_dismiss_steering()` | 381-420 | Auto-dismiss Steering menu with Tab+Enter |
| `_looks_like_idle_deepseek_pane()` | 461-471 | Checks has_idle AND NOT has_busy AND NOT has_queued |
| `_looks_like_busy_deepseek_pane()` | 482-487 | Checks any busy/queued marker |
| `_pane_can_accept_new_kanban_task()` | 491-494 | Returns `_looks_like_idle_deepseek_pane()` |
| `_observe_pane_progress()` | 534-586 | Stalled-busy timer logic |
| `claim_and_inject_one()` | 885-896 | Checks `_pane_can_accept_new_kanban_task()` before injection |
| Watcher main loop | 1171-1280 | Idle pane reclaim (600s) and stalled_busy reclaim |
| `build_deepseek_cmd()` | 1522-1535 | FIXED: added `log_path` param, print(stderr) → log_line() |
| `launcher_main()` + `start_watcher()` | 1659-1704 | FIXED: all print() → log_line() |
| `build_deepseek_cmd(args, log_path=log_path)` | 1708 | FIXED: passes log_path |

## PTY direction diagram

```
                    PTY master side              PTY slave side
                    ──────────────              ──────────────
User keyboard ──→  zellij PTY master  ──→  codewhale stdin (read)
                    (via terminal)          (codewhale reads from here)

Watcher inject ──→  zellij action       ──→  codewhale stdin (read)
  write-chars       write-chars              (same entry point as keyboard)

Launcher print ──→  /dev/pts/N (stderr) ──→  codewhale screen (display only)
  to stderr         PTY slave write          (does NOT loop back to stdin!)
```

Key: PTY slave write (launcher print/stderr) goes to screen output only. PTY master write (zellij write-chars, user keyboard) goes to codewhale stdin. These are independent pathways.
