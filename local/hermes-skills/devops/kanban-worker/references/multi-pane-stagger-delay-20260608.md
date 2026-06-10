# Multi-pane Startup Stagger Delay (2026-06-08)

## Problem

When `start-kanban.sh` launches 4 panes simultaneously via zellij layout, all panes start their interactive listener (watcher) at the same time. This causes DB lock conflicts and claim races — multiple panes may try to claim the same task in the same polling cycle.

## Fix

`start-kanban.sh` (`/home/wyr/bin/start-kanban.sh`) now adds stagger delays via `sleep N &&` prefix in `build_role_command()`:

- coordinator: `sleep 0 &&` (no delay)
- planner: `sleep 0.5 &&` (500ms)
- implementer: `sleep 1.0 &&` (1s)
- critic: `sleep 1.5 &&` (1.5s)

The stagger_s variable is set in `build_role_command()` based on role name, and prepended to all agent commands (hermes, codex, deepseek-tui, deepseek-reasonix).

## Verification

```bash
bash /home/wyr/bin/start-kanban.sh -b egomotion4d -n 2>&1 | grep "sleep"
```

Should show `sleep 0`, `sleep 0.5`, `sleep 1.0`, `sleep 1.5` in the 4 pane commands.

## Diagnostic

If you see repeated DB lock errors or double-claims right after kanban startup, check that the stagger delays are still in place. The `-n` (dry-run) flag prints the generated zellij layout without starting anything.