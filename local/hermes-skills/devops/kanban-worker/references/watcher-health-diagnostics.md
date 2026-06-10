# Watcher Health Diagnostics

Quick reference for diagnosing kanban interactive pane health issues.

## Process tree model

Each pane has a 3-level process hierarchy:

```
launcher (kanban_interactive.py --profile X --auto-start)
├── watcher (kanban_interactive.py --watch-child --profile X ...)
└── TUI (reasonix chat / codex resume / deepseek-tui)
```

The launcher is the parent. It starts both the watcher and TUI as child processes.
If the watcher dies, the launcher should restart it (after Failure 10 fix).
If the launcher dies, everything dies — no auto-restart.

## Finding processes

```bash
# All launchers (--auto-start, NOT --watch-child)
ps aux | grep "kanban_interactive.py.*--auto-start" | grep -v grep | grep -v watch-child

# All watchers (--watch-child only)
ps aux | grep "kanban_interactive.*watch-child" | grep -v grep

# Children of a specific launcher
pgrep -P <launcher_pid> | xargs -I{} ps -p {} -o pid,cmd
```

For each launcher, there should be exactly 2 children: one TUI and one watcher.
If only 1 child (TUI, no watcher), the watcher died and wasn't restarted.

## Watcher log location

```
~/.hermes/kanban/boards/<board>/logs/<agent>-interactive-<profile>.log
```

Examples:
- `reasonix-interactive-implementer.log`
- `reasonix-interactive-critic.log`
- `codex-interactive-planner.log`
- `deepseek-interactive-implementer.log`

## Key log patterns

| Pattern | Meaning |
|---------|---------|
| `interactive watcher started profile=...` | Watcher just started |
| `claimed+injected t_xxx` | Task claimed and injected into pane |
| `skip claim: Reasonix pane still shows an active/pending Kanban prompt` | Pane screen has stale text (Failure 11) |
| `interactive watcher stopped` | Watcher exiting (normal or after signal) |
| `watcher exited cleanly rc=0 while Reasonix is still running; not restarting` | **BUG** — launcher won't restart (Failure 10, pre-fix) |
| `watcher exited rc=X and Reasonix is gone; not restarting` | Expected — TUI also dead, no point restarting |
| `DB error in active-task loop` | SQLite WAL/FD issue (Failure 7) |
| `idle pane reclaim` | Watcher detected idle pane and reclaimed task |
| `active task left running state: t_xxx status=done` | Task completed normally |

## Automatic watcher health check (coordinator /listen-kanban)

Implemented 2026-06-07. The coordinator's `/listen-kanban` loop runs `_check_watcher_health()` every 180 seconds during idle polling.

**How it works**:
1. Scans `/proc` for all `kanban_interactive` processes matching the board
2. Identifies launcher processes (`--auto-start` without `--watch-child`) and watcher processes (`--watch-child`)
3. Matches each to a profile via `--profile` CLI arg
4. Detects coordinator's hermes process separately (no launcher/watcher subprocess)
5. Reports anomalies: launcher alive but watcher missing

**Output in coordinator pane**:
- Healthy: `[kanban-listener] watcher health check OK (4 panes)`
- Anomaly: `[kanban-listener] WATCHER HEALTH CHECK: 1 pane(s) missing watcher subprocess!`
  - Followed by per-profile details and recovery command

**Configuration**: The 180s interval is hardcoded in `_listener_loop`. To change it, modify the `>= 180.0` threshold in `kanban_listener.py`. The check only runs when `state.assignee == "coordinator"`.

**Limitations**: Detection only — does NOT auto-restart dead watchers. Recovery requires manual intervention (restart the kanban session or the specific pane).

## Quick health check script (manual)

```bash
#!/bin/bash
# Check if all expected watchers are alive for a given board
BOARD="egomotion4d"

echo "=== Launchers ==="
ps aux | grep "kanban_interactive.py.*--auto-start" | grep -v grep | grep -v watch-child | \
  awk '{for(i=1;i<=NF;i++) if($i ~ /--profile/) print $2, $i}'

echo ""
echo "=== Watchers ==="
ps aux | grep "kanban_interactive.*watch-child" | grep -v grep | \
  awk '{for(i=1;i<=NF;i++) if($i ~ /--profile/) print $2, $i}'

echo ""
echo "=== Missing watchers (launcher has TUI but no watch-child) ==="
for launcher_pid in $(ps aux | grep "kanban_interactive.py.*--auto-start" | grep -v grep | grep -v watch-child | awk '{print $2}'); do
    children=$(pgrep -P $launcher_pid 2>/dev/null)
    has_watcher=false
    for cpid in $children; do
        cmd=$(ps -p $cpid -o cmd= 2>/dev/null)
        if echo "$cmd" | grep -q "watch-child"; then
            has_watcher=true
            break
        fi
    done
    if ! $has_watcher; then
        echo "  ⚠️ Launcher PID $launcher_pid has NO watch-child!"
    fi
done

echo ""
echo "=== Recent watcher log tails ==="
LOG_DIR="$HOME/.hermes/kanban/boards/$BOARD/logs"
if [ -d "$LOG_DIR" ]; then
    for log in "$LOG_DIR"/*interactive*.log; do
        echo "--- $(basename $log) ---"
        tail -3 "$log" 2>/dev/null
        echo ""
    done
fi
```

## Coordinator config for kanban

The coordinator's hermes profile (`~/.hermes/profiles/coordinator/config.yaml`) has:

```yaml
kanban:
  auto_listen: true
  current_board: egomotion4d
  dispatch_in_gateway: false
  dispatch_interval_seconds: 60
```

`auto_listen: true` enables the `/listen-kanban` listener (15s poll for tasks).
`dispatch_in_gateway: false` prevents the gateway from spawning headless workers.
`dispatch_interval_seconds: 60` is only used when `dispatch_in_gateway: true` (currently disabled).

The watcher health check runs inside the `/listen-kanban` loop at 180s intervals, independent of the dispatch_interval_seconds config.
