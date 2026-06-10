# Kanban CLI Syntax Quirks

Commands that have non-obvious argument handling. Check this before guessing flags.

## `hermes kanban unblock`
- Takes **positional task IDs** only: `hermes kanban --board <slug> unblock <task_id> [task_id2...]`
- Does NOT accept `--reason` or any other flags beyond the standard `--board`
- Contrast: `hermes kanban block <task_id> --reason "..."` DOES accept `--reason`

## `hermes kanban dispatch`
- `--dry-run --json` previews without spawning
- `--max N --json` limits spawned tasks to N (use `--max 1` for cautious watchdog dispatch)
- Returns JSON with `spawned`, `reclaimed`, `crashed`, `timed_out`, `promoted`, `auto_blocked`, `auto_archived` arrays

## `hermes kanban complete`
- `--summary "text"` for result summary
- No `--result` flag; the summary IS the result

## `hermes kanban show`
- Always returns full task detail including events, runs, comments
- No `--short` or `--json` flag; output is always JSON-formatted when called from CLI with `2>&1`

## `hermes kanban stats`
- Returns counts by status and by assignee
- No filtering flags; shows the whole board

## `hermes kanban list`
- `--status <status> --json` filters by status, returns array of task objects
- `--assignee <profile> --status <status>` further filters