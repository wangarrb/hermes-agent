# Reasonix v0.x → v1.x Migration: Kanban Impact

> Findings from evaluating reasonix 0.53.2 → 1.3.0 upgrade impact on listen-kanban.
> Last updated: 2026-06-08

## Version Info

| | v0.53.2 | v1.1.0+ (Go rewrite) |
|---|---|---|
| Runtime | Node.js | Go binary |
| Subcommand | `reasonix code <workspace>` | `reasonix chat --dir <workspace>` |
| Config | `~/.reasonix/config.json` | `reasonix.toml` (auto-migrated, old file kept) |
| Install | `npm i -g reasonix` | `npm i -g reasonix` (Go binary via platform-specific optional deps) |

## Breaking CLI Changes

| Flag / Arg | v0.53.2 | v1.1.0+ | Impact on Kanban |
|---|---|---|---|
| `reasonix code <dir>` | position arg | No `code` subcommand | **FATAL**: Must use `chat --dir <dir>` |
| `--system-append-file <path>` | injects role instructions | Removed | **FATAL**: Must use `system_prompt_file` in `reasonix.toml` |
| `--no-mouse` | disables mouse in TUI | Removed | LOW: `--yolo` mode handles this |
| `--no-dashboard` | disables dashboard | Removed | LOW: Cosmetic only |
| `--new` | force new session | Removed | MEDIUM: New session is default |
| `--continue` | resume recent session | still supported | OK |
| `--model <id>` | provider name | still supported | OK |
| `--yolo` | N/A | auto-approve all tools | NEW: replaces `--no-mouse --no-dashboard` |
| `--dir <path>` | N/A | set project root | NEW: replaces position arg |
| `--max-steps N` | N/A | max tool-call rounds | NEW: 0=unlimited, default=6 |

## System Prompt Injection: v1.x Mechanism

v1.1.0+ removes `--system-append-file` CLI flag. Replacement is `system_prompt_file` in `reasonix.toml`:

```toml
# reasonix.toml (project-level)
system_prompt_file = ".reasonix-kanban/<board>/<role>/role-instructions.md"

[agent]
max_steps = 0       # 0 = unlimited; default is 6 which is too low for kanban tasks
auto_plan = "off"   # prevent auto plan-mode popup that stalls kanban workers
```

Config resolution order: `flag > ./reasonix.toml > ~/.config/reasonix/config.toml > built-in defaults`

### Auto-generated reasonix.toml

The kanban listener's `_write_role_instructions()` writes `reasonix.toml` in the workspace root each time a pane starts. It includes `system_prompt_file`, `max_steps = 0`, and `auto_plan = "off"`. This is necessary because:

1. **`max_steps` default is 6** — Reasonix's built-in default is 6 tool-call rounds, which is far too low for kanban tasks (typically 10-30 rounds). Without explicitly setting `max_steps = 0`, workers pause at 6 rounds waiting for manual continuation.
2. **`auto_plan = "off"`** — With `auto_plan = "ask"` (global default), Reasonix may prompt to enter plan mode for complex tasks. In YOLO/kanban mode this creates a stall. Setting it to `"off"` prevents this.

### Race condition: shared reasonix.toml across panes

**All panes share the same workspace and therefore the same `reasonix.toml`.** The last pane to call `_write_role_instructions()` wins — its `system_prompt_file` overwrites the others. In practice this is mitigated because:

1. Reasonix reads `reasonix.toml` only at startup and caches the config
2. Panes start with a few seconds of stagger (layout KDL runs them sequentially)
3. If panes start simultaneously, the wrong role instructions may be loaded

**Known issue, low priority.** If it causes real problems, the fix would be per-pane config directories (e.g., `--dir workspace/.reasonix-kanban/<role>/` with symlinked source), but this is complex and the current stagger-start works.

## Kanban DB Journal Mode

Reasonix kanban workers (3 watchers + gateway) cause concurrent writes to `kanban.db`. Under WAL journal mode (the default), this causes repeated "database disk image is malformed" corruption.

**Fix**: Force DELETE journal mode via `HERMES_KANBAN_FORCE_DELETE_JOURNAL` env var (default "1") in `kanban_db.py:connect()`. DELETE mode serializes writes via rollback journal — slower but far more robust for kanban's lightweight write workload.

**Critical**: Stale watcher processes holding WAL-mode connections block `PRAGMA journal_mode=DELETE` from succeeding. Kill all stale watchers before switching modes.

## Migration Checklist (COMPLETED)

- [x] Update `build_reasonix_cmd()` in `reasonix_kanban_interactive.py`
- [x] Replace `reasonix code <dir>` → `reasonix chat --dir <dir>`
- [x] Replace `--system-append-file <path>` → generate `reasonix.toml` with `system_prompt_file`
- [x] Remove `--no-mouse` and `--no-dashboard` flags
- [x] Add `--yolo` for kanban auto-approve mode
- [x] Handle `--new` removal (v1.x creates new session by default)
- [x] Add `max_steps = 0` to auto-generated `reasonix.toml`
- [x] Add `auto_plan = "off"` to auto-generated `reasonix.toml`
- [x] Force DELETE journal mode for kanban DB
- [x] Fix watcher restart on rc=0 exit (Bug1: `_should_restart_watcher`)
- [x] Fix stale KANBAN_TASK_BOUNDARY markers blocking claim (Bug2: `_pane_can_accept_new_kanban_task`)

## Watcher Bug Fixes (2026-06-08)

### Bug1: Watcher not restarted on rc=0 exit

`_should_restart_watcher(returncode)` previously only restarted on non-zero exit codes. But watchers can exit cleanly (rc=0) due to DB errors, yet the Reasonix TUI is still alive and needs the watcher. Fix: restart on ANY exit code when Reasonix is alive.

### Bug2: Stale Kanban injection markers blocking claim

`_pane_can_accept_new_kanban_task()` detected idle markers but also saw stale `KANBAN_TASK_BOUNDARY` text from previous injections, treating them as "busy". Fix: when idle markers are present, only truly active markers (processing/running) block claims; stale injection markers are ignored.

## Prompt Architecture (Three-Tier)

1. **`role-instructions.md`** (persistent via `system_prompt_file`): General rules + role definition + complete/block command templates. Loaded once at Reasonix startup. ~600-800 bytes per role.
2. **zellij `inject_text`**: Task-specific info only — `KANBAN_TASK_BOUNDARY` + task ID/title + role + file path + complete/block command. ~120 chars per task.
3. **Prompt file** (`<workspace>/.reasonix-kanban/<board>/<role>/<task>.md`): Task context body. ~500 chars.

Total per-task injection: ~620 chars (was ~2900 chars before optimization).

## Rollback

```bash
npm i -g reasonix@0.53.2
# Auto-migration does NOT delete old config.json, so rollback is clean
```
