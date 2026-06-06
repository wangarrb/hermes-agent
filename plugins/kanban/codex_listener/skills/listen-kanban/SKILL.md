---
name: listen-kanban
description: Use when Codex should act as a Hermes Kanban profile/worker. Provides `codex-kanban-interactive` for visible self-poll mode and `codex-kanban-listen` for headless exec mode.
---

# Codex Hermes Kanban Listener

This plugin exposes Hermes Kanban tasks to Codex CLI without duplicating Hermes Kanban logic.

Canonical source lives in:

`/home/wyr/.hermes/hermes-agent/plugins/kanban/codex_listener/`

Use the headless worker wrappers only when no visible TUI is needed:

```bash
codex-kanban-listen --profile planner
deepseek-kanban-listen --profile implementer
```

Optional flags:

```bash
codex-kanban-listen --profile planner --board egomotion4d
codex-kanban-listen --profile planner --once
codex-kanban-listen --profile planner --sandbox read-only
codex-kanban-listen --profile planner --model gpt-5.5
codex-kanban-listen --profile planner --poll 6   # explicit override; default uses shared Hermes policy
```

Default visible self-poll mode:

```bash
codex-kanban-interactive --profile planner --board egomotion4d --workspace /home/wyr/code/Egomotion4D --task-delivery self-poll
```

Self-poll mode is the default for visible Codex/DeepSeek Kanban panes. It
starts the TUI with one startup prompt, then the agent must call:

```bash
hermes kanban --board <board> next --profile <profile> --claim-assignees <csv> --workspace <repo> --listener-kind codex-self-poll --owner <pane-owner> --json
```

The JSON response contains `context_path`, `complete_command`,
`block_command`, and `heartbeat_command`.  The worker must read `context_path`,
finish or block the task, then poll again.  To abandon only the current
self-poll claim, use `hermes kanban ... reset-current ... --json`; do not use a
generic `/reset` command name.

For operator-facing reset, the interactive wrappers expose:

```bash
codex-kanban-interactive --profile <profile> --board <board> --workspace <repo> --reset-kanban
deepseek-kanban-interactive --profile <profile> --board <board> --workspace <repo> --reset-kanban
```

Use `--task-delivery inject` only when deliberately falling back to the legacy
per-task Zellij injection watcher.  New launcher sessions should use
`--task-delivery self-poll` by default; use `--task-delivery worker` only for
headless non-interactive workers.

Timing policy source:

```text
/home/wyr/.hermes/hermes-agent/hermes_cli/kanban_listener_policy.py
```

Behavior:

1. Polls the active Hermes Kanban board for `ready` tasks assigned to the selected profile.
2. Atomically claims the task using Hermes' own `hermes_cli.kanban_db` source.
3. Builds the same worker context used by Hermes workers: task body, parents, prior runs, comments.
4. Runs `codex exec` in the resolved task workspace.
5. Requires Codex's final answer to be structured JSON.
6. Writes `done` or `blocked` back to Hermes Kanban and records Codex log path in metadata.

Important:

- Do not call `hermes kanban complete` manually from inside Codex; the listener completes/blocks based on the final JSON.
- If creating follow-up tasks, use `hermes kanban --board <board> create ... --json` and include only real returned ids in `metadata.created_task_ids`.
- For visible multi-agent zellij mode, run this listener in the planner pane instead of interactive `codex`.
- For self-poll visible mode, the `--owner` in the startup prompt is pane/process scoped.  Do not replace it with only the role name, or two panes with the same claim assignees can share one running claim.
