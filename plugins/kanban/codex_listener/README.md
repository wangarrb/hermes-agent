# Codex Kanban Listener

Canonical source for running Codex CLI as a Hermes Kanban profile.

This directory is the single source of truth.  The Codex plugin exports at
`~/.agents/plugins/codex-kanban` and `/home/wyr/plugins/codex-kanban` are
symlinks to this directory; do not fork or copy the implementation elsewhere.

The listener imports both:

```python
from hermes_cli import kanban_db
from hermes_cli import kanban_listener_policy
```

So state flow and timing policy are shared with Hermes `/listen-kanban`.

## Runtime commands

Default headless worker mode:

```bash
codex-kanban-listen --profile planner
deepseek-kanban-listen --profile implementer
```

This is the fallback mode for `start-kanban.sh`: a local listener polls Kanban
without model tokens while idle, claims one task at a time, runs `codex exec` or
`deepseek-tui exec`, and completes/blocks from the final structured result.

Interactive visible Codex mode:

```bash
codex-kanban-interactive --profile planner --board egomotion4d --workspace /home/wyr/code/Egomotion4D
```

The interactive bridge starts a real Codex TUI in the given workspace and runs a
small background watcher.  The watcher claims ready Kanban tasks assigned to the
profile, writes a task prompt file under `.codex-kanban/<board>/<profile>/`,
injects a short instruction into the Codex pane via Zellij, and heartbeats the
claim until Codex completes/blocks the task with `hermes kanban`.

Explicit self-poll visible mode:

```bash
codex-kanban-interactive --profile planner --board egomotion4d --workspace /home/wyr/code/Egomotion4D --task-delivery self-poll
deepseek-kanban-interactive --profile critic --board egomotion4d --workspace /home/wyr/code/Egomotion4D --task-delivery self-poll
```

Self-poll mode is now the default visible-pane mode.  It replaces per-task
`KANBAN_TASK_BOUNDARY` injection with a machine-readable claim interface.  The
bridge gives the TUI one startup prompt that tells it to run
`hermes kanban next --json`, read the emitted `context_path`, complete/block
the task, then poll again.  It writes task contexts under
`.hermes-kanban/<board>/<profile>/`.

The legacy per-task Zellij injection mode is still available with
`--task-delivery inject` for rollback/debugging.  Use `--task-delivery
worker` only when you deliberately want a headless listener and no visible TUI.

Control-plane commands:

```bash
hermes kanban --board <board> next --profile <profile> --claim-assignees <csv> --workspace <repo> --listener-kind codex-self-poll --owner <pane-owner> --json
hermes kanban --board <board> reset-current --profile <profile> --claim-assignees <csv> --workspace <repo> --listener-kind codex-self-poll --owner <pane-owner> --json

Interactive wrappers also support a local reset shortcut:

```bash
codex-kanban-interactive --profile planner --board egomotion4d --workspace /home/wyr/code/Egomotion4D --reset-kanban
deepseek-kanban-interactive --profile critic --board egomotion4d --workspace /home/wyr/code/Egomotion4D --reset-kanban
```
```

The `--owner` value is pane/process scoped, not just role scoped.  This prevents
two panes with the same profile or assist assignees from adopting each other's
current running claim.  `/reset-kanban` and `/listen-kanban --reset` remain the
Hermes CLI reset names; self-poll workers use `kanban reset-current`.

Wrappers are installed as:

```text
/home/wyr/.local/bin/codex-kanban-listen -> /home/wyr/.hermes/hermes-agent/plugins/kanban/codex_listener/bin/codex-kanban-listen
/home/wyr/.local/bin/codex-kanban-interactive -> /home/wyr/.hermes/hermes-agent/plugins/kanban/codex_listener/bin/codex-kanban-interactive
/home/wyr/.local/bin/deepseek-kanban-listen -> /home/wyr/.hermes/hermes-agent/plugins/kanban/deepseek_listener/bin/deepseek-kanban-listen
```

## Zellij integration

`~/.config/zellij/layouts/multi-agent.kdl` historically ran the planner pane as:

```bash
/home/wyr/.local/bin/codex-kanban-listen --profile planner
```

The canonical launcher `/home/wyr/bin/start-kanban.sh -b <board> -w <workspace>`
now defaults Codex/DeepSeek panes to visible self-poll TUI sessions:

```bash
/home/wyr/.local/bin/codex-kanban-interactive --profile planner --board <board> --workspace <repo> --task-delivery self-poll
/home/wyr/.local/bin/deepseek-kanban-interactive --profile implementer --board <board> --workspace <repo> --task-delivery self-poll
```

Headless listener delivery remains available with `--task-delivery worker`.

Backups:

```text
~/.config/zellij/layouts/multi-agent-planner-hermes-backup.kdl
~/.config/zellij/layouts/multi-agent-codex-interactive-backup.kdl
```

## Codex plugin export

Plugin manifest:

```text
.codex-plugin/plugin.json
```

Personal marketplace entry:

```text
/home/wyr/.agents/plugins/marketplace.json
/home/wyr/.agents/plugins/codex-kanban -> this directory
/home/wyr/plugins/codex-kanban -> this directory
```

Codex marketplace was registered with:

```bash
codex plugin marketplace add /home/wyr
```

Keep the marketplace path live-linked to this Hermes source.  Running
`codex plugin add codex-kanban@local-plugins` creates a Codex cache copy under
`~/.codex/plugins/cache/`; that cache is disposable and should not become the
maintained source.

## Verification

Syntax:

```bash
python3 -m py_compile codex_kanban_listener.py
```

Fake-Codex smoke completed successfully on board `codex-listener-smoke` task
`t_56bbb812`: ready -> claimed -> spawned -> heartbeat -> done.

A real-Codex smoke hit CCH 503 and was correctly blocked, proving provider
failures flow to kanban block instead of leaving tasks stuck running.
