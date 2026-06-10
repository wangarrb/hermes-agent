---
name: kanban-orchestrator
description: Decomposition playbook + specialist-roster conventions + anti-temptation rules for an orchestrator profile routing work through Kanban. The "don't do the work yourself" rule and the basic lifecycle are auto-injected into every kanban worker's system prompt; this skill is the deeper playbook when you're specifically playing the orchestrator role.
version: 2.0.0
metadata:
  hermes:
    tags: [kanban, multi-agent, orchestration, routing]
    related_skills: [kanban-worker]
---

# Kanban Orchestrator — Decomposition Playbook

> The **core worker lifecycle** (including the `kanban_create` fan-out pattern and the "decompose, don't execute" rule) is auto-injected into every kanban process via the `KANBAN_GUIDANCE` system-prompt block. This skill is the deeper playbook when you're an orchestrator profile whose whole job is routing.

## When to use the board (vs. just doing the work)

Create Kanban tasks when any of these are true:

1. **Multiple specialists are needed.** Research + analysis + writing is three profiles.
2. **The work should survive a crash or restart.** Long-running, recurring, or important.
3. **The user might want to interject.** Human-in-the-loop at any step.
4. **Multiple subtasks can run in parallel.** Fan-out for speed.
5. **Review / iteration is expected.** A reviewer profile loops on drafter output.
6. **The audit trail matters.** Board rows persist in SQLite forever.

If *none* of those apply — it's a small one-shot reasoning task — use `delegate_task` instead or answer the user directly.

## The anti-temptation rules

Your job description says "route, don't execute." The rules that enforce that:

- **Do not execute the work yourself.** Your restricted toolset usually doesn't even include terminal/file/code/web for implementation. If you find yourself "just fixing this quickly" — stop and create a task for the right specialist.
- **For any concrete task, create a Kanban task and assign it.** Every single time.
- **If no specialist fits, ask the user which profile to create.** Do not default to doing it yourself under "close enough."
- **Decompose, route, and summarize — that's the whole job.**
- **UPDATE, don't re-create.** If a task's body/title/priority needs fixing after creation, use `kanban update <task_id> --body "..."` (or `--title`, `--priority`, `--assignee`). Do NOT create a duplicate task and archive the original. The only exception is when the task has already been claimed/started by a worker — in that case, block it first, then create a replacement with a clear `supersedes` comment.
- **Use `--idempotency-key` on `kanban create`.** Every `kanban create` in batch scripts or wave-publishing code must include `--idempotency-key <wave-prefix>-<task-slug>` (e.g. `SSP4-B`). If the script retries after a failure, the same key returns the existing task instead of creating a duplicate.

## The standard specialist roster (convention)

Unless the user's setup has customized profiles, assume these exist. Adjust to whatever the user actually has — ask if you're unsure.

| Profile | Does | Typical workspace |
|---|---|---|
| `researcher` | Reads sources, gathers facts, writes findings | `scratch` |
| `analyst` | Synthesizes, ranks, de-dupes. Consumes multiple `researcher` outputs | `scratch` |
| `writer` | Drafts prose in the user's voice | `scratch` or `dir:` into their Obsidian vault |
| `reviewer` | Reads output, leaves findings, gates approval | `scratch` |
| `backend-eng` | Writes server-side code | `worktree` |
| `frontend-eng` | Writes client-side code | `worktree` |
| `ops` | Runs scripts, manages services, handles deployments | `dir:` into ops scripts repo |
| `pm` | Writes specs, acceptance criteria | `scratch` |

**Same-profile concurrent workers:** The assignee label is just a profile name. The dispatcher can spawn multiple concurrent workers from the same profile (each in an independent Hermes session). You don't need 8 separate profiles for the 8 roles above — a lean setup with 2-3 profiles works for a full kanban pipeline, as long as task workspaces don't conflict (default `scratch` workspaces are isolated per task). Only create a new profile when you need different credentials, model, or skills for that role.

## Decomposition playbook

### Step 1 — Understand the goal

Ask clarifying questions if the goal is ambiguous. Cheap to ask; expensive to spawn the wrong fleet.

### Step 2 — Sketch the task graph

Before creating anything, draft the graph out loud (in your response to the user). Example for "Analyze whether we should migrate to Postgres":

```
T1  researcher        research: Postgres cost vs current
T2  researcher        research: Postgres performance vs current
T3  analyst           synthesize migration recommendation       parents: T1, T2
T4  writer            draft decision memo                       parents: T3
```

Show this to the user. Let them correct it before you create anything.

### Step 3 — Create tasks and link

```python
t1 = kanban_create(
    title="research: Postgres cost vs current",
    assignee="researcher",
    body="Compare estimated infrastructure costs, migration costs, and ongoing ops costs over a 3-year window. Sources: AWS/GCP pricing, team time estimates, current Postgres bills from peers.",
    tenant=os.environ.get("HERMES_TENANT"),
)["task_id"]

t2 = kanban_create(
    title="research: Postgres performance vs current",
    assignee="researcher",
    body="Compare query latency, throughput, and scaling characteristics at our expected data volume (~500GB, 10k QPS peak). Sources: benchmark papers, public case studies, pgbench results if easy.",
)["task_id"]

t3 = kanban_create(
    title="synthesize migration recommendation",
    assignee="analyst",
    body="Read the findings from T1 (cost) and T2 (performance). Produce a 1-page recommendation with explicit trade-offs and a go/no-go call.",
    parents=[t1, t2],
)["task_id"]

t4 = kanban_create(
    title="draft decision memo",
    assignee="writer",
    body="Turn the analyst's recommendation into a 2-page memo for the CTO. Match the tone of previous decision memos in the team's knowledge base.",
    parents=[t3],
)["task_id"]
```

`parents=[...]` gates promotion — children stay in `todo` until every parent reaches `done`, then auto-promote to `ready`. No manual coordination needed; the dispatcher and dependency engine handle it.

### Step 4 — Complete your own task

If you were spawned as a task yourself (e.g. `planner` profile was assigned `T0: "investigate Postgres migration"`), mark it done with a summary of what you created:

```python
kanban_complete(
    summary="decomposed into T1-T4: 2 researchers parallel, 1 analyst on their outputs, 1 writer on the recommendation",
    metadata={
        "task_graph": {
            "T1": {"assignee": "researcher", "parents": []},
            "T2": {"assignee": "researcher", "parents": []},
            "T3": {"assignee": "analyst", "parents": ["T1", "T2"]},
            "T4": {"assignee": "writer", "parents": ["T3"]},
        },
    },
)
```

### Step 5 — Report back to the user

Tell them what you created in plain prose:

> I've queued 4 tasks:
> - **T1** (researcher): cost comparison
> - **T2** (researcher): performance comparison, in parallel with T1
> - **T3** (analyst): synthesizes T1 + T2 into a recommendation
> - **T4** (writer): turns T3 into a CTO memo
>
> The dispatcher will pick up T1 and T2 now. T3 starts when both finish. You'll get a gateway ping when T4 completes. Use the dashboard or `hermes kanban tail <id>` to follow along.

## Coordinator Agent Workflow (Visible Multi-Agent)

When the human operator wants to **see** what agents are doing and route work directly, use this alternative to auto-dispatch. The coordinator AI (you) stays in the main conversation with the human while routing tasks to visible worker sessions.

### Setting up multi-agent panes from scratch

**1. Create profiles for each role**

```bash
hermes profile create coordinator --no-skills   # saves ~10s of skill sync
hermes profile create planner --no-skills
hermes profile create implementer --no-skills
hermes profile create critic --no-skills
```

`--no-skills` avoids copying all bundled skills (which takes time and disk space). The profiles will inherit skills from the Hermes skill directory at runtime.

**2. Bootstrap config**

New profiles have minimal config. Copy the active profile's config as a starting point:

```bash
for p in coordinator planner implementer critic; do
    cp ~/.hermes/config.yaml ~/.hermes/profiles/$p/config.yaml
done
```

**3. Verify kanban toolset is available**

Check that the profile's `platform_toolsets.cli` includes `kanban`:

```bash
grep -A12 'platform_toolsets:' ~/.hermes/profiles/coordinator/config.yaml
```

Expected output shows `- kanban` in the list. If not present, add it. The kanban tools are in `_HERMES_CORE_TOOLS` and gated on the profile having `kanban` in its toolsets. When using CLI mode (`hermes -p <profile> chat`), `platform_toolsets.cli` determines which toolsets are active.

**4. Create a zellij layout**

`~/.config/zellij/layouts/multi-agent.kdl`:

```kdl
layout {
    pane split_direction="horizontal" {
        pane split_direction="vertical" {
            pane command="bash" cwd="/home/user/workspace" {
                args "-c" "hermes -p coordinator --continue chat"
            }
            pane command="bash" cwd="/home/user/workspace" {
                args "-c" "hermes -p planner --continue chat"
            }
        }
        pane split_direction="vertical" {
            pane command="bash" cwd="/home/user/workspace" {
                args "-c" "hermes -p implementer --continue chat"
            }
            pane command="bash" cwd="/home/user/workspace" {
                args "-c" "hermes -p critic --continue chat"
            }
        }
    }
}
```

Note: Use `--continue` (not `--resume <session-id>`) in layout commands. `--continue` automatically resumes the most recent session. `sessions list` does NOT support `--json` output, so don't try to script session ID discovery for resume. The `--continue` flag takes an optional session name argument, but omitting it also works — Hermes picks the most recent session automatically.

Note the KDL syntax: multiple `args` values, not a single quoted string.

**5. Set as default layout**

`~/.config/zellij/config.kdl`:

```kdl
default_layout "multi-agent"
```

Now every new zellij session opens with the 4-pane grid.

**6. Launch**\n\n```bash\nzellij  # picks up default_layout\n# Or use the packaged script for one-shot:\nbash ~/.hermes/skills/devops/kanban-orchestrator/scripts/start-kanban.sh\n```

Note: zellij requires a real TTY. Cannot be launched from a background terminal process. If you need to attach later: `zellij list-sessions` → `zellij attach <session>`.

**Codex as visible planner profile:** If the user wants Codex in the planner pane but still wants normal Kanban task triggering/flow, prefer the shared-source listener exported from Hermes to Codex instead of manual `tmux/zellij send-keys` injection. In this install, right-top planner runs `/home/wyr/.local/bin/codex-kanban-interactive --profile planner`; the canonical source is `/home/wyr/.hermes/hermes-agent/plugins/kanban/codex_listener/`. This keeps Hermes and Codex on one Kanban implementation while letting tasks assigned to `planner` flow through `ready -> claimed -> codex exec -> done/blocked` like ordinary profiles. Note: `codex-kanban-listen` (worker mode) is deprecated — only `codex-kanban-interactive` (inject mode) is used.

**DeepSeek-TUI as visible implementer profile:** The left-bottom implementer pane now runs DeepSeek-TUI instead of Hermes. The bridge is at `/home/wyr/.hermes/hermes-agent/plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py` with shell wrapper `/home/wyr/.local/bin/deepseek-kanban-interactive`. It follows the same claim/inject/heartbeat pattern as the Codex bridge: a background watcher claims `assignee=implementer,status=ready` tasks, writes a prompt file to `.deepseek-kanban/<board>/implementer/<task>.md`, and injects it into the DeepSeek pane via zellij write-chars. DeepSeek runs in YOLO mode by default (auto-approve all tool calls). To revert to Hermes implementer: `cp /home/wyr/bin/start-kanban.sh.bak.20260520 /home/wyr/bin/start-kanban.sh`.

**Canonical launch/stop scripts for this install:** The user normally starts visible Kanban with `/home/wyr/bin/start-kanban.sh -b <board>` (for example `-b egomotion4d`), not by directly opening `multi-agent.kdl`. That script rewrites `~/.config/zellij/layouts/kanban-launcher.kdl` on each run, so if the visible pane layout is wrong, patch `/home/wyr/bin/start-kanban.sh`, not only `multi-agent.kdl`.

**⚠️ Codewhale provider/model override chain**: `start-kanban.sh` originally forced `--provider openrouter` on all codewhale panes, and if no `--deepseek-model` was passed, the bridge (`deepseek_kanban_interactive.py`) fell back to `default_model_for_provider("openrouter") → "deepseek-v4-flash"`. This **ignored codewhale's own config file** entirely. Fixed 2026-06-08:
- `start-kanban.sh` no longer passes `--provider` to the bridge. If no `--deepseek-provider`/`--deepseek-model` flags are given, no provider/model override happens.
- `bridge` (`normalize_provider` in `deepseek_kanban_interactive.py`) returns `("", "")` on `None`/empty instead of falling back to `"openrouter"`.
- `build_env()` only sets `DEEPSEEK_PROVIDER`/`DEEPSEEK_MODEL`/base URL env vars when the provider or model was **explicitly** passed. If neither is given, codewhale reads its own `~/.config/codewhale/settings.toml` (or `~/.config/deepseek/settings.toml`).
- If you explicitly pass `--deepseek-provider openrouter` or `--deepseek-model deepseek-v4-pro`, the override still applies — useful for debugging or temporary config changes.
- **Troubleshooting**: If codewhale kanban panes use the wrong model, check whether `start-kanban.sh` is passing `--deepseek-provider` or `--deepseek-model` (grep the `codewhale)` branch in `build_role_command`). Also check `echo $DEEPSEEK_MODEL` and `echo $DEEPSEEK_PROVIDER` inside the zellij pane — if they're set, the override is active.

`start-kanban.sh` now keeps the four roles fixed while allowing the backend agent to be selected per role:
- `-o/--coordinator-agent <hermes|codex|deepseek-tui|deepseek-reasonix>`
- `-p/--planner-agent <hermes|codex|deepseek-tui|deepseek-reasonix>`
- `-i/--implementer-agent <hermes|codex|deepseek-tui|deepseek-reasonix>`
- `-c/--critic-agent <hermes|codex|deepseek-tui|deepseek-reasonix>`

Defaults remain: coordinator=Hermes, planner=Codex interactive, implementer=DeepSeek-TUI interactive, critic=Hermes. Role prompts are passed as `--profile <role>` and must stay role-bound rather than agent-type-bound. Use `start-kanban.sh -b <board> -n` to verify the generated layout without starting/stopping Zellij. The matching shutdown script is `/home/wyr/bin/stop-kanban.sh`; use `stop-kanban.sh -n` to dry-run before stopping.

**Task delivery**: Only `inject` mode is supported. Self-poll and worker modes were removed on 2026-06-06. The `--task-delivery` flag in `start-kanban.sh` is accepted for backward compatibility but defaults to `inject` and forces all other values to `inject`. Do not pass `--task-delivery` to the Python interactive scripts — they no longer accept it.

**DeepSeek-Reasonix as alternative to DeepSeek-TUI (2026-05-29, updated 2026-06-05):** Reasonix (`npm install -g reasonix`) is a DeepSeek-native agent framework with Ink TUI. It can replace DeepSeek-TUI in any kanban pane via `-i deepseek-reasonix` or `-c deepseek-reasonix`. Key differences from DeepSeek-TUI:
- **v0.53.2 (Node.js)**: Launches via `reasonix code <dir>`, uses `--system-append-file <path>` for role injection, `--no-mouse --no-dashboard` for kanban mode, `--new` for new session
- **v1.1.0+ (Go rewrite, 2026-06-04)**: Launches via `reasonix chat --dir <dir>`, `--system-append-file` REMOVED (use `system_prompt_file` in `reasonix.toml`), `--no-mouse`/`--no-dashboard` REMOVED, `--new` REMOVED (new session is default; `--continue` resumes), adds `--yolo` for auto-approve
- **NO `--session` parameter** — reasonix auto-generates session names from the workspace directory name. It does NOT support `--session <name>` for per-role isolation. Passing `--session` causes `error: unknown option '--session'`.
- Model selection via `--model <id>` (e.g., `deepseek-v4-pro`); no provider abstraction needed
- Idle markers: "输入任何内容 · / 使用命令 · @ 引用文件" (zh) / "ask anything · slash for commands · at-sign for files" (en)
- Implementation: `/home/wyr/.hermes/hermes-agent/plugins/kanban/reasonix_listener/reasonix_kanban_interactive.py`, wrapper at `~/.local/bin/reasonix-kanban-interactive`
- Does NOT support `worker` task delivery mode — only `inject` (self-poll removed 2026-06-06)
- User preference: **always use `inject`** mode. This is now the default in `start-kanban.sh` (changed 2026-06-06). Do NOT pass `--task-delivery` to the Python interactive scripts — the flag was removed from all three.
- Example: `start-kanban.sh -b egomotion4d -i deepseek-reasonix` (no `--task-delivery` needed)
- **Pitfall**: All three kanban interactive scripts (codex/deepseek/reasonix) need `log_path.parent.mkdir(parents=True, exist_ok=True)` before `log_path.open()` — otherwise `FileNotFoundError` on new boards where the `logs/` directory doesn't exist yet.
- **Reasonix MCP configuration (v0.x)**: To add MCP servers (like jcodemunch) to reasonix, edit `~/.reasonix/config.json` field `"mcp"`. The field accepts a **string array**, NOT object format. Correct: `["jcodemunch-mcp serve"]`. Incorrect: `[{"name": "jcodemunch", "command": "jcodemunch-mcp", "args": ["serve"]}]` — reasonix will silently drop non-string items with `config "mcp" had N non-string item(s) — dropped`. After updating config, restart the reasonix TUI for MCP tools to load.
- **Reasonix MCP configuration (v1.x)**: Config migrated to `reasonix.toml` format. Auto-migration reads v0.x `config.json` and writes to TOML on first run. Old file preserved.
- **⚠️ v1.x UPGRADE BREAKS LISTEN-KANBAN**: The `reasonix_kanban_interactive.py` plugin currently uses `reasonix code <dir>` + `--system-append-file <path>` + `--no-mouse` + `--no-dashboard` + `--new`, ALL of which are removed in v1.1.0. The plugin MUST be updated before upgrading. See `references/reasonix-v1-migration-kanban-impact.md` for the full migration checklist and replacement mechanisms.

**Zellij session lifecycle critical pitfall**: `zellij kill-session` only kills processes — the session becomes EXITED (dead) but the name stays occupied. Running `zellij --session <name> --new-session-with-layout` after that fails with "Session with name X already exists, but is dead". Both `start-kanban.sh` and `stop-kanban.sh` must call `zellij delete-session <name>` after `kill-session` to fully remove the session record and allow same-name recreation. If you encounter dead sessions blocking startup, clean up with: `zellij list-sessions --no-formatting --short | while read s; do zellij delete-session "$s"; done`.

**Zellij auto_close cascade**: When all panes exit (even normally with rc=0), zellij's default `auto_close` deletes the session entirely. This makes a single pane failure look like total system failure. When diagnosing "session disappeared," check watcher logs for the FIRST pane to exit — the others are likely cascade victims. See `kanban-worker` skill Failure 9 for the full diagnostic.

**⚠️ Reasonix v1.x `max_steps` default is 6**: The built-in default for `agent.max_steps` is 6 tool-call rounds — far too low for kanban tasks that need 10-30 rounds (search code + remote execution + verification). The auto-generated `reasonix.toml` sets `max_steps = 0` (unlimited), but if this file is missing or overwritten, Reasonix silently falls back to 6 and workers pause mid-task with "paused after 6 tool-call rounds (agent.max_steps)". Always verify `reasonix.toml` has `[agent] max_steps = 0` after restarting kanban sessions.

**⚠️ Watcher idle-reclaim loop (fixed 2026-06-08)**: The watcher's idle detection used to only check screen content, ignoring that Reasonix session files were still being updated (proving the agent was working). This caused repeated 600s reclaims that interrupted API waits. The fix adds a session-liveness gate: if the session `.jsonl` file was modified within `idle_pane_reclaim_s`, the idle timer resets. Additionally, `_sessions_latest_mtime()` was scanning the wrong directory (`~/.reasonix/sessions/*.json` instead of `~/.config/reasonix/sessions/*.jsonl`), making the liveness check completely ineffective. See `references/kanban-watcher-idle-and-lock.md` for the full diagnosis and fix details.

**⚠️ CodeWhale v0.8.53+ "active ctx" false busy marker (fixed 2026-06-09)**: CodeWhale v0.8.53 changed the status bar format — even at idle, the bottom line shows `编写任务或使用 /。` plus a status line like `Ctrl+O Activity: thinking  active ctx 5%  ...`. The `active ctx` substring hit the watcher's `_DEEPSEEK_BUSY_MARKERS` list, so `_looks_like_idle_deepseek_pane()` always returned False and the watcher permanently printed `skip claim: DeepSeek pane still shows an active/pending Kanban prompt`. **Fix**: Removed `" active ctx"` from `_DEEPSEEK_BUSY_MARKERS` in `deepseek_kanban_interactive.py`. The `" active ·"` marker remains and correctly matches genuine busy state. `active ctx` is only a context-usage indicator (e.g. "5% of context used"), not an activity indicator. **Diagnostic**: If a watcher log shows continuous `skip claim` lines for a pane that is visibly idle, dump the pane screen (`zellij --session <s> action dump-screen --pane-id <id> --full | tail -5`) and check which BUSY markers match the tail text. **After editing markers, restart the watcher process** — the launcher and watcher must be killed and relaunched for the Python module change to take effect.

**⚠️ Watcher silent-fail loop when no ready tasks (Bug 4)**: When `claim_and_inject_one()` finds no ready tasks (e.g., because the task's parent is still `running`), it returns `(None, None)` without logging. The watcher sleeps 60s and loops silently, appearing dead (no log entries for 20+ min). Most common cause: dependency-gated tasks where a `todo` child has a `running` parent and cannot become `ready`. Fix: add `log_line()` at silent return paths. See `references/kanban-watcher-idle-and-lock.md` Bug 4.

**⚠️ Launcher print() PTY contamination (Bug 5, fixed 2026-06-09/10)**: Launcher `print()` calls wrote to the PTY master, overwriting codewhale's idle marker on screen, causing permanent skip-claim. Fix: replaced all `print()` with `log_line()`. See `references/kanban-watcher-idle-and-lock.md` Bug 5.

**⚠️ "turn stalled" not recognized as busy (Bug 6, fixed 2026-06-09)**: When codewhale entered "Turn stalled", the watcher injected kanban text into stdin, corrupting the input buffer. Fix: added `"turn stalled"` to `_DEEPSEEK_BUSY_MARKERS`. See `references/kanban-watcher-idle-and-lock.md` Bug 6.

**Quick diagnostic: watcher appears dead** — check in order:
1. `ps aux | grep deepseek_kanban_interactive.*<profile>` — process alive?
2. `tail -20 <watcher-log>` — `skip claim` = Bug 3 (false busy); no entries 10+ min = Bug 4 (no ready tasks)
3. `hermes kanban --board <board> list` — find `todo` tasks with `running` parents = Bug 4 confirmed
4. `zellij dump-screen` — check for busy/idle marker mismatches

**⚠️ Double launcher prevention (added 2026-06-08)**: Running two launcher+watcher instances for the same (board, profile) causes task injection chaos. The watcher now uses `flock()` on `~/.hermes/kanban/boards/<board>/logs/watcher-<profile>.lock` to enforce single-instance per slot. If a second launcher starts, it prints the old PID and exits immediately. See `references/kanban-watcher-idle-and-lock.md` for design rationale and edge cases.

**⚠️ Reasonix v1.x `auto_plan` should be "off"**: The global default `auto_plan = "ask"` can prompt workers to enter plan mode for complex tasks, stalling kanban execution. The auto-generated `reasonix.toml` sets `auto_plan = "off"`, but same caveat as `max_steps` — if the file is missing or overwritten, workers may stall.

**⚠️ Shared `reasonix.toml` race condition**: All kanban panes share the same workspace and therefore the same `reasonix.toml`. The last pane to start writes its `system_prompt_file` path, overwriting the others. In practice this is mitigated because Reasonix reads `reasonix.toml` only at startup and caches the config, and panes start with a few seconds of stagger. If panes start simultaneously, the wrong role instructions may be loaded — verify by checking each pane's Reasonix status bar shows the correct role.

**⚠️ Kanban DB corruption under WAL mode**: With 3+ concurrent writers (watchers + gateway), WAL journal mode causes repeated "database disk image is malformed". Force DELETE journal mode via `HERMES_KANBAN_FORCE_DELETE_JOURNAL=1` (default on). When switching modes, kill ALL stale watcher processes first — they hold WAL-mode connections that block `PRAGMA journal_mode=DELETE`.

**Go TUI tools need PTY**: `reasonix` and `deepseek-tui` are Go/bubbletea binaries that require `open("/dev/tty")` and terminal raw mode. They crash immediately with `bubbletea: error opening TTY` when stdin is a pipe (Hermes agent terminal, cron, SSH without `-t`). You cannot launch or test them from Hermes agent's `terminal()` tool. Use `start-kanban.sh` from a real terminal, or wrap with `script -qc 'command' /dev/null` if you need a PTY wrapper.

**Kanban DB corruption kills all workers**: If every pane worker crashes within seconds of launch, check `file ~/.hermes/kanban/boards/<board>/kanban.db` — if it says `data` instead of `SQLite 3.x database`, the DB is corrupted. All workers crash at `sqlite3.DatabaseError: file is not a database` in `kanban_db.py:connect()`. Look for `kanban.db.repair-test-*` backups, validate with `file` + `sqlite3`, then replace the corrupted DB. See `kanban-worker` skill for the full recovery procedure.

**Codex planner idle behavior:** The right-top `planner-codex` pane is a listener, not an interactive Codex chat UI. When healthy but idle it only prints `codex-kanban listener started: profile=planner board=<board> poll=...`. It starts real `codex exec` output only after a `ready` task assigned to `planner` exists. Verification sequence before changing code: `zellij --session <session> action list-panes --all --json`, `zellij --session <session> action dump-screen -p terminal_<id> -f`, `ps ... codex_kanban_listener.py`, then `hermes kanban --board <board> list --assignee planner --status ready`.

**DeepSeek implementer idle behavior:** The left-bottom `implementer-deepseek-interactive` pane shows a "按 Enter 进入 interactive DeepSeek-TUI" prompt on startup. After pressing Enter, DeepSeek-TUI runs in YOLO mode. The background watcher claims `assignee=implementer,status=ready` tasks and injects them via zellij write-chars. DeepSeek itself must run `hermes kanban complete/block` to finalize tasks. Prompt files are stored in `<workspace>/.deepseek-kanban/<board>/implementer/`. Logs: `~/.hermes/hermes_use/kanban-logs/deepseek-interactive-implementer.log`.

**Team composition examples (real setups):**

| Pane | Role | What they do | Toolsets |
|:----:|------|-------------|:--------:|
| 0 | Coordinator (you/AI) | Decompose tasks, route, review, merge | kanban, skills, web |
| 1 | Planner | Research, design experiments, write specs | web, file, kanban |
| 2 | Implementer | Write code, run experiments, debug | terminal, file, kanban |
| 3 | Reviewer/Critic | Code review, validate conclusions, find flaws | web, file, terminal, kanban |

**User's actual 4-role setup (this install):** coordinator / planner / implementer / critic — these profiles already exist under `~/.hermes/profiles/`. Do NOT create new profiles (researcher/analyst/worker) without explicit user request. The zellij layout `multi-agent.kdl` launches these four directly. Current pane assignments: coordinator (Hermes), planner (Codex interactive + kanban watcher), implementer (DeepSeek-TUI interactive + kanban watcher), critic (Hermes).

The reviewer/critic is particularly valuable for catching "self-referential metric traps" (optimizing your own outputs without external ground truth). This role explicitly distrusts results until verified against independent GT.

**Pitfalls:**
- Don't create profiles with `hermes profile create --model ... --provider ...` — the CLI doesn't accept model/provider flags for `profile create`. Set model after creation via the config.yaml.
- **Don't create duplicate profiles for roles that already exist.** Check `ls ~/.hermes/profiles/` first. If coordinator/planner/implementer/critic already exist, use them — don't create researcher/analyst/worker variants without the user explicitly asking.
- Profiles created with `--no-skills` get a `.no-bundled-skills` sentinel file. Delete it to opt back into bundled skill delivery on next `hermes update`.
- **Critical: `.env` files are per-profile.** New profiles (even with `--no-skills`) start without a `.env` file. If you copy `config.yaml` from the active profile (which sets provider/model), the profile will fail with HTTP 401 because it has no API key. Fix: copy the main `.env` to each new profile: `cp ~/.hermes/.env ~/.hermes/profiles/<name>/.env`. The main `.env` contains all API keys; profiles without their own `.env` inherit from environment variables at shell launch, which may not include all keys.
- All profiles share the same API keys from `.env` unless you create per-profile `.env` files.
- **The dispatcher (`dispatch_in_gateway: true`) spawns headless workers for any task whose assignee matches a real profile name.** In manual pane mode, you want the dispatcher to **skip** your manually-run tasks — use an assignee label that doesn't match any profile, or disable the dispatcher if it's competing for tasks.
- **⚠️ Do NOT run `hermes kanban daemon` or `hermes kanban dispatch` when interactive listeners are active.** These are two competing task delivery mechanisms (see `kanban-worker` skill Failure 8). The user's workflow is: interactive listeners (start-kanban.sh → zellij panes → watcher injects prompts) ONLY. Dispatch daemon runs headless `hermes chat -q "work kanban task <id>"` that claims tasks but does NOT inject into visible panes — the user sees "status=running" but nothing happening in their TUI. If a task was stolen by dispatch, reclaim it: `hermes kanban --board <slug> reclaim <id> --reason "headless dispatch; need visible pane"` and the interactive listener will pick it up.
- Zellij pane titles default to the running command. Set meaningful pane names via `pane name="coordinator"` in the layout KDL for clarity.

### Interaction flow

**Setup:**
1. Create tmux/zellij panes (2×2 grid), each running `hermes -p <role> chat`
2. You (coordinator AI) talk to the human in pane 0
3. Kanban is used for **persistence only** — task definitions, structured outputs, handoff data
4. Enable mouse mode: zellij has `mouse_mode true` in config.kdl; tmux uses `tmux set -g mouse on`

**Flow:**

```
Human: @researcher 查一下XX技术的优劣
You (coordinator):
    kanban_create(assignee="researcher", title="...", body="...")  # persist task
    tmux send-keys -t researcher:0.1 '查一下XX技术的优劣' Enter      # dispatch to visible window

    # Researcher runs in its tmux pane — human can SEE real-time output
    # When done, you capture the result:
    result = $(tmux capture-pane -t researcher:0.1 -p | tail -30)

Human: 把结果给 analyst 分析
You (coordinator):
    kanban_create(assignee="analyst", body=<includes researcher output>, parents=[researcher-task-id])
    tmux send-keys -t analyst:0.2 '分析以下结果：<result>' Enter
```

**Key differences from auto-spawn:**

| Aspect | Auto-Spawn (Standard) | Coordinator Agent (This Mode) |
|--------|----------------------|-------------------------------|
| Worker execution | Headless `subprocess.Popen()` | tmux pane, human-visible |
| Dispatch trigger | Dispatcher 60s tick | `tmux send-keys` immediate |
| Routing | Assignee must match real profile | Any label works (dispatcher skips) |
| Info bridge | `parents[]` automatic in `build_worker_context()` | Coordinator manually captures+forwards |
| Human visibility | Logs / dashboard only | Live terminal output in adjacent pane |
| Result retrieval | `kanban show / log` | `tmux capture-pane` |

**Kanban still provides:**
- Task definitions and structured outputs survive tmux restart
- Parent-child dependencies for workflow structure
- `kanban_complete(summary=..., metadata=...)` preserves cross-task handoff
- Audit trail persists in SQLite

**Pitfalls:**
- **Coordinator must manually bridge info** — workers in tmux panes cannot read each other's output. You must capture-pane from worker A and forward to worker B.
- tmux session can be killed (`tmux kill-session`); kanban DB survives intact
- For large data transfer (>10 lines), write to `/tmp/kanban-share/` and tell the worker to read that file instead of pasting into send-keys
- Each tmux pane needs enough startup time (5-15s) for Hermes to boot before sending keys
- tmux is single-cursor — the human can only type in one pane at a time. Use mouse mode for quick switching.
- Hermes in tmux panes runs with full tool access; be careful not to send destructive commands blindly

**When to use which mode:**

| Situation | Use |
|-----------|-----|
| Routine batch processing (pure automation) | Auto-spawn via dispatcher |
| Exploratory research, debugging, learning | Coordinator + tmux (this mode) |
| Mixed — pipeline with human gates | Auto-spawn for routine steps, coordinator for review gates |
| Single person, no need for multi-role | Just talk to the coordinator AI directly — kanban/tmux unnecessary |

## Common patterns

**Fan-out + fan-in (research → synthesize):** N `researcher` tasks with no parents, one `analyst` task with all of them as parents.

**Pipeline with gates:** `pm → backend-eng → reviewer`. Each stage's `parents=[previous_task]`. Reviewer blocks or completes; if reviewer blocks, the operator unblocks with feedback and respawns.

**Same-profile queue:** 50 tasks, all assigned to `translator`, no dependencies between them. Dispatcher serializes — translator processes them in priority order, accumulating experience in their own memory.

**Human-in-the-loop:** Any task can `kanban_block()` to wait for input. Dispatcher respawns after `/unblock`. The comment thread carries the full context.

## listen-kanban Timing Configuration

Visible Kanban listeners share timing/retry policy through `hermes_cli/kanban_listener_policy.py`. Both Hermes `/listen-kanban` and the Codex-backed planner listener import this module, so changes apply consistently across visible panes.

| Parameter | Day (09:00~00:59) | Night (01:00~08:59) | Config location |
|-----------|:---:|:---:|:---:|
| Poll interval (find ready tasks) | 6s | 30s | `poll_seconds()` / `DAY_POLL_SECONDS` / `NIGHT_POLL_SECONDS` |
| Health check (claim audit, orphan reclaim) | 60s | 1800s | `health_check_seconds()` / `DAY_HEALTH_CHECK_SECONDS` / `NIGHT_HEALTH_CHECK_SECONDS` |
| Watcher health check (coordinator only) | 180s | 180s | `_check_watcher_health()` in `kanban_listener.py`, gated by `ListenerState._last_watcher_check` |
| Claim TTL | 3600s | 3600s | `LISTENER_HEALTH_CLAIM_TTL_SECONDS` |
| Provider retry cooldown | 600s (10min) | 600s (10min) | `RETRY_COOLDOWN_SECONDS` |

This user has a clear preference: fast daytime response, quota-saving night. Poll, health, TTL, and provider cooldown are independent values — no single constant drives them.

Operational nuance: provider/API failures such as 503/429/quota are not immediately blocked. Hermes and Codex listeners hold/heartbeat the claim during `RETRY_COOLDOWN_SECONDS`, then requeue; provider failures get at least `MIN_PROVIDER_FAILURE_SILENT_RETRIES` before surfacing as blocked.

To change timing, edit `hermes_cli/kanban_listener_policy.py` and restart the affected visible listener process/pane. The listener does not reload timing config at runtime.

## Kanban Watchdog Patrol (Cron-Driven Health Check)

When running as a scheduled cron job to monitor kanban board health (common for long-running multi-iteration research pipelines), follow this procedure:

### Step 1: Board stats and scope filter
```bash
hermes kanban --board <slug> stats
```
If all tasks are `done` and none are `ready/running/blocked`, the pipeline is either complete or stalled upstream. Check the latest final-review task (typically the planner review) for its `children` — if children exist, inspect them.

**Efficiency tip**: Run `stats` first, then `list --status running --json` and `list --status blocked --json` to get only the active anomalies. Then inspect those specific tasks with `show`. Only `show` the tracked task IDs from your cron config if `list` doesn't cover them. This avoids unnecessary `show` calls on the hundreds of `done` tasks.

**Efficiency tip — finding final-review tasks in a sea of done**: When you need to locate specific campaign tasks among hundreds of done entries, pipe `list --status done --json` through a Python one-liner that filters by title pattern:
```bash
hermes kanban --board <slug> list --status done --json | python3 -c "
import json, sys
tasks = json.load(sys.stdin)
for t in tasks:
    if 'W15-R' in t['title'] or 'W16-R' in t['title'] or ('final review' in t['title'].lower() and 'W1' in t['title']):
        print(f\"{t['task_id']}: {t['title']} (done, completed_at={t.get('completed_at')})\")
"
```
This avoids calling `show` on each of the 300+ done tasks individually.

**Campaign scope filter**: If your cron job monitors specific campaign prefixes (e.g., W13-*), filter the `list` output by title prefix to ignore other campaigns. You can do this with: `hermes kanban --board <slug> list --status running --json | python3 -c "import json,sys; [print(json.dumps(t)) for t in json.load(sys.stdin) if 'W13' in t.get('title','')]"`

### Step 2: Check critical task IDs
For each tracked task ID (provided in the cron job config), run:
```bash
hermes kanban --board <slug> show <task_id>
```
Focus on: status, assignee, completion time, result summary, children (next-wave tasks).

### Step 3: Detect anomalies
| Anomaly | Detection | Action |
|---------|-----------|--------|
| Ready task unclaimed >10 min | `status: ready`, no runs with recent heartbeats | Run `dispatch --dry-run --json` first, then `dispatch --max 1 --json` if sensible |
| Stale-lock reclaim loop | Task shows multiple consecutive runs all reclaimed via `stale_lock`, each lasting minutes/hours before the worker process dies or hangs | **For campaign-scoped tasks: Do NOT dispatch** — dispatch will just spawn another run that also fails. Root cause is worker runtime instability (listener/TUI process crash, Zellij pane conflict, environment issue). Flag for human intervention: the agent process itself needs debugging, not more task spawns. Add a comment noting the reclaim count and suggesting manual intervention or profile reconfiguration. **For non-campaign tasks**: just note them in your report as out-of-scope anomalies; do NOT spend dispatch cycles or create fix tasks for them — they belong to a different work stream that may have its own watchdog or manual operator. |
| Mass-stall: N tasks claimed by same listener PID, all zero heartbeats | Multiple `running` tasks share the same `claimed` lock PID, each with only 2 events (created + claimed), no heartbeats, and the listener process is still alive (`ps -p <pid>` shows it running) | **Do NOT dispatch more workers** — they'll also stall. The listener is alive but not processing (e.g., deepseek-listen stuck after claiming). Reclaim all stalled tasks first: `for tid in <ids>; do hermes kanban --board <slug> reclaim $tid --reason "watchdog: mass-stall, listener PID alive but zero heartbeats on N tasks"; done`. Then investigate the listener (check pane output, confirm TUI is stuck). After reclaiming, dispatch ONE task at a time and verify the next worker actually produces heartbeats before dispatching the rest. This pattern is distinct from a single stale heartbeat — it's a systemic listener stall affecting all its claimed tasks simultaneously. |
| Running task stalled >60 min | `status: running`, last heartbeat >60 min ago | Comment on task; reclaim if worker is dead |
| Critic skipped artifact check | Critic result doesn't mention validator output or artifact inspection | Add comment requesting Output Contract Self-Check; create fix+re-review task if needed |
| Blocked task with fixable cause | `status: blocked`, blocker is environmental/config | Create a fix task assigned to implementer, link as parent of blocked task |
| Blocked task whose dependency completed | `status: blocked`, but all parent tasks are now `done` and passed | Unblock via `hermes kanban --board <slug> unblock <task_id>` then dispatch; note: `unblock` takes positional task IDs only, no `--reason` flag |
| Stale blocked run after dependency resolves | Task shows `status: running` but its only run is in `blocked` state and the blocking condition has since resolved (parent critic completed, task was unblocked) | Run `hermes kanban --board <slug> dispatch --dry-run --json` — dispatch will reclaim the stale run and respawn the task. Do NOT just `unblock` — the run itself is blocked, not the task. `dispatch` handles the reclaim+respawn atomically. |
| Final review missing next-wave | Final review `done` but no children/published wave IDs | This breaks the iteration pipeline — flag as critical |
| Evidence-fail without fix task | Critic returns `CODE_PASS_EVIDENCE_FAIL` or `CODE_FAIL` but no fix task created | Create repair task linked from critic, re-review after fix |

### Step 4: Dispatch if needed
```bash
# Preview first
hermes kanban --board <slug> dispatch --dry-run --json
# Then dispatch one task at a time
hermes kanban --board <slug> dispatch --max 1 --json
```
Do NOT grab tasks assigned to profiles that aren't `ready`.

**Dispatch on already-running tasks**: `dispatch --dry-run` may list a currently `running` task in its `spawned` array. This happens when the task has a valid lock but dispatch wants to create an additional run. If the existing run is actually working (recent heartbeats), do NOT dispatch — you'll create a conflict. If the existing run is stale (no heartbeats for >30 min), dispatch will typically reclaim first, then spawn. The `reclaimed` count in the dispatch response tells you whether stale locks were cleaned up.

**Campaign-prefix filtering**: When monitoring a specific campaign (e.g., W13-W16), use task title prefixes to quickly filter out irrelevant running/blocked tasks from other campaigns (e.g., GFPR-*, GGPT-*). This avoids wasted `show` calls on tasks outside your scope.

### Step 5: Act on missing artifacts / incomplete reviews
- If a critic reviewed without inspecting output artifacts first, add a precise comment: "Critic must run validator on artifacts before code review. Re-review required with artifact-first order."
- If products are missing (e.g., required visuals/tables not in the output root), create a fix task with specific missing items.
- If a research task closed with NO_CLAIM but the no-claim analysis packet is insufficient (e.g., `failure_modes.md` is <200 bytes, `run_manifest.json` is <300 bytes), create a repair task.

### Step 6: Ensure pipeline continuity
If all critics for a wave have passed (even with NO_CLAIM verdicts) and the final planner review task is ready but not started, either:
- Flag it for the next planner dispatch cycle, or
- If you're acting as the coordinator/planner, claim and execute the review to prevent the pipeline from stalling.

### Output: [SILENT] when clean
If no anomalies found and all issues already have handling tasks, respond with exactly `[SILENT]` to suppress delivery. Do NOT combine `[SILENT]` with other content.

### Completed-campaign detection (when to go [SILENT])

When ALL tracked task IDs from the cron config are `done`, **trace forward through children/waves before declaring [SILENT]**. Cron configs often list only the first wave's task IDs (e.g., W13-A0 through W13-R). When those are all `done`, check the final review task's `children` to find next-wave task IDs (e.g., W14-*), then `show` those, and repeat until you reach the campaign's final review (e.g., W16-R). Only after confirming the entire pipeline from first wave to final review is `done` should you return `[SILENT]`. This avoids the false-negative where only the first wave's tasks are done but later waves are stalled.

Once confirmed complete, if the final review's children reference a different campaign prefix (e.g., W16-R children are `FPMR-*` or `GFPR-*`), the campaign has concluded and transitioned. The watchdog should return `[SILENT]` — this is a healthy terminal state, not a stall. The new campaign is outside the current watchdog's scope and will have its own cron config or manual kick-off.

Similarly, if `stats` shows `ready=0, running=0, blocked=0` and all tracked tasks are `done`, the board is idle between campaigns. Do NOT dispatch just because there are `todo` tasks from an unrelated campaign — those are waiting for their own promotion/dependency resolution.

### Key anti-patterns to watch for
- **Renaming a failed iteration as "not a real iteration"** — if a task ran and produced output (even NO_CLAIM), it counts as an iteration. The planner final review is the iteration boundary.
- **Stale blocked tasks** — a task may be `blocked` because a parent/dependency hadn't completed yet, but if that dependency has since completed (e.g., a critic task finished PASS), the blocked task should be unblocked immediately. Use `hermes kanban --board <slug> unblock <task_id>` (positional IDs only, no flags) then dispatch. This is especially common in multi-campaign boards where campaign transitions create blocked review tasks whose blocking condition resolves hours later.
- **Critic passing without artifact inspection** — the whole point of critic gates is catching bad evidence early. A critic that only reads code and skips output validation is a broken gate.
- **Final review completing without next wave** — this breaks the iteration cycle and leaves the pipeline dead.
- **Untracked scripts in done tasks** — if a task result claims "commit X pushed" but `git ls-tree X` doesn't contain the claimed script, the evidence provenance is broken. The critic should catch this; if not, create a fix task.
- **Completed task with empty result** — `result: null` or result preview `<empty>` on a `done` task means the implementer finished without recording what they did. The critic review should flag this and request evidence or a re-run. Do not let empty-result tasks pass through to final review.
- **Ready tasks with live listeners but no claims** — `needs-worker-health` events with `live_idle_unclaimed` indicate the listener process exists but isn't picking up ready tasks. Dispatch (`hermes kanban --board <slug> dispatch --max N --json`) usually unsticks this. Check for multiple ready tasks sharing the same assignee before dispatching — the dispatcher spawns one worker per ready task.

### Multi-campaign board handling
A single board often hosts multiple campaigns. When `stats` shows hundreds of `done` tasks and 0 `ready/running`:
- This is **normal and healthy** — old campaigns leave their completed tasks on the board permanently.
- **Only investigate** `ready/running/blocked` counts. If all are zero, the board is idle between campaigns.
- **Blocked tasks from inactive campaigns** (e.g., a GFPR-R blocked waiting for a different critic, or a user test task) should NOT be flagged as anomalies. They belong to separate work streams.
- To determine which campaign a blocked task belongs to, check its title prefix (e.g., `GFPR-*` ≠ `W1[3-6]-*`) and whether its parent tasks are from the active campaign.
- When transitioning between campaigns (e.g., W16-R → FPMR), verify the new campaign's first tasks are progressing within 10 minutes of being `ready`.

### Campaign boundaries and wave transitions

When a research campaign reaches its planned iteration limit (e.g., 6 of 6), the final planner review typically publishes a **new campaign** with different task structure. This is a wave transition, not a pipeline stall.

Signs of a healthy campaign boundary:
- The final review (e.g., `W16-R`) completes with `done` status and its `children` contain task IDs for the new campaign
- New campaign tasks appear in `todo`/`ready` with different naming conventions (e.g., `FPMR-A0` instead of `W17-A0`)
- The new campaign may have a different DAG shape (e.g., `A0→B0→C0/G0/V0→Critic→R` instead of `A0→A1→G1→C1→R`)
- Board stats show `done` count jumping and new `todo` tasks appearing

Watchdog actions at campaign boundaries:
- Do NOT flag the completed campaign's final review as "missing next wave" if it has published new campaign tasks as children
- Do verify the new campaign's first tasks are progressing (claimed/running within 10 min of ready)
- If the new campaign uses different output contract patterns (e.g., DepthEvidencePacket instead of source_gauge_manifest), the critic should adapt — do not force old campaign verification patterns onto new campaign tasks
- If the new campaign's initial gate task (A0) completes but no critic review is auto-created, create one via the coordinator-audit pattern or manually
- **Recognize multi-campaign boards**: When `stats` shows hundreds of `done` tasks, the board is healthy — just focus on `ready/running/blocked` counts. A board with `ready=0, running=0, blocked=0, todo>0` means tasks aren't being promoted; check if parent tasks are all `done` and children should auto-promote.

## Pitfall: Duplicate tasks from the same wave

Three root causes for "same task published multiple times":

### 1. Planner publishes, then immediately re-publishes a "stricter" replacement

The planner creates a task, re-reads the body 10-30 seconds later, decides it's not strict enough, and creates a second task with the same title but tighter acceptance criteria. The original gets archived or commented as "superseded."

**This is now a HARD RULE violation.** The anti-temptation rules explicitly require UPDATE over re-create.

**Prevention (mandatory):** Review all task bodies BEFORE calling `kanban_create`. If you must fix a body after creation, use `hermes kanban update <task_id> --body "..."` (or `--title`, `--priority`, `--assignee`) to modify the existing task — do NOT create a duplicate and archive the original. If the task has already been claimed/started by a worker, block it first, then create a replacement with a clear `supersedes` comment. Batch wave-publishing scripts must use `--idempotency-key` on every `kanban create` call.

### 2. Script retry after `kanban create` JSON field mismatch

If a batch script expects `task_id` in the `kanban create --json` response but the CLI returns `id` (or vice versa), the script gets a KeyError, retries, and creates the same task again. The `id`↔`task_id` alias was added in May 2026, but any script using other field names can hit the same pattern.

**Prevention:** Always capture and validate the response from `kanban_create` before proceeding. Use `idempotency_key` parameter (the DB has an `idempotency_key` column) to prevent duplicate creation on retry — same key returns the existing task instead of creating a new one.

### 3. DB index corruption revives old tasks

After a REINDEX or `.recover` operation, the DB can revert to an older timeline — new tasks disappear and old (superseded/completed) tasks come back alive. The planner sees the old tasks and re-creates the wave, producing duplicates when the DB eventually catches up.

**Prevention:** After any DB repair, verify the task count and latest task IDs match expectations before publishing new waves. Compare `SELECT count(*) FROM tasks` and `SELECT id FROM tasks ORDER BY created_at DESC LIMIT 5` before and after repair.

**Detection query** — find duplicate titles on the board:
```bash
sqlite3 ~/.hermes/kanban/boards/<board>/kanban.db \
  "SELECT title, COUNT(*) as cnt FROM tasks GROUP BY title HAVING cnt > 1 ORDER BY cnt DESC;"
```

## Pitfalls

**Reassignment vs. new task.** If a reviewer blocks with "needs changes," create a NEW task linked from the reviewer's task — don't re-run the same task with a stern look. The new task is assigned to the original implementer profile.

**Argument order for links.** `kanban_link(parent_id=..., child_id=...)` — parent first. Mixing them up demotes the wrong task to `todo`.

**Don't pre-create the whole graph if the shape depends on intermediate findings.** If T3's structure depends on what T1 and T2 find, let T3 exist as a "synthesize findings" task whose own first step is to read parent handoffs and plan the rest. Orchestrators can spawn orchestrators.

**Tenant inheritance.** If `HERMES_TENANT` is set in your env, pass `tenant=os.environ.get("HERMES_TENANT")` on every `kanban_create` call so child tasks stay in the same namespace.

## Recovering stuck workers

When a worker profile keeps crashing, hallucinating, or getting blocked by its own mistakes (usually: wrong model, missing skill, broken credential), the kanban dashboard flags the task with a ⚠ badge and opens a **Recovery** section in the drawer. Three primary actions:

1. **Reclaim** (or `hermes kanban reclaim <task_id>`) — abort the running worker immediately and reset the task to `ready`. The existing claim TTL is ~15 min; this is the fast path out.
2. **Reassign** (or `hermes kanban reassign <task_id> <new-profile> --reclaim`) — switch the task to a different profile and let the dispatcher pick it up with a fresh worker.
3. **Change profile model** — the dashboard prints a copy-paste hint for `hermes -p <profile> model` since profile config lives on disk; edit it in a terminal, then Reclaim to retry with the new model.

Hallucination warnings appear on tasks where a worker's `kanban_complete(created_cards=[...])` claim included card ids that don't exist or weren't created by the worker's profile (the gate blocks the completion), or where the free-form summary references `t_<hex>` ids that don't resolve (advisory prose scan, non-blocking). Both produce audit events that persist even after recovery actions — the trail stays for debugging.

## Architecture FAQ

For conceptual questions about kanban design — profile vs session identity, worker visibility, manual claiming (control-plane lanes), and role communication patterns — see `references/kanban-architecture.md`.

## Task Rejection / Return Operations

Users sometimes need to reject or return kanban tasks without deleting them. The board supports different operations depending on intent:

- **Block**: Task stays in the board with `status=blocked` and a human-readable reason. Listeners skip blocked tasks.
- **Reclaim**: Force-release a stale claim (even before TTL expiry) and reset to ready, available to the next worker.
- **Reassign**: Change the assignee profile, optionally reclaiming the current run first.
- **Complete with REJECT**: Mark as done with a summary/result that explains the rejection decision.

See `references/kanban-task-rejection-operations.md` for exact commands, decision table, and status check recipes.

For the **rework flow** (打回返工) — when a review fails and the task goes back to the original implementer for fixes — see `references/kanban-task-rework-flow.md`. This covers the `update --reopen` flow (done → ready with comment), mandatory comment format, reassign-to-different-worker, and iteration limits (max 3 reworks). This is the standard pattern for critic/planner review failures in Egomotion4D's visible multi-agent setup.

For CLI syntax quirks (e.g., `unblock` takes positional IDs only, no `--reason`), see `references/kanban-cli-syntax-quirks.md`.

**Note**: `hermes kanban update <task_id> --body/--title/--priority/--assignee` modifies a non-done, non-archived task in-place. For done tasks, add `--reopen` to transition the task back to `ready` and allow updates simultaneously. `--reopen` also closes any dangling runs and records a `reopened` audit event. REFUSED on archived tasks (use `promote` first).

For a real six-iteration research campaign example (Egomotion4D W13–W16, A/G/C lines, gate-first pattern, NO_CLAIM outcomes, confidence normalization, pose provenance repair), see `references/egomotion4d-six-iteration-research-campaign-2026-05-24.md`.
