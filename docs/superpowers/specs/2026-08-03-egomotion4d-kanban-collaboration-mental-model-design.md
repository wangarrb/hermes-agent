# Egomotion4D Kanban Collaboration Mental Model Design

**Date:** 2026-08-03  
**Status:** proposed  
**Logical ID:** `egomotion4d-kanban-collaboration`

## 1. Goal

Add one maintained, user-auditable mental model that explains the current
Zellij + interactive watcher + Hermes Kanban collaboration system. The model
has two explicit layers:

1. generic mechanics shared by projects: pane/watcher topology, task and goal
   lifecycle, durable handback, safe injection, result notification and retry;
2. the Egomotion4D overlay: role authority, workspaces, review gates, route
   probability/reset policy, resource modes and project-specific commands.

It is a bounded consumption cache, not a new rule source. Runtime code and
automatically loaded project instructions remain authoritative.

## 2. Existing Gap

The current registry contains dynamic actor, pose/gauge, research guardrails
and static surface models. None covers roles, Zellij panes, watcher behavior,
Kanban lifecycle or workspace ownership. `egomotion4d-research-guardrails` is
algorithm-only and must not absorb operational process material.

## 3. Truth Ownership

The model is derived from a small canonical source manifest at
`~/.hermes/mental-models/egomotion4d/sources/kanban_collaboration_sources.json`.
Each entry uses one of `whole_file`, `markdown_heading`, `python_symbol` or
`shell_function`; selectors must resolve exactly once.

### Generic sources

- `/home/wyr/.hermes/hermes-agent-repo/docs/superpowers/specs/2026-08-03-kanban-publisher-result-notifications-design.md`
  (`whole_file`);
- `/home/wyr/.hermes/hermes-agent-repo/docs/superpowers/specs/2026-08-03-multi-owner-project-workspaces-design.md`
  (`whole_file`);
- `/home/wyr/.hermes/hermes-agent-repo/local/bin/start-kanban.sh`
  (`shell_function: workspace_for_role`, `shell_function: build_role_command`,
  plus the usage block bounded by `usage()`);
- `/home/wyr/.hermes/hermes-agent-repo/plugins/kanban/base_listener.py`
  (`python_symbol: BaseInteractiveListener.wait_for_stable_composer_input`,
  `_handle_idle_task_followup`, `pump_control_messages`,
  `pump_result_notifications`);
- `/home/wyr/.hermes/hermes-agent-repo/hermes_cli/kanban_db.py`
  (`python_symbol: create_task`, `return_task_for_rework`,
  `lease_result_notifications`, `result_wait_state`).

### Egomotion4D overlay sources

- `/home/wyr/code/Egomotion4D/AGENTS.md`
  (`markdown_heading: ## 8. Kanban 任务系统`, through the next level-2
  heading or EOF);
- `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/reviewer/kanban-system-prompt.md`
  (`whole_file`);
- `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/planner/kanban-system-prompt.md`
  (`whole_file`, also authoritative for designer behavior);
- `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/implementer/kanban-system-prompt.md`
  (`whole_file`);
- `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/coordinator/kanban-system-prompt.md`
  (`whole_file`);
- `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/continuous-execution-mode-protocol.md`
  (`whole_file`).

The source manifest stores absolute path, semantic section selector and SHA256.
A deterministic builder emits
`sources/kanban_collaboration_current_evidence.md` containing only bounded
extracts plus the complete source/hash inventory. It never invents policy and
never reads generated mental-model exports as input.

## 4. Maintenance and Staleness

The only scheduled entrypoint remains
`local/hermes-scripts/daily_mental_model_wrapper.py`, deployed as the current
Hermes cron wrapper. It invokes the deployed
`~/.hermes/scripts/hindsight_daily_noagent.py --mental-model-daily`. At the
start of `_refresh_evidence_bundle()`, that script calls the new deterministic
builder `~/.hermes/scripts/kanban_collaboration_evidence.py`, whose maintained
source lives at
`local/hermes-scripts/kanban_collaboration_evidence.py`. The builder atomically
regenerates
`~/.hermes/mental-models/egomotion4d/sources/kanban_collaboration_current_evidence.md`
from the manifest before source hashes are recomputed.

Any canonical source hash or selected section change changes this model's
evidence SHA and makes it stale. The normal inactive-slot refresh,
adjudication, smoke and atomic PASS_PUBLISH switch then apply without a second
maintenance system.

Missing files, missing selected headings, duplicate selectors or an empty
extract fail closed as `BLOCK_INVALID_EVIDENCE_BUNDLE`; the previous accepted
revision remains current. No historical backfill or auto-rewrite of canonical
rules is performed.

## 5. Model Contract

The accepted content is Chinese, at most 4096 tokens, and must contain these
sections in order:

1. `系统拓扑与唯一分发路径`
2. `角色职责与决策权`
3. `任务、Goal 与审核生命周期`
4. `Workspace、分支与并发写集`
5. `Watcher 安全注入与结果通知`
6. `Reviewer 结论、成功率与路线重置`
7. `资源模式与委派边界`
8. `故障恢复与禁止事项`
9. `事实源与更新方式`

Required facts include:

- interactive watcher + Zellij inject is the only dispatch path;
- planner uses the project main directory, designer/coordinator use their
  registered long-lived workspaces, and implementer defaults to publisher
  workspace;
- large owner work uses a real Kanban goal and final reviewer acceptance is
  part of completion;
- reviewer returns one deterministic four-state verdict and owns formal route
  probability/reset decisions;
- cross-profile create subscribes the publisher by default, same-profile
  create does not unless explicitly requested;
- result events enqueue immediately, watcher drains FIFO at a safe boundary,
  and waiting goals use the 120-minute insurance interval;
- source labels `[by watcher]` / `[by <profile>]`, durable comments as truth,
  no notification/continuation cards, no headless dispatch;
- current code/artifacts and `AGENTS.md` override stale memories or exports.

The terminal marker is `END_KANBAN_COLLABORATION`.

## 6. Publication Gates

Add `specs/kanban-collaboration.json`, an A/B physical pair, registry entry,
review-export manifest and a focused benchmark file. Smoke questions must cover
at least:

- allowed dispatch path and why headless dispatch is rejected;
- same-profile versus cross-profile notification policy;
- goal completion and final reviewer acceptance;
- planner/designer/coordinator workspace ownership;
- reviewer four-state verdict and route reset threshold;
- busy composer behavior and durable FIFO delivery;
- reassign versus reverse notification tasks;
- authority when model content conflicts with current code/`AGENTS.md`.

Each smoke item has an exact required assertion and forbidden assertion. For
example: cross-profile create must say “default subscribe” while same-profile
must say “default off unless explicit”; a busy/non-stable composer must say the
queue remains durable and must not say it injects immediately; goal completion
must include final reviewer `通过|带病通过` and must reject “submission means
done”.

Publication requires the existing candidate completeness check, adjudicator and
target-isolated smoke gate. Runtime identity is exactly:

- registry owner:
  `~/.hermes/mental-models/egomotion4d/registry.json`;
- logical ID: `egomotion4d-kanban-collaboration`;
- physical IDs: `egomotion4d-kanban-collaboration-a` and
  `egomotion4d-kanban-collaboration-b`;
- generation spec:
  `~/.hermes/mental-models/egomotion4d/specs/kanban-collaboration.json`;
- benchmark:
  `~/.hermes/mental-models/egomotion4d/benchmark/questions-kanban-collaboration.json`;
- review manifest owner:
  `~/.hermes/mental-models/egomotion4d/review_exports.json`.

`mental_model_maintain()` may refresh only the inactive physical ID and records
the candidate transaction. Only `mental_model_adjudicate()` may atomically flip
`registry.models[logical_id].active_slot` after completeness, adjudication and
`_run_smoke_regression()` pass. It then invokes
`_export_accepted_consumers()` for accepted current/history exports and
`_publish_review_exports()` for accepted review current/history exports; no
builder or recreate script may write an accepted revision directly. Review
exports embed the exact generation spec, derived evidence and source hash
inventory. `_render_mental_model_index()` adds the model to the wiki registry
index automatically after PASS_PUBLISH.

## 7. Consumer Boundary

The current export is primarily for user audit, reviewer/planner/designer
orientation and explicit `mental_model_preflight` retrieval. It is not added as
a mandatory per-turn prompt and does not replace `AGENTS.md`, role prompts or
task bodies. This avoids paying the model's token cost on every task while
keeping one maintained current overview available.

## 8. Minimal File Scope

Implementation changes only the custom mental-model/Kanban overlay:

- Hermes repo `local/mental-models/egomotion4d/` and the existing daily mental
  model wrapper/tests;
- live `~/.hermes/mental-models/egomotion4d/` registry/spec/source/benchmark
  state through the existing recreate/maintenance path;
- generated wiki mental-model current/review/history/index outputs.

It does not change core Hindsight, generic Hermes Agent runtime, Egomotion4D
algorithm code or old mental-model history.

For the initial implementation commit, the wiki write allowlist is limited to:

- `auto-maintenance/project/egomotion4d/mental-models/README.md`;
- `.../exports/current/egomotion4d-kanban-collaboration.md`;
- `.../exports/history/egomotion4d-kanban-collaboration-*.md`;
- `.../exports/review/current/egomotion4d-kanban-collaboration.md`;
- `.../exports/review/history/egomotion4d-kanban-collaboration-*.md`.

Daily reports may be generated for operational evidence but are not staged by
this change. No pre-existing wiki modification or deletion outside the
allowlist is staged.

## 9. Verification

Tests must prove deterministic source extraction and hash-driven staleness,
fail-closed missing selectors, registry/index inclusion, generation contract
completeness, target-isolated smoke routing and no regression of the four
existing models. A final audit verifies PASS_PUBLISH metadata, current/review
exports and exact hashes. Existing unrelated dirty wiki files are preserved and
excluded from the implementation commit.
