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

The model is derived from a small canonical source manifest.

### Generic sources

- Hermes custom Kanban notification and watcher design specs;
- `local/bin/start-kanban.sh` launcher contract;
- `plugins/kanban/base_listener.py` safe-boundary and delivery behavior;
- `hermes_cli/kanban_db.py` task lifecycle, control and result outbox behavior.

### Egomotion4D overlay sources

- `/home/wyr/code/Egomotion4D/AGENTS.md`, especially §8.0–§8.4;
- reviewer, planner, implementer and coordinator project prompts under
  `.hermes-kanban/egomotion4d/`;
- `continuous-execution-mode-protocol.md`.

The source manifest stores absolute path, semantic section selector and SHA256.
A deterministic builder emits
`sources/kanban_collaboration_current_evidence.md` containing only bounded
extracts plus the complete source/hash inventory. It never invents policy and
never reads generated mental-model exports as input.

## 4. Maintenance and Staleness

Before the existing evidence-bundle refresh, daily maintenance regenerates the
derived source snapshot atomically. Any canonical source hash or selected
section change changes this model's evidence SHA and makes it stale. The normal
inactive-slot refresh, adjudication, smoke and atomic PASS_PUBLISH switch then
apply without a second maintenance system.

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

Publication requires the existing candidate completeness check, adjudicator and
target-isolated smoke gate. Review exports embed the exact generation spec,
derived evidence and source hash inventory. The wiki registry index gains the
model automatically after PASS_PUBLISH.

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

## 9. Verification

Tests must prove deterministic source extraction and hash-driven staleness,
fail-closed missing selectors, registry/index inclusion, generation contract
completeness, target-isolated smoke routing and no regression of the four
existing models. A final audit verifies PASS_PUBLISH metadata, current/review
exports and exact hashes. Existing unrelated dirty wiki files are preserved and
excluded from the implementation commit.
