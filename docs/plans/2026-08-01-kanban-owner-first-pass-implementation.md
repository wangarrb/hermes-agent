# Kanban Owner First-Pass Reliability Implementation Plan

status: completed

> **For agentic workers:** REQUIRED: Execute locally with TDD. Do not create a
> worktree or delegate; this is a focused change in the existing Hermes main
> tree plus one project role-prompt line.

**Goal:** Keep current corrective evidence dominant, prevent unsafe verdict
wakes, and make planner/designer preflight evidence-complete.

**Architecture:** Change only prompt rendering and Hermes pane-state
classification in the Hermes repository. Add one concise behavioral rule to
the Egomotion4D planner/designer prompt; do not add a schema or lifecycle gate.

**Tech Stack:** Python, SQLite, Zellij screen fixtures, pytest, Markdown.

---

## Chunk 1: Goal context hygiene

### Task 1: Current-generation goal context

**Files:**
- Modify: `hermes_cli/kanban_db.py`
- Modify: `tests/hermes_cli/test_kanban_core_functionality.py`

- [x] Add a failing test with a generation-2 goal, an old failed run, a
  current-generation run, a corrective `REVIEWER_RESULT` comment, and recent
  cross-task role history.
- [x] Verify RED: corrective handback is not front-loaded, old run text and
  cross-task history are still injected.
- [x] Render the latest corrective comment before attempts; for goal tasks,
  show full attempts only when `run.generation == task.generation`, collapse
  older attempts to a count, and omit cross-task role history.
- [x] Verify the new goal fixture and existing ordinary-task context tests.
- [x] Commit the focused context change.

## Chunk 2: Safe Hermes idle boundary

### Task 2: Distinguish idle from active `msg=interrupt`

**Files:**
- Modify: `plugins/kanban/hermes_listener/hermes_kanban_interactive.py`
- Modify: `tests/plugins/test_kanban_idle_continuation.py`

- [x] Add a failing fixture for active `⚕ msg=interrupt ...` without `❯` and
  assert claim/followup readiness is false.
- [x] Verify the exact idle fixture `⚕ ❯ msg=interrupt ...` remains true.
- [x] Tighten the Hermes override to require the exact idle status pattern;
  the active lookalike remains busy.
- [x] Run focused listener tests and commit.

## Chunk 3: Evidence-complete owner preflight

### Task 3: Project role guidance

**Files:**
- Modify:
  `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/planner/kanban-system-prompt.md`

- [x] Add one sentence requiring a 5–8 row success-gate evidence matrix for
  formal review and requiring the primary production-path falsifier in the
  final acceptance modality.
- [x] Confirm the rule does not duplicate workflow, add a mechanical gate, or
  mention Graphify; project architecture understanding remains Serena +
  RepoWise per `AGENTS.md §9/§10.4`.
- [x] Run `git diff --check`, commit only the role prompt, and record its SHA.

### 更新记录

| 日期 | 关键数据/结果 | 结论 | 关键转折及原因 |
|------|--------------|------|---------------|
| 2026-08-02 00:40 | Hermes `c795bd2db`, `2a578f5cc`, `ea94ec1d4`; Egomotion4D `2a6444b`, `6914784`; planner prompt SHA256 `96306415196f62fadec51e032b83625ca85e7770414e7ea97527201402975183` | Chunks 1–3 complete | Production idle regex already enforced the required distinction; the change pinned it with a regression instead of adding redundant logic. |

## Chunk 4: Regression and activation

### Task 4: Verify and reload

**Files:**
- Test only.

- [x] Run focused context and listener suites plus the existing Kanban
  lifecycle regression.
- [x] Run Python compilation and `git diff --check` in both repositories.
- [x] Restart only planner/designer watchers if the listener source changed;
  preserve active TUI sessions/tasks and verify single watcher instances.
- [x] Notify planner at its safe boundary. Do not inject into the active
  designer turn; carry the same evidence-matrix rule in its durable terminal
  handback and let the restarted watcher re-inject the original goal.

### 更新记录

| 日期 | 关键数据/结果 | 结论 | 关键转折及原因 |
|------|--------------|------|---------------|
| 2026-08-02 00:49 | 196 related tests PASS in 85.33s; compileall PASS; planner watcher `2104130`, designer watcher `2105661`; designer goal restored as run `2006`; test alignment commit `31ca06939` | ✅ PASS | Active-listener restart reclaims its run, so recovery relies on the restarted watcher re-injecting the same goal at a safe boundary; no worktree/code was lost and no continuation was created. |
