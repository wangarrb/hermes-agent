# Kanban Owner First-Pass Reliability Implementation Plan

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

- [ ] Add a failing test with a generation-2 goal, an old failed run, a
  current-generation run, a corrective `REVIEWER_RESULT` comment, and recent
  cross-task role history.
- [ ] Verify RED: corrective handback is not front-loaded, old run text and
  cross-task history are still injected.
- [ ] Render the latest corrective comment before attempts; for goal tasks,
  show full attempts only when `run.generation == task.generation`, collapse
  older attempts to a count, and omit cross-task role history.
- [ ] Verify the new goal fixture and existing ordinary-task context tests.
- [ ] Commit the focused context change.

## Chunk 2: Safe Hermes idle boundary

### Task 2: Distinguish idle from active `msg=interrupt`

**Files:**
- Modify: `plugins/kanban/hermes_listener/hermes_kanban_interactive.py`
- Modify: `tests/plugins/test_kanban_idle_continuation.py`

- [ ] Add a failing fixture for active `⚕ msg=interrupt ...` without `❯` and
  assert claim/followup readiness is false.
- [ ] Verify the exact idle fixture `⚕ ❯ msg=interrupt ...` remains true.
- [ ] Tighten the Hermes override to require the exact idle status pattern;
  the active lookalike remains busy.
- [ ] Run focused listener tests and commit.

## Chunk 3: Evidence-complete owner preflight

### Task 3: Project role guidance

**Files:**
- Modify:
  `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/planner/kanban-system-prompt.md`

- [ ] Add one sentence requiring a 5–8 row success-gate evidence matrix for
  formal review and requiring the primary production-path falsifier in the
  final acceptance modality.
- [ ] Confirm the rule does not duplicate workflow, add a mechanical gate, or
  mention Graphify; project architecture understanding remains Serena +
  RepoWise per `AGENTS.md §9/§10.4`.
- [ ] Run `git diff --check`, commit only the role prompt, and record its SHA.

## Chunk 4: Regression and activation

### Task 4: Verify and reload

**Files:**
- Test only.

- [ ] Run focused context and listener suites plus the existing Kanban
  lifecycle regression.
- [ ] Run Python compilation and `git diff --check` in both repositories.
- [ ] Restart only planner/designer watchers if the listener source changed;
  preserve active TUI sessions/tasks and verify single watcher instances.
- [ ] At a safe idle boundary, notify planner/designer of the role-prompt SHA;
  do not inject while either pane is active.
