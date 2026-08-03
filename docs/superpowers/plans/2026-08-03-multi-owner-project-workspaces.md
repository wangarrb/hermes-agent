# Multi-owner Project Workspaces Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make planner, designer, and coordinator equal project owners; give designer/coordinator safe per-project long-lived worktrees; support explicit single-owner project switching; reserve implementer guidance for reviewer assistance.

**Architecture:** Keep all runtime behavior in the custom Kanban plugin/local layer. A focused shell library owns Git worktree derivation and synchronization, the launcher consumes it, and a detached pane-switch helper reuses the existing reviewer-switch safety pattern. Project role documents provide the guidance-only implementer responsibility change.

**Tech Stack:** Bash, Python/pytest, Zellij CLI, Git worktrees, Hermes interactive Kanban plugin.

**Spec:** `docs/superpowers/specs/2026-08-03-multi-owner-project-workspaces-design.md`

---

## Chunk 1: Shared role semantics

### Task 1: Make three effective owner roles share the project owner prompt

**Files:**
- Modify: `plugins/kanban/role_context.py`
- Modify: `plugins/kanban/base_listener.py`
- Modify: `tests/plugins/test_kanban_role_context_matrix.py`
- Modify: `tests/plugins/test_kanban_control_delivery.py`

- [ ] Add failing tests proving `coordinator` retains effective role
  `coordinator` while loading the project planner prompt, matching designer.
- [ ] Add failing guidance assertions proving planner/designer/coordinator share
  the owner capability boundary and implementer is a bounded reviewer assistant.
- [ ] Assert generic owner guidance sends long work to background subagents,
  calls the detached helper for an explicit project switch, and makes
  owner-to-implementer delegation an explicit non-default exception.
- [ ] Run the two focused test files and confirm the new assertions fail.
- [ ] Extend `_ROLE_SOURCE_ALIASES` to map coordinator to planner and replace the
  stale per-role listener strings with a shared owner string plus the new
  implementer boundary.
- [ ] Re-run focused tests and confirm PASS.
- [ ] Commit only these plugin/test files with
  `feat(kanban): align three owner roles`.

### Task 2: Encode the guidance-only implementer responsibility in project docs

**Files:**
- Modify: `/home/wyr/code/Egomotion4D/AGENTS.md`
- Modify: `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/planner/kanban-system-prompt.md`
- Modify: `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/reviewer/kanban-system-prompt.md`
- Modify: `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/implementer/kanban-system-prompt.md`
- Modify: `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/coordinator/kanban-system-prompt.md`
- Modify: `/home/wyr/code/Egomotion4D/.hermes-kanban/egomotion4d/continuous-execution-mode-protocol.md`
- Create: `/home/wyr/code/Egomotion4D/tests/test_role_guidance_contract.py`

- [ ] Add a static guidance test reading the project files and rejecting language
  that makes owner-to-implementer delegation the default long-work path.
- [ ] Verify RED against current project guidance.
- [ ] Update the smallest guaranteed sections: owners use background subagents
  for long work; implementer assists reviewer; reviewer retains probability,
  route, verdict, and handback; writable implementer cards state absolute
  workspace/branch/base/write-set/commit owner and otherwise stay read-only.
- [ ] Update coordinator references so it is a third owner using the shared
  planner prompt and its own long-lived workspace; replace the old coordinator
  lifecycle-only prompt rather than leaving an alternate stale definition.
- [ ] Require designer/coordinator to rerun the owner-workspace helper before a
  writable task and record absolute workspace, actual role branch, and frozen
  integration SHA. Require natural-language project-switch requests to invoke
  the detached switch helper after background-work clearance.
- [ ] Re-run the static guidance test and `git diff --check` in Egomotion4D.
- [ ] Commit only the listed Egomotion4D guidance files and its repo-local test with
  `docs: define three owners and reviewer implementer`.

## Chunk 2: Long-lived owner worktrees

### Task 3: Implement a project-neutral owner worktree library

**Files:**
- Create: `local/lib/owner_workspace.sh`
- Create: `local/bin/hermes-kanban-owner-workspace`
- Create: `tests/local/test_owner_workspace.sh`

- [ ] Write RED shell tests using temporary Git repositories for sibling path
  derivation, `designer/mainline`, `coordinator/mainline`, integration-branch
  detection, and preservation of role-only commits.
- [ ] Add RED cases for dirty trees, unexpected branches, merge conflict abort,
  partial-creation cleanup, and a same-named worktree/branch belonging to an
  unrelated Git common-dir; the unrelated-repository case must be zero-change.
- [ ] Implement pure helpers for role validation, derived paths, integration SHA,
  and current branch.
- [ ] Implement `prepare` and `sync`: create missing linked worktree, require the
  expected role branch, merge the frozen integration SHA, and never reset.
- [ ] Make `--dry-run` print the exact path/branch/base without repository writes.
- [ ] Run the shell suite, `bash -n`, and `git diff --check`.
- [ ] Commit with `feat(kanban): manage owner workspaces`.

### Task 4: Teach the launcher to derive and prepare both owner workspaces

**Files:**
- Modify: `local/bin/start-kanban.sh`
- Modify: `tests/local/test_start_kanban_designer.py`

- [ ] Add RED tests using an arbitrary `/tmp/OtherRepo` workspace and assert
  derived sibling designer/coordinator paths, role cwd, and override behavior.
- [ ] Add RED assertions proving launcher help, top-level examples, and assist
  regression fixtures no longer promote `designer/coordinator:implementer` and
  instead document `reviewer:implementer`.
- [ ] Add RED tests proving dry-run has no Git side effects and actual launch
  preparation delegates to the owner-workspace helper.
- [ ] Run the focused launcher test file and confirm the new path, guidance, and
  preparation assertions fail before changing the launcher.
- [ ] Add `--coordinator-workspace`; derive both role paths only after resolving
  the primary workspace; map coordinator in `workspace_for_role`.
- [ ] Replace the launcher help/examples and assist fixture with
  `reviewer:implementer`; retain the generic assist mechanism without
  advertising the retired owner pattern.
- [ ] Call the helper for enabled owner panes before layout generation while
  preserving `--dry-run` zero-write behavior.
- [ ] Update help/output without an Egomotion4D hard-coded default.
- [ ] Re-run launcher tests and shell syntax checks.
- [ ] Commit with `feat(kanban): launch coordinator worktree`.

## Chunk 3: Explicit owner project switch

### Task 5: Implement safe detached owner-pane switching

**Files:**
- Create: `local/lib/owner_project_switch.sh`
- Create: `local/bin/hermes-kanban-switch-owner-project`
- Modify: `local/bin/start-kanban.sh`
- Create: `tests/local/test_owner_project_switch.sh`

- [ ] Add RED tests for role validation, project registry resolution,
  board/workspace same-binding validation, and unbound explicit input logging.
- [ ] Add RED safety tests for missing live reviewer lane, running owner task,
  absent background-work-clear confirmation, unstable composer, changed pane
  identity, and final pre-replacement recheck.
- [ ] Add RED success dry-run proving only the selected owner pane is replaced,
  target board/workspace are used, and the target conversation is fresh rather
  than resumed.
- [ ] Parameterize success for planner/designer/coordinator and prove their
  resolved target workspaces are respectively primary, sibling designer, and
  sibling coordinator.
- [ ] Add the two-phase failure test: target preparation completes, final source
  pane recheck fails, the pane stays unchanged, the prepared target remains,
  and the log records `PREPARED_NOT_SWITCHED` with before/after SHA.
- [ ] Reuse or extract the existing reviewer-switch idle/signature helpers; do
  not duplicate a second busy-marker parser.
- [ ] Implement the detached worker and target preparation. Log fully prepared
  but unswitched targets as `PREPARED_NOT_SWITCHED` with before/after SHA.
- [ ] Expose launcher forwarding through
  `start-kanban.sh --switch-owner-project <role>` plus explicit target options.
- [ ] Run the switch suite, reviewer-mode regression tests, and `bash -n`.
- [ ] Commit with `feat(kanban): switch owner project safely`.

## Chunk 4: Egomotion4D migration and end-to-end verification

### Task 6: Migrate the two Egomotion4D owner worktrees without losing delivery

**Files:**
- Existing: `/home/wyr/code/Egomotion4D-designer`
- Create worktree: `/home/wyr/code/Egomotion4D-coordinator`

- [ ] Confirm both primary and designer worktrees are clean enough for migration;
  stop if new uncommitted changes appeared.
- [ ] Freeze `M=$(git -C /home/wyr/code/Egomotion4D rev-parse master)`, original
  designer branch, and original designer HEAD in the migration log.
- [ ] Before any write, require the frozen original designer HEAD to equal the
  inspected migration starting HEAD and require `d43c9ea` to be its ancestor;
  otherwise stop with zero repository change.
- [ ] Create `designer/mainline` from the original designer HEAD and merge `M`;
  on conflict abort and restore the original checkout without reset.
- [ ] Create coordinator worktree and `coordinator/mainline` from `M`.
- [ ] Verify `M` is ancestor of both role branches and `d43c9ea` is ancestor of
  designer/mainline; verify all three worktree paths/branches/status.
- [ ] Do not delete or push legacy role branches.

### Task 7: Run full focused regression and publish

**Files:**
- All files changed by Tasks 1-5.

- [ ] Run plugin role-context/control tests and project guidance tests.
- [ ] Run owner workspace, launcher, owner switch, reviewer switch, and watcher
  idle-detection focused suites.
- [ ] Run `bash -n` on every changed/new shell file and `git diff --check` in
  both repositories.
- [ ] Inspect staged name-status in each repository and exclude unrelated dirty
  files, especially `scripts/wechat_inject.py` and existing Egomotion4D work.
- [ ] Record final commit SHAs, worktree ancestry evidence, and dry-run commands.
- [ ] Record local Hermes and Egomotion4D commit SHAs. Do not push unless the user
  gives separate explicit push authorization after this delivery.
