# Codex Root Session Resume Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the Codex kanban reviewer launcher from cold-resuming a multi-agent sub-agent thread, and restore the failed reviewer pane on its existing parent thread.

**Architecture:** Keep workspace and recency selection in `latest_codex_thread`, but add a small row-level source classifier that rejects known sub-agent rows without relying on SQLite JSON functions. Preserve legacy Codex schemas by selecting `NULL` aliases for missing metadata columns. Recover the live pane only after tests pass and after verifying its failed processes are gone.

**Tech Stack:** Python 3, SQLite, pytest, Bash/Zellij, Codex CLI 0.153.4

---

## Chunk 1: Selector regression and fix

### Task 1: Reproduce sub-agent selection

**Files:**
- Modify: `tests/plugins/test_codex_listener_project_sessions.py`
- Reference: `plugins/kanban/session_scope.py`

- [ ] **Step 1: Generalize the temporary database helper**

Allow the helper to create the source-column combinations needed by each test while keeping the existing legacy-schema test unchanged.

- [ ] **Step 2: Add the current-schema regression test**

Create one root row and a newer `thread_source='subagent'` row for the requested workspace, plus a newer unrelated-project root. Assert `CodexInteractiveListener` builds `codex resume <root-id>`.

- [ ] **Step 3: Add compatibility cases**

Add separate tests for:

- JSON `source.subagent` when `thread_source` is absent.
- JSON `source.subagent` when `thread_source` exists but is `NULL` or empty.
- Plain and malformed `source` values that must remain eligible and must not abort the database scan.
- Neither source column, retaining legacy behavior.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
rtk pytest -q tests/plugins/test_codex_listener_project_sessions.py
```

Expected: the new sub-agent cases fail by selecting the newer sub-agent ID; existing legacy/workspace cases pass.

### Task 2: Filter non-resumable Codex sub-agent rows

**Files:**
- Modify: `plugins/kanban/session_scope.py:37`
- Test: `tests/plugins/test_codex_listener_project_sessions.py`

- [ ] **Step 1: Add a private row classifier**

Import `json` and add a private function that:

1. Returns true for normalized `thread_source == 'subagent'`.
2. When `thread_source` is missing/empty, returns true for plain `source == 'subagent'`.
3. Best-effort parses `source`; returns true only for a mapping with a top-level `subagent` member.
4. Treats absent, plain root, or malformed values as non-sub-agent.

- [ ] **Step 2: Select optional metadata compatibly**

Build `thread_source_expr` and `source_expr` as their column names when present and `NULL` otherwise. Select both alongside ID, CWD, and recency, then skip rows classified as sub-agents before comparing workspace and recency.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run:

```bash
rtk pytest -q tests/plugins/test_codex_listener_project_sessions.py
```

Expected: all tests pass.

- [ ] **Step 4: Run surrounding selector/listener tests**

Run:

```bash
rtk pytest -q tests/local/test_kanban_session_scope.py \
  tests/plugins/test_codex_listener_project_sessions.py \
  tests/plugins/test_codex_listener_idle_detection.py
```

Expected: all tests pass with no warnings or errors.

- [ ] **Step 5: Commit the code fix**

```bash
rtk git add plugins/kanban/session_scope.py \
  tests/plugins/test_codex_listener_project_sessions.py
rtk git commit -m "fix(kanban): resume Codex root threads only"
```

## Chunk 2: Live reviewer recovery

### Task 3: Replace the failed pane and verify runtime state

**Files:**
- Read: `local/bin/hermes-kanban-switch-reviewer-mode`
- Read: `/home/wyr/.hermes/kanban/boards/seqscale/logs/codex-interactive-reviewer.log`
- Runtime: Zellij session `kanban-seqscale`, reviewer pane 2

- [ ] **Step 1: Confirm the selector returns the parent**

Call `latest_codex_thread` against `/home/wyr/.codex-kanban/reviewer` and `/home/wyr/code/SeqScale`. Expected ID:

```text
01a08481-82dd-7b00-8798-2fc8ac752592
```

- [ ] **Step 2: Check replacement preconditions**

Confirm the pane still shows the bootstrap error and no reviewer `--watch-child` or reviewer-owned Codex process is live. Abort replacement if work is active.

- [ ] **Step 3: Generate the canonical launch command**

Run:

```bash
rtk local/bin/hermes-kanban-switch-reviewer-mode \
  --board seqscale --mode balanced --session kanban-seqscale \
  --workspace /home/wyr/code/SeqScale --dry-run
```

Use the emitted command verbatim for the replacement.

- [ ] **Step 4: Replace only reviewer pane 2**

Record the focused pane, focus pane 2, and run `zellij action new-pane --in-place --close-replaced-pane` with the generated launch command. Restore the prior focus. Do not restart the full board or mutate saved sessions.

- [ ] **Step 5: Verify end to end**

Confirm:

- Exactly one reviewer watcher child is live.
- Exactly one reviewer Codex TUI process is live.
- Its command contains `resume 01a08481-82dd-7b00-8798-2fc8ac752592`.
- The reviewer pane reaches an empty interactive composer without the bootstrap error.
- Planner, implementer, designer, and coordinator panes remain live.

- [ ] **Step 6: Record final repository state**

Run `rtk git status --short` and report the code commit, tests, selected session ID, and runtime process counts.
