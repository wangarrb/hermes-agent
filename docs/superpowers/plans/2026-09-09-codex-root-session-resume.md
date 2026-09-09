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
- Runtime: Zellij session `kanban-seqscale`, reviewer pane resolved from its
  `terminal_command` immediately before replacement

- [ ] **Step 1: Confirm the selector returns the parent**

Call `latest_codex_thread` against `/home/wyr/.codex-kanban/reviewer` and `/home/wyr/code/SeqScale`. Expected ID:

```text
01a08481-82dd-7b00-8798-2fc8ac752592
```

- [ ] **Step 2: Capture and check replacement preconditions**

Resolve exactly one reviewer pane, require it to be exited/held, capture the
error-screen fingerprint, and require no reviewer watcher or Codex process:

```bash
SESSION=kanban-seqscale
PANES_JSON="$(rtk proxy zellij --session "$SESSION" action list-panes --all --json)"
REVIEWER_COUNT="$(jq '[.[] | select(.is_plugin == false) | select((.terminal_command // "") | contains("--profile reviewer"))] | length' <<<"$PANES_JSON")"
test "$REVIEWER_COUNT" = 1
REVIEWER_PANE="$(jq -r '.[] | select(.is_plugin == false) | select((.terminal_command // "") | contains("--profile reviewer")) | .id' <<<"$PANES_JSON")"
jq -e --argjson id "$REVIEWER_PANE" '.[] | select(.is_plugin == false and .id == $id) | .exited == true and .is_held == true' <<<"$PANES_JSON"
ERROR_SCREEN="$(rtk proxy zellij --session "$SESSION" action dump-screen --full --pane-id "$REVIEWER_PANE")"
grep -Fq 'cannot resume an unloaded multi-agent v2 sub-agent' <<<"$ERROR_SCREEN"
ERROR_SIGNATURE="$(sha256sum <<<"$ERROR_SCREEN" | awk '{print $1}')"
test -z "$(ps -eo args= | awk '/[c]odex_kanban_interactive.py --watch-child/ && /--profile reviewer/ && /--board seqscale/ {print}')"
test -z "$(ps -eo args= | awk '/[c]odex resume/ && /01a084(ee-dcf0-7140-9c71-7dba149e6138|81-82dd-7b00-8798-2fc8ac752592)/ {print}')"
```

Abort replacement if any assertion fails.

- [ ] **Step 3: Generate the canonical launch command**

Capture only the emitted launch command (the dry-run also prints one metadata
line):

```bash
LAUNCH_CMD="$(rtk proxy local/bin/hermes-kanban-switch-reviewer-mode \
  --board seqscale --mode balanced --session "$SESSION" \
  --workspace /home/wyr/code/SeqScale --dry-run | tail -n 1)"
test -n "$LAUNCH_CMD"
```

Use this exact captured line for the replacement.

- [ ] **Step 4: Replace only reviewer pane 2**

Immediately re-resolve the pane and recheck its identity, exited state, screen
fingerprint, and process absence. Then install an EXIT trap before changing
focus so failures restore the user's prior focus:

```bash
CURRENT_JSON="$(rtk proxy zellij --session "$SESSION" action list-panes --all --json)"
CURRENT_REVIEWER_COUNT="$(jq '[.[] | select(.is_plugin == false) | select((.terminal_command // "") | contains("--profile reviewer"))] | length' <<<"$CURRENT_JSON")"
test "$CURRENT_REVIEWER_COUNT" = 1
CURRENT_REVIEWER_PANE="$(jq -r '.[] | select(.is_plugin == false) | select((.terminal_command // "") | contains("--profile reviewer")) | .id' <<<"$CURRENT_JSON")"
test "$CURRENT_REVIEWER_PANE" = "$REVIEWER_PANE"
jq -e --argjson id "$REVIEWER_PANE" '.[] | select(.is_plugin == false and .id == $id) | .exited == true and .is_held == true' <<<"$CURRENT_JSON"
CURRENT_SCREEN="$(rtk proxy zellij --session "$SESSION" action dump-screen --full --pane-id "$REVIEWER_PANE")"
test "$(sha256sum <<<"$CURRENT_SCREEN" | awk '{print $1}')" = "$ERROR_SIGNATURE"
grep -Fq 'cannot resume an unloaded multi-agent v2 sub-agent' <<<"$CURRENT_SCREEN"
test -z "$(ps -eo args= | awk '/[c]odex_kanban_interactive.py --watch-child/ && /--profile reviewer/ && /--board seqscale/ {print}')"
test -z "$(ps -eo args= | awk '/[c]odex resume/ && /01a084(ee-dcf0-7140-9c71-7dba149e6138|81-82dd-7b00-8798-2fc8ac752592)/ {print}')"

ORIGINAL_FOCUS="$(jq -r '.[] | select(.is_plugin == false and .is_focused == true) | .id' <<<"$CURRENT_JSON" | head -n 1)"
restore_focus() {
  if test -n "$ORIGINAL_FOCUS"; then
    rtk proxy zellij --session "$SESSION" action focus-pane-id "$ORIGINAL_FOCUS" >/dev/null 2>&1 || true
  fi
}
trap restore_focus EXIT
rtk proxy zellij --session "$SESSION" action focus-pane-id "$REVIEWER_PANE"
NEW_PANE="$(rtk proxy zellij --session "$SESSION" action new-pane \
  --in-place --close-replaced-pane --name codex-kanban \
  --cwd /home/wyr/code/SeqScale -- bash -lc "$LAUNCH_CMD")"
restore_focus
trap - EXIT
test -n "$NEW_PANE"
```

Do not restart the full board or mutate saved sessions.

- [ ] **Step 5: Verify end to end**

Confirm:

- Exactly one reviewer watcher child is live.
- Exactly one reviewer Codex TUI process is live.
- Its command contains `resume 01a08481-82dd-7b00-8798-2fc8ac752592`.
- The reviewer pane reaches an empty interactive composer without the bootstrap error.
- Planner, implementer, designer, and coordinator panes remain live.

- [ ] **Step 6: Record final repository state**

Run `rtk git status --short` and report the code commit, tests, selected session ID, and runtime process counts.
