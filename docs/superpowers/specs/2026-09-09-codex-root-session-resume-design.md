# Codex Kanban Root Session Resume Design

## Problem

The Codex kanban launcher resolves the most recently updated thread for a
profile workspace and passes it to `codex resume`. Codex persists both the
interactive root thread and its multi-agent v2 sub-agent threads in the same
`threads` table. A sub-agent can therefore sort ahead of its parent.

Codex cannot cold-resume such a sub-agent through the TUI. It requires the
parent thread to be loaded first, so selecting a sub-agent causes the reviewer
pane to fail during bootstrap.

## Root cause

`plugins/kanban/session_scope.py::latest_codex_thread` filters by archive state
and canonical workspace only. It does not distinguish resumable root threads
from `thread_source = 'subagent'` rows. On 2026-09-09 it selected
`01a084ee-dcf0-7140-9c71-7dba149e6138` instead of its existing parent
`01a08481-82dd-7b00-8798-2fc8ac752592`.

## Design

Keep session selection in `latest_codex_thread`, but restrict candidates to
root threads when the installed Codex state schema exposes source metadata.

- If `thread_source` exists, exclude rows whose value is `subagent`.
- Otherwise, if `source` exists, exclude rows whose value is the structured
  sub-agent source or whose JSON source object has a `subagent` member.
- Preserve compatibility with older schemas that expose neither column by
  retaining the existing workspace-and-recency selection.
- Do not mutate, archive, rename, or delete session rows or rollout files.
- Keep the listener API and launch command unchanged; it will receive the
  selected root thread ID as before.

The source filtering belongs in the selector rather than the launcher because
the selector owns the resumability decision and can be tested independently.

## Tests

Add a regression case using a temporary Codex state database containing:

1. An older root thread for the requested workspace.
2. A newer sub-agent thread for the same workspace.
3. An unrelated newer root thread for another workspace.

The selector must return the requested workspace's root thread. Existing tests
must continue to cover exact canonical-workspace matching and legacy schemas.

Run the focused selector/listener tests, then the surrounding kanban listener
test group.

## Operational recovery

After tests pass, restart only the failed reviewer pane through the existing
reviewer-mode switch/launcher path. Confirm its process command resumes parent
thread `01a08481-82dd-7b00-8798-2fc8ac752592`, the pane reaches an interactive
composer, and the watcher child remains live. Other kanban panes and saved
sessions remain untouched.
