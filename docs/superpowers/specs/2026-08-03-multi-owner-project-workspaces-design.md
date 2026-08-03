# Multi-owner project workspaces design

Status: approved design
Date: 2026-08-03

## Goal

Turn `coordinator` into a third long-running project owner equivalent to
`planner` and `designer`, while giving `designer` and `coordinator` independent
long-lived worktrees and branches for every project.  One owner pane may be
explicitly switched to another project without restarting the other panes or
carrying the source project's working context into the target project.

## Scope boundary

The implementation belongs to the custom Kanban layer hosted in the Hermes
repository:

- `plugins/kanban/`
- `local/bin/`
- `local/lib/`
- focused tests for those paths
- project-local role guidance such as Egomotion4D's `AGENTS.md` and
  `.hermes-kanban/`

Do not add role-specific behavior to the general Hermes agent runtime, generic
database layer, or unrelated `hermes_cli/` code.  The existing designer check
in `hermes_cli/kanban_db.py` is not expanded in this change.

## Role model

`planner`, `designer`, and `coordinator` are three equal owner lanes.  Each may
design a route, write plans, create and supervise Kanban tasks, implement code,
request reviewer decisions at load-bearing checkpoints, and drive its goal to
reviewer-accepted completion.  `coordinator` no longer owns generic lifecycle
coordination; the watcher and each task owner maintain task state.

The generic listener guidance must describe one shared owner contract.  At the
project layer, `designer` and `coordinator` reuse the planner project prompt
when a project does not provide an intentional role-specific override.  This
keeps the behavior useful outside Egomotion4D without copying its paths or
algorithm rules into Hermes.

### Implementer responsibility

The implementer is no longer the default worker pool for planner, designer, or
coordinator.  An owner keeps bounded work in its own goal and preferentially
uses a background subagent for long-running or parallel implementation,
experiment, or inventory work.  The owner remains responsible for integration,
evidence, and goal completion.  It may still create an implementer Kanban card
when a background subagent is unsuitable, the contract is frozen, and durable
cross-session Kanban ownership is worth the communication cost.

Implementer capacity is primarily reserved for reviewer assistance.  A reviewer may
delegate deterministic diff inspection, focused tests, artifact and metric
inventory, reproduction, bounded tool-heavy investigation, or a small repair
whose acceptance contract is already frozen.  Implementer cost is treated as
negligible relative to reviewer tokens, so the reviewer should delegate when
the saved reviewer work exceeds the communication cost.  Formal success
probability, algorithm direction, route reset, review verdict, and final
handback remain reviewer-only decisions.

Every reviewer-to-implementer task records an absolute workspace, actual
branch, base SHA, write set, and commit ownership when writes are allowed.  If
those fields are absent, the task is read-only and the implementer may report
evidence but must not repair code.  This is especially important when the
reviewed delivery belongs to a designer or coordinator worktree rather than
the primary repository.

This responsibility change is guidance-only.  It is expressed in the
guaranteed project instructions and role prompts; this design deliberately
does not add task-creator validation, claim rejection, database policy, or a
new notification protocol.  A mistaken owner-to-implementer assignment is
corrected through role guidance rather than a mechanical gate.

## Workspace convention

Given a primary repository `/parent/Repo`:

| Role | Workspace | Long-lived branch |
|---|---|---|
| planner | `/parent/Repo` | the repository integration branch |
| designer | `/parent/Repo-designer` | `designer/mainline` |
| coordinator | `/parent/Repo-coordinator` | `coordinator/mainline` |

The integration branch is detected from the primary workspace rather than
hard-coded as `master`; Egomotion4D currently resolves to `master`.  Explicit
`--designer-workspace` and `--coordinator-workspace` overrides remain
available.

A small custom helper creates, inspects, and synchronizes owner worktrees.  It
must preserve role-only commits and synchronize by making the current
integration SHA an ancestor of the owner branch.  It must never use reset to
discard delivery commits.

## Startup and task flow

`start-kanban.sh --workspace /abs/path/Repo` derives both owner workspaces and
ensures their contracts before launching panes.  A missing worktree is created
from the current integration branch.  An existing worktree must be clean and
on its expected role branch before automatic synchronization.

Before a new designer/coordinator implementation task, the owner helper is run
again.  The task records the actual absolute workspace, role branch, and the
integration SHA observed at creation.  Synchronization is a default action,
not permission to overwrite dirty work or silently resolve conflicts.  A
read-only task may still inspect a deliberately unsynchronized workspace when
its task contract says so.

## Explicit owner project switch

The normal user interface is a direct instruction to the owner, for example:

> Switch to `/home/wyr/code/OtherProject`, board `other-project`.

If the Hermes project registry uniquely maps a project name to its primary
path and board, the user may provide only the project name.  Otherwise the
owner asks only for the missing path or board.  When both `--board` and
`--workspace` are explicit and a registry binding exists, they must match that
same binding.  An unbound pair requires explicit user input and is recorded in
the switch log rather than silently inferred.  The deterministic command is:

```bash
hermes-kanban-switch-owner-project \
  --session kanban-egomotion4d \
  --role coordinator \
  --board other-project \
  --workspace /home/wyr/code/OtherProject
```

The command is a pane control operation, not a Kanban task.  The owner launches
a detached switch worker so replacing its own pane cannot kill the operation.
The worker:

1. accepts only `planner`, `designer`, or `coordinator`;
2. confirms the target board already has a live reviewer lane capable of
   accepting its checkpoint and final-review tasks;
3. confirms the selected pane has no running Kanban task, no active owner
   subagent, and no background job still using the source workspace;
4. confirms an idle prompt and unchanged composer three times at ten-second
   intervals;
5. prepares and synchronizes the target owner workspace;
6. immediately before replacement, re-resolves the same pane and confirms once
   more that it has no running task and that its composer signature is still
   the accepted signature;
7. starts the target board/profile in a fresh project conversation;
8. replaces only the selected pane in place.

The natural-language owner flow verifies and explicitly confirms that its
subagent/background-work ledger is clear before invoking the switch helper.
The helper requires that confirmation and refuses an unattended switch without
it.  An owner goal cannot complete while an owned background subagent remains
active or has not handed back its result.

The role profile and durable memory are retained, but the previous project's
conversation is not resumed.  Other panes remain on their current boards and
projects.  A switch back uses the same command with the original project.

## Failure semantics

- Dirty owner workspace: report paths and stop before merge or pane changes.
- Unexpected owner branch: report expected/actual branch and stop.
- Merge conflict: abort the merge, restore the pre-switch repository state,
  and keep the old pane running.
- Running task, busy prompt, or changing composer: do not switch.
- Missing target reviewer lane or active source-project subagent/background
  work: do not switch.
- Missing or ambiguous project registry binding: request the missing value;
  never guess a repository or board.
- Any failure before pane replacement is a zero-pane-change failure and is
  written to a dedicated switch log.  Repository preparation has its own
  explicit boundary: a partially created worktree/ref is removed when its
  preparation fails; a fully synchronized target worktree may be retained when
  the final source-pane check later fails, but must be logged as
  `PREPARED_NOT_SWITCHED` with its before/after SHA.  The helper never claims
  that this case is a zero-filesystem-change failure.

## Egomotion4D migration

The existing designer tree is clean but has substantial unmerged dynamic actor
delivery.  At migration start freeze `M=$(git rev-parse master)` and record the
original designer branch and HEAD.  Create `designer/mainline` from its current
HEAD `d43c9ea`, then merge `M`; do not recreate it from `master` and do not
delete the old designer branch.  On conflict, abort and restore the original
designer checkout/branch without changing its delivery state.  Create
`/home/wyr/code/Egomotion4D-coordinator` and `coordinator/mainline` from `M`.
Accept the migration only when `M` is an ancestor of both role branches and
`d43c9ea` remains an ancestor of `designer/mainline`.  Archive old branches
only after the new worktrees and ancestry have been verified.

## Verification

- Generic role guidance gives all three owners the same capability boundary.
- Focused guidance tests prove owner prompts use background subagents instead
  of implementer cards, reviewer guidance permits only bounded implementer
  delegation, and success probability, algorithm direction, verdict, and
  handback remain reviewer-owned.
- Project role-context tests prove designer and coordinator reuse the planner
  prompt while retaining their effective role names.
- Launcher tests derive sibling paths from arbitrary temporary repository
  names and honor explicit overrides.
- Git integration tests prove worktree creation, clean synchronization,
  preservation of role-only commits, and zero-change behavior for dirty trees
  and merge conflicts.
- Switch tests prove running-task and unstable-composer rejection, project
  registry resolution, fresh target conversation, and single-pane replacement.
- Egomotion4D checks prove both role worktrees use the expected branches and
  that the selected `master` SHA is an ancestor after migration.

## Non-goals

- No board metadata schema or dashboard configuration UI is added.
- No automatic per-task temporary project switching.
- No automatic pushing of owner branches.
- No deletion of legacy worktrees or branches.
- No role-specific changes in the generic Hermes runtime or database schema.
- No database or listener enforcement of who may create implementer tasks.
