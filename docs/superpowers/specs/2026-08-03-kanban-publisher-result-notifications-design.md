# Kanban Publisher Result Notifications Design

**Date:** 2026-08-03  
**Status:** proposed  
**Scope:** custom Kanban plugin/listener and Kanban DB only

## 1. Goal

When a pane publishes a Kanban task, task-result notification is enabled by
default. Completing or otherwise ending that task records a durable
notification for the publishing profile and returns immediately. The target
profile's watcher later drains those notifications in FIFO order and injects a
short wake-up only at a safe input boundary.

The mechanism is generic. It does not encode reviewer/owner policy. Current
project policy mainly uses it for reviewer-created owner tasks and
owner-created reviewer tasks, but any profile can publish and subscribe.

## 2. Non-goals

- Do not replace tasks that perform implementation, review, algorithm
  decisions, or independent evidence work.
- Do not create notification or continuation Kanban cards.
- Do not make the publishing LLM poll, wait, or consume tokens while the task
  runs.
- Do not inject into a busy pane and do not let notification delivery race
  ordinary watcher injection.
- Do not change task ownership merely to return a result.

## 3. Durable model

Add two small board-local tables.

### `kanban_result_subscriptions`

One row means that a profile wants actionable result events for a task.

- `task_id`
- `target_profile`
- `created_at`
- `active`
- primary key: `(task_id, target_profile)`

### `kanban_result_queue`

One row is one durable delivery item.

- monotonic `id`
- `task_id`
- source `event_id`
- `target_profile`
- `event_kind`
- compact JSON `payload`
- `status`: `pending | leased | delivered`
- `lease_owner`, `lease_expires`
- `created_at`, `delivered_at`
- unique key: `(event_id, target_profile)`

The queue is independent from the gateway-oriented `kanban_notify_subs` table.
Gateway delivery and local pane delivery have different routing and safety
requirements and must not overload each other's platform semantics.

## 4. Default subscription policy

The launcher exports the logical pane identity, including
`HERMES_KANBAN_ORIGIN_PROFILE=<profile>`. Create APIs accept an explicit origin
override for non-pane callers.

On `create_task`:

1. If an origin profile is available and differs from the assignee, add a
   result subscription in the same transaction as task creation.
2. Cross-profile notification is the default; `--no-notify-origin` is the
   explicit opt-out.
3. Same-profile publication does not subscribe by default because the same
   role already owns the task lifecycle. A caller may explicitly request it
   with `--notify-origin` for an unusual multi-pane case.
4. Dashboard/user/API creation without a logical profile does not invent a
   pane target. Existing gateway subscriptions remain available for humans.
5. Idempotent task creation also idempotently ensures the requested
   subscription exists.

Reassignment preserves existing subscriptions and does not create a reverse
task. An optional explicit operation may add or transfer a subscription when
actual result ownership changes; ordinary `reassign` only changes the
executor.

## 5. Event production

The existing task state transaction enqueues result items for subscriptions.
The operation is an indexed `INSERT OR IGNORE` and never waits for a pane.

Default actionable events:

- `completed`
- `blocked`
- `gave_up`
- `returned_for_rework`
- control-level supersession/invalidation that requires the subscriber to act

Intermediate crash/timeout attempts that will be retried do not notify. A
terminal circuit-breaker transition does. Event payloads contain only task ID,
kind, generation, short summary/reason, and relevant control/rework IDs; full
truth remains in task comments, events, runs, and artifacts.

## 6. Queue delivery

Each interactive watcher drains only rows addressed to its effective profile.
Delivery has higher priority than goal-completion reminders and ordinary ready
task discovery.

1. Check whether the pane is at a safe input boundary using the existing
   backend-specific busy/composer rules.
2. Inspect the oldest undelivered row for that profile. If it has an unexpired
   lease owned by another watcher, stop; never skip it to deliver newer rows.
   Otherwise atomically reclaim an expired head lease or lease a bounded,
   contiguous FIFO prefix of eligible rows. Refresh the lease immediately
   before injection so it covers the whole paste/Enter operation.
3. Coalesce a bounded consecutive batch into one message while preserving
   queue order.
4. Inject the message plus Enter using the existing tagged injection path.
5. Mark rows delivered only after injection succeeds; release the lease on a
   known injection failure.

Example:

```text
[TASK_RESULTS_READY] t_a=completed; t_b=blocked. Read the durable
summary/comments/events and continue the existing objective. [by watcher]
```

Enqueue and task completion return immediately after the board transaction. No
caller waits for pane idle. If the pane or watcher is down, rows remain pending
and are delivered after it returns. Unique keys prevent duplicate enqueue and
FIFO leases prevent concurrent delivery. Delivery is honestly at-least-once:
a watcher crash after Enter but before the delivered update may repeat a
message. Every injected item therefore carries stable queue/event IDs so the
subscriber can recognize the replay; the design does not claim impossible
exactly-once terminal injection.

## 7. Goal-completion interaction

A goal pane is considered `waiting_on_result` when either condition holds:

- it has an active subscription whose task is not terminal; or
- its result queue has pending/leased items.

An explicitly subscribed task currently executing in this pane is excluded
from the first condition; an unusual same-profile subscription must not make a
goal look as though it is waiting on itself.

While waiting:

- queued result delivery always wins over `GOAL_COMPLETION_CHECK`;
- the normal daytime two-minute goal reminder is suppressed;
- an insurance reminder is allowed at most once per 120 minutes;
- after the last watched task becomes actionable, its queued result is
  delivered, and no other watched task remains nonterminal, reset the
  goal-reminder timer and restore the normal interval.

The 120-minute reminder states which watched tasks are still nonterminal. It
must not tell the agent to create a notification task or poll continuously.
Existing overnight policy remains unchanged when no result is outstanding.

## 8. Reassignment and handback semantics

- A task result returns through its subscription, not by B creating or
  assigning a mirror task back to A.
- `reassign` changes `assignee` and preserves subscriptions.
- B assigns work to A only when A must perform new substantive work.
- `return-for-rework` continues to reopen the original execution card and
  emits its actionable result event; it is not modeled as an ordinary reverse
  assignment.

## 9. Failure behavior

- Queue DB failure aborts the corresponding state transition transaction so a
  committed terminal event cannot silently lose its subscribed notification.
- Zellij unavailable or pane busy leaves the queue untouched.
- A stale/missing profile watcher does not reroute to another role.
- Superseded generations retain generation/event identity so stale callbacks
  cannot acknowledge a newer delivery.
- Malformed payloads fail closed and remain inspectable in the queue.

## 10. Verification

Focused tests must prove:

- cross-profile create defaults to origin subscription, same-profile create
  defaults off, and both explicit opt-in/opt-out work;
- generic profiles work without reviewer/owner branching;
- create/event enqueue is nonblocking and does not call Zellij;
- completion, block, terminal gave-up, and return-for-rework enqueue once;
- retryable crash/timeout does not enqueue;
- reassign preserves the original subscriber;
- FIFO leasing, lease expiry, restart recovery, bounded batching, and
  duplicate event suppression;
- busy panes retain pending rows and safe panes inject in order with
  `[by watcher]`;
- pending/outstanding results suppress normal goal checks and enforce the
  120-minute insurance interval;
- result delivery resets the goal timer and resumes normal behavior.

## 11. Rollout

Implement behind a board/listener feature flag for focused tests, then enable
it by default for interactive watcher task creation. Existing tasks without a
subscription keep current behavior. No historical backfill is required.
