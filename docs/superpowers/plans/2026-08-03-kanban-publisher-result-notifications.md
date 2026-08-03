# Kanban Publisher Result Notifications Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver terminal Kanban task results asynchronously to the publishing pane, defaulting on only for cross-profile task creation, while suppressing noisy goal reminders during the wait.

**Architecture:** Keep local pane delivery separate from gateway notifications. Task-state transactions append an event and enqueue one idempotent result row for each durable profile subscription; the target profile's existing interactive watcher later leases the FIFO head and injects a bounded batch only at a verified safe composer boundary. The launcher exports the pane's logical profile, and the CLI turns that identity into an explicit subscription request so the DB layer stays transport-agnostic.

**Tech Stack:** Python 3, SQLite, argparse, pytest, Bash/Zellij launcher.

---

## Chunk 1: Durable cross-profile result delivery

### Task 1: Add the subscription and FIFO outbox primitives

**Files:**
- Modify: `hermes_cli/kanban_db.py` (schema, task creation, event append, queue API)
- Create: `tests/hermes_cli/test_kanban_result_notifications.py`

- [x] **Step 1: Write schema and subscription policy tests**

  Add isolated-board tests that assert fresh and reopened DBs contain
  `kanban_result_subscriptions` and `kanban_result_queue`; an explicitly passed
  `result_subscriber` is inserted once; idempotent create ensures a missing
  requested subscription on the existing task; and `reassign_task` leaves the
  original subscriber unchanged.

- [x] **Step 2: Run the focused tests and verify RED**

  Run:

  ```bash
  python3 -m pytest tests/hermes_cli/test_kanban_result_notifications.py -q
  ```

  Expected: FAIL because the tables and `result_subscriber` API do not exist.

- [x] **Step 3: Add the minimal durable schema and subscription API**

  In `SCHEMA_SQL`, create:

  ```sql
  CREATE TABLE IF NOT EXISTS kanban_result_subscriptions (
      task_id TEXT NOT NULL,
      target_profile TEXT NOT NULL,
      created_at INTEGER NOT NULL,
      active INTEGER NOT NULL DEFAULT 1,
      PRIMARY KEY (task_id, target_profile)
  );
  CREATE TABLE IF NOT EXISTS kanban_result_queue (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      task_id TEXT NOT NULL,
      event_id INTEGER NOT NULL,
      target_profile TEXT NOT NULL,
      event_kind TEXT NOT NULL,
      payload TEXT,
      status TEXT NOT NULL DEFAULT 'pending',
      lease_owner TEXT,
      lease_expires INTEGER,
      created_at INTEGER NOT NULL,
      delivered_at INTEGER,
      UNIQUE (event_id, target_profile)
  );
  ```

  Add indexes on `(target_profile, status, id)` and `(task_id, active)`. Extend
  `create_task(..., result_subscriber=None)` so task insertion and the
  `INSERT OR IGNORE` subscription share one write transaction. On the
  idempotency fast path, ensure the explicitly requested subscription in a
  short write transaction before returning the existing ID. Normalize profile
  names with the existing canonicalizer. Do not infer origin or assignee policy
  in this DB function.

- [x] **Step 4: Run the schema/subscription tests and verify GREEN**

  Run the focused file. Expected: subscription tests PASS.

- [x] **Step 5: Write event enqueue and queue lease tests**

  Add tests proving:

  - `completed`, human-action `blocked`, terminal `gave_up`, root
    `returned_for_rework`, and descendant `invalidated_for_rework` enqueue one
    row per subscriber;
  - retryable `timed_out`/`crashed` events and dependency waits do not enqueue;
  - duplicate event delivery is suppressed by `(event_id, target_profile)`;
  - the oldest unexpired lease owned by another watcher blocks newer rows;
  - an expired head lease is reclaimable and a bounded contiguous prefix is
    returned in `id` order;
  - marking delivered requires the lease owner; failed delivery can release a
    lease back to pending;
  - `result_wait_state(profile, exclude_task_id=...)` reports pending queue IDs
    and nonterminal subscribed task IDs without counting a same-profile goal's
    own task.

- [x] **Step 6: Run the focused tests and verify RED**

  Expected: FAIL because actionable enqueue and lease APIs are absent.

- [x] **Step 7: Implement event-coupled enqueue and strict FIFO leasing**

  Make `_append_event` return its integer event ID and, for the fixed actionable
  event-kind set, call a private `_enqueue_result_notifications` before the
  surrounding state transaction commits. Queue payloads contain only task ID,
  event kind, generation, short summary/reason, and rework/control IDs found in
  the event payload. Implement:

  ```python
  lease_result_notifications(conn, *, target_profile, lease_owner,
                             limit=8, lease_seconds=60, now=None)
  mark_result_notifications_delivered(conn, ids, *, lease_owner, now=None)
  release_result_notification_lease(conn, ids, *, lease_owner)
  result_wait_state(conn, target_profile, *, exclude_task_id=None)
  ```

  Leasing uses one `BEGIN IMMEDIATE` transaction: inspect the oldest
  undelivered row; return no rows if its lease is unexpired and owned by
  another watcher; otherwise reclaim the expired head and lease only the
  contiguous eligible prefix, stopping before any unexpired foreign lease.
  State transitions do no pane or Zellij I/O.

- [x] **Step 8: Run DB notification tests and the nearby DB regression suite**

  Run:

  ```bash
  python3 -m pytest tests/hermes_cli/test_kanban_result_notifications.py tests/hermes_cli/test_kanban_db.py -q
  ```

  Expected: PASS.

### Task 2: Apply cross-profile defaults at the CLI boundary

**Files:**
- Modify: `hermes_cli/kanban.py` (create flags and origin resolution)
- Modify: `tests/hermes_cli/test_kanban_result_notifications.py`

- [x] **Step 1: Write CLI policy tests**

  Test `_cmd_create` with `HERMES_KANBAN_ORIGIN_PROFILE` and assert:

  - `planner -> reviewer` subscribes planner by default;
  - `planner -> planner` has no default subscription;
  - same-profile `--notify-origin` subscribes explicitly;
  - cross-profile `--no-notify-origin` opts out;
  - `--notify-profile coordinator` explicitly targets coordinator;
  - no origin environment creates no pane subscription;
  - `--origin-profile PROFILE` applies the same cross-/same-profile default
    policy for a non-pane caller and overrides the environment;
  - `HERMES_KANBAN_RESULT_NOTIFICATIONS=0` disables implicit subscription but
    leaves an explicit `--notify-profile` request available;
  - arbitrary profile names follow the same rules, with no reviewer/owner
    branches.

- [x] **Step 2: Run those tests and verify RED**

  Expected: FAIL because the create flags and resolution helper are absent.

- [x] **Step 3: Implement the minimal create options**

  Add a mutually exclusive `--notify-origin` / `--no-notify-origin` pair with
  default `None`, plus `--notify-profile PROFILE` and
  `--origin-profile PROFILE`. Resolve origin from the explicit flag first and
  `HERMES_KANBAN_ORIGIN_PROFILE` second. Resolve the subscriber as follows:
  explicit notify profile wins; explicit opt-out returns `None`; explicit
  opt-in requires a nonempty origin; otherwise, while
  `HERMES_KANBAN_RESULT_NOTIFICATIONS` is not false-like, subscribe only when
  origin exists and canonical origin differs from canonical assignee. Pass
  only the resolved value to `kb.create_task`.

- [x] **Step 4: Run CLI and DB tests and verify GREEN**

  Run:

  ```bash
  python3 -m pytest tests/hermes_cli/test_kanban_result_notifications.py tests/hermes_cli/test_kanban_core_functionality.py -q
  ```

  Expected: PASS.

### Task 3: Export the logical origin profile from every launcher pane

**Files:**
- Modify: `local/bin/start-kanban.sh` (`build_role_command`)
- Modify: `tests/local/test_start_kanban_designer.py`

- [x] **Step 1: Add failing launcher assertions**

  Extend the generated-command tests to require
  `HERMES_KANBAN_ORIGIN_PROFILE=<logical role>` for Hermes, Codex, CodeWhale,
  DeepSeek/Reasonix, and Claude panes, independent of the underlying agent.

- [x] **Step 2: Run the launcher test and verify RED**

  Run:

  ```bash
  python3 -m pytest tests/local/test_start_kanban_designer.py -q
  ```

  Expected: FAIL because the environment variable is absent.

- [x] **Step 3: Add one common origin environment prefix**

  Compute `origin_profile_env="HERMES_KANBAN_ORIGIN_PROFILE=${role_q}"` once in
  `build_role_command` and include it in every backend's launched environment.
  Do not alter claim assignees or infer origin from the underlying model.

- [x] **Step 4: Run launcher tests and verify GREEN**

  Expected: PASS.

### Task 4: Drain result notifications before other watcher injections

**Files:**
- Modify: `plugins/kanban/base_listener.py` (central watcher loop and idle follow-up)
- Create: `tests/plugins/test_kanban_result_delivery.py`
- Modify: `tests/plugins/test_kanban_idle_continuation.py`

- [x] **Step 1: Write safe-delivery behavior tests**

  Test a minimal `BaseInteractiveListener` subclass and assert:

  - a busy pane leaves the FIFO queue pending and injects nothing;
  - a safe pane leases a bounded ordered batch, injects one single-line
    `[TASK_RESULTS_READY]` message with stable queue/event IDs and
    `[by watcher]`, then marks all batch rows delivered;
  - a known `zellij_inject=False` releases the lease;
  - delivery runs before ordinary ready-task claim and before idle follow-up;
  - a process restart can reclaim an expired lease;
  - result delivery never calls task create/reassign and never waits for the
    publishing caller.
  - `HERMES_KANBAN_RESULT_NOTIFICATIONS=0` leaves the queue untouched and
    performs no result injection, while the default value enables delivery.

- [x] **Step 2: Run delivery tests and verify RED**

  Run:

  ```bash
  python3 -m pytest tests/plugins/test_kanban_result_delivery.py -q
  ```

  Expected: FAIL because the drain method is absent.

- [x] **Step 3: Implement one central result-drain path**

  Add `pump_result_notifications(args, conn, log_path) -> bool`. It targets
  `args.profile`, checks the existing backend-specific safe boundary before
  leasing, refreshes/leases the FIFO batch, formats a compact one-line prompt,
  injects once, and marks delivered only on success. Call it centrally in the
  watcher loop both while an active task is idle and before idle hooks/claim;
  when it returns true, skip every other injection and claim for that tick.
  Keep control-plane supersession higher priority than result notifications.
  Guard the listener drain with the same default-on
  `HERMES_KANBAN_RESULT_NOTIFICATIONS` environment flag used by Task 2. A
  false-like value disables implicit subscription creation and pane delivery;
  explicit subscription requests remain durable and queued until the feature
  is re-enabled, so rollback does not delete result ownership or queue data.

- [x] **Step 4: Write goal-wait throttling tests**

  Extend idle continuation tests to prove a running goal with a nonterminal
  subscribed cross-profile task suppresses the normal two-minute reminder;
  emits at most one waiting-aware insurance reminder per 120 minutes; pending
  or leased result rows also suppress the normal reminder; the goal itself is
  excluded; and after the final result is delivered with no remaining watched
  tasks, the normal goal interval is restored from a reset timer.

- [x] **Step 5: Run goal tests and verify RED**

  Expected: FAIL because wait-state-aware throttling is absent.

- [x] **Step 6: Add the 120-minute waiting state to idle follow-up**

  Add `RESULT_WAIT_GOAL_INTERVAL_S = 120 * 60`. Query
  `kb.result_wait_state` before selecting a goal interval. Seed/reset a
  per-goal waiting timer on state transitions so entering wait does not emit an
  immediate reminder and leaving wait restores the existing daytime/overnight
  schedule. The insurance prompt lists bounded watched task IDs/statuses and
  explicitly says to wait for watcher delivery rather than poll or create a
  continuation card.

- [x] **Step 7: Run watcher and idle regression tests and verify GREEN**

  Run:

  ```bash
  python3 -m pytest \
    tests/plugins/test_kanban_result_delivery.py \
    tests/plugins/test_kanban_idle_continuation.py \
    tests/plugins/test_codex_listener_idle_detection.py -q
  ```

  Expected: PASS.

### Task 5: Integration verification and focused commit

**Files:**
- Modify: `docs/superpowers/specs/2026-08-03-kanban-publisher-result-notifications-design.md` (accepted status only)
- Track: `docs/superpowers/plans/2026-08-03-kanban-publisher-result-notifications.md`

- [x] **Step 1: Run all focused suites together**

  ```bash
  python3 -m pytest \
    tests/hermes_cli/test_kanban_result_notifications.py \
    tests/hermes_cli/test_kanban_db.py \
    tests/hermes_cli/test_kanban_core_functionality.py \
    tests/plugins/test_kanban_result_delivery.py \
    tests/plugins/test_kanban_idle_continuation.py \
    tests/plugins/test_codex_listener_idle_detection.py \
    tests/local/test_start_kanban_designer.py -q
  ```

  Expected: PASS with no warnings attributable to this change.

- [x] **Step 2: Inspect scope and unrelated dirt**

  Run `git status --short`, `git diff --stat`, and focused diffs. Confirm
  `scripts/wechat_inject.py` remains untouched and unstaged.

- [x] **Step 3: Commit only the notification implementation**

  Stage only the files listed in this plan and commit with:

  ```bash
  git commit -m "feat(kanban): notify publishers of task results"
  ```

- [x] **Step 4: Report rollout behavior**

  Report the commit SHA, exact test counts, no-push status, and that existing
  panes need a watcher/launcher restart before new CLI creates inherit
  `HERMES_KANBAN_ORIGIN_PROFILE`. Existing tasks without subscriptions are not
  backfilled.
