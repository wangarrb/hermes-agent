# Kanban Injection Reliability Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Codex/Hermes Kanban injection fail closed when the pane is stale, confirm that the TUI consumed the prompt before acknowledging delivery, and prevent composer/retry/input races.

**Architecture:** Keep the durable claim/lease flow, but split the shared terminal operation into (1) validated text write and (2) one raw-CR submit. Every injection carries the listener's expected pane-role prefix; the helper validates a live non-plugin pane before each write and catches transport failures. Backend hooks return `confirmed`, `known_unsubmitted`, or `unknown` using the exact injected marker plus a pre-write composer snapshot. Only `confirmed` updates task/queue delivery. Both failure states produce a fenced reclaim/release transition within the bounded post-confirm window (at-least-once retry; no task is left running forever), and never emit a success ACK. Codex and Hermes parse only their current composers and confirm a marker transition.

**Tech Stack:** Python 3, subprocess, Zellij 0.44.1, Codex/Hermes TUI screen dumps, SQLite-backed Kanban leases, pytest.

---

## Chunk 1: Shared transport and delivery contract

### Task 1: Add fail-closed Zellij primitives

**Files:**
- Modify: `plugins/kanban/base_listener.py:323-423,840-850,1889-1948`
- Modify for compatibility: `plugins/kanban/claude_listener/claude_kanban_interactive.py:173-190`, `plugins/kanban/deepseek_listener/deepseek_kanban_interactive.py:616-660`
- Tests: `tests/plugins/test_kanban_injection_transport.py` (create)

- [ ] Write failing tests for invalid pane IDs returning rc=0, missing panes, stale/reassigned/foreign-role pane identity, write timeout/OSError, LF/CR input rejection, and exact one-text/one-CR command sequencing.
- [ ] Run the new focused tests and verify they fail for the current helper.
- [ ] Implement pane enumeration/identity validation (`expected_pane_prefix` is required by listener callers). Normalize `title.strip().casefold()` and require an exact token boundary after the backend prefix (`end`, whitespace, or `[`), including the role-preserving pause form (`codex-kanban [pause …]` / `hermes-kanban [pause …]`); titles such as `codex-kanban-evil` are foreign. When available, also require matching `terminal_command`/`pane_command` role tokens. Update the control rename call sites to preserve that prefix. Add bounded subprocess timeouts, explicit `zellij_submit_enter`, and correlation-aware transport logging. `zellij_inject` must reject control/newline bytes and must not treat an empty/unknown/foreign-role pane as success.
- [ ] Change follow-up/API retry callers and existing Claude/DeepSeek post-hooks to use `zellij_submit_enter` rather than recursively calling `zellij_inject("\r")`; make all transport exceptions return false and leave durable work pending. Preserve their current queue semantics and add/adjust their command-sequence tests.
- [ ] Run focused transport tests and the existing injection safety/control/result suites; expected all pass.
- [ ] Commit: `fix(kanban): fail closed on zellij injection transport`.

### Task 2: Make semantic delivery explicit

**Files:**
- Modify: `plugins/kanban/base_listener.py:840-850,1530-1660,1889-1948`
- Tests: `tests/plugins/test_kanban_delivery_contract.py` (create)

- [ ] Add failing tests proving transport-true plus `known_unsubmitted` reclaims a task, transport-true plus `unknown` does not leave a running task forever, and all task/control/result paths invoke the post-confirm hook before marking delivery.
- [ ] Run them to verify current code marks delivery despite the failed semantic hook.
- [ ] Add a tri-state post-injection contract with backward-compatible `confirmed` default for non-Codex/Hermes backends. The hook receives `injected_marker` and `pre_write_composer`; it returns `confirmed` only after observing marker-present→marker-gone or a live busy transition, `known_unsubmitted` when the marker remains after retries, and `unknown` on probe timeout/empty unsupported screen. For task claims, both non-confirmed states use the claim fence to reclaim immediately with an injection-failure reason; for control/result leases, both release the lease. Check the fenced CAS result and log/alert a `delivery_reclaim_race` if it fails, so a concurrent state change cannot silently leave the task running. This gives a bounded at-least-once retry and prevents a permanently running claim.
- [ ] Add structured log events with task/control/result correlation, run/generation, pane ID/prefix, and states `transport_accepted`, `submit_sent`, `delivery_confirmed`, `delivery_requeued`, `delivery_unknown`.
- [ ] Run the focused contract and existing Kanban lifecycle tests; commit `fix(kanban): require semantic injection acknowledgement`.

## Chunk 2: Codex and Hermes TUI behavior

### Task 3: Harden Codex composer and submit confirmation

**Files:**
- Modify: `plugins/kanban/codex_listener/codex_kanban_interactive.py:49-274`
- Tests: `tests/plugins/test_codex_listener_idle_detection.py`

- [ ] Add RED cases for a bare `›` idle prompt, non-empty draft rejection, transcript text that must not count as queued composer input, and terminal retry failure after the maximum attempts.
- [ ] Implement composer-only detection, accept bare `›` while rejecting typed drafts, and make `on_post_inject(injected_marker, pre_write_composer)` return the tri-state result after bounded confirmation; inspect the current composer rather than arbitrary transcript tail. Replace existing tests that allow a stable non-empty draft after 30s with fail-closed draft-protection assertions.
- [ ] Run Codex listener and transport tests; commit `fix(kanban): confirm Codex composer submission`.

### Task 4: Add Hermes composer and submit confirmation

**Files:**
- Modify: `plugins/kanban/hermes_listener/hermes_kanban_interactive.py:120-340`
- Tests: `tests/plugins/test_hermes_listener_delivery.py` (create)

- [ ] Add RED cases for Hermes bare/role idle prompts, non-empty composer protection on claim/result/control paths, successful submit, and bounded failed submit.
- [ ] Implement Hermes `composer_input_text`, strict idle reuse for every injection path, and an `on_post_inject(injected_marker, pre_write_composer)` bounded submit probe with explicit tri-state results for claim/control/result paths.
- [ ] Narrow API-error matching to the live error box rather than stale scrollback and preserve retry state keyed by task/error kind with deterministic eviction/reset tests.
- [ ] Run Hermes listener, control/result, and transport tests; commit `fix(kanban): confirm Hermes composer submission`.

## Chunk 3: Input and generation safety

### Task 5: Sanitize injected payloads and bind prompts to runs

**Files:**
- Modify: `plugins/kanban/base_listener.py:284-346,356-370,1420-1450,1530-1660`
- Tests: `tests/plugins/test_kanban_injection_safety.py`, `tests/plugins/test_kanban_prompt_generation.py` (create)

- [ ] Add RED tests for titles containing LF/CR/control bytes and for two generations of one task producing distinct prompt paths.
- [ ] Implement strict single-line sanitization/rejection at the shared boundary and include run/generation in prompt filenames; update `_mark_prompt_superseded`, control/result prompt references, and readers so old `task-<id>.md` paths remain readable during migration.
- [ ] Run all injection/listener tests; commit `fix(kanban): isolate prompt generations and sanitize input`.

## Chunk 4: Integration and live verification

### Task 6: Verify runtime behavior and integrate

**Files:**
- Read: `local/bin/kanban-watcher-supervisor.py`, current board logs, Zellij pane inventory.

- [ ] Run the complete focused suite: transport, delivery contract, Codex, Hermes, control, result, prompt-generation, and supervisor tests.
- [ ] Run a non-mutating invalid-pane smoke proving the helper now returns false and logs a failure.
- [ ] Inspect watcher/pane identity counts and existing logs for false-positive `claimed+injected` patterns.
- [ ] Merge the reviewed feature branch into the current main without touching pre-existing user changes.
- [ ] Restart only affected panes if needed and confirm one watcher per pane, a current pane identity, composer transition, and requeue on failed confirmation.
- [ ] Report test counts, commit SHAs, runtime counts, and any residual limitations.
