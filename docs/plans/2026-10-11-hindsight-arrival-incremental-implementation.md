# Hindsight Arrival-Incremental Implementation Plan

> For agentic workers: use subagent-driven-development for disjoint component implementation and independent review. Follow the approved design at `docs/plans/2026-10-11-hindsight-arrival-incremental-design.md`.

Goal: preserve existing memory, eliminate automatic historical LLM replay, process frozen new-arrival batches, and resume only incomplete stages with truthful success receipts.

Architecture: one bank-scoped local SQLite ledger plus immutable payload files, three reusable components (capture, delivery, local canonical merge), and a small orchestration entrypoint. Existing executors/filters/API/quality gates are reused; production wrappers switch only after isolation verification. Old date-repair paths are not used by default cron/full.

Tech stack: Python stdlib SQLite/json/hashlib/uuid/fcntl, existing requests and REST client, existing local reducers/publish gates; no new package, service, provider or profile loop.

## Global constraints

- Worktree: `/home/wyr/.hermes/hermes-agent-repo/.worktrees/hindsight-arrival-incremental`, branch `feat/hindsight-arrival-incremental`, base `b5b53ffa768`.
- Read worktree AGENTS.md, design, domain skills and existing related implementation before edits. The project has no local/hermes-scripts AGENTS.md or dedicated domain map; approved design plus actual code/runbooks supply those constraints. RepoWise does not index this repo, so native file/symbol search is the documented substitute.
- No production API writes, paid LLM/embedding, old-state migration, service restarts, cron resume, external push, or deletion during component work. Cron daily/weekly are user-authorized paused; stats/Codex snapshots remain enabled.
- Do not touch main, `.env`, live config, other profiles, daily_report changes, package-lock or untracked placeholders. Children do not commit; parent commits reviewed disjoint file groups.
- Test through `HERMES_PYTHON=/home/wyr/miniconda/bin/python scripts/run_tests.sh -j 2 ...`; pytest is absent from both repo venvs, available in this existing interpreter. Explicit test fixtures/probes use Hermes scratch and never production HOME/API. Source tests mock clients, real-path E2E uses loopback mock REST.
- Baseline RED: `test_execute_does_not_update_submit_state_when_async_operation_fails` currently fails because failed operation is swallowed; this is part of the requested fix, not a new regression.
- Defaults stay 12 units / 500000 input chars per round; split pending across rounds rather than expanding history or budgets. Actual retry calls are separately capped and reported.
- Input identity excludes model/label/endpoint/week/path/index provenance; sealed successful input cannot become pending on model/cache changes. Bank and source epoch remain identity boundaries.
- No automatic baseline=now/latest-date success; old unknowns have explicit non-success legacy_review/excluded status.
- Every API state write and publish is read back before completion; bad/empty JSON, async failed/unknown, stale gate, unsafe manifest, pending consolidation and NULL embeddings stay fail-closed.

## Shared integration interfaces

Paths below are relative to `local/hermes-scripts/`. Disjoint implementers may refine implementation details but keep these call boundaries or report a concrete proposed signature correction before parent integration.

`hindsight_incremental_ledger.py` exports `IncrementalLedger(root: Path, bank: str, *, create: bool = False)` and an explicit initializer; opening a missing ledger for a normal run is an error, not auto migration. Methods:

- `initialize_legacy(source_snapshots: list[dict], seed: dict, unknown: list[dict], *, confirm: str) -> dict`: local explicit bootstrap with policy preserve-no-replay, seed hashes and uncompleted legacy inventory; no LLM/network.
- `capture_snapshot(snapshot: dict, *, max_input_chars: int = 60000) -> list[dict]`: atomically capture only unseen new events after source baseline into immutable batches; snapshot schema contains source_key, source_epoch, messages/event identities and checked source hash.
- `list_batches(*, phases: tuple[str, ...] | None = None) -> list[dict]`, `get_batch(batch_id: str) -> dict`, `pending_batches(*, now: float, max_units: int = 12, max_chars: int = 500000) -> list[dict]`.
- `set_phase(batch_id: str, phase: str, *, receipt: dict | None = None) -> dict`: validated monotone durable transition, terminal failed/deferred are not success, source/bank/hash mismatch rejected.
- `defer(batch_id: str, *, error_class: str, next_retry_at: float, attempts: int) -> dict`: preserve last completed stage so later recovery does not regenerate known success.
- `locked()` context manager: one bank-scoped producer, no stale-owner stealing during live work.

Batch dictionaries include bank, batch_id, source_key, source_epoch, phase, immutable payload path/hash, source event IDs, document_id, operation_id, receipts and next_retry_at. Every operation UUID is bound to the exact immutable payload hash before POST. Content stays only in private payload/artifact files.

`hindsight_incremental_sources.py` exports `scan_source_snapshots(hermes_home: Path, *, profile_mode: str = 'all') -> list[dict]`: readonly JSON/Codex snapshot adapter and, where needed for persisted unsnapshotted messages, a readonly source SQLite adapter with epoch/cursor contract. Reuse existing cleaning and source/profile detection. No full SQLite history paid processing, no core persistence changes.

`hindsight_incremental_delivery.py` exports `deliver_batch(ledger, batch: dict, *, client, generate=None, validate=None, retry_policy=None) -> dict`: uses ledger methods above and exact operation receipts. generate is an injected bounded new-input function; no existing generated payload calls it again. Its output includes status/phase/operation IDs/attempts and non-sensitive failure class.

`hindsight_incremental_merge.py` exports `merge_observations(seed: list[dict], delta: list[dict]) -> list[dict]`, `write_generation(root: Path, *, bank: str, parent: dict, observations: list[dict], consumed_batch_ids: list[str]) -> dict`, `validate_generation(manifest_path: Path) -> dict`. Legacy opaque rows remain byte/content-equivalent and provenance-bound; typed new rows preserve distinct conditions/evidence. Manifest enumerates exact artifact files and hashes.

`hindsight_incremental_pipeline.py` supplies initialize/plan/run/status CLI and `run_incremental(config: dict, *, ledger, client, llm, clock, execute: bool) -> dict` used by tests and wrappers. Dependency injection is concrete for tests, not a production feature framework.

## Task 1: Ledger, frozen new-arrival source batches

Files: create `hindsight_incremental_ledger.py`, `hindsight_incremental_sources.py`; tests `tests/test_hindsight_incremental_ledger.py`, `tests/test_hindsight_incremental_sources.py`. Do not modify old manifest/runner/retry/publisher in this task.

- [ ] Search existing SQLite state, atomic writes, source normalizers, message identity/epoch and filtering; reuse rather than clone logic. Confirm dependency management requires no new package.
- [ ] Add behavior tests before implementation. Required relationships:

```python
ledger.initialize_legacy([snapshot_old], seed, unknown, confirm='initialize-hindsight-incremental')
assert ledger.capture_snapshot(snapshot_old) == []
new_batches = ledger.capture_snapshot(snapshot_old_plus_new_message)
assert [event for b in new_batches for event in b['source_event_ids']] == [new_event_id]
assert ledger.capture_snapshot(snapshot_old_plus_new_message) == []
```

- [ ] Run ledger/source tests RED through canonical runner; failures must be missing behavior, not an incidental environment failure.
- [ ] Implement explicit local initialization, bank/epoch binding, event identity and private immutable payloads; preserve old history as legacy, not completed. Atomic source capture commits payload coverage before moving discovery cursor. Open read-only/dry-run without mutating absent state. No filesystem mtime-only success rule.
- [ ] Implement growing-session range capture, source stability checks, restart recovery, same-timestamp distinct events, source epoch/compaction/shrink diagnostics, source bank isolation. New late old event metadata must not reopen old day.
- [ ] Implement monotone stage receipts and pending selection with max units/chars, actual persisted earlier stage; secure paths/permissions/symlink controls and one-bank lock.
- [ ] GREEN and self-review. Return exact APIs, file hashes and commands. Parent integrates only after reading tests/diff; no production migration.

## Task 2: Correct failure classification, async receipts, generated-result reuse

Files: create `hindsight_incremental_delivery.py`, optional small topical `hindsight_offline_retry.py`; modify `hindsight_native_client.py`, `hindsight_session_retain_runner.py`, narrow `call_llm`/error boundary in `offline_hindsight_reflect_consolidate.py`; their focused tests and new `tests/test_hindsight_incremental_delivery.py`. Do not edit ledger/sources/merge/pipeline wrappers.

- [ ] Start with existing failing async-operation test. Fix failed/cancelled/error handling, including parent/children checks and per-batch confirmation; failed records never enter successful state. Preserve successful independent batches on a later failure.
- [ ] Add client retain `operation_id` option using existing request helper. Bind UUID to immutable payload/hash in new delivery, not to volatile run timestamps. Verify UUID/payload mismatch and lost ACK with exact GET reconciliation.
- [ ] Test state boundary independently with a fake ledger implementing shared methods:

```python
first = deliver_batch(ledger, batch, client=client_lost_ack, generate=generator)
second = deliver_batch(ledger, ledger.get_batch(batch['batch_id']), client=client_completed, generate=generator)
assert generator.calls == 1
assert client_lost_ack.operation_id == client_completed.operation_id
assert second['phase'] == 'retained'
```

- [ ] Generated JSON valid and durable before POST; parse/required fields fail before remote writes. Recovery from generated/submitted/unknown preserves payload UUID and input hash; succeeded phases are never repeated.
- [ ] Fix RateLimitError broad capture; permanent400/401 immediately propagate. Add bounded attempts/request timeout/total deadline/actual-call cap and parse both Retry-After forms. After deadline defer the one unit with last phase and next_retry_at rather than infinite sleep or erased progress. Do not add fallback providers or change keys/env.
- [ ] Requests/logs error summaries exclude prompts/secrets; preserve selected request IDs and failure categories without inventing gateway backend causes.
- [ ] Test continuous502,429→success,401,invalidJSON, ACK loss, crash after generated/POST receipt, unrelated successful unit, and per-round call cap. Use scratch and Fake/loopback clients only.
- [ ] GREEN plus self-review; report exact APIs and diff. Parent cross-checks delivery with real ledger once Task1 lands.

## Task 3: Lossless local seed+delta merge and exact candidate manifests

Files: create `hindsight_incremental_merge.py`; tests `tests/test_hindsight_incremental_merge.py`; narrow manifest-selection fix in `hindsight_offline_v2_publish.py` with tests. Do not edit ledger/delivery/sources/orchestration/call_llm.

- [ ] Read existing Observation/build_cards/write index/load_cards publisher and identify reusable components. Old insight-only dedup is not the identity rule for complete long-term state.
- [ ] Define/test typed semantic identity with topic/type/insight/applicability/validity and source contribution hash. Opaque legacy seed is preserved, not guessed from rendered text. Required relationships:

```python
assert merge_observations(seed, []) == seed
merged = merge_observations(seed, [conditional_a, conditional_b])
assert conditional_a in merged and conditional_b in merged
assert merge_observations(merged, [conditional_a]) == merged
```

- [ ] Write immutable generation with exact manifest files/hashes and complete observation/evidence index; compact cards are projections. Preserve old generation and parent seed; no directories cleanup/deletion.
- [ ] Validate manifest bank/root/hash, controls/symlinks, duplicate IDs, missing files/malformed JSONL. Publisher loads exactly declared files, not glob leftover91 from current directory; reject unexpected files or safely exclude according frozen manifest contract.
- [ ] Expose seed export/import pure functions sufficient for parent to connect readonly DB rows; do not perform live SQL or use latest local cards as authoritative baseline. Existing production memory preservation is mandatory.
- [ ] No paid weekly all-history; no active embedding/publish during Task3. Repeat merge/different insertion order and added duplicate generation should have stable semantic result, no identity invalidation by model/label/path.
- [ ] GREEN and self-review; parent verifies exact file selection tests and old publisher compatibility without weakening gates.

## Task 4: Unified incremental orchestration and safe active entrypoints (parent integration)

Files: create `hindsight_incremental_pipeline.py` and focused `tests/test_hindsight_incremental_pipeline.py`; modify `hindsight_memory_pipeline.py`, `hindsight_daily_noagent.py`, `hindsight_weekly_wrapper.py`, `hindsight_pipeline_common.py` and focused tests only after components APIs pass review. For god-file changes keep entry dispatch thin; new state/logic resides in topical modules.

- [ ] Review concrete Task1–3 interfaces and approved spec; resolve signature gaps before wiring. Test-run old focused suites and distinguish pre-existing failures.
- [ ] CLI initialize requires explicit `initialize-hindsight-incremental` confirmation, emits migration manifest with verified sealed/legacy_unknown inventory. Plan/status are read-only. Run on missing migration state fails before any paid action.
- [ ] Both daily/full normal entrypoints route to arrival-incremental runner; no implicit old date-repair/all-history branch. Explicit history repair is never scheduled and requires dedicated confirmation.
- [ ] Capture snapshot fence, select bounded pending, deliver new source batches; build new daily summary only from confirmed new retain/fact contributions tied to those operation IDs. Event dates are attributes; no entire-day DB query from current bank. Successful prior batch never reconstructed.
- [ ] Local merge only confirmed new summaries with frozen parent seed. Persist generated/gated candidate receipts; blocked/failed publish retains existing memory and retries frozen next stage only.
- [ ] Shared bank lock and observation owner for all producers, including full-skip. On exception inspect exact operations/backlog before restore; cleanup_pending and run failure are durable. No producer/unknown queue means not falsely safe; same bank concurrently cannot progress.
- [ ] Stream logs and always write result (completed/no_op/deferred/blocked/cleanup_pending). Optional wiki/digest independently bounded and degraded; legacy pre-clean bypass is not invoked on normal new path.
- [ ] Dependency-injected integration test real modules + loopback REST: initialize old seed, no-op0 calls, one appended source,429/502 recovery, exact ID confirmation, deterministic merge, gate/publish failure isolation, restart-success no paidcalls; bankA→B→A isolation. Direct CLI smoke uses isolated HOME.
- [ ] Independently review spec and code before mark complete. Do not claim delivery from stub/helper tests only.

## Task 5: Packaging, documentation and acceptance (parent)

Files: approved spec/implementation plan, implementation docs plus canonical installer payload references. Bundled skill scripts live outside repository and are updated only after canonical reviewed implementation; their ownership constraints remain respected.

- [ ] Sync approved new rule (no history LLM replay / stage receipts) into operations/hardening runbooks, preserving separate explicit manual-repair safety.
- [ ] Installer arrays contain transitive new ledger/source/delivery/merge/entrypoint dependencies and focused behavior tests. Verify canonical/bundle parity and standalone isolated install with mock REST, no production writes.
- [ ] Focused tests plus full local/hermes-scripts suite through canonical runner; record exact source provenance and pre-existing failures without pretending allgreen.
- [ ] Final independent reviewer checks no-op, sealed stability, late data, false async success, retry deadline/budget, generation identity, cleanup/window ownership, manifest-only publication and gate freshness. Parent reproduces key acceptance cases.
- [ ] Per-feature commits only task paths; preserve concurrent main changes. Merge/cutover and production source-state migration require explicit confirmed scope with backups; external push requires confirmation. Until that gate, report tested worktree not deployed runtime.

## Task 6: Authorized production cutover and incremental smoke (after review)

- [ ] Recheck actual API/queue/source/seed and worktree revision, no writers. Export/backup existing ledger/progress/canonical seed read-only, verified by IDs/hashes and exact bank.
- [ ] Explain pending56 consolidation and exact cleanup/migration/call-budget scope to user; get authorization before PATCH, source-state migration or paid smoke.
- [ ] Deploy reviewed source, local preserve-no-replay migration with explicit legacy_unknown/exclusion inventory; do not move success cursor over unknowns. Verify same deployment paths/bundle and seed contents.
- [ ] One smallest real new-input incremental run within existing12/500k/call cap. Verify remote operation/child success, native drain, exact gate/published candidate readback if eligible; then a no-op repeat must make0 paidcalls.
- [ ] Restore only the previously paused daily/weekly cron after newpath verified; read back exact enabled states and preserve stats/Codex jobs. If blocked, retain pause and state precise gate; do not resume old full script as fallback.

## Completion criteria

An implementation is not done merely because this plan exists. Parent must show tests and real-path scratch output, independently approved integration, verified artifact/seed invariants, explicit deployed-or-not status, actual paid/no-op counters, safe observation cleanup and truthful stage success. If production gate is not yet authorized, continue independent implementation and acceptance work, then report that exact boundary rather than fabricate a completed production run.
