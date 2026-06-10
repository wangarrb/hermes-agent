---
name: hindsight-consolidation-operations
description: Use when operating, monitoring, tuning, or debugging Hindsight offline/session consolidation drains, including LLM-bound backlogs, safe restarts, read-only status snapshots, and production-safe parallelization.
version: 1.1.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hindsight, consolidation, offline-processing, drain, docker, llm, postgres, observability]
    related_skills: [hindsight-local-deployment]
---

# Hindsight Consolidation Operations

## Pipeline Full Flow Steps (2026-05-27 confirmed)

The `hindsight_memory_pipeline.py full` mode executes these steps in order:

1. preflight (environment checks)
2. status (DB/API health)
3. queue_drain_before_daily
4. session_manifest (scan new sessions, generate manifest)
5. retain_session (incremental retain → extract source facts)
6. daily_reflect (offline daily consolidation, calls LLM)
7. native_consolidation_drain_after_daily
8. v2_rebuild (daily → canonical observations)
9. native_consolidation_drain_before_weekly
10. weekly_reflect (offline weekly consolidation, calls LLM)
11. native_consolidation_drain_after_weekly
12. v2_rebuild (weekly → canonical observations)
13. conflict_audit (conflict detection)
14. repair_zone_proposals (repair proposals, skippable with --skip-repair-zone)
15. proposal_review (proposal LLM review, skippable with --skip-proposal-review)
16. wiki_auto_maintenance (requires --include-wiki flag)

Key flags: `--execute --confirm run-hindsight-pipeline`, `--include-wiki`, `--skip-daily`, `--history all`, `--no-wait-native-consolidation`.

Failure recovery: earlier steps may have completed successfully; only re-run the failed step rather than starting from scratch. Eval step 502 errors are usually due to container restart (embedding model ~25s load time); wait for `/health` healthy + recall test before retrying.

## Overview

This skill is the long-running operations guide for Hindsight offline/session consolidation. It is narrower than `hindsight-local-deployment`: use it when the question is about offline drains, observations, LLM-bound consolidation, async operation queues, status accounting, safe restarts, and production-safe parallelization.

Core idea: keep the workflow read-only until a human explicitly accepts a mutation. Consolidation drains can run for hours; a flat counter is not always a stall, and an impatient restart can turn healthy work into partial/failure state.

## When to Use

Use this skill when:

- Hindsight offline/session consolidation is slow, stuck, backlogged, or interrupted.
- `async_operations` contains `consolidation`, `retain`, or observation-processing work.
- You need to decide whether it is safe to restart/recreate the Hindsight service.
- You need to tune batch size, LLM batch size, official `max_memories_per_round`, consolidation recall budget/source-fact caps, external wrapper `parallel_batches`, or recall/search fanout.
- You need progress/status for daily reports or long-running monitors.
- You need to distinguish drain completion from conflict/lineage quality gates.
- You need production-safe repair/publish boundaries after offline processing.

Use `hindsight-local-deployment` for installation, provider setup, session retain manifests, proposal-review packaging, and broader memory-provider governance.

## Non-Negotiable Safety Rules

1. **Do not change Hindsight LLM model configuration (llm_profile, base_url, api_key) without explicit user confirmation.** This includes pipeline_config.json, .env overrides, and container env vars. Changing models mid-pipeline can cause 403/502 failures, incorrect consolidation, or cost overruns. Always ask first.
2. Do not edit `.env` from this workflow.
3. Do not print API keys, tokens, passwords, bearer strings, or raw credential files.
3. Treat restart/recreate as a potentially disruptive operation. If active/pending work exists or status is unknown, wait for idle or get explicit human approval.
4. Prefer read-only status snapshots before any mutation.
5. Do not claim “done” from one counter. Split status into drain completion, quality/conflict gate, and runtime auto-retain/config gate.
6. Do not convert proposal output into production retain/merge. Proposal-only → local review → optional advisory LLM → human go/no-go → separate rollback/quarantine plan.
7. Keep LLM parallelism bounded by recall/search fanout and provider limits, not by wishful speedup.

## Packaged Read-Only Tool

This skill packages a stdlib-only read-only status helper and installer:

```bash
bash scripts/install_hindsight_consolidation_tools.sh
python3 scripts/hindsight_consolidation_status.py --json
```

Typical installed usage:

```bash
python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --json
```

It checks:

- `/health`
- `/v1/<tenant>/banks/<bank>/stats`
- v0.6.1 `/operations?exclude_parents=true` counts and recent samples without payloads
- v0.6.1 `/stats/memories-timeseries` and `/audit-logs/stats` for report/anomaly context
- optional PostgreSQL `async_operations` counts, if `psql` is available

Important properties:

- no database writes,
- no Hindsight mutation APIs,
- no `.env` edit,
- output is JSON for cron/daily-report integration,
- defaults are environment/config friendly: `HINDSIGHT_API_URL`, `HINDSIGHT_TENANT`, `HINDSIGHT_BANK`, `HINDSIGHT_PSQL`, `HINDSIGHT_PGHOST`, `HINDSIGHT_PGPORT`, `HINDSIGHT_PGUSER`, `HINDSIGHT_PGDATABASE`.

If `psql` is unavailable, use API-only mode:

```bash
python3 scripts/hindsight_consolidation_status.py --skip-psql --json
```

For old Hindsight versions without the operations API, add `--skip-operations-api` and keep psql enabled if available.

## First Response Workflow

When diagnosing an offline drain, do this order:

1. Read-only health/status:
   ```bash
   python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --skip-psql --json
   ```
2. If the helper is not installed, use API-only checks:
   ```bash
   curl -s http://127.0.0.1:8888/health
   curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats
   curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?exclude_parents=true&limit=5'
   ```
3. Check whether active work exists before restart:
   - `processing`/`running` > 0 means active work.
   - `pending`/`queued` > 0 means work remains.
   - Unknown async status is not proof of idle.
4. Inspect recent logs/monitor snapshots before changing code.
5. **Detect observation gaps caused by 'refusing to switch'.** Check the latest daily pipeline log for these two patterns:
   - `refusing to switch to minimax with existing queue` — retain ran without observations
   - `enable_observations: false` after retain — confirm observations were skipped
   Even when `pending_consolidation=0` and `failed_consolidation=0`, the pipeline may have silently skipped observations. Cross-reference `total_observations` change: if no increment after a daily run that had session retain work, observations were likely skipped. The fix is to run a `full` pipeline or explicitly POST `/consolidate` with observations enabled.
5. If a foreground pipeline/orchestrator timed out, do not equate that with Hindsight failure. Check whether native operations, logs, observation counts, or consolidation batches are still advancing; if yes, use a watchdog/wait-for-idle + submit-state reconciliation + resume pattern instead of restart/resubmit.
6. If V2 publish appears to hang after files are written, inspect `docker top hindsight` and the process tree. A high-CPU `docker exec` running `SentenceTransformer`/`BAAI/bge-m3` indicates embedding computation is still active; wait unless CPU is idle for a sustained period or stats stop changing.
7. If `pending_consolidation > 0` and `processing_ops == 0` persist without advancing, suspect the silent consolidation stall: after a retain phase restores `enable_observations=false`, new source facts never get auto-consolidated. The pipeline gate (`hindsight_wait_native_consolidation.py`) can self-heal via its `--trigger-on-stall` (default on) which POSTs `/consolidate` after 2 idle polling cycles. If you are monitoring a live drain outside the pipeline, manually POST `/consolidate` to kick-start background consolidation before deep-diving other causes.
7. Only then decide between: keep watching, tune next run, wait-for-idle restart, targeted API repair, or confirmed reset.

### Failed Operation Triage

When the user asks "do these failed ops need handling?", follow `references/failed-op-triage.md`. The short answer: check `pending_consolidation` and `failed_base` from bank stats. If both are 0 and no pipeline is actively stuck, failed ops are historical noise and do not block anything. Classify errors by message: FK violations, `generator didn't stop`, `Failed to search memories`, and `datetime serialization` are all non-blocking with exhausted retries.

## Completion Gates

Report three gates separately:

1. Native drain/consolidation gate
   - no active/pending consolidation/retain work, or clearly explained active work;
   - unconsolidated source backlog is zero or decreasing;
   - failed source items are zero or triaged.
2. Quality/conflict gate
   - conflict/lineage audit has no blocking issues, or outputs a proposal-only review packet.
3. Runtime/config gate
   - `auto_retain=false` when expected;
   - provider/tuning restored after paid/offline windows;
   - daily report has accurate LLM call/token and stage counters.

Do not collapse these into “Hindsight is done”.

## Tuning Principles

### Batch and fanout

Stable production default (2026-05-14 onward): 64x8 with decoupled parallel patch

```text
consolidation_batch_size=64
consolidation_llm_batch_size=8
max_memories_per_round=64
max_memories_per_job=64
parallel_batches=8
consolidation_recall_max_concurrent=60
consolidation_llm_max_concurrent=8
consolidation_recall_budget=low
source_facts_max_tokens=4096
source_facts_max_tokens_per_observation=256
```

Fallback for DB-pool-constrained or no-patch environments: 20x3

```text
consolidation_batch_size=20
consolidation_llm_batch_size=20
max_memories_per_round=60
parallel_batches=3
estimated recall/search fanout ≈ 60
```

On Hindsight v0.6.x, prefer official bounded-fanout defaults: `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET=low`, `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096`, `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION=256`, and `HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA=false`. Do not treat old `*_MAX_MEMORIES_PER_JOB` or `parallel_batches` as upstream-supported runtime keys.

If jobs time out before LLM output, suspect recall/search DB pressure rather than provider latency. Estimate fanout as:

```text
llm_batch_size * active parallelism * observation-scope parallelism
```

Prefer balanced configurations such as `20x3` or `25x2` over raw high parallelism.

### Worker slot semantics

For Hindsight v0.6.1, do not equate total worker slots with same-bank consolidation parallelism.

- `HINDSIGHT_API_WORKER_MAX_SLOTS` is total task capacity.
- `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS` is a per-operation reservation; upstream default is 2.
- Local paid/offline wrappers may deliberately set consolidation slots to 1 while retain concurrency is 8, yielding `WORKER_MAX_SLOTS=9` and `WORKER_CONSOLIDATION_MAX_SLOTS=1`. This is conservative, not an upstream requirement.
- A reasonable normal-runtime setting is consolidation slots 2 with worker max slots equal to retain concurrency plus consolidation slots, e.g. 8 + 2 = 10.
- Increasing slots does not speed a currently active single-bank consolidation: `submit_async_consolidation` uses `dedupe_by_bank=True`, and the worker claim query excludes banks already processing consolidation. More slots help multiple banks or mixed queued tasks, not one active `hermes` job.
- Change slot settings only after idle or with explicit interruption approval, because it normally requires container restart/recreate.

### Decoupled native-consolidation concurrency

The parallel consolidator patch is production-default in this environment. Same-bank consolidation is serialized in upstream, so worker slots mostly help multiple banks or mixed queues. The local `patch_hindsight_consolidator_parallel.py` patch changes the `for llm_batch in llm_batches` serial loop into `asyncio.create_task`-driven wave scheduling — this is what enables 8-way LLM concurrency for all consolidation operations, not just large backlogs. Writes remain serialized.

Production default profile (active since 2026-05-14, set via `hindsight_minimax_import.py normal-local`):

```text
HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE=64
HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=8
HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=64
HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES=8
HINDSIGHT_API_CONSOLIDATION_RECALL_MAX_CONCURRENT=60
HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT=8
HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=2
HINDSIGHT_API_WORKER_MAX_SLOTS=10
```

Default adaptive 429 behavior in the patched consolidator:

```text
On provider 429/rate-limit:
1. release the active LLM slot;
2. reduce live LLM concurrency by half for future acquisitions, down to 1;
3. sleep HINDSIGHT_API_CONSOLIDATION_429_BACKOFF_SECONDS (default 300s);
4. retry the same batch without consuming the normal bad-input attempt budget.
```

Repeated 429s after each cooldown continue halving (8→4→2→1). 429s that arrive inside the same cooldown join the same sleep window and do not double-halve immediately. Override knobs:

```text
HINDSIGHT_API_CONSOLIDATION_429_BACKOFF_SECONDS=300
HINDSIGHT_API_CONSOLIDATION_429_MAX_RETRIES=0   # 0 = unlimited 429 cooldown/retry loop
```

The old manual `consolidation-half-downgrade` command is only an emergency/operator preview path; the default behavior is now automatic inside the running consolidation loop once the patched module is loaded.

Expected log signatures:

```text
[CONSOLIDATION] ... parallel_batches=8 llm_limit=8 recall_limit=60 llm_batch_size=8
[CONSOLIDATION] ... limits: batch_parallel=8 llm=8 recall=60
```

The patch bounds recall/search, LLM calls, and local batch tasks separately, while serializing observation writes and `consolidated_at` commits. Monitor `pending_acquires`, DB waits, `Failed to search memories`, asyncpg timeouts, and exact duplicate observations after rollout. If DB pressure appears, reduce `PARALLEL_BATCHES` or `RECALL_MAX_CONCURRENT` before reducing LLM provider max.

**Pre-fetch pipeline (since 2026-05-14):** the patched consolidator now overlaps next-round DB fetches with in-flight LLM waves. When the current wave of batches starts running, a background `asyncio.create_task` fetches the next round's unconsolidated memories so they're ready immediately when the current wave finishes. This eliminates the serial fetch→wait→LLM→fetch→wait pattern and is visible in logs as `rate_limit_backoff=300s` alongside the existing `parallel_batches=8 llm_limit=8` signature.

### LLM-bound tail

For slow final-tail consolidation:

- verify active logs before declaring a stall;
- compare provider timeouts/rate limits;
- switch provider only after idle or with explicit approval;
- keep queue monitors tolerant of DB connection pressure.

### Writes

If implementing parallel consolidation in Hindsight internals:

- parallelize LLM calls cautiously;
- serialize database mutation paths;
- mark source units processed/failed exactly once;
- use dedupe/row-level locks for observation create/update/delete;
- make monitor loops survive temporary `too many clients already` rather than crashing.

## Observability and Daily Reports

Daily reports should not infer cost and health from one state DB. Use explicit Hindsight/offline pipeline stats where possible.

A durable statistics interface should include, per bank/run/stage:

- LLM call count and token usage;
- created/updated/deleted observation or document counts;
- conflict count;
- pending-human-review count;
- exception/failure count;
- retry/rate-limit count;
- stage durations and active/pending/completed/failed operation counts.

Use `references/hindsight-consolidation-telemetry-contract.md` as the target contract when extending Hindsight or daily-report integrations.

## v0.6.1 API Repair and Operation Controls

Prefer official targeted controls before DB edits:

```bash
# inspect failed child operations
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&limit=20'

# inspect one operation without payload first
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations/<operation_id>?include_payload=false'
```

Mutation boundaries:

- Retry one reviewed failed operation through `POST /operations/{id}/retry`; confirm token `retry-hindsight-operation`.
- Cancel/delete one reviewed pending operation through `DELETE /operations/{id}`; confirm token `delete-hindsight-operation`.
- Recover failed consolidation through `POST /consolidation/recover`; snapshot/export first and confirm `recover-hindsight-consolidation` in scripts.
- Reprocess one document through `POST /documents/{id}/reprocess`; confirm `reprocess-hindsight-document` in scripts.

Do not use these as broad cleanup loops. Execute one scoped repair, then re-run status, recall smoke, and audit/conflict checks.

## Safe Restart Pattern

**Standard practice: direct restart, recover stuck ops after.**

The established workflow in this environment does NOT wait for the queue to drain. `patch_json_parser_and_restart()` and `patch_hindsight_container_and_restart()` do `docker restart hindsight` immediately — no queue check, no idle wait. This is faster and the recovery is mechanical.

Only after restart if needed:

1. If a consolidation op was mid-flight, it will be stuck in `processing` state after restart. The worker's `dedupe_by_bank=True` mechanism prevents a new consolidation from being picked up while the bank appears to have an active op. Mark it as `failed` via asyncpg to unblock:

```python
import asyncpg, asyncio
async def unstick():
    conn = await asyncpg.connect("postgresql://hindsight@127.0.0.1:5432/hindsight")
    await conn.execute(
        "UPDATE async_operations SET status='failed', completed_at=NOW(), "
        "error_message='container restarted while op was in-flight' "
        "WHERE operation_id=$1 AND status='processing'",
        'STUCK_OPERATION_UUID'
    )
    await conn.close()
asyncio.run(unstick())
```

Or use the API endpoints (preferred when they work):
```bash
curl -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/consolidation/recover' \
  -H 'Content-Type: application/json' -d '{"confirm":"recover-hindsight-consolidation"}'
```

2. Stop external drain drivers before restarting so they don't enqueue new work.

3. After restart, wait for `/health` healthy.

4. Verify new code loaded (check disk vs process mtime or probe for expected markers).

5. Submit new consolidation if auto-re-queue didn't fire:
```bash
curl -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/consolidate' \
  -H 'Content-Type: application/json' -d '{}'
```

6. Monitor backlog/observations advancing.

**The idle-watcher approach** (wait for `processing==0` then restart) is only used when the user explicitly chooses conservative handling. In practice, the re-queue gap between consolidation rounds is sub-second and extremely hard to catch with polling.

## References

Load only the needed reference:

- `references/parallel-consolidation-drain.md` — safe LLM-parallel consolidation and idle-restart rollout.
- `references/recall-fanout-balanced-tuning.md` — tune by recall/search fanout, not raw parallelism.
- `references/llm-batch-sizing.md` — quality-first LLM batch-size limits.
- `references/connection-pool-pressure.md` — DB pool saturation and monitor survival.
- `references/log-based-drain-watcher.md` — non-DB watcher when psql probes add pressure.
- `references/log-based-pool-cap-idle-restart.md` — staged DB pool cap with idle restart.
- `references/direct-restart-vs-poll-waiting.md` — when to use direct restart vs poll-waiting for patch application during active consolidation.
- `references/prefetch-pipeline-overlap.md` — background DB fetch during LLM waves eliminates inter-round delay from fresh DB queries.
- `references/tail-slowdown-and-pool-cap.md` — final-tail slowdown diagnosis and safe pool cap handling.
- `references/provider-tail-switching.md` — safe provider/model switch after idle.
- `references/hindsight-recall-is-not-evidence.md` — hindsight recall/reflect is not ground truth; verify against actual system state.
- `references/monitor-window-interpretation.md` — flat backlog vs live consolidation interpretation.
- `references/drain-complete-vs-quality-gates.md` — separate drain, conflict, and config gates.
- `references/consolidation-benchmarks.md` — throughput benchmarks from 2026-05-14 validation run (1754 memories, 8-way parallel, 2h20m, 0 429s).\n- `references/container-recreate-log-loss.md` — docker `rm -f` + `run` destroys log history; `daily_stats.py` misses LLM tokens after recreate; mitigations.
- `references/consolidation-benchmarks.md` — throughput benchmarks from 2026-05-14 validation run (1754 memories, 8-way parallel, 2h20m, 0 429s).
- `references/restart-stuck-op-recovery.md` — asyncpg recipe for recovering stuck `processing` operations after `docker restart` without waiting for drain.
- `references/orphan-bypass-psycopg2-fallback.md` — psycopg2 / psql CLI fallback chain for orphaned consolidation bypass, including `consolidation_failed_at` reset and retry limits.
- `references/orphaned-consolidation-psycopg2-fallback.md` — orphaned consolidation after container restart + psycopg2 missing in hermes-agent venv causes infinite pipeline stall; 3-layer fix (install psycopg2, psql CLI fallback, bounded bypass retries).
- `references/orphaned-consolidation-units.md` — Orphaned `memory_units` with `consolidated_at=NULL` after consolidation completes: detection, `fix_orphaned_consolidation.py`, auto-bypass in wait loop, and safety guard against bypassing during active work.
- `references/full-pipeline-run-and-recall-smoke.md` — full pipeline invocation, pitfalls, recall smoke verification (API JSON control-char issue), and post-run data reference.\n- `references/failed-op-triage.md` — classify failed Hindsight operations by error type, decide if blocking, decide if cleanup is needed.

## Pipeline Execution Verification

After running `hindsight_memory_pipeline.py daily|full --execute`, verify completion by checking three independent signals:

```bash
# 1. Pipeline log: all steps returncode=0
tail -50 /home/wyr/.hermes/logs/hindsight-offline-pipeline/latest.log | grep -E '"status": "ok"' | wc -l

# 2. Hindsight stats: no pending/failed operations
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats | python3 -c 'import json,sys;d=json.load(sys.stdin);print(f"docs={d[\"total_documents\"]} obs={d[\"total_observations\"]} pend={d[\"pending_consolidation\"]} fail={d[\"failed_consolidation\"]}")'

# 3. Recall smoke: query known content from the window
curl -s -X POST http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall \
  -H 'Content-Type: application/json' -d '{"query":"[known term from recent sessions]", "limit":3}' | \
  python3 -c 'import json,sys;d=json.load(sys.stdin);print(f"hits={len(d.get(\"results\",[]))}")'
```

If pipeline log shows OK but stats delta is zero, check whether the session manifest step found any new records (pipeline log → `build_session_manifest` → `records` field). Zero records means there were no new Hermes sessions to ingest.

## Orphaned Consolidation Cleanup Safety

After a container restart, `memory_units` with `consolidated_at=NULL` and no matching `async_operations` can block `wait_native_consolidation`. The fix script `fix_orphaned_consolidation.py` and the inline fallback `_fix_orphaned_via_db()` in `hindsight_wait_native_consolidation.py` handle this.

**Critical pitfall: never bypass ALL unconsolidated rows while consolidation is actively running.** The `--bypass` flag marks `consolidated_at=NOW()` on matching rows. If a consolidation worker is mid-flight processing those same rows, the bypass will mark them as done before the worker finishes — the worker then skips them, and those facts are never actually consolidated (no observations created, no links formed).

**Safe pattern: always check for active/pending consolidation operations first:**

```python
cur.execute(
    "SELECT COUNT(*) FROM async_operations "
    "WHERE bank_id=%s AND operation_type='consolidation' "
    "AND status IN ('processing','pending')",
    (bank,),
)
if cur.fetchone()[0] > 0:
    # SKIP — consolidation is running, do not touch rows
    return
```

Both `fix_orphaned_consolidation.py` (as of 2026-05-26) and `_fix_orphaned_via_db()` now include this guard. The `--force` flag exists for emergencies but should never be used during active consolidation.

**When to run the cleanup:**
- After a confirmed idle state (no processing/pending consolidation ops)
- As a pre-check in `hindsight_wait_native_consolidation.py` before entering the wait loop
- Never as a cron or periodic task while consolidation might be running

## Common Pitfalls
## Orphaned Consolidation Bypass (2026-05-29)

After container restart, `memory_units` may have `consolidated_at=NULL` with no matching `async_operations`, causing `pending_consolidation` to block indefinitely. The wait script (`hindsight_wait_native_consolidation.py`) handles this with a 3-stage approach:

1. **Stage 1**: POST `/consolidate` to trigger the worker
2. **Stage 2**: If still stalled, run orphan bypass via `_fix_orphaned_via_db()`
3. **Stage 3**: If bypass also fails after 3 attempts, give up and let timeout handle it

**`_fix_orphaned_via_db()` has 3-level fallback for DB access**:
- Level 1: `psycopg2` (preferred, now installed in hermes-agent venv)
- Level 2: `psql` CLI (found via `shutil.which` or `~/.hindsight-docker/installation/18.1.0/bin/psql`)
- Level 3: `docker exec hindsight psql` (last resort)

**Critical**: bypass must also reset `consolidation_failed_at=NULL` for failed consolidation units, not just mark unconsolidated units as consolidated. Otherwise `failed_consolidation` stays non-zero and can block the pipeline if `--block-on-failed-consolidation` is used.

**Hermes-agent venv vs miniconda**: Pipeline scripts run under `~/.hermes/hermes-agent/venv/bin/python` (3.11), not miniconda (3.13). After Hermes upgrades that recreate the venv, `psycopg2-binary` must be reinstalled. The psql CLI fallback exists precisely because psycopg2 may be missing.

**Manual cleanup commands** (when scripts aren't enough):
```sql
-- Find orphaned units
SELECT fact_type, count(*) FROM memory_units
WHERE consolidated_at IS NULL AND consolidation_failed_at IS NULL
  AND fact_type IN ('experience','world') GROUP BY fact_type;

-- Bypass: mark as consolidated
UPDATE memory_units SET consolidated_at=NOW()
WHERE bank_id='hermes' AND consolidated_at IS NULL AND fact_type IN ('experience','world');

-- Reset failed flags
UPDATE memory_units SET consolidation_failed_at=NULL
WHERE bank_id='hermes' AND consolidation_failed_at IS NOT NULL;

-- Cancel stuck processing operations
UPDATE async_operations SET status='cancelled', completed_at=NOW()
WHERE status='processing' AND operation_type='consolidation';
```

## Orphaned Consolidation Fix (psycopg2 / psql CLI / docker exec)

After container restart, `memory_units` may have `consolidated_at=NULL` with no matching `async_operations`, causing `pending_consolidation` to block indefinitely. The pipeline's `hindsight_wait_native_consolidation.py` and `fix_orphaned_consolidation.py` handle this, but they need database access.

**Three-tier fallback** (implemented in `hindsight_wait_native_consolidation.py`):

1. **psycopg2** (preferred) — needs `psycopg2-binary` installed in the venv that runs the pipeline.
2. **psql CLI** — looks for `psql` in PATH, then `~/.hindsight-docker/installation/18.1.0/bin/psql`.
3. **docker exec** — `docker exec hindsight psql -U hindsight -d hindsight -c "..."`.

**Critical pitfall**: The offline pipeline runs under the Hermes-agent venv (`~/.hermes/hermes-agent/venv/bin/python`), NOT the system python or miniconda. If psycopg2 is missing in that venv, orphan bypass silently fails every cycle, and the pipeline loops forever waiting for `pending_consolidation=0` that never arrives. Install with:

```bash
~/.hermes/hermes-agent/venv/bin/pip install psycopg2-binary -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**Bypass retry limit**: The wait script caps bypass attempts at `MAX_BYPASS_ATTEMPTS=3` to prevent infinite loops when all database paths fail. After 3 failed attempts, it stops retrying and lets the overall timeout handle the stall.

**Resetting `failed_consolidation`**: The fix scripts also clear `consolidation_failed_at` flags on `memory_units`, which otherwise cause `failed_consolidation > 0` to persist in API stats even after the underlying issues are resolved.

## Common Pitfalls

1. Restarting because one counter looks flat while logs/observations are still advancing.
2. Treating `completed:consolidation` rows as proof that quality gates passed.
3. Treating historical `failed:consolidation` async rows as current failure without checking source failed counts.
4. Increasing external `parallel_batches` or worker slots without considering recall/search fanout and DB pool limits.
5. Running many external `psql` monitors during DB-pool pressure.
6. Calling proposal-review output "approved". It is only review material.
7. Assuming `total_observations=0` is always failure; some workflows intentionally disable observations.
8. Treating a foreground orchestration timeout or watchdog `rc=-15` as native Hindsight failure while logs/operations/observation counts are still advancing. `rc=-15` usually means SIGTERM/external stop. Prefer watchdog log inspection, wait-for-idle, submit-state/progress reconciliation, resume from the next safe stage, and post-run recall smoke.
11. Forgetting to restore provider/tuning/auto-retain after paid/offline windows.
12. Misdiagnosing V2 publish as hung while local embeddings are still running. A long-lived `docker exec` with high CPU on `SentenceTransformer(... BAAI/bge-m3 ...)` can be normal during publish; verify with `docker top hindsight`, Hindsight stats, and eventual pipeline rc before interrupting.
13. Attempting to poll-wait for `processing==0` to catch a restart window — the auto-re-queue gap between consolidation rounds is sub-second, and even tight polling (3s) consistently misses it. Direct restart is the established practice in this environment (see `references/direct-restart-vs-poll-waiting.md`).
14. **Orphan-fix scripts that bypass ALL unconsolidated records.** After container restart, `fix_orphaned_consolidation.py --bypass` or `_fix_orphaned_via_db()` must NOT mark `consolidated_at=NOW()` on all records with `consolidated_at=NULL`. If a consolidation operation is in-flight (status `processing`/`pending`), those source facts are about to be consolidated — marking them as done corrupts the in-flight work. **Always check `async_operations` for active consolidation ops first.** Both scripts have been patched (2026-05-26) with a `has_active_consolidation()` guard. Use `--force` only when you are certain no work is in-flight.
15. **Container startup failure due to missing embedding model.** If the Hindsight container starts but the local embedding model (e.g. `BAAI/bge-m3`) is not cached or is incomplete, the app will crash with `OSError: does not appear to have a file named pytorch_model.bin or model.safetensors`. The HF cache is mounted from host (`~/.cache/huggingface` → `/home/hindsight/.cache/huggingface`). Download the model to host cache **before** restarting the container. Note: HF hub is case-sensitive for cache dirs — `BAAI/bge-m3` vs `BAAi/bge-m3` create separate cache trees, and the container uses the exact case from `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL`.
11. Forgetting to restore provider/tuning/auto-retain after paid/offline windows.
12. Misdiagnosing V2 publish as hung while local embeddings are still running. A long-lived `docker exec` with high CPU on `SentenceTransformer(BAAI/bge-m3)` can be normal during publish; verify with `docker top hindsight`, Hindsight stats, and eventual pipeline rc before interrupting.
13. Attempting to poll-wait for `processing==0` to catch a restart window. The auto-re-queue gap between consolidation rounds is sub-second, and even tight polling (3s) consistently misses it. Direct restart is the established practice in this environment (see `references/direct-restart-vs-poll-waiting.md`).
14. Assuming parallel env vars are sufficient without the patched consolidator. `HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES=8` and `LLM_MAX_CONCURRENT=8` in the container env do NOT enable parallel LLM batches on their own; only effective when the patched `consolidator.py` (with `AdaptiveLLMConcurrencyLimiter` and `asyncio.create_task` wave scheduling) is loaded. Verify with `docker logs | grep 'limits: batch_parallel=8 llm=8/8'`.
14. Assuming parallel env vars are sufficient without the patched consolidator. `HINDSIGHT_API_CONSOLIDATION_PARALLEL_BATCHES=8` and `LLM_MAX_CONCURRENT=8` in the container env do NOT enable parallel LLM batches on their own — they only have effect when the patched `consolidator.py` (with `AdaptiveLLMConcurrencyLimiter` and `asyncio.create_task` wave scheduling) is loaded. Verify with `docker logs | grep 'limits: batch_parallel=8 llm=8/8'`.
15. Treating operation-level failures as data-level failures. v0.7.1's `search_vector` GENERATED ALWAYS column causes some retain/consolidation operations to fail with `cannot insert a non-DEFAULT value into column "search_vector"`, but the data is committed before the error occurs. When `wait_for_operation_ids` reports `RetainOperationFailed`, check documents/observations delta from `/stats` before concluding data was lost. See `hindsight-local-deployment/references/hindsight-v071-upgrade-and-search-vector-issue.md`.
15. **Orphaned consolidation units after container restart + psycopg2 missing = infinite pipeline stall.** After a container force-restart, `memory_units` may have `consolidated_at=NULL` with no matching `async_operations`, causing `pending_consolidation > 0`. The `hindsight_wait_native_consolidation.py` script tries orphan bypass via `_fix_orphaned_via_db()`, which requires `psycopg2`. The Hermes-agent venv (`~/.hermes/hermes-agent/venv/`) does NOT automatically include `psycopg2` — the pipeline script runs under this venv, so `import psycopg2` fails and bypass never executes, causing the pipeline to loop forever. Fix: (a) install `psycopg2-binary` into the venv, (b) add psql CLI fallback in the bypass function (3-level: psycopg2 → local psql binary → `docker exec` psql), (c) limit bypass retry count (e.g. max 3 attempts) to prevent infinite loops, (d) also reset `consolidation_failed_at` flags during bypass. The `fix_orphaned_consolidation.py` script has the same psycopg2 dependency and needs the same fallback chain.
16. **The Hermes-agent venv is NOT the user's system Python.** Pipeline scripts like `hindsight_memory_pipeline.py` are invoked via `sys.executable` from `hindsight_daily_noagent.py`, which runs under `~/.hermes/hermes-agent/venv/bin/python`. Packages installed in the user's miniconda or system Python are NOT available. Always verify with the venv's python: `~/.hermes/hermes-agent/venv/bin/python -c "import X"`.
15. Assuming psycopg2 is available in hermes-agent venv. Pipeline scripts run under `~/.hermes/hermes-agent/venv/bin/python` (3.11), which is separate from miniconda (3.13). After Hermes upgrades recreate the venv, psycopg2-binary must be reinstalled. The orphan bypass `_fix_orphaned_via_db()` and `fix_orphaned_consolidation.py` now have psql CLI fallback, but psycopg2 should still be installed for efficiency.
16. Orphan bypass looping infinitely when DB access fails. The wait script now has `bypass_attempts` capped at 3; after 3 failed bypass attempts, it gives up and lets the timeout handle it. Without this cap, a missing psycopg2 + missing psql would cause the script to retry bypass every poll cycle forever.
15. Orphan bypass failing silently because `psycopg2` is missing from the pipeline's Python venv. The pipeline runs under `hermes-agent/venv/bin/python`, not the system Python — if psycopg2 is only installed in miniconda, the venv import fails with `ModuleNotFoundError` and the bypass loop retries forever. Fix: (a) install psycopg2-binary in the venv, (b) add psql CLI fallback in `_fix_orphaned_via_db()` and `fix_orphaned_consolidation.py`, (c) cap bypass retry attempts (default 3) so the pipeline doesn't loop forever on persistent failure. The psql fallback tries: system PATH → `~/.hindsight-docker/installation/18.1.0/bin/psql` → `docker exec hindsight psql`.
16. `consolidation_failed_at` flag on `memory_units` preventing re-consolidation. When orphan bypass marks units as consolidated, it must also reset `consolidation_failed_at=NULL` for the same bank, otherwise `failed_consolidation` stays non-zero and `--block-on-failed-consolidation` gates never open. Both `_fix_orphaned_via_db()` and `fix_orphaned_consolidation.py --bypass` must issue `UPDATE memory_units SET consolidation_failed_at=NULL WHERE bank_id=? AND consolidation_failed_at IS NOT NULL` before the consolidated_at bypass update.
15. **Orphaned consolidation + missing psycopg2 = infinite pipeline stall.** After a container restart, `memory_units` with `consolidated_at=NULL` and no matching `async_operations` cause `pending_consolidation` to remain >0. The wait script's orphan bypass calls `_fix_orphaned_via_db()` which requires `psycopg2`. The Hermes-agent venv (`~/.hermes/hermes-agent/venv/`) is a separate Python from `miniconda` — psycopg2 may be in miniconda but NOT in the venv. When psycopg2 is missing, the bypass silently fails every poll cycle, and the pipeline loops forever. Fix: (a) install psycopg2-binary into the venv, (b) ensure scripts fallback to psql CLI or `docker exec psql` when psycopg2 is unavailable, (c) cap bypass retries to prevent infinite loops. See `references/orphaned-consolidation-psycopg2-fallback.md`.
16. **The `fix_orphaned_consolidation.py` script also needs psycopg2.** Its `import psycopg2` with `sys.exit(2)` on ImportError means it crashes outright when run from the venv without psycopg2. Fix: graceful degradation to psql CLI fallback, same as the wait script.
15. Using `topenrouter` as Hindsight `--llm-profile` argument. It is NOT a valid Hindsight profile name — valid profiles include `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`. `topenrouter` is the Hermes provider routing name for ChinaDataPay's tp-api.chinadatapay.com, not a Hindsight concept. To route Hindsight LLM calls through topenrouter, set `llm_profile=deepseek-v4-flash` in pipeline_config.json and override via .env: `HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1`, `HINDSIGHT_OFFLINE_LLM_MODEL=deepseek-v4-flash`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY`.
16. Using model name `deepseek/deepseek-v4-flash` with topenrouter. TopenRouter registers models WITHOUT the provider prefix — correct name is `deepseek-v4-flash`, not `deepseek/deepseek-v4-flash`. The prefixed version returns 403 "This token has no access". This applies ONLY to topenrouter; DeepSeek's official API (`api.deepseek.com`) uses `deepseek/deepseek-v4-flash`.
17. **orphan_bypass failing due to missing psycopg2**: When the daily pipeline's `orphan_bypass` tries to query PostgreSQL to resolve stalled pending operations, it requires `psycopg2` in the pipeline script's Python venv. If missing, orphan_bypass fails with `No module named 'psycopg2'`, leaving `pending_consolidation` stuck at a non-zero count indefinitely (infinite wait loop: `ready=false`, `pending_consolidation>0`, `processing=0`). Fix: install psycopg2 in the venv, or modify orphan_bypass to use pg8000 (pure-Python) as fallback. Verify: `~/.hermes/hermes-agent/venv/bin/python -c "import psycopg2"`.
18. **pending_consolidation stalled with no processing operations**: If `pending_consolidation > 0` and `processing_operations = 0` for hours, and manual `POST /consolidate` only briefly creates a processing op that immediately disappears, the engine may be blocked by orphaned `processing` entries in `async_operations`. Check: `SELECT operation_id, status, error_message FROM async_operations WHERE status IN ('processing', 'failed') AND operation_type LIKE '%consolidation%' ORDER BY created_at DESC LIMIT 10`. Mark stuck `processing` ops as `failed` via asyncpg or the `/consolidation/recover` endpoint.
17. Confusing `topenrouter` with `openrouter.ai`. They are different services. `topenrouter` (tp-api.chinadatapay.com) is a ChinaDataPay domestic relay, no proxy needed. `openrouter.ai` is the international OpenRouter service, requires proxy. Never use openrouter.ai URLs for topenrouter configuration.
18. Running eval or recall-dependent steps immediately after Hindsight container restart. The bge-m3 embedding model takes ~25 seconds to load during which the API returns 502/connection-refused. Always verify `/health` returns `healthy` AND a recall test succeeds before running eval or pipeline steps that depend on the recall API.
15. Using `topenrouter` as an `--llm-profile` value. It is not a valid Hindsight internal profile name. Valid profiles: `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`. OpenRouter (`https://openrouter.ai/api/v1`) has its own API format and is NOT openai-compatible for Hindsight's `hindsight_provider='openai'`. See `hindsight-local-deployment/references/hindsight-llm-profile-naming.md`.
16. Overriding `HINDSIGHT_OFFLINE_LLM_BASE_URL` in `.env` with a third-party proxy that may expire. The default `base_url` in each profile works out of the box. Only override when intentionally routing through a known-good alternative. If LLM calls suddenly return 403/401, check `.env` for stale overrides before debugging the profile or API key.
15. Using `topenrouter` as `llm_profile` in `pipeline_config.json` or `--llm-profile` CLI flag. `topenrouter` is a provider routing name, not a valid Hindsight llm-profile. Valid internal profiles: `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`. Using the wrong name causes immediate rejection from `hindsight_minimax_import.py`. Always use `deepseek-v4-flash` as `llm_profile`.
15. Bypassing orphaned consolidation units while consolidation is actively processing. The `fix_orphaned_consolidation.py` script and `_fix_orphaned_via_db()` both guard against this by checking for active/pending operations, but a naive `UPDATE memory_units SET consolidated_at=NOW() WHERE consolidated_at IS NULL` without this check will corrupt in-flight work. See `references/orphaned-consolidation-units.md`.
16. Downloading HuggingFace models with wrong case (e.g., `BAAi/bge-m3` instead of `BAAI/bge-m3`). The HF cache creates separate directories per case variant, and the container will fail to find the model with `OSError: does not appear to have a file named pytorch_model.bin`. Always match the exact model ID from the container's config.
15. Leaving residual orphaned `memory_units` (1–5 world/experience facts with `consolidated_at=NULL`) after a consolidation drain. The worker considers the bank done and won't pick them up, but `pending_consolidation` stays non-zero, blocking `wait_native_consolidation`. Trigger `POST /consolidate` — if it processes 0, these are truly orphaned and must be bypassed with `fix_orphaned_consolidation.py --bypass`. The `hindsight_wait_native_consolidation.py` script has auto-recovery for this pattern. See `references/orphaned-units-after-consolidation.md`.
16. Running `fix_orphaned_consolidation.py --bypass` or `_fix_orphaned_via_db()` while consolidation is actively processing. This marks in-flight source facts as consolidated, causing the worker to skip them. Both scripts now have `has_active_consolidation()` safety checks; `--force` overrides but risks data inconsistency.
15. Running `fix_orphaned_consolidation.py --bypass` or `_fix_orphaned_via_db()` while consolidation operations are in `processing`/`pending` state. This marks `consolidated_at=NOW()` on rows the worker is still processing, causing them to be skipped without actual consolidation (no observations, no links). Always check `async_operations` for active work first — the scripts now include this guard automatically.
15. **bge-m3 model missing → Hindsight 502/startup failure**. If the container cache (`~/.cache/huggingface/hub/models--BAAI--bge-m3/`) is empty, incomplete, or only has a wrong-case variant (e.g. `models--BAAi--bge-m3` instead of `models--BAAI--bge-m3`), Hindsight will fail to start with `OSError: BAAI/bge-m3 does not appear to have a file named pytorch_model.bin or model.safetensors`. The container mount maps `~/.cache/huggingface` → `/home/hindsight/.cache/huggingface`. Fix: download with correct repo id `BAAI/bge-m3` (case-sensitive) via `snapshot_download('BAAI/bge-m3')` with proxy set, then restart the container. Verify inside container: `docker exec hindsight ls /home/hindsight/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/`.
16. **fix_orphaned_consolidation.py requires psycopg2**. The script at `~/.hermes/scripts/fix_orphaned_consolidation.py` uses psycopg2 which is not in the default Hermes Python environment. Either install `psycopg2-binary` or rewrite to use subprocess+psql.
15. Using a non-whitelisted provider name when switching LLM providers for consolidation. Hindsight validates `HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER` (and the other 3 provider vars) against a hardcoded list at startup — any value not in the list causes an immediate crash. For OpenAI-compatible gateways (TopenRouter, etc.), use `provider=openai` with the custom `BASE_URL`. See `hindsight-local-deployment/references/hindsight-llm-provider-switching.md` for the full switching runbook.
15. Hindsight recall API JSON contains control characters in observation text. Python's `json.loads()` rejects these. Always strip with `re.sub(rb'[\x00-\x08\x0b\x0c\x0e-\x1f]', b'', raw)` before parsing recall response bytes.

Before declaring this skill publish-ready or an offline workflow stable:

- [ ] `SKILL.md` frontmatter includes name, description, version, author, license, and metadata.
- [ ] `SKILL.md` is comfortably below 100,000 chars.
- [ ] All referenced files exist.
- [ ] No hardcoded user paths, profile names, or credentials remain.
- [ ] Packaged scripts compile with `python3 -m py_compile`.
- [ ] Packaged tests pass with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- [ ] Read-only status helper works in API-only mode.
- [ ] Independent review finds no blocker.
