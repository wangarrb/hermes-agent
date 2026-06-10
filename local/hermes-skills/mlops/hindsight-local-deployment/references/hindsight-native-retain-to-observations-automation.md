# Hindsight native retain → consolidation/observations automation

Session learning from reviewing how to automate high-dimensional processing of retained facts using the local Hindsight native API.

## Correct mental model for the local implementation

The current local Hindsight implementation does **not** expose the reference-style slash/API commands below as real interfaces:

- `/hindsight_consolidate --session_id ...`
- `/hindsight_observations ...`
- `/hindsight_reflect --auto_consolidate true --generate_observations true`
- `POST /v1/observations`
- Hermes `post_retain_hooks` in `~/.hermes/hindsight/config.json`

Verified native endpoints are:

- `POST /v1/default/banks/{bank_id}/memories` — retain
- `POST /v1/default/banks/{bank_id}/memories/recall` — recall
- `POST /v1/default/banks/{bank_id}/reflect` — generate a reflected answer; not the default persistent observation writer
- `POST /v1/default/banks/{bank_id}/consolidate` — schedules native consolidation that writes `fact_type='observation'`
- `GET/PATCH /v1/default/banks/{bank_id}/config`
- `GET /v1/default/banks/{bank_id}/operations`

## What is automatic upstream

In the inspected source, retain triggers consolidation only when bank config/env resolves `enable_observations=true`:

- retain extracts/stores facts and basic links/indexes;
- after retain, `memory_engine` calls `submit_async_consolidation(...)` if observations are enabled;
- worker claims `operation_type='consolidation'` only when consolidation slots are available;
- consolidator reads unconsolidated `experience/world` facts and creates/updates/deletes `fact_type='observation'` with `source_memory_ids`.

Do **not** assume built-in idle-session, daily, weekly, threshold, or post-retain-hook schedulers unless verified in the running version. In this environment, daily/weekly jobs are Hermes local offline cron, not Hindsight native background scheduling.

## Recommended architecture

For this user, prefer native-first data plane with local control plane:

1. Retain layer: get content into Hindsight as facts. Keep `retain_chunk_size=8000`, `retain_extraction_mode=concise` unless a quality/cost experiment says otherwise.
2. Native high-dimensional layer: use Hindsight `/consolidate` as the primary persistent observation path. Treat `/reflect` as validation/interactive reasoning, not default durable high-level memory publication.
3. Local guard/orchestration layer: use scripts for preflight, provider switching, cost budget, payload_null quarantine, retries/backoff, patch reapplication, dry-run estimates, and restore-local.
4. Offline pipeline: when native observations become the source of truth, demote offline V2 to audit/eval/wiki-candidate/proposal unless explicitly authorized to publish. Avoid two competing high-level canonical stores.

## Safe automation pattern

Do **not** immediately keep paid observations/consolidation always on when there is a large unconsolidated backlog. Use guarded windows first:

```bash
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py run-native-consolidation-paid \
  --jobs 1 \
  --facts-per-job 50
```

This is dry-run only. Execution requires explicit confirmation:

```bash
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py run-native-consolidation-paid \
  --jobs 1 \
  --facts-per-job 50 \
  --execute \
  --confirm run-native-paid-consolidation
```

The runner should:

- verify health, provider, observations state, pending/failed operations, active payload_null;
- estimate full backlog and current window LLM calls;
- switch to paid/native profile only inside the window;
- set observations on and worker consolidation slots safely;
- apply JSON/429/budget patches;
- POST native `/consolidate`;
- wait for operation completion;
- restore normal-local in `finally`.

## Batch/cost settings

Quality-first default for this user:

- `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=50` — official upstream cap in Hindsight >=0.5.3 / 0.6.x; older local patch used `MAX_MEMORIES_PER_JOB`. Prevents one operation from consuming the whole bank.
- `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE=50` — fetch round size; in this version mainly env-controlled, not bank-config PATCH.
- `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=50` — facts per LLM call; directly affects call count.
- `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096`.
- `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES=1`, `HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS=1` for paid windows.
- `HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS=300`.

Important distinction:

- per-job cap limits blast radius; it does not reduce total full-backlog calls;
- `consolidation_llm_batch_size` reduces calls by packing more facts per LLM call;
- Hindsight exact tag grouping can split one 50-fact job into multiple LLM calls. Example from this bank: first 50 facts estimated `tag_groups=3`, `llm_calls=3`.

Avoid `consolidation_llm_batch_size=8` for call reduction; with ~5944 facts its theoretical lower bound is ~743 calls before tag grouping, versus current 50-batch estimate of ~178 calls.

## Stage 0 validation gap

Existing guard covers system-level preflight: health, DB, provider, queue, payload_null, observations switch, candidate counts. It does not yet fully validate a specific retain/session/document.

Future enhancement: add `validate-retain --document-id ...` or equivalent to check:

- memory count for document/session > 0;
- key metadata present (`document_id`, `mentioned_at`, `occurred_start/end`, tags as applicable);
- recall self-test can retrieve the retained facts;
- no new failed/payload_null operations;
- optional manual retain补全 when critical facts are missing.

## When to move to always-on background processing

Only consider always-on native background consolidation after backlog is drained or small. Then a stable configuration can keep:

- `enable_observations=true`;
- `worker_consolidation_max_slots=1`;
- `worker_max_slots > worker_consolidation_max_slots`;
- per-job cap and 50/50 batch settings still active;
- daily guard/status checks.

Until then, use scheduled guarded windows (e.g. 1 job/50 facts pilot, then 5 jobs/250 facts daily if quality is good).