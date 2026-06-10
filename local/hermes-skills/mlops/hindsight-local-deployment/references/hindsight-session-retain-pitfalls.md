# Hindsight session retain pitfalls

Session context: 2026-05-09 clean paid retain for the earliest production week after DB reset.

## What happened
- The first retain attempt became inconsistent because submit/wait were split across failures, then the same manifest was re-run and queued duplicate retain operations.
- The duplicate queue was large enough that a clean restart was cheaper than trying to reason about the mixed state.
- After a PostgreSQL drop/recreate, the container could still report `/health` OK while the schema was incomplete; manual migrations were still required.

## Correct recovery sequence
1. Back up the DB first.
2. Stop old retain/wait processes.
3. Reset the DB only if the queue state is polluted or duplicated.
4. Run migrations manually after reset if startup migrations are disabled.
5. Before `--execute`, verify the exact manifest path, record count, date window, `action=production` count, `include_content`, content policy, and sha256 against the user's stated target. Do not substitute a newer `latest`/all-history manifest for a previously approved cleaned manifest.
6. Check Hermes cron status before attributing an unexpected run to cron. Paused Hindsight daily/weekly jobs are not the cause; manual background wrappers and Docker workers can continue independently.
7. Submit the manifest once with a fresh submit-state.
8. If switching LLM provider, keep the existing queue and only switch the worker/provider env; do not resubmit the manifest.

See also `session-retain-operator-safety.md` for the full paid-retain pre-execute checklist and 2026-05-10 wrong-manifest/concise-config incident.

## Provider-switch pitfalls
- `session-manifest-retain-llm --allow-existing-queue` is not a safe way to switch providers if the queue may already contain submitted work; it can still create duplicates.
- `import-llm --llm-profile minimax --allow-existing-queue` is the safer switch path for an already-submitted queue because it restarts worker/provider without resubmitting the manifest.
- Critical config pitfall: wrapper restart/provider-switch paths can re-apply `base_env()` defaults and overwrite bank config, especially `retain_extraction_mode`, custom instructions, causal-links, and chunk size. On 2026-05-10 this caused a paid bge-m3 retain attempt to run with `retain_extraction_mode=concise` even though the intended plan was `custom`. The wrapper was patched to default to `custom`, user+assistant-only custom instructions, `retain_extract_causal_links=false`, and env-overridable `HINDSIGHT_NATIVE_RETAIN_*`; still verify `/banks/<bank>/config` and container env immediately after every provider switch or container recreate before allowing paid work to continue.
- `hindsight_minimax_import.py` does not auto-switch providers on HTTP 429; the patched Hindsight client only does long backoff/retry (`HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS=300`). Provider changes require explicit `import-llm` / `session-manifest-retain-llm` execution.
- DeepSeek can show long slow-call / stuck behavior before any explicit 429s are visible.
- 2026-05-10 pitfall/update: after correcting `deepseek-v4-flash` to the official DeepSeek API (`https://api.deepseek.com/v1`, `DEEPSEEK_API_KEY`), direct Chat Completions smoke returned HTTP 200 and valid JSON. An early Hindsight native retain smoke still produced 0 memory_units, so it was unsafe to use. Later guarded temp-bank smokes passed: retain `1 synthetic doc -> 5 memory_units` and observations/consolidation `5 facts -> 3 observations` with no JSON parse errors. Production use is allowed only through the guarded paid window: exact manifest preflight, temp/small-batch smoke, observations disabled unless explicitly testing, and restore `normal-local` immediately after.
- MiniMax can respond faster, but it may hit quota/429 even at low concurrency; lower concurrency by restarting the worker only and do not resubmit the manifest.
- Bailian/DashScope GLM-5 is available via profile alias `bailian` -> `glm` (`model=glm-5`, OpenAI-compatible base URL `https://coding.dashscope.aliyuncs.com/v1`). To continue an existing queue with it, use `HINDSIGHT_OFFLINE_LLM_CONCURRENCY=8 python3 ~/.hermes/scripts/hindsight_minimax_import.py import-llm --llm-profile bailian --allow-existing-queue`; this only restarts the worker/provider and does not resubmit the manifest, but still check the config pitfall above. For offline reflect/consolidation, start at concurrency 8 by default; 16 hit DashScope/Bailian `concurrency allocated quota exceeded` 429 during weekly all-history even after daily backfill succeeded, so use 8 as the stable ceiling unless doing a guarded experiment.
- Pitfall observed during provider switch: the container/env can already be updated to `glm-5` while the API still crash-loops in embeddings startup and never reaches health. In that case, verify with `hindsight_minimax_import.py status` and `docker inspect` rather than assuming the provider switch failed, and do not resubmit the manifest.
- 2026-05-11 crash-loop root cause chain: Chinese-capable local embeddings (`BAAI/bge-m3`) were related. The model cache existed, but without `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`, `SentenceTransformer`/Transformers still did an optional HuggingFace metadata probe (`adapter_config.json` / PEFT check) during startup and crashed with `RuntimeError: Cannot send a request, as the client has been closed`. Fix: keep bge-m3 cached on host and set both offline env vars in the Hindsight container wrapper.
- 2026-05-11 second blocker: after fixing embeddings, API failed on PostgreSQL because the production pg0 data is under `~/.hindsight-docker/instances/hindsight/data`, not the stale `~/.pg0/instances/hindsight/data`. Always start/check the production `.hindsight-docker` pg0 before Hindsight and query it for `async_operations`; the stale `.pg0` DB showed old 2026-05-04 counts and can mislead recovery.
- 2026-05-11 offline reflect/consolidation blocker: `offline_hindsight_reflect_consolidate.py` still hardcoded `PSQL = $HOME/.pg0/installation/18.1.0/bin/psql`. If you run weekly/daily backfill after the stale DB was deleted, either patch that constant or export `HINDSIGHT_PSQL=$HOME/.hindsight-docker/installation/18.1.0/bin/psql` before starting the wrapper, otherwise it crashes before any paid calls are submitted.
- Protected Hindsight production paths: keep `~/.hindsight-docker/instances/hindsight/data`, `~/.hindsight-docker/instances/hindsight/instance.json`, `~/.hindsight-docker/installation/`, and `~/.cache/huggingface/hub/models--BAAI--bge-m3`. Cleanup candidates are old backups/stale DBs such as `~/.pg0`, `~/.hindsight-docker/backups`, and `~/.hindsight-docker/instances/hindsight/data.pre-reset-*`; inventory and confirm before deleting.
- Switching/restarting provider can move previously `processing` retain ops back to `pending`; this is normal reclaim behavior if `duplicate_ops` remains 0.

## Confirming provider switch without extra paid calls
- Prefer local evidence before doing any paid smoke: `hindsight_minimax_import.py status`, container env, Hindsight startup logs, and production DB queue progress. A provider switch to Bailian/DashScope may appear internally as OpenAI-compatible `provider=openai`, `model=glm-5`, `base_url=https://coding.dashscope.aliyuncs.com/v1`; do not expect the alias string `bailian` to appear in Hindsight logs.
- Strong confirmation bundle:
  1. `/health` is healthy and database connected.
  2. Container env shows `HINDSIGHT_API_RETAIN_LLM_PROVIDER=openai`, `HINDSIGHT_API_RETAIN_LLM_MODEL=glm-5`, and DashScope base URL for retain; check reflect/consolidation separately only if those workers are enabled.
  3. Docker logs show `LLM (retain): provider=openai, model=glm-5` and `OpenAI-compatible client initialized ... base_url=https://coding.dashscope.aliyuncs.com/v1`.
  4. Production DB `async_operations` shows child `retain` rows moving from pending/processing to completed, `failed=0`, and `documents` count increasing after the switch timestamp.
- If the user reports that Bailian/DashScope console usage does not match local expectations, explicitly separate the two model classes before drawing conclusions:
  - Remote paid LLM for retain/reflect/consolidation is controlled by `HINDSIGHT_API_*_LLM_PROVIDER/MODEL/BASE_URL` (e.g. `openai` + `glm-5` + DashScope endpoint).
  - Local embeddings can simultaneously be `BAAI/bge-m3` in offline mode; this is not a local chat/retain LLM and should not be counted as GLM-5 usage.
  - Do not say “it must be console delay” as the only explanation. Also consider wrong console/account/key/channel, OpenAI-compatible endpoint accounting differences, shared key calls, or no active LLM work remaining.
- After a paid retain queue drains, verify follow-on work explicitly instead of assuming it is running: query `async_operations` grouped by `operation_type,status`; check for `consolidation` and `observation(s)` rows, and check `HINDSIGHT_API_ENABLE_OBSERVATIONS` plus `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS`. In the 2026-05-11 GLM-5 recovery, final state was `retain|completed`, `batch_retain|completed`, `consolidation|completed|2`, no pending/processing/failed, and observations disabled/absent.
- Do not use a naive `pending_operations` total for ETA because non-claimable parent `batch_retain` rows can inflate it. Estimate real remaining work from claimable/child `retain` pending + processing rows.
- Live `ss`/netstat sampling is weak evidence: Hindsight provider HTTP calls can be short-lived, may not be captured in a small sampling window, and unrelated Hermes processes can own other 443 connections. Absence of a captured 443/proxy connection is not evidence that the worker is not calling the configured provider.
- If the user asks for provider-side proof, explain the boundary: local evidence confirms Hindsight's active config and queue progress, but vendor-console confirmation requires checking DashScope/Bailian billing/logs or running an explicit paid smoke call. Do not initiate an extra paid smoke call without explicit user approval.

## Call-count and cost sanity checks
- Do not equate "one week" or "83 records" with a small retain job. Inspect actual manifest payload size first: `content_omitted=true` records can still contain very large `content` strings, while metadata fields such as `content_chars` may represent a compressed/filtered estimate and undercount the prompt-sized payload.
- Hindsight native retain LLM cost scales with chunk count, not document count. Estimate chunks from actual retained text and `HINDSIGHT_API_RETAIN_CHUNK_SIZE`; each chunk can generate one or more provider calls due to retries, JSON/parser repair, 429 backoff, or output-too-long splitting.
- In the 2026-05-09 raw run, 83 records produced a 32 MB manifest with ~28M actual content characters and ~3,548 chunks@8000; provider bills in the thousands were therefore plausible even without duplicate submission.
- The durable fix was not just lower concurrency; it was input cleaning. The accepted policy for paid session retain is user+assistant-only: keep user inputs and Hermes assistant outputs, drop tool/command/search/thinking/procedural traces before native retain.
- For user+assistant-only v2 on the same week: 1,152,534 chars, 187 chunks@8000, 142 chunks@12000, 124 chunks@16000. Quality-first choice was 8000; balanced default 12000; 4000/6000 cost more with uncertain extra value.
- Hindsight native retain call scale should be estimated from chunk count plus provider logs/billing, not from Hermes `state.db` `api_call_count`. Hermes main-session calls, auxiliary compression, and Hindsight Docker LLM calls are separate channels and can easily be mixed in a daily/cost report.
- Cross-check all three channels before concluding where cost came from:
  1. Hindsight DB: `SELECT operation_type,status,count(*) FROM async_operations GROUP BY 1,2;`
  2. Chunk sizing from manifest actual `content` length and current `retain_chunk_size`.
  3. Hermes main session usage: `~/.hermes/state.db` `sessions.api_call_count`, which is not the same as Hindsight retain calls.

## Verification to wait for before audit
- `duplicate_ops == 0`
- `failed_operations == 0`
- no claimable child `retain` rows remain
- no active `processing` retain rows remain
- `docs_without_units` is understood and acceptable, or investigated if nonzero
- then run facts audit and recall smoke

## Reset / wait-audit race pitfalls
- A wait/audit job that spans an intentional DB reset can report `queue drained` and `0 documents / 0 memory_units` after the reset. Do not treat this as a successful retain audit; read the full wait log timeline. If counts were increasing before a reset and then become zero, this is a stale watcher crossing reset boundary.
- Record reset evidence explicitly: backup path, pre-reset counts, embedding type before/after. Example bge-m3 reset evidence: pre-reset partial backup had `memory_units.embedding=vector(384)`, 74 documents, 714 memory_units; post-reset schema was `vector(1024)` and empty.
- If a `session-manifest-retain-llm` wrapper was launched with a short `--wait-timeout-s`, it can time out and run its `finally` restore-to-normal-local path while Hindsight still has pending paid retain work. For long queues, prefer either a large timeout (8-12h) or decouple submission from waiting: submit once, kill/stop only the local watcher if needed, then run a separate `wait-queue -> audit -> normal-local` monitor.
- If killing a wrapper to prevent premature restore, do not kill the Hindsight container/worker. Kill only the local wrapper/runner process after confirming operations are already queued in DB; then verify `/health`, `async_operations` pending/processing counts, and that the worker remains in paid provider mode.

## wait_for_operation_ids partial-failure misdiagnosis (2026-06-02)

- `hindsight_session_retain_runner.py`'s `wait_for_operation_ids()` raises `RetainOperationFailed` when any operation in the tracked set fails after max retries (default 3). It does NOT distinguish "2 out of 200 operations failed" from "all operations failed".
- In the 2026-06-02 daily run, 2 retain operations failed (due to v0.7.1 `search_vector` GENERATED ALWAYS column issue), but 54 documents and 159 observations were successfully committed before those operations hit the error. The runner threw `RetainOperationFailed` and the noagent wrapper reported exit code 1, making it look like the entire retain failed.
- **Correct diagnosis pattern**: when `wait_for_operation_ids` or the daily noagent wrapper reports failure, check data-level evidence BEFORE concluding retain failed:
  1. Compare `total_documents` and `total_observations` before and after the run (from `/v1/default/banks/hermes/stats`).
  2. Check `submit_state.json` for the manifest's document IDs — if they show `status=completed` or the document count increased, data landed.
  3. Inspect the specific failed operation error messages — `search_vector` errors are v0.7.1 non-blocking known issues (see `references/hindsight-v071-upgrade-and-search-vector-issue.md`).
  4. Only if documents did NOT increase should you treat the failure as a true retain failure requiring re-run.
- **Do not blindly re-run the daily pipeline** when the exit code is non-zero but data has landed. Re-running will re-submit the same manifest, potentially creating duplicate documents.
- The 11:45 re-run on the same day hit `Connection refused` because the 00:01 run's restore/recreate cycle was still restarting the container. Avoid scheduling overlapping daily runs.

## Pending parent-operation caveat
- Native session manifest retain can leave `batch_retain` parent rows with `status='pending'` and `task_payload IS NULL` while child `retain` operations do the real work. These rows are not claimable (`idx_async_operations_pending_claim` excludes null payloads) and can make a naive `wait-queue` / stats-only watcher look stuck.
- Diagnose with:
  - `SELECT operation_type,status,count(*) FROM async_operations GROUP BY 1,2 ORDER BY 1,2;`
  - `SELECT operation_type,count(*) FROM async_operations WHERE status='pending' AND task_payload IS NOT NULL AND (next_retry_at IS NULL OR next_retry_at <= now()) GROUP BY 1;`
  - `SELECT operation_type,count(*) FROM async_operations WHERE status='pending' AND task_payload IS NULL GROUP BY 1;`
- Do not declare retain complete while claimable `retain` rows or `processing` rows remain. If only non-claimable `batch_retain` parent rows remain, treat it as a parent-status bookkeeping issue and base the audit decision on completed child retain count, `docs_without_units`, `failed_operations`, and duplicate checks rather than waiting forever on the parent rows.

## MiniMax `generator didn't stop after athrow()` during retain

See `references/minimax-retain-generator-throw-2026-05-18.md` for diagnosis and recovery.

Summary: This is a transient MiniMax streaming bug, not a content/db issue. Check failed ops with the operations API, retry with full UUID. After retry, the pipeline can resume with `--skip-daily` once consolidation drains.
