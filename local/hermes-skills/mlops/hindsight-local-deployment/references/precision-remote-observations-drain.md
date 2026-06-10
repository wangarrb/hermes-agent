# Hindsight precision-remote observations drain

Use this reference when a Hindsight offline reflect/consolidation or observations recovery run is interrupted, half-drained, or needs to move from cost-saving local inference to precision-first remote LLM inference.

## Core decisions captured from 2026-05-11 session

- Prefer precision over cost when the user asks for higher-quality Hindsight recall/reflect results.
- Keep `auto_retain=false` during offline/import work to avoid noisy redundant facts interfering with curated offline processing.
- Set active Hermes Hindsight config to:
  - `memory_mode=hybrid`
  - `recall_prefetch_method=reflect`
  - `recall_budget=high`
  - `recall_cache_enabled=false` when precision and freshness matter more than speed
  - `enable_observations=true` when repairing or draining observation/consolidation backlog
- Keep `recall_cache_enabled=false` through active ingestion/repair/drain/governance. Turning it on is reasonable only after observations and derived state are stable; otherwise cached reflect/recall can hide newly generated observations or recently corrected/quarantined facts.
- Use remote paid LLM for retain/reflect/consolidation by default in precision mode; local models are only for embeddings unless the user explicitly asks for local inference.
- Cap paid concurrency conservatively (8 was chosen) even if `.env` contains a larger value. High concurrency can reduce stability and precision through rate limits/timeouts.
- Keep mental model refresh disabled during large offline processing/drain runs. One core purpose of offline consolidation is to reduce downstream LLM calls by building observations; enabling `refresh_after_consolidation=true` can trigger a reflect/LLM refresh after every consolidation job for every matching mental model, multiplying calls roughly by `consolidation_jobs × matching_mental_models`. Only refresh mental models manually or via low-frequency scheduled jobs after the backlog is drained.

## Network/proxy lesson

Hindsight Docker with `HTTP_PROXY=http://127.0.0.1:7890` can fail because `127.0.0.1` inside the container is the container itself, not the host proxy. Symptoms include LLM consolidation/observations failures even though host network looks fine.

Preferred order:
1. Test provider direct connectivity from the container with proxy env unset.
2. If direct works for DeepSeek/MiniMax/DashScope/Bailian, remove global `HTTP_PROXY`/`HTTPS_PROXY` injection from the Hindsight container environment.
3. Only enable proxy for specific providers that require it, not globally.
4. If an in-flight consolidation must finish before container recreation, a temporary in-container direct proxy on `127.0.0.1:7890` is acceptable as a bridge, but remove it after the queue drains and restart the container with clean env.

## Safe interrupted-run recovery sequence

1. Inspect current state before modifying anything:
   - active config JSON
   - async operations grouped by status/type
   - offline document/unit coverage
   - observation count and latest timestamp
   - failed/processing consolidation count
2. Do not kill active consolidation unless necessary. If a single processing job is still moving and `failed_base=0`, wait and monitor.
3. Start a visible monitor in tmux for long drains. Track:
   - `observations` count and latest timestamp
   - `unconsolidated_base`
   - `failed_base`
   - processing consolidation count
   - offline docs total/with_units/zero
   - short-window ETA, labeled as approximate
4. If switching model/provider or removing proxy, prefer a watcher that waits for idle, then applies the switch and restarts the container. Avoid interrupting in-flight LLM operations.
5. Do not try to parallelize same-bank consolidation by increasing worker slots alone. Same-bank consolidation remains effectively serialized because the submit path dedupes by bank and the worker claim path refuses a second `processing` consolidation for the same bank. More importantly, the consolidation fetch path selects unconsolidated `memory_units` without per-memory claim/lock, so forcing multiple same-bank jobs risks duplicate processing, duplicate observations, extra spend, and memory pollution.
6. Understand what remains after each consolidation job:
   - mark processed base `world`/`experience` units with `consolidated_at`
   - create/update/merge `observation` units and embeddings
   - optionally enqueue `refresh_mental_model` only if matching mental models have `trigger.refresh_after_consolidation=true`
   - mark async operation complete and optionally fire consolidation webhooks
   - a drain script then triggers the next `/consolidate` job until `unconsolidated_base=0`
   If `mental_models` is empty or no model has `refresh_after_consolidation=true`, post-consolidation overhead is light; the expensive part is the repeated LLM structured consolidation jobs.
7. Avoid duplicate drain/watchers. After an idle-switch watcher recreates the container and starts a drain, verify with `pgrep -af hindsight_observations_drain.py` and keep only one drain process; duplicate drains create noisy duplicate logs and can race to trigger redundant consolidation attempts.
8. For long paid/remote observation drains, do not leave short watchdog ceilings such as `8*3600` in `hindsight_observations_drain.py`. An 8h ceiling can make the run stop cleanly halfway with `timeout_8h` while the monitor keeps repainting flat counts. Use at least a 48h ceiling (`MAX_RUNTIME_SECONDS = 48 * 3600`, `timeout_48h`) or a run-until-empty design, then restart the drain worker so the new ceiling is in the active process.
9. After changing a drain script while it is already running, compile/check the script, kill only the old drain worker (not the monitor or active Hindsight container), restart one drain process, and verify the latest `*-drain.jsonl` shows fresh samples plus `processing:consolidation` or declining `unconsolidated_base`.
10. After idle/restart, run final audit:
   - container/status healthy
   - no stale proxy env unless intentionally configured
   - `auto_retain=false`, `memory_mode=hybrid`, `recall_prefetch_method=reflect`, `recall_budget=high`
   - queue empty or expected residual failures documented
   - observations continue to grow after a small smoke query/run

## Completion false-positive trap

A monitor that keeps refreshing is not proof of progress. In one interrupted drain session, the monitor tmux window stayed alive while the actual drain worker had disappeared; later, a restarted drain worker resumed and the JSONL log showed real progress. Before telling the user that Hindsight is “done”, verify all three:

- live queue counts changed recently in the JSONL monitor/drain log, not just the terminal repaint
- the drain worker is present (`pgrep -af hindsight_observations_drain.py` or the expected process list)
- `unconsolidated_base` actually declines across samples, or reaches zero

If `observations`, `unconsolidated_base`, and `failed_base` are flat for multiple samples, treat the run as stalled/incomplete even if the monitor window still looks active.

When recovery is needed, restart the drain worker and confirm the new log file gets fresh JSONL rows; do not rely on the tmux session alone as proof that the pipeline is still making forward progress.

## Parallelization design note

If observation consolidation is too slow, first verify timing from docker logs before changing architecture. In the 2026-05-12 run, evidence showed `llm=~500-725s` per job while `db_write=~0.7-11s`, so the bottleneck was LLM generation, not PostgreSQL writes.

Do **not** speed this up by simply starting multiple same-bank consolidation jobs or increasing worker slots:

- `submit_async_consolidation(..., dedupe_by_bank=True)` dedupes pending same-bank consolidation.
- Worker claim SQL refuses another `consolidation` for the same bank while one is already `processing`.
- The consolidator currently fetches source `memory_units` by `consolidated_at IS NULL` without a per-memory claim/lease; parallel jobs would likely read the same sources, duplicate LLM spend, and race on observations.

The safer architecture is claim-then-parallel-process-then-commit:

1. Atomically claim source `world`/`experience` rows before LLM work, using `FOR UPDATE SKIP LOCKED` or explicit `consolidation_claimed_at/consolidation_claimed_by` lease columns. Claiming, not writing observations, is the critical anti-duplication step.
2. Run multiple claimed LLM batches concurrently after source rows are reserved. Keep mental model refresh disabled during large drains.
3. In the write phase, use short transactions and row locks for target observations (`SELECT ... FOR UPDATE` before update/delete/merge) and mark source rows `consolidated_at` only after their actions have been applied.
4. On failure, release or expire stale claims so recovery can resume without permanently orphaning source memories.
5. Keep final audit checks: `failed_base=0`, declining `unconsolidated_base`, no duplicate drain workers, and observation counts increasing or queue empty.
6. If refactoring update/create history writes, remember that asyncpg returns DB timestamps as Python `datetime` objects. Any JSONB history payload built from existing row fields must use explicit serialization, e.g. `json.dumps(payload, default=_json_default)` where datetimes become `isoformat()` and UUIDs become strings; otherwise the job can fail after spending a full LLM call with `Object of type datetime is not JSON serializable`.
7. When a hot patch must be loaded after a failing in-flight consolidation, a watcher waiting for `processing=0 AND pending=0` can deadlock on retry-blocked/failed operations. For patch reload, wait for `processing=0`, restart the container, health-check it, then start exactly one drain worker. If the watcher uses `os.execvp(...)`, ensure `import os` is present.

## Reporting style for the user

## Reporting style for the user

Report concise operational facts first: monitor location, current counts, failures, ETA, config changes, and remaining risk. Avoid long narrative unless asked. Explicitly mark ETA as approximate and derived from recent slope.
