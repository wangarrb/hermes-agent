# Hindsight consolidation tail slowdown + DB pool cap pattern

Use this note when a large Hindsight consolidation drain initially moves fast but the final few hundred `unconsolidated_base` memories slow sharply, while monitors intermittently show `FATAL: sorry, too many clients already`.

## Observed symptoms from 2026-05-12 drain

- Backlog moved quickly earlier, then near the tail flattened:
  - recent 10 min: `uncon_delta` ~2
  - recent 20-35 min: `uncon_delta` ~3
  - recent 60 min: `uncon_delta` ~11
  - recent 120 min: `uncon_delta` ~425
- `observations` still increased and `processing:consolidation=1`, so this was not a full stop.
- Container logs showed one long-running op with `stage=llm.openai.consolidation+structured` or `stage=task.consolidation` for 1000s+.
- Tail batches became tiny but extremely expensive:
  - `1 memories, 4 llm calls`, avg ~1466s/memory
  - `2 memories, 3 llm calls`, avg ~1594s/memory
  - `2 memories, 3 llm calls`, avg ~953s/memory
- Recent completed batch metrics exposed root causes:
  - recall spikes: `recall=5685.983s`, `4785.387s`, `753.976s`
  - slow/timeout LLM calls: 100-400s+ and `APIConnectionError: Request timed out`
  - embedding/DB write stalls: `embedding=2281.130s`, `db_write=429.303s`
  - DB waits on semantic expansion, insert, update paths.

## Interpretation

Do not interpret flat `unconsolidated_base` alone as a stop. In tail phases, source memories may be marked only at batch/job commit, while observations can still be created or updated. Use three signals together:

1. `processing:consolidation` / active op still present.
2. Container logs show active `slow llm call`, `llm_batch`, DB wait, or worker stage updates.
3. `failed_base` remains 0.

If these hold, the drain is slow, not dead.

## Why the tail is slow

Common causes:

- Remaining memories have broad tags/scopes and trigger large recall over many existing observations.
- Prompts grow because source text/context or recalled observations are large.
- Structured JSON output becomes unstable, causing retries or split batches.
- Observation update/dedupe touches large histories and requires embedding / DB writes.
- API DB pool pressure makes both internal writes and external monitor psql probes slower.

## DB pool cap lesson

There are two different limits:

- PostgreSQL server `max_connections` (in this setup: `$HOME/.hindsight-docker/instances/hindsight/data/postgresql.conf`, `max_connections = 100`).
- Hindsight API asyncpg pool max (`pool: size=100 limits=5-100` in logs; default came from `DEFAULT_DB_POOL_MAX_SIZE = 100`).

If both are 100, the API can occupy every PostgreSQL client slot. External monitor/psql then fails with `sorry, too many clients already`.

Preferred first fix is usually not to raise API pool size. Lower the API pool max to leave headroom, e.g. stage `DEFAULT_DB_POOL_MAX_SIZE = 60`, then apply after the active op finishes. Raising PostgreSQL `max_connections` is a second-order option and can increase memory/CPU pressure; do it only with capacity reasoning.

## Safe pool-cap application pattern

1. Patch/stage the API pool cap in the container/source copy and compile-check it.
2. Do not immediately restart during `processing:consolidation=1` unless the user explicitly accepts losing in-flight work.
3. Stop external drain driver/watchers so they cannot enqueue a new job after the current op completes.
4. Run a log-based idle/restart watcher that:
   - identifies current consolidation op from container logs,
   - waits for `Marked async operation as completed: <op>` or failed,
   - restarts the Hindsight container,
   - verifies `/health`,
   - restarts the drain in tmux.
5. Keep tmux monitor alive by showing cached JSONL snapshots and container-log liveness, not by retrying psql aggressively.

## When to change drain parameters

If the tail remains slow after pool cap/restart:

- Conservative: continue if logs show activity and `failed_base=0`.
- More stable tail mode: reduce failure radius:
  - `max_memories_per_round`: 200 -> 50/80
  - `consolidation_llm_batch_size`: 50 -> 20/25
  - `parallel_batches`: 8 -> 4-6
- Tradeoff: more calls and orchestration overhead, but less chance one pathological 50-memory batch blocks progress for a long time.

## Reporting language

Be direct: “slow tail, not stopped” when logs show activity. Explain that backlog is stepwise and can stay flat until batch/job commit. Mention concrete evidence: active op id, stage age, recent slow LLM call, recent observations, failed_base=0, and recent deltas.
