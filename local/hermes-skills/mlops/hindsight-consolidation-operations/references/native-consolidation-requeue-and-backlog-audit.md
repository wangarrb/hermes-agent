# Native consolidation requeue and backlog audit

Session-derived note for diagnosing unexpectedly slow Hindsight `pending_consolidation` drains.

## Key mechanism: native consolidation self-requeues

Hindsight native consolidation does not necessarily mean "process only the latest increment". It scans the whole bank for unconsolidated source units:

```sql
SELECT COUNT(*)
FROM memory_units
WHERE bank_id = $1
  AND consolidated_at IS NULL
  AND consolidation_failed_at IS NULL
  AND fact_type IN ('experience', 'world')
```

A single operation processes up to `consolidation_max_memories_per_round`. When it hits that round limit and there is likely more work, `consolidator.py` calls `submit_async_consolidation(...)` again. This is normal full-bank drain behavior, not by itself an infinite-loop bug.

Operational consequence: a small current session/import can appear to trigger thousands of pending items if historical `experience`/`world` rows were never consolidated.

## Fast read-only audit for "why are there thousands?"

Use API status first. If direct DB inspection is needed and safe, count the backlog by fact type, creation hour, and document prefix. In the container, use a non-printing credential path; do not echo DB URLs.

```bash
docker exec -i hindsight python - <<'PY'
import os, asyncio, asyncpg, json

async def main():
    conn = await asyncpg.connect(os.environ['HINDSIGHT_API_DATABASE_URL'])
    try:
        queries = {
            'by_fact_type': """
                SELECT fact_type, COUNT(*) AS n
                FROM memory_units
                WHERE bank_id=$1 AND consolidated_at IS NULL AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience','world')
                GROUP BY fact_type ORDER BY n DESC
            """,
            'by_created_hour_desc': """
                SELECT date_trunc('hour', created_at) AS h, COUNT(*) AS n
                FROM memory_units
                WHERE bank_id=$1 AND consolidated_at IS NULL AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience','world')
                GROUP BY h ORDER BY h DESC LIMIT 20
            """,
            'by_doc_prefix': """
                SELECT COALESCE(split_part(document_id, '::', 1), '[null]') AS doc_prefix,
                       COUNT(*) AS n, MIN(created_at) AS min_created, MAX(created_at) AS max_created
                FROM memory_units
                WHERE bank_id=$1 AND consolidated_at IS NULL AND consolidation_failed_at IS NULL
                  AND fact_type IN ('experience','world')
                GROUP BY doc_prefix ORDER BY n DESC LIMIT 20
            """,
        }
        for name, sql in queries.items():
            rows = await conn.fetch(sql, 'hermes')
            print(name, json.dumps([dict(r) for r in rows], default=str, ensure_ascii=False))
    finally:
        await conn.close()

asyncio.run(main())
PY
```

Interpretation pattern:

- If most rows are under historical `hermes-session` or `hermes-offline-consolidation` document prefixes, the backlog is historical, not the current small increment.
- If creation hours cluster around prior import/offline windows, do not blame the latest run.
- Report `pending_consolidation` separately from active/pending async operations; `pending_operations=0` can still coexist with thousands of unconsolidated source units.

## Speed diagnosis

Check recent worker logs before tuning:

```bash
docker logs --since 10m hindsight 2>&1 | grep -E 'WORKER_STATS|WORKER_TASK|CONSOLIDATION|slow llm call|DB_WAITS|Failed to search memories|TimeoutError' | tail -80
```

If logs show `stage=llm.<provider>.consolidation+structured` and batch logs like `llm=300s`, the bottleneck is provider/LLM batch latency, not worker slots.

## Why worker slots alone often do not help

`HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS` controls how many consolidation operations may be claimed, but same-bank consolidation is serialized by the claim query: a bank already `processing` consolidation is treated as busy. Increasing slots helps multiple banks or mixed operation types; it will not accelerate one currently active `hermes` consolidation job.

## Practical acceleration options

1. If the goal is to verify downstream weekly/V2/conflict/proposal stages, do not wait for a full-bank native drain unless required. Wait for a safe operation boundary, temporarily prevent further self-requeue/observation drain if explicitly approved, then run the downstream stages.
2. For real backlog drain speed, enable/restore batch-level parallelism inside the consolidator: process multiple LLM batches concurrently while serializing observation writes. The safe target profile is `batch=20`, `llm_batch=20`, `max_round=60`, `parallel_batches=3` (estimated recall fanout ~60).
3. Avoid blind high parallelism. `_process_memory_batch` already launches per-fact recall with `asyncio.gather`, so outer parallelism multiplies DB/search fanout.
4. Consider provider/model changes only after confirming LLM-bound logs and with quality/cost tradeoffs explicit.

## Pitfalls

- Do not call native self-requeue a bug without checking source unit backlog.
- Do not describe `pending_operations=0` as idle when `processing=1` and `pending_consolidation` remains high.
- Do not increase `WORKER_CONSOLIDATION_MAX_SLOTS` expecting same-bank speedup.
- Do not let a downstream verification watchdog wait for full-bank drain by default; it can turn a small validation into a 10+ hour backlog drain.
