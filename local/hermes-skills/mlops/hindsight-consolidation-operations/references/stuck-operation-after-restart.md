# Stuck Operation Cleanup After Container Restart

## Problem

When Hindsight container is restarted (`docker restart`) while a consolidation
operation is in-flight (status=`processing`), the operation stays stuck as
`processing` forever.  The worker poller marks it `[STUCK?]` but does not
auto-recover it.  The API refuses to cancel or retry a `processing` operation
(`DELETE /operations/{id}` only accepts `pending`; `POST /operations/{id}/retry`
only accepts `failed`/`cancelled`).

The deduplication logic (`dedupe_by_bank=True`) prevents a new consolidation
from starting while any operation for the same bank is still `processing`.

## Fix

Mark the stuck operation as `failed` in the database, then submit a new
consolidation via the normal API.

### Step 1: Mark stuck operations as failed

```bash
# Script: /tmp/fix_hindsight_ops.py (or any path inside container)
import asyncpg, asyncio, os

async def fix():
    url = os.environ.get("HINDSIGHT_API_DATABASE_URL",
                         "postgresql://hindsight@127.0.0.1:5432/hindsight")
    conn = await asyncpg.connect(dsn=url)
    try:
        rows = await conn.fetch(
            "SELECT operation_id FROM async_operations WHERE status='processing'"
        )
        for row in rows:
            await conn.execute(
                "UPDATE async_operations SET status='failed', completed_at=NOW(), "
                "error_message='container restarted while op was in-flight' "
                "WHERE operation_id=$1",
                row["operation_id"]
            )
            print(f"marked failed: {row['operation_id']}")
    finally:
        await conn.close()

asyncio.run(fix())
```

Run inside the container:
```bash
sg docker -c "docker cp /tmp/fix_hindsight_ops.py hindsight:/tmp/ && \
  docker exec hindsight /app/api/.venv/bin/python3 /tmp/fix_hindsight_ops.py"
```

Note: the column is `operation_id` (not `id`).

### Step 2: Submit a new consolidation

```bash
curl -s -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/consolidate' \
  -H 'Content-Type: application/json' -d '{}'
```

### Timing

The restart itself does not lose data: memories that were in-flight simply
lack `consolidated_at` and will be picked up by the next consolidation round.
A stuck operation only blocks new work; fixing it and resubmitting is safe.

## When to use

- After any `docker restart hindsight` that interrupted an active consolidation
- When the worker log shows `[STUCK?]` on a consolidation operation
- When `processing=1` persists and `pending_consolidation` is not decreasing

## Pitfalls

- Do not run `DELETE FROM async_operations` directly; use the targeted UPDATE.
- Always check `AND status='processing'` to avoid touching completed rows.
- After restart, verify the new code actually loaded (check for marker strings
  in the container file) before kicking off new consolidation.
