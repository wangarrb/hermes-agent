# Restart Stuck Operation Recovery

## Problem

After `docker restart` during active consolidation, the Hindsight worker's
`dedupe_by_bank=True` mechanism prevents a new consolidation from being picked
up while the bank appears to have an active `processing` operation. The
Operations API refuses to cancel (`status='processing', only 'pending' can be
cancelled`) and refuses to retry (`expected 'failed' or 'cancelled'`).

## Recovery Recipe

### Step 1: Find stuck operation IDs

```bash
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?exclude_parents=true&status=processing&limit=5' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); [print(op['id']) for op in d.get('operations',[])]"
```

### Step 2: Try API recovery first

```bash
# v0.6.1 recovery endpoint (best effort)
curl -s -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/consolidation/recover' \
  -H 'Content-Type: application/json' -d '{"confirm":"recover-hindsight-consolidation"}'
```

### Step 3: DB-level unsticking (when API recovery returns retried_count=0)

Container has no `psql` binary. Use the container's own Python + asyncpg:

```bash
# Write fix script and execute inside container
cat > /tmp/fix_stuck_ops.py <<'PY'
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
PY

sg docker -c "docker cp /tmp/fix_stuck_ops.py hindsight:/tmp/fix_stuck_ops.py"
sg docker -c "docker exec hindsight /app/api/.venv/bin/python3 /tmp/fix_stuck_ops.py"
```

### Step 4: Submit new consolidation

```bash
curl -s -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/consolidate' \
  -H 'Content-Type: application/json' -d '{}'
```

## Column Name Trap

The `async_operations` table uses `operation_id` (not `id`). Using `id` gives:
```
asyncpg.exceptions.UndefinedColumnError: column "id" does not exist
```

Always use `operation_id` in WHERE clauses.

## Verify

```bash
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/stats' \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pending_consolidation',0))"
```
