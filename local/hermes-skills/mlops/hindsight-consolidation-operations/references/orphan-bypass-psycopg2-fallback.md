# Orphan Consolidation Bypass: psycopg2 / psql CLI Fallback

## Problem

After a Hindsight container restart, some `memory_units` have `consolidated_at=NULL` with no matching active `async_operations`. The consolidation worker won't pick them up, causing `pending_consolidation > 0` indefinitely.

The pipeline's `hindsight_wait_native_consolidation.py` has an orphan bypass (`_fix_orphaned_via_db()`) that was psycopg2-only. But the pipeline runs under `hermes-agent/venv/bin/python`, which may not have psycopg2 installed (it was only in miniconda). When psycopg2 import fails, the bypass silently fails every poll cycle, and the pipeline loops forever.

## Fix (2026-05-29)

### 1. Install psycopg2 in the venv

```bash
/home/wyr/.hermes/hermes-agent/venv/bin/pip install psycopg2-binary \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

### 2. psql CLI fallback chain

`_fix_orphaned_via_db()` now tries three paths:

1. **psycopg2** (preferred — gets rowcount)
2. **psql CLI** — search order: `shutil.which("psql")` → `~/.hindsight-docker/installation/18.1.0/bin/psql` → skip
3. **docker exec psql** — `docker exec hindsight psql -U hindsight -d hindsight -c "..."`

`fix_orphaned_consolidation.py` similarly falls back to psql CLI when psycopg2 is unavailable (instead of `sys.exit(2)`).

### 3. Retry limit

`_fix_orphaned_via_db()` returns int (units bypassed). The main loop tracks `bypass_attempts` (max 3). If bypass succeeds (returns n > 0), counter resets. After 3 consecutive failures, the loop gives up and lets the timeout handle exit.

### 4. Reset `consolidation_failed_at`

Both the inline bypass and `fix_orphaned_consolidation.py --bypass` now also reset `consolidation_failed_at=NULL` for the bank before marking units as consolidated. Without this, `failed_consolidation` stays non-zero and `--block-on-failed-consolidation` gates never open.

### 5. Clean up stale failed operations

Historical `async_operations` with `status='failed'` and `error_message='generator didn't stop after athrow()'` from old sessions can be safely cancelled:

```sql
UPDATE async_operations SET status='cancelled', completed_at=NOW()
WHERE status='failed' AND created_at < '2026-05-28';
```

## Diagnostic Commands

```bash
# Check pending/failed consolidation
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats | python3 -c "
import sys,json; d=json.load(sys.stdin)
print(f'pending_consolidation={d[\"pending_consolidation\"]} failed_consolidation={d[\"failed_consolidation\"]}')
"

# Find orphaned units (no active consolidation ops)
/home/wyr/.hermes/hermes-agent/venv/bin/python ~/.hermes/scripts/fix_orphaned_consolidation.py --bank hermes --dry-run

# Fix orphans with bypass
/home/wyr/.hermes/hermes-agent/venv/bin/python ~/.hermes/scripts/fix_orphaned_consolidation.py --bank hermes --bypass

# Verify clean state
/home/wyr/.hermes/hermes-agent/venv/bin/python ~/.hermes/scripts/hindsight_wait_native_consolidation.py --once
```

## Key Tables

- `memory_units`: `consolidated_at`, `consolidation_failed_at`, `fact_type`, `bank_id`
- `async_operations`: `operation_id`, `operation_type`, `status`, `bank_id`
- API stats endpoint: `/v1/default/banks/hermes/stats` → `pending_consolidation`, `failed_consolidation`
