# Orphaned Units After Consolidation Drain

## Problem

After a consolidation drain completes, a small number of `memory_units` may
remain with `consolidated_at=NULL` and `fact_type IN ('experience','world')`.
The consolidation worker no longer picks them up (it considers the bank
fully processed), but `pending_consolidation` stays non-zero, blocking
`wait_native_consolidation` and the downstream pipeline.

Typical count: 1–5 units, usually `fact_type='world'`.

## Root Cause

These units have insufficient source links or fail internal consolidation
eligibility filters. The worker processes 0 of them even when triggered
manually:

```
[CONSOLIDATION] bank=hermes completed: 0 processed
```

## Diagnosis

1. Check pending count:
   ```bash
   curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats \
     | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pending_consolidation',0))"
   ```

2. If non-zero but worker is idle (no processing/pending async_operations):
   ```python
   import psycopg2
   conn = psycopg2.connect("postgresql://hindsight@127.0.0.1:5432/hindsight")
   cur = conn.cursor()
   cur.execute("SELECT fact_type, COUNT(*) FROM memory_units WHERE bank_id='hermes' AND consolidated_at IS NULL GROUP BY fact_type")
   for r in cur.fetchall(): print(r)
   cur.execute("SELECT status, COUNT(*) FROM async_operations WHERE bank_id='hermes' AND operation_type='consolidation' GROUP BY status")
   for r in cur.fetchall(): print(r)
   conn.close()
   ```

3. Trigger consolidation manually to confirm it processes 0:
   ```bash
   curl -s -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/consolidate' \
     -H 'Content-Type: application/json' -d '{}'
   ```
   Then check logs for `completed: 0 processed`.

## Fix: Bypass Orphaned Units

Mark them as consolidated to unblock the pipeline:

```bash
python3 ~/.hermes/scripts/fix_orphaned_consolidation.py --bypass
```

The script has a safety check: it **refuses** to run when active
consolidation operations exist, preventing corruption of in-flight work.
Use `--force` only when you are certain the active ops are stale.

## Prevention: Auto-Recovery in wait_native_consolidation

The `hindsight_wait_native_consolidation.py` script has built-in
auto-recovery for this scenario:

1. Detects stalled consolidation (pending>0, no processing ops, N consecutive polls)
2. Triggers `POST /consolidate` to retry
3. If consolidation processes 0 (orphaned), automatically calls
   `_fix_orphaned_via_db()` to bypass the stuck units
4. Pipeline unblocks without manual intervention

The `fix_orphaned_consolidation.py` pre-check at script startup also
handles orphaned units from container restarts, but only when no active
ops exist.

## Safety Rules

- Never run `--bypass` while consolidation is actively processing.
  It would mark in-flight source facts as consolidated, causing them
  to be skipped by the worker.
- The `has_active_consolidation()` check in both scripts prevents this.
- If you must force-bypass despite active ops, use `--force` and accept
  the risk of data inconsistency.
