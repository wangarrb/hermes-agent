# Orphaned Consolidation Units

## Problem

After container restart or when consolidation completes, a small number of
`memory_units` rows may have `consolidated_at=NULL` with `fact_type` in
`('experience','world')` but **no active/pending consolidation operation** to
process them. The consolidation worker considers its job done and goes idle,
but `pending_consolidation` remains > 0, blocking `wait_native_consolidation`
and the offline pipeline.

Typical scenario: 194 unconsolidated → worker processes 192 → 2 remain
orphaned → worker reports "0 processed" on subsequent runs → pipeline stuck.

## Root Cause

Hindsight v0.6.1 does not guarantee that every source fact with
`consolidated_at=NULL` will be picked up by a consolidation round. Some world
facts may lack sufficient source links or fall outside the consolidation
batch window, leaving them permanently unprocessed.

## Detection

```bash
# Check pending_consolidation
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pending_consolidation',0))"

# Check if worker is idle but pending > 0
python3 -c "
import psycopg2
conn = psycopg2.connect('postgresql://hindsight@127.0.0.1:5432/hindsight')
cur = conn.cursor()
cur.execute(\"SELECT fact_type, COUNT(*) FROM memory_units WHERE bank_id='hermes' AND consolidated_at IS NULL GROUP BY fact_type\")
for r in cur.fetchall(): print(f'  {r[0]}: {r[1]}')
cur.execute(\"SELECT status, COUNT(*) FROM async_operations WHERE bank_id='hermes' AND operation_type='consolidation' GROUP BY status\")
for r in cur.fetchall(): print(f'  ops {r[0]}: {r[1]}')
conn.close()
"
```

If `pending_consolidation > 0` but all async_operations are
completed/cancelled (no processing/pending), the remaining units are orphaned.

## Fix Scripts

### fix_orphaned_consolidation.py

`~/.hermes/scripts/fix_orphaned_consolidation.py`

Two modes:
- **Default (no --bypass)**: Creates a new `async_operations` row for the
  worker to pick up. Preferred — lets consolidation actually process the
  units.
- **--bypass**: Marks orphaned units as consolidated directly. Use when
  consolidation has already tried and processed 0 (units are genuinely
  unprocessable).

**Safety**: Both modes check for active/pending consolidation operations
first and refuse to run if any exist (prevents corrupting in-flight work).
Use `--force` to override this check (dangerous).

```bash
# Dry run
python3 ~/.hermes/scripts/fix_orphaned_consolidation.py --dry-run

# Re-queue for worker (preferred)
python3 ~/.hermes/scripts/fix_orphaned_consolidation.py

# Bypass unprocessable units
python3 ~/.hermes/scripts/fix_orphaned_consolidation.py --bypass
```

### hindsight_wait_native_consolidation.py Auto-Bypass

The wait script has a three-phase auto-recovery loop:

1. **Stall detected** (pending>0, processing=0, stall_count>=threshold):
   → `POST /consolidate` to trigger the worker
2. **Wait one cycle**: Let worker process what it can
3. **Still stalled** (same pending count, still no processing):
   → Whatever remains is orphaned → `_fix_orphaned_via_db()` bypass
   → pipeline unblocked

This means the offline pipeline will never permanently block on orphaned
units — it will auto-bypass after at most 2 stall cycles.

## Key Pitfalls

1. **Never bypass while consolidation is active**: The `_fix_orphaned_via_db`
   and `fix_orphaned_consolidation.py` both check `async_operations` for
   processing/pending rows before touching `memory_units`. Without this
   guard, bypassing would mark in-flight source facts as consolidated,
   causing the worker to skip them and produce incomplete observations.

2. **Bypass ≠ consolidated**: Units bypassed by `--bypass` have
   `consolidated_at` set but were not actually processed by the
   consolidation LLM. The timestamp distinguishes them (set by the fix
   script rather than the consolidation worker), but there is no explicit
   `failed` status on individual `memory_units`. The bypass is logged to
   stderr for traceability.

3. **`psycopg2` dependency**: `fix_orphaned_consolidation.py` requires
   `psycopg2-binary`. The inline fallback `_fix_orphaned_via_db()` also
   uses it. If unavailable, the pre-check in `hindsight_wait_native_consolidation.py`
   will silently skip and rely on the auto-bypass loop instead.

4. **Model download case sensitivity**: When downloading embedding models
   for Hindsight (e.g., `BAAI/bge-m3`), the HuggingFace cache is
   case-sensitive. A download with wrong case (`BAAi/bge-m3`) creates a
   separate cache directory (`models--BAAi--bge-m3/`) that the container
   won't find when it looks for `BAAI/bge-m3`. Always use the exact
   model ID from the container's config.
