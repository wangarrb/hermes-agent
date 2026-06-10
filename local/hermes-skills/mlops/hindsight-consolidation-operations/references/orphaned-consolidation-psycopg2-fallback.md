# Orphaned Consolidation + psycopg2 Fallback

## Problem

After a Hindsight container restart, `memory_units` may have `consolidated_at=NULL` with no
matching active `async_operations`. The API reports `pending_consolidation > 0` but the
consolidation worker cannot pick them up. The pipeline's `hindsight_wait_native_consolidation.py`
detects a stall, tries orphan bypass, which calls `_fix_orphaned_via_db()` requiring `psycopg2`.

The offline pipeline runs under the **Hermes-agent venv** Python
(`~/.hermes/hermes-agent/venv/bin/python`), NOT the system `miniconda` Python. If `psycopg2`
is not installed in the venv, every bypass attempt fails with `ModuleNotFoundError: No module
named 'psycopg2'`, and the wait loop polls indefinitely (up to 86400s timeout), stalling the
entire daily pipeline.

## Root Cause Chain

```
Container restart
  → some memory_units left with consolidated_at=NULL, no async_operations
    → API pending_consolidation > 0
      → wait script detects stall
        → POST /consolidate (trigger) — worker picks up nothing (orphaned)
          → stall persists
            → orphan bypass → _fix_orphaned_via_db() → import psycopg2 → ImportError
              → bypass fails silently, retry next poll cycle
                → infinite loop until timeout (24h)
```

## Fix (3 layers)

### Layer 1: Install psycopg2 in the venv

```bash
# Use Tsinghua mirror — PyPI direct is slow from China
HTTPS_PROXY=127.0.0.1:7890 \
  /home/wyr/.hermes/hermes-agent/venv/bin/pip install psycopg2-binary \
  -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

Verify: `/home/wyr/.hermes/hermes-agent/venv/bin/python -c "import psycopg2; print(psycopg2.__version__)"`

### Layer 2: Multi-path DB fallback in scripts

Both `hindsight_wait_native_consolidation.py` and `fix_orphaned_consolidation.py` should
try DB access in this order:

1. **psycopg2** (preferred — parameterized queries, row counts)
2. **psql CLI** — check `shutil.which("psql")`, then known path
   `/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql`
3. **docker exec psql** — `docker exec hindsight psql -U hindsight -d hindsight -c "..."`

When psycopg2 raises `ImportError`, fall through to psql CLI instead of crashing.

### Layer 3: Bounded bypass retries

The wait script's main loop must cap orphan bypass attempts:

```python
bypass_attempts: int = 0
MAX_BYPASS_ATTEMPTS: int = 3

# In the stall detection block:
if bypass_attempts >= MAX_BYPASS_ATTEMPTS:
    print("[orphan_bypass] giving up after N attempts, pending remains stuck")
    # Let the timeout handle it instead of infinite retry
else:
    bypass_attempts += 1
    n = _fix_orphaned_via_db(args)
    if n and n > 0:
        bypass_attempts = 0  # reset on success
```

## Manual Recovery

If the pipeline is already stuck, manual cleanup via psql:

```bash
PSQL="/home/wyr/.hindsight-docker/installation/18.1.0/bin/psql"
DB="postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight"

# 1. Check for orphaned units
$PSQL $DB -t -c "
SELECT fact_type, count(*) FROM memory_units
WHERE consolidated_at IS NULL AND consolidation_failed_at IS NULL
  AND fact_type IN ('experience','world')
GROUP BY fact_type;"

# 2. Check for failed consolidation
$PSQL $DB -t -c "
SELECT fact_type, count(*) FROM memory_units
WHERE consolidation_failed_at IS NOT NULL
GROUP BY fact_type;"

# 3. Cancel stuck processing operations
$PSQL $DB -c "
UPDATE async_operations SET status='cancelled', completed_at=NOW()
WHERE status IN ('processing','pending');"

# 4. Fix orphaned (bypass mode — marks as consolidated)
$PSQL $DB -c "
UPDATE memory_units SET consolidation_failed_at=NULL
WHERE consolidation_failed_at IS NOT NULL;"
$PSQL $DB -c "
UPDATE memory_units SET consolidated_at=NOW()
WHERE consolidated_at IS NULL AND fact_type IN ('experience','world');"

# 5. Or use the fix script (now with psql fallback)
/home/wyr/.hermes/hermes-agent/venv/bin/python \
  ~/.hermes/scripts/fix_orphaned_consolidation.py --bank hermes --bypass
```

## Verification

After cleanup, check API stats:

```bash
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats | \
  python3 -c "import sys,json; d=json.load(sys.stdin); \
  print(f'pending_consolidation: {d[\"pending_consolidation\"]}'); \
  print(f'failed_consolidation: {d[\"failed_consolidation\"]}');"
```

Both should be 0. Then test the wait script:

```bash
/home/wyr/.hermes/hermes-agent/venv/bin/python \
  ~/.hermes/scripts/hindsight_wait_native_consolidation.py --once
```

Should return `"ready": true`.

## Key Tables

- `memory_units`: has `consolidated_at` and `consolidation_failed_at` columns
- `async_operations`: has `operation_type='consolidation'`, `status` in (pending/processing/completed/failed/cancelled)
- `pending_consolidation` (API stat): counts units the consolidation worker should process but hasn't
- `failed_consolidation` (API stat): counts units where consolidation failed (`consolidation_failed_at IS NOT NULL`)

## venv vs miniconda

| Python | Path | psycopg2 |
|--------|------|----------|
| miniconda | `/home/wyr/miniconda/bin/python3` (3.13) | Installed by default |
| hermes-agent venv | `/home/wyr/.hermes/hermes-agent/venv/bin/python` (3.11) | Must install separately |

The offline pipeline uses the venv Python. Always verify psycopg2 availability in the venv
after Hermes upgrades (which may recreate the venv).
