# Hindsight v0.7.1 Upgrade and PostgreSQL Tuning

Upgrade notes and performance tuning for Hindsight v0.7.1, recorded 2026-06-02.

## Upgrade from 0.6.1 to 0.7.1

### Pre-upgrade state (2026-06-02)

- Running image: `ghcr.io/vectorize-io/hindsight:0.6.1` (also tagged as `:latest`)
- DB Alembic version: `m3rg3h3ad5f6` (unchanged from 0.6.1)
- Data: 5795 documents, 80868 memory_units, 1052660 memory_links
- PostgreSQL: system service at `/home/wyr/.hindsight-docker/instances/hindsight/data`

### Upgrade steps

1. Pull new image:
   ```bash
   docker pull ghcr.io/vectorize-io/hindsight:0.7.1
   ```

2. Stop and remove old container (preserves DB data):
   ```bash
   docker stop hindsight && docker rm hindsight
   ```

3. Recreate with same env vars:
   ```bash
   # Export env from old container first (if not already saved)
   docker inspect hindsight-old --format '{{range .Config.Env}}{{println .}}{{end}}' > /tmp/hindsight_env.txt

   # Recreate with new image
   docker run -d --name hindsight --network host --env-file /tmp/hindsight_env.txt \
     ghcr.io/vectorize-io/hindsight:0.7.1
   ```

4. Verify:
   ```bash
   curl -s http://127.0.0.1:8888/version
   # Expected: {"api_version":"0.7.1", "features":{"observations":true,...}}
   ```

5. Clean up old images:
   ```bash
   docker rmi ghcr.io/vectorize-io/hindsight:0.6.1
   docker rmi swr.cn-north-4.myhuaweicloud.com/ddn-k8s/ghcr.io/vectorize-io/hindsight:0.6.1
   ```

### Schema changes in 0.7.1

- `search_vector` column in `memory_units` is now `GENERATED ALWAYS` (auto-computed tsvector)
- No new Alembic migration required if already on `m3rg3h3ad5f6`
- API paths unchanged from 0.6.1: `/v1/default/banks/{bank_id}/memories`, `/recall`, `/reflect`

## PostgreSQL Tuning for Vector Search

### Problem: Recall timeout (60s+)

After upgrading to 0.7.1, recall queries timed out (60s+). Root cause: PostgreSQL `shared_buffers` was only 128MB for a 1.7GB `memory_units` table with HNSW vector indexes.

Symptoms from logs:
```
[DB POOL] Slow acquire: 60.831s
[DB_WAITS] pid=... wait=IO.DataFileRead state=active age=57s query='... embedding <=> ...'
```

### Solution: Tune PostgreSQL for vector workload

Edit `/home/wyr/.hindsight-docker/instances/hindsight/data/postgresql.conf`:

```ini
# --- Tuned for Hindsight 0.7.1 vector search (2026-06-02) ---
shared_buffers = 2GB
effective_cache_size = 12GB
work_mem = 256MB
random_page_cost = 1.1
max_parallel_workers_per_gather = 4
```

Then restart PostgreSQL:
```bash
/home/wyr/.hindsight-docker/installation/18.1.0/bin/pg_ctl \
  -D /home/wyr/.hindsight-docker/instances/hindsight/data restart
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Recall latency | 60s+ (timeout) | 2-2.5s |
| Stats latency | 15s+ (timeout) | 1.6-3s |
| Reflect latency | 72-145s | 30s |
| Retain latency | N/A | 200ms |

Cache hit ratio improved from ~93% to expected >98% after warmup.

### Why these settings

- **shared_buffers=2GB**: HNSW indexes are random-IO heavy. Default 128MB causes constant disk reads for a 1.7GB table. 2GB fits the hot portion of `memory_units` + indexes.
- **random_page_cost=1.1**: Default 4.0 penalizes random IO (HNSW traversals). Lower value lets the planner choose index scans over sequential scans.
- **effective_cache_size=12GB**: Hints to planner that much of the 16GB system RAM is available for caching (realistic for dedicated Hindsight workloads).
- **work_mem=256MB**: Allows larger in-memory sorts/hashes for graph traversal queries.
- **max_parallel_workers_per_gather=4**: Enables limited parallelism for complex queries.

### System requirements

- Minimum 16GB RAM for 2GB shared_buffers + OS + other services
- SSD storage (HNSW random IO on HDD would still be slow)
- Dedicated or near-dedicated Hindsight workload (shared DB servers may need lower settings)

## Known Issues in 0.7.1

### search_vector generated column error

Symptom:
```
asyncpg.exceptions.GeneratedAlwaysError: cannot insert a non-DEFAULT value into column "search_vector"
DETAIL: Column "search_vector" is a generated column.
```

This error appears in logs (23 occurrences observed) but is caught internally and does not block recall/reflect/retain operations. It indicates 0.7.1 internal code paths that attempt to INSERT into the generated column, but the error is handled gracefully.

Status: Non-blocking, monitor for future upstream fix.

### topenrouter model name prefix

When using topenrouter (`tp-api.chinadatapay.com:8000/v1`):
- ✅ Correct: `deepseek-v4-flash` (no prefix)
- ❌ Wrong: `openai/deepseek-v4-flash` (with prefix) → 401 Invalid token

Hindsight config uses the correct format. Failed operations (60 observed) were from other sources using the wrong prefix.

## API Verification Commands

```bash
# Health (should be <10ms)
curl -s -w "\n%{time_total}s" http://127.0.0.1:8888/health

# Version
curl -s http://127.0.0.1:8888/version

# Stats (should be <5s after tuning)
curl -s -w "\n%{time_total}s" http://127.0.0.1:8888/v1/default/banks/hermes/stats

# Recall (should be <5s after tuning)
curl -s -m 30 -w "\n%{time_total}s" -X POST \
  http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}'

# Retain
curl -s -m 30 -X POST \
  http://127.0.0.1:8888/v1/default/banks/hermes/memories \
  -H "Content-Type: application/json" \
  -d '{"items": [{"content": "test content"}]}'

# Reflect (may take 30-120s depending on query)
curl -s -m 120 -X POST \
  http://127.0.0.1:8888/v1/default/banks/hermes/reflect \
  -H "Content-Type: application/json" \
  -d '{"query": "project status"}'
```

## Data Integrity After Upgrade

PostgreSQL data is independent of container version. Verify counts:

```bash
~/.hermes/hermes-agent/venv/bin/python3 -c "
import psycopg2
conn = psycopg2.connect(host='127.0.0.1', port=5432, dbname='hindsight', user='hindsight', password='hindsight')
cur = conn.cursor()
cur.execute('SELECT count(*) FROM documents')
print('Documents:', cur.fetchone()[0])
cur.execute('SELECT count(*) FROM memory_units')
print('Memory units:', cur.fetchone()[0])
cur.execute('SELECT count(*) FROM memory_links')
print('Memory links:', cur.fetchone()[0])
conn.close()
"
```

Expected (2026-06-02): 5710+ documents, 80000+ memory_units, 1M+ memory_links.
