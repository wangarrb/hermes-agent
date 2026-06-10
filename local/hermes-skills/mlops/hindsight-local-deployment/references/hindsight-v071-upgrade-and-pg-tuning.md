# Hindsight 0.6.1 → 0.7.1 Upgrade & PostgreSQL Tuning

## Upgrade Summary (2026-06-02)

Upgraded from `ghcr.io/vectorize-io/hindsight:0.6.1` to `:0.7.1`.

### What Changed

1. **API path restructuring** — all bank-scoped endpoints moved from `/v1/banks/{bank_id}/...` to `/v1/default/banks/{bank_id}/...`
2. **Schema changes** — `observations` table removed; all data now in `memory_units` with `fact_type` discriminator. `operations` → `async_operations`. `consolidated_at` column removed from `documents`.
3. **Retain API format** — now requires `{"items": [...]}` wrapper instead of flat `{"content": ..., "source": ...}`. `search_vector` is a generated column; attempting to insert non-DEFAULT values fails.
4. **Package rename** — `hindsight-api` → `hindsight-api-slim` in 0.7.1.
5. **New features** — `bank_config_api`, `file_upload_api`, `mcp` support in `/version` response.

### API Path Mapping (0.6.1 → 0.7.1)

| 0.6.1 | 0.7.1 |
|-------|-------|
| `POST /v1/banks/{bank}/recall` | `POST /v1/default/banks/{bank}/memories/recall` |
| `POST /v1/banks/{bank}/retain` | `POST /v1/default/banks/{bank}/memories` |
| `POST /v1/banks/{bank}/reflect` | `POST /v1/default/banks/{bank}/reflect` |
| `GET /v1/banks/{bank}/stats` | `GET /v1/default/banks/{bank}/stats` |
| `GET /v1/banks/{bank}/operations` | `GET /v1/default/banks/{bank}/operations` |
| `GET /v1/banks/{bank}/documents` | `GET /v1/default/banks/{bank}/documents` |
| `POST /v1/banks/{bank}/consolidate` | `POST /v1/default/banks/{bank}/consolidate` |
| `GET /v1/banks/{bank}/export` | `GET /v1/default/banks/{bank}/export` |
| `POST /v1/banks/{bank}/import` | `POST /v1/default/banks/{bank}/import` |

Use `GET /openapi.json` to discover the full route list for any version.

### Upgrade Procedure

1. Pull new image: `docker pull ghcr.io/vectorize-io/hindsight:0.7.1`
2. Export current env vars: `docker inspect hindsight > /tmp/hindsight_inspect.json`, extract `Env` array
3. Write env file: `jq -r '.[0].Env[]' /tmp/hindsight_inspect.json > /tmp/hindsight_env_file.txt`
4. Stop old container: `docker stop hindsight`
5. Recreate with same env: `docker run -d --name hindsight --env-file /tmp/hindsight_env_file.txt --network host --restart unless-stopped ghcr.io/vectorize-io/hindsight:0.7.1`
6. Verify: `curl -s http://127.0.0.1:8888/version` should show `api_version: 0.7.1`
7. Data persists in PostgreSQL — no data loss from container swap
8. Remove old image: `docker rmi ghcr.io/vectorize-io/hindsight:0.6.1`

### Known Issues After Upgrade

- **Retain via direct API** fails with `cannot insert a non-DEFAULT value into column "search_vector"` — the column is now auto-generated. Hermes SDK may need update.
- **Recall parameter `limit`** is ignored with warning: `Unknown parameters ignored: [limit]`
- **`observations` table queries** fail — use `memory_units WHERE fact_type='observation'` instead
- **`operations` table queries** fail — use `async_operations` instead
- **`consolidated_at` column** removed from `documents` — use `async_operations` status instead

## PostgreSQL Tuning for Vector Search

### Problem

After upgrading to 0.7.1, recall queries were timing out (60s+). Root cause: PostgreSQL `shared_buffers=128MB` (default) was far too small for the 1.7GB `memory_units` table with 75K+ embeddings. HNSW vector similarity search and graph traversal were doing heavy `IO.DataFileRead` waits (57-112s per query).

### Diagnosis

```bash
# Check PG config
psql -c "SHOW shared_buffers"  # was 128MB

# Check table sizes
psql -c "SELECT pg_size_pretty(pg_total_relation_size('memory_units'))"  # 1764 MB

# Check cache hit ratio
psql -c "SELECT sum(heap_blks_hit)/(sum(heap_blks_hit)+sum(heap_blks_read))*100 FROM pg_statio_user_tables"  # was 93.1%

# Check HNSW indexes
psql -c "SELECT indexname, indexdef FROM pg_indexes WHERE tablename='memory_units' AND indexdef LIKE '%vector%'"

# Watch DB waits in real-time
docker logs hindsight --since 5m 2>&1 | grep "DB_WAITS"
```

Key log patterns indicating the problem:
- `[DB POOL] Slow acquire: 60.831s` — connection pool starving
- `[DB_WAITS] wait=IO.DataFileRead state=active age=57s` — disk IO bottleneck
- `[LinkExpansion] Entity expansion timed out after 10.0s` — graph traversal hitting disk
- `RuntimeError: generator didn't stop after athrow()` — asyncpg timeout cascade

### Fix

PG config file: `/home/wyr/.hindsight-docker/instances/hindsight/data/postgresql.conf`

```ini
# --- Tuned for Hindsight 0.7.1 vector search (2026-06-02) ---
shared_buffers = 2GB
effective_cache_size = 12GB
work_mem = 256MB
random_page_cost = 1.1
max_parallel_workers_per_gather = 4
```

Restart PG:
```bash
/home/wyr/.hindsight-docker/installation/18.1.0/bin/pg_ctl -D /home/wyr/.hindsight-docker/instances/hindsight/data restart
```

### Results

| Metric | Before | After |
|--------|--------|-------|
| Recall latency | 60s+ (timeout) | 2-2.5s |
| Stats latency | 1.6s | 3s |
| Health latency | 10ms | 3ms |
| DB IO waits | 57-112s | <1s |

### Rationale

- **shared_buffers=2GB**: memory_units table is 1.7GB; 2GB buffers can hold the hot working set in RAM. System has 16GB total, PG was only using 128MB.
- **effective_cache_size=12GB**: tells the query planner that up to 12GB of OS cache is available, encouraging index scans over sequential scans.
- **work_mem=256MB**: graph traversal and vector search use large sorts/hash joins; 4MB was causing temp files on disk.
- **random_page_cost=1.1**: HNSW index access is random IO; default 4.0 penalizes it too heavily, making PG avoid the index. 1.1 treats SSD random access as nearly free.
- **max_parallel_workers_per_gather=4**: allows parallel query execution for large scans.

### PG Installation Details

Hindsight's PostgreSQL is NOT a Docker container — it's a system-level installation at:
- Binary: `/home/wyr/.hindsight-docker/installation/18.1.0/bin/postgres`
- Data: `/home/wyr/.hindsight-docker/instances/hindsight/data`
- Config: same as data dir (`postgresql.conf`)
- Connection: `host=127.0.0.1, port=5432, user=hindsight, dbname=hindsight`

No local `psql` client installed; use Python `psycopg2` from Hermes venv (`~/.hermes/hermes-agent/venv/bin/python3`) for DB queries.

### Stale Bank Indexes

The `memory_units` table has HNSW indexes for many old temporary banks (`hermes_tmp_*`, `external_*`). These indexes consume space and slow down writes. Consider dropping indexes for deleted/unused banks:

```sql
-- List all vector indexes with their bank_id filter
SELECT indexname, indexdef FROM pg_indexes
WHERE tablename='memory_units' AND indexdef LIKE '%vector_cosine_ops%';

-- Drop indexes for banks that no longer exist
-- (check bank existence first: SELECT id FROM banks)
```
