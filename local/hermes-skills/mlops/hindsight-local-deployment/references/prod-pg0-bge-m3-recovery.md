# Hindsight production pg0 + bge-m3 recovery notes

Session distilled from 2026-05-11 recovery after a mid-retain crash/reboot.

## Trigger
Use this when Hindsight API crash-loops during startup, especially after switching to local Chinese-capable embeddings (`BAAI/bge-m3`) or after a machine reboot while paid retain work is still queued.

## Protected production paths
Do not delete or overwrite these without explicit production-reset approval:
- `$HOME/.hindsight-docker/instances/hindsight/data` — production pg0 PostgreSQL data directory.
- `$HOME/.hindsight-docker/instances/hindsight/instance.json` — production pg0 metadata.
- `$HOME/.hindsight-docker/installation/` — pg0/PostgreSQL binaries used by production DB.
- `$HOME/.cache/huggingface/hub/models--BAAI--bge-m3` — local bge-m3 cache.

Known stale/misleading path:
- `$HOME/.pg0/instances/hindsight/data` — old/stale DB in this environment; do not use it for production queue status.

## Root cause pattern
1. Hindsight env can show the right paid provider (`glm-5` / DashScope) while API still fails before health.
2. With `HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-m3`, `SentenceTransformer` can still do optional HuggingFace metadata probes even when model weights are cached.
3. In startup lifespan this can crash as:
   - `RuntimeError: Cannot send a request, as the client has been closed`
   - stack through `SentenceTransformer(...)`, `transformers.utils.peft_utils.find_adapter_config_file`, `hf_hub_download`.
4. After embeddings are fixed, a second failure can occur if the wrong/stale pg0 is queried or if production pg0 is not running.

## Required env for cached bge-m3 startup
When bge-m3 is already cached locally, force offline mode:
- `HF_HUB_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_HOME=/home/hindsight/.cache/huggingface`
- `HUGGINGFACE_HUB_CACHE=/home/hindsight/.cache/huggingface/hub`
- `TRANSFORMERS_CACHE=/home/hindsight/.cache/huggingface/hub`

Minimal bge-m3 cache test inside the Hindsight image:
```bash
sg docker -c 'docker run --rm --network host --entrypoint /app/api/.venv/bin/python3 \
  -v $HOME/.cache/huggingface:/home/hindsight/.cache/huggingface \
  -v $HOME/.cache/torch:/home/hindsight/.cache/torch \
  -e HF_HOME=/home/hindsight/.cache/huggingface \
  -e HUGGINGFACE_HUB_CACHE=/home/hindsight/.cache/huggingface/hub \
  -e TRANSFORMERS_CACHE=/home/hindsight/.cache/huggingface/hub \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  ghcr.io/vectorize-io/hindsight:latest \
  -c "from sentence_transformers import SentenceTransformer; SentenceTransformer(\"BAAI/bge-m3\"); print(\"load_ok\")"'
```

## Production pg0 checks
Start/check production pg0, not stale `~/.pg0`:
```bash
$HOME/.hindsight-docker/installation/18.1.0/bin/pg_ctl \
  -D $HOME/.hindsight-docker/instances/hindsight/data status || true

$HOME/.hindsight-docker/installation/18.1.0/bin/pg_ctl \
  -D $HOME/.hindsight-docker/instances/hindsight/data \
  -l $HOME/.hindsight-docker/instances/hindsight/data/start.log \
  -w -t 60 start
```

Query production queue:
```bash
$HOME/.hindsight-docker/installation/18.1.0/bin/psql \
  -h 127.0.0.1 -p 5432 -U hindsight -d hindsight -v ON_ERROR_STOP=1 <<'SQL'
SELECT operation_type,status,count(*)
FROM async_operations
GROUP BY 1,2
ORDER BY 1,2;

SELECT operation_type,count(*)
FROM async_operations
WHERE status='pending'
  AND task_payload IS NOT NULL
  AND (next_retry_at IS NULL OR next_retry_at <= now())
GROUP BY 1;

SELECT operation_type,count(*)
FROM async_operations
WHERE status='pending' AND task_payload IS NULL
GROUP BY 1;
SQL
```

## `parent_pending_null_payload` interpretation
`batch_retain` rows with `status='pending'` and `task_payload IS NULL` are parent bookkeeping rows.
They are not claimable worker jobs and should not trigger manifest resubmission.
Focus on:
- claimable `retain` pending rows (`task_payload IS NOT NULL`),
- `retain` processing rows,
- failed rows,
- document/unit growth.

If `retain_pending=0` and `retain_processing=0` but parent rows remain, then do a separate targeted bookkeeping cleanup/mark-complete after snapshot and verification.

## Cleanup policy
Only after explicit user confirmation, safe cleanup candidates include:
- `$HOME/.pg0` stale DB,
- `$HOME/.hindsight-docker/backups`,
- `$HOME/.hindsight-docker/instances/hindsight/data.pre-reset-*`,
- `$HOME/.hermes/hindsight/backups`,
- old stopped containers such as `hindsight-minimax-disabled` and `tmp_hs_pg0_pkg_*`.

Before deleting, verify candidate paths do not contain protected paths and production `/health` is healthy or production pg0 is confirmed running.
