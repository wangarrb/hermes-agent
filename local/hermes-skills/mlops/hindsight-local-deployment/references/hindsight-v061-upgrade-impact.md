# Hindsight v0.6.x Upgrade Impact

Use this when evaluating an upgrade from a v0.5.2-era local Hindsight container to v0.6.x.

## Local baseline observed on 2026-05-13 (upgrade execution)

- Pre-upgrade running image: `ghcr.io/vectorize-io/hindsight:latest`, `org.opencontainers.image.version=0.5.2`, revision `712a8628`.
- 0.6.1 image present locally (id `79069181071d`, 6.26GB) but `:latest` still pointed to 0.5.2.
- DB Alembic version: `m3rg3h3ad5f6` (already the 0.6.1 head; migrations were a fast no-op).
- Pre-upgrade counts: 1485 documents, 13184 memory_units, 254349 memory_links.
- `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false` was set; migrations had to be run manually.

## Important upstream changes in v0.6.x

- Official consolidation cap is `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND`; prefer this over the old local `MAX_MEMORIES_PER_JOB` naming.
- Consolidation fanout is bounded by default with:
  - `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET=low`
  - `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096`
  - `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION=256`
  - `HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA=false`
- Worker slot reservations are official: `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS`, `HINDSIGHT_API_WORKER_RETAIN_MAX_SLOTS`, `HINDSIGHT_API_WORKER_REFRESH_MENTAL_MODEL_MAX_SLOTS`, etc. They are reservations inside `WORKER_MAX_SLOTS`, not additive pools.
- `HINDSIGHT_API_DB_STATEMENT_TIMEOUT` is available as a Postgres query safety net.
- Async operations gained better observability: `processing`/`cancelled`, retry metadata, parent filtering, cancellation support, and improved parent/child error propagation.
- Retain and consolidation received reliability fixes: reduced retain memory pressure, batch-retain atomicity, deferred memory_links FK to reduce deadlock risk, observation source dedupe/upsert fixes, and recall/entity fixes.
- v0.6.1 adds optional read-only DB backend for recall queries (`HINDSIGHT_API_READ_DATABASE_URL`).
- Integrations changed, especially OpenClaw/Claude Code/CLI. Self-driving-agents CLI was removed to its own repo; not relevant unless this local deployment uses it.

## Upgrade risk assessment

Impact is moderate, not a trivial image pull:

1. Database migrations are required. Do not upgrade a production bank without snapshot/export and a rollback plan.
2. This local deployment has `HINDSIGHT_API_RUN_MIGRATIONS_ON_STARTUP=false`; pulling/recreating the container alone may start new code against an old schema.
3. Replace local-wrapper env assumptions with official v0.6.x envs before rollout.
4. Keep `auto_retain=false` and avoid production writes during the upgrade window.
5. Upgrade only when read-only status shows no active/pending work.

## Recommended safe rollout

1. Read-only status:
   ```bash
   python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --json
   ```
2. Stop any active pipeline processes first, then stop the container:
   ```bash
   # Kill pipeline orchestrator + watchdog + any worker subprocesses
   pgrep -af 'hindsight_memory_pipeline|hindsight_full_pipeline_watchdog|hindsight_minimax_import|offline_hindsight_reflect' | awk '{print $1}' | xargs -r kill -TERM
   # Force-stop and remove the container (login shell may lack docker group → use sg)
   sg docker -c "docker rm -f hindsight"
   ```
3. Snapshot/export the affected DB/bank. Record pre-upgrade counts:
   ```bash
   BACKUP_DIR=$HOME/.hindsight-docker/backups/upgrade-0.6.1-$(date +%Y%m%d-%H%M%S)
   mkdir -p "$BACKUP_DIR"
   PSQL="$HOME/.hindsight-docker/installation/18.1.0/bin/psql -h 127.0.0.1 -U hindsight -d hindsight"
   export PGPASSWORD=hindsight
   $PSQL -c "SELECT version_num FROM alembic_version;" -t > "$BACKUP_DIR/alembic_pre.txt"
   $PSQL -c "SELECT count(*) FROM documents;" -t > "$BACKUP_DIR/doc_count_pre.txt"
   $PSQL -c "SELECT count(*) FROM memory_units;" -t > "$BACKUP_DIR/mu_count_pre.txt"
   $HOME/.hindsight-docker/installation/18.1.0/bin/pg_dump -h 127.0.0.1 -U hindsight -d hindsight -Fc -f "$BACKUP_DIR/hindsight_pre_upgrade.dump"
   $HOME/.hindsight-docker/installation/18.1.0/bin/pg_dumpall -h 127.0.0.1 -U hindsight --globals-only -f "$BACKUP_DIR/globals_pre_upgrade.sql"
   sha256sum "$BACKUP_DIR"/hindsight_pre_upgrade.dump "$BACKUP_DIR"/globals_pre_upgrade.sql > "$BACKUP_DIR/SHA256SUMS"
   ```
4. **Permanently fix the `:latest` tag** so future `recreate_container()` calls (from any wrapper mode) pick up the new version. Without this, setting `HINDSIGHT_IMAGE` for one invocation does NOT survive subsequent pipeline stages that also recreate the container:
   ```bash
   sg docker -c "docker tag ghcr.io/vectorize-io/hindsight:0.6.1 ghcr.io/vectorize-io/hindsight:latest"
   ```
   Verify with: `docker image inspect ghcr.io/vectorize-io/hindsight:latest --format '{{index .Config.Labels "org.opencontainers.image.version"}}'` → `0.6.1`.
   Then clean up old image tags (export to trash first, then delete):
   ```bash
   TRASH_DIR="$HOME/.local/share/Trash/docker-images"; mkdir -p "$TRASH_DIR"
   OLD_ID=f0b40a3cc33d  # the old 0.5.2 image ID
   sg docker -c "docker save $OLD_ID" > "$TRASH_DIR/hindsight-0.5.2.tar"
   sg docker -c "docker rmi swr.cn-north-4.myhuaweicloud.com/ddn-k8s/ghcr.io/vectorize-io/hindsight:latest" 2>/dev/null || true
   ```

5. Run migrations deliberately using a temporary 0.6.1 container (required because `RUN_MIGRATIONS_ON_STARTUP=false`):
   ```bash
   sg docker -c "
   docker run --rm --network host \
     ghcr.io/vectorize-io/hindsight:0.6.1 \
     python3 -c '
   import sys
   sys.path.insert(0, \"/app\")
   from hindsight_api.migrations import run_migrations
   run_migrations(\"postgresql://hindsight:hindsight@127.0.0.1:5432/hindsight\")
   print(\"MIGRATIONS_COMPLETE\")
   '
   "
   ```
   Note: `run_migrations()` requires the database_url as a positional argument (not from env).
6. Start the container via the wrapper. After the permanent tag fix (step 4), the `HINDSIGHT_IMAGE` override is no longer needed; `normal-local` will use the now-correct `:latest`:
   ```bash
   python3 $HERMES_HOME/scripts/hindsight_minimax_import.py normal-local
   ```
   This recreates the container, copies runtime tuning, applies best-effort patches
   (which will correctly report "not applicable" on 0.6.1), and restores the normal-local
   provider config. Old 0.5.2-era patches are not upgrade blockers; their absence on
   0.6.1 is expected.
7. Verify:
   - `curl -s http://127.0.0.1:8888/version` → `"api_version":"0.6.1"` and `"features":{"observations":true,...}`
   - `/health` → `{"status":"healthy","database":"connected"}`
   - Document/memory_unit counts match pre-upgrade snapshot
   - Bank stats from `/v1/default/banks/<bank>/stats`
   - Bank config now supports `consolidation_max_memories_per_round` (was unsupported on 0.5.2)
   - `consolidation_recall_budget` may still appear as "not supported" even on 0.6.1; this is acceptable
   - Recall smoke via `hindsight_recall` or `/recall` endpoint returns meaningful results
   - Preflight passes with `blocking_count=0`
   - async operations have no unexpected pending/failed rows
9. Only then resume scheduled/offline pipelines.

## Legacy local patches and v0.6.x impact

The local deployment previously carried two in-container best-effort patches for v0.5.2-era images:

1. `patch_hindsight_minimax_json_parser.py`
   - Target: `/app/api/hindsight_api/engine/providers/openai_compatible_llm.py`.
   - Purpose: make structured LLM JSON parsing more tolerant of fenced JSON, `<think>...</think>`, trailing commas, and provider text before the JSON object; also added a conservative 429/rate-limit backoff hook.
   - v0.6.1 status: the exact retry block anchor changed, so the patch is now skipped as not applicable. Upstream still has basic `_strip_code_fences`, but not all local V4 repairs. MiniMax-M2.7 retain smoke passed without it; monitor logs for JSON parse errors or 429 bursts before reintroducing any new patch.

2. `patch_hindsight_native_consolidation_budget.py`
   - Target: `/app/api/hindsight_api/engine/consolidation/consolidator.py`.
   - Purpose: add local `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB` so one `/consolidate` operation could not drain the whole production bank.
   - v0.6.1 status: superseded by official `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND`; the consolidator now tracks `round_remaining` and caps fetch size by the round budget. Skipping this patch is expected when `MAX_MEMORIES_PER_ROUND` is configured.

Operational rule: patch scripts must be best-effort compatibility hooks, not upgrade blockers. If their anchors are missing on v0.6.x, log a concise "not applicable" message and continue, then verify with retain/recall/consolidation smoke.

## Skill/script consequences

- Skill docs and preflight should use `max_memories_per_round` as canonical and keep `max_memories_per_job` only as a legacy alias.
- Do not write `consolidation_parallel_batches` into bank config; upstream does not expose it. Keep `parallel_batches` only as an external/offline pipeline fanout budget.
- Status helpers should prefer API status when possible and keep psql as optional forensic fallback. Current packaged helper: `hindsight_consolidation_status.py --skip-psql --json` uses `/operations?exclude_parents=true`, `/stats/memories-timeseries`, and `/audit-logs/stats`.
- Preflight should verify the official v0.6.x runtime env keys from Docker inspect and distinguish worker-slot reservations from additive worker pools.
- Prefer v0.6.1 operation APIs before DB forensics: list operations with `exclude_parents`, inspect a single operation with `include_payload` only when needed, retry failed operations through `/operations/{id}/retry`, and cancel pending work through `DELETE /operations/{id}`.
- Use `/stats/memories-timeseries` and `/audit-logs/stats` for daily reports and anomaly checks instead of only reading local state DBs.
- Use `/export`, `/import`, and `/v1/bank-template-schema` for reversible temp-bank/rollback workflows where possible.
- Use targeted repair endpoints (`documents/{id}/reprocess`, `entities/{id}/regenerate`, `mental-models/{id}/refresh`, `consolidation/recover`) before full DB resets.
- Consider `HINDSIGHT_API_READ_DATABASE_URL` only if a real read-only replica/backend exists; do not point it blindly at the same production writer and call that a safety improvement.
