# Hindsight v0.6.1 API-First Operations

Use this when operating a Hindsight v0.6.1+ bank. The point is to use official APIs for status, reporting, reversible movement, and targeted repair before falling back to direct PostgreSQL.

## Read-only status first

Preferred queue/status snapshot:

```bash
python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --skip-psql --json
```

What it uses:

- `GET /health`
- `GET /v1/default/banks/{bank_id}/stats`
- `GET /v1/default/banks/{bank_id}/operations?exclude_parents=true`
- `GET /v1/default/banks/{bank_id}/stats/memories-timeseries?period=1d&time_field=created_at`
- `GET /v1/default/banks/{bank_id}/audit-logs/stats?period=1d`

Why `exclude_parents=true`: parent batch operations can look pending even when claimable child work is complete/in-flight elsewhere. For drain status, count child/real work first. Include parents only when debugging batch orchestration.

## Safe operation inspection

```bash
HINDSIGHT_BASE_URL="${HINDSIGHT_BASE_URL:-http://127.0.0.1:8888}"

# List failed child operations without payloads
curl -s "$HINDSIGHT_BASE_URL/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&limit=20"

# Inspect one operation; payload off by default
curl -s "$HINDSIGHT_BASE_URL/v1/default/banks/hermes/operations/<operation_id>?include_payload=false"
```

Rules:

- Do not use `include_payload=true` for broad scans. Payloads can be large and may include private source text.
- Keep operation retry/cancel scoped to one explicit operation ID or a reviewed list.
- Retrying a failed operation can create new provider calls; treat as a mutation requiring confirmation.
- Deleting/cancelling an operation is destructive and can interrupt work; require explicit confirmation.

Script confirm tokens:

```text
retry-hindsight-operation
cancel/delete: delete-hindsight-operation
```

## Observability for reports

Use v0.6.1 endpoints in daily/weekly reports:

```bash
HINDSIGHT_BASE_URL="${HINDSIGHT_BASE_URL:-http://127.0.0.1:8888}"

curl -s "$HINDSIGHT_BASE_URL/v1/default/banks/hermes/stats/memories-timeseries?period=7d&time_field=created_at"
curl -s "$HINDSIGHT_BASE_URL/v1/default/banks/hermes/audit-logs/stats?period=7d"
curl -s "$HINDSIGHT_BASE_URL/v1/default/banks/hermes/audit-logs?limit=100"
```

Interpretation:

- `created_at` = ingest time. Use it for operational throughput.
- `mentioned_at` / `occurred_start` = event time. Use it for migrated corpora when ingest time is one burst and not the real knowledge timeline.
- Missing observability endpoint is `unknown`, not zero.
- Daily report should still include offline wrapper/provider state and auxiliary compression costs; Hindsight API endpoints do not cover every local wrapper cost.

## Export/import snapshots

Endpoints:

```bash
HINDSIGHT_BASE_URL="${HINDSIGHT_BASE_URL:-http://127.0.0.1:8888}"

curl -s "$HINDSIGHT_BASE_URL/v1/bank-template-schema"
curl -s "$HINDSIGHT_BASE_URL/v1/default/banks/hermes/export" > bank-template.json
curl -s -X POST "$HINDSIGHT_BASE_URL/v1/default/banks/temp-bank/import?dry_run=true" \
  -H 'Content-Type: application/json' --data @bank-template.json
```

Use cases:

- temp-bank validation,
- proposal review fixtures,
- lightweight rollback material before targeted repair,
- bank-template portability checks.

Limits:

- Treat export/import as bank-template movement, not full physical DB backup.
- For high-risk migrations or schema upgrades, still take a DB snapshot/pg_dump or filesystem snapshot.
- `dry_run=false` import is a production mutation and needs explicit user go/no-go plus rollback plan.

Confirm token for script-level non-dry-run import:

```text
import-hindsight-bank-template
```

## Targeted repair before reset

Prefer this order for local damage:

1. `POST /documents/{document_id}/reprocess`
2. `POST /consolidation/recover`
3. `POST /mental-models/{mental_model_id}/refresh`
4. `POST /entities/{entity_id}/regenerate` when entity-specific repair is justified; note OpenAPI marks it deprecated.
5. Only then consider quarantine/temp-bank rebuild or full DB reset.

Script confirm tokens:

```text
reprocess-hindsight-document
recover-hindsight-consolidation
refresh-hindsight-mental-model
regenerate-hindsight-entity
```

Production rule: export/snapshot first, identify exact IDs, dry-run/preview locally where possible, execute one targeted repair, then verify recall/search/audit before proceeding to the next.

## Worker slot semantics

v0.6.1 worker reservation log shape:

```text
max_slots=9, reservations=[consolidation=1], shared_pool=8
```

Interpretation:

- `WORKER_MAX_SLOTS` is the total pool.
- `WORKER_*_MAX_SLOTS` are reservations/limits inside the total pool, not additive workers.
- Raising external `parallel_batches` without raising safe DB/recall capacity can increase timeouts.
- Tune by recall/search fanout:

```text
llm_batch_size * active_parallelism * observation-scope_parallelism
```

Default publication profile remains `20 x 3` with `MAX_MEMORIES_PER_ROUND=60`, `RECALL_BUDGET=low`, source fact cap `4096`, and per-observation cap `256`.

## Read-only DB backend

`HINDSIGHT_API_READ_DATABASE_URL` is useful only when it points to a real read-only backend/replica.

Do not set it to the same writer URL and claim safety. That only adds configuration complexity without reducing mutation risk.

## PostgreSQL fallback policy

Direct DB reads are acceptable for:

- schema/migration forensics,
- exact queries not yet exposed by API,
- validating API counters when a bug is suspected.

Direct DB writes are not part of the normal workflow. If unavoidable, they require a separate plan, snapshot, rollback path, and explicit user confirmation.
