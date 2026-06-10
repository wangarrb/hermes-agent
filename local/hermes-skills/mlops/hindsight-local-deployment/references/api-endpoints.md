# Hindsight API Endpoints Reference

实测 Docker 版本 Hindsight API endpoints（与官方文档略有不同）。

## Base URL

- API: `http://localhost:8888`
- Control Plane UI: `http://localhost:9999`

## Endpoints

### Health Check

```
GET /health
```

Response:
```json
{"status":"healthy","database":"connected"}
```

### Banks

#### List Banks

```
GET /v1/default/banks
```

Response:
```json
{
  "banks": [
    {
      "bank_id": "hermes",
      "name": "hermes",
      "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
      "mission": "",
      "created_at": "...",
      "updated_at": "..."
    }
  ]
}
```

#### Create Bank

Bank 在首次 retain 时自动创建，无需显式创建。

### Memories

#### Retain Memories

```
POST /v1/default/banks/{bank_id}/memories
```

Request:
```json
{
  "items": [
    {
      "content": "对话内容...",
      "document_id": "session_20260410_xxx",
      "context": "hermes_conversation"
    }
  ]
}
```

Response:
```json
{
  "success": true,
  "bank_id": "hermes-sessions",
  "items_count": 1,
  "async": false,
  "usage": {
    "input_tokens": 2805,
    "output_tokens": 573,
    "total_tokens": 3378
  }
}
```

#### Recall Memories

```
POST /v1/default/banks/{bank_id}/memories/recall
```

Request:
```json
{
  "query": "搜索内容",
  "max_tokens": 4096
}
```

### Documents

```
GET /v1/default/banks/{bank_id}/documents
```

### Entities

```
GET /v1/default/banks/{bank_id}/entities
```

## Pitfalls

1. **Not `/api/v1/banks`** — 正确路径是 `/v1/default/banks`
2. **Not `/retain`** — 正确路径是 `/memories`（POST）
3. **Bank 自动创建** — 无需显式调用 create bank API

## Rate Limit Handling

当遇到 429 错误时：

```json
{
  "type": "error",
  "error": {
    "type": "rate_limit_error",
    "message": "usage limit exceeded, 5-hour usage limit reached for Token Plan Plus (1500/1500 used), resets at 2026-05-03T10:00:00+08:00"
  }
}
```

解决方案：
1. 等待 reset time
2. 切换 LLM provider
3. 降低请求频率

## v0.6.1 Useful Endpoints

### Async Operations

```
GET    /v1/default/banks/{bank_id}/operations
GET    /v1/default/banks/{bank_id}/operations/{operation_id}?include_payload=false
POST   /v1/default/banks/{bank_id}/operations/{operation_id}/retry
DELETE /v1/default/banks/{bank_id}/operations/{operation_id}
```

Notes:
- `operations` supports `status`, `type`, `limit`, `offset`, and `exclude_parents` query params.
- Known statuses: `pending`, `processing`, `completed`, `failed`, `cancelled`.
- Known types include `retain`, `consolidation`, `refresh_mental_model`, `file_convert_retain`, `webhook_delivery`.
- Prefer this API for queue/drain status before direct DB reads.
- Default status checks should use `exclude_parents=true` to avoid parent batch-operation false positives.
- Only use `include_payload=true` when debugging a specific operation, because payloads can be large or sensitive.
- `POST /retry` can trigger new provider work; treat it as mutation. Local wrapper confirm token: `retry-hindsight-operation`.
- `DELETE` cancels pending operations; treat it as mutation. Local wrapper confirm token: `delete-hindsight-operation`.

### Reports / Observability

```
GET /v1/default/banks/{bank_id}/stats/memories-timeseries?period=7d&time_field=created_at
GET /v1/default/banks/{bank_id}/audit-logs/stats?period=7d
GET /v1/default/banks/{bank_id}/audit-logs
```

Use these for daily/weekly Hindsight reports instead of relying only on local state DB counters.

`time_field=created_at` means ingest time. Use `mentioned_at` or `occurred_start` when auditing migrated corpora by event time.

### Reversible Bank Movement

```
GET  /v1/default/banks/{bank_id}/export
POST /v1/default/banks/{bank_id}/import
GET  /v1/bank-template-schema
```

Use export/import for temp-bank, rollback, and proposal-review workflows where possible. `POST /import?dry_run=true` validates a template without applying it; `dry_run=false` is a mutation requiring explicit approval. This is a bank-template snapshot, not a replacement for pg_dump/filesystem snapshots before schema upgrades.

### Targeted Repair

```
POST /v1/default/banks/{bank_id}/documents/{document_id}/reprocess
POST /v1/default/banks/{bank_id}/entities/{entity_id}/regenerate
POST /v1/default/banks/{bank_id}/mental-models/{mental_model_id}/refresh
POST /v1/default/banks/{bank_id}/consolidation/recover
```

Prefer targeted repair before full DB reset. Still snapshot/export first for production banks. `entities/{id}/regenerate` is marked deprecated in the v0.6.1 OpenAPI observed locally; prefer document reprocess, consolidation recover, or mental-model refresh when they match the failure.

## Docker Logs

```bash
docker logs hindsight --tail 50
```

查看 rate limit 错误：
```bash
docker logs hindsight 2>&1 | grep -E "429|rate_limit"
```