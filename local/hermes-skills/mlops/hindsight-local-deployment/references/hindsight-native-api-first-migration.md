# Hindsight Native API-First Migration Notes

## Context

When extending the local Hermes/Hindsight offline pipeline, prefer reusing Hindsight's native high-level API/SDK instead of duplicating Hindsight internals in local scripts. Local code should act as orchestration/governance around Hindsight, not as a parallel memory system.

## Native interfaces verified locally

Local Hindsight exposes OpenAPI at `/openapi.json` and has the `hindsight_client` Python package installed in the `hindsight` conda env.

Useful native endpoints/classes:

- `hindsight_client.Hindsight(base_url, api_key=None, timeout=...)`
- `retain`, `retain_batch`
- `recall(types=..., tags=..., trace=..., include_source_facts=..., include_chunks=...)`
- `reflect(response_schema=..., tags=..., fact_types=..., include_facts=...)`
- `list_memories(type=..., search_query=..., limit=..., offset=...)`
- `get_memory`, `get_observation_history`
- `get_bank_config`, `update_bank_config`
- `list_directives`, `create_directive`, `update_directive`
- `list_mental_models`, `create_mental_model`, `refresh_mental_model`
- REST-only/important endpoints observed via OpenAPI:
  - `GET /v1/default/banks/{bank_id}/documents`
  - `GET /v1/default/banks/{bank_id}/documents/{document_id}`
  - `PATCH /v1/default/banks/{bank_id}/documents/{document_id}`
  - `DELETE /v1/default/banks/{bank_id}/documents/{document_id}`
  - `GET /v1/default/banks/{bank_id}/operations`
  - `DELETE /v1/default/banks/{bank_id}/operations/{operation_id}`
  - `POST /v1/default/banks/{bank_id}/operations/{operation_id}/retry`
  - `GET /v1/default/banks/{bank_id}/entities`
  - `GET /v1/default/banks/{bank_id}/graph`
  - `GET /v1/default/banks/{bank_id}/stats`
  - `GET/PATCH /v1/default/banks/{bank_id}/config`

## Target architecture

Hindsight should own the data plane:

- durable storage
- embeddings
- entity/link/graph maintenance
- recall/rerank
- reflect/consolidation/mental models/directives
- operations/status/config

Local Hermes scripts should own the control/governance plane:

- SQLite/session extraction
- paid LLM cost budget and provider switching
- cron orchestration/locking/logging
- eval gates
- conflict severity policy
- human confirmation for destructive operations
- wiki candidate generation
- raw provenance scanning in Hermes session DB

## Recommended migration sequence

### P0: Add a native client adapter

Create one `hindsight_native_client.py`-style wrapper for local scripts:

1. Prefer `hindsight_client.Hindsight` where SDK covers the operation.
2. Use REST/OpenAPI for endpoints the SDK does not expose.
3. Use direct PostgreSQL only as a named fallback with warning and tests.
4. Keep all API/DB quirks in the adapter; do not scatter `requests`, `urllib`, or `psql` calls through business scripts.

### P1: Move read-only paths API-first

Migrate audit/lineage/status/eval reads before changing writes:

- status/config/operations -> `/health`, `/stats`, `/operations`, `/config`
- documents -> `/documents` and `/documents/{id}`
- memory units -> `list_memories`, `get_memory`, `get_observation_history`
- entities/graph -> `/entities`, `/graph`

Current local implementation status (2026-05-07):

- `hindsight_native_client.py` centralizes health/stats/config/documents/memories/operations/entities/graph/recall/reflect access and guards destructive operations with confirm tokens.
- `cancel_hindsight_bank_pending.py` uses official operations API; dry-run by default, execution requires `--execute --confirm-cancel delete-hindsight-operation`.
- `hindsight_conflict_audit.py`, `hindsight_lineage_trace.py`, `hindsight_offline_audit.py`, and `hindsight_offline_v2_audit.py` are API-first for Hindsight data-plane reads; local SQLite is retained only for Hermes raw-span provenance.
- `hindsight_minimax_import.py --purge-sqlite` is now official-API cleanup and dry-run by default; destructive execution requires `--execute-purge --confirm-purge-documents delete-hindsight-document --confirm-purge-operations delete-hindsight-operation`.

DB fallback is acceptable only for missing API features, e.g. exact memory-units-by-document queries, and should be explicit rather than hidden in normal maintenance paths.

### P2: Promote only selected high-level principles to mental-models/directives

After a full rebuild succeeds, treat Hindsight canonical observations/documents as the primary source of truth. `v2_cards/local canonical` is an intermediate sidecar/proposal/gate artifact, not a second canonical store. Do not bulk-copy local cards into mental-models/directives.

Only promote a small number of stable cross-cycle principles after review:

- stable user preferences that affect future reasoning -> mental model
- long-lived project decisions/state that should survive rebuilds -> mental model
- reusable tooling lessons or quality principles -> mental model
- open questions that should stay active across cycles -> mental model
- hard behavioral/system rules -> directive only when they should be injected into prompts

Do not turn ordinary project facts or transient canonical observations into directives; directives affect prompt behavior and can become over-strong historical instructions.

### P3: Retire direct DB publish from the main path

Local direct DB publishing of canonical observations duplicates Hindsight internals (`documents`, `chunks`, `memory_units`, `entities`, `unit_entities`, `memory_links`, embeddings). This is fragile across Hindsight upgrades and should not be the default.

Preferred path:

- publish high-level summaries through native `retain_batch`/documents/mental-models when acceptable;
- if exact observation upsert semantics are required and Hindsight lacks the endpoint, keep local proposal only or add/upstream a native endpoint;
- retain direct DB publish only as an explicit, tested compatibility fallback with preflight and backup.

### P4: Move offline reflect/consolidation toward Hindsight `/reflect`

Local scripts may keep budget/cache/gate orchestration, but should prefer Hindsight `/reflect` with `response_schema`, `tags`, `fact_types`, and `include_facts` instead of maintaining a separate OpenAI-compatible LLM caller, retry system, JSON parser, and prompt schema indefinitely.

## Upstream feature gaps worth requesting/patching

- list memory units by `document_id`, tags, time range, and fact type
- exact upsert/import observation or memory-unit API
- batch delete/replace documents by prefix/tag with backup/dry-run
- native cost/dry-run estimate for retain/consolidation/reflect
- full bank export/import including documents and memory units, not just bank template
- worker queue status that reflects in-memory pending work, not only DB-visible operations
- provenance API: observation -> source facts -> source documents/raw chunks

## Verification checklist

Before replacing any direct DB path:

1. Compare API output vs current DB query on representative docs/memories.
2. Run existing hardening tests with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
3. Verify status remains healthy and pending/processing/failed are zero.
4. Confirm no paid provider switch or publish occurs during read-only migration tests.
5. Keep destructive writes behind explicit confirmation and fail-closed preflight.
