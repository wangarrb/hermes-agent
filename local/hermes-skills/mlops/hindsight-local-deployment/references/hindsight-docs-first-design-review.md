# Hindsight docs-first design review

Use this reference when the task is to design or change Hermes/Hindsight memory ingestion, staging, consolidation, or cleanup flows.

## Lesson from session

Do not start by inventing a multi-layer architecture or pre-classifier. For Hindsight-related architecture decisions, first ground the proposal in:

1. Official Hindsight docs.
2. The live instance's `/openapi.json`.
3. The live bank config/stats.
4. Existing local scripts/patches, if changes are planned.
5. Only then compare designs.

This was a user correction: the user explicitly objected to blind guessing/design and asked to read docs/search related content before deciding.

## Quick evidence-gathering checklist

### Official docs to check

- Overview: `https://hindsight.vectorize.io/`
- Retain API: `https://hindsight.vectorize.io/developer/api/retain`
- Retain architecture: `https://hindsight.vectorize.io/developer/retain`
- Recall API: `https://hindsight.vectorize.io/developer/api/recall`
- Memory banks/config: `https://hindsight.vectorize.io/developer/api/memory-banks`
- Documents: `https://hindsight.vectorize.io/developer/api/documents`
- Reflect: `https://hindsight.vectorize.io/developer/api/reflect`
- Mental models: `https://hindsight.vectorize.io/developer/api/mental-models`
- Webhooks: `https://hindsight.vectorize.io/developer/api/webhooks`
- Performance: `https://hindsight.vectorize.io/developer/performance`

### Live API probes

```bash
curl -s http://127.0.0.1:8888/health
curl -s http://127.0.0.1:8888/openapi.json > /tmp/hindsight-openapi.json
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/config | jq
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/stats | jq
```

Useful OpenAPI paths to inspect before designing:

- `POST /v1/default/banks/{bank_id}/memories`
- `GET /v1/default/banks/{bank_id}/memories/list`
- `GET /v1/default/banks/{bank_id}/memories/{memory_id}`
- `GET /v1/default/banks/{bank_id}/memories/{memory_id}/history`
- `DELETE /v1/default/banks/{bank_id}/memories/{memory_id}/observations`
- `POST /v1/default/banks/{bank_id}/memories/recall`
- `GET/PATCH/DELETE /v1/default/banks/{bank_id}/config`
- `GET/PATCH/DELETE /v1/default/banks/{bank_id}/documents/{document_id}`
- `POST /v1/default/banks/{bank_id}/consolidate`
- `GET/DELETE /v1/default/banks/{bank_id}/operations/{operation_id}`
- `GET /v1/default/banks/{bank_id}/graph`
- `GET /v1/default/banks/{bank_id}/entities`
- `GET /v1/default/banks/{bank_id}/export`
- `POST /v1/default/banks/{bank_id}/import`
- webhook endpoints for `retain.completed` and `consolidation.completed`

## Facts verified in the session

- Hindsight docs say a full conversation should usually be retained as one item, not message-by-message.
- Retain extracts structured facts/entities/relationships/time; it does not merely store raw text.
- Recall returns structured facts, not raw documents.
- Recall supports `types`, `tags`, `tags_match`, and include options such as chunks/source facts/entities.
- Memory banks are official isolation units: memories in one bank are not visible to another.
- `MemoryItem` in the live OpenAPI includes `content`, `timestamp`, `context`, `metadata`, `document_id`, `entities`, `tags`, `observation_scopes`, `strategy`, and `update_mode` (`replace`/`append`).
- `observation_scopes` supports `combined`, `per_tag`, `all_combinations`, or explicit tag-group lists.
- Live `hermes` bank at the time of the session had `retain_extraction_mode=concise`, `retain_chunk_size=8000`, `enable_observations=false`, `consolidation_llm_batch_size=50`, no pending/failed operations.
- Source/code review found native `retain_extraction_mode="chunks"`: it bypasses LLM fact/entity extraction and stores chunks directly as memory units. This is useful for a separate raw/evidence index bank, not for the clean fact bank.
- Source/OpenAPI/docs review did not find native automatic topic discovery, document-level topic clustering, or a pre-retain stratified sampler. Tags/scopes/entity labels are available primitives, but they do not automatically discover topic strata before retain.
- `entity_labels` with `tag: true` can classify extracted facts into controlled labels/tags during retain. It is post-retain classification and cannot reduce pre-retain LLM calls.
- Provider Batch API support in the checked source is limited to OpenAI/Groq; MiniMax falls back and should not be treated as a way to reduce MiniMax retain work.

## Design implications

- Prefer native Hindsight retain/recall/reflect/consolidation APIs over local reimplementation.
- Do not assume a custom Bronze/Silver/Gold pipeline is necessary until native capabilities and gaps are checked.
- If the goal is to reduce paid retain calls fundamentally, prefer a separate chunks-mode raw index bank plus local topic-selection control plane before clean retain. Do not use `estimated_retain_chunks` as the selection algorithm; use it only as a budget/audit guard.
- If using `entity_labels`, keep labels closed-set and stable; avoid free-text labels as primary topic tags because they become consolidation scopes.
- If considering staging banks, justify them from bank isolation semantics and promotion limitations, not from generic data-lake intuition.
- Before relying on destructive cleanup, verify current-version document deletion/observation lineage behavior. Old GitHub issue reports have described orphan observations after document deletion; treat this as a risk to test, not as current truth.
- If exact fact/observation upsert is needed, first verify whether a native endpoint exists. If not, options are curated re-retain, local proposal only, upstream feature request, or explicit DB fallback with backup/preflight.

## Additional verified pitfalls (2026-05-08)

- Live OpenAPI may expose document/list/recall/consolidate/export/import but still lack exact memory-unit/fact/observation upsert/update/delete endpoints. Do not design “promotion” or “quarantine” as if exact fact editing is natively available.
- `GET /export` in the checked instance behaved like bank template/config export, not full memory export. Do not assume it can promote facts/observations between banks.
- Native consolidation groups facts by exact tag set before LLM calls. Wrong tags are therefore a primary quality risk: they directly become wrong observation scopes.
- Broad source/system tags such as `hermes`, `sqlite`, `incremental` should not be treated as semantic consolidation scopes. Use stable domain/project/topic tags and/or explicit `observation_scopes`.
- `proof_count` can be misleading: in the checked DB all observations had `proof_count=1`; evidence strength was better assessed via `source_memory_ids` length and source_fact lineage.
- Upstream Hindsight now has an official safe round cap: `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND`. For paid backlog runs, use that rather than the older local `MAX_MEMORIES_PER_JOB` patch; still keep proposal/lineage audits before expanding native consolidation.
- Before expanding paid native consolidation, run a read-only contamination/lineage check: document tag distribution, memory-unit tag/text mismatch, observation source ids, missing source targets, and `observation_scopes` distribution.
- Temporary-bank smoke (2026-05-08) confirmed same `document_id` default re-retain replaced old document text/facts; `PATCH /documents/{document_id}` tags propagated to existing memory_units; but `update_mode=append` appended document text while the appended delta fact lost item-level `observation_scopes` (`NULL`). Treat append as unsafe for live-session governance until this is understood or worked around; prefer replace/re-retain for corrected canonical documents.
- `/consolidate` obeys bank `enable_observations`; posting it while observations are disabled can complete a no-op consolidation. Controlled smoke requires enabling observations for the target temp bank, enabling local consolidation worker slots, then disabling/restoring afterwards.
- Temporary-bank smoke showed `per_tag` creates separate observations per tag; broad/system/source tags can therefore create noisy single-tag observations. Keep semantic observation scopes limited to stable domain/project/topic tags, not `hermes`/`sqlite`/`incremental`/generic source tags.
- V4 MiniMax smoke (`hermes_v4_minimax_smoke_20260508`) added an important quality lesson: a bank can have pending=0, docs_without_units=0, broad/system-tag count=0, and still fail semantic recall precision. Recall-smoke showed `patent`/`openclaw`/`user_pref`/`cch` queries returning Egomotion4D/Hindsight/tooling facts because tags co-occurred too broadly. Future gates must include query-level contamination checks and topic/tag co-occurrence drift, not only count/tag audits.
- V4 operational lessons: provider/container switch may exceed 90s due to local embeddings reload; paid runs need generous health timeout and background+notify. If MiniMax JSON parse retry/STUCK persists but docs already have units and no useful progress, cancel only target-bank pending/processing ops, restore normal-local, then audit partial output.
- V4 safety lessons: route credential-like sessions to `manual_review:secret_or_credential_material`, and redact credential-like strings before JSON/Markdown/stdout audit reports.
- Temporary-bank delete smoke confirmed current-version document deletion removed observations sourced from deleted document facts and left no missing source ids in the remaining observations; still run this in temp bank after upgrades before applying destructive cleanup to `hermes`.
- Production cleanup should be discard-first: before deleting a production document, replace/re-retain of an existing production document, clearing derived observations, or doing DB fallback edits, snapshot affected documents/facts/observations/source ids into a discard/quarantine area (e.g. local `~/.hermes/hindsight/discard/` plus optional `hermes_discard` bank with observations disabled). Only mutate production after snapshot verification, backup, and explicit confirmation.
- For session/json-based rebuilds, first run the non-mutating manifest dry-run script `$HOME/.hermes/scripts/hindsight_session_manifest.py`. Default manifest output should omit full conversation content and keep source paths/hashes/cost estimates. Route sessions with multiple project tags or more than four semantic tags to `manual_review` rather than production, because broad mixed scopes are exactly what polluted native observations before. Also route bootstrap/identity/environment-diagnostic sessions to `manual_review` reason `bootstrap_or_environment_diagnostic`; early sessions often contain the assistant self-introduction with AEB/ADAS/autodrive keywords, which otherwise falsely labels Hermes/OpenClaw/debug sessions as `domain:autodrive`. Do not use generic `trajectory/轨迹/尺度/ego-motion` as `project:egomotion4d` triggers; they polluted L2 mono distance/speed sessions. Route broad aggregate memory summaries such as “read all detailed notes/project memories/topics” to `manual_review` reason `broad_aggregate_summary`. Drop deterministic self-reflection skill prompts such as `Review the conversation above and consider saving or updating a skill...` as noise before retain. Lightweight candidate filtering is intentionally simple and deterministic: skip short/chitchat-only conversations such as `hi`/`你好`, pure `继续`, `ok`, short acknowledgements and common assistant boilerplate, but keep semantic prompts such as “继续讨论 Hindsight native consolidation”. Record `candidate_filter_version` in metadata and include it in submit-state comparison so filter changes force preview/re-retain instead of silently reusing old successful state.
- Current legacy offline daily/weekly cron jobs may still publish `hermes-offline-canonical::*` documents to the current `hermes` bank. Before enabling a new native/session pipeline, pause or update those jobs to audit/candidate-only mode so offline V2 does not keep competing with native observations. In the 2026-05-08 implementation pass, the two legacy jobs were paused: `225636f0c7bf` (`hindsight-daily-retain-reflect-consolidate`) and `cabc42eab0e0` (`hindsight-weekly-consolidation-after-daily`). Daily report/wiki cron stayed enabled.
- The first guarded retain runner is `$HOME/.hermes/scripts/hindsight_session_retain_runner.py`. It consumes a lean manifest, rehydrates source JSON content, filters only `action=production`, serializes all metadata values to strings for live Hindsight API compatibility, uses `update_mode=replace`, defaults to dry-run, and requires `--execute --confirm retain-hindsight-session-manifest` before any Hindsight submit. It is incremental-safe: session JSON can grow/change after an earlier manifest; manifest records `source_mtime_ns`, source size/file hash and content hashes; the runner skips unchanged documents only from successful-submit state for the same target bank + `document_id` + `content_sha256`, and updates that state only after successful execute; for async retain, it waits for operation ids to complete successfully before advancing submit-state; dry-run never updates it. Latest broad dry-run on manifest `20260508-021827-session-manifest.jsonl` would submit 1080 records and submitted 0. Later 5-doc smokes on `hermes_v3` exposed taxonomy/cleaning issues; failed smoke docs were cleaned discard-first. Current tiny local-model seed after hardening used manifest `20260508-092638-session-manifest.jsonl`, operation `f15fddbc-a349-41a7-9145-0c454a80fd73`, and yielded 5 documents / 28 experience facts / 0 observations with pending=0 failed=0. A paid MiniMax session/json wrapper now exists: `$HOME/.hermes/scripts/hindsight_minimax_import.py session-manifest-retain-minimax`; dry-run does not switch provider, execute mode requires the retain confirm token, switches 8888 to MiniMax, verifies/patches the target bank, runs the session retain runner, waits, and restores `normal-local` in `finally`. MiniMax smoke #1 on temporary bank `hermes_v3_minimax_smoke_20260508` used the same first 5 production docs, operation `d78514f5-a881-4ae6-b541-4eaed98713f6`, and yielded 5 documents / 15 facts (`world`=13, `experience`=2) / 0 observations with pending=0 failed=0. MiniMax kept more numeric/detail-rich L2 mono distance/speed facts than the local seed, but 3/5 docs still produced no facts and the first production window still contained chat-memo aggregates plus inline reasoning/tool noise in some MEMORY.md sessions; do not expand by blind `--limit N`. Treat both seeds as evidence/fact smoke only; do not enable observations/consolidation yet.
- Operational pitfall from curated MiniMax run: do not run paid `session-manifest-retain-minimax --execute` under a foreground terminal timeout that can kill the wrapper before its `finally` restores local mode. Use background+notify or a runner with enough timeout. If interrupted, immediately run status, restore `normal-local`, inspect the target temp bank stats, and cancel pending operations for that bank before local Ollama continues them. Also do not leave provider-switch health wait at a tight 90s: Hindsight may need >90s to reload local embeddings after a container/provider switch. The wrapper now exposes `--health-timeout-s` (default 300); use a larger value for paid/background smoke runs if startup is slow. During a background paid smoke, treat repeated MiniMax JSON parse retry/STUCK on one op as a budget/safety condition: if target documents already have units and stats show no useful progress, cancel only that target bank's pending/processing operations, restore `normal-local`, and audit the partial bank before any further action.
- The repeatable bank quality audit script is `$HOME/.hermes/scripts/hindsight_bank_quality_audit.py`. It is read-only and writes JSON/Markdown reports under `$HOME/.hermes/hindsight/reports/`. Command shape: `python3 $HOME/.hermes/scripts/hindsight_bank_quality_audit.py --bank hermes --recall-smoke --stem <stem>`. Native list APIs may omit `document_id`/`source_memory_ids`; the script defaults to `--db-fallback auto` and uses read-only PostgreSQL only when API lineage looks sparse. It now redacts credential-like strings in JSON/Markdown/stdout; still do not intentionally query or expose secrets. This DB fallback is audit-only and must not become a mutation path. Latest scripted audit on `hermes` (`2026-05-08-hermes-bank-quality-audit-scripted-v3`) matched manual SQL metrics: documents=186, memory_units=6622, observations=393, source_refs=4499, missing_source_refs=2, docs_without_units=10; conclusion unchanged: archive/evidence recall OK, broad native consolidation not OK.

## Recommended output format for future design review

1. Evidence summary: docs/API/live-state facts.
2. Capability matrix: native supported / unclear / missing.
3. Risks and unknowns.
4. 2-3 design options with tradeoffs.
5. Minimal safe next experiment, preferably read-only or dry-run first.
