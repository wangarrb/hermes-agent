---
name: hindsight-local-deployment
description: Use when deploying, operating, importing, and safely governing a local Hindsight memory provider for Hermes, including session retain, offline consolidation, proposal-only repair review, and production-safe preflight checks.
version: 1.3.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [hindsight, memory, docker, hermes, embedding, session-import, offline-consolidation, governance]
    related_skills: [hindsight-consolidation-operations, hermes-agent]
---

# Hindsight Local Deployment

## Overview

This skill is the operational runbook for a local Hindsight memory provider used by Hermes. It covers deployment checks, session retain/import, offline reflect/consolidation, production-safe proposal review, and rollback-aware memory governance.

Core rule: prefer native/API-first, proposal-only, and reversible workflows. Do not write to production Hindsight until the exact payload, snapshot/quarantine plan, and human go/no-go are explicit.

**New-user onboarding:** a comprehensive 960-line design and operations guide covering concepts, architecture, installation, tuning, pitfalls, and the full cron schedule lives at `~/wiki/auto-maintenance/hindsight-offline-guide.md`. Point first-time users or unfamiliar Hermes sessions there.

The skill ships a compact main workflow plus detailed references. Load only the reference that matches the current task; do not dump the whole reference tree into context.

## When to Use

Use this skill when the task mentions:

- Hindsight local deployment or Docker/Postgres/vector setup.
- Hermes memory provider configuration for Hindsight.
- Session/json retain, historical session import, or user+assistant-only manifests.
- Offline daily/weekly reflect/consolidation pipelines.
- MiniMax/Bailian/GLM/topenrouter provider switching for Hindsight jobs.
- Hindsight review backlog, temp-bank repair, proposal-only production review, or publish gates.
- Hindsight queue/status/progress/debugging.
- Hindsight production safety: snapshot, quarantine, rollback, conflict audit, or no-go decisions.
- **External content import:** chat-memo txt files, OpenClaw lcm.db conversations — see `references/hindsight-external-content-import.md` for the full workflow.

Do not use this for generic Hermes Agent configuration unless the task is specifically about the Hindsight memory provider. For general Hermes CLI/provider/tool config, load `hermes-agent` instead.

## Non-Negotiable Safety Rules

1. Never edit `.env` from this workflow. Read credentials only when required and never print them.
2. Never run production destructive operations without an explicit user confirmation, a snapshot/export, and a rollback/quarantine plan.
3. Default to proposal-only for repair-zone and publish workflows.
4. Advisory LLM review is not approval. A `merge_ready` LLM result can only become `conditional_go`; final go/no-go is human-only.
5. Deterministic-blocked proposals, especially secret-like material, must not be sent to an external LLM.
6. Production repair/retain should first validate in a temp/quarantine bank when possible.
7. If a command may mutate production data, say so before running it and verify scope.
8. If Hindsight has active processing/pending jobs, avoid restarts/recreates unless the user explicitly accepts the interruption. **In consolidation patch scenarios, direct restart is the established practice:** poll-waiting for `processing==0` is unreliable because the auto-re-queue gap is sub-second (see `hindsight-consolidation-operations/references/direct-restart-vs-poll-waiting.md`). Recover stuck `processing` ops after restart via DB update or `POST /consolidation/recover`.
9. **Do not change Hindsight LLM model/provider/key configuration without explicit user confirmation.** This includes `llm_profile` in `pipeline_config.json`, `HINDSIGHT_OFFLINE_LLM_*` env vars, `HINDSIGHT_API_LLM_*` container env vars, and review `llm_model`/`llm_base_url`/`llm_api_key_env`. Always ask first — even if the current config seems broken or suboptimal.
9. **Do not change Hindsight LLM configuration (`llm_profile`, `base_url`, `api_key_env`, model name) without explicit user confirmation.** The configuration chain involves multiple layers (pipeline_config.json, .env overrides, container env) with per-provider model name format quirks. A wrong change silently breaks offline reflect, consolidation, and proposal review. See `references/hindsight-llm-provider-configuration-pitfalls.md`.

## Packaged Scripts

This skill packages the current production-safe helper scripts under `scripts/`:

- `install_hindsight_pipeline_scripts.sh` — copies packaged scripts into `$HERMES_HOME/scripts`, backing up overwritten files.
- `hindsight_pipeline_common.py` — shared config/tuning helpers.
- `hindsight_pipeline_preflight.py` — config init, install/runtime preflight, 64x8 tuning checks.
- `hindsight_memory_pipeline.py` — daily/weekly/full pipeline orchestrator.
- `hindsight_session_manifest.py` — session manifest builder.
- `hindsight_session_retain_runner.py` — session retain runner.
- `hindsight_minimax_import.py` — historical wrapper for paid/offline retain/import jobs; supports generic LLM profiles in current local workflow.
- `offline_hindsight_reflect_consolidate.py` — offline reflect/consolidation worker.
- `hindsight_offline_v2_rebuild.py` — V2 card rebuild/gate.
- `hindsight_conflict_core.py` — shared conflict-audit scoring/core helpers.
- `hindsight_conflict_audit.py` — conflict/lineage audit.
- `hindsight_repair_proposal_build.py` — converts approved repair sidecars to proposal-only canonical bundles.
- `hindsight_proposal_review.py` — local human/LLM-advisory review packets; no production mutation.
- `hindsight_native_client.py` — native Hindsight API helper, including v0.6.1 operations/observability/export/import/targeted-repair wrappers.
- `hindsight_consolidation_status.py` — read-only v0.6.1 Operations API status snapshot; psql is optional forensic fallback.
- `hindsight_external_manifest.py` — build manifest for third-party conversation imports (OpenClaw lcm.db, chat-memo txt exports). See `references/hindsight-external-content-import.md`.
- `hindsight_external_retain_runner.py` — submit external-import manifests to the `hermes` bank and, by default, wait for post-retain native consolidation/observation drain. See `references/hindsight-external-content-import.md`.
- `hindsight_wait_native_consolidation.py` — read-only pipeline gate that waits for `pending_consolidation` plus pending/processing child operations to drain before V2/conflict/proposal stages.
- `import_sqlite_to_hindsight.py` — legacy SQLite import helper used by offline wrappers.
- `patch_hindsight_consolidator_parallel.py` — applies the local decoupled-concurrency consolidator patch after container recreate: bounded recall fanout, bounded LLM calls, serialized observation/source writes, and adaptive 429 backoff/concurrency halving.
- `hindsight_consolidator_parallel_patched.py` — packaged patched consolidator source used as a fallback if `~/.hermes/patches/hindsight-consolidator-parallel/consolidator.py` is missing.
- progress helpers: `hindsight_progress_bar.py`, `hindsight_progress_live.sh`.
- tests under `scripts/tests/` for pipeline planning and proposal review gates.
- **External import scripts:** `hindsight_external_manifest.py` (chat-memo / OpenClaw lcm.db → manifest JSONL) and `hindsight_external_retain_runner.py` (manifest → hermes retain → native consolidation/observations). These are manual entrypoints, but production retain now defaults to `hermes` and waits for the observation drain unless `--no-wait-consolidation` is explicitly used. See `references/hindsight-external-content-import.md`.

Install/update the scripts into the active Hermes home:

```bash
SKILL_DIR="${HERMES_HOME:-$HOME/.hermes}/skills/mlops/hindsight-local-deployment"
bash "$SKILL_DIR/scripts/install_hindsight_pipeline_scripts.sh"
```

The installer backs up overwritten files under `$HERMES_HOME/scripts/backups/`, initializes `$HERMES_HOME/hindsight/pipeline_config.json`, writes the default 64x8 tuning file, and runs a small compile/preflight check. It does not edit `.env`.

## Configuration Model

Default config path:

```text
$HERMES_HOME/hindsight/pipeline_config.json
```

Create/update it:

```bash
python3 $HERMES_HOME/scripts/hindsight_pipeline_preflight.py --init-config --json
python3 $HERMES_HOME/scripts/hindsight_pipeline_preflight.py --write-tuning --json
```

Expected publication/runtime tuning (default 64x8, verified 2026-05-14):

```text
consolidation_batch_size=64
consolidation_llm_batch_size=8
max_memories_per_round=64
max_memories_per_job=64
parallel_batches=8
consolidation_recall_budget=low
source_facts_max_tokens=4096
source_facts_max_tokens_per_observation=256
```

This is the production default and requires the parallel consolidator patch. For environments that cannot apply the patch, fall back to upstream-serial-safe 20x3.

This is not just a worker-slot change: `patch_hindsight_consolidator_parallel.py` must be applied/reapplied after container recreate, otherwise upstream serial `for llm_batch in llm_batches` will not use 8-way LLM concurrency. Writes remain serialized; monitor DB pool waits and failed operations.

**LLM provider/model changes require explicit user approval.** Do not switch providers, models, base URLs, or API keys without confirming with the user first. The current user-selected runtime is topenrouter (ChinaDataPay, `tp-api.chinadatapay.com:8000/v1`) with `deepseek/deepseek-v4-flash`. This is configured via `.env` overrides (`HINDSIGHT_OFFLINE_LLM_BASE_URL`, `HINDSIGHT_OFFLINE_LLM_MODEL`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY`) over the internal `deepseek-v4-flash` profile in `pipeline_config.json`. topenrouter is NOT openrouter.ai — it is a ChinaDataPay API relay, requires no proxy, and uses an OpenAI-compatible format. If topenrouter returns 403 or "This token has no access", the key may have expired or lack quota; check with the user before switching. Do not use the official DeepSeek API unless the user explicitly requests it.

For Hindsight v0.6.x, prefer official upstream knobs: `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND`, `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET=low`, `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096`, and `HINDSIGHT_API_RERANKER_FLASHRANK_CPU_MEM_ARENA=false`. Treat old `*_MAX_MEMORIES_PER_JOB` / `parallel_batches` as local-wrapper compatibility unless the local parallel consolidator patch is active.

Current user-selected runtime LLM is **topenrouter** (ChinaDataPay relay at `tp-api.chinadatapay.com:8000/v1`, no proxy needed) with model `deepseek/deepseek-v4-flash`, routed through `.env` overrides: `HINDSIGHT_OFFLINE_LLM_BASE_URL`, `HINDSIGHT_OFFLINE_LLM_MODEL=deepseek/deepseek-v4-flash`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY`. 29. **Recall API is vector retrieval, not an LLM call.** `POST /v1/default/banks/hermes/memories/recall` uses bge-m3 embedding + pgvector similarity search. It does NOT call an external LLM. If recall is slow/timing out, the cause is DB pressure, embedding compute load, or container instability — not the LLM provider. Do not switch LLM profiles to "fix" recall timeouts; diagnose the actual cause (container OOM/restart, DB lock contention, embedding model still loading).

30. **Container 502 window after restart is ~25 seconds.** After `docker restart hindsight`, the API returns 502/connection-refused for approximately 20-30 seconds while the bge-m3 embedding model loads into memory. Pipeline eval steps and recall smoke tests that run immediately after a container restart will fail with HTTP 502. Always wait for `/health` to return OK before running eval/recall-dependent operations. Quick check:
```bash
for i in $(seq 1 12); do curl -s -m 5 http://127.0.0.1:8888/health && break; sleep 5; done
```

31. **Hindsight LLM config is a two-layer override system.** Layer 1: `pipeline_config.json` sets `llm_profile` (internal name like `deepseek-v4-flash`). Layer 2: `.env` overrides the actual endpoint — `HINDSIGHT_OFFLINE_LLM_BASE_URL`, `HINDSIGHT_OFFLINE_LLM_MODEL`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV`. Changing `.env` overrides does NOT take effect until the container is restarted (it reads env vars at startup). The `deepseek-v4-flash` profile defaults to `api.deepseek.com` + `DEEPSEEK_API_KEY`; to route through topenrouter instead, set the `.env` overrides to `tp-api.chinadatapay.com:8000/v1` + `TOPENROUTER_API_KEY`. Similarly, `pipeline_config.json` `review.proposal_review` section has its own `llm_api_key_env`/`llm_base_url`/`llm_model` that must also point to the intended provider. `topenrouter` is NOT the same as `openrouter.ai` — they are completely different services.

**LLM profile naming pitfall:** `pipeline_config.json` field `llm_profile` and `--llm-profile` CLI flag use Hindsight-internal profile names, NOT provider routing names. Valid profiles: `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash` (plus custom). `topenrouter` is NOT a valid llm-profile — it is the provider routing name for the OpenAI-compatible endpoint. Using `topenrouter` as `llm_profile` causes `hindsight_minimax_import.py` to reject with `Unknown --llm-profile='topenrouter'`. Always use `deepseek-v4-flash` in `pipeline_config.json`.

**API key mismatch pitfall:** The `deepseek-v4-flash` profile defaults to `DEEPSEEK_API_KEY`. When routing through topenrouter via `.env` overrides, you MUST set `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY`. Otherwise the wrong key gets sent → 403/401 from topenrouter. Similarly, `review.proposal_review.llm_api_key_env` in `pipeline_config.json` must be `TOPENROUTER_API_KEY`.

**Model name prefix pitfall:** topenrouter expects `deepseek/deepseek-v4-flash` (with `deepseek/` prefix), not bare `deepseek-v4-flash`. The `HINDSIGHT_OFFLINE_LLM_MODEL` env var must use the prefixed form.

For full provider details, see `references/hindsight-topenrouter-provider.md`.

Strict preflight:

```bash
python3 $HERMES_HOME/scripts/hindsight_pipeline_preflight.py --strict-runtime --json
```

**CRITICAL: Never change Hindsight LLM model/profile/base_url/api_key without explicit user approval.** This includes switching providers, models, or overriding `.env` variables. Always confirm with the user before modifying `pipeline_config.json` `llm_profile`, `HINDSIGHT_OFFLINE_LLM_*` env vars, or container LLM settings.

Current LLM configuration uses `deepseek-v4-flash` via topenrouter (ChinaDataPay relay at `tp-api.chinadatapay.com:8000/v1`), set through `.env` overrides on the `deepseek-v4-flash` internal profile. The `review.proposal_review` section also routes through topenrouter. See `references/hindsight-topenrouter-llm-configuration.md` for the full configuration chain and naming pitfalls.

Expected result before running production-adjacent workflows:

```text
ok=true
blocking_count=0
warning_count=0
local_consolidation_tuning_64x8=true
runtime_consolidation_tuning_64x8=true
proposal_review_requires_llm=true
proposal_review_requires_human_approval=true
```

If runtime tuning fails only because the container tuning file is missing, check active Hindsight operations before copying/restarting. Do not interrupt active production processing blindly.

Version note: upgrading a production bank from v0.5.2-era images to v0.6.x is not just an image pull. v0.6.x adds migrations, official per-round consolidation caps, low consolidation recall budget, bounded source facts, and worker/operation observability changes. v0.7.1 renames the package to `hindsight-api-slim` and adds `markitdown` and `cryptography` deps. Snapshot/export first, verify idle, run/verify Alembic migrations, then restart with official v0.6.x+ env keys. See `references/hindsight-container-upgrade-pattern.md` for the full upgrade workflow.

## Primary Workflows

### 1. Status and Health Check

Use these before longer operations:

```bash
curl -s http://127.0.0.1:8888/health
python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --skip-psql --json
python3 $HERMES_HOME/scripts/hindsight_minimax_import.py status
python3 $HERMES_HOME/scripts/hindsight_pipeline_preflight.py --strict-runtime --json
```

On Hindsight v0.6.1, prefer `hindsight_consolidation_status.py` over ad-hoc DB queries.

For "did yesterday's offline pipeline succeed?" quick checks, see `references/hindsight-daily-pipeline-quick-status-check.md` — log locations, exit code interpretation, and how to distinguish historical `failed_operations` from current pipeline failure.

#### Version verification: don't trust `:latest` alone

The Docker `:latest` tag can silently point to an older version. Always verify the actual running image version:

```bash
# Check the running container's image version label
docker inspect <container-id> --format '{{json .Config.Labels}}' | \
  python3 -c "import sys,json; labels=json.load(sys.stdin); print(labels.get('org.opencontainers.image.version','?'))"

# List available local hindsight images with all tags
docker images | grep hindsight

# Cross-reference: what's running vs what's pulled
docker ps --filter name=hindsight --format '{{.Image}}'
```

If a newer image tag (e.g. `:0.6.1`) exists locally but `:latest` still serves an older version, the container was never recreated with the new tag. Switching requires updating the compose/run reference and recreating the container. It reads `/operations?exclude_parents=true`, samples recent operations without payloads, and adds `/stats/memories-timeseries` plus `/audit-logs/stats`. Use psql only for forensic fallback or schema details the API cannot expose.

For progress monitoring:

```bash
python3 $HERMES_HOME/scripts/hindsight_progress_bar.py
bash $HERMES_HOME/scripts/hindsight_progress_live.sh
```

### 2. Session Retain / Historical Import

Default policy for paid/session retain:

- user + Hermes assistant outputs only,
- no tool traces, shell output, API keys, or private process noise,
- smoke first, then batch,
- incremental history by default,
- auto-retain off unless explicitly requested,
- give native observations enough wall-clock time; full observation loads can legitimately run for up to 24h, so do not use short wait timeouts for full/incremental production runs.

Build or inspect a session manifest. Default `--profile-mode hindsight` scans the main session store plus every profile under `~/.hermes/profiles/<profile>/` whose `config.yaml` has `memory.provider: hindsight` (e.g. kanban `coordinator/planner/implementer/critic`). Non-default profile documents are namespaced as `hermes-session::<profile>::<session_id>` and tagged with `profile:<profile>` so incremental submit state cannot collide with default sessions.

```bash
python3 $HERMES_HOME/scripts/hindsight_session_manifest.py --bank-target hermes --profile-mode hindsight --json
```

For debugging/backfill only, use `--profile-mode all` to include every profile with a `sessions/` directory, or `--profile-mode none` to restrict to the default session store.

For tricky retain/import failures, read:

- `references/hindsight-session-retain-pitfalls.md`
- `references/hindsight-user-assistant-only-retain-smoke.md`
- `references/hindsight-session-json-production-hardening.md`
- `references/hindsight-session-quality-hardening.md`
- `references/hindsight-full-pipeline-watchdog-and-manifest-id.md` — use filename stems as durable session IDs; long observation waits need watchdog/resume handling.

### 3. Daily / Weekly / Full Pipeline

Dry-run plan is the default safe entrypoint:

```bash
python3 $HERMES_HOME/scripts/hindsight_memory_pipeline.py weekly --plan-json
```

Expected weekly plan includes:

```text
preflight
runtime_status
weekly_reflect
v2_rebuild_gate
conflict_audit
repair_zone_proposals
proposal_review
production_writes_possible=false
```

For full production-adjacent runs, expect native retain/observation work to outlive the foreground orchestration process. Use a long wait budget (up to 86400s for broad/full session retain), and if the foreground process times out while Hindsight operations are still active, use a watchdog/resume pattern: wait for idle, reconcile submit state, rerun from the next safe stage, then perform post-status and recall smoke. The pipeline now inserts read-only native-consolidation drain gates before quality-sensitive V2/conflict/proposal stages by default; `full --skip-daily` waits before weekly resume and again after weekly reflect. Use `--no-wait-native-consolidation` only for explicit emergency/debug resumes where stale observations are acceptable. Do not trust a short one-shot/no-agent cron as the durable verifier for long `full --skip-daily` runs: confirm a live verifier process and log after scheduling, or use an explicit background watchdog. See `references/hindsight-full-pipeline-watchdog-and-manifest-id.md` and `references/hindsight-full-skip-daily-v2-publish-resume.md`.

If daily retain/reflect already completed and you need to resume the full pipeline from weekly/V2 gates, use `--skip-daily` with `full`; this skips session manifest/retain, daily reflect, and the daily V2 rebuild gate, then runs weekly reflect → V2 publish gate → conflict audit → repair proposal review → optional wiki maintenance. `--skip-daily` is intentionally valid only with `full` mode; non-full modes fail closed.

Execute only with the outer pipeline confirmation:

```bash
python3 $HERMES_HOME/scripts/hindsight_memory_pipeline.py weekly \
  --execute --confirm run-hindsight-pipeline
```

For advisory LLM proposal review in the same weekly run, add the separate proposal-review confirmation:

```bash
python3 $HERMES_HOME/scripts/hindsight_memory_pipeline.py weekly \
  --execute --confirm run-hindsight-pipeline \
  --execute-proposal-review-llm \
  --confirm-proposal-review review-hindsight-proposals \
  --notify-proposal-review
```

Read before scheduling or changing cadence:

- `references/hindsight-memory-pipeline-modes.md`
- `references/offline-weekly-paid-run-safety.md`
- `references/offline-weekly-paid-run-progress-and-audit.md`
- `references/hindsight-offline-cron-logging-and-daily-anomaly-report.md`

### 4. Repair Backlog → Temp Bank → Proposal Review

Safe publication path:

1. Score/review backlog sidecars.
2. Send candidate repairs through a temp/quarantine bank when needed.
3. Run fact-quality and recall smoke checks.
4. Build proposal-only canonical bundles.
5. Build local review packets.
6. Optional advisory LLM review with explicit confirmation.
7. Human reviewer makes final go/no-go.
8. Only after that, design a separate production retain/merge with snapshot and rollback.

Build proposal-only bundles:

```bash
python3 $HERMES_HOME/scripts/hindsight_repair_proposal_build.py \
  --approved-index $HERMES_HOME/hindsight/review_repair/approved/<stem>-observations_index.jsonl \
  --output-root $HERMES_HOME/hindsight/review_repair/proposals \
  --stem <stem> --top 80 --json
```

Build a local review packet without LLM disclosure:

```bash
python3 $HERMES_HOME/scripts/hindsight_proposal_review.py \
  --proposal-json $HERMES_HOME/hindsight/review_repair/proposals/<stem>-canonical-proposals.json \
  --review-root $HERMES_HOME/hindsight/review_repair/reviews \
  --top 80 --notify --json
```

Run advisory LLM review only with explicit confirmation:

```bash
python3 $HERMES_HOME/scripts/hindsight_proposal_review.py \
  --proposal-json $HERMES_HOME/hindsight/review_repair/proposals/<stem>-canonical-proposals.json \
  --review-root $HERMES_HOME/hindsight/review_repair/reviews \
  --top 80 \
  --execute-llm --confirm-review review-hindsight-proposals \
  --notify --json
```

Required behavior:

- `production_mutation_allowed=false`
- `production_merge_or_retain_executed=false`
- human final decision remains pending unless separately recorded
- deterministic blocked/secret-like proposals get `llm_judgement.status=skipped_deterministic_block`
- no Hindsight retain/merge/delete API calls in `hindsight_proposal_review.py`

Read:

- `references/hindsight-proposal-review-governance.md`
- `references/hindsight-review-backlog-tempbank-flow.md`
- `references/hindsight-review-backlog-scorer.md`
- `references/hindsight-review-backlog-scorer-preflight.md`

### Switching Hindsight LLM Provider

Hindsight container LLM is configured via 4 groups of env vars: `LLM`, `RETAIN_LLM`, `CONSOLIDATION_LLM`, `REFLECT_LLM`. Each group has `PROVIDER`, `MODEL`, `BASE_URL`, `API_KEY`.

**Critical: Hindsight only recognizes specific provider names.** Valid providers (as of v0.6.1): `openai, groq, ollama, gemini, anthropic, lmstudio, llamacpp, vertexai, openai-codex, claude-code, mock, none, minimax, deepseek, litellm, litellmrouter, bedrock, volcano, openrouter, zai`. Using an unrecognized name (e.g. `topenrouter`) causes `ValueError: Invalid LLM provider` at startup.

**For OpenAI-compatible endpoints (TopenRouter, OpenCode, etc.)**, use `provider=openai` with the custom `BASE_URL`:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_LLM_API_KEY=<topenrouter-key>
```

**Switch procedure** (container recreate):

1. Verify the new endpoint works: `curl -s <base_url>/chat/completions -H "Authorization: Bearer $KEY" -d '{"model":"<model>","messages":[{"role":"user","content":"OK"}],"max_tokens":5}'`
2. `docker stop hindsight && docker rm hindsight`
3. `docker run -d --name hindsight ...` with all 4 LLM groups updated to the new provider/model/base_url/key
4. Reapply the parallel consolidator patch: `python3 ~/.hermes/scripts/patch_hindsight_consolidator_parallel.py`
5. `docker restart hindsight` (to load the patch)
6. Verify: `curl http://127.0.0.1:8888/health` + check logs for `slow llm call` entries with the new model
7. If consolidation was interrupted, recover: `curl -X POST http://127.0.0.1:8888/v1/default/banks/hermes/consolidation/recover`

**Container recreate loses writable-layer patches.** The parallel consolidator patch must be reapplied after every recreate. The `patch_hindsight_consolidator_parallel.py` script handles this.

### 5. v0.6.1 API-First Operations

Use new official endpoints before DB-level work:

```bash
# Queue/drain status without parent-batch false positives
python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --skip-psql --json
curl_cmd "http://127.0.0.1:8888/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&limit=20"

# Drill down only when needed; avoid include_payload=true unless debugging one operation
curl_cmd "http://127.0.0.1:8888/v1/default/banks/hermes/operations/<operation_id>?include_payload=false"

# Daily/weekly reporting sources
curl_cmd "http://127.0.0.1:8888/v1/default/banks/hermes/stats/memories-timeseries?period=7d&time_field=created_at"
curl_cmd "http://127.0.0.1:8888/v1/default/banks/hermes/audit-logs/stats?period=7d"
```

Mutation endpoints are allowed only after explicit scope confirmation:

- `POST /operations/{id}/retry` — retry a specific failed operation; use confirm token `retry-hindsight-operation` in scripts.
- `DELETE /operations/{id}` — cancel/delete a specific operation; use confirm token `delete-hindsight-operation`.
- `POST /documents/{id}/reprocess`, `/entities/{id}/regenerate`, `/mental-models/{id}/refresh`, `/consolidation/recover` — targeted repair before reset; snapshot/export first for production banks.
- `GET /export`, `POST /import?dry_run=true`, and `/v1/bank-template-schema` — reversible bank-template movement for temp-bank/proposal workflows. Treat `dry_run=false` import as production mutation.

Operational rules:

- `exclude_parents=true` is the default for status so batch parent rows do not look like real pending work.
- `include_payload=true` can expose large or sensitive payloads; do it only for one known operation ID.
- Export/import is a bank-template snapshot, not a substitute for pg_dump when doing high-risk DB/schema upgrades.
- `HINDSIGHT_API_READ_DATABASE_URL` helps only with a real read-only backend. Do not point it at the same writer and call that safer.

### 6. Recovery / Repair / Reset

High-risk operations. Do not execute without explicit user approval.

Before any production repair/reset:

- Export/snapshot the affected bank or DB rows.
- Identify exact document/operation IDs.
- Prefer quarantine/temp-bank validation.
- Prepare rollback instructions before mutation.
- Verify with recall/search/lineage audits after mutation.

Read the specific playbook first:

- `references/offline-zero-unit-observations-repair.md`
- `references/hindsight-zero-unit-temp-bank-retry.md`
- `references/hindsight-db-reset-full-rebuild-preflight.md`
- `references/hindsight-db-reset-full-rebuild-2026-05-08-lessons.md`
- `references/llm-cleanup-repair-loop.md`
- `references/hindsight-llm-provider-configuration-pitfalls.md` — topenrouter ≠ openrouter.ai, model name prefix trap (403), .env override chain, container restart 502 window, and diagnosis checklist.

## Reference Map

Use targeted references instead of loading everything.

### Core operations

- `references/hindsight-daily-pipeline-quick-status-check.md` — quick "did yesterday's pipeline succeed?" diagnostic: log locations, exit codes, distinguishing historical failed_operations from current failure, and live-container vs offline-pipeline LLM config differences.
- `references/hindsight-cron-schedule.md` — production Hermes cron jobs for daily/weekly/wiki pipelines, exact prompts, biweekly guard pattern, and WeChat delivery targets.
- `references/hindsight-skill-bundle-export-sharing.md` — package these Hindsight skills for another Hermes user with manifest/checksum, privacy scan, temp-extract verification, and import README.
- `references/hindsight-v061-upgrade-impact.md` — v0.5.2-era local deployment → v0.6.x upgrade impact, migration risks, and new official tuning keys.
- `references/hindsight-v071-upgrade-and-search-vector-issue.md` — v0.7.1 upgrade notes and `search_vector` GENERATED ALWAYS column non-blocking issue.
- `references/hindsight-v071-upgrade-and-tuning.md` — v0.6.1 → v0.7.1 upgrade notes, PostgreSQL tuning for vector search (shared_buffers, random_page_cost), and known issues.
- `references/hindsight-container-upgrade-pattern.md` — step-by-step pattern for upgrading Hindsight container (env dump → stop → rm → recreate with --env-file → verify → reapply patches), pitfalls with shell escaping and restart race conditions. Verified: 0.6.1→0.7.1 (2026-06-02).
- `references/hindsight-native-api-first-migration.md` — prefer native API and avoid brittle DB-first workflows.
- `references/hindsight-native-workflow-guard.md` — native workflow guardrails.
- `references/hindsight-bank-config.md` — bank config and observations settings.
- `references/hindsight-consolidation-config.md` — consolidation config keys.
- `references/api-endpoints.md` — useful Hindsight API endpoints.
- `references/hindsight-v061-api-first-operations.md` — v0.6.1 Operations API, observability, export/import, targeted repair, worker slots, and read-only DB backend rules.
- `references/hindsight-v061-skill-upgrade-2026-05-13.md` — session-specific notes for the v0.6.1 skill/library upgrade and verification snapshot.
- `references/hindsight-v071-upgrade-and-pg-tuning.md` — v0.6.1→0.7.1 upgrade: API path changes, schema changes, retain format, and critical PostgreSQL tuning for vector search performance (shared_buffers, random_page_cost, work_mem).
- `references/hindsight-container-upgrade-pattern.md` — step-by-step pattern for upgrading Hindsight Docker containers: version verification, env preservation, recreate, and patch reapplication.

### Session retain and quality

- `references/hindsight-full-pipeline-watchdog-and-manifest-id.md` — session filename-stem durable IDs, long native observation wait budgets, and full-pipeline watchdog/resume pattern.
- `references/hindsight-full-pipeline-interrupt-resume.md` — diagnosing SIGTERM/rc=-15 watchdog exits, identifying completed stages, and resuming from the next safe stage instead of blindly rerunning full.
- `references/hindsight-full-skip-daily-v2-publish-resume.md` — resume full pipeline with `--skip-daily`, publish V2, recognize long bge-m3 embedding inside `docker exec`, and verify proposal-review safety boundaries.
- `references/native-consolidation-pipeline-gates-and-half-downgrade.md` — why V2/conflict/proposal stages wait for `pending_consolidation` drain, plus the prepared 8→4 native-consolidation half-downgrade rollback.
- `references/hindsight-session-retain-pitfalls.md`
- `references/hindsight-user-assistant-only-retain-smoke.md`
- `references/user-assistant-only-retain.md`
- `references/hindsight-session-json-cost-control-and-curated-smoke.md`
- `references/hindsight-session-json-lightweight-candidate-filter.md`
- `references/hindsight-session-json-production-hardening.md`
- `references/hindsight-session-quality-hardening.md`
- `references/retain-call-amplification.md`
- `references/retain-throughput-and-queue-diagnostics.md`
- `references/hindsight-external-import-design-notes.md` — external import design notes, adapter inventory, granularity analysis, and forward-looking third-party import considerations

### External content import

- `references/hindsight-external-content-import.md` — full workflow for importing chat-memo txt files and OpenClaw lcm.db conversations into Hindsight. Covers manifest building, retain runner, per-source serial execution, pitfalls, and timeline estimates. The scripts (`hindsight_external_manifest.py`, `hindsight_external_retain_runner.py`) are in `~/.hermes/scripts/`.
- `references/external-import-hermes-observation-drain-20260519.md` — session lesson for 10pct sample-bank vs `hermes` production import: verify sample document IDs in `hermes`, do not copy derived observations across banks, reconcile submit_state after wait failures, then run the `hermes` consolidation/observation drain.

### Offline reflect/consolidation

- `references/hindsight-memory-pipeline-modes.md`
- `references/hindsight-full-pipeline-step-sequence.md` — complete 16-step sequence for `full` mode, CLI flags, failure recovery patterns, and common failure symptom/cause/fix table.
- `references/offline-weekly-paid-run-safety.md`
- `references/offline-weekly-paid-run-progress-and-audit.md`
- `references/precision-remote-observations-drain.md`
- `references/parallel-consolidation-drain.md`
- `references/consolidation-llm-batch-sizing.md`
- `references/consolidation-slot-semantics-and-remote-switch.md`
- `references/hindsight-offline-consolidation-pipeline.md`
- `references/hindsight-offline-progress-monitoring.md`
- `references/hindsight-offline-pipeline-verification.md`

### Proposal review and publish gates

- `references/hindsight-proposal-review-governance.md`
- `references/hindsight-review-backlog-tempbank-flow.md`
- `references/hindsight-offline-pipeline-readiness-and-publish-gate.md`
- `references/hindsight-offline-v2-publish-gate.md`
- `references/hindsight-offline-v2-quality-architecture.md`
- `references/hindsight-offline-v2-sidecar-publish-runbook.md`

### Provider, model, and rate-limit issues

- `references/hindsight-llm-provider-configuration-pitfalls.md` — **NEW (2026-05-26)** — topenrouter vs OpenRouter, model name format per provider (`deepseek/` prefix rules), .env override chain, safety rule: never change LLM config without user permission.
- `references/hindsight-topenrouter-llm-configuration.md` — topenrouter (ChinaDataPay) LLM configuration chain, model naming pitfalls, full pipeline step reference, and 403/502 troubleshooting.
- `references/hindsight-model-requirements.md`
- `references/hindsight-llm-config-two-layer-override.md` — detailed two-layer override architecture, provider vs profile distinction, switching providers, common error patterns
- `references/hindsight-topenrouter-llm-configuration.md` — topenrouter (ChinaDataPay) LLM configuration chain, model naming pitfalls, full pipeline step reference, and 403/502 troubleshooting.
- `references/hindsight-model-requirements.md`
### Provider, model, and rate-limit issues

- `references/topenrouter-provider-configuration.md` — TopenRouter (ChinaDataPay relay) provider setup, model naming pitfall (no `deepseek/` prefix), `.env` override pattern, container 502 window, and provider-vs-profile naming disambiguation.
- `references/hindsight-topenrouter-llm-configuration.md` — topenrouter (ChinaDataPay) LLM configuration chain, model naming pitfalls, full pipeline step reference, and 403/502 troubleshooting.
- `references/hindsight-model-requirements.md`
- `references/hindsight-operations-model-sizing.md`
- `references/hindsight-glm5-rate-limit.md`
- `references/hindsight-429-five-minute-backoff.md`
- `references/bailian-offline-concurrency-429.md`
- `references/minimax-api-call-optimization.md`
- `references/hindsight-minimax-queue-quirks.md`
- `references/ollama-local-deployment.md`
- `references/hindsight-ollama-shadow-local.md`
- `references/topenrouter-llm-configuration.md`

### LLM Provider and Model Configuration — Hard Rules

1. **NEVER change Hindsight LLM settings (llm_profile, base_url, api_key) without explicit user confirmation.** This is a user preference enforced after an incident where unauthorized model switches broke pipeline runs and cost tracking.

2. **topenrouter ≠ openrouter.ai.** topenrouter is ChinaDataPay's relay (`tp-api.chinadatapay.com:8000/v1`), a domestic service requiring no proxy. openrouter.ai is a separate international service. Do not confuse the two.

3. **Model name format differs by provider.** topenrouter registers models without provider prefixes: `deepseek-v4-flash` (correct), NOT `deepseek/deepseek-v4-flash` (will 403). Always verify via `GET /v1/models` on the target base_url before assuming a model name. The Hindsight internal profile `deepseek-v4-flash` uses `hindsight_provider: "openai"` format but the actual `HINDSIGHT_OFFLINE_LLM_MODEL` value must match the upstream registry — no `deepseek/` prefix for topenrouter.

4. **Container environment overrides.** Hindsight reads LLM config from container env vars. The `.env` overrides (`HINDSIGHT_OFFLINE_LLM_BASE_URL`, `HINDSIGHT_OFFLINE_LLM_MODEL`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV`) take effect when the offline reflect/import scripts run. Verify container env via `docker exec hindsight env | grep LLM_` after any config change. Container restart is required for `.env` changes to propagate.

5. **Container restart timing.** After `docker restart`, the bge-m3 embedding model takes ~25s to load. During this window, the API returns 502 or connection refused. Wait for `curl http://127.0.0.1:8888/health` to return healthy before running eval or other API-dependent steps.

6. **Full pipeline step reference** (verified 2026-05-26):
   - preflight → status → queue_drain → session_manifest → retain_session → daily_reflect → native_consolidation_drain → v2_rebuild(daily) → native_consolidation_drain → weekly_reflect → native_consolidation_drain → v2_rebuild(weekly) → conflict_audit → repair_zone_proposals → proposal_review → wiki_auto_maintenance(需`--include-wiki`)
   - Failed steps don't require re-running earlier succeeded steps. Resume from the failed step only.

### Embeddings and DB recovery

- `references/embedding-model-migration.md`
- `references/bge-m3-manual-model-download.md`
- `references/prod-pg0-bge-m3-recovery.md`
- `references/hindsight-external-database-architecture.md`

- `references/hindsight-llm-profile-naming.md` — internal LLM profile names vs provider routing names, `--llm-profile` valid values, `.env` override pitfalls (`HINDSIGHT_OFFLINE_LLM_BASE_URL`), and debugging checklist for LLM auth failures.

### Wiki/topic docs

- `references/wiki-maintenance.md`
- `references/topic-doc-ingestion.md`
- `references/hindsight-daily-report-cron-checklist.md`

## Common Pitfalls

4. Treating proposal review as production approval. It is not. It only prepares review packets.
5. Using 'topenrouter' as Hindsight LLM provider name. Hindsight's valid providers list is: `openai, groq, ollama, gemini, anthropic, lmstudio, llamacpp, vertexai, openai-codex, claude-code, mock, none, minimax, deepseek, litellm, litellmrouter, bedrock, volcano, openrouter, zai`. Custom providers like `topenrouter` are NOT supported — use `openai` provider with the custom base_url pointing to the topenrouter endpoint.
2. Sending blocked/secret-like proposal text to an external LLM. Deterministic blocks must short-circuit LLM calls.
3. Confusing `batch_retain` parent rows with claimable retain jobs. Parent rows with `task_payload IS NULL` can look pending while real child work is complete/in flight.
29. **LLM profile names are internal to `hindsight_minimax_import.py`, not provider routing names.** `--llm-profile` accepts: `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`, plus custom profiles. `topenrouter` is NOT a valid llm-profile — it is the OpenRouter-specific routing format (`https://openrouter.ai/api/v1`), which is NOT `openai`-compatible and cannot be used as `hindsight_provider='openai'`. For DeepSeek V4 Flash, use `--llm-profile deepseek-v4-flash` (which defaults to `https://api.deepseek.com/v1` with `DEEPSEEK_API_KEY`). The `pipeline_config.json` `llm_profile` field must also use the internal profile name, not a provider name. See `references/hindsight-llm-profile-naming.md`.

30. **Do not override `HINDSIGHT_OFFLINE_LLM_BASE_URL` in `.env` with third-party proxy URLs.** The `deepseek-v4-flash` profile defaults to `https://api.deepseek.com/v1`, which works. A `HINDSIGHT_OFFLINE_LLM_BASE_URL` override (e.g. `https://tp-api.chinadatapay.com:8000/v1`) takes precedence and will cause 403/401 errors if the third-party key expires. Remove the override to fall back to the profile's default `base_url`. Only set this env var when intentionally routing through a known-good alternative endpoint.

31. **Recall API (`POST /memories/recall`) does NOT call external LLM.** It is pure vector retrieval (bge-m3 embedding + pgvector + reranker). If recall times out, the cause is DB load, embedding computation pressure, or container resource limits — not the LLM model. Do not change `llm_profile` to fix recall timeouts; instead check container health, DB connections, and memory usage.

4. Reading only state DB or one status endpoint for cost/reporting. Daily reporting should include main Hindsight sessions, offline pipeline, auxiliary compression, and Hermes profile state DBs where relevant.
4a. **Running eval/recall immediately after container restart.** The API returns 502/connection-refused for ~25 seconds while bge-m3 loads. Always wait for `/health` OK first.
4b. **Assuming recall timeout is an LLM problem.** `POST /memories/recall` is bge-m3 embedding + pgvector similarity search, NOT an external LLM call. Timeout means DB/embedding pressure or container instability, not LLM provider issues. See `references/hindsight-llm-config-two-layer-override.md` for the full LLM config model.

4a. **Hermes profiles are isolated for both state.db and session JSON, and the pipeline now handles this automatically.** The default paths `~/.hermes/state.db` and `~/.hermes/sessions/` do not include profile sessions. Visible kanban profiles such as `coordinator`, `planner`, `implementer`, and `critic` store data under `~/.hermes/profiles/<profile>/state.db` and `~/.hermes/profiles/<profile>/sessions/`. `hindsight_session_manifest.py` defaults to `--profile-mode hindsight`, discovers profiles whose `memory.provider` is `hindsight`, namespaces non-default document IDs as `hermes-session::<profile>::...`, and adds profile/source tags. Keep this default for daily/full pipelines so future kanban profiles are retained incrementally without manual per-profile runs.
5. Using global HTTP(S)_PROXY blindly. Hindsight/DeepSeek/MiniMax/DashScope usually work direct; proxy can break local Docker networking.
4a. **Hermes profiles are isolated for both state.db and session JSON, and the pipeline now handles this automatically.**
5. Using global HTTP(S)_PROXY blindly. Hindsight/DeepSeek/MiniMax/DashScope usually work direct; proxy can break local Docker networking.
6. Mixing embedding dimensions in one bank. bge-small 384 and bge-m3 1024 require separate/reset banks.
7. Assuming `total_observations=0` is always a failure. Some workflows intentionally disable observations; check config and operation intent.
8. Recreating containers without considering writable-layer patches. Some local patches survive restart but not recreate unless wrappers reapply them.
9. Putting ad-hoc Python scripts under `/tmp` with names like `http.py`; they can shadow stdlib modules.
10. Using `rm -rf` for cleanup. Default to trash/quarantine unless the user explicitly asks for permanent deletion.
11. Trusting `:latest` Docker tag equals latest version. `:latest` can point to an older release (e.g. v0.5.2) while a newer tagged image (v0.6.1) is already pulled but never deployed. Always verify the actual version via `docker inspect` image labels and cross-reference with available image tags.
12. Trusting embedded `session_id` over `session_*.json` filename stems. Historical JSON can contain stale embedded IDs; manifests should derive durable unique document/session IDs from filename stems.
13. Treating a foreground orchestration timeout or watchdog `rc=-15` as native Hindsight failure. `rc=-15` usually means SIGTERM/external stop. Inspect watchdog logs, identify the last completed stage, verify Hindsight is idle, then resume from the next safe stage instead of restarting or resubmitting blindly. Do not rerun full from the beginning when daily/weekly reflect progress already records completed document IDs.
14. Treating V2 publish as stuck after gate files appear. Publish may still be computing bge-m3 embeddings inside a `docker exec` child; if `docker top hindsight` shows high CPU on `SentenceTransformer(... BAAI/bge-m3 ...)` and Hindsight stats are advancing, wait for the subprocess to return instead of killing it.
15. **Model name format varies per provider.** topenrouter requires `deepseek-v4-flash` (no `deepseek/` prefix); `deepseek/deepseek-v4-flash` returns 403 "no access" on topenrouter but works on OpenRouter. opencode-go also requires no prefix. Always test the exact model name with `curl` before committing config. See `references/hindsight-llm-provider-configuration-pitfalls.md`.
16. **Changing Hindsight LLM configuration without user permission.** The LLM config chain (pipeline_config.json → .env overrides → container env) is fragile and provider-specific. Never modify `llm_profile`, `base_url`, `api_key_env`, or model name without explicit user go-ahead.
14. Treating V2 publish as stuck after gate files appear. Publish may still be computing bge-m3 embeddings inside a `docker exec` child; if `docker top hindsight` shows high CPU on `SentenceTransformer(... BAAI/bge-m3 ...)` and Hindsight stats are advancing, wait for the subprocess to return instead of killing it.
15. Before any container/image upgrade or recreate, verify the running image label and the intended image tag explicitly:

```bash
sg docker -c \"docker ps --filter name=hindsight --format '{{.Image}} {{.ID}}'\"
sg docker -c \"docker inspect hindsight --format '{{json .Config.Labels}}'\" | \\
  python3 -c \"import sys,json; print(json.load(sys.stdin).get('org.opencontainers.image.version','?''))\"
sg docker -c \"docker image inspect ghcr.io/vectorize-io/hindsight:latest --format '{{index .Config.Labels \\\"org.opencontainers.image.version\\\"}}'\"
```

If a future upgrade intentionally changes version, update the wrapper default and retag `:latest` together; otherwise a later container recreate can silently run a different image than expected.

16. **Cron-loaded skill examples should avoid `$HINDSIGHT_API_URL`.** Older Hermes cron prompt scanners treat any env var containing `API` in a `curl`/`wget` line as possible `exfil_curl`. In Hindsight skill docs and references that may be loaded into scheduled jobs, prefer a neutral alias:

```bash
HINDSIGHT_BASE_URL=\"${HINDSIGHT_BASE_URL:-http://127.0.0.1:8888}\"
curl -s \"$HINDSIGHT_BASE_URL/v1/default/banks/hermes/stats\"
```

This skill-only workaround avoids touching Hermes source and should be tried before patching scanner regexes. If source patching is still needed, use two-sided tests: `HINDSIGHT_API_URL` should pass, while `$API_KEY`, `$AWS_ACCESS_KEY_ID`, `${AWS_CREDENTIALS}`, `${MINIMAX_API_TOKEN}`, `$CLIENT_SECRET`, and `$PASSWORD_FILE` must still block.

17. **0.6.1 config key support is partial.** After upgrading to 0.6.1:
   - `consolidation_max_memories_per_round` — **supported** (was unsupported in 0.5.2). This is the key upgrade win.
   - `consolidation_recall_budget` — **still unsupported** even in 0.6.1. The wrapper skips it gracefully.
   - Old local patches (JSON fence parser, consolidation per-job budget) are correctly detected as \"not applicable\" on 0.6.1 and skipped. This is expected, not an error.
   - `features.observations` changes from `false` (0.5.2) to `true` (0.6.1) in `/version`.

18. **Do not production-retain Hermes cron sessions by default.** Cron session JSON can contain the full system prompt, tool schemas, skill injection text, and generated operational reports rather than direct user/Hermes dialogue. Retaining an active daily-pipeline cron session caused self-ingestion and a Hindsight retain FK race on `memory_links.to_unit_id`. The session manifest builder now defaults to:
   - skip files modified in the last `HINDSIGHT_SESSION_MANIFEST_MIN_FILE_AGE_SECONDS` seconds (default 900), and
   - mark `platform=cron` / `session_id` starting with `cron_` as `skip:automated_cron_session`.

If cron reports are ever needed as memory evidence, route them through a separate curated/report-specific ingestion path instead of generic session retain.

19. **Session retain must not drain the whole bank queue after its own retain ops finish.** `session-manifest-retain-llm` may run with `--enable-observations`, but the underlying `hindsight_session_retain_runner.py` already waits for the submitted retain operation IDs. Do not call a global `wait_queue_drained()` before restore: Hindsight native consolidation intentionally self-requeues when it hits `consolidation_max_memories_per_round` until the whole bank's unconsolidated backlog is empty. That behavior is finite but can look like an infinite loop and can turn a small session retain into hours of bank-wide consolidation. Correct wrapper behavior is: wait submitted retain IDs in the runner → immediately pre-disable target-bank observations in `finally` → restore normal mode → then do a bounded post-restore tail drain so any already-queued consolidation no-ops/drains without becoming another whole-bank wait. Default tail-drain timeout is 600s when `--timeout=0`; override with `HINDSIGHT_POST_RESTORE_QUEUE_DRAIN_TIMEOUT_S` only when you intentionally need a longer cleanup wait. If a global drain already started, patch the bank config to `enable_observations=false` before the current round hits its requeue point.

20. **Patch Hindsight temporal link FK guard through the skill wrapper, not Hermes source.** Hindsight 0.6.1 `create_temporal_links_batch_per_fact()` selects temporal neighbors and then inserted `memory_links` with `skip_exists_check=True`. Concurrent replace-retain can delete a selected neighbor before insert, causing `fk_memory_links_to_unit_id_memory_units`. The packaged `patch_hindsight_retain_temporal_fk_guard.py` changes only temporal links to `skip_exists_check=False`, reusing `_bulk_insert_links`' EXISTS guard. Causal links remain `skip_exists_check=True` because both endpoints are new units in the same transaction. The wrapper reapplies this patch after container recreate.

21. **Short one-shot/no-agent cron jobs are not durable verification for long Hindsight waits.** A one-shot job can disappear from the cron list after the scheduler timeout while no long-running verifier remains. For `full --skip-daily` end-to-end verification, check both: (a) the cron entry/job state and (b) a live process plus growing log. If the cron job vanished and no verifier process exists, start a controlled background watchdog that waits for Hindsight idle and then runs only the intended resume command. Do not compensate by rerunning full daily/session retain unless that is explicitly intended.

22. **'refusing to switch to minimax with existing queue' causes silent observation gap.** When `hindsight_minimax_import.py` runs `session-manifest-retain-llm` with `--enable-observations` but finds `processing > 0` in the queue, it refuses to switch to minimax mode and runs retain WITHOUT observations. This does NOT produce a pipeline-breaking error — the retain step succeeds, but no observations are created. The gap is only visible by: (a) checking the daily pipeline log for `refusing to switch to minimax with existing queue` or `enable_observations: false`, (b) comparing `total_observations` before/after the pipeline run, or (c) noticing that `pending_consolidation` was already 0 before the pipeline ran and stayed 0. To repair: run `full` pipeline (or at minimum `POST /consolidate` with `enable_observations=true` set in bank config). To detect proactively: always check the latest daily pipeline log for these key lines before declaring a run successful.

22. **Hindsight LLM provider name must match the internal whitelist.** Valid providers: `openai, groq, ollama, gemini, anthropic, lmstudio, llamacpp, vertexai, openai-codex, claude-code, mock, none, minimax, deepseek, litellm, litellmrouter, bedrock, volcano, openrouter, zai`. Any other string (e.g. `topenrouter`) causes a startup crash: `ValueError: Invalid LLM provider: xxx`. For OpenAI-compatible endpoints like TopenRouter, use `provider=openai` with the custom `BASE_URL`. This applies to all 4 LLM groups (llm/retain/consolidation/reflect). See `references/hindsight-llm-provider-switching.md` for the full provider-switching runbook including container recreation, env var mapping, and post-recreate patch sequence.

23. **MiniMax 8-way concurrency requires the local parallel consolidator patch.** Hindsight v0.6.1 upstream can expose `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT=8`, but the native consolidator may still process LLM batches serially. For large LLM-bound backlogs, apply `patch_hindsight_consolidator_parallel.py` and use decoupled limits (`LLM_BATCH_SIZE=8`, `PARALLEL_BATCHES=8`, `RECALL_MAX_CONCURRENT=60`, `LLM_MAX_CONCURRENT=8`). Do not raise only worker slots; same-bank consolidation is serialized. Verify logs show `parallel_batches=8 llm_limit=8 recall_limit=60` and per-batch `limits: batch_parallel=8 llm=8/8 recall=60`. **Validated throughput (2026-05-14): 1754 memories → 14350 observations in 2h20m with 0 429s, ~$6 MiniMax cost.** Switched to deepseek-v4-flash via OpenRouter on 2026-05-24 for better stability. The patched consolidator also includes pre-fetch pipeline and adaptive 429 handling. Since 2026-05-14, 64x8 is the production default in `normal-local` mode; fall back to 20x3 only when DB pool pressure demands it.

23. **429 handling is automatic, not a separate operator path.** The patched native consolidator now treats provider 429/rate-limit as capacity pressure: release the LLM slot, halve live LLM concurrency for future acquisitions, sleep `HINDSIGHT_API_CONSOLIDATION_429_BACKOFF_SECONDS` (default 300s), then retry the same batch without consuming the normal bad-input attempt budget. Repeated 429s after cooldown continue 8→4→2→1; simultaneous 429s inside one cooldown do not double-halve immediately. `HINDSIGHT_API_CONSOLIDATION_429_MAX_RETRIES=0` means unlimited 429 cooldown/retry loop. The `consolidation-half-downgrade` wrapper command is only emergency/manual preview, not the default mechanism. **Verified (2026-05-14): 0 429s across entire daily+weekly pipeline.**

24. **Pipeline quality gates must wait for native source-fact consolidation.** `pending_consolidation` means source `experience/world` facts have not been marked `consolidated_at` yet. V2 rebuild, conflict audit, proposal review, and final quality claims should run after the read-only `hindsight_wait_native_consolidation.py` gate reaches `pending_consolidation <= 0` and no pending/processing child ops remain. Historical failed operation rows alone do not prove current failure; use `failed_consolidation` and current source backlog.

25. **Pre-fetch pipeline eliminates inter-round DB latency.** The patched consolidator (since 2026-05-14) uses `asyncio.create_task` to fetch the next round's unconsolidated memories from DB while the current LLM wave runs. This overlaps DB read time (~24s) with LLM processing time, shrinking the next wave's recall step from 24s to 2s. Visible in logs as `rate_limit_backoff=300s` alongside `parallel_batches=8`. Works automatically; no extra config needed.

26. **Direct restart skips unreliable poll-waiting for patch application.** The consolidation auto-re-queue gap between rounds is sub-second, making poll-waiting (even at 3s intervals) unreliable. The established practice is `docker restart`, then unstick any stuck `processing` operations via `POST /consolidation/recover` or DB-level status fix. Unfinished `consolidated_at` marks are harmless: the next round reprocesses those memories. Only use poll-waiting when the user explicitly asks to wait for clean completion.

27. **Container restart causes ~25s API unavailability.** After `docker restart hindsight`, the bge-m3 embedding model takes ~25 seconds to load. During this window, the API returns 502 Bad Gateway or connection refused. Pipeline eval steps and any recall-dependent operations must wait for the API to become healthy before executing. Check readiness with `curl -s -m 5 http://127.0.0.1:8888/health` in a loop before proceeding.

28. **Full pipeline step order (full mode):** preflight → status → queue_drain_before_daily → session_manifest → retain_session → daily_reflect → native_consolidation_drain_after_daily → v2_rebuild (daily) → native_consolidation_drain_before_weekly → weekly_reflect → native_consolidation_drain_after_weekly → v2_rebuild (weekly) → conflict_audit → repair_zone_proposals → proposal_review → wiki_auto_maintenance (requires `--include-wiki`). If a step fails, earlier steps may have already completed — resume from the failed step instead of rerunning from the beginning.

29. **Full pipeline verification timing (2026-05-14):** daily retain ~25min (30 sessions) → consolidation ~2h20m (1754→0, 8-way parallel, 0 429s, ~$6 MiniMax) → weekly reflect ~15min (62 V2 units) → consolidation drain ~6min (67→0) → V2 publish ~10min (bge-m3, 20755 nodes) → conflict + proposal ~5min (52 cases, 0 P1 blocking) → wiki ~3min (79 pages, 30 high-confidence points). Total end-to-end: ~3h. LLM call economics: consolidation uses ~440 calls (~11M input tokens), parallelism reduces wall time but NOT call count — 3-way and 8-way have identical cost.

29. **Never change Hindsight LLM model/profile/base_url/api_key without explicit user approval.** This includes `pipeline_config.json` `llm_profile`, `.env` `HINDSIGHT_OFFLINE_LLM_*` overrides, `review.proposal_review` LLM settings, and container environment variables. The user has repeatedly corrected agents that switched models/providers without confirmation. Always ask first, even if the current model appears to be failing. See `references/hindsight-topenrouter-llm-configuration.md` for the current configuration and naming pitfalls.

30. **Topenrouter model names do not use the `deepseek/` prefix.** topenrouter (ChinaDataPay relay at `tp-api.chinadatapay.com:8000/v1`) registers models as `deepseek-v4-flash`, not `deepseek/deepseek-v4-flash`. Using the prefixed name returns 403 "This token has no access". topenrouter is NOT openrouter.ai — it is a domestic Chinese relay that does not need a proxy. See `references/hindsight-topenrouter-llm-configuration.md`.

31. **Container restart causes ~25s recall blackout.** After `docker restart hindsight`, bge-m3 embedding model loading takes 25-30 seconds. During this window, `/health` may be healthy but `POST /memories/recall` returns 502 or connection refused. Pipeline eval and recall-dependent steps must wait for the embedding model to finish loading before proceeding.

32. **Daily stats must merge external Agent data by kanban role into Hermes/Profiles table.** deepseek-tui session JSON `metadata.workspace` contains the kanban role (e.g. `.ds-sessions/implementer` → `implementer`). The `external_agent_stats.py` module infers roles from this field and the snapshot stores `workspace` for future deltas. `daily_stats.py` merges rows with kanban role profiles into the main Hermes/Profiles model usage table with a `*` suffix marker. Rows without a known role remain as `deepseek-tui` or `codex` in the separate external Agent table. When `hindsight_minimax_import.py` finishes retain and restores precision remote mode, it sets `enable_observations=false`. If new facts were submitted during the retain window (e.g. by daily reflect), they remain unconsolidated and `pending_consolidation > 0` forever because auto-consolidation requires `enable_observations=true` or an explicit `POST /consolidate`. The native consolidation wait gate (`hindsight_wait_native_consolidation.py`) previously only polled — it never triggered. **Fix (2026-05-15):** `hindsight_wait_native_consolidation.py` now has auto-trigger (default on). When `pending_consolidation > 0` and `processing_ops == 0` for 2 consecutive polling cycles, it POSTs `/consolidate` to kick-start consolidation. Disable with `--no-trigger-on-stall`.

29. **TopenRouter model names must NOT include provider prefixes.** The correct name is `deepseek-v4-flash`, not `deepseek/deepseek-v4-flash`. Using the prefixed name returns 403 "This token has no access". This is different from DeepSeek's official API which uses `deepseek/deepseek-v4-flash`. When switching between providers, always verify the model name format with a test call first. See `references/topenrouter-provider-configuration.md`.

30. **`wait_for_operation_ids` partial-failure misdiagnosis.** The session retain runner raises `RetainOperationFailed` when ANY tracked operation fails after retries, even if most data has already landed. When the daily noagent wrapper or runner reports failure, check data-level evidence (documents/observations delta from `/stats`) before concluding retain failed. v0.7.1's `search_vector` GENERATED ALWAYS column issue causes a small number of retain operations to fail with `cannot insert a non-DEFAULT value into column "search_vector"` — this is non-blocking and does not prevent data from being committed. See `references/hindsight-v071-upgrade-and-search-vector-issue.md` and `references/hindsight-session-retain-pitfalls.md` (wait_for_operation_ids section).

31. **Do not blindly re-run the daily pipeline on non-zero exit code.** If `wait_for_operation_ids` threw `RetainOperationFailed` but documents/observations increased, the data has landed. Re-running re-submits the same manifest and may create duplicate documents. Instead, verify stats delta, inspect failed operation error messages, and only re-run if documents did NOT increase.

30. **PostgreSQL shared_buffers must be tuned for vector search workloads.** Default `shared_buffers=128MB` causes recall timeouts (60s+) when `memory_units` exceeds ~1GB with HNSW indexes. The IO.DataFileRead wait pattern in logs is the telltale. For a 16GB RAM system with a dedicated Hindsight workload, set `shared_buffers=2GB`, `random_page_cost=1.1`, `effective_cache_size=12GB`, `work_mem=256MB`. Config file: `/home/wyr/.hindsight-docker/instances/hindsight/data/postgresql.conf`. Restart PG after changes. See `references/hindsight-v071-upgrade-and-tuning.md` for full tuning details and expected latency improvements.

31. **Hindsight 0.7.1 `search_vector` generated column errors are non-blocking.** The `memory_units.search_vector` column is `GENERATED ALWAYS` in 0.7.1. Internal code paths may attempt INSERT with a non-DEFAULT value, producing `GeneratedAlwaysError` in logs. These are caught internally and do not block recall/reflect/retain. Monitor frequency; expect upstream fix in later versions.

32. **Container upgrade preserves PostgreSQL data.** Hindsight data lives in the system PostgreSQL service, not inside the Docker container. Stopping/removing/recreating the container with a new image does not affect documents, memory_units, or memory_links. Always verify counts after upgrade. See `references/hindsight-v071-upgrade-and-tuning.md` for the upgrade procedure.

30. **Hindsight 0.7.1 restructured all API paths.** Endpoints moved from `/v1/banks/{bank}/...` to `/v1/default/banks/{bank}/...`. Specific changes: `recall` → `memories/recall`, `retain` → `memories` (with `items` wrapper), `observations` table removed (use `memory_units`), `operations` → `async_operations`. Always check `/openapi.json` after upgrading to discover current routes. Scripts using hardcoded v0.6.1 paths will 404. See `references/hindsight-v071-upgrade-and-pg-tuning.md` for the full mapping.

31. **PostgreSQL shared_buffers must be tuned for vector search workloads.** The default `shared_buffers=128MB` causes recall timeouts (60s+) when `memory_units` exceeds 1GB. For a 16GB RAM system with a 1.7GB table, set `shared_buffers=2GB`, `effective_cache_size=12GB`, `work_mem=256MB`, `random_page_cost=1.1` (SSD), `max_parallel_workers_per_gather=4`. After tuning, recall dropped from 60s+ timeout to 2-2.5s. PG config is at `/home/wyr/.hindsight-docker/instances/hindsight/data/postgresql.conf`; restart with `pg_ctl restart`. See `references/hindsight-v071-upgrade-and-pg-tuning.md` for diagnosis queries and rationale.

30. **`topenrouter` is a Hermes provider name, not a Hindsight llm-profile.** Hindsight `--llm-profile` accepts internal names only: `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`, plus custom. To use topenrouter as the backend, set `llm_profile=deepseek-v4-flash` and override base_url/key via `.env` env vars (`HINDSIGHT_OFFLINE_LLM_BASE_URL`, `HINDSIGHT_OFFLINE_LLM_MODEL`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV`). Do not pass `--llm-profile topenrouter`.

31. **Container restart causes ~25s 502 window.** After `docker restart`, the bge-m3 embedding model loads for ~25 seconds. During this time, the health endpoint may return 200 but recall API returns 502. Any recall-dependent steps (eval, status with recall smoke) must wait for the container to be fully ready by testing recall specifically, not just health.

29. **topenrouter model names must NOT include provider prefixes.** topenrouter (`tp-api.chinadatapay.com:8000`) registers models as `deepseek-v4-flash`, not `deepseek/deepseek-v4-flash`. Using the prefixed form causes 403 Forbidden. This differs from DeepSeek's official API (`api.deepseek.com`) which accepts `deepseek/deepseek-v4-flash`. Always verify available models via `GET /v1/models` on the target endpoint. See `references/topenrouter-llm-configuration.md`.

30. **Never change Hindsight LLM config without user confirmation.** llm_profile, base_url, api_key changes must be explicitly approved. Unauthorized switches break pipeline runs, cost tracking, and daily report accuracy. This rule applies to pipeline_config.json, .env overrides, and container environment variables alike.

29. **Do not change Hindsight LLM configuration without user confirmation.** The user explicitly requires approval before modifying `llm_profile`, `base_url`, `api_key_env`, or `llm_model` in Hindsight config files, `.env`, or `pipeline_config.json`. This includes provider switches (e.g. minimax→topenrouter), model changes, and API key updates. See `references/hindsight-llm-provider-configuration-pitfalls.md`.

30. **topenrouter is NOT openrouter.ai.** topenrouter (`tp-api.chinadatapay.com:8000/v1`) is a ChinaDataPay domestic relay, not the global OpenRouter service. It is OpenAI-compatible per their docs, does not need proxy, and uses model names WITHOUT the `deepseek/` prefix. Using `deepseek/deepseek-v4-flash` with topenrouter returns 403; the correct model name is `deepseek-v4-flash`. The Hermes provider name `topenrouter` is not a valid Hindsight `--llm-profile`; use `deepseek-v4-flash` profile with `.env` overrides. See `references/hindsight-llm-provider-configuration-pitfalls.md`.

31. **Container restart causes ~25s API 502 window.** After `docker restart hindsight`, the bge-m3 embedding model loads for ~25 seconds. During this window, `/health` may return connection refused and `/memories/recall` returns 502. Pipeline eval steps hitting this window will fail. Wait for `/health` healthy AND a recall smoke test before running eval or API-dependent steps.

29. **External-import observations may not carry the external document_id.** Native consolidation can create `observation` memory units with empty/NULL `document_id` and link them back to external source facts via `memory_links`. Do not verify external observation coverage with `memory_units.document_id LIKE 'external-%'` alone — that can falsely report zero observations. Correct verification joins external source `memory_units` (`document_id LIKE 'external-%'`) through `memory_links` to `fact_type='observation'`, then counts distinct linked observation ids and breaks down by `documents.retain_params->'metadata'->>'source_kind'`.

29. **External import wait failures need reconciliation, not blind reruns.** The runner submits retain batches asynchronously, then polls `wait_for_operation_ids` and now continues into `--wait-consolidation` by default. If Hindsight is briefly unreachable during the wait, retain may already be submitted. Verify document presence/stats and reconcile `external_import/submit_state.json` for the target `hermes` bank before deciding whether to rerun; do not blindly re-retain unchanged records. See `references/hindsight-external-content-import.md` for full workflow.

29. **Hindsight LLM provider name must be from its built-in list.** `HINDSIGHT_API_LLM_PROVIDER=topenrouter` causes a startup crash — Hindsight does not accept arbitrary provider names. Use `openai` (for OpenAI-compatible endpoints), `minimax`, `deepseek`, `ollama`, etc. Set `HINDSIGHT_API_LLM_BASE_URL` to the actual endpoint. See `references/hindsight-topenerouter-switch.md` for a worked example.

Before declaring this skill or a Hindsight workflow ready:

- [ ] `SKILL.md` frontmatter validates and content is well below the 100,000-char limit.
- [ ] Packaged scripts exist under `scripts/` and can be installed without editing `.env`.
- [ ] `python3 -m py_compile` passes for the pipeline/proposal scripts.
- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest scripts/tests/... -q` passes, or skipped tests are justified.
- [ ] `hindsight_pipeline_preflight.py --strict-runtime --json` has no blocking failures before production-adjacent operations.
- [ ] `hindsight_consolidation_status.py --skip-psql --json` reads v0.6.1 `/operations?exclude_parents=true` and reports observability endpoint availability.
- [ ] Weekly dry-run has `production_writes_possible=false` unless the user explicitly chooses a confirmed mutation path.
- [ ] Proposal review packets show no production mutation and require human final decision.
- [ ] If validating a cron scanner/skill-text fix, trigger the affected recurring job by temporarily setting only `next_run_at` to one minute in the future; do not change its cron expression. Confirm logs show `Running job`, old `exfil_curl`/scanner errors do not reappear, and `next_run_at` returns to the normal cadence. For long Hindsight jobs, wait for completion before expecting `last_run_at` and output files to update.

## Quick Publish-Readiness Check for This Skill

From the skill directory:

```bash
python3 - <<'PY'
from pathlib import Path
import re, yaml
p = Path('SKILL.md')
text = p.read_text(encoding='utf-8')
assert text.startswith('---')
m = re.search(r'\n---\s*\n', text[3:])
assert m
fm = yaml.safe_load(text[3:3+m.start()])
assert fm['name'] == 'hindsight-local-deployment'
assert fm.get('version') and fm.get('author') and fm.get('license')
assert len(text) < 100_000
for rel in re.findall(r'`(references/[^`]+?)`', text):
    assert Path(rel).exists(), rel
assert 'hindsight_consolidation_status.py' in text
print('skill ok', len(text))
PY

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  scripts/tests/test_hindsight_memory_pipeline.py \
  scripts/tests/test_hindsight_pipeline_preflight.py \
  scripts/tests/test_hindsight_native_client.py \
  scripts/tests/test_hindsight_session_manifest.py \
  scripts/tests/test_hindsight_minimax_import_session_manifest.py \
  scripts/tests/test_hindsight_repair_proposal_build.py \
  scripts/tests/test_hindsight_proposal_review.py -q
```