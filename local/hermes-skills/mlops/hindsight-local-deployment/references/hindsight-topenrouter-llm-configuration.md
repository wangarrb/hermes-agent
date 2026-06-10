# Hindsight LLM Configuration via TopenRouter

## Overview

Hindsight's LLM is configured to use `deepseek-v4-flash` via topenrouter (ChinaDataPay relay), not through DeepSeek's official API or OpenRouter.ai.

## Configuration Chain

1. **`pipeline_config.json`** sets `llm_profile = "deepseek-v4-flash"` (Hindsight internal profile name)
2. **`.env` overrides** redirect the profile to topenrouter:
   - `HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1`
   - `HINDSIGHT_OFFLINE_LLM_MODEL=deepseek-v4-flash`
   - `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY`
   - `HINDSIGHT_OFFLINE_HINDSIGHT_PROVIDER=openai`
3. **Container env** mirrors these via Hindsight's env→config mapping:
   - `HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1`
   - `HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash`
   - `HINDSIGHT_API_LLM_PROVIDER=openai`
   - Same for `RETAIN`, `CONSOLIDATION`, `REFLECT` sub-profiles
4. **`review.proposal_review`** in `pipeline_config.json`:
   - `llm_api_key_env=TOPENROUTER_API_KEY`
   - `llm_base_url=https://tp-api.chinadatapay.com:8000/v1`
   - `llm_model=deepseek-v4-flash`

## Key Naming Pitfalls

### 1. topenrouter ≠ openrouter.ai

- `topenrouter` is ChinaDataPay's relay service at `tp-api.chinadatapay.com:8000/v1`
- It is NOT `openrouter.ai` (which has its own API format and requires a proxy)
- topenrouter does NOT need a proxy — it's a domestic Chinese service
- Do NOT use `openrouter.ai/api/v1` as the base_url

### 2. Model name must NOT have `deepseek/` prefix

- ✅ Correct: `deepseek-v4-flash`
- ❌ Wrong: `deepseek/deepseek-v4-flash` → returns 403 "This token has no access"
- The `deepseek/` prefix is used by OpenRouter.ai, not by topenrouter
- Available models can be queried: `GET https://tp-api.chinadatapay.com:8000/v1/models`

### 3. Hindsight `--llm-profile` uses internal names

- Valid internal profiles: `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`
- `topenrouter` is NOT a valid llm-profile — it's a provider routing name in Hermes config
- To use topenrouter, set `llm_profile=deepseek-v4-flash` and override base_url/key via `.env`

### 4. Provider name must be `openai`

- In Hindsight container: `HINDSIGHT_API_LLM_PROVIDER=openai`
- topenrouter's API is OpenAI-compatible, so the provider is `openai`
- Do NOT set provider to `topenrouter`

## Container Restart Timing

After `docker restart hindsight`, the bge-m3 embedding model takes ~25 seconds to load. During this time:
- `/health` may return healthy but `/memories/recall` returns 502 or connection refused
- Wait at least 30s after restart before running eval or pipeline steps that call recall
- Test with: `curl -s -m 60 -X POST http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall -H 'Content-Type: application/json' -d '{"query":"test","limit":1}'`

## Full Pipeline Steps (16 steps)

```
1.  preflight
2.  status
3.  queue_drain_before_daily
4.  session_manifest
5.  retain_session
6.  daily_reflect
7.  native_consolidation_drain_after_daily
8.  v2_rebuild (daily)
9.  native_consolidation_drain_before_weekly
10. weekly_reflect
11. native_consolidation_drain_after_weekly
12. v2_rebuild (weekly)
13. conflict_audit
14. repair_zone_proposals (skip with --skip-repair-zone)
15. proposal_review (skip with --skip-proposal-review)
16. wiki_auto_maintenance (requires --include-wiki)
```

Run with: `python3 hindsight_memory_pipeline.py full --execute --confirm run-hindsight-pipeline --include-wiki`

If pipeline fails partway, earlier steps may have succeeded — only rerun the failed step instead of starting from scratch.

## Troubleshooting 403/502 Errors

| Error | Likely Cause | Fix |
|-------|-------------|-----|
| 403 "This token has no access to model deepseek/deepseek-v4-flash" | Model name has `deepseek/` prefix | Change to `deepseek-v4-flash` (no prefix) |
| 403 with correct model name | API key expired or quota depleted | Check topenrouter dashboard |
| 502 Bad Gateway during recall | Container just restarted, embedding model loading | Wait 30s and retry |
| 401 Authentication Fails | Wrong API key for the endpoint | Verify key matches the service (topenrouter key ≠ deepseek key) |
