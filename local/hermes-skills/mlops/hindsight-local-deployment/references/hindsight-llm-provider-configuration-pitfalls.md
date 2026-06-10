# Hindsight LLM Provider Configuration Pitfalls (2026-05-26)

## Critical: Do Not Change Hindsight Model Config Without User Confirmation

The user explicitly requires confirmation before any changes to `llm_profile`, `base_url`, `api_key_env`, or `llm_model` in Hindsight configuration files. Do not silently switch providers, models, or keys.

## topenrouter ≠ openrouter.ai

- **topenrouter** is ChinaDataPay's API relay: `https://tp-api.chinadatapay.com:8000/v1`
- It is NOT the same as `openrouter.ai` (OpenRouter's global service at `https://openrouter.ai/api/v1`)
- topenrouter is OpenAI-compatible per their docs (https://topenrouter.com/docs), supports `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, etc.
- topenrouter does NOT need proxy — it's a domestic China service, direct connection works
- Available models (as of 2026-05): `deepseek-v4-flash`, `deepseek-v4-pro`, and others listed at `/v1/models`

## Model Name Prefix Trap (403 Root Cause)

topenrouter registers models WITHOUT the `deepseek/` prefix:
- **Correct**: `deepseek-v4-flash` → returns 200 OK
- **Wrong**: `deepseek/deepseek-v4-flash` → returns 403 "This token has no access to model"
- **Wrong**: `DeepSeek-V4-Flash` → returns 403
- **Wrong**: `deepseek_v4_flash` → returns 403

This is the opposite of OpenRouter (`openrouter.ai`), which uses `deepseek/deepseek-v4-flash` format.

When switching between topenrouter and direct DeepSeek API (`api.deepseek.com/v1`), the model name format differs:
- topenrouter: `deepseek-v4-flash` (no prefix)
- DeepSeek official: `deepseek-chat` or model-specific names with different conventions

## Hindsight llm-profile Internal Names

Hindsight's `--llm-profile` flag accepts internal profile names defined in `LLM_PROFILES` dict inside `hindsight_minimax_import.py`:
- `minimax`, `glm`, `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `opencode-go-deepseek-v4-flash`
- `topenrouter` is NOT a valid llm-profile — it's a Hermes provider routing name
- Profile aliases exist: `bailian`→`glm`, `opencode-go`→`opencode-go-deepseek-v4-flash`

To route Hindsight through topenrouter while using the `deepseek-v4-flash` profile, override via `.env`:
```
HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_OFFLINE_LLM_MODEL=deepseek-v4-flash          # NOT deepseek/deepseek-v4-flash
HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY
```

The profile's default `base_url` and `api_key_env` get overridden by these env vars. The `hindsight_provider` inside the profile (`openai`) remains unchanged because topenrouter is OpenAI-compatible.

## Container Restart 502 Window

After `docker restart hindsight`, the bge-m3 embedding model takes ~25s to load. During this window:
- `/health` may return connection refused or empty response
- `/memories/recall` returns 502 Bad Gateway
- All dependent pipeline steps (eval, V2 publish) will fail if they hit this window

**Mitigation**: wait for `/health` to return `{"status":"healthy","database":"connected"}` AND verify a recall smoke test succeeds before running eval or other API-dependent steps.

## pipeline_config.json Review Section

The `review.proposal_review` section also needs consistent LLM config:
```json
{
  "llm_api_key_env": "TOPENROUTER_API_KEY",
  "llm_base_url": "https://tp-api.chinadatapay.com:8000/v1",
  "llm_model": "deepseek-v4-flash"
}
```

All three fields (key_env, base_url, model) must match the same provider. Mixing providers (e.g. key from one, base_url from another) causes authentication failures.

## Full Pipeline Step Reference

Hindsight full pipeline (`--execute --confirm run-hindsight-pipeline`):

1. preflight (environment checks)
2. status (DB/API health)
3. queue_drain_before_daily (wait for existing queue)
4. session_manifest (scan new sessions, generate manifest)
5. retain_session (incremental retain → extract source facts)
6. daily_reflect (offline daily consolidation, calls LLM)
7. native_consolidation_drain_after_daily
8. v2_rebuild (daily → canonical observations)
9. native_consolidation_drain_before_weekly
10. weekly_reflect (offline weekly consolidation, calls LLM)
11. native_consolidation_drain_after_weekly
12. v2_rebuild (weekly → canonical observations)
13. conflict_audit
14. repair_zone_proposals (skip with `--skip-repair-zone`)
15. proposal_review (needs LLM, skip with `--skip-proposal-review`)
16. wiki_auto_maintenance (only with `--include-wiki`)

Key flags: `--include-wiki`, `--skip-daily`, `--history all`, `--no-wait-native-consolidation`

## Diagnosis Checklist for 403/502 Errors

When Hindsight LLM calls fail with 403:
1. Check model name format (topenrouter: no `deepseek/` prefix)
2. Check `api_key_env` matches the actual key for the `base_url`
3. Check key balance/status on the provider dashboard
4. Check `.env` overrides vs pipeline_config.json vs container env for consistency

When eval/pipeline fails with 502:
1. Check container uptime — if <30s, embedding model is still loading
2. Wait for `/health` healthy + recall smoke test
3. Re-run only the failed step, not the full pipeline from scratch