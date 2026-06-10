# TopenRouter LLM Configuration for Hindsight

## What is TopenRouter

TopenRouter is ChinaDataPay's AI model relay service. It is NOT OpenRouter (openrouter.ai).

- **Base URL**: `https://tp-api.chinadatapay.com:8000/v1`
- **Docs**: https://topenrouter.com/docs
- **Format**: OpenAI-compatible API
- **Proxy**: Not needed (domestic service, direct connect)
- **Auth**: `Authorization: Bearer <TOPENROUTER_API_KEY>`

## Model Name Convention

**Critical**: topenrouter registers models WITHOUT provider prefixes.

| Correct (topenrouter) | Wrong (will 403) |
|---|---|
| `deepseek-v4-flash` | `deepseek/deepseek-v4-flash` |
| `deepseek-v4-pro` | `deepseek/deepseek-v4-pro` |

Always verify available models:
```bash
curl -s 'https://tp-api.chinadatapay.com:8000/v1/models' \
  -H "Authorization: Bearer $TOPENROUTER_API_KEY" | \
  python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]"
```

## Hindsight Configuration

### pipeline_config.json

```json
{
  "llm_profile": "deepseek-v4-flash",
  "review": {
    "proposal_review": {
      "llm_api_key_env": "TOPENROUTER_API_KEY",
      "llm_base_url": "https://tp-api.chinadatapay.com:8000/v1",
      "llm_model": "deepseek-v4-flash"
    }
  }
}
```

### .env overrides

The offline reflect/import scripts read these overrides:

```bash
HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_OFFLINE_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY
```

### Container environment

Verify after config changes:
```bash
docker exec hindsight env | grep 'LLM_MODEL\|LLM_BASE_URL\|LLM_API_KEY'
```

Expected:
```
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
```

## Hindsight Internal Profile Names

The `--llm-profile` flag uses Hindsight-internal names, NOT provider names:

- `deepseek-v4-flash` (Hindsight profile → uses `api.deepseek.com/v1` by default, overridden by `.env`)
- `deepseek-v4-pro`
- `glm`
- `minimax`
- `opencode-go-deepseek-v4-flash`

`topenrouter` is NOT a valid `--llm-profile` value. It is a Hermes provider routing name.

## 403 Forbidden Root Causes

1. **Model name with prefix**: `deepseek/deepseek-v4-flash` → 403. Use `deepseek-v4-flash`.
2. **Wrong API key**: Using `DEEPSEEK_API_KEY` against topenrouter endpoint → 401/403. Must use `TOPENROUTER_API_KEY`.
3. **Key quota exhausted**: Check balance at tp-api.chinadatapay.com dashboard. Key may be valid but out of credits.

## Verification

```bash
# Test topenrouter connectivity
TOR_KEY=$(grep TOPENROUTER_API_KEY ~/.hermes/.env | head -1 | cut -d= -f2)
curl -s -m 15 -X POST 'https://tp-api.chinadatapay.com:8000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOR_KEY" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'
```

Expected: `{"choices":[{"message":{"content":"ok",...}}],...}`

## Pipeline Full Step Reference (2026-05-26)

1. preflight
2. status
3. queue_drain_before_daily
4. session_manifest
5. retain_session
6. daily_reflect (calls LLM)
7. native_consolidation_drain_after_daily
8. v2_rebuild (daily → canonical observations)
9. native_consolidation_drain_before_weekly
10. weekly_reflect (calls LLM)
11. native_consolidation_drain_after_weekly
12. v2_rebuild (weekly → canonical observations)
13. conflict_audit
14. repair_zone_proposals (can `--skip-repair-zone`)
15. proposal_review (calls LLM, can `--skip-proposal-review`)
16. wiki_auto_maintenance (requires `--include-wiki`)

Key flags:
- `--execute --confirm run-hindsight-pipeline` (execution mode)
- `--include-wiki` (include wiki step)
- `--skip-daily` (skip daily, run weekly only — only valid with `full`)
- `--no-wait-native-consolidation` (don't wait for consolidation drain)

Failure recovery: earlier succeeded steps don't need re-running. Resume from the failed step.
