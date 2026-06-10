# TopenRouter Provider Configuration for Hindsight

## What is TopenRouter

TopenRouter (tp-api.chinadatapay.com) is a ChinaDataPay AI API relay service. It provides an OpenAI-compatible API format for calling DeepSeek, GLM and other models. **Not** the same as openrouter.ai — it is a domestic relay, does not require proxy, and has its own model naming conventions.

API base URL: `https://tp-api.chinadatapay.com:8000/v1`

Docs: https://topenrouter.com/docs

## Model Naming Pitfall (CRITICAL)

TopenRouter registers models **without provider prefixes**. The correct model name is:

```
deepseek-v4-flash      ← CORRECT (topenrouter naming)
deepseek/deepseek-v4-flash  ← WRONG (returns 403 "This token has no access")
```

Other models follow the same pattern: `deepseek-v4-pro`, `glm-5`, etc. Always use the name without the `deepseek/` prefix when calling topenrouter.

To verify available models for your token:

```bash
curl -s 'https://tp-api.chinadatapay.com:8000/v1/models' \
  -H "Authorization: Bearer $TOPENROUTER_API_KEY" | \
  python3 -c "import sys,json; [print(m['id']) for m in json.load(sys.stdin).get('data',[])]"
```

## Hindsight Configuration

Hindsight uses `llm_profile` internal names (deepseek-v4-flash, minimax, glm, etc.) — `topenrouter` is NOT a valid llm-profile value. To make Hindsight use topenrouter, override through `.env`:

```env
HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_OFFLINE_LLM_MODEL=deepseek-v4-flash        # NO deepseek/ prefix!
HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY
```

In `pipeline_config.json`:
- `llm_profile`: `deepseek-v4-flash` (internal name, NOT `topenrouter`)
- `review.proposal_review.llm_model`: `deepseek-v4-flash` (again, no prefix)
- `review.proposal_review.llm_base_url`: `https://tp-api.chinadatapay.com:8000/v1`
- `review.proposal_review.llm_api_key_env`: `TOPENROUTER_API_KEY`

The offline pipeline script (`hindsight_minimax_import.py`) reads `.env` overrides at runtime and replaces the profile's default base_url/model/key with the override values. This means `llm_profile=deepseek-v4-flash` internally defaults to `api.deepseek.com/v1`, but `.env` overrides redirect it to topenrouter.

## Container Environment

When the Hindsight Docker container starts, it reads `.env` and sets:

```
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_LLM_API_KEY=<TOPENROUTER_API_KEY value>
HINDSIGHT_API_LLM_PROVIDER=openai
```

All four LLM channels (llm/retain/consolidation/reflect) get the same base_url/model/key override.

## Container Restart 502 Window

After container restart, the bge-m3 embedding model takes ~25 seconds to load. During this window, the API returns 502 Bad Gateway or connection refused. Any recall-dependent steps (eval, status) must wait for the container to be fully ready:

```bash
# Wait for API readiness
for i in $(seq 1 12); do
  if curl -s -m 5 http://127.0.0.1:8888/health | grep -q 'healthy'; then
    echo "API ready after ${i}x5s"
    break
  fi
  sleep 5
done

# Then test recall specifically (health endpoint may respond before embedding is loaded)
curl -s -m 60 -X POST 'http://127.0.0.1:8888/v1/default/banks/hermes/memories/recall' \
  -H 'Content-Type: application/json' -d '{"query":"test","limit":3}'
```

## Quick Connectivity Test

```bash
curl -s -m 15 -X POST 'https://tp-api.chinadatapay.com:8000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOPENROUTER_API_KEY" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'
```

Expected: `{"choices":[...]}` with content "ok".
If you get 403 "This token has no access" — you probably used `deepseek/deepseek-v4-flash` instead of `deepseek-v4-flash`.
If you get 401 — key is invalid or expired.

## Provider vs Profile Naming

| Concept | Value | Used Where |
|---------|-------|------------|
| Hermes provider name | `topenrouter` | Hermes `config.yaml` providers section |
| Hindsight llm_profile | `deepseek-v4-flash` | `pipeline_config.json`, `--llm-profile` CLI arg |
| Hindsight provider | `openai` | Container env `HINDSIGHT_API_LLM_PROVIDER` |
| TopenRouter model name | `deepseek-v4-flash` | API `model` field in chat/completions |
| DeepSeek official model name | `deepseek/deepseek-v4-flash` | NOT used with topenrouter |

Do NOT mix these. `topenrouter` is a Hermes routing name, not a Hindsight profile. The Hindsight container must use `provider=openai` because topenrouter is OpenAI-compatible format, not a custom provider.