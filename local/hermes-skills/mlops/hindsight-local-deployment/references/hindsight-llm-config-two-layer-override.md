# Hindsight LLM Configuration: Two-Layer Override

## Architecture

Hindsight LLM config has two layers:

1. **Layer 1: `pipeline_config.json`** — sets `llm_profile` (internal profile name)
2. **Layer 2: `.env` overrides** — `HINDSIGHT_OFFLINE_LLM_*` vars override the profile's base_url, model, and API key

The container reads env vars at startup; `.env` changes require `docker restart hindsight`.

## Current Production Config (as of 2026-05-26)

### pipeline_config.json
```json
{
  "llm_profile": "deepseek-v4-flash",
  "review": {
    "proposal_review": {
      "llm_api_key_env": "TOPENROUTER_API_KEY",
      "llm_base_url": "https://tp-api.chinadatapay.com:8000/v1",
      "llm_model": "deepseek/deepseek-v4-flash"
    }
  }
}
```

### .env overrides
```
HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_OFFLINE_LLM_MODEL=deepseek/deepseek-v4-flash
HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY
```

## Valid LLM Profile Names (internal)

`deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash` plus custom.

**`topenrouter` is NOT a valid llm_profile** — it is a provider routing name, not an internal profile. Using it causes `Unknown --llm-profile='topenrouter'` error.

## Provider vs Profile Distinction

| Concept | Example | Scope |
|---------|---------|-------|
| LLM profile | `deepseek-v4-flash` | Hindsight internal, used in config/CLI |
| Provider routing | `topenrouter` | Hermes config.yaml, NOT Hindsight |
| API endpoint | `tp-api.chinadatapay.com:8000/v1` | topenrouter (ChinaDataPay relay, domestic) |
| API endpoint | `openrouter.ai/api/v1` | openrouter.ai (international, needs proxy) |
| API endpoint | `api.deepseek.com/v1` | DeepSeek official (international, may need proxy) |

**topenrouter ≠ openrouter.ai** — completely different services, different keys, different behavior. topenrouter needs no proxy; openrouter.ai does.

## Common Errors

1. **403 Forbidden**: Wrong API key for the endpoint. topenrouter endpoint requires `TOPENROUTER_API_KEY`, not `DEEPSEEK_API_KEY`. Check `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` matches the endpoint.

2. **502 Bad Gateway**: Container just restarted; embedding model loading takes ~25s. Wait for `/health` OK before running eval/recall.

3. **Recall timeout**: `POST /memories/recall` is bge-m3 + pgvector vector search, NOT an LLM call. Timeout means DB/embedding pressure, not LLM provider issues. Don't switch LLM profiles to "fix" recall timeouts.

4. **Unknown llm-profile**: `topenrouter` is not valid. Use `deepseek-v4-flash` in pipeline_config.json, then override endpoint/key via `.env`.

## Switching Providers

To switch from topenrouter to DeepSeek official:

```bash
# Remove .env overrides, let profile defaults take over
sed -i '/HINDSIGHT_OFFLINE_LLM_BASE_URL/d' ~/.hermes/.env
sed -i '/HINDSIGHT_OFFLINE_LLM_MODEL/d' ~/.hermes/.env
sed -i '/HINDSIGHT_OFFLINE_LLM_API_KEY_ENV/d' ~/.hermes/.env

# Update pipeline_config.json review section
# llm_api_key_env -> DEEPSEEK_API_KEY
# llm_base_url -> https://api.deepseek.com/v1

# Restart container
docker restart hindsight
```

To switch from topenrouter to opencode-go (direct, ~1s latency):

```bash
# Change .env overrides
HINDSIGHT_OFFLINE_LLM_BASE_URL=https://opencode.ai/zen/go/v1
HINDSIGHT_OFFLINE_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=OPENCODE_GO_API_KEY

# Change pipeline_config.json llm_profile
"llm_profile": "opencode-go-deepseek-v4-flash"

docker restart hindsight
```