# Hindsight LLM Provider Configuration

## Current Production Config (2026-05-26)

| Component | Value |
|-----------|-------|
| `pipeline_config.json` `llm_profile` | `deepseek-v4-flash` (internal name, NOT `topenrouter`) |
| `HINDSIGHT_OFFLINE_LLM_BASE_URL` | `https://tp-api.chinadatapay.com:8000/v1` |
| `HINDSIGHT_OFFLINE_LLM_MODEL` | `deepseek/deepseek-v4-flash` |
| `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` | `TOPENROUTER_API_KEY` |
| `review.proposal_review.llm_api_key_env` | `TOPENROUTER_API_KEY` |
| `review.proposal_review.llm_base_url` | `https://tp-api.chinadatapay.com:8000/v1` |
| `review.proposal_review.llm_model` | `deepseek/deepseek-v4-flash` |

## Two-Layer Override System

1. **Layer 1 (profile defaults):** `pipeline_config.json` → `llm_profile` → selects a profile in `hindsight_minimax_import.py` `LLM_PROFILES` dict. Each profile defines `base_url`, `model`, `api_key_envs`.
2. **Layer 2 (.env overrides):** `HINDSIGHT_OFFLINE_LLM_BASE_URL`, `HINDSIGHT_OFFLINE_LLM_MODEL`, `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` override the profile defaults at runtime.

The container reads env vars at startup. Changing `.env` requires `docker restart hindsight` to take effect.

## Valid llm-profile Names

Internal names accepted by `--llm-profile` and `pipeline_config.json`:
- `deepseek` — deepseek-chat at api.deepseek.com
- `deepseek-v4-flash` — deepseek-v4-flash at api.deepseek.com (default key: DEEPSEEK_API_KEY)
- `deepseek-v4-pro` — deepseek-v4-pro at api.deepseek.com
- `glm` — glm-5 at dashscope.aliyuncs.com
- `minimax` — MiniMax-M2.7 at api.minimaxi.com
- `opencode-go-deepseek-v4-flash` — deepseek-v4-flash at opencode.ai

**`topenrouter` is NOT a valid llm-profile.** It is a provider routing name for the ChinaDataPay relay.

## Provider Distinction

| Name | Base URL | Needs Proxy | API Key Env |
|------|----------|-------------|-------------|
| topenrouter | `https://tp-api.chinadatapay.com:8000/v1` | No | `TOPENROUTER_API_KEY` |
| openrouter.ai | `https://openrouter.ai/api/v1` | Yes | `OPENROUTER_API_KEY` |
| deepseek direct | `https://api.deepseek.com/v1` | No | `DEEPSEEK_API_KEY` |

## Debugging Path for 403/LLM Failures

1. Check which endpoint the pipeline is actually hitting: look at `hindsight_minimax_import.py` output for `base_url`
2. Check which API key is being used: `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` in `.env`
3. Verify the key works against the endpoint: `curl -s -m 15 -X POST '<base_url>/chat/completions' -H "Authorization: Bearer $KEY" -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'`
4. If 403/401: key doesn't match the endpoint (e.g. DEEPSEEK_API_KEY against topenrouter URL)
5. If key is wrong, fix `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` in `.env` and restart container
6. If base_url is wrong, fix `HINDSIGHT_OFFLINE_LLM_BASE_URL` in `.env` and restart container

## Container Startup 502 Window

After `docker restart hindsight`, API returns 502 for ~25s while bge-m3 loads. Wait for health check before running eval/recall:

```bash
for i in $(seq 1 12); do curl -s -m 5 http://127.0.0.1:8888/health && break; sleep 5; done
```
