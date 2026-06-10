# Hindsight LLM Profile Naming and Provider Configuration

**Session**: 2026-05-26
**Context**: Pipeline failed because `--llm-profile topenrouter` is invalid, and `HINDSIGHT_OFFLINE_LLM_BASE_URL` override pointed to a dead third-party proxy (403 Forbidden).

---

## Internal LLM Profile Names

`hindsight_minimax_import.py` defines these profiles in `LLM_PROFILES`:

| Profile Name | Model | Base URL | API Key Env | Provider |
|---|---|---|---|---|
| `minimax` | MiniMax-M2.7 | `https://api.minimaxi.com/v1` | `MINIMAX_CN_API_KEY` | `minimax` |
| `glm` | glm-5 | `https://coding.dashscope.aliyuncs.com/v1` | `BAILIAN_API_KEY` | `openai` |
| `deepseek` | deepseek-chat | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `openai` |
| `deepseek-v4-flash` | deepseek-v4-flash | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `openai` |
| `deepseek-v4-pro` | deepseek-v4-pro | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` | `openai` |
| `opencode-go-deepseek-v4-flash` | deepseek-v4-flash | `https://opencode.ai/zen/go/v1` | `OPENCODE_GO_API_KEY` | `openai` |

Plus `PROFILE_ALIASES`: `bailian`/`dashscope`/`zai`/`z.ai` → `glm`, `opencode-go` → `opencode-go-deepseek-v4-flash`.

## What `topenrouter` Is (And Why It Doesn't Work Here)

- `topenrouter` is the Hermes config provider name for routing through OpenRouter (`https://openrouter.ai/api/v1`).
- OpenRouter has its own API format (with `HTTP-Referer`, `X-Title` headers, model routing like `deepseek/deepseek-v4-flash`).
- It is **NOT** `openai`-compatible in the way Hindsight expects. Setting `hindsight_provider='openai'` with an OpenRouter base URL produces authentication and format errors.
- Hindsight's `--llm-profile` does not accept `topenrouter`. Use one of the internal profile names above.

## `.env` Override Pitfalls

`hindsight_minimax_import.py` reads these env vars to override profile defaults:

| Env Var | Overrides | Risk |
|---|---|---|
| `HINDSIGHT_OFFLINE_LLM_MODEL` | `profile["model"]` | Wrong model name → API errors |
| `HINDSIGHT_OFFLINE_LLM_BASE_URL` | `profile["base_url"]` | Dead/proxied URL → 403/401 |
| `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` | `profile["api_key_env"]` | Wrong key → auth failures |

**Key lesson (2026-05-26):** `HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1` was set in `.env`, overriding the default `https://api.deepseek.com/v1`. The third-party proxy key expired, causing 403 on every LLM call. Removing the override restored normal operation.

**Rule:** Only set these override env vars when intentionally routing through a known-good alternative. If LLM calls start failing with auth errors, check `.env` for stale overrides before debugging the profile or key itself.

## `pipeline_config.json` LLM Configuration

The `llm_profile` field in `pipeline_config.json` must use the internal profile name:

```json
{
  "llm_profile": "deepseek-v4-flash",
  "review": {
    "proposal_review": {
      "llm_api_key_env": "DEEPSEEK_API_KEY",
      "llm_base_url": "https://api.deepseek.com/v1",
      "llm_model": "deepseek/deepseek-v4-flash"
    }
  }
}
```

Do NOT set `llm_profile` to a provider routing name like `topenrouter`.

## Debugging Checklist for LLM Call Failures

1. Check `pipeline_config.json` → `llm_profile` is a valid internal name
2. Check `.env` for `HINDSIGHT_OFFLINE_LLM_BASE_URL` / `HINDSIGHT_OFFLINE_LLM_MODEL` / `HINDSIGHT_OFFLINE_LLM_API_KEY_ENV` overrides
3. Test the target API directly:
   ```bash
   curl -s -m 10 -o /dev/null -w "%{http_code}" \
     -X POST 'https://api.deepseek.com/v1/chat/completions' \
     -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
     -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
   # Expected: 200
   ```
4. If the direct API returns 200 but the pipeline fails, the `.env` override is likely pointing to a different (broken) endpoint
5. Remove stale `.env` overrides and re-run

---

_Updated: 2026-05-26_
