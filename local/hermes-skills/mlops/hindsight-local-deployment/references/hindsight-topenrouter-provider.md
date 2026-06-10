# Hindsight LLM Provider: topenrouter (ChinaDataPay)

Session: 2026-05-26
Context: Switching Hindsight pipeline from MiniMax-M2.7 to deepseek-v4-flash via topenrouter

## What is topenrouter

`topenrouter` is a ChinaDataPay-hosted API relay at `https://tp-api.chinadatapay.com:8000/v1`.
It provides OpenAI-compatible chat completions with models like `deepseek/deepseek-v4-flash`.

Key characteristics:
- **No proxy needed** — direct access from China, no VPN/Clash required
- **OpenAI-compatible format** — works with `hindsight_provider: "openai"` in Hindsight profiles
- **NOT openrouter.ai** — `topenrouter` ≠ `openrouter.ai/api/v1`. They are completely different services
- **Model naming** — use `deepseek/deepseek-v4-flash` (with `deepseek/` prefix), not bare `deepseek-v4-flash`

## Hermes config.yaml Entry

```yaml
providers:
  topenrouter:
    base_url: https://tp-api.chinadatapay.com:8000/v1
    key_env: TOPENROUTER_API_KEY
    provider: topenrouter
    models:
      deepseek-v4-flash:
        context_length: 1024000
```

## Hindsight Integration

### Method 1: .env Override (Recommended for pipeline)

The offline pipeline reads env overrides that take priority over profile defaults:

```bash
# In ~/.hermes/.env
HINDSIGHT_OFFLINE_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_OFFLINE_LLM_MODEL=deepseek/deepseek-v4-flash
HINDSIGHT_OFFLINE_LLM_API_KEY_ENV=TOPENROUTER_API_KEY
HINDSIGHT_OFFLINE_HINDSIGHT_PROVIDER=openai
```

These override whatever the `llm_profile` in `pipeline_config.json` specifies for base_url/model/api_key.

### Method 2: New LLM Profile

Add a custom profile to `LLM_PROFILES` in `hindsight_minimax_import.py`:

```python
"topenrouter-v4-flash": {
    "label": "topenrouter-v4-flash",
    "hindsight_provider": "openai",
    "model": "deepseek/deepseek-v4-flash",
    "base_url": "https://tp-api.chinadatapay.com:8000/v1",
    "api_key_envs": ["TOPENROUTER_API_KEY"],
    "response_format": True,
    "strict_schema": False,
},
```

Then set `llm_profile: topenrouter-v4-flash` in `pipeline_config.json`.

### Method 3: Existing Profile + .env Override (Current Production)

Use `llm_profile: deepseek-v4-flash` in `pipeline_config.json` (which defaults to `api.deepseek.com`),
but override via `.env` to route through topenrouter. This is the current production setup.

## pipeline_config.json Review Section

The `review.proposal_review` section also needs to point to topenrouter:

```json
{
  "llm_api_key_env": "TOPENROUTER_API_KEY",
  "llm_base_url": "https://tp-api.chinadatapay.com:8000/v1",
  "llm_model": "deepseek/deepseek-v4-flash"
}
```

## Pitfalls

1. **403 Forbidden when using wrong API key** — topenrouter requires `TOPENROUTER_API_KEY`, not `DEEPSEEK_API_KEY`. The `deepseek-v4-flash` profile defaults to `DEEPSEEK_API_KEY`. If using this profile without .env override, the wrong key gets sent to topenrouter → 403.

2. **`topenrouter` is NOT a valid `--llm-profile` name** — Hindsight's `--llm-profile` accepts only internal profile names: `deepseek`, `deepseek-v4-flash`, `deepseek-v4-pro`, `glm`, `minimax`, `opencode-go-deepseek-v4-flash`, plus custom profiles added to the source. `topenrouter` is a Hermes provider name, not a Hindsight profile name.

3. **Model name must include prefix** — topenrouter expects `deepseek/deepseek-v4-flash`, not bare `deepseek-v4-flash`. The `.env` override `HINDSIGHT_OFFLINE_LLM_MODEL` must use the prefixed form.

4. **Don't confuse with openrouter.ai** — `openrouter.ai/api/v1` is a different service that requires proxy access and has its own auth format. `topenrouter` (ChinaDataPay) is domestic, direct-access.

5. **Duplicate HINDSIGHT_OFFLINE_LLM_MODEL entries in .env** — If both `deepseek-v4-flash` and `deepseek/deepseek-v4-flash` exist, the script reads the first match. Clean up duplicates keeping only the prefixed version.

6. **Review section key mismatch** — `pipeline_config.json` `review.proposal_review.llm_api_key_env` must match the provider's key env. If set to `DEEPSEEK_API_KEY` while the base_url points to topenrouter, calls will 403/401.

## Verification

```bash
# Quick smoke test
TOR_KEY=$(python3 -c "
from pathlib import Path
for line in Path('$HOME/.hermes/.env').read_text().splitlines():
    if line.strip().startswith('TOPENROUTER_API_KEY='):
        print(line.split('=',1)[1])
")
curl -s -m 15 -X POST 'https://tp-api.chinadatapay.com:8000/v1/chat/completions' \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer $TOR_KEY" \
  -d '{"model":"deepseek/deepseek-v4-flash","messages":[{"role":"user","content":"say ok"}],"max_tokens":5}'
# Expect: 200 with content "ok"
```

---

_Updated: 2026-05-26_
