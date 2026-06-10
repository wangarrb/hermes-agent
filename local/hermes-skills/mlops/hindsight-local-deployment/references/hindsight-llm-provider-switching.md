# Hindsight LLM Provider Switching Runbook

**Session**: 2026-05-24
**Context**: Switching Hindsight from MiniMax-M2.7 to TopenRouter/deepseek-v4-flash

---

## Valid LLM Provider Names

Hindsight validates the `HINDSIGHT_API_LLM_PROVIDER` value against a hardcoded whitelist at startup:

```
openai, groq, ollama, gemini, anthropic, lmstudio, llamacpp, vertexai,
openai-codex, claude-code, mock, none, minimax, deepseek, litellm,
litellmrouter, bedrock, volcano, openrouter, zai
```

**Any string not in this list causes an immediate startup crash:**

```
ValueError: Invalid LLM provider: <name>. Must be one of: openai, groq, ...
```

### Using OpenAI-Compatible Endpoints (TopenRouter, etc.)

For endpoints that speak OpenAI-compatible chat completions API but aren't named in the whitelist (e.g. TopenRouter, 讯飞, custom proxies), use `provider=openai` with the custom `BASE_URL`:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_LLM_API_KEY=<your-key>
```

**Do NOT use** `provider=topenrouter` or any custom name — it will crash.

Note: `openrouter` IS a valid built-in provider name (pointing to openrouter.ai), but that's a different service from TopenRouter (tp-api.chinadatapay.com). If you want the Chinese TopenRouter gateway, use `provider=openai` + TopenRouter's base_url.

---

## 4 LLM Groups That Must Be Changed Together

Hindsight has 4 independent LLM configuration groups. All must be switched consistently:

| Group | Provider var | Model var | Base URL var | API Key var |
|-------|-------------|-----------|-------------|-------------|
| Main LLM | `HINDSIGHT_API_LLM_PROVIDER` | `HINDSIGHT_API_LLM_MODEL` | `HINDSIGHT_API_LLM_BASE_URL` | `HINDSIGHT_API_LLM_API_KEY` |
| Retain | `HINDSIGHT_API_RETAIN_LLM_PROVIDER` | `HINDSIGHT_API_RETAIN_LLM_MODEL` | `HINDSIGHT_API_RETAIN_LLM_BASE_URL` | `HINDSIGHT_API_RETAIN_LLM_API_KEY` |
| Consolidation | `HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER` | `HINDSIGHT_API_CONSOLIDATION_LLM_MODEL` | `HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL` | `HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY` |
| Reflect | `HINDSIGHT_API_REFLECT_LLM_PROVIDER` | `HINDSIGHT_API_REFLECT_LLM_MODEL` | `HINDSIGHT_API_REFLECT_LLM_BASE_URL` | `HINDSIGHT_API_REFLECT_LLM_API_KEY` |

If retain/consolidation/reflect vars are omitted, they inherit from the main LLM group. But explicit is safer and avoids ambiguity.

---

## Provider Switch Procedure

### 1. Verify the new endpoint works

```bash
curl -s --max-time 30 <base_url>/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"只回复OK"}],"max_tokens":16,"temperature":0}'
```

Must return a valid chat completion response before proceeding.

### 2. Check for active operations

```bash
curl -s http://127.0.0.1:8888/v1/default/banks/hermes/operations?exclude_parents=true&limit=5
```

If there are `processing` operations, they will be interrupted. Hindsight auto-recovers stuck ops after restart, but be aware.

### 3. Save current container env vars (for rollback)

```bash
docker inspect hindsight --format '{{range .Config.Env}}{{println .}}{{end}}' > ~/hindsight-env-backup-$(date +%Y%m%d).txt
```

### 4. Stop and remove the old container

```bash
docker stop hindsight
docker rm hindsight
```

### 5. Recreate with new LLM env vars

Use `docker run` with ALL the same env vars (copy from the backup), changing only the 4 groups of LLM vars. Key template for TopenRouter:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_LLM_API_KEY=$TOPENROUTER_KEY

HINDSIGHT_API_RETAIN_LLM_PROVIDER=openai
HINDSIGHT_API_RETAIN_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_RETAIN_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_RETAIN_LLM_API_KEY=$TOPENROUTER_KEY

HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER=openai
HINDSIGHT_API_CONSOLIDATION_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY=$TOPENROUTER_KEY

HINDSIGHT_API_REFLECT_LLM_PROVIDER=openai
HINDSIGHT_API_REFLECT_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_REFLECT_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_REFLECT_LLM_API_KEY=$TOPENROUTER_KEY
```

### 6. Reapply writable-layer patches

Container recreate loses all writable-layer patches. Must reapply:

```bash
python3 ~/.hermes/scripts/patch_hindsight_consolidator_parallel.py
python3 ~/.hermes/scripts/patch_hindsight_retain_temporal_fk_guard.py  # if used
```

### 7. Restart container (so patches take effect)

```bash
docker restart hindsight
```

### 8. Verify

```bash
# Health check
curl -s http://127.0.0.1:8888/health

# Check logs show correct provider/model
docker logs hindsight 2>&1 | grep -E "provider=|model=" | head -8

# Check consolidation is working with new model
docker logs hindsight --tail 20 2>&1 | grep "llm call"
```

Expected log output:
```
LLM: provider=openai, model=deepseek-v4-flash
LLM (retain): provider=openai, model=deepseek-v4-flash
LLM (reflect): provider=openai, model=deepseek-v4-flash
LLM (consolidation): provider=openai, model=deepseek-v4-flash
```

And LLM calls should show:
```
slow llm call: scope=consolidation, model=openai/deepseek-v4-flash, input_tokens=..., output_tokens=...
```

---

## Rollback

If the new provider fails, reverse the procedure:

1. Stop and remove the current container
2. Recreate with the original env vars from the backup file
3. Reapply patches
4. Restart

---

## Provider Comparison (as of 2026-05-24)

| Provider | Endpoint | Model | Cost | Latency | Notes |
|----------|----------|-------|------|---------|-------|
| MiniMax | api.minimaxi.com/v1 | MiniMax-M2.7 | ~¥0.01/1K tokens | 8-15s/consolidation call | Direct, no proxy needed |
| TopenRouter | tp-api.chinadatapay.com:8000/v1 | deepseek-v4-flash | ~¥0.001/1K tokens (10x cheaper) | 12-40s/consolidation call | Domestic, no proxy; `provider=openai` in Hindsight |
| DeepSeek | api.deepseek.com/v1 | deepseek-chat | ~¥0.001/1K tokens | 10-30s/consolidation call | Direct, no proxy; `provider=deepseek` in Hindsight |

---

_Updated: 2026-05-24_
