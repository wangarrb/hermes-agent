# Hindsight LLM Provider Naming Pitfall

## Valid Provider Names

Hindsight's `HINDSIGHT_API_LLM_PROVIDER` only accepts a fixed, hardcoded list. Arbitrary provider names like `topenrouter` will cause a startup crash:

```
ValueError: Invalid LLM provider: topenrouter. Must be one of:
openai, groq, ollama, gemini, anthropic, lmstudio, llamacpp, vertexai,
openai-codex, claude-code, mock, none, minimax, deepseek, litellm,
litellmrouter, bedrock, volcano, openrouter, zai
```

## Switching to an OpenAI-Compatible Provider (e.g. TopenRouter)

When targeting an OpenAI-compatible endpoint that isn't on the list:

- Set `HINDSIGHT_API_LLM_PROVIDER=openai` (or `minimax`/`deepseek` if the endpoint speaks that specific protocol)
- Set `HINDSIGHT_API_LLM_BASE_URL` to the provider's v1 endpoint
- Set `HINDSIGHT_API_LLM_API_KEY` to the appropriate key
- Set `HINDSIGHT_API_LLM_MODEL` to the model name the provider expects

Example switching from MiniMax to TopenRouter/deepseek-v4-flash:

```bash
HINDSIGHT_API_LLM_PROVIDER=openai
HINDSIGHT_API_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash
HINDSIGHT_API_LLM_API_KEY=sk-xxx
```

All 4 groups (llm, retain, consolidation, reflect) must be changed together for consistency.

## Verification

After switching, check the container logs for the LLM config line:

```
2026-05-24 03:33:21,344 - INFO - hindsight_api.config - LLM: provider=openai, model=deepseek-v4-flash
```

And look for successful LLM calls in the worker logs:

```
slow llm call: scope=consolidation, model=openai/deepseek-v4-flash,
input_tokens=2696, output_tokens=385, total_tokens=3081
```

## Notes

- The `openai` provider maps to `OpenAI-compatible LLM class` internally — it works with any OpenAI-format endpoint.
- Docker container recreate will lose the parallel consolidator patch; reapply `patch_hindsight_consolidator_parallel.py` after first startup.
- There is no separate `topenrouter` provider class in Hindsight; always use `openai` as the bridge.