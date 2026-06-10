# Hindsight Consolidation Configuration Reference

From official docs: https://hindsight.vectorize.io/developer/configuration

## LLM Provider (独立配置)

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER` | 继承主 LLM | Consolidation 专用 provider |
| `HINDSIGHT_API_CONSOLIDATION_LLM_API_KEY` | 继承主 LLM | Consolidation 专用 API key |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MODEL` | 继承主 LLM | Consolidation 专用模型 |
| `HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL` | 继承主 LLM | Consolidation 专用 endpoint |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_CONCURRENT` | 继承主 LLM | 并发请求数 |
| `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES` | 继承主 LLM | 重试次数 |
| `HINDSIGHT_API_CONSOLIDATION_LLM_TIMEOUT` | 继承主 LLM | 超时（秒） |

## 批处理控制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND` | 100 | 每轮最大记忆数（0=无限） |
| `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE` | 8 | 单次 LLM 调用发送的 facts 数（1=禁用批处理） |
| `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE` | 50 | 内部优化：每批加载记忆数 |
| `HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS` | 3 | 外层重试次数 |

## Recall 预算控制

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HINDSIGHT_API_CONSOLIDATION_RECALL_BUDGET` | low | 内部 recall 预算（low/mid/high） |
| `HINDSIGHT_API_CONSOLIDATION_MAX_TOKENS` | 1024 | 查找相关 observations 的 max tokens |
| `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS` | 4096 | Source facts 总 token 上限（-1=无限） |
| `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS_PER_OBSERVATION` | 256 | 每个 observation 的 source facts token 上限 |

## 功能开关

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HINDSIGHT_API_ENABLE_OBSERVATIONS` | true | 是否启用 observation consolidation |
| `HINDSIGHT_API_MAX_OBSERVATIONS_PER_SCOPE` | -1 | 每个 tag scope 最大 observations（-1=无限） |
| `HINDSIGHT_API_OBSERVATIONS_MISSION` | - | 自定义 consolidation 规则 |

## Worker Slots

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS` | 2 | Consolidation 专用 worker slots |

## Webhook

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HINDSIGHT_API_WEBHOOK_EVENT_TYPES` | consolidation.completed | 事件类型通知 |

## 成本优化建议

**降低频率**：
```bash
HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=50  # 每轮处理更少
```

**降低 token**：
```bash
HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=2048  # 更小 prompt
HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=4              # 更小 batch
```

**用更便宜的模型**：
```bash
HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER=openai
HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL=https://tp-api.chinadatapay.com:8000/v1
HINDSIGHT_API_CONSOLIDATION_LLM_MODEL=deepseek-v4-flash  # TopenRouter via openai-compatible
```

## Valid LLM Provider Names (v0.6.1)

Hindsight validates `*_LLM_PROVIDER` against this list at startup. Using an unrecognized name causes `ValueError: Invalid LLM provider` and the container exits immediately.

```
openai, groq, ollama, gemini, anthropic, lmstudio, llamacpp, vertexai,
openai-codex, claude-code, mock, none, minimax, deepseek, litellm,
litellmrouter, bedrock, volcano, openrouter, zai
```

**OpenAI-compatible third-party endpoints** (TopenRouter, OpenCode, etc.) must use `provider=openai` with a custom `BASE_URL`. Do NOT invent provider names — Hindsight will reject them.

## Switching LLM Provider (Operational)

All 4 LLM groups must be updated together when switching providers:
- `HINDSIGHT_API_LLM_*` (general)
- `HINDSIGHT_API_RETAIN_LLM_*`
- `HINDSIGHT_API_CONSOLIDATION_LLM_*`
- `HINDSIGHT_API_REFLECT_LLM_*`

Each group has: `PROVIDER`, `MODEL`, `BASE_URL`, `API_KEY`.

After container recreate, reapply the parallel consolidator patch:
```bash
python3 ~/.hermes/scripts/patch_hindsight_consolidator_parallel.py
docker restart hindsight
```

If consolidation was interrupted by the switch, recover stuck operations:
```bash
curl -X POST http://127.0.0.1:8888/v1/default/banks/hermes/consolidation/recover
```