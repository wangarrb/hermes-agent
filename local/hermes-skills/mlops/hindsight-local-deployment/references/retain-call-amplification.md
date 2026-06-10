# Hindsight retain 调用量放大：chunk 不是 document

## 背景

2026-05-09 排查 paid retain 时，用户质疑“83 条数据为何触发 MiniMax 约 1500 次调用”。复核源码和队列后确认：Hindsight retain 的 LLM 调用单位不是 document/session，而是 retain chunk。

## 关键事实

- `extract_facts_from_text()` 会先用 `chunk_text(..., max_chars=8000)` 切分输入。
- 每个 chunk 至少触发一次结构化 facts extraction LLM 调用。
- 每个 chunk 还有两层放大：
  - chunk 级重试，例如 `MAX_CHUNK_RETRIES = 3`。
  - LLM 调用级重试，例如 `llm_config.call(... max_retries=2)`，也就是一次 chunk 失败时最多 3 次请求。
- 如果输出过长或结构化 JSON 失败，可能继续二分/重跑，进一步放大。
- 高并发只会更容易撞 provider rate limit，并不会减少总调用量。

## 量级 sanity check

排查样例：

- 原始 83 个超长 session/document 总字符约 2805 万。
- 按 `retain_chunk_size=8000`，实际约 3548 个 chunk。
- MiniMax 窗口约 10 个 doc / 621 个 chunk；近 5 小时约 1500 次调用。
- 1500 / 621 ≈ 2.4 calls/chunk，符合 chunk 重试 + JSON 重试 + rate-limit backoff 的正常放大量级。
- deepseek-v4-flash 初始窗口约 57 个 doc / 1951 个 chunk，因此“上千次调用”也合理。

## 诊断顺序

1. 不要用 document 数估算成本；先统计字符数和 chunk 数。
2. 按 provider 切时间窗，估算每个窗口覆盖的 document/chunk 数。
3. 计算 `calls_per_chunk = provider_calls / chunk_count`。
4. 若 calls/chunk 在 1~3 左右，优先认为是正常重试放大；若远高于 3，再查 JSON parse loop、429 backoff、输出过长二分、重复提交。
5. 同时查 async_operations 是否存在真实重复提交：重复 `retain` 子任务才算重复，`batch_retain` parent rows 且 `task_payload IS NULL` 不应算成可执行重复任务。

## 官方依据与配置杠杆

- 官方 scaling 文档明确：retain cost scales with input；内容先 chunk，LLM fact extraction 是 dominant cost；一 chunk 一次 extraction。
- 官方配置默认 `HINDSIGHT_API_RETAIN_CHUNK_SIZE=3000`；更大 chunk 会减少 calls，但可能损失上下文/增大输出失败风险。
- `HINDSIGHT_API_RETAIN_EXTRACTION_MODE=chunks` 是 zero LLM cost：只存 chunk + embedding，无实体/时间/结构化事实，适合 raw evidence side bank，不宜直接替代 production facts。
- `HINDSIGHT_API_RETAIN_BATCH_ENABLED=true` 只在兼容 provider Batch API 时降成本（官方明确 OpenAI/Groq）；不要假设任意 OpenAI-compatible endpoint（opencode/Bailian/MiniMax）支持 `/files`/`/batches`。
- 本地 0.5.x 源码存在多层 retry：`_extract_chunk_with_retry`（3 次）× `_extract_facts_from_chunk` 外层 retry × provider `llm_config.call(max_retries=...)` 内层 retry；所以 calls/chunk > 1 是正常的。

## 止血动作

- 发现调用量异常先停 Hindsight 容器或暂停 worker，避免继续烧 paid provider。
- 立刻 kill stale wait/audit watcher，避免后续恢复容器时自动继续导入或切回 normal-local。
- 不要只降并发；降调用量应从四个方向一起看：
  1. 输入清洗：先统计 role/block 字符占比，尤其 tool output；raw tool dump 往往是主因。
  2. chunk size / chunk 数：必要时提高到 8k/16k，但配合 custom prompt 限制事实数。
  3. retry 次数和 backoff：`HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES=1` 可减少失败放大；不能设 0（部分代码路径会无尝试）。
  4. batch / window 切分：Batch API 只对确认支持的 provider 开启。
  5. 预过滤或更稳的结构化输出解析：drop/cap tool output、skip process chatter、关闭 causal links、使用 custom extraction 限制 max facts/chunk。

## 2026-05-09 production session 经验

- 83 条 session manifest 共 28,050,435 chars、3,548 chunks@8000；其中 tool blocks 26,737,904 chars（95.6%）。
- 用户最终偏好：paid retain manifest 只保留用户输入 `[user]` 和 Hermes 助手输出 `[assistant]`；排除 tool/command/search/thinking/procedural traces，不把原始工具 dump 交给 Hindsight retain。
- dry-run：user+assistant-only v2 降到 1,152,534 chars、187 chunks@8000 / 142 chunks@12000 / 124 chunks@16000；5 个最长文档 smoke 为 222,853 chars、30 chunks@8000 / 20 chunks@12000 / 15 chunks@16000。
- smoke 结果（test DB + bank，不动 production）：5 docs 全完成，67 memory_units，0 failed，0 pending，observations=0；日志无 429，出现少量 APIConnectionError retry，质量明显优于 raw tool dump。
- chunk size 取舍：8000 细节召回最高但调用多；16000 调用最省但长 chunk 容易漏细节；对清洗后的 user+assistant-only production retain，推荐 12000 作为质量/成本折中（比 16000 仅多约 18 chunks，比 8000 少 45 chunks）。
- 推荐路线：不要继续 raw manifest；先用 user+assistant-only cleaned manifest 小批 smoke，记录 provider calls/chunk 与 facts density，再决定全量 replace/re-retain。

## 给用户汇报时的表述

应直接纠正口径：

> 这不是“83 条数据打了 1500 次”的问题，而是“83 条超长 session 被切成几千个 chunk，每个 chunk 还会重试和拆分”的问题。
