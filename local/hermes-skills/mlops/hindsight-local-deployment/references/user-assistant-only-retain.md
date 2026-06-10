# User+assistant-only Hindsight session retain

Use this when importing Hermes session history into Hindsight with paid native retain.

## Policy

The production paid-retain input should contain only:

- `[user]` blocks: the user's inputs.
- `[assistant]` blocks: Hermes assistant outputs.

Exclude before submission:

- `[tool]` outputs and raw command/search/file dumps.
- Thinking/analysis/commentary/procedural traces.
- Logs, file listings, raw API responses, and temporary progress chatter.
- Credential-like strings; redact or route to manual review.

Rationale: Hindsight retain cost is chunk-based. In the 2026-05-09 week import, raw tool output was ~95.6% of 28M chars and inflated the job to ~3,548 chunks@8000. User+assistant-only v2 reduced the same week to 1,152,534 chars / 187 chunks@8000.

## Recommended sizing

For cleaned user+assistant-only manifests:

- `chunk_size=8000`: quality-first production import. For the 83-record week: 187 chunks, max doc 6 chunks.
- `chunk_size=12000`: balanced default if cost pressure is higher. Same week: 142 chunks, max doc 4 chunks.
- `chunk_size=16000`: lowest cost, but higher risk of missing details when custom instructions cap facts/chunk. Same week: 124 chunks.
- `4000`/`6000`: usually not worth it after cleaning; they increase calls substantially and can create duplicate facts/cross-chunk fragmentation.

## Smoke first

Before mutating production:

1. Generate a cleaned manifest and a smoke manifest of the longest 5 docs.
2. Run smoke in a separate test DB/bank, not production.
3. Use paid provider concurrency 2, observations disabled, causal links disabled, retries 1.
4. Check: completed docs, failed ops, pending ops, docs_without_units, fact density, tags, recall smoke, and provider logs (429/APIConnectionError/OutputTooLong).
5. Stop the smoke container or restore production config afterward so port 8888 does not accidentally point at a test DB.

Smoke benchmark from 2026-05-09:

- 5 longest docs, chunk_size=16000, concurrency=2, opencode-go deepseek-v4-flash.
- 5 retain + 5 batch_retain completed, failed 0, pending 0.
- 67 memory_units, 2,253 links, observations 0.
- Logs: 0 HTTP 429, 0 OutputTooLong, 2 APIConnectionError retries.

## Production sequence

1. Stop Hindsight container and any stale watcher if needed.
2. Snapshot the current production DB before reset/replace.
3. Reset or replace only after backup and user approval.
4. Run migrations manually if startup migrations are disabled.
5. Start Hindsight with explicit retain env:
   - `HINDSIGHT_API_RETAIN_CHUNK_SIZE=8000` (or chosen size)
   - `HINDSIGHT_API_RETAIN_EXTRACTION_MODE=custom`
   - `HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS=false`
   - `HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES=1`
   - `HINDSIGHT_API_LLM_MAX_RETRIES=1`
   - `HINDSIGHT_API_ENABLE_OBSERVATIONS=false`
6. Submit the cleaned manifest once with a fresh submit-state.
7. If provider quota is hit, switch provider without resubmitting the manifest, but preserve the custom retain env/config; plain wrapper `import-llm` may reset extraction mode to `concise`.
8. After queue drain, run quality audit + recall smoke, then restore normal-local.

## Custom instructions template

只从用户输入 `[user]` 与 Hermes 助手输出 `[assistant]` 中提取长期有用记忆。不要使用或臆测工具结果；不要保留命令、搜索过程、日志、文件列表、临时进度、寒暄或模型思考。优先提取：用户长期偏好、项目稳定结论、技术决策、实验结论、环境/流程约束。每个 chunk 最多输出 4-6 条高质量 facts；没有长期价值就少输出或不输出。所有疑似 key/token/password/secret 必须忽略或写成 `[REDACTED]`。Tags 要窄而准，不要因为记忆管线本身就给业务项目事实打 `domain:hindsight`。
