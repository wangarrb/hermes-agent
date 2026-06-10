# Hindsight user+assistant-only retain smoke

## Purpose

This reference captures the user-preferred cleaning strategy for paid Hindsight retain when the source corpus is a raw Hermes session transcript:

- keep only `[user]` and `[assistant]` role blocks
- drop `[tool]` / command / search / process / thinking traces
- redact credential-like strings
- validate on a small smoke bank before full production retain

## Why this exists

A 2026-05-09 production week manifest showed that tool output dominated raw size and drove retain call amplification. The user explicitly preferred a corpus that uses only their input and Hermes assistant output, not process material.

The resulting cleaned manifest was much smaller and produced a safer, lower-noise smoke run.

## Recommended cleaning policy

1. Parse each record into role blocks.
2. Keep only `[user]` and `[assistant]` blocks.
3. Remove all other role blocks entirely.
4. Redact any credential-like strings (`api_key`, `token`, `secret`, `password`, `sk-...`, `Bearer ...`).
5. Recompute content length and chunk estimates.
6. Use the cleaned manifest for smoke first; do not reuse raw tool-dump manifests for paid production retain.

## Chunk-size guidance

For cleaned user+assistant-only manifests:

- `chunk_size=8000`: quality-first production import. In the 2026-05-09 83-record week this was 187 chunks, max doc 6 chunks.
- `chunk_size=12000`: balanced default when cost pressure is higher; same week 142 chunks, max doc 4 chunks.
- `chunk_size=16000`: lowest cost, but higher risk of missing details when custom instructions cap facts/chunk; same week 124 chunks.
- `4000`/`6000`: usually not worth it after cleaning; they increase calls substantially and can create duplicate facts/cross-chunk fragmentation.

## Smoke validation pattern

- Select the 5 longest cleaned documents first.
- Use a disposable test DB/bank, not production.
- Keep observations disabled, causal links disabled, retries low, and provider concurrency around 2.
- Verify:
  - completed operations
  - failed/pending operations
  - memory_units count and facts/doc density
  - recall smoke quality
  - tags are narrow and not polluted by memory-pipeline tags
  - absence of 429, output-too-long loops, or repeated APIConnectionError retries
- Stop the smoke container or restore normal production config afterward so port 8888 does not accidentally point at the smoke DB.

2026-05-09 smoke benchmark: 5 longest docs, chunk_size=16000, concurrency=2, opencode-go deepseek-v4-flash; 5 retain + 5 batch_retain completed; failed 0, pending 0; 67 memory_units, 2,253 links, observations 0; logs had 0 HTTP 429, 0 OutputTooLong, 2 APIConnectionError retries.

## Production sequence

1. Stop Hindsight container and stale watcher if needed.
2. Snapshot current production DB before reset/replace.
3. Reset or replace only after backup and user approval.
4. Run migrations manually if startup migrations are disabled.
5. Start Hindsight with explicit retain env: custom extraction, chosen chunk size, `HINDSIGHT_API_RETAIN_EXTRACT_CAUSAL_LINKS=false`, retain/global LLM retries 1, observations disabled.
6. Submit the cleaned manifest once with a fresh submit-state.
7. If provider quota is hit, switch provider without resubmitting the manifest, but preserve the custom retain env/config. Plain wrapper `import-llm` may reset extraction mode to `concise`; verify bank config immediately after switch.
8. After queue drain, run quality audit + recall smoke, then restore normal-local.

## User-specific note

For this user, the clean-manifest preference is stronger than the older “keep tool摘要” compromise. If there is a choice between:

- preserving tool summaries, or
- strictly keeping only user+assistant content

prefer strict user+assistant-only cleaning unless the user explicitly asks to keep tool evidence.

## Related files

- `references/retain-call-amplification.md`
- `references/hindsight-session-retain-pitfalls.md`
- `references/hindsight-session-json-cost-control-and-curated-smoke.md`
