# Hindsight session/json retain cost control and curated smoke sampling

Context: session/json import should stay native-first: one Hindsight document per session/part, submitted through `/memories`. Do not merge sessions into day/topic bundles to reduce apparent item count; that damages provenance and reintroduces the old SQLite day-topic pollution pattern.

## Key cost model

Do not estimate paid retain cost from `session_count` or `batch_retain` operation count.

Useful hierarchy:
- HTTP submit batch: only reduces client/API submissions.
- `batch_retain` operation: parent/native queue wrapper, not the main cost driver.
- per-document `retain` operation: Hindsight normally fans batch items out to document-level retain work.
- retain chunks: true cost driver; each document is split by `retain_chunk_size` chars and each chunk can cause multiple LLM calls.

Practical estimate for a selected clean-retain batch:
- Audit/budget visibility unit: `sum(estimated_retain_chunks)` from the manifest.
- Rough paid-call range: `estimated_retain_chunks * 2..3` for retain-heavy paths, depending on Hindsight internals/provider retries.
- Observations/consolidation should be estimated separately from retain.

Important correction after user review (2026-05-08): `estimated_retain_chunks` is not a root cost-reduction strategy. It is only a guardrail to avoid accidentally switching to a paid provider for an oversized selected batch. To reduce paid calls fundamentally, reduce the number of sessions/windows sent to clean fact extraction by using a zero-LLM raw-chunks index plus topic/coverage selection, or accept an explicit quality tradeoff such as `retain_extraction_mode="chunks"` for raw evidence only.

Example from 2026-05-08 smoke:
- curated 15 production records = 419,981 chars = 61 estimated retain chunks.
- Even though only 15 sessions and 3 submit batches, rough retain calls could be 122-183.
- full manifest production subset = 393 records = 696 estimated chunks, roughly 1,392-2,088 retain LLM calls.
- manual_review subset = 1,363 records = 4,136 estimated chunks; never send this automatically to paid retain.

## Guardrails to add/use

0. Root cost reduction: do not send broad production history directly to paid clean retain.
   - First build/use a separate chunks-mode raw index or source-manifest topic discovery step.
   - Select source sessions/windows by topic coverage, recency, source value, and risk gates.
   - Then use chunk budget only on the selected clean-retain batch.

1. Prefer `--max-estimated-chunks` or equivalent preflight budget over `--limit N`.
   - `--limit 15` can still be expensive if sessions are long.
   - Fail closed before provider switch if chunk budget is exceeded.

2. Split phases:
   - Phase A: production-only retain, observations disabled.
   - Phase B: audit facts/tag/scope/lineage.
   - Phase C: run native consolidation/observations in bounded windows, e.g. 50 facts/job.

3. Keep manual_review isolated.
   - Credential/secret/token/API-key-triggered sessions should stay review-only until explicitly cleared.
   - Do not use paid provider on manual_review by default.

4. Do not solve cost by merging unrelated sessions.
   - Merging reduces document count but breaks session boundaries, provenance, and scope purity.
   - It can harm native consolidation more than it saves cost.

5. Retain chunk size tuning requires A/B.
   - Increasing `retain_chunk_size` from 8k chars to 16k/24k may cut chunks, but can lose numeric/config details.
   - Test on the same smoke set and compare fact count, numeric preservation, recall smoke, contamination, and elapsed/call estimate.

## Curated smoke vs balanced sampling

`curated` should mean a small, non-mutating smoke manifest for pipeline validation, not a replacement for production import.

Current selector behavior is high-signal smoke, not statistically balanced sampling:
- production-only
- weighted toward high-value tags such as Hindsight/Egomotion4D/OpenClaw
- one per session root to avoid selecting many split parts from the same session

Use cases:
- Validate native retain wrapper, provider restore, queue drain, submit_state incrementality, audit scripts.
- Expose long-session/timeout/tag/scope issues before broad import.

Do not use curated smoke as production coverage; it will underrepresent normal sessions and some topics.

Balanced smoke, when needed, should stratify by:
- time: recent/history
- length: short/mid/long/very long
- topic/domain/project tags
- action: production and selected manual_review boundary samples, but not auto-retain manual_review
- structure: tool-heavy, compress/resume, multi-part sessions
- risk: secret-like/manual-review cases retained only as local audit examples

## User-facing explanation pattern

When asked whether native fan-out “multiplies calls”, answer precisely:
- Fan-out from batch to per-document retain is normal and preserves session boundaries.
- The real multiplier is chunking inside each document plus internal LLM passes/retries.
- Therefore cost control should budget by estimated chunks, not by batch count or session count.
