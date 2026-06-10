# Hindsight session/json production hardening

Use this reference after a session/json native retain trial finishes but quality gates fail, especially when `docs_without_units` or query-level recall contamination appear.

## Goal

Build a generic, long-term stable production pipeline. Do not patch for one run, one project name, or one recall query. The pipeline should be reproducible, auditable, source-preserving, and rollback-safe.

## Non-goals

- Do not write hard-coded fixes for queries like `patent`, `openclaw`, `user_pref`, or `cch`.
- Do not write project-specific rules for Egomotion4D / Hindsight / OpenClaw in the core selector.
- Do not merge unrelated sessions just to reduce paid calls.
- Do not run consolidation when retain facts gate fails.
- Do not directly mutate production; destructive actions require discard/quarantine + proposal + explicit confirmation.

## Stable architecture

1. **Raw session inventory**
   - JSON sessions remain the source of truth.
   - Manifest records source path, session id, time, size, hashes, message counts, cleaning stats, filter versions, and provenance.

2. **Generic structural scoring**
   - Score by content structure, not by project keywords.
   - Suggested dimensions: semantic density, transcript noise ratio, retainability risk, provenance quality.
   - Suggested generic classes: user_preference, durable_decision, project_state, experiment_result, tool_lesson, error_root_cause, environment_fact, open_question, low_signal.

3. **Raw evidence bank**
   - Optional but preferred for cost control and provenance recall.
   - Use a separate raw bank with Hindsight `retain_extraction_mode="chunks"` so raw evidence indexing costs 0 LLM calls.
   - Keep observations/consolidation disabled on raw banks.
   - Never mix raw chunk facts into the clean production fact bank.

4. **Clean retain selection**
   - Prefer whole session for context preservation.
   - Use deterministic contiguous windows only for overlong/noisy/multi-scope sessions.
   - Record parent session id, span boundaries, part index, before/after context, and hashes.
   - Routes should be generic: production_whole_session, production_windowed, raw_only, manual_review, skip.

5. **Paid native retain**
   - Use Hindsight native retain API and fresh bank/run submit-state.
   - Keep `enable_observations=false` during retain.
   - Run paid provider in background/long-timeout mode and always restore normal-local.

6. **Post-retain quality gate**
   - Required checks: pending/processing/failed=0; retain child payload_null=0; `docs_without_units` acceptable and classified; fact density acceptable; no secret leaks; no broad/system tag pollution; no missing lineage; recall smoke precision acceptable.
   - Completed retain operations and `failed_operations=0` are not enough. Hindsight can complete retain while many documents produce no facts.

7. **Generic recall benchmark**
   - Maintain three query classes:
     - stable seed queries for regression;
     - manifest-derived queries generated from tags/classes/entities/time;
     - adversarial negative queries to detect dominant-topic hijacking.
   - Metrics: top-k precision, MRR/first relevant rank, dominant tag ratio, off-topic rate, false positive rate, per-class coverage.

8. **Zero-unit diagnosis loop**
   - Zero-unit docs are a first-class quality signal.
   - Classify as true low-signal, noisy transcript, overlong/multi-scope, extraction too strict, cleaning lost context, or prompt/model failure.
   - Retry only high-value zero-unit samples in a temporary bank, comparing whole-session vs windowed vs custom mission.
   - Treat temp-bank recovery as diagnostic evidence, not production data: do not merge recovered facts directly into production.
   - Promote only generalizable rules to manifest/selector/audit.
   - See `references/hindsight-zero-unit-temp-bank-retry.md` for the 2026-05-08 Phase B pattern where 13/20 high-value zero-unit docs recovered facts on temp-bank re-retain.

9. **Consolidation**
   - Run only after facts gate and recall gate pass.
   - Default quality-first window: 1 job / 50 facts.
   - Re-audit after consolidation.

10. **Repair / quarantine**
    - Bad recall, bad observation, wiki candidate anomaly, user-reported wrong claim, and benchmark regression all enter the same repair loop.
    - Flow: case intake -> lineage trace -> source facts/docs/raw span -> repair proposal -> discard snapshot -> explicit confirmation -> mutate -> rerun gate.

## Recommended implementation order

1. Add/extend zero-unit report with structural features and generic value classes.
   - Current helper: `~/.hermes/scripts/hindsight_session_quality_hardening.py`.
   - Example read-only command:
     ```bash
     python3 ~/.hermes/scripts/hindsight_session_quality_hardening.py \
       --bank hermes \
       --manifest <week-production.jsonl> \
       --audit-json <retain-audit-parsed.json> \
       --output-dir <run-root> \
       --stem phase-a-session-quality-hardening
     ```
   - It writes JSON/Markdown reports, classifies zero-unit docs by generic structure, scores recall-smoke rows, and emits reviewable manifest-derived benchmark candidates. It does not mutate Hindsight.
2. Add recall benchmark metrics beyond sample rows: top-k precision, dominant tag ratio, off-topic rate, per-class coverage.
3. Optionally add an LLM value scorer as a review-backlog sidecar/rescue-router, not as a hard pre-retain filter. Questionable/noisy/low-score samples should first land in a recoverable backlog (`review_backlog.jsonl` or optional review/discard bank) with `content_sha256`, source path, source `event_date`, deterministic anomaly labels, and current retain outcome. Current helpers: `~/.hermes/scripts/hindsight_review_backlog.py` builds this index read-only from a manifest, hardening JSON, retry fact-quality gates, and optional PostgreSQL per-document unit counts; `~/.hermes/scripts/hindsight_review_backlog_cleanup.py` keeps the active/hot backlog on a 3-calendar-month time window by default and archives older records to cold JSONL; `~/.hermes/scripts/hindsight_review_backlog_sampler.py` selects a content-free representative scorer sample from the active backlog; `~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py` turns that sample into bounded scorer packages where one package equals one future LLM call and, only with `--execute-score --confirm-score score-review-backlog`, writes score sidecar JSONL. Default scorer cadence/cost guard is weekly 10 packages/calls (`--max-batches 10`, `--batch-size 5`), configurable up to larger pilots such as 50 calls, with overflow deferred rather than dropped. The backlog omits content by default and rehydrates only locally for hashing/anomaly detection. Use MiniMax/LLM scoring later to re-evaluate individual records or topic clusters for value/density/durability, anomalies, suggested spans, and rescue routes (`wait`, `raw_only`, `whole_session`, `windowed`, `manual_review`, `repair_note_candidate`, `cluster_revisit`). Do not feed scorer summaries back into Hindsight as retain content; do not write free-form scorer labels into semantic tags/LLM-visible metadata; do not delete evidence solely because of a low score. Cache by content hash and compare scorer decisions against actual retain metrics and later repair yield before using it for production routing.
4. Run a temporary-bank experiment on 10-20 high-value zero-unit docs.
5. Promote only stable, generic improvements into manifest builder/selector.
6. Rerun the one-week trial; only if facts + recall gates pass, run bounded native consolidation.
7. Scale from one week -> two weeks -> one month -> all history, preserving budget/quality/recall/lineage reports each time.
