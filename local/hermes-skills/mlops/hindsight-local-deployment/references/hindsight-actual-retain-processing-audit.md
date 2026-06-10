# Hindsight actual retain processing audit: raw session transcript vs retain-ready evidence

Use this reference when Hindsight session/json retain quality is poor despite operations completing, facts increasing, or temp banks showing no failed operations.

## Verified processing chain

Hindsight `batch_retain` does not understand Hermes session semantics. It treats each item as a plain text document:

1. API/worker creates async `batch_retain` task.
2. `MemoryEngine._handle_batch_retain()` reads `bank_id`, `contents`, `document_tags`, `operation_id`, `strategy`.
3. It calls `retain_batch_async()` / retain orchestrator.
4. Hindsight writes the original item text to `documents.original_text`.
5. It chunks text into `chunks.chunk_text` using retain chunk config.
6. LLM fact extraction runs over chunks/content.
7. Extracted facts are written to `memory_units.text` with `fact_type` such as `experience` or `world`.
8. If observations/consolidation are enabled, consolidation later reads `memory_units`, not raw documents, and writes `fact_type='observation'` rows back into `memory_units` with `source_memory_ids` lineage.

Implication: consolidation cannot recover facts that retain never extracted from the original transcript. If low-level facts are noisy or missing, native observations will amplify that problem.

## Hermes session/json runner item shape

The local session retain runner converts manifest records roughly into:

```python
{
    "content": rec.get("content") or "",
    "document_id": rec.get("document_id"),
    "context": rec.get("context") or "hermes_session",
    "event_date": rec.get("event_date") or (rec.get("metadata") or {}).get("event_date") or (rec.get("metadata") or {}).get("started_at"),
    "metadata": normalize_metadata_for_hindsight(rec.get("metadata") or {}),
    "tags": rec.get("tags") or [],
    "observation_scopes": rec.get("observation_scopes") or [],
    "update_mode": rec.get("update_mode") or "replace",
}
```

`content_omitted=true` in a manifest does not necessarily mean empty input. The runner may rehydrate the content from a JSON path before submitting; verify `documents.original_text` in DB before blaming the manifest.

## Diagnostic pattern from 2026-05-08 week trial C2

A paid all-routes rerun completed and produced more facts, but quality remained poor:

- submitted documents: 227
- chunks: 211
- memory_units: 1544
- experience facts: 1409
- world facts: 135
- observations: 0
- parent coverage improved only from 83/188 to 93/188
- parent zero-unit still 95/188
- recall macro precision about 0.40
- failed operations: 0

Actual route mix:

- `keep_clean_session`: 88
- `production_windowed`: 94
- `retry_less_aggressive_cleaning`: 22
- `retry_custom_mission`: 23

Observed input shape included raw Hermes transcript features:

- context compaction handoff text
- repeated short user commands such as “继续”
- tool/status/plan summaries
- provider/queue/log/debug noise
- project decisions mixed with temporary execution state
- custom mission preambles prepended to the transcript
- windowed fragments that lost full session context

Some clean full-session records generated useful technical facts, while custom-mission, relaxed-cleaning, and windowed samples could still produce zero units despite non-empty long text.

## Interpretation rules

Do not equate these signals:

- `completed` operation != quality passed
- `failed_operations=0` != every document produced useful memory units
- more facts != better recall precision or parent coverage
- `task_payload IS NULL` on parent `batch_retain` != child retain failure by itself
- zero-unit document != necessarily empty input

When quality is bad, first verify actual data before another paid run:

1. Inspect Hindsight source/API path for what operation actually processes.
2. Compare manifest records with DB `documents.original_text`.
3. Count documents/chunks/memory_units by bank and fact type.
4. Parent-level coverage: how many source sessions have at least one unit.
5. Query-level recall smoke and top-k contamination.
6. Topic/tag co-occurrence drift.
7. Sample zero-unit parents by route and text shape.

## Design lesson

Raw Hermes session transcript is not a good default clean retain document. It is high-provenance evidence, but it should usually be converted into a retain-ready evidence note before paid/native Hindsight retain.

A retain-ready note should keep:

- work goal / research question
- durable decision
- root cause and correction
- stable result / metric / configuration
- project evolution and turning point
- user preference or workflow rule
- explicit provenance back to source session/span

It should filter or demote:

- tool-call流水 and log tails
- progress-only status
- short continuation commands
- context handoff template text
- provider/queue/debug noise unless the debugging fact itself is durable
- credential-like material, which must go to manual review

## Evidence-note pilot result: shape matters

2026-05-09 small local pilot on source document `hermes-session::20260501_124934_18c742::phase-c2-window-01` showed that “evidence note” is not enough by itself; the exact input shape affects extraction.

Setup:
- temp bank: `hermes_tmp_evidence_note_pilot_20260509`
- source C2 route: `production_windowed`
- source C2 unit count: 0
- runtime: normal-local Ollama, observations disabled, `retain_extraction_mode=concise`, `retain_chunk_size=8000`

Variants:
- same raw transcript re-retained in the pilot bank: 7 units
- Markdown-style evidence note with metadata/header sections: 0 units
- conversation-wrapped fact list (`User: 请把下面这些事实作为长期记忆保留` + `Assistant: 可保留的长期事实如下` + explicit `Fact N:` lines): 12 units and recall top-k was dominated by the evidence-note-dialog facts

Interpretation:
- Hindsight/LLM extraction may handle conversation-like fact lists better than pure Markdown notes with source metadata headers.
- Do not assume a “structured note” works because it looks clean to humans; retain each candidate note shape in a temp bank and verify `memory_units` + recall before scaling.
- The local extractor still introduced formatting artifacts in some facts (truncated paths, odd `When` strings), so shape success must be paired with quality/precision gates, not just unit count.

Practical pilot template:
1. Use a temp bank with observations disabled.
2. Submit raw baseline + one or more retain-ready variants under different document_ids.
3. Prefer a conversation-wrapped explicit fact list for first pass.
4. Compare units per variant, inspect sample facts, and run a query-level recall smoke.
5. Only promote a shape after it improves coverage without adding formatting/path/timestamp contamination.

## Fact-text artifact root cause

If extracted facts contain bad fragments such as `When: 2026- paths`, `When: ... JSON`, generic `Involving: 用户`, truncated file paths, or incomplete code identifiers, inspect retain extraction before blaming recall.

Verified in 2026-05-09 pilot:
- Hindsight's retain schema requires/favors `what`, `when`, `where`, `who`, `why`, and `fact_type`.
- The concise prompt tells the LLM to use the input `Event Date` as temporal reference.
- Orchestrator defaults missing `event_date` to current `utcnow()`.
- Parser builds persisted fact text as `what | When: {when} | Involving: {who} | {why}`.
- It drops only exact empty/`N/A` fields; malformed non-empty strings are accepted.
- Storage only sanitizes and inserts; it does not validate semantic date/person/path quality.

Pilot evidence:
- dialog-wrapped fact list gave better coverage (12 units) but still had formatting artifacts.
- Adding `retain_mission` that explicitly said not to invent dates/users did not fix local Ollama extraction: 12/12 facts still had `When`, 12/12 had `Involving: 用户`, and truncated identifiers remained.
- Sending `event_date: null` also did not solve it in practice.

Interpretation:
- The artifact is introduced during LLM fact extraction / field filling, then persisted by Hindsight's field concatenation.
- This is a low-level retain problem, not recall or consolidation.
- Current local `qwen3.5:9b-local` may produce valid JSON but semantically corrupt `when/who/why` fields for technical evidence notes.

Handling:
1. Always submit source conversation time as top-level `event_date` for retain/retry/repair documents. In the Hermes session/json pipeline this is now generated from `session_start` and forwarded by `hindsight_session_retain_runner.py`; do not rely on Hindsight's missing-date default because it becomes the current run date.
2. Add a post-retain quality gate for temp banks before promotion: malformed `When`, generic `Involving: 用户` on technical facts, truncated paths/identifiers, odd header fragments (`JSON`, `paths`, `provided`, `chunk`) should quarantine or fail.
3. For clean technical fact banks, consider a wrapper/Hindsight patch that drops or stores `when/who/why` as metadata instead of concatenating them into `memory_units.text`, or sanitizes invalid fields before storage.
4. `retain_mission` alone is not a sufficient control for this artifact class.
5. Stronger paid retain may reduce artifacts, but must still pass the same fact-quality gate.

## Review backlog control and cleanup

A recoverable backlog is useful, but it must not grow without bound. Prefer a time-window policy over a hard count quota:

- generate the full backlog read-only from the manifest + hardening / retry evidence + optional DB unit counts;
- keep `event_date`, `content_sha256`, `source path`, current retain outcome, and deterministic anomaly labels in the active/cold indexes;
- default active/hot backlog policy is the last 3 calendar months (`--retention-months 3`);
- run cleanup monthly or on another periodic cadence so older items move to cold archive and are covered by monthly review;
- archive older records to a cold JSONL instead of deleting them;
- keep `--max-records` / bucket quotas only as an optional safety valve for abnormal backlog explosions, not as the primary policy;
- use the sampler script to produce a content-free scorer subset from the active 3-month backlog; do not feed scorer output back into Hindsight content/tags;
- LLM scorer data should be planned as batches/packages: **one scorer batch = one future LLM call**. Default cadence is weekly and default cap is 10 packages/calls per week; keep `--max-llm-calls` / `--max-batches` configurable (e.g. 50 for a larger pilot), and defer overflow rows rather than dropping them.
- Actual LLM scoring remains fail-closed: without `--execute-score --confirm-score score-review-backlog`, `hindsight_review_backlog_llm_scorer.py` only plans/dry-runs and writes no scores. When executed, it writes sidecar JSONL only (`hindsight-review-backlog-llm-score-v1`), skips credential-like records before LLM, and forces `hindsight_submit_allowed=false` / `production_mutation_allowed=false`.

Current helpers:
- `~/.hermes/scripts/hindsight_review_backlog.py`
- `~/.hermes/scripts/hindsight_review_backlog_sampler.py`
- `~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py` (planner + explicit-confirm sidecar scorer; default `--batch-size 5 --max-batches 10 --cadence weekly`, no LLM/Hindsight calls unless explicitly executed)
- `~/.hermes/scripts/hindsight_review_backlog_cleanup.py`

Cleanup result from the 2026-05-09 pilot:
- 3-month time-window cleanup produced active backlog 188, archive 0 because the sample is only one week old;
- cutoff_event_date was `2026-02-09T00:00:00+00:00` for the one-week run;
- the older `active120/archive-overflow` files are retained only as count-cap experiment artifacts, not the default policy;
- active set remained event-date complete and content-free by default.

## Hindsight-compatible preprocessing: do not over-summarize at retain ingress

2026-05-09 correction: do not turn project decisions, experiment results, technical conclusions, paths/configs/metrics, or root causes into a high-level summary before retain. Those are often high-dimensional project-evolution information. Retain ingress should preserve low-level evidence and atomic facts; Hindsight native consolidation/observations/reflect should do the higher-level synthesis later, after the low-level facts pass quality gates.

Preferred upstream shaping is therefore **low-level Hindsight-compatible cleaning**, not high-level evidence-note summarization. Keep it lightweight by default: no extra LLM summarizer, no global project-decision/root-cause synthesis, and no large custom prompt preamble in production input.

- keep original conversation/document semantics and provenance
- remove credential-like material, explicit reasoning/thinking/chain-of-thought fields or blocks, tool-log tails, repeated handoff boilerplate, pure queue/status noise, and short no-signal continuations
- preserve natural `User:` / `Assistant:` conversation slices when possible
- split into chunks that Hindsight itself would see cleanly
- avoid adding large Markdown metadata/header blocks into the text body
- keep durable atomic claims explicit and local, but do not invent a global decision/root-cause narrative that the source did not state directly
- put metadata in metadata fields when possible; do not mix it into searchable text unless it is itself evidence

The earlier “conversation-wrapped Fact N list” result is useful as an extraction-shape signal, not as permission to pre-compress the session into high-level conclusions. Use it only for atomic source-backed facts or temp-bank experiments.

## Native preprocess simulation before paid reruns

When investigating truncation or field pollution, simulate Hindsight's own preprocessing before another paid/native run:

1. In the Hindsight container, run from `/app/api` with `/app/api/.venv/bin/python`; default `python` may not import `hindsight_api`.
2. Import and call Hindsight's own `chunk_text`, prompt/schema builder, and user-message builder for the exact candidate `content`.
3. Record `input_chars`, `chunk_count`, per-chunk chars, prompt/user_message chars, and whether critical needles are present in the generated `user_message`.
4. Only then call the same provider/model with Hindsight's schema and inspect the parsed extraction output.
5. Attribute truncation based on where the needle first disappears or corrupts:
   - missing in `content` → local cleaning/manifest bug
   - present in `content`, missing in `user_message` → Hindsight chunk/prompt/header issue
   - present in `user_message`, corrupted only in parsed facts → LLM extraction/output-budget/schema-pressure issue
   - clean parsed facts, corrupted in DB → Hindsight formatting/storage issue

Verified local pilot numbers for one dialog fact-list sample:

- input: 1883 chars
- Hindsight `chunk_text`: 1 chunk, 1883 chars
- prompt: about 6065 chars
- user_message: about 2120 chars
- critical needles remained present in both content and user_message, including long file paths, `can_correction_edge_policy="gated"`, `ATE_m=1.011961`, and `direction_agreement_median_cos≈0.9988`
- local Ollama extraction output already contained generic `who: 用户`, timestamp `when`, and a malformed truncated `when: "202 "`

Conclusion for that sample: the truncation/dirty `When`/`Involving` artifact was not caused by Hindsight chunking or user-message preprocessing; it was introduced by LLM extraction and then persisted by Hindsight's fact-text assembly.

## Clean-v2 native-like sample result

2026-05-09 clean-v2 sample on six high-value zero-unit/noisy one-week documents:

- `deterministic-clean-v2` strips explicit reasoning/thinking/COT fields and `<think>...</think>` blocks.
- Same one-week production source set dry-run stayed stable: 188 old records → 188 v2 records, all production, changed common document count 0 for that week subset.
- Direct Hindsight preprocessing on six samples produced natural `User:` / `Assistant:` inputs with no context compaction, no reasoning literals, no think blocks, and no custom mission preamble.
- Local retain into temp bank `hermes_tmp_weektrial_clean_v2_native_sample_20260509` produced 6 documents, 17 memory_units, 0 observations, but still 3/6 docs had zero units.
- Artifact gate for this sample found 0 reasoning-like units, 0 suspicious `When`, and 0 generic `Involving: 用户` units.

Interpretation: clean-v2/native-like preprocessing is safer and cleaner, but it does not by itself solve local extractor zero-unit behavior. Do not scale to production or paid solely on this change; keep using temp-bank pilots and coverage/recall gates.

## Clean-v2 window pilot: metadata/narrator hygiene matters

Follow-up pilot on the three zero-unit parents from the clean-v2 native sample:

- Variants tested: `first-useful-window` and `local-semantic-context-window`, with no high-level summarization or custom mission.
- `first-useful-window` recovered all 3 previously zero-unit parent sessions.
- `local-semantic-context-window` produced 0 units for all 3 parents in the local extractor.
- Passing verbose manifest metadata caused fact contamination because Hindsight includes metadata in the extraction user message.
- Omitting metadata still did not fully prevent leakage if the temp bank profile name equals the long temp bank ID; Hindsight passes bank profile `name` as `Narrator`.
- A temp bank with profile name set to `Hermes` and empty metadata removed temp-bank/prompt leaks in the coarse gate, but local qwen still emitted malformed temporal fields in some facts.

Operational guidance:
1. Keep retain text natural and lightweight, but also minimize metadata passed to Hindsight extraction. If provenance metadata is needed, prefer storing it outside the LLM-visible payload or in a companion manifest joined by `document_id`.
2. For temp banks, set profile `name` to a neutral short agent name (e.g. `Hermes`) so the `Narrator` section does not leak a long temp bank ID into facts.
3. Treat `first-useful-window` as a candidate lightweight zero-unit recovery route, but do not promote it until a larger temp sample passes fact-quality and recall gates.
4. `local-semantic-context-window` selected by simple keyword scoring can over-focus on noisy tail/status/skill-update spans; do not use it as production default.

Expanded sample result (2026-05-09): a 24-parent `first-useful-window` temp run with empty metadata and neutral bank profile name `Hermes` produced only 33 memory units and recovered 6/24 parents (25%); 18/24 docs stayed zero-unit. Artifact hygiene improved (no temp-bank/prompt/reasoning/context leak; only 3 generic `Involving: 用户` units), but coverage was too low and recall smoke was poor (`macro_precision_at_k=0.0625`). Do not treat the earlier 3/3 result as generalizable. Keep `first-useful-window` as one route in a controlled experiment, not as the default production rewrite.

Contrast follow-up (2026-05-09): sample 8 of the 18 failed parents and retain two variants per parent (`first-useful-window` vs `whole-session`) in a temp bank with empty metadata and neutral profile name. Result: `first-useful-window` recovered 0/8 parents, while `whole-session` recovered 6/8 parents with 38 units and no artifact-count hits in the gate. Interpretation: for this subset, the dominant failure was window selection/truncation losing extractable evidence, not complete local-extractor inability. Still, 2/8 whole-session parents remained zero, so whole-session is not a blanket production repair. Before production mutation, inspect those two and build a backup-first/proposal-first re-retain plan; never merge temp-bank facts directly.

Dialogue simulation follow-up (2026-05-09): controlled natural dialogue sims reproduced the same class of Hindsight/local qwen behavior, but not a fixed winning variant. Real contrast8: `first-useful-window` 0/8 vs `whole-session` 6/8. Simulation A (focused/resume/status/low-signal): `first-useful-window` 2/4 vs `whole-session` 0/4, with generic-involving/malformed-When/prompt-context artifacts. Simulation B (early vague window + later summary): mixed 1/3 vs 1/3, no artifacts. Conclusion: zero-unit docs and variant flips are normal input/segmentation sensitivity for local retain; they are not alone evidence of production pipeline corruption. This does not make results production-ready; still require per-parent proposal, backup-first repair, and quality/recall gates.

MiniMax comparison (2026-05-09): re-retaining the exact same contrast8 manifest in a MiniMax temp bank increased density but did not change structural coverage. Local qwen: 38 units, parent coverage 6/8, `first-useful-window` 0/8, `whole-session` 6/8. MiniMax: 61 units, parent coverage 6/8, `first-useful-window` 0/8, `whole-session` 6/8; the same two parents stayed zero. Interpretation: MiniMax is better for fact richness on recovered whole-session inputs, but bad windows and hard-zero parents remain bad. Do not treat provider switch as a cure for zero-unit; fix/select input route first, then optionally use MiniMax for final high-quality extraction.

Hard-zero local variant probe (2026-05-09): the two parents that stayed zero under both local qwen and MiniMax contrast8 were probed in a local-only temp bank with 8 deterministic variants. Parent coverage became 2/2, but doc coverage was only 3/8 and one `whole-retry` fact had malformed `When: 2026-05- paths`. The Egomotion4D parent recovered useful technical facts only when the final explicit skill-correction/lesson cue was included (`core-plus-skill-note`: 10 units); shorter core-analysis/metric-core variants still produced zero units. The Hindsight native-consolidation parent had ingress hygiene issues because internal English planning blocks survived in the transcript; removing them made focused user-facing variants zero, while whole-retry produced only two generic user-intent facts. Interpretation: even hard-zero can recover via retry/shape, but the winning shape may be parent-specific and not production-safe. Use this as evidence for per-parent repair proposals, not blanket whole-session, first-window, or provider-switch rules. Also fix event_date handling before promotion; otherwise facts may get the run date rather than the source session date.

Practical scripts from this expansion:
- `$HOME/.hermes/scripts/hindsight_first_useful_window_manifest.py` builds first-useful-window manifests from high-value zero-unit candidates while keeping Hindsight-visible metadata `{}` and writing provenance to a sidecar JSON.
- `$HOME/.hermes/scripts/hindsight_fact_quality_gate.py` is a read-only temp-bank gate for parent coverage, zero-unit docs, artifact counters, and operation/payload status.

## Operational rule

Before enabling native consolidation/observations on a session-json bank, require low-level retain facts to pass coverage + recall precision gates. If facts are noisy, keep consolidation off; otherwise observations may crystallize polluted or incomplete facts into higher-level memory.
