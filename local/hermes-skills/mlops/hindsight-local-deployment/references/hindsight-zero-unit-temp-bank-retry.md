# Hindsight zero-unit temp-bank retry pattern

Use this when a session/json native retain trial reports many `docs_without_units` / zero-unit documents after all retain operations completed successfully.

## Why this matters

A completed retain operation with `failed_operations=0` can still be a quality failure: the document may produce zero memory units. Do not treat zero-unit as automatically low value. A controlled temp-bank retry can separate true low-signal sessions from retain stochasticity, cleaning loss, overlong context, prompt/model sensitivity, or windowing needs.

## Baseline case: Phase B zero-unit temp-bank retry, 2026-05-08

Run artifacts:

- parent run: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855`
- temp bank: `hermes_tmp_zero_unit_phase_b_20260508`
- candidate manifest: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-tempbank-candidates.jsonl`
- paid retain log: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-paid-retain.log`
- audit JSON: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-tempbank-retain-audit.json`
- quality report: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-tempbank-quality-hardening.md`
- final summary: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-tempbank-final-summary.md`

Observed result:

- submitted_items: 20
- estimated_retain_chunks: 35
- content_chars: 209,538
- queue drained: yes
- provider restored normal-local: yes
- failed_operations: 0
- documents: 20
- memory_units: 112
- observations: 0
- docs with units: 13 / 20 = 65%
- docs still zero-unit: 7 / 20 = 35%

Key lesson: 65% of high-value production zero-unit candidates recovered facts on whole-session temp-bank re-retain. This means zero-unit often indicates retry/windowing/extraction instability, not absence of durable content.

## Route-specific case: Phase B2 zero-unit retry, 2026-05-08

Artifacts:

- temp bank: `hermes_tmp_zero_unit_phase_b2_route_retry_20260508`
- route-specific manifest: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b2-route-specific-tempbank-candidates.jsonl`
- local retain log: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b2-route-specific-local-retain.log`
- audit JSON: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b2-route-specific-tempbank-retain-audit.json`
- final summary: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b2-route-specific-tempbank-final-summary.md`

Observed result:

- remaining Phase-B zero-unit parent sessions retried: 7
- route-specific records submitted: 13
- recovered parent sessions: 7 / 7
- memory_units: 52
- failed_operations: 0
- observations: 0
- route success:
  - `production_windowed`: 4 / 4 parents recovered
  - `retry_less_aggressive_cleaning`: 2 / 2 parents recovered
  - `retry_custom_mission`: 1 / 1 parent recovered

Key lesson: route-specific retry can recover all remaining high-value zero-unit parents, but recovered facts still need quality filtering. This temp bank had recall macro precision only `0.125` and included some command-log/generic meta facts, so it is evidence for a repair rule, not a merge source.

Operational notes:

- If paid/MiniMax credentials are unavailable or intentionally disabled, a direct local retry using the current Ollama-backed Hindsight instance can still validate route logic. Create/patch only a temp bank, keep observations disabled, and set retain concurrency to 1 for local models.
- Hindsight may store item metadata on `memory_units` rather than `documents`; for zero-record/parent aggregation, join DB document counts with the manifest rather than relying on `documents.metadata`.

## Preserve-key-context repair manifest case: Phase B3, 2026-05-08

Artifacts:

- source candidate manifest: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-tempbank-candidates.jsonl`
- repair manifest: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-repair-manifest-preserve-key-context.jsonl`
- summary markdown: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-repair-manifest-preserve-key-context.md`
- summary JSON: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b-zero-unit-repair-manifest-preserve-key-context.json`
- target temp bank: `hermes_tmp_zero_unit_phase_b3_preserve_20260508`

Manifest construction details:

- record_count: 20
- total_content_chars: 209,538
- records keep `content_omitted=true` and rely on `metadata.json_path` rehydration by `hindsight_session_retain_runner.py`.
- `action` is set to `production` only to allow retain runner submission; `bank_target` is redirected to the temp bank, not production `hermes`.
- metadata adds:
  - `phase_b3_preserve_priority`: `["work", "research", "decision", "turning_point", "result"]`
  - `phase_b3_repair_note`: `Preserve working/research/decision/turning-point/result context; filter only low-value noise.`

Key lesson: when repairing high-value zero-unit sessions for this user, do not over-filter semantic project evolution. The default repair posture is preserve-first: keep work/research/decisions/turning points/results, filter only low-value noise and credential-like material, then use post-retain quality gates to stop pollution.

Observed retain result (same phase, local Ollama/qwen3.5 temp-bank run):

- temp bank: `hermes_tmp_zero_unit_phase_b3_preserve_20260508`
- final summary: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-b3-preserve-tempbank-final-summary.md`
- submitted/docs: 20
- docs_with_units: 13 / 20
- remaining zero-unit docs: 7 / 20
- memory_units: 154
- failed_operations: 0
- observations: 0
- recall macro precision: `0.3125`
- broad/system tag total: 0
- exact contamination counters in the audit were 0
- transient issue seen: one retain retry hit PostgreSQL/Hindsight entity index limit (`idx_entities_canonical_name` row size exceeds btree maximum) but final async operations completed.

Interpretation: preserve-first custom instructions increased extracted fact volume compared with the Phase B whole-session baseline, but did not reduce the 7 high-value zero-unit documents. Do not rerun the full week with preserve-only whole-session settings. The next production-quality retry should combine preserve-first semantics with route-specific rules (`production_windowed`, `retry_less_aggressive_cleaning`, `retry_custom_mission`) and then gate in a new temp bank.

Before running this manifest or any derived one-week v2 manifest, do a key-like/secret scan of the new manifest, verify target temp bank is empty or explicitly recreate it, and keep observations/consolidation disabled. Retain completion should be followed by bank quality audit, session quality hardening, recall smoke/top-k contamination checks, and a production repair proposal. Do not directly merge temp-bank facts.

## Recommended workflow

1. Do not mutate production and do not merge recovered temp-bank facts.
2. Build a high-value zero-unit candidate manifest from the production trial.
3. Use preserve-first semantics for this user: retain work/research/decision/turning-point/result context; filter only low-value noise and credential-like material.
4. Before execution, secret-scan the repair manifest and verify the target temp bank is empty or intentionally recreated.
5. Run native retain into a clearly temporary bank with observations disabled; use background+notify for long/paid runs.
6. Wait for pending/processing=0, then verify DB counts, child retain operations, and docs-without-units.
7. Run a quality hardening report on the temp bank.
8. Interpret recall smoke carefully: a zero-unit temp bank is intentionally biased and should not be treated as a production recall gate.
9. Use the remaining zero-unit docs to design route-specific experiments.
10. Promote only generalizable rules into manifest builder/selector.
11. For a production-quality one-week rerun, combine preserve-first default handling with route-specific records rather than rerunning preserve-only whole sessions. The 2026-05-08 Phase C candidate manifest initially used only the default hardening sample list: 188 source records -> 214 output records, routed parents=30, estimated_retain_chunks=304. That run is incomplete as an all-routes test because the report had `high_value_retry_candidate_count=100` but only 30 sampled `high_value_retry_candidates`. Correct procedure: rerun hardening with `--max-samples` above the reported candidate count, then regenerate an all-routes manifest and verify `routed_parent_count == high_value_retry_candidate_count` before paid execution.
12. Corrected Phase C2 paid run (MiniMax, 16 concurrency) result: manifest 188 source parents -> 227 output records, routed parents=100/100, submitted=227, documents=227, memory_units=1544, docs_with_units=100, parent coverage=93/188, failed_operations=0, observations=0, recall macro precision=0.3958. It improved over baseline (83/188 parent coverage, 852 facts) but still left 95/188 parents zero and recall contamination remained for cch/openclaw/patent/user_pref, so do not mutate production or enable consolidation yet. Artifacts: `$HOME/.hermes/hindsight/runs/hindsight-reset-weektrial-20260508-201855/phase-c2-week-v2-allroutes-quality-hardening.md` and parent coverage JSON.
13. If another paid rerun is required, require explicit paid-run authorization because hundreds of chunks can fan out to many retain LLM calls.

## 2026-05-10 bge-m3 zero-unit rehydrate / relaxed prompt experiment

Context: corrected production retain used 83 cleaned user+assistant-only v2 docs, `BAAI/bge-m3`, `vector(1024)`, `custom`, chunk_size=8000. It completed with 638 memory units and 23 `docs_without_units`.

Controlled temp-bank experiments:

1. **Rehydrate source dialogue, original document IDs**
   - temp bank: `hermes_tmp_zero_unit_bge_m3_v2_rehydrate_20260510`
   - manifest: `$HOME/.hermes/hindsight/runs/zero_unit_repair_20260510/zero_unit_tempbank_retry_manifest_rehydrate.jsonl`
   - input: 23 zero-unit parent docs, `content_omitted=true`, original `document_id`, valid `metadata.json_path` pointing to the source Hermes JSON
   - result: documents=23, memory_units=44, docs_without_units=11, failed=0, observations=0
   - recovery: 12 / 23 parent docs recovered units

2. **Extra-relaxed custom instructions on the remaining zero-unit docs**
   - temp bank: `hermes_tmp_zero_unit_bge_m3_v2_relaxed2_20260510`
   - manifest: `$HOME/.hermes/hindsight/runs/zero_unit_repair_20260510/zero_unit_tempbank_retry_manifest_relaxed2.jsonl`
   - input: 11 remaining zero-unit parent docs, source rehydration, stronger instruction: extract any durable fact/decision/result/preference/config/path/test outcome/project direction; keep terse sessions if meaningful; skip greetings/chatter/secrets
   - result: documents=11, memory_units=30, docs_without_units=5, failed=0, observations=0
   - additional recovery: 6 / 11 docs, combined potential recovery 18 / 23, leaving 5 persistent zero-unit docs

Important lessons:

- Temp bank is not only emergency fallback; it is the validation/staging sandbox for relaxed production rules. Do not mutate production until the relaxed rule is proven in temp bank and quality-gated.
- For `content_omitted=true`, `hindsight_session_retain_runner.py` rehydrates by matching `document_id` against records generated from `metadata.json_path`. Synthetic document IDs like `zero-retry::<id>::w00` will not rehydrate. Use the original `hermes-session::<session_id>` document ID when relying on source rehydration.
- Rehydrating original source JSON with user+assistant-only cleaning is the first safe relaxation: it preserves dialogue context without reintroducing tool traces.
- Extra-relaxed prompts improve coverage but can increase broad/less relevant recall smoke results. Treat recovered facts as candidates; run quality hardening before production repair or consolidation.

## 2026-05-10 production repair after temp-bank validation

After temp-bank validation, do not merge temp-bank facts. Convert the winning rule into production repair manifests and re-run against the production bank with original `document_id` / `metadata.json_path` lineage.

Observed production repair on bank `hermes`:

- baseline: 83 docs, 638 memory_units, 23 docs_without_units, observations=0
- model/profile as executed historically: `deepseek-v4-flash` under the old wrapper mapping, which routed through OpenCode Go; user later corrected that future `deepseek-v4-flash` must mean DeepSeek provider / official DeepSeek API, not OpenCode Go. The wrapper profile was patched accordingly.
- concurrency: 4 (chosen over high parallelism to reduce 429/provider instability risk)
- pass1 `rehydrate`: 638 -> 668 units, 23 -> 15 zero docs
- pass2 `relaxed2`: 668 -> 679 units, 15 -> 12 zero docs
- pass3 `relaxed3`: final 700 units, 7 zero docs
- final audit: duplicate_exact_text_groups=0, contamination counters=0, missing_source_refs=0, units_missing_document=0, observations=0
- wrapper restored Hindsight to normal-local; accidental LLM calls use local Ollama after completion

Keep these decisions for future repairs:

- Prefer a generic route/prompt class over topic/session special-cases: `rehydrate`, `relaxed short durable signal`, `windowed`, and `less aggressive cleaning` are acceptable; `special-case openclaw/patent/cch/session-date` is not.
- Before production mutation, check coverage for both the zero subset and the whole production set. Use bank-scoped DB/audit counts as truth; the observed document API `memory_unit_count` can be cross-bank polluted when temp banks reuse production `document_id`.
- For same-content in-place production repair, pass `--ignore-submit-state` or a dedicated repair submit-state; otherwise the runner may skip records whose content hash was already submitted.
- Stop before chasing zero at any cost. If an already relaxed generic prompt leaves a small hard tail, switch to manual review or deterministic window/explicit-content micro-routes. Further prompt broadening raises noise/hallucination risk.
- If final audit shows newly added units with null/empty observation scope, keep observations/consolidation disabled until scopes are investigated or regenerated.

## Route-specific next experiments

For remaining zero-unit docs, split by generic class and test in separate temp banks:

- `production_windowed`: deterministic contiguous windows for noisy / overlong / multi-scope sessions. If using `content_omitted=true`, keep original document IDs; if using synthetic window IDs, include explicit `content` because runner rehydration cannot match synthetic IDs.
- `retry_less_aggressive_cleaning`: retain more surrounding context when cleaning may have dropped important cues; source rehydration with original doc IDs is preferred before synthetic rewriting.
- `retry_custom_mission`: use a generic durable-memory extraction mission for semantically dense or terse sessions where default extraction appears too strict.
- `extra_relaxed_short_signal`: for short docs with durable technical/user signal, allow one concise fact instead of requiring rich semantic context.

Do not write project/query-specific fixes such as special handling for `openclaw`, `patent`, `cch`, or a single dated run. The selector rule must be structural and reusable.

## 2026-05-10 bge-m3 full-run note

After a clean 83-record user+assistant-only bge-m3/vector1024 production retain, the main residual issue was `docs_without_units=23` / `83` with no contamination and no failed ops. The right next step was not consolidation; it was a temp-bank retry.

Observed temp-bank repair inputs:
- 23 zero-unit parent docs
- 103,134 total chars
- 0 tool markers
- 1 doc required redaction of secret-like text before rehydration
- routes used:
  - `production_windowed` for long/overlong parents
  - `retry_less_aggressive_cleaning` for medium-signal parents
  - `retry_custom_mission` for short or weak-signal parents
- output manifest: 31 records / ~31 chunks@8000, dry-run clean

Operational lesson:
- If the audit shows healthy counts but leftover zero-unit parents, do not jump to consolidation or observations.
- First diagnose zero-unit docs, build a temp-bank retry manifest, run a secret-like scan, and dry-run before any paid execution.
- Keep the retry target bank clearly temporary and do not merge recovered facts into production until the winning route is generalized and re-run in a production trial.

## Verification checklist

- Hindsight mode restored to normal-local after paid run.
- `HINDSIGHT_API_ENABLE_OBSERVATIONS=false` and consolidation slots disabled unless explicitly testing consolidation.
- `failed_operations=0`.
- `pending_operations=0`, `processing_operations=0`.
- `retain` child operations have `payload_null=0`.
- Parent `batch_retain` operations may have `payload_null` after completion; do not confuse this with child retain payload loss.
- Count documents and memory_units directly from DB/API.
- Record remaining zero-unit document ids and classes.
- Do not run consolidation until facts and recall gates pass.

## Pitfalls

- `failed_operations=0` is not enough; check `docs_without_units`.
- A temp bank biased toward zero-unit docs will often have poor recall benchmark precision; do not use that as production recall success/failure alone.
- For windowed/route-expanded manifests, document-level zero ratio can be misleading because output docs are not source sessions. Always compute parent-level coverage: original parent sessions, parents_with_units, parent_zero_count, and route-parent coverage. Use this for baseline/v2 comparison.
- Hardening reports may cap `high_value_retry_candidates` to a sample while separately reporting a larger `high_value_retry_candidate_count`. Before any paid or full rerun, verify the manifest's routed parent count equals the reported candidate count; otherwise stop, regenerate full samples, and record the earlier run as incomplete.
- When quality looks bad, do an explicit attribution split instead of blaming Hindsight or the agent generically: (1) operator/pipeline mistakes such as sample-vs-full candidate errors, (2) ingress/cleaning/windowing/route design gaps for noisy Hermes transcripts, (3) Hindsight limitations such as completed zero-unit docs, recall topic drift, weak coverage guarantees, and entity/index edge cases. Own confirmed agent mistakes first.
- Docker logs may be lost after the wrapper recreates the container back to normal-local; preserve relevant worker logs before restore if retry traces matter.
- Live MiniMax JSON parse retries may occur on command/log-heavy sessions yet still drain successfully; distinguish transient retries from failed operations.
- Do not merge temp-bank facts into production just because they recovered. First convert the winning retry route into a generic production rule and rerun the production trial.
- When temp banks reuse the same `document_id` as production, do **not** trust `GET /banks/<bank>/documents/<id>` `memory_unit_count` for zero-unit decisions in the observed Hindsight version: the source query joins `memory_units` on `document_id` without `bank_id`, so temp-bank units can make a production zero-unit document look nonzero. Use DB lineage with `mu.bank_id = d.bank_id` or the audit DB fallback for authoritative counts.
- Production repair with the same source document IDs/content can be skipped by the session runner submit-state because the content hash is unchanged. For intentional in-place zero-unit repair, use `--ignore-submit-state` or a dedicated repair submit-state, and keep `update_mode=replace` so the document is reprocessed from scratch.