# Review-backlog A bucket temp repair runbook

Use when the Hindsight review backlog/scorer has produced an `A_next_repair_candidate` or `A-temp-repair-candidates-normalized` JSONL and the user asks to run the “待捞回区/捞回区” processing.

Related readiness/publishability checklist: `references/hindsight-offline-pipeline-readiness-and-publish-gate.md`.

## Safety boundary

- Do **not** submit A bucket candidates directly to production `hermes`.
- Submit only to an explicit temporary bank, e.g. `hermes_tmp_review_repair_a_sample40_YYYYMMDD`.
- Keep `~/.hermes/hindsight/config.json` `auto_retain=false`; verify after the run.
- Do not use raw full-session content if it contains credentials, context compaction, or tool logs.

## Important pitfall

`$HOME/.hermes/scripts/hindsight_retain_a_group_production.py` rehydrates sessions by concatenating raw `messages` from the session JSON. In the 2026-05-12 A-bucket run, its dry-run source scan found credential-like material in several candidates and context-compaction/raw assistant noise in the preview. Treat this script as unsafe for real paid/temp retain unless it is patched to use the cleaned session-manifest rehydration path.

Preferred real-submit path:

1. Build a **cleaned lean manifest** from the candidate JSONL:
   - use `hindsight_session_manifest.records_from_json_file(source_json_path, bank_target=<temp_bank>)`;
   - match the candidate `document_id`;
   - omit `content` in the output manifest;
   - keep `metadata.json_path` so `hindsight_session_retain_runner.py` rehydrates cleaned content;
   - add candidate metadata/tags, but avoid putting the temp bank id into fact-worthy content.
2. Build a sidecar JSON for gates with `docs[]` entries: `document_id`, `parent_document_id`, `variant`, and `candidate` fields.
3. Dry-run with:
   - `python3 hindsight_session_retain_runner.py --manifest <cleaned_manifest.jsonl> --bank <temp_bank> --action production --batch-size 3 --ignore-submit-state --json`
4. Before real submit, scan rehydrated cleaned content for credential-like strings.
5. Execute temp retain only:
   - `python3 hindsight_session_retain_runner.py --manifest <cleaned_manifest.jsonl> --bank <temp_bank> --action production --batch-size 3 --ignore-submit-state --execute --confirm retain-hindsight-session-manifest --wait-timeout-s 2400 --poll-s 10 --json > <execute.json>`
6. Wait until the temp bank has no `pending/processing` retain or consolidation ops. In this setup temp retain may auto-trigger consolidation/observations even if production is untouched; wait it out rather than interrupting.
7. Run quality gates:
   - `python3 hindsight_fact_quality_gate.py --bank <temp_bank> --sidecar <sidecar.json> --output-json <fq.json> --output-md <fq.md> --json`
   - `python3 hindsight_bank_quality_audit.py --bank <temp_bank> --recall-smoke --db-fallback always --output-dir <run_dir> --stem <stem> --json > <audit.json>`
8. Promotion rule:
   - Direct production remains **NO-GO** if `docs_without_units > 0`, artifact flags include real leaks, recall smoke shows broad irrelevant retrieval, or contamination counts are nonzero.
   - If only false-positive artifact regex hits remain, document why before proposing production.

## How to integrate this temp repair lane into the main flow

Do not inherit or insert the old raw-submit script as-is. Integrate the cleaned/temp-bank pattern as a first-class staged lane inside the normal offline pipeline:

1. Candidate input stays read-only: scorer/review-backlog emits A candidates plus source `document_id`/`source_json_path`.
2. A repair-manifest builder creates a lean cleaned manifest from `hindsight_session_manifest.records_from_json_file(...)`, matching each candidate by `document_id`. It must omit `content`, keep only `metadata.json_path` for cleaned rehydration, and put provenance/routing/debug fields into a sidecar, not Hindsight-visible metadata.
3. Preflight gates run before submit: source exists, cleaned content non-empty, credential/context-compaction/reasoning/tool-log scan clean, no temp bank/window ids in LLM-visible content, and `auto_retain=false` verified.
4. Execution always targets an explicit temp bank first. Never write A bucket directly to production. Use `hindsight_session_retain_runner.py` with `--execute --confirm retain-hindsight-session-manifest` and wait for retain plus consolidation/observations to finish.
5. Post-run gates are mandatory: `hindsight_fact_quality_gate.py` with sidecar, then `hindsight_bank_quality_audit.py --recall-smoke`. A run is production-eligible only when expected documents exist, parent coverage is acceptable, contamination/missing lineage are zero, artifact hits are explained or zero, and recall smoke does not cause broad irrelevant retrieval.
6. Failed parents are routed to a second-stage narrower repair: windowed extract or human-authored repair note. Do not keep retrying whole-session raw content.
7. Production write-back should be a separate proposal/apply stage inside the main flow, not an automatic follow-on from temp retain. It must reference the temp bank, gate reports, exact candidate list, and planned production manifest.

8. Weekly conflict/high-dimensional extraction can search the recall/repair zone. If it finds useful knowledge, promote the distilled fact through confirmation + canonical repair proposal; do not copy temp-bank units directly into production. After promotion, mark the sidecar item as promoted/superseded so the same knowledge does not remain as a duplicate truth source.

Integration target: add/keep tests around (a) raw tool/system messages are dropped, (b) lean manifest rehydrates cleaned content, (c) sidecar carries provenance while Hindsight-visible metadata stays minimal, (d) secret-like content blocks submit, and (e) temp gate blocks production when any parent has zero units or real artifact leakage.

## 2026-05-12 observed temp result

A cleaned A-bucket temp run submitted 13 candidates to `hermes_tmp_review_repair_a_sample40_20260509`: 13 docs, 271 memory units, 185 observations, no active ops, no production mutation. Gate was **not production-ready**: 4/13 docs had zero units and artifact flags included temp-bank/secret-like false positives, so production promotion was blocked pending narrower repair notes / metadata cleanup.
