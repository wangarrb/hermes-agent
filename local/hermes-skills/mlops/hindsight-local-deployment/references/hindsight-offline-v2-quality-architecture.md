# Hindsight Offline v2 Quality Architecture Notes

Session: 2026-05-06. Use when improving Hermes/Hindsight offline memory quality beyond default automatic Hindsight retain/consolidation.

## Core correction

The P0 issue was initially misdiagnosed as “offline consolidation does not write back to Hindsight.” That is wrong. `~/.hermes/scripts/offline_hindsight_reflect_consolidate.py` already has `post_to_hindsight()` and posts retain-friendly markdown to `/v1/default/banks/{bank}/memories`.

The real P0 is semantic layering:

- Offline summaries are written back, but Hindsight re-extracts them as ordinary `experience/world` facts.
- Built-in Hindsight observations/consolidation are intentionally disabled for daily safety.
- Offline JSON schema did not emit a dedicated `canonical_observations` layer.
- High-level recall often still ranks raw SQLite facts before offline daily/weekly summaries.

So the fix is not “post more”; it is to add explicit canonical/high-level memory layers and layered retrieval.

## Shareability / generalization rule

This offline Hindsight skill must remain shareable. Do not hard-code a specific user's projects, research domain, model names, tools, benchmark numbers, or entity aliases into default scripts/prompts.

Generic defaults must:

- auto-discover topics from Hindsight documents, offline output metadata, entity names, and content clusters;
- auto-discover alias fragmentation by normalization/patterns, not by a hand-written list of local entities;
- use bilingual Chinese/English query probes where possible, because user memory may be Chinese, English, or mixed;
- keep local/domain-specific benchmarks in `.local` files or user config, never as default skill data;
- use placeholders such as `topic:<auto-discovered-domain>` in documentation instead of local project names.

Local examples may be documented only as examples/regression artifacts and must be clearly labeled non-default.

### Genericization checklist from the v2 audit session

When turning an offline Hindsight workflow into a reusable skill or default script:

- Move domain-specific benchmark cases into `benchmark_queries.local.jsonl`; keep `benchmark_queries.jsonl` limited to generic classes such as user preferences, project decisions, tooling lessons, risks/open questions, evidence lookup, and offline pipeline quality.
- Audit entity fragmentation by automatic normalization only: strip path basename, file extensions, generic prefixes (`raw`, `local`, `global`, `canonical`, `version`, etc.) and generic suffixes (`config`, `provider`, `baseline`, `script`, etc.). Do not maintain hand-written alias groups for a user's projects or tools.
- Keep recall probes generic and bilingual by default; auto-add topic probes from local offline output metadata or Hindsight entities instead of hard-coding local project names.
- Document any project/model/number examples as private regression artifacts, not as default expectations for shared skills.
- Verify genericization with both a no-recall audit smoke test and the generic eval benchmark before declaring the workflow shareable.

## Audit facts from 2026-05-06

Current healthy baseline:

- `documents=183`
- `memory_units=4049`
- `observations=22`
- queue clean: `pending=0`, `processing=0`, `failed=0`
- DB layers: `offline_daily=57`, `offline_weekly=23`, `sqlite_import=91`, `other=12`
- local offline JSON vs DB reconciliation passed: `local_not_in_db=0`, `db_not_local=0`

Quality issues:

- active offline JSON outputs had `canonical_observations=0`
- weekly all-history is a chunked map pass, not a true global reduce; `weekly single_wrapped=12`
- language inconsistency: `english_only offline outputs=7`
- entity alias fragmentation is now audited by generic normalization, not by project presets; local 2026-05-06 examples merely showed multiple auto-discovered alias groups with count >= 10.
- high-level recall was weak for some auto-discovered/local benchmark topics because raw SQLite facts dominated.

## Scripts added

Read-only / safe Phase 0 scripts:

```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_audit.py
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode dry-run --scope all
python3 ~/.hermes/scripts/hindsight_recall_layered.py --mode high-level 'QUERY'
python3 ~/.hermes/scripts/hindsight_eval.py
```

Files:

- `~/.hermes/scripts/hindsight_offline_v2_audit.py`: read-only audit for health, layers, local-vs-DB reconciliation, observation gap, language, entity fragmentation, recall source mix.
- `~/.hermes/scripts/hindsight_offline_v2_reduce.py`: no-LLM/no-Hindsight-write local reducer that reads existing daily/weekly JSON and writes local-only L3/L4 cards under `~/.hermes/hindsight/offline_reflect/v2_cards` when `--mode local` is explicit. It excludes `_bad-output-backup*` by default, keeps type diversity so preferences/decisions/risks/open questions are not crowded out by technical lessons, and writes `observations_index.jsonl` so detailed numeric observations not selected into compact cards remain retrievable.
- `~/.hermes/scripts/hindsight_offline_v2_gate.py`: no-LLM/no-Hindsight-write publish gate. It compares default eval vs `--use-local-cards` eval for at least generic + local benchmark pairs, blocks unless term recall and expected layer hits both improve, and emits only local proposal preview files when all checks pass.
- `~/.hermes/scripts/hindsight_recall_layered.py`: no-LLM layered recall helper; modes `high-level`, `evidence`, `mixed`; local rerank with layer weighting, meta-log penalty, significant-token/type/numeric scoring, and a cap on experimental local canonical results. Local v2 cards are experimental and fail-closed: pass `--use-local-cards` to enable them.
- `~/.hermes/scripts/hindsight_eval.py`: no-LLM eval harness comparing direct Hindsight recall vs layered recall. Local cards are also opt-in via `--use-local-cards`.
- `~/.hermes/hindsight/eval/benchmark_queries.jsonl`: generic bilingual benchmark template.
- `~/.hermes/hindsight/eval/benchmark_queries.local.jsonl`: local/private regression benchmark; may contain user/project-specific topics and numbers, but must not be used as the shareable default.
- `~/.hermes/hindsight/offline_reflect/OFFLINE_HINDSIGHT_V2_DESIGN.md`: full design draft.

## Evaluation baseline

First eval run:

```text
direct_avg_score: 64.86
layered_avg_score: 75.83
direct_avg_term_recall: 0.554
layered_avg_term_recall: 0.533
direct_expected_layer_hits: 15
layered_expected_layer_hits: 45
```

Interpretation:

- Layered recall significantly improves high-level source hits.
- Term recall did not improve yet, proving that rerank alone is insufficient.
- Need actual `canonical_observations` / topic cards / global reduce, not just retrieval tuning.

Good/weak case labels in local runs are private regression examples, not skill defaults. For the shareable workflow, evaluate generic categories first:

- user preferences / recurring corrections
- project decisions / architecture choices
- tooling lessons / environment configuration
- risks / blockers / open questions
- evidence lookup with concrete values, paths, or commands
- offline pipeline quality

Domain-specific cases can be stored in `benchmark_queries.local.jsonl`.

## Parameterized main-script extension

`offline_hindsight_reflect_consolidate.py` was extended without changing defaults:

```bash
--emit-observations       # default off
--output-language zh|en|auto  # default auto
```

When enabled, the LLM schema includes:

```json
"canonical_observations": [
  {
    "id": "stable deterministic id",
    "insight": "Chinese high-level insight",
    "type": "user_preference|project_decision|technical_lesson|risk|method_comparison|open_question|system_rule",
    "applicability": "scope/conditions",
    "evidence_ids": ["source fact/document ids"],
    "supersedes": ["old observation ids"],
    "confidence": "high|medium|low",
    "valid_from": "...",
    "valid_until": null,
    "tags": ["..."]
  }
]
```

`--output-language zh` asks for Chinese by default while preserving paths, commands, variable names, model names, and English technical terms.

Validation used:

```bash
python3 -m py_compile ~/.hermes/scripts/offline_hindsight_reflect_consolidate.py ~/.hermes/scripts/hindsight_offline_v2_audit.py ~/.hermes/scripts/hindsight_recall_layered.py ~/.hermes/scripts/hindsight_eval.py
python3 ~/.hermes/scripts/offline_hindsight_reflect_consolidate.py --scope daily --date 2026-05-05 --daily-source facts --mode dry-run --prefilter safe --emit-observations --output-language zh
```

## Recommended next phase

Phase 1 local reducer is now implemented and validated as a local-only experiment:

1. `hindsight_offline_v2_reduce.py` reads existing daily/weekly local outputs.
2. It generates local-only topic/global cards under `offline_reflect/v2_cards`.
3. It makes no LLM calls and performs no Hindsight writes.
4. It excludes `_bad-output-backup*` JSON by default.
5. It preserves type diversity so preferences, decisions, risks, and open questions are not crowded out by technical lessons.
6. It writes a local `observations_index.jsonl` with all deduped observations, so detailed numeric observations remain retrievable even when compact cards only keep top observations.
7. Layered recall scores local observations with significant tokens, type hints, numeric/measurement signals, and a local-result cap so broad canonical matches do not crowd out precise source evidence.

Validation commands:

```bash
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode dry-run --scope all
python3 ~/.hermes/scripts/hindsight_offline_v2_reduce.py --mode local --scope all
python3 ~/.hermes/scripts/hindsight_eval.py --benchmark ~/.hermes/hindsight/eval/benchmark_queries.jsonl --output-dir ~/.hermes/hindsight/eval/runs-generic-default
python3 ~/.hermes/scripts/hindsight_eval.py --benchmark ~/.hermes/hindsight/eval/benchmark_queries.jsonl --use-local-cards --output-dir ~/.hermes/hindsight/eval/runs-generic-cards
python3 ~/.hermes/scripts/hindsight_eval.py --benchmark ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl --output-dir ~/.hermes/hindsight/eval/runs-local-default
python3 ~/.hermes/scripts/hindsight_eval.py --benchmark ~/.hermes/hindsight/eval/benchmark_queries.local.jsonl --use-local-cards --output-dir ~/.hermes/hindsight/eval/runs-local-cards
python3 ~/.hermes/scripts/hindsight_offline_v2_gate.py \
  --pair generic ~/.hermes/hindsight/eval/runs-generic-default/latest.json ~/.hermes/hindsight/eval/runs-generic-cards/latest.json \
  --pair local ~/.hermes/hindsight/eval/runs-local-default/latest.json ~/.hermes/hindsight/eval/runs-local-cards/latest.json \
  --emit-proposal
```

Latest local validation after observation-index retrieval and conservative local-card rerank:

```text
reducer local build: input_files=82, backup_excluded=11, collected=2319, cards=21, observation_index_count=2099
local benchmark default layered: score=75.83, term_recall=0.533, layer_hits=45
local benchmark with --use-local-cards: score=87.55, term_recall=0.640, layer_hits=47
generic benchmark default layered: score=70.28, term_recall=0.461, layer_hits=28
generic benchmark with --use-local-cards: score=76.94, term_recall=0.528, layer_hits=29
publish gate: eligible_for_local_proposal; generic/local both pass; no case-level term recall regressions
proposal preview: ~/.hermes/hindsight/offline_reflect/v2_publish_gate/canonical-retain-proposal.{jsonl,md}
```

Important: local cards are still experimental and are not enabled by default. Keep `--use-local-cards` opt-in until repeated eval shows no regression. `hindsight_offline_v2_gate.py` must pass on generic + local benchmark pairs before any canonical docs retain/post is considered. A passing gate may only emit local proposal preview files; actual Hindsight retain still requires a separate explicit workflow and user authorization.

Target architecture:

```text
L0 raw SQLite transcripts
  -> controlled retain / first-pass extraction
L1 processed Hindsight facts
  -> daily topic consolidation
L2 daily summaries + daily observations
  -> topic-history reduce
L3 topic canonical cards
  -> global reduce
L4 global canonical observations
  -> layered recall
```

## Pitfalls

- Do not confuse “observation count low” with “offline docs not posted.” Check DB document prefixes first.
- Do not let offline layer boost outrank textual relevance; generic “performed offline consolidation” meta-facts can dominate unless penalized.
- Do not claim weekly all-history is a real global refresh if it is only size-based chunks. A true global reduce needs topic/card stages.
- Do not make new observation schema default immediately; keep `--emit-observations` opt-in until eval validates improvement.
- Do not write directly to `memory_units.fact_type='observation'` as a first step; safer route is canonical documents with evidence IDs.
