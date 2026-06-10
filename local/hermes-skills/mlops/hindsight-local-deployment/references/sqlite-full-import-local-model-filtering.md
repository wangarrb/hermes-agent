# SQLite full import filtering with local Ollama models

Updated: 2026-05-06

## Purpose

Use this reference when backfilling Hermes SQLite history into Hindsight and the user wants full import to be cost-controlled without losing important long-term memory.

Core principle: local models are auxiliary gates, not the final memory extractor. Keep original text for Hindsight retain whenever possible; do not first summarize with a small local model, because that loses numeric evidence, paths, experimental conditions, and decision boundaries.

## Tested local model suitability

Latest verification after Ollama upgrade to `0.23.0` (2026-05-05): `gemma4:4b-local` now loads and runs, but it is still **not suitable as an automatic drop gate** for SQLite import filtering.

Models observed on this machine:

- `gemma4:4b-local`
- `llama3.1:8b-local`
- `qwen2:7b-instruct`
- `qwen3:8b-local`
- `deepseek-r1:7b-distill-qwen-local`

Gate-test set: 12 snippets covering short acknowledgements, compaction handoff, Egomotion4D technical conclusion, Hindsight config, OpenClaw tool quirk, weather query, user preference, raw importer progress, command/env fact, current-time query, merge lesson, pure logs.

| Model | JSON stability | Gate accuracy | Observed problem | Role |
|------|----------------|---------------|------------------|------|
| `llama3.1:8b-local` | 12/12 | 11/12 | One false drop on terse user preference; backup rescues it | **Primary local gate** |
| `qwen2:7b-instruct` | 12/12 after manifest repair | 10/12 | Over-keeps compaction/progress; conservative rescue model | **Backup/rescue gate** |
| `gemma4:4b-local` | 12/12 parse but often `{}` | 6/12 | Runs now, but over-keeps all low-value cases under strict JSON prompt | Do not use for auto drop |
| `qwen3:8b-local` | 9/12 | 5/12 | Invalid JSON / over-keeps | Do not use for auto drop |
| `deepseek-r1:7b-distill-qwen-local` | 11/12 plus timeout on smoke | 5/12 | Reasoning model; slow/schema-unstable/over-keeps | Do not use for auto drop |

Qwen2 note: after upgrade/import, `ollama list` showed `qwen2:7b-instruct` but `/api/show` initially returned 404. Running `ollama run qwen2:7b-instruct ...` pulled/wrote the official manifest and repaired API usage. If this reappears, run a smoke `ollama run` before benchmarking.

Recommendation: keep `llama3.1:8b-local` as primary and `qwen2:7b-instruct` as backup. Do not use `gemma4:4b-local` for automatic drops despite runtime availability; it is acceptable only as a conservative keep-biased smoke/retain fallback after separate quality tests.

## Pipeline design

### Stage 0: Split below session level

Do not only filter at session level. Long sessions mix valuable facts with tool logs, compaction blocks, repeated project recall, and transient progress.

Prefer:
- message-level filtering, or
- small turn windows (2-6 user/assistant messages), preserving session_id/timestamp/order.

Then re-bundle kept text by day-topic.

### Stage 1: deterministic rules first

Hard keep (local model cannot override):
- User preferences / persistent rules: 默认, 必须, 不要, 记住, 用户偏好, 教训.
- Project decisions / conclusions: 结论, 决定, 推荐, 不建议, 取舍, 根因, 风险.
- Verification/debug evidence: 已验证, 通过, 失败, error, traceback, 修复, bug, metrics.
- Environment/tool facts: ports, providers, model names, Docker/container state, paths, commands, config, endpoints.
- Numeric evidence: ATE/RPE, chunks, token, calls, percentages, experiment comparisons.
- Core topics: Egomotion4D, Hindsight, Hermes, OpenClaw, Pi3X, DAGE, GTSAM, etc.

Hard drop if no hard-keep signal:
- Short acknowledgements: 好, 收到, 谢谢, 继续, ok.
- CONTEXT COMPACTION / reference-only handoff blocks.
- Untrusted metadata wrappers themselves.
- Weather/temporary one-off queries.
- Pure progress/log text without durable conclusion.
- Duplicate project-recall summaries; keep latest/highest-quality representative if needed.

Everything else is gray zone.

### Stage 2: local-model gray-zone gate

Primary model: `llama3.1:8b-local`.

Expected strict JSON:

```json
{
  "keep": true,
  "priority": "high|medium|low|drop",
  "confidence": 0.0,
  "info_types": ["preference", "decision", "verification", "config", "project_fact", "temporary", "noise"],
  "reason": "<=40 chars"
}
```

Decision policy:
- `keep=true` or priority high/medium: keep.
- `drop` with confidence >= 0.80 and deterministic rule score is low: drop.
- `drop` but weak technical signal exists or confidence < 0.80: ask backup model `qwen2:7b-instruct`.
- Both models drop: drop.
- Models disagree: keep as low priority.
- JSON parse failure / timeout: keep as low priority; never lose content because a local gate malfunctioned.

### Stage 3: local duplicate filtering

Before Hindsight retain, use deterministic dedup:
- normalized exact hash,
- simhash/minhash for near-duplicate long text,
- direct removal of compaction/handoff blocks,
- representative selection for repeated “回忆项目” summaries.

Do not do per-item Hindsight recall for full dedup by default. It does not burn LLM, but is slow/complex; use recall for sampling and post-import validation.

### Stage 4: retain parameters

Current default safe import baseline after bug fixes:

```bash
--full --group-by day-topic --no-main \
--prefilter safe \
--retain-chunk-size 8000
```

Reason: call volume is now controllable, and `safe` only removes obvious low-value noise. Current full dry-run shows removing `safe` increases about +8.6% chars / +8.9% retain chunks. Use `none` when maximum coverage is desired; use balanced/strict only for emergency cost-control full rebuilds after sample audit.

Historical aggressive estimates on this machine:
- none + 8k: about 1144 chunks in the earlier run; current run is 1003 chunks.
- balanced + 16k: about 478 chunks (~58% reduction in the earlier run).
- balanced threshold15 + 16k: about 383 chunks earlier; current run is 363 chunks.

Do not default to 48000. Large chunks can miss numeric evidence, paths, experimental conditions, and decision boundaries.

### Stage 5: audit before submit

Dry-run must show:
- kept/dropped counts,
- dropped-by-reason,
- estimated retain chunks,
- max bundle chars,
- random sample of dropped items,
- random sample of kept items,
- hard-keep count,
- model-drop count.

Acceptance criterion: dropped samples must not contain obvious high-value facts. If they do, lower threshold or prefer keep.

Post-import recall spot checks should cover:
- Egomotion4D key conclusions,
- Hindsight architecture/config,
- OpenClaw quirks,
- user preferences,
- retain_chunk_size / auto_retain / observations decisions.

## Implemented importer controls

As of 2026-05-05, `$HOME/.hermes/scripts/import_sqlite_to_hindsight.py` implements the conservative local gate and audit path.

Available options:

```bash
--prefilter none|safe|balanced|strict
--prefilter-threshold N   # defaults after 2026-05-05: safe=1, balanced=7, strict=12
--retain-chunk-size N
--local-filter llama3.1:8b-local
--backup-filter qwen2:7b-instruct
--drop-policy single|consensus
--local-filter-timeout SECONDS
--local-filter-max-calls N
--sample-report N
--sample-seed N
```

Operational semantics:
- deterministic rules run first;
- local model is only an auxiliary gate for gray-zone content the rules are prepared to drop;
- hard-keep content is not overridden by the local model;
- recommended `--drop-policy consensus`: only drop when primary and backup both decide drop;
- model timeout, HTTP error, JSON parse failure, or `--local-filter-max-calls` exhaustion all conservatively keep the content;
- Ollama calls are serial and batch-ordered: primary runs across all candidates first, then backup runs across primary-drop candidates, so the script avoids repeated primary/backup load thrashing on 8GB GPUs;
- current implementation is session-level gray-zone review after message-level filtering. Balanced mode with an explicit threshold plus dual model rescue is safer than strict mode, because strict mode can drop content at message level before the model sees it.

Dry-run `--sample-report N` prints kept/dropped counts, reason counts, local/backup call counts, and random kept/dropped examples. Always inspect dropped samples before a full MiniMax submit.

## Recommended command shape

2026-05-06 revisit: after JSON parser / payload-null / queue-drain bugs were fixed, SQLite import call volume is controllable enough that the default should be quality-first and fail-open.

Current default for scheduled/normal import:

```bash
python3 ~/.hermes/scripts/hindsight_minimax_import.py sqlite-import-minimax -- \
  --mode submit --group-by day-topic --prefilter safe
```

`hindsight_minimax_import.py` auto-injects `--prefilter safe` when no `--prefilter` is supplied. It does **not** enable `--local-filter` by default.

Current dry-run on this machine (`--full --group-by day-topic --no-main`, 2026-05-06):

| Config | Records | Bundles | Retain chunks | Chars | Delta if removed |
|---|---:|---:|---:|---:|---:|
| safe + 8k | 457 | 108 | 921 | 6,903,075 | baseline |
| none + 8k | 490 | 119 | 1003 | 7,499,712 | +7.2% records / +10.2% bundles / +8.9% chunks / +8.6% chars |
| balanced7 + 8k | 417 | 88 | 790 | 5,978,490 | removing balanced -> +17.5% records / +35.2% bundles / +27.0% chunks / +25.4% chars |
| balanced15 + 16k | 370 | 79 | 363 | 5,100,003 | vs none+16k: +32.7% records / +50.6% bundles / +48.8% chunks / +47.2% chars |

Current incremental dry-run showed `safe` and `none` identical (35 records, 6 bundles, 46 chunks), so safe filtering adds little cost risk in normal daily operation.

Use the older dual local-model gate only for an emergency full rebuild where paid-call volume again becomes the dominant risk, and only after inspecting dropped samples:

```bash
python3 ~/.hermes/scripts/import_sqlite_to_hindsight.py \
  --mode dry-run --full --group-by day-topic --no-main \
  --prefilter balanced --prefilter-threshold 15 \
  --retain-chunk-size 16000 \
  --local-filter llama3.1:8b-local \
  --backup-filter qwen2:7b-instruct \
  --drop-policy consensus \
  --sample-report 12
```

Do not make this aggressive path the default: it saves calls but can drop weakly expressed preferences or rare details. Prefer `safe` or even `none` when quality/coverage matters more than a ~9% chunk increase.
