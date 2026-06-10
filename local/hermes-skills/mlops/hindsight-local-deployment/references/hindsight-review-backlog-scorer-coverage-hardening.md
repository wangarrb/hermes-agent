# Hindsight review-backlog scorer coverage hardening

Session pattern captured 2026-05-09. Use when running paid LLM scorer sidecars over review backlog samples.

## Problem

A MiniMax scorer run with batch_size=5 completed safely but returned scores for only 12 of 39 prompted records. The script defaulted 27 missing records to `wait`, which is safe but not a valid routing signal. Do not treat defaulted `wait` as model judgment.

Root cause pattern: the prompt/schema showed `scores: [{...}]` but did not strongly require one score per input `document_id`, so the model often scored one item per batch.

## Required hardening

Before trusting scorer routes, verify the scorer script/prompt has these features:

- User prompt includes `required_document_ids` and `required_score_count`.
- Constraints explicitly say to return exactly one score object for every required document_id.
- Parser accepts list output and map-shaped output such as `scores_by_document_id`, `score_by_document_id`, or dict-shaped `scores`.
- Summary records:
  - `records_prompted_to_llm`
  - `valid_scores_from_llm`
  - `missing_scores_from_llm`
  - `missing_document_ids`
  - `score_coverage`
  - `coverage_ok`

Coverage gate: `missing_scores_from_llm == 0` and `score_coverage == 1.0` for small pilots. If this fails, stop before production repair.

## Safe execution pattern

1. Preflight remains sidecar-only:
   - Hindsight `/health` healthy.
   - Bank stats stable: pending=0, failed=0.
   - No native consolidation/observations or production mutation.
2. Run a small hardened pilot first:
   - Example: `--batch-size 5 --max-llm-calls 2` over the sample.
   - If credential-like records are skipped before LLM, coverage denominator is prompted records, not total sample rows.
3. Verify pilot:
   - `coverage_ok=true`
   - `missing_scores_from_llm=0`
   - Hindsight document/node/observation counts unchanged.
4. If pilot passes, either rerun all sample rows or run only remaining rows to avoid duplicate LLM cost.
5. Combine sidecars, deduplicate by `document_id`, and compute route/risk/value distributions.
6. Generate routing buckets only as local candidate files. Do not submit to Hindsight or mutate production from scorer output directly.

## Recommended routing buckets

- `A_next_repair_candidate`: route in `cluster_revisit/windowed/whole_session`, value_level >= 4, risk not high. Next action: temp-bank candidate only.
- `B_manual_or_risk_review`: route is `manual_review`, risk high, or anomalies present (`tool_log_heavy`, `overlong_multi_scope`, etc.). Next action: manual/cleaning review; no auto retain.
- `C_cluster_later`: cluster/window/whole route with value_level == 3. Next action: defer until same-topic cluster has enough context.
- `D_raw_only_archive`: route `raw_only`. Keep raw evidence; do not retain now.
- `E_wait_defer`: no action now.
- `S_secret_or_credential_manual_review`: skipped credential-like material. Do not send to LLM or retain automatically.

## Handling A bucket

A bucket should become a candidate-only manifest, not a production write. Include:

- `document_id`, `parent_document_id`, `event_date`, `content_sha256`
- source session JSON path
- scorer topic/route/scores/risk/classes/reason
- original retain outcome and hardening overlay
- recommended processing mode:
  - `clustered_repair_note_then_temp_retain`
  - `windowed_temp_retain`
  - `whole_session_temp_retain`
- `hindsight_submit_allowed=false`
- `production_mutation_allowed=false`
- proposed temporary bank name

Then run temp retain + fact-quality gate before considering any production repair.

## Command skeleton

```bash
PY=$HOME/.hermes/hermes-agent/venv/bin/python
RUN=<run_dir>
export MINIMAX_API_KEY=$(python3 - <<'PY'
from pathlib import Path
for p in [Path.home()/'.hermes/.env', Path.home()/'.hermes/profiles/<profile>/.env']:
    if not p.exists():
        continue
    vals={}
    for raw in p.read_text(encoding='utf-8', errors='ignore').splitlines():
        line=raw.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k,v=line.split('=',1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    key=vals.get('MINIMAX_API_KEY') or vals.get('MINIMAX_CN_API_KEY')
    if key:
        print(key)
        break
PY
)

NO_PROXY=api.minimaxi.com,api.minimax.chat,127.0.0.1,localhost \
no_proxy=api.minimaxi.com,api.minimax.chat,127.0.0.1,localhost \
$PY ~/.hermes/scripts/hindsight_review_backlog_llm_scorer.py \
  --input <sample.jsonl> \
  --score-output <sidecar.jsonl> \
  --summary-json <summary.json> \
  --batch-size 5 \
  --max-llm-calls 2 \
  --cadence weekly-hardened-pilot \
  --execute-score \
  --confirm-score score-review-backlog \
  --llm-model MiniMax-M2.7 \
  --llm-base-url https://api.minimaxi.com/v1 \
  --llm-api-key-env MINIMAX_API_KEY \
  --json
```

Notes:
- Use the Hermes venv Python to avoid unrelated conda import noise.
- Load profile-specific MiniMax key without printing it; never put API keys in command arguments.
- Keep provider direct for MiniMax via `NO_PROXY`.
