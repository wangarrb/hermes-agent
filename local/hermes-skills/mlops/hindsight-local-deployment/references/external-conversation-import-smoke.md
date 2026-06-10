# External Conversation Import Smoke Workflow

Use when importing third-party conversations (chat-memo exports, OpenClaw `lcm.db`) into Hindsight. Keep this path manual and bank-isolated until explicitly promoted.

## Durable lessons

- Do not over-specialize tag rules. The user prefers practical, general rules; "good enough" tagging is better than brittle heuristics that lose generality.
- External imports are not the daily Hermes session pipeline. Keep `manual_only=true`, `daily_pipeline_integrated=false`, and use a separate smoke bank first.
- Full validation means real retain + native consolidation/observations + recall + isolation checks, not just dry-run.
- For OpenClaw, remove untrusted metadata and tool traces before retain:
  - `Conversation info (untrusted metadata)` / `Sender (untrusted metadata)` / `System (untrusted ...)` blocks.
  - structured assistant JSON `thinking`, `toolCall`, `toolResult` items; keep only human-visible text/content.
- Use source/platform tags for provenance, but do not let generic source headers (e.g. `OpenClaw conversation N`, `Platform: OpenClaw`) drive semantic tags.
- Include tag/cleaning versions in metadata and submit-state keys so revised filters do not get skipped as unchanged.

## Current helper scripts

```bash
python3 $HERMES_HOME/scripts/hindsight_external_manifest.py --help
python3 $HERMES_HOME/scripts/hindsight_external_retain_runner.py --help
```

Key options:

```text
--source chat-memo | openclaw-lcm
--path / --chat-memo-dir       # chat-memo txt directory
--db ~/.openclaw/lcm.db        # OpenClaw source
--bank-target <smoke_bank>
--include-observation-scopes
--sample-records N
--sample-action production
```

## Smoke flow template

1. Build a sampled manifest:

```bash
python3 $HERMES_HOME/scripts/hindsight_external_manifest.py \
  --source chat-memo \
  --path /path/to/chat-memo-dir \
  --bank-target external_chatmemo_flow_smoke \
  --min-file-age-seconds 0 \
  --include-observation-scopes \
  --sample-records 3 \
  --sample-action production \
  --json | tee /tmp/external_chatmemo_manifest_summary.json
```

OpenClaw variant:

```bash
python3 $HERMES_HOME/scripts/hindsight_external_manifest.py \
  --source openclaw-lcm \
  --db ~/.openclaw/lcm.db \
  --bank-target external_openclaw_flow_smoke \
  --include-observation-scopes \
  --sample-records 3 \
  --sample-action production \
  --json | tee /tmp/external_openclaw_manifest_summary.json
```

2. Inspect sampled records before writing. Rehydrate omitted content if needed via `hindsight_external_retain_runner.record_to_memory_item()` and verify no secrets/tool traces.

3. Dry-run retain:

```bash
MANIFEST=$(python3 - <<'PY'
import json
print(json.load(open('/tmp/external_chatmemo_manifest_summary.json'))['paths']['manifest'])
PY
)
python3 $HERMES_HOME/scripts/hindsight_external_retain_runner.py \
  --manifest "$MANIFEST" \
  --bank external_chatmemo_flow_smoke \
  --batch-size 2 \
  --limit 3 \
  --ignore-submit-state \
  --json
```

4. Execute retain only to the smoke bank:

```bash
python3 $HERMES_HOME/scripts/hindsight_external_retain_runner.py \
  --manifest "$MANIFEST" \
  --bank external_chatmemo_flow_smoke \
  --batch-size 2 \
  --limit 3 \
  --ignore-submit-state \
  --execute \
  --confirm retain-hindsight-external-manifest \
  --wait-timeout-s 600 \
  --poll-s 5 \
  --json | tee /tmp/external_chatmemo_retain_result.json
```

5. Wait for native consolidation/observations:

```bash
python3 $HERMES_HOME/scripts/hindsight_wait_native_consolidation.py \
  --bank external_chatmemo_flow_smoke \
  --timeout-s 1800 \
  --poll-s 10 \
  --block-on-failed-consolidation \
  --json | tee /tmp/external_chatmemo_wait.json
```

6. Verify:

- `pending_consolidation == 0`
- `pending_operations == 0`
- `failed_operations == 0`
- `failed_consolidation == 0`
- `total_observations > 0` when `--include-observation-scopes` was used and bank observations are enabled.
- recall finds relevant facts in the smoke bank.
- `hermes` main bank query for `external-chatmemo` / `external-openclaw` returns 0.

## Representative validated results

Session validated the flow with:

```text
chat-memo smoke bank: external_chatmemo_flow_smoke_v7
  3 docs -> 22 nodes -> 13 observations; pending_consolidation=0; failed=0

OpenClaw smoke bank: external_openclaw_flow_smoke_v8
  3 docs -> 33 nodes -> 23 observations; pending_consolidation=0; failed=0
```

Final evaluation report from that session:

```text
/home/wyr/.hermes/hindsight/external_import/evals/20260518-2108-external-flow-smoke-final-v7.json
```

## Pitfalls

- Hindsight bank `stats.operations_by_status` can include parent rows; use `/operations?exclude_parents=true` for operation counts.
- A short, context-poor third-party conversation can be imported correctly but still produce weak recall. Do not treat weak recall on weak source text as an import failure.
- Intermediate smoke banks from experiments should not be deleted without explicit confirmation.
