# Pipeline: session retain fails inside `full` — retry & resume

Observed 2026-05-18: `hindsight_memory_pipeline.py full --execute` failed at `retain_session_manifest` stage with `RetainOperationFailed`. The pipeline's `finally` block correctly restored `normal-local` mode and attempted post-restore queue drain (which timed out at 600s). This pattern differs from an external SIGTERM (`rc=-15`) — the pipeline self-terminated on a business error.

## Symptom

```
RetainOperationFailed: retain async operation failed: {'<op-uuid>': 'failed'}
session manifest retain failed with code 1; restoring local mode
ERROR: post-restore queue wait failed for hermes: queue did not drain within 600s
```

Post-restore queue drain timeout is expected and harmless: after a failed retain, Hindsight may still be consolidating earlier successful submits, and the drain has no fixed deadline for the `finally` block.

## Root cause

Failed ops had error: `generator didn't stop after athrow()`. This is a MiniMax LLM streaming generation bug — the generator raised/cancelled mid-stream before yielding all structured output. Not a content/data issue. Each affected op had `retry_count=3`, `next_retry_at` in the past, so no auto-retry would fire.

## Resume sequence

1. **Don't rerun full from the beginning.** The `finally` block already restored local mode and did the post-restore queue drain. Manifest was already submitted (partially successfully). submit_state.json already marks successful items as submitted, so a second full run would skip already-submitted sessions — but the failed ops' documents would NOT be in submit_state (runner doesn't record submit_state until the op completes).

2. **Identify failed ops** that are worth retrying:
```bash
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&created_after=<TIMESTAMP>&limit=10' | python3 -c '
import json,sys
for op in json.load(sys.stdin).get("operations",[]):
    if "generator didn" in (op.get("error_message") or ""):
        print(op["id"])
'
```

3. **Retry via API** (not via pipeline; the pipeline's wrapper would switch provider again):
```bash
curl -s -X POST "http://127.0.0.1:8888/v1/default/banks/hermes/operations/${OP_ID}/retry" \
  -H 'Content-Type: application/json' \
  -d '{"confirm":"retry-hindsight-operation"}'
```

4. **Wait for retries + consolidation**:
```bash
python3 $HERMES_HOME/scripts/hindsight_wait_native_consolidation.py \
  --api-url http://127.0.0.1:8888 --bank hermes --timeout-s 3600 --poll-s 60 --json
```

5. **Resume pipeline after retain** by skipping the already-done retain stage. Use `full --skip-daily` if daily retain/reflect already completed, or run weekly/V2 stages manually:
```bash
# If daily retain completed (including retries), skip to weekly:
python3 $HERMES_HOME/scripts/hindsight_memory_pipeline.py full \
  --skip-daily --include-wiki \
  --execute --confirm run-hindsight-pipeline
```

## Important distinction: partial pipeline failure vs SIGTERM

| Pattern | Signal | Restore state | Resume |
|---------|--------|---------------|--------|
| Full self-failure (this bug) | `exit code 1`, `RetainOperationFailed` | `finally` ran, local mode restored | Skip completed stages, retry failed ops first |
| External SIGTERM | `rc=-15`, watchdog | Process may or may not have `finally` | Inspect last completed stage, resume from there |

## Why post-restore drain timed out

After `finally` runs `hindsight_minimax_import.py` restore-local mode, the script calls `wait_queue_drained()` with a 600s timeout. If Hindsight is still processing consolidation from previous successful retain batches (not from the retry), this can time out. This is cosmetic — it runs outside the mutating path and local mode is already restored. The pipeline's exit code is already 1 from the retain failure; the drain timeout is secondary noise.