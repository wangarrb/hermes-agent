# Failed Operation Triage

When the user asks "do these failed operations need handling?" or you see failed ops in a daily report, follow this triage flow.

## Quick Decision

**`pending_consolidation=0` + `failed_base=0` + no ongoing pipeline** → failed ops are historical noise. Do not block pipeline execution.

## Step-by-Step

### 1. Check bank-level health first

```bash
python3 /home/wyr/.hermes/scripts/hindsight_consolidation_status.py --skip-psql --json
```

Key fields from the response:
- `bank_summary.operations_by_status` — shows completed vs failed counts
- `checks.bank_stats.data.pending_consolidation` — if 0, no unconsolidated source facts
- `checks.bank_stats.data.failed_consolidation` — counts consolidation-specific failures
- `checks.bank_stats.data.failed_operations` — total failed ops
- `checks.bank_stats.data.last_consolidated_at` — when consolidation last completed OK

### 2. Fetch failed operation details

```bash
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&limit=20'
```

Note: `curl` output goes to stdout; use `| python3 -c` or pipe to `head` for viewing. The API returns JSON directly — if you get "FAILED: curl" that means the curl itself failed (likely shell redirect issue), not the API. Use `-vs` for verbose debugging.

### 3. Classify by error message

| Error message | Root cause | Blocks? | Action |
|---|---|---|---|
| `generator didn't stop after athrow()` | LLM generator boundary bug in Hindsight 0.6.1 AsyncIterator protocol. Has `retry_count=3` | No. Already exhausted retries. `next_retry_at` in past. | None needed |
| `FK violation: memory_links.to_unit_id` | Temporal link FK guard not applied yet. `patch_hindsight_retain_temporal_fk_guard.py` fixes this | No for new ops (patch is applied on restore). Historical remain failed | Can delete if cosmetic; not needed |
| `Failed to search memories:` | bge-m3 recall/search temporarily unavailable during embedding restart | No. Had `retry_count=3`, exhausted | None |
| `Object of type datetime is not JSON serializable` | Python/datetime serialization edge case in earlier version | No. Retries exhausted | None |
| Other (unrecognized) | Inspect `error_message` for clues | Depends on `pending_consolidation` | If blocking, consider `POST /consolidation/recover` or `curl -X POST` |

### 4. Check if failed ops block the pipeline

Blocking conditions:
- `pending_consolidation > 0` AND `processing_ops == 0` AND no new consolidation starts → consolidation stall
- `failed_base > 0` (failed base memory units, not just operations) → source-level failures need triage
- Active pipeline waiting on native consolidation drain that never completes

Non-blocking conditions:
- `pending_consolidation = 0` — no unconsolidated facts waiting
- Failed ops are `retry_count=3, next_retry_at` in the past — system has given up on them
- `failed_base = 0` — no source units failed, only operations

### 5. When to clean up

Deleting old failed ops is purely cosmetic. Only do it when explicitly asked:

```bash
# Get operation IDs
curl -s 'http://127.0.0.1:8888/v1/default/banks/hermes/operations?status=failed&exclude_parents=true&limit=20' | \
  python3 -c "import json,sys; ops=json.load(sys.stdin).get('operations',[]); [print(o['id']) for o in ops]"

# Delete one (requires confirm token)
curl -X DELETE "http://127.0.0.1:8888/v1/default/banks/hermes/operations/<id>"
```

## Example: This session's triage (2026-05-18)

```
bank status: pending_consolidation=0, failed_base=0, failed_operations=8
```

Found 7 ops classified as non-blocking:
- 2x consolidation (`generator didn't stop after athrow()`) — 05-17, retries exhausted
- 1x retain (FK violation) — 05-14, fixed by patch, won't recur
- 4x consolidation (`Failed to search memories` / `datetime`) — 05-12, retries exhausted

Verdict: **Blocking: 0. Historical noise: 7. No action needed.** Pipeline can proceed.

## Common traps

- `curl` returning empty or "FAILED" on first try: retry with `-vs` for debugging. The API returns valid JSON even with large error messages.
- Don't judge blocking by operation count alone. 8 failed ops with `pending_consolidation=0` is noise; 1 failed op with `pending_consolidation=50` is a signal.
- `next_retry_at` in the past means auto-retry has given up. These ops stay `failed` permanently unless manually retried.
- `retry_count=3` is the system default. Don't try to manually retry ops that already failed 3 times — the same error will recur unless the root cause is fixed.