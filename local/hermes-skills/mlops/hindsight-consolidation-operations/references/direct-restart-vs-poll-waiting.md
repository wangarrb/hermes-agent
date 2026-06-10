# Direct Restart vs Poll-Waiting for Consolidator Patch Application

## Why Poll-Waiting Fails

Hindsight's native consolidation auto-re-queues (`submit_async_consolidation`) at the end of
every round that hits `max_memories_per_round`. The gap between "current round finishes" and
"next round starts processing" is **sub-second**. Even tight 3-second polling consistently
misses this window:

```
[11:10:23] processing=1 pending=0
[11:10:26] processing=1 pending=0   ← next round already started
[11:10:29] processing=1 pending=0
...
[11:15:57] processing=1 pending=0   ← still processing, round 2
[11:16:00] processing=1 pending=0
[11:16:03] processing=1 pending=0
...
[11:16:33] processing=1 pending=0   ← round 2 ends
[11:16:36] processing=1 pending=0   ← round 3 already started (3s gap!)
```

The watchdog (`hindsight_restart_on_idle.py`) was tested for 11 minutes and never caught
the idle window because 8-way parallel rounds finish fast and re-queue immediately.

## Established Practice: Direct Restart

The production scripts `patch_json_parser_and_restart()` and
`patch_hindsight_container_and_restart()` do `docker restart hindsight` with **no queue
check, no idle wait**. This has been the operational pattern since May 2026 and has never
caused data loss.

### Why It's Safe

1. **No data loss**: Unfinished `consolidated_at` marks for the current batch are simply
   missing; the next consolidation picks up those memories as `consolidated_at IS NULL`
   and reprocesses them.

2. **Deduplication prevents double-work**: The Hindsight worker's `dedupe_by_bank=True`
   prevents TWO consolidations running on the same bank. After restart, one stuck
   `processing` op blocks the bank. Unstick it (see `restart-stuck-op-recovery.md`) and
   submit fresh consolidation.

3. **Observations are idempotent**: The patched consolidator handles create/update/delete
   with dedup guards. Reprocessing the same memories produces the same observations.

### When to Use Poll-Waiting Instead

Only when:
- The user explicitly says "wait for this to finish" (quality-conscious choice)
- The consolidation is nearly complete (e.g., 7 remaining) and you want clean completion
- You're NOT in a rush and the next round hasn't started yet

Otherwise: direct restart → unstick → resubmit.
