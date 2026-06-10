# Hindsight Full Pipeline Step Sequence

Complete step sequence for `hindsight_memory_pipeline.py full` mode (verified 2026-05-27):

## Step Order

1. **preflight** — environment checks, config validation, tuning file
2. **status** — DB/API health, current queue state
3. **queue_drain_before_daily** — wait for any existing async queue before starting paid LLM work
4. **session_manifest** — scan new/modified sessions, build manifest JSONL
5. **retain_session** — manifest incremental retain → extract source facts
6. **daily_reflect** — offline daily consolidation (calls LLM per topic per day)
7. **native_consolidation_drain_after_daily** — wait for native source-fact consolidation to complete
8. **v2_rebuild (daily)** — daily facts → canonical observations
9. **native_consolidation_drain_before_weekly** — wait before weekly stage
10. **weekly_reflect** — offline weekly consolidation (calls LLM per topic per week)
11. **native_consolidation_drain_after_weekly** — wait after weekly reflect
12. **v2_rebuild (weekly)** — weekly facts → canonical observations
13. **conflict_audit** — detect entity alias fragmentation, recall weakness, schema issues
14. **repair_zone_proposals** — generate repair proposals (can `--skip-repair-zone`)
15. **proposal_review** — LLM-advisory + human review (can `--skip-proposal-review`)
16. **wiki_auto_maintenance** — long-cycle wiki candidate maintenance (requires `--include-wiki`)

## Key CLI Flags

```
--execute --confirm run-hindsight-pipeline   # required for execution
--include-wiki                                # include wiki step (16)
--skip-daily                                  # skip steps 3-8, start from weekly
--skip-repair-zone                            # skip step 14
--skip-proposal-review                        # skip step 15
--no-wait-native-consolidation                # skip consolidation drain gates
--history all                                 # force re-retain of all sessions
```

## Failure Recovery

When the pipeline fails mid-way:
- Steps 1-8 may succeed even if later steps fail (e.g., 403 on LLM, 502 on recall API)
- Check the pipeline output JSON for `"status": "ok"` per step
- Resume with `--skip-daily` if daily stage completed
- For eval-only failures (502 due to container restart), just rerun eval separately
- Do NOT rerun from the beginning when earlier stages already completed

## Common Failure Patterns

| Symptom | Cause | Fix |
|---------|-------|-----|
| 403 on LLM calls | Wrong model name (e.g., `deepseek/deepseek-v4-flash` instead of `deepseek-v4-flash`) | Check `.env` HINDSIGHT_OFFLINE_LLM_MODEL |
| 403 on LLM calls | API key expired/no balance | Verify key and balance on provider dashboard |
| 502 on recall API | Container just restarted, bge-m3 loading (~25s) | Wait for API readiness, then rerun the failed step |
| Pipeline exits 1 at eval | Same as 502 — eval calls recall during restart window | Separate eval run after container is stable |
| Unknown --llm-profile | `topenrouter` is not a valid Hindsight profile name | Use `deepseek-v4-flash` instead, override base_url via `.env` |

## Timing Reference (2026-05-27, topenrouter deepseek-v4-flash)

- Session manifest + retain: ~25 min
- Daily reflect: varies by LLM speed
- Weekly reflect: varies by LLM speed
- Eval: ~1-2 min (when API is ready)
- Wiki maintenance: ~3 min

Full pipeline with wiki: multi-hour depending on LLM throughput and consolidation backlog.
