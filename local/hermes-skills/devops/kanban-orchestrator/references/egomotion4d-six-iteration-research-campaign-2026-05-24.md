# Egomotion4D Six-Iteration Research Campaign (W13–W16)

## Campaign scope

Completed 2026-05-24. Six iterations (W13=iter3, W14=iter4, W15=iter5, W16=iter6/final) on the egomotion4d board, investigating three research lines:
- **A-line**: Does policy-based multi-source TSDF fusion beat best single source on strict heldout metrics?
- **Ground-line**: Does a ground reconstruction method beat best single frontend under an independent denominator?
- **RoMA/C-line**: Does RoMA/SfM sparse guidance improve pose/geometry or downstream fusion?

## Final conclusions (W16-R)

All three lines closed as no-claim/weak-prior. Automatic iteration stopped; no W17 published.

| Line | Final claim label | Bottleneck | Reopen condition |
|------|------------------|-----------|-----------------|
| A-line | PARETO_DIAGNOSTIC | Fusion improves coverage/depth on 6/6 folds, but strict near-field p90 gates fail on scene4/scene22; fusion-policy precision, not source contract readiness | Tune policy weights with heldout-blind validation; increase near-field metric precision |
| Ground | NO_CLAIM_DEDICATED_GROUND_BUG_OR_BAD_SIGNAL | Best single (vggt4d) beats generic multi-source and dedicated ground surface; dedicated has height_scale_or_coordinate_frame_mismatch (p90=376m, std=237m) | Fix dedicated ground coordinate frame/scale; validate height units match BEV denominator |
| RoMA | WEAK_PRIOR_ONLY_FINAL | 37/37 valid pairs, inlier ratio 1.0, but median reprojection 20.3px and only 2/37 pairs improve | Reduce stride (adjacent pairs already tested), use different RoMA config, or acquire independent pose GT |

## Key intermediate gates that passed

| Gate | Wave | Result | Why it mattered |
|------|------|--------|----------------|
| W13-A0 source contract gate | iter3 | CLAIM_PASS 8/8 contracts, no scene0 fallback | Proved scene4/scene22 manifest infrastructure is sound |
| W14-A0 source-load confidence gate | iter4 | CLAIM_PASS_SOURCE_LOAD_READY | Identified pi3x/dage strict-ready, vggt4d/any4d uint8_0_255 |
| W14-G0 ground denominator triage | iter4 | CLAIM_PASS 4269 BEV cells (sam2) | Reopened ground line with independent denominator |
| W15-A0 pose/gauge repair | iter5 | CLAIM_PASS 8/8, no scene0 pose leak | Removed the scene0_fused_poses.txt blocker |

## Patterns observed across all iterations

1. **Gate-first, benchmark-second**: Every wave starts with a blocking gate (source contracts, confidence, pose/gauge) before running the benchmark. Skipping gates wastes gpuserver time.
2. **NO_CLAIM is a valid iteration outcome**: An iteration with NO_CLAIM but validator-clean artifacts and ranked bottlenecks is a real iteration — it reduces uncertainty even without a positive claim.
3. **Critic evidence-fail catches provenance issues**: W13-G1R and W13-C1R both returned CODE_PASS_EVIDENCE_FAIL because run_manifest.json was terse and scripts were untracked. This pattern repeated enough that W14/W15 tasks all required richer manifest/script evidence.
4. **DeepSeek-TUI listener instability**: Most implementer tasks saw 2-4 reclaim cycles due to DeepSeek interactive listener stopping before completion. Planner had to reclaim and complete many tasks manually. This is a known operational issue, not a research finding.
5. **Confidence normalization must be explicit**: W14-A0 discovered vggt4d/any4d confidence is uint8 [0,255], not [0,1]. The accepted transform is divide-by-255, not minmax or clipping. Any deviation from this is an evidence failure.
6. **Scene-specific pose paths are mandatory**: All scene4/scene22 manifests originally pointed to scene0_fused_poses.txt. Repairing this to scene-specific paths was a full wave (W15-A0). Any future multi-scene benchmark must audit pose provenance first.

## Board stats after campaign

330 done, 0 ready, 0 running, 1 blocked (user test task t_ea1cc133, unrelated to research). All W13-W16 campaign tasks completed. Board is idle between campaigns.

## Post-campaign board state

After W16-R completed, separate campaigns (GFPR, GGPT, P1-SV, etc.) were created with different task structures. This is a normal campaign transition — the board hosts both the completed W13-W16 campaign tasks and tasks from newer campaigns. Watchdog巡检 for the W13-W16 scope should immediately detect completion (all wave tasks `done`) and return `[SILENT]`, while ignoring tasks from other campaigns.

## Post-campaign operational patterns (2026-05-25 watchdog observation)

After W16-R completed, some non-mainline tasks showed **stale-lock reclaim loops**:
- Tasks with 3+ consecutive stale_lock reclaims over 6+ hours
- Root cause: listener/TUI processes die or hang before completing, lock expires, gets reclaimed, next run also fails

**Watchdog guidance**: When a task has 3+ consecutive stale_lock reclaims, do NOT dispatch. Flag for human intervention and add a comment noting the reclaim count and suggesting the pane/profile needs debugging.

**Non-campaign stale-lock nuance**: If the stale-lock task is outside your monitored campaign scope, just note it as an out-of-scope anomaly. Do NOT create fix tasks or dispatch cycles for it — it belongs to a different work stream.

**Mass-stall pattern (2026-05-25 12:17 watchdog)**: When one listener PID claims 5+ tasks and all show zero heartbeats for 3+ hours, the listener itself is the bottleneck (alive but not processing). The fix is to reclaim ALL stalled tasks at once, then dispatch one at a time to verify the next worker actually produces heartbeats before scaling up.

## Watchdog forward-trace technique (2026-05-25 observation)

Cron job configs often list only the current wave's task IDs (e.g., W13-A0 through W13-R). When all of those are `done`, the watchdog must trace forward through the final review's `children` to find the next wave's task IDs (e.g., W14-*), then `show` those, and repeat until reaching the campaign's final review (e.g., W16-R). Only after confirming the entire pipeline from first wave to final review is `done` should the watchdog return `[SILENT]`. This avoids false-negatives where only the first wave's tasks are done but later waves are stalled or unstarted.

**Efficient forward-trace sequence** (validated 2026-05-25):
1. `hermes kanban --board <slug> stats` → if ready=0, running=0, blocked is only unrelated tasks → campaign is likely complete
2. `show` each tracked task ID from cron config → confirm all `done`
3. For each `done` final-review task, check `children` → `show` those children
4. Recursively follow children chains (W13-R→W14→W15→W16-R) until reaching the campaign's terminal review
5. Confirm terminal review is `done` and either published new-campaign children or explicitly stopped iteration
6. `dispatch --dry-run --json` as final confirmation (should show spawned=[], reclaimed=0)
7. Return `[SILENT]`

**Key efficiency**: When stats show 0 ready/0 running, do NOT `show` all 330+ done tasks. Only `show` the tracked task IDs and their children chains. Skip unrelated done tasks entirely.
