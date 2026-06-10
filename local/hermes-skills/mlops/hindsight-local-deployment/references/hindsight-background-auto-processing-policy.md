# Hindsight background auto-processing policy

Context: user realized Hindsight background automatic processing (observations/consolidation) is important, but wants it enabled without causing raw/offline pipeline blow-up or degrading precision.

## Core policy

Use Hindsight native background consolidation/observations as the primary high-level memory path, but feed it processed facts, not raw transcripts.

Corrected position (2026-05-08): Hindsight official design does support automatic after-retain consolidation/observations when `enable_observations=true`, consolidation worker slots are available, the LLM provider is usable, and the queue is healthy. If observations are not running in this environment, treat that as a deliberate safety configuration (`enable_observations=false`, `worker_consolidation_max_slots=0`) rather than a Hindsight capability gap.

Recommended layering:
1. Raw conversation / SQLite transcript -> Hindsight retain extracts structured facts, entities, links, graph/temporal/semantic indexes. Retain is not a raw dump.
2. Native background consolidation consumes facts (`experience`/`world`) and emits evidence-grounded observations.
3. Reflect remains native and paid/strong-model by explicit/manual or controlled window. It is for reasoning/querying, not the default persistent-observation writer.
4. Mental models hold a small curated set of stable high-value reflect outputs / user models / SOPs.

Avoid: raw transcript -> offline full consolidation directly. That tends to increase calls, lower precision, and create maintenance burden.

Offline pipeline role after this correction: audit/eval/conflict lineage/wiki candidates/temporary compatibility, not the default high-level source of truth competing with native observations. Do not run v2 canonical publish as the routine answer to “update Hindsight” unless the user explicitly asks for the legacy offline publish path.

## Cost/call-count controls

Call count is mainly controlled by `consolidation_llm_batch_size`, not by `MAX_MEMORIES_PER_ROUND` alone.

Current quality-first defaults for this environment:
- `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND=50` (official in Hindsight >=0.5.3 / 0.6.x; older local wrappers used `MAX_MEMORIES_PER_JOB`)
- `HINDSIGHT_API_CONSOLIDATION_BATCH_SIZE=50`
- `HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=50`
- `HINDSIGHT_API_CONSOLIDATION_SOURCE_FACTS_MAX_TOKENS=4096`
- `HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=1` only inside controlled paid/native window

Important distinction:
- `MAX_MEMORIES_PER_ROUND` prevents one native round from consuming the full backlog; `MAX_MEMORIES_PER_JOB` was a legacy local-wrapper name.
- `CONSOLIDATION_LLM_BATCH_SIZE` determines how many facts can be packed into one LLM call.
- Native Hindsight groups by exact tag set, so 50 facts can still become multiple LLM calls if split across tag groups.

## Background modes

Daily/local mode:
- provider: Ollama/local
- observations: false
- consolidation worker slots: 0
- safe for recall and local fallback; no paid background spend.

Controlled paid/native background window:
- preflight first: health, provider, queue empty, failed=0, active payload_null=0
- enable paid provider and native observations only through wrapper/guard
- run 1 job / 50 facts first
- monitor operation until completion
- restore normal-local in `finally`

Preferred runner:
```bash
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py run-native-consolidation-paid --jobs 1 --facts-per-job 50
```
Default is dry-run. Paid execution requires:
```bash
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py run-native-consolidation-paid \
  --jobs 1 \
  --facts-per-job 50 \
  --execute \
  --confirm run-native-paid-consolidation
```

## First guarded native paid window result (2026-05-08)

Executed:
```bash
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py run-native-consolidation-paid \
  --jobs 1 --facts-per-job 50 --execute --confirm run-native-paid-consolidation
```

Observed:
- Preflight OK; active payload_null=0; pending/processing/failed=0 before start.
- Dry-run estimate for 50 facts: 3 tag groups / ~3 LLM calls; full backlog at that moment: 6144 facts / ~187 tag groups/calls.
- Native operation completed in ~150s and wrapper restored normal-local.
- Backlog reduced 6144 -> 6094 native candidates.
- Observations increased 356 -> 392 (+36); total_nodes 6500 -> 6536.
- Final state: provider=ollama, `enable_observations=false`, consolidation slots=0, pending/processing/failed=0.

Quality warning / RCA from spot check:
- The first 50 facts came from the oldest unconsolidated SQLite-import facts, not from a curated clean topic window: 5 `hermes-sqlite::day-topic::*` documents across 2026-03-15/21/24/25 and 3 sorted tag scopes.
- Native Hindsight behaved as designed: it fetches unconsolidated `experience/world` facts ordered by `created_at ASC`, groups by exact sorted tag set, then recalls existing observations with strict tag matching. Newer upstream uses `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_ROUND`; the older local patch used `HINDSIGHT_API_CONSOLIDATION_MAX_MEMORIES_PER_JOB=50`. Neither chooses cleaner facts.
- `proof_count=1` on new observations is partly an implementation issue: `_create_observation_directly()` inserts `proof_count=1` even when `source_memory_ids` contains multiple facts. Use `array_length(source_memory_ids,1)` for quality audit. Still, most first-window observations were single-source, so the low-synthesis warning remains real.
- There were zero pre-existing observations with the exact native scopes `general/hermes/incremental/sqlite`, `hermes/incremental/openclaw/sqlite`, or `egomotion4d/hermes/incremental/sqlite`; offline-v2 observations use different `canonical/offline-v2/...` tags. Therefore native consolidation mostly created new observations instead of updating/merging existing ones.
- Tag/scope pollution root cause is mostly our SQLite import, not native consolidation: `import_sqlite_to_hindsight.py` posts tags as `["hermes", "sqlite", "incremental", bundle.topic]` and sets no `observation_scopes`. `bundle.topic` is produced by naive substring counts over whole session content.
- The topic classifier has known failure modes: the short keyword `ate` for ATE matches substrings in words such as `patent`, `statement`, `date`, etc.; and OpenClaw approval/gateway/tooling noise can dominate the topic even when the durable content is patent/OA1. There is no `patent` topic, so patent facts can end up under `general`, `egomotion4d`, or `openclaw` scopes.
- With `observations_mission=None`, Hindsight uses its default mission: "Track every detail ... prefer specifics over abstractions." That encourages one-off operational facts, file-read approvals, patent document details, and similar low-durability facts to become observations.
- Do not scale directly to `--jobs 5` until retain tags / observation_scopes / topic classifier / observations_mission are fixed or a better source-filter window is selected.
- Next native test should either target a cleaner tag/scope subset or first improve retain tags (`domain:*`, `topic:*`, `project:*`, `source:hermes`) and then re-run a small 50-fact quality sample.

- Do not reduce `consolidation_llm_batch_size` to 8 when the user asks to save calls; 8 improves prompt compactness but increases call count substantially.
- Do not process raw transcripts directly in offline consolidation by default; use processed facts first.
- Keep `payload_null` guard fail-closed before paid windows. Active `payload_null` should be quarantined, not deleted from source data.
- Weak local models may complete consolidation but produce low-quality observations; for high-level observations, use strong paid model in small controlled windows.

## Decision rule

If the goal is "enable useful background processing": use native Hindsight background consolidation on processed facts with guarded paid windows.

If the goal is "minimize paid calls": increase fact packing up to the quality limit (currently 50), not raw aggregation or unlimited full-backlog runs.

If 50-fact quality is poor: do not build a complex local replacement consolidation system; prefer waiting for/upgrading Hindsight upstream or improving tags/source grouping.