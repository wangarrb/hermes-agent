# Native consolidation gates and half-downgrade rollback

## When this matters

Use this note when Hindsight daily/weekly/full pipeline stages finish their retain/offline reflect work but `/stats.pending_consolidation` remains high. In that state, source `experience`/`world` facts are in the bank but have not all been turned into native `observation` facts or marked `consolidated_at`.

Quality-sensitive stages should not run against a partially consolidated bank unless the user explicitly accepts stale/incomplete observations.

## Correct pipeline ordering

For daily/full runs that ingest or reflect new source facts:

1. session retain / offline reflect submits data into Hindsight.
2. Wait for submitted retain operation IDs if applicable.
3. Wait for native source-fact consolidation drain:
   - `pending_consolidation <= 0` normally;
   - child operations `pending=0` and `processing=0` with `exclude_parents=true`;
   - `failed_consolidation=0` or explicitly triaged.
4. Only then run V2 rebuild, conflict audit, repair-zone proposal build, and proposal review.

For `full --skip-daily` resume runs, add a gate before weekly reflect because previous daily/session work may still be draining, and another gate after weekly reflect before V2/conflict/proposal.

Current helper:

```bash
python3 $HERMES_HOME/scripts/hindsight_wait_native_consolidation.py \
  --bank hermes --timeout-s 86400 --poll-s 60 --json
```

Escape hatch:

```bash
python3 $HERMES_HOME/scripts/hindsight_memory_pipeline.py full --skip-daily \
  --no-wait-native-consolidation --plan-json
```

Use the escape hatch only for emergency/debug resumes where stale or incomplete observations are acceptable.

## Distinguish the counters

- `pending_consolidation`: source facts not yet native-consolidated. This is the gate for quality completeness.
- `operations?status=pending/processing&exclude_parents=true`: active queue rows. Parent batch rows can be false positives if parents are not excluded.
- Historical `failed` operations: useful for audit, but not always a current blocker. Check `failed_consolidation` and current source backlog.
- `observation.consolidated_at` being null is expected; observations are outputs, not source facts to be consolidated.

## Half-downgrade rollback

Keep the stable 8-way profile unless logs show provider/DB pressure. The rollback is prepared but not automatic.

Preview:

```bash
python3 $HERMES_HOME/scripts/hindsight_minimax_import.py consolidation-half-downgrade
```

Typical result from the 8-way profile:

```text
HINDSIGHT_NATIVE_CONSOLIDATION_LLM_MAX_CONCURRENT=4
HINDSIGHT_NATIVE_CONSOLIDATION_PARALLEL_BATCHES=4
HINDSIGHT_NATIVE_CONSOLIDATION_RECALL_MAX_CONCURRENT=30
```

Apply only after idle unless the user explicitly accepts interruption:

```bash
python3 $HERMES_HOME/scripts/hindsight_minimax_import.py consolidation-half-downgrade \
  --execute --confirm halve-hindsight-consolidation-concurrency
```

The downgrade is consolidation-only: retain/reflect global paid-LLM concurrency can remain unchanged.

## Verification checklist

- `python3 -m py_compile` on changed scripts.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest ... -q` for pipeline/minimax/preflight tests.
- `full --skip-daily --plan-json` shows native gates before weekly and before V2/conflict/proposal.
- `daily --plan-json` shows a native gate before V2.
- Half-downgrade preview is dry-run and does not recreate the container.
- One-shot wait gate snapshot reports current `pending_consolidation`, pending/processing ops, and `ready=false/true` as expected.
