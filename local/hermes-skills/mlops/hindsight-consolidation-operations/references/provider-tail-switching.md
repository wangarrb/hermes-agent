# Provider tail switching for slow Hindsight consolidation

Session-derived pattern from a long offline/session consolidation drain that reached the final tail and then became LLM-bound.

## Symptoms

- Backlog is small but each consolidation batch is very slow.
- Logs show repeated slow LLM calls and occasional timeout / APIConnectionError.
- `failed_base=0`, but `unconsolidated_base` stops dropping for long windows.
- Tail batches can be tiny yet expensive, e.g. 1–2 memories requiring several observation-scope calls with multi-minute latency.

## Diagnosis

Check provider and recent LLM behavior before assuming the drain code is stuck:

```bash
# Confirm current provider/model/env inside container; redact keys in reports
docker inspect hindsight --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | grep -E 'HINDSIGHT_API_(LLM|CONSOLIDATION_LLM|DB_POOL)'

# Inspect recent logs for provider initialization, slow calls, timeouts, and worker stats
docker logs --since 60m hindsight 2>&1 \
  | grep -E 'OpenAI-compatible client initialized|slow llm|timeout|APIConnectionError|WORKER_STATS|CONSOLIDATION'

# Check drain/monitor tmux panes without spawning excessive DB queries
tmux capture-pane -t hindsight-obs-monitor -p -S -80 | tail -80
tmux capture-pane -t hindsight-obs-drain -p -S -80 | tail -80
```

Useful interpretation:
- Slow provider tail is likely when `failed_base=0` and logs continue to show active consolidation, but recent LLM calls are hundreds of seconds each.
- A failed async `consolidation` operation is not automatically a data failure; distinguish `failed:consolidation` from `failed_base`.

## Safe provider switch pattern

Do not hard-kill an active consolidation just to switch providers. Use an idle switcher / watcher:

1. Wait until current active consolidation finishes or idles (`processing:consolidation=0` and no pending consolidation that would immediately claim).
2. Stop the old drain/pool watcher so it cannot queue the next operation during restart.
3. Recreate/restart the Hindsight container with the new provider env.
4. Re-apply local patches into the container after recreation:
   - patched consolidator,
   - config/tuning files,
   - DB pool cap patch if used.
5. Verify syntax/health.
6. Restart monitor and drain.
7. Confirm logs show the intended provider/model.

In the observed run, switching the tail from GLM to MiniMax was useful when GLM showed many slow calls/timeouts. The target env looked like:

```text
HINDSIGHT_API_LLM_PROVIDER=minimax
HINDSIGHT_API_LLM_MODEL=MiniMax-M2.7
HINDSIGHT_API_LLM_BASE_URL=https://api.minimaxi.com/v1
HINDSIGHT_API_CONSOLIDATION_LLM_PROVIDER=minimax
HINDSIGHT_API_CONSOLIDATION_LLM_MODEL=MiniMax-M2.7
HINDSIGHT_API_CONSOLIDATION_LLM_BASE_URL=https://api.minimaxi.com/v1
HINDSIGHT_API_DB_POOL_MAX_SIZE=60
```

Verify with logs:

```text
OpenAI-compatible client initialized: provider=minimax, model=MiniMax-M2.7
```

## Post-switch watch conditions

Watch 10–20 minutes after the switch:

- slow LLM call duration drops relative to the previous provider;
- timeout / APIConnectionError count does not keep climbing;
- `unconsolidated_base` eventually drops after job commit;
- `failed_base` remains 0;
- new `failed:consolidation` entries are explained and do not repeat.

## Reporting guidance

Be explicit that the switch is a mitigation for LLM-bound tail latency, not proof the whole drain is complete. Report:

- old provider symptom (slow calls/timeouts),
- new provider/model actually live,
- current backlog and failed_base,
- new operation id if available,
- next watch condition.
