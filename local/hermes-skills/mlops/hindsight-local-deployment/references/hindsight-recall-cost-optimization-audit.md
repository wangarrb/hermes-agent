# Hindsight Recall / Reflect Cost Optimization Audit

## Context

Use when reviewing cost-saving advice that proposes Hindsight env vars for recall caching, conditional recall, dreaming/reflect scheduling, embedding cache, or entity merge scheduling.

## Key finding

Do not accept invented `HINDSIGHT_*` knobs at face value. In the current Hermes + Hindsight setup, many plausible-looking variables are not implemented and silently do nothing if added to env/config.

Verified unsupported in the current Hermes Hindsight provider and Hindsight API container source:

```text
HINDSIGHT_AUTO_RECALL_MODE
HINDSIGHT_RECALL_TRIGGER_KEYWORDS
HINDSIGHT_RECALL_SIMILARITY_FILTER
HINDSIGHT_RECALL_CACHE_ENABLED
HINDSIGHT_RECALL_CACHE_TTL
HINDSIGHT_EMBEDDING_CACHE_ENABLED
HINDSIGHT_EMBEDDING_CACHE_TTL
HINDSIGHT_REFLECT_MODE
HINDSIGHT_REFLECT_TRIGGER_CONDITIONS
HINDSIGHT_REFLECT_BATCH_MODE
HINDSIGHT_REFLECT_FILTER_CHITCHAT
HINDSIGHT_ENTITY_MERGE_REALTIME
HINDSIGHT_ENTITY_MERGE_URGENT_TRIGGER
HINDSIGHT_ENTITY_MERGE_SCHEDULED
HINDSIGHT_ENTITY_MERGE_CRON
```

## Actual current behavior

Hermes Hindsight provider:

- `auto_recall=true` means each completed turn queues a background prefetch for the next turn.
- `memory_mode=context` injects prefetched recall results into context and hides Hindsight tools.
- `recall_prefetch_method=recall` uses Hindsight recall; `reflect` would call LLM and should not be used for daily prefetch.
- `auto_retain=false` prevents online writes, but does not by itself disable auto-recall.

Hindsight recall:

- Generates a query embedding for each recall request.
- Runs semantic/BM25/graph/temporal retrieval depending on config/budget.
- Does not call an LLM when using `recall` rather than `reflect`.

Current deployment already uses local recall primitives:

```text
HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
HINDSIGHT_API_RERANKER_PROVIDER=rrf
```

Therefore, in this environment recall does not burn MiniMax/paid embedding API calls. The real costs are local embedding compute, DB work, and extra main-model input tokens from injected recall context.

## Real supported knobs

Hermes config (`~/.hermes/hindsight/config.json`):

```json
{
  "auto_recall": true,
  "auto_retain": false,
  "memory_mode": "context",
  "recall_budget": "mid",
  "recall_max_tokens": 4096,
  "recall_max_input_chars": 800,
  "recall_prefetch_method": "recall"
}
```

Hindsight API env/config examples:

```text
HINDSIGHT_API_EMBEDDINGS_PROVIDER=local
HINDSIGHT_API_EMBEDDINGS_LOCAL_MODEL=BAAI/bge-small-en-v1.5
HINDSIGHT_API_RERANKER_PROVIDER=rrf|local|flashrank|...
HINDSIGHT_API_RECALL_CONNECTION_BUDGET=<int>
HINDSIGHT_API_REFLECT_MAX_ITERATIONS=<int>
HINDSIGHT_API_REFLECT_WALL_TIMEOUT=<seconds>
HINDSIGHT_API_ENABLE_OBSERVATIONS=true|false
HINDSIGHT_API_WORKER_CONSOLIDATION_MAX_SLOTS=<int>
HINDSIGHT_API_CONSOLIDATION_LLM_BATCH_SIZE=<int>
```

## Recommended policy for this user

1. Keep daily Hindsight recall on `recall`, not `reflect`.
2. Keep embeddings local; do not claim embedding API savings unless `HINDSIGHT_API_EMBEDDINGS_PROVIDER` is remote.
3. If reducing per-turn recall overhead is desired, implement it in the Hermes Hindsight provider, not by adding nonexistent Hindsight env vars.
4. Candidate Hermes-side feature: `auto_recall_mode=always|conditional`, with keyword/short-input/slash-command gates and an in-process TTL cache.
5. If using `auto_recall=false`, pair it with `memory_mode=hybrid` or `tools`; `memory_mode=context + auto_recall=false` leaves Hindsight mostly invisible because tools are hidden.
6. Changing local embedding model (e.g. `bge-small-en` -> `bge-small-zh`) is not a free config flip. Existing stored vectors are in the old model's vector space; switching requires DB backup, re-embedding/rebuild, and recall benchmark.
7. Treat Hindsight `reflect` as explicit/manual or mental-model refresh machinery, not a default daily dreaming job. This user's offline daily/weekly reflect pipeline is local orchestration, not Hindsight native automatic reflect.
8. Entity resolution/links are part of retain pipeline; current source does not expose independent LLM-heavy entity-merge scheduling knobs. Do not optimize phantom `ENTITY_MERGE_*` vars.

## Quick verification commands

```bash
# Check Hermes Hindsight provider config
cat ~/.hermes/hindsight/config.json | jq .

# Check live Hindsight status/provider/observations
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py status

# Check runtime env without leaking secrets
newgrp docker <<'SH'
docker exec hindsight sh -lc 'env | sort | grep -E "^HINDSIGHT_API_(EMBEDDINGS|RERANKER|RECALL|REFLECT_MAX|REFLECT_WALL|ENABLE_OBSERVATIONS|WORKER_CONSOLIDATION)" | sed -E "s/(API_KEY|TOKEN|SECRET)=.*/\1=[REDACTED]/"'
SH

# Grep for proposed env vars in current container source
newgrp docker <<'SH'
docker exec hindsight sh -lc 'grep -R "HINDSIGHT_AUTO_RECALL_MODE\|HINDSIGHT_RECALL_CACHE\|HINDSIGHT_REFLECT_MODE\|HINDSIGHT_ENTITY_MERGE" -n /app/api/hindsight_api || true'
SH
```

## Pitfall wording

Avoid saying "zero precision loss" for conditional recall. Any gate can miss history-dependent turns. Safer phrasing: reduces useless recall and context pollution, with manual/tool recall as fallback.
