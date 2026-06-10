# Hindsight native workflow guard: reflect / consolidation / observations

Session learning from validating Hindsight native workflows without triggering provider call explosions.

## Problem class

Hindsight native `/reflect`, `/consolidate`, and `enable_observations` are useful, but paid-provider runs can explode when:
- `payload_null` async operations are active;
- worker queues are non-empty before provider switch;
- all unconsolidated memories are eligible at once;
- provider returns 429/quota errors and Hindsight retries quickly;
- structured JSON responses are fenced, causing parse retry loops;
- `worker_max_slots <= worker_consolidation_max_slots`, leaving retain no slots.

## Guard script added in this environment

Path:

```bash
~/.hermes/scripts/hindsight_native_workflow_guard.py
```

Useful commands:

```bash
# Read-only status: health, stats, provider env, async op breakdown, payload_null, unconsolidated candidates
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py status

# Fail-closed preflight before paid/native workflow
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py preflight \
  --expect-provider ollama \
  --require-observations-disabled \
  --max-unconsolidated 100

# Dry-run active payload_null cleanup; execution requires explicit token
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py cleanup-payload-null
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py cleanup-payload-null \
  --execute --confirm cleanup-hindsight-payload-null

# Apply in-container JSON fence + 429 long-backoff patch
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py apply-patch
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py apply-patch --restart

# Native reflect smoke through Hindsight API, local provider only
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py smoke-reflect-local \
  --query 'Give a one-line JSON answer confirming native reflect is reachable.'

# Native consolidation smoke on a temporary bank; finally deletes bank and restores normal-local
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py smoke-consolidation-local --cleanup

# Restore normal daily mode
python3 ~/.hermes/scripts/hindsight_native_workflow_guard.py restore-local
```

## Validated facts from the run

- Native `/reflect` works with local Ollama and can return `structured_output`, but weak local models may produce noisy extra text; only treat this as API/workflow reachability, not quality approval.
- Native `/consolidate` worked on a temporary bank using local provider and restored normal-local afterward.
- Main bank had `pending=0`, `failed=0`, `active_payload_null=0`, provider `ollama`, `enable_observations=false`, `worker_consolidation_max_slots=0` after restore.
- Historical completed `batch_retain` rows can have `task_payload IS NULL`; if status is completed and active count is zero, they are not an immediate cost risk.
- Main bank had thousands of unconsolidated candidates (`experience` + `world`), so opening native consolidation against the whole bank should be blocked by preflight unless intentionally authorized.

## Recommended sequence

1. Run status and save evidence.
2. Run preflight with a strict `--max-unconsolidated` threshold before any provider switch.
3. If active `payload_null > 0`, do not switch paid provider. Use dry-run cleanup first; execution is a DB write and requires confirmation.
4. Apply/verify the JSON fence + 429 backoff patch before paid runs.
5. Prefer `smoke-reflect-local`, then `smoke-consolidation-local` on a temp bank.
6. Only after local smoke succeeds, consider paid MiniMax/GLM single-slot smoke on an isolated sample; never open all-bank native consolidation by default.
7. Always finish with `normal-local` and verify provider/slots/queue.

## Implementation details to preserve

- `hindsight_minimax_import.py` should keep low retry env defaults for paid windows:
  - `HINDSIGHT_API_LLM_MAX_RETRIES=2`
  - `HINDSIGHT_API_RETAIN_LLM_MAX_RETRIES=2`
  - `HINDSIGHT_API_REFLECT_LLM_MAX_RETRIES=2`
  - `HINDSIGHT_API_CONSOLIDATION_LLM_MAX_RETRIES=1`
  - `HINDSIGHT_API_CONSOLIDATION_MAX_ATTEMPTS=1`
  - `HINDSIGHT_API_RATE_LIMIT_BACKOFF_SECONDS=300`
- `ollama_env()` supports overriding local model and temporarily enabling consolidation worker slots while keeping `enable_observations=false`.
- Worker slots must satisfy `worker_max_slots > worker_consolidation_max_slots` when consolidation worker is enabled.
- Direct DB status checks are still valuable because stats API may omit processing or hide payload_null detail.
