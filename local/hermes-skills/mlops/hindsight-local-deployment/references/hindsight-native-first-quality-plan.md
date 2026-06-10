# Hindsight Native-First Quality Plan (Session Note)

Date: 2026-05-07

Purpose: capture the current operating policy for native reflect/consolidation/observations with minimal local fallback.

## Verified current state
- API: `http://127.0.0.1:8888`
- health: healthy
- runtime provider: ollama
- observations: false
- worker consolidation slots: 0
- pending operations: 0
- failed operations: 0
- active payload_null: 0
- native consolidation candidates: 5944
- bank config: `enable_observations=false`, `consolidation_llm_batch_size=50`, `consolidation_source_facts_max_tokens=4096`

## Policy
1. Prefer native Hindsight endpoints for reflect/consolidation/observations.
2. Local code may only add control-plane guardrails:
   - preflight/status
   - payload_null quarantine
   - JSON fence parsing hardening
   - long 429 backoff
   - per-job consolidation cap
3. Quality-first default for paid/native consolidation:
   - 50 facts per LLM batch/job
   - no large default batches for the main bank
   - run one job first, then audit observations before scaling
4. Fail closed when any of the following are true:
   - queue not empty
   - active payload_null exists
   - failed operations exist
   - provider mismatch
   - observations enabled unexpectedly
   - patch markers missing

## payload_null handling
- Dry-run first: `cleanup-payload-null`
- Execute only for active `pending`/`processing` rows with `task_payload IS NULL`
- Mark them failed with audit metadata instead of deleting data
- Do not touch completed historical null-payload rows unless they are proven to be re-claimed

## Native consolidation estimates at 50 facts/job
- Fetch rounds: about 119 for 5944 candidates
- Exact-tag grouping means actual LLM calls can be higher than the fetch-round count
- Observed estimate for current backlog: about 180 calls at 50/50 due to tag grouping

## Support files
- `references/hindsight-native-workflow-guard.md`
- `references/hindsight-native-api-first-migration.md`
