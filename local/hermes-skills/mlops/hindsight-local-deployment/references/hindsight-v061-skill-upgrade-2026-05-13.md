# Hindsight v0.6.1 Skill Upgrade Notes — 2026-05-13

Use this as session-specific detail behind the class-level v0.6.1 workflow. The durable rule is in `hindsight-v061-api-first-operations.md`; this file records what was actually implemented and verified locally.

## Implemented skill/library upgrades

- Added v0.6.1 API-first status helper to the local-deployment skill package: `scripts/hindsight_consolidation_status.py`.
- Mirrored the same status helper into `hindsight-consolidation-operations` so both Hindsight umbrellas use the same read-only operational view.
- Enhanced `scripts/hindsight_native_client.py` with wrappers for:
  - `/operations` with `exclude_parents`;
  - `/operations/{id}` with `include_payload=false` default;
  - operation retry/cancel confirm-token guards;
  - `/stats/memories-timeseries` and `/audit-logs/stats`;
  - `/export`, `/import?dry_run=...`, `/v1/bank-template-schema`;
  - targeted repair endpoints: document reprocess, consolidation recover, mental-model refresh, entity regenerate.
- Enhanced `hindsight_pipeline_preflight.py` to check:
  - Operations API availability with `exclude_parents=true`;
  - no active/pending work before production-adjacent changes;
  - v0.6.1 observability endpoints;
  - bank-template schema endpoint;
  - official v0.6.x runtime env keys from Docker inspect;
  - worker-slot reservation sanity.
- Switched default psql discovery away from a hardcoded `18.1.0` path; it now discovers `~/.hindsight-docker/installation/*/bin/psql` and falls back to `psql`.

## Confirm tokens added/standardized

```text
retry-hindsight-operation
import-hindsight-bank-template
reprocess-hindsight-document
regenerate-hindsight-entity
refresh-hindsight-mental-model
recover-hindsight-consolidation
```

Existing destructive token retained:

```text
delete-hindsight-operation
```

## Verification snapshot

Installed to `$HERMES_HOME/scripts` and verified:

```bash
python3 $HERMES_HOME/scripts/hindsight_pipeline_preflight.py --strict-runtime --json
# ok=true, blocking_count=0, warning_count=0

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
  $HERMES_HOME/scripts/tests/test_hindsight_memory_pipeline.py \
  $HERMES_HOME/scripts/tests/test_hindsight_pipeline_preflight.py \
  $HERMES_HOME/scripts/tests/test_hindsight_native_client.py \
  $HERMES_HOME/scripts/tests/test_hindsight_repair_proposal_build.py \
  $HERMES_HOME/scripts/tests/test_hindsight_proposal_review.py -q
# 17 passed

python3 $HERMES_HOME/scripts/hindsight_consolidation_status.py --skip-psql --json
# schema_version=hindsight-consolidation-status-v2
# async_status_source=operations_api
# has_active_work=false
# memories_timeseries=true, audit_log_stats=true
```

Observed Operations API counts with `exclude_parents=true` at verification time:

```text
pending=0
processing=0
completed=805
failed=4
cancelled=0
```

## Lessons / pitfalls

- Absence of `HINDSIGHT_API_READ_DATABASE_URL` is not a warning by itself. It is an optional feature and should only be enabled with a real read-only backend/replica.
- `exclude_parents=true` gives a more useful drain-status view than raw `operations_by_status`, because parent batch rows can inflate completed/pending counts.
- Do not request `include_payload=true` during broad operations scans. Inspect one operation ID at a time.
- Export/import is useful for template movement and temp-bank validation, but high-risk schema upgrades still need DB/filesystem snapshots.
