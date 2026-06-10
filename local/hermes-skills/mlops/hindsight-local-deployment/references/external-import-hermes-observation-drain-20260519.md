# External import into hermes: sample merge + observation drain lessons (2026-05-19)

## What happened

A third-party external import flow for chat-memo txt and OpenClaw `lcm.db` was validated in isolated 10pct sample banks, then full production records were retained into the production `hermes` bank.

The mistake was treating sample-bank observation success as enough evidence for the later `hermes` import. The 10pct sample banks had observations, but the full import into `hermes` needed its own post-retain consolidation drain.

## Correct handling

1. Do not copy observations directly from sample banks into `hermes`.
   - Sample-bank observations are derived artifacts with bank-local links/scopes.
   - Correct production path is to retain/replace the source documents into `hermes`, then run native consolidation in `hermes`.

2. Before re-retaining sample records, verify whether the same production `document_id`s are already present in `hermes`.
   - In this case, chatmemo 10pct production ids were a subset of the full chatmemo manifest and already existed in `hermes`.
   - OpenClaw 10pct production ids were a subset of the full OpenClaw manifest and already existed in `hermes`.

3. If retain wait crashes after submission, reconcile instead of blindly rerunning.
   - `wait_for_operation_ids` can fail with transient `Connection refused` after retain batches have already been submitted.
   - Verify document presence/stats, then update `external_import/submit_state.json` for the `hermes::<document_id>` keys only after confirming documents exist.
   - Keep a submit-state backup before reconciliation.

4. Production external import should be a single end-to-end gate:
   - build manifest targeting `hermes`;
   - retain `--action production`;
   - wait retain async operation ids;
   - patch `enable_observations=true`;
   - `POST /consolidate`;
   - wait `pending_consolidation == 0` and no child pending/processing operations;
   - require `failed_consolidation == 0` before calling observations complete.

## Verification commands

Check current `hermes` drain state:

```bash
python3 $HERMES_HOME/scripts/hindsight_wait_native_consolidation.py --bank hermes --once --json
```

Dry-run an already-imported manifest against submit state; unchanged records should skip:

```bash
python3 $HERMES_HOME/scripts/hindsight_external_retain_runner.py \
  --manifest /path/to/external-manifest.jsonl \
  --bank hermes \
  --action production \
  --json
```

Expected after submit-state reconciliation: `would_submit_items=0` and `skipped.unchanged=<production_count>` for already-imported production records.

## Pitfalls

- 10pct finalize only proves the sample bank can consolidate; it does not prove the later `hermes` import has observations.
- `enable_observations=true` must be effective on the target bank when consolidation runs, not merely during a previous sample-bank test.
- Do not mark submit state as successful based only on operation submission; verify async completion or confirmed document presence after a transient wait failure.
