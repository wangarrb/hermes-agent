# Exact SQLite header transition candidate

This is a source-reader candidate, not a deployed migration or production approval.
The normal pipeline still supplies no transition and therefore fails closed.
No baseline, policy, ledger epoch, batch, seed, or source payload is rewritten.

## Explicit SDK entry

`scan_source_snapshots` accepts two additional keyword arguments together:

```python
snapshots = scan_source_snapshots(
    hermes_home,
    profile_mode='all',
    physical_baseline=original_baseline,
    source_unknown_policy=original_policy,
    source_transition_certificate=reviewed_certificate,
    approved_transition_sha256=separately_approved_certificate_sha256,
)
```

The digest must be supplied explicitly by the trusted caller after independent
review. A certificate's own approval metadata cannot authorize it. There is no
config flag, registration command, implicit lookup, or fallback to a new feed.

`inspect_sqlite_transition_header(path)` reads prospective target metadata only;
it neither proves continuity nor creates approval. Actual scanner comparison of
the structure hash occurs in the same read-only transaction as its complete row
validation. The inspection connection is never used as a scanner trust fence.

## Certificate fields

The exact schema is `hindsight-approved-sqlite-header-transition/1`:

- `bank`, `hermes_home`, `profile` (only `default`), `state_db`;
- `physical_baseline_sha256`, `source_unknown_policy_sha256` (or null when no
  policy exists), bound to the original immutable inputs;
- `authorization`: `explicit-exact-sqlite-header-transition`;
- `approval_reference`: a review/approval identifier, and
  `continuity_proof_sha256`: the independently reviewed proof's hash;
- `native_copy_receipt_precondition`: policy `current-epoch-exact-inventory`,
  the exact container session count, and the validated inventory `sha256`
  (`sessions: 0`, `sha256: null` when absent);
- `old`: original `identity`, `epoch`, `schema_version`, `application_id`, and
  `message_columns`, exactly equal to the original descriptor;
- `new`: those same fields plus `structure_sha256`, binding the current sorted
  SQLite schema objects (`type`, `name`, `tbl_name`, `sql`) in the scanner snapshot.

The application ID and message columns cannot change. A new physical identity is
required. The authorization covers that exact replacement only: rereading it is
idempotent, while a third inode, schema-cookie change, or structure drift fails.

The certificate changes only the default header equality comparison. The original
scanner still verifies every baseline row hash, every baseline session identity,
all approved-unknown exact records, native copy receipts, all JSON prefixes, and
all other profiles. Missing or changed evidence fails the entire read. It returns
the original physical feed epochs, and records the certificate digest in the
source fence, preserving the existing ledger and future-arrival identities.

Native copy receipts retain their original strict physical-epoch validation.
This candidate **does not translate old-epoch receipts preserved by a DB copy**;
such a database still fails closed. Only absent receipts or an independently
pinned exact inventory already valid for the new physical epoch can pass.
The inventory count/hash is checked in that same scanner transaction after full
native receipt validation. Later inventory additions require renewed explicit
digest approval; ordinary message appends do not. No receipt is rewritten.

## Fixture verification

```bash
rtk proxy env HERMES_PYTHON=/home/wyr/miniconda/bin/python \
  HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh -j 1 \
  local/hermes-scripts/tests/test_hindsight_source_epoch_transition.py \
  -q --tb=short --basetemp=/tmp/hindsight-epoch-transition-tests
```

Fixtures use actual temporary SQLite replacement, immutable baseline and unknown
policy bindings, JSON exports, two homes and a second profile, and the real
incremental ledger. Future SQLite and late JSON events capture once; the repeated
batch capture adds zero. Negative probes cover approval/binding loss, third
replacement, schema cookie and independently hidden DDL drift, old/unknown row
loss or mutation, session loss or mutation, other profiles, JSON prefix changes,
invalid native copy receipts, and a structurally valid old-epoch receipt. A
valid separately approved new-epoch receipt keeps copied future input no-op.

## Remaining production steps (not performed)

1. Independently review the candidate and fixtures, then review the exact proof
   and the finite proposed old/new header pair. Confirm the original exclusion
   set stays fixed, including all existing policy rows; preserve later arrivals.
2. Obtain explicit approval for the exact certificate digest and production
   source-reader change. The three-session LLM consent is a separate scope.
3. Design the owning caller's secure immutable certificate loading and explicit
   approved-digest propagation. No production caller is wired by this candidate.
4. Under the existing bank lock, verify exact approved inputs and run one complete
   original all-profile read-only admission scan. Any failed guard stops the run.
5. Only after those checks, invoke the existing approved-three runner with its
   original external-data allowlist; keep other local sources from paid work.

Never use `initialize`, replace the baseline, reset a cursor, edit ledger metadata,
or choose an isolated feed to make the current production error disappear.
