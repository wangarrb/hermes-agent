# Exact SQLite header transition candidate

This source-reader mechanism is deployed with an explicit per-profile
certificate set supplied by the cron wrappers (see Production wiring below).
Without supplied certificates the reader still fails closed.
No baseline, policy, ledger epoch, batch, seed, or source payload is rewritten.

## Explicit SDK entry

`scan_source_snapshots` accepts two additional keyword arguments together:

```python
snapshots = scan_source_snapshots(
    hermes_home,
    profile_mode='all',
    physical_baseline=original_baseline,
    source_unknown_policy=original_policy,
    source_transition_certificates=[reviewed_certificate, ...],
    approved_transition_sha256s=[separately_approved_certificate_sha256, ...],
)
```

Every certificate must match exactly one approved digest, every approved digest
must be consumed by exactly one certificate, and at most one certificate may
exist per profile. Any profile without a certificate stays strict. The digests
must be supplied explicitly by the trusted caller after independent review. A
certificate's own approval metadata cannot authorize it. There is no config
flag, registration command, implicit lookup, or fallback to a new feed.

`inspect_sqlite_transition_header(path)` reads prospective target metadata only;
it neither proves continuity nor creates approval. Actual scanner comparison of
the structure hash occurs in the same read-only transaction as its complete row
validation. The inspection connection is never used as a scanner trust fence.

## Certificate fields

The exact schema is `hindsight-approved-sqlite-header-transition/1`:

- `bank`, `hermes_home`, `profile` (any baseline profile; at most one
  certificate per profile), `state_db`;
- `physical_baseline_sha256`, `source_unknown_policy_sha256` (or null when no
  policy exists), bound to the original immutable inputs;
- `authorization`: `explicit-exact-sqlite-header-transition`;
- `approval_reference`: a review/approval identifier, and
  `continuity_proof_sha256`: the independently reviewed proof's hash;
- `native_copy_receipt_precondition`: either policy
  `current-epoch-exact-inventory` with the exact container session count and
  validated inventory `sha256` (`sessions: 0`, `sha256: null` when absent), or
  policy `current-epoch-superset-inventory` adding `operations`
  (`{session_id: {operation_id: operation_digest}}`); every pinned operation
  must still exist unmodified, while later legitimate appends are admitted
  after the same native receipt validation;
- `old`: original `identity`, `epoch`, `schema_version`, `application_id`, and
  `message_columns`, exactly equal to the original descriptor;
- `new`: those same fields plus `structure_sha256`, binding the current sorted
  SQLite schema objects (`type`, `name`, `tbl_name`, `sql`) in the scanner snapshot.

The application ID and message columns cannot change. The new header may be a
physical replacement (new identity) or an in-place schema rebuild (same
identity, same epoch, changed schema cookie/structure). The authorization
covers that exact new header only: rereading it is idempotent, while another
identity, schema-cookie, or structure drift fails.

The certificate changes only the covered profile's header equality comparison.
The original scanner still verifies every baseline row hash, every baseline
session identity, all approved-unknown exact records, native copy receipts, all
JSON prefixes, and every other profile. Missing or changed evidence fails the
entire read. It returns the original physical feed epochs, and records the
certificate digest in the source fence, preserving the existing ledger and
future-arrival identities.

Native copy receipts retain their original strict physical-epoch validation.
This mechanism **does not translate old-epoch receipts preserved by a DB copy**;
such a database still fails closed. Absent receipts pass; an independently
pinned exact inventory passes; a superset inventory admits only later appends
whose operations were validated for the current epoch while every pinned
operation stays present. Later exact-inventory additions require renewed
explicit digest approval; ordinary message appends do not. No receipt is
rewritten.

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

## Production wiring

The owning callers supply the frozen set explicitly:

- `hindsight_source_transitions.py` holds the independently pinned digests
  (`APPROVED_SOURCE_TRANSITION_SHA256`) and loads the certificate files from
  `$HERMES_HOME/hindsight/source-transitions/`. A present directory with a
  missing or digest-mismatched certificate fails before any scan; an absent
  directory supplies nothing.
- `hindsight_daily_noagent.py` (also used by the weekly wrapper) loads that
  set and passes it through `run_alias` → the pipeline CLI
  (`--transition-certificate` / `--approved-transition-sha256`, repeatable)
  → `run_incremental` → `scan_source_snapshots`. No production caller is
  otherwise wired; direct CLI use must pass both explicitly.

Certificate generation and review evidence live outside the repository under
the parent's task evidence; the certificates themselves remain local private
files. The original exclusions stay fixed and later arrivals are preserved.

Never use `initialize`, replace the baseline, reset a cursor, edit ledger metadata,
or choose an isolated feed to make a production error disappear.

## Frozen external-input guard follow-up

Independent review found that a session/count-only allowlist did not pin content.
The owning runner now pins the three original frozen file SHA256 values in code,
verifies one read_bytes result, then parses those same bytes. It rejects equal-count
content substitution before runtime/configuration. Four focused canonical-runner
tests passed, including a read-then-path-replacement case. This runner change is
a review candidate only: no external run, baseline rebuild, or transition apply
was performed. Evidence: task/evidence/approved-run-frozen-hash-fix.json.
