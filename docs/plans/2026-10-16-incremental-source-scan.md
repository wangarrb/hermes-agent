# Incremental source scan — daily must never full-scan (design)

Status: design for review. Owner: parent (Hermes). Date: 2026-10-16.

## Goal and constraints

- Hard requirement: the daily incremental run must NOT scan all sources.
- Simplest effective design; easy to maintain.
- Precision boundary stated honestly; no silent weakening.

## Core mechanism: per-source fingerprints + tiered reads

1. **Per-source fingerprint** (one aggregate pass per db, cheap):
   - db session: `MAX(message id)` + `COUNT(*)` + the session-row hash
     (includes copy receipts / model_config projection).
   - JSON export file: `(size, mtime, inode)` from stat.
2. **Per-db pre-check**: `(identity, size, mtime)` — an unchanged db skips
   even the fingerprint pass.
3. **Compare** against the stored fingerprints from the last scan.
4. **Changed/new sources → the existing strict read path** (unchanged code,
   zero precision loss); **unchanged sources → skipped** (not read, not
   captured — this also eliminates the 5000+ no-op capture transactions).
5. **Global always-reads** (cheap): sessions table, structure hash, receipts.
6. **Fingerprint store update** after a successful scan.

## Precision boundary (decided by user 2026-10-16)

- Covered: new messages (max id), deletions (count), session/receipt changes
  (row hash), any normal write (file mtime).
- Gap: an in-place content edit of an OLD message that keeps count/max id
  unchanged. **The user explicitly accepts this small gap in exchange for
  never full-scanning.** No periodic full backstop is used (the weekly also
  runs incrementally).
- One-time exceptions (not recurring): baseline initialization/transition
  verification and an explicit diagnostic `--full-scan`. The fingerprint
  store itself can be seeded from cheap aggregates without any full read, so
  even the first incremental run never full-scans.

## Relationship to current code

- The changed-source read path reuses the existing strict logic verbatim.
- New: the fingerprint layer + the skip set + a mode switch
  (`--full-scan` forces the old behavior).
- Full-scan paths retained: initialization, transitions (certificates/policy
  changes), weekly, explicit flag.

## Acceptance (differential proof)

- **Differential test**: incremental mode and full mode on the same state
  produce byte-identical snapshots (the strongest correctness evidence).
- Timing: daily scan drops from ~10 min to seconds on typical days.
- Independent review of the diff before deployment.
- After deployment: consecutive unattended nightly runs (with the overall
  7-run acceptance).

## Implementation notes

- Fingerprints: one `SELECT session_id, MAX(id), COUNT(*) ... GROUP BY
  session_id` per db + the sessions table; JSON exports by stat.
- Skip logic: set membership in the existing scan loop.
- Store: sidecar JSON with atomic write (simplest) vs ledger meta —
  decide in review.
- Remaining blocker for full unattended operation (out of scope here, needs
  user decision): new compressor successor pairs still require approval; a
  bounded pre-authorization would close that loop.
