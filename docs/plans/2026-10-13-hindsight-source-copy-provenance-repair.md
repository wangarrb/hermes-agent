# Source copy provenance repair — approved scope

## Approval and protected state

User explicitly approved only the diagnosed exact unknown rows and future atomic native copy receipts. The approved planner SID is `20261008_134609_9c9471`; rows are exactly `98282`, `98283`, `98284`. Their full record bindings are in `/home/wyr/.hermes/cache/scratch/19185a607e2f1b24fd307235af4a399cb50a6f3ba2abc1e27c425fce94b0f004.ambiguous-native-row-inventory.json`. Baseline artifact SHA is `bd2f42385ad95f1f5728e363de18d63f3a07263e0d0bcbbdb65c293eb63bdde3`. This is an explicit finite unknown exclusion, not a proven-copy or success receipt. Possible missed independent occurrences are the accepted trade-off.

Do not recapture/rewrite the original descriptor, reset epochs/cursors, broaden exclusions to a SID/profile/time range, replace models/configs, replay history, alter prior six published batches/canonical seed, or call production during implementation. Unknown rows outside the exact approval still block. Preserve independent append even when body/time/display grouping match a copy. No current CLI process is hot-patched or restarted implicitly.

## Workstream A — finite source exception

Own `local/hermes-scripts/hindsight_incremental_sources.py`, `hindsight_incremental_ledger.py`, `hindsight_incremental_pipeline.py` and focused new tests. Extend existing private immutable-artifact and ledger-meta infrastructure; avoid alternate source readers. A separately versioned hash-bound approved-unknown manifest binds bank/home, original descriptor SHA, profile/native feed epoch/SID, and exact row IDs plus all record hashes. Persist a private immutable envelope under the owning bank lock, referenced by hashed ledger metadata. Its status is unknown/noncompletion. Normal discovery consumes and verifies it; absent/malformed/drifted/missing-row/epoch/schema cases fail closed. Only the three full bindings may be bypassed, and only for SQL source records; no textual/mtime/max-ID exclusions. Native SQL readonly confirmation occurs before installing policy, never paid work. Reader fence accounts for every row. Report policy separately from genuine input completion. Preserve true no-op semantics.

## Workstream B — native atomic copy provenance

Own `hermes_state_messages.py`, a small topical `hermes_state_copy_provenance.py` if no existing equivalent, `hermes_state_sessions.py` only when necessary to preserve owned metadata, and focused `tests/hermes_state/` tests. Reuse native transaction/config merge primitives. No SQLite schema/epoch migration: persist fixed-schema content-free native receipts in the existing private session model_config field `_message_copy_receipts_v1`, atomically in the same transaction as compaction/copy. This is runtime session metadata, not a provider/profile config change.

Canonical interface for the reader (coordinate any justified change before changing this):

    container = {
      'schema_version': 'hermes-message-copy-receipts/1',
      'session_id': <native SID>,
      'source_epoch': <sqlite: SHA256({inode:[dev,ino],application_id:int})>,
      'operations': [operation, ...]
    }
    operation = {
      'operation_id': <canonical UUID>,
      'kind': 'inplace_compaction',
      'copies': [{
        'source_row_id': <int>, 'target_row_id': <int>,
        'source_row_hash': <64-lower-hex>, 'target_row_hash': <64-lower-hex>,
        'copy_hash': <64-lower-hex>
      }, ...],
      'receipt_sha256': <canonical hash of operation WITHOUT receipt_sha256>
    }

The row hashes equal the existing physical source definition: complete SQLite row except `active`, `compacted`, `display_order`; copy_hash additionally excludes `id`, `session_id`; recursive BLOB mapping and canonical JSON must match Sources `_sql_hash`/digest. Receipt requires exact same immutable copy fields. Emit only from an explicitly supplied, verified native `_row_id` causal reference or a trusted SQL-clone source ID; never infer copying from content/timestamps/display grouping. Independent append emits no receipt. Carry old IDs through sanctioned compaction callers if they currently discard them; no wire/history semantic changes. Bind file-backed epoch and SID; do not fabricate durable evidence for in-memory stores. Preserve receipts on session create/resume metadata updates, reject corrupted/spoofed owned containers, retain append-only history, and do not silently clear it. A failing receipt save rolls back message writes. No bodies, vectors, secrets or exception reprs in receipts/logs.

## Parent integration / native reader consumer

After A owns and finishes its Source edits, parent extends Source receipt consumption against B's exact final schema. Keep only this private fixed-schema field plus existing three relation markers in SQLite projection, never full provider configuration. Validate literal types/UUIDs/hashes, epoch/SID/operation uniqueness, old/new exact current row hashes and copy equality, missing targets/parents and cycles. Require actual readonly transaction evidence. Verified copies of exact old baseline or approved unknown rows are not new occurrences; captured future events keep their original identity across verified copies. Same-SID equal body/time without a receipt remains blocked. No new native IDs for already ambiguous rows; three policy exceptions remain explicitly separate.

## Acceptance and rollout

TDD red→green on real temporary SQLite and Core writers; same-SID legitimate copy positive, independent identical append negative, changed source/receipt, wrong scope, missing IDs, hash mismatch, cycles, rollback, copy chains, cross-profile A→B→A, and approved-unknown noncompletion. Original strict physical guards remain. Run canonical `scripts/run_tests.sh` with explicit supported interpreter and private scratch, unique per-file JUnit, no overlapping real source scans. Fresh installed byte parity plus standalone CLI/no-op. Independent final review must exercise actual Core receipt→normal Source→ledger path, not synthetic success flags. After no producer/write window is active, back up exact owned files/state, deploy final verified bytes, install only approved exception manifest, read back exact policy reference, run complete real normal all-profile discovery/admission validation with no paid operations, and reverify original frozen six-batch no-op and bank flags. Do not claim new paid Full completion from source admission alone.

## 13. Post-review hardening (independent findings closed)

The first narrow independent review of the frozen consumer raised one P1 and one P2; both are fixed in the frozen bytes below, with the review's own falsifiers re-executed by the parent.

- P2 escaped-key bypass: `_raw_mentions_copy_receipts()` fails closed for every encoded spelling of the protected key (JSON `\uXXXX` escapes, duplicate keys, malformed JSON, BLOBs) instead of projecting `{}`; a valid escaped key still parses and validates.
- P1 hidden downgrade: `_read_db` now collects the cheap identity tuple of every yieldable row (excluding `id`/`active`/`compacted`/`display_order`), decodes only hidden rows sharing a tuple inside the same read-only transaction, and raises `ambiguous hidden SQLite twin without copy proof` when a yielded no-origin event matches a hidden unproven twin — matching the physical reader's hidden-row falsifier.
- Intentional contract change: an uncaused in-place compaction copy (no explicit native `_row_id`, so no receipt) now fail-closes on rescan instead of silently keeping or losing the input. `test_real_core_compaction_keeps_observed_input_noop` now exercises the receipt-carrying real-caller path; `test_uncaused_real_core_compaction_copy_fails_closed` pins the new fail-closed expectation. Production callers preserve `_row_id`, so the noop path is unchanged for real flows.
- Parent-run evidence: sources 107 / native 58 / unknown 65 zero-failure lanes; the review's 10-case file shows its two P1 probes now raising the fixed error at the first scan and its escaped-key probes passing.
- Frozen consumer bytes: `local/hermes-scripts/hindsight_incremental_sources.py` `ecab301a1bfdfcca2c3295cbf247f3058aa7501c90b96e8baf3f7671e2bcb78e`; `tests/test_hindsight_native_copy_receipts.py` `2cdcb1658da7068a62bc65e1ea5d8a5b727d06f9d2a284563d7600f719e9bc8b`; `tests/test_hindsight_incremental_sources.py` `427396a0c04e6c49dc3e098b1314c1063b83a604892d5b2a11bb93883df9a02a`.
- Final delta review (read-only, own falsifiers): 30/30 green — P2 escaped-family (uppercase `\u005F`, duplicate, malformed, BLOB, non-dict/list, trailing garbage, literal-key BLOB) all fail closed at first scan; P1 witnesses (container-deleted / empty-operations / first-operation-deleted, with and without target deletion plus independent equal append) all raise at the first damaged scan; single-compaction, tail0 and baseline-future variants likewise sealed; no false blocks (receipt-present chains, watermark, hidden without twin, fresh-profile residual plus new arrivals captured once, origin rows, cross-session same content). Declared limitations (non-blocking): a non-UTF-8 BLOB containing an escaped key projects `{}` and relies on the hidden-twin/ledger guards; the main loop streams hidden rows instead of SQL-side filtering (constant memory, no correctness impact); owned 107/58/65 lanes not re-run inside this review (parent canonical XMLs cover them).
Trailing-newline note: the final `test_hindsight_native_copy_receipts.py` drops one trailing blank line for `git diff --check`; the reviewed bytes reconstruct exactly as final bytes + one newline (`2cdcb165...` = sha256(final + "\n")).
