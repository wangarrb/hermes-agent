# Bounded App source candidate — review only

The App runner remains plan-only. It reads the original three frozen sessions,
their existing production exports and the existing ledger; it never registers
a contract, captures a batch, constructs the runtime or calls a model.
The original447 code-side raw-byte pins and one-read hash/parse stay unchanged.
Newer items, other sessions and same-count content substitutions stay excluded.

The supported API supplies opaque stable item IDs: the real447 contains444
`msg_...` items and3 UUID items. UUID is not the item-ID contract. Session,
turn, item and the exact composite origin are checked together. The task's
existing approved source scope and bank/home/ledger target provide the local
account boundary; no native account-ID field or new namespace is invented.

The App view uses the same final physical source_key, normalized event_id and
original default feed_epoch as its already-present production JSON export.
The immutable payload is unchanged. Both views therefore use the existing
per-source ledger cursor and deterministic batch/operation IDs. No global
alias/ownership table or additional capture hook is needed.

Each App observation is bounded to its own source key. It does not claim to
observe the complete default physical feed and cannot shrink its existing
source inventory. The normal physical reader retains its baseline, finite
unknown-row exclusions, copy/ambiguity, epoch and schema checks.

The read-only plan verifies actual source state and any relevant existing
batch payload. It rejects a changed origin/body, competing known owner,
mixed unauthorized source batch/derived parents and unrelated pending work.
An existing unknown POST remains attached to its original operation UUID and
phase; the plan does not call GET/POST or report it as completed.

Unrelated historical batches do not need new dual-view proofs. An existing
unknown-row policy does not forbid all future sessions. Exact legacy/unknown
identity or raw-export matches remain exclusions; incomplete historical
coverage stays explicitly unknown, rather than becoming an unbounded demand
to disprove every possible semantic copy.

The earlier finite-contract framework was an overbroad candidate. It has been
removed from the implementation; its prior fixture results do not establish
real-data admission. The narrowed candidate must pass real opaque-ID fixtures,
App/physical replay in both orders, immutable-payload/authorization failures,
operation reconciliation and the existing physical rejection regressions.

This candidate authorizes no deployment or external extraction. A real447
read-only plan and independent review are required before any later execution.

Canonical validation:11 test files,562 passed,0 failed with automatic file
retries disabled. This includes21 bounded-adapter cases,7 boundary/delivery/
concurrency/crash cases and4 original raw-pin cases, plus the existing ledger,
physical-source, native-copy, finite-unknown, transition and pipeline checks.

The real frozen447 plan now passes:3 sources,447 new messages,0 known messages
and0 relevant conflicts. The existing60000-character capture budget projects6
source batches; these are metadata calculations, not captured files or LLM-call
counts. Ledger tables, baseline/unknown artifacts, original frozen/production
exports and the3438-key physical-feed inventory stayed unchanged. Execution
and native-runtime preflight remain disabled and unattempted.
