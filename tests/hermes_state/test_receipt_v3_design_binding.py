"""Receipt r4 milestone 2D: design binding, expected precedence, disclosure.

Contract from the frozen effective r4 machine spec (amendment section 5 rows
"版本/优先级/审查状态" / "受影响 expected 增补" / "准确披露") and the gates
inventory section 2 (A1/A2/A3):

  * the acceptance apparatus resolves each gate's expectation with
    ``r4_expected`` if present, else the original unchanged ``r3_expected``
    (``protocol.precedence`` / ``amendment_lineage.expected_precedence``);
  * exactly the ten adapted gates carry an ``r4_expected`` extending the
    retained ``r3_expected`` prefix (supplement-only derivation) and every
    historical field stays retained;
  * the binding is auditable derivation metadata, explicitly NOT protocol
    record: no binding value may enter the wire -- the protocol revision
    stays 3;
  * the disclosure fields match the frozen r4 bytes AND the owner-approved
    state (2026-10-17: F/context conservative pause approved, isolated
    implementation per r4 design; migration/activation/production and
    bridge/certificate/pin/ledger stay unauthorized; unknowns stay truthful).

The fixture bytes below were exported verbatim from the frozen effective r4
JSON (sha256 70fec478...), the frozen r3 base (5d2af129...) and the owner
decision record (d052ad48...). RED against the M2C slice: these surfaces do
not exist there. Synthetic fixtures only; no real store is opened and no
production data is touched.
"""

from __future__ import annotations

import hashlib
import json

import pytest

def _m():
    import hermes_state_copy_provenance as module
    return module


def _canon(x) -> bytes:
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8", "strict")


EXPORT = json.loads(r"""{
 "source_shas": {
  "source_json_sha256": "70fec4785772aabd1ca075fe8c1d65e3fbc6da1e9c216bb74fbd26bc7fac54f1",
  "base_json_sha256": "5d2af1296bf6ea7dd018b01c32d7b2b670b54a6ba98cc99215345397ac726ade",
  "amendments_sha256": "7b49d7e90f2d8782d24ba4708f484b728f8d62623afbebfec228b116ca3ad879",
  "owner_decision_sha256": "d052ad48922b37171efc0eb5b9a1dcc4ef1919975bcd2113d817e04bd9304a6d",
  "gates_inventory_sha256": "a0e0cfd8b94e8ed08e0e39ab1777bd157c4190e2f7f53a4adfd5e839c79ef7a7"
 },
 "metadata": {
  "artifact_version": "schema31-receipt-v2-design/4",
  "revision": "review-candidate-r4-focused-amendments",
  "status": "READY_FOR_FOCUSED_INDEPENDENT_DESIGN_REVIEW",
  "precedence": "Exact named owner SHA > r4 amendments and this self-contained effective r4 JSON (contradiction stops) > unaffected pinned r3 MD SHA 22729cadf86090464e67f3a14b04b5b031b0190c940a95d1388e977980d56018. r1/r2 only exact previously enumerated unchanged references; no fallback hybrid. r4 is a design-document revision, not a wire/schema/domain/epoch revision. All historical r3_expected retained; per-row r4_expected, if present, is the effective expectation.",
  "expected_precedence": "r4_expected if present else original unchanged r3_expected; all original inputs/hashes/owner/status/disposition retained",
  "amendment_lineage": {
   "base_json_sha256": "5d2af1296bf6ea7dd018b01c32d7b2b670b54a6ba98cc99215345397ac726ade",
   "shared_r3_md_sha256": "22729cadf86090464e67f3a14b04b5b031b0190c940a95d1388e977980d56018",
   "amendments_sha256": "7b49d7e90f2d8782d24ba4708f484b728f8d62623afbebfec228b116ca3ad879",
   "only_findings": [
    "R3-C01",
    "R3-C02",
    "R3-Q01"
   ],
   "owner_approved": false,
   "implementation": false,
   "runtime_pass": false
  }
 },
 "rows": [
  {
   "id": "A01",
   "disposition": "carry",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Carry historical nonempty v1 full0/3 old30 3/3 observations only; raw/epoch unchanged; no approved bridge still refuses."
  },
  {
   "id": "A02",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Independent writer/standalone3.11+3.14 codec/page/frame/P/F/joint vectors identical; all unsupported types/versions refused. This turn stdlib only."
  },
  {
   "id": "A03",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Literal actual30 physical manifest,20 payload,2 ref,8 explicit excluded; unknown/missing/reordered schema or any4 UID unproved drift refuses. Excluded changes do not claim full proof."
  },
  {
   "id": "A04",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "schema/3+revision3+exact P domain required; unknown family/escaped decoded duplicates/page/header/hash/group downgrade refuses before yield; old reader ignorance not acceptance."
  },
  {
   "id": "A05",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Original transition /1 unchanged; new /4 exact independently approved genuine snapshot bridge+pins and BOTH scanner seams. Missing/extra/misbound proof refuses."
  },
  {
   "id": "A06",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "v1 original key/value lexical tokens/op hashes/old epoch preserved through native head and generic config writes; journal append does not rewrite history. Whole model_config may change legitimately."
  },
  {
   "id": "A07",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "All original config routes plus set_meta(cursor)/prefix/list/import: omit/echo preserves stored bytes; cannot install/clear/rewind head/journal; no partial writes."
  },
  {
   "id": "A08",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Only exact real persisted-parent receipt/head stripping on trusted native relation; parent unchanged; child own g0; no copied state_meta ownership."
  },
  {
   "id": "A09",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Explicit trusted donor+actual inserted row mapping only; same text/time/UID independent append no edge; truly derivative P no copy. Excluded-only difference can be projected source copy, not full row."
  },
  {
   "id": "A10",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Named early-return branch with unseen/gap/rewind donor covered; selected actual lastrowid and same-tx pages/head/counts; no UI audit requirement."
  },
  {
   "id": "A11",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Watermark/tail actual mapping preserves late old-dated rows and four UID stored payload; no guessed ID ranges or UID-only success."
  },
  {
   "id": "A12",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Real postturn stamp no display read before compaction, both unseen/watermark -> invisible donor NULL vs fixed target allowed under P; changed protected P or unknown donor refuses. Pruned checkpoint is excluded, no witness."
  },
  {
   "id": "A13",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Source-proof head/held P fence catches compaction or payload repair concurrent winner; excluded-only metadata must NOT stale slow summary; new append retains native coverage."
  },
  {
   "id": "A14",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Actual parent→child exact P only, native lease/closure/counts same tx; donor observation no parent ownership transfer; group complete."
  },
  {
   "id": "A15",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Inject faults at original native seams + pages/header/index/head CAS/rollover; whole native txn rollback and all live UID/CAS/row/timestamp/canonical fields restored. No permanent metadata journal."
  },
  {
   "id": "A16",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "BUSY whole callback retries same request/token/donor memo, one committed op/targets; changed request rejects; rollback rowid cannot become donor."
  },
  {
   "id": "A17",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Commit-unknown/ACK loss read original token/result on admitted epoch before stale; no blind replay; missing token needs reload."
  },
  {
   "id": "A18",
   "name": "多代链与删除降级",
   "disposition": "adapt",
   "original_input": "v1→v2→v2，删旧op/删整v2key/隐藏row缺receipt/循环/同target双donor",
   "r1_input_sha256": "14bc8ac6c85982bf0b3fa72da8f3eb32377f3675d38a14467186f62a48f3d15e",
   "r2_expected_sha256": "7b16f4f4acca9f4dfdf57d0df103dc1a4b899eebb556bd80ca87cb6b92cf46db",
   "r3_expected": "v1 bridge→copy→necessary derivation→copy→epoch continuation; remove any referenced page/header/row/head/alias or duplicate target/cycle/group gap rejects; UI audit removal intentionally not tested.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "v1 bridge→copy→necessary derivation→copy→epoch continuation; remove any referenced page/header/row/head/alias or duplicate target/cycle/group gap rejects; UI audit removal intentionally not tested. Also independently replay every complete/closed-partial epoch-local range key and native context predecessor; missing old page or prior-context ref refuses, no mutable checkpoint downgrade.",
   "r4_adaptation_reason": "R3-C01/Q01"
  },
  {
   "id": "A19",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "One ro snapshot actual identity fence; current20 P leaves/UIDs + approved anchors/derivations checked; replacement requires exact joint-proof continuation. No claim current-full30 evolution."
  },
  {
   "id": "A20",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Invalid proof/changed captured feed => zero capture/POST; old source_key/feed_epoch/epoch_scope/event_id/message_hash and ledger-rooted delta-only contract unchanged."
  },
  {
   "id": "A21",
   "disposition": "carry",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Carry multi-shard sparse rowid backfill reopen/interruption requirements: progress monotonic, preserve nonNULL, NULL0+done; tiny fixture not full budget proof."
  },
  {
   "id": "A22",
   "disposition": "carry",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Unmodified old3.11 writer remains disabled after31/recovery; read-only old API success no write permission; NULL pending != pass."
  },
  {
   "id": "A23",
   "name": "drain与恢复writer",
   "disposition": "adapt",
   "original_input": "owner批准维护窗口后清点全部写句柄；故意留一个旧CLI/cron writer",
   "r1_input_sha256": "82963388737e2ede9271042bb2edb56baf1c052fcabce5547b8c6890f5f315e2",
   "r2_expected_sha256": "8859ea4f0b7252e779f6a830d0e132037419b9d2754f28636161c26ae5c18f88",
   "r3_expected": "All writer/scanner and generic meta paths capability roster; approved drain only, no pane kill. Partial recovery owner roster not active.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "All writer/scanner and generic meta paths capability roster; approved drain only, no pane kill. Partial recovery owner roster not active. Quiet native recovery admission is explicitly proposed and separately approved; no constructor/repair/usage/FTS maintenance before proof-to-ALL-readback release, including crash/reopen. Holder drain remains external and approved, no pane kill.",
   "r4_adaptation_reason": "R3-C02"
  },
  {
   "id": "A24",
   "disposition": "carry",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Carry exact R0/R1 boundary; any constructor/DDL/backfill/write => R2 with latest data preservation, not code-only rollback."
  },
  {
   "id": "A25",
   "name": "R2保新数据",
   "disposition": "adapt",
   "original_input": "target产生新消息/usage/config/v2后故障，完整snapshot+故障侧车保留",
   "r1_input_sha256": "d546e6f81da6689b4c53e251a0614c9adbc6691f50a4c77975b239369756473c",
   "r2_expected_sha256": "8d4763f0d158e9cd6ef92a11f3590869204bf049cd29a717383b07bc4de748fe",
   "r3_expected": "Latest complete all-table typed JOINT identity+values proof (incl all UI/sidecars/UID/usage/config/state_meta), approved same-epoch build and new-inode continuation; no oldsnapshot rollback.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "Latest complete all-table typed JOINT identity+values proof (incl all UI/sidecars/UID/usage/config/state_meta), approved same-epoch build and new-inode continuation; no oldsnapshot rollback. Selected byte/data-complete replacement: preconstructor and post-quiet-open joint equality to ORIGINAL latest, proof then approval then ALL continuation and final independent exact named identity-aware inverse joint, including FTS/config/shadows; release only after separate activation.",
   "r4_adaptation_reason": "R3-C02"
  },
  {
   "id": "A26",
   "disposition": "carry",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Crash/journal/disk-full tests distinguish SQL atomicity from powerloss/fsync; preserve raw originals/sidecars, recover only approved copies."
  },
  {
   "id": "A27",
   "name": "资源限额",
   "disposition": "adapt",
   "original_input": "真实大小分布的多profile副本；慢chunk/写锁/空间/恢复时限",
   "r1_input_sha256": "6b207bb9dc80d7c486831ae72d126a4b5d136db61212fd7c1538bdcf592ea0f9",
   "r2_expected_sha256": "13e3ac618a6a23d4b26da1cea341c1b3b88664323c80789aedde1a7d0fafc1a2",
   "r3_expected": "2001+ visible native rows no proof cap; fixed-page streaming, >former8MiB cumulative journal works; finite measured reserve, maintenance relocation and crash resume retain all pages; budgets actual approval required.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "2001+ visible native rows no proof cap; fixed-page streaming, >former8MiB cumulative journal works; finite measured reserve, maintenance relocation and crash resume retain all pages; budgets actual approval required. Include same-epoch two full1024 blocks, non-g1 epoch start, closed partial tail, multiple epochs; reserve/relocation passes quiet constructor/FTS recovery gates before normal bounded maintenance resumes; no native resource PASS this turn.",
   "r4_adaptation_reason": "R3-C01/C02"
  },
  {
   "id": "A28",
   "disposition": "carry",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "A→B→A profile exact opened handle identity; sameSID otherDB never shares proof; birth_ref versus logical feed epoch distinct; no credential cold imports/network."
  },
  {
   "id": "A29",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Real slow/proactive/micro/rotated compaction callers carry source head+held P/op token. Receipt failure must not claim success or leak unproved compaction into later flush. Metadata sidecar failure not source-copy audit failure."
  },
  {
   "id": "A30",
   "name": "原生元数据后写",
   "disposition": "adapt",
   "original_input": "reaction/display_metadata/api_content/platform/codex checkpoint在receipt后合法写入",
   "r1_input_sha256": "512366ba329ff57bfb6fb51f6fc0e0f76d5fea8fdcf94b4832b0c073bbb13cc3",
   "r2_expected_sha256": "5264caa3e88de7a99a0047df476fd657a03d77914d8d6d93817604433cfbd995",
   "r3_expected": "Repeated native reaction/seen/API/display/checkpoint excluded writes succeed without receipt op/sealing ordinary row; platform origin/core/UID remain protected; necessary native CAS repair D/F exact derivation and ingestion policy explicit. D/F is raw feed-input Q classification; protected raw-content same-clean repair still pauses ingestion pending policy, native app does not roll back solely for this policy pause.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "Repeated native reaction/seen/API/display/checkpoint excluded writes succeed without receipt op/sealing ordinary row; platform origin/core/UID remain protected; necessary native CAS repair D/F exact derivation and ingestion policy explicit. D/F is raw feed-input Q classification; protected raw-content same-clean repair still pauses ingestion pending policy, native app does not roll back solely for this policy pause. Qsame D additionally requires immutable readable prior native context + approved namespace/lineage recomputation; unknown->cli/cron can change binding with unchanged P/Q. Legal native update preserved; reader context refusal/correction policy is separate and unapproved.",
   "r4_adaptation_reason": "R3-Q01"
  },
  {
   "id": "A31",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Final separately approved implementation under pinned PM canonical runner, red controls then real native green; static formatting/algebra never runtime acceptance."
  },
  {
   "id": "A32",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Hash-bind r3 source/input/owner decision, freeze new docs+JSON/evidence/operator only; old r1/r2/reviews/source/operators unchanged. No DB/App/producer/scanner/API executed in this turn."
  },
  {
   "id": "B01",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Replace full30-column evolution commitment objective with20-P/ref membership+actual30 manifest, typed joint recovery and domain/version controls. Excluded leaf absent intentionally named; unknown P leaf or F drift refuses."
  },
  {
   "id": "B02",
   "name": "legacy boundary bridge + later real metadata",
   "disposition": "adapt",
   "original_input": "nonempty true legacy v1 fixture duplicate rows and exact donor; approved test bridge computed from real old/new closed snapshots, then native api/reaction writes",
   "r1_input_sha256": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
   "r2_expected_sha256": "c7e3e402a83dbab5f5d542313d635e0c791261e139458b46f25fb6457abc4c0b",
   "r3_expected": "Genuine nonempty legacy finite bridge and later excluded metadata+D native repair independently validated; old bytes/full domain/epoch unchanged; added UID drift without approved derivation refuses. D/F is raw feed-input Q classification; protected raw-content same-clean repair still pauses ingestion pending policy, native app does not roll back solely for this policy pause.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "Genuine nonempty legacy finite bridge and later excluded metadata+D native repair independently validated; old bytes/full domain/epoch unchanged; added UID drift without approved derivation refuses. D/F is raw feed-input Q classification; protected raw-content same-clean repair still pauses ingestion pending policy, native app does not roll back solely for this policy pause. Boundary owner inventory carries actual readable typed four-field/marker context, legacy digest and approved relationship facts. New owner uses immutable first protected header, not finite old baseline pretending to cover all future owners.",
   "r4_adaptation_reason": "R3-Q01"
  },
  {
   "id": "B03",
   "disposition": "owner-policy-retire",
   "status": "RETIRED_BY_SELECTED_DESIGN_DIRECTION_NOT_RUNTIME_TEST",
   "r3_expected": "Retire permanent metadata/peer event-closure objective by exact owner B decision; native peers/reconcile/reaction remain correct business but no audit witness. Replacement negative A12/A30 verifies no proof and no F/core loophole."
  },
  {
   "id": "B04",
   "name": "metadata concurrency + rollback + late finality",
   "disposition": "adapt",
   "original_input": "metadata after edge twice, slow summary sampled before another metadata tx, forced receipt/CAS/space failure and ACK loss",
   "r1_input_sha256": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
   "r2_expected_sha256": "f76280e06a598ee2f425b83466ee1247ccb872d3be3bda10795b86e3d2fd939a",
   "r3_expected": "Adapt concurrency/latefinality: excluded changes no new op/no stale; protected D bumps head; F ingestion stop; payload receipt fault rollback+same-token readback; no permanent metadata timeline. D/F is raw feed-input Q classification; protected raw-content same-clean repair still pauses ingestion pending policy, native app does not roll back solely for this policy pause.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "Adapt concurrency/latefinality: excluded changes no new op/no stale; protected D bumps head; F ingestion stop; payload receipt fault rollback+same-token readback; no permanent metadata timeline. D/F is raw feed-input Q classification; protected raw-content same-clean repair still pauses ingestion pending policy, native app does not roll back solely for this policy pause. Missing/drifted prior context is not Fsame or writer_kind self-authorization. Compare actual immutable before-context to current native binding without rolling back legal session updates; no audit of ordinary excluded metadata added.",
   "r4_adaptation_reason": "R3-Q01"
  },
  {
   "id": "B05",
   "name": "new inode authorization and continued append",
   "disposition": "adapt",
   "original_input": "byte-identical closed latest source copied to chosen inode then copy again; approved test proof for ONLY first chosen file",
   "r1_input_sha256": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
   "r2_expected_sha256": "8ba5daadaa198601d14393abe7ece31b15c8b0f205932fdd7c3eff71cddc39b2",
   "r3_expected": "Exact new inode proof then continuation and ordinary copy/repair append; second unapproved identical copy refuses; old op bytes/epoch exact.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "Exact new inode proof then continuation and ordinary copy/repair append; second unapproved identical copy refuses; old op bytes/epoch exact. Quiet unchanged open + final ALL-owner full joint and separate activation precede deferred FTS merge; no constructor settle/rebaseline, no FTS shadow waiver, next unapproved inode still refuses.",
   "r4_adaptation_reason": "R3-C02"
  },
  {
   "id": "B06",
   "name": "two epochs generation/row namespace",
   "disposition": "adapt",
   "original_input": "g0 v1-only/no-receipt owners plus multi-op owners; two successive approved replacements, same SID/ROWID in other DB",
   "r1_input_sha256": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
   "r2_expected_sha256": "c74640189068ab800927fb0d8eb17384ec9d7de4b8882b98e6820a2029c4edae",
   "r3_expected": "g0/v1/no-receipt owners all included; two approved epochs continue g, exact finite actual-row aliases only; no UID reconstruction or copy edge; monotonic ids.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "g0/v1/no-receipt owners all included; two approved epochs continue g, exact finite actual-row aliases only; no UID reconstruction or copy edge; monotonic ids. Epoch-local ranges start at actual first g (continuation g may not1), same-epoch second range and closing partial have distinct immutable keys, exact counts/root and unchanged old pages; g/epoch never reset.",
   "r4_adaptation_reason": "R3-C01"
  },
  {
   "id": "B07",
   "name": "complete R2 all-table equality and partial initialization",
   "disposition": "adapt",
   "original_input": "new messages/usage/config/UID/tool arrays plus receipts before failure; equal row count but altered PK/ROWID, missing one session/config table, power loss half roster",
   "r1_input_sha256": "74234e98afe7498fb5daf1f36ac2d78acc339464f950703b8c019892f982b90b",
   "r2_expected_sha256": "5c49e3b7d531317951b0f49d1190407a2c825dde9459de80407711623920a6a4",
   "r3_expected": "Swap rowid/value association with unchanged separate summaries must fail JOINT hash, including nonalias rowid/FTS shadow; latest all-table full values preserved; partial roster crash idempotent maintenance only.",
   "reason": "exact owner selected source-payload/no permanent UI-sidecar audit; preserve causal/feed/recovery floor",
   "authority_sha256": "540803934c1c05fb76a57c101931ac3b59d16c9da1368550ab653c061514648d",
   "status": "REQUIRED_NOT_RUN",
   "r4_expected": "Swap rowid/value association with unchanged separate summaries must fail JOINT hash, including nonalias rowid/FTS shadow; latest all-table full values preserved; partial roster crash idempotent maintenance only. Real FTS segments/usermerge cadence cases999/1000/1001+ owners, partial roster crash/quiet reopen, request-named identity-aware inverse and final independent ALL readback required after separate implementation approval; post-activation FTS/usage/config may legitimately evolve.",
   "r4_adaptation_reason": "R3-C02"
  },
  {
   "id": "B08",
   "disposition": "adapt",
   "status": "REQUIRED_NOT_RUN",
   "r3_expected": "Excluded-only + authorized replacement +1 fresh real event -> no historical POST and exactly1 delta; no plaintext/UID/config emitted; page-bound/spill/reserve/relocation failure no truncation, not permanent owner cap."
  }
 ],
 "disclosure": {
  "known_published_gap": "Owner selected B design direction only. F pause/correction NOT approved: current strict r3 proposal blocks ANY protected raw-Q-change, including not-yet-captured activation/recovery anchors, copy endpoints, tool/summary nonfeed rows, rawsame-clean and changed-back; not merely published/captured data. N already captured but not anchor/ref remains under original ledger SourceChanged. No ignore/rePOST/baseline reset. Q01 context missing/drift is a distinct newly explicit unapproved policy boundary, not F authorization.",
  "remaining_policy_gates": [
   "focused non-author independent review of exact r4 freeze and owner exact design approval",
   "ANY protected Q-change F pause/correction policy NOT approved: uncaptured anchors/copy endpoints/tool-summary nonfeed/rawsame-clean/changed-back included; N already captured retains original ledger SourceChanged",
   "Q01 unknown/missing/drifted context policy and any exception/correction NOT approved; no rollback of legal native unknown-to-surface changes",
   "private native quiet-recovery admission/defer and external holder fence design/implementation/drain/activation require separate approval; actual constructor/FTS/partial-roster tests NOT_RUN",
   "finite representative resource maxima/provisioned reserve/operator maintenance budget before deployment",
   "legacy snapshot bridge+independent exact proof/cert/pin/source/ledger activation approval, migration/production/publication remain unauthorized",
   "all39 real native writer/reader/recovery/durability/negative gates after separate implementation+copy-write approval"
  ],
  "finding_disposition_I02_normative_sections": "repair D/F unchanged raw-Q classification; F proposal covers ANY protected Q-change (not only published). Q01 explicit prior context+approved namespace chain/refusal, still unapproved. Native business remains usable; no independent closure or runtime proof."
 },
 "owner": {
  "question": "F/context政策＋隔离实施",
  "user_response": "两项均按保守暂停批准，并授权按r4设计进入隔离实现（不迁库/不激活/不写生产）",
  "state": "POLICIES_APPROVED_AND_IMPLEMENTATION_AUTHORIZED_ISOLATED",
  "limits": [
   "no DB migration",
   "no activation",
   "no production writes",
   "no legacy bridge/certificate/pin/ledger activation"
  ]
 },
 "writer_timeline": {
  "r4": [
   "Ordinary native path retains existing _execute_write replacement fence/BEGIN IMMEDIATE and approval/domain/epoch/head checks. Recovery continuation ONLY uses proposed private quiet admission and same original transaction core per recovery.quiet_window; no alternate producer or ordinary-constructor settlement before proof.",
   "Check existing token request result before expected-head stale; check native leases/relations/coverage/held P and original source-proof fence.",
   "Use actual row-state index and selected stored rows; run normal native INSERT/CAS repair/archive/rewind/prune/reconcile; capture necessary D/F derivation before/after, not excluded metadata.",
   "Bind real donor→actual inserted lastrowid at observations above; source<target within canonical current AUTOINCREMENT order; cross-epoch aliases only approved boundary identity, one incoming donor across v1/v3.",
   "Stream fixed-size pages INSERT state_meta through same conn; append hash-bound operation header; update derived row/token indexes and fixed head by old raw-config+generation+hash CAS. Counters/closure/flags/rows/all receipt data same tx.",
   "Only _execute_write commits native op+receipt/index/head/counts; callback faults roll all back and preserve caller state. Actual native postcommit _write_count++ remains; quiet recovery defers FTS usermerge/merge until ALL-owner independent full-joint readback and separate activation. WAL checkpoint is not logical table mutation. Ordinary postcommit maintenance behavior unchanged."
  ],
  "r3": [
   "Existing _execute_write replacement fence and BEGIN IMMEDIATE only; approved header/domain/epoch capability and owner head checked. No alternate connection/sidecar transaction.",
   "Check existing token request result before expected-head stale; check native leases/relations/coverage/held P and original source-proof fence.",
   "Use actual row-state index and selected stored rows; run normal native INSERT/CAS repair/archive/rewind/prune/reconcile; capture necessary D/F derivation before/after, not excluded metadata.",
   "Bind real donor→actual inserted lastrowid at observations above; source<target within canonical current AUTOINCREMENT order; cross-epoch aliases only approved boundary identity, one incoming donor across v1/v3.",
   "Stream fixed-size pages INSERT state_meta through same conn; append hash-bound operation header; update derived row/token indexes and fixed head by old raw-config+generation+hash CAS. Counters/closure/flags/rows/all receipt data same tx.",
   "Only _execute_write commits; all faults roll back native op+receipt/index/head/counts; caller copy failure not falsely persisted by later ordinary flush."
  ]
 }
}""")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_design_metadata_binding_matches_frozen_r4_and_stays_outside_protocol():
    m = _m()
    binding = m.receipt_design_binding()
    meta = EXPORT["metadata"]
    assert binding["artifact_version"] == meta["artifact_version"] == "schema31-receipt-v2-design/4"
    assert binding["revision"] == meta["revision"]
    assert binding["status"] == meta["status"]
    assert binding["precedence"] == meta["precedence"]
    assert binding["expected_precedence"] == meta["expected_precedence"]
    lineage, frozen = binding["amendment_lineage"], meta["amendment_lineage"]
    for key in ("base_json_sha256", "shared_r3_md_sha256", "amendments_sha256"):
        assert lineage[key] == frozen[key]
    assert list(lineage["only_findings"]) == frozen["only_findings"] == ["R3-C01", "R3-C02", "R3-Q01"]
    assert lineage["owner_approved_at_authoring"] is False
    assert lineage["implementation_at_authoring"] is False
    assert lineage["runtime_pass_at_authoring"] is False
    assert binding["source_json_sha256"] == EXPORT["source_shas"]["source_json_sha256"]
    # Not a wire revision: the protocol constants are untouched.
    assert binding["wire_or_schema_revision"] is False
    assert binding["protocol_record"] is False
    assert m._PROTOCOL_REVISION == 3
    assert m._HEAD_SCHEMA == "hermes-message-copy-receipts/3"
    assert m._OPERATION_SCHEMA == "hermes-receipt-operation/3"
    assert binding["gate_rows"] == 40 and binding["adapted_gate_count"] == 10


def test_resolve_gate_expected_prefers_r4_over_r3_and_retains_history():
    m = _m()
    rows = EXPORT["rows"]
    adapted = [(index, row) for index, row in enumerate(rows) if "r4_expected" in row]
    assert [index for index, _row in adapted] == [17, 22, 24, 26, 29, 33, 35, 36, 37, 38]
    for _index, row in adapted:
        resolved = m.resolve_gate_expected(row)
        assert resolved["effective_expected"] == row["r4_expected"]
        assert resolved["effective_expected"].startswith(row["r3_expected"] + " ")
        assert resolved["adapted"] is True
        assert resolved["r4_adaptation_reason"] == row["r4_adaptation_reason"]
        for field in ("id", "name", "disposition", "original_input", "r1_input_sha256",
                      "r2_expected_sha256", "r3_expected", "reason", "authority_sha256", "status"):
            assert resolved["retained"][field] == row[field]


def test_resolve_gate_expected_falls_back_to_unchanged_r3():
    m = _m()
    statuses = {}
    for row in EXPORT["rows"]:
        if "r4_expected" in row:
            continue
        resolved = m.resolve_gate_expected(row)
        assert resolved["effective_expected"] == row["r3_expected"]
        assert resolved["adapted"] is False
        assert resolved["r4_adaptation_reason"] is None
        assert resolved["retained"]["r3_expected"] == row["r3_expected"]
        assert resolved["retained"]["status"] == row["status"]
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    # The fallback is the unchanged historical disposition: the 29 still
    # required runtime gates plus the one owner-retired row (B03).
    assert statuses == {"REQUIRED_NOT_RUN": 29,
                        "RETIRED_BY_SELECTED_DESIGN_DIRECTION_NOT_RUNTIME_TEST": 1}


def test_resolve_gate_expected_refuses_malformed_rows():
    m = _m()
    for bad in ({}, {"id": "X"}, {"id": "X", "r3_expected": ""}, {"id": "X", "r3_expected": 3},
                {"id": "X", "r3_expected": "e", "r4_expected": ""},
                {"id": "X", "r3_expected": "e", "r4_expected": None},
                {"id": "X", "r3_expected": "e", "r4_expected": "e x"},
                {"id": "X", "r3_expected": "e", "r4_expected": "e x", "r4_adaptation_reason": ""},
                ["not", "a", "dict"]):
        with pytest.raises(ValueError):
            m.resolve_gate_expected(bad)


def test_adapted_gate_binding_is_exactly_the_ten_frozen_rows():
    m = _m()
    rows = EXPORT["rows"]
    expected = [(index, row["id"], row["r4_adaptation_reason"])
                for index, row in enumerate(rows) if "r4_expected" in row]
    assert list(m.R4_ADAPTED_GATES) == expected
    assert m.R4_ADAPTED_GATE_INDICES == (17, 22, 24, 26, 29, 33, 35, 36, 37, 38)
    assert m.R4_GATE_ROWS == len(rows) == 40
    summary = m.verify_r4_gate_binding(rows)
    assert summary["rows"] == 40 and summary["fallback_count"] == 30
    assert summary["adapted_indices"] == [17, 22, 24, 26, 29, 33, 35, 36, 37, 38]
    assert summary["adapted_ids"] == [item[1] for item in expected]


def test_gate_binding_refuses_derivation_drift():
    m = _m()
    rows = EXPORT["rows"]
    def copy_rows():
        return [dict(row) for row in rows]
    mutants = []
    mutant = copy_rows(); del mutant[17]["r4_expected"]; mutants.append(mutant)
    mutant = copy_rows(); mutant[22]["r4_expected"] = "X" + mutant[22]["r3_expected"]; mutants.append(mutant)
    mutant = copy_rows(); mutant[24]["r4_adaptation_reason"] = "R3-C99"; mutants.append(mutant)
    mutant = copy_rows(); mutant[0]["r4_expected"] = mutant[0]["r3_expected"] + " extra"
    mutant[0]["r4_adaptation_reason"] = "R3-C01"; mutants.append(mutant)
    mutant = copy_rows(); mutant[38] = dict(mutant[39]); mutants.append(mutant)  # duplicated id
    mutants.append(rows[:-1])
    for mutant in mutants:
        with pytest.raises(ValueError):
            m.verify_r4_gate_binding(mutant)
    assert m.verify_r4_gate_binding(rows)["adapted_indices"] == [17, 22, 24, 26, 29, 33, 35, 36, 37, 38]


def test_disclosure_matches_frozen_bytes_and_owner_status():
    m = _m()
    disclosure, owner = EXPORT["disclosure"], EXPORT["owner"]
    pins = m.R4_DISCLOSURE_PINS
    assert pins["known_published_gap_sha256"] == _sha(disclosure["known_published_gap"])
    assert pins["remaining_policy_gates_sha256"] == \
        hashlib.sha256(_canon(disclosure["remaining_policy_gates"])).hexdigest()
    assert pins["finding_disposition_I02_normative_sections_sha256"] == \
        _sha(disclosure["finding_disposition_I02_normative_sections"])
    assert pins["remaining_policy_gates_count"] == len(disclosure["remaining_policy_gates"]) == 7
    assert "F pause/correction NOT approved" in disclosure["known_published_gap"]
    assert "Q01 context missing/drift" in disclosure["known_published_gap"]
    assert "still unapproved" in disclosure["finding_disposition_I02_normative_sections"]
    out = m.receipt_design_disclosure()
    assert out["frozen_r4"]["known_published_gap_sha256"] == pins["known_published_gap_sha256"]
    assert out["frozen_r4"]["remaining_policy_gates_count"] == 7
    assert out["frozen_r4"]["source_json_sha256"] == EXPORT["source_shas"]["source_json_sha256"]
    assert out["disclosure_only"] is True and out["protocol_record"] is False
    status = out["owner_status"]
    assert status["F_policy"] == "APPROVED_AS_CONSERVATIVE_PAUSE"
    assert status["context_policy"] == "APPROVED_AS_CONSERVATIVE_PAUSE"
    assert status["isolated_implementation_per_r4_design"] == "AUTHORIZED"
    assert tuple(status["limits"]) == tuple(owner["limits"]) == \
        ("no DB migration", "no activation", "no production writes",
         "no legacy bridge/certificate/pin/ledger activation")
    assert status["owner_decision_sha256"] == EXPORT["source_shas"]["owner_decision_sha256"]
    assert status["native_runtime_gates"] == "REQUIRED_NOT_RUN"
    assert any("39 native runtime gates" in item for item in out["still_open"])
    assert any("NOT AUTHORIZED" in item for item in out["still_open"])
    assert "never claimed as verified" in out["truthfulness"]
    json.dumps(out)   # the disclosure view must stay JSON-serializable


def test_writer_timeline_override_binding_matches_frozen_pair():
    m = _m()
    assert m.R4_WRITER_TIMELINE_OVERRIDDEN_INDICES == (0, 5)
    assert m.R4_WRITER_TIMELINE_UNCHANGED_INDICES == (1, 2, 3, 4)
    timeline = EXPORT["writer_timeline"]
    assert len(timeline["r4"]) == len(timeline["r3"]) == 6
    for index in m.R4_WRITER_TIMELINE_UNCHANGED_INDICES:
        assert timeline["r4"][index] == timeline["r3"][index]
    for index in m.R4_WRITER_TIMELINE_OVERRIDDEN_INDICES:
        assert timeline["r4"][index] != timeline["r3"][index]
