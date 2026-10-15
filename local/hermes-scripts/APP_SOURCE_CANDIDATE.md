# Finite App source candidate — fixture review only

This candidate is not deployed and does not authorize external processing.
The approved runner's App dispatch is plan-only. Its original physical dispatch
keeps every physical-source rejection. The original447 byte pins remain fixed;
the newer70 items and other sessions do not enter that authorization.

The existing ledger deduplicates within source_key. Sharing that ledger alone
does not prevent App→CLI or CLI→App double capture. App provenance headers also
make the two raw payloads different. An event-ID rewrite or equal text is not
native identity evidence.

The local candidate therefore uses a finite, independently reviewed contract
with account/session/turn/item identity, both immutable view payloads, canonical
payload binding and complete snapshot/event lists. The independent contract pin
means an operator has reviewed concrete native identity evidence; calculating a
hash of a newly generated contract does not supply that approval. A proof hash
alone is an integrity reference, not proof an alias exists.
The validator checks the proof's declared identity and both original view
bindings. It cannot verify the meaning or provenance of evidence named by that
hash; review of that evidence and approval of the exact contract are external
preconditions. The synthetic test authority supplies no production evidence.

Ownership is checked inside the existing bank lock and capture transaction.
The contract and a per-contract integrity marker are registered atomically;
missing or inconsistent pairs and orphan owners are rejected before either
reader can fall back to ordinary capture.
The first bound event owns its original payload, batch and deterministic
operation UUID; the other view records coverage without a second batch. The
original payload is never rewritten. Owner phase is reported separately:
captured, unknown or pending does not mean retained or published. Mixed batches
with unauthorized events, legacy exclusions, finite unknown bindings, changed
payloads or unresolved ownership are rejected.

The closed-world rule is deliberately conservative: once a finite contract is
registered, a source outside its complete scopes is ambiguous. Registration
also rejects existing sources/batches that the contract cannot account for.
This blocks unbound third-reader aliases; it is not a usable general production
source registry. No new bank, ledger or global alias inference system is used.
This version also refuses any physical source-unknown policy, including the
existing69 bindings: no reviewed disjointness proof exists. A larger contract
or broader session allowlist cannot repair this evidence gap by itself.
An active scope is immutable: appended or truncated snapshots are refused.
This finite candidate is not a production incremental collector; the existing
local archival collector remains the separate resumable discovery path.

The real production ledger has110 old published batches and a physical legacy
baseline whose logical events do not reconstruct old CLI ownership. No complete
native cross-view contract has been approved for the original447. Therefore
their current result is **ambiguity/no capture**, even if synthetic fixtures
pass. They must not be admitted from equal text, missing UUID overlap, a fixture
contract or a self-generated approval pin.

Required local checks cover both ingestion orders, repeat capture, view-content
conflicts, same-count raw replacement, an unbound third source, legacy/unknown
overlap, mixed unauthorized batches, account/session item collisions, atomic
owner creation and crash recovery. Unknown remote POST outcomes must reconcile
the same existing operation UUID before any further POST. Delivery and publish
remain owned by the existing runtime and gates; tests use local fakes only.

Validation completed through the repository's canonical scripts/run_tests.sh:
17 files,578 tests passed,0 failed;71 are the new ownership/adapter/delivery
cases. The set includes prior physical sources, ledger, delivery, entrypoint,
pipeline, unknown-policy, source-transition, App archival/producer and UTC+8
usage/day-boundary checks. The raw447 same-count replacement tests also passed.
All test ledgers, payloads and delivery clients were temporary local fixtures.
Independent static review found no blocking issue within this fixture-only,
plan-only scope. These results do not establish production447 readiness.
