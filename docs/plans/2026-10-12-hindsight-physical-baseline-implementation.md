# Hindsight physical legacy baseline implementation

> Agentic workers use the existing subagent-driven-development / executing-plans workflow. Source and ledger have disjoint ownership; parent owns pipeline integration and production acceptance.

Goal: implement approved arrival design §10.3–10.4 for pre-cutover inputs whose logical identity cannot be proved, without weakening identity for future arrivals.
Architecture: one private immutable physical exclusion artifact, bound to bank/home and source feed epochs, referenced by the existing bank ledger. It proves finite observation/exclusion, never successful historical processing. Reuse secure freezing, SHA-256, the producer lock and SQLite metadata; no new service, dependency, provider, behavior env var or per-profile process.
Tech stack: existing stdlib SQLite / JSON / hashlib, incremental source/ledger/pipeline, canonical test harness.

## Global constraints

- Preserve bank data, canonical seed, vector bytes and all source/profile coverage. No production native/LLM/embedding/publish calls in implementation tests.
- No replay or success receipt for legacy_unknown. Native row IDs and JSON positions locate exclusions only; never become future logical event IDs.
- No guessed timestamp, root, timezone, lineage or clone relation. Future ambiguity is source_changed, not automatic exclusion or admission.
- All-path coverage, epoch/inode replacement, same-inode reset, missing prefix records and JSON truncation remain fail-closed.
- Keep normal no-op free of SQL backend/credentials/native/LLM requirements; local source reads and immutable artifact validation remain necessary.
- Working tree: .worktrees/hindsight-physical-baseline, based on d81193180cc6660c10b1ffb737c968c589d7abf0 and the recorded nine-file streaming snapshot. Do not stage/commit/push or touch unrelated main modifications.
- Original daily/weekly writers stay paused; five-minute read-only monitor remains active. Production full follows only after final review and concrete boundary validation, within 12 units / 500000 chars / 16 offline LLM requests and the unchanged native call limit.

## Task 1: finite physical source descriptor and normal reader

Files: modify local/hermes-scripts/hindsight_incremental_sources.py and tests/test_hindsight_incremental_sources.py only.
Interfaces:
    capture_physical_baseline(hermes_home: Path, *, bank: str, profile_mode: str = 'all') -> dict
    validate_physical_baseline(baseline: dict, *, bank: str, hermes_home: Path) -> None
    physical_baseline_snapshots(baseline: dict) -> list[dict]
    scan_source_snapshots(hermes_home: Path, *, profile_mode: str = 'all', physical_baseline: dict | None = None) -> list[dict]

Descriptor header is schema_version='hindsight-physical-legacy-baseline/1', bank, hermes_home, status='legacy_unknown', completion_proven=False, and profiles. The concrete per-feed descriptor is owned by Task 1; other components treat it opaquely after validation. Store raw native identity fields and exact body/row hashes, not raw bodies, prompts, reasoning or credentials. Use exact source paths, complete physical record coverage, feed epoch and raw native session associations. Read SQLite with mode=ro / query_only within one snapshot transaction; double-read each JSON export and preserve its raw fence hash. Do not run _message or whole historical _lineage_root merely to exclude legacy physical records.

- [ ] Reproduce malformed historical logical identity rejection in isolated SQLite/JSON fixtures.
- [ ] Capture a finite descriptor without inventing logical events. All profiles, DB-only feeds and old malformed inputs remain represented, not filtered.
- [ ] Emit empty baseline snapshots using explicitly named native cutover-session anchors, not claimed historical roots. Existing feed epoch/source-key coverage fencing is retained.
- [ ] In normal scans, verify the frozen descriptor then exclude only exact legacy locations with matching content binding. No broad id<=H/mtime/text blacklist. Compare every bound old row/file prefix; reject replacement, in-place reset, deletion, prefix rewrite and truncation.
- [ ] Strictly process future records. A real new origin with old text/event_date is admitted once; an exact no-origin new row matching old unknown material cannot silently no-op. Verified future cross-session compression/exports may be no-op; unprovable same-session clone or rewritten export blocks. Old malformed history cannot fabricate a new root; a native cutover anchor is explicitly nonhistorical and future forward lineage must be proved.
- [ ] Run old source tests plus isolated physical cases through canonical harness, using distinct scratch roots.

Example boundary falsifiers (the fixture creates ordinary temporary source files, never production paths):
    legacy = capture_physical_baseline(home, bank='bank-a')
    validate_physical_baseline(legacy, bank='bank-a', hermes_home=home)
    assert all(not s['messages'] for s in physical_baseline_snapshots(legacy))
    assert all(not s['messages'] for s in scan_source_snapshots(home, physical_baseline=legacy))
    append_new_native_origin(home, origin='new-origin', event_date='2020-01-01T00:00:00Z')
    assert admitted_origins(scan_source_snapshots(home, physical_baseline=legacy)) == {'new-origin'}
    append_unprovable_same_body_time_row(home)
    with raises(SourceChangedError):
        scan_source_snapshots(home, physical_baseline=legacy)

## Task 2: exact ledger binding and crash-safe noncompletion

Files: modify hindsight_incremental_ledger.py and tests/test_hindsight_incremental_ledger.py only.
Interfaces:
    initialize_legacy(source_snapshots, seed, unknown, *, confirm, physical_baseline=None) -> dict
    physical_baseline_reference() -> dict | None

Reference fields: schema_version='hindsight-physical-legacy-baseline/1', bank, hermes_home, artifact_path, artifact_hash. The private artifact holds Task 1's complete descriptor. Validate bank/home/path/hash/type/policy binding before committing initialization. Preserve existing valid logical-baseline callers when no physical reference is provided. Bind the reference atomically in meta/legacy with seed identity and empty source-anchor snapshots; no batch and no success stage is created. Accessor returns the immutable reference, not physical material.

- [ ] RED: malformed/tampered/wrong-bank/outside-root/private-mode/alias reference cannot initialize.
- [ ] Reuse existing lock, private immutable artifact validators and SQL transaction. No new database or external API.
- [ ] Persist physical exclusion metadata explicitly as noncompleted; physical record count is not logical event count.
- [ ] Validate crash before/after artifact freeze versus initialization transaction. Frozen valid orphans may be reused; partial/tampered artifacts and already initialized incompatible state cannot be overwritten or silently rebaselined.
- [ ] Run existing ledger regressions and new physical binding cases, with no production sources or credentials.

Example assertions:
    manifest = ledger.initialize_legacy(empty_anchors, seed, unknown,
        confirm='initialize-hindsight-incremental', physical_baseline=reference)
    assert manifest['legacy_status'] == 'legacy_unknown'
    assert ledger.physical_baseline_reference() == reference
    assert ledger.list_batches() == []
    assert not any_success_receipts(ledger)

## Task 3: parent pipeline integration and exact end-to-end acceptance

Files: parent owns hindsight_incremental_pipeline.py and tests/test_hindsight_incremental_pipeline.py. Source and ledger implementers do not edit them.

- [ ] Add physical_baseline=False to SDK initialize for compatibility. Explicit CLI initialize selects True; no implicit baseline is created by plan/run/cron.
- [ ] Physical initialization calls Task 1 capture and validation, freezes the complete descriptor under the bank ledger private root using existing delivery _freeze, obtains empty anchor snapshots, and calls Task 2 initialize_legacy with the exact receipt. Canonical seed stays hash bound through the already verified streaming loader.
- [ ] Normal run loads Task 2 reference, reads the exact hash-bound artifact, validates bank/home and passes physical_baseline to the source reader. No fallback to the old reader if a reference is missing, malformed or mismatched; no new SQL backend/profile credentials demanded by a no-op.
- [ ] RED→GREEN integration: malformed legacy initialize gives zero batches, unchanged rescan zero paid work; real appended origin / late event admitted once; budget/run errors, snapshot growth during capture, raw and derived stage receipts keep previous contracts.
- [ ] Independent final review before production use: exact coverage, clone/late-export/overwrite ambiguity, epoch/reset/truncation and crash boundaries. Keep real native worker and publication acceptance separate from fixtures.
- [ ] Production: capture/validate finite physical descriptor using explicit approved initialization; never pay for legacy_unknown. Execute bounded new full, scoped drain and readback; verify unchanged seed/56 inherited pending/flags restoration and zero-paid no-op, then restore the two paused writer schedules. Report optional wiki degraded honestly until its bounded lane exists.

Canonical commands per file, replacing scratch root with a unique private path:
    HERMES_PYTHON=/home/wyr/miniconda/bin/python HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh -j 1 local/hermes-scripts/tests/test_hindsight_incremental_sources.py -q --tb=short --basetemp=/home/wyr/.hermes/cache/scratch/physical-sources-green
    HERMES_PYTHON=/home/wyr/miniconda/bin/python HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh -j 1 local/hermes-scripts/tests/test_hindsight_incremental_ledger.py -q --tb=short --basetemp=/home/wyr/.hermes/cache/scratch/physical-ledger-green
    HERMES_PYTHON=/home/wyr/miniconda/bin/python HERMES_TEST_FILE_RETRIES=0 scripts/run_tests.sh -j 1 local/hermes-scripts/tests/test_hindsight_incremental_pipeline.py -q --tb=short --basetemp=/home/wyr/.hermes/cache/scratch/physical-pipeline-green
