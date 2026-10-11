"""Real filesystem/SQLite contracts for preserve-no-replay capture."""
import hashlib
import importlib
import json
import multiprocessing
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def api():
    assert (ROOT / 'hindsight_incremental_ledger.py').exists(), 'arrival ledger is not implemented'
    return importlib.import_module('hindsight_incremental_ledger')


def snap(*ids, epoch='epoch-1', key='default:hermes:session-1', date='2000-01-01'):
    sources = importlib.import_module('hindsight_incremental_sources')
    value = {'source_key': key, 'source_epoch': epoch, 'profile': 'default',
             'source_hash_before': 'a' * 64, 'source_hash_after': 'a' * 64,
             'messages': [{'event_id': i, 'role': 'user', 'timestamp': date,
                           'origin': i, 'content': 'A durable technical fact: ' + i} for i in ids]}
    value['snapshot_hash'] = sources.snapshot_hash(value)
    return value


def initialized(tmp_path, old=None, bank='A'):
    ledger = api().IncrementalLedger(tmp_path / 'ledger', bank, create=True)
    result = ledger.initialize_legacy([old] if old else [], {'seed_hash': 'b' * 64},
                                    [{'reason': 'coverage_unproven'}],
                                    confirm='initialize-hindsight-incremental')
    assert result['policy'] == 'preserve-no-replay'
    assert result['legacy_status'] == 'legacy_unknown'
    return ledger


def _try_lock(root, bank, queue):
    try:
        with api().IncrementalLedger(Path(root), bank).locked():
            queue.put('acquired')
    except api().LedgerLockedError:
        queue.put('locked')


def _abort_after_batch_write(root):
    ledger = api().IncrementalLedger(Path(root), 'A')
    original = ledger._store_batch
    def abort(con, batch):
        original(con, batch)
        os._exit(42)
    ledger._store_batch = abort
    ledger.capture_snapshot(snap('one'))


def test_missing_state_and_uninitialized_state_fail_without_migration(tmp_path):
    mod = api()
    missing = tmp_path / 'missing'
    with pytest.raises(mod.LedgerNotInitializedError):
        mod.IncrementalLedger(missing, 'A')
    assert not missing.exists()
    ledger = mod.IncrementalLedger(missing, 'A', create=True)
    with pytest.raises(mod.LedgerNotInitializedError):
        ledger.capture_snapshot(snap('new'))
    with pytest.raises(ValueError, match='confirm'):
        ledger.initialize_legacy([], {}, [], confirm='yes')


def test_legacy_is_exclusion_not_success_then_tail_and_late_event_capture_once(tmp_path):
    old = snap('old')
    ledger = initialized(tmp_path, old)
    assert ledger.capture_snapshot(old) == []
    assert ledger.list_batches() == []
    first = ledger.capture_snapshot(snap('old', 'tail'))
    late = snap('old', 'tail', 'late')
    late['messages'][-1]['timestamp'] = '1990-01-01'
    late['snapshot_hash'] = importlib.import_module('hindsight_incremental_sources').snapshot_hash(late)
    second = ledger.capture_snapshot(late)
    assert [e for b in first + second for e in b['source_event_ids']] == ['tail', 'late']
    assert first[0]['document_id'] != second[0]['document_id']
    assert ledger.capture_snapshot(late) == []
    body = json.loads(Path(first[0]['payload_path']).read_text())
    assert [m['event_id'] for m in body['messages']] == ['tail']
    assert body['bank'] == 'A'
    assert 'old' not in first[0]['source_event_ids']


def test_bank_A_B_A_and_model_provenance_do_not_reopen_captured_input(tmp_path):
    mod = api()
    a = initialized(tmp_path, bank='A')
    batch = a.capture_snapshot(snap('one'))[0]
    b = initialized(tmp_path, bank='B')
    other = b.capture_snapshot(snap('one'))[0]
    assert batch['document_id'] != other['document_id']
    assert batch['operation_id'] != other['operation_id']
    assert a.root == a.bank_dir
    assert b.root == b.bank_dir
    changed = snap('one')
    changed.update(model='different', label='week-new', artifact_path='copy.json')
    assert mod.IncrementalLedger(tmp_path / 'ledger', 'A').capture_snapshot(changed) == []


def test_restart_preserves_uuid_receipts_and_defer_last_completed_phase(tmp_path):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    uuid.UUID(batch['operation_id'])
    generated = generated_receipt(ledger, batch)
    ledger.set_phase(batch['batch_id'], 'generated', receipt=generated)
    ledger.defer(batch['batch_id'], error_class='transport', next_retry_at=100, attempts=2)
    recovered = api().IncrementalLedger(tmp_path / 'ledger', 'A')
    current = recovered.get_batch(batch['batch_id'])
    assert current['phase'] == 'generated'
    assert current['operation_id'] == batch['operation_id']
    assert current['receipts']['generated']['payload_hash'] == generated['payload_hash']
    assert recovered.pending_batches(now=99) == []
    assert recovered.pending_batches(now=100)[0]['batch_id'] == batch['batch_id']
    for phase in ('submitted', 'unknown', 'retained', 'consolidated', 'locally_merged', 'gated', 'published'):
        receipt = {k: generated[k] for k in ('bank', 'batch_id', 'operation_id', 'payload_hash')}
        if phase == 'retained':
            receipt.update(bank='A', batch_id=batch['batch_id'], operation_id=batch['operation_id'],
                           input_hash=batch['payload_hash'], status='completed', children=[])
        if phase in ('consolidated', 'locally_merged', 'gated', 'published'):
            receipt = postretain_receipt(recovered, recovered.get_batch(batch['batch_id']), phase)
        recovered.set_phase(batch['batch_id'], phase, receipt=receipt)
    assert recovered.pending_batches(now=200) == []
    with pytest.raises(ValueError, match='phase'):
        recovered.set_phase(batch['batch_id'], 'generated', receipt={'proof': 'again'})


def test_receipt_identity_mismatch_and_uuid_rebinding_are_rejected(tmp_path):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    for receipt in ({'bank': 'B'}, {'payload_hash': 'f' * 64}, {'operation_id': str(uuid.uuid4())}):
        with pytest.raises(ValueError, match='mismatch'):
            ledger.set_phase(batch['batch_id'], 'generated', receipt=receipt)
    generated = generated_receipt(ledger, batch)
    ledger.set_phase(batch['batch_id'], 'generated', receipt=generated)
    with pytest.raises(ValueError, match='mismatch'):
        ledger.set_phase(batch['batch_id'], 'generated', receipt={**generated, 'operation_payload_hash': 'e' * 64})


def test_record_boundary_limits_and_pending_budget(tmp_path):
    ledger = initialized(tmp_path)
    value = snap('a', 'b', 'c')
    size = len(value['messages'][0]['content'])
    batches = ledger.capture_snapshot(value, max_input_chars=size)
    assert len(batches) == 3
    assert len(ledger.pending_batches(now=0, max_units=2)) == 2
    assert len(ledger.pending_batches(now=0, max_chars=size)) == 1
    huge = snap('d', key='huge')
    with pytest.raises(ValueError, match='payload_blocked'):
        ledger.capture_snapshot(huge, max_input_chars=1)
    assert ledger.capture_snapshot(huge)


def test_snapshot_hash_epoch_shrink_and_changed_event_fail_closed(tmp_path):
    ledger = initialized(tmp_path, snap('old'))
    changed = snap('old', 'new')
    changed['messages'][-1]['content'] = 'mutated after source fence'
    with pytest.raises(ValueError, match='hash'):
        ledger.capture_snapshot(changed)
    unstable = snap('old', 'new')
    unstable['source_hash_after'] = 'e' * 64
    with pytest.raises(ValueError, match='source_changed'):
        ledger.capture_snapshot(unstable)
    for value in (snap('old', epoch='epoch-2'), snap(), snap('old', date='2020-01-01')):
        with pytest.raises(api().SourceChangedError):
            ledger.capture_snapshot(value)
    assert ledger.list_batches() == []


def test_payload_and_state_hash_tampering_rejected(tmp_path):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    path = Path(batch['payload_path'])
    assert path.stat().st_mode & 0o777 == 0o600
    assert ledger.db_path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    path.write_text('{}')
    with pytest.raises(api().LedgerIntegrityError, match='hash'):
        ledger.get_batch(batch['batch_id'])
    with sqlite3.connect(ledger.db_path) as con:
        con.execute("UPDATE batches SET state_hash = ?", ('0' * 64,))
    with pytest.raises(api().LedgerIntegrityError, match='state'):
        ledger.list_batches()


def test_path_escape_and_symlink_rejected(tmp_path):
    mod = api()
    for bank in ('../B', '.', '..', 'A/B', 'A\nB'):
        with pytest.raises(ValueError):
            mod.IncrementalLedger(tmp_path / 'ledger', bank, create=True)
    outside = tmp_path / 'outside'
    outside.mkdir()
    link = tmp_path / 'link'
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(mod.LedgerIntegrityError, match='symlink'):
        mod.IncrementalLedger(link / 'state', 'A', create=True)
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    payload = Path(batch['payload_path'])
    payload.unlink()
    payload.symlink_to(outside / 'secret')
    with pytest.raises(mod.LedgerIntegrityError):
        ledger.get_batch(batch['batch_id'])


def test_live_bank_lock_blocks_other_process_without_stealing(tmp_path):
    ledger = initialized(tmp_path)
    initialized(tmp_path, bank='B')
    ctx = multiprocessing.get_context('fork')
    with ledger.locked():
        for bank, expected in [('A', 'locked'), ('B', 'acquired')]:
            queue = ctx.Queue()
            child = ctx.Process(target=_try_lock, args=(str(tmp_path / 'ledger'), bank, queue))
            child.start()
            assert queue.get(timeout=10) == expected
            child.join(10)
            assert child.exitcode == 0
        with ledger.locked():
            ledger.capture_snapshot(snap('new'))


def test_capture_failure_before_database_commit_recovers_same_payload(tmp_path, monkeypatch):
    ledger = initialized(tmp_path)
    original = ledger._store_batch
    def crash(*args, **kwargs):
        raise RuntimeError('crash before coverage commit')
    monkeypatch.setattr(ledger, '_store_batch', crash)
    with pytest.raises(RuntimeError, match='crash'):
        ledger.capture_snapshot(snap('one'))
    assert ledger.list_batches() == []
    monkeypatch.setattr(ledger, '_store_batch', original)
    batch = ledger.capture_snapshot(snap('one'))[0]
    assert ledger.capture_snapshot(snap('one')) == []
    assert len(list(Path(batch['payload_path']).parent.glob('*.json'))) == 1


def test_derived_summary_requires_retained_parent_and_has_separate_identity(tmp_path):
    ledger = initialized(tmp_path)
    parent = ledger.capture_snapshot(snap('one'))[0]
    assert parent['kind'] == 'source'
    assert parent['parent_batch_ids'] == []
    summary = [{'event_id': 'summary', 'role': 'assistant', 'timestamp': '2000-01-01',
                'origin': 'summary', 'content': 'Confirmed new source summary'}]
    with pytest.raises(ValueError, match='parent'):
        ledger.capture_derived([parent['batch_id']], summary, kind='daily_summary')
    retained_batch(ledger, parent)
    child = ledger.capture_derived([parent['batch_id']], summary, kind='daily_summary')[0]
    assert child['kind'] == 'daily_summary'
    assert child['phase'] == 'captured'
    assert child['parent_batch_ids'] == [parent['batch_id']]
    assert child['document_id'] != parent['document_id']
    assert child['operation_id'] != parent['operation_id']
    assert ledger.capture_derived([parent['batch_id']], summary, kind='daily_summary') == []
    changed = [{**summary[0], 'content': 'A different model rewrote the same source'}]
    with pytest.raises(ValueError, match='derived'):
        ledger.capture_derived([parent['batch_id']], changed, kind='daily_summary')
    assert ledger.get_batch(parent['batch_id'])['phase'] == 'retained'
    forged = snap('fake')
    forged.update(kind='daily_summary', parent_batch_ids=[parent['batch_id']], phase='retained')
    with pytest.raises(ValueError, match='derived'):
        ledger.capture_snapshot(forged)


def test_abrupt_process_exit_rolls_back_coverage_and_reuses_durable_orphan(tmp_path):
    ledger = initialized(tmp_path)
    child = multiprocessing.get_context('fork').Process(target=_abort_after_batch_write,
        args=(str(tmp_path / 'ledger'),))
    child.start()
    child.join(10)
    assert child.exitcode == 42
    recovered = api().IncrementalLedger(tmp_path / 'ledger', 'A')
    assert recovered.list_batches() == []
    before = list(ledger.payload_dir.glob('*.json'))
    assert len(before) == 1
    batch = recovered.capture_snapshot(snap('one'))[0]
    assert Path(batch['payload_path']) == before[0]
    assert recovered.capture_snapshot(snap('one')) == []


def test_frozen_input_has_new_document_memory_items(tmp_path):
    ledger = initialized(tmp_path, snap('old'))
    batch = ledger.capture_snapshot(snap('old', 'new'))[0]
    payload = json.loads(Path(batch['payload_path']).read_text())
    assert payload['items'][0]['document_id'] == batch['document_id']
    assert 'new' in payload['items'][0]['content']
    assert 'old' not in payload['items'][0]['content']


def test_generated_submit_hash_is_distinct_from_input_and_bound_before_post(tmp_path):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    raw = json.dumps({'async': True, 'operation_id': batch['operation_id'],
                      'items': [{'document_id': batch['document_id'], 'content': 'new fact'}]}).encode()
    path = ledger.bank_dir / 'retain.json'
    path.write_bytes(raw)
    path.chmod(0o600)
    generated = {'bank': 'A', 'batch_id': batch['batch_id'], 'input_hash': batch['payload_hash'],
                 'operation_id': batch['operation_id'], 'payload_path': str(path),
                 'payload_hash': hashlib.sha256(raw).hexdigest(), 'artifact_kind': 'raw_source'}
    current = ledger.set_phase(batch['batch_id'], 'generated', receipt=generated)
    assert current['operation_payload_hash'] == generated['payload_hash']
    binding = {k: generated[k] for k in ('bank', 'batch_id', 'operation_id', 'payload_hash')}
    ledger.set_phase(batch['batch_id'], 'submitted', receipt=binding)
    ledger.set_phase(batch['batch_id'], 'unknown', receipt=binding)
    # Reconcile + same-UUID retry cannot regress the uncertain phase.
    ledger.set_phase(batch['batch_id'], 'submitted', receipt=binding)
    assert ledger.get_batch(batch['batch_id'])['phase'] == 'unknown'
    receipt = {'bank': 'A', 'batch_id': batch['batch_id'], 'operation_id': batch['operation_id'],
               'input_hash': batch['payload_hash'], 'payload_hash': generated['payload_hash'],
               'status': 'completed', 'children': []}
    ledger.set_phase(batch['batch_id'], 'retained', receipt=receipt)


def test_failed_parent_receipt_cannot_authenticate_derived_summary(tmp_path):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    generated = generated_receipt(ledger, batch)
    ledger.set_phase(batch['batch_id'], 'generated', receipt=generated)
    ledger.set_phase(batch['batch_id'], 'submitted', receipt={k: generated[k] for k in
        ('bank', 'batch_id', 'operation_id', 'payload_hash')})
    with pytest.raises(ValueError, match='receipt'):
        ledger.set_phase(batch['batch_id'], 'retained', receipt={'status': 'failed'})


@pytest.mark.parametrize('child_id', [True, 123, None, [], {}, '', ' ', 'child\nid'])
def test_retained_child_requires_literal_nonempty_string_identity(tmp_path, child_id):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    generated = generated_receipt(ledger, batch)
    ledger.set_phase(batch['batch_id'], 'generated', receipt=generated)
    binding = {k: generated[k] for k in ('bank', 'batch_id', 'operation_id', 'payload_hash')}
    ledger.set_phase(batch['batch_id'], 'submitted', receipt=binding)
    before = ledger.db_path.read_bytes()
    receipt = {**binding, 'input_hash': batch['payload_hash'], 'status': 'completed',
        'children': [{'bank': ledger.bank, 'status': 'completed', 'operation_id': child_id, 'children': []}]}
    with pytest.raises(ValueError, match='child receipt'):
        ledger.set_phase(batch['batch_id'], 'retained', receipt=receipt)
    assert ledger.db_path.read_bytes() == before
    assert ledger.get_batch(batch['batch_id'])['phase'] == 'submitted'


def test_noop_capture_does_not_rewrite_ledger_state(tmp_path):
    ledger = initialized(tmp_path, snap('old'))
    before = ledger.db_path.read_bytes()
    assert ledger.capture_snapshot(snap('old')) == []
    assert ledger.db_path.read_bytes() == before
    audit = ledger.initialization_manifest()
    assert audit['excluded_source_events'] == 1
    assert audit['legacy_status'] == 'legacy_unknown'


def generated_receipt(ledger, batch):
    from hindsight_incremental_delivery import _freeze
    payload = {'async': True, 'operation_id': batch['operation_id'],
               'items': [{'document_id': batch['document_id'], 'content': 'A new confirmed fact'}]}
    path, checksum = _freeze(ledger.root, payload, 'test-retain')
    result = {'bank': ledger.bank, 'batch_id': batch['batch_id'], 'operation_id': batch['operation_id'],
              'input_hash': batch['payload_hash'], 'payload_path': path, 'payload_hash': checksum,
              'artifact_kind': 'raw_source', 'artifact_path': None, 'artifact_hash': None}
    if batch['kind'] != 'source':
        result['artifact_kind'] = 'summary'
        result['artifact_path'], result['artifact_hash'] = _freeze(ledger.root,
            {'canonical_observations': [{'bank': ledger.bank, 'topic': 'technical', 'type': 'fact',
                'insight': 'A new confirmed fact', 'applicability': 'measured', 'evidence_ids': ['fact-1']}]}, 'test-summary')
    return result


def retained_batch(ledger, batch):
    generated = generated_receipt(ledger, batch)
    ledger.set_phase(batch['batch_id'], 'generated', receipt=generated)
    binding = {k: generated[k] for k in ('bank', 'batch_id', 'operation_id', 'payload_hash')}
    ledger.set_phase(batch['batch_id'], 'submitted', receipt=binding)
    return ledger.set_phase(batch['batch_id'], 'retained', receipt={**binding,
        'input_hash': batch['payload_hash'], 'status': 'completed', 'children': []})


def postretain_receipt(ledger, batch, phase, manifest=None):
    from hindsight_incremental_sources import digest
    base = {'bank': ledger.bank, 'batch_id': batch['batch_id']}
    if phase == 'consolidated':
        return {**base, 'scoped_proof': {'bank': ledger.bank,
            'scope': [f'arrival-batch:{batch["batch_id"]}'], 'pending_in_scope': 0, 'failed_in_scope': 0,
            'ready': True, 'operation_ids': []}}
    if manifest is None:
        return {**base, 'no_canonical_delta': True,
                'retained_receipt_hash': digest(batch['receipts']['retained'])}
    gate = {'bank': ledger.bank, 'manifest_sha256': manifest['manifest_sha256'],
        'decision': 'eligible_for_local_proposal', 'checks': {key: True for key in
        ('schema', 'coverage', 'conflict', 'generic_eval', 'local_eval', 'scoped_drain')}}
    if phase == 'locally_merged':
        return {**base, 'manifest': manifest}
    if phase == 'gated':
        return {**base, 'manifest': manifest, 'gate': gate}
    return {**base, 'manifest': manifest, 'sequence': 1, 'publication': {'bank': ledger.bank,
        'manifest_sha256': manifest['manifest_sha256'], 'published': True,
        'readback_verified': True, 'legacy_seed_preserved': True}}


def seal_no_delta(ledger, batch):
    batch = retained_batch(ledger, batch)
    for phase in ('consolidated', 'locally_merged', 'gated', 'published'):
        batch = ledger.set_phase(batch['batch_id'], phase, receipt=postretain_receipt(ledger, batch, phase))
    return batch


def _abort_in_payload_publish_window(root):
    ledger = api().IncrementalLedger(Path(root), 'A')
    # Abort at the real publication syscall, before any cleanup/coverage commit.
    for name in ('link', 'rename', 'replace'):
        original = getattr(os, name)
        def abort(*args, _original=original, **kwargs):
            _original(*args, **kwargs)
            os._exit(43)
        setattr(os, name, abort)
    ledger.capture_snapshot(snap('one'))


def test_payload_publish_window_abrupt_exit_leaves_single_link_recoverable_orphan(tmp_path, monkeypatch):
    ledger = initialized(tmp_path)
    child = multiprocessing.get_context('fork').Process(target=_abort_in_payload_publish_window,
        args=(str(tmp_path / 'ledger'),))
    child.start()
    child.join(10)
    assert child.exitcode == 43
    files = list(ledger.payload_dir.glob('*.json'))
    assert len(files) == 1
    assert files[0].stat().st_nlink == 1
    recovered = api().IncrementalLedger(tmp_path / 'ledger', 'A')
    assert recovered.list_batches() == []
    import stat
    directory_syncs = []
    original = os.fsync
    def synced(fd):
        info = os.fstat(fd)
        if stat.S_ISDIR(info.st_mode):
            directory_syncs.append(info.st_ino)
        return original(fd)
    monkeypatch.setattr(os, 'fsync', synced)
    assert recovered.capture_snapshot(snap('one'))[0]['payload_path'] == str(files[0])
    assert recovered.payload_dir.stat().st_ino in directory_syncs
    assert recovered.capture_snapshot(snap('one')) == []
    os.link(files[0], tmp_path / 'external-alias.json')
    with pytest.raises(api().LedgerIntegrityError, match='hard-linked'):
        recovered.get_batch(recovered.list_batches()[0]['batch_id'])


@pytest.mark.parametrize('bad', [{'operation_payload_hash': 'd' * 64}, {'artifact_hash': 'c' * 64},
    {'status': 'failed'}, {'status': 'unknown'}, {'readback_verified': False}])
def test_generated_requires_frozen_operation_payload_not_self_declared_proof(tmp_path, bad):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    with pytest.raises(ValueError, match='receipt|generated'):
        ledger.set_phase(batch['batch_id'], 'generated', receipt=bad)
    assert ledger.get_batch(batch['batch_id'])['phase'] == 'captured'


@pytest.mark.parametrize('phase', ['consolidated', 'locally_merged', 'gated', 'published'])
@pytest.mark.parametrize('bad', [{'status': 'failed'}, {'status': 'unknown'}, {'readback_verified': False}])
def test_postretained_negative_receipt_cannot_advance_or_seal(tmp_path, phase, bad):
    ledger = initialized(tmp_path)
    batch = retained_batch(ledger, ledger.capture_snapshot(snap('one'))[0])
    stages = ('consolidated', 'locally_merged', 'gated', 'published')
    for before in stages[:stages.index(phase)]:
        batch = ledger.set_phase(batch['batch_id'], before, receipt=postretain_receipt(ledger, batch, before))
    before_phase = batch['phase']
    with pytest.raises(ValueError, match='receipt|proof'):
        ledger.set_phase(batch['batch_id'], phase, receipt={**postretain_receipt(ledger, batch, phase), **bad})
    assert ledger.get_batch(batch['batch_id'])['phase'] == before_phase
    assert ledger.pending_batches(now=0)


def test_parent_contributions_cannot_be_consumed_by_overlapping_derived_groups(tmp_path):
    ledger = initialized(tmp_path)
    batches = ledger.capture_snapshot(snap('A', 'B', 'C'), max_input_chars=27)
    assert len(batches) == 3
    for batch in batches:
        retained_batch(ledger, batch)
    parents = [b['batch_id'] for b in batches]
    summary = [{'event_id': 'summary', 'role': 'assistant', 'timestamp': '2000-01-01',
                'origin': 'summary', 'content': 'A confirmed summary of only new facts'}]
    first = ledger.capture_derived(parents[:2], summary)
    recovered = api().IncrementalLedger(tmp_path / 'ledger', 'A')
    assert recovered.capture_derived(parents[:2], summary) == []
    for overlap in ([parents[0]], parents[1:], parents):
        with pytest.raises(ValueError, match='consum'):
            recovered.capture_derived(overlap, summary)
    assert recovered.capture_derived([parents[2]], summary)
    assert len([b for b in recovered.list_batches() if b['kind'] == 'daily_summary']) == len(first) + 1


def test_missing_sealed_raw_payload_diagnosed_without_blocking_new_pending(tmp_path):
    ledger = initialized(tmp_path)
    sealed = seal_no_delta(ledger, ledger.capture_snapshot(snap('old'))[0])
    Path(sealed['payload_path']).unlink()
    recovered = api().IncrementalLedger(tmp_path / 'ledger', 'A')
    assert recovered.get_batch(sealed['batch_id'])['phase'] == 'published'
    assert recovered.pending_batches(now=0) == []
    delta = recovered.capture_snapshot(snap('old', 'new'))
    assert [b['batch_id'] for b in recovered.pending_batches(now=0)] == [delta[0]['batch_id']]
    diagnostics = recovered.payload_diagnostics()
    assert diagnostics == [{'bank': 'A', 'batch_id': sealed['batch_id'], 'phase': 'published', 'status': 'missing'}]


def canonical_summary(ledger):
    from hindsight_incremental_merge import write_generation
    parent = retained_batch(ledger, ledger.capture_snapshot(snap('one'))[0])
    messages = [{'event_id': 'summary', 'role': 'assistant', 'timestamp': '2000-01-01',
                 'origin': 'summary', 'content': 'Confirmed new summary'}]
    batch = retained_batch(ledger, ledger.capture_derived([parent['batch_id']], messages)[0])
    batch = ledger.set_phase(batch['batch_id'], 'consolidated', receipt=postretain_receipt(ledger, batch, 'consolidated'))
    artifact = json.loads(Path(batch['receipts']['generated']['artifact_path']).read_text())
    manifest = write_generation(ledger.root / 'canonical', bank=ledger.bank,
        parent={'bank': ledger.bank, 'seed_sha256': 'b' * 64},
        observations=artifact['canonical_observations'], consumed_batch_ids=[batch['batch_id']])
    return parent, batch, manifest


@pytest.mark.parametrize('negative', [{'status': 'failed'}, {'status': 'unknown'}, {'readback_verified': False}])
def test_generated_even_real_frozen_artifact_cannot_carry_negative_proof(tmp_path, negative):
    ledger = initialized(tmp_path)
    batch = ledger.capture_snapshot(snap('one'))[0]
    receipt = {**generated_receipt(ledger, batch), **negative}
    with pytest.raises(ValueError, match='receipt|proof'):
        ledger.set_phase(batch['batch_id'], 'generated', receipt=receipt)
    assert ledger.get_batch(batch['batch_id'])['phase'] == 'captured'


@pytest.mark.parametrize('field,value', [('bank', 'B'), ('scope', ['global']),
    ('pending_in_scope', False), ('pending_in_scope', None), ('pending_in_scope', 1),
    ('ready', False), ('failed_in_scope', 1), ('pending_operations', False)])
def test_consolidation_requires_exact_positive_scope_not_nonempty_dict(tmp_path, field, value):
    ledger = initialized(tmp_path)
    batch = retained_batch(ledger, ledger.capture_snapshot(snap('one'))[0])
    receipt = postretain_receipt(ledger, batch, 'consolidated')
    receipt['scoped_proof'][field] = value
    with pytest.raises(ValueError, match='receipt|proof'):
        ledger.set_phase(batch['batch_id'], 'consolidated', receipt=receipt)
    assert ledger.get_batch(batch['batch_id'])['phase'] == 'retained'


@pytest.mark.parametrize('missing', ['failed_in_scope', 'operation_ids'])
def test_consolidation_missing_real_native_proof_fields_is_unknown_not_success(tmp_path, missing):
    ledger = initialized(tmp_path)
    batch = retained_batch(ledger, ledger.capture_snapshot(snap('one'))[0])
    receipt = postretain_receipt(ledger, batch, 'consolidated')
    receipt['scoped_proof'].pop(missing)
    with pytest.raises(ValueError, match='receipt|proof'):
        ledger.set_phase(batch['batch_id'], 'consolidated', receipt=receipt)
    assert ledger.get_batch(batch['batch_id'])['phase'] == 'retained'


def test_canonical_stage_schema_binds_exact_manifest_gate_and_readback_then_parent_coverage(tmp_path):
    ledger = initialized(tmp_path)
    parent, batch, manifest = canonical_summary(ledger)
    for weak in ({'proof': 'merged'}, {'manifest': {'bank': 'A'}}, {'manifest': {**manifest, 'bank': 'B'}}):
        with pytest.raises(ValueError, match='receipt|manifest'):
            ledger.set_phase(batch['batch_id'], 'locally_merged', receipt={'bank': 'A', 'batch_id': batch['batch_id'], **weak})
    batch = ledger.set_phase(batch['batch_id'], 'locally_merged', receipt=postretain_receipt(ledger, batch, 'locally_merged', manifest))
    gate_receipt = postretain_receipt(ledger, batch, 'gated', manifest)
    for changed in ({'bank': 'B'}, {'manifest_sha256': 'f' * 64}, {'decision': 'blocked_keep_local_only'},
                    {'checks': {}}, {'checks': {**gate_receipt['gate']['checks'], 'coverage': False}}, {'status': 'unknown'}):
        with pytest.raises(ValueError, match='receipt|proof'):
            ledger.set_phase(batch['batch_id'], 'gated', receipt={**gate_receipt, 'gate': {**gate_receipt['gate'], **changed}})
    batch = ledger.set_phase(batch['batch_id'], 'gated', receipt=gate_receipt)
    published = postretain_receipt(ledger, batch, 'published', manifest)
    for changed in ({'bank': 'B'}, {'manifest_sha256': 'f' * 64}, {'published': False},
                    {'readback_verified': False}, {'legacy_seed_preserved': False}, {'status': 'failed'}):
        with pytest.raises(ValueError, match='receipt|proof'):
            ledger.set_phase(batch['batch_id'], 'published', receipt={**published, 'publication': {**published['publication'], **changed}})
    with pytest.raises(ValueError, match='receipt|proof'):
        ledger.set_phase(batch['batch_id'], 'published', receipt={**published, 'sequence': True})
    # Raw source cannot use a summary candidate until every child is read back.
    parent = ledger.set_phase(parent['batch_id'], 'consolidated', receipt=postretain_receipt(ledger, parent, 'consolidated'))
    with pytest.raises(ValueError, match='coverage'):
        ledger.set_phase(parent['batch_id'], 'locally_merged', receipt=postretain_receipt(ledger, parent, 'locally_merged', manifest))
    ledger.set_phase(batch['batch_id'], 'published', receipt=published)
    for phase in ('locally_merged', 'gated', 'published'):
        parent = ledger.set_phase(parent['batch_id'], phase, receipt=postretain_receipt(ledger, parent, phase, manifest))
    reopened = api().IncrementalLedger(tmp_path / 'ledger', 'A')
    assert reopened.pending_batches(now=0) == []
    assert reopened.capture_snapshot(snap('one')) == []


def test_no_delta_seal_requires_exact_retained_hash_and_cannot_forge_publication(tmp_path):
    ledger = initialized(tmp_path)
    batch = retained_batch(ledger, ledger.capture_snapshot(snap('one'))[0])
    batch = ledger.set_phase(batch['batch_id'], 'consolidated', receipt=postretain_receipt(ledger, batch, 'consolidated'))
    proof = postretain_receipt(ledger, batch, 'locally_merged')
    for changed in ({'retained_receipt_hash': 'f' * 64}, {'published': True},
                    {'publication': {'published': True}}, {'no_canonical_delta': False}):
        with pytest.raises(ValueError, match='receipt'):
            ledger.set_phase(batch['batch_id'], 'locally_merged', receipt={**proof, **changed})
    for phase in ('locally_merged', 'gated', 'published'):
        batch = ledger.set_phase(batch['batch_id'], phase, receipt=postretain_receipt(ledger, batch, phase))
    assert batch['receipts']['published']['no_canonical_delta'] is True
    assert 'publication' not in batch['receipts']['published']
    assert ledger.pending_batches(now=0) == []
    with pytest.raises(ValueError, match='parent|sealed'):
        ledger.capture_derived([batch['batch_id']], [{'event_id': 'late', 'role': 'assistant',
            'origin': 'late', 'timestamp': '2000-01-01', 'content': 'Cannot reopen sealed input'}])


def test_bootstrap_reinitialization_is_not_a_now_watermark(tmp_path):
    ledger = initialized(tmp_path, snap('old'))
    with pytest.raises(ValueError, match='already'):
        ledger.initialize_legacy([snap('old', 'new')], {}, [], confirm='initialize-hindsight-incremental')
    assert ledger.capture_snapshot(snap('old', 'new'))[0]['source_event_ids'] == ['new']
