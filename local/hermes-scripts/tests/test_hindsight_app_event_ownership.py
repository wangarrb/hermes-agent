"""Synthetic finite evidence only; these fixtures authorize no real messages."""
import copy
import importlib
import json
import multiprocessing
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hindsight_incremental_sources import digest, message_hash, snapshot_hash
from hindsight_incremental_ledger import IncrementalLedger, LedgerLockedError
import hindsight_app_event_ownership as ownership


def fixture_snapshot(view, count=1):
    messages = []
    for i in range(count):
        messages.append({'event_id': f'{view}-{i}', 'role': 'user',
                         'timestamp': '2026-10-15T00:00:00Z',
                         'origin': (f'codex-app:session:turn-{i}:item-{i}' if view == 'app'
                                    else f'native:session:turn-{i}:item-{i}'),
                         'content': ('App provenance\n' if view == 'app' else '') + f'Fixture fact {i}'})
    result = {'source_key': ('app-contract::account::session' if view == 'app'
                             else 'default:hermes:native-session'),
              'source_epoch': view + '-epoch', 'profile': 'default',
              'source_hash_before': 'a' * 64, 'source_hash_after': 'a' * 64,
              'messages': messages}
    result['snapshot_hash'] = snapshot_hash(result)
    return result


def fixture_contract(ledger, app, physical):
    """Explicit synthetic proof helper for independent integration fixtures."""
    events = []
    for i, (a, p) in enumerate(zip(app['messages'], physical['messages'])):
        identity = {'account': 'fixture-account', 'session': 'session',
                    'turn': f'turn-{i}', 'item': f'item-{i}'}
        views = [{'view': view, 'source_key': snap['source_key'],
                  'source_epoch': snap['source_epoch'], 'event_id': msg['event_id'],
                  'message': dict(msg), 'message_hash': message_hash(msg),
                  'content_prefix': 'App provenance\n' if view == 'app' else ''}
                 for view, snap, msg in [('app', app, a), ('physical', physical, p)]]
        payload = {k: p[k] for k in ('role', 'timestamp', 'content')}
        events.append({'identity': identity, 'canonical_payload': payload,
                       'canonical_payload_sha256': digest(payload), 'views': views,
                       'identity_proof': {'kind': 'native-account-session-turn-item',
                           'sha256': digest({'synthetic-native-proof': identity}),
                           'identity': identity, 'bindings': sorted([
                               {'source_key': v['source_key'], 'event_id': v['event_id'],
                                'origin': v['message']['origin'], 'message_hash': v['message_hash']}
                               for v in views], key=lambda v: v['source_key'])}})
    scopes = []
    for view, snap in [('app', app), ('physical', physical)]:
        scope = {'view': view, 'source_key': snap['source_key'],
                 'source_epoch': snap['source_epoch'], 'snapshot_hash': snap['snapshot_hash'],
                 'event_ids': [m['event_id'] for m in snap['messages']]}
        for key in ('profile', 'epoch_scope', 'scope_source_keys'):
            if key in snap:
                scope[key] = snap[key]
        scopes.append(scope)
    return {'schema': ownership.SCHEMA, 'bank': ledger.bank,
            'ledger_root': str(ledger.storage_root), 'account': 'fixture-account',
            'scopes': scopes, 'events': events}


def fixture_ledger(tmp_path, old=None):
    ledger = IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy([old] if old else [], {}, [],
                             confirm='initialize-hindsight-incremental')
    return ledger


def fixture_case(tmp_path, count=1):
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app', count), fixture_snapshot('physical', count)
    contract = fixture_contract(ledger, app, physical)
    ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    return ledger, app, physical, contract


@pytest.mark.parametrize('first_view', ['app', 'physical'])
def test_two_real_sources_capture_once_payload_unchanged_and_phase_is_not_completion(tmp_path, first_view):
    ledger, app, physical, _ = fixture_case(tmp_path, 2)
    first, second = (app, physical) if first_view == 'app' else (physical, app)
    before = copy.deepcopy(first)
    batch = ledger.capture_snapshot(first)[0]
    assert ledger.capture_snapshot(second) == []
    assert ledger.capture_snapshot(first) == []
    assert len(ledger.list_batches()) == 1
    assert json.loads(Path(batch['payload_path']).read_text())['messages'] == first['messages']
    assert first == before
    plan = ownership.plan_snapshot(ledger, second)
    assert all(r['status'] == 'owned' and r['phase'] == 'captured' and not r['completed']
               and r['operation_id'] == batch['operation_id'] for r in plan['events'])
    with ledger._connection() as con:
        assert con.execute("SELECT count(*) FROM meta WHERE key LIKE 'app-owner:%'").fetchone()[0] == 2


def test_unregistered_app_no_capture_and_no_contract_is_inferred(tmp_path):
    ledger = fixture_ledger(tmp_path)
    with pytest.raises(ownership.OwnershipAmbiguousError, match='no independently approved'):
        ledger.capture_snapshot(fixture_snapshot('app'))
    assert ledger.list_batches() == []
    assert list(ledger.payload_dir.iterdir()) == []


@pytest.mark.parametrize('case', ['contract-before-capture', 'contract-after-owner',
                                  'all-contract-metadata-after-owner', 'marker-missing', 'marker-mismatch'])
def test_missing_contract_or_required_marker_cannot_downgrade_physical_gate(tmp_path, case):
    ledger, app, physical, contract = fixture_case(tmp_path)
    pin = digest(contract)
    if case in {'contract-after-owner', 'all-contract-metadata-after-owner'}:
        ledger.capture_snapshot(app)
    with ledger.locked(), ledger._connection(write=True) as con:
        if case.startswith('contract') or case == 'all-contract-metadata-after-owner':
            con.execute('DELETE FROM meta WHERE key=?', (ownership.CONTRACT_PREFIX + pin,))
        if case in {'marker-missing', 'all-contract-metadata-after-owner'}:
            con.execute('DELETE FROM meta WHERE key=?', (ownership.REQUIRED_PREFIX + pin,))
        if case == 'marker-mismatch':
            ledger._put(con, 'meta', ownership.REQUIRED_PREFIX + pin,
                        dict(ownership._required_marker(ledger, pin), bank='wrong-bank'))
    before = len(ledger.list_batches())
    with pytest.raises(ownership.OwnershipAmbiguousError):
        ledger.capture_snapshot(physical)
    with pytest.raises(ownership.OwnershipAmbiguousError):
        ownership.plan_snapshot(ledger, physical)
    assert len(ledger.list_batches()) == before


def test_contract_and_required_marker_registration_rollback_together(tmp_path, monkeypatch):
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    contract = fixture_contract(ledger, app, physical)
    original = ledger._put
    def crash(con, table, key, value):
        if key.startswith(ownership.REQUIRED_PREFIX):
            raise RuntimeError('fixture interrupted registration')
        return original(con, table, key, value)
    monkeypatch.setattr(ledger, '_put', crash)
    with pytest.raises(RuntimeError, match='interrupted registration'):
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    with ledger._connection() as con:
        assert con.execute("SELECT count(*) FROM meta WHERE key LIKE 'app-contract%'").fetchone()[0] == 0
    monkeypatch.setattr(ledger, '_put', original)
    ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    assert len(ledger.capture_snapshot(app)) == 1
    assert ledger.capture_snapshot(physical) == []


@pytest.mark.parametrize('case', ['one-contract-lost', 'pair-lost-owner-remains',
                                  'owner-outside-events', 'owner-missing-contract', 'owner-missing-batch'])
def test_another_valid_contract_cannot_hide_missing_finite_metadata(tmp_path, case):
    ledger, app, physical, first = fixture_case(tmp_path)
    ledger.capture_snapshot(app)
    other_app, other_physical = fixture_snapshot('app'), fixture_snapshot('physical')
    for snapshot in (other_app, other_physical):
        snapshot['source_key'] += '-distinct'
        snapshot['messages'][0]['origin'] += '-distinct'
        snapshot['snapshot_hash'] = snapshot_hash(snapshot)
    second = fixture_contract(ledger, other_app, other_physical)
    second['events'][0]['identity']['item'] = 'distinct-item'
    ownership.register_contract(ledger, second, approved_contract_sha256=digest(second))
    ledger.capture_snapshot(other_app)
    first_pin = digest(first)
    canonical = ownership.canonical_event_key(first['events'][0]['identity'])
    with ledger.locked(), ledger._connection(write=True) as con:
        if case in {'one-contract-lost', 'pair-lost-owner-remains'}:
            con.execute('DELETE FROM meta WHERE key=?', (ownership.CONTRACT_PREFIX + first_pin,))
        if case == 'pair-lost-owner-remains':
            con.execute('DELETE FROM meta WHERE key=?', (ownership.REQUIRED_PREFIX + first_pin,))
        if case.startswith('owner-'):
            owner = ledger._read(con, 'meta', ownership.OWNER_PREFIX + canonical)
            if case == 'owner-outside-events':
                ledger._put(con, 'meta', ownership.OWNER_PREFIX + 'f' * 64,
                            dict(owner, canonical_key='f' * 64))
            elif case == 'owner-missing-contract':
                ledger._put(con, 'meta', ownership.OWNER_PREFIX + canonical,
                            dict(owner, contract_sha256='f' * 64))
            else:
                con.execute('DELETE FROM batches WHERE batch_id=?', (owner['batch_id'],))
    with pytest.raises(ownership.OwnershipAmbiguousError):
        ledger.capture_snapshot(other_physical)
    with pytest.raises(ownership.OwnershipAmbiguousError):
        ownership.plan_snapshot(ledger, other_physical)


@pytest.mark.parametrize('mutation', ['third-source', 'third-without-origin', 'append', 'rewrite', 'truncate'])
def test_active_contract_is_closed_world_for_both_readers(tmp_path, mutation):
    ledger, app, physical, _ = fixture_case(tmp_path, 2)
    ledger.capture_snapshot(app)
    altered = copy.deepcopy(physical)
    if mutation.startswith('third'):
        altered['source_key'] = 'default:hermes:third'
        if mutation.endswith('without-origin'):
            altered['messages'][0]['origin'] = None
    elif mutation == 'append':
        altered['messages'].append(dict(altered['messages'][0], event_id='new'))
    elif mutation == 'rewrite':
        altered['messages'][0]['content'] += '!'
    else:
        altered['messages'].pop()
    altered['snapshot_hash'] = snapshot_hash(altered)
    with pytest.raises(ownership.OwnershipAmbiguousError, match='outside finite'):
        ledger.capture_snapshot(altered)
    assert len(ledger.list_batches()) == 1


def test_existing_cli_batch_adopted_only_from_complete_payload_and_original_operation(tmp_path):
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    batch = ledger.capture_snapshot(physical)[0]
    contract = fixture_contract(ledger, app, physical)
    ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    assert ledger.capture_snapshot(app) == []
    planned = ownership.plan_snapshot(ledger, app)['events'][0]
    assert planned['phase'] == 'captured' and planned['operation_id'] == batch['operation_id']
    assert not planned['completed'] and len(ledger.list_batches()) == 1


@pytest.mark.parametrize('register_first', [True, False])
def test_real_retained_parent_with_legitimate_derived_child_remains_one_source_owner(tmp_path, register_first):
    from test_hindsight_incremental_ledger import retained_batch, postretain_receipt
    from hindsight_incremental_merge import write_generation
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    contract = fixture_contract(ledger, app, physical)
    if register_first:
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    parent = retained_batch(ledger, ledger.capture_snapshot(physical)[0])
    summary = [{'event_id': 'summary-item', 'role': 'assistant',
                'timestamp': '2026-10-15T00:00:00Z', 'origin': 'derived-summary',
                'content': 'Synthetic summary'}]
    child = ledger.capture_derived([parent['batch_id']], summary, kind='daily_summary')[0]
    # Actual receipt transitions and a real frozen manifest; no phase mutation.
    child = retained_batch(ledger, child)
    child = ledger.set_phase(child['batch_id'], 'consolidated',
                             receipt=postretain_receipt(ledger, child, 'consolidated'))
    artifact = json.loads(Path(child['receipts']['generated']['artifact_path']).read_text())
    manifest = write_generation(ledger.root / 'canonical', bank=ledger.bank,
        parent={'bank': ledger.bank, 'seed_sha256': 'b' * 64},
        observations=artifact['canonical_observations'], consumed_batch_ids=[child['batch_id']])
    for phase in ('locally_merged', 'gated', 'published'):
        child = ledger.set_phase(child['batch_id'], phase,
                                 receipt=postretain_receipt(ledger, child, phase, manifest))
    if not register_first:
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    assert ledger.capture_snapshot(app) == []
    row = ownership.plan_snapshot(ledger, app)['events'][0]
    assert row['phase'] == 'retained' and not row['completed']
    assert row['operation_id'] == parent['operation_id']
    assert len(ledger.list_batches()) == 2
    for phase in ('consolidated', 'locally_merged', 'gated', 'published'):
        parent = ledger.set_phase(parent['batch_id'], phase,
                                  receipt=postretain_receipt(ledger, parent, phase, manifest))
    assert ledger.capture_snapshot(app) == []
    assert ownership.plan_snapshot(ledger, app)['events'][0]['completed'] is True


def test_published_source_alias_still_verifies_immutable_payload_and_is_complete(tmp_path):
    from test_hindsight_incremental_ledger import seal_no_delta
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    batch = seal_no_delta(ledger, ledger.capture_snapshot(physical)[0])
    contract = fixture_contract(ledger, app, physical)
    ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    assert ledger.capture_snapshot(app) == []
    assert ownership.plan_snapshot(ledger, app)['events'][0]['completed'] is True
    Path(batch['payload_path']).write_bytes(b'{}')
    with pytest.raises(ValueError, match='payload hash'):
        ledger.capture_snapshot(app)


def test_historical_derived_child_cannot_hide_unapproved_source_parent(tmp_path):
    from test_hindsight_incremental_ledger import retained_batch
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    reviewed = retained_batch(ledger, ledger.capture_snapshot(physical)[0])
    other = copy.deepcopy(physical)
    other['source_key'] = 'outside:physical'
    other['snapshot_hash'] = snapshot_hash(other)
    unreviewed = retained_batch(ledger, ledger.capture_snapshot(other)[0])
    ledger.capture_derived([reviewed['batch_id'], unreviewed['batch_id']],
        [{'event_id': 'summary', 'role': 'assistant', 'timestamp': '2026-10-15',
          'origin': 'derived', 'content': 'Synthetic mixed summary'}])
    contract = fixture_contract(ledger, app, physical)
    with pytest.raises(ownership.OwnershipAmbiguousError, match='mixed or unauthorized'):
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))


@pytest.mark.parametrize('case', ['legacy-excluded', 'outside-source', 'mixed-batch', 'unknown-policy'])
def test_registration_cannot_turn_excluded_unknown_or_uncovered_history_into_owner(tmp_path, case):
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    ledger = fixture_ledger(tmp_path, physical if case == 'legacy-excluded' else None)
    if case == 'outside-source':
        other = copy.deepcopy(physical)
        other['source_key'] = 'unreviewed-source'
        other['snapshot_hash'] = snapshot_hash(other)
        ledger.capture_snapshot(other)
    elif case == 'mixed-batch':
        mixed = fixture_snapshot('physical', 2)
        ledger.capture_snapshot(mixed)
    elif case == 'unknown-policy':
        with ledger.locked(), ledger._connection(write=True) as con:
            ledger._put(con, 'meta', 'source-unknown-policy', {'fixture': 'unknown69'})
    contract = fixture_contract(ledger, app, physical)
    with pytest.raises(ownership.OwnershipAmbiguousError):
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))
    with ledger._connection() as con:
        assert con.execute("SELECT count(*) FROM meta WHERE key LIKE 'app-contract:%'").fetchone()[0] == 0


@pytest.mark.parametrize('case', ['pin', 'account', 'proof', 'origin', 'view-hash', 'canonical', 'scope-hash'])
def test_independent_pin_and_both_exact_view_proofs_are_required(tmp_path, case):
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    contract = fixture_contract(ledger, app, physical)
    if case == 'account':
        contract['events'][0]['identity']['account'] = 'other-account'
    elif case == 'proof':
        contract['events'][0]['identity_proof']['bindings'] = []
    elif case == 'origin':
        contract['events'][0]['views'][1]['message']['origin'] = None
    elif case == 'view-hash':
        contract['events'][0]['views'][1]['message_hash'] = 'b' * 64
    elif case == 'canonical':
        contract['events'][0]['canonical_payload']['content'] = 'different'
    elif case == 'scope-hash':
        contract['scopes'][0]['snapshot_hash'] = 'b' * 64
    pin = '0' * 64 if case == 'pin' else digest(contract)
    with pytest.raises(ownership.OwnershipAmbiguousError):
        ownership.register_contract(ledger, contract, approved_contract_sha256=pin)


def test_structured_identity_prevents_tuple_string_and_cross_account_collisions():
    base = {'account': 'a', 'session': 'b:c', 'turn': 'd', 'item': 'e'}
    assert ownership.canonical_event_key(base) != ownership.canonical_event_key(dict(base, session='b', turn='c:d'))
    assert ownership.canonical_event_key(base) != ownership.canonical_event_key(dict(base, account='other'))
    assert ownership.canonical_event_key(base) != ownership.canonical_event_key(dict(base, session='other'))


@pytest.mark.parametrize('case', ['cross-session', 'cross-turn', 'physical-origin', 'app-origin', 'cross-view-origin'])
def test_shared_native_identity_is_ambiguity_in_common_validator(tmp_path, case):
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app', 2), fixture_snapshot('physical', 2)
    if case.endswith('origin'):
        snap = physical if case == 'physical-origin' else app
        if case == 'cross-view-origin':
            snap['messages'][1]['origin'] = physical['messages'][0]['origin']
        else:
            snap['messages'][1]['origin'] = snap['messages'][0]['origin']
        snap['snapshot_hash'] = snapshot_hash(snap)
    contract = fixture_contract(ledger, app, physical)
    if case in {'cross-session', 'cross-turn'}:
        identity = contract['events'][1]['identity']
        identity['item'] = contract['events'][0]['identity']['item']
        if case == 'cross-session':
            identity['session'] = 'another-session'
    with pytest.raises(ownership.OwnershipAmbiguousError, match='shared'):
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))


def test_shared_native_item_across_independently_registered_contracts_is_rejected(tmp_path):
    ledger, _, _, _ = fixture_case(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    app['source_key'] += '-other'
    physical['source_key'] += '-other'
    for snap in (app, physical):
        snap['snapshot_hash'] = snapshot_hash(snap)
    contract = fixture_contract(ledger, app, physical)
    contract['events'][0]['identity']['session'] = 'another-session'
    with pytest.raises(ownership.OwnershipAmbiguousError, match='cross-contract shared native item'):
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))


def test_corrupt_old_payload_is_not_accepted_as_owner(tmp_path):
    ledger = fixture_ledger(tmp_path)
    app, physical = fixture_snapshot('app'), fixture_snapshot('physical')
    batch = ledger.capture_snapshot(physical)[0]
    Path(batch['payload_path']).write_bytes(b'{}')
    contract = fixture_contract(ledger, app, physical)
    with pytest.raises(ValueError):
        ownership.register_contract(ledger, contract, approved_contract_sha256=digest(contract))


def test_owner_and_source_rollback_reuses_one_durable_orphan(tmp_path, monkeypatch):
    ledger, app, physical, _ = fixture_case(tmp_path)
    original = ledger._put
    def crash(con, table, key, value):
        if key.startswith(ownership.OWNER_PREFIX):
            raise RuntimeError('fixture crash before transaction commit')
        return original(con, table, key, value)
    monkeypatch.setattr(ledger, '_put', crash)
    with pytest.raises(RuntimeError, match='fixture crash'):
        ledger.capture_snapshot(app)
    assert ledger.list_batches() == []
    with ledger._connection() as con:
        assert con.execute('SELECT count(*) FROM sources').fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM meta WHERE key LIKE 'app-owner:%'").fetchone()[0] == 0
    orphan = list(ledger.payload_dir.glob('*.json'))
    assert len(orphan) == 1
    monkeypatch.setattr(ledger, '_put', original)
    batch = ledger.capture_snapshot(app)[0]
    assert batch['payload_path'] == str(orphan[0])
    assert ledger.capture_snapshot(physical) == []
    assert len(list(ledger.payload_dir.glob('*.json'))) == 1


def _capture_worker(root, snapshot, start, queue):
    start.wait(10)
    ledger = IncrementalLedger(Path(root), 'A')
    try:
        queue.put(('ok', len(ledger.capture_snapshot(snapshot))))
    except LedgerLockedError:
        queue.put(('locked', 0))


def _crash_capture_worker(root, snapshot, point):
    ledger = IncrementalLedger(Path(root), 'A')
    original = ledger._put
    def crash(con, table, key, value):
        original(con, table, key, value)
        if ((point == 'owner' and key.startswith(ownership.OWNER_PREFIX))
                or (point == 'source' and table == 'sources')):
            os._exit(47)
    ledger._put = crash
    ledger.capture_snapshot(snapshot)


@pytest.mark.parametrize('point', ['owner', 'source'])
def test_process_crash_cannot_commit_owner_without_coverage_or_batch(tmp_path, point):
    ledger, app, physical, _ = fixture_case(tmp_path)
    ctx = multiprocessing.get_context('fork')
    worker = ctx.Process(target=_crash_capture_worker, args=(ledger.storage_root, app, point))
    worker.start()
    worker.join(15)
    assert worker.exitcode == 47
    recovered = IncrementalLedger(ledger.storage_root, ledger.bank)
    assert recovered.list_batches() == []
    with recovered._connection() as con:
        assert con.execute('SELECT count(*) FROM sources').fetchone()[0] == 0
        assert con.execute("SELECT count(*) FROM meta WHERE key LIKE 'app-owner:%'").fetchone()[0] == 0
    orphan = list(recovered.payload_dir.glob('*.json'))
    assert len(orphan) == 1
    batch = recovered.capture_snapshot(app)[0]
    assert batch['payload_path'] == str(orphan[0])
    assert recovered.capture_snapshot(physical) == []
    assert len(recovered.list_batches()) == 1


def test_concurrent_readers_have_one_bank_owner_and_one_payload(tmp_path):
    ledger, app, physical, _ = fixture_case(tmp_path)
    ctx = multiprocessing.get_context('fork')
    start, queue = ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=_capture_worker, args=(ledger.storage_root, snap, start, queue))
               for snap in (app, physical)]
    for worker in workers:
        worker.start()
    start.set()
    results = [queue.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    # A lock loser retries the same source rather than manufacturing a new owner.
    assert ledger.capture_snapshot(app) == []
    assert ledger.capture_snapshot(physical) == []
    assert sum(r[1] for r in results) == 1
    assert len(ledger.list_batches()) == 1
    with ledger._connection() as con:
        assert con.execute("SELECT count(*) FROM meta WHERE key LIKE 'app-owner:%'").fetchone()[0] == 1
