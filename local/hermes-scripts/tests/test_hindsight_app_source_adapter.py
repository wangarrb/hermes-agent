"""Finite synthetic native proofs; no fixture authorizes production App history."""
import copy
import json
from pathlib import Path
import sys
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hindsight_app_source_adapter as adapter
import hindsight_app_approved_run as runner
import hindsight_app_event_ownership as ownership
from hindsight_incremental_ledger import IncrementalLedger
from hindsight_incremental_sources import _message, digest, message_hash, snapshot_hash


def finite_fixture(root, count=2, bank='fixture', shared_item=False, same_session=False):
    account = 'synthetic-native-account'
    account_hash = digest(account)
    epoch = 'app-contract-v1::' + account_hash
    sessions, events, scopes, app_snapshots, physical_messages = [], [], [], [], []
    ids = [str(uuid.UUID(int=100 + (0 if same_session else i))) for i in range(count)]
    keys = [f'app-contract::{account_hash}::{sid}' for sid in ids]
    physical_key, physical_epoch = 'default:hermes:synthetic-native', 'synthetic-physical-v1'
    for i, (sid, key) in enumerate(zip(ids, keys)):
        turn = str(uuid.UUID(int=200 + i))
        item = str(uuid.UUID(int=300 + (0 if shared_item else i)))
        body = f'Independent fixture observation number {i} remains attributable.'
        raw = {'role': 'user', 'timestamp': '2026-10-15T00:00:00Z',
               'content': 'App fixture provenance\n' + body,
               'source_turn_id': turn, 'source_item_id': item,
               'origin_message_id': f'codex-app:{sid}:{turn}:{item}'}
        session = {'source_session_id': sid, 'session_id': 'codex-app-' + sid,
                   'source': 'codex-app', 'platform': 'backfill-codex-app', 'messages': [raw]}
        sessions.append(session)
        app_message = _message(raw, codex=True)
        origin = f'native-fixture:{sid}:{turn}:{item}'
        physical_message = {'event_id': digest({'origin': origin}), 'origin': origin,
            'role': app_message['role'], 'timestamp': app_message['timestamp'],
            'content': 'Physical fixture provenance\n' + body}
        physical_messages.append(physical_message)
        snap = {'source_key': key, 'source_epoch': epoch, 'epoch_scope': epoch,
            'scope_source_keys': sorted(set(keys)), 'profile': 'default',
            'messages': [app_message], 'source_hash_before': 'a' * 64, 'source_hash_after': 'a' * 64}
        snap['snapshot_hash'] = snapshot_hash(snap)
        app_snapshots.append(snap)
        identity = {'account': account, 'session': sid, 'turn': turn, 'item': item}
        views = []
        for view, source_key, source_epoch, msg, prefix in [
                ('app', key, epoch, app_message, 'App fixture provenance\n'),
                ('physical', physical_key, physical_epoch, physical_message, 'Physical fixture provenance\n')]:
            views.append({'view': view, 'source_key': source_key, 'source_epoch': source_epoch,
                'event_id': msg['event_id'], 'message': msg, 'message_hash': message_hash(msg),
                'content_prefix': prefix})
        payload = {'role': app_message['role'], 'timestamp': app_message['timestamp'], 'content': body}
        events.append({'identity': identity, 'canonical_payload': payload,
            'canonical_payload_sha256': digest(payload), 'views': views,
            'identity_proof': {'kind': 'native-account-session-turn-item',
                'sha256': digest({'synthetic-independent-native-record': identity}),
                'identity': dict(identity), 'bindings': sorted([
                    {'source_key': v['source_key'], 'event_id': v['event_id'],
                     'origin': v['message']['origin'], 'message_hash': v['message_hash']}
                    for v in views], key=lambda v: v['source_key'])}})
    if same_session:
        sessions[0]['messages'] = [s['messages'][0] for s in sessions]
        sessions = sessions[:1]
        app_snapshots[0]['messages'] = [s['messages'][0] for s in app_snapshots]
        app_snapshots[0]['snapshot_hash'] = snapshot_hash(app_snapshots[0])
        app_snapshots = app_snapshots[:1]
    physical = {'source_key': physical_key, 'source_epoch': physical_epoch,
        'profile': 'default', 'messages': physical_messages,
        'source_hash_before': 'b' * 64, 'source_hash_after': 'b' * 64}
    physical['snapshot_hash'] = snapshot_hash(physical)
    for view, snap in [('app', s) for s in app_snapshots] + [('physical', physical)]:
        scope = {k: snap[k] for k in ('source_key', 'source_epoch', 'profile', 'snapshot_hash')}
        scope.update(view=view, event_ids=[m['event_id'] for m in snap['messages']])
        for key in ('epoch_scope', 'scope_source_keys'):
            if key in snap:
                scope[key] = snap[key]
        scopes.append(scope)
    contract = {'schema': ownership.SCHEMA, 'bank': bank, 'ledger_root': str(root.absolute()),
                'account': account, 'scopes': scopes, 'events': events}
    return sessions, contract, physical


def build(sessions, fixture_contract, root, **overrides):
    kwargs = {'contract': fixture_contract, 'approved_contract_sha256': digest(fixture_contract),
              'bank': fixture_contract['bank'], 'ledger_root': root,
              'approved_session_ids': {s['source_session_id'] for s in sessions}}
    kwargs.update(overrides)
    return adapter.build_app_snapshots(sessions, **kwargs)


@pytest.mark.parametrize('case', [
    'missing-contract', 'missing-pin', 'wrong-pin', 'wrong-bank', 'wrong-root',
    'partial', 'outside-session', 'malformed-session', 'unknown-native-id', 'missing-origin',
    'truncated', 'changed-public-body', 'filtered-tool', 'canonical-payload', 'native-proof',
    'unapproved-contract-replacement', 'cross-session-item', 'cross-turn-item',
])
def test_ambiguous_whole_frozen_scope_never_yields_capture_candidates(tmp_path, case):
    root = tmp_path / 'ledger'
    sessions, contract, _ = finite_fixture(root)
    options = {}
    if case == 'missing-contract':
        options['contract'] = None
    elif case == 'missing-pin':
        options['approved_contract_sha256'] = None
    elif case == 'wrong-pin':
        options['approved_contract_sha256'] = '0' * 64
    elif case == 'wrong-bank':
        options['bank'] = 'other'
    elif case == 'wrong-root':
        options['ledger_root'] = tmp_path / 'other'
    elif case == 'partial':
        _, contract, _ = finite_fixture(root, count=1)
    elif case == 'outside-session':
        options['approved_session_ids'] = {sessions[0]['source_session_id']}
    elif case == 'malformed-session':
        sessions[1] = None
        options['approved_session_ids'] = {str(uuid.UUID(int=100)), str(uuid.UUID(int=101))}
    elif case == 'unknown-native-id':
        sessions[0]['messages'][0]['source_item_id'] = 'not-a-native-uuid'
    elif case == 'missing-origin':
        sessions[0]['messages'][0].pop('origin_message_id')
    elif case == 'truncated':
        sessions[0]['messages'][0]['truncated'] = True
    elif case == 'changed-public-body':
        sessions[0]['messages'][0]['content'] += ' changed'
    elif case == 'filtered-tool':
        sessions[0]['messages'][0]['role'] = 'tool'
    elif case == 'canonical-payload':
        contract['events'][0]['canonical_payload']['content'] += ' changed'
    elif case == 'native-proof':
        contract['events'][0]['identity_proof']['bindings'] = []
    elif case == 'unapproved-contract-replacement':
        path = tmp_path / 'contract.json'
        original_pin = digest(contract)
        contract['account'] = 'substituted-account'
        path.write_text(json.dumps(contract))
        with pytest.raises(adapter.AppSourceAmbiguity, match='pin mismatch'):
            adapter.load_contract(path, approved_contract_sha256=original_pin)
        assert not root.exists()
        return
    elif case in {'cross-session-item', 'cross-turn-item'}:
        # Both individual dual-view proofs remain valid; current schema provides
        # no cross-session App alias evidence for one native item UUID.
        sessions, contract, _ = finite_fixture(root, shared_item=True,
                                              same_session=case == 'cross-turn-item')
    if case == 'malformed-session':
        with pytest.raises((adapter.AppSourceAmbiguity, ownership.OwnershipAmbiguousError)):
            adapter.build_app_snapshots(sessions, contract=contract,
                approved_contract_sha256=digest(contract), bank=contract['bank'], ledger_root=root,
                approved_session_ids=options['approved_session_ids'])
    else:
        with pytest.raises((adapter.AppSourceAmbiguity, ownership.OwnershipAmbiguousError)):
            build(sessions, contract, root, **options)
    assert not root.exists()


@pytest.mark.parametrize('first_view', ['app', 'physical'])
def test_finite_adapter_uses_same_real_owner_gate_and_plan_never_runs_runtime(tmp_path, monkeypatch, capsys, first_view):
    root = tmp_path / 'ledger'
    sessions, contract, physical = finite_fixture(root, bank='hermes')
    pin = digest(contract)  # Synthetic fixture approval only; never a production pin.
    snapshots = build(sessions, contract, root)
    ledger = IncrementalLedger(root, 'hermes', create=True)
    ledger.initialize_legacy([], {}, [], confirm='initialize-hindsight-incremental')
    ownership.register_contract(ledger, contract, approved_contract_sha256=pin)
    first, second = (snapshots, [physical]) if first_view == 'app' else ([physical], snapshots)
    batches = [b for snap in first for b in ledger.capture_snapshot(snap)]
    assert batches
    assert all(ledger.capture_snapshot(snap) == [] for snap in second + first)
    assert len(ledger.list_batches()) == len(batches)
    for snap in snapshots:
        plan = ownership.plan_snapshot(ledger, snap)
        assert all(e['phase'] == 'captured' and not e['completed'] for e in plan['events'])

    contract_path = tmp_path / 'synthetic-contract.json'
    contract_path.write_text(json.dumps(contract))
    monkeypatch.setattr(runner, '_load_approved_sessions', lambda: copy.deepcopy(sessions))
    monkeypatch.setattr(runner, 'APPROVED', frozenset(s['source_session_id'] for s in sessions))
    monkeypatch.setattr(runner, 'APP_LEDGER_ROOT', root)
    def forbidden(*args, **kwargs):
        pytest.fail('planning must not load config, construct runtime/ledger, capture, or call models')
    for name in ('load_config', 'ArrivalRuntime', 'IncrementalLedger', 'run_incremental', 'sync_sessions'):
        monkeypatch.setattr(runner, name, forbidden)
    assert runner.main(['--source-mode', 'app', '--plan-only', '--app-contract', str(contract_path),
                        '--approved-app-contract-sha256', pin]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['status'] == 'planning_only' and output['admission_disabled']
    assert output['model_calls'] == 0 and output['app_history_complete'] is None
    assert output['ledger_capture'] == output['runtime_preflight'] == 'not_attempted'
    assert output['contract_covers_frozen_event_set'] and len(output['sources']) == len(sessions)
    for args in [ ['--source-mode', 'app'], ['--source-mode', 'app', '--plan-only'],
                 ['--source-mode', 'app', '--app-contract', str(contract_path),
                  '--approved-app-contract-sha256', pin], ['--plan-only'],
                 ['--source-mode', 'app', '--plan-only', '--app-contract', str(contract_path),
                  '--approved-app-contract-sha256', pin, '--approved-transition-sha256', pin] ]:
        with pytest.raises(SystemExit):
            runner.main(args)
    assert len(ledger.list_batches()) == len(batches)
