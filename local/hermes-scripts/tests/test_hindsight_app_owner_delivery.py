"""Local-only cross-entry admission and existing delivery safety contracts."""
import importlib
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sources = importlib.import_module('hindsight_incremental_sources')
ownership = importlib.import_module('hindsight_app_event_ownership')
ledger_module = importlib.import_module('hindsight_incremental_ledger')
delivery = importlib.import_module('hindsight_incremental_delivery')
retry = importlib.import_module('hindsight_offline_retry')


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'home'))


def physical_fixture(tmp_path):
    home = tmp_path / 'physical'
    (home / 'sessions').mkdir(parents=True)
    database = home / 'state.db'
    with sqlite3.connect(database) as connection:
        connection.executescript('''
            CREATE TABLE sessions(id TEXT PRIMARY KEY, started_at REAL,
                parent_session_id TEXT, end_reason TEXT, source TEXT);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT,
                role TEXT, content TEXT, timestamp REAL, origin_message_id TEXT);
        ''')
        connection.execute('INSERT INTO sessions VALUES (?,?,?,?,?)',
                           ('20261008_141855_0038ce', 1791420000.0, None, None, 'cli'))
    return home, database, sources.capture_physical_baseline(home, bank='fixture')


def test_physical_identical_future_rows_still_rejected(tmp_path):
    home, database, baseline = physical_fixture(tmp_path)
    with sqlite3.connect(database) as connection:
        for identity in (1, 2):
            connection.execute('INSERT INTO messages VALUES (?,?,?,?,?,?)',
                (identity, '20261008_141855_0038ce', 'assistant',
                 'The measured result requires independent immutable event identity.',
                 1792022400.0, None))
    with pytest.raises(sources.SourceChangedError, match='ambiguous identical future SQLite'):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_schema_drift_still_rejected(tmp_path):
    home, database, baseline = physical_fixture(tmp_path)
    with sqlite3.connect(database) as connection:
        previous = connection.execute('PRAGMA schema_version').fetchone()[0]
        connection.execute(f'PRAGMA schema_version={previous + 1}')
    with pytest.raises(sources.SourceChangedError, match='epoch/schema/session loss'):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def finite_fixture(tmp_path):
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'fixture', create=True)
    ledger.initialize_legacy([], {}, [], confirm='initialize-hindsight-incremental')
    canonical = {'role': 'assistant', 'timestamp': '2026-10-15T01:00:00+00:00',
                 'content': 'A measured local fixture preserves the original source and request identity.'}
    identity = {'account': 'synthetic-account', 'session': 'synthetic-session',
                'turn': 'synthetic-turn', 'item': 'synthetic-item'}
    snapshots, scopes, views = {}, [], []
    for view, prefix in [('app', '[source: fixture-App]\n'), ('physical', '')]:
        key = 'app-contract::fixture' if view == 'app' else 'default:hermes:fixture'
        message = {**canonical, 'origin': f'fixture-native-{view}',
                   'event_id': sources.digest({'view': view}), 'content': prefix + canonical['content']}
        snapshot = {'source_key': key, 'source_epoch': 'fixture-epoch-' + view,
                    'messages': [message], 'source_hash_before': 'a' * 64,
                    'source_hash_after': 'a' * 64}
        snapshot['snapshot_hash'] = sources.snapshot_hash(snapshot)
        snapshots[view] = snapshot
        scopes.append({k: snapshot[k] for k in ('source_key', 'source_epoch', 'snapshot_hash')}
                      | {'view': view, 'event_ids': [message['event_id']]})
        views.append({'view': view, 'source_key': key, 'source_epoch': snapshot['source_epoch'],
                      'event_id': message['event_id'], 'message': message,
                      'message_hash': sources.message_hash(message), 'content_prefix': prefix})
    # This is a synthetic authority for an isolated fixture, not an approval
    # derived from real App data, a matching text body or an observed UUID gap.
    contract = {'schema': ownership.SCHEMA, 'bank': ledger.bank,
                'ledger_root': str(ledger.storage_root), 'account': identity['account'],
                'scopes': scopes, 'events': [{'identity': identity,
                    'identity_proof': {'kind': 'native-account-session-turn-item',
                                       'sha256': sources.digest({'synthetic-native-proof': identity}),
                                       'identity': identity,
                                       'bindings': sorted([
                                           {'source_key': v['source_key'], 'event_id': v['event_id'],
                                            'origin': v['message']['origin'],
                                            'message_hash': v['message_hash']}
                                           for v in views], key=lambda v: v['source_key'])},
                    'canonical_payload': canonical, 'canonical_payload_sha256': sources.digest(canonical),
                    'views': views}]}
    ownership.register_contract(ledger, contract, approved_contract_sha256=sources.digest(contract))
    return ledger, snapshots


@pytest.mark.parametrize('first_view', ['app', 'physical'])
def test_unknown_post_alias_reconciles_same_operation_without_second_post(tmp_path, first_view):
    ledger, snapshots = finite_fixture(tmp_path)
    first = ledger.capture_snapshot(snapshots[first_view])[0]
    operation = first['operation_id']

    class Client:
        bank = ledger.bank
        completed = False
        posts = []
        gets = []

        def retain_items(self, items, *, operation_id, **kwargs):
            self.posts.append(operation_id)
            raise TimeoutError('synthetic lost acknowledgement')

        def get_operation(self, operation_id, **kwargs):
            self.gets.append(operation_id)
            return {'operation_id': operation_id, 'bank_id': self.bank,
                    'status': 'completed' if self.completed else 'pending'}

    client = Client()
    clock = [1000.0]

    def policy():
        return retry.RetryPolicy(clock=lambda: clock[0], wall_clock=lambda: clock[0],
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds), jitter=lambda: 0)

    pending = delivery.deliver_batch(ledger, first, client=client,
                                    retry_policy=policy(), single_poll=True)
    assert pending['phase'] == 'unknown'
    other_view = 'physical' if first_view == 'app' else 'app'
    assert ledger.capture_snapshot(snapshots[other_view]) == []
    planned = ownership.plan_snapshot(ledger, snapshots[other_view])['events'][0]
    assert planned['phase'] == 'unknown' and not planned['completed']
    assert planned['operation_id'] == operation
    client.completed = True
    retained = delivery.deliver_batch(ledger, ledger.get_batch(first['batch_id']),
        client=client, retry_policy=policy(), single_poll=True)
    assert retained['phase'] == 'retained'
    assert client.posts == [operation]
    assert client.gets and all(value == operation for value in client.gets)
    assert ledger.get_batch(first['batch_id'])['operation_id'] == operation
    assert not ownership.plan_snapshot(ledger, snapshots[other_view])['events'][0]['completed']
