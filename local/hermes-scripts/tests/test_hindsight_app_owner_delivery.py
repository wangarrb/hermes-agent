"""Local-only cross-entry admission and existing delivery safety contracts."""
import importlib
import multiprocessing
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sources = importlib.import_module('hindsight_incremental_sources')
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
    epoch, old = 'fixture-physical-feed', 'default:cutover-session:legacy'
    anchor = {'source_key': old, 'source_epoch': epoch, 'messages': [],
              'epoch_scope': 'default:physical-feed', 'scope_source_keys': [old],
              'source_hash_before': 'a' * 64, 'source_hash_after': 'a' * 64}
    anchor['snapshot_hash'] = sources.snapshot_hash(anchor)
    ledger.initialize_legacy([anchor], {}, [], confirm='initialize-hindsight-incremental')
    # Real API identity, deliberately fictional body; no conversation text.
    sid = '01a13ed4-a7da-71fa-ba33-e9879482c024'
    turn = '01a13ed4-b6ee-7033-9775-b565d258549f'
    item = 'msg_03886b2b9e4f39ea016ad098cfee788197912c5642bf94d92e'
    message = sources._message({'role': 'assistant', 'timestamp': '2026-10-15T01:00:00+00:00',
        'content': 'A measured local fixture preserves the original source and request identity.',
        'origin_message_id': f'codex-app:{sid}:{turn}:{item}'}, codex=True)
    key = 'default:hermes:codex-app-' + sid
    snapshots = {}
    for view in ('app', 'physical'):
        snapshot = {'source_key': key, 'source_epoch': epoch, 'messages': [message],
                    'source_hash_before': 'b' * 64, 'source_hash_after': 'b' * 64}
        if view == 'physical':
            snapshot.update(epoch_scope='default:physical-feed', scope_source_keys=[old, key])
        snapshot['snapshot_hash'] = sources.snapshot_hash(snapshot)
        snapshots[view] = snapshot
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
    assert ledger.get_batch(first['batch_id'])['phase'] == 'unknown'
    assert ledger.get_batch(first['batch_id'])['operation_id'] == operation
    client.completed = True
    retained = delivery.deliver_batch(ledger, ledger.get_batch(first['batch_id']),
        client=client, retry_policy=policy(), single_poll=True)
    assert retained['phase'] == 'retained'
    assert client.posts == [operation]
    assert client.gets and all(value == operation for value in client.gets)
    assert ledger.get_batch(first['batch_id'])['operation_id'] == operation
    assert ledger.get_batch(first['batch_id'])['phase'] != 'published'
    assert len(ledger.list_batches()) == 1


def _capture_worker(root, snapshot, start, queue):
    start.wait(10)
    ledger = ledger_module.IncrementalLedger(Path(root), 'fixture')
    try:
        queue.put(len(ledger.capture_snapshot(snapshot)))
    except ledger_module.LedgerLockedError:
        queue.put(0)


def test_same_native_source_concurrent_views_capture_one_payload(tmp_path):
    ledger, snapshots = finite_fixture(tmp_path)
    ctx = multiprocessing.get_context('fork')
    start, queue = ctx.Event(), ctx.Queue()
    workers = [ctx.Process(target=_capture_worker,
        args=(ledger.storage_root, snapshot, start, queue)) for snapshot in snapshots.values()]
    for worker in workers:
        worker.start()
    start.set()
    results = [queue.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    assert sum(results) == 1
    assert all(ledger.capture_snapshot(s) == [] for s in snapshots.values())
    assert len(ledger.list_batches()) == len(list(ledger.payload_dir.glob('*.json'))) == 1


def _crash_capture_worker(root, snapshot, point):
    ledger = ledger_module.IncrementalLedger(Path(root), 'fixture')
    original = ledger._put
    def crash(con, table, key, value):
        original(con, table, key, value)
        if table == point:
            os._exit(47)
    ledger._put = crash
    ledger.capture_snapshot(snapshot)


@pytest.mark.parametrize('point', ['batches', 'sources'])
def test_source_batch_transaction_crash_reuses_one_payload(tmp_path, point):
    ledger, snapshots = finite_fixture(tmp_path)
    ctx = multiprocessing.get_context('fork')
    worker = ctx.Process(target=_crash_capture_worker,
        args=(ledger.storage_root, snapshots['app'], point))
    worker.start()
    worker.join(15)
    assert worker.exitcode == 47
    assert ledger.list_batches() == []
    orphan = list(ledger.payload_dir.glob('*.json'))
    assert len(orphan) == 1
    recovered = ledger.capture_snapshot(snapshots['app'])[0]
    assert recovered['payload_path'] == str(orphan[0])
    assert ledger.capture_snapshot(snapshots['physical']) == []
    assert len(ledger.list_batches()) == 1
