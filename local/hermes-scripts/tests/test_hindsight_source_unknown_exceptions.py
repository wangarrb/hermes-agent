"""Finite approved unknown rows are exclusions, never captured/success receipts.

All source, policy, and ledger I/O is real and confined to temporary fixtures.
"""
import copy
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def modules():
    return tuple(importlib.import_module('hindsight_incremental_' + name)
                 for name in ('sources', 'ledger', 'delivery', 'pipeline'))


def fixture(tmp_path):
    src, ld, delivery, flow = modules()
    home = tmp_path / 'home'
    home.mkdir()
    planner = home / 'profiles' / 'planner'
    planner.mkdir(parents=True)
    db = planner / 'state.db'
    with sqlite3.connect(db) as con:
        con.executescript('''CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL,
            parent_session_id TEXT, end_reason TEXT, source TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            timestamp REAL, content TEXT, active INTEGER, compacted INTEGER,
            display_order INTEGER, display_identity BLOB);
            INSERT INTO sessions VALUES ('native-session',100,NULL,NULL,'cli');''')
        for n in range(1, 4):
            con.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?)',
                        (n, 'native-session', 'user', 100 + n, 'Measured braking result ' + str(n),
                         1, 0, n, b'identity-' + bytes([n])))
    baseline = src.capture_physical_baseline(home, bank='fixture-bank')
    ledger = ld.IncrementalLedger(tmp_path / 'ledger', 'fixture-bank', create=True)
    with ledger.locked():
        path, checksum = delivery._freeze(ledger.root, baseline, 'physical-baseline')
        seed_path, seed_hash = delivery._freeze(ledger.root, [], 'seed')
        reference = {'schema_version': baseline['schema_version'], 'bank': ledger.bank,
                     'hermes_home': str(home), 'artifact_path': path, 'artifact_hash': checksum}
        ledger.initialize_legacy(src.physical_baseline_snapshots(baseline),
            {'bank': ledger.bank, 'artifact_path': seed_path, 'artifact_hash': seed_hash,
             'provenance': {'kind': 'synthetic-seed'}}, [],
            confirm=ld.CONFIRM_INITIALIZE, physical_baseline=reference)
    with sqlite3.connect(db) as con:
        for n in range(1, 4):
            con.execute('INSERT INTO messages SELECT id+10,session_id,role,timestamp,content,0,1,'
                        'display_order+10,display_identity FROM messages WHERE id=?', (n,))
        con.row_factory = sqlite3.Row
        records = [src._physical_record(dict(row), sqlite_row=True)
                   for row in con.execute('SELECT * FROM messages WHERE id IN (11,12,13) ORDER BY id')]
    profile = next(p for p in baseline['profiles'] if p['profile'] == 'planner')
    policy = {'schema_version': 'hindsight-source-approved-unknown/1', 'bank': ledger.bank,
              'hermes_home': str(home), 'physical_baseline_sha256': checksum,
              'authorization': 'explicit-finite-source-row-exclusion',
              'status': 'unknown', 'completion_proven': False,
              'scopes': [{'profile': 'planner', 'feed_epoch': profile['feed_epoch'],
                          'source_epoch': profile['sqlite']['epoch'],
                          'session_id': 'native-session', 'rows': records}]}
    return src, ld, delivery, flow, home, db, baseline, ledger, policy


def scan(src, home, baseline, policy):
    return src.scan_source_snapshots(home, physical_baseline=baseline, source_unknown_policy=policy)


def flatten(snapshots):
    return [m for s in snapshots for m in s['messages']]


def register(ledger, policy):
    return ledger.register_source_unknown_policy(policy, confirm='approve-hindsight-source-unknown')


def test_exact_three_unknown_bindings_are_not_input_or_completion(tmp_path):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    original = ledger.initialization_manifest()
    before_db = db.read_bytes()
    with pytest.raises(src.SourceChangedError, match='ambiguous'):
        src.scan_source_snapshots(home, physical_baseline=baseline)
    snapshots = scan(src, home, baseline, policy)
    assert not flatten(snapshots)
    assert [b for s in snapshots for b in ledger.capture_snapshot(s)] == []
    assert ledger.list_batches() == []
    assert ledger.initialization_manifest() == original
    assert db.read_bytes() == before_db
    assert policy['status'] == 'unknown' and policy['completion_proven'] is False
    assert snapshots == scan(src, home, baseline, policy)
    assert next(s for s in snapshots if s['profile'] == 'planner')['source_hash_before'] != (
        next(s for s in src.physical_baseline_snapshots(baseline) if s['profile'] == 'planner')['source_hash_before'])


def test_post_baseline_session_rows_admissible_only_by_exact_exclusion(tmp_path):
    """A session created after the descriptor is future; its exact ambiguous
    rows may be explicitly excluded, and the kept visible twin is captured once."""
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO sessions VALUES ('new-session',200,NULL,NULL,'cli')")
        con.execute("INSERT INTO messages VALUES (21,'new-session','user',201.0,'New measured result',0,0,21,NULL)")
        con.execute('INSERT INTO messages SELECT 31,session_id,role,timestamp,content,1,0,31,display_identity'
                    ' FROM messages WHERE id=21')
        con.row_factory = sqlite3.Row
        record = src._physical_record(dict(con.execute('SELECT * FROM messages WHERE id=21').fetchone()), sqlite_row=True)
    with pytest.raises(src.SourceChangedError, match='ambiguous'):
        src.scan_source_snapshots(home, physical_baseline=baseline)
    profile = next(p for p in baseline['profiles'] if p['profile'] == 'planner')
    scope = {'profile': 'planner', 'feed_epoch': profile['feed_epoch'],
             'source_epoch': profile['sqlite']['epoch'], 'session_id': 'new-session', 'rows': [record]}
    combined = copy.deepcopy(policy)
    combined['scopes'].append(scope)
    register(ledger, combined)
    snapshots = scan(src, home, baseline, combined)
    messages = flatten(snapshots)
    assert len(messages) == 1 and messages[0]['content'] == 'New measured result'
    assert len([b for s in snapshots for b in ledger.capture_snapshot(s)]) == 1
    assert [b for s in snapshots for b in ledger.capture_snapshot(s)] == []
    # Live drift of the excluded row still blocks.
    with sqlite3.connect(db) as con:
        con.execute("UPDATE messages SET content='Changed' WHERE id=21")
    with pytest.raises(src.SourceChangedError, match='binding drift'):
        scan(src, home, baseline, combined)


def test_registration_is_explicit_private_immutable_and_readback_bound(tmp_path):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    original = ledger.initialization_manifest()
    assert ledger.source_unknown_policy_reference() is None
    with pytest.raises(ValueError):
        ledger.register_source_unknown_policy(policy, confirm='not-approved')
    reference = register(ledger, policy)
    assert reference == ledger.source_unknown_policy_reference()
    assert delivery._read(ledger.root, reference['artifact_path'], reference['artifact_hash']) == policy
    assert reference['physical_baseline_sha256'] == ledger.physical_baseline_reference()['artifact_hash']
    assert Path(reference['artifact_path']).stat().st_mode & 0o777 == 0o400
    assert Path(reference['artifact_path']).parent.stat().st_mode & 0o777 == 0o700
    assert Path(reference['artifact_path']).stat().st_nlink == 1
    assert register(ledger, policy) == reference
    assert ledger.list_batches() == []
    assert ledger.initialization_manifest() == original
    changed = copy.deepcopy(policy)
    changed['scopes'][0]['rows'].pop()
    with pytest.raises(ld.LedgerIntegrityError):
        register(ledger, changed)
    assert ledger.source_unknown_policy_reference() == reference


@pytest.mark.parametrize('kind', ['missing-field', 'extra-field', 'schema', 'bank', 'home', 'baseline',
    'authorization', 'completion', 'completion-zero', 'status', 'empty', 'scopes-type', 'profile',
    'feed-epoch', 'source-epoch', 'sid', 'rows-type', 'empty-rows', 'row-shape', 'row-hash',
    'copy-hash', 'export-hash', 'origin-hash', 'boolean-id', 'string-id', 'wrong-row-sid',
    'duplicate-row', 'duplicate-scope', 'legacy-row'])
def test_policy_schema_scope_and_full_bindings_fail_closed(tmp_path, kind):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    bad = copy.deepcopy(policy)
    scope = bad['scopes'][0]
    row = scope['rows'][0]
    if kind == 'missing-field':
        del bad['completion_proven']
    elif kind == 'extra-field':
        bad['max_id'] = 13
    elif kind in {'schema', 'bank', 'home', 'baseline', 'authorization', 'status'}:
        key = {'schema': 'schema_version', 'baseline': 'physical_baseline_sha256',
               'home': 'hermes_home'}.get(kind, kind)
        bad[key] = 'wrong'
    elif kind in {'completion', 'completion-zero'}:
        bad['completion_proven'] = True if kind == 'completion' else 0
    elif kind == 'empty':
        bad['scopes'] = []
    elif kind == 'scopes-type':
        bad['scopes'] = {}
    elif kind in {'profile', 'feed-epoch', 'source-epoch', 'sid'}:
        scope[{'feed-epoch': 'feed_epoch', 'source-epoch': 'source_epoch', 'sid': 'session_id'}.get(kind, kind)] = 'wrong'
    elif kind == 'rows-type':
        scope['rows'] = {}
    elif kind == 'empty-rows':
        scope['rows'] = []
    elif kind == 'row-shape':
        del row['origin_hash']
    elif kind in {'row-hash', 'copy-hash', 'export-hash', 'origin-hash'}:
        row[kind.replace('-', '_')] = 'invalid'
    elif kind in {'boolean-id', 'string-id'}:
        row['id'] = True if kind == 'boolean-id' else '11'
    elif kind == 'wrong-row-sid':
        row['session_id'] = 'other'
    elif kind == 'duplicate-row':
        scope['rows'].append(copy.deepcopy(row))
    elif kind == 'duplicate-scope':
        bad['scopes'].append(copy.deepcopy(scope))
    else:
        scope['rows'][0] = copy.deepcopy(next(p for p in baseline['profiles']
            if p['profile'] == 'planner')['sqlite']['rows'][0])
    with pytest.raises(ValueError):
        scan(src, home, baseline, bad)
    with pytest.raises(ValueError):
        register(ledger, bad)
    assert ledger.source_unknown_policy_reference() is None


@pytest.mark.parametrize('change', ['body', 'timestamp', 'association', 'delete', 'truncate', 'epoch',
    'schema', 'baseline-body', 'baseline-delete', 'session', 'replace', 'copy-hash', 'export-hash', 'origin-hash'])
def test_live_sql_or_policy_drift_never_consumes_unknown_rows(tmp_path, change):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    register(ledger, policy)
    if change in {'copy-hash', 'export-hash', 'origin-hash'}:
        policy['scopes'][0]['rows'][0][change.replace('-', '_')] = 'a' * 64
    elif change == 'replace':
        db.rename(db.with_suffix('.old'))
        db.write_bytes(db.with_suffix('.old').read_bytes())
    else:
        sql = {'body': "UPDATE messages SET content='Changed source' WHERE id=11",
               'timestamp': 'UPDATE messages SET timestamp=999 WHERE id=11',
               'association': "UPDATE messages SET session_id='other' WHERE id=11",
               'delete': 'DELETE FROM messages WHERE id=13', 'truncate': 'DELETE FROM messages',
               'epoch': 'PRAGMA application_id=99', 'schema': 'ALTER TABLE messages ADD COLUMN other TEXT',
               'baseline-body': "UPDATE messages SET content='Changed baseline' WHERE id=1",
               'baseline-delete': 'DELETE FROM messages WHERE id=1',
               'session': "UPDATE sessions SET started_at=999 WHERE id='native-session'"}[change]
        with sqlite3.connect(db) as con:
            con.execute(sql)
    with pytest.raises(ValueError):
        scan(src, home, baseline, policy)
    assert ledger.list_batches() == []


@pytest.mark.parametrize('where', ['same-sid', 'other-sid', 'json'])
def test_unknown_policy_does_not_widen_to_other_identical_rows_or_json(tmp_path, where):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    register(ledger, policy)
    if where == 'json':
        directory = db.parent / 'sessions'
        directory.mkdir()
        (directory / 'session_native-session.json').write_text(json.dumps({
            'session_id': 'native-session', 'session_start': 100,
            'messages': [{'role': 'user', 'timestamp': 101, 'content': 'Measured braking result 1'}]}))
    else:
        with sqlite3.connect(db) as con:
            sid = 'native-session' if where == 'same-sid' else 'unrelated-session'
            if where == 'other-sid':
                con.execute('INSERT INTO sessions VALUES (?,200,NULL,NULL,\'cli\')', (sid,))
            con.execute('INSERT INTO messages SELECT 14,?,role,timestamp,content,1,0,14,display_identity '
                        'FROM messages WHERE id=1', (sid,))
    with pytest.raises(src.SourceChangedError):
        scan(src, home, baseline, policy)


def test_new_old_date_input_arrives_once_without_unknown_payloads(tmp_path):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    register(ledger, policy)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (0,'native-session','user',1,'New arrival about an old event',1,0,0,NULL)")
    snapshots = scan(src, home, baseline, policy)
    assert [m['content'] for m in flatten(snapshots)] == ['New arrival about an old event']
    batches = [b for s in snapshots for b in ledger.capture_snapshot(s)]
    assert len(batches) == 1 and len(batches[0]['source_event_ids']) == 1
    assert [b for s in scan(src, home, baseline, policy) for b in ledger.capture_snapshot(s)] == []
    frozen = delivery._read(ledger.root, batches[0]['payload_path'], batches[0]['payload_hash'])
    assert all('Measured braking result' not in m['content'] for m in frozen['messages'])


@pytest.mark.parametrize('damage', ['null', 'missing-field', 'type', 'schema', 'hash', 'path', 'permissions',
    'hardlink', 'symlink', 'artifact-missing', 'artifact-bytes', 'baseline-bytes', 'meta-hash'])
def test_reference_corruption_blocks_normal_pipeline_without_paid_preflight(tmp_path, monkeypatch, damage):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    reference = register(ledger, policy)
    bad = copy.deepcopy(reference)
    target = Path(reference['artifact_path'])
    if damage == 'null':
        bad = None
    elif damage == 'missing-field':
        del bad['artifact_hash']
    elif damage == 'type':
        bad['artifact_hash'] = 123
    elif damage == 'schema':
        bad['schema_version'] = 'wrong'
    elif damage == 'hash':
        bad['artifact_hash'] = 'a' * 64
    elif damage == 'path':
        bad['artifact_path'] = str(target) + '\n'
    elif damage == 'permissions':
        target.chmod(0o600)
    elif damage == 'hardlink':
        os.link(target, tmp_path / 'alias')
    elif damage == 'symlink':
        target.rename(tmp_path / 'original')
        target.symlink_to(tmp_path / 'original')
    elif damage == 'artifact-missing':
        target.unlink()
    elif damage in {'artifact-bytes', 'baseline-bytes'}:
        if damage == 'baseline-bytes':
            target = Path(ledger.physical_baseline_reference()['artifact_path'])
        target.chmod(0o600)
        target.write_bytes(target.read_bytes() + b' ')
        target.chmod(0o400)
    if damage in {'null', 'missing-field', 'type', 'schema', 'hash', 'path', 'meta-hash'}:
        with ledger.locked(), ledger._connection(write=True) as con:
            ledger._put(con, 'meta', 'source-unknown-policy', bad)
            if damage == 'meta-hash':
                con.execute("UPDATE meta SET state_hash='bad' WHERE key='source-unknown-policy'")
    class Forbidden:
        def preflight(self, **kw):
            raise AssertionError('unexpected preflight')
        def open_window(self, *args):
            raise AssertionError('unexpected producer')
    def forbidden(*args, **kw):
        raise AssertionError('unexpected LLM/reporter')
    monkeypatch.setattr(flow, 'reporting_run', forbidden)
    report = flow.run_incremental({'hermes_home': str(home)}, ledger=ledger,
                                 client=Forbidden(), llm=forbidden, execute=True)
    assert report['status'] == 'blocked'
    assert report['calls'] == report['llm_calls'] == 0
    assert ledger.list_batches() == []


def test_actual_normal_pipeline_reads_policy_and_noop_needs_no_native_or_threads(tmp_path, monkeypatch):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    reference = register(ledger, policy)
    def forbidden(*args, **kw):
        raise AssertionError('no-op must not start paid/native/progress work')
    class Forbidden:
        preflight = open_window = forbidden
    monkeypatch.setattr(flow, 'reporting_run', forbidden)
    report = flow.run_incremental({'hermes_home': str(home)}, ledger=ledger,
                                 client=Forbidden(), llm=forbidden, execute=True)
    assert report['status'] == 'no_op'
    assert report['calls'] == report['llm_calls'] == 0
    assert report['source_unknown_policy'] == {'artifact_hash': reference['artifact_hash'],
        'status': 'unknown', 'completion_proven': False, 'excluded_sql_rows': 3}
    assert ledger.list_batches() == []


def test_policy_cannot_be_used_without_original_physical_baseline(tmp_path):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    with pytest.raises(ValueError):
        src.scan_source_snapshots(home, source_unknown_policy=policy)
    logical = ld.IncrementalLedger(tmp_path / 'logical', 'fixture-bank', create=True)
    logical.initialize_legacy([], {}, [], confirm=ld.CONFIRM_INITIALIZE)
    with pytest.raises(ValueError):
        register(logical, policy)


def test_registration_checks_readonly_current_row_coverage_before_persisting(tmp_path):
    src, ld, delivery, flow, home, db, baseline, ledger, policy = fixture(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute('DELETE FROM messages WHERE id=13')
    with pytest.raises(ValueError):
        register(ledger, policy)
    assert ledger.source_unknown_policy_reference() is None
    assert list((ledger.root / 'delivery-artifacts').glob('*.source-unknown-policy.json')) == []
