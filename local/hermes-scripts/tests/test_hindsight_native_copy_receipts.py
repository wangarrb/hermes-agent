"""Real synthetic Core -> readonly Sources -> ledger/normal no-op entrypoint.

Fixture-only synchronous=OFF proves transaction/read consistency, not durability
across a power failure. No production source, policy, bank or native API is used.
"""
import copy
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hermes_state import SessionDB

KEY = '_message_copy_receipts_v1'
SID = 'synthetic-session'


def modules():
    return tuple(importlib.import_module('hindsight_incremental_' + n)
                 for n in ('sources', 'ledger', 'delivery', 'pipeline'))


@pytest.fixture(autouse=True)
def synthetic_pragmas(tmp_path, monkeypatch):
    import hermes_state
    original = hermes_state.apply_database_pragmas

    def apply(con, *args, **kwargs):
        assert all(not row[2] or Path(row[2]).resolve().is_relative_to(tmp_path.resolve())
                   for row in con.execute('PRAGMA database_list'))
        original(con, *args, **kwargs)
        con.execute('PRAGMA synchronous=OFF')

    monkeypatch.setattr(hermes_state, 'apply_database_pragmas', apply)


@pytest.fixture
def native(tmp_path):
    src, ld, delivery, flow = modules()
    home = tmp_path / 'home'
    home.mkdir()
    db = SessionDB(home / 'state.db')
    db.create_session(SID, 'cli', model_config={'provider': 'synthetic', 'private': 'not-projected'})
    db.append_message(SID, 'user', content='Synthetic measured legacy braking result', timestamp=100.0)
    baseline = src.capture_physical_baseline(home, bank='synthetic-bank')
    ledger = ld.IncrementalLedger(tmp_path / 'ledger', 'synthetic-bank', create=True)
    with ledger.locked():
        path, checksum = delivery._freeze(ledger.root, baseline, 'physical-baseline')
        seed_path, seed_hash = delivery._freeze(ledger.root, [], 'seed')
        ref = {'schema_version': baseline['schema_version'], 'bank': ledger.bank,
               'hermes_home': str(home), 'artifact_path': path, 'artifact_hash': checksum}
        ledger.initialize_legacy(src.physical_baseline_snapshots(baseline),
            {'bank': ledger.bank, 'artifact_path': seed_path, 'artifact_hash': seed_hash,
             'provenance': {'kind': 'synthetic-seed'}}, [],
            confirm=ld.CONFIRM_INITIALIZE, physical_baseline=ref)
    yield src, ld, delivery, flow, home, db, baseline, ledger
    db.close()


def receipt(db):
    return json.loads(db.get_session(SID)['model_config'])[KEY]


def save_receipt(db, value):
    with sqlite3.connect(db.db_path) as con:
        con.execute('UPDATE sessions SET model_config=? WHERE id=?',
                    (json.dumps({KEY: value, 'private': 'not-projected'}), SID))


def carried(db, row_id=None):
    with db._read_ctx() as con:
        rows = [dict(r) for r in con.execute('SELECT * FROM messages WHERE session_id=? ORDER BY id', (SID,))]
    row_id = row_id or rows[-1]['id']
    row = next(r for r in rows if r['id'] == row_id)
    return {**row, 'content': db._decode_content(row['content']), '_row_id': row_id}


def compact(db, msg, tail_count=0):
    db.archive_and_compact(SID, [{"role": "assistant", "content": "synthetic summary", '_compressed_summary': True}, msg],
                           tail_count=tail_count)
    return msg['_row_id']


def scan(native, policy=None):
    src, _, _, _, home, _, baseline, _ = native
    options = {'physical_baseline': baseline}
    if policy is not None:
        options['source_unknown_policy'] = policy
    return src.scan_source_snapshots(home, **options)


def flatten(snapshots):
    return [m for s in snapshots for m in s['messages']]


def capture(native, snapshots):
    ledger = native[-1]
    return [b for s in snapshots for b in ledger.capture_snapshot(s)]


def assert_noop(native, monkeypatch, expected_unknown=None):
    _, _, _, flow, home, _, _, ledger = native
    def forbidden(*args, **kwargs):
        raise AssertionError('no-op invoked native/LLM/reporter work')
    class Forbidden:
        preflight = open_window = forbidden
    monkeypatch.setattr(flow, 'reporting_run', forbidden)
    report = flow.run_incremental({'hermes_home': str(home)}, ledger=ledger,
                                 client=Forbidden(), llm=forbidden, execute=True)
    assert report['status'] == 'no_op', report
    assert report['calls'] == report['llm_calls'] == 0
    if expected_unknown:
        assert report['source_unknown_policy'] == expected_unknown
    assert ledger.list_batches() == []
    return report


@pytest.mark.parametrize('tail_count', [0, 1])
def test_real_core_legacy_copy_chain_is_zero_input_normal_pipeline_noop(native, monkeypatch, tail_count):
    src, _, _, _, _, db, baseline, ledger = native
    initial = ledger.initialization_manifest()
    for _ in range(2):
        compact(db, carried(db), tail_count)
        assert receipt(db)['operations']
        assert flatten(scan(native)) == []
        assert capture(native, scan(native)) == []
        assert_noop(native, monkeypatch)
    assert ledger.initialization_manifest() == initial


def test_real_core_sql_clone_of_legacy_tail_is_not_new_input(native, monkeypatch):
    _, _, _, _, _, db, _, _ = native
    db.archive_and_compact(SID, [{'role': 'assistant', 'content': 'synthetic summary', '_compressed_summary': True}], watermark=0)
    assert receipt(db)['operations'][0]['copies'][0]['source_row_id'] == 1
    assert flatten(scan(native)) == []
    assert_noop(native, monkeypatch)


def test_independent_equal_append_with_display_group_is_still_ambiguous(native):
    src, _, _, _, _, db, _, ledger = native
    compact(db, carried(db))
    prior = copy.deepcopy(receipt(db))
    db.append_message(SID, 'user', content='Synthetic measured legacy braking result', timestamp=100.0)
    assert receipt(db) == prior
    with pytest.raises(src.SourceChangedError, match='ambiguous'):
        scan(native)
    assert ledger.list_batches() == []


@pytest.mark.parametrize('tail_count', [0, 1])
def test_future_late_event_survives_two_compactions_exactly_once(native, tail_count):
    src, _, _, _, _, db, _, ledger = native
    row_id = db.append_message(SID, 'user', content='New arrival about a synthetic old event', timestamp=1.0)
    snapshots = scan(native)
    event = flatten(snapshots)[0]
    batches = capture(native, snapshots)
    assert len(batches) == 1
    original_batch = ledger.get_batch(batches[0]['batch_id'])
    for _ in range(2):
        row_id = compact(db, carried(db, row_id), tail_count)
        assert flatten(scan(native)) == [event]
        assert capture(native, scan(native)) == []
        assert ledger.get_batch(batches[0]['batch_id']) == original_batch
    # Even a hidden original is exact immutable evidence, not forgotten history.
    assert event['timestamp'].startswith('1970-01-01')


def test_new_late_event_after_native_legacy_copy_arrives_once(native):
    _, _, _, _, _, db, _, _ = native
    compact(db, carried(db))
    db.append_message(SID, 'user', content='Different synthetic arrival about an old date', timestamp=1.0)
    assert len(flatten(scan(native))) == 1
    assert len(capture(native, scan(native))) == 1
    assert capture(native, scan(native)) == []


def unknown_policy(native, row_id):
    src, _, _, _, home, db, baseline, ledger = native
    profile = baseline['profiles'][0]
    with sqlite3.connect(db.db_path) as con:
        con.row_factory = sqlite3.Row
        row = dict(con.execute('SELECT * FROM messages WHERE id=?', (row_id,)).fetchone())
    return {'schema_version': 'hindsight-source-approved-unknown/1', 'bank': ledger.bank,
            'hermes_home': str(home), 'physical_baseline_sha256': ledger.physical_baseline_reference()['artifact_hash'],
            'authorization': 'explicit-finite-source-row-exclusion', 'status': 'unknown', 'completion_proven': False,
            'scopes': [{'profile': profile['profile'], 'feed_epoch': profile['feed_epoch'],
                        'source_epoch': profile['sqlite']['epoch'], 'session_id': SID,
                        'rows': [src._physical_record(row, sqlite_row=True)]}]}


def test_approved_unknown_root_copy_never_becomes_input_or_completion(native, monkeypatch):
    src, _, _, _, _, db, _, ledger = native
    row_id = db.append_message(SID, 'user', content='Synthetic measured legacy braking result', timestamp=100.0)
    policy = unknown_policy(native, row_id)
    with pytest.raises(src.SourceChangedError):
        scan(native)
    reference = ledger.register_source_unknown_policy(policy, confirm='approve-hindsight-source-unknown')
    for _ in range(2):
        row_id = compact(db, carried(db, row_id))
        assert flatten(scan(native, policy)) == []
        assert capture(native, scan(native, policy)) == []
        assert_noop(native, monkeypatch, {'artifact_hash': reference['artifact_hash'],
            'status': 'unknown', 'completion_proven': False, 'excluded_sql_rows': 1})
    assert policy['completion_proven'] is False
    assert ledger.source_unknown_policy_reference() == reference


@pytest.mark.parametrize('damage', ['schema', 'sid', 'epoch', 'extra', 'operations-type', 'op-id', 'nil-id',
    'bool-id', 'source-hash', 'target-hash', 'copy-hash', 'uppercase', 'receipt-hash', 'kind',
    'duplicate-op', 'duplicate-target', 'cycle', 'wrong-owner', 'source-body', 'target-body',
    'source-blob', 'target-blob', 'missing-source', 'missing-target', 'deleted', 'malformed'])
def test_native_receipt_or_current_row_tampering_fails_closed(native, damage):
    src, _, _, _, _, db, _, ledger = native
    compact(db, carried(db))
    value = copy.deepcopy(receipt(db))
    op = value['operations'][0]
    edge = op['copies'][0]
    if damage in {'schema', 'sid', 'epoch'}:
        value[{'schema': 'schema_version', 'sid': 'session_id', 'epoch': 'source_epoch'}[damage]] = 'wrong'
    elif damage == 'extra':
        edge['body'] = 'not-allowed'
    elif damage == 'operations-type':
        value['operations'] = {}
    elif damage in {'op-id', 'nil-id'}:
        op['operation_id'] = str(uuid.UUID(int=0)) if damage == 'nil-id' else str(uuid.uuid4()).upper()
    elif damage == 'bool-id':
        edge['source_row_id'] = True
    elif damage in {'source-hash', 'target-hash', 'copy-hash'}:
        edge[{'source-hash': 'source_row_hash', 'target-hash': 'target_row_hash', 'copy-hash': 'copy_hash'}[damage]] = 'a' * 64
    elif damage == 'uppercase':
        edge['source_row_hash'] = edge['source_row_hash'].upper()
    elif damage == 'receipt-hash':
        op['receipt_sha256'] = 'b' * 64
    elif damage == 'kind':
        op['kind'] = 'independent_append'
    elif damage == 'duplicate-op':
        value['operations'].append(copy.deepcopy(op))
    elif damage == 'duplicate-target':
        op['copies'].append(copy.deepcopy(edge))
    elif damage == 'cycle':
        edge['source_row_id'], edge['target_row_id'] = edge['target_row_id'], edge['source_row_id']
    elif damage == 'deleted':
        with sqlite3.connect(db.db_path) as con:
            con.execute('UPDATE sessions SET model_config=? WHERE id=?', ('{}', SID))
    elif damage == 'malformed':
        with sqlite3.connect(db.db_path) as con:
            con.execute('UPDATE sessions SET model_config=? WHERE id=?', ('{"' + KEY + '":broken}', SID))
    else:
        target = edge['source_row_id'] if damage.startswith('source-') or damage == 'missing-source' else edge['target_row_id']
        with sqlite3.connect(db.db_path) as con:
            if damage.startswith('missing-'):
                con.execute('DELETE FROM messages WHERE id=?', (target,))
            elif damage == 'wrong-owner':
                db.create_session('other', 'cli')
                con.execute('UPDATE messages SET session_id=? WHERE id=?', ('other', target))
            elif damage.endswith('-blob'):
                con.execute('UPDATE messages SET display_identity=? WHERE id=?', (b'CASE-ALIAS', target))
            else:
                con.execute('UPDATE messages SET content=? WHERE id=?', ('Changed synthetic payload', target))
    if damage not in {'deleted', 'malformed'}:
        if damage != 'receipt-hash':
            for operation in value['operations']:
                operation['receipt_sha256'] = src.digest({k: v for k, v in operation.items() if k != 'receipt_sha256'})
        save_receipt(db, value)
    with pytest.raises(src.SourceChangedError):
        scan(native)
    assert ledger.list_batches() == []


@pytest.mark.parametrize('damage', ['container-deleted', 'first-operation-deleted', 'empty-operations'])
@pytest.mark.parametrize('tail_count', [0, 1])
def test_hidden_future_original_cannot_downgrade_when_receipt_history_is_removed(native, damage, tail_count):
    src, _, _, _, _, db, _, ledger = native
    row_id = db.append_message(SID, 'user', content='Synthetic future causal result', timestamp=1.0)
    original = flatten(scan(native))
    capture(native, scan(native))
    for _ in range(2):
        row_id = compact(db, carried(db, row_id), tail_count)
    value = receipt(db)
    if damage == 'container-deleted':
        with sqlite3.connect(db.db_path) as con:
            con.execute('UPDATE sessions SET model_config=? WHERE id=?', ('{}', SID))
    else:
        value['operations'] = value['operations'][1:] if damage == 'first-operation-deleted' else []
        save_receipt(db, value)
    with pytest.raises(src.SourceChangedError, match='ambiguous'):
        scan(native)
    assert len(ledger.list_batches()) == 1
    assert original[0]['content'] == 'Synthetic future causal result'


@pytest.mark.parametrize('tail_count', [0, 1])
def test_real_core_receipts_work_on_unanchored_normal_reader(native, tail_count):
    src, _, _, _, home, db, _, _ = native
    first = src.scan_source_snapshots(home)
    for _ in range(2):
        compact(db, carried(db), tail_count)
        current = src.scan_source_snapshots(home)
        assert flatten(current) == flatten(first)
        assert current[0]['source_key'] == first[0]['source_key']
        assert current[0]['source_epoch'] == first[0]['source_epoch']


def test_native_receipts_are_home_scoped_A_B_A_and_wrong_epoch_cannot_leak(native, tmp_path):
    src, _, _, _, a, db_a, baseline_a, _ = native
    b = tmp_path / 'other-home'
    b.mkdir()
    db_b = SessionDB(b / 'state.db')
    try:
        db_b.create_session(SID, 'cli')
        db_b.append_message(SID, 'user', content='Different home synthetic legacy result', timestamp=100.0)
        baseline_b = src.capture_physical_baseline(b, bank='other-bank')
        compact(db_a, carried(db_a))
        compact(db_b, carried(db_b))
        first = src.scan_source_snapshots(a, physical_baseline=baseline_a)
        assert flatten(src.scan_source_snapshots(b, physical_baseline=baseline_b)) == []
        assert src.scan_source_snapshots(a, physical_baseline=baseline_a) == first
        saved = copy.deepcopy(receipt(db_b))
        save_receipt(db_b, receipt(db_a))
        with pytest.raises(src.SourceChangedError):
            src.scan_source_snapshots(b, physical_baseline=baseline_b)
        assert src.scan_source_snapshots(a, physical_baseline=baseline_a) == first
        save_receipt(db_b, saved)
        assert flatten(src.scan_source_snapshots(b, physical_baseline=baseline_b)) == []
    finally:
        db_b.close()


def test_receipt_validation_queries_only_referenced_rows_bounded_in_same_ro_transaction(native, monkeypatch):
    src, _, _, _, home, db, _, _ = native
    batch = [{'role': 'assistant' if n % 2 == 0 else 'user', 'timestamp': 200.0 + n,
              'content': 'Synthetic bounded copy result ' + str(n)} for n in range(128)]
    assert db.append_messages_batch(SID, batch) == 128
    baseline = src.capture_physical_baseline(home, bank='synthetic-bank')
    with db._read_ctx() as con:
        ids = [r[0] for r in con.execute('SELECT id FROM messages ORDER BY id')]
    db.archive_and_compact(SID, [carried(db, row_id) for row_id in ids], tail_count=len(ids))
    assert len(receipt(db)['operations'][0]['copies']) == 129
    reads = []
    real_connect = sqlite3.connect

    class ReadProbe(sqlite3.Connection):
        def execute(self, query, parameters=(), **kwargs):
            if query.startswith('SELECT * FROM messages'):
                assert self.in_transaction
                assert super().execute('PRAGMA query_only').fetchone()[0] == 1
                reads.append((query, len(parameters)))
            return super().execute(query, parameters, **kwargs)

    def connect(*args, **kwargs):
        assert args[0].endswith('?mode=ro')
        return real_connect(*args, factory=ReadProbe, **kwargs)
    monkeypatch.setattr(src.sqlite3, 'connect', connect)
    assert flatten(src.scan_source_snapshots(home, physical_baseline=baseline)) == []
    chunks = [count for query, count in reads if 'WHERE id IN' in query]
    assert len(chunks) == 2 and sum(chunks) == 258 and max(chunks) <= 250
    assert sum('ORDER BY id' in query for query, _ in reads) == 1


def test_projection_keeps_only_owned_receipts_and_stable_relation_markers(native):
    src, _, _, _, _, db, _, _ = native
    compact(db, carried(db))
    value = receipt(db)
    with sqlite3.connect(db.db_path) as con:
        con.execute('UPDATE sessions SET model_config=? WHERE id=?',
            (json.dumps({KEY: value, '_delegate_from': 'native-parent', '_branched_from': 'branch-parent',
                         '_reset_from': 'reset-parent', 'private': 'never-projected'}), SID))
    header, sessions = src._read_physical_db(Path(db.db_path))
    assert set(sessions[SID]['model_config']) == {KEY, '_delegate_from', '_branched_from', '_reset_from'}
    assert 'never-projected' not in json.dumps({'header': header, 'sessions': sessions})


@pytest.mark.parametrize('damage', ['null', 'list', 'duplicate-key', 'blob'])
def test_owned_metadata_has_no_malformed_or_alias_decode_fallback(native, damage):
    src, _, _, _, _, db, _, _ = native
    compact(db, carried(db))
    value = receipt(db)
    if damage == 'null':
        raw = json.dumps({KEY: None})
    elif damage == 'list':
        raw = json.dumps([{KEY: value}])
    elif damage == 'duplicate-key':
        raw = '{"' + KEY + '":' + json.dumps(value) + ',"' + KEY + '":null}'
    else:
        raw = json.dumps({KEY: value}).encode()
    with sqlite3.connect(db.db_path) as con:
        con.execute('UPDATE sessions SET model_config=? WHERE id=?', (raw, SID))
    with pytest.raises(src.SourceChangedError) as caught:
        scan(native)
    assert str(caught.value) == 'source_changed: invalid native message copy receipts'


def test_standalone_source_consumer_needs_no_core_writer_import(native, tmp_path):
    import subprocess
    src, _, _, _, home, db, baseline, _ = native
    compact(db, carried(db))
    baseline_path = tmp_path / 'synthetic-baseline.json'
    baseline_path.write_text(json.dumps(baseline))
    code = ('import importlib.util,json,pathlib,sys; sys.path.insert(0,sys.argv[1]); '
            'assert importlib.util.find_spec("hermes_state") is None; '
            'import hindsight_incremental_sources as s; '
            'result=s.scan_source_snapshots(pathlib.Path(sys.argv[2]),physical_baseline=json.loads(pathlib.Path(sys.argv[3]).read_text())); '
            'assert not [m for view in result for m in view["messages"]]; print("verified-standalone-noop")')
    out = subprocess.run([sys.executable, '-I', '-c', code, str(ROOT), str(home), str(baseline_path)],
                         cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'verified-standalone-noop'

@pytest.mark.parametrize('damage', ['container-deleted', 'empty-operations'])
@pytest.mark.parametrize('tail_count', [0, 1])
def test_new_profile_hidden_twin_cannot_downgrade_after_receipt_loss(native, damage, tail_count):
    src, _, _, _, home, _, baseline, ledger = native
    profile_home = home / 'profiles' / 'new-profile'
    profile_home.mkdir(parents=True)
    db = SessionDB(profile_home / 'state.db')
    try:
        db.create_session(SID, 'cli')
        row_id = db.append_message(SID, 'user', content='New-profile genuine old-date future result', timestamp=1.0)
        assert len(capture(native, src.scan_source_snapshots(home, physical_baseline=baseline))) == 1
        for _ in range(2):
            row_id = compact(db, carried(db, row_id), tail_count)
            assert capture(native, src.scan_source_snapshots(home, physical_baseline=baseline)) == []
        with sqlite3.connect(db.db_path) as con:
            if damage == 'container-deleted':
                con.execute('UPDATE sessions SET model_config=? WHERE id=?', ('{}', SID))
            else:
                value = receipt(db)
                value['operations'] = []
                con.execute('UPDATE sessions SET model_config=? WHERE id=?', (json.dumps({KEY: value}), SID))
            con.execute('DELETE FROM messages WHERE id=?', (row_id,))
        # A genuine independent append with identical display/body/time has no
        # causal receipt; losing the receipt history cannot silently admit it.
        db.append_message(SID, 'user', content='New-profile genuine old-date future result', timestamp=1.0)
        with pytest.raises(src.SourceChangedError, match='ambiguous'):
            src.scan_source_snapshots(home, physical_baseline=baseline)
        assert len(ledger.list_batches()) == 1
    finally:
        db.close()


@pytest.mark.parametrize('damage', ['duplicate-escaped-key', 'malformed-escaped-key', 'blob-escaped-key'])
def test_escaped_owned_receipt_key_aliases_fail_closed(native, damage):
    src, _, _, _, _, db, _, ledger = native
    row_id = db.append_message(SID, 'tool', content='Synthetic tool response', timestamp=123.0)
    compact(db, carried(db, row_id))
    value = receipt(db)
    assert value['operations'][0]['copies'][0]['source_row_id'] == row_id
    escaped = '\\u005fmessage_copy_receipts_v1'
    if damage == 'duplicate-escaped-key':
        raw = '{"' + escaped + '":' + json.dumps(value) + ',"' + escaped + '":null}'
    elif damage == 'malformed-escaped-key':
        raw = '{"' + escaped + '":broken}'
    else:
        raw = ('{"' + escaped + '":' + json.dumps(value) + '}').encode()
    with sqlite3.connect(db.db_path) as con:
        con.execute('UPDATE sessions SET model_config=? WHERE id=?', (raw, SID))
    with pytest.raises(src.SourceChangedError, match='copy receipts'):
        scan(native)
    assert ledger.list_batches() == []


def test_escaped_valid_owned_key_still_projects(native, monkeypatch):
    src, _, _, _, home, db, _, _ = native
    row_id = db.append_message(SID, 'tool', content='Synthetic tool response', timestamp=123.0)
    compact(db, carried(db, row_id))
    value = receipt(db)
    escaped = '\\u005fmessage_copy_receipts_v1'
    with sqlite3.connect(db.db_path) as con:
        con.execute('UPDATE sessions SET model_config=? WHERE id=?',
                    ('{"' + escaped + '":' + json.dumps(value) + '}', SID))
    assert flatten(scan(native)) == []
    assert_noop(native, monkeypatch)
