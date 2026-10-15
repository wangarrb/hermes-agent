"""Approved physical header replacements never expand legacy exclusions.

All SQLite, JSON, policy, ledger and approval inputs are real temporary fixtures.
No production certificate is generated, registered or consumed by this suite.
"""
import importlib
import json
from pathlib import Path
import sqlite3
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def database(home):
    home.mkdir(parents=True, exist_ok=True)
    path = home / 'state.db'
    with sqlite3.connect(path) as con:
        con.executescript('''CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL,
            parent_session_id TEXT, end_reason TEXT, source TEXT, model_config TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            timestamp REAL, content TEXT, active INTEGER, compacted INTEGER,
            origin_message_id TEXT, display_identity BLOB);
            INSERT INTO sessions VALUES ('native',100,NULL,NULL,'cli',NULL);
            INSERT INTO messages VALUES (1,'native','user',100,'Old measured result',1,0,'old-event',x'01');''')
    return path


def replace_db(path):
    staging = path.with_suffix('.replacement')
    with sqlite3.connect(path) as source, sqlite3.connect(staging) as target:
        source.backup(target)
        target.execute('CREATE INDEX IF NOT EXISTS idx_transition_marker ON messages(role)')
    staging.replace(path)


def fixture(tmp_path):
    src = importlib.import_module('hindsight_incremental_sources')
    home = tmp_path / 'home'
    db = database(home)
    reviewer = database(home / 'profiles' / 'reviewer')
    directory = home / 'sessions'
    directory.mkdir()
    export = directory / 'session_json-native.json'
    export.write_text(json.dumps({'session_id': 'json-native', 'session_start': 100,
        'messages': [{'role': 'user', 'timestamp': 100, 'content': 'Legacy JSON evidence',
                      'origin_message_id': 'old-json'}]}))
    baseline = src.capture_physical_baseline(home, bank='fixture')
    default = next(p for p in baseline['profiles'] if p['profile'] == 'default')
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'native','user',101,'Finite unknown event',1,0,'unknown-event',x'02')")
        con.row_factory = sqlite3.Row
        unknown = src._physical_record(dict(con.execute('SELECT * FROM messages WHERE id=2').fetchone()), sqlite_row=True)
    policy = {'schema_version': src.SOURCE_UNKNOWN_POLICY_SCHEMA, 'bank': 'fixture',
        'hermes_home': str(home), 'physical_baseline_sha256': src.digest(baseline),
        'authorization': 'explicit-finite-source-row-exclusion', 'status': 'unknown',
        'completion_proven': False, 'scopes': [{'profile': 'default',
            'feed_epoch': default['feed_epoch'], 'source_epoch': default['sqlite']['epoch'],
            'session_id': 'native', 'rows': [unknown]}]}
    fields = ('identity', 'epoch', 'schema_version', 'application_id', 'message_columns')
    old = {k: default['sqlite'][k] for k in fields}
    replace_db(db)
    certificate = {'schema_version': src.SOURCE_TRANSITION_SCHEMA, 'bank': 'fixture',
        'hermes_home': str(home), 'profile': 'default', 'state_db': str(db),
        'physical_baseline_sha256': src.digest(baseline),
        'source_unknown_policy_sha256': src.digest(policy),
        'authorization': 'explicit-exact-sqlite-header-transition',
        'approval_reference': 'fixture-only-explicit-approval',
        'continuity_proof_sha256': src.digest({'fixture': 'old rows are checked live'}),
        'native_copy_receipt_precondition': {
            'policy': 'current-epoch-exact-inventory', 'sessions': 0, 'sha256': None},
        'old': old, 'new': src.inspect_sqlite_transition_header(db)}
    return src, home, db, reviewer, export, baseline, policy, certificate


def scan(f, *, certificate=True, approved=True):
    src, home, db, reviewer, export, baseline, policy, cert = f
    options = {'physical_baseline': baseline, 'source_unknown_policy': policy}
    if certificate:
        options['source_transition_certificates'] = [cert]
    if approved:
        options['approved_transition_sha256s'] = [src.digest(cert)]
    return src.scan_source_snapshots(home, **options)


def install_real_receipt(src, db, source_id, target_id, epoch, *, append=False):
    with sqlite3.connect(db) as con:
        con.execute('INSERT INTO messages SELECT ?,session_id,role,timestamp,content,active,compacted,'
                    'origin_message_id,display_identity FROM messages WHERE id=?', (target_id, source_id))
        con.row_factory = sqlite3.Row
        rows = {r['id']: src._physical_record(dict(r), sqlite_row=True)
                for r in con.execute('SELECT * FROM messages WHERE id IN (?,?)', (source_id, target_id))}
        operation = {'operation_id': str(uuid.uuid4()), 'kind': 'inplace_compaction', 'copies': [{
            'source_row_id': source_id, 'target_row_id': target_id,
            'source_row_hash': rows[source_id]['row_hash'], 'target_row_hash': rows[target_id]['row_hash'],
            'copy_hash': rows[source_id]['copy_hash']}]}
        operation['receipt_sha256'] = src.digest(operation)
        operations = [operation]
        if append:
            current = json.loads(con.execute(
                "SELECT model_config FROM sessions WHERE id='native'").fetchone()[0])
            operations = [*current[src._COPY_RECEIPTS_KEY]['operations'], operation]
        container = {'schema_version': src._COPY_RECEIPTS_SCHEMA, 'session_id': 'native',
                     'source_epoch': epoch, 'operations': operations}
        config = {src._COPY_RECEIPTS_KEY: container}
        con.execute('UPDATE sessions SET model_config=? WHERE id=?', (json.dumps(config), 'native'))
        # Establish that this is a valid native receipt for the stated epoch.
        roots, sources, checksum = src._native_copy_bindings(con, {'native': {'model_config': config}}, epoch)
        assert roots.get(target_id) == source_id and source_id in sources
        return checksum, operation


def test_approved_transition_keeps_existing_ledger_epoch_and_captures_future_once(tmp_path):
    f = fixture(tmp_path / 'A')
    src, home, db, reviewer, export, baseline, policy, cert = f
    ledger_mod = importlib.import_module('hindsight_incremental_ledger')
    delivery = importlib.import_module('hindsight_incremental_delivery')
    ledger = ledger_mod.IncrementalLedger(tmp_path / 'ledger', 'fixture', create=True)
    path, checksum = delivery._freeze(ledger.root, baseline, 'physical-baseline')
    reference = {k: baseline[k] for k in ('schema_version', 'bank', 'hermes_home')}
    reference.update(artifact_path=path, artifact_hash=checksum)
    ledger.initialize_legacy(src.physical_baseline_snapshots(baseline), {}, [],
        confirm='initialize-hindsight-incremental', physical_baseline=reference)
    initial = ledger.initialization_manifest()
    baseline_before, policy_before = src.digest(baseline), src.digest(policy)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (3,'native','assistant',102,'New independent measurement',1,0,'future-event',x'03')")
    data = json.loads(export.read_text())
    data['messages'].append({'role': 'user', 'timestamp': 1, 'content': 'Late independent JSON event',
                             'origin_message_id': 'future-json'})
    export.write_text(json.dumps(data))
    first = scan(f)
    assert {m['origin'] for s in first for m in s['messages']} == {'future-event', 'future-json'}
    expected_epochs = {p['profile']: p['feed_epoch'] for p in baseline['profiles']}
    assert all(s['source_epoch'] == expected_epochs[s['profile']] for s in first)
    batches = [b for s in first for b in ledger.capture_snapshot(s)]
    assert len(batches) == 2
    # A different home cannot consume A's certificate, even with its approved digest.
    other = fixture(tmp_path / 'B')
    with pytest.raises(ValueError, match='scope'):
        src.scan_source_snapshots(other[1], physical_baseline=other[5], source_unknown_policy=other[6],
            source_transition_certificates=[cert], approved_transition_sha256s=[src.digest(cert)])
    assert scan(f) == first  # A -> B -> A has no process-global approval state.
    assert [b for s in scan(f) for b in ledger.capture_snapshot(s)] == []
    assert ledger.initialization_manifest() == initial
    assert ledger.physical_baseline_reference() == reference
    assert (src.digest(baseline), src.digest(policy)) == (baseline_before, policy_before)
    receipt_hash, _ = install_real_receipt(src, db, 3, 4, cert['new']['epoch'])
    # Original empty inventory approval cannot silently admit new receipts.
    with pytest.raises(src.SourceChangedError, match='receipt inventory'):
        scan(f)
    cert['native_copy_receipt_precondition'].update(sessions=1, sha256=receipt_hash)
    # A separately pinned exact current-epoch inventory keeps causal copies no-op.
    assert {m['origin'] for s in scan(f) for m in s['messages']} == {'future-event', 'future-json'}
    assert [b for s in scan(f) for b in ledger.capture_snapshot(s)] == []
    assert ledger.initialization_manifest() == initial


@pytest.mark.parametrize('damage', ['absent', 'missing-approval', 'wrong-approved-hash',
    'wrong-baseline', 'wrong-policy', 'third-inode', 'schema-cookie', 'structure',
    'application-id', 'columns', 'old-row', 'missing-old-row', 'unknown', 'missing-unknown',
    'session', 'missing-session', 'other-profile', 'other-profile-row', 'json-prefix', 'copy-receipt',
    'legacy-copy-receipt'])
def test_transition_never_weakens_other_source_proofs(tmp_path, damage):
    f = fixture(tmp_path)
    src, home, db, reviewer, export, baseline, policy, cert = f
    if damage == 'absent':
        with pytest.raises(src.SourceChangedError):
            scan(f, certificate=False, approved=False)
        return
    if damage == 'missing-approval':
        with pytest.raises(ValueError):
            scan(f, approved=False)
        return
    if damage == 'wrong-approved-hash':
        with pytest.raises(ValueError, match='digest'):
            src.scan_source_snapshots(home, physical_baseline=baseline, source_unknown_policy=policy,
                source_transition_certificates=[cert], approved_transition_sha256s=['0' * 64])
        return
    if damage in {'wrong-baseline', 'wrong-policy'}:
        cert['physical_baseline_sha256' if damage == 'wrong-baseline' else 'source_unknown_policy_sha256'] = '0' * 64
        with pytest.raises(ValueError, match='scope'):
            scan(f)
        return
    if damage == 'third-inode':
        replace_db(db)
    elif damage == 'other-profile':
        replace_db(reviewer)
    elif damage == 'json-prefix':
        data = json.loads(export.read_text())
        data['messages'][0]['content'] = 'Rewritten old JSON'
        export.write_text(json.dumps(data))
    elif damage == 'legacy-copy-receipt':
        receipt_hash, _ = install_real_receipt(src, db, 1, 4, cert['old']['epoch'])
        cert['native_copy_receipt_precondition'].update(sessions=1, sha256=receipt_hash)
        # Even an approved inventory cannot translate old receipt epochs.
    else:
        statements = {
            'schema-cookie': 'PRAGMA schema_version=999',
            'structure': 'CREATE INDEX idx_unapproved_structure ON messages(content)',
            'application-id': 'PRAGMA application_id=123',
            'columns': 'ALTER TABLE messages ADD COLUMN unexpected TEXT',
            'old-row': "UPDATE messages SET content='Rewritten immutable row' WHERE id=1",
            'missing-old-row': 'DELETE FROM messages WHERE id=1',
            'unknown': "UPDATE messages SET content='Drifted unknown' WHERE id=2",
            'missing-unknown': 'DELETE FROM messages WHERE id=2',
            'session': 'UPDATE sessions SET started_at=200',
            'missing-session': 'DELETE FROM sessions',
            'other-profile-row': "UPDATE messages SET content='Unrelated old source drift' WHERE id=1",
            'copy-receipt': "UPDATE sessions SET model_config='{' || '\"_message_copy_receipts_v1\":{}}'",
        }
        with sqlite3.connect(reviewer if damage == 'other-profile-row' else db) as con:
            con.execute(statements[damage])
            if damage == 'structure':
                # Even a restored cookie cannot hide unapproved DDL.
                con.execute('PRAGMA schema_version=' + str(cert['new']['schema_version']))
    with pytest.raises(src.SourceChangedError):
        scan(f)


def make_certificate(src, baseline, profile, db_path, policy, home):
    record = next(p for p in baseline['profiles'] if p['profile'] == profile)
    fields = ('identity', 'epoch', 'schema_version', 'application_id', 'message_columns')
    old = {k: record['sqlite'][k] for k in fields}
    return {'schema_version': src.SOURCE_TRANSITION_SCHEMA, 'bank': 'fixture',
        'hermes_home': str(home), 'profile': profile, 'state_db': str(db_path),
        'physical_baseline_sha256': src.digest(baseline),
        'source_unknown_policy_sha256': src.digest(policy),
        'authorization': 'explicit-exact-sqlite-header-transition',
        'approval_reference': f'fixture-only-explicit-approval-{profile}',
        'continuity_proof_sha256': src.digest({'fixture': f'{profile} old rows are checked live'}),
        'native_copy_receipt_precondition': {'policy': 'current-epoch-exact-inventory',
            'sessions': 0, 'sha256': None},
        'old': old, 'new': src.inspect_sqlite_transition_header(db_path)}


def test_multi_profile_transition_set_covers_each_drifted_profile(tmp_path):
    f = fixture(tmp_path)
    src, home, db, reviewer, export, baseline, policy, cert = f
    replace_db(reviewer)
    reviewer_cert = make_certificate(src, baseline, 'reviewer', reviewer, policy, home)
    certificates = [cert, reviewer_cert]
    approved = [src.digest(cert), src.digest(reviewer_cert)]
    snapshots = src.scan_source_snapshots(home, physical_baseline=baseline,
        source_unknown_policy=policy, source_transition_certificates=certificates,
        approved_transition_sha256s=approved)
    assert {s['profile'] for s in snapshots} == {'default', 'reviewer'}
    # The same read without the reviewer certificate stays strict per profile.
    with pytest.raises(src.SourceChangedError):
        src.scan_source_snapshots(home, physical_baseline=baseline,
            source_unknown_policy=policy, source_transition_certificates=[cert],
            approved_transition_sha256s=[src.digest(cert)])


def test_transition_set_requires_matched_unique_pins(tmp_path):
    f = fixture(tmp_path)
    src, home, db, reviewer, export, baseline, policy, cert = f
    scan_args = {'physical_baseline': baseline, 'source_unknown_policy': policy}
    with pytest.raises(ValueError, match='together'):
        src.scan_source_snapshots(home, **scan_args, source_transition_certificates=[cert],
            approved_transition_sha256s=None)
    with pytest.raises(ValueError, match='together'):
        src.scan_source_snapshots(home, **scan_args, source_transition_certificates=None,
            approved_transition_sha256s=[src.digest(cert)])
    with pytest.raises(ValueError, match='mismatch'):
        src.scan_source_snapshots(home, **scan_args, source_transition_certificates=[cert],
            approved_transition_sha256s=['0' * 64])
    with pytest.raises(ValueError, match='unused'):
        src.scan_source_snapshots(home, **scan_args, source_transition_certificates=[cert],
            approved_transition_sha256s=[src.digest(cert), '0' * 64])
    with pytest.raises(ValueError, match='duplicate approved'):
        src.scan_source_snapshots(home, **scan_args, source_transition_certificates=[cert],
            approved_transition_sha256s=[src.digest(cert), src.digest(cert)])
    twin = json.loads(json.dumps(cert))
    twin['approval_reference'] = 'fixture-only-explicit-approval-second'
    with pytest.raises(ValueError, match='duplicate source transition profile'):
        src.scan_source_snapshots(home, **scan_args, source_transition_certificates=[cert, twin],
            approved_transition_sha256s=[src.digest(cert), src.digest(twin)])


def test_inplace_schema_transition_without_replacement_is_coverable(tmp_path):
    src = importlib.import_module('hindsight_incremental_sources')
    home = tmp_path / 'home'
    db = database(home)
    baseline = src.capture_physical_baseline(home, bank='fixture')
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'native','user',101,'Finite unknown event',1,0,'unknown-event',x'02')")
        con.row_factory = sqlite3.Row
        unknown = src._physical_record(dict(con.execute('SELECT * FROM messages WHERE id=2').fetchone()), sqlite_row=True)
    default = next(p for p in baseline['profiles'] if p['profile'] == 'default')
    policy = {'schema_version': src.SOURCE_UNKNOWN_POLICY_SCHEMA, 'bank': 'fixture',
        'hermes_home': str(home), 'physical_baseline_sha256': src.digest(baseline),
        'authorization': 'explicit-finite-source-row-exclusion', 'status': 'unknown',
        'completion_proven': False, 'scopes': [{'profile': 'default',
            'feed_epoch': default['feed_epoch'], 'source_epoch': default['sqlite']['epoch'],
            'session_id': 'native', 'rows': [unknown]}]}
    # The live db evolves its schema in place; inode and epoch stay put.
    with sqlite3.connect(db) as con:
        con.execute('CREATE INDEX idx_inplace_evolution ON messages(role)')
    inplace = make_certificate(src, baseline, 'default', db, policy, home)
    assert inplace['old']['identity'] == inplace['new']['identity']
    assert inplace['old']['epoch'] == inplace['new']['epoch']
    assert inplace['old']['schema_version'] != inplace['new']['schema_version']
    snapshots = src.scan_source_snapshots(home, physical_baseline=baseline,
        source_unknown_policy=policy, source_transition_certificates=[inplace],
        approved_transition_sha256s=[src.digest(inplace)])
    assert {s['profile'] for s in snapshots} == {'default'}
    # A further unapproved in-place change still fails closed.
    with sqlite3.connect(db) as con:
        con.execute('CREATE INDEX idx_unapproved_evolution ON messages(content)')
    with pytest.raises(src.SourceChangedError):
        src.scan_source_snapshots(home, physical_baseline=baseline,
            source_unknown_policy=policy, source_transition_certificates=[inplace],
            approved_transition_sha256s=[src.digest(inplace)])


def test_superset_receipt_inventory_pins_existing_operations(tmp_path):
    f = fixture(tmp_path)
    src, home, db, reviewer, export, baseline, policy, cert = f
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (3,'native','assistant',102,'New independent measurement',1,0,'future-event',x'03')")
    checksum, op1 = install_real_receipt(src, db, 3, 4, cert['new']['epoch'])
    cert['native_copy_receipt_precondition'] = {'policy': 'current-epoch-superset-inventory',
        'sessions': 1, 'sha256': checksum,
        'operations': {'native': {op1['operation_id']: src.digest(op1)}}}
    assert scan(f)  # The pinned operation persists; the scan passes.
    # A later legitimate append keeps the pinned operation and is admitted.
    _, op2 = install_real_receipt(src, db, 1, 5, cert['new']['epoch'], append=True)
    assert scan(f)
    # Removing the pinned operation fails closed even while the rest stays valid.
    with sqlite3.connect(db) as con:
        current = json.loads(con.execute(
            "SELECT model_config FROM sessions WHERE id='native'").fetchone()[0])
        current[src._COPY_RECEIPTS_KEY]['operations'] = [
            op for op in current[src._COPY_RECEIPTS_KEY]['operations']
            if op['operation_id'] != op1['operation_id']]
        con.execute('UPDATE sessions SET model_config=? WHERE id=?', (json.dumps(current), 'native'))
    with pytest.raises(src.SourceChangedError, match='operation lost'):
        scan(f)
