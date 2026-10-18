from pathlib import Path
import importlib.util
import json
import multiprocessing
import os
import socket
import sqlite3
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]


def modules():
    sys.path.insert(0, str(ROOT))
    import hindsight_incremental_sources as src
    import hindsight_incremental_ledger as ld
    import hindsight_incremental_delivery as delivery
    import hindsight_incremental_pipeline as flow
    return src, ld, delivery, flow


def snapshot(src, count):
    messages = []
    for i in range(count):
        row = {'role': 'user', 'timestamp': f'2026-10-11T12:{i:02d}:00+00:00',
               'origin': f'message-{i}', 'content': f'New collision avoidance finding {i}: keep the measured evidence.'}
        row['event_id'] = src.digest(row)
        messages.append(row)
    value = {'source_key': 'default:session-A', 'source_epoch': 'epoch-A', 'profile': 'default',
             'messages': messages, 'source_hash_before': src.digest(messages), 'source_hash_after': src.digest(messages)}
    value['snapshot_hash'] = src.snapshot_hash(value)
    return value


class Runtime:
    def __init__(self, ledger, *, gate_pass=True, publish_fails=False):
        self.bank = ledger.bank
        self.ledger = ledger
        self.posts = []
        self.operations = {}
        self.generator_calls = 0
        self.llm_calls = 0
        self.windows = 0
        self.cleanups = 0
        self.gates = 0
        self.publishes = 0
        self.published = {}
        self.gate_pass = gate_pass
        self.publish_fails = publish_fails
        self.empty_facts = False

    def retain_items(self, items, *, async_mode=True, operation_id=None, timeout=None):
        assert async_mode is True
        assert operation_id is not None
        assert all(isinstance(v, str) for item in items for v in item.get('metadata', {}).values())
        assert all(item.get('observation_scopes') == [[f'arrival-batch:{item["document_id"].split("::")[-1]}']] for item in items)
        if operation_id not in self.operations:
            self.posts.append(operation_id)
            self.operations[operation_id] = {'operation_id': operation_id, 'bank_id': self.bank,
                                            'status': 'completed', 'child_operations': None}
        return {'operation_id': operation_id, 'status': 'queued'}

    def get_operation(self, operation_id, timeout=None):
        return self.operations.get(operation_id, {'operation_id': operation_id, 'status': 'not_found'})

    def open_window(self, ledger):
        self.windows += 1
        return {'owner': 'scoped-only', 'legacy_pending': 56}

    def cleanup_window(self, ledger, window):
        self.cleanups += 1
        return 'restored'

    def facts_for_batch(self, batch):
        if self.empty_facts:
            return []
        return [{'id': str(uuid.uuid5(uuid.NAMESPACE_URL, batch['batch_id'])),
                 'document_id': batch['document_id'], 'text': 'A verified new collision avoidance fact.',
                 'type': 'world', 'created_at': '2026-10-11T13:00:00+00:00'}]

    def consolidate_batch(self, batch, policy, *, single_poll=False):
        return {'bank': self.bank, 'scope': [f'arrival-batch:{batch["batch_id"]}'],
                'pending_in_scope': 0, 'failed_in_scope': 0, 'operation_ids': [],
                'legacy_pending': 56, 'ready': True}

    def generate_summary(self, frozen, *, retry_policy=None):
        self.generator_calls += 1
        self.llm_calls += 1
        facts = [json.loads(m['content']) for m in frozen['messages']]
        observation = {'bank': self.bank, 'insight': facts[0]['text'], 'topic': 'aeb', 'type': 'fact',
                       'applicability': 'measured scene', 'evidence_ids': [f['id'] for f in facts]}
        artifact = {'canonical_observations': [observation]}
        return {'items': [{'document_id': frozen['items'][0]['document_id'],
                           'content': json.dumps(artifact), 'metadata': {'kind': 'daily_summary'}}],
                'artifact': artifact}

    def validate_summary(self, artifact):
        assert isinstance(artifact.get('canonical_observations'), list)
        return True

    def gate(self, manifest):
        self.gates += 1
        return {'bank': self.bank, 'manifest_sha256': manifest['manifest_sha256'],
                'decision': 'eligible_for_local_proposal' if self.gate_pass else 'blocked_keep_local_only',
                'checks': {k: self.gate_pass for k in ('schema', 'coverage', 'conflict', 'generic_eval', 'local_eval', 'scoped_drain')}}

    def publish_generation(self, manifest, gate):
        self.publishes += 1
        if self.publish_fails:
            raise ConnectionError('test transport failure after frozen gate')
        self.published[manifest['manifest_sha256']] = {'bank': self.bank,
            'manifest_sha256': manifest['manifest_sha256'], 'published': True,
            'readback_verified': True, 'legacy_seed_preserved': True}

    def readback_generation(self, manifest):
        return self.published.get(manifest['manifest_sha256'], {'published': False})


@pytest.fixture
def setup(tmp_path):
    src, ld, delivery, flow = modules()
    ledger = ld.IncrementalLedger(tmp_path / 'state', 'bank-a', create=True)
    path, checksum = delivery._freeze(ledger.root, [], 'seed')
    seed = {'bank': ledger.bank, 'artifact_path': path, 'artifact_hash': checksum,
            'provenance': {'bank': ledger.bank, 'seed_sha256': src.digest([])}}
    old = snapshot(src, 1)
    ledger.initialize_legacy([old], seed, [], confirm='initialize-hindsight-incremental')
    cfg = {'hermes_home': str(tmp_path / 'home'), 'bank': ledger.bank}
    return src, ledger, flow, cfg


def test_cli_initialize_explicitly_selects_physical_noncompletion_baseline(setup, monkeypatch):
    _src, _ledger, flow, cfg = setup
    monkeypatch.setattr(flow, 'load_config', lambda *_a, **_k: cfg)
    seen = []
    monkeypatch.setattr(flow, 'initialize', lambda config, **kwargs: seen.append(kwargs) or {'legacy_status': 'legacy_unknown'})
    assert flow.main(['initialize', '--seed-file', '/isolated/seed.json', '--confirm', 'initialize-hindsight-incremental']) == 0
    assert seen[0]['physical_baseline'] is True


def test_normal_discovery_loads_exact_physical_receipt_without_paid_work(setup, monkeypatch):
    src, ledger, flow, cfg = setup
    from hindsight_incremental_delivery import _freeze
    descriptor = {'schema_version': 'hindsight-physical-legacy-baseline/1', 'bank': ledger.bank,
                  'hermes_home': cfg['hermes_home'], 'status': 'legacy_unknown', 'completion_proven': False,
                  'profiles': []}
    path, digest = _freeze(ledger.root, descriptor, 'physical-baseline')
    reference = {k: descriptor[k] for k in ('schema_version', 'bank', 'hermes_home')}
    reference.update(artifact_path=path, artifact_hash=digest)
    original = ledger.initialization_manifest()
    monkeypatch.setattr(ledger, 'initialization_manifest', lambda: dict(original, physical_baseline=reference))
    monkeypatch.setattr(ledger, 'physical_baseline_reference', lambda: reference, raising=False)
    monkeypatch.setattr(src, 'validate_physical_baseline', lambda value, **kw: None, raising=False)
    seen = []
    def reader(home, **kwargs):
        seen.append((home, kwargs))
        return [snapshot(src, 1)]
    runtime = Runtime(ledger)
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                  source_reader=reader, execute=True)
    assert report['status'] == 'no_op', report
    assert seen[0][1]['physical_baseline'] == descriptor
    assert runtime.posts == [] and runtime.windows == runtime.generator_calls == 0


def test_physical_receipt_hash_failure_never_falls_back_to_old_discovery(setup, monkeypatch):
    src, ledger, flow, cfg = setup
    from hindsight_incremental_delivery import _freeze
    path, _hash = _freeze(ledger.root, {'bank': ledger.bank}, 'physical-baseline')
    reference = {'schema_version': 'hindsight-physical-legacy-baseline/1', 'bank': ledger.bank,
                 'hermes_home': cfg['hermes_home'], 'artifact_path': path, 'artifact_hash': 'f' * 64}
    original = ledger.initialization_manifest()
    monkeypatch.setattr(ledger, 'initialization_manifest', lambda: dict(original, physical_baseline=reference))
    monkeypatch.setattr(ledger, 'physical_baseline_reference', lambda: reference, raising=False)
    runtime = Runtime(ledger)
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_kw: pytest.fail('old reader reached after a broken physical fence'), execute=True)
    assert report['status'] == 'blocked' and report['calls'] == report['llm_calls'] == 0
    assert runtime.posts == [] and ledger.list_batches() == []


def test_real_physical_initializer_then_new_origin_full_and_repeat_noop(tmp_path):
    import sqlite3
    src, ld, _delivery, flow = modules()
    home = tmp_path / 'physical-home'
    home.mkdir()
    (home / 'sessions').mkdir()
    db = home / 'state.db'
    con = sqlite3.connect(db)
    con.executescript('''
        CREATE TABLE sessions(id TEXT PRIMARY KEY, started_at REAL, parent_session_id TEXT, end_reason TEXT, source TEXT);
        CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT,
            created_at REAL, platform_message_id TEXT, active INTEGER, compacted INTEGER, display_order INTEGER);
        INSERT INTO sessions VALUES('native-session', 1, 'missing-old-parent', NULL, 'cli');
        INSERT INTO messages VALUES(1, 'native-session', 'user', 'A malformed legacy source with exact retained content.', NULL, NULL, 1, 0, 0);
        INSERT INTO messages VALUES(2, 'native-session', 'user', 'A malformed legacy source with exact retained content.', NULL, NULL, 1, 0, 1);
    ''')
    con.commit(); con.close()
    seed = {'bank': 'bank-a', 'observations': [], 'provenance': {'bank': 'bank-a', 'seed_sha256': src.digest([])},
            'remote_seed': None, 'legacy_unknown': [{'status': 'legacy_unknown', 'reason': 'noncompletion fixture'}]}
    seed['export_sha256'] = flow.canonical_hash(seed)
    path = tmp_path / 'external-seed.json'
    path.write_text(json.dumps(seed))
    cfg = {'hermes_home': str(home), 'bank': 'bank-a', 'incremental': {'root': str(tmp_path / 'state')}}
    manifest = flow.initialize(cfg, seed_file=path, confirm='initialize-hindsight-incremental', physical_baseline=True)
    ledger = ld.IncrementalLedger(Path(cfg['incremental']['root']), 'bank-a')
    assert manifest['legacy_status'] == 'legacy_unknown' and manifest['completion_proven'] is False
    assert manifest['excluded_source_events'] == 0 and ledger.list_batches() == []
    runtime = Runtime(ledger)
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary, execute=True)
    assert first['status'] == 'no_op', first
    assert runtime.posts == [] and runtime.generator_calls == runtime.windows == runtime.publishes == 0
    con = sqlite3.connect(db)
    con.execute('INSERT INTO messages VALUES(3, ?, ?, ?, ?, ?, 1, 0, 2)',
                ('native-session', 'user', 'A malformed legacy source with exact retained content.', 1600000000, 'real-new-native-origin'))
    con.commit(); con.close()
    new = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary, execute=True)
    assert new['status'] == 'completed' and new['published'] is True, new
    before = (len(runtime.posts), runtime.generator_calls, runtime.windows, runtime.publishes)
    repeated = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary, execute=True)
    assert repeated['status'] == 'no_op', repeated
    assert before == (len(runtime.posts), runtime.generator_calls, runtime.windows, runtime.publishes)
    assert ledger.initialization_manifest()['completion_proven'] is False


@pytest.fixture
def physical_preparation(tmp_path, monkeypatch):
    src, ld, delivery, flow = modules()
    home = tmp_path / 'physical-home'
    home.mkdir(mode=0o700)
    db = home / 'state.db'
    with sqlite3.connect(db) as con:
        con.executescript('''
            CREATE TABLE sessions(id TEXT PRIMARY KEY, started_at REAL, parent_session_id TEXT, end_reason TEXT, source TEXT);
            CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL,
                content TEXT, active INTEGER, compacted INTEGER, origin_message_id TEXT);
            INSERT INTO sessions VALUES('native-old', NULL, NULL, NULL, 'cli');
            INSERT INTO messages VALUES(1, 'native-old', 'user', NULL, 'PRIVATE_LEGACY_BODY', 1, 0, NULL);
            INSERT INTO messages VALUES(2, 'native-old', 'tool', NULL, 'PRIVATE_LEGACY_TOOL', 1, 0, NULL);
        ''')
    seed = {'bank': 'bank-a', 'observations': [],
            'provenance': {'bank': 'bank-a', 'seed_sha256': src.digest([])},
            'remote_seed': None, 'legacy_unknown': [{'reason': 'unproven'}]}
    seed['export_sha256'] = flow.canonical_hash(seed)
    seed_file = tmp_path / 'external-seed.json'
    seed_file.write_text(json.dumps(seed))
    cfg = {'hermes_home': str(home), 'bank': 'bank-a',
           'incremental': {'root': str(tmp_path / 'ledger')}}
    calls = []
    def forbidden_connect(*args, **kwargs):
        calls.append(args)
        raise AssertionError('isolated initialization must never connect')
    monkeypatch.setattr(socket.socket, 'connect', forbidden_connect)
    yield src, ld, delivery, flow, cfg, db, seed_file
    assert calls == []


def _exit_physical_initialization(cfg, seed_file, boundary):
    _src, ld, _delivery, flow = modules()
    original_freeze = flow._freeze
    def interrupted_freeze(root, value, suffix):
        result = original_freeze(root, value, suffix)
        if suffix == boundary:
            os._exit(74)
        return result
    setattr(flow, '_freeze', interrupted_freeze)
    if boundary in {'plan-before-chmod', 'plan-before-directory-fsync'}:
        original_chmod = os.chmod
        def interrupted_chmod(path, mode, *args, **kwargs):
            if str(path).endswith('.initialization-plan.json') and boundary == 'plan-before-chmod':
                os._exit(74)
            result = original_chmod(path, mode, *args, **kwargs)
            if str(path).endswith('.initialization-plan.json'):
                os._exit(74)
            return result
        os.chmod = interrupted_chmod
    if boundary in {'transaction', 'committed'}:
        original_put = ld.IncrementalLedger._put
        def interrupted_put(self, con, table, key, value):
            result = original_put(self, con, table, key, value)
            if boundary == 'transaction' and table == 'meta' and key == 'legacy':
                os._exit(74)
            return result
        ld.IncrementalLedger._put = interrupted_put
        original_initialize = ld.IncrementalLedger.initialize_legacy
        def interrupted_initialize(self, *args, **kwargs):
            result = original_initialize(self, *args, **kwargs)
            if boundary == 'committed':
                os._exit(74)
            return result
        ld.IncrementalLedger.initialize_legacy = interrupted_initialize
    flow.initialize(cfg, seed_file=seed_file, confirm='initialize-hindsight-incremental', physical_baseline=True)


def _crash_preparation(cfg, seed_file, boundary):
    process = multiprocessing.get_context('fork').Process(target=_exit_physical_initialization,
        args=(cfg, seed_file, boundary))
    process.start()
    process.join(90)
    if process.is_alive():
        process.terminate()
        process.join(10)
        pytest.fail('initialization child did not reach its crash boundary')
    assert process.exitcode == 74


@pytest.mark.parametrize('boundary', ['initialization-plan', 'plan-before-directory-fsync',
                                      'physical-baseline', 'transaction', 'committed'])
def test_physical_initialization_exit_reuses_frozen_scope_and_keeps_late_origin(physical_preparation, monkeypatch, boundary):
    src, ld, delivery, flow, cfg, db, seed_file = physical_preparation
    _crash_preparation(cfg, seed_file, boundary)
    root = Path(cfg['incremental']['root'])
    directory = root / cfg['bank'] / 'delivery-artifacts'
    plans = list(directory.glob('*.initialization-plan.json'))
    assert len(plans) == 1
    frozen = plans[0].read_bytes()
    plan_inode = plans[0].stat().st_ino
    plan = json.loads(frozen)
    assert plan['seed_export_sha256'] == json.loads(seed_file.read_text())['export_sha256']
    assert len(plan['physical_baseline']['profiles'][0]['sqlite']['rows']) == 2
    assert b'PRIVATE_LEGACY_' not in frozen
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES(3, 'native-old', 'user', 1, 'Late old-date observation', 1, 0, 'late-new-origin')")
    def no_recapture(*args, **kwargs):
        pytest.fail('crash retry expanded the frozen exclusion scope')
    monkeypatch.setattr(src, 'capture_physical_baseline', no_recapture)
    if boundary == 'committed':
        with pytest.raises(ValueError, match='already initialized'):
            flow.initialize(cfg, seed_file=seed_file, confirm='initialize-hindsight-incremental', physical_baseline=True)
    else:
        with pytest.raises(ld.LedgerNotInitializedError):
            ld.IncrementalLedger(root, cfg['bank'])
        flow.initialize(cfg, seed_file=seed_file, confirm='initialize-hindsight-incremental', physical_baseline=True)
    ledger = ld.IncrementalLedger(root, cfg['bank'])
    reference = ledger.physical_baseline_reference()
    descriptor = delivery._read(ledger.root, reference['artifact_path'], reference['artifact_hash'])
    assert descriptor == plan['physical_baseline']
    assert plans[0].read_bytes() == frozen and plans[0].stat().st_ino == plan_inode
    assert len(list(directory.glob('*.physical-baseline.json'))) == 1
    manifest = ledger.initialization_manifest()
    assert manifest['completion_proven'] is False and manifest['excluded_source_events'] == 0
    assert ledger.list_batches() == []
    snapshots = src.scan_source_snapshots(Path(cfg['hermes_home']), physical_baseline=descriptor)
    messages = [m for s in snapshots for m in s['messages']]
    assert [m['origin'] for m in messages] == ['late-new-origin']
    assert messages[0]['timestamp'].startswith('1970-01-01')
    # Only the external Native boundary is fake; source/freeze/ledger/run are real.
    runtime = Runtime(ledger)
    runtime.empty_facts = True
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary, execute=True)
    assert report['status'] == 'completed', report
    assert len(runtime.posts) == 1 and runtime.generator_calls == 0
    assert len(ledger.list_batches()) == 1
    source = ledger.list_batches()[0]
    assert source['kind'] == 'source' and source['phase'] == 'published'
    assert source['source_event_ids'] == [messages[0]['event_id']]
    counts = (len(runtime.posts), runtime.windows, runtime.publishes)
    repeat = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary, execute=True)
    assert repeat['status'] == 'no_op' and repeat['calls'] == repeat['llm_calls'] == 0
    assert counts == (len(runtime.posts), runtime.windows, runtime.publishes)


@pytest.mark.parametrize('damage', ['changed-seed', 'invalid-seed-hash', 'changed-home', 'logical-retry',
    'wrong-bank', 'wrong-home', 'wrong-seed', 'invalid-descriptor', 'extra-field', 'schema',
    'duplicate', 'partial-bytes', 'writable', 'hardlink', 'symlink', 'public-directory',
    'wrong-filename', 'no-hash-name', 'partial-name', 'partial-neighbor', 'orphan-physical',
    'no-hash-physical', 'other-physical', 'changed-bank', 'crash-before-chmod', 'seed-null'])
def test_physical_initialization_preparation_binding_rejects_without_recapture(physical_preparation, monkeypatch, damage):
    src, ld, delivery, flow, cfg, _db, seed_file = physical_preparation
    if damage == 'crash-before-chmod':
        _crash_preparation(cfg, seed_file, 'plan-before-chmod')
    ledger = ld.IncrementalLedger(Path(cfg['incremental']['root']), cfg['bank'], create=True)
    descriptor = src.capture_physical_baseline(Path(cfg['hermes_home']), bank=cfg['bank'])
    seed = json.loads(seed_file.read_text())
    plan = {'schema_version': 'hindsight-physical-initialization-plan/1', 'bank': cfg['bank'],
            'hermes_home': cfg['hermes_home'], 'seed_export_sha256': seed['export_sha256'],
            'physical_baseline': descriptor}
    fields = {'wrong-bank': ('bank', 'different-bank'), 'wrong-home': ('hermes_home', '/wrong-home'),
              'wrong-seed': ('seed_export_sha256', 'f' * 64), 'invalid-descriptor': ('physical_baseline', None),
              'extra-field': ('unexpected', True), 'schema': ('schema_version', False)}
    if damage in fields:
        key, value = fields[damage]
        plan[key] = value
    path, checksum = delivery._freeze(ledger.root, plan, 'initialization-plan')
    path = Path(path)
    if damage in {'changed-seed', 'invalid-seed-hash'}:
        seed['legacy_unknown'].append({'reason': 'changed inventory'})
        if damage == 'changed-seed':
            seed['export_sha256'] = flow.canonical_hash({k: v for k, v in seed.items() if k != 'export_sha256'})
        seed_file.write_text(json.dumps(seed))
    elif damage == 'changed-home':
        other = seed_file.parent / 'other-home'
        other.mkdir(mode=0o700)
        cfg = dict(cfg, hermes_home=str(other))
    elif damage == 'duplicate':
        delivery._freeze(ledger.root, dict(plan, seed_export_sha256='e' * 64), 'initialization-plan')
    elif damage == 'partial-bytes':
        path.chmod(0o600); path.write_text('{'); path.chmod(0o400)
    elif damage == 'writable':
        path.chmod(0o600)
    elif damage == 'hardlink':
        os.link(path, path.parent / 'alias')
    elif damage == 'symlink':
        path.rename(path.parent / 'target')
        path.symlink_to(path.parent / 'target')
    elif damage == 'public-directory':
        path.parent.chmod(0o755)
    elif damage == 'wrong-filename':
        path.rename(path.parent / ('e' * 64 + '.initialization-plan.json'))
    elif damage == 'no-hash-name':
        path.rename(path.parent / 'initialization-plan.json')
    elif damage == 'partial-name':
        path.rename(path.with_suffix('.json.partial'))
    elif damage == 'partial-neighbor':
        (path.parent / ('f' * 64 + '.initialization-plan.json.partial')).write_text('{')
    elif damage == 'changed-bank':
        cfg = dict(cfg, bank='bank-b')
    elif damage == 'seed-null':
        seed_file.write_text('null')
    elif damage == 'other-physical':
        delivery._freeze(ledger.root, dict(descriptor, bank='bank-b'), 'physical-baseline')
    elif damage in {'orphan-physical', 'no-hash-physical'}:
        path.rename(path.parent / 'unrelated-report.json')
        physical_path, _ = delivery._freeze(ledger.root, descriptor, 'physical-baseline')
        if damage == 'no-hash-physical':
            Path(physical_path).rename(path.parent / 'physical-baseline.json')
    before = ledger.db_path.read_bytes()
    artifacts = {str(p): p.read_bytes() for p in path.parent.iterdir() if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail('invalid preparation must not recapture or fall back to logical discovery')
    monkeypatch.setattr(src, 'capture_physical_baseline', forbidden)
    from hindsight_offline_retry import RemoteCallError
    with pytest.raises((ValueError, RemoteCallError)):
        flow.initialize(cfg, seed_file=seed_file, confirm='initialize-hindsight-incremental',
                        physical_baseline=damage != 'logical-retry', source_reader=forbidden)
    assert ledger.db_path.read_bytes() == before
    assert {str(p): p.read_bytes() for p in path.parent.iterdir() if p.is_file()} == artifacts
    with sqlite3.connect(ledger.db_path) as con:
        assert con.execute("SELECT count(*) FROM meta WHERE key='legacy'").fetchone()[0] == 0
        assert con.execute('SELECT count(*) FROM sources').fetchone()[0] == 0
        assert con.execute('SELECT count(*) FROM batches').fetchone()[0] == 0


def test_no_new_input_is_zero_delivery_llm_window_or_publish(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == 'no_op'
    assert runtime.posts == []
    assert (runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes) == (0, 0, 0, 0)


def test_resume_only_path_initializes_scan_fingerprints_guard(setup):
    """Regression: resume-only skips the scan branch but still reaches the store guard.

    scan_fingerprints was bound only inside the non-resume branch and then read
    unconditionally after capture, so --resume-only raised UnboundLocalError
    before it could drain preserved work.
    """
    _src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)

    def forbidden_reader(*_a, **_k):
        raise AssertionError('resume-only must not rescan sources')

    resume_cfg = {'hermes_home': cfg['hermes_home'], 'bank': cfg['bank'],
                  'incremental': {'resume_only': True}}
    out = flow.run_incremental(resume_cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=forbidden_reader, execute=True)
    assert out['status'] == 'no_op'
    assert runtime.posts == []
    assert (runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes) == (0, 0, 0, 0)


def test_append_only_new_arrival_completes_and_repeat_costs_zero(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    read = lambda *_a, **_k: [snapshot(src, 2)]
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                source_reader=read, execute=True)
    assert first['status'] == 'completed', first
    assert len(runtime.posts) == 2
    assert runtime.generator_calls == 1
    assert first['published'] is True and first['cleanup'] == 'restored'
    assert {b['kind'] for b in ledger.list_batches()} == {'source', 'daily_summary'}
    assert all(b['phase'] == 'published' for b in ledger.list_batches())
    counts = (len(runtime.posts), runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes)
    repeat = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=read, execute=True)
    assert repeat['status'] == 'no_op'
    assert counts == (len(runtime.posts), runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes)


def test_publish_failure_resumes_only_frozen_publish_not_successful_inputs(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger, publish_fails=True)
    read = lambda *_a, **_k: [snapshot(src, 2)]
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                source_reader=read, execute=True)
    assert first['status'] == 'blocked'
    child = next(b for b in ledger.list_batches() if b['kind'] == 'daily_summary')
    assert child['phase'] == 'gated'
    posts, generated, gate_count = len(runtime.posts), runtime.generator_calls, runtime.gates
    runtime.publish_fails = False
    second = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=read, execute=True)
    assert second['status'] == 'completed', second
    assert (posts, generated, gate_count) == (len(runtime.posts), runtime.generator_calls, runtime.gates)


def test_blocked_quality_does_not_publish_or_rewind_retained(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger, gate_pass=False)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'deferred'
    assert runtime.publishes == 0
    child = next(b for b in ledger.list_batches() if b['kind'] == 'daily_summary')
    assert child['phase'] == 'locally_merged'
    assert child['receipts']['retained']['status'] == 'completed'


def test_new_input_budget_does_not_reconstruct_existing_history(setup):
    src, ledger, flow, cfg = setup
    cfg['incremental'] = {'max_pending_units': 1, 'max_pending_chars': 500000}
    runtime = Runtime(ledger)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'deferred'
    assert len(runtime.posts) == 1 and runtime.generator_calls == 0
    assert out['used_units'] == 1
    assert ledger.initialization_manifest()['legacy_status'] == 'legacy_unknown'


@pytest.mark.parametrize('knobs,accepted', [
    ({'max_pending_units': 200, 'max_pending_chars': 20_000_000}, True),
    ({'max_pending_units': 48, 'max_pending_chars': 2_000_000}, True),
    ({'max_pending_units': 201, 'max_pending_chars': 500000}, False),
    ({'max_pending_units': 12, 'max_pending_chars': 20_000_001}, False),
    ({'max_pending_units': 0, 'max_pending_chars': 500000}, False),
])
def test_normal_incremental_budget_uses_safety_ceilings(setup, knobs, accepted):
    """Heavy loads must be able to drain in one run: the caps are safety
    ceilings, while per-unit deadlines and attempt limits stay the guards."""
    src, ledger, flow, cfg = setup
    cfg['incremental'] = dict(knobs)
    runtime = Runtime(ledger)
    if accepted:
        out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                   source_reader=lambda *_a, **_k: [], execute=False)
        assert out['status'] == 'plan'
    else:
        with pytest.raises(ValueError, match='normal_incremental_budget_exceeded'):
            flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=lambda *_a, **_k: [], execute=False)


def test_plan_does_not_capture_or_mutate_ledger(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    before = ledger.db_path.read_bytes()
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=False)
    assert out['status'] == 'plan'
    assert ledger.db_path.read_bytes() == before
    assert len(ledger.list_batches()) == 0 and runtime.posts == []


def test_successful_zero_fact_source_does_not_remain_permanently_pending(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    runtime.empty_facts = True
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'completed', out
    assert runtime.generator_calls == 0 and runtime.publishes == 0
    repeat = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert repeat['status'] == 'no_op'


def test_opening_failure_reconciles_durable_window_before_returning(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    original = runtime.open_window
    def partial_open(value):
        setattr(runtime, '_window', dict(original(value), status='active'))
        raise ConnectionError('fixture ACK lost after window opened')
    runtime.open_window = partial_open
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'blocked'
    assert out['cleanup'] == 'restored'
    assert runtime.cleanups == 1 and runtime.posts == []


def test_native_deferral_persists_exact_retry_and_blocks_unfinished_parent_publish(setup):
    src, ledger, flow, cfg = setup
    from hindsight_offline_retry import RetryDeferred
    runtime = Runtime(ledger)
    original = runtime.consolidate_batch
    def postpone_source(batch, policy, *, single_poll=False):
        if batch['kind'] == 'source':
            raise RetryDeferred('call_cap', next_retry_at=99999999999)
        return original(batch, policy, single_poll=single_poll)
    runtime.consolidate_batch = postpone_source
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    source = next(b for b in ledger.list_batches() if b['kind'] == 'source')
    assert source['phase'] == 'retained' and source['receipts']['retained']['status'] == 'completed'
    assert source['next_retry_at'] == 99999999999
    assert out['status'] == 'deferred'
    assert runtime.publishes == 0 and out['published'] is False


def test_window_budget_refusal_is_deferred_before_any_retain(setup):
    src, ledger, flow, cfg = setup
    from hindsight_offline_retry import RetryDeferred
    runtime = Runtime(ledger)
    def no_budget(_ledger):
        raise RetryDeferred('call_cap', next_retry_at=99999999999)
    runtime.open_window = no_budget
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    batch = ledger.list_batches()[0]
    assert batch['phase'] == 'captured' and batch['next_retry_at'] == 99999999999
    assert out['status'] == out['core_status'] == 'deferred'
    assert runtime.posts == [] and runtime.generator_calls == 0


def test_publication_preflight_blocks_before_any_new_producer(setup, monkeypatch):
    src, ledger, flow, cfg = setup
    from hindsight_incremental_runtime import ArrivalRuntime
    runtime = ArrivalRuntime(dict(cfg, api_url='http://isolated.invalid'))
    runtime.ledger = ledger
    attempts = []
    def missing_adapter():
        attempts.append('adapter')
        raise RuntimeError('fixture configured interpreter lacks SQL driver')
    monkeypatch.setattr(runtime, '_publication', missing_adapter)
    monkeypatch.setattr(runtime.native, 'request', lambda *_a, **_k: attempts.append('network'))
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'blocked'
    assert attempts == ['adapter'] and runtime.native_calls == runtime.llm_calls == 0
    assert out['cleanup'] == 'not_opened'
    assert ledger.list_batches()[0]['phase'] == 'captured'


def test_noop_does_not_require_adapters_or_credentials_for_sealed_history(setup, monkeypatch):
    src, ledger, flow, cfg = setup
    from hindsight_incremental_runtime import ArrivalRuntime
    runtime = ArrivalRuntime(dict(cfg, api_url='http://isolated.invalid'))
    runtime.ledger = ledger
    def unavailable():
        raise AssertionError('no-op must not import SQL driver or resolve credentials')
    monkeypatch.setattr(runtime, '_publication', unavailable)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == 'no_op' and out['calls'] == out['llm_calls'] == 0


def test_requested_wiki_tail_is_explicitly_degraded_not_silently_successful(setup):
    src, ledger, flow, cfg = setup
    cfg['incremental'] = {'mode': 'full', 'include_wiki': True}
    runtime = Runtime(ledger)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == out['core_status'] == 'no_op'
    assert out['mode'] == 'full'
    assert out['optional_tails']['wiki']['requested'] is True
    assert out['optional_tails']['wiki']['status'] == 'degraded'
    assert out['optional_tails']['wiki']['reason'] == 'bounded_tail_not_implemented'
    assert runtime.posts == [] and runtime.generator_calls == 0


def test_real_cli_records_requested_mode_and_wiki_lane(setup, monkeypatch, capsys):
    src, ledger, flow, cfg = setup
    cfg['incremental'] = {'root': str(ledger.root.parent)}
    monkeypatch.setattr(flow, 'load_config', lambda *_a, **_k: cfg)
    import hindsight_incremental_runtime as module
    monkeypatch.setattr(module, 'ArrivalRuntime', lambda config: Runtime(ledger))
    rc = flow.main(['run', '--mode', 'full', '--include-wiki', '--confirm', 'run-hindsight-pipeline'])
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert rc == 0 and out['mode'] == 'full'
    assert out['optional_tails']['wiki']['requested'] is True
    assert out['optional_tails']['wiki']['status'] != 'completed'


class FakeClock:
    def __init__(self):
        self.value = 1000.0
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


def snapshot_keyed(src, key, count=1):
    messages = []
    for i in range(count):
        row = {'role': 'user', 'timestamp': f'2026-10-11T12:{i:02d}:00+00:00',
               'origin': f'message-{key}-{i}',
               'content': f'New collision avoidance finding {key}-{i}: keep the measured evidence.'}
        row['event_id'] = src.digest(row)
        messages.append(row)
    value = {'source_key': f'default:session-{key}', 'source_epoch': f'epoch-{key}', 'profile': 'default',
             'messages': messages, 'source_hash_before': src.digest(messages), 'source_hash_after': src.digest(messages)}
    value['snapshot_hash'] = src.snapshot_hash(value)
    return value


class PipelinedRuntime(Runtime):
    """Operations complete after N polls; records the exact call order."""
    def __init__(self, ledger, *, polls_to_complete=1, **kw):
        super().__init__(ledger, **kw)
        self.polls_to_complete = polls_to_complete
        self.operation_polls = {}
        self.call_log = []

    def retain_items(self, items, *, async_mode=True, operation_id=None, timeout=None):
        ack = super().retain_items(items, async_mode=async_mode, operation_id=operation_id, timeout=timeout)
        self.operation_polls.setdefault(operation_id, 0)
        self.call_log.append(('post', operation_id))
        return ack

    def get_operation(self, operation_id, timeout=None):
        if operation_id not in self.operations:
            self.call_log.append(('poll', operation_id))
            return {'operation_id': operation_id, 'status': 'not_found'}
        self.operation_polls[operation_id] += 1
        done = self.operation_polls[operation_id] >= self.polls_to_complete
        self.call_log.append(('complete' if done else 'poll', operation_id))
        if done:
            return self.operations[operation_id]
        return {'operation_id': operation_id, 'bank_id': self.bank, 'status': 'pending'}

    def consolidate_batch(self, batch, policy, *, single_poll=False):
        return {'bank': self.bank, 'scope': [f'arrival-batch:{batch["batch_id"]}'],
                'pending_in_scope': 0, 'failed_in_scope': 0, 'operation_ids': [],
                'legacy_pending': 56, 'ready': True}


def test_heavy_load_posts_batches_before_earlier_operations_complete(setup):
    src, ledger, flow, cfg = setup
    runtime = PipelinedRuntime(ledger, polls_to_complete=3)
    cfg = dict(cfg, incremental={'max_inflight_units': 3})
    reader = lambda *_a, **_k: [snapshot_keyed(src, k) for k in ('a', 'b', 'c')]
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                  source_reader=reader, execute=True)
    assert report['status'] == 'completed', report
    log = runtime.call_log
    posts = [i for i, entry in enumerate(log) if entry[0] == 'post']
    first_complete = next(i for i, entry in enumerate(log) if entry[0] == 'complete')
    assert posts[2] < first_complete, log


def test_deferred_operation_is_retried_within_the_same_run(setup):
    src, ledger, flow, cfg = setup
    clock = FakeClock()
    runtime = PipelinedRuntime(ledger, polls_to_complete=4)
    retry = flow.RetryPolicy(unit_deadline_s=15, poll_s=5, defer_delay=10, max_calls=1000,
                             clock=clock.now, wall_clock=clock.now, sleep=clock.sleep, jitter=lambda: 0)
    cfg = dict(cfg, incremental={'max_inflight_units': 2})
    reader = lambda *_a, **_k: [snapshot_keyed(src, 'a')]
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                  source_reader=reader, execute=True, clock=clock.now, retry_policy=retry)
    assert report['status'] == 'completed', report
    assert report['remaining_batches'] == 0
    assert any(s['stage'] == 'native_stage' and s.get('status') == 'deferred'
               for s in report['stages']), report['stages']


def test_publish_resume_loads_frozen_gate_reference_without_regating(setup, tmp_path):
    """Gated receipts reference the frozen gate bytes; a resume must load the
    reference (verify hash) instead of duplicating 466KB per batch or re-gating."""
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger, publish_fails=True)
    real_gate = runtime.gate
    def gate_with_freeze(manifest):
        value = real_gate(manifest)
        value['gate_path'] = str(tmp_path / 'gate.json')
        value['gate_sha256'] = flow.canonical_hash(value)
        Path(value['gate_path']).write_text(json.dumps(value, sort_keys=True))
        return value
    runtime.gate = gate_with_freeze
    read = lambda *_a, **_k: [snapshot(src, 2)]
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=read, execute=True)
    assert first['status'] == 'blocked', first
    child = next(b for b in ledger.list_batches() if b['kind'] == 'daily_summary')
    receipt = child['receipts']['gated']
    assert 'gate' not in receipt and isinstance(receipt.get('gate_reference'), dict)
    reference = receipt['gate_reference']
    assert reference['gate_sha256'] and reference['gate_path'].endswith('gate.json')
    runtime.publish_fails = False
    gates_before = runtime.gates
    second = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                  source_reader=read, execute=True)
    assert second['status'] == 'completed', second
    assert runtime.gates == gates_before  # resumed from the frozen reference
