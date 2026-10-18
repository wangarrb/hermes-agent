"""Same physical identity for finite App input, using only real temporary files."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import uuid

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hindsight_app_source_adapter as adapter
import hindsight_app_approved_run as runner
import hindsight_incremental_sources as sources
from hindsight_incremental_delivery import _freeze
from hindsight_incremental_ledger import IncrementalLedger, LedgerIntegrityError

ACTUAL_OPAQUE_ITEM = 'msg_03886b2b9e4f39ea016ad098cfee788197912c5642bf94d92e'


def fixture(tmp_path, counts=(2, 2, 2), with_database=False, baseline_app_anchor=False):
    home = tmp_path / 'home'
    directory = home / 'sessions'
    directory.mkdir(parents=True)
    if baseline_app_anchor:
        sid = sorted(runner.APPROVED)[0]
        write_export(home, {'session_id': 'codex-app-' + sid,
                            'session_start': 1792022400, 'messages': []})
    if with_database:
        with sqlite3.connect(home / 'state.db') as con:
            con.executescript('CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL, '
                'parent_session_id TEXT, end_reason TEXT, source TEXT); '
                'CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, '
                'timestamp REAL, content TEXT, active INTEGER, compacted INTEGER, origin_message_id TEXT); '
                "INSERT INTO sessions VALUES ('native',100,NULL,NULL,'cli'); "
                "INSERT INTO messages VALUES (1,'native','user',100,'Legacy fixture observation',1,0,'old-native');")
    baseline = sources.capture_physical_baseline(home, bank='hermes')
    ledger = IncrementalLedger(tmp_path / 'ledger', 'hermes', create=True)
    path, checksum = _freeze(ledger.root, baseline, 'physical-baseline')
    reference = {k: baseline[k] for k in ('schema_version', 'bank', 'hermes_home')}
    reference.update(artifact_path=path, artifact_hash=checksum)
    ledger.initialize_legacy(sources.physical_baseline_snapshots(baseline), {}, [],
        confirm='initialize-hindsight-incremental', physical_baseline=reference)
    sessions, pins = [], {}
    for i, (sid, count) in enumerate(zip(sorted(runner.APPROVED), counts)):
        messages = []
        for j in range(count):
            # Opaque item reused in distinct sessions remains distinct composite
            # identity. A string's format does not establish a cross-view alias.
            item = ACTUAL_OPAQUE_ITEM if j == 0 else f'opaque-native-item-{j}'
            turn = str(uuid.UUID(int=200)) if j == 0 else f'turn_opaque_{j}'
            messages.append({'role': 'assistant', 'timestamp': 1792022400 + j,
                'content': f'Independent fixture result for source {i}, observation {j}.',
                'source_turn_id': turn, 'source_item_id': item, 'source_item_order': j,
                'origin_message_id': f'codex-app:{sid}:{turn}:{item}'})
        session = {'session_id': 'codex-app-' + sid, 'source_session_id': sid,
            'source': 'codex-app', 'platform': 'backfill-codex-app',
            'session_start': 1792022400, 'messages': messages}
        sessions.append(session)
        pins[sid] = write_export(home, session)
    return home, ledger, baseline, sessions, pins


def write_export(home, session):
    raw = (json.dumps(session, sort_keys=True) + '\n').encode()
    path = home / 'sessions' / ('session_' + session['session_id'] + '.json')
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def options(home, ledger, sessions, pins):
    return {'ledger': ledger, 'hermes_home': home,
        'approved_session_ids': frozenset(pins), 'approved_source_sha256': pins}


def file_fence(ledger):
    # Explicit bounded fixture directories, never production roots.
    paths = [p for p in ledger.root.glob('*') if p.is_file()]
    paths += list(ledger.payload_dir.glob('*.json'))
    paths += list((ledger.root / 'delivery-artifacts').glob('*.json'))
    return {str(p): (p.stat().st_ino, p.stat().st_size, p.stat().st_mode,
                    hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}


@pytest.mark.parametrize('first_view', ['app', 'physical'])
def test_actual_opaque_item_preserves_physical_identity_dedup_and_readonly_plan(tmp_path, monkeypatch, capsys, first_view):
    home, ledger, baseline, sessions, pins = fixture(tmp_path, counts=(165, 265, 17))
    kwargs = options(home, ledger, sessions, pins)
    before = file_fence(ledger)
    with ledger._connection() as con:
        original_feed = ledger._read(con, 'meta', 'epoch:default:physical-feed')
    snapshots = adapter.build_app_snapshots(sessions, **kwargs)
    plan = adapter.plan_app_input(sessions, **kwargs)
    assert plan['status'] == 'planning_only' and plan['admission_disabled']
    assert plan['new_messages'] == 447 and plan['known_messages'] == 0
    assert plan['conflicts'] == [] and plan['model_calls'] == 0
    assert file_fence(ledger) == before
    physical = [s for s in sources.scan_source_snapshots(home, physical_baseline=baseline)
                if s['source_key'] in {s['source_key'] for s in snapshots}]
    assert len(physical) == len(snapshots) == 3
    by_key = {s['source_key']: s for s in physical}
    for snap in snapshots:
        native = by_key[snap['source_key']]
        assert snap['source_epoch'] == native['source_epoch'] == baseline['profiles'][0]['feed_epoch']
        assert snap.get('epoch_scope', snap['source_key']) == snap['source_key']
        assert snap.get('scope_source_keys', [snap['source_key']]) == [snap['source_key']]
        assert snap['messages'] == native['messages']
        sources.validate_snapshot(snap)
        sources.validate_snapshot(native)
    assert len({m['event_id'] for s in snapshots for m in s['messages']}) == 447
    first, second = (snapshots, physical) if first_view == 'app' else (physical, snapshots)
    batches = [b for snap in first for b in ledger.capture_snapshot(snap)]
    if first_view == 'app':
        with ledger._connection() as con:
            assert ledger._read(con, 'meta', 'epoch:default:physical-feed') == original_feed
    assert batches and all(ledger.capture_snapshot(snap) == [] for snap in second + first)
    with ledger._connection() as con:
        current_feed = ledger._read(con, 'meta', 'epoch:default:physical-feed')
        assert current_feed['epoch'] == original_feed['epoch']
        assert set(original_feed['source_keys']) <= set(current_feed['source_keys'])
    before = file_fence(ledger)
    plan = adapter.plan_app_input(sessions, **kwargs)
    assert plan['known_messages'] == 447 and plan['new_messages'] == 0
    assert all(b['phase'] == 'captured' for b in plan['related_batches'])
    assert file_fence(ledger) == before

    monkeypatch.setattr(runner, 'SOURCE', home / 'sessions')
    monkeypatch.setattr(runner, 'APPROVED_SOURCE_SHA256', pins)
    monkeypatch.setattr(runner, 'APP_LEDGER_ROOT', ledger.storage_root)
    monkeypatch.setattr(runner, 'APP_HERMES_HOME', home)
    def forbidden(*args, **kwargs):
        pytest.fail('plan must not load config, instantiate runtime, capture, or run models')
    for name in ('load_config', 'ArrivalRuntime', 'run_incremental', 'sync_sessions'):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(IncrementalLedger, 'capture_snapshot', forbidden)
    assert runner.main(['--source-mode', 'app', '--plan-only']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['admission_disabled'] and report['ledger_capture'] == 'not_attempted'
    assert report['runtime_preflight'] == 'not_attempted' and report['model_calls'] == 0
    assert file_fence(ledger) == before
    for args in [ ['--source-mode', 'app'], ['--plan-only'],
                 ['--source-mode', 'app', '--plan-only', '--app-contract', 'obsolete'],
                 ['--source-mode', 'app', '--plan-only', '--approved-transition-sha256', 'a' * 64] ]:
        with pytest.raises(SystemExit):
            runner.main(args)


@pytest.mark.parametrize('case', [
    'raw-replacement', 'missing-origin', 'changed-origin', 'changed-item', 'unapproved-70', 'other-session',
    'known-conflict', 'other-source-owner', 'published-payload-changed', 'mixed-batch',
    'mixed-parents', 'legacy-hit', 'unknown-hit', 'unknown-disjoint',
    'original-app-anchor', 'cli-filename-overlap',
    'source-bank', 'batch-without-source', 'unrelated-pending',
])
def test_only_concrete_scope_or_history_conflicts_block_finite_app_plan(tmp_path, monkeypatch, case):
    home, ledger, baseline, sessions, pins = fixture(tmp_path,
        with_database=case.startswith('unknown-'), baseline_app_anchor=case == 'original-app-anchor')
    kwargs = options(home, ledger, sessions, pins)
    if case == 'raw-replacement':
        changed = copy.deepcopy(sessions[0])
        changed['messages'][0]['content'] += ' Unapproved replacement.'
        write_export(home, changed)  # Original independently supplied fixture pin unchanged.
    elif case in {'missing-origin', 'changed-origin', 'changed-item'}:
        raw = sessions[0]['messages'][0]
        if case == 'missing-origin':
            raw.pop('origin_message_id')
        elif case == 'changed-item':
            raw['source_item_id'] = 'msg_unapproved_replacement'
        else:
            raw['origin_message_id'] += '-changed'
        pins[sessions[0]['source_session_id']] = write_export(home, sessions[0])
    elif case == 'unapproved-70':
        sessions[0]['messages'] += [dict(sessions[0]['messages'][0]) for _ in range(70)]
        write_export(home, sessions[0])  # No authority to widen the frozen raw pin.
    elif case == 'other-session':
        sessions.append(dict(sessions[0], source_session_id=str(uuid.UUID(int=999))))
    elif case == 'cli-filename-overlap':
        collision = home / 'sessions' / f"session_codex-{sessions[0]['source_session_id']}.json"
        collision.write_bytes(b'An unrelated CLI body must not be opened for filename overlap checks.')
        read_bytes = adapter._read_bytes
        def do_not_read_cli(path):
            assert Path(path) != collision, 'planning opened the unrelated CLI body'
            return read_bytes(path)
        monkeypatch.setattr(adapter, '_read_bytes', do_not_read_cli)
    elif case == 'original-app-anchor':
        pass
    else:
        snapshots = adapter.build_app_snapshots(sessions, **kwargs)
        target = copy.deepcopy(snapshots[0])
        if case in {'source-bank', 'batch-without-source'}:
            ledger.capture_snapshot(target)
            with ledger.locked(), ledger._connection(write=True) as con:
                if case == 'source-bank':
                    state = ledger._read(con, 'sources', target['source_key'])
                    ledger._put(con, 'sources', target['source_key'], dict(state, bank='other-bank'))
                else:
                    con.execute('DELETE FROM sources WHERE source_key=?', (target['source_key'],))
        elif case == 'unrelated-pending':
            target = dict(target, source_key='default:hermes:unapproved-native',
                          epoch_scope='default:hermes:unapproved-native',
                          scope_source_keys=['default:hermes:unapproved-native'],
                          messages=[dict(target['messages'][0], event_id='unapproved-event',
                                         origin='unapproved-origin')])
            target['snapshot_hash'] = sources.snapshot_hash(target)
            ledger.capture_snapshot(target)
        elif case == 'known-conflict':
            target['messages'][0]['content'] += ' Conflicting captured payload.'
            target['snapshot_hash'] = sources.snapshot_hash(target)
            ledger.capture_snapshot(target)
        elif case == 'other-source-owner':
            target['source_key'] = 'default:hermes:competing-native-source'
            target['epoch_scope'] = target['source_key']
            target['scope_source_keys'] = [target['source_key']]
            target['snapshot_hash'] = sources.snapshot_hash(target)
            ledger.capture_snapshot(target)
        elif case == 'published-payload-changed':
            # Reuse the repository's existing local Runtime fixture through the
            # real pipeline; it supplies no external service or model calls.
            spec = importlib.util.spec_from_file_location('adapter_pipeline_fixture',
                Path(__file__).with_name('test_hindsight_incremental_pipeline.py'))
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            runtime = helper.Runtime(ledger)
            runtime.empty_facts = True
            import hindsight_incremental_pipeline as flow
            def no_model(*args, **kwargs):
                pytest.fail('zero-fact fixture must not call a model')
            report = flow.run_incremental({'bank': ledger.bank, 'hermes_home': str(home)},
                ledger=ledger, client=runtime, llm=no_model,
                source_reader=lambda *a, **k: snapshots, execute=True)
            assert report['status'] == 'completed', report
            batch = ledger.list_batches()[0]
            assert batch['phase'] == 'published' and runtime.generator_calls == 0
            Path(batch['payload_path']).write_bytes(b'{}')
        elif case in {'mixed-batch', 'mixed-parents'}:
            approved_batch = ledger.capture_snapshot(target)[0] if case == 'mixed-parents' else None
            unapproved = {'event_id': 'unapproved-event', 'origin': 'native-unapproved-origin',
                'role': 'user', 'timestamp': target['messages'][0]['timestamp'],
                'content': 'Unapproved fixture event must not enter an approved document.'}
            if case == 'mixed-parents':
                target = dict(target, source_key='default:hermes:unapproved-native', messages=[unapproved])
            else:
                target['messages'].append(unapproved)
            target['snapshot_hash'] = sources.snapshot_hash(target)
            batch = ledger.capture_snapshot(target)[0]
            if case == 'mixed-parents':
                # Model a historical derived batch through the ledger's existing
                # payload writer; no delivery/retain success is fabricated.
                with ledger.locked(), ledger._connection(write=True) as con:
                    derived = dict(target, source_key='derived:fixture', source_epoch='fixture-derived')
                    derived['snapshot_hash'] = sources.snapshot_hash(derived)
                    ledger._make_batch(con, derived, [target['messages'][0]], kind='daily_summary',
                                       parents=(approved_batch['batch_id'], batch['batch_id']))
        elif case == 'legacy-hit':
            ledger.capture_snapshot(target)
            with ledger.locked(), ledger._connection(write=True) as con:
                state = ledger._read(con, 'sources', target['source_key'])
                state['legacy_event_ids'] = [target['messages'][0]['event_id']]
                ledger._put(con, 'sources', target['source_key'], state)
        elif case.startswith('unknown-'):
            origin = (sessions[0]['messages'][0]['origin_message_id'] if case == 'unknown-hit'
                      else 'unrelated-native-unknown-origin')
            with sqlite3.connect(home / 'state.db') as con:
                con.execute("INSERT INTO messages VALUES (2,'native','assistant',1792022400,"
                    "'Independent unknown fixture event',1,0,?)", (origin,))
                con.row_factory = sqlite3.Row
                record = sources._physical_record(dict(con.execute('SELECT * FROM messages WHERE id=2').fetchone()),
                                                  sqlite_row=True)
            profile = baseline['profiles'][0]
            policy = {'schema_version': sources.SOURCE_UNKNOWN_POLICY_SCHEMA, 'bank': ledger.bank,
                'hermes_home': str(home), 'physical_baseline_sha256': sources.digest(baseline),
                'authorization': 'explicit-finite-source-row-exclusion', 'status': 'unknown',
                'completion_proven': False, 'scopes': [{'profile': 'default',
                    'feed_epoch': profile['feed_epoch'], 'source_epoch': profile['sqlite']['epoch'],
                    'session_id': 'native', 'rows': [record]}]}
            ledger.register_source_unknown_policy(policy, confirm='approve-hindsight-source-unknown')
    before = file_fence(ledger)
    if case in {'raw-replacement', 'missing-origin', 'changed-origin', 'changed-item', 'unapproved-70',
                'other-session', 'published-payload-changed', 'original-app-anchor', 'cli-filename-overlap'}:
        reason = {'original-app-anchor': 'approved_app_session_is_original_cutover_anchor',
                  'cli-filename-overlap': 'known_cli_session_overlap'}.get(case)
        with pytest.raises((adapter.AppSourceAmbiguity, LedgerIntegrityError), match=reason):
            adapter.plan_app_input(sessions, **kwargs)
    else:
        result = adapter.plan_app_input(sessions, **kwargs)
        if case == 'unknown-disjoint':
            assert result['status'] == 'planning_only' and not result['conflicts']
        else:
            assert result['status'] == 'blocked' and result['conflicts']
            expected = {'known-conflict': 'source_event_changed',
                'other-source-owner': 'native_event_in_other_source',
                'mixed-batch': 'mixed_unauthorized_source_batch',
                'mixed-parents': 'mixed_unauthorized_derived_parent',
                'legacy-hit': 'legacy_excluded_event', 'unknown-hit': 'finite_unknown_collision',
                'source-bank': 'source_bank_changed',
                'batch-without-source': 'batch_event_without_source_coverage',
                'unrelated-pending': 'unrelated_pending_batch'}
            assert expected[case] in {c['code'] for c in result['conflicts']}
        assert result['model_calls'] == 0 and result['ledger_capture'] == 'not_attempted'
    assert file_fence(ledger) == before
