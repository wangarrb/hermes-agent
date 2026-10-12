"""Read-only source fences and stable arrival identity, using real files/DBs.

The two *_core_* cases require the real Hermes SessionDB and are reported in
a separate core E2E lane. Script-only installs explicitly skip just those cases;
the standalone NUL-sentinel codec test below has no Hermes-core prerequisite.
"""
import hashlib
import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def api():
    assert (ROOT / 'hindsight_incremental_sources.py').exists(), 'arrival source adapter is not implemented'
    return importlib.import_module('hindsight_incremental_sources')


def write_session(home, sid, messages, **extra):
    target = home / 'sessions' / f'session_{sid}.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'session_id': sid, 'session_start': 100, 'messages': messages, **extra}))
    return target


def message(text='Useful technical fact', timestamp: int | float | str | None = 100, **extra):
    return {'role': 'user', 'timestamp': timestamp, 'content': text, **extra}


def create_db(home):
    home.mkdir(exist_ok=True)
    path = home / 'state.db'
    with sqlite3.connect(path) as con:
        con.executescript('''CREATE TABLE sessions (id TEXT PRIMARY KEY, started_at REAL,
            parent_session_id TEXT, end_reason TEXT, source TEXT);
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
            timestamp REAL, content TEXT, active INTEGER, compacted INTEGER);
            INSERT INTO sessions VALUES ('s',100,NULL,NULL,'cli');
            INSERT INTO messages VALUES (1,'s','user',100,'Useful technical fact',1,0);''')
    return path


def test_json_tail_late_timestamp_and_duplicate_artifact_identity(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message()])
    first = sources.scan_source_snapshots(home)
    assert len(first) == 1
    event = first[0]['messages'][0]['event_id']
    assert first[0]['source_hash_before'] == first[0]['source_hash_after']
    assert first[0]['snapshot_hash'] == sources.snapshot_hash(first[0])
    clone = home / 'sessions' / 'session_copy.json'
    clone.write_bytes(path.read_bytes())
    repeat = sources.scan_source_snapshots(home)
    assert len(repeat) == 1
    assert repeat[0]['messages'][0]['event_id'] == event
    write_session(home, 's', [message(), message('Late old event', 1), message('Same time distinct', 100)])
    # Copy remains old; union must not replay its prefix or lose the new tail.
    grown = sources.scan_source_snapshots(home)[0]
    assert grown['source_key'] == first[0]['source_key']
    assert len(grown['messages']) == 3
    assert grown['messages'][0]['event_id'] == event
    assert len({m['event_id'] for m in grown['messages']}) == 3


def test_profiles_A_B_A_cleaning_codex_and_cron_exclusion(tmp_path):
    sources = api()
    a = tmp_path / 'A'
    b = tmp_path / 'B'
    write_session(a, 's', [message('Keep this <think>secret reasoning</think> useful result')])
    write_session(b, 's', [message('Different home')])
    write_session(a / 'profiles' / 'reviewer', 'codex-x', [message('# AGENTS.md instructions boilerplate'),
        message('Codex real result', 101)], platform='backfill-codex', model='codex')
    write_session(a, 'cron-job', [message('Automated duplicated reporting')], platform='cron')
    (a / 'profiles' / 'reviewer' / 'config.yaml').write_text('memory:\n  provider: hindsight\n')
    first = sources.scan_source_snapshots(a)
    sources.scan_source_snapshots(b)
    assert sources.scan_source_snapshots(a) == first
    assert {s['profile'] for s in first} == {'default', 'reviewer'}
    assert all('secret reasoning' not in m['content'] for s in first for m in s['messages'])
    assert all('# AGENTS' not in m['content'] for s in first for m in s['messages'])
    assert len(sources.scan_source_snapshots(a, profile_mode='none')) == 1
    assert len(sources.scan_source_snapshots(a, profile_mode='hindsight')) == 2


def test_source_sqlite_unsnapshotted_tail_and_compaction_clones_are_readonly(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    path = create_db(home)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    old = sources.scan_source_snapshots(home)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    event = old[0]['messages'][0]['event_id']
    write_session(home, 's', [message()])
    assert sources.scan_source_snapshots(home)[0]['messages'][0]['event_id'] == event
    with sqlite3.connect(path) as con:
        con.execute("UPDATE sessions SET end_reason='compression' WHERE id='s'")
        con.execute("INSERT INTO sessions VALUES ('child',101,'s',NULL,'cli')")
        con.execute("INSERT INTO messages VALUES (2,'child','user',100,'Useful technical fact',1,0)")
        con.execute("INSERT INTO messages VALUES (3,'child','assistant',102,'New unsnapshotted fact',1,0)")
    current = sources.scan_source_snapshots(home)
    assert len(current) == 1
    assert current[0]['source_epoch'] == old[0]['source_epoch']
    assert current[0]['source_key'] == old[0]['source_key']
    assert current[0]['messages'][0]['event_id'] == event
    assert len(current[0]['messages']) == 2


def test_db_replacement_epoch_changes_and_ledger_refuses_replay(tmp_path):
    sources = api()
    ledger_mod = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    path = create_db(home)
    old = sources.scan_source_snapshots(home)
    ledger = ledger_mod.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(old, {}, [], confirm='initialize-hindsight-incremental')
    saved = home / 'previous.db'
    path.rename(saved)
    create_db(home)
    with sqlite3.connect(path) as con:
        con.execute("UPDATE sessions SET id='replacement-session'")
        con.execute("UPDATE messages SET session_id='replacement-session'")
    new = sources.scan_source_snapshots(home)
    assert new[0]['source_epoch'] != old[0]['source_epoch']
    assert new[0]['source_key'] != old[0]['source_key']
    with pytest.raises(ledger_mod.SourceChangedError, match='epoch'):
        ledger.capture_snapshot(new[0])
    assert ledger.list_batches() == []


def test_sqlite_inplace_reset_or_empty_feed_cannot_hide_coverage_loss(tmp_path):
    sources = api()
    ledger_mod = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    path = create_db(home)
    old = sources.scan_source_snapshots(home)
    ledger = ledger_mod.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(old, {}, [], confirm='initialize-hindsight-incremental')
    with sqlite3.connect(path) as con:
        con.execute('DELETE FROM messages')
        con.execute('DELETE FROM sessions')
    empty = sources.scan_source_snapshots(home)
    assert empty, 'empty SQLite still needs an epoch/coverage diagnostic fence'
    with pytest.raises(ledger_mod.SourceChangedError):
        ledger.capture_snapshot(empty[0])
    with sqlite3.connect(path) as con:
        con.execute("INSERT INTO sessions VALUES ('new-session',100,NULL,NULL,'cli')")
        con.execute("INSERT INTO messages VALUES (1,'new-session','user',100,'Old replay',1,0)")
    new = sources.scan_source_snapshots(home)
    assert new[0]['source_epoch'] == old[0]['source_epoch']
    with pytest.raises(ledger_mod.SourceChangedError):
        ledger.capture_snapshot(new[0])
    assert ledger.list_batches() == []


def test_source_symlink_escape_and_invalid_json_fail_closed(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    target = write_session(home, 's', [message()])
    outside = tmp_path / 'outside.json'
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(sources.SourceChangedError, match='symlink'):
        sources.scan_source_snapshots(home)
    target.unlink()
    target.write_text('{broken')
    with pytest.raises(sources.SourceChangedError, match='invalid'):
        sources.scan_source_snapshots(home)


def test_file_mutation_between_reads_fails_source_fence(tmp_path, monkeypatch):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message()])
    original = sources._read_bytes
    reads = []
    def mutate(target):
        result = original(target)
        if Path(target) == path and not reads:
            reads.append(True)
            write_session(home, 's', [message(), message('Arrived during scan', 101)])
        return result
    monkeypatch.setattr(sources, '_read_bytes', mutate)
    with pytest.raises(sources.SourceChangedError, match='source_changed'):
        sources.scan_source_snapshots(home)


def test_missing_timestamp_without_origin_is_not_an_array_index_identity(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [{'role': 'user', 'content': 'Cannot prove stable identity'}])
    with pytest.raises(sources.SourceChangedError, match='identity'):
        sources.scan_source_snapshots(home)


def test_db_can_enrich_timestamp_less_json_without_inventing_message_identity(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    create_db(home)
    old = sources.scan_source_snapshots(home)[0]
    write_session(home, 's', [{'role': 'user', 'content': 'Useful technical fact'}])
    current = sources.scan_source_snapshots(home)[0]
    assert current['messages'] == old['messages']


def test_same_timestamp_text_with_distinct_original_ids_is_not_deduped(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [message(message_id='first'), message(message_id='second')])
    snapshots = sources.scan_source_snapshots(home)
    assert len(snapshots[0]['messages']) == 2
    assert len({m['event_id'] for m in snapshots[0]['messages']}) == 2


def test_ambiguous_identical_events_without_origin_fail_instead_of_losing_one(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [message(), message()])
    with pytest.raises(sources.SourceChangedError, match='ambiguous'):
        sources.scan_source_snapshots(home)


def test_core_typed_content_is_clean_before_identity_and_late_export_is_noop(tmp_path):
    SessionDB = pytest.importorskip('hermes_state', reason='real Hermes core E2E lane requires SessionDB').SessionDB
    sources = api()
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    home.mkdir()
    blocks = [{'type': 'thinking', 'text': 'Private reasoning must not be retained'},
              {'type': 'text', 'text': 'Useful typed technical fact'}]
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session('s', 'cli')
        db.append_message('s', 'assistant', content=blocks, timestamp=100)
        old = sources.scan_source_snapshots(home)
        assert old[0]['messages'][0]['content'] == 'Useful typed technical fact'
        ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
        ledger.initialize_legacy(old, {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
        write_session(home, 's', [{'role': 'assistant', 'timestamp': 100, 'content': blocks}])
        current = sources.scan_source_snapshots(home)
        assert current[0]['messages'] == old[0]['messages']
        assert [b for snapshot in current for b in ledger.capture_snapshot(snapshot)] == []
    finally:
        db.close()


def test_standalone_typed_content_codec_uses_nul_sentinel_and_preserves_identity(tmp_path, monkeypatch):
    import builtins
    original_import = builtins.__import__
    blocked = []
    def without_core(name, *args, **kwargs):
        if name == 'hermes_state':
            blocked.append(name)
            raise ImportError('standalone script bundle has no Hermes core')
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_core)
    sources = api()
    home = tmp_path / 'home'
    path = create_db(home)
    blocks = [{'type': 'thinking', 'text': 'Private reasoning must not be retained'},
              {'type': 'text', 'text': 'Useful typed technical fact'}]
    tagged = '\x00json:' + json.dumps(blocks)
    with sqlite3.connect(path) as con:
        con.execute('UPDATE messages SET content=? WHERE id=1', (tagged,))
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    old = sources.scan_source_snapshots(home)
    assert blocked, 'must exercise the standalone decoder, not an available core'
    assert old[0]['messages'][0]['content'] == 'Useful typed technical fact'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    write_session(home, 's', [{'role': 'user', 'timestamp': 100, 'content': blocks}])
    assert sources.scan_source_snapshots(home)[0]['messages'] == old[0]['messages']
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(old, {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
    assert ledger.capture_snapshot(sources.scan_source_snapshots(home)[0]) == []
    # Ordinary JSON-looking prose must not be decoded; malformed tagged content
    # must fail closed, never retain serialized reasoning as raw text.
    assert sources._decode_db_content(json.dumps(blocks)) == json.dumps(blocks)
    assert sources._decode_db_content('json:' + json.dumps(blocks)) == 'json:' + json.dumps(blocks)
    with sqlite3.connect(path) as con:
        con.execute('UPDATE messages SET content=? WHERE id=1', ('\x00json:{broken',))
    with pytest.raises(sources.SourceChangedError, match='codec'):
        sources.scan_source_snapshots(home)


def test_json_parent_compression_proof_collapses_prefix_and_preserves_root_epoch(tmp_path):
    sources = api()
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    write_session(home, 'parent', [message()], end_reason='compression')
    old = sources.scan_source_snapshots(home)
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(old, {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
    child = write_session(home, 'child', [message()], parent_session_id='parent')
    data = json.loads(child.read_text())
    data['session_start'] = 101
    child.write_text(json.dumps(data))
    current = sources.scan_source_snapshots(home)
    assert len(current) == 1
    assert current[0]['source_key'] == old[0]['source_key']
    assert current[0]['source_epoch'] == old[0]['source_epoch']
    assert ledger.capture_snapshot(current[0]) == []
    write_session(home, 'child', [message(), message('New child tail', 102)], parent_session_id='parent')
    delta = ledger.capture_snapshot(sources.scan_source_snapshots(home)[0])
    assert len(delta) == 1
    payload = json.loads(Path(delta[0]['payload_path']).read_text())
    assert [m['content'] for m in payload['messages']] == ['New child tail']


@pytest.mark.parametrize('parent_exists', [False, True])
def test_json_child_without_parent_end_reason_proof_is_review_blocked(tmp_path, parent_exists):
    sources = api()
    home = tmp_path / 'home'
    if parent_exists:
        write_session(home, 'parent', [message()])
    write_session(home, 'child', [message()], parent_session_id='parent')
    with pytest.raises(sources.SourceChangedError, match='lineage'):
        sources.scan_source_snapshots(home)


def test_stable_origin_conflict_cannot_be_masked_by_stale_export(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    original = message(message_id='m1')
    changed = message('Edited history, not a new arrival', message_id='m1')
    write_session(home, 's', [original])
    baseline = sources.scan_source_snapshots(home)[0]
    write_session(home, 'stale', [original], session_id='s')
    write_session(home, 's', [changed])
    with pytest.raises(sources.SourceChangedError, match='conflict'):
        sources.scan_source_snapshots(home)
    # Removing the stale export still leaves the same identity with new bytes.
    (home / 'sessions' / 'session_stale.json').unlink()
    edited = sources.scan_source_snapshots(home)[0]
    assert edited['messages'][0]['event_id'] == baseline['messages'][0]['event_id']
    assert sources.message_hash(edited['messages'][0]) != sources.message_hash(baseline['messages'][0])


@pytest.mark.parametrize('flags', [(1, 0), (0, 1)])
def test_distinct_sqlite_rows_need_clone_proof_not_just_equal_body_and_time(tmp_path, flags):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    sources.scan_source_snapshots(home)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'s','user',100,'Useful technical fact',?,?)", flags)
    with pytest.raises(sources.SourceChangedError, match='ambiguous'):
        sources.scan_source_snapshots(home)


def test_real_core_compaction_keeps_observed_input_noop(tmp_path):
    SessionDB = pytest.importorskip('hermes_state', reason='real Hermes core E2E lane requires SessionDB').SessionDB
    sources = api()
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    home.mkdir()
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session('s', 'cli')
        db.append_message('s', 'user', content='Useful real core fact', timestamp=100)
        old = sources.scan_source_snapshots(home)
        ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
        ledger.initialize_legacy(old, {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
        carried = db.get_messages_as_conversation('s')
        db.archive_and_compact('s', [{'role': 'assistant', 'content': 'Compressed context',
            '_compressed_summary': True, 'timestamp': 101}, *carried], tail_count=len(carried))
        current = sources.scan_source_snapshots(home)
        assert current[0]['messages'] == old[0]['messages']
        assert ledger.capture_snapshot(current[0]) == []
    finally:
        db.close()


def test_large_unretained_artifacts_and_sqlite_rows_have_bounded_scan_memory(tmp_path):
    import tracemalloc

    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    ballast = 'unretained tool and prompt ballast ' * 32768
    for i in range(20):
        write_session(home, f'export-{i}', [message(), {'role': 'tool', 'content': ballast}],
                      session_id='s', system_prompt=ballast, tools=[{'description': ballast}])
    with sqlite3.connect(db) as con:
        con.executemany('INSERT INTO messages VALUES (?,\'s\',\'tool\',100,?,1,0)',
                        [(i + 2, ballast) for i in range(20)])
    raw_hashes = [hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted((home / 'sessions').glob('session_*.json'))]
    db_only_home = home / 'profiles' / 'db_only'
    db_only_home.mkdir(parents=True)
    db_only = create_db(db_only_home)
    # Memory must follow retained input, not unrelated tool/prompt history.
    # A generous absolute bound avoids relying on runtime/scheduling speed.
    tracemalloc.start()
    try:
        snapshots = sources.scan_source_snapshots(home)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * 1024 * 1024
    assert {s['profile'] for s in snapshots} == {'default', 'db_only'}
    snapshot = next(s for s in snapshots if s['profile'] == 'default')
    assert snapshot['messages'] == [sources._message(message())]
    assert snapshot['source_hash_before'] == sources.digest(sorted([
        sources.digest(snapshot['messages']), *raw_hashes]))
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(snapshots, {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
    assert [b for s in sources.scan_source_snapshots(home) for b in ledger.capture_snapshot(s)] == []
    with sqlite3.connect(db_only) as con:
        con.execute("INSERT INTO messages VALUES (2,'s','assistant',1,'Late DB-only arrival',1,0)")
    delta = [b for s in sources.scan_source_snapshots(home) for b in ledger.capture_snapshot(s)]
    assert len(delta) == 1 and delta[0]['source_key'].startswith('db_only:hermes:')
    payload = json.loads(Path(delta[0]['payload_path']).read_text())
    assert [m['content'] for m in payload['messages']] == ['Late DB-only arrival']
    assert [b for s in sources.scan_source_snapshots(home) for b in ledger.capture_snapshot(s)] == []


def test_sqlite_skips_unconsumed_payloads_but_still_checks_visible_orphans(tmp_path, monkeypatch):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.executemany('INSERT INTO messages VALUES (?,\'s\',\'tool\',100,?,1,0)',
                        [(i + 2, 'Large unconsumed tool result ' * 32768) for i in range(3)])
        con.execute("INSERT INTO messages VALUES (5,'missing','tool',100,'Hidden orphan',0,0)")
    yielded_roles = []
    real_connect = sqlite3.connect

    class ProbeConnection(sqlite3.Connection):
        def execute(self, query, *args, **kwargs):
            result = super().execute(query, *args, **kwargs)
            if query.startswith('SELECT * FROM messages'):
                def observe():
                    for row in result:
                        yielded_roles.append(row['role'])
                        yield row
                return observe()
            return result

    def connect(*args, **kwargs):
        return real_connect(*args, factory=ProbeConnection, **kwargs)
    monkeypatch.setattr(sources.sqlite3, 'connect', connect)
    assert sources.scan_source_snapshots(home)[0]['messages'] == [sources._message(message())]
    # SQLite must not deserialize full unretained tool rows into Python.
    assert yielded_roles == ['user']
    with real_connect(db) as con:
        con.execute('UPDATE messages SET active=1 WHERE id=5')
    with pytest.raises(sources.SourceChangedError, match='missing session'):
        sources.scan_source_snapshots(home)


def test_secret_material_filtered_not_frozen_as_plaintext(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [message('api_key=secretcredentialvalue'), message('Safe result', 101)])
    snapshots = sources.scan_source_snapshots(home)
    assert [m['content'] for m in snapshots[0]['messages']] == ['Safe result']


def malformed_physical_home(tmp_path):
    home = tmp_path / 'physical'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute("ALTER TABLE messages ADD COLUMN origin_message_id TEXT")
        con.execute("ALTER TABLE messages ADD COLUMN display_order INTEGER")
        con.execute("UPDATE sessions SET parent_session_id='missing-parent' WHERE id='s'")
        con.execute("UPDATE messages SET timestamp=NULL WHERE id=1")
        con.execute("INSERT INTO messages VALUES (9,'orphan','tool',NULL,'private tool body',0,0,NULL,NULL)")
    path = write_session(home, 's', [{'role': 'user', 'content': 'Historical missing time'},
        {'role': 'tool', 'content': 'private tool body'}], parent_session_id='missing-parent',
        system_prompt='private prompt', reasoning='private reasoning')
    return home, db, path


def flattened(snapshots):
    return [m for snapshot in snapshots for m in snapshot['messages']]


@pytest.mark.parametrize('field', ['active', 'compacted', 'display_order'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), float('-inf')])
def test_physical_sql_omitted_mutable_fields_still_reject_nonfinite(field, invalid):
    sources = api()
    row = {'id': 1, 'session_id': 's', 'role': 'tool', 'timestamp': 100,
           'content': 'All immutable source fields stay bound', field: invalid}
    with pytest.raises(ValueError):
        sources._physical_record(row, sqlite_row=True)


def test_physical_sql_exact_emitted_hashes_match_canonical_full_field_definition():
    sources = api()
    for role in ('user', 'assistant', 'system', 'tool', 'other'):
        row = {'id': 1, 'session_id': 's', 'role': role, 'timestamp': 100,
               'content': '\\x00json:' + json.dumps([{'type': 'text', 'text': 'Exact legacy bytes'}]),
               'reasoning': 'Private synthetic reasoning remains hashed', 'api_content': 'Complete synthetic API field',
               'display_identity': b'exact-synthetic-blob', 'origin_message_id': 'origin-A',
               'active': 1, 'compacted': 0, 'display_order': 2}
        immutable = {k: v for k, v in row.items() if k not in sources._MUTABLE_ROW_FIELDS}
        expected = {'id': row['id'], 'session_id': row['session_id'],
                    'row_hash': sources.digest(sources._sql_hash(immutable)),
                    'copy_hash': sources.digest(sources._sql_hash({k: v for k, v in immutable.items() if k not in {'id', 'session_id'}})),
                    'export_hash': sources._raw_export_binding(row, sqlite_content=True),
                    'origin_hash': sources.digest('origin-A')}
        assert sources._physical_record(row, sqlite_row=True) == expected
        assert sources._physical_record(row)['record_hash'] == sources.digest(sources._sql_hash(row))


@pytest.mark.parametrize('lane', ['sqlite', 'json'])
def test_old_copy_cannot_cross_first_non_target_cutover_anchor(tmp_path, lane):
    sources = api()
    home = tmp_path / 'anchor-home'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET parent_session_id='old-b', end_reason='compression' WHERE id='s'")
        con.execute("INSERT INTO sessions VALUES ('old-b',99,'s','compression','cli')")
        con.execute("INSERT INTO messages VALUES (2,'old-b','user',101,'Unknown repeated cross-anchor content',1,0)")
    baseline = sources.capture_physical_baseline(home, bank='anchor-bank')
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO sessions VALUES ('future-x',102,'s',NULL,'cli')")
        if lane == 'sqlite':
            con.execute("INSERT INTO messages VALUES (3,'future-x','user',101,'Unknown repeated cross-anchor content',1,0)")
    if lane == 'json':
        write_session(home, 'future-x', [message('Unknown repeated cross-anchor content', 101)],
                      session_start=102, parent_session_id='s')
    with pytest.raises(sources.SourceChangedError, match='ambiguous legacy copy'):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_old_compression_proof_stops_at_anchor_but_allows_target_and_forward_chain():
    sources = api()
    evidence = {'old-a': {'parent_session_id': 'old-b', 'end_reason': 'compression'},
                'old-b': {'parent_session_id': 'old-a', 'end_reason': 'compression'},
                'future-x': {'parent_session_id': 'old-a', 'end_reason': 'compression'},
                'future-y': {'parent_session_id': 'future-x'}}
    anchors = {'old-a', 'old-b'}
    assert sources._compression_copy('old-a', 'future-y', evidence, cutover_anchors=anchors) is True
    assert sources._compression_copy('future-x', 'future-y', evidence, cutover_anchors=anchors) is True
    assert sources._compression_copy('old-b', 'future-y', evidence, cutover_anchors=anchors) is False


@pytest.mark.parametrize('marker', ['_delegate_from', '_branched_from', '_reset_from'])
@pytest.mark.parametrize('parent_end', [None, 'compression'])
def test_native_independent_parent_marker_does_not_require_active_parent_to_end(tmp_path, marker, parent_end):
    sources = api()
    home = tmp_path / 'native-independent'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute('ALTER TABLE sessions ADD COLUMN model_config TEXT')
        con.execute('ALTER TABLE messages ADD COLUMN origin_message_id TEXT')
    baseline = sources.capture_physical_baseline(home, bank='native-relations')
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET end_reason=? WHERE id='s'", (parent_end,))
        con.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?)',
                    ('new-child', 101, 's', None, 'subagent', json.dumps({marker: 's'})))
        con.execute('INSERT INTO messages VALUES(?,?,?,?,?,1,0,?)',
                    (2, 'new-child', 'user', 1, 'Independent delegated finding with an old event date', 'new-origin'))
    result = sources.scan_source_snapshots(home, physical_baseline=baseline)
    admitted = [s for s in result if s['messages']]
    assert len(admitted) == 1 and admitted[0]['source_key'] == 'default:hermes:new-child'
    assert admitted[0]['messages'][0]['origin'] == 'new-origin'


def test_inherited_delegate_marker_does_not_hide_real_compression_or_missing_proof(tmp_path):
    sources = api()
    home = tmp_path / 'native-inherited'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute('ALTER TABLE sessions ADD COLUMN model_config TEXT')
        con.execute('ALTER TABLE messages ADD COLUMN origin_message_id TEXT')
        con.execute("UPDATE sessions SET source='subagent', model_config=? WHERE id='s'", (json.dumps({'_delegate_from': 'grand-root'}),))
    baseline = sources.capture_physical_baseline(home, bank='native-relations')
    with sqlite3.connect(db) as con:
        con.execute('INSERT INTO sessions VALUES(?,?,?,?,?,?)',
                    ('new-child', 101, 's', None, 'subagent', json.dumps({'_delegate_from': 'grand-root'})))
        con.execute('INSERT INTO messages VALUES(?,?,?,?,?,1,0,?)',
                    (2, 'new-child', 'user', 1, 'A genuinely new compressed child finding', 'new-origin'))
    with pytest.raises(sources.SourceChangedError, match='missing forward lineage'):
        sources.scan_source_snapshots(home, physical_baseline=baseline)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET end_reason='compression' WHERE id='s'")
    result = sources.scan_source_snapshots(home, physical_baseline=baseline)
    admitted = [s for s in result if s['messages']]
    assert len(admitted) == 1 and admitted[0]['source_key'] == 'default:cutover-session:s'


def test_physical_future_clone_proof_stops_at_cyclic_legacy_anchor(tmp_path):
    import subprocess
    sources = api()
    home = tmp_path / 'cycle-home'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET parent_session_id='old-b', end_reason='compression' WHERE id='s'")
        con.execute("INSERT INTO sessions VALUES ('old-b',99,'s','compression','cli')")
    baseline = sources.capture_physical_baseline(home, bank='cycle-bank')
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(baseline))
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO sessions VALUES ('future-x',101,'s',NULL,'cli')")
        con.execute("INSERT INTO sessions VALUES ('future-y',102,'s',NULL,'cli')")
        for row_id, sid in [(2, 'future-x'), (3, 'future-y')]:
            con.execute('INSERT INTO messages VALUES (?,?,?,?,?,1,0)',
                        (row_id, sid, 'user', 101, 'Future duplicate without proved clone identity'))
    code = '''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import hindsight_incremental_sources as source
try:
    source.scan_source_snapshots(Path(sys.argv[2]), physical_baseline=json.loads(Path(sys.argv[3]).read_text()))
except source.SourceChangedError:
    print('BLOCKED_WITHOUT_HANG')
else:
    raise AssertionError('unproved sibling clone accepted')
'''
    result = subprocess.run([sys.executable, '-c', code, str(ROOT), str(home), str(path)],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'BLOCKED_WITHOUT_HANG'


def test_physical_malformed_history_is_finite_noncompletion_private_and_readonly(tmp_path, monkeypatch):
    sources = api()
    home, db, path = malformed_physical_home(tmp_path)
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home)
    before = {p: p.read_bytes() for p in (db, path)}
    # Exclusion must not parse historical logical identity or construct core DB.
    def forbidden(*args, **kwargs):
        raise AssertionError('historical logical identity was traversed')
    monkeypatch.setattr(sources, '_message', forbidden)
    monkeypatch.setattr(sources, '_lineage_root', forbidden)
    baseline = sources.capture_physical_baseline(home, bank='A')
    sources.validate_physical_baseline(baseline, bank='A', hermes_home=home)
    assert baseline['schema_version'] == 'hindsight-physical-legacy-baseline/1'
    assert baseline['status'] == 'legacy_unknown' and baseline['completion_proven'] is False
    assert {p: p.read_bytes() for p in (db, path)} == before
    encoded = json.dumps(baseline)
    assert all(secret not in encoded for secret in ('private tool body', 'private prompt',
        'private reasoning', 'Historical missing time', 'Useful technical fact'))
    snapshots = sources.physical_baseline_snapshots(baseline)
    assert snapshots and not flattened(snapshots)
    assert all(':cutover-session:' in s['source_key'] or ':physical:empty' in s['source_key'] for s in snapshots)
    assert sources.scan_source_snapshots(home, physical_baseline=baseline) == snapshots


def test_physical_profiles_and_home_A_B_A_do_not_discard_db_only_or_legacy_roles(tmp_path):
    sources = api()
    a, db, path = malformed_physical_home(tmp_path)
    b = tmp_path / 'B'
    write_session(b, 's', [message(timestamp=None)])
    db_only = a / 'profiles' / 'db-only'
    db_only.mkdir(parents=True)
    create_db(db_only)
    base_a = sources.capture_physical_baseline(a, bank='A')
    base_b = sources.capture_physical_baseline(b, bank='A')
    first = sources.scan_source_snapshots(a, physical_baseline=base_a)
    assert {s['profile'] for s in first} == {'default', 'db-only'}
    assert len(base_a['profiles'][0]['sqlite']['rows']) == 2
    assert len(base_a['profiles'][0]['artifacts'][0]['records']) == 2
    assert not flattened(sources.scan_source_snapshots(b, physical_baseline=base_b))
    assert sources.scan_source_snapshots(a, physical_baseline=base_a) == first
    with pytest.raises(ValueError):
        sources.validate_physical_baseline(base_a, bank='B', hermes_home=a)
    with pytest.raises(ValueError):
        sources.validate_physical_baseline(base_a, bank='A', hermes_home=b)
    with pytest.raises(ValueError):
        sources.scan_source_snapshots(a, profile_mode='none', physical_baseline=base_a)


@pytest.mark.parametrize('change', ['body', 'timestamp', 'association', 'delete', 'reset', 'replace',
    'json-prefix', 'json-truncate', 'json-delete', 'json-replace', 'json-sid'])
def test_physical_exact_old_bindings_reject_rewrite_reset_truncation_and_epoch(tmp_path, change):
    sources = api()
    home, db, path = malformed_physical_home(tmp_path)
    baseline = sources.capture_physical_baseline(home, bank='A')
    if change in {'body', 'timestamp', 'association', 'delete', 'reset'}:
        sql = {'body': "UPDATE messages SET content='Changed' WHERE id=1",
               'timestamp': 'UPDATE messages SET timestamp=100 WHERE id=1',
               'association': "UPDATE messages SET session_id='other' WHERE id=1",
               'delete': 'DELETE FROM messages WHERE id=9', 'reset': 'DELETE FROM messages'}[change]
        with sqlite3.connect(db) as con:
            con.execute(sql)
    elif change == 'replace':
        db.rename(home / 'old.db')
        create_db(home)
    elif change == 'json-delete':
        path.rename(path.with_suffix('.old'))
    elif change == 'json-replace':
        old = path.read_bytes()
        path.rename(path.with_suffix('.old'))
        path.write_bytes(old)
    else:
        value = json.loads(path.read_text())
        if change == 'json-prefix':
            value['messages'][0]['content'] = 'Changed'
        elif change == 'json-truncate':
            value['messages'].pop()
        else:
            value['session_id'] = 'other'
        path.write_text(json.dumps(value))
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_visibility_changes_are_not_content_rewrites_and_late_origin_arrives_once(tmp_path):
    sources = api()
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    home, db, path = malformed_physical_home(tmp_path)
    baseline = sources.capture_physical_baseline(home, bank='A')
    anchors = sources.physical_baseline_snapshots(baseline)
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(anchors, {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
    with sqlite3.connect(db) as con:
        con.execute('UPDATE messages SET active=0, compacted=1, display_order=77 WHERE id=1')
        con.execute("INSERT INTO messages VALUES (0,'s','user',1,'Useful technical fact',1,0,'new-origin',78)")
    current = sources.scan_source_snapshots(home, physical_baseline=baseline)
    assert [m['origin'] for m in flattened(current)] == ['new-origin']
    delta = [b for s in current for b in ledger.capture_snapshot(s)]
    assert len(delta) == 1
    assert [b for s in sources.scan_source_snapshots(home, physical_baseline=baseline)
            for b in ledger.capture_snapshot(s)] == []


@pytest.mark.parametrize('sid,proof', [('s', False), ('child', False), ('child', True)])
def test_physical_old_no_origin_copy_requires_forward_cross_session_proof(tmp_path, sid, proof):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        if sid == 'child':
            if proof:
                con.execute("UPDATE sessions SET end_reason='compression' WHERE id='s'")
            con.execute("INSERT INTO sessions VALUES ('child',101,'s',NULL,'cli')")
        con.execute('INSERT INTO messages VALUES (2,?,\'user\',100,\'Useful technical fact\',1,0)', (sid,))
    if proof:
        assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))
    else:
        with pytest.raises(sources.SourceChangedError):
            sources.scan_source_snapshots(home, physical_baseline=baseline)


@pytest.mark.parametrize('kind', ['missing-time', 'missing-parent', 'orphan-tool', 'same-body-time'])
def test_physical_future_unknowns_remain_strict_not_automatically_legacy(tmp_path, kind):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        if kind == 'missing-parent':
            con.execute("INSERT INTO sessions VALUES ('child',101,'missing',NULL,'cli')")
            row = (2, 'child', 'user', 101, 'New fact', 1, 0)
        elif kind == 'orphan-tool':
            row = (2, 'missing', 'tool', 101, 'Tool body', 1, 0)
        elif kind == 'same-body-time':
            row = (2, 's', 'user', 100, 'Useful technical fact', 1, 0)
        else:
            row = (2, 's', 'user', None, 'New fact', 1, 0)
        con.execute('INSERT INTO messages VALUES (?,?,?,?,?,?,?)', row)
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_json_prefix_growth_admits_native_late_origin_not_old_text_blacklist(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message(timestamp=None)], parent_session_id='gone')
    baseline = sources.capture_physical_baseline(home, bank='A')
    data = json.loads(path.read_text())
    data['messages'].append(message(timestamp=1, origin_message_id='new-late-origin'))
    path.write_text(json.dumps(data))
    current = sources.scan_source_snapshots(home, physical_baseline=baseline)
    assert [m['origin'] for m in flattened(current)] == ['new-late-origin']
    assert current[0]['source_key'] == sources.physical_baseline_snapshots(baseline)[0]['source_key']


@pytest.mark.parametrize('origin', [None, 'native-old'])
def test_physical_new_late_export_needs_native_origin_or_exact_export_copy_binding(tmp_path, origin):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute('ALTER TABLE messages ADD COLUMN origin_message_id TEXT')
        con.execute('UPDATE messages SET origin_message_id=? WHERE id=1', (origin,))
    baseline = sources.capture_physical_baseline(home, bank='A')
    write_session(home, 's', [message(**({'origin_message_id': origin} if origin else {}))])
    if origin:
        assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))
    else:
        with pytest.raises(sources.SourceChangedError):
            sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_exact_new_export_copy_is_noop_but_rewritten_copy_is_not(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message(timestamp=None)])
    baseline = sources.capture_physical_baseline(home, bank='A')
    copy = path.with_name('session_copy.json')
    copy.write_bytes(path.read_bytes())
    assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))
    data = json.loads(copy.read_text())
    data['messages'][0]['content'] = 'Different unknown history'
    copy.write_text(json.dumps(data))
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_capture_double_read_guard_and_raw_path_controls(tmp_path, monkeypatch):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message(timestamp=None)])
    real = sources._read_bytes
    def mutate(target):
        value = real(target)
        if Path(target) == path:
            path.write_bytes(value + b' ')
        return value
    monkeypatch.setattr(sources, '_read_bytes', mutate)
    with pytest.raises(sources.SourceChangedError):
        sources.capture_physical_baseline(home, bank='A')
    with pytest.raises(sources.SourceChangedError):
        sources.capture_physical_baseline(Path(str(home) + '\n'), bank='A')


def test_physical_old_conflicting_logical_lineage_does_not_block_exact_exclusion(tmp_path):
    sources = api()
    home, db, path = malformed_physical_home(tmp_path)
    value = json.loads(path.read_text())
    value['parent_session_id'] = 'a-different-missing-parent'
    path.write_text(json.dumps(value))
    baseline = sources.capture_physical_baseline(home, bank='A')
    assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (10,'s','assistant',1,'New late fact',1,0,'native-late',0)")
    assert [m['origin'] for m in flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))] == ['native-late']


def test_physical_forward_lineage_conflict_still_blocks_clone_proof(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    write_session(home, 's', [message()], end_reason='closed')
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET end_reason='compression' WHERE id='s'")
        con.execute("INSERT INTO sessions VALUES ('child',101,'s',NULL,'cli')")
        con.execute("INSERT INTO messages VALUES (2,'child','user',100,'Useful technical fact',1,0)")
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_new_json_cross_session_copy_and_real_future_tail_have_current_fence(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [message(timestamp=None)], end_reason='compression')
    baseline = sources.capture_physical_baseline(home, bank='A')
    write_session(home, 'child', [message(timestamp=None), message('Genuinely new late tail', 1)],
                  parent_session_id='s')
    current = sources.scan_source_snapshots(home, physical_baseline=baseline)
    assert [m['content'] for m in flattened(current)] == ['Genuinely new late tail']
    assert current[0]['source_hash_before'] != sources.physical_baseline_snapshots(baseline)[0]['source_hash_before']


@pytest.mark.parametrize('mutation', ['raw-body', 'raw-role', 'raw-reasoning', 'origin', 'session-start'])
def test_physical_old_full_raw_prefix_not_cleaned_text_only(tmp_path, mutation):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message(timestamp=None, reasoning='private old reasoning')])
    baseline = sources.capture_physical_baseline(home, bank='A')
    value = json.loads(path.read_text())
    if mutation == 'session-start':
        value['session_start'] = 200
    else:
        key = {'raw-body': 'content', 'raw-role': 'role', 'raw-reasoning': 'reasoning', 'origin': 'message_id'}[mutation]
        value['messages'][0][key] = 'rewritten'
    path.write_text(json.dumps(value))
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_sqlite_snapshot_streams_all_roles_queryonly_and_no_core_constructor(tmp_path, monkeypatch):
    sources = api()
    home, db, path = malformed_physical_home(tmp_path)
    real_connect = sqlite3.connect
    probes = []
    class ReadOnlyConnection(sqlite3.Connection):
        def execute(self, query, *args, **kwargs):
            probes.append(query)
            result = super().execute(query, *args, **kwargs)
            if query.startswith('SELECT * FROM messages'):
                assert self.in_transaction
                assert super().execute('PRAGMA query_only').fetchone()[0] == 1
                def stream():
                    for row in result:
                        yield row
                return stream()
            return result
    def connect(url, *args, **kwargs):
        assert url.endswith('?mode=ro') and kwargs['uri'] is True
        return real_connect(url, *args, factory=ReadOnlyConnection, **kwargs)
    monkeypatch.setattr(sources.sqlite3, 'connect', connect)
    baseline = sources.capture_physical_baseline(home, bank='A')
    assert len(baseline['profiles'][0]['sqlite']['rows']) == 2
    assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))
    assert sum(q == 'BEGIN' for q in probes) == 2


@pytest.mark.parametrize('change', ['directory', 'profile', 'app-id', 'session-id', 'raw-tool', 'display-identity'])
def test_physical_all_path_and_native_epoch_coverage_stays_fail_closed(tmp_path, change):
    sources = api()
    home, db, path = malformed_physical_home(tmp_path)
    with sqlite3.connect(db) as con:
        con.execute('ALTER TABLE messages ADD COLUMN display_identity BLOB')
        con.execute("UPDATE messages SET display_identity=? WHERE id=1", (b'opaque-display',))
    profile_home = home / 'profiles' / 'db-only'
    profile_home.mkdir(parents=True)
    create_db(profile_home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    if change == 'directory':
        directory = path.parent
        directory.rename(directory.with_name('old-sessions'))
        directory.mkdir()
        path.write_bytes((home / 'old-sessions' / path.name).read_bytes())
    elif change == 'profile':
        profile_home.rename(home / 'removed-profile')
    else:
        with sqlite3.connect(db) as con:
            sql = {'app-id': 'PRAGMA application_id=123', 'session-id': "UPDATE sessions SET started_at=102 WHERE id='s'",
                'raw-tool': "UPDATE messages SET content='changed private tool' WHERE id=9",
                'display-identity': "UPDATE messages SET display_identity=x'00' WHERE id=1"}[change]
            con.execute(sql)
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


@pytest.mark.parametrize('corruption', ['completed', 'raw', 'prefix', 'row-hash', 'outside', 'duplicate', 'empty-profiles'])
def test_physical_descriptor_validation_rejects_policy_raw_material_and_bad_bindings(tmp_path, corruption):
    sources = api()
    home, db, path = malformed_physical_home(tmp_path)
    baseline = sources.capture_physical_baseline(home, bank='A')
    if corruption == 'completed':
        baseline['completion_proven'] = True
    elif corruption == 'raw':
        baseline['profiles'][0]['sqlite']['rows'][0]['content'] = 'Private leaked body'
    elif corruption == 'prefix':
        baseline['profiles'][0]['artifacts'][0]['prefix_hash'] = '0' * 64
    elif corruption == 'row-hash':
        baseline['profiles'][0]['sqlite']['rows'][0]['row_hash'] = 'not-sha256'
    elif corruption == 'outside':
        baseline['profiles'][0]['artifacts'][0]['path'] = str(tmp_path / 'outside.json')
    elif corruption == 'duplicate':
        baseline['profiles'][0]['sqlite']['rows'].append(baseline['profiles'][0]['sqlite']['rows'][0])
    else:
        baseline['profiles'] = []
    with pytest.raises(ValueError):
        sources.validate_physical_baseline(baseline, bank='A', hermes_home=home)


def test_physical_future_typed_content_secret_and_codex_filters_remain_private(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    blocks = [{'type': 'thinking', 'text': 'Private future reasoning'}, {'type': 'text', 'text': 'Real future typed fact'}]
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'s','assistant',1,?,1,0)", ('\x00json:' + json.dumps(blocks),))
        con.execute("INSERT INTO messages VALUES (3,'s','user',2,'api_key=do-not-freeze-me',1,0)")
    write_session(home, 'codex-new', [message('# AGENTS.md instructions unretained', 2),
        message('Codex future fact', 1, origin_message_id='codex-native')], platform='backfill-codex')
    assert {m['content'] for m in flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))} == {
        'Real future typed fact', 'Codex future fact'}


def test_physical_new_same_sid_export_with_rewritten_valid_identity_is_not_new_arrival(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    path = write_session(home, 's', [message()])
    baseline = sources.capture_physical_baseline(home, bank='A')
    write_session(home, 'late-copy', [message('Rewritten old export', 101)], session_id='s')
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_in_place_identical_schema_recreation_is_still_reset(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        con.executescript('''DROP TABLE messages;
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT,
                timestamp REAL, content TEXT, active INTEGER, compacted INTEGER);
            INSERT INTO messages VALUES (1,'s','user',100,'Useful technical fact',1,0);''')
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_verified_clone_of_legacy_invalid_codec_can_remain_excluded(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute('UPDATE messages SET content=? WHERE id=1', ('\x00json:{old-malformed',))
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET end_reason='compression' WHERE id='s'")
        con.execute("INSERT INTO sessions VALUES ('child',101,'s',NULL,'cli')")
        con.execute("INSERT INTO messages SELECT 2,'child',role,timestamp,content,active,compacted FROM messages WHERE id=1")
    assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))


def test_physical_json_clone_requires_exact_raw_record_not_cleaned_tuple_only(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [message(reasoning='old-private')], end_reason='compression')
    baseline = sources.capture_physical_baseline(home, bank='A')
    write_session(home, 'child', [message(reasoning='different-private')], parent_session_id='s')
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_new_profile_is_strict_and_not_discarded(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    write_session(home / 'profiles' / 'new-profile', 'fresh', [message('A new profile fact', 1)])
    current = sources.scan_source_snapshots(home, physical_baseline=baseline)
    assert {s['profile'] for s in current} == {'default', 'new-profile'}
    assert [m['content'] for m in flattened(current)] == ['A new profile fact']


def test_physical_retained_future_tail_survives_cross_sid_compaction_noop(tmp_path):
    sources = api()
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(sources.physical_baseline_snapshots(baseline), {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'s','user',101,'New after cutover',1,0)")
    assert len([b for s in sources.scan_source_snapshots(home, physical_baseline=baseline) for b in ledger.capture_snapshot(s)]) == 1
    with sqlite3.connect(db) as con:
        con.execute("UPDATE sessions SET end_reason='compression' WHERE id='s'")
        con.execute("INSERT INTO sessions VALUES ('child',102,'s',NULL,'cli')")
        con.execute("INSERT INTO messages SELECT 3,'child',role,timestamp,content,active,compacted FROM messages WHERE id=1")
        con.execute("INSERT INTO messages SELECT 4,'child',role,timestamp,content,active,compacted FROM messages WHERE id=2")
    current = sources.scan_source_snapshots(home, physical_baseline=baseline)
    assert [m['content'] for m in flattened(current)] == ['New after cutover']
    assert [b for s in current for b in ledger.capture_snapshot(s)] == []


@pytest.mark.parametrize('field', ['profile-mode', 'columns'])
def test_physical_bad_descriptor_types_raise_validation_error(tmp_path, field):
    sources = api()
    home = tmp_path / 'home'
    create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    if field == 'profile-mode':
        baseline['profile_mode'] = []
    else:
        baseline['profiles'][0]['sqlite']['message_columns'] = [[]]
    with pytest.raises(ValueError):
        sources.validate_physical_baseline(baseline, bank='A', hermes_home=home)


def test_physical_new_self_parent_is_invalid_even_with_noncompression_end_reason(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO sessions VALUES ('self',101,'self','closed','cli')")
        con.execute("INSERT INTO messages VALUES (2,'self','user',101,'Not a provable lineage',1,0)")
    with pytest.raises(sources.SourceChangedError):
        sources.scan_source_snapshots(home, physical_baseline=baseline)


def test_physical_same_text_time_distinct_new_origins_are_all_admitted(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    with sqlite3.connect(db) as con:
        con.execute('ALTER TABLE messages ADD COLUMN origin_message_id TEXT')
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (0,'s','user',100,'Useful technical fact',1,0,'origin-zero')")
        con.execute("INSERT INTO messages VALUES (2,'s','user',100,'Useful technical fact',1,0,'origin-two')")
    assert {m['origin'] for m in flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))} == {
        'origin-zero', 'origin-two'}


def test_physical_empty_anchor_keeps_feed_coverage_when_first_json_arrives(tmp_path):
    sources = api()
    ledger_module = importlib.import_module('hindsight_incremental_ledger')
    home = tmp_path / 'home'
    home.mkdir()
    baseline = sources.capture_physical_baseline(home, bank='A')
    ledger = ledger_module.IncrementalLedger(tmp_path / 'ledger', 'A', create=True)
    ledger.initialize_legacy(sources.physical_baseline_snapshots(baseline), {}, [], confirm=ledger_module.CONFIRM_INITIALIZE)
    assert ledger.list_batches() == []
    write_session(home, 'first', [message('First finite arrival', 1)])
    assert len([b for s in sources.scan_source_snapshots(home, physical_baseline=baseline) for b in ledger.capture_snapshot(s)]) == 1
    assert [b for s in sources.scan_source_snapshots(home, physical_baseline=baseline) for b in ledger.capture_snapshot(s)] == []


def test_physical_capture_and_rescan_do_not_retain_unconsumed_history_bodies(tmp_path):
    import tracemalloc
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    ballast = 'unretained private payload ' * 32768
    for i in range(12):
        write_session(home, f'large-{i}', [message(timestamp=None), {'role': 'tool', 'content': ballast}],
                      session_id='s', system_prompt=ballast, reasoning=ballast)
    with sqlite3.connect(db) as con:
        con.executemany("INSERT INTO messages VALUES (?,'s','tool',NULL,?,1,0)", [(i + 2, ballast) for i in range(12)])
    tracemalloc.start()
    try:
        baseline = sources.capture_physical_baseline(home, bank='A')
        current = sources.scan_source_snapshots(home, physical_baseline=baseline)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert not flattened(current)
    assert len(baseline['profiles'][0]['sqlite']['rows']) == 13
    assert len(baseline['profiles'][0]['artifacts']) == 12
    assert peak < 16 * 1024 * 1024


def test_physical_future_unretained_payloads_skip_codec_but_all_role_orphans_block(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    db = create_db(home)
    baseline = sources.capture_physical_baseline(home, bank='A')
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'s','tool',NULL,?,1,0)", ('\x00json:{not-message-codec',))
    assert not flattened(sources.scan_source_snapshots(home, physical_baseline=baseline))
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (3,'missing','tool',NULL,'Hidden future orphan',0,0)")
    with pytest.raises(sources.SourceChangedError, match='missing session'):
        sources.scan_source_snapshots(home, physical_baseline=baseline)
