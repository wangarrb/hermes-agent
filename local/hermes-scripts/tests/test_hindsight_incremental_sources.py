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


def message(text='Useful technical fact', timestamp=100, **extra):
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


def test_secret_material_filtered_not_frozen_as_plaintext(tmp_path):
    sources = api()
    home = tmp_path / 'home'
    write_session(home, 's', [message('api_key=secretcredentialvalue'), message('Safe result', 101)])
    snapshots = sources.scan_source_snapshots(home)
    assert [m['content'] for m in snapshots[0]['messages']] == ['Safe result']
