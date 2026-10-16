"""Incremental scan: unchanged sources are never read; changed sources keep
the full strict path. A differential test proves incremental output equals
full-scan output for the same state.
"""
import importlib
import json
import sqlite3
import sys
from pathlib import Path

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


@pytest.fixture
def fixture(tmp_path):
    src = importlib.import_module('hindsight_incremental_sources')
    home = tmp_path / 'home'
    db = database(home)
    (home / 'sessions').mkdir(exist_ok=True)
    baseline = src.capture_physical_baseline(home, bank='fixture')
    return src, home, db, baseline


def spy_reader(src, monkeypatch):
    calls = []
    original = src._read_physical_db

    def spy(path, **kwargs):
        calls.append({'path': str(path), 'row_filter': kwargs.get('row_filter')})
        return original(path, **kwargs)

    monkeypatch.setattr(src, '_read_physical_db', spy)
    return calls


def test_unchanged_sources_are_never_read(fixture, monkeypatch):
    src, home, db, baseline = fixture
    calls = spy_reader(src, monkeypatch)
    first = src.scan_source_snapshots(home, physical_baseline=baseline)
    assert calls and calls[-1]['row_filter'] is None  # seed scan reads fully
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    before = len(calls)
    second = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=current)
    assert second == []
    assert len(calls) == before  # unchanged profile: no db read at all
    assert first  # the seed still produced the baseline snapshots


def test_changed_source_reads_only_its_rows(fixture, monkeypatch):
    src, home, db, baseline = fixture
    seed = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'native','assistant',101,"
                    "'New incremental fact',1,0,'new-event',x'02')")
    calls = spy_reader(src, monkeypatch)
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    snapshots = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=seed)
    assert calls and calls[-1]['row_filter'] == frozenset({'native'})
    messages = [m for s in snapshots for m in s['messages']]
    assert {m['origin'] for m in messages} == {'new-event'}


def test_differential_incremental_matches_full(fixture, monkeypatch):
    src, home, db, baseline = fixture
    seed = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (2,'native','assistant',101,"
                    "'New incremental fact',1,0,'new-event',x'02')")
    full = src.scan_source_snapshots(home, physical_baseline=baseline)
    monkeypatch.undo()
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    incremental = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=seed)
    full_by_key = {s['source_key']: s['messages'] for s in full}
    inc_by_key = {s['source_key']: s['messages'] for s in incremental}
    assert inc_by_key
    for key, messages in inc_by_key.items():
        assert messages == full_by_key[key]


def test_absent_store_seeds_with_full_profile(fixture, monkeypatch):
    src, home, db, baseline = fixture
    calls = spy_reader(src, monkeypatch)
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    snapshots = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=None)
    assert calls and calls[-1]['row_filter'] is None  # seed reads fully
    assert snapshots  # baseline snapshots produced


def test_json_view_change_forces_profile_read(fixture, monkeypatch):
    src, home, db, baseline = fixture
    seed = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    directory = home / 'sessions'
    directory.mkdir(exist_ok=True)
    (directory / 'session_json-late.json').write_text(json.dumps(
        {'session_id': 'json-late', 'session_start': 200,
         'messages': [{'role': 'user', 'timestamp': 200, 'content': 'Late view fact',
                       'origin_message_id': 'late-json'}]}))
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    calls = spy_reader(src, monkeypatch)
    snapshots = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=seed)
    # F7: only the changed view's session is dirty, not the whole profile.
    assert calls and calls[-1]['row_filter'] == frozenset({'json-late'})
    messages = [m for s in snapshots for m in s['messages']]
    assert any(m['origin'] == 'late-json' for m in messages)


def test_ledger_scan_store_roundtrip(tmp_path):
    ledger_mod = importlib.import_module('hindsight_incremental_ledger')
    ledger = ledger_mod.IncrementalLedger(tmp_path / 'ledger', 'fixture', create=True)
    assert ledger.load_scan_fingerprints() is None
    profiles = {'default': {'db': [1, 2, 3, 4], 'sessions': {'native': [1, 1, 'x']},
                            'artifacts': {}, 'scope_sids': ['native']}}
    ledger.save_scan_fingerprints(profiles)
    assert ledger.load_scan_fingerprints() == profiles
    (ledger.bank_dir / 'scan-fingerprints.json').write_text('{not json')
    assert ledger.load_scan_fingerprints() is None  # invalid store: seed run


def test_compression_successor_without_export_matches_full(fixture, monkeypatch):
    # Two post-baseline sessions in one compression chain, no JSON exports:
    # the closure must come from the sessions table, not artifact-only lineage.
    src, home, db, baseline = fixture
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO sessions VALUES ('chain-a',200,NULL,'compression','cli',NULL)")
        con.execute("INSERT INTO sessions VALUES ('chain-b',300,'chain-a',NULL,'cli',NULL)")
        con.execute("INSERT INTO messages VALUES (2,'chain-a','user',200,'Chain A fact',1,0,'chain-a-e1',x'02')")
        con.execute("INSERT INTO messages VALUES (3,'chain-b','user',300,'Chain B fact',1,0,'chain-b-e1',x'03')")
    seed = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO messages VALUES (4,'chain-b','assistant',301,"
                    "'Chain B new fact',1,0,'chain-b-e2',x'04')")
    full = src.scan_source_snapshots(home, physical_baseline=baseline)
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    incremental = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=seed)
    full_by_key = {s['source_key']: s['messages'] for s in full}
    inc_by_key = {s['source_key']: s['messages'] for s in incremental}
    assert inc_by_key  # the dirty chain must produce output
    for key, messages in inc_by_key.items():
        assert key in full_by_key, key
        assert messages == full_by_key[key], key


def test_vanished_session_is_reported_for_fail_closed_capture(fixture):
    src, home, db, baseline = fixture
    with sqlite3.connect(db) as con:
        con.execute("INSERT INTO sessions VALUES ('post-baseline',400,NULL,NULL,'cli',NULL)")
        con.execute("INSERT INTO messages VALUES (2,'post-baseline','user',400,'Ephemeral fact',1,0,'eph-e1',x'02')")
    seed = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    for entry in seed.values():  # the pipeline stores the full scope
        entry['scope_sids'] = sorted(entry['sessions'].keys())
    with sqlite3.connect(db) as con:
        con.execute("DELETE FROM messages WHERE session_id='post-baseline'")
        con.execute("DELETE FROM sessions WHERE id='post-baseline'")
    current = src.compute_scan_fingerprints(home, physical_baseline=baseline)
    snapshots = src.scan_source_snapshots(home, physical_baseline=baseline,
        scan_fingerprints=current, scan_previous=seed)
    vanished = [s for s in snapshots if s['source_key'].endswith('post-baseline')]
    assert vanished and not vanished[0]['messages']  # empty -> ledger capture fails closed
