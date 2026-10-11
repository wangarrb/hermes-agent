from pathlib import Path
import importlib
import json
import sqlite3
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    sys.path.insert(0, str(ROOT))
    return importlib.import_module('hindsight_incremental_monitor')


def test_uninitialized_monitor_is_readonly_and_does_not_claim_legacy_drain(tmp_path):
    module = load()
    cfg = {'hermes_home': str(tmp_path / 'home'), 'bank': 'hermes', 'api_url': 'http://test.invalid'}
    calls = []
    def read(path):
        calls.append(path)
        if path == '/health':
            return {'status': 'healthy', 'database': 'connected'}
        if path.endswith('stats'):
            return {'pending_operations': 0, 'pending_consolidation': 56, 'operations_by_status': {'completed': 10}}
        return {'config': {'enable_observations': False, 'enable_auto_consolidation': True}}
    report = module.collect(cfg, request=read, process_reader=lambda: [], now=lambda: 1000)
    assert report['ledger']['initialized'] is False
    assert report['backend']['pending_consolidation'] == 56
    assert 'legacy_backlog' in report['alerts']
    assert len(calls) == 3
    assert not Path(cfg['hermes_home']).exists()


def test_monitor_never_emits_content_secrets_or_lease_token(tmp_path):
    module = load()
    bank_dir = tmp_path / 'state' / 'hermes'
    bank_dir.mkdir(parents=True)
    con = sqlite3.connect(bank_dir / 'ledger.sqlite3')
    con.execute('CREATE TABLE batches (batch_id TEXT PRIMARY KEY,state TEXT,state_hash TEXT)')
    con.execute('INSERT INTO batches VALUES (?,?,?)', ('b1', json.dumps({'phase': 'unknown',
        'kind': 'source', 'next_retry_at': 900, 'content': 'private conversation content',
        'api_key': 'never emit this', 'last_error_class': 'transport'}), 'unused'))
    con.commit(); con.close()
    (bank_dir / 'runtime-window.json').write_text(json.dumps({'status': 'cleanup_pending',
        'owner': {'pid': 100, 'token': 'private-fencing-token'}, 'legacy_pending': 56}))
    cfg = {'hermes_home': str(tmp_path / 'home'), 'bank': 'hermes', 'incremental': {'root': str(tmp_path / 'state')}}
    def read(path):
        return {'status': 'healthy', 'config': {'enable_observations': True}, 'pending_consolidation': 56}
    report = module.collect(cfg, request=read, process_reader=lambda: [], now=lambda: 1000)
    rendered = json.dumps(report)
    assert 'private conversation' not in rendered and 'never emit' not in rendered
    assert 'fencing-token' not in rendered
    assert report['ledger']['phase_counts']['unknown'] == 1
    assert report['ledger']['retry_classes'] == {'transport': 1}
    assert report['ledger']['next_retry_at'] == 900
    assert 'cleanup_pending' in report['alerts']


def test_backend_timeout_is_a_status_not_a_fabricated_healthy_result(tmp_path):
    module = load()
    cfg = {'hermes_home': str(tmp_path / 'home'), 'bank': 'hermes'}
    def read(_path):
        raise TimeoutError('secret upstream body should not be emitted')
    report = module.collect(cfg, request=read, process_reader=lambda: [], now=lambda: 1000)
    assert report['backend']['healthy'] is False
    assert report['backend']['pending_consolidation'] is None
    assert 'backend_unavailable' in report['alerts']
    assert 'secret upstream' not in json.dumps(report)


def test_monitor_rejects_malformed_bank_instead_of_repairing(tmp_path):
    module = load()
    with pytest.raises(ValueError):
        module.collect({'hermes_home': str(tmp_path), 'bank': '../wrong'}, request=lambda _: {})


def test_retry_diagnostics_are_content_free_even_for_corrupt_state(tmp_path):
    module = load()
    root = tmp_path / 'state' / 'hermes'
    root.mkdir(parents=True)
    con = sqlite3.connect(root / 'ledger.sqlite3')
    con.execute('CREATE TABLE batches (batch_id TEXT PRIMARY KEY,state TEXT,state_hash TEXT)')
    rows = [dict(phase='generated', kind='daily_summary', next_retry_at=2000, last_error_class='http_502'),
            dict(phase='private-phase-content', kind='private-kind-content', next_retry_at=3000,
                 last_error_class='private-error-content')]
    for i, row in enumerate(rows):
        con.execute('INSERT INTO batches VALUES (?,?,?)', (str(i), json.dumps(row), 'unused'))
    con.commit(); con.close()
    report = module.collect({'bank': 'hermes', 'hermes_home': str(tmp_path),
        'incremental': {'root': str(tmp_path / 'state')}}, request=lambda _: {},
        process_reader=lambda: [], now=lambda: 1000)
    assert report['ledger']['due_batches'] == 0
    assert report['ledger']['retry_classes'] == {'http_502': 1, 'unrecognized': 1}
    assert report['ledger']['next_retry_at'] == 2000
    assert 'private-' not in json.dumps(report)
