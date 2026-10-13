"""Owned copy receipts are nonportable session runtime metadata, not model config."""
import json
from pathlib import Path

import pytest
from hermes_state import SessionDB

KEY = '_message_copy_receipts_v1'

@pytest.fixture
def native_copy_parent(tmp_path, monkeypatch):
    import hermes_state
    original = hermes_state.apply_database_pragmas
    def private_pragmas(conn, *args, **kwargs):
        assert all(not r[2] or Path(r[2]).is_relative_to(tmp_path) for r in conn.execute('PRAGMA database_list'))
        original(conn, *args, **kwargs)
        conn.execute('PRAGMA synchronous=OFF')
    monkeypatch.setattr(hermes_state, 'apply_database_pragmas', private_pragmas)
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('parent', 'cli', model_config={'model': 'synthetic-owner'})
    old_id = db.append_message('parent', 'user', content='Immutable original input', timestamp=100)
    carry = next(m for m in db.get_messages('parent') if m['id'] == old_id)
    carry['_row_id'] = old_id
    db.archive_and_compact('parent', [
        {'role': 'assistant', 'content': 'Synthetic carrier', 'timestamp': 101, '_compressed_summary': True}, carry
    ], tail_count=1)
    config = json.loads(db.get_session('parent')['model_config'])
    assert KEY in config
    yield db, config
    db.close()

@pytest.mark.parametrize('kind', ['compression', 'delegation', 'branch', 'reset'])
def test_native_child_does_not_inherit_parent_owned_copy_receipts(native_copy_parent, kind):
    db, parent_config = native_copy_parent
    child_config = dict(parent_config)
    if kind == 'compression':
        db.publish_compression_child(parent_session_id='parent', child_session_id='child',
            source='cli', model_config=child_config, require_compression_lease=False,
            messages=[{'role': 'user', 'content': 'New child input', 'timestamp': 102}])
    else:
        marker = {'delegation': '_delegate_from', 'branch': '_branched_from', 'reset': '_reset_from'}[kind]
        child_config[marker] = 'parent'
        db.create_session('child', 'subagent' if kind == 'delegation' else 'cli',
            parent_session_id='parent', model_config=child_config)
    child = json.loads(db.get_session('child')['model_config'] or '{}')
    assert KEY not in child, 'parent receipt cannot authorize any row in a different native session'
    assert child['model'] == 'synthetic-owner'
    assert json.loads(db.get_session('parent')['model_config'])[KEY] == parent_config[KEY]

@pytest.mark.parametrize('kind', ['compression', 'delegation'])
def test_native_child_rejects_forged_receipt_without_changing_parent(native_copy_parent, kind):
    db, parent_config = native_copy_parent
    incoming = json.loads(json.dumps(parent_config))
    incoming[KEY]['operations'][0]['copies'][0]['target_row_hash'] = 'a' * 64
    before = db.get_session('parent')['model_config']
    with pytest.raises(ValueError):
        if kind == 'compression':
            db.publish_compression_child(parent_session_id='parent', child_session_id='child',
                source='cli', model_config=incoming, require_compression_lease=False,
                messages=[{'role': 'user', 'content': 'New child input', 'timestamp': 102}])
        else:
            incoming['_delegate_from'] = 'parent'
            db.create_session('child', 'subagent', parent_session_id='parent', model_config=incoming)
    assert db.get_session('child') is None
    assert db.get_session('parent')['model_config'] == before
    assert db.get_session('parent')['end_reason'] is None
