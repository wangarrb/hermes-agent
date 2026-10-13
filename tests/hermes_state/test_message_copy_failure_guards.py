"""A copy receipt authenticates a newly inserted target, not lastrowid alone."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from hermes_state import SessionDB

KEY = '_message_copy_receipts_v1'

@pytest.fixture
def native_store(tmp_path, monkeypatch):
    import hermes_state
    pragmas = hermes_state.apply_database_pragmas
    def fixture_pragmas(conn, *args, **kwargs):
        assert all(not x[2] or Path(x[2]).is_relative_to(tmp_path) for x in conn.execute('PRAGMA database_list'))
        pragmas(conn, *args, **kwargs)
        conn.execute('PRAGMA synchronous=OFF')
    monkeypatch.setattr(hermes_state, 'apply_database_pragmas', fixture_pragmas)
    db = SessionDB(db_path=tmp_path/'state.db')
    db.create_session('s', 'cli')
    yield db
    db.close()

def snapshot(db):
    return ([dict(r) for r in db._conn.execute('SELECT * FROM messages ORDER BY id')],
            [dict(r) for r in db._conn.execute('SELECT * FROM sessions ORDER BY id')])

@pytest.mark.parametrize('path', ['carried_insert', 'watermark_clone'])
def test_ignored_native_insert_rolls_back_instead_of_attesting_preexisting_target(native_store, path):
    db = native_store
    first = db.append_message('s', 'user', content='Same exact event fields', timestamp=100)
    db.append_message('s', 'user', content='Same exact event fields', timestamp=100)
    carried = next(m for m in db.get_messages('s') if m['id'] == first)
    carried['_row_id'] = first
    with db._conn:
        db._conn.execute("CREATE TRIGGER synthetic_ignore BEFORE INSERT ON messages WHEN NEW.role='user' BEGIN SELECT RAISE(IGNORE); END")
    before = snapshot(db)
    original = copy.deepcopy(carried)
    with pytest.raises(ValueError):
        if path == 'carried_insert':
            db.archive_and_compact('s', [carried])
        else:
            db.archive_and_compact('s', [{'role':'assistant','content':'Synthetic carrier','timestamp':101,'_compressed_summary':True}], watermark=first)
    assert snapshot(db) == before
    assert carried == original
    cfg = json.loads(db.get_session('s')['model_config'] or '{}')
    assert KEY not in cfg

def test_nil_uuid_with_valid_recomputed_receipt_digest_is_rejected(native_store):
    db = native_store
    mid = db.append_message('s', 'user', content='Real writer receipt event', timestamp=100)
    carried = next(m for m in db.get_messages('s') if m['id'] == mid)
    carried['_row_id'] = mid
    db.archive_and_compact('s', [carried])
    cfg = json.loads(db.get_session('s')['model_config'])
    operation = cfg[KEY]['operations'][0]
    operation['operation_id'] = '00000000-0000-0000-0000-000000000000'
    payload = {k:v for k,v in operation.items() if k!='receipt_sha256'}
    operation['receipt_sha256'] = hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with db._conn:
        db._conn.execute('UPDATE sessions SET model_config=? WHERE id=?',(json.dumps(cfg),'s'))
    before = snapshot(db)
    with pytest.raises(ValueError):
        db.update_session_meta('s', '{}')
    assert snapshot(db) == before
