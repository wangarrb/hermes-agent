import importlib.util
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name='daily_report'):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def make_state_db(path: Path, *, session_id: str, model: str, msg_ts: float, started_at: float, calls: int, input_tokens: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        'CREATE TABLE sessions (id TEXT PRIMARY KEY, model TEXT, message_count INTEGER, api_call_count INTEGER, input_tokens INTEGER, cache_read_tokens INTEGER, cache_write_tokens INTEGER, output_tokens INTEGER, started_at REAL)'
    )
    con.execute('CREATE TABLE messages (session_id TEXT, timestamp REAL)')
    con.execute(
        'INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (session_id, model, 7, calls, input_tokens, 11, 13, 17, started_at),
    )
    con.execute('INSERT INTO messages VALUES (?, ?)', (session_id, msg_ts))
    con.commit()
    con.close()


def test_hermes_model_usage_aggregates_default_and_profile_state_dbs(tmp_path):
    mod = load_module()
    start = datetime(2026, 5, 18, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
    before_window = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc).timestamp()
    inside_window = datetime(2026, 5, 18, 3, 0, tzinfo=timezone.utc).timestamp()

    mod.STATE_DB = tmp_path / 'state.db'
    mod.PROFILE_DIR = tmp_path / 'profiles'
    make_state_db(mod.STATE_DB, session_id='default-session', model='main-model', msg_ts=inside_window, started_at=before_window, calls=2, input_tokens=100)
    make_state_db(mod.PROFILE_DIR / 'planner' / 'state.db', session_id='planner-session', model='kanban-model', msg_ts=inside_window, started_at=before_window, calls=5, input_tokens=300)

    rows = mod.hermes_model_usage(start, end)

    by_profile = {r['profile']: r for r in rows}
    assert set(by_profile) == {'default', 'planner'}
    assert by_profile['default']['model'] == 'main-model'
    assert by_profile['default']['calls'] == 2
    assert by_profile['planner']['model'] == 'kanban-model'
    assert by_profile['planner']['calls'] == 5
    assert by_profile['planner']['input_tokens'] == 300
