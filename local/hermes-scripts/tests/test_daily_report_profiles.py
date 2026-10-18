import importlib.util
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name='daily_report'):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'daily_report.py')
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


def test_hermes_model_usage_prefers_per_call_agent_log_over_startup_model(tmp_path):
    mod = load_module('daily_report_per_call_model')
    start = datetime(2026, 5, 18, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc)
    inside_window = datetime(2026, 5, 18, 3, 0, tzinfo=timezone.utc).timestamp()

    mod.HERMES_HOME = tmp_path
    mod.STATE_DB = tmp_path / 'state.db'
    mod.PROFILE_DIR = tmp_path / 'profiles'
    make_state_db(
        mod.STATE_DB,
        session_id='default-session',
        model='startup-model',
        msg_ts=inside_window,
        started_at=inside_window,
        calls=1,
        input_tokens=999,
    )
    agent_log = tmp_path / 'logs' / 'agent.log'
    agent_log.parent.mkdir()
    agent_log.write_text(
        '2026-05-18 03:00:00 [abc123] API call #1: '
        'model=actual-per-call-model provider=openai in=100 out=20 '
        'total=120 latency=0.1s cache=30/0\n',
        encoding='utf-8',
    )

    assert mod.hermes_model_usage(start, end) == [{
        'profile': 'default',
        'model': 'actual-per-call-model',
        'sessions': 1,
        'turns': 0,
        'calls': 1,
        'input_tokens': 100,
        'cache_read_tokens': 30,
        'cache_write_tokens': 0,
        'output_tokens': 20,
    }]


def test_mental_model_rows_report_exact_readiness_and_window_change(tmp_path):
    mod = load_module('daily_report_models')
    registry = {
        'models': {
            'egomotion4d-static-surface': {
                'active_slot': 'b',
                'last_verdict': 'PASS_PUBLISH',
                'verdict_detail': 'quality=95',
                'source_evidence_sha': 'e' * 64,
                'accepted_revision': {
                    'slot': 'b',
                    'content_sha': 'c' * 64,
                    'source_evidence_sha': 'e' * 64,
                    'accepted_at': '2026-07-21T01:00:00Z',
                },
            },
            'egomotion4d-dynamic-actor': {
                'active_slot': 'a',
                'last_verdict': 'REJECT',
                'source_evidence_sha': 'f' * 64,
            },
        }
    }
    path = tmp_path / 'registry.json'
    path.write_text(json.dumps(registry), encoding='utf-8')
    start = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)

    rows = mod.mental_model_rows(start, end, registry_path=path)

    by_id = {row['logical_id']: row for row in rows}
    assert by_id['egomotion4d-static-surface']['state'] == 'READY'
    assert by_id['egomotion4d-static-surface']['change'] == 'PUBLISHED'
    assert by_id['egomotion4d-static-surface']['revision'] == 'c' * 12
    assert by_id['egomotion4d-static-surface']['quality'] == '95'
    assert by_id['egomotion4d-dynamic-actor']['state'] == 'REJECT'
    assert by_id['egomotion4d-dynamic-actor']['change'] == 'NO_ACCEPTED_REVISION'


def test_operation_summary_excludes_historical_terminal_queue_rows():
    mod = load_module('daily_report_operations')
    window_ops = [
        ['consolidation', 'completed', '7'],
        ['refresh_mental_model', 'completed', '25'],
        ['refresh_mental_model', 'failed', '1'],
    ]
    all_time_queue = [
        ['completed', 'retain', '17694'],
        ['cancelled', 'retain', '21'],
        ['failed', 'retain', '3'],
        ['processing', 'consolidation', '1'],
    ]

    summary = mod.window_operation_summary(window_ops)
    alerts = mod.current_operation_alerts(window_ops, all_time_queue)

    assert 'refresh_mental_model=25 ok/1 failed' in summary
    assert alerts == ['active queue: consolidation processing x1']


def test_load_research_digest_hides_generation_metadata(tmp_path):
    mod = load_module('daily_report_digest')
    mod.RESEARCH_DIGEST_DIR = tmp_path
    (tmp_path / '2026-07-20.md').write_text(
        '<!--\nllm: openai/model\nbase_url: http://oneapi/v1\n-->\n\n# Digest\nbody\n',
        encoding='utf-8',
    )

    assert mod.load_research_digest('2026-07-20') == '# Digest\nbody'


def test_parse_prev_report_accepts_compact_daily_header(tmp_path):
    mod = load_module('daily_report_prev_compact')
    mod.WIKI_DAILY_DIR = tmp_path
    (tmp_path / '2026-07-21.md').write_text(
        '概要: Documents=27,886 (Δ+29)  Observations=22,065 (Δ+136)  '
        'unconsolidated=0  failed_base=0\n'
        'Consolidation: base_done=26  remaining=0  failed=0\n',
        encoding='utf-8',
    )

    assert mod.parse_prev_report('2026-07-21') == {
        'documents': 27886,
        'observations': 22065,
        'consolidated': 26,
        'unconsolidated': 0,
    }


def test_pitfall_summary_uses_lifecycle_timestamps_not_coarse_date(tmp_path):
    mod = load_module('daily_report_pitfalls')
    root = tmp_path / 'mental-models' / 'egomotion4d'
    root.mkdir(parents=True)
    mod.MENTAL_MODEL_ROOT = root
    (root / 'pitfall_index.json').write_text(
        json.dumps(
            {
                'entries': [
                    {
                        'p_id': 'P1',
                        'status': 'candidate',
                        'title': 'timestamped',
                        'date': '2026-07-21',
                        'created_at': '2026-07-21T01:00:00Z',
                    },
                    {
                        'p_id': 'P2',
                        'status': 'candidate',
                        'title': 'coarse legacy date only',
                        'date': '2026-07-21',
                    },
                ]
            }
        ),
        encoding='utf-8',
    )
    start = datetime(2026, 7, 21, 0, 0, tzinfo=timezone.utc)
    end = datetime(2026, 7, 22, 0, 0, tzinfo=timezone.utc)

    summary = mod.pitfall_summary(start, end)

    assert [entry['p_id'] for entry in summary['changes']] == ['P1']
