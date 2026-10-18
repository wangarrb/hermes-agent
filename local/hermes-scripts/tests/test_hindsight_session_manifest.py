import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_stable_session_id_prefers_session_file_stem_when_embedded_id_is_stale():
    manifest = load_module('hindsight_session_manifest')
    path = Path('/tmp/session_20260425_221657_918c61.json')
    session_data = {'session_id': '20260425_191816_90f637'}
    assert manifest.stable_session_id(path, session_data) == '20260425_221657_918c61'


def test_stable_session_id_uses_embedded_id_when_filename_is_not_session_pattern():
    manifest = load_module('hindsight_session_manifest')
    path = Path('/tmp/manual_export.json')
    session_data = {'session_id': 'embedded-id'}
    assert manifest.stable_session_id(path, session_data) == 'embedded-id'


def test_cron_sessions_are_skipped_from_production(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    path = tmp_path / 'session_cron_job_123.json'
    path.write_text(
        '{"session_id":"cron_job_123","platform":"cron","messages":[{"role":"user","content":"重要的 Hindsight pipeline 操作报告，包含 project:hindsight"}]}',
        encoding='utf-8',
    )
    records = manifest.records_from_json_file(path, bank_target='hermes')
    assert len(records) == 1
    assert records[0]['action'] == 'skip'
    assert records[0]['reason'] == 'automated_cron_session'


def test_recent_session_files_are_not_yielded(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    recent = tmp_path / 'session_recent.json'
    recent.write_text('{"session_id":"recent","messages":[]}', encoding='utf-8')
    assert list(manifest.iter_json_session_files(tmp_path, min_file_age_seconds=900)) == []
    assert list(manifest.iter_json_session_files(tmp_path, min_file_age_seconds=0)) == [recent]


def test_profile_session_records_are_namespaced_and_tagged(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    path = tmp_path / 'session_kanban_001.json'
    path.write_text(json.dumps({
        'session_id': 'stale-embedded-id',
        'messages': [
            {'role': 'user', 'content': 'Hindsight session retain should include project:hindsight profile data.'},
            {'role': 'assistant', 'content': 'Profile namespace avoids incremental collisions.'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    default_record = manifest.records_from_json_file(path, bank_target='hermes')[0]
    profile_record = manifest.records_from_json_file(path, bank_target='hermes', source_profile='coordinator')[0]

    assert default_record['document_id'] == 'hermes-session::kanban_001'
    assert profile_record['document_id'] == 'hermes-session::coordinator::kanban_001'
    assert profile_record['metadata']['source_profile'] == 'coordinator'
    assert profile_record['metadata']['source_label'] == 'hermes-profile:coordinator'
    assert 'profile:coordinator' in profile_record['tags']
    assert 'source:kanban-profile' in profile_record['tags']


def test_discover_session_sources_includes_only_hindsight_profiles_by_default(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    default_sessions = tmp_path / 'sessions'
    default_sessions.mkdir()
    default_state = tmp_path / 'state.db'
    profile_root = tmp_path / 'profiles'
    for name, provider in [('planner', 'hindsight'), ('critic', 'holographic')]:
        profile = profile_root / name
        (profile / 'sessions').mkdir(parents=True)
        (profile / 'config.yaml').write_text(f'memory:\n  provider: {provider}\n', encoding='utf-8')

    sources = manifest.discover_session_sources(
        sessions_dir=default_sessions,
        state_db=default_state,
        profile_root=profile_root,
        profile_mode='hindsight',
    )

    assert [s['profile'] for s in sources] == ['default', 'planner']


def test_build_manifest_from_session_sources_counts_profile_records(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    default_sessions = tmp_path / 'sessions'
    profile_sessions = tmp_path / 'profiles' / 'planner' / 'sessions'
    default_sessions.mkdir()
    profile_sessions.mkdir(parents=True)
    default_sessions.joinpath('session_default_001.json').write_text(json.dumps({
        'messages': [{'role': 'user', 'content': 'Hindsight default session project:hindsight'}]
    }, ensure_ascii=False), encoding='utf-8')
    profile_sessions.joinpath('session_planner_001.json').write_text(json.dumps({
        'messages': [{'role': 'user', 'content': 'Hindsight planner session project:hindsight'}]
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.build_manifest_from_session_sources(
        sources=[
            {'profile': 'default', 'sessions_dir': default_sessions, 'state_db': tmp_path / 'state.db'},
            {'profile': 'planner', 'sessions_dir': profile_sessions, 'state_db': tmp_path / 'profiles' / 'planner' / 'state.db'},
        ],
        bank_target='hermes',
        min_file_age_seconds=0,
    )
    summary = manifest.summarize_records(records)

    assert {r['document_id'] for r in records} == {
        'hermes-session::default_001',
        'hermes-session::planner::planner_001',
    }
    assert summary['by_profile'] == {'default': 1, 'planner': 1}
