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


def test_codex_rollout_records_keep_dialogue_and_drop_instruction_tool_noise(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    path = tmp_path / 'rollout-2026-05-20T12-24-43-019e43a0-demo.jsonl'
    events = [
        {
            'type': 'session_meta',
            'payload': {
                'id': '019e43a0-demo',
                'timestamp': '2026-05-20T04:24:43.244Z',
                'cwd': '/home/wyr/桌面',
                'originator': 'codex-tui',
                'cli_version': '0.128.0',
                'model': 'gpt-5.5',
                'base_instructions': {'text': 'DO NOT RETAIN BASE INSTRUCTIONS'},
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'developer',
                'content': [{'type': 'input_text', 'text': 'developer instruction should be dropped'}],
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': '# AGENTS.md instructions for /tmp\n\n<environment_context>noise</environment_context>'}],
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': '把 Codex 对话加入 Hindsight daily 流程'}],
            },
        },
        {
            'type': 'response_item',
            'payload': {'type': 'reasoning', 'encrypted_content': 'secret reasoning'},
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'function_call_output',
                'output': 'tool output should be dropped sk-test-secret',
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'assistant',
                'phase': 'final_answer',
                'content': [{'type': 'output_text', 'text': '已接入 Hindsight daily manifest。'}],
            },
        },
    ]
    path.write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in events), encoding='utf-8')

    records = manifest.records_from_codex_rollout_file(path, bank_target='hermes')

    assert len(records) == 1
    rec = records[0]
    assert rec['document_id'] == 'codex-session::019e43a0-demo'
    assert rec['context'] == 'codex_session'
    assert rec['event_date'] == '2026-05-20T04:24:43.244Z'
    assert rec['metadata']['source_kind'] == 'codex_rollout_jsonl'
    assert rec['metadata']['source_label'] == 'codex'
    assert rec['metadata']['cwd'] == '/home/wyr/桌面'
    assert 'source:codex-session' in rec['tags']
    assert rec['action'] == 'production'
    assert '把 Codex 对话加入 Hindsight daily 流程' in rec['content']
    assert '已接入 Hindsight daily manifest' in rec['content']
    assert 'DO NOT RETAIN' not in rec['content']
    assert 'developer instruction' not in rec['content']
    assert 'AGENTS.md instructions' not in rec['content']
    assert 'tool output should be dropped' not in rec['content']
    assert 'secret reasoning' not in rec['content']


def test_recent_codex_rollout_files_are_not_yielded(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    root = tmp_path / 'codex' / 'sessions' / '2026' / '05' / '20'
    root.mkdir(parents=True)
    recent = root / 'rollout-2026-05-20T12-24-43-019e43a0-demo.jsonl'
    recent.write_text('', encoding='utf-8')
    assert list(manifest.iter_codex_rollout_files(tmp_path / 'codex' / 'sessions', min_file_age_seconds=900)) == []
    assert list(manifest.iter_codex_rollout_files(tmp_path / 'codex' / 'sessions', min_file_age_seconds=0)) == [recent]


def test_codex_rollout_records_include_successfully_written_markdown_artifacts(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    artifact = tmp_path / 'notes' / 'codex-output.md'
    artifact.parent.mkdir()
    artifact.write_text('# Codex Output\n\nImportant Hindsight daily source design.', encoding='utf-8')
    ignored = tmp_path / 'notes' / 'existing-reference.md'
    ignored.write_text('# Existing Reference\n\nShould not be imported just because user mentioned it.', encoding='utf-8')
    rollout = tmp_path / 'rollout-2026-05-20T12-24-43-019e43a0-demo.jsonl'
    events = [
        {
            'type': 'session_meta',
            'payload': {
                'id': '019e43a0-demo',
                'timestamp': '2026-05-20T04:24:43.244Z',
                'cwd': str(tmp_path),
                'originator': 'codex-tui',
                'model': 'gpt-5.5',
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'user',
                'content': [{'type': 'input_text', 'text': f'请参考已有文档 {ignored}'}],
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'custom_tool_call',
                'name': 'apply_patch',
                'call_id': 'patch-1',
                'input': f'*** Begin Patch\n*** Add File: {artifact}\n+# Codex Output\n*** End Patch\n',
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'custom_tool_call_output',
                'call_id': 'patch-1',
                'output': '{"output":"Success. Updated the following files:\\nA ' + str(artifact) + '\\n"}',
            },
        },
        {
            'type': 'response_item',
            'payload': {
                'type': 'message',
                'role': 'assistant',
                'phase': 'final_answer',
                'content': [{'type': 'output_text', 'text': '已写入 Markdown artifact。'}],
            },
        },
    ]
    rollout.write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in events), encoding='utf-8')

    records = manifest.records_from_codex_rollout_file(rollout, bank_target='hermes')
    artifact_records = [r for r in records if r['metadata']['source_kind'] == 'codex_markdown_artifact']

    assert len(artifact_records) == 1
    rec = artifact_records[0]
    assert rec['document_id'].startswith('codex-artifact::')
    assert rec['context'] == 'codex_markdown_artifact'
    assert rec['metadata']['source_path'] == str(artifact)
    assert rec['metadata']['producer'] == 'codex_apply_patch'
    assert 'source:codex-artifact' in rec['tags']
    assert 'topic:markdown-artifact' in rec['tags']
    assert 'Important Hindsight daily source design' in rec['content']
    assert str(ignored) not in [r['metadata'].get('source_path') for r in artifact_records]
