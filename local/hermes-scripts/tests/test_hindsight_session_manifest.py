import importlib.util
import json
import os
import sqlite3
import sys
import time
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


def test_hermes_json_records_include_write_file_markdown_artifacts(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    artifact = tmp_path / 'plans' / 'hermes-output.md'
    artifact.parent.mkdir()
    artifact.write_text('# Hermes Output\n\nEgomotion4D daily markdown artifact from deepseek tui.', encoding='utf-8')
    session = tmp_path / 'session_hermes_md.json'
    session.write_text(json.dumps({
        'session_id': 'stale',
        'messages': [
            {'role': 'user', 'content': '把 Egomotion4D 计划写成 md 文件。'},
            {
                'role': 'assistant',
                'content': '已写入 markdown 文件。',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'write_file',
                            'arguments': json.dumps({'path': str(artifact), 'content': '# Hermes Output'}),
                        }
                    }
                ],
            },
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(
        session,
        bank_target='hermes',
        source_profile='planner',
        include_markdown_artifacts=True,
        min_markdown_file_age_seconds=0,
    )
    artifact_records = [r for r in records if r['metadata']['source_kind'] == 'hermes_markdown_artifact']

    assert len(artifact_records) == 1
    rec = artifact_records[0]
    assert rec['context'] == 'hermes_markdown_artifact'
    assert rec['metadata']['producer'] == 'hermes_write_file_tool'
    assert rec['metadata']['source_profile'] == 'planner'
    assert rec['metadata']['source_file_md5']
    assert 'Egomotion4D daily markdown artifact from deepseek tui' in rec['content']


def test_daily_manifest_dedupes_markdown_artifacts_by_md5_across_sources(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    shared = '# Shared Markdown\n\nEgomotion4D duplicate markdown artifact content.'
    hermes_artifact = tmp_path / 'hermes-output.md'
    codex_artifact = tmp_path / 'codex-output.md'
    hermes_artifact.write_text(shared, encoding='utf-8')
    codex_artifact.write_text(shared, encoding='utf-8')
    session = tmp_path / 'session_hermes_md.json'
    session.write_text(json.dumps({
        'session_id': 'hermes-md',
        'messages': [
            {'role': 'user', 'content': 'Egomotion4D markdown artifact'},
            {
                'role': 'assistant',
                'content': '已写入 markdown 文件。',
                'tool_calls': [
                    {
                        'function': {
                            'name': 'write_file',
                            'arguments': json.dumps({'path': str(hermes_artifact)}),
                        }
                    }
                ],
            },
        ],
    }, ensure_ascii=False), encoding='utf-8')
    rollout = tmp_path / 'rollout-2026-05-20T12-24-43-019e43a0-demo.jsonl'
    events = [
        {'type': 'session_meta', 'payload': {'id': '019e43a0-demo', 'timestamp': '2026-05-20T04:24:43.244Z', 'cwd': str(tmp_path)}},
        {
            'type': 'response_item',
            'payload': {
                'type': 'custom_tool_call',
                'name': 'apply_patch',
                'call_id': 'patch-1',
                'input': f'*** Begin Patch\n*** Add File: {codex_artifact}\n+# Shared Markdown\n*** End Patch\n',
            },
        },
        {'type': 'response_item', 'payload': {'type': 'custom_tool_call_output', 'call_id': 'patch-1', 'output': '{"output":"Success. Updated the following files:\\nA ' + str(codex_artifact) + '\\n"}'}},
    ]
    rollout.write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in events), encoding='utf-8')

    records = []
    records.extend(manifest.records_from_json_file(session, bank_target='hermes', include_markdown_artifacts=True, min_markdown_file_age_seconds=0))
    records.extend(manifest.records_from_codex_rollout_file(rollout, bank_target='hermes'))

    deduped, diagnostics = manifest.dedupe_manifest_records(records)
    artifact_records = [
        r for r in deduped
        if (r.get('metadata') or {}).get('source_kind') in {'codex_markdown_artifact', 'hermes_markdown_artifact'}
    ]

    assert len(artifact_records) == 1
    assert artifact_records[0]['metadata']['full_content_md5']
    assert diagnostics['duplicate_markdown_content_md5'] == {artifact_records[0]['metadata']['full_content_md5']: 2}


def test_kanban_prompt_markdown_records_are_imported_from_workspace_roots(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    prompt = tmp_path / 'Egomotion4D' / '.codex-kanban' / 'egomotion4d' / 'planner' / 't_plan.md'
    prompt.parent.mkdir(parents=True)
    prompt.write_text('# Hermes Kanban task for role: planner\n\nKanban prompt md should enter daily Hindsight.', encoding='utf-8')

    records = manifest.build_manifest_from_kanban_sources(
        workspace_roots=[tmp_path],
        board_db_paths=[],
        bank_target='hermes',
        min_file_age_seconds=0,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec['document_id'].startswith('kanban-markdown::')
    assert rec['context'] == 'kanban_prompt_markdown'
    assert rec['metadata']['source_kind'] == 'kanban_prompt_markdown'
    assert rec['metadata']['board'] == 'egomotion4d'
    assert rec['metadata']['profile'] == 'planner'
    assert rec['metadata']['task_id'] == 't_plan'
    assert 'source:kanban-markdown' in rec['tags']
    assert 'Kanban prompt md should enter daily Hindsight' in rec['content']


def test_kanban_comment_records_are_imported_from_board_db(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    db = tmp_path / 'kanban.db'
    con = sqlite3.connect(db)
    con.executescript(
        '''
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT,
            body TEXT,
            assignee TEXT,
            status TEXT,
            created_at INTEGER,
            completed_at INTEGER,
            result TEXT
        );
        CREATE TABLE task_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        '''
    )
    now = int(time.time()) - 3600
    con.execute(
        'INSERT INTO tasks (id,title,body,assignee,status,created_at,result) VALUES (?,?,?,?,?,?,?)',
        ('t_123', 'Fix daily Kanban ingestion', 'include comments', 'planner', 'done', now, 'implemented'),
    )
    con.execute(
        'INSERT INTO task_comments (task_id,author,body,created_at) VALUES (?,?,?,?)',
        ('t_123', 'critic', 'Comment thread should be retained for Hindsight recall.', now + 10),
    )
    con.commit()
    con.close()

    records = manifest.build_manifest_from_kanban_sources(
        workspace_roots=[],
        board_db_paths=[('egomotion4d', db)],
        bank_target='hermes',
        min_file_age_seconds=0,
    )

    assert len(records) == 1
    rec = records[0]
    assert rec['document_id'] == 'kanban-comment::egomotion4d::t_123::1'
    assert rec['context'] == 'kanban_comment'
    assert rec['metadata']['source_kind'] == 'kanban_task_comment'
    assert rec['metadata']['board'] == 'egomotion4d'
    assert rec['metadata']['task_id'] == 't_123'
    assert rec['metadata']['comment_id'] == 1
    assert 'source:kanban-comment' in rec['tags']
    assert 'Fix daily Kanban ingestion' in rec['content']
    assert 'Comment thread should be retained for Hindsight recall' in rec['content']


def test_kanban_prompt_markdown_dedupes_same_file_and_same_task(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    older = tmp_path / 'repo-a' / '.codex-kanban' / 'egomotion4d' / 'planner' / 't_same.md'
    newer = tmp_path / 'repo-b' / '.deepseek-kanban' / 'egomotion4d' / 'planner' / 't_same.md'
    older.parent.mkdir(parents=True)
    newer.parent.mkdir(parents=True)
    older.write_text('# Older Prompt\n\nOld task prompt should lose same-task dedupe.', encoding='utf-8')
    newer.write_text('# Newer Prompt\n\nNew task prompt should win same-task dedupe.', encoding='utf-8')
    old_ts = time.time() - 600
    new_ts = time.time() - 60
    os_utime = getattr(__import__('os'), 'utime')
    os_utime(older, (old_ts, old_ts))
    os_utime(newer, (new_ts, new_ts))

    records = manifest.build_manifest_from_kanban_sources(
        workspace_roots=[tmp_path, tmp_path],
        board_db_paths=[],
        bank_target='hermes',
        min_file_age_seconds=0,
    )

    assert len(records) == 1
    assert records[0]['metadata']['source_path'] == str(newer)
    assert 'New task prompt should win' in records[0]['content']


def test_kanban_sources_apply_incremental_cutoff_to_files_and_comments(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    cutoff_seconds = int(time.time()) - 300
    cutoff_ns = cutoff_seconds * 1_000_000_000

    old_prompt = tmp_path / 'repo' / '.codex-kanban' / 'egomotion4d' / 'planner' / 't_old.md'
    new_prompt = tmp_path / 'repo' / '.codex-kanban' / 'egomotion4d' / 'planner' / 't_new.md'
    old_prompt.parent.mkdir(parents=True)
    old_prompt.write_text('# Old Prompt\n\nShould be skipped by since cutoff.', encoding='utf-8')
    new_prompt.write_text('# New Prompt\n\nShould pass since cutoff.', encoding='utf-8')
    os_utime = getattr(__import__('os'), 'utime')
    os_utime(old_prompt, (cutoff_seconds - 100, cutoff_seconds - 100))
    os_utime(new_prompt, (cutoff_seconds + 100, cutoff_seconds + 100))

    db = tmp_path / 'kanban.db'
    con = sqlite3.connect(db)
    con.executescript(
        '''
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT,
            body TEXT,
            assignee TEXT,
            status TEXT,
            created_at INTEGER,
            completed_at INTEGER,
            result TEXT
        );
        CREATE TABLE task_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        '''
    )
    con.execute('INSERT INTO tasks (id,title,body,assignee,status,created_at,result) VALUES (?,?,?,?,?,?,?)', ('t_old', 'old', '', 'planner', 'done', cutoff_seconds - 200, ''))
    con.execute('INSERT INTO tasks (id,title,body,assignee,status,created_at,result) VALUES (?,?,?,?,?,?,?)', ('t_new', 'new', '', 'planner', 'done', cutoff_seconds + 100, ''))
    con.execute('INSERT INTO task_comments (task_id,author,body,created_at) VALUES (?,?,?,?)', ('t_old', 'critic', 'old comment skipped', cutoff_seconds - 100))
    con.execute('INSERT INTO task_comments (task_id,author,body,created_at) VALUES (?,?,?,?)', ('t_new', 'critic', 'new comment kept', cutoff_seconds + 100))
    con.commit()
    con.close()

    records = manifest.build_manifest_from_kanban_sources(
        workspace_roots=[tmp_path],
        board_db_paths=[('egomotion4d', db)],
        bank_target='hermes',
        since_mtime_ns=cutoff_ns,
        min_file_age_seconds=0,
    )
    bodies = '\n'.join(r.get('content', '') for r in records)

    assert 'Should pass since cutoff' in bodies
    assert 'new comment kept' in bodies
    assert 'Should be skipped by since cutoff' not in bodies
    assert 'old comment skipped' not in bodies


def test_cli_scan_state_filters_known_sessions_but_keeps_first_kanban_scan(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    old_session = sessions / 'session_old.json'
    new_session = sessions / 'session_new.json'
    old_session.write_text(json.dumps({'session_id': 'old', 'messages': [{'role': 'user', 'content': 'old Hindsight memory'}]}, ensure_ascii=False), encoding='utf-8')
    new_session.write_text(json.dumps({'session_id': 'new', 'messages': [{'role': 'user', 'content': 'new Hindsight memory'}]}, ensure_ascii=False), encoding='utf-8')
    prompt = tmp_path / 'repo' / '.codex-kanban' / 'egomotion4d' / 'planner' / 't_first.md'
    prompt.parent.mkdir(parents=True)
    prompt.write_text('# First Kanban Prompt\n\nFirst Kanban source scan must not be skipped by session cutoff.', encoding='utf-8')
    now = int(time.time())
    old_ns = (now - 3000) * 1_000_000_000
    new_ns = (now - 2000) * 1_000_000_000
    prompt_ns = (now - 4000) * 1_000_000_000
    for path, ns in [(old_session, old_ns), (new_session, new_ns), (prompt, prompt_ns)]:
        os.utime(path, ns=(ns, ns))
    state_path = tmp_path / 'scan-state.json'
    state_path.write_text(json.dumps({
        'schema_version': 'session-manifest-scan-state-v1',
        'sources': {'hermes_json': {'max_source_mtime_ns': old_ns + 1}},
    }), encoding='utf-8')
    out_dir = tmp_path / 'out'

    rc = manifest.main([
        '--sessions-dir', str(sessions),
        '--profile-mode', 'none',
        '--no-include-codex',
        '--include-kanban',
        '--kanban-workspace-root', str(tmp_path),
        '--min-file-age-seconds', '0',
        '--scan-state', str(state_path),
        '--write-scan-state',
        '--output-dir', str(out_dir),
        '--json',
    ])

    assert rc == 0
    latest = json.loads((out_dir / 'latest.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in Path(latest['manifest']).read_text(encoding='utf-8').splitlines() if line.strip()]
    source_kinds = [(r.get('metadata') or {}).get('source_kind') for r in records]
    assert 'hermes_json' in source_kinds
    assert 'kanban_prompt_markdown' in source_kinds
    assert [r['metadata']['session_id'] for r in records if r['metadata'].get('source_kind') == 'hermes_json'] == ['new']
    saved_state = json.loads(state_path.read_text(encoding='utf-8'))
    assert saved_state['sources']['hermes_json']['max_source_mtime_ns'] == new_ns
    assert saved_state['sources']['kanban_prompt_markdown']['max_source_mtime_ns'] == prompt_ns


def test_cli_scan_state_read_only_without_explicit_write_flag(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    session = sessions / 'session_new.json'
    session.write_text(json.dumps({'session_id': 'new', 'messages': [{'role': 'user', 'content': 'new Hindsight memory'}]}, ensure_ascii=False), encoding='utf-8')
    state_path = tmp_path / 'scan-state.json'
    state_path.write_text(json.dumps({'schema_version': 'session-manifest-scan-state-v1', 'sources': {}}), encoding='utf-8')
    before = state_path.read_text(encoding='utf-8')

    rc = manifest.main([
        '--sessions-dir', str(sessions),
        '--profile-mode', 'none',
        '--no-include-codex',
        '--no-include-kanban',
        '--min-file-age-seconds', '0',
        '--scan-state', str(state_path),
        '--output-dir', str(tmp_path / 'out'),
        '--json',
    ])

    assert rc == 0
    assert state_path.read_text(encoding='utf-8') == before
