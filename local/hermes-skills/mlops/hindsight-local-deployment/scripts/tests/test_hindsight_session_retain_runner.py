import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_session(path: Path, session_id='s1', text='Hindsight observations consolidation'):
    path.write_text(json.dumps({
        'session_id': session_id,
        'model': 'test-model',
        'platform': 'cli',
        'session_start': '2026-05-08T02:00:00',
        'last_updated': '2026-05-08T02:30:00',
        'messages': [
            {'role': 'user', 'content': text},
            {'role': 'assistant', 'content': '使用 session retain 和 observation_scopes。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')


def test_retain_runner_rehydrates_lean_manifest_content_and_filters_production(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s1.json'
    make_session(session_file, session_id='s1')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    manual = dict(prod)
    manual['document_id'] = 'manual-doc'
    manual['action'] = 'manual_review'
    paths = manifest_mod.write_manifest([prod, manual], tmp_path, include_content=False)

    entries = runner.load_manifest(paths['manifest'])
    items, skipped = runner.prepare_retain_items(entries, action='production')

    assert len(items) == 1
    assert skipped['manual_review'] == 1
    item = items[0]
    assert item['document_id'] == 'hermes-session::s1'
    assert 'Hindsight observations consolidation' in item['content']
    assert item['update_mode'] == 'replace'
    assert item['event_date'] == '2026-05-08T02:00:00'
    assert ['domain:hindsight'] in item['observation_scopes']
    assert item['metadata']['source_kind'] == 'hermes_json'


def test_retain_runner_rehydrates_profile_namespaced_lean_manifest(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_profile_001.json'
    make_session(session_file, session_id='embedded-stale', text='Hindsight profile retain project:hindsight')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3', source_profile='planner')[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)

    item = runner.prepare_retain_items(runner.load_manifest(paths['manifest']), action='production')[0][0]

    assert item['document_id'] == 'hermes-session::planner::profile_001'
    assert item['metadata']['source_profile'] == 'planner'
    assert item['metadata']['source_label'] == 'hermes-profile:planner'
    assert 'Hindsight profile retain' in item['content']


def test_retain_runner_rehydrates_codex_rollout_lean_manifest(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    rollout = tmp_path / 'rollout-2026-05-20T12-24-43-019e43a0-demo.jsonl'
    events = [
        {
            'type': 'session_meta',
            'payload': {
                'id': '019e43a0-demo',
                'timestamp': '2026-05-20T04:24:43.244Z',
                'cwd': str(tmp_path),
                'model': 'gpt-5.5',
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
            'payload': {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': '已接入 Codex rollout manifest。'}],
            },
        },
    ]
    rollout.write_text('\n'.join(json.dumps(e, ensure_ascii=False) for e in events), encoding='utf-8')
    prod = manifest_mod.records_from_codex_rollout_file(rollout, bank_target='hermes')[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)

    item = runner.prepare_retain_items(runner.load_manifest(paths['manifest']), action='production')[0][0]

    assert item['document_id'] == 'codex-session::019e43a0-demo'
    assert item['context'] == 'codex_session'
    assert item['metadata']['source_kind'] == 'codex_rollout_jsonl'
    assert '把 Codex 对话加入 Hindsight daily 流程' in item['content']
    assert '已接入 Codex rollout manifest' in item['content']


def test_retain_runner_rehydrates_codex_markdown_artifact_lean_manifest(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    artifact = tmp_path / 'codex-output.md'
    artifact.write_text('# Codex Output\n\nMarkdown artifact from Codex daily source.', encoding='utf-8')
    prod = manifest_mod.record_from_codex_markdown_artifact(
        artifact,
        bank_target='hermes',
        producer='codex_apply_patch',
        source_rollout_path=str(tmp_path / 'rollout-demo.jsonl'),
        session_id='019e43a0-demo',
    )[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)

    item = runner.prepare_retain_items(runner.load_manifest(paths['manifest']), action='production')[0][0]

    assert item['document_id'].startswith('codex-artifact::')
    assert item['context'] == 'codex_markdown_artifact'
    assert item['metadata']['source_kind'] == 'codex_markdown_artifact'
    assert item['metadata']['producer'] == 'codex_apply_patch'
    assert 'Markdown artifact from Codex daily source' in item['content']


def test_retain_runner_serializes_metadata_values_to_strings_for_hindsight_api(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s6.json'
    make_session(session_file, session_id='s6')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    prod['metadata']['nested'] = {'a': 1}
    prod['metadata']['number'] = 42
    prod['metadata']['flag'] = True

    item = runner.record_to_memory_item(prod)

    assert item['metadata']['source_kind'] == 'hermes_json'
    assert item['metadata']['number'] == '42'
    assert item['metadata']['flag'] == 'true'
    assert item['event_date'] == '2026-05-08T02:00:00'
    assert 'event_date' not in item['metadata']
    assert json.loads(item['metadata']['nested']) == {'a': 1}
    assert all(isinstance(v, str) for v in item['metadata'].values())


def test_retain_runner_dry_run_does_not_call_client(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s2.json'
    make_session(session_file, session_id='s2')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)

    class FakeClient:
        def __init__(self):
            self.calls = []
        def retain_items(self, items, async_mode=True):
            self.calls.append((items, async_mode))
            return {'operation_id': 'should-not-call'}

    client = FakeClient()
    result = runner.run_manifest(paths['manifest'], client=client, dry_run=True, confirm=None)

    assert result['dry_run'] is True
    assert result['submitted_items'] == 0
    assert result['would_submit_items'] == 1
    assert client.calls == []


def test_retain_runner_submit_requires_confirm_token(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s3.json'
    make_session(session_file, session_id='s3')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)

    class FakeClient:
        def __init__(self):
            self.calls = []
        def retain_items(self, items, async_mode=True):
            self.calls.append((items, async_mode))
            return {'operation_id': 'op-1'}
        def iter_operations(self, max_items=1000):
            return [{'operation_id': 'op-1', 'status': 'completed'}]

    client = FakeClient()
    try:
        runner.run_manifest(paths['manifest'], client=client, dry_run=False, confirm=None)
    except runner.UnsafeRetainOperation as exc:
        assert 'confirm' in str(exc).lower()
    else:
        raise AssertionError('submit must require confirm token')
    assert client.calls == []

    result = runner.run_manifest(paths['manifest'], client=client, dry_run=False, confirm=runner.RETAIN_CONFIRM)
    assert result['dry_run'] is False
    assert result['submitted_items'] == 1
    assert result['responses'] == [{'operation_id': 'op-1'}]
    assert len(client.calls) == 1
    assert client.calls[0][1] is True


def test_incremental_filter_skips_unchanged_document_and_keeps_changed(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s4.json'
    make_session(session_file, session_id='s4', text='Hindsight incremental retain state')
    prod_v1 = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    state = {
        'documents': {
            prod_v1['document_id']: {
                'bank': 'hermes_v3',
                'content_sha256': prod_v1['metadata']['content_sha256'],
                'source_mtime_ns': prod_v1['metadata']['source_mtime_ns'],
            }
        }
    }

    items, skipped = runner.prepare_retain_items([prod_v1], action='production', submit_state=state, bank='hermes_v3')
    assert items == []
    assert skipped['unchanged'] == 1

    # The same document/content must still be submitted to a different bank; the
    # submit-state is per successful bank write, not a global document cache.
    items, skipped = runner.prepare_retain_items([prod_v1], action='production', submit_state=state, bank='hermes_v3_minimax_smoke')
    assert len(items) == 1
    assert skipped['unchanged'] == 0

    make_session(session_file, session_id='s4', text='Hindsight incremental retain state changed with new recall details')
    prod_v2 = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    items, skipped = runner.prepare_retain_items([prod_v2], action='production', submit_state=state, bank='hermes_v3')
    assert len(items) == 1
    assert skipped['unchanged'] == 0
    assert items[0]['metadata']['content_sha256'] != state['documents'][prod_v1['document_id']]['content_sha256']


def test_dry_run_does_not_update_submit_state_but_execute_success_does(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s5.json'
    make_session(session_file, session_id='s5')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)
    state_path = tmp_path / 'submit-state.json'

    class FakeClient:
        def retain_items(self, items, async_mode=True):
            return {'operation_id': 'op-ok', 'status': 'queued'}
        def iter_operations(self, max_items=1000):
            return [{'operation_id': 'op-ok', 'status': 'completed'}]

    dry = runner.run_manifest(paths['manifest'], client=FakeClient(), dry_run=True, submit_state_path=state_path)
    assert dry['would_submit_items'] == 1
    assert not state_path.exists()

    done = runner.run_manifest(
        paths['manifest'],
        client=FakeClient(),
        dry_run=False,
        confirm=runner.RETAIN_CONFIRM,
        submit_state_path=state_path,
    )
    assert done['submitted_items'] == 1
    saved = json.loads(state_path.read_text(encoding='utf-8'))
    doc_state = saved['documents'][runner.submit_state_document_key(prod['document_id'], 'hermes_v3')]
    assert doc_state['content_sha256'] == prod['metadata']['content_sha256']
    assert doc_state['source_mtime_ns'] == prod['metadata']['source_mtime_ns']
    assert doc_state['last_submit_manifest'] == str(paths['manifest'])


def test_execute_does_not_update_submit_state_when_async_operation_fails(tmp_path):
    manifest_mod = load_module('hindsight_session_manifest')
    runner = load_module('hindsight_session_retain_runner')
    session_file = tmp_path / 'session_s7.json'
    make_session(session_file, session_id='s7')
    prod = manifest_mod.records_from_json_file(session_file, bank_target='hermes_v3')[0]
    paths = manifest_mod.write_manifest([prod], tmp_path, include_content=False)
    state_path = tmp_path / 'submit-state.json'

    class FakeClient:
        def retain_items(self, items, async_mode=True):
            return {'operation_id': 'op-fail', 'status': 'queued'}
        def iter_operations(self, max_items=1000):
            return [{'operation_id': 'op-fail', 'status': 'failed'}]

    try:
        runner.run_manifest(paths['manifest'], client=FakeClient(), dry_run=False, confirm=runner.RETAIN_CONFIRM, submit_state_path=state_path)
    except runner.RetainOperationFailed as exc:
        assert 'op-fail' in str(exc)
    else:
        raise AssertionError('failed async operation should raise')
    assert not state_path.exists()
