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


def test_session_manifest_cleans_noise_and_keeps_user_assistant_text():
    manifest = load_module('hindsight_session_manifest')
    session = {
        'session_id': '20260508_test',
        'model': 'gpt-test',
        'platform': 'cli',
        'session_start': '2026-05-08T00:00:00',
        'last_updated': '2026-05-08T00:10:00',
        'messages': [
            {'role': 'system', 'content': 'SYSTEM PROMPT SHOULD NOT BE RETAINED'},
            {'role': 'user', 'content': '继续讨论 Hindsight native consolidation 和 recall cache'},
            {'role': 'assistant', 'reasoning': 'hidden chain of thought', 'reasoning_content': 'hidden reasoning content', 'codex_reasoning_items': [{'type': 'reasoning', 'text': 'hidden codex reasoning'}], 'content': '结论：先做 session manifest dry-run。'},
            {'role': 'assistant', 'content': [
                {'type': 'thinking', 'text': 'hidden list thinking'},
                {'type': 'text', 'text': '<think>hidden xml thinking</think>最终结论：继续用原生 retain。'},
            ]},
            {'role': 'tool', 'content': 'terminal output should be dropped'},
            {'role': 'assistant', 'content': '<memory-context>old recalled memory</memory-context>\n这段应该只保留真正回复。'},
            {'role': 'user', 'content': 'NUL\x00字符要清理'},
        ],
    }

    text, stats = manifest.extract_clean_conversation(session)

    assert '继续讨论 Hindsight native consolidation' in text
    assert '结论：先做 session manifest dry-run。' in text
    assert '最终结论：继续用原生 retain。' in text
    assert 'NUL字符要清理' in text
    assert 'SYSTEM PROMPT SHOULD NOT BE RETAINED' not in text
    assert 'terminal output should be dropped' not in text
    assert 'hidden chain of thought' not in text
    assert 'hidden reasoning content' not in text
    assert 'hidden codex reasoning' not in text
    assert 'hidden list thinking' not in text
    assert 'hidden xml thinking' not in text
    assert '<memory-context>' not in text
    assert stats['kept_messages'] == 5
    assert stats['dropped_messages'] == 2


def test_session_manifest_drops_self_reflection_skill_prompt_noise():
    manifest = load_module('hindsight_session_manifest')
    session = {
        'session_id': 'reflection-noise',
        'messages': [
            {'role': 'user', 'content': 'OpenClaw 升级后 PATH 多版本冲突怎么处理？'},
            {'role': 'assistant', 'content': '检查 npm root -g 和 which openclaw。'},
            {'role': 'user', 'content': 'Review the conversation above and consider saving or updating a skill if appropriate.\n\nFocus on: was a non-trivial approach used to complete a task?'},
        ],
    }

    text, stats = manifest.extract_clean_conversation(session)

    assert 'OpenClaw 升级后 PATH 多版本冲突' in text
    assert 'Review the conversation above' not in text
    assert stats['kept_messages'] == 2
    assert stats['dropped_noise_messages'] == 1


def test_manifest_records_use_semantic_tags_and_metadata_for_source_labels(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_20260508_abcd.json'
    session_path.write_text(json.dumps({
        'session_id': '20260508_abcd',
        'model': 'gpt-test',
        'platform': 'cli',
        'session_start': '2026-05-08T00:00:00',
        'last_updated': '2026-05-08T00:10:00',
        'messages': [
            {'role': 'user', 'content': 'Hindsight memory provider 的 observations/consolidation 怎么治理？'},
            {'role': 'assistant', 'content': '应该用 native-first，session retain，显式 observation_scopes。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3', max_document_chars=10000)

    assert len(records) == 1
    rec = records[0]
    assert rec['document_id'] == 'hermes-session::20260508_abcd'
    assert rec['action'] == 'production'
    assert 'domain:hindsight' in rec['tags']
    assert 'topic:memory-management' in rec['tags']
    assert 'hermes' not in rec['tags']
    assert 'sqlite' not in rec['tags']
    assert rec['metadata']['source_kind'] == 'hermes_json'
    assert rec['metadata']['source_label'] == 'hermes'
    assert rec['metadata']['started_at'] == '2026-05-08T00:00:00'
    assert rec['event_date'] == '2026-05-08T00:00:00'
    assert 'event_date' not in rec['metadata']
    assert ['domain:hindsight'] in rec['observation_scopes']
    assert ['topic:memory-management'] in rec['observation_scopes']
    assert rec['update_mode'] == 'replace'


def test_manifest_splits_large_session_without_append(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_long.json'
    long_text = 'Egomotion4D ATE metric discussion. ' * 80
    session_path.write_text(json.dumps({
        'session_id': 'long',
        'messages': [
            {'role': 'user', 'content': long_text},
            {'role': 'assistant', 'content': '保留轨迹、尺度和窗口配置。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3', max_document_chars=400)

    assert len(records) > 1
    assert records[0]['document_id'] == 'hermes-session::long::part-000'
    assert records[1]['document_id'] == 'hermes-session::long::part-001'
    assert {r['update_mode'] for r in records} == {'replace'}
    assert all(r['metadata']['part_count'] == len(records) for r in records)
    assert all(r['action'] == 'production' for r in records)
    assert all('project:egomotion4d' in r['tags'] for r in records)


def test_manifest_routes_overbroad_multiscope_sessions_to_manual_review(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_mixed.json'
    session_path.write_text(json.dumps({
        'session_id': 'mixed',
        'messages': [
            {'role': 'user', 'content': 'Egomotion4D ATE、OpenClaw approval、VGGT-Long loop closure、paper arxiv、Hindsight observations 都在同一段里。'},
            {'role': 'assistant', 'content': '这种 mixed session 需要人工确认 tag/scope，不能直接生产 consolidation。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'manual_review'
    assert records[0]['reason'] == 'multi_scope_or_overbroad_tags'
    assert 'project:egomotion4d' in records[0]['tags']
    assert 'project:openclaw' in records[0]['tags']


def test_manifest_routes_broad_memory_aggregate_summaries_to_manual_review(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_broad_aggregate.json'
    session_path.write_text(json.dumps({
        'session_id': 'broadagg',
        'messages': [
            {'role': 'user', 'content': '阅读所有详细笔记文件，提取核心技术知识。文件路径：~/.hermes/memories/details/，输出结构化总结。'},
            {'role': 'assistant', 'content': '包含端到端自动驾驶、Egomotion4D、VGGT4D、RK3588、论文、项目记忆等多个主题。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'manual_review'
    assert records[0]['reason'] == 'broad_aggregate_summary'


def test_manifest_does_not_tag_generic_scale_or_ego_motion_as_egomotion4d(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_l2_mono.json'
    session_path.write_text(json.dumps({
        'session_id': 'l2mono',
        'messages': [
            {'role': 'user', 'content': 'L2自动驾驶单目测距测速总结：单目没有绝对尺度，需要 ego-motion 补偿和地面建模。'},
            {'role': 'assistant', 'content': '输出测速架构：极坐标EKF、CUSUM、AEB safe双输出。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'production'
    assert 'domain:autodrive' in records[0]['tags']
    assert 'project:egomotion4d' not in records[0]['tags']


def test_manifest_skips_low_signal_greeting_and_ack_sessions(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_greeting.json'
    session_path.write_text(json.dumps({
        'session_id': 'greeting',
        'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': '在，老王。有什么要我处理的？'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'skip'
    assert records[0]['reason'] == 'low_signal_short_or_chitchat'


def test_manifest_skips_continue_or_ok_only_sessions(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_continue_only.json'
    session_path.write_text(json.dumps({
        'session_id': 'continue_only',
        'messages': [
            {'role': 'user', 'content': '继续'},
            {'role': 'assistant', 'content': '好，我继续处理。'},
            {'role': 'user', 'content': 'ok'},
            {'role': 'assistant', 'content': '收到。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'skip'
    assert records[0]['reason'] == 'low_signal_short_or_chitchat'


def test_manifest_keeps_continue_when_it_has_semantic_topic(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_continue_semantic.json'
    session_path.write_text(json.dumps({
        'session_id': 'continue_semantic',
        'messages': [
            {'role': 'user', 'content': '继续讨论 Hindsight native consolidation 和 recall cache'},
            {'role': 'assistant', 'content': '结论：保留 native-first，session retain 只做 dry-run。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'production'
    assert 'domain:hindsight' in records[0]['tags']


def test_manifest_skips_identity_question_sessions(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    for phrase in ['你是谁', 'who aru u', 'who r u', 'who are you', 'who are u']:
        session_path = tmp_path / f"session_identity_{manifest.sha256_text(phrase)[:8]}.json"
        session_path.write_text(json.dumps({
            'session_id': f"identity-{manifest.sha256_text(phrase)[:8]}",
            'messages': [
                {'role': 'user', 'content': phrase},
                {'role': 'assistant', 'content': '我是臭臭，自动驾驶领域 AI 助手。'},
            ],
        }, ensure_ascii=False), encoding='utf-8')

        records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

        assert len(records) == 1
        assert records[0]['action'] == 'skip', phrase
        assert records[0]['reason'] == 'low_signal_short_or_chitchat', phrase


def test_manifest_routes_secret_or_api_key_sessions_to_manual_review(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_secret_key.json'
    session_path.write_text(json.dumps({
        'session_id': 'secret_key',
        'messages': [
            {'role': 'user', 'content': 'Hindsight provider 配置好了，API key: sk-tes...3456，需要确认 retain 是否可用'},
            {'role': 'assistant', 'content': '配置验证通过，但不能把 key 写入长期记忆。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'manual_review'
    assert records[0]['reason'] == 'secret_or_credential_material'


def test_manifest_records_mark_candidate_filter_version(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_selector_version.json'
    session_path.write_text(json.dumps({
        'session_id': 'selector_version',
        'messages': [
            {'role': 'user', 'content': 'Hindsight session candidate selector 要排除 hi/ok/继续 等低价值会话'},
            {'role': 'assistant', 'content': '使用轻量确定性规则，不调用 LLM。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert records[0]['metadata']['candidate_filter_version'] == manifest.CANDIDATE_FILTER_VERSION


def test_manifest_routes_bootstrap_identity_sessions_to_manual_review_even_if_autodrive_intro(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_bootstrap.json'
    session_path.write_text(json.dumps({
        'session_id': 'bootstrap',
        'messages': [
            {'role': 'user', 'content': 'who are u'},
            {'role': 'assistant', 'content': '我是臭臭，自动驾驶领域 AI 助手。擅长 AEB、ADAS、车道线、单目测速。'},
            {'role': 'user', 'content': '检查下你的环境，我使用 hermes claw migrate 提示成功了，但看起来没起到作用'},
            {'role': 'assistant', 'content': '让我检查环境和 migrate 的效果。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'manual_review'
    assert records[0]['reason'] == 'bootstrap_or_environment_diagnostic'


def test_manifest_routes_memory_recall_bootstrap_sessions_to_manual_review(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_memory_recall_bootstrap.json'
    session_path.write_text(json.dumps({
        'session_id': 'memory_recall_bootstrap',
        'messages': [
            {'role': 'user', 'content': '回忆Egomotion4D'},
            {'role': 'assistant', 'content': '先把项目记忆和图谱都捞一遍，我直接帮你回忆。'},
            {'role': 'assistant', 'content': 'Egomotion4D 之前 scene0 ATE_metric=0.51m，Hindsight recall 里还有 DAGE 记录。'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'manual_review'
    assert records[0]['reason'] == 'memory_recall_or_context_bootstrap'


def test_manifest_routes_context_resume_sessions_to_manual_review(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_context_resume.json'
    session_path.write_text(json.dumps({
        'session_id': 'context_resume',
        'messages': [
            {'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': '在，老王。有什么要我处理的？'},
            {'role': 'user', 'content': '刚聊到哪儿了'},
            {'role': 'assistant', 'content': '我这边当前窗口只看到这句话。从长期记忆里能看到你在做 Egomotion4D。'},
            {'role': 'user', 'content': '所有尝试都失败了，所以应该如何提升 RPE？'},
        ],
    }, ensure_ascii=False), encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    assert records[0]['action'] == 'manual_review'
    assert records[0]['reason'] == 'context_resume_or_handoff'


def test_manifest_drops_preserved_task_list_noise():
    manifest = load_module('hindsight_session_manifest')
    session = {
        'session_id': 'task_list_noise',
        'messages': [
            {'role': 'user', 'content': '[Your active task list was preserved across context compression]\n- [>] internal todo'},
            {'role': 'user', 'content': 'Egomotion4D ATE_metric 需要继续验证。'},
        ],
    }

    text, stats = manifest.extract_clean_conversation(session)

    assert 'Your active task list' not in text
    assert 'internal todo' not in text
    assert 'Egomotion4D ATE_metric' in text
    assert stats['dropped_noise_messages'] == 1


def test_manifest_removes_model_switch_notes_from_message_text():
    manifest = load_module('hindsight_session_manifest')

    cleaned = manifest.clean_text('[Note: model was just switched from MiniMax-M2.7 to gpt-5.5 via cch. Adjust your self-identification accordingly.]\n\nhi')

    assert cleaned == 'hi'


def test_manifest_records_include_source_file_mtime_size_and_hash(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    session_path = tmp_path / 'session_incremental.json'
    payload = {
        'session_id': 'incremental',
        'last_updated': '2026-05-08T02:30:00',
        'messages': [
            {'role': 'user', 'content': 'Hindsight retain session 增量处理要看文件修改时间'},
            {'role': 'assistant', 'content': '同 document_id 变化后要 replace/re-retain。'},
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False)
    session_path.write_text(raw, encoding='utf-8')

    records = manifest.records_from_json_file(session_path, bank_target='hermes_v3')

    assert len(records) == 1
    meta = records[0]['metadata']
    assert meta['source_mtime_ns'] == session_path.stat().st_mtime_ns
    assert meta['source_size_bytes'] == session_path.stat().st_size
    assert meta['source_file_sha256'] == manifest.sha256_bytes(session_path.read_bytes())
    assert meta['session_last_updated'] == '2026-05-08T02:30:00'


def test_build_manifest_can_filter_by_source_mtime_ns_without_submit_state(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    old_file = tmp_path / 'session_old.json'
    new_file = tmp_path / 'session_new.json'
    old_file.write_text(json.dumps({'session_id': 'old', 'messages': [{'role': 'user', 'content': 'old Hindsight memory'}]}, ensure_ascii=False), encoding='utf-8')
    new_file.write_text(json.dumps({'session_id': 'new', 'messages': [{'role': 'user', 'content': 'new Hindsight memory'}]}, ensure_ascii=False), encoding='utf-8')
    cutoff = new_file.stat().st_mtime_ns - 1

    records = manifest.build_manifest_from_json_dir(sessions_dir=tmp_path, bank_target='hermes_v3', since_mtime_ns=cutoff)

    assert [r['metadata']['session_id'] for r in records] == ['new']


def test_summarize_records_includes_reason_counts_for_candidate_audit():
    manifest = load_module('hindsight_session_manifest')
    records = [
        {'action': 'skip', 'reason': 'low_signal_short_or_chitchat', 'content_chars': 10, 'estimated_retain_chunks': 0, 'tags': []},
        {'action': 'skip', 'reason': 'low_signal_short_or_chitchat', 'content_chars': 12, 'estimated_retain_chunks': 0, 'tags': []},
        {'action': 'manual_review', 'reason': 'no_semantic_tags', 'content_chars': 80, 'estimated_retain_chunks': 1, 'tags': []},
    ]

    summary = manifest.summarize_records(records)

    assert summary['by_action'] == {'skip': 2, 'manual_review': 1}
    assert summary['by_reason']['skip:low_signal_short_or_chitchat'] == 2
    assert summary['by_reason']['manual_review:no_semantic_tags'] == 1


def test_write_manifest_defaults_to_lean_records_without_full_content(tmp_path):
    manifest = load_module('hindsight_session_manifest')
    records = [{
        'document_id': 'doc-1',
        'content': 'sensitive full conversation',
        'content_chars': 27,
        'estimated_retain_chunks': 1,
        'action': 'production',
        'tags': ['domain:hindsight'],
        'metadata': {'content_sha256': 'abc'},
    }]

    paths = manifest.write_manifest(records, tmp_path, include_content=False)
    first = json.loads(Path(paths['manifest']).read_text(encoding='utf-8').splitlines()[0])

    assert 'content' not in first
    assert first['content_omitted'] is True
    assert first['metadata']['content_sha256'] == 'abc'


def test_discard_manager_snapshots_document_facts_and_observations(tmp_path):
    discard = load_module('hindsight_discard_manager')

    class FakeClient:
        def get_document(self, document_id):
            return {'id': document_id, 'original_text': 'source text', 'tags': ['domain:hindsight']}

        def iter_memories(self, types=None, max_items=None):
            rows = [
                {'id': 'fact-1', 'document_id': 'doc-1', 'type': 'world', 'text': 'fact text', 'tags': ['domain:hindsight']},
                {'id': 'obs-1', 'document_id': None, 'type': 'observation', 'text': 'obs text', 'source_memory_ids': ['fact-1'], 'tags': ['domain:hindsight']},
                {'id': 'obs-2', 'document_id': None, 'type': 'observation', 'text': 'unrelated', 'source_memory_ids': ['other'], 'tags': ['domain:hindsight']},
            ]
            for row in rows:
                yield row

    out = discard.snapshot_document(
        bank='hermes_v3',
        document_id='doc-1',
        case_id='case-1',
        output_root=tmp_path,
        client=FakeClient(),
        reason='test snapshot',
    )

    case_dir = Path(out['case_dir'])
    assert out['dry_run'] is True
    assert out['document_id'] == 'doc-1'
    assert out['counts']['documents'] == 1
    assert out['counts']['facts'] == 1
    assert out['counts']['derived_observations'] == 1
    assert (case_dir / 'document.json').exists()
    assert (case_dir / 'facts.jsonl').exists()
    assert (case_dir / 'derived_observations.jsonl').exists()
    manifest = json.loads((case_dir / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['case_id'] == 'case-1'
    assert manifest['mutation_allowed'] is False


def test_discard_verify_fails_when_snapshot_counts_do_not_match(tmp_path):
    discard = load_module('hindsight_discard_manager')
    case_dir = tmp_path / 'bad-case'
    case_dir.mkdir()
    (case_dir / 'manifest.json').write_text(json.dumps({
        'case_id': 'bad-case',
        'counts': {'documents': 1, 'facts': 2, 'derived_observations': 0},
    }), encoding='utf-8')
    (case_dir / 'document.json').write_text('{}', encoding='utf-8')
    (case_dir / 'facts.jsonl').write_text('{"id":"only-one"}\n', encoding='utf-8')
    (case_dir / 'derived_observations.jsonl').write_text('', encoding='utf-8')

    result = discard.verify_case(case_dir)

    assert result['ok'] is False
    assert any('facts' in err for err in result['errors'])
