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


def write_jsonl(path, records):
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def test_review_backlog_keeps_event_date_and_no_content_by_default(tmp_path):
    backlog = load_module('hindsight_review_backlog')
    manifest = tmp_path / 'manifest.jsonl'
    write_jsonl(manifest, [{
        'document_id': 'hermes-session::s1',
        'event_date': '2026-05-01T10:00:00',
        'content': 'User: 讨论 token cost，不包含真实 token: value。\nAssistant: 结论：保留。',
        'content_chars': 52,
        'action': 'production',
        'reason': 'semantic_tags_detected',
        'tags': ['domain:hindsight', 'topic:memory-management'],
        'observation_scopes': [['domain:hindsight']],
        'metadata': {'json_path': '/tmp/session_s1.json', 'content_sha256': 'abc'},
    }])

    rows, summary = backlog.build_backlog(
        manifest_path=manifest,
        hardening_paths=[],
        retry_gate_paths=[],
        unit_counts={'hermes-session::s1': {'memory_unit_count': 2}},
        bank='hermes',
    )

    assert summary['record_count'] == 1
    assert summary['missing_event_date'] == 0
    assert rows[0]['event_date'] == '2026-05-01T10:00:00'
    assert rows[0]['current_retain_outcome']['status'] == 'has_units'
    assert rows[0]['review']['recommended_route'] == 'monitor'
    assert 'content' not in rows[0]
    assert 'content_preview' not in rows[0]
    assert 'credential_like' not in rows[0]['deterministic_anomalies']


def test_review_backlog_zero_unit_hardening_routes_to_cluster_revisit(tmp_path):
    backlog = load_module('hindsight_review_backlog')
    manifest = tmp_path / 'manifest.jsonl'
    hardening = tmp_path / 'hardening.json'
    write_jsonl(manifest, [{
        'document_id': 'hermes-session::s2',
        'metadata': {'started_at': '2026-05-02T10:00:00', 'json_path': '/tmp/session_s2.json'},
        'content': 'User: 这里有实验结果和错误根因。\nAssistant: 结论：需要后续重捞。',
        'content_chars': 42,
        'action': 'production',
        'reason': 'semantic_tags_detected',
        'tags': ['domain:autodrive', 'project:egomotion4d'],
        'observation_scopes': [['domain:autodrive']],
    }])
    hardening.write_text(json.dumps({
        'zero_unit_report': {
            'high_value_retry_candidates': [{
                'document_id': 'hermes-session::s2',
                'zero_unit_class': 'extraction_too_strict_candidate',
                'recommended_route': 'production_windowed',
                'primary_value_classes': ['experiment_result', 'error_root_cause'],
                'semantic_score': 37,
            }]
        }
    }, ensure_ascii=False), encoding='utf-8')

    rows, summary = backlog.build_backlog(
        manifest_path=manifest,
        hardening_paths=[str(hardening)],
        retry_gate_paths=[],
        unit_counts={'hermes-session::s2': {'memory_unit_count': 0}},
        bank='hermes',
    )

    assert rows[0]['event_date'] == '2026-05-02T10:00:00'
    assert rows[0]['current_retain_outcome']['status'] == 'zero_units'
    assert rows[0]['review']['recommended_route'] == 'cluster_revisit'
    assert rows[0]['hardening_overlay']['zero_unit_class'] == 'extraction_too_strict_candidate'
    assert rows[0]['value_class_guess'] == ['experiment_result', 'error_root_cause']


def test_review_backlog_sampler_outputs_content_free_stratified_sample(tmp_path):
    sampler = load_module('hindsight_review_backlog_sampler')
    rows = []
    for i in range(6):
        route = 'cluster_revisit' if i < 3 else 'monitor'
        status = 'zero_units' if i < 3 else 'has_units'
        rows.append({
            'document_id': f'hermes-session::s{i}',
            'event_date': '2026-05-01T00:00:00',
            'topic_key': 'domain:hindsight' if i % 2 else 'domain:autodrive',
            'content_chars': 100 + i,
            'current_retain_outcome': {'status': status, 'memory_unit_count': 0 if status == 'zero_units' else 2},
            'review': {'recommended_route': route},
            'retry_evidence': [{'bank': 'tmp'}] if i == 0 else [],
            'value_class_guess': ['tool_lesson'],
        })

    sample, summary = sampler.select_sample(rows, size=4, quotas={'zero_with_retry_evidence': 1, 'zero_without_retry_evidence': 2, 'monitor_has_units': 1})

    assert len(sample) == 4
    assert summary['contains_content_fields'] == 0
    assert summary['with_event_date'] == 4
    assert any(r['sample']['bucket'] == 'zero_with_retry_evidence' for r in sample)
    assert all(r['sample']['llm_call_allowed'] is False for r in sample)
    assert all(r['sample']['hindsight_submit_allowed'] is False for r in sample)


def test_review_backlog_llm_scorer_batches_records_with_default_weekly_10_package_cap():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    rows = [{'document_id': f'doc-{i}', 'content_chars': 100, 'event_date': '2026-05-01T00:00:00'} for i in range(31)]

    batches, summary = scorer.plan_scorer_batches(rows, batch_size=2)

    assert summary['cadence'] == 'weekly'
    assert summary['max_llm_calls'] == 10
    assert summary['llm_calls_planned'] == 10
    assert summary['records_planned'] == 20
    assert summary['records_deferred_by_call_cap'] == 11
    assert summary['capped_by_max_llm_calls'] is True
    assert len(batches) == 10
    assert all(batch['llm_call_index'] == i + 1 for i, batch in enumerate(batches))
    assert all(len(batch['records']) == 2 for batch in batches)


def test_review_backlog_llm_scorer_allows_configurable_call_cap_above_weekly_default():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    rows = [{'document_id': f'doc-{i}', 'content_chars': 100, 'event_date': '2026-05-01T00:00:00'} for i in range(151)]

    batches, summary = scorer.plan_scorer_batches(rows, batch_size=2, max_llm_calls=50)

    assert summary['max_llm_calls'] == 50
    assert summary['llm_calls_planned'] == 50
    assert summary['records_planned'] == 100
    assert summary['records_deferred_by_call_cap'] == 51
    assert [batch['record_count'] for batch in batches] == [2] * 50


def test_review_backlog_llm_scorer_normalizes_response_to_sidecar_schema():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    record = {
        'document_id': 'hermes-session::s1',
        'event_date': '2026-05-01T00:00:00',
        'content_sha256': 'abc123',
        'review': {'recommended_route': 'cluster_revisit'},
    }
    raw = {
        'document_id': 'hermes-session::s1',
        'value_level': 7,
        'information_density': -2,
        'durability': 4,
        'actionability': 3,
        'topic': ['hindsight', 'memory'],
        'value_classes': ['tool_lesson', 'unknown_class'],
        'retainability_risk': 'invalid',
        'recommended_route': 'repair_note_candidate',
        'anomalies': ['tool_log_heavy', 'unknown_anomaly'],
        'reason_brief': 'source backed',
        'suggested_spans': [{'start': 1, 'end': 2}],
    }

    out = scorer.normalize_score(record, raw, batch_id='b1', llm_call_index=1)

    assert out['schema_version'] == 'hindsight-review-backlog-llm-score-v1'
    assert out['document_id'] == 'hermes-session::s1'
    assert out['event_date'] == '2026-05-01T00:00:00'
    assert out['content_sha256'] == 'abc123'
    assert out['scores']['value_level'] == 5
    assert out['scores']['information_density'] == 0
    assert out['scores']['durability'] == 4
    assert out['scores_normalized']['value_level'] == 1.0
    assert out['scores_normalized']['information_density'] == 0.0
    assert out['scores_normalized']['durability'] == 0.8
    assert out['scores_normalized']['actionability'] == 0.6
    assert out['score_total_0_20'] == 12
    assert out['score_mean_0_1'] == 0.6
    assert out['value_classes'] == ['tool_lesson']
    assert out['retainability_risk'] == 'medium'
    assert out['recommended_route'] == 'repair_note_candidate'
    assert out['anomalies'] == ['tool_log_heavy']
    assert out['hindsight_submit_allowed'] is False
    assert out['production_mutation_allowed'] is False


def test_review_backlog_llm_scorer_dry_run_does_not_call_llm():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    rows = [{'document_id': 'doc-1', 'content': 'User: 需要评估。', 'event_date': '2026-05-01T00:00:00'}]
    batches, _ = scorer.plan_scorer_batches(rows, batch_size=1, max_llm_calls=1)
    calls = {'n': 0}

    def fake_llm(_messages):
        calls['n'] += 1
        return {'scores': []}

    scores, summary = scorer.score_batches(batches, execute=False, confirm='', llm_fn=fake_llm)

    assert scores == []
    assert calls['n'] == 0
    assert summary['llm_calls_made'] == 0
    assert summary['execute'] is False


def test_review_backlog_llm_scorer_execute_requires_confirm():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    rows = [{'document_id': 'doc-1', 'content': 'User: 需要评估。', 'event_date': '2026-05-01T00:00:00'}]
    batches, _ = scorer.plan_scorer_batches(rows, batch_size=1, max_llm_calls=1)

    try:
        scorer.score_batches(batches, execute=True, confirm='', llm_fn=lambda _messages: {'scores': []})
    except SystemExit as exc:
        assert 'confirm' in str(exc).lower()
    else:
        raise AssertionError('execute scoring should require explicit confirm token')


def test_review_backlog_llm_scorer_prompt_requires_all_document_ids():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    prompt_records = [
        {'document_id': 'doc-1', 'content_excerpt': 'User: A'},
        {'document_id': 'doc-2', 'content_excerpt': 'User: B'},
    ]

    messages = scorer.build_scorer_messages({'batch_id': 'b1'}, prompt_records)
    payload = json.loads(messages[1]['content'])

    assert payload['required_document_ids'] == ['doc-1', 'doc-2']
    assert payload['required_score_count'] == 2
    assert any('every document_id' in c for c in payload['constraints'])
    assert payload['final_check_before_answer']['scores_length_must_equal'] == 2


def test_review_backlog_llm_scorer_accepts_scores_by_document_id_map():
    scorer = load_module('hindsight_review_backlog_llm_scorer')

    rows = scorer.scores_from_llm_obj({
        'scores_by_document_id': {
            'doc-1': {'value_level': 3, 'recommended_route': 'wait'},
            'doc-2': {'document_id': 'doc-2', 'value_level': 4, 'recommended_route': 'cluster_revisit'},
        }
    })

    assert {r['document_id'] for r in rows} == {'doc-1', 'doc-2'}


def test_review_backlog_llm_scorer_reports_missing_score_coverage():
    scorer = load_module('hindsight_review_backlog_llm_scorer')
    rows = [
        {'document_id': 'doc-1', 'content': 'User: A', 'event_date': '2026-05-01T00:00:00'},
        {'document_id': 'doc-2', 'content': 'User: B', 'event_date': '2026-05-01T00:00:00'},
    ]
    batches, _ = scorer.plan_scorer_batches(rows, batch_size=2, max_llm_calls=1)

    def fake_llm(_messages):
        return {'scores': [{'document_id': 'doc-1', 'value_level': 3, 'recommended_route': 'wait'}]}

    scores, summary = scorer.score_batches(
        batches,
        execute=True,
        confirm='score-review-backlog',
        llm_fn=fake_llm,
    )

    assert len(scores) == 2
    assert summary['records_prompted_to_llm'] == 2
    assert summary['valid_scores_from_llm'] == 1
    assert summary['missing_scores_from_llm'] == 1
    assert summary['missing_document_ids'] == ['doc-2']
    assert summary['score_coverage'] == 0.5
    assert summary['coverage_ok'] is False
