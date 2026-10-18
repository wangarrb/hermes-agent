import importlib.util
import sys
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_zero_unit_classifier_uses_generic_structure_not_project_names():
    mod = load_module('hindsight_session_quality_hardening')
    doc = {
        'id': 'doc-high-value',
        'original_text': 'User: 我决定废弃旧路线。Assistant: 结论：根因是配置没有生效，修复方案是改为稳定的生产流程。需要后续验证指标。',
        'tags': ['domain:anything'],
        'unit_count': 0,
        'retain_params': {'metadata': {'model': 'test', 'message_count': 4, 'cleaning_stats': {'kept_messages': 4, 'dropped_messages': 0}}},
    }
    out = mod.classify_doc(doc)
    assert out['features']['semantic_score'] >= 4
    assert out['zero_unit_class'] == 'extraction_too_strict_candidate'
    assert out['recommended_route'] == 'retry_custom_mission'
    assert 'durable_decision' in out['features']['primary_value_classes']


def test_zero_unit_classifier_routes_noisy_high_value_to_windowing():
    mod = load_module('hindsight_session_quality_hardening')
    noisy = '\n'.join([
        'RUN: python script.py', 'EXIT 0', 'STDOUT {...}', 'Traceback placeholder',
        '/home/user/project/file.py sha256 abcdef1234567890abcdef1234567890',
    ] * 5)
    text = noisy + '\n结论：实验失败，根因是服务配置错误，下一步需要修复并验证。'
    doc = {'id': 'doc-noisy', 'original_text': text, 'tags': ['topic:generic'], 'unit_count': 0}
    out = mod.classify_doc(doc)
    assert out['zero_unit_class'] == 'noisy_high_value_transcript'
    assert out['recommended_route'] == 'production_windowed'
    assert out['features']['noise']['transcript_noise_ratio'] >= 0.35


def test_recall_scoring_is_query_term_based_and_reports_dominance():
    mod = load_module('hindsight_session_quality_hardening')
    rows = [
        {'type': 'world', 'doc_prefix': 'a', 'tags': ['topic:memory'], 'text': 'unrelated memory management fact'},
        {'type': 'world', 'doc_prefix': 'a', 'tags': ['project:openclaw'], 'text': 'OpenClaw gateway probe returned No session found'},
        {'type': 'world', 'doc_prefix': 'b', 'tags': ['topic:memory'], 'text': 'another unrelated fact'},
    ]
    out = mod.score_recall_rows('OpenClaw gateway probe No session found', rows)
    assert out['k'] == 3
    assert out['relevant_count'] == 1
    assert out['precision_at_k'] == 0.3333
    assert out['first_relevant_rank'] == 2
    assert out['mrr'] == 0.5
    assert out['dominant_tag'] == 'topic:memory'
    assert out['dominant_tag_ratio'] == 0.6667


def test_zero_unit_summary_picks_high_value_retry_candidates():
    mod = load_module('hindsight_session_quality_hardening')
    docs = [
        mod.classify_doc({'id': 'a', 'original_text': '决定采用新方案，根因已定位，需要验证。', 'unit_count': 0}),
        mod.classify_doc({'id': 'b', 'original_text': 'hi', 'unit_count': 0}),
        mod.classify_doc({'id': 'c', 'original_text': '正常事实', 'unit_count': 2}),
    ]
    report = mod.summarize_zero_units(docs)
    assert report['total_documents'] == 3
    assert report['zero_unit_documents'] == 2
    assert report['by_zero_unit_class']['extraction_too_strict_candidate'] == 1
    assert report['by_zero_unit_class']['true_low_signal'] == 1
    assert report['high_value_retry_candidate_count'] == 1


def test_manifest_derived_benchmark_candidates_are_generic_tag_based():
    mod = load_module('hindsight_session_quality_hardening')
    docs = [
        mod.classify_doc({'id': 'a', 'original_text': '决定采用稳定方案，根因已定位。', 'tags': ['domain:memory'], 'unit_count': 2}),
        mod.classify_doc({'id': 'b', 'original_text': '实验结果通过，需要继续验证。', 'tags': ['domain:memory'], 'unit_count': 0}),
        mod.classify_doc({'id': 'c', 'original_text': '其他内容。', 'tags': ['domain:other'], 'unit_count': 1}),
    ]
    out = mod.build_manifest_derived_benchmark_candidates(docs)
    assert out[0]['id'] == 'derived_tag_domain_memory'
    assert out[0]['tag'] == 'domain:memory'
    assert out[0]['support_doc_count'] == 2
    assert out[0]['unit_doc_count'] == 1
    assert out[0]['zero_unit_doc_count'] == 1
