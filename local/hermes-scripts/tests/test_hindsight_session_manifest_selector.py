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


def test_selector_keeps_only_production_and_dedupes_session_roots():
    selector = load_module('hindsight_session_manifest_selector')
    records = [
        {
            'document_id': 'hermes-session::a::part-000',
            'action': 'production',
            'content_chars': 200,
            'estimated_retain_chunks': 1,
            'tags': ['domain:hindsight'],
            'metadata': {'session_id': 'a'},
        },
        {
            'document_id': 'hermes-session::a::part-001',
            'action': 'production',
            'content_chars': 900,
            'estimated_retain_chunks': 1,
            'tags': ['domain:hindsight'],
            'metadata': {'session_id': 'a'},
        },
        {
            'document_id': 'hermes-session::b',
            'action': 'manual_review',
            'content_chars': 5000,
            'estimated_retain_chunks': 1,
            'tags': ['domain:hindsight'],
            'metadata': {'session_id': 'b'},
        },
        {
            'document_id': 'hermes-session::c',
            'action': 'production',
            'content_chars': 300,
            'estimated_retain_chunks': 1,
            'tags': ['project:egomotion4d', 'domain:autodrive'],
            'metadata': {'session_id': 'c'},
        },
    ]

    selected = selector.select_records(records, limit=10)

    assert [r['document_id'] for r in selected] == ['hermes-session::c', 'hermes-session::a::part-001']


def test_selector_writes_manifest_and_summary_without_rehydrating_content(tmp_path):
    selector = load_module('hindsight_session_manifest_selector')
    records = [
        {
            'document_id': 'hermes-session::a',
            'action': 'production',
            'reason': 'semantic_tags_detected',
            'content_chars': 200,
            'estimated_retain_chunks': 1,
            'tags': ['domain:hindsight'],
            'metadata': {'session_id': 'a', 'candidate_filter_version': 'lightweight-candidate-filter-v2'},
            'content_omitted': True,
        },
    ]

    paths = selector.write_curated_manifest(records, tmp_path, stem='curated-test')

    manifest_line = json.loads(Path(paths['manifest']).read_text(encoding='utf-8').strip())
    summary = json.loads(Path(paths['summary']).read_text(encoding='utf-8'))
    assert manifest_line['document_id'] == 'hermes-session::a'
    assert 'content' not in manifest_line
    assert manifest_line['content_omitted'] is True
    assert summary['selected_records'] == 1
    assert summary['by_tag']['domain:hindsight'] == 1
