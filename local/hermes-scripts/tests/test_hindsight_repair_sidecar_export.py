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


def test_repair_sidecar_exports_clean_observation_with_source_lineage():
    exporter = load_module('hindsight_repair_sidecar_export')
    memories = [
        {
            'id': 'src-1',
            'fact_type': 'experience',
            'document_id': 'hermes-session::s1',
            'text': 'source fact',
            'tags': ['domain:hindsight', 'value:tool_lesson'],
        },
        {
            'id': 'obs-1',
            'fact_type': 'observation',
            'document_id': None,
            'text': 'Clean repair-zone lesson about safe sidecar promotion.',
            'tags': ['domain:hindsight', 'value:tool_lesson'],
            'source_memory_ids': ['src-1'],
            'proof_count': 1,
        },
    ]
    sidecar = {
        'docs': [
            {
                'document_id': 'hermes-session::s1',
                'parent_document_id': 'hermes-session::s1',
                'candidate': {'candidate_id': 'a-1', 'topic_group': 'hermes_hindsight_ops'},
                'variant': 'clustered_repair_note_then_temp_retain',
            }
        ]
    }

    result = exporter.build_sidecar_records(memories=memories, sidecar=sidecar, bank='hermes_tmp_demo')

    assert result['counts']['approved'] == 1
    assert result['counts']['rejected'] == 0
    rec = result['approved'][0]
    assert rec['layer'] == 'approved_repair_sidecar'
    assert rec['insight'] == 'Clean repair-zone lesson about safe sidecar promotion.'
    assert rec['evidence_ids'] == ['src-1']
    assert rec['source_documents'] == ['hermes-session::s1']
    assert rec['topic'] == 'hindsight'
    assert rec['type'] == 'tooling_lesson'
    assert rec['status'] == 'approved'
    assert rec['provenance']['candidate_id'] == 'a-1'
    assert rec['provenance']['source_bank_hash']
    assert 'hermes_tmp_demo' not in json.dumps(rec, ensure_ascii=False)


def test_repair_sidecar_rejects_artifact_leaks_and_missing_sources():
    exporter = load_module('hindsight_repair_sidecar_export')
    memories = [
        {
            'id': 'obs-artifact',
            'fact_type': 'observation',
            'document_id': None,
            'text': 'This leaks hermes_tmp_demo into the fact text.',
            'tags': ['domain:hindsight'],
            'source_memory_ids': ['missing-src'],
        },
        {
            'id': 'obs-nosource',
            'fact_type': 'observation',
            'document_id': None,
            'text': 'Clean text but no usable source lineage.',
            'tags': ['domain:hindsight'],
            'source_memory_ids': [],
        },
    ]

    result = exporter.build_sidecar_records(memories=memories, sidecar={'docs': []}, bank='hermes_tmp_demo')

    assert result['counts']['approved'] == 0
    reasons = {r['id']: r['reason'] for r in result['rejected']}
    assert reasons['obs-artifact'].startswith('artifact_flags:')
    assert reasons['obs-nosource'] == 'missing_source_memory_ids'
