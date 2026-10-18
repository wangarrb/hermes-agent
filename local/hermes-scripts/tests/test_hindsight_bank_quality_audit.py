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


def test_bank_quality_audit_detects_tag_pollution_and_lineage_gaps(tmp_path):
    audit = load_module('hindsight_bank_quality_audit')
    docs = [
        {'id': 'hermes-sqlite::a'},
        {'id': 'hermes-offline-canonical::b'},
        {'id': 'empty-doc'},
    ]
    memories = [
        {
            'id': 'fact-egopatent',
            'type': 'experience',
            'document_id': 'hermes-sqlite::a',
            'text': 'PCN221584I 专利 OA1 审查意见 权利要求',
            'tags': ['egomotion4d', 'hermes', 'sqlite', 'incremental'],
            'observation_scopes': None,
        },
        {
            'id': 'fact-open-e4d',
            'type': 'world',
            'document_id': 'missing-doc',
            'text': 'Egomotion4D VGGT DAGE ATE trajectory issue',
            'tags': ['openclaw', 'hermes'],
            'observation_scopes': None,
        },
        {
            'id': 'obs-native',
            'type': 'observation',
            'document_id': None,
            'text': 'OpenClaw approval No session found',
            'tags': ['egomotion4d', 'hermes', 'sqlite', 'incremental'],
            'source_memory_ids': ['fact-egopatent', 'missing-source'],
            'proof_count': 1,
            'observation_scopes': None,
        },
        {
            'id': 'obs-canonical',
            'type': 'observation',
            'document_id': 'hermes-offline-canonical::b',
            'text': 'Useful canonical observation',
            'tags': ['offline-v2', 'canonical', 'observation', 'topic', 'egomotion4d'],
            'source_memory_ids': ['fact-egopatent', 'fact-open-e4d'],
            'proof_count': 1,
            'observation_scopes': [['project:egomotion4d']],
        },
    ]

    result = audit.audit_collections(memories=memories, documents=docs, operations=[])

    assert result['counts']['memory_units'] == 4
    assert result['counts']['documents'] == 3
    assert result['tag_quality']['broad_system_tag_counts']['hermes'] == 3
    assert result['contamination_counts']['egomotion_tag_patent_terms'] == 1
    assert result['contamination_counts']['egomotion_tag_openclaw_terms'] == 1
    assert result['contamination_counts']['openclaw_tag_egomotion_terms'] == 1
    assert result['lineage']['missing_source_refs'] == 1
    assert result['lineage']['observations_with_missing_source'] == 1
    assert result['lineage']['docs_without_units'] == 1
    assert result['lineage']['units_missing_document'] == 1
    assert result['observation_quality']['source_count_distribution']['2'] == 2
    assert result['observation_quality']['proof_count_distribution']['1'] == 2
    assert result['observation_quality']['observation_doc_prefixes']['<null>'] == 1
    assert result['scope_quality']['null_observation_scopes'] == 3


def test_run_audit_uses_db_fallback_when_api_lineage_fields_are_missing(tmp_path):
    audit = load_module('hindsight_bank_quality_audit')

    class SparseApiClient:
        def health(self):
            return {'status': 'healthy'}
        def stats(self):
            return {'pending_operations': 0, 'processing_operations': 0, 'failed_operations': 0}
        def get_config(self):
            return {'config': {}}
        def list_all_documents(self, max_items=None):
            return [{'id': 'doc-1'}]
        def iter_memories(self, types=None, max_items=None):
            # API list endpoint omits document_id/source_memory_ids on this Hindsight version.
            yield {'id': 'fact-1', 'fact_type': 'world', 'text': 'fact', 'tags': ['hermes']}
            yield {'id': 'obs-1', 'fact_type': 'observation', 'text': 'obs', 'tags': ['hermes']}
        def iter_operations(self, max_items=None):
            return iter([])

    def fake_db_loader(bank):
        return {
            'documents': [{'id': 'doc-1'}],
            'memories': [
                {'id': 'fact-1', 'fact_type': 'world', 'document_id': 'doc-1', 'text': 'fact', 'tags': ['hermes']},
                {'id': 'obs-1', 'fact_type': 'observation', 'document_id': None, 'text': 'obs', 'tags': ['hermes'], 'source_memory_ids': ['fact-1']},
            ],
            'operations': [],
            'source': 'postgresql',
        }

    result = audit.run_audit(client=SparseApiClient(), bank='hermes', db_fallback='auto', db_loader=fake_db_loader)

    assert result['data_source'] == 'postgresql_fallback'
    assert result['audit']['lineage']['source_refs'] == 1
    assert result['audit']['lineage']['docs_without_units'] == 0
    assert result['audit']['composition']['memory_units_by_doc_prefix']['doc-1'] == 1


def test_run_audit_uses_db_fallback_when_api_document_ids_do_not_match_memory_lineage(tmp_path):
    audit = load_module('hindsight_bank_quality_audit')

    class MismatchedApiClient:
        def health(self):
            return {'status': 'healthy'}
        def stats(self):
            return {'pending_operations': 0, 'processing_operations': 0, 'failed_operations': 0}
        def get_config(self):
            return {'config': {}}
        def list_all_documents(self, max_items=None):
            # Simulates an API page exposing opaque ids instead of durable document_id.
            return [{'id': 'opaque-api-doc-id'}]
        def iter_memories(self, types=None, max_items=None):
            yield {'id': 'fact-1', 'fact_type': 'world', 'document_id': 'doc-1', 'text': 'fact', 'tags': ['hermes']}
        def iter_operations(self, max_items=None):
            return iter([])

    def fake_db_loader(bank):
        return {
            'documents': [{'id': 'doc-1'}],
            'memories': [{'id': 'fact-1', 'fact_type': 'world', 'document_id': 'doc-1', 'text': 'fact', 'tags': ['hermes']}],
            'operations': [],
            'source': 'postgresql',
        }

    result = audit.run_audit(client=MismatchedApiClient(), bank='hermes', db_fallback='auto', db_loader=fake_db_loader)

    assert result['data_source'] == 'postgresql_fallback'
    assert result['audit']['lineage']['docs_without_units'] == 0


def test_bank_quality_audit_recall_smoke_and_report_writing(tmp_path):
    audit = load_module('hindsight_bank_quality_audit')

    class FakeClient:
        def health(self):
            return {'status': 'healthy'}
        def stats(self):
            return {'pending_operations': 0, 'processing_operations': 0, 'failed_operations': 0}
        def get_config(self):
            return {'config': {'enable_observations': False}}
        def list_all_documents(self, max_items=None):
            return [{'id': 'doc-1'}]
        def iter_memories(self, types=None, max_items=None):
            yield {'id': 'm1', 'type': 'world', 'document_id': 'doc-1', 'text': 'Hindsight memory fact', 'tags': ['domain:hindsight']}
        def iter_operations(self, max_items=None):
            return iter([])
        def recall(self, query, types=None, limit=None, budget='mid', max_tokens=2048):
            return {'results': [
                {'type': 'world', 'document_id': 'doc-1', 'tags': ['domain:hindsight'], 'text': 'Hindsight session json retain fact with key sk-test-abcdefghijklmnopqrstuvwxyz123456'}
            ]}

    result = audit.run_audit(client=FakeClient(), bank='hermes', recall_smoke=True)
    assert result['runtime']['health']['status'] == 'healthy'
    assert result['recall_smoke']['hindsight_arch']['count'] == 1

    paths = audit.write_reports(result, tmp_path, stem='audit-test')
    assert Path(paths['json']).exists()
    assert Path(paths['md']).exists()
    saved = json.loads(Path(paths['json']).read_text(encoding='utf-8'))
    assert saved['bank'] == 'hermes'
    assert '[REDACTED]' in json.dumps(saved, ensure_ascii=False)
    assert 'sk-tes...' not in json.dumps(saved, ensure_ascii=False)
    md = Path(paths['md']).read_text(encoding='utf-8')
    assert 'Hindsight bank quality audit' in md
    assert 'hindsight_arch' in md
