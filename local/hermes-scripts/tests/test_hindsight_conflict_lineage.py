import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


core = load_module('hindsight_conflict_core')
audit = load_module('hindsight_conflict_audit')


def test_offline_consolidation_alias_matches_hash_changed_document():
    old = 'hermes-offline-consolidation::daily::2026-04-09::general::00::eb55dba6f5fd'
    new = 'hermes-offline-consolidation::daily::2026-04-09::general::00::9a3e90ffbdbd'
    alias = core.offline_consolidation_doc_alias(new)

    assert alias
    assert core._source_ref_known(old, {alias}, set(), set()) is True


def test_partial_dangling_source_doc_is_nonblocking_when_another_source_is_traceable():
    obs = {
        'id': 'obs1',
        'insight': 'durable fact',
        'source_documents': [
            'hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8',
            'hermes-sqlite::day-topic::2026-03-21__general::0001::adae52fe8fb8',
        ],
        'evidence_ids': [],
    }
    cases = core.build_conflict_cases(
        [obs],
        known_document_ids={'hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8'},
        known_memory_ids=set(),
        known_file_paths=set(),
    )

    assert [c['type'] for c in cases] == ['partial_dangling_source_document']
    assert cases[0]['severity'] == 'P3'


def test_stale_memory_uuid_is_nonblocking_when_source_document_is_traceable():
    obs = {
        'id': 'obs2',
        'insight': 'durable fact',
        'source_documents': ['hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8'],
        'evidence_ids': ['fact_id=57cefe48-6865-45e5-a5f8-87e644c7058a'],
    }
    cases = core.build_conflict_cases(
        [obs],
        known_document_ids={'hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8'},
        known_memory_ids={'another-id'},
        known_file_paths=set(),
    )

    assert [c['type'] for c in cases] == ['stale_evidence_id']
    assert cases[0]['severity'] == 'P3'


def test_local_lineage_ids_from_daily_weekly_json(tmp_path):
    p = tmp_path / 'daily' / '2026-03-21' / 'general__00__50215bca9ea8.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        'document_id': 'hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8',
        'unit': {'source_ids': ['57cefe48-6865-45e5-a5f8-87e644c7058a']},
        'llm_json': {'canonical_observations': [
            {'evidence_ids': ['fact_id=4354d4dc-e707-4c35-a61c-f2108d663627']}
        ]},
    }), encoding='utf-8')

    docs, mems = audit.known_local_lineage_ids(tmp_path)

    assert 'hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8' in docs
    assert core.offline_consolidation_doc_alias('hermes-offline-consolidation::daily::2026-03-21::general::00::50215bca9ea8') in docs
    assert '57cefe48-6865-45e5-a5f8-87e644c7058a' in mems
    assert '4354d4dc-e707-4c35-a61c-f2108d663627' in mems
