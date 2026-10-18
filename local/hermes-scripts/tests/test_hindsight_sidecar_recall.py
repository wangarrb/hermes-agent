import importlib.util
import json
from pathlib import Path

ROOT = Path('/home/wyr/.hermes/scripts')


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_cards_root(tmp_path):
    root = tmp_path / 'cards'
    root.mkdir()
    obs = {
        'id': 'obs:local',
        'topic': 'topic-a',
        'type': 'technical_lesson',
        'confidence': 0.9,
        'insight': 'local sidecar carries special-term evidence',
        'evidence_ids': ['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'],
        'tags': ['special-term'],
    }
    (root / 'observations_index.jsonl').write_text(json.dumps(obs) + '\n', encoding='utf-8')
    return root


def make_repair_sidecar_root(tmp_path):
    root = tmp_path / 'repair'
    root.mkdir()
    obs = {
        'id': 'repair:obs',
        'layer': 'approved_repair_sidecar',
        'topic': 'topic-a',
        'type': 'tooling_lesson',
        'status': 'approved',
        'insight': 'repair sidecar carries repair-term evidence',
        'evidence_ids': ['ffffffff-bbbb-cccc-dddd-eeeeeeeeeeee'],
        'source_documents': ['hermes-session::repair-source'],
        'tags': ['repair-term'],
    }
    (root / 'observations_index.jsonl').write_text(json.dumps(obs) + '\n', encoding='utf-8')
    return root


def test_layered_recall_appends_local_sidecar_without_replacing_nonlocal(monkeypatch, tmp_path):
    layered = load_module('hindsight_recall_layered')
    cards_root = make_cards_root(tmp_path)

    def fake_recall(api, bank, query, limit, include_observations=True):
        return [
            {'id': 'n1', 'document_id': 'hermes-sqlite::1', 'type': 'world', 'text': 'nonlocal one topic-a'},
            {'id': 'n2', 'document_id': 'hermes-sqlite::2', 'type': 'world', 'text': 'nonlocal two topic-a'},
        ][:limit]

    monkeypatch.setattr(layered, 'recall', fake_recall)
    results = layered.layered_recall('api', 'bank', 'special-term topic-a', 'mixed', raw_limit=2, limit=2, cards_root=cards_root, include_observations=True, local_sidecar_limit=1)
    assert [r['id'] for r in results[:2]] == ['n1', 'n2']
    assert len(results) == 3
    assert results[2]['layer'] == 'local_canonical'
    assert results[2]['_sidecar'] is True


def test_layered_recall_appends_approved_repair_sidecar_without_replacing_nonlocal(monkeypatch, tmp_path):
    layered = load_module('hindsight_recall_layered')
    repair_root = make_repair_sidecar_root(tmp_path)

    def fake_recall(api, bank, query, limit, include_observations=True):
        return [
            {'id': 'n1', 'document_id': 'hermes-sqlite::1', 'type': 'world', 'text': 'nonlocal one topic-a'},
            {'id': 'n2', 'document_id': 'hermes-sqlite::2', 'type': 'world', 'text': 'nonlocal two topic-a'},
        ][:limit]

    monkeypatch.setattr(layered, 'recall', fake_recall)
    results = layered.layered_recall(
        'api', 'bank', 'repair-term topic-a', 'mixed', raw_limit=2, limit=2,
        cards_root=None, repair_sidecar_root=repair_root, include_observations=True,
        repair_sidecar_limit=1,
    )
    assert [r['id'] for r in results[:2]] == ['n1', 'n2']
    assert len(results) == 3
    assert results[2]['layer'] == 'approved_repair_sidecar'
    assert results[2]['_sidecar'] is True
    assert results[2]['sidecar_reason'] == 'approved_repair_source_preserving_augmentation'


def test_eval_top_k_preserves_nonlocal_and_adds_sidecar():
    eval_mod = load_module('hindsight_eval')
    case = {'expected_terms': ['special-term'], 'expected_layers': ['canonical']}
    results = [
        {'id': 'n1', 'document_id': 'hermes-sqlite::1', 'layer': 'sqlite_import', 'text': 'nonlocal one'},
        {'id': 'n2', 'document_id': 'hermes-sqlite::2', 'layer': 'sqlite_import', 'text': 'nonlocal two'},
        {'id': 'l1', 'document_id': 'offline-v2-observation::topic::obs', 'layer': 'local_canonical', '_sidecar': True, 'text': 'special-term from sidecar', 'source_fact_ids': ['aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee']},
    ]
    scored = eval_mod.evaluate_results(case, results, k=2)
    assert scored['evaluated_non_sidecar_count'] == 2
    assert scored['evaluated_sidecar_count'] == 1
    assert scored['term_recall'] == 1.0
    assert scored['expected_layer_hits_top_k'] == 1
