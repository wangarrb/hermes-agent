"""Publisher selects frozen exact artifacts, never stale directory globs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path('/home/wyr/.hermes/cache/scratch')


@pytest.fixture
def work():
    return Path(tempfile.mkdtemp(prefix='exact-publish-test-', dir=SCRATCH))


@pytest.fixture
def publisher(monkeypatch):
    # Import-only dependency; any DB/embedding call fails the test.
    def forbidden(*_a, **_kw):
        raise AssertionError('production calls forbidden in manifest tests')
    extras = types.SimpleNamespace(RealDictCursor=object, execute_values=forbidden)
    monkeypatch.setitem(sys.modules, 'psycopg2', types.SimpleNamespace(connect=forbidden, extras=extras))
    monkeypatch.setitem(sys.modules, 'psycopg2.extras', extras)
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('publish_exact_manifest', ROOT / 'hindsight_offline_v2_publish.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy(work, *, extra=False):
    root = work / 'legacy'
    (root / 'topics').mkdir(parents=True)
    obs = {'id': 'obs-a', 'insight': 'canonical fact', 'topic': 'geometry', 'type': 'lesson',
           'applicability': 'static', 'evidence_ids': ['fact-a']}
    card = {'card_id': 'card-a', 'scope': 'topic', 'topic': 'geometry',
            'canonical_observations': [obs]}
    (root / 'topics' / 'geometry.json').write_text(json.dumps(card))
    manifest = {'card_count': 1, 'observation_index_count': 1,
                'cards': [{'card_id': 'card-a', 'scope': 'topic', 'topic': 'geometry', 'observation_count': 1}]}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'observations_index.jsonl').write_text(json.dumps(obs) + '\n')
    if extra:
        (root / 'topics' / 'stale.json').write_text(json.dumps(dict(card, card_id='stale', topic='stale')))
    return root


def test_stale_extra_cards_are_refused_and_preserved(publisher, work):
    root = legacy(work, extra=True)
    before = (root / 'topics' / 'stale.json').read_bytes()
    with pytest.raises(ValueError, match='extra|unexpected|undeclared'):
        publisher.load_cards(root)
    assert (root / 'topics' / 'stale.json').read_bytes() == before


def test_complete_old_reducer_manifest_is_compatible(publisher, work):
    cards = publisher.load_cards(legacy(work))
    assert [c['card_id'] for c in cards] == ['card-a']
    assert [o['id'] for c in cards for o in c['canonical_observations']] == ['obs-a']


@pytest.mark.parametrize('damage', ['bad-jsonl', 'missing-card', 'wrong-card-id', 'count', 'duplicate-id', 'symlink'])
def test_legacy_manifest_requires_complete_valid_exact_artifacts(publisher, work, damage):
    root = legacy(work)
    path = root / 'topics' / 'geometry.json'
    index = root / 'observations_index.jsonl'
    if damage == 'bad-jsonl':
        index.write_text(index.read_text() + '{bad}\n')
    elif damage == 'missing-card':
        path.rename(work / 'preserved-card.json')
    elif damage == 'wrong-card-id':
        value = json.loads(path.read_text())
        value['card_id'] = 'mixed-generation'
        path.write_text(json.dumps(value))
    elif damage == 'count':
        value = json.loads((root / 'manifest.json').read_text())
        value['observation_index_count'] = 99
        (root / 'manifest.json').write_text(json.dumps(value))
    elif damage == 'duplicate-id':
        index.write_text(index.read_text() * 2)
    elif damage == 'symlink':
        saved = work / 'saved-card.json'
        path.rename(saved)
        path.symlink_to(saved)
    with pytest.raises(ValueError):
        publisher.load_cards(root)


def test_immutable_generation_uses_full_index_not_top_n(publisher, work):
    import hindsight_incremental_merge as merge
    rows = [{'insight': f'exact numerical detail {i}', 'topic': 'geometry',
             'type': 'technical_lesson', 'applicability': 'static', 'evidence_ids': [f'fact-{i}']}
            for i in range(135)]
    m = merge.write_generation(work, bank='hermes',
        parent={'bank': 'hermes', 'seed_sha256': hashlib.sha256(b'readonly-seed').hexdigest()},
        observations=rows, consumed_batch_ids=['batch-a'])
    cards = publisher.load_cards(Path(m['root']), bank='hermes')
    assert sum(len(c['canonical_observations']) for c in cards) == len(rows)
    assert any(c['compact_card_observation_count'] < len(c['canonical_observations']) for c in cards)
    with pytest.raises(ValueError, match='bank'):
        publisher.load_cards(Path(m['root']), bank='other')
    index = Path(m['root']) / 'observations_index.jsonl'
    index.write_text(index.read_text().replace('exact numerical detail', 'tampered detail'))
    with pytest.raises(ValueError, match='hash'):
        publisher.load_cards(Path(m['root']), bank='hermes')


def test_opaque_seed_is_not_silently_converted_by_existing_lossy_publisher(publisher, work):
    import hindsight_incremental_merge as merge
    raw = {'bank_id': 'hermes', 'id': 'row-a', 'text': 'opaque | type=old',
           'metadata': {'custom-provenance': ['keep']}}
    seed = merge.import_legacy_seed([raw], bank='hermes',
                                   provenance={'bank': 'hermes', 'seed_sha256': hashlib.sha256(b'seed').hexdigest()})
    m = merge.write_generation(work, bank='hermes',
        parent={'bank': 'hermes', 'seed_sha256': hashlib.sha256(b'seed').hexdigest()},
        observations=seed, consumed_batch_ids=[])
    # Task3 is a local state migration, not a license to rewrite opaque DB rows.
    with pytest.raises(ValueError, match='opaque|legacy'):
        publisher.load_cards(Path(m['root']), bank='hermes')
