"""Lossless sealed seed + deterministic local delta; real scratch artifacts only."""
import copy
import hashlib
import importlib.util
import itertools
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
    return Path(tempfile.mkdtemp(prefix='incremental-merge-test-', dir=SCRATCH))


@pytest.fixture
def merge():
    path = ROOT / 'hindsight_incremental_merge.py'
    assert path.is_file(), 'sealed-state local merge module has not been implemented'
    spec = importlib.util.spec_from_file_location('hindsight_incremental_merge', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def row(**changes):
    value = {'insight': 'Use relative motion, not absolute pose.', 'topic': 'geometry',
             'type': 'technical_lesson', 'applicability': 'static regions',
             'valid_from': '2026-10-01', 'valid_until': None,
             'evidence_ids': ['fact-a'], 'source_documents': ['doc-a'],
             'confidence': 'high', 'tags': ['pose'], 'supersedes': [],
             'open_questions': ['scale?'], 'conflicts': [{'with': 'node-b', 'condition': 'dynamic'}]}
    value.update(changes)
    return value


def parent(bank='hermes'):
    return {'bank': bank, 'seed_sha256': hashlib.sha256(b'authoritative-readonly-seed').hexdigest(),
            'rebuild_id': 'offline-v2-rebuild::20261002-165118::7bce4665'}


@pytest.fixture
def publication(merge, monkeypatch):
    calls = []
    def forbidden(*_a, **_kw):
        calls.append('external I/O')
        raise AssertionError('SQL, network, LLM and embeddings forbidden in Task3 tests')
    extras = types.SimpleNamespace(RealDictCursor=object, execute_values=forbidden)
    monkeypatch.setitem(sys.modules, 'psycopg2', types.SimpleNamespace(connect=forbidden, extras=extras))
    monkeypatch.setitem(sys.modules, 'psycopg2.extras', extras)
    monkeypatch.syspath_prepend(str(ROOT))
    spec = importlib.util.spec_from_file_location('task3_publication_probe', ROOT / 'hindsight_incremental_publication.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = module.PublicationAdapter({'dsn': 'forbidden', 'script_dir': str(ROOT)}, 'hermes',
        connect=forbidden, run_json=forbidden, embeddings=forbidden, execute_values=forbidden)
    return adapter, calls


def generation(merge, work, observations=None, **kwargs):
    return merge.write_generation(work, bank='hermes', parent=parent(),
                                  observations=observations if observations is not None else [row()],
                                  consumed_batch_ids=kwargs.get('consumed_batch_ids', ['batch-a']))


def test_empty_delta_preserves_opaque_seed_and_does_not_mutate(merge):
    seed = [{'text': '原始 | type=unknown | applicability=x', 'metadata': {'opaque': ['retain', 7]}}, row()]
    before = copy.deepcopy(seed)
    assert merge.merge_observations(seed, []) == before
    result = merge.merge_observations(seed, [row(applicability='dynamic regions')])
    assert before[0] in result
    assert seed == before


@pytest.mark.parametrize('field,value', [('topic', 'robotics'), ('type', 'risk'),
    ('applicability', 'dynamic regions'), ('valid_from', '2026-11-01'),
    ('valid_until', '2027-01-01'), ('validity', {'status': 'invalid', 'when': 'dynamic'})])
def test_typed_conditions_are_distinct_nodes(merge, field, value):
    a, b = row(), row(**{field: value})
    assert a in merge.merge_observations([], [a, b])
    assert b in merge.merge_observations([], [a, b])
    assert merge.observation_id(a) != merge.observation_id(b)


def test_merge_is_idempotent_order_independent_and_associative(merge):
    a = row(evidence_ids=['a'], source_documents=['a'], confidence='low')
    b = row(evidence_ids=['b'], source_documents=['b'], confidence='high',
            evidence=[{'source': 'b', 'quote': 'numerical fact', 'score': 0.91}])
    c = row(applicability='dynamic', evidence_ids=['c'])
    batch = merge.merge_observations([], [a, b, c, a])
    assert batch == merge.merge_observations([], [c, b, a])
    assert batch == merge.merge_observations(merge.merge_observations([], [a, c]), [b])
    assert merge.merge_observations(batch, [a, b, c]) == batch
    combined = next(x for x in batch if x['applicability'] == 'static regions')
    assert set(combined['evidence_ids']) == {'a', 'b'}
    assert combined['evidence'][0]['quote'] == 'numerical fact'
    assert len(combined['contributions']) == 2


def test_full_evidence_and_unknown_fields_survive_merge_and_index(merge, work):
    a = row(evidence_ids=[f'fact-{i}' for i in range(85)],
            source_documents=[f'doc-{i}' for i in range(45)],
            evidence=[{'source_id': f's-{i}', 'nested': {'quote': f'exact {i}', 'values': [i, 0.3]}}
                      for i in range(70)], numeric_fact={'unit': 'mm', 'value': 0.125})
    b = row(evidence_ids=['new-fact'], supersedes=['old-observation'],
            open_questions=['what about translation?'])
    merged = merge.merge_observations([], [a, b])
    node = merged[0]
    assert len(node['evidence_ids']) == 86
    assert len(node['source_documents']) == 46
    assert len(node['evidence']) == 70
    assert node['numeric_fact'] == a['numeric_fact']
    manifest = generation(merge, work, merged)
    validated = merge.validate_generation(Path(manifest['manifest_path']))
    assert validated == manifest
    index = Path(manifest['root']) / 'observations_index.jsonl'
    assert [json.loads(line) for line in index.read_text().splitlines()] == merged


def test_semantic_hash_and_node_id_ignore_execution_recipe(merge):
    a = row(model='old', label='week-1', output_path='/old', batch_id='batch-x', id='old-model-id')
    b = row(model='new', label='week-2', output_path='/new', batch_id='batch-y', id='new-model-id')
    assert merge.observation_id(a) == merge.observation_id(b)
    assert merge.observation_content_hash(a) == merge.observation_content_hash(b)
    merged = merge.merge_observations([], [a, b])
    assert len(merged) == 1
    assert len(merged[0]['contributions']) == 1
    assert merge.observation_content_hash(merged[0]) == merge.observation_content_hash(a)


def test_readonly_seed_adapter_preserves_original_rows_without_parsing_text(merge, work):
    raw = {'id': 'old-db-uuid', 'bank_id': 'hermes', 'text': '  中文 | type=looks-like-a-field\n',
           'metadata': {'observation_id': 'shared-old-id', 'evidence_ids': '["fact-z"]',
                        'applicability': '  literal  ', 'unknown': 'keep'},
           'source_memory_ids': ['old-source'], 'document_id': 'old-canonical-document',
           'context': 'exact old context', 'tags': ['canonical'], 'embedding': [0.1, 0.2]}
    original = copy.deepcopy(raw)
    seed = merge.import_legacy_seed([raw], bank='hermes', provenance=parent())
    assert seed[0]['kind'] == 'opaque_legacy'
    assert seed[0]['legacy_row'] == original
    assert raw == original
    assert merge.merge_observations(seed, []) == seed
    manifest = generation(merge, work, merge.merge_observations(seed, [row()]))
    index = Path(manifest['root']) / 'observations_index.jsonl'
    assert any(x['legacy_row'] == original for x in map(json.loads, index.read_text().splitlines())
               if x.get('kind') == 'opaque_legacy')
    projection = merge.publication_observation(seed[0])
    assert projection['insight'] == raw['text']
    assert projection['legacy_row'] == raw
    with pytest.raises(ValueError, match='bank'):
        merge.import_legacy_seed([dict(raw, bank_id='other')], bank='hermes', provenance=parent())


def test_generation_is_immutable_exact_and_keeps_parent_files(merge, work):
    first = generation(merge, work)
    path = Path(first['manifest_path'])
    original_bytes = path.read_bytes()
    original_stat = path.stat().st_mtime_ns
    repeated = generation(merge, work, consumed_batch_ids=['batch-a', 'batch-a'])
    assert repeated == first
    assert path.read_bytes() == original_bytes and path.stat().st_mtime_ns == original_stat
    second = merge.write_generation(work, bank='hermes', parent=first,
        observations=merge.merge_observations([row()], [row(applicability='dynamic')]),
        consumed_batch_ids=['batch-b', 'batch-a'])
    assert second['root'] != first['root']
    assert second['parent_sha256'] == merge.canonical_hash(first)
    assert path.read_bytes() == original_bytes
    assert merge.validate_generation(Path(second['manifest_path'])) == second
    assert {x['path'] for x in second['files']} == {
        str(p.relative_to(second['root'])) for p in Path(second['root']).rglob('*')
        if p.is_file() and p.name != 'manifest.json'}


def test_generation_hash_is_stable_across_order_and_output_root(merge, work):
    rows = [row(), row(applicability='dynamic')]
    a = generation(merge, work / 'one', rows, consumed_batch_ids=['b', 'a'])
    b = generation(merge, work / 'two', list(reversed(rows)), consumed_batch_ids=['a', 'b'])
    assert a['generation_id'] == b['generation_id']
    assert a['observations_sha256'] == b['observations_sha256']
    assert {f['path']: f['sha256'] for f in a['files']} == {f['path']: f['sha256'] for f in b['files']}


def reseal_manifest(merge, path, manifest):
    manifest['manifest_sha256'] = merge.canonical_hash({k: v for k, v in manifest.items()
                                                       if k != 'manifest_sha256'})
    path.write_text(json.dumps(manifest), encoding='utf-8')


@pytest.mark.parametrize('corruption', ['hash', 'jsonl', 'missing', 'extra', 'duplicate-node',
                                       'duplicate-file', 'bank', 'root', 'parent', 'card'])
def test_validation_fails_closed_on_real_artifact_corruption(merge, work, corruption):
    m = generation(merge, work)
    path, root = Path(m['manifest_path']), Path(m['root'])
    index = root / 'observations_index.jsonl'
    if corruption == 'hash':
        index.write_text(index.read_text() + ' ')
    elif corruption == 'jsonl':
        index.write_text('{not-json}\n')
    elif corruption == 'missing':
        # Move rather than delete: preserve all original test artifacts.
        index.rename(work / 'preserved-index.jsonl')
    elif corruption == 'extra':
        (root / 'topics' / 'stale.json').write_text('{}')
    elif corruption == 'duplicate-node':
        index.write_text(index.read_text() * 2)
        m['observation_count'] = 2
    elif corruption == 'duplicate-file':
        m['files'].append(copy.deepcopy(m['files'][0]))
    elif corruption == 'bank':
        m['bank'] = 'other'
    elif corruption == 'root':
        m['root'] = str(work)
    elif corruption == 'parent':
        m['parent']['seed_sha256'] = '0' * 64
    elif corruption == 'card':
        card = next(f for f in m['files'] if f['kind'] == 'card')
        (root / card['path']).write_text('{bad card}')
    if corruption in {'jsonl', 'duplicate-node', 'card'}:
        for f in m['files']:
            p = root / f['path']
            f.update(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size)
    if corruption != 'hash':
        reseal_manifest(merge, path, m)
    with pytest.raises(ValueError):
        merge.validate_generation(path)


@pytest.mark.parametrize('bad_path', ['../outside.json', '/absolute.json', 'topics/../x.json',
                                     'topics//x.json', 'topics/./x.json', 'topics/x\n.json',
                                     'topics\\x.json'])
def test_manifest_rejects_unsafe_raw_paths_before_normalization(merge, work, bad_path):
    m = generation(merge, work)
    path = Path(m['manifest_path'])
    m['files'][0]['path'] = bad_path
    reseal_manifest(merge, path, m)
    with pytest.raises(ValueError, match='path'):
        merge.validate_generation(path)


def test_generation_rejects_symlink_roots_and_artifacts(merge, work):
    real = work / 'real'
    real.mkdir()
    link = work / 'linked'
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match='symlink'):
        generation(merge, link)
    m = generation(merge, real)
    index = Path(m['root']) / 'observations_index.jsonl'
    saved = work / 'saved-index.jsonl'
    index.rename(saved)
    index.symlink_to(saved)
    with pytest.raises(ValueError, match='symlink'):
        merge.validate_generation(Path(m['manifest_path']))


def test_writer_rejects_bank_mismatch_duplicate_ids_and_bad_parent(merge, work):
    with pytest.raises(ValueError, match='bank'):
        merge.write_generation(work, bank='hermes', parent=parent('other'),
                               observations=[row()], consumed_batch_ids=[])
    with pytest.raises(ValueError, match='duplicate'):
        generation(merge, work, [row(id='same'), row(applicability='dynamic', id='same')])
    with pytest.raises(ValueError, match='parent'):
        merge.write_generation(work, bank='hermes', parent={'bank': 'hermes'},
                               observations=[row()], consumed_batch_ids=[])
    assert not (work / 'generations').exists()


def test_parent_artifact_is_bound_not_merely_an_unchecked_dictionary(merge, work):
    first = generation(merge, work)
    second = merge.write_generation(work, bank='hermes', parent=first,
        observations=merge.merge_observations([row()], [row(applicability='dynamic')]),
        consumed_batch_ids=['batch-a', 'batch-b'])
    path = Path(first['manifest_path'])
    original = path.read_bytes()
    path.write_bytes(original + b'\ncorruption')
    with pytest.raises(ValueError, match='parent|JSON'):
        merge.validate_generation(Path(second['manifest_path']))


@pytest.mark.parametrize('changes', [
    {'evidence_ids': {'not': 'a list'}}, {'source_documents': [42]},
    {'contributions': 'not-an-array'}, {'contribution_provenance': [None]},
])
def test_schema_rejects_malformed_evidence_or_contributions_before_writing(merge, work, changes):
    with pytest.raises(ValueError):
        generation(merge, work, [row(**changes)])
    assert not (work / 'generations').exists()


def test_declaring_stale_file_does_not_make_it_a_generation_artifact(merge, work):
    m = generation(merge, work)
    root, path = Path(m['root']), Path(m['manifest_path'])
    stale = root / 'topics' / 'stale.md'
    stale.write_text('stale-card')
    m['files'].append({'path': 'topics/stale.md', 'kind': 'markdown',
                      'size': stale.stat().st_size, 'sha256': hashlib.sha256(stale.read_bytes()).hexdigest()})
    reseal_manifest(merge, path, m)
    with pytest.raises(ValueError, match='artifact|projection'):
        merge.validate_generation(path)


def test_source_contribution_list_order_does_not_invalidate_content_hash(merge):
    a = row(evidence_ids=['x', 'y'], source_documents=['doc-y', 'doc-x'],
            evidence=[{'id': 'x', 'quote': 'a'}, {'id': 'y', 'quote': 'b'}])
    b = row(evidence_ids=['y', 'x'], source_documents=['doc-x', 'doc-y'],
            evidence=list(reversed(a['evidence'])))
    assert merge.observation_id(a) == merge.observation_id(b)
    assert merge.observation_content_hash(a) == merge.observation_content_hash(b)
    merged = merge.merge_observations([], [a, b])
    assert len(merged) == 1 and len(merged[0]['contributions']) == 1


def test_opaque_rows_with_repeated_old_observation_metadata_are_all_preserved(merge, work):
    rows = [{'id': f'uuid-{i}', 'bank_id': 'hermes', 'text': 'identical old text',
             'metadata': {'observation_id': 'duplicate-old-id'}, 'context': f'original-{i}'}
            for i in range(3)]
    seed = merge.import_legacy_seed(rows, bank='hermes', provenance=parent())
    m = generation(merge, work, seed)
    recovered = [x['legacy_row'] for x in map(json.loads,
        (Path(m['root']) / 'observations_index.jsonl').read_text().splitlines())]
    assert sorted(recovered, key=lambda x: x['id']) == rows


def test_aggregate_cannot_drop_original_evidence_even_after_rehashing(merge, work):
    merged = merge.merge_observations([], [row(evidence_ids=['a']), row(evidence_ids=['b'])])
    corrupted = copy.deepcopy(merged)
    corrupted[0]['evidence_ids'] = ['a']
    with pytest.raises(ValueError, match='contribution|aggregate'):
        generation(merge, work, corrupted)
    assert not (work / 'generations').exists()


def test_bad_root_string_and_hardlink_artifact_are_rejected(merge, work):
    m = generation(merge, work)
    path = Path(m['manifest_path'])
    original = copy.deepcopy(m)
    m['root'] = m['root'].replace('/generations/', '//generations/')
    reseal_manifest(merge, path, m)
    with pytest.raises(ValueError, match='path|root'):
        merge.validate_generation(path)
    reseal_manifest(merge, path, original)
    import os
    os.link(Path(m['root']) / 'observations_index.jsonl', work / 'aliased-index.jsonl')
    with pytest.raises(ValueError, match='unaliased'):
        merge.validate_generation(path)


def opaque(merge, **changes):
    raw = {'id': 'exact-db-id', 'bank_id': 'hermes', 'text': '  opaque | type=literal\n',
           'metadata': '{"observation_id":"old-shared-id","unknown":[1,null]}',
           'source_memory_ids': [f'source-{i}' for i in range(65)],
           'embedding': '[0.1,0.2]', 'future_field': {'retain': [None, True, 'exact']}}
    raw.update(changes)
    return merge.import_legacy_seed([raw], bank='hermes', provenance=parent())[0]


def test_opaque_provenance_is_lossless_associative_and_order_independent(merge, work):
    a = opaque(merge)
    b = copy.deepcopy(a)
    b['provenance'] = dict(parent(), export_note={'full': list(range(150))})
    c = copy.deepcopy(a)
    c.update(model='different-recipe', label='second-export', output_path='/not-identity')
    variants = [a, b, c]
    before = copy.deepcopy(variants)
    expected = merge.merge_observations([a], [b, c])
    for order in itertools.permutations(variants):
        assert merge.merge_observations([order[0]], list(order[1:])) == expected
        partial = merge.merge_observations([order[0]], [order[1]])
        assert merge.merge_observations(partial, [order[2]]) == expected
    assert merge.merge_observations(expected, variants * 2) == expected
    assert merge.merge_observations(expected, []) == expected
    assert variants == before
    assert expected[0]['legacy_row'] == a['legacy_row']
    assert expected[0]['opaque_provenance'] == sorted(variants, key=merge._json)
    m = generation(merge, work / 'one', expected)
    swapped = generation(merge, work / 'two', merge.merge_observations([c], [b, a]))
    assert m['generation_id'] == swapped['generation_id']
    assert m['files'] == swapped['files']
    assert merge.load_manifest_artifacts(Path(m['root']), bank='hermes')[2] == expected
    assert merge.validate_generation(Path(m['manifest_path'])) == m
    # Unchanged imports must retain the exact singleton wrapper expected by the publisher.
    assert merge.merge_observations([a], [copy.deepcopy(a)]) == [a]
    for corrupted in (dict(expected[0], provenance=parent('other')),
                      dict(expected[0], opaque_provenance='not-an-array')):
        with pytest.raises(ValueError, match='provenance|parent|bank'):
            generation(merge, work / 'invalid-provenance', [corrupted])
    corrupted = copy.deepcopy(expected[0])
    corrupted['opaque_provenance'][0]['provenance']['bank'] = 'other'
    with pytest.raises(ValueError, match='bank|provenance'):
        generation(merge, work / 'invalid-variant', [corrupted])
    changed_projection = dict(expected[0], provenance=dict(parent(), export_note='must not disappear'))
    with pytest.raises(ValueError, match='provenance'):
        merge.merge_observations([changed_projection], [a])


def test_opaque_semantics_exclude_recipe_but_artifacts_bind_all_provenance(merge, work):
    a = opaque(merge)
    b = copy.deepcopy(a)
    b.update(provenance=dict(parent(), export_note='second exact export'), model='new')
    combined = merge.merge_observations([a], [b])
    assert merge.observation_id(a) == merge.observation_id(b)
    assert merge.observation_content_hash(a) == merge.observation_content_hash(b)
    assert merge.observation_content_hash(combined[0]) == merge.observation_content_hash(a)
    first = generation(merge, work / 'one', [a])
    second = generation(merge, work / 'two', combined)
    assert first['generation_id'] == second['generation_id']
    assert first['observations_sha256'] == second['observations_sha256']
    assert first['files'] != second['files']
    with pytest.raises(ValueError, match='immutable generation differs'):
        generation(merge, work / 'one', combined)
    changed = opaque(merge, metadata='different exact metadata')
    assert merge.observation_content_hash(changed) != merge.observation_content_hash(a)


@pytest.mark.parametrize('case', ['same-id-same-row', 'same-id-different-text',
    'missing-id', 'empty-id', 'blank-id', 'numeric-id', 'null-id'])
def test_importer_requires_unique_original_primary_keys(merge, case):
    a = opaque(merge)['legacy_row']
    rows = [copy.deepcopy(a)]
    if case.startswith('same-id'):
        rows.append(dict(a, text='other exact text') if case.endswith('different-text') else copy.deepcopy(a))
    elif case == 'missing-id':
        rows[0].pop('id')
    else:
        rows[0]['id'] = {'empty-id': '', 'blank-id': '  ', 'numeric-id': 17, 'null-id': None}[case]
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError, match='row id|primary key|duplicate legacy'):
        merge.import_legacy_seed(rows, bank='hermes', provenance=parent())
    assert rows == before


@pytest.mark.parametrize('case', ['same-id-different-text', 'missing-id', 'empty-id', 'blank-id', 'numeric-id'])
def test_generation_cannot_bypass_real_primary_keys_with_opaque_wrappers(merge, work, case):
    a = opaque(merge)
    bad = copy.deepcopy(a)
    if case == 'same-id-different-text':
        bad['legacy_row']['text'] = 'conflicting value for one real primary key'
    elif case == 'missing-id':
        bad['legacy_row'].pop('id')
    else:
        bad['legacy_row']['id'] = {'empty-id': '', 'blank-id': '  ', 'numeric-id': 17}[case]
    bad['legacy_sha256'] = merge.canonical_hash(bad['legacy_row'])
    bad['id'] = merge.observation_id(bad)
    invalid = [a, bad] if case.startswith('same-id') else [bad]
    with pytest.raises(ValueError, match='row id|primary key|duplicate legacy'):
        generation(merge, work / 'blocked', invalid)
    assert not (work / 'blocked' / 'generations').exists()
    # Integrity hashes alone cannot certify a hand-crafted invalid seed index.
    m = generation(merge, work / 'reader', [a])
    index = Path(m['root']) / 'observations_index.jsonl'
    index.write_text(''.join(merge._json(r) + '\n' for r in invalid), encoding='utf-8')
    entry = next(e for e in m['files'] if e['kind'] == 'observations')
    entry.update(sha256=hashlib.sha256(index.read_bytes()).hexdigest(), size=index.stat().st_size)
    m.update(observation_count=len(invalid), observations_sha256=merge._state_hash(invalid))
    reseal_manifest(merge, Path(m['manifest_path']), m)
    with pytest.raises(ValueError, match='row id|primary key|duplicate legacy'):
        merge.load_manifest_artifacts(Path(m['root']), bank='hermes')


@pytest.mark.parametrize('kind', ['observations', 'card'])
@pytest.mark.parametrize('damage', ['missing', 'corrupt'])
@pytest.mark.parametrize('ancestor', ['parent', 'grandparent'])
def test_parent_artifact_ancestry_must_be_restorable(merge, work, publication, kind, damage, ancestor):
    first = generation(merge, work)
    second = merge.write_generation(work, bank='hermes', parent=first,
        observations=[row()], consumed_batch_ids=['batch-b'])
    third = merge.write_generation(work, bank='hermes', parent=second,
        observations=[row()], consumed_batch_ids=['batch-c'])
    adapter, calls = publication
    ledger = types.SimpleNamespace(bank='hermes', root=work)
    adapter._initial_seed = lambda _: ([], parent(), {'document_ids': []})
    assert adapter._candidate(third, ledger)[0] == third
    target = second if ancestor == 'parent' else first
    entry = next(e for e in target['files'] if e['kind'] == kind)
    artifact = Path(target['root']) / entry['path']
    if damage == 'missing':
        artifact.rename(work / 'preserved-parent-artifact')
    else:
        artifact.write_bytes(artifact.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='artifact'):
        merge.validate_generation(Path(third['manifest_path']))
    with pytest.raises(ValueError, match='artifact'):
        merge.load_manifest_artifacts(Path(third['root']), bank='hermes')
    with pytest.raises(ValueError, match='artifact'):
        merge._parent(third, 'hermes')
    blocked = work / 'blocked-child'
    with pytest.raises(ValueError, match='artifact'):
        merge.write_generation(blocked, bank='hermes', parent=third,
            observations=[row()], consumed_batch_ids=['batch-d'])
    assert not blocked.exists()
    with pytest.raises(ValueError, match='artifact'):
        merge.import_legacy_seed([opaque(merge)['legacy_row']], bank='hermes', provenance=third)
    with pytest.raises(ValueError, match='artifact'):
        adapter.publish_generation(third, {}, ledger=ledger)
    assert calls == []


def test_ancestry_validation_has_cycle_and_depth_guards(merge, work, monkeypatch):
    first = generation(merge, work)
    second = merge.write_generation(work, bank='hermes', parent=first,
        observations=[row()], consumed_batch_ids=['batch-b'])
    third = merge.write_generation(work, bank='hermes', parent=second,
        observations=[row()], consumed_batch_ids=['batch-c'])
    path = Path(third['manifest_path'])
    with pytest.raises(ValueError, match='cycle'):
        merge._validated(path, _ancestors=(path,))
    monkeypatch.setattr(merge, 'MAX_ANCESTRY_DEPTH', 2)
    with pytest.raises(ValueError, match='depth'):
        merge.validate_generation(path)
    blocked = work / 'depth-blocked'
    with pytest.raises(ValueError, match='depth'):
        merge.write_generation(blocked, bank='hermes', parent=second,
            observations=[row()], consumed_batch_ids=['new'])
    assert not blocked.exists()


def test_authoritative_singleton_seed_compatibility_and_extra_provenance_refusal(merge, work, publication):
    adapter, calls = publication
    a = opaque(merge)
    seed = [a]
    ledger = types.SimpleNamespace(bank='hermes', root=work)
    adapter._initial_seed = lambda _: (seed, parent(), {'document_ids': ['old-untouched-document']})
    candidate = generation(merge, work / 'normal', merge.merge_observations(seed, [copy.deepcopy(a), row()]))
    bound, rows, previous, cards, _ = adapter._candidate(candidate, ledger)
    assert bound == candidate and previous == seed and len(cards) == 1
    assert next(r for r in rows if r.get('kind') == 'opaque_legacy') == a
    assert merge.merge_observations(rows, []) == rows
    b = dict(a, provenance=dict(parent(), export_note='additional export'))
    extended = generation(merge, work / 'additional-provenance', merge.merge_observations(seed, [b, row()]))
    assert merge.validate_generation(Path(extended['manifest_path'])) == extended
    # Publication intentionally requires exact authoritative wrappers. Local
    # provenance enrichment is not permission to rewrite that seed binding.
    with pytest.raises(ValueError, match='opaque seed differs|rewrites opaque'):
        adapter._candidate(extended, ledger)
    assert calls == []


def test_actual_legacy_reducer_typed_manifests_remain_exactly_loadable(merge, work, publication):
    adapter, calls = publication
    spec = importlib.util.spec_from_file_location('task3_legacy_reducer', ROOT / 'hindsight_offline_v2_reduce.py')
    assert spec is not None and spec.loader is not None
    reducer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reducer
    spec.loader.exec_module(reducer)
    observations = [reducer.Observation(id=f'old-{i}', insight=f'Unique exact fact {i}',
        type='technical_lesson', topic=topic, applicability='under exact conditions',
        evidence_ids=[f'fact-{i}'], source_documents=[f'doc-{i}'], source_section='lesson', source_period='old')
        for i, topic in enumerate(['几何运动', 'robotics'])]
    cards = reducer.build_cards(observations, scope='all', max_topics=20, max_obs=120)
    legacy_root = work / 'actual-old-reducer'
    reducer.write_cards(cards, legacy_root, observations=observations)
    manifest, recovered_cards, rows = merge.load_manifest_artifacts(legacy_root, bank='hermes')
    assert manifest['card_count'] == len(cards) and recovered_cards == cards
    assert sorted(rows, key=lambda r: r['id']) == [o.as_dict() for o in observations]
    loaded = adapter.publisher.load_cards(legacy_root, bank='hermes')
    assert sorted(o['id'] for c in loaded for o in c['canonical_observations']) == [o.id for o in observations]
    frozen = generation(merge, work / 'new-typed', rows)
    assert merge.load_manifest_artifacts(Path(frozen['root']), bank='hermes')[2] == sorted(rows, key=merge.observation_id)
    projected = [o for c in adapter.publisher.load_cards(Path(frozen['root']), bank='hermes')
                 for o in c['canonical_observations']]
    assert sorted(projected, key=merge.observation_id) == sorted(
        [merge.publication_observation(r) for r in rows], key=merge.observation_id)
    assert calls == []


def test_streamed_canonical_bytes_equal_json_dumps_and_reject_nonfinite(merge, work):
    values = [None, True, -0.0, 1e-20, 1.2345678901234567, (1, '中文😀', None),
              {'z': ['\n\t\\\"', {'nested': [False, 7]}], 'a': '𐀀', 'float': 1e30}]
    for i, value in enumerate(values):
        expected = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(',', ':'), allow_nan=False).encode('utf-8')
        assert merge.canonical_hash(value) == hashlib.sha256(expected).hexdigest()
        path = work / f'canonical-{i}.json'
        merge._write(path, merge._json_chunks(value))
        assert path.read_bytes() == expected
        assert merge._read_json(path, expected_hash=hashlib.sha256(expected).hexdigest()) == json.loads(expected)
        with pytest.raises(ValueError, match='artifact hash'):
            merge._read_json(path, expected_hash='0' * 64)
    for value in (float('nan'), {'nested': [float('inf')]}, float('-inf')):
        with pytest.raises(ValueError):
            merge.canonical_hash(value)
    with pytest.raises(TypeError):
        merge.canonical_hash({'invalid': object()})
    jsonl = b'{"x":1}\r\n{"x":2}\r{"x":3}\n'
    for cut in range(1, len(jsonl)):
        assert merge._index([jsonl[:cut], jsonl[cut:]]) == merge._index(jsonl)
    for malformed in (b'{}\n\n', b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'[]\n'):
        with pytest.raises(ValueError):
            merge._index([malformed])


def test_streamed_write_failure_preserves_partial_evidence_without_manifest(merge, work, monkeypatch):
    original = merge._write
    def interrupted(path, chunks):
        def partial():
            yield next(iter(chunks))
            raise OSError('isolated injected disk failure')
        return original(path, partial())
    monkeypatch.setattr(merge, '_write', interrupted)
    with pytest.raises(OSError, match='injected disk failure'):
        generation(merge, work)
    partial = list((work / 'generations').iterdir())
    assert len(partial) == 1 and not (partial[0] / 'manifest.json').exists()
    index = partial[0] / 'observations_index.jsonl'
    assert index.exists() and index.stat().st_mode & 0o777 == 0o600
    before = index.read_bytes()
    monkeypatch.setattr(merge, '_write', original)
    with pytest.raises(ValueError, match='artifact'):
        generation(merge, work)
    assert index.read_bytes() == before and not (partial[0] / 'manifest.json').exists()


@pytest.mark.linux_only
@pytest.mark.parametrize('operation', ['hash', 'generation'])
def test_large_opaque_serialization_under_bounded_address_space(merge, work, operation):
    """Old whole-object Unicode serialization exceeds the child-only memory cap."""
    import os
    import subprocess
    program = r'''
import hashlib, json, pathlib, resource, sys
sys.path.insert(0, sys.argv[1])
import hindsight_incremental_merge as merge
root, operation = pathlib.Path(sys.argv[2]), sys.argv[3]
parent = {'bank': 'synthetic', 'seed_sha256': 'a' * 64}
raw = [{'id': f'unit-{i:05d}', 'bank_id': 'synthetic', 'text': f'原样😀 {i}',
        'embedding': '[' + ','.join(['0.12345678901'] * 1024) + ']',
        'metadata': {'opaque': [None, True, {'tuple_json': [1, '𐀀']}]},
        'embedding_dimension': 1024, 'document_id': 'synthetic-document'}
       for i in range(4000)]
rows = merge.import_legacy_seed(raw, bank='synthetic', provenance=parent)
encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
def golden(value):
    digest = hashlib.sha256()
    for part in encoder.iterencode(value):
        digest.update(part.encode('utf-8'))
    return digest.hexdigest()
expected = golden(rows)
size = int(next(line.split()[1] for line in pathlib.Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmSize:'))) * 1024
resource.setrlimit(resource.RLIMIT_AS, (size + 128 * 1024 * 1024,) * 2)
if operation == 'hash':
    assert merge.canonical_hash(rows) == expected
else:
    manifest = merge.write_generation(root, bank='synthetic', parent=parent,
                                      observations=rows, consumed_batch_ids=['synthetic-batch'])
    bound, cards, recovered = merge.load_manifest_artifacts(pathlib.Path(manifest['root']), bank='synthetic')
    assert bound == manifest and len(recovered) == len(rows)
    assert len(cards[0]['canonical_observations']) == 120
    assert recovered == sorted(rows, key=merge.observation_id)
    index = pathlib.Path(manifest['root']) / 'observations_index.jsonl'
    digest, byte_count = hashlib.sha256(), 0
    for row in sorted(rows, key=merge.observation_id):
        data = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        digest.update(data); byte_count += len(data)
    entry = next(e for e in manifest['files'] if e['kind'] == 'observations')
    assert entry['sha256'] == digest.hexdigest() and entry['size'] == byte_count
    del recovered, cards
    # Reuse is a verification, not an overwrite; corrupt evidence is a blocker.
    before = index.stat().st_mtime_ns
    assert merge.write_generation(root, bank='synthetic', parent=parent, observations=rows,
                                  consumed_batch_ids=['synthetic-batch']) == manifest
    assert index.stat().st_mtime_ns == before
    with index.open('ab') as stream:
        stream.write(b'\n')
    try:
        merge.validate_generation(pathlib.Path(manifest['manifest_path']))
    except ValueError:
        pass
    else:
        raise AssertionError('blank JSONL line was silently discarded')
print(json.dumps({'operation': operation, 'rows': len(rows), 'golden_sha256': expected,
                  'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
'''
    result = subprocess.run([sys.executable, '-c', program, str(ROOT), str(work), operation],
                            capture_output=True, text=True, timeout=180,
                            env={**os.environ, 'HERMES_HOME': str(work / 'isolated-home')})
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(result.stdout)
    assert proof['rows'] == 4000 and proof['operation'] == operation
    (work / f'memory-{operation}.json').write_text(result.stdout)
    print(result.stdout.strip())
