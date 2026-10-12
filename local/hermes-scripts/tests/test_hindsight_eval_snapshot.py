"""Invocation-scoped recall projection; real sealed files, only HTTP is replayed."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
API = 'http://snapshot.test'


def load_module(name, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    merge = load_module('hindsight_incremental_merge', monkeypatch)
    layered = load_module('hindsight_recall_layered', monkeypatch)
    evaluator = load_module('hindsight_eval', monkeypatch)
    # Exercise the evaluator's real dynamic import, never installed/main bytes.
    monkeypatch.setattr(evaluator, 'LAYERED_PATH', SCRIPTS / 'hindsight_recall_layered.py')
    return merge, layered, evaluator


def typed(marker, i):
    return {'bank': 'bank-a', 'insight': f'{marker} geometry relative motion value {i}.25',
            'topic': 'geometry', 'type': 'technical_lesson',
            'applicability': 'static regions', 'valid_from': '2026-10-01',
            'evidence_ids': [f'fact-{i}'], 'source_documents': [f'doc-{i}'],
            'tags': ['geometry', marker], 'conditions': {'region': 'static'},
            'validity': {'status': 'current'}}


def generation(merge, root, *, bank='bank-a', marker='alpha', with_parent=True, typed_count=24):
    provenance = {'bank': bank, 'seed_sha256': hashlib.sha256(bank.encode()).hexdigest()}
    raw = [{'id': f'{bank}-original-{i}', 'bank_id': bank,
            'document_id': f'hermes-offline-canonical::{bank}::{i}',
            'text': f'  {marker} legacy Ω 中文 😀 untouched | type=not-a-field {i}\n',
            'metadata': {'original': ['opaque', i], 'observation_id': 'shared-old-label'},
            'embedding': [0.125, -0.5, i], 'source_memory_ids': [f'old-{i}']}
           for i in range(3)]
    seed = merge.import_legacy_seed(raw, bank=bank, provenance=provenance)
    rows = seed + [dict(typed(marker, i), bank=bank) for i in range(typed_count)]
    parent = merge.write_generation(root, bank=bank, parent=provenance,
        observations=rows, consumed_batch_ids=['parent-batch'])
    if not with_parent:
        return parent
    rows = merge.merge_observations(rows, [dict(typed(f'{marker}-tail', 99), bank=bank)])
    return merge.write_generation(root, bank=bank, parent=parent,
        observations=rows, consumed_batch_ids=['parent-batch', 'new-batch'])


def cases(marker='alpha'):
    return [
        {'id': 'opaque', 'query': f'{marker} legacy', 'mode': 'high-level',
         'expected_terms': ['untouched', '中文', '😀'], 'expected_layers': ['canonical']},
        {'id': 'typed', 'query': f'{marker} geometry relative motion', 'mode': 'mixed',
         'expected_terms': ['relative motion', 'static regions'], 'expected_layers': ['canonical']},
        {'id': 'tail', 'query': f'{marker}-tail geometry 99.25', 'mode': 'high-level',
         'expected_terms': ['99.25', f'{marker}-tail'], 'expected_layers': ['canonical']},
        {'id': 'default-mode', 'query': f'{marker} geometry',
         'expected_terms': ['geometry', 'static regions'], 'expected_layers': ['canonical']},
        {'id': 'evidence-only', 'query': 'native exact value', 'mode': 'evidence',
         'expected_terms': ['native', '42.5'], 'expected_layers': ['sqlite_import']},
        {'id': 'no-match', 'query': 'unrelated zzzzxxxx', 'mode': 'mixed',
         'expected_terms': ['unobtainable'], 'expected_layers': ['canonical']},
    ]


@pytest.fixture
def replay(monkeypatch):
    calls = []
    hook = {'before_response': None}
    def urlopen(request, *, timeout):
        assert request.full_url.startswith(API + '/v1/default/banks/')
        assert request.get_method() == 'POST'
        payload = json.loads(request.data)
        calls.append({'url': request.full_url, 'payload': payload, 'timeout': timeout})
        if hook['before_response']:
            hook['before_response']()
        # Stable wire responses, including a duplicate opaque ID for dedup tests.
        bank = request.full_url.split('/banks/')[1].split('/')[0]
        rows = [
            {'id': 'native-1', 'document_id': 'hermes-sqlite::native', 'type': 'world',
             'text': 'native exact value 42.5 geometry', 'metadata': {'source_count': 1}},
            {'id': f'{bank}-original-0', 'document_id': f'hermes-offline-canonical::{bank}::0',
             'type': 'observation', 'text': 'native legacy summary'},
            {'id': 'native-2', 'document_id': 'hermes-offline-consolidation::weekly::native',
             'type': 'experience', 'text': 'weekly geometry overview'},
        ]
        return io.BytesIO(json.dumps({'results': rows[:payload['limit']]}).encode())
    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    return calls, hook


def run(evaluator, rows, root=None, *, bank='bank-a', sidecar_limit=None):
    return evaluator.run_eval(rows, api=API, bank=bank, top_k=2, raw_limit=5, limit=2,
                              cards_root=str(root) if root else None,
                              include_observations=False, local_sidecar_limit=sidecar_limit)


def standalone_report(evaluator, layered, rows, root, *, bank='bank-a', sidecar_limit=None):
    results, ranked = [], []
    for case in rows:
        query, mode = case['query'], case.get('mode') or 'mixed'
        direct = evaluator.direct_recall(layered, API, bank, query, 5, include_observations=False)
        recall = layered.layered_recall(API, bank, query, mode, 5, 2, cards_root=str(root),
            include_observations=False, local_sidecar_limit=sidecar_limit)
        ranked.append(recall)
        results.append({'id': case['id'], 'query': query, 'mode': mode,
            'direct': evaluator.evaluate_results(case, direct, 2),
            'layered': evaluator.evaluate_results(case, recall, 2)})
    n = max(len(rows), 1)
    summary: dict = {'case_count': len(rows), 'top_k': 2}
    for method in ('direct', 'layered'):
        summary[f'{method}_avg_score'] = round(sum(r[method]['score'] for r in results) / n, 2)
        summary[f'{method}_avg_term_recall'] = round(sum(r[method]['term_recall'] for r in results) / n, 3)
        summary[f'{method}_expected_layer_hits'] = sum(r[method]['expected_layer_hits_top_k'] for r in results)
    return {'cards_root': str(root), 'include_hindsight_observations': False,
            'summary': summary, 'results': results}, ranked


def count_loader(merge, monkeypatch):
    calls = []
    original = merge.load_manifest_artifacts
    def counted(root, **kwargs):
        calls.append((str(root), kwargs))
        return original(root, **kwargs)
    monkeypatch.setattr(merge, 'load_manifest_artifacts', counted)
    return calls


def disk_fingerprint(merge, manifests):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for m in manifests for p in Path(m['root']).rglob('*') if p.is_file()}


@pytest.mark.parametrize('sidecar_limit', [None, 1, 3])
def test_eval_loads_complete_generation_once_with_identical_scores_and_ranking(
        modules, replay, monkeypatch, tmp_path, sidecar_limit):
    merge, layered, evaluator = modules
    manifest = generation(merge, tmp_path / 'state', typed_count=124)
    root = Path(manifest['root'])
    before = disk_fingerprint(merge, [manifest, manifest['parent']])
    projection = layered.load_local_cards(root)
    _, compact, authoritative = merge.load_manifest_artifacts(root)
    assert len(projection) == len(authoritative) == manifest['observation_count']
    assert len(projection) > sum(len(c['canonical_observations']) for c in compact)
    by_id = {r['id']: r for r in projection}
    for obs in authoritative:
        if obs.get('kind') == 'opaque_legacy':
            assert by_id[obs['legacy_row']['id']]['text'] == obs['legacy_row']['text']
            assert by_id[obs['legacy_row']['id']]['document_id'] == obs['legacy_row']['document_id']
    loads = count_loader(merge, monkeypatch)
    reference, ranked_before = standalone_report(evaluator, layered, cases(), root,
                                                sidecar_limit=sidecar_limit)
    standalone_loads = len(loads)
    assert standalone_loads == sum((c.get('mode') or 'mixed') in {'high-level', 'mixed'} for c in cases())
    loads.clear()
    http_before = copy.deepcopy(replay[0])
    replay[0].clear()
    ranked_after = []
    actual_load = evaluator.load_layered_module
    def observed_module():
        mod = actual_load()
        original = mod.layered_recall
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            ranked_after.append(copy.deepcopy(result))
            return result
        mod.layered_recall = observed
        return mod
    monkeypatch.setattr(evaluator, 'load_layered_module', observed_module)
    report = run(evaluator, cases(), root, sidecar_limit=sidecar_limit)
    report.pop('generated_at')
    assert report == reference
    assert ranked_after == ranked_before
    assert replay[0] == http_before
    assert disk_fingerprint(merge, [manifest, manifest['parent']]) == before
    receipt = {'summary': report['summary'], 'case_ids': [r['id'] for r in report['results']],
               'report_sha256': merge.canonical_hash(report),
               'ranked_sha256': merge.canonical_hash(ranked_after),
               'projection_sha256': merge.canonical_hash(projection),
               'projection_count': len(projection), 'opaque_count': 3,
               'authoritative_files_unchanged': True,
               'standalone_strict_loads': standalone_loads, 'eval_strict_loads': len(loads),
               'http_requests': len(replay[0]), 'source_dir': str(SCRIPTS)}
    (tmp_path / 'differential.json').write_text(json.dumps(receipt, indent=2))
    assert len(loads) == 1, 'evaluation must validate the full frozen index once, not per applicable query'


def test_snapshot_is_fresh_and_bank_root_isolated_between_invocations(modules, replay, monkeypatch, tmp_path):
    merge, _, evaluator = modules
    a = generation(merge, tmp_path / 'a', marker='alpha')
    b = generation(merge, tmp_path / 'b', marker='bravo')
    foreign = generation(merge, tmp_path / 'foreign', bank='bank-b', marker='charlie')
    loads = count_loader(merge, monkeypatch)
    reports = [run(evaluator, cases(marker), m['root'], bank=bank)
               for m, marker, bank in [(a, 'alpha', 'bank-a'), (b, 'bravo', 'bank-a'),
                                       (foreign, 'charlie', 'bank-b'), (a, 'alpha', 'bank-a')]]
    assert reports[0]['results'] == reports[-1]['results']
    for report, marker in zip(reports, ['alpha', 'bravo', 'charlie', 'alpha']):
        assert marker in report['results'][0]['layered']['top_preview'][-1]['text']
    assert [p for p, _ in loads] == [a['root'], b['root'], foreign['root'], a['root']]


@pytest.mark.parametrize('selection', ['no-cards', 'evidence-only', 'zero-sidecars', 'empty'])
def test_unused_local_cards_keep_the_legacy_call_shape_and_do_not_load(
        modules, replay, monkeypatch, tmp_path, selection):
    merge, _, evaluator = modules
    m = generation(merge, tmp_path / 'a')
    loads = count_loader(merge, monkeypatch)
    rows = cases() if selection not in {'evidence-only', 'empty'} else (
        [cases()[4]] if selection == 'evidence-only' else [])
    root = None if selection == 'no-cards' else m['root']
    sidecar_limit = 0 if selection == 'zero-sidecars' else None
    actual_load = evaluator.load_layered_module
    def legacy_module():
        mod = actual_load()
        original = mod.layered_recall
        # Old substitutes do not accept a prepared snapshot keyword.
        def old_shape(api, bank, query, mode, raw_limit, limit, *, cards_root,
                      include_observations, local_sidecar_limit):
            return original(api, bank, query, mode, raw_limit, limit, cards_root=cards_root,
                include_observations=include_observations, local_sidecar_limit=local_sidecar_limit)
        mod.layered_recall = old_shape
        return mod
    monkeypatch.setattr(evaluator, 'load_layered_module', legacy_module)
    report = run(evaluator, rows, root, sidecar_limit=sidecar_limit)
    assert report['summary']['case_count'] == len(rows)
    assert loads == []


def test_standalone_recall_still_loads_each_call_and_preserves_default_options(
        modules, replay, monkeypatch, tmp_path):
    merge, layered, _ = modules
    m = generation(merge, tmp_path / 'a')
    loads = count_loader(merge, monkeypatch)
    first = layered.layered_recall(API, 'bank-a', 'alpha geometry', 'mixed', 5, 2, m['root'])
    second = layered.layered_recall(API, 'bank-a', 'alpha geometry', 'mixed', 5, 2, m['root'])
    assert first == second
    assert len(loads) == 2
    assert all(c['payload']['types'] == ['observation', 'world', 'experience'] for c in replay[0])
    assert all(c['timeout'] == 120 for c in replay[0])


@pytest.mark.parametrize('mutation', ['index', 'card', 'manifest', 'parent', 'extra', 'schema', 'bank', 'raw-vector'])
def test_mutation_before_eval_is_rejected_by_real_strict_loader(
        modules, replay, tmp_path, mutation):
    merge, _, evaluator = modules
    m = generation(merge, tmp_path / 'a')
    root = Path(m['root'])
    if mutation in {'index', 'card', 'parent'}:
        target = (root / 'observations_index.jsonl' if mutation == 'index' else
                  root / m['cards'][0]['path'] if mutation == 'card' else
                  Path(m['parent']['root']) / 'observations_index.jsonl')
        target.write_bytes(target.read_bytes() + b' ')
    elif mutation == 'extra':
        (root / 'undeclared.json').write_text('{}')
    elif mutation == 'manifest':
        (root / 'manifest.json').write_text('{}')
    elif mutation == 'schema':
        m['schema_version'] = 'unsupported/9'
        (root / 'manifest.json').write_text(json.dumps(m))
    elif mutation == 'raw-vector':
        target = root / 'observations_index.jsonl'
        rows = [json.loads(line) for line in target.read_text().splitlines()]
        opaque = next(r for r in rows if r.get('kind') == 'opaque_legacy')
        opaque['legacy_row']['embedding'][0] = 0.126
        target.write_text('\n'.join(map(json.dumps, rows)) + '\n')
        artifact = next(f for f in m['files'] if f['kind'] == 'observations')
        artifact.update(sha256=hashlib.sha256(target.read_bytes()).hexdigest(), size=target.stat().st_size)
        m['manifest_sha256'] = merge.canonical_hash({k: v for k, v in m.items() if k != 'manifest_sha256'})
        (root / 'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):
        run(evaluator, cases(), root, bank='bank-b' if mutation == 'bank' else 'bank-a')


def mutate_same_size(path):
    previous = path.stat()
    payload = path.read_bytes()
    # Parseable whitespace change of identical size; restoring mtime is not enough.
    assert payload.endswith(b'\n')
    path.write_bytes(payload[:-1] + b' ')
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert path.stat().st_size == previous.st_size
    assert path.stat().st_mtime_ns == previous.st_mtime_ns


@pytest.mark.parametrize('target_kind', ['index', 'parent', 'extra', 'replacement', 'symlink'])
def test_during_eval_mutation_blocks_instead_of_mixing_candidate_generations(
        modules, replay, monkeypatch, tmp_path, target_kind):
    merge, _, evaluator = modules
    m = generation(merge, tmp_path / 'a')
    root = Path(m['root'])
    actual_load = evaluator.load_layered_module
    first_results = []
    def observed_module():
        mod = actual_load()
        original = mod.layered_recall
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            if not first_results:
                first_results.extend(copy.deepcopy(result))
                target = root / 'observations_index.jsonl'
                if target_kind in {'index', 'parent'}:
                    if target_kind == 'parent':
                        target = Path(m['parent']['root']) / target.name
                    mutate_same_size(target)
                elif target_kind == 'extra':
                    (root / 'extra.json').write_text('{}')
                elif target_kind == 'replacement':
                    replacement = tmp_path / 'replacement.jsonl'
                    replacement.write_bytes(target.read_bytes())
                    os.replace(replacement, target)
                else:
                    (root / 'alias').symlink_to(tmp_path, target_is_directory=True)
            return result
        mod.layered_recall = observed
        return mod
    monkeypatch.setattr(evaluator, 'load_layered_module', observed_module)
    with pytest.raises(ValueError):
        run(evaluator, cases(), root)
    assert any(r.get('_sidecar') for r in first_results)


def test_mutation_during_preparation_is_rejected_even_after_strict_loader_returns(
        modules, replay, monkeypatch, tmp_path):
    merge, _, evaluator = modules
    m = generation(merge, tmp_path / 'a')
    original = merge.load_manifest_artifacts
    def changed_after_validation(root, **kwargs):
        result = original(root, **kwargs)
        mutate_same_size(Path(root) / 'observations_index.jsonl')
        return result
    monkeypatch.setattr(merge, 'load_manifest_artifacts', changed_after_validation)
    with pytest.raises(ValueError):
        run(evaluator, cases(), m['root'])
    assert replay[0] == [], 'snapshot validation must finish before the first recall request'


def test_prepared_view_is_readonly_scope_bound_and_expires(modules, replay, tmp_path):
    merge, layered, _ = modules
    m = generation(merge, tmp_path / 'a')
    b = generation(merge, tmp_path / 'b', marker='bravo')
    with layered.prepare_local_cards(m['root'], bank='bank-a') as snapshot:
        kwargs = {'cards_root': m['root'], '_local_cards_snapshot': snapshot}
        first = layered.layered_recall(API, 'bank-a', 'alpha geometry', 'high-level', 5, 2, **kwargs)
        frozen = copy.deepcopy(first)
        sidecar = next(r for r in first if r.get('_sidecar') and r['type'] == 'technical_lesson')
        sidecar['metadata']['tags'].append('caller-change')
        sidecar['source_fact_ids'].append('caller-change')
        assert layered.layered_recall(API, 'bank-a', 'alpha geometry', 'high-level', 5, 2, **kwargs) == frozen
        for bank, root in [('bank-b', m['root']), ('bank-a', b['root']), ('bank-a', None)]:
            with pytest.raises(ValueError):
                layered.layered_recall(API, bank, 'alpha geometry', 'mixed', 5, 2,
                    cards_root=root, _local_cards_snapshot=snapshot)
        with pytest.raises((TypeError, ValueError)):
            layered.layered_recall(API, 'bank-a', 'alpha geometry', 'mixed', 5, 2,
                cards_root=m['root'], _local_cards_snapshot=[])
    with pytest.raises(ValueError):
        layered.layered_recall(API, 'bank-a', 'alpha geometry', 'mixed', 5, 2, **kwargs)


def test_last_evidence_query_mutation_cannot_return_a_successful_report(modules, replay, tmp_path):
    merge, _, evaluator = modules
    m = generation(merge, tmp_path / 'a')
    def change_on_evidence():
        if replay[0][-1]['payload']['query'].startswith('native exact value'):
            mutate_same_size(Path(m['root']) / 'observations_index.jsonl')
            replay[1]['before_response'] = None
    replay[1]['before_response'] = change_on_evidence
    with pytest.raises(ValueError):
        run(evaluator, [cases()[0], cases()[4]], m['root'])


def test_eval_preserves_exhaustive_parse_proof_and_authoritative_coverage_policy(
        modules, replay, monkeypatch, tmp_path):
    merge, layered, evaluator = modules
    publication = load_module('hindsight_incremental_publication', monkeypatch)
    m = generation(merge, tmp_path / 'a', typed_count=124)
    before = publication.candidate_parse_proof(m['root'], SCRIPTS)
    report = run(evaluator, cases(), m['root'])
    publication._validate_eval(report, sorted(c['id'] for c in cases()))
    after = publication.candidate_parse_proof(m['root'], SCRIPTS)
    assert before == after
    assert after['parse_errors'] == 0
    assert after['observation_count'] == m['observation_count']
    assert after['opaque_count'] == 3
    assert after['eval_view_sha256'] == merge.canonical_hash(layered.load_local_cards(m['root']))
    (tmp_path / 'parse-provenance.json').write_text(json.dumps(after, indent=2))
    mutate_same_size(Path(m['parent']['root']) / 'observations_index.jsonl')
    with pytest.raises(ValueError):
        publication.candidate_parse_proof(m['root'], SCRIPTS)


def test_unmanifested_legacy_projection_retains_identical_eval_behavior(
        modules, replay, tmp_path):
    _, layered, evaluator = modules
    root = tmp_path / 'legacy'
    root.mkdir()
    legacy = [dict(typed('alpha', i), id=f'old-typed-{i}') for i in range(3)]
    (root / 'observations_index.jsonl').write_text('\n'.join(map(json.dumps, legacy)) + '\n')
    expected, _ = standalone_report(evaluator, layered, cases(), root)
    actual = run(evaluator, cases(), root)
    actual.pop('generated_at')
    assert actual == expected


def test_missing_optional_root_stays_empty_but_appearance_during_eval_blocks(
        modules, replay, tmp_path):
    _, _, evaluator = modules
    root = tmp_path / 'not-created'
    report = run(evaluator, cases(), root)
    assert all(r['layered']['evaluated_sidecar_count'] == 0 for r in report['results'])
    def appear():
        root.mkdir()
        replay[1]['before_response'] = None
    replay[1]['before_response'] = appear
    with pytest.raises(ValueError):
        run(evaluator, cases(), root)


def offchain_generation(merge, root, kind, *, typed_count=1):
    """Real strict dependencies that cannot be discovered in the root manifest."""
    seed = {'bank': 'bank-a', 'seed_sha256': hashlib.sha256(b'offchain-seed').hexdigest()}
    raw = [{'id': 'offchain-original', 'bank_id': 'bank-a',
            'document_id': 'hermes-offline-canonical::offchain',
            'text': 'alpha legacy exact Ω 中文 😀',
            'metadata': {'opaque': [None, ['unchanged', 1]]}, 'embedding': [0.125, -0.5]}]
    leaf = merge.write_generation(root / 'leaf', bank='bank-a', parent=seed,
        observations=[typed('leaf', 0)], consumed_batch_ids=['leaf'])
    provenance = leaf
    dependencies = [leaf]
    if kind == 'nested-seed-generation':
        # The imported seed itself refers to a second generation, neither root
        # appears in the candidate's manifest.parent tree.
        imported = merge.import_legacy_seed(raw, bank='bank-a', provenance=leaf)
        provenance = merge.write_generation(root / 'seed-generation', bank='bank-a',
            parent=seed, observations=imported, consumed_batch_ids=['seed-generation'])
        dependencies.append(provenance)
    wrapped = merge.import_legacy_seed(raw, bank='bank-a', provenance=provenance)
    if kind == 'opaque-variant':
        original = merge.import_legacy_seed(raw, bank='bank-a', provenance=seed)
        original[0]['model'], wrapped[0]['model'] = 'a-original', 'z-offchain'
        wrapped = merge.merge_observations(original, wrapped)
        assert wrapped[0]['provenance'] == seed
        assert wrapped[0]['opaque_provenance'][1]['provenance'] == leaf
    candidate = merge.write_generation(root / 'candidate', bank='bank-a', parent=seed,
        observations=wrapped + [typed('alpha', i) for i in range(typed_count)],
        consumed_batch_ids=['candidate'])
    return candidate, dependencies


@pytest.mark.parametrize('kind', ['row', 'opaque-variant', 'nested-seed-generation'])
@pytest.mark.parametrize('when', ['before', 'first-recall', 'during-read', 'after-validation'])
def test_actual_offchain_dependencies_are_fenced_before_reads_and_after_validation(
        modules, replay, monkeypatch, tmp_path, kind, when):
    merge, layered, evaluator = modules
    candidate, dependencies = offchain_generation(merge, tmp_path, kind)
    root = Path(candidate['root'])
    leaf = dependencies[0]
    assert Path(leaf['root']) not in layered._snapshot_roots(root), 'old manifest-only scope precondition'
    target = Path(leaf['root']) / 'observations_index.jsonl'
    original_chunks, validated_reads = merge._file_chunks, []
    def is_target(stream):
        opened, expected = os.fstat(stream.fileno()), target.stat()
        return (opened.st_dev, opened.st_ino) == (expected.st_dev, expected.st_ino)
    def counted_chunks(stream):
        if is_target(stream):
            validated_reads.append(target)
        yield from original_chunks(stream)
    with monkeypatch.context() as counting:
        counting.setattr(merge, '_file_chunks', counted_chunks)
        assert merge.load_manifest_artifacts(root, bank='bank-a')[0] == candidate
    assert validated_reads
    mutations = []
    def mutate():
        if not mutations:
            mutations.append(target)
            mutate_same_size(target)
    if when == 'before':
        mutate()
    elif when == 'first-recall':
        replay[1]['before_response'] = mutate
    elif when == 'during-read':
        reads = []
        def traced_chunks(stream):
            matched = is_target(stream)
            yield from original_chunks(stream)
            if matched:
                reads.append(target)
                # Mutate during the final real validation read, not early enough
                # for an unrelated repeated strict read to mask the fence gap.
                if len(reads) == len(validated_reads):
                    mutate()
        monkeypatch.setattr(merge, '_file_chunks', traced_chunks)
    else:
        original = merge.load_manifest_artifacts
        def changed_after_validation(path, **kwargs):
            result = original(path, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(merge, 'load_manifest_artifacts', changed_after_validation)
    with pytest.raises(ValueError):
        run(evaluator, cases(), root)
    assert mutations == [target]
    if when != 'first-recall':
        assert replay[0] == [], 'invalid discovery/validation must block before any recall'


@pytest.mark.parametrize('damage', ['extra-file', 'extra-directory', 'symlink', 'ancestor-symlink', 'replacement'])
@pytest.mark.parametrize('when', ['during-discovery', 'last-recall'])
def test_offchain_inventory_and_path_changes_fail_closed(
        modules, replay, monkeypatch, tmp_path, damage, when):
    merge, _, evaluator = modules
    candidate, dependencies = offchain_generation(merge, tmp_path, 'opaque-variant')
    dep_root = Path(dependencies[0]['root'])
    mutations = []
    def mutate():
        if mutations:
            return
        mutations.append(damage)
        if damage == 'extra-file':
            (dep_root / 'undeclared.json').write_text('{}')
        elif damage == 'extra-directory':
            (dep_root / 'undeclared').mkdir()
        elif damage == 'symlink':
            (dep_root / 'alias').symlink_to(tmp_path, target_is_directory=True)
        elif damage == 'ancestor-symlink':
            ancestor = dep_root.parent
            saved = ancestor.with_name('saved-generations')
            ancestor.rename(saved)
            ancestor.symlink_to(saved, target_is_directory=True)
        else:
            target = dep_root / 'observations_index.jsonl'
            replacement = tmp_path / 'replacement.jsonl'
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)
    if when == 'during-discovery':
        original = merge._inventory
        def changed_after_inventory(root, expected):
            original(root, expected)
            if root == dep_root:
                mutate()
        monkeypatch.setattr(merge, '_inventory', changed_after_inventory)
    else:
        def last_recall():
            if replay[0][-1]['payload']['query'].startswith('native exact value'):
                mutate()
                replay[1]['before_response'] = None
        replay[1]['before_response'] = last_recall
    with pytest.raises(ValueError):
        run(evaluator, [cases()[0], cases()[4]], candidate['root'])
    assert mutations == [damage]
    if when == 'during-discovery':
        assert replay[0] == []


def test_offchain_full_projection_and_results_equal_uncached_with_one_strict_load(
        modules, replay, monkeypatch, tmp_path):
    merge, layered, evaluator = modules
    candidate, dependencies = offchain_generation(merge, tmp_path, 'opaque-variant', typed_count=124)
    root = Path(candidate['root'])
    manifests = [candidate, *dependencies]
    before = disk_fingerprint(merge, manifests)
    _, compact, rows = merge.load_manifest_artifacts(root, bank='bank-a')
    projection = layered.load_local_cards(root, bank='bank-a')
    assert len(projection) == len(rows) > sum(len(c['canonical_observations']) for c in compact)
    reference, ranking = standalone_report(evaluator, layered, cases(), root)
    requests = copy.deepcopy(replay[0])
    replay[0].clear()
    loads = count_loader(merge, monkeypatch)
    actual_load = evaluator.load_layered_module
    ranked, reads, query_read_counts, snapshots = [], [], [], []
    from contextlib import contextmanager
    original_open = merge._open_read
    @contextmanager
    def counted_open(path):
        reads.append(path)
        for trace in merge._validation_read_fences.get():
            assert any(path in inventory for inventory in trace._inventories.values()), (
                'every strict artifact inventory must be observed before its first read')
        with original_open(path) as stream:
            yield stream
    monkeypatch.setattr(merge, '_open_read', counted_open)
    def observed_module():
        mod = actual_load()
        original = mod.layered_recall
        def observed(*args, **kwargs):
            snapshot = kwargs['_local_cards_snapshot']
            if not snapshots:
                snapshots.append(snapshot)
                assert layered._copy_projection(snapshot._cards) == projection
                assert set(snapshot._read_fence._files) == {Path(p) for p in before}
                assert set(snapshot._read_fence._inventories) == {Path(m['root']) for m in manifests}
            query_read_counts.append(len(reads))
            result = original(*args, **kwargs)
            assert len(reads) == query_read_counts[-1], 'cheap query fences must not reread artifacts'
            ranked.append(copy.deepcopy(result))
            return result
        mod.layered_recall = observed
        return mod
    monkeypatch.setattr(evaluator, 'load_layered_module', observed_module)
    def write_unrelated_ledger_siblings():
        (root.parent / 'GateReports.json').write_text('{}')
        (root.parent.parent / 'monitor.db').write_bytes(b'private sibling')
    replay[1]['before_response'] = write_unrelated_ledger_siblings
    actual = run(evaluator, cases(), root)
    actual.pop('generated_at')
    assert actual == reference and ranked == ranking and replay[0] == requests
    assert loads == [(str(root), {'bank': 'bank-a'})]
    assert query_read_counts and all(count == len(reads) for count in query_read_counts)
    assert disk_fingerprint(merge, manifests) == before
    (tmp_path / 'offchain-differential.json').write_text(json.dumps({
        'report_sha256': merge.canonical_hash(actual), 'ranking_sha256': merge.canonical_hash(ranked),
        'projection_sha256': merge.canonical_hash(projection), 'projection_count': len(projection),
        'strict_loads': len(loads), 'dependency_roots': [m['root'] for m in dependencies],
        'read_paths': sorted(set(map(str, reads))), 'query_read_counts': query_read_counts,
        'inventory_roots': sorted(map(str, snapshots[0]._read_fence._inventories)),
        'authoritative_files_unchanged': True}, indent=2))


def seed_artifact(merge, publication, root):
    bank = 'bank-a'
    document_id = 'hermes-offline-canonical::seed'
    text = 'exact opaque seed Ω 中文 😀'
    docs = [{'id': document_id, 'bank_id': bank, 'original_text': text,
             'content_hash': hashlib.sha256(text.encode()).hexdigest()}]
    raw = [{'id': 'seed-original', 'bank_id': bank, 'document_id': document_id,
            'text': text, 'fact_type': 'observation', 'embedding': [0.125] * 1024,
            'embedding_dimension': 1024, 'metadata': {'unknown': ['exact', None]}}]
    remote = publication._remote_identity(bank, docs, raw)
    provenance = {'bank': bank, 'seed_sha256': merge.canonical_hash(
        {'bank': bank, 'observations': raw, 'remote_seed': remote})}
    wrapped = merge.import_legacy_seed(raw, bank=bank, provenance=provenance)
    value = {'bank': bank, 'observations': wrapped, 'provenance': provenance, 'remote_seed': remote}
    value['export_sha256'] = merge.canonical_hash(value)
    path = root / 'seed.json'
    merge._write(path, merge._artifact_chunks('card', value))
    seed = {'bank': bank, 'artifact_path': str(path), 'artifact_hash': hashlib.sha256(path.read_bytes()).hexdigest(),
            'provenance': provenance, 'remote_seed': remote}
    from types import SimpleNamespace
    ledger = SimpleNamespace(root=root, bank=bank, initialization_manifest=lambda: {'seed': seed})
    # _initial_seed has no SQL/API work; use that real validator, not an invented
    # seed-path contract in generation provenance (which only binds a seed hash).
    adapter = object.__new__(publication.PublicationAdapter)
    adapter.bank = bank
    return path, adapter, ledger, (wrapped, provenance, remote)


@pytest.mark.parametrize('when', ['before', 'during-read', 'after-validation', 'late'])
def test_actual_read_trace_fences_seed_blob_without_fencing_ledger_siblings(
        modules, monkeypatch, tmp_path, when):
    merge, _, _ = modules
    publication = load_module('hindsight_incremental_publication', monkeypatch)
    path, adapter, ledger, expected = seed_artifact(merge, publication, tmp_path / 'ledger')
    mutations = []
    def mutate():
        mutations.append(path)
        mutate_same_size(path)
    if when == 'before':
        mutate()
    elif when == 'during-read':
        original = merge._load_json
        def changed_during_read(stream, **kwargs):
            result = original(stream, **kwargs)
            mutate()
            return result
        monkeypatch.setattr(merge, '_load_json', changed_during_read)
    if when != 'late':
        with pytest.raises(ValueError):
            with merge._validation_read_scope():
                assert adapter._initial_seed(ledger) == expected
                if when == 'after-validation':
                    mutate()
    else:
        with merge._validation_read_scope() as trace:
            assert adapter._initial_seed(ledger) == expected
        assert set(trace._files) == {path} and trace._inventories == {}
        (path.parent / 'GateReports.json').write_text('{}')
        (path.parent / 'monitor.db').write_bytes(b'private sibling')
        trace.check()
        mutate()
        with pytest.raises(ValueError):
            trace.check()
    assert mutations == [path]
    assert merge._validation_read_fences.get() == ()


def test_trace_nested_scopes_restore_on_errors_and_do_not_leak_between_roots(modules, tmp_path):
    merge, _, _ = modules
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_bytes(b'{"a":1}\n')
    b.write_bytes(b'{"b":2}\n')
    with merge._validation_read_scope() as outer:
        assert merge._read(a) == a.read_bytes()
        with pytest.raises(RuntimeError, match='private nested error'):
            with merge._validation_read_scope() as inner:
                assert merge._read(b) == b.read_bytes()
                raise RuntimeError('private nested error')
        assert merge._read(a) == a.read_bytes()
    assert set(outer._files) == {a, b} and set(inner._files) == {b}
    with merge._validation_read_scope() as fresh:
        assert merge._read(a) == a.read_bytes()
    assert set(fresh._files) == {a}
    mutate_same_size(b)
    fresh.check()
    with pytest.raises(ValueError):
        outer.check()
    # Neither a success nor an exception leaves a tracker in the caller context.
    assert merge._validation_read_fences.get() == ()


def test_trace_failed_initial_validation_does_not_poison_other_bank_or_root(
        modules, replay, monkeypatch, tmp_path):
    merge, _, evaluator = modules
    invalid, dependencies = offchain_generation(merge, tmp_path / 'invalid', 'row')
    foreign = generation(merge, tmp_path / 'foreign', bank='bank-b', marker='charlie')
    mutate_same_size(Path(dependencies[0]['root']) / 'observations_index.jsonl')
    with pytest.raises(ValueError):
        run(evaluator, cases(), invalid['root'])
    loads = count_loader(merge, monkeypatch)
    report = run(evaluator, cases('charlie'), foreign['root'], bank='bank-b')
    assert report['summary']['case_count'] == len(cases('charlie'))
    assert loads == [(foreign['root'], {'bank': 'bank-b'})]
    assert merge._validation_read_fences.get() == ()
