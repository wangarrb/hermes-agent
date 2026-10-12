"""Isolated SQL-double tests exercise the real publisher transaction and gate."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import hindsight_incremental_merge as merge


class Cursor:
    def __init__(self, db):
        self.db, self.rows, self.rowcount = db, [], 0
    def __enter__(self):
        return self
    def __exit__(self, *_):
        return False
    def execute(self, sql, params=None):
        self.db.sql.append((sql, params))
        assert params is not None
        text = ' '.join(sql.split())
        self.rows = []
        if text.startswith('SELECT'):
            bank = params[0]
            if 'FROM unit_entities' in text:
                self.rows = [{'row': copy.deepcopy(r)} for r in self.db.data.get('unit_entities', []) if r['unit_id'] in params[0]]
            elif 'FROM chunks' in text:
                self.rows = [{'row': copy.deepcopy(r)} for r in self.db.data.get('chunks', []) if r['bank_id'] == bank and r['document_id'] in params[1]]
            elif 'FROM memory_links' in text:
                self.rows = [{'row': copy.deepcopy(r)} for r in self.db.data.get('links', []) if r['bank_id'] == bank and (r['from_unit_id'] in params[1] or r['to_unit_id'] in params[2])]
            elif 'source_memory_ids &&' in text:
                self.rows = [{'row': copy.deepcopy(r)} for r in self.db.data['units'] if r['bank_id'] == bank and set(r.get('source_memory_ids') or []) & set(params[1])]
            elif 'FROM entities' in text:
                rows = [r for r in self.db.data.get('entities', []) if r['bank_id'] == bank]
                if 'to_jsonb' in text:
                    self.rows = [{'row': copy.deepcopy(r)} for r in rows if r['id'] in params[1]]
                else:
                    self.rows = [(r['id'],) for r in rows if r['canonical_name'].lower() == params[1].lower()]
            elif 'to_jsonb' in text:
                table = 'documents' if 'FROM documents' in text else 'units'
                rows = [r for r in self.db.data[table] if r['bank_id'] == bank]
                if 'ANY' in text:
                    key = 'id' if table == 'documents' else 'document_id'
                    rows = [r for r in rows if r[key] in params[1]]
                else:
                    key = 'id' if table == 'documents' else 'document_id'
                    rows = [r for r in rows if r[key].startswith('hermes-offline-canonical::')
                            and not r[key].startswith('hermes-offline-canonical::incremental::')]
                self.rows = [{'row': copy.deepcopy(r)} for r in sorted(rows, key=lambda r: r['id'])]
            elif 'FROM documents' in text or 'FROM memory_units' in text:
                table = 'documents' if 'FROM documents' in text else 'units'
                key = 'id' if table == 'documents' else 'document_id'
                ids = params[1]
                self.rows = [copy.deepcopy(r) for r in self.db.data[table]
                             if r['bank_id'] == bank and (r[key] in ids if isinstance(ids, list) else r[key].startswith('hermes-offline-canonical::'))]
        elif text.startswith('DELETE FROM documents'):
            assert 'LIKE' not in text, 'global canonical DELETE is forbidden'
            bank, ids = params
            assert all(i.startswith('hermes-offline-canonical::incremental::') for i in ids)
            before = len(self.db.data['documents'])
            self.db.data['documents'] = [d for d in self.db.data['documents'] if not (d['bank_id'] == bank and d['id'] in ids)]
            self.db.data['units'] = [u for u in self.db.data['units'] if not (u['bank_id'] == bank and u['document_id'] in ids)]
            self.rowcount = before - len(self.db.data['documents'])
        elif text.startswith('INSERT INTO entities'):
            name, bank, metadata = params
            entities = self.db.data.setdefault('entities', [])
            existing = next((r for r in entities if r['bank_id'] == bank and r['canonical_name'].lower() == name.lower()), None)
            if existing is None:
                existing = {'id': str(uuid.uuid4()), 'canonical_name': name, 'bank_id': bank, 'metadata': json.loads(metadata)}
                entities.append(existing)
            self.rows = [(existing['id'],)]
        else:
            raise AssertionError(text)
    def fetchall(self):
        return self.rows
    def fetchone(self):
        return self.rows[0]


class Connection:
    def __init__(self, db):
        self.db = db
        self.saved = copy.deepcopy(db.data)
    def cursor(self, **_):
        return Cursor(self.db)
    def set_session(self, **kw):
        self.db.sessions.append(kw)
    def commit(self):
        self.db.commits += 1
        self.saved = copy.deepcopy(self.db.data)
        if self.db.lose_ack:
            self.db.lose_ack = False
            raise OSError('commit acknowledgement lost')
    def rollback(self):
        self.db.data = copy.deepcopy(self.saved)
    def close(self):
        pass


class Database:
    def __init__(self):
        old = 'hermes-offline-canonical::topic::old::123456789abc'
        self.data = {'documents': [{'id': old, 'bank_id': 'bank-a', 'original_text': 'opaque doc',
                      'content_hash': hashlib.sha256(b'opaque doc').hexdigest(), 'metadata': {'unknown': [1, {'x': True}]}}],
                     'units': [{'id': str(uuid.uuid4()), 'bank_id': 'bank-a', 'document_id': old,
                      'text': 'not parsed | applicability=unknown', 'fact_type': 'observation',
                      'metadata': {'legacy_nested': '{"not":"decoded"}'}, 'embedding': '[' + ','.join(['0'] * 1024) + ']',
                      'embedding_dimension': 1024, 'future_field': {'keep': [None, 'x']}}]}
        self.sql, self.sessions, self.embeds = [], [], []
        self.commits, self.lose_ack = 0, False
    def connect(self, dsn):
        assert dsn == 'scratch-only'
        return Connection(self)
    def encode(self, texts, *_a, **_kw):
        self.embeds.append(texts)
        return [[0.25] * 1024 for _ in texts], 'fixture'
    def values(self, cur, sql, rows, **kwargs):
        self.sql.append((sql, rows))
        if 'INSERT INTO documents' in sql:
            for r in rows:
                existing = next((d for d in self.data['documents'] if d['id'] == r[0] and d['bank_id'] == r[1]), None)
                new = dict(id=r[0], bank_id=r[1], original_text=r[2], content_hash=r[3], metadata=json.loads(r[4]))
                if existing:
                    assert 'ON CONFLICT' in sql, 'plain INSERT must fail on existing document'
                    new['metadata'] = dict(existing['metadata'], **new['metadata'])
                    existing.update(new)
                else:
                    self.data['documents'].append(new)
        elif 'INSERT INTO chunks' in sql:
            for r in rows:
                existing = next((d for d in self.data.get('chunks', []) if d['chunk_id'] == r[0]), None)
                if existing:
                    assert 'ON CONFLICT' in sql
                    existing.update(chunk_text=r[4], content_hash=r[5])
                else:
                    self.data.setdefault('chunks', []).append(dict(chunk_id=r[0], document_id=r[1], bank_id=r[2], chunk_index=r[3], chunk_text=r[4], content_hash=r[5]))
        elif 'INSERT INTO memory_units' in sql:
            result = []
            for r in rows:
                explicit = len(r) == 17
                uid = r[0] if explicit else str(uuid.uuid4())
                r = r[1:] if explicit else r
                new = dict(id=uid, bank_id=r[0], text=r[1], embedding=r[2],
                    embedding_dimension=1024, fact_type=r[8], metadata=json.loads(r[9]), document_id=r[11],
                    source_memory_ids=r[15], chunk_id=r[10])
                existing = next((u for u in self.data['units'] if u['id'] == uid), None)
                if existing:
                    assert 'ON CONFLICT' in sql
                    new['metadata'] = dict(existing['metadata'], **new['metadata'])
                    new['source_memory_ids'] = sorted(set(existing.get('source_memory_ids', []) + new['source_memory_ids']))
                    existing.update(new)
                else:
                    self.data['units'].append(new)
                result.append((uid,))
            return result
        elif 'INSERT INTO unit_entities' in sql:
            for uid, eid in rows:
                value = {'unit_id': uid, 'entity_id': eid}
                if value not in self.data.setdefault('unit_entities', []):
                    self.data['unit_entities'].append(value)
        elif 'INSERT INTO memory_links' in sql:
            for uid, sid, typ, weight, bank in rows:
                value = dict(from_unit_id=uid, to_unit_id=sid, link_type=typ, weight=weight, bank_id=bank)
                if value not in self.data.setdefault('links', []):
                    self.data['links'].append(value)
        return []


class Ledger:
    bank = 'bank-a'
    def __init__(self, root):
        self.root = root
        self.seed = {}
    def initialization_manifest(self):
        return {'bank': self.bank, 'seed': self.seed}
    def get_batch(self, batch_id):
        return {'bank': self.bank, 'batch_id': batch_id, 'payload_sha256': 'a' * 64,
                'phase': 'locally_merged', 'receipts': {'consolidated': {'verified': True}}}


@pytest.fixture
def env(tmp_path):
    mod = importlib.import_module('hindsight_incremental_publication')
    db = Database()
    config = {'dsn': 'scratch-only', 'script_dir': str(ROOT), 'publisher_python': sys.executable,
              'api': 'http://127.0.0.1:1', 'hermes_home': str(tmp_path),
              'generic_benchmark': str(tmp_path / 'generic.jsonl'), 'local_benchmark': str(tmp_path / 'local.jsonl')}
    for label in ('generic', 'local'):
        (tmp_path / f'{label}.jsonl').write_text('{"id":"case"}\n')
    adapter = mod.PublicationAdapter(config, 'bank-a', connect=db.connect,
        embeddings=db.encode, execute_values=db.values, run_json=fake_run)
    ledger = Ledger(tmp_path)
    export = adapter.export_seed(tmp_path / 'seed.json')
    ledger.seed = {'bank': ledger.bank, 'artifact_path': str(tmp_path / 'seed.json'),
        'artifact_hash': hashlib.sha256((tmp_path / 'seed.json').read_bytes()).hexdigest(),
        'provenance': export['provenance'], 'remote_seed': export['remote_seed']}
    seed = export['observations']
    row = {'insight': 'Use exact condition', 'topic': 'topic-a', 'type': 'rule', 'applicability': 'under nested conditions',
           'conditions': {'and': [{'version': {'gte': 3}}, {'not': ['unknown']}]},
           'evidence_ids': [f'evidence-{i}' for i in range(150)],
           'evidence': [{'complex': {'facts': [1, None, True]}}], 'source_documents': ['source-new']}
    candidate = merge.write_generation(tmp_path / 'canonical', bank=ledger.bank, parent=export['provenance'],
        observations=merge.merge_observations(seed, [row]), consumed_batch_ids=['batch-a'])
    proof = {'bank': ledger.bank, 'scope': ['arrival-batch:batch-a'], 'ready': True,
             'pending_in_scope': 0, 'pending_operations': 0}
    return mod, adapter, ledger, db, candidate, proof, row


def fake_run(cmd, **_):
    name = Path(cmd[1]).name
    arg = lambda k: cmd[cmd.index(k) + 1]
    if name == 'hindsight_conflict_audit.py':
        return {'bank': arg('--bank'), 'cards_root': arg('--cards-root'), 'offline_root': arg('--offline-root'),
                'observation_count': len(merge.load_manifest_artifacts(Path(arg('--cards-root')))[2]), 'api': arg('--api'),
                'cases': [], 'summary': {'blocking_cases': 0}, 'decision': 'pass', 'source_scan_errors': []}
    assert name == 'hindsight_eval.py'
    use_cards = '--use-local-cards' in cmd
    from hindsight_incremental_publication import candidate_parse_proof
    return {'cards_root': arg('--cards-root') if use_cards else None,
            'candidate_parse_proof': candidate_parse_proof(arg('--cards-root')) if use_cards else None,
            'invocation': {'bank': arg('--bank'), 'api': arg('--api'), 'benchmark': arg('--benchmark')},
            'summary': {'case_count': 1, 'layered_avg_term_recall': 0.6 if use_cards else 0.5,
                        'layered_expected_layer_hits': 2 if use_cards else 1, 'layered_avg_score': 1},
            'results': [{'id': 'case', 'direct': {'term_recall': 0.5, 'expected_layer_hits_top_k': 1, 'score': 1},
                        'layered': {'term_recall': 0.6 if use_cards else 0.5,
                                    'expected_layer_hits_top_k': 2 if use_cards else 1, 'score': 1}}]}


def test_remote_cursor_validation_streams_instead_of_serializing_full_list(monkeypatch):
    module = importlib.import_module('hindsight_incremental_publication')
    cursor = type('FixtureCursor', (), {'fetchall': lambda _self: [{'row': {'id': 'b', 'text': '原样😀'}}, {'row': {'id': 'a', 'text': '保留'}}]})()
    monkeypatch.setattr(merge, '_json', lambda *_a: pytest.fail('whole-list JSON validation reached'))
    assert [row['id'] for row in module._rows_from_cursor(cursor)] == ['a', 'b']


@pytest.mark.parametrize('invalid_hash', [None, '', False, 0, 'a' * 63, 'A' * 64])
def test_initial_seed_requires_literal_sha256_even_for_parseable_unchanged_json(env, invalid_hash):
    _mod, adapter, fixture_ledger, db, _manifest, _proof, _row = env
    from hindsight_incremental_ledger import IncrementalLedger
    with Path(fixture_ledger.seed['artifact_path']).open('ab') as stream:
        stream.write(b' ' * 70000)
    root = fixture_ledger.root / 'mandatory-hash-ledger'
    ledger = IncrementalLedger(root, fixture_ledger.bank, create=True)
    frozen_path = ledger.root / 'seed.json'
    frozen_path.write_bytes(Path(fixture_ledger.seed['artifact_path']).read_bytes())
    frozen_path.chmod(0o600)
    seed = dict(fixture_ledger.seed, artifact_path=str(frozen_path), artifact_hash=invalid_hash)
    ledger.initialize_legacy([], seed, [], confirm='initialize-hindsight-incremental')
    with pytest.raises(ValueError, match='^invalid initial seed artifact hash$'):
        adapter._initial_seed(ledger)
    assert db.commits == 0 and db.embeds == []


def test_initial_seed_uses_hash_bound_stream_decoder(env, monkeypatch):
    _mod, adapter, ledger, db, _manifest, _proof, _row = env
    monkeypatch.setattr(merge, '_read', lambda *_a: pytest.fail('whole-file seed read reached'))
    rows, provenance, remote = adapter._initial_seed(ledger)
    assert [row['legacy_row'] for row in rows] == db.data['units']
    assert provenance == ledger.seed['provenance']
    assert remote == ledger.seed['remote_seed']


def test_initializer_uses_stream_decoder_before_freezing_seed(env, monkeypatch):
    _mod, adapter, fixture_ledger, db, _manifest, _proof, _row = env
    import hindsight_incremental_pipeline as pipeline
    from hindsight_incremental_ledger import IncrementalLedger
    seed_file = Path(fixture_ledger.seed['artifact_path'])
    original = Path.read_text
    def reject_whole_seed(path, *args, **kwargs):
        if path == seed_file:
            pytest.fail('whole-file seed text read reached')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', reject_whole_seed)
    config = dict(adapter.config, bank=fixture_ledger.bank,
                  incremental={'root': str(fixture_ledger.root / 'stream-ledger')})
    pipeline.initialize(config, seed_file=seed_file, confirm='initialize-hindsight-incremental',
                        source_reader=lambda *_a, **_kw: [])
    ledger = IncrementalLedger(Path(config['incremental']['root']), fixture_ledger.bank)
    _parent, rows, sequence = pipeline._seed_state(ledger)
    assert sequence == 0 and [row['legacy_row'] for row in rows] == db.data['units']


def test_export_is_authoritative_frozen_native_and_readonly(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    exported = json.loads(Path(ledger.seed['artifact_path']).read_text())
    assert [r['legacy_row'] for r in exported['observations']] == db.data['units']
    assert all(r['kind'] == 'opaque_legacy' for r in exported['observations'])
    assert exported['export_sha256'] == merge.canonical_hash({k: v for k, v in exported.items() if k != 'export_sha256'})
    assert exported['provenance']['bank'] == ledger.bank
    assert db.sessions and all(s['readonly'] is True for s in db.sessions)
    assert db.commits == 0 and db.embeds == []


def test_opaque_projection_uses_exact_original_text_and_memory_id(env):
    mod, adapter, ledger, db, manifest, proof, _ = env
    import hindsight_conflict_audit as conflict
    import hindsight_recall_layered as layered
    before = Path(manifest['root']) / 'observations_index.jsonl'
    frozen = before.read_bytes()
    raw = db.data['units'][0]
    audited = conflict.load_observations(Path(manifest['root']))
    evaluated = layered.load_local_cards(manifest['root'])
    assert next(r for r in audited if r['kind'] == 'opaque_legacy')['id'] == raw['id']
    assert next(r for r in evaluated if r['type'] == 'opaque_legacy')['id'] == raw['id']
    assert next(r for r in evaluated if r['type'] == 'opaque_legacy')['text'] == raw['text']
    assert before.read_bytes() == frozen
    receipt = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    parsed = receipt['candidate_parse_proof']
    assert parsed['opaque_count'] == 1 and parsed['typed_count'] == 1 and parsed['observation_count'] == 2
    assert parsed['parse_errors'] == 0 and parsed['manifest_sha256'] == manifest['manifest_sha256']


def test_export_real_initializer_and_seed_state_interoperate_without_reparsing(env):
    mod, adapter, fixture_ledger, db, manifest, proof, row = env
    import hindsight_incremental_pipeline as pipeline
    from hindsight_incremental_ledger import IncrementalLedger
    config = dict(adapter.config, bank=fixture_ledger.bank,
                  incremental={'root': str(fixture_ledger.root / 'real-ledger')})
    pipeline.initialize(config, seed_file=Path(fixture_ledger.seed['artifact_path']),
        confirm='initialize-hindsight-incremental', source_reader=lambda *_a, **_kw: [])
    ledger = IncrementalLedger(Path(config['incremental']['root']), fixture_ledger.bank)
    parent, seed, sequence = pipeline._seed_state(ledger)
    assert sequence == 0 and all(r['kind'] == 'opaque_legacy' for r in seed)
    assert [r['legacy_row'] for r in seed] == db.data['units']
    candidate = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=parent,
        observations=merge.merge_observations(seed, [row]), consumed_batch_ids=['new-cohort'])
    bound, rows, previous, cards, remote = adapter._candidate(candidate, ledger)
    assert bound == candidate and previous == seed and len(cards) == 1
    assert remote == ledger.initialization_manifest()['seed']['remote_seed']
    assert db.commits == 0 and db.embeds == []


def test_publish_preserves_opaque_seed_and_full_metadata_and_is_idempotent(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    old = copy.deepcopy(db.data)
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    result = adapter.publish_generation(manifest, gate, ledger=ledger)
    assert result['published'] and result['readback_verified'] and result['legacy_seed_preserved']
    assert db.data['documents'][0] == old['documents'][0]
    assert db.data['units'][0] == old['units'][0]
    new = db.data['units'][1]
    structured = json.loads(new['metadata']['structured_observation'])
    assert structured['conditions'] == row['conditions'] and structured['evidence_ids'] == row['evidence_ids']
    assert structured['evidence'] == row['evidence']
    assert new['metadata']['observation_content_sha256'] == merge.observation_content_hash(row)
    assert new['metadata']['generation_sha256'] == manifest['manifest_sha256']
    assert len(db.embeds) == 1 and db.commits == 1
    again = adapter.publish_generation(manifest, gate, ledger=ledger)
    assert again['readback_verified'] and len(db.embeds) == 1 and db.commits == 1
    delete = [p for s, p in db.sql if s.strip().startswith('DELETE')]
    assert delete == []
    assert all(not isinstance(p, tuple) or len(p) < 2 or not isinstance(p[1], str) or '%' not in p[1]
               for s, p in db.sql if 'FROM documents' in s and 'backup' in s)


@pytest.mark.parametrize('field', ['manifest_sha256', 'bank', 'root', 'decision'])
def test_stale_or_forged_gate_blocks_before_sql_and_embeddings(env, field):
    mod, adapter, ledger, db, manifest, proof, row = env
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    bad = copy.deepcopy(gate)
    bad[field] = 'wrong'
    before = len(db.sql)
    with pytest.raises(ValueError):
        adapter.publish_generation(manifest, bad, ledger=ledger)
    assert len(db.sql) == before and db.embeds == [] and db.commits == 0


def test_manifest_tampering_blocks_before_sql_and_embeddings(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    bad = dict(manifest, manifest_sha256='0' * 64)
    before = len(db.sql)
    with pytest.raises(ValueError):
        adapter.publish_generation(bad, gate, ledger=ledger)
    assert len(db.sql) == before and db.embeds == []


def test_lost_commit_ack_readback_skips_reembedding(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    db.lose_ack = True
    result = adapter.publish_generation(manifest, gate, ledger=ledger)
    assert result['readback_verified'] and result['legacy_seed_preserved']
    assert len(db.embeds) == 1 and db.commits == 1
    assert adapter.publish_generation(manifest, gate, ledger=ledger)['published']
    assert len(db.embeds) == 1


@pytest.mark.parametrize('vectors', [lambda texts: [None for _ in texts], lambda texts: [[0.] * 3 for _ in texts]])
def test_bad_embeddings_fail_closed_without_writes(env, vectors):
    mod, adapter, ledger, db, manifest, proof, row = env
    adapter.embeddings = lambda texts, *_a, **_kw: (vectors(texts), 'fixture')
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    old = copy.deepcopy(db.data)
    with pytest.raises(RuntimeError):
        adapter.publish_generation(manifest, gate, ledger=ledger)
    assert db.data == old and db.commits == 0


def test_gate_runs_fresh_generic_local_pairs_with_exact_scripts_and_root(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    calls = []
    adapter.run_json = lambda cmd, **kw: (calls.append(cmd), fake_run(cmd, **kw))[1]
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    assert gate['decision'] == 'eligible_for_local_proposal' and all(gate['checks'].values())
    assert len(calls) == 5 and all(Path(c[1]).parent == ROOT for c in calls)
    conflict = calls[0]
    assert conflict[conflict.index('--offline-root') + 1] == manifest['root']
    assert all('--no-hindsight-observations' not in c for c in calls[1:])
    assert {c[c.index('--benchmark') + 1] for c in calls[1:]} == {adapter.config['generic_benchmark'], adapter.config['local_benchmark']}
    assert not Path(gate['gate_path']).is_relative_to(Path(manifest['root']))
    assert len(gate['artifacts']) >= 7


def test_gate_wrong_root_or_missing_pair_or_no_gain_blocks(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    def stale(cmd, **kw):
        r = fake_run(cmd, **kw)
        if Path(cmd[1]).name == 'hindsight_eval.py' and '--use-local-cards' in cmd:
            r['cards_root'] = 'stale-root'
        return r
    adapter.run_json = stale
    with pytest.raises(ValueError):
        adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    def no_gain(cmd, **kw):
        r = fake_run(cmd, **kw)
        if 'summary' in r and 'layered_avg_term_recall' in r['summary']:
            r['summary']['layered_avg_term_recall'] = 0.5
            r['results'][0]['layered']['term_recall'] = 0.5
        return r
    adapter.run_json = no_gain
    g = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    assert g['decision'] == 'blocked_keep_local_only'
    with pytest.raises(ValueError):
        adapter.publish_generation(manifest, g, ledger=ledger)
    assert db.embeds == []


def test_scoped_proof_strict_zero_and_exact_cohort(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    for changed in ({'pending_in_scope': False}, {'pending_operations': None}, {'scope': ['old']}, {'bank': 'bank-b'}):
        with pytest.raises(ValueError):
            adapter.gate(manifest, ledger=ledger, scoped_proof=dict(proof, **changed))
    assert db.embeds == []


def test_runtime_aggregate_scoped_proof_is_bound_to_exact_ledger_receipts(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    native = {'bank': ledger.bank, 'scope': ['arrival-batch:batch-a'], 'ready': True,
              'pending_in_scope': 0, 'failed_in_scope': 0, 'operation_ids': ['native-operation']}
    original = ledger.get_batch
    ledger.get_batch = lambda bid: dict(original(bid), receipts={'consolidated': {'scoped_proof': native}})
    aggregate = {'bank': ledger.bank, 'ready': True, 'pending_in_scope': 0,
                 'scopes': [native['scope']], 'batches': [native], 'legacy_pending': 56}
    assert adapter.gate(manifest, ledger=ledger, scoped_proof=aggregate)['checks']['scoped_drain']
    for changed in ({'pending_in_scope': False}, {'failed_in_scope': None}, {'scope': ['arrival-batch:wrong']}):
        with pytest.raises(ValueError):
            adapter.gate(manifest, ledger=ledger, scoped_proof=dict(aggregate, batches=[dict(native, **changed)]))
    ledger.get_batch = original
    with pytest.raises(ValueError):
        adapter.gate(manifest, ledger=ledger, scoped_proof=aggregate)


def test_changed_node_replaces_only_itself_not_other_incremental_nodes(env):
    mod, adapter, ledger, db, first, proof, row = env
    gate = adapter.gate(first, ledger=ledger, scoped_proof=proof)
    adapter.publish_generation(first, gate, ledger=ledger)
    first_doc = db.data['documents'][1]['id']
    rows = merge.load_manifest_artifacts(Path(first['root']), bank=ledger.bank)[2]
    assert rows is not None
    other = dict(row, insight='Other node', conditions={'different': True})
    second = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=merge.merge_observations(rows, [other]), consumed_batch_ids=['batch-b'])
    second_proof = dict(proof, scope=['arrival-batch:batch-b'])
    adapter.publish_generation(second, adapter.gate(second, ledger=ledger, scoped_proof=second_proof), ledger=ledger)
    assert len(db.data['units']) == 3
    contribution = dict(row, evidence_ids=['extra-evidence'])
    rows2 = merge.load_manifest_artifacts(Path(second['root']), bank=ledger.bank)[2]
    assert rows2 is not None
    third = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=second,
        observations=merge.merge_observations(rows2, [contribution]), consumed_batch_ids=['batch-c'])
    proof3 = dict(proof, scope=['arrival-batch:batch-c'])
    result = adapter.publish_generation(third, adapter.gate(third, ledger=ledger, scoped_proof=proof3), ledger=ledger)
    assert result['target_document_ids'] == [first_doc] and len(db.data['units']) == 3
    new = next(u for u in db.data['units'] if u['document_id'] == first_doc)
    details = json.loads(new['metadata']['structured_observation'])
    assert len(details['contributions']) == 2 and len(details['evidence_ids']) == 151
    assert len(db.embeds) == 3


def test_readback_refuses_null_vectors_mutated_seed_and_forged_metadata(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    adapter.publish_generation(manifest, adapter.gate(manifest, ledger=ledger, scoped_proof=proof), ledger=ledger)
    saved = copy.deepcopy(db.data)
    db.data['units'][1]['embedding'] = None
    assert not adapter.readback_generation(manifest, ledger=ledger)['readback_verified']
    db.data = copy.deepcopy(saved)
    db.data['units'][1]['metadata']['structured_observation'] = '{}'
    assert not adapter.readback_generation(manifest, ledger=ledger)['readback_verified']
    db.data = copy.deepcopy(saved)
    db.data['units'][0]['future_field']['keep'].append('corruption')
    result = adapter.readback_generation(manifest, ledger=ledger)
    assert not result['legacy_seed_preserved'] and not result['published']


def test_missing_fresh_pair_and_changed_report_hash_block_before_embedding(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    good = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    forged = copy.deepcopy(good)
    forged['pairs'] = forged['pairs'][:1]
    forged['gate_sha256'] = merge.canonical_hash({k: v for k, v in forged.items() if k != 'gate_sha256'})
    Path(forged['gate_path']).write_text(json.dumps(forged))
    with pytest.raises(ValueError):
        adapter.publish_generation(manifest, forged, ledger=ledger)
    Path(good['gate_path']).write_text(json.dumps(good))
    Path(good['pairs'][0]['cards_eval']).write_text('{}')
    with pytest.raises(ValueError):
        adapter.publish_generation(manifest, good, ledger=ledger)
    assert db.embeds == [] and db.commits == 0


def test_export_null_or_wrong_dimension_seed_blocks_without_artifact(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    for changes in ({'embedding': None}, {'embedding_dimension': 768}):
        original = copy.deepcopy(db.data['units'][0])
        db.data['units'][0].update(changes)
        path = ledger.root / ('bad-seed-' + uuid.uuid4().hex + '.json')
        with pytest.raises(ValueError):
            adapter.export_seed(path)
        assert not path.exists()
        db.data['units'][0] = original
    assert db.embeds == [] and db.commits == 0


def test_seed_list_artifact_and_provenance_binding_match_real_initializer(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    exported = json.loads(Path(ledger.seed['artifact_path']).read_text())
    path = ledger.root / 'opaque-array.json'
    path.write_text(json.dumps(exported['observations']))
    ledger.seed.update(artifact_path=str(path), artifact_hash=hashlib.sha256(path.read_bytes()).hexdigest())
    assert adapter.gate(manifest, ledger=ledger, scoped_proof=proof)['checks']['coverage']
    ledger.seed['provenance'] = dict(ledger.seed['provenance'], seed_sha256='f' * 64)
    with pytest.raises(ValueError):
        adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    assert db.embeds == []


def test_sql_failure_rolls_back_delta_and_preserves_seed(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    before = copy.deepcopy(db.data)
    def fail_after_document_insert(cur, sql, rows, **kw):
        result = db.values(cur, sql, rows, **kw)
        if 'INSERT INTO chunks' in sql:
            raise RuntimeError('isolated write failure')
        return result
    adapter.execute_values = fail_after_document_insert
    with pytest.raises(RuntimeError):
        adapter.publish_generation(manifest, gate, ledger=ledger)
    assert db.data == before and db.commits == 0
    assert adapter.readback_generation(manifest, ledger=ledger)['legacy_seed_preserved']


def test_selective_backup_contains_full_old_delta_vector_only(env):
    mod, adapter, ledger, db, first, proof, row = env
    adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    previous = copy.deepcopy(db.data['units'][1])
    rows = merge.load_manifest_artifacts(Path(first['root']), bank=ledger.bank)[2]
    assert rows is not None
    contribution = dict(row, evidence_ids=['new'])
    second = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=merge.merge_observations(rows, [contribution]), consumed_batch_ids=['batch-b'])
    proof2 = dict(proof, scope=['arrival-batch:batch-b'])
    result = adapter.publish_generation(second, adapter.gate(second, ledger=ledger, scoped_proof=proof2), ledger=ledger)
    backup = json.loads(Path(result['publish_report']['backup_path']).read_text())
    assert backup['document_count'] == 1 and backup['unit_count'] == 1
    assert backup['memory_units'] == [previous] and backup['memory_units'][0]['embedding'] is not None
    assert all(d['id'].startswith(mod.INCREMENTAL_PREFIX) for d in backup['documents'])


def test_unchanged_candidate_and_lossy_parent_rewrite_block(env):
    mod, adapter, ledger, db, first, proof, row = env
    rows = merge.load_manifest_artifacts(Path(first['root']), bank=ledger.bank)[2]
    assert rows is not None
    unchanged = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=rows, consumed_batch_ids=['batch-b'])
    with pytest.raises(ValueError):
        adapter.gate(unchanged, ledger=ledger, scoped_proof=dict(proof, scope=['arrival-batch:batch-b']))
    lost = [dict(r, evidence_ids=['lost-other-evidence']) if r.get('kind') != 'opaque_legacy' else r for r in rows]
    rewritten = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=lost, consumed_batch_ids=['batch-c'])
    with pytest.raises(ValueError):
        adapter.gate(rewritten, ledger=ledger, scoped_proof=dict(proof, scope=['arrival-batch:batch-c']))
    assert db.embeds == [] and db.commits == 0


def test_missing_sql_driver_blocks_without_paid_or_exec_fallback(env, monkeypatch):
    mod, adapter, ledger, db, manifest, proof, row = env
    real = mod.importlib.import_module
    def missing(name):
        if name == 'hindsight_offline_v2_publish':
            raise ModuleNotFoundError('psycopg2')
        return real(name)
    monkeypatch.setattr(mod.importlib, 'import_module', missing)
    with pytest.raises(RuntimeError, match='SQL driver unavailable'):
        mod.PublicationAdapter(adapter.config, ledger.bank, embeddings=db.encode, connect=db.connect)
    assert db.embeds == [] and db.commits == 0


def test_generated_source_artifact_hash_is_rechecked_before_publication(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    path = ledger.root / 'generated-source.json'
    path.write_text('{"new":"only"}')
    original = ledger.get_batch
    def batch(bid):
        result = original(bid)
        result['receipts']['generated'] = {'artifact_path': str(path),
            'artifact_hash': hashlib.sha256(b'{"new":"only"}').hexdigest()}
        return result
    ledger.get_batch = batch
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    path.write_text('{"changed":"after-freeze"}')
    before = len(db.sql)
    with pytest.raises(ValueError, match='source coverage artifact hash mismatch'):
        adapter.publish_generation(manifest, gate, ledger=ledger)
    assert len(db.sql) == before and db.embeds == []


@pytest.mark.parametrize('bad_report', ['missing_metrics', 'duplicate_cases', 'missing_conflict_decision'])
def test_malformed_fresh_gate_reports_fail_closed(env, bad_report):
    mod, adapter, ledger, db, manifest, proof, row = env
    def malformed(command, **kw):
        report = fake_run(command, **kw)
        if Path(command[1]).name == 'hindsight_eval.py':
            if bad_report == 'missing_metrics':
                report['summary'].pop('layered_avg_score')
            elif bad_report == 'duplicate_cases':
                report['results'] *= 2
                report['summary']['case_count'] = 2
        elif bad_report == 'missing_conflict_decision':
            report.pop('decision')
        return report
    adapter.run_json = malformed
    with pytest.raises(ValueError):
        adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    assert db.embeds == [] and db.commits == 0


def test_eval_summary_must_match_exhaustive_per_case_metrics(env):
    _, adapter, ledger, db, manifest, proof, _ = env
    def wrong_aggregate(cmd, **kw):
        report = fake_run(cmd, **kw)
        if Path(cmd[1]).name == 'hindsight_eval.py' and '--use-local-cards' in cmd:
            report['summary']['layered_avg_score'] += 1
        return report
    adapter.run_json = wrong_aggregate
    with pytest.raises(ValueError, match='aggregate'):
        adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    assert db.embeds == [] and db.commits == 0


@pytest.mark.parametrize('mode', ['baseline', 'cards'])
@pytest.mark.parametrize('defect', ['missing_case', 'extra_case', 'nan_summary', 'nan_case', 'missing_case_metric'])
def test_benchmark_exact_set_and_finite_per_case_metrics(env, mode, defect):
    _, adapter, ledger, db, manifest, proof, _ = env
    def invalid(cmd, **kw):
        report = fake_run(cmd, **kw)
        if Path(cmd[1]).name != 'hindsight_eval.py' or ('--use-local-cards' in cmd) != (mode == 'cards'):
            return report
        if defect == 'missing_case':
            for label in ('generic', 'local'):
                Path(adapter.config[label + '_benchmark']).write_text('{"id":"case"}\n{"id":"omitted"}\n')
        elif defect == 'extra_case':
            report['results'][0]['id'] = 'undeclared'
        elif defect == 'nan_summary':
            report['summary']['layered_avg_score'] = float('nan')
        elif defect == 'nan_case':
            report['results'][0]['layered']['term_recall'] = float('nan')
        else:
            report['results'][0]['layered'].pop('term_recall')
        return report
    adapter.run_json = invalid
    with pytest.raises(ValueError):
        adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    assert db.embeds == [] and db.commits == 0


def test_incremental_update_keeps_dependency_graph_and_backups(env):
    _, adapter, ledger, db, first, proof, row = env
    adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    old_unit = copy.deepcopy(db.data['units'][1])
    uid = old_unit['id']
    # Explicit SQL-double graph: document DELETE must cascade these, UPDATE must not.
    db.data.update(chunks=[{'chunk_id': old_unit['document_id'] + '::chunk::0000',
        'document_id': old_unit['document_id'], 'bank_id': ledger.bank, 'extra': 'kept'}],
        unit_entities=[{'unit_id': uid, 'entity_id': 'external-entity'}],
        links=[{'from_unit_id': db.data['units'][0]['id'], 'to_unit_id': uid, 'bank_id': ledger.bank},
               {'from_unit_id': uid, 'to_unit_id': db.data['units'][0]['id'], 'bank_id': ledger.bank}])
    external = {'id': str(uuid.uuid4()), 'bank_id': ledger.bank, 'document_id': 'external-source', 'source_memory_ids': [uid]}
    db.data['units'].append(external)
    before_graph = copy.deepcopy({k: db.data[k] for k in ('chunks', 'unit_entities', 'links')})
    rows = merge.load_manifest_artifacts(Path(first['root']), bank=ledger.bank)[2]
    assert rows is not None
    second = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=merge.merge_observations(rows, [dict(row, evidence_ids=['new-evidence'])]), consumed_batch_ids=['batch-b'])
    out = adapter.publish_generation(second, adapter.gate(second, ledger=ledger,
        scoped_proof=dict(proof, scope=['arrival-batch:batch-b'])), ledger=ledger)
    updated = next(u for u in db.data['units'] if u['document_id'] == old_unit['document_id'])
    assert updated['id'] == uid and external in db.data['units']
    assert all(r in db.data[k] for k, rows in before_graph.items() if k != 'chunks' for r in rows)
    assert db.data['chunks'][0]['chunk_id'] == before_graph['chunks'][0]['chunk_id']
    assert db.data['chunks'][0]['extra'] == 'kept'
    assert not any(s.strip().startswith('DELETE') for s, _ in db.sql)
    backup = json.loads(Path(out['publish_report']['backup_path']).read_text())
    assert backup['chunks'] == before_graph['chunks']
    assert backup['memory_links'] == before_graph['links']
    assert backup['unit_entities'] == before_graph['unit_entities']


@pytest.mark.parametrize('mutate', ['remove', 'vector', 'metadata'])
def test_all_parent_rows_block_drift_before_embedding(env, mutate):
    _, adapter, ledger, db, first, proof, row = env
    adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    rows = merge.load_manifest_artifacts(Path(first['root']), bank=ledger.bank)[2]
    assert rows is not None
    second = merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=merge.merge_observations(rows, [dict(row, insight='another node')]), consumed_batch_ids=['batch-b'])
    gate = adapter.gate(second, ledger=ledger, scoped_proof=dict(proof, scope=['arrival-batch:batch-b']))
    if mutate == 'remove':
        db.data['units'].pop()
    elif mutate == 'vector':
        db.data['units'][1]['embedding'] = '[' + ','.join(['1'] * 1024) + ']'
    else:
        db.data['units'][1]['metadata']['foreign_change'] = 'unexpected'
    before = len(db.embeds)
    with pytest.raises((ValueError, RuntimeError)):
        adapter.publish_generation(second, gate, ledger=ledger)
    assert len(db.embeds) == before and db.commits == 1


def test_default_eval_runner_binds_transitive_layered_module_to_script_dir(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    scripts = ledger.root / 'fixture-scripts'
    scripts.mkdir()
    (scripts / 'hindsight_recall_layered.py').write_text('VALUE = "configured-fixture"\n')
    (scripts / 'hindsight_eval.py').write_text(
        'import importlib.util,json,pathlib\n'
        'LAYERED_PATH = pathlib.Path("/definitely-not-installed/layered.py")\n'
        'def main():\n'
        '    spec=importlib.util.spec_from_file_location("probe",LAYERED_PATH)\n'
        '    module=importlib.util.module_from_spec(spec)\n'
        '    spec.loader.exec_module(module)\n'
        '    print(json.dumps({"loaded":module.VALUE,"path":str(LAYERED_PATH)}))\n')
    result = adapter._run_json([sys.executable, str(scripts / 'hindsight_eval.py'), '--json'])
    assert result == {'loaded': 'configured-fixture', 'path': str(scripts / 'hindsight_recall_layered.py')}
    assert db.embeds == [] and db.commits == 0


def test_publication_freeze_exact_bytes_and_invalid_value_has_no_artifact(tmp_path):
    mod = importlib.import_module('hindsight_incremental_publication')
    value = {'observations': [{'vector': '[0.1,-0.0]', 'text': '原样😀𐀀',
                              'nested': (None, 1e-20, False)}], 'float': -0.0}
    expected = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                           allow_nan=False) + '\n').encode('utf-8')
    path = tmp_path / 'export.json'
    receipt = mod._freeze(path, value)
    assert path.read_bytes() == expected
    assert receipt == {'path': str(path), 'sha256': hashlib.sha256(expected).hexdigest()}
    assert path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        mod._freeze(path, {'changed': True})
    assert path.read_bytes() == expected
    invalid = tmp_path / 'invalid.json'
    with pytest.raises(ValueError):
        mod._freeze(invalid, {'nonfinite': float('nan')})
    assert not invalid.exists()


@pytest.mark.linux_only
@pytest.mark.parametrize('operation', ['freeze', 'artifact'])
def test_publication_large_seed_serialization_in_memory_bounded_child(tmp_path, operation):
    import os
    import subprocess
    program = r'''
import hashlib, json, pathlib, resource, sys
sys.path.insert(0, sys.argv[1])
import hindsight_incremental_publication as publication
root, operation = pathlib.Path(sys.argv[2]), sys.argv[3]
value = {'observations': [{'id': f'unit-{i}', 'text': f'原样😀 {i}',
                          'vector': '[' + ','.join(['0.12345678901'] * 1024) + ']',
                          'opaque': {'nested': [None, True, '𐀀']}}
                         for i in range(4000)], 'bank': 'synthetic'}
encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
digest, byte_count = hashlib.sha256(), 0
for part in encoder.iterencode(value):
    data = part.encode('utf-8'); digest.update(data); byte_count += len(data)
digest.update(b'\n'); byte_count += 1
expected = digest.hexdigest()
path = root / 'large-seed.json'
if operation == 'artifact':
    with path.open('wb') as stream:
        for part in encoder.iterencode(value):
            stream.write(part.encode('utf-8'))
        stream.write(b'\n')
    path.chmod(0o600)
size = int(next(line.split()[1] for line in pathlib.Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmSize:'))) * 1024
headroom = 128 if operation == 'freeze' else 16
resource.setrlimit(resource.RLIMIT_AS, (size + headroom * 1024 * 1024,) * 2)
receipt = publication._freeze(path, value) if operation == 'freeze' else publication._artifact(path)
assert receipt == {'path': str(path), 'sha256': expected}
assert path.stat().st_size == byte_count
if operation == 'freeze':
    assert publication.merge._read_json(path, expected_hash=expected) == value
print(json.dumps({'operation': operation, 'rows': len(value['observations']), 'bytes': byte_count,
                  'golden_sha256': expected, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
'''
    result = subprocess.run([sys.executable, '-c', program, str(ROOT), str(tmp_path), operation],
                            capture_output=True, text=True, timeout=120,
                            env={**os.environ, 'HERMES_HOME': str(tmp_path / 'isolated-home')})
    assert result.returncode == 0, result.stdout + result.stderr
    proof = json.loads(result.stdout)
    assert proof['rows'] == 4000 and proof['operation'] == operation
    (tmp_path / f'memory-{operation}.json').write_text(result.stdout)
    print(result.stdout.strip())
