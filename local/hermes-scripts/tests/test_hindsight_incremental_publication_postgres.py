"""Real disposable PostgreSQL/pgvector, reconstructed minimal native schema.

No production DSN/service/API is used. Embeddings and gate reports remain explicit
fixtures. This tests actual psycopg2 SQL, FK/row-lock/rollback/graph semantics, not
native retain/consolidation workers or deployed production schema acceptance.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_hindsight_incremental_publication import Ledger, fake_run
import hindsight_incremental_merge as merge


@pytest.fixture(scope='module')
def cluster(tmp_path_factory):
    psycopg2 = pytest.importorskip('psycopg2')
    candidates = [Path(p).parent for p in [shutil.which('postgres')] if p]
    candidates += [Path.home() / '.hindsight-docker/installation/18.1.0/bin']
    binaries = next((p for p in candidates if all((p / n).is_file() for n in ('initdb', 'pg_ctl', 'postgres'))), None)
    if binaries is None or os.getuid() == 0:
        pytest.skip('disposable PG requires existing initdb/pg_ctl/postgres binaries and non-root user')
    scratch = tmp_path_factory.getbasetemp().parent
    root = Path(tempfile.mkdtemp(prefix='pgpub-', dir=scratch))
    data, sock = root / 'db', root / 's'
    sock.mkdir(mode=0o700)
    subprocess.run([str(binaries / 'initdb'), '-D', str(data), '-U', 'pub_scratch', '-A', 'trust', '--no-locale'],
        check=True, capture_output=True, text=True, timeout=30)
    started = False
    try:
        subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-l', str(root / 'postgres.log'),
            '-o', f"-k {sock} -h '' -p 6543 -F -c max_connections=10", '-w', 'start'],
            check=True, capture_output=True, text=True, timeout=30)
        started = True
        dsn = f'dbname=postgres user=pub_scratch host={sock} port=6543'
        with psycopg2.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute('CREATE EXTENSION vector')
                cur.execute('SELECT version(), extversion FROM pg_extension WHERE extname=%s', ('vector',))
                print('DISPOSABLE_REAL_PG', cur.fetchone())
        yield dsn
    finally:
        if started:
            subprocess.run([str(binaries / 'pg_ctl'), '-D', str(data), '-m', 'immediate', '-w', 'stop'],
                check=True, capture_output=True, text=True, timeout=30)
        # Retain logs/schema/test artifacts in scratch; do not remove user data.


SCHEMA = """
CREATE TABLE documents (id text, bank_id text, original_text text, content_hash text, metadata jsonb DEFAULT '{}',
    retain_params jsonb, tags text[], created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(), PRIMARY KEY (id,bank_id));
CREATE TABLE chunks (chunk_id text PRIMARY KEY, document_id text, bank_id text, chunk_index int, chunk_text text, content_hash text,
    extra text, FOREIGN KEY(document_id,bank_id) REFERENCES documents(id,bank_id) ON DELETE CASCADE);
CREATE TABLE memory_units (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text, text text, embedding vector(1024),
    event_date timestamptz, occurred_start timestamptz, occurred_end timestamptz, mentioned_at timestamptz,
    context text, fact_type text, metadata jsonb DEFAULT '{}', chunk_id text REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    document_id text, tags text[], observation_scopes jsonb, text_signals text, source_memory_ids uuid[],
    created_at timestamptz DEFAULT now(), updated_at timestamptz DEFAULT now(),
    FOREIGN KEY(document_id,bank_id) REFERENCES documents(id,bank_id) ON DELETE CASCADE);
CREATE TABLE entities (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), canonical_name text, bank_id text, metadata jsonb,
    mention_count int, last_seen timestamptz DEFAULT now());
CREATE UNIQUE INDEX entity_name_bank ON entities(bank_id,lower(canonical_name));
CREATE TABLE unit_entities (unit_id uuid REFERENCES memory_units(id) ON DELETE CASCADE,
    entity_id uuid REFERENCES entities(id) ON DELETE CASCADE, PRIMARY KEY(unit_id,entity_id));
CREATE TABLE memory_links (from_unit_id uuid REFERENCES memory_units(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    to_unit_id uuid REFERENCES memory_units(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
    link_type text DEFAULT 'semantic', weight float, bank_id text, entity_id uuid REFERENCES entities(id),
    PRIMARY KEY(from_unit_id,to_unit_id,link_type));
"""


@pytest.fixture
def pg_env(cluster, tmp_path):
    import psycopg2
    from psycopg2.extras import execute_values
    schema = 'case_' + uuid.uuid4().hex
    with psycopg2.connect(cluster) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE SCHEMA ' + schema)
            cur.execute('SET search_path TO ' + schema + ',public')
            cur.execute(SCHEMA)
            old = 'hermes-offline-canonical::topic::legacy::123456789abc'
            cur.execute('INSERT INTO documents(id,bank_id,original_text,content_hash,metadata) VALUES(%s,%s,%s,%s,%s::jsonb)',
                (old, 'bank-a', 'opaque doc', hashlib.sha256(b'opaque doc').hexdigest(), '{"unknown":[1,null]}'))
            legacy_id = str(uuid.uuid4())
            cur.execute("INSERT INTO memory_units(id,bank_id,document_id,text,fact_type,embedding,metadata) VALUES(%s::uuid,%s,%s,%s,'observation',%s::vector,%s::jsonb)",
                (legacy_id, 'bank-a', old, 'opaque original | not guessed', '[' + ','.join(['0'] * 1024) + ']', '{"legacy":"unchanged"}'))
    dsn = cluster + f' options=-csearch_path={schema},public'
    calls = []
    def connect(exact):
        assert exact == dsn and 'pub_scratch' in exact and 'pgpub-' in exact
        return psycopg2.connect(exact)
    def encode(texts, *_a, **_kw):
        calls.append(list(texts))
        return [[0.25] * 1024 for _ in texts], 'explicit_fake_embedding'
    config = {'dsn': dsn, 'script_dir': str(ROOT), 'publisher_python': sys.executable,
        'api': 'http://127.0.0.1:1', 'generic_benchmark': str(tmp_path / 'generic.jsonl'),
        'local_benchmark': str(tmp_path / 'local.jsonl')}
    for label in ('generic', 'local'):
        Path(config[label + '_benchmark']).write_text('{"id":"case"}\n')
    adapter = importlib.import_module('hindsight_incremental_publication').PublicationAdapter(
        config, 'bank-a', connect=connect, run_json=fake_run, embeddings=encode, execute_values=execute_values)
    ledger = Ledger(tmp_path)
    exported = adapter.export_seed(tmp_path / 'seed.json')
    ledger.seed = {'bank': ledger.bank, 'artifact_path': str(tmp_path / 'seed.json'),
        'artifact_hash': hashlib.sha256((tmp_path / 'seed.json').read_bytes()).hexdigest(),
        'provenance': exported['provenance'], 'remote_seed': exported['remote_seed']}
    row = {'insight': 'a stable typed node', 'topic': 'pg', 'type': 'rule',
        'applicability': 'under the exact condition', 'evidence_ids': ['evidence-original']}
    manifest = merge.write_generation(tmp_path / 'canonical', bank=ledger.bank, parent=exported['provenance'],
        observations=merge.merge_observations(exported['observations'], [row]), consumed_batch_ids=['batch-a'])
    proof = {'bank': ledger.bank, 'scope': ['arrival-batch:batch-a'], 'ready': True, 'pending_in_scope': 0, 'pending_operations': 0}
    return adapter, ledger, manifest, proof, row, legacy_id, connect, calls


def next_generation(ledger, first, row):
    rows = merge.load_manifest_artifacts(Path(first['root']), bank=ledger.bank)[2]
    assert rows is not None
    return merge.write_generation(ledger.root / 'canonical', bank=ledger.bank, parent=first,
        observations=merge.merge_observations(rows, [row]), consumed_batch_ids=['batch-b'])


def test_real_pg_exact_seed_parent_locks_block_noncooperating_writers(pg_env):
    import psycopg2
    from psycopg2.extras import execute_values
    adapter, ledger, first, proof, row, legacy_id, connect, calls = pg_env
    out = adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    with connect(adapter.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id::text FROM memory_units WHERE document_id=%s', (out['target_document_ids'][0],))
            typed_id = cur.fetchone()[0]
    second = next_generation(ledger, first, dict(row, evidence_ids=['another']))
    proof2 = dict(proof, scope=['arrival-batch:batch-b'])
    blocked = []
    def interleave(cur, sql, rows, **kwargs):
        if 'INSERT INTO documents' in sql:
            statements = [
                ("UPDATE memory_units SET text='foreign-writer' WHERE id=%s::uuid", (uid,))
                for uid in (legacy_id, typed_id)]
            statements += [
                ("UPDATE documents SET metadata='{}'::jsonb WHERE id=%s", (out['target_document_ids'][0],)),
                ("INSERT INTO memory_units(bank_id,document_id,text,fact_type) SELECT bank_id,document_id,'phantom','observation' FROM memory_units WHERE id=%s::uuid", (legacy_id,)),
            ]
            for statement, params in statements:
                other = connect(adapter.dsn)
                try:
                    with other.cursor() as foreign:
                        foreign.execute("SET lock_timeout='150ms'")
                        with pytest.raises(psycopg2.errors.LockNotAvailable):
                            foreign.execute(statement, params)
                    blocked.append(statement)
                finally:
                    other.rollback(); other.close()
        return execute_values(cur, sql, rows, **kwargs)
    adapter.execute_values = interleave
    result = adapter.publish_generation(second, adapter.gate(second, ledger=ledger, scoped_proof=proof2), ledger=ledger)
    assert result['readback_verified'] and len(blocked) == 4
    assert len(calls) == 2


def test_real_pg_seed_change_between_embedding_and_fence_rolls_back(pg_env):
    adapter, ledger, first, proof, row, legacy_id, connect, calls = pg_env
    original = adapter.embeddings
    def changed_seed(*args, **kwargs):
        result = original(*args, **kwargs)
        with connect(adapter.dsn) as other:
            with other.cursor() as cur:
                cur.execute("UPDATE memory_units SET metadata=metadata || '{\"foreign\":true}'::jsonb WHERE id=%s::uuid", (legacy_id,))
        return result
    adapter.embeddings = changed_seed
    with pytest.raises(ValueError, match='changed before transaction'):
        adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    with connect(adapter.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents WHERE id LIKE 'hermes-offline-canonical::incremental::%'")
            assert cur.fetchone()[0] == 0
            cur.execute('SELECT metadata FROM memory_units WHERE id=%s::uuid', (legacy_id,))
            assert cur.fetchone()[0]['foreign'] is True
    assert len(calls) == 1


def test_real_pg_inplace_graph_identity_and_full_parent_drift(pg_env):
    adapter, ledger, first, proof, row, legacy_id, connect, calls = pg_env
    out = adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    docid = out['target_document_ids'][0]
    with connect(adapter.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id::text,chunk_id FROM memory_units WHERE document_id=%s', (docid,))
            uid, chunk = cur.fetchone()
            cur.execute("UPDATE chunks SET extra='preserved' WHERE chunk_id=%s", (chunk,))
            # Graph writes do not rewrite sealed seed/typed rows. Native FK paths
            # are exercised by inbound/outbound references and old associations.
            cur.execute("INSERT INTO memory_links(from_unit_id,to_unit_id,bank_id,weight) VALUES(%s::uuid,%s::uuid,'bank-a',0.9),(%s::uuid,%s::uuid,'bank-a',0.8)", (legacy_id,uid,uid,legacy_id))
            cur.execute('SELECT unit_id::text,entity_id::text FROM unit_entities WHERE unit_id=%s::uuid', (uid,))
            old_entities = cur.fetchall()
    second = next_generation(ledger, first, dict(row, evidence_ids=[legacy_id]))
    proof2 = dict(proof, scope=['arrival-batch:batch-b'])
    result = adapter.publish_generation(second, adapter.gate(second, ledger=ledger, scoped_proof=proof2), ledger=ledger)
    backup = json.loads(Path(result['publish_report']['backup_path']).read_text())
    assert len(backup['memory_links']) == 2 and backup['chunks'][0]['extra'] == 'preserved'
    with connect(adapter.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id::text,chunk_id,source_memory_ids::text[] FROM memory_units WHERE document_id=%s', (docid,))
            assert cur.fetchone() == (uid,chunk,[legacy_id])
            cur.execute('SELECT count(*) FROM memory_links WHERE from_unit_id=%s::uuid OR to_unit_id=%s::uuid', (uid,uid))
            assert cur.fetchone()[0] == 2
            cur.execute('SELECT unit_id::text,entity_id::text FROM unit_entities WHERE unit_id=%s::uuid', (uid,))
            assert set(old_entities) <= set(cur.fetchall())
            cur.execute('SELECT extra FROM chunks WHERE chunk_id=%s', (chunk,))
            assert cur.fetchone()[0] == 'preserved'
    third = next_generation(ledger, second, dict(row, insight='new independent node'))
    gate = adapter.gate(third, ledger=ledger, scoped_proof=proof2)
    with connect(adapter.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE memory_units SET embedding=%s::vector WHERE id=%s::uuid", ('[' + ','.join(['1']*1024) + ']',uid))
    before = len(calls)
    with pytest.raises(ValueError, match='typed parent'):
        adapter.publish_generation(third, gate, ledger=ledger)
    assert len(calls) == before


def test_real_pg_failed_changed_write_rolls_back_whole_graph(pg_env):
    from psycopg2.extras import execute_values
    adapter, ledger, first, proof, row, legacy_id, connect, calls = pg_env
    out = adapter.publish_generation(first, adapter.gate(first, ledger=ledger, scoped_proof=proof), ledger=ledger)
    with connect(adapter.dsn) as conn:
        with conn.cursor() as cur:
            cur.execute('SELECT id::text FROM memory_units WHERE document_id=%s', (out['target_document_ids'][0],))
            uid = cur.fetchone()[0]
            cur.execute("INSERT INTO memory_links(from_unit_id,to_unit_id,bank_id,weight) VALUES(%s::uuid,%s::uuid,'bank-a',0.9),(%s::uuid,%s::uuid,'bank-a',0.8)", (legacy_id,uid,uid,legacy_id))
    def snapshot():
        result = {}
        with connect(adapter.dsn) as conn:
            with conn.cursor() as cur:
                for table in ('documents', 'chunks', 'memory_units', 'entities', 'unit_entities', 'memory_links'):
                    cur.execute(f'SELECT to_jsonb(t) FROM {table} t ORDER BY to_jsonb(t)::text')
                    result[table] = cur.fetchall()
        return result
    before = snapshot()
    second = next_generation(ledger, first, dict(row, evidence_ids=[legacy_id]))
    gate = adapter.gate(second, ledger=ledger, scoped_proof=dict(proof, scope=['arrival-batch:batch-b']))
    def fail_after_update(cur, sql, rows, **kwargs):
        result = execute_values(cur, sql, rows, **kwargs)
        if 'INSERT INTO unit_entities' in sql:
            raise RuntimeError('explicit isolated write interruption')
        return result
    adapter.execute_values = fail_after_update
    with pytest.raises(RuntimeError, match='isolated write interruption'):
        adapter.publish_generation(second, gate, ledger=ledger)
    assert snapshot() == before
