"""Native 0.9 metadata gates: no text reads, no engine/provider writes."""
import copy
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_hindsight_incremental_runtime import make_runtime, batch, policy
from hindsight_incremental_delivery import _freeze, _json_bytes, _digest


def isolation_module():
    return importlib.import_module('hindsight_incremental_native_isolation')


class SQL:
    """Only metadata columns in the installed public schema are permitted."""
    def __init__(self, current):
        self.calls = []
        self.models = []
        self.documents = []
        self.live = []
        self.archive = []
        self.binding = []
        self.operation_exists = False
        self.current = current
    def cursor(self):
        return self
    def __enter__(self):
        return self
    def __exit__(self, *_):
        pass
    def execute(self, query, params=()):
        self.calls.append((query, copy.deepcopy(params)))
        assert query.lstrip().startswith('SELECT'), 'non-readonly SQL'
        assert 'original_text' not in query and 'SELECT text' not in query
        if 'arrival-operation-exists' in query:
            self.rows = [(self.operation_exists,)]
        elif 'mental_models' in query:
            self.rows = [(r,) for r in self.models]
        elif 'arrival-binding' in query:
            self.rows = self.binding
        elif 'invalidated_memory_units' in query:
            self.rows = [(r,) for r in self.archive]
        elif 'memory_units' in query:
            self.rows = [(r,) for r in self.live]
        elif 'documents' in query:
            self.rows = [(r,) for r in self.documents]
        else:
            raise AssertionError(query)
    def fetchall(self):
        return copy.deepcopy(self.rows)


def owned_payload(ledger, current, items):
    payload = {'async': True, 'operation_id': current['operation_id'], 'items': items}
    path, digest = _freeze(ledger.root, payload, 'retain')
    receipt = {'bank': current['bank'], 'batch_id': current['batch_id'],
               'operation_id': current['operation_id'], 'input_hash': 'a' * 64,
               'payload_path': path, 'payload_hash': digest}
    current['operation_payload_hash'] = digest
    current['payload_hash'] = 'a' * 64
    current['receipts']['generated'] = receipt
    current['receipts']['retained'].update(payload_hash=digest, input_hash='a' * 64,
                                          batch_id=current['batch_id'])
    current['receipts']['submitted'] = {k: receipt[k] for k in ('bank', 'batch_id', 'operation_id', 'payload_hash')}
    return payload


def setup(tmp_path, monkeypatch, *, existing=False):
    mod = isolation_module()
    runtime, rest, ledger = make_runtime(tmp_path)
    tag = rest.scope[0]
    item = {'document_id': batch()['document_id'], 'content': 'new immutable input',
            'tags': [tag], 'observation_scopes': [[tag]],
            'timestamp': '2026-10-11T00:00:00Z', 'metadata': {'kind': 'source'}}
    current = ledger.batches[0]
    payload = owned_payload(ledger, current, [item])
    sql = SQL(current)
    proof = {'pid': 123, 'boot_id': 'boot-fixture', 'start_ticks': '42',
             'executable': '/isolated/python', 'source_sha256': mod.NATIVE_TREE_SHA256,
             'memories': 'hindsight_api.engine.memories.postgres.PostgresMemories',
             'operation_validator': None, 'database_backend': 'postgresql'}
    observed = [proof]
    monkeypatch.setattr(mod, 'local_provenance', lambda *_: copy.deepcopy(observed[-1]))
    @contextmanager
    def readonly():
        yield sql
    runtime._publication_adapter = type('Publication', (), {'dsn': 'scratch-only', '_readonly': staticmethod(readonly)})()
    runtime._native_isolation_adapter = mod.NativeIsolation(runtime)
    if existing:
        sql.documents = [{'id': current['document_id'], 'bank_id': current['bank'], 'tags': [tag],
                          'content_hash': _digest(item['content'].encode())}]
        sql.live = [{'id': 'fact-a', 'bank_id': current['bank'], 'document_id': current['document_id'],
                     'fact_type': 'world', 'tags': [tag], 'observation_scopes': [[tag]], 'source_memory_ids': None},
                    {'id': 'fact-b', 'bank_id': current['bank'], 'document_id': current['document_id'],
                     'fact_type': 'experience', 'tags': [tag], 'observation_scopes': [[tag]], 'source_memory_ids': None}]
        sql.binding = [(True,)]
    return runtime, rest, ledger, sql, observed, item


def test_unknown_backend_blocks_before_any_window_write(tmp_path):
    # No explicit local proof, and an arbitrary REST schema is not attestation.
    runtime, rest, ledger = make_runtime(tmp_path)
    with pytest.raises(Exception, match='native_'):
        runtime.open_window(ledger)
    assert not any(c[0] in {'PATCH', 'POST'} for c in rest.calls)
    assert not (ledger.root / 'runtime-window.json').exists()


@pytest.mark.parametrize('defect', ['extension', 'validator', 'backend', 'hash', 'unknown', 'mental_auto', 'mental_cron', 'mental_unknown'])
def test_contract_is_dynamic_and_blocks_before_producer(tmp_path, monkeypatch, defect):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch)
    runtime.open_window(ledger)
    if defect == 'extension': observed[-1]['memories'] = 'custom.UnsafeMemories'
    elif defect == 'validator': observed[-1]['operation_validator'] = 'custom.WidenTags'
    elif defect == 'backend': observed[-1]['database_backend'] = 'remote'
    elif defect == 'hash': observed[-1]['source_sha256'] = 'f' * 64
    elif defect == 'unknown': observed[-1] = None
    elif defect == 'mental_auto': sql.models = [{'id': 'global-model', 'trigger': {'refresh_after_consolidation': True}}]
    elif defect == 'mental_cron': sql.models = [{'id': 'global-model', 'trigger': {'refresh_after_consolidation': False, 'refresh_cron': '0 * * * *'}}]
    else: sql.models = [{'id': 'global-model', 'trigger': None}]
    before = len(rest.calls)
    with pytest.raises(Exception, match='native_'):
        runtime.retain_items([item], operation_id=batch()['operation_id'])
    assert not any(c[0] == 'POST' for c in rest.calls[before:])


@pytest.mark.parametrize('defect', ['old_tag', 'superset', 'shared_fact', 'old_source', 'empty_source', 'archive', 'extra_fact', 'other_document', 'wrong_receipt'])
def test_scope_and_source_ids_are_proven_not_assumed(tmp_path, monkeypatch, defect):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch, existing=True)
    runtime.open_window(ledger)
    tag = rest.scope[0]
    if defect == 'old_tag': sql.live.append({'id': 'old', 'bank_id': ledger.bank, 'document_id': 'old-doc', 'fact_type': 'world', 'tags': [tag]})
    elif defect == 'superset': sql.live[0]['tags'] = [tag, 'legacy']
    elif defect == 'shared_fact': sql.live[0]['observation_scopes'] = [[]]
    elif defect in {'old_source', 'empty_source'}:
        sql.live.append({'id': 'observation-new', 'bank_id': ledger.bank, 'document_id': None,
                         'fact_type': 'observation', 'tags': [tag], 'observation_scopes': None,
                         'source_memory_ids': ['old-fact'] if defect == 'old_source' else []})
    elif defect == 'archive': sql.archive = [copy.deepcopy(sql.live[0])]
    elif defect == 'extra_fact': sql.live.append({**sql.live[0], 'id': 'unlisted-fact'})
    elif defect == 'other_document': sql.documents.append({'id': 'old-doc', 'bank_id': ledger.bank, 'tags': [tag]})
    else: sql.binding = [(False,)]
    with pytest.raises(Exception, match='native_'):
        runtime.consolidate_batch(batch(), policy())
    assert not any(c[0] == 'POST' for c in rest.calls)
    assert all('text' not in r for r in sql.live)


def test_readback_receipt_is_lossless_metadata_and_revalidated(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch, existing=True)
    window = runtime.open_window(ledger)
    facts = runtime.facts_for_batch(ledger.batches[0])
    assert {f['id'] for f in facts} == {'fact-a', 'fact-b'}
    proof = runtime.consolidate_batch(ledger.batches[0], policy())
    isolation = proof['native_isolation']
    assert isolation['source_ids'] == ['fact-a', 'fact-b']
    assert isolation['scope'] == rest.scope
    assert isolation['backend']['source_sha256'] == isolation_module().NATIVE_TREE_SHA256
    private = json.loads((ledger.root / 'runtime-window.json').read_text())
    assert private['scopes'][rest.scope[0]]['native_isolation'] == isolation
    raw = _json_bytes(isolation).decode()
    assert 'new immutable input' not in raw and 'scratch-only' not in raw and 'text' not in raw
    observed[-1]['start_ticks'] = '43'
    with pytest.raises(Exception, match='native_backend_changed'):
        runtime.consolidate_batch(ledger.batches[0], policy())


def test_reserved_collision_rejected_before_open_patch(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch)
    sql.documents = [{'id': 'old-doc', 'bank_id': ledger.bank, 'tags': rest.scope}]
    with pytest.raises(Exception, match='native_'):
        runtime.open_window(ledger)
    assert not any(c[0] in {'POST', 'PATCH'} for c in rest.calls)


def test_exact_wire_binding_and_scope_required_even_for_first_post(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch)
    runtime.open_window(ledger)
    for altered in ({**item, 'tags': [*rest.scope, 'old']},
                    {**item, 'content': 'different payload'}, {**item, 'update_mode': 'append'}):
        with pytest.raises(Exception):
            runtime.retain_items([altered], operation_id=batch()['operation_id'])
    assert not any(c[0] == 'POST' for c in rest.calls)
    assert runtime.retain_items([item], operation_id=batch()['operation_id'])['operation_id'] == batch()['operation_id']


def test_reserved_uuid_collision_without_facts_is_not_a_free_scope(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch)
    sql.operation_exists = True
    sql.binding = [(False,)]
    with pytest.raises(Exception, match='native_remote_payload_binding_unproved'):
        runtime.open_window(ledger)
    assert not rest.calls and not (ledger.root / 'runtime-window.json').exists()


def test_unknown_sql_truth_is_not_positive_ownership(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch, existing=True)
    sql.binding = [(1,)]
    with pytest.raises(Exception, match='native_remote_payload_binding_unproved'):
        runtime.open_window(ledger)
    assert not rest.calls


def test_old_private_observation_even_with_new_links_needs_owned_trigger(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch, existing=True)
    runtime.open_window(ledger)
    sql.live.append({'id': 'old-reserved-observation', 'bank_id': ledger.bank, 'document_id': None,
                     'fact_type': 'observation', 'tags': rest.scope, 'observation_scopes': None,
                     'source_memory_ids': ['fact-a']})
    with pytest.raises(Exception, match='native_observation_ownership_unproved'):
        runtime.consolidate_batch(ledger.batches[0], policy())
    assert not any(c[0] == 'POST' for c in rest.calls)


def test_clean_owned_observations_keep_all_ids_and_links(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch, existing=True)
    runtime.open_window(ledger)
    runtime.consolidate_batch(ledger.batches[0], policy())
    sql.live.append({'id': 'owned-observation', 'bank_id': ledger.bank, 'document_id': None,
                     'fact_type': 'observation', 'tags': rest.scope, 'observation_scopes': None,
                     'source_memory_ids': ['fact-a', 'fact-b']})
    proof = runtime.consolidate_batch(ledger.batches[0], policy())
    assert proof['native_isolation']['source_ids'] == ['fact-a', 'fact-b']
    assert proof['native_isolation']['observation_ids'] == ['owned-observation']
    assert sum(c[0] == 'POST' for c in rest.calls) == 1


def test_backend_drift_preserves_pending_cleanup_not_unproved_patch(tmp_path, monkeypatch):
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch)
    window = runtime.open_window(ledger)
    before = len(rest.calls)
    observed[-1]['start_ticks'] = '43'
    assert runtime.cleanup_window(ledger, window) == 'pending'
    assert not any(c[0] == 'PATCH' for c in rest.calls[before:])


def test_mental_auto_refresh_blocks_summary_llm_before_paid_call(tmp_path, monkeypatch):
    from test_hindsight_incremental_runtime import frozen_input
    runtime, rest, ledger, sql, observed, item = setup(tmp_path, monkeypatch, existing=True)
    runtime.open_window(ledger)
    sql.models = [{'id': 'global-old-history', 'trigger': {'refresh_after_consolidation': True}}]
    before = len(rest.calls)
    with pytest.raises(Exception, match='native_mental_model_refresh_unproved'):
        runtime.generate_summary(frozen_input(), retry_policy=policy())
    assert runtime.llm_calls == 0 and len(rest.calls) == before


@pytest.mark.parametrize('environment', [
    {'HINDSIGHT_API_MEMORIES_EXTENSION': 'unsafe:Store'},
    {'HINDSIGHT_API_OPERATION_VALIDATOR_EXTENSION': 'unsafe:WidenRecall'},
    {'HINDSIGHT_API_TENANT_EXTENSION': 'unsafe:ChangeSchema'},
    {'HINDSIGHT_API_DATABASE_BACKEND': 'oracle'},
    {'HINDSIGHT_API_DATABASE_SCHEMA': 'other'},
    {'HINDSIGHT_API_WORKERS': '2'}, {'HINDSIGHT_API_WORKER_ENABLED': 'false'},
    {'PYTHONPATH': '/unsafe'}, {'COVERAGE_PROCESS_START': '/unknown-config'},
])
def test_effective_environment_contract_is_not_an_operator_boolean(environment):
    mod = isolation_module()
    with pytest.raises(Exception, match='native_'):
        mod._environment(environment)
    assert mod._environment({}) is None


def test_source_changed_after_boot_is_never_a_loaded_code_attestation(tmp_path):
    import os
    import time
    mod = isolation_module()
    source = tmp_path / 'audited.py'
    source.write_text('trusted fixture')
    started = time.time() + 1
    assert mod._file_bytes(source, started) == b'trusted fixture'
    os.utime(source, (started + 10, started + 10))
    with pytest.raises(Exception, match='native_source_changed_since_start'):
        mod._file_bytes(source, started)


def test_relatime_access_bump_is_not_a_source_change(tmp_path):
    """Reading the file may update atime under relatime; that is access
    metadata, not a content change, and must not fail the probe."""
    import os
    import time
    mod = isolation_module()
    source = tmp_path / 'audited_relatime.py'
    source.write_text('trusted fixture')
    now = time.time()
    os.utime(source, (now - 2 * 86400, now - 3 * 86400))
    started = now + 1
    assert mod._file_bytes(source, started) == b'trusted fixture'


def test_real_runtime_noop_never_invokes_any_native_gate_or_network(tmp_path, monkeypatch):
    from test_hindsight_incremental_pipeline import modules, snapshot
    src, ld, delivery, flow = modules()
    from hindsight_incremental_runtime import ArrivalRuntime
    ledger = ld.IncrementalLedger(tmp_path / 'state', 'bank-a', create=True)
    path, checksum = delivery._freeze(ledger.root, [], 'seed')
    seed = {'bank': ledger.bank, 'artifact_path': path, 'artifact_hash': checksum,
            'provenance': {'bank': ledger.bank, 'seed_sha256': src.digest([])}}
    ledger.initialize_legacy([snapshot(src, 1)], seed, [], confirm='initialize-hindsight-incremental')
    config = {'bank': ledger.bank, 'api_url': 'http://unproved.invalid', 'hermes_home': str(tmp_path / 'home')}
    runtime = ArrivalRuntime(config)
    def forbidden(*_a, **_k):
        raise AssertionError('no-op attempted backend/network work')
    monkeypatch.setattr(isolation_module(), 'local_provenance', forbidden)
    runtime.native.transport = forbidden
    out = flow.run_incremental(config, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == 'no_op' and out['calls'] == 0
    assert runtime.llm_calls == runtime.native_calls == 0


def test_network_namespace_is_real_same_process_and_unknown_rejected(tmp_path):
    mod = isolation_module()
    mod._network_namespace(Path('/proc/self'))
    with pytest.raises(Exception, match='native_network_namespace_unproved'):
        mod._network_namespace(tmp_path / 'absent-process')


def test_different_namespace_inode_cannot_attest_host_loopback(tmp_path):
    mod = isolation_module()
    fake_process = tmp_path / 'different-process'
    (fake_process / 'ns').mkdir(parents=True)
    (fake_process / 'ns/net').write_text('distinct namespace fixture inode')
    with pytest.raises(Exception, match='native_network_namespace_unproved'):
        mod._network_namespace(fake_process)


def test_wire_binding_matches_installed_native_http_function_without_worker_io(monkeypatch):
    """Execute the real 0.9 handler's conversion; no lifespan/worker/provider."""
    import asyncio
    import socket
    from types import SimpleNamespace
    http = pytest.importorskip('hindsight_api.api.http')
    from hindsight_api.models import RequestContext
    mod = isolation_module()
    def no_network(*_a, **_k):
        raise AssertionError('native conversion attempted network')
    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(http, 'load_extension', lambda *_a, **_k: None)
    captures = []
    async def submit(bank, contents, **kwargs):
        captures.append(contents)
        return {'operation_id': kwargs['operation_id'], 'items_count': len(contents)}
    app = http.create_app(SimpleNamespace(submit_async_retain=submit, audit_logger=None), initialize_memory=False)
    endpoint = next(r.endpoint for r in app.routes if getattr(r, 'path', None) == '/v1/default/banks/{bank_id}/memories' and 'POST' in getattr(r, 'methods', set()))
    handler = endpoint.__wrapped__  # bypass optional audit logger, not conversion
    tag = ['arrival-batch:' + 'a' * 64]
    for timestamp in ('2026-10-11T00:00:00Z', 'unset'):
        item = {'content': 'new-only fixture', 'document_id': 'new-only-document',
                'timestamp': timestamp, 'tags': tag, 'observation_scopes': [tag],
                'metadata': {'kind': 'source'}, 'context': 'new',
                'entities': [{'text': 'Fixture'}]}
        request = http.RetainRequest.model_validate({'items': [item], 'async': True,
                                                   'operation_id': batch()['operation_id']})
        asyncio.run(handler('isolated-fixture-bank', request, RequestContext(), None))
        native = json.loads(json.dumps(captures[-1], default=lambda value: value.isoformat()))
        assert mod._contents([item]) == native
