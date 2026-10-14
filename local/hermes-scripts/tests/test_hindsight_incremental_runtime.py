"""Concrete REST and lease contracts, never a production endpoint."""
import copy
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def runtime_module():
    assert (ROOT / 'hindsight_incremental_runtime.py').exists(), 'concrete ArrivalRuntime is missing'
    return importlib.import_module('hindsight_incremental_runtime')


@pytest.fixture(autouse=True)
def isolated_native_metadata(monkeypatch):
    """These tests own REST/lease flow, not production process attestation.

    The sibling isolation suite exercises the real gate with adversarial SQL
    metadata and the readonly deployed-source audit tests real provenance.
    """
    class Isolation:
        def __init__(self, runtime):
            self.runtime = runtime
        def attest(self):
            return {'pid': 123, 'boot_id': 'fixture', 'start_ticks': '42', 'source_sha256': 'fixture'}
        def verify(self, current, *, expected_ids=None, items=None):
            return {'backend': self.attest(), 'bank': self.runtime.bank,
                    'scope': self.runtime._scope(current), 'batch_id': current['batch_id'],
                    'source_ids': sorted(expected_ids if expected_ids is not None else ['fact-a', 'fact-b'])}
    monkeypatch.setattr(runtime_module().ArrivalRuntime, '_isolation', lambda runtime: Isolation(runtime))


def batch(phase='retained'):
    from hindsight_incremental_delivery import _digest
    bid = _digest(b'new immutable arrival')
    return {'bank': 'bank-A', 'batch_id': bid, 'document_id': f'hindsight-arrival::bank-A::{bid}',
            'operation_id': '11111111-1111-4111-8111-111111111111', 'phase': phase,
            'receipts': {'retained': {'operation_id': '11111111-1111-4111-8111-111111111111',
                         'bank': 'bank-A', 'status': 'completed', 'children': []}}}


class Ledger:
    def __init__(self, root):
        self.bank, self.root = 'bank-A', root
        root.mkdir(mode=0o700)
        self.batches = [batch()]

    def locked(self):
        from contextlib import nullcontext
        return nullcontext(self)

    def list_batches(self, **kwargs):
        return copy.deepcopy(self.batches)

    def get_batch(self, bid):
        return copy.deepcopy(next(b for b in self.batches if b['batch_id'] == bid))


class REST:
    def __init__(self):
        self.flags = {'enable_observations': False, 'enable_auto_consolidation': True}
        self.calls = []
        self.legacy_pending = 56
        self.new_pending = 2
        self.new_failed = 0
        self.operation_status = 'completed'
        self.disable_on_trigger = False
        self.scope = [f"arrival-batch:{batch()['batch_id']}"]
        self.operation_id = '22222222-2222-4222-8222-222222222222'
        self.operations = {}
        self.facts = [{'id': 'fact-a', 'document_id': batch()['document_id'], 'text': 'New measured fact A',
                       'type': 'world', 'date': '1990-01-01', 'score': 99},
                      {'id': 'fact-b', 'document_id': batch()['document_id'], 'text': 'New measured fact B',
                       'type': 'experience', 'score': 23}]
        self.document = {'id': batch()['document_id'], 'bank_id': 'bank-A',
                         'created_at': '2026-10-11T00:00:00+00:00', 'memory_unit_count': 2,
                         'nodes_by_fact_type': {'world': 1, 'experience': 1}}
        self.on_patch = None
        self.override_total = None
        self.active = []

    def __call__(self, method, path, *, payload=None, params=None):
        params = params or {}
        self.calls.append((method, path, copy.deepcopy(payload), copy.deepcopy(params)))
        if path.endswith('/config'):
            if method == 'PATCH':
                if self.on_patch:
                    self.on_patch()
                self.flags.update(payload['updates'])
            return {'bank_id': 'bank-A', 'config': copy.deepcopy(self.flags)}
        if '/documents/' in path:
            return copy.deepcopy(self.document)
        if path.endswith('/memories/list'):
            if 'consolidation_state' in params:
                scoped = params.get('tags') == self.scope and params.get('tags_match') == 'all_strict'
                count = self.new_failed if params['consolidation_state'] == 'failed' else self.new_pending
                if not scoped and params['consolidation_state'] == 'pending':
                    count += self.legacy_pending
                return {'items': [], 'total': count, 'limit': 0, 'offset': 0}
            assert params.get('document_id') == self.document['id'], 'unbounded/history fact scan'
            records = [f for f in self.facts if f['type'] == params['type']]
            total = len(records) if self.override_total is None else self.override_total
            offset, limit = int(params['offset']), int(params['limit'])
            wire = [{k: v for k, v in r.items() if k != 'type'} | {'fact_type': r['type']}
                    for r in records[offset:offset + limit]]
            return {'items': copy.deepcopy(wire), 'total': total,
                    'limit': limit, 'offset': offset}
        if path.endswith('/operations'):
            items = [op for op in self.active if op['status'] == params.get('status')]
            return {'operations': copy.deepcopy(items), 'total': len(items)}
        if path.endswith('/consolidate'):
            assert self.flags == {'enable_observations': True, 'enable_auto_consolidation': False}
            assert payload == {'observation_scopes': [self.scope]}, 'shared/history scope reached'
            op = {'operation_id': self.operation_id, 'bank_id': 'bank-A', 'status': self.operation_status,
                  'child_operations': None, 'task_payload': {'observation_scopes': [self.scope]}}
            self.operations[self.operation_id] = op
            if self.operation_status == 'completed':
                self.new_pending = 0
            if self.disable_on_trigger:
                self.flags['enable_observations'] = False
            return {'operation_id': self.operation_id, 'deduplicated': False}
        if '/operations/' in path:
            oid = path.split('/')[-1]
            return copy.deepcopy(self.operations.get(oid, {'operation_id': oid, 'status': 'completed',
                                                           'bank_id': 'bank-A', 'child_operations': None}))
        if path.endswith('/memories') and method == 'POST':
            return {'operation_id': payload['operation_id']}
        raise AssertionError((method, path, payload, params))


def make_runtime(tmp_path, rest=None):
    mod = runtime_module()
    from hindsight_native_client import HindsightNativeClient
    rest = rest or REST()
    ledger = Ledger(tmp_path / 'private-bank')
    runtime = mod.ArrivalRuntime({'bank': ledger.bank, 'api_url': 'http://isolated.invalid',
                                 'incremental': {'retry': {'max_calls': 100, 'base_delay': 0, 'poll_s': 0}}})
    runtime.native = HindsightNativeClient(api='http://isolated.invalid', bank=ledger.bank, transport=rest)
    runtime.ledger = ledger
    return runtime, rest, ledger


def policy(**kwargs):
    from hindsight_offline_retry import RetryPolicy
    return RetryPolicy(max_calls=100, base_delay=0, max_delay=0, poll_s=0, **kwargs)


def test_private_scope_window_is_durable_and_never_drains_legacy(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    rest.on_patch = lambda: assert_lease_exists(ledger)
    window = runtime.open_window(ledger)
    proof = runtime.consolidate_batch(batch(), policy())
    assert proof['ready'] is True and proof['pending_in_scope'] == 0
    assert proof['scope'] == rest.scope and proof['legacy_pending'] == 56
    assert runtime.cleanup_window(ledger, window) == 'restored'
    assert rest.legacy_pending == 56
    patches = [c[2]['updates'] for c in rest.calls if c[0] == 'PATCH']
    assert patches == [{'enable_auto_consolidation': False, 'enable_observations': True},
                       {'enable_auto_consolidation': True, 'enable_observations': False}]
    scoped = [c[3] for c in rest.calls if c[1].endswith('/memories/list') and c[3].get('tags')]
    assert scoped and all(p['tags_match'] == 'all_strict' for p in scoped)


def assert_lease_exists(ledger):
    path = ledger.root / 'runtime-window.json'
    lease = json.loads(path.read_text())
    assert lease['bank'] == ledger.bank and lease['owner']['pid'] == os.getpid()
    assert 'before_config' in lease and 'allowed_scopes' in lease
    assert path.stat().st_mode & 0o077 == 0


def test_zero_new_pending_has_zero_trigger(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    rest.new_pending = 0
    window = runtime.open_window(ledger)
    proof = runtime.consolidate_batch(batch(), policy())
    assert proof['ready'] and proof['legacy_pending'] == 56
    assert not any(c[0] == 'POST' for c in rest.calls)
    assert runtime.cleanup_window(ledger, window) == 'restored'


@pytest.mark.parametrize('failure', ['failed', 'disabled', 'pending', 'failed_facts'])
def test_failed_disabled_or_pending_consolidation_cannot_be_ready(tmp_path, failure):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    if failure == 'disabled':
        rest.disable_on_trigger = True
    elif failure == 'failed_facts':
        rest.new_failed = 1
    else:
        rest.operation_status = failure
    with pytest.raises(Exception):
        runtime.consolidate_batch(batch(), policy())
    if failure == 'pending':
        assert runtime.cleanup_window(ledger, runtime._window) == 'pending'
        assert rest.flags['enable_observations'] is True
        assert rest.flags['enable_auto_consolidation'] is False
    assert rest.legacy_pending == 56


def test_live_owner_and_unowned_leaked_window_are_rejected(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    other = runtime_module().ArrivalRuntime(runtime.config)
    other.native, other.ledger = runtime.native, ledger
    with pytest.raises(Exception, match='owner'):
        other.open_window(ledger)
    assert len([c for c in rest.calls if c[0] == 'PATCH']) == 1


def test_leaked_observations_without_lease_are_not_claimed(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    rest.flags['enable_observations'] = True
    with pytest.raises(Exception, match='window'):
        runtime.open_window(ledger)
    assert not any(c[0] == 'PATCH' for c in rest.calls)


def test_crash_lease_recovers_only_exact_owned_scope(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    runtime.consolidate_batch(batch(), policy())
    path = ledger.root / 'runtime-window.json'
    lease = json.loads(path.read_text())
    lease['owner']['pid'] = 99999999
    path.write_text(json.dumps(lease))
    recovered = runtime_module().ArrivalRuntime(runtime.config)
    recovered.native, recovered.ledger = runtime.native, ledger
    window = recovered.open_window(ledger)
    assert runtime_module().ArrivalRuntime(runtime.config).bank == ledger.bank
    assert window['allowed_scopes'] == [rest.scope]
    assert recovered.cleanup_window(ledger, window) == 'restored'
    assert rest.legacy_pending == 56


def test_exact_document_facts_are_stable_and_timestamp_is_arrival_not_event_date(tmp_path):
    runtime, rest, _ = make_runtime(tmp_path)
    facts = runtime.facts_for_batch(batch())
    assert [f['id'] for f in facts] == ['fact-a', 'fact-b']
    assert all(set(f) == {'id', 'text', 'type', 'document_id', 'created_at'} for f in facts)
    assert all(f['created_at'] == rest.document['created_at'] for f in facts)
    assert not any(c[3].get('date') or c[3].get('start_date') for c in rest.calls)


@pytest.mark.parametrize('defect', ['count', 'document', 'page_total', 'duplicate', 'unconfirmed'])
def test_unknown_or_mismatched_fact_coverage_fails_closed(tmp_path, defect):
    runtime, rest, _ = make_runtime(tmp_path)
    current = batch()
    if defect == 'count':
        rest.document['memory_unit_count'] = 3
    elif defect == 'document':
        rest.document['id'] = 'old-document'
    elif defect == 'page_total':
        rest.override_total = 2
    elif defect == 'duplicate':
        rest.facts[1]['id'] = rest.facts[0]['id']
    else:
        current['phase'] = 'submitted'
    with pytest.raises(Exception):
        runtime.facts_for_batch(current)


def frozen_input():
    source = REST()
    facts = [{k: f[k] for k in ('id', 'text', 'type', 'document_id')} | {'created_at': source.document['created_at']}
             for f in source.facts]
    return {'bank': 'bank-A', 'kind': 'daily_summary', 'parent_batch_ids': [batch()['batch_id']],
            'messages': [{'origin': f['id'], 'timestamp': f['created_at'], 'content': json.dumps(f)} for f in facts],
            'items': [{'document_id': batch()['document_id'], 'content': 'frozen new facts', 'metadata': {}}]}


def observation():
    return {'id': 'obs-original-id', 'topic': 'measurements', 'insight': 'Evidence A supports safe bounds',
            'type': 'technical_lesson', 'applicability': 'new apparatus only', 'evidence_ids': ['fact-a'],
            'supersedes': [], 'confidence': 'high', 'valid_from': '2026-10-11', 'valid_until': None}


def test_summary_uses_only_frozen_facts_and_existing_configured_llm(tmp_path, monkeypatch):
    runtime, _, _ = make_runtime(tmp_path)
    mod = runtime_module()
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    supplied = {'model': 'configured-OneAPI', 'base_url': 'http://loopback-only/v1',
                'label': 'exact-label', 'api_key_env': 'FAKE_LLM_KEY', 'response_format': True}
    monkeypatch.setattr(profiles, 'get_llm_profile', lambda name: dict(supplied))
    seen = []
    def llm(unit, **kwargs):
        kwargs['retry_policy'].charge()
        seen.append((unit, kwargs))
        return {'canonical_observations': [observation()]}, 'private raw JSON'
    monkeypatch.setattr(reflect, 'call_llm', llm)
    out = runtime.generate_summary(frozen_input(), retry_policy=policy())
    assert out['items'][0]['document_id'] == frozen_input()['items'][0]['document_id']
    assert runtime.validate_summary(out['artifact']) is True
    row = out['artifact']['canonical_observations'][0]
    assert row['id'] == observation()['id'] and row['evidence_ids'] == ['fact-a']
    assert row['bank'] == 'bank-A' and row['source_documents'] == [batch()['document_id']]
    unit, kwargs = seen[0]
    assert unit.source_ids == ['fact-a', 'fact-b']
    assert 'New measured fact A' in unit.content and 'New measured fact B' in unit.content
    assert kwargs['model'] == supplied['model'] and kwargs['base_url'] == supplied['base_url']
    assert kwargs['emit_observations'] is True and runtime.llm_calls == 1


@pytest.mark.parametrize('defect', ['empty', 'bad_json', 'missing_typed', 'forged_evidence', 'over_budget'])
def test_invalid_summary_is_rejected_before_retain(tmp_path, monkeypatch, defect):
    runtime, rest, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    monkeypatch.setattr(profiles, 'get_llm_profile', lambda name: {'model': 'M', 'base_url': 'http://fake/v1',
                                                                 'label': 'L', 'api_key_env': 'FAKE'})
    obj = {'canonical_observations': [observation()]}
    frozen = frozen_input()
    if defect == 'empty':
        obj['canonical_observations'] = []
    elif defect == 'bad_json':
        obj = None
    elif defect == 'missing_typed':
        obj['canonical_observations'][0].pop('applicability')
    elif defect == 'forged_evidence':
        obj['canonical_observations'][0]['evidence_ids'] = ['sealed-history-fact']
    else:
        frozen['messages'][0]['content'] += 'x' * 60001
    monkeypatch.setattr(reflect, 'call_llm', lambda *a, **k: (obj, 'raw'))
    with pytest.raises(Exception):
        runtime.generate_summary(frozen, retry_policy=policy())
    assert not any(c[0] == 'POST' for c in rest.calls)


def test_publication_methods_delegate_with_ledger_and_scoped_proof(tmp_path, monkeypatch):
    runtime, _, ledger = make_runtime(tmp_path)
    calls = []
    class Publication:
        def __init__(self, config, bank):
            assert config is runtime.config and bank == ledger.bank
        def gate(self, manifest, *, ledger, scoped_proof):
            calls.append(('gate', manifest, ledger, scoped_proof)); return {'decision': 'real-adapter'}
        def publish_generation(self, manifest, gate, *, ledger):
            calls.append(('publish', manifest, gate, ledger)); return {'published': False}
        def readback_generation(self, manifest, *, ledger):
            calls.append(('readback', manifest, ledger)); return {'readback_verified': False}
        def export_seed(self, path):
            calls.append(('export', path)); return {'path': str(path)}
    import types
    monkeypatch.setitem(sys.modules, 'hindsight_incremental_publication', types.SimpleNamespace(PublicationAdapter=Publication))
    manifest = {'bank': ledger.bank, 'consumed_batch_ids': [batch()['batch_id']]}
    ledger.batches[0]['receipts']['consolidated'] = {'scoped_proof': {'bank': ledger.bank,
        'scope': [f"arrival-batch:{batch()['batch_id']}"], 'pending_in_scope': 0, 'ready': True}}
    assert runtime.gate(manifest)['decision'] == 'real-adapter'
    assert runtime.publish_generation(manifest, {'gate': 'bound'})['published'] is False
    assert runtime.readback_generation(manifest)['readback_verified'] is False
    assert runtime.export_seed(tmp_path / 'seed.json')['path'].endswith('seed.json')
    assert calls[0][2] is ledger and calls[0][3]['ready'] is True
    assert [c[0] for c in calls] == ['gate', 'publish', 'readback', 'export']


def test_real_loopback_rest_wire_contract(tmp_path):
    rest = REST()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def handle_request(self):
            parts = urlsplit(self.path)
            params = {k: (v if k == 'tags' else v[0]) for k, v in parse_qs(parts.query).items()}
            for key in ('limit', 'offset'):
                if key in params: params[key] = int(params[key])
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length)) if length else None
            response = rest(self.command, parts.path, payload=payload, params=params)
            data = json.dumps(response).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)
        do_GET = do_PATCH = do_POST = handle_request
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        runtime, _, ledger = make_runtime(tmp_path, rest)
        from hindsight_native_client import HindsightNativeClient
        runtime.native = HindsightNativeClient(api=f'http://127.0.0.1:{server.server_port}', bank=ledger.bank)
        window = runtime.open_window(ledger)
        assert len(runtime.facts_for_batch(batch())) == 2
        assert runtime.consolidate_batch(batch(), policy())['ready']
        assert runtime.cleanup_window(ledger, window) == 'restored'
        assert rest.legacy_pending == 56
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


def test_poll_budget_counts_gets_and_stops_at_actual_call_cap(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    rest.operation_status = 'pending'
    before = len(rest.calls)
    from hindsight_offline_retry import RetryDeferred, RetryPolicy
    bounded = RetryPolicy(max_calls=40, base_delay=0, max_delay=0, poll_s=0)
    with pytest.raises(RetryDeferred) as caught:
        runtime.consolidate_batch(batch(), bounded)
    polls = [c for c in rest.calls[before:] if '/operations/' in c[1]]
    assert len(polls) > 4 and bounded.calls_used == bounded.max_calls
    assert caught.value.error_class in {'operation_pending', 'call_cap'}
    assert sum(c[0] == 'POST' for c in rest.calls) == 1


def test_successful_fifth_native_status_poll_is_not_a_transport_failure(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    rest.operation_status = 'pending'
    original = rest.__call__
    polls = []
    def eventual(method, path, **kwargs):
        if '/operations/' in path:
            polls.append(path)
            if len(polls) >= 5:
                rest.operations[rest.operation_id]['status'] = 'completed'
                rest.new_pending = 0
        return original(method, path, **kwargs)
    runtime.native.transport = eventual
    assert runtime.consolidate_batch(batch(), policy())['ready'] is True
    assert len(polls) == 5
    assert sum(c[0] == 'POST' for c in rest.calls) == 1


def test_native_continuation_blocks_zero_fact_cleanup(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    window = runtime.open_window(ledger)
    runtime.consolidate_batch(batch(), policy())
    rest.active = [{'operation_id': 'native-continuation', 'status': 'processing', 'operation_type': 'consolidate'}]
    assert rest.new_pending == 0
    assert runtime.cleanup_window(ledger, window) == 'pending'
    assert rest.flags == {'enable_observations': True, 'enable_auto_consolidation': False}


def test_scope_dedup_to_shared_operation_never_produces_ready(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    original = rest.__call__
    def shared(method, path, **kwargs):
        value = original(method, path, **kwargs)
        if '/operations/' in path:
            value['task_payload'] = {'chosen_scope': []}
        return value
    runtime.native.transport = shared
    with pytest.raises(Exception, match='scope'):
        runtime.consolidate_batch(batch(), policy())
    assert runtime.cleanup_window(ledger, runtime._window) == 'pending'


def test_unknown_trigger_ack_cannot_be_blindly_reposted_or_closed(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    window = runtime.open_window(ledger)
    original = rest.__call__
    def lost(method, path, **kwargs):
        value = original(method, path, **kwargs)
        if path.endswith('/consolidate'):
            raise TimeoutError('lost acknowledgement')
        return value
    runtime.native.transport = lost
    with pytest.raises(TimeoutError):
        runtime.consolidate_batch(batch(), policy())
    with pytest.raises(Exception, match='unknown'):
        runtime.consolidate_batch(batch(), policy())
    assert sum(c[0] == 'POST' for c in rest.calls) == 1
    assert runtime.cleanup_window(ledger, window) == 'pending'


def test_failed_open_readback_keeps_durable_cleanup_evidence(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    original = rest.__call__
    patches = []
    def lost(method, path, **kwargs):
        value = original(method, path, **kwargs)
        if method == 'PATCH':
            patches.append(value)
        elif path.endswith('/config') and patches:
            raise TimeoutError('bank config readback unavailable')
        return value
    runtime.native.transport = lost
    with pytest.raises(Exception):
        runtime.open_window(ledger)
    lease = json.loads((ledger.root / 'runtime-window.json').read_text())
    assert lease['status'] in {'opening', 'cleanup_pending'}
    assert lease['before_config']['enable_observations'] is False
    assert rest.flags['enable_auto_consolidation'] is False
    runtime.native.transport = original
    assert runtime.cleanup_window(ledger, runtime._window) == 'restored'


def test_delivery_proxy_keeps_exact_uuid_payload_and_admits_scope_before_post(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    item = {'document_id': batch()['document_id'], 'content': 'private body',
            'tags': rest.scope, 'observation_scopes': [rest.scope],
            'metadata': {'private-key': 'string'}, 'timestamp': '2026-10-11T00:00:00Z'}
    original = copy.deepcopy(item)
    result = runtime.retain_items([item], operation_id=batch()['operation_id'], timeout=0.5)
    assert result['operation_id'] == batch()['operation_id'] and item == original
    lease = json.loads((ledger.root / 'runtime-window.json').read_text())
    assert lease['allowed_scopes'] == [rest.scope]
    assert lease['scopes'][rest.scope[0]]['retain_operation_id'] == batch()['operation_id']
    with pytest.raises(Exception, match='scope'):
        runtime.retain_items([{**item, 'observation_scopes': []}], operation_id=batch()['operation_id'])


def test_real_engine_runtime_reflect_and_publication_callbacks_are_incremental(tmp_path, monkeypatch):
    """Real adapters and wire HTTP; only native workers/SQL/eval/embedding are Fake."""
    import hindsight_incremental_ledger as ld
    import hindsight_incremental_sources as sources
    import hindsight_incremental_merge as merge
    import hindsight_incremental_pipeline as flow
    from hindsight_incremental_publication import PublicationAdapter
    # Reuse the publisher contract's SQL double rather than invent a second SQL
    # protocol. This still exercises the real publication transaction/readback.
    sys.path.insert(0, str(ROOT / 'tests'))
    from test_hindsight_incremental_publication import Database, fake_run
    db = Database()
    inherited = copy.deepcopy(db.data)
    calls, flags, documents, facts, operations = [], dict(OPEN_FLAGS), {}, {}, {}
    pending = set()
    chat_attempts = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
        def request(self):
            parts = urlsplit(self.path)
            params = {k: v for k, v in parse_qs(parts.query).items()}
            length = int(self.headers.get('Content-Length', 0))
            payload = json.loads(self.rfile.read(length)) if length else None
            calls.append((self.command, parts.path, copy.deepcopy(payload), params))
            status = 200
            if parts.path == '/v1/chat/completions':
                chat_attempts.append(payload)
                assert 'New arrival native fact' in payload['messages'][1]['content']
                assert 'sealed old conversation' not in payload['messages'][1]['content']
                if len(chat_attempts) < 3:
                    status = 429 if len(chat_attempts) == 1 else 502
                    response = {'error': 'isolated transient'}
                else:
                    row = observation(); row['evidence_ids'] = ['raw-fact-original-id']
                    response = {'choices': [{'message': {'content': json.dumps({'canonical_observations': [row]})}}]}
            elif parts.path.endswith('/config'):
                if self.command == 'PATCH':
                    flags.update(payload['updates'])
                response = {'bank_id': 'bank-a', 'config': dict(flags)}
            elif parts.path.endswith('/operations'):
                response = {'operations': [], 'total': 0}
            elif '/operations/' in parts.path:
                oid = parts.path.rsplit('/', 1)[1]
                response = operations.get(oid, {'operation_id': oid, 'status': 'not_found', 'bank_id': 'bank-a'})
            elif parts.path.endswith('/memories'):
                assert flags == {'enable_auto_consolidation': False, 'enable_observations': True}
                oid = payload['operation_id']
                item = payload['items'][0]
                doc = item['document_id']
                tag = 'arrival-batch:' + doc.split('::')[-1]
                assert item['tags'] == [tag] and item['observation_scopes'] == [[tag]]
                assert all(isinstance(v, str) for v in item['metadata'].values())
                pending.add(tag)
                created = '2026-10-11T00:00:00+00:00'
                documents[doc] = {'id': doc, 'bank_id': 'bank-a', 'created_at': created,
                                  'memory_unit_count': 1, 'nodes_by_fact_type': {'world': 1}}
                fid = 'summary-fact' if item['metadata']['kind'] == 'daily_summary' else 'raw-fact-original-id'
                facts[doc] = [{'id': fid, 'text': 'New arrival native fact', 'type': 'world',
                               'document_id': doc, 'created_at': created}]
                operations[oid] = {'operation_id': oid, 'bank_id': 'bank-a', 'status': 'completed', 'child_operations': None}
                response = {'operation_id': oid}
            elif '/documents/' in parts.path:
                from urllib.parse import unquote
                response = documents[unquote(parts.path.rsplit('/', 1)[1])]
            elif parts.path.endswith('/memories/list'):
                if 'consolidation_state' in params:
                    tags = params.get('tags')
                    if tags:
                        assert params['tags_match'] == ['all_strict']
                    count = 0 if params['consolidation_state'] == ['failed'] else (
                        sum(t in pending for t in tags) if tags else 56 + len(pending))
                    response = {'items': [], 'total': count, 'limit': 0, 'offset': 0}
                else:
                    doc = params['document_id'][0]
                    assert doc in documents
                    rows = [f for f in facts[doc] if f['type'] == params['type'][0]]
                    wire = [{k: v for k, v in r.items() if k != 'type'} | {'fact_type': r['type']}
                            for r in rows]
                    response = {'items': wire, 'total': len(rows), 'limit': 100, 'offset': 0}
            elif parts.path.endswith('/consolidate'):
                scopes = payload['observation_scopes']
                assert len(scopes) == 1 and len(scopes[0]) == 1 and scopes[0][0].startswith('arrival-batch:')
                pending.discard(scopes[0][0])
                oid = str(__import__('uuid').uuid4())
                operations[oid] = {'operation_id': oid, 'bank_id': 'bank-a', 'status': 'completed',
                                   'child_operations': None, 'task_payload': {'chosen_scope': scopes[0]}}
                response = {'operation_id': oid}
            else:
                raise AssertionError(parts.path)
            data = json.dumps(response).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Retry-After', '0'); self.send_header('Content-Length', str(len(data)))
            self.end_headers(); self.wfile.write(data)
        do_GET = do_POST = do_PATCH = request
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        api = f'http://127.0.0.1:{server.server_port}'
        for key, value in {'HINDSIGHT_OFFLINE_LLM_MODEL': 'same-OneAPI-model',
                           'HINDSIGHT_OFFLINE_LLM_BASE_URL': api + '/v1',
                           'HINDSIGHT_OFFLINE_LLM_API_KEY_ENV': 'LOOPBACK_TEST_KEY',
                           'LOOPBACK_TEST_KEY': 'isolated-fixture-key'}.items():
            monkeypatch.setenv(key, value)
        config = {'bank': 'bank-a', 'api_url': api, 'api': api, 'hermes_home': str(tmp_path),
                  'llm_profile': 'custom', 'dsn': 'scratch-only', 'script_dir': str(ROOT),
                  'generic_benchmark': str(tmp_path / 'generic.jsonl'),
                  'local_benchmark': str(tmp_path / 'local.jsonl'),
                  'incremental': {'root': str(tmp_path / 'state'),
                                  'retry': {'max_calls': 100, 'max_llm_calls': None,
                                            'poll_s': 0, 'base_delay': 0, 'max_delay': 0}}}
        for label in ('generic', 'local'):
            (tmp_path / f'{label}.jsonl').write_text('{"id":"case"}\n')
        adapter = PublicationAdapter(config, 'bank-a', connect=db.connect,
                                     embeddings=db.encode, execute_values=db.values, run_json=fake_run)
        export = adapter.export_seed(tmp_path / 'seed-export.json')
        def snapshot(count):
            rows = []
            for i in range(count):
                row = {'role': 'user', 'timestamp': '1990-01-01T00:00:00Z', 'origin': f'm-{i}',
                       'content': 'sealed old conversation' if i == 0 else 'new arrival only'}
                row['event_id'] = sources.digest(row); rows.append(row)
            value = {'source_key': 'default:session', 'source_epoch': 'epoch-A', 'profile': 'default',
                     'messages': rows, 'source_hash_before': sources.digest(rows), 'source_hash_after': sources.digest(rows)}
            value['snapshot_hash'] = sources.snapshot_hash(value)
            return value
        flow.initialize(config, seed_file=tmp_path / 'seed-export.json', confirm='initialize-hindsight-incremental',
                        source_reader=lambda *_a, **_k: [snapshot(1)])
        ledger = ld.IncrementalLedger(tmp_path / 'state', 'bank-a')
        runtime = runtime_module().ArrivalRuntime(config); runtime.ledger = ledger
        runtime._publication_adapter = adapter
        first = flow.run_incremental(config, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                     source_reader=lambda *_a, **_k: [snapshot(2)], execute=True)
        assert first['status'] == 'completed', first
        assert first['cleanup'] == 'restored' and first['published'] is True
        assert first['llm_calls'] == 3 and len(chat_attempts) == 3
        assert runtime.native_calls == len([c for c in calls if c[1] != '/v1/chat/completions'])
        assert db.commits == 1 and len(db.embeds) == 1  # Fake embedding/SQL, not production telemetry.
        assert db.data['documents'][0] == inherited['documents'][0]
        assert db.data['units'][0] == inherited['units'][0]
        assert flags == OPEN_FLAGS and not pending
        before = len(calls), db.commits, len(db.embeds)
        second = flow.run_incremental(config, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                      source_reader=lambda *_a, **_k: [snapshot(2)], execute=True)
        assert second['status'] == 'no_op' and second['calls'] == 0
        assert (len(calls), db.commits, len(db.embeds)) == before
        assert all(b['phase'] == 'published' for b in ledger.list_batches())
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)


OPEN_FLAGS = {'enable_observations': False, 'enable_auto_consolidation': True}


def test_unknown_ack_recovers_only_active_job_with_exact_private_scope(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    rest.operation_status = 'pending'
    original = rest.__call__
    def lost(method, path, **kwargs):
        value = original(method, path, **kwargs)
        if path.endswith('/consolidate'):
            raise TimeoutError('lost acknowledgement')
        return value
    runtime.native.transport = lost
    with pytest.raises(TimeoutError):
        runtime.consolidate_batch(batch(), policy())
    rest.active = [copy.deepcopy(rest.operations[rest.operation_id])]
    from hindsight_offline_retry import RetryDeferred
    with pytest.raises(RetryDeferred):
        runtime.consolidate_batch(batch(), policy())
    lease = json.loads((ledger.root / 'runtime-window.json').read_text())
    state = lease['scopes'][rest.scope[0]]
    assert state['consolidation_operation_ids'] == [rest.operation_id]
    assert state['trigger_unknown'] is False
    rest.operations[rest.operation_id]['status'] = 'completed'
    rest.new_pending = 0; rest.active = []
    assert runtime.consolidate_batch(batch(), policy())['ready']
    assert sum(c[0] == 'POST' for c in rest.calls) == 1


def test_begin_run_shares_native_call_cap_and_resets_observed_counters(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    from hindsight_offline_retry import RetryDeferred
    runtime.llm_calls = 7; runtime.native_calls = 9
    run_policy = policy()
    runtime.begin_run(run_policy)
    assert runtime.llm_calls == runtime.native_calls == 0
    runtime.open_window(ledger)
    assert run_policy.calls_used == len(rest.calls) == runtime.native_calls
    run_policy.calls_used = run_policy.max_calls
    with pytest.raises(RetryDeferred):
        runtime.facts_for_batch(batch())


def test_pending_lease_can_be_recovered_without_admitting_new_producer(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    runtime.consolidate_batch(batch(), policy())
    path = ledger.root / 'runtime-window.json'
    lease = json.loads(path.read_text()); lease['owner']['pid'] = 99999999
    path.write_text(json.dumps(lease))
    recovered = runtime_module().ArrivalRuntime(runtime.config)
    recovered.native = runtime.native
    before = len([c for c in rest.calls if c[0] == 'PATCH'])
    window = recovered.resume_window(ledger)
    assert window is not None and len([c for c in rest.calls if c[0] == 'PATCH']) == before
    assert recovered.cleanup_window(ledger, window) == 'restored'
    assert recovered.resume_window(ledger) is None


def test_disabled_completed_native_operation_is_not_a_success(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    original = rest.__call__
    def skipped(method, path, **kwargs):
        result = original(method, path, **kwargs)
        if '/operations/' in path:
            result['result_metadata'] = {'skipped': True}
        return result
    runtime.native.transport = skipped
    with pytest.raises(Exception, match='disabled'):
        runtime.consolidate_batch(batch(), policy())
    assert runtime.cleanup_window(ledger, runtime._window) == 'pending'


def test_existing_live_native_work_is_not_drained_or_claimed(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    rest.active = [{'id': 'inherited-operation', 'task_type': 'consolidate', 'status': 'processing'}]
    with pytest.raises(Exception, match='unowned_active_native_work'):
        runtime.open_window(ledger)
    assert not any(c[0] in {'POST', 'PATCH'} for c in rest.calls)


def test_conflicting_native_scope_fields_fail_closed(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    original = rest.__call__
    def conflict(method, path, **kwargs):
        result = original(method, path, **kwargs)
        if '/operations/' in path:
            result['task_payload'] = {'observation_scopes': [[]], 'chosen_scope': rest.scope}
        return result
    runtime.native.transport = conflict
    with pytest.raises(Exception, match='scope'):
        runtime.consolidate_batch(batch(), policy())


def test_disabled_native_child_blocks_completed_parent(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    runtime.open_window(ledger)
    original = rest.__call__
    def disabled_child(method, path, **kwargs):
        result = original(method, path, **kwargs)
        if '/operations/' in path:
            result['child_operations'] = [{'operation_id': 'child-operation', 'status': 'completed',
                                           'result_metadata': {'disabled': True}}]
        return result
    runtime.native.transport = disabled_child
    with pytest.raises(Exception, match='disabled'):
        runtime.consolidate_batch(batch(), policy())


@pytest.mark.parametrize('boundary', ['profile', 'llm_key'])
def test_llm_configuration_exit_is_reportable_without_provider_fallback(tmp_path, monkeypatch, boundary):
    runtime, rest, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    def unavailable(*_args, **_kwargs):
        raise SystemExit('isolated configuration missing')
    if boundary == 'profile':
        monkeypatch.setattr(profiles, 'get_llm_profile', unavailable)
    else:
        monkeypatch.setattr(profiles, 'get_llm_profile', lambda _name: {
            'model': 'same-model', 'base_url': 'http://isolated.invalid/v1', 'label': 'same', 'api_key_env': 'NO_KEY'})
        monkeypatch.setattr(reflect, 'get_llm_key', unavailable)
    with pytest.raises(Exception, match='llm_configuration'):
        runtime.generate_summary(frozen_input(), retry_policy=policy())
    assert runtime.llm_calls == 0 and not any(c[0] == 'POST' for c in rest.calls)


def test_explicit_summary_overrides_precede_key_selection_without_changing_env(tmp_path, monkeypatch):
    runtime, rest, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    monkeypatch.setattr(profiles, 'read_dotenv', lambda: {})
    monkeypatch.delenv('ONEAPI_API_KEY', raising=False)
    monkeypatch.setenv('ARRIVAL_FIXTURE_KEY', 'test-only-not-a-real-key')
    monkeypatch.setenv('HINDSIGHT_OFFLINE_LLM_MODEL', 'env-model')
    runtime.config['llm_profile'] = 'oneapi-auto-cheep'
    runtime.config['incremental']['llm_override'] = {
        'model': 'explicit-model.v2', 'base_url': 'http://isolated.invalid/explicit/v1',
        'api_key_env': 'ARRIVAL_FIXTURE_KEY', 'provider': 'openai'}
    seen = []
    def llm(_unit, **kwargs):
        kwargs['retry_policy'].charge(kind='llm')
        seen.append(kwargs)
        return {'canonical_observations': [observation()]}, 'fixture'
    monkeypatch.setattr(reflect, 'call_llm', llm)
    out = runtime.generate_summary(frozen_input(), retry_policy=policy())
    assert runtime.validate_summary(out['artifact'])
    assert seen[0]['model'] == 'explicit-model.v2'
    assert seen[0]['base_url'] == 'http://isolated.invalid/explicit/v1'
    assert seen[0]['api_key_env'] == 'ARRIVAL_FIXTURE_KEY'
    assert os.environ['HINDSIGHT_OFFLINE_LLM_MODEL'] == 'env-model'
    assert not any(c[0] == 'POST' for c in rest.calls)


def test_noncompatible_explicit_provider_is_rejected_not_silently_ignored(tmp_path, monkeypatch):
    runtime, rest, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    monkeypatch.setattr(profiles, 'read_dotenv', lambda: {})
    monkeypatch.setenv('ARRIVAL_FIXTURE_KEY', 'test-only-not-a-real-key')
    runtime.config['llm_profile'] = 'oneapi-auto-cheep'
    runtime.config['incremental']['llm_override'] = {'provider': 'anthropic', 'api_key_env': 'ARRIVAL_FIXTURE_KEY'}
    generated = []
    monkeypatch.setattr(reflect, 'call_llm', lambda *_a, **_k: (generated.append(True), 'fixture'))
    with pytest.raises(Exception, match='unsupported_llm_provider'):
        runtime.generate_summary(frozen_input(), retry_policy=policy())
    assert generated == [] and runtime.llm_calls == 0
    assert not any(c[0] == 'POST' for c in rest.calls)


def test_runtime_accepts_actual_09_list_fact_type_contract(tmp_path):
    runtime, rest, ledger = make_runtime(tmp_path)
    # Installed list API returns fact_type, NOT type; created_at is absent.
    for row in rest.facts:
        row['fact_type'] = row.pop('type')
    original = rest.__call__
    def actual_list(method, path, *, payload=None, params=None):
        if path.endswith('/memories/list') and 'type' in (params or {}):
            rows = [r for r in rest.facts if r['fact_type'] == params['type']]
            return {'items': copy.deepcopy(rows), 'total': len(rows), 'limit': 100, 'offset': 0}
        return original(method, path, payload=payload, params=params)
    runtime.native.transport = actual_list
    facts = runtime.facts_for_batch(batch())
    assert {f['type'] for f in facts} == {'world', 'experience'}
    assert all(f['created_at'] == rest.document['created_at'] for f in facts)


@pytest.mark.parametrize('llm_calls', [0, 20])
def test_exhausted_request_cap_does_not_create_unknown_unattempted_trigger(tmp_path, llm_calls):
    from hindsight_offline_retry import RetryPolicy
    runtime, rest, ledger = make_runtime(tmp_path)
    policy = RetryPolicy(base_delay=0, max_delay=0, poll_s=0, max_llm_calls=None)
    runtime.begin_run(policy)
    runtime.open_window(ledger)
    runtime.facts_for_batch(batch())
    from hindsight_offline_retry import RetryBudget
    for _ in range(llm_calls):
        RetryBudget(policy).charge(kind='llm')
    # Leave capacity for only the three pre-trigger reads. This boundary can
    # occur with any finite cap, including the separately capped API policy.
    policy.calls_used = policy.llm_calls_used + policy.max_calls - 3
    from hindsight_offline_retry import RetryDeferred
    with pytest.raises(RetryDeferred, match='call_cap'):
        runtime.consolidate_batch(batch(), policy)
    assert not any(c[1].endswith('/consolidate') for c in rest.calls)
    assert policy.api_calls_used == policy.max_calls and policy.llm_calls_used == llm_calls
    lease = json.loads((ledger.root / 'runtime-window.json').read_text())
    assert lease['scopes'][rest.scope[0]]['trigger_unknown'] is False, (
        'No request was attempted, but recovery would now require finding an impossible active operation')


def test_summary_generation_resamples_invalid_output_then_succeeds(tmp_path, monkeypatch):
    runtime, _, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    monkeypatch.setattr(profiles, 'get_llm_profile', lambda name: {'model': 'M', 'base_url': 'http://fake/v1',
                                                                 'label': 'L', 'api_key_env': 'FAKE'})
    bad = {'canonical_observations': [observation() | {'confidence': 'moderate'}]}
    good = {'canonical_observations': [observation()]}
    calls = []
    def llm(unit, **kwargs):
        kwargs['retry_policy'].charge()
        calls.append(1)
        return (bad if len(calls) == 1 else good), 'raw'
    monkeypatch.setattr(reflect, 'call_llm', llm)
    out = runtime.generate_summary(frozen_input(), retry_policy=policy())
    assert len(calls) == 2
    assert runtime.validate_summary(out['artifact']) is True


def test_summary_generation_exhausts_resamples_then_raises(tmp_path, monkeypatch):
    runtime, _, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    monkeypatch.setattr(profiles, 'get_llm_profile', lambda name: {'model': 'M', 'base_url': 'http://fake/v1',
                                                                 'label': 'L', 'api_key_env': 'FAKE'})
    calls = []
    def llm(unit, **kwargs):
        kwargs['retry_policy'].charge()
        calls.append(1)
        return {'canonical_observations': [observation() | {'confidence': 'moderate'}]}, 'raw'
    monkeypatch.setattr(reflect, 'call_llm', llm)
    from hindsight_offline_retry import RemoteCallError
    with pytest.raises(RemoteCallError, match='invalid_output'):
        runtime.generate_summary(frozen_input(), retry_policy=policy(max_attempts=2))
    assert len(calls) == 2


def test_summary_identity_fields_are_overwritten_not_trusted(tmp_path, monkeypatch):
    runtime, _, _ = make_runtime(tmp_path)
    import hindsight_minimax_import as profiles
    import offline_hindsight_reflect_consolidate as reflect
    monkeypatch.setattr(profiles, 'get_llm_profile', lambda name: {'model': 'M', 'base_url': 'http://fake/v1',
                                                                 'label': 'L', 'api_key_env': 'FAKE'})
    obj = {'canonical_observations': [observation() | {'bank': 'wrong-bank', 'source_ids': ['bogus'],
                                                        'source_documents': ['bogus-doc']}]}
    monkeypatch.setattr(reflect, 'call_llm', lambda *a, **k: (obj, 'raw'))
    out = runtime.generate_summary(frozen_input(), retry_policy=policy())
    row = out['artifact']['canonical_observations'][0]
    assert row['bank'] == 'bank-A'
    assert row['source_ids'] == ['fact-a']
    assert row['source_documents'] == [batch()['document_id']]
