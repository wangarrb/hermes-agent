"""Scratch-only delivery contracts; no live bank, home or provider."""
import copy
import hashlib
import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(tmp_path / 'hermes'))


class Ledger:
    def __init__(self, tmp_path):
        self.root = tmp_path
        payload = {'items': [{'document_id': 'delta::1', 'content': 'new fact'}]}
        path = tmp_path / 'payload.json'
        raw = json.dumps(payload).encode()
        path.write_bytes(raw)
        path.chmod(0o600)
        self.batch = {'bank': 'isolated', 'batch_id': 'b1', 'phase': 'captured',
                      'document_id': 'delta::1', 'payload_path': str(path),
                      'payload_hash': hashlib.sha256(raw).hexdigest(), 'receipts': {}}
    def get_batch(self, batch_id):
        assert batch_id == 'b1'
        return copy.deepcopy(self.batch)
    def set_phase(self, batch_id, phase, *, receipt=None):
        self.batch['phase'] = phase
        self.batch['receipts'][phase] = copy.deepcopy(receipt or {})
        return self.get_batch(batch_id)
    def defer(self, batch_id, *, error_class, next_retry_at, attempts):
        self.batch.update(last_error_class=error_class, next_retry_at=next_retry_at, attempts=attempts)
        return self.get_batch(batch_id)


class Clock:
    value = 1000.0
    def now(self):
        return self.value
    def sleep(self, seconds):
        self.value += seconds


def policy(**kwargs):
    retry = importlib.import_module('hindsight_offline_retry')
    clock = Clock()
    return retry.RetryPolicy(clock=clock.now, wall_clock=clock.now, sleep=clock.sleep,
                             jitter=lambda: 0, **kwargs)


class Client:
    bank = 'isolated'
    def __init__(self):
        self.posts = []
        self.gets = []
        self.status = 'completed'
        self.fail_post = False
    def retain_items(self, items, *, async_mode, operation_id, timeout):
        self.posts.append((operation_id, copy.deepcopy(items)))
        if self.fail_post:
            raise TimeoutError('sensitive prompt must never be logged')
        return {'operation_id': operation_id}
    def get_operation(self, operation_id, **kwargs):
        self.gets.append(operation_id)
        return {'operation_id': operation_id, 'bank_id': self.bank, 'status': self.status}


def deliver(ledger, client, **kwargs):
    module = importlib.import_module('hindsight_incremental_delivery')
    return module.deliver_batch(ledger, ledger.get_batch('b1'), client=client, **kwargs)


def test_ack_loss_reconciles_exact_uuid_without_regeneration(tmp_path):
    ledger = Ledger(tmp_path)
    client = Client()
    client.fail_post = True
    client.status = 'pending'
    calls = []
    def generate(payload, *, retry_policy):
        calls.append(1)
        return {'artifact': {'canonical_observations': [{'insight': 'delta'}]}, 'items': payload['items']}
    first = deliver(ledger, client, generate=generate, validate=lambda a: True, retry_policy=policy(max_calls=2))
    assert first['status'] == 'deferred'
    assert ledger.batch['phase'] in {'submitted', 'unknown'}
    assert 'sensitive' not in json.dumps(first)
    client.status = 'completed'
    resumed_policy = policy()
    resumed_policy.sleep(61)  # only due pending is eligible on the next round
    second = deliver(ledger, client, generate=generate, validate=lambda a: True, retry_policy=resumed_policy)
    assert second['phase'] == 'retained'
    assert len(calls) == 1
    assert len(client.posts) == 1
    assert client.gets[-1] == client.posts[0][0]
    assert ledger.batch['receipts']['generated']['artifact_hash']
    assert deliver(ledger, client, generate=generate, validate=lambda a: True, retry_policy=policy())['status'] == 'no_op'
    assert len(client.posts) == 1


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'error'])
def test_terminal_operation_never_sealed(tmp_path, status):
    ledger, client = Ledger(tmp_path), Client()
    client.status = status
    result = deliver(ledger, client, retry_policy=policy())
    assert result['status'] == 'blocked'
    assert result['phase'] != 'retained'


def test_native_pre_io_call_cap_preserves_deferred_retry_not_permanent_block(tmp_path):
    retry = importlib.import_module('hindsight_offline_retry')
    ledger, client = Ledger(tmp_path), Client()
    def refused_before_post(*_a, **_k):
        raise retry.RetryDeferred('call_cap', next_retry_at=2000)
    client.retain_items = refused_before_post
    result = deliver(ledger, client, retry_policy=policy())
    assert result['status'] == 'deferred' and result['next_retry_at'] == 2000
    assert ledger.batch['next_retry_at'] == 2000
    assert ledger.batch['receipts']['generated']['payload_hash']
    assert ledger.batch['phase'] != 'retained' and client.posts == []


@pytest.mark.parametrize('artifact', [None, {}, {'canonical_observations': []}])
def test_invalid_generated_artifact_never_posts(tmp_path, artifact):
    ledger, client = Ledger(tmp_path), Client()
    result = deliver(ledger, client, generate=lambda p, **kw: {'artifact': artifact, 'items': p['items']},
                     validate=lambda value: bool(value and value.get('canonical_observations')),
                     retry_policy=policy())
    assert result['status'] == 'blocked'
    assert result['error_class'] == 'invalid_output'
    assert not client.posts


def test_tampered_generated_file_cannot_reuse_uuid(tmp_path):
    ledger, client = Ledger(tmp_path), Client()
    client.status = 'pending'
    deliver(ledger, client, retry_policy=policy(max_calls=2))
    receipt = ledger.batch['receipts']['generated']
    Path(receipt['payload_path']).chmod(0o600)
    Path(receipt['payload_path']).write_text('{"items":[{"content":"changed","document_id":"delta::1"}]}')
    resumed_policy = policy()
    resumed_policy.sleep(61)
    result = deliver(ledger, client, retry_policy=resumed_policy)
    assert result['status'] == 'blocked'
    assert result['error_class'] == 'payload_mismatch'
    assert len(client.posts) == 1


def test_permanent_http_not_retried_and_generated_reused_after_transient(tmp_path):
    retry = importlib.import_module('hindsight_offline_retry')
    ledger, client = Ledger(tmp_path), Client()
    calls = []
    def generate(payload, **kwargs):
        calls.append(1)
        return {'artifact': {'summary': ['fact']}, 'items': payload['items']}
    def forbidden(*args, **kwargs):
        client.posts.append(kwargs['operation_id'])
        raise retry.RemoteCallError('http_401', status_code=401)
    client.retain_items = forbidden
    result = deliver(ledger, client, generate=generate, validate=lambda a: True, retry_policy=policy())
    assert result['status'] == 'blocked'
    assert len(client.posts) == 1
    assert ledger.batch['phase'] != 'retained'
    assert ledger.batch['receipts']['generated']['artifact_hash']
    assert len(calls) == 1


def test_real_loopback_lost_ack_readback(tmp_path):
    native = importlib.import_module('hindsight_native_client')
    ledger = Ledger(tmp_path)
    posts, operations = [], {}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            posts.append(body)
            operations[body['operation_id']] = {'operation_id': body['operation_id'], 'status': 'completed', 'bank_id': 'isolated'}
            self.connection.shutdown(2)
            self.connection.close()
        def do_GET(self):
            op_id = self.path.split('?', 1)[0].rsplit('/', 1)[-1]
            body = json.dumps(operations[op_id]).encode()
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = native.HindsightNativeClient(api=f'http://127.0.0.1:{server.server_port}', bank='isolated')
        result = deliver(ledger, client, retry_policy=policy())
        assert result['phase'] == 'retained'
        assert len(posts) == 1
        assert result['receipt']['operation_id'] == posts[0]['operation_id']
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_installed_090_leaf_shape_accepts_nullable_children(tmp_path):
    ledger, client = Ledger(tmp_path), Client()
    client.get_operation = lambda op_id, **kw: {'operation_id': op_id, 'status': 'completed',
                                               'operation_type': 'retain', 'child_operations': None,
                                               'result_metadata': None, 'task_payload': None}
    assert deliver(ledger, client, retry_policy=policy())['phase'] == 'retained'


def test_090_not_found_after_lost_ack_reposts_only_same_uuid_payload(tmp_path):
    ledger, client = Ledger(tmp_path), Client()
    original = client.retain_items
    def lose_first(items, **kwargs):
        if not client.posts:
            client.posts.append((kwargs['operation_id'], copy.deepcopy(items)))
            raise TimeoutError()
        return original(items, **kwargs)
    client.retain_items = lose_first
    def get(op_id, **kwargs):
        client.gets.append(op_id)
        return {'operation_id': op_id, 'status': 'not_found' if len(client.posts) == 1 else 'completed',
                'child_operations': None}
    client.get_operation = get
    result = deliver(ledger, client, retry_policy=policy())
    assert result['phase'] == 'retained'
    assert client.posts[0] == client.posts[1]
    assert client.gets == [client.posts[0][0], client.posts[0][0]]


def test_same_uuid_changed_payload_even_with_new_file_hash_is_rejected(tmp_path):
    ledger, client = Ledger(tmp_path), Client()
    client.status = 'pending'
    deliver(ledger, client, retry_policy=policy(max_calls=2))
    receipt = ledger.batch['receipts']['generated']
    body = json.loads(Path(receipt['payload_path']).read_text())
    body['items'][0]['content'] = 'different'
    raw = json.dumps(body).encode()
    altered = tmp_path / 'newpayload.json'
    altered.write_bytes(raw)
    altered.chmod(0o600)
    receipt['payload_path'] = str(altered)
    receipt['payload_hash'] = hashlib.sha256(raw).hexdigest()
    p = policy()
    p.sleep(61)
    result = deliver(ledger, client, retry_policy=p)
    assert result['error_class'] == 'payload_mismatch'
    assert len(client.posts) == 1


def test_child_get_at_callcap_defers_not_blocks(tmp_path):
    ledger, client = Ledger(tmp_path), Client()
    client.get_operation = lambda op_id, **kw: {'operation_id': op_id, 'status': 'completed', 'child_operation_ids': ['child']}
    result = deliver(ledger, client, retry_policy=policy(max_calls=2))
    assert result['status'] == 'deferred'
    assert result['error_class'] == 'call_cap'
    assert result['calls'] == 2


def test_generate_requires_schema_validator_before_paid_generation(tmp_path):
    ledger, client = Ledger(tmp_path), Client()
    calls = []
    def generate(p, **kw):
        calls.append(1)
        return {'artifact': {'wrong': 'schema'}, 'items': p['items']}
    assert deliver(ledger, client, generate=generate, retry_policy=policy())['status'] == 'blocked'
    assert not calls
    assert not client.posts


@pytest.mark.parametrize('field', ['bank', 'batch_id', 'operation_id', 'input_hash', 'payload_hash', 'children'])
def test_retained_noop_requires_full_durable_binding_and_child_proof(tmp_path, field):
    ledger, client = Ledger(tmp_path), Client()
    assert deliver(ledger, client, retry_policy=policy())['phase'] == 'retained'
    before = (len(client.posts), len(client.gets))
    ledger.batch['receipts']['retained'][field] = None
    result = deliver(ledger, client, retry_policy=policy())
    assert result['status'] == 'blocked'
    assert result['error_class'] == 'retained_receipt_mismatch'
    assert result['phase'] == 'retained' and result['calls'] == 0
    assert before == (len(client.posts), len(client.gets))
    assert 'last_error_class' not in ledger.batch  # Never regress an already retained phase.


@pytest.mark.parametrize('response', [
    {'status': 'completed', 'child_operations': None, 'total_children': 1, 'completed_children': 1},
    {'status': 'completed', 'child_operations': [{'id': 'unexpected', 'status': 'completed'}],
     'child_operation_ids': ['expected']},
    {'status': 'completed', 'child_operations': [{'id': 'child', 'status': 'failed'}]},
])
def test_parent_completion_without_exact_successful_children_cannot_seal(tmp_path, response):
    ledger, client = Ledger(tmp_path), Client()
    client.get_operation = lambda operation_id, **kw: {'operation_id': operation_id, **response}
    result = deliver(ledger, client, retry_policy=policy(max_calls=2))
    assert result['phase'] != 'retained'
    assert len(client.posts) == 1


def test_child_http404_does_not_authorize_reposting_parent(tmp_path):
    from hindsight_offline_retry import RemoteCallError
    ledger, client = Ledger(tmp_path), Client()
    def get(operation_id, **kw):
        if operation_id == 'child':
            raise RemoteCallError('http_404', status_code=404)
        return {'operation_id': operation_id, 'status': 'completed', 'child_operation_ids': ['child']}
    client.get_operation = get
    result = deliver(ledger, client, retry_policy=policy())
    assert result['phase'] != 'retained'
    assert len(client.posts) == 1


@pytest.mark.parametrize('child_id', [True, 123, None, [], {}, '', ' ', 'child\nid'])
def test_normalized_child_operation_identity_cannot_be_coerced(child_id):
    from hindsight_incremental_delivery import operation_receipt
    from hindsight_offline_retry import RemoteCallError
    response = {'operation_id': 'parent', 'bank_id': 'isolated', 'status': 'completed',
        'child_operations': [{'operation_id': child_id, 'bank_id': 'isolated',
                              'status': 'completed', 'child_operations': None}]}
    with pytest.raises(RemoteCallError, match='operation_unknown'):
        operation_receipt(response, 'parent', 'isolated')


def test_real_ledger_strict_wire_generated_crash_ack_loss_and_restart(tmp_path):
    """Real capture/SQLite/delivery/native HTTP; server rejects invalid wire items."""
    from datetime import datetime
    from hindsight_incremental_ledger import IncrementalLedger
    from hindsight_incremental_sources import snapshot_hash
    from hindsight_incremental_delivery import deliver_batch
    from hindsight_native_client import HindsightNativeClient

    def snapshot(*ids):
        value = {'source_key': 'default:hermes:session', 'source_epoch': 'epoch-1',
                 'profile': 'default', 'source_hash_before': 'a' * 64,
                 'source_hash_after': 'a' * 64,
                 'messages': [{'event_id': event, 'origin': event, 'role': 'user',
                               'timestamp': '1990-01-01T02:30:00Z', 'content': 'Fact ' + event}
                              for event in ids]}
        value['snapshot_hash'] = snapshot_hash(value)
        return value

    old, new = snapshot('old'), snapshot('old', 'new')
    root = tmp_path / 'state'
    ledger = IncrementalLedger(root, 'isolated', create=True)
    ledger.initialize_legacy([old], {'seed_hash': 'b' * 64}, [],
                             confirm='initialize-hindsight-incremental')
    assert ledger.capture_snapshot(old) == []
    raw = ledger.capture_snapshot(new)[0]
    posts, gets, rejected, operations = [], [], [], {}
    server_state = {'summary_id': None, 'summary_completed': False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
        def send_json(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            valid = value.get('async') is True and bool(value.get('items'))
            for item in value.get('items', []):
                valid = valid and isinstance(item.get('content'), str) and bool(item.get('document_id'))
                metadata = item.get('metadata', {})
                valid = valid and isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values())
                try:
                    datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
                except (ValueError, TypeError, KeyError):
                    valid = False
            if not valid:
                rejected.append(value)
                self.send_json(400, {'error': 'invalid MemoryItem schema'})
                return
            posts.append(value)
            operation_id = value['operation_id']
            operations[operation_id] = value
            if operation_id == server_state['summary_id']:
                self.connection.shutdown(2)
                self.connection.close()
                return
            self.send_json(200, {'operation_id': operation_id})
        def do_GET(self):
            operation_id = self.path.split('?', 1)[0].rsplit('/', 1)[-1]
            gets.append(operation_id)
            if operation_id == server_state['summary_id']:
                response = {'operation_id': operation_id, 'bank_id': 'isolated', 'status': 'completed',
                            'child_operation_ids': ['summary-child'], 'child_operations': None}
            elif operation_id == 'summary-child':
                response = {'operation_id': operation_id, 'bank_id': 'isolated', 'child_operations': None,
                            'status': 'completed' if server_state['summary_completed'] else 'pending'}
            else:
                response = {'operation_id': operation_id, 'bank_id': 'isolated',
                            'status': 'completed' if operation_id in operations else 'not_found',
                            'child_operations': None}
            self.send_json(200, response)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = HindsightNativeClient(api=f'http://127.0.0.1:{server.server_port}', bank='isolated')
    generation_calls = []
    def generate(source, *, retry_policy):
        retry_policy.charge()  # Fake paid boundary participates in the real shared budget.
        generation_calls.append(source)
        items = copy.deepcopy(source['items'])
        items[0]['content'] = 'A schema-validated summary of only the new parent.'
        return {'items': items, 'artifact': {'canonical_observations': [{'insight': 'new parent fact'}]}}
    def validate(artifact):
        return bool(artifact.get('canonical_observations'))
    try:
        with ledger.locked():
            retained_raw = deliver_batch(ledger, raw, client=client, retry_policy=policy())
        assert retained_raw['phase'] == 'retained', retained_raw
        wire = posts[0]['items'][0]
        assert wire['timestamp'] == new['messages'][-1]['timestamp']
        assert json.loads(wire['metadata']['source_event_ids']) == ['new']
        assert json.loads(wire['metadata']['parent_batch_ids']) == []
        assert retained_raw['receipt']['input_hash'] == raw['payload_hash']
        derived_messages = [{'event_id': 'summary-input', 'origin': raw['batch_id'], 'role': 'assistant',
                             'timestamp': new['messages'][-1]['timestamp'], 'content': 'New fact only'}]
        summary = ledger.capture_derived([raw['batch_id']], derived_messages)[0]
        server_state['summary_id'] = summary['operation_id']
        with ledger.locked():
            generated = deliver_batch(ledger, summary, client=client, generate=generate, validate=validate,
                                      retry_policy=policy(max_calls=1))
        assert generated['phase'] == 'generated' and generated['status'] == 'deferred', generated
        assert len(posts) == 1 and len(generation_calls) == 1
        generated_receipt = ledger.get_batch(summary['batch_id'])['receipts']['generated']
        assert generated_receipt['input_hash'] == summary['payload_hash']
        assert generated_receipt['payload_hash'] != summary['payload_hash']

        ledger = IncrementalLedger(root, 'isolated')
        due = policy(max_calls=2)
        due.sleep(61)
        with ledger.locked():
            unknown = deliver_batch(ledger, summary, client=client, generate=generate, validate=validate,
                                    retry_policy=due)
        assert unknown['status'] == 'deferred' and unknown['phase'] == 'unknown', unknown
        assert unknown['calls'] == 2 and len(posts) == 2 and len(generation_calls) == 1
        assert ledger.get_batch(raw['batch_id'])['phase'] == 'retained'
        assert ledger.get_batch(summary['batch_id'])['receipts']['generated'] == generated_receipt

        ledger = IncrementalLedger(root, 'isolated')
        server_state['summary_completed'] = True
        due = policy()
        due.sleep(122)
        with ledger.locked():
            retained_summary = deliver_batch(ledger, summary, client=client, generate=generate,
                                             validate=validate, retry_policy=due)
        assert retained_summary['phase'] == 'retained', retained_summary
        assert retained_summary['calls'] == 2 and len(posts) == 2 and len(generation_calls) == 1
        receipt = retained_summary['receipt']
        assert receipt['bank'] == 'isolated' and receipt['batch_id'] == summary['batch_id']
        assert receipt['input_hash'] == summary['payload_hash']
        assert receipt['operation_id'] == posts[1]['operation_id'] == summary['operation_id']
        assert receipt['payload_hash'] == generated_receipt['payload_hash']
        assert receipt['children'][0]['operation_id'] == 'summary-child'
        assert receipt['children'][0]['status'] == 'completed'
        frozen = json.loads(Path(generated_receipt['payload_path']).read_text())
        assert posts[1] == frozen
        assert json.loads(posts[1]['items'][0]['metadata']['parent_batch_ids']) == [raw['batch_id']]
        assert ledger.capture_snapshot(new) == []
        assert ledger.capture_derived([raw['batch_id']], derived_messages) == []
        before = (len(posts), len(gets), len(generation_calls))
        with ledger.locked():
            repeat = [deliver_batch(ledger, ledger.get_batch(batch['batch_id']), client=client,
                                    generate=generate, validate=validate, retry_policy=policy())
                      for batch in (raw, summary)]
        assert all(r['status'] == 'no_op' and r['calls'] == 0 for r in repeat)
        assert before == (len(posts), len(gets), len(generation_calls))
        assert not rejected
        evidence = {'raw': retained_raw, 'generated_restart': generated, 'ack_loss': unknown,
                    'summary_resume': retained_summary, 'repeat': repeat,
                    'post_count': len(posts), 'get_count': len(gets),
                    'generation_count': len(generation_calls), 'schema_rejections': len(rejected)}
        (tmp_path / 'integration-evidence.json').write_text(json.dumps(evidence, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_delivery_freeze_and_streamed_reader_preserve_exact_json(tmp_path):
    import io
    import hindsight_incremental_merge as merge
    module = importlib.import_module('hindsight_incremental_delivery')
    values = [None, -0.0, 1e-20, (1, '中文😀', None),
              {'items': [{'nested': (None, True, '𐀀'), 'vector': '[0.1,-0.0]'}]}]
    for i, value in enumerate(values):
        expected = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                              allow_nan=False).encode('utf-8')
        path, digest = module._freeze(tmp_path, value, f'golden-{i}')
        assert Path(path).read_bytes() == expected
        assert digest == hashlib.sha256(expected).hexdigest()
        assert Path(path).stat().st_mode & 0o777 == 0o400
        assert module._read(tmp_path, path, digest) == json.loads(expected)
        assert module._freeze(tmp_path, value, f'golden-{i}') == (path, digest)
        for encoding in ('utf-8', 'utf-8-sig', 'utf-16', 'utf-32'):
            assert merge._load_json(io.BytesIO(expected.decode().encode(encoding))) == json.loads(expected)
    assert merge._load_json(io.BytesIO(b'{"x":0,"x":1}')) == {'x': 1}
    for malformed in (b'', b'[1,]', b'{"x":1,}', b'{}[]', b'{1:2}', b'[1e]'):
        with pytest.raises(ValueError):
            merge._load_json(io.BytesIO(malformed))
    # Numeric tokens and escaped/non-BMP strings may straddle a read boundary.
    for padding in (65530, 65531, 65532, 65533, 65534, 65535):
        raw = b' ' * padding + b'[1e-20,"\\ud83d\\ude00",{"z":[null,false]}]'
        assert merge._load_json(io.BytesIO(raw)) == json.loads(raw)
    with pytest.raises(ValueError):
        module._freeze(tmp_path, {'invalid': float('nan')}, 'bad')


@pytest.mark.linux_only
@pytest.mark.parametrize('operation', ['freeze', 'read'])
def test_delivery_large_initial_seed_in_memory_bounded_child(tmp_path, operation):
    import os
    import subprocess
    program = r'''
import hashlib, json, pathlib, resource, sys
sys.path.insert(0, sys.argv[1])
import hindsight_incremental_delivery as delivery
root, operation = pathlib.Path(sys.argv[2]), sys.argv[3]
value = [{'id': f'unit-{i}', 'text': f'原样😀 {i}',
          'vector': '[' + ','.join(['0.12345678901'] * 1024) + ']',
          'opaque': {'nested': [None, True, '𐀀']}} for i in range(4000)]
encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
digest, byte_count = hashlib.sha256(), 0
for part in encoder.iterencode(value):
    data = part.encode('utf-8'); digest.update(data); byte_count += len(data)
expected = digest.hexdigest()
path = root / 'large-seed.json'
if operation == 'read':
    with path.open('wb') as stream:
        for part in encoder.iterencode(value):
            stream.write(part.encode('utf-8'))
    path.chmod(0o400)
size = int(next(line.split()[1] for line in pathlib.Path('/proc/self/status').read_text().splitlines()
                if line.startswith('VmSize:'))) * 1024
resource.setrlimit(resource.RLIMIT_AS, (size + 128 * 1024 * 1024,) * 2)
if operation == 'freeze':
    path, actual = delivery._freeze(root, value, 'seed')
    path = pathlib.Path(path)
    assert actual == expected and path.stat().st_size == byte_count
    assert path.stat().st_mode & 0o777 == 0o400
else:
    recovered = delivery._read(root, path, expected)
    assert recovered == value and len(recovered) == 4000
print(json.dumps({'operation': operation, 'rows': len(value), 'bytes': byte_count,
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


def test_single_poll_returns_pending_without_sleeping_or_deferring(tmp_path):
    """Rotation step: one poll, no wait, no deferral; the scheduler paces."""
    ledger = Ledger(tmp_path)
    client = Client()
    client.status = 'pending'
    p = policy()
    first = deliver(ledger, client, retry_policy=p, single_poll=True)
    assert first['status'] == 'pending'
    assert first['phase'] in {'submitted', 'unknown'}
    assert ledger.batch['phase'] in {'submitted', 'unknown'}
    assert len(client.posts) == 1 and len(client.gets) == 1
    assert ledger.batch.get('last_error_class') is None
    assert ledger.batch.get('next_retry_at') is None
    assert p.clock() == 1000.0  # never slept while pending
    client.status = 'completed'
    second = deliver(ledger, client, retry_policy=policy(), single_poll=True)
    assert second['status'] == 'completed' and second['phase'] == 'retained'
    assert len(client.posts) == 1 and len(client.gets) == 2
