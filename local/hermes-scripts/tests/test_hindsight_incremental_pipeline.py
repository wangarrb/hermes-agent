from pathlib import Path
import importlib.util
import json
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]


def modules():
    sys.path.insert(0, str(ROOT))
    import hindsight_incremental_sources as src
    import hindsight_incremental_ledger as ld
    import hindsight_incremental_delivery as delivery
    import hindsight_incremental_pipeline as flow
    return src, ld, delivery, flow


def snapshot(src, count):
    messages = []
    for i in range(count):
        row = {'role': 'user', 'timestamp': f'2026-10-11T12:{i:02d}:00+00:00',
               'origin': f'message-{i}', 'content': f'New collision avoidance finding {i}: keep the measured evidence.'}
        row['event_id'] = src.digest(row)
        messages.append(row)
    value = {'source_key': 'default:session-A', 'source_epoch': 'epoch-A', 'profile': 'default',
             'messages': messages, 'source_hash_before': src.digest(messages), 'source_hash_after': src.digest(messages)}
    value['snapshot_hash'] = src.snapshot_hash(value)
    return value


class Runtime:
    def __init__(self, ledger, *, gate_pass=True, publish_fails=False):
        self.bank = ledger.bank
        self.ledger = ledger
        self.posts = []
        self.operations = {}
        self.generator_calls = 0
        self.llm_calls = 0
        self.windows = 0
        self.cleanups = 0
        self.gates = 0
        self.publishes = 0
        self.published = {}
        self.gate_pass = gate_pass
        self.publish_fails = publish_fails
        self.empty_facts = False

    def retain_items(self, items, *, async_mode=True, operation_id=None, timeout=None):
        assert async_mode is True
        assert operation_id is not None
        assert all(isinstance(v, str) for item in items for v in item.get('metadata', {}).values())
        assert all(item.get('observation_scopes') == [[f'arrival-batch:{item["document_id"].split("::")[-1]}']] for item in items)
        if operation_id not in self.operations:
            self.posts.append(operation_id)
            self.operations[operation_id] = {'operation_id': operation_id, 'bank_id': self.bank,
                                            'status': 'completed', 'child_operations': None}
        return {'operation_id': operation_id, 'status': 'queued'}

    def get_operation(self, operation_id, timeout=None):
        return self.operations.get(operation_id, {'operation_id': operation_id, 'status': 'not_found'})

    def open_window(self, ledger):
        self.windows += 1
        return {'owner': 'scoped-only', 'legacy_pending': 56}

    def cleanup_window(self, ledger, window):
        self.cleanups += 1
        return 'restored'

    def facts_for_batch(self, batch):
        if self.empty_facts:
            return []
        return [{'id': str(uuid.uuid5(uuid.NAMESPACE_URL, batch['batch_id'])),
                 'document_id': batch['document_id'], 'text': 'A verified new collision avoidance fact.',
                 'type': 'world', 'created_at': '2026-10-11T13:00:00+00:00'}]

    def consolidate_batch(self, batch, policy):
        return {'bank': self.bank, 'scope': [f'arrival-batch:{batch["batch_id"]}'],
                'pending_in_scope': 0, 'failed_in_scope': 0, 'operation_ids': [],
                'legacy_pending': 56, 'ready': True}

    def generate_summary(self, frozen, *, retry_policy=None):
        self.generator_calls += 1
        self.llm_calls += 1
        facts = [json.loads(m['content']) for m in frozen['messages']]
        observation = {'bank': self.bank, 'insight': facts[0]['text'], 'topic': 'aeb', 'type': 'fact',
                       'applicability': 'measured scene', 'evidence_ids': [f['id'] for f in facts]}
        artifact = {'canonical_observations': [observation]}
        return {'items': [{'document_id': frozen['items'][0]['document_id'],
                           'content': json.dumps(artifact), 'metadata': {'kind': 'daily_summary'}}],
                'artifact': artifact}

    def validate_summary(self, artifact):
        assert isinstance(artifact.get('canonical_observations'), list)
        return True

    def gate(self, manifest):
        self.gates += 1
        return {'bank': self.bank, 'manifest_sha256': manifest['manifest_sha256'],
                'decision': 'eligible_for_local_proposal' if self.gate_pass else 'blocked_keep_local_only',
                'checks': {k: self.gate_pass for k in ('schema', 'coverage', 'conflict', 'generic_eval', 'local_eval', 'scoped_drain')}}

    def publish_generation(self, manifest, gate):
        self.publishes += 1
        if self.publish_fails:
            raise ConnectionError('test transport failure after frozen gate')
        self.published[manifest['manifest_sha256']] = {'bank': self.bank,
            'manifest_sha256': manifest['manifest_sha256'], 'published': True,
            'readback_verified': True, 'legacy_seed_preserved': True}

    def readback_generation(self, manifest):
        return self.published.get(manifest['manifest_sha256'], {'published': False})


@pytest.fixture
def setup(tmp_path):
    src, ld, delivery, flow = modules()
    ledger = ld.IncrementalLedger(tmp_path / 'state', 'bank-a', create=True)
    path, checksum = delivery._freeze(ledger.root, [], 'seed')
    seed = {'bank': ledger.bank, 'artifact_path': path, 'artifact_hash': checksum,
            'provenance': {'bank': ledger.bank, 'seed_sha256': src.digest([])}}
    old = snapshot(src, 1)
    ledger.initialize_legacy([old], seed, [], confirm='initialize-hindsight-incremental')
    cfg = {'hermes_home': str(tmp_path / 'home'), 'bank': ledger.bank}
    return src, ledger, flow, cfg


def test_no_new_input_is_zero_delivery_llm_window_or_publish(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == 'no_op'
    assert runtime.posts == []
    assert (runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes) == (0, 0, 0, 0)


def test_append_only_new_arrival_completes_and_repeat_costs_zero(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    read = lambda *_a, **_k: [snapshot(src, 2)]
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                source_reader=read, execute=True)
    assert first['status'] == 'completed', first
    assert len(runtime.posts) == 2
    assert runtime.generator_calls == 1
    assert first['published'] is True and first['cleanup'] == 'restored'
    assert {b['kind'] for b in ledger.list_batches()} == {'source', 'daily_summary'}
    assert all(b['phase'] == 'published' for b in ledger.list_batches())
    counts = (len(runtime.posts), runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes)
    repeat = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=read, execute=True)
    assert repeat['status'] == 'no_op'
    assert counts == (len(runtime.posts), runtime.generator_calls, runtime.windows, runtime.gates, runtime.publishes)


def test_publish_failure_resumes_only_frozen_publish_not_successful_inputs(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger, publish_fails=True)
    read = lambda *_a, **_k: [snapshot(src, 2)]
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                source_reader=read, execute=True)
    assert first['status'] == 'blocked'
    child = next(b for b in ledger.list_batches() if b['kind'] == 'daily_summary')
    assert child['phase'] == 'gated'
    posts, generated, gate_count = len(runtime.posts), runtime.generator_calls, runtime.gates
    runtime.publish_fails = False
    second = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=read, execute=True)
    assert second['status'] == 'completed', second
    assert (posts, generated, gate_count) == (len(runtime.posts), runtime.generator_calls, runtime.gates)


def test_blocked_quality_does_not_publish_or_rewind_retained(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger, gate_pass=False)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'deferred'
    assert runtime.publishes == 0
    child = next(b for b in ledger.list_batches() if b['kind'] == 'daily_summary')
    assert child['phase'] == 'locally_merged'
    assert child['receipts']['retained']['status'] == 'completed'


def test_new_input_budget_does_not_reconstruct_existing_history(setup):
    src, ledger, flow, cfg = setup
    cfg['incremental'] = {'max_pending_units': 1, 'max_pending_chars': 500000}
    runtime = Runtime(ledger)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'deferred'
    assert len(runtime.posts) == 1 and runtime.generator_calls == 0
    assert out['used_units'] == 1
    assert ledger.initialization_manifest()['legacy_status'] == 'legacy_unknown'


def test_plan_does_not_capture_or_mutate_ledger(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    before = ledger.db_path.read_bytes()
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=False)
    assert out['status'] == 'plan'
    assert ledger.db_path.read_bytes() == before
    assert len(ledger.list_batches()) == 0 and runtime.posts == []


def test_successful_zero_fact_source_does_not_remain_permanently_pending(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    runtime.empty_facts = True
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                              source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'completed', out
    assert runtime.generator_calls == 0 and runtime.publishes == 0
    repeat = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert repeat['status'] == 'no_op'


def test_opening_failure_reconciles_durable_window_before_returning(setup):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    original = runtime.open_window
    def partial_open(value):
        setattr(runtime, '_window', dict(original(value), status='active'))
        raise ConnectionError('fixture ACK lost after window opened')
    runtime.open_window = partial_open
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'blocked'
    assert out['cleanup'] == 'restored'
    assert runtime.cleanups == 1 and runtime.posts == []


def test_native_deferral_persists_exact_retry_and_blocks_unfinished_parent_publish(setup):
    src, ledger, flow, cfg = setup
    from hindsight_offline_retry import RetryDeferred
    runtime = Runtime(ledger)
    original = runtime.consolidate_batch
    def postpone_source(batch, policy):
        if batch['kind'] == 'source':
            raise RetryDeferred('call_cap', next_retry_at=99999999999)
        return original(batch, policy)
    runtime.consolidate_batch = postpone_source
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    source = next(b for b in ledger.list_batches() if b['kind'] == 'source')
    assert source['phase'] == 'retained' and source['receipts']['retained']['status'] == 'completed'
    assert source['next_retry_at'] == 99999999999
    assert out['status'] == 'deferred'
    assert runtime.publishes == 0 and out['published'] is False


def test_window_budget_refusal_is_deferred_before_any_retain(setup):
    src, ledger, flow, cfg = setup
    from hindsight_offline_retry import RetryDeferred
    runtime = Runtime(ledger)
    def no_budget(_ledger):
        raise RetryDeferred('call_cap', next_retry_at=99999999999)
    runtime.open_window = no_budget
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    batch = ledger.list_batches()[0]
    assert batch['phase'] == 'captured' and batch['next_retry_at'] == 99999999999
    assert out['status'] == out['core_status'] == 'deferred'
    assert runtime.posts == [] and runtime.generator_calls == 0


def test_publication_preflight_blocks_before_any_new_producer(setup, monkeypatch):
    src, ledger, flow, cfg = setup
    from hindsight_incremental_runtime import ArrivalRuntime
    runtime = ArrivalRuntime(dict(cfg, api_url='http://isolated.invalid'))
    runtime.ledger = ledger
    attempts = []
    def missing_adapter():
        attempts.append('adapter')
        raise RuntimeError('fixture configured interpreter lacks SQL driver')
    monkeypatch.setattr(runtime, '_publication', missing_adapter)
    monkeypatch.setattr(runtime.native, 'request', lambda *_a, **_k: attempts.append('network'))
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 2)], execute=True)
    assert out['status'] == 'blocked'
    assert attempts == ['adapter'] and runtime.native_calls == runtime.llm_calls == 0
    assert out['cleanup'] == 'not_opened'
    assert ledger.list_batches()[0]['phase'] == 'captured'


def test_noop_does_not_require_adapters_or_credentials_for_sealed_history(setup, monkeypatch):
    src, ledger, flow, cfg = setup
    from hindsight_incremental_runtime import ArrivalRuntime
    runtime = ArrivalRuntime(dict(cfg, api_url='http://isolated.invalid'))
    runtime.ledger = ledger
    def unavailable():
        raise AssertionError('no-op must not import SQL driver or resolve credentials')
    monkeypatch.setattr(runtime, '_publication', unavailable)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == 'no_op' and out['calls'] == out['llm_calls'] == 0


def test_requested_wiki_tail_is_explicitly_degraded_not_silently_successful(setup):
    src, ledger, flow, cfg = setup
    cfg['incremental'] = {'mode': 'full', 'include_wiki': True}
    runtime = Runtime(ledger)
    out = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
        source_reader=lambda *_a, **_k: [snapshot(src, 1)], execute=True)
    assert out['status'] == out['core_status'] == 'no_op'
    assert out['mode'] == 'full'
    assert out['optional_tails']['wiki']['requested'] is True
    assert out['optional_tails']['wiki']['status'] == 'degraded'
    assert out['optional_tails']['wiki']['reason'] == 'bounded_tail_not_implemented'
    assert runtime.posts == [] and runtime.generator_calls == 0


def test_real_cli_records_requested_mode_and_wiki_lane(setup, monkeypatch, capsys):
    src, ledger, flow, cfg = setup
    cfg['incremental'] = {'root': str(ledger.root.parent)}
    monkeypatch.setattr(flow, 'load_config', lambda *_a, **_k: cfg)
    import hindsight_incremental_runtime as module
    monkeypatch.setattr(module, 'ArrivalRuntime', lambda config: Runtime(ledger))
    rc = flow.main(['run', '--mode', 'full', '--include-wiki', '--confirm', 'run-hindsight-pipeline'])
    out = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert rc == 0 and out['mode'] == 'full'
    assert out['optional_tails']['wiki']['requested'] is True
    assert out['optional_tails']['wiki']['status'] != 'completed'
