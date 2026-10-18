"""Publication observability over tiny SQL-double inventory / named fake reports.

No production SQL, provider, eval or embeddings: reused fixture I/O doubles drive
real candidate/gate/publisher code. Lifecycle tests use event-controlled blocking.
"""
import copy
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from test_hindsight_incremental_publication import env
from test_hindsight_incremental_pipeline import setup, Runtime, snapshot


def progress():
    assert importlib.util.find_spec('hindsight_incremental_progress') is not None, 'runtime progress helper missing'
    return importlib.import_module('hindsight_incremental_progress')


def events(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_heartbeat_tracks_blocked_deepest_operation_with_one_thread_and_counts(monkeypatch):
    p = progress()
    stream = io.StringIO()
    beat = threading.Event()
    release = threading.Event()
    ready = threading.Event()
    started = []
    original = threading.Thread.start
    def start(thread):
        if thread.name == 'hindsight-publication-heartbeat':
            started.append(thread)
        original(thread)
    monkeypatch.setattr(threading.Thread, 'start', start)
    class Sink:
        def write(self, line):
            stream.write(line)
            if json.loads(line)['event'] == 'heartbeat':
                beat.set()
        def flush(self):
            pass
    owner = object()
    now = [10.0]
    failures = []
    def work():
        try:
            with p.reporting_run(owner, interval=0.01, clock=lambda: now[0], sink=Sink()) as reporter:
                with p.reporting_run(owner) as nested:
                    assert nested is reporter
                    with p.span('readback'):
                        with p.span('rows_fetch', row_count=2) as counts:
                            counts.update(unit_count=2)
                            now[0] = 14.0
                            ready.set()
                            assert release.wait(5)
        except BaseException as error:
            failures.append(error)
    worker = threading.Thread(target=work)
    worker.start()
    try:
        assert ready.wait(5) and beat.wait(5)
        active = [e for e in events(stream) if e['event'] == 'heartbeat']
        assert active and all(e['operation'] == 'rows_fetch' and e['row_count'] == e['unit_count'] == 2 for e in active)
        assert all(e['elapsed_seconds'] == 4 and e['run_elapsed_seconds'] == 4 for e in active)
        assert len(started) == 1 and started[0].is_alive()
        assert all(type(e[k]) in (int, float) for e in active for k in e if k.startswith('process_'))
    finally:
        release.set()
        worker.join(5)
    assert not failures and not worker.is_alive() and not started[0].is_alive()
    assert p.active_reporter() is None
    done = [e for e in events(stream) if e['operation'] == 'rows_fetch' and e['event'] == 'done']
    assert done[0]['row_count'] == done[0]['unit_count'] == 2


@pytest.mark.parametrize('failure', [False, True])
def test_context_restores_parent_and_stops_threads_on_success_or_original_exception(failure):
    p = progress()
    stream = io.StringIO()
    error = ValueError('PRIVATE_EXCEPTION_DSN_KEY_BODY_VECTOR')
    parent = object()
    with p.reporting_run(parent, sink=stream) as outer:
        try:
            with p.reporting_run(object(), sink=stream) as inner:
                with p.span('candidate'):
                    if failure:
                        raise error
            assert not failure
        except ValueError as caught:
            assert failure and caught is error
        assert p.active_reporter() is outer and not inner.thread.is_alive()
    assert p.active_reporter() is None and not outer.thread.is_alive()
    assert 'PRIVATE_EXCEPTION' not in stream.getvalue()
    errors = [e for e in events(stream) if e['event'] == 'error']
    assert bool(errors) == failure
    if failure:
        assert all(e['error_type'] == 'ValueError' for e in errors)


@pytest.mark.parametrize('sink_error', [BrokenPipeError, OSError])
def test_sink_failures_never_mask_result_or_original_gate_exception(sink_error):
    p = progress()
    class Sink:
        def write(self, _line):
            raise sink_error('PRIVATE_SINK_MESSAGE')
        def flush(self):
            raise AssertionError('unreachable')
    failure = RuntimeError('PRIVATE_GATE_MESSAGE')
    with p.reporting_run(object(), sink=Sink()) as reporter:
        with p.span('candidate'):
            result = {'untouched': True}
        assert result == {'untouched': True}
        with pytest.raises(RuntimeError) as caught:
            with p.span('validate_gate'):
                raise failure
        assert caught.value is failure
    assert not reporter.thread.is_alive()


def test_progress_rejects_freeform_operations_and_counts_without_logging_them():
    p = progress()
    stream = io.StringIO()
    with p.reporting_run(object(), sink=stream):
        with pytest.raises(ValueError):
            with p.span('postgres://PRIVATE_DSN'):
                pytest.fail('unsafe operation accepted')
        with pytest.raises(TypeError):
            with p.span('candidate', text='PRIVATE_BODY'):
                pytest.fail('unsafe field accepted')
        with pytest.raises(ValueError):
            with p.span('candidate', row_count='PRIVATE_VECTOR'):
                pytest.fail('nonnumeric count accepted')
    assert 'PRIVATE' not in stream.getvalue()


def test_record_preserves_report_and_stdout_without_exposing_private_values(capsys):
    import hindsight_incremental_pipeline as flow
    report = {'stages': []}
    flow._record(report, 'delivery', batch_id='a' * 64, receipt={'key': 'PRIVATE_KEY'}, calls=2)
    assert report['stages'][0]['batch_id'] == 'a' * 64
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {'stage': 'delivery', 'batch_id': 'a' * 64, 'calls': 2}
    assert captured.err == '' and 'PRIVATE' not in captured.out
    p = progress()
    with p.reporting_run(object()):
        flow._record(report, 'delivery', batch_id='a' * 64, error_class='transport', calls=2)
        print(json.dumps({'final': 'unchanged'}))
    captured = capsys.readouterr()
    assert [json.loads(line) for line in captured.out.splitlines()][-1] == {'final': 'unchanged'}
    assert 'PRIVATE' not in captured.err
    assert any(e['operation'] == 'delivery' and e['calls'] == 2 for e in map(json.loads, captured.err.splitlines()))


def test_native_fetch_hash_timings_preserve_rows_identity_and_never_log_inventory(env):
    p = progress()
    mod, adapter, ledger, db, manifest, proof, row = env
    original = copy.deepcopy(db.data)
    stream = io.StringIO()
    with p.reporting_run(adapter, sink=stream):
        with adapter._readonly() as conn:
            docs, units = mod._read_remote(conn, ledger.bank)
        actual = mod._remote_identity(ledger.bank, docs, units)
    assert actual == ledger.seed['remote_seed'] and db.data == original
    recorded = events(stream)
    for operation, count_key, count in [('rows_fetch', 'row_count', 1), ('rows_hash', 'row_count', 1),
            ('remote_documents_hash', 'document_count', len(docs)), ('remote_units_hash', 'unit_count', len(units)),
            ('remote_combined_hash', 'unit_count', len(units))]:
        assert any(e['operation'] == operation and e['event'] == 'done' and e[count_key] == count for e in recorded)
    for private in [adapter.dsn, docs[0]['original_text'], units[0]['text'], units[0]['embedding'], units[0]['id']]:
        assert private not in stream.getvalue()
    assert db.embeds == [] and db.commits == 0


def test_passed_frozen_gate_publisher_resume_is_observable_without_new_eval(env):
    p = progress()
    mod, adapter, ledger, db, manifest, proof, row = env
    gate = adapter.gate(manifest, ledger=ledger, scoped_proof=proof)
    gate_before, manifest_before = copy.deepcopy(gate), copy.deepcopy(manifest)
    frozen_before = {str(path): path.read_bytes() for path in Path(gate['gate_path']).parent.iterdir() if path.is_file()}
    adapter.run_json = lambda *_a, **_kw: pytest.fail('frozen gate resume must not call eval')
    stream = io.StringIO()
    with p.reporting_run(adapter, sink=stream) as reporter:
        result = adapter.publish_generation(manifest, gate, ledger=ledger)
        again = adapter.publish_generation(manifest, gate, ledger=ledger)
    assert result['readback_verified'] and again['readback_verified']
    assert len(db.embeds) == db.commits == 1 and not reporter.thread.is_alive()
    assert gate == gate_before and manifest == manifest_before
    assert all(Path(path).read_bytes() == value for path, value in frozen_before.items())
    recorded = events(stream)
    required = {'publish_generation', 'readback_generation', 'readback', 'candidate', 'initial_seed',
                'seed_read', 'candidate_load', 'validate_gate', 'seed_preserved', 'readonly_connect',
                'publisher_publish', 'prewrite_check', 'precommit_check', 'embeddings'}
    assert required <= {e['operation'] for e in recorded if e['event'] == 'done'}
    embedding = [e for e in recorded if e['operation'] == 'embeddings' and e['event'] == 'done']
    assert len(embedding) == 1 and embedding[0]['requested_count'] == embedding[0]['received_count'] == 1
    assert adapter.dsn not in stream.getvalue() and row['insight'] not in stream.getvalue()


def test_real_pipeline_noop_starts_no_progress_thread_or_credentials(setup, monkeypatch, capsys):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    runtime.preflight = lambda **_kw: pytest.fail('no-op credential preflight')
    starts = []
    original = threading.Thread.start
    monkeypatch.setattr(threading.Thread, 'start', lambda thread: (starts.append(thread), original(thread))[1])
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                source_reader=lambda *_a, **_kw: [snapshot(src, 1)], execute=True)
    assert report['status'] == 'no_op' and report['calls'] == report['llm_calls'] == 0
    assert runtime.posts == [] and runtime.windows == runtime.gates == runtime.publishes == 0
    captured = capsys.readouterr()
    assert starts == [] and captured.err == ''
    assert [json.loads(line) for line in captured.out.splitlines()] == report['stages']


def test_pipeline_publish_retry_has_one_reporter_and_restores_context(setup, monkeypatch, capsys):
    p = progress()
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger, publish_fails=True)
    read = lambda *_a, **_kw: [snapshot(src, 2)]
    runtime.preflight = lambda **_kw: None
    threads = []
    original = threading.Thread.start
    def start(thread):
        if thread.name == 'hindsight-publication-heartbeat':
            threads.append(thread)
        original(thread)
    monkeypatch.setattr(threading.Thread, 'start', start)
    first = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                source_reader=read, execute=True)
    assert first['status'] == 'blocked' and p.active_reporter() is None
    before = (len(runtime.posts), runtime.generator_calls, runtime.gates)
    runtime.publish_fails = False
    second = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                 source_reader=read, execute=True)
    assert second['status'] == 'completed' and p.active_reporter() is None
    assert before == (len(runtime.posts), runtime.generator_calls, runtime.gates)
    assert len(threads) == 2 and all(not thread.is_alive() for thread in threads)
    output = capsys.readouterr()
    assert all('stage' in json.loads(line) for line in output.out.splitlines())
    recorded = [json.loads(line) for line in output.err.splitlines()]
    assert any(e['operation'] == 'publish_generation' and e['event'] == 'error' and e['error_type'] == 'ConnectionError' for e in recorded)
    assert any(e['operation'] == 'saved_manifest_validate' and e['event'] == 'done' for e in recorded)


def test_unready_pipeline_noop_is_silent_and_restores_parent_reporter(setup):
    p = progress()
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    stream = io.StringIO()
    with p.reporting_run(object(), sink=stream) as parent:
        report = flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                                     source_reader=lambda *_a, **_kw: [snapshot(src, 1)], execute=True)
        assert report['status'] == 'no_op' and p.active_reporter() is parent
    assert [(e['operation'], e['event']) for e in events(stream)] == [('pipeline', 'start'), ('pipeline', 'done')]


def test_callback_without_owner_does_not_borrow_unrelated_run():
    p = progress()
    stream = io.StringIO()
    @p.timed('prewrite_check', reporter=None)
    def callback(value):
        return value
    value = object()
    with p.reporting_run(object(), sink=stream) as parent:
        assert callback(value) is value and p.active_reporter() is parent
    assert [(e['operation'], e['event']) for e in events(stream)] == [('pipeline', 'start'), ('pipeline', 'done')]


def test_final_report_failure_preserves_exception_stops_thread_and_emits_error(setup, monkeypatch, capsys):
    p = progress()
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    original = flow._freeze
    error = OSError('PRIVATE_REPORT_DSN_KEY')
    def freeze(root, value, suffix):
        if suffix == 'run-report':
            raise error
        return original(root, value, suffix)
    monkeypatch.setattr(flow, '_freeze', freeze)
    threads = []
    start = threading.Thread.start
    def track(thread):
        if thread.name == 'hindsight-publication-heartbeat':
            threads.append(thread)
        start(thread)
    monkeypatch.setattr(threading.Thread, 'start', track)
    with pytest.raises(OSError) as caught:
        flow.run_incremental(cfg, ledger=ledger, client=runtime, llm=runtime.generate_summary,
                             source_reader=lambda *_a, **_kw: [snapshot(src, 2)], execute=True)
    assert caught.value is error and p.active_reporter() is None
    assert len(threads) == 1 and not threads[0].is_alive()
    output = capsys.readouterr()
    assert 'PRIVATE' not in output.err
    assert any(e['operation'] == 'pipeline' and e['event'] == 'error' and e['error_type'] == 'OSError'
               for e in map(json.loads, output.err.splitlines()))


def test_cli_stdout_is_exactly_one_final_json_report(setup, monkeypatch, capsys):
    src, ledger, flow, cfg = setup
    cfg = dict(cfg, incremental={'root': str(ledger.storage_root)})
    runtime = Runtime(ledger)
    import hindsight_incremental_runtime as runtime_module
    monkeypatch.setattr(runtime_module, 'ArrivalRuntime', lambda _cfg: runtime)
    monkeypatch.setattr(flow, 'load_config', lambda *_a, **_kw: cfg)
    original = flow.run_incremental
    def run(config, **kwargs):
        return original(config, source_reader=lambda *_a, **_kw: [snapshot(src, 2)], **kwargs)
    monkeypatch.setattr(flow, 'run_incremental', run)
    assert flow.main(['run', '--confirm', flow.CONFIRM_RUN]) == 0
    output = capsys.readouterr()
    stdout = [json.loads(line) for line in output.out.splitlines()]
    report = stdout[-1]
    assert stdout[:-1] == report['stages']
    assert sum('schema_version' in event for event in stdout) == 1
    assert report['status'] == 'completed' and report['published'] is True
    assert json.loads(Path(report['report_path']).read_text()) == {k: v for k, v in report.items() if k not in {'report_path', 'report_sha256'}}
    assert all(json.loads(line)['event'] in {'start', 'done', 'error', 'heartbeat'} for line in output.err.splitlines())


def test_readonly_span_covers_generator_use_not_creation(env):
    p = progress()
    _mod, adapter, _ledger, db, _manifest, _proof, _row = env
    stream = io.StringIO()
    started, ended = threading.Event(), threading.Event()
    class Sink:
        def write(self, line):
            stream.write(line)
            event = json.loads(line)
            if event['operation'] == 'readonly_transaction':
                (started if event['event'] == 'start' else ended).set()
        def flush(self):
            pass
    with p.reporting_run(adapter, sink=Sink()):
        context = adapter._readonly()
        assert not started.is_set()
        with context:
            assert started.wait(5)
            assert not ended.is_set()
        assert ended.wait(5)
    assert db.commits == 0


def test_callback_reuses_run_across_thread_context_and_nested_same_operation():
    p = progress()
    stream = io.StringIO()
    failures = []
    with p.reporting_run(object(), sink=stream, clock=lambda: 1.0) as owner:
        @p.timed('prewrite_check', reporter=owner)
        def callback():
            with p.span('readback'):
                with p.span('readback'):
                    assert p.active_reporter() is owner
            assert owner.stack[-1]['operation'] == 'prewrite_check'
        def work():
            try:
                assert p.active_reporter() is None
                callback()
                assert p.active_reporter() is None
            except BaseException as error:
                failures.append(error)
        thread = threading.Thread(target=work)
        thread.start()
        thread.join(5)
        assert not thread.is_alive() and not failures
        assert len(owner.stack) == 1 and owner.stack[0]['operation'] == 'pipeline'
    assert len([e for e in events(stream) if e['operation'] == 'prewrite_check' and e['event'] == 'done']) == 1


def test_original_stage_stdout_contract_is_preserved(capsys):
    import hindsight_incremental_pipeline as flow
    report = {'stages': []}
    values = {'batch_id': 'a' * 64, 'calls': 2, 'status': 'retained',
              'receipt': {'private': 'must remain in report only'}}
    flow._record(report, 'delivery', **values)
    captured = capsys.readouterr()
    expected = {'stage': 'delivery', 'batch_id': 'a' * 64, 'calls': 2, 'status': 'retained'}
    assert captured.out == json.dumps(expected, ensure_ascii=False) + '\n'
    assert captured.err == ''
    assert report['stages'] == [{'stage': 'delivery', **values}]


def test_source_parent_receipt_finalization_has_safe_phase_timings(setup, capsys):
    src, ledger, flow, cfg = setup
    runtime = Runtime(ledger)
    report = flow.run_incremental(cfg, ledger=ledger, client=runtime,
                                 llm=runtime.generate_summary,
                                 source_reader=lambda *_a, **_kw: [snapshot(src, 2)], execute=True)
    assert report['status'] == 'completed'
    output = capsys.readouterr()
    recorded = [json.loads(line) for line in output.err.splitlines()]
    for operation in ('source_parent_merge', 'source_parent_gate', 'source_parent_publish'):
        assert any(e['operation'] == operation and e['event'] == 'start' for e in recorded)
        assert any(e['operation'] == operation and e['event'] == 'done' for e in recorded)
    assert all(b['phase'] == 'published' for b in ledger.list_batches())


@pytest.mark.parametrize('block_at', ['write', 'flush'])
def test_blocked_diagnostic_sink_never_stalls_business_error_or_cleanup(block_at):
    p = progress()
    blocked, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = RuntimeError('PRIVATE_ORIGINAL_GATE_FAILURE')
    errors, reporters = [], []
    class Sink:
        heartbeat = False
        def write(self, line):
            self.heartbeat = json.loads(line)['event'] == 'heartbeat'
            if block_at == 'write' and self.heartbeat:
                blocked.set()
                assert release.wait(8)
        def flush(self):
            if block_at == 'flush' and self.heartbeat:
                blocked.set()
                assert release.wait(8)
    def business():
        try:
            with p.reporting_run(object(), interval=0.01, sink=Sink()) as reporter:
                reporters.append(reporter)
                assert blocked.wait(5)
                with p.span('validate_gate'):
                    raise original
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()
    worker = threading.Thread(target=business)
    worker.start()
    try:
        assert blocked.wait(5)
        assert completed.wait(2), 'diagnostic I/O must not hold the business thread'
        assert errors == [original]
    finally:
        release.set()
        worker.join(5)
        for reporter in reporters:
            reporter.thread.join(5)
    assert not worker.is_alive() and all(not r.thread.is_alive() for r in reporters)


def test_bounded_diagnostic_queue_disables_on_congestion_without_delaying_result():
    p = progress()
    blocked, release, completed = threading.Event(), threading.Event(), threading.Event()
    reporters, results, errors = [], [], []
    sentinel = object()
    class Sink:
        def write(self, _line):
            blocked.set()
            assert release.wait(8)
        def flush(self):
            pass
    def business():
        try:
            with p.reporting_run(object(), sink=Sink(), queue_size=2) as reporter:
                reporters.append(reporter)
                assert blocked.wait(5)
                for _ in range(20):
                    with p.span('candidate'):
                        pass
                assert reporter.disabled
                assert reporter.pending.qsize() <= 2
                results.append(sentinel)
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()
    worker = threading.Thread(target=business)
    worker.start()
    try:
        assert blocked.wait(5) and completed.wait(2)
        assert results == [sentinel] and errors == []
    finally:
        release.set()
        worker.join(5)
        for reporter in reporters:
            reporter.thread.join(5)
    assert not worker.is_alive() and all(not r.thread.is_alive() for r in reporters)


@pytest.mark.parametrize('capacity', [0, -1, True, None, '2'])
def test_diagnostic_queue_rejects_unbounded_or_nonliteral_capacity(capacity):
    p = progress()
    with pytest.raises(ValueError, match='finite positive capacity'):
        with p.reporting_run(object(), sink=io.StringIO(), queue_size=capacity):
            pytest.fail('invalid diagnostic capacity accepted')
