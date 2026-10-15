"""Execute the real App branch locally with fictional text and an in-memory API.

No production sources, credentials, runtime, service, or network are used.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hindsight_app_approved_run as runner
import hindsight_incremental_sources as sources
from hindsight_incremental_delivery import _freeze


def load_helper(filename, name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter_fixtures = load_helper('test_hindsight_app_source_adapter.py', 'execution_adapter_fixtures')
pipeline_fixtures = load_helper('test_hindsight_incremental_pipeline.py', 'execution_pipeline_fixtures')


class LocalRuntime(pipeline_fixtures.Runtime):
    """Existing pipeline fixture plus the actual runner's runtime interface."""

    def __init__(self, ledger):
        super().__init__(ledger)
        self.constructed = 0
        self.preflights = []
        self.begins = 0
        self.resumes = 0
        self.gets = []
        self.lose_ack = False
        self.lost_operation = None
        self.complete_lost = False
        self.unknown_readbacks = []
        self.on_begin = None
        self.summary_profile = {'base_url': 'http://127.0.0.1:3000/v1', 'model': 'auto_cheep_model'}
        self.summary_profile_forbidden = False

    def _summary_profile(self):
        if self.summary_profile_forbidden:
            pytest.fail('publish-only/no-op recovery must not require a summary route')
        return dict(self.summary_profile)

    def preflight(self, *, require_llm):
        self.preflights.append(require_llm)

    def begin_run(self, policy):
        self.begins += 1
        self.llm_calls = self.native_calls = 0
        if self.on_begin is not None:
            self.on_begin()

    def resume_window(self, ledger):
        self.resumes += 1
        return None

    def retain_items(self, items, *, operation_id=None, **kwargs):
        response = super().retain_items(items, operation_id=operation_id, **kwargs)
        if self.lose_ack and self.lost_operation is None:
            self.lost_operation = operation_id
            self.operations[operation_id]['status'] = 'pending'
            raise TimeoutError('Fictional acknowledgement loss after accepted POST')
        return response

    def get_operation(self, operation_id, **kwargs):
        self.gets.append(operation_id)
        if operation_id == self.lost_operation:
            batch = next(b for b in self.ledger.list_batches() if b['operation_id'] == operation_id)
            self.unknown_readbacks.append(batch['phase'])
            if self.complete_lost:
                self.operations[operation_id]['status'] = 'completed'
        return super().get_operation(operation_id, **kwargs)


@pytest.fixture
def execution(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    home, ledger, baseline, sessions, pins = adapter_fixtures.fixture(tmp_path, counts=(165, 265, 17))
    monkeypatch.setenv('HERMES_HOME', str(home))
    # The full publication path consumes the existing immutable seed reference.
    seed_path, seed_hash = _freeze(ledger.root, [], 'seed')
    seed = {'bank': ledger.bank, 'artifact_path': seed_path, 'artifact_hash': seed_hash,
            'provenance': {'bank': ledger.bank, 'seed_sha256': sources.digest([])}}
    with ledger.locked(), ledger._connection(write=True) as con:
        manifest = ledger._read(con, 'meta', 'legacy')
        manifest.update(seed=seed, seed_identity_hash=sources.digest(seed))
        ledger._put(con, 'meta', 'legacy', manifest)
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    for path in (home / 'sessions').glob('*.json'):
        (frozen / path.name).write_bytes(path.read_bytes())
    report_path = tmp_path / 'execution-report.json'
    cfg = {'bank': ledger.bank, 'api_url': 'http://127.0.0.1:8888', 'hermes_home': str(home),
           'incremental': {'root': str(ledger.storage_root), 'mode': 'weekly', 'include_wiki': True,
               'resume_only': True,
               'retry': {'max_attempts': 1, 'max_calls': 1000, 'unit_deadline_s': 2,
                         'defer_delay': 0, 'poll_s': 0.01, 'base_delay': 0, 'max_delay': 0}}}
    runtime = LocalRuntime(ledger)
    def runtime_factory(config):
        assert config['hermes_home'] == str(home)
        assert config['bank'] == ledger.bank
        assert config['incremental']['resume_only'] is False
        assert config['incremental']['mode'] == 'daily'
        assert config['incremental']['include_wiki'] is False
        runtime.constructed += 1
        return runtime
    monkeypatch.setattr(runner, 'SOURCE', frozen)
    monkeypatch.setattr(runner, 'APPROVED_SOURCE_SHA256', pins)
    monkeypatch.setattr(runner, 'APP_LEDGER_ROOT', ledger.storage_root)
    monkeypatch.setattr(runner, 'APP_HERMES_HOME', home)
    monkeypatch.setattr(runner, 'REPORT', report_path)
    monkeypatch.setattr(runner, 'load_config', lambda: copy.deepcopy(cfg))
    monkeypatch.setattr(runner, 'ArrivalRuntime', runtime_factory)
    def no_sync(*args, **kwargs):
        pytest.fail('App execution must not rewrite already approved production exports')
    monkeypatch.setattr(runner, 'sync_sessions', no_sync)
    return {'home': home, 'ledger': ledger, 'baseline': baseline, 'sessions': sessions,
            'pins': pins, 'runtime': runtime, 'cfg': cfg, 'report': report_path, 'frozen': frozen}


def execute(case):
    code = runner.main(['--source-mode', 'app', '--execute'])
    report = json.loads(case['report'].read_bytes())
    assert report['source_mode'] == 'app' and report['approved_messages'] == 447
    return code, report


def counters(runtime):
    return (len(runtime.posts), runtime.generator_calls, runtime.windows,
            runtime.gates, runtime.publishes)


@pytest.mark.parametrize('scenario', ['complete', 'unknown-post', 'publish-failure', 'post-admission-source-change'])
def test_real_execution_and_recovery_keep_one_capture_operation_and_publication(execution, scenario):
    case = execution
    runtime, ledger = case['runtime'], case['ledger']
    with ledger._connection() as con:
        original_feed = ledger._read(con, 'meta', 'epoch:default:physical-feed')
    runtime.lose_ack = scenario == 'unknown-post'
    runtime.publish_fails = scenario == 'publish-failure'
    if scenario == 'post-admission-source-change':
        def mutate_export_after_admission():
            path = sorted((case['home'] / 'sessions').glob('*.json'))[0]
            value = json.loads(path.read_bytes())
            value['messages'][0]['content'] += ' Mutation after initial admission.'
            path.write_text(json.dumps(value))
        runtime.on_begin = mutate_export_after_admission
    code, report = execute(case)
    assert runtime.constructed == runtime.begins == runtime.resumes == 1
    if scenario == 'post-admission-source-change':
        assert code == 2 and report['runtime_report']['status'] == 'blocked'
        assert ledger.list_batches() == []
        assert not runtime.posts and not runtime.preflights
        assert runtime.generator_calls == runtime.windows == runtime.publishes == 0
        assert report['runtime_report']['calls'] == report['runtime_report']['offline_llm_calls'] == 0
        return
    assert runtime.preflights and runtime.windows > 0
    source_batches = [b for b in ledger.list_batches() if b['kind'] == 'source']
    assert sum(len(b['source_event_ids']) for b in source_batches) == 447
    assert report['input_plan']['new_messages'] == 447
    original_operations = {b['batch_id']: b['operation_id'] for b in source_batches}
    if scenario == 'unknown-post':
        assert code == 2 and report['runtime_report']['status'] == 'deferred'
        lost = runtime.lost_operation
        assert lost is not None and runtime.posts.count(lost) == 1
        assert next(b for b in ledger.list_batches() if b['operation_id'] == lost)['phase'] == 'unknown'
        assert 'unknown' in runtime.unknown_readbacks
        runtime.complete_lost = True
        code, report = execute(case)
        assert runtime.posts.count(lost) == 1 and lost in runtime.gets
    elif scenario == 'publish-failure':
        assert code == 2 and report['runtime_report']['status'] == 'blocked'
        assert any(b['phase'] == 'gated' for b in ledger.list_batches() if b['kind'] == 'daily_summary')
        before = counters(runtime)
        runtime.publish_fails = False
        runtime.summary_profile_forbidden = True
        code, report = execute(case)
        assert counters(runtime)[:2] == before[:2]
        assert runtime.windows == before[2] and runtime.gates == before[3]
        assert runtime.preflights[-1] is False
    assert code == 0 and report['runtime_report']['status'] == 'completed'
    assert report['runtime_report']['published']
    assert runtime.generator_calls > 0 and runtime.publishes > 0
    assert all(b['phase'] == 'published' for b in ledger.list_batches())
    assert all(ledger.get_batch(bid)['operation_id'] == op for bid, op in original_operations.items())
    before, batch_count = counters(runtime), len(ledger.list_batches())
    code, report = execute(case)
    assert code == 0 and report['runtime_report']['status'] == 'no_op'
    assert report['input_plan']['new_messages'] == 0
    assert counters(runtime) == before and len(ledger.list_batches()) == batch_count
    assert report['runtime_report']['llm_calls'] == report['runtime_report']['offline_llm_calls'] == 0
    with ledger._connection() as con:
        assert ledger._read(con, 'meta', 'epoch:default:physical-feed') == original_feed


@pytest.mark.parametrize('scenario', ['frozen-pin', 'production-pin', 'wrong-home', 'wrong-api',
    'wrong-bank', 'wrong-ledger-root', 'wrong-summary-route', 'unrelated-pending', 'mixed-derived-parents'])
def test_admission_blocks_before_runtime_or_resume_and_preserves_existing_work(execution, scenario):
    case = execution
    runtime, ledger = case['runtime'], case['ledger']
    if scenario in {'unrelated-pending', 'mixed-derived-parents'}:
        runtime.publish_fails = True
        code, report = execute(case)
        assert code == 2 and report['runtime_report']['status'] == 'blocked'
        raw = {'role': 'user', 'timestamp': 1792022400, 'content': 'Unrelated fictional pending event.',
               'origin_message_id': 'unrelated-pending-fixture-origin'}
        snapshot = {'source_key': 'default:hermes:unrelated-pending',
            'source_epoch': case['baseline']['profiles'][0]['feed_epoch'],
            'messages': [sources._message(raw)], 'source_hash_before': 'a' * 64,
            'source_hash_after': 'a' * 64}
        snapshot['snapshot_hash'] = sources.snapshot_hash(snapshot)
        foreign = ledger.capture_snapshot(snapshot)[0]
        if scenario == 'mixed-derived-parents':
            own = next(b for b in ledger.list_batches() if b['kind'] == 'source'
                       and b['source_key'] != snapshot['source_key'])
            with ledger.locked(), ledger._connection(write=True) as con:
                child = dict(snapshot, source_key='derived:mixed-fixture', source_epoch='mixed-parent-fixture')
                child['snapshot_hash'] = sources.snapshot_hash(child)
                ledger._make_batch(con, child, child['messages'], kind='daily_summary',
                                   parents=(own['batch_id'], foreign['batch_id']))
    elif scenario in {'frozen-pin', 'production-pin'}:
        directory = case['frozen'] if scenario == 'frozen-pin' else case['home'] / 'sessions'
        path = sorted(directory.glob('*.json'))[0]
        value = json.loads(path.read_bytes())
        value['messages'][0]['content'] += ' Same-count unapproved substitution.'
        path.write_text(json.dumps(value))
    elif scenario == 'wrong-home':
        case['cfg']['hermes_home'] = str(case['home'].parent / 'unapproved-home')
    elif scenario == 'wrong-api':
        case['cfg']['api_url'] = 'http://127.0.0.1:9999'
    elif scenario == 'wrong-bank':
        case['cfg']['bank'] = 'unapproved-bank'
    elif scenario == 'wrong-ledger-root':
        case['cfg']['incremental']['root'] = str(case['home'].parent / 'unapproved-ledger')
    elif scenario == 'wrong-summary-route':
        runtime.summary_profile['base_url'] = 'http://127.0.0.1:3999/v1'
    before = (runtime.constructed, runtime.begins, runtime.resumes, counters(runtime))
    fence = adapter_fixtures.file_fence(ledger)
    if scenario in {'unrelated-pending', 'mixed-derived-parents'}:
        assert runner.main(['--source-mode', 'app', '--execute']) == 2
    else:
        with pytest.raises(ValueError):
            runner.main(['--source-mode', 'app', '--execute'])
    if scenario == 'wrong-summary-route':
        assert runtime.constructed == before[0] + 1
        assert (runtime.begins, runtime.resumes, counters(runtime)) == before[1:]
    else:
        assert (runtime.constructed, runtime.begins, runtime.resumes, counters(runtime)) == before
    assert adapter_fixtures.file_fence(ledger) == fence
    for args in [ ['--source-mode', 'app', '--execute', '--plan-only'],
                  ['--source-mode', 'app', '--execute', '--transition-certificate', 'not-read.json',
                   '--approved-transition-sha256', 'a' * 64] ]:
        with pytest.raises(SystemExit):
            runner.main(args)
