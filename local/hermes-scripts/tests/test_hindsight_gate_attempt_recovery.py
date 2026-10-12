"""Normal gate recovery on tiny immutable generations; no production I/O."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from test_hindsight_incremental_publication import env, fake_report as fake_run, ROOT
import hindsight_incremental_merge as merge

STEPS = ['conflict', 'generic-baseline', 'generic-cards', 'local-baseline', 'local-cards']


def read(path):
    return json.loads(Path(path).read_text())


def reference(ledger, manifest):
    return ledger.root / 'publication' / manifest['manifest_sha256'] / 'gate-attempt-reference.json'


def binding(ledger, manifest):
    ref = read(reference(ledger, manifest))
    return ref, read(ref['input_binding']['path'])


class Runner:
    def __init__(self):
        self.calls = []
        self.fail = None
        self.callback = None

    def __call__(self, command, **kwargs):
        assert kwargs['timeout'] == 3600
        args = lambda k: command[command.index(k) + 1]
        step = Path(args('--output-dir')).name
        self.calls.append((step, list(command)))
        if self.fail == step:
            self.fail = None
            raise RuntimeError('isolated child interrupted')
        result = fake_run(command)
        directory = Path(args('--output-dir'))
        directory.mkdir(parents=True, exist_ok=True)
        disk = directory / ('named-report-' + str(len(self.calls)) + '.json')
        disk.write_text(json.dumps(result))
        result['json_path'] = str(disk)
        if self.callback:
            self.callback(step, result)
        return result


@pytest.fixture
def recovery(env):
    mod, adapter, ledger, db, manifest, proof, row = env
    scripts = ledger.root / 'scripts'
    scripts.mkdir()
    for name in ('hindsight_conflict_audit.py', 'hindsight_conflict_core.py',
                 'hindsight_eval.py', 'hindsight_recall_layered.py'):
        shutil.copyfile(ROOT / name, scripts / name)
    adapter.script_dir = scripts
    runner = Runner()
    adapter.run_json = runner
    return mod, adapter, ledger, db, manifest, proof, runner


def invoke(recovery):
    _, adapter, ledger, _, manifest, proof, _ = recovery
    return adapter.gate(manifest, ledger=ledger, scoped_proof=proof)


def test_retry_runs_only_missing_children_and_returns_independently_validated_gate(recovery):
    mod, adapter, ledger, db, manifest, proof, runner = recovery
    runner.fail = 'local-baseline'
    with pytest.raises(RuntimeError, match='interrupted'):
        invoke(recovery)
    before = list(runner.calls)
    gate = invoke(recovery)
    assert [step for step, _ in runner.calls[len(before):]] == ['local-baseline', 'local-cards']
    assert gate['decision'] == 'eligible_for_local_proposal'
    adapter._validate_gate(manifest, gate, ledger)
    ref, inputs = binding(ledger, manifest)
    assert ref['bank'] == ledger.bank and inputs['manifest_sha256'] == manifest['manifest_sha256']
    assert inputs['thresholds'] == mod.THRESHOLDS
    assert inputs['recipe']['eval_environment'] == {'HINDSIGHT_EVAL_REQUEST_TIMEOUT': '600'}
    assert len(inputs['steps']) == 5
    for step in STEPS:
        completion = read(Path(ref['attempt_root']) / (step + '.completion.json'))
        assert Path(completion['disk_report']['path']).stat().st_mode & 0o777 == 0o600
    for label in ('generic', 'local'):
        bench = inputs['benchmarks'][label]
        assert Path(bench['snapshot']['path']).read_bytes() == Path(bench['source']['path']).read_bytes()
        assert bench['snapshot']['sha256'] == bench['source']['sha256']
        assert bench['ids'] == ['case']
        assert Path(bench['snapshot']['path']).stat().st_mode & 0o777 == 0o600
        for step, cmd in runner.calls:
            if step.startswith(label):
                assert cmd[cmd.index('--benchmark') + 1] == bench['snapshot']['path']
                assert '--no-hindsight-observations' not in cmd
                assert cmd[cmd.index('--raw-limit') + 1] == '40'
                assert cmd[cmd.index('--limit') + 1] == '10'
                assert cmd[cmd.index('--top-k') + 1] == '5'
    before = len(runner.calls)
    assert invoke(recovery) == gate
    assert len(runner.calls) == before
    assert db.commits == 0 and db.embeds == []


def test_validator_failure_keeps_last_completed_report_and_retry_calls_no_children(recovery, monkeypatch):
    mod, adapter, ledger, db, manifest, proof, runner = recovery
    original = mod._validate_eval
    def false_validator(report, ids):
        if len(runner.calls) == 5:
            raise ValueError('synthetic old validator rounding defect')
        original(report, ids)
    monkeypatch.setattr(mod, '_validate_eval', false_validator)
    with pytest.raises(ValueError, match='rounding defect'):
        invoke(recovery)
    ref, inputs = binding(ledger, manifest)
    directory = Path(ref['input_binding']['path']).parent
    receipt = read(directory / 'local-cards.completion.json')
    disk = Path(receipt['disk_report']['path'])
    saved = disk.read_bytes()
    assert receipt['input_binding'] == ref['input_binding']
    monkeypatch.setattr(mod, '_validate_eval', original)
    gate = invoke(recovery)
    assert len(runner.calls) == 5 and disk.read_bytes() == saved
    adapter._validate_gate(manifest, gate, ledger)


@pytest.mark.parametrize('change', ['source_benchmark', 'snapshot', 'helper', 'coverage', 'scoped',
                                   'config', 'bank', 'candidate', 'receipt', 'disk_report', 'stdout', 'schema'])
def test_changed_bound_inputs_or_receipts_block_before_any_more_queries(recovery, change):
    mod, adapter, ledger, db, manifest, proof, runner = recovery
    runner.fail = 'local-baseline'
    with pytest.raises(RuntimeError):
        invoke(recovery)
    ref, inputs = binding(ledger, manifest)
    directory = Path(ref['input_binding']['path']).parent
    if change == 'source_benchmark':
        with Path(adapter.config['generic_benchmark']).open('ab') as out:
            out.write(b'{"id":"new-case"}\n')
    elif change == 'snapshot':
        Path(inputs['benchmarks']['generic']['snapshot']['path']).write_text('{"id":"changed"}\n')
    elif change == 'helper':
        with (adapter.script_dir / 'hindsight_eval.py').open('ab') as out:
            out.write(b'\n# changed helper\n')
    elif change == 'coverage':
        original = ledger.get_batch
        ledger.get_batch = lambda bid: dict(original(bid), source_event_ids=['unexpected'])
    elif change == 'scoped':
        proof['unexpected'] = True
    elif change == 'config':
        adapter.config['raw_limit'] = 41
    elif change == 'bank':
        adapter.bank = 'bank-b'
    elif change == 'candidate':
        manifest['root'] = str(ledger.root / 'wrong-root')
    elif change == 'schema':
        inputs['schema_version'] = 'unknown'
        Path(ref['input_binding']['path']).write_text(json.dumps(inputs))
    else:
        receipt = read(directory / 'generic-baseline.completion.json')
        path = directory / 'generic-baseline.completion.json' if change == 'receipt' else Path(
            receipt['disk_report' if change == 'disk_report' else 'stdout']['path'])
        with path.open('ab') as out:
            out.write(b' ')
    before = len(runner.calls)
    with pytest.raises(ValueError):
        invoke(recovery)
    assert len(runner.calls) == before
    assert db.commits == 0 and db.embeds == []


def test_original_benchmark_mutation_during_last_eval_blocks_final_gate(recovery):
    _, adapter, ledger, db, manifest, _, runner = recovery
    def mutate(step, report):
        if step == 'local-cards':
            with Path(adapter.config['local_benchmark']).open('ab') as out:
                out.write(b'\n# changed original bytes\n')
    runner.callback = mutate
    with pytest.raises(ValueError):
        invoke(recovery)
    assert len(runner.calls) == 5
    ref, _ = binding(ledger, manifest)
    assert not (Path(ref['input_binding']['path']).parent / 'gate.json').exists()


def test_durable_binding_survives_crash_before_reference_and_does_not_recapture(recovery, monkeypatch):
    import hindsight_incremental_gate_attempts as attempts
    _, adapter, ledger, _, manifest, _, runner = recovery
    original = attempts.publish_reference
    monkeypatch.setattr(attempts, 'publish_reference', lambda *_a: (_ for _ in ()).throw(RuntimeError('before reference')))
    with pytest.raises(RuntimeError, match='before reference'):
        invoke(recovery)
    assert runner.calls == [] and not reference(ledger, manifest).exists()
    monkeypatch.setattr(attempts, 'publish_reference', original)
    with Path(adapter.config['generic_benchmark']).open('ab') as out:
        out.write(b'\n# changed before first child\n')
    with pytest.raises(ValueError):
        invoke(recovery)
    assert runner.calls == [] and not reference(ledger, manifest).exists()


def test_unchanged_durable_orphan_binding_reconciles_exact_reference(recovery, monkeypatch):
    import hindsight_incremental_gate_attempts as attempts
    _, _, ledger, _, manifest, _, runner = recovery
    original = attempts.publish_reference
    monkeypatch.setattr(attempts, 'publish_reference', lambda *_a: (_ for _ in ()).throw(RuntimeError('before reference')))
    with pytest.raises(RuntimeError):
        invoke(recovery)
    monkeypatch.setattr(attempts, 'publish_reference', original)
    assert invoke(recovery)['decision'] == 'eligible_for_local_proposal'
    assert len(runner.calls) == 5
    ref, _ = binding(ledger, manifest)
    assert ref['manifest_sha256'] == manifest['manifest_sha256']


@pytest.mark.parametrize('defect', ['ids', 'nonfinite', 'summary', 'false_quality'])
def test_recovery_never_turns_invalid_quality_evidence_into_pass(recovery, defect):
    _, adapter, ledger, _, manifest, _, runner = recovery
    def corrupt(step, report):
        if step != 'generic-cards':
            return
        if defect == 'ids':
            report['results'][0]['id'] = 'undeclared'
        elif defect == 'nonfinite':
            report['results'][0]['layered']['score'] = float('nan')
        elif defect == 'summary':
            report['summary']['layered_avg_score'] = 100
        else:
            report['summary']['layered_avg_term_recall'] = 0.5
            report['results'][0]['layered']['term_recall'] = 0.5
        disk = Path(report['json_path'])
        disk.write_text(json.dumps({k: v for k, v in report.items() if k != 'json_path'}))
    runner.callback = corrupt
    if defect == 'false_quality':
        gate = invoke(recovery)
        assert gate['decision'] == 'blocked_keep_local_only'
        with pytest.raises(ValueError):
            adapter._validate_gate(manifest, gate, ledger)
        count = len(runner.calls)
        assert invoke(recovery) == gate
        assert len(runner.calls) == count
    else:
        with pytest.raises(ValueError):
            invoke(recovery)
        count = len(runner.calls)
        with pytest.raises(ValueError):
            invoke(recovery)
        assert len(runner.calls) == count


def test_explicit_unbound_attempt_reference_is_provenance_missing_not_a_report_import(recovery):
    _, _, ledger, _, manifest, _, runner = recovery
    root = reference(ledger, manifest).parent
    old = root / 'old-unbound'
    old.mkdir(parents=True)
    (old / 'conflict.json').write_text('{}')
    reference(ledger, manifest).write_text(json.dumps({'attempt_root': str(old)}))
    with pytest.raises(ValueError, match='provenance_missing'):
        invoke(recovery)
    assert runner.calls == [] and (old / 'conflict.json').read_text() == '{}'


def test_partial_preparation_is_not_recaptured_even_when_originals_change(recovery, monkeypatch):
    import hindsight_incremental_gate_attempts as attempts
    _, adapter, ledger, _, manifest, _, runner = recovery
    original = attempts.atomic_write
    def crash(path, payload):
        if Path(path).name == 'attempt-inputs.json':
            merge._write(Path(path).with_name('attempt-inputs.json.preparing'), b'{"partial":')
            raise RuntimeError('partial binding crash')
        return original(path, payload)
    monkeypatch.setattr(attempts, 'atomic_write', crash)
    with pytest.raises(RuntimeError, match='partial binding'):
        invoke(recovery)
    monkeypatch.setattr(attempts, 'atomic_write', original)
    with Path(adapter.config['generic_benchmark']).open('ab') as out:
        out.write(b'\n# changed after partial prepare\n')
    with pytest.raises(ValueError, match='provenance_missing'):
        invoke(recovery)
    assert runner.calls == []
    partial = reference(ledger, manifest).parent / attempts.SLOT / 'attempt-inputs.json.preparing'
    assert partial.read_bytes() == b'{"partial":'


def test_multiple_input_binding_candidates_are_ambiguous_not_latest_selected(recovery):
    _, _, ledger, _, manifest, _, runner = recovery
    runner.fail = 'local-baseline'
    with pytest.raises(RuntimeError):
        invoke(recovery)
    ref, _ = binding(ledger, manifest)
    extra = reference(ledger, manifest).parent / 'another-attempt'
    extra.mkdir()
    shutil.copyfile(ref['input_binding']['path'], extra / 'attempt-inputs.json')
    count = len(runner.calls)
    with pytest.raises(ValueError, match='ambiguous'):
        invoke(recovery)
    assert len(runner.calls) == count


def test_crash_after_completion_before_stdout_resumes_without_requery(recovery, monkeypatch):
    import hindsight_incremental_gate_attempts as attempts
    _, adapter, ledger, _, manifest, _, runner = recovery
    original = attempts.atomic_write
    def crash(path, payload):
        if Path(path).name == 'local-cards.json':
            raise RuntimeError('after completed child before stdout')
        return original(path, payload)
    monkeypatch.setattr(attempts, 'atomic_write', crash)
    with pytest.raises(RuntimeError, match='before stdout'):
        invoke(recovery)
    ref, _ = binding(ledger, manifest)
    directory = Path(ref['attempt_root'])
    assert not (directory / 'local-cards.json').exists()
    completion = read(directory / 'local-cards.completion.json')
    disk_bytes = Path(completion['disk_report']['path']).read_bytes()
    monkeypatch.setattr(attempts, 'atomic_write', original)
    gate = invoke(recovery)
    assert len(runner.calls) == 5 and gate['checks']['local_eval'] is True
    assert (directory / 'local-cards.json').read_text() == completion['stdout_json']
    assert Path(completion['disk_report']['path']).read_bytes() == disk_bytes
    adapter._validate_gate(manifest, gate, ledger)


def legacy_gate(recovery):
    """A genuinely completed pre-binding gate, with its full original closure."""
    mod, adapter, ledger, _, manifest, proof, runner = recovery
    directory = reference(ledger, manifest).parent / 'legacy-completed'
    directory.mkdir(parents=True)
    coverage = adapter._coverage(manifest, ledger)
    parsed = mod.candidate_parse_proof(manifest['root'], adapter.script_dir)
    artifacts = [mod._freeze(directory / name, value) for name, value in (
        ('coverage.json', coverage), ('candidate-parse-proof.json', parsed), ('scoped-proof.json', proof))]
    artifacts.extend(mod._artifact(Path(adapter.config[label + '_benchmark'])) for label in ('generic', 'local'))
    artifacts.extend(mod._artifact(adapter.script_dir / name) for name in (
        'hindsight_conflict_audit.py', 'hindsight_conflict_core.py', 'hindsight_eval.py', 'hindsight_recall_layered.py'))
    api = adapter.config['api']
    common = [adapter.python, str(adapter.script_dir / 'hindsight_conflict_audit.py'),
              '--cards-root', manifest['root'], '--offline-root', manifest['root'], '--api', api,
              '--bank', ledger.bank, '--output-dir', str(directory / 'conflict'), '--json']
    adapter._run_report(common, directory / 'conflict.json', artifacts)
    conflict_check = mod.evaluate_conflict_audit(directory / 'conflict.json', required=True)
    pairs, ids = [], {}
    for label in ('generic', 'local'):
        benchmark = adapter.config[label + '_benchmark']
        ids[label] = mod._benchmark_ids(Path(benchmark))
        for mode in ('baseline', 'cards'):
            step = label + '-' + mode
            command = [adapter.python, str(adapter.script_dir / 'hindsight_eval.py'),
                '--benchmark', benchmark, '--cards-root', manifest['root'], '--bank', ledger.bank,
                '--api', api, '--output-dir', str(directory / step), '--json',
                '--no-local-cards' if mode == 'baseline' else '--use-local-cards']
            adapter._run_report(command, directory / (step + '.json'), artifacts)
        pairs.append(mod.evaluate_pair(label, directory / (label + '-baseline.json'),
                                       directory / (label + '-cards.json'), **mod.THRESHOLDS))
    gate = {'schema_version': 'hindsight-incremental-gate/1', 'bank': ledger.bank,
        'manifest_sha256': manifest['manifest_sha256'], 'root': manifest['root'],
        'parent_sha256': manifest['parent_sha256'], 'coverage': coverage, 'scoped_proof': proof,
        'artifacts': artifacts, 'pairs': pairs, 'thresholds': copy.deepcopy(mod.THRESHOLDS),
        'conflict_check': conflict_check, 'candidate_parse_proof': parsed, 'benchmark_ids': ids,
        'checks': {key: True for key in mod.CHECKS}, 'decision': 'eligible_for_local_proposal',
        'gate_path': str(directory / 'gate.json')}
    gate['gate_sha256'] = merge.canonical_hash(gate)
    mod._freeze(directory / 'gate.json', gate)
    return gate


def test_legacy_completed_hash_bound_gate_keeps_original_benchmark_paths(recovery):
    mod, adapter, ledger, _, manifest, _, runner = recovery
    gate = legacy_gate(recovery)
    saved = Path(gate['gate_path']).read_bytes()
    count = len(runner.calls)
    adapter._validate_gate(manifest, gate, ledger)
    assert len(runner.calls) == count and not reference(ledger, manifest).exists()
    assert Path(gate['gate_path']).read_bytes() == saved
    with Path(adapter.config['generic_benchmark']).open('ab') as out:
        out.write(b'\n# old bound source changed\n')
    with pytest.raises(ValueError, match='dependency hash'):
        adapter._validate_gate(manifest, gate, ledger)


def test_config_mutation_after_first_child_blocks_before_next_query(recovery):
    _, adapter, _, _, _, _, runner = recovery
    def change(step, report):
        if step == 'conflict':
            adapter.config['raw_limit'] = 41
    runner.callback = change
    with pytest.raises(ValueError):
        invoke(recovery)
    assert [step for step, _ in runner.calls] == ['conflict']


def test_independent_validator_does_not_recreate_missing_attempt_reference(recovery):
    _, adapter, ledger, _, manifest, _, runner = recovery
    gate = invoke(recovery)
    ref_path = reference(ledger, manifest)
    saved = ref_path.read_bytes()
    ref_path.rename(ref_path.with_suffix('.saved'))
    forged = copy.deepcopy(gate)
    forged['artifacts'] = [item for item in forged['artifacts'] if item['path'] != str(ref_path)]
    forged['gate_sha256'] = merge.canonical_hash({k: v for k, v in forged.items() if k != 'gate_sha256'})
    Path(forged['gate_path']).write_text(json.dumps(forged))
    count = len(runner.calls)
    with pytest.raises(ValueError):
        adapter._validate_gate(manifest, forged, ledger)
    assert not ref_path.exists() and ref_path.with_suffix('.saved').read_bytes() == saved
    assert len(runner.calls) == count


def reseal_gate(mod, gate):
    gate['gate_sha256'] = merge.canonical_hash({k: v for k, v in gate.items() if k != 'gate_sha256'})
    Path(gate['gate_path']).write_text(json.dumps(gate))


@pytest.mark.parametrize('missing', ['all', 'benchmark', 'helper', 'coverage', 'parse', 'scoped', 'stdout', 'disk'])
def test_legacy_requires_complete_original_artifact_closure(recovery, missing):
    mod, adapter, ledger, db, manifest, _, runner = recovery
    gate = legacy_gate(recovery)
    path = Path(gate['gate_path']).parent
    omitted = {
        'benchmark': adapter.config['generic_benchmark'],
        'helper': str(adapter.script_dir / 'hindsight_recall_layered.py'),
        'coverage': str(path / 'coverage.json'), 'parse': str(path / 'candidate-parse-proof.json'),
        'scoped': str(path / 'scoped-proof.json'), 'stdout': str(path / 'generic-baseline.json'),
        'disk': read(path / 'generic-cards.json')['json_path'],
    }
    gate['artifacts'] = [] if missing == 'all' else [a for a in gate['artifacts'] if a['path'] != omitted[missing]]
    reseal_gate(mod, gate)
    before = len(runner.calls)
    with pytest.raises(ValueError):
        adapter._validate_gate(manifest, gate, ledger)
    assert len(runner.calls) == before and db.commits == 0 and not db.embeds


@pytest.mark.parametrize('trace', ['slot', 'reference', 'artifact'])
def test_new_gate_never_falls_back_to_legacy_when_binding_fields_disappear(recovery, trace):
    mod, adapter, ledger, _, manifest, _, runner = recovery
    gate = invoke(recovery)
    ref, bound = binding(ledger, manifest)
    gate.pop('input_binding')
    gate.pop('attempt_reference')
    for label in ('generic', 'local'):
        adapter.config[label + '_benchmark'] = bound['benchmarks'][label]['snapshot']['path']
    if trace != 'slot':
        gate['gate_path'] = str(Path(ref['attempt_root']).parent / 'relocated-gate.json')
    if trace == 'artifact':
        # A referenced new binding is still a new trace when the saved reference
        # was moved; validators cannot repair it or bless a legacy fallback.
        reference(ledger, manifest).rename(reference(ledger, manifest).with_suffix('.saved'))
    reseal_gate(mod, gate)
    before = len(runner.calls)
    with pytest.raises(ValueError):
        adapter._validate_gate(manifest, gate, ledger)
    assert len(runner.calls) == before


@pytest.mark.parametrize('stage', ['initial', 'independent'])
@pytest.mark.parametrize('field,value', [
    ('observations', False), ('observations', None), ('observations', 'missing'),
    ('top_k', 1), ('top_k', True), ('top_k', None), ('top_k', 'missing'),
])
def test_report_options_are_literal_command_bound_even_when_metrics_are_correct(recovery, stage, field, value):
    mod, adapter, ledger, db, manifest, _, runner = recovery
    def mutate(report):
        target, key = (report, 'include_hindsight_observations') if field == 'observations' else (report['summary'], 'top_k')
        if value == 'missing':
            target.pop(key)
        else:
            target[key] = value
        mod._validate_eval(report, ['case'])  # Options must not redefine metric policy.
    if stage == 'initial':
        def corrupt(step, report):
            if step == 'generic-baseline':
                mutate(report)
                Path(report['json_path']).write_text(json.dumps({k: v for k, v in report.items() if k != 'json_path'}))
        runner.callback = corrupt
        with pytest.raises(ValueError):
            invoke(recovery)
        assert [s for s, _ in runner.calls] == ['conflict', 'generic-baseline']
    else:
        gate = legacy_gate(recovery)
        stdout = Path(gate['pairs'][0]['baseline_eval'])
        report = read(stdout)
        mutate(report)
        disk = Path(report['json_path'])
        stdout.write_text(json.dumps(report))
        disk.write_text(json.dumps({k: v for k, v in report.items() if k not in {'json_path', 'markdown_path'}}))
        gate['artifacts'] = [mod._artifact(Path(a['path'])) if a['path'] in {str(stdout), str(disk)} else a for a in gate['artifacts']]
        reseal_gate(mod, gate)
        with pytest.raises(ValueError):
            adapter._validate_gate(manifest, gate, ledger)
    assert db.commits == 0 and not db.embeds


@pytest.mark.parametrize('drift', ['coverage', 'scope', 'native_receipt', 'seed', 'index', 'inventory', 'completion', 'stdout', 'disk'])
def test_live_dependencies_and_all_completed_evidence_fenced_before_missing_child(recovery, drift, monkeypatch):
    mod, adapter, ledger, db, manifest, proof, runner = recovery
    native = {'bank': ledger.bank, 'scope': ['arrival-batch:batch-a'], 'ready': True,
              'pending_in_scope': 0, 'failed_in_scope': 0, 'operation_ids': ['actual-operation']}
    original = ledger.get_batch
    ledger.get_batch = lambda bid: dict(original(bid), receipts={'consolidated': {'scoped_proof': native}})
    proof.clear()
    proof.update(bank=ledger.bank, ready=True, pending_in_scope=0,
                 scopes=[native['scope']], batches=[copy.deepcopy(native)], legacy_pending=56)
    candidate_calls, seed_reads = [], []
    candidate = adapter._candidate
    def checked_candidate(*args, **kwargs):
        candidate_calls.append(1)
        return candidate(*args, **kwargs)
    monkeypatch.setattr(adapter, '_candidate', checked_candidate)
    decoder = merge._read_json
    def checked_read(path, **kwargs):
        if str(path) == ledger.seed['artifact_path']:
            seed_reads.append(1)
        return decoder(path, **kwargs)
    monkeypatch.setattr(merge, '_read_json', checked_read)
    def change(step, report):
        assert len(candidate_calls) == 1 and len(seed_reads) == 1, 'missing-child fence must not reparse the seed/candidate'
        if step != 'generic-baseline':
            return
        if drift == 'coverage':
            ledger.get_batch = lambda bid: dict(original(bid), source_event_ids=['new'], receipts={'consolidated': {'scoped_proof': native}})
        elif drift == 'scope':
            proof['pending_in_scope'] = 1
        elif drift == 'native_receipt':
            native['operation_ids'] = ['different-actual-operation']
        elif drift in {'seed', 'index'}:
            target = Path(ledger.seed['artifact_path']) if drift == 'seed' else Path(manifest['root']) / 'observations_index.jsonl'
            with target.open('ab') as stream:
                stream.write(b' ')
        elif drift == 'inventory':
            (Path(manifest['root']) / 'unexpected.json').write_text('{}')
        else:
            ref, _ = binding(ledger, manifest)
            path = Path(ref['attempt_root'])
            completion = read(path / 'conflict.completion.json')
            target = path / 'conflict.completion.json' if drift == 'completion' else Path(completion['stdout' if drift == 'stdout' else 'disk_report']['path'])
            with target.open('ab') as stream:
                stream.write(b' ')
    runner.callback = change
    with pytest.raises(ValueError):
        invoke(recovery)
    assert [s for s, _ in runner.calls] == ['conflict', 'generic-baseline']
    assert db.commits == 0 and not db.embeds


def test_shared_benchmark_keeps_distinct_snapshots_and_commands(recovery):
    mod, adapter, ledger, db, manifest, _, runner = recovery
    adapter.config['local_benchmark'] = adapter.config['generic_benchmark']
    gate = invoke(recovery)
    adapter._validate_gate(manifest, gate, ledger)
    _, bound = binding(ledger, manifest)
    paths = [a['path'] for a in bound['artifacts']]
    assert paths.count(adapter.config['generic_benchmark']) == 1 and len(paths) == len(set(paths))
    generic, local = (bound['benchmarks'][k] for k in ('generic', 'local'))
    assert generic['source'] == local['source'] and generic['snapshot']['path'] != local['snapshot']['path']
    assert generic['ids'] == local['ids'] == ['case']
    for step, cmd in runner.calls[1:]:
        assert cmd[cmd.index('--benchmark') + 1] == bound['benchmarks'][step.split('-')[0]]['snapshot']['path']
    assert db.commits == 0 and not db.embeds
