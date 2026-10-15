#!/usr/bin/env python3
"""Durable, bank-owned gate attempts. The caller holds the producer flock.

One deterministic attempt slot per manifest avoids latest/glob selection. A
partial preparation is a blocker, never permission to recapture changed inputs.
Completion evidence precedes quality validation; it is not a passing gate.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import hindsight_incremental_merge as merge

SCHEMA = 'hindsight-incremental-gate-inputs/1'
REFERENCE_SCHEMA = 'hindsight-incremental-gate-attempt-reference/1'
COMPLETION_SCHEMA = 'hindsight-incremental-gate-child-completion/1'
SLOT = 'gate-attempt-v1'
SOURCES = ('hindsight_conflict_audit.py', 'hindsight_conflict_core.py',
           'hindsight_eval.py', 'hindsight_recall_layered.py')
STEPS = ('conflict', 'generic-baseline', 'generic-cards', 'local-baseline', 'local-cards')


def artifact(path: Path) -> dict:
    return {'path': str(merge._absolute(path)), 'sha256': merge._file_info(path)['sha256']}


def unique_artifacts(items: list[dict]) -> list[dict]:
    """Deduplicate exact paths only; a second digest is not another identity."""
    by_path = {}
    for item in items:
        if (not isinstance(item, dict) or set(item) != {'path', 'sha256'}
                or not isinstance(item['path'], str)
                or str(merge._absolute(item['path'])) != item['path']):
            raise ValueError('provenance_missing: invalid gate artifact reference')
        merge._hash(item['sha256'], 'gate artifact')
        if by_path.setdefault(item['path'], item) != item:
            raise ValueError('gate artifact path has conflicting hashes')
    return list(by_path.values())


def requires_binding(path: Path, gate: dict, ledger, manifest: dict) -> bool:
    """Missing fields cannot turn persisted new-attempt evidence into legacy."""
    root = merge._absolute(ledger.root) / 'publication' / manifest['manifest_sha256']
    trace_names = {'attempt-inputs.json', 'attempt-inputs.json.preparing',
                   'gate-attempt-reference.json', 'gate-attempt-reference.json.preparing'}
    if ('input_binding' in gate or 'attempt_reference' in gate or SLOT in path.parts
            or (root / SLOT).exists()
            or any((root / name).exists() for name in trace_names)):
        return True
    for item in gate.get('artifacts', []):
        p = merge._absolute(item['path'])
        if SLOT in p.parts or p.name in trace_names or p.name.endswith('.completion.json'):
            return True
    # Inventory rejects traces; it never selects or imports an old report.
    return root.exists() and any(child.is_dir() and any((child / name).exists()
        for name in trace_names) for child in root.iterdir())


def report_outputs_match(stdout: dict, disk: dict) -> bool:
    """Accept the two real writers' optional transport paths, never contradictions.

    Audit persists its self-references; eval adds them only to stdout. If the
    disk includes either key, require its literal value to match stdout before
    comparing the complete payload. Exact raw hashes remain independently bound.
    """
    paths = {'json_path', 'markdown_path'}
    if any(key in disk and (type(disk[key]) is not str or disk[key] != stdout.get(key))
           for key in paths):
        return False
    payload = lambda value: {key: item for key, item in value.items() if key not in paths}
    # Preserve invalid NaN evidence for the separate finite-quality validator.
    return json.dumps(payload(disk), sort_keys=True) == json.dumps(payload(stdout), sort_keys=True)


def validate_legacy_closure(adapter, gate: dict, directory: Path, benchmarks: dict) -> None:
    """Require the original pre-query closure, not a self-authored passing flag."""
    required = [artifact(path) for path in benchmarks.values()]
    required.extend(artifact(adapter.script_dir / name) for name in SOURCES)
    for name, key in (('coverage.json', 'coverage'), ('scoped-proof.json', 'scoped_proof'),
                      ('candidate-parse-proof.json', 'candidate_parse_proof')):
        path = directory / name
        if merge._object(merge._read(path), 'legacy bound proof') != gate[key]:
            raise ValueError('legacy frozen proof differs from current gate binding')
        required.append(artifact(path))
    reports = {'conflict': gate['conflict_check']['audit_path']}
    for pair in gate['pairs']:
        reports[pair['label'] + '-baseline'] = pair['baseline_eval']
        reports[pair['label'] + '-cards'] = pair['cards_eval']
    if set(reports) != set(STEPS):
        raise ValueError('provenance_missing: incomplete legacy report closure')
    for step, token in reports.items():
        path = directory / (step + '.json')
        if token != str(path):
            raise ValueError('legacy report path does not belong to frozen attempt')
        report = merge._object(merge._read(path), 'legacy completed stdout')
        required.append(artifact(path))
        if report.get('json_path'):
            produced = merge._absolute(report['json_path'])
            if (str(produced) != report['json_path'] or not produced.is_relative_to(directory)
                    or produced == path or produced.name == 'latest.json'):
                raise ValueError('legacy report requires exact named disk path in its attempt')
            disk = merge._object(merge._read(produced), 'legacy named disk report')
            if not report_outputs_match(report, disk):
                raise ValueError('legacy command stdout/report artifact mismatch')
            required.append(artifact(produced))
    expected = {a['path']: a for a in unique_artifacts(required)}
    actual = {a['path']: a for a in unique_artifacts(gate['artifacts'])}
    if actual != expected:
        raise ValueError('provenance_missing: incomplete legacy gate artifact closure')


def fsync_directory(path: Path) -> None:
    fd = os.open(merge._absolute(path), os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(path: Path, payload) -> dict:
    """Immutable same-directory rename after fsync; preserve incomplete evidence.

    The bank flock excludes other producers. Neither a destination nor an orphan
    .preparing file is overwritten. Only a fully published binding may reconcile
    its missing reference; incomplete preparation requires explicit review.
    """
    path = merge._absolute(path)
    pending = path.with_name(path.name + '.preparing')
    if path.exists() or pending.exists():
        raise ValueError('provenance_missing: incomplete or existing immutable gate artifact')
    merge._write(pending, payload)
    if path.exists():
        raise ValueError('ambiguous gate writer; producer flock required')
    os.rename(pending, path)
    fsync_directory(path.parent)
    return artifact(path)


def freeze(path: Path, value: dict) -> dict:
    merge.canonical_hash(value)
    return atomic_write(path, merge._artifact_chunks('card', value))


def seal(value: dict, key: str) -> dict:
    return dict(value, **{key: merge.canonical_hash(value)})


def load_sealed(path: Path, key: str, *, expected=None) -> dict:
    # Check raw bytes as well as self-hash: even whitespace mutation is drift.
    value = merge._read_json(path, expected_hash=expected) if expected else merge._object(merge._read(path), 'gate evidence')
    if (not isinstance(value, dict) or value.get(key) != merge.canonical_hash(
            {k: v for k, v in value.items() if k != key})):
        raise ValueError('provenance_missing: gate evidence hash mismatch')
    if merge._file_info(path) != merge._payload_info(merge._artifact_chunks('card', value)):
        raise ValueError('gate evidence raw bytes changed')
    return value


def publish_reference(path: Path, value: dict) -> dict:
    return freeze(path, seal(value, 'reference_sha256'))


def recipe(adapter, *, interpreter=None) -> dict:
    from hindsight_offline_v2_rebuild import eval_environment
    resolved = shutil.which(adapter.python)
    if resolved is None:
        raise ValueError('gate interpreter unavailable')
    # Bind the executed binary, including a symlink's current target. The SHA
    # is captured once and verified with the full artifact closure in check().
    executed = Path(resolved).resolve(strict=True)
    if interpreter is None:
        interpreter = artifact(executed)
    elif str(executed) != interpreter['path']:
        raise ValueError('gate executed interpreter target changed')
    return {'python': adapter.python, 'interpreter': interpreter,
            'script_dir': str(adapter.script_dir),
            'api': str(adapter.config.get('api') or adapter.config.get('api_url') or 'http://127.0.0.1:8888'),
            'raw_limit': str(adapter.config.get('raw_limit', 40)),
            'limit': str(adapter.config.get('limit', 10)),
            'top_k': str(adapter.config.get('top_k', 5)),
            'timeout': 3600, 'eval_environment': eval_environment({}),
            # Config can contain a DSN. Persist its hash, never credentials.
            'config_sha256': merge.canonical_hash(adapter.config)}


def commands(adapter, manifest: dict, directory: Path, benchmarks: dict, invocation: dict) -> dict:
    root, api = manifest['root'], invocation['api']
    delta_root = directory.parent / 'delta-cards'
    if (delta_root / 'observations_index.jsonl').exists():
        root = str(delta_root)
    result = {'conflict': [adapter.python, str(adapter.script_dir / SOURCES[0]),
        '--cards-root', root, '--offline-root', root, '--output-dir', str(directory / 'conflict'),
        '--api', api, '--bank', adapter.bank, '--json']}
    for label in ('generic', 'local'):
        common = [adapter.python, str(adapter.script_dir / 'hindsight_eval.py'),
            '--benchmark', benchmarks[label]['snapshot']['path'], '--api', api, '--bank', adapter.bank,
            '--raw-limit', invocation['raw_limit'], '--limit', invocation['limit'], '--top-k', invocation['top_k'],
            '--cards-root', root, '--json']
        for mode in ('baseline', 'cards'):
            step = label + '-' + mode
            result[step] = common + ['--no-local-cards' if mode == 'baseline' else '--use-local-cards',
                                    '--output-dir', str(directory / step)]
    return result


class GateAttempt:
    def __init__(self, adapter, manifest: dict, ledger, coverage: dict, scoped: dict, parse: dict, thresholds: dict,
                 *, recover: bool = True, candidate_fence=None):
        self.adapter, self.manifest, self.ledger = adapter, manifest, ledger
        self.recover = recover
        # Retain the actual caller's proof reference, not a cached copy. Capture
        # metadata from real strict reads once; never reparse the seed per child.
        self.scoped = scoped
        self.manifest_identity = merge.canonical_hash(manifest)
        if candidate_fence is None:
            with merge._validation_read_scope() as candidate_fence:
                _m, scope_rows, _, _cards, _r = adapter._scope_candidate(manifest, ledger)
                if adapter._scope_parse_proof(manifest, ledger, scope_rows) != parse:
                    raise ValueError('candidate parse proof differs from strict dependency closure')
        self.candidate_fence = candidate_fence
        self.root = merge._absolute(ledger.root) / 'publication' / manifest['manifest_sha256']
        self.directory = self.root / SLOT
        self.reference_path = self.root / 'gate-attempt-reference.json'
        self.binding_path = self.directory / 'attempt-inputs.json'
        if self.root.exists():
            # Inventory is used only to REJECT competing bindings, never to pick
            # a report/attempt. Old unbound UUID directories are not adopted.
            for child in self.root.iterdir():
                child = merge._absolute(child)
                if child != self.directory and child.is_dir() and any(
                        (child / name).exists() for name in ('attempt-inputs.json', 'attempt-inputs.json.preparing')):
                    raise ValueError('ambiguous gate input binding candidates; explicit review required')
        self.context = {'schema_version': SCHEMA, 'bank': adapter.bank, 'ledger_root': str(merge._absolute(ledger.root)),
            'manifest_sha256': manifest['manifest_sha256'], 'manifest_schema': manifest['schema_version'],
            'root': manifest['root'], 'parent_sha256': manifest['parent_sha256'],
            'candidate_manifest': artifact(Path(manifest['manifest_path'])),
            'coverage_sha256': merge.canonical_hash(coverage), 'scoped_sha256': merge.canonical_hash(scoped),
            'parse_sha256': merge.canonical_hash(parse), 'thresholds': dict(thresholds),
            'recipe': recipe(adapter), 'sources': [artifact(adapter.script_dir / name) for name in SOURCES]}
        self.source_paths = {label: merge._absolute(adapter.config[label + '_benchmark']) for label in ('generic', 'local')}
        if self.reference_path.exists():
            ref = load_sealed(self.reference_path, 'reference_sha256')
            if (ref.get('schema_version') != REFERENCE_SCHEMA or ref.get('bank') != adapter.bank
                    or ref.get('manifest_sha256') != manifest['manifest_sha256']
                    or ref.get('attempt_root') != str(self.directory)
                    or not isinstance(ref.get('input_binding'), dict)
                    or ref['input_binding'].get('path') != str(self.binding_path)):
                raise ValueError('provenance_missing: unsupported or unbound gate attempt reference')
            self.binding = load_sealed(self.binding_path, 'binding_sha256', expected=merge._hash(
                ref['input_binding'].get('sha256'), 'gate input binding'))
        elif not recover:
            raise ValueError('provenance_missing: independent validation requires the saved gate attempt reference')
        elif self.directory.exists():
            if not self.binding_path.exists():
                raise ValueError('provenance_missing: incomplete gate input preparation; explicit review required')
            # A crash between binding and reference is recoverable only after all
            # original source/config/candidate bytes still match, before queries.
            self.binding = load_sealed(self.binding_path, 'binding_sha256')
        else:
            if self.reference_path.with_name(self.reference_path.name + '.preparing').exists():
                raise ValueError('provenance_missing: orphan gate reference; explicit review required')
            self.directory.mkdir(parents=True, mode=0o700)
            fsync_directory(self.root)
            fsync_directory(self.root.parent)
            fsync_directory(merge._absolute(ledger.root))
            self.binding = self._prepare(coverage, scoped, parse)
        self.binding_artifact = artifact(self.binding_path)
        self.check()
        if not self.reference_path.exists():
            publish_reference(self.reference_path, {'schema_version': REFERENCE_SCHEMA, 'bank': adapter.bank,
                'manifest_sha256': manifest['manifest_sha256'], 'attempt_root': str(self.directory),
                'input_binding': self.binding_artifact})
        self.reference_artifact = artifact(self.reference_path)
        self.artifacts = [*self.binding['artifacts'], self.binding_artifact, self.reference_artifact]
        self.completed = {}
        # Verify ALL completed evidence before launching ANY missing step.
        for step in STEPS:
            path = self.directory / (step + '.completion.json')
            if path.exists():
                self.completed[step] = self._completion(step)
            elif (path.with_name(path.name + '.preparing').exists()
                  or (self.directory / (step + '.json')).exists()):
                raise ValueError('provenance_missing: orphan child completion; explicit review required')
        self.check()

    def _prepare(self, coverage, scoped, parse) -> dict:
        from hindsight_incremental_publication import _benchmark_ids
        artifacts = [freeze(self.directory / name, value) for name, value in (
            ('coverage.json', coverage), ('candidate-parse-proof.json', parse), ('scoped-proof.json', scoped))]
        benchmarks = {}
        for label, path in self.source_paths.items():
            source = artifact(path)
            snapshot = self.directory / (label + '-benchmark.jsonl')
            snapshot_artifact = atomic_write(snapshot, merge._read(path))
            if snapshot_artifact['sha256'] != source['sha256'] or artifact(path) != source:
                raise ValueError('benchmark changed during durable input preparation')
            benchmarks[label] = {'source': source, 'snapshot': snapshot_artifact, 'ids': _benchmark_ids(snapshot)}
            artifacts.extend((source, snapshot_artifact))
        artifacts.extend(self.context['sources'])
        artifacts.extend((self.context['recipe']['interpreter'], self.context['candidate_manifest']))
        value = dict(self.context, benchmarks=benchmarks, artifacts=unique_artifacts(artifacts),
                     steps=commands(self.adapter, self.manifest, self.directory, benchmarks, self.context['recipe']))
        value = seal(value, 'binding_sha256')
        freeze(self.binding_path, value)
        return value

    def check(self) -> None:
        from hindsight_incremental_publication import _benchmark_ids
        self.candidate_fence.check()
        coverage = self.adapter._coverage(self.manifest, self.ledger)
        self.adapter._proof(self.manifest, self.scoped)
        if (merge.canonical_hash(coverage) != self.binding['coverage_sha256']
                or merge.canonical_hash(self.scoped) != self.binding['scoped_sha256']
                or ('batches' in self.scoped and self.scoped['batches'] != [
                    b['consolidated'].get('scoped_proof') for b in coverage['batches']])
                or merge.canonical_hash(self.manifest) != self.manifest_identity
                or self.adapter.bank != self.context['bank']
                or str(merge._absolute(self.ledger.root)) != self.context['ledger_root']):
            raise ValueError('live gate candidate/coverage/scoped binding changed before child query')
        if self.context['recipe'] != recipe(self.adapter, interpreter=self.context['recipe']['interpreter']):
            raise ValueError('gate interpreter/config recipe changed before child query')
        if any(self.binding.get(k) != v for k, v in self.context.items()):
            raise ValueError('gate bound candidate/coverage/scoped/config/source inputs changed')
        if set(self.binding.get('benchmarks', {})) != {'generic', 'local'}:
            raise ValueError('provenance_missing: incomplete frozen benchmarks')
        expected_paths = {str(self.directory / name) for name in (
            'coverage.json', 'candidate-parse-proof.json', 'scoped-proof.json')}
        for label, benchmark in self.binding['benchmarks'].items():
            source, snapshot = benchmark['source'], benchmark['snapshot']
            if (source['path'] != str(self.source_paths[label])
                    or snapshot['path'] != str(self.directory / (label + '-benchmark.jsonl'))
                    or source['sha256'] != snapshot['sha256']
                    or benchmark['ids'] != _benchmark_ids(Path(snapshot['path']))):
                raise ValueError('frozen benchmark source/snapshot binding mismatch')
            expected_paths.update((source['path'], snapshot['path']))
        expected_paths.update(s['path'] for s in self.context['sources'])
        expected_paths.update((self.context['recipe']['interpreter']['path'], self.context['candidate_manifest']['path']))
        if (set(a['path'] for a in self.binding['artifacts']) != expected_paths
                or len(self.binding['artifacts']) != len(expected_paths)
                or self.binding['steps'] != commands(self.adapter, self.manifest, self.directory,
                    self.binding['benchmarks'], self.context['recipe'])):
            raise ValueError('provenance_missing: incomplete gate input/command binding')
        # Read each exact input/completion/stdout/disk dependency once per fence.
        # No recovery routine is called: it could create stdout or append proof.
        dependencies = [*self.binding['artifacts'], *getattr(self, 'artifacts', ())]
        dependencies.extend(getattr(self, name) for name in ('binding_artifact', 'reference_artifact')
                            if hasattr(self, name))
        for item in unique_artifacts(dependencies):
            if artifact(Path(item['path'])) != item:
                raise ValueError('gate frozen input/completed report bytes changed before child query')
        for name, key in (('coverage.json', 'coverage_sha256'), ('scoped-proof.json', 'scoped_sha256'),
                          ('candidate-parse-proof.json', 'parse_sha256')):
            value = merge._object(merge._read(self.directory / name), 'gate bound proof')
            if merge.canonical_hash(value) != self.binding[key]:
                raise ValueError('gate bound proof mismatch')
        self.candidate_fence.check()

    def _completion(self, step: str) -> dict:
        path = self.directory / (step + '.completion.json')
        value = load_sealed(path, 'completion_sha256')
        if (value.get('schema_version') != COMPLETION_SCHEMA or value.get('step') != step
                or value.get('input_binding') != self.binding_artifact
                or value.get('command') != self.binding['steps'][step]
                or value.get('stdout', {}).get('path') != str(self.directory / (step + '.json'))
                or not isinstance(value.get('stdout_json'), str)):
            raise ValueError('provenance_missing: child completion input/command mismatch')
        payload = value['stdout_json'].encode('utf-8')
        if value['stdout']['sha256'] != hashlib.sha256(payload).hexdigest():
            raise ValueError('child stdout completion bytes mismatch')
        result = merge._object(payload, 'completed child stdout')
        disk = value.get('disk_report')
        if (not isinstance(disk, dict) or result.get('json_path') != disk.get('path')
                or not merge._absolute(disk['path']).is_relative_to(self.directory / step)
                or Path(disk['path']).name == 'latest.json'
                or artifact(Path(disk['path'])) != disk):
            raise ValueError('provenance_missing: exact named child disk report mismatch')
        report = merge._object(merge._read(Path(disk['path'])), 'completed child disk report')
        if not report_outputs_match(result, report):
            raise ValueError('command stdout/report artifact mismatch')
        stdout = Path(value['stdout']['path'])
        if not stdout.exists():
            if not self.recover:
                raise ValueError('provenance_missing: independent validation cannot repair missing stdout')
            atomic_write(stdout, payload)  # Reconstruct only exact persisted stdout, never a report.
        if artifact(stdout) != value['stdout']:
            raise ValueError('completed child stdout bytes changed')
        self.artifacts.extend((artifact(path), value['stdout'], disk))
        return result

    def run(self, step: str) -> dict:
        if step in self.completed:
            return self.completed[step]
        self.check()
        result = self.adapter.run_json(self.binding['steps'][step], timeout=3600)
        if not isinstance(result, dict) or not result.get('json_path'):
            raise ValueError('provenance_missing: gate child requires exact named disk report')
        produced = merge._absolute(result['json_path'])
        if not produced.is_relative_to(self.directory / step) or produced.name == 'latest.json':
            raise ValueError('gate command returned an old report path')
        # Only newly returned child output is made private and durable here;
        # recovery never chmods or rewrites an old report.
        with merge._open_read(produced) as stream:
            os.fchmod(stream.fileno(), 0o600)
            os.fsync(stream.fileno())
        fsync_directory(produced.parent)
        # Capture stdout + disk SHA + binding BEFORE validation. The completion is
        # written first and contains the exact serialized stdout, so a crash before
        # stdout materialization cannot lose already completed expensive work.
        stdout_json = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=True) + '\n'
        value = seal({'schema_version': COMPLETION_SCHEMA, 'step': step,
            'input_binding': self.binding_artifact, 'command': self.binding['steps'][step],
            'disk_report': artifact(produced), 'stdout_json': stdout_json,
            'stdout': {'path': str(self.directory / (step + '.json')),
                       'sha256': hashlib.sha256(stdout_json.encode('utf-8')).hexdigest()}}, 'completion_sha256')
        freeze(self.directory / (step + '.completion.json'), value)
        self.completed[step] = self._completion(step)
        # Preserve completed expensive output, then detect drift that happened
        # during the child before permitting any further query or passing gate.
        self.check()
        return self.completed[step]
