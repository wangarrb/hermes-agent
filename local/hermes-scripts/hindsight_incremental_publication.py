#!/usr/bin/env python3
"""Frozen gate and delta-only canonical publication; never rewrite opaque seed.

The caller holds the bank ledger lock. SQL export/readback are repeatable-read,
read-only. Only new/changed typed nodes enter the optional in-place publisher path.
Exact seed/typed-parent row locks plus a precommit full-state check fence writes;
opaque source rows are never rewritten and incremental publication never deletes.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import runpy
import sys
import uuid

import hindsight_incremental_merge as merge
from hindsight_incremental_progress import active_reporter, reporter_context, span, timed
from hindsight_offline_v2_gate import evaluate_pair, evaluate_conflict_audit
from hindsight_offline_v2_rebuild import eval_environment, run_json as default_run_json

INCREMENTAL_PREFIX = 'hermes-offline-canonical::incremental::'
CHECKS = ('schema', 'coverage', 'conflict', 'generic_eval', 'local_eval', 'scoped_drain')
THRESHOLDS: dict = dict(min_term_recall_delta=0.001, min_layer_hit_delta=0,
    min_score_delta=0.0, max_case_term_regressions=2, case_regression_tolerance=0.05)


def _bytes_hash(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact(path: Path) -> dict:
    return {'path': str(merge._absolute(path)), 'sha256': merge._file_info(path)['sha256']}


def _freeze(path: Path, value: dict) -> dict:
    merge.canonical_hash(value)  # Reject invalid data before creating evidence.
    merge._write(path, merge._artifact_chunks('card', value))
    return _artifact(path)


def _rows_from_cursor(cur) -> list[dict]:
    with span('rows_fetch') as counts:
        rows = [r['row'] if isinstance(r, dict) else r[0] for r in cur.fetchall()]
        counts.update(row_count=len(rows))
    # JSONB conversion in SQL, not default=str, defines the immutable representation.
    with span('rows_hash', row_count=len(rows)):
        merge.canonical_hash(rows)
    return sorted(rows, key=lambda r: str(r['id']))


@timed('read_remote')
def _read_remote(conn, bank: str, document_ids: list[str] | None = None, *, for_update=False) -> tuple[list, list]:
    # NULL embeddings stay explicit. to_jsonb preserves arbitrary future columns,
    # arrays, timestamps, unknown metadata and complete vector text without parsing.
    if document_ids is None:
        doc_where = 'id LIKE %s AND id NOT LIKE %s'
        unit_where = 'document_id LIKE %s AND document_id NOT LIKE %s'
        params = (bank, 'hermes-offline-canonical::%', INCREMENTAL_PREFIX + '%')
    else:
        doc_where, unit_where = 'id = ANY(%s)', 'document_id = ANY(%s)'
        params = (bank, document_ids)
    lock = ' FOR UPDATE' if for_update else ''
    with conn.cursor() as cur:
        with span('remote_documents_fetch') as counts:
            cur.execute(f'SELECT to_jsonb(d) AS row FROM documents d '
                        f'WHERE bank_id=%s AND {doc_where} ORDER BY id{lock}', params)
            docs = _rows_from_cursor(cur)
            counts.update(document_count=len(docs))
        with span('remote_units_fetch') as counts:
            cur.execute("SELECT (to_jsonb(m) - 'embedding') || jsonb_build_object("
                        "'embedding', embedding::text, 'embedding_dimension', vector_dims(embedding)) AS row "
                        f'FROM memory_units m WHERE bank_id=%s AND {unit_where} ORDER BY id{lock}', params)
            units = _rows_from_cursor(cur)
            counts.update(unit_count=len(units))
    return docs, units


@timed('remote_identity')
def _remote_identity(bank: str, docs: list, units: list) -> dict:
    ids = [d['id'] for d in docs]
    if len(set(ids)) != len(ids) or len({u['id'] for u in units}) != len(units):
        raise ValueError('duplicate authoritative seed IDs')
    if any(d.get('bank_id') != bank for d in docs) or any(
            u.get('bank_id') != bank or u.get('document_id') not in ids for u in units):
        raise ValueError('authoritative seed bank/document mismatch')
    if any(not isinstance(d['id'], str) or not d['id'].startswith('hermes-offline-canonical::')
            or d['id'].startswith(INCREMENTAL_PREFIX) for d in docs):
        raise ValueError('seed is not the old canonical namespace')
    if any(not isinstance(d.get('original_text'), str) or d.get('content_hash') !=
           _bytes_hash(d['original_text'].encode('utf-8')) for d in docs):
        raise ValueError('authoritative seed document content hash mismatch')
    if any(not isinstance(u.get('text'), str) or u.get('fact_type') != 'observation' or not _valid_vector(u) for u in units):
        raise ValueError('seed requires non-NULL 1024-dimensional observations')
    with span('remote_documents_hash', document_count=len(docs)):
        documents_hash = merge.canonical_hash(docs)
    with span('remote_units_hash', unit_count=len(units)):
        units_hash = merge.canonical_hash(units)
    with span('remote_combined_hash', document_count=len(docs), unit_count=len(units)):
        combined_hash = merge.canonical_hash({'documents': docs, 'memory_units': units})
    return {'bank': bank, 'document_ids': ids, 'document_count': len(docs),
            'observation_count': len(units), 'documents_sha256': documents_hash,
            'observations_sha256': units_hash, 'row_fingerprint_sha256': combined_hash}


def _valid_vector(row: dict) -> bool:
    return row.get('embedding') is not None and type(row.get('embedding_dimension')) is int and row['embedding_dimension'] == 1024


def _delta_cards(rows: list[dict], publisher) -> list[dict]:
    cards = []
    for row in sorted(rows, key=merge.observation_id):
        obs = merge.publication_observation(row)
        # Retain the full 256-bit node identity in the document topic component,
        # not merely the old publisher's twelve-character suffix.
        digest = merge.observation_id(row).split(':', 1)[1]
        card = {'card_id': digest, 'schema_version': 'hindsight-incremental-node/1',
                'scope': 'incremental', 'topic': digest, 'generated_at': 'sealed-local-state',
                'executive_summary': [], 'entity_aliases': [], 'evidence_index': [],
                'canonical_observations': [obs]}
        if not publisher.card_document_id(card).startswith(INCREMENTAL_PREFIX):
            raise ValueError('unsafe incremental document namespace')
        cards.append(card)
    return cards


def candidate_parse_proof(root: str | Path, script_dir: Path | None = None) -> dict:
    """Evidence from both real consumers of the exact sealed, exhaustive index."""
    manifest, _, rows = merge.load_manifest_artifacts(Path(root))
    if rows is None:
        raise ValueError('candidate parse proof requires a complete index')
    scripts = script_dir or Path(__file__).parent
    conflict = runpy.run_path(str(scripts / 'hindsight_conflict_audit.py'))['load_observations'](Path(root))
    layered = runpy.run_path(str(scripts / 'hindsight_recall_layered.py'))['load_local_cards'](root)
    def view_id(row):
        return row['legacy_row']['id'] if row.get('kind') == 'opaque_legacy' else merge.observation_id(row)
    ids = sorted(view_id(r) for r in rows)
    for actual, key in ((conflict, 'insight'), (layered, 'text')):
        if sorted(r['id'] for r in actual) != ids:
            raise ValueError('candidate consumer omitted or duplicated declared rows')
        by_id = {r['id']: r for r in actual}
        for row in rows:
            expected_text = row['legacy_row']['text'] if row.get('kind') == 'opaque_legacy' else row['insight']
            got = by_id[view_id(row)][key]
            if (got != expected_text if row.get('kind') == 'opaque_legacy' or key == 'insight' else not got.startswith(expected_text)):
                raise ValueError('candidate consumer changed declared observation text')
    return {'manifest_sha256': manifest['manifest_sha256'], 'observation_count': len(rows),
        'opaque_count': sum(r.get('kind') == 'opaque_legacy' for r in rows),
        'typed_count': sum(r.get('kind') != 'opaque_legacy' for r in rows),
        'ids_sha256': merge.canonical_hash(ids), 'parse_errors': 0,
        'conflict_view_sha256': merge.canonical_hash(conflict), 'eval_view_sha256': merge.canonical_hash(layered)}


def delta_cards_root(ledger_root: Path, manifest: dict) -> Path:
    """Stable delta-cards root beside the frozen gate attempt directory."""
    return merge._absolute(ledger_root) / 'publication' / manifest['manifest_sha256'] / 'delta-cards'


def freeze_delta_cards(ledger_root: Path, manifest: dict, rows: list[dict]) -> Path:
    """Write (or verify) the delta-cards root: the delta rows as the exact
    observations index the conflict and layered readers consume directly."""
    import os
    from hindsight_incremental_gate_attempts import fsync_directory
    root = delta_cards_root(ledger_root, manifest)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / 'observations_index.jsonl'
    payload = b''.join(merge._artifact_chunks('observations', rows))
    entry = next((f for f in manifest['files'] if f['path'] == 'delta_index.jsonl'), None)
    if entry is None:
        raise ValueError('delta cards root requires a sealed delta index')
    pointer = {'manifest_sha256': manifest['manifest_sha256'], 'delta_sha256': entry['sha256'],
               'delta_count': len(rows)}
    pointer_payload = json.dumps(pointer, ensure_ascii=False, sort_keys=True).encode('utf-8')
    for target, data in ((path, payload), (root / 'delta-manifest.json', pointer_payload)):
        if target.exists():
            if target.read_bytes() != data:
                raise ValueError('delta cards root differs from the sealed delta')
        else:
            tmp = target.with_name('.' + target.name + '.preparing')
            with open(tmp, 'wb') as handle:
                handle.write(data)
            os.replace(tmp, target)
    fsync_directory(root)
    return root


def delta_parse_proof(delta_root: Path, script_dir: Path | None = None) -> dict:
    """Exhaustive parse proof over the sealed delta root through both real
    consumers (conflict audit + layered local cards), never the full index.
    The root carries its own pointer (manifest_sha256/delta_sha256/count)."""
    root = merge._absolute(delta_root)
    pointer = merge._object(merge._read(root / 'delta-manifest.json'), 'delta manifest pointer')
    payload = (root / 'observations_index.jsonl').read_bytes()
    if (not isinstance(pointer, dict) or pointer.get('delta_sha256') != hashlib.sha256(payload).hexdigest()
            or type(pointer.get('delta_count')) is not int):
        raise ValueError('delta parse proof pointer mismatch')
    rows = [json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    if len(rows) != pointer['delta_count']:
        raise ValueError('delta parse proof count mismatch')
    scripts = script_dir or Path(__file__).parent
    conflict = runpy.run_path(str(scripts / 'hindsight_conflict_audit.py'))['load_observations'](root)
    layered = runpy.run_path(str(scripts / 'hindsight_recall_layered.py'))['load_local_cards'](root)
    ids = sorted(merge.observation_id(r) for r in rows)

    def view_row(value):
        # The raw delta may preserve a model's noncanonical id. Compare both
        # views using typed identity without changing the sealed source bytes.
        value = dict(value)
        value['id'] = merge.observation_id(value)
        return value

    conflict = [view_row(r) for r in conflict]
    if sorted(r['id'] for r in conflict) != ids:
        raise ValueError('delta consumer omitted or duplicated declared rows')
    by_id = {r['id']: r for r in conflict}
    for row in rows:
        if by_id[merge.observation_id(row)]['insight'] != row['insight']:
            raise ValueError('delta consumer changed declared observation text')
    # The layered view carries no raw identity fields; bind one-to-one by
    # exact declared text (visible text starts with the insight).
    if len(layered) != len(rows):
        raise ValueError('delta consumer omitted or duplicated declared rows')
    remaining = list(layered)
    for row in rows:
        expected = row['insight']
        for index, candidate in enumerate(remaining):
            got = candidate.get('text')
            if isinstance(got, str) and got.startswith(expected):
                remaining.pop(index)
                break
        else:
            raise ValueError('delta consumer changed declared observation text')
    return {'manifest_sha256': pointer.get('manifest_sha256'), 'delta_count': len(rows),
        'observation_count': len(rows),
        'delta_sha256': pointer['delta_sha256'], 'ids_sha256': merge.canonical_hash(ids), 'parse_errors': 0,
        'conflict_view_sha256': merge.canonical_hash(conflict), 'eval_view_sha256': merge.canonical_hash(layered)}


def _benchmark_ids(path: Path) -> list[str]:
    rows = [json.loads(line) for line in merge._read(path).decode('utf-8').splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    if not rows or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) or not r['id'] for r in rows):
        raise ValueError('benchmark requires explicit nonempty case IDs')
    ids = [r['id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate benchmark case IDs')
    return sorted(ids)


def _validate_eval(report: dict, ids: list[str]) -> None:
    summary, results = report.get('summary'), report.get('results')
    if (not isinstance(summary, dict) or type(summary.get('case_count')) is not int
            or not isinstance(results, list) or summary['case_count'] != len(ids)
            or len(results) != len(ids) or any(not isinstance(r, dict) or not isinstance(r.get('id'), str) for r in results)
            or sorted(r['id'] for r in results) != ids):
        raise ValueError('eval report lacks exact declared benchmark case coverage')
    def finite(value):
        return type(value) in (int, float) and math.isfinite(value)
    for metric in ('layered_avg_term_recall', 'layered_expected_layer_hits', 'layered_avg_score'):
        if not finite(summary.get(metric)):
            raise ValueError('eval report has missing or nonfinite metrics')
    for result in results:
        for mode in ('direct', 'layered'):
            metrics = result.get(mode)
            if not isinstance(metrics, dict) or any(not finite(metrics.get(k)) for k in
                    ('term_recall', 'expected_layer_hits_top_k', 'score')):
                raise ValueError('eval case has missing or nonfinite metrics')
    # Reproduce run_eval's ordered +=, not sum(): Python's compensated sum
    # can land on the other side of a rounding tie for the same case metrics.
    term_total, score_total, layer_hits = 0.0, 0.0, 0
    for result in results:
        term_total += result['layered']['term_recall']
        score_total += result['layered']['score']
        layer_hits += result['layered']['expected_layer_hits_top_k']
    aggregate = {
        'layered_avg_term_recall': round(term_total / len(ids), 3),
        'layered_avg_score': round(score_total / len(ids), 2),
        'layered_expected_layer_hits': layer_hits,
    }
    if any(summary[k] != value for k, value in aggregate.items()):
        raise ValueError('eval aggregate differs from exhaustive per-case metrics')


def _validate_eval_options(report: dict, top_k) -> None:
    # These fields are emitted by the real evaluator. Command provenance alone
    # does not establish that the report used observations or the requested k.
    summary = report.get('summary')
    if (report.get('include_hindsight_observations') is not True
            or not isinstance(summary, dict) or type(summary.get('top_k')) is not int
            or summary['top_k'] != int(top_k)):
        raise ValueError('eval report options contradict bound observations/top-k command')


def _validate_conflict_report(report: dict, bank: str, root: str, api: str, proof: dict) -> None:
    if (report.get('bank') != bank or report.get('cards_root') != root
            or report.get('offline_root') != root or not isinstance(report.get('cases'), list)
            or not isinstance(report.get('summary'), dict)
            or type(report.get('summary', {}).get('blocking_cases')) is not int
            or report['summary']['blocking_cases'] < 0
            or type(report.get('observation_count')) is not int
            or report.get('observation_count') != proof['observation_count']
            or report.get('api') != api
            or report.get('decision') not in {'pass', 'blocked_conflict_review_required'}
            or any(not isinstance(c, dict) or c.get('severity') not in {'OK', 'P0', 'P1', 'P2', 'P3'}
                   for c in report['cases']) or report.get('source_scan_errors')):
        raise ValueError('conflict report incomplete or candidate/bank mismatch')


class PublicationAdapter:
    """Concrete runtime adapter. Injection replaces only I/O in isolated tests.

    Config keys: dsn (or db_dsn), script_dir, publisher_python, api,
    generic_benchmark, local_benchmark, embedding_model/base_url/batch_size.
    Keys can be supplied under incremental.publication or publication as well.
    scoped_proof requires exact bank/cohort scope, ready=True and strict integer
    pending_in_scope=0, pending_operations=0 (bool/None are not zero).
    """
    def __init__(self, config: dict, bank: str, *, connect=None, run_json=None,
                 embeddings=None, execute_values=None):
        merge._name(bank, 'bank')
        self.bank = bank
        self.config = dict(config)
        self.config.update((config.get('incremental') or {}).get('publication') or {})
        self.config.update(config.get('publication') or {})
        # Never exec an alternate interpreter on import: absent SQL driver blocks
        # construction before any eval/embedding side effect.
        try:
            self.publisher = importlib.import_module('hindsight_offline_v2_publish')
        except ImportError as exc:
            raise RuntimeError('publisher SQL driver unavailable in runtime Python') from exc
        self.connect = connect or self.publisher.connect
        self.run_json = run_json or self._run_json
        self.embeddings = embeddings or self.publisher.maybe_embeddings
        self.execute_values = execute_values or self.publisher.execute_values
        self.dsn = self.config.get('dsn') or self.config.get('db_dsn') or self.publisher.DEFAULT_DB_DSN
        self.script_dir = merge._absolute(self.config.get('script_dir') or self.config.get('scripts_dir') or Path(__file__).parent)
        self.python = str(self.config.get('publisher_python') or sys.executable)

    def _run_json(self, command: list[str], **kwargs) -> dict:
        if Path(command[1]).name == 'hindsight_eval.py':
            # Arrival publication must retain the legacy eval transport policy;
            # the default recall deadline is shorter than legitimate graph search.
            kwargs['env'] = eval_environment(kwargs.get('env'))
            # The legacy evaluator hardcodes ~/.hermes/scripts for its layered
            # module. Bind its real globals to the configured adjacent module;
            # executing a worktree eval must not silently import installed code.
            bootstrap = """import pathlib,runpy,sys
path=pathlib.Path(sys.argv.pop(1))
sys.path.insert(0,str(path.parent))
module=runpy.run_path(str(path))
g=module['main'].__globals__
g['LAYERED_PATH']=path.parent/'hindsight_recall_layered.py'
if 'run_eval' in g:
    original=g['run_eval']
    def checked_eval(*args, **kwargs):
        from hindsight_incremental_publication import candidate_parse_proof, delta_parse_proof
        proof=None
        if kwargs.get('cards_root'):
            root=pathlib.Path(str(kwargs['cards_root']))
            proof=(candidate_parse_proof(kwargs['cards_root'], path.parent)
                   if (root/'manifest.json').exists() else delta_parse_proof(kwargs['cards_root'], path.parent))
        report=original(*args, **kwargs)
        report['candidate_parse_proof']=proof
        report['invocation']={'bank':kwargs['bank'],'api':kwargs['api'],
            'benchmark':sys.argv[sys.argv.index('--benchmark')+1]}
        return report
    g['run_eval']=checked_eval
module['main']()
"""
            command = [command[0], '-c', bootstrap, command[1], *command[2:]]
        return default_run_json(command, **kwargs)

    @contextmanager
    def _readonly(self):
        # Never include a DSN or potentially secret driver exception in reports.
        with span('readonly_connect'):
            try:
                conn = self.connect(self.dsn)
            except Exception:
                raise RuntimeError('canonical SQL connection unavailable') from None
        try:
            with span('readonly_transaction'):
                conn.set_session(readonly=True, isolation_level='REPEATABLE READ', autocommit=False)
                yield conn
        finally:
            conn.rollback()
            conn.close()

    def export_seed(self, path: Path) -> dict:
        """Explicit readonly authoritative old canonical export, not local latest."""
        path = merge._absolute(path)
        with self._readonly() as conn:
            docs, rows = _read_remote(conn, self.bank)
        if not docs or not rows:
            raise ValueError('authoritative canonical seed is empty; explicit review required')
        remote = _remote_identity(self.bank, docs, rows)
        provenance = {'bank': self.bank, 'seed_sha256': merge.canonical_hash(
            {'bank': self.bank, 'observations': rows, 'remote_seed': remote})}
        opaque = merge.import_legacy_seed(rows, bank=self.bank, provenance=provenance)
        payload = {'bank': self.bank, 'observations': opaque, 'provenance': provenance,
                   'remote_seed': remote, 'legacy_unknown': [
                       {'status': 'legacy_unknown', 'reason': 'published seed does not prove historical source coverage'}]}
        payload['export_sha256'] = merge.canonical_hash(payload)
        _freeze(path, payload)
        return payload

    @staticmethod
    def _has_delta(manifest: dict) -> bool:
        return any(isinstance(f, dict) and f.get('path') == 'delta_index.jsonl'
                   for f in manifest.get('files', []))

    def _delta_candidate(self, manifest: dict, ledger, remote=None):
        """Delta-scoped candidate: reads ONLY the sealed delta index. The full
        state and seed stay bound by manifest hashes as recorded evaluation
        evidence (user policy: hashes for evaluation, never a per-gate full
        re-read or a blocking obstacle)."""
        root = merge._absolute(manifest['root'])
        if not root.is_relative_to(merge._absolute(ledger.root)):
            raise ValueError('candidate outside private ledger root')
        with span('delta_load') as counts:
            bound, delta = merge.load_delta_artifacts(root, bank=self.bank)
            counts.update(row_count=len(delta))
        if bound != manifest:
            raise ValueError('candidate differs from exact frozen manifest')
        # Manifest-level ancestry walk (94KB manifests; never parent indexes).
        ancestor = manifest['parent']
        while ancestor.get('schema_version') == merge.SCHEMA:
            parent_root = merge._absolute(ancestor['root'])
            if not parent_root.is_relative_to(merge._absolute(ledger.root)):
                raise ValueError('ancestor outside private ledger root')
            merge._bound_parent(ancestor, self.bank)
            stored = merge._object(merge._read(parent_root / 'manifest.json'), 'ancestor manifest')
            if stored != ancestor:
                raise ValueError('ancestor manifest differs from the bound parent')
            ancestor = ancestor['parent']
        # The chain must root in THIS ledger's authoritative initial export.
        # The ledger manifest carries the seed provenance; no seed artifact read.
        seed = ledger.initialization_manifest().get('seed') or {}
        if seed.get('bank') != self.bank:
            raise ValueError('initial seed bank mismatch')
        if ancestor != seed.get('provenance'):
            raise ValueError('parent seed provenance mismatch')
        cards = _delta_cards(delta, self.publisher)
        targets = [self.publisher.card_document_id(c) for c in cards]
        if len(set(targets)) != len(targets):
            raise ValueError('incremental target overlaps another node')
        if remote is not None and set(targets) & set(remote['document_ids']):
            raise ValueError('incremental target overlaps legacy seed or another node')
        return manifest, delta, None, cards, remote

    def _scope_candidate(self, manifest: dict, ledger, remote=None):
        """Delta scope when the generation seals a delta index; legacy full
        candidate otherwise."""
        if self._has_delta(manifest):
            return self._delta_candidate(manifest, ledger, remote)
        return self._candidate(manifest, ledger)

    def _scope_parse_proof(self, manifest: dict, ledger, rows: list[dict] | None = None) -> dict:
        """Parse proof over the exact scope the gate reads: delta rows when a
        sealed delta exists, otherwise the full candidate index."""
        if self._has_delta(manifest):
            if rows is None:
                rows = merge.load_delta_artifacts(merge._absolute(manifest['root']), bank=self.bank)[1]
            root = freeze_delta_cards(ledger.root, manifest, rows)
            return delta_parse_proof(root, self.script_dir)
        return candidate_parse_proof(manifest['root'], self.script_dir)

    @timed('initial_seed')
    def _initial_seed(self, ledger) -> tuple[list[dict], dict, dict]:
        if ledger.bank != self.bank:
            raise ValueError('publication ledger bank mismatch')
        seed = ledger.initialization_manifest()['seed']
        if seed.get('bank') != self.bank:
            raise ValueError('initial seed bank mismatch')
        path = merge._absolute(seed['artifact_path'])
        if not path.is_relative_to(merge._absolute(ledger.root)):
            raise ValueError('seed artifact outside private ledger root')
        expected_hash = merge._hash(seed.get('artifact_hash'), 'initial seed artifact')
        with span('seed_read') as counts:
            value = merge._read_json(path, expected_hash=expected_hash)
            if isinstance(value, list):
                counts.update(row_count=len(value))
            elif isinstance(value, dict) and isinstance(value.get('observations'), list):
                counts.update(row_count=len(value['observations']))
        provenance, remote = seed['provenance'], seed['remote_seed']
        merge._parent(provenance, self.bank)
        if isinstance(value, dict):
            if value.get('bank') != self.bank or value.get('export_sha256') != merge.canonical_hash(
                    {k: v for k, v in value.items() if k != 'export_sha256'}):
                raise ValueError('seed export hash/bank mismatch')
            if value.get('provenance') != provenance or value.get('remote_seed') != remote:
                raise ValueError('initial seed export binding mismatch')
            value = value['observations']
        if not isinstance(value, list):
            raise ValueError('seed artifact requires opaque row array')
        if value and all(r.get('kind') == 'opaque_legacy' for r in value):
            rows = value
            raw = [r['legacy_row'] for r in value]
        else:
            raw = value
            rows = merge.import_legacy_seed(raw, bank=self.bank, provenance=provenance)
        if sorted(rows, key=merge.observation_id) != merge.import_legacy_seed(raw, bank=self.bank, provenance=provenance):
            raise ValueError('opaque seed wrappers differ from exact authoritative import')
        if (remote.get('bank') != self.bank or remote.get('observations_sha256') != merge.canonical_hash(
                sorted(raw, key=lambda r: str(r['id']))) or remote.get('observation_count') != len(raw)
                or len(set(remote.get('document_ids', []))) != remote.get('document_count')):
            raise ValueError('remote seed fingerprint/coverage mismatch')
        if provenance['seed_sha256'] != merge.canonical_hash(
                {'bank': self.bank, 'observations': sorted(raw, key=lambda r: str(r['id'])), 'remote_seed': remote}):
            raise ValueError('authoritative seed provenance hash mismatch')
        return rows, provenance, remote

    @timed('candidate')
    def _candidate(self, manifest: dict, ledger) -> tuple[dict, list, list, list, dict]:
        seed, provenance, remote = self._initial_seed(ledger)
        root = merge._absolute(manifest['root'])
        if not root.is_relative_to(merge._absolute(ledger.root)):
            raise ValueError('candidate outside private ledger root')
        with span('candidate_load') as counts:
            bound, _cards, rows = merge.load_manifest_artifacts(root, bank=self.bank)
            if rows is not None:
                counts.update(row_count=len(rows))
        if bound != manifest or rows is None:
            raise ValueError('candidate differs from exact frozen manifest')
        parent = manifest['parent']
        if parent.get('schema_version') == merge.SCHEMA:
            parent_root = merge._absolute(parent['root'])
            if not parent_root.is_relative_to(merge._absolute(ledger.root)):
                raise ValueError('parent outside private ledger root')
            with span('parent_load') as counts:
                parent_bound, _, previous = merge.load_manifest_artifacts(parent_root, bank=self.bank)
                if previous is not None:
                    counts.update(row_count=len(previous))
            if parent_bound != parent or previous is None:
                raise ValueError('parent state mismatch')
            # Bind every ancestor to this ledger's authoritative initial export.
            ancestor = parent
            while ancestor.get('schema_version') == merge.SCHEMA:
                if not merge._absolute(ancestor['root']).is_relative_to(merge._absolute(ledger.root)):
                    raise ValueError('ancestor outside private ledger root')
                ancestor = ancestor['parent']
            if ancestor != provenance:
                raise ValueError('parent seed provenance mismatch')
        else:
            if parent != provenance:
                raise ValueError('candidate initial parent mismatch')
            previous = seed
        old = {merge.observation_id(r): r for r in previous}
        new = {merge.observation_id(r): r for r in rows}
        if not old.keys() <= new.keys():
            raise ValueError('candidate drops sealed parent observations')
        baseline = {merge.observation_id(r): r for r in seed}
        for key, row in old.items():
            if row.get('kind') == 'opaque_legacy':
                if new[key] != row:
                    raise ValueError('candidate rewrites opaque legacy seed')
            elif merge.merge_observations([row], [new[key]]) != merge.merge_observations([], [new[key]]):
                raise ValueError('candidate drops sealed typed contributions')
        if {key: r for key, r in new.items() if r.get('kind') == 'opaque_legacy'} != baseline:
            raise ValueError('candidate opaque seed differs from authoritative export')
        delta = [r for key, r in new.items() if r.get('kind') != 'opaque_legacy' and
                 (key not in old or merge.observation_content_hash(r) != merge.observation_content_hash(old[key]))]
        cards = _delta_cards(delta, self.publisher)
        targets = [self.publisher.card_document_id(c) for c in cards]
        if len(set(targets)) != len(targets) or set(targets) & set(remote['document_ids']):
            raise ValueError('incremental target overlaps legacy seed or another node')
        return manifest, rows, previous, cards, remote

    def _coverage(self, manifest: dict, ledger) -> dict:
        batches = manifest['consumed_batch_ids']
        if not batches:
            raise ValueError('gate requires an actual newly consumed cohort')
        coverage = []
        for bid in batches:
            b = ledger.get_batch(bid)
            if (b.get('bank') != self.bank or b.get('batch_id') != bid
                    or b.get('phase') not in {'consolidated', 'locally_merged', 'gated', 'published'}
                    or not isinstance(b.get('receipts', {}).get('consolidated'), dict)):
                raise ValueError('candidate cohort lacks consolidated source coverage')
            payload_hash = merge._hash(b.get('payload_sha256') or b.get('payload_hash'), 'source payload')
            artifacts = []
            declared: list[tuple] = [(b.get('payload_path'), payload_hash)]
            generated = b['receipts'].get('generated') or {}
            declared.extend((generated.get(k + '_path'), generated.get(k + '_hash'))
                            for k in ('artifact', 'payload'))
            for artifact_path, expected_hash in declared:
                if artifact_path is None:
                    continue
                path = merge._absolute(artifact_path)
                if not path.is_relative_to(merge._absolute(ledger.root)):
                    raise ValueError('source coverage artifact outside private bank root')
                actual = _artifact(path)
                if actual['sha256'] != expected_hash:
                    raise ValueError('source coverage artifact hash mismatch')
                artifacts.append(actual)
            coverage.append({'batch_id': bid, 'bank': self.bank,
                'payload_sha256': payload_hash, 'artifacts': artifacts,
                'source_event_ids': b.get('source_event_ids', []),
                'consolidated': b['receipts']['consolidated'],
                'retained': b['receipts'].get('retained'),
                'generated': b['receipts'].get('generated')})
        return {'bank': self.bank, 'manifest_sha256': manifest['manifest_sha256'],
                'parent_sha256': manifest['parent_sha256'], 'batches': coverage}

    def _proof(self, manifest: dict, proof: dict) -> None:
        expected = sorted('arrival-batch:' + b for b in manifest['consumed_batch_ids'])
        def zero(value):
            return type(value) is int and value == 0
        if (not isinstance(proof, dict) or proof.get('bank') != self.bank
                or proof.get('ready') is not True or not zero(proof.get('pending_in_scope'))):
            raise ValueError('scoped native drain proof missing, stale or unknown')
        if 'batches' in proof:
            # ArrivalRuntime supplies the original per-batch native proofs, not a
            # historical bank backlog total. Preserve and validate each one.
            batches = proof['batches']
            scopes = [[tag] for tag in expected]
            if (not isinstance(batches, list) or len(batches) != len(scopes)
                    or proof.get('scopes') != scopes
                    or any(not isinstance(p, dict) or p.get('bank') != self.bank
                           or p.get('scope') != scope or p.get('ready') is not True
                           or not zero(p.get('pending_in_scope')) or not zero(p.get('failed_in_scope'))
                           or not isinstance(p.get('operation_ids'), list)
                           for p, scope in zip(batches, scopes))):
                raise ValueError('per-batch scoped native proof mismatch')
        elif proof.get('scope') != expected or not zero(proof.get('pending_operations')):
            raise ValueError('scoped operation drain proof missing, stale or unknown')

    def _run_report(self, command: list[str], path: Path, artifacts: list[dict]) -> dict:
        result = self.run_json(command, timeout=3600)
        if not isinstance(result, dict):
            raise ValueError('gate command did not return a report object')
        # If a script emits a disk report, bind its exact non-latest path too;
        # never accept a stale report outside this fresh attempt directory.
        if result.get('json_path'):
            produced = merge._absolute(result['json_path'])
            if not produced.is_relative_to(path.parent) or produced.name == 'latest.json':
                raise ValueError('gate command returned an old report path')
            disk = merge._object(merge._read(produced), 'produced gate report')
            if any(disk.get(k) != v for k, v in result.items() if k not in {'json_path', 'markdown_path'}):
                raise ValueError('command stdout/report artifact mismatch')
            artifacts.append(_artifact(produced))
        artifacts.append(_freeze(path, result))
        return result

    def gate(self, manifest: dict, *, ledger, scoped_proof: dict) -> dict:
        delta_scope = self._has_delta(manifest)
        remote = None
        if delta_scope:
            # One seed read for the legacy-seed overlap anchor; the full state
            # itself stays recorded evidence (never a blocking re-read).
            _seed_rows, _provenance, remote = self._initial_seed(ledger)
        with merge._validation_read_scope() as candidate_fence:
            manifest, scope_rows, _, cards, _ = self._scope_candidate(manifest, ledger, remote)
            parse_proof = self._scope_parse_proof(manifest, ledger, scope_rows)
        if not cards:
            raise ValueError('unchanged candidate requires parent no-op, not a fabricated gate')
        self._proof(manifest, scoped_proof)
        coverage = self._coverage(manifest, ledger)
        if 'batches' in scoped_proof and scoped_proof['batches'] != [
                b['consolidated'].get('scoped_proof') for b in coverage['batches']]:
            raise ValueError('scoped proof differs from exact ledger native receipts')
        from hindsight_incremental_gate_attempts import GateAttempt
        attempt = GateAttempt(self, manifest, ledger, coverage, scoped_proof, parse_proof, THRESHOLDS,
                              candidate_fence=candidate_fence)
        directory, artifacts = attempt.directory, attempt.artifacts
        benchmarks = {label: Path(value['snapshot']['path']) for label, value in attempt.binding['benchmarks'].items()}
        benchmark_ids = {label: value['ids'] for label, value in attempt.binding['benchmarks'].items()}
        root, api = manifest['root'], attempt.binding['recipe']['api']
        scope_root = str(delta_cards_root(ledger.root, manifest)) if delta_scope else root
        conflict = attempt.run('conflict')
        _validate_conflict_report(conflict, self.bank, scope_root, api, parse_proof)
        conflict_check = evaluate_conflict_audit(directory / 'conflict.json', required=True)
        pairs = []
        for label in ('generic', 'local'):
            bench = benchmarks[label]
            paths = []
            for mode in ('baseline', 'cards'):
                path = directory / f'{label}-{mode}.json'
                report = attempt.run(f'{label}-{mode}')
                expected = None if mode == 'baseline' else (str(delta_cards_root(ledger.root, manifest))
                                                             if delta_scope else root)
                if report.get('cards_root') != expected:
                    raise ValueError('eval report does not match exact candidate root')
                _validate_eval(report, benchmark_ids[label])
                _validate_eval_options(report, attempt.binding['recipe']['top_k'])
                if (report.get('invocation') != {'bank': self.bank, 'api': api, 'benchmark': str(bench)}
                        or report.get('candidate_parse_proof') != (parse_proof if mode == 'cards' else None)):
                    raise ValueError('eval invocation or exhaustive parse proof mismatch')
                paths.append(path)
            pairs.append(evaluate_pair(label, *paths, **THRESHOLDS))
        # Detect mutation during eval, including source receipts/benchmark files.
        attempt.check()
        with merge._validation_read_scope() as final_fence:
            _m2, final_rows, _, _c2, _r2 = self._scope_candidate(manifest, ledger, remote)
            final_parse = self._scope_parse_proof(manifest, ledger, final_rows)
        if coverage != self._coverage(manifest, ledger):
            raise ValueError('source coverage changed during gate')
        self._proof(manifest, scoped_proof)
        if parse_proof != final_parse:
            raise ValueError('candidate parse proof changed during gate')
        # Rebuild the recipe/source/coverage binding, not merely the remembered
        # in-memory hashes. This is also used independently before publication.
        final_attempt = GateAttempt(self, manifest, ledger, self._coverage(manifest, ledger),
                                    scoped_proof, parse_proof, THRESHOLDS, recover=False,
                                    candidate_fence=final_fence)
        if final_attempt.binding_artifact != attempt.binding_artifact:
            raise ValueError('gate attempt input binding changed during eval')
        for item in artifacts:
            if _artifact(Path(item['path'])) != item:
                raise ValueError('gate input/report changed during eval')
        checks = dict(schema=True, coverage=True, conflict=conflict_check['passed'],
                      generic_eval=pairs[0]['passed'], local_eval=pairs[1]['passed'], scoped_drain=True)
        delta_entry = next((f for f in manifest['files'] if f['path'] == 'delta_index.jsonl'), None)
        receipt = {'schema_version': 'hindsight-incremental-gate/1', 'bank': self.bank,
            'gate_scope': 'delta' if delta_scope else 'full',
            'seed_evidence': ({'observations_sha256': manifest.get('observations_sha256'),
                               'observation_count': manifest.get('observation_count'),
                               'parent_sha256': manifest.get('parent_sha256'),
                               'delta_sha256': delta_entry['sha256'] if delta_entry else None,
                               'delta_count': manifest.get('delta_count'),
                               'policy': 'recorded_not_blocking'} if delta_scope else None),
            'manifest_sha256': manifest['manifest_sha256'], 'root': root,
            'parent_sha256': manifest['parent_sha256'], 'coverage': coverage,
            'scoped_proof': copy.deepcopy(scoped_proof), 'artifacts': artifacts,
            'pairs': pairs, 'thresholds': copy.deepcopy(THRESHOLDS), 'conflict_check': conflict_check,
            'candidate_parse_proof': parse_proof, 'benchmark_ids': benchmark_ids,
            'input_binding': attempt.binding_artifact, 'attempt_reference': attempt.reference_artifact,
            'checks': checks, 'decision': 'eligible_for_local_proposal' if all(checks.values()) else 'blocked_keep_local_only',
            'gate_path': str(directory / 'gate.json')}
        receipt['gate_sha256'] = merge.canonical_hash(receipt)
        if Path(receipt['gate_path']).exists():
            if merge._object(merge._read(Path(receipt['gate_path'])), 'frozen gate') != receipt:
                raise ValueError('existing immutable gate differs from recovered evidence')
        else:
            from hindsight_incremental_gate_attempts import freeze
            freeze(Path(receipt['gate_path']), receipt)
        if receipt['decision'] == 'eligible_for_local_proposal':
            self._validate_gate(manifest, receipt, ledger)
        return receipt

    @timed('validate_gate')
    def _validate_gate(self, manifest: dict, gate: dict, ledger) -> None:
        if (gate.get('bank') != self.bank or gate.get('manifest_sha256') != manifest['manifest_sha256']
                or gate.get('root') != manifest['root'] or gate.get('parent_sha256') != manifest['parent_sha256']
                or gate.get('decision') != 'eligible_for_local_proposal'
                or any(gate.get('checks', {}).get(k) is not True for k in CHECKS)
                or gate.get('thresholds') != THRESHOLDS
                or gate.get('gate_sha256') != merge.canonical_hash({k: v for k, v in gate.items() if k != 'gate_sha256'})):
            raise ValueError('gate is blocked, forged, stale or bound to another candidate')
        from hindsight_incremental_gate_attempts import GateAttempt, STEPS, requires_binding, unique_artifacts, validate_legacy_closure
        path = merge._absolute(gate['gate_path'])
        if (not path.is_relative_to(merge._absolute(ledger.root) / 'publication' / manifest['manifest_sha256'])
                or path.is_relative_to(Path(manifest['root']))
                or merge._object(merge._read(path), 'frozen gate') != gate):
            raise ValueError('frozen gate artifact mismatch')
        for item in unique_artifacts(gate['artifacts']):
            if _artifact(Path(item['path'])) != item:
                raise ValueError('frozen gate dependency hash mismatch')
        if gate['coverage'] != self._coverage(manifest, ledger):
            raise ValueError('gate source coverage mismatch')
        self._proof(manifest, gate['scoped_proof'])
        if 'batches' in gate['scoped_proof'] and gate['scoped_proof']['batches'] != [
                b['consolidated'].get('scoped_proof') for b in gate['coverage']['batches']]:
            raise ValueError('frozen native proofs differ from ledger receipts')
        if [p.get('label') for p in gate['pairs']] != ['generic', 'local']:
            raise ValueError('fresh generic/local eval pairs both required')
        delta_scope = self._has_delta(manifest)
        expected_cards = str(delta_cards_root(ledger.root, manifest)) if delta_scope else manifest['root']
        with merge._validation_read_scope() as candidate_fence:
            _m, scope_rows, _, _c, _r = self._scope_candidate(manifest, ledger)
            proof = self._scope_parse_proof(manifest, ledger, scope_rows)
        if gate.get('candidate_parse_proof') != proof:
            raise ValueError('frozen candidate parse proof mismatch')
        benchmarks = {label: merge._absolute(self.config[label + '_benchmark']) for label in ('generic', 'local')}
        top_k = self.config.get('top_k', 5)
        if requires_binding(path, gate, ledger, manifest):
            attempt = GateAttempt(self, manifest, ledger, self._coverage(manifest, ledger),
                                  gate['scoped_proof'], proof, THRESHOLDS, recover=False,
                                  candidate_fence=candidate_fence)
            if (gate.get('input_binding') != attempt.binding_artifact
                    or gate.get('attempt_reference') != attempt.reference_artifact
                    or path != attempt.directory / 'gate.json'
                    or set(attempt.completed) != set(STEPS) or gate['artifacts'] != attempt.artifacts):
                raise ValueError('frozen gate attempt completion/input binding mismatch')
            benchmarks = {label: Path(value['snapshot']['path']) for label, value in attempt.binding['benchmarks'].items()}
            top_k = attempt.binding['recipe']['top_k']
        else:
            # Genuine completed legacy gates keep their original hash-bound
            # paths; never reconstruct missing provenance from current reports.
            validate_legacy_closure(self, gate, path.parent, benchmarks)
        api = str(self.config.get('api') or self.config.get('api_url') or 'http://127.0.0.1:8888')
        for pair in gate['pairs']:
            ids = _benchmark_ids(benchmarks[pair['label']])
            if gate.get('benchmark_ids', {}).get(pair['label']) != ids:
                raise ValueError('frozen benchmark case IDs mismatch')
            for mode in ('baseline', 'cards'):
                eval_path = pair[mode + '_eval']
                if eval_path != str(path.parent / (pair['label'] + '-' + mode + '.json')):
                    raise ValueError('frozen eval path differs from exact attempt output')
                report = merge._object(merge._read(Path(eval_path)), 'frozen eval')
                _validate_eval(report, ids)
                _validate_eval_options(report, top_k)
                benchmark = str(benchmarks[pair['label']])
                if (report.get('cards_root') != (expected_cards if mode == 'cards' else None)
                        or report.get('invocation') != {'bank': self.bank, 'api': api, 'benchmark': benchmark}
                        or report.get('candidate_parse_proof') != (proof if mode == 'cards' else None)):
                    raise ValueError('frozen eval invocation/parse proof mismatch')
            actual = evaluate_pair(pair['label'], Path(pair['baseline_eval']), Path(pair['cards_eval']), **THRESHOLDS)
            if actual != pair or actual['passed'] is not True or actual['cards_root'] != expected_cards:
                raise ValueError('frozen eval pair no longer passes original thresholds')
        conflict_path = path.parent / 'conflict.json'
        if gate['conflict_check']['audit_path'] != str(conflict_path):
            raise ValueError('frozen conflict path differs from exact attempt output')
        _validate_conflict_report(merge._object(merge._read(conflict_path), 'frozen conflict'),
                                  self.bank, expected_cards, api, proof)
        if evaluate_conflict_audit(conflict_path, required=True) != gate['conflict_check']:
            raise ValueError('frozen conflict audit mismatch')
        candidate_fence.check()
        for item in unique_artifacts(gate['artifacts']):
            if _artifact(Path(item['path'])) != item:
                raise ValueError('frozen gate dependency hash mismatch')

    @timed('seed_preserved')
    def _seed_preserved(self, conn, remote: dict, *, for_update=False) -> bool:
        docs, units = _read_remote(conn, self.bank, remote['document_ids'], for_update=for_update)
        try:
            return _remote_identity(self.bank, docs, units) == remote
        except ValueError:
            return False

    def _state_path(self, manifest, ledger):
        return merge._absolute(ledger.root) / 'publication' / manifest['manifest_sha256'] / 'remote-state.json'

    def _load_state(self, manifest, ledger):
        value = merge._object(merge._read(self._state_path(manifest, ledger)), 'frozen remote generation')
        if (value.get('bank') != self.bank or value.get('manifest_sha256') != manifest['manifest_sha256']
                or value.get('state_sha256') != merge.canonical_hash({k: v for k, v in value.items() if k != 'state_sha256'})):
            raise ValueError('frozen remote generation fingerprint mismatch')
        return value

    def _typed_match(self, card, doc, group, generation, uid):
        unit = group[0] if len(group) == 1 else {}
        obs = card['canonical_observations'][0]
        text = self.publisher.render_card_document(card, lossless=True)
        expected = self.publisher.memory_fact_metadata({'structured_observation': obs,
            'observation_content_sha256': merge.observation_content_hash(obs),
            'generation_sha256': generation, 'observation_id': obs['id']})
        return (doc.get('bank_id') == self.bank and doc.get('original_text') == text
            and doc.get('content_hash') == _bytes_hash(text.encode('utf-8'))
            and (doc.get('metadata') or {}).get('generation_sha256') == generation
            and unit.get('id') == uid and unit.get('bank_id') == self.bank
            and unit.get('fact_type') == 'observation' and _valid_vector(unit)
            and unit.get('text') == self.publisher.observation_text(obs, card, lossless=True)
            and all((unit.get('metadata') or {}).get(k) == v for k, v in expected.items()))

    @timed('readback')
    def _readback(self, conn, manifest, ledger, *, for_update=False):
        _, rows, previous, delta, remote = self._candidate(manifest, ledger)
        all_cards = _delta_cards([r for r in rows if r.get('kind') != 'opaque_legacy'], self.publisher)
        parent_cards = _delta_cards([r for r in previous if r.get('kind') != 'opaque_legacy'], self.publisher)
        parent_state = self._load_state(manifest['parent'], ledger) if parent_cards else {'documents': [], 'memory_units': []}
        parent_docs = {d['id']: d for d in parent_state['documents']}
        parent_units = {u['document_id']: u for u in parent_state['memory_units']}
        if (set(parent_docs) != {self.publisher.card_document_id(c) for c in parent_cards}
                or len(parent_units) != len(parent_cards)):
            raise ValueError('frozen parent remote generation coverage mismatch')
        for card in parent_cards:
            docid = self.publisher.card_document_id(card)
            u = parent_units.get(docid, {})
            revision = (u.get('metadata') or {}).get('generation_sha256')
            if not revision or not self._typed_match(card, parent_docs[docid], [u], revision, u.get('id')):
                raise ValueError('frozen parent remote generation semantic mismatch')
        legacy = self._seed_preserved(conn, remote, for_update=for_update)
        ids = [self.publisher.card_document_id(c) for c in all_cards]
        docs, units = _read_remote(conn, self.bank, ids, for_update=for_update)
        by_doc = {d['id']: d for d in docs}
        by_unit = {docid: [] for docid in ids}
        for unit in units:
            by_unit[unit['document_id']].append(unit)
        targets = [self.publisher.card_document_id(c) for c in delta]
        ready, verified, unchanged_ok = True, True, True
        unit_ids = {}
        for card in all_cards:
            docid = self.publisher.card_document_id(card)
            group, doc = by_unit[docid], by_doc.get(docid, {})
            prior = parent_units.get(docid)
            uid = prior['id'] if prior else self.publisher.semantic_memory_unit_id(self.bank, docid, card['canonical_observations'][0]['id'])
            if prior:
                unit_ids[docid] = uid
            matches_parent = prior is not None and doc == parent_docs[docid] and group == [prior]
            if docid not in targets:
                unchanged_ok = unchanged_ok and matches_parent
                verified = verified and matches_parent
                ready = ready and matches_parent
            else:
                matches_candidate = self._typed_match(card, doc, group, manifest['manifest_sha256'], uid)
                verified = verified and matches_candidate
                ready = ready and (matches_candidate or matches_parent or (prior is None and not doc and not group))
        state = {'bank': self.bank, 'manifest_sha256': manifest['manifest_sha256'], 'documents': docs, 'memory_units': units}
        with span('remote_state_hash', document_count=len(docs), unit_count=len(units)):
            state['state_sha256'] = merge.canonical_hash(state)
        # A generation already frozen after commit may not be silently repaired or
        # used as a new baseline after an unrelated writer changed any field/vector.
        if self._state_path(manifest, ledger).exists() and self._load_state(manifest, ledger) != state:
            ready = verified = False
        success = bool(delta) and bool(verified and legacy) and len(docs) == len(ids) and len(units) == len(ids)
        return {'bank': self.bank, 'manifest_sha256': manifest['manifest_sha256'],
            'published': success, 'readback_verified': success, 'legacy_seed_preserved': legacy,
            'typed_parent_preserved': bool(unchanged_ok), 'write_ready': bool(ready and legacy),
            'target_document_ids': targets, 'target_document_count': len(delta), 'target_observation_count': len(delta),
            'full_typed_document_count': len(ids), 'remote_document_count': len(docs), 'remote_observation_count': len(units),
            'incremental_unit_ids': unit_ids, 'remote_state_sha256': state['state_sha256']}, state

    @timed('readback_generation')
    def readback_generation(self, manifest: dict, *, ledger) -> dict:
        with self._readonly() as conn:
            receipt, state = self._readback(conn, manifest, ledger)
        if receipt['readback_verified'] and not self._state_path(manifest, ledger).exists():
            _freeze(self._state_path(manifest, ledger), state)
        return receipt

    @timed('publish_generation')
    def publish_generation(self, manifest: dict, gate: dict, *, ledger) -> dict:
        manifest, _, _, cards, remote = self._scope_candidate(manifest, ledger)
        self._validate_gate(manifest, gate, ledger)
        if not cards:
            raise ValueError('unchanged candidate requires no-op')
        # Commit may have happened despite an unknown response. Exact readback
        # ALWAYS precedes the embedding request and any SQL write on retries.
        readback = self.readback_generation(manifest, ledger=ledger)
        if readback['readback_verified']:
            return readback
        if not readback['legacy_seed_preserved']:
            raise ValueError('authoritative legacy seed changed; publication blocked')
        if not readback['write_ready']:
            raise ValueError('frozen typed parent changed or removed; publication blocked')
        reporter = active_reporter()
        @timed('prewrite_check', reporter=reporter)
        def prewrite(conn):
            # FOR UPDATE on every declared document also fences new child rows
            # through PostgreSQL's document FK KEY SHARE lock; units are locked
            # too, so non-cooperating writers cannot change parent revisions.
            current, _ = self._readback(conn, manifest, ledger, for_update=True)
            if not current['write_ready']:
                raise ValueError('seed or typed parent changed before transaction')
            self._scope_candidate(manifest, ledger)
            self._validate_gate(manifest, gate, ledger)
        @timed('precommit_check', reporter=reporter)
        def precommit(conn):
            current, _ = self._readback(conn, manifest, ledger)
            if not current['readback_verified']:
                raise ValueError('seed or full candidate drift before commit')
            self._validate_gate(manifest, gate, ledger)
        def embeddings(texts, *args, **kwargs):
            with reporter_context(reporter), span('embeddings', requested_count=len(texts)) as counts:
                result = self.embeddings(texts, *args, **kwargs)
                # The native API returns (vectors, provider). Count only the
                # outer list; never traverse/log vector values or change errors.
                if isinstance(result, tuple) and result and isinstance(result[0], list):
                    counts.update(received_count=len(result[0]))
                return result
        try:
            with span('publisher_publish', document_count=len(cards)):
                report = self.publisher.publish(cards, dsn=self.dsn, bank=self.bank, mode='publish', replace=False,
                embedding_provider='ollama', embedding_model=str(self.config.get('embedding_model') or 'bge-m3'),
                embedding_base_url=str(self.config.get('embedding_base_url') or self.publisher.DEFAULT_EMBEDDING_BASE_URL),
                embedding_batch_size=int(self.config.get('embedding_batch_size', 128)),
                backup_dir=merge._absolute(ledger.root) / 'publication-backups',
                generation_sha256=manifest['manifest_sha256'], lossless=True,
                backup_document_ids=readback['target_document_ids'], required_embedding_dimension=1024,
                connect_fn=self.connect, embedding_fn=embeddings, execute_values_fn=self.execute_values,
                prewrite_check=prewrite, precommit_check=precommit, incremental=True,
                incremental_unit_ids={k: v for k, v in readback['incremental_unit_ids'].items()
                                      if k in readback['target_document_ids']})
        except Exception:
            after = self.readback_generation(manifest, ledger=ledger)
            if after['readback_verified']:
                return after
            raise
        after = self.readback_generation(manifest, ledger=ledger)
        if not after['readback_verified']:
            raise RuntimeError('canonical commit readback did not match exact frozen full generation')
        after['publish_report'] = report
        _freeze(merge._absolute(ledger.root) / 'publication' / manifest['manifest_sha256'] /
                ('readback-' + uuid.uuid4().hex + '.json'), after)
        return after
