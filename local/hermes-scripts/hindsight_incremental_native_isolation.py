"""Fail-closed isolation for the audited native 0.9 PostgreSQL deployment.

REST schema/version is not backend provenance. Only a local single-process
systemd CLI with the audited source tree, unchanged since boot, is supported.
Caller and server must share a network namespace. Deployment must separately
exclude noncooperating producers and unaudited standalone workers throughout
the owned window; a local ledger lock/empty queue is not a server write fence.
Metadata-only SQL fills the public list API's missing per-fact scope/source
fields; this never fetches historical text or runs native/paid functions.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import NoReturn
from urllib.parse import urlsplit

from hindsight_incremental_delivery import _digest, _json_bytes, _read, operation_receipt
from hindsight_offline_retry import RemoteCallError

# This is the installed, function-exercised 0.9 audit, not a configurable trust
# boolean/hash. Unknown versions must be audited, not silently opted into.
NATIVE_TREE_SHA256 = 'c55ad66d980c637130cb9f4c1f9f3dc53ce7dcc1ac9dd907a89ea241febfdde2'
ENTRY_SHA256 = 'a9b870ca46c53f633ce23be25c0b98521e73e2a00dbcb80f333435757f582eae'
NATIVE_MEMORIES = 'hindsight_api.engine.memories.postgres.PostgresMemories'
# Reviewed startup hooks: Conda path correction, distutils shim, dormant
# coverage, and a finder restricted to pointcept. Unreviewed hooks block.
STARTUP_HASHES = {
    'sitecustomize.py': '4d28d27c2fa0bf05113f4eef7c5d3e58a4011b74ba50674b6303b167b31f514d',
    'distutils-precedence.pth': '2638ce9e2500e572a5e0de7faed6661eb569d1b696fcba07b0dd223da5f5d224',
    'a1_coverage.pth': 'ef2ed06d19867ec669c09a804060666a9cd5e383af0a9d11aa2de79b77d448e8',
    '__editable__.pointcept-0.0.0.pth': '2b0e50847f737e3369dd8cd1a7c0198344c70ebf11740bc9c075e60dc5b8e375',
    '__editable___pointcept_0_0_0_finder.py': 'c8d8d080da5d9d03b5dc0328bbc94276bc4d6fe31d962c63508cf00e30da5623',
}


def _fail(code) -> NoReturn:
    raise RemoteCallError(code)


def _stat_key(info):
    # Content-relevant identity only: atime is access metadata that relatime
    # updates on a plain read and must not be reported as a source change.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _file_bytes(path, started):
    before = path.stat()
    if path.is_symlink() or max(before.st_mtime, before.st_ctime) > started:
        _fail('native_source_changed_since_start')
    raw = path.read_bytes()
    if _stat_key(path.stat()) != _stat_key(before):
        _fail('native_source_changed_during_probe')
    return raw


def _identity(proc):
    fields = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()
    return {'pid': int(proc.name), 'start_ticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def _network_namespace(proc):
    """Caller loopback must name the same namespace as the serving socket."""
    try:
        caller = Path('/proc/self/ns/net').stat()
        server = (proc / 'ns/net').stat()
    except OSError:
        _fail('native_network_namespace_unproved')
    if (caller.st_dev, caller.st_ino) != (server.st_dev, server.st_ino):
        _fail('native_network_namespace_unproved')


def _environment(env):
    if any(v for k, v in env.items() if k.endswith('_EXTENSION') and k.startswith('HINDSIGHT_API_')):
        _fail('native_extension_unproved')
    if any(env.get(k) for k in ('PYTHONPATH', 'PYTHONHOME', 'COVERAGE_PROCESS_START', 'COVERAGE_PROCESS_CONFIG')):
        _fail('native_startup_environment_unproved')
    if env.get('HINDSIGHT_API_DATABASE_SCHEMA', 'public') != 'public':
        _fail('native_schema_unproved')
    if env.get('HINDSIGHT_API_DATABASE_BACKEND', 'postgresql') != 'postgresql':
        _fail('native_backend_unproved')
    if (env.get('HINDSIGHT_API_WORKERS', '1') != '1'
            or env.get('HINDSIGHT_API_WORKER_ENABLED', 'true').lower() != 'true'):
        _fail('native_worker_contract_unproved')


def _startup_paths(site, proc, env, started):
    home = Path(env.get('HOME', ''))
    if not home.is_absolute():
        _fail('native_startup_environment_unproved')
    # CLI sys.path[0] is bin, not cwd. Conda's reviewed sitecustomize can add
    # two other sites; user-site and simple path .pth files can shadow imports.
    others = [site.parent.parent / 'python3.1' / 'site-packages',
              site.parent.parent / 'python' / 'site-packages',
              home / '.local' / 'lib' / site.parent.name / 'site-packages',
              (proc / 'exe').resolve().parent]
    for hook in [*site.glob('*.pth'), site.parent / 'sitecustomize.py']:
        if not hook.exists():
            continue
        raw = _file_bytes(hook, started)
        if any(line.startswith((b'import ', b'import\t')) for line in raw.splitlines()) or hook.name == 'sitecustomize.py':
            if hashlib.sha256(raw).hexdigest() != STARTUP_HASHES.get(hook.name):
                _fail('native_startup_hook_unproved')
            if hook.name == '__editable__.pointcept-0.0.0.pth':
                finder = site / '__editable___pointcept_0_0_0_finder.py'
                if hashlib.sha256(_file_bytes(finder, started)).hexdigest() != STARTUP_HASHES[finder.name]:
                    _fail('native_startup_hook_unproved')
        else:
            for line in raw.decode().splitlines():
                if line.strip() and not line.lstrip().startswith('#'):
                    target = site / line.strip()
                    if target.exists():
                        others.append(target)
    for directory in others:
        if not directory.exists():
            continue
        if (any(directory.glob('*.pth')) or any((directory / n).exists() for n in
                ('hindsight_api', 'hindsight_api.py', 'sitecustomize.py', 'usercustomize.py'))):
            _fail('native_import_path_unproved')
    if (site / 'sitecustomize.py').exists() or (site / 'usercustomize.py').exists():
        _fail('native_import_path_unproved')


def local_provenance(api, dsn, config):
    """Observe the serving process; credentials stay in memory, never receipts."""
    try:
        url = urlsplit(api)
        if (url.scheme != 'http' or url.hostname not in {'127.0.0.1', 'localhost'}
                or url.path or url.query or url.fragment or url.username or url.password):
            _fail('native_local_backend_required')
        unit = ((config.get('incremental') or {}).get('native_isolation') or {}).get('service_unit', 'hindsight.service')
        if not isinstance(unit, str) or not re.fullmatch(r'[A-Za-z0-9_.@-]+\.service', unit):
            _fail('native_service_unproved')
        result = subprocess.run(['systemctl', 'show', unit, '--property=MainPID,ActiveState', '--value'],
                                capture_output=True, text=True, timeout=5, check=True)
        values = result.stdout.splitlines()
        pid = next((int(v) for v in values if v.isdigit() and int(v) > 0), None)
        if pid is None or 'active' not in values:
            _fail('native_service_unproved')
        proc = Path(f'/proc/{pid}')
        identity = _identity(proc)
        _network_namespace(proc)
        boot = next(float(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime '))
        started = boot + int(identity['start_ticks']) / os.sysconf('SC_CLK_TCK')
        exe = (proc / 'exe').resolve()
        if not re.fullmatch(r'python3\.\d+', exe.name):
            _fail('native_entrypoint_unproved')
        entry = exe.parent / 'hindsight-api'
        argv = (proc / 'cmdline').read_bytes().split(b'\0')[:-1]
        if argv not in ([str(exe).encode(), str(entry).encode()],
                        [str(exe).encode(), str(entry).encode(), b'--log-level', b'info']):
            _fail('native_entrypoint_unproved')
        if hashlib.sha256(_file_bytes(entry, started)).hexdigest() != ENTRY_SHA256:
            _fail('native_entrypoint_unproved')
        # Match the listening socket inode to the service PID, not a version
        # string or an argv substring. No proxy/remote backend can pass this.
        inodes = {p.readlink().name[8:-1] for p in (proc / 'fd').iterdir()
                  if p.readlink().name.startswith('socket:[')}
        port = url.port or 80
        listening = {line.split()[9] for line in (proc / 'net' / 'tcp').read_text().splitlines()[1:]
                     if line.split()[3] == '0A' and int(line.split()[1].split(':')[1], 16) == port}
        if not inodes & listening:
            _fail('native_socket_unproved')
        env = dict(pair.decode().split('=', 1) for pair in (proc / 'environ').read_bytes().split(b'\0') if b'=' in pair)
        _environment(env)
        # main.load_dotenv_for_entrypoint overrides process env. We only
        # support a stable discovered dotenv with no Hindsight/Python settings;
        # interpolation or guessing effective secrets would not be proof.
        cwd = (proc / 'cwd').resolve()
        for parent in (cwd, *cwd.parents):
            dotenv = parent / '.env'
            if dotenv.exists():
                raw = _file_bytes(dotenv, started).decode()
                if re.search(r'HINDSIGHT_API_|PYTHON|COVERAGE', raw):
                    _fail('native_dotenv_override_unproved')
                break
        from psycopg2.extensions import parse_dsn
        actual_dsn = env.get('HINDSIGHT_API_DATABASE_URL')
        if not actual_dsn or parse_dsn(actual_dsn) != parse_dsn(dsn):
            _fail('native_database_identity_unproved')
        site = exe.parent.parent / 'lib' / exe.name / 'site-packages'
        _startup_paths(site, proc, env, started)
        root = site / 'hindsight_api'
        files = {str(p.relative_to(root)): hashlib.sha256(_file_bytes(p, started)).hexdigest()
                 for p in sorted(root.rglob('*.py'))}
        digest = _digest(_json_bytes(files))
        if not files or digest != NATIVE_TREE_SHA256:
            _fail('native_source_contract_unproved')
        if _identity(proc) != identity:
            _fail('native_backend_changed')
        _network_namespace(proc)
        return {**identity, 'executable': str(exe), 'source_sha256': digest,
                'memories': NATIVE_MEMORIES, 'operation_validator': None, 'database_backend': 'postgresql'}
    except RemoteCallError:
        raise
    except Exception:
        _fail('native_backend_provenance_unknown')


def _contents(items):
    """Installed MemoryItem → retain child JSON for an exact ownership query.

    Restrict to one immutable document and no strategy/append rather than
    reimplementing native grouping. This is a wire binding, not a producer.
    """
    if len(items) != 1:
        _fail('native_wire_payload_unproved')
    item = items[0]
    allowed = {'content', 'document_id', 'timestamp', 'context', 'metadata', 'entities',
               'tags', 'observation_scopes', 'update_mode', 'strategy'}
    if not set(item) <= allowed or item.get('strategy') or item.get('update_mode') not in {None, 'replace'}:
        _fail('native_wire_payload_unproved')
    content = {k: item[k] for k in ('content', 'document_id', 'context', 'metadata', 'tags') if item.get(k)}
    content['observation_scopes'] = item['observation_scopes']
    if item.get('timestamp'):
        value = item['timestamp']
        content['event_date'] = None if value.lower() == 'unset' else datetime.fromisoformat(value.replace('Z', '+00:00')).isoformat()
    if item.get('entities'):
        content['entities'] = [{'text': e['text'], 'type': e.get('type') or 'CONCEPT'} for e in item['entities']]
    if item.get('update_mode') is not None:
        content['update_mode'] = item['update_mode']
    return [content]


def _payload(runtime, batch, items=None):
    generated = batch.get('receipts', {}).get('generated')
    expected = {'bank': runtime.bank, 'batch_id': batch['batch_id'], 'operation_id': batch['operation_id']}
    if (not isinstance(generated, dict) or any(generated.get(k) != v for k, v in expected.items())
            or generated.get('input_hash') != (batch.get('payload_hash') or batch.get('payload_sha256'))
            or generated.get('payload_hash') != batch.get('operation_payload_hash')):
        _fail('native_scope_ownership_unproved')
    payload = _read(runtime.ledger.root, generated['payload_path'], generated['payload_hash'])
    if (payload.get('operation_id') != batch['operation_id'] or payload.get('async') is not True
            or not isinstance(payload.get('items'), list) or (items is not None and payload['items'] != items)):
        _fail('native_wire_binding_mismatch')
    tag = runtime._scope(batch)
    if any(i.get('document_id') != batch['document_id'] or i.get('tags') != tag
           or i.get('observation_scopes') != [tag] for i in payload['items']):
        _fail('native_private_scope_mismatch')
    _contents(payload['items'])
    return payload, generated['payload_hash']


def _rows(cur, query, params):
    cur.execute(query, params)
    values = cur.fetchall()
    if any(len(row) != 1 or not isinstance(row[0], dict) for row in values):
        _fail('native_metadata_unknown')
    return [row[0] for row in values]


def remote_binding(cur, bank, doc, operation_id, contents):
    """Return only exact child-payload equality, using the actual 0.9 schema."""
    cur.execute("""SELECT /* arrival-binding */ EXISTS (
      SELECT 1 FROM public.async_operations p JOIN public.async_operations c
        ON c.result_metadata->>'parent_operation_id'=p.operation_id::text
      WHERE p.operation_id=%s::uuid AND p.bank_id=%s AND p.operation_type='batch_retain'
        AND p.result_metadata->>'document_id'=%s
        AND c.bank_id=p.bank_id AND c.operation_type='retain'
        AND c.task_payload->>'type'='batch_retain'
        AND c.task_payload->>'bank_id'=p.bank_id
        AND c.task_payload->>'operation_id'=c.operation_id::text
        AND c.task_payload->'contents'=%s::jsonb
        AND NOT (c.task_payload ?| ARRAY['strategy','document_tags','_tenant_id','_api_key_id'])
        AND (SELECT count(*) FROM public.async_operations x WHERE
             x.result_metadata->>'parent_operation_id'=p.operation_id::text)=1)""",
                (operation_id, bank, doc, _json_bytes(contents).decode()))
    matched = cur.fetchall()
    return len(matched) == 1 and len(matched[0]) == 1 and matched[0][0] is True


class NativeIsolation:
    def __init__(self, runtime):
        self.runtime = runtime

    def attest(self):
        runtime = self.runtime
        proof = local_provenance(runtime.native.api, runtime._publication().dsn, runtime.config)
        if (not isinstance(proof, dict) or proof.get('source_sha256') != NATIVE_TREE_SHA256
                or proof.get('memories') != NATIVE_MEMORIES or proof.get('operation_validator', 'unknown') is not None
                or proof.get('database_backend') != 'postgresql' or type(proof.get('pid')) is not int
                or any(not isinstance(proof.get(k), str) or not proof[k] for k in ('boot_id', 'start_ticks', 'executable'))):
            _fail('native_backend_contract_unproved')
        lease = runtime._load_lease(runtime.ledger) if runtime.ledger is not None else None
        if lease and lease.get('status') != 'restored' and lease.get('native_backend') not in (None, proof):
            _fail('native_backend_changed')
        try:
            with runtime._publication()._readonly() as conn, conn.cursor() as cur:
                models = _rows(cur, "SELECT jsonb_build_object('id',id,'trigger',trigger) FROM public.mental_models WHERE bank_id=%s ORDER BY id", (runtime.bank,))
            for model in models:
                trigger = model.get('trigger')
                if (not isinstance(trigger, dict) or trigger.get('refresh_after_consolidation') is not False
                        or trigger.get('refresh_cron') is not None):
                    _fail('native_mental_model_refresh_unproved')
        except RemoteCallError:
            raise
        except Exception:
            _fail('native_metadata_sql_unproved')
        return proof

    def verify(self, batch, *, expected_ids=None, items=None):
        """Prove all strict-tag supersets and all source links before text/POST."""
        runtime = self.runtime
        backend = self.attest()
        tag, doc = runtime._scope(batch), batch['document_id']
        payload = digest = None
        if items is not None:
            payload, digest = _payload(runtime, batch, items)
        try:
            with runtime._publication()._readonly() as conn, conn.cursor() as cur:
                docs = _rows(cur, "SELECT jsonb_build_object('id',id,'bank_id',bank_id,'tags',tags,'content_hash',content_hash) FROM public.documents WHERE bank_id=%s AND (id=%s OR tags && %s::varchar[]) ORDER BY id", (runtime.bank, doc, tag))
                projection = "jsonb_build_object('id',id::text,'bank_id',bank_id,'document_id',document_id,'fact_type',fact_type,'tags',tags,'observation_scopes',observation_scopes,'source_memory_ids',source_memory_ids)"
                query = f'SELECT {projection} FROM public.{{table}} WHERE bank_id=%s AND (document_id=%s OR tags && %s::varchar[]) ORDER BY id'
                live = _rows(cur, query.format(table='memory_units'), (runtime.bank, doc, tag))
                archive = _rows(cur, query.format(table='invalidated_memory_units'), (runtime.bank, doc, tag))
                if archive or any(d.get('id') != doc or d.get('bank_id') != runtime.bank or d.get('tags') != tag for d in docs):
                    _fail('native_reserved_scope_collision')
                cur.execute('SELECT /* arrival-operation-exists */ EXISTS (SELECT 1 FROM public.async_operations WHERE operation_id=%s::uuid)', (batch['operation_id'],))
                exists = cur.fetchall()
                if len(exists) != 1 or len(exists[0]) != 1 or type(exists[0][0]) is not bool:
                    _fail('native_remote_operation_unknown')
                if docs or live or exists == [(True,)]:
                    payload, digest = _payload(runtime, batch, items)
                    binding = batch.get('receipts', {}).get('submitted') or batch.get('receipts', {}).get('unknown') or batch.get('receipts', {}).get('retained')
                    if (not isinstance(binding, dict) or binding.get('operation_id') != batch['operation_id']
                            or binding.get('payload_hash') != digest or binding.get('bank') != runtime.bank
                            or binding.get('batch_id') != batch['batch_id']):
                        _fail('native_scope_ownership_unproved')
                    # The native parent has NULL task_payload; exact immutable
                    # contents live on its one child. Compare in SQL and return
                    # only a boolean, never task bodies or historical text.
                    if not remote_binding(cur, runtime.bank, doc, batch['operation_id'], _contents(payload['items'])):
                        _fail('native_remote_payload_binding_unproved')
                    if docs and docs[0].get('content_hash') != _digest(payload['items'][0]['content'].encode()):
                        _fail('native_document_payload_mismatch')
                sources, observations = [], []
                seen = set()
                for row in live:
                    if (not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen
                            or row.get('bank_id') != runtime.bank or row.get('tags') != tag):
                        _fail('native_reserved_scope_collision')
                    seen.add(row['id'])
                    if row.get('fact_type') in {'world', 'experience'}:
                        if row.get('document_id') != doc or row.get('observation_scopes') != [tag]:
                            _fail('native_fact_scope_mismatch')
                        sources.append(row['id'])
                    elif row.get('fact_type') == 'observation':
                        observations.append(row)
                    else:
                        _fail('native_reserved_scope_collision')
                if live and not docs:
                    _fail('native_document_ownership_unproved')
                if expected_ids is not None and (len(expected_ids) != len(set(expected_ids)) or set(expected_ids) != set(sources)):
                    _fail('native_document_source_ids_mismatch')
                for row in observations:
                    ids = row.get('source_memory_ids')
                    if not isinstance(ids, list) or not ids or any(i not in sources for i in ids):
                        _fail('native_observation_source_ids_unproved')
                if observations:
                    lease = runtime._load_lease(runtime.ledger)
                    state = (lease or {}).get('scopes', {}).get(tag[0], {})
                    sealed = batch.get('receipts', {}).get('consolidated', {}).get('scoped_proof', {})
                    if not state.get('consolidation_operation_ids') and not sealed.get('operation_ids'):
                        _fail('native_observation_ownership_unproved')
                proof = {'backend': backend, 'bank': runtime.bank, 'batch_id': batch['batch_id'], 'scope': tag,
                         'document_id': doc, 'operation_id': batch['operation_id'], 'payload_sha256': digest,
                         'source_ids': sorted(sources), 'observation_ids': sorted(r['id'] for r in observations),
                         'metadata_sha256': _digest(_json_bytes({'documents': docs, 'live': live}))}
                return proof
        except RemoteCallError:
            raise
        except Exception:
            _fail('native_metadata_sql_unproved')
