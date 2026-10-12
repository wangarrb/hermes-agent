#!/usr/bin/env python3
"""Bank-scoped, fail-closed arrival ledger and private immutable input files.

Only explicit bootstrap establishes the preserve-no-replay exclusion boundary.
Legacy inventory is never a success watermark. SQLite commits coverage only
AFTER durable payload creation; an interrupted capture leaves reusable orphan
payloads, never skipped input. No network, model or event-date decisions here.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import tempfile
import threading
import uuid
from pathlib import Path

from hindsight_incremental_sources import (
    SourceChangedError, canonical_json, digest, message_hash, snapshot_hash, validate_snapshot,
)

PHASES = ('captured', 'generated', 'submitted', 'unknown', 'retained', 'consolidated',
          'locally_merged', 'gated', 'published')
SUCCESS_PHASES = ('captured', 'generated', 'retained', 'consolidated', 'locally_merged', 'gated', 'published')
CONFIRM_INITIALIZE = 'initialize-hindsight-incremental'
PHYSICAL_BASELINE_SCHEMA = 'hindsight-physical-legacy-baseline/1'
_SCHEMA = '''
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, state TEXT NOT NULL, state_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources (source_key TEXT PRIMARY KEY, state TEXT NOT NULL, state_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, state TEXT NOT NULL, state_hash TEXT NOT NULL);
'''


class LedgerIntegrityError(ValueError):
    """A ledger or frozen payload no longer matches its bound identity."""


class LedgerNotInitializedError(RuntimeError):
    """Normal operation cannot create or infer a migration boundary."""


class LedgerLockedError(RuntimeError):
    """Another live producer owns this bank; never steal its lock."""


def _safe_path(path: Path) -> Path:
    raw = Path(path)
    if '..' in raw.parts:
        raise LedgerIntegrityError('path escape')
    absolute = Path(os.path.abspath(raw))
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise LedgerIntegrityError('symlink ledger path')
    return absolute


def _private(path: Path, *, directory=False):
    _safe_path(path)
    st = path.stat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(st.st_mode) or st.st_uid != os.geteuid() or st.st_mode & 0o077:
        raise LedgerIntegrityError('unsafe ledger type, owner or permissions')
    if not directory and st.st_nlink != 1:
        raise LedgerIntegrityError('hard-linked ledger file')
    return st


def _finite(value, field):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'invalid {field}')


class IncrementalLedger:
    def __init__(self, root: Path, bank: str, *, create: bool = False):
        if not isinstance(bank, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', bank) or bank in {'.', '..'}:
            raise ValueError('invalid bank identifier')
        self.storage_root, self.bank = _safe_path(root), bank
        self.bank_dir = self.storage_root / bank
        # Dependencies writing artifacts receive only this bank's private root.
        self.root = self.bank_dir
        self.payload_dir = self.bank_dir / 'payloads'
        self.db_path = self.bank_dir / 'ledger.sqlite3'
        self.lock_path = self.bank_dir / 'producer.lock'
        self._local = threading.local()
        if not self.db_path.exists():
            if not create:
                raise LedgerNotInitializedError('missing incremental ledger; explicit initialize required')
            self.storage_root.mkdir(mode=0o700, parents=True, exist_ok=True)
            for path in (self.bank_dir, self.payload_dir):
                _safe_path(path).mkdir(mode=0o700, exist_ok=True)
                _private(path, directory=True)
            for path in (self.db_path, self.lock_path):
                fd = os.open(_safe_path(path), os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
                os.close(fd)
            self._db_inode = (self.db_path.stat().st_dev, self.db_path.stat().st_ino)
            with self.locked(), self._connection(write=True, initialized=False) as con:
                con.executescript(_SCHEMA)
                if con.execute("SELECT 1 FROM meta WHERE key='identity'").fetchone() is None:
                    self._put(con, 'meta', 'identity', {'bank': bank, 'schema': 1})
        self._db_inode = (self.db_path.stat().st_dev, self.db_path.stat().st_ino)
        self._lock_inode = (self.lock_path.stat().st_dev, self.lock_path.stat().st_ino)
        with self._connection(initialized=False) as con:
            identity = self._read(con, 'meta', 'identity')
            if identity != {'bank': bank, 'schema': 1}:
                raise LedgerIntegrityError('bank/schema identity mismatch')
            if not create:
                self._require_initialized(con)

    def _check_paths(self):
        for path in (self.bank_dir, self.payload_dir):
            _private(path, directory=True)
        st = _private(self.db_path)
        if (st.st_dev, st.st_ino) != self._db_inode:
            raise LedgerIntegrityError('ledger database replaced')
        lock = _private(self.lock_path)
        if hasattr(self, '_lock_inode') and (lock.st_dev, lock.st_ino) != self._lock_inode:
            raise LedgerIntegrityError('ledger lock replaced')
        for suffix in ('-journal', '-wal', '-shm'):
            path = _safe_path(Path(str(self.db_path) + suffix))
            if path.exists():
                _private(path)

    @contextlib.contextmanager
    def locked(self):
        """Nonblocking process lock, reentrant only for the owning thread/PID."""
        self._check_paths()
        owner = getattr(self._local, 'owner', None)
        if owner == os.getpid():
            yield self
            return
        fd = os.open(self.lock_path, os.O_RDWR | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            if (st.st_dev, st.st_ino) != (self.lock_path.stat().st_dev, self.lock_path.stat().st_ino):
                raise LedgerIntegrityError('ledger lock replaced')
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise LedgerLockedError(f'bank {self.bank} has a live producer') from exc
            self._local.owner = os.getpid()
            try:
                self._check_paths()
                yield self
            finally:
                self._local.owner = None
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @contextlib.contextmanager
    def _connection(self, *, write=False, initialized=True):
        self._check_paths()
        mode = 'rw' if write else 'ro'
        con = sqlite3.connect(self.db_path.as_uri() + '?mode=' + mode, uri=True, timeout=0)
        try:
            con.execute('PRAGMA synchronous=FULL')
            if not write:
                con.execute('PRAGMA query_only=ON')
            con.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            if initialized:
                self._require_initialized(con)
            yield con
            if write:
                con.commit()
        except sqlite3.Error as exc:
            raise LedgerIntegrityError('invalid ledger SQLite state') from exc
        finally:
            con.close()

    def _put(self, con, table, key, value):
        column = {'meta': 'key', 'sources': 'source_key', 'batches': 'batch_id'}[table]
        text = canonical_json(value)
        con.execute(f'INSERT INTO {table} ({column},state,state_hash) VALUES (?,?,?) '
                    f'ON CONFLICT({column}) DO UPDATE SET state=excluded.state,state_hash=excluded.state_hash',
                    (key, text, hashlib.sha256(text.encode()).hexdigest()))

    def _read(self, con, table, key):
        column = {'meta': 'key', 'sources': 'source_key', 'batches': 'batch_id'}[table]
        row = con.execute(f'SELECT state,state_hash FROM {table} WHERE {column}=?', (key,)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
            raise LedgerIntegrityError(f'{table} state hash mismatch')
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise LedgerIntegrityError('invalid ledger state JSON') from exc

    def _physical_reference(self, reference):
        """Validate the literal private pointer without loading profile material."""
        fields = {'schema_version', 'bank', 'hermes_home', 'artifact_path', 'artifact_hash'}
        try:
            if (not isinstance(reference, dict) or set(reference) != fields or
                    any(not isinstance(v, str) or not v or any(ord(c) < 32 or ord(c) == 127 for c in v)
                        for v in reference.values()) or
                    reference['schema_version'] != PHYSICAL_BASELINE_SCHEMA or reference['bank'] != self.bank):
                raise ValueError('invalid reference')
            home = _safe_path(Path(reference['hermes_home']))
            if str(home) != reference['hermes_home']:
                raise ValueError('nonliteral home')
            path = self._artifact_pointer(reference['artifact_path'], reference['artifact_hash'])
            expected = self.root / 'delivery-artifacts' / (reference['artifact_hash'] + '.physical-baseline.json')
            if path != expected:
                raise ValueError('nonliteral frozen path')
            _private(path.parent, directory=True)
            info = _private(path)
            if stat.S_IMODE(info.st_mode) != 0o400:
                raise ValueError('unfinished or writable artifact')
            return dict(reference), info
        except (OSError, ValueError, TypeError):
            raise LedgerIntegrityError('invalid physical baseline reference') from None

    def _bound_physical_reference(self, manifest):
        reference, _ = self._physical_reference(manifest.get('physical_baseline'))
        try:
            if (manifest.get('baseline_kind') != 'physical' or
                    manifest.get('hermes_home') != reference['hermes_home'] or
                    manifest.get('legacy_status') != 'legacy_unknown' or
                    manifest.get('completion_proven') is not False or
                    type(manifest.get('excluded_source_events')) is not int or manifest['excluded_source_events'] != 0 or
                    not isinstance(manifest.get('seed'), dict) or
                    manifest.get('seed_identity_hash') != digest(manifest.get('seed'))):
                raise ValueError('invalid noncompletion binding')
        except (ValueError, TypeError):
            raise LedgerIntegrityError('invalid physical baseline metadata binding') from None
        return reference

    @staticmethod
    def _artifact_fence(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    def _validate_physical_artifact(self, reference, source_snapshots):
        from hindsight_incremental_delivery import _read
        from hindsight_incremental_merge import canonical_hash
        from hindsight_offline_retry import RemoteCallError
        import hindsight_incremental_sources as sources

        reference, before = self._physical_reference(reference)
        try:
            descriptor = _read(self.root, reference['artifact_path'], reference['artifact_hash'])
            if (not isinstance(descriptor, dict) or
                    any(descriptor.get(k) != reference[k] for k in ('schema_version', 'bank', 'hermes_home')) or
                    descriptor.get('status') != 'legacy_unknown' or descriptor.get('completion_proven') is not False or
                    not isinstance(descriptor.get('profiles'), list) or
                    canonical_hash(descriptor) != reference['artifact_hash']):
                raise ValueError('invalid descriptor binding')
            # The source adapter owns all per-feed fields; never invent a second DSL.
            sources.validate_physical_baseline(descriptor, bank=self.bank, hermes_home=Path(reference['hermes_home']))
            anchors = sources.physical_baseline_snapshots(descriptor)
            order = lambda s: s['source_key']
            if sorted(source_snapshots, key=order) != sorted(anchors, key=order):
                raise ValueError('source anchor coverage differs from physical descriptor')
            # Complete the freeze window if its owner exited after chmod but
            # before the directory fsync. No DB reference precedes durability.
            directory = os.open(Path(reference['artifact_path']).parent,
                                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            _, after = self._physical_reference(reference)
            if self._artifact_fence(before) != self._artifact_fence(after):
                raise ValueError('artifact changed during validation')
        except (OSError, ValueError, TypeError, KeyError, RemoteCallError):
            raise LedgerIntegrityError('invalid physical baseline artifact binding') from None
        return reference, self._artifact_fence(after)

    def _require_initialized(self, con):
        value = self._read(con, 'meta', 'legacy')
        if value is None:
            raise LedgerNotInitializedError('explicit initialize required; no implicit now baseline')
        if not isinstance(value, dict) or value.get('bank') != self.bank or value.get('policy') != 'preserve-no-replay':
            raise LedgerIntegrityError('legacy bank/policy mismatch')
        if any(k in value for k in ('physical_baseline', 'baseline_kind', 'hermes_home')):
            self._bound_physical_reference(value)
        return value

    def initialize_legacy(self, source_snapshots: list[dict], seed: dict, unknown: list[dict], *,
                          confirm: str, physical_baseline: dict | None = None) -> dict:
        if confirm != CONFIRM_INITIALIZE:
            raise ValueError('confirm must be initialize-hindsight-incremental')
        if not isinstance(seed, dict) or not isinstance(unknown, list):
            raise ValueError('invalid legacy seed/inventory')
        for snapshot in source_snapshots:
            validate_snapshot(snapshot)
        if physical_baseline is not None and any(s['messages'] for s in source_snapshots):
            raise LedgerIntegrityError('physical baseline requires empty logical source anchors')
        manifest = {'bank': self.bank, 'policy': 'preserve-no-replay', 'legacy_status': 'legacy_unknown',
                    'seed': seed, 'seed_identity_hash': digest(seed), 'unknown': unknown,
                    'excluded_source_events': sum(len(s['messages']) for s in source_snapshots),
                    'source_snapshots': [{'source_key': s['source_key'], 'source_epoch': s['source_epoch'],
                                          'snapshot_hash': s['snapshot_hash']} for s in source_snapshots]}
        with self.locked(), self._connection(write=True, initialized=False) as con:
            if self._read(con, 'meta', 'legacy') is not None:
                raise ValueError('ledger already initialized')
            reference, artifact_fence = None, None
            if physical_baseline is not None:
                reference, artifact_fence = self._validate_physical_artifact(physical_baseline, source_snapshots)
                manifest.update(physical_baseline=reference, hermes_home=reference['hermes_home'],
                                baseline_kind='physical', completion_proven=False)
            seen = set()
            for snapshot in source_snapshots:
                key = snapshot['source_key']
                if key in seen:
                    raise ValueError('duplicate baseline source')
                seen.add(key)
                self._bind_source_epoch(con, snapshot)
                state = self._source_state(snapshot)
                state['legacy_event_ids'] = list(state['events'])
                self._put(con, 'sources', key, state)
            self._put(con, 'meta', 'legacy', manifest)
            if physical_baseline is not None:
                _, final = self._physical_reference(reference)
                if self._artifact_fence(final) != artifact_fence:
                    raise LedgerIntegrityError('physical baseline artifact changed before initialization commit')
        return manifest

    def physical_baseline_reference(self) -> dict | None:
        """Return only the exact bound pointer; the normal reader validates bytes.

        Absence is compatible only with the original logical legacy mode. A
        malformed or missing pointer on a physical ledger is never a fallback.
        """
        with self._connection() as con:
            manifest = self._require_initialized(con)
            if 'physical_baseline' not in manifest:
                return None
            return dict(manifest['physical_baseline'])

    def initialization_manifest(self) -> dict:
        """Read-only audit of the non-success baseline/seed inventory."""
        with self._connection() as con:
            return self._require_initialized(con)

    def _source_state(self, snapshot):
        return {'bank': self.bank, 'source_key': snapshot['source_key'], 'source_epoch': snapshot['source_epoch'],
                'snapshot_hash': snapshot['snapshot_hash'], 'legacy_event_ids': [],
                'events': {m['event_id']: message_hash(m) for m in snapshot['messages']}}

    def _bind_source_epoch(self, con, snapshot):
        scope = snapshot.get('epoch_scope', snapshot['source_key'])
        key = 'epoch:' + scope
        visible = set(snapshot.get('scope_source_keys', [snapshot['source_key']]))
        prior = self._read(con, 'meta', key)
        if prior is not None:
            if prior['epoch'] != snapshot['source_epoch']:
                raise SourceChangedError('source_changed: feed epoch mismatch')
            if not set(prior['source_keys']).issubset(visible):
                raise SourceChangedError('source_changed: feed source coverage truncated')
        value = {'epoch': snapshot['source_epoch'], 'source_keys': sorted(visible)}
        if value != prior:
            self._put(con, 'meta', key, value)

    def _write_payload(self, path, data):
        """Publish a fsynced single-link file under the owning bank lock.

        An exclusive temporary plus same-directory rename never creates an
        alias, even if the process exits between publication and DB commit.
        Existing identities are verified and reused, never rewritten.
        """
        if getattr(self._local, 'owner', None) != os.getpid():
            raise LedgerLockedError('payload publication requires bank lock')
        _safe_path(path)
        if path.exists():
            _private(path)
            if path.read_bytes() != data:
                raise LedgerIntegrityError('immutable orphan payload hash mismatch')
            # A previous process may have exited after rename but before this
            # directory fsync. Finish its durability fence before DB coverage.
            self._sync_payload_directory()
            return
        fd, name = tempfile.mkstemp(prefix='.capture-', dir=self.payload_dir)
        tmp = Path(name)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.rename(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        self._sync_payload_directory()

    def _sync_payload_directory(self):
        directory = os.open(self.payload_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _make_batch(self, con, snapshot, messages, *, kind='source', parents=()):
        payload = {'bank': self.bank, 'source_key': snapshot['source_key'], 'source_epoch': snapshot['source_epoch'],
                   'profile': snapshot.get('profile', 'default'), 'kind': kind, 'parent_batch_ids': list(parents),
                   'cleaning_version': snapshot.get('cleaning_version', 'deterministic-clean-v2'),
                   'messages': messages}
        batch_id = digest(payload)
        document_id = f'hindsight-arrival::{self.bank}::{batch_id}'
        payload['items'] = [{'document_id': document_id,
                             'content': '\n\n'.join(f"{m['role'].title()}: {m['content']}" for m in messages),
                             'metadata': {'bank': self.bank, 'kind': kind,
                                          'source_key': payload['source_key'], 'source_epoch': payload['source_epoch'],
                                          'source_event_ids': [m['event_id'] for m in messages],
                                          'parent_batch_ids': list(parents)}}]
        data = canonical_json(payload).encode('utf-8')
        payload_hash = hashlib.sha256(data).hexdigest()
        if self._read(con, 'batches', batch_id) is not None:
            return None
        path = self.payload_dir / (batch_id + '.json')
        self._write_payload(path, data)
        batch = {'bank': self.bank, 'batch_id': batch_id, 'kind': kind, 'parent_batch_ids': list(parents),
                 'profile': payload['profile'], 'source_key': snapshot['source_key'],
                 'source_epoch': snapshot['source_epoch'], 'phase': 'captured', 'last_successful_phase': 'captured',
                 'payload_path': str(path), 'payload_hash': payload_hash, 'payload_sha256': payload_hash,
                 'source_event_ids': [m['event_id'] for m in messages],
                 'document_id': f'hindsight-arrival::{self.bank}::{batch_id}',
                 'operation_id': str(uuid.uuid5(uuid.NAMESPACE_URL, f'hindsight-arrival:{self.bank}:{batch_id}')),
                 'operation_payload_hash': None, 'receipts': {}, 'input_chars': sum(len(m['content']) for m in messages),
                 'attempts': 0, 'next_retry_at': 0.0, 'last_error_class': None}
        self._store_batch(con, batch)
        return batch

    def _store_batch(self, con, batch):
        self._put(con, 'batches', batch['batch_id'], batch)

    def _split(self, messages, max_input_chars):
        if not isinstance(max_input_chars, int) or max_input_chars < 1:
            raise ValueError('invalid max_input_chars')
        groups, current, used = [], [], 0
        for msg in messages:
            size = len(msg['content'])
            if size > max_input_chars:
                raise ValueError('payload_blocked: message exceeds input budget; no truncation')
            if current and used + size > max_input_chars:
                groups.append(current)
                current, used = [], 0
            current.append(msg)
            used += size
        if current:
            groups.append(current)
        return groups

    def capture_snapshot(self, snapshot: dict, *, max_input_chars: int = 60000) -> list[dict]:
        if snapshot.get('kind', 'source') != 'source' or snapshot.get('parent_batch_ids'):
            raise ValueError('derived batches require capture_derived and verified parent receipts')
        validate_snapshot(snapshot)
        with self.locked(), self._connection(write=True) as con:
            self._bind_source_epoch(con, snapshot)
            state = self._source_state(snapshot)
            previous = self._read(con, 'sources', snapshot['source_key'])
            if previous is not None:
                if previous['bank'] != self.bank or previous['source_epoch'] != state['source_epoch']:
                    raise SourceChangedError('source_changed: source bank/epoch mismatch')
                for event, event_hash in previous['events'].items():
                    if state['events'].get(event) != event_hash:
                        raise SourceChangedError('source_changed: source truncated or captured event changed')
                state['legacy_event_ids'] = previous['legacy_event_ids']
            known = previous['events'] if previous else {}
            unseen = [m for m in snapshot['messages'] if m['event_id'] not in known]
            groups = self._split(unseen, max_input_chars)
            batches = [self._make_batch(con, snapshot, group) for group in groups]
            # Coverage + cursor advance in the SAME transaction as batch state.
            if state != previous:
                self._put(con, 'sources', snapshot['source_key'], state)
            return [b for b in batches if b is not None]

    def capture_derived(self, parent_batch_ids: list[str], messages: list[dict], *, kind: str = 'daily_summary',
                        max_input_chars: int = 60000) -> list[dict]:
        """Freeze new summary children from retained raw parents, never date queries.

        Source fields cannot authenticate success. Parent phase and immutable
        receipts are loaded from this same bank ledger within the transaction.
        A derived batch owns its own generated/submitted/retained lifecycle.
        """
        if kind == 'source' or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', kind):
            raise ValueError('invalid derived kind')
        parents = sorted(set(parent_batch_ids))
        if not parents:
            raise ValueError('derived batch needs parent receipts')
        with self.locked(), self._connection(write=True) as con:
            parent_states = [self._batch(con, parent) for parent in parents]
            for parent in parent_states:
                if parent['kind'] != 'source' or PHASES.index(parent['phase']) < PHASES.index('retained') or not parent['receipts'].get('retained'):
                    raise ValueError('parent must be confirmed retained source with receipt')
                self._validate_retained_receipt(parent, parent['receipts']['retained'])
                if parent['receipts'].get('published', {}).get('no_canonical_delta') is True:
                    raise ValueError('sealed no-delta parent cannot acquire a new derived contribution')
            identity = [{'batch_id': p['batch_id'], 'payload_hash': p['payload_hash'],
                         'operation_id': p['operation_id'], 'retained_receipt_hash': digest(p['receipts']['retained'])}
                        for p in parent_states]
            snapshot = {'source_key': 'derived:' + kind + ':' + digest(identity),
                        'source_epoch': digest(identity), 'profile': parent_states[0]['profile'], 'messages': messages,
                        'source_hash_before': digest(identity), 'source_hash_after': digest(identity)}
            snapshot['snapshot_hash'] = snapshot_hash(snapshot)
            validate_snapshot(snapshot)
            prior = self._read(con, 'sources', snapshot['source_key'])
            if prior is not None:
                if prior.get('snapshot_hash') != snapshot['snapshot_hash']:
                    raise ValueError('derived payload mismatch: parent inputs already frozen')
                return []
            groups = self._split(messages, max_input_chars)
            if not groups:
                raise ValueError('derived contribution requires nonempty messages')
            for parent in parents:
                if self._read(con, 'meta', 'derived-parent:' + parent) is not None:
                    raise ValueError('parent contribution already consumed by derived batch')
            # Also fence ledgers created by the pre-fix helper, whose derived
            # batches already own a contribution but have no per-parent map.
            for (existing_id,) in con.execute('SELECT batch_id FROM batches'):
                existing = self._read(con, 'batches', existing_id)
                if set(existing.get('parent_batch_ids', [])) & set(parents):
                    raise ValueError('parent contribution already consumed by derived batch')
            batches = [self._make_batch(con, snapshot, group, kind=kind, parents=parents) for group in groups]
            self._put(con, 'sources', snapshot['source_key'], self._source_state(snapshot))
            for parent in identity:
                self._put(con, 'meta', 'derived-parent:' + parent['batch_id'],
                          {'bank': self.bank, **parent, 'derived_source_key': snapshot['source_key'],
                           'derived_batch_ids': [b['batch_id'] for b in batches]})
            return [b for b in batches if b is not None]

    def _batch(self, con, batch_id, *, verify_payload=None):
        batch = self._read(con, 'batches', batch_id)
        if batch is None:
            raise KeyError(batch_id)
        if batch.get('bank') != self.bank or batch.get('batch_id') != batch_id or batch.get('phase') not in PHASES:
            raise LedgerIntegrityError('batch identity/state mismatch')
        expected = self.payload_dir / (batch_id + '.json')
        if batch.get('payload_path') != str(expected):
            raise LedgerIntegrityError('batch payload path escape')
        self._validate_batch_metadata(con, batch)
        if verify_payload is None:
            verify_payload = batch['phase'] != 'published'
        if not verify_payload:
            return batch
        _private(expected)
        fd = os.open(expected, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as handle:
            data = handle.read()
        if hashlib.sha256(data).hexdigest() != batch['payload_hash']:
            raise LedgerIntegrityError('payload hash mismatch')
        payload = json.loads(data)
        if digest({k: v for k, v in payload.items() if k != 'items'}) != batch_id:
            raise LedgerIntegrityError('batch input identity hash mismatch')
        for key in ('bank', 'source_key', 'source_epoch', 'profile', 'kind', 'parent_batch_ids'):
            if payload[key] != batch[key]:
                raise LedgerIntegrityError('payload identity mismatch')
        if [m['event_id'] for m in payload['messages']] != batch['source_event_ids']:
            raise LedgerIntegrityError('payload coverage mismatch')
        return batch

    def get_batch(self, batch_id: str) -> dict:
        with self._connection() as con:
            return self._batch(con, batch_id)

    def list_batches(self, *, phases: tuple[str, ...] | None = None) -> list[dict]:
        if phases is not None and any(p not in PHASES for p in phases):
            raise ValueError('invalid phase filter')
        with self._connection() as con:
            ids = [row[0] for row in con.execute('SELECT batch_id FROM batches ORDER BY rowid')]
            batches = [self._batch(con, batch_id, verify_payload=False) for batch_id in ids]
            return [b for b in batches if phases is None or b['phase'] in phases]

    def pending_batches(self, *, now: float, max_units: int = 12, max_chars: int = 500000) -> list[dict]:
        _finite(now, 'now')
        if not isinstance(max_units, int) or not isinstance(max_chars, int) or min(max_units, max_chars) < 0:
            raise ValueError('invalid pending budget')
        selected, used = [], 0
        for batch in self.list_batches():
            if batch['phase'] == 'published' or (batch['next_retry_at'] is not None and batch['next_retry_at'] > now):
                continue
            if len(selected) >= max_units:
                break
            if used + batch['input_chars'] > max_chars:
                continue
            selected.append(self.get_batch(batch['batch_id']))
            used += batch['input_chars']
        return selected

    def payload_diagnostics(self, *, sealed_only: bool = True) -> list[dict]:
        """Read-only missing/corrupt raw-file audit; never requeue sealed input.

        Normal selection verifies ledger/receipt identity without reopening
        sealed raw history. Status tooling can surface these separate blockers
        for explicit history review, while genuine new input remains eligible.
        """
        issues = []
        with self._connection() as con:
            ids = [r[0] for r in con.execute('SELECT batch_id FROM batches ORDER BY rowid')]
            for batch_id in ids:
                batch = self._batch(con, batch_id, verify_payload=False)
                if sealed_only and batch['phase'] != 'published':
                    continue
                try:
                    self._batch(con, batch_id, verify_payload=True)
                except FileNotFoundError:
                    status = 'missing'
                except (LedgerIntegrityError, ValueError, OSError):
                    status = 'corrupt'
                else:
                    continue
                issues.append({'bank': self.bank, 'batch_id': batch_id, 'phase': batch['phase'], 'status': status})
        return issues

    def _receipt_file(self, path: str, expected_hash: str | None):
        target = _safe_path(Path(path))
        if not target.is_relative_to(self.root):
            raise LedgerIntegrityError('receipt artifact path escape')
        _private(target)
        with target.open('rb') as handle:
            raw = handle.read()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError('receipt artifact hash mismatch')
        return json.loads(raw)

    def _validate_retained_receipt(self, batch, receipt):
        required = {'bank': self.bank, 'batch_id': batch['batch_id'], 'operation_id': batch['operation_id'],
                    'input_hash': batch['payload_hash'], 'payload_hash': batch['operation_payload_hash'],
                    'status': 'completed'}
        if not batch['operation_payload_hash'] or any(receipt.get(k) != v for k, v in required.items()):
            raise ValueError('retained receipt identity/status mismatch')
        def check_children(value):
            children = value.get('children')
            if not isinstance(children, list):
                raise ValueError('retained receipt lacks child completeness proof')
            for child in children:
                if (not isinstance(child, dict) or child.get('status') != 'completed' or child.get('bank') != self.bank
                        or not isinstance(child.get('operation_id'), str) or not child['operation_id'].strip()
                        or any(ord(c) < 32 or ord(c) == 127 for c in child['operation_id'])):
                    raise ValueError('retained child receipt mismatch')
                check_children(child)
        check_children(receipt)

    def _receipt_binding(self, batch, phase, receipt):
        if not isinstance(receipt, dict) or not receipt:
            raise ValueError('phase requires a durable receipt')
        for key in ('bank', 'batch_id'):
            if receipt.get(key) != batch[key]:
                raise ValueError(f'receipt {key} mismatch')
        for key in ('source_key', 'source_epoch', 'payload_sha256', 'operation_id'):
            if key in receipt and receipt[key] != batch[key]:
                raise ValueError(f'receipt {key} mismatch')
        if 'input_hash' in receipt and receipt['input_hash'] != batch['payload_hash']:
            raise ValueError('receipt input hash mismatch')
        if phase != 'generated':
            for key in ('payload_hash', 'operation_payload_hash'):
                if key in receipt and receipt[key] != batch['operation_payload_hash']:
                    raise ValueError('receipt operation payload hash mismatch')
        # These are evidence fields, not freeform provenance. A successful
        # schema cannot simultaneously carry negative or uncertain evidence.
        def evidence(value):
            if not isinstance(value, dict):
                raise ValueError('invalid receipt proof')
            if 'status' in value and value['status'] != 'completed':
                raise ValueError('receipt status is failed or unconfirmed')
            for key in ('ready', 'readback_verified', 'published', 'legacy_seed_preserved'):
                if key in value and value[key] is not True:
                    raise ValueError('receipt contains negative proof')
            for key in ('scoped_proof', 'gate', 'publication'):
                if key in value:
                    evidence(value[key])
        if phase not in {'submitted', 'unknown'}:
            evidence(receipt)

    def _artifact_pointer(self, path, checksum):
        if not isinstance(path, str) or not path or not isinstance(checksum, str) or not re.fullmatch(r'[0-9a-f]{64}', checksum):
            raise ValueError('generated receipt needs artifact path/hash')
        target = _safe_path(Path(path))
        if str(target) != path or not target.is_relative_to(self.root):
            raise LedgerIntegrityError('receipt artifact path escape')
        return target

    def _children_for(self, con, batch_id):
        children = []
        for (key,) in con.execute('SELECT batch_id FROM batches'):
            state = self._read(con, 'batches', key)
            if batch_id in state.get('parent_batch_ids', []):
                children.append(state)
        return children

    def _manifest_proof(self, con, batch, manifest, *, verify_files):
        from hindsight_incremental_merge import SCHEMA, validate_generation
        if not isinstance(manifest, dict) or manifest.get('bank') != self.bank or manifest.get('schema_version') != SCHEMA:
            raise ValueError('manifest receipt bank/schema mismatch')
        checksum = manifest.get('manifest_sha256')
        path = self._artifact_pointer(manifest.get('manifest_path'), checksum)
        if manifest.get('root') != str(path.parent) or path.name != 'manifest.json':
            raise ValueError('manifest receipt root/path mismatch')
        if digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != checksum:
            raise ValueError('manifest receipt hash mismatch')
        consumed = manifest.get('consumed_batch_ids')
        if not isinstance(consumed, list) or consumed != sorted(set(consumed)):
            raise ValueError('manifest receipt coverage mismatch')
        if batch['kind'] == 'source':
            children = self._children_for(con, batch['batch_id'])
            if not children or any(c['phase'] != 'published' or c['batch_id'] not in consumed or
                    c['receipts'].get('published', {}).get('manifest') != manifest for c in children):
                raise ValueError('source manifest receipt lacks published child coverage')
        elif batch['batch_id'] not in consumed:
            raise ValueError('manifest receipt batch coverage mismatch')
        if verify_files:
            _private(path)
            if validate_generation(path) != manifest:
                raise ValueError('manifest receipt differs from frozen generation')
            for entry in manifest['files']:
                _private(path.parent / entry['path'])
        return manifest

    def _validate_stage_receipt(self, con, batch, phase, receipt, *, verify_files=False):
        self._receipt_binding(batch, phase, receipt)
        if phase == 'generated':
            if receipt.get('input_hash') != batch['payload_hash'] or receipt.get('operation_id') != batch['operation_id']:
                raise ValueError('generated receipt input/operation mismatch')
            self._artifact_pointer(receipt.get('payload_path'), receipt.get('payload_hash'))
            if batch['operation_payload_hash'] not in (None, receipt['payload_hash']):
                raise ValueError('generated receipt operation payload hash mismatch')
            if 'operation_payload_hash' in receipt and receipt['operation_payload_hash'] != receipt['payload_hash']:
                raise ValueError('generated receipt operation payload hash mismatch')
            expected_kind = 'raw_source' if batch['kind'] == 'source' else 'summary'
            if receipt.get('artifact_kind') != expected_kind:
                raise ValueError('generated receipt artifact kind mismatch')
            if expected_kind == 'summary':
                self._artifact_pointer(receipt.get('artifact_path'), receipt.get('artifact_hash'))
            elif receipt.get('artifact_path') is not None or receipt.get('artifact_hash') is not None:
                raise ValueError('raw generated receipt cannot claim summary artifact')
            if verify_files:
                payload = self._receipt_file(receipt['payload_path'], receipt['payload_hash'])
                if not isinstance(payload, dict) or payload.get('operation_id') != batch['operation_id'] or payload.get('async') is not True:
                    raise ValueError('generated receipt operation payload mismatch')
                items = payload.get('items')
                if not isinstance(items, list) or not items or any(not isinstance(i, dict) or
                        i.get('document_id') != batch['document_id'] or not isinstance(i.get('content'), str) or
                        not i['content'].strip() for i in items):
                    raise ValueError('generated receipt document/content mismatch')
                if expected_kind == 'summary':
                    artifact = self._receipt_file(receipt['artifact_path'], receipt['artifact_hash'])
                    if not isinstance(artifact, dict) or not isinstance(artifact.get('canonical_observations'), list) or not artifact['canonical_observations']:
                        raise ValueError('generated receipt summary schema invalid')
            return
        if phase in {'submitted', 'unknown'}:
            if not batch['operation_payload_hash'] or receipt.get('operation_id') != batch['operation_id'] or receipt.get('payload_hash') != batch['operation_payload_hash']:
                raise ValueError('submitted receipt requires exact prebound operation payload hash')
            return
        if phase == 'retained':
            self._validate_retained_receipt(batch, receipt)
            return
        if phase == 'consolidated':
            proof = receipt.get('scoped_proof')
            if (not isinstance(proof, dict) or proof.get('bank') != self.bank or
                    proof.get('scope') != [f"arrival-batch:{batch['batch_id']}"] or
                    type(proof.get('pending_in_scope')) is not int or proof['pending_in_scope'] != 0 or
                    type(proof.get('failed_in_scope')) is not int or proof['failed_in_scope'] != 0 or
                    proof.get('ready') is not True):
                raise ValueError('consolidated receipt requires exact scoped drain proof')
            operations = proof.get('operation_ids')
            if (not isinstance(operations, list) or any(not isinstance(op, str) or not op or
                    any(ord(c) < 32 for c in op) for op in operations) or len(operations) != len(set(operations))):
                raise ValueError('consolidated receipt lacks exact native operation ID proof')
            for key in ('pending_operations',):
                if key in proof and (type(proof[key]) is not int or proof[key] != 0):
                    raise ValueError('consolidated receipt has failed/uncertain drain proof')
            return
        if receipt.get('no_canonical_delta') is True:
            if (batch['kind'] != 'source' or self._children_for(con, batch['batch_id']) or
                    receipt.get('retained_receipt_hash') != digest(batch['receipts']['retained']) or
                    any(k in receipt for k in ('manifest', 'gate', 'publication', 'sequence', 'published', 'readback_verified'))):
                raise ValueError('no-canonical-delta receipt identity/coverage mismatch')
            if phase != 'locally_merged' and not batch['receipts']['locally_merged'].get('no_canonical_delta'):
                raise ValueError('no-canonical-delta receipt cannot replace frozen canonical proof')
            return
        if 'no_canonical_delta' in receipt or 'retained_receipt_hash' in receipt:
            raise ValueError('invalid no-canonical-delta receipt')
        manifest = self._manifest_proof(con, batch, receipt.get('manifest'), verify_files=verify_files)
        if phase != 'locally_merged' and manifest != batch['receipts']['locally_merged'].get('manifest'):
            raise ValueError('manifest receipt differs from frozen merge proof')
        if phase == 'gated':
            gate = receipt.get('gate')
            required = ('schema', 'coverage', 'conflict', 'generic_eval', 'local_eval', 'scoped_drain')
            if (not isinstance(gate, dict) or gate.get('bank') != self.bank or
                    gate.get('manifest_sha256') != manifest['manifest_sha256'] or
                    gate.get('decision') != 'eligible_for_local_proposal' or
                    not isinstance(gate.get('checks'), dict) or any(gate['checks'].get(k) is not True for k in required)):
                raise ValueError('gated receipt requires exact eligible gate proof')
        elif phase == 'published':
            publication = receipt.get('publication')
            if (not isinstance(publication, dict) or publication.get('bank') != self.bank or
                    publication.get('manifest_sha256') != manifest['manifest_sha256'] or
                    any(publication.get(k) is not True for k in ('published', 'readback_verified', 'legacy_seed_preserved')) or
                    type(receipt.get('sequence')) is not int or receipt['sequence'] < 1 or
                    manifest != batch['receipts']['gated'].get('manifest')):
                raise ValueError('published receipt requires exact positive readback proof')

    def _validate_batch_metadata(self, con, batch):
        expected_operation = str(uuid.uuid5(uuid.NAMESPACE_URL, f"hindsight-arrival:{self.bank}:{batch['batch_id']}"))
        if (not re.fullmatch(r'[0-9a-f]{64}', batch['batch_id']) or
                batch.get('document_id') != f"hindsight-arrival::{self.bank}::{batch['batch_id']}" or
                batch.get('operation_id') != expected_operation or batch.get('payload_sha256') != batch.get('payload_hash') or
                not isinstance(batch.get('payload_hash'), str) or not re.fullmatch(r'[0-9a-f]{64}', batch['payload_hash']) or
                not isinstance(batch.get('receipts'), dict)):
            raise LedgerIntegrityError('batch metadata identity/hash mismatch')
        required = [p for p in PHASES[1:PHASES.index(batch['phase']) + 1] if p != 'unknown']
        if batch['phase'] == 'unknown':
            required.append('unknown')
        if any(p not in batch['receipts'] for p in required):
            raise LedgerIntegrityError('batch lacks durable stage receipt')
        try:
            for phase, receipt in batch['receipts'].items():
                if phase not in PHASES or PHASES.index(phase) > PHASES.index(batch['phase']):
                    raise ValueError('receipt phase mismatch')
                self._validate_stage_receipt(con, batch, phase, receipt)
            if batch['receipts'].get('generated', {}).get('payload_hash') != batch['operation_payload_hash']:
                raise ValueError('receipt operation hash mismatch')
        except (ValueError, TypeError, KeyError) as exc:
            raise LedgerIntegrityError('batch receipt identity/proof mismatch') from exc

    def set_phase(self, batch_id: str, phase: str, *, receipt: dict | None = None) -> dict:
        if phase not in PHASES:
            raise ValueError('invalid phase')
        with self.locked(), self._connection(write=True) as con:
            batch = self._batch(con, batch_id)
            # Unknown and submitted belong to the same unconfirmed tier. An
            # exact same-UUID retry preserves unknown, never regresses success.
            retry_unknown = batch['phase'] == 'unknown' and phase == 'submitted'
            old = PHASES.index(batch['phase'])
            new = PHASES.index(phase)
            if not retry_unknown and (new < old or (new > old + 1 and not (batch['phase'] == 'submitted' and phase == 'retained'))):
                raise ValueError('nonmonotone or skipped phase')
            if phase != 'captured':
                self._validate_stage_receipt(con, batch, phase, receipt, verify_files=True)
            receipt = dict(receipt or {})
            if phase == 'generated':
                batch['operation_payload_hash'] = receipt['payload_hash']
            existing = batch['receipts'].get(phase)
            if existing and existing != receipt:
                raise ValueError('phase receipt mismatch; frozen receipts cannot be replaced')
            if receipt:
                batch['receipts'][phase] = receipt
            if not retry_unknown:
                batch['phase'] = phase
            if phase in SUCCESS_PHASES:
                batch['last_successful_phase'] = phase
            if phase == 'published':
                batch['completion_kind'] = ('no_canonical_delta' if receipt.get('no_canonical_delta') is True
                                            else 'canonical_readback_verified')
            batch['next_retry_at'], batch['last_error_class'] = 0.0, None
            self._store_batch(con, batch)
            return batch

    def defer(self, batch_id: str, *, error_class: str, next_retry_at: float, attempts: int) -> dict:
        if not isinstance(error_class, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', error_class):
            raise ValueError('invalid non-sensitive error_class')
        _finite(next_retry_at, 'next_retry_at')
        if not isinstance(attempts, int) or attempts < 0:
            raise ValueError('invalid attempts')
        with self.locked(), self._connection(write=True) as con:
            batch = self._batch(con, batch_id)
            if batch['phase'] == 'published' or attempts < batch['attempts']:
                raise ValueError('cannot defer sealed phase or decrease attempts')
            batch.update(last_error_class=error_class, next_retry_at=next_retry_at, attempts=attempts)
            self._store_batch(con, batch)
            return batch
